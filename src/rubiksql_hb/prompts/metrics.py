"""The metric registry, exposure predicates and the exact paired test.

Every metric declares how it is valued on one eval row and which rows are *exposed* to it,
reading the row as ``scripts/eval_bird.py`` writes it (``id``, ``ex_score``, ``pred_sql``,
``candidates``, ``judge_debug``, ``candidates_debug``), the per-agent ``prompt_identity``
snapshots the runtime records (``rubiksql.render/1``) and, when the caller attaches them,
lab fields under ``row["lab"]``. The repository's readings have always been conditional
behaviour rates of this kind (docs/measured-rejections.md); EX alone needs thousands of
questions to separate two arms.

Two arms are compared on every question both of them value (:func:`paired`): the discordant
pairs ``(b, c)`` give the net effect in percentage points (:func:`effect_pp`) and the exact
one-sided McNemar test (:func:`mcnemar_exact`). The promotion rule that reads them lives in
:mod:`evolve`; the other registered metrics are reported beside EX as information.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Iterable, Mapping

from . import entries as E

__all__ = [
    "METRICS",
    "EvalRow",
    "ExposureContext",
    "McNemar",
    "Metric",
    "MetricValue",
    "PairedResult",
    "compute",
    "effect_pp",
    "mcnemar_exact",
    "paired",
    "resolve_metric",
    "slot_stats",
]

_GROUP_RE = re.compile(r"Candidate (\d+):\s*\[result-group ([A-Z]+): (?:(\d+) candidate\(s\) return this exact result|unique result)\]")
_WS_RE = re.compile(r"\s+")
#: The submit-nudge surfaces; a snapshot names whichever row provided them (either layout).
_CANDIDATE_NUDGE = "candidate.nudge.submit"
_JUDGE_NUDGE = "judge.nudge.submit"


def _norm_sql(sql: Any) -> str:
    return _WS_RE.sub(" ", str(sql or "")).strip().rstrip(";").strip().lower()


# ---------------------------------------------------------------------------------------
# Rows


class EvalRow:
    """One eval result row with the accessors the metrics read.

    Attributes:
        raw (Mapping[str, Any]): The row as written to ``<db_id>.json`` ``results``, optionally with a ``lab``
            mapping (``{"judge": {...}, "candidates": [...]}``) attached by the caller.
    """

    def __init__(self, raw: Mapping[str, Any]) -> None:
        self.raw = raw

    @property
    def qid(self) -> str:
        """Question id.

        Returns:
            str: ``row["id"]``.
        """
        return str(self.raw.get("id"))

    @property
    def db_id(self) -> str:
        """Database id.

        Returns:
            str: ``row["db_id"]``.
        """
        return str(self.raw.get("db_id", ""))

    @property
    def ex(self) -> float | None:
        """Official EX of the submitted SQL.

        Returns:
            float | None: ``1.0`` or ``0.0``; ``None`` without gold or when EX scoring failed.
            A retained unscorable gold contributes fixed zero in both arms.
            Legacy ``exd_score`` and ``score_errors=["exd"]`` are read too.
        """
        if "no_gold" in (self.raw.get("reasons") or ()):
            return None
        # Legacy scored rows may omit gold_sql entirely, but an explicitly empty
        # gold_sql cannot support an EX score even when its placeholder is 0.
        if "gold_sql" in self.raw and not str(self.raw["gold_sql"] or "").strip():
            return None
        score_errors = self.raw.get("score_errors") or ()
        if "ex" in score_errors or "exd" in score_errors:
            return None
        if (self.raw.get("gold_execution") or {}).get("status") == "gold_unscorable":
            return 0.0
        score = self.raw.get("ex_score")
        if score is None:
            score = self.raw.get("exd_score")
        if score is None:
            return None
        return 1.0 if float(score) >= 1.0 else 0.0

    @property
    def pred_sql(self) -> str:
        """Submitted SQL.

        Returns:
            str: ``row["pred_sql"]`` or ``""``.
        """
        return str(self.raw.get("pred_sql") or "")

    @property
    def judge_debug(self) -> Mapping[str, Any]:
        """The judge's debug record.

        Returns:
            Mapping[str, Any]: ``row["judge_debug"]`` or ``{}``.
        """
        value = self.raw.get("judge_debug")
        return value if isinstance(value, Mapping) else {}

    @property
    def candidates(self) -> list[Mapping[str, Any]]:
        """Candidate summaries ``{sql, row_count, error}`` in pool order.

        Returns:
            list[Mapping[str, Any]]: ``row["candidates"]`` or ``[]``.
        """
        value = self.raw.get("candidates")
        return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []

    @property
    def candidates_debug(self) -> list[Mapping[str, Any]]:
        """Candidate debug records (transcripts, ``prompt_identity``) in pool order.

        Returns:
            list[Mapping[str, Any]]: ``row["candidates_debug"]`` or ``[]``.
        """
        value = self.raw.get("candidates_debug")
        return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []

    @property
    def lab(self) -> Mapping[str, Any]:
        """Lab fields attached by the caller.

        Returns:
            Mapping[str, Any]: ``row["lab"]`` or ``{}``.
        """
        value = self.raw.get("lab")
        return value if isinstance(value, Mapping) else {}

    def identities(self, role: str | None = None) -> list[Mapping[str, Any]]:
        """Every ``prompt_identity`` snapshot the row carries.

        Args:
            role (str | None): Keep only snapshots of this role (``candidate``, ``judge``, ``reviewer``).

        Returns:
            list[Mapping[str, Any]]: Snapshots from the row, the judge, the reviewer and each candidate debug record.
        """
        found: list[Mapping[str, Any]] = []
        for holder in (self.raw, self.judge_debug, self.raw.get("reviewer_debug") or {}, *self.candidates_debug):
            snapshot = holder.get("prompt_identity") if isinstance(holder, Mapping) else None
            if isinstance(snapshot, Mapping) and (role is None or snapshot.get("role") == role):
                found.append(snapshot)
        return found

    def fired(self, role: str | None = None) -> frozenset[str]:
        """Entries any agent of the row rendered.

        Args:
            role (str | None): Restrict to one role.

        Returns:
            frozenset[str]: Entry names across the matching snapshots.
        """
        names: set[str] = set()
        for snapshot in self.identities(role):
            fired = snapshot.get("fired")
            if isinstance(fired, Mapping):
                names.update(str(name) for name in fired)
        return frozenset(names)

    def fired_count(self, snapshot: Mapping[str, Any], surface: str) -> int:
        """How many times one snapshot rendered a surface.

        Args:
            snapshot (Mapping[str, Any]): A ``rubiksql.render/1`` document.
            surface (str): Surface id, e.g. ``candidate.nudge.submit``; the entries of either layout that provide it
                are counted.

        Returns:
            int: The summed ``n`` counters, or 0.
        """
        fired = snapshot.get("fired")
        if not isinstance(fired, Mapping):
            return 0
        return sum(int((data or {}).get("n", 0)) for entry, data in fired.items() if surface in E.surfaces_of((str(entry),)))

    def groups(self) -> dict[int, tuple[str, int]]:
        """Result groups of the judged pool, parsed from the judge's user message.

        Returns:
            dict[int, tuple[str, int]]: Candidate number (1-based) to ``(group label, group size)``; empty when
                the judge message is absent.
        """
        message = self.judge_debug.get("user_msg")
        if not isinstance(message, str):
            return {}
        groups: dict[int, tuple[str, int]] = {}
        for match in _GROUP_RE.finditer(message):
            groups[int(match.group(1))] = (match.group(2), int(match.group(3) or 1))
        return groups

    def valid_candidates(self) -> dict[int, Mapping[str, Any]]:
        """Candidates that executed, by 1-based number.

        Returns:
            dict[int, Mapping[str, Any]]: Number to summary for every candidate without an error.
        """
        return {index: item for index, item in enumerate(self.candidates, 1) if not item.get("error")}

    @property
    def judge_mode(self) -> str:
        """How the judge ended (``complete_submit``, ``truncated``, ``reexec_*``, ``fallback*``).

        Returns:
            str: ``judge_debug.mode`` or ``""``.
        """
        return str(self.judge_debug.get("mode") or "")

    @property
    def fallback(self) -> bool | None:
        """Whether the fallback selection produced the answer.

        Returns:
            bool | None: ``True`` for ``fallback*`` modes or a pool-guard fallback, ``False`` for
                ``complete_submit``, ``None`` when the mode leaves it unrecorded (the lab's own rule).
        """
        if self.raw.get("pool_guard_fallback"):
            return True
        mode = self.judge_mode
        if mode.startswith("fallback"):
            return True
        if mode == "complete_submit":
            return False
        lab_value = self.lab.get("judge", {}).get("fallback") if isinstance(self.lab.get("judge"), Mapping) else None
        return lab_value if isinstance(lab_value, bool) else None

    def width(self, key: str) -> int | None:
        """Column count of the first preview row.

        Args:
            key (str): ``"pred_rows_preview"`` or ``"gold_rows_preview"``.

        Returns:
            int | None: Number of columns, or ``None`` when the preview is absent or empty.
        """
        preview = self.raw.get(key)
        if not isinstance(preview, list) or not preview:
            return None
        first = preview[0]
        if isinstance(first, (list, tuple, Mapping)):
            return len(first)
        return 1


@dataclass(frozen=True)
class ExposureContext:
    """What the exposure predicates need beyond the row.

    Attributes:
        changed_surfaces (frozenset[str]): Surfaces whose static text differs between the two arms; entry names of
            either layout (an older outcome's ``changed_entries``) are read as the surfaces they provide.
        k (int): The ``@k`` threshold of parametrised metrics.
        changed_knobs (frozenset[str]): Knobs whose value differs between the two arms' releases
            (:data:`entries.KNOBS`); the pairing reads them, no metric does.
    """

    changed_surfaces: frozenset[str] = frozenset()
    k: int = 1
    changed_knobs: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------------------
# Metric registry


@dataclass(frozen=True)
class Metric:
    """A binary per-row metric with an exposure predicate.

    Attributes:
        name (str): Registry name, e.g. ``ex@exposed`` or ``submit_after_nudge@k``.
        description (str): What the value and the exposure mean.
        exposed (Callable[[EvalRow, ExposureContext], bool]): Whether the row is in the metric's denominator.
        value (Callable[[EvalRow], float | None]): 1.0/0.0 on an exposed row, ``None`` when unmeasurable.
    """

    name: str
    description: str
    exposed: Callable[[EvalRow, ExposureContext], bool]
    value: Callable[[EvalRow], float | None]


def _always(_row: EvalRow, _ctx: ExposureContext) -> bool:
    return True


def _fired_changed(row: EvalRow, ctx: ExposureContext) -> bool:
    return bool(E.fired_surfaces(row.fired()) & E.surfaces_of(ctx.changed_surfaces))


def _pool_4of5(row: EvalRow, _ctx: ExposureContext) -> bool:
    valid = row.valid_candidates()
    groups = row.groups()
    if len(valid) != 5 or not groups:
        return False
    sizes = {groups[index][1] for index in valid if index in groups}
    return max(sizes, default=0) == 4


def _overturn_4of5(row: EvalRow) -> float | None:
    if row.fallback:
        return None
    groups = row.groups()
    consensus = {_norm_sql(item.get("sql")) for index, item in row.valid_candidates().items() if index in groups and groups[index][1] == 4}
    if not consensus or not row.pred_sql:
        return None
    return 0.0 if _norm_sql(row.pred_sql) in consensus else 1.0


def _pool_unanimous(row: EvalRow, _ctx: ExposureContext) -> bool:
    valid = row.valid_candidates()
    groups = row.groups()
    if len(valid) < 2 or not groups:
        return False
    labels = {groups[index][0] for index in valid if index in groups}
    return len(labels) == 1 and all(index in groups for index in valid)


def _escape_unanimous(row: EvalRow) -> float | None:
    if row.fallback or not row.pred_sql:
        return None
    pool = {_norm_sql(item.get("sql")) for item in row.valid_candidates().values()}
    return 0.0 if _norm_sql(row.pred_sql) in pool else 1.0


def _judged(row: EvalRow, _ctx: ExposureContext) -> bool:
    return bool(row.judge_debug) or bool(row.raw.get("pool_guard_fallback"))


def _fallback_value(row: EvalRow) -> float | None:
    fallback = row.fallback
    return None if fallback is None else float(fallback)


def _submitted(row: EvalRow, _ctx: ExposureContext) -> bool:
    return bool(row.pred_sql.strip()) and not row.raw.get("placeholder")


def _empty_submit(row: EvalRow) -> float | None:
    rows = row.raw.get("pred_rows")
    if rows is None:
        return None
    return 1.0 if int(rows) == 0 else 0.0


def _nudged_units(row: EvalRow, k: int) -> list[tuple[str, Mapping[str, Any], int]]:
    units = []
    for index, debug in enumerate(row.candidates_debug):
        snapshot = debug.get("prompt_identity")
        if isinstance(snapshot, Mapping) and row.fired_count(snapshot, _CANDIDATE_NUDGE) >= k:
            units.append(("candidate", debug, index))
    judge = row.judge_debug.get("prompt_identity")
    if isinstance(judge, Mapping) and row.fired_count(judge, _JUDGE_NUDGE) >= k:
        units.append(("judge", row.judge_debug, -1))
    return units


def _nudged(row: EvalRow, ctx: ExposureContext) -> bool:
    return bool(_nudged_units(row, ctx.k))


def _submit_after_nudge(k: int) -> Callable[[EvalRow], float | None]:
    def value(row: EvalRow) -> float | None:
        units = _nudged_units(row, k)
        if not units:
            return None
        for role, record, index in units:
            if role == "candidate":
                summary = row.candidates[index] if index < len(row.candidates) else {}
                submitted = bool((record.get("sql") or summary.get("sql") or "").strip()) and not summary.get("error")
            else:
                submitted = row.judge_mode == "complete_submit"
            if not submitted:
                return 0.0
        return 1.0

    return value


def _both_widths(row: EvalRow, _ctx: ExposureContext) -> bool:
    return row.width("pred_rows_preview") is not None and row.width("gold_rows_preview") is not None


def _colcount_mismatch(row: EvalRow) -> float | None:
    pred, gold = row.width("pred_rows_preview"), row.width("gold_rows_preview")
    if pred is None or gold is None:
        return None
    return 0.0 if pred == gold else 1.0


def _nudge_metric(k: int) -> Metric:
    return Metric(
        f"submit_after_nudge@{k}",
        f"Among rows where a candidate or the judge received the submit nudge at least {k} times, whether every nudged agent went on to submit.",
        lambda row, ctx: _nudged(row, ExposureContext(ctx.changed_surfaces, k, ctx.changed_knobs)),
        _submit_after_nudge(k),
    )


#: The registry. ``submit_after_nudge@k`` is registered for k = 1..3 and resolved for any k.
METRICS: Mapping[str, Metric] = {
    metric.name: metric
    for metric in (
        Metric("ex", "Official EX of the submitted SQL over every question.", _always, lambda row: row.ex),
        Metric("ex@exposed", "EX restricted to rows where some agent rendered a surface the proposal changed.", _fired_changed, lambda row: row.ex),
        Metric(
            "judge_overturn_4of5",
            "In pools of five executed candidates where four agree, whether the judge submitted SQL outside that consensus.",
            _pool_4of5,
            _overturn_4of5,
        ),
        Metric(
            "pool_escape_unanimous",
            "In pools where every executed candidate returns the same result, whether the judge submitted SQL that is in none of them.",
            _pool_unanimous,
            _escape_unanimous,
        ),
        Metric("fallback_rate", "Among judged rows, whether the fallback selection produced the answer.", _judged, _fallback_value),
        Metric("empty_submit_rate", "Among rows with real submitted SQL, whether it returned zero rows.", _submitted, _empty_submit),
        _nudge_metric(1),
        _nudge_metric(2),
        _nudge_metric(3),
        Metric("colcount_mismatch", "Among rows with both previews, whether the predicted column count differs from gold.", _both_widths, _colcount_mismatch),
    )
}


def resolve_metric(name: str) -> Metric:
    """Look a metric up by name, materialising ``submit_after_nudge@k`` for any positive k.

    Args:
        name (str): Registry name.

    Returns:
        Metric: The metric.

    Raises:
        KeyError: If the name is unknown.
    """
    if name in METRICS:
        return METRICS[name]
    match = re.fullmatch(r"submit_after_nudge@(\d+)", name)
    if match and int(match.group(1)) >= 1:
        return _nudge_metric(int(match.group(1)))
    raise KeyError(f"unknown metric {name!r}; registered: {sorted(METRICS)}")


@dataclass(frozen=True)
class MetricValue:
    """One metric over one arm."""

    metric: str
    n_exposed: int
    n_valued: int
    mean: float | None
    values: Mapping[str, float] = field(default_factory=dict)


def compute(metric: Metric | str, rows: Iterable[Mapping[str, Any] | EvalRow], ctx: ExposureContext = ExposureContext()) -> MetricValue:
    """Value a metric over one arm.

    Args:
        metric (Metric | str): The metric or its registry name.
        rows (Iterable[Mapping[str, Any] | EvalRow]): Eval rows of the arm.
        ctx (ExposureContext): Exposure inputs.

    Returns:
        MetricValue: Exposed and valued counts, the mean over valued rows, and per-qid values.
    """
    metric = resolve_metric(metric) if isinstance(metric, str) else metric
    values: dict[str, float] = {}
    exposed = 0
    for raw in rows:
        row = raw if isinstance(raw, EvalRow) else EvalRow(raw)
        if not metric.exposed(row, ctx):
            continue
        exposed += 1
        value = metric.value(row)
        if value is not None:
            values[row.qid] = value
    mean = sum(values.values()) / len(values) if values else None
    return MetricValue(metric.name, exposed, len(values), mean, values)


@dataclass(frozen=True)
class PairedResult:
    """A metric paired between a parent arm and a child arm on shared questions.

    Attributes:
        metric (str): Metric name.
        n (int): Questions valued in both arms (the paired denominator).
        b (int): Parent 1, child 0.
        c (int): Parent 0, child 1.
        effect_pp (float): ``(c - b) / n × 100``; 0.0 when n is 0.
        qids (tuple[str, ...]): The paired questions.
    """

    metric: str
    n: int
    b: int
    c: int
    effect_pp: float
    qids: tuple[str, ...] = ()


def slot_stats(rows: Iterable[Mapping[str, Any] | EvalRow]) -> dict[str, dict[str, Any]]:
    """Per candidate slot of a run: how often the judge answered with that slot's SQL, and how those answers scored.

    A slot is the candidate's position in the pool, counted from 1 in the order of the candidates spec — the
    candidate a ``candidate.s<k>`` document serves. On a question the slot is *picked* when the submitted SQL equals
    that slot's SQL up to whitespace, case and a trailing ``;``; slots that wrote the same SQL are picked together,
    and a fallback answer picks nothing. ``own_ex`` is the slot's own correctness where the rows carry it — the lab
    attaches ``row["lab"]["candidates"][k - 1]["scores"]["ex"]`` — and stays ``None`` on plain eval rows: scoring a
    candidate means executing its SQL against the gold, work this statistic never does.

    Args:
        rows (Iterable[Mapping[str, Any] | EvalRow]): Eval rows of one run.

    Returns:
        dict[str, dict[str, Any]]: Slot (``"1"``, ``"2"``, …) to ``{"n", "valid", "picked", "picked_ex", "own_n",
            "own_ex"}``: questions the slot ran on, of those where it returned SQL without an error, where it was picked
            and the mean EX of those answers (over the scored ones; ``None`` when none was), and where its own EX is known
            with its mean.
    """
    out: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = raw if isinstance(raw, EvalRow) else EvalRow(raw)
        lab = row.lab.get("candidates") if isinstance(row.lab.get("candidates"), list) else []
        pred = _norm_sql(row.pred_sql) if row.pred_sql and not row.fallback else ""
        for index, candidate in enumerate(row.candidates):
            bucket = out.setdefault(str(index + 1), {"n": 0, "valid": 0, "picked": 0, "picked_n": 0, "picked_ex_sum": 0.0, "own_n": 0, "own_ex_sum": 0.0})
            bucket["n"] += 1
            valid = not candidate.get("error") and bool(candidate.get("sql"))
            bucket["valid"] += int(valid)
            if valid and pred and _norm_sql(candidate.get("sql")) == pred:
                bucket["picked"] += 1
                if row.ex is not None:
                    bucket["picked_n"] += 1
                    bucket["picked_ex_sum"] += row.ex
            scores = lab[index].get("scores") if index < len(lab) and isinstance(lab[index], Mapping) else None
            if isinstance(scores, Mapping) and isinstance(scores.get("ex"), (int, float)):
                bucket["own_n"] += 1
                bucket["own_ex_sum"] += 1.0 if float(scores["ex"]) >= 1.0 else 0.0
    return {
        slot: {
            "n": bucket["n"],
            "valid": bucket["valid"],
            "picked": bucket["picked"],
            "picked_ex": round(bucket["picked_ex_sum"] / bucket["picked_n"], 6) if bucket["picked_n"] else None,
            "own_n": bucket["own_n"],
            "own_ex": round(bucket["own_ex_sum"] / bucket["own_n"], 6) if bucket["own_n"] else None,
        }
        for slot, bucket in sorted(out.items(), key=lambda item: int(item[0]))
    }


def effect_pp(b: int, c: int, n: int) -> float:
    """Net effect in percentage points from discordant counts.

    Args:
        b (int): Parent right, child wrong.
        c (int): Parent wrong, child right.
        n (int): Paired denominator.

    Returns:
        float: ``(c - b) / n × 100``, or 0.0 when ``n`` is 0.
    """
    return 0.0 if n == 0 else (c - b) / n * 100.0


def paired(metric: Metric | str, parent_rows: Iterable[Mapping[str, Any] | EvalRow], child_rows: Iterable[Mapping[str, Any] | EvalRow], ctx: ExposureContext = ExposureContext()) -> PairedResult:
    """Pair a metric between two arms on every question both value.

    Exposure is taken from the union of the two arms: a question the child exposed but the parent
    did not (or vice versa) is still paired when both arms value it, so a proposal that changes
    exposure itself does not silently shrink the denominator.

    Args:
        metric (Metric | str): The metric or its registry name.
        parent_rows (Iterable[Mapping[str, Any] | EvalRow]): Parent arm rows.
        child_rows (Iterable[Mapping[str, Any] | EvalRow]): Child arm rows.
        ctx (ExposureContext): Exposure inputs.

    Returns:
        PairedResult: Discordant counts and the net effect.
    """
    metric = resolve_metric(metric) if isinstance(metric, str) else metric
    parent = {row.qid: row for row in (r if isinstance(r, EvalRow) else EvalRow(r) for r in parent_rows)}
    child = {row.qid: row for row in (r if isinstance(r, EvalRow) else EvalRow(r) for r in child_rows)}
    b = c = 0
    qids = []
    for qid in sorted(set(parent) & set(child)):
        prow, crow = parent[qid], child[qid]
        if not (metric.exposed(prow, ctx) or metric.exposed(crow, ctx)):
            continue
        pv, cv = metric.value(prow), metric.value(crow)
        if pv is None or cv is None:
            continue
        qids.append(qid)
        if pv > cv:
            b += 1
        elif cv > pv:
            c += 1
    return PairedResult(metric.name, len(qids), b, c, effect_pp(b, c, len(qids)), tuple(qids))


# ---------------------------------------------------------------------------------------
# Exact test


def _half_tail_ge(k: int, n: int) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, 1/2)``, the McNemar null.

    The tail is a ratio of integers and is evaluated exactly with :class:`fractions.Fraction`, so
    ``math.comb(n, i)`` never has to fit a float (it overflows near ``n = 1030``).
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return float(Fraction(sum(math.comb(n, i) for i in range(k, n + 1)), 2**n))


@dataclass(frozen=True)
class McNemar:
    """Result of the exact one-sided McNemar test on discordant pairs."""

    b: int
    c: int
    p: float


def mcnemar_exact(b: int, c: int) -> McNemar:
    """Exact one-sided McNemar test on discordant pairs: does the child win more of them than a fair coin?

    Under the null the ``b + c`` discordant pairs split as Binomial(b + c, 1/2); the p-value is
    ``P(X >= c)``, the child winning at least ``c`` of them.

    Args:
        b (int): Parent right, child wrong.
        c (int): Parent wrong, child right.

    Returns:
        McNemar: The counts and the exact p-value (1.0 when there are no discordant pairs).

    Raises:
        ValueError: If a count is negative.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    return McNemar(b, c, _half_tail_ge(c, n) if n else 1.0)
