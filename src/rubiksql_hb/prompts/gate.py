"""The offline gate: every check a proposal's release must pass before anything is published.

Zero model calls. The gate re-derives, on a loaded ``Release``, what the runtime loader
already enforces (names, closed meta, reasons, text, spec and document policies, closure) and
adds what only an author-side pass can afford: the render matrix run twice (the stdlib
interpreter against an injected HeavenBase reference renderer) with a per-cell thread CPU time cap,
structural coverage with a hint per gap, surface contracts, a per-cell diff against the parent
release, the change ratio of every changed row (information: a proposal that rewrites more than
a fifth of a document draws a warning, never a failure), the real-corpus ``k of N changed``
count, the keyspace movement plus the cost estimate of re-evaluating from a root run's fired
sets, and, for a release that changes the general rule corpus, the rule-evolution contract
(:func:`check_rules_corpus`: labelled provenance, winning adjudications, the text audits of
``scripts/audit_rule_corpus.py --check``). Changes are counted per surface
(:func:`changed_surfaces`): a release that cuts the same texts into other rows changes none. The result is a :class:`GateReport` that
serialises to JSON and to a short text summary; a report with any failed check must never
publish (:func:`require_passed`).

HeavenBase reference renderer protocol::

    hb_render(row: Mapping, part_rows: Mapping[str, Mapping], params: Mapping) -> str

``row`` is the exported HB row of the entry, ``part_rows`` every row its closure reaches
keyed by row name (a spec row's ``meta.parts`` maps slots to those names), ``params`` the cell
arguments. When no renderer is injected the HB half is skipped and the report says so.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import entries as E
from .canon import canon, sha256_hex
from .cells import Gap, all_cells, coverage_report
from .corpus import Corpus, CorpusStats, read_rows, stats
from .doc import DocError, Document, compile_document
from .entity_model import CLOSURE_V2, MIGRATION_KIND, EntityModelError, closure_view
from .matrix import Cell
from .metrics import METRICS, resolve_metric
from .policy import (
    OUTPUT_MAX_CHARS,
    PARAM_MAX_CHARS,
    PROPOSAL_RE,
    RELEASE_RE,
    SPEC_MAX_BYTES,
    SPEC_MAX_DEPTH,
    SPEC_MAX_NODES,
    PolicyError,
    check_document,
    check_meta_keys,
    check_name,
    check_reason,
    check_text,
    check_version,
)
from .rules import change_summary, corpus_text
from .spec import INT_MAX_CHARS, Composer, RenderContext, RenderError, TextLeaf
from .store import PACKAGE_STORE_DIR, PromptStoreError, Store, V0Scope, rule_corpus_path

__all__ = [
    "CHANGE_RATIO_WARN",
    "EXPOSURES",
    "HBRender",
    "LOOP_AUTHOR_SUFFIX",
    "MECHANISMS",
    "PLAN_FIELDS",
    "PLAN_METRICS",
    "PRIOR_RELATIONS",
    "RENDER_CONTRACTS",
    "RETIRED_PLAN_FIELDS",
    "RULE_AUDIT_SCRIPT",
    "SCHEMA_GATE",
    "ChangeStat",
    "Check",
    "CostEstimate",
    "GateError",
    "GateReport",
    "RenderContract",
    "RootRun",
    "Unit",
    "change_stat",
    "changed_surfaces",
    "check_change_ratio",
    "check_closure",
    "check_contracts",
    "check_corpus",
    "check_coverage",
    "check_keyspace_cost",
    "check_names_meta_reason",
    "check_parent_cells",
    "check_required_rows",
    "check_rules_corpus",
    "check_spec_policy",
    "check_text_policy",
    "closure_rows",
    "estimate_cost",
    "keyspace_moves",
    "prior_problems",
    "require_passed",
    "row_text",
    "run_gate",
    "run_matrix",
]

SCHEMA_GATE = "rubiksql.gate/1"
#: Hypothesis mechanisms a proposal may claim; ``refactor`` re-cuts texts into other rows without changing a byte.
MECHANISMS = ("wording", "structure", "composition", "addition", "removal", "refactor")
#: Exposure predicates a plan may name (``evolve.EXPOSURES``; repeated here so the gate imports nothing from evolve).
EXPOSURES = ("diverged:prompt", "fired:changed")
#: The fields a plan holds: the metric the promotion rule reads, the exposure slice reported beside it, and the
#: control run it pairs with (``{"namespace": <cache namespace>, "max_age_days": <drift window>}``).
PLAN_FIELDS = ("metric", "exposure", "control")
#: The metric a plan may name: the promotion rule is an EX gain on every shared question (``evolve.promotion_verdict``).
PLAN_METRICS = ("ex",)
#: Plan fields of the retired holdout/alpha design; a plan carrying one is refused so its author learns they mean nothing.
RETIRED_PLAN_FIELDS = ("alpha", "direction", "equivalence_bound_pp", "eval_on", "holdout_rule", "min_effect_pp")
#: How a proposal relates to an earlier proposal or row version it names in ``hypothesis.prior``: it carries that idea
#: further (``builds_on``), reverts or contradicts part of it (``departs_from``), or proposes again what did not promote,
#: on a base that changed since (``retries``).
PRIOR_RELATIONS = ("builds_on", "departs_from", "retries")
#: The fields of one ``hypothesis.prior`` entry.
_PRIOR_FIELDS = ("ref", "relation", "note")
#: How the evolution loop's author label ends (``agent:<claude|codex>/evolve-loop``): the gate requires
#: ``hypothesis.prior`` of an agent proposal so labelled, and checks the field of any other proposal only where it is
#: written, so a person's proposal and every proposal written before the field stay valid.
LOOP_AUTHOR_SUFFIX = "/evolve-loop"
#: A HeavenBase reference renderer: ``(row, part_rows, params) -> text``.
HBRender = Callable[[Mapping[str, Any], Mapping[str, Mapping[str, Any]], Mapping[str, Any]], str]

_CANDIDATE_LINE_RE = re.compile(r"(?m)^Candidate (\d+):")
#: A typed version a ``hypothesis.prior`` ref names: a typed closure key (:func:`entity_model.closure_view`) and the first
#: 16 hex of its value in a release of the history: ``rule:<title>@<16 hex>``, ``tool:<name>@…``, ``parameters:knobs@…``,
#: ``rules:corpus@…`` or ``tools:loader@…``.
_TYPED_PRIOR_RE = re.compile(r"(?P<key>(?:rule|tool):[^@\s:]+|rules:corpus|tools:loader|parameters:knobs)"
                             r"@(?P<version>[0-9a-f]{16})")


class GateError(RuntimeError):
    """A gate report with failed checks was asked to publish."""


@dataclass(frozen=True)
class Check:
    """One gate check.

    Attributes:
        name (str): Check name.
        status (str): ``pass``, ``fail``, ``warn`` or ``skip``.
        detail (str): One line for humans.
        data (Mapping[str, Any]): JSON-compatible evidence (diffs, counts, hints).
    """

    name: str
    status: str
    detail: str
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class GateReport:
    """The gate's verdict on one release.

    Attributes:
        release (str): Release id checked.
        parent (str | None): Parent release id compared against.
        checks (list[Check]): Checks in pipeline order.
    """

    release: str
    parent: str | None
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether no check failed.

        Returns:
            bool: ``True`` when every check is ``pass``, ``warn`` or ``skip``.
        """
        return all(check.status != "fail" for check in self.checks)

    def failures(self) -> list[Check]:
        """The failed checks.

        Returns:
            list[Check]: Checks with status ``fail``.
        """
        return [check for check in self.checks if check.status == "fail"]

    def to_json(self) -> dict[str, Any]:
        """Serialise the report.

        Returns:
            dict[str, Any]: ``{"schema": "rubiksql.gate/1", "release", "parent", "passed", "checks": [...]}``.
        """
        return {
            "schema": SCHEMA_GATE,
            "release": self.release,
            "parent": self.parent,
            "passed": self.passed,
            "checks": [{"name": check.name, "status": check.status, "detail": check.detail, "data": _jsonable(check.data)} for check in self.checks],
        }

    def summary(self) -> str:
        """The short text form: a header and one line per check.

        Returns:
            str: Multi-line text.
        """
        header = f"gate {self.release}" + (f" vs {self.parent}" if self.parent else "") + f": {'PASS' if self.passed else 'FAIL'}"
        lines = [header]
        for check in self.checks:
            lines.append(f"  {check.status.upper():<4} {check.name:<18} {check.detail}")
        return "\n".join(lines)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Gap):
        return {"row": value.row, "path": value.path, "kind": value.kind, "outcome": value.outcome, "entries": list(value.entries), "driver": value.driver, "hint": value.hint}
    return value


def require_passed(report: GateReport) -> None:
    """Refuse to continue on a failed gate.

    Args:
        report (GateReport): The report.

    Returns:
        None: This function does not return a value.

    Raises:
        GateError: If any check failed; the message lists them. A failed gate publishes nothing.
    """
    failures = report.failures()
    if failures:
        raise GateError("gate failed; nothing is published: " + "; ".join(f"{check.name}: {check.detail}" for check in failures))


# ---------------------------------------------------------------------------------------
# Row-level policies


def _meta(release: Any, name: str) -> Mapping[str, Any]:
    return release.row(name)["manifest"]["meta"]


def _reason(release: Any, name: str) -> str:
    return str(release.row(name)["manifest"].get("reason", ""))


def check_names_meta_reason(release: Any, parent: Any | None = None, proposal: Mapping[str, Any] | None = None, store: Store | None = None) -> Check:
    """Names, closed meta tables and reasons of every row, plus the proposal's reason, hypothesis and plan.

    A row whose content differs from the parent's row of the same name must carry a reason that
    is not the parent's. An agent proposal must carry a hypothesis (claim, mechanism, expected
    metric/direction/min_effect_pp, evidence, risks, falsifier) and a plan (:data:`PLAN_FIELDS`).
    The expected direction may be ``+``, ``-`` or ``0`` (a refactor, such as the one-document
    re-cut); it is information, because the promotion rule promotes an EX gain only, so a ``-`` or
    ``0`` proposal reaches BASELINE only by a person's ``promote --override-verdict``. A hypothesis
    that names earlier proposals or row versions (``prior``) must name them well
    (:func:`prior_problems`), and an agent proposal of the evolution loop (its author ends with
    :data:`LOOP_AUTHOR_SUFFIX`) must carry the field, so its author learns of a missing ``prior`` in
    the same report as of everything else. Every row changed from the base release, including removed rows and
    changes brought by a merge, must cite its current version or a proposal that actually changed it. A first slot
    document may cite the shared candidate it inherits; a genuinely new row needs a reason naming it in ``prior_note``.
    A proposal whose parents lie in the store's v0 history joins that history: its refs, the proposals that cover a
    row and the earlier versions that make a row re-added all come from it (:func:`prior_problems`).

    Args:
        release (Release): The release under test.
        parent (Release | None): The parent release.
        proposal (Mapping[str, Any] | None): The proposal body.
        store (Store | None): The store the refs of ``hypothesis.prior`` must exist in. Without one, row references
            can cover the base's current versions, but proposal and added-row history cannot be verified.

    Returns:
        Check: ``pass`` or ``fail`` with every violation listed.
    """
    problems: list[str] = []
    for name in sorted(release.refs):
        try:
            check_name(name)
            check_meta_keys(_meta(release, name))
            parent_reason = None
            if parent is not None and name in parent.refs and parent.refs[name] != release.refs[name]:
                parent_reason = _reason(parent, name)
            check_reason(_reason(release, name), parent_reason)
        except PolicyError as error:
            problems.append(f"{name}: {error}")
    if proposal is not None:
        problems.extend(_proposal_problems(proposal, store))
        if proposal.get("origin", "agent") == "agent":
            problems.extend(_prior_coverage_problems(release, parent, proposal, store))
    if problems:
        return Check("names_meta_reason", "fail", f"{len(problems)} violation(s)", {"violations": problems})
    return Check("names_meta_reason", "pass", f"{len(release.refs)} rows" + (", proposal ok" if proposal is not None else ""))


def _proposal_problems(proposal: Mapping[str, Any], store: Store | None = None) -> list[str]:
    problems: list[str] = []
    try:
        check_reason(proposal.get("reason"))
    except PolicyError as error:
        problems.append(f"proposal.reason: {error}")
    origin = proposal.get("origin", "agent")
    if origin in {"cutover", MIGRATION_KIND}:
        return problems  # Authority alone admits these exact, explicitly authorized unpaid closures.
    if origin not in {"agent", "human"}:
        problems.append(f"proposal.origin: unsupported publication origin {origin!r}")
    hypothesis = proposal.get("hypothesis")
    plan = proposal.get("plan")
    if origin == "agent" or hypothesis is not None:
        if not isinstance(hypothesis, Mapping):
            problems.append("proposal.hypothesis: required for agent proposals")
        else:
            for key in ("claim", "mechanism", "expected", "evidence", "risks", "falsifier"):
                if key not in hypothesis:
                    problems.append(f"proposal.hypothesis.{key}: missing")
            if hypothesis.get("mechanism") not in MECHANISMS:
                problems.append(f"proposal.hypothesis.mechanism: must be one of {list(MECHANISMS)}")
            expected = hypothesis.get("expected")
            if not isinstance(expected, Mapping):
                problems.append("proposal.hypothesis.expected: must be {metric, direction, min_effect_pp}")
            else:
                try:
                    resolve_metric(str(expected.get("metric")))
                except KeyError:
                    problems.append(f"proposal.hypothesis.expected.metric: unknown metric {expected.get('metric')!r}; registered {sorted(METRICS)}")
                if expected.get("direction") not in ("+", "-", "0"):
                    problems.append("proposal.hypothesis.expected.direction: must be '+', '-' or '0'")
                if not isinstance(expected.get("min_effect_pp"), (int, float)) or isinstance(expected.get("min_effect_pp"), bool):
                    problems.append("proposal.hypothesis.expected.min_effect_pp: must be a number")
    if isinstance(hypothesis, Mapping):
        problems.extend(prior_problems(hypothesis, store, required=origin == "agent", parents=_parent_releases(proposal)))
    if origin == "agent" or plan is not None:
        if not isinstance(plan, Mapping):
            problems.append("proposal.plan: required for agent proposals")
        else:
            problems.extend(_plan_problems(plan))
    return problems


def prior_problems(hypothesis: Mapping[str, Any], store: Store | None = None, *, required: bool = False, parents: Iterable[str] = ()) -> list[str]:
    """What is wrong with the earlier proposals and row versions a hypothesis says it relates to (``prior``).

    ``prior`` is a list of ``{"ref", "relation", "note"}``: ``ref`` a proposal id (``p<16 hex>``), a prompt row version
    (``name:version``, the 16-hex version, as the lab's prompt versions and the dossier's row pointers write them) or a
    typed version of a closure/2 release (``<typed key>@<16 hex>``: ``rule:<title>``, ``tool:<name>``,
    ``parameters:knobs``, ``rules:corpus`` or ``tools:loader`` and the first 16 hex of its value in a release of the
    history, :func:`entity_model.closure_view`; for a rule, tool or parameter set that is the row's version),
    ``relation`` one of :data:`PRIOR_RELATIONS` and ``note`` a non-empty string saying how. An empty list needs
    ``prior_note``, a reason (:func:`policy.check_reason`) why no earlier version or proposal is relevant. The field
    names what the author read and cites; it restricts nothing a proposal may change. This function checks the
    references themselves; :func:`check_names_meta_reason` also checks that a loop proposal covers every changed row.

    Evolution restarted at v0 (the user's order of 2026-09-30), and the lab the proposer reads holds that history alone:
    a proposal whose parents all lie in the store's v0 history (:meth:`Store.v0_scope`) joins it, so each ref must be a
    proposal of that history or a row version one of its releases carries. A proposal outside it, a stored pre-v0 one
    among them, keeps the rule it was written under.

    Args:
        hypothesis (Mapping[str, Any]): The proposal's hypothesis.
        store (Store | None): The store every ref must exist in; without one only the refs' form is checked.
        required (bool): Refuse a hypothesis without ``prior``: the gate requires it of the evolution loop's agent
            proposals (:data:`LOOP_AUTHOR_SUFFIX`). Otherwise the field is checked only when present, so every proposal
            written before it existed, and a person's, stays valid.
        parents (Iterable[str]): The release ids of the proposal's parents (base and merges), which decide whether it
            joins the v0 history; none checks the refs against the whole store.

    Returns:
        list[str]: One line per problem, each naming its field (``proposal.hypothesis.prior[1].relation: …``); empty
            when the field is valid, or absent and not required.

    Raises:
        PromptStoreError: When the store's v0 history cannot be read (:meth:`Store.v0_scope`).
    """
    where = "proposal.hypothesis.prior"
    if "prior" not in hypothesis:
        if required:
            return [f"{where}: missing"]
        return [f"{where}: missing; prior_note explains an empty prior list"] if "prior_note" in hypothesis else []
    prior = hypothesis["prior"]
    if not isinstance(prior, list):
        return [f"{where}: must be a list of {{ref, relation, note}} objects"]
    scope = _v0_history(store, parents)
    views: dict[str, dict[str, str]] = {}
    problems: list[str] = []
    if not prior:
        try:
            check_reason(hypothesis.get("prior_note"))
        except PolicyError as error:
            problems.append(f"proposal.hypothesis.prior_note: an empty prior list needs a note saying why no earlier version or proposal is relevant ({error})")
    for index, item in enumerate(prior):
        at = f"{where}[{index}]"
        if not isinstance(item, Mapping):
            problems.append(f"{at}: must be an object {{ref, relation, note}}")
            continue
        unknown = sorted(str(key) for key in item if key not in _PRIOR_FIELDS)
        if unknown:
            problems.append(f"{at}: unknown field(s) {unknown}; an entry holds {list(_PRIOR_FIELDS)}")
        if item.get("relation") not in PRIOR_RELATIONS:
            problems.append(f"{at}.relation: {item.get('relation')!r} is not one of {list(PRIOR_RELATIONS)}")
        if not isinstance(item.get("note"), str) or not item["note"].strip():
            problems.append(f"{at}.note: say how this proposal relates to the ref")
        problem = _prior_ref_problem(item.get("ref"), store, scope, views)
        if problem:
            problems.append(f"{at}.ref: {problem}")
    return problems


def _prior_ref_problem(ref: Any, store: Store | None, scope: V0Scope | None = None,
                       views: dict[str, dict[str, str]] | None = None) -> str:
    """Why a ``prior`` ref names no proposal, row version or typed version (of ``store``, when given; of ``scope``, the
    v0 history the proposal joins, when given), or ``""``; ``views`` caches release closure keys across refs."""
    typed = _TYPED_PRIOR_RE.fullmatch(ref) if isinstance(ref, str) else None
    if typed is not None:
        if store is None:
            return ""
        if typed["version"] in _key_versions(store, typed["key"], _history_releases(store, scope), views):
            return ""
        held = f"the v0 history rooted at {scope.root}" if scope is not None else "the store"
        return f"no release of {held} carries version {typed['version']} of {typed['key']}"
    if isinstance(ref, str) and PROPOSAL_RE.fullmatch(ref):
        if store is None:
            return ""
        try:
            store.proposal(ref)
        except PromptStoreError:
            return f"no proposal {ref} in the store"
        return _before_v0("proposal", ref, scope) if scope is not None and ref not in scope.proposals else ""
    name, _, version = ref.rpartition(":") if isinstance(ref, str) else ("", "", "")
    try:
        check_name(name)
        check_version(version)
    except PolicyError:
        return (f"{ref!r} is neither a proposal id (p<16 hex>), a row version (name:version, the 16-hex version) nor a "
                "typed version (<typed key>@<16 hex>)")
    if store is None:
        return ""
    try:
        store.object(name, version)
    except PromptStoreError:
        return f"no row version {ref} in the store"
    return _before_v0("row version", ref, scope) if scope is not None and version not in store.v0_versions(scope, name) else ""


def _before_v0(kind: str, ref: str, scope: V0Scope) -> str:
    """The refusal of a ``prior`` ref that lies outside the v0 history its proposal joins."""
    return (f"{kind} {ref} lies outside the v0 history rooted at {scope.root}: evolution restarted at v0, and the lab's "
            "History recipes read that history alone; cite a proposal or row version from it")


def _parent_releases(proposal: Mapping[str, Any]) -> list[str]:
    """The release ids a proposal body names as its parents (base and merges)."""
    return [str(parent.get("release")) for parent in proposal.get("parents") or [] if isinstance(parent, Mapping)]


def _v0_history(store: Store | None, parents: Iterable[str]) -> V0Scope | None:
    """The v0 history a proposal with these parents joins: the store's, when every parent lies in it; else ``None``."""
    held = set(parents)
    if store is None or not held:
        return None
    scope = store.v0_scope()
    return scope if scope is not None and held <= scope.releases else None


def _typed(release: Any) -> bool:
    """Whether a loaded release is a closure/2 release (typed tools, rules and parameters)."""
    return getattr(release, "rules", None) == CLOSURE_V2


def _release_view(release: Any) -> dict[str, str]:
    """A loaded release's comparable closure keys: prompt names and, for closure/2, its typed keys."""
    if not _typed(release):
        return dict(release.refs)
    return closure_view(release.manifest, release.typed_rows)


def _store_view(store: Store, rid: str, views: dict[str, dict[str, str]] | None = None) -> dict[str, str]:
    """A stored release's closure keys (:func:`entity_model.closure_view`), cached in ``views`` when given."""
    if views is not None and rid in views:
        return views[rid]
    document = store.release(rid).data
    try:
        view = closure_view(document, store.typed_index() if document.get("rules") == CLOSURE_V2 else {})
    except EntityModelError as error:
        raise PromptStoreError(f"release {rid}: {error}") from error
    if views is not None:
        views[rid] = view
    return view


def _stored_ids(store: Store, directory: str, pattern: re.Pattern[str]) -> set[str]:
    """The ids of the files one store directory holds (``releases`` or ``proposals``)."""
    return {path.stem for layer in store.layers for path in (layer.path / directory).glob("*.json")
            if pattern.fullmatch(path.stem)}


def _history_releases(store: Store, scope: V0Scope | None) -> set[str]:
    """The releases a citation may name a version of: the v0 history's, else every release the store holds."""
    return set(scope.releases) if scope is not None else _stored_ids(store, "releases", RELEASE_RE)


def _key_versions(store: Store, key: str, releases: Iterable[str],
                  views: dict[str, dict[str, str]] | None = None) -> set[str]:
    """The 16-hex versions of one closure key that the given releases carry."""
    return {view[key][:16] for rid in sorted(releases) if key in (view := _store_view(store, rid, views))}


def _earlier_releases(store: Store, proposal: Mapping[str, Any], scope: V0Scope | None) -> set[str]:
    """Releases of the history other than the proposal's own that no later proposal alone produced, so later
    publications do not change a stored proposal's validity."""
    created = proposal.get("created_at")
    producers: dict[str, list[Any]] = {}
    pids = scope.proposals if scope is not None else _stored_ids(store, "proposals", PROPOSAL_RE)
    for pid in sorted(pids):
        if pid != proposal.get("id"):
            body = store.proposal(pid).data
            producers.setdefault(str(body.get("release")), []).append(body.get("created_at"))

    def earlier(rid: str) -> bool:
        times = producers.get(rid)
        return not times or not _number(created) or any(not _number(at) or at <= created for at in times)

    return {rid for rid in _history_releases(store, scope) if rid != proposal.get("release") and earlier(rid)}


def _proposal_changed_rows(store: Store, pid: str) -> set[str]:
    """Rows and typed keys a published proposal actually changed, derived from its releases rather than its change
    summary (:func:`entity_model.closure_view`; a closure/1 release's keys are its prompt rows)."""
    proposal = store.proposal(pid).data
    try:
        bases = [parent["release"] for parent in proposal["parents"] if parent.get("role", "base") == "base"]
        if len(bases) != 1:
            raise PromptStoreError(f"proposal {pid}: expected one base release")
        before = _store_view(store, str(bases[0]))
        after = _store_view(store, str(proposal["release"]))
        return {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    except (KeyError, TypeError, AttributeError) as error:
        raise PromptStoreError(f"proposal {pid}: invalid base or child release references") from error


def _row_has_prior_version(store: Store, name: str, proposal: Mapping[str, Any], scope: V0Scope | None = None) -> bool:
    """Whether a row existed before this proposal, in ``scope`` (the v0 history it joins) when given; later
    publications do not change a stored proposal's validity."""
    held = store.v0_versions(scope, name) if scope is not None else None
    for layer in store.layers:
        for path in (layer.path / "objects" / name).glob("*.json"):
            if held is not None and path.stem not in held:
                continue  # a version before v0 is history the lab does not hold, and no v0 proposal could cover it
            row = store.object(name, path.stem).data["row"]
            origin = row["manifest"]["meta"].get("proposal")
            if origin and origin == proposal.get("id"):
                continue
            if origin and _number(proposal.get("created_at")):
                try:
                    created = store.proposal(origin).data.get("created_at")
                except PromptStoreError:
                    return True  # an untracked object's history cannot be waived
                if _number(created) and created > proposal["created_at"]:
                    continue
            return True
    return False


def _prior_coverage_problems(release: Any, parent: Any | None, proposal: Mapping[str, Any], store: Store | None) -> list[str]:
    """Coverage of the actual base-to-child diff by the loop's historical citations; extra context remains allowed.

    The diff is every closure key (:func:`entity_model.closure_view`): prompt rows, and on closure/2 releases each
    changed ``rule:<title>``, ``tool:<name>``, ``parameters:knobs``, ``rules:corpus`` and ``tools:loader``. A key the
    parent carries is covered by its current version (``name:<16 hex>`` for a prompt row, ``<key>@<16 hex>`` for a typed
    key) or a proposal that actually changed it; an added typed key with earlier versions in the history needs such a
    proposal, and a genuinely new one a ``prior_note`` naming it.
    """
    where = "proposal.hypothesis.prior"
    if parent is None:
        return [f"{where}: cannot verify changed rows without the base release"]
    hypothesis = proposal.get("hypothesis")
    if not isinstance(hypothesis, Mapping) or not isinstance(hypothesis.get("prior"), list):
        return []  # the shape check already names the missing or malformed field
    before, after = _release_view(parent), _release_view(release)
    changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    if not changed:
        return []
    scope = _v0_history(store, _parent_releases(proposal))
    covered: set[str] = set()
    versions: set[str] = set()
    problems: list[str] = []
    for index, item in enumerate(hypothesis["prior"]):
        if not isinstance(item, Mapping) or not isinstance(item.get("ref"), str):
            continue
        ref = item["ref"]
        if ref == proposal.get("id"):
            problems.append(f"{where}[{index}].ref: a proposal cannot cite itself ({ref})")
        elif PROPOSAL_RE.fullmatch(ref):
            if store is None:
                problems.append(f"{where}[{index}].ref: cannot verify the rows changed by {ref} without a store")
                continue
            if scope is not None and ref not in scope.proposals:
                continue  # outside the v0 history: the ref check refuses it, and it covers no row
            try:
                covered.update(_proposal_changed_rows(store, ref))
            except PromptStoreError as error:
                if not _prior_ref_problem(ref, store):  # a missing proposal is already reported by the shape check
                    problems.append(f"{where}[{index}].ref: cannot verify the rows changed by {ref}: {error}")
        else:
            versions.add(ref)
    for name in sorted(changed - covered):
        current = before.get(name)
        typed_key = ":" in name  # a prompt name never holds ':'
        if current:
            ref = f"{name}@{current[:16]}" if typed_key else f"{name}:{current[:16]}"
            if ref not in versions:
                problems.append(f"{where}: uncovered changed {'typed version' if typed_key else 'row'} {name}; cite current {ref} "
                                f"or a proposal that actually changed {name}")
            continue
        if store is None:
            problems.append(f"{where}: cannot verify whether added {'typed version' if typed_key else 'row'} {name} has earlier "
                            "versions without a store")
            continue
        if typed_key:
            subject = name.split(":", 1)[1]
            if _key_versions(store, name, _earlier_releases(store, proposal, scope)):
                problems.append(f"{where}: uncovered re-added typed version {name}; cite a proposal that actually changed {name}")
                continue
            note = hypothesis.get("prior_note")
            try:
                check_reason(note)
                if not re.search(rf"(?<![\w.-]){re.escape(subject)}(?![\w.-])", note):
                    raise PolicyError(f"name the new {subject}")
            except PolicyError as error:
                problems.append(f"proposal.hypothesis.prior_note: new {name} has no earlier version or proposal; explain why it is "
                                f"new and name it ({error})")
            continue
        if _row_has_prior_version(store, name, proposal, scope):
            problems.append(f"{where}: uncovered re-added row {name}; cite a proposal that actually changed {name}")
        elif name in {E.slot_entry(slot) for slot in range(1, E.CANDIDATE_SLOTS + 1)} and "candidate" in parent.refs:
            ref = f"candidate:{parent.refs['candidate'][:16]}"
            if ref not in versions:
                problems.append(f"{where}: uncovered new slot {name}; cite its inherited current {ref} or a proposal that actually changed {name}")
        else:
            note = hypothesis.get("prior_note")
            try:
                check_reason(note)
                if not re.search(rf"(?<![\w.-]){re.escape(name)}(?![\w.-])", note):
                    raise PolicyError(f"name the new row {name}")
            except PolicyError as error:
                problems.append(f"proposal.hypothesis.prior_note: new row {name} has no earlier version or proposal; explain why it is new and name it ({error})")
    return problems


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _plan_problems(plan: Mapping[str, Any]) -> list[str]:
    """Every field of a plan is validated: the agent authors them, the outcome trusts them.

    A plan holds :data:`PLAN_FIELDS` and nothing else. The promotion rule itself (its gain threshold,
    its question set, the attempt cap) is policy in :mod:`evolve`, never a plan field, and the
    fields of the retired holdout/alpha design are refused by name.
    """
    problems: list[str] = []
    for key in ("metric", "exposure"):
        if key not in plan:
            problems.append(f"proposal.plan.{key}: missing")
    for key in sorted(plan):
        if key in RETIRED_PLAN_FIELDS:
            problems.append(f"proposal.plan.{key}: retired; promotion is one fixed rule on the complete evolution question set (an EX gain of at least evolve.PROMOTE_MIN_GAIN_PP), so remove the field")
        elif key not in PLAN_FIELDS:
            problems.append(f"proposal.plan.{key}: unknown field; a plan holds {list(PLAN_FIELDS)}")
    if "metric" in plan and plan["metric"] not in PLAN_METRICS:
        problems.append(f"proposal.plan.metric: must be one of {list(PLAN_METRICS)} (the promotion rule reads EX; the other metrics are reported as information)")
    if "exposure" in plan and plan["exposure"] not in EXPOSURES:
        problems.append(f"proposal.plan.exposure: must be one of {list(EXPOSURES)}")
    control = plan.get("control")
    if "control" in plan:
        if not isinstance(control, Mapping):
            problems.append("proposal.plan.control: must be {namespace, max_age_days}")
        else:
            extra = sorted(set(control) - {"namespace", "max_age_days"})
            if extra:
                problems.append(f"proposal.plan.control: unknown field(s) {extra}; control holds namespace and max_age_days")
            if "namespace" in control and not isinstance(control["namespace"], str):
                problems.append("proposal.plan.control.namespace: must be a string (the cache namespace both runs share)")
            if "max_age_days" in control and (not _number(control["max_age_days"]) or not 0 < control["max_age_days"] <= 14):
                problems.append("proposal.plan.control.max_age_days: must be a number > 0 and <= 14 (the drift window)")
    return problems


def check_text_policy(release: Any) -> Check:
    """Apply the text field policy to every text row and the document policy, grammar and tag rule to every doc row.

    Args:
        release (Release): The release under test.

    Returns:
        Check: ``pass`` or ``fail`` listing the rows that violate it.
    """
    problems = []
    texts = documents = 0
    for name in sorted(release.refs):
        meta = _meta(release, name)
        try:
            if meta.get("kind") == "text":
                texts += 1
                check_text(meta.get("text"), meta.get("slots"), name=name)
            elif meta.get("kind") == "doc":
                documents += 1
                compile_document(name, check_document(meta.get("text")), meta.get("params") or {})
        except (PolicyError, DocError) as error:
            problems.append(f"{name}: {error}")
    if problems:
        return Check("text_policy", "fail", f"{len(problems)} text or doc row(s) violate their policy", {"violations": problems})
    return Check("text_policy", "pass", f"{texts} text rows, {documents} doc rows")


def _text_bounds(entry: E.Entry) -> dict[str, int]:
    return {slot: PARAM_MAX_CHARS if declared == "str" else INT_MAX_CHARS for slot, declared in entry.params.items()}


def check_spec_policy(release: Any) -> Check:
    """Grammar, types, size bounds and the static output bound of every spec and doc row and entry.

    The 16 KiB spec-JSON limit applies to stored specs; a document's derived spec is held to the node and
    depth limits only (its size is bounded by ``DOC_MAX_CHARS`` in the document policy).

    Args:
        release (Release): The release under test; its specs are already parsed, so this re-checks bounds.

    Returns:
        Check: ``pass`` with the per-entry bounds, or ``fail`` naming the offending row.
    """
    problems = []
    bounds: dict[str, int] = {}
    specs = 0
    for name in sorted(release.refs):
        meta = _meta(release, name)
        if meta.get("kind") not in ("spec", "doc"):
            continue
        specs += 1
        composer = release.renderable(name)
        if not isinstance(composer, Composer) or isinstance(composer, Document) != (meta.get("kind") == "doc"):
            problems.append(f"{name}: {meta.get('kind')} row did not resolve to its executor")
            continue
        if meta.get("kind") == "spec":
            size = len(canon(meta["spec"]).encode("utf-8"))
            if size > SPEC_MAX_BYTES:
                problems.append(f"{name}: spec is {size} bytes; the limit is {SPEC_MAX_BYTES}")
        if composer.spec.node_count > SPEC_MAX_NODES:
            problems.append(f"{name}: {composer.spec.node_count} nodes; the limit is {SPEC_MAX_NODES}")
        if composer.spec.depth > SPEC_MAX_DEPTH:
            problems.append(f"{name}: nesting depth {composer.spec.depth}; the limit is {SPEC_MAX_DEPTH}")
    for name in release.entries:
        known = E.ENTRIES.get(name)
        renderable = release.renderable(name)
        bound = renderable.bound(_text_bounds(known) if known is not None and isinstance(renderable, TextLeaf) else ({slot: PARAM_MAX_CHARS for slot in renderable.slots} if isinstance(renderable, TextLeaf) else None))
        bounds[name] = bound
        if bound > OUTPUT_MAX_CHARS:
            # The bound saturates at OUTPUT_MAX_CHARS + 1 once the traversal crosses the cap
            # (spec._SATURATED), so the exact figure is unknown and would mislead.
            problems.append(f"{name}: static output bound exceeds {OUTPUT_MAX_CHARS}")
    if problems:
        return Check("spec_policy", "fail", f"{len(problems)} violation(s)", {"violations": problems, "bounds": bounds})
    return Check("spec_policy", "pass", f"{specs} spec/doc rows; largest static bound {max(bounds.values(), default=0)} chars", {"bounds": bounds})


def closure_rows(release: Any, entry: str) -> dict[str, Mapping[str, Any]]:
    """Every row an entry's render can reach, keyed by name, excluding the entry itself.

    Args:
        release (Release): The release.
        entry (str): Entry name.

    Returns:
        dict[str, Mapping[str, Any]]: Row name to exported HB row.
    """
    rows: dict[str, Mapping[str, Any]] = {}
    stack = [release.renderable(entry)]
    while stack:
        item = stack.pop()
        if isinstance(item, Composer):
            for part in item.parts.values():
                if part.name not in rows and part.name != entry:
                    rows[part.name] = release.row(part.name)
                    stack.append(part)
    return rows


def check_closure(release: Any) -> Check:
    """The refs are exactly the closure of the entries and the parts form no cycle.

    Args:
        release (Release): The release under test.

    Returns:
        Check: ``pass`` or ``fail`` naming orphans, missing rows or the cycle.
    """
    reachable: set[str] = set()
    problems: list[str] = []

    def walk(name: str, chain: tuple[str, ...]) -> None:
        if name in chain:
            problems.append("cycle: " + " -> ".join(chain + (name,)))
            return
        if name in reachable:
            return
        reachable.add(name)
        item = release.renderable(name)
        if isinstance(item, Composer):
            for part in item.parts.values():
                walk(part.name, chain + (name,))

    for entry in release.entries:
        walk(entry, ())
    orphans = sorted(set(release.refs) - reachable)
    missing = sorted(reachable - set(release.refs))
    if orphans:
        problems.append(f"refs carry rows no entry reaches: {orphans}")
    if missing:
        problems.append(f"entries reach rows outside refs: {missing}")
    if problems:
        return Check("closure", "fail", "; ".join(problems), {"orphans": orphans, "missing": missing})
    return Check("closure", "pass", f"{len(reachable)} rows reachable from {len(release.entries)} entries; acyclic")


def _carries(release: Any, name: str) -> bool:
    """Whether a release carries a required row: a renderable entry, or a closure/2 release's typed knobs or corpus."""
    return (_typed(release) and name in (E.KNOBS_ENTRY, E.RULES_ENTRY)) or release.has(name)


def _explicit_knobs(release: Any) -> dict[str, Any] | None:
    """The knob values a release sets explicitly: its parameter set's, its ``config.knobs`` text's, or ``None``."""
    if _typed(release):
        return dict(release.parameter_values())
    if not release.has(E.KNOBS_ENTRY):
        return None
    return json.loads(release.row(E.KNOBS_ENTRY)["manifest"]["meta"]["text"])


def check_required_rows(release: Any, parent: Any | None) -> Check:
    """The optional rows and explicit knob settings a release must retain from its parent.

    The knobs, the rule corpus and the generator documents (:data:`entries.RELEASE_REQUIRED`) became rows of the
    release on 2026-09-25; every release proposed from one that carries them keeps them — a proposal edits them and
    never removes them — and a ``config.knobs`` it changes keeps every knob its parent explicitly lists. A knob
    absent from an older release's row runs at its code default, including when a proposal edits another knob;
    declaring a new knob must not invalidate historical proposals or require edits to immutable store rows.
    Removing an explicit parent setting is still refused, even when that setting equals its default.
    Slot documents stay optional. A closure/2 release carries the knobs and the rule corpus through its typed parameter
    set and rule references, which its load always resolves; its explicit knobs are the parameter set's values, and
    every explicit parent knob, from a ``config.knobs`` text or a parameter set, must stay explicit in the child.

    Args:
        release (Release): The release under test.
        parent (Release | None): The base parent; ``skip`` when absent.

    Returns:
        Check: ``pass`` or ``fail`` naming the dropped rows and explicit parent knobs a changed row leaves out.
    """
    if parent is None:
        return Check("required_rows", "skip", "no parent release given")
    dropped = [name for name in E.RELEASE_REQUIRED if _carries(parent, name) and not _carries(release, name)]
    unlisted: list[str] = []
    explicit, parent_explicit = _explicit_knobs(release), _explicit_knobs(parent)
    if explicit is not None and parent_explicit is not None and explicit != parent_explicit:
        unlisted = sorted(set(parent_explicit) - set(explicit))
    problems = ([f"drops {', '.join(dropped)}, which its parent carries"] if dropped else []) + ([f"{E.KNOBS_ENTRY} leaves out {', '.join(unlisted)}"] if unlisted else [])
    carried = [name for name in E.RELEASE_REQUIRED if _carries(release, name)]
    if problems:
        return Check("required_rows", "fail", "; ".join(problems) + "; a release keeps the knobs, the rule corpus and the generator documents it inherits, and a knob table it edits keeps every explicit parent setting", {"dropped": dropped, "unlisted_knobs": unlisted})
    return Check("required_rows", "pass", f"carries {len(carried)} of the {len(E.RELEASE_REQUIRED)} store-extension rows" + ("" if carried else " (its parent carries none)"), {"carried": carried})


#: The audit a changed rule corpus runs through, relative to the checkout that holds this package.
RULE_AUDIT_SCRIPT = ("scripts", "audit_rule_corpus.py")
_REPO = Path(__file__).resolve().parents[3]
#: Seconds the rule audit may take; it reads the Mini-Dev 500 evolution questions and the 1534 dev questions once and runs
#: no model.
_RULE_AUDIT_TIMEOUT_S = 600


def _corpus_text(release: Any) -> str | None:
    """The exact text of a release's rule corpus, whatever carries it, or ``None`` when it carries none.

    ``rules.corpus_text`` of :meth:`runtime.Release.rules_general`: a closure/1 release's ``rules.general`` text (the
    loader holds it to that spelling) or the corpus a closure/2 release's rule versions reconstruct.
    """
    try:
        return corpus_text(release.rules_general())
    except KeyError:
        return None


def _run_rule_audit(child: str, parent: str | None) -> tuple[list[str], str | None]:
    """Run ``audit_rule_corpus.py --check`` on a corpus text, against its parent's; ``(problems, error)``."""
    script = _REPO.joinpath(*RULE_AUDIT_SCRIPT)
    if not script.is_file():
        return [], f"{script} is missing; a change to {E.RULES_ENTRY} is audited from a checkout"
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    # This static audit reads the explicit corpus files below. Inherited answering pins
    # would bootstrap an unrelated execution bundle before it can inspect those files.
    for name in ("RUBIKSQL_TOOL_BUNDLE", "RUBIKSQL_PROMPT_SNAPSHOT", "RUBIKSQL_PROMPT_RELEASE"):
        environment.pop(name, None)
    with tempfile.TemporaryDirectory(prefix="rubiksql-rule-audit-") as scratch:
        root = Path(scratch)
        (root / "child.json").write_text(child, encoding="utf-8")
        command = [sys.executable, str(script), "--check", "--rules", str(root / "child.json"), "--out", str(root / "report.json")]
        if parent is not None:
            (root / "parent.json").write_text(parent, encoding="utf-8")
            command += ["--parent", str(root / "parent.json")]
        try:
            completed = subprocess.run(command, cwd=_REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=_RULE_AUDIT_TIMEOUT_S, env=environment)
        except subprocess.TimeoutExpired:
            return [], f"the rule audit did not finish within {_RULE_AUDIT_TIMEOUT_S}s"
        report = root / "report.json"
        if not report.is_file():
            tail = " ".join(completed.stderr.strip().splitlines()[-3:])
            return [], f"the rule audit wrote no report (exit {completed.returncode}): {tail}"
        return [str(problem) for problem in json.loads(report.read_text(encoding="utf-8")).get("problems") or []], None


def _rules_change_text(summary: Mapping[str, Any]) -> str:
    """Describe the change and its recorded advisory shares."""
    def named(titles: Sequence[str]) -> str:
        return ", ".join(titles[:4]) + (f", and {len(titles) - 4} more" if len(titles) > 4 else "")

    parts = [f"{label} {len(summary[key])} ({named(summary[key])})" for key, label in (("added", "added"), ("edited", "edited"), ("removed", "removed")) if summary[key]]
    shares = "; ".join(f"{item['new']} over {item['existing']} {item['follows_new']}:{item['follows_existing']} of {item['applicable']}" for item in summary["adjudications"])
    return f"{E.RULES_ENTRY}: " + (", ".join(parts) or "provenance only") + (f"; advisory adjudication {shares}" if shares else "") + f"; {summary['rules']} rules, {summary['dev_evolved']} dev-evolved; every knowledge fingerprint moves, so every question re-runs"


def check_rules_corpus(release: Any, parent: Any | None) -> Check:
    """Audit a changed corpus's structure, provenance and model-facing text.

    Explicit rule replacements retain advisory adjudications. The static auditor
    checks current evolution input pins, append-only provenance, trigger and text
    conventions, and blocking question/hint/gold leaks. Its mining doctrine
    registry cannot veto an authored rule merely because that title is guarded.
    Closure/1 carriers and closure/2 reconstructed corpora use the same audit.

    Args:
        release (Release): The release under test.
        parent (Release | None): The base parent; ``skip`` when absent.

    Returns:
        Check: ``pass`` when the corpus is unchanged or the change follows the contract (the detail counts the rules
            added, edited and removed and the advisory shares recorded), ``fail`` naming every problem. ``data`` holds
            ``summary`` (:func:`rules.change_summary`, when there is a parent corpus) and ``problems``.
    """
    if parent is None:
        return Check("rules_corpus", "skip", "no parent release given")
    after = _corpus_text(release)
    if after is None:
        if _typed(release):
            return Check("rules_corpus", "fail", f"a {CLOSURE_V2} release carries no reconstructed rule corpus")
        return Check("rules_corpus", "pass", f"carries no {E.RULES_ENTRY}")
    before = _corpus_text(parent)
    if before is None:
        return Check("rules_corpus", "fail", "parent release has no pinned rules.general corpus")
    if after == before:
        return Check("rules_corpus", "pass", f"{E.RULES_ENTRY} unchanged")
    try:
        summary = change_summary(json.loads(before), json.loads(after)) if before is not None else None
    except (ValueError, TypeError, AttributeError):
        summary = None
    problems, error = _run_rule_audit(after, before)
    data = {"summary": summary, "problems": problems}
    if error is not None:
        return Check("rules_corpus", "fail", error, data)
    if problems:
        return Check("rules_corpus", "fail", f"{len(problems)} problem(s) in the {E.RULES_ENTRY} change: " + "; ".join(problems[:3]) + (f"; and {len(problems) - 3} more" if len(problems) > 3 else ""), data)
    return Check("rules_corpus", "pass", (_rules_change_text(summary) if summary is not None else f"{E.RULES_ENTRY}: a corpus with no parent corpus to compare") + "; audit clean", data)


# ---------------------------------------------------------------------------------------
# Matrix, coverage, contracts


@dataclass
class MatrixRun:
    """Rendered matrix cells: the whole render per cell and the text of every surface it holds."""

    check: Check
    texts: dict[tuple[str, str], str] = field(default_factory=dict)
    cells: dict[tuple[str, str], Cell] = field(default_factory=dict)
    surfaces: dict[tuple[str, str], dict[str, str]] = field(default_factory=dict)


def _diff(a: str, b: str, label_a: str, label_b: str, limit: int = 40) -> str:
    lines = list(difflib.unified_diff(a.splitlines(True), b.splitlines(True), label_a, label_b, n=1))
    if len(lines) > limit:
        lines = lines[:limit] + [f"... {len(lines) - limit} more lines\n"]
    return "".join(lines)


def run_matrix(release: Any, matrix: Iterable[Cell], hb_render: HBRender | None = None, *, cell_time_cap_ms: float = 50.0) -> MatrixRun:
    """Render every cell with the interpreter, and with HeavenBase when a renderer is injected.

    Args:
        release (Release): The release under test.
        matrix (Iterable[Cell]): Cells; cells of entries the release lacks are skipped.
        hb_render (HBRender | None): The HeavenBase reference renderer, or ``None`` to skip that half.
        cell_time_cap_ms (float): Interpreter thread CPU time cap per cell. Scheduler delays and work on
            other threads do not consume the budget.

    Returns:
        MatrixRun: The dual-run check plus the interpreter texts by ``(entry, cell)``; the check is ``skip`` (not
            ``pass``) when no renderer was injected, and its data carries ``hb: "skipped"`` or ``"ran"``.
    """
    texts: dict[tuple[str, str], str] = {}
    cells: dict[tuple[str, str], Cell] = {}
    surfaces: dict[tuple[str, str], dict[str, str]] = {}
    slow: list[str] = []
    errors: list[str] = []
    mismatches: dict[str, str] = {}
    count = 0
    for cell in matrix:
        if not release.has(cell.entry):
            continue
        key = (cell.entry, cell.name)
        count += 1
        cells[key] = cell
        # Rendering is synchronous: bound its CPU work, not how long the host leaves this thread unscheduled.
        started = time.thread_time()
        try:
            text = "".join(piece for piece, _src in release.renderable(cell.entry).render_chunks(RenderContext(), cell.params))
        except (RenderError, KeyError, TypeError, ValueError) as error:
            errors.append(f"{cell.entry}/{cell.name}: {type(error).__name__}: {error}")
            continue
        elapsed_ms = (time.thread_time() - started) * 1000.0
        texts[key] = text
        known = E.ENTRIES.get(cell.entry)
        if known is not None and known.surfaces:
            surfaces[key] = dict(zip(known.surfaces, release.messages(cell.entry, **cell.params)))
        if elapsed_ms > cell_time_cap_ms:
            slow.append(f"{cell.entry}/{cell.name}: {elapsed_ms:.1f} ms thread CPU")
        if hb_render is not None:
            try:
                hb_text = hb_render(release.row(cell.entry), closure_rows(release, cell.entry), cell.params)
            except Exception as error:  # a reference renderer failure is a gate failure, whatever its type
                errors.append(f"{cell.entry}/{cell.name}: hb_render raised {type(error).__name__}: {error}")
                continue
            if hb_text != text:
                mismatches[f"{cell.entry}/{cell.name}"] = _diff(text, hb_text, "interpreter", "heavenbase")
    data: dict[str, Any] = {"cells": count, "slow": slow, "errors": errors, "mismatches": mismatches, "hb": "skipped" if hb_render is None else "ran", "time_cap_ms": cell_time_cap_ms, "time_basis": "thread_cpu"}
    if errors or mismatches or slow:
        parts = []
        if mismatches:
            parts.append(f"{len(mismatches)} cell(s) differ between interpreter and HeavenBase")
        if errors:
            parts.append(f"{len(errors)} render error(s)")
        if slow:
            parts.append(f"{len(slow)} cell(s) over {cell_time_cap_ms:g} ms thread CPU")
        return MatrixRun(Check("matrix_dual_run", "fail", "; ".join(parts), data), texts, cells, surfaces)
    if hb_render is None:
        # the interpreter half ran clean, but the design's dual run needs both halves: a skip, never a pass
        return MatrixRun(Check("matrix_dual_run", "skip", f"{count} cells rendered by the interpreter; HB reference execution skipped (no renderer injected), so the dual run did not happen", data), texts, cells, surfaces)
    return MatrixRun(Check("matrix_dual_run", "pass", f"{count} cells byte-equal under interpreter and HeavenBase", data), texts, cells, surfaces)


def check_coverage(release: Any, matrix: Iterable[Cell]) -> Check:
    """Structural coverage of every spec row under the matrix, with a hint per gap.

    Args:
        release (Release): The release under test.
        matrix (Iterable[Cell]): Cells.

    Returns:
        Check: ``pass`` or ``fail`` whose data carries the gaps and hints.
    """
    report = coverage_report(release, list(matrix))
    if report.gaps:
        hints = [gap.line() for gap in report.gaps]
        return Check("coverage", "fail", f"{len(report.gaps)} uncovered branch(es); add cells: " + " | ".join(gap.hint for gap in report.gaps[:3]), {"gaps": list(report.gaps), "hints": hints})
    obligations = sum(len(composer.spec.branch_points()) for composer in release.composers())
    return Check("coverage", "pass", f"{obligations} structural obligations met by {len(report.renders)} cell renders")


@dataclass(frozen=True)
class RenderContract:
    """A check on the rendered text of a parameterised surface, evaluated on every matrix cell that renders it.

    Attributes:
        surface (str): Surface id; the contract reads that message of the cell's render, whichever row provides it.
        name (str): Contract name printed with a violation.
        check (Callable[[Mapping[str, Any], str], str | None]): ``(params, text) -> violation or None``.
    """

    surface: str
    name: str
    check: Callable[[Mapping[str, Any], str], str | None]


def _candidate_numbering(params: Mapping[str, Any], text: str) -> str | None:
    n = min(len(params.get("candidates") or []), 64)
    numbers = [int(match) for match in _CANDIDATE_LINE_RE.findall(text)]
    expected = list(range(1, min(n, 16) + 1))
    if sorted(numbers) != expected:
        missing = sorted(set(expected) - set(numbers))
        extra = sorted(set(numbers) - set(expected))
        duplicates = sorted({number for number in numbers if numbers.count(number) > 1})
        return f"candidate numbering gap: rendered {numbers}, expected the set 1..{len(expected)} (missing {missing}, unexpected {extra}, duplicated {duplicates})"
    return None


def _verbatim(*keys: str) -> Callable[[Mapping[str, Any], str], str | None]:
    def check(params: Mapping[str, Any], text: str) -> str | None:
        for key in keys:
            value = params.get(key)
            if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value) not in text:
                return f"{key} is not rendered verbatim"
        return None

    return check


def _hits_present(params: Mapping[str, Any], text: str) -> str | None:
    for index, hit in enumerate(params.get("hits") or []):
        for key in ("syn", "target"):
            if str(hit.get(key)) not in text:
                return f"hits[{index}].{key} is missing from the rendered block"
    return None


#: Contracts on parameterised surfaces. Contracts on surfaces without parameters live in ``entries.py``.
RENDER_CONTRACTS: tuple[RenderContract, ...] = (
    RenderContract("judge.user", "judge.user.candidate_numbering", _candidate_numbering),
    RenderContract("judge.user", "judge.user.question_verbatim", _verbatim("question")),
    RenderContract("reviewer.user", "reviewer.user.question_and_sql_verbatim", _verbatim("question", "sql")),
    RenderContract("candidate.nudge.time", "candidate.nudge.time.minutes_verbatim", _verbatim("mins")),
    RenderContract("shared.synonyms", "shared.synonyms.hits_present", _hits_present),
)


def check_contracts(release: Any, run: MatrixRun) -> Check:
    """Surface contracts: the ones on surfaces without parameters from ``entries.py`` and the render contracts over every cell.

    Args:
        release (Release): The release under test.
        run (MatrixRun): Output of :func:`run_matrix`.

    Returns:
        Check: ``fail`` on an error-severity violation, ``warn`` on warning-severity ones, else ``pass``.
    """
    errors: list[str] = []
    warnings: list[str] = list(release.warnings)
    passed: list[str] = []
    for surface in E.SURFACES.values():
        if surface.params or not surface.contracts or not release.has_surface(*surface.id.split(".", 1)):
            continue
        rendered = release.surface(*surface.id.split(".", 1))
        for contract in surface.contracts:
            violation = contract.check(rendered)
            if violation is None:
                passed.append(contract.name)
            elif contract.severity == "error":
                errors.append(f"{contract.name}: {violation}")
    for contract in RENDER_CONTRACTS:
        hit = False
        for key, texts in run.surfaces.items():
            if contract.surface not in texts:
                continue
            hit = True
            violation = contract.check(run.cells[key].params, texts[contract.surface])
            if violation is not None:
                errors.append(f"{contract.name} @ {key[1]}: {violation}")
        if hit and not any(line.startswith(contract.name) for line in errors):
            passed.append(contract.name)
    data = {"errors": errors, "warnings": warnings, "passed": sorted(set(passed))}
    if errors:
        return Check("contracts", "fail", f"{len(errors)} contract violation(s): " + errors[0], data)
    if warnings:
        return Check("contracts", "warn", f"{len(set(passed))} contracts pass; {len(warnings)} warning(s) in the seeded content", data)
    return Check("contracts", "pass", f"{len(set(passed))} contracts pass", data)


def _surface_renders(release: Any, run: MatrixRun) -> dict[tuple[str, str], str]:
    """Every surface text a matrix run rendered, keyed by surface and the arguments that surface took.

    Args:
        release (Release): The release the run rendered.
        run (MatrixRun): Its matrix run.

    Returns:
        dict[tuple[str, str], str]: ``(surface id, canonical JSON of its arguments)`` to the rendered message.
    """
    renders: dict[tuple[str, str], str] = {}
    for key, texts in run.surfaces.items():
        params = run.cells[key].params
        for surface, text in texts.items():
            renders[(surface, canon({name: params[name] for name in E.SURFACES[surface].params}))] = text
    return renders


def check_parent_cells(release: Any, parent: Any | None, run: MatrixRun, parent_run: MatrixRun | None) -> Check:
    """Per-cell comparison against the parent release, per-surface comparison across layouts, and the changed rows.

    Cells are compared by entry, so a release that re-cuts the parent's texts into other rows shows its cells
    as added (``+``) and the parent's as removed (``-``). The surface comparison meets both layouts on the
    texts the code sends: every surface message the two matrix runs rendered with the same arguments is
    compared byte for byte, whichever row rendered it (a twin cell renders the same surface with the same
    arguments under the other layout). A surface that renders differently while its identity text is unchanged
    (:func:`changed_surfaces`) fails the check: its keyspaces would not move, so the two releases would share
    caches, resumed stages and outcome pairings while sending different bytes.

    Args:
        release (Release): The release under test.
        parent (Release | None): The parent release; ``skip`` when absent.
        run (MatrixRun): The release's matrix run.
        parent_run (MatrixRun | None): The parent's matrix run over the same cells.

    Returns:
        Check: ``pass`` with ``=``/``~``/``+``/``-`` per cell and a diff per changed cell, and ``surfaces``: how many
            shared surface renders are equal, a diff per differing one, and ``unexplained``: the surfaces whose
            renders differ under an unchanged identity, which make the check ``fail``.
    """
    if parent is None or parent_run is None:
        return Check("parent_diff", "skip", "no parent release given")
    cells: dict[str, str] = {}
    diffs: dict[str, str] = {}
    for key in sorted(set(run.texts) | set(parent_run.texts)):
        label = f"{key[0]}/{key[1]}"
        if key not in parent_run.texts:
            cells[label] = "+"
        elif key not in run.texts:
            cells[label] = "-"
        elif run.texts[key] == parent_run.texts[key]:
            cells[label] = "="
        else:
            cells[label] = "~"
            diffs[label] = _diff(parent_run.texts[key], run.texts[key], parent.id, release.id)
    rows: dict[str, dict[str, str | None]] = {}
    for name in sorted(set(release.refs) | set(parent.refs)):
        before, after = parent.refs.get(name), release.refs.get(name)
        if before != after:
            rows[name] = {"from": before[:16] if before else None, "to": after[:16] if after else None}
    mine, theirs = _surface_renders(release, run), _surface_renders(parent, parent_run)
    shared = sorted(set(mine) & set(theirs))
    differing = [(surface, params) for surface, params in shared if mine[(surface, params)] != theirs[(surface, params)]]
    surface_diffs = {f"{surface} {params}": _diff(theirs[(surface, params)], mine[(surface, params)], parent.id, release.id) for surface, params in differing}
    moved = changed_surfaces(release, parent)
    unexplained = sorted({surface for surface, _params in differing if surface not in moved})
    surfaces = {"shared": len(shared), "equal": len(shared) - len(surface_diffs), "diffs": surface_diffs, "unexplained": unexplained}
    changed = sum(1 for mark in cells.values() if mark == "~")
    detail = f"{len(rows)} row(s) changed; cells: {changed} ~, {sum(1 for m in cells.values() if m == '=')} =, {sum(1 for m in cells.values() if m == '+')} +, {sum(1 for m in cells.values() if m == '-')} -"
    detail += f"; surface renders: {surfaces['equal']} =, {len(surface_diffs)} ~ of {len(shared)} shared"
    data = {"cells": cells, "diffs": diffs, "rows": rows, "surfaces": surfaces}
    if unexplained:
        return Check("parent_diff", "fail", detail + f"; {', '.join(unexplained)} render(s) differ under an unchanged identity text, so no keyspace would move", data)
    return Check("parent_diff", "pass", detail, data)


# ---------------------------------------------------------------------------------------
# Change ratio


#: A document whose text changes by more than this fraction of its characters makes :func:`check_change_ratio` warn,
#: never fail: one proposal should test one idea.
CHANGE_RATIO_WARN = 0.20
#: Words and single other characters, whitespace aside: the tokens a replaced run of lines is compared by, and the
#: characters a change ratio counts.
_CHANGE_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
#: Differing middles that multiply past this many pairs of lines or tokens are matched with difflib's popularity
#: heuristic instead of exactly: exact matching of repetitive text grows with the cube of its length.
_CHANGE_EXACT_PAIRS = 1_000_000
#: Changed rows a change-ratio line names before it counts the rest.
_CHANGE_LINE_ROWS = 6


def _percent(ratio: float) -> str:
    """``ratio`` as a percentage with one decimal, or with as many more (up to four) as it takes to print on its own
    side of :data:`CHANGE_RATIO_WARN`; a change too small for one decimal prints ``<0.1%``."""
    if 0.0 < ratio < 0.0005:
        return "<0.1%"
    for digits in range(1, 5):
        shown = float(f"{ratio * 100:.{digits}f}") / 100
        if ratio == CHANGE_RATIO_WARN or (shown != CHANGE_RATIO_WARN and (shown > CHANGE_RATIO_WARN) == (ratio > CHANGE_RATIO_WARN)):
            return f"{ratio * 100:.{digits}f}%"
    return f"{ratio * 100:.4f}%"


@dataclass(frozen=True)
class ChangeStat:
    """How far one text moved to another (:func:`change_stat`).

    Attributes:
        lines_added (int): Lines of the new text in inserted or replaced runs of lines.
        lines_removed (int): Lines of the old text in deleted or replaced runs of lines.
        chars_before (int): Characters of the old text, whitespace aside.
        chars_after (int): Characters of the new text, whitespace aside.
        ratio (float): The fraction of the document's characters changed, ``1 - matched / max(chars_before,
            chars_after)``: 0.0 for texts that differ at most in whitespace, 1.0 for texts with nothing in common or
            a text against none.
    """

    lines_added: int
    lines_removed: int
    chars_before: int
    chars_after: int
    ratio: float

    def summary(self) -> str:
        """The counts in one phrase.

        Returns:
            str: ``+4/-2 lines, 1.8%``.
        """
        return f"+{self.lines_added}/-{self.lines_removed} lines, {_percent(self.ratio)}"


def _content_chars(text: str) -> int:
    """Characters of ``text`` a change ratio counts: every one but whitespace."""
    return sum(map(len, _CHANGE_TOKEN_RE.findall(text)))


def _equal_runs(old: Sequence[str], new: Sequence[str]) -> list[tuple[int, int, int]]:
    """``(i, j, size)`` runs equal in both sequences, in order: the common ends first, then difflib's matching blocks of
    the middles, exact unless they multiply past :data:`_CHANGE_EXACT_PAIRS` pairs (then difflib's popularity heuristic
    joins in), so an edit in one place of a long run compares only what lies between its equal ends."""
    limit = min(len(old), len(new))
    head = 0
    while head < limit and old[head] == new[head]:
        head += 1
    tail = 0
    while tail < limit - head and old[len(old) - 1 - tail] == new[len(new) - 1 - tail]:
        tail += 1
    runs = [(0, 0, head)] if head else []
    middle_old, middle_new = old[head : len(old) - tail], new[head : len(new) - tail]
    if middle_old and middle_new:
        matcher = difflib.SequenceMatcher(None, middle_old, middle_new, autojunk=len(middle_old) * len(middle_new) > _CHANGE_EXACT_PAIRS)
        runs.extend((head + i, head + j, size) for i, j, size in matcher.get_matching_blocks() if size)
    if tail:
        runs.append((len(old) - tail, len(new) - tail, tail))
    return runs


def change_stat(before: str, after: str) -> ChangeStat:
    """Measure how far ``before`` moved to ``after``: lines added and removed, and the fraction of characters changed.

    The measure is difflib's, in two steps: the texts are matched line by line, then each run of lines replaced by
    other lines token by token (words and single other characters, whitespace aside), so an edit inside a long line
    counts its words, not the line, and a re-wrapped paragraph counts nothing. Both steps take the common ends first
    and match what lies between them with ``SequenceMatcher``, exactly unless the two sides multiply past a million
    pairs (difflib's popularity heuristic then joins in, faster and coarser on repetitive text). ``matched`` counts
    the characters, whitespace aside, of the equal lines and of the equal tokens inside replaced runs, and ``ratio =
    1 - matched / max(chars_before, chars_after)``: the larger of the share of the old text deleted and the share of
    the new text inserted. Rewriting a fifth of a document, adding a quarter to it and deleting a fifth of it each
    measure 20%.

    Args:
        before (str): The old text; ``""`` for none.
        after (str): The new text; ``""`` for none.

    Returns:
        ChangeStat: The line counts, the character counts and the ratio.
    """
    old, new = before.splitlines(keepends=True), after.splitlines(keepends=True)
    runs = _equal_runs(old, new)
    matched = sum(_content_chars("".join(old[i : i + size])) for i, _j, size in runs)
    old_at = new_at = 0
    for i, j, size in [*runs, (len(old), len(new), 0)]:
        if i > old_at and j > new_at:  # lines replaced by other lines: compare their tokens
            old_tokens, new_tokens = _CHANGE_TOKEN_RE.findall("".join(old[old_at:i])), _CHANGE_TOKEN_RE.findall("".join(new[new_at:j]))
            matched += sum(len(token) for ti, _tj, tsize in _equal_runs(old_tokens, new_tokens) for token in old_tokens[ti : ti + tsize])
        old_at, new_at = i + size, j + size
    kept = sum(size for _i, _j, size in runs)
    chars_before, chars_after = _content_chars(before), _content_chars(after)
    longest = max(chars_before, chars_after)
    return ChangeStat(len(new) - kept, len(old) - kept, chars_before, chars_after, 1.0 - matched / longest if longest else 0.0)


def row_text(meta: Mapping[str, Any]) -> str:
    """The text a row's change is measured on.

    Args:
        meta (Mapping[str, Any]): The row's ``manifest.meta``.

    Returns:
        str: A text or doc row's ``meta.text``; a spec row's spec JSON as ``prompts draft --spec`` prints it.
    """
    if meta.get("kind") == "spec":
        return json.dumps(meta["spec"], ensure_ascii=False, indent=1) + "\n"
    return str(meta["text"])


def _stand_in(name: str, release: Any) -> str | None:
    """The row that renders in place of ``name`` in a release without it: a slot document's shared document, or ``None``.

    A slot document (``candidate.s2``, :func:`entries.slot_entry`) renders in place of the shared ``candidate`` document
    for its slot (``Release.slot_surface``), so one added or removed is measured against the shared document.
    """
    entry = E.ENTRIES.get(name)
    shared = "candidate" if entry is not None and entry.slot is not None else None
    return shared if shared is not None and shared in release.refs else None


def check_change_ratio(release: Any, parent: Any | None) -> Check:
    """How much each changed row's document moved against the parent: information, ``warn`` at most, never ``fail``.

    Every row whose content key differs from the parent's is measured with :func:`change_stat` on :func:`row_text`:
    the lines added and removed and the fraction of the document's characters changed. A slot document added or
    removed (``candidate.s2``) is measured against the shared document (``candidate``) its slot rendered before or
    renders after; any other row added or removed is a whole document, listed as ``added``/``removed`` and never
    measured against the threshold. A measured document that changes by more than :data:`CHANGE_RATIO_WARN` makes
    the check ``warn``: one proposal should test one idea.

    Args:
        release (Release): The release under test.
        parent (Release | None): The parent release; ``skip`` when absent.

    Returns:
        Check: ``pass`` or ``warn`` with one summary line. ``data`` holds ``threshold``; ``rows``, row name to its
            ``before`` and ``after`` rows (``None`` for none), ``lines_added``, ``lines_removed``, ``chars_before``,
            ``chars_after``, ``ratio`` and ``summary``; ``max_ratio``, the largest ratio of a measured document (0.0
            when none); and ``over``, the measured documents above the threshold.
    """
    if parent is None:
        return Check("change_ratio", "skip", "no parent release given")
    rows: dict[str, dict[str, Any]] = {}
    measured: dict[str, float] = {}
    for name in sorted(set(release.refs) | set(parent.refs)):
        if release.refs.get(name) == parent.refs.get(name):
            continue
        before = name if name in parent.refs else _stand_in(name, parent)
        after = name if name in release.refs else _stand_in(name, release)
        old = row_text(parent.row(before)["manifest"]["meta"]) if before else ""
        stat = change_stat(old, row_text(release.row(after)["manifest"]["meta"]) if after else "")
        if before and after:
            measured[name] = stat.ratio
            if before == after:
                summary = stat.summary()
            elif after == name:  # a slot document added: its slot rendered the shared document before
                summary = f"added, against {before}: {stat.summary()}"
            else:  # a slot document removed: its slot renders the shared document now
                summary = f"removed, against {after}: {stat.summary()}"
        else:
            count = stat.lines_added if after else stat.lines_removed
            summary = f"{'added' if after else 'removed'}, {count} line{'' if count == 1 else 's'}"
        rows[name] = {
            "before": before,
            "after": after,
            "lines_added": stat.lines_added,
            "lines_removed": stat.lines_removed,
            "chars_before": stat.chars_before,
            "chars_after": stat.chars_after,
            "ratio": round(stat.ratio, 6),
            "summary": summary,
        }
    over = sorted(name for name, ratio in measured.items() if ratio > CHANGE_RATIO_WARN)
    data = {"threshold": CHANGE_RATIO_WARN, "rows": rows, "max_ratio": round(max(measured.values(), default=0.0), 6), "over": over}
    if not rows:
        return Check("change_ratio", "pass", "no row changed", data)
    named = [f"{name} {row['summary']}" for name, row in rows.items()]
    detail = f"{len(rows)} row(s) changed: " + ", ".join(named[:_CHANGE_LINE_ROWS]) + (f", and {len(named) - _CHANGE_LINE_ROWS} more" if len(named) > _CHANGE_LINE_ROWS else "")
    if over:
        spread = ", ".join(f"{name} ({_percent(measured[name])})" for name in over)
        return Check("change_ratio", "warn", f"{detail}; more than {CHANGE_RATIO_WARN:.0%} of the characters changed in {spread}: one proposal should test one idea", data)
    if measured:
        largest = max(measured, key=lambda name: (measured[name], name))
        detail += f"; largest {largest} {_percent(measured[largest])}, none over {CHANGE_RATIO_WARN:.0%}"
    return Check("change_ratio", "pass", detail, data)


# ---------------------------------------------------------------------------------------
# Corpus


def check_corpus(corpus: Corpus | None, parent: Any | None, release: Any, *, samples: int = 3) -> Check:
    """Real-corpus ``k of N changed`` between the parent and the release, with sample diffs.

    Args:
        corpus (Corpus | None): A locally built corpus; ``skip`` when absent.
        parent (Release | None): The parent release; ``skip`` when absent.
        release (Release): The release under test.
        samples (int): Sample diffs to attach.

    Returns:
        Check: ``pass`` (informational) with the counts, diffs and the changed entries the corpus never reaches.
    """
    if corpus is None or parent is None:
        return Check("corpus", "skip", "no corpus or no parent given")
    result: CorpusStats = stats(corpus, parent, release, samples=samples, changed=changed_surfaces(release, parent))
    data = {"n": result.n, "k": result.k, "skipped": result.skipped, "uncovered": list(result.uncovered), "per_entry": {entry: {"n": n, "k": k} for entry, (n, k) in result.per_entry.items()}, "samples": [{"qid": item.sample.qid, "role": item.sample.role, "message": item.sample.message, "diff": item.diff} for item in result.samples]}
    return Check("corpus", "pass", result.summary(), data)


# ---------------------------------------------------------------------------------------
# Keyspace movement and cost


def changed_surfaces(release: Any, parent: Any) -> frozenset[str]:
    """Surfaces whose static text differs between two releases (surfaces one of them lacks included).

    The comparison reads :meth:`runtime.Release.surface_identity`: the text as written, whatever rows
    carry it, so a refactor that re-cuts the texts into other rows changes nothing, and a change to a
    part of a composition reaches the surfaces that include it.

    Args:
        release (Release): The child.
        parent (Release): The parent.

    Returns:
        frozenset[str]: Surface ids.
    """
    mine, theirs = release.surface_identity(), parent.surface_identity()
    return frozenset(surface for surface in set(mine) | set(theirs) if mine.get(surface) != theirs.get(surface))


def keyspace_moves(release: Any, parent: Any) -> dict[str, tuple[str, str]]:
    """Keyspaces whose value differs between two releases.

    Args:
        release (Release): The child.
        parent (Release): The parent.

    Returns:
        dict[str, tuple[str, str]]: Label (``role`` or ``role/route``, and ``candidate[s<k>]``-style for a slot either
            release gives a keyspace of its own) to ``(parent keyspace, child keyspace)``.
    """
    moves = {}
    for role, route in E.KEYSPACES:
        before, after = parent.keyspace(role, route), release.keyspace(role, route)
        if before != after:
            moves[E.route_label(role, route)] = (before, after)
    for route, slot in sorted(_own_slots(release) | _own_slots(parent), key=lambda key: (key[1], E.ROUTES["candidate"].index(key[0]))):
        before, after = parent.keyspace("candidate", route, slot), release.keyspace("candidate", route, slot)
        if before != after:
            moves[E.route_label("candidate", route, slot)] = (before, after)
    return moves


def _own_slots(release: Any) -> set[tuple[str | None, int]]:
    """``(route, slot)`` of every candidate slot the release keys on a keyspace of its own."""
    labels = release.keyspaces()
    return {(route, slot) for route in E.ROUTES["candidate"] for slot in range(1, E.CANDIDATE_SLOTS + 1) if E.route_label("candidate", route, slot) in labels}


def _unit_label(unit: "Unit", release: Any, parent: Any) -> str:
    """The keyspace label a unit's cache key follows: its slot's own when either release gives the slot one."""
    if unit.role == "candidate" and unit.slot is not None:
        label = E.route_label("candidate", unit.route, unit.slot)
        if label in release.keyspaces() or label in parent.keyspaces():
            return label
    return E.route_label(unit.role, unit.route)


@dataclass(frozen=True)
class Unit:
    """One agent run of a root run: what it rendered and, when known, when.

    Attributes:
        qid (str): Question id.
        role (str): ``candidate``, ``judge`` or ``reviewer``.
        route (str | None): Route or ``None``.
        fired (frozenset[str]): Entries the run rendered, as the snapshot names them (either layout).
        turns (int | None): Model turns the run took, when the snapshot recorded them.
        fired_turns (Mapping[str, int] | None): Entry to the 1-based turn at which it first fired (the turn index),
            when available.
        slot (int | None): A candidate's position in the pool, counted from 1; ``None`` for a judge or reviewer.
    """

    qid: str
    role: str
    route: str | None
    fired: frozenset[str]
    turns: int | None = None
    fired_turns: Mapping[str, int] | None = None
    slot: int | None = None


@dataclass(frozen=True)
class RootRun:
    """The fired sets of a root run, read from its result rows' ``prompt_identity`` snapshots."""

    name: str
    units: tuple[Unit, ...]

    @classmethod
    def from_rows(cls, name: str, rows: Iterable[Mapping[str, Any]], *, fired_turns: Mapping[tuple[str, str, int], Mapping[str, int]] | None = None) -> "RootRun":
        """Collect units from eval rows.

        Args:
            name (str): Run name.
            rows (Iterable[Mapping[str, Any]]): Eval rows carrying ``prompt_identity`` on the judge, the reviewer and
                each ``candidates_debug`` record.
            fired_turns (Mapping[tuple[str, str, int], Mapping[str, int]] | None): Optional turn index keyed by
                ``(qid, role, index)`` (index is the candidate slot, ``-1`` for judge/reviewer).

        Returns:
            RootRun: The units.
        """
        units = []
        for row in rows:
            qid = str(row.get("id"))
            holders: list[tuple[int, Mapping[str, Any]]] = [(-1, row.get("judge_debug") or {}), (-1, row.get("reviewer_debug") or {})]
            holders.extend((index, debug) for index, debug in enumerate(row.get("candidates_debug") or []))
            for index, holder in holders:
                snapshot = holder.get("prompt_identity") if isinstance(holder, Mapping) else None
                if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("fired"), Mapping):
                    continue
                role = str(snapshot.get("role"))
                replay = snapshot.get("replay") if isinstance(snapshot.get("replay"), Mapping) else {}
                turns = replay.get("turns")
                index_key = (qid, role, index)
                slot = index + 1 if role == "candidate" and index >= 0 else None
                units.append(Unit(qid, role, snapshot.get("route"), frozenset(str(entry) for entry in snapshot["fired"]), int(turns) if isinstance(turns, int) and turns > 0 else None, dict((fired_turns or {}).get(index_key, {})) or None, slot))
        return cls(name, tuple(units))

    @classmethod
    def from_run_dir(cls, run_dir: Any) -> "RootRun":
        """Collect units from a run directory.

        Args:
            run_dir (Path): Directory with result files.

        Returns:
            RootRun: The units, named after the directory.
        """
        from pathlib import Path

        run_dir = Path(run_dir)
        return cls.from_rows(run_dir.name, read_rows(run_dir))


@dataclass(frozen=True)
class CostEstimate:
    """What a re-evaluation of the release against the root run would have to regenerate.

    Attributes:
        moved (Mapping[str, tuple[str, str]]): Keyspaces that moved.
        changed_surfaces (tuple[str, ...]): Surfaces whose static text changed.
        exposed (Mapping[str, int]): Keyspace label to units that rendered a changed entry (they diverge).
        replayed (Mapping[str, int]): Keyspace label to units in a moved keyspace that never rendered a changed entry.
        suffix_turns (int): Turns to regenerate, summed over exposed units with a turn count (from the first changed
            render when the turn index is known, else the whole run as an upper bound).
        units_without_turns (int): Exposed units whose snapshot recorded no turn count.
        units_without_index (int): Exposed units estimated from the fired set alone (no turn index).
        judge_reruns (int): Questions whose judge must run afresh: an exposed judge unit, or any exposed candidate.
        exposed_qids (tuple[str, ...]): Questions with at least one exposed unit.
    """

    moved: Mapping[str, tuple[str, str]]
    changed_surfaces: tuple[str, ...]
    exposed: Mapping[str, int]
    replayed: Mapping[str, int]
    suffix_turns: int
    units_without_turns: int
    units_without_index: int
    judge_reruns: int
    exposed_qids: tuple[str, ...]

    def summary(self) -> str:
        """One line for the gate report.

        Returns:
            str: Moved keyspaces, exposed units and the turn estimate.
        """
        moved = ", ".join(sorted(self.moved)) or "none"
        exposed = ", ".join(f"{label} {count}" for label, count in sorted(self.exposed.items())) or "0"
        turns = f"≈ {self.suffix_turns} suffix turns" + (f" (+{self.units_without_turns} units without a turn count)" if self.units_without_turns else "")
        index_note = " estimated from fired sets only" if self.units_without_index else ""
        return f"keyspaces moved: {moved}; exposed units: {exposed}; {turns}{index_note}; judge re-runs on {self.judge_reruns} question(s)"


def estimate_cost(release: Any, parent: Any, root_run: RootRun) -> CostEstimate:
    """Estimate the divergence and generation cost of evaluating the release from a root run.

    A unit diverges when its keyspace moved and it rendered an entry providing a surface whose text
    changed; a unit in a moved keyspace that rendered no such entry replays turn by turn. Fired sets
    are read through the surface table (:func:`entries.fired_surfaces`), so runs and pinned
    releases meet on their declared surfaces, and the candidate's
    first user message, which the code's fallback renders without a record, counts as rendered with the
    candidate system message. Without a turn index the estimate uses the fired sets alone and charges
    each exposed unit's full turn count.

    Args:
        release (Release): The child.
        parent (Release): The parent (the release the root run ran under).
        root_run (RootRun): Fired sets of the root run.

    Returns:
        CostEstimate: The estimate.
    """
    changed = changed_surfaces(release, parent)
    moved = keyspace_moves(release, parent)
    exposed: dict[str, int] = {}
    replayed: dict[str, int] = {}
    suffix = 0
    without_turns = 0
    without_index = 0
    exposed_qids: set[str] = set()
    judge_qids: set[str] = set()
    for unit in root_run.units:
        label = _unit_label(unit, release, parent)
        if label not in moved:
            continue
        # a candidate renders its slot's own document when the child gives it one, whatever row it rendered before
        slot_surfaces = {f"candidate.s{unit.slot}.system", f"candidate.s{unit.slot}.user"} if unit.role == "candidate" and unit.slot is not None else set()
        hits = {entry for entry in unit.fired if E.fired_surfaces((entry,)) & changed}
        if slot_surfaces & changed:
            hits.add(E.slot_entry(unit.slot))
        if not hits:
            replayed[label] = replayed.get(label, 0) + 1
            continue
        exposed[label] = exposed.get(label, 0) + 1
        exposed_qids.add(unit.qid)
        if unit.role in ("candidate", "judge"):
            judge_qids.add(unit.qid)
        if unit.turns is None:
            without_turns += 1
            continue
        if unit.fired_turns and any(entry in unit.fired_turns for entry in hits):
            first = min(unit.fired_turns[entry] for entry in hits if entry in unit.fired_turns)
            suffix += max(unit.turns - first + 1, 0)
        else:
            without_index += 1
            suffix += unit.turns
    return CostEstimate(moved, tuple(sorted(changed)), exposed, replayed, suffix, without_turns, without_index, len(judge_qids), tuple(sorted(exposed_qids)))


def check_keyspace_cost(release: Any, parent: Any | None, root_run: RootRun | None) -> Check:
    """Keyspace movement and, with a root run, the cost estimate.

    Args:
        release (Release): The release under test.
        parent (Release | None): The parent; ``skip`` when absent.
        root_run (RootRun | None): The root run's fired sets; movement alone is reported when absent.

    Returns:
        Check: ``pass`` (informational).
    """
    if parent is None:
        return Check("keyspace_cost", "skip", "no parent release given")
    moved = keyspace_moves(release, parent)
    changed = sorted(changed_surfaces(release, parent))
    if root_run is None:
        detail = f"keyspaces moved: {', '.join(sorted(moved)) or 'none'}; changed surfaces: {', '.join(changed) or 'none'}; no root run given for a cost estimate"
        return Check("keyspace_cost", "pass", detail, {"moved": moved, "changed_surfaces": changed, "cost": None})
    estimate = estimate_cost(release, parent, root_run)
    data = {
        "moved": moved,
        "changed_surfaces": changed,
        "cost": {
            "root_run": root_run.name,
            "exposed": dict(estimate.exposed),
            "replayed": dict(estimate.replayed),
            "suffix_turns": estimate.suffix_turns,
            "units_without_turns": estimate.units_without_turns,
            "units_without_index": estimate.units_without_index,
            "judge_reruns": estimate.judge_reruns,
            "exposed_qids": list(estimate.exposed_qids),
        },
    }
    return Check("keyspace_cost", "pass", estimate.summary(), data)


# ---------------------------------------------------------------------------------------
# Pipeline


def run_gate(
    release: Any,
    *,
    parent: Any | None = None,
    store: Store | None = None,
    proposal: Mapping[str, Any] | None = None,
    hb_render: HBRender | None = None,
    corpus: Corpus | None = None,
    root_run: RootRun | None = None,
    matrix: Sequence[Cell] | None = None,
    cell_time_cap_ms: float = 50.0,
) -> GateReport:
    """Run every gate check in the design's order and collect the report.

    Args:
        release (Release): The release a proposal produced, already loaded and verified.
        parent (Release | None): The base parent release, for diffs, corpus and cost.
        store (Store | None): Store whose data cells join the code cells when ``matrix`` is omitted.
        proposal (Mapping[str, Any] | None): The proposal body (reason, hypothesis, plan).
        hb_render (HBRender | None): HeavenBase reference renderer; skipped when ``None``.
        corpus (Corpus | None): Locally built real-input corpus.
        root_run (RootRun | None): Fired sets of the root run.
        matrix (Sequence[Cell] | None): Cells; :func:`cells.all_cells` when omitted.
        cell_time_cap_ms (float): Per-cell interpreter thread CPU time cap; scheduler delays are excluded.

    Returns:
        GateReport: Every check; ``passed`` tells whether publication may proceed.
    """
    cells = tuple(matrix) if matrix is not None else all_cells(store)
    report = GateReport(release.id, parent.id if parent is not None else None)
    report.checks.append(check_names_meta_reason(release, parent, proposal, store))
    report.checks.append(check_text_policy(release))
    report.checks.append(check_spec_policy(release))
    report.checks.append(check_closure(release))
    report.checks.append(check_required_rows(release, parent))
    report.checks.append(check_rules_corpus(release, parent))
    run = run_matrix(release, cells, hb_render, cell_time_cap_ms=cell_time_cap_ms)
    report.checks.append(run.check)
    report.checks.append(check_coverage(release, cells))
    report.checks.append(check_contracts(release, run))
    parent_run = run_matrix(parent, cells, None, cell_time_cap_ms=float("inf")) if parent is not None else None
    report.checks.append(check_parent_cells(release, parent, run, parent_run))
    report.checks.append(check_change_ratio(release, parent))
    report.checks.append(check_corpus(corpus, parent, release))
    report.checks.append(check_keyspace_cost(release, parent, root_run))
    return report
