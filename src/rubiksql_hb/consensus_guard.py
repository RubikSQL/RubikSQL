"""Deterministic post-review guard that restores a candidate consensus the final left by adding structure.

The judge sees candidates grouped by identical execution results. When it leaves the unique
largest group of at least two members, holding a non-empty answer, for an answer that reads an
extra table or adds a de-duplication step, this guard submits the group's first member instead. It makes no model call and reuses the candidate
results already executed in the vote. Its answerless fallback then replaces a final that carries
no answer (no rows, only NULL values, or a single numeric zero) with the largest agreeing group of
at least two answered candidates, and its defect fallback replaces a final whose outer SELECT is a
non-constant bare aggregate or an aggregate-only inert superlative (:mod:`.result_shape`) with
the largest answered group whose representative is neither, unless an unflagged candidate returns
the inert superlative's own result. Voting and offline replay share this module.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any

from .question import HINT_SEPARATOR
from .result_shape import bare_aggregate, inert_superlative
from .submission_gate import SQL_READER_VERSION as _READER_VERSION
from .submission_gate import (
    _bindings, _Catalog, _pairs, _scalar, _selects, _successful, _Table, _tokens, _Token, _updated, _values,
)


#: The guard reads SQL with the submission gate's parser, so the gate's version is part of its own.
IMPLEMENTATION_VERSION = f"consensus-guard-4+{_READER_VERSION}"
#: The opt-in guard that bypasses the hint-column exemption for added tables.
STRICT_ADDED_TABLES_VERSION = f"consensus-guard-5+{_READER_VERSION}"
FEATURE_NAMES = ("new_source", "added_dedup")
#: The answerless fallback that runs after the guard; recorded separately so the guard's version is unchanged.
ANSWERLESS_FALLBACK_VERSION = "answerless-fallback-1"
#: Members a group of answered candidates needs before it can replace an answerless final.
ANSWERLESS_FALLBACK_MIN_GROUP = 2
#: The defect fallback that runs after the answerless fallback; recorded separately for the same reason.
#: Version 2 also flags inert superlatives (version 1 read bare aggregates only) and records the final's ``shape``.
DEFECT_FALLBACK_VERSION = "defect-fallback-2"
#: The defect shapes a record's ``shape`` names.
DEFECT_SHAPES = ("inert-superlative", "bare-aggregate")

_DEDUP_CUE = re.compile(r"\b(?:distinct|unique|different|repetitive|duplicat\w*)\b", re.I)
_SEMI_JOINS = {"IN", "EXISTS"}


@dataclass(frozen=True)
class _Shape:
    tables: frozenset[_Table]
    dedup: bool


def result_groups(
    candidates: Sequence[dict[str, Any]], result_sig: Callable[[dict[str, Any]], Hashable],
) -> list[list[int]]:
    """Group non-failed candidates by execution result, largest group first.

    This is the grouping the judge reads as ``group_label``/``group_size``: label ``A`` is the
    first returned group, ``B`` the second, and so on.

    Args:
        candidates (Sequence[dict[str, Any]]): Executed candidate pool in slot order. A candidate
            with a truthy ``error`` belongs to no group.
        result_sig (Callable[[dict[str, Any]], Hashable]): Official execution-result signature.

    Returns:
        list[list[int]]: 1-based slots per group, ordered by descending size and then by the
            group's lowest slot. Slots within a group ascend.
    """
    by_signature: dict[Hashable, list[int]] = {}
    for slot, candidate in enumerate(candidates, 1):
        if not candidate.get("error"):
            by_signature.setdefault(result_sig(candidate), []).append(slot)
    return sorted(by_signature.values(), key=lambda group: (-len(group), group[0]))


def plurality_slot(
    candidates: Sequence[dict[str, Any]], result_sig: Callable[[dict[str, Any]], Hashable],
) -> int | None:
    """The candidate a judge-less answer falls back to: the plurality result's first slot.

    The largest result group wins. Between equal groups an answered one (not `answerless`)
    beats an answerless one, and then the one holding the lowest slot wins, so a pool without
    a plurality keeps the old positional choice; a larger group is never displaced. Within the
    group the lowest slot is taken (all of its members return the same result).

    Args:
        candidates (Sequence[dict[str, Any]]): Executed candidate pool in slot order; failed
            candidates (truthy ``error``) are never chosen.
        result_sig (Callable[[dict[str, Any]], Hashable]): Official execution-result signature.

    Returns:
        int | None: The chosen 1-based slot, or ``None`` when every candidate failed.
    """
    groups = result_groups(candidates, result_sig)
    if not groups:
        return None
    tied = [group for group in groups if len(group) == len(groups[0])]
    answered = [group for group in tied if not answerless(candidates[group[0] - 1])]
    return (answered or tied)[0][0]


def fallback_candidate_sql(candidates: list | None, debug: list | None = None) -> str:
    """Pick the native candidate net's SQL when every judge pass is answerless.

    The plurality of recorded results wins. Without complete recorded results,
    choose the first non-errored SQL, or the first SQL if every seat errored.
    This inert policy is shared by live evaluation and frozen-pool reproduction.

    Args:
        candidates (list | None): Ordered candidate summaries with SQL and error.
        debug (list | None): Aligned producer records, optionally carrying rows.

    Returns:
        str: Selected SQL, or an empty string if no seat produced SQL text.
    """
    from .metrics.official_ex import result_signature

    candidates = candidates or []
    executed = []
    for index, candidate in enumerate(candidates):
        record = debug[index] if debug and index < len(debug) else None
        usable = (isinstance(candidate, dict) and isinstance(record, dict) and not candidate.get("error")
                  and not record.get("error") and "rows" in record and (candidate.get("sql") or "").strip())
        executed.append({"sql": candidate["sql"], "rows": record["rows"]} if usable else {"error": "no recorded result"})
    slot = plurality_slot(executed, result_signature)
    if slot is not None:
        return executed[slot - 1]["sql"].strip()
    pool = [c for c in candidates
            if isinstance(c, dict) and (c.get("sql") or "").strip()]
    pick = next((c for c in pool if not c.get("error")), pool[0] if pool else None)
    return (pick.get("sql") or "").strip() if pick else ""


def _key(table: _Table) -> _Table:
    return _Table(table.name.casefold(), table.schema.casefold())


def _semi_join_select(tokens: list[_Token], index: int) -> bool:
    """``IN (SELECT`` or ``EXISTS (SELECT`` at ``index``, excluding the ``NOT`` anti-join forms."""
    return (tokens[index].word in _SEMI_JOINS
            and (index == 0 or tokens[index - 1].word != "NOT")
            and index + 2 < len(tokens) and tokens[index + 1].text == "("
            and tokens[index + 2].word == "SELECT")


def _shape(sql: Any) -> _Shape | None:
    """Read the tables and de-duplication steps of one SQL statement.

    Comments, string literals and quoted identifiers never count as keywords. CTE names and
    derived tables are not tables. ``None`` means the SQL is missing, cannot be tokenized or
    has a FROM clause that cannot be read, so no structural comparison involving it is trusted.
    """
    if not isinstance(sql, str):
        return None
    try:
        tokens = _tokens(sql)
    except ValueError:
        return None
    pairs = _pairs(tokens)
    scopes = _selects(tokens)
    tables: set[_Table] = set()
    for scope in scopes:
        bindings = _bindings(tokens, scope, pairs)
        if scope.from_index is not None and not bindings:
            return None
        tables.update(_key(binding.table) for binding in bindings if binding.table is not None)
    dedup = any(
        (token.word == "COUNT" and i + 2 < len(tokens) and tokens[i + 1].text == "("
         and tokens[i + 2].word == "DISTINCT")
        or (token.word == "IN" and _semi_join_select(tokens, i) and i + 3 < len(tokens)
            and tokens[i + 3].word == "DISTINCT")
        for i, token in enumerate(tokens))
    for scope in scopes if not dedup else ():
        own = [i for i in range(scope.start, scope.end) if tokens[i].depth == scope.depth]
        if (any(tokens[i].word == "COUNT" and i + 1 < len(tokens) and tokens[i + 1].text == "(" for i in own)
                and any(_semi_join_select(tokens, i) for i in own)):
            dedup = True
            break
    return _Shape(frozenset(tables), dedup)


def _answerless(result: dict[str, Any]) -> bool:
    """No rows, or only NULL values: a consensus that carries no answer to restore."""
    rows = result.get("rows") or []
    return all(value is None for row in rows for value in _values(row))


def answerless(result: dict[str, Any]) -> bool:
    """Whether a successful result carries no answer: no rows, only NULL values, or one numeric zero.

    A single row holding a single ``0`` or ``0.0`` (not a boolean) counts as answerless, because a
    filter that matches nothing often surfaces as a zero count or sum.

    Args:
        result (dict[str, Any]): Executed result with ``rows``.

    Returns:
        bool: True for an answerless result.
    """
    rows = result.get("rows") or []
    values = [value for row in rows for value in _values(row)]
    if all(value is None for value in values):
        return True
    if len(rows) == 1 and len(values) == 1:
        value = values[0]
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0
    return False


def apply_answerless_fallback(
    chosen: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    result_sig: Callable[[dict[str, Any]], Hashable],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace an answerless final with the largest agreeing group of answered candidates.

    The final is left unchanged when it failed or is not :func:`answerless`. Otherwise the result
    groups (:func:`result_groups`) that are answered and hold at least
    :data:`ANSWERLESS_FALLBACK_MIN_GROUP` candidates are kept, so a single stray candidate never
    wins; the lowest slot of the largest kept group (ties go to the lower slot) is submitted with
    its recorded result. Without such a group the final stays. It makes no model or database call.

    Args:
        chosen (dict[str, Any]): Current executed final with SQL and full rows.
        candidates (Sequence[dict[str, Any]]): Executed candidate pool in slot order.
        result_sig (Callable[[dict[str, Any]], Hashable]): Official execution-result signature.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: The final (a shallow copy whose rows and columns
            are detached from the candidate when it fired, else ``chosen`` itself) and a
            version-1 record: ``v``, ``implementation_version``, ``pre_fallback_sql``, ``fired``,
            ``from_slot`` (the lowest candidate slot sharing the final's result, else ``None``),
            ``to_slot`` and ``group_size`` (the submitted slot and its group's size when fired,
            else ``None`` and ``0``) and ``skipped`` (``final-failed``, ``answered`` or
            ``no-supported-answered-group`` when it did not fire, else ``None``).
    """
    debug: dict[str, Any] = {"v": 1, "implementation_version": ANSWERLESS_FALLBACK_VERSION,
                             "pre_fallback_sql": chosen.get("sql"), "fired": False, "from_slot": None,
                             "to_slot": None, "group_size": 0, "skipped": None}
    if not isinstance(chosen.get("sql"), str) or not _successful(chosen):
        debug["skipped"] = "final-failed"
        return chosen, debug
    if not answerless(chosen):
        debug["skipped"] = "answered"
        return chosen, debug
    groups = result_groups(candidates, result_sig)
    signature = result_sig(chosen)
    debug["from_slot"] = next((group[0] for group in groups
                               if result_sig(candidates[group[0] - 1]) == signature), None)
    answered = [group for group in groups
                if len(group) >= ANSWERLESS_FALLBACK_MIN_GROUP and not answerless(candidates[group[0] - 1])]
    if not answered:
        debug["skipped"] = "no-supported-answered-group"
        return chosen, debug
    group = answered[0]
    member = candidates[group[0] - 1]
    debug.update(fired=True, to_slot=group[0], group_size=len(group))
    return _updated(chosen, member, member["sql"]), debug


def defect_shape(sql: Any) -> str | None:
    """The defect shape a SQL may have, before any probe: one of :data:`DEFECT_SHAPES`, else ``None``.

    A bare aggregate (:func:`.result_shape.bare_aggregate`) is defective only when its constant
    probe counts more than one value (:func:`defect`); this holds with or without ORDER BY and
    LIMIT, so a bare column constant over the filtered rows keeps its exemption. An inert
    superlative (:func:`.result_shape.inert_superlative`) with no bare column is defective by its
    shape alone.

    Args:
        sql (Any): The SQL to read.

    Returns:
        str | None: ``"inert-superlative"``, ``"bare-aggregate"`` or ``None``.
    """
    if bare_aggregate(sql) is not None:
        return "bare-aggregate"
    return "inert-superlative" if inert_superlative(sql) else None


async def defect(sql: Any, execute: Callable[[str], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
    """Decide whether one SQL is a non-constant bare aggregate or an inert superlative.

    The shape comes from :func:`defect_shape`. A bare aggregate is defective only when its
    constant probe succeeds and counts more than one distinct value of the bare columns; an
    inert superlative (aggregates only, no bare column) is defective without a probe.

    Args:
        sql (Any): The SQL to read.
        execute (Callable[[str], Awaitable[dict[str, Any]]]): Read-only base-database executor
            for the probe; it runs only for a bare aggregate.

    Returns:
        dict[str, Any]: ``flagged`` (bool), ``shaped`` (bool), ``shape`` (the
            :func:`defect_shape`), ``distinct`` (the probe's count, else ``None``) and
            ``probe_error`` (True when a bare aggregate's probe failed twice, which leaves
            ``flagged`` false).
    """
    shape = bare_aggregate(sql)
    if shape is None:
        inert = inert_superlative(sql)
        return {"flagged": inert, "shaped": inert, "shape": "inert-superlative" if inert else None,
                "distinct": None, "probe_error": False}
    distinct = None
    for _ in range(2):   # one retry: a probe can fail transiently under server load
        probe = await execute(shape.probe_sql)
        distinct = _scalar(probe) if _successful(probe) else None
        if isinstance(distinct, int):
            break
    if not isinstance(distinct, int):
        return {"flagged": False, "shaped": True, "shape": "bare-aggregate", "distinct": None, "probe_error": True}
    return {"flagged": distinct > 1, "shaped": True, "shape": "bare-aggregate", "distinct": distinct,
            "probe_error": False}


async def apply_defect_fallback(
    chosen: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    result_sig: Callable[[dict[str, Any]], Hashable],
    execute: Callable[[str], Awaitable[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace a defective final with an answered, unflagged group.

    A final is defective (:func:`defect`) when it aggregates without GROUP BY next to a column
    that is not constant over its rows, and so returns one arbitrary row's value (wrong under any
    reading of the question), or when it is an aggregate-only inert superlative, whose ORDER BY
    and LIMIT rank a single aggregated row. The inert superlative returns the plain aggregate's
    result, which is right only when the question asks for that aggregate rather than for a
    ranking, so its result stays whenever an unflagged answered candidate returns it too
    (``result-supported``). Otherwise a defective final gives way to the result group
    (:func:`result_groups`; largest first, ties to the lower slot, one member suffices) that is
    answered (not :func:`answerless`) and whose representative is not flagged. A group's
    representative is its lowest slot whose SQL is not an inert superlative by shape alone, so a
    group of only such SQLs is flagged without a probe and the representative is the member
    submitted. A probe that fails is retried once; when a representative's probe still fails,
    its group is unchecked and the final stays, so a smaller group is never chosen over an
    unchecked larger one. Without such a group, and for any final that failed, is not shaped, is
    constant or cannot be probed, the final stays. Only probes run; no model is called.

    Args:
        chosen (dict[str, Any]): Current executed final with SQL and full rows.
        candidates (Sequence[dict[str, Any]]): Executed candidate pool in slot order.
        result_sig (Callable[[dict[str, Any]], Hashable]): Official execution-result signature.
        execute (Callable[[str], Awaitable[dict[str, Any]]]): Read-only base-database executor
            for the constant probes.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: The final (a shallow copy whose rows and columns
            are detached from the candidate when it fired, else ``chosen`` itself) and a
            ``v`` 1 record: ``v``, ``implementation_version``, ``pre_fallback_sql``, ``fired``,
            ``shape`` (the final's :func:`defect_shape`), ``distinct`` (the final's probe count,
            else ``None``), ``from_slot`` (the lowest
            candidate slot sharing the final's result, else ``None``), ``to_slot`` and
            ``group_size`` (when fired, else ``None`` and ``0``), ``flagged_slots`` (checked
            representatives that were flagged; the lowest slot for a group of only inert
            superlatives) and ``skipped`` (``final-failed``, ``no-defect``, ``probe-failed``,
            ``constant``, ``result-supported``, ``representative-probe-failed`` or
            ``no-unflagged-answered-group`` when it did not fire, else ``None``).
    """
    debug: dict[str, Any] = {"v": 1, "implementation_version": DEFECT_FALLBACK_VERSION,
                             "pre_fallback_sql": chosen.get("sql"), "fired": False, "shape": None,
                             "distinct": None, "from_slot": None, "to_slot": None, "group_size": 0,
                             "flagged_slots": [], "skipped": None}
    if not isinstance(chosen.get("sql"), str) or not _successful(chosen):
        debug["skipped"] = "final-failed"
        return chosen, debug
    final = await defect(chosen["sql"], execute)
    debug.update(shape=final["shape"], distinct=final["distinct"])
    if not final["flagged"]:
        debug["skipped"] = ("no-defect" if not final["shaped"] else "probe-failed" if final["probe_error"]
                            else "constant")
        return chosen, debug
    groups = result_groups(candidates, result_sig)
    signature = result_sig(chosen)
    debug["from_slot"] = next((group[0] for group in groups
                               if result_sig(candidates[group[0] - 1]) == signature), None)
    verdicts: dict[int, tuple[int, dict[str, Any]]] = {}

    async def checked(index: int) -> tuple[int, dict[str, Any]]:
        # One check per group: its representative is the lowest slot not inert by shape alone.
        if index not in verdicts:
            group = groups[index]
            slot = next((slot for slot in group
                         if defect_shape(candidates[slot - 1].get("sql")) != "inert-superlative"), None)
            verdicts[index] = ((group[0], {"flagged": True, "probe_error": False}) if slot is None
                               else (slot, await defect(candidates[slot - 1].get("sql"), execute)))
        return verdicts[index]

    order = [index for index, group in enumerate(groups) if not answerless(candidates[group[0] - 1])]
    if final["shape"] == "inert-superlative":
        own = next((index for index in order if result_sig(candidates[groups[index][0] - 1]) == signature), None)
        if own is not None:
            slot, verdict = await checked(own)
            if verdict["probe_error"]:
                debug["skipped"] = "representative-probe-failed"
                return chosen, debug
            if not verdict["flagged"]:
                debug["skipped"] = "result-supported"
                return chosen, debug
    for index in order:
        slot, verdict = await checked(index)
        if verdict["probe_error"]:
            debug["skipped"] = "representative-probe-failed"
            return chosen, debug
        if verdict["flagged"]:
            if slot not in debug["flagged_slots"]:
                debug["flagged_slots"].append(slot)
            continue
        member = candidates[slot - 1]
        debug.update(fired=True, to_slot=slot, group_size=len(groups[index]))
        return _updated(chosen, member, member["sql"]), debug
    debug["skipped"] = "no-unflagged-answered-group"
    return chosen, debug


def _names_column(hint: str, column: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])", hint, re.I) is not None


async def _columns(catalog: _Catalog, table: _Table) -> tuple[str, ...]:
    columns = await catalog.table_columns(table)
    if not columns:
        # Every table here was read by SQL that executed, so missing columns mean the metadata read failed.
        raise LookupError(f"no column metadata for {table.schema}.{table.name}")
    return columns


async def _hint_tables(
    hint: str, added: frozenset[_Table], group_tables: frozenset[_Table], catalog: _Catalog,
) -> list[_Table]:
    """Added tables holding a column the hint names that no group table holds.

    Raises:
        LookupError: A table's column metadata could not be read.
    """
    if not hint.strip():
        return []
    shared = {column.casefold() for table in group_tables for column in await _columns(catalog, table)}
    exempt = []
    for table in sorted(added, key=lambda item: (item.schema, item.name)):
        columns = await _columns(catalog, table)
        if any(column.casefold() not in shared and _names_column(hint, column) for column in columns):
            exempt.append(table)
    return exempt


async def apply_consensus_guard(
    question: str,
    chosen: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    execute: Callable[[str], Awaitable[dict[str, Any]]],
    result_sig: Callable[[dict[str, Any]], Hashable],
    *,
    strict_added_tables: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restore the candidate consensus when the final departs from it by adding structure.

    G is the unique largest result group of at least two non-failed candidates
    (:func:`result_groups`; a tie for the largest group, or a G whose result has no rows or only
    NULL values, leaves the final unchanged). When the final's result differs from G's and the
    reviewer did not change the answer, the final SQL is compared with every SQL in G:

    - ``new_source``: it reads a table no G member reads, unless that table holds a column,
      absent from every table G reads, that the question's hint names. A failed metadata read
      abstains from ``new_source``. A final supported by at least two successful
      executed candidates is also exempt. ``strict_added_tables=True`` bypasses only
      the hint-column exemption.
    - ``added_dedup``: it uses ``COUNT(DISTINCT ...)``, ``IN (SELECT DISTINCT ...)`` or a COUNT
      in the same SELECT as an ``IN``/``EXISTS`` sub-select, and no G member does, unless the
      question or hint uses de-duplication wording (distinct, unique, different, repetitive,
      duplicate).

    Any firing feature replaces the final with G's lowest-slot member and its recorded result.

    Args:
        question (str): Question text, optionally followed by ``\\n[Hint]: `` and the hint.
        chosen (dict[str, Any]): Current executed final with SQL, full rows and optional
            ``reviewer_debug``.
        candidates (Sequence[dict[str, Any]]): Executed candidate pool in slot order.
        execute (Callable[[str], Awaitable[dict[str, Any]]]): Read-only base-database executor.
            Only table-column metadata for the hint exemption is read, and only when an added
            table and a hint exist and ``strict_added_tables`` is false.
        result_sig (Callable[[dict[str, Any]], Hashable]): Official execution-result signature.
        strict_added_tables (bool): If true, added tables are not exempted by hint columns.
            Defaults to false, preserving the hint exemption and its metadata-failure behavior.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: The final (a shallow copy whose rows and columns
            are detached from the candidate when it fired, else ``chosen`` itself) and a
            version-1 record: ``v``, ``implementation_version``, ``pre_guard_sql``, ``fired``,
            ``features`` (fired names in :data:`FEATURE_NAMES` order), ``group_label`` and
            ``group_size`` (``None`` and ``0`` without a consensus group), ``slot`` (the 1-based
            submitted candidate when fired, else ``None``) and ``skipped`` (why the comparison
            did not run, else ``None``). ``new_tables`` lists the non-exempt added tables when
            ``new_source`` fires. This function does not attach the record.
    """
    version = STRICT_ADDED_TABLES_VERSION if strict_added_tables else IMPLEMENTATION_VERSION
    debug: dict[str, Any] = {"v": 1, "implementation_version": version,
                             "pre_guard_sql": chosen.get("sql"), "fired": False, "features": [],
                             "group_label": None, "group_size": 0, "slot": None, "skipped": None}
    groups = result_groups(candidates, result_sig)
    if not groups or len(groups[0]) < 2 or (len(groups) > 1 and len(groups[1]) == len(groups[0])):
        debug["skipped"] = "no-consensus"
        return chosen, debug
    group = groups[0]
    debug.update(group_label="A", group_size=len(group))
    if _answerless(candidates[group[0] - 1]):
        debug["skipped"] = "empty-consensus"
        return chosen, debug
    reviewer = chosen.get("reviewer_debug")
    if isinstance(reviewer, dict) and reviewer.get("changed"):
        debug["skipped"] = "reviewer-changed"
        return chosen, debug
    sql = chosen.get("sql")
    if not isinstance(sql, str) or not _successful(chosen):
        debug["skipped"] = "final-failed"
        return chosen, debug
    if result_sig(chosen) == result_sig(candidates[group[0] - 1]):
        debug["skipped"] = "agrees"
        return chosen, debug
    final_shape = _shape(sql)
    group_shapes = [_shape(candidates[slot - 1].get("sql")) for slot in group]
    if final_shape is None or any(shape is None for shape in group_shapes):
        debug["skipped"] = "unparsed"
        return chosen, debug
    group_tables = frozenset().union(*(shape.tables for shape in group_shapes))
    features = []
    added = final_shape.tables - group_tables
    supporters = [slot for slot, candidate in enumerate(candidates, 1)
                  if (candidate.get("sql") or "").strip() and _successful(candidate)
                  and result_sig(candidate) == result_sig(chosen)]
    if added and len(supporters) >= 2:
        debug["new_source_exempt_slots"] = supporters
    if added and len(supporters) < 2:
        exempt = []
        if not strict_added_tables:
            try:
                exempt = await _hint_tables(question.partition(HINT_SEPARATOR)[2], added, group_tables,
                                            _Catalog(execute))
            except Exception as exc:
                # An unknown exemption abstains from new_source: the feature cannot be established.
                debug["errors"] = [{"feature": "new_source", "type": type(exc).__name__}]
                exempt = list(added)
        new_tables = sorted(f"{table.schema}.{table.name}" for table in added - set(exempt))
        if new_tables:
            features.append("new_source")
            debug["new_tables"] = new_tables
    if (final_shape.dedup and not any(shape.dedup for shape in group_shapes)
            and not _DEDUP_CUE.search(question)):
        features.append("added_dedup")
    if not features:
        debug["skipped"] = "no-feature"
        return chosen, debug
    member = candidates[group[0] - 1]
    debug.update(fired=True, features=features, slot=group[0])
    return _updated(chosen, member, member["sql"]), debug
