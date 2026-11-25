"""Diagnose uncued joined-row counts while preserving the selected final result."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Hashable, Sequence
from typing import Any

from .submission_gate import SQL_READER_VERSION as _READER_VERSION
from .submission_gate import (
    _identifier, _NOT_ALIAS, _outer_select, _pairs, _replace, _selects, _successful, _tokens,
)


#: SQL reader changes also change this hook's implementation identity.
IMPLEMENTATION_VERSION = f"count-weight-2+{_READER_VERSION}"

_DEDUP_CUE = re.compile(r"\b(?:distinct|unique|different|repetitive|duplicat\w*|repeated)\b", re.I)


def _variant(sql: str) -> tuple[str | None, str | None]:
    """Remove DISTINCT tokens only from aggregate counts in the outer joined projection."""
    try:
        tokens = _tokens(sql)
    except ValueError:
        return None, "unparsed"
    outer = _outer_select(tokens)
    if outer is None:
        return None, "unparsed"
    pairs = _pairs(tokens)
    nested = set()
    for scope in _selects(tokens):
        if not outer.start < scope.start < outer.end or scope.depth <= outer.depth:
            continue
        # Exclude the whole query parentheses, including WITH prefixes and compound arms.
        opening = next(i for i, end in pairs.items()
                       if i < scope.start < end and tokens[i].depth == scope.depth - 1)
        nested.update(range(opening, pairs[opening] + 1))
    own = [i for i in range(outer.start + 1, outer.end) if i not in nested]
    # WINDOW is also a legal unquoted alias/CTE name. The gate's generic list_end can
    # stop there, so find the reserved FROM keyword rather than treating a name as a clause.
    from_index = next((i for i in own if tokens[i].depth == outer.depth and tokens[i].word == "FROM"), None)
    if from_index is None:
        return None, "no_join"
    clauses = []
    windowed = False
    for i in own:
        token = tokens[i]
        if token.depth == outer.depth:
            if (token.word in {"GROUP", "ORDER"} and i + 1 < outer.end and tokens[i + 1].word == "BY"
                    or token.word == "WINDOW" and i + 3 < outer.end and _identifier(tokens[i + 1])
                    and tokens[i + 2].word == "AS" and tokens[i + 3].text == "("
                    or token.word in {"WHERE", "HAVING", "LIMIT", "RETURNING"}):
                clauses.append(i)
                windowed |= token.word in {"GROUP", "HAVING", "WINDOW"}
        # OVER can sit inside casts/arithmetic. Aliases, column names and CTE sources
        # named over are not window invocations.
        if (token.word == "OVER" and i > 0 and tokens[i - 1].text == ")" and i + 1 < outer.end
                and (tokens[i + 1].text == "(" or _identifier(tokens[i + 1])
                     and tokens[i + 1].word not in _NOT_ALIAS)):
            windowed = True
    if windowed:
        return None, "grouped_or_windowed"
    from_end = next((i for i in clauses if i > from_index), outer.end)
    if not any(from_index < i < from_end and tokens[i].word == "JOIN" for i in own):
        return None, "no_join"
    replacements = []
    for i in own:
        if (i + 2 < from_index and tokens[i].word == "COUNT" and tokens[i + 1].text == "("
                and tokens[i + 2].word == "DISTINCT" and pairs[i + 1] < from_index):
            distinct = tokens[i + 2]
            replacements.append((distinct.start, distinct.end, ""))
    return (_replace(sql, replacements), None) if replacements else (None, "no_distinct_count")


async def apply_count_weight(
    question: str,
    chosen: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    execute: Callable[[str], Awaitable[dict[str, Any]]],
    result_sig: Callable[[dict[str, Any]], Hashable],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check joined-row counts without replacing the selected final SQL or result.

    The final and every candidate must have successful, complete results. De-duplication
    wording in the question or hint exempts the answer. An outer SELECT with its own JOIN
    and no GROUP BY, HAVING or windows is eligible: remove DISTINCT from all its projection
    COUNT calls, leaving nested queries and the rest of the SQL intact. The variant must
    execute completely. Equality with a candidate is recorded as a veto; a different
    result also keeps the final, because absence from the pool is not evidence that
    removing de-duplication is correct. Equal results retain the original SQL too.

    Args:
        question (str): Question text, including any appended hint.
        chosen (dict[str, Any]): Executed final with SQL and full result rows.
        candidates (Sequence[dict[str, Any]]): Executed candidate pool with full result rows.
        execute (Callable[[str], Awaitable[dict[str, Any]]]): Read-only base-database executor.
        result_sig (Callable[[dict[str, Any]], Hashable]): Official execution-result signature.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: ``chosen`` itself and an unattached
            version-1 debug record.
            The record holds ``v``, ``implementation_version``, ``pre_hook_sql``, ``fired``,
            ``skipped``, ``variant_sql`` when built and ``errors`` on
            exceptions. Failed or incomplete executions use the corresponding ``*_failed``
            skip reason. Any exception keeps the original final.
    """
    debug: dict[str, Any] = {"v": 1, "implementation_version": IMPLEMENTATION_VERSION,
                             "pre_hook_sql": chosen.get("sql"), "fired": False, "skipped": None}
    try:
        sql = chosen.get("sql")
        if not isinstance(sql, str) or not _successful(chosen):
            debug["skipped"] = "final_failed"
        elif any(not _successful(candidate) for candidate in candidates):
            debug["skipped"] = "candidate_failed"
        elif _DEDUP_CUE.search(question):
            debug["skipped"] = "dedup_cue"
        else:
            variant_sql, debug["skipped"] = _variant(sql)
            if variant_sql is not None:
                debug["variant_sql"] = variant_sql
                variant = await execute(variant_sql)
                if not _successful(variant):
                    debug["skipped"] = "variant_failed"
                else:
                    signature = result_sig(variant)
                    if signature == result_sig(chosen):
                        debug["skipped"] = "variant_matches_final"
                    elif any(signature == result_sig(candidate) for candidate in candidates):
                        debug["skipped"] = "variant_matches_candidate"
                    else:
                        debug["skipped"] = "variant_changes_final"
    except Exception as exc:
        debug["skipped"] = "error"
        debug["errors"] = [{"feature": "count_weight", "type": type(exc).__name__}]
    return chosen, debug
