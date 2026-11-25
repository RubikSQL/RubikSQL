"""Find a repeated row inside an executed ``ORDER BY ... LIMIT N`` window (``candidate.note.result_shape``).

A top-N list that returns the same row twice inside its window usually lists one item
multiplied by a one-to-many join, or a category value, where the question asks for N
different items. Repeats are legitimate when they are different rows that share a displayed
value, so the finding only states the fact and fires under a narrow gate:

- the outer query is a single SELECT with ``LIMIT N`` (N > 1), no DISTINCT and no GROUP BY;
- two rows inside the first N are identical and carry a non-numeric value;
- the outer FROM joins tables, or the repeated row is low-cardinality: the distinct combinations of
  its non-numeric output columns number at most :data:`MAX_DISTINCT` and at most :data:`MAX_SHARE`
  of the FROM rows, measured by one probe query (:attr:`Repeat.probe`).

The module reads only the submitted SQL and its executed rows; it never sees the question.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .submission_gate import _COMPOUND, _outer_select, _select_items, _tokens, _Token

__all__ = ["IMPLEMENTATION_VERSION", "MAX_DISTINCT", "MAX_SHARE", "Repeat", "find", "fires", "low_cardinality"]

IMPLEMENTATION_VERSION = "limit-repeats-2"
MAX_DISTINCT = 200
MAX_SHARE = 0.05

_FROM_END = {"WHERE", "GROUP", "ORDER", "LIMIT", "HAVING", "WINDOW"}
_KEYWORDS = {"END", "NULL", "AND", "OR", "NOT", "IS", "ELSE", "THEN", "WHEN", "CASE", "DISTINCT", "AS", "TRUE", "FALSE"}


@dataclass(frozen=True)
class Repeat:
    """Two identical rows inside a LIMIT window.

    Attributes:
        limit (int): The outer LIMIT N.
        first (int): 1-based position of the first of the two rows.
        second (int): 1-based position of its repeat.
        column (str): The first non-numeric output column of the repeated row.
        join (bool): The outer FROM joins two or more tables, which is gate enough.
        probe (str | None): For a single-table FROM, the probe counting distinct combinations of every non-numeric
            output column against the FROM rows, which must pass :func:`low_cardinality`; ``None`` when the gate
            cannot be measured (an item that is ``*`` or holds a subquery).
    """

    limit: int
    first: int
    second: int
    column: str
    join: bool
    probe: str | None


def _limit(tokens: list[_Token], start: int, end: int, depth: int) -> int | None:
    at = next((i for i in range(start, end) if tokens[i].depth == depth and tokens[i].word == "LIMIT"), None)
    if at is None or at + 1 >= end or tokens[at + 1].kind != "number":
        return None
    count = tokens[at + 1]
    if at + 3 < end and tokens[at + 2].text == "," and tokens[at + 3].kind == "number":
        count = tokens[at + 3]     # LIMIT offset, count
    try:
        return int(count.text)
    except ValueError:
        return None


def _values(row: Any, columns: Sequence[str]) -> tuple[Any, ...]:
    if isinstance(row, Mapping):
        return tuple(row.get(column) for column in columns)
    return tuple(row)


def _numeric(value: Any) -> bool:
    return value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))


def _item_expression(sql: str, tokens: list[_Token], start: int, end: int) -> str | None:
    if end - start >= 2 and tokens[end - 2].word == "AS":
        end -= 2
    elif (end - start >= 2 and tokens[end - 1].kind in {"word", "identifier"}
          and (tokens[end - 2].text == ")" or tokens[end - 2].kind == "identifier"
               or (tokens[end - 2].kind == "word" and tokens[end - 2].word not in _KEYWORDS))
          and tokens[end - 1].word not in _KEYWORDS):
        end -= 1    # implicit alias: "expr alias"
    if end <= start or any(token.text == "*" for token in tokens[start:end]):
        return None
    return sql[tokens[start].start:tokens[end - 1].end]


def find(sql: str, columns: Sequence[str], rows: Sequence[Any]) -> Repeat | None:
    """Locate the first repeated row inside the outer LIMIT window of an executed final SQL.

    Args:
        sql (str): The SQL that produced ``rows``.
        columns (Sequence[str]): The result column names, in order.
        rows (Sequence[Any]): The executed rows (dicts keyed by column, or sequences).

    Returns:
        Repeat | None: The finding, or ``None`` when the shape does not qualify or no row repeats.
    """
    try:
        tokens = _tokens(sql)
    except ValueError:
        return None
    scope = _outer_select(tokens)
    if scope is None or scope.from_index is None or any(t.depth == 0 and t.word in _COMPOUND for t in tokens):
        return None
    if tokens[scope.start + 1].word == "DISTINCT":
        return None
    tail = range(scope.list_end, scope.end)
    if any(tokens[i].depth == scope.depth and tokens[i].word == "GROUP" for i in tail):
        return None
    limit = _limit(tokens, scope.list_end, scope.end, scope.depth)
    if limit is None or limit < 2:
        return None
    window = [_values(row, columns) for row in list(rows)[:limit]]
    seen: dict[tuple[Any, ...], int] = {}
    pair = None
    for position, values in enumerate(window):
        try:
            key = tuple(values)
            hash(key)
        except TypeError:
            return None
        if key in seen:
            pair = (seen[key], position)
            break
        seen[key] = position
    if pair is None:
        return None
    repeated = window[pair[0]]
    text_at = next((k for k, value in enumerate(repeated) if not _numeric(value)), None)
    if text_at is None:
        return None
    from_end = next((i for i in range(scope.from_index + 1, scope.end)
                     if tokens[i].depth == scope.depth and tokens[i].word in _FROM_END), scope.end)
    from_tokens = tokens[scope.from_index + 1:from_end]
    join = any(t.depth == scope.depth and (t.word == "JOIN" or t.text == ",") for t in from_tokens)
    probe = None
    items = _select_items(tokens, scope)
    text_columns = [k for k, value in enumerate(repeated) if not _numeric(value)]
    if not join and len(items) == len(repeated) and from_tokens:
        # The whole repeated row must be low-cardinality: every non-numeric output column together, so two
        # different people who share a low-cardinality label are never mistaken for one repeated item.
        # An item holding a subquery would make the probe run it per row, so such a query gets no probe.
        expressions = [_item_expression(sql, tokens, *items[k]) for k in text_columns]
        nested = any(token.word == "SELECT" for k in text_columns for token in tokens[items[k][0]:items[k][1]])
        if not nested and all(expressions):
            prefix = sql[:tokens[scope.start].start]
            source = sql[from_tokens[0].start:from_tokens[-1].end]
            probe = (f"{prefix}SELECT (SELECT COUNT(*) FROM (SELECT DISTINCT {', '.join(expressions)} FROM {source})) AS d, "
                     f"(SELECT COUNT(*) FROM {source}) AS n")
    column = str(columns[text_at]) if text_at < len(columns) else str(text_at + 1)
    return Repeat(limit, pair[0] + 1, pair[1] + 1, column, join, probe)


def low_cardinality(result: object) -> bool:
    """Tell whether a probe result shows a low-cardinality column.

    Args:
        result (object): The executed probe (``columns`` and one row of two counts).

    Returns:
        bool: ``True`` when distinct values are at most :data:`MAX_DISTINCT` and at most :data:`MAX_SHARE` of the rows.
    """
    if not isinstance(result, Mapping) or result.get("error") or not result.get("rows"):
        return False
    row = result["rows"][0]
    values = list(row.values()) if isinstance(row, Mapping) else list(row)
    try:
        distinct, total = int(values[0]), int(values[1])
    except (TypeError, ValueError, IndexError):
        return False
    return total > 0 and distinct <= MAX_DISTINCT and distinct <= MAX_SHARE * total


def fires(repeat: Repeat | None, probe_result: object = None) -> bool:
    """Apply the gate: a join, or a passing cardinality probe.

    Args:
        repeat (Repeat | None): The finding from :func:`find`.
        probe_result (object): The executed :attr:`Repeat.probe`, when one was run.

    Returns:
        bool: Whether the note fires.
    """
    return repeat is not None and (repeat.join or (repeat.probe is not None and low_cardinality(probe_result)))
