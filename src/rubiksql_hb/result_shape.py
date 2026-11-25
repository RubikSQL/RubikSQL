"""Read-only shape checks of a final SQL's outer SELECT, shared by voting and offline replay.

Two shapes are read here, both on an outer SELECT without GROUP BY that makes an aggregate call:

- The bare aggregate puts an aggregate call next to a column that is not aggregated. SQLite then
  returns a single row whose bare column holds the value of one arbitrary input row, so the
  answer is wrong under any reading of the question unless that column is constant over the
  filtered rows.
- The inert superlative adds ORDER BY and LIMIT to such a query. The aggregate collapses the
  rows to one before the ordering applies, so the ranking the SQL was written to perform never
  happens: the query aggregates over every filtered row instead of picking the extreme one. With
  aggregates only, its result is the plain aggregate's, which is right only when the question
  asks for that aggregate; the shape signals a ranking that was lost, not a wrong result under
  every reading.

The SQL is read with the submission gate's tokenizer, so comments, string literals and quoted
identifiers (for example a backticked column whose name holds parentheses) never count as calls
or keywords.
"""

from __future__ import annotations

from dataclasses import dataclass

from .submission_gate import _COMPOUND, _outer_select, _pairs, _select_items, _tokens, _Token

#: Aggregate calls. MIN and MAX are aggregates only with one argument; with more they are scalar.
AGGREGATES = frozenset({"COUNT", "SUM", "AVG", "TOTAL", "GROUP_CONCAT", "MIN", "MAX"})
#: SQLite's documented extreme-row idiom: when the query makes exactly one min() or max() call,
#: bare columns come from the row holding that extreme, whatever other aggregates sit beside it
#: (SQLite reuses one evaluation for repeated identical calls), so such a query is exempt.
EXTREMES = frozenset({"MIN", "MAX"})

# Words that never name a column inside a select item. A word missing here only costs the
# constant probe one more column; it cannot make a constant item fire.
_KEYWORDS = frozenset({
    "CASE", "WHEN", "THEN", "ELSE", "END", "AND", "OR", "NOT", "NULL", "IS", "IN", "LIKE", "GLOB",
    "REGEXP", "MATCH", "BETWEEN", "AS", "CAST", "DISTINCT", "ALL", "TRUE", "FALSE", "ESCAPE",
    "EXISTS", "COLLATE", "ISNULL", "NOTNULL", "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP",
})
# Clauses after which the probe stops: they reorder or cut rows but never change the row set.
_PROBE_STOP = frozenset({"ORDER", "LIMIT", "WINDOW"})


@dataclass(frozen=True)
class BareAggregate:
    """An outer SELECT that aggregates without GROUP BY next to non-aggregated columns.

    Attributes:
        aggregates (tuple[str, ...]): Aggregate calls of the select list in order, one entry
            per call (``MAX(x) - MIN(x)`` contributes two).
        bare_items (tuple[str, ...]): Source text of the non-aggregated column items.
        probe_sql (str): ``SELECT COUNT(*) FROM (SELECT DISTINCT <bare items> <FROM .. WHERE ..>)``
            over the same rows, without ORDER BY, LIMIT or WINDOW; a CTE prefix is kept.
    """

    aggregates: tuple[str, ...]
    bare_items: tuple[str, ...]
    probe_sql: str


def _subquery_spans(tokens: list[_Token], pairs: dict[int, int]) -> list[tuple[int, int]]:
    return [(opening, pairs[opening]) for opening, token in enumerate(tokens)
            if token.text == "(" and opening + 1 < len(tokens) and tokens[opening + 1].word in {"SELECT", "WITH"}]


def _aggregate_calls(tokens: list[_Token], indices: list[int], pairs: dict[int, int]) -> list[tuple[str, str]]:
    """Aggregate calls among ``indices`` as (name, normalized call text).

    MIN/MAX with two or more arguments are scalar functions. The call text drops whitespace and
    upper-cases words, so repeated identical calls compare equal.
    """
    calls = []
    for i in indices:
        word = tokens[i].word
        if word not in AGGREGATES or i + 1 >= len(tokens) or tokens[i + 1].text != "(":
            continue
        if i > 0 and tokens[i - 1].text == ".":
            continue
        opening = i + 1
        if word in EXTREMES and any(tokens[j].text == "," and tokens[j].depth == tokens[opening].depth + 1
                                    for j in range(opening + 1, pairs[opening])):
            continue
        text = "".join(token.text.upper() if token.kind == "word" else token.text
                       for token in tokens[i:pairs[opening] + 1])
        calls.append((word, text))
    return calls


def _operand(token: _Token) -> bool:
    return (token.kind in {"identifier", "number", "string"} or token.text == ")"
            or token.kind == "word" and (token.word not in _KEYWORDS or token.word in {"END", "NULL"}))


def _without_alias(tokens: list[_Token], start: int, end: int) -> int:
    """End of one select item without its alias (``AS name`` or a trailing bare name)."""
    last = end - 1
    if (end - start >= 2 and tokens[last].kind in {"identifier", "word"} and tokens[last].word not in _KEYWORDS
            and (tokens[last - 1].word == "AS" or _operand(tokens[last - 1])) and tokens[last - 1].text != "."):
        return last - 1 if tokens[last - 1].word == "AS" else last
    return end


def _references_column(tokens: list[_Token], indices: list[int]) -> bool:
    """Whether one alias-free select item reads a column, ignoring type and function names."""
    names = list(indices)
    for position, i in enumerate(names):
        token = tokens[i]
        previous = tokens[names[position - 1]] if position else None
        following = tokens[i + 1] if i + 1 < len(tokens) else None
        if token.text == "*" and (previous is None or previous.text == "."):
            return True
        if token.kind == "identifier" and not (previous is not None and previous.word in {"AS", "COLLATE"}):
            return True
        if (token.kind == "word" and token.word not in _KEYWORDS
                and not (following is not None and following.text == "(")
                and not (previous is not None and previous.word in {"AS", "COLLATE"})):
            return True
    return False


def bare_aggregate(sql: object) -> BareAggregate | None:
    """Read whether a SQL's outer SELECT is a bare aggregate.

    It is when the outer SELECT is not compound, has a FROM clause and no GROUP BY, makes at
    least one aggregate call in its select list, and holds an item that is neither aggregated
    nor a literal, a scalar subquery, a constant expression or a window. Aggregates are counted
    per call. A query whose MIN/MAX calls are all one identical call is exempt, whatever other
    aggregates it makes (SQLite's extreme-row idiom); two different MIN/MAX calls, as in
    ``MAX(x) - MIN(x)``, are not. The bare column's constancy is not decided here: the caller runs
    :attr:`BareAggregate.probe_sql` and treats a count of at most one as constant.

    Args:
        sql (object): The SQL text; anything else, or SQL that cannot be tokenized, reads as no defect.

    Returns:
        BareAggregate | None: The shape and its constant probe, or ``None``.
    """
    if not isinstance(sql, str):
        return None
    try:
        tokens = _tokens(sql)
    except ValueError:
        return None
    if not tokens or tokens[0].word not in {"SELECT", "WITH"}:
        return None
    if any(token.depth == 0 and token.word in _COMPOUND for token in tokens):
        return None
    scope = _outer_select(tokens)
    if scope is None or scope.from_index is None:
        return None
    own = range(scope.list_end, scope.end)
    if any(tokens[i].depth == scope.depth and tokens[i].word == "GROUP" for i in own):
        return None
    pairs = _pairs(tokens)
    spans = _subquery_spans(tokens, pairs)
    aggregates: list[tuple[str, str]] = []
    bare: list[tuple[int, int]] = []
    for start, end in _select_items(tokens, scope):
        # The alias is read over the whole item first: `(SELECT ...) best` ends in an alias too.
        indices = [i for i in range(start, _without_alias(tokens, start, end))
                   if not any(low <= i <= high for low, high in spans)]
        if any(tokens[i].word == "OVER" for i in indices):
            continue
        calls = _aggregate_calls(tokens, indices, pairs)
        if calls:
            aggregates.extend(calls)
        elif _references_column(tokens, indices):
            bare.append((start, end))
    extremes = {text for name, text in aggregates if name in EXTREMES}
    if not aggregates or not bare or len(extremes) == 1:
        return None
    stop = next((i for i in range(scope.from_index, scope.end)
                 if tokens[i].depth == scope.depth and tokens[i].word in _PROBE_STOP), scope.end)
    while stop > scope.from_index and tokens[stop - 1].text == ";":
        stop -= 1
    items = tuple(sql[tokens[start].start:tokens[end - 1].end] for start, end in bare)
    source = sql[tokens[scope.from_index].start:tokens[stop - 1].end]
    prefix = sql[:tokens[scope.start].start]
    probe = f"{prefix}SELECT COUNT(*) FROM (SELECT DISTINCT {', '.join(items)} {source})"
    return BareAggregate(tuple(name for name, _ in aggregates), items, probe)


def inert_superlative(sql: object) -> bool:
    """Read whether a SQL's outer SELECT is an aggregate whose ORDER BY and LIMIT rank nothing.

    It is when the SQL starts with SELECT (a CTE is not read), the outer SELECT is not compound,
    has a FROM clause, no GROUP BY and no window, makes at least one aggregate call in its select
    list outside any subquery, and carries both ORDER BY and LIMIT of its own. Such a query returns
    at most one row, so its ORDER BY and LIMIT change nothing. Bare columns beside the aggregate do
    not matter. A select list whose MIN/MAX calls are all one identical call is SQLite's
    extreme-row idiom and exempt, as in :func:`bare_aggregate`. Aggregates inside a scalar
    subquery are not the outer query's, so ``SELECT id, (SELECT MAX(x) FROM t) FROM u ORDER BY d
    LIMIT 1`` ranks its rows and is not inert.

    Args:
        sql (object): The SQL text; anything else, or SQL that cannot be tokenized, reads as no shape.

    Returns:
        bool: True for an inert superlative.
    """
    if not isinstance(sql, str):
        return False
    try:
        tokens = _tokens(sql)
    except ValueError:
        return False
    if not tokens or tokens[0].word != "SELECT":
        return False
    if any(token.depth == 0 and token.word in _COMPOUND for token in tokens):
        return False
    scope = _outer_select(tokens)
    if scope is None or scope.from_index is None:
        return False
    own = {tokens[i].word for i in range(scope.start, scope.end) if tokens[i].depth == scope.depth}
    if "GROUP" in own or "OVER" in own or "WINDOW" in own or not {"ORDER", "LIMIT"} <= own:
        return False
    pairs = _pairs(tokens)
    spans = _subquery_spans(tokens, pairs)
    aggregates: list[tuple[str, str]] = []
    for start, end in _select_items(tokens, scope):
        indices = [i for i in range(start, _without_alias(tokens, start, end))
                   if not any(low <= i <= high for low, high in spans)]
        aggregates.extend(_aggregate_calls(tokens, indices, pairs))
    extremes = {text for name, text in aggregates if name in EXTREMES}
    return bool(aggregates) and len(extremes) != 1
