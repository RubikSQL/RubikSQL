"""Narrow answer-selection conventions shared by voting and saved-result replay."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass
from typing import Any

from .submission_gate import SQL_READER_VERSION as _READER_VERSION
from .submission_gate import (
    _COMPOUND, _NOT_ALIAS, _Select, _Token, _identifier, _pairs, _reference, _replace,
    _row_count, _selects, _successful, _tokens, _updated, _values,
)


#: The source reader's version is part of the hook's implementation identity.
IMPLEMENTATION_VERSION = f"selection-normalization-1+{_READER_VERSION}"
OPERATION_NAMES = ("bare_selection_cast", "uncued_single_row_selection")

# Frozen cast_limit/scan.json:cue_spec, including its reviewed additions.
_CUE_WORDS = """
highest lowest most least biggest smallest largest shortest longest best worst
top bottom peak maximum minimum max min greatest fewest fastest slowest
richest poorest tallest oldest youngest latest earliest recent recently newest
last first current previous prior next final initial rank ranks ranked ranking
rankings greater less more fewer higher lower bigger smaller larger shorter
longer better worse faster slower earlier later older younger above below
leading trailing second third fourth fifth sixth seventh eighth ninth tenth
eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth
eighteenth nineteenth twentieth penultimate
favorite favourite mostly champion majority deepest hottest widest main
sort sorted sorting order ordered ordering ascending descending ascend descend
""".split()
_CUE = re.compile(r"\b(?:" + "|".join(_CUE_WORDS) + r")\b", re.I)
_COUNT_WORDS = (
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    r"fifty|sixty|seventy|eighty|ninety|hundred|thousand|\d+"
)
_VERBS = r"pick|name|list|show|give|return|select|find|retrieve|provide|display|identify|get"
_COUNT_CUE = re.compile(
    rf"\b(?:{_VERBS})\s+(?:me\s+)?(?:the\s+)?(?:only\s+)?(?:{_COUNT_WORDS})\b"
    rf"|\b(?:{_COUNT_WORDS})\s+(?:rows?|records?|results?|entries|items?)\b"
    r"|\b\d+(?:st|nd|rd|th)\b"
    rf"|\b(?:{_VERBS})\s+(?:me\s+)?(?:the\s+)?(?:any|exactly)\s+(?:{_COUNT_WORDS})\b",
    re.I,
)
_CLAUSE_ORDER = {"FROM": 0, "WHERE": 1, "GROUP": 2, "HAVING": 3,
                 "WINDOW": 4, "ORDER": 5, "LIMIT": 6, "OFFSET": 7}
_NON_COLUMNS = {"NULL", "TRUE", "FALSE", "CURRENT_TIME", "CURRENT_DATE", "CURRENT_TIMESTAMP"}


@dataclass(frozen=True)
class _Expression:
    """A validated expression span; subqueries are separate SELECT scopes."""

    kind: str
    start: int
    end: int
    children: tuple[_Expression, ...] = ()


def _unparen(node: _Expression) -> _Expression:
    while node.kind == "paren":
        node = node.children[0]
    return node


def _split(tokens: list[_Token], start: int, end: int, text: str = ",") -> list[tuple[int, int]]:
    depth = tokens[start].depth if start < end else -1
    boundaries = [i for i in range(start, end) if tokens[i].depth == depth and tokens[i].text == text]
    return list(zip([start, *(i + 1 for i in boundaries)], [*boundaries, end]))


def _clauses(tokens: list[_Token], scope: _Select) -> list[tuple[str, int, int]]:
    starts = []
    for i in range(scope.start + 1, scope.end):
        token = tokens[i]
        if token.depth != scope.depth:
            continue
        word = token.word
        if (word in {"FROM", "WHERE", "HAVING", "LIMIT", "OFFSET"}
                or word in {"GROUP", "ORDER"} and i + 1 < scope.end and tokens[i + 1].word == "BY"
                or word == "WINDOW" and i + 3 < scope.end and _identifier(tokens[i + 1])
                and tokens[i + 2].word == "AS" and tokens[i + 3].text == "("):
            starts.append((word, i))
    if any(_CLAUSE_ORDER[a[0]] >= _CLAUSE_ORDER[b[0]] for a, b in zip(starts, starts[1:])):
        raise ValueError("Uncertain SELECT clause order")
    return [(word, start, starts[n + 1][1] if n + 1 < len(starts) else scope.end)
            for n, (word, start) in enumerate(starts)]


def _query(sql: str) -> tuple[list[_Token], dict[int, int], list[_Select]]:
    tokens = _tokens(sql)
    if tokens and tokens[-1].text == ";":
        tokens.pop()
    if (not tokens or tokens[0].word not in {"SELECT", "WITH"}
            or any(token.text == ";" for token in tokens)):
        raise ValueError("Expected one SELECT query")
    pairs, scopes = _pairs(tokens), _selects(tokens)
    if not any(scope.depth == 0 for scope in scopes):
        raise ValueError("Missing outer SELECT")
    for i, token in enumerate(tokens):
        if token.word == "WITH":
            _with_prefix(tokens, i, pairs)
        elif token.word in _COMPOUND:
            arm = i + 1
            if arm < len(tokens) and tokens[arm].word in {"ALL", "DISTINCT"}:
                arm += 1
            if arm >= len(tokens) or tokens[arm].word != "SELECT" or tokens[arm].depth != token.depth:
                raise ValueError("Uncertain compound SELECT")
    for scope in scopes:
        clauses = _clauses(tokens, scope)
        projection_end = clauses[0][1] if clauses else scope.end
        first = scope.start + 1
        if first < projection_end and tokens[first].word in {"DISTINCT", "ALL"}:
            first += 1
        if first == projection_end or any(start == end for start, end in _split(tokens, first, projection_end)):
            raise ValueError("Empty SELECT expression")
        for name, start, end in clauses:
            if start + (2 if name in {"ORDER", "GROUP"} else 1) >= end:
                raise ValueError("Empty SELECT clause")
        for start, end, context in _roots(tokens, scope, pairs):
            reader = _order_expression if context == "ORDER" else _projection_expression if context == "projection" else _expression
            reader(tokens, start, end, pairs)
    return tokens, pairs, scopes


def _with_prefix(tokens: list[_Token], start: int, pairs: dict[int, int]) -> None:
    i = start + 1
    if i < len(tokens) and tokens[i].word == "RECURSIVE":
        i += 1
    while i < len(tokens):
        if not _identifier(tokens[i]):
            raise ValueError("Uncertain CTE name")
        i += 1
        if i < len(tokens) and tokens[i].text == "(":
            closing = pairs[i]
            if any(b != a + 1 or not _identifier(tokens[a]) for a, b in _split(tokens, i + 1, closing)):
                raise ValueError("Uncertain CTE columns")
            i = closing + 1
        if i >= len(tokens) or tokens[i].word != "AS":
            raise ValueError("Missing CTE AS")
        i += 1
        if i < len(tokens) and tokens[i].word == "NOT":
            i += 1
            if i >= len(tokens) or tokens[i].word != "MATERIALIZED":
                raise ValueError("Uncertain CTE materialization")
        if i < len(tokens) and tokens[i].word == "MATERIALIZED":
            i += 1
        if i + 1 >= len(tokens) or tokens[i].text != "(" or tokens[i + 1].word not in {"SELECT", "WITH"}:
            raise ValueError("Uncertain CTE query")
        i = pairs[i] + 1
        if i < len(tokens) and tokens[i].text == ",":
            i += 1
            continue
        if i >= len(tokens) or tokens[i].word != "SELECT":
            raise ValueError("Missing SELECT after CTE")
        return
    raise ValueError("Unclosed CTE")


def _sources(tokens: list[_Token], start: int, end: int, pairs: dict[int, int]) -> list[tuple[int, int, str]]:
    """Validate FROM/JOIN boundaries and expose only their expression children."""
    roots, i, depth = [], start, tokens[start].depth
    join_words = {"LEFT", "RIGHT", "FULL", "OUTER", "INNER", "CROSS", "NATURAL"}
    while i < end:
        if tokens[i].text == "(":
            closing = pairs[i]
            if closing >= end:
                raise ValueError("Uncertain FROM parentheses")
            if tokens[i + 1].word not in {"SELECT", "WITH"}:
                roots.extend(_sources(tokens, i + 1, closing, pairs))
            i = closing + 1
        else:
            reference = _reference(tokens, i, end)
            if reference is None or len(reference[1]) > 2:
                raise ValueError("Uncertain table source")
            source_start, i = i, reference[0]
            if i < end and tokens[i].text == "(":
                i = pairs[i] + 1
                roots.append((source_start, i, "source"))
        if i < end and tokens[i].word == "AS":
            i += 1
            if i >= end or not _identifier(tokens[i]):
                raise ValueError("Missing table alias")
            i += 1
        elif i < end and _identifier(tokens[i]) and tokens[i].word not in _NOT_ALIAS:
            i += 1
        if i < end and tokens[i].word == "INDEXED":
            if i + 2 >= end or tokens[i + 1].word != "BY" or not _identifier(tokens[i + 2]):
                raise ValueError("Uncertain index clause")
            i += 3
        elif i + 1 < end and tokens[i].word == "NOT" and tokens[i + 1].word == "INDEXED":
            i += 2
        if i < end and tokens[i].word == "ON":
            stop = next((j for j in range(i + 1, end) if tokens[j].depth == depth
                         and (tokens[j].word in join_words | {"JOIN"} or tokens[j].text == ",")), end)
            roots.append((i + 1, stop, "ON"))
            i = stop
        elif i < end and tokens[i].word == "USING":
            if i + 1 >= end or tokens[i + 1].text != "(":
                raise ValueError("Uncertain USING clause")
            closing = pairs[i + 1]
            if any(b != a + 1 or not _identifier(tokens[a]) for a, b in _split(tokens, i + 2, closing)):
                raise ValueError("Uncertain USING columns")
            i = closing + 1
        if i == end:
            break
        if tokens[i].text == ",":
            i += 1
        else:
            while i < end and tokens[i].word in join_words:
                i += 1
            if i >= end or tokens[i].word != "JOIN":
                raise ValueError("Uncertain JOIN boundary")
            i += 1
        if i == end:
            raise ValueError("Missing joined source")
    return roots


def _roots(tokens: list[_Token], scope: _Select, pairs: dict[int, int]) -> list[tuple[int, int, str]]:
    clauses = _clauses(tokens, scope)
    projection_end = clauses[0][1] if clauses else scope.end
    first = scope.start + 1
    if tokens[first].word in {"DISTINCT", "ALL"}:
        first += 1
    roots = [(a, b, "projection") for a, b in _split(tokens, first, projection_end)]
    for name, start, end in clauses:
        if name in {"ORDER", "GROUP"}:
            roots.extend((a, b, name) for a, b in _split(tokens, start + 2, end))
        elif name in {"WHERE", "HAVING", "LIMIT", "OFFSET"}:
            roots.append((start + 1, end, name))
        elif name == "FROM":
            roots.extend(_sources(tokens, start + 1, end, pairs))
    return roots


def _expression(tokens: list[_Token], start: int, end: int, pairs: dict[int, int]) -> _Expression:
    """Read expression structure conservatively, never infer ancestors from substring matches."""
    if start >= end:
        raise ValueError("Empty SQL expression")
    token, depth = tokens[start], tokens[start].depth
    if token.text == "(" and pairs[start] == end - 1:
        if tokens[start + 1].word in {"SELECT", "WITH"}:
            return _Expression("subquery", start, end)
        return _Expression("paren", start, end, (_expression(tokens, start + 1, end - 1, pairs),))

    # CASE separators and BETWEEN's AND belong to their expression rather than the
    # surrounding binary operator. Leave unfamiliar CASE shapes unparsed.
    own = []
    case_depth, between = 0, False
    for i in range(start, end):
        part = tokens[i]
        if part.depth != depth:
            continue
        if part.word == "CASE":
            case_depth += 1
        elif part.word == "END":
            case_depth -= 1
        if case_depth:
            continue
        if part.word == "BETWEEN":
            between = True
        elif part.word == "AND" and between:
            between = False
            continue
        own.append(i)
    if case_depth:
        raise ValueError("Unclosed CASE")
    for operators in ({"OR"}, {"AND"}, {"=", "==", "!=", "<>", "<", ">", "<=", ">=", "IS", "IN", "LIKE", "GLOB", "MATCH", "REGEXP", "BETWEEN"},
                      {"|", "&", "<<", ">>"}, {"+", "-"}, {"*", "/", "%"}, {"||", "->", "->>"}):
        hits = [i for i in own if start < i < end - 1 and (tokens[i].word or tokens[i].text) in operators]
        if not hits:
            continue
        i = hits[-1]
        word = tokens[i].word or tokens[i].text
        left_end = i - 1 if tokens[i - 1].word == "NOT" and word in {"IN", "LIKE", "GLOB", "MATCH", "REGEXP", "BETWEEN"} else i
        right_start = i + 1
        if word == "IS" and tokens[right_start].word == "NOT":
            right_start += 1
        if word == "BETWEEN":
            middle = next((j for j in range(right_start, end) if tokens[j].depth == depth and tokens[j].word == "AND"), None)
            if middle is None:
                raise ValueError("Unclosed BETWEEN")
            children = (_expression(tokens, start, left_end, pairs),
                        _expression(tokens, right_start, middle, pairs), _expression(tokens, middle + 1, end, pairs))
        elif word == "IN" and tokens[right_start].text == "(" and pairs[right_start] == end - 1:
            if tokens[right_start + 1].word in {"SELECT", "WITH"}:
                rhs = (_Expression("subquery", right_start, end),)
            else:
                rhs = tuple(_expression(tokens, a, b, pairs) for a, b in _split(tokens, right_start + 1, end - 1))
            children = (_expression(tokens, start, left_end, pairs), *rhs)
        else:
            children = (_expression(tokens, start, left_end, pairs), _expression(tokens, right_start, end, pairs))
        return _Expression("division" if word == "/" else "binary", start, end, children)
    if token.text in {"+", "-", "~"} or token.word in {"NOT", "EXISTS"}:
        return _Expression("unary", start, end, (_expression(tokens, start + 1, end, pairs),))
    if len(own) >= 2 and tokens[own[-2]].word == "COLLATE" and _identifier(tokens[own[-1]]):
        return _Expression("collate", start, end, (_expression(tokens, start, own[-2], pairs),))
    if token.word == "CASE" and tokens[end - 1].word == "END":
        separators = []
        nested = 0
        for i in range(start + 1, end - 1):
            if tokens[i].depth != depth:
                continue
            nested += (tokens[i].word == "CASE") - (tokens[i].word == "END")
            if nested == 0 and tokens[i].word in {"WHEN", "THEN", "ELSE"}:
                separators.append(i)
        if not separators or tokens[separators[0]].word != "WHEN":
            raise ValueError("Uncertain CASE")
        chunks = []
        if separators[0] > start + 1:
            chunks.append((start + 1, separators[0]))
        chunks.extend((i + 1, separators[n + 1] if n + 1 < len(separators) else end - 1)
                      for n, i in enumerate(separators))
        return _Expression("case", start, end, tuple(_expression(tokens, a, b, pairs) for a, b in chunks))
    reference = _reference(tokens, start, end)
    if reference is not None and reference[0] == end and token.word not in _NON_COLUMNS:
        return _Expression("column", start, end)
    if (reference is not None and reference[0] + 2 == end
            and tokens[end - 2].text == "." and tokens[end - 1].text == "*"):
        return _Expression("wildcard", start, end)
    if end == start + 1 and (token.kind in {"number", "string"} or token.word in _NON_COLUMNS or token.text == "*"):
        return _Expression("literal", start, end)
    if start + 1 < end and _identifier(token) and tokens[start + 1].text == "(":
        closing = pairs[start + 1]
        if closing >= end:
            raise ValueError("Uncertain function span")
        if closing + 1 < end:
            suffix = closing + 1
            # A window ancestor blocks every candidate in its function or window spec.
            if tokens[suffix].word == "OVER" and suffix + 1 < end:
                if (tokens[suffix + 1].text == "(" and pairs[suffix + 1] == end - 1
                        or suffix + 2 == end and _identifier(tokens[suffix + 1])):
                    return _Expression("window", start, end, (_expression(tokens, start, closing + 1, pairs),))
            if (tokens[suffix].word == "FILTER" and suffix + 3 < end and tokens[suffix + 1].text == "("
                    and tokens[suffix + 2].word == "WHERE" and pairs[suffix + 1] == end - 1):
                return _Expression("filter", start, end, (_expression(tokens, start, closing + 1, pairs),
                                                           _expression(tokens, suffix + 3, end - 1, pairs)))
            raise ValueError("Uncertain function suffix")
        if token.word == "CAST":
            as_indices = [i for i in range(start + 2, closing) if tokens[i].depth == depth + 1 and tokens[i].word == "AS"]
            if len(as_indices) != 1 or as_indices[0] + 2 != closing:
                raise ValueError("Uncertain CAST")
            i = as_indices[0]
            child = _expression(tokens, start + 2, i, pairs)
            return _Expression("cast" if tokens[i + 1].word in {"INTEGER", "REAL"} else "other_cast", start, end, (child,))
        argument_start = start + 2
        distinct = argument_start < closing and tokens[argument_start].word == "DISTINCT"
        if argument_start < closing and tokens[argument_start].word in {"DISTINCT", "ALL"}:
            argument_start += 1
        arguments = (() if argument_start == closing else
                     tuple(_expression(tokens, a, b, pairs) for a, b in _split(tokens, argument_start, closing)))
        kind = token.word.lower() if token.word in {"MIN", "MAX"} and not distinct else "function"
        return _Expression(kind, start, end, arguments)
    raise ValueError("Unsupported SQL expression")


def _projection_expression(tokens: list[_Token], start: int, end: int, pairs: dict[int, int]) -> _Expression:
    try:
        return _expression(tokens, start, end, pairs)
    except ValueError:
        if end - start >= 3 and tokens[end - 2].word == "AS" and _identifier(tokens[end - 1]):
            return _expression(tokens, start, end - 2, pairs)
        if end - start >= 2 and _identifier(tokens[end - 1]):
            return _expression(tokens, start, end - 1, pairs)
        raise


def _order_expression(tokens: list[_Token], start: int, end: int, pairs: dict[int, int]) -> _Expression:
    if end - start >= 2 and tokens[end - 2].word == "NULLS" and tokens[end - 1].word in {"FIRST", "LAST"}:
        end -= 2
    if end > start and tokens[end - 1].word in {"ASC", "DESC"}:
        end -= 1
    return _expression(tokens, start, end, pairs)


def _cast_spans(node: _Expression, *, ordered: bool = False, blocked: bool = False) -> list[_Expression]:
    result = []
    bare = _unparen(node)
    if ordered and not blocked and bare.kind == "cast" and _unparen(bare.children[0]).kind == "column":
        result.append(bare)
    blocked |= node.kind in {"division", "window"}
    if node.kind in {"min", "max"} and len(node.children) == 1:
        argument = _unparen(node.children[0])
        if not blocked and argument.kind == "cast" and _unparen(argument.children[0]).kind == "column":
            result.append(argument)
    for child in node.children:
        result.extend(_cast_spans(child, blocked=blocked))
    return result


def _compound_scope(tokens: list[_Token], scope: _Select, pairs: dict[int, int]) -> bool:
    opening = next((i for i, closing in pairs.items() if i < scope.start < closing
                    and tokens[i].text == "(" and tokens[i].depth == scope.depth - 1), None)
    first, last = (0, len(tokens)) if opening is None else (opening + 1, pairs[opening])
    return any(token.depth == scope.depth and token.word in _COMPOUND for token in tokens[first:last])


def bare_selection_cast(sql: str) -> tuple[str | None, str | None]:
    """Build a source-preserving bare INTEGER/REAL selection-CAST variant without execution.

    Args:
        sql (str): One SQLite SELECT query, optionally with CTEs or compound arms.

    Returns:
        tuple[str | None, str | None]: Variant and no skip reason, or no variant and a
            structural skip reason. Unsupported query or expression structure abstains.
    """
    try:
        tokens, pairs, scopes = _query(sql)
        casts: dict[tuple[int, int], _Expression] = {}
        for scope in scopes:
            for start, end, context in _roots(tokens, scope, pairs):
                reader = _order_expression if context == "ORDER" else _projection_expression if context == "projection" else _expression
                root = reader(tokens, start, end, pairs)
                # An ORDER attached to a compound query is not a SELECT's ORDER.
                for cast in _cast_spans(root, ordered=context == "ORDER" and not _compound_scope(tokens, scope, pairs)):
                    casts[cast.start, cast.end] = cast
        replacements = []
        for cast in casts.values():
            operand = cast.children[0]
            replacements.append((tokens[cast.start].start, tokens[cast.end - 1].end,
                                 sql[tokens[operand.start].start:tokens[operand.end - 1].end]))
        return (_replace(sql, replacements), None) if replacements else (None, "no_bare_selection_cast")
    except (ValueError, IndexError, KeyError, RecursionError):
        return None, "unparsed"


def _has_window(tokens: list[_Token]) -> bool:
    return any((token.word == "OVER" and i > 0 and tokens[i - 1].text == ")" and i + 1 < len(tokens)
                and (tokens[i + 1].text == "(" or _identifier(tokens[i + 1]) and tokens[i + 1].word not in _NOT_ALIAS)
                or token.word == "WINDOW" and i + 3 < len(tokens) and _identifier(tokens[i + 1])
                and tokens[i + 2].word == "AS" and tokens[i + 3].text == "(")
               for i, token in enumerate(tokens))


def uncued_single_row_selection(question: str, sql: str) -> tuple[str | None, str | None]:
    """Remove only uncued outer ORDER BY and literal LIMIT 1 without executing SQL.

    Args:
        question (str): Question text together with its hint/evidence.
        sql (str): One SQLite SELECT query.

    Returns:
        tuple[str | None, str | None]: Variant and no skip reason, or no variant and
            the cue, parse, or selection-shape skip reason.
    """
    if _CUE.search(question) or _COUNT_CUE.search(question):
        return None, "selection_cue"
    try:
        tokens, pairs, scopes = _query(sql)
        outer = next(scope for scope in scopes if scope.depth == 0)
        if any(token.depth == 0 and token.word in _COMPOUND for token in tokens):
            return None, "compound_query"
        clauses = {name: (start, end) for name, start, end in _clauses(tokens, outer)}
        if "GROUP" in clauses or "HAVING" in clauses or "OFFSET" in clauses:
            return None, "group_having_or_offset"
        if _has_window(tokens):
            return None, "windowed"
        if any(name == "LIMIT" for scope in scopes if scope is not outer for name, _, _ in _clauses(tokens, scope)):
            return None, "nested_limit"
        if "ORDER" not in clauses or "LIMIT" not in clauses:
            return None, "no_order_limit"
        limit_start, limit_end = clauses["LIMIT"]
        if (limit_end != limit_start + 2 or tokens[limit_start + 1].kind != "number"
                or not tokens[limit_start + 1].text.isdigit() or int(tokens[limit_start + 1].text) != 1):
            return None, "not_limit_one"
        order_start, order_end = clauses["ORDER"]
        keys = _split(tokens, order_start + 2, order_end)
        if len(keys) != 1 or _unparen(_order_expression(tokens, *keys[0], pairs)).kind != "column":
            return None, "not_single_bare_order_key"
        return _replace(sql, [(tokens[order_start].start, tokens[order_end - 1].end, ""),
                              (tokens[limit_start].start, tokens[limit_end - 1].end, "")]), None
    except (ValueError, IndexError, KeyError, StopIteration, RecursionError):
        return None, "unparsed"


def _complete(result: dict[str, Any]) -> bool:
    return (_successful(result) and not any(result.get(key) for key in
                                           ("truncated", "timeout", "timed_out", "partial", "incomplete")))


def _arity(result: dict[str, Any]) -> int | None:
    widths = {len(_values(row)) for row in result["rows"]}
    columns = result.get("columns")
    if columns is not None:
        if not isinstance(columns, (list, tuple)):
            return None
        widths.add(len(columns))
    return next(iter(widths)) if len(widths) == 1 else None


async def apply_selection_normalization(
    question: str,
    chosen: dict[str, Any],
    execute: Callable[[str], Awaitable[dict[str, Any]]],
    result_sig: Callable[[dict[str, Any]], Hashable],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Try bare-selection CAST removal, then uncued single-row selection removal.

    Each operation uses the preceding accepted final. Both input and variant must be
    successful, complete, nonempty and have equal output arity. An unchanged official
    signature or any exception retains that operation's input. Candidate results have
    no veto. This is a deterministic benchmark convention, not SQL equivalence.

    Args:
        question (str): Question text, including its appended hint/evidence.
        chosen (dict[str, Any]): Executed final containing SQL and full result rows.
        execute (Callable[[str], Awaitable[dict[str, Any]]]): Read-only base-database executor.
        result_sig (Callable[[dict[str, Any]], Hashable]): Official execution-result signature.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: Final with detached rows/columns when an
            operation is accepted, otherwise ``chosen`` itself, plus an unattached
            version-1 debug record. ``operations`` contains both ordered operation
            records, including each input SQL, built variant, acceptance and skip.
            Top ``fired`` means at least one accepted operation; top ``variant_sql``
            is the accepted final SQL when fired, otherwise the last attempted SQL.
            Exceptions record only their feature and type and retain prior results.
    """
    debug: dict[str, Any] = {"v": 1, "implementation_version": IMPLEMENTATION_VERSION,
                             "pre_hook_sql": chosen.get("sql"), "fired": False, "skipped": None,
                             "operations": []}
    current, attempted_skip = chosen, None
    for name in OPERATION_NAMES:
        record: dict[str, Any] = {"name": name, "pre_hook_sql": current.get("sql"), "fired": False, "skipped": None}
        debug["operations"].append(record)
        try:
            sql = current.get("sql")
            if not isinstance(sql, str) or not _complete(current):
                record["skipped"] = "final_failed"
            elif _row_count(current) == 0:
                record["skipped"] = "final_empty"
            elif _arity(current) is None:
                record["skipped"] = "arity_changed"
            else:
                variant_sql, record["skipped"] = (bare_selection_cast(sql) if name == "bare_selection_cast"
                                                   else uncued_single_row_selection(question, sql))
                if variant_sql is not None:
                    record["variant_sql"] = variant_sql
                    debug["variant_sql"] = variant_sql
                    variant = await execute(variant_sql)
                    if not _complete(variant):
                        record["skipped"] = "variant_failed"
                    elif _row_count(variant) == 0:
                        record["skipped"] = "variant_empty"
                    elif _arity(variant) is None or _arity(variant) != _arity(current):
                        record["skipped"] = "arity_changed"
                    elif result_sig(variant) == result_sig(current):
                        record["skipped"] = "variant_matches_final"
                    else:
                        current = _updated(current, variant, variant_sql)
                        record["fired"] = True
                        debug["fired"] = True
        except Exception as exc:
            record["skipped"] = "error"
            record["errors"] = [{"feature": name, "type": type(exc).__name__}]
            debug.setdefault("errors", []).extend(record["errors"])
        if "variant_sql" in record or record["skipped"] in {"error", "final_failed", "final_empty", "arity_changed"}:
            attempted_skip = record["skipped"]
        debug["skipped"] = attempted_skip or record["skipped"]
    if debug["fired"]:
        debug["skipped"] = None
        debug["variant_sql"] = current["sql"]
    return current, debug
