"""Deterministic, read-only submission rules shared by voting and offline replay."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Awaitable, Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any

from .question import HINT_SEPARATOR

IMPLEMENTATION_VERSION = "submission-gate-3"
# The opt-in rule runs after the unchanged reviewer loop. Its knob moves the
# final keyspace while this separate identity preserves identical reviewer memos.
HINT_AGGREGATE_VERSION = "submission-gate-hint-aggregate-2"
# The shared SQL reader did not change with the LEFT-to-INNER support policy.
SQL_READER_VERSION = "submission-gate-2"
RULE_NAMES = (
    "percent_scale",
    "left_join_to_inner",
    "date_granularity",
    "derived_text_stored_integer",
    "multiply_first_percentage",
)

_PERCENT = re.compile(r"\bpercent(?:age)?s?\b", re.I)
_SCALE = re.compile(r"\*\s*100(\.0*)?\b|\b100(\.0*)?\s*\*")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_DATETIME = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}")
_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_HUNDRED = re.compile(r"100(?:\.0*)?\Z")
_HINT_AGGREGATE = re.compile(r"\b(AVG|SUM)\s*\(\s*([^()]+?)\s*\)", re.I)
_CLAUSES = {"FROM", "WHERE", "GROUP", "ORDER", "LIMIT", "HAVING", "WINDOW",
            "UNION", "INTERSECT", "EXCEPT", "RETURNING"}
_COMPOUND = {"UNION", "INTERSECT", "EXCEPT"}
_NOT_ALIAS = _CLAUSES | {"JOIN", "ON", "USING", "LEFT", "RIGHT", "FULL", "OUTER",
                         "INNER", "CROSS", "NATURAL", "INDEXED", "NOT", "AS"}
_GROUP_WORDS = _NOT_ALIAS | {"SELECT", "AND", "OR", "IN", "EXISTS", "WHEN", "THEN",
                            "ELSE", "BY", "BETWEEN", "DISTINCT", "ALL"}
_EXPR_OPERATORS = {"+", "-", "*", "/", "%", "&", "|", "<<", ">>", "~",
                   "||", "->", "->>", "."}


@dataclass(frozen=True)
class _Token:
    text: str
    start: int
    end: int
    kind: str
    depth: int

    @property
    def word(self) -> str:
        return self.text.upper() if self.kind == "word" else ""

    @property
    def name(self) -> str:
        if self.kind == "identifier":
            if self.text[0] == "[":
                return self.text[1:-1].replace("]]", "]")
            quote = self.text[0]
            return self.text[1:-1].replace(quote * 2, quote)
        return self.text


@dataclass(frozen=True)
class _Select:
    start: int
    list_end: int
    from_index: int | None
    end: int
    depth: int


@dataclass(frozen=True)
class _Table:
    name: str
    schema: str = "main"

    @property
    def sql(self) -> str:
        return f"{_quote_identifier(self.schema)}.{_quote_identifier(self.name)}"


@dataclass(frozen=True)
class _Binding:
    alias: str
    table: _Table | None


def _tokens(sql: str) -> list[_Token]:
    tokens: list[_Token] = []
    i, depth = 0, 0
    while i < len(sql):
        start, char = i, sql[i]
        if char.isspace():
            i += 1
            continue
        if sql.startswith("--", i):
            end = sql.find("\n", i + 2)
            i = len(sql) if end < 0 else end + 1
            continue
        if sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            if end < 0:
                raise ValueError("Unclosed SQL comment")
            i = end + 2
            continue
        if char in "'\"`[":
            closing = "]" if char == "[" else char
            i += 1
            while i < len(sql):
                if sql[i] == closing:
                    if i + 1 < len(sql) and sql[i + 1] == closing:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            else:
                raise ValueError("Unclosed SQL quote")
            kind = "string" if char == "'" else "identifier"
        elif char.isalpha() or char == "_":
            i += 1
            while i < len(sql) and (sql[i].isalnum() or sql[i] in "_$"):
                i += 1
            kind = "word"
        elif (match := _NUMBER.match(sql, i)) is not None:
            i = match.end()
            kind = "number"
        else:
            operator = next((op for op in ("->>", "||", "->", "<<", ">>", "<=", ">=", "==", "!=", "<>")
                             if sql.startswith(op, i)), char)
            i += len(operator)
            kind = "symbol"
        if char == ")" and kind == "symbol":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced SQL parentheses")
        tokens.append(_Token(sql[start:i], start, i, kind, depth))
        if char == "(" and kind == "symbol":
            depth += 1
    if depth:
        raise ValueError("Unbalanced SQL parentheses")
    return tokens


def _pairs(tokens: list[_Token]) -> dict[int, int]:
    pairs: dict[int, int] = {}
    stack: list[int] = []
    for i, token in enumerate(tokens):
        if token.text == "(":
            stack.append(i)
        elif token.text == ")":
            opening = stack.pop()
            pairs[opening] = i
            pairs[i] = opening
    return pairs


def _selects(tokens: list[_Token]) -> list[_Select]:
    result: list[_Select] = []
    for i, token in enumerate(tokens):
        if token.word != "SELECT":
            continue
        end = next((j for j in range(i + 1, len(tokens))
                    if tokens[j].depth < token.depth
                    or (tokens[j].depth == token.depth and tokens[j].word in _COMPOUND)),
                   len(tokens))
        list_end = next((j for j in range(i + 1, end)
                         if tokens[j].depth == token.depth and tokens[j].word in _CLAUSES), end)
        from_index = list_end if list_end < end and tokens[list_end].word == "FROM" else None
        result.append(_Select(i, list_end, from_index, end, token.depth))
    return result


def _outer_select(tokens: list[_Token]) -> _Select | None:
    return next((scope for scope in _selects(tokens) if scope.depth == 0), None)


def _code_sql(sql: str, tokens: list[_Token]) -> str:
    masked = [" "] * len(sql)
    for token in tokens:
        if token.kind not in {"string", "identifier"}:
            masked[token.start:token.end] = token.text
    return "".join(masked)


def _replace(sql: str, replacements: list[tuple[int, int, str]]) -> str:
    for start, end, value in sorted(replacements, reverse=True):
        sql = sql[:start] + value + sql[end:]
    return sql


def _values(row: Any) -> tuple[Any, ...]:
    return tuple(row.values()) if isinstance(row, dict) else tuple(row)


def _row_count(result: dict[str, Any]) -> int:
    return int(result.get("row_count", len(result.get("rows") or [])))


def _successful(result: dict[str, Any]) -> bool:
    rows = result.get("rows")
    try:
        return (not result.get("error") and isinstance(rows, (list, tuple))
                and _row_count(result) == len(rows))
    except (TypeError, ValueError, OverflowError):
        return False


def _scalar(result: dict[str, Any]) -> float | int | None:
    if not _successful(result) or _row_count(result) != 1:
        return None
    values = _values(result["rows"][0])
    if len(values) != 1 or len(result.get("columns", values)) != 1:
        return None
    value = values[0]
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def _updated(chosen: dict[str, Any], result: dict[str, Any], sql: str) -> dict[str, Any]:
    updated = dict(chosen)
    updated.update(sql=sql, rows=copy.deepcopy(result["rows"]),
                   columns=copy.deepcopy(result.get("columns", [])), row_count=_row_count(result))
    return updated


def _join_spans(tokens: list[_Token], kinds: set[str]) -> list[tuple[int, int]]:
    spans = []
    for i, token in enumerate(tokens):
        if token.word not in kinds:
            continue
        j = i + 1
        if j < len(tokens) and tokens[j].word == "OUTER":
            j += 1
        if j < len(tokens) and tokens[j].word == "JOIN":
            spans.append((i, j))
    return spans


def _prefer_multiply_first(sql: str) -> bool:
    tokens = _tokens(sql)
    outer = _outer_select(tokens)
    if outer is None:
        return False
    expression = tokens[outer.start + 1:outer.list_end]
    division = next((token.start for token in expression if token.text == "/"), len(sql))
    start = tokens[outer.start].end
    end = tokens[outer.list_end].start if outer.list_end < len(tokens) else len(sql)
    scale = _SCALE.search(_code_sql(sql, tokens), start, end)
    return scale is not None and scale.start() < division


def _percent_scale(
    question: str, chosen: dict[str, Any], candidates: Sequence[dict[str, Any]],
    result_sig: Callable[[dict[str, Any]], Hashable],
) -> tuple[dict[str, Any], int | None]:
    value = _scalar(chosen)
    if not _PERCENT.search(question.split("\n[Hint]: ", 1)[0]) or value in (None, 0):
        return chosen, None
    sql = chosen["sql"]
    tokens = _tokens(sql)
    outer = _outer_select(tokens)
    if (_SCALE.search(_code_sql(sql, tokens)) or outer is None
            or not any(token.text == "/" for token in tokens[outer.start + 1:outer.list_end])):
        return chosen, None
    qualifying = []
    groups: dict[Hashable, int] = {}
    for slot, candidate in enumerate(candidates, 1):
        number = _scalar(candidate)
        candidate_sql = candidate.get("sql")
        if number is None or not isinstance(candidate_sql, str):
            continue
        if (abs(number - 100 * value) > 1e-9 * max(1, abs(number))
                or not _SCALE.search(_code_sql(candidate_sql, _tokens(candidate_sql)))):
            continue
        signature = result_sig(candidate)
        groups[signature] = groups.get(signature, 0) + 1
        qualifying.append((slot, candidate, signature))
    if not qualifying:
        return chosen, None
    slot, candidate, _ = min(qualifying, key=lambda entry: (
        -groups[entry[2]], not _prefer_multiply_first(entry[1]["sql"]), entry[0]))
    return _updated(chosen, candidate, candidate["sql"]), slot


async def _left_join_to_inner(
    chosen: dict[str, Any], candidates: Sequence[dict[str, Any]],
    execute: Callable[[str], Awaitable[dict[str, Any]]],
    result_sig: Callable[[dict[str, Any]], Hashable],
    *, min_support_margin: int = 1,
) -> dict[str, Any]:
    sql = chosen["sql"]
    tokens = _tokens(sql)
    spans = _join_spans(tokens, {"LEFT"})
    if not spans:
        return chosen
    agreement = [candidate for candidate in candidates
                 if _successful(candidate) and isinstance(candidate.get("sql"), str)
                 and not _join_spans(_tokens(candidate["sql"]), {"LEFT", "RIGHT", "FULL"})]
    if not agreement:
        return chosen
    replacements = []
    for first, last in spans:
        if (sql[tokens[first].end:tokens[first + 1].start].isspace()
                and (last == first + 1
                     or sql[tokens[first + 1].end:tokens[last].start].isspace())):
            replacements.append((tokens[first].start, tokens[last].end, "INNER JOIN"))
            continue
        # Comments between keywords remain in place.
        replacements.append((tokens[first].start, tokens[first].end, "INNER"))
        if last == first + 2:
            replacements.append((tokens[first + 1].start, tokens[first + 1].end, ""))
    variant_sql = _replace(sql, replacements)
    variant = await execute(variant_sql)
    if (_successful(variant) and _row_count(variant) > 0
            and any(result_sig(variant) == result_sig(candidate) for candidate in agreement)):
        chosen_support = sum(_successful(candidate) and result_sig(candidate) == result_sig(chosen)
                             for candidate in candidates)
        inner_support = sum(result_sig(candidate) == result_sig(variant) for candidate in agreement)
        if inner_support - chosen_support < min_support_margin:
            return chosen
        return _updated(chosen, variant, variant_sql)
    return chosen


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _quote_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _identifier(token: _Token) -> bool:
    return token.kind in {"word", "identifier"}


def _reference(tokens: list[_Token], start: int, end: int) -> tuple[int, tuple[str, ...]] | None:
    if start >= end or not _identifier(tokens[start]):
        return None
    names = [tokens[start].name]
    i = start + 1
    while i + 1 < end and tokens[i].text == "." and _identifier(tokens[i + 1]):
        names.append(tokens[i + 1].name)
        i += 2
    return (i, tuple(names)) if len(names) <= 3 else None


def _reference_before(tokens: list[_Token], end: int) -> tuple[int, tuple[str, ...]] | None:
    start = end - 1
    if start < 0 or not _identifier(tokens[start]):
        return None
    while start >= 2 and tokens[start - 1].text == "." and _identifier(tokens[start - 2]):
        start -= 2
    reference = _reference(tokens, start, end)
    return (start, reference[1]) if reference is not None and reference[0] == end else None


def _cte_names(tokens: list[_Token], pairs: dict[int, int], scope: _Select) -> set[str]:
    names = set()
    for i, token in enumerate(tokens):
        if token.word != "WITH" or i >= scope.start or token.depth > scope.depth:
            continue
        enclosing = next((closing for opening, closing in pairs.items()
                          if opening < i < closing and tokens[opening].depth == token.depth - 1),
                         len(tokens))
        if scope.start >= enclosing:
            continue
        j = i + 1
        if j < len(tokens) and tokens[j].word == "RECURSIVE":
            j += 1
        while j < len(tokens) and _identifier(tokens[j]):
            name = tokens[j].name.casefold()
            j += 1
            if j < len(tokens) and tokens[j].text == "(":
                j = pairs[j] + 1
            if j >= len(tokens) or tokens[j].word != "AS":
                break
            j += 1
            while j < len(tokens) and tokens[j].word in {"NOT", "MATERIALIZED"}:
                j += 1
            if j >= len(tokens) or tokens[j].text != "(":
                break
            names.add(name)
            j = pairs[j] + 1
            if j >= len(tokens) or tokens[j].text != ",":
                break
            j += 1
    return names


def _bindings(tokens: list[_Token], scope: _Select, pairs: dict[int, int]) -> list[_Binding]:
    if scope.from_index is None:
        return []
    ctes = _cte_names(tokens, pairs, scope)
    end = next((j for j in range(scope.from_index + 1, scope.end)
                if tokens[j].depth == scope.depth and tokens[j].word in _CLAUSES), scope.end)
    bindings = []
    i, needs_source = scope.from_index + 1, True
    while i < end:
        token = tokens[i]
        if token.depth != scope.depth:
            i += 1
            continue
        if token.word == "JOIN" or token.text == ",":
            needs_source = True
            i += 1
            continue
        if not needs_source:
            i += 1
            continue
        table, default_alias = None, ""
        grouped: list[_Binding] = []
        if token.text == "(":
            closing = pairs[i]
            if not any(inner.word in {"SELECT", "WITH"} and inner.depth == scope.depth + 1
                       for inner in tokens[i + 1:closing]):
                grouped = _bindings(tokens, _Select(scope.start, i, i, closing, scope.depth + 1), pairs)
            i = closing + 1
        else:
            reference = _reference(tokens, i, end)
            if reference is None or len(reference[1]) > 2:
                return []
            i, names = reference
            default_alias = names[-1]
            if len(names) == 2 or default_alias.casefold() not in ctes:
                table = _Table(default_alias, names[0] if len(names) == 2 else "main")
            if i < end and tokens[i].text == "(":
                table = None
                i = pairs[i] + 1
        if i < end and tokens[i].word == "AS":
            i += 1
        alias = default_alias
        if i < end and _identifier(tokens[i]) and tokens[i].word not in _NOT_ALIAS:
            alias = tokens[i].name
            i += 1
        if alias and len(grouped) == 1 and grouped[0].table is not None:
            table = grouped[0].table
            grouped = []
        bindings.extend(grouped)
        if not grouped or alias:
            bindings.append(_Binding(alias, table))
        needs_source = False
    return bindings


class _Catalog:
    def __init__(self, execute: Callable[[str], Awaitable[dict[str, Any]]]) -> None:
        self.execute = execute
        self.columns: dict[_Table, tuple[str, ...]] = {}
        self.date_columns: dict[tuple[_Table, str], bool] = {}

    async def table_columns(self, table: _Table) -> tuple[str, ...]:
        if table not in self.columns:
            metadata = await self.execute(
                f"SELECT name FROM {_quote_identifier(table.schema)}.sqlite_master "
                f"WHERE type = 'table' AND lower(name) = lower({_quote_string(table.name)})")
            if not _successful(metadata) or _row_count(metadata) != 1:
                self.columns[table] = ()
            else:
                columns = await self.execute(
                    f"SELECT name FROM pragma_table_info({_quote_string(table.name)}, "
                    f"{_quote_string(table.schema)})")
                self.columns[table] = (tuple(str(_values(row)[0]) for row in columns["rows"])
                                       if _successful(columns) else ())
        return self.columns[table]

    async def resolve(self, names: tuple[str, ...], bindings: list[_Binding]) -> tuple[_Table, str] | None:
        column = names[-1].casefold()
        if len(names) > 1:
            relevant = [binding for binding in bindings
                        if binding.alias.casefold() == names[-2].casefold()]
            if len(names) == 3:
                relevant = [binding for binding in relevant if binding.table is not None
                            and binding.table.schema.casefold() == names[0].casefold()]
        else:
            # Unknown derived outputs can also contain the unqualified name.
            if any(binding.table is None for binding in bindings):
                return None
            relevant = bindings
        resolved = []
        for binding in relevant:
            if binding.table is None:
                continue
            columns = await self.table_columns(binding.table)
            resolved.extend((binding.table, name) for name in columns if name.casefold() == column)
        return resolved[0] if len(resolved) == 1 else None

    async def has_datetime(self, table: _Table, column: str) -> bool:
        key = table, column
        if key not in self.date_columns:
            col = _quote_identifier(column)
            sample = await self.execute(
                f"SELECT {col} FROM {table.sql} WHERE {col} IS NOT NULL LIMIT 200")
            present = (_successful(sample) and any(
                isinstance(value := _values(row)[0], str) and _DATETIME.match(value)
                for row in sample["rows"]))
            valid = False
            if present:
                invalid = await self.execute(
                    f"SELECT COUNT(*) FROM {table.sql} "
                    f"WHERE {col} IS NOT NULL AND "
                    f"(date({col}) IS NULL OR date({col}) <> substr({col}, 1, 10))")
                valid = _scalar(invalid) == 0
            self.date_columns[key] = bool(present and valid)
        return self.date_columns[key]


def _function_open(tokens: list[_Token], opening: int) -> bool:
    if opening == 0:
        return False
    previous = tokens[opening - 1]
    return (_identifier(previous) and previous.word not in _GROUP_WORDS
            or previous.kind in {"number", "string"})


def _bare_reference(
    tokens: list[_Token], start: int, end: int, scope: _Select, pairs: dict[int, int],
) -> bool:
    if any(opening < start < closing and opening >= scope.start and _function_open(tokens, opening)
           for opening, closing in pairs.items() if opening < closing):
        return False
    if start > 0:
        previous = tokens[start - 1]
        if (previous.text in _EXPR_OPERATORS or previous.text == ")"
                or previous.kind in {"identifier", "number", "string"}
                or previous.kind == "word" and previous.word not in _GROUP_WORDS):
            return False
    if end < len(tokens):
        following = tokens[end]
        if (following.text in _EXPR_OPERATORS or following.text == "("
                or following.word == "COLLATE"):
            return False
    return True


def _bare_literal(tokens: list[_Token], index: int) -> bool:
    if index > 0 and tokens[index - 1].text in _EXPR_OPERATORS:
        return False
    if index + 1 >= len(tokens):
        return True
    following = tokens[index + 1]
    if (following.text in _EXPR_OPERATORS or following.text == "("
            or following.kind in {"identifier", "number", "string"}):
        return False
    return (following.kind != "word"
            or following.word in _CLAUSES | {"AND", "OR", "THEN", "ELSE", "END", "AS"})


async def _resolve_date_column(
    names: tuple[str, ...], index: int, tokens: list[_Token], scopes: list[_Select],
    pairs: dict[int, int], catalog: _Catalog,
) -> tuple[_Table, str] | None:
    visible = sorted((scope for scope in scopes if scope.start < index < scope.end),
                     key=lambda scope: (scope.depth, scope.start), reverse=True)
    for scope in visible:
        bindings = _bindings(tokens, scope, pairs)
        if len(names) > 1:
            if any(binding.alias.casefold() == names[-2].casefold() for binding in bindings):
                return await catalog.resolve(names, bindings)
            continue
        if any(binding.table is None for binding in bindings):
            return None
        columns = [await catalog.table_columns(binding.table) for binding in bindings
                   if binding.table is not None]
        if any(not table_columns for table_columns in columns):
            return None
        if any(column.casefold() == names[0].casefold() for table_columns in columns
               for column in table_columns):
            return await catalog.resolve(names, bindings)
    return None


async def _date_granularity(chosen: dict[str, Any], catalog: _Catalog) -> dict[str, Any]:
    sql = chosen["sql"]
    tokens = _tokens(sql)
    pairs, scopes = _pairs(tokens), _selects(tokens)
    replacements: dict[tuple[int, int], str] = {}
    flipped = {">": "<", "<": ">", ">=": "<=", "<=": ">=", "=": "=", "==": "=="}
    for i, token in enumerate(tokens):
        if (token.kind != "string" or not _DATE.fullmatch(token.text[1:-1].replace("''", "'"))
                or not _bare_literal(tokens, i)):
            continue
        scope = max((scope for scope in scopes if scope.start < i < scope.end),
                    key=lambda scope: (scope.depth, scope.start), default=None)
        if scope is None:
            continue
        reference, start, end = None, 0, 0
        if i >= 2 and tokens[i - 1].text in {">", "<=", "=", "=="}:
            backward = _reference_before(tokens, i - 1)
            if backward is not None:
                start, reference = backward
                end = i - 1
        elif i + 2 < len(tokens) and flipped.get(tokens[i + 1].text) in {">", "<=", "=", "=="}:
            forward = _reference(tokens, i + 2, scope.end)
            if forward is not None:
                start, (end, reference) = i + 2, forward
        elif i >= 2 and tokens[i - 1].word == "AND":
            between = next((j for j in range(i - 2, scope.start, -1)
                            if tokens[j].depth == token.depth
                            and tokens[j].word in {"BETWEEN", "AND", "OR", "WHERE", "ON"}), None)
            if between is not None and tokens[between].word == "BETWEEN":
                lower_bound = tokens[between + 1:i - 1]
                if (len(lower_bound) != 1 or lower_bound[0].kind != "string"
                        or not _DATE.fullmatch(lower_bound[0].text[1:-1].replace("''", "'"))):
                    continue
                end = between - 1 if tokens[between - 1].word == "NOT" else between
                backward = _reference_before(tokens, end)
                if backward is not None:
                    start, reference = backward
        if reference is None or not _bare_reference(tokens, start, end, scope, pairs):
            continue
        resolved = await _resolve_date_column(reference, i, tokens, scopes, pairs, catalog)
        if resolved is not None and await catalog.has_datetime(*resolved):
            first, last = tokens[start].start, tokens[end - 1].end
            replacements[first, last] = f"date({sql[first:last]})"
    if not replacements:
        return chosen
    variant_sql = _replace(sql, [(start, end, value) for (start, end), value in replacements.items()])
    variant = await catalog.execute(variant_sql)
    return _updated(chosen, variant, variant_sql) if _successful(variant) else chosen


def _select_items(tokens: list[_Token], scope: _Select) -> list[tuple[int, int]]:
    start = scope.start + 1
    if start < scope.list_end and tokens[start].word in {"DISTINCT", "ALL"}:
        start += 1
    items = []
    for i in range(start, scope.list_end):
        if tokens[i].depth == scope.depth and tokens[i].text == ",":
            items.append((start, i))
            start = i + 1
    items.append((start, scope.list_end))
    return items


def _derived_call(
    tokens: list[_Token], start: int, end: int, pairs: dict[int, int],
) -> tuple[int, int, tuple[str, ...]] | None:
    if end > start + 2 and tokens[end - 2].word == "AS" and _identifier(tokens[end - 1]):
        end -= 2
    elif end > start + 1 and _identifier(tokens[end - 1]) and tokens[end - 2].text == ")":
        end -= 1
    if (end <= start + 2 or tokens[start].word not in {"STRFTIME", "SUBSTR", "SUBSTRING"}
            or tokens[start + 1].text != "(" or pairs[start + 1] != end - 1):
        return None
    references = []
    i = start + 2
    while i < end - 1:
        token = tokens[i]
        if token.kind in {"string", "number"} or token.text in {",", "+", "-"}:
            i += 1
            continue
        reference = _reference(tokens, i, end - 1)
        if reference is None:
            return None
        after, names = reference
        if after < end - 1 and tokens[after].text not in {","}:
            return None
        references.append((i, after, names))
        i = after
    if len(references) != 1:
        return None
    source_start, _, names = references[0]
    return end, source_start, names


async def _derived_text_stored_integer(chosen: dict[str, Any], catalog: _Catalog) -> dict[str, Any]:
    sql = chosen["sql"]
    tokens = _tokens(sql)
    scope = _outer_select(tokens)
    if (scope is None or scope.from_index is None
            or any(token.depth == scope.depth and token.word in _COMPOUND for token in tokens)):
        return chosen
    clauses = {token.word for token in tokens[scope.list_end:scope.end] if token.depth == scope.depth}
    # Integer ordering differs from text ordering (999 vs 1000), so with a LIMIT the cast may
    # change which rows are kept; only when an ORDER BY term reads the cast item itself.
    order_terms = _order_terms(tokens, scope) if {"ORDER", "LIMIT"} <= clauses else []
    pairs = _pairs(tokens)
    bindings = _bindings(tokens, scope, pairs)
    replacements = []
    for ordinal, (start, item_end) in enumerate(_select_items(tokens, scope), start=1):
        call = _derived_call(tokens, start, item_end, pairs)
        if call is None:
            continue
        end, source_start, names = call
        if order_terms and _orders_by_item(tokens, order_terms, ordinal, start, end, item_end):
            continue
        resolved = await catalog.resolve(names, bindings)
        if resolved is None:
            continue
        table, column = resolved
        source_end = _reference(tokens, source_start, end)[0]
        expression = sql[tokens[start].start:tokens[end - 1].end]
        offset = tokens[start].start
        probe_expression = _replace(expression, [(
            tokens[source_start].start - offset, tokens[source_end - 1].end - offset,
            _quote_identifier(column))])
        for stored_column in await catalog.table_columns(table):
            if stored_column.casefold() == column.casefold():
                continue
            stored = _quote_identifier(stored_column)
            checked = await catalog.execute(
                f"SELECT COUNT(*) AS n, SUM(CASE WHEN typeof({stored}) = 'integer' "
                f"AND {probe_expression} = CAST({stored} AS TEXT) THEN 1 ELSE 0 END) AS matched "
                f"FROM {table.sql} WHERE {_quote_identifier(column)} IS NOT NULL")
            if not _successful(checked) or _row_count(checked) != 1:
                continue
            values = _values(checked["rows"][0])
            if len(values) == 2 and values[0] >= 10 and values[1] == values[0]:
                replacements.append((tokens[start].start, tokens[end - 1].end,
                                     f"CAST({expression} AS INTEGER)"))
                break
    if not replacements:
        return chosen
    variant_sql = _replace(sql, replacements)
    variant = await catalog.execute(variant_sql)
    if _successful(variant) and _row_count(variant) == _row_count(chosen):
        return _updated(chosen, variant, variant_sql)
    return chosen


def _order_terms(tokens: list[_Token], scope: _Select) -> list[list[_Token]]:
    """Return the outer ORDER BY terms, each without its ASC/DESC/NULLS/COLLATE suffix."""
    level = [token for token in tokens[scope.list_end:scope.end] if token.depth == scope.depth or token.depth > scope.depth]
    words = [token.word if token.depth == scope.depth else "" for token in level]
    if "ORDER" not in words:
        return []
    i = words.index("ORDER") + 1
    if i < len(level) and words[i] == "BY":
        i += 1
    terms, current = [], []
    for token, word in zip(level[i:], words[i:]):
        if word in {"LIMIT", "OFFSET"}:
            break
        if token.depth == scope.depth and token.text == ",":
            terms.append(current)
            current = []
            continue
        current.append(token)
    terms.append(current)
    suffix = {"ASC", "DESC", "NULLS", "FIRST", "LAST"}
    cleaned = []
    for term in terms:
        if any(token.word == "COLLATE" for token in term):
            term = term[:next(k for k, token in enumerate(term) if token.word == "COLLATE")]
        while term and term[-1].word in suffix:
            term = term[:-1]
        if term:
            cleaned.append(term)
    return cleaned


def _orders_by_item(tokens: list[_Token], terms: list[list[_Token]], ordinal: int,
                    start: int, end: int, item_end: int) -> bool:
    """Whether an ORDER BY term reads this select item by ordinal, alias or expression."""
    alias = None
    if item_end - end == 2 and tokens[end].word == "AS" and _identifier(tokens[end + 1]):
        alias = tokens[end + 1].name.casefold()
    elif item_end - end == 1 and _identifier(tokens[end]):
        alias = tokens[end].name.casefold()
    expression = [token.text.casefold() for token in tokens[start:end]]
    for term in terms:
        texts = [token.text.casefold() for token in term]
        if len(term) == 1 and term[0].kind == "number" and term[0].text == str(ordinal):
            return True
        if alias is not None and any(_identifier(token) and token.name.casefold() == alias for token in term):
            return True
        if any(texts[k:k + len(expression)] == expression for k in range(len(texts) - len(expression) + 1)):
            return True
    return False


def _primary(tokens: list[_Token], start: int, end: int, pairs: dict[int, int]) -> int | None:
    i = start
    if i < end and tokens[i].text in {"+", "-"}:
        i += 1
    if i >= end:
        return None
    token = tokens[i]
    if token.text == "(":
        closing = pairs[i]
        return closing + 1 if closing < end else None
    if token.word == "CASE":
        nested = 1
        for j in range(i + 1, end):
            nested += (tokens[j].word == "CASE") - (tokens[j].word == "END")
            if nested == 0:
                return j + 1
        return None
    if token.kind in {"number", "string"}:
        return i + 1
    reference = _reference(tokens, i, end)
    if reference is None:
        return None
    after = reference[0]
    if after < end and tokens[after].text == "(":
        closing = pairs[after]
        return closing + 1 if closing < end else None
    return after


def _scale_tail(tokens: list[_Token], hundred: int) -> bool:
    return (tokens[hundred].kind == "number" and _HUNDRED.fullmatch(tokens[hundred].text) is not None
            and (hundred + 1 >= len(tokens)
                 or tokens[hundred + 1].text not in {"||", "->", "->>", "."}
                 and tokens[hundred + 1].word != "COLLATE"))


def _multiply_once(sql: str) -> str:
    tokens = _tokens(sql)
    pairs = _pairs(tokens)
    for i, token in enumerate(tokens):
        if token.text == "/":
            denominator_end = _primary(tokens, i + 1, len(tokens), pairs)
            if (denominator_end is not None and denominator_end + 1 < len(tokens)
                    and tokens[denominator_end].text == "*" and _scale_tail(tokens, denominator_end + 1)):
                denominator = sql[token.end:tokens[denominator_end - 1].end]
                gap = sql[tokens[denominator_end - 1].end:tokens[denominator_end].start]
                scale = sql[tokens[denominator_end].end:tokens[denominator_end + 1].end]
                return _replace(sql, [(token.start, tokens[denominator_end + 1].end,
                                       "*" + scale + gap + "/" + denominator)])
        if token.text != "(" or _function_open(tokens, i):
            continue
        if i > 0 and tokens[i - 1].text in {"/", "%", "||"}:
            continue
        closing = pairs[i]
        if (closing + 2 >= len(tokens) or tokens[closing + 1].text != "*"
                or not _scale_tail(tokens, closing + 2)):
            continue
        cursor = _primary(tokens, i + 1, closing, pairs)
        last_division = None
        while cursor is not None and cursor < closing and tokens[cursor].text in {"*", "/"}:
            last_division = cursor if tokens[cursor].text == "/" else None
            cursor = _primary(tokens, cursor + 1, closing, pairs)
        if cursor != closing or last_division is None:
            continue
        division = tokens[last_division]
        scale = sql[tokens[closing + 1].end:tokens[closing + 2].end]
        outside_gap = sql[tokens[closing].end:tokens[closing + 1].start]
        replacement = (sql[token.start:division.start] + "*" + scale + outside_gap
                       + "/" + sql[division.end:tokens[closing].end])
        return _replace(sql, [(token.start, tokens[closing + 2].end, replacement)])
    return sql


def _ordered_near(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if (_row_count(before) != _row_count(after)
            or len(before.get("columns", [])) != len(after.get("columns", []))):
        return False
    for old_row, new_row in zip(before["rows"], after["rows"]):
        old_values, new_values = _values(old_row), _values(new_row)
        if len(old_values) != len(new_values):
            return False
        for old, new in zip(old_values, new_values):
            if old == new:
                continue
            if (not isinstance(old, (int, float)) or isinstance(old, bool)
                    or not isinstance(new, (int, float)) or isinstance(new, bool)
                    or not math.isfinite(old) or not math.isfinite(new)
                    or abs(old - new) > 1e-9 * max(abs(old), abs(new))):
                return False
    return True


async def _multiply_first_percentage(
    chosen: dict[str, Any], execute: Callable[[str], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    sql = chosen["sql"]
    variant_sql = sql
    for _ in range(10):
        rewritten = _multiply_once(variant_sql)
        if rewritten == variant_sql:
            break
        variant_sql = rewritten
    if variant_sql == sql:
        return chosen
    variant = await execute(variant_sql)
    if _successful(variant) and _ordered_near(chosen, variant):
        return _updated(chosen, variant, variant_sql)
    return chosen


def _hint_aggregate_fidelity(
    question: str, chosen: dict[str, Any], candidates: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], int | None]:
    """Select an otherwise identical candidate preserving a directly hinted AVG or SUM.

    Only one direct-column aggregate in a numeric scalar projection may change.
    Conflicting operators, nested SELECT scopes and any other SQL-token change abstain.
    """
    if _scalar(chosen) is None:
        return chosen, None
    hint = question.partition(HINT_SEPARATOR)[2]
    requested: dict[str, set[str]] = {}
    for match in _HINT_AGGREGATE.finditer(hint):
        try:
            argument = _tokens(match[2])
        except ValueError:
            # Prose and malformed pseudo-SQL establish no direct-column formula.
            continue
        reference = _reference(argument, 0, len(argument))
        if reference is not None and reference[0] == len(argument):
            requested.setdefault(reference[1][-1].casefold(), set()).add(match[1].upper())
    if not requested:
        return chosen, None
    tokens = _tokens(chosen["sql"])
    scopes = _selects(tokens)
    if len(scopes) != 1 or scopes[0].depth != 0:
        return chosen, None
    scope, pairs = scopes[0], _pairs(tokens)
    replacements = []
    for i in range(scope.start + 1, scope.list_end - 2):
        token = tokens[i]
        if token.word not in {"AVG", "SUM"} or tokens[i + 1].text != "(":
            continue
        closing = pairs[i + 1]
        reference = _reference(tokens, i + 2, closing)
        if reference is None or reference[0] != closing:
            continue
        operators = requested.get(reference[1][-1].casefold(), set())
        if len(operators) == 1 and token.word not in operators:
            replacements.append((token.start, token.end, next(iter(operators))))
    if len(replacements) != 1:
        return chosen, None
    variant_sql = _replace(chosen["sql"], replacements)
    variant = [(token.kind, token.word if token.kind == "word" else token.text)
               for token in _tokens(variant_sql) if token.text != ";"]
    for slot, candidate in enumerate(candidates, 1):
        if not _successful(candidate) or not isinstance(candidate.get("sql"), str):
            continue
        candidate_tokens = [(token.kind, token.word if token.kind == "word" else token.text)
                            for token in _tokens(candidate["sql"]) if token.text != ";"]
        if candidate_tokens == variant:
            return _updated(chosen, candidate, variant_sql), slot
    return chosen, None


async def apply_submission_gate(
    question: str,
    chosen: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    execute: Callable[[str], Awaitable[dict[str, Any]]],
    result_sig: Callable[[dict[str, Any]], Hashable],
    *, left_inner_min_support_margin: int = 1, hint_aggregate_fidelity: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the ordered submission rules without changing cached records.

    Args:
        question (str): Question text, optionally followed by ``\n[Hint]: ``.
        chosen (dict[str, Any]): Current executed final, including SQL and full rows.
        candidates (Sequence[dict[str, Any]]): Executed candidate pool in slot order.
        execute (Callable[[str], Awaitable[dict[str, Any]]]): Read-only base-database
            executor under the normal final-SQL budget. Returns full rows, columns,
            row_count, and an optional error. SQL need not be included in its result.
        result_sig (Callable[[dict[str, Any]], Hashable]): Official execution-result
            signature, ignoring row order and duplicates while retaining cell order.
        left_inner_min_support_margin (int): Required excess of matching valid
            INNER candidates over successful chosen-result candidates. At least 1.
        hint_aggregate_fidelity (bool): Enable exact-candidate SUM/AVG selection
            from a direct current-hint formula, before the five base rules.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: Shallow copy of the final and version-1
            debug record with the gate's implementation version. Changed rows and
            columns are detached from executed results; retained transcripts are shared.
            A firing records its rule, prior SQL, new SQL or 1-based candidate slot,
            and integer row counts. A rule exception preserves the current final;
            later rules can still run, and an optional errors list records its rule
            and exception type. This function does not attach the debug record.

    Raises:
        ValueError: The support margin is not a positive integer, or a configured
            rule name has no declared implementation.
    """
    if (not isinstance(left_inner_min_support_margin, int) or isinstance(left_inner_min_support_margin, bool)
            or left_inner_min_support_margin < 1):
        raise ValueError("left_inner_min_support_margin must be a positive integer")
    current = dict(chosen)
    version = HINT_AGGREGATE_VERSION if hint_aggregate_fidelity else IMPLEMENTATION_VERSION
    debug: dict[str, Any] = {"v": 1, "implementation_version": version,
                             "pre_gate_sql": chosen.get("sql"), "fired": []}
    catalog = _Catalog(execute)
    rules = {
        "hint_aggregate_fidelity": lambda: _hint_aggregate_fidelity(question, current, candidates),
        "percent_scale": lambda: _percent_scale(question, current, candidates, result_sig),
        "left_join_to_inner": lambda: _left_join_to_inner(
            current, candidates, execute, result_sig, min_support_margin=left_inner_min_support_margin),
        "date_granularity": lambda: _date_granularity(current, catalog),
        "derived_text_stored_integer": lambda: _derived_text_stored_integer(current, catalog),
        "multiply_first_percentage": lambda: _multiply_first_percentage(current, execute),
    }
    rule_names = (("hint_aggregate_fidelity",) + RULE_NAMES if hint_aggregate_fidelity else RULE_NAMES)
    for rule in rule_names:
        if rule not in rules:
            raise ValueError(f"Unknown submission-gate rule: {rule!r}")
    if not isinstance(current.get("sql"), str) or not _successful(current):
        return current, debug
    for rule in rule_names:
        before = current
        slot = None
        try:
            outcome = rules[rule]()
            if isinstance(outcome, tuple):
                after, slot = outcome
            else:
                after = await outcome
            if after is before or result_sig(after) == result_sig(before):
                continue
            firing = {"rule": rule, "sql_before": before["sql"],
                      "rows_before": _row_count(before), "rows_after": _row_count(after)}
            firing.update({"slot": slot} if slot is not None else {"sql_after": after["sql"]})
            debug["fired"].append(firing)
            current = after
        except Exception as exc:
            # Submission is already valid; a failed deterministic check cannot replace it.
            debug.setdefault("errors", []).append({"rule": rule, "type": type(exc).__name__})
    return current, debug
