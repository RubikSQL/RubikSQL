"""Narrow conditional entity percentages, shared by post-review voting and replay."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Hashable
from dataclasses import dataclass
from typing import Any

from .selection_normalization import _arity, _complete
from .submission_gate import _scalar, _updated


IMPLEMENTATION_VERSION = "entity-percentage-count-1"

_MEASURE_WORDS = frozenset(
    "amount cost price value rate ratio total sum average mean consumption "
    "increase decrease growth change difference increment".split()
)
_MEASURE_WORDS = _MEASURE_WORDS | frozenset(word + "s" for word in _MEASURE_WORDS)
_FOCUS = re.compile(
    r"\bpercent(?:age)?\s+of\s+(?:the\s+)?"
    r"(?P<focus>[a-z]+(?:\s+[a-z]+){0,3}?)\s+(?:that|who|whose|which|with)\b",
    re.I,
)
_LEX = re.compile(
    r"(?P<space>\s+)|(?P<comment>--[^\n]*|/\*[\s\S]*?\*/)"
    r"|(?P<string>'(?:''|[^'])*')"
    r'|(?P<quoted>"(?:""|[^"])*"|`(?:``|[^`])*`|\[(?:\]\]|[^\]])*\])'
    r"|(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<word>[A-Za-z_][A-Za-z_0-9$]*)"
    r"|(?P<symbol><=|>=|<>|!=|==|[=<>+*/%(),.;-])"
)


@dataclass(frozen=True)
class _Token:
    text: str
    kind: str
    start: int
    end: int

    @property
    def word(self) -> str:
        return self.text.upper() if self.kind == "word" else ""


def _tokens(sql: str) -> list[_Token]:
    result = []
    cursor = 0
    while cursor < len(sql):
        match = _LEX.match(sql, cursor)
        if match is None:
            raise ValueError("unsupported_token")
        kind = match.lastgroup
        if kind not in {"space", "comment"}:
            result.append(_Token(match.group(), kind, match.start(), match.end()))
        cursor = match.end()
    return result


def _tops(tokens: list[_Token]) -> list[int]:
    result, depth, cases = [], 0, 0
    for i, token in enumerate(tokens):
        if token.text == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced_parentheses")
        if depth == 0:
            if token.word == "CASE":
                cases += 1
            if cases == 0:
                result.append(i)
            if token.word == "END":
                cases -= 1
                if cases < 0:
                    raise ValueError("unbalanced_case")
        if token.text == "(":
            depth += 1
    if depth or cases:
        raise ValueError("unbalanced_expression")
    return result


def _unparen(tokens: list[_Token]) -> list[_Token]:
    while tokens and tokens[0].text == "(" and tokens[-1].text == ")":
        depth = 0
        for i, token in enumerate(tokens):
            depth += (token.text == "(") - (token.text == ")")
            if depth == 0:
                break
        if i != len(tokens) - 1:
            break
        tokens = tokens[1:-1]
    return tokens


def _number(tokens: list[_Token], value: int) -> bool:
    tokens = _unparen(tokens)
    return len(tokens) == 1 and tokens[0].kind == "number" and float(tokens[0].text) == value


def _column(tokens: list[_Token]) -> tuple[str, ...] | None:
    tokens = _unparen(tokens)
    if not tokens or len(tokens) % 2 != 1:
        return None
    if any(token.text != "." for token in tokens[1::2]):
        return None
    parts = []
    for token in tokens[::2]:
        if token.kind not in {"word", "quoted"} or token.word in {"NULL", "TRUE", "FALSE"}:
            return None
        text = token.text
        if token.kind == "quoted":
            delimiter = "]" if text[0] == "[" else text[0]
            text = text[1:-1].replace(delimiter * 2, delimiter)
        parts.append(text.casefold())
    return tuple(parts)


def _operand(tokens: list[_Token]) -> bool:
    tokens = _unparen(tokens)
    if _column(tokens) is not None:
        return True
    if len(tokens) == 2 and tokens[0].text in {"+", "-"}:
        tokens = tokens[1:]
    return len(tokens) == 1 and (
        tokens[0].kind in {"number", "string"} or tokens[0].word in {"NULL", "TRUE", "FALSE"}
    )


def _predicate(tokens: list[_Token]) -> bool:
    tokens = _unparen(tokens)
    if not tokens:
        return False
    tops = _tops(tokens)
    for word in ("OR", "AND"):
        cuts = [i for i in tops if tokens[i].word == word]
        if cuts:
            boundaries = [-1, *cuts, len(tokens)]
            return all(_predicate(tokens[a + 1:b]) for a, b in zip(boundaries, boundaries[1:]))
    if tokens[0].word == "NOT":
        return _predicate(tokens[1:])
    comparisons = [i for i in tops if tokens[i].text in {"=", "==", "!=", "<>", "<", ">", "<=", ">="}
                   or tokens[i].word == "IS"]
    if len(comparisons) != 1:
        return False
    i = comparisons[0]
    right = tokens[i + 1:]
    if tokens[i].word == "IS" and right and right[0].word == "NOT":
        right = right[1:]
    return _operand(tokens[:i]) and _operand(right)


def _call(tokens: list[_Token], name: str) -> list[_Token] | None:
    tokens = _unparen(tokens)
    if len(tokens) < 3 or tokens[0].word != name or tokens[1].text != "(" or tokens[-1].text != ")":
        return None
    if 1 not in _tops(tokens):
        return None
    return tokens[2:-1]


def _factors(tokens: list[_Token]) -> tuple[list[list[_Token]], list[list[_Token]]]:
    tokens = _unparen(tokens)
    cuts = [i for i in _tops(tokens) if tokens[i].text in {"*", "/"}]
    if not cuts:
        return [tokens], []
    i = cuts[-1]
    left_num, left_den = _factors(tokens[:i])
    right_num, right_den = _factors(tokens[i + 1:])
    if tokens[i].text == "*":
        return left_num + right_num, left_den + right_den
    return left_num + right_den, left_den + right_num


def _numerator(tokens: list[_Token], denominator: tuple[str, ...]) -> list[_Token] | None:
    tokens = _unparen(tokens)
    cast = _call(tokens, "CAST")
    if cast is not None:
        places = [i for i in _tops(cast) if cast[i].word == "AS"]
        if len(places) != 1:
            return None
        i = places[0]
        if len(cast) != i + 2 or cast[-1].word not in {"REAL", "FLOAT", "DOUBLE", "NUMERIC"}:
            return None
        tokens = _unparen(cast[:i])
    body = _call(tokens, "SUM")
    if body is None:
        return None
    body = _unparen(body)
    if _predicate(body):
        return body
    if not body or body[0].word != "CASE" or body[-1].word != "END":
        return None
    places = {word: [i for i, token in enumerate(body) if token.word == word]
              for word in ("CASE", "WHEN", "THEN", "ELSE", "END")}
    if any(len(indices) != 1 for indices in places.values()) or places["WHEN"][0] != 1:
        return None
    w, t, e, end = (places[word][0] for word in ("WHEN", "THEN", "ELSE", "END"))
    if not w < t < e < end:
        return None
    predicate, payload, otherwise = body[w + 1:t], body[t + 1:e], body[e + 1:end]
    if not _predicate(predicate) or not _number(otherwise, 0):
        return None
    if not _number(payload, 1) and _column(payload) != denominator:
        return None
    return predicate


def build_variant(question: str, sql: str) -> tuple[str | None, str]:
    """Build the frozen, narrow row-percentage variant without executing SQL.

    Eligibility uses the question before its appended hint and SQL structure only.
    It preserves predicate text, output aliases and the full source/filter tail.
    This benchmark convention does not establish SQL equivalence or entity grain.

    Args:
        question (str): Question text, including any appended ``\n[Hint]:`` evidence.
        sql (str): Executed final query to inspect.

    Returns:
        tuple[str | None, str]: Variant and ``eligible``, or no variant and the
            frozen lexical or structural abstention reason. Unsupported syntax
            abstains; numeric result guards belong to the execution wrapper.
    """
    wording = question.split("\n[Hint]:", 1)[0]
    focuses = [match.group("focus").casefold().split() for match in _FOCUS.finditer(wording)]
    if len(focuses) != 1:
        return None, "no_unique_percentage_focus"
    words = focuses[0]
    if not (words[-1].endswith("s") and not words[-1].endswith("ss")
            and not (set(words) & _MEASURE_WORDS)):
        return None, "no_entity_percentage_focus"
    try:
        tokens = _tokens(sql)
        if tokens and tokens[-1].text == ";":
            tokens = tokens[:-1]
        if not tokens or tokens[0].word != "SELECT" or any(token.text == ";" for token in tokens):
            return None, "not_one_select"
        if sum(token.word == "SELECT" for token in tokens) != 1:
            return None, "nested_select"
        forbidden = {"GROUP", "HAVING", "UNION", "INTERSECT", "EXCEPT", "WINDOW", "OVER", "FILTER", "LIMIT", "OFFSET"}
        if any(token.word in forbidden for token in tokens):
            return None, "non_scalar_scope"
        tops = _tops(tokens)
        froms = [i for i in tops if tokens[i].word == "FROM"]
        if len(froms) != 1:
            return None, "uncertain_from"
        end = froms[0]
        projection = tokens[1:end]
        ptop = _tops(projection)
        if any(projection[i].text == "," for i in ptop):
            return None, "multiple_projection"
        expression = projection
        aliases = [i for i in ptop if projection[i].word == "AS"]
        if aliases:
            if len(aliases) != 1 or aliases[0] != len(projection) - 2 or projection[-1].kind not in {"word", "quoted"}:
                return None, "uncertain_alias"
            expression = projection[:aliases[0]]
        elif len(projection) > 1 and projection[-1].kind in {"word", "quoted"}:
            expression = projection[:-1]
        if sum(token.text == "/" for token in expression) != 1:
            return None, "not_one_division"
        numerator, denominator = _factors(expression)
        if len(numerator) != 2 or len(denominator) != 1:
            return None, "not_scaled_ratio"
        hundreds = [i for i, factor in enumerate(numerator) if _number(factor, 100)]
        if len(hundreds) != 1:
            return None, "not_percent_scale"
        denominator_body = _call(denominator[0], "SUM")
        column = _column(denominator_body) if denominator_body is not None else None
        if column is None:
            return None, "denominator_not_sum_column"
        predicate = _numerator(numerator[1 - hundreds[0]], column)
        if predicate is None:
            return None, "numerator_not_conditional_sum"
        predicate_text = sql[predicate[0].start:predicate[-1].end]
        canonical = f"CAST(SUM(CASE WHEN {predicate_text} THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*)"
        variant = sql[:expression[0].start] + canonical + sql[expression[-1].end:]
        return variant, "eligible"
    except (IndexError, ValueError, OverflowError):
        return None, "unsupported_syntax"


async def apply_entity_percentage_count(
    question: str,
    chosen: dict[str, Any],
    execute: Callable[[str], Awaitable[dict[str, Any]]],
    result_sig: Callable[[dict[str, Any]], Hashable],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Try a conditional row percentage after selection normalization.

    Both executions must be successful, complete, finite numeric 1×1 results.
    The variant must be in [0,100] and change the official result signature.
    Candidates have no veto. This is a deterministic benchmark convention.

    Args:
        question (str): Question text, including any appended hint/evidence.
        chosen (dict[str, Any]): Executed final containing SQL and full results.
        execute (Callable[[str], Awaitable[dict[str, Any]]]): Read-only database executor.
        result_sig (Callable[[dict[str, Any]], Hashable]): Official result signature.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: Updated final with detached result
            rows/columns if accepted, otherwise ``chosen`` itself, and an unattached
            version-1 decision record. The record includes input SQL, acceptance,
            skip reason, variant SQL when built, and feature/error type on exceptions.
            Every exception retains the original final.
    """
    debug: dict[str, Any] = {"v": 1, "implementation_version": IMPLEMENTATION_VERSION,
                             "pre_hook_sql": chosen.get("sql"), "fired": False, "skipped": None}
    try:
        sql = chosen.get("sql")
        if not isinstance(sql, str) or not _complete(chosen):
            debug["skipped"] = "final_failed"
        elif _arity(chosen) != 1 or _scalar(chosen) is None:
            debug["skipped"] = "final_not_numeric_scalar"
        else:
            variant_sql, reason = build_variant(question, sql)
            if variant_sql is None:
                debug["skipped"] = reason
            else:
                debug["variant_sql"] = variant_sql
                variant = await execute(variant_sql)
                if not _complete(variant):
                    debug["skipped"] = "variant_failed"
                elif _arity(variant) != 1 or (value := _scalar(variant)) is None:
                    debug["skipped"] = "variant_not_numeric_scalar"
                elif not 0 <= value <= 100:
                    debug["skipped"] = "variant_out_of_range"
                elif result_sig(variant) == result_sig(chosen):
                    debug["skipped"] = "variant_matches_final"
                else:
                    updated = _updated(chosen, variant, variant_sql)
                    debug["fired"] = True
                    return updated, debug
    except Exception as exc:
        debug["skipped"] = "error"
        debug["errors"] = [{"feature": "entity_percentage_count", "type": type(exc).__name__}]
    return chosen, debug
