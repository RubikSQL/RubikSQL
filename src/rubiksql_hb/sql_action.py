"""The SQL-bearing tool names.

Every capability, SQL included, is a native provider tool call (``sql_native_v2``; see
docs/architecture.md, *Model/runtime protocol*). The candidate and judge loops tell these tools
apart from the knowledge and schema tools to apply the SQL-action semantics (salvage, the
final-check bounce, the accepting submit, the ``sql_actions`` record), and the judge's execution
identity lists them as ``sql_actions``. The plain-text action protocol that once carried them
(7f02fdc to sql_native_v2) is retired with its parser. The r0 export reads this file as it stood at
commit 9140477 (its SQL-results note), never the working tree.
"""

import os
import re

from .question import HINT_SEPARATOR

SQL_ACTIONS = frozenset({
    "run_sql",
    "run_sql_base",
    "submit_final_sql",
    "save_sql_as_table",
    "audit_measure_grain",
})

# Shared with the reviewer's rank-request gate. Hints and stored-rank wording do
# not turn an attribute lookup into an imperative request.
_RANK_REQUEST_VERB = re.compile(
    r"(?:^\s*|\bplease\s+|[,;.!?]\s*|\b(?:can|could|would)\s+you\s+)"
    r"rank\s+(?!of\b)[^.!?]*\bby\b", re.IGNORECASE)


def rank_request(question: str | None) -> bool:
    """Return whether the question itself requests an imperative ranking."""
    text = str(question or "").partition(HINT_SEPARATOR)[0]
    return bool(_RANK_REQUEST_VERB.search(text))


def rank_publication_enabled() -> bool:
    """Whether the adopted rank publication contract is active (default on)."""
    return os.environ.get("RUBIKSQL_RANK_PUBLICATION", "1") == "1"


def missing_rank_output(question: str, sql: str, columns: list[str]) -> bool:
    """Check an executed answer's projected ordinal for an imperative ranking.

    Follow window outputs through CTEs and derived tables. A window used only
    for filtering, sorting, or a scalar subquery is not a projected ordinal.
    Stored ordinal references are also valid; merely naming an expression
    ``rank`` does not make it an ordinal. Unreadable SQL abstains.

    Args:
        question: Question with its optional canonical hint block.
        sql: Successfully executed final SQL.
        columns: Executed output column names, including wildcard expansion.

    Returns:
        Whether the answer visibly omits the ordinal or its accompanying
        entity/ordering measure. This is a corrective nudge, never a rewrite.
    """
    return rank_output_contract(question, sql, columns)["status"] == "missing"


def rank_output_contract(question: str, sql: str, columns: list[str]) -> dict:
    """Describe an executed projection's imperative-rank contract without rewriting SQL.

    Args:
        question: Canonical question, optionally including evidence.
        sql: Successfully executed answer SQL.
        columns: Full executed output columns.

    Returns:
        dict: A status (not_requested, missing, unknown or verified) and reason.
            Only verified answers may displace a noncompliant fallback. Verification
            requires an entity, a projected measure equal to the window's ordering
            expression, and lexical support for that measure in the request/evidence.
            Stored ordinals and unsupported SQL remain unknown rather than proof.
    """
    def result(status, reason):
        return {"v": 1, "status": status, "reason": reason}

    if not rank_request(question):
        return result("not_requested", "no imperative rank request")
    if len(columns) < 3:
        return result("missing", "entity, measure and ordinal are not all projected")
    from .submission_gate import _identifier, _outer_select, _pairs, _reference, _selects, _tokens

    try:
        tokens = _tokens(sql)
        pairs = _pairs(tokens)
        outer = _outer_select(tokens)
        if outer is None:
            return result("unknown", "unreadable outer projection")
        if any(t.depth == 0 and t.word in {"UNION", "EXCEPT", "INTERSECT"} for t in tokens):
            return result("unknown", "compound output lineage is not established")
        scopes = _selects(tokens)
        ctes = {}
        for i, token in enumerate(tokens):
            if token.word == "AS" and i and i + 1 in pairs and _identifier(tokens[i - 1]):
                scope = next((s for s in scopes if s.start == i + 2), None)
                if scope is not None:
                    ctes[tokens[i - 1].name.casefold()] = scope

        def projection(scope, active):
            if scope.start in active:
                raise ValueError("recursive projection")
            active = active | {scope.start}
            sources = {}
            if scope.from_index is not None:
                end = next((j for j in range(scope.from_index + 1, scope.end)
                            if tokens[j].depth == scope.depth and tokens[j].word in
                            {"WHERE", "GROUP", "ORDER", "LIMIT", "HAVING", "WINDOW"}), scope.end)
                i, needed = scope.from_index + 1, True
                while i < end:
                    t = tokens[i]
                    if t.depth != scope.depth:
                        i += 1
                        continue
                    if t.word == "JOIN" or t.text == ",":
                        needed = True
                        i += 1
                        continue
                    if not needed:
                        i += 1
                        continue
                    if t.text == "(":
                        nested = next((s for s in scopes if s.start == i + 1), None)
                        if nested is None:
                            raise ValueError("unknown derived source")
                        source = projection(nested, active)
                        i, alias = pairs[i] + 1, ""
                    else:
                        ref = _reference(tokens, i, end)
                        if ref is None:
                            raise ValueError("unknown source")
                        i, names = ref
                        alias = names[-1].casefold()
                        source = projection(ctes[alias], active) if alias in ctes else alias
                    if i < end and tokens[i].word == "AS":
                        i += 1
                    if i < end and _identifier(tokens[i]) and tokens[i].word not in {
                            "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "ON", "USING"}:
                        alias = tokens[i].name.casefold()
                        i += 1
                    sources[alias] = source
                    needed = False

            def reference(names):
                name = names[-1].casefold()
                eligible = ([sources[names[-2].casefold()]] if len(names) > 1
                            and names[-2].casefold() in sources else list(sources.values()))
                values = [source[name] if isinstance(source, dict) else
                          (("column", source, name), None) for source in eligible
                          if not isinstance(source, dict) or name in source]
                if len(values) != 1:
                    raise ValueError("ambiguous column lineage")
                return values[0]

            def expression(start, end):
                ref = _reference(tokens, start, end)
                if ref and ref[0] == end:
                    return reference(ref[1])
                if tokens[start].word in {"RANK", "DENSE_RANK", "ROW_NUMBER"}:
                    if start + 4 >= end or tokens[start + 3].word != "OVER" or tokens[start + 4].text != "(":
                        raise ValueError("unsupported window")
                    lo, hi = start + 5, pairs[start + 4]
                    if any(tokens[j].word == "PARTITION" for j in range(lo, hi)):
                        raise ValueError("partition scope is not established")
                    order = next((j + 2 for j in range(lo, hi - 1)
                                  if tokens[j].word == "ORDER" and tokens[j + 1].word == "BY"
                                  and tokens[j].depth == tokens[lo].depth), None)
                    if order is None:
                        raise ValueError("window has no ordering measure")
                    stop = hi
                    if tokens[stop - 1].word in {"ASC", "DESC"}:
                        stop -= 1
                    if any(tokens[j].text == "," and tokens[j].depth == tokens[order].depth for j in range(order, stop)):
                        raise ValueError("multiple ordering expressions")
                    measure, _ = expression(order, stop)
                    return ("ordinal", tokens[start].word, measure), measure
                parts = []
                i = start
                while i < end:
                    t = tokens[i]
                    if t.word == "SELECT":
                        return ("scalar-subquery",), None
                    ref = _reference(tokens, i, end) if _identifier(t) else None
                    if ref and (ref[0] == end or tokens[ref[0]].text != "(") and t.word not in {"DISTINCT", "AS"}:
                        value, ordinal = reference(ref[1])
                        if ordinal is not None:
                            raise ValueError("transformed ordinal")
                        parts.append(value)
                        i = ref[0]
                    else:
                        parts.append(t.text.casefold())
                        i += 1
                return tuple(parts), None

            outputs = {}
            cuts = [scope.start + 1] + [j + 1 for j in range(scope.start + 1, scope.list_end)
                                        if tokens[j].depth == scope.depth and tokens[j].text == ","]
            ends = [j - 1 for j in cuts[1:]] + [scope.list_end]
            for start, end in zip(cuts, ends):
                if tokens[start].word in {"DISTINCT", "ALL"}:
                    start += 1
                name = tokens[end - 1].name.casefold()
                aliased = False
                if end - start > 1 and tokens[end - 2].word == "AS":
                    aliased = True
                    end -= 2
                elif end - start > 1 and _identifier(tokens[end - 1]) and tokens[end - 2].text != ".":
                    aliased = True
                    end -= 1
                ref = _reference(tokens, start, end)
                if not aliased and not (ref and ref[0] == end):
                    # SQLite names an anonymous expression by its source text.
                    # Using its final ')' would merge distinct sibling functions.
                    name = sql[tokens[start].start:tokens[end - 1].end].casefold()
                if tokens[end - 1].text == "*":
                    selected = sources.values() if end - start == 1 else [sources[tokens[start].name.casefold()]]
                    for source in selected:
                        if not isinstance(source, dict):
                            raise ValueError("base wildcard lineage is unknown")
                        outputs.update(source)
                else:
                    outputs[name] = expression(start, end)
            return outputs

        outputs = list(projection(outer, set()).values())
        ordinals = [order for _, order in outputs if order is not None]
        if not ordinals:
            # A stored ordinal can be valid, but its name alone is not semantic proof.
            stored = any(expr and expr[0] == "column" and re.fullmatch(
                r"rank|ranking|ordinal|position|rn|rnk|row_number|dense_rank", expr[-1], re.I) for expr, _ in outputs)
            return result("unknown" if stored else "missing", "no traced projected window ordinal")
        measures = [expr for expr, order in outputs if order is None]
        if not any(order in measures for order in ordinals):
            return result("missing", "window ordering measure is not projected")
        if len(columns) != 3 or len(outputs) != 3 or len(ordinals) != 1:
            return result("unknown", "additional output roles are not established")
        measure = ordinals[0]
        entity = next((expr for expr in measures if expr != measure), None)
        if not entity or entity[0] != "column":
            return result("unknown", "entity lineage is not a direct column")
        raw_question = question.partition(HINT_SEPARATOR)[0]
        request = _RANK_REQUEST_VERB.search(raw_question)
        text = re.split(r"[,;.!?]", raw_question[request.end():], maxsplit=1)[0].casefold()
        direct_measure = measure[0] == "column"
        aggregate = (len(measure) == 4 and measure[0] in {"count", "sum", "avg", "min", "max"}
                     and measure[1] == "(" and measure[-1] == ")")
        if not direct_measure and not aggregate:
            return result("unknown", "transformed ordering measure is not established")
        cues = {"count": r"\b(?:count|number|popularity|frequency|common)\b", "avg": r"\b(?:average|mean)\b",
                "sum": r"\b(?:sum|total)\b", "min": r"\b(?:minimum|lowest|least)\b", "max": r"\b(?:maximum|highest|most)\b"}
        if aggregate and not re.search(cues[measure[0]], text):
            return result("unknown", "aggregate operation lacks request support")
        def metric_words(expr):
            if isinstance(expr, tuple) and expr and expr[0] == "column":
                return set(re.findall(r"[a-z]+", expr[-1])) - {"id", "value", "total", "number", "num"}
            return set().union(*(metric_words(item) for item in expr if isinstance(item, tuple))) if isinstance(expr, tuple) else set()
        words = metric_words(measure)
        mentioned = any(re.search(r"\b" + re.escape(word) + r"\b", text) for word in words)
        counted = "count" in measure and bool(re.search(r"\b(?:count|number|popularity|frequency|common)\b", text))
        if not mentioned and not counted:
            return result("unknown", "ordering measure lacks request or evidence support")
        return result("verified", "projected entity, supported measure and its window ordinal")
    except (ValueError, IndexError, KeyError, TypeError):
        return result("unknown", "expression lineage cannot be established")


def publish_rank_result(question: str, chosen: dict, alternatives: list[dict], *, enabled: bool = False) -> dict:
    """Record compliance and optionally retain a verified alternative at publication.

    Args:
        question: Canonical request and evidence.
        chosen: Existing executed decision, including any synthesis and review.
        alternatives: Same-pool executed results in existing preference order.
        enabled: Enable replacement; recording itself requires no model call.

    Returns:
        dict: The original decision, with a compliance receipt for rank requests;
            a definitely incomplete decision may use the first verified alternative.
            Unknown lineage, failed executions and non-imperatives never cause replacement.
    """
    from .submission_gate import _successful

    if not rank_request(question):
        return chosen
    def contract(item):
        if not item.get("sql") or not _successful(item):
            return {"v": 1, "status": "unknown", "reason": "no complete successful execution"}
        return rank_output_contract(question, item["sql"], item.get("columns", []))
    before = contract(chosen)
    receipts = [contract(item) for item in alternatives]
    index = next((i for i, receipt in enumerate(receipts) if receipt["status"] == "verified"), None)
    fired = enabled and before["status"] == "missing" and index is not None
    kept = chosen
    if fired:
        alternate = alternatives[index]
        kept = {**chosen, **{name: alternate.get(name) for name in ("sql", "rows", "columns", "error")},
                "row_count": alternate.get("row_count", len(alternate["rows"])),
                "source": "rank_contract",
                "judge_debug": alternate.get("judge_debug") or
                    {"mode": "fallback_rank_contract", "previous": chosen.get("judge_debug")},
                "reviewer_debug": alternate.get("reviewer_debug") or
                    {"active": False, "skipped": "rank_contract", "previous": chosen.get("reviewer_debug")}}
    previous = chosen.get("rank_contract") or {}
    prior_fired = previous.get("fired_any", previous.get("fired", False))
    return {**kept, "rank_contract": {"v": 1, "enabled": enabled, "fired": fired, "fired_any": fired or prior_fired,
                                     "before": before, "alternatives": receipts,
                                     "selected_alternative": index if fired else None,
                                     "pre_contract_sql": chosen.get("sql"), "after": contract(kept),
                                     **({"prior": previous} if prior_fired else {})}}
