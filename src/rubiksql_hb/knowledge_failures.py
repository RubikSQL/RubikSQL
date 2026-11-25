"""Exact knowledge-failure wire contracts and retained-transcript detection."""
from collections.abc import Mapping
import json
import re


RULES_RETRIEVAL_FAILED_ERROR = (
    "get_rules retrieval FAILED (knowledge backend error) — zero rules were returned "
    "because the rule store could not be searched, NOT because no rules matched"
)
KNOWLEDGE_TOOLS = frozenset({
    "get_rules", "get_examples", "expand_examples", "search_values", "search_columns",
    "query_knowledge", "query_spec", "match_rule_triggers", "get_db_info", "get_tab_info",
    "get_col_info", "expand_synonyms", "save_rule", "update_rule", "save_example", "save_synonym",
})


def _failure_text(tool: str, text: object) -> bool:
    if not isinstance(tool, str) or not isinstance(text, str):
        return False
    # Retained assistant calls keep gateway-added namespaces even though live
    # dispatch strips these separators before executing the canonical tool.
    if tool not in KNOWLEDGE_TOOLS:
        for separator in (":", ".", "__"):
            tail = tool.rsplit(separator, 1)[-1]
            if tail in KNOWLEDGE_TOOLS:
                tool = tail
                break
    if tool not in KNOWLEDGE_TOOLS:
        return False
    return (
        tool == "get_rules" and text == RULES_RETRIEVAL_FAILED_ERROR
        # Exact legacy OpenAI SDK payloads emitted by the tools' broad except
        # branches. Never classify arbitrary "failed:" text as an outage.
        or text in {f"{tool} failed: Connection error.", f"{tool} failed: Request timed out."}
    )


def degraded_knowledge_result(tool: str, result: object) -> bool:
    """Recognise only error rows emitted by a knowledge tool's failure boundary."""
    rows = result if isinstance(result, list) else [result]
    return any(_failure_text(tool, row.get("error") if isinstance(row, Mapping) else row) for row in rows)


def _degraded_content(tool: str, content: object) -> bool:
    if not isinstance(content, str):
        return degraded_knowledge_result(tool, content)
    try:
        if degraded_knowledge_result(tool, json.loads(content)):
            return True
    except (ValueError, TypeError):
        pass
    # model_render emits either one ERROR line or a final error section. Its
    # collision-free SQL fences can contain arbitrary section-looking SQL text.
    if content.startswith("ERROR: "):
        return _failure_text(tool, content.removeprefix("ERROR: "))
    lines = content.splitlines()
    fence = None
    for index, line in enumerate(lines):
        if fence is not None:
            if line == fence:
                fence = None
            continue
        if opening := re.fullmatch(r"(`{3,}|~{3,})sql", line):
            fence = opening[1]
            continue
        if line not in {"Tool errors", "Knowledge errors"} or (index and lines[index - 1] != ""):
            continue
        # Error records are contiguous and end the renderer's output. Never
        # carry a heading across a blank line, SQL block or ordinary prose.
        errors = [re.fullmatch(r"(?:- |\[[^\]\r\n]+\] )(.+)", row) for row in lines[index + 1:]]
        if errors and all(errors):
            return any(_failure_text(tool, error[1]) for error in errors)
    return False


def degraded_record(record: Mapping) -> bool:
    """Detect flagged or legacy knowledge failures in one role's retained evidence.

    Only tool responses are inspected, with their corresponding call's tool name.
    Assistant/user text, SQL errors and healthy empty retrievals cannot mark a role.
    """
    if record.get("degraded_retrieval"):
        return True
    if _failure_text("get_rules", record.get("baseline_rules_error")):
        return True
    if any(isinstance(call, Mapping) and _failure_text(call.get("name"), call.get("error"))
           for call in record.get("tool_calls") or []):
        return True
    calls = {}
    for message in record.get("messages") or []:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                calls[call.get("id")] = (call.get("function") or {}).get("name")
            for block in content if isinstance(content, list) else []:
                if isinstance(block, Mapping) and block.get("type") == "tool_use":
                    calls[block.get("id")] = block.get("name")
        elif message.get("role") == "tool":
            if _degraded_content(calls.get(message.get("tool_call_id")), content):
                return True
        elif message.get("role") == "user" and isinstance(content, list):
            for block in content:
                if (isinstance(block, Mapping) and block.get("type") == "tool_result"
                        and _degraded_content(calls.get(block.get("tool_use_id")), block.get("content"))):
                    return True
    return False
