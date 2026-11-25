"""DashScope cache hints at the chat wire boundary, outside all replay identities."""
from __future__ import annotations

from copy import deepcopy
import inspect
import os
from typing import Any


def wire_request(request: dict[str, Any], model: str = "") -> dict[str, Any]:
    """Mark a detached Qwen conversation's system and newest text blocks.

    Args:
        request: SDK kwargs or raw chat body. The caller's messages remain untouched.
        model: Wire model fallback when the request omits its model.

    Returns:
        Original request for other models or RUBIKSQL_PROMPT_CACHE=0/off/false;
        otherwise a shallow request copy with deeply copied messages and at most
        four cache hints. System/newest hints take priority over existing hints
        when the request would otherwise exceed the four-marker allowance.
        Empty/non-text final messages fall back to the nearest preceding text.
    """
    model = str(request.get("model") or model).lower().rsplit("/", 1)[-1]
    if (not model.startswith("qwen")
            or os.environ.get("RUBIKSQL_PROMPT_CACHE", "1").strip().lower() in {"0", "off", "false"}
            or not isinstance(request.get("messages"), list)):
        return request
    messages = deepcopy(request["messages"])
    def last_text(message: dict[str, Any]) -> dict[str, Any] | None:
        content = message.get("content")
        if isinstance(content, str):
            if not content:
                return None
            content = [{"type": "text", "text": content}]
            message["content"] = content
        if isinstance(content, list):
            for part in reversed(content):
                if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                    return part
        return None

    targets = []
    for message in messages:
        if message.get("role") == "system":
            if (part := last_text(message)) is not None:
                targets.append(part)
            break
    for message in reversed(messages):
        if (part := last_text(message)) is not None:
            targets.append(part)
            break
    for part in targets:
        part.setdefault("cache_control", {"type": "ephemeral"})
    priority = {id(part) for part in targets}
    remaining = 4 - len(priority)
    for message in messages:
        content = message.get("content")
        for part in content if isinstance(content, list) else []:
            if isinstance(part, dict) and "cache_control" in part and id(part) not in priority:
                if remaining:
                    remaining -= 1
                else:
                    del part["cache_control"]
    return {**request, "messages": messages}


def response_usage(response: Any, *, stream: bool = False) -> Any:
    """Expose numeric cache aliases before HB's shallow tool-loop aggregation.

    Args:
        response: Owned SDK completion or stream. Original nested usage is retained.
        stream: Whether to adapt each streamed chunk and close its owned stream.

    Returns:
        Completion, or a sync/async iterator, with flat ``cached_tokens`` and
        ``cache_creation_input_tokens`` aliases when the provider reports them.
        These aliases are authoritative for HB dashboard multi-turn totals; they
        describe the same tokens as the nested counters and must not be added to them.
    """
    def expose(value: Any) -> Any:
        usage = value.get("usage") if isinstance(value, dict) else getattr(value, "usage", None)
        details = usage.get("prompt_tokens_details") if isinstance(usage, dict) else getattr(usage, "prompt_tokens_details", None)
        for name in ("cached_tokens", "cache_creation_input_tokens"):
            count = details.get(name) if isinstance(details, dict) else getattr(details, name, None)
            if isinstance(count, (int, float)) and not isinstance(count, bool):
                if isinstance(usage, dict):
                    usage.setdefault(name, count)
                elif not hasattr(usage, name):
                    setattr(usage, name, count)
        return value

    if not stream:
        return expose(response)
    if hasattr(response, "__aiter__"):
        async def aiterate():
            try:
                async for chunk in response:
                    yield expose(chunk)
            finally:
                close = getattr(response, "aclose", None) or getattr(response, "close", None)
                if close is not None:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
        return aiterate()

    def iterate():
        try:
            for chunk in response:
                yield expose(chunk)
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()
    return iterate()
