"""Question-local example masking at the host read boundary."""
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from .question import normalized_question


READERS = frozenset({"get_examples", "expand_examples", "query_knowledge", "query_spec", "read_examples"})
QUESTION_META = "rubiksql_original_question"
_QUESTION = ContextVar("rubiksql_example_question", default="")


def current_question() -> str:
    """Return the question owned by the current host example read, or empty outside a read."""
    return _QUESTION.get()


@contextmanager
def question_scope(question: str):
    """Carry one read's question into projected queries without altering pinned tools."""
    token = _QUESTION.set(question)
    try:
        yield
    finally:
        _QUESTION.reset(token)


def mask_example_rows(rows: list, question: str) -> list:
    """Withhold same-question examples without changing stored rows or other questions.

    Args:
        rows: Example records, including any error records.
        question: Runtime-owned current question, optionally carrying its hint.

    Returns:
        The original list when no collision exists; otherwise a filtered list retaining
        the original rows and order. Empty question text masks nothing.
    """
    target = normalized_question(question)
    if not target:
        return rows
    kept = [row for row in rows if not isinstance(row, dict) or row.get("error")
            or normalized_question(row.get("question")) != target]
    return rows if len(kept) == len(rows) else kept


def mask_tool_result(name: str, arguments: Mapping[str, Any], result: Any) -> Any:
    """Apply question-local masking to example reads, including pre-injection rows.

    Args:
        name: Host-exposed reader name.
        arguments: Bound callable arguments; original_question comes from the answering runtime.
        result: The pinned reader's unchanged result.

    Returns:
        The original result for non-example reads and non-colliding questions, otherwise
        the applicable example rows. The library read_examples helper uses its question
        argument when no separate original_question is supplied.
    """
    if name not in READERS or not isinstance(result, list):
        return result
    if name == "query_knowledge" and str(arguments.get("entity") or "").strip().lower() != "example":
        return result
    return mask_example_rows(result, arguments.get("original_question") or arguments.get("question") or "")
