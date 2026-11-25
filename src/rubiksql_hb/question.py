"""Canonical question and hint composition."""
from collections.abc import Iterable
from typing import Any


HINT_SEPARATOR = "\n[Hint]: "


def normalized_question(question: str | None) -> str:
    """Return casefolded question text with whitespace collapsed and its stored hint removed.

    Args:
        question: Raw or hint-bearing question text; None is empty.

    Returns:
        Normalized text for exact question equality, preserving punctuation and wording.
    """
    text = str(question or "").casefold().split("\n[hint]:", 1)[0]
    return " ".join(text.split())


def normalize_hints(hints: Any = None) -> tuple[str, ...]:
    """Normalize one hint or a sequence of hints without changing their order.

    Args:
        hints: A hint string, an iterable of hint values, or ``None``.

    Returns:
        tuple[str, ...]: Non-empty stripped hints with exact duplicates removed.
    """
    if hints is None:
        return ()
    values: Iterable[Any] = (hints,) if isinstance(hints, str) else (
        hints if isinstance(hints, Iterable) else (hints,)
    )
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def question_hints(question: str) -> tuple[str, ...]:
    """Return the canonical hint block already carried by a question.

    Args:
        question: Raw or hint-bearing question text.

    Returns:
        tuple[str, ...]: An empty tuple or the existing hint block as one value.
    """
    _, separator, hint = str(question or "").partition(HINT_SEPARATOR)
    return (hint.strip(),) if separator and hint.strip() else ()


def question_with_hint(question: str, hints: Any = None) -> str:
    """Compose the semantic question used by every reasoning and retrieval path.

    The operation is idempotent for canonical input: passing a question that already carries
    the same hint block returns it unchanged. New hints are appended after any existing block
    without duplicating exact values.

    Args:
        question: Raw or already hint-bearing question text.
        hints: A hint string, an iterable of hint values, or ``None``.

    Returns:
        str: The untouched question when no hint exists, otherwise
        ``question + "\\n[Hint]: " + "; ".join(hints)``.
    """
    text = str(question or "")
    raw, separator, existing = text.partition(HINT_SEPARATOR)
    incoming = normalize_hints(hints)
    if not separator:
        return text if not incoming else raw + HINT_SEPARATOR + "; ".join(incoming)

    current = existing.strip()
    if not incoming:
        return text
    joined = "; ".join(incoming)
    if joined == current:
        return text
    merged = [current] if current else []
    merged.extend(hint for hint in incoming if hint not in merged)
    return raw + HINT_SEPARATOR + "; ".join(merged)
