"""Pinned implementation of expand_examples."""


def expand_examples(
    db_id: str,
    virtual_ids: list[str],
    session_id: str = "",
    original_question: str = "",
) -> list:
    """Read applicable cached question + SQL payloads for this session's virtual ids. Optional example_scope describes that precedent only; it cannot narrow the current request."""
    from ..server import knowledge_sessions
    from .example_scope import example_applicable, example_scope_note

    try:
        if not session_id:
            raise ValueError("session_id is required for virtual example ids")
        out = []
        for virtual_id in list(virtual_ids or [])[:12]:
            payload = knowledge_sessions.resolve(
                session_id, "example", db_id, virtual_id,
            )
            if payload is None:
                out.append({
                    "virtual_id": virtual_id,
                    "error": "unknown example virtual id for this session and database",
                })
                continue
            if not example_applicable(payload, original_question):
                continue
            out.append({
                "virtual_id": str(virtual_id).strip().upper(),
                "question": payload.get("question"),
                "sql": payload.get("sql"),
            })
            scope = example_scope_note(payload)
            if scope:
                out[-1]["example_scope"] = scope
        return out
    except Exception as e:
        return [{"error": f"expand_examples failed: {e}"}]
