"""Pinned implementation of expand_examples."""


def expand_examples(
    db_id: str,
    virtual_ids: list[str],
    session_id: str = "",
) -> list:
    """Read cached question + SQL payloads for virtual ids from the current session."""
    from ..server import knowledge_sessions

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
            out.append({
                "virtual_id": str(virtual_id).strip().upper(),
                "question": payload.get("question"),
                "sql": payload.get("sql"),
            })
        return out
    except Exception as e:
        return [{"error": f"expand_examples failed: {e}"}]
