"""Pinned implementation of update_rule."""


def update_rule(
    db_id: str,
    virtual_id: str,
    desc: str,
    scope: str = "",
    session_id: str = "",
) -> dict:
    """Update a retained rule by its session-local R-id. Non-empty scope must be global or database."""
    from ..server import knowledge_sessions, _update_rule, _ws_for

    try:
        if not session_id:
            raise ValueError("session_id is required for virtual rule ids")
        source_id = knowledge_sessions.source_id(
            session_id, "rule", db_id, virtual_id,
        )
        if source_id is None:
            raise ValueError(
                "unknown rule virtual id for this session and database"
            )
        updated = _update_rule(
            _ws_for(db_id),
            db_id,
            source_id,
            desc,
            scope=scope,
        )
        if "error" in updated:
            return updated
        normalized_id = knowledge_sessions.retain(
            session_id, "rule", db_id, source_id, updated,
        )
        return {
            "virtual_id": normalized_id,
            **{key: value for key, value in updated.items() if key != "entity_id"},
        }
    except Exception as e:
        return {"error": f"update_rule failed: {e}"}
