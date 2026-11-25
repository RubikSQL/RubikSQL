"""Pinned implementation of save_rule."""


def save_rule(
    db_id: str,
    title: str,
    desc: str,
    scope: str = "global",
    tags: list[str] = [],
    triggers: list[str] = [],
    example: str = "",
    session_id: str = "",
) -> dict:
    """Store a rule with scope global or database and return its session-local R-id."""
    from ..server import normalize_rule_scope, _save_rule, _ws_for, knowledge_sessions

    try:
        if not session_id:
            raise ValueError("session_id is required for virtual rule ids")
        normalized_scope = normalize_rule_scope(scope)
        source_id = _save_rule(
            _ws_for(db_id),
            db_id,
            title,
            desc,
            scope=normalized_scope,
            tags=tags,
            triggers=triggers,
            example=example,
        )
        if not source_id:
            raise RuntimeError("rule writer returned no persistent id")
        payload = {
            "entity_id": source_id,
            "scope": normalized_scope,
            "title": title,
            "desc": desc,
            "tags": tags or [],
        }
        if example:
            payload["example"] = example
        virtual_id = knowledge_sessions.retain(
            session_id, "rule", db_id, source_id, payload,
        )
        return {
            "virtual_id": virtual_id,
            **{key: value for key, value in payload.items() if key != "entity_id"},
        }
    except Exception as e:
        return {"error": f"save_rule failed: {e}"}
