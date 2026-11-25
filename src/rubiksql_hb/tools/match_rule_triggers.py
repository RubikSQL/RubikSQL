"""Pinned implementation of match_rule_triggers."""
from typing import Any
import heavenbase as hb
from ..schema import RuleEntity
from ..knowledge_host import _trigger_matches, _rule_projection


def read_rule_triggers(
    ws: hb.HeavenBase,
    db_id: str,
    text: str,
    exclude_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return every not-yet-triggered rule whose literal trigger occurs in ``text``.

    Matching is case-insensitive after NFKC normalization. Underscores and hyphens are
    treated as spaces. Latin and numeric triggers require phrase boundaries, while CJK
    triggers use literal substring matching.

    Args:
        ws: Workspace holding the rules.
        db_id: Database whose triggers may match.
        text: One new user or assistant natural-language message.
        exclude_ids: Rule ids already injected during this session.

    Returns:
        list[dict[str, Any]]: All new matches ordered by title then entity id.
    """
    excluded = set(exclude_ids or [])
    rows = (
        ws.query(RuleEntity)
        .where({"db_id": {"$eq": db_id}})
        .execute()
        .rows()
    )
    matches = [
        row for row in rows
        if str(row.get("object_id") or "") not in excluded
        and any(_trigger_matches(text, trigger) for trigger in row.get("triggers") or [])
    ]
    matches.sort(key=lambda row: (
        str(row.get("title") or ""),
        str(row.get("object_id") or ""),
    ))
    return [_rule_projection(row) for row in matches]


def match_rule_triggers(
    db_id: str,
    text: str,
    exclude_ids: list[str] = [],
    session_id: str = "",
) -> list:
    """Runtime-only exact trigger matcher; this tool is hidden from model allowlists."""
    from ..server import knowledge_sessions, _ws_for

    try:
        if not session_id:
            raise ValueError("session_id is required for virtual rule ids")
        persistent_excludes = [
            source_id
            for virtual_id in exclude_ids
            if (
                source_id := knowledge_sessions.source_id(
                    session_id, "rule", db_id, virtual_id,
                )
            ) is not None
        ]
        rows = read_rule_triggers(
            _ws_for(db_id), db_id, text, exclude_ids=persistent_excludes,
        )
        out = []
        for row in rows:
            source_id = str(row.get("entity_id") or "")
            virtual_id = knowledge_sessions.retain(
                session_id, "rule", db_id, source_id, row,
            )
            out.append({
                "virtual_id": virtual_id,
                **{key: value for key, value in row.items() if key != "entity_id"},
            })
        return out
    except Exception as e:
        return [{"error": f"match_rule_triggers failed: {e}"}]
