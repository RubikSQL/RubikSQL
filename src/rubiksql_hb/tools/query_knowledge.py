"""Pinned implementation of query_knowledge."""


def query_knowledge(
    db_id: str,
    entity: str,
    where: dict = {},
    limit: int = 50,
    session_id: str = "",
    original_question: str = "",
) -> list:
    """Query the KNOWLEDGE BASE with exact filters — the business rules, NL→SQL examples, synonyms and schema metadata curated for this database. It does NOT read the SQL database's data; run_sql does that.
    It complements get_rules, get_examples and search_values, which return only the top-k most SIMILAR records. This one returns EVERY record matching an exact filter (up to 200), so you can enumerate and count instead of sampling.
    `entity` selects what to read, and each carries these fields: "rule" — scope, title, desc, tags, triggers, example. "example" — question, sql, tables, difficulty. "synonym" — table_name, col_name, synonym, canonical. "column" — table_name, name, datatype_orig, datatype_anno, desc, profile_desc, is_pk, fk_ref, disabled. "table" — name, desc, n_rows, pks, disabled. "enum_value" — table_name, col_name, value. Rule/example results carry session-local `virtual_id`; other entities may carry `entity_id`.
    `where` filters those fields exactly; omit it to enumerate every record of that entity. Operators are $eq $ne $in $contains $like $ilike $starts_with $gt $gte $lt $lte, plus $and / $or / $not over nested filters; for the list-valued fields (tables, tags, triggers, pks) use $contains, not $in.
    Worked examples. Every rule specific to the current database, which get_rules may not contain: query_knowledge(entity="rule", where={"scope": {"$eq": "database"}}). Every synonym defined for this database, which NO other tool exposes (expand_synonyms only rewrites the question): query_knowledge(entity="synonym"). Every CATEGORICAL column: query_knowledge(entity="column", where={"datatype_anno": {"$eq": "CATEGORICAL"}}). Every stored example whose SQL touches a table: query_knowledge(entity="example", where={"tables": {"$contains": "cards"}}).
    The filter is always ANDed with this database, so another database's knowledge is never readable. An unknown entity name, or a filter naming a field the entity does not have, comes back as one `error` record listing the valid names.

    Args:
        db_id: Registered database identifier, injected by the agent runtime.
        entity: Knowledge type to read; one of the six names described above.
        where: Exact filter mapping over that entity's own fields, as described above.
        limit: Maximum records to return (default 50, hard cap 200).

    Returns:
        list: Matching records with embedding vectors stripped, or a single-element list
        holding an `error` field when the entity name or the filter is invalid.
    """
    from ..server import _query_knowledge, _ws_for, knowledge_sessions
    from .example_scope import example_applicable, example_scope_note

    try:
        normalized_entity = str(entity or "").strip().lower()
        rows = _query_knowledge(
            _ws_for(db_id), db_id, normalized_entity, where=where or None, limit=limit,
        )
        if normalized_entity not in {"rule", "example"}:
            return rows
        if not session_id:
            raise ValueError(
                "session_id is required for virtual rule/example ids"
            )
        out = []
        for row in rows:
            if normalized_entity == "example" and not example_applicable(row, original_question):
                continue
            source_id = str(row.get("entity_id") or "")
            virtual_id = knowledge_sessions.retain(
                session_id, normalized_entity, db_id, source_id, row,
            )
            out.append({
                "virtual_id": virtual_id,
                **{key: value for key, value in row.items() if key != "entity_id"},
            })
            if normalized_entity == "example":
                scope = example_scope_note(row)
                if scope:
                    out[-1]["example_scope"] = scope
        return out
    except Exception as e:
        return [{"error": f"query_knowledge failed: {e}"}]
