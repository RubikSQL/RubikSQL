"""Pinned implementation of query_spec."""
_SPEC_HIDDEN_IDS = frozenset({"rule", "example"})


def query_spec(db_id: str, entity: str, spec: dict = {}) -> dict:
    """COMPUTE over the KNOWLEDGE BASE with a full query spec — aggregation, projection and ordering, where query_knowledge only filters and lists. Same six entities (rule | example | synonym | column | table | enum_value), same scope: this database's curated knowledge, never the SQL database's data (run_sql is for data).
    `spec` keys: filter (same operators as query_knowledge's where), select (list of fields to return), group_by (list of fields) with aggregate mapping each output name to {op: {"field": <field>}} where op is count | min | max | sum | avg, having, order (field names, "-" prefix for descending), offset, and limit (default 50, hard cap 200). Per-group counts are exact over ALL matching records, but the result lists at most `limit` rows or group rows (up to 200) — when more exist the result carries "truncated": true; raise limit or page with offset.
    Worked examples. How many enum values each column of a table carries: query_spec(entity="enum_value", spec={"filter": {"table_name": {"$eq": "cards"}}, "group_by": ["col_name"], "aggregate": {"n": {"count": {"field": "entity_id"}}}, "order": ["-n"]}). Every table's size at a glance: query_spec(entity="table", spec={"select": ["name", "n_rows"], "order": ["-n_rows"]}). Stored examples per difficulty: query_spec(entity="example", spec={"group_by": ["difficulty"], "aggregate": {"n": {"count": {"field": "entity_id"}}}}).
    Embedding fields are never returned. rule/example rows come back WITHOUT ids (and min/max/sum/avg over their entity_id is refused — count is fine) — aggregate them here, but read them through query_knowledge or get_rules/get_examples when you need ids for update_rule or expand_examples. The filter is always ANDed with this database, so another database's knowledge is never readable.

    Args:
        db_id: Registered database identifier, injected by the agent runtime.
        entity: Knowledge type to read; one of the six names described above.
        spec: Query specification as described above; {} scans with the default limit.

    Returns:
        dict: A {"columns", "rows", "n", "truncated"} result frame, or a mapping with
        an `error` field when the entity name or the spec is invalid.
    """
    from ..server import refuse_code_reduce, _query_frame, _ws_for, _QUERY_ENTITIES

    try:
        # Before the id guard reads the spec: a spec sent as JSON text fails there unnamed,
        # and the code_reduce refusal must be the answer whatever shape the spec has.
        refuse_code_reduce(spec)
        normalized = str(entity or "").strip().lower()
        aggregate = (spec or {}).get("aggregate")
        aggregate = aggregate if isinstance(aggregate, dict) else {}
        if normalized == "example":
            # Raw projections/aggregates of question or SQL bypass scoped admission.
            # Keep non-content corpus counts; row reads use the scoped exact path.
            groups = (spec or {}).get("group_by") or []
            if (not aggregate or any(op != "count" for metric in aggregate.values()
                                     for op in metric)
                    or any(field not in {"difficulty", "tables", "db_id"} for field in groups)
                    or any(field not in {*groups, *aggregate} for field in ((spec or {}).get("select") or []))):
                return {"error": "Example content requires request-scoped admission; use query_knowledge(entity='example')"}
        if normalized in _SPEC_HIDDEN_IDS:
            # min/max/sum/avg re-expose a persistent id under any output alias; count
            # yields an integer and stays safe.
            for metric in aggregate.values():
                for op, params in (metric.items() if isinstance(metric, dict) else ()):
                    field = params.get("field") if isinstance(params, dict) else None
                    if op != "count" and field in ("entity_id", "object_id"):
                        return {"error": f"query_spec failed: {op} over entity_id is not "
                                         f"available for {normalized} rows — count is; read "
                                         f"ids through query_knowledge"}
        frame = _query_frame(
            _ws_for(db_id), _QUERY_ENTITIES, normalized,
            spec=spec or None, scope=("db_id", db_id),
        )
        # Strip the persistent id column unless "entity_id" is an aggregate output alias
        # here — then the column is a caller-named metric (a count), not a row id.
        if normalized in _SPEC_HIDDEN_IDS and "entity_id" not in aggregate:
            frame["columns"] = [col for col in frame["columns"] if col != "entity_id"]
            for row in frame["rows"]:
                row.pop("entity_id", None)
        return frame
    except Exception as e:
        return {"error": f"query_spec failed: {e}"}
