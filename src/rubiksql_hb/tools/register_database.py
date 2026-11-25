"""Pinned implementation of register_database."""


def register_database(db_id: str, url: str, desc: str = "") -> str:
    """Register an external database. Introspects its schema and stores TableEntity + ColumnEntity records."""
    from ..server import _register_db, _ws_for, _DB_REGISTRY, _persist_registry

    result = _register_db(db_id=db_id, url=url, workspace=_ws_for(db_id), desc=desc)
    _DB_REGISTRY[db_id] = url
    _persist_registry()
    return f"Registered '{db_id}': {result['tables']} tables, {result['columns']} columns, {result['enum_values']} enum values."
