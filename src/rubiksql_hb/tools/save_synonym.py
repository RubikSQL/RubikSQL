"""Pinned implementation of save_synonym."""


def save_synonym(db_id: str, table_name: str, synonym: str, canonical: str, col_name: str = "") -> str:
    """Store a synonym mapping so user terms are resolved to canonical schema names."""
    from ..server import add_synonym, _ws_for

    return add_synonym(_ws_for(db_id), db_id, table_name, synonym, canonical, col_name=col_name)
