"""Pinned implementation of list_temp_tables."""


def list_temp_tables(db_id: str, session_id: str = "") -> list:
    """List the intermediate tables saved this run via save_sql_as_table."""
    from ..server import list_session_tables

    return list_session_tables(session_id)
