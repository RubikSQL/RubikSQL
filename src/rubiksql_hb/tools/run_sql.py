"""Pinned implementation of run_sql."""


def run_sql(db_id: str, sql: str, session_id: str = "") -> dict:
    """Execute a SQL query and return results. Sees the base tables plus any intermediate
    tables saved this run via save_sql_as_table (session_id is injected by the agent)."""
    from ..server import _db_url, exec_session
    from .sql_result import sql_result

    url = _db_url(db_id)
    if not url:
        return {"error": f"db_id '{db_id}' not registered. Call register_database first."}
    return sql_result(exec_session(url, sql, session_id))
