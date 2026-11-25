"""Pinned implementation of run_sql_base."""


def run_sql_base(db_id: str, sql: str, session_id: str = "") -> dict:
    """Execute a SQL query against the BASE TABLES ONLY — the SAME conditions as
    submit_final_sql (your run's intermediate tables are NOT visible). Use this to verify a
    self-contained final answer before submitting: if it passes here it will pass submit."""
    from ..server import _db_url, exec_sql
    from .sql_result import sql_result

    url = _db_url(db_id)
    if not url:
        return {"error": f"db_id '{db_id}' not registered. Call register_database first."}
    return sql_result(exec_sql(url, sql))
