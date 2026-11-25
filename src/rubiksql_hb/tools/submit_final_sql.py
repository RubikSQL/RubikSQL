"""Pinned implementation of submit_final_sql."""


def submit_final_sql(db_id: str, sql: str, session_id: str = "") -> dict:
    """Validate SQL syntax then execute it as the confirmed final answer."""
    from ..server import _db_url, _submit_sql
    from .sql_result import sql_result

    url = _db_url(db_id)
    if not url:
        return {"error": f"db_id '{db_id}' not registered."}
    return sql_result(_submit_sql(url, sql))
