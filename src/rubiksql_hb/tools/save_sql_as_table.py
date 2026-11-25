"""Pinned implementation of save_sql_as_table."""


def save_sql_as_table(db_id: str, sql: str, table_name: str, session_id: str = "") -> dict:
    """Run SQL and save the result as a named intermediate table for THIS run, reusable by
    name in later run_sql. It is private to this run and never touches the base DB.
    submit_final_sql runs WITHOUT these intermediates, so inline them into a self-contained
    base-table SQL before submitting."""
    from ..server import _db_url, save_temp

    url = _db_url(db_id)
    if not url:
        return {"error": f"db_id '{db_id}' not registered."}
    return save_temp(url, sql, table_name, session_id)
