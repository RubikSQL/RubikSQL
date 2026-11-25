"""Pinned implementation of get_col_info."""
from typing import Any
import heavenbase as hb
from ..schema import ColumnEntity, TableEntity


def _fuzzy_suggest(candidates: list[str], query: str, top_k: int = 3, thres: float = 0.3) -> list[str]:
    from difflib import SequenceMatcher
    scored = [(c, SequenceMatcher(None, query.lower(), c.lower()).ratio()) for c in candidates]
    return [c for c, s in sorted(scored, key=lambda x: -x[1]) if s >= thres][:top_k]


def col_info(ws: hb.HeavenBase, db_id: str, table_name: str, col_name: str = "") -> Any:
    """Column profile_desc only (value formats, nulls, ranges, sentinels). Structural fields
    (type, PK, FK, desc) are in get_db_info / get_tab_info — not repeated here.

    Measured lines (Values / Range / NULL / Rows / Docs-vs-data) are executed facts — where the
    documented text disagrees with them, trust the measurement. The standing rule is stated ONCE
    to the model, in the served ``get_col_info`` description (``server.py``, pinned by
    ``tests/mcp_contract_baseline.json``); this docstring mirrors it for the Python caller,
    which reads the profile through this function instead.

    - col_name omitted or "": return profile_desc for every column in the table
    - col_name provided: return profile_desc for that single column (fuzzy name correction built in)
    """
    # resolve table_name with fuzzy correction
    all_tables = [
        r["name"]
        for r in ws.query(TableEntity)
        .where({"db_id": {"$eq": db_id}})
        .select("name")
        .execute()
        .rows()
    ]
    msgs = []
    resolved_table = table_name
    if table_name not in all_tables:
        suggestions = _fuzzy_suggest(all_tables, table_name)
        if not suggestions:
            return {"error": f"Table '{table_name}' not found in db_id='{db_id}'. Available: {all_tables}"}
        resolved_table = suggestions[0]
        msgs.append(f"Table '{table_name}' not found. Closest match: '{resolved_table}'.")

    # no col_name → return profile_desc for all columns in the table
    if not col_name:
        cols = (
            ws.query(ColumnEntity)
            .where({"db_id": {"$eq": db_id}, "table_name": {"$eq": resolved_table},
                    "disabled": {"$eq": False}})
            .select("name", "profile_desc")
            .execute()
            .rows()
        )
        result: Any = {
            "table": resolved_table,
            "columns": [{"name": c["name"], "profile_desc": c.get("profile_desc") or ""} for c in cols],
        }
        if msgs:
            result["warnings"] = msgs
        return result

    # specific col_name → resolve with fuzzy correction
    all_cols = [
        r["name"]
        for r in ws.query(ColumnEntity)
        .where({"db_id": {"$eq": db_id}, "table_name": {"$eq": resolved_table}})
        .select("name")
        .execute()
        .rows()
    ]
    resolved_col = col_name
    if col_name not in all_cols:
        suggestions = _fuzzy_suggest(all_cols, col_name)
        if not suggestions:
            return {"error": f"Column '{col_name}' not found in '{resolved_table}'. Available: {all_cols}"}
        resolved_col = suggestions[0]
        msgs.append(f"Column '{col_name}' not found. Closest match: '{resolved_col}'.")

    cols = (
        ws.query(ColumnEntity)
        .where({"db_id": {"$eq": db_id}, "table_name": {"$eq": resolved_table},
                "name": {"$eq": resolved_col}})
        .select("name", "table_name", "profile_desc")
        .execute()
        .rows()
    )
    if not cols:
        return {"error": f"Column '{resolved_col}' not found in '{resolved_table}'"}
    c = cols[0]
    result = {"name": c["name"], "table_name": c["table_name"], "profile_desc": c.get("profile_desc") or ""}
    if msgs:
        result = {"warnings": msgs, **result}
    return result


def get_col_info(db_id: str, table_name: str, col_name: str = "") -> dict:
    """Return profile_desc for columns (value formats, nulls, ranges, sentinels). Structural info
    (type, PK/FK, desc) is already in get_db_info — this tool adds the statistical layer only.
    Measured lines (Values / Range / NULL / Rows / Docs-vs-data) are executed facts — where the
    documented text disagrees with them, trust the measurement.
    col_name omitted: return profile_desc for all columns in the table.
    col_name provided: return profile_desc for that single column (fuzzy name correction built in)."""
    from ..server import _ws_for

    try:
        return col_info(_ws_for(db_id), db_id, table_name, col_name)
    except Exception as e:
        return {"error": f"get_col_info failed: {e}"}
