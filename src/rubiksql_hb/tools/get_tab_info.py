"""Pinned implementation of get_tab_info."""
from typing import Any
import heavenbase as hb
from ..schema import ColumnEntity, TableEntity
from .schema_rows import _col_row, _table_desc


def tab_info(ws: hb.HeavenBase, db_id: str, table_name: str) -> dict[str, Any]:
    """Single-table variant of get_db_info. Use when you only need one table's schema."""
    tables = (
        ws.query(TableEntity)
        .where({"db_id": {"$eq": db_id}, "name": {"$eq": table_name}})
        .select("name", "desc", "profile_desc", "n_rows", "pks")
        .execute()
        .rows()
    )
    if not tables:
        return {"error": f"Table '{table_name}' not found in db_id='{db_id}'"}

    t = tables[0]
    cols = (
        ws.query(ColumnEntity)
        .where({"db_id": {"$eq": db_id}, "table_name": {"$eq": table_name}, "disabled": {"$eq": False}})
        .select("name", "datatype_orig", "is_pk", "fk_ref", "desc")
        .execute()
        .rows()
    )
    return {
        "table":   t["name"],
        "desc":    _table_desc(t),
        "n_rows":  t.get("n_rows"),
        "pks":     t.get("pks", []),
        "columns": [_col_row(c) for c in cols],
    }


def get_tab_info(db_id: str, table_name: str) -> dict:
    """Return metadata and columns for a single table."""
    from ..server import _ws_for

    try:
        return tab_info(_ws_for(db_id), db_id, table_name)
    except Exception as e:
        return {"error": f"get_tab_info failed: {e}"}
