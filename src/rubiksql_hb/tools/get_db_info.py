"""Pinned implementation of get_db_info."""
import re
from typing import Any
import heavenbase as hb
from ..schema import ColumnEntity, DatabaseEntity, TableEntity
from .schema_rows import _col_row, _table_desc

_BRIEF_CAP = 160
_BRIEF_MARK = " …(full: get_tab_info)"


def _brief_doc(doc: str) -> str:
    """User ruling (08-06): serve the stored bytes — parts keep their stored newlines, the
    label lives in storage, and the only surgery is the length cut, backed off to the last
    whole word because a truncated "USD" reads as the different word "US".
    """
    doc = (doc or "").strip()
    if len(doc) <= _BRIEF_CAP:
        return doc
    cut = doc[:_BRIEF_CAP]
    if not cut[-1].isspace() and not doc[_BRIEF_CAP].isspace():
        m = re.match(r"[\s\S]*\s", cut)
        if m:
            cut = m.group(0)
    return cut.rstrip() + _BRIEF_MARK


def db_info(ws: hb.HeavenBase, db_id: str) -> dict[str, Any]:
    """Full schema: all tables with column names, types, PK/FK, and semantic desc.
    Sufficient to write SQL. Does NOT include profile_desc — use get_col_info for that."""
    tables = (
        ws.query(TableEntity)
        .where({"db_id": {"$eq": db_id}, "disabled": {"$eq": False}})
        .select("name", "desc", "profile_desc", "n_rows", "pks")
        .execute()
        .rows()
    )
    if not tables:
        return {"error": f"No schema found for db_id='{db_id}'"}

    result = []
    for t in tables:
        cols = (
            ws.query(ColumnEntity)
            .where({"db_id": {"$eq": db_id}, "table_name": {"$eq": t["name"]}, "disabled": {"$eq": False}})
            .select("name", "datatype_orig", "is_pk", "fk_ref", "desc")
            .execute()
            .rows()
        )
        # Brief 是生产唯一形态(08-06 打平): 列文档一律地图化,全文在 get_tab_info。
        col_rows = [_col_row(c) for c in cols]
        for c in col_rows:
            c["desc"] = _brief_doc(c["desc"])
        result.append({
            "table":   t["name"],
            # Served desc composes the two layers, mirroring how a reader wants them:
            # official docs first, LLM profile appended. Either alone serves as-is.
            "desc":    _table_desc(t),
            "n_rows":  t.get("n_rows"),
            "pks":     t.get("pks", []),
            "columns": col_rows,
        })
    out: dict[str, Any] = {"db_id": db_id, "tables": result}
    # Database-level overview (cross-table disambiguation, grain map). Added only when a
    # profile exists, so databases without one keep today's byte-identical payload.
    try:
        overview_rows = (
            ws.query(DatabaseEntity)
            .where({"db_id": {"$eq": db_id}})
            .select("profile_desc")
            .execute()
            .rows()
        )
    except Exception:  # noqa: BLE001 — a pre-DatabaseEntity workspace that could not be
        # migrated (e.g. opened read-only) must degrade to "no overview", never break
        # get_db_info for the whole database.
        overview_rows = []
    overview = (overview_rows[0].get("profile_desc") or "").strip() if overview_rows else ""
    if overview:
        out["overview"] = overview
    return out


def get_db_info(db_id: str) -> dict:
    """Return the full schema (all tables and columns) for a registered database."""
    from ..server import _ws_for

    try:
        return db_info(_ws_for(db_id), db_id)
    except Exception as e:
        return {"error": f"get_db_info failed: {e}"}
