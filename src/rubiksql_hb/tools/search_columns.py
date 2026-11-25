"""Pinned implementation of search_columns."""
from typing import Any
import heavenbase as hb
from ..retrieval import search_vectors
from ..schema import ColumnEntity


def fuzzy_enum(
    ws: hb.HeavenBase,
    db_id: str,
    keyword: str,
    table_names: list[str] | None = None,
    top_k: int = 10,
    include_profile: bool = False,
) -> list[dict[str, Any]]:
    """
    Vector-search column descriptions to find columns whose values likely
    match the given keyword (enum/categorical lookup).
    include_profile additionally returns each hit's empirical profile_desc
    (k-bounded — this is the search shortlist, never the whole schema).
    Ranked by HB's exact cosine scan over the table-filtered rows (see
    :func:`~rubiksql_hb.retrieval.search_vectors`); there is no fallback. A vector store
    that cannot be queried raises; one that is merely incomplete does not — a column whose
    ``desc_vec`` is missing from Chroma is skipped by the exact scan, so the result is
    shorter, not an error. Completeness is checked when the workspace is built
    (``scripts/setup_dbs.py``), not here.
    """
    from ..embedding import embed
    tables = set(table_names or ())
    fields = ["object_id", "table_name", "name", "datatype_orig", "datatype_anno", "desc"]
    if include_profile:
        fields.append("profile_desc")
    rows = search_vectors(ws, ColumnEntity, db_id, "desc_vec", embed(keyword), top_k,
                          where={"table_name": {"$in": sorted(tables)}} if tables else None)
    return [{f: row.get(f) for f in fields} for row in rows]


def search_columns(db_id: str, keyword: str, table_names: list[str] = [], top_k: int = 10) -> list:
    """Vector-search columns by keyword to find enum/categorical matches."""
    from ..server import _ws_for, _SEARCH_COLUMNS_PROFILE

    try:
        return fuzzy_enum(_ws_for(db_id), db_id, keyword, table_names=table_names or None,
                          top_k=top_k, include_profile=_SEARCH_COLUMNS_PROFILE)
    except Exception as e:
        return [{"error": f"search_columns failed: {e}"}]
