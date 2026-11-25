"""Pinned implementation of search_values."""
from typing import Any
import heavenbase as hb
import sqlalchemy as sa
from ..prompts import current as _prompt_release
from ..retrieval import search_vectors
from ..schema import ColumnEntity, EnumValueEntity
from ..sql_host import _engine, _quote_ident

_PROBE_MAX_COLS = 12
_SYMBOL_NEAR = _prompt_release().knob("symbol_near")


def _probe_values(url: str, ws: hb.HeavenBase, db_id: str, keyword: str,
                  tables: set[str], cols: set[str], top_k: int) -> list[dict[str, Any]]:
    """Read candidate cell values straight from the database, for columns with no stored enum.

    ``EnumValueEntity`` only holds columns with at most 100 distinct values (see
    ``register.register_database``), so every high-cardinality column — the fact table's name
    columns, which are what a question usually points at — has an empty pool and no amount of
    vector search can find anything in it. This reads the live table instead: a bounded
    ``LIKE '%keyword%'`` over the same CATEGORICAL columns the indexer considered, on the
    read-only engine every other tool here uses.

    It buys coverage, not semantics. A literal probe cannot cross ``已发货`` → ``shipped``; that
    jump needs the embedding, and the columns where it is needed are low-cardinality and thus
    already indexed. The two paths are complementary, which is why this one runs only when the
    indexed path came back with nothing.

    Args:
        url (str): SQLAlchemy URL of the benchmark database. ``_engine`` forces read-only.
        ws (hb.HeavenBase): Workspace holding this database's ``ColumnEntity`` rows.
        db_id (str): Database to scope to.
        keyword (str): The term to look for, matched as a substring, case-insensitively.
        tables (set[str]): Restrict to these tables; empty means any.
        cols (set[str]): Restrict to these columns; empty means any.
        top_k (int): Stop once this many values have been collected.

    Returns:
        list[dict[str, Any]]: ``{table_name, col_name, value, score, via}`` per hit, best first,
        where ``score`` is string similarity to the keyword and ``via`` is ``"substring"`` so the
        caller can see these did not come from the vector index. Empty when nothing matched, the
        schema has no categorical column in scope, or the database cannot be read — a probe that
        fails must degrade to the old empty answer, never to an exception.
    """
    from difflib import SequenceMatcher

    try:
        rows = ws.query(ColumnEntity).where({"db_id": {"$eq": db_id}}).execute().rows()
    except Exception:                                            # noqa: BLE001
        return []
    scope = [(r["table_name"], r["name"]) for r in rows
             if (r.get("datatype_anno") == "CATEGORICAL")
             and (not tables or r["table_name"] in tables)
             and (not cols or r["name"] in cols)]
    if not scope:
        return []

    # LIKE's own wildcards have to be neutralised or a keyword containing % or _ silently
    # matches far more than the model asked for.
    esc = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    out: list[dict[str, Any]] = []
    try:
        engine = _engine(url)
        with engine.connect() as conn:
            for table, col in scope[:_PROBE_MAX_COLS]:
                try:
                    found = conn.execute(sa.text(
                        f"SELECT DISTINCT {_quote_ident(col)} AS v FROM {_quote_ident(table)} "
                        f"WHERE {_quote_ident(col)} LIKE :kw ESCAPE '\\' LIMIT :lim"
                    ), {"kw": f"%{esc}%", "lim": top_k}).fetchall()
                except Exception:                                # noqa: BLE001
                    continue                                     # one unreadable column, not a failure
                for (val,) in found:
                    if val is None:
                        continue
                    out.append({
                        "table_name": table, "col_name": col, "value": str(val),
                        "score": round(SequenceMatcher(None, keyword.lower(),
                                                       str(val).lower()).ratio(), 4),
                        "via": "substring",
                    })
                if len(out) >= top_k:
                    break
    except Exception:                                            # noqa: BLE001
        return []
    # Ties on `score` are broken by (table, col, value) so repeated calls agree with each other;
    # DISTINCT gives no order guarantee and the caller compares these lists across processes.
    out.sort(key=lambda x: (-x["score"], x["table_name"], x["col_name"], x["value"]))
    return out[:top_k]


def _is_symbol_value(value: str) -> bool:
    """True when the value contains no ASCII letter or digit anywhere ('∞', '=', '*²')."""
    return bool(value) and not any(c.isascii() and c.isalnum() for c in value)


def fuzzy_enum_values(
    ws: hb.HeavenBase,
    db_id: str,
    keyword: str,
    table_names: list[str] | None = None,
    col_names: list[str] | None = None,
    top_k: int = 10,
    thres: float = 0.45,
    alpha: float = 0.4,
    strong: float = 0.75,
    url: str | None = None,
) -> dict[str, Any]:
    """
    Vector-search actual cell values (enum values) to find which values in the
    database semantically match the given keyword.

    Hybrid score = alpha * string_similarity + (1 - alpha) * vector_cosine, in [0, 1];
    1.0 ≈ exact. Matches below `thres` are dropped. A glyph-only value ('=', '#', '∞') scores by
    its cosine alone and is dropped below :data:`_SYMBOL_NEAR` (the release's ``symbol_near``
    knob) instead. If the best score is below `strong`,
    the keyword is likely NOT a stored categorical value (e.g. a number / time /
    measurement literal) — the result carries strong_match=False plus a note so the
    caller falls back to get_col_info + direct SQL instead of trusting a weak hit.

    When the index holds nothing for the columns in scope — the normal state of any column with
    over 100 distinct values, which is never indexed — `url` lets the search fall back to a
    bounded literal `LIKE` read of the live table, and `matches` then carries entries tagged
    `via="substring"`. Those are candidate spellings found by containment, not by meaning.

    Returns {strong_match, best_score, matches:[{table_name, col_name, value, score,
    semantic_score, spelling_score}], note?}.
    """
    from difflib import SequenceMatcher
    from ..embedding import embed

    # The ×10 oversample is the rerank pool, not an index workaround: the hybrid score below
    # can promote a weak vector match with a strong string match, so the vector stage has to
    # hand down more than top_k rows. HB applies the complete table/column filter BEFORE its
    # exact top-k — the old ANN `.near()` took a global top-k and then intersected, which
    # starved a table/col-scoped search of candidates and could drop a true nearest neighbour
    # outright. Rubik owns only the spelling/semantic blend that follows.
    tables = set(table_names or ())
    cols = set(col_names or ())
    filters: list[dict[str, Any]] = []
    if tables:
        filters.append({"table_name": {"$in": sorted(tables)}})
    if cols:
        filters.append({"col_name": {"$in": sorted(cols)}})
    where = filters[0] if len(filters) == 1 else {"$and": filters} if filters else None
    candidates = search_vectors(ws, EnumValueEntity, db_id, "value_vec",
                                embed(keyword), top_k * 10, where=where)

    def str_sim(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    scored = []
    for row in candidates:
        vec = float(row.get("score") or 0.0)          # native cosine similarity
        ss  = str_sim(keyword, row["value"])
        hybrid = vec if _is_symbol_value(row["value"]) else alpha * ss + (1.0 - alpha) * vec
        scored.append({
            "table_name": row["table_name"],
            "col_name":   row["col_name"],
            "value":      row["value"],
            "score":      round(hybrid, 4),
            "semantic_score": round(vec, 4),
            "spelling_score": round(ss, 4),
        })
    # Stable sort over a deterministic input order (cosine desc, then object_id), so the
    # many ties that `round(..., 4)` manufactures resolve the same way in every process.
    scored.sort(key=lambda x: x["score"], reverse=True)
    scored = [s for s in scored
              if s["score"] >= (_SYMBOL_NEAR if _is_symbol_value(s["value"]) else thres)
              ][:top_k]
    best = scored[0]["score"] if scored else 0.0

    result: dict[str, Any] = {
        "strong_match": best >= strong,
        "best_score":   round(best, 4),
        "matches":      scored,
    }
    if best < strong:
        result["note"] = (
            f"No confident value match for {keyword!r} (best={best:.2f} < {strong}). "
            "It may not be a stored categorical value (e.g. a number, time, or measurement). "
            "Use get_col_info(table, col) to read the stored format, then filter directly."
        )
    # An empty pool and a pool that simply had nothing close are different failures, and only
    # the second one the note above describes. The first means the columns in scope were never
    # indexed at all — over 100 distinct values — which is the normal state of a fact table's
    # name column, so the answer was never in here to be found. Read the table instead of
    # telling the model something false about its own schema.
    if not scored and url:
        probe = _probe_values(url, ws, db_id, keyword, tables, cols, top_k)
        if probe:
            result["matches"] = probe
            result["note"] = (
                f"No indexed enum values for {keyword!r} — the columns in scope hold more than "
                "100 distinct values, so their cell values are not in the vector index. The "
                "matches above were read from the database just now by literal substring, NOT "
                "by meaning: they are candidate spellings, not a confirmed answer. Pick the one "
                "the question means and verify with run_sql before relying on it."
            )
    return result


def search_values(db_id: str, keyword: str, table_names: list[str] = [], col_names: list[str] = [], top_k: int = 10) -> dict:
    """Vector-search actual cell values to find enum values matching a keyword.
    Use this when you know the concept (e.g. '已发货') but need to find the exact stored value
    (e.g. 'shipped'). `score` is the ranking key; `semantic_score` (meaning closeness) and
    `spelling_score` (character overlap) are its two components. The model-facing result
    uses strict JSON with positional rows;
    interpret values with normal JSON types and decode JSON string escapes before copying
    a stored value into SQL."""
    from ..server import _ws_for, _db_url

    try:
        return fuzzy_enum_values(_ws_for(db_id), db_id, keyword, table_names=table_names or None, col_names=col_names or None, top_k=top_k, url=_db_url(db_id))
    except Exception as e:
        return {"error": f"search_values failed: {e}"}
