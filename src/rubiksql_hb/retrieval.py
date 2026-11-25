"""Business-scoped exact vector queries — HeavenBase owns the ranking.

Every knowledge tool used to rank through an application-side ``VectorPool``: one in-memory
float32 matrix per (workspace, entity, database, field), scored by brute-force cosine and
tie-broken on ``object_id``. It existed because Chroma's HNSW graph was never flushed at these
collection sizes and each process rebuilt a different one, so ``.near()`` could drop a true
nearest neighbour. The vendored HeavenBase build now offers ``near(..., exact=True)``: an
exhaustive scan after the complete structured filter, ranked by ``(-score, object_id)`` in
binary64, with an optional versioned candidate cache. The pool is therefore gone and this
module is one function that states the business scope and lets HB execute it.

Behaviour relative to the pool it replaces: scores are HB ``cosine_score`` semantics
(binary64; a zero-norm row or query scores ``0.0``), so they differ from the float32 pool by
about 1e-8; ordering and tie-breaking are the same. Without a ``retrieval_version`` on the
workspace every query is a fresh full scan — the same work the pool did on its first load,
repeated per call. ``scripts/compare_exact_retrieval.py`` measures the parity offline.

Two differences are not parity and are stated here because nothing at query time reports
them. A row whose vector is missing from the vector backend (``question_vec``, ``value_vec``
and ``desc_vec`` live only in Chroma) is SKIPPED by HB's exact scan — it scores ``None`` and
drops out — where the pool refused to load and every caller answered with an error. An
interrupted ``sync_vector_index`` (delete-then-add, not atomic) or a killed build therefore
reads as a healthy, smaller result set. Integrity is a build-time gate, not a query-time
one: ``scripts/setup_dbs.py`` compares Chroma and SQLite counts (``_verify_chroma_setup``)
after each build, and ``scripts/sync_chroma_index.py --force`` rewrites the collections of
a built workspace from its stored vectors. And HB caps ``top_k`` at its configured
``heavenbase.query.near.max_top_k`` (1000 in the vendored build); the pool had no ceiling and
returned the whole pool past it, so :func:`search_vectors` clamps to that maximum instead
of letting the request fail.
"""
from __future__ import annotations

from typing import Any

import heavenbase as hb

#: HB's ``heavenbase.query.near.max_top_k`` when the workspace carries no configuration
#: (test doubles); the vendored build's ``resources/configs/default.yaml`` sets 1000.
NEAR_MAX_TOP_K_DEFAULT = 1000


def near_max_top_k(ws: hb.HeavenBase) -> int:
    """The widest ``top_k`` one HB ``near`` query on ``ws`` accepts.

    HB rebinds every executed ``Near`` against ``heavenbase.query.near.max_top_k`` and
    rejects a wider request with ``ValueError``. Callers that let a model choose a width
    clamp to this value rather than surface that error.

    Args:
        ws (heavenbase.HeavenBase): Workspace whose context configuration answers; an object
            without a context (a test double) gets :data:`NEAR_MAX_TOP_K_DEFAULT`.

    Returns:
        int: The configured maximum, at least 1.
    """
    context = getattr(ws, "context", None)
    config = getattr(context, "config", None)
    if config is None or not callable(getattr(config, "get", None)):
        return NEAR_MAX_TOP_K_DEFAULT
    return max(1, int(config.get("heavenbase.query.near.max_top_k",
                                 default=NEAR_MAX_TOP_K_DEFAULT)))


def search_vectors(
    ws: hb.HeavenBase,
    entity: type[hb.Entity],
    db_id: str,
    field: str,
    query_vector: list[float],
    top_k: int,
    *,
    where: dict[str, Any] | None = None,
    exact: bool | None = None,
) -> list[dict[str, Any]]:
    """Rank one entity's rows for one business database by cosine similarity.

    The query is ``db_id`` scope, then ``where``, then HB ``near``; HB applies the whole filter
    before it ranks, so the result is the true top-k of the admitted subset rather than a
    post-filtered global top-k. Embedding fields are projected out so no vector reaches a
    prompt, and ``object_id`` plus ``score`` ride along because callers key on both.

    Args:
        ws: Workspace to query. Nothing is retained in an application cache; HB's own
            candidate cache is used only when the workspace carries a ``retrieval_version``.
        entity: Knowledge entity to search.
        db_id: Business database whose rows may be returned.
        field: Vector field to rank on, e.g. ``"desc_vec"``.
        query_vector: Embedded query; need not be normalised.
        top_k: Maximum rows to return. Zero or negative returns nothing without querying;
            a width above :func:`near_max_top_k` is clamped to it, so a subset no larger
            than that maximum is still returned whole, as the pool returned it.
        where: Additional JSON filter (``$eq``, ``$in``, ``$and``, ``$not``, ...) applied by
            HB before ranking. ``$nin`` is not registered in the vendored build; write an
            exclusion as ``{"$not": {"field": {"$in": [...]}}}``.
        exact: Exhaustive requirement. ``None`` reads the workspace's ``retrieval_exact``
            setting, which :func:`rubiksql_hb.workspace.open_workspace` defaults to ``True``;
            ``True`` forces an exact scan for a caller that rescores a known candidate set.

    Returns:
        list[dict[str, Any]]: HB-ranked rows, highest ``score`` first and ties broken on
        ``object_id`` ascending, each carrying every non-vector field plus ``score``.
    """
    if top_k <= 0:
        return []
    top_k = min(top_k, near_max_top_k(ws))
    query = ws.query(entity).where({"db_id": {"$eq": db_id}})
    if where:
        query = query.where(where)
    required = getattr(ws, "_rubiksql_retrieval_exact", True) if exact is None else exact
    fields = [name for name in entity.schema().fields if not name.endswith("_vec")]
    return (
        query.near(
            field,
            query_vector,
            top_k=top_k,
            metric="cosine",
            exact=required,
            cache_version=getattr(ws, "_rubiksql_retrieval_version", None) if required else None,
        )
        .select(*fields, "score")
        .execute()
        .rows()
    )
