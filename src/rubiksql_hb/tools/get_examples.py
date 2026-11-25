"""Pinned implementation of get_examples."""
from typing import Any
import heavenbase as hb
from ..retrieval import search_vectors
from ..schema import ExampleEntity
from ..prompts import current as _served_release
from .knowledge_errors import _retrieval_failed

_EXAMPLES_TOP_K = _served_release().knob("examples_top_k")


def read_examples(
    ws: hb.HeavenBase,
    db_id: str,
    question: str,
    top_k: int = 12,
    menu: bool = False,
) -> list[dict[str, Any]]:
    """
    Retrieve the most relevant historical NL→SQL examples for a question.
    Uses HB exact vector similarity on the question field. There is no full-scan fallback: a
    failed QUERY (embedding endpoint, workspace, HB error) is reported on stderr by
    ``_retrieval_failed`` and returns no examples. An example whose ``question_vec`` is
    missing from Chroma is not a failure to HB — its exact scan skips the row silently — so
    an incomplete vector store reads as a smaller, healthy result; the build-time count
    gate in ``scripts/setup_dbs.py`` is what guards against that (see ``retrieval``).
    """
    from ..embedding import embed
    qv = embed(question)

    if menu:
        # Two-stage flow (issue #8), stage 1: retrieve a wider pool with the full payload.
        # server.py retains every question + SQL pair in the runtime session before projecting
        # id + question + sql_length to the model. Keeping SQL here is what
        # lets expand_examples resolve a later E-id without another database read.
        # Ranked by HB's exact scan (search_vectors), not an approximate `.near()`: Chroma's
        # HNSW graph is rebuilt per process from the write-ahead log and the rebuild is not
        # reproducible, so ANN made this lookup disagree with itself across processes on 12 of
        # 36 slots. search_vectors keeps object_id and strips question_vec, so no embedding
        # leaks to the model; the projection below picks the prompt fields.
        try:
            rows = search_vectors(ws, ExampleEntity, db_id, "question_vec", qv, top_k)
        except Exception as e:
            return _retrieval_failed("read_examples (two-stage menu)", db_id, e)
        rows_out = [{"entity_id": r.get("object_id"), "question": r.get("question"),
                     "sql": r.get("sql"), "tables": r.get("tables") or [],
                     "difficulty": r.get("difficulty")}
                    for r in rows]
        return rows_out

    try:
        rows = search_vectors(ws, ExampleEntity, db_id, "question_vec", qv, top_k)
    except Exception as e:
        return _retrieval_failed("read_examples (top-k)", db_id, e)
    # object_id rides along: callers key on it (the cross-process determinism check compares
    # returned id lists), and `.select()` is what used to drop it.
    return [{"object_id": r.get("object_id"), "question": r.get("question"), "sql": r.get("sql"),
             "tables": r.get("tables") or [], "difficulty": r.get("difficulty")}
            for r in rows]


def get_examples(
    db_id: str,
    question: str,
    top_k: int = _EXAMPLES_TOP_K,
    session_id: str = "",
) -> list:
    """Retrieve similar NL→SQL examples or a session-retained three-column menu.

    In the default two-stage flow the model-facing menu contains only ``id``, ``question``,
    and ``sql_length``; ``id`` is backed by the structured row's session-local ``virtual_id``.
    The full SQL and retrieval metadata remain in the session payload for
    :func:`expand_examples`.
    """
    from ..server import _ws_for, knowledge_sessions

    try:
        if not session_id:
            raise ValueError("session_id is required for virtual example ids")
        rows = read_examples(
            _ws_for(db_id), db_id, question, top_k=top_k, menu=True,
        )
        out = []
        for row in rows:
            if row.get("error"):
                out.append(row)
                continue
            source_id = str(row.get("entity_id") or row.get("object_id") or "")
            virtual_id = knowledge_sessions.retain(
                session_id, "example", db_id, source_id, row,
            )
            public = {
                "virtual_id": virtual_id,
                "question": row.get("question"),
                "sql_length": len(row.get("sql") or ""),
            }
            out.append(public)
        return out
    except Exception as e:
        return [{"error": f"get_examples failed: {e}"}]


get_examples.__doc__ = (
    "Retrieve a MENU of similar historical questions with exactly id, question, and sql_length "
    "(Python Unicode character count), WITHOUT SQL or persistent ids. Each id is a session-local "
    "E-id. Full question + SQL payloads and retrieval metadata are already retained in this "
    "session. Pick the few most relevant and call expand_examples(virtual_ids) with their E-ids."
)
