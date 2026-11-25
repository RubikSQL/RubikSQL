"""Pinned implementation of get_rules."""
import re
from typing import Any
import heavenbase as hb
from ..embedding import embed, rule_with_example
from ..retrieval import near_max_top_k, search_vectors
from ..schema import RuleEntity
from ..knowledge_host import RULES_RETRIEVAL_FAILED_ERROR, _normalize_rule_text, _rule_projection
from ..prompts import current as _served_release
from .knowledge_errors import _retrieval_failed

_RULES_TOP_K = _served_release().knob("rules_top_k")
RANK_DESC, RANK_TITLE, RANK_TEXT = 0.6, 0.4, 0.5
_RULE_STOPWORDS = frozenset("""
a an the this that these those there here
is are was were be been being am do does did done
of in on at to for from by with without into over under between among as
and or but not no nor if then than so such
i you he she it we they me him her them my your his its our their
what which who whom whose when where why how
have has had can could will would shall should may might must
all any both each few more most other some only own same very
s t don now just also
""".split())


def _rule_tokens(text: str) -> set[str]:
    """Return Latin words, numbers, and individual CJK characters from normalized text."""
    return set(re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", _normalize_rule_text(text)))


def _content_tokens(text: str) -> set[str]:
    """Return the tokens of ``text`` that carry retrieval signal.

    Function words are dropped because they match every rule equally: an overlap term
    computed over them ranks rules by how much prose they contain rather than by how well
    that prose fits the question.
    """
    return _rule_tokens(text) - _RULE_STOPWORDS


def _phrase_hit(question: str, field: str) -> bool:
    """Return whether ``field`` appears in ``question`` as a complete phrase.

    Requires at least two content tokens. A single-token field is already credited by the
    overlap term, and rewarding it again here would rank a rule carrying one broad tag above
    a rule carrying four precise ones \u2014 a difference in field length, not in relevance.
    """
    if len(_content_tokens(field)) < 2:
        return False
    query = _normalize_rule_text(question)
    value = _normalize_rule_text(field)
    return bool(query) and bool(value) and (value in query or query in value)


def _field_score(question: str, field: str) -> float:
    """Return a deterministic lexical relevance score for one rule field.

    Overlap is normalized by the question's own content tokens, so the four fields share a
    denominator while their numerators still scale with field length \u2014 which is what the
    per-field weights in :func:`_text_score` exist to offset.

    Args:
        question: Natural-language retrieval query.
        field: One rule field's text.

    Returns:
        float: ``overlap + 2.0`` when the field also matches as a complete phrase.
    """
    query_tokens = _content_tokens(question)
    value_tokens = _content_tokens(field)
    if not query_tokens or not value_tokens:
        return 0.0
    overlap = len(query_tokens & value_tokens) / len(query_tokens)
    return overlap + (2.0 * float(_phrase_hit(question, field)))


def _text_score(question: str, row: dict[str, Any]) -> float:
    """Score a rule on its semantic fields after vector candidate generation.

    The weights offset field length: :func:`_field_score` normalizes by the question, so a
    four-token title and a hundred-token ``desc + example`` reach very different overlaps
    for the same quality of match. Retrieval returns ``2 * k`` of a database's rules, so
    this ordering decides which rules reach the model, not merely how they are ordered.

    Two fields, not four. Triggers have their own delivery path — :func:`match_rule_triggers`
    runs on the question and on every model turn and injects each match in full, so scoring
    the same match here spent a retrieval slot re-delivering a rule the conversation already
    held. Tags are a controlled facet vocabulary (``aggregate``, ``join``, ``ordering``)
    matched against natural language by lexical overlap, which is why they scored 0.00 at the
    99th percentile across 37327 question/rule pairs on the shipped corpus.
    """
    return (
        4.0 * _field_score(question, str(row.get("title") or ""))
        + _field_score(
            question,
            rule_with_example(
                str(row.get("desc") or ""),
                str(row.get("example") or ""),
            ),
        )
    )


def _rank_score(question: str, row: dict[str, Any]) -> float:
    """Combine both vector branches and the scalar text score into one rank.

    Replaces a lexicographic key whose primary term was the text score. That key let any
    non-zero lexical overlap outrank any cosine, and measured over the shipped corpus 74% of
    those promotions rested on a single shared content word — weak evidence overriding
    strong, by construction rather than by tuning. Summing lets strong lexical evidence win
    and stops weak evidence from winning at all.

    ``max`` over the two branches is likewise gone. It scored a rule both branches agree on
    exactly as it scored one branch's favourite, discarding the agreement the two views had
    already computed.

    Args:
        question: Natural-language retrieval query.
        row: Merged candidate carrying ``_title_score`` and ``_desc_score``.

    Returns:
        float: Rank score; higher sorts first. Callers break ties on ``object_id``.
    """
    return (
        RANK_DESC * float(row.get("_desc_score") or 0.0)
        + RANK_TITLE * float(row.get("_title_score") or 0.0)
        + RANK_TEXT * _text_score(question, row)
    )


def read_rules(
    ws: hb.HeavenBase,
    db_id: str,
    question: str,
    k: int = 10,
    exclude_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve rules through two vector branches followed by scalar text ranking.

    ``title_vec`` and ``desc_vec`` each nominate their top ``k`` through HB's exact scan, so
    at most ``2 * k`` distinct rules enter the rerank. Every candidate is then rescored on
    BOTH branches by a second exact query restricted to the union — one arriving through a
    single branch would otherwise carry no similarity from the other, and a weighted sum
    cannot use an absence. ``desc_vec`` embeds ``example +
    desc`` using an explicit ``[Rule]`` separator. The two cosines and the scalar text
    score are combined by :func:`_rank_score`, ties broken on entity id, and the result
    trimmed to ``k``. Tags and triggers are returned but do not rank: triggers have their
    own delivery path through :func:`match_rule_triggers`, and tags are a controlled
    vocabulary that lexical overlap against a question cannot match.

    ``k`` is the result width. It used to be the per-branch width with the trim set to
    ``2 * k`` — so ``k=10`` returned 20 records, roughly 3,400 tokens, from a parameter
    whose name promises 10. Callers that genuinely want 20 now ask for 20. The width is
    capped at half of HB's ``near`` maximum (:func:`~rubiksql_hb.retrieval.near_max_top_k`,
    1000 in the vendored build): the two branches' union must fit the one exact rescoring
    query below, which HB would otherwise reject and this function report as a failure.

    ``exclude_ids`` drops rules before ranking rather than after. A rule already delivered
    by the conversation's trigger hook is removed downstream anyway, but by then it has won
    one of the ``k`` slots and the rule it displaced is gone: the caller ends up with fewer
    rules than it asked for, and never sees the one it lost.

    Args:
        ws: Workspace holding the rules.
        db_id: Database whose rules may be returned.
        question: Natural-language retrieval query.
        k: Result width, and each branch's nomination width; defaults to 10.
        exclude_ids: Entity ids to leave out, typically rules already injected this run.

    Returns:
        list[dict[str, Any]]: At most ``k`` prompt-safe rule records. A retrieval whose
        machinery ITSELF failed (embedding endpoint, workspace, vector query) answers
        ``[{"error": RULES_RETRIEVAL_FAILED_ERROR}]`` instead of an empty list, so a
        backend outage is never mistaken for an empty rule store downstream.
    """
    excluded = {str(entity_id) for entity_id in exclude_ids or []}
    width = max(0, int(k))
    if width == 0:
        return []
    try:
        width = min(width, near_max_top_k(ws) // 2)
        query_vec = embed(question)
        branches = {"title": "title_vec", "desc": "desc_vec"}
        # Exclusions belong to the HB filter, ahead of each branch's top-k, so an
        # already-injected rule cannot occupy one of a branch's k nominations and push out a
        # rule the caller has never seen. (`$nin` is not registered in the vendored build.)
        excluding = {"$not": {"object_id": {"$in": sorted(excluded)}}} if excluded else None
        ranked_by = {
            branch: search_vectors(ws, RuleEntity, db_id, field, query_vec, width, where=excluding)
            for branch, field in branches.items()
        }
        # Each branch nominates its own top-k; the union is at most 2 * k distinct rules.
        merged: dict[str, dict[str, Any]] = {}
        for rows in ranked_by.values():
            for row in rows:
                entity_id = str(row.get("object_id") or "")
                if entity_id:
                    merged.setdefault(entity_id, dict(row))
        if not merged:
            return []
        # A candidate nominated by one branch must receive the other's real cosine: a
        # weighted sum cannot use an absence, and scoring it zero would mark a rule as
        # unrelated to a view that merely did not rank it in its own top-k. The union is
        # rescored exactly on both vectors, restricted to the candidates by object_id.
        candidate_ids = list(merged)
        scores = {
            branch: {
                str(row["object_id"]): float(row["score"])
                for row in search_vectors(
                    ws, RuleEntity, db_id, field, query_vec, len(candidate_ids),
                    where={"object_id": {"$in": candidate_ids}}, exact=True,
                )
            }
            for branch, field in branches.items()
        }
        # Missing hydration is an integrity failure, not a semantic score of zero.
        for entity_id, row in merged.items():
            for branch in branches:
                if entity_id not in scores[branch]:
                    raise RuntimeError(
                        f"rule candidate {entity_id!r} could not be scored on {branch}_vec"
                    )
                row[f"_{branch}_score"] = scores[branch][entity_id]
    except Exception as e:
        _retrieval_failed("get_rules", db_id, e)
        # Not the shared empty return: an empty list is indistinguishable from a healthy
        # zero-match, and a live outage shipped 200 degraded answers that way with only
        # stderr knowing. The error row rides the existing error-row channel end to end
        # (server passthrough → "Knowledge errors" render → transcript), where eval_bird
        # keys the row's ``degraded_retrieval`` flag on this exact text.
        return [{"error": RULES_RETRIEVAL_FAILED_ERROR}]

    ranked = sorted(
        merged.values(),
        key=lambda row: (-_rank_score(question, row), str(row.get("object_id") or "")),
    )
    return [_rule_projection(row) for row in ranked[:width]]


def get_rules(
    db_id: str,
    question: str,
    k: int = _RULES_TOP_K,
    session_id: str = "",
    exclude_ids: list[str] = [],
) -> list:
    """Retrieve and rerank rules with session-local virtual ids.

    Returns at most k rules — the same number you ask for. Two vector branches each
    contribute k candidates and the merged set is reranked before being cut back to k, so a
    larger k widens the search as well as the answer.
    """
    from ..server import knowledge_sessions, _ws_for

    try:
        if not session_id:
            raise ValueError("session_id is required for virtual rule ids")
        # Same virtual-to-source resolution match_rule_triggers performs, for the same
        # reason: the caller holds this run's R-ids, while the store is keyed by entity id.
        persistent_excludes = [
            source_id
            for virtual_id in exclude_ids
            if (
                source_id := knowledge_sessions.source_id(
                    session_id, "rule", db_id, virtual_id,
                )
            ) is not None
        ]
        rows = read_rules(
            _ws_for(db_id), db_id, question, k=k, exclude_ids=persistent_excludes,
        )
        out = []
        for row in rows:
            if row.get("error"):
                out.append(row)
                continue
            source_id = str(row.get("entity_id") or "")
            virtual_id = knowledge_sessions.retain(
                session_id, "rule", db_id, source_id, row,
            )
            out.append({
                "virtual_id": virtual_id,
                **{key: value for key, value in row.items() if key != "entity_id"},
            })
        return out
    except Exception as e:
        return [{"error": f"get_rules failed: {e}"}]
