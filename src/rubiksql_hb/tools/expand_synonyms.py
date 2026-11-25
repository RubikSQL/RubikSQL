"""Pinned implementation of expand_synonyms."""
import sys
import heavenbase as hb
from ..schema import SynonymEntity

_EMPTY_SYNONYM_REPORTED: set[str] = set()


def _no_synonyms(db_id: str) -> None:
    """Report, once per db_id, that this database has no synonym rows at all.

    An empty ``synonym_entity`` is indistinguishable from a working one: ``resolve_synonyms``
    answers with the question verbatim either way, which is also exactly what a correct
    "nothing needed rewriting" answer looks like — the same silent-empty failure mode
    ``_retrieval_failed`` exists to remove. It matters here because the candidate prompt makes
    ``expand_synonyms`` a MANDATORY Phase 1 step, so a database with zero synonyms spends one
    tool round-trip per candidate on a guaranteed identity function and nothing says so.

    A thread race can at worst print the line twice (same trade-off as
    ``nl2sql._cache_report_broken``); it can never suppress it or spam it.
    """
    if db_id in _EMPTY_SYNONYM_REPORTED:
        return
    _EMPTY_SYNONYM_REPORTED.add(db_id)
    print(f"[knowledge] resolve_synonyms found ZERO synonyms for db={db_id!r} — every question "
          f"on this database is returned VERBATIM, so expand_synonyms is a no-op here. "
          f"save_synonym is the only writer of this table. Reported once per database.",
          file=sys.stderr, flush=True)


def resolve_synonyms(
    ws: hb.HeavenBase,
    db_id: str,
    text: str,
) -> str:
    """
    Replace known synonyms in a natural language query with their canonical names.
    Simple string substitution pass before SQL generation.

    A database with no stored synonyms gets ``text`` back unchanged and is reported once on
    stderr by :func:`_no_synonyms`; the return value is unaffected either way.
    """
    synonyms = (
        ws.query(SynonymEntity)
        .where({"db_id": {"$eq": db_id}})
        .select("synonym", "canonical")
        .execute()
        .rows()
    )
    if not synonyms:
        _no_synonyms(db_id)
    for s in synonyms:
        text = text.replace(s["synonym"], s["canonical"])
    return text


def expand_synonyms(db_id: str, question: str) -> str:
    """Replace known synonyms in a question with canonical schema names."""
    from ..server import _ws_for

    return resolve_synonyms(_ws_for(db_id), db_id, question)
