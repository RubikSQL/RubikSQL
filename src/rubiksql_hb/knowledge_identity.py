"""Read-only identities of the model-visible knowledge actually consumed by a run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import sqlite3
from typing import Any, Iterable


class KnowledgeIdentityError(ValueError):
    """A scientific source cannot be matched to an explicit knowledge workspace."""


def question_subset_identity(selected: list[dict[str, Any]], original: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify an answering roster against the knowledge build's question versions.

    Args:
        selected: Normalized answering records; gold and scoring metadata are ignored.
        original: Normalized original build records, independently checked by its ledger.

    Returns:
        Counts and content digests of the original and selected question identities.

    Raises:
        KnowledgeIdentityError: Either roster is empty, duplicates IDs, or a selected
            ID, database, question or evidence differs from the original build.
    """
    def identities(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = row.get("id")
            if not isinstance(key, str) or not key or key in result:
                raise KnowledgeIdentityError("Knowledge question rosters require unique nonempty IDs")
            if row.get("question_id") is not None:
                alias = str(row["question_id"])
                canonical = f"Q{int(alias):05d}" if alias.isdigit() else alias
                if canonical != key:
                    raise KnowledgeIdentityError("Knowledge question ID aliases disagree: " + key)
            if not isinstance(row.get("database"), str) or not row["database"] or not isinstance(row.get("question"), str) or not row["question"]:
                raise KnowledgeIdentityError("Knowledge questions require a database and question text")
            preferences = row.get("context", {}).get("user_profile", {}).get("preferences", [])
            evidence = row.get("evidence", preferences)
            evidence = [evidence] if isinstance(evidence, str) and evidence else evidence or []
            result[key] = {"id": key,
                           "database": row["database"], "question": row["question"],
                           "evidence": evidence, "preferences": preferences}
        if not result:
            raise KnowledgeIdentityError("Knowledge question rosters must be nonempty")
        return result

    complete, subset = identities(original), identities(selected)
    for key, row in subset.items():
        if complete.get(key) != row:
            raise KnowledgeIdentityError("Answering question is not an exact knowledge-build subset: " + key)
    def digest(rows: dict[str, dict[str, Any]]) -> str:
        return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"selected_count": len(subset), "build_count": len(complete),
            "selected_sha256": digest(subset), "build_sha256": digest(complete)}


def recorded_knowledge_path(value: object) -> str:
    """Validate a retained absolute path from either producer operating system.

    The spelling identifies the producer location, not the current local copy.
    Callers must compare the copy's complete knowledge fingerprints separately.

    Args:
        value: Original manifest's knowledge root, without rewriting its bytes.

    Returns:
        The canonical, absolute producer path.

    Raises:
        KnowledgeIdentityError: The path is relative, ambiguous or noncanonical.
    """
    if isinstance(value, str):
        for kind in (PurePosixPath, PureWindowsPath):
            path = kind(value)
            if path.is_absolute() and str(path) == value and ".." not in path.parts:
                return value
    raise KnowledgeIdentityError("recorded knowledge root must be a canonical absolute producer path")


def knowledge_fingerprint(database: Path, *, embedding_model: str, strict: bool = False) -> str:
    """Hash the existing knowledge payload without opening a writable database.

    Strict publication checks refuse missing/incomplete schemas; ordinary cache
    callers preserve the historical empty-on-unavailable result and byte spelling.
    """
    database = Path(database).resolve()
    if not database.is_file():
        if strict:
            raise KnowledgeIdentityError("recorded knowledge database is missing")
        return ""
    conn = None
    try:
        conn = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        conn.execute("PRAGMA query_only=ON")
        rules = conn.execute(
            'SELECT scope, title, "desc", tags, triggers, example '
            "FROM rule_entity ORDER BY object_id"
        ).fetchall()
        examples = conn.execute(
            "SELECT question, sql, tables FROM example_entity ORDER BY object_id"
        ).fetchall()
        example_fields = {row[1] for row in conn.execute("PRAGMA table_info(example_entity)")}
        contracts = conn.execute(
            "SELECT object_id, contract FROM example_entity ORDER BY object_id"
        ).fetchall() if "contract" in example_fields else []
        synonyms = conn.execute(
            "SELECT synonym, canonical, table_name, col_name FROM synonym_entity ORDER BY object_id"
        ).fetchall()
        col_descs = conn.execute(
            'SELECT table_name, name, "desc" FROM column_entity ORDER BY object_id'
        ).fetchall()
        try:
            col_profiles = conn.execute(
                "SELECT table_name, name, profile_desc FROM column_entity ORDER BY object_id"
            ).fetchall()
        except sqlite3.OperationalError:
            if strict:
                raise   # legacy workspace without the column
            col_profiles = []
        try:
            table_descs = conn.execute(
                'SELECT name, "desc", profile_desc FROM table_entity ORDER BY object_id'
            ).fetchall()
        except sqlite3.OperationalError:
            if strict:
                raise   # minimal/legacy workspace without the table —
            # an unguarded miss here nulled the WHOLE fingerprint to "" and every
            # knowledge state hashed equal (caught by the fingerprint test suite).
            table_descs = []
        try:
            overviews = conn.execute(
                "SELECT db_id, profile_desc FROM database_entity ORDER BY object_id"
            ).fetchall()
        except sqlite3.OperationalError:
            if strict:
                raise   # pre-08-05 workspace without the table
            overviews = []
        payload = {"rules": rules, "examples": examples, "synonyms": synonyms,
                   # The vectors served by get_examples/get_rules/search_* are derived
                   # from these texts under ONE embedding model; a model change moves
                   # every retrieval without touching a stored text, so the model name is
                   # part of the knowledge identity (added 2026-09-23 with the DashScope
                   # switch - every entry keyed before it stops matching, by design).
                   "embedding_model": embedding_model}
        if any(value not in (None, "", "{}", "null") for _, value in contracts):
            payload["example_contracts"] = contracts
        # Official column documentation is knowledge: a desc-backfilled state must
        # never replay cache entries produced without it. Added ONLY when non-empty,
        # so an empty-desc state hashes like one without the column. That once kept
        # the pre-desc production cache replayable; the unconditional embedding_model
        # salt above re-keyed EVERY fingerprint on 2026-09-23, so no entry written
        # before it replays now, whatever its desc state. The only-when-non-empty
        # rule buys byte-stability among post-salt states only.
        if any((d or "").strip() for _, _, d in col_descs):
            payload["column_desc"] = col_descs
        # Column profiles are served verbatim by get_col_info — the system prompt even
        # instructs per-column calls — so they are model-visible knowledge exactly like
        # the table profiles below. This hash omitted them until 08-07 (violating this
        # function's own docstring), so swapping profiles did not re-key the cache and
        # any profile A/B replayed stale answers. Same only-when-non-empty pattern.
        if any((p or "").strip() for _, _, p in col_profiles):
            payload["column_profile"] = col_profiles
        # Table profiles and the database overview render inside get_db_info/get_tab_info,
        # so they are model-visible knowledge and must move this hash. Same only-when-
        # non-empty pattern: states without profiles keep byte-identical fingerprints.
        # database_entity may not exist in a pre-08-05 workspace — probed separately so
        # its absence cannot hide a populated table_desc.
        if any((d or "").strip() or (p or "").strip() for _, d, p in table_descs):
            payload["table_desc"] = table_descs
        if any((o or "").strip() for _, o in overviews):
            payload["db_overview"] = overviews
        content = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode()).hexdigest()
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        if strict:
            raise KnowledgeIdentityError("recorded knowledge database is unreadable or incomplete") from exc
        return ""
    finally:
        if conn is not None:
            conn.close()


def knowledge_fingerprints(root: Path, dbs: Iterable[str], *, embedding_model: str) -> dict[str, str]:
    """Verify an exact per-database knowledge root for scientific publication."""
    root = Path(root).resolve()
    result = {}
    for db in sorted(set(dbs)):
        if not isinstance(db, str) or not db or db in {".", ".."} or "/" in db or "\\" in db:
            raise KnowledgeIdentityError("knowledge database identity is invalid")
        database = (root / db / f"{db}.db").resolve()
        if not database.is_relative_to(root):
            raise KnowledgeIdentityError("knowledge database escapes its recorded root")
        result[db] = knowledge_fingerprint(database, embedding_model=embedding_model, strict=True)
    return result
