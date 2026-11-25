"""Content identities and exact question bindings for a fresh lab."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

import heavenbase as hb

from .lab_schema import ExperimentEntity, QAEntity
from .atlas import sqlite_content_fingerprint

_SHA256 = re.compile(r"(?:sha256:)?([0-9a-f]{64})\Z")
_QA_THREAD_LOCKS: dict[tuple[str, str], Lock] = {}
_QA_THREAD_LOCKS_GUARD = Lock()


@contextmanager
def qa_gold_lock(lab_dir: str | Path, qa_id: str) -> Iterator[None]:
    """Serialize creation and gold publication for one QA across local processes.

    Args:
        lab_dir: Directory of the lab whose QA row is being written.
        qa_id: Exact versioned QA object id.

    Yields:
        None: The cross-process lock is held through the caller's read and write.
    """
    root = Path(lab_dir).resolve()
    key = (str(root), qa_id)
    with _QA_THREAD_LOCKS_GUARD:
        thread_lock = _QA_THREAD_LOCKS.setdefault(key, Lock())
    digest = hashlib.sha256(qa_id.encode("utf-8")).hexdigest()
    path = root / ".qa-gold-locks" / f"{digest}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with thread_lock, path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def content_version(value: str) -> str:
    """Normalize a SHA-256 content identity, rejecting labels and missing values.

    Args:
        value: A lowercase hex digest, optionally prefixed by ``sha256:``.

    Returns:
        str: The ``sha256:<hex>`` content identity.

    Raises:
        ValueError: The value is not a complete SHA-256 digest.
    """
    match = _SHA256.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"invalid SHA-256 content identity: {value!r}")
    return f"sha256:{match.group(1)}"


def question_file(path: str | Path) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Load question records and name the exact source bytes used.

    Args:
        path: JSON question file, as a list or a mapping with ``questions``.

    Returns:
        tuple: Records, ``sha256:<hex>`` version, and source evidence.

    Raises:
        ValueError: The file does not contain a question list.
    """
    source = Path(path).resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    records = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"question file {source} must contain a list of question records")
    digest = hashlib.sha256(raw).hexdigest()
    return records, f"sha256:{digest}", {"path": str(source), "sha256": digest, "bytes": len(raw)}


def qa_object_id(dataset_version: str, database_version: str, qid: str) -> str:
    """Return the immutable QA id for one dataset, database, and public question id.

    Args:
        dataset_version: Question file content identity.
        database_version: Raw SQLite content identity.
        qid: Public question id, such as ``Q00371``.

    Returns:
        str: Stable HeavenBase object id.
    """
    if not re.fullmatch(r"Q[0-9]{5,}", qid):
        raise ValueError(f"invalid public question id: {qid!r}")
    return hb.utils.hash_id(content_version(dataset_version), content_version(database_version), qid)


def database_path(db_dir: str | Path, db_id: str) -> Path:
    """Resolve a raw BIRD SQLite file in either supported layout.

    Args:
        db_dir: Raw database directory.
        db_id: Database name recorded by the question file.

    Returns:
        Path: Existing SQLite source file.

    Raises:
        ValueError: The database id is unsafe.
        FileNotFoundError: Neither layout exists.
    """
    if not re.fullmatch(r"[A-Za-z0-9_]+", db_id):
        raise ValueError(f"invalid database id: {db_id!r}")
    root = Path(db_dir).resolve()
    for path in (root / f"{db_id}.sqlite", root / db_id / f"{db_id}.sqlite"):
        if path.is_file():
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"raw SQLite database {db_id!r} escapes {root}: {resolved}")
            return resolved
    raise FileNotFoundError(f"raw SQLite database {db_id!r} not found under {root}")


def database_content_version(db_path: str | Path, *, deadline_monotonic: float | None = None) -> str:
    """Fingerprint a raw SQLite file only when its main bytes are complete.

    A nonempty WAL may contain writes absent from the main SQLite file, so its
    presence makes a main-file hash insufficient as an input identity.

    Args:
        db_path: Main raw SQLite file.
        deadline_monotonic: Optional monotonic wall deadline. Hashing checks it
            between 1 MiB chunks when supplied.

    Returns:
        str: ``sha256:<hex>`` identity of the main file.

    Raises:
        ValueError: A pending WAL exists.
        OSError: The source cannot be read.
        TimeoutError: The deadline expired during hashing.
    """
    source = Path(db_path)
    wal = Path(f"{source}-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError(f"database {source} has nonempty WAL {wal}; checkpoint the source before registration")
    if deadline_monotonic is None:
        actual = f"sha256:{sqlite_content_fingerprint(str(source))}"
    else:
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            while True:
                if time.monotonic() >= deadline_monotonic:
                    raise TimeoutError(f"database fingerprint deadline expired for {source}")
                chunk = handle.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
        actual = f"sha256:{digest.hexdigest()}"
    if wal.exists() and wal.stat().st_size:
        raise ValueError(f"database {source} acquired a nonempty WAL during verification")
    return actual


def validate_database_version(db_path: str | Path, expected_version: str, *,
                              deadline_monotonic: float | None = None) -> str:
    """Reject a source whose current bytes differ from its recorded identity.

    Args:
        db_path: Main raw SQLite file.
        expected_version: Expected raw hex or ``sha256:<hex>`` digest.
        deadline_monotonic: Optional wall deadline for a bounded live hash.

    Returns:
        str: Normalized expected version when the live file matches.

    Raises:
        ValueError: A pending WAL exists or the bytes do not match.
        OSError: The source cannot be read.
    """
    expected = content_version(expected_version)
    actual = database_content_version(db_path, deadline_monotonic=deadline_monotonic)
    if actual != expected:
        raise ValueError(f"database content fingerprint mismatch for {db_path}: expected {expected}, found {actual}")
    return expected


def qa_for_trace(ws: hb.HeavenBase, trace: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve a Trace's exact versioned QA and reject missing or drifted bindings.

    Args:
        ws: Open lab workspace.
        trace: Trace row with required ``qa_id``, public ``qid`` and ``db_id``.

    Returns:
        dict: The bound QA row.

    Raises:
        ValueError: The trace lacks a valid exact QA reference.
    """
    qa_id = trace.get("qa_id")
    if not isinstance(qa_id, str) or not qa_id:
        raise ValueError(f"trace {trace.get('exp_id')} {trace.get('qid')} has no qa_id")
    row = ws.get(qa_id, QAEntity)
    if row is None:
        raise ValueError(f"trace {trace.get('exp_id')} {trace.get('qid')} references missing QA {qa_id}")
    qa = row if isinstance(row, dict) else row.to_dict()
    if qa.get("name") != trace.get("qid") or qa.get("db_id") != trace.get("db_id"):
        raise ValueError(f"trace {trace.get('exp_id')} {trace.get('qid')} QA {qa_id} has mismatched question or database")
    if qa.get("object_id") != qa_object_id(str(qa.get("dataset_version")), str(qa.get("database_version")), str(qa.get("name"))):
        raise ValueError(f"trace {trace.get('exp_id')} {trace.get('qid')} QA {qa_id} has invalid version identity")
    exp = ws.get(str(trace.get("exp_id") or ""), ExperimentEntity)
    if exp is None:
        raise ValueError(f"trace {trace.get('exp_id')} {trace.get('qid')} has no experiment input provenance")
    experiment = exp if isinstance(exp, dict) else exp.to_dict()
    provenance = experiment.get("provenance")
    inputs = provenance.get("inputs") if isinstance(provenance, Mapping) else None
    db_versions = inputs.get("db_fingerprints") if isinstance(inputs, Mapping) else None
    if (not isinstance(db_versions, Mapping)
            or inputs.get("dataset_version") != qa["dataset_version"]
            or db_versions.get(qa["db_id"]) != qa["database_version"]):
        raise ValueError(f"trace {trace.get('exp_id')} {trace.get('qid')} QA {qa_id} conflicts with experiment input provenance")
    return qa
