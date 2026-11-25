"""Immutable cache provenance, separate from replaceable lookup keys.

Only the evaluation worker binds a producer identity. Direct API callers and legacy
cache entries stay unattributed; replay never attributes old work to its consumer.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import asyncio
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from typing import Any, Callable
from .knowledge_failures import degraded_record

SCHEMA = "rubiksql.cache-source/1"
REF = "llm_source:"
_EXECUTION: ContextVar[dict[str, Any] | None] = ContextVar("cache_execution", default=None)
_CAPTURE: ContextVar[AttemptCapture | None] = ContextVar("attempt_capture", default=None)
_OBSERVER: ContextVar[str | None] = ContextVar("attempt_observer", default=None)


class AttemptCapture:
    """An attempt-local mailbox of immutable snapshots made by their owning event loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._callbacks: dict[str, tuple[asyncio.AbstractEventLoop, Callable[[], dict[str, Any]]]] = {}
        self._records: dict[str, dict[str, Any]] = {}
        self._closed = False

    def register(self, key: str, callback: Callable[[], dict[str, Any]]) -> None:
        with self._lock:
            if not self._closed:
                self._callbacks[key] = (asyncio.get_running_loop(), callback)

    def checkpoint(self, reason: str, *, key: str | None = None, reply: Any = None) -> None:
        """Copy live state on its owner loop; readers never touch mutable agent dictionaries."""
        loop = asyncio.get_running_loop()
        with self._lock:
            callbacks = [(name, fn) for name, (owner, fn) in self._callbacks.items()
                         if owner is loop and (key is None or name == key)]
        for name, callback in callbacks:
            try:
                record = deepcopy(callback())
                if isinstance(reply, Mapping):
                    record.setdefault("messages", []).append(deepcopy(dict(reply)))
                record.update(transcript_status="interrupted", capture_reason=reason)
            except (KeyError, TypeError, ValueError, RuntimeError):
                # The last successful checkpoint remains available when metadata fails.
                continue
            with self._lock:
                if not self._closed:
                    self._records[name] = record

    def release(self, key: str, record: Mapping[str, Any] | None = None) -> None:
        """Drop live callbacks when a role ends, retaining only its detached final snapshot."""
        frozen = deepcopy(dict(record)) if record is not None else None
        with self._lock:
            self._callbacks.pop(key, None)
            if not self._closed and frozen is not None:
                self._records[key] = frozen

    def snapshot(self, reason: str, timeout: float = 1.0) -> list[dict[str, Any]]:
        """Request a bounded checkpoint; a blocked loop leaves its earlier observed prefix."""
        with self._lock:
            loops = {loop for loop, _ in self._callbacks.values()}
        deadline = time.monotonic() + max(0.0, min(timeout, 1.0))
        events = []
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        for loop in loops:
            done = threading.Event()
            def collect(done=done):
                try:
                    self.checkpoint(reason)
                finally:
                    done.set()
            try:
                if loop is current_loop:
                    collect()
                else:
                    loop.call_soon_threadsafe(collect)
                events.append(done)
            except RuntimeError:
                pass  # A stopped loop cannot provide more than its last checkpoint.
        for done in events:
            done.wait(max(0.0, deadline - time.monotonic()))
        with self._lock:
            records = deepcopy(list(self._records.values()))
        for record in records:
            if record.get("transcript_status") == "interrupted":
                record["last_checkpoint_reason"] = record.get("capture_reason")
                record["capture_reason"] = reason
        return records

    def close(self) -> None:
        """Release this attempt's callbacks and snapshots; late workers cannot republish."""
        with self._lock:
            self._closed = True
            self._callbacks.clear()
            self._records.clear()


def observe(producer: Mapping[str, Any], callback: Callable[[], dict[str, Any]]) -> tuple[Any, ...] | None:
    """Register an active role's snapshot callback in the current evaluation attempt."""
    capture = _CAPTURE.get()
    if capture is None:
        return None
    key = str(producer["agent_attempt"])
    capture.register(key, callback)
    return capture, key, _OBSERVER.set(key)


def checkpoint(reason: str, reply: Any = None) -> None:
    """Publish one immutable prefix at model-message boundaries, never per token."""
    capture = _CAPTURE.get()
    if capture is not None:
        capture.checkpoint(reason, key=_OBSERVER.get(), reply=reply)


def finish(observer: tuple[Any, ...] | None, record: Mapping[str, Any] | None = None) -> None:
    """Release an observer on its owner task, including cancellation and error paths."""
    if observer is not None:
        capture, key, token = observer
        capture.release(key, record)
        _OBSERVER.reset(token)


def note(role: str, record: Mapping[str, Any]) -> None:
    """Publish the latest state of a non-agent step under ``role``, replacing its earlier state.

    Args:
        role (str): The step's capture role; agents use ``candidate``, ``judge`` and ``reviewer``.
        record (Mapping[str, Any]): The step's state, copied when published.
    """
    capture = _CAPTURE.get()
    if capture is not None:
        capture.release(role, {**record, "execution": {"role": role}})


def noted(records: list[dict[str, Any]], role: str) -> dict[str, Any]:
    """Return the state :func:`note` published under ``role`` in an attempt snapshot, else ``{}``."""
    for record in records:
        if (record.get("execution") or {}).get("role") == role:
            return {key: value for key, value in record.items() if key != "execution"}
    return {}


def captured_prediction(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Project observed role snapshots into the evaluator's existing failure-record shape."""
    candidates: dict[int, list[dict[str, Any]]] = {}
    roles: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        origin = record.get("execution") or {}
        role = origin.get("role")
        if role == "candidate" and type(origin.get("slot")) is int:
            candidates.setdefault(origin["slot"], []).append(record)
        elif role in ("judge", "reviewer"):
            roles.setdefault(role, []).append(record)
    def latest(items):
        return {**items[-1], "attempts": [*items[-1].get("attempts", []), *items[:-1]]}
    debug = [latest(candidates[slot]) if slot in candidates else {"transcript_status": "not_run", "messages": []}
             for slot in range(1, max(candidates, default=0) + 1)]
    result: dict[str, Any] = {"candidates_debug": debug,
                              "candidates": [{"sql": item.get("sql", ""), "error": item.get("error")} for item in debug]}
    if "judge" in roles:
        result["judge_debug"] = latest(roles["judge"])
    if "reviewer" in roles:
        result["reviewer_debug"] = {"active": True, "changed": False, "route_debug": latest(roles["reviewer"])}
    return result


def _canonical(value: Any) -> str:
    from .model_render import binary_json_default

    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=binary_json_default)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@contextmanager
def evaluation(run: str, db: str, qid: str, attempt: int, capture: AttemptCapture | None = None) -> Iterator[None]:
    """Bind the actual executor's question attempt, including its async children."""
    token = _EXECUTION.set({"run": run, "db": db, "qid": qid, "attempt": attempt})
    capture_token = _CAPTURE.set(capture)
    try:
        yield
    finally:
        _CAPTURE.reset(capture_token)
        _EXECUTION.reset(token)


def identity(*, db: str, role: str, slot: int | None, release: str,
             knowledge_fingerprint: str) -> dict[str, Any]:
    """Return producer metadata; absent evaluation context stays explicitly unknown."""
    context = _EXECUTION.get() or {}
    return {"run": context.get("run"), "db": db, "qid": context.get("qid"),
            "attempt": context.get("attempt"), "role": role, "slot": slot,
            "agent_attempt": uuid.uuid4().hex,
            "release": release, "knowledge_fingerprint": knowledge_fingerprint}


def owned_by_evaluation(record: Mapping[str, Any]) -> bool:
    """Whether this question's run produced the record, across retry attempts.

    Immutable provenance takes precedence over consumption metadata. Unattributed
    records and cache consumers without a source cannot establish ownership.

    Args:
        record (Mapping[str, Any]): The producing role's candidate, judge or reviewer record.

    Returns:
        bool: True when the producer matches the bound run, database and question.
    """
    context = _EXECUTION.get() or {}
    source = record.get("cache_source")
    if source is not None:
        if not valid_descriptor(source):
            return False
        origin = source["identity"]
    else:
        origin = record.get("execution") or {}
        if origin.get("source_kind") == "cache":
            return False
    return all(context.get(key) and context[key] == origin.get(key) for key in ("run", "db", "qid"))


def content_sha(record: Mapping[str, Any]) -> str:
    """Hash the immutable answer/transcript, excluding current reuse and cost metadata."""
    prompt = record.get("prompt_identity") if isinstance(record.get("prompt_identity"), Mapping) else {}
    return _sha({"messages": record.get("messages", []), "reasoning": record.get("reasoning", []),
                 "sql": record.get("sql"), "prompt": {key: prompt.get(key) for key in ("release", "fired", "msg")}})


def retain_source(store: Any, record: dict[str, Any], *, payload: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Best-effort archive; unavailable provenance never discards an observed conversation."""
    if store is None:
        return None
    producer = record.get("execution") or {}
    try:
        source = store.write_source(record if payload is None else payload, producer)
    except Exception as error:
        record["cache_source_error"] = {"type": type(error).__name__, "message": str(error)}
        return None
    if source is not None:
        record["cache_source"] = source
    elif producer.get("run") and producer.get("qid"):
        # TurnStore fails soft after reporting storage errors. Its None result
        # carries no exact cause, so retain only the observed unavailability.
        record["cache_source_error"] = {"type": "SourceUnavailable", "message": "immutable source archive returned no record"}
    return source


def descriptor(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the content-addressed proof header retained in raw and HB records."""
    return {key: document[key] for key in ("schema", "id", "identity", "payload_sha256", "content_sha256")}


def valid_descriptor(source: Any, *, attributed: bool = False) -> bool:
    """Check the header digest and, when required, a complete producer identity."""
    if not isinstance(source, Mapping) or source.get("schema") != SCHEMA:
        return False
    keys = ("schema", "identity", "payload_sha256", "content_sha256")
    if any(key not in source for key in keys) or source.get("id") != _sha({key: source[key] for key in keys}):
        return False
    origin = source.get("identity")
    if not isinstance(origin, Mapping):
        return False
    if attributed:
        if any(not isinstance(origin.get(key), str) or not origin[key] for key in
               ("run", "db", "qid", "role", "release", "knowledge_fingerprint")):
            return False
        if type(origin.get("attempt")) is not int or origin["attempt"] < 1:
            return False
        if origin["role"] == "candidate" and (type(origin.get("slot")) is not int or origin["slot"] < 1):
            return False
    return True


def write(conn: sqlite3.Connection, record: Mapping[str, Any], producer: Mapping[str, Any]) -> dict[str, Any]:
    """Append a full source and verify stored bytes; lookup-key replacement cannot alter it."""
    payload = {key: value for key, value in record.items() if key not in ("cache_source", "execution")}
    header = {"schema": SCHEMA, "identity": dict(producer), "payload_sha256": _sha(payload),
              "content_sha256": content_sha(payload)}
    document = {**header, "id": _sha(header), "payload": payload}
    conn.execute("CREATE TABLE IF NOT EXISTS llm_source (id TEXT PRIMARY KEY, document TEXT NOT NULL)")
    conn.execute("INSERT OR IGNORE INTO llm_source (id, document) VALUES (?, ?)", (document["id"], _canonical(document)))
    stored = read(conn, REF + document["id"])
    if stored != document:
        raise ValueError("immutable cache source failed read-back verification")
    return descriptor(document)


def read(conn: sqlite3.Connection, ref: str) -> dict[str, Any] | None:
    """Read and hash-verify a source; corrupted or missing evidence is never trusted."""
    if not ref.startswith(REF):
        return None
    row = conn.execute("SELECT document FROM llm_source WHERE id = ?", (ref[len(REF):],)).fetchone()
    if row is None:
        return None
    from .model_render import restore_sql_results

    document = restore_sql_results(json.loads(row[0]))
    if (not valid_descriptor(document) or document["id"] != ref[len(REF):]
            or not isinstance(document.get("payload"), dict)
            or _sha(document["payload"]) != document["payload_sha256"]
            or content_sha(document["payload"]) != document["content_sha256"]):
        raise ValueError("immutable cache source digest mismatch")
    return document


def candidate_cache_key(unit: Mapping[str, Any], document: Mapping[str, Any] | None, *, prompt_keyed: bool) -> str:
    """Resolve a candidate's lookup key from its digest-verified immutable evidence.

    Args:
        unit (Mapping[str, Any]): The candidate projection, including its source proof and keys.
        document (Mapping[str, Any] | None): The corresponding archived ``llm_source``
            document, or ``None`` when the archive is missing.
        prompt_keyed (bool): Whether this candidate's key includes a prompt term. A legacy
            record without its full key can use its memo base key only when this is false.

    Returns:
        str: The exact whole-candidate cache key, distinct from the turn memo's base key.

    Raises:
        ValueError: The archive or projected keys disagree, or legacy prompt-keyed evidence
            cannot establish the full lookup key.
    """
    if document is None:
        raise ValueError("candidate source document is missing from llm_source")
    payload = document.get("payload")
    if (not valid_descriptor(document, attributed=True) or not isinstance(payload, Mapping)
            or _sha(payload) != document["payload_sha256"]
            or content_sha(payload) != document["content_sha256"]):
        raise ValueError("candidate source document fails its digest check")
    if unit.get("source_proof") != descriptor(document):
        raise ValueError("candidate source document does not match its retained proof")
    base_key = unit.get("memo_base_key")
    if not isinstance(base_key, str) or not base_key or payload.get("memo_base_key") != base_key:
        raise ValueError("candidate memo base key does not match its immutable source")
    key, archived_key = unit.get("cache_key"), payload.get("cache_key")
    if key is None and archived_key is None:
        if prompt_keyed:
            raise ValueError("legacy prompt-keyed candidate has no verifiable full cache key")
        return base_key
    if not isinstance(key, str) or not key or key != archived_key:
        raise ValueError("candidate full cache key does not match its immutable source")
    return key


def replay(record: Mapping[str, Any], *, role: str | None = None, slot: int | None = None,
           capture: bool = False) -> dict[str, Any]:
    """Mark a whole-cache consumption without changing its producer or charging old usage."""
    result = dict(record)
    if degraded_record(record):
        result["degraded_retrieval"] = True
    prompt = dict(record.get("prompt_identity") or {})
    source = record.get("cache_source")
    turns = sum(message.get("role") == "assistant" for message in (record.get("messages") or []) if isinstance(message, Mapping))
    verified = valid_descriptor(source) and source["content_sha256"] == content_sha(record)
    if verified:
        prompt["replay"] = {"turns": turns, "replayed": turns, "diverged_at": None, "cause": None,
                            "source": REF + source["id"],
                            "sources": [{"source": source, "idx": index} for index, message in enumerate(record.get("messages") or [])
                                        if isinstance(message, Mapping) and message.get("role") == "assistant"]}
    result.update(usage={}, duration_sec=None, from_cache=True, prompt_identity=prompt)
    # Historical timing belongs to the immutable source, never this consumption.
    # The live role/submission wrapper attaches its own measurements on return.
    result.pop("timing", None)
    result.pop("shared_session_timing", None)
    for field in ("judge_debug", "reviewer_debug"):
        if isinstance(result.get(field), Mapping):
            result[field] = {key: value for key, value in result[field].items() if key != "timing"}
            if field == "reviewer_debug" and isinstance(result[field].get("route_debug"), Mapping):
                result[field]["route_debug"] = {key: value for key, value in result[field]["route_debug"].items() if key != "timing"}
    # This is consumption metadata only, never the origin of the cached answer.
    result["execution"] = {**(_EXECUTION.get() or {}), "role": role, "slot": slot,
                           "agent_attempt": uuid.uuid4().hex, "source_kind": "cache"}
    observer = _CAPTURE.get()
    if capture and observer is not None:
        # This is a snapshot of an already-consumed result, never a new producer
        # and never the stale candidate pool carried inside a cached judge bundle.
        observer.release(result["execution"]["agent_attempt"], result)
    return result
