"""Sanitized records of the provider's error answers on the lent file-key route.

A refused request (HTTP 4xx/5xx, or a 200 whose body carries an ``error``) is recorded with what the provider said and
nothing secret: HTTP status, the JSON ``error.code`` / ``error.type`` / ``error.message`` (Volcengine Ark shape, also
accepted without the ``error`` wrapper), the model, the key's file name and opaque broker id, and the time. The message
is redacted (:func:`redact`) and truncated before it leaves the request. Records go to the host broker's
``broker_errors`` table on a background thread (never on an event loop) and to a log line rate-limited per status and
code. ``python -m rubiksql_hb.key_broker errors --since 2h`` lists them by code.

Recording never fails or delays a request: :func:`report` only queues. The writer stores queued records in batches; a
batch that meets a busy broker (a burst of refusals from many processes contends for its one writer lock) goes back to
the front of the queue and is retried with a bounded, jittered exponential backoff, joined by records that arrived
meanwhile. A record is dropped, with a rate-limited warning, only after :data:`STORE_ATTEMPTS` busy attempts, on any
other storage error, or when the queue is full.

Main has no general redaction module (the one on ``enterprise_main`` is database-oriented and must not be merged
wholesale), so this module carries the small redaction the provider route needs.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
from typing import Any

#: Characters of a provider message kept after redaction.
MAX_MESSAGE = 500


def model_refusal(status: int | None, body: Any) -> bool:
    """Recognize structured model-configuration errors without treating keys as invalid."""
    if status not in (400, 404, 410):
        return False
    if isinstance(body, (bytes, str)):
        try:
            body = json.loads(body)
        except (ValueError, UnicodeError):
            return False
    if not isinstance(body, Mapping):
        return False
    error = body.get("error", body)
    if not isinstance(error, Mapping):
        return False
    code = re.sub(r"[^a-z]", "", str(error.get("code", error.get("Code", ""))).lower())
    return (str(error.get("param", "")).lower() == "model" or
            "model" in code and any(word in code for word in
                ("notfound", "notexist", "unsupported", "notsupported", "unavailable", "notavailable",
                 "notopen", "disabled", "accessdenied", "invalid")))


#: Characters of raw provider text redaction ever scans (redaction runs after this cut, so its cost is bounded).
MAX_SCAN = 4 * MAX_MESSAGE
#: One log line per status and code per process in this many seconds; every event is still stored.
LOG_INTERVAL_S = 60.0
#: Unwritten records beyond this are dropped (a refusal storm must not grow memory).
MAX_PENDING = 2000
#: Records one broker transaction stores at most.
BATCH = 200
#: Attempts to store a record while the broker's writer lock stays busy; the last busy failure drops it.
STORE_ATTEMPTS = 8
#: Backoff before the first retry of a busy batch, doubled per attempt up to :data:`RETRY_MAX_S` (with 50-100% jitter):
#: about 7 s of waiting over all attempts.
RETRY_BASE_S = 0.05
RETRY_MAX_S = 2.0

# Long names may end a compound word; the short ones only a whole word or a ``_``/``-`` segment (not risk, task, peak).
_NAME = (r"(?:[a-z0-9_-]{0,40}(?:api[_-]?key|access[_-]?key|secret[_-]?key|access[_-]?token|secret|token|password)"
         r"|(?:[a-z0-9_-]{0,40}[_-])?(?:key|ak|sk))")
_BEARER = re.compile(r"(?i)\b(bearer|basic|authorization)(\s*[:=]?\s*)(?:(?:bearer|basic)\s+)?[^\s\"',;]+")
_ASSIGNMENT = re.compile(r"(?i)\b((?:\"|')?" + _NAME + r"(?:\"|')?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;)}\]]+)")
_KEY_LIKE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|AKIA[A-Z0-9]{16}|LTAI[A-Za-z0-9]{16,}|"
                       r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b")
#: Masked echoes such as ``abcd****wxyz``.
_MASKED = re.compile(r"[A-Za-z0-9]{2,}\*{3,}[A-Za-z0-9]{2,}")
#: Long runs that may be a key (base64 included); redacted only when they mix letters and digits.
_LONG_RUN = re.compile(r"[A-Za-z0-9_+/=-]{32,}")
_LOGGER = logging.getLogger(__name__)
_LOCK = threading.Condition()
#: ``(broker, record, failed busy attempts)``, oldest first.
_QUEUE: deque[tuple[Any, dict[str, Any], int]] = deque()
_BUSY = False
_WRITER: threading.Thread | None = None
_LOGGED: dict[tuple[int | None, str], float] = {}
_STORE_WARNED = float("-inf")
#: The writer's backoff wait; tests replace it to run the retries on a fake clock.
_SLEEP = time.sleep


def _reset_after_fork() -> None:
    """A forked child has no writer thread and may inherit a held lock: start empty."""
    global _LOCK, _QUEUE, _BUSY, _WRITER
    _LOCK = threading.Condition()
    _QUEUE = deque()
    _BUSY = False
    _WRITER = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _long_run(match: re.Match[str]) -> str:
    text = match.group(0)
    if any(character.isdigit() for character in text) and any(character.isalpha() for character in text):
        return "[REDACTED]"
    return text


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Remove credentials from a provider diagnostic; callers cut the text to :data:`MAX_SCAN` first.

    Args:
        text: Diagnostic text.
        secrets: Exact values to remove first (the request's key).

    Returns:
        Text with known secrets, Authorization/Bearer/Basic values, key and secret assignments (``key=``, ``ak=``,
        ``sk=``, ``access_key``, ``secret_key``, ``api_key``, ``token``, ...), masked echoes (``abcd****wxyz``) and
        key-like tokens (``sk-…``, cloud access-key ids, UUID-shaped keys, long letter-digit or base64 runs) replaced
        by ``[REDACTED]``. Every pattern is linear in the text length.
    """
    for secret in sorted((value for value in secrets if value), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = _BEARER.sub(lambda match: match.group(1) + match.group(2) + "[REDACTED]", text)
    text = _ASSIGNMENT.sub(lambda match: match.group(1) + "[REDACTED]", text)
    text = _KEY_LIKE.sub("[REDACTED]", text)
    text = _MASKED.sub("[REDACTED]", text)
    return _LONG_RUN.sub(_long_run, text)


def _fields(body: Mapping[str, Any]) -> tuple[str, str, str]:
    """``(code, type, message)`` of one error object, in lower-case or Volcengine OpenAPI capitalised keys."""
    def text(*names: str) -> str:
        for name in names:
            value = body.get(name)
            if value not in (None, ""):
                return str(value)
        return ""
    return text("code", "Code", "CodeN"), text("type", "Type"), text("message", "Message", "msg", "detail")


def parse_error(body: Any) -> tuple[str, str, str]:
    """Extract ``(code, type, message)`` from a provider error body.

    Accepted shapes: Ark/OpenAI ``{"error": {...}}`` or the inner object; ``{"error": "text"}``; Volcengine OpenAPI
    ``{"ResponseMetadata": {"Error": {"Code", "Message"}}}``; top-level ``Code``/``Message``. When none of these yields
    anything, the message is the body's compact JSON (or text), so the stored event is never empty.

    Args:
        body: Parsed JSON, raw bytes/text, or anything else.

    Returns:
        Strings, unredacted and uncut; :func:`event` cuts and redacts them.
    """
    if isinstance(body, (bytes, bytearray)):
        body = bytes(body[:65536]).decode("utf-8", "replace")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            return "", "", body
    if body is None:
        return "", "", ""
    if not isinstance(body, Mapping):
        return "", "", str(body)
    error = body.get("error")
    if isinstance(error, Mapping):
        found = _fields(error)
    elif error not in (None, "", False):
        found = ("", "", str(error))
    else:
        metadata = body.get("ResponseMetadata")
        nested = metadata.get("Error") if isinstance(metadata, Mapping) else None
        found = _fields(nested) if isinstance(nested, Mapping) else _fields(body)
    if any(found):
        return found
    try:
        return "", "", json.dumps(body, ensure_ascii=False, separators=(",", ":"), default=str)[:MAX_SCAN]
    except (TypeError, ValueError):
        return "", "", str(body)[:MAX_SCAN]


def error_body(error: BaseException) -> Any:
    """The provider's error body of an SDK status error: its parsed ``body``, else the response text."""
    body = getattr(error, "body", None)
    if body is not None:
        return body
    response = getattr(error, "response", None)
    try:
        return getattr(response, "text", None)
    except Exception:  # noqa: BLE001 - an unread streamed body has no text
        return None


def event(*, status: int | None, body: Any, model: str, file_name: str, key_id: str, secret: str = "",
          now: float | None = None) -> dict[str, Any]:
    """Build one sanitized record; nothing of the request body or credential is kept.

    Args:
        status: HTTP status (200 for an error carried by a successful response).
        body: The provider's error body.
        model: Requested model.
        file_name: The key's approved file name (not secret).
        key_id: The broker's opaque key id.
        secret: The request's key, removed from every field.
        now: POSIX seconds; the current time by default.

    Returns:
        ``at, status, code, type, message, model, file_name, key_id``.
    """
    code, kind, message = parse_error(body)

    def clean(text: str, size: int) -> str:
        # The exact key goes first, on the uncut text (linear), so no key straddles the cut. The regexes then
        # scan at most MAX_SCAN characters, whatever the provider sent.
        if secret:
            text = text.replace(secret, "[REDACTED]")
        return redact(text[:MAX_SCAN], [secret])[:size]

    message = message.replace(secret, "[REDACTED]") if secret else message
    return {"at": time.time() if now is None else now, "status": status, "code": clean(code, 128),
            "type": clean(kind, 128), "message": clean(" ".join(message[:MAX_SCAN].split()), MAX_MESSAGE),
            "model": clean(model, 128), "file_name": file_name, "key_id": key_id}


def _warn_store(error: BaseException, outcome: str) -> None:
    """Log (rate-limited) a store failure and what became of the affected records."""
    global _STORE_WARNED
    now = time.monotonic()
    if now - _STORE_WARNED >= LOG_INTERVAL_S:
        _STORE_WARNED = now
        _LOGGER.warning("[lent-provider-error] Could not store error events (%s); %s.", type(error).__name__, outcome)


def _busy(error: BaseException) -> bool:
    """Whether a store failure is a busy or locked SQLite writer, which waiting can resolve."""
    code = getattr(error, "sqlite_errorcode", None)
    return (isinstance(error, sqlite3.OperationalError) and type(code) is int
            and (code & 0xff) in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED))


def _backoff(attempts: int) -> float:
    """Seconds to wait after a batch's ``attempts``-th busy failure."""
    return min(RETRY_MAX_S, RETRY_BASE_S * 2 ** (attempts - 1)) * random.uniform(0.5, 1.0)


def _drain() -> None:
    """Writer thread: store queued records in one transaction per broker and batch, retrying busy batches."""
    global _BUSY
    while True:
        with _LOCK:
            while not _QUEUE:
                _BUSY = False
                _LOCK.notify_all()
                _LOCK.wait()
            _BUSY = True
            batch = [_QUEUE.popleft() for _ in range(min(BATCH, len(_QUEUE)))]
        by_broker: dict[int, tuple[Any, list[tuple[dict[str, Any], int]]]] = {}
        for broker, record, attempts in batch:
            by_broker.setdefault(id(broker), (broker, []))[1].append((record, attempts))
        retry: list[tuple[Any, dict[str, Any], int]] = []
        for broker, entries in by_broker.values():
            try:
                broker.record_errors([record for record, _attempts in entries])
            except Exception as error:  # noqa: BLE001 - a lost diagnostic never fails a request
                kept = ([(broker, record, attempts + 1) for record, attempts in entries if attempts + 1 < STORE_ATTEMPTS]
                        if _busy(error) else [])
                if len(kept) < len(entries):
                    _warn_store(error, f"{len(entries) - len(kept)} record(s) dropped")
                retry.extend(kept)
        if retry:
            with _LOCK:
                # Back at the front, in order; the queue may exceed MAX_PENDING by at most one batch.
                _QUEUE.extendleft(reversed(retry))
            _SLEEP(_backoff(max(attempts for _broker, _record, attempts in retry)))


def report(broker: Any, record: dict[str, Any]) -> None:
    """Log (rate-limited) and queue one sanitized record for the writer thread; never blocks on SQLite.

    Args:
        broker: Host broker with ``record_errors``.
        record: Output of :func:`event`.
    """
    global _WRITER
    key = (record["status"], record["code"] or record["type"])
    now = time.monotonic()
    start_error: RuntimeError | None = None
    with _LOCK:
        log = now - _LOGGED.get(key, float("-inf")) >= LOG_INTERVAL_S
        if log:
            _LOGGED[key] = now
        if len(_QUEUE) < MAX_PENDING:
            _QUEUE.append((broker, record, 0))
            if _WRITER is None:
                writer = threading.Thread(target=_drain, name="lent-errors", daemon=True)
                try:
                    writer.start()
                    _WRITER = writer
                except RuntimeError as error:
                    start_error = error
            _LOCK.notify_all()
    if start_error is not None:
        _warn_store(start_error, "queued records wait for a later report")
    if log:
        _LOGGER.warning("[lent-provider-error] HTTP %s code=%s type=%s model=%s key=%s message=%s",
                        record["status"], record["code"] or "-", record["type"] or "-", record["model"],
                        record["file_name"], record["message"] or "-")


def flush(timeout: float = 5.0) -> None:
    """Wait until queued records are written (tests and orderly shutdown)."""
    deadline = time.monotonic() + timeout
    with _LOCK:
        while (_QUEUE or _BUSY) and time.monotonic() < deadline:
            _LOCK.wait(max(0.0, deadline - time.monotonic()))
