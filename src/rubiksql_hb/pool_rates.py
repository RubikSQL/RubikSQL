"""Durable request pacing shared by provider workers and processes."""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import closing
from pathlib import Path
import random
import sqlite3
import threading
import time
from collections.abc import Mapping
from typing import Any


_STATE_LOCKS: dict[str, threading.RLock] = {}
_STATE_LOCKS_MUTEX = threading.Lock()
_PACING_SCHEMAS: dict[str, tuple[int, int, int, int]] = {}


def state_lock(state_path: Path) -> threading.RLock:
    """Return the process-local transaction lock for one resolved state file."""
    identity = os.path.normcase(str(state_path.resolve()))
    with _STATE_LOCKS_MUTEX:
        lock = _STATE_LOCKS.get(identity)
        if lock is None:
            lock = threading.RLock()
            _STATE_LOCKS[identity] = lock
        return lock


def begin_immediate(connection: sqlite3.Connection) -> None:
    """Acquire the shared writer before any allocation or reservation work.

    A competing database process can exhaust SQLite's 30-second busy wait. Retry
    only acquisition, so account cursors and quota reservations are never replayed.
    All non-busy failures and failures inside an existing transaction propagate.

    Args:
        connection: An open connection with its busy handler configured.

    Raises:
        sqlite3.Error: Acquisition failed after at most three attempts.
    """
    for attempt in range(3):
        try:
            connection.execute("BEGIN IMMEDIATE")
            return
        except sqlite3.OperationalError as error:
            if (not _retryable_busy(error)
                    or connection.in_transaction or attempt == 2):
                raise
            time.sleep(random.uniform(0.05, 0.15))


def _retryable_busy(error: sqlite3.Error) -> bool:
    code = getattr(error, "sqlite_errorcode", None)
    return (type(code) is int and (code & 0xff) == sqlite3.SQLITE_BUSY
            and code != sqlite3.SQLITE_BUSY_SNAPSHOT)


def commit(connection: sqlite3.Connection) -> None:
    """Retry COMMIT on the same live transaction, never replaying its body.

    Rollback-journal readers can delay a writer's commit after its updates have
    succeeded. SQLite leaves that transaction active on BUSY. Each of three
    attempts uses the connection's busy timeout; other errors propagate so the
    owning context rolls back. BUSY_SNAPSHOT cannot be repaired by waiting.
    """
    for attempt in range(3):
        try:
            connection.commit()
            return
        except sqlite3.OperationalError as error:
            if not _retryable_busy(error) or not connection.in_transaction or attempt == 2:
                raise
            time.sleep(random.uniform(0.05, 0.15))


def token_reservation(payload: Any) -> int:
    """Bound input tokens by UTF-8 bytes and include explicitly requested output.

    Args:
        payload: Request text or a detached chat request without credentials.

    Returns:
        Conservative input reservation plus a requested output token limit, if any.
    """
    if isinstance(payload, Mapping) and isinstance(payload.get("messages"), list):
        messages = []
        for message in payload["messages"]:
            content = message.get("content")
            if isinstance(content, list) and all(isinstance(part, dict) and part.get("type") == "text" for part in content):
                message = {**message, "content": "".join(part.get("text", "") for part in content)}
            messages.append(message)
        payload = {**payload, "messages": messages}
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, default=str)
    output = 0
    if isinstance(payload, Mapping):
        for name in ("max_completion_tokens", "max_tokens"):
            value = payload.get(name)
            if type(value) is int and value > 0:
                output = value
                break
    return max(1, len(text.encode("utf-8"))) + output


def reserve(
    state_path: Path, *, bucket: str, model: str, requests_per_minute: int | None,
    tokens_per_minute: int | None, tokens: int,
) -> float:
    """Reserve one account/model request without holding a lock while waiting.

    Args:
        state_path: Existing private allocation database, shared by all workers.
        bucket: Explicit quota bucket identity; never an API credential.
        model: Exact provider model id.
        requests_per_minute: Configured positive request budget, or no request pacing.
        tokens_per_minute: Configured positive token budget, or no token pacing.
        tokens: Conservative input and requested output token reservation.

    Returns:
        Seconds until this request's reserved slot. Callers wait outside SQLite.

    Raises:
        ValueError: Invalid limits or a request larger than the token budget.
        sqlite3.Error: The shared durable reservation could not be committed.
    """
    if (not bucket or not model or type(tokens) is not int or tokens < 1
            or any(value is not None and (type(value) is not int or value < 1)
                   for value in (requests_per_minute, tokens_per_minute))):
        raise ValueError("provider pacing requires explicit positive limits and identities")
    if tokens_per_minute is not None and tokens > tokens_per_minute:
        raise ValueError("provider request exceeds its configured token budget")
    if requests_per_minute is None and tokens_per_minute is None:
        return 0.0
    identity = hashlib.sha256(json.dumps([bucket, model], separators=(",", ":")).encode()).hexdigest()
    state_identity = os.path.normcase(str(state_path.resolve()))
    with (
        state_lock(state_path),
        closing(sqlite3.connect(state_path, timeout=30, isolation_level=None)) as connection,
        connection,
    ):
        begin_immediate(connection)
        stat = state_path.stat()
        schema_identity = (os.getpid(), stat.st_dev, stat.st_ino,
                           connection.execute("PRAGMA schema_version").fetchone()[0])
        if _PACING_SCHEMAS.get(state_identity) != schema_identity:
            connection.execute("CREATE TABLE IF NOT EXISTS gateway_request_pacing ("
                               "bucket_model TEXT PRIMARY KEY, next_request REAL NOT NULL, next_token REAL NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS gateway_token_reservations ("
                               "reservation_id INTEGER PRIMARY KEY,bucket_model TEXT NOT NULL,start_at REAL NOT NULL,tokens INTEGER NOT NULL)")
            connection.execute("CREATE INDEX IF NOT EXISTS gateway_token_reservations_bucket ON gateway_token_reservations(bucket_model,start_at)")
            schema_identity = (*schema_identity[:3], connection.execute("PRAGMA schema_version").fetchone()[0])
        row = connection.execute("SELECT next_request,next_token FROM gateway_request_pacing WHERE bucket_model=?",
                                 (identity,)).fetchone()
        now = time.time()
        request_at, token_at = row if row is not None else (now, now)
        start = max(now, request_at if requests_per_minute else now, token_at if tokens_per_minute else now)
        connection.execute("DELETE FROM gateway_token_reservations WHERE start_at<=?", (now - 60,))
        if tokens_per_minute is not None:
            reservations = connection.execute("SELECT start_at,tokens FROM gateway_token_reservations "
                                              "WHERE bucket_model=? AND start_at>? ORDER BY start_at,reservation_id",
                                              (identity, start - 60)).fetchall()
            while sum(value for at, value in reservations if at > start - 60) + tokens > tokens_per_minute:
                start = next(at for at, _value in reservations if at > start - 60) + 60
        next_request = start + 60 / requests_per_minute if requests_per_minute else now
        next_token = start + 60 * tokens / tokens_per_minute if tokens_per_minute else now
        connection.execute("INSERT INTO gateway_request_pacing VALUES (?,?,?) "
                           "ON CONFLICT(bucket_model) DO UPDATE SET next_request=excluded.next_request,next_token=excluded.next_token",
                           (identity, next_request, next_token))
        if tokens_per_minute is not None:
            connection.execute("INSERT INTO gateway_token_reservations (bucket_model,start_at,tokens) VALUES (?,?,?)",
                               (identity, start, tokens))
        commit(connection)
        _PACING_SCHEMAS[state_identity] = schema_identity
    return max(0.0, start - now)


def wait(state_path: Path, **request: Any) -> None:
    """Wait for a durable provider request reservation without blocking its peers."""
    delay = reserve(state_path, **request)
    if delay:
        time.sleep(delay)
