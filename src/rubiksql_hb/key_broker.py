"""Hand out provider keys one request at a time, in strict round-robin order shared by a host's processes.

Every request takes the next key of its pool in the pool's own fixed order; after the last key the next round starts
again from the first. A key the provider reported out of daily quota is skipped for :data:`EXHAUSTION_SKIP_S` and then
probed again; every mark lapses at the quota day's boundary (local midnight, Asia/Shanghai, :data:`QUOTA_TZ`). Keys an
account-level failure retired are excluded by the caller (``GatewayPool`` keeps that durable status).

The broker never sees a credential. Callers pass opaque key ids (``<pool index>-<sha256 prefix>``, :func:`key_id`) and
map the chosen id back to the secret in their own process. :class:`SqliteKeyBroker` keeps the cursor and the marks in a
host-wide SQLite file (WAL, every update one ``BEGIN IMMEDIATE`` transaction), so concurrent processes on Windows and
Linux share one order. :class:`LocalKeyBroker` keeps the same state in process memory; it serves the embedding pool when
the host broker is switched off (:data:`BROKER_ENV`). Inline pools use only a cursor. The lent file pool additionally
uses expiring per-request reservations, FULL-synchronous durable status marks, counters and file-name identities.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import argparse
import atexit
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from uuid import uuid4

from .pool_rates import begin_immediate, commit
from .process_identity import OwnerLiveness, owner_tag

_LOGGER = logging.getLogger(__name__)

#: ``0``, ``off``, ``false`` or ``no`` switch the host broker off: chat returns to the per-database group rotation and
#: embeddings rotate within each process. Any other value, or no value, keeps the shared host broker.
BROKER_ENV = "RUBIKSQL_KEY_BROKER"
#: Optional path of the host broker's SQLite file; by default it sits beside the gateway allocation state. Point it at a
#: root excluded from on-access antivirus scanning (e.g. a Defender-excluded work root) to keep each turn cheap.
BROKER_STATE_ENV = "RUBIKSQL_KEY_BROKER_STATE"
#: How long a key reported out of daily quota is skipped before it is probed again.
EXHAUSTION_SKIP_S = 1800.0
#: The providers' quota day: Asia/Shanghai has no daylight saving, so a fixed offset is exact.
QUOTA_TZ = timezone(timedelta(hours=8))
_OFF = frozenset({"0", "off", "false", "no"})
_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS broker_cursors (pool TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, "
    "next_position INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS broker_marks (pool TEXT NOT NULL, key_id TEXT NOT NULL, "
    "exhausted_until REAL NOT NULL, quota_day TEXT NOT NULL, PRIMARY KEY(pool, key_id))",
    "CREATE TABLE IF NOT EXISTS broker_file_keys (key_id TEXT PRIMARY KEY, file_name TEXT NOT NULL, "
    "cooling_until REAL NOT NULL DEFAULT 0, exhausted_until REAL NOT NULL DEFAULT 0, retired INTEGER NOT NULL DEFAULT 0, "
    "requests INTEGER NOT NULL DEFAULT 0, ok INTEGER NOT NULL DEFAULT 0, rate_limited INTEGER NOT NULL DEFAULT 0, "
    "exhausted INTEGER NOT NULL DEFAULT 0, invalid INTEGER NOT NULL DEFAULT 0, errors INTEGER NOT NULL DEFAULT 0, "
    "peak INTEGER NOT NULL DEFAULT 0)",
    "CREATE TABLE IF NOT EXISTS broker_leases (token TEXT PRIMARY KEY, key_id TEXT NOT NULL, expires REAL NOT NULL)",
    "CREATE INDEX IF NOT EXISTS broker_leases_key ON broker_leases(key_id)",
    "CREATE INDEX IF NOT EXISTS broker_leases_expiry ON broker_leases(expires)",
    "CREATE TABLE IF NOT EXISTS broker_totals (id INTEGER PRIMARY KEY CHECK(id=1), peak INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS broker_notices (name TEXT PRIMARY KEY, last_at REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS broker_capbal (id INTEGER PRIMARY KEY CHECK(id=1), "
    "cap INTEGER NOT NULL CHECK(cap BETWEEN 0 AND 60))",
    # One row per process and kind (HostCounter): admissions waiting for a slot ("waiting"), questions held at the
    # question-start gate ("gate_held"), questions held at or running past the gate ("gate_active"). A row is live
    # while its heartbeat is newer than COUNT_TTL_S and its owner process still runs (process_identity).
    "CREATE TABLE IF NOT EXISTS broker_counts (kind TEXT NOT NULL, owner TEXT NOT NULL, value INTEGER NOT NULL, "
    "updated REAL NOT NULL, PRIMARY KEY(kind, owner))",
    # Sanitized provider error answers (provider_errors.event): no credential, header or request body.
    "CREATE TABLE IF NOT EXISTS broker_errors (id INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL, status INTEGER, "
    "code TEXT NOT NULL, type TEXT NOT NULL, message TEXT NOT NULL, model TEXT NOT NULL, file_name TEXT NOT NULL, "
    "key_id TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS broker_errors_at ON broker_errors(at)",
    # Question-start slot gate (rubiksql_hb.question_gate, slots mode). A reservation is one admitted question's
    # request slots (its candidate seats, then its judge passes); a queue ticket is one question waiting to start,
    # served in ``seq`` order. Both name their owner process (process_identity) and carry a heartbeat.
    "CREATE TABLE IF NOT EXISTS broker_reservations (token TEXT PRIMARY KEY, owner TEXT NOT NULL, "
    "slots INTEGER NOT NULL, phase TEXT NOT NULL, updated REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS broker_gate_queue (seq INTEGER PRIMARY KEY AUTOINCREMENT, owner TEXT NOT NULL, "
    "need INTEGER NOT NULL, touched REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS broker_gate_totals (id INTEGER PRIMARY KEY CHECK(id=1), admitted INTEGER NOT NULL)",
    # The capacity the slot gate last admitted against, its source (override or keys) and which process used it.
    "CREATE TABLE IF NOT EXISTS broker_gate_capacity (id INTEGER PRIMARY KEY CHECK(id=1), capacity INTEGER NOT NULL, "
    "source TEXT NOT NULL, owner TEXT NOT NULL, at REAL NOT NULL)",
)
#: Provider error events older than this are pruned when new ones are written.
ERROR_RETENTION_S = 7 * 86400.0
#: A process's published count older than this is ignored: the process stopped or died without publishing zero.
#: A process with a nonzero count republishes at least every :data:`COUNT_HEARTBEAT_S`, so a live row survives about
#: 17 consecutive busy-broker skips. An orderly exit publishes zero. A killed process (no ``atexit``) is recognized
#: sooner: readers ignore a row whose owner pid no longer runs or was reused (:mod:`.process_identity`), so its last
#: count lingers this long only when its owner cannot be checked.
COUNT_TTL_S = 90.0
COUNT_HEARTBEAT_S = 5.0
#: Publishers delete dead owners' rows at most this often, so readers without the liveness check (the cap
#: controller's host helper) stop counting them too.
COUNT_SWEEP_S = 10.0
#: Consecutive failed publications after which a :class:`HostCounter` warns that its row may expire.
COUNT_WARN_FAILURES = 10
#: The per-process counts processes publish (:class:`HostCounter`).
COUNT_KINDS = ("waiting", "gate_held", "gate_active")
#: A slot reservation whose owner stopped refreshing it for this long is ignored (its process hung or died
#: unrecognized). A gate process refreshes its rows on every cycle, about once a second.
RESERVATION_TTL_S = 90.0
#: A queue ticket not touched for this long loses its place to later tickets until its waiter touches it again.
TICKET_TTL_S = 30.0
#: Reservations and tickets untouched for this long are deleted.
GATE_ROW_PURGE_S = 3600.0
#: The capacity a gate cycle stored is the host's current one for this long (stats, conflicting overrides).
GATE_CAPACITY_FRESH_S = 10.0
_BROKERS: dict[str, "SqliteKeyBroker"] = {}
_BROKERS_LOCK = threading.Lock()


@dataclass(frozen=True)
class GateRequest:
    """One gate process's state sent to the broker in one cycle (:meth:`SqliteKeyBroker.gate_cycle`).

    Attributes:
        owner: The process's ``<pid>-<birth>-<nonce>`` owner id (:mod:`.process_identity`).
        enqueue: New questions to queue, ``(token, need, phase)`` each, in arrival order.
        waiting: Queued questions still waiting, ``(token, seq, need, phase)`` each.
        leave: Queue tickets (``seq``) whose questions left without starting.
        reservations: Every admitted question's current reservation, ``(token, slots, phase)`` each; this refreshes
            their heartbeat.
        release: Tokens of questions that finished.
        admit: Admit queued questions in this cycle; ``False`` only refreshes and queues (after a wall-clock jump,
            when other processes' rows may look stale until they refresh them).
    """

    owner: str
    enqueue: tuple[tuple[str, int, str], ...] = ()
    waiting: tuple[tuple[str, int, int, str], ...] = ()
    leave: tuple[int, ...] = ()
    reservations: tuple[tuple[str, int, str], ...] = ()
    release: tuple[str, ...] = ()
    admit: bool = True


@dataclass(frozen=True)
class GateCycle:
    """The broker's answer to one :class:`GateRequest`.

    Attributes:
        capacity: Host request slots (the manual override, or eligible keys x per-key cap).
        reserved: Slots reserved by live questions, this cycle's admissions included.
        running: Live reservations (admitted questions not yet finished) on the host.
        queued: Live queue tickets left on the host.
        queued_slots: Slots those tickets need.
        inflight: Unexpired key leases.
        waiting: Admissions waiting for a key slot on live processes.
        admitted_total: Questions the host's gate has admitted since the broker file was created.
        enqueued: ``{token: seq}`` of this request's new tickets.
        admitted: Tokens admitted in this cycle.
        capacity_source: ``override`` (the manual capacity) or ``keys`` (computed).
        capacity_conflict: ``(capacity, source)`` another live process stored within
            :data:`GATE_CAPACITY_FRESH_S` when it differs from this cycle's manual capacity; else ``None``.
    """

    capacity: int
    reserved: int
    running: int = 0
    queued: int = 0
    queued_slots: int = 0
    inflight: int = 0
    waiting: int = 0
    admitted_total: int = 0
    enqueued: dict[str, int] = field(default_factory=dict)
    admitted: frozenset[str] = frozenset()
    capacity_source: str = "keys"
    capacity_conflict: tuple[int, str] | None = None


def _busy(error: sqlite3.OperationalError) -> bool:
    """Whether an SQLite error is a busy writer lock (the caller retries later)."""
    return (getattr(error, "sqlite_errorcode", 0) & 0xff) == sqlite3.SQLITE_BUSY


def broker_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether requests take their keys from the shared host broker.

    Args:
        environ (Mapping[str, str] | None): Environment to read; the process environment by default.

    Returns:
        bool: ``False`` only when :data:`BROKER_ENV` names an off value.
    """
    env = os.environ if environ is None else environ
    return env.get(BROKER_ENV, "").strip().lower() not in _OFF


def key_id(index: int, secret: str) -> str:
    """The broker's opaque id of one key: its pool index and a 12-hex-digit SHA-256 prefix, never the value.

    Args:
        index (int): The key's position in its pool's fixed order.
        secret (str): The key itself, hashed in-process.

    Returns:
        str: ``<index>-<sha256 prefix>``.
    """
    return f"{index}-{hashlib.sha256(secret.encode('utf-8')).hexdigest()[:12]}"


def quota_day(now: float) -> str:
    """The provider quota day a timestamp falls in (ISO date in :data:`QUOTA_TZ`).

    Args:
        now (float): POSIX seconds.

    Returns:
        str: ``YYYY-MM-DD``.
    """
    return datetime.fromtimestamp(now, QUOTA_TZ).date().isoformat()


def _fingerprint(key_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(key_ids).encode("utf-8")).hexdigest()


def _choose(key_ids: Sequence[str], eligible: set[str], marks: Mapping[str, tuple[float, str]],
            cursor: int, now: float) -> int | None:
    """The first position from ``cursor`` on, cyclically, whose key is eligible and not skipped for exhaustion."""
    today = quota_day(now)
    for step in range(len(key_ids)):
        position = (cursor + step) % len(key_ids)
        mark = marks.get(key_ids[position])
        if key_ids[position] in eligible and not (mark is not None and mark[1] == today and mark[0] > now):
            return position
    return None


class LocalKeyBroker:
    """Keep the round-robin cursor and exhaustion marks of each pool in this process only.

    Args:
        clock (Callable[[], float]): POSIX-seconds clock; tests pass a controllable one.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._cursors: dict[str, tuple[str, int]] = {}
        self._marks: dict[tuple[str, str], tuple[float, str]] = {}

    def next(self, pool: str, key_ids: Sequence[str], eligible: set[str], *, advance: bool = True) -> str | None:
        """Take the next eligible key of ``pool`` in its fixed order.

        Args:
            pool (str): Key pool name, e.g. ``volcengine-chat`` or ``dashscope-embedding``.
            key_ids (Sequence[str]): Every key id of the pool, in the pool's fixed order.
            eligible (set[str]): The ids not retired by an account-level failure.
            advance (bool): Move the cursor past the chosen key; ``False`` only previews it.

        Returns:
            str | None: The chosen key id, or ``None`` when every key is retired or skipped for exhaustion.
        """
        with self._lock:
            now = self._clock()
            fingerprint = _fingerprint(key_ids)
            stored = self._cursors.get(pool)
            cursor = stored[1] if stored is not None and stored[0] == fingerprint else 0
            marks = {key: mark for (name, key), mark in self._marks.items() if name == pool}
            position = _choose(key_ids, eligible, marks, cursor, now)
            if position is None:
                return None
            if advance:
                self._cursors[pool] = (fingerprint, (position + 1) % len(key_ids))
            return key_ids[position]

    def mark_exhausted(self, pool: str, key: str) -> None:
        """Skip ``key`` for :data:`EXHAUSTION_SKIP_S`, or until the quota day ends.

        Args:
            pool (str): Key pool name.
            key (str): The exhausted key's id.
        """
        with self._lock:
            now = self._clock()
            self._marks[(pool, key)] = (now + EXHAUSTION_SKIP_S, quota_day(now))

    def marks(self, pool: str) -> dict[str, tuple[float, str]]:
        """The pool's exhaustion marks, ``{key id: (skip until, quota day)}``, for status reports and tests."""
        with self._lock:
            return {key: mark for (name, key), mark in self._marks.items() if name == pool}


class SqliteKeyBroker:
    """Share each pool's round-robin cursor and exhaustion marks across every process on a host.

    The file holds pool names, key ids, positions and timestamps only. Each call is one short ``BEGIN IMMEDIATE``
    transaction, so two processes never take the same turn. Inline pools have no reservations. File pools use
    acquire/finish leases, with expiry for crashed callers and a permanent file-name identity for retirement.

    Args:
        path (Path): Private SQLite file, created with its parent directory when missing.
        clock (Callable[[], float]): POSIX-seconds clock; tests pass a controllable one.
        owner_dead (Callable[[str], bool] | None): Whether a count row's owner process is provably gone; a cached
            :class:`.process_identity.OwnerLiveness` check by default, tests pass a fake.
    """

    def __init__(self, path: Path, clock: Callable[[], float] = time.time,
                 owner_dead: Callable[[str], bool] | None = None) -> None:
        self.path = Path(path)
        self._clock = clock
        self._owner_dead = owner_dead or OwnerLiveness().dead
        self._swept_at = float("-inf")
        self._ready = False
        self._registered_files: tuple[str, ...] = ()
        self._lock = threading.Lock()

    @classmethod
    def for_path(cls, path: Path) -> "SqliteKeyBroker":
        """The process's broker for one resolved state file, so its schema is created once."""
        identity = os.path.normcase(str(Path(path).resolve()))
        with _BROKERS_LOCK:
            broker = _BROKERS.get(identity)
            if broker is None:
                broker = _BROKERS[identity] = cls(Path(identity))
            return broker

    def _connect(self) -> sqlite3.Connection:
        if not self._ready:
            with self._lock:
                if not self._ready:
                    self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    created = not self.path.exists()
                    with closing(sqlite3.connect(self.path, timeout=30, isolation_level=None)) as connection:
                        connection.execute("PRAGMA journal_mode=WAL")
                        for statement in _SCHEMA:
                            connection.execute(statement)
                    if created and os.name != "nt":
                        self.path.chmod(0o600)
                    self._ready = True
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=30000")
        # WAL with NORMAL sync stays consistent after a crash; at worst the last few turns or marks are lost.
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def next(self, pool: str, key_ids: Sequence[str], eligible: set[str], *, advance: bool = True) -> str | None:
        """Take the next eligible key of ``pool`` in its fixed order, atomically across processes.

        Args:
            pool (str): Key pool name, e.g. ``volcengine-chat`` or ``dashscope-embedding``.
            key_ids (Sequence[str]): Every key id of the pool, in the pool's fixed order. A changed list restarts the
                cursor at the first key.
            eligible (set[str]): The ids not retired by an account-level failure.
            advance (bool): Move the shared cursor past the chosen key; ``False`` only previews it.

        Returns:
            str | None: The chosen key id, or ``None`` when every key is retired or skipped for exhaustion.
        """
        fingerprint = _fingerprint(key_ids)
        with closing(self._connect()) as connection:
            begin_immediate(connection)
            try:
                now = self._clock()
                row = connection.execute("SELECT fingerprint, next_position FROM broker_cursors WHERE pool=?",
                                         (pool,)).fetchone()
                cursor = row[1] if row is not None and row[0] == fingerprint and 0 <= row[1] < len(key_ids) else 0
                marks = {key: (until, day) for key, until, day in connection.execute(
                    "SELECT key_id, exhausted_until, quota_day FROM broker_marks WHERE pool=?", (pool,))}
                position = _choose(key_ids, eligible, marks, cursor, now)
                if position is not None and advance:
                    connection.execute(
                        "INSERT INTO broker_cursors VALUES (?,?,?) ON CONFLICT(pool) DO UPDATE SET "
                        "fingerprint=excluded.fingerprint, next_position=excluded.next_position",
                        (pool, fingerprint, (position + 1) % len(key_ids)))
                commit(connection)
            finally:
                if connection.in_transaction:
                    connection.rollback()
        return None if position is None else key_ids[position]

    def mark_exhausted(self, pool: str, key: str) -> None:
        """Skip ``key`` for :data:`EXHAUSTION_SKIP_S`, or until the quota day ends, on every process of the host.

        Args:
            pool (str): Key pool name.
            key (str): The exhausted key's id.
        """
        with closing(self._connect()) as connection:
            begin_immediate(connection)
            try:
                now = self._clock()
                connection.execute(
                    "INSERT INTO broker_marks VALUES (?,?,?,?) ON CONFLICT(pool, key_id) DO UPDATE SET "
                    "exhausted_until=excluded.exhausted_until, quota_day=excluded.quota_day",
                    (pool, key, now + EXHAUSTION_SKIP_S, quota_day(now)))
                commit(connection)
            finally:
                if connection.in_transaction:
                    connection.rollback()

    def marks(self, pool: str) -> dict[str, tuple[float, str]]:
        """The pool's exhaustion marks, ``{key id: (skip until, quota day)}``, for status reports and tests."""
        with closing(self._connect()) as connection:
            return {key: (until, day) for key, until, day in connection.execute(
                "SELECT key_id, exhausted_until, quota_day FROM broker_marks WHERE pool=?", (pool,))}

    def acquire(self, files: Sequence[str], *, cap: int, ttl: float) -> tuple[str, str] | None:
        """Atomically reserve the next available file key, or return None while capped/stalled.

        Args:
            files: Validated file names in the owner's fixed order. No credentials.
            cap: Maximum simultaneous leases per key on this host, 0–60 (0 pauses admission).
            ttl: Crash recovery lifetime, longer than the request's absolute deadline.

        Returns:
            A unique lease token and file name, or None. Expired leases are reclaimed.
        """
        if type(cap) is not int or not 0 <= cap <= 60 or not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("Invalid file-key lease cap or lifetime.")
        ids = [file_key_id(name) for name in files]
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            # Admission runs on async callers too. A busy writer means wait outside SQLite,
            # rather than holding an event loop in the inline broker's long busy ladder.
            connection.execute("PRAGMA busy_timeout=25")
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as error:
                if _busy(error):
                    return None
                raise
            try:
                # A cap-file read can precede this transaction by an arbitrary delay.
                # The controller's durable ceiling shares the admission writer lock,
                # so a stale caller cannot spend capacity transferred to another host.
                ceiling = connection.execute("SELECT cap FROM broker_capbal WHERE id=1").fetchone()
                if ceiling is not None:
                    cap = min(cap, ceiling[0])
                now = self._clock()
                connection.execute("DELETE FROM broker_leases WHERE expires<=?", (now,))
                if self._registered_files != tuple(files):
                    connection.executemany("INSERT OR IGNORE INTO broker_file_keys(key_id,file_name) VALUES (?,?)", zip(ids, files))
                row = connection.execute("SELECT next_position FROM broker_cursors WHERE pool='lent-volcano-files'").fetchone()
                cursor = row[0] if row else 0
                states = {row[0]: row[1:] for row in connection.execute(
                    "SELECT k.key_id,k.cooling_until,k.exhausted_until,k.retired,count(l.token) "
                    "FROM broker_file_keys k LEFT JOIN broker_leases l ON l.key_id=k.key_id GROUP BY k.key_id")}
                chosen = None
                for step in range(len(ids)):
                    position = (cursor + step) % len(ids)
                    key = ids[position]
                    state = states[key]
                    active = state[3]
                    if not state[2] and max(state[:2]) <= now and active < cap:
                        chosen = (uuid4().hex, files[position])
                        connection.execute("INSERT INTO broker_leases VALUES (?,?,?)", (chosen[0], key, now + ttl))
                        connection.execute("UPDATE broker_file_keys SET requests=requests+1,peak=max(peak,?) WHERE key_id=?", (active + 1, key))
                        total = sum(state[3] for state in states.values()) + 1
                        connection.execute("INSERT INTO broker_totals VALUES(1,?) ON CONFLICT(id) DO UPDATE SET peak=max(peak,excluded.peak)", (total,))
                        connection.execute("INSERT INTO broker_cursors VALUES('lent-volcano-files',?,?) ON CONFLICT(pool) DO UPDATE SET next_position=excluded.next_position",
                                           (_fingerprint(ids), (position + 1) % len(ids)))
                        break
                commit(connection)
                self._registered_files = tuple(files)
                return chosen
            finally:
                if connection.in_transaction:
                    connection.rollback()

    def finish(self, token: str, status: int | None, *, retry_after: float = 0) -> None:
        """Release exactly this lease, count the outcome and persist owner-defined status marks.

        Args:
            token: Token from acquire; an expired or already finished token is harmless.
            status: HTTP status, or None for a transport/cancellation failure.
            retry_after: Validated cooldown seconds for HTTP 429.
        """
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            begin_immediate(connection)
            try:
                row = connection.execute("SELECT key_id FROM broker_leases WHERE token=?", (token,)).fetchone()
                if row:
                    key, now = row[0], self._clock()
                    counter = {429: "rate_limited", 403: "exhausted", 401: "invalid"}.get(status,
                        "ok" if status is not None and 200 <= status < 300 else "errors")
                    connection.execute(f"UPDATE broker_file_keys SET {counter}={counter}+1 WHERE key_id=?", (key,))
                    if status == 429:
                        connection.execute("UPDATE broker_file_keys SET cooling_until=max(cooling_until,?) WHERE key_id=?", (now + retry_after, key))
                    elif status == 403:
                        today = datetime.fromtimestamp(now, QUOTA_TZ).date()
                        reset = datetime.combine(today, datetime.min.time(), QUOTA_TZ) + timedelta(minutes=5)
                        if reset.timestamp() <= now:
                            reset += timedelta(days=1)
                        connection.execute("UPDATE broker_file_keys SET exhausted_until=max(exhausted_until,?) WHERE key_id=?", (reset.timestamp(), key))
                    elif status == 401:
                        connection.execute("UPDATE broker_file_keys SET retired=1 WHERE key_id=?", (key,))
                    connection.execute("DELETE FROM broker_leases WHERE token=?", (token,))
                commit(connection)
            finally:
                if connection.in_transaction:
                    connection.rollback()

    def stats(self) -> dict[str, object]:
        """Return counters, marks and active leases, identified only by opaque id and file name."""
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            rows = [dict(row) for row in connection.execute("SELECT * FROM broker_file_keys ORDER BY file_name")]
            for row in rows:
                row["inflight"] = connection.execute("SELECT count(*) FROM broker_leases WHERE key_id=? AND expires>?", (row["key_id"], self._clock())).fetchone()[0]
            peak = connection.execute("SELECT peak FROM broker_totals WHERE id=1").fetchone()
            return {"keys": rows, "peak_total_inflight": peak[0] if peak else 0}

    def publish_count(self, kind: str, owner: str, value: int) -> bool:
        """Record one process's count of ``kind``; skipped while the writer is busy.

        One short autocommit statement with a 250 ms busy timeout (a publisher writes about once a second, off any
        event loop): a busy broker returns ``False`` and the caller publishes again on its next tick.

        Args:
            kind: One of :data:`COUNT_KINDS`.
            owner: Opaque per-process id (no host, user or key material).
            value: The process's current count.

        Returns:
            Whether the count was written.
        """
        if kind not in COUNT_KINDS:
            raise ValueError(f"unknown host count: {kind}")
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA busy_timeout=250")
            now = self._clock()
            try:
                connection.execute("INSERT INTO broker_counts VALUES (?,?,?,?) ON CONFLICT(kind, owner) DO UPDATE SET "
                                   "value=excluded.value, updated=excluded.updated", (kind, owner, int(value), now))
                connection.execute("DELETE FROM broker_counts WHERE updated<?", (now - 3600,))
            except sqlite3.OperationalError as error:
                if _busy(error):
                    return False
                raise
            if now - self._swept_at >= COUNT_SWEEP_S:
                self._swept_at = now
                self._sweep_dead(connection, now)
        return True

    def _sweep_dead(self, connection: sqlite3.Connection, now: float) -> None:
        """Delete the live-looking rows of owners that provably exited; best effort, the count is already written."""
        try:
            owners = [owner for (owner,) in connection.execute(
                "SELECT DISTINCT owner FROM broker_counts WHERE updated>=?", (now - COUNT_TTL_S,)).fetchall()]
            dead = [owner for owner in owners if self._owner_dead(owner)]
            if dead:
                connection.executemany("DELETE FROM broker_counts WHERE owner=?", [(owner,) for owner in dead])
        except Exception as error:  # noqa: BLE001 - a skipped sweep only delays it; readers still ignore dead owners
            _LOGGER.debug("[host-count] Dead-owner sweep skipped (%s).", type(error).__name__)

    def _live_counts(self, connection: sqlite3.Connection) -> list[tuple[str, int]]:
        """``(kind, value)`` of each nonzero count whose heartbeat is within :data:`COUNT_TTL_S` and whose owner runs."""
        rows = connection.execute("SELECT kind, owner, value FROM broker_counts WHERE updated>=? AND value>0",
                                  (self._clock() - COUNT_TTL_S,)).fetchall()
        return [(kind, int(value)) for kind, owner, value in rows if not self._owner_dead(owner)]

    def host_counts(self) -> dict[str, int]:
        """Sum every live process's published counts, ``{kind: total}`` for each of :data:`COUNT_KINDS`."""
        with closing(self._connect()) as connection:
            rows = self._live_counts(connection)
        return {kind: sum(value for name, value in rows if name == kind) for kind in COUNT_KINDS}

    def backlog(self, files: Sequence[str]) -> dict[str, int]:
        """Read the host's key-slot backlog without taking the writer lock.

        Args:
            files: The pool's file names, used only when no key has been registered yet.

        Returns:
            ``waiting`` (admissions waiting on live processes), ``holders`` (live processes holding questions at the
            question-start gate), ``gates`` (live processes with questions held at or running past the gate),
            ``inflight`` (unexpired leases), ``usable_keys``
            (registered keys neither retired nor out of daily quota, or ``len(files)`` before any registration) and
            ``ceiling`` (the controller's durable per-key ceiling, 60 when none is set).
        """
        with closing(self._connect()) as connection:
            now = self._clock()
            counts = self._live_counts(connection)
            inflight = connection.execute("SELECT count(*) FROM broker_leases WHERE expires>?", (now,)).fetchone()[0]
            registered, usable = connection.execute(
                "SELECT count(*), coalesce(sum(retired=0 AND exhausted_until<=?),0) FROM broker_file_keys "
                f"WHERE file_name IN ({','.join('?' for _ in files)})",
                (now, *files)).fetchone()
            ceiling = connection.execute("SELECT cap FROM broker_capbal WHERE id=1").fetchone()
        return {"waiting": sum(value for kind, value in counts if kind == "waiting"),
                "holders": sum(1 for kind, _ in counts if kind == "gate_held"),
                "gates": sum(1 for kind, _ in counts if kind == "gate_active"), "inflight": int(inflight),
                "usable_keys": len(files) - int(registered) + int(usable),
                "ceiling": int(ceiling[0]) if ceiling is not None else 60}

    def _slot_capacity(self, connection: sqlite3.Connection, files: Sequence[str], cap: int, now: float) -> int:
        """Eligible keys x per-key cap: keys neither cooling, exhausted nor retired, capped by the controller ceiling."""
        ceiling = connection.execute("SELECT cap FROM broker_capbal WHERE id=1").fetchone()
        per_key = min(cap, ceiling[0]) if ceiling is not None else cap
        registered, eligible = connection.execute(
            "SELECT count(*), coalesce(sum(retired=0 AND cooling_until<=? AND exhausted_until<=?),0) "
            f"FROM broker_file_keys WHERE file_name IN ({','.join('?' for _ in files)})", (now, now, *files)).fetchone()
        return max(0, int(len(files) - registered + eligible) * max(0, per_key))

    def _live_gate_rows(self, connection: sqlite3.Connection, now: float) -> tuple[list[tuple], list[tuple]]:
        """Live reservations ``(token, owner, slots)`` and live tickets ``(seq, owner, need)`` in queue order."""
        reservations = [row for row in connection.execute(
            "SELECT token, owner, slots FROM broker_reservations WHERE updated>=?", (now - RESERVATION_TTL_S,))
            if not self._owner_dead(row[1])]
        tickets = [row for row in connection.execute(
            "SELECT seq, owner, need FROM broker_gate_queue WHERE touched>=? ORDER BY seq", (now - TICKET_TTL_S,))
            if not self._owner_dead(row[1])]
        return reservations, tickets

    def gate_cycle(self, request: GateRequest, *, files: Sequence[str], cap: int,
                   capacity: int | None = None) -> GateCycle | None:
        """Apply one gate process's changes and admit its queued questions that fit, in one transaction.

        A ticket is admitted only when the host's free slots cover it and every live ticket queued before it
        (``capacity - reserved >= sum of needs up to and including it``), so admission is first come, first served
        across processes and databases: a later question never takes the room an earlier one is waiting for, and
        every earlier ticket that fits is admitted on its own process's next cycle. An admitted ticket becomes a
        reservation of its ``need`` slots. Reservations and tickets of owners that provably exited are deleted.

        Args:
            request: The process's changes and its current reservations.
            files: The pool's file names; before any key is registered every file counts as eligible.
            cap: The host's per-key cap (cap file or ``RUBIKSQL_KEY_MAX_INFLIGHT``); the controller's durable
                ceiling lowers it.
            capacity: A manual total slot capacity replacing the computed one (``RUBIKSQL_QUESTION_START_CAPACITY``).

        Returns:
            The host state after the cycle, or ``None`` when the broker was busy (nothing changed; retry later).
        """
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA busy_timeout=250")
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as error:
                if _busy(error):
                    return None
                raise
            try:
                now = self._clock()
                purge = now - GATE_ROW_PURGE_S
                connection.execute("DELETE FROM broker_reservations WHERE updated<?", (purge,))
                connection.execute("DELETE FROM broker_gate_queue WHERE touched<?", (purge,))
                owners = {owner for (owner,) in connection.execute(
                    "SELECT owner FROM broker_reservations UNION SELECT owner FROM broker_gate_queue")}
                dead = [(owner,) for owner in owners if owner != request.owner and self._owner_dead(owner)]
                if dead:
                    connection.executemany("DELETE FROM broker_reservations WHERE owner=?", dead)
                    connection.executemany("DELETE FROM broker_gate_queue WHERE owner=?", dead)
                connection.executemany("DELETE FROM broker_reservations WHERE token=? AND owner=?",
                                       [(token, request.owner) for token in request.release])
                connection.executemany("DELETE FROM broker_gate_queue WHERE seq=? AND owner=?",
                                       [(seq, request.owner) for seq in request.leave])
                connection.executemany(
                    "INSERT INTO broker_reservations VALUES (?,?,?,?,?) ON CONFLICT(token) DO UPDATE SET "
                    "slots=excluded.slots, phase=excluded.phase, updated=excluded.updated",
                    [(token, request.owner, int(slots), phase, now) for token, slots, phase in request.reservations])
                # Touching a ticket keeps its place; one deleted meanwhile comes back at its original position.
                connection.executemany(
                    "INSERT INTO broker_gate_queue VALUES (?,?,?,?) ON CONFLICT(seq) DO UPDATE SET touched=excluded.touched",
                    [(seq, request.owner, int(need), now) for _token, seq, need, _phase in request.waiting])
                enqueued = {}
                for token, need, _phase in request.enqueue:
                    enqueued[token] = connection.execute(
                        "INSERT INTO broker_gate_queue(owner, need, touched) VALUES (?,?,?)",
                        (request.owner, int(need), now)).lastrowid
                mine = {seq: (token, phase) for token, seq, _need, phase in request.waiting}
                mine.update({seq: (token, phase) for (token, _need, phase), seq in
                             zip(request.enqueue, enqueued.values())})
                source = "override" if capacity is not None else "keys"
                total = capacity if capacity is not None else self._slot_capacity(connection, files, cap, now)
                conflict = self._capacity_conflict(connection, request.owner, total, source, now)
                connection.execute(
                    "INSERT INTO broker_gate_capacity VALUES (1,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                    "capacity=excluded.capacity, source=excluded.source, owner=excluded.owner, at=excluded.at",
                    (int(total), source, request.owner, now))
                reservations, tickets = self._live_gate_rows(connection, now)
                reserved = sum(slots for _token, _owner, slots in reservations)
                admitted, ahead, granted, remaining = set(), 0, 0, []
                for seq, owner, need in tickets:
                    ahead += need
                    if (request.admit and owner == request.owner and seq in mine
                            and reserved + ahead <= total):
                        token, phase = mine[seq]
                        connection.execute("DELETE FROM broker_gate_queue WHERE seq=?", (seq,))
                        connection.execute("INSERT INTO broker_reservations VALUES (?,?,?,?,?)",
                                           (token, request.owner, int(need), phase, now))
                        admitted.add(token)
                        granted += need
                    else:
                        remaining.append(need)
                if admitted:
                    connection.execute("INSERT INTO broker_gate_totals VALUES (1, ?) ON CONFLICT(id) DO UPDATE SET "
                                       "admitted=admitted+excluded.admitted", (len(admitted),))
                reserved += granted
                inflight = connection.execute("SELECT count(*) FROM broker_leases WHERE expires>?", (now,)).fetchone()[0]
                waiting = sum(value for kind, value in self._live_counts(connection) if kind == "waiting")
                row = connection.execute("SELECT admitted FROM broker_gate_totals WHERE id=1").fetchone()
                commit(connection)
            finally:
                if connection.in_transaction:
                    connection.rollback()
        return GateCycle(capacity=int(total), reserved=int(reserved), running=len(reservations) + len(admitted),
                         queued=len(remaining), queued_slots=sum(remaining), inflight=int(inflight),
                         waiting=int(waiting), admitted_total=int(row[0]) if row else 0, enqueued=enqueued,
                         admitted=frozenset(admitted), capacity_source=source, capacity_conflict=conflict)

    def _capacity_conflict(self, connection: sqlite3.Connection, owner: str, total: int, source: str,
                           now: float) -> tuple[int, str] | None:
        """The capacity another live process stored moments ago, when it disagrees with a manual one here."""
        row = connection.execute("SELECT capacity, source, owner, at FROM broker_gate_capacity WHERE id=1").fetchone()
        if row is None or row[2] == owner or now - row[3] > GATE_CAPACITY_FRESH_S or self._owner_dead(row[2]):
            return None
        if "override" not in (source, row[1]) or (row[1], int(row[0])) == (source, int(total)):
            return None
        return int(row[0]), str(row[1])

    def gate_state(self, files: Sequence[str], *, cap: int | None, capacity: int | None = None) -> dict[str, object]:
        """Read the question-start slot gate's host state without taking the writer lock.

        Args:
            files: The pool's file names, used only when no key has been registered yet.
            cap: The host's per-key cap, or ``None`` when it cannot be read (the computed capacity is then unknown).
            capacity: The manual capacity override, when one is configured.

        The capacity the host's gate processes admitted against within :data:`GATE_CAPACITY_FRESH_S` wins over
        ``capacity`` and ``cap``, so a reader whose environment differs from the eval processes' still reports
        what the gate uses (``capacity_observed`` is then true).

        Returns:
            ``capacity`` (``None`` when unknown), ``capacity_source`` (``override``, ``keys`` or ``unknown``),
            ``capacity_observed``, ``reserved``, ``running``, ``queued``, ``queued_slots``, ``inflight``,
            ``waiting`` and ``admitted_total``.
        """
        with closing(self._connect()) as connection:
            now = self._clock()
            stored = connection.execute("SELECT capacity, source, at FROM broker_gate_capacity WHERE id=1").fetchone()
            observed = stored is not None and now - stored[2] <= GATE_CAPACITY_FRESH_S
            if observed:
                total, source = int(stored[0]), str(stored[1])
            elif capacity is not None:
                total, source = capacity, "override"
            elif cap is not None:
                total, source = self._slot_capacity(connection, files, cap, now), "keys"
            else:
                total, source = None, "unknown"
            reservations, tickets = self._live_gate_rows(connection, now)
            inflight = connection.execute("SELECT count(*) FROM broker_leases WHERE expires>?", (now,)).fetchone()[0]
            waiting = sum(value for kind, value in self._live_counts(connection) if kind == "waiting")
            row = connection.execute("SELECT admitted FROM broker_gate_totals WHERE id=1").fetchone()
        return {"capacity": total, "capacity_source": source, "capacity_observed": observed,
                "reserved": sum(slots for _token, _owner, slots in reservations), "running": len(reservations),
                "queued": len(tickets), "queued_slots": sum(need for _seq, _owner, need in tickets),
                "inflight": int(inflight), "waiting": int(waiting), "admitted_total": int(row[0]) if row else 0}

    def record_error(self, record: Mapping[str, object]) -> None:
        """Store one sanitized provider error event (:func:`.provider_errors.event`); see :meth:`record_errors`."""
        self.record_errors([record])

    def record_errors(self, records: Sequence[Mapping[str, object]]) -> None:
        """Store sanitized provider error events in one transaction and prune old ones.

        The writer waits at most 250 ms for the lock, so it never queues long in front of request writers; a busy
        broker raises, the transaction rolls back whole, and :mod:`.provider_errors` retries the batch with backoff.

        Args:
            records: ``at, status, code, type, message, model, file_name, key_id`` each; already redacted.
        """
        if not records:
            return
        fields = ("at", "status", "code", "type", "message", "model", "file_name", "key_id")
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA busy_timeout=250")
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.executemany("INSERT INTO broker_errors(at,status,code,type,message,model,file_name,key_id) "
                                       "VALUES (?,?,?,?,?,?,?,?)", [tuple(row[name] for name in fields) for row in records])
                newest = max(float(row["at"]) for row in records)
                connection.execute("DELETE FROM broker_errors WHERE at<?", (newest - ERROR_RETENTION_S,))
                commit(connection)
            finally:
                if connection.in_transaction:
                    connection.rollback()

    def errors(self, since: float) -> list[dict[str, object]]:
        """Provider error events at or after ``since`` (POSIX seconds), oldest first."""
        with closing(self._connect()) as connection:
            connection.row_factory = sqlite3.Row
            return [dict(row) for row in connection.execute(
                "SELECT at,status,code,type,message,model,file_name,key_id FROM broker_errors WHERE at>=? ORDER BY at, id",
                (since,))]

    def stall_notice(self) -> bool:
        """Permit at most one unavailable-pool warning per minute across the host's processes."""
        with closing(self._connect()) as connection:
            return bool(connection.execute("INSERT INTO broker_notices VALUES ('lent-volcano-files',?) "
                "ON CONFLICT(name) DO UPDATE SET last_at=excluded.last_at WHERE last_at<=?",
                (self._clock(), self._clock() - 60)).rowcount)

    def import_marks(self, marks: object) -> None:
        """Merge a secret-free marks export; retirement and later exhaustion can only increase.

        Args:
            marks: A JSON list of id/name/exhausted_until/retired records from export-marks.

        Raises:
            ValueError: Unknown fields, unapproved names, ids or invalid timestamps.
        """
        from .lent_keys import valid_key_name
        if not isinstance(marks, list):
            raise ValueError("Expected a file-key marks list.")
        for row in marks:
            if (not isinstance(row, dict) or set(row) != {"key_id", "file_name", "exhausted_until", "retired"}
                    or not valid_key_name(row["file_name"]) or row["key_id"] != file_key_id(row["file_name"])
                    or type(row["retired"]) is not int or row["retired"] not in (0, 1)
                    or type(row["exhausted_until"]) not in (int, float)
                    or not math.isfinite(row["exhausted_until"]) or row["exhausted_until"] < 0):
                raise ValueError("Invalid file-key mark; only approved names and ids are accepted.")
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA synchronous=FULL")
            begin_immediate(connection)
            try:
                for row in marks:
                    connection.execute("INSERT INTO broker_file_keys(key_id,file_name,exhausted_until,retired) VALUES (?,?,?,?) "
                        "ON CONFLICT(key_id) DO UPDATE SET exhausted_until=max(exhausted_until,excluded.exhausted_until),retired=max(retired,excluded.retired)",
                        (row["key_id"], row["file_name"], row["exhausted_until"], row["retired"]))
                commit(connection)
            finally:
                if connection.in_transaction:
                    connection.rollback()


class HostCounter:
    """Count something in this process in memory and publish it to the host broker for the host's other processes.

    Callers change the count in memory only (no SQLite on an event loop). A daemon thread publishes it about once a
    second when it changed, and every :data:`COUNT_HEARTBEAT_S` while it is nonzero; a dead process's row expires
    after :data:`COUNT_TTL_S`, or as soon as a reader finds its owner gone. Kinds: ``waiting`` (file-key admissions
    waiting for a slot, read by the question-start gate), ``gate_held`` (questions held at the question-start gate,
    read by the cap controller's host helper) and ``gate_active`` (questions held at or running past the gate: the
    gate divides its start budget by the processes with such questions).

    Args:
        broker: The host broker receiving the count.
        kind: One of :data:`COUNT_KINDS`.
    """

    def __init__(self, broker: "SqliteKeyBroker", kind: str) -> None:
        if kind not in COUNT_KINDS:
            raise ValueError(f"unknown host count: {kind}")
        self.broker = broker
        self.kind = kind
        # <pid>-<birth>-<nonce>: readers drop the row as soon as this process is gone (process_identity.owner_dead).
        self.owner = f"{owner_tag()}-{uuid4().hex[:12]}"
        self._lock = threading.Lock()
        self._count = 0
        self._published: int | None = None
        self._published_at = float("-inf")
        self._failures = 0
        self._thread: threading.Thread | None = None

    @property
    def count(self) -> int:
        """This process's current count."""
        with self._lock:
            return self._count

    def change(self, delta: int) -> None:
        """Add ``delta`` (never below zero) and start the publisher if needed."""
        with self._lock:
            self._count = max(0, self._count + delta)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name=f"host-count-{self.kind}", daemon=True)
                self._thread.start()

    def publish(self) -> None:
        """Publish the count now when it changed, or when a nonzero count's heartbeat is due."""
        with self._lock:
            value = self._count
            due = value != self._published or (value and time.monotonic() - self._published_at >= COUNT_HEARTBEAT_S)
        if not due:
            return
        try:
            written = self.broker.publish_count(self.kind, self.owner, value)
        except Exception as error:  # noqa: BLE001 - a lost count only weakens a gate or a report; never fail work
            _LOGGER.warning("[host-count] Could not publish the %s count (%s).", self.kind, type(error).__name__)
            written = False
        with self._lock:
            if written:
                self._published, self._published_at = value, time.monotonic()
                self._failures = 0
                return
            self._failures += 1
            warn = self._failures == COUNT_WARN_FAILURES
        if warn:
            _LOGGER.warning("[host-count] %d consecutive %s publications failed (broker busy); its row expires after "
                            "%.0f s without one.", COUNT_WARN_FAILURES, self.kind, COUNT_TTL_S)

    def close(self) -> None:
        """Publish zero at an orderly exit so this process's count leaves the host total at once."""
        with self._lock:
            published = self._published
            self._count = 0
        if published and self.broker.path.exists():
            self.publish()

    def _run(self) -> None:
        # Event.wait, not time.sleep: the publisher's period must not follow a caller's patched or paced sleep.
        tick = threading.Event()
        while True:
            tick.wait(1.0)
            self.publish()


_COUNTERS: dict[tuple[str, str], HostCounter] = {}
_COUNTERS_LOCK = threading.Lock()


def host_counter(broker: "SqliteKeyBroker", kind: str) -> HostCounter:
    """The process's counter of ``kind`` for one host broker."""
    with _COUNTERS_LOCK:
        counter = _COUNTERS.get((str(broker.path), kind))
        if counter is None or counter.broker is not broker:
            counter = _COUNTERS[(str(broker.path), kind)] = HostCounter(broker, kind)
        return counter


@atexit.register
def _close_counters() -> None:
    with _COUNTERS_LOCK:
        counters = list(_COUNTERS.values())
    for counter in counters:
        counter.close()


def file_key_id(name: str) -> str:
    """Return a stable cross-host opaque id derived from a file name, never from its secret."""
    return hashlib.sha256(("lent-volcano/" + name).encode()).hexdigest()[:24]


def _duration(text: str) -> float:
    """Seconds in ``90``, ``90s``, ``30m``, ``2h`` or ``1d``."""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    text = text.strip().lower()
    scale = units.get(text[-1:], None)
    value = float(text[:-1] if scale else text) * (scale or 1)
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid duration")
    return value


def error_report(events: Sequence[Mapping[str, object]], *, now: float, samples: int = 3) -> dict[str, object]:
    """Group provider error events by status, code and type, with counts, keys, models and recent messages.

    Args:
        events: Rows from :meth:`SqliteKeyBroker.errors`.
        now: POSIX seconds, for the event ages.
        samples: Distinct recent messages kept per group.

    Returns:
        ``{"events": n, "groups": [...]}``, largest group first.
    """
    groups: dict[tuple, dict[str, object]] = {}
    for row in events:
        key = (row["status"], row["code"], row["type"])
        group = groups.setdefault(key, {"status": row["status"], "code": row["code"], "type": row["type"], "count": 0,
                                        "keys": set(), "models": set(), "first_age_s": None, "last_age_s": None,
                                        "messages": []})
        group["count"] += 1
        group["keys"].add(row["file_name"])
        group["models"].add(row["model"])
        age = round(now - float(row["at"]), 1)
        group["first_age_s"] = age if group["first_age_s"] is None else group["first_age_s"]
        group["last_age_s"] = age
        if row["message"] and row["message"] not in group["messages"]:
            group["messages"] = ([row["message"]] + group["messages"])[:samples]
    rendered = []
    for group in sorted(groups.values(), key=lambda item: -item["count"]):
        rendered.append({**group, "keys": sorted(group["keys"]), "key_count": len(group["keys"]),
                         "models": sorted(group["models"])})
    return {"events": len(events), "groups": rendered}


def _keyproxy_report(state: Path) -> dict[str, object]:
    """The host key proxy's live stats (when ``RUBIKSQL_KEYPROXY_URL`` answers) and today's ledger summary."""
    from . import keyproxy
    report: dict[str, object] = {"state": str(state)}
    url = (os.environ.get(keyproxy.URL_ENV) or "").strip().rstrip("/")
    if url:
        try:   # the URL is validated before the token is sent to it
            endpoint = keyproxy.Endpoint(keyproxy.origin(url),
                                         Path(os.environ.get(keyproxy.TOKEN_FILE_ENV) or state / "token").expanduser())
            report["live"] = endpoint.call("/keyproxy/stats", timeout=5.0)
        except (OSError, ValueError, RuntimeError) as error:
            report["live"] = {"status": "unreachable", "error": type(error).__name__}
    ledger = state / "ledger" / f"{keyproxy.quota_day(time.time())}.jsonl"
    if ledger.exists():
        report["ledger_today"] = keyproxy.summarize_ledger(ledger)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Inspect host counters, exchange marks, or write the nonsecret lent pool configuration."""
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Lent file-pool host caps: RUBIKSQL_KEY_MAX_INFLIGHT or the live RUBIKSQL_KEY_CAP_FILE, "
        "integers 0-60 (0 pauses). The sum across all hosts must stay <= 60 per key."))
    parser.add_argument("command", choices=("stats", "export-marks", "import-marks", "make-pool", "errors"))
    parser.add_argument("--since", default="2h", help="errors: how far back, e.g. 90s, 30m, 2h, 1d (default 2h)")
    parser.add_argument("--state", type=Path, default=os.environ.get(BROKER_STATE_ENV))
    parser.add_argument("--file", type=Path)
    parser.add_argument("--source", type=Path)
    from .lent_keys import configured_key_dir, configured_key_names
    parser.add_argument("--key-dir", type=Path, default=configured_key_dir())
    parser.add_argument("--keyproxy-state", type=Path, default=os.environ.get("RUBIKSQL_KEYPROXY_STATE"),
                        help="stats/errors: also read the host key proxy's ledger in this state directory "
                             "(default RUBIKSQL_KEYPROXY_STATE, else beside the key directory when it exists)")
    args = parser.parse_args(argv)
    try:
        from .lent_keys import is_key_path
        if any(is_key_path(path) for path in (args.state, args.file, args.source) if path is not None):
            parser.error("broker state, reports and source configuration must be outside the lent key directory")
        if args.command == "make-pool":
            from .lent_keys import make_pool
            if args.source is None or args.file is None:
                parser.error("make-pool requires --source and --file")
            make_pool(args.source, args.file, args.key_dir)
            return 0
        state = args.state
        names = None
        configured_pool = os.environ.get("RUBIKSQL_GATEWAY_POOL")
        if configured_pool:
            config = json.loads(Path(configured_pool).expanduser().read_text(encoding="utf-8-sig"))
            if isinstance(config, dict) and "key_files" in config:
                from .lent_keys import validate_files
                paths = validate_files(config["key_files"])
                names = tuple(path.name for path in paths)
                state = state or paths[0].parent.parent / "volcano-host-broker.sqlite3"
        if names is None:
            names = configured_key_names(args.key_dir)
        broker = SqliteKeyBroker(state or args.key_dir.parent / "volcano-host-broker.sqlite3")
        from . import keyproxy
        proxy_state = (args.keyproxy_state or keyproxy.default_state_dir(args.key_dir)).expanduser()
        if args.command == "errors":
            now = time.time()
            events = broker.errors(now - _duration(args.since))
            if proxy_state.exists():
                events = sorted([*events, *keyproxy.ledger_errors(proxy_state, now - _duration(args.since))],
                                key=lambda row: float(row["at"]))
            print(json.dumps(error_report(events, now=now), indent=2))
            return 0
        if args.command == "import-marks":
            if args.file is None:
                parser.error("import-marks requires --file")
            broker.import_marks(json.loads(args.file.read_text(encoding="utf-8")))
        else:
            result = broker.stats()
            if args.command == "stats":
                from .lent_keys import CAP_ENV, CAP_FILE_ENV, read_host_cap, window_label
                from .question_gate import capacity_override
                cap_file = Path(os.environ.get(CAP_FILE_ENV) or broker.path.with_suffix(".cap")).expanduser()
                try:
                    cap = read_host_cap(cap_file, os.environ.get(CAP_ENV, "1"))
                except RuntimeError:
                    cap = None
                try:
                    override, override_error = capacity_override(os.environ), None
                except ValueError as error:
                    override, override_error = None, str(error)
                gate = broker.gate_state(names, cap=cap, capacity=override)
                if override_error is not None:
                    gate["capacity_override_error"] = override_error
                result = {**result, "window_end": window_label(), "question_gate": gate}
                if proxy_state.exists():
                    result["keyproxy"] = _keyproxy_report(proxy_state)
            if args.command == "export-marks":
                result = [{name: row[name] for name in ("key_id", "file_name", "exhausted_until", "retired")} for row in result["keys"]]
            rendered = json.dumps(result, indent=2)
            if args.file:
                args.file.write_text(rendered + "\n", encoding="utf-8")
            else:
                print(rendered)
        return 0
    except (OSError, ValueError, sqlite3.Error):
        parser.exit(2, "Cannot process file-key configuration or broker state; details withheld.\n")


if __name__ == "__main__":
    raise SystemExit(main())
