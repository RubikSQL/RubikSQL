"""Turn memo: replay the identical-prefix turns of an agent run from a recorded sibling.

When a prompt release changes one conditional text of a role's closure — a nudge, a note, a
grain correction, a hook frame — that role's cache key moves to a new namespace and every
run of the role would be re-sampled from scratch, at full cost and with fresh sampling
noise mixed into the measured effect. The memo removes both: a turn whose request bytes
are identical to a recorded turn reuses the recorded reply, and only the turns that really
saw the changed text are generated again. The run then reproduces its parent byte for byte
up to the first turn the change reached, and is a fresh sample from there on.

Recording is always on and writes only values, never keys. Two tables live in the same
SQLite database as ``llm_cache``; the memo's own write path creates them
(:func:`ensure_schema`), so a read-only or frozen cache file still serves ``llm_cache`` hits
and only loses recording and replay:

- ``llm_turn(turn_key, ref, idx, t, ns)``: one row per assistant message of a recorded run,
  plus one chain-start row per run (``idx`` = :data:`START_IDX`, ``turn_key`` = ``h0``) that
  says a transcript was recorded under this base key at all. ``turn_key`` is the hash-chain
  value of the request that produced the message, ``ref`` names the transcript
  (``llm_cache:<key>`` for a candidate, whose cached value already holds ``messages``;
  ``llm_episode:<key>`` for a judge or reviewer), ``idx`` is the message's index in that
  transcript, ``t`` the logical-clock reading when the reply arrived, ``ns`` the sampling
  universe. ``turn_key`` is the primary key and rows are written ``INSERT OR REPLACE``: the
  same request bytes recorded twice keep the latest transcript, and a row whose ``ref`` was
  later overwritten by another transcript fails the prefix check below and counts as a miss.
  A candidate whose whole run replays from ``llm_cache`` backfills these rows from the
  stored transcript when they are missing (:func:`transcript_rows`), so a root run served
  from an older cache still leaves a replay source.
- ``llm_episode(key, messages_z, think_z, t, ns, release)``: the judge and reviewer
  replay transcripts, written independently of successful answer caching, including captured
  fallback and failed attempts. A reply with a call its loop certainly fails on
  (:func:`reply_replayable`) remains in the audit transcript but is excluded from the replay
  index; lookup also rejects such replies indexed by older writers. Valid replies interrupted
  by tool failures remain replayable.
  The current result also carries its own messages for lab ingestion.
  ``messages_z`` is the zlib-compressed
  JSON of the messages, ``think_z`` the same for the per-turn reasoning texts, ``release``
  the prompt release that produced the run. Also ``INSERT OR REPLACE`` on ``key``.

The turn key is a hash chain over the request bytes (``rubiksql.turn/1``)::

    h0  = sha256(canon(["rubiksql.turn/1", base_key, ns, system or ""]))
    h_k = sha256(h_{k-1} + "\\n" + canon(messages appended since the previous request))

``base_key`` is the role's whole cache key computed WITHOUT its ``prompt`` term — the key
r0 rules would produce — so only runs that agree on everything but prompt content (model,
system prompt, seed, knowledge fingerprint, message shape, protocols, route, environment
knobs, universe) share turns. Lookups happen only when the role's ``key_extra`` is
non-empty (its closure differs from r0's) and neither ``--no-cache`` nor skip-barren is
active; under r0 with no namespace nothing is looked up, the provider is called exactly
as before and only the index rows are written. A hit rebuilds the assistant turn from the
recorded message with usage ``{}`` (the turn cost nothing), and aligns the run's
:class:`RunClock` so that ``elapsed()`` reads the source's ``t``: the time nudge then fires
on the same turn with the same ``{mins}``. Tools always execute live; a tool result that
differs from the source changes the next request's bytes, and the run turns to live
generation from there.

The first miss of a run that looks up is classified and written to the
``prompt_identity.replay`` fields (``turns``, ``replayed``, ``diverged_at``, ``cause``,
``source``): ``no-source`` when nothing was recorded under the base key, the recorded
transcript was overwritten with other bytes, or an interrupted source's exact prefix has
been exhausted — there is no recorded reply to compare against, so the continuation is
sampled live rather than classified as a prompt divergence; ``initial`` when a source exists and
the very first request differs from it; ``prompt:<entry>`` when an entry whose surface text
differs from the source release's rendered on the way to the differing request; ``tool`` when
the first differing message is a tool result; ``clock`` when the time nudge fired on one
side only; ``unknown`` otherwise. A change to a text the base key itself hashes (the
candidate, judge or reviewer system prompt) moves the base key, so the paired parent's
turns cannot replay. A matching earlier child attempt can still supply turns; otherwise
the run reports ``no-source``. The outcome settles it from first-request message digests. This module
depends on the standard library and the stdlib-only prompt runtime alone; it never imports
heavenbase.

``python -m rubiksql_hb.turn_memo gc --keep-ns <ns> ...`` drops the episodes and index
rows of universes that are no longer replayed; ``ls`` counts rows per universe.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import threading
import zlib
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence

from .prompts.canon import canon, sha256_hex
from . import cache_lineage
from . import session_timing

__all__ = [
    "CAUSE_CLOCK",
    "CAUSE_INITIAL",
    "CAUSE_NO_SOURCE",
    "CAUSE_PROMPT",
    "CAUSE_TOOL",
    "CAUSE_UNKNOWN",
    "ENV_CACHE_NS",
    "REF_CACHE",
    "REF_EPISODE",
    "SCHEMA_TURN",
    "START_IDX",
    "TIME_NUDGE_SURFACE",
    "Hit",
    "RunClock",
    "Source",
    "TurnMemo",
    "TurnRow",
    "TurnStore",
    "cache_db_path",
    "chain_extend",
    "chain_over",
    "chain_start",
    "changed_entries",
    "connect_cache",
    "ensure_schema",
    "main",
    "reply_replayable",
    "start_row",
    "transcript_rows",
]

SCHEMA_TURN = "rubiksql.turn/1"
ENV_CACHE_NS = "RUBIKSQL_CACHE_NS"
#: ``ref`` prefixes of the two transcript stores an index row can point into.
REF_CACHE = "llm_cache:"
REF_EPISODE = "llm_episode:"
#: Divergence causes; ``CAUSE_PROMPT`` is a prefix completed with the entry name.
CAUSE_INITIAL = "initial"
CAUSE_NO_SOURCE = "no-source"
CAUSE_PROMPT = "prompt:"
CAUSE_TOOL = "tool"
CAUSE_CLOCK = "clock"
CAUSE_UNKNOWN = "unknown"
#: ``idx`` of the chain-start row a recorded run writes next to its per-message rows; no
#: assistant message can sit before the first message, so it never matches a lookup.
START_IDX = -1
#: The one surface whose firing depends on the clock rather than on the transcript.
TIME_NUDGE_SURFACE = "candidate.nudge.time"
#: Shortest static fragment of the time-nudge template that identifies it in a message.
_MARKER_MIN = 8


# ---------------------------------------------------------------------------------------
# Database location and schema


def cache_db_path(environ: Mapping[str, str] | None = None) -> str:
    """Return the path of the LLM cache database, the file the memo tables share.

    ``RUBIKSQL_CACHE_DB`` names the file directly; otherwise it is ``llm_cache.db`` under
    ``RUBIKSQL_DATA_DIR`` (default ``~/.rubiksql_hb``). ``nl2sql._cache_db_path`` delegates
    here, so both readers resolve the same file.

    Args:
        environ (Mapping[str, str] | None): The environment of the process that will use the
            cache, when it is not this one (a launcher resolving its workers' file);
            ``os.environ`` when None.

    Returns:
        str: Absolute path of the database file.
    """
    environ = os.environ if environ is None else environ
    override = environ.get("RUBIKSQL_CACHE_DB")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    data_dir = environ.get("RUBIKSQL_DATA_DIR", os.path.expanduser("~/.rubiksql_hb"))
    return os.path.join(data_dir, "llm_cache.db")


def connect_cache(path: str) -> sqlite3.Connection:
    """Open the shared cache with the knowledge database's two-minute busy wait.

    Args:
        path: Cache file selected by :func:`cache_db_path` or an explicit CLI path.

    Returns:
        A connection owned by the caller. Existing journal mode and read-only
        cache-hit behavior are preserved; this does not create any tables.

    Raises:
        OSError: The cache directory cannot be created.
        sqlite3.Error: SQLite cannot open the cache file.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return sqlite3.connect(path, timeout=120, check_same_thread=False)


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the ``llm_turn`` and ``llm_episode`` tables when they are missing.

    :class:`TurnStore` runs this on its write path only, so opening the cache database for
    an ``llm_cache`` read never needs write access to it.

    Args:
        conn (sqlite3.Connection): An open connection to the cache database.

    Returns:
        None: This function does not return a value.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_turn (
            turn_key TEXT PRIMARY KEY,
            ref      TEXT NOT NULL,
            idx      INTEGER NOT NULL,
            t        REAL NOT NULL,
            ns       TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_episode (
            key        TEXT PRIMARY KEY,
            messages_z BLOB NOT NULL,
            think_z    BLOB NOT NULL,
            t          REAL NOT NULL,
            ns         TEXT NOT NULL DEFAULT '',
            release    TEXT NOT NULL DEFAULT ''
        )
        """
    )


# ---------------------------------------------------------------------------------------
# Logical clock


class RunClock:
    """The logical clock of one agent loop: real elapsed time plus a replay offset.

    The candidate loop reads ``elapsed()`` for its hard deadline, its bounce suppression and
    its time nudge. Every replayed turn aligns the clock to the source run's reading at
    that turn, so the nudge fires on the same turn with the same ``{mins}``; with no replay
    the offset stays zero and ``elapsed()`` equals the wall clock exactly as before.
    ``real_elapsed()`` never carries the offset and feeds ``duration_sec``.
    """

    def __init__(self, now: Callable[[], float]) -> None:
        """Start the clock.

        Args:
            now (Callable[[], float]): The monotonic time source; the loop passes its own
                module's ``time.monotonic`` so a patched clock stays in charge.
        """
        self._now = now
        self._start = now()
        self._offset = 0.0

    @property
    def offset(self) -> float:
        """The replay offset in seconds, zero when nothing was replayed.

        Returns:
            float: Seconds added to the wall clock by :meth:`align`.
        """
        return self._offset

    def real_elapsed(self) -> float:
        """Seconds of wall clock since the clock started.

        Args:
            None.

        Returns:
            float: Real elapsed seconds.
        """
        return self._now() - self._start

    def elapsed(self) -> float:
        """The logical reading: wall clock plus the replay offset.

        Args:
            None.

        Returns:
            float: Logical elapsed seconds.
        """
        return self.real_elapsed() + self._offset

    def align(self, t: float) -> None:
        """Move the offset so that ``elapsed()`` reads ``t`` right now.

        Args:
            t (float): The source run's logical reading at the replayed turn.

        Returns:
            None: This method does not return a value.
        """
        self._offset = t - self.real_elapsed()


# ---------------------------------------------------------------------------------------
# Hash chain


def chain_start(base_key: str, ns: str, system: str | None = None) -> str:
    """The chain's first link: universe, base key and the separate system prompt, if any.

    Args:
        base_key (str): The role's cache key computed without its ``prompt`` term.
        ns (str): ``RUBIKSQL_CACHE_NS``, empty when unset.
        system (str | None): A system prompt sent outside ``messages`` (the Anthropic shape);
            ``None`` when the system prompt is the first message.

    Returns:
        str: ``h0`` as 64 hex characters.
    """
    return sha256_hex(canon([SCHEMA_TURN, base_key, ns, system or ""]))


def chain_extend(previous: str, appended: Sequence[Mapping[str, Any]]) -> str:
    """One more link: the messages appended since the previous request.

    Args:
        previous (str): ``h_{k-1}``.
        appended (Sequence[Mapping[str, Any]]): The new messages, in request order.

    Returns:
        str: ``h_k`` as 64 hex characters.
    """
    return sha256_hex(previous + "\n" + canon(list(appended)))


def chain_over(base_key: str, ns: str, messages: Sequence[Mapping[str, Any]], system: str | None = None) -> dict[int, str]:
    """Recompute the turn key of every assistant message of a recorded transcript.

    A request ends right before each assistant message, so the transcript is cut into the
    segments the live loop appended between two requests and chained in order.

    Args:
        base_key (str): The role's base key.
        ns (str): The universe.
        messages (Sequence[Mapping[str, Any]]): The full transcript.
        system (str | None): The separate system prompt, if the shape has one.

    Returns:
        dict[int, str]: Assistant message index to its turn key.
    """
    keys: dict[int, str] = {}
    link = chain_start(base_key, ns, system)
    start = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        link = chain_extend(link, messages[start:index])
        keys[index] = link
        start = index
    return keys


# ---------------------------------------------------------------------------------------
# Rows and the store


@dataclass(frozen=True)
class TurnRow:
    """One ``llm_turn`` row.

    Attributes:
        turn_key (str): Hash-chain value of the request (``h0`` for the chain-start row).
        ref (str): ``llm_cache:<key>`` or ``llm_episode:<key>``.
        idx (int): Index of the assistant message in the referenced transcript, or
            :data:`START_IDX` for the chain-start row.
        t (float): Logical-clock reading when the reply arrived (``0.0`` for the chain-start row).
    """

    turn_key: str
    ref: str
    idx: int
    t: float


def start_row(base_key: str, ns: str, system: str | None = None) -> TurnRow:
    """The chain-start row of a recorded run: ``h0`` marks that a transcript exists under the base key.

    Args:
        base_key (str): The role's base key.
        ns (str): The universe.
        system (str | None): The separate system prompt, if the shape has one.

    Returns:
        TurnRow: ``idx`` :data:`START_IDX`, ``t`` 0.0, ``ref`` left empty for the writer to fill.
    """
    return TurnRow(chain_start(base_key, ns, system), "", START_IDX, 0.0)


def transcript_rows(base_key: str, ns: str, messages: Sequence[Mapping[str, Any]], system: str | None = None) -> list[TurnRow]:
    """Index rows of a recorded transcript whose clock readings are unknown.

    A candidate served whole from ``llm_cache`` has its transcript but not the logical-clock
    readings its turns were recorded with, so every row carries ``t = 0.0``: a later replay
    aligned to it never re-fires the time nudge, and a source that did fire it (the nudge is a
    user message of the transcript) diverges there as ``clock``, exactly as an unknown clock
    should.

    Args:
        base_key (str): The role's base key.
        ns (str): The universe.
        messages (Sequence[Mapping[str, Any]]): The full transcript.
        system (str | None): The separate system prompt, if the shape has one.

    Returns:
        list[TurnRow]: The chain-start row followed by one row per assistant message, ``ref`` empty.
    """
    rows = [start_row(base_key, ns, system)]
    rows += [TurnRow(key, "", idx, 0.0) for idx, key in chain_over(base_key, ns, messages, system).items()]
    return rows


@dataclass(frozen=True)
class Source:
    """A recorded transcript an index row points into.

    Attributes:
        ref (str): The row's ``ref``.
        messages (list[dict[str, Any]]): The transcript.
        thinks (list[str]): Per-assistant-message reasoning texts; empty when unrecorded.
        release (str | None): The prompt release that produced it, when recorded.
        interrupted (bool): The immutable source explicitly records an interrupted attempt.
    """

    ref: str
    messages: list[dict[str, Any]]
    thinks: list[str]
    release: str | None
    proof: dict[str, Any] | None = None
    interrupted: bool = False


@dataclass(frozen=True)
class Hit:
    """A verified replayable turn.

    Attributes:
        row (TurnRow): The index row.
        source (Source): The transcript it points into.
        message (dict[str, Any]): The recorded assistant message.
        think (str): Its recorded reasoning text, empty when none was kept.
    """

    row: TurnRow
    source: Source
    message: dict[str, Any]
    think: str


_STORE_ERRORS = (sqlite3.Error, OSError, json.JSONDecodeError, zlib.error, ValueError, TypeError)


class TurnStore:
    """The memo tables behind a connection factory, failing soft like the LLM cache.

    Every operation opens a connection through ``connect`` (``nl2sql._get_cache_conn``),
    runs under ``lock`` and closes the connection. Writes create the two tables first; a
    read against a database that has none is a plain miss, since nothing was recorded. A
    storage error is reported once per store through ``report`` (the memo's own alarm,
    never the LLM cache's) and turns into a miss or a dropped write, never into an
    exception: the memo only ever saves money, it never decides correctness.
    """

    def __init__(
        self,
        connect: Callable[[], sqlite3.Connection],
        *,
        lock: "threading.Lock | None" = None,
        report: Callable[[str, BaseException], None] | None = None,
    ) -> None:
        """Bind the store to a connection factory.

        Args:
            connect (Callable[[], sqlite3.Connection]): Opens the cache database.
            lock (threading.Lock | None): Serializes access; a private lock when ``None``.
            report (Callable[[str, BaseException], None] | None): Receives ``(operation, error)``
                on the first storage failure; silent when ``None``.
        """
        self._connect = connect
        self._lock = lock if lock is not None else threading.Lock()
        self._report = report
        self._reported = False

    def _run(self, op: str, fn: Callable[[sqlite3.Connection], Any], *, write: bool = False) -> Any:
        with session_timing.locked(self._lock, "cache_lock"), session_timing.measure("gate", f"turn_memo_{op}"):
            try:
                conn = self._connect()
                try:
                    if write:
                        ensure_schema(conn)
                    result = fn(conn)
                    if write:
                        conn.commit()
                finally:
                    conn.close()
                return result
            except sqlite3.OperationalError as exc:
                # A database nothing has recorded into yet has no memo tables: a miss, not a fault.
                if not write and str(exc).startswith("no such table"):
                    return None
                self._fail(op, exc)
                return None
            except _STORE_ERRORS as exc:
                self._fail(op, exc)
                return None

    def _fail(self, op: str, exc: BaseException) -> None:
        if self._report is None or self._reported:
            return
        self._reported = True
        self._report(op, exc)

    def write_turns(self, ref: str, ns: str, turns: Iterable[TurnRow]) -> bool:
        """Write the index rows of one run (``INSERT OR REPLACE`` on ``turn_key``).

        Args:
            ref (str): The transcript reference every row points to.
            ns (str): The universe.
            turns (Iterable[TurnRow]): Rows; their own ``ref`` is ignored in favour of ``ref``.

        Returns:
            bool: Whether the rows were written.
        """
        rows = [(turn.turn_key, ref, int(turn.idx), float(turn.t), ns) for turn in turns]

        def write(conn: sqlite3.Connection) -> bool:
            conn.executemany("INSERT OR REPLACE INTO llm_turn (turn_key, ref, idx, t, ns) VALUES (?, ?, ?, ?, ?)", rows)
            return True

        return bool(self._run("turn index write", write, write=True))

    def write_source(self, record: Mapping[str, Any], producer: Mapping[str, Any]) -> dict[str, Any] | None:
        """Persist immutable producer evidence before publishing any lookup reference."""
        if not producer.get("run") or not producer.get("qid"):
            return None
        return self._run("immutable source write", lambda conn: cache_lineage.write(conn, record, producer), write=True)

    def write_episode(self, key: str, *, messages: Sequence[Mapping[str, Any]], thinks: Sequence[str], t: float, ns: str, release: str, turns: Iterable[TurnRow], source: Mapping[str, Any] | None = None) -> bool:
        """Write a judge or reviewer transcript and its index rows.

        Args:
            key (str): The run's cache key (the ``llm_cache`` row it accompanies).
            messages (Sequence[Mapping[str, Any]]): The transcript.
            thinks (Sequence[str]): Reasoning text per assistant message, in order.
            t (float): The clock reading at the end of the run.
            ns (str): The universe.
            release (str): The prompt release id.
            turns (Iterable[TurnRow]): The run's index rows.

        Returns:
            bool: Whether everything was written.
        """
        messages_z = zlib.compress(canon(list(messages)).encode("utf-8"))
        think_z = zlib.compress(canon(list(thinks)).encode("utf-8"))
        ref = cache_lineage.REF + source["id"] if cache_lineage.valid_descriptor(source) else REF_EPISODE + key
        rows = [(turn.turn_key, ref, int(turn.idx), float(turn.t), ns) for turn in turns]

        def write(conn: sqlite3.Connection) -> bool:
            conn.execute(
                "INSERT OR REPLACE INTO llm_episode (key, messages_z, think_z, t, ns, release) VALUES (?, ?, ?, ?, ?, ?)",
                (key, messages_z, think_z, float(t), ns, release),
            )
            conn.executemany("INSERT OR REPLACE INTO llm_turn (turn_key, ref, idx, t, ns) VALUES (?, ?, ?, ?, ?)", rows)
            return True

        return bool(self._run("episode write", write, write=True))

    def lookup(self, turn_key: str) -> TurnRow | None:
        """Read the index row of one turn key.

        Args:
            turn_key (str): The chain value of the request.

        Returns:
            TurnRow | None: The row, or ``None`` when absent or unreadable.
        """
        def read(conn: sqlite3.Connection) -> TurnRow | None:
            row = conn.execute("SELECT ref, idx, t FROM llm_turn WHERE turn_key = ?", (turn_key,)).fetchone()
            return None if row is None else TurnRow(turn_key, str(row[0]), int(row[1]), float(row[2]))

        return self._run("turn index read", read)

    def read_source(self, ref: str) -> Source | None:
        """Load the transcript an index row points into.

        Args:
            ref (str): ``llm_cache:<key>`` or ``llm_episode:<key>``.

        Returns:
            Source | None: The transcript, or ``None`` when the reference is unknown, missing or malformed.
        """
        def read(conn: sqlite3.Connection) -> Source | None:
            if ref.startswith(cache_lineage.REF):
                document = cache_lineage.read(conn, ref)
                if document is None:
                    return None
                payload = document["payload"]
                messages, thinks = payload.get("memo_messages", payload.get("messages")), payload.get("reasoning", [])
                if not isinstance(messages, list) or not isinstance(thinks, list):
                    return None
                return Source(ref, messages, [str(item) for item in thinks], document["identity"].get("release"),
                              cache_lineage.descriptor(document), interrupted=payload.get("transcript_status") == "interrupted")
            if ref.startswith(REF_CACHE):
                row = conn.execute("SELECT response FROM llm_cache WHERE cache_key = ?", (ref[len(REF_CACHE):],)).fetchone()
                if row is None:
                    return None
                record = json.loads(row[0])
                messages = record.get("messages") if isinstance(record, dict) else None
                if not isinstance(messages, list):
                    return None
                identity = record.get("prompt_identity") if isinstance(record.get("prompt_identity"), dict) else {}
                release = identity.get("release")
                reasoning = record.get("reasoning")
                thinks = [item if isinstance(item, str) else "" for item in reasoning] if isinstance(reasoning, list) else []
                return Source(ref, messages, thinks, release if isinstance(release, str) else None)
            if ref.startswith(REF_EPISODE):
                row = conn.execute("SELECT messages_z, think_z, release FROM llm_episode WHERE key = ?", (ref[len(REF_EPISODE):],)).fetchone()
                if row is None:
                    return None
                messages = json.loads(zlib.decompress(row[0]).decode("utf-8"))
                thinks = json.loads(zlib.decompress(row[1]).decode("utf-8"))
                if not isinstance(messages, list) or not isinstance(thinks, list):
                    return None
                return Source(ref, messages, [str(item) for item in thinks], str(row[2]) or None)
            return None

        return self._run("transcript read", read)

    def counts(self) -> dict[str, dict[str, int]]:
        """Count index rows and episodes per universe.

        Args:
            None.

        Returns:
            dict[str, dict[str, int]]: ``{ns: {"turns": n, "episodes": m}}``; the default universe is ``""``.
        """
        def read(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
            out: dict[str, dict[str, int]] = {}
            for ns, n in conn.execute("SELECT ns, COUNT(*) FROM llm_turn GROUP BY ns"):
                out.setdefault(str(ns), {"turns": 0, "episodes": 0})["turns"] = int(n)
            for ns, n in conn.execute("SELECT ns, COUNT(*) FROM llm_episode GROUP BY ns"):
                out.setdefault(str(ns), {"turns": 0, "episodes": 0})["episodes"] = int(n)
            return out

        return self._run("count", read) or {}

    def gc(self, keep_ns: Iterable[str], *, drop_default: bool = False, dry_run: bool = False) -> dict[str, int]:
        """Drop the episodes and index rows of universes that are no longer replayed.

        Args:
            keep_ns (Iterable[str]): Universes to keep.
            drop_default (bool): Also drop the default universe (``ns == ""``), kept otherwise.
            dry_run (bool): Count without deleting.

        Returns:
            dict[str, int]: ``{"turns": rows dropped, "episodes": rows dropped}``; zeros when the store failed.
        """
        keep = set(keep_ns)
        if not drop_default:
            keep.add("")
        placeholders = ",".join("?" for _ in keep) or "''"
        params = tuple(sorted(keep))

        def run(conn: sqlite3.Connection) -> dict[str, int]:
            where = f"ns NOT IN ({placeholders})" if keep else "1"
            turns = int(conn.execute(f"SELECT COUNT(*) FROM llm_turn WHERE {where}", params).fetchone()[0])
            episodes = int(conn.execute(f"SELECT COUNT(*) FROM llm_episode WHERE {where}", params).fetchone()[0])
            if not dry_run:
                conn.execute(f"DELETE FROM llm_turn WHERE {where}", params)
                conn.execute(f"DELETE FROM llm_episode WHERE {where}", params)
            return {"turns": turns, "episodes": episodes}

        return self._run("gc", run, write=True) or {"turns": 0, "episodes": 0}


# ---------------------------------------------------------------------------------------
# Divergence classification helpers


def changed_entries(current: Any, source_release: str | None, entries: Iterable[str]) -> frozenset[str]:
    """Entries of the running release whose surfaces read differently in the source run's release.

    The comparison is by surface identity text (``Release.surface_identity``: the text as written,
    whatever rows carry it), so a release that re-cuts the same texts into other rows blames
    nothing, and an entry is blamed exactly when a surface it provides changed. An entry the code
    table does not know is blamed when the source release lacks it or carries it with other
    content. Exact recorded source releases must exist in the verified execution
    snapshot; an absent source refuses attribution.

    Args:
        current (Release): The release the process runs.
        source_release (str | None): The release id recorded with the source transcript.
        entries (Iterable[str]): The running release's entries in the run's role/route closure.

    Returns:
        frozenset[str]: Entry names of the running release whose surface text differs.
    """
    if source_release == current.id:
        return frozenset()
    from .prompts import runtime as RT

    if source_release is None:
        from .prompts.store import PromptStoreError

        raise PromptStoreError("recorded turn-memo source has no exact prompt release")
    source = RT.load_release(source_release, allow_retired=True)
    from .prompts import entries as E

    mine, theirs = current.surface_identity(), source.surface_identity()
    changed = set()
    for entry in entries:
        if entry not in current.entries:
            continue
        known = E.ENTRIES.get(entry)
        if known is None:
            if source.refs.get(entry) != current.refs.get(entry):
                changed.add(entry)
        elif any(mine.get(surface) != theirs.get(surface) for surface in known.surfaces):
            changed.add(entry)
    return frozenset(changed)


def _is_tool_message(message: Mapping[str, Any]) -> bool:
    if message.get("role") == "tool":
        return True
    content = message.get("content")
    if isinstance(content, list):
        return any(isinstance(block, Mapping) and block.get("type") == "tool_result" for block in content)
    return False


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, Mapping):
                value = block.get("text") if block.get("type") == "text" else block.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)
    return ""


def _nudge_marker(release: Any) -> str:
    """The longest static fragment of the time-nudge text, or ``""`` when unavailable."""
    try:
        template = str(release.surface_identity()[TIME_NUDGE_SURFACE])
    except Exception:  # noqa: BLE001 - a release without the surface has no clock-driven text
        return ""
    pieces = sorted((piece.strip() for piece in template.split("{mins}")), key=len, reverse=True)
    return pieces[0] if pieces and len(pieces[0]) >= _MARKER_MIN else ""


# ---------------------------------------------------------------------------------------
# The per-run memo


@dataclass
class _TurnRecord:
    """What the memo keeps about one turn of the running loop."""

    idx: int
    turn_key: str
    t: float
    think: str
    replayed: bool


def reply_replayable(message: Mapping[str, Any],
                     validate_calls: Callable[[list[tuple[str, Any]]], bool] | None = None) -> bool:
    """Reject a recorded reply its loop certainly fails on, without inspecting live tool results.

    The agent loops decode every call of a reply before any guard runs, so a malformed call
    list or arguments that do not decode disqualify the reply wherever they appear.
    ``validate_calls`` then judges the calls its dispatcher certainly reaches.

    Args:
        message (Mapping[str, Any]): Recorded assistant reply.
        validate_calls (Callable | None): The owning dispatcher's judgement of the decoded
            calls, given as ``(recorded name, decoded arguments)`` pairs in reply order.

    Returns:
        bool: Whether the reply may replay.
    """
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        return False
    decoded: list[tuple[str, Any]] = []
    for call in calls:
        if not isinstance(call, Mapping):
            return False
        function = call.get("function") or {}
        if not isinstance(function, Mapping):
            return False
        raw = str(function.get("arguments") or "{}")
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            try:
                # Match nl2sql._parse_tool_args: a complete object followed by
                # gateway debris is salvageable and must keep replaying.
                arguments, _ = json.JSONDecoder().raw_decode(raw)
            except json.JSONDecodeError:
                return False
        decoded.append((str(function.get("name") or ""), arguments))
    return validate_calls is None or validate_calls(decoded)


class TurnMemo:
    """Record and replay the turns of one agent run.

    Wrap each provider call as ``await memo.turn(call, messages)``: the memo extends the hash
    chain with the messages appended since the previous request, replays a verified hit or
    awaits ``call()``, records the turn, and classifies the first miss. After the run,
    :meth:`write_candidate_index` persists a cached candidate's rows; :meth:`write_episode`
    records judge/reviewer turns independently of answer caching. :meth:`replay_fields`
    gives the recorder its ``replay`` block.
    """

    def __init__(
        self,
        store: TurnStore | None,
        *,
        base_key: str,
        ns: str,
        clock: RunClock,
        rebuild: Callable[[dict[str, Any], str], Any],
        lookup: bool = False,
        system: str | None = None,
        recorder: Any = None,
        release: Any = None,
        role: str = "candidate",
        route: str | None = None,
        slot: int | None = None,
        validate_calls: Callable[[list[tuple[str, Any]]], bool] | None = None,
    ) -> None:
        """Start a memo for one run.

        Args:
            store (TurnStore | None): The memo tables; ``None`` records in memory only.
            base_key (str): The role's cache key without its ``prompt`` term.
            ns (str): The sampling universe (``RUBIKSQL_CACHE_NS``), empty when unset.
            clock (RunClock): The run's logical clock, aligned on every hit.
            rebuild (Callable[[dict[str, Any], str], Any]): Builds the loop's turn object from a
                recorded assistant message and its reasoning text.
            lookup (bool): Whether to replay; recording happens regardless.
            system (str | None): A system prompt sent outside ``messages``, for the chain start.
            recorder (RenderRecorder | None): The run's render recorder, read for divergence attribution.
            release (Release | None): The running release, read for divergence attribution.
            role (str): ``candidate``, ``judge`` or ``reviewer``.
            route (str | None): The route, ``None`` for the default.
            slot (int | None): A candidate's position in the pool; its own document (``candidate.s<k>``) joins the
                entries a divergence can be blamed on.
            validate_calls (Callable | None): The dispatcher's judgement of a reply's decoded
                calls (:func:`reply_replayable`), shared by current index writes and legacy
                index lookups.
        """
        self._store = store
        self._base_key = base_key
        self._ns = ns
        self._clock = clock
        self._rebuild = rebuild
        self._lookup = bool(lookup) and store is not None
        self._system = system
        self._start = chain_start(base_key, ns, system)
        self._link = self._start
        self._stale = False
        self._seen = 0
        self._turn = 0
        self._recorder = recorder
        self._release = release
        self._role = role
        self._route = route
        self._slot = slot
        self._validate_calls = validate_calls
        self._records: list[_TurnRecord] = []
        self._source: Source | None = None
        self._source_keys: dict[int, str] = {}
        self._fired: dict[str, int] = self._fired_now()
        self._changed: frozenset[str] | None = None
        self.diverged_at: int | None = None
        self.cause: str | None = None
        self.source_ref: str | None = None
        self.sources: list[dict[str, Any]] = []

    # -- public surface --------------------------------------------------------------

    @property
    def lookup(self) -> bool:
        """Whether this run replays (``key_extra`` non-empty and no cache bypass).

        Returns:
            bool: The lookup switch.
        """
        return self._lookup

    @property
    def turns(self) -> int:
        """Number of requests made so far.

        Returns:
            int: Turn count.
        """
        return self._turn

    @property
    def replayed(self) -> int:
        """Number of turns served from the memo.

        Returns:
            int: Hit count.
        """
        return sum(1 for record in self._records if record.replayed)

    def rows(self) -> list[TurnRow]:
        """The index rows of the turns so far, ``ref`` left empty for the writer to fill.

        Args:
            None.

        Returns:
            list[TurnRow]: The chain-start row, then one row per assistant message, in order.
        """
        rows = [start_row(self._base_key, self._ns, self._system)]
        rows += [TurnRow(record.turn_key, "", record.idx, record.t) for record in self._records]
        return rows

    def thinks(self) -> list[str]:
        """Reasoning text per turn, in order.

        Args:
            None.

        Returns:
            list[str]: One entry per assistant message.
        """
        return [record.think for record in self._records]

    def replay_fields(self, turns: int | None = None) -> dict[str, Any]:
        """The ``prompt_identity.replay`` block of this run.

        Args:
            turns (int | None): The loop's own turn counter, when it should be reported instead of
                the number of requests (a loop that stopped at its deadline counts the turn it
                did not send, exactly as it did before the memo existed).

        Returns:
            dict[str, Any]: ``{"turns", "replayed", "diverged_at", "cause", "source"}``.
        """
        return {
            "turns": self._turn if turns is None else int(turns),
            "replayed": self.replayed,
            "diverged_at": self.diverged_at,
            "cause": self.cause,
            "source": self.source_ref,
            **({"sources": list(self.sources)} if any(item["source"] is not None for item in self.sources) else {}),
        }

    async def turn(self, call: Callable[[], Awaitable[Any]], messages: Sequence[Mapping[str, Any]]) -> Any:
        """Serve one model turn: a verified replay, or the live call.

        Args:
            call (Callable[[], Awaitable[Any]]): Performs the provider call for ``messages``.
            messages (Sequence[Mapping[str, Any]]): The request as the loop holds it; the
                assistant reply will be appended at ``len(messages)``.

        Returns:
            Any: The turn object — the recorded one rebuilt through ``rebuild`` on a hit, else ``call()``'s.
        """
        appended = list(messages[self._seen:])
        self._link = chain_extend(self._link, appended)
        self._seen = len(messages)
        self._turn += 1
        idx = len(messages)
        hit = self._find(self._link, idx) if self._lookup else None
        cache_lineage.checkpoint("awaiting_model_reply")
        if hit is not None:
            self._clock.align(hit.row.t)
            if self.source_ref is None:
                self.source_ref = hit.row.ref
            self.sources.append({"source": hit.source.proof, "ref": hit.row.ref, "idx": hit.row.idx})
            self._records.append(_TurnRecord(idx, self._link, hit.row.t, hit.think, True))
            self._fired = self._fired_now()
            response = self._rebuild(dict(hit.message), hit.think)
            cache_lineage.checkpoint("model_reply_received", reply=getattr(response, "message", None))
            return response
        if self._lookup and self.diverged_at is None:
            self.diverged_at = self._turn
            self.cause = self._classify(messages)
        response = await call()
        think = str(getattr(response, "think", "") or "")
        self._records.append(_TurnRecord(idx, self._link, self._clock.elapsed(), think, False))
        self._fired = self._fired_now()
        cache_lineage.checkpoint("model_reply_received", reply=getattr(response, "message", None))
        return response

    def write_candidate_index(self, cache_key: str, messages: Sequence[Mapping[str, Any]],
                              source: Mapping[str, Any] | None = None) -> bool:
        """Persist the index rows of a candidate run next to its cache row.

        Args:
            cache_key (str): The candidate's cache key (the ``llm_cache`` row holding ``messages``).
            messages (Sequence[Mapping[str, Any]]): The complete audit transcript.

        Returns:
            bool: Whether the rows were written.
        """
        if self._store is None or not self._records:
            return False
        ref = cache_lineage.REF + source["id"] if cache_lineage.valid_descriptor(source) else REF_CACHE + cache_key
        return self._store.write_turns(ref, self._ns, self._replay_rows(messages))

    def write_episode(self, cache_key: str, messages: Sequence[Mapping[str, Any]], release: str, source: Mapping[str, Any] | None = None) -> bool:
        """Persist a judge or reviewer transcript with its reasoning and index rows.

        Args:
            cache_key (str): The run's cache key.
            messages (Sequence[Mapping[str, Any]]): The full transcript at the end of the run.
            release (str): The prompt release id.

        Returns:
            bool: Whether everything was written.
        """
        if self._store is None or not self._records:
            return False
        turns = self._replay_rows(messages)
        return self._store.write_episode(cache_key, messages=messages, thinks=self.thinks(), t=self._clock.elapsed(), ns=self._ns, release=release, turns=turns, source=source)

    # -- internals -------------------------------------------------------------------

    def _fired_now(self) -> dict[str, int]:
        fired = getattr(self._recorder, "fired", None)
        if not isinstance(fired, Mapping):
            return {}
        return {entry: int(data.get("n", 0)) for entry, data in fired.items() if isinstance(data, Mapping)}

    def _fired_delta(self) -> set[str]:
        now = self._fired_now()
        return {entry for entry, n in now.items() if n > self._fired.get(entry, 0)}

    def _replay_rows(self, messages: Sequence[Mapping[str, Any]]) -> list[TurnRow]:
        return [row for row in self.rows() if row.idx == START_IDX
                or (row.idx < len(messages) and reply_replayable(messages[row.idx], self._validate_calls))]

    def _find(self, turn_key: str, idx: int) -> Hit | None:
        """Look the turn up and verify the source prefix hashes to the same key.

        A row that exists but whose transcript no longer verifies (overwritten, unreadable,
        not an assistant message) is a stale index: the memo notes it, because on the first
        turn it means the recorded source is gone rather than that the request differs.
        """
        self._stale = False
        row = self._store.lookup(turn_key) if self._store is not None else None
        if row is None or row.idx != idx:
            return None
        source = self._source if self._source is not None and self._source.ref == row.ref else None
        if source is None:
            source = self._store.read_source(row.ref)
            if source is None:
                self._stale = True
                return None
            self._source = source
            self._source_keys = chain_over(self._base_key, self._ns, source.messages, self._system)
        if self._source_keys.get(row.idx) != turn_key:
            self._stale = True
            return None
        message = source.messages[row.idx]
        if (not isinstance(message, dict) or message.get("role") != "assistant"
                or not reply_replayable(message, self._validate_calls)):
            self._stale = True
            return None
        ordinal = sum(1 for item in source.messages[: row.idx] if isinstance(item, dict) and item.get("role") == "assistant")
        think = source.thinks[ordinal] if ordinal < len(source.thinks) else ""
        return Hit(row, source, message, think)

    def _classify(self, messages: Sequence[Mapping[str, Any]]) -> str:
        """Name the cause of the first miss (see the module docstring)."""
        if self._turn == 1:
            # The first request found nothing: either no transcript was ever recorded under
            # this base key (or the one recorded was overwritten), so there is nothing to
            # replay against, or a source exists and this run's first request differs from it.
            if self._stale or not self._source_recorded():
                return CAUSE_NO_SOURCE
            return CAUSE_INITIAL
        delta = self._fired_delta()
        if self._changed is None:
            self._changed = self._changed_entries()
        blamed = sorted(delta & self._changed)
        if blamed:
            return CAUSE_PROMPT + blamed[0]
        source_messages = self._source.messages if self._source is not None else []
        if (len(messages) < len(source_messages)
                and canon(list(messages)) == canon(source_messages[:len(messages)])
                and not reply_replayable(source_messages[len(messages)], self._validate_calls)):
            # The request still matches, but its recorded reply cannot execute.
            return CAUSE_NO_SOURCE
        if (self._source is not None and self._source.interrupted and source_messages
                and len(messages) >= len(source_messages)
                and canon(list(messages[:len(source_messages)])) == canon(source_messages)):
            # A retry reached the end of a retained partial attempt. Its missing
            # continuation is not an observed tool, clock or request difference.
            return CAUSE_NO_SOURCE
        first_new: Mapping[str, Any] | None = None
        first_old: Mapping[str, Any] | None = None
        start = self._records[-1].idx if self._records else 0
        for position in range(start, max(len(messages), len(source_messages))):
            new = messages[position] if position < len(messages) else None
            old = source_messages[position] if position < len(source_messages) else None
            if new != old:
                first_new, first_old = new, old
                break
        if first_new is not None and _is_tool_message(first_new):
            return CAUSE_TOOL
        if first_new is None and first_old is not None and _is_tool_message(first_old):
            return CAUSE_TOOL
        marker = _nudge_marker(self._release) if self._release is not None else ""
        from .prompts import entries as E

        if TIME_NUDGE_SURFACE in E.surfaces_of(delta):
            return CAUSE_CLOCK
        if marker and first_old is not None and marker in _message_text(first_old):
            return CAUSE_CLOCK
        if marker and first_new is not None and marker in _message_text(first_new):
            return CAUSE_CLOCK
        return CAUSE_UNKNOWN

    def _source_recorded(self) -> bool:
        """Whether a readable transcript was recorded under this base key (its chain-start row exists)."""
        if self._store is None:
            return False
        row = self._store.lookup(self._start)
        if row is None or row.idx != START_IDX:
            return False
        return self._store.read_source(row.ref) is not None

    def _changed_entries(self) -> frozenset[str]:
        if self._release is None or self._source is None:
            return frozenset()
        from .prompts import entries as E

        try:
            providers = (self._release.provider(surface) for surface in E.closure(self._role, self._route))
            names = {provider[0] for provider in providers if provider is not None}
            if self._slot is not None:
                # a slot document stands in for the candidate document, in either release of the comparison
                names.add(E.slot_entry(self._slot))
            closure = tuple(sorted(names))
        except KeyError:
            closure = tuple(self._release.entries)
        source_release = self._source.release if self._source is not None else None
        return changed_entries(self._release, source_release, closure)


# ---------------------------------------------------------------------------------------
# Command line


def _default_store(path: str) -> TurnStore:
    def connect() -> sqlite3.Connection:
        return connect_cache(path)

    def report(op: str, exc: BaseException) -> None:
        print(f"turn_memo: {op} failed ({type(exc).__name__}: {exc}) at {path}", file=sys.stderr)

    return TurnStore(connect, report=report)


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``ls`` or ``gc`` against the cache database.

    Args:
        argv (Sequence[str] | None): Command line without the program name; ``sys.argv[1:]`` when ``None``.

    Returns:
        int: Process exit status.
    """
    parser = argparse.ArgumentParser(prog="python -m rubiksql_hb.turn_memo", description=__doc__.split("\n", 1)[0])
    parser.add_argument("--db", default=None, help="cache database (default: $RUBIKSQL_CACHE_DB or $RUBIKSQL_DATA_DIR/llm_cache.db)")
    verbs = parser.add_subparsers(dest="verb", required=True)
    verbs.add_parser("ls", help="count index rows and episodes per universe")
    gc = verbs.add_parser("gc", help="drop the episodes and index rows of universes not listed")
    gc.add_argument("--keep-ns", action="append", default=[], metavar="NS", help="universe to keep (repeatable)")
    gc.add_argument("--drop-default", action="store_true", help="also drop the default universe (no RUBIKSQL_CACHE_NS)")
    gc.add_argument("--dry-run", action="store_true", help="count without deleting")
    args = parser.parse_args(argv)
    path = os.path.abspath(os.path.expanduser(args.db)) if args.db else cache_db_path()
    store = _default_store(path)
    if args.verb == "ls":
        json.dump({"db": path, "universes": store.counts()}, sys.stdout, indent=1, sort_keys=True)
        print()
        return 0
    dropped = store.gc(args.keep_ns, drop_default=args.drop_default, dry_run=args.dry_run)
    kept = sorted(set(args.keep_ns) | (set() if args.drop_default else {""}))
    json.dump({"db": path, "dry_run": bool(args.dry_run), "kept": kept, "dropped": dropped}, sys.stdout, indent=1, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
