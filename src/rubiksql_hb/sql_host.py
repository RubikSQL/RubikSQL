"""SQL execution tools.

Three execution paths:
  • exec_session  — run_sql: execute against the base DB plus THIS run's per-session
                    intermediate tables (read path used during exploration).
  • save_temp     — save_sql_as_table: materialise an intermediate into the run's own
                    session file (real table, session-private).
  • submit_sql    — submit_final_sql: execute the final answer against the base DB ONLY
                    (no session file), so the committed SQL must be self-contained over the
                    base tables — a SQL still referencing an intermediate fails here.

Each run gets its own session file (sess_<uuid>.sqlite), so concurrent workers/candidates
sharing the same MCP process never collide, and intermediates never touch the base DB file.
"""
import sys
from collections.abc import Mapping
import json
from contextlib import contextmanager
from typing import Any
import os
import shutil
import sqlite3
import tempfile
import time
from urllib.parse import quote
import sqlalchemy as sa
from heavenbase.database.readonly import is_sql_readonly


# Per-statement execution budget in seconds of ACTUAL CPU time (not wall-clock). A correct
# single-pass query over the largest BIRD tables (~400k rows) burns at most a few seconds of
# CPU; a pathological correlated subquery that re-aggregates for every outer row burns CPU for
# minutes. 60s of CPU separates the two with wide margin. Measuring CPU (progress handler +
# time.thread_time, see _install_progress_guard) instead of wall-clock means a query that is
# merely blocked on a lock, GIL-starved, or swapped out is NEVER falsely killed — only a real
# CPU-bound runaway is. Tunable via env; <=0 disables.
_SQL_TIMEOUT_SEC = float(os.environ.get("RUBIKSQL_SQL_TIMEOUT_SEC", "1200"))  # 20 min CPU-time

# Budget for a statement offered as the FINAL ANSWER — the shared floor of exec_sql, which is
# the one executor behind BOTH run_sql_base and submit_final_sql. Exploration (run_sql, via
# exec_session) keeps the 20-minute budget above: a slow scratch query is legitimate, a slow
# ANSWER is not.
#
# 30 is not a round number picked for comfort — it is the official BIRD execution limit, so the
# guard says exactly what the benchmark says. Under the old 1200s budget a pathologically slow
# but logically correct query ran to completion here, looked accepted, and scored 0 officially:
# a failure class the harness was structurally blind to.
#
# Measured across 1751 submitted statements from two stored runs against raw source databases:
#   • 2 exceeded 30s
#   • the slowest within-limit submit took 8.74s wall / 7.94s CPU
#     (worst of two single-threaded trials; it measures 5.2s under a 2-way pool)
#   • the 8.74s-30s band is EMPTY — 1749 of 1751 finish under 5.3s, 1731 of them under 1s
# So the margin is that empty gap, not a fudge factor: ~3.8x of CPU headroom above the slowest
# honest query in either run. Moving the threshold DOWN buys nothing (nothing sits between)
# and would start cancelling queries the benchmark accepts; moving it UP admits queries the
# benchmark rejects. 30 is simultaneously the rule and the empirically safe point.
#
# Measured as CPU time, not wall clock (see _install_progress_guard). The two differ only by
# time the query was NOT running — I/O waits, lock waits, descheduling under load — so
# cpu <= wall ALWAYS. That inequality is the whole argument: a cancellation at 30s CPU proves
# the statement also burned >=30s of wall clock here, so every rejection is a query that
# genuinely exceeded the official limit and there are NO false positives from machine load.
# It matters more here than it did at 1200s, because the verdict now lands while the model is
# still authoring, where a false positive costs a rewrite of a query that would have passed —
# and the campaign runs several arms x 16 workers on one box, exactly the contention that made
# the old wall-clock guard kill fast queries. The failure class itself is pure CPU and loses
# nothing to the choice: the two CPU-heavy overruns measured cpu/wall = 0.98 and 0.96 single-threaded.
#
# The residual risk is the other direction, and it is real but unobserved: an I/O-bound query
# can sit under a CPU budget while exceeding the official WALL limit. One measured statement
# demonstrates that shape (4.97s wall, 0.49s CPU — cpu/wall = 0.10), so the
# divergence is not hypothetical. It costs nothing at this scale: for such a query to slip
# through it would need >30s wall on under 30s CPU, and the slowest non-offender measured
# anywhere in the 1751 is 8.74s wall. Revisit if a future run shows a submitted query above
# ~20s wall with a low cpu/wall ratio.
# Tunable via env; <=0 disables. Operational, like its 1200s sibling — NOT a behaviour arm
# (neither budget is namespaced into nl2sql._cache_key), but it IS recorded in the pipeline's
# run identity so a run that moves it cannot silently resume onto another run's answers.
_FINAL_SQL_TIMEOUT_SEC = float(os.environ.get("RUBIKSQL_FINAL_SQL_TIMEOUT_SEC", "30"))

# Stable substring of _TIMEOUT_MSG, so callers can recognise OUR cancellation without
# re-matching the prose (which is model-visible and edited more often than this contract).
# is_timeout_error() is the only supported test; nothing should grep the message itself.
_TIMEOUT_MARK = "execution limit and was cancelled"

# Returned (as a normal {"error": ...}) when a query is cancelled by the timeout. The text is
# the ONLY guidance the agent gets about slow SQL — it only appears for candidates that
# actually hit the limit, so we keep the always-on system prompt lean.
_TIMEOUT_MSG = (
    "query exceeded the {t}s execution limit and was cancelled. This is almost always a "
    "correlated subquery that re-computes an aggregate for every outer row. Rewrite it as a "
    "single-pass aggregate: compute the aggregate ONCE in a CTE and join to it (a CTE stays "
    "fast after it is inlined into the final SQL). For exploration you can also materialise "
    "the reusable part with save_sql_as_table — but note a correlated subquery does NOT get "
    "faster on submit, so prefer the single-pass rewrite."
)

# Appended ONLY on the final-answer path (exec_sql). The message above is shared with the
# 20-minute exploration budget, where "this is the limit you are scored under" would simply be
# false — so the scoring framing lives here, where it is true, instead of in the shared text.
# It exists to stop the one wrong reaction to a cancellation: resubmitting the same statement
# and hoping. The limit is not transient and the query will not get faster.
_FINAL_TIMEOUT_NOTE = (
    " That limit is the one your FINAL ANSWER is scored under, so this SQL cannot be "
    "submitted as it stands — it needs a rewrite, not a retry."
)


def is_timeout_error(result: object) -> bool:
    """True when ``result`` is the error dict OUR execution guard produced.

    Distinguishes a query WE cancelled for exceeding its budget from every other
    ``{"error": ...}`` — a syntax error, a missing table, a dead session. Callers that
    react to slowness specifically (the agent loop's rewrite counter) need that split, and
    matching on the rendered prose would break the first time the wording is tuned.

    Args:
        result (object): Any tool result; non-dicts and non-errors are simply False.

    Returns:
        bool: Whether the guard cancelled this statement.
    """
    return (
        isinstance(result, dict)
        and _TIMEOUT_MARK in str(result.get("error") or "")
    )


# VM opcodes between progress-handler callbacks. Large enough that the per-callback overhead
# is negligible even on a heavy legitimate scan, small enough that a query crossing the CPU
# budget is aborted within milliseconds of doing so.
_PROGRESS_OPS = 1_000_000


_READONLY_ACTIONS = frozenset({
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
})


def _readonly_uri(url: str) -> str:
    """Build a SQLite URI that the driver opens in read-only mode."""
    if not url.lower().startswith("sqlite:///"):
        raise ValueError("read-only audit tools currently support SQLite URLs only")
    path = os.path.abspath(_sqlite_path(url))
    # BIRD databases are immutable benchmark inputs. `immutable=1` keeps the connection
    # physically read-only while also allowing WAL-header databases to open before any
    # writable connection has created their transient -wal/-shm sidecars.
    return f"file:{quote(path, safe='/:')}?mode=ro&immutable=1"


def _readonly_authorizer(action: int, arg1: str | None, arg2: str | None,
                         _db: str | None, _source: str | None) -> int:
    """Allow only query actions and reject extension loading."""
    if action not in _READONLY_ACTIONS:
        return sqlite3.SQLITE_DENY
    unsafe_functions = {"load_extension", "readfile", "writefile"}
    if action == sqlite3.SQLITE_FUNCTION and (arg1 or arg2 or "").lower() in unsafe_functions:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _install_progress_guard(conn: sqlite3.Connection, timeout_sec: float | None = None,
                            *, deadline: float | None = None):
    """Abort a query after _SQL_TIMEOUT_SEC of actual CPU time — NOT wall-clock — so a query
    that is merely blocked on a lock, starved of the GIL, or swapped out is never falsely
    cancelled; only a genuinely CPU-bound runaway (e.g. a correlated subquery re-aggregating
    for every outer row) is killed.

    SQLite calls the progress handler every _PROGRESS_OPS VM opcodes — i.e. ONLY while the
    statement is actively executing — and aborts it (raising sqlite3.OperationalError) when the
    handler returns non-zero. We compare per-thread CPU time (time.thread_time, which excludes
    sleep / lock-wait / descheduled / swapped-out time) against the budget, so non-running time
    never counts toward the limit. This fixes the old wall-clock threading.Timer guard, which
    measured elapsed real time and so killed fast queries that were merely stalled under load.

    Returns `fired` (fired['v'] True iff WE aborted) so the caller can tell our abort from an
    unrelated OperationalError. No-op when the guard is disabled (<=0)."""
    fired = {"v": False, "reason": None}
    budget = _SQL_TIMEOUT_SEC if timeout_sec is None else float(timeout_sec)
    if budget <= 0 and deadline is None:
        return fired
    start = time.thread_time()

    def _cb() -> int:
        if deadline is not None and time.monotonic() >= deadline:
            fired["v"] = True
            fired["reason"] = "wall"
            return 1
        if budget > 0 and time.thread_time() - start > budget:
            fired["v"] = True
            fired["reason"] = "cpu"
            return 1   # non-zero → SQLite aborts the running statement
        return 0

    conn.set_progress_handler(_cb, _PROGRESS_OPS)
    return fired


@contextmanager
def guarded_readonly_cursor(
    url: str,
    sql: str,
    params: tuple[object, ...] | dict[str, object] = (),
    *,
    timeout_sec: float | None = None,
    deadline: float | None = None,
    max_sqlite_length: int | None = None,
    max_columns: int | None = None,
):
    """Open one guarded SQLite SELECT/WITH cursor for incremental fetching.

    Args:
        url: SQLite database URL; the database is opened read-only.
        sql: Exactly one SELECT/WITH statement.
        params: Driver-bound parameters.
        timeout_sec: Existing per-thread CPU budget; zero disables it.
        deadline: Optional monotonic wall deadline, including cursor fetches.
        max_sqlite_length: Optional SQLite length limit applied before execution.
        max_columns: Optional SQLite result-column limit applied before execution.

    Yields:
        Cursor and mutable cancellation marker. Consumers must fetch while inside
        the context and inspect ``fired['reason']`` after SQLite interruption.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("sql must be a non-empty SELECT/WITH statement")
    statement = sql.strip()
    if not is_sql_readonly(statement, dialect="sqlite"):
        raise ValueError("only one read-only SELECT/WITH statement is allowed")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("diagnostic SQL wall deadline exceeded")
    conn = sqlite3.connect(_readonly_uri(url), uri=True)
    try:
        if max_sqlite_length is not None:
            if max_sqlite_length <= 0:
                raise ValueError("max_sqlite_length must be positive")
            conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, max_sqlite_length)
        if max_columns is not None:
            if max_columns <= 0:
                raise ValueError("max_columns must be positive")
            conn.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, max_columns)
        fired = _install_progress_guard(conn, timeout_sec, deadline=deadline)
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_readonly_authorizer)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("diagnostic SQL wall deadline exceeded")
        cursor = conn.execute(statement, params)
        yield cursor, fired
    finally:
        conn.close()


# Session files live under a PER-PROCESS subdir {_SESSION_DIR}/{pid}/ so that a sibling MCP
# subprocess (another db, or another parallel-eval process — all share /tmp) never deletes
# our live intermediate-table files. sweep_sessions() only clears our own pid's subdir and
# the subdirs of already-DEAD processes.
_SESSION_DIR = os.path.join(tempfile.gettempdir(), "rubiksql_sessions")

# A session file older than this is certainly a stale leftover — no single agent run lasts
# hours. Used only as a backstop to reclaim a dir whose pid got recycled by an unrelated
# live process (so the liveness check keeps the dir, but its files are ancient).
_SESSION_TTL_SEC = 12 * 3600


def _proc_session_dir() -> str:
    """Path of THIS process's session subdir. Pure — does not create it (only the write
    path, _connect, materialises the dir, so read/cleanup helpers never leave empty dirs)."""
    return os.path.join(_SESSION_DIR, str(os.getpid()))


def _readonly_file_path(url: str) -> str | None:
    """Absolute path of the on-disk SQLite file behind ``url``, or None if it has none.

    Returns None for in-memory databases (``:memory:``, the empty ``sqlite://``), for URLs the
    caller already expressed as a URI, and for non-SQLite URLs — none of those can be reopened
    read-only by rewriting a path, and an in-memory database is private to its connection, so
    there is nothing to protect.
    """
    if not url.lower().startswith("sqlite:///"):
        return None
    path = _sqlite_path(url)
    if not path or path == ":memory:" or path.startswith("file:"):
        return None
    return os.path.abspath(path)


def _engine(url: str) -> sa.Engine:
    """Read-only SQLAlchemy engine on a benchmark database.

    Every caller runs model-authored or model-parameterised SQL, and the benchmark database is
    also the evaluator's scoring ground truth, so no engine here may be able to write to it.
    In-memory and non-SQLite URLs are passed through unchanged (see ``_readonly_file_path``).
    """
    path = _readonly_file_path(url)
    if path is None:
        return sa.create_engine(url)
    # `uri=true` must ride in the query string: passing connect_args={"uri": True} instead
    # makes the pysqlite dialect hand the driver a path it reads literally, which fails to open.
    return sa.create_engine(f"sqlite:///file:{quote(path, safe='/:')}?mode=ro&uri=true")


def _sqlite_path(url: str) -> str:
    """Filesystem path from a sqlite:/// URL."""
    return url[len("sqlite:///"):] if url.lower().startswith("sqlite:///") else url


def _session_path(session_id: str) -> str:
    return os.path.join(_proc_session_dir(), f"sess_{session_id}.sqlite")


def _quote_ident(name: str) -> str:
    """Quote a SQL identifier, doubling any embedded double-quotes so an arbitrary
    (model-chosen) table_name can't break out of the quoting."""
    return '"' + str(name).replace('"', '""') + '"'


def _connect(url: str, session_id: str = "") -> sqlite3.Connection:
    """Open the base DB READ-ONLY; ATTACH the per-session file as `sess` when session_id is given.

    Unqualified table names then resolve base tables from main and this run's intermediates
    from sess; other sessions' intermediates (separate file) are invisible.

    The base database is the benchmark's immutable input AND the evaluator's scoring ground
    truth, so it is opened `mode=ro`: model-authored DDL used to run in SQLite's autocommit
    mode and commit immediately, returning a success-shaped empty result while the table was
    already gone. Only `sess` stays writable, which is all save_temp needs (it qualifies every
    statement with `sess.`). `immutable=1` is deliberately NOT used here — unlike the audit
    path it would make SQLite ignore a shipped -wal and answer from a stale snapshot; it is
    only reached as a fallback when the plain read-only open fails, which happens for a
    WAL-header database whose directory is not writable (no -shm sidecar can be created).

    NB: ATTACH to a not-yet-existing path CREATES it, so the first run_sql of a run
    materialises an (empty) session file even if it never saves an intermediate. That's
    harmless — close_session() drops it at end-of-run, and sweep_sessions() reclaims any
    leftover at startup."""
    path = _readonly_file_path(url)
    if path is None:                                  # in-memory / URI / non-SQLite: nothing to protect
        conn = sqlite3.connect(_sqlite_path(url))
    else:
        try:
            conn = sqlite3.connect(f"file:{quote(path, safe='/:')}?mode=ro", uri=True)
        except sqlite3.OperationalError as e:
            conn = sqlite3.connect(_readonly_uri(url), uri=True)
            print(f"[sql] read-only open of {path} needed the immutable fallback ({e}); a shipped "
                  f"-wal file (if any) is NOT being read — verify the database is fully "
                  f"checkpointed.", file=sys.stderr, flush=True)
    if session_id:
        os.makedirs(_proc_session_dir(), exist_ok=True)  # ATTACH creates the file; its dir must exist
        conn.execute("ATTACH DATABASE ? AS sess", (_session_path(session_id),))
    return conn


def _dedupe_cols(cols: list) -> list:
    """Uniquify duplicate result-column names positionally (name, name_2, ...).

    ``dict(zip(cols, row))`` silently DROPS columns when a SELECT outputs the same
    name twice (e.g. ``T2.name, T1.name`` — common in BIRD golds): the later key
    overwrites the earlier one, so scoring compared a 2-column reference against a
    3-column prediction and the model's own run_sql preview lost a
    column too. Positional suffixing preserves every column in SELECT order.
    """
    seen: dict = {}
    out = []
    for c in cols:
        n = seen.get(c, 0)
        seen[c] = n + 1
        out.append(c if n == 0 else f"{c}_{n + 1}")
    return out


def exec_sql(url: str, sql: str) -> dict[str, Any]:
    """Execute a SQL query against the base DB only and return rows as list of dicts.

    The single executor behind BOTH ``run_sql_base`` and ``submit_final_sql``, which is why
    the final-answer budget lives here: the model's own verification and the harness's verdict
    on submission run the same statement through the same code with the same threshold, so
    they agree BY CONSTRUCTION rather than by two constants being kept in sync. A model that
    watches its SQL pass run_sql_base has therefore seen exactly what submission will do —
    which is what that tool already promises ("the SAME conditions as submit_final_sql").
    """
    result = exec_session(url, sql, "", timeout_sec=_FINAL_SQL_TIMEOUT_SEC)
    if is_timeout_error(result):
        return {"error": str(result["error"]) + _FINAL_TIMEOUT_NOTE}
    return result


def query_readonly(
    url: str,
    sql: str,
    params: tuple[object, ...] | dict[str, object] = (),
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    """Execute one guarded read query against a SQLite database.

    This path is intended for higher-level diagnostic tools that wrap model-provided
    source SQL. It accepts one `SELECT`/`WITH` statement, opens the database with
    `mode=ro`, enables SQLite's query-only mode, and installs an authorizer that
    rejects writes, pragmas, attachment, transactions, and extension loading.

    Args:
        url (str): SQLite database URL.
        sql (str): Single read-only `SELECT`/`WITH` statement.
        params (tuple[object, ...] | dict[str, object]): Bound SQL parameters.

    Returns:
        dict[str, Any]: `columns`, `rows`, and `row_count` on success, or an
        `error` field when validation or execution fails.
    """
    budget = _SQL_TIMEOUT_SEC if timeout_sec is None else float(timeout_sec)
    fired = {"v": False}
    try:
        with guarded_readonly_cursor(url, sql, params, timeout_sec=budget) as (cur, fired):
            cols = [desc[0] for desc in cur.description] if cur.description else []
            cols = _dedupe_cols(cols)
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
            return {"columns": cols, "rows": rows, "row_count": len(rows)}
    except sqlite3.OperationalError as exc:
        # Cancellation inside conn.execute happens before the context manager yields
        # its marker. This readonly connection exposes no other interrupt source.
        if fired["v"] or getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT:
            return {"error": _TIMEOUT_MSG.format(t=int(budget))}
        return {"error": str(exc)}
    except (sqlite3.Error, ValueError, OSError) as exc:
        return {"error": str(exc)}


def exec_session(url: str, sql: str, session_id: str = "",
                 timeout_sec: float | None = None) -> dict[str, Any]:
    """run_sql path: execute against base + this session's intermediates.

    Also the executor for run_sql_base / submit_final_sql (via exec_sql), so the timeout guard
    here covers every read/submit path. ``timeout_sec`` selects WHICH budget: exec_sql passes
    the 30s final-answer budget, while run_sql leaves it None and keeps the 20-minute
    exploration budget — a slow scratch query is legitimate, a slow answer is not."""
    conn = _connect(url, session_id)
    budget = _SQL_TIMEOUT_SEC if timeout_sec is None else float(timeout_sec)
    fired = _install_progress_guard(conn, budget)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        cols = _dedupe_cols(cols)
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        return {"columns": cols, "rows": rows, "row_count": len(rows)}
    except sqlite3.OperationalError as e:
        if fired["v"]:
            return {"error": _TIMEOUT_MSG.format(t=int(budget))}
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


def save_temp(url: str, sql: str, table_name: str, session_id: str) -> dict[str, Any]:
    """save_sql_as_table path: materialise an intermediate into THIS run's session file."""
    if not session_id:
        return {"error": "save_sql_as_table is only available inside an agent run (no session)."}
    conn = _connect(url, session_id)
    tbl = _quote_ident(table_name)
    fired = _install_progress_guard(conn)
    try:
        conn.execute(f'DROP TABLE IF EXISTS sess.{tbl}')
        conn.execute(f'CREATE TABLE sess.{tbl} AS {sql}')
        n = conn.execute(f'SELECT COUNT(*) FROM sess.{tbl}').fetchone()[0]
        conn.commit()
        note = (f"Intermediate '{table_name}' saved for this run — reference it by name in "
                f"run_sql. submit_final_sql runs WITHOUT these intermediates, so before "
                f"submitting rewrite your answer as ONE self-contained SQL over the base "
                f"tables (inline intermediates as CTEs). Don't name an intermediate the "
                f"same as a base table.")
        return {
            "table": table_name,
            "row_count": n,
            "note": note,
        }
    except sqlite3.OperationalError as e:
        if fired["v"]:
            # `budget`, not the module default: the caller may be running under the 30s
            # final-answer budget, and reporting 1200 there tells the model its query ran for
            # twenty minutes when it was cancelled at thirty seconds — a number that makes the
            # advice underneath it look absurd and invites the wrong rewrite.
            return {"error": _TIMEOUT_MSG.format(t=int(budget))}
        return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


def list_session_tables(session_id: str) -> list[str]:
    """Names of the intermediate tables saved in this run's session file."""
    if not session_id or not os.path.exists(_session_path(session_id)):
        return []
    conn = sqlite3.connect(_session_path(session_id))
    try:
        return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    except Exception:
        return []
    finally:
        conn.close()


def close_session(session_id: str) -> None:
    """Delete this run's session file at end-of-run (in our own per-process subdir).
    Leftovers from a crashed run are cleared by sweep_sessions() when a process next finds
    our pid dead."""
    if not session_id:
        return
    base = _session_path(session_id)
    for p in (base, base + "-wal", base + "-shm", base + "-journal"):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


def _pid_alive(pid: int) -> bool:
    """True if a process with this pid currently exists.

    Windows MUST NOT probe via ``os.kill(pid, 0)``: there the call maps to
    TerminateProcess semantics (it can KILL a live sibling MCP server) and raises
    WinError 87 for dead pids, which leaked as a C-level pending exception and
    crashed server startup (SystemError at the next builtin call). Use an
    information-only OpenProcess handle instead.
    """
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, int(pid))
        if not h:
            # access denied (5) → exists but not ours; invalid parameter (87) → dead
            return ctypes.get_last_error() == 5 or kernel32.GetLastError() == 5
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(h, ctypes.byref(code)):
                return True   # be conservative — don't delete on uncertainty
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists but owned by someone else
    except Exception:
        return True   # be conservative — don't delete on uncertainty
    return True


def sweep_sessions() -> int:
    """Startup hygiene. Remove ONLY this process's own session subdir (a prior incarnation
    that reused our pid is dead) and the subdirs of already-DEAD processes — NEVER a live
    sibling MCP subprocess's subdir (that would delete intermediates a concurrent run is
    still using). Also drops any legacy flat session files from the pre-isolation layout."""
    if not os.path.isdir(_SESSION_DIR):
        return 0
    me = os.getpid()
    removed = 0
    for name in os.listdir(_SESSION_DIR):
        path = os.path.join(_SESSION_DIR, name)
        if not os.path.isdir(path):
            try:                       # legacy flat sess_*.sqlite from before per-pid subdirs
                os.remove(path); removed += 1
            except Exception:
                pass
            continue
        try:
            pid = int(name)
        except ValueError:
            continue                   # unknown subdir — leave it alone
        if pid == me or not _pid_alive(pid):
            try:                       # count the session files we're about to drop
                removed += sum(1 for e in os.scandir(path) if e.is_file())
            except Exception:
                pass
            shutil.rmtree(path, ignore_errors=True)
        else:
            # An ALIVE pid we don't own — normally a live sibling MCP server, so leave its
            # in-flight files alone. But the pid may have been recycled by an unrelated
            # process (the original server died); drop only files too old to belong to any
            # in-flight run, so a recycled pid can't pin leftovers forever.
            removed += _drop_stale_files(path)
    return removed


def _drop_stale_files(dir_path: str) -> int:
    """Remove session files older than _SESSION_TTL_SEC, then the dir if it's left empty."""
    cutoff = time.time() - _SESSION_TTL_SEC
    removed = 0
    try:
        entries = list(os.scandir(dir_path))
    except Exception:
        return 0
    for e in entries:
        try:
            if e.is_file() and e.stat().st_mtime < cutoff:
                os.remove(e.path)
                removed += 1
        except Exception:
            pass
    try:
        if not os.listdir(dir_path):
            os.rmdir(dir_path)
    except Exception:
        pass
    return removed


def submit_sql(url: str, sql: str) -> dict[str, Any]:
    """
    submit_final_sql path: validate syntax then execute the final answer against the base
    DB ONLY (no session file). The committed SQL must therefore be self-contained over the
    base tables — one still referencing a run's intermediate fails here ('no such table').

    Runs EXPLAIN before executing so syntax errors are caught early. Returns the same
    structure as exec_sql plus 'final: true' to signal this is the committed answer.
    """
    engine = _engine(url)
    try:
        with engine.connect() as conn:
            conn.execute(sa.text(f"EXPLAIN {sql}"))
    except Exception:
        # EXPLAIN can reject valid SQL (e.g. SQLite rejects double-quoted
        # identifiers in EXPLAIN even though execution succeeds). Fall through
        # and let exec_sql surface any genuine errors.
        pass
    finally:
        engine.dispose()

    result = exec_sql(url, sql)
    if "error" not in result:
        result["final"] = True
    return result


def reexecute_final(url: str, sql: str, tool_result: Mapping[str, Any]) -> dict[str, Any]:
    """Reexecute final SQL on host-owned base tables and retain tool evidence.

    Args:
        url (str): Source database URL, guarded read-only by this module.
        sql (str): Self-contained final SQL that the agent submitted.
        tool_result (Mapping[str, Any]): Tool-returned rows/error; these never
            determine gold execution or the host's EX score.

    Returns:
        dict[str, Any]: tool_result, host_result, result_mismatch and a reason.
            The host result uses exactly the final-answer timeout.
    """
    host = submit_sql(url, sql)
    tool = {key: tool_result[key] for key in ("columns", "rows", "row_count", "error") if key in tool_result}
    observed = {key: host[key] for key in ("columns", "rows", "row_count", "error") if key in host}
    keys = set(tool) | set(observed)
    differing = [key for key in sorted(keys) if json.dumps(tool.get(key), sort_keys=True, ensure_ascii=False, default=str)
                 != json.dumps(observed.get(key), sort_keys=True, ensure_ascii=False, default=str)]
    return {"tool_result": tool, "host_result": host, "result_mismatch": bool(differing),
            "mismatch_reason": ", ".join(differing) + " differ between tool evidence and host reexecution" if differing else None}
