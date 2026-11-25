"""RubikSQL-HB MCP server — installable package entry point."""
import os
os.environ.setdefault("LOG_LEVEL", "WARNING")

try:
    from dotenv import load_dotenv
    load_dotenv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"),
        override=False,
    )
except ImportError:
    pass

import json
import importlib
import inspect
import sys
import tempfile
import threading
import time
import traceback
from functools import lru_cache, partial, wraps
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any
from weakref import WeakValueDictionary

import anyio
from anyio.lowlevel import RunVar, current_token

from rubiksql_hb.prompts.binding import load_binding, select_binding_from_argv
from rubiksql_hb.tool_bundle import bootstrap_head_bundle

if __name__ == "__main__":
    select_binding_from_argv(sys.argv[1:])
_BINDING = load_binding()
_EXECUTION_SNAPSHOT = bootstrap_head_bundle(binding=_BINDING)

import heavenbase as hb

from rubiksql_hb import rate_limits, tool_timing
from rubiksql_hb.example_mask import QUESTION_META, READERS as _EXAMPLE_READERS, mask_tool_result, question_scope
from rubiksql_hb.knowledge_session import knowledge_sessions
from rubiksql_hb.prompts import current as _prompt_release
from rubiksql_hb.query_policy import code_reduce_disabled, refuse_code_reduce
from rubiksql_hb.schema import normalize_rule_scope
from rubiksql_hb.workspace import check_retrieval_version, open_workspace
from rubiksql_hb.register import register_db as _register_db
from rubiksql_hb.sql_host import (
    exec_sql, exec_session, save_temp, list_session_tables,
    close_session as _close_session, sweep_sessions, submit_sql as _submit_sql,
)
from rubiksql_hb.knowledge_host import (
    add_example, add_synonym,
    save_rule as _save_rule, update_rule as _update_rule,
    query_knowledge as _query_knowledge, QUERY_LIMIT_MAX as _QUERY_LIMIT_MAX,
    QUERY_ENTITIES as _QUERY_ENTITIES, query_frame as _query_frame,
)
from rubiksql_hb.tool_declarations import project_tool_functions

_TOOL_FUNCTIONS = {function.__name__: function for function in project_tool_functions()}

# Historical six-module snapshots import these helpers from their host server.
# Resolve the export from the pinned callable's own module so those releases
# keep their original implementation without importing any current aggregate.
_PINNED_HELPER_EXPORTS = {
    "db_info": ("get_db_info", "db_info"), "tab_info": ("get_tab_info", "tab_info"),
    "col_info": ("get_col_info", "col_info"), "fuzzy_enum": ("search_columns", "fuzzy_enum"),
    "fuzzy_enum_values": ("search_values", "fuzzy_enum_values"),
    "_fd_check": ("check_functional_dependency", "fd_check"),
    "read_examples": ("get_examples", "read_examples"), "resolve_synonyms": ("expand_synonyms", "resolve_synonyms"),
    "_get_rules": ("get_rules", "read_rules"), "_match_rule_triggers": ("match_rule_triggers", "read_rule_triggers"),
    "_SPEC_HIDDEN_IDS": ("query_spec", "_SPEC_HIDDEN_IDS"),
    "_audit_measure_grain": ("audit_measure_grain", "measure_grain"),
    "_grain_alias_list": ("audit_measure_grain", "_grain_alias_list"),
    "_split_outside_quotes": ("audit_measure_grain", "_split_outside_quotes"),
    "render_sql_template": ("render_template", "render_sql_template"),
}

# ── Data directory ────────────────────────────────────────────────────────────
# Set RUBIKSQL_DATA_DIR to control where db, chroma, and registry are stored.
# Default: ~/.rubiksql_hb/
# Experiment flags — read before ANY tool definition below uses them.
# Two-stage example retrieval (issue #8): promoted to DEFAULT ON — standard hidden-state
# 498q A/B measured +2.21pp (0.7269→0.7490) with the same +11q at the oracle ceiling
# (candidate-pool gain, not judge luck), 10/11 dbs non-negative (PLAN §5.14).
# §11 row 16b arm: search_columns hits carry profile_desc (k-bounded). Default OFF;
# the flag is part of the candidate cache key (nl2sql._cache_key env namespace).
_SEARCH_COLUMNS_PROFILE = os.environ.get("RUBIKSQL_SEARCH_COLUMNS_PROFILE", "") == "1"
# Structured knowledge-base read arm. Every knowledge channel the models have today is
# top-k-by-vector-similarity (get_examples, get_rules, search_values top-10
# above 0.45): nothing is enumerable, nothing is filterable, and the synonym table is not
# visible at all. This flag serves query_knowledge alongside them. Default OFF; it is part of
# BOTH cache keys (nl2sql._cache_key env namespace + the vote judge key), and the same env
# var gates the tool in the candidate/judge allowlists — _mcp_params passes {**os.environ}
# to this subprocess, so the two sides cannot disagree within one run.

# Retrieval widths are knobs of the pinned prompt release (config.knobs, prompts/entries.py):
# the k get_rules and the top_k get_examples use when the caller passes none. The pin reaches
# this subprocess through the environment _mcp_params hands it, so a run's candidates, judge
# and tools read one release; a value other than the default moves the keyspaces of every role
# that calls these tools, since the tool schema it serves carries the default.
_RULES_TOP_K = _prompt_release().knob("rules_top_k")
_EXAMPLES_TOP_K = _prompt_release().knob("examples_top_k")
# Bind the served callables to the pinned release; declaration reflection normalizes
# these two defaults to the committed KNOBS entries for implementation identity.
for _name, _value in (("get_examples", _EXAMPLES_TOP_K), ("get_rules", _RULES_TOP_K)):
    _function = _TOOL_FUNCTIONS[_name]
    _function.__defaults__ = (_value, *_function.__defaults__[1:])

for _fn in (_TOOL_FUNCTIONS["query_knowledge"], _TOOL_FUNCTIONS["query_spec"]):
    for _stated in (f"(up to {_QUERY_LIMIT_MAX})", f"hard cap {_QUERY_LIMIT_MAX}"):
        if _stated not in (_fn.__doc__ or ""):
            raise RuntimeError(f"{_fn.__name__} docstring must state {_stated!r} to match "
                               f"tools.knowledge.QUERY_LIMIT_MAX")

_DATA_DIR = os.environ.get("RUBIKSQL_DATA_DIR", os.path.expanduser("~/.rubiksql_hb"))
_WORKSPACE_ID = os.environ.get("RUBIKSQL_WORKSPACE_ID", "rubiksql_server")

# HeavenBase applies sqlite pragmas natively (per-engine, from backend config)
# since 0.1.1.x — the old global SQLAlchemy connect-event hack is retired. The pragmas
# list REPLACES the provider defaults, so restate them all; busy_timeout is
# raised from the 6 s default to the 120 s value tuned for concurrent eval.
_SQLITE_PRAGMAS = [
    "PRAGMA foreign_keys = ON;",
    "PRAGMA journal_mode = WAL;",
    "PRAGMA synchronous = NORMAL;",
    "PRAGMA busy_timeout = 120000;",
]

_POOL = {"connect_args": {"timeout": 60}}

#: The process-wide workspace and the toolkit built from it, created on first use.
#: Building them cost 16.5 s of the 17 s that ``import rubiksql_hb.server`` used to take (the
#: import chain itself is 1.3 s): ``HeavenBase(...)`` is 8.8 s and each of the six
#: ``register()`` calls ~1.1 s — flat per entity, not per vector field. Paying that at *import*
#: time meant anything that merely wanted a name out of this module built a whole SQLite +
#: Chroma workspace as a side effect; two subprocess tests only needed ``_db_url`` and were
#: timing out at 120 s under load. Nothing here needs the workspace to exist at import time —
#: the tool functions are plain functions and reach it through :func:`_workspace`.
_WS: hb.HeavenBase | None = None
_TOOLKIT = None
#: Deferring construction reintroduces a race the module-level version could not have: nl2sql
#: runs tool calls from threads (``threading.Thread`` for the agent loop, ``concurrent.futures``
#: for the candidates), so two first-callers could each see ``None`` and build a second
#: workspace over the same SQLite + Chroma directory. One lock covers both getters because the
#: toolkit is built from the workspace.
_BOOT_LOCK = threading.RLock()   # 可重入:_mcp_toolkit 持锁时会调 _workspace


def _retrieval_open_kwargs() -> dict[str, Any]:
    """How every workspace this server opens asks HeavenBase to rank.

    Exact is not optional here: BIRD needs the same rows from the same workspace in every
    process, and only HB's exhaustive scan gives that (see ``retrieval.search_vectors``).
    ``RUBIKSQL_RETRIEVAL_VERSION`` is the opt-in warm path — an immutable knowledge-generation
    token that lets HB reuse decoded candidates across queries. Setting it ASSERTS that no
    other process writes this workspace's knowledge for the lifetime of this server: HB
    replays the cached generation without re-checking the backend, and this server lives as
    long as the eval that launched it, so an external ``inject_*`` or rulegen write made
    meanwhile is served stale — while the eval's own knowledge fingerprint, read fresh from
    SQLite, would key those stale answers into the persistent result cache under the new
    knowledge. This process's own saves invalidate the cache. Surrounding whitespace (a
    shell artefact) is stripped; a token with inner whitespace or a control character is
    refused at startup, because it also travels into cache keys.

    Returns:
        dict[str, Any]: Keyword arguments for :func:`rubiksql_hb.workspace.open_workspace`.

    Raises:
        ValueError: ``RUBIKSQL_RETRIEVAL_VERSION`` is set to something other than one
            printable word (see :func:`rubiksql_hb.workspace.check_retrieval_version`).
    """
    version = os.environ.get("RUBIKSQL_RETRIEVAL_VERSION", "").strip()
    if not version:
        return {"retrieval_exact": True}
    try:
        check_retrieval_version(version)
    except ValueError as exc:
        raise ValueError(f"RUBIKSQL_RETRIEVAL_VERSION: {exc}") from None
    return {"retrieval_exact": True, "retrieval_version": version}


def _workspace() -> hb.HeavenBase:
    """The workspace this module serves from, opened on first use.

    Returns:
        heavenbase.HeavenBase: The process-wide workspace, built at most once per process.
        ``toolkit=True`` only here — this is the one whose ``to_mcp()`` builds the served toolkit.
    """
    global _WS
    if _WS is None:                                   # 无锁快路径:建好之后每次调用都走这里
        with _BOOT_LOCK:
            if _WS is None:                           # 拿到锁后重查 —— 可能已被别的线程建好
                os.makedirs(_DATA_DIR, exist_ok=True)
                _WS = open_workspace(_DATA_DIR, pragmas=_SQLITE_PRAGMAS, pool=_POOL,
                                     workspace_id_value=_WORKSPACE_ID, toolkit=True,
                                     **_retrieval_open_kwargs())
    return _WS

# ── Per-db workspace routing ──────────────────────────────────────────────────
# Per-db workspaces live at bird_dbs/{db_id}/ (project-local) or
# ~/.rubiksql_hb/bird_dbs/{db_id}/. _ws_for() opens them lazily and caches
# them so interactive MCP calls (e.g. from Claude Desktop) read the right
# workspace even when RUBIKSQL_DATA_DIR is not set.

_BIRD_DBS_DIR = os.environ.get("RUBIKSQL_BIRD_DBS_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "bird_dbs",
)
_WS_CACHE: dict[str, hb.HeavenBase] = {}
#: One lock per db_id, so two threads asking for the SAME database open it once while two asking
#: for DIFFERENT databases still open in parallel — the dashboard prewarms all 11 concurrently,
#: and a single lock would serialise that into 11 × ~16 s.
#:
#: Without this, ``_ws_for`` was a plain check-then-set: measured 6 threads asking for one db_id
#: got 6 separate workspaces over the same directory. The wasted construction is the small half.
#: The real cost is that each ``HeavenBase`` brings its own Chroma client, and the HNSW graph is
#: never persisted at these collection sizes — so every copy rebuilds a different in-memory graph
#: and the same question retrieves different examples depending on which copy served it. That is
#: precisely the non-determinism the exact HB scan (``retrieval.search_vectors``) exists to remove.
_WS_LOCKS: dict[str, threading.RLock] = {}
_WS_LOCKS_GUARD = threading.Lock()


def _ws_lock(db_id: str) -> threading.RLock:
    """The lock for one database's workspace, created on first ask.

    Args:
        db_id (str): Database identifier.

    Returns:
        threading.RLock: The same lock object for the same ``db_id``, always.
    """
    with _WS_LOCKS_GUARD:
        return _WS_LOCKS.setdefault(db_id, threading.RLock())


def _paths_equal(a: str, b: str) -> bool:
    """Case-insensitive path comparison on Windows, case-sensitive elsewhere."""
    a, b = os.path.normpath(a), os.path.normpath(b)
    if os.name == "nt":
        return a.lower() == b.lower()
    return a == b


def _ws_for(db_id: str) -> hb.HeavenBase:
    """The workspace serving one database, opened once per process.

    Args:
        db_id (str): Database identifier.

    Returns:
        heavenbase.HeavenBase: Its workspace — the global one when this process is already
        running inside that database's directory, otherwise the per-db workspace under
        ``bird_dbs/<db_id>/``, or the global one as a last resort.
    """
    hit = _WS_CACHE.get(db_id)
    if hit is not None:                       # 无锁快路径:开过之后每次调用都走这里
        return hit
    with _ws_lock(db_id):
        hit = _WS_CACHE.get(db_id)
        if hit is not None:                   # 拿到锁后重查 —— 可能已被别的线程开好
            return hit
        return _open_ws_for(db_id)


def _open_ws_for(db_id: str) -> hb.HeavenBase:
    """Resolve and open one database's workspace. Caller holds :func:`_ws_lock` for ``db_id``."""
    # Fast path: if RUBIKSQL_DATA_DIR was already set to this db's directory
    # (done by _mcp_params in nl2sql.py), the global ws is already correct.
    # Use basename check — avoids fragile cross-platform path comparison.
    if os.path.basename(os.path.normpath(_DATA_DIR)) == db_id:
        _WS_CACHE[db_id] = _workspace()
        return _WS_CACHE[db_id]
    search_bases = (
        _BIRD_DBS_DIR,
        os.path.join(os.path.expanduser("~/.rubiksql_hb"), "bird_dbs"),
    )
    for base in search_bases:
        per_db = os.path.join(base, db_id)
        if not os.path.isdir(per_db):
            continue
        # Already running inside this per-db workspace — reuse the global one.
        if _paths_equal(per_db, _DATA_DIR):
            _WS_CACHE[db_id] = _workspace()
            return _WS_CACHE[db_id]
        # open_workspace runs assert_workspace_dim: a workspace whose vectors are a
        # different width than EMBEDDING_DIM, or another model's by its knowledge ledger,
        # answers every near() with confident nonsense.
        per_ws = open_workspace(
            per_db,
            workspace_id_value=db_id,
            pragmas=_SQLITE_PRAGMAS,
            pool=_POOL,
            **_retrieval_open_kwargs(),
        )
        _WS_CACHE[db_id] = per_ws
        return per_ws
    _WS_CACHE[db_id] = _workspace()
    return _WS_CACHE[db_id]


# ── DB registry ───────────────────────────────────────────────────────────────
_DB_REGISTRY: dict[str, str] = {}
_REGISTRY_FILE = os.path.join(_DATA_DIR, "db_registry.json")


def _url_to_relative(url: str) -> str:
    if url.lower().startswith("sqlite:///"):
        path = url[len("sqlite:///"):]
        if os.path.isabs(path):
            try:
                return "sqlite:///" + os.path.relpath(path, _DATA_DIR)
            except ValueError:
                pass
    return url


def _url_to_absolute(url: str) -> str:
    if url.lower().startswith("sqlite:///"):
        path = url[len("sqlite:///"):]
        if not os.path.isabs(path):
            return "sqlite:///" + os.path.normpath(os.path.join(_DATA_DIR, path))
    return url


def _load_registry() -> None:
    if not os.path.exists(_REGISTRY_FILE):
        return
    with open(_REGISTRY_FILE, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not all(
        isinstance(db_id, str) and isinstance(url, str)
        for db_id, url in data.items()
    ):
        raise TypeError("db_registry.json must map string db_ids to string SQLite URLs")
    for db_id, url in data.items():
        _DB_REGISTRY[db_id] = _url_to_absolute(url)


def _persist_registry() -> None:
    """Write the registry atomically: a reader never sees a half-written file."""
    data = {db_id: _url_to_relative(url) for db_id, url in _DB_REGISTRY.items()}
    target_dir = os.path.dirname(os.path.abspath(_REGISTRY_FILE))
    os.makedirs(target_dir, exist_ok=True)
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=target_dir,
            prefix=f".{os.path.basename(_REGISTRY_FILE)}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, _REGISTRY_FILE)
    finally:
        if temporary and os.path.exists(temporary):
            os.remove(temporary)


_load_registry()


# Optional per-process override of the EXECUTION database directory. When set (e.g. by
# eval_bird via the RUBIKSQL_DB_DIR env var), every query/temp-table/submit runs against
# {RUBIKSQL_DB_DIR}/{db_id}.sqlite. As of the registry/profiling unification this is the
# SAME db (BirdSQL_raw) that the persisted registry points at and that scripts/setup_profiling.py
# profiled — so registry == profiling == execution on one DB. The override therefore just
# pins that DB explicitly per-eval (and keeps the agent on exactly the file the evaluator
# scores gold on, isolating it from any local re-registration). Schema/values come from the
# per-db workspace, which is built from this same raw DB. Unset → use the persisted registry.
_DB_DIR_OVERRIDE = os.environ.get("RUBIKSQL_DB_DIR")


def _db_url(db_id: str) -> str | None:
    """Resolve the execution URL for db_id, honoring RUBIKSQL_DB_DIR if set.

    Supports the flat RubikBench layout (<db>.sqlite) AND the official nested BIRD layout
    (<db>/<db>.sqlite). Deliberately STRICTER than rulegen.resolve_db_path (this
    module must not import repo-root scripts): that resolver may fall back to the flat path
    because its callers exists-check before opening, whereas every caller here hands the URL
    straight to SQLAlchemy — which CREATES a 0-byte database instead of failing. A mis-set
    RUBIKSQL_DB_DIR would then look healthy while every query answered "no such table", so a
    missing source database is raised here, naming both probed layouts.

    Raises:
        FileNotFoundError: RUBIKSQL_DB_DIR is set but holds neither layout for ``db_id``.
    """
    if _DB_DIR_OVERRIDE:
        flat = os.path.join(_DB_DIR_OVERRIDE, f"{db_id}.sqlite")
        if os.path.exists(flat):
            return "sqlite:///" + flat
        nested = os.path.join(_DB_DIR_OVERRIDE, db_id, f"{db_id}.sqlite")
        if os.path.exists(nested):
            return "sqlite:///" + nested
        raise FileNotFoundError(
            f"source database for '{db_id}' not found under RUBIKSQL_DB_DIR="
            f"'{_DB_DIR_OVERRIDE}'; probed '{flat}' (flat layout) and '{nested}' (official "
            f"nested layout). Point RUBIKSQL_DB_DIR at the directory holding the .sqlite "
            f"files — an empty database is never created to paper over this.")
    return _DB_REGISTRY.get(db_id)


# ── MCP tool functions ────────────────────────────────────────────────────────

# ── Build toolkit ─────────────────────────────────────────────────────────────

TOOL_WORKERS_ENV = "RUBIKSQL_TOOL_WORKERS"
TOOL_WORKERS_DEFAULT = 32
TOOL_WORKERS_MAX = 1024


def tool_worker_limit(raw: str | None) -> int:
    """The tool-worker token count one served event loop admits, from ``RUBIKSQL_TOOL_WORKERS``.

    Unset or empty keeps the default of 32. Otherwise the value, stripped of surrounding
    whitespace, must be a decimal integer from 1 to 1024; anything else fails server startup
    rather than falling back, so a mistyped launch never runs at an unintended width.

    Args:
        raw (str | None): The environment value, or ``None`` when the variable is unset.

    Returns:
        int: The configured limit.

    Raises:
        ValueError: ``raw`` is nonempty and not a decimal integer in ``[1, 1024]``.
    """
    if raw is None or not raw.strip():
        return TOOL_WORKERS_DEFAULT
    value = raw.strip()
    if not (value.isascii() and value.isdigit()) or not 1 <= int(value) <= TOOL_WORKERS_MAX:
        raise ValueError(f"{TOOL_WORKERS_ENV} must be an integer from 1 to {TOOL_WORKERS_MAX}, got {raw!r}")
    return int(value)


_TOOL_LOCKS: WeakValueDictionary[tuple[object, str], anyio.Lock] = WeakValueDictionary()
_TOOL_WORKER_LOCKS: WeakValueDictionary[str, Any] = WeakValueDictionary()
_TOOL_WORKER_LOCKS_GUARD = threading.Lock()
#: Tool worker tokens per served event loop, read once at startup (see :func:`tool_worker_limit`).
_TOOL_WORKER_LIMIT = tool_worker_limit(os.environ.get(TOOL_WORKERS_ENV))
_TOOL_LIMITER: RunVar[anyio.CapacityLimiter] = RunVar("rubiksql_hb.tool_limiter")
_TOOL_ACTIVITY = threading.Condition()
_TOOL_READERS = 0
_TOOL_WRITERS_WAITING = 0
_TOOL_WRITING = False


@contextmanager
def _tool_activity(*, registering: bool) -> Iterator[None]:
    """Keep registration atomic against tool readers without serializing readers."""
    global _TOOL_READERS, _TOOL_WRITERS_WAITING, _TOOL_WRITING
    with _TOOL_ACTIVITY:
        if registering:
            _TOOL_WRITERS_WAITING += 1
            _TOOL_ACTIVITY.notify_all()
            try:
                _TOOL_ACTIVITY.wait_for(lambda: not _TOOL_WRITING and _TOOL_READERS == 0)
                _TOOL_WRITING = True
            finally:
                _TOOL_WRITERS_WAITING -= 1
        else:
            _TOOL_ACTIVITY.wait_for(lambda: not _TOOL_WRITING and _TOOL_WRITERS_WAITING == 0)
            _TOOL_READERS += 1
    try:
        yield
    finally:
        with _TOOL_ACTIVITY:
            if registering:
                _TOOL_WRITING = False
            else:
                _TOOL_READERS -= 1
            _TOOL_ACTIVITY.notify_all()


def _example_call(name, function, args, kwargs):
    """Bind host question context without changing a pinned callable's schema."""
    signature = inspect.signature(function)
    clean = dict(kwargs)
    original = clean.get("original_question", "")
    if name in _EXAMPLE_READERS and "original_question" not in signature.parameters:
        clean.pop("original_question", None)
    arguments = dict(signature.bind(*args, **clean).arguments)
    if name in _EXAMPLE_READERS:
        from fastmcp.server.dependencies import get_context
        try:
            context = get_context().request_context
        except RuntimeError:
            context = None
        meta = context.meta if context else None
        question = meta.get(QUESTION_META) if meta else None
        arguments["original_question"] = question or original
    return clean, arguments


class _ConcurrentTool:
    """Keep pinned synchronous Capsules off the MCP event loop.

    HB guards workspace operations with its existing lifecycle locks. Session tools
    additionally enter a fair async lock before worker dispatch, so cleanup follows
    earlier requests without occupying worker slots. Weak references release those
    locks after the final caller. Tools use a separate bounded limiter so saturation
    cannot starve MCP's stdio work in AnyIO's default thread pool.
    """

    def __init__(self, tool: Any) -> None:
        self._tool = tool

    def __getattr__(self, name: str) -> Any:
        """Retain the native tool's Capsule, schemas, metadata and serialization."""
        return getattr(self._tool, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Apply host read policy after executing the unchanged pinned callable."""
        if self.name not in _EXAMPLE_READERS:
            return self._tool.run(*args, **kwargs)
        clean, arguments = _example_call(self.name, self.capsule.to_func(), args, kwargs)
        with question_scope(arguments.get("original_question") or arguments.get("question") or ""):
            value = self._tool.run(*args, **clean)
        return mask_tool_result(self.name, arguments, value)

    def _run_sync(self, *args: Any, **kwargs: Any) -> Any:
        _, arguments = _example_call(self.name, self.capsule.to_func(), args, kwargs)
        session_id = arguments.get("session_id")
        with _tool_activity(registering=self.name == "register_database"):
            if not session_id:
                return self.run(*args, **kwargs)
            # MCP's AnyIO cancellation waits for active workers. Keep worker-side
            # ownership too: a native asyncio Task.cancel() can bypass that shield.
            with _TOOL_WORKER_LOCKS_GUARD:
                lock = _TOOL_WORKER_LOCKS.get(session_id)
                if lock is None:
                    lock = threading.RLock()
                    _TOOL_WORKER_LOCKS[session_id] = lock
            with lock:
                return self.run(*args, **kwargs)

    async def arun(self, *args: Any, **kwargs: Any) -> Any:
        """Copy request context and finish active work under MCP's AnyIO cancellation."""
        function = self.capsule.to_func()
        clean, arguments = _example_call(self.name, function, args, kwargs)
        if inspect.iscoroutinefunction(function):
            with question_scope(arguments.get("original_question") or arguments.get("question") or ""):
                value = await self._tool.arun(*args, **clean)
            return mask_tool_result(self.name, arguments, value)
        session_id = arguments.get("session_id")
        limiter = _TOOL_LIMITER.get(None)
        if limiter is None:
            limiter = anyio.CapacityLimiter(_TOOL_WORKER_LIMIT)
            _TOOL_LIMITER.set(limiter)
        call = partial(self._run_sync, *args, **kwargs)
        entered = time.perf_counter()
        if session_id:
            # No await between lookup and publication: callers on this event loop
            # share one lock, while another loop owns independent async primitives.
            key = (current_token(), session_id)
            lock = _TOOL_LOCKS.get(key)
            if lock is None:
                lock = anyio.Lock()
                _TOOL_LOCKS[key] = lock
            async with lock:
                value = await self._run_limited(call, limiter, entered)
        else:
            value = await self._run_limited(call, limiter, entered)
        return await value if inspect.isawaitable(value) else value

    async def _run_limited(self, call: partial, limiter: anyio.CapacityLimiter, entered: float) -> Any:
        """Run ``call`` on a tool worker, noting its session wait, worker admission and execution."""
        span: dict[str, float] = {}

        def measured() -> Any:
            span["start"], span["borrowed"] = time.perf_counter(), limiter.borrowed_tokens
            try:
                return call()
            finally:
                span["end"] = time.perf_counter()

        asked, load = time.perf_counter(), limiter.statistics()   # on the loop thread, before this call queues
        try:
            return await anyio.to_thread.run_sync(measured, limiter=limiter)
        finally:
            if "end" in span:
                tool_timing.observe(self.name, session_wait=asked - entered, admission_wait=span["start"] - asked,
                                    execution=span["end"] - span["start"], borrowed=int(span["borrowed"]),
                                    waiting=load.tasks_waiting, limit=int(load.total_tokens))

    async def arun_to_str(self, *args: Any, **kwargs: Any) -> str:
        """Keep serialized calls on the concurrent execution path too."""
        return await self._tool.ato_str(await self.arun(*args, **kwargs))


def _mcp_toolkit():
    """The served MCP toolkit, built on first use.

    Registration is deferred together with the workspace: ``to_mcp()`` needs it, and building
    it is the 16.5 s this defers. The ordered callable declaration itself needs no workspace.

    Returns:
        The toolkit with every project tool added. Cached — repeated calls return the same one.
    """
    global _TOOLKIT
    if _TOOLKIT is None:                              # 无锁快路径
        with _BOOT_LOCK:                              # RLock:体内的 _workspace() 会再取一次
            if _TOOLKIT is None:
                toolkit = _workspace().to_mcp(name="rubiksql-hb")
                if toolkit._config.get("heavenbase.mcp.cache.enabled", default=False):
                    raise RuntimeError("answering MCP cache must be disabled for pinned session state")
                from .tool_bundle import process_bundle

                bundle = process_bundle()
                # answering-bundle/2 pins callable Capsules by id; /3 pins ToolEntity versions (no callable Capsule).
                if bundle.tool_rows:
                    pins = {"tool_entity_id": {row["name"]: row["object_id"] for row in bundle.tool_rows}}
                else:
                    pins = {"native_capsule_id": {reference["name"]: reference["capsule_id"]
                                                  for reference in bundle.payload.get("tools", [])}}
                from .hb_tool_entities import answering_selection

                for fn, record in zip(project_tool_functions(), answering_selection(bundle.definitions), strict=True):
                    # The workspace's artifact-loaded Toolkit has its own Tool class.
                    # Register natively before adapting execution, or add() recaptures
                    # a foreign Tool as a callable and loses the pinned Capsule.
                    toolkit.add(fn, name=record["name"], version=bundle.tool_digests[record["name"]], namespace="rubiksql.answering.tool")
                    toolkit.tools[record["name"]].metadata.update({"native_capsule_version": bundle.tool_digests[record["name"]],
                        **{key: ids[record["name"]] for key, ids in pins.items() if record["name"] in ids}})
                    toolkit.tools[record["name"]] = _ConcurrentTool(toolkit.tools[record["name"]])
                from .tool_bundle import assert_startup_parity

                assert_startup_parity(project_tool_functions(), tuple(toolkit.tools.values()), _prompt_release())
                _TOOLKIT = toolkit    # 最后才发布 —— 半建好的 toolkit 不得被别的线程看见
    return _TOOLKIT


@lru_cache(maxsize=None)
def _question_bound_reader(name: str):
    """Apply the same host policy to public library tools and pre-injection readers."""
    if name in _TOOL_FUNCTIONS:
        function = _TOOL_FUNCTIONS[name]
    else:
        owner, attribute = _PINNED_HELPER_EXPORTS[name]
        function = getattr(importlib.import_module(_TOOL_FUNCTIONS[owner].__module__), attribute)

    @wraps(function)
    def read(*args, **kwargs):
        clean, arguments = _example_call(name, function, args, kwargs)
        with question_scope(arguments.get("original_question") or arguments.get("question") or ""):
            return mask_tool_result(name, arguments, function(*args, **clean))

    return read


def __getattr__(name: str):
    """Keep ``server.toolkit`` working now that it is lazy (PEP 562).

    Module ``__getattr__`` fires on attribute access from outside — which is exactly how the
    contract tests reach ``server.toolkit`` — but NOT on a bare global lookup inside this file,
    so code in here must call :func:`_mcp_toolkit` directly.

    Args:
        name (str): Attribute being looked up.

    Returns:
        The toolkit, built on demand.

    Raises:
        AttributeError: For any other name, so typos still fail loudly.
    """
    if name == "toolkit":
        return _mcp_toolkit()
    if name in _EXAMPLE_READERS:
        return _question_bound_reader(name)
    if name in _TOOL_FUNCTIONS:
        return _TOOL_FUNCTIONS[name]
    if name in _PINNED_HELPER_EXPORTS:
        owner, attribute = _PINNED_HELPER_EXPORTS[name]
        module = importlib.import_module(_TOOL_FUNCTIONS[owner].__module__)
        return getattr(module, attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> int:
    """Serve over stdio; diagnostics go to stderr because stdout is the JSON-RPC channel.

    The launching parent (``nl2sql``'s ``stdio_client``) never reads this process's exit
    code — it sees a dead connection and this process's stderr, which is therefore the only
    channel that can carry the root cause. A failure anywhere in the serve lifetime (the
    workspace open, ``to_mcp``, or the stdio loop itself — ``serve`` covers all three) is
    printed with its full traceback; one summary line would hide the cause behind
    ``ExceptionGroup: unhandled errors in a TaskGroup``.

    The serve lifetime runs inside :func:`~rubiksql_hb.query_policy.code_reduce_disabled`:
    every query spec this process answers was written by a model, so HeavenBase refuses its
    ``code_reduce`` aggregate here on every tool, its native ``query`` included, without
    writing the switch into any HeavenBase home.

    Args:
        None.

    Returns:
        int: ``0`` after a clean shutdown, ``1`` after a failure.
    """
    try:
        sweep_sessions()   # clear any session files left by a hard-killed previous run
    except Exception as exc:   # noqa: BLE001 — hygiene must NEVER kill server startup
        print(f"[sweep] skipped (non-fatal): {exc}", file=sys.stderr, flush=True)
    # The knowledge tools embed their queries here, in this process: when the launching eval_bird
    # names its rate-limit record directory, the counts go there, current at every request —
    # eval_bird kills its servers at its exit, so a record written at the end would never land.
    record_dir = os.environ.get(rate_limits.RECORD_ENV)
    if record_dir:
        rate_limits.keep_record(record_dir, f"mcp-{_WORKSPACE_ID}")
        tool_timing.keep_record(record_dir, f"mcp-{_WORKSPACE_ID}")
    try:
        with code_reduce_disabled():
            _mcp_toolkit().serve(transport="stdio")
    except Exception as exc:  # noqa: BLE001 — the traceback IS the diagnostic; see above
        print(f"[server] failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return 1
    return 0


if __name__ == "__main__":
    # Tool declarations import this canonical name only when they execute.
    # The -m entrypoint must expose its active runtime under the same name.
    sys.modules["rubiksql_hb.server"] = sys.modules[__name__]
    raise SystemExit(main())
