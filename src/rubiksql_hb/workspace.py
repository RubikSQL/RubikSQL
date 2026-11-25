"""Workspace construction — the one place that knows HeavenBase's locator grammar.

Every per-database knowledge store carries its database identity end to end. For
``financial`` that means a directory named ``financial/``, scalar storage in
``financial.db``, a HeavenBase workspace id of ``financial``, and the Chroma database
namespace ``financial``. Call sites go through this module so the three identities cannot
drift:

* **Locators.** HeavenBase 0.1.2.x accepts only ``file://`` URIs for local backends — a
  native path raises ``ValueError``, and a Windows path raises again on the backslashes.
  :func:`location` does the conversion once, correctly, for both platforms.
* **Workspace lifecycle.** Normal readers use detached facades. Checkpoint ③ setup may
  create a durable registration, including a schema manifest, so HeavenBase's native
  ``hb dashboard`` command can reopen the workspace without RubikSQL-specific dashboard
  code. Checkpoint ② keeps using detached facades and therefore cannot replace checkpoint
  ③'s registry entry.
"""
from __future__ import annotations

import os
import contextlib
import sys
import threading
from collections import OrderedDict
import unicodedata
import warnings
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping

import heavenbase as hb
from heavenbase.utils import check_workspace_id
from heavenbase.utils.files import path_to_file_uri

if TYPE_CHECKING:
    from heavenbase.workspace import WorkspaceSpec

from .embedding import assert_workspace_dim
from .schema import (
    ColumnEntity,
    DatabaseEntity,
    EnumValueEntity,
    ExampleEntity,
    RuleEntity,
    SynonymEntity,
    TableEntity,
)

#: The six entity types a complete workspace carries, in registration order.
ALL_ENTITIES: tuple[type, ...] = (
    TableEntity, ColumnEntity, EnumValueEntity, ExampleEntity, SynonymEntity, RuleEntity,
    # Appended last so the first six keep their registration order (and the manifests of
    # workspaces registered before 08-05 differ only by this one addition). ws.register()
    # is additive-idempotent: opening an older workspace creates the new table on demand
    # without touching existing rows.
    DatabaseEntity,
)

CHROMA_DIR   = "chroma_db"

#: HNSW knobs :func:`sync_vector_index` writes into every collection it builds. Chroma's
#: defaults (``sync_threshold`` 1000, ``batch_size`` 100) are larger than any collection this
#: project creates — the biggest is 728 rows — so the graph was never written to disk and
#: every process rebuilt one from the write-ahead log instead. That rebuild is not
#: reproducible (20 rebuilds of one byte-identical workspace gave 9 distinct graphs), and it
#: cost 219 ms per collection on first access; flushing every 8 records makes the graph a
#: build-time artifact, read back in 11 ms. Sized to flush, not tuned: any value below the
#: smallest collection works, and 8 leaves at most 7 records replayed from the log.
SYNC_HNSW: dict[str, int] = {"sync_threshold": 8, "batch_size": 8}

#: The open-file limit :func:`ensure_file_limit` asks for before this process opens a Chroma
#: client. Chroma keeps every collection's HNSW index in an in-process pool of (open-file
#: limit) // 5 indexes split across 64 shards. A read of a collection this process has read
#: before, whose index the pool has since evicted, reloads only what was flushed to disk (every
#: ``sync_threshold`` records): nothing, for a small collection, and the read fails with ``Error
#: creating hnsw segment reader: Nothing found on disk`` until the collection is written again.
#: Windows' default of 512 stdio streams gives a pool of 102, about one index per shard: a
#: second read of 80 three-row collections in one process failed 34 of 160 times at 512, on the
#: Windows box and on macOS alike, and never at 2048, the most that msvcrt.dll (the C runtime
#: Chroma asks) accepts. The separate Universal CRT stream table also needs this capacity:
#: NumPy's FILE* conversion and OpenSSL's certificate reads use it, not msvcrt.dll's table.
FILE_LIMIT = 2048


def location(path: str | os.PathLike[str]) -> str:
    """Return ``path`` as the ``file://`` URI HeavenBase requires for a local backend.

    Delegates the encoding to ``heavenbase.utils.files.path_to_file_uri``, the exact inverse
    of the ``file_uri_to_path`` that HeavenBase's sqlite and chroma backends parse with — so
    the producing and consuming sides cannot drift. That helper demands an absolute path,
    while callers here pass repo-relative directories, so the path is absolutised first.

    Args:
        path (str | os.PathLike[str]): Native filesystem path, absolute or relative. It does
            not have to exist.

    Returns:
        str: Canonical ``file:///...`` URI. Windows drive letters and separators are
        normalised (``C:\\x\\y`` becomes ``file:///C:/x/y``) because HeavenBase rejects
        backslashes outright, and spaces are percent-encoded.

    Raises:
        ValueError: ``path`` resolves to a UNC or network location.
    """
    return path_to_file_uri(os.path.abspath(path))


def ensure_file_limit() -> int:
    """Raise this process's open-file limit to :data:`FILE_LIMIT`, so Chroma sizes its index pool for it.

    Chroma reads the limit when this process opens its first client on a directory, so
    :func:`open_workspace` and :func:`sync_vector_index` call this first, and so does any
    caller that builds a Chroma backend another way. A limit at or above :data:`FILE_LIMIT`
    is left as it is. One the operating system refuses to raise stays too, with a
    ``RuntimeWarning``: Chroma then keeps the smaller pool. On Windows, raise both the
    legacy ``msvcrt`` table Chroma reads and the separate ``ucrtbase`` table native
    file I/O uses; increasing only the former leaves NumPy and TLS reads capped at 512.

    Returns:
        int: The limit Chroma reads: ``msvcrt._getmaxstdio()`` on Windows, the soft
        ``RLIMIT_NOFILE`` elsewhere.
    """
    if os.name == "nt":
        import ctypes

        limits = {}
        for name in ("msvcrt", "ucrtbase"):
            crt = ctypes.CDLL(name)
            if crt._getmaxstdio() < FILE_LIMIT:
                crt._setmaxstdio(FILE_LIMIT)
            limits[name] = crt._getmaxstdio()
        if min(limits.values()) < FILE_LIMIT:
            warnings.warn(
                f"the Windows C runtime stream limits stay at {limits}, below {FILE_LIMIT}; "
                "native file I/O can exhaust a different stream table from Chroma's pool limit",
                RuntimeWarning, stacklevel=2)
        return limits["msvcrt"]
    else:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft != resource.RLIM_INFINITY and soft < FILE_LIMIT:
            wanted = FILE_LIMIT if hard == resource.RLIM_INFINITY else min(FILE_LIMIT, hard)
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
            except (OSError, ValueError):
                pass
        limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if limit < FILE_LIMIT:
        warnings.warn(
            f"the open-file limit stays at {limit}, below {FILE_LIMIT}: Chroma's HNSW index pool "
            f"holds {limit // 5} indexes in 64 shards, and a collection whose unflushed index "
            f"it evicts fails every read in this process until the collection is written again",
            RuntimeWarning, stacklevel=2)
    return limit


def workspace_id(data_dir: str | os.PathLike[str], value: str | None = None) -> str:
    """Return the validated identity for a workspace directory.

    Args:
        data_dir: Workspace directory. Its final path component is the default id.
        value: Explicit id when the directory name is not the intended identity.

    Returns:
        A valid HeavenBase workspace id.

    Raises:
        ValueError: The explicit value or directory basename is not a valid id.
    """
    candidate = value or os.path.basename(os.path.normpath(os.fspath(data_dir)))
    return check_workspace_id(candidate)


def workspace_db_path(
    data_dir: str | os.PathLike[str],
    *,
    workspace_id_value: str | None = None,
) -> str:
    """Return the scalar SQLite path for one workspace.

    Args:
        data_dir: Workspace directory.
        workspace_id_value: Explicit workspace id. Defaults to the directory basename.

    Returns:
        Native path ``<data_dir>/<workspace_id>.db``.
    """
    ws_id = workspace_id(data_dir, workspace_id_value)
    return os.path.join(os.fspath(data_dir), f"{ws_id}.db")


def backend_config(data_dir: str | os.PathLike[str], *,
                   workspace_id_value: str | None = None,
                   pragmas: list[str] | None = None,
                   pool: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Return the ``backends`` mapping for the workspace rooted at ``data_dir``.

    Args:
        data_dir: Directory holding ``<workspace_id>.db`` and ``chroma_db/``.
        workspace_id_value: Explicit workspace id. Defaults to the directory basename.
        pragmas: sqlite pragmas to apply per engine. HeavenBase REPLACES its provider
            defaults with this list, so pass every pragma that must hold, not a delta.
        pool: SQLAlchemy pool options, e.g. ``{"connect_args": {"timeout": 60}}``.

    Returns:
        A mapping with the ``main`` (sqlite) and ``chroma`` (vector) backend aliases.
    """
    main: dict[str, Any] = {
        "type": "sqlite",
        "database": location(workspace_db_path(
            data_dir, workspace_id_value=workspace_id_value,
        )),
    }
    if pool:
        main["pool"] = dict(pool)
    if pragmas:
        main["pragmas"] = list(pragmas)
    return {
        "main":   main,
        "chroma": {"type": "chroma", "path": location(os.path.join(data_dir, CHROMA_DIR))},
    }


def _registry_manifest(ws: hb.HeavenBase) -> dict[str, Any]:
    """Return a manifest that native HeavenBase processes can replay.

    RubikSQL opens operational workspaces from the Python Entity classes, which retain
    their computes. A plain ``hb dashboard`` process has not registered application
    callables (and does not register optional HeavenBase utility computes either), so
    replaying their qualified references would fail before the workspace could be browsed.
    The dashboard only needs the field and storage contracts; remove compute hooks from the
    durable manifest. This does not alter the operational Entity classes or stored rows.

    Args:
        ws: Fully registered RubikSQL workspace.

    Returns:
        Replayable workspace manifest for HeavenBase's native registry.
    """
    manifest = ws.to_manifest()
    for entity in manifest.get("entities", []):
        for field in entity.get("fields", {}).values():
            for key in ("computes", "query_computes"):
                field.pop(key, None)
    return manifest


@contextlib.contextmanager
def _registered_live_workspace(context: hb.Context, workspace_id: str) -> Iterator[hb.HeavenBase | None]:
    """Inspect the pinned HB live cache without opening a saved descriptor.

    The pinned artifact exposes no public side-effect-free lookup for this cache.
    Its owning lifecycle accessor only reads it; publication remains the public
    HeavenBase.from_manifest path. An unavailable cache contract fails closed.

    Yields:
        The cached facade, or None, with the Context lock held through validation
        and the caller's public from_manifest publication.
    """
    try:
        from heavenbase.workspace.setup.lifecycle import _registered_workspace
        from heavenbase.utils.registry import Registry

        lock = context._lock
        if not isinstance(context._workspace_registry, Registry):
            raise TypeError("unexpected live-workspace cache")
    except (ImportError, AttributeError, TypeError) as error:
        raise RuntimeError("pinned HeavenBase live-workspace lookup is unavailable") from error
    with lock:
        try:
            live = _registered_workspace(context, workspace_id)
            if live is not None and not isinstance(live, hb.HeavenBase):
                raise TypeError("unexpected live-workspace value")
        except (AttributeError, TypeError) as error:
            raise RuntimeError("pinned HeavenBase live-workspace lookup is unavailable") from error
        yield live


def check_retrieval_version(token: str) -> str:
    """Validate one HB ``cache_version`` token and return it unchanged.

    The token names an immutable knowledge generation. HB keys its candidate cache on it
    and replays a cached generation WITHOUT re-checking the backend, so passing one asserts
    that no other process writes this workspace's knowledge for as long as the process that
    opened it lives — this process's own writes invalidate the cache; an external writer
    needs a new token or ``ws.refresh()``. The token travels through environment variables
    and cache keys, so it must be one printable word: no whitespace, no control characters.

    Args:
        token (str): Candidate generation token.

    Returns:
        str: ``token`` itself, when it is acceptable.

    Raises:
        ValueError: ``token`` is not a string, is empty, or contains whitespace or a control
            character.
    """
    if not isinstance(token, str) or not token.strip():
        raise ValueError("retrieval_version must be a non-empty string")
    if any(ch.isspace() or unicodedata.category(ch) == "Cc" for ch in token):
        raise ValueError(
            f"retrieval_version must be one printable token without whitespace or control "
            f"characters, got {token!r}")
    return token


def open_workspace(data_dir: str | os.PathLike[str], *,
                   entities: Iterable[type] = ALL_ENTITIES,
                   workspace_id_value: str | None = None,
                   context: hb.Context | None = None,
                   retrieval_exact: bool = True,
                   retrieval_version: str | None = None,
                   pragmas: list[str] | None = None,
                   pool: Mapping[str, Any] | None = None,
                   toolkit: bool = False) -> hb.HeavenBase:
    """Open (or create) the workspace rooted at ``data_dir`` with ``entities`` registered.

    Args:
        data_dir: Directory holding ``<workspace_id>.db`` and ``chroma_db/``. Not created —
            callers that build a fresh workspace make it themselves.
        entities: Entity classes to register. Defaults to :data:`ALL_ENTITIES`; pass a
            narrower tuple when only one table is touched.
        workspace_id_value: Explicit HeavenBase workspace id. Defaults to the directory
            basename. The same id selects the SQLite filename and Chroma database namespace.
        context: HeavenBase machine context. Defaults to the process context.
        retrieval_exact: Require exhaustive HB ranking for every
            :func:`~rubiksql_hb.retrieval.search_vectors` call that does not say otherwise.
            ``True`` — the BIRD default — is what makes retrieval reproducible across
            processes; ``False`` permits the vector backend's approximate route.
        retrieval_version: Explicit immutable knowledge-generation token handed to HB as
            ``cache_version``, so repeated exact scans reuse decoded candidates instead of
            re-reading the store. ``None`` keeps live reads. The token asserts that no
            OTHER process writes this workspace's knowledge for the lifetime of this one —
            HB replays the cached generation without re-checking the backend; this
            process's own writes invalidate the cache, external ones need a new token or
            ``ws.refresh()``. Validated by :func:`check_retrieval_version`: one printable
            word, no whitespace or control characters.
        pragmas: See :func:`backend_config`.
        pool: See :func:`backend_config`.
        toolkit: Enable the ``toolkit`` extension. Required by, and only by, the workspace
            that calls ``to_mcp()`` — a detached workspace comes up with ``capsule`` and
            ``prompt`` enabled but not ``toolkit``, and ``to_mcp()`` then fails with
            ``Registry contribution toolkit_function:workspace_toolkit is unavailable``.
    Returns:
        A detached :class:`heavenbase.HeavenBase` with ``entities`` registered.

    Raises:
        ValueError: ``retrieval_exact`` is not a boolean, or ``retrieval_version`` fails
            :func:`check_retrieval_version`.
        RuntimeError: Stored vector width disagrees with ``EMBEDDING_DIM``, or the workspace's knowledge ledger
            records vectors of another embedding model (:func:`~rubiksql_hb.embedding.assert_workspace_dim`).
    """
    if not isinstance(retrieval_exact, bool):
        raise ValueError("retrieval_exact must be a boolean")
    if retrieval_version is not None:
        check_retrieval_version(retrieval_version)
    if os.path.isdir(data_dir):
        from .knowledge_safety import scan_knowledge

        scan_knowledge(data_dir)
    assert_workspace_dim(data_dir)   # another width or model = confident nonsense; see its docstring
    ws_id = workspace_id(data_dir, workspace_id_value)
    ensure_file_limit()
    ws = hb.HeavenBase(
        ws_id,
        context=context,
        backends=backend_config(
            data_dir,
            workspace_id_value=ws_id,
            pragmas=pragmas,
            pool=pool,
        ),
        detached=True,
    )
    # Read back by retrieval.search_vectors; HB itself has no per-workspace exactness default.
    ws._rubiksql_retrieval_exact = retrieval_exact
    ws._rubiksql_retrieval_version = retrieval_version
    if toolkit:
        ws.enable_extension("toolkit")
    for ent in entities:
        ws.register(ent)
    if retrieval_version is not None:
        reuse_query_results(ws)
    return ws


#: Remembered results per workspace; the least recently used are dropped beyond either bound.
QUERY_REUSE_MAX = 4096
QUERY_REUSE_MAX_BYTES = 64 * 1024 * 1024


def _frame_bytes(frame: Any) -> int:
    """Conservatively account the Python-owned cells of one result frame (as HB sizes its candidates)."""
    def size(value: Any) -> int:
        if isinstance(value, Mapping):
            return sys.getsizeof(value) + sum(size(key) + size(item) for key, item in value.items())
        if isinstance(value, (list, tuple)):
            return sys.getsizeof(value) + sum(size(item) for item in value)
        return sys.getsizeof(value)

    columns = getattr(frame, "_columns", None)
    return size(columns) if isinstance(columns, Mapping) else sys.getsizeof(frame)


def _holds_vectors(frame: Any) -> bool:
    names = getattr(frame, "col_names", None)
    return callable(names) and any(str(name).endswith("_vec") for name in names())


def reuse_query_results(ws: hb.HeavenBase) -> bool:
    """Serve a repeated plain query of an immutable knowledge generation from its first result.

    An answering server asks the same questions of its workspace over and over: every
    ``get_db_info`` runs one query per table, every ``match_rule_triggers`` reads all of the
    database's rules, and each such HB query costs tens of milliseconds of Python held under
    the HB Context lock, which serializes every query in the process. Under answering load
    the queries queue on that lock, and tool round trips grow far beyond their own work.

    With a ``retrieval_version`` the workspace's knowledge is asserted immutable for the life
    of the process (:func:`check_retrieval_version`), so a plain query (no ``near``, no
    ``traverse``) returns what it returned before; the request is identified by its complete
    dataclass rendering. Its :class:`heavenbase.ResultFrame` is detached and immutable, and
    every public accessor (``rows()``, ``scalar()`` …) returns copies, so one frame serves
    every later caller with unchanged results. Vector queries keep HB's own versioned
    candidate reuse. A frame that carries an embedding column (``*_vec``, e.g. an unprojected
    ``query_knowledge``) is never kept, and the kept frames are bounded by
    :data:`QUERY_REUSE_MAX` entries and :data:`QUERY_REUSE_MAX_BYTES`, least recently used first.

    HB announces each write, schema registration, refresh and release of this workspace by
    calling its candidate cache's ``invalidate``; such a call from outside the cache discards
    every remembered result, and a query that began before it never stores its frame. The
    cache's own calls to ``invalidate`` (a candidate miss, a version change, an eviction under
    its byte or entry bound) change no data and leave remembered results alone.

    Args:
        ws (heavenbase.HeavenBase): A workspace opened with a ``retrieval_version``.

    Returns:
        bool: Whether reuse was installed; ``False`` leaves live reads when this HB build
        does not expose the invalidation it relies on.
    """
    candidates = getattr(getattr(getattr(ws, "_router", None), "_near", None), "cache", None)
    invalidate = getattr(candidates, "invalidate", None)
    execute = getattr(ws, "execute", None)
    if not callable(invalidate) or not callable(execute):
        return False
    frames: OrderedDict[Any, tuple[Any, int]] = OrderedDict()
    held = [0]
    guard = threading.Lock()
    generation = [0]

    def invalidating(*args: Any, **kwargs: Any) -> Any:
        # The cache's own methods call ``self.invalidate`` for misses and evictions; only a caller
        # outside it (writer, schema, refresh/release) reports a change of the workspace's data.
        if sys._getframe(1).f_locals.get("self") is not candidates:
            with guard:
                frames.clear()
                held[0] = 0
                generation[0] += 1
        return invalidate(*args, **kwargs)

    def reusing(query: Any) -> Any:
        if getattr(query, "near", None) is not None or getattr(query, "traverse", None) is not None:
            return execute(query)
        # The request's full dataclass rendering: filter trees hold lists (unhashable), and equality
        # alone would take ``False`` for ``0``; ``repr`` keeps every field and each value's type.
        key = (type(query), repr(query))
        with guard:
            entry = frames.get(key)
            if entry is not None:
                frames.move_to_end(key)
            seen = generation[0]
        if entry is not None:
            return entry[0]
        frame = execute(query)
        if _holds_vectors(frame):
            return frame
        size = _frame_bytes(frame)
        with guard:
            if generation[0] == seen and key not in frames and size <= QUERY_REUSE_MAX_BYTES:
                frames[key] = (frame, size)
                held[0] += size
                while len(frames) > QUERY_REUSE_MAX or held[0] > QUERY_REUSE_MAX_BYTES:
                    held[0] -= frames.popitem(last=False)[1][1]
        return frame

    candidates.invalidate = invalidating
    ws.execute = reusing
    return True


def register_workspace(
    data_dir: str | os.PathLike[str],
    *,
    workspace_id_value: str | None = None,
    entities: Iterable[type] = ALL_ENTITIES,
    context: hb.Context | None = None,
    retrieval_exact: bool = True,
    pragmas: list[str] | None = None,
    pool: Mapping[str, Any] | None = None,
) -> WorkspaceSpec:
    """Persist one workspace for discovery by HeavenBase's native dashboard.

    Operational RubikSQL readers stay detached. This function snapshots their schema and
    backend construction into ``WorkspaceSpec`` without leaving a live workspace in the
    process registry, which is important on Windows where 11 retained Chroma clients keep
    HNSW files locked.

    Args:
        data_dir: Existing per-database workspace directory.
        workspace_id_value: Explicit id. Defaults to the directory basename.
        entities: Entity classes to expose in the dashboard.
        context: HeavenBase machine context. Defaults to the process context.
        retrieval_exact: Exhaustive requirement; see :func:`open_workspace`.
        pragmas: See :func:`backend_config`.
        pool: See :func:`backend_config`.

    Returns:
        The durable :class:`heavenbase.workspace.WorkspaceSpec`.
    """
    from heavenbase.workspace import WorkspaceSpec

    ws = open_workspace(
        data_dir,
        entities=entities,
        workspace_id_value=workspace_id_value,
        context=context,
        retrieval_exact=retrieval_exact,
        pragmas=pragmas,
        pool=pool,
        toolkit=True,
    )
    return WorkspaceSpec.register(
        ws.id,
        manifest=_registry_manifest(ws),
        context=ws.context,
    )


def sync_vector_index(
    data_dir: str | os.PathLike[str],
    *,
    workspace_id_value: str | None = None,
    force: bool = False,
) -> list[tuple[str, int]]:
    """Rewrite ``data_dir``'s Chroma collections so their HNSW graph is stored on disk.

    Reads each collection's vectors back out and re-adds them under :data:`SYNC_HNSW`, which
    is small enough that Chroma's writer flushes. Purely local: no embedding is recomputed
    and no API is called, so this is safe to run on a workspace built hours ago. Chroma keeps
    a collection's stored configuration across ``get_or_create_collection``, so later writes
    through HeavenBase — which only ever passes ``metadata={"hnsw:space": ...}`` — keep
    flushing too.

    Idempotent: a collection already at or below :data:`SYNC_HNSW`'s ``sync_threshold`` is
    left untouched unless ``force`` is set. Chroma drops a deleted collection's row from its
    ``segments`` table but leaves the segment's directory of ``.bin`` files behind, so the
    replaced (all-zero) index directories are swept afterwards — otherwise every rewrite adds
    another 424 KB of dead weight per collection to a workspace that gets packaged and shipped.
    Windows may keep an obsolete HNSW file memory-mapped until the Chroma client exits; that
    specific sharing violation defers only the space-saving sweep and is retried on the next
    sync. The replacement collection is already committed and remains authoritative.

    Args:
        data_dir: Workspace directory, i.e. the parent of ``chroma_db/``.
        workspace_id_value: Explicit Chroma database namespace. Defaults to the workspace
            directory basename.
        force: Rebuild every collection even when its configuration already flushes. Use it
            to repair a workspace whose stored graph is suspect.

    Returns:
        list[tuple[str, int]]: ``(collection_name, rows_rewritten)`` for each collection this
        call rebuilt, in name order. Empty when everything was already synced.

    Raises:
        FileNotFoundError: ``data_dir`` holds no Chroma store to sync.
        RuntimeError: A rebuilt collection came back with a different row count than it went
            in with, i.e. vectors were lost in the rewrite.
    """
    import chromadb

    path = os.path.join(data_dir, CHROMA_DIR)
    if not os.path.exists(os.path.join(path, "chroma.sqlite3")):
        raise FileNotFoundError(
            f"no Chroma store at {path} — build the workspace with scripts/setup_dbs.py first")
    ensure_file_limit()
    client = chromadb.PersistentClient(
        path=path,
        database=workspace_id(data_dir, workspace_id_value),
    )
    rebuilt: list[tuple[str, int]] = []
    for name in sorted(c.name for c in client.list_collections()):
        old = client.get_collection(name)
        hnsw = (old.configuration_json or {}).get("hnsw") or {}
        if not force and hnsw.get("sync_threshold", 1 << 30) <= SYNC_HNSW["sync_threshold"]:
            continue
        data = old.get(include=["embeddings", "metadatas", "documents"])
        ids = list(data["ids"])
        client.delete_collection(name)
        fresh = client.create_collection(
            name=name,
            configuration={"hnsw": {"space": hnsw.get("space", "cosine"), **SYNC_HNSW}},
        )
        if ids:
            fresh.add(
                ids=ids,
                embeddings=[[float(x) for x in vec] for vec in data["embeddings"]],
                metadatas=[meta or {"_hb": True} for meta in data["metadatas"]],
                documents=[doc if doc is not None else "" for doc in data["documents"]],
            )
        if fresh.count() != len(ids):
            raise RuntimeError(
                f"{name} in {path} came back with {fresh.count()} of {len(ids)} vectors after "
                f"the rewrite — the collection is now incomplete and every lookup against it "
                f"would silently return the wrong neighbours. Rebuild this workspace.")
        rebuilt.append((name, len(ids)))
    # Sweep on no-op calls too: a Windows process may have deferred an obsolete
    # segment while its predecessor still held data_level0.bin open.
    _sweep_orphan_segments(path)
    from .knowledge_safety import seal_knowledge

    seal_knowledge(data_dir)
    return rebuilt


def _sweep_orphan_segments(chroma_path: str) -> list[str]:
    """Delete index directories Chroma no longer lists as segments.

    Chroma can keep a deleted collection's HNSW files memory-mapped on Windows.
    ``WinError 32`` and ``33`` therefore mean only that an already-orphaned
    directory cannot be reclaimed until the client process exits. Those two
    sharing violations are reported and deferred; every other filesystem error
    remains fatal.

    Args:
        chroma_path: The ``chroma_db/`` directory to sweep.

    Returns:
        Paths whose deletion was deferred because Windows still held an HNSW
        file open. Empty when every orphan was removed.
    """
    import sqlite3

    from heavenbase.utils import delete_path

    conn = sqlite3.connect(os.path.join(chroma_path, "chroma.sqlite3"))
    try:
        live = {row[0] for row in conn.execute("SELECT id FROM segments")}
    finally:
        conn.close()
    deferred = []
    for name in os.listdir(chroma_path):
        entry = os.path.join(chroma_path, name)
        if os.path.isdir(entry) and name not in live:
            try:
                delete_path(entry)
            except OSError as exc:
                if getattr(exc, "winerror", None) not in {32, 33}:
                    raise
                deferred.append(entry)
    if deferred:
        print(
            f"    HNSW cleanup deferred for {len(deferred)} obsolete segment(s): "
            "Windows still has data_level0.bin open; retrieval uses the replacement "
            "segments and the next process will retry cleanup.",
            flush=True,
        )
    return deferred
