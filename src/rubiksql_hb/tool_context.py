"""Trusted read capabilities for independently deployed assistant and META tools.

The host owns workspace selection and native read/execution boundaries. Tool source
owns project behavior. No callable accepts these capabilities as model arguments.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname


def _fixed_identity(value: Any) -> Any:
    """Detach JSON identity metadata and freeze all containers recursively."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("read tool identity mappings require string keys")
        return MappingProxyType({key: _fixed_identity(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_fixed_identity(child) for child in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("read tool identities contain only inert JSON metadata")


@dataclass(frozen=True)
class NativeReadServices:
    """Explicit native HB read operations bound to the context's workspace.

    Native factories may supply read primitives; this is not a project service
    registry. Optional services are available only to their deployed consumer.
    """

    list_entities: Callable[..., Any]
    describe_entity: Callable[..., Any]
    query: Callable[..., Any]
    explain: Callable[..., Any]
    get: Callable[..., Any] | None = None
    count: Callable[..., Any] | None = None
    sample_rows: Callable[..., Any] | None = None
    column_stats: Callable[..., Any] | None = None
    search_workspace: Callable[..., Any] | None = None


@dataclass
class ToolContextState:
    """Invocation-owned evidence and diagnostic state, never formal score state."""

    evidence_reader: Any = None
    diagnostic_sql: Any = None
    evidence_lock: Any = field(default_factory=threading.RLock)
    diagnostic_lock: Any = field(default_factory=threading.RLock)


@dataclass(frozen=True)
class ToolExecutionContext:
    """Immutable host binding for read-only tools and optional META diagnostics.

    Args:
        role: Trusted consumer, ``assistant`` or ``meta``.
        workspace: HB facade backed by physically guarded SQLite readers.
        workspace_identity: Captured workspace/binding metadata; ``db_path`` when
            present must match the actual read backend.
        host_identity: Deployed host and vendored native implementation identities.
        native_readers: Explicit workspace-bound native read primitives.
        lab_entities: META alias-to-entity map, absent for assistant.
        authority_reader: META's read-only history primitive, never an authority writer.
        history_reader: META's immutable store read primitive for recorded row history.
        db_root: Fixed original database root for META diagnostics.
        analysis_cache_dir: Invocation-owned temporary diagnostic cache.
        analysis_scope: Fixed diagnostic request identity scope.
        state: Evidence snapshots, diagnostic helper and their locks.

    Raises:
        ValueError: Role, identity, native services or resource ownership is invalid.
    """

    role: str
    workspace: Any
    workspace_identity: Mapping[str, Any]
    host_identity: Mapping[str, Any]
    native_readers: NativeReadServices
    lab_entities: Mapping[str, Any] = field(default_factory=dict)
    authority_reader: Any = None
    history_reader: Callable[..., Any] | None = None
    db_root: Path | None = None
    analysis_cache_dir: Path | None = None
    analysis_scope: str | None = None
    state: ToolContextState = field(default_factory=ToolContextState)

    def __post_init__(self) -> None:
        if self.role not in {"assistant", "meta"}:
            raise ValueError("read tool contexts bind only assistant or meta")
        if (not isinstance(self.workspace_identity, Mapping) or not self.workspace_identity
                or not isinstance(self.host_identity, Mapping) or not self.host_identity
                or not isinstance(self.native_readers, NativeReadServices)
                or not isinstance(self.lab_entities, Mapping) or not isinstance(self.state, ToolContextState)):
            raise ValueError("read tool context needs fixed identities and explicit read services")
        for name in NativeReadServices.__dataclass_fields__:
            service = getattr(self.native_readers, name)
            if service is not None and not callable(service):
                raise ValueError(f"native read service {name} is not callable")
        if any(getattr(self.native_readers, name) is None for name in ("list_entities", "describe_entity", "query", "explain")):
            raise ValueError("read tool context needs all shared native readers")
        if self.role == "assistant" and (self.lab_entities or self.authority_reader is not None or self.history_reader is not None
                or any(value is not None for value in (self.db_root, self.analysis_cache_dir, self.analysis_scope))):
            raise ValueError("assistant context cannot bind META resources")
        if self.authority_reader is not None:
            if not callable(self.authority_reader) or any(hasattr(self.authority_reader, name)
                    for name in ("publish", "publish_tools", "deploy_tools", "promote", "allocate", "score", "writer")):
                raise ValueError("tool context cannot contain authority write or scoring capabilities")
        if self.history_reader is not None and not callable(self.history_reader):
            raise ValueError("history reader must be a trusted read primitive")
        if self.analysis_scope is not None and (not isinstance(self.analysis_scope, str) or not self.analysis_scope.strip()):
            raise ValueError("analysis scope must be a fixed nonempty identity")
        if (self.analysis_cache_dir is None) != (self.analysis_scope is None):
            raise ValueError("diagnostic cache and invocation scope must be bound together")
        object.__setattr__(self, "workspace_identity", _fixed_identity(self.workspace_identity))
        object.__setattr__(self, "host_identity", _fixed_identity(self.host_identity))
        object.__setattr__(self, "lab_entities", MappingProxyType(dict(self.lab_entities)))
        for name in ("db_root", "analysis_cache_dir"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())


_CONTEXT: ContextVar[ToolExecutionContext | None] = ContextVar("rubiksql_read_tool_context", default=None)


def _validate_read_workspace(context: ToolExecutionContext) -> None:
    """Verify the real guarded backend and live connection, not a read-only flag."""
    from .prompts.readonly import ReadOnlySQLiteBackend

    registry = getattr(context.workspace, "backends", None)
    if registry is None:
        raise ValueError("read tool context requires an actual guarded HB workspace")
    backends = dict(registry.items())
    if not backends or str(getattr(context.workspace, "id", "")) != str(context.workspace_identity.get(
            "workspace_id", getattr(context.workspace, "id", ""))):
        raise ValueError("read tool workspace identity differs from its captured binding")
    expected_path = context.workspace_identity.get("db_path")
    paths = set()
    for backend in backends.values():
        if not isinstance(backend, ReadOnlySQLiteBackend) or backend._binding:
            raise ValueError("tool workspace must enforce physical read-only access on every backend")
        engine = backend.db.engine
        uri = getattr(engine, "_rubiksql_readonly_uri", "")
        parsed = urlparse(uri)
        if parsed.scheme != "file" or parsed.query != "mode=ro" or parsed.params or parsed.fragment:
            raise ValueError("tool workspace connection is not physically mode=ro")
        # File URIs encode Windows drive paths as /C:/... and UNC hosts in
        # netloc. Decode with the platform converter before checking SQLite's
        # live path; Path('/C:/...') is not the Windows path C:\\... .
        uri_path = f"//{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
        path = Path(url2pathname(uri_path)).resolve()
        paths.add(path)
        with engine.connect() as connection:
            if connection.exec_driver_sql("PRAGMA query_only").scalar() != 1:
                raise ValueError("tool workspace connection does not enforce query_only")
            actual = connection.exec_driver_sql("PRAGMA database_list").all()
            if not any(row[1] == "main" and Path(row[2]).resolve() == path for row in actual):
                raise ValueError("tool workspace live database differs from its read-only URI")
    if expected_path is not None and paths != {Path(expected_path).expanduser().resolve()}:
        raise ValueError("tool workspace database differs from its captured binding")


def current_tool_context(*, role: str | None = None) -> ToolExecutionContext:
    """Return the trusted invocation context and reject an absent or wrong role.

    Args:
        role: Optional required consumer role, never a model argument.

    Returns:
        ToolExecutionContext: Currently bound host read capabilities.

    Raises:
        ValueError: There is no context or its consumer differs.
    """
    context = _CONTEXT.get()
    if context is None or role is not None and context.role != role:
        raise ValueError("tool invocation has no matching trusted read context")
    return context


@contextmanager
def bind_tool_context(context: ToolExecutionContext) -> Iterator[ToolExecutionContext]:
    """Bind one verified read context for a complete invocation and always reset it.

    Args:
        context: Immutable capabilities selected by the host.

    Yields:
        ToolExecutionContext: The verified binding while the invocation runs.

    Raises:
        ValueError: Context or actual workspace access is unsafe.
    """
    if not isinstance(context, ToolExecutionContext) or context.role not in {"assistant", "meta"}:
        raise ValueError("only trusted assistant/meta contexts may be bound")
    _validate_read_workspace(context)
    token = _CONTEXT.set(context)
    try:
        yield context
    finally:
        _CONTEXT.reset(token)
