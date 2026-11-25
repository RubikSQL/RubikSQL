"""The single HB version authority and its deterministic, recoverable exports.

Readers hold one SQLite transaction and never construct a HeavenBase workspace.
Writers own the instance fence and process lock. Immutable native rows are written
first; a pointer decision is the commit point, and an export certifies the batch.

The closed version scope is ``store.TABLES``: the six native sets plus the canonical typed
Tool, rule and parameter-set versions (``store.TYPED_TABLES``; only rows with a canonical
``format`` belong to them). New exports write format 4 (``rubiksql.authority-head/4``, 16
sets with row-addressed evidence); format 3 retains the same logical sets. Formats 1 and 2
(the six legacy sets) remain readable, restorable and fresh only while no canonical typed row exists.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from .evolve import OutcomeComputation

from . import binding as B
from . import entity_model as M
from .canon import canon, sha256_hex
from .store import LEGACY_TABLES, POINTER_KINDS, TABLES, TYPED_TABLES, _bytes, _canon_json, _json_value, project_version_files
from .types import AuthorityBinding, AuthorityError, AuthorityHead, AuthorityRevision, GitAuthorityBinding

SCHEMA_AUTHORITY = "rubiksql.authority/1"
SCHEMA_GIT_AUTHORITY = "rubiksql.authority/2"
SCHEMA_EXPORT = "rubiksql.authority-export/1"
SCHEMA_HEAD = "rubiksql.authority-head/1"
SCHEMA_WORKSPACE_EXPORT = "rubiksql.authority-export/2"
SCHEMA_WORKSPACE_HEAD = "rubiksql.authority-head/2"
SCHEMA_WORKSPACE = "rubiksql.workspace-state/1"
SCHEMA_TYPED_EXPORT = "rubiksql.authority-export/3"
SCHEMA_TYPED_HEAD = "rubiksql.authority-head/3"
SCHEMA_TYPED_WORKSPACE = "rubiksql.workspace-state/2"
SCHEMA_ROW_EXPORT = "rubiksql.authority-export/4"
SCHEMA_ROW_HEAD = "rubiksql.authority-head/4"
#: Export format -> (HEAD schema, receipt schema, STATE schema or None for version-only, closed version sets).
EXPORT_FORMATS: Mapping[int, tuple[str, str, str | None, Mapping[str, str]]] = {
    1: (SCHEMA_HEAD, SCHEMA_EXPORT, None, LEGACY_TABLES),
    2: (SCHEMA_WORKSPACE_HEAD, SCHEMA_WORKSPACE_EXPORT, SCHEMA_WORKSPACE, LEGACY_TABLES),
    3: (SCHEMA_TYPED_HEAD, SCHEMA_TYPED_EXPORT, SCHEMA_TYPED_WORKSPACE, TABLES),
    4: (SCHEMA_ROW_HEAD, SCHEMA_ROW_EXPORT, SCHEMA_TYPED_WORKSPACE, TABLES)}
#: The format every new export writes.
EXPORT_FORMAT = 4
ANCILLARY_FILES = frozenset({"metadata/tool-tiers.json"})
DECISION_KINDS = POINTER_KINDS | {"reject", "abandon", "retire", "unretire", "revoke", "unrevoke"}
#: Columns this typed-entity change added; a table lacking only these predates it and holds no canonical row.
_TYPED_COLUMNS: Mapping[str, frozenset[str] | None] = {
    M.TOOL_ENTITY: None, M.RULE_ENTITY: frozenset({"format", "version", "digest", "layout"}),
    M.PARAMETER_ENTITY: frozenset({"format", "digest"})}
_TYPED_BATCH = ("tool_rows", "rule_rows", "parameter_rows")
_MIGRATION_CLAIM = {"claim": "Representation change only; no EX gain claimed"}
_EPHEMERAL_FIELDS = frozenset({"created_at", "updated_at"})
#: Rows one HB batch write carries during recovery and projection catch-up.
_RESTORE_BATCH_ROWS = 2048
_KEY_PATTERN = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|LTAI[A-Za-z0-9]{16,})\b")
_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_LOCK_STATE = threading.local()
_MemoValue = TypeVar("_MemoValue")


def _adoption_requested(kind: str, payload: Mapping[str, Any]) -> bool:
    """Validate the explicit adoption mode shared by writes and history verification."""
    adopt = payload.get("adopt_no_harm", False)
    if not isinstance(adopt, bool):
        raise AuthorityError("adopt_no_harm must be a boolean")
    if adopt:
        rationale = payload.get("rationale")
        if kind != "promote" or "override_verdict" in payload:
            raise AuthorityError("no-harm adoption requires promote without a verdict override")
        if not isinstance(rationale, str) or not rationale.strip():
            raise AuthorityError("no-harm adoption requires a nonempty evidence-backed rationale")
    return adopt


@dataclass(frozen=True)
class PublicationBatch:
    """Complete checked immutable publication; the proposal body is verbatim.

    A closure/2 release carries its new canonical Tool, rule and parameter-set versions in the
    trailing typed fields; a closure/1 publication leaves them empty.
    """

    prompt_rows: tuple[Mapping[str, Any], ...]
    capsule_rows: tuple[Mapping[str, Any], ...]
    toolkit_rows: tuple[Mapping[str, Any], ...]
    release: Mapping[str, Any]
    proposal: Mapping[str, Any]
    gate: Mapping[str, Any]
    tool_rows: tuple[Mapping[str, Any], ...] = ()
    rule_rows: tuple[Mapping[str, Any], ...] = ()
    parameter_rows: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class PublishResult:
    proposal: str
    release: str
    created: int
    revision: AuthorityRevision
    export_commit: str | None = None


@dataclass(frozen=True)
class DecisionResult:
    decision: str | None
    previous: str
    release: str
    generation: str
    revision: AuthorityRevision
    export_commit: str | None = None


@dataclass(frozen=True)
class VerifyResult:
    ok: bool
    revision: AuthorityRevision
    head: AuthorityHead
    export_fresh: bool
    export_commit: str | None
    violations: tuple[str, ...]
    bound: bool = True


@dataclass(frozen=True)
class ExportResult:
    commit: str
    head_sha256: str
    covered_revision: AuthorityRevision
    live_revision: AuthorityRevision


@dataclass(frozen=True)
class RestoreResult:
    db: Path
    head: AuthorityHead
    covered_revision: AuthorityRevision
    export_anchor: Path
    verified: bool = True
    bound: bool = False


@dataclass(frozen=True)
class SecretInputs:
    """Secret values held only in memory while scanning authoring/export bytes."""

    values: tuple[str, ...] = ()


def secret_inputs() -> SecretInputs:
    """Read configured credential values for scanning without exposing matches.

    Returns:
        Nonempty configured keys, never written into the export.
    """
    values = [value for key, value in os.environ.items() if key in {"DASHSCOPE_API_KEY", "RUBIKSQL_EMBED_API_KEY"} and value]
    token_file = os.environ.get("RUBIKSQL_GATEWAY_COORDINATOR_TOKEN")
    if token_file is not None:
        try:
            token = Path(token_file).expanduser().read_text(encoding="utf-8").strip()
            if not token or len(token) > 4096:
                raise ValueError("invalid private coordinator token")
        except (OSError, ValueError) as error:
            raise AuthorityError("private gateway coordinator token could not be read for secret scanning") from error
        values.append(token)
    configured = os.environ.get("RUBIKSQL_GATEWAY_POOL", "").strip()
    pool = Path(configured).expanduser() if configured else Path("/home/tester/.config/rubiksql-hb/credentials/experiment-gateway-20260928.json")
    if configured and not pool.is_file():
        raise AuthorityError("configured approved credential pool is missing; secret scanning refuses")
    if pool.is_file():
        try:
            from ..experiment_gateway import GatewayPool

            state = os.environ.get("RUBIKSQL_GATEWAY_STATE", "").strip()
            selected = GatewayPool(pool, state_path=Path(state) if state else None).configuration_path
            document = json.loads(selected.read_text(encoding="utf-8"))
        except (OSError, ValueError, RuntimeError) as error:
            raise AuthorityError("approved credential pool could not be read for secret scanning") from error

        def collect(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    if isinstance(item, str) and key.lower() in {"api_key", "apikey", "key", "token"} and item:
                        values.append(item)
                    elif key.lower() == "keys" and isinstance(item, list):
                        values.extend(secret for secret in item if isinstance(secret, str) and secret)
                        for nested in item:
                            if isinstance(nested, Mapping):
                                collect(nested)
                    else:
                        collect(item)
            elif isinstance(value, list):
                for item in value:
                    collect(item)

        collect(document)
    return SecretInputs(tuple(sorted(set(values))))


def scan_secrets(files: Mapping[str, bytes], secrets: SecretInputs) -> None:
    """Refuse secret values or key patterns without printing any matched text."""
    for name, value in files.items():
        text = value.decode("utf-8", errors="replace")
        if _KEY_PATTERN.search(text) or any(secret.encode("utf-8") in value for secret in secrets.values):
            raise AuthorityError(f"secret scan refused export or candidate file {name}")


def semantic_row(value: Any, *, _nested: bool = False) -> Any:
    """Canonical complete semantics, omitting only native timestamps/capture paths.

    Args:
        value: Native row or nested value.

    Returns:
        Detached JSON data retaining status, tags, descriptions and trust fields.
    """
    if callable(value) and hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, Mapping):
        if set(value) == {"__obj_type__", "__obj_data__"} and value["__obj_type__"] == "callable-ref":
            return value["__obj_data__"]
        native = "object_id" in value or all(key in value for key in ("id", "namespace", "restore_layers"))
        captured_source = isinstance(value.get("source"), str) and isinstance(value.get("source_hash"), str)
        result = {key: semantic_row(item, _nested=True) for key, item in value.items()
                  if not (native and key in _EPHEMERAL_FIELDS)
                  and not (not _nested and "object_id" in value and item is None)
                  and not (captured_source and key == "filename" and isinstance(item, str) and item.startswith("/"))}
        if native and "restore_layers" in result:
            provenance = result.get("provenance")
            if isinstance(provenance, dict):
                # Rebuild instead of popping: a callable reference returns its stored data
                # itself, and digests of retained rows must never edit their input.
                result["provenance"] = {key: item for key, item in provenance.items() if key != "captured_at"}
        if "capsule" in result and isinstance(result.get("fingerprints"), dict):
            result["fingerprints"] = {**result["fingerprints"], "capsule": sha256_hex(canon(result["capsule"]))}
        return result
    if isinstance(value, (list, tuple)):
        return [semantic_row(item, _nested=True) for item in value]
    return value


def _revision(events: Sequence[Mapping[str, Any]]) -> AuthorityRevision:
    return AuthorityRevision(len(events), str(events[-1]["object_id"]) if events else None)


def _batch_digest(batch: PublicationBatch) -> str:
    # A typed field enters the digest only when non-empty, so historical prepared batches keep their digest.
    return sha256_hex(canon({"prompt_rows": [semantic_row(row) for row in batch.prompt_rows],
        "capsule_rows": [semantic_row(row) for row in batch.capsule_rows],
        "toolkit_rows": [semantic_row(row) for row in batch.toolkit_rows],
        "release": batch.release, "proposal": batch.proposal, "gate": batch.gate,
        **{key: [semantic_row(row) for row in rows] for key in _TYPED_BATCH if (rows := getattr(batch, key))}}))


def _batch_document(batch: PublicationBatch) -> dict[str, Any]:
    document = _json_value(dataclasses.asdict(batch))
    return {key: value for key, value in document.items() if key not in _TYPED_BATCH or value}


def _schemas() -> Mapping[str, Any]:
    from heavenbase.capsule.runtime import Capsule
    from heavenbase.extensions.prompt.prompt import Prompt
    from heavenbase.toolkit.toolkit import Toolkit
    from .control_schema import GeneralRuleEntity, KnobSetEntity, PromptEventEntity, ProposalEntity, ReleaseEntity, ToolEntity

    return {"sys-prompt": Prompt.schema(), "sys-capsule": Capsule.schema(), "sys-toolkit": Toolkit.schema(),
            "release-entity": ReleaseEntity.schema(), "proposal-entity": ProposalEntity.schema(),
            "prompt-event-entity": PromptEventEntity.schema(), M.TOOL_ENTITY: ToolEntity.schema(),
            M.RULE_ENTITY: GeneralRuleEntity.schema(), M.PARAMETER_ENTITY: KnobSetEntity.schema()}


def _entities(control: Any) -> Mapping[str, Any]:
    from heavenbase.capsule.runtime import Capsule
    from heavenbase.toolkit.toolkit import Toolkit
    from .control_schema import GeneralRuleEntity, KnobSetEntity, PromptEventEntity, ProposalEntity, ReleaseEntity, ToolEntity

    return {"sys-prompt": control.Prompt, "sys-capsule": Capsule, "sys-toolkit": Toolkit,
            "release-entity": ReleaseEntity, "proposal-entity": ProposalEntity, "prompt-event-entity": PromptEventEntity,
            M.TOOL_ENTITY: ToolEntity, M.RULE_ENTITY: GeneralRuleEntity, M.PARAMETER_ENTITY: KnobSetEntity}


def _export_format(schema: Any, *, receipt: bool = False) -> int:
    """Return the export format number of a HEAD or receipt schema.

    Args:
        schema: Recorded HEAD (or, with ``receipt``, export receipt) schema.
        receipt: Classify an export event receipt instead of a HEAD file.

    Returns:
        The key of :data:`EXPORT_FORMATS`.

    Raises:
        ValueError: The schema names no supported export format.
    """
    for number, (head, export, _state, _tables) in EXPORT_FORMATS.items():
        if schema == (export if receipt else head):
            return number
    raise ValueError(f"unsupported authority export {'receipt' if receipt else 'HEAD'} schema {schema!r}")


def _verify_typed_row(entity: str, row: Mapping[str, Any], *, policy: bool = True) -> None:
    """Verify one canonical Tool, rule or parameter-set version by its own identity.

    Args:
        entity: Typed entity id.
        row: Candidate or stored row.
        policy: Also apply the current rule and knob policy; stored history passes False so a later
            code change cannot invalidate a retained version.

    Raises:
        ValueError: A projection row, an unknown format, or content that differs from its identity.
    """
    from ..hb_tool_entities import verify_tool_row

    if not M.is_canonical(entity, row):
        raise M.EntityModelError(f"{entity} {row.get('object_id')!r} is a historical projection row")
    if entity == M.TOOL_ENTITY:
        verify_tool_row(row)
    elif entity == M.RULE_ENTITY:
        M.verify_rule_row(row, policy=policy)
    else:
        M.verify_parameter_row(row, policy=policy)


def _carries_tools(row: Mapping[str, Any]) -> bool:
    if row.get("rules") == M.CLOSURE_V2:
        return bool((row.get("content_refs") or {}).get("tools"))
    return "tools.answering" in row["refs"]


def _git(directory: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(directory), *args], check=True, capture_output=True,
                              text=True, timeout=90).stdout.strip()
    except subprocess.SubprocessError as error:
        raise AuthorityError(f"authority export git operation failed: {args[0]}") from error


def default_export_dir(binding: AuthorityBinding) -> Path:
    """Return the authority-owned export checkout beside its control directory."""
    directory = binding.db_path.parent
    return (directory.parent if directory.name == "control" else directory) / "authority-export"


@contextlib.contextmanager
def authority_lock(path: Path) -> Iterator[None]:
    """Hold the cross-process authority lock, reentrant for writer projection.

    Args:
        path: Canonical authority DB identity used by all process drivers.

    Yields:
        Nothing. Acquire the lab lock only after entering this context.
    """
    from .evolve import store_lock

    key = Path(path).resolve()
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(key, threading.RLock())
    with lock:
        active = getattr(_LOCK_STATE, "active", set())
        if key in active:
            yield
            return
        with store_lock(key):
            _LOCK_STATE.active = active | {key}
            try:
                yield
            finally:
                _LOCK_STATE.active = active


class AuthorityRead:
    """One read-only native SQLite view, held from BEGIN through context exit."""

    def __init__(self, connection: sqlite3.Connection, db: Path, binding: AuthorityBinding | GitAuthorityBinding,
                 export_anchor: Path | None = None) -> None:
        self.connection, self.path, self.binding = connection, db, binding
        self.export_anchor = export_anchor
        self._schemas = _schemas()
        self._view_active = True
        self._writer_owner: Authority | None = None
        self._writer_token: object | None = None
        self.git_binding: GitAuthorityBinding | None = None
        self.authority_commit: str | None = None
        self._git_projection: Any = None
        self._ancillary_files: Mapping[str, bytes] = {}
        self._read_memo: dict[tuple[Any, ...], Any] = {}
        self._read_revision: tuple[int, int, int] | None = None

    def _database_revision(self) -> tuple[int, int, int]:
        # data_version catches commits from HB/other connections; total_changes and
        # schema_version also catch direct writes and DDL on this connection.
        return (int(self.connection.execute("PRAGMA data_version").fetchone()[0]),
                int(self.connection.execute("PRAGMA schema_version").fetchone()[0]),
                self.connection.total_changes)

    def _memo_entry(self, key: tuple[Any, ...], compute: Callable[[], _MemoValue]) -> tuple[_MemoValue, bool]:
        """Return a value and whether it is the retained entry of this revision."""
        # Only native read views may retain decoded values. A writable connection
        # can roll back to a savepoint without changing total_changes; caching its
        # uncommitted rows would hide that rollback. Normal mode=ro readers never
        # perform local writes, even when their owning HB writer changes the DB.
        if self.connection.total_changes or not self.connection.execute("PRAGMA query_only").fetchone()[0]:
            self._read_revision, self._read_memo = None, {}
            return compute(), False
        revision = self._database_revision()
        if revision != self._read_revision:
            self._read_revision, self._read_memo = revision, {}
        if key not in self._read_memo:
            self._read_memo[key] = compute()
        return self._read_memo[key], True

    def _memoized(self, key: tuple[Any, ...], compute: Callable[[], _MemoValue]) -> _MemoValue:
        value, retained = self._memo_entry(key, compute)
        # Readers historically returned newly decoded, mutable JSON. Keep cached
        # integrity inputs private even when callers modify their returned rows.
        return copy.deepcopy(value) if retained else value

    def _retained(self, key: tuple[Any, ...], compute: Callable[[], _MemoValue]) -> _MemoValue:
        """Inspect a memo entry in place; deep-copy whatever leaves this view.

        Digests, serialization and selections read retained rows without first copying
        every row. Only callers that never mutate their input may use this.
        """
        return self._memo_entry(key, compute)[0]

    @property
    def writer_lock_held(self) -> bool:
        """Whether this live view belongs to the current thread's active writer."""
        owner = self._writer_owner
        return bool(self._view_active and owner is not None and owner._writer_active
                    and self._writer_token is owner._writer_token
                    and owner._writer_thread == threading.get_ident()
                    and self.path in getattr(_LOCK_STATE, "active", set()))

    def rows(self, entity: str, *, namespace: str | None = None) -> tuple[Mapping[str, Any], ...]:
        """Decode all rows of an allowlisted entity in this transaction.

        Args:
            entity: Native entity id, never a SQL identifier from the caller.
            namespace: Optional exact native capsule/toolkit namespace filter.

        Returns:
            Decoded rows sorted by object id. A typed entity yields only its canonical versions
            (historical projection rows are not version authority); a missing typed table, or one
            that lacks only columns this typed change added, predates it and yields ``()``.

        Raises:
            AuthorityError: Unknown entity or schema/column mismatch.
            ValueError: A typed row names an unknown format.
        """
        return self._memoized(("rows", entity, namespace), lambda: self._decode_rows(entity, namespace=namespace))

    def _held_rows(self, entity: str) -> tuple[Mapping[str, Any], ...]:
        """Return the retained rows of :meth:`rows`; read them only (see :meth:`_retained`)."""
        return self._retained(("rows", entity, None), lambda: self._decode_rows(entity, namespace=None))

    def _decode_rows(self, entity: str, *, namespace: str | None) -> tuple[Mapping[str, Any], ...]:
        if entity not in TABLES:
            raise AuthorityError(f"unknown authority entity {entity!r}")
        table = TABLES[entity]
        columns = {row[1] for row in self.connection.execute(f'PRAGMA table_info("{table}")')}
        if not columns:
            if entity in {"sys-capsule", "sys-toolkit"} or entity in TYPED_TABLES:
                return ()
            raise AuthorityError(f"authority native table {table} is missing")
        schema = self._schemas[entity]
        expected = set(schema.fields) | {"object_id"}
        if entity in TYPED_TABLES and columns < expected:
            if expected - columns <= (_TYPED_COLUMNS[entity] or set(schema.fields)):
                return ()
        if columns != expected:
            raise AuthorityError(f"authority native schema differs for {entity}: missing {sorted(expected - columns)}, extra {sorted(columns - expected)}")
        if namespace is not None and "namespace" not in columns:
            raise AuthorityError(f"{entity} has no namespace")
        sql = f'SELECT * FROM "{table}"'
        parameters: tuple[str, ...] = ()
        if namespace is not None:
            sql += " WHERE namespace = ?"
            parameters = (namespace,)
        sql += " ORDER BY object_id"
        from heavenbase.utils.serialize import loads_json

        result = []
        for native in self.connection.execute(sql, parameters):
            row = dict(native)
            for name, field in schema.fields.items():
                if row.get(name) is None:
                    continue
                value = row[name]
                if type(field.dtype).identifier in {"json", "array"}:
                    value = loads_json(value)
                row[name] = field.dtype.decode(value)
            if entity not in TYPED_TABLES or M.is_canonical(entity, row):
                result.append(row)
        return tuple(result)

    def typed_index(self) -> dict[str, dict[str, Mapping[str, Any]]]:
        """Return every canonical typed version by entity id and object id."""
        return {entity: {str(row["object_id"]): row for row in self.rows(entity)} for entity in TYPED_TABLES}

    def typed_bundle(self, release: str) -> Mapping[str, Any] | None:
        """Return a closure/2 release's executable ``rubiksql.answering-bundle/3`` document.

        Args:
            release: Exact release id.

        Returns:
            ``hb_tool_entities.typed_bundle(content_refs)``; None for a closure/1 release, whose
            bundle is its ``tools.answering`` carrier (:meth:`answering`).

        Raises:
            AuthorityError: The release is absent or its typed block is malformed.
        """
        from ..hb_tool_entities import typed_bundle

        row = self.row("release-entity", release)
        if row["rules"] != M.CLOSURE_V2:
            return None
        try:
            return typed_bundle(M.check_release_refs(row["content_refs"]))
        except M.EntityModelError as error:
            raise AuthorityError(f"release {release} typed references are invalid: {error}") from error

    def row(self, entity: str, object_id: str) -> Mapping[str, Any]:
        """Read one exact native identity, refusing absence or duplicates."""
        found = [row for row in self._held_rows(entity) if row["object_id"] == object_id]
        if len(found) != 1:
            raise AuthorityError(f"{entity} {object_id!r} is absent from authority")
        return copy.deepcopy(found[0])

    def _held_events(self) -> tuple[Mapping[str, Any], ...]:
        """Return the retained ordered events of :meth:`events`; read them only."""
        return self._retained(("events",), lambda: tuple(sorted(self._held_rows("prompt-event-entity"),
            key=lambda row: (float(row["ts"]), str(row["object_id"])))))

    def events(self, kind: str | None = None) -> tuple[Mapping[str, Any], ...]:
        """Return canonical timestamp/object-id order, optionally filtered by kind."""
        # Select before copying: a kind filter detaches only its own events.
        return copy.deepcopy(tuple(row for row in self._held_events() if kind is None or row["kind"] == kind))

    def _altered_events(self) -> tuple[str, ...]:
        """Ids of events whose content no longer derives their id, hashed once per revision."""
        # Fetch inside the computation: a retained result must describe its own revision.
        return self._retained(("altered_events",), lambda: tuple(
            event["object_id"] for event in self._held_events()
            if event["object_id"] != "e" + sha256_hex(canon({key: value for key, value in event.items() if key != "object_id"}))[:16]))

    def marker(self) -> Mapping[str, Any]:
        """Return the first mark event; a rebind can never replace epoch identity."""
        markers = [row for row in self.events("authority") if row["payload"].get("operation") == "mark"]
        if len(markers) != 1:
            raise AuthorityError("authority requires exactly one initial marker")
        return markers[0]["payload"]

    def epoch(self) -> Mapping[str, Any]:
        """Return the immutable restarted epoch identity."""
        return self.marker()["epoch"]

    def _binding_marker(self) -> Mapping[str, Any]:
        events = self.events("authority")
        if not events:
            raise AuthorityError("authority has no binding marker")
        payload = events[-1]["payload"]
        return payload["new"] if payload.get("operation") == "rebind" else payload

    def _binding_matches(self) -> bool:
        if self._git_projection is not None:
            return self._git_projection.owns_reader(self)
        try:
            marker = self._binding_marker()
            if isinstance(self.binding, GitAuthorityBinding):
                return marker.get("schema") == SCHEMA_GIT_AUTHORITY and marker.get("binding") == B.binding_document(self.binding)
            return all(marker.get(key) == value for key, value in {
                "authority_uuid": self.binding.authority_uuid, "instance_id": self.binding.instance_id,
                "db_path": str(self.binding.db_path)}.items()) and self.path == self.binding.db_path
        except AuthorityError:
            return False

    def _row_digest(self, tables: Mapping[str, str]) -> str:
        return self._memoized(("row_digest", tuple(tables.items())), lambda: sha256_hex(canon({
            entity: [semantic_row(row) for row in self._held_rows(entity)]
            for entity in tables if entity != "prompt-event-entity"})))

    def _complete_version_digest(self, tables: Mapping[str, str]) -> str:
        return self._memoized(("complete_version_digest", tuple(tables.items())), lambda: sha256_hex(_canon_json({
            entity: self._held_rows(entity) for entity in tables if entity != "prompt-event-entity"})))

    def _typed_rows_exist(self) -> bool:
        return any(self._held_rows(entity) for entity in TYPED_TABLES)

    def workspace_rows(self, entity: str) -> tuple[Mapping[str, Any], ...]:
        """Read canonical QA, experiment and conversation evidence without opening HB."""
        from .workspace_state import evidence_rows

        return self._memoized(("workspace_rows", entity), lambda: evidence_rows(self.connection, entity))

    def _held_evidence(self, entity: str) -> tuple[Mapping[str, Any], ...]:
        """Return the retained rows of :meth:`workspace_rows`; read them only."""
        from .workspace_state import evidence_rows

        return self._retained(("workspace_rows", entity), lambda: evidence_rows(self.connection, entity))

    def _evidence_bytes(self, entity: str) -> bytes:
        """Serialize one exported evidence file once per native revision."""
        return self._retained(("evidence_bytes", entity), lambda: _bytes(self._held_evidence(entity)))

    def operational_state(self) -> Mapping[str, Any]:
        """Fingerprint full evidence rows and schemas independently of version events."""
        from .workspace_state import EVIDENCE_TABLES, schema_facts

        # Each digest covers the canonical rows that the exported file holds before its newline.
        def compute() -> Mapping[str, Any]:
            return {entity: (self._evidence_anchor(entity) or {
                "count": len(self._held_evidence(entity)), "rows_digest": sha256_hex(self._evidence_bytes(entity)[:-1]),
                "schema": schema_facts(self.connection, table)}) for entity, table in EVIDENCE_TABLES.items()}
        return self._memoized(("operational_state",), compute)

    def _evidence_anchor(self, entity: str) -> Mapping[str, Any] | None:
        """Reuse evidence semantics only while its complete raw table and DDL agree."""
        from .projection_proof import unchanged

        projection = self._git_projection
        if (projection is not None and projection.owns_reader(self) and projection.proof is not None
                and unchanged(self, projection.proof, entity)):
            return copy.deepcopy(projection.proof["export"]["workspace_entities"][entity])
        return None

    def workspace_state(self, *, export_format: int = EXPORT_FORMAT) -> Mapping[str, Any]:
        """Describe every canonical entity set of a complete export format, including explicit empty sets.

        Args:
            export_format: Complete export format (2: the six legacy version sets, 3–4: all nine) plus
                the seven evidence sets.

        Returns:
            ``{"schema", "entities": {entity: {count, rows_digest, schema}}}``.

        Raises:
            AuthorityError: A version-only format has no workspace state.
        """
        from .workspace_state import schema_facts

        _head, _receipt, state, tables = EXPORT_FORMATS[export_format]
        if state is None:
            raise AuthorityError("a version-only export format has no workspace state")
        def compute() -> Mapping[str, Any]:
            versions = {entity: {"count": len(rows), "rows_digest": sha256_hex(_canon_json(rows)),
                                "schema": schema_facts(self.connection, table)}
                        for entity, table in tables.items() for rows in (self._held_rows(entity),)}
            return {"schema": state, "entities": {**versions, **self.operational_state()}}

        return self._memoized(("workspace_state", export_format), compute)

    def _workspace_schemas_digest(self, tables: Mapping[str, str]) -> str:
        from .workspace_state import EVIDENCE_TABLES, schema_facts

        return self._memoized(("workspace_schemas_digest", tuple(tables.items())), lambda: sha256_hex(canon({
            entity: schema_facts(self.connection, table) for entity, table in {**tables, **EVIDENCE_TABLES}.items()})))

    def workspace_files(self, *, tools_release: str | None = None,
                        export_format: int = EXPORT_FORMAT) -> Mapping[str, bytes]:
        """Map complete recovery state of one export format; answering and history use store_files instead."""
        from .workspace_state import EVIDENCE_TABLES, encode_evidence_rows

        state = self.workspace_state(export_format=export_format)
        if any(not value["schema"]["present"] for value in state["entities"].values()):
            raise AuthorityError("complete workspace export requires every canonical entity table")
        files = dict(self.store_files(tools_release=tools_release, tables=EXPORT_FORMATS[export_format][3]))
        files["workspace/STATE.json"] = _bytes(state)
        try:
            for entity in EVIDENCE_TABLES:
                projection = self._git_projection
                if (export_format == 4 and self._evidence_anchor(entity) is not None
                        and projection._proof_files):
                    files.update({name: raw for name, raw in projection._proof_files.items()
                                  if name.startswith(f"workspace/{entity}/")})
                else:
                    files.update(encode_evidence_rows(entity, self._held_evidence(entity), row_files=export_format == 4))
        except ValueError as error:
            raise AuthorityError(f"complete workspace evidence export refused: {error}") from error
        files.update(self._ancillary_files)
        return files

    def tool_tiers(self) -> Mapping[str, str]:
        """Read executable tier policy from this exact pinned authority commit.

        Returns:
            Module -> hb-executed or host-verified. Historical ECS readers retain
            the adjacent policy file; Git readers require their covered metadata.

        Raises:
            AuthorityError: Missing pinned metadata or malformed policy.
        """
        if self.git_binding is not None:
            raw = self._ancillary_files.get("metadata/tool-tiers.json")
            if raw is None:
                raise AuthorityError("Git authority has no pinned tool tier policy")
            tiers = json.loads(raw)
        else:
            directory = self.path.parent
            path = (directory.parent if directory.name == "control" else directory) / "tool-tiers.json"
            tiers = json.loads(path.read_bytes()) if path.is_file() else {}
        if (not isinstance(tiers, dict) or any(not isinstance(name, str) or not name
                or tier not in {"hb-executed", "host-verified"} for name, tier in tiers.items())):
            raise AuthorityError("authority tool tier configuration is invalid")
        return tiers

    def _export_state(self) -> tuple[bool, str | None]:
        # Inspect retained events in place; only the returned commit leaves this view.
        events = self._held_events()
        nonexport = [row["object_id"] for row in events if row["kind"] != "export"]
        exports = [row for row in events if row["kind"] == "export"]
        if not exports:
            return False, None
        receipt = exports[-1]["payload"]
        certified = exports[-1]["object_id"]
        before = events[:next(index for index, event in enumerate(events) if event["object_id"] == certified)]
        intact_events = not self._altered_events()
        try:
            # Freshness is computed in the receipt's own format: a legacy receipt never covered
            # typed sets, so it stays fresh only while no canonical typed row exists.
            _head, _schema, state, tables = EXPORT_FORMATS[_export_format(receipt.get("schema"), receipt=True)]
            operational = self.operational_state()
            covered_workspace = (receipt.get("workspace_digest") == sha256_hex(canon(operational))
                                 and receipt.get("workspace_schemas_digest") == self._workspace_schemas_digest(tables)
                                 and receipt.get("version_rows_digest") == self._complete_version_digest(tables) if state is not None
                                 else not any(value["count"] for value in operational.values()))
            covered_workspace = covered_workspace and (set(TYPED_TABLES) <= set(tables) or not self._typed_rows_exist())
            version_digest = self._row_digest(tables)
        except (AuthorityError, ValueError, KeyError, TypeError, sqlite3.Error):
            return False, copy.deepcopy(receipt.get("export_commit"))
        fresh = (intact_events and covered_workspace and receipt.get("covered_non_export_ids") == nonexport
                 and receipt.get("rows_digest") == version_digest
                 and receipt.get("covered_revision") == dataclasses.asdict(_revision(before))
                 and all(isinstance(receipt.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", receipt[key])
                         for key in ("head_sha256", "file_map_digest"))
                 and isinstance(receipt.get("export_commit"), str) and re.fullmatch(r"[0-9a-f]{40}", receipt["export_commit"]) is not None)
        return fresh, copy.deepcopy(receipt.get("export_commit"))

    def covered_head(self, files: Mapping[str, bytes], *, export_format: int) -> Mapping[str, Any]:
        """Derive every covered HEAD field of one export format from this exact native read view.

        Args:
            files: The exact export file map the HEAD certifies.
            export_format: 1 (version-only), 2 (complete, legacy version sets) or 3 (complete, all sets).

        Returns:
            The HEAD document; digests cover exactly that format's version sets.
        """
        schema, _receipt, state, tables = EXPORT_FORMATS[export_format]
        current = self.head()
        marker = self._binding_marker()
        binding = (marker["binding"] if marker.get("schema") == SCHEMA_GIT_AUTHORITY else
                   {"schema": B.SCHEMA_BINDING, **{key: marker[key] for key in ("authority_uuid", "instance_id", "db_path")}})
        file_map = {name: sha256_hex(value) for name, value in sorted(files.items())}
        return {"schema": schema, "binding": binding, "epoch": self.epoch(),
                "release": current.release, "generation": current.generation, "decision": current.decision,
                "covered_revision": dataclasses.asdict(current.revision), "covered_event_ids": [row["object_id"] for row in self._held_events()],
                "covered_non_export_ids": [row["object_id"] for row in self._held_events() if row["kind"] != "export"],
                "rows_digest": self._row_digest(tables), "files": file_map, "file_map_digest": sha256_hex(canon(file_map)),
                **({"workspace_digest": sha256_hex(canon(self.operational_state())),
                    "workspace_schemas_digest": self._workspace_schemas_digest(tables),
                    "version_rows_digest": self._complete_version_digest(tables)} if state is not None else {})}

    def head(self) -> AuthorityHead:
        """Derive the current selection from verified state and pointer events."""
        # Retained events are only inspected; the head holds strings and counts.
        events = self._held_events()
        if any(event["kind"] == "decision" and event["payload"].get("kind") == "refresh-tools"
               or event["kind"] == "publish" and event["payload"].get("body", {}).get("origin") == "refresh-tools"
               for event in events):
            raise AuthorityError("obsolete refresh history requires the fenced retire-refresh migration")
        try:
            root = str(self.epoch()["release"])
        except AuthorityError:
            roots = self.rows("release-entity")
            if len(roots) != 1:
                raise AuthorityError("unmarked authority must contain exactly one seeded root")
            root = str(roots[0]["object_id"])
        state = self.current_state()
        moves = [event for event in events if event["kind"] == "decision"
                 and event["payload"].get("kind") in POINTER_KINDS
                 or state is not None and event["object_id"] == state["id"]]
        latest = moves[-1] if moves else None
        release = str(latest["payload"]["release"] if latest["kind"] == "register" else latest["payload"]["to"]) if latest else root
        fresh, commit = self._export_state()
        if self._git_projection is not None:
            fresh = self._git_projection.clean(self)
            commit = self.authority_commit
        return AuthorityHead(release, release, str(latest["object_id"]) if latest else None,
                             _revision(events), fresh, commit, self.authority_commit)

    def current_state(self) -> Mapping[str, Any] | None:
        """Read the verified standalone admission created by the fenced cleanup."""
        from .authority_state import validate_current_state
        from .fresh_epoch import validate_admission

        def compute() -> Mapping[str, Any] | None:
            event = validate_admission(self) or validate_current_state(self)
            return {**copy.deepcopy(event["payload"]), "id": event["object_id"]} if event is not None else None

        # Both validators read only this native snapshot; a failure is never retained.
        return self._memoized(("current_state",), compute)

    def source_admission(self, release: str) -> Mapping[str, Any] | None:
        """Return the exact detached source proof for the adopted release, if any."""
        from copy import deepcopy

        state = self.current_state()
        return deepcopy(state["source_admission"]) if state is not None and state["release"] == release else None

    def resolve(self, pin: str) -> str:
        """Resolve baseline, an exact release or an exact published proposal."""
        if pin == "baseline":
            return self.head().release
        if any(row["object_id"] == pin for row in self.rows("release-entity")):
            return pin
        return str(self.proposal(pin)["release"])

    def proposal(self, pid: str) -> Mapping[str, Any]:
        """Read the complete verbatim body from its publication event."""
        from .store import proposal_id

        self.row("proposal-entity", pid)
        events = [event for event in self._held_events() if event["kind"] == "publish" and event["subject"] == pid
                  and isinstance(event["payload"].get("body"), Mapping)]
        if len(events) != 1:
            raise AuthorityError(f"proposal {pid} has no unique complete publication event")
        body = copy.deepcopy(events[0]["payload"]["body"])
        if body.get("id") != pid or proposal_id(body) != pid:
            raise AuthorityError(f"proposal {pid} body integrity differs")
        return body

    def pending_publications(self) -> Mapping[str, Mapping[str, Any]]:
        """Return durable prepared batches lacking their completion event."""
        prepared: dict[str, Mapping[str, Any]] = {}
        publications = [event for event in self._held_events() if event["kind"] == "publish"]
        complete = {event["subject"] for event in publications if isinstance(event["payload"].get("body"), Mapping)}
        for event in publications:
            if event["payload"].get("phase") == "prepared" and event["subject"] not in complete:
                if event["subject"] in prepared:
                    raise AuthorityError("publication has multiple prepared batches")
                prepared[str(event["subject"])] = event["payload"]
        return copy.deepcopy(prepared)

    def pending_tool_registrations(self) -> Mapping[str, Mapping[str, Any]]:
        """Return verified interrupted catalog/deployment preparations.

        Args:
            None.

        Returns:
            Mapping[str, Mapping[str, Any]]: Pending events by registration id.
        """
        from ..tools_catalog import catalog_history

        return catalog_history(self, allow_pending=True)[2]

    def tool_publication(self, publication_id: str) -> Mapping[str, Any]:
        """Read an exact completed, source-verified catalog publication.

        Args:
            publication_id (str): Immutable publication identity.

        Returns:
            Mapping[str, Any]: Detached publication and exact register events.

        Raises:
            AuthorityError: The publication is absent, incomplete, or altered.
        """
        from copy import deepcopy
        from ..tools_catalog import catalog_history

        publications, _deployments, _pending = catalog_history(self)
        if publication_id not in publications:
            raise AuthorityError(f"no completed tool publication {publication_id}")
        return deepcopy(publications[publication_id])

    def tool_deployment(self, consumer: str, deployment_id: str | None = None) -> Mapping[str, Any]:
        """Read a consumer's latest or exact historical completed deployment.

        Args:
            consumer (str): ``assistant`` or ``meta``.
            deployment_id (str | None): Exact pin; omitted only for startup resolution.

        Returns:
            Mapping[str, Any]: Detached deployment and exact register events.

        Raises:
            AuthorityError: The consumer or completed pin is absent or altered.
        """
        from copy import deepcopy
        from ..tools_catalog import CONSUMERS, catalog_history

        if consumer not in CONSUMERS:
            raise AuthorityError("independent tools deploy only to assistant or meta")
        _publications, deployments, _pending = catalog_history(self)
        matches = [record for record in deployments.values() if record["consumer"] == consumer
                   and (deployment_id is None or record["id"] == deployment_id)]
        if not matches:
            raise AuthorityError(f"no completed tool deployment for {consumer}")
        return deepcopy(matches[-1])

    def resolve_tools(self, deployment_id: str) -> Mapping[str, Any]:
        """Resolve an exact consumer deployment as detached inert native rows.

        Args:
            deployment_id (str): Immutable deployment identity.

        Returns:
            Mapping[str, Any]: Deployment, publication, Tool and Capsule object-id maps.

        Raises:
            AuthorityError: Registration, source, gate or selected-row integrity differs.
        """
        from copy import deepcopy
        from ..tools_catalog import catalog_history, validate_resolved_tools

        publications, deployments, _pending = catalog_history(self)
        if deployment_id not in deployments:
            raise AuthorityError(f"no completed tool deployment {deployment_id}")
        deployment = deployments[deployment_id]
        publication = publications[deployment["publication_id"]]
        content = publication["content"]
        capsule_ids = {reference["capsule_id"] for reference in content["loader"]["modules"]}
        capsule_ids.add(content["loader"]["serializer"]["capsule_id"])
        resolved = {"deployment": deepcopy(deployment), "publication": deepcopy(publication),
                    "tool_rows": {reference["object_id"]: deepcopy(self.row(M.TOOL_ENTITY, reference["object_id"]))
                                  for reference in content["tools"]},
                    "capsule_rows": {identifier: deepcopy(self.row("sys-capsule", identifier)) for identifier in capsule_ids}}
        validate_resolved_tools(resolved)
        return resolved

    def store_files(self, *, tools_release: str | None = None,
                    tables: Mapping[str, str] = TABLES) -> Mapping[str, bytes]:
        """Map the complete epoch to deterministic store files plus native rows.

        Args:
            tools_release: Release whose executable tool closure is materialized.
            tables: Closed version scope: ``TABLES`` (current) or ``LEGACY_TABLES`` (historical
                export formats 1 and 2).

        Returns:
            Relative path to exact UTF-8 bytes. No file is written here.
        """
        selected = self.resolve(tools_release) if tools_release is not None else None
        tool_key = None
        if selected is not None:
            release = self.row("release-entity", selected)
            tool_key = (("typed", canon(release["content_refs"])) if release["rules"] == M.CLOSURE_V2
                        else ("legacy", release["refs"].get("tools.answering")))
        # Projecting the same immutable view and execution bytes is independent
        # of which sibling prompt release selected them. Native rows stay private;
        # callers receive their own mapping of immutable byte strings.
        files = self._retained(("store_files", tuple(sorted(tables)), tool_key), lambda: project_version_files(
            {entity: self._held_rows(entity) for entity in tables}, tools_release=selected))
        return dict(files)

    def carrier_text(self, release: str, name: str) -> str:
        """Return the exact text of one prompt carrier a closure/1 release references.

        Args:
            release: Exact release id.
            name: ``tools.answering``, ``rules.general`` or ``config.knobs``.

        Returns:
            The carrier row's ``manifest.meta.text``.

        Raises:
            AuthorityError: The release does not reference exactly one such row.
        """
        key = self.row("release-entity", release)["refs"].get(name)
        rows = [row for row in self.rows("sys-prompt") if row["name"] == name and row["manifest"]["meta"]["content_key"] == key]
        if key is None or len(rows) != 1:
            raise AuthorityError(f"release {release} has no exact {name} row")
        return rows[0]["manifest"]["meta"]["text"]

    def answering(self, release: str) -> Mapping[str, Any]:
        """Decode the exact tools.answering data row of a closure/1 release (closure/2: :meth:`typed_bundle`)."""
        if self.row("release-entity", release)["rules"] == M.CLOSURE_V2:
            raise AuthorityError(f"release {release} uses typed tool references; read its typed bundle")
        return json.loads(self.carrier_text(release, "tools.answering"))

    def verify(self, *, require_export: bool = True, export_anchor: Path | None = None) -> VerifyResult:
        """Verify immutable rows, epoch, decisions and exact export coverage.

        Args:
            require_export: Require a fresh certifying export or explicit anchor.
            export_anchor: Verified export checkout for a restored unbound copy.

        Returns:
            Structured violations; no authority state is changed.
        """
        anchor = export_anchor if export_anchor is not None else self.export_anchor
        projection = self._git_projection
        if (projection is not None and projection.owns_reader(self) and projection.proof is not None
                and anchor is not None and Path(anchor).resolve() == projection.export.resolve()):
            from .projection_proof import raw_state

            if (raw_state(self) == projection.proof["raw"]
                    and {key: value for key, value in projection.proof["export"].items()
                         if key != "workspace_entities"} == projection.head):
                # Bootstrap already checked remote ancestry, actual export bytes,
                # runtime identity and the closed database hash. Recheck the anchor
                # so an operation-local mutation cannot inherit that proof.
                try:
                    head, _files, commit = projection.read_export(anchor)
                except (AuthorityError, ValueError, KeyError, TypeError) as error:
                    return self._verify(require_export=require_export, export_anchor=export_anchor, _export_error=error)
                if head == projection.head and commit == self.authority_commit:
                    current = self.head()
                    return VerifyResult(True, current.revision, current, True, commit, (), self._binding_matches())
        exported = None
        if anchor is not None:
            try:
                exported = (self._git_projection.read_export(anchor) if self._git_projection is not None
                            and self._git_projection.owns_reader(self) else _read_export(anchor))
            except (AuthorityError, ValueError, KeyError, TypeError) as error:
                # Preserve ordinary structured export violations. A prior valid
                # snapshot never suppresses a changed or unavailable anchor.
                return self._verify(require_export=require_export, export_anchor=export_anchor, _export_error=error)
        identity = None
        if exported is not None:
            assert anchor is not None
            identity = (str(Path(anchor).resolve()), exported[2], sha256_hex(canon(exported[0])))
        key = ("verify", require_export, identity, canon(B.binding_document(self.binding)),
               self._binding_matches(), self.authority_commit, id(self._git_projection),
               tuple(sorted(self._ancillary_files.items())),
               None if self._git_projection is None else canon(self._git_projection.head))

        def compute() -> VerifyResult:
            projection = self._git_projection
            if exported is not None and projection is not None and projection.owns_reader(self):
                previous = projection._verified_reads.get(key)
                if previous is None and not require_export:
                    # Stored proofs are OK, so their anchored state is fresh: requiring the
                    # export added no problem, and the same proof answers this request.
                    previous = projection._verified_reads.get(("verify", True, *key[2:]))
                if previous is not None:
                    try:
                        number = _export_format(exported[0]["schema"])
                        current = self.head()
                        files = self.workspace_files(tools_release=current.release, export_format=number)
                        if (files == exported[1]
                                and self.covered_head(files, export_format=number) == exported[0]):
                            # A new SQLite view or HB auxiliary repair may change
                            # the DB revision. Reuse its history proof only after
                            # complete canonical rows AND schema bytes agree.
                            return copy.deepcopy(previous)
                    except (AuthorityError, ValueError, KeyError, TypeError):
                        pass  # Ordinary verification reports structured damage.
            result = self._verify(require_export=require_export, export_anchor=export_anchor, _exported=exported)
            if result.ok and exported is not None and projection is not None and projection.owns_reader(self):
                projection._verified_reads[key] = copy.deepcopy(result)
            return result

        return self._memoized(key, compute)

    def _verify(self, *, require_export: bool, export_anchor: Path | None,
                _exported: tuple[Mapping[str, Any], Mapping[str, bytes], str] | None = None,
                _export_error: Exception | None = None) -> VerifyResult:
        """Replay history once per unchanged native snapshot and exact export identity."""
        from .integrity import check_prompt_row
        from .store import release_document_id
        from .projection_proof import verified_row

        projection = self._git_projection
        proof = projection.proof if projection is not None and projection.owns_reader(self) else None

        problems: list[str] = []
        head = self.head()
        bound = self._binding_matches()
        anchor = export_anchor if export_anchor is not None else self.export_anchor
        try:
            epoch = self.epoch()
            if not bound and anchor is None:
                problems.append("package binding and authority marker differ")
            problems.extend(f"event {object_id} has altered content" for object_id in self._altered_events())
            initial = self.marker()
            fresh_epoch = initial.get("schema") == SCHEMA_GIT_AUTHORITY
            identity = (initial["binding"] if fresh_epoch else
                        {"schema": B.SCHEMA_BINDING, **{key: initial.get(key) for key in ("authority_uuid", "instance_id", "db_path")}})
            for event in self.events("authority"):
                payload = event["payload"]
                if payload.get("schema") not in {SCHEMA_AUTHORITY, SCHEMA_GIT_AUTHORITY} or not str(payload.get("user_instruction", "")).strip():
                    problems.append(f"authority event {event['object_id']} lacks schema or authorization evidence")
                if payload.get("schema") == SCHEMA_GIT_AUTHORITY:
                    if (payload.get("operation") != "mark" or payload.get("binding") != identity
                            or not isinstance(self.binding, GitAuthorityBinding) or identity != B.binding_document(self.binding)):
                        problems.append(f"Git authority marker {event['object_id']} differs from its exact binding")
                    continue
                if payload.get("operation") == "rebind":
                    new = payload.get("new") or {}
                    if fresh_epoch:
                        # A fresh Git epoch keeps its marker's binding: no Git read of
                        # its ref verifies a rebind, so a writer must never publish one.
                        problems.append(f"authority rebind {event['object_id']} follows a fresh Git epoch's marker")
                    elif (payload.get("prior") != identity or payload.get("epoch") != epoch
                            or new.get("authority_uuid") != identity["authority_uuid"] or new.get("schema") != B.SCHEMA_BINDING
                            or not isinstance(payload.get("export_commit"), str) or re.fullmatch(r"[0-9a-f]{40}", payload["export_commit"]) is None
                            or not isinstance(payload.get("head_sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", payload["head_sha256"]) is None):
                        problems.append(f"authority rebind {event['object_id']} differs from its prior binding or export evidence")
                    identity = new
                elif payload.get("operation") != "mark":
                    problems.append(f"authority event {event['object_id']} has unknown operation")
            prompts = {}
            for row in self.rows("sys-prompt"):
                if not verified_row(proof, "sys-prompt", row):
                    check_prompt_row(row)
                prompts[(row["name"], row["manifest"]["meta"]["content_key"])] = row
            from ..hb_tool_entities import row_digest

            for entity in ("sys-capsule", "sys-toolkit"):
                for row in self.rows(entity):
                    if not verified_row(proof, entity, row):
                        row_digest(row, capsule=entity == "sys-capsule")
            # Stored history is verified for integrity only (policy=False), like the closure/1 carriers.
            typed_index = self.typed_index()
            for entity, held in typed_index.items():
                for object_id, row in held.items():
                    try:
                        if not verified_row(proof, entity, row):
                            _verify_typed_row(entity, row, policy=False)
                    except ValueError as error:
                        problems.append(f"{entity} {object_id} is not an exact canonical version: {error}")
            from .toolkit_retirement import verify_history as verify_toolkit_history
            capsules = {item["object_id"]: item for item in self.rows("sys-capsule")}
            toolkits = {item["object_id"]: item for item in self.rows("sys-toolkit")}
            verify_toolkit_history(self.rows("release-entity"), self.rows("sys-prompt"), capsules, toolkits, self.events())
            for row in self.rows("release-entity"):
                typed = row["rules"] == M.CLOSURE_V2
                try:
                    identical = release_document_id(row, epoch=epoch["epoch"]) == row["object_id"]
                except ValueError:
                    identical = False
                if not identical:
                    problems.append(f"release {row['object_id']} content identity differs")
                if any((name, key) not in prompts for name, key in row["refs"].items()):
                    problems.append(f"release {row['object_id']} closure is incomplete")
                if typed:
                    try:
                        M.check_release_document(row)
                        M.resolve_closure(row["content_refs"], typed_index, policy=False,
                                          capsule_rows={item["object_id"]: item for item in self.rows("sys-capsule")})
                    except ValueError as error:
                        problems.append(f"release {row['object_id']} typed closure is invalid: {error}")
                elif row.get("content_refs") is not None:
                    problems.append(f"release {row['object_id']} carries typed references outside closure/2")
                elif "tools.answering" in row["refs"]:
                    from ..hb_tool_entities import validate_answering

                    payload = self.answering(str(row["object_id"]))
                    if payload.get("schema") != "rubiksql.answering-bundle/1":
                        validate_answering(payload, capsule_rows=capsules)
            for row in self.rows("proposal-entity"):
                try:
                    body = self.proposal(str(row["object_id"]))
                except AuthorityError as error:
                    problems.append(str(error))
                    continue
                expected = _proposal_row(body, row.get("gate") or {})
                if semantic_row(row) != semantic_row(expected):
                    problems.append(f"proposal {row['object_id']} native row differs from complete publication")
            from .store import PromptStoreError, publication_ancestry

            origins = {row["object_id"]: {} for row in self.rows("release-entity")}
            state = self.current_state()
            roots = {epoch["release"]} | ({state["release"]} if state is not None else set())
            publications = {}
            for event in self.events("publish"):
                body = event["payload"].get("body")
                if not isinstance(body, Mapping):
                    continue
                publications[body["id"]] = body
                rid = body["release"]
                if rid in origins and rid not in roots and not origins[rid]:
                    origins[rid] = {"proposal": body["id"], "parents": body["parents"]}
            try:
                publication_ancestry(origins, publications)
            except PromptStoreError as error:
                problems.append(str(error))
            from ..tools_catalog import catalog_history

            try:
                catalog_history(self)
            except (AuthorityError, KeyError, TypeError, ValueError) as error:
                problems.append(str(error))
            from .workspace_state import EVIDENCE_TABLES
            from .workspace_migration import pending_migration, validate_evidence

            try:
                # Validation only reads its rows; it never needs a private copy of all evidence.
                if not all(self._evidence_anchor(entity) is not None for entity in EVIDENCE_TABLES):
                    validate_evidence({entity: self._held_evidence(entity) for entity in EVIDENCE_TABLES})
            except (AuthorityError, ValueError, TypeError, KeyError) as error:
                problems.append(f"canonical evidence invalid: {error}")
            try:
                if pending_migration(self):
                    problems.append("lab migration is incomplete")
            except AuthorityError as error:
                problems.append(f"lab migration invalid: {error}")
            for pid, prepared in self.pending_publications().items():
                batch = _prepared_batch(prepared)
                if batch.proposal.get("id") != pid or prepared.get("batch_digest") != _batch_digest(batch):
                    problems.append(f"prepared publication {pid} has altered batch content")
                for entity, candidates in prepared.get("candidates", {}).items():
                    if entity not in TABLES or entity == "prompt-event-entity":
                        problems.append(f"prepared publication {pid} has unsupported entity")
                        continue
                    held = {row["object_id"]: row for row in self.rows(entity)}
                    for candidate in candidates:
                        if candidate["object_id"] in held and semantic_row(held[candidate["object_id"]]) != semantic_row(candidate):
                            problems.append(f"prepared publication {pid} native candidate differs")
            state = self.current_state()
            previous, current = None, str(epoch["release"])
            recorded_outcomes: dict[str, Mapping[str, Any]] = {}
            acted_outcomes: set[str] = set()
            for event in self.events():
                if state is not None and event["object_id"] == state["id"]:
                    previous, current = str(event["object_id"]), str(state["release"])
                    continue
                if event["kind"] == "outcome" and event["payload"].get("target") is not None:
                    from .evolve import SCHEMA_OUTCOME, verify_outcome

                    document = event["payload"]
                    anchor_ok = (document.get("baseline_decision") == previous and previous is not None
                                 if "baseline_decision" in document else
                                 state is not None and document.get("authority_state") == state["id"]
                                 and document.get("target") == state["release"])
                    if (not anchor_ok or document.get("schema") != SCHEMA_OUTCOME
                            or document.get("target") != current
                            or event["subject"] != current or document.get("generation") != current
                            ):
                        problems.append(f"outcome {event['object_id']} has no exact adopted current-state target")
                    problems.extend(f"outcome {event['object_id']}: {problem}" for problem in verify_outcome(document))
                if event["kind"] == "outcome":
                    recorded_outcomes[str(event["payload"].get("id"))] = event
                if event["kind"] != "decision":
                    continue
                payload = event["payload"]
                if payload.get("schema") != "rubiksql.decision/1" or payload.get("kind") not in DECISION_KINDS or not str(payload.get("reason", "")).strip():
                    problems.append(f"decision {event['object_id']} has invalid kind, schema or reason")
                if payload.get("previous_decision") != previous:
                    problems.append(f"decision {event['object_id']} previous pointer differs")
                try:
                    adopt = _adoption_requested(str(payload.get("kind")), payload)
                    chosen_event = recorded_outcomes.get(str(payload.get("outcome")))
                    chosen = chosen_event["payload"] if chosen_event is not None else {}
                    if adopt:
                        from .evolve import plan_view, verify_outcome

                        proposal = self.proposal(str(payload["target"]))
                        if (chosen_event is None or chosen_event["subject"] != payload["target"]
                                or chosen.get("proposal") != payload["target"]
                                or chosen.get("release") != payload.get("to")
                                or chosen.get("generation") != current
                                or chosen.get("verdict") != "adoptable"
                                or chosen.get("id") in acted_outcomes
                                or verify_outcome(chosen, plan=plan_view(proposal))):
                            raise AuthorityError("adoption needs an unused reproducible adoptable outcome for this pointer move")
                    elif payload.get("kind") == "promote" and chosen.get("verdict") == "adoptable" and not payload.get("override_verdict"):
                        raise AuthorityError("an adoptable outcome requires explicit no-harm adoption")
                except (AuthorityError, KeyError, TypeError, ValueError) as error:
                    problems.append(f"decision {event['object_id']} has invalid adoption: {error}")
                if payload.get("kind") == "promote" and isinstance(payload.get("outcome"), str):
                    acted_outcomes.add(payload["outcome"])
                if payload.get("kind") in POINTER_KINDS:
                    if payload.get("from") != current:
                        problems.append(f"decision {event['object_id']} source pointer differs")
                    if previous is None and payload.get("kind") != "cutover":
                        problems.append("first pointer decision is not cutover")
                    previous, current = str(event["object_id"]), str(payload["to"])
                if payload.get("generation") != current:
                    problems.append(f"decision {event['object_id']} generation differs from its pointer")
                if payload.get("kind") == M.MIGRATION_KIND:
                    try:
                        proposal = self.proposal(str(payload["target"]))
                        if (proposal["parents"] != [{"release": payload.get("from"), "role": "base"}]
                                or proposal["release"] != payload.get("to")):
                            raise AuthorityError("entity-model decision differs from its proposal parent or release")
                        check = _check_entity_migration(self, proposal, self.row("release-entity", str(payload["to"])),
                                                        str(payload.get("user_instruction", "")), policy=False)
                        prior = [item for item in self.events() if (float(item["ts"]), str(item["object_id"]))
                                 < (float(event["ts"]), str(event["object_id"]))]
                        _prebaseline_history(prior, operation="entity-model migration", typed=False)
                        _check_entity_gate(self, str(payload["target"]), check, payload.get("deployed_commit"))
                    except (AuthorityError, KeyError, TypeError, ValueError) as error:
                        problems.append(f"decision {event['object_id']} has invalid entity-model migration: {error}")
            fresh, commit = self._export_state()
            if anchor is not None:
                if _export_error is not None:
                    raise AuthorityError(str(_export_error))
                export_head, exported_files, commit = _exported if _exported is not None else _read_export(anchor)
                number = _export_format(export_head["schema"])
                _schema, _receipt, state, tables = EXPORT_FORMATS[number]
                if not set(TYPED_TABLES) <= set(tables) and self._typed_rows_exist():
                    problems.append("historical export format cannot cover canonical typed versions")
                else:
                    native_files = (self.workspace_files(tools_release=head.release, export_format=number) if state is not None
                                    else self.store_files(tools_release=head.release, tables=tables))
                    if state is None and any(value["count"] for value in self.operational_state().values()):
                        problems.append("historical version-only export cannot cover canonical evidence")
                    if export_head != self.covered_head(native_files, export_format=number):
                        problems.append("restored complete HEAD differs from export anchor")
                    if exported_files != native_files:
                        problems.append("restored canonical mapped state differs from export anchor")
                fresh = not problems
            if require_export and not fresh:
                problems.append("authority has unexported or altered rows/events")
        except (AuthorityError, ValueError, KeyError, TypeError) as error:
            problems.append(str(error))
            fresh, commit = False, None
        return VerifyResult(not problems, head.revision, head, fresh, commit, tuple(problems), bound)

    def status(self) -> Mapping[str, Any]:
        """Return binding, epoch, current generation quota and export state."""
        head = self.head()
        epoch = self.epoch() if self.events("authority") else None
        return {"binding": B.binding_document(self.git_binding or self.binding), "epoch": epoch, "head": dataclasses.asdict(head),
                "revision": dataclasses.asdict(head.revision), "attempts": self.attempts(),
                "export": {"fresh": head.exported, "commit": head.export_commit}}

    def attempts(self) -> Mapping[str, int]:
        """Count committed child reservations and unmatched historical attachments once."""
        from .paid_launch import counted_attempts
        from .evolve import MAX_ATTEMPTS_PER_GENERATION, PROMOTE_QUESTION_SETS

        return {"used": len(counted_attempts(self.events(), self.head().generation,
                                           question_set=next(iter(PROMOTE_QUESTION_SETS)))),
                "cap": MAX_ATTEMPTS_PER_GENERATION}

    def gate(self, target: str, *, parent: str | None = None, corpus: Any = None, root_run: Any = None) -> Any:
        """Run the existing offline scientific gate on a verified authority view."""
        from . import views
        from .gate import run_gate

        with views.history_store(self, require_export=not self.writer_lock_held) as store:
            release = views.load(store, target, allow_retired=True)
            proposal = self.proposal(target) if target.startswith("p") else None
            if parent is None and proposal is not None:
                parent = next(item["release"] for item in proposal["parents"] if item["role"] == "base")
            base = views.load(store, parent, allow_retired=True) if parent else None
            return run_gate(release, parent=base, store=store, proposal=proposal, corpus=corpus, root_run=root_run)


def _proposal_row(body: Mapping[str, Any], gate: Mapping[str, Any]) -> dict[str, Any]:
    return {"object_id": body["id"], **{key: value for key, value in body.items() if key not in {"id", "schema", "plan", "revert_of", "user_instruction"}},
            "analysis_plan": body.get("plan"), "gate": dict(gate)}


def _prepared_batch(payload: Mapping[str, Any]) -> PublicationBatch:
    batch = payload["batch"]
    return PublicationBatch(tuple(batch["prompt_rows"]), tuple(batch["capsule_rows"]), tuple(batch["toolkit_rows"]),
                            batch["release"], batch["proposal"], batch["gate"],
                            *(tuple(batch.get(key, ())) for key in _TYPED_BATCH))


def _recoverable_violations(reader: AuthorityRead) -> tuple[str, ...]:
    """Permit only a prepared batch's known row-before-completion interruption."""
    allowed = {f"proposal {pid} has no unique complete publication event" for pid in reader.pending_publications()}
    allowed.update(f"tool registration {identifier} is incomplete" for identifier in reader.pending_tool_registrations())
    return tuple(problem for problem in reader.verify(require_export=False).violations if problem not in allowed)


def _prebaseline_history(events: Sequence[Mapping[str, Any]], *, operation: str,
                         typed: bool | None = None) -> None:
    """Require cutover and at most one unpaid representation migration."""
    decisions = [event["payload"].get("kind") for event in events if event["kind"] == "decision"]
    if (not decisions or decisions[0] != "cutover"
            or any(kind not in {M.MIGRATION_KIND} for kind in decisions[1:])
            or decisions.count(M.MIGRATION_KIND) > 1
            or (typed is not None and (M.MIGRATION_KIND in decisions) != typed)):
        raise AuthorityError(f"{operation} is available only after cutover, with at most one entity-model migration, "
                             "and before measured evolution or other lifecycle decisions")
    if any(event["kind"] in {"eval-reserve", "eval-claim", "eval-attach", "outcome"} for event in events):
        raise AuthorityError(f"{operation} refuses any measured attempt or outcome in the epoch")


def _batch_typed_index(reader: AuthorityRead, batch: PublicationBatch) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Merge a batch's validated canonical typed rows over the authority's own canonical rows.

    Args:
        reader: Authority view.
        batch: Publication carrying ``tool_rows``, ``rule_rows`` and ``parameter_rows``.

    Returns:
        Entity id -> object id -> row.

    Raises:
        AuthorityError: A row with fields outside its entity, a repeated row, a non-canonical or
            self-inconsistent version, or an id held with different complete semantics.
    """
    index = {entity: dict(rows) for entity, rows in reader.typed_index().items()}
    for entity, key in zip(M.TYPED_ENTITY_IDS, _TYPED_BATCH):
        seen: set[str] = set()
        expected = set(reader._schemas[entity].fields) | {"object_id"}
        for row in getattr(batch, key):
            object_id = str(row.get("object_id"))
            if set(row) != expected or object_id in seen:
                raise AuthorityError(f"publication {entity} row {object_id} has inexact fields or repeats")
            seen.add(object_id)
            try:
                _verify_typed_row(entity, row)
            except ValueError as error:
                raise AuthorityError(f"publication {entity} row {object_id} is invalid: {error}") from error
            held = index[entity].get(object_id)
            if held is not None and semantic_row(held) != semantic_row(row):
                raise AuthorityError(f"{entity} {object_id} already exists with different complete semantics")
            index[entity][object_id] = row
    return index


def _check_entity_migration(reader: AuthorityRead, proposal: Mapping[str, Any], release: Mapping[str, Any],
                            user_instruction: str, *, rows: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
                            capsules: Mapping[str, Mapping[str, Any]] | None = None, policy: bool = True) -> dict[str, Any]:
    """Recompute the representation-only equivalence of an entity-model migration.

    The child must keep the base's prompt references exactly, and its typed closure must
    reproduce the base's ``tools.answering`` (schema2), ``rules.general`` and ``config.knobs``
    carriers: identical Tool versions, loader pins and executable identities, the exact corpus
    text and knowledge digest, and the exact knob text.

    Args:
        reader: Authority view holding the immutable base release and its carrier rows.
        proposal: Complete migration proposal body.
        release: Child release document or native row.
        user_instruction: Verbatim authorization recorded with the proposal (and decision).
        rows: Canonical typed rows by entity and id, the authority's own rows by default.
        capsules: Native Capsule rows by id, the authority's own rows by default.
        policy: Also apply the current rule and knob policy (publication and decision); ``verify()``
            of stored history passes False and checks integrity and equivalence only.

    Returns:
        The gate fields ``{"release", "base", "bundle_digest", "legacy_bundle_digest", "equivalence"}``.

    Raises:
        AuthorityError: Authorization, parent, prompt references, typed closure, content
            equivalence, plan namespace or proposal changes differ.
    """
    from ..hb_tool_entities import convert_answering, digest, implementation_version, typed_bundle
    from . import rules as R
    from .entries import CLOSURE_RULES, KNOBS_ENTRY, RULES_ENTRY, knobs_text

    if (not user_instruction.strip() or proposal.get("origin") != M.MIGRATION_KIND
            or proposal.get("user_instruction") != user_instruction):
        raise AuthorityError("entity-model migration requires its exact nonempty user instruction")
    parents = proposal.get("parents")
    if (not isinstance(parents, list) or len(parents) != 1 or not isinstance(parents[0], Mapping)
            or set(parents[0]) != {"release", "role"} or parents[0]["role"] != "base"):
        raise AuthorityError("entity-model migration requires exactly one base parent")
    epoch = reader.epoch()
    base = reader.row("release-entity", str(parents[0]["release"]))
    rid = str(release.get("id", release.get("object_id")))
    if proposal.get("release") != rid:
        raise AuthorityError("entity-model proposal does not pin its release")
    if (base["rules"] != CLOSURE_RULES or base.get("content_refs") is not None
            or any(name not in base["refs"] for name in M.CARRIERS)):
        raise AuthorityError("entity-model migration requires a closure/1 base carrying tools.answering, rules.general "
                             "and config.knobs")
    refs = {name: key for name, key in base["refs"].items() if name not in M.CARRIERS}
    entries = sorted(name for name in base["entries"] if name not in M.CARRIERS)
    if dict(release.get("refs") or {}) != refs or list(release.get("entries") or []) != entries:
        raise AuthorityError("entity-model migration changes prompt references")
    if proposal.get("plan", {}).get("control", {}).get("namespace") != f"v0-{epoch['release']}":
        raise AuthorityError("entity-model migration must retain the recorded epoch cache namespace")
    if (proposal.get("hypothesis") != _MIGRATION_CLAIM
            or proposal.get("hypothesis_digest") != sha256_hex(canon(_MIGRATION_CLAIM))):
        raise AuthorityError("entity-model migration claims no EX gain")
    legacy = reader.answering(str(base["object_id"]))
    if legacy.get("schema") != "rubiksql.answering-bundle/2":
        raise AuthorityError("entity-model migration requires a schema2 tools.answering bundle")
    texts = {name: reader.carrier_text(str(base["object_id"]), name) for name in (RULES_ENTRY, KNOBS_ENTRY)}
    capsule_rows = capsules if capsules is not None else {row["object_id"]: row for row in reader.rows("sys-capsule")}
    index = rows if rows is not None else reader.typed_index()
    try:
        M.check_release_document(release)
        tools = convert_answering(legacy, capsule_rows)
        closure = M.resolve_closure(release["content_refs"], index, capsule_rows=capsule_rows, policy=policy)
        corpus = M.corpus_from_text(texts[RULES_ENTRY])
        values = M.values_from_text(texts[KNOBS_ENTRY]) if policy else json.loads(texts[KNOBS_ENTRY])
        changes = M.closure_changes(base, release, index)
    except ValueError as error:
        raise AuthorityError(f"entity-model migration is not an exact representation: {error}") from error
    content = release["content_refs"]
    if (canon(list(content["tools"])) != canon(list(tools.content["tools"]))
            or canon(dict(content["loader"])) != canon(dict(tools.content["loader"]))):
        raise AuthorityError("entity-model tools differ from the pinned answering bundle")
    if closure.corpus_text != texts[RULES_ENTRY] or content["rules"]["knowledge_digest"] != R.rules_digest(corpus["rules"]):
        raise AuthorityError("entity-model rules differ from the rules.general carrier")
    if knobs_text(closure.values) != texts[KNOBS_ENTRY] or dict(closure.values) != values:
        raise AuthorityError("entity-model parameters differ from the config.knobs carrier")
    if proposal.get("changes") != changes:
        raise AuthorityError("entity-model proposal changes differ from its closure diff")
    implementations = {row["name"]: implementation_version(definition, row["implementation"])
                       for row, definition in zip(tools.tool_rows, tools.definitions)}
    equivalence = {"prompt_refs": digest(refs),
                   "rules": {"carrier": base["refs"][RULES_ENTRY], "carrier_sha256": sha256_hex(texts[RULES_ENTRY]),
                             "reconstructed_sha256": sha256_hex(closure.corpus_text),
                             "knowledge_digest": content["rules"]["knowledge_digest"]},
                   "knobs": {"carrier": base["refs"][KNOBS_ENTRY], "carrier_sha256": sha256_hex(texts[KNOBS_ENTRY]),
                             "values_sha256": sha256_hex(knobs_text(closure.values))},
                   "tools": {"carrier": base["refs"]["tools.answering"], "definitions": digest(list(tools.definitions)),
                             "loader": digest(dict(content["loader"])), "implementations": implementations}}
    return {"release": rid, "base": str(base["object_id"]), "bundle_digest": digest(typed_bundle(content)),
            "legacy_bundle_digest": digest(legacy), "equivalence": equivalence}


def _require_entity_gate(gate: Mapping[str, Any], check: Mapping[str, Any]) -> str:
    """Require an entity-model gate payload to equal a re-verified equivalence; return its deployed commit.

    Raises:
        AuthorityError: The gate differs from ``check`` or lacks a 40-hex deployed commit.
    """
    body = {key: value for key, value in gate.items() if key != "scientific_gate"}
    commit = body.get("deployed_commit")
    if (body != {"schema": M.SCHEMA_ENTITY_GATE, **check, "deployed_commit": commit}
            or not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None):
        raise AuthorityError("entity-model gate differs from its re-verified equivalence")
    return commit


def _check_entity_gate(reader: AuthorityRead, pid: str, check: Mapping[str, Any], deployed_commit: Any) -> None:
    """Require the migration's recorded gate event to re-verify and name the decision's deployed commit.

    Raises:
        AuthorityError: No entity-model gate, a gate that differs from ``check``, or another deployed commit.
    """
    gates = [event["payload"] for event in reader.events("gate")
             if event["subject"] == pid and event["payload"].get("schema") == M.SCHEMA_ENTITY_GATE]
    if not gates:
        raise AuthorityError("entity-model migration has no entity-model gate")
    if _require_entity_gate(gates[-1], check) != deployed_commit:
        raise AuthorityError("entity-model migration deployed commit differs from its gate")


class Authority:
    """A writer holding the authority lock and verified ECS binding."""

    def __init__(self, control: Any, binding: AuthorityBinding | GitAuthorityBinding) -> None:
        self.control, self.binding = control, binding
        self.path = control.path if isinstance(binding, GitAuthorityBinding) else binding.db_path
        if control is not None:
            self.control.ws._rubiksql_authority_binding = binding
            self.control.ws._rubiksql_authority_db = str(self.path)
            self.control.ws._rubiksql_lab_dir = str(self.path.parent)
        self._batch_dirty = False
        self._writer_active = False
        self._writer_thread: int | None = None
        self._writer_token: object | None = None
        self._retire_refresh_only = False
        self.git_binding: GitAuthorityBinding | None = None
        self._git_projection: Any = None
        self._read_monitor: sqlite3.Connection | None = None
        self._read_memo_version: int | None = None
        self._read_memo: dict[tuple[Any, ...], Any] = {}

    def _attach_read_memo(self, reader: AuthorityRead) -> None:
        """Share decoded state only across this writer's unchanged committed views."""
        if self._read_monitor is None:
            self._read_monitor = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=60)
            self._read_monitor.execute("PRAGMA query_only=ON")
        before = int(self._read_monitor.execute("PRAGMA data_version").fetchone()[0])
        # Reading schema_version establishes the reader's deferred BEGIN snapshot.
        revision = reader._database_revision()
        after = int(self._read_monitor.execute("PRAGMA data_version").fetchone()[0])
        if before != after:
            # A concurrent commit raced snapshot capture. Keep this reader isolated;
            # never associate its old snapshot with the monitor's new revision.
            self._read_memo_version, self._read_memo = None, {}
            return
        if after != self._read_memo_version:
            # Existing live readers retain their old dictionaries and snapshots.
            self._read_memo_version, self._read_memo = after, {}
        reader._read_revision, reader._read_memo = revision, self._read_memo

    def _close_read_memo(self) -> None:
        """Release this writer's SQLite revision monitor and decoded state."""
        if self._read_monitor is not None:
            self._read_monitor.close()
            self._read_monitor = None
        self._read_memo_version, self._read_memo = None, {}

    @classmethod
    def active_writer(cls, db: Path | None = None, *, binding: GitAuthorityBinding | None = None) -> Authority | None:
        """Return this thread's already-fenced writer for the exact database, if active."""
        if binding is not None:
            from .git_authority import active_writer

            return active_writer(binding)
        if db is None:
            return None
        writer = getattr(_LOCK_STATE, "writers", {}).get(Path(db).resolve())
        return writer if writer is not None and writer._writer_active and writer._writer_thread == threading.get_ident() else None

    @classmethod
    @contextlib.contextmanager
    def read(cls, db: Path | None = None, *, binding: AuthorityBinding | GitAuthorityBinding,
             export_anchor: Path | None = None, commit: str | None = None) -> Iterator[AuthorityRead]:
        """Open mode=ro, query_only and one BEGIN transaction without HB bootstrap.

        Args:
            db: Native authority database.
            binding: Explicit verified package identity.
            export_anchor: Exact export proving a restored unbound database.

        Yields:
            One stable read view until exit.

        Raises:
            AuthorityError: Missing database or native read failure.
        """
        if isinstance(binding, GitAuthorityBinding):
            if db is not None or export_anchor is not None:
                raise AuthorityError("Git authority reads select a commit, not a local database or export fallback")
            from .git_authority import GitAuthority

            with GitAuthority(binding).read(commit=commit) as reader:
                yield reader
            return
        if commit is not None:
            raise AuthorityError("a historical ECS database has no Git authority commit selector")
        path = Path(db if db is not None else binding.db_path).resolve()
        with cls._read_native(path, binding=binding, export_anchor=export_anchor) as reader:
            yield reader

    @classmethod
    @contextlib.contextmanager
    def _read_native(cls, db: Path, *, binding: AuthorityBinding | GitAuthorityBinding,
                     export_anchor: Path | None = None) -> Iterator[AuthorityRead]:
        """Read a private projection or historical native database without remote dispatch."""
        path = Path(db).resolve()
        try:
            with contextlib.closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=60)) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA query_only=ON")
                connection.execute("BEGIN")
                reader = AuthorityRead(connection, path, binding, export_anchor)
                owner = cls.active_writer(path)
                if owner is not None:
                    owner._attach_read_memo(reader)
                if export_anchor is not None:
                    _head, files, _commit = _read_export(export_anchor)
                    reader._ancillary_files = {name: value for name, value in files.items() if name in ANCILLARY_FILES}
                try:
                    yield reader
                finally:
                    reader._view_active = False
                    connection.rollback()
        except sqlite3.Error as error:
            raise AuthorityError(f"authority read refused: {error}") from error

    @classmethod
    def transact(cls, operation: Callable[[Authority], _MemoValue], *,
                 binding: AuthorityBinding | GitAuthorityBinding) -> _MemoValue:
        """Commit a replay-safe domain operation, rebuilding after Git contention.

        Args:
            operation: Callback that rechecks every precondition and rebuilds
                its changes from the supplied writer. No unconfirmed execution
                effects or retained staged candidates are permitted.
            binding: Exact authority identity, unchanged across attempts.

        Returns:
            Callback result, only after successful writer exit and publication.

        Raises:
            AuthorityError: Domain refusal or non-conflict failure immediately;
                GitAuthorityConflict after thirty-two conflicting attempts or once
                ``git_authority.WRITE_RETRY_SECONDS`` elapsed. Confirmed checkpoints
                and unconfirmed pushes are never replayed.
        """
        from .git_authority import retry_conflicts

        def attempt() -> _MemoValue:
            with cls.write(binding=binding) as writer:
                return operation(writer)

        if not isinstance(binding, GitAuthorityBinding) or cls.active_writer(binding=binding) is not None:
            return attempt()
        return retry_conflicts(attempt)

    @classmethod
    @contextlib.contextmanager
    def write(cls, db: Path | None = None, *, binding: AuthorityBinding | GitAuthorityBinding,
              expected_commit: str | None = None) -> Iterator[Authority]:
        """Fence instance/path, acquire cross-process lock, then open the HB writer.

        Args:
            db: Canonical package-bound native database path.
            binding: Package binding; cannot be overridden through environment.

        Yields:
            Writer for one complete publication/accounting/export batch.

        Raises:
            AuthorityError: Wrong instance, path, marker or altered authority.
        """
        if isinstance(binding, GitAuthorityBinding):
            if db is not None:
                raise AuthorityError("Git authority writes never select a local authoritative database")
            from .git_authority import GitAuthority

            with GitAuthority(binding).write(expected_commit=expected_commit) as writer:
                yield writer
            return
        if expected_commit is not None:
            raise AuthorityError("a historical ECS writer has no Git compare-and-swap commit")
        configured = B.load_binding()
        if isinstance(configured, GitAuthorityBinding):
            raise AuthorityError("Git authority is configured; the historical ECS database is no longer writable authority")
        if Path(db if db is not None else binding.db_path).resolve() != binding.db_path or B.metadata_instance_id() != binding.instance_id:
            raise AuthorityError("authority writer instance or database binding differs")
        existing = cls.active_writer(binding.db_path)
        if existing is not None:
            if existing.binding != binding:
                raise AuthorityError("nested authority writer binding differs")
            yield existing
            return
        from .control import Control
        with authority_lock(binding.db_path):
            retiring = False
            with cls.read(binding.db_path, binding=binding) as reader:
                if reader.events("authority"):
                    if not reader._binding_matches():
                        raise AuthorityError("authority writer marker and package binding differ")
                    retiring = any(event["kind"] == "decision" and event["payload"].get("kind") == "refresh-tools"
                                   or event["kind"] == "publish" and event["payload"].get("body", {}).get("origin") == "refresh-tools"
                                   for event in reader.events())
                    if retiring:
                        from .authority_state import migration_preimage

                        migration_preimage(reader)
                    else:
                        violations = _recoverable_violations(reader)
                        if violations:
                            raise AuthorityError("authority writer integrity refused: " + "; ".join(violations))
                elif any(event["object_id"] != "e" + sha256_hex(canon({key: value for key, value in event.items() if key != "object_id"}))[:16]
                         for event in reader.events()):
                    raise AuthorityError("unmarked authority has altered events")
                else:
                    from .integrity import check_prompt_row

                    prompts = reader.rows("sys-prompt")
                    for row in prompts:
                        check_prompt_row(row)
                    releases = reader.rows("release-entity")
                    keys = {(row["name"], row["manifest"]["meta"]["content_key"]) for row in prompts}
                    if len(releases) != 1 or any((name, key) not in keys for name, key in releases[0]["refs"].items()):
                        raise AuthorityError("unmarked authority has an incomplete seeded root")
            control = Control.open(binding.db_path, create=False)
            writer = cls(control, binding)
            writer._retire_refresh_only = retiring
            writer._writer_active = True
            writer._writer_thread = threading.get_ident()
            writer._writer_token = object()
            writers = getattr(_LOCK_STATE, "writers", None)
            if writers is None:
                writers = _LOCK_STATE.writers = {}
            writers[binding.db_path] = writer
            try:
                yield writer
            finally:
                writers.pop(binding.db_path, None)
                writer._writer_active = False
                writer._writer_token = None
                writer._close_read_memo()
                # Existing native facades (including lab callers) share this engine.
                # Release lookup ownership without disposing their database resources.
                control.ws.release()

    @contextlib.contextmanager
    def reader(self) -> Iterator[AuthorityRead]:
        """Open a new committed reader while the writer lock stays held."""
        if self._git_projection is not None:
            with self._git_projection.reader() as reader:
                yield reader
            return
        with self.read(self.path, binding=self.binding) as reader:
            if self._writer_active and self._writer_thread == threading.get_ident():
                reader._writer_owner = self
                reader._writer_token = self._writer_token
            yield reader

    def head(self) -> AuthorityHead:
        with self.reader() as reader:
            return reader.head()

    @classmethod
    def project(cls, *, binding: GitAuthorityBinding, context: Any = None, commit: str | None = None) -> Any:
        """Open a read-only local HB facade pinned to a verified Git commit.

        Args:
            binding: Canonical Git identity, with no local DB authority.
            context: Optional caller's HB context for the read facade.
            commit: Reachable exact SHA; current remote ref by default.

        Returns:
            Control; close its workspace to dispose of the private projection.
        """
        from .git_authority import GitAuthority

        return GitAuthority(binding).project(context=context, commit=commit)

    def commit(self) -> ExportResult:
        """Confirm this staged batch on the canonical Git remote before effects.

        Returns:
            Exact confirmed commit and covered/live event revision.

        Raises:
            AuthorityError: Not a Git writer, stale lease, invalid state, or
                unconfirmed remote publication. Domain work is never retried.
        """
        if self._git_projection is None:
            raise AuthorityError("commit requires a Git authority writer")
        self._require_writer(allow_pending=True, allow_tools=True)
        return self._git_projection.publish(secret_inputs())

    def _require_writer(self, *, allow_pending: bool = False, allow_tools: bool = False) -> None:
        if (not self._writer_active or self._writer_thread != threading.get_ident()
                or self.path not in getattr(_LOCK_STATE, "active", set())):
            raise AuthorityError("authority mutation requires the active instance-bound writer context")
        if self._retire_refresh_only:
            raise AuthorityError("obsolete refresh preimage permits only the fenced retire-refresh migration")
        with self.reader() as reader:
            violations = _recoverable_violations(reader)
            if violations:
                raise AuthorityError("authority mutation integrity refused: " + "; ".join(violations))
            if reader.pending_publications() and not allow_pending:
                raise AuthorityError("complete the exact prepared publication before another mutation")
            if reader.pending_tool_registrations() and not allow_tools:
                raise AuthorityError("complete the exact prepared tool registration before another mutation")
            from .workspace_migration import pending_migration

            if pending_migration(reader):
                raise AuthorityError("incomplete historical lab migration prevents authority mutation")

    def events(self, kind: str | None = None) -> tuple[Mapping[str, Any], ...]:
        """Read current events while retaining this writer's exclusive lock."""
        with self.reader() as reader:
            return reader.events(kind)

    def record_workspace_change(self, operation: str) -> str:
        """Record an evidence write using counts and full-row digests, without row payloads.

        Args:
            operation: Nonempty operation label owned by the canonical evidence writer.

        Returns:
            Immutable operational event id. The writer must export before completion.

        Raises:
            AuthorityError: No active fenced writer, incomplete publication or blank label.
        """
        from .workspace_migration import validate_evidence
        from .workspace_state import EVIDENCE_TABLES

        self._require_writer()
        if not operation.strip():
            raise AuthorityError("workspace change requires a nonempty operation")
        with self.reader() as reader:
            validate_evidence({entity: reader._held_evidence(entity) for entity in EVIDENCE_TABLES})
            state = reader.operational_state()
            payload = {"schema": "rubiksql.workspace-change/1", "operation": operation,
                       "workspace_digest": sha256_hex(canon(state)),
                       "counts": {entity: value["count"] for entity, value in state.items()},
                       "entity_digests": {entity: value["rows_digest"] for entity, value in state.items()},
                       "schemas_digest": sha256_hex(canon({entity: value["schema"] for entity, value in state.items()}))}
            return self._event("workspace-change", reader.head().release, payload, "agent:workspace")

    def _expected(self, expected: AuthorityHead, *, require_export: bool = True, full_revision: bool = False) -> None:
        current = self.head()
        if expected.authority_commit is not None and expected.authority_commit != current.authority_commit:
            raise AuthorityError("expected Git authority commit is stale")
        if (current.release, current.generation, current.decision) != (expected.release, expected.generation, expected.decision):
            raise AuthorityError("expected authority head is stale")
        if full_revision and current.revision != expected.revision:
            raise AuthorityError("expected authority revision is stale")
        if require_export and not self._batch_dirty and not current.exported:
            raise AuthorityError("authority has unexported rows/events; finish export before mutation")

    def _event(self, kind: str, subject: str, payload: Mapping[str, Any], actor: str) -> str:
        event = self.control.record_event(kind, subject, payload, actor=actor)
        self._batch_dirty = True
        return event

    def mark(self, epoch: Mapping[str, Any], *, user_instruction: str,
             request_id: str | None = None, seed: Mapping[str, Any] | None = None) -> str:
        """Record the sole initial marker on an existing unmeasured seeded root."""
        if not user_instruction.strip():
            raise AuthorityError("initial mark needs the verbatim user instruction")
        from .store import release_document_id

        if (epoch.get("label") != "v0" or not all(isinstance(epoch.get(key), str) and re.fullmatch(r"[0-9a-f]{40}", epoch[key])
                for key in ("source_commit", "delivery_commit")) or epoch.get("epoch") != f"v0:{epoch.get('delivery_commit')}"):
            raise AuthorityError("authority epoch provenance is malformed")
        with self.reader() as reader:
            if reader.events("authority"):
                if (reader.marker()["epoch"] != dict(epoch) or reader.marker().get("user_instruction") != user_instruction
                        or request_id is not None and reader.marker().get("request_id") != request_id
                        or seed is not None and reader.marker().get("seed") != dict(seed)):
                    raise AuthorityError("existing marker epoch or user instruction differs")
                return str(reader.events("authority")[0]["object_id"])
            if reader.rows("proposal-entity") or any(reader.events(kind) for kind in ("decision", "outcome", "eval-attach")):
                raise AuthorityError("cutover root has prior decisions, proposals or measured runs")
            releases = reader.rows("release-entity")
            if len(releases) != 1 or releases[0]["object_id"] != epoch["release"]:
                raise AuthorityError("cutover needs exactly the current seeded root")
            root = releases[0]
            if release_document_id(root, epoch=str(epoch["epoch"])) != root["object_id"]:
                raise AuthorityError("authority epoch does not identify the seeded root")
        if isinstance(self.binding, GitAuthorityBinding):
            if not request_id or not seed or self._git_projection is None:
                raise AuthorityError("fresh Git marker requires an exact request, seed and private projection")
            return self._event("authority", str(epoch["release"]), {"schema": SCHEMA_GIT_AUTHORITY, "operation": "mark",
                "binding": B.binding_document(self.binding), "epoch": dict(epoch), "request_id": request_id,
                "seed": dict(seed), "user_instruction": user_instruction}, "agent:META/fresh-epoch")
        return self._event("authority", str(epoch["release"]), {"schema": SCHEMA_AUTHORITY, "operation": "mark",
            "authority_uuid": self.binding.authority_uuid, "instance_id": self.binding.instance_id,
            "db_path": str(self.path), "epoch": dict(epoch), "user_instruction": user_instruction}, "agent:META/user-2026-10-09-hb-cutover")

    def _put(self, entity: str, row: Mapping[str, Any]) -> bool:
        entity_type = _entities(self.control)[entity]
        held = self.control.ws.get(str(row["object_id"]), entity=entity_type)
        if held is not None:
            if semantic_row(held) != semantic_row(row):
                raise AuthorityError(f"{entity} {row['object_id']} already exists with different complete semantics")
            return False
        self.control.ws.upsert(entity_type, dict(row), overwrite=False)
        return True

    def publish(self, batch: PublicationBatch, *, expected: AuthorityHead, actor: str,
                parent_run: Path | None = None, dbs_dir: Path | None = None,
                user_instruction: str | None = None, prepare_receipt: bool = True) -> PublishResult:
        """Publish a fully validated native batch against its actual parent evidence."""
        return self._publish(batch, expected=expected, actor=actor, parent_run=parent_run, dbs_dir=dbs_dir,
                             user_instruction=user_instruction, prepare_receipt=prepare_receipt)

    def _publish(self, batch: PublicationBatch, *, expected: AuthorityHead, actor: str,
                 parent_run: Path | None = None, dbs_dir: Path | None = None,
                 request: Mapping[str, Any] | None = None, user_instruction: str | None = None,
                 dry_run: bool = False, prepare_receipt: bool = True) -> PublishResult:
        """Validate the entire batch before any row write, then publish create-only.

        A closure/2 release (``rules`` ``rubiksql.closure/2``, schema ``rubiksql.release/2``) carries
        its new canonical typed rows in the batch; every typed reference must resolve over the
        authority's rows plus the batch, the proposal ``changes`` must equal the exact closure diff,
        and a changed typed tool bundle needs its tool receipts.

        Args:
            batch: Complete prompt/module/toolkit/typed/release/proposal/receipt content.
            expected: Head whose facts produced this publication.
            actor: Author recorded in the publication event.
            dry_run: Validate the entire batch and return without writing authority state.

        Returns:
            Immutable identities and the resulting live revision.

        Raises:
            AuthorityError: Stale facts, policy/closure/receipt failure or collision.
        """
        self._require_writer(allow_pending=True)
        if not prepare_receipt and self._git_projection is None:
            raise AuthorityError("omitting prepared publication receipts requires an atomic Git writer")
        from .control import check_row_integrity
        from .store import proposal_id, release_document_id

        cutover = batch.proposal.get("origin") == "cutover"
        typed = batch.release.get("rules") == M.CLOSURE_V2
        self._expected(expected, require_export=False)
        with self.reader() as reader:
            epoch = reader.epoch()
            release, proposal = dict(batch.release), dict(batch.proposal)
            rid, pid = str(release["id"]), str(proposal["id"])
            pending = reader.pending_publications()
            if pending and (set(pending) != {pid} or pending[pid].get("batch_digest") != _batch_digest(batch)):
                raise AuthorityError("publication differs from the exact prepared batch")
            try:
                identical = release_document_id(release, epoch=epoch["epoch"]) == rid
            except ValueError:
                identical = False
            if not identical or proposal_id(proposal) != pid or proposal["release"] != rid:
                raise AuthorityError("publication release or proposal identity differs")
            allowed = {"schema", "id", "refs", "entries", "rules", "reason", "origin", "epoch", "parents", "proposal"}
            if (set(release) - (allowed | ({"content_refs"} if typed else set()))
                    or release.get("schema") != (M.SCHEMA_RELEASE_V2 if typed else "rubiksql.release/1")
                    or release.get("epoch", epoch["epoch"]) != epoch["epoch"]):
                raise AuthorityError("publication release carries unsupported semantic fields")
            if not typed and any(getattr(batch, key) for key in _TYPED_BATCH):
                raise AuthorityError("a closure/1 publication carries no typed Tool, rule or parameter rows")
            if cutover and typed:
                raise AuthorityError("cutover publication is closure/1-only")
            capsules = {row["object_id"]: row for row in reader.rows("sys-capsule")}
            toolkits = {row["object_id"]: row for row in reader.rows("sys-toolkit")}
            capsules.update({row["object_id"]: row for row in batch.capsule_rows})
            toolkits.update({row["object_id"]: row for row in batch.toolkit_rows})
            typed_index = _batch_typed_index(reader, batch) if typed else {}
            existing_release = next((row for row in reader.rows("release-entity") if row["object_id"] == rid), None)
            release_published = any(event["payload"].get("release") == rid and isinstance(event["payload"].get("body"), Mapping)
                                    for event in reader.events("publish"))
            if existing_release is not None and release_published:
                previous = json.loads(reader.store_files()[f"releases/{rid}.json"])
                # Content-addressed releases can be proposed again from a later base.
                # Their first publication metadata stays immutable; the new proposal owns its parents.
                if any(key in release and release[key] != previous.get(key) for key in ("parents", "proposal", "origin", "reason")):
                    raise AuthorityError("existing release semantic metadata differs")
                release = dict(previous)
            elif (release.get("parents", proposal["parents"]) != proposal["parents"]
                    or release.get("proposal", pid) != pid
                    or release.get("origin", "proposal") != ("cutover" if cutover else "proposal")):
                raise AuthorityError("release and proposal semantic metadata differ")
            base = next((item["release"] for item in proposal["parents"] if item.get("role", "base") == "base"), None)
            if base != expected.release:
                raise AuthorityError("proposal base is not the current head")
            base_row = reader.row("release-entity", str(base))
            if not typed and base_row["rules"] == M.CLOSURE_V2:
                raise AuthorityError("a closure/2 base admits only closure/2 children")
            migrations = [event["payload"] for event in reader.events("decision")
                          if event["payload"].get("kind") == M.MIGRATION_KIND]
            if not typed and migrations and proposal.get("origin") in {"agent", "human"}:
                raise AuthorityError(f"the head {base} is a retained historical closure/1 release; after the entity-model "
                                     f"migration only closure/2 children are published; return with "
                                     f"`authority rollback {migrations[-1]['to']}`")
            for parent in proposal["parents"]:
                if parent.get("role") != "merge":
                    continue
                pool = [reader.proposal(str(row["object_id"])) for row in reader.rows("proposal-entity")
                        if row["release"] == parent["release"]]
                if not any(any(item.get("role", "base") == "base" and item.get("release") == expected.generation
                               for item in body["parents"]) for body in pool):
                    raise AuthorityError("merge parent is not a proposal of this generation's pool")
            from .evolve import release_measurement_blocker

            if release_measurement_blocker(rid, expected.generation,
                    attachments=(event["payload"] for event in reader.events("eval-attach")),
                    outcomes=(event["payload"] for event in reader.events("outcome")),
                    reservations=(event["payload"] for event in reader.events("eval-reserve"))) is not None:
                raise AuthorityError("release was already measured in this generation")
            if cutover:
                root = reader.row("release-entity", str(epoch["release"]))
                if (len(root["refs"]) != 34 or "tools.answering" in root["refs"]
                        or expected.release != epoch["release"] or not user_instruction
                        or user_instruction != reader.marker().get("user_instruction")):
                    raise AuthorityError("cutover publication requires the exact authorized initial marker and unchanged root34")
                if reader.events("decision") or reader.events("outcome") or reader.events("eval-attach"):
                    raise AuthorityError("cutover is only permitted before any measured decision")
                if proposal["parents"] != [{"release": epoch["release"], "role": "base"}] or release["refs"] != {**root["refs"], "tools.answering": release["refs"].get("tools.answering")}:
                    raise AuthorityError("cutover changes content beyond adding tools.answering")
                if "tools.answering" not in release["refs"]:
                    raise AuthorityError("cutover has no tools.answering")
                if proposal.get("changes") != {"tools.answering": {"from": None, "to": release["refs"]["tools.answering"]}}:
                    raise AuthorityError("cutover publication changes only the new tools.answering reference")
                if proposal.get("plan", {}).get("control", {}).get("namespace") != f"v0-{epoch['release']}":
                    raise AuthorityError("cutover plan must retain the recorded epoch cache namespace")
                if any(row["object_id"] != pid and reader.proposal(str(row["object_id"]))["origin"] == "cutover"
                       for row in reader.rows("proposal-entity")):
                    raise AuthorityError("cutover admits exactly one proposal")
            elif proposal.get("origin") not in {"agent", "human"}:
                raise AuthorityError("child publication origin must be agent or human")
            if not cutover:
                with self._publication_store(reader) as store:
                    facts = self._proposal_facts(reader, store, parent_run=parent_run, dbs_dir=dbs_dir)
                    self._check_publication_sources(reader, batch, store, facts)
            rows = {row["object_id"]: row for row in reader.rows("sys-prompt")}
            for row in batch.prompt_rows:
                check_row_integrity(row)
                rows[row["object_id"]] = row
            by_key = {(row["name"], row["manifest"]["meta"]["content_key"]): row for row in rows.values()}
            if any((name, key) not in by_key for name, key in release["refs"].items()):
                raise AuthorityError("publication closure is incomplete")
            if batch.toolkit_rows and reader.events("toolkit-retirement"):
                raise AuthorityError("retired Toolkit persistence cannot be published")
            if typed:
                self._check_typed_publication(reader, batch, base_row, typed_index, capsules)
            elif "tools.answering" in release["refs"]:
                from ..hb_tool_entities import validate_answering, validate_tool_receipts

                payload = json.loads(by_key[("tools.answering", release["refs"]["tools.answering"])]["manifest"]["meta"]["text"])
                if payload.get("schema") == "rubiksql.answering-bundle/1":
                    from ..hb_tool_entities import _validate_legacy_answering

                    if reader.events("toolkit-retirement") or proposal.get("origin") != "cutover":
                        raise AuthorityError("retired Toolkit persistence cannot be published")
                    _validate_legacy_answering(payload, capsule_rows=capsules, toolkit_rows=toolkits)
                else:
                    validate_answering(payload, capsule_rows=capsules)
                parent = reader.answering(str(base)) if "tools.answering" in base_row["refs"] else None
                if parent != payload:
                    from ..hb_tool_entities import digest as bundle_digest

                    validate_tool_receipts(payload, batch.gate, parent_bundle=bundle_digest(parent) if parent else None,
                                           launch=False, tiers=self._tool_tiers())
            release_row = {"object_id": rid, **{key: release[key] for key in ("refs", "entries", "rules")},
                           **({"content_refs": release["content_refs"]} if typed else {}),
                           "reason": release.get("reason", proposal.get("reason", "")), "created_at": proposal.get("created_at", 0.0)}
            if existing_release is not None and release_published:
                release_row = dict(existing_release)
            report = self._gate_batch(reader, batch)
            from .evolve import _gate_summary

            scientific_gate = _gate_summary(report)
            proposal_row = _proposal_row(proposal, scientific_gate)
            # Typed rows precede the release row, so an interrupted batch never holds an unresolvable release.
            candidates = ([("sys-prompt", row) for row in batch.prompt_rows] + [("sys-capsule", row) for row in batch.capsule_rows]
                          + [("sys-toolkit", row) for row in batch.toolkit_rows]
                          + [(entity, row) for entity, key in zip(M.TYPED_ENTITY_IDS, _TYPED_BATCH) for row in getattr(batch, key)]
                          + [("release-entity", release_row), ("proposal-entity", proposal_row)])
            for entity, candidate in candidates:
                existing = next((row for row in reader.rows(entity) if row["object_id"] == candidate["object_id"]), None)
                if existing is not None and semantic_row(existing) != semantic_row(candidate):
                    raise AuthorityError(f"{entity} {candidate['object_id']} already exists with different complete semantics")
            verified = reader.verify(require_export=False)
            incomplete = f"proposal {pid} has no unique complete publication event"
            if any(problem != incomplete for problem in verified.violations):
                raise AuthorityError("publication cannot repair unrelated authority integrity violations")
            publications = [event for event in reader.events("publish") if event["subject"] == pid
                            and isinstance(event["payload"].get("body"), Mapping)]
            digest = _batch_digest(batch)
            if publications:
                if len(publications) != 1 or publications[0]["payload"].get("batch_digest") != digest:
                    raise AuthorityError("existing publication batch differs")
                gates = [event for event in reader.events("gate") if event["subject"] == pid]
                if not gates and not dry_run:
                    self._event("gate", pid, {**dict(batch.gate), "scientific_gate": scientific_gate,
                        **({"schema": "rubiksql.scientific-gate/1", "release": rid} if not batch.gate.get("schema") else {})}, actor)
                return PublishResult(pid, rid, 0, self.head().revision, reader.head().export_commit)
            if reader.head().revision != expected.revision:
                raise AuthorityError("expected authority revision is stale")
            if not cutover and not pending and not reader.head().exported and not self._batch_dirty:
                nonexport = [event["object_id"] for event in reader.events() if event["kind"] != "export"]
                exports = reader.events("export")
                same_events = bool(exports) and exports[-1]["payload"].get("covered_non_export_ids") == nonexport
                held_candidates = any(any(row["object_id"] == candidate["object_id"] for row in reader.rows(entity))
                                      for entity, candidate in candidates if entity != "release-entity")
                if not same_events or not held_candidates:
                    raise AuthorityError("authority has an unfinished unrelated batch; export it before publication")
            scan_secrets({"publication.json": _bytes(dataclasses.asdict(batch)), "request.json": _bytes(request)}, secret_inputs())
            if dry_run:
                return PublishResult(pid, rid, 0, reader.head().revision, reader.head().export_commit)
        if not pending and prepare_receipt:
            self._event("publish", pid, {"schema": "rubiksql.publication/1", "phase": "prepared", "proposal": pid,
                "release": rid, "batch": _batch_document(batch), "batch_digest": digest,
                "request": _json_value(request),
                "candidates": {entity: [_json_value(row) for kind, row in candidates if kind == entity]
                               for entity in sorted({kind for kind, _row in candidates})}}, actor)
        created = sum(self._put(entity, row) for entity, row in candidates)
        self._event("publish", pid, {"schema": "rubiksql.publication/1", "proposal": pid, "release": rid, "body": proposal,
            "prompt_ids": [row["object_id"] for row in batch.prompt_rows], "capsule_ids": [row["object_id"] for row in batch.capsule_rows],
            "toolkit_ids": [row["object_id"] for row in batch.toolkit_rows], "batch_digest": digest,
            **{f"{key[:-5]}_ids": [row["object_id"] for row in getattr(batch, key)] for key in _TYPED_BATCH if getattr(batch, key)}},
            actor)
        self._event("gate", pid, {**dict(batch.gate), "scientific_gate": scientific_gate,
            **({"schema": "rubiksql.scientific-gate/1", "release": rid} if not batch.gate.get("schema") else {})}, actor)
        return PublishResult(pid, rid, created, self.head().revision)

    def _check_typed_publication(self, reader: AuthorityRead, batch: PublicationBatch, base: Mapping[str, Any],
                                 typed_index: Mapping[str, Mapping[str, Mapping[str, Any]]],
                                 capsules: Mapping[str, Mapping[str, Any]]) -> None:
        """Validate a closure/2 child's exact typed closure, diff, row set and tool receipts.

        Args:
            reader: The publication's authority view.
            batch: Publication whose release is closure/2.
            base: The base release row (the current head).
            typed_index: Authority canonical rows merged with the batch's validated typed rows.
            capsules: Authority Capsules merged with the batch's Capsules.

        Raises:
            AuthorityError: Unresolvable or altered references, unreferenced batch rows or Capsules,
                Toolkit rows, changes other than the closure diff, or invalid tool receipts.
        """
        from ..hb_tool_entities import digest as bundle_digest, typed_bundle, validate_tool_receipts

        release, proposal = batch.release, batch.proposal
        try:
            M.check_release_document(release)
            M.resolve_closure(release["content_refs"], typed_index, capsule_rows=capsules)
            changes = M.closure_changes(base, release, typed_index)
        except ValueError as error:
            raise AuthorityError(f"publication typed closure is invalid: {error}") from error
        content = release["content_refs"]
        referenced = {(M.TOOL_ENTITY, ref["object_id"]) for ref in content["tools"]}
        referenced |= {(M.RULE_ENTITY, ref["object_id"]) for ref in content["rules"]["items"]}
        referenced.add((M.PARAMETER_ENTITY, content["parameters"]["object_id"]))
        if any((entity, row["object_id"]) not in referenced
               for entity, key in zip(M.TYPED_ENTITY_IDS, _TYPED_BATCH) for row in getattr(batch, key)):
            raise AuthorityError("publication carries typed rows its release does not reference")
        pins = {ref["capsule_id"] for ref in content["loader"]["modules"]} | {content["loader"]["serializer"]["capsule_id"]}
        if batch.toolkit_rows or any(row["object_id"] not in pins for row in batch.capsule_rows):
            raise AuthorityError("closure/2 publication carries Toolkit rows or Capsules outside its loader pins")
        if proposal.get("changes") != changes:
            raise AuthorityError("publication changes differ from the exact closure diff")
        child = typed_bundle(content)
        parent = (typed_bundle(base["content_refs"]) if base["rules"] == M.CLOSURE_V2
                  else reader.answering(str(base["object_id"])) if "tools.answering" in base["refs"] else None)
        if parent is None or bundle_digest(child) != bundle_digest(parent):
            validate_tool_receipts(child, batch.gate, parent_bundle=bundle_digest(parent) if parent is not None else None,
                                   launch=False, tiers=self._tool_tiers())

    @contextlib.contextmanager
    def _publication_store(self, reader: AuthorityRead) -> Iterator[Any]:
        """Map verified current facts during an exact incomplete-batch recovery."""
        from .store import Store

        with tempfile.TemporaryDirectory(prefix="rubiksql-publication-facts-") as temporary:
            root = Path(temporary)
            for name, value in reader.store_files().items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
            yield Store(package=root)

    def _check_publication_sources(self, reader: AuthorityRead, batch: PublicationBatch,
                                   store: Any, facts: Mapping[str, Any]) -> None:
        """Enforce authoring constraints on the actual native base-to-child diff.

        A closure/2 child's diff is :func:`entity_model.closure_changes` over the head row, the batch release and the
        canonical rows the authority and the batch hold. Compiled typed keys are checked through their canonical
        authoring selector (``rules.general``, ``parameters:knobs`` or ``tools:loader``). Per-tool, per-rule and corpus
        diff keys are not external proposal names. Rule changes are traced to their dossier failures through the
        reconstructed corpora. An ordinary publication keeps its head's release format.
        """
        from . import evolve, rules
        from .entity_model import (CLOSURE_V2, PARAMETER_ENTITY, RULE_ENTITY, TOOL_ENTITY, EntityModelError, change_kind,
                                   closure_changes, reconstruct_corpus)
        from .runtime import load_release

        parent = load_release(reader.head().release, stores=store, allow_retired=True)
        proposed = batch.proposal
        head_row = reader.row("release-entity", reader.head().release)
        typed = batch.release.get("rules") == CLOSURE_V2
        if typed != (head_row["rules"] == CLOSURE_V2):
            raise AuthorityError("publication changes its head's release format; only the entity-model migration "
                                 "converts a closure/1 release")
        typed_ops = {"rules": ("rules.general", "rules"), "knobs": ("parameters:knobs", "knobs"),
                     "tool": ("tools:loader", "tools")}
        index: dict[str, dict[str, Mapping[str, Any]]] = {}
        if typed:
            index = {entity: dict(rows) for entity, rows in reader.typed_index().items()}
            for entity, added in ((TOOL_ENTITY, batch.tool_rows), (RULE_ENTITY, batch.rule_rows),
                                  (PARAMETER_ENTITY, batch.parameter_rows)):
                index.setdefault(entity, {}).update({str(row["object_id"]): row for row in added})
            try:
                actual = closure_changes(head_row, batch.release, index)
            except EntityModelError as error:
                raise AuthorityError(f"publication typed references do not resolve: {error}") from error
        else:
            actual = {name: {"from": parent.refs.get(name), "to": batch.release["refs"].get(name)}
                      for name in sorted(set(parent.refs) | set(batch.release["refs"]))
                      if parent.refs.get(name) != batch.release["refs"].get(name)}
        if not actual or actual != proposed.get("changes"):
            raise AuthorityError("publication changes differ from the actual parent-to-child refs")
        if (proposed.get("plan", {}).get("control", {}).get("namespace") != facts["namespace"]):
            raise AuthorityError("publication plan namespace differs from its verified parent run")
        rows = {(row["name"], row["manifest"]["meta"]["content_key"]): row
                for row in (*reader.rows("sys-prompt"), *batch.prompt_rows)}
        for name, change in actual.items():
            if ":" in name:  # a typed key; a prompt name never holds ':'
                selector, op = typed_ops[change_kind(name)]
                evolve._check_change({"name": selector, "op": op}, parent)
                continue
            row = rows.get((name, change["to"])) if change["to"] is not None else None
            kind = (row or {}).get("manifest", {}).get("meta", {}).get("kind")
            op = "remove" if row is None else "rules" if name == "rules.general" else "doc" if kind == "doc" else "text" if kind == "text" else "spec"
            evolve._check_change({"name": name, "op": op, "slots": (row or {}).get("manifest", {}).get("meta", {}).get("slots", [])}, parent)
        if typed and any(change_kind(name) == "rules" for name in actual):
            before = parent.rules_general()
            try:
                after = reconstruct_corpus(batch.release["content_refs"]["rules"], index.get(RULE_ENTITY, {}))
            except EntityModelError as error:
                raise AuthorityError(f"publication rule references do not resolve: {error}") from error
        elif "rules.general" in actual:
            before = json.loads(parent.row("rules.general")["manifest"]["meta"]["text"])
            after = json.loads(rows[("rules.general", batch.release["refs"]["rules.general"])]["manifest"]["meta"]["text"])
        else:
            before = after = None
        if before is not None and after is not None:
            old = {row["title"]: row for row in before["rules"]}
            provenance = {record["title"]: record for record in after.get("provenance", {}).get(rules.EVOLVED, [])}
            brought = [{"title": row["title"], "dossier_qids": provenance.get(row["title"], {}).get("dossier_qids", [])}
                       for row in after["rules"] if old.get(row["title"]) != row]
            if any(not entry["dossier_qids"] for entry in brought):
                raise AuthorityError("published rule changes require their native failure dossier provenance")
            evolve._check_rule_sources([{"op": "rules", "add": brought}], facts["failures"], facts["titles"])

    def _gate_batch(self, reader: AuthorityRead, batch: PublicationBatch) -> Any:
        """Verify candidate rendering and science from a detached local projection.

        The batch's canonical Tool, rule and parameter rows and its source Capsules join the projection's
        ``native/<entity>.json`` beside the authority's own, and its release file is written in its own format
        (``rubiksql.release/2`` for closure/2), so both the head and the candidate load from the projection.
        """
        from . import evolve
        from .control import HbRenderer
        from .entity_model import CLOSURE_V2, PARAMETER_ENTITY, RULE_ENTITY, TOOL_ENTITY
        from .gate import run_gate
        from .runtime import load_release
        from .store import Store

        with tempfile.TemporaryDirectory(prefix="rubiksql-authority-gate-") as temporary:
            directory = Path(temporary)
            files = dict(reader.store_files())
            for row in batch.prompt_rows:
                files[f"objects/{row['name']}/{row['version']}.json"] = _bytes({"schema": "rubiksql.object/1", "row": row,
                    "export": {"content_key": row["manifest"]["meta"]["content_key"], "name": row["name"], "version": row["version"], "control_object_id": row["object_id"]}})
            typed = batch.release.get("rules") == CLOSURE_V2
            for entity, added in ((TOOL_ENTITY, batch.tool_rows), (RULE_ENTITY, batch.rule_rows),
                                  (PARAMETER_ENTITY, batch.parameter_rows), ("sys-capsule", batch.capsule_rows)):
                if typed and added:
                    held = {str(row["object_id"]): row for row in json.loads(files.get(f"native/{entity}.json", b"[]"))}
                    held.update({str(row["object_id"]): row for row in added})
                    files[f"native/{entity}.json"] = _bytes([held[key] for key in sorted(held)])
            files[f"releases/{batch.release['id']}.json"] = _bytes({**batch.release, "epoch": reader.epoch()["epoch"]})
            files[f"proposals/{batch.proposal['id']}.json"] = _bytes(batch.proposal)
            for name, data in files.items():
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            store = Store(package=directory)
            child = load_release(str(batch.release["id"]), stores=store, allow_retired=True)
            parent = load_release(reader.head().release, stores=store, allow_retired=True)
            report = run_gate(child, parent=parent, store=store, proposal=batch.proposal,
                              hb_render=evolve._hb_adapter(HbRenderer.for_release(self.control.ws, child)))
            if not report.passed:
                raise AuthorityError("publication gate failed: " + "; ".join(check.detail for check in report.failures()))
            return report

    def propose(self, path: Path, *, expected: AuthorityHead, actor: str, source_commit: str | None = None,
                tool_source: Path | None = None, receipts: Mapping[str, Any] | None = None,
                corpus: Any = None, root_run: Any = None, parent_run: Path | None = None,
                dbs_dir: Path | None = None, dry_run: bool = False, prepare_receipt: bool = True) -> PublishResult:
        """Compile and scientifically gate a local draft against the authority view.

        On a closure/2 head the draft's typed changes publish canonical rows: an ``op: tools`` change takes the Tool
        versions built from the Git checkout (``source_commit``) or the direct HB input (``tool_source``, exactly
        ``module_sources``, ``definitions`` and ``serializer_source``) and passes the typed tool gate; rule and
        parameter changes publish only the rule versions and parameter set that are new, beside only new Capsules.

        Once the authority has an entity-model decision, a closure/1 head is a retained historical release: it may be
        pinned, evaluated or rolled back to, but no new content builds on it.

        ``dry_run`` exercises the same compilation and complete publication checks,
        then returns before writing any authority row, event or Git commit.

        Args:
            path: Local proposal JSON and its dossier-relative documents.
            expected: Authority head whose current facts authorize the proposal.
            actor: Author recorded on the proposal and publication events.
            source_commit: Git Tool source commit, mutually exclusive with ``tool_source``.
            tool_source: Direct HB Tool authoring input, mutually exclusive with ``source_commit``.
            receipts: Required review and invariant evidence for changed Tools.
            corpus: Optional corpus supplied to the scientific gate.
            root_run: Optional root-run evidence supplied to the scientific gate.
            parent_run: Current generation's complete parent experiment.
            dbs_dir: That parent's recorded knowledge workspaces.
            dry_run: Validate without writing authority rows, events or commits.
            prepare_receipt: Retain the legacy partial-publication recovery intent.
                Atomic generation Git batches omit it: failure discards the entire
                unpushed projection, while complete publication and gate remain.

        Returns:
            Proposal and release identities and the live revision. A dry-run returns
            zero created rows and does not provide a publication receipt.

        Raises:
            AuthorityError: Stale facts, a failed gate, tool input and an ``op: tools`` change that disagree, or a
                closure/1 head after the entity-model migration.
        """
        self._require_writer(allow_pending=True)
        from . import evolve, views
        from .entity_model import CLOSURE_V2, MIGRATION_KIND

        with self.reader() as reader:
            current = reader.head().release
            typed_head = reader.row("release-entity", current)["rules"] == CLOSURE_V2
            migrations = [event["object_id"] for event in reader.events("decision") if event["payload"].get("kind") == MIGRATION_KIND]
        if migrations and not typed_head:
            raise AuthorityError(f"head {current} is a retained historical rubiksql.closure/1 release: the authority migrated to "
                                 f"typed tool, rule and parameter-set versions (entity-model decision {migrations[0]}), so "
                                 f"proposals build on a {CLOSURE_V2} release; return to one with authority rollback <typed release>")
        built = None
        if tool_source is not None or source_commit is not None:
            from ..hb_tool_entities import convert_answering, prepare_answering_rows

            built = prepare_answering_rows(source_commit=source_commit, tool_source=tool_source)
            if typed_head:
                # The schema2 builder reads exactly the authoring source; its conversion proves the Tool versions.
                built = convert_answering(built.payload, {row["object_id"]: row for row in built.capsule_rows})
        request = self._authoring_request(path, actor=actor, parent_run=parent_run, dbs_dir=dbs_dir,
            source_commit=source_commit, tool_source=tool_source, receipts=receipts, built=built,
            corpus=corpus, root_run=root_run)
        with self.reader() as reader:
            matches = [event["payload"] for event in reader.events("publish")
                       if event["payload"].get("phase") == "prepared" and event["payload"].get("request") == request]
            if len(matches) > 1:
                raise AuthorityError("raw proposal has multiple prepared authoring receipts")
            if reader.pending_publications() and not matches:
                raise AuthorityError("raw proposal differs from the exact prepared authoring request")
        if matches:
            return self._publish(_prepared_batch(matches[0]), expected=expected, actor=actor,
                                 parent_run=parent_run, dbs_dir=dbs_dir, request=request, dry_run=dry_run,
                                 prepare_receipt=prepare_receipt)
        with self.reader() as reader, views.history_store(reader, require_export=False) as store:
            self._expected(expected)
            facts = self._proposal_facts(reader, store, parent_run=parent_run, dbs_dir=dbs_dir)
            base = views.load(store, reader.head().release)
            typed = base.rules == CLOSURE_V2
            pool = [str(row["object_id"]) for row in reader.rows("proposal-entity")
                    if any(item.get("role", "base") == "base" and item.get("release") == reader.head().generation
                           for item in reader.proposal(str(row["object_id"]))["parents"])]
            body, _kinds = evolve.validate_proposal(path, target="any", release=base, pool=pool, author=actor,
                namespace=facts["namespace"], failures=facts["failures"], database_titles=facts["titles"])
            if typed and sum(change.get("op") == "tools" for change in body["changes"]) != (built is not None):
                raise AuthorityError("a closure/2 tool change is one op tools change with --source-commit or --tool-source, "
                                     "and tool authoring input needs that op tools change")
            proposal = evolve.Proposal.from_json(body, base_dir=path.parent)
            result = evolve.propose(proposal, store, corpus=corpus, root_run=root_run, tools=built if typed else None)
            if not result.report.passed:
                raise AuthorityError("proposal gate failed: " + "; ".join(check.detail for check in result.report.failures()))
            capsules: tuple[Mapping[str, Any], ...] = ()
            toolkits: tuple[Mapping[str, Any], ...] = ()
            gate: dict[str, Any] = {"scientific_gate": evolve._gate_summary(result.report)}
            if typed:
                capsules = tuple(result.capsule_rows)
            elif built is not None:
                capsules, toolkits = tuple(built.capsule_rows), tuple(built.toolkit_rows)
            if built is not None:
                gate.update(self._tool_gate(reader, built, receipts=receipts or {}, mode="hb" if tool_source is not None else "git"))
            from heavenbase.utils import sha256hash

            # A merged pool proposal's rows are already published under that proposal; they are reused as held.
            held = {(row["name"], row["manifest"]["meta"]["content_key"]) for row in reader.rows("sys-prompt")}
            changed_rows = []
            for name, key in result.refs.items():
                if result.base.refs.get(name) == key or (name, key) in held:
                    continue
                row = json.loads(json.dumps(result.child.row(name)))
                row["manifest"]["meta"]["proposal"] = result.id
                row["fingerprints"]["manifest"] = sha256hash(row["manifest"])
                changed_rows.append(row)
            release = result.store.release(result.release).data
            if typed:
                batch = PublicationBatch(tuple(changed_rows), capsules, toolkits, release, result.body, gate,
                                         tool_rows=tuple(result.tool_rows), rule_rows=tuple(result.rule_rows),
                                         parameter_rows=tuple(result.parameter_rows))
            else:
                batch = PublicationBatch(tuple(changed_rows), capsules, toolkits, release, result.body, gate)
        return self._publish(batch, expected=expected, actor=actor, parent_run=parent_run, dbs_dir=dbs_dir,
                             request=request, dry_run=dry_run, prepare_receipt=prepare_receipt)

    def _authoring_request(self, path: Path, *, actor: str, parent_run: Path | None, dbs_dir: Path | None,
                           source_commit: str | None, tool_source: Path | None, receipts: Mapping[str, Any] | None,
                           built: Any, corpus: Any, root_run: Any) -> Mapping[str, Any]:
        """Bind exact raw inputs to a durable prepared publication before any row."""
        path = Path(path).resolve()
        raw = path.read_bytes()
        body = json.loads(raw)
        if not isinstance(body, Mapping) or not isinstance(body.get("changes"), list):
            raise AuthorityError("raw proposal must hold an object with a changes list")
        files: dict[str, str] = {}
        for change in body.get("changes", []):
            if not isinstance(change, Mapping) or "text_file" not in change:
                continue
            if not isinstance(change["text_file"], str):
                raise AuthorityError("proposal document must be a dossier-relative filename")
            target = (path.parent / str(change["text_file"])).resolve()
            if not target.is_relative_to(path.parent):
                raise AuthorityError("proposal document is outside its authored dossier")
            files[str(change["text_file"])] = sha256_hex(target.read_bytes())
        from ..hb_tool_entities import TypedTools, digest, typed_bundle

        def context(value: Any) -> Any:
            return _json_value(dataclasses.asdict(value)) if dataclasses.is_dataclass(value) else _json_value(value)

        bundle = None
        if built is not None:  # typed Tool versions bind the bundle a closure/2 release pins (answering-bundle/3)
            bundle = digest(typed_bundle(built.content)) if isinstance(built, TypedTools) else digest(built.payload)
        return {"path": str(path), "sha256": sha256_hex(raw), "document_files": files, "actor": actor,
                "parent_run": str(Path(parent_run).resolve()) if parent_run is not None else None,
                "dbs_dir": str(Path(dbs_dir).resolve()) if dbs_dir is not None else None,
                "source_commit": source_commit, "tool_source": str(Path(tool_source).resolve()) if tool_source is not None else None,
                "tool_bundle": bundle,
                "receipts_sha256": sha256_hex(canon(_json_value(receipts))),
                "corpus_sha256": sha256_hex(canon(context(corpus))), "root_run_sha256": sha256_hex(canon(context(root_run)))}

    def _proposal_facts(self, reader: AuthorityRead, store: Any, *, parent_run: Path | None,
                        dbs_dir: Path | None) -> Mapping[str, Any]:
        """Read fixed parent experiment and knowledge evidence at publication.

        The parent run's snapshot must pin exactly the authoritative head: the complete release row (refs, entries,
        closure rules and, for closure/2, the typed block), every prompt row it references and every canonical Tool,
        rule and parameter row a closure/2 release references, compared row for row. Every retained role unit,
        interrupted ones included, carries its own attributed producer proof with a matching content hash, a declared
        implementation context and the actual knowledge fingerprint. An interrupted unit is retained failure evidence;
        a child never verifies a replay of it, so at outcome pairing it settles as that unit's own no-source cause.
        """
        from . import evolve, metrics, rules
        from .. import cache_lineage
        from ..lab_ingest import run_snapshot
        from ..knowledge_identity import knowledge_fingerprints, recorded_knowledge_path
        from ..llm_runtime import EMBEDDING_MODEL
        from .runtime import load_release
        from .store import Store

        if parent_run is None or dbs_dir is None:
            raise AuthorityError("raw proposal publication requires --parent-run and --dbs-dir for verified current facts")
        run = evolve.read_run(Path(parent_run))
        head = reader.head()
        if run.release != head.release or not run.ns or run.release_problem(store=store):
            raise AuthorityError("proposal parent run is not the current generation's unmixed release and namespace")
        snapshot = run_snapshot(run.path, run.manifest)
        if snapshot.release != head.release or snapshot.root != reader.epoch()["release"]:
            raise AuthorityError("proposal parent snapshot is outside the current authority generation")
        parent = load_release(snapshot.release, stores=Store(package=snapshot.path), allow_retired=True)
        authoritative = reader.row("release-entity", head.release)
        fields = ("refs", "entries", "rules", "content_refs")
        if ({key: _json_value(parent.manifest.get(key)) for key in fields}
                != {key: _json_value(authoritative.get(key)) for key in fields} or parent.refs != authoritative["refs"]):
            raise AuthorityError("proposal parent snapshot release differs from authority")
        current_rows = {(row["name"], row["manifest"]["meta"]["content_key"]): row for row in reader.rows("sys-prompt")}
        if any(semantic_row(parent.row(name)) != semantic_row(current_rows[(name, key)]) for name, key in parent.refs.items()):
            raise AuthorityError("proposal parent snapshot rows differ from authority")
        held = reader.typed_index()
        if any(semantic_row(row) != semantic_row(held.get(entity, {}).get(object_id, {}))
               for entity, rows in parent.typed_rows.items() for object_id, row in rows.items()):
            raise AuthorityError("proposal parent snapshot typed rows differ from authority")
        question_file = Path(__file__).resolve().parents[3] / evolve.EVOLUTION_SET.path
        raw = question_file.read_bytes()
        records = json.loads(raw)
        databases = sorted({str(record["db_id"]) for record in records})
        knowledge_root = Path(dbs_dir).resolve()
        recorded_root = run.manifest.get("knowledge_root")
        recorded_fingerprints = run.manifest.get("knowledge_fingerprints")
        # Transport preserves the producer's manifest bytes and path. Content,
        # including every attributed role proof below, identifies the local copy.
        recorded_knowledge_path(recorded_root)
        actual_fingerprints = knowledge_fingerprints(knowledge_root, databases, embedding_model=EMBEDDING_MODEL)
        if recorded_fingerprints != actual_fingerprints:
            raise AuthorityError("proposal knowledge fingerprints differ from the actual recorded parent knowledge")
        ids = {str(record["question_id"]) if "question_id" in record else str(record["id"]) for record in records}
        ids = {qid if qid.startswith("Q") else f"Q{int(qid):05d}" for qid in ids}
        if (run.diagnostic or len(ids) != evolve.EVOLUTION_SET.count
                or "sha256:" + sha256_hex(raw) != evolve.EVOLUTION_QUESTION_SET
                or run.identity.get("question_set") != evolve.EVOLUTION_QUESTION_SET
                or run.questions != evolve.question_set_record(ids) or set(run.rows) != ids):
            raise AuthorityError("proposal parent must retain the exact complete Mini-Dev 500 scope")
        context = evolve._implementation_contexts(run)
        if context is None or context.get("source") != "manifest" or not context.get("role_digests"):
            raise AuthorityError("proposal parent lacks recorded implementation role digests")
        from ..implementation_provenance import verify_role_records

        declared_roles = evolve._implementation_records(context["role_digests"])
        if not declared_roles:
            raise AuthorityError("proposal parent lacks valid implementation role digests")
        try:
            verify_role_records(snapshot, run.manifest["implementation"], declared_roles)
        except ValueError as error:
            raise AuthorityError(str(error)) from error
        failures = []
        for qid, row in run.rows.items():
            score = metrics.EvalRow(row).ex
            units = evolve._role_units(row)
            if score not in (0.0, 1.0) or not units or row.get("final_score_source") != "host-reexecution":
                raise AuthorityError("proposal parent has unscored rows or missing role source proof")
            for unit in units.values():
                proof = unit.get("source_proof")
                if (not cache_lineage.valid_descriptor(proof, attributed=True)
                        or proof["content_sha256"] != unit.get("content_sha") or not unit.get("source_attributed")
                        or evolve._unit_implementation(unit, context) is None
                        or proof["identity"].get("knowledge_fingerprint") != actual_fingerprints.get(row["db_id"])):
                    raise AuthorityError("proposal parent role source proof is missing or altered")
            if score == 0.0:
                failures.append(qid)
        titles: dict[str, str] = {}
        for db in databases:
            state = rules.WorkspaceRules.read(knowledge_root / db)
            for title in state.database_titles:
                titles.setdefault(title, db)
        reread = evolve.read_run(run.path)
        if reread.manifest_sha != run.manifest_sha or reread.result_files != run.result_files:
            raise AuthorityError("proposal parent evidence changed while current facts were being verified")
        return {"namespace": run.ns, "failures": failures, "titles": titles,
                "knowledge_root": str(knowledge_root), "knowledge_fingerprints": actual_fingerprints}

    def _tool_tiers(self) -> Mapping[str, str]:
        if self._git_projection is not None:
            with self.reader() as reader:
                return reader.tool_tiers()
        directory = self.path.parent
        tier_path = (directory.parent if directory.name == "control" else directory) / "tool-tiers.json"
        tiers = json.loads(tier_path.read_text(encoding="utf-8")) if tier_path.is_file() else {}
        if not isinstance(tiers, dict) or any(tier not in {"hb-executed", "host-verified"} for tier in tiers.values()):
            raise AuthorityError("authority tool tier configuration is invalid")
        return tiers

    def _tool_gate(self, reader: AuthorityRead, built: Any, *, receipts: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
        """Run the publisher's leak audit over exact native module/contract texts.

        A closure/1 head audits a ``tools.answering`` bundle (``AnsweringRows``) against its parent carrier. A
        closure/2 head audits canonical Tool versions (``TypedTools``): the payload is the candidate's typed bundle
        document (``rubiksql.answering-bundle/3``), the parent payload the head's typed bundle, definitions come from
        Tool rows and module sources from the source Capsules, so the receipts bind those two bundle digests; the
        audit base is the head's own tool gate chain (:meth:`_typed_parent_commit`).

        Raises:
            AuthorityError: Tool input that does not fit the head's format, a typed head without a tool gate chain,
                or an invalid tier configuration.
        """
        from ..tool_receipts import publish_tool_gate
        from ..hb_tool_entities import TypedTools, answering_definitions, digest as bundle_digest
        from .entity_model import CLOSURE_V2

        modules = {row["manifest"]["source"]["module"]: row["manifest"]["source"]["source"]
                   for row in built.capsule_rows if row["namespace"] == "rubiksql.answering.module"}
        head = reader.head()
        head_row = reader.row("release-entity", head.release)
        if isinstance(built, TypedTools) or head_row["rules"] == CLOSURE_V2:
            payload, parent, parent_sources, parent_contracts, definitions, serializer = self._typed_tool_inputs(
                reader, built, head_row)
            parent_commit = self._typed_parent_commit(reader, bundle_digest(parent))
        else:
            payload = built.payload
            parent = reader.answering(head.release) if "tools.answering" in head_row["refs"] else None
            parent_sources = None
            parent_contracts = None
            parent_commit = "ccc996feebc7dd92e3c2683df64d4c8ab51cc9a6"
            if parent is not None:
                capsules = {row["object_id"]: row for row in reader.rows("sys-capsule")}
                parent_sources = {ref["module"]: capsules[ref["capsule_id"]]["manifest"]["source"]["source"] for ref in parent["modules"]}
                parent_contracts = answering_definitions(parent, capsules,
                    toolkit_rows={row["object_id"]: row for row in reader.rows("sys-toolkit")})
                gates = [event["payload"] for event in reader.events("gate") if event["payload"].get("bundle_digest") == bundle_digest(parent)]
                if gates:
                    parent_commit = str(gates[-1].get("source_commit") or gates[-1]["audit_implementation_commit"])
            definitions = answering_definitions(built.payload, {row["object_id"]: row for row in built.capsule_rows},
                toolkit_rows={row["object_id"]: row for row in built.toolkit_rows})
            serializer = next(row["manifest"]["source"]["source"] for row in built.capsule_rows
                              if row["object_id"] == built.payload["serializer"]["capsule_id"])
        tiers = self._tool_tiers()
        if any(module not in modules for module in tiers):
            raise AuthorityError("authority tool tier configuration is invalid")
        return publish_tool_gate(Path(__file__).resolve().parents[3], payload, module_sources=modules,
            parent_payload=parent, parent_sources=parent_sources, parent_source_commit=parent_commit,
            receipts=receipts, mode=mode, secret_values=secret_inputs().values,
            contracts=definitions, parent_contracts=parent_contracts,
            serializer_source=serializer, tiers=tiers)

    @staticmethod
    def _typed_tool_inputs(reader: AuthorityRead, built: Any, head_row: Mapping[str, Any]
                           ) -> tuple[dict, dict, dict[str, str], list[dict], list[dict], str]:
        """The typed tool gate's candidate and parent payloads, sources and definitions over a closure/2 head.

        Returns:
            ``(typed bundle, parent typed bundle, parent module sources, parent definitions, definitions,
            serializer source)``.

        Raises:
            AuthorityError: The input is not canonical Tool versions over a closure/2 head.
        """
        from ..hb_tool_entities import TypedTools, tool_definition, typed_bundle
        from .entity_model import CLOSURE_V2, TOOL_ENTITY

        if not isinstance(built, TypedTools) or head_row["rules"] != CLOSURE_V2:
            raise AuthorityError("typed tool authoring publishes canonical Tool versions over a closure/2 head only")
        refs = head_row["content_refs"]
        capsules = {row["object_id"]: row for row in reader.rows("sys-capsule")}
        tools = {row["object_id"]: row for row in reader.rows(TOOL_ENTITY)}
        parent_sources = {ref["module"]: capsules[ref["capsule_id"]]["manifest"]["source"]["source"]
                          for ref in refs["loader"]["modules"]}
        parent_contracts = [tool_definition(tools[ref["object_id"]]) for ref in refs["tools"]]
        serializer = next(row["manifest"]["source"]["source"] for row in built.capsule_rows
                          if row["object_id"] == built.content["loader"]["serializer"]["capsule_id"])
        return (typed_bundle(built.content), typed_bundle(refs), parent_sources, parent_contracts,
                [dict(record) for record in built.definitions], serializer)

    @staticmethod
    def _typed_parent_commit(reader: AuthorityRead, bundle: str) -> str:
        """The trusted audit base of a closure/2 head's tools: the source commit of the tool gate that admitted them.

        Only ``rubiksql.tool-gate/1`` gates count. A tool gate binding the head's typed bundle admits it directly; a
        typed bundle the entity-model migration admitted is attested only by its ``rubiksql.entity-model-gate/1``
        gate, whose ``legacy_bundle_digest`` names the legacy bundle of the same executable source, and that bundle's
        tool gate is the audit base. There is no default base.

        Args:
            reader: The writer's current view.
            bundle: Digest of the head's typed bundle.

        Returns:
            The latest admitting tool gate's ``source_commit`` (or its ``audit_implementation_commit``).

        Raises:
            AuthorityError: No tool gate admits the bundle, directly or through its migration gate.
        """
        from .entity_model import SCHEMA_ENTITY_GATE
        from ..hb_tool_entities import digest, typed_bundle

        state = reader.current_state()
        if state is not None:
            adopted = reader.row("release-entity", state["release"])
            if digest(typed_bundle(adopted["content_refs"])) == bundle:
                gate = state["source_admission"]["gate"]
                return str(gate.get("source_commit") or gate["audit_implementation_commit"])

        payloads = [event["payload"] for event in reader.events("gate")]
        digests = {bundle}
        if not any(payload.get("schema") == "rubiksql.tool-gate/1" and payload.get("bundle_digest") == bundle for payload in payloads):
            digests = {payload["legacy_bundle_digest"] for payload in payloads
                       if payload.get("schema") == SCHEMA_ENTITY_GATE and payload.get("bundle_digest") == bundle
                       and isinstance(payload.get("legacy_bundle_digest"), str)}
        gates = [payload for payload in payloads
                 if payload.get("schema") == "rubiksql.tool-gate/1" and payload.get("bundle_digest") in digests]
        commit = (gates[-1].get("source_commit") or gates[-1].get("audit_implementation_commit")) if gates else None
        if not isinstance(commit, str) or not commit:
            raise AuthorityError("the head's typed tools have no tool gate chain to a trusted audit base; "
                                 "a typed tool change cannot be audited against them")
        return commit

    def gate(self, target: str, *, expected: AuthorityHead, actor: str, record: str | None = None,
             parent: str | None = None, corpus: Any = None, root_run: Any = None) -> Any:
        """Evaluate the scientific gate and optionally append its immutable receipt."""
        self._require_writer()
        self._expected(expected)
        with self.reader() as reader:
            report = reader.gate(target, parent=parent, corpus=corpus, root_run=root_run)
        if record is not None:
            self._event("gate", target, {"schema": "rubiksql.scientific-gate/1", "label": record,
                "release": report.release, "passed": report.passed,
                "checks": {check.name: check.status for check in report.checks}}, actor)
        return report

    def decide(self, kind: str, target: str, *, expected: AuthorityHead, actor: str,
               reason: str, payload: Mapping[str, Any] | None = None) -> DecisionResult:
        """Validate and append one decision; only pointer kinds move generation.

        Args:
            kind: Cutover, promote, rollback or an immutable lifecycle decision.
            target: Exact release, proposal or name:version reference.
            expected: Current pointer facts, checked under the writer lock.
            actor: Recorded decision actor.
            reason: Human-readable reason.
            payload: Outcome/override/user-instruction evidence and dry-run flag.
                For explicit no-harm adoption, set ``adopt_no_harm=True`` and supply a nonempty
                ``rationale`` explaining the sound shared mechanism and its supporting evidence.
                Only an ``adoptable`` outcome qualifies; a verdict override cannot be combined.

        Returns:
            Proposed or committed pointer and live revision.
        """
        self._require_writer()
        from . import evolve, views

        extra = dict(payload or {})
        adopt = _adoption_requested(kind, extra)
        if kind == "abandon" and not reason.strip():
            reason = "The author withdrew this candidate before promotion."
        if kind == M.MIGRATION_KIND or kind not in DECISION_KINDS or not reason.strip():
            raise AuthorityError("decision kind and nonempty reason are required")
        self._expected(expected, require_export=kind != "cutover")
        if kind in POINTER_KINDS and kind != "cutover" and expected.decision is None:
            raise AuthorityError("the first pointer decision must be cutover")
        if kind in evolve.HUMAN_ONLY_VERBS or extra.get("override_verdict"):
            evolve.ensure_human(kind if not extra.get("override_verdict") else "promote --override-verdict")
        if kind == "cutover" and not str(extra.get("user_instruction", "")).strip():
            raise AuthorityError("cutover needs the verbatim user instruction")
        with self.reader() as reader, views.history_store(reader, require_export=False) as store:
            rid = reader.resolve(target) if kind in POINTER_KINDS | {"reject", "abandon", "retire", "unretire"} else expected.release
            if kind in POINTER_KINDS:
                if rid == expected.release:
                    raise AuthorityError("pointer target already is the current release")
                if not _carries_tools(reader.row("release-entity", rid)):
                    raise AuthorityError("pointer target has no executable answering bundle")
                selected_row = reader.row("release-entity", rid)
                if (selected_row["rules"] != M.CLOSURE_V2
                        and reader.answering(rid).get("schema") == "rubiksql.answering-bundle/1"):
                    raise AuthorityError("schema1 Toolkit execution is permanently retired")
                if store.is_retired(rid) or evolve._revoked_rows(store, rid):
                    raise AuthorityError("pointer target carries retired or revoked content")
            if kind == "cutover":
                if reader.events("decision") or reader.events("outcome") or reader.events("eval-attach") or expected.decision:
                    raise AuthorityError("cutover must be the sole first pointer decision at 0/10")
                proposal = reader.proposal(target)
                if proposal.get("origin") != "cutover":
                    raise AuthorityError("cutover target is not the unique cutover proposal")
                runs = self.path.parent.parent / "runs"
                for manifest in runs.rglob(".prompt_manifest.json") if runs.exists() else ():
                    if json.loads(manifest.read_text(encoding="utf-8")).get("release") == expected.release:
                        raise AuthorityError("cutover root already has a measured run")
                gate = [row["payload"] for row in reader.events("gate")
                        if row["subject"] == target and row["payload"].get("schema") == "rubiksql.tool-gate/1"]
                if not gate or gate[-1].get("source_commit") != extra.get("deployed_commit"):
                    raise AuthorityError("cutover deployed commit differs from tool gate source")
            if kind == "promote":
                body = reader.proposal(target)
                if next(item["release"] for item in body["parents"] if item.get("role", "base") == "base") != expected.release:
                    raise AuthorityError("promotion proposal base is stale")
                docs = [row["payload"] for row in reader.events("outcome") if row["subject"] == target]
                chosen = next((doc for doc in docs if doc["id"] == extra.get("outcome")), None) if extra.get("outcome") else evolve.latest_verdict(docs, expected.generation)
                if chosen is None and not extra.get("override_verdict"):
                    raise AuthorityError("promotion needs a recorded outcome in this generation")
                if chosen is not None and (chosen.get("generation") != expected.generation or chosen.get("release") != rid):
                    raise AuthorityError("promotion outcome does not measure this pointer move")
                if chosen is not None and (chosen.get("coverage") or {}).get("question_set") != evolve.EVOLUTION_QUESTION_SET:
                    raise AuthorityError("new adoption requires a Mini-Dev 500 outcome")
                if adopt and chosen is not None and any(row["payload"].get("kind") == "promote"
                        and row["payload"].get("outcome") == chosen["id"] for row in reader.events("decision")):
                    raise AuthorityError("promotion outcome was already acted on")
                if not extra.get("override_verdict"):
                    required_verdict = "adoptable" if adopt else "promotable"
                    if chosen.get("verdict") != required_verdict or evolve.verify_outcome(chosen, plan=evolve.plan_view(body)):
                        raise AuthorityError("promotion outcome fails reproducible verdict checks")
                    if evolve.deep_verify_outcome(chosen, store=store, plan=evolve.plan_view(body)):
                        raise AuthorityError("promotion run-source integrity verification failed")
                    if evolve.promotion_veto(store, rid) or evolve._withdrawal(store, target):
                        raise AuthorityError("promotion target was rejected, rolled back or abandoned")
                extra["outcome"] = chosen["id"] if chosen is not None else None
            if kind == "abandon":
                body = reader.proposal(target)
                if actor != body.get("author"):
                    evolve.ensure_human("abandon")
            if kind == "unretire":
                target_row = reader.row("release-entity", rid)
                if (target_row["rules"] != M.CLOSURE_V2 and "tools.answering" in target_row["refs"]
                        and reader.answering(rid).get("schema") == "rubiksql.answering-bundle/1"):
                    raise AuthorityError("schema1 Toolkit execution cannot be unretired")
            if kind == "retire" and rid in {expected.release, reader.epoch()["release"]}:
                raise AuthorityError("cannot retire the current release or epoch root")
            if kind in {"revoke", "unrevoke"}:
                name, version, key = evolve._row_ref(target, store)
                if kind == "revoke" and any(key in reader.row("release-entity", release)["refs"].values() for release in {expected.release, str(reader.epoch()["release"])}):
                    raise AuthorityError("cannot revoke content of the current release or epoch root")
                extra.update(name=name, version=version, content_key=key)
        decision_payload = {"schema": "rubiksql.decision/1", "kind": kind, "previous_decision": expected.decision,
            "target": target, "generation": rid if kind in POINTER_KINDS else expected.generation,
            "reason": reason, **{key: value for key, value in extra.items() if key != "dry_run"}}
        if kind in POINTER_KINDS:
            decision_payload.update({"from": expected.release, "to": rid})
        else:
            decision_payload["release"] = rid
        did = None if extra.get("dry_run") else self._event("decision", target, decision_payload, actor)
        return DecisionResult(did, expected.release, rid if kind in POINTER_KINDS else expected.release,
                              rid if kind in POINTER_KINDS else expected.generation, self.head().revision)

    def record_outcome(self, proposal: str, child: Path, *, parent: Path | None, expected: AuthorityHead,
                       actor: str, write: bool = True, allow_over_cap: bool = False,
                       search: Sequence[Path] = (), decision_rationale: str = "", auto_promote: bool = True,
                       computation: OutcomeComputation | None = None,
                       registration: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        """Record a proposal outcome or descriptive adopted-current baseline.

        The exact current release is the only non-proposal target.
        Its descriptive outcome does not reserve an attempt or promote content.
        """
        self._require_writer()
        self._expected(expected)
        from . import evolve, views

        with self.reader() as reader, views.history_store(reader, require_export=False) as store:
            result = evolve.authority_outcome(self, reader, proposal, child, store, parent_run_dir=parent,
                search=search, decision_rationale=decision_rationale, write=write, allow_over_cap=allow_over_cap,
                actor=actor, computation=computation, registration=registration)
        if write and result.get("proposal") is not None and result["verdict"] == "promotable" and auto_promote:
            decision = self.decide("promote", proposal, expected=self.head(), actor=evolve.AUTO_ACTOR,
                                   reason=decision_rationale or "host-scored EX passed the promotion rule", payload={"outcome": result["id"]})
            return {**result, "promotion": dataclasses.asdict(decision)}
        return result

    def retire_refresh(self, *, expected: AuthorityHead, request_id: str,
                       user_instruction: str, actor: str = "agent:META") -> Mapping[str, Any]:
        """Atomically detach the existing current state from obsolete provenance.

        Args:
            expected (AuthorityHead): Exact exported preimage and full revision.
            request_id (str): Unique cleanup request, reused only for exact replay.
            user_instruction (str): Verbatim authorization for the deletion.
            actor (str): Recorded author of the cleanup.

        Returns:
            Mapping[str, Any]: Completed anchor identity, deletion proof and fresh export.

        Raises:
            AuthorityError: The preimage, reference closure or replay facts differ.
        """
        from .authority_state import apply_retirement, plan_retirement

        if (not self._writer_active or self._writer_thread != threading.get_ident()
                or self.path not in getattr(_LOCK_STATE, "active", set())):
            raise AuthorityError("refresh retirement requires the active instance-bound writer context")
        if (not isinstance(request_id, str) or not request_id.strip()
                or not isinstance(user_instruction, str) or not user_instruction.strip()
                or not isinstance(actor, str) or not actor.strip()):
            raise AuthorityError("refresh retirement needs exact request, actor and authorization")
        with self.reader() as reader:
            state = reader.current_state()
            if state is not None and state.get("schema") == "rubiksql.fresh-epoch/1":
                raise AuthorityError("fresh source admission has no obsolete refresh history to retire")
            if state is None:
                plan = plan_retirement(reader, expected, request_id=request_id,
                                       user_instruction=user_instruction, actor=actor)
            else:
                if any(state.get(key) != value for key, value in
                       {"request_id": request_id, "user_instruction": user_instruction, "actor": actor}.items()):
                    raise AuthorityError("refresh retirement already completed with another exact request")
                original = {key: state["expected"][key] for key in ("release", "generation", "decision", "revision")}
                if self._tool_expected(expected) not in (original, self._tool_expected(reader.head())):
                    raise AuthorityError("refresh retirement replay expected facts are stale")
                plan = None
        if plan is not None:
            apply_retirement(self, plan)
        self._retire_refresh_only = False
        with self.reader() as reader:
            checked = reader.verify(require_export=False)
            if not checked.ok:
                raise AuthorityError("retired current state verification refused: " + "; ".join(checked.violations))
            state = reader.current_state()
            if state is None:
                raise AuthorityError("refresh retirement has no completed current-state anchor")
        exported = None
        if not self.head().exported:
            exported = self.export(export_dir=default_export_dir(self.binding), secret_inputs=secret_inputs())
        with self.reader() as reader:
            checked = reader.verify()
            if not checked.ok:
                raise AuthorityError("refresh retirement export verification refused: " + "; ".join(checked.violations))
        return {"id": state["id"], "event_id": state["id"], "operation": "retire-refresh", "phase": "completed",
                "release": state["release"], "generation": state["generation"], "removed": state["removed"],
                "revision": dataclasses.asdict(self.head().revision), "export_commit": self.head().export_commit,
                "export": dataclasses.asdict(exported) if exported else {"commit": self.head().export_commit}}

    def export(self, *, export_dir: Path, secret_inputs: SecretInputs) -> ExportResult:
        """Commit a complete deterministic export, then append its self-excluded receipt.

        Every new export uses format 4: HEAD ``rubiksql.authority-head/4``, STATE
        ``rubiksql.workspace-state/2`` over the nine version sets and seven evidence sets, and
        receipt ``rubiksql.authority-export/4``.

        Args:
            export_dir: Authority-owned Git checkout on branch authority/v0.
            secret_inputs: In-memory configured keys used only by the scanner.

        Returns:
            Export commit and honestly distinct covered/live revisions.
        """
        self._require_writer(allow_pending=True, allow_tools=True)
        if self._git_projection is not None:
            return self._git_projection.publish(secret_inputs)
        directory = Path(export_dir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / ".git").exists():
            _git(directory, "init", "-b", "authority/v0")
            _git(directory, "config", "user.name", "RubikSQL authority")
            _git(directory, "config", "user.email", "authority@rubiksql.local")
        if _git(directory, "branch", "--show-current") != "authority/v0":
            raise AuthorityError("export checkout must use authority/v0")
        with self.reader() as reader:
            verified = reader.verify(require_export=False)
            if not verified.ok:
                raise AuthorityError("cannot export altered authority: " + "; ".join(verified.violations))
            current = reader.head()
            files = dict(reader.workspace_files(tools_release=current.release, export_format=EXPORT_FORMAT))
            head = reader.covered_head(files, export_format=EXPORT_FORMAT)
            files["HEAD.json"] = _bytes(head)
            scan_secrets(files, secret_inputs)
        previous = _git(directory, "ls-files").splitlines()
        for relative in previous:
            if relative not in files:
                path = directory / relative
                if not path.resolve().is_relative_to(directory):
                    raise AuthorityError("export tracked path escapes checkout")
                path.unlink(missing_ok=True)
        for relative, value in files.items():
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() == value:
                continue
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_bytes(value)
            os.replace(temporary, path)
        _git(directory, "add", "--all")
        if _git(directory, "diff", "--cached", "--name-only"):
            _git(directory, "commit", "-m", f"Export authority {current.release} at event {current.revision.event_count}")
        commit = _git(directory, "rev-parse", "HEAD")
        _read_export(directory)
        self._event("export", current.release, {"schema": EXPORT_FORMATS[EXPORT_FORMAT][1], "export_commit": commit,
            "head_sha256": sha256_hex(files["HEAD.json"]), "covered_revision": dataclasses.asdict(current.revision),
            "covered_non_export_ids": head["covered_non_export_ids"], "rows_digest": head["rows_digest"],
            "file_map_digest": head["file_map_digest"], "workspace_digest": head["workspace_digest"],
            "workspace_schemas_digest": head["workspace_schemas_digest"],
            "version_rows_digest": head["version_rows_digest"]}, "authority:export")
        self._batch_dirty = False
        return ExportResult(commit, sha256_hex(files["HEAD.json"]), current.revision, self.head().revision)

    def backup(self, *, into: Path) -> Mapping[str, str]:
        """Online-backup without opening or initializing any HB workspace."""
        if isinstance(self.binding, GitAuthorityBinding):
            raise AuthorityError("Git authority backups are its exact committed exports")
        return backup_authority(self.path, binding=self.binding, into=into)

    @staticmethod
    def _tool_expected(expected: AuthorityHead) -> dict[str, Any]:
        return {"release": expected.release, "generation": expected.generation, "decision": expected.decision,
                "revision": dataclasses.asdict(expected.revision)}

    def _tool_retry(self, reader: AuthorityRead, request: Mapping[str, Any], expected: AuthorityHead
                    ) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
        from ..tools_catalog import DEPLOYMENT_SCHEMA, PUBLICATION_SCHEMA, catalog_history

        publications, deployments, pending = catalog_history(reader, allow_pending=True)
        completed = {**publications, **deployments}
        matches = [event for event in reader.events("register")
                   if event["payload"].get("schema") in {PUBLICATION_SCHEMA, DEPLOYMENT_SCHEMA}
                   and event["payload"].get("phase") == "prepared"
                   and event["payload"].get("request", {}).get("request_id") == request["request_id"]]
        if len(matches) > 1:
            raise AuthorityError("tool request has multiple preparations")
        if not matches:
            if pending:
                raise AuthorityError("tool request differs from the exact pending registration")
            self._expected(expected, full_revision=True)
            return None, None
        prepared = matches[0]
        held = prepared["payload"]["request"]
        if ({key: value for key, value in held.items() if key != "expected"}
                != {key: value for key, value in request.items() if key != "expected"}
                or self._tool_expected(expected) not in (held["expected"], self._tool_expected(reader.head()))):
            raise AuthorityError("tool request differs from its exact immutable preparation")
        identifier = prepared["payload"]["id"]
        if identifier in completed:
            if pending:
                raise AuthorityError("complete the pending tool registration before replaying another request")
            return prepared, completed[identifier]
        if set(pending) != {identifier}:
            raise AuthorityError("tool request is not the unique interrupted registration")
        return prepared, None

    def _complete_tool_registration(self, document: Mapping[str, Any], *, prepared: Mapping[str, Any] | None,
                                    tool_rows: Sequence[Mapping[str, Any]] = (),
                                    capsule_rows: Sequence[Mapping[str, Any]] = ()) -> Mapping[str, Any]:
        from copy import deepcopy
        from ..tools_catalog import _check_document

        if prepared is None:
            payload = {**deepcopy(dict(document)), "phase": "prepared", "tool_rows": deepcopy(list(tool_rows)),
                       "capsule_rows": deepcopy(list(capsule_rows))}
            _check_document(payload)
            preparation = self._event("register", str(document["id"]),
                payload, str(document["request"]["actor"]))
        else:
            preparation = str(prepared["object_id"])
            tool_rows, capsule_rows = prepared["payload"]["tool_rows"], prepared["payload"]["capsule_rows"]
        for entity, rows in (("sys-capsule", capsule_rows), (M.TOOL_ENTITY, tool_rows)):
            for row in rows:
                self._put(entity, row)
        completion = self._event("register", str(document["id"]),
            {**deepcopy(dict(document)), "phase": "completed", "prepared_event": preparation}, str(document["request"]["actor"]))
        return self._tool_result(document, completion)

    def _tool_result(self, document: Mapping[str, Any], completion: str) -> Mapping[str, Any]:
        head = self.head()
        exported = None
        if not head.exported:
            exported = (self.commit() if self._git_projection is not None else
                        self.export(export_dir=default_export_dir(self.binding), secret_inputs=secret_inputs()))
        with self.reader() as reader:
            verified = reader.verify()
            if not verified.ok:
                raise AuthorityError("tool registration export verification refused: " + "; ".join(verified.violations))
        return {"id": document["id"], "phase": "completed", "event_id": completion,
                "publication_id": document["id"] if document["schema"] == "rubiksql.tool-publication/1"
                                  else document["publication_id"],
                "content_digest": document.get("content_digest", document.get("publication_digest")),
                "revision": dataclasses.asdict(self.head().revision), "export_commit": self.head().export_commit,
                "export": dataclasses.asdict(exported) if exported else {"commit": self.head().export_commit}}

    def publish_tools(self, *, expected: AuthorityHead, request_id: str, receipts: Path | Mapping[str, Any],
                      source_commit: str | None = None, tool_source: Path | None = None,
                      actor: str = "agent:META") -> Mapping[str, Any]:
        """Publish a source-audited immutable catalog and its fresh authority export.

        Args:
            expected (AuthorityHead): Exact full revision and BIRD pointer used to prepare.
            request_id (str): Durable unique request identity, reused only for exact retries.
            receipts (Path | Mapping[str, Any]): Independent designated-model review and
                credential-free, active offline invariant evidence.
            source_commit (str | None): Exact deployed Git HEAD; omitted uses HEAD.
            tool_source (Path | None): Reviewed direct-HB source; mutually exclusive with Git.
            actor (str): Author identity recorded in the registration.

        Returns:
            Mapping[str, Any]: Exact publication, changed authority revision and verified export.

        Raises:
            AuthorityError: Stale facts, conflicting retry, invalid source or audit evidence.
        """
        from copy import deepcopy
        from ..hb_tool_entities import prepare_catalog_rows, tool_definition, typed_bundle, validate_typed_tools
        from ..tools_catalog import PUBLICATION_SCHEMA, _document, catalog_parent, registration_id
        from ..tool_receipts import publish_tool_gate

        if not isinstance(request_id, str) or not request_id.strip() or not actor.strip():
            raise AuthorityError("tool publication requires an exact request identity and actor")
        if source_commit is not None and tool_source is not None:
            raise AuthorityError("tool publication uses exactly one Git or direct-HB source")
        self._require_writer(allow_tools=True)
        built = prepare_catalog_rows(source_commit=source_commit, tool_source=tool_source)
        tool_rows = {row["object_id"]: row for row in built.tool_rows}
        capsule_rows = {row["object_id"]: row for row in built.capsule_rows}
        definitions = validate_typed_tools(built.content, tool_rows=tool_rows, capsule_rows=capsule_rows)
        receipt_body = deepcopy(dict(receipts)) if isinstance(receipts, Mapping) else json.loads(Path(receipts).read_text(encoding="utf-8"))
        checkout = Path(__file__).resolve().parents[3]
        deployed = _git(checkout, "rev-parse", "HEAD")
        source = {"mode": "hb" if tool_source is not None else "git",
                  "source_commit": None if tool_source is not None else deployed, "deployed_commit": deployed}
        request = {"operation": "publish-tools", "request_id": request_id, "actor": actor,
                   "expected": self._tool_expected(expected), "content_digest": sha256_hex(canon(built.content)),
                   "receipts_sha256": sha256_hex(canon(receipt_body)), "source": source}
        with self.reader() as reader:
            prepared, completed = self._tool_retry(reader, request, expected)
            if completed is not None:
                document = _document(completed)
                completion = str(completed["event"]["object_id"])
            elif prepared is not None:
                document, completion = _document(prepared["payload"]), None
            else:
                provenance, parent_content, parent_commit = catalog_parent(reader)
                capsules = {row["object_id"]: row for row in reader.rows("sys-capsule")}
                parent_sources = {ref["module"]: capsules[ref["capsule_id"]]["manifest"]["source"]["source"]
                                  for ref in parent_content["loader"]["modules"]}
                parent_definitions = [tool_definition(reader.row(M.TOOL_ENTITY, ref["object_id"]))
                                      for ref in parent_content["tools"]]
                module_sources = {ref["module"]: capsule_rows[ref["capsule_id"]]["manifest"]["source"]["source"]
                                  for ref in built.content["loader"]["modules"]}
                serializer = capsule_rows[built.content["loader"]["serializer"]["capsule_id"]]["manifest"]["source"]["source"]
                tiers = self._tool_tiers()
                if any(module not in module_sources for module in tiers):
                    raise AuthorityError("authority tool tier configuration is invalid")
                gate = publish_tool_gate(checkout, typed_bundle(built.content), module_sources=module_sources,
                    parent_payload=typed_bundle(parent_content), parent_sources=parent_sources,
                    parent_source_commit=parent_commit, receipts=receipt_body, mode=source["mode"],
                    secret_values=secret_inputs().values, tiers=tiers, contracts=definitions,
                    parent_contracts=parent_definitions, serializer_source=serializer, catalog_parent=provenance)
                document = {"schema": PUBLICATION_SCHEMA, "request": request, "content": deepcopy(dict(built.content)),
                            "content_digest": request["content_digest"], "source": source, "gate": gate,
                            "parent_provenance": provenance}
                document["id"] = registration_id(document)
                completion = None
        if completion is not None:
            return self._tool_result(document, completion)
        return self._complete_tool_registration(document, prepared=prepared,
                                               tool_rows=built.tool_rows, capsule_rows=built.capsule_rows)

    def deploy_tools(self, *, consumer: str, publication_id: str, expected: AuthorityHead,
                     previous_deployment: str | None, request_id: str, actor: str = "agent:META") -> Mapping[str, Any]:
        """Select an immutable catalog publication for one independent consumer.

        Args:
            consumer (str): ``assistant`` or ``meta``; no BIRD activation is available.
            publication_id (str): Exact completed and source-verified publication.
            expected (AuthorityHead): Full authority revision and pointer facts.
            previous_deployment (str | None): Exact previous consumer pin, including None initially.
            request_id (str): Durable unique request identity for exact retry recovery.
            actor (str): Deployment actor recorded in the registration.

        Returns:
            Mapping[str, Any]: Exact deployment, changed authority revision and fresh export.

        Raises:
            AuthorityError: Source, role, previous pin, revision or retry identity differs.
        """
        from copy import deepcopy
        from ..hb_tool_entities import validate_typed_tools
        from ..tools_catalog import CONSUMERS, DEPLOYMENT_SCHEMA, _document, catalog_history, registration_id, selected_refs

        if consumer not in CONSUMERS or not isinstance(request_id, str) or not request_id.strip() or not actor.strip():
            raise AuthorityError("tool deployment needs assistant/meta, an exact request identity and actor")
        self._require_writer(allow_tools=True)
        request = {"operation": "deploy-tools", "request_id": request_id, "actor": actor,
                   "expected": self._tool_expected(expected), "consumer": consumer,
                   "publication_id": publication_id, "previous_deployment": previous_deployment}
        with self.reader() as reader:
            prepared, completed = self._tool_retry(reader, request, expected)
            if completed is not None:
                document, completion = _document(completed), str(completed["event"]["object_id"])
            elif prepared is not None:
                document, completion = _document(prepared["payload"]), None
            else:
                publications, deployments, _pending = catalog_history(reader)
                publication = publications.get(publication_id)
                if publication is None:
                    raise AuthorityError("tool deployment needs an exact completed publication")
                previous = [record["id"] for record in deployments.values() if record["consumer"] == consumer]
                if previous_deployment != (previous[-1] if previous else None):
                    raise AuthorityError("expected previous tool deployment is stale")
                definitions = validate_typed_tools(publication["content"],
                    tool_rows={row["object_id"]: row for row in reader.rows(M.TOOL_ENTITY)},
                    capsule_rows={row["object_id"]: row for row in reader.rows("sys-capsule")})
                document = {"schema": DEPLOYMENT_SCHEMA, "request": request, "consumer": consumer,
                            "publication_id": publication_id, "publication_digest": publication["content_digest"],
                            "tools": selected_refs(publication["content"], definitions, consumer),
                            "previous_deployment": previous_deployment}
                document["id"] = registration_id(document)
                completion = None
        if completion is not None:
            return self._tool_result(document, completion)
        return self._complete_tool_registration(deepcopy(document), prepared=prepared)

    def cutover(self, *, user_instruction: str, through: str = "decision", expected: AuthorityHead | None = None,
                receipts: Path | None = None) -> Mapping[str, Any]:
        """Resume marker, exact R1 publication and first pointer decision at 0/10."""
        if through not in {"marker", "r1", "decision"} or not user_instruction.strip():
            raise AuthorityError("cutover needs a phase and the verbatim user instruction")
        if expected is not None:
            self._expected(expected, require_export=False)
        with self.reader() as reader:
            if any(event["payload"].get("kind") == "cutover" for event in reader.events("decision")):
                raise AuthorityError("cutover already committed")
            marked = bool(reader.events("authority"))
            if marked and reader.marker().get("user_instruction") != user_instruction:
                raise AuthorityError("cutover resume instruction differs from its recorded authorization")
        if not marked:
            # The only frozen-store read is the explicitly authorized cutover
            # inheritance check; runtime and ordinary evolution never use it.
            package = Path(__file__).with_name("store")
            marker = json.loads((package / "V0.json").read_text(encoding="utf-8"))
            root = json.loads((package / "releases" / f"{marker['release']}.json").read_text(encoding="utf-8"))
            epoch = {key: marker[key] for key in ("label", "release", "source_release", "source_commit", "delivery_commit")}
            epoch["epoch"] = root["epoch"]
            with self.reader() as reader:
                native = reader.row("release-entity", epoch["release"])
                if native["refs"] != root["refs"] or len(native["refs"]) != 34:
                    raise AuthorityError("seeded root inheritance differs from frozen content")
                for row in reader.rows("sys-prompt"):
                    if root["refs"].get(row["name"]) != row["manifest"]["meta"]["content_key"]:
                        raise AuthorityError("seeded prompt inheritance differs")
            self.mark(epoch, user_instruction=user_instruction)
        with self.reader() as reader:
            epoch = reader.epoch()
            pending = reader.pending_publications()
            unfinished = [payload for payload in pending.values() if _prepared_batch(payload).proposal.get("origin") == "cutover"]
            existing = [row for row in reader.rows("proposal-entity") if row["object_id"] not in pending
                        and reader.proposal(str(row["object_id"])).get("origin") == "cutover"]
        if through == "marker":
            return {"phase": "marker", "root": epoch["release"], "r1": None, "decision": None, "revision": dataclasses.asdict(self.head().revision), "export_commit": None}
        if unfinished:
            if len(unfinished) != 1 or existing or len(pending) != 1:
                raise AuthorityError("cutover has multiple candidate publications")
            batch = _prepared_batch(unfinished[0])
            self.publish(batch, expected=self.head(), actor=str(batch.proposal["author"]), user_instruction=user_instruction)
            pid, rid = str(batch.proposal["id"]), str(batch.release["id"])
        elif not existing:
            from ..hb_tool_entities import prepare_answering_rows
            from . import entries, evolve
            from .control import TextDraft, capture_text
            from .store import proposal_id, release_id

            built = prepare_answering_rows()
            payload = built.payload
            draft = TextDraft("tools.answering", canon(payload), [], "Pinned native answering tool bundle", ["rubiksql", "owner:tools", "kind:text"],
                              "Carry unchanged prompt content with HB-owned answering module versions", origin="human")
            row = capture_text(self.control, draft).to_dict()
            row["manifest"]["reason"] = draft.reason
            from .integrity import native_hash

            row["fingerprints"]["manifest"] = native_hash(row["manifest"])
            with self.reader() as reader:
                root_row = reader.row("release-entity", str(epoch["release"]))
                refs = {**root_row["refs"], "tools.answering": draft.content_key}
                names = sorted(set(root_row["entries"]) | ({"tools.answering"} if "tools.answering" in entries.ENTRIES else set()))
                rid = release_id(refs, names, root_row["rules"], epoch=epoch["epoch"])
            plan = evolve.Proposal(parents=({"release": epoch["release"], "role": "base"},), author="agent:META/user-2026-10-09-hb-cutover", changes=(), reason=draft.reason).frozen_plan()
            plan["control"]["namespace"] = f"v0-{epoch['release']}"
            body = {"schema": "rubiksql.proposal/1", "parents": [{"release": epoch["release"], "role": "base"}], "release": rid,
                    "changes": {"tools.answering": {"from": None, "to": draft.content_key}}, "reason": draft.reason,
                    "hypothesis": {"claim": "Representation change only; no EX gain claimed"},
                    "hypothesis_digest": sha256_hex(canon({"claim": "Representation change only; no EX gain claimed"})),
                    "plan": plan, "plan_digest": sha256_hex(canon(plan)), "author": "agent:META/user-2026-10-09-hb-cutover",
                    "origin": "cutover", "created_at": time.time()}
            body["id"] = proposal_id(body)
            receipt_path = Path(receipts) if receipts is not None else self.path.parent / "migration-tool-receipts.json"
            if not receipt_path.is_file():
                raise AuthorityError("cutover requires migration-tool-receipts.json from independent review and invariant validation")
            receipt_body = json.loads(receipt_path.read_text(encoding="utf-8"))
            with self.reader() as reader:
                gate = self._tool_gate(reader, built, receipts=receipt_body, mode="git")
            release = {"schema": "rubiksql.release/1", "id": rid, "refs": refs, "entries": names, "rules": root_row["rules"], "origin": "cutover", "epoch": epoch["epoch"]}
            self.publish(PublicationBatch((row,), tuple(built.capsule_rows), tuple(built.toolkit_rows), release, body, gate),
                         expected=self.head(), actor=body["author"], user_instruction=user_instruction)
            pid = body["id"]
        else:
            if len(existing) != 1:
                raise AuthorityError("cutover has multiple candidate publications")
            pid, rid = str(existing[0]["object_id"]), str(existing[0]["release"])
            with self.reader() as reader:
                prepared = [event["payload"] for event in reader.events("publish")
                            if event["subject"] == pid and event["payload"].get("phase") == "prepared"]
            if prepared:
                if len(prepared) != 1:
                    raise AuthorityError("cutover has multiple prepared batches")
                batch = _prepared_batch(prepared[0])
                self.publish(batch, expected=self.head(), actor=str(batch.proposal["author"]), user_instruction=user_instruction)
        if through == "r1":
            return {"phase": "r1", "root": epoch["release"], "r1": rid, "decision": None, "revision": dataclasses.asdict(self.head().revision), "export_commit": None}
        commit = _git(Path(__file__).resolve().parents[3], "rev-parse", "HEAD")
        result = self.decide("cutover", pid, expected=self.head(), actor="agent:META/user-2026-10-09-hb-cutover",
                             reason="HB authority cutover at 0/10", payload={"user_instruction": user_instruction, "deployed_commit": commit})
        return {"phase": "decision", "root": epoch["release"], "r1": rid, "decision": result.decision,
                "revision": dataclasses.asdict(result.revision), "export_commit": None}


def backup_authority(db: Path, *, binding: AuthorityBinding, into: Path) -> Mapping[str, str]:
    """Back up the complete bound workspace with SQLite online APIs.

    Args:
        db: Canonical database named in the package binding.
        binding: Instance/path fence checked before acquiring the authority lock.
        into: New private backup directory.

    Returns:
        Exact relative file SHA-256 map. No source workspace is initialized.

    Raises:
        AuthorityError: Wrong instance/path or existing destination.
    """
    source_db = Path(db).resolve()
    if source_db != binding.db_path or B.metadata_instance_id() != binding.instance_id:
        raise AuthorityError("authority backup instance or database binding differs")
    destination = Path(into).resolve()
    with authority_lock(source_db):
        if destination.exists():
            raise AuthorityError("backup destination must be new")
        destination.mkdir(parents=True, mode=0o700)
        with sqlite3.connect(source_db.as_uri() + "?mode=ro", uri=True) as source, sqlite3.connect(destination / "control.db") as output:
            source.execute("PRAGMA query_only=ON")
            source.backup(output)
        return {path.relative_to(destination).as_posix(): sha256_hex(path.read_bytes())
                for path in sorted(destination.rglob("*")) if path.is_file()}



def _export_paths(directory: Path) -> dict[str, Path]:
    """Enumerate ordinary export files once, refusing redirects at every level."""
    files = {}
    pending = [directory]
    while pending:
        parent = pending.pop()
        with os.scandir(parent) as entries:
            for entry in entries:
                if parent == directory and entry.name == ".git":
                    continue
                info = entry.stat(follow_symlinks=False)
                if (stat.S_ISLNK(info.st_mode)
                        or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
                    raise ValueError("unsafe export path")
                path = Path(entry.path)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(info.st_mode):
                    files[path.relative_to(directory).as_posix()] = path
                else:
                    raise ValueError("unsafe export file")
    return files


def _read_export(directory: Path, *, _verified: Mapping[str, Any] | None = None
                 ) -> tuple[Mapping[str, Any], Mapping[str, bytes], str]:
    """Read and check one committed export of format 1–4.

    Args:
        directory: Export checkout.

    Returns:
        ``(HEAD document, exact file map, Git commit)``. A complete export (2–4) holds exactly
        its format's ``native/<entity>.json`` version sets, the seven evidence sets and a STATE whose
        counts and digests match; a typed native set holds only canonical rows.

    Raises:
        AuthorityError: Unknown format, unsafe, missing, changed or extra files, or contradictory coverage.
    """
    directory = Path(directory).resolve()
    try:
        paths = _export_paths(directory)
        head = json.loads(paths["HEAD.json"].read_text(encoding="utf-8"))
        number = _export_format(head["schema"])
        _schema, _receipt, state_schema, tables = EXPORT_FORMATS[number]
        files = {}
        for relative, digest in head["files"].items():
            value = paths[relative].read_bytes()
            if sha256_hex(value) != digest:
                raise ValueError(f"export file {relative} differs")
            files[relative] = value
        present = set(paths)
        if present != set(files) | {"HEAD.json"} or sha256_hex(canon(head["files"])) != head["file_map_digest"]:
            raise ValueError("export contains missing, changed or extra files")
        metadata = {name for name in files if name.startswith("metadata/")}
        if not metadata <= ANCILLARY_FILES:
            raise ValueError("export contains unsupported ancillary metadata")
        if "metadata/tool-tiers.json" in files:
            tiers = json.loads(files["metadata/tool-tiers.json"])
            if (not isinstance(tiers, dict) or any(not isinstance(name, str) or not name
                    or tier not in {"hb-executed", "host-verified"} for name, tier in tiers.items())):
                raise ValueError("export tool tier policy is invalid")
        workspace_paths = {name for name in files if name.startswith("workspace/")}
        from .workspace_state import EVIDENCE_TABLES, decode_evidence_rows

        if state_schema is None:
            if workspace_paths:
                raise ValueError("historical version export contains canonical workspace evidence")
        else:
            expected_paths = ({"workspace/STATE.json", *(f"workspace/{entity}.json" for entity in EVIDENCE_TABLES)}
                              if number != 4 else {"workspace/STATE.json"} | {
                                  name for name in workspace_paths
                                  if any(name.startswith(f"workspace/{entity}/") for entity in EVIDENCE_TABLES)})
            if workspace_paths != expected_paths:
                raise ValueError("complete export lacks exact canonical evidence sets")
            if {name for name in files if name.startswith("native/")} != {f"native/{entity}.json" for entity in tables}:
                raise ValueError("complete export lacks the exact native version sets of its format")
            state = json.loads(files["workspace/STATE.json"])
            if state.get("schema") != state_schema or set(state.get("entities", {})) != set(tables) | set(EVIDENCE_TABLES):
                raise ValueError("complete workspace schema differs")
            for entity, metadata in state["entities"].items():
                # A sealed projection has already proved these exact bytes and
                # their count/schema/semantic digest with this runtime. Changed
                # sets still take the complete structural verification below.
                if _verified is not None and head["schema"] == _verified["schema"]:
                    prefix = f"native/{entity}.json" if entity in tables else f"workspace/{entity}"
                    selected = {name: digest for name, digest in head["files"].items()
                                if name == prefix or name.startswith(prefix + "/") or name == prefix + ".json"}
                    previous = {name: digest for name, digest in _verified["files"].items()
                                if name == prefix or name.startswith(prefix + "/") or name == prefix + ".json"}
                    if (selected == previous and metadata == _verified["workspace_entities"][entity]):
                        continue
                rows = (json.loads(files[f"native/{entity}.json"]) if entity in tables
                        else decode_evidence_rows(entity, files, row_files=number == 4))
                if (not isinstance(rows, list) or metadata["count"] != len(rows)
                        or metadata["rows_digest"] != sha256_hex(canon(rows)) or not metadata["schema"]["present"]
                        or len({row["object_id"] for row in rows}) != len(rows)):
                    raise ValueError(f"complete workspace row identity/count/digest differs for {entity}")
                if entity in TYPED_TABLES and not all(M.is_canonical(entity, row) for row in rows):
                    raise ValueError(f"complete export holds a non-canonical {entity} row")
            operational = {entity: state["entities"][entity] for entity in EVIDENCE_TABLES}
            schemas = {entity: value["schema"] for entity, value in state["entities"].items()}
            if (head.get("workspace_digest") != sha256_hex(canon(operational))
                    or head.get("workspace_schemas_digest") != sha256_hex(canon(schemas))
                    or head.get("version_rows_digest") != sha256_hex(canon({entity: json.loads(files[f"native/{entity}.json"])
                        for entity in tables if entity != "prompt-event-entity"}))):
                raise ValueError("complete workspace digest or schema coverage differs")
        commit = _git(directory, "rev-parse", "HEAD")
        if _git(directory, "status", "--porcelain"):
            raise ValueError("export checkout differs from its Git commit")
        return head, files, commit
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise AuthorityError(f"authority export refused: {error}") from error


def _restore_workspace_schema(db: Path, state: Mapping[str, Any], *, tables_only: bool,
                              versions: Mapping[str, str]) -> None:
    """Reproduce verified canonical structures of an export's own sets in a new task-owned recovery DB.

    Args:
        db: New recovery database.
        state: The export's verified STATE document.
        tables_only: Create tables (before rows) or only indexes and triggers (after rows).
        versions: The export format's version sets; tables outside them are left to the opener.
    """
    from .workspace_state import EVIDENCE_TABLES

    tables = {**versions, **EVIDENCE_TABLES}
    with contextlib.closing(sqlite3.connect(db)) as connection:
        for entity, table in tables.items():
            schema = state["entities"][entity]["schema"]
            table_objects = [row for row in schema["objects"] if row[0] == "table"]
            if len(table_objects) != 1 or table_objects[0][1:3] != [table, table]:
                raise AuthorityError(f"recovery schema has no exact canonical table for {entity}")
            for kind, name, owner, sql in schema["objects"]:
                if kind not in {"table", "index", "trigger"} or owner != table:
                    raise AuthorityError(f"recovery schema has an unsupported object for {entity}")
                if tables_only != (kind == "table") or sql is None:
                    continue
                prefix = {"table": r"^CREATE\s+TABLE\b", "index": r"^CREATE\s+(?:UNIQUE\s+)?INDEX\b", "trigger": r"^CREATE\s+TRIGGER\b"}[kind]
                if not isinstance(sql, str) or re.match(prefix, sql.strip(), re.I) is None:
                    raise AuthorityError(f"recovery schema SQL has an unsupported declaration for {entity}")
                existing = connection.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE name=?", (name,)).fetchone()
                if existing is not None:
                    if list(existing) != [kind, name, owner, sql]:
                        raise AuthorityError(f"recovery schema conflicts with a native declaration for {entity}")
                else:
                    connection.execute(sql)
            if tables_only:
                from .workspace_state import schema_facts

                actual = schema_facts(connection, table)
                if actual["columns"] != schema["columns"]:
                    raise AuthorityError(f"recovery table columns differ for {entity}")
        connection.commit()


def restore_authority(export: Path, *, into: Path, context: Any = None, _projection: Any = None) -> RestoreResult:
    """Restore exact covered native identities into a new private, unbound DB.

    A format 1 or 2 export restores its six legacy version sets; the opener creates the typed
    tables empty, and verification refuses if the restored database would hold canonical typed rows.
    All three formats support recovery under a Git package binding without granting write
    authority. Any failure removes the newly created destination after releasing owned handles.

    Args:
        export: Verified committed authority export checkout.
        into: New directory; existing destinations refuse.
        context: Caller-owned HeavenBase context, if supplied.

    Returns:
        Exact covered revision and explicit anchor; writer authorization is false.

    Raises:
        AuthorityError: Existing destination, invalid export, or failed anchored verification.
    """
    directory = Path(into).resolve()
    if _projection is not None:
        from .git_authority import _Projection

        if not isinstance(_projection, _Projection) or not _projection.permits_control(directory / "control.db"):
            raise AuthorityError("authority restore requires its owning private staged projection")
    head, files, _commit = _projection.read_export(export) if _projection is not None else _read_export(export)
    if directory.exists():
        raise AuthorityError("restore destination must be a new private directory")
    directory.mkdir(parents=True, mode=0o700)
    try:
        return _restore_native(export, directory=directory, head=head, files=files, context=context,
                               projection=_projection)
    except BaseException as error:
        # Keep the exact path required by Git projections and HB contexts. Only this
        # invocation owns the directory; an existing destination never reaches here.
        try:
            shutil.rmtree(directory)
        except OSError as cleanup_error:
            error.add_note(f"restore destination cleanup failed: {cleanup_error}")
        raise


def _restore_native(export: Path, *, directory: Path, head: Mapping[str, Any], files: Mapping[str, bytes],
                    context: Any, projection: Any) -> RestoreResult:
    """Materialize and verify an export inside the caller's newly owned recovery directory."""
    from .control import Control

    number = _export_format(head["schema"])
    _schema, _receipt, state_schema, tables = EXPORT_FORMATS[number]
    db = directory / "control.db"
    if state_schema is not None:
        state = json.loads(files["workspace/STATE.json"])
        _restore_workspace_schema(db, state, tables_only=True, versions=tables)
    # Complete exports create exact table definitions before HB registers its entities.
    # This private construction does not open an existing authoritative DB for mutation.
    control = Control._open_native(db, context=context, owns_database=True)

    def insert(entity_cls: Any, table: str, rows: Sequence[Mapping[str, Any]]) -> None:
        # HB's overwrite=False probes storage once per row. An export without a repeated
        # identity loaded into a table that holds no row makes every row new: prove both
        # once instead, so no existing identity can ever be replaced. Each row still passes
        # HB's materialization; a batch commits once, and fresh Catalog rows have nothing
        # to merge in a new DB.
        try:
            repeated = len({row["object_id"] for row in rows}) != len(rows)
        except (KeyError, TypeError) as error:
            raise AuthorityError(f"restore export rows of {table} lack exact identities") from error
        if repeated:
            raise AuthorityError(f"restore export repeats an identity of {table}")
        with contextlib.closing(sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)) as connection:
            if connection.execute(f'SELECT 1 FROM "{table}" LIMIT 1').fetchone() is not None:
                raise AuthorityError(f"restore destination {table} already holds rows")
        for start in range(0, len(rows), _RESTORE_BATCH_ROWS):
            control.ws.upsert_many(entity_cls, rows[start:start + _RESTORE_BATCH_ROWS], overwrite=True,
                                   catalog_mode="fresh")

    try:
        for entity, table in tables.items():
            insert(_entities(control)[entity], table, json.loads(files[f"native/{entity}.json"]))
        if state_schema is not None:
            from .workspace_state import EVIDENCE_TABLES, decode_evidence_rows, evidence_entities

            for entity in evidence_entities():
                insert(control.ws.entities[entity], EVIDENCE_TABLES[entity],
                       decode_evidence_rows(entity, files, row_files=number == 4))
    finally:
        control.close()
    if state_schema is not None:
        # User/native index and trigger definitions are restored after exact rows,
        # so their insert side effects cannot rewrite the covered recovery state.
        _restore_workspace_schema(db, state, tables_only=False, versions=tables)
    record = head["binding"]
    if record.get("schema") == B.SCHEMA_GIT_BINDING:
        binding = GitAuthorityBinding(**record)
        if projection is not None and projection.owner.binding != binding:
            raise AuthorityError("restored Git identity differs from its owning projection")
    else:
        binding = AuthorityBinding(record["schema"], record["authority_uuid"], record["instance_id"], Path(record["db_path"]))
    with Authority._read_native(db, binding=binding, export_anchor=None if projection is not None else export) as reader:
        if projection is not None:
            projection.decorate(reader)
        result = reader.verify(export_anchor=export)
        if not result.ok:
            raise AuthorityError("restored authority verification failed: " + "; ".join(result.violations))
        return RestoreResult(db, reader.head(), reader.head().revision, Path(export).resolve())


def mark_binding(*, instance_id: str, db_path: Path, authority_uuid: str,
                 binding_path: Path | None = None) -> AuthorityBinding:
    """Prepare the ordinary package binding without opening or marking a DB."""
    binding = AuthorityBinding(B.SCHEMA_BINDING, authority_uuid, instance_id, Path(db_path).resolve())
    B.write_binding(binding, binding_path)
    return binding
