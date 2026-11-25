"""The control workspace: where prompt rows are published, described, exported and imported.

The control workspace is a detached, main-only sqlite HeavenBase workspace at
the ordinary package-bound ``control/control.db``. It holds canonical prompt rows, canonical
Tool, rule and parameter-set versions, immutable release/proposal/event history, evaluation
evidence and native conversations. GeneralRule and KnobSet rows with an empty ``format`` are
historical read projections of the immutable ``rules.general``/``config.knobs`` prompt carriers.
Explicit paths remain available for scratch initialization; production adapters validate the binding.
Runtime clients read verified per-run snapshots with the standard library.

Two functions here are the only ``Capsule.from_code`` call sites of the repository:
:func:`capture_text` goes through HeavenBase's own template path (``Prompt(template, ...)``
calls ``from_code`` with ``fn_name="render"``) and :func:`capture_compose` captures the
``compose(parts, params)`` source :func:`spec.compile_spec` generated — from a spec row's
``meta.spec`` or from a doc row's ``meta.text`` (:func:`doc.compile_document`). Both assert
the single source layer, the ``<capsule-code>`` filename, empty ``global_refs`` and
``trust.verify_source_hash``.

Publication is create-only: describe first; an existing row with the same content key is
idempotent (an event is recorded), a different content under the same ``name:version`` is a
hard error, an absent row is registered with ``overwrite=False`` and an "already exists"
race is settled by describing again. ``register(overwrite=True)`` is never called.
"""
from __future__ import annotations

import os
import shutil
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .canon import SCHEMA_PROMPT, canon, content_key, escape, sha256_hex, template_code
from .control_schema import CONTROL_ENTITIES, EVENT_KINDS, PromptEventEntity, ProposalEntity, ReleaseEntity
from .doc import DocError, compile_document
from .entries import RULES_ENTRY
from .policy import PolicyError, check_document, check_meta_keys, check_name, check_reason, check_text, check_version
from .spec import compile_spec, parse_spec
from .store import SCHEMA_OBJECT, PromptStoreError, canonical_bytes, write_if_absent

__all__ = [
    "CONTROL_DB_NAME",
    "CONTROL_WS_ID",
    "SQLITE_PRAGMAS",
    "Control",
    "DocDraft",
    "HbRenderer",
    "PromptControlError",
    "PublishOutcome",
    "SharedHomeGuard",
    "SpecDraft",
    "TextDraft",
    "baseline_text_files",
    "capture_compose",
    "capture_text",
    "check_row_source",
    "check_row_integrity",
    "control_db_path",
    "ensure_private_dir",
    "export_objects",
    "main_only_backends",
    "object_file",
    "read_objects",
]

CONTROL_DB_NAME = "control.db"
CONTROL_WS_ID = "rubiksql_evolution"
#: The sqlite pragmas server.py applies to every RubikSQL workspace; HeavenBase replaces its defaults with this list.
SQLITE_PRAGMAS = (
    "PRAGMA foreign_keys = ON;",
    "PRAGMA journal_mode = WAL;",
    "PRAGMA synchronous = NORMAL;",
    "PRAGMA busy_timeout = 120000;",
)
SQLITE_POOL = {"connect_args": {"timeout": 60}}
_LAYER_SHAPE = {"type": "source", "priority": 10, "filename": "<capsule-code>", "global_refs": {}}


class PromptControlError(RuntimeError):
    """A control-workspace operation was refused; the message names the row and the rule."""


def control_db_path() -> Path:
    """Return the canonical database named by the ordinary package binding.

    Returns:
        Path: The existing bound database path; no environment or HOME override.

    Raises:
        BindingError: The ordinary binding is absent or invalid.
    """
    from .binding import load_binding
    from .types import GitAuthorityBinding

    binding = load_binding()
    if isinstance(binding, GitAuthorityBinding):
        raise PromptControlError("Git authority has no canonical local database; open an Authority projection")
    return binding.db_path


def ensure_private_dir(directory: Path) -> Path:
    """Create a directory readable by its owner only (mode 0700).

    Args:
        directory (Path): Directory to create or tighten.

    Returns:
        Path: The directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, stat.S_IRWXU)
    except OSError:
        pass
    return directory


def main_only_backends(db_path: Path) -> dict[str, dict[str, Any]]:
    """The ``backends`` mapping of a main-only sqlite workspace with server.py's pragmas.

    Args:
        db_path (Path): The sqlite file.

    Returns:
        dict[str, dict[str, Any]]: ``{"main": {...}}`` and no vector backend.
    """
    from ..workspace import location

    return {"main": {"type": "sqlite", "database": location(db_path), "pragmas": list(SQLITE_PRAGMAS), "pool": dict(SQLITE_POOL)}}


# ---------------------------------------------------------------------------------------
# Drafts


def _meta(kind: str, body: Mapping[str, Any], origin: str, author: str, parent: str | None, proposal: str | None) -> dict[str, Any]:
    meta: dict[str, Any] = {"schema": SCHEMA_PROMPT, "kind": kind, **body, "origin": origin}
    if author:
        meta["author"] = author
    if parent:
        meta["parent"] = parent
    if proposal:
        meta["proposal"] = proposal
    return meta


@dataclass(frozen=True)
class TextDraft:
    """A text row to publish.

    Attributes:
        name (str): Row name.
        text (str): Human-readable text with single-brace slots.
        slots (list[str]): Declared slots.
        description (str): Row description.
        tags (list[str]): Row tags.
        reason (str): ``manifest.reason``.
        origin (str): ``agent`` or ``human``.
        author (str): Author label, kept in ``meta.author`` when non-empty.
        parent (str | None): Parent version's content key, when known.
        proposal (str | None): Proposal id that introduces the row, when known.
    """

    name: str
    text: str
    slots: list[str]
    description: str
    tags: list[str]
    reason: str
    origin: str = "human"
    author: str = ""
    parent: str | None = None
    proposal: str | None = None
    kind: str = field(default="text", init=False)

    @property
    def meta(self) -> dict[str, Any]:
        """``manifest.meta`` without ``content_key``.

        Args:
            None.

        Returns:
            dict[str, Any]: The closed-table meta.
        """
        return _meta("text", {"text": self.text, "slots": list(self.slots)}, self.origin, self.author, self.parent, self.proposal)

    @property
    def content_key(self) -> str:
        """The row's content key.

        Args:
            None.

        Returns:
            str: 64 hex digits.
        """
        return content_key(self.name, self.meta)

    @property
    def version(self) -> str:
        """The HB version column.

        Args:
            None.

        Returns:
            str: ``content_key[:16]``.
        """
        return self.content_key[:16]


@dataclass(frozen=True)
class SpecDraft:
    """A spec row to publish.

    Attributes:
        name (str): Row name.
        spec (Any): ``rubiksql.spec/1`` JSON.
        parts (dict[str, str]): Slot to row name.
        params (dict[str, Any]): Parameter types.
        description (str): Row description.
        tags (list[str]): Row tags.
        reason (str): ``manifest.reason``.
        origin (str): ``agent`` or ``human``.
        author (str): Author label.
        parent (str | None): Parent version's content key.
        proposal (str | None): Proposal id.
    """

    name: str
    spec: Any
    parts: dict[str, str]
    params: dict[str, Any]
    description: str
    tags: list[str]
    reason: str
    origin: str = "human"
    author: str = ""
    parent: str | None = None
    proposal: str | None = None
    #: Slot to the 16-hex version of the part row the spec is type-checked against; the newest active row
    #: of that name when a slot is absent (C3c: a proposal names the exact versions of its closure).
    part_versions: Mapping[str, str] | None = None
    kind: str = field(default="spec", init=False)

    @property
    def meta(self) -> dict[str, Any]:
        """``manifest.meta`` without ``content_key``.

        Args:
            None.

        Returns:
            dict[str, Any]: The closed-table meta.
        """
        return _meta("spec", {"spec": self.spec, "parts": dict(self.parts), "params": dict(self.params)}, self.origin, self.author, self.parent, self.proposal)

    @property
    def content_key(self) -> str:
        """The row's content key.

        Args:
            None.

        Returns:
            str: 64 hex digits.
        """
        return content_key(self.name, self.meta)

    @property
    def version(self) -> str:
        """The HB version column.

        Args:
            None.

        Returns:
            str: ``content_key[:16]``.
        """
        return self.content_key[:16]


@dataclass(frozen=True)
class DocDraft:
    """A doc row to publish: one document, compiled into a function prompt.

    Attributes:
        name (str): Row name.
        text (str): The document (``rubiksql.doc/1``).
        params (dict[str, Any]): Parameter types.
        description (str): Row description.
        tags (list[str]): Row tags.
        reason (str): ``manifest.reason``.
        origin (str): ``agent`` or ``human``.
        author (str): Author label.
        parent (str | None): Parent version's content key.
        proposal (str | None): Proposal id.
    """

    name: str
    text: str
    params: dict[str, Any]
    description: str
    tags: list[str]
    reason: str
    origin: str = "human"
    author: str = ""
    parent: str | None = None
    proposal: str | None = None
    kind: str = field(default="doc", init=False)

    @property
    def meta(self) -> dict[str, Any]:
        """``manifest.meta`` without ``content_key``: the document and its parameters, nothing derived.

        Args:
            None.

        Returns:
            dict[str, Any]: The closed-table meta.
        """
        return _meta("doc", {"text": self.text, "params": dict(self.params)}, self.origin, self.author, self.parent, self.proposal)

    @property
    def content_key(self) -> str:
        """The row's content key.

        Args:
            None.

        Returns:
            str: 64 hex digits.
        """
        return content_key(self.name, self.meta)

    @property
    def version(self) -> str:
        """The HB version column.

        Args:
            None.

        Returns:
            str: ``content_key[:16]``.
        """
        return self.content_key[:16]


@dataclass(frozen=True)
class PublishOutcome:
    """What :meth:`Control.publish` did for one draft.

    Attributes:
        status (str): ``"created"`` or ``"identical"`` (the row existed with the same content key).
        name (str): Row name.
        version (str): Row version.
        content_key (str): Full content key.
        row (Mapping[str, Any]): The row as the workspace holds it after the call.
    """

    status: str
    name: str
    version: str
    content_key: str
    row: Mapping[str, Any]


# ---------------------------------------------------------------------------------------
# Capsule capture: the repository's only Capsule.from_code call sites


def _check_layers(manifest: Any, expected_source: str, qualname: str, name: str) -> None:
    layers = list(manifest.restore_layers)
    if len(layers) != 1:
        raise PromptControlError(f"{name}: HeavenBase captured {len(layers)} restore layers; exactly one source layer is required")
    layer = layers[0]
    for key, value in {**_LAYER_SHAPE, "qualname": qualname}.items():
        if layer.get(key) != value:
            raise PromptControlError(f"{name}: restore layer {key} is {layer.get(key)!r}, expected {value!r}")
    if layer.get("source") != expected_source:
        raise PromptControlError(f"{name}: the captured source differs from the source regenerated from meta")
    if layer.get("source_hash") != sha256_hex(expected_source):
        raise PromptControlError(f"{name}: the captured source hash differs from sha256(source)")
    manifest.trust["verify_source_hash"] = True


def capture_text(control: "Control", draft: TextDraft) -> Any:
    """Build the ``Prompt`` object of a text row through HeavenBase's own template path.

    ``Prompt(escape(text, slots), ...)`` makes HeavenBase generate ``render(*args, tr=str,
    **kwargs)`` from ``_template_code`` and capture it with ``Capsule.from_code`` — the same
    bytes :func:`canon.template_code` regenerates at load time.

    Args:
        control (Control): The control workspace (supplies the Prompt class and the workspace).
        draft (TextDraft): The row to capture.

    Returns:
        Any: An unregistered ``Prompt`` bound to the control workspace.

    Raises:
        PromptControlError: If the text fails the field policy, the reason fails its policy, or the
            captured capsule does not have the single-source-layer shape.
    """
    try:
        check_name(draft.name)
        template = check_text(draft.text, list(draft.slots), name=draft.name)
        check_reason(draft.reason)
    except PolicyError as error:
        raise PromptControlError(f"{draft.name}: {error}") from error
    meta = draft.meta
    meta["content_key"] = draft.content_key
    prompt = control.Prompt(template, name=draft.name, version=draft.version, ws=control.ws, description=draft.description, tags=list(draft.tags), **meta)
    _check_layers(prompt.manifest, template_code(draft.name, template, []), "render", draft.name)
    return prompt


def capture_compose(control: "Control", draft: SpecDraft | DocDraft) -> Any:
    """Build the ``Prompt`` object of a spec or doc row from the compiled ``compose`` source.

    Args:
        control (Control): The control workspace.
        draft (SpecDraft | DocDraft): The row to capture. A spec draft's parts must already exist as rows so their
            signatures can type-check the spec; a doc draft's document is parsed against its parameters.

    Returns:
        Any: An unregistered ``Prompt`` whose capsule holds ``compile_spec(...)`` or ``compile_document(...)``.

    Raises:
        PromptControlError: If the spec fails parsing against its parts or the document fails its policy or grammar,
            the reason fails its policy, or the captured capsule does not have the single-source-layer shape.
    """
    from heavenbase.capsule import Capsule

    try:
        check_name(draft.name)
        check_reason(draft.reason)
        if isinstance(draft, DocDraft):
            check_document(draft.text)
            source = compile_document(draft.name, draft.text, draft.params)
        else:
            sigs = {slot: control.part_signature(target, (draft.part_versions or {}).get(slot)) for slot, target in draft.parts.items()}
            parse_spec(draft.spec, params=draft.params, part_sigs=sigs)
            source = compile_spec(draft.name, draft.spec, draft.parts, draft.params)
    except (PolicyError, ValueError) as error:
        raise PromptControlError(f"{draft.name}: {error}") from error
    meta = draft.meta
    meta["content_key"] = draft.content_key
    prompt = control.Prompt(name=draft.name, version=draft.version, ws=control.ws, description=draft.description, tags=list(draft.tags), **meta)
    capsule = Capsule.from_code(
        source,
        fn_name="compose",
        name=draft.name,
        namespace="prompt",
        version=draft.version,
        description=draft.description or "Compose a prompt from its parts.",
        tags=list(draft.tags),
        layers=("source",),
        resolver=control.ws.module_resolver,
        config=control.ws.context.config,
    )
    prompt._adopt_capsule(capsule, capsule.to_func())
    _check_layers(prompt.manifest, source, "compose", draft.name)
    return prompt


def check_row_source(row: Mapping[str, Any]) -> str:
    """Regenerate a stored row's capsule source from its meta and require byte equality.

    This is the check every path that hands a row to HeavenBase for execution must pass
    first (trust boundary §3): the only source HeavenBase ever executes is the template
    source it generates itself or the compiled spec source, both regenerated here. The name
    and version are validated before anything else, because the export path builds
    ``objects/<name>/<version>.json`` from them and a control workspace may hold rows other
    agents wrote through the HB API without any check.

    Args:
        row (Mapping[str, Any]): A ``sys-prompt`` row.

    Returns:
        str: The row kind, ``"text"``, ``"spec"`` or ``"doc"``.

    Raises:
        PromptControlError: If the row's name or version is not a valid path component, or its meta,
            source, layer shape, hashes or trust flag disagree.
    """
    try:
        name = check_name(row.get("name"))
        version = check_version(row.get("version"))
    except PolicyError as error:
        raise PromptControlError(f"row {row.get('name')!r}: {error}") from error
    meta = dict((row.get("manifest") or {}).get("meta") or {})
    try:
        check_meta_keys(meta)
    except PolicyError as error:
        raise PromptControlError(f"{name}: {error}") from error
    kind = meta.get("kind")
    if kind == "text":
        try:
            template = check_text(meta.get("text"), meta.get("slots"), name=name)
        except PolicyError as error:
            raise PromptControlError(f"{name}: {error}") from error
        expected, qualname = template_code(name, template, []), "render"
    elif kind == "spec":
        try:
            expected = compile_spec(name, meta["spec"], meta["parts"], meta["params"])
        except (KeyError, ValueError) as error:
            raise PromptControlError(f"{name}: {error}") from error
        qualname = "compose"
    elif kind == "doc":
        stray = [key for key in ("spec", "parts", "slots") if key in meta]
        if stray:
            raise PromptControlError(f"{name}: a doc row stores no {stray}; they derive from meta.text")
        try:
            check_document(meta.get("text"))
            expected = compile_document(name, meta["text"], meta["params"])
        except (KeyError, PolicyError, DocError) as error:
            raise PromptControlError(f"{name}: {error}") from error
        qualname = "compose"
    else:
        raise PromptControlError(f"{name}: unknown row kind {kind!r}")
    if content_key(name, meta) != meta.get("content_key") or version != str(meta.get("content_key"))[:16]:
        raise PromptControlError(f"{name}: content key or version does not match the row's meta")
    capsule = row.get("capsule") or {}
    layers = capsule.get("restore_layers") or []
    if len(layers) != 1:
        raise PromptControlError(f"{name}: expected exactly one restore layer, found {len(layers)}")
    layer = layers[0]
    for key, value in {**_LAYER_SHAPE, "qualname": qualname}.items():
        if layer.get(key) != value:
            raise PromptControlError(f"{name}: restore layer {key} is {layer.get(key)!r}, expected {value!r}")
    if layer.get("source") != expected:
        raise PromptControlError(f"{name}: stored capsule source differs from the source regenerated from meta")
    digest = sha256_hex(expected)
    if layer.get("source_hash") != digest or (capsule.get("source") or {}).get("source_hash") != digest or (capsule.get("fingerprints") or {}).get("source_hash") != digest:
        raise PromptControlError(f"{name}: the three source hashes must equal sha256(source)")
    if (capsule.get("trust") or {}).get("verify_source_hash") is not True:
        raise PromptControlError(f"{name}: capsule.trust.verify_source_hash must be true")
    return kind


def check_row_integrity(row: Mapping[str, Any]) -> str:
    """Verify regenerated source, complete mirrored prompt fields and hashes.

    Args:
        row: Complete native sys-prompt row.

    Returns:
        Verified prompt kind.

    Raises:
        PromptControlError: Source, native mirrors or fingerprints differ.
    """
    from .integrity import PromptIntegrityError, check_prompt_row

    try:
        return check_prompt_row(row)
    except PromptIntegrityError as error:
        raise PromptControlError(str(error)) from error


# ---------------------------------------------------------------------------------------
# The control workspace


def _close_control_resources(ws: Any, *, backend: Any = None, projection: Any = None,
                             dispose_backends: bool = True) -> None:
    """Release detached resources, attempting every close and retaining the first failure."""
    backends = list(ws.backends.items()) if ws is not None else []
    if backend is not None and not any(value is backend for _name, value in backends):
        backends.append(("main", backend))
    failures: list[BaseException] = []
    if ws is not None:
        try:
            ws.release()
        except BaseException as error:
            failures.append(error)
    if dispose_backends:
        for _name, value in backends:
            try:
                dispose = getattr(getattr(value, "db", None), "dispose", None)
                if callable(dispose):
                    dispose()
            except BaseException as error:
                failures.append(error)
    primary = failures[0] if failures else None
    if primary is not None:
        for error in failures[1:]:
            primary.add_note(f"control cleanup also failed: {error}")
    try:
        if primary is not None:
            raise primary
    finally:
        # The projection must see failed disposal before considering cache sealing.
        if projection is not None:
            try:
                projection.close()
            except BaseException as error:
                if primary is None:
                    raise
                primary.add_note(f"control cleanup also failed: {error}")


class Control:
    """An open control workspace.

    Attributes:
        ws (Any): The detached HeavenBase facade.
        path (Path): The sqlite file.
        Prompt (type): The ``sys-prompt`` entity class of this workspace's artifact generation.
        Translation (type): The ``sys-translation`` entity class.
    """

    def __init__(self, ws: Any, path: Path, Prompt: type, Translation: type) -> None:
        self.ws = ws
        self.path = path
        self.Prompt = Prompt
        self.Translation = Translation
        self._projection_owner: Any = None
        self._closed = False

    def close(self) -> None:
        """Release this facade and its DB resources without changing stored data.

        A Git facade also closes its projection lease. A caller-supplied HB
        context remains owned by the caller and is never closed here.
        """
        if self._closed:
            return
        self._closed = True
        _close_control_resources(self.ws, projection=self._projection_owner)

    @classmethod
    def open(cls, path: str | os.PathLike[str] | None = None, *, context: Any = None, create: bool = True,
             read_only: bool = False, _projection: Any = None) -> "Control":
        """Open (or create) the control workspace.

        Args:
            path (str | os.PathLike[str] | None): Database file or directory; :func:`control_db_path` by default.
            context (Any): HeavenBase machine context; the process default when omitted.
            create (bool): Create the directory and database when absent.
            read_only (bool): Reflect existing storage without DDL or persisted schema
                publication. Every row mutation is refused.

        Returns:
            Control: The canonical workspace with content (including the typed Tool, rule and
                parameter-set entities), operational evidence and historical projection schemas registered.

        Raises:
            PromptControlError: If ``create`` is false and the database does not exist.
        """
        db = control_db_path() if path is None else Path(path).expanduser()
        if db.suffix != ".db":
            db = db / CONTROL_DB_NAME
        if not read_only and db.is_file():
            from .binding import load_binding
            from .types import GitAuthorityBinding

            configured = load_binding()
            if isinstance(configured, GitAuthorityBinding):
                from .git_authority import _Projection

                if not isinstance(_projection, _Projection) or not _projection.permits_control(db):
                    raise PromptControlError("Git authority permits writes only inside its private staged projection")
        return cls._open_native(db, context=context, create=create, read_only=read_only)

    @classmethod
    def _open_native(cls, db: Path, *, context: Any = None, create: bool = True,
                     read_only: bool = False, owns_database: bool = False) -> "Control":
        """Construct a native facade after its caller establishes storage ownership.

        Ordinary opens pass through ``open`` and its Git write fence. Recovery alone
        calls this directly for a newly owned private database, whose exported schema
        may already exist; that construction grants no authority writer capability.
        ``owns_database`` permits failed-open disposal of that private database;
        ordinary writable opens can share an engine with another caller.
        """
        import heavenbase as hb
        from heavenbase.context import context_or_default
        from heavenbase.entity.registry import load_entity_type
        from heavenbase.workspace import WorkspaceManifest

        if not create and not db.exists():
            raise PromptControlError(f"control workspace {db} does not exist; run `python -m rubiksql_hb.prompts init` first")
        if read_only and not db.is_file():
            raise PromptControlError(f"control database does not exist: {db}")
        context = context_or_default(context)
        backend = None
        ws = None
        try:
            if read_only:
                from .readonly import ReadOnlySQLiteBackend

                backend = ReadOnlySQLiteBackend("main", db.resolve().as_uri(), ws_id=CONTROL_WS_ID,
                                                resolver=context.modules(), config=context.config)
            else:
                from heavenbase.backends import BackendType

                ensure_private_dir(db.parent)
                # Match HB's module and workspace configuration scopes while retaining
                # the backend if manifest initialization fails before assigning ws.
                with context.config.scoped(context.config.base_scope):
                    resolver = context.modules()
                    resolver.setup()
                with context.config.scoped(f"{context.config.base_scope}.{CONTROL_WS_ID}"):
                    backend = BackendType.build("main", main_only_backends(db)["main"], ws_id=CONTROL_WS_ID,
                                                resolver=resolver, config=context.config)
            # Manifest bootstrap excludes pending tracked classes from another facade.
            # Dashboard replay strips computes; this opener binds the canonical declarations.
            ws = hb.HeavenBase.from_manifest(WorkspaceManifest(id=CONTROL_WS_ID), context=context,
                                            backends={"main": backend}, detached=True)
            ws.enable_extension("prompt")
            ws.enable_extension("toolkit")
            ws.enable_extension("agent")
            for entity in CONTROL_ENTITIES:
                ws.register(entity)
            # The canonical opener owns every schema. Import locally because lab_schema
            # reuses control_schema and the package facade imports this module.
            from ..lab_schema import LAB_ANALYTICS_ENTITIES
            from .control_schema import GeneralRuleEntity, KnobSetEntity

            for entity in (*LAB_ANALYTICS_ENTITIES, GeneralRuleEntity, KnobSetEntity):
                ws.register(entity)
            if read_only:
                backend.finish_binding()
            setattr(ws, "_rubiksql_authority_db", str(db.resolve()))
            setattr(ws, "_rubiksql_lab_dir", str(db.resolve().parent))
            prompt_type = load_entity_type("sys-prompt", resolver=ws.module_resolver)
            translation_type = load_entity_type("sys-translation", resolver=ws.module_resolver)
            return cls(ws, db, prompt_type, translation_type)
        except BaseException as error:
            try:
                _close_control_resources(ws, backend=backend, dispose_backends=read_only or owns_database)
            except BaseException as cleanup_error:
                error.add_note(f"control cleanup also failed: {cleanup_error}")
                for note in getattr(cleanup_error, "__notes__", ()):
                    error.add_note(note)
            raise

    # -- reading ---------------------------------------------------------------------

    def describe(self, name: str, version: str) -> Mapping[str, Any] | None:
        """Return one prompt row, tombstoned or not.

        Args:
            name (str): Row name.
            version (str): Row version.

        Returns:
            Mapping[str, Any] | None: The row, or ``None`` when absent.
        """
        try:
            return self.Prompt.describe(f"{name}:{version}", ws=self.ws)
        except KeyError:
            return None

    def rows(self, *, status: str | None = None, prefix: str | None = None) -> list[Mapping[str, Any]]:
        """List prompt rows.

        Args:
            status (str | None): ``active``, ``deleted``, or ``None`` for every row.
            prefix (str | None): Dotted-name prefix filter.

        Returns:
            list[Mapping[str, Any]]: Rows sorted by name and version.
        """
        rows = self.Prompt.list(prefix=prefix, status=status, ws=self.ws)
        return sorted(rows, key=lambda row: (str(row.get("name")), str(row.get("version"))))

    def translations(self) -> list[Mapping[str, Any]]:
        """Every ``sys-translation`` row; the design requires this to stay empty.

        Args:
            None.

        Returns:
            list[Mapping[str, Any]]: Translation rows.
        """
        return list(self.ws.query(self.Translation).execute().rows())

    def part_signature(self, name: str, version: str | None = None) -> Any:
        """The part signature of one row named ``name``, for type-checking a spec draft.

        Args:
            name (str): Part row name.
            version (str | None): The 16-hex version to read; the newest active row of the name when ``None``.

        Returns:
            PartSig: Kind and parameter types.

        Raises:
            PromptControlError: If no active row carries the name (or the named version is absent).
        """
        from .spec import PartSig, check_type

        if version is not None:
            row = self.describe(name, version)
            if row is None:
                raise PromptControlError(f"part {name}:{version} is not in the control workspace; publish parts before the spec that uses them")
            rows = [row]
        else:
            rows = [row for row in self.rows(status="active") if row.get("name") == name]
        if not rows:
            raise PromptControlError(f"part {name!r} has no active row in the control workspace; publish parts before the spec that uses them")
        meta = rows[-1]["manifest"]["meta"]
        if meta["kind"] == "text":
            return PartSig("text", {slot: None for slot in meta["slots"]})
        return PartSig("spec", {param: check_type(value) for param, value in meta["params"].items()})

    # -- writing ---------------------------------------------------------------------

    def _judge_existing(self, existing: Mapping[str, Any], draft: TextDraft | SpecDraft | DocDraft, actor: str) -> PublishOutcome:
        stored_key = (existing.get("manifest") or {}).get("meta", {}).get("content_key")
        recomputed = content_key(draft.name, dict((existing.get("manifest") or {}).get("meta") or {}))
        candidate = (capture_text(self, draft) if isinstance(draft, TextDraft) else capture_compose(self, draft)).to_dict()
        candidate["manifest"]["reason"] = draft.reason
        from heavenbase.utils import sha256hash
        from .authority import semantic_row

        candidate["fingerprints"]["manifest"] = sha256hash(candidate["manifest"])
        check_row_source(existing)
        if stored_key == draft.content_key == recomputed and semantic_row(existing) == semantic_row(candidate):
            self.record_event("publish", f"{draft.name}:{draft.version}", {"status": "identical", "content_key": draft.content_key}, actor=actor)
            return PublishOutcome("identical", draft.name, draft.version, draft.content_key, existing)
        raise PromptControlError(
            f"{draft.name}:{draft.version} already exists with different content (stored content key {stored_key}, draft {draft.content_key}); "
            "the 16-hex version collided or the row was tampered with — publication never replaces an existing row"
        )

    def publish(self, draft: TextDraft | SpecDraft | DocDraft, *, actor: str = "human") -> PublishOutcome:
        """Publish one row create-only.

        Args:
            draft (TextDraft | SpecDraft | DocDraft): The row.
            actor (str): Who publishes, recorded on the event.

        Returns:
            PublishOutcome: ``created`` or ``identical``.

        Raises:
            PromptControlError: If a row with the same ``name:version`` holds different content, or the
                draft fails a policy.
        """
        prompt = capture_text(self, draft) if isinstance(draft, TextDraft) else capture_compose(self, draft)
        existing = self.describe(draft.name, draft.version)
        if existing is not None:
            return self._judge_existing(existing, draft, actor)
        try:
            prompt.register(self.ws, overwrite=False, reason=draft.reason)
        except ValueError as error:
            if "already exists" not in str(error):
                raise
            existing = self.describe(draft.name, draft.version)
            if existing is None:
                raise PromptControlError(f"{draft.name}:{draft.version}: register reported an existing row that describe cannot find") from error
            return self._judge_existing(existing, draft, actor)
        row = self.describe(draft.name, draft.version)
        if row is None:
            raise PromptControlError(f"{draft.name}:{draft.version}: the row is missing right after registration")
        check_row_source(row)
        self.record_event("publish", f"{draft.name}:{draft.version}", {"status": "created", "content_key": draft.content_key, "origin": draft.origin}, actor=actor)
        return PublishOutcome("created", draft.name, draft.version, draft.content_key, row)

    def import_objects(self, objects: Iterable[Mapping[str, Any]], *, actor: str = "human") -> tuple[int, int]:
        """Import exported object files, preserving their timestamps.

        Args:
            objects (Iterable[Mapping[str, Any]]): ``rubiksql.object/1`` documents.
            actor (str): Who imports.

        Returns:
            tuple[int, int]: Rows created and rows already present with the same content.

        Raises:
            PromptControlError: If an object is malformed, fails :func:`check_row_source`, or collides with
                a row of different content.
        """
        from heavenbase.extensions.prompt.name import prompt_row_id

        created = skipped = 0
        for obj in objects:
            if not isinstance(obj, Mapping) or obj.get("schema") != SCHEMA_OBJECT or not isinstance(obj.get("row"), Mapping):
                raise PromptControlError(f"object is not a {SCHEMA_OBJECT} document")
            row = dict(obj["row"])
            check_row_source(row)
            key = row["manifest"]["meta"]["content_key"]
            # Key the row the way HeavenBase keys prompt rows so describe/load by name:version resolve
            # (rows exported by HeavenBase already carry this id; synthetic objects may not).
            row["object_id"] = prompt_row_id(row["name"], row["version"])
            if isinstance(row.get("manifest"), Mapping):
                row["manifest"] = {**row["manifest"], "object_id": row["object_id"]}
            try:
                self.ws.upsert(self.Prompt, row, overwrite=False)
            except ValueError as error:
                if "already exists" not in str(error):
                    raise
                existing = self.describe(row["name"], row["version"])
                from .authority import semantic_row

                if existing is not None:
                    check_row_source(existing)
                if existing is None or semantic_row(existing) != semantic_row(row):
                    raise PromptControlError(f"{row['name']}:{row['version']}: an existing row holds different content") from error
                skipped += 1
                continue
            created += 1
        return created, skipped

    def put_release(self, rid: str, refs: Mapping[str, str], entries: Iterable[str], rules: str, *, actor: str = "human", reason: str = "") -> str:
        """Record a release (create-only).

        Args:
            rid (str): Release id.
            refs (Mapping[str, str]): Row name to content key.
            entries (Iterable[str]): Entry names.
            rules (str): Closure rules version.
            actor (str): Who records it.
            reason (str): Why the release exists.

        Returns:
            str: ``rid``.

        Raises:
            PromptControlError: If a release with this id exists with different content.
        """
        row = {"object_id": rid, "refs": dict(refs), "entries": sorted(entries), "rules": rules, "reason": reason, "created_at": time.time()}
        try:
            self.ws.upsert(ReleaseEntity, row, overwrite=False)
        except ValueError as error:
            if "already exists" not in str(error):
                raise
            existing = self.ws.get(rid, entity=ReleaseEntity) or {}
            if existing.get("refs") != row["refs"] or existing.get("entries") != row["entries"] or existing.get("rules") != rules:
                raise PromptControlError(f"release {rid} exists with different content") from error
            return rid
        self.record_event("publish", rid, {"entries": len(row["entries"]), "refs": len(row["refs"])}, actor=actor)
        return rid

    def put_proposal(self, pid: str, body: Mapping[str, Any], *, actor: str = "human") -> str:
        """Record a proposal (create-only).

        Args:
            pid (str): Proposal id.
            body (Mapping[str, Any]): Proposal fields (``parents``, ``release``, ``changes``, ``reason``, ...).
            actor (str): Who records it.

        Returns:
            str: ``pid``.

        Raises:
            PromptControlError: If the id exists with a different body.
        """
        row = {"object_id": pid, **{key: value for key, value in body.items() if key != "id"}}
        row.setdefault("created_at", time.time())
        try:
            self.ws.upsert(ProposalEntity, row, overwrite=False)
        except ValueError as error:
            if "already exists" not in str(error):
                raise
            existing = self.ws.get(pid, entity=ProposalEntity) or {}
            from .authority import semantic_row

            if semantic_row(existing) != semantic_row(row):
                raise PromptControlError(f"proposal {pid} exists with a different complete body") from error
            return pid
        self.record_event("draft", pid, {"release": row.get("release")}, actor=actor)
        return pid

    def record_event(self, kind: str, subject: str, payload: Mapping[str, Any], *, actor: str = "human") -> str:
        """Append one event.

        Args:
            kind (str): One of :data:`control_schema.EVENT_KINDS`.
            subject (str): ``name:version``, a release id or a proposal id.
            payload (Mapping[str, Any]): Event details.
            actor (str): Who caused the event.

        Returns:
            str: The event's object id.

        Raises:
            PromptControlError: If ``kind`` is unknown.
        """
        if kind not in EVENT_KINDS:
            raise PromptControlError(f"unknown event kind {kind!r}; expected one of {EVENT_KINDS}")
        previous = self.events()
        ts = max(time.time(), float(previous[-1]["ts"]) + 0.000001) if previous else time.time()
        body = {"kind": kind, "subject": subject, "payload": dict(payload), "actor": actor, "ts": ts}
        event_id = "e" + sha256_hex(canon(body))[:16]
        self.ws.upsert(PromptEventEntity, {"object_id": event_id, **body}, overwrite=False)
        return event_id

    def events(self, kind: str | None = None) -> list[Mapping[str, Any]]:
        """List events, oldest first.

        Args:
            kind (str | None): Restrict to one kind.

        Returns:
            list[Mapping[str, Any]]: Event rows.
        """
        rows = list(self.ws.query(PromptEventEntity).execute().rows())
        if kind is not None:
            rows = [row for row in rows if row.get("kind") == kind]
        return sorted(rows, key=lambda row: (float(row.get("ts") or 0.0), str(row.get("object_id") or "")))


# ---------------------------------------------------------------------------------------
# HB reference execution


class HbRenderer:
    """Render rows through HeavenBase's own restore path, bound to a workspace, with ``tr=str``.

    Text rows execute the template source HeavenBase generated; spec rows execute the compiled
    ``compose`` with part renderers built recursively from the same release; doc rows execute the
    ``compose`` compiled from their document, with no parts, and return every message joined by
    U+001E. Every row passes :func:`check_row_source` before it is handed to HeavenBase.
    """

    def __init__(self, ws: Any, prompt_type: type, rows: Mapping[str, Mapping[str, Any]]) -> None:
        """Bind the renderer.

        Args:
            ws (Any): Workspace supplying the module resolver (a temporary or the control workspace).
            prompt_type (type): The ``sys-prompt`` class resolved through that workspace.
            rows (Mapping[str, Mapping[str, Any]]): Row by name; one version per name, as a release has.
        """
        self.ws = ws
        self.Prompt = prompt_type
        self.rows = dict(rows)
        self._prompts: dict[str, Any] = {}
        self._callables: dict[str, Callable[..., str]] = {}

    @classmethod
    def for_release(cls, ws: Any, release: Any) -> "HbRenderer":
        """Bind a renderer to a loaded runtime release.

        Args:
            ws (Any): Workspace supplying the module resolver.
            release (Release): A verified release.

        Returns:
            HbRenderer: Over the release's rows.
        """
        from heavenbase.entity.registry import load_entity_type

        prompt_type = load_entity_type("sys-prompt", resolver=ws.module_resolver)
        return cls(ws, prompt_type, {name: release.row(name) for name in release.refs})

    def _prompt(self, name: str) -> Any:
        if name not in self._prompts:
            row = self.rows[name]
            check_row_source(row)
            self._prompts[name] = self.Prompt._from_row(dict(row), ws=self.ws)
        return self._prompts[name]

    def callable_for(self, name: str) -> Callable[..., str]:
        """A keyword-argument renderer for one row.

        Args:
            name (str): Row name.

        Returns:
            Callable[..., str]: ``f(**values) -> str``; text rows are called with ``tr=str``.
        """
        if name in self._callables:
            return self._callables[name]
        prompt = self._prompt(name)
        meta = self.rows[name]["manifest"]["meta"]
        if meta["kind"] == "text":
            def render_text(**values: Any) -> str:
                return str(prompt(tr=str, **values))

            fn: Callable[..., str] = render_text
        elif meta["kind"] == "doc":
            def render_doc(**values: Any) -> str:
                return str(prompt({}, dict(values)))

            fn = render_doc
        else:
            parts = {slot: self.callable_for(target) for slot, target in meta["parts"].items()}

            def render_spec(**values: Any) -> str:
                return str(prompt(parts, dict(values)))

            fn = render_spec
        self._callables[name] = fn
        return fn

    def render(self, name: str, **params: Any) -> str:
        """Render one row through HeavenBase.

        Args:
            name (str): Row name.
            **params (Any): Slot or parameter values.

        Returns:
            str: The rendered text.
        """
        return self.callable_for(name)(**params)


# ---------------------------------------------------------------------------------------
# Store export / import


def object_file(row: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a control row as a ``rubiksql.object/1`` store document.

    Args:
        row (Mapping[str, Any]): A ``sys-prompt`` row.

    Returns:
        dict[str, Any]: ``{"schema", "row", "export"}``; the export block is deterministic so a
            re-export of the same row is byte-identical.
    """
    key = row["manifest"]["meta"]["content_key"]
    return {"schema": SCHEMA_OBJECT, "row": dict(row), "export": {"content_key": key, "name": row["name"], "version": row["version"], "control_object_id": row["object_id"]}}


def export_objects(store_dir: Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, int]:
    """Write rows into ``<store>/objects/<name>/<version>.json`` write-if-absent.

    An object already present is kept when it carries the same content key (the same content
    published from another control workspace differs only in timestamps); a present object with
    a different content key is an error.

    Args:
        store_dir (Path): Store root.
        rows (Iterable[Mapping[str, Any]]): Rows to export.

    Returns:
        tuple[int, int]: Objects written and objects already present.

    Raises:
        PromptStoreError: If a present object holds a different content key.
    """
    written = kept = 0
    for row in rows:
        check_row_source(row)
        doc = object_file(row)
        path = store_dir / "objects" / row["name"] / f"{row['version']}.json"
        if path.is_file():
            import json

            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("schema") != SCHEMA_OBJECT or (existing.get("row") or {}).get("manifest", {}).get("meta", {}).get("content_key") != doc["export"]["content_key"]:
                raise PromptStoreError(f"{path}: exists with a different content key; store objects are write-once")
            kept += 1
            continue
        write_if_absent(path, doc)
        written += 1
    return written, kept


def read_objects(store_dir: Path) -> list[dict[str, Any]]:
    """Read every object file of a store directory.

    Args:
        store_dir (Path): Store root.

    Returns:
        list[dict[str, Any]]: Documents sorted by path.
    """
    import json

    directory = store_dir / "objects"
    if not directory.is_dir():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*/*.json"))]


def baseline_text_files(release: Any) -> dict[str, bytes]:
    """The human-readable mirror of a release: one file per row it renders.

    Text and doc rows become ``<name>.md`` holding exactly ``meta.text`` (a role's whole document,
    for the one-document layout); spec rows become ``<name>.spec.json`` holding the spec, parts,
    params, description and reason in canonical JSON. A dropped entry the release still lists
    (``Release.dropped_entries``) is never rendered and has no file. A ``rubiksql.closure/2`` release
    carries no data carrier rows; its parameter set and rule versions mirror as ``config.knobs.md`` and
    ``rules.general.md`` holding exactly the legacy texts, so the mirror of a representation-only
    migration is byte-identical to its closure/1 parent's. The runtime never reads these files; they
    exist so a promotion PR shows a readable diff.

    Args:
        release (Release): A loaded release.

    Returns:
        dict[str, bytes]: Relative path to file bytes.
    """
    from . import entries as E
    from .entity_model import CLOSURE_V2
    from .rules import corpus_text

    files: dict[str, bytes] = {}
    dropped = set(getattr(release, "dropped_entries", ()))
    for name in sorted(set(release.refs) - dropped):
        row = release.row(name)
        meta = row["manifest"]["meta"]
        if meta["kind"] in ("text", "doc"):
            files[f"{name}.md"] = meta["text"].encode("utf-8")
        else:
            files[f"{name}.spec.json"] = canonical_bytes({"name": name, "version": row["version"], "description": row["description"], "reason": row["manifest"].get("reason", ""), "params": meta["params"], "parts": meta["parts"], "spec": meta["spec"]})
    if getattr(release, "rules", None) == CLOSURE_V2:
        files[f"{E.KNOBS_ENTRY}.md"] = E.knobs_text(release.parameter_values()).encode("utf-8")
        files[f"{E.RULES_ENTRY}.md"] = corpus_text(release.rules_general()).encode("utf-8")
    return files


class SharedHomeGuard:
    """Snapshot the machine-wide HeavenBase home's workspace registry and active selection.

    Tests take a snapshot before opening any workspace and assert nothing changed afterwards:
    author-side work must never register or activate a workspace in ``~/.heavenbase`` by
    accident. Only ``prompts register`` may, and tests mock that call.
    """

    def __init__(self, context: Any = None) -> None:
        self.context = context
        self.before = self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Read the registry rows and the active workspace of the default (or given) context.

        Args:
            None.

        Returns:
            dict[str, Any]: ``{"active": id | None, "workspaces": {id: {"updated_at", "manifest_sha", "construction_sha"}}}``.
        """
        from heavenbase.context import context_or_default
        from heavenbase.workspace import WorkspaceSpec

        context = context_or_default(self.context)
        specs = WorkspaceSpec.list(context=context)
        return {
            "active": context.active_workspace_id,
            "workspaces": {
                spec.id: {"updated_at": spec.updated_at, "manifest_sha": sha256_hex(canon(spec.manifest)), "construction_sha": sha256_hex(canon(spec.construction.to_dict()))}
                for spec in specs
            },
        }

    def changes(self) -> list[str]:
        """Describe what changed since the first snapshot.

        Args:
            None.

        Returns:
            list[str]: Human-readable differences; empty when nothing changed.
        """
        after = self.snapshot()
        diffs = []
        if after["active"] != self.before["active"]:
            diffs.append(f"active workspace {self.before['active']!r} -> {after['active']!r}")
        for ws_id in sorted(set(self.before["workspaces"]) | set(after["workspaces"])):
            old, new = self.before["workspaces"].get(ws_id), after["workspaces"].get(ws_id)
            if old != new:
                diffs.append(f"workspace {ws_id!r}: {old} -> {new}")
        return diffs

    def assert_unchanged(self) -> None:
        """Raise when the shared home changed.

        Args:
            None.

        Returns:
            None: This method does not return a value.

        Raises:
            AssertionError: Listing every change.
        """
        diffs = self.changes()
        if diffs:
            raise AssertionError("the shared HeavenBase home changed: " + "; ".join(diffs))
