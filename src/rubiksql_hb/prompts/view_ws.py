"""Build the private verified dashboard view of the current authority release.

Only the derived view is registered. The authority stays detached and immutable;
older versions and decisions are read through authority-backed history tools.
All current prompt rows in this view are active, so native bare-name loads select
the exact current version. Registration remains a human-only CLI operation.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .canon import canon, sha256_hex
from .control import PromptControlError, check_row_source, ensure_private_dir, main_only_backends
from .store import Store

__all__ = [
    "DEFAULT_VIEW_ROOT",
    "ENV_VIEW",
    "VIEW_DB_NAME",
    "VIEW_WS_ID",
    "RegisterResult",
    "ViewBuild",
    "build_view",
    "register_view",
    "unregister_view",
    "view_root",
]

ENV_VIEW = "RUBIKSQL_PROMPT_VIEW"
DEFAULT_VIEW_ROOT = Path("~/.rubiksql_hb/prompt_view")
VIEW_WS_ID = "rubiksql_evolution_view"
VIEW_DB_NAME = "rubiksql_evolution_view.db"


@dataclass
class ViewBuild:
    """A built snapshot view.

    Attributes:
        directory (Path): The final directory holding the sqlite file.
        ws_id (str): Workspace id.
        baseline (str): The BASELINE release id the view is projected against.
        digest (str): Digest of the BASELINE release id (the directory name). The id is a content key over the
            closure, the only content the view mirrors, so an unchanged BASELINE rebuilds the same directory.
        counts (dict[str, int]): ``active`` (the BASELINE closure's rows) and ``translations`` (always 0; a
            translation row fails the build).
        ws (Any): The reopened workspace at ``directory`` (detached).
    """

    directory: Path
    ws_id: str
    baseline: str
    digest: str
    counts: dict[str, int] = field(default_factory=dict)
    ws: Any = None


@dataclass
class RegisterResult:
    """What ``prompts register`` did."""

    build: ViewBuild
    previous_active: str | None
    restore_command: str
    registered: bool


def view_root(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the directory that holds view builds.

    Args:
        env (Mapping[str, str] | None): Environment to read; ``os.environ`` by default.

    Returns:
        Path: ``RUBIKSQL_PROMPT_VIEW`` or ``~/.rubiksql_hb/prompt_view``.
    """
    raw = (env if env is not None else os.environ).get(ENV_VIEW, "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_VIEW_ROOT.expanduser()


def _release_backends(ws: Any) -> None:
    """Dispose the sqlite engines of a detached workspace so its directory can be moved."""
    for name in list(ws.backends.names()):
        backend = ws.backends.get(name)
        database = getattr(backend, "db", None)
        for method in ("dispose", "close"):
            fn = getattr(database, method, None)
            if callable(fn):
                try:
                    fn()
                except Exception:  # best effort: the handles are dropped with the object anyway
                    pass
                break


def _open(directory: Path, context: Any) -> Any:
    import heavenbase as hb

    ws = hb.HeavenBase(VIEW_WS_ID, context=context, backends=main_only_backends(directory / VIEW_DB_NAME), detached=True)
    ws.enable_extension("prompt")
    return ws


def build_view(store: Store, *, root: Path | None = None, context: Any = None) -> ViewBuild:
    """Rebuild the snapshot view from a store.

    Args:
        store (Store): The store; the BASELINE pointer and integrity files come from its package layer.
        root (Path | None): Directory receiving the build; :func:`view_root` by default.
        context (Any): HeavenBase machine context; the process default when omitted.

    Returns:
        ViewBuild: The build, reopened at its final directory.

    Raises:
        PromptStoreError: If BASELINE cannot be loaded.
        PromptControlError: If an object fails :func:`control.check_row_source` or a translation row exists.
    """
    from heavenbase.entity.registry import load_entity_type
    from heavenbase.extensions.prompt.name import prompt_row_id

    from .runtime import load_release

    root = root if root is not None else view_root()
    pointer = store.baseline()
    baseline = load_release(pointer.release, stores=store)
    digest = sha256_hex(canon({"baseline": baseline.id}))[:12]
    final = root / digest
    build_dir = root / f".build-{digest}-{os.getpid()}"
    ensure_private_dir(root)
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()
    counts = {"active": 0, "translations": 0}
    ws = _open(build_dir, context)
    try:
        prompt_type = load_entity_type("sys-prompt", resolver=ws.module_resolver)
        translation_type = load_entity_type("sys-translation", resolver=ws.module_resolver)
        for name, key in sorted(baseline.refs.items()):
            relative = f"objects/{name}/{key[:16]}.json"
            row = dict(baseline.row(name))
            try:
                check_row_source(row)
            except PromptControlError as error:
                raise PromptControlError(f"{relative}: {error}") from error
            if row["manifest"]["meta"]["content_key"] != key:
                raise PromptControlError(f"{relative}: the row's content key differs from BASELINE's ref")
            row["status"] = "active"
            # Key the view row the way HeavenBase keys prompt rows, so describe/load by name:version resolve.
            row["object_id"] = prompt_row_id(row["name"], row["version"])
            ws.upsert(prompt_type, row)
            counts["active"] += 1
        translations = list(ws.query(translation_type).execute().rows())
        counts["translations"] = len(translations)
        if translations:
            raise PromptControlError(f"the view carries {len(translations)} sys-translation rows; translation is never enabled")
        if counts["active"] != len(baseline.refs):
            raise PromptControlError(f"{counts['active']} active rows but BASELINE {baseline.id} has {len(baseline.refs)} refs")
    finally:
        _release_backends(ws)
        del ws
    if final.exists():
        retired = root / f".old-{digest}-{os.getpid()}"
        os.replace(final, retired)
        shutil.rmtree(retired, ignore_errors=True)
    os.replace(build_dir, final)
    reopened = _open(final, context)
    return ViewBuild(final, VIEW_WS_ID, baseline.id, digest, counts, reopened)


def register_view(store: Store, *, root: Path | None = None, context: Any = None) -> RegisterResult:
    """Build the view and register it in the HeavenBase registry (the ``register`` verb).

    Args:
        store (Store): The store to project.
        root (Path | None): View root directory.
        context (Any): HeavenBase machine context.

    Returns:
        RegisterResult: The build, the previously active workspace and the command that restores it.
    """
    from heavenbase.context import context_or_default
    from heavenbase.workspace import WorkspaceSpec

    from ..workspace import _registry_manifest

    build = build_view(store, root=root, context=context)
    selected = context_or_default(context)
    previous = selected.active_workspace_id
    WorkspaceSpec.register(build.ws_id, manifest=_registry_manifest(build.ws), context=build.ws.context)
    restore = f"hb ws use {previous}" if previous else "hb ws use <your workspace>  (no workspace was active)"
    return RegisterResult(build, previous, restore, True)


def unregister_view(*, context: Any = None) -> bool:
    """Remove the view from the HeavenBase registry (the ``unregister`` verb); the directory stays.

    Args:
        context (Any): HeavenBase machine context.

    Returns:
        bool: Whether a registration existed.
    """
    from heavenbase.workspace import WorkspaceSpec

    return bool(WorkspaceSpec.unregister(VIEW_WS_ID, context=context))
