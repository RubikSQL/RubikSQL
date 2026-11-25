"""Load the ordinary package authority binding without runtime or HB imports."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import urllib.request
import uuid
from collections.abc import Sequence
from pathlib import Path

from .types import AuthorityBinding, GitAuthorityBinding

SCHEMA_BINDING = "rubiksql.authority-binding/1"
SCHEMA_GIT_BINDING = "rubiksql.authority-binding/2"
AUTHORITY_REF = "refs/heads/authority/v0"
AUTHORITY_REMOTE = "git@github.com:edsml-jz2321/RubikSQL-HB.git"
_REMOTES = {AUTHORITY_REMOTE, "https://github.com/edsml-jz2321/RubikSQL-HB.git"}
PACKAGE_BINDING = Path(__file__).with_name("AUTHORITY.json")
_REF_PATTERN = re.compile(r"refs/heads/authority/[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_SELECTED_BINDING: tuple[Path, str, AuthorityBinding | GitAuthorityBinding] | None = None
_SELECTION_LOCK = threading.Lock()


class BindingError(RuntimeError):
    """An authority binding or the machine identity is invalid."""


def valid_authority_ref(ref: object) -> bool:
    """Accept one explicit safe branch name within the authority namespace."""
    return isinstance(ref, str) and _REF_PATTERN.fullmatch(ref) is not None


def load_binding(path: Path | None = None) -> AuthorityBinding | GitAuthorityBinding:
    """Load and validate the package binding or an explicit authoring file.

    Args:
        path: Explicit binding JSON; otherwise the selected immutable file or
            the installed package's AUTHORITY.json.

    Returns:
        The validated, canonical binding. No environment override is accepted.

    Raises:
        BindingError: Missing, malformed or noncanonical binding.
    """
    if path is None and _SELECTED_BINDING is not None:
        source, expected, selected = _SELECTED_BINDING
        try:
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise BindingError(f"selected authority binding {source} changed during this process")
        except OSError as error:
            raise BindingError(f"selected authority binding {source} is unavailable") from error
        return selected
    source = path if path is not None else PACKAGE_BINDING
    return _read_binding(Path(source))


def _read_binding(source: Path, raw: bytes | None = None) -> AuthorityBinding | GitAuthorityBinding:
    try:
        body = json.loads(source.read_bytes() if raw is None else raw)
        if not isinstance(body, dict):
            raise ValueError("expected an authority identity object")
        if str(uuid.UUID(body["authority_uuid"])) != body["authority_uuid"]:
            raise ValueError("authority_uuid must be a canonical UUID")
        if body.get("schema") == SCHEMA_GIT_BINDING:
            if set(body) != {"schema", "authority_uuid", "remote", "ref"}:
                raise ValueError("Git binding requires exactly schema, authority_uuid, remote and ref")
            if body["remote"] not in _REMOTES or not valid_authority_ref(body["ref"]):
                raise ValueError("Git authority remote or full ref differs from the canonical repository")
            return GitAuthorityBinding(SCHEMA_GIT_BINDING, body["authority_uuid"], body["remote"], body["ref"])
        if body.get("schema") != SCHEMA_BINDING or set(body) != {"schema", "authority_uuid", "instance_id", "db_path"}:
            raise ValueError("unsupported authority binding schema or fields")
        instance = body["instance_id"]
        if not isinstance(instance, str) or not instance.startswith("i-") or not instance[2:].isalnum():
            raise ValueError("instance_id must be an ECS instance id")
        db_path = body["db_path"]
        if (not isinstance(db_path, str) or not Path(db_path).is_absolute()
                or Path(db_path).suffix != ".db" or str(Path(db_path).resolve()) != db_path):
            raise ValueError("db_path must be an absolute canonical path")
        return AuthorityBinding(SCHEMA_BINDING, body["authority_uuid"], instance, Path(db_path))
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise BindingError(f"authority binding {source} refused: {error}") from error


def select_binding(path: Path) -> AuthorityBinding | GitAuthorityBinding:
    """Select one immutable explicit binding file for this process and its threads.

    Args:
        path: Absolute binding file path supplied by the caller.

    Returns:
        The validated binding shared by every thread in this process.

    Raises:
        BindingError: Relative or invalid file, changed bytes, or a second
            selection of a different file or identity.
    """
    source = Path(path)
    if not source.is_absolute():
        raise BindingError("explicit authority binding path must be absolute")
    source = source.resolve()
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise BindingError(f"authority binding {source} is unavailable") from error
    selected = _read_binding(source, raw)
    selection = (source, hashlib.sha256(raw).hexdigest(), selected)
    global _SELECTED_BINDING
    with _SELECTION_LOCK:
        if _SELECTED_BINDING is not None and _SELECTED_BINDING != selection:
            raise BindingError("authority binding selection cannot change during this process")
        _SELECTED_BINDING = selection
    return load_binding()


def selected_binding_path() -> Path | None:
    """Return the verified explicitly selected file for child-process propagation."""
    if _SELECTED_BINDING is None:
        return None
    load_binding()
    return _SELECTED_BINDING[0]


def binding_arguments() -> list[str]:
    """Return the same explicit authority selector for each child entry point."""
    path = selected_binding_path()
    return ["--authority-binding", str(path)] if path is not None else []


def select_binding_from_argv(argv: Sequence[str]) -> AuthorityBinding | GitAuthorityBinding | None:
    """Select an explicit CLI binding before runtime bootstrap without consuming argv.

    Args:
        argv: The command arguments excluding the executable name. Arguments
            after ``--`` belong to a child command and are not scanned.

    Returns:
        The explicitly selected binding, or None when the flag is absent.

    Raises:
        BindingError: Duplicate selector, missing value, or invalid selection.
    """
    paths = []
    for index, argument in enumerate(argv):
        if argument == "--":
            break
        if argument == "--authority-binding":
            if index + 1 == len(argv) or argv[index + 1].startswith("--"):
                raise BindingError("--authority-binding requires an absolute file path")
            paths.append(argv[index + 1])
        elif argument.startswith("--authority-binding="):
            paths.append(argument.partition("=")[2])
    if len(paths) > 1:
        raise BindingError("--authority-binding may be supplied only once")
    return select_binding(Path(paths[0])) if paths else None


def binding_document(binding: AuthorityBinding | GitAuthorityBinding) -> dict[str, str]:
    """Return the exact JSON representation of a binding."""
    if isinstance(binding, GitAuthorityBinding):
        return {"schema": binding.schema, "authority_uuid": binding.authority_uuid,
                "remote": binding.remote, "ref": binding.ref}
    return {"schema": binding.schema, "authority_uuid": binding.authority_uuid,
            "instance_id": binding.instance_id, "db_path": str(binding.db_path)}


def write_binding(binding: AuthorityBinding | GitAuthorityBinding, path: Path | None = None) -> Path:
    """Atomically publish a validated ordinary binding file.

    Args:
        binding: Complete identity to publish.
        path: Package default or explicit task package path.

    Returns:
        Written binding path.
    """
    target = path if path is not None else PACKAGE_BINDING
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        temporary.write_text(json.dumps(binding_document(binding), sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        load_binding(temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def metadata_instance_id() -> str:
    """Read ECS instance identity through IMDS with proxies disabled.

    Returns:
        Instance id reported by the local metadata service.

    Raises:
        BindingError: Identity is unavailable or malformed. This has no fallback.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = "http://100.100.100.200"
    try:
        request = urllib.request.Request(base + "/latest/api/token", method="PUT",
                                         headers={"X-aliyun-ecs-metadata-token-ttl-seconds": "60"})
        with opener.open(request, timeout=3) as response:
            token = response.read(4096).decode("utf-8").strip()
        request = urllib.request.Request(base + "/latest/meta-data/instance-id",
                                         headers={"X-aliyun-ecs-metadata-token": token})
        with opener.open(request, timeout=3) as response:
            instance = response.read(4096).decode("utf-8").strip()
        if not instance.startswith("i-") or not instance[2:].isalnum():
            raise ValueError("invalid instance id")
        return instance
    except (OSError, ValueError) as error:
        raise BindingError("ECS instance identity unavailable; authority writes refuse") from error


instance_id = metadata_instance_id
