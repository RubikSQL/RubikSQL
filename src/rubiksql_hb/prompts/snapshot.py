"""Verified read-only execution snapshots, without HB or provider bootstrap.

A snapshot is the deterministic projection of its own native version rows plus the execution
closure of one pinned release. New snapshots are ``rubiksql.prompt-snapshot/2``: their closed scope
(:data:`store.TABLES`) includes the canonical Tool, rule and parameter-set versions, and a
``rubiksql.closure/2`` release executes the typed answering bundle its references pin. Historical
``rubiksql.prompt-snapshot/1`` snapshots (:data:`store.LEGACY_TABLES`, closure/1 releases only)
still verify exactly as recorded.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
import errno
from functools import partial
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import threading
from typing import Any, TYPE_CHECKING

from rubiksql_hb.file_identity import file_identity

from .binding import load_binding
from .types import AuthorityBinding, AuthorityRevision, GitAuthorityBinding, VerifiedSnapshot

if TYPE_CHECKING:
    from .authority import AuthorityRead

ENV_SNAPSHOT = "RUBIKSQL_PROMPT_SNAPSHOT"
ENV_BUNDLE = "RUBIKSQL_TOOL_BUNDLE"
#: Historical snapshot schema: the six historical version sets and closure/1 releases only.
SCHEMA_SNAPSHOT = "rubiksql.prompt-snapshot/1"
#: Current snapshot schema: the nine current version sets, typed rows included (possibly empty).
SCHEMA_SNAPSHOT_V2 = "rubiksql.prompt-snapshot/2"
#: Gate receipt schema of an answering-tool publication.
_TOOL_GATE = "rubiksql.tool-gate/1"

# A content address is admitted only after every file has passed its hash check.
# Reuse that admission only while the complete inventory/stat identity is unchanged.
_VERIFIED_CONTENT: set[tuple[Path, str, AuthorityBinding | GitAuthorityBinding]] = set()
_VERIFIED_EPOCHS: set[str] = set()
_SnapshotIdentity = tuple[tuple[str | int, ...], ...]
_VERIFIED_SNAPSHOTS: dict[
    tuple[Path, AuthorityBinding | GitAuthorityBinding, str | None, str | None, str | None, str | None],
    tuple[_SnapshotIdentity, VerifiedSnapshot],
] = {}
_VERIFIED_CONTENT_LOCK = threading.Lock()
_BYTE_VERIFICATION_REQUIRED = False


class SnapshotError(RuntimeError):
    """Snapshot content or its immutable pin could not be verified."""


def reject_execution_controls(environ: Mapping[str, str]) -> None:
    """Refuse removed runtime controls while keeping explicit historical reads available."""
    if environ.get("RUBIKSQL_PROMPT_STORE", "").strip():
        raise SnapshotError("runtime prompt overlays are removed; use the authority snapshot")
    if environ.get("RUBIKSQL_PROMPT_ALLOW_RETIRED", "").strip():
        raise SnapshotError("retired-release execution control is removed; use explicit history reads")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, ValueError) as exc:
        raise SnapshotError(f"snapshot cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SnapshotError(f"snapshot {path}: expected JSON object")
    return value


def _relative(name: object) -> str:
    if not isinstance(name, str) or not name or "\\" in name:
        raise SnapshotError("snapshot file name must be a POSIX relative path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts) or str(path) != name:
        raise SnapshotError(f"snapshot has unsafe path {name!r}")
    return name


def _payload(path: Path, release: Mapping[str, Any]) -> Mapping[str, Any] | None:
    key = release.get("refs", {}).get("tools.answering")
    if key is None:
        return None
    row = _json(path / "objects" / "tools.answering" / f"{str(key)[:16]}.json").get("row", {})
    try:
        payload = json.loads(row["manifest"]["meta"]["text"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotError("snapshot tools.answering is not canonical JSON data") from exc
    if not isinstance(payload, dict):
        raise SnapshotError("snapshot tools.answering must be a JSON object")
    return payload


def _scope(schema: object) -> Mapping[str, str]:
    """Return the closed native version scope of a snapshot schema."""
    from .store import LEGACY_TABLES, TABLES

    return TABLES if schema == SCHEMA_SNAPSHOT_V2 else LEGACY_TABLES


def _release_rules(name: str, document: Mapping[str, Any], schema: object) -> str:
    """Check a release file's schema/rules pair, its snapshot format and closure/2 invariants."""
    from .entity_model import CLOSURE_V2, EntityModelError, check_release_document
    from .store import PromptStoreError, check_release_format

    try:
        rules = check_release_format(document)
    except PromptStoreError as exc:
        raise SnapshotError(f"snapshot release {name}: {exc}") from exc
    if rules == CLOSURE_V2:
        if schema != SCHEMA_SNAPSHOT_V2:
            raise SnapshotError(f"snapshot {schema} cannot hold the rubiksql.closure/2 release {name}")
        try:
            check_release_document(document)
        except EntityModelError as exc:
            raise SnapshotError(f"snapshot release {name} has an invalid typed closure: {exc}") from exc
    return rules


def _verify_epoch(path: Path, listed: set[str], manifest: Mapping[str, Any]) -> None:
    """Validate every retained release and row, including historical-only refs.

    The native version files of the snapshot's scope must all exist; every canonical typed row must
    verify its own identity, and every closure/2 release must resolve its exact typed closure.
    """
    from rubiksql_hb.hb_tool_entities import validate_answering, verify_tool_row
    from .canon import canon, sha256_hex
    from .entity_model import (CLOSURE_V2, PARAMETER_ENTITY, RULE_ENTITY, TOOL_ENTITY, EntityModelError, closure_changes,
                               is_canonical, resolve_closure, verify_parameter_row, verify_rule_row)
    from .integrity import check_prompt_row
    from .runtime import _Loader
    from .store import SCHEMA_OBJECT, SCHEMA_PROPOSAL, TYPED_TABLES, Store, project_version_files, publication_ancestry

    # Retained history is verified for integrity only; current rule and knob policy applies at publication and load.
    verifiers = {TOOL_ENTITY: verify_tool_row, RULE_ENTITY: partial(verify_rule_row, policy=False),
                 PARAMETER_ENTITY: partial(verify_parameter_row, policy=False)}
    try:
        releases = {}
        proposals = {}
        objects = {}
        for name in sorted(listed):
            if name.startswith("objects/"):
                document = _json(path / name)
                row = document["row"]
                if document.get("schema") != SCHEMA_OBJECT:
                    raise SnapshotError(f"snapshot object schema mismatch {name}")
                check_prompt_row(row)
                key = row["manifest"]["meta"]["content_key"]
                expected_export = {"content_key": key, "name": row["name"], "version": row["version"],
                                   "control_object_id": row["object_id"]}
                if document.get("export") != expected_export:
                    raise SnapshotError(f"snapshot object export identity mismatch {name}")
                objects[row["object_id"]] = row
            elif name.startswith("releases/"):
                document = _json(path / name)
                rid = document["id"]
                _release_rules(name, document, manifest["schema"])
                if name != f"releases/{rid}.json":
                    raise SnapshotError(f"snapshot release schema/filename mismatch {name}")
                if document.get("epoch") != manifest["epoch"]:
                    raise SnapshotError(f"snapshot release belongs to another epoch {rid}")
                releases[rid] = document
            elif name.startswith("proposals/"):
                document = _json(path / name)
                pid = document["id"]
                if document.get("schema") != SCHEMA_PROPOSAL or name != f"proposals/{pid}.json":
                    raise SnapshotError(f"snapshot proposal schema/filename mismatch {name}")
                proposals[pid] = document
        root = _json(path / "V0.json")["release"]
        if root not in releases or manifest["release"] not in releases:
            raise SnapshotError("snapshot lacks its current or epoch-root release")
        if _json(path / "BASELINE.json").get("release") not in releases:
            raise SnapshotError("snapshot baseline points outside its epoch")
        native = {}
        for entity in _scope(manifest["schema"]):
            name = f"native/{entity}.json"
            if name not in listed:
                raise SnapshotError("snapshot lacks a complete closed native version projection")
            rows = json.loads((path / name).read_bytes())
            if (not isinstance(rows, list) or any(not isinstance(row, dict)
                    or not isinstance(row.get("object_id"), str) for row in rows)):
                raise SnapshotError("snapshot native version file must contain an array of identified rows")
            native[entity] = rows
        typed: dict[str, dict[str, Mapping[str, Any]]] = {entity: {} for entity in TYPED_TABLES}
        for entity in TYPED_TABLES:
            for row in native.get(entity, ()):
                try:
                    if not is_canonical(entity, row):
                        raise EntityModelError("a projection row is not a canonical typed version")
                    verifiers[entity](row)
                except ValueError as exc:
                    raise SnapshotError(f"snapshot typed {entity} row {row['object_id']!r} is invalid: {exc}") from exc
                typed[entity][row["object_id"]] = row
        capsules = {row["object_id"]: row for row in native["sys-capsule"]}
        toolkits = {row["object_id"]: row for row in native["sys-toolkit"]}
        from .toolkit_retirement import verify_history as verify_toolkit_history

        verify_toolkit_history(list(releases.values()), list(objects.values()), capsules, toolkits,
                              sorted(native["prompt-event-entity"], key=lambda item: (float(item["ts"]), str(item["object_id"]))))
        store = Store(package=path)
        from .fresh_epoch import validate_native_admission
        state = validate_native_admission(native)
        if any(event["kind"] == "register" and event["payload"].get("schema") == "rubiksql.authority-state/1"
               for event in native["prompt-event-entity"]):
            from .authority_state import validate_native_state

            state = validate_native_state(native)
        for rid, release in releases.items():
            # This loader is pure: it regenerates prompt source and checks every
            # ref, entry, spec/document closure and data-row contract. Historical
            # revocation is recorded evidence, not an execution override here.
            _Loader(store, allow_retired=True, check_revoked=False, policy=False).load(
                rid, resolved_from="snapshot-verification", anchor=None)
            if release["rules"] == CLOSURE_V2:
                try:
                    resolve_closure(release["content_refs"], typed, capsule_rows=capsules, policy=False)
                except EntityModelError as exc:
                    raise SnapshotError(f"snapshot release {rid} does not resolve its typed closure: {exc}") from exc
            pid = release.get("proposal")
            if state is not None and rid == state["payload"]["release"]:
                expected_origin = "v0" if state["payload"]["schema"] == "rubiksql.fresh-epoch/1" else "state"
                if (release.get("origin") != expected_origin or release.get("authority_state") != state["object_id"]
                        or pid is not None or release.get("parents")):
                    raise SnapshotError(f"snapshot release differs from its exact state admission {rid}")
            elif rid != root and (pid not in proposals or proposals[pid]["release"] != rid):
                raise SnapshotError(f"snapshot release lacks its publication proposal {rid}")
            if pid is not None and release.get("parents") != proposals[pid]["parents"]:
                raise SnapshotError(f"snapshot release/proposal parent links differ {rid}")
        for pid, proposal in proposals.items():
            rid = proposal["release"]
            if rid not in releases:
                raise SnapshotError(f"snapshot proposal names an absent release {pid}")
            bases = [item["release"] for item in proposal["parents"] if item.get("role", "base") == "base"]
            refs = [item["release"] for item in proposal["parents"]]
            if len(bases) != 1 or any(parent not in releases or parent == rid for parent in refs):
                raise SnapshotError(f"snapshot proposal parent closure is invalid {pid}")
            # For a closure/1 pair this is exactly the per-name refs diff.
            expected = closure_changes(releases[bases[0]], releases[rid], typed)
            changes = proposal["changes"]
            if (not isinstance(changes, Mapping) or set(changes) != set(expected)
                    or any(not isinstance(change, Mapping) or change.get("from") != expected[name]["from"]
                           or change.get("to") != expected[name]["to"] for name, change in changes.items())):
                raise SnapshotError(f"snapshot proposal changes disagree with its release closures {pid}")
        publication_ancestry(releases, proposals)
        events = [json.loads(line) for line in (path / "events.jsonl").read_bytes().splitlines() if line]
        if events != sorted(events, key=lambda event: (float(event["ts"]), str(event["object_id"]))):
            raise SnapshotError("snapshot event history is not ordered")
        ids = [event["object_id"] for event in events]
        if len(set(ids)) != len(ids):
            raise SnapshotError("snapshot event history repeats an event identity")
        for event in events:
            body = {key: event[key] for key in ("kind", "subject", "payload", "actor", "ts")}
            if event["object_id"] != "e" + sha256_hex(canon(body))[:16]:
                raise SnapshotError("snapshot event body differs from its identity")
        revision = manifest["authority_revision"]
        if revision != {"event_count": len(events), "latest_event_id": ids[-1] if ids else None}:
            raise SnapshotError("snapshot revision differs from its complete event history")
        if {row["object_id"]: row for row in native["sys-prompt"]} != objects:
            raise SnapshotError("snapshot native prompts and object history differ")
        if sorted(native["prompt-event-entity"], key=lambda event: (float(event["ts"]), str(event["object_id"]))) != events:
            raise SnapshotError("snapshot native events and event history differ")
        for release in releases.values():
            payload = _payload(path, release)
            if payload is not None:
                if payload.get("schema") == "rubiksql.answering-bundle/1":
                    if release["id"] == manifest["release"]:
                        raise SnapshotError("schema1 Toolkit execution is permanently retired")
                else:
                    validate_answering(payload, capsule_rows=capsules)
        # The regenerated projection uses the scope the snapshot schema names.
        projected = project_version_files(native, tools_release=manifest["release"])
        if set(projected) != listed or any((path / name).read_bytes() != raw for name, raw in projected.items()):
            raise SnapshotError("snapshot closed version projection paths or bodies differ from native history")
    except SnapshotError:
        raise
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise SnapshotError(f"snapshot epoch closure is invalid: {exc}") from exc


def _verify_typed_tools(path: Path, listed: set[str], release: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    """Verify a closure/2 pin's typed bundle digest and exact inert execution files.

    Returns:
        str: The digest of the release's ``rubiksql.answering-bundle/3`` document.
    """
    from rubiksql_hb.hb_tool_entities import digest, typed_bundle, typed_snapshot_files, validate_typed_tools

    content_refs = release["content_refs"]
    bundle_digest = digest(typed_bundle(content_refs))
    if bundle_digest != manifest.get("bundle_digest"):
        raise SnapshotError("snapshot bundle digest mismatch")
    try:
        capsules = {item.stem: _json(item) for item in (path / "tools" / "capsules").glob("*.json")}
        tool_rows = {row["object_id"]: row for row in json.loads((path / "native" / "tool-entity.json").read_bytes())}
        expected_tools = typed_snapshot_files(content_refs, capsules)
        validate_typed_tools(content_refs, tool_rows=tool_rows, capsule_rows=capsules)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SnapshotError(f"snapshot typed tool closure is invalid: {exc}") from exc
    if {name for name in listed if name.startswith("tools/")} != set(expected_tools):
        raise SnapshotError("snapshot tool files exceed the pinned execution closure")
    if any((path / name).read_bytes() != raw for name, raw in expected_tools.items()):
        raise SnapshotError("snapshot module source files differ from their pinned source Capsules")
    return bundle_digest


def _snapshot_identity(path: Path) -> _SnapshotIdentity:
    """Inspect every directory/member and reject symlinks before reusing verified bytes."""
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise SnapshotError("snapshot root symlink refused")
    if (path / "SNAPSHOT.json").is_symlink():
        raise SnapshotError("snapshot manifest symlink refused")
    try:
        identity = [("", *file_identity(path))]
        for item in sorted(path.rglob("*")):
            if item.is_symlink():
                raise SnapshotError(f"snapshot symlink refused: {item.relative_to(path)}")
            identity.append((item.relative_to(path).as_posix(), *file_identity(item)))
        return tuple(identity)
    except OSError as exc:
        raise SnapshotError(f"snapshot cannot inspect {path}: {exc}") from exc


def require_snapshot_byte_verification() -> None:
    """Permanently require complete snapshot byte and closure checks in this process.

    Call during trusted preparation, before an audit guard or concurrent snapshot
    verification starts. No native file-identity API is used in this mode. Cache
    resets and fork handlers preserve this requirement.

    Returns:
        None: This function does not return a value.
    """
    global _BYTE_VERIFICATION_REQUIRED
    with _VERIFIED_CONTENT_LOCK:
        _BYTE_VERIFICATION_REQUIRED = True
        _VERIFIED_CONTENT.clear()
        _VERIFIED_EPOCHS.clear()
        _VERIFIED_SNAPSHOTS.clear()


def verify_snapshot(path: Path, *, expected_release: str | None = None,
                    expected_digest: str | None = None,
                    expected_authority_commit: str | None = None) -> VerifiedSnapshot:
    """Verify only version/execution bytes, never canonical QA or conversations.

    Every call checks the complete inventory, symlinks and authority binding.
    Normally, unchanged stat identities and expected pins reuse fully verified bytes
    without taking the cold-verification lock; changed identities recheck all bytes.
    Requiring byte verification disables native identity reads and both verification
    memos, checking all bytes and semantic closure under the lock on every call.

    Args:
        path: Directory containing SNAPSHOT.json.
        expected_release: Exact release id already recorded by a launcher.
        expected_digest: Digest already recorded by a run manifest.
        expected_authority_commit: Exact Git authority commit recorded by a launcher.

    Returns:
        VerifiedSnapshot: Immutable provenance for the verified directory.

    Raises:
        SnapshotError: Missing files, unsafe paths, altered hashes or identities, an unknown snapshot or
            release schema, or a typed row, reference or closure that differs.
    """
    return _verified_snapshot(path, expected_release=expected_release, expected_digest=expected_digest,
                              expected_authority_commit=expected_authority_commit)


def _verified_snapshot(path: Path, *, expected_release: str | None = None, expected_digest: str | None = None,
                       expected_authority_commit: str | None = None, pin: str | None = None) -> VerifiedSnapshot:
    """Share verification while separating a runtime ref's immutable selection from launcher checks."""
    path = Path(path).absolute()
    binding = load_binding()
    if _BYTE_VERIFICATION_REQUIRED:
        with _VERIFIED_CONTENT_LOCK:
            return _verify_snapshot_bytes(path, binding, expected_release=expected_release,
                                          expected_digest=expected_digest,
                                          expected_authority_commit=expected_authority_commit)
    key = (path, binding, expected_release, expected_digest, expected_authority_commit, pin)
    identity = _snapshot_identity(path)
    hit = _VERIFIED_SNAPSHOTS.get(key)
    if hit is not None and hit[0] == identity:
        return replace(hit[1], tiers=dict(hit[1].tiers))
    with _VERIFIED_CONTENT_LOCK:
        if _BYTE_VERIFICATION_REQUIRED:
            return _verify_snapshot_bytes(path, binding, expected_release=expected_release,
                                          expected_digest=expected_digest,
                                          expected_authority_commit=expected_authority_commit)
        # Single-flight cold admission; verified hot callers never acquire this lock.
        identity = _snapshot_identity(path)
        hit = _VERIFIED_SNAPSHOTS.get(key)
        if hit is not None and hit[0] == identity:
            return replace(hit[1], tiers=dict(hit[1].tiers))
        manifest = _json(path / "SNAPSHOT.json")
        verified = _verify_snapshot_bytes(path, binding, expected_release=expected_release,
                                          expected_digest=expected_digest,
                                          expected_authority_commit=expected_authority_commit, _manifest=manifest)
        if identity != _snapshot_identity(path):
            raise SnapshotError("snapshot changed during verification")
        _VERIFIED_CONTENT.add((verified.path, verified.digest, binding))
        _VERIFIED_EPOCHS.add(_epoch_digest(manifest))
        _VERIFIED_SNAPSHOTS[key] = (identity, verified)
        return replace(verified, tiers=dict(verified.tiers))


def _verify_snapshot_bytes(path: Path, binding: AuthorityBinding | GitAuthorityBinding, *,
                           expected_release: str | None, expected_digest: str | None,
                           expected_authority_commit: str | None,
                           _manifest: Mapping[str, Any] | None = None) -> VerifiedSnapshot:
    """Perform complete byte checks under the cold lock before semantic admission."""
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise SnapshotError("snapshot root symlink refused")
    if (path / "SNAPSHOT.json").is_symlink():
        raise SnapshotError("snapshot manifest symlink refused")
    for item in path.rglob("*"):
        if item.is_symlink():
            raise SnapshotError(f"snapshot symlink refused: {item.relative_to(path)}")
    path = path.resolve()
    manifest = dict(_json(path / "SNAPSHOT.json") if _manifest is None else _manifest)
    digest = manifest.pop("digest", None)
    if manifest.get("schema") not in (SCHEMA_SNAPSHOT, SCHEMA_SNAPSHOT_V2) or digest != _sha(_canonical(manifest)):
        raise SnapshotError("snapshot manifest digest/schema mismatch")
    if path.name != digest:
        raise SnapshotError("snapshot directory does not match its content address")
    if expected_digest is not None and digest != expected_digest:
        raise SnapshotError("snapshot differs from recorded run digest")
    if expected_release is not None and manifest.get("release") != expected_release:
        raise SnapshotError("snapshot differs from recorded run release")
    if manifest.get("authority_uuid") != binding.authority_uuid:
        raise SnapshotError("snapshot authority UUID differs from package binding")
    authority_commit = manifest.get("authority_commit")
    if ((authority_commit is not None and (not isinstance(authority_commit, str)
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", authority_commit) is None))
            or isinstance(binding, GitAuthorityBinding) and authority_commit is None):
        raise SnapshotError("snapshot lacks a valid Git authority commit")
    if expected_authority_commit is not None and authority_commit != expected_authority_commit:
        raise SnapshotError("snapshot differs from recorded Git authority commit")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise SnapshotError("snapshot manifest files must be a list")
    listed: set[str] = set()
    for record in files:
        if not isinstance(record, dict):
            raise SnapshotError("snapshot manifest file record is malformed")
        relative = _relative(record.get("path"))
        if relative in listed or relative == "SNAPSHOT.json":
            raise SnapshotError(f"snapshot duplicate/self-referenced file {relative}")
        listed.add(relative)
        target = path / relative
        if target.is_symlink() or any(parent.is_symlink() for parent in target.parents if parent != path.parent):
            raise SnapshotError(f"snapshot symlink refused: {relative}")
        try:
            raw = target.read_bytes()
        except OSError as exc:
            raise SnapshotError(f"snapshot missing file {relative}") from exc
        if len(raw) != record.get("size") or _sha(raw) != record.get("sha256"):
            raise SnapshotError(f"snapshot altered file {relative}")
    actual = {str(item.relative_to(path).as_posix()) for item in path.rglob("*") if item.is_file()}
    if actual != listed | {"SNAPSHOT.json"}:
        raise SnapshotError("snapshot contains missing or unmanifested files")
    # A copied or atomically renamed snapshot has the same complete manifest
    # digest. Every file above was still byte-hashed here; semantic closure is
    # a property of those exact bytes, not their former directory name. Only a
    # successful, stable admission in this process can supply this proof.
    admitted = any(held_digest == digest and held_binding == binding
                   for _held_path, held_digest, held_binding in _VERIFIED_CONTENT)
    if _BYTE_VERIFICATION_REQUIRED or not admitted:
        _verify_snapshot_content(path, listed, manifest)
    revision = manifest["authority_revision"]
    return VerifiedSnapshot(path, str(digest), str(manifest["release"]), str(manifest["root"]),
                            AuthorityRevision(int(revision["event_count"]), revision.get("latest_event_id")),
                            manifest.get("bundle_digest"), dict(manifest.get("tiers") or {}), authority_commit)


def _reset_verified_content() -> None:
    """Forget process-local byte verification and semantic admissions."""
    with _VERIFIED_CONTENT_LOCK:
        _VERIFIED_CONTENT.clear()
        _VERIFIED_EPOCHS.clear()
        _VERIFIED_SNAPSHOTS.clear()


def _reset_verified_content_after_fork() -> None:
    """Drop inherited admissions and a lock another parent thread may hold."""
    global _VERIFIED_CONTENT_LOCK
    _VERIFIED_CONTENT_LOCK = threading.Lock()
    _reset_verified_content()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_verified_content_after_fork)


def _epoch_digest(manifest: Mapping[str, Any]) -> str:
    """Identify all actual version bytes independently of the selected pin."""
    body = {key: value for key, value in manifest.items()
            if key not in {"digest", "release", "bundle_digest", "authority_commit"}}
    body["files"] = sorted(body["files"], key=lambda record: record["path"])
    return _sha(_canonical(body))


def _verify_snapshot_epoch(path: Path, listed: set[str], manifest: Mapping[str, Any]) -> None:
    """Verify every native row, its projected files and the complete lineage."""
    from .canon import content_key
    from .store import proposal_id, release_document_id

    for name in listed:
        if name.startswith("objects/"):
            row = _json(path / name).get("row", {})
            try:
                meta = row["manifest"]["meta"]
                key = content_key(row["name"], dict(meta))
                valid = (meta["content_key"] == key and row["version"] == key[:16]
                         and name == f"objects/{row['name']}/{row['version']}.json")
            except (KeyError, TypeError, ValueError) as exc:
                raise SnapshotError(f"snapshot invalid prompt object {name}") from exc
            if not valid:
                raise SnapshotError(f"snapshot prompt content key mismatch {name}")
        elif name.startswith("releases/"):
            release = _json(path / name)
            _release_rules(name, release, manifest["schema"])
            try:
                identity = release_document_id(release)
            except (KeyError, TypeError, ValueError) as exc:
                raise SnapshotError(f"snapshot release identity mismatch {name}") from exc
            if identity != release.get("id"):
                raise SnapshotError(f"snapshot release identity mismatch {name}")
        elif name.startswith("proposals/"):
            proposal = _json(path / name)
            if proposal_id(proposal) != proposal.get("id"):
                raise SnapshotError(f"snapshot proposal identity mismatch {name}")
    _verify_epoch(path, listed, manifest)


def _verify_snapshot_content(path: Path, listed: set[str], manifest: Mapping[str, Any]) -> None:
    """Verify common history and this pin's exact inert execution closure."""
    from .entity_model import CLOSURE_V2

    # Every listed file was byte-hashed before this call. A successful stable
    # snapshot can prove the same entire history for another pin in that history.
    # Selection and execution files are still checked separately for every pin.
    if _BYTE_VERIFICATION_REQUIRED or _epoch_digest(manifest) not in _VERIFIED_EPOCHS:
        _verify_snapshot_epoch(path, listed, manifest)
    selected = f"releases/{manifest['release']}.json"
    if selected not in listed:
        raise SnapshotError("snapshot lacks its selected release")
    release = _json(path / selected)
    root = _json(path / "V0.json").get("release")
    if root != manifest.get("root") or release.get("epoch") != manifest.get("epoch"):
        raise SnapshotError("snapshot release/root epoch mismatch")
    if release.get("rules") == CLOSURE_V2:
        bundle_digest = _verify_typed_tools(path, listed, release, manifest)
        payload = None
    else:
        payload = _payload(path, release)
        if payload is not None and payload.get("schema") == "rubiksql.answering-bundle/1":
            raise SnapshotError("schema1 Toolkit execution is permanently retired")
        bundle_digest = _sha(_canonical(payload)) if payload is not None else None
        if bundle_digest != manifest.get("bundle_digest"):
            raise SnapshotError("snapshot bundle digest mismatch")
    if payload is not None:
        from rubiksql_hb.tool_bundle import validate_snapshot_tools

        capsules = {item.stem: _json(item) for item in (path / "tools" / "capsules").glob("*.json")}
        toolkits = {item.stem: _json(item) for item in (path / "tools" / "toolkits").glob("*.json")}
        modules = {record["module"]: (path / "tools" / "modules" / (record["module"].replace(".", "/")
                   + ("/__init__.py" if record["module"] == "rubiksql_hb.tools" else ".py"))).read_text(encoding="utf-8")
                   for record in payload["modules"]}
        validate_snapshot_tools(payload, capsules, toolkits, modules)
        from rubiksql_hb.hb_tool_entities import snapshot_files

        expected_tools = snapshot_files(payload, capsules, toolkits)
        actual_tools = {name for name in listed if name.startswith("tools/")}
        if actual_tools != set(expected_tools):
            raise SnapshotError("snapshot tool files exceed the pinned execution closure")
        if any((path / name).read_bytes() != raw for name, raw in expected_tools.items()):
            raise SnapshotError("snapshot module source files differ from their pinned source Capsules")
    elif bundle_digest is None and any(name.startswith("tools/") for name in listed):
        raise SnapshotError("snapshot without an answering bundle contains execution files")


def _sealed_snapshot(path: Path, *, release: str, digest: str) -> VerifiedSnapshot:
    """Finish a verified publication, including a post-rename interrupted seal."""
    verify_snapshot(path, expected_release=release, expected_digest=digest)
    for item in path.rglob("*"):
        item.chmod(0o555 if item.is_dir() else 0o444)
    path.chmod(0o555)
    return verify_snapshot(path, expected_release=release, expected_digest=digest)


def _discard_snapshot_stage(path: Path) -> None:
    """Reopen disposable staging files before deletion, including Windows read-only attributes."""
    path.chmod(0o755)
    for item in path.rglob("*"):
        item.chmod(0o755 if item.is_dir() else 0o644)
    shutil.rmtree(path)


def write_snapshot(reader: AuthorityRead, pin: str, *, out_dir: Path,
                   tiers: Mapping[str, str] | None = None, staged: bool = False) -> VerifiedSnapshot:
    """Publish a read-only snapshot atomically from one open authority transaction.

    Args:
        reader: Open authority transaction whose ``store_files`` is the current nine-set projection.
        pin: ``baseline``, a release id or a published proposal id.
        out_dir: Directory holding ``.prompt_snapshot/<digest>``.
        tiers: Explicit operator-owned module execution policy. Git reads default to HB execution;
            historical database bindings retain their adjacent ``tool-tiers.json`` policy.
        staged: Compile a plan inside its active writer before export. All native
            integrity checks still run. This temporary planning snapshot is never
            an execution admission; confirmation makes a fresh committed snapshot.

    Returns:
        VerifiedSnapshot: The sealed ``rubiksql.prompt-snapshot/2`` snapshot. Its ``bundle_digest`` binds
            the ``tools.answering`` payload of a closure/1 pin or the typed bundle of a closure/2 pin.

    Raises:
        SnapshotError: The authority, tiers or written snapshot fail verification.
    """
    from rubiksql_hb.hb_tool_entities import digest, typed_bundle
    from .entity_model import CLOSURE_V2

    if staged and not reader.writer_lock_held:
        raise SnapshotError("a staged planning snapshot requires the active authority writer")
    verified = reader.verify(require_export=False) if staged else reader.verify()
    if verified is not None and not verified.ok:
        raise SnapshotError("authority failed verification: " + "; ".join(verified.violations))
    release = reader.resolve(pin)
    selected = reader.row("release-entity", release)
    if (selected["rules"] != CLOSURE_V2 and "tools.answering" in selected["refs"]
            and reader.answering(release).get("schema") == "rubiksql.answering-bundle/1"):
        raise SnapshotError("schema1 Toolkit execution is permanently retired")
    files = dict(reader.store_files(tools_release=release))
    head = reader.head()
    marker = json.loads(files["V0.json"])
    body = json.loads(files[f"releases/{release}.json"])
    if isinstance(getattr(reader, "git_binding", None), GitAuthorityBinding):
        pinned_tiers = dict(reader.tool_tiers())
        if tiers is not None and dict(tiers) != pinned_tiers:
            raise SnapshotError("snapshot tool tiers differ from the pinned Git authority policy")
        tiers = pinned_tiers
    elif tiers is None:
        tier_file = reader.binding.db_path.parent.parent / "tool-tiers.json"
        tiers = _json(tier_file) if tier_file.is_file() else {}
    if any(tier not in {"hb-executed", "host-verified"} for tier in tiers.values()):
        raise SnapshotError("tool tier configuration contains an unknown execution tier")
    if body.get("rules") == CLOSURE_V2:
        bundle_digest = digest(typed_bundle(body["content_refs"]))
    else:
        key = body["refs"].get("tools.answering")
        payload = None
        if key:
            row = json.loads(files[f"objects/tools.answering/{key[:16]}.json"])["row"]
            payload = json.loads(row["manifest"]["meta"]["text"])
        bundle_digest = _sha(_canonical(payload)) if payload else None
    files.pop("HEAD.json", None)
    files.pop("SNAPSHOT.json", None)
    manifest = {"schema": SCHEMA_SNAPSHOT_V2, "release": release, "root": marker["release"],
                "epoch": body["epoch"], "authority_uuid": reader.binding.authority_uuid,
                "authority_revision": {"event_count": head.revision.event_count, "latest_event_id": head.revision.latest_event_id},
                "bundle_digest": bundle_digest, "tiers": dict(tiers or {}),
                "files": [{"path": _relative(name), "sha256": _sha(raw), "size": len(raw)} for name, raw in sorted(files.items())]}
    if getattr(reader, "authority_commit", None) is not None:
        manifest["authority_commit"] = reader.authority_commit
    manifest["digest"] = _sha(_canonical(manifest))
    directory = Path(out_dir) / ".prompt_snapshot"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / manifest["digest"]
    if target.exists():
        return _sealed_snapshot(target, release=release, digest=manifest["digest"])
    staging_parent = Path(tempfile.mkdtemp(prefix=".writing-", dir=directory))
    temporary = staging_parent / manifest["digest"]
    temporary.mkdir()
    try:
        for name, raw in files.items():
            destination = temporary / _relative(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        (temporary / "SNAPSHOT.json").write_bytes(_canonical(manifest) + b"\n")
        verify_snapshot(temporary, expected_release=release, expected_digest=manifest["digest"])
        for item in temporary.rglob("*"):
            item.chmod(0o555 if item.is_dir() else 0o444)
        try:
            # Moving a directory between parents updates its '..' entry. Keep
            # the staging root writable until that move, then seal the final
            # root before exposing the returned pin. Its files are already sealed.
            temporary.rename(target)
        except OSError as exc:
            if exc.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not target.is_dir():
                raise
            _discard_snapshot_stage(temporary)
        staging_parent.rmdir()
        return _sealed_snapshot(target, release=release, digest=manifest["digest"])
    except BaseException:
        if temporary.exists():
            _discard_snapshot_stage(temporary)
        if staging_parent.exists():
            staging_parent.rmdir()
        raise


def snapshot_record(snapshot: VerifiedSnapshot) -> dict[str, Any]:
    """Manifest evidence kept outside scientific run identity."""
    return {"path": str(snapshot.path), "digest": snapshot.digest, "release": snapshot.release,
            "authority_revision": {"event_count": snapshot.revision.event_count, "latest_event_id": snapshot.revision.latest_event_id},
            "bundle_digest": snapshot.bundle_digest, "tiers": dict(snapshot.tiers),
            **({"authority_commit": snapshot.authority_commit} if snapshot.authority_commit is not None else {})}


def snapshot_env(snapshot: VerifiedSnapshot, *, bundle: bool = False) -> dict[str, str]:
    """Return only execution pin variables; no authority writer/config credential."""
    return {ENV_SNAPSHOT: str(snapshot.path), "RUBIKSQL_PROMPT_RELEASE": snapshot.release,
            **({ENV_BUNDLE: "1"} if bundle else {})}


def execution_snapshot(*, out_dir: Path, pin: str, binding: AuthorityBinding | GitAuthorityBinding,
                       inherited_env: Mapping[str, str], manifest: Mapping[str, Any] | None = None) -> VerifiedSnapshot:
    """Resolve only the outermost run; nested and resumed runs verify their recorded pin."""
    reject_execution_controls(inherited_env)
    record = (manifest or {}).get("prompt_snapshot") or {}
    source = inherited_env.get(ENV_SNAPSHOT) or record.get("path")
    if source:
        snapshot = verify_snapshot(Path(source), expected_digest=record.get("digest"), expected_release=record.get("release"),
                                   expected_authority_commit=record.get("authority_commit"))
        expected = pin
        if pin.startswith("p"):
            expected = str(_json(snapshot.path / "proposals" / f"{pin}.json")["release"])
        if pin != "baseline" and expected != snapshot.release:
            raise SnapshotError("nested/resumed launcher requested a different snapshot pin")
        return snapshot
    from .authority import Authority

    with Authority.read(binding=binding) as reader:
        return write_snapshot(reader, pin, out_dir=out_dir)


def head_snapshot(binding: AuthorityBinding | GitAuthorityBinding, *, into: Path) -> VerifiedSnapshot:
    """Bootstrap current-head interactive execution before importing prompt consumers."""
    return execution_snapshot(out_dir=into, pin="baseline", binding=binding, inherited_env={})


def _base_bundle_digest(reader: AuthorityRead, gate_event: Mapping[str, Any]) -> str | None:
    """Digest of the executable bundle of a receipt's publication base, or None when the base has none.

    A fresh epoch records its seed admission gate against the admitted release, not a proposal; that
    admission carries its own parent bundle, so children inheriting the seed tools verify against it.
    """
    from rubiksql_hb.hb_tool_entities import digest
    from .entity_model import CLOSURE_V2

    admission = reader.source_admission(str(gate_event["subject"]))
    if admission is not None and admission["gate"] == gate_event["payload"]:
        return admission["parent_bundle_digest"]
    publication = reader.proposal(str(gate_event["subject"]))
    bases = [item["release"] for item in publication["parents"] if item.get("role", "base") == "base"]
    if len(bases) != 1:
        raise SnapshotError("tool publication has no unique authoritative base parent")
    base = reader.row("release-entity", str(bases[0]))
    if base["rules"] == CLOSURE_V2:
        return digest(reader.typed_bundle(str(bases[0])))
    return _sha(_canonical(reader.answering(str(bases[0])))) if "tools.answering" in base["refs"] else None


def _entity_gate_payload(reader: AuthorityRead, gate_event: Mapping[str, Any]) -> Mapping[str, Any]:
    """Re-verify an entity-model gate against the authority and return its base's legacy payload.

    The representation-only migration has no tool-gate receipt of its own: its typed bundle is
    admitted because the migrated release's typed tools convert exactly from the base's pinned
    ``tools.answering`` bundle, whose own tool gate the caller validates next.
    """
    from rubiksql_hb.hb_tool_entities import canonical, convert_answering, digest
    from .entity_model import CLOSURE_V2, MIGRATION_KIND

    gate = gate_event["payload"]
    target, base = str(gate.get("release")), str(gate.get("base"))
    publication = reader.proposal(str(gate_event["subject"]))
    bases = [item["release"] for item in publication["parents"] if item.get("role", "base") == "base"]
    if publication.get("origin") != MIGRATION_KIND or publication.get("release") != target or bases != [base]:
        raise SnapshotError("entity-model gate differs from its migration publication")
    row = reader.row("release-entity", target)
    if row["rules"] != CLOSURE_V2 or digest(reader.typed_bundle(target)) != gate.get("bundle_digest"):
        raise SnapshotError("entity-model gate does not bind its release's typed bundle")
    base_row = reader.row("release-entity", base)
    if base_row["rules"] == CLOSURE_V2 or "tools.answering" not in base_row["refs"]:
        raise SnapshotError("entity-model gate base does not pin a tools.answering bundle")
    legacy = reader.answering(base)
    if _sha(_canonical(legacy)) != gate.get("legacy_bundle_digest"):
        raise SnapshotError("entity-model gate legacy bundle digest differs from its base's tools.answering")
    try:
        converted = convert_answering(legacy, {item["object_id"]: item for item in reader.rows("sys-capsule")})
    except ValueError as exc:
        raise SnapshotError(f"entity-model base bundle does not convert to typed tools: {exc}") from exc
    content = row["content_refs"]
    if canonical(converted.content) != canonical({"tools": content["tools"], "loader": content["loader"]}):
        raise SnapshotError("entity-model typed tools are not the exact conversion of the base's pinned bundle")
    return legacy


def validate_launch_snapshot(snapshot: VerifiedSnapshot, *, receipt: Mapping[str, Any] | None = None,
                             reader: Any = None, source_commit: str = "HEAD",
                             binding: AuthorityBinding | GitAuthorityBinding | None = None) -> None:
    """Compare a launcher pin with immutable published authority content and receipts.

    Answering, scoring and stage children only verify snapshot bytes. Launchers use this
    read-only check without resolving their pin again; no source account is allocated. A closure/2
    pin also compares its typed references and rows with the authority, and binds its typed bundle
    to its current-state source admission, a tool-gate receipt, or an entity-model gate whose base
    bundle carries a valid tool-gate receipt.

    Args:
        snapshot: Verified launcher pin.
        receipt: Optional batch admission checked in this same verified view,
            avoiding a second authority read before activation.
        reader: Existing verified authority view, for atomic batch admission.
            An active writer may validate its unexported planning snapshot.
        source_commit: Exact measured launcher source, or the calling checkout's
            HEAD. Deployment ancestry must hold for this commit.
        binding: Explicit authority for a caller-owned writer, or the selected
            package binding. The view must carry this exact identity.

    Raises:
        SnapshotError: Content, bundle or receipts differ from the authority, or the checkout does not
            descend from a deployed tool authority commit.
    """
    from .authority import Authority, semantic_row, scan_secrets, secret_inputs
    from .entity_model import CLOSURE_V2, MIGRATION_KIND, PARAMETER_ENTITY, RULE_ENTITY, SCHEMA_ENTITY_GATE, TOOL_ENTITY
    from .store import Store
    from rubiksql_hb.hb_tool_entities import canonical, typed_bundle, validate_tool_receipts

    if snapshot.bundle_digest is None:
        raise SnapshotError("evaluation requires tools.answering; the seed root cannot be evaluated")
    binding = binding if binding is not None else load_binding()
    if source_commit != "HEAD" and re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise SnapshotError("launch source requires a full Git commit")
    if isinstance(binding, GitAuthorityBinding) and snapshot.authority_commit is None:
        raise SnapshotError("launch snapshot lacks its exact Git authority commit")
    options = {"commit": snapshot.authority_commit} if isinstance(binding, GitAuthorityBinding) else {}
    with (nullcontext(reader) if reader is not None else Authority.read(binding=binding, **options)) as reader:
        if getattr(reader, "authority_commit", None) != snapshot.authority_commit:
            raise SnapshotError("launch snapshot differs from the exact authority view")
        if isinstance(binding, GitAuthorityBinding) and getattr(reader, "git_binding", None) != binding:
            raise SnapshotError("launch authority view has a different Git binding")
        verified = reader.verify(require_export=False) if getattr(reader, "writer_lock_held", False) else reader.verify()
        if not verified.ok:
            raise SnapshotError("authority launch verification failed: " + "; ".join(verified.violations))
        if isinstance(binding, GitAuthorityBinding) and dict(snapshot.tiers) != dict(reader.tool_tiers()):
            raise SnapshotError("launch snapshot tool tiers differ from the pinned Git authority policy")
        release = _json(snapshot.path / "releases" / f"{snapshot.release}.json")
        held = reader.row("release-entity", snapshot.release)
        typed = release["rules"] == CLOSURE_V2
        if (release["refs"] != held["refs"] or release["entries"] != held["entries"]
                or release["rules"] != held["rules"] or (typed and release["content_refs"] != held.get("content_refs"))):
            raise SnapshotError("snapshot release is not the authority's immutable release")
        prompts: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for row in reader.rows("sys-prompt"):
            prompts.setdefault((row["name"], row["version"]), []).append(row)
        for name, key in release["refs"].items():
            obj = _json(snapshot.path / "objects" / name / f"{key[:16]}.json")["row"]
            candidates = prompts.get((name, key[:16]), [])
            if len(candidates) != 1 or semantic_row(obj) != semantic_row(candidates[0]):
                raise SnapshotError(f"snapshot prompt {name} differs from authority")
        candidate_files = {str(path.relative_to(snapshot.path)): path.read_bytes()
                           for path in (snapshot.path / "tools").rglob("*") if path.is_file()}
        gates = [event for event in reader.events("gate")
                 if event["payload"].get("bundle_digest") == snapshot.bundle_digest]
        admission = reader.source_admission(snapshot.release)
        if not typed:
            payload = _payload(snapshot.path, release)
            if payload is not None and payload.get("schema") == "rubiksql.answering-bundle/1":
                raise SnapshotError("schema1 Toolkit execution is permanently retired")
            if payload != reader.answering(snapshot.release):
                raise SnapshotError("snapshot bundle differs from authority")
            if admission is not None:
                gate = admission["gate"]
                validate_tool_receipts(payload, gate, parent_bundle=admission["parent_bundle_digest"], launch=True,
                                       tiers=snapshot.tiers)
            elif not gates:
                raise SnapshotError("pinned bundle has no publication receipt")
            else:
                gate = gates[-1]["payload"]
                validate_tool_receipts(payload, gate, parent_bundle=_base_bundle_digest(reader, gates[-1]), launch=True,
                                       tiers=snapshot.tiers)
            candidate_files["tools.answering"] = _canonical(payload)
        else:
            content = release["content_refs"]
            mine, theirs = Store(package=snapshot.path).typed_index(), reader.typed_index()
            references = [*((TOOL_ENTITY, ref) for ref in content["tools"]),
                          *((RULE_ENTITY, ref) for ref in content["rules"]["items"]), (PARAMETER_ENTITY, content["parameters"])]
            for entity, ref in references:
                row, authority_row = mine[entity].get(ref["object_id"]), theirs.get(entity, {}).get(ref["object_id"])
                if row is None or authority_row is None or semantic_row(row) != semantic_row(authority_row):
                    raise SnapshotError(f"snapshot {entity} {ref['object_id']} differs from authority")
                candidate_files[f"typed/{entity}/{ref['object_id']}.json"] = _canonical(row)
            bundle = typed_bundle(content)
            authority_bundle = reader.typed_bundle(snapshot.release)
            if authority_bundle is None or canonical(authority_bundle) != canonical(bundle):
                raise SnapshotError("snapshot bundle differs from authority")
            tool_gates = [event for event in gates if event["payload"].get("schema") == _TOOL_GATE]
            entity_gates = [event for event in gates if event["payload"].get("schema") == SCHEMA_ENTITY_GATE]
            if admission is not None:
                gate = admission["gate"]
                validate_tool_receipts(bundle, gate, parent_bundle=admission["parent_bundle_digest"], launch=True,
                                       tiers=snapshot.tiers)
            elif tool_gates:
                gate = tool_gates[-1]["payload"]
                validate_tool_receipts(bundle, gate, parent_bundle=_base_bundle_digest(reader, tool_gates[-1]), launch=True,
                                       tiers=snapshot.tiers)
            elif entity_gates:
                gate = entity_gates[-1]["payload"]
                legacy = _entity_gate_payload(reader, entity_gates[-1])
                legacy_gates = [event for event in reader.events("gate")
                                if event["payload"].get("bundle_digest") == gate["legacy_bundle_digest"]]
                if not legacy_gates:
                    raise SnapshotError("entity-model base bundle has no publication receipt")
                validate_tool_receipts(legacy, legacy_gates[-1]["payload"], launch=True, tiers=snapshot.tiers,
                                       parent_bundle=_base_bundle_digest(reader, legacy_gates[-1]))
                candidate_files["legacy-publication-gate.json"] = _canonical(legacy_gates[-1]["payload"])
            else:
                raise SnapshotError("pinned bundle has no publication receipt")
            candidate_files["typed-bundle.json"] = _canonical(bundle)
        candidate_files["publication-gate.json"] = _canonical(gate)
        if admission is not None:
            candidate_files["source-admission.json"] = _canonical(admission)
        scan_secrets(candidate_files, secret_inputs())
        commits = []
        if admission is not None:
            commits.append(gate.get("source_commit") or gate["audit_implementation_commit"])
        for event in reader.events("decision"):
            detail = event["payload"]
            if detail.get("kind") in {"cutover", MIGRATION_KIND}:
                commits.append(detail["deployed_commit"])
        for commit in dict.fromkeys(commits):
            import subprocess
            source = Path(__file__).resolve().parents[3]
            result = subprocess.run(["git", "merge-base", "--is-ancestor", str(commit), source_commit],
                                    cwd=source, check=False, capture_output=True, timeout=30)
            if result.returncode:
                raise SnapshotError("loaded checkout does not descend from deployed tool authority commit")
        if receipt is not None:
            from .paid_launch import _CONFIRMED_ADMISSIONS, _read_committed_claim
            from .canon import canon

            if (receipt.get("claim_commit") != snapshot.authority_commit
                    or receipt.get("snapshot_digest") != snapshot.digest or receipt.get("release") != snapshot.release):
                raise SnapshotError("batch receipt differs from its pushed execution snapshot")
            reservation = _read_committed_claim(binding, receipt, reader=reader)
            _CONFIRMED_ADMISSIONS[(binding, canon(dict(receipt)))] = reservation
