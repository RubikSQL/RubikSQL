"""Read-only mapped authority/snapshot files and canonical content identities.

The frozen package store is test/history data. Default Store construction requires a
verified execution snapshot; explicit package paths are read-only authority adapters.
No runtime overlay or mirror writer exists.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .canon import canon, sha256_hex
from .policy import PolicyError, check_name, check_proposal_id, check_release_id, check_version
from .types import AuthorityError

__all__ = [
    "ENV_STORE",
    "DECISION_KINDS",
    "LEGACY_TABLES",
    "PACKAGE_ONLY",
    "PACKAGE_STORE_DIR",
    "POINTER_KINDS",
    "REPO_ENV_FILE",
    "RULE_CORPUS_FILE",
    "SCHEMA_BASELINE",
    "SCHEMA_CELL",
    "SCHEMA_DECISION",
    "SCHEMA_OBJECT",
    "SCHEMA_PROPOSAL",
    "SCHEMA_RELEASE",
    "SCHEMA_RETIRED",
    "SCHEMA_REVOKED",
    "SCHEMA_V0",
    "TABLES",
    "TYPED_TABLES",
    "BaselinePointer",
    "V0Scope",
    "Layer",
    "Loaded",
    "PromptStoreError",
    "Store",
    "canonical_bytes",
    "check_env_file",
    "check_release_format",
    "env_file_bindings",
    "env_file_keys",
    "git_blob_sha",
    "proposal_id",
    "project_version_files",
    "publication_ancestry",
    "release_document_id",
    "release_id",
    "rule_corpus_path",
    "write_if_absent",
    "write_pointer",
]

ENV_STORE = "RUBIKSQL_PROMPT_STORE"
#: The git-tracked package store, always the first layer.
PACKAGE_STORE_DIR = Path(__file__).resolve().parent / "store"
#: The tracked ``.env`` nl2sql loads at import; prompt variables must never appear in it.
REPO_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"
#: Paths that only the package layer may provide.
PACKAGE_ONLY = ("BASELINE.json", "V0.json", "revoked.json", "retired", "decisions", "epochs")
#: The committed general rule corpus, relative to the checkout root: the generated artifact of BASELINE's
#: ``rules.general`` row, rewritten by every promotion and rollback, so tools that read the file read BASELINE's corpus.
RULE_CORPUS_FILE = ("resources", "generated_rules", "general_rules.json")

SCHEMA_OBJECT = "rubiksql.object/1"
SCHEMA_RELEASE = "rubiksql.release/1"
SCHEMA_PROPOSAL = "rubiksql.proposal/1"
SCHEMA_BASELINE = "rubiksql.baseline/1"
SCHEMA_REVOKED = "rubiksql.revoked/1"
SCHEMA_RETIRED = "rubiksql.retired/1"
SCHEMA_DECISION = "rubiksql.decision/1"
SCHEMA_CELL = "rubiksql.cell/1"
SCHEMA_V0 = "rubiksql.v0/1"
#: Decisions that move the current release. ``entity-model`` is the representation-only typed migration.
POINTER_KINDS = frozenset({"cutover", "promote", "rollback", "entity-model"})
#: Decision kinds accepted by inert native projections and recorded decision files.
DECISION_KINDS = ("cutover", "entity-model", "promote", "rollback", "reject", "abandon", "retire", "unretire",
                  "revoke", "unrevoke")
#: The six native version sets of every historical format (snapshot /1, export /1 and /2).
LEGACY_TABLES = {"sys-prompt": "sys_prompt", "sys-capsule": "sys_capsule", "sys-toolkit": "sys_toolkit",
                 "release-entity": "release_entity", "proposal-entity": "proposal_entity",
                 "prompt-event-entity": "prompt_event_entity"}
#: Canonical typed version sets; only rows with a canonical ``format`` belong to them.
TYPED_TABLES = {"tool-entity": "tool_entity", "general-rule-entity": "general_rule_entity",
                "knob-set-entity": "knob_set_entity"}
#: The current closed version scope (snapshot /2, export /3).
TABLES = {**LEGACY_TABLES, **TYPED_TABLES}

_COMMIT = re.compile(r"[0-9a-f]{40}\Z")

_GUARDED_ENV_PREFIX = "RUBIKSQL_PROMPT_"
_GUARDED_ENV_KEYS = ("RUBIKSQL_CACHE_NS",)

# python-dotenv 1.2's binding grammar (dotenv/parser.py), so the guard sees exactly the keys nl2sql's
# load_dotenv would bind: leading whitespace may span lines; `export` needs at least one blank after it;
# a key is single-quoted or a run free of `=`, `#` and whitespace; a value is single- or double-quoted
# (both may span lines, escapes decoded) or unquoted to the end of the line minus a trailing comment.
_ENV_WHITESPACE = re.compile(r"\s*")
_ENV_EXPORT = re.compile(r"(?:export[^\S\r\n]+)?")
_ENV_QUOTED_KEY = re.compile(r"'([^']+)'")
_ENV_KEY = re.compile(r"([^=\#\s]+)")
_ENV_BLANKS = re.compile(r"[^\S\r\n]*")
_ENV_EQUAL = re.compile(r"=[^\S\r\n]*")
_ENV_SINGLE_QUOTED = re.compile(r"'((?:\\'|[^'])*)'")
_ENV_DOUBLE_QUOTED = re.compile(r'"((?:\\"|[^"])*)"')
_ENV_UNQUOTED = re.compile(r"[^\r\n]*")
_ENV_COMMENT_AND_EOL = re.compile(r"(?:[^\S\r\n]*#[^\r\n]*)?[^\S\r\n]*(?:\r\n|\n|\r|$)")
_ENV_REST_OF_LINE = re.compile(r"[^\r\n]*(?:\r|\n|\r\n)?")
_ENV_SINGLE_ESCAPES = re.compile(r"\\[\\']")
_ENV_DOUBLE_ESCAPES = re.compile(r"\\[\\'\"abfnrtv]")


class PromptStoreError(RuntimeError):
    """The store, a release, an object or the environment failed a check; the message names the path and the rule."""


@dataclass(frozen=True)
class Layer:
    """One store directory and how it is reported (``package`` or ``overlay:<dir>``)."""

    path: Path
    origin: str


@dataclass(frozen=True)
class Loaded:
    """A JSON file read through the layers, with the origin that supplied it."""

    data: Any
    origin: str
    relative: str
    raw: bytes


@dataclass(frozen=True)
class BaselinePointer:
    """The release ``BASELINE.json`` points at, plus the git blob sha of the pointer file."""

    release: str
    blob_sha: str


@dataclass(frozen=True)
class V0Scope:
    """The new experiment history rooted at one parentless v0 release."""

    root: str
    releases: frozenset[str]
    proposals: frozenset[str]


def canonical_bytes(obj: Any) -> bytes:
    """Serialize a store file: sorted keys, indent 1, UTF-8, LF, trailing newline.

    Args:
        obj (Any): JSON-compatible value.

    Returns:
        bytes: The exact bytes the store holds for this value.
    """
    return (json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def git_blob_sha(data: bytes) -> str:
    """Compute git's blob sha1 of file contents, so a pointer can be cited as ``baseline@<sha>``.

    Args:
        data (bytes): File contents.

    Returns:
        str: 40 hex characters, equal to ``git hash-object`` of the same bytes.
    """
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def release_id(refs: Mapping[str, str], entries: Iterable[str], rules: str, *, epoch: str | None = None,
               content_refs: Mapping[str, Any] | None = None) -> str:
    """Derive a release id from its content.

    Args:
        refs (Mapping[str, str]): Row name to full 64-hex content key of every prompt row in the closure.
        entries (Iterable[str]): Names of the rows that are entries.
        rules (str): Closure rules version.
        epoch (str | None): ``"v0:<delivery_commit>"`` for a fresh v0 lineage, ``"v0"`` for the
            original v0 seed, or ``None`` for pre-v0 releases. The epoch is part of the content identity.
        content_refs (Mapping[str, Any] | None): The typed block of a ``rubiksql.closure/2`` release;
            ignored for closure/1, whose identity is unchanged.

    Returns:
        str: ``"r"`` + the first 16 hex digits of ``sha256(canon({"refs", "entries", "rules"}))``; a
            closure/2 body also hashes ``schema`` ``rubiksql.release/2`` and ``content_refs``.

    Raises:
        ValueError: An unsupported epoch, or a closure/2 release without a typed block.
    """
    from .entity_model import CLOSURE_V2, SCHEMA_RELEASE_V2

    if epoch not in (None, "v0") and (not isinstance(epoch, str) or not re.fullmatch(r"v0:[0-9a-f]{40}", epoch)):
        raise ValueError(f"unsupported release epoch {epoch!r}")
    body: dict[str, Any] = {"refs": dict(refs), "entries": sorted(entries), "rules": rules}
    if epoch is not None:
        body["epoch"] = epoch
    if rules == CLOSURE_V2:
        if not isinstance(content_refs, Mapping):
            raise ValueError("a rubiksql.closure/2 release id needs its content_refs")
        body.update(schema=SCHEMA_RELEASE_V2, content_refs=dict(content_refs))
    return "r" + sha256_hex(canon(body))[:16]


def release_document_id(document: Mapping[str, Any], *, epoch: str | None = None) -> str:
    """Recompute the id of a release file body or native row of either closure format.

    Args:
        document (Mapping[str, Any]): ``refs``, ``entries``, ``rules`` and, for closure/2, ``content_refs``.
        epoch (str | None): The epoch for a native row; a file body's own ``epoch`` is used when omitted.

    Returns:
        str: The content-derived release id.
    """
    return release_id(document["refs"], document["entries"], document["rules"],
                      epoch=epoch if epoch is not None else document.get("epoch"),
                      content_refs=document.get("content_refs"))


def check_release_format(document: Mapping[str, Any]) -> str:
    """Check that a release file pairs its schema with its closure rules.

    Readers accept exactly ``(rubiksql.release/1, rubiksql.closure/1)`` and
    ``(rubiksql.release/2, rubiksql.closure/2)``; any other schema, closure rules or pairing fails.

    Args:
        document (Mapping[str, Any]): Release file body.

    Returns:
        str: The closure rules version.

    Raises:
        PromptStoreError: Unknown schema or closure rules, or a mismatched pair.
    """
    from .entity_model import CLOSURE_V2, SCHEMA_RELEASE_V2
    from .entries import CLOSURE_RULES

    pairs = {SCHEMA_RELEASE: CLOSURE_RULES, SCHEMA_RELEASE_V2: CLOSURE_V2}
    schema, rules = document.get("schema"), document.get("rules")
    if schema not in pairs or pairs[schema] != rules:
        raise PromptStoreError(f"release {document.get('id')!r} pairs schema {schema!r} with closure rules {rules!r}; "
                               f"known pairs are {sorted(pairs.items())}")
    return str(rules)


def proposal_id(body: Mapping[str, Any]) -> str:
    """Derive a proposal id from its body (every field except ``id``).

    Args:
        body (Mapping[str, Any]): Proposal file contents; an ``id`` key is ignored.

    Returns:
        str: ``"p"`` + the first 16 hex digits of the sha256 of the canonical body.
    """
    return "p" + sha256_hex(canon({key: value for key, value in body.items() if key != "id"}))[:16]


def publication_ancestry(releases: Mapping[str, Mapping[str, Any]],
                         proposals: Mapping[str, Mapping[str, Any]]) -> dict[str, set[str]]:
    """Verify publication dependencies and return immutable first-publication ancestry.

    Args:
        releases: Release documents keyed by content identity. Their ``proposal`` and ``parents``
            describe the first publication only; epoch and state roots have neither.
        proposals: Every publication, including later proposals reproducing existing content.

    Returns:
        Release-to-parent edges from first publications only. A later proposal depends on the
        first publication of each parent, without changing the ancestry of its output release.

    Raises:
        PromptStoreError: A publication link is missing, inconsistent, self-referential or cyclic.
    """
    ancestry = {}
    for rid, release in releases.items():
        pid = release.get("proposal")
        parents = release.get("parents") or []
        if pid is not None and (pid not in proposals or proposals[pid].get("release") != rid
                                or proposals[pid].get("parents") != parents):
            raise PromptStoreError(f"release {rid} differs from its first publication")
        if pid is None and parents:
            raise PromptStoreError(f"release {rid} has parents without a first publication")
        ancestry[rid] = {item["release"] for item in parents}
    dependencies = {}
    for pid, proposal in proposals.items():
        rid, parents = proposal["release"], proposal["parents"]
        if (rid not in releases or not isinstance(parents, list)
                or sum(item.get("role", "base") == "base" for item in parents) != 1
                or any(item.get("release") not in releases or item["release"] == rid for item in parents)):
            raise PromptStoreError(f"proposal {pid} parent closure is invalid")
        dependencies[pid] = {releases[item["release"]]["proposal"] for item in parents
                             if releases[item["release"]].get("proposal") is not None}
    visiting, completed = set(), set()

    def visit(pid: str) -> None:
        if pid in visiting:
            raise PromptStoreError("proposal history contains a parent cycle")
        if pid in completed:
            return
        visiting.add(pid)
        for parent in dependencies[pid]:
            visit(parent)
        visiting.remove(pid)
        completed.add(pid)

    for pid in proposals:
        visit(pid)
    return ancestry


def _bytes(value: Any) -> bytes:
    return (_canon_json(value) + "\n").encode("utf-8")


def _canon_json(value: Any) -> str:
    """Return ``canon(_json_value(value))`` without a Python walk of plain decoded JSON.

    ``json.dumps`` serializes dicts, lists, tuples and scalars exactly as their
    ``_json_value`` detachment would, and refuses every value that detachment
    converts (sets, paths, callables and other Mappings). A refusal therefore
    falls back to the detaching walk, whose result is identical.
    """
    try:
        return canon(value)
    except (TypeError, ValueError):
        return canon(_json_value(value))


def _json_value(value: Any) -> Any:
    """Detach native decoded JSON without discarding timestamps or metadata."""
    if callable(value) and hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return str(value)
    return value


def project_version_files(rows: Mapping[str, Sequence[Mapping[str, Any]]], *,
                          tools_release: str | None = None) -> Mapping[str, bytes]:
    """Derive the exact closed version projection from inert native rows/events.

    The historical scope (:data:`LEGACY_TABLES`, snapshot /1) projects exactly as before and holds
    only ``rubiksql.closure/1`` releases. The current scope (:data:`TABLES`) adds the canonical
    typed version sets and projects a closure/2 release as a ``rubiksql.release/2`` file.

    Args:
        rows: Exactly the six historical or the nine current version entity sets, including explicit
            empty sets. Typed sets hold canonical rows only; the caller filters projection rows.
        tools_release: Exact selected release whose execution closure is included: the pinned
            ``tools.answering`` closure of a closure/1 release, or the module and serializer source
            closure of a closure/2 release (its Tool rows travel in ``native/tool-entity.json``).

    Returns:
        Canonical paths and bytes; no database, HB workspace or provider is opened.

    Raises:
        AuthorityError: Native sets, epoch, publication identity or closure differs, the historical
            scope holds a closure/2 release, or a typed set holds a non-canonical row.
    """
    from .entity_model import CLOSURE_V2, SCHEMA_RELEASE_V2, EntityModelError, is_canonical

    if set(rows) not in (set(LEGACY_TABLES), set(TABLES)):
        raise AuthorityError("version projection requires exactly the six historical or the nine current native entity sets")
    native = {entity: tuple(sorted(values, key=lambda row: str(row["object_id"]))) for entity, values in rows.items()}
    if any(len({row["object_id"] for row in values}) != len(values) for values in native.values()):
        raise AuthorityError("version projection repeats a native identity")
    if set(rows) == set(LEGACY_TABLES) and any(row.get("rules") == CLOSURE_V2 for row in native["release-entity"]):
        raise AuthorityError("the historical six-set version projection cannot hold a rubiksql.closure/2 release")
    for entity in TYPED_TABLES:
        for row in native.get(entity, ()):
            try:
                canonical = is_canonical(entity, row)
            except EntityModelError as error:
                raise AuthorityError(f"version projection {entity} row is invalid: {error}") from error
            if not canonical:
                raise AuthorityError(f"version projection {entity} holds the non-canonical row {row.get('object_id')!r}")
    events = sorted(native["prompt-event-entity"], key=lambda event: (float(event["ts"]), str(event["object_id"])))
    if any(event["kind"] == "decision" and event["payload"].get("kind") not in DECISION_KINDS for event in events):
        raise AuthorityError("version projection decision has an unsupported kind")
    for event in events:
        if event["kind"] != "publish":
            continue
        body = event["payload"].get("body") or (event["payload"].get("batch") or {}).get("proposal")
        if isinstance(body, Mapping) and body.get("origin") not in {"cutover", "entity-model", "agent", "human"}:
            raise AuthorityError("version projection publication has an unsupported origin")
    markers = [event for event in events if event["kind"] == "authority" and event["payload"].get("operation") == "mark"]
    if len(markers) != 1:
        raise AuthorityError("version projection requires the unique epoch marker")
    epoch = markers[0]["payload"]["epoch"]
    state = None
    if any(event["kind"] == "register" and event["payload"].get("schema") == "rubiksql.authority-state/1"
           for event in events):
        from .authority_state import validate_native_state

        state = validate_native_state(native)
    states = [state] if state is not None else []
    from .fresh_epoch import validate_native_admission
    fresh = validate_native_admission(native)
    if fresh is not None:
        states.append(fresh)
    moves = [event for event in events if event in states or
             (event["kind"] == "decision" and event["payload"].get("kind") in POINTER_KINDS)]
    baseline = (str(moves[-1]["payload"].get("to", moves[-1]["payload"].get("release")))
                if moves else str(epoch["release"]))
    publications: dict[str, Mapping[str, Any]] = {}
    by_proposal: dict[str, Mapping[str, Any]] = {}
    for event in events:
        body = event["payload"].get("body") if event["kind"] == "publish" else None
        if not isinstance(body, Mapping):
            continue
        pid = str(event["subject"])
        if pid in by_proposal or body.get("id") != pid or proposal_id(body) != pid:
            raise AuthorityError("version projection has no exact unique publication body")
        if body.get("origin") not in {"cutover", "entity-model", "agent", "human"}:
            raise AuthorityError("version projection publication has an unsupported origin")
        by_proposal[pid] = body
        publications.setdefault(str(body["release"]), body)
    files: dict[str, bytes] = {}
    releases = {str(row["object_id"]): row for row in native["release-entity"]}
    anchors = {}
    for event in states:
        state = event["payload"]
        rid = str(state["release"])
        row = releases.get(rid)
        if (row is None or state.get("epoch") != epoch or state.get("generation") != rid
                or state.get("content_digest") != sha256_hex(canon({key: row.get(key)
                    for key in ("refs", "entries", "rules", "content_refs")}))):
            raise AuthorityError("version projection current-state anchor differs from its exact release")
        if rid in anchors:
            raise AuthorityError("version projection repeats a current-state release anchor")
        anchors[rid] = event["object_id"]
    for rid, row in releases.items():
        typed = row["rules"] == CLOSURE_V2
        body = {"schema": SCHEMA_RELEASE_V2 if typed else SCHEMA_RELEASE, "id": rid, "refs": row["refs"],
                "entries": row["entries"], "rules": row["rules"], **({"content_refs": row["content_refs"]} if typed else {}),
                "reason": row.get("reason", ""), "origin": "v0" if rid == epoch["release"] else "proposal", "epoch": epoch["epoch"]}
        if rid == epoch["release"]:
            body.update({key: epoch[key] for key in ("source_release", "source_commit", "delivery_commit")})
            if rid in anchors:
                body["authority_state"] = anchors[rid]
        elif rid in anchors:
            body.update(origin="state", authority_state=anchors[rid])
        elif rid in publications:
            proposal = publications[rid]
            body.update(origin="cutover" if proposal["origin"] == "cutover" else "proposal",
                        parents=proposal["parents"], proposal=proposal["id"])
        files[f"releases/{rid}.json"] = _bytes(body)
    prompts = {}
    for row in native["sys-prompt"]:
        key = row["manifest"]["meta"]["content_key"]
        prompts[(row["name"], key)] = row
        files[f"objects/{row['name']}/{row['version']}.json"] = _bytes({"schema": SCHEMA_OBJECT, "row": row,
            "export": {"content_key": key, "name": row["name"], "version": row["version"], "control_object_id": row["object_id"]}})
    for row in native["proposal-entity"]:
        if row["object_id"] in by_proposal:
            files[f"proposals/{row['object_id']}.json"] = _bytes(by_proposal[row["object_id"]])
    files["BASELINE.json"] = _bytes({"schema": SCHEMA_BASELINE, "release": baseline})
    files["V0.json"] = _bytes({"schema": "rubiksql.v0/1", **{key: epoch[key] for key in ("label", "release", "source_release", "source_commit", "delivery_commit")}})
    revoked, retired = {}, {}
    for event in events:
        payload = event["payload"]
        if event["kind"] == "outcome":
            target = payload.get("proposal") or payload.get("target")
            if not isinstance(target, str):
                raise AuthorityError("version projection outcome has no exact proposal or release target")
            files[f"outcomes/{target}/{payload['id']}.json"] = _bytes(payload)
        elif event["kind"] == "eval-attach":
            files[f"attempts/{payload['generation']}/{payload['id']}.json"] = _bytes(payload)
        elif event["kind"] == "eval-reserve":
            files[f"reservations/{payload['id']}.json"] = _bytes(payload)
        elif event["kind"] == "gate":
            files[f"attestations/{event['object_id']}-gate.json"] = _bytes(payload)
        elif event["kind"] == "toolkit-retirement":
            rid = payload["archived"]["release"]
            retired[rid] = {"schema": SCHEMA_RETIRED, "release": rid, "decision": event["object_id"],
                            "reason": "schema1 Toolkit execution permanently retired", "actor": event["actor"],
                            "created_at": event["ts"]}
        elif event["kind"] == "decision":
            decision = {**payload, "schema": SCHEMA_DECISION, "id": event["object_id"], "actor": event["actor"],
                        "created_at": event["ts"], "previous": payload.get("from"), "release": payload.get("to", payload.get("release"))}
            files[f"decisions/{event['object_id']}.json"] = _bytes(decision)
            kind, target = payload.get("kind"), str(payload.get("target", event["subject"]))
            if kind == "retire":
                retired[target] = {"schema": SCHEMA_RETIRED, "release": target, "decision": event["object_id"], "reason": payload["reason"], "actor": event["actor"], "created_at": event["ts"]}
            elif kind == "unretire":
                retired.pop(target, None)
            elif kind == "revoke":
                revoked[str(payload["content_key"])] = {"ref": target, "decision": event["object_id"], "reason": payload["reason"], "actor": event["actor"]}
            elif kind == "unrevoke":
                revoked.pop(str(payload["content_key"]), None)
    files.update({f"retired/{rid}.json": _bytes(body) for rid, body in retired.items()})
    files["revoked.json"] = _bytes({"schema": SCHEMA_REVOKED, "revoked": revoked})
    files["events.jsonl"] = b"".join(_bytes(event) for event in events)
    files.update({f"native/{entity}.json": _bytes(values) for entity, values in native.items()})
    if tools_release is not None:
        release = releases[tools_release]
        if release["rules"] == CLOSURE_V2:
            from ..hb_tool_entities import typed_snapshot_files

            try:
                files.update(typed_snapshot_files(release["content_refs"], {row["object_id"]: row for row in native["sys-capsule"]}))
            except (KeyError, TypeError) as error:
                raise AuthorityError(f"release {tools_release} pins an absent or malformed source Capsule: {error}") from error
        elif "tools.answering" in release["refs"]:
            from ..hb_tool_entities import snapshot_files

            row = prompts[("tools.answering", release["refs"]["tools.answering"])]
            payload = json.loads(row["manifest"]["meta"]["text"])
            files.update(snapshot_files(payload, {row["object_id"]: row for row in native["sys-capsule"]},
                                        {row["object_id"]: row for row in native["sys-toolkit"]}))
    return files


def write_if_absent(path: Path, obj: Any) -> bool:
    """Write a canonical store file once; an identical existing file is a no-op, a different one an error.

    Args:
        path (Path): Destination; parent directories are created.
        obj (Any): JSON-compatible value.

    Returns:
        bool: ``True`` when the file was created, ``False`` when identical bytes already existed.

    Raises:
        PromptStoreError: If the path exists with different bytes.
    """
    data = canonical_bytes(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "xb") as handle:
            handle.write(data)
    except FileExistsError:
        if path.read_bytes() != data:
            raise PromptStoreError(f"{path}: exists with different bytes; store files are write-once") from None
        return False
    return True


def write_pointer(path: Path, obj: Any) -> None:
    """Write or replace a mutable pointer file (``BASELINE.json``) atomically.

    Args:
        path (Path): Destination.
        obj (Any): JSON-compatible value.

    Returns:
        None: This function does not return a value.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(canonical_bytes(obj))
    os.replace(temp, path)


def rule_corpus_path(package: Path) -> None:
    """Corpus mirrors are frozen history; runtime readers use pinned rules.general."""
    return None




class _EnvSyntax(Exception):
    """A dotenv line the grammar rejects; python-dotenv skips it with a warning, so does the parser here."""


def _env_value(text: str, pos: int) -> tuple[str, int]:
    if pos < len(text) and text[pos] == "'":
        match = _ENV_SINGLE_QUOTED.match(text, pos)
        if match is None:
            raise _EnvSyntax
        return _ENV_SINGLE_ESCAPES.sub(lambda m: codecs.decode(m.group(0), "unicode-escape"), match.group(1)), match.end()
    if pos < len(text) and text[pos] == '"':
        match = _ENV_DOUBLE_QUOTED.match(text, pos)
        if match is None:
            raise _EnvSyntax
        return _ENV_DOUBLE_ESCAPES.sub(lambda m: codecs.decode(m.group(0), "unicode-escape"), match.group(1)), match.end()
    if pos >= len(text) or text[pos] in "\r\n":
        return "", pos
    match = _ENV_UNQUOTED.match(text, pos)
    return re.sub(r"\s+#.*", "", match.group(0)).rstrip(), match.end()


def env_file_bindings(text: str) -> list[tuple[str, str | None]]:
    """Parse a dotenv text the way python-dotenv 1.2 does and list the bindings it would produce.

    ``export KEY=value`` (any run of blanks after ``export``), ``'KEY'=value``, ``KEY = value``, quoted
    values spanning lines and ``# comments`` are all read as python-dotenv reads them. A key without
    ``=`` is listed with value ``None``: ``dotenv_values`` reports it, ``load_dotenv`` does not export it,
    and the guard refuses it either way. A line the grammar rejects is skipped, as python-dotenv skips
    it with a warning.

    Args:
        text (str): File contents.

    Returns:
        list[tuple[str, str | None]]: ``(key, value)`` pairs in file order, repeated keys included.
    """
    bindings: list[tuple[str, str | None]] = []
    pos, end = 0, len(text)
    while pos < end:
        pos = _ENV_WHITESPACE.match(text, pos).end()
        if pos >= end:
            break
        try:
            pos = _ENV_EXPORT.match(text, pos).end()
            key: str | None = None
            if pos < end and text[pos] == "#":
                pass  # a comment line: consumed below with the rest of the line
            else:
                match = (_ENV_QUOTED_KEY if pos < end and text[pos] == "'" else _ENV_KEY).match(text, pos)
                if match is None:
                    raise _EnvSyntax
                key, pos = match.group(1), match.end()
            pos = _ENV_BLANKS.match(text, pos).end()
            value: str | None = None
            if pos < end and text[pos] == "=":
                value, pos = _env_value(text, _ENV_EQUAL.match(text, pos).end())
            match = _ENV_COMMENT_AND_EOL.match(text, pos)
            if match is None:
                raise _EnvSyntax
            pos = match.end()
        except _EnvSyntax:
            pos = _ENV_REST_OF_LINE.match(text, pos).end()
            continue
        if key is not None:
            bindings.append((key, value))
    return bindings


def env_file_keys(text: str) -> list[str]:
    """List the variable names a dotenv file binds, as python-dotenv would bind them.

    Args:
        text (str): File contents; see :func:`env_file_bindings` for the grammar.

    Returns:
        list[str]: Keys in file order.
    """
    return [key for key, _value in env_file_bindings(text)]


def check_env_file(path: Path | None = None) -> None:
    """Refuse a tracked ``.env`` that would pin prompt variables behind the launcher's back.

    nl2sql loads this file with ``override=False`` before anything else, so a
    ``RUBIKSQL_PROMPT_*`` or ``RUBIKSQL_CACHE_NS`` key there would silently steer every process.
    The file is read with python-dotenv's own grammar (:func:`env_file_bindings`), so ``export``,
    quoted keys and spaced assignments cannot slip past the guard.

    Args:
        path (Path | None): The dotenv file; ``REPO_ENV_FILE`` when omitted. A missing file passes.

    Returns:
        None: This function does not return a value.

    Raises:
        PromptStoreError: If the file binds any ``RUBIKSQL_PROMPT_*`` key or ``RUBIKSQL_CACHE_NS``.
    """
    path = Path(path) if path is not None else REPO_ENV_FILE
    if not path.exists():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return
    offending = [key for key in env_file_keys(text) if key.startswith(_GUARDED_ENV_PREFIX) or key in _GUARDED_ENV_KEYS]
    if offending:
        raise PromptStoreError(f"{path}: assigns {offending}; prompt variables must come from the process environment, never from the tracked .env")


class Store:
    """One explicit read-only authority/snapshot view.

    Attributes:
        layers (tuple[Layer, ...]): Package layer first, then overlays in search-path order.
    """

    def __init__(self, overlays: Sequence[str | os.PathLike[str]] = (), *, package: Path | None = None) -> None:
        """Assemble the layers and reject overlays that carry package-only files.

        Args:
            overlays (Sequence[str | os.PathLike[str]]): Extra store directories; order does not matter because
                conflicting bytes are an error rather than a precedence question.
            package (Path | None): The package layer; ``PACKAGE_STORE_DIR`` when omitted.

        Raises:
            PromptStoreError: If the package layer or an overlay is not a directory, or an overlay contains one of
                ``BASELINE.json``, ``revoked.json``, ``retired/`` or ``decisions/``.
        """
        if overlays:
            raise PromptStoreError("prompt store overlays are removed; publish and pin an authority release")
        if package is None:
            from .snapshot import ENV_SNAPSHOT, verify_snapshot
            raw = os.environ.get(ENV_SNAPSHOT)
            if not raw:
                raise PromptStoreError("no verified prompt snapshot; the frozen Git store is not a runtime source")
            package = verify_snapshot(Path(raw)).path
        package = Path(package)
        if not package.is_dir():
            raise PromptStoreError(f"prompt read view {package} is not a directory")
        self.layers = (Layer(package, "package"),)
        self._native: dict[str, tuple[Mapping[str, Any], ...]] = {}

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Store":
        """Build the store from ``RUBIKSQL_PROMPT_STORE``.

        Args:
            env (Mapping[str, str] | None): Environment to read; ``os.environ`` by default.

        Returns:
            Store: Package layer plus every non-empty ``os.pathsep``-separated overlay.
        """
        raw = (env if env is not None else os.environ).get(ENV_STORE, "")
        if raw.strip():
            raise PromptStoreError("runtime prompt overlays are removed")
        return cls()

    @property
    def package(self) -> Path:
        """The package layer directory.

        Returns:
            Path: First layer's path.
        """
        return self.layers[0].path

    def _read(self, relative: str, *, package_only: bool = False) -> Loaded | None:
        found: list[tuple[Layer, bytes]] = []
        for layer in self.layers[:1] if package_only else self.layers:
            path = layer.path / relative
            if path.is_file():
                found.append((layer, path.read_bytes()))
        if not found:
            return None
        first_layer, first_bytes = found[0]
        for layer, raw in found[1:]:
            if raw != first_bytes:
                raise PromptStoreError(f"{relative}: present in {first_layer.origin} and {layer.origin} with different bytes")
        try:
            data = json.loads(first_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PromptStoreError(f"{relative} in {first_layer.origin}: not valid UTF-8 JSON ({error})") from error
        return Loaded(data, first_layer.origin, relative, first_bytes)

    def require(self, relative: str, what: str, *, package_only: bool = False) -> Loaded:
        """Read one JSON file through the layers, failing loudly when it is absent.

        Args:
            relative (str): Path inside a store layer, already validated by the caller.
            what (str): Human name of the object for the error message.
            package_only (bool): Read the package layer alone (integrity files, matrix cells).

        Returns:
            Loaded: The parsed file with the origin that supplied it.

        Raises:
            PromptStoreError: If no layer has the file, layers disagree on its bytes, or it is not UTF-8 JSON.
        """
        loaded = self._read(relative, package_only=package_only)
        if loaded is None:
            raise PromptStoreError(f"{what} not found at {relative} in any store layer {[layer.origin for layer in self.layers]}")
        return loaded

    def object(self, name: str, version: str) -> Loaded:
        """Read one exported row.

        Args:
            name (str): Row name; validated before it becomes a path.
            version (str): 16-hex version; validated before it becomes a path.

        Returns:
            Loaded: The object file and its origin.

        Raises:
            PromptStoreError: If a component is invalid, the file is missing, unreadable, or differs across layers.
        """
        try:
            check_name(name)
            check_version(version)
        except PolicyError as error:
            raise PromptStoreError(str(error)) from error
        return self.require(f"objects/{name}/{version}.json", f"object {name}:{version}")

    def release(self, rid: str) -> Loaded:
        """Read one release manifest.

        Args:
            rid (str): Release id; validated before it becomes a path.

        Returns:
            Loaded: The manifest and its origin.

        Raises:
            PromptStoreError: If the id is invalid, the file is missing, unreadable, or differs across layers.
        """
        try:
            check_release_id(rid)
        except PolicyError as error:
            raise PromptStoreError(str(error)) from error
        return self.require(f"releases/{rid}.json", f"release {rid}")

    def proposal(self, pid: str) -> Loaded:
        """Read one proposal.

        Args:
            pid (str): Proposal id; validated before it becomes a path.

        Returns:
            Loaded: The proposal and its origin.

        Raises:
            PromptStoreError: If the id is invalid, the file is missing, unreadable, or differs across layers.
        """
        try:
            check_proposal_id(pid)
        except PolicyError as error:
            raise PromptStoreError(str(error)) from error
        loaded = self.require(f"proposals/{pid}.json", f"proposal {pid}")
        if loaded.data.get("origin", "agent") not in {"agent", "human", "cutover", "entity-model"}:
            raise PromptStoreError(f"proposal {pid}: unsupported publication origin")
        return loaded

    def native(self, entity: str) -> tuple[Mapping[str, Any], ...]:
        """Read the complete native row set ``native/<entity>.json`` of one version entity.

        Args:
            entity (str): An entity id of :data:`TABLES`.

        Returns:
            tuple[Mapping[str, Any], ...]: The rows as projected (sorted by object id); empty when the
                file is absent, as in a snapshot /1 for a typed entity. Read once per Store; callers must
                not mutate the rows.

        Raises:
            PromptStoreError: Unknown entity, layers disagree on the bytes, or the file is not an array of
                rows with distinct string object ids.
        """
        if entity not in TABLES:
            raise PromptStoreError(f"{entity!r} is not a native version entity; known {sorted(TABLES)}")
        if entity not in self._native:
            loaded = self._read(f"native/{entity}.json")
            rows = () if loaded is None else loaded.data
            if (not isinstance(rows, (list, tuple)) or any(not isinstance(row, Mapping) or not isinstance(row.get("object_id"), str)
                                                           for row in rows)
                    or len({row["object_id"] for row in rows}) != len(rows)):
                raise PromptStoreError(f"native/{entity}.json: expected an array of rows with distinct string object ids")
            self._native[entity] = tuple(rows)
        return self._native[entity]

    def typed_rows(self, entity: str) -> Mapping[str, Mapping[str, Any]]:
        """Read the canonical typed versions of one typed entity.

        Args:
            entity (str): ``tool-entity``, ``general-rule-entity`` or ``knob-set-entity``.

        Returns:
            Mapping[str, Mapping[str, Any]]: Canonical rows by object id; empty when the file is absent.
                Rows are classified, not verified; :func:`entity_model.resolve_closure` verifies a release.

        Raises:
            PromptStoreError: Not a typed entity, or the file holds a projection row or an unknown format.
        """
        from .entity_model import EntityModelError, is_canonical

        if entity not in TYPED_TABLES:
            raise PromptStoreError(f"{entity!r} is not a typed version entity; known {sorted(TYPED_TABLES)}")
        rows = {}
        for row in self.native(entity):
            try:
                canonical = is_canonical(entity, row)
            except EntityModelError as error:
                raise PromptStoreError(f"native/{entity}.json: {error}") from error
            if not canonical:
                raise PromptStoreError(f"native/{entity}.json: row {row['object_id']!r} is not a canonical typed version")
            rows[row["object_id"]] = row
        return rows

    def typed_index(self) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
        """Read every canonical typed version, as closure checks and diffs consume them.

        Returns:
            Mapping[str, Mapping[str, Mapping[str, Any]]]: Typed entity id -> object id -> canonical row.

        Raises:
            PromptStoreError: A typed native file is malformed or holds a non-canonical row.
        """
        return {entity: self.typed_rows(entity) for entity in TYPED_TABLES}

    def baseline(self) -> BaselinePointer:
        """Read the package ``BASELINE.json``.

        Returns:
            BaselinePointer: The pointed release id and the pointer file's git blob sha.

        Raises:
            PromptStoreError: If the current-reference file is absent or malformed.
        """
        loaded = self._read("BASELINE.json", package_only=True)
        if loaded is None:
            raise PromptStoreError(
                f"{self.package / 'BASELINE.json'} is absent: authority current reference is absent "
                "so no release can be resolved from 'baseline'"
            )
        data = loaded.data
        if not isinstance(data, Mapping) or data.get("schema") != SCHEMA_BASELINE:
            raise PromptStoreError(f"BASELINE.json: expected schema {SCHEMA_BASELINE!r}")
        try:
            rid = check_release_id(data.get("release"))
        except PolicyError as error:
            raise PromptStoreError(f"BASELINE.json: {error}") from error
        return BaselinePointer(rid, git_blob_sha(loaded.raw))

    def v0_root(self) -> str | None:
        """Read the write-once v0 boundary and validate its release identity.

        Fresh roots use the marker's full delivery commit as their epoch and carry
        the same source fields. The unsalted ``"v0"`` epoch belongs only to the
        original root ``redc46b38b8238949``.

        Returns:
            str | None: The validated root release, or ``None`` when the marker is absent.

        Raises:
            PromptStoreError: The marker, root closure, epoch or source identity is invalid.
        """
        loaded = self._read("V0.json", package_only=True)
        if loaded is None:
            return None
        data = loaded.data
        if not isinstance(data, Mapping) or data.get("schema") != SCHEMA_V0 or data.get("label") != "v0":
            raise PromptStoreError("V0.json: invalid v0 marker")
        try:
            root = check_release_id(data.get("release"))
            check_release_id(data.get("source_release"))
        except PolicyError as error:
            raise PromptStoreError(f"V0.json: {error}") from error
        for field in ("source_commit", "delivery_commit"):
            value = data.get(field)
            if not isinstance(value, str) or not _COMMIT.fullmatch(value):
                raise PromptStoreError(f"V0.json: {field} must be a full 40-hex commit")
        manifest = self.release(root).data
        from .entity_model import CLOSURE_V2, SCHEMA_RELEASE_V2
        valid_schema = SCHEMA_RELEASE_V2 if isinstance(manifest, Mapping) and manifest.get("rules") == CLOSURE_V2 else SCHEMA_RELEASE
        if (not isinstance(manifest, Mapping) or manifest.get("schema") != valid_schema or manifest.get("id") != root
                or manifest.get("origin") != "v0"
                or manifest.get("parents") or manifest.get("proposal")):
            raise PromptStoreError(f"V0.json: release {root} is not a v0 root")
        epoch = manifest.get("epoch")
        if epoch != f"v0:{data['delivery_commit']}":
            raise PromptStoreError(f"V0.json: release {root} epoch differs from the marker's delivery_commit")
        for field in ("source_release", "source_commit", "delivery_commit"):
            if manifest.get(field) != data[field]:
                raise PromptStoreError(f"V0.json: release {root} {field} differs from the marker")
        try:
            valid_id = release_document_id(manifest)
        except (KeyError, TypeError, ValueError) as error:
            raise PromptStoreError(f"V0.json: release {root} has an invalid closure") from error
        if valid_id != root:
            raise PromptStoreError(f"V0.json: release {root} has a different v0 content identity")
        return root

    def require_seeded_v0(self) -> None:
        """Require the current authority epoch; archived/unseeded stores never run."""
        if self.v0_root() is None:
            raise PromptStoreError("verified prompt read view has no seeded authority epoch")


    def v0_scope(self) -> V0Scope | None:
        """Find releases admitted by the current-state anchor or proposals descending from v0.

        A missing marker keeps scratch and historical test stores usable. A marked
        store rejects a BASELINE outside its new history instead of silently
        projecting the old store or counting its attempts.
        """
        root = self.v0_root()
        if root is None:
            return None
        epoch = self.release(root).data["epoch"]
        releases = {root}
        events = self.native("prompt-event-entity")
        has_state = any(event["kind"] == "register" and event["payload"].get("schema") in {"rubiksql.authority-state/1", "rubiksql.fresh-epoch/1"}
                        for event in events)
        anchor = None
        try:
            if has_state:
                from .authority_state import validate_native_state
                from .fresh_epoch import validate_native_admission

                native = {entity: self.native(entity) for entity in TABLES}
                anchor = validate_native_admission(native) or validate_native_state(native)
        except (AuthorityError, KeyError, TypeError, ValueError) as error:
            raise PromptStoreError(f"invalid current-state anchor: {error}") from error
        if anchor is not None:
            state = anchor["payload"]
            rid = str(state.get("release"))
            release = self.release(rid).data
            expected_origin = "v0" if state["schema"] == "rubiksql.fresh-epoch/1" else "state"
            if (release.get("origin") != expected_origin or release.get("authority_state") != anchor["object_id"]
                    or release.get("epoch") != epoch or state.get("epoch", {}).get("epoch") != epoch
                    or state.get("generation") != rid
                    or state.get("content_digest") != sha256_hex(canon({key: release.get(key)
                        for key in ("refs", "entries", "rules", "content_refs")}))
                    or release_document_id(release) != rid):
                raise PromptStoreError(f"release {rid}: invalid current-state anchor")
            releases.add(rid)
        proposals: set[str] = set()
        paths = sorted((self.package / "proposals").glob("p*.json"))
        while True:
            grew = False
            for path in paths:
                pid = path.stem
                if pid in proposals:
                    continue
                body = self.proposal(pid).data
                if not isinstance(body, Mapping):
                    raise PromptStoreError(f"proposals/{pid}.json: expected an object")
                parents = body.get("parents")
                if (isinstance(parents, list) and parents
                        and all(isinstance(parent, Mapping) and parent.get("release") in releases for parent in parents)):
                    rid = body.get("release")
                    if not isinstance(rid, str):
                        raise PromptStoreError(f"proposals/{pid}.json: missing release")
                    release = self.release(rid).data
                    if not isinstance(release, Mapping) or release.get("epoch") != epoch:
                        raise PromptStoreError(f"proposals/{pid}.json: release {rid} is outside the v0 epoch")
                    check_release_format(release)
                    try:
                        identity = release_document_id(release)
                    except (KeyError, TypeError, ValueError) as error:
                        raise PromptStoreError(f"proposals/{pid}.json: release {rid} has an invalid closure") from error
                    if identity != rid:
                        raise PromptStoreError(f"proposals/{pid}.json: release {rid} has a different v0 content identity")
                    releases.add(rid)
                    proposals.add(pid)
                    grew = True
            if not grew:
                break
        baseline = self.baseline().release
        if baseline not in releases:
            raise PromptStoreError(f"BASELINE.json: {baseline} lies outside v0 history rooted at {root}")
        return V0Scope(root, frozenset(releases), frozenset(proposals))

    def v0_versions(self, scope: V0Scope, name: str) -> set[str]:
        """The versions of one row that a v0 history holds: those its releases carry, the v0 root's included.

        Args:
            scope (V0Scope): The v0 history (:meth:`v0_scope`).
            name (str): The row name.

        Returns:
            set[str]: The 16-hex versions; empty when no release of the history carries the row.
        """
        return {str(key)[:16] for key in (self.release(rid).data["refs"].get(name) for rid in scope.releases) if key}

    def revoked(self) -> Mapping[str, str]:
        """Read the package revocation list.

        Returns:
            Mapping[str, str]: Content key to decision id; empty when the file is absent.

        Raises:
            PromptStoreError: If the file exists but is malformed.
        """
        loaded = self._read("revoked.json", package_only=True)
        if loaded is None:
            return {}
        data = loaded.data
        if not isinstance(data, Mapping) or data.get("schema") != SCHEMA_REVOKED or not isinstance(data.get("revoked"), Mapping):
            raise PromptStoreError(f"revoked.json: expected {{'schema': {SCHEMA_REVOKED!r}, 'revoked': {{content_key: decision}}}}")
        return dict(data["revoked"])

    def is_retired(self, rid: str) -> bool:
        """Tell whether the package layer marks a release as retired.

        Args:
            rid (str): Release id, already validated.

        Returns:
            bool: ``True`` when ``retired/<rid>.json`` exists in the package layer.
        """
        return (self.package / "retired" / f"{rid}.json").is_file()

    def decisions(self) -> tuple[Mapping[str, Any], ...]:
        """Read every decision file from the package layer.

        Returns:
            tuple[Mapping[str, Any], ...]: Decisions sorted by file name.

        Raises:
            PromptStoreError: If a decision file is malformed.
        """
        directory = self.package / "decisions"
        if not directory.is_dir():
            return ()
        decisions = []
        for path in sorted(directory.glob("*.json")):
            loaded = self.require(f"decisions/{path.name}", f"decision {path.stem}", package_only=True)
            if not isinstance(loaded.data, Mapping) or loaded.data.get("schema") != SCHEMA_DECISION:
                raise PromptStoreError(f"{loaded.relative}: expected schema {SCHEMA_DECISION!r}")
            if loaded.data.get("kind") not in DECISION_KINDS:
                raise PromptStoreError(f"{loaded.relative}: unsupported decision kind {loaded.data.get('kind')!r}")
            decisions.append(loaded.data)
        return tuple(decisions)
