"""Strict, read-only loading of retained candidate pools for judge diagnostics."""
from __future__ import annotations

from copy import deepcopy
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sqlite3
from typing import TYPE_CHECKING

from . import cache_lineage
from .question import question_with_hint

if TYPE_CHECKING:
    from .prompts.runtime import Release
    from .prompts.snapshot import VerifiedSnapshot


SCHEMA = "rubiksql.saved-candidates/1"
VARIANT_SCHEMA = "rubiksql.saved-candidates/2"


def _candidate_knobs() -> dict:
    from .prompts.entries import KNOBS

    return {name: knob for name, knob in KNOBS.items() if any(role == "candidate" for role, _ in knob.visible)}


def _validate_surface(surface: object) -> None:
    """Check the complete candidate proof carried by a cross-release admission."""
    from .prompts import entries as E

    if not isinstance(surface, dict) or set(surface) != {"keyspaces", "knobs", "tools"}:
        raise ValueError("invalid saved candidate surface evidence")
    bases = {E.route_label("candidate", route) for route in E.ROUTES["candidate"]}
    labels = bases | {E.route_label("candidate", route, slot) for route in E.ROUTES["candidate"]
                      for slot in range(1, E.CANDIDATE_SLOTS + 1)}
    spaces, tools = surface["keyspaces"], surface["tools"]
    if (not isinstance(spaces, dict) or not bases <= spaces.keys() <= labels
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{16}", value) for value in spaces.values())
            or not isinstance(tools, dict) or set(tools) != bases
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in tools.values())):
        raise ValueError("invalid saved candidate keyspace/tool evidence")
    knobs = surface["knobs"]
    required = _candidate_knobs().keys() - {"examples_inline_sql"}
    if not isinstance(knobs, dict) or not required <= knobs.keys() <= _candidate_knobs().keys():
        raise ValueError("invalid saved candidate knob evidence")
    for name, knob in _candidate_knobs().items():
        # Absence is the historical OFF spelling; ON must carry an explicit bool.
        if knob.violation(knobs.get(name, False)) is not None:
            raise ValueError(f"invalid saved candidate knob evidence: {name}")


def _validate_runtime(runtime: object) -> None:
    """Require the recorded non-release candidate controls for every mixed source."""
    if (not isinstance(runtime, dict) or set(runtime) != {"router", "disable_tools", "model_protocol", "transport_protocol",
                                                        "search_columns_profile", "final_sql_timeout_sec"}
            or runtime["router"] is not None and not isinstance(runtime["router"], str)
            or not isinstance(runtime["disable_tools"], list)
            or any(not isinstance(name, str) or not name for name in runtime["disable_tools"])
            or runtime["disable_tools"] != sorted(set(runtime["disable_tools"]))
            or any(not isinstance(runtime[name], str) or not runtime[name] for name in ("model_protocol", "transport_protocol"))
            or type(runtime["search_columns_profile"]) is not bool
            or type(runtime["final_sql_timeout_sec"]) not in (int, float)
            or type(runtime["final_sql_timeout_sec"]) is float and not math.isfinite(runtime["final_sql_timeout_sec"])
            or runtime["final_sql_timeout_sec"] <= 0):
        raise ValueError("invalid saved candidate runtime evidence")


@dataclass(frozen=True)
class SavedPool:
    """A complete ordered pool with its source, database and knowledge identities."""

    identity: dict
    candidates: str
    input_fingerprints: dict[str, str]
    knowledge_fingerprints: dict[str, str]
    rows: dict[tuple[str, str], list[dict]]


def validate_identity(identity: object) -> None:
    """Reject malformed saved-pool identities before recording an admission.

    Args:
        identity: JSON-shaped identity returned by :func:`load_pool`.

    Returns:
        None: The identity has the complete source selection and evidence hashes.

    Raises:
        ValueError: The schema, ordered source selection or evidence hashes are invalid.
    """
    def digest(value):
        return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value)

    variant = isinstance(identity, dict) and identity.get("schema") == VARIANT_SCHEMA
    fields = {"schema", "release", "question_set", "knowledge_fingerprints", "sources"}
    if variant:
        fields |= {"candidate_surface", "candidate_runtime"}
    if (not isinstance(identity, dict) or set(identity) != fields
            or identity.get("schema") not in (SCHEMA, VARIANT_SCHEMA) or not isinstance(identity.get("release"), str) or not identity["release"]
            or not digest(identity.get("question_set")) or not isinstance(identity.get("sources"), list) or not identity["sources"]):
        raise ValueError("invalid saved candidate identity")
    if variant:
        if not re.fullmatch(r"r[0-9a-f]{16}", identity["release"]):
            raise ValueError("invalid saved candidate pin release")
        _validate_surface(identity["candidate_surface"])
        _validate_runtime(identity["candidate_runtime"])
    knowledge = identity["knowledge_fingerprints"]
    if (not isinstance(knowledge, dict) or not knowledge
            or any(not isinstance(db, str) or not db or not isinstance(value, str) or not value for db, value in knowledge.items())):
        raise ValueError("invalid saved candidate knowledge fingerprints")
    seen, reservations = set(), set()
    for source in identity["sources"]:
        fields = {"run", "cache_db", "reservation_id", "claim_commit", "manifest_sha256", "inputs_sha256", "results_sha256", "slots", "models", "payloads_sha256", "payload_count", "recorded_failure_count"}
        if variant:
            fields |= {"release", "candidate_surface"}
        # Present only when a selected failed seat has no producer evidence, so
        # identities of pools without such seats keep their original fields.
        if isinstance(source, dict) and "unattributed_failure_count" in source:
            fields |= {"unattributed_failure_count"}
        if (not isinstance(source, dict) or set(source) != fields
                or any(not isinstance(source.get(key), str)
                       or not (PurePosixPath(source[key]).is_absolute() or PureWindowsPath(source[key]).is_absolute()) for key in ("run", "cache_db"))
                or not isinstance(source.get("reservation_id"), str) or not source["reservation_id"].strip()
                or not isinstance(source.get("claim_commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", source["claim_commit"])
                or any(not digest(source.get(key)) for key in ("manifest_sha256", "inputs_sha256"))):
            raise ValueError("invalid saved candidate source identity")
        if variant:
            if not isinstance(source["release"], str) or not re.fullmatch(r"r[0-9a-f]{16}", source["release"]):
                raise ValueError("invalid saved candidate source release")
            _validate_surface(source["candidate_surface"])
            if source["candidate_surface"] != identity["candidate_surface"]:
                raise ValueError("saved candidate source surface differs from the pin")
        if source["reservation_id"] in reservations:
            raise ValueError("duplicate saved candidate source reservation")
        reservations.add(source["reservation_id"])
        slots, models = source["slots"], source["models"]
        if (not isinstance(slots, list) or not slots or any(type(slot) is not int or slot < 1 for slot in slots)
                or len(slots) != len(set(slots)) or not isinstance(models, list) or len(models) != len(slots)
                or any(not isinstance(model, dict) or set(model) != {"provider", "model"}
                       or any(not isinstance(value, str) or not value for value in model.values()) for model in models)):
            raise ValueError("invalid saved candidate model selection")
        if any((source["run"], slot) in seen for slot in slots):
            raise ValueError("duplicate saved candidate source selection")
        seen.update((source["run"], slot) for slot in slots)
        files = source["results_sha256"]
        if (not isinstance(files, dict) or not files or any(not isinstance(name, str) or not name or not digest(value) for name, value in files.items())
                or not digest(source["payloads_sha256"]) or type(source["payload_count"]) is not int or source["payload_count"] < 0
                or type(source["recorded_failure_count"]) is not int or source["recorded_failure_count"] < 0
                or source["payload_count"] + source["recorded_failure_count"] < 1):
            raise ValueError("invalid saved candidate evidence hashes")
        if "unattributed_failure_count" in source and (
                type(source["unattributed_failure_count"]) is not int
                or not 1 <= source["unattributed_failure_count"] <= source["recorded_failure_count"]):
            raise ValueError("invalid saved candidate unattributed failure count")


def _read(path: Path) -> tuple[dict, str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError) as error:
        raise ValueError(f"saved candidates cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"saved candidates expected an object in {path}")
    return value, "sha256:" + hashlib.sha256(raw).hexdigest()


def _models(identity: dict) -> list[dict]:
    spec = identity.get("candidates")
    if spec is None:
        configured = identity.get("provider")
        provider = configured.split(":") if isinstance(configured, str) else []
        if len(provider) != 2 or not all(provider):
            raise ValueError("saved candidates source lacks its candidate model")
        return [{"provider": provider[0], "model": provider[1]}]
    if not isinstance(spec, str) or not spec:
        raise ValueError("saved candidates source has an invalid candidate specification")
    models = []
    for entry in spec.split(","):
        fields = entry.strip().split(":")
        if len(fields) < 3 or not all(fields[:2]) or not fields[2].isdigit() or int(fields[2]) < 1:
            raise ValueError(f"saved candidates source has an invalid model specification: {entry!r}")
        models.extend([{"provider": fields[0], "model": fields[1]}] * int(fields[2]))
    return models


def _verify_manifest(manifest: dict, *, release: str, question_set: str, questions: dict) -> VerifiedSnapshot:
    from .prompts.snapshot import SnapshotError, snapshot_record, verify_snapshot

    def role_record(record, role):
        if not isinstance(record, dict) or not isinstance(record.get("context"), dict):
            return False
        context = record["context"]
        return (context.get("role") == role and isinstance(context.get("tools"), list)
                and all(isinstance(tool, str) and tool for tool in context["tools"])
                and (role != "candidate" or isinstance(context.get("route"), str) and bool(context["route"]))
                and isinstance(record.get("digest"), str) and re.fullmatch(r"[0-9a-f]{64}", record["digest"]))

    implementation = manifest.get("implementation")
    identity = manifest.get("identity")
    if (manifest.get("schema") != "rubiksql.prompt-manifest/1" or manifest.get("release") != release
            or not isinstance(release, str) or not re.fullmatch(r"r[0-9a-f]{16}", release)
            or manifest.get("store") != "snapshot" or manifest.get("mixed") or manifest.get("releases")
            or not isinstance(implementation, dict) or not isinstance(identity, dict)
            or implementation.get("clean") is not True or not isinstance(implementation.get("commit"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", implementation["commit"])
            or any(not isinstance(implementation.get(key), str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", implementation[key])
                   for key in ("uv_lock_sha256", "wheel_sha256")) or identity.get("question_set") != question_set
            or manifest.get("questions") != questions):
        raise ValueError("saved candidates require a clean, verified, unmixed source of the exact release and complete question set")
    roles = implementation.get("role_digests")
    if (not isinstance(roles, dict) or not isinstance(roles.get("candidate"), list) or not roles["candidate"]
            or not all(role_record(record, "candidate") for record in roles["candidate"])
            or any(not role_record(roles.get(role), role) for role in ("judge", "reviewer"))):
        raise ValueError("saved candidates source lacks complete implementation role digests")
    record = manifest.get("prompt_snapshot")
    if not isinstance(record, dict) or record.get("release") != release or not record.get("path") or not record.get("digest"):
        raise ValueError("saved candidates source lacks verified prompt snapshot provenance")
    try:
        snapshot = verify_snapshot(Path(record["path"]), expected_release=release, expected_digest=record["digest"],
                                   expected_authority_commit=record.get("authority_commit"))
    except SnapshotError as error:
        raise ValueError(f"saved candidates source snapshot cannot be verified: {error}") from error
    if snapshot_record(snapshot) != record:
        raise ValueError("saved candidates source snapshot provenance differs from its verified snapshot")
    return snapshot


def _candidate_surface(snapshot: VerifiedSnapshot) -> tuple[Release, dict]:
    """Derive candidate inputs from verified content, never manifest assertions."""
    from .prompts import entries as E
    from .prompts.runtime import load_release
    from .prompts.store import Store
    from .tool_bundle import read_bundle, snapshot_role_contract
    from .tool_definitions import select_definitions

    release = load_release(snapshot.release, stores=Store(package=snapshot.path), allow_retired=True)
    settings = release.knob_settings()
    knobs = {name: list(settings[name]) if isinstance(settings[name], tuple) else settings[name] for name in _candidate_knobs()}
    if knobs.get("examples_inline_sql") is False:
        del knobs["examples_inline_sql"]  # Preserve pre-knob saved identities and diagnostic namespaces.
    # Rule/generator rows affect candidates through materialized knowledge,
    # whose fingerprints are checked separately against every source and pin.
    definitions = read_bundle(snapshot).definitions
    contracts = {}
    for route in E.ROUTES["candidate"]:
        context = {"role": "candidate", "route": route or "default",
                   "tools": [item["name"] for item in select_definitions(definitions, "candidate", route or "default")]}
        contracts[E.route_label("candidate", route)] = snapshot_role_contract(snapshot, context)["digest"]
    surface = {"keyspaces": {name: value for name, value in release.keyspaces().items()
                             if name == "candidate" or name.startswith(("candidate/", "candidate["))},
               "knobs": knobs, "tools": contracts}
    _validate_surface(surface)
    return release, surface


def _source_runtime(manifest: dict, snapshot: VerifiedSnapshot, pin_snapshot: VerifiedSnapshot, release: Release) -> dict:
    """Verify effective candidate controls and the source's recorded implementation."""
    from .prompts.entries import KNOBS
    from .tool_bundle import snapshot_role_contract

    identity = manifest["identity"]
    names = ("router", "disable_tools", "model_protocol", "transport_protocol", "search_columns_profile", "final_sql_timeout_sec")
    runtime = {name: identity[name] for name in names if name in identity}
    if "disable_tools" in runtime:
        disabled = runtime["disable_tools"]
        if disabled is not None and not isinstance(disabled, str):
            raise ValueError("invalid saved candidate disabled-tools evidence")
        runtime["disable_tools"] = sorted({name for name in (disabled or "").split(",") if name})
    _validate_runtime(runtime)
    overrides, recorded = identity.get("knobs_env", {}), manifest.get("knobs")
    if (not isinstance(overrides, dict) or overrides.keys() - KNOBS.keys() or not isinstance(recorded, dict)):
        raise ValueError("saved candidates source lacks effective candidate knob evidence")
    settings = release.knob_settings()
    for name, knob in _candidate_knobs().items():
        record = recorded.get(name)
        if (name == "examples_inline_sql" and name not in recorded and name not in overrides
                and name not in release.parameter_values() and settings[name] is False):
            # Only a verified source with default OFF can omit historical evidence.
            continue
        if (not isinstance(record, dict) or set(record) != {"value", "source"}
                or record["source"] not in ("default", "release", "env")
                or knob.violation(record["value"]) is not None or knob.normalize(record["value"]) != settings[name]
                or (record["source"] == "env") != (name in overrides)
                or name in overrides and (knob.violation(overrides[name]) is not None
                                          or knob.normalize(overrides[name]) != settings[name])):
            raise ValueError(f"saved candidates source effective candidate knob differs from its release: {name}")
    for record in manifest["implementation"]["role_digests"]["candidate"]:
        context = record["context"]
        if (snapshot_role_contract(snapshot, context) != record
                or snapshot_role_contract(pin_snapshot, context) != record):
            raise ValueError("saved candidates source candidate implementation differs from the verified snapshots")
    return runtime


def _ordered_rows(rows: list, columns: list, *, label: str) -> list:
    """Restore SELECT order after source verification; leave positional rows unchanged."""
    if any(isinstance(row, dict) for row in rows):
        if (not all(isinstance(column, str) for column in columns)
                or not all(isinstance(row, dict) and set(row) == set(columns) for row in rows)):
            raise ValueError(f"saved candidates mapping rows differ from recorded columns: {label}")
        # Repeated column names retain the first insertion order of dict(zip(...)).
        return [{column: row[column] for column in columns} for row in rows]
    return deepcopy(rows)


def _invalidates_candidates(record: dict) -> bool:
    return bool(record.get("pool_guard_fallback") or record.get("degraded_retrieval") or record.get("placeholder"))


def _needs_reanswer(record: dict) -> bool:
    return (_invalidates_candidates(record) or bool(record.get("error"))
            or record.get("decider_source") == "fallback_candidate")


def _recorded_failure(record: dict) -> bool:
    """Match a live failure, including archives written before empty results were attached."""
    return (isinstance(record.get("error"), str) and bool(record["error"].strip())
            and isinstance(record.get("sql"), str) and not record["sql"].strip()
            and record.get("rows", []) == [] and type(record.get("row_count", 0)) is int
            and record.get("row_count", 0) == 0 and not _invalidates_candidates(record)
            and record.get("decider_source") != "fallback_candidate")


#: Fields an unattributed failed seat may carry with a value: the recorded failure itself,
#: its exception classification and source timings. Every other field must be blank.
_UNATTRIBUTED_FAILURE_FIELDS = frozenset({"sql", "rows", "row_count", "error", "_exc_type",
                                          "_dead_session", "_mcp_error_code", "provider_failure", "timing", "shared_session_timing"})


def _blank(value: object) -> bool:
    """Absent producer data as the candidate projection spells it: None, False or an empty list."""
    return value is None or value is False or type(value) is list and not value


def _unattributed_failure(summary: dict, debug: dict) -> bool:
    """Match a failed seat the run recorded without any candidate producer evidence.

    The candidate gather boundary records a seat it stopped before an attempt record
    existed (for example, a candidate-cap cancellation) as its error and classification: no execution
    identity, transcript, prompt or implementation record was ever produced. The source
    result file, its hash and the seat's position are the only evidence, and the only
    content: an empty failure the judge reads as that error. A record carrying any
    producer evidence, or any success, is never this class and keeps the strict checks.

    Args:
        summary: The seat's ``candidates`` entry.
        debug: The seat's ``candidates_debug`` entry.

    Returns:
        bool: Both entries record the same failure and nothing else.
    """
    return (_recorded_failure(summary) and _recorded_failure(debug) and summary["error"] == debug["error"]
            and set(summary) <= {"sql", "error", "row_count", "rows", "provider_failure"}
            and all(name in _UNATTRIBUTED_FAILURE_FIELDS or _blank(value) for name, value in debug.items()))


def _seat_sha(summary: dict, debug: dict) -> str:
    """Hash one recorded seat's exact JSON content, independent of key order."""
    canonical = json.dumps({"summary": summary, "debug": debug}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _candidate(conn: sqlite3.Connection, summary: dict, debug: dict, *, run: Path, slot: int,
               key: tuple[str, str], model: dict, manifest_sha: str, result_sha: str, knowledge_fingerprint: str,
               candidate_release: Release | None = None, candidate_contracts: list[dict] | None = None) -> dict:
    label = f"{run} {key[0]}/{key[1]} slot {slot}"
    proof = debug.get("cache_source")
    execution = debug.get("execution")
    failed = _recorded_failure(summary) and _recorded_failure(debug) and summary["error"] == debug["error"]
    # A failed seat without producer evidence is proven only by its source file and slot;
    # there is no candidate execution, keyspace or implementation to verify or invent.
    unattributed = failed and _unattributed_failure(summary, debug)
    if not unattributed and (
            _invalidates_candidates(summary) or _invalidates_candidates(debug)
            or (not failed and (_needs_reanswer(summary) or _needs_reanswer(debug)
                                or debug.get("transcript_status") != "complete" or debug.get("recording_error")))
            or ((proof is not None or not failed) and not cache_lineage.valid_descriptor(proof, attributed=True))
            or not isinstance(execution, dict)
            or execution.get("run") != run.name or execution.get("db") != key[0]
            or str(execution.get("qid")) != key[1] or execution.get("role") != "candidate"
            or type(execution.get("slot")) is not int or execution["slot"] != slot):
        raise ValueError(f"saved candidates missing complete candidate evidence: {label}")
    payload = {}
    if proof is not None:
        origin = proof["identity"]
        if (origin["db"] != key[0] or origin["qid"] != key[1] or origin["role"] != "candidate"
                or cache_lineage.content_sha(debug) != proof["content_sha256"]):
            raise ValueError(f"saved candidates source attribution/transcript mismatch: {label}")
        if origin["knowledge_fingerprint"] != knowledge_fingerprint:
            raise ValueError(f"saved candidates producer knowledge fingerprint differs from its source: {label}")
        document = cache_lineage.read(conn, cache_lineage.REF + proof["id"])
        if document is None or cache_lineage.descriptor(document) != proof:
            raise ValueError(f"saved candidates immutable payload is missing or changed: {label}")
        payload = document["payload"]
    if candidate_release is not None and not unattributed:
        from .prompts.entries import ROUTES

        prompt = (payload if proof is not None else debug).get("prompt_identity")
        recorded = debug.get("prompt_identity")
        producer_release = proof["identity"]["release"] if proof is not None else candidate_release.id
        if (not isinstance(prompt, dict) or not isinstance(recorded, dict)
                or not {"release", "role", "route", "keyspace"} <= prompt.keys() & recorded.keys()
                or prompt.get("role") != "candidate" or prompt.get("route") not in ROUTES["candidate"]
                or prompt.get("release") != producer_release
                or prompt.get("keyspace") != candidate_release.keyspace("candidate", prompt.get("route"), slot)
                or any(recorded.get(name) != prompt.get(name) for name in ("release", "role", "route", "keyspace"))):
            raise ValueError(f"saved candidates producer candidate keyspace cannot be verified: {label}")
        implementation = (payload if proof is not None else debug).get("implementation_context")
        if (not isinstance(implementation, dict) or implementation not in (candidate_contracts or [])
                or implementation["context"]["route"] != (prompt.get("route") or "default")
                or debug.get("implementation_context") != implementation):
            raise ValueError(f"saved candidates producer candidate implementation cannot be verified: {label}")
    if failed:
        # Failures never enter the answer cache, but may have an immutable turn archive.
        # content_sha does not cover errors/status/results, so verify these separately.
        if proof is not None and (not _recorded_failure(payload) or payload["error"] != debug["error"]
                                  or payload.get("transcript_status") != debug.get("transcript_status")):
            raise ValueError(f"saved candidates immutable payload differs from recorded failure: {label}")
        record = {**deepcopy(payload), **deepcopy(debug), "sql": "", "rows": [], "columns": [], "row_count": 0}
    else:
        rows, columns, count = payload.get("rows"), payload.get("columns"), payload.get("row_count")
        messages = payload.get("messages")
        if (not isinstance(payload.get("sql"), str) or not payload["sql"].strip()
                or payload["sql"] != summary.get("sql") or payload["sql"] != debug.get("sql")
                or _needs_reanswer(payload) or payload.get("transcript_status") != "complete"
                or payload.get("recording_error") or not isinstance(rows, list) or not isinstance(columns, list)
                or type(count) is not int or count < len(rows) or summary.get("row_count") != count
                or not isinstance(messages, list) or not messages
                or not all(isinstance(message, dict) for message in messages)
                or not any(message.get("role") == "assistant" for message in messages)):
            raise ValueError(f"saved candidates immutable payload lacks complete SQL/results/transcript: {label}")
        # Preserve source replay/consumer metadata as well as the immutable producer.
        record = {**deepcopy(payload), **deepcopy(debug), "rows": _ordered_rows(rows, columns, label=label),
                  "columns": deepcopy(columns), "row_count": count}
    record["saved_candidate_source"] = {
        "run": str(run), "slot": slot, **model, "manifest_sha256": manifest_sha,
        "result_sha256": result_sha, "cache_source": deepcopy(proof),
        **({"recorded_failure": True} if failed else {}),
        **({"unattributed_failure": True, "seat_sha256": _seat_sha(summary, debug)} if unattributed else {}),
        "usage": deepcopy((payload if proof is not None else debug).get("usage")),
        "duration_sec": (payload if proof is not None else debug).get("duration_sec"),
    }
    # Source timings and attempts describe historical work, never this consumption.
    for field in ("timing", "shared_session_timing", "attempts"):
        historical = record.pop(field, None)
        if historical is None:
            historical = deepcopy(payload.get(field))
        if historical is not None:
            record["saved_candidate_source"][field] = historical
    record.update(from_cache=True, usage={}, duration_sec=None)
    return record


def load_pool(path: Path, *, release: str, queries: list[dict], question_set: str) -> SavedPool:
    """Load every selected candidate without executing SQL or generating replacements.

    Args:
        path: JSON descriptor with schema ``rubiksql.saved-candidates/1`` and ordered
            ``sources`` objects containing absolute ``run`` and ``cache_db`` paths and
            one-based, distinct ``slots``.
        release: Diagnostic pin. Different source releases qualify only with equal
            verified candidate surfaces and compatible recorded candidate controls.
        queries: Complete normalized questions, including their hint preferences.
        question_set: SHA-256 identity of the original questions file.

    Returns:
        SavedPool: Detached candidate records keyed by database and string question id,
            their ordered model specification, immutable source identity, database hashes
            and the common source knowledge fingerprints. Each source identity counts
            selected recorded failures; these retain errors without SQL or result rows.
            Failed seats recorded without any producer evidence (a candidate stopped
            before its attempt record existed) are proven by their source file and
            slot alone; their provenance says ``unattributed_failure`` with the seat's
            content hash, and the source identity counts them separately.
            Cross-release identities use ``rubiksql.saved-candidates/2`` and record
            the pin, every source release, and their candidate equivalence evidence.
            Same-release identities retain the original version and behavior.

    Raises:
        ValueError: A source differs or any selected question, slot, result or transcript
            cannot be verified. No missing evidence is reconstructed or regenerated.
    """
    descriptor, _ = _read(Path(path))
    if descriptor.get("schema") != SCHEMA or not isinstance(descriptor.get("sources"), list) or not descriptor["sources"]:
        raise ValueError(f"saved candidates descriptor must use {SCHEMA} with nonempty sources")
    expected = {(str(query["database"]), str(query["id"])): question_with_hint(
        query["question"], query.get("context", {}).get("user_profile", {}).get("preferences", [])) for query in queries}
    if not expected or len(expected) != len(queries):
        raise ValueError("saved candidates require a nonempty, distinct complete question set")
    from .prompts.evolve import question_set_record
    questions = question_set_record(key[1] for key in expected)
    result: dict[tuple[str, str], list[dict]] = {key: [] for key in expected}
    identities, selected_models, seen, reservations = [], [], set(), set()
    producers = {key: {"id": set(), "content_sha256": set()} for key in expected}
    fingerprints, knowledge_fingerprints = None, None
    sources = []
    for source in descriptor["sources"]:
        if not isinstance(source, dict) or not isinstance(source.get("run"), str) or not isinstance(source.get("cache_db"), str):
            raise ValueError("saved candidates sources require absolute run and cache_db paths")
        run, cache = Path(source["run"]), Path(source["cache_db"])
        slots = source.get("slots")
        if (not run.is_absolute() or not cache.is_absolute() or not isinstance(slots, list) or not slots
                or any(type(slot) is not int or slot < 1 for slot in slots) or len(slots) != len(set(slots))):
            raise ValueError("saved candidates sources require absolute paths and distinct positive slots")
        run, cache = run.resolve(), cache.resolve()
        if any((run, slot) in seen for slot in slots):
            raise ValueError("saved candidates cannot select a source slot twice")
        seen.update((run, slot) for slot in slots)
        manifest, manifest_sha = _read(run / ".prompt_manifest.json")
        snapshot = _verify_manifest(manifest, release=manifest.get("release"), question_set=question_set, questions=questions)
        sources.append((run, cache, slots, manifest, manifest_sha, snapshot))
    variant = any(manifest["release"] != release for _, _, _, manifest, _, _ in sources)
    pin_surface, common_runtime = None, None
    if variant:
        from .prompts.snapshot import ENV_SNAPSHOT, SnapshotError, verify_snapshot
        from .prompts.runtime import PromptStoreError

        try:
            path = os.environ.get(ENV_SNAPSHOT)
            if not path:
                raise ValueError("saved candidates cross-release replay requires a verified pin snapshot")
            pin_snapshot = verify_snapshot(Path(path), expected_release=release)
            pin, pin_surface = _candidate_surface(pin_snapshot)
            for name, value in pin.knob_settings().items():
                if name in _candidate_knobs() and pin.knob(name) != value:
                    raise ValueError(f"saved candidates pin overrides a candidate knob: {name}")
        except (SnapshotError, PromptStoreError) as error:
            raise ValueError(f"saved candidates pin snapshot cannot be verified: {error}") from error
    for run, cache, slots, manifest, manifest_sha, snapshot in sources:
        source_release = manifest["release"]
        loaded, surface = None, None
        if variant:
            try:
                loaded, surface = _candidate_surface(snapshot)
                if surface != pin_surface:
                    raise ValueError(f"saved candidates candidate surface differs from the pin: {run}")
                runtime = _source_runtime(manifest, snapshot, pin_snapshot, loaded)
            except PromptStoreError as error:
                raise ValueError(f"saved candidates source release cannot be verified: {error}") from error
            if common_runtime is not None and runtime != common_runtime:
                raise ValueError(f"saved candidates source candidate runtime differs across the pool: {run}")
            common_runtime = runtime
        knowledge = manifest.get("knowledge_fingerprints")
        if (not isinstance(knowledge, dict) or set(knowledge) != {key[0] for key in expected}
                or not all(isinstance(value, str) and value for value in knowledge.values())
                or (knowledge_fingerprints is not None and knowledge_fingerprints != knowledge)):
            raise ValueError(f"saved candidates source has missing or differing knowledge fingerprints: {run}")
        knowledge_fingerprints = knowledge
        models = _models(manifest["identity"])
        if max(slots) > len(models):
            raise ValueError(f"saved candidates requested an undeclared source slot in {run}")
        inputs, inputs_sha = _read(run / ".evaluation_inputs.json")
        dbs = inputs.get("db_fingerprints")
        if (inputs.get("schema") != "rubiksql.evaluation-inputs/1" or inputs.get("question_set") != question_set
                or inputs.get("verified_after") is not True or not isinstance(dbs, dict)
                or set(dbs) != {key[0] for key in expected}
                or not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in dbs.values())
                or (fingerprints is not None and fingerprints != dbs)):
            raise ValueError(f"saved candidates source has unverified or differing evaluation inputs: {run}")
        admission = manifest.get("paid_launch")
        if (not isinstance(admission, dict) or admission.get("schema") != "rubiksql.paid-launch/1"
                or not isinstance(admission.get("reservation_id"), str) or not admission["reservation_id"].strip()
                or not isinstance(admission.get("claim_commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", admission["claim_commit"])
                or not isinstance(admission.get("out_dir"), str) or not Path(admission["out_dir"]).is_absolute()
                or Path(admission["out_dir"]).resolve() != run or admission.get("run") != run.name or admission.get("release") != source_release):
            raise ValueError(f"saved candidates source lacks a committed admission bound to its run and release: {run}")
        if admission["reservation_id"] in reservations:
            raise ValueError(f"saved candidates cannot repeat a source reservation: {run}")
        reservations.add(admission["reservation_id"])
        if admission.get("inputs") != {key: inputs[key] for key in ("schema", "question_set", "db_fingerprints")}:
            raise ValueError(f"saved candidates source input fingerprints differ from its admission: {run}")
        if not isinstance(admission.get("knowledge"), dict) or admission["knowledge"].get("fingerprints") != knowledge:
            raise ValueError(f"saved candidates source knowledge fingerprints differ from its admission: {run}")
        fingerprints = dbs
        result_hashes, source_rows, payloads = {}, {}, {}
        recorded_failure_count = unattributed_failure_count = 0
        for db in sorted(dbs):
            file = run / f"{db}.json"
            body, digest = _read(file)
            result_hashes[file.relative_to(run).as_posix()] = digest
            if not isinstance(body.get("results"), list):
                raise ValueError(f"saved candidates source lacks result rows: {file}")
            for row in body["results"]:
                if not isinstance(row, dict):
                    raise ValueError(f"saved candidates source has malformed result rows: {file}")
                key = (str(row.get("db_id")), str(row.get("id")))
                if key in source_rows or key not in expected or key[0] != db or row.get("prompt_release") != source_release or row.get("question") != expected[key]:
                    raise ValueError(f"saved candidates source has duplicate/foreign questions or release drift: {file}")
                source_rows[key] = (row, digest)
        if set(source_rows) != set(expected):
            raise ValueError(f"saved candidates source does not contain the complete question set: {run}")
        try:
            with closing(sqlite3.connect(cache.as_uri() + "?mode=ro", uri=True)) as conn:
                conn.execute("PRAGMA query_only=ON")
                for key, (row, digest) in source_rows.items():
                    summaries, debug = row.get("candidates"), row.get("candidates_debug")
                    if not isinstance(summaries, list) or not isinstance(debug, list) or max(slots) > min(len(summaries), len(debug)):
                        raise ValueError(f"saved candidates source is missing selected slots: {run} {key}")
                    if len(summaries) != len(models) or len(debug) < len(models):
                        raise ValueError(f"saved candidates source lacks its complete declared candidate pool: {run} {key}")
                    # Sweep/resume retains old-only pool-guard slots as attempt history.
                    if any(not isinstance(item, dict) or set(item) != {"transcript_status", "attempts"}
                           or item["transcript_status"] != "not_run" or not isinstance(item["attempts"], list)
                           or not item["attempts"] or any(not isinstance(attempt, dict) for attempt in item["attempts"])
                           for item in debug[len(models):]):
                        raise ValueError(f"saved candidates source has malformed trailing candidate history: {run} {key}")
                    debug = debug[:len(models)]
                    # The source judge/reviewer verdict is replaced. Explicit pool health
                    # markers still invalidate candidate evidence, including aggregate
                    # degraded retrieval whose origin may be absent from the projection.
                    if _invalidates_candidates(row):
                        raise ValueError(f"saved candidates source row requires a fresh re-answer: {run} {key}")
                    for slot in slots:
                        if not isinstance(summaries[slot - 1], dict) or not isinstance(debug[slot - 1], dict):
                            raise ValueError(f"saved candidates source has a malformed selected slot: {run} {key}")
                    for summary, item in zip(summaries, debug):
                        if not isinstance(summary, dict) or not isinstance(item, dict):
                            raise ValueError(f"saved candidates source lacks its complete declared candidate pool: {run} {key}")
                        failed = _recorded_failure(summary) and _recorded_failure(item) and summary["error"] == item["error"]
                        if not failed and (_needs_reanswer(summary) or _needs_reanswer(item)
                                           or not isinstance(summary.get("sql"), str) or not summary["sql"].strip()):
                            raise ValueError(f"saved candidates source lacks its complete declared candidate pool: {run} {key}")
                    for slot in slots:
                        record = _candidate(conn, summaries[slot - 1], debug[slot - 1], run=run, slot=slot, key=key,
                                            model=models[slot - 1], manifest_sha=manifest_sha, result_sha=digest,
                                            knowledge_fingerprint=knowledge[key[0]], candidate_release=loaded,
                                            candidate_contracts=manifest["implementation"]["role_digests"]["candidate"] if variant else None)
                        if variant:
                            record["saved_candidate_source"].update(release=source_release, rejudge_release=release,
                                                                    candidate_surface=deepcopy(surface))
                        proof = record.get("cache_source")
                        if record["saved_candidate_source"].get("recorded_failure"):
                            recorded_failure_count += 1
                            unattributed_failure_count += bool(record["saved_candidate_source"].get("unattributed_failure"))
                        else:
                            if any(proof[field] in producers[key][field] for field in producers[key]):
                                raise ValueError(f"saved candidates cannot repeat an immutable producer or transcript: {run} {key} slot {slot}")
                            for field in producers[key]:
                                producers[key][field].add(proof[field])
                        result[key].append(record)
                        if proof is not None:
                            payloads[proof["id"]] = proof["payload_sha256"]
        except (OSError, sqlite3.Error) as error:
            raise ValueError(f"saved candidates cannot read immutable cache {cache}: {error}") from error
        selected = [models[slot - 1] for slot in slots]
        selected_models.extend(selected)
        identities.append({"run": str(run), "cache_db": str(cache), "manifest_sha256": manifest_sha,
                           **({"release": source_release, "candidate_surface": deepcopy(surface)} if variant else {}),
                           "reservation_id": admission["reservation_id"], "claim_commit": admission["claim_commit"],
                           "inputs_sha256": inputs_sha, "results_sha256": result_hashes,
                           "slots": slots, "models": selected, "payload_count": len(payloads),
                           "recorded_failure_count": recorded_failure_count,
                           **({"unattributed_failure_count": unattributed_failure_count} if unattributed_failure_count else {}),
                           "payloads_sha256": "sha256:" + hashlib.sha256(json.dumps(payloads, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    groups = []
    for model in selected_models:
        name = f"{model['provider']}:{model['model']}"
        if groups and groups[-1][0] == name:
            groups[-1][1] += 1
        else:
            groups.append([name, 1])
    candidates = ",".join(f"{name}:{count}" for name, count in groups)
    identity = {"schema": VARIANT_SCHEMA if variant else SCHEMA, "release": release, "question_set": question_set,
                **({"candidate_surface": pin_surface, "candidate_runtime": common_runtime} if variant else {}),
                "knowledge_fingerprints": deepcopy(knowledge_fingerprints), "sources": identities}
    validate_identity(identity)
    return SavedPool(identity, candidates, deepcopy(fingerprints), deepcopy(knowledge_fingerprints), result)
