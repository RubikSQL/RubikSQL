"""Git-confirmed, one-use admission for a paid evolution execution.

The reservation charges a child and binds its one-use owner before transport
allocation. Interruption requires inspecting the saved execution, never automatic
relaunch. Claim identifiers are public provenance, not provider credentials.
"""
from __future__ import annotations

import json
import math
import os
import re
import socket
import subprocess
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from .canon import canon, sha256_hex
from .types import AuthorityError
from ..evaluation_questions import EVOLUTION_SET

ENV_LAUNCH = "RUBIKSQL_PAID_LAUNCH"
SCHEMA_RESERVATION = "rubiksql.paid-reservation/1"
SCHEMA_CLAIM = "rubiksql.paid-claim/1"
SCHEMA_RECEIPT = "rubiksql.paid-launch/1"
SCHEMA_BUILD_RESUME = "rubiksql.paid-build-resume/1"
SCHEMA_BUILD_CONTINUATION = "rubiksql.paid-build-continuation/1"
SCHEMA_ANSWER_CONTINUATION = "rubiksql.paid-answer-continuation/1"
SCHEMA_ANSWER_RESUME = "rubiksql.paid-answer-resume/1"
SCHEMA_CHILD_BUILD_CONTINUATION = "rubiksql.paid-child-build-continuation/1"
SCHEMA_CHILD_BUILD_RESUME = "rubiksql.paid-child-build-resume/1"
SCHEMA_CHILD_OPERATIONAL_CONTINUATION = "rubiksql.paid-child-build-operational-continuation/1"
SCHEMA_CHILD_OPERATIONAL_RESUME = "rubiksql.paid-child-build-operational-resume/1"
SCHEMA_CHILD_NATURAL_CONTINUATION = "rubiksql.paid-child-build-natural-continuation/1"
SCHEMA_CHILD_DIRECT_OPERATIONAL_CONTINUATION = "rubiksql.paid-child-build-direct-operational-continuation/1"
OWNER_FIELDS = ("owner_host", "owner_pid", "owner_started")
CHILD_CONTINUATION_EVIDENCE = ("original_request", "helper_stopped", "abandon_audit", "abandon_action",
                               "abandon_verified", "provider_assignment", "previous_receipt_file")
CHILD_OPERATIONAL_EVIDENCE = ("original_request", "previous_request", "helper_started", "helper_continued",
                              "helper_stopped", "pipeline_exit", "wrapper_terminal", "provider_assignment",
                              "previous_receipt_file")
CHILD_NATURAL_EVIDENCE = ("launch_context", "launch_wrapper", "launch_log", "pipeline_state", "pipeline_exit",
                          "stopped_verification", "process_inventory", "provider_assignment", "previous_receipt_file")
CHILD_DIRECT_OPERATIONAL_EVIDENCE = ("launch_context", *CHILD_OPERATIONAL_EVIDENCE[1:])
CHILD_KNOWLEDGE_STAGES = frozenset(("setup", "profiling", "table_profiles", "db_profile", "general_rules",
                                    "hint_rules", "synonyms", "examples"))
#: Pre-answering records an answering continuation retains, then re-prepares under its corrected source.
PRE_ANSWERING_RECORDS = (".prompt_manifest.json", ".prompt_manifest.lock", ".evaluation_inputs.json", ".live_preflight.json")
#: Seconds after its launch-context observation that natural child evidence stays fresh enough to admit.
_NATURAL_OBSERVATION_MAX_AGE = 300


def check_fresh_knowledge(root: Path, cache: Path | None = None) -> None:
    """Refuse previous generated knowledge or answering cache in a fresh admission."""
    root = Path(root)
    if root.is_symlink() or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
        raise AuthorityError("fresh knowledge requires a new empty physical root")
    if cache is not None and any(Path(str(cache) + suffix).exists() for suffix in ("", "-wal", "-shm")):
        raise AuthorityError("fresh knowledge requires an absent answering cache and sidecars")


def _checked_build_plan(knowledge: Mapping[str, Any], inputs: Mapping[str, Any], release: str, *, fresh: bool) -> None:
    """Check every stage/database decision, including summaries, rather than a vacuous empty plan."""
    from . import knowledge as K

    plan = knowledge.get("build_plan")
    dbs = set(inputs.get("db_fingerprints") or {})
    if (not isinstance(plan, Mapping) or plan.get("schema") != K.PLAN_SCHEMA or plan.get("release") != release
            or plan.get("root") != knowledge.get("root") or not dbs
            or set(plan.get("stages") or {}) != set(K.STAGE_NAMES)
            or set(plan.get("summary") or {}) != set(K.STAGE_NAMES)):
        raise AuthorityError("knowledge readiness requires a complete verified build plan")
    for stage in K.STAGE_NAMES:
        decisions = plan["stages"][stage]
        if not isinstance(decisions, Mapping) or set(decisions) != dbs:
            raise AuthorityError("knowledge build plan changed its database scope")
        summary: dict[str, list[str]] = {}
        for db, decision in decisions.items():
            expected = "run" if fresh else "skip"
            if (not isinstance(decision, Mapping) or decision.get("action") != expected
                    or (fresh and decision.get("reasons") != [K.NO_RECORD])):
                raise AuthorityError("fresh knowledge needs all first-build stages" if fresh else
                                     "answering requires a verified empty knowledge plan")
            summary.setdefault(expected, []).append(db)
        if plan["summary"][stage] != {action: sorted(names) for action, names in summary.items()}:
            raise AuthorityError("knowledge build plan summary differs from its decisions")


def _process(pid: int) -> tuple[int, str]:
    """Parent and start identity of one live local process."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessEntry(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                        ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                        ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        for name in ("Process32FirstW", "Process32NextW"):
            function = getattr(kernel, name)
            function.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
            function.restype = wintypes.BOOL
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        process = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            created, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetProcessTimes(process, ctypes.byref(created), ctypes.byref(exited),
                                          ctypes.byref(kernel_time), ctypes.byref(user_time)):
                raise ctypes.WinError(ctypes.get_last_error())
            snapshot = kernel.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
            if snapshot == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                entry = ProcessEntry()
                entry.dwSize = ctypes.sizeof(entry)
                found = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
                while found:
                    if entry.th32ProcessID == pid:
                        return int(entry.th32ParentProcessID), str((created.dwHighDateTime << 32) | created.dwLowDateTime)
                    found = kernel.Process32NextW(snapshot, ctypes.byref(entry))
                error = ctypes.get_last_error()
                if error != 18:  # ERROR_NO_MORE_FILES
                    raise ctypes.WinError(error)
                raise ProcessLookupError("paid launch owner is no longer a live process")
            finally:
                kernel.CloseHandle(snapshot)
        finally:
            kernel.CloseHandle(process)
    proc = Path(f"/proc/{pid}/stat")
    if proc.is_file():
        fields = proc.read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
        return int(fields[1]), fields[19]
    result = subprocess.run(["ps", "-o", "ppid=,lstart=", "-p", str(pid)], capture_output=True,
                            text=True, timeout=5, check=True).stdout.strip().split(maxsplit=1)
    if len(result) != 2:
        raise AuthorityError("paid launch owner is no longer a live process")
    return int(result[0]), result[1]


def launcher_identity() -> dict[str, Any]:
    """Capture the live owner for a local or cross-host batch admission.

    Args:
        None.

    Returns:
        Host, PID and process birth identity; the owner must stay alive while its
        admitted evaluator descendants run.
    """
    return {"owner_pid": os.getpid(), "owner_host": socket.gethostname(), "owner_started": _process(os.getpid())[1]}


def _owned_process(receipt: Mapping[str, Any]) -> bool:
    if receipt.get("owner_host") != socket.gethostname() or type(receipt.get("owner_pid")) is not int:
        return False
    pid = os.getpid()
    child_started = None
    try:
        for _ in range(128):
            parent, started = _process(pid)
            if os.name == "nt":
                # A retained parent PID must not identify a process created after its child.
                birth = int(started)
                if child_started is not None and birth > child_started:
                    return False
                child_started = birth
            if pid == receipt["owner_pid"]:
                return started == receipt.get("owner_started")
            if parent <= 0 or parent == pid:
                return False
            pid = parent
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    return False


def attempt_question_set(record: Mapping[str, Any]) -> str:
    """Read the content scope of immutable launch accounting.

    Args:
        record: A reservation or evaluation attachment.

    Returns:
        The question-set digest. Unscoped historical attachments belong to fixed 330.
    """
    from ..evaluation_questions import QUESTION_SETS

    return str(record.get("question_set") or (record.get("inputs") or {}).get("question_set")
               or QUESTION_SETS["330"].digest)


def counted_attempts(events: Iterable[Mapping[str, Any]], generation: str, *,
                     question_set: str = EVOLUTION_SET.digest) -> list[dict[str, Any]]:
    """Count reserved children and unmatched attachments once within one question set.

    Args:
        events: Complete immutable authority events.
        generation: Base release whose budget is requested.
        question_set: Content digest; the current evolution set by default. Historical
            verification supplies its recorded digest explicitly.

    Returns:
        Counted records in their original attempt order, excluding diagnostics.
    """
    rows = list(events)
    diagnostic_ids = {str(event["payload"]["id"]) for event in rows
                      if event["kind"] == "eval-reserve" and event["payload"].get("diagnostic") is True}
    reservations = {str(event["payload"]["id"]): dict(event["payload"]) for event in rows
                    if event["kind"] == "eval-reserve" and event["payload"].get("counted") is True
                    and event["payload"].get("generation") == generation
                    and attempt_question_set(event["payload"]) == question_set}
    attached = {str(event["payload"]["id"]): dict(event["payload"]) for event in rows
                if event["kind"] == "eval-attach" and event["payload"].get("generation") == generation
                and attempt_question_set(event["payload"]) == question_set
                and event["payload"].get("reservation_id") not in reservations
                and event["payload"].get("reservation_id") not in diagnostic_ids}
    return sorted([*reservations.values(), *attached.values()],
                  key=lambda row: (int(row.get("attempt", {}).get("used", 0)), float(row.get("created_at", 0)), str(row["id"])))


def require_evolution_regime(reader: Any) -> None:
    """Require the recorded complete baseline control before counted 500 work.

    Args:
        reader: Verified, consistent authority reader for this admission.

    Raises:
        AuthorityError: No outcome records the Mini-Dev 500 control boundary.
    """
    if not any(event["payload"].get("evolution_regime", {}).get("question_set") == EVOLUTION_SET.digest
               for event in reader.events("outcome")):
        raise AuthorityError("record the Mini-Dev 500 baseline control outcome before counted work")


def diagnostic_namespace(epoch_namespace: str, *, pin: str, out_dir: Path, candidates: str, judge: str,
                         saved_candidates: Mapping[str, Any] | None = None) -> str:
    """Derive a run-specific diagnostic cache namespace from its immutable configuration.

    Args:
        epoch_namespace: Existing cache namespace supplied by ``meta_ops env``.
        pin: Concrete release identity.
        out_dir: Owned diagnostic output directory.
        candidates: Explicit candidate specification.
        judge: Explicit judge specification.
        saved_candidates: Immutable source-pool identity for a saved-candidate rejudge.

    Returns:
        An isolated namespace stable only for this release, configuration and output.
    """
    identity = {"release": pin, "out_dir": str(Path(out_dir).resolve()), "candidates": candidates, "judge": judge}
    if saved_candidates is not None:
        identity["saved_candidates"] = dict(saved_candidates)
    return epoch_namespace + ":diagnostic:" + sha256_hex(canon(identity))[:16]


def evaluation_configuration(snapshot: Any) -> dict[str, str]:
    """The models a non-diagnostic run of a verified snapshot's pinned release uses.

    Args:
        snapshot: Verified execution snapshot (``snapshot.verify_snapshot``); its release loads from the
            snapshot's own store, with the runtime's full checks.

    Returns:
        dict[str, str]: ``{"candidates": "qwen:qwen3.7-plus:3", "judge": <the release's judge_model>}``; the
            reviewer follows the judge.

    Raises:
        AuthorityError: The pinned release does not load from the snapshot.
    """
    from .runtime import load_release
    from .store import PromptStoreError, Store

    try:
        release = load_release(snapshot.release, stores=Store(package=snapshot.path))
    except PromptStoreError as error:
        raise AuthorityError(f"paid admission cannot read the pinned release's judge_model: {error}") from error
    return {"candidates": "qwen:qwen3.7-plus:3", "judge": release.knob_settings()["judge_model"]}


def verify_evaluation_configuration(receipt: Mapping[str, Any], *, diagnostic: bool, candidates: str, judge: str,
                                    saved_candidates: Mapping[str, Any] | None = None) -> None:
    """Bind a worker's mode, models and saved knowledge to its paid admission.

    Historical receipts without an explicit counted flag retain the fixed
    evolution models locally; this check never opens the authority. A counted
    receipt's committed configuration names the judge its admission verified
    against the pinned release's ``judge_model`` (:func:`evaluation_configuration`).

    Args:
        receipt: Verified committed launch receipt, or the existing empty offline receipt.
        diagnostic: Whether the worker was explicitly launched as a diagnostic.
        candidates: Actual ordered candidate specification selected by the worker; slot order is identity.
        judge: Actual judge specification selected by the worker.
        saved_candidates: Exact saved source-pool identity, absent for candidate generation.

    Returns:
        None: The selected configuration satisfies the receipt's local policy.

    Raises:
        AuthorityError: A diagnostic lacks its exact receipt, changes models, or is
            presented as a normal run; saved candidates used different knowledge;
            a counted child changes its fixed candidates or its admitted judge.
    """
    from rubiksql_hb.judge_policy import JUDGE, JUDGES, require_diagnostic_configuration

    configuration: dict[str, Any] = {"candidates": candidates, "judge": judge}
    if saved_candidates is not None:
        if (not diagnostic or not isinstance(receipt, Mapping)
                or not isinstance(saved_candidates, Mapping) or not saved_candidates):
            raise AuthorityError("saved candidates require a diagnostic source-pool identity")
        from rubiksql_hb.saved_candidates import validate_identity

        try:
            validate_identity(dict(saved_candidates))
        except ValueError as error:
            raise AuthorityError(str(error)) from None
        inputs = receipt.get("inputs")
        if (saved_candidates["release"] != receipt.get("release") or not isinstance(inputs, Mapping)
                or saved_candidates["question_set"] != inputs.get("question_set")):
            raise AuthorityError("saved candidate admission changed its source release or question set")
        knowledge = receipt.get("knowledge")
        if (not isinstance(knowledge, Mapping)
                or saved_candidates["knowledge_fingerprints"] != knowledge.get("fingerprints")):
            raise AuthorityError("saved candidate knowledge fingerprints differ from the admitted knowledge")
        configuration["saved_candidates"] = dict(saved_candidates)
    if diagnostic:
        if not isinstance(candidates, str) or not isinstance(judge, str):
            raise AuthorityError("diagnostic model specifications must be strings")
        try:
            require_diagnostic_configuration(candidates, judge, saved_candidates=saved_candidates is not None)
        except ValueError as error:
            raise AuthorityError(str(error)) from None
        if saved_candidates is not None:
            models = [model for source in saved_candidates["sources"] for model in source["models"]]
            entries = [entry.split(":") for entry in candidates.split(",")]
            if sum(int(count) for _, _, count in entries) != len(models):
                raise AuthorityError("saved candidate models differ from the diagnostic candidate specification")
            selected = [(provider, model) for provider, model, count in entries for _ in range(int(count))]
            if selected != [(model["provider"], model["model"]) for model in models]:
                raise AuthorityError("saved candidate models differ from the diagnostic candidate specification")
        if (not isinstance(receipt, Mapping) or receipt.get("diagnostic") is not True
                or receipt.get("configuration") != configuration
                or receipt.get("counted") is not False):
            raise AuthorityError("diagnostic worker differs from its committed mode or model configuration")
    elif receipt.get("diagnostic") is True:
        raise AuthorityError("a diagnostic launch cannot be used by a non-diagnostic worker")
    else:
        if receipt.get("configuration") is not None and receipt["configuration"] != configuration:
            raise AuthorityError("worker model configuration differs from its committed paid admission")
        # Historical receipts predate the explicit counted/configuration fields.
        # They used the fixed production models; keep that conservative local
        # policy without reopening the authority in each descendant process.
        # A newer receipt's configuration names the judge its admission checked
        # against the pinned release's judge_model.
        counted = receipt.get("counted", bool(receipt))
        admitted = (receipt.get("configuration") or {}).get("judge", JUDGE)
        if counted is True and (candidates != "qwen:qwen3.7-plus:3" or judge not in JUDGES or judge != admitted):
            raise AuthorityError("counted evolution requires qwen:qwen3.7-plus:3 candidates and the judge its "
                                 "admission verified against the pinned release's judge_model")


def _commit(writer: Any) -> str:
    result = writer.commit()
    commit = result.commit
    if not isinstance(commit, str) or len(commit) != 40:
        raise AuthorityError("paid admission has no remotely confirmed authority commit")
    return commit


def _save_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(canon(receipt) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _admission_context(reader: Any, pin: str, release: str, *, diagnostic: bool = False) -> dict[str, Any]:
    """Capture plan inputs that authority accounting must never retarget."""
    from .entity_model import PARAMETER_ENTITY, RULE_ENTITY, TOOL_ENTITY

    head = reader.head()
    selected = reader.row("release-entity", release)
    prompts = {(row["name"], row["manifest"]["meta"]["content_key"]): row
               for row in reader.rows("sys-prompt")}
    rows = [prompts[(name, key)] for name, key in selected["refs"].items()]
    content = selected.get("content_refs")
    if content:
        rows.extend(reader.row(entity, ref["object_id"]) for entity, ref in [
            *((TOOL_ENTITY, ref) for ref in content["tools"]),
            *((RULE_ENTITY, ref) for ref in content["rules"]["items"]),
            (PARAMETER_ENTITY, content["parameters"])])
        rows.extend(reader.row("sys-capsule", ref["capsule_id"])
                    for ref in [*content["loader"]["modules"], content["loader"]["serializer"]])
    # Include the complete admitting gate chain, including a legacy migration
    # base, rather than accepting a new receipt for the same executable bundle.
    from rubiksql_hb.hb_tool_entities import digest, typed_bundle

    bundles = ({digest(typed_bundle(content))} if content else
               {sha256_hex(canon(reader.answering(release)))} if "tools.answering" in selected["refs"] else set())
    gates: dict[str, Any] = {}
    while True:
        found = [event for event in reader.events("gate")
                 if (event["payload"].get("bundle_digest") in bundles
                     or event["subject"] in {pin, release} or event["payload"].get("release") == release)
                 and event["object_id"] not in gates]
        if not found:
            break
        for event in found:
            gates[event["object_id"]] = event
            bundles.update(value for key, value in event["payload"].items()
                           if key in {"legacy_bundle_digest", "parent_bundle_digest"} and isinstance(value, str))
    targets = {pin, release, *(f"{name}:{key[:16]}" for name, key in selected["refs"].items()),
               *(row["object_id"] for row in reader.rows("proposal-entity") if row.get("release") == release)}
    decisions = {event["object_id"]: event for event in reader.events("decision")
                 if event["payload"].get("kind") in {"retire", "unretire", "revoke", "unrevoke", "reject", "abandon"}
                 and (event["payload"].get("target") in targets
                      or event["payload"].get("content_key") in selected["refs"].values())}
    return {"head": None if diagnostic else (head.release, head.generation, head.decision), "epoch": reader.epoch(),
            "release": reader.resolve(pin), "content": selected, "rows": rows,
            "tools": {name: raw for name, raw in reader.store_files(tools_release=release).items()
                      if name.startswith("tools/")},
            "admission": reader.source_admission(release), "gates": gates, "decisions": decisions,
            "proposal": {"body": reader.proposal(pin), "row": reader.row("proposal-entity", pin)} if pin.startswith("p") else None,
            "tiers": reader.tool_tiers()}


def _check_admission_context(writer: Any, *, commit: str, context: Mapping[str, Any],
                             events: tuple[Mapping[str, Any], ...], pin: str, release: str,
                             diagnostic: bool = False) -> None:
    """Prove the plan's history and admission inputs survive unrelated appends."""
    from .git_authority import _git

    projection = writer._git_projection
    if _git(projection.export, "merge-base", "--is-ancestor", commit,
            writer.head().authority_commit, check=False).returncode:
        raise AuthorityError("paid plan belongs to a stale authority commit; check it again")
    with writer.reader() as reader:
        current = reader.events()
        # Event timestamps are host clocks, not append order. Verify the entire
        # old set survives byte-for-byte even when a new event sorts before it.
        before = {event["object_id"]: event for event in events}
        after = {event["object_id"]: event for event in current}
        unrelated = {"eval-reserve", "eval-claim", "eval-ready", "eval-attach", "outcome", "publish", "gate",
                     "workspace-change", "register", "decision"}
        if (any(after.get(key) != event for key, event in before.items())
                or any(event["kind"] not in unrelated for key, event in after.items() if key not in before)
                or _admission_context(reader, pin, release, diagnostic=diagnostic) != context):
            raise AuthorityError("paid plan belongs to a stale authority commit; check it again")


def generation_budget(reader: Any, requested: int | None = None) -> int:
    """Resolve the generation's recorded cap, with an explicit extension up to twenty.

    A later arm cannot silently lower an already recorded cap. The default remains
    ten; an extension is recorded in each reservation and its outcome rule.

    Args:
        reader: Consistent authority view for the current generation.
        requested: Optional explicit limit from ten through twenty.

    Returns:
        The default or largest recorded/requested bounded limit.

    Raises:
        AuthorityError: A requested or recorded cap is invalid.
    """
    from . import evolve as EV

    if requested is not None and (type(requested) is not int or not 10 <= requested <= 20):
        raise AuthorityError("an explicit generation budget_cap must be an integer from 10 to 20")
    caps = [EV.MAX_ATTEMPTS_PER_GENERATION]
    generation = reader.head().generation
    for event in reader.events("eval-reserve"):
        row = event["payload"]
        if row.get("generation") == generation and row.get("counted") is True:
            cap = row.get("budget_cap", EV.MAX_ATTEMPTS_PER_GENERATION)
            if type(cap) is not int or not 0 <= cap <= 20:
                raise AuthorityError("recorded generation budget cap is invalid")
            caps.append(cap)
    return max(*caps, requested if requested is not None else EV.MAX_ATTEMPTS_PER_GENERATION)


def _prepare_admission(*, pin: str, out_dir: Path, snapshot: Any, inputs: Mapping[str, Any],
                      knowledge: Mapping[str, Any], source_commit: str, namespace: str,
                      parent_run: Path | None, environment: Mapping[str, str], requires_ready: bool = False,
                      fresh_knowledge: bool = False, diagnostic: bool = False,
                      configuration: Mapping[str, Any] | None = None,
                      owner: Mapping[str, Any] | None = None, delivery_root: Path | None = None,
                      budget_cap: int | None = None, binding: Any = None) -> Callable[[Any], PreparedAdmission]:
    """Stage a checked reservation and its owner in the caller's transaction.

    No receipt or execution is exposed before the caller confirms the complete
    batch with ``writer.commit()`` and calls ``PreparedAdmission.confirm``.

    Args:
        pin: Exact published proposal or current baseline release.
        out_dir: Owned output directory; its name is the canonical run identity.
        snapshot: Verified source snapshot whose commit supplied the checked plan.
        inputs: Allowlisted diagnostic or Mini-Dev 500 question and SQLite fingerprints.
        knowledge: Per-database knowledge fingerprints and the checked build plan.
        source_commit: Committed execution source revision.
        namespace: Preserved epoch cache namespace.
        parent_run: Complete paired parent for a child; absent for a baseline.
        environment: Private provider configuration. Credentials are never persisted.
        diagnostic: Uncounted descriptive run of a concrete release, with its own namespace.
        configuration: Diagnostic candidate and judge specifications, plus an optional
            immutable saved-candidate pool identity. Non-diagnostic runs accept only
            :func:`evaluation_configuration` of the snapshot (the fixed candidates and
            the pinned release's ``judge_model``), which an absent value defaults to;
            the reservation and receipt record it.
        budget_cap: Explicit generation cap extension, from ten through twenty.
        binding: Caller-owned writer identity, otherwise the selected package binding.

    Returns:
        Admission to confirm after the entire batch is pushed.

    Raises:
        AuthorityError: Stale authority, invalid source, exhausted quota, duplicate
            launch, missing private checkpoints, or unconfirmed push.
    """
    from rubiksql_hb.experiment_gateway import GatewayPool
    from .authority import Authority
    from .binding import load_binding
    from . import evolve as EV
    from .types import GitAuthorityBinding

    binding = binding if binding is not None else load_binding()
    if not isinstance(binding, GitAuthorityBinding):
        raise AuthorityError("paid evolution requires the Git primary authority binding")
    pool = GatewayPool.from_env(environment)
    status = pool.status()
    if status.get("profile") != "volcengine" or not status.get("initialized"):
        raise AuthorityError("paid admission requires the approved pool and its existing private allocation checkpoint")
    databases = inputs.get("db_fingerprints")
    if not isinstance(databases, Mapping) or not 1 <= len(databases) <= 11:
        raise AuthorityError("paid admission requires a distinct scope of one to eleven databases")
    available_chat = {row["index"] for row in status.get("accounts", ()) if row["status"] == "available"}
    if len(available_chat) < len(databases):
        raise AuthorityError("paid admission requires at least one available chat account per database")
    paired = status.get("paired", {})
    available_embedding = {row["index"] for row in paired.get("embedding_accounts", ())
                           if row["status"] == "available"}
    if paired.get("configured") is not True or not available_embedding:
        raise AuthorityError("paid admission requires the embedding-only key pool with at least one available account")
    from rubiksql_hb.evaluation_questions import QUESTION_SETS

    allowed = {item.digest for item in QUESTION_SETS.values()} if diagnostic else {EV.EVOLUTION_QUESTION_SET}
    if inputs.get("question_set") not in allowed or not snapshot.authority_commit:
        scope = "an allowlisted diagnostic source" if diagnostic else "the exact Mini-Dev 500 source"
        raise AuthorityError(f"paid admission requires {scope} and authority commit")
    if len(source_commit) != 40 or not namespace:
        raise AuthorityError("paid admission requires committed code and the existing epoch namespace")
    if diagnostic:
        if (re.fullmatch(r"r[0-9a-f]{16}", pin) is None or pin != snapshot.release
                or parent_run is not None or fresh_knowledge or requires_ready
                or environment.get("RUBIKSQL_EVALUATION_MODE") != "diagnostic"
                or environment.get("RUBIKSQL_CACHE_NS") != namespace
                or environment.get("RUBIKSQL_GATEWAY_EXPERIMENT_ID") != "rubiksql:diagnostic:" + namespace
                or not isinstance(configuration, Mapping)
                or set(configuration) not in ({"candidates", "judge"}, {"candidates", "judge", "saved_candidates"})):
            raise AuthorityError("diagnostic admission requires a concrete release, explicit models and ready knowledge without a parent")
        verify_evaluation_configuration({"diagnostic": True, "counted": False, "configuration": dict(configuration),
                                         "release": snapshot.release, "inputs": inputs, "knowledge": knowledge},
                                        diagnostic=True, candidates=configuration["candidates"], judge=configuration["judge"],
                                        saved_candidates=configuration.get("saved_candidates"))
        _checked_build_plan(knowledge, inputs, snapshot.release, fresh=False)
        fingerprints = knowledge.get("fingerprints")
        if (not isinstance(fingerprints, Mapping) or set(fingerprints) != set(databases)
                or any(not isinstance(value, str) or not value for value in fingerprints.values())):
            raise AuthorityError("diagnostic admission requires every ready knowledge fingerprint")
    else:
        if environment.get("RUBIKSQL_EVALUATION_MODE") == "diagnostic":
            raise AuthorityError("diagnostic environment requires an explicit diagnostic admission")
        fixed = evaluation_configuration(snapshot)
        configuration = dict(configuration) if configuration is not None else fixed
        if configuration != fixed:
            raise AuthorityError("non-diagnostic paid evolution requires the fixed candidates and the pinned "
                                 f"release's judge_model {fixed['judge']}")
    if fresh_knowledge:
        if len(databases) != 11 or parent_run is not None or not requires_ready or knowledge.get("fingerprints") != {}:
            raise AuthorityError("fresh knowledge is an uncounted baseline build only")
        cache = environment.get("RUBIKSQL_CACHE_DB")
        if not cache:
            raise AuthorityError("fresh knowledge requires an explicit isolated answering cache")
        check_fresh_knowledge(Path(knowledge["root"]), Path(cache))
        _checked_build_plan(knowledge, inputs, snapshot.release, fresh=True)
    output = Path(out_dir).resolve()
    if diagnostic and (Path(out_dir).is_symlink() or (output.exists() and (
            not output.is_dir() or any(path.name != ".prompt_snapshot" for path in output.iterdir())))):
        raise AuthorityError("diagnostic output already has a paid reservation or previous execution artifacts; use a fresh output directory")
    if fresh_knowledge and output.exists() and any(path.name != ".prompt_snapshot" for path in output.iterdir()):
        raise AuthorityError("fresh build output contains previous execution artifacts")
    parent = EV.read_run(parent_run) if parent_run is not None else None
    parent_identity = ({"run": parent.name, "release": parent.release, "namespace": parent.ns,
                        "created_at": parent.created_at, "manifest_sha256": parent.manifest_sha,
                        "result_files": parent.result_files} if parent else None)
    identity = {"pin": pin, "release": snapshot.release, "run": output.name, "out_dir": str(output),
                "namespace": namespace, "source_commit": source_commit,
                "source_authority_commit": snapshot.authority_commit, "snapshot_digest": snapshot.digest,
                "inputs": dict(inputs), "knowledge": dict(knowledge), "parent": parent_identity,
                "requires_ready": requires_ready, "configuration": dict(configuration)}
    if fresh_knowledge:
        identity["fresh_knowledge"] = True
    if diagnostic:
        identity["diagnostic"] = True
    reservation_id = "o" + sha256_hex(canon(identity))[:16]
    owner = dict(owner) if owner is not None else launcher_identity()
    if (set(owner) != set(OWNER_FIELDS) or type(owner["owner_pid"]) is not int or owner["owner_pid"] < 1
            or any(not isinstance(owner[key], str) or not owner[key] for key in ("owner_host", "owner_started"))
            or (owner["owner_host"] != socket.gethostname() and delivery_root is None)):
        raise AuthorityError("batch admission needs an exact launcher identity and a delivery root for a foreign host")
    admission: dict[str, Any] = {}

    def admit(reader: Any) -> None:
        # Captured once from a verified view of exactly the snapshot commit; every attempt compares with it.
        context = _admission_context(reader, pin, snapshot.release, diagnostic=diagnostic)
        if context["release"] != snapshot.release:
            raise AuthorityError("paid plan snapshot differs from its pinned authority release")
        admission.update(context=context, events=reader.events())

    def read_admission() -> None:
        with Authority.read(binding=binding, commit=snapshot.authority_commit) as reader:
            admit(reader)

    def check_context(writer: Any) -> None:
        _check_admission_context(writer, commit=snapshot.authority_commit, context=admission["context"],
                                 events=admission["events"], pin=pin, release=snapshot.release, diagnostic=diagnostic)
        if parent_run is not None:
            current_parent = EV.read_run(parent_run)
            if ({"run": current_parent.name, "release": current_parent.release, "namespace": current_parent.ns,
                 "created_at": current_parent.created_at, "manifest_sha256": current_parent.manifest_sha,
                 "result_files": current_parent.result_files} != parent_identity):
                raise AuthorityError("paid admission paired parent changed after its plan was checked")
            with writer.reader() as reader:
                plan = EV.plan_view(reader.proposal(pin))
            if time.time() > current_parent.created_at + plan.max_age_days * 86400:
                raise AuthorityError("paid admission paired parent expired before its claim")

    def reserve(writer: Any) -> dict[str, Any] | None:
        if not admission:
            if writer.head().authority_commit != snapshot.authority_commit:
                return None  # The head moved after the remote check: admit from the exact snapshot commit first.
            # This writer's projection is verified exactly at the snapshot commit, as that read's would be.
            with writer.reader() as reader:
                admit(reader)
        check_context(writer)
        head = writer.head()
        with writer.reader() as reader:
            adopted = not diagnostic and pin == head.release and parent is None
            if fresh_knowledge and not adopted:
                raise AuthorityError("fresh knowledge requires the current baseline without a paired parent")
            if diagnostic:
                epoch_namespace = "v0-" + str(reader.epoch()["release"])
                expected = diagnostic_namespace(epoch_namespace, pin=pin, out_dir=output,
                                                candidates=configuration["candidates"], judge=configuration["judge"],
                                                saved_candidates=configuration.get("saved_candidates"))
                if namespace != expected or reader.resolve(pin) != snapshot.release:
                    raise AuthorityError("diagnostic admission changed its release or isolated epoch cache namespace")
            elif adopted:
                if parent is not None:
                    raise AuthorityError("baseline admission accepts no paired parent")
                epoch = reader.epoch()
                if namespace != "v0-" + str(epoch["release"]):
                    raise AuthorityError("baseline admission changed the epoch namespace")
            else:
                proposal = reader.proposal(pin)
                plan = EV.plan_view(proposal)
                if proposal["release"] != snapshot.release or plan.base_parent != head.generation:
                    raise AuthorityError("child admission does not match the current published generation")
                if (parent is None or parent.release != head.generation or parent.ns != namespace
                        or plan.namespace != namespace or time.time() > parent.created_at + plan.max_age_days * 86400):
                    raise AuthorityError("child admission requires its complete current paired parent within the pairing window")
                parent_problem = parent.release_problem(store=None)
                if parent_problem:
                    raise AuthorityError("paired parent source refused: " + parent_problem)
                try:
                    EV.require_evolution_run(parent)
                except EV.EvolveError as error:
                    raise AuthorityError(str(error)) from error
                require_evolution_regime(reader)
            events = reader.events()
            for event in reader.events("eval-reserve"):
                held = event["payload"]
                if held.get("id") == reservation_id:
                    raise AuthorityError("this execution was already reserved; inspect its durable claim before recovery")
                if diagnostic:
                    if held.get("out_dir") == str(output) or held.get("namespace") == namespace or held.get("run") == output.name:
                        raise AuthorityError("this diagnostic execution already has a paid reservation; follow that execution")
                    continue
                duplicate = (adopted and held.get("counted") is False and held.get("diagnostic") is not True
                             and held.get("epoch") == reader.epoch()["epoch"]
                             and attempt_question_set(held) == inputs.get("question_set"))
                if duplicate and held.get("release") == snapshot.release:
                    raise AuthorityError("this release already has a paid reservation in the epoch; follow that execution")
            if not adopted and not diagnostic and EV.release_measurement_blocker(snapshot.release, head.generation,
                    attachments=(event["payload"] for event in reader.events("eval-attach")),
                    outcomes=(event["payload"] for event in reader.events("outcome")),
                    reservations=(event["payload"] for event in reader.events("eval-reserve"))) is not None:
                raise AuthorityError("this release already has a paid reservation in the epoch; follow that execution")
            counted = counted_attempts(events, head.generation)
            cap = generation_budget(reader, budget_cap)
            if not adopted and not diagnostic and len(counted) >= cap:
                raise AuthorityError("generation has exhausted its paid child evaluation budget")
            previous_ids = sorted(str(row["id"]) for row in counted)
            token = uuid.uuid4().hex
            reservation = {"schema": SCHEMA_RESERVATION, "id": reservation_id, **identity,
                           "epoch": reader.epoch()["epoch"], "generation": head.generation,
                           "proposal": None if adopted or diagnostic else pin, "counted": not adopted and not diagnostic, "token": token,
                           "attempt": {"used": len(previous_ids), "counted": previous_ids, "allow_over_cap": False},
                           "created_at": time.time(), "admission": "reserved-owner/1", "budget_cap": cap, **owner}
        writer._require_writer()
        writer._event("eval-reserve", pin, reservation, "agent:paid-launch")
        return reservation

    def stage(writer: Any) -> PreparedAdmission:
        if writer.head().authority_commit != snapshot.authority_commit and not admission:
            read_admission()
        reservation = reserve(writer)
        if reservation is None:
            read_admission()
            reservation = reserve(writer)
        if reservation is None:
            raise AuthorityError("paid admission could not bind its original snapshot")
        return PreparedAdmission(pin, output, reservation, delivery_root, parent_run)

    return stage


def stage_admission(writer: Any, **kwargs: Any) -> PreparedAdmission:
    """Stage one checked, owner-bound reservation without a commit or receipt.

    Args:
        writer: Active authority generation writer.
        **kwargs: Exact execution inputs documented by :func:`_prepare_admission`.

    Returns:
        Reservation that the caller confirms only after its batch push succeeds.

    Raises:
        AuthorityError: Configuration, source, pairing, cap or duplicate check fails.
    """
    return _prepare_admission(binding=writer.git_binding, **kwargs)(writer)


@dataclass(frozen=True)
class PreparedAdmission:
    """Checked domain reservation; confirmation is the only execution boundary."""

    pin: str
    output: Path
    reservation: Mapping[str, Any]
    delivery_root: Path | None = None
    parent_run: Path | None = None

    def confirm(self, writer: Any, commit: str, *, activate: bool = True,
                reader: Any = None) -> tuple[dict[str, Any], Any]:
        """Expose a receipt only for this writer's successfully pushed commit.

        Args:
            writer: The writer that staged the admission and confirmed the push.
            commit: Exact SHA returned by that writer's commit operation.
            activate: Consume the local execution lock now, or let the evaluator do so.
            reader: Optional committed view owned by this exact active writer,
                shared while confirming all receipts in one batch.

        Returns:
            Locally saved one-use receipt and its committed execution snapshot.

        Raises:
            AuthorityError: Commit, reservation, owner or the permanent host lock differs.
        """
        from .snapshot import write_snapshot

        held = self.reservation
        with (nullcontext(reader) if reader is not None else writer.reader()) as current:
            if (not current.writer_lock_held or current._git_projection is not writer._git_projection
                    or current.git_binding != writer.git_binding or current.authority_commit != commit):
                raise AuthorityError("admission confirmation requires this writer's exact committed view")
            head = current.head()
            if head.authority_commit != commit or not head.exported:
                raise AuthorityError("admission requires the exact pushed batch commit")
            if [event["payload"] for event in current.events("eval-reserve")
                    if event["payload"].get("id") == held["id"]] != [held]:
                raise AuthorityError("admission changed after its reservation")
            if self.parent_run is not None:
                from . import evolve as EV

                parent = EV.read_run(self.parent_run)
                current_parent = {"run": parent.name, "release": parent.release, "namespace": parent.ns,
                                  "created_at": parent.created_at, "manifest_sha256": parent.manifest_sha,
                                  "result_files": parent.result_files}
                if current_parent != held["parent"]:
                    raise AuthorityError("paid admission paired parent changed before execution")
            snapshot = write_snapshot(current, self.pin, out_dir=self.delivery_root or self.output)
        receipt = {"schema": SCHEMA_RECEIPT, "reservation_id": held["id"], "reservation_commit": commit,
                   "admission": "reserved-owner/1",
                   "claim_commit": commit, "token": held["token"], "phase": "claimed",
                   "snapshot_digest": snapshot.digest,
                   **{key: held[key] for key in ("run", "out_dir", "release", "source_commit", "namespace", "inputs",
                       "requires_ready", "counted", "configuration", "parent", *OWNER_FIELDS)},
                   "knowledge": {key: held["knowledge"][key] for key in ("root", "fingerprints")}}
        for key in ("fresh_knowledge", "diagnostic"):
            if held.get(key):
                receipt[key] = True
        try:
            _save_receipt((self.delivery_root or self.output) / ".paid-launch.json", receipt)
        except OSError as error:
            error.add_note(f"Reservation {commit} is committed. Recover by reading the authority at that commit; "
                           "never relaunch as a new reservation.")
            raise
        # commit() already verified the entire export and confirmed the remote
        # fast-forward. Retain those exact facts; do not open another projection.
        _CONFIRMED_ADMISSIONS[(writer.git_binding, canon(receipt))] = dict(held)
        if held["owner_host"] == socket.gethostname():
            from .git_authority import _private_open

            confirmation = _execution_lock(writer.git_binding, receipt).with_suffix(".confirmed.json")
            with open(confirmation, "x", encoding="utf-8", opener=_private_open) as stream:
                stream.write(canon({"receipt": receipt, "reservation": dict(held)}) + "\n")
        if activate:
            activate_admission(receipt, snapshot=snapshot)
        return receipt, snapshot


def reserve_and_claim(**kwargs: Any) -> tuple[dict[str, Any], Any]:
    """Admit one run in one transaction; batch callers use ``stage_admission``.

    Args:
        **kwargs: The keyword arguments documented by :func:`stage_admission`,
            excluding its caller-owned writer.

    Returns:
        Pushed admission receipt and execution snapshot.

    Raises:
        AuthorityError: Any admission, quota, duplicate or publication check fails.
    """
    from .authority import Authority
    from .binding import load_binding

    stage = _prepare_admission(**kwargs)

    def admit(writer: Any) -> tuple[dict[str, Any], Any]:
        prepared = stage(writer)
        return prepared.confirm(writer, _commit(writer))

    return Authority.transact(admit, binding=load_binding())


_CONFIRMED_ADMISSIONS: dict[tuple[Any, str], dict[str, Any]] = {}


def _execution_lock(binding: Any, receipt: Mapping[str, Any]) -> Path:
    from .git_authority import _account_home, _check_cache_path, _check_cache_permissions

    locks = _account_home() / ".rhb" / "launch-locks"
    _check_cache_path(locks)
    locks.mkdir(mode=0o700, parents=True, exist_ok=True)
    _check_cache_permissions(locks)
    return locks / (binding.authority_uuid + "-" + receipt["reservation_id"] + ".json")


def activate_admission(receipt: Mapping[str, Any], *, snapshot: Any) -> None:
    """Consume a pushed batch admission once on its recorded launcher host.

    Args:
        receipt: Delivered receipt, retaining the pre-recorded live owner identity.
        snapshot: Delivered, byte-verified snapshot of exactly the pushed commit.

    Returns:
        None: The permanent host lock admits this root's descendants.

    Raises:
        AuthorityError: Another execution consumed it, or owner/commit/content differs.
    """
    from .binding import load_binding
    from .git_authority import _private_open

    binding = load_binding()
    if (not _owned_process(receipt) or snapshot.authority_commit != receipt.get("claim_commit")
            or snapshot.digest != receipt.get("snapshot_digest") or snapshot.release != receipt.get("release")):
        raise AuthorityError("batch activation differs from its pushed commit, snapshot or live launcher")
    _verify_committed_claim(binding, canon(dict(receipt)))
    lock = _execution_lock(binding, receipt)
    try:
        with open(lock, "x", encoding="utf-8", opener=_private_open) as stream:
            stream.write(canon(dict(receipt)) + "\n")
    except FileExistsError as error:
        raise AuthorityError("paid reservation already has a permanent host execution lock") from error


def local_confirmation(binding: Any, receipt: Mapping[str, Any]) -> dict[str, Any] | None:
    """Read the private push confirmation handed to a local batch descendant.

    Args:
        binding: Selected authority identity.
        receipt: Exact receipt exposed after the coordinator's confirmed push.

    Returns:
        Committed reservation, or None for a foreign/legacy handoff. A present
        mismatching confirmation refuses instead of falling back to remote reads.

    Raises:
        AuthorityError: The private confirmation was redirected or altered.
    """
    if not isinstance(receipt, Mapping):
        raise AuthorityError("paid launch receipt must be an object")
    if receipt.get("admission") != "reserved-owner/1" or receipt.get("owner_host") != socket.gethostname():
        return None
    from .git_authority import _check_cache_path, _check_cache_permissions

    path = _execution_lock(binding, receipt).with_suffix(".confirmed.json")
    _check_cache_path(path)
    if not path.exists():
        return None
    _check_cache_permissions(path)
    proof = json.loads(path.read_bytes())
    if proof.get("receipt") != dict(receipt):
        # Readiness and authorized continuations have their own committed checks.
        if receipt.get("phase") == "ready" or receipt.get("resume_commit") or receipt.get("answer_resume_commit"):
            return None
        raise AuthorityError("paid receipt differs from its private pushed-commit confirmation")
    return dict(proof["reservation"])


def reservation_claims(reader: Any, reservation: Mapping[str, Any], commit: str | None = None) -> list[Mapping[str, Any]]:
    """Read legacy claims or the owner embedded in a new one-use reservation."""
    claims = [event["payload"] for event in reader.events("eval-claim")
              if event["payload"].get("reservation_id") == reservation.get("id")]
    if reservation.get("admission") == "reserved-owner/1":
        if claims or any(not reservation.get(key) for key in OWNER_FIELDS):
            raise AuthorityError("fused reservation has conflicting or incomplete ownership")
        return [{**reservation, "reservation_id": reservation["id"], "reservation_commit": commit}]
    return claims


def verify_claim(*, environment: Mapping[str, str] | None = None, out_dir: Path | None = None,
                 snapshot: Any = None, answering: bool = False) -> dict[str, Any]:
    """Verify owner provenance once; admit its descendants against the local receipt.

    Owners save each committed claim, readiness or continuation receipt to
    ``.paid-launch.json`` before verification so a failed read retains recovery
    evidence. They return it for inheritance only after verification succeeds.
    A live descendant must inherit that exact receipt and pass the same local
    execution checks without an authority read. Repeated owner checks reuse the
    immutable committed verification.

    Args:
        environment: Inherited launch environment; defaults to this process's.
        out_dir: Expected run directory, when the caller owns that argument.
        snapshot: Verified local prompt snapshot, when supplied by the caller.
        answering: Require final knowledge readiness before answering.

    Returns:
        dict[str, Any]: The checked receipt, or an empty mapping for an offline
            authority binding.

    Raises:
        AuthorityError: The receipt, owner ancestry or local execution identity
            differs; an owner's remote committed verification fails.
    """
    from .binding import load_binding
    from .types import GitAuthorityBinding

    env = os.environ if environment is None else environment
    binding = load_binding()
    if not isinstance(binding, GitAuthorityBinding):
        return {}
    try:
        receipt = json.loads(env.get(ENV_LAUNCH, ""))
    except (TypeError, ValueError) as error:
        raise AuthorityError("paid worker has no inherited committed launch receipt") from error
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA_RECEIPT or receipt.get("phase") not in {"claimed", "ready"}:
        raise AuthorityError("paid worker has no claimed launch receipt")
    if "counted" in receipt and type(receipt["counted"]) is not bool:
        raise AuthorityError("paid worker changed its counted admission identity")
    if answering and receipt.get("requires_ready") and receipt.get("phase") != "ready":
        raise AuthorityError("answering requires the committed final knowledge readiness receipt")
    if not _owned_process(receipt):
        raise AuthorityError("paid worker is not descended from its live one-use launch owner")
    if receipt.get("admission") == "reserved-owner/1":
        try:
            lock = json.loads(_execution_lock(binding, receipt).read_bytes())
        except (OSError, ValueError) as error:
            raise AuthorityError("paid worker has no permanent host execution lock") from error
        if any(lock.get(key) != receipt.get(key) for key in ("reservation_id", "reservation_commit", "token")):
            raise AuthorityError("paid worker changed its permanent host execution lock")
    if env.get("RUBIKSQL_CACHE_NS") != receipt.get("namespace"):
        raise AuthorityError("paid worker changed the reserved epoch cache namespace")
    if (env.get("RUBIKSQL_EVALUATION_MODE") == "diagnostic") != (receipt.get("diagnostic") is True):
        raise AuthorityError("paid worker changed the committed diagnostic mode")
    commits = ("reservation_commit", "claim_commit") + (("execution_commit",) if receipt.get("phase") == "ready" else ())
    if receipt.get("resume_commit") is not None:
        commits += ("resume_commit",)
    if receipt.get("answer_resume_commit") is not None:
        commits += ("answer_resume_commit",)
    if any(not isinstance(receipt.get(key), str) or re.fullmatch(r"[0-9a-f]{40}", receipt[key]) is None for key in commits):
        raise AuthorityError("paid worker requires exact committed admission identities")
    if out_dir is not None and Path(receipt.get("out_dir", "")).resolve() != Path(out_dir).resolve():
        raise AuthorityError("paid worker output differs from its claimed run")
    if snapshot is not None and (snapshot.authority_commit != receipt.get("execution_commit", receipt.get("resume_commit", receipt.get("claim_commit")))
                                 or snapshot.digest != receipt.get("snapshot_digest")
                                 or snapshot.release != receipt.get("release")):
        raise AuthorityError("paid worker snapshot differs from its exact claim commit")
    if receipt.get("owner_pid") == os.getpid():
        _verify_committed_claim(binding, canon(receipt))
    else:
        # Owners save the commit before verification, then return the receipt
        # for inheritance only on success. Older inherited receipts fail here.
        # Compare the whole receipt so fields checked remotely by the owner
        # (including mode, models and knowledge) cannot change in a worker.
        try:
            recorded = json.loads((Path(receipt["out_dir"]) / ".paid-launch.json").read_text(encoding="utf-8"))
            matches = canon(recorded) == canon(receipt)
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise AuthorityError("paid worker cannot read its owner's local launch receipt") from error
        if not matches:
            raise AuthorityError("paid worker receipt differs from its owner's local launch receipt")
    return receipt


def verify_remote_claim(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    """Verify a coordinator caller's committed claim without local process ancestry.

    The caller retains its own live-process ownership check. This transport-side
    check validates exact committed owner facts and returns an immutable view of
    the reservation that owns the database scope and canonical run identity.

    Args:
        receipt: The caller's claimed or ready paid-launch receipt.

    Returns:
        The exact remotely committed reservation, with nested values immutable.

    Raises:
        AuthorityError: Receipt shape, authority binding or committed provenance
            is invalid. No process on the coordinator claims the caller's run.
    """
    from .binding import load_binding
    from .types import GitAuthorityBinding

    binding = load_binding()
    if not isinstance(binding, GitAuthorityBinding):
        raise AuthorityError("remote paid admission requires the Git primary authority binding")
    if (not isinstance(receipt, Mapping) or receipt.get("schema") != SCHEMA_RECEIPT
            or not isinstance(receipt.get("phase"), str) or receipt["phase"] not in {"claimed", "ready"}):
        raise AuthorityError("remote paid worker has no claimed launch receipt")
    allowed = {"schema", "reservation_id", "reservation_commit", "claim_commit", "token", "phase", "run", "out_dir",
               "release", "snapshot_digest", "source_commit", "namespace", "inputs", "knowledge", "requires_ready",
               "owner_host", "owner_pid", "owner_started", "execution_commit", "fresh_knowledge", "resume_commit", "resume_event",
               "execution_source_commit", "answer_resume_commit", "answer_resume_event", "diagnostic", "configuration", "counted", "parent", "admission"}
    if set(receipt) - allowed or (receipt["phase"] != "ready" and "execution_commit" in receipt):
        raise AuthorityError("remote paid worker receipt has an invalid execution identity")
    if (("resume_commit" in receipt) != ("resume_event" in receipt)
            or ("fresh_knowledge" in receipt and receipt["fresh_knowledge"] is not True)
            or ("diagnostic" in receipt and receipt["diagnostic"] is not True)
            or ("counted" in receipt and type(receipt["counted"]) is not bool)):
        raise AuthorityError("remote paid worker receipt has an invalid build resume identity")
    if ("answer_resume_commit" in receipt) != ("answer_resume_event" in receipt) or (
            "answer_resume_commit" in receipt and receipt["phase"] != "ready"):
        raise AuthorityError("remote paid worker receipt has an invalid answering continuation identity")
    required = ("reservation_id", "token", "run", "out_dir", "release", "snapshot_digest", "namespace",
                "owner_host", "owner_started")
    if (any(not isinstance(receipt.get(key), str) or not receipt[key].strip() for key in required)
            or type(receipt.get("owner_pid")) is not int or receipt["owner_pid"] < 1
            or type(receipt.get("requires_ready")) is not bool
            or not isinstance(receipt.get("knowledge"), Mapping)
            or not isinstance(receipt.get("inputs"), Mapping)):
        raise AuthorityError("remote paid worker receipt has an invalid execution identity")
    commits = ("reservation_commit", "claim_commit", "source_commit") + (("execution_commit",) if receipt["phase"] == "ready" else ())
    if receipt.get("resume_commit") is not None:
        commits += ("resume_commit",)
    if receipt.get("answer_resume_commit") is not None:
        commits += ("answer_resume_commit",)
    if "execution_source_commit" in receipt:
        if receipt.get("resume_commit") is None and receipt.get("answer_resume_commit") is None:
            raise AuthorityError("continued execution source requires a committed build resume or answering continuation")
        commits += ("execution_source_commit",)
    if any(not isinstance(receipt.get(key), str) or re.fullmatch(r"[0-9a-f]{40}", receipt[key]) is None for key in commits):
        raise AuthorityError("remote paid worker requires exact committed admission identities")
    databases = receipt["inputs"].get("db_fingerprints")
    if (not isinstance(databases, Mapping) or not 1 <= len(databases) <= 11
            or any(not isinstance(key, str) or not key.strip() or not isinstance(value, str) or not value
                   for key, value in databases.items())):
        raise AuthorityError("remote paid worker has an invalid reserved database scope")
    try:
        reservation = _verify_committed_claim(binding, canon(dict(receipt)))
    except (TypeError, ValueError) as error:
        raise AuthorityError("remote paid worker receipt is not a serializable committed identity") from error

    def immutable(value: Any) -> Any:
        if isinstance(value, dict):
            return MappingProxyType({key: immutable(item) for key, item in value.items()})
        if isinstance(value, list):
            return tuple(immutable(item) for item in value)
        return value

    return immutable(reservation)


@lru_cache(maxsize=64)
def _verify_committed_claim(binding: Any, encoded_receipt: str) -> dict[str, Any]:
    """Cache verified provenance; retry only transient Git reads, never mismatches."""
    from .git_authority import GitAuthorityConflict

    confirmed = _CONFIRMED_ADMISSIONS.get((binding, encoded_receipt))
    if confirmed is not None:
        return dict(confirmed)
    receipt = json.loads(encoded_receipt)
    confirmed = local_confirmation(binding, receipt)
    if confirmed is not None:
        return confirmed
    for delay in (0.1, 0.2):
        try:
            return _read_committed_claim(binding, receipt)
        except AuthorityError as error:
            # _git has no transport subtype: match only its remote read errors.
            transport = str(error) in {
                "Git authority operation unavailable: ls-remote", "Git authority operation refused: ls-remote",
                "Git authority operation unavailable: fetch", "Git authority operation refused: fetch",
            }
            if not (isinstance(error, GitAuthorityConflict) or transport):
                raise
            time.sleep(delay)
    return _read_committed_claim(binding, receipt)


def _read_committed_claim(binding: Any, receipt: Mapping[str, Any], *, reader: Any = None) -> dict[str, Any]:
    """Read and check one immutable receipt, including its continuation history."""
    from .authority import Authority

    commit = receipt.get("answer_resume_commit") or receipt.get("execution_commit", receipt.get("resume_commit", receipt.get("claim_commit")))
    with (nullcontext(reader) if reader is not None else Authority.read(binding=binding, commit=commit)) as reader:
        if getattr(reader, "authority_commit", commit) != commit:
            raise AuthorityError("paid claim reader differs from its exact pushed commit")
        reservations = [event["payload"] for event in reader.events("eval-reserve")
                        if event["payload"].get("id") == receipt.get("reservation_id")]
        claims = reservation_claims(reader, reservations[0], receipt.get("reservation_commit")) if len(reservations) == 1 else []
        if (len(reservations) == 1 and reservations[0].get("admission") == "reserved-owner/1"
                and receipt.get("claim_commit") != receipt.get("reservation_commit")):
            raise AuthorityError("fused reservation and claim must name the same pushed commit")
        if (len(reservations) != 1 or len(claims) != 1
                or claims[0].get("token") != receipt.get("token")
                or claims[0].get("reservation_commit") != receipt.get("reservation_commit")
                or reservations[0].get("token") != receipt.get("token")):
            raise AuthorityError("paid worker claim is not the exact committed reservation")
        reservation = reservations[0]
        if reservation.get("admission") == "reserved-owner/1" and receipt.get("admission") != "reserved-owner/1":
            raise AuthorityError("paid worker changed its fused admission identity")
        owner = claims[0]
        if receipt.get("resume_commit") is not None:
            resumed = [event["payload"] for event in reader.events("register")
                       if event.get("object_id") == receipt.get("resume_event")]
            if len(resumed) == 1 and resumed[0].get("schema") == SCHEMA_CHILD_BUILD_RESUME:
                _verify_child_build_resume(resumed[0], receipt, reservation, claims[0], binding.authority_uuid)
            elif len(resumed) == 1 and resumed[0].get("schema") == SCHEMA_CHILD_OPERATIONAL_RESUME:
                _child_build_chain(binding, reader, receipt, reservation, claims[0])
            elif (len(resumed) != 1 or resumed[0].get("schema") != SCHEMA_BUILD_RESUME
                    or not reservation.get("fresh_knowledge")
                    or any(resumed[0].get(key) != receipt.get(key) for key in ("reservation_id", "claim_commit", "token"))):
                raise AuthorityError("paid build owner is not its committed same-claim resume")
            owner = resumed[0]
        if receipt.get("answer_resume_commit") is not None:
            answered = [event["payload"] for event in reader.events("register")
                        if event.get("object_id") == receipt.get("answer_resume_event")
                        and event["payload"].get("schema") == SCHEMA_ANSWER_RESUME]
            if (len(answered) != 1 or receipt.get("phase") != "ready" or not reservation.get("fresh_knowledge")
                    or any(answered[0].get(key) != receipt.get(key)
                           for key in ("reservation_id", "claim_commit", "token", "execution_commit"))):
                raise AuthorityError("paid answering owner is not its committed same-claim continuation")
            owner = answered[0]
        if any(owner.get(key) != receipt.get(key) for key in OWNER_FIELDS):
            raise AuthorityError("paid worker claim is not the exact committed reservation")
        if any(reservation.get(key) != receipt.get(key) for key in ("release", "run", "out_dir", "source_commit", "namespace", "inputs")):
            raise AuthorityError("paid worker receipt changed its reserved execution identity")
        if (reservation.get("diagnostic") != receipt.get("diagnostic")
                or ("parent" in receipt and reservation.get("parent") != receipt["parent"])
                or reservation.get("configuration") != receipt.get("configuration")
                or ("configuration" in reservation and reservation.get("counted") != receipt.get("counted"))
                or ("counted" in receipt and reservation.get("counted") != receipt["counted"])):
            raise AuthorityError("paid worker changed its committed evaluation mode or models")
        if reservation.get("diagnostic") is True:
            if (reservation.get("counted") is not False or reservation.get("proposal") is not None
                    or reservation.get("parent") is not None or reservation.get("requires_ready")
                    or receipt.get("phase") != "claimed"):
                raise AuthorityError("diagnostic claim cannot become a counted child or knowledge build")
            configuration = receipt.get("configuration")
            if (not isinstance(configuration, Mapping)
                    or set(configuration) not in ({"candidates", "judge"}, {"candidates", "judge", "saved_candidates"})):
                raise AuthorityError("diagnostic claim has no exact model configuration")
            verify_evaluation_configuration(receipt, diagnostic=True,
                                            candidates=configuration["candidates"], judge=configuration["judge"],
                                            saved_candidates=configuration.get("saved_candidates"))
        if receipt.get("execution_source_commit") != owner.get("execution_source_commit"):
            raise AuthorityError("paid worker changed its committed continuation source")
        if reservation.get("requires_ready") != receipt.get("requires_ready"):
            raise AuthorityError("paid worker changed its knowledge build admission")
        if bool(reservation.get("fresh_knowledge")) != bool(receipt.get("fresh_knowledge")):
            raise AuthorityError("paid worker changed its fresh knowledge admission")
        if receipt.get("phase") == "ready":
            ready = [event["payload"] for event in reader.events("eval-ready")
                     if event["payload"].get("reservation_id") == receipt.get("reservation_id")]
            if (len(ready) != 1 or ready[0].get("claim_commit") != receipt.get("claim_commit")
                    or ready[0].get("token") != receipt.get("token")
                    or {key: ready[0].get("knowledge", {}).get(key) for key in ("root", "fingerprints")} != receipt.get("knowledge")):
                raise AuthorityError("paid worker final knowledge readiness is not the exact committed claim")
            _checked_build_plan(ready[0]["knowledge"], receipt["inputs"], receipt["release"], fresh=False)
        elif {key: reservation.get("knowledge", {}).get(key) for key in ("root", "fingerprints")} != receipt.get("knowledge"):
            raise AuthorityError("paid worker changed the reserved knowledge fingerprints")
        return dict(reservation)


def _assert_stopped_build(receipt: Mapping[str, Any], source_dir: Path) -> None:
    """Refuse an alive, foreign, inaccessible or possibly orphaned build owner."""
    if receipt.get("owner_host") != socket.gethostname():
        raise AuthorityError("build resume must remain on its original execution host")
    try:
        _, started = _process(receipt["owner_pid"])
    except ProcessLookupError:
        pass
    except OSError as error:
        if getattr(error, "winerror", None) != 87:
            raise AuthorityError("cannot establish that the original build owner stopped") from error
    except subprocess.CalledProcessError:
        pass  # The bounded ps query found no live PID.
    else:
        if started == receipt.get("owner_started"):
            raise AuthorityError("original build owner is still alive; follow that job")
    ancestors: set[int] = set()
    pid = os.getpid()
    while pid > 0 and pid not in ancestors:
        ancestors.add(pid)
        try:
            pid, _ = _process(pid)
        except (OSError, subprocess.SubprocessError):
            break
    if os.name == "nt":
        command = ("[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false); "
                   "@(Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine) | ConvertTo-Json -Compress")
        output = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                capture_output=True, text=True, encoding="utf-8", check=True, timeout=30).stdout
        rows = json.loads(output)
    else:
        output = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True, timeout=30).stdout
        rows = [{"ProcessId": int(parts[0]), "CommandLine": parts[1]} for line in output.splitlines()
                if len(parts := line.strip().split(maxsplit=1)) == 2]
    if not isinstance(rows, list):
        raise AuthorityError("build resume could not verify the process inventory")
    paths = [str(source_dir.resolve()), receipt["out_dir"], receipt["knowledge"]["root"]]
    paths = [path.replace("\\", "/").casefold() for path in paths]
    for row in rows:
        command = str(row.get("CommandLine") or "").replace("\\", "/").casefold()
        if row["ProcessId"] in ancestors or not any(name in command for name in (
                "eval_evolution.py", "run_official_pipeline.py", "eval_parallel.py", "eval_bird.py", "mcp_server.py",
                "inject_examples.py", "inject_examples_topk.py", "dedup_examples.py", "rubiksql_hb.server")):
            continue
        if any(path in command for path in paths) or not re.search(r"(?:[a-z]:/|/).*python", command):
            raise AuthorityError("a build/evaluation worker may still be running; inspect and follow it before resume")


def check_build_continuation(continuation: Mapping[str, Any], receipt: Mapping[str, Any],
                             source_commit: str) -> dict[str, Any]:
    """Bind an explicit user-authorized continuation to one stopped receipt and source.

    Args:
        continuation: Operator-supplied record with verbatim authorization, authority,
            claim, previous receipt hash, old checkout and exact corrected source.
        receipt: Preserved current build receipt; its reservation is never replaced.
        source_commit: Validated execution checkout's exact Git commit.

    Returns:
        A detached authorization record suitable for the existing resume event.

    Raises:
        AuthorityError: The explicit authorization or any bound identity is absent
            or differs. This does not waive ownership, readiness or scope guards.
    """
    from .binding import load_binding

    fields = {"schema", "request_id", "authorization", "authority_uuid", "reservation_id", "claim_commit",
              "previous_receipt_sha256", "previous_source_commit", "source_commit", "previous_source_dir",
              "waive_retry_limit"}
    if (not isinstance(continuation, Mapping) or set(continuation) != fields
            or continuation.get("schema") != SCHEMA_BUILD_CONTINUATION
            or continuation.get("waive_retry_limit") is not True
            or not isinstance(continuation.get("authorization"), str) or not continuation["authorization"].strip()
            or not isinstance(continuation.get("request_id"), str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", continuation["request_id"]) is None
            or continuation.get("authority_uuid") != load_binding().authority_uuid
            or any(continuation.get(key) != receipt.get(key) for key in ("reservation_id", "claim_commit"))
            or continuation.get("previous_receipt_sha256") != sha256_hex(canon(receipt))
            or continuation.get("previous_source_commit") != receipt.get("execution_source_commit", receipt.get("source_commit"))
            or continuation.get("source_commit") != source_commit
            or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
            or not isinstance(continuation.get("previous_source_dir"), str)
            or not Path(continuation["previous_source_dir"]).is_absolute()):
        raise AuthorityError("build continuation requires explicit authorization bound to this receipt, authority and source")
    return dict(continuation)


def _child_build_continuation(continuation: Mapping[str, Any], receipt: Mapping[str, Any],
                              source_commit: str, authority_uuid: str) -> dict[str, Any]:
    """Validate the closed, one-use authorization independently of local evidence files."""
    schema = continuation.get("schema") if isinstance(continuation, Mapping) else None
    schema = schema if isinstance(schema, str) else None
    operational = schema in (SCHEMA_CHILD_OPERATIONAL_CONTINUATION, SCHEMA_CHILD_DIRECT_OPERATIONAL_CONTINUATION)
    evidence_names = {SCHEMA_CHILD_BUILD_CONTINUATION: CHILD_CONTINUATION_EVIDENCE,
                      SCHEMA_CHILD_OPERATIONAL_CONTINUATION: CHILD_OPERATIONAL_EVIDENCE,
                      SCHEMA_CHILD_NATURAL_CONTINUATION: CHILD_NATURAL_EVIDENCE,
                      SCHEMA_CHILD_DIRECT_OPERATIONAL_CONTINUATION: CHILD_DIRECT_OPERATIONAL_EVIDENCE}
    fields = {"schema", "request_id", "authorization", "authority_uuid", "pin", "reservation_id", "claim_commit",
              "previous_receipt_sha256", "previous_source_commit", "source_commit", "previous_source_dir",
              "previous_owner", "evidence"}
    if operational:
        fields.add("build_attempt")
    if (not isinstance(continuation, Mapping) or set(continuation) != fields
            or schema not in evidence_names
            or (operational and (type(continuation.get("build_attempt")) is not int or not 3 <= continuation["build_attempt"] <= 4))
            or not isinstance(continuation.get("authorization"), str) or not continuation["authorization"].strip()
            or not isinstance(continuation.get("request_id"), str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", continuation["request_id"]) is None
            or continuation.get("authority_uuid") != authority_uuid
            or not isinstance(continuation.get("pin"), str) or re.fullmatch(r"p[0-9a-f]{16}", continuation["pin"]) is None
            or any(continuation.get(key) != receipt.get(key) for key in ("reservation_id", "claim_commit"))
            or continuation.get("previous_receipt_sha256") != sha256_hex(canon(receipt))
            or continuation.get("previous_source_commit") != receipt.get("execution_source_commit" if operational else "source_commit")
            or continuation.get("source_commit") != source_commit
            or not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
            or continuation.get("previous_owner") != {key: receipt.get(key) for key in OWNER_FIELDS}
            or not isinstance(continuation.get("previous_source_dir"), str)
            or not Path(continuation["previous_source_dir"]).is_absolute()
            or not isinstance(continuation.get("evidence"), Mapping)
            or set(continuation["evidence"]) != set(evidence_names[schema])):
        raise AuthorityError("child build continuation requires exact receipt-bound user authorization")
    for item in continuation["evidence"].values():
        if (not isinstance(item, Mapping) or set(item) != {"path", "sha256"}
                or not isinstance(item["path"], str) or not Path(item["path"]).is_absolute()
                or not isinstance(item["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None):
            raise AuthorityError("child build continuation evidence needs absolute paths and exact hashes")
    return json.loads(canon(continuation))


def _verify_child_build_resume(event: Mapping[str, Any], receipt: Mapping[str, Any], reservation: Mapping[str, Any],
                               claim: Mapping[str, Any], authority_uuid: str) -> None:
    """Admit only the distinct counted-child event; baseline resume rules stay separate."""
    fields = {"schema", "reservation_id", "claim_commit", "token", "previous_receipt", "previous_owner",
              "previous_receipt_sha256", "created_at", "execution_source_commit", "continuation", *OWNER_FIELDS}
    previous = event.get("previous_receipt")
    previous_fields = {"schema", "reservation_id", "reservation_commit", "claim_commit", "token", "phase", "run",
                       "out_dir", "release", "snapshot_digest", "source_commit", "namespace", "inputs", "knowledge",
                       "requires_ready", *OWNER_FIELDS}
    if "configuration" in reservation:
        previous_fields.update(("configuration", "counted"))
    if reservation.get("admission") == "reserved-owner/1":
        previous_fields.add("admission")
    if isinstance(previous, Mapping) and "parent" in previous:
        previous_fields.add("parent")
    if (set(event) != fields or event.get("schema") != SCHEMA_CHILD_BUILD_RESUME or not isinstance(previous, dict)
            or set(previous) != previous_fields
            or previous.get("schema") != SCHEMA_RECEIPT or previous.get("phase") != "claimed"
            or any(key in previous for key in ("resume_commit", "answer_resume_commit", "execution_commit", "execution_source_commit"))
            or reservation.get("counted") is not True or reservation.get("fresh_knowledge")
            or reservation.get("requires_ready") is not True or previous.get("requires_ready") is not True
            or any(event.get(key) != receipt.get(key) or previous.get(key) != receipt.get(key)
                   for key in ("reservation_id", "claim_commit", "token"))
            or previous.get("reservation_commit") != claim.get("reservation_commit")
            or ("parent" in previous and previous["parent"] != reservation.get("parent"))
            or event.get("previous_owner") != {key: previous.get(key) for key in OWNER_FIELDS}
            or any(previous.get(key) != claim.get(key) for key in OWNER_FIELDS)
            or any(previous.get(key) != reservation.get(key)
                   for key in ("release", "run", "out_dir", "source_commit", "namespace", "inputs"))
            or any(previous.get(key) != reservation.get(key) for key in ("configuration", "counted")
                   if key in reservation and key in previous_fields)
            or previous.get("admission") != reservation.get("admission")
            or previous.get("knowledge") != {key: reservation.get("knowledge", {}).get(key) for key in ("root", "fingerprints")}
            or previous.get("fresh_knowledge")
            or event.get("previous_receipt_sha256") != sha256_hex(canon(previous))):
        raise AuthorityError("paid child owner is not its exact committed counted-child continuation")
    authorization = _child_build_continuation(event.get("continuation"), previous,
                                               event.get("execution_source_commit"), authority_uuid)
    if (authorization["schema"] not in (SCHEMA_CHILD_BUILD_CONTINUATION, SCHEMA_CHILD_NATURAL_CONTINUATION)
            or reservation.get("pin") != authorization["pin"] or reservation.get("proposal") != authorization["pin"]):
        raise AuthorityError("paid child continuation changed its reserved proposal")


def _child_build_chain(binding: Any, reader: Any, receipt: Mapping[str, Any], reservation: Mapping[str, Any],
                       claim: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    """Verify a bounded, unforked child chain using committed data, never local evidence paths."""
    from .authority import Authority

    rows = [dict(row) for row in reader.events("register")
            if row["payload"].get("reservation_id") == receipt.get("reservation_id")]
    remaining = {row["object_id"] for row in rows}
    requests: set[str] = set()
    current = dict(receipt)
    first: dict[str, Any] = {}
    expected = 0
    previous_fields = {"schema", "reservation_id", "reservation_commit", "claim_commit", "token", "phase", "run",
                       "out_dir", "release", "snapshot_digest", "source_commit", "namespace", "inputs", "knowledge",
                       "requires_ready", "resume_commit", "resume_event", "execution_source_commit", *OWNER_FIELDS}
    if "configuration" in reservation:
        previous_fields.update(("configuration", "counted"))
    if reservation.get("admission") == "reserved-owner/1":
        previous_fields.add("admission")
    event_fields = {"schema", "reservation_id", "claim_commit", "token", "previous_receipt", "previous_owner",
                    "previous_receipt_sha256", "created_at", "execution_source_commit", "continuation", "build_attempt", *OWNER_FIELDS}
    for _ in range(3):
        matches = [row["payload"] for row in rows if row["object_id"] == current.get("resume_event")]
        if len(matches) != 1 or current.get("resume_event") not in remaining:
            raise AuthorityError("child operational continuation has no unique committed predecessor")
        event = dict(matches[0])
        request_id = event.get("continuation", {}).get("request_id") if isinstance(event.get("continuation"), dict) else None
        if not isinstance(request_id, str) or request_id in requests:
            raise AuthorityError("child operational continuation reuses an authorization request")
        requests.add(request_id)
        attempt = 2 if event.get("schema") == SCHEMA_CHILD_BUILD_RESUME else event.get("build_attempt")
        if (type(attempt) is not int or not 2 <= attempt <= 4 or (expected and attempt != expected)
                or any(event.get(key) != current.get(key) for key in
                       ("reservation_id", "claim_commit", "token", "execution_source_commit", *OWNER_FIELDS))
                or any(current.get(key) != reservation.get(key) for key in
                       ("release", "run", "out_dir", "source_commit", "namespace", "inputs"))
                or current.get("reservation_commit") != claim.get("reservation_commit")):
            raise AuthorityError("child operational continuation changed its predecessor identity or build count")
        if not first:
            first = event
        remaining.remove(current["resume_event"])
        if attempt == 2:
            _verify_child_build_resume(event, current, reservation, claim, binding.authority_uuid)
            if remaining:
                raise AuthorityError("child operational continuation refuses a fork or stale chain tail")
            return int(first.get("build_attempt", 2)), first
        previous = event.get("previous_receipt")
        fields = previous_fields | ({"parent"} if isinstance(previous, Mapping) and "parent" in previous else set())
        if (set(event) != event_fields or event.get("schema") != SCHEMA_CHILD_OPERATIONAL_RESUME
                or not isinstance(previous, dict) or set(previous) != fields
                or previous.get("schema") != SCHEMA_RECEIPT or previous.get("phase") != "claimed"
                or ("parent" in previous and previous["parent"] != reservation.get("parent"))
                or previous.get("requires_ready") is not True
                or any(previous.get(key) != reservation.get(key) for key in ("configuration", "counted")
                       if key in reservation and key in previous_fields)
                or previous.get("admission") != reservation.get("admission")
                or any(previous.get(key) != receipt.get(key) for key in ("reservation_id", "claim_commit", "token"))
                or previous.get("knowledge") != {key: reservation.get("knowledge", {}).get(key) for key in ("root", "fingerprints")}
                or any(not isinstance(previous.get(key), str) or re.fullmatch(r"[0-9a-f]{40}", previous[key]) is None
                       for key in ("resume_commit", "execution_source_commit"))
                or event.get("previous_owner") != {key: previous.get(key) for key in OWNER_FIELDS}
                or event.get("previous_receipt_sha256") != sha256_hex(canon(previous))):
            raise AuthorityError("paid child owner is not its exact committed operational continuation")
        authorization = _child_build_continuation(event.get("continuation"), previous,
                                                   event.get("execution_source_commit"), binding.authority_uuid)
        if (authorization.get("schema") not in (SCHEMA_CHILD_OPERATIONAL_CONTINUATION, SCHEMA_CHILD_DIRECT_OPERATIONAL_CONTINUATION)
                or authorization["build_attempt"] != attempt
                or reservation.get("pin") != authorization["pin"] or reservation.get("proposal") != authorization["pin"]):
            raise AuthorityError("child operational continuation changed its proposal or build count")
        # The predecessor must already exist at its recorded commit, with exactly the remaining chain.
        with Authority.read(binding=binding, commit=previous["resume_commit"]) as ancestor:
            preceding = [dict(row) for row in ancestor.events("register")
                         if row["payload"].get("reservation_id") == receipt.get("reservation_id")]
        if ({row["object_id"] for row in preceding} != remaining
                or any(row not in rows for row in preceding)):
            raise AuthorityError("child operational continuation changed its committed predecessor history")
        predecessor = next((row["payload"] for row in preceding if row["object_id"] == previous["resume_event"]), {})
        previous_authorization = predecessor.get("continuation", {})
        if not isinstance(previous_authorization, dict):
            raise AuthorityError("child operational continuation has an invalid launch origin")
        previous_direct = previous_authorization.get("schema") in (
            SCHEMA_CHILD_NATURAL_CONTINUATION, SCHEMA_CHILD_DIRECT_OPERATIONAL_CONTINUATION)
        if ((previous_direct and authorization["schema"] != SCHEMA_CHILD_DIRECT_OPERATIONAL_CONTINUATION)
                or (authorization["schema"] == SCHEMA_CHILD_DIRECT_OPERATIONAL_CONTINUATION
                    and (not previous_direct or any(authorization["evidence"][name] !=
                         previous_authorization.get("evidence", {}).get(name) for name in ("launch_context", "provider_assignment"))))):
            raise AuthorityError("child operational continuation changed its immutable launch origin")
        current, rows, expected = previous, preceding, attempt - 1
    raise AuthorityError("child operational continuation exceeds four root build admissions")


def _read_child_evidence(continuation: Mapping[str, Any]) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Read physical evidence by exact byte hash; only the natural wrapper/log are raw."""
    evidence = continuation["evidence"]
    raw: dict[str, bytes] = {}
    documents: dict[str, Any] = {}
    for name, item in evidence.items():
        path = Path(item["path"])
        direct = continuation["schema"] in (SCHEMA_CHILD_NATURAL_CONTINUATION, SCHEMA_CHILD_DIRECT_OPERATIONAL_CONTINUATION)
        if path.is_symlink() or not path.is_file() or (direct and path.resolve() != path):
            raise AuthorityError("child continuation evidence is not a physical file: " + name)
        raw[name] = path.read_bytes()
        if sha256_hex(raw[name]) != item["sha256"]:
            raise AuthorityError("child continuation evidence bytes changed: " + name)
        if continuation["schema"] == SCHEMA_CHILD_NATURAL_CONTINUATION and name in ("launch_wrapper", "launch_log"):
            continue
        try:
            documents[name] = json.loads(raw[name])
        except (ValueError, UnicodeError) as error:
            raise AuthorityError("child continuation evidence is not JSON: " + name) from error
        if not isinstance(documents[name], dict):
            raise AuthorityError("child continuation evidence is not an object: " + name)
    return raw, documents


def _child_process_inventory() -> list[dict[str, Any]]:
    """Inspect local processes without an optional Python platform dependency."""
    try:
        if os.name == "nt":
            command = ("$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); "
                       "@(Get-CimInstance Win32_Process | ForEach-Object { "
                       "@{pid=[int]$_.ProcessId; parent_pid=[int]$_.ParentProcessId; image=$_.Name; "
                       "started=$(if ($_.CreationDate) {$_.CreationDate.ToUniversalTime().ToFileTimeUtc().ToString()} "
                       "else {''}); command_line=$_.CommandLine} }) | ConvertTo-Json -Compress")
            result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                                    capture_output=True, text=True, encoding="utf-8", check=True, timeout=30)
            rows = json.loads(result.stdout)
        else:
            result = subprocess.run(["ps", "-eo", "pid=,ppid=,comm=,args="], capture_output=True,
                                    text=True, check=True, timeout=30)
            rows = []
            for line in result.stdout.splitlines():
                parts = line.strip().split(maxsplit=3)
                if len(parts) != 4:
                    raise AuthorityError("child continuation could not read a complete process inventory")
                pid, parent = int(parts[0]), int(parts[1])
                try:
                    _, started = _process(pid)
                except (OSError, subprocess.SubprocessError):
                    started = ""  # The row may have exited; unreadable possible workers are refused below.
                rows.append({"pid": pid, "parent_pid": parent, "started": started,
                             "image": parts[2], "command_line": parts[3]})
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise AuthorityError("child continuation could not verify its live process inventory") from error
    if not isinstance(rows, list):
        raise AuthorityError("child continuation could not verify its live process inventory")
    return rows


def _check_child_processes(rows: Any, receipt: Mapping[str, Any], source_dir: Path) -> None:
    """Refuse the old owner, its surviving descendants and ambiguous build workers."""
    fields = {"pid", "parent_pid", "started", "image", "command_line"}
    if (not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) != fields
            or type(row["pid"]) is not int or row["pid"] < 0
            or type(row["parent_pid"]) is not int or row["parent_pid"] < 0
            or not isinstance(row["started"], str) or not isinstance(row["image"], str)
            or (row["command_line"] is not None and not isinstance(row["command_line"], str)) for row in rows)
            or len({row["pid"] for row in rows}) != len(rows)):
        raise AuthorityError("child continuation has an invalid process inventory")
    ancestors: set[int] = set()
    pid = os.getpid()
    while pid > 0 and pid not in ancestors:
        ancestors.add(pid)
        try:
            pid, _ = _process(pid)
        except (OSError, subprocess.SubprocessError):
            break
    descendants = {receipt["owner_pid"]}
    root = next((row for row in rows if row["pid"] == receipt["owner_pid"]), None)

    def after_reused_owner(row: Mapping[str, Any]) -> bool:
        if root is None or root["started"] == receipt["owner_started"] or row["parent_pid"] != root["pid"]:
            return False
        try:
            if root["started"].isdigit() and row["started"].isdigit():
                return int(row["started"]) > int(root["started"])
            return datetime.strptime(row["started"], "%a %b %d %H:%M:%S %Y") > datetime.strptime(
                root["started"], "%a %b %d %H:%M:%S %Y")
        except ValueError:
            return False  # Unknown birth order cannot rule out an original orphan.

    for _ in range(len(rows)):
        children = {row["pid"] for row in rows if row["parent_pid"] in descendants and not after_reused_owner(row)}
        if children.issubset(descendants):
            break
        descendants.update(children)
    paths = [str(source_dir.resolve()), receipt["out_dir"], receipt["knowledge"]["root"]]
    paths = [value.replace("\\", "/").casefold() for value in paths]
    workers = ("eval_evolution.py", "run_official_pipeline.py", "eval_parallel.py", "eval_bird.py", "mcp_server.py",
               "rubiksql_hb.server", "setup_profiling.py", "setup_table_profiles.py", "setup_db_profile.py",
               "inject_examples.py", "inject_examples_topk.py", "dedup_examples.py", "rulegen.py",
               "continue_counted_child.py")
    for row in rows:
        if row["pid"] == receipt["owner_pid"] and row["started"] == receipt["owner_started"]:
            raise AuthorityError("original build owner is still alive; follow that job")
        if row["pid"] in ancestors:
            continue
        command = (row["command_line"] or "").replace("\\", "/").casefold()
        image = Path(row["image"].replace("\\", "/")).name.casefold()
        possible_worker = any(name in image for name in ("python", "powershell", "pwsh", "cmd.exe"))
        if ((row["pid"] in descendants and row["pid"] != receipt["owner_pid"])
                or (possible_worker and (not command or not row["started"]))
                or (any(name in command for name in workers)
                    and (any(path in command for path in paths)
                         or not re.search(r"(?:[a-z]:/|/).*python", command)))
                or (possible_worker and any(path in command for path in paths))):
            raise AuthorityError("a build/evaluation worker may still be running; inspect and follow it before resume")


def _assert_stopped_direct_build(receipt: Mapping[str, Any], source_dir: Path) -> None:
    _assert_stopped_build(receipt, source_dir)
    _check_child_processes(_child_process_inventory(), receipt, source_dir)


def _child_direct_context(continuation: Mapping[str, Any], receipt: Mapping[str, Any], context: Mapping[str, Any],
                          *, initial: bool) -> None:
    """Validate explicitly retrospective direct-launch context, never a fictional helper request."""
    from .binding import PACKAGE_BINDING, binding_document, load_binding, selected_binding_path

    fields = {"schema", "provenance", "observed_at", "proposal", "source", "binding", "namespace", "inputs",
              "knowledge_fingerprints", "paths", "queries_sha256", "parent_sha256", "original_owner", "launch_evidence"}
    source_fields = {"directory", "commit", "interpreter", "home", "host"}
    path_fields = {"output", "knowledge", "parent", "records", "cache", "databases", "queries"}
    binding = load_binding()
    if (set(context) != fields or context.get("schema") != "rubiksql.child-direct-launch-context/1"
            or context.get("provenance") != "retrospective"
            or context.get("proposal") != {"pin": continuation["pin"], "release": receipt["release"]}
            or context.get("namespace") != receipt["namespace"] or context.get("inputs") != receipt["inputs"]
            or context.get("knowledge_fingerprints") != receipt["knowledge"]["fingerprints"]
            or not isinstance(context.get("source"), dict) or set(context["source"]) != source_fields
            or context["source"].get("commit") != receipt["source_commit"]
            or context["source"].get("host") != receipt["owner_host"]
            or any(not isinstance(context["source"][key], str) or not Path(context["source"][key]).is_absolute()
                   for key in ("directory", "interpreter", "home"))
            or not isinstance(context.get("paths"), dict) or set(context["paths"]) != path_fields
            or any(not isinstance(value, str) or not Path(value).is_absolute() for value in context["paths"].values())
            or context["paths"]["output"] != receipt["out_dir"]
            or context["paths"]["knowledge"] != receipt["knowledge"]["root"]
            or not isinstance(context.get("original_owner"), dict) or set(context["original_owner"]) != set(OWNER_FIELDS)
            or (initial and (context["original_owner"] != continuation["previous_owner"]
                             or context["source"]["directory"] != continuation["previous_source_dir"]))
            or not isinstance(context.get("binding"), dict)
            or set(context["binding"]) != {"path", "sha256", "authority_uuid", "ref"}
            or context["binding"].get("authority_uuid") != continuation["authority_uuid"]
            or context["binding"].get("ref") != binding.ref
            or context["binding"].get("path") != str(selected_binding_path() or PACKAGE_BINDING)
            or not isinstance(context.get("launch_evidence"), dict)
            or set(context["launch_evidence"]) != {"launch_wrapper", "launch_log", "pipeline_state", "pipeline_exit",
                                                    "provider_assignment", "previous_receipt_file"}
            or any(not isinstance(context.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", context[key]) is None
                   for key in ("queries_sha256", "parent_sha256"))):
        raise AuthorityError("child natural evidence changed its retrospective direct-launch context")
    item = context["binding"]
    path = Path(item["path"])
    try:
        body = path.read_bytes()
        if (path.is_symlink() or path.resolve() != path or sha256_hex(body) != item["sha256"]
                or json.loads(body) != binding_document(binding)):
            raise AuthorityError("child natural evidence changed its original authority binding bytes")
    except (OSError, ValueError, UnicodeError) as error:
        raise AuthorityError("child natural evidence cannot verify its original authority binding bytes") from error
    queries = Path(context["paths"]["queries"])
    if (queries.is_symlink() or queries.resolve() != queries or not queries.is_file()
            or sha256_hex(queries.read_bytes()) != context["queries_sha256"]):
        raise AuthorityError("child natural evidence changed its original knowledge queries")


def _check_natural_observation(continuation: Mapping[str, Any], receipt: Mapping[str, Any],
                               documents: Mapping[str, Any]) -> None:
    context, inventory, verified, exited = (documents[name] for name in
        ("launch_context", "process_inventory", "stopped_verification", "pipeline_exit"))
    times = [exited.get("ts"), context.get("observed_at"), inventory.get("observed_at"), verified.get("observed_at")]
    now = time.time()
    if (any(type(value) not in (int, float) or not math.isfinite(value) for value in times)
            or not 0 < times[0] <= times[1] <= times[2] <= times[3] <= now
            or now - times[1] > _NATURAL_OBSERVATION_MAX_AGE):
        raise AuthorityError("child natural evidence has stale or unordered observation timestamps")
    if os.name == "nt":
        try:
            owner_birth = int(receipt["owner_started"]) / 10_000_000 - 11_644_473_600
        except (TypeError, ValueError, OverflowError) as error:
            raise AuthorityError("child natural evidence has an invalid original process birth") from error
        if not math.isfinite(owner_birth) or owner_birth > times[0]:
            raise AuthorityError("child natural evidence predates the original process birth")
    if (set(inventory) != {"schema", "observed_at", "host", "processes"}
            or inventory.get("schema") != "rubiksql.child-process-inventory/1"
            or inventory.get("host") != receipt["owner_host"]):
        raise AuthorityError("child natural evidence has an invalid process inventory")
    _check_child_processes(inventory.get("processes"), receipt, Path(continuation["previous_source_dir"]))


def _child_natural_evidence(continuation: Mapping[str, Any], receipt: Mapping[str, Any]) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Bind a fresh stopped observation to the genuine direct launch and failed knowledge stage."""
    raw, documents = _read_child_evidence(continuation)
    evidence, output = continuation["evidence"], Path(receipt["out_dir"])
    context, state, exited, verified = (documents[name] for name in
        ("launch_context", "pipeline_state", "pipeline_exit", "stopped_verification"))
    _child_direct_context(continuation, receipt, context, initial=True)
    if (any(context["launch_evidence"][name] != evidence[name] for name in context["launch_evidence"])
            or documents["previous_receipt_file"] != dict(receipt)
            or any(Path(evidence[name]["path"]).resolve() != output / filename for name, filename in
                   (("previous_receipt_file", ".paid-launch.json"), ("provider_assignment", ".gateway-experiment.json"),
                    ("pipeline_state", ".pipeline_state.json"), ("pipeline_exit", "_exit.json")))
            or documents["provider_assignment"].get("pin") != continuation["pin"]
            or not isinstance(documents["provider_assignment"].get("experiment_id"), str)
            or not documents["provider_assignment"]["experiment_id"].strip()
            or state.get("db_fingerprints") != receipt["inputs"].get("db_fingerprints")
            or not isinstance(state.get("run_identity"), dict)
            or state["run_identity"].get("prompt_release") != receipt["release"]
            or state["run_identity"].get("cache_ns") != receipt["namespace"]
            or any(key not in {"run_identity", "db_fingerprints", *CHILD_KNOWLEDGE_STAGES} for key in state)
            or type(exited.get("exit_code")) is not int or exited["exit_code"] == 0
            or not isinstance(exited.get("stage"), str) or exited["stage"] not in CHILD_KNOWLEDGE_STAGES):
        raise AuthorityError("child natural evidence changed its original execution or failed knowledge stage")
    try:
        wrapper, log = (raw[name].decode("utf-8-sig") for name in ("launch_wrapper", "launch_log"))
    except UnicodeError as error:
        raise AuthorityError("child natural evidence wrapper and log must retain readable original bytes") from error
    normalized = wrapper.replace("\\", "/").casefold()

    def mentions(value: str) -> bool:
        value = value.replace("\\", "/").casefold()
        return re.search(r"(?<![a-z0-9_./:\-])" + re.escape(value) + r"(?![a-z0-9_./:\-])", normalized) is not None

    required = [receipt["source_commit"], continuation["pin"],
                context["source"]["directory"], context["source"]["home"], context["binding"]["path"],
                context["paths"]["knowledge"], context["paths"]["databases"], context["paths"]["queries"]]
    if (any(not mentions(value) for value in required)
            or re.search(r"(?:^|[/\s'\"])eval_evolution\.py(?:$|[\s'\"])", normalized) is None
            or continuation["pin"] not in log or re.search(r"\bdry_run\s*=\s*true\b", log, re.IGNORECASE)
            or re.search(r"\bexit\s*=\s*" + re.escape(str(exited["exit_code"])) + r"\b", log, re.IGNORECASE) is None):
        raise AuthorityError("child natural evidence does not corroborate the original direct invocation and exit")
    for path in (context["source"]["interpreter"], context["paths"]["output"], context["paths"]["parent"],
                 context["paths"]["cache"]):
        target = Path(path)
        if mentions(path):
            continue
        roots = (Path(context["source"]["directory"]), output.parent, output.parent.parent)
        represented = False
        for root in roots:
            if root != target and root in target.parents:
                relative = str(target.relative_to(root)).replace("\\", "/").casefold()
                # PowerShell's original run expression concatenates 'runs/' and $proposalId.
                relative_present = mentions(relative)
                if target == output:
                    relative_present |= mentions(relative.rsplit("/", 1)[0] + "/") and mentions(target.name)
                represented |= mentions(str(root)) and relative_present
        if not represented:
            raise AuthorityError("child natural evidence changed a derived original invocation path")
    verification_fields = {"schema", "provenance", "observed_at", "reservation_id", "claim_commit", "previous_owner",
                           "receipt_phase", "original_owner_alive", "original_build_processes_remaining", "ready_exists",
                           "pre_answering_exists", "answer_file_count", "counted_slot_retained", "receipt_sha256",
                           "provider_assignment_sha256", "process_inventory_sha256", "launch_context_sha256"}
    if (set(verified) != verification_fields or verified.get("schema") != "rubiksql.child-natural-stopped-verification/1"
            or verified.get("provenance") != "observed_after_natural_exit"
            or any(verified.get(key) != receipt[key] for key in ("reservation_id", "claim_commit"))
            or verified.get("previous_owner") != continuation["previous_owner"]
            or verified.get("receipt_phase") != "claimed"
            or any(verified.get(key) is not False for key in
                   ("original_owner_alive", "ready_exists", "pre_answering_exists"))
            or any(type(verified.get(key)) is not int or verified[key] != 0
                   for key in ("original_build_processes_remaining", "answer_file_count"))
            or verified.get("receipt_sha256") != sha256_hex(canon(receipt))
            or any(verified.get(name + "_sha256") != evidence[name]["sha256"]
                   for name in ("provider_assignment", "process_inventory", "launch_context"))):
        raise AuthorityError("child natural evidence changed its fresh stopped verification")
    _check_natural_observation(continuation, receipt, documents)
    return raw, documents


def _child_operational_evidence(continuation: Mapping[str, Any], receipt: Mapping[str, Any]) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Bind a natural failed build to its actual helper and wrapper records."""
    raw, documents = _read_child_evidence(continuation)
    evidence = continuation["evidence"]
    direct = continuation["schema"] == SCHEMA_CHILD_DIRECT_OPERATIONAL_CONTINUATION
    origin = "launch_context" if direct else "original_request"
    request, previous, started, continued, terminal = (documents[name] for name in
        (origin, "previous_request", "helper_started", "helper_continued", "wrapper_terminal"))
    output = Path(receipt["out_dir"])
    if (any(not isinstance(request.get(key), dict) for key in ("source", "binding", "paths"))
            or not isinstance(previous.get("source"), dict)
            or not isinstance(previous.get("records"), str) or not Path(previous["records"]).is_absolute()
            or not isinstance(request.get("paths", {}).get("records"), str)
            or not Path(request["paths"]["records"]).is_absolute()
            or not isinstance(request.get("paths", {}).get("parent"), str)
            or not Path(request["paths"]["parent"]).is_absolute()):
        raise AuthorityError("child operational evidence has invalid execution records")
    records = Path(previous["records"])
    if direct:
        _child_direct_context(continuation, receipt, request, initial=False)
        if (set(previous) != {"schema", "launch_context", "source", "records", "continuation", "helper_sha256",
                             "incremental_plan_sha256", "runtime"}
                or set(previous["source"]) != {"directory", "commit", "interpreter", "home", "host"}
                or any(not isinstance(previous.get(key), str) or re.fullmatch(r"[0-9a-f]{64}", previous[key]) is None
                       for key in ("helper_sha256", "incremental_plan_sha256"))
                or not isinstance(previous.get("runtime"), dict)
                or any(started.get(key) != receipt[key] for key in OWNER_FIELDS)):
            raise AuthorityError("child operational evidence changed its direct-origin helper provenance")
    if (request.get("schema") != ("rubiksql.child-direct-launch-context/1" if direct else "rubiksql.new-knowledge-child/1")
            or request.get("proposal") != {"pin": continuation["pin"], "release": receipt["release"]}
            or request.get("namespace") != receipt["namespace"] or request.get("inputs") != receipt["inputs"]
            or request.get("knowledge_fingerprints") != receipt["knowledge"]["fingerprints"]
            or request["source"].get("commit") != receipt["source_commit"]
            or request["binding"].get("authority_uuid") != continuation["authority_uuid"]
            or request["paths"].get("output") != receipt["out_dir"]
            or request["paths"].get("knowledge") != receipt["knowledge"]["root"]
            or previous.get("schema") != ("rubiksql.child-direct-build-follow/1" if direct else "rubiksql.child-build-follow/1")
            or previous.get(origin) != evidence[origin]
            or previous["source"].get("commit") != continuation["previous_source_commit"]
            or previous["source"].get("directory") != continuation["previous_source_dir"]
            or previous["source"].get("host") != receipt["owner_host"]
            or started.get("request") != previous or started.get("request_sha256") != sha256_hex(canon(previous))
            or started.get("owner_pid") != receipt["owner_pid"]
            or continued != {**{key: receipt[key] for key in ("reservation_id", "claim_commit", "resume_commit", "resume_event")},
                             "receipt_sha256": sha256_hex(canon(receipt))}
            or documents["helper_stopped"] != {"error_type": "Refused", "retry_allowed": False}
            or any(Path(evidence[name]["path"]).resolve() != records / filename for name, filename in
                   (("helper_started", "started.json"), ("helper_continued", "continued.json"), ("helper_stopped", "stopped.json")))
            or documents["previous_receipt_file"] != dict(receipt)
            or Path(evidence["previous_receipt_file"]["path"]).resolve() != output / ".paid-launch.json"
            or Path(evidence["provider_assignment"]["path"]).resolve() != output / ".gateway-experiment.json"
            or documents["provider_assignment"].get("pin") != continuation["pin"]
            or not isinstance(documents["provider_assignment"].get("experiment_id"), str)
            or not documents["provider_assignment"]["experiment_id"].strip()):
        raise AuthorityError("child operational evidence changed its stopped helper or original execution")
    if (Path(evidence["pipeline_exit"]["path"]).resolve() != output / "_exit.json"
            or type(documents["pipeline_exit"].get("exit_code")) is not int or documents["pipeline_exit"]["exit_code"] == 0
            or (not isinstance(documents["pipeline_exit"].get("stage"), str)
                or documents["pipeline_exit"]["stage"] not in CHILD_KNOWLEDGE_STAGES if direct
                else documents["pipeline_exit"].get("stage") != "examples")
            or terminal.get("schema") != "rubiksql.child-continuation-wrapper/1" or terminal.get("phase") != "launch"
            or terminal.get("source") != continuation["previous_source_commit"]
            or terminal.get("request_sha256") != evidence["previous_request"]["sha256"]
            or any(terminal.get(key) != receipt[key] for key in ("reservation_id", "claim_commit"))
            or any(terminal.get(key) is not False for key in ("success", "new_reservation"))
            or type(terminal.get("python_exit")) is not int or terminal["python_exit"] == 0
            or any(not isinstance(terminal.get(key), str) or not terminal[key].strip() for key in ("started_utc", "finished_utc"))):
        raise AuthorityError("child operational evidence requires a naturally failed examples build and terminal wrapper")
    try:
        start, finish = (datetime.fromisoformat(terminal[key]) for key in ("started_utc", "finished_utc"))
        created, exited = started.get("created_at"), documents["pipeline_exit"].get("ts")
        valid_times = (start.utcoffset() is not None and finish.utcoffset() is not None
                       and all(type(value) in (int, float) and math.isfinite(value) for value in (created, exited))
                       and start.timestamp() <= created <= exited <= finish.timestamp())
    except (ValueError, OverflowError, OSError) as error:
        raise AuthorityError("child operational evidence has malformed terminal timestamps") from error
    if not valid_times:
        raise AuthorityError("child operational evidence has stale or unordered terminal timestamps")
    if direct and finish.timestamp() > time.time():
        raise AuthorityError("child operational evidence has a future terminal timestamp")
    return raw, documents


def _child_build_evidence(continuation: Mapping[str, Any], receipt: Mapping[str, Any]) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Read exact retained evidence and check its cancellation chain, without provider access."""
    evidence = continuation["evidence"]
    raw, documents = _read_child_evidence(continuation)
    request, audit, action, verified = (documents[name] for name in
                                       ("original_request", "abandon_audit", "abandon_action", "abandon_verified"))
    if (any(not isinstance(request.get(key), dict) for key in ("source", "binding", "paths"))
            or not isinstance(audit.get("build"), dict)
            or not isinstance(audit["build"].get("subtree"), list)
            or any(not isinstance(row, dict) for row in audit["build"]["subtree"])):
        raise AuthorityError("child continuation evidence has invalid execution records")
    output = Path(receipt["out_dir"])
    if (documents["previous_receipt_file"] != dict(receipt)
            or Path(evidence["previous_receipt_file"]["path"]).resolve() != output / ".paid-launch.json"
            or Path(evidence["provider_assignment"]["path"]).resolve() != output / ".gateway-experiment.json"
            or documents["provider_assignment"].get("pin") != continuation["pin"]
            or not isinstance(documents["provider_assignment"].get("experiment_id"), str)
            or not documents["provider_assignment"]["experiment_id"].strip()
            or documents["helper_stopped"] != {"error_type": "Refused", "retry_allowed": False}
            or request.get("schema") != "rubiksql.new-knowledge-child/1"
            or request.get("proposal") != {"pin": continuation["pin"], "release": receipt["release"]}
            or request.get("namespace") != receipt["namespace"] or request.get("inputs") != receipt["inputs"]
            or request.get("knowledge_fingerprints") != receipt["knowledge"]["fingerprints"]
            or request.get("source", {}).get("commit") != receipt["source_commit"]
            or request.get("source", {}).get("directory") != continuation["previous_source_dir"]
            or request.get("source", {}).get("host") != receipt["owner_host"]
            or request.get("binding", {}).get("authority_uuid") != continuation["authority_uuid"]
            or request.get("paths", {}).get("output") != receipt["out_dir"]
            or request.get("paths", {}).get("knowledge") != receipt["knowledge"]["root"]
            or not isinstance(request.get("paths", {}).get("parent"), str)
            or not Path(request["paths"]["parent"]).is_absolute()
            or not isinstance(request.get("paths", {}).get("records"), str)
            or not Path(request["paths"]["records"]).is_absolute()
            or Path(evidence["helper_stopped"]["path"]).resolve() != Path(request["paths"]["records"]) / "stopped.json"):
        raise AuthorityError("child continuation evidence changed the original execution or provider assignment")
    identity = {"proposal": continuation["pin"], "reservation": receipt["reservation_id"], "claim_commit": receipt["claim_commit"]}
    if (audit.get("schema") != "rubiksql.pre-ready-content-abandon/1"
            or any(audit.get(key) != value for key, value in identity.items())
            or audit.get("release") != receipt["release"] or audit.get("source_commit") != receipt["source_commit"]
            or audit.get("request_sha256") != evidence["original_request"]["sha256"]
            or audit.get("owner_pid") != receipt["owner_pid"] or audit.get("owner_started") != receipt["owner_started"]
            or any(audit.get(key) is not False for key in ("ready_exists", "authority_writes", "accepted"))
            or action.get("schema") != "rubiksql.owned-build-cancellation/1"
            or action.get("audit_sha256") != evidence["abandon_audit"]["sha256"]
            or type(action.get("taskkill_exit")) is not int or action["taskkill_exit"] != 0
            or any(action.get(key) is not False for key in ("owner_targeted", "shared_tunnel_targeted", "authority_writes"))
            or not isinstance(audit.get("build", {}).get("subtree"), list) or not audit["build"]["subtree"]
            or action.get("subtree_before") != audit["build"]["subtree"]
            or action.get("target_pid") not in [row.get("pid") for row in action["subtree_before"]]
            or verified.get("schema") != "rubiksql.pre-ready-abandon-verification/1"
            or any(verified.get(key) != value for key, value in identity.items())
            or verified.get("receipt_phase") != "claimed" or verified.get("stopped_error_type") != "Refused"
            or any(verified.get(key) is not False for key in ("original_owner_alive", "ready_exists", "knowledge_complete"))
            or any(type(verified.get(key)) is not int or verified[key] != 0
                   for key in ("original_build_processes_remaining", "answer_file_count"))
            or verified.get("audit_sha256") != evidence["abandon_audit"]["sha256"]
            or verified.get("action_sha256") != evidence["abandon_action"]["sha256"]
            or verified.get("stopped_sha256") != evidence["helper_stopped"]["sha256"]):
        raise AuthorityError("child continuation cancellation evidence is not the exact stopped execution chain")
    return raw, documents


def resume_child_build_claim(receipt: Mapping[str, Any], *, pin: str, out_dir: Path, snapshot: Any,
                             inputs: Mapping[str, Any], knowledge_root: Path, source_commit: str,
                             namespace: str, source_dir: Path,
                             continuation: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    """Transfer a stopped counted child before readiness without admitting another evaluation.

    Args:
        receipt: Original claimed receipt for cancellation/natural v1, or the latest operational receipt.
        pin: Original proposal identity, distinct from its release identity.
        out_dir: Original output directory containing the receipt and provider assignment.
        snapshot: Verified snapshot of this receipt's claim or latest resume and exact digest.
        inputs: Unchanged Mini-Dev 500 evaluation inputs.
        knowledge_root: Original partially built knowledge root.
        source_commit: Exact clean execution checkout commit authorized by the user.
        namespace: Original epoch namespace.
        source_dir: Physical execution checkout; the original checkout remains preserved.
        continuation: Closed cancellation, natural or operational record binding this receipt and evidence.

    Returns:
        Updated receipt and snapshot after one confirmed register-event publication.

    Raises:
        AuthorityError: Evidence, scope, source, owner inventory or current authority
            differs, readiness/answering started, or the claim already continued.
        GitAuthorityConflict: A competing publication wins; no transfer is retried.
    """
    from .authority import Authority
    from .binding import load_binding
    from . import evolve as EV
    from .snapshot import write_snapshot

    binding, output = load_binding(), Path(out_dir).resolve()
    continuation = _child_build_continuation(continuation, receipt, source_commit, binding.authority_uuid)
    natural = continuation["schema"] == SCHEMA_CHILD_NATURAL_CONTINUATION
    direct = continuation["schema"] in (SCHEMA_CHILD_NATURAL_CONTINUATION, SCHEMA_CHILD_DIRECT_OPERATIONAL_CONTINUATION)
    operational = continuation["schema"] in (SCHEMA_CHILD_OPERATIONAL_CONTINUATION, SCHEMA_CHILD_DIRECT_OPERATIONAL_CONTINUATION)
    if (receipt.get("phase") != "claimed" or receipt.get("fresh_knowledge") or receipt.get("requires_ready") is not True
            or any(key in receipt for key in ("answer_resume_commit", "execution_commit"))
            or (operational and any(key not in receipt for key in ("resume_commit", "resume_event", "execution_source_commit")))
            or (not operational and any(key in receipt for key in ("resume_commit", "execution_source_commit")))
            or continuation["pin"] != pin or receipt.get("out_dir") != str(output)
            or receipt.get("namespace") != namespace or receipt.get("inputs") != dict(inputs)
            or inputs.get("question_set") != EV.EVOLUTION_QUESTION_SET
            or receipt.get("knowledge", {}).get("root") != str(knowledge_root.resolve())
            or snapshot.authority_commit != receipt.get("resume_commit", receipt.get("claim_commit"))
            or snapshot.digest != receipt.get("snapshot_digest")
            or snapshot.release != receipt.get("release")):
        raise AuthorityError("child continuation differs from its original claimed execution")
    verify_remote_claim(receipt)
    validator = _child_operational_evidence if operational else (_child_natural_evidence if natural else _child_build_evidence)
    raw, documents = validator(continuation, receipt)
    origin = "launch_context" if direct else "original_request"
    request = documents[origin]
    record_directories = {Path(request["paths"]["records"])}
    if operational:
        record_directories.add(Path(documents["previous_request"]["records"]))

    def before_readiness() -> None:
        forbidden = [*PRE_ANSWERING_RECORDS, ".build-paused.json", ".knowledge-ready.json"]
        if (any((output / name).exists() for name in forbidden)
                or any((records / name).exists() for records in record_directories for name in
                       ("ready.json", "completed.json", "answer-go.json", "decision-observed.json"))
                or answering_artifacts(output, inputs["db_fingerprints"])):
            raise AuthorityError("child continuation refuses readiness or answering artifacts")

    def before_transfer() -> None:
        before_readiness()
        if direct:
            _child_direct_context(continuation, receipt, request, initial=natural)
        if natural:
            _check_natural_observation(continuation, receipt, documents)
        sources = [(source_dir, source_commit),
                   (Path(continuation["previous_source_dir"]), continuation["previous_source_commit"])]
        if direct and operational:
            sources.append((Path(request["source"]["directory"]), request["source"]["commit"]))
        for directory, commit in sources:
            options = dict(cwd=directory, capture_output=True, text=True, check=True, timeout=30)
            if (directory.is_symlink() or (direct and directory.resolve() != directory)
                    or subprocess.run(["git", "rev-parse", "HEAD"], **options).stdout.strip() != commit
                    or subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], **options).stdout.strip()):
                raise AuthorityError("child continuation requires both exact clean source checkouts")
            (_assert_stopped_direct_build if direct else _assert_stopped_build)(receipt, directory)
        if any((direct and Path(continuation["evidence"][name]["path"]).resolve()
                != Path(continuation["evidence"][name]["path"]))
               or Path(continuation["evidence"][name]["path"]).read_bytes() != body for name, body in raw.items()):
            raise AuthorityError("child continuation evidence changed during stopped-owner inspection")
        before_readiness()

    before_transfer()
    parent = EV.read_run(Path(request["paths"]["parent"]))
    parent_identity = {"run": parent.name, "release": parent.release, "namespace": parent.ns,
                       "created_at": parent.created_at, "manifest_sha256": parent.manifest_sha, "result_files": parent.result_files}
    if (sha256_hex(canon(parent_identity)) != request.get("parent_sha256") or parent.release_problem(store=None)
            or parent.identity.get("question_set") != EV.EVOLUTION_QUESTION_SET or len(parent.rows) != EV.EVOLUTION_SET.count
            or any(EV.M.EvalRow(row).ex is None for row in parent.rows.values())):
        raise AuthorityError("child continuation changed its complete paired parent")
    owner = {"owner_host": socket.gethostname(), "owner_pid": os.getpid(), "owner_started": _process(os.getpid())[1]}
    with Authority.write(binding=binding) as writer:
        reservations = [event["payload"] for event in writer.events("eval-reserve")
                        if event["payload"].get("id") == receipt["reservation_id"]]
        transfers = [event["payload"] for event in writer.events("register")
                     if event["payload"].get("schema") in {SCHEMA_BUILD_RESUME, SCHEMA_CHILD_BUILD_RESUME,
                                                          SCHEMA_CHILD_OPERATIONAL_RESUME, SCHEMA_ANSWER_RESUME}]
        if (len(reservations) != 1 or reservations[0].get("counted") is not True or reservations[0].get("fresh_knowledge")
                or reservations[0].get("requires_ready") is not True
                or reservations[0].get("pin") != pin or reservations[0].get("proposal") != pin
                or reservations[0].get("parent") != parent_identity
                or writer.head().generation != reservations[0].get("generation")
                or any((not operational and row.get("reservation_id") == receipt["reservation_id"])
                       or row.get("continuation", {}).get("request_id") == continuation["request_id"] for row in transfers)
                or any(event["payload"].get("reservation_id") == receipt["reservation_id"]
                       for kind in ("eval-ready", "eval-attach") for event in writer.events(kind))):
            raise AuthorityError("counted child already advanced or its reserved generation/parent changed")
        reservation = reservations[0]
        with writer.reader() as reader:
            if direct:
                claims = reservation_claims(reader, reservation, receipt["reservation_commit"])
                if (len(claims) != 1
                        or request["original_owner"] != {key: claims[0].get(key) for key in OWNER_FIELDS}):
                    raise AuthorityError("child natural evidence changed its original committed claim")
                if natural:
                    created, exited = claims[0].get("created_at"), documents["pipeline_exit"].get("ts")
                    if type(created) not in (int, float) or not math.isfinite(created) or created > exited:
                        raise AuthorityError("child natural evidence exit predates its committed claim")
            if operational:
                claims = reservation_claims(reader, reservation, receipt["reservation_commit"])
                if len(claims) != 1:
                    raise AuthorityError("child operational continuation changed its original claim")
                attempt, predecessor = _child_build_chain(binding, reader, receipt, reservation, claims[0])
                if continuation["build_attempt"] != attempt + 1:
                    raise AuthorityError("child operational continuation must increment the root build count")
                if (documents["previous_request"].get("continuation") != predecessor["continuation"]
                        or any(continuation["evidence"][name] != predecessor["continuation"]["evidence"].get(name)
                               for name in (origin, "provider_assignment"))
                        or (direct and predecessor["continuation"]["schema"] not in
                            (SCHEMA_CHILD_NATURAL_CONTINUATION, SCHEMA_CHILD_DIRECT_OPERATIONAL_CONTINUATION))):
                    raise AuthorityError("child operational continuation changed its predecessor request or provider assignment")
                if direct and predecessor.get("created_at", math.inf) > documents["pipeline_exit"]["ts"]:
                    raise AuthorityError("child operational evidence exit predates its admitted owner")
                for row in transfers:
                    if row.get("reservation_id") == receipt["reservation_id"] and row.get("schema") == SCHEMA_CHILD_OPERATIONAL_RESUME:
                        item = row["continuation"]["evidence"]["previous_request"]
                        previous_raw = Path(item["path"]).read_bytes()
                        if sha256_hex(previous_raw) != item["sha256"]:
                            raise AuthorityError("child operational continuation lost a previous helper record")
                        record_directories.add(Path(json.loads(previous_raw)["records"]))
            if operational or direct:
                for event in reader.events("eval-reserve"):
                    other = event["payload"]
                    if (other.get("id") != receipt["reservation_id"]
                            and Path(other.get("knowledge", {}).get("root", "")).resolve() == knowledge_root.resolve()):
                        raise AuthorityError("child operational continuation has ambiguous root build history")
            proposal = reader.proposal(pin)
            plan = EV.plan_view(proposal)
            if (reader.epoch()["epoch"] != reservation.get("epoch") or proposal["release"] != receipt["release"]
                    or plan.base_parent != reservation["generation"] or plan.namespace != namespace
                    or parent.release != reservation["generation"] or parent.ns != namespace):
                raise AuthorityError("child continuation changed its epoch, proposal or paired generation")
        used = reservation.get("attempt", {}).get("used")
        if (type(used) is not int or used < 0 or (not operational and
                (type(documents["stopped_verification" if natural else "abandon_verified"].get("counted_slot_retained")) is not int
                 or documents["stopped_verification" if natural else "abandon_verified"]["counted_slot_retained"] != used + 1))):
            raise AuthorityError("child continuation changed its counted reservation slot")
        before_transfer()
        history = output / ".child_build_history" / continuation["previous_receipt_sha256"]
        if direct and history.resolve() != history:
            raise AuthorityError("child continuation history requires an owned physical directory")
        history.mkdir(parents=True, exist_ok=True)
        retained = {**raw, "authorization": (canon(continuation) + "\n").encode("utf-8")}
        for name, body in retained.items():
            extension = {"launch_wrapper": ".ps1", "launch_log": ".log"}.get(name, ".json")
            path = history / (name + extension)
            if path.exists():
                if path.is_symlink() or path.read_bytes() != body:
                    raise AuthorityError("retained child continuation evidence differs; preserve both records")
            else:
                with path.open("xb") as stream:
                    stream.write(body)
                    stream.flush()
                    os.fsync(stream.fileno())
        payload = {"schema": SCHEMA_CHILD_OPERATIONAL_RESUME if operational else SCHEMA_CHILD_BUILD_RESUME,
                   "reservation_id": receipt["reservation_id"],
                   "claim_commit": receipt["claim_commit"], "token": receipt["token"], "previous_receipt": dict(receipt),
                   "previous_owner": {key: receipt[key] for key in OWNER_FIELDS},
                   "previous_receipt_sha256": continuation["previous_receipt_sha256"], "created_at": time.time(),
                   **owner, "execution_source_commit": source_commit, "continuation": continuation}
        if operational:
            payload["build_attempt"] = continuation["build_attempt"]
        writer._require_writer()
        event_id = writer._event("register", pin, payload, "agent:paid-launch")
        commit = _commit(writer)
        with writer.reader() as reader:
            resumed_snapshot = write_snapshot(reader, pin, out_dir=output)
    resumed = {**receipt, **owner, "resume_commit": commit, "resume_event": event_id,
               "execution_source_commit": source_commit, "snapshot_digest": resumed_snapshot.digest}
    _save_receipt(output / ".paid-launch.json", resumed)
    _verify_committed_claim(binding, canon(resumed))
    return resumed, resumed_snapshot


def resume_build_claim(receipt: Mapping[str, Any], *, pin: str, out_dir: Path, snapshot: Any,
                       inputs: Mapping[str, Any], knowledge_root: Path, source_commit: str,
                       namespace: str, source_dir: Path,
                       continuation: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], Any]:
    """Explicitly transfer a stopped fresh build's owner; retain its one reservation and claim.

    Only a terminal failed build checkpoint can resume. An interrupted admission, a
    live owner, unknown workers or any run that reached answering requires inspection
    and is never automatically relaunched. A competing resumer loses the Git lease.
    A separately supplied, receipt-bound user authorization can waive only the
    build retry limit and record a corrected execution source. Historical source,
    reservation, claim and snapshots remain unchanged.
    """
    from .authority import Authority
    from .binding import load_binding
    from .snapshot import write_snapshot

    if inputs.get("question_set") != EVOLUTION_SET.digest:
        raise AuthorityError("new baseline continuation requires Mini-Dev 500")
    output = Path(out_dir).resolve()
    if continuation is not None:
        continuation = check_build_continuation(continuation, receipt, source_commit)
    if (receipt.get("phase") != "claimed" or not receipt.get("fresh_knowledge") or not receipt.get("requires_ready")
            or receipt.get("out_dir") != str(output) or receipt.get("release") != pin
            or (continuation is None and receipt.get("execution_source_commit", receipt.get("source_commit")) != source_commit)
            or receipt.get("namespace") != namespace
            or receipt.get("inputs") != dict(inputs) or receipt.get("knowledge", {}).get("root") != str(knowledge_root.resolve())
            or snapshot.authority_commit != receipt.get("resume_commit", receipt.get("claim_commit"))
            or snapshot.digest != receipt.get("snapshot_digest")):
        raise AuthorityError("build resume differs from its existing fresh claim")
    paused = output / ".build-paused.json"
    if (not paused.is_file() or json.loads(paused.read_text(encoding="utf-8")).get("receipt_sha256") != sha256_hex(canon(receipt))
            or (output / ".prompt_manifest.json").exists() or (output / ".evaluation_inputs.json").exists()
            or (output / ".live_preflight.json").exists()):
        raise AuthorityError("build resume needs its terminal failed-build checkpoint; answering cannot be relaunched")
    binding = load_binding()
    _verify_committed_claim(binding, canon(receipt))
    _assert_stopped_build(receipt, source_dir)
    if continuation is not None:
        old_source = Path(continuation["previous_source_dir"])
        previous_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=old_source, capture_output=True,
                                        text=True, check=True, timeout=30).stdout.strip()
        if previous_commit != continuation["previous_source_commit"]:
            raise AuthorityError("preserved continuation checkout differs from its previous source pin")
        if old_source.resolve() != source_dir.resolve():
            _assert_stopped_build(receipt, old_source)
    owner = {"owner_host": socket.gethostname(), "owner_pid": os.getpid(), "owner_started": _process(os.getpid())[1]}
    with Authority.write(binding=binding) as writer:
        reservations = [event["payload"] for event in writer.events("eval-reserve")
                        if event["payload"].get("id") == receipt["reservation_id"]]
        resumes = [event for event in writer.events("register") if event["payload"].get("schema") == SCHEMA_BUILD_RESUME
                   and event["payload"].get("reservation_id") == receipt["reservation_id"]]
        if (len(reservations) != 1 or writer.head().generation != reservations[0].get("generation")
                or reservations[0].get("counted") is not False
                or (resumes[-1]["object_id"] if resumes else None) != receipt.get("resume_event")
                or any(event["payload"].get("reservation_id") == receipt["reservation_id"] for event in writer.events("eval-ready"))):
            raise AuthorityError("build claim already advanced or its generation changed; follow the existing execution")
        if len(resumes) >= 3 and continuation is None:
            raise AuthorityError("fresh knowledge root has used its four builds; preserve the existing claim")
        history = output / ".build_history" / (sha256_hex(canon(receipt)) + ".json")
        retained = {"receipt": dict(receipt), "paused": json.loads(paused.read_text(encoding="utf-8"))}
        history.parent.mkdir(exist_ok=True)
        if history.exists():
            if json.loads(history.read_text(encoding="utf-8")) != retained:
                raise AuthorityError("retained build checkpoint differs; preserve both records")
        else:
            with history.open("x", encoding="utf-8") as stream:
                stream.write(canon(retained) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        extra: dict[str, Any] = ({"execution_source_commit": source_commit}
                                if continuation is not None or "execution_source_commit" in receipt else {})
        if continuation is not None:
            extra["continuation"] = continuation
        writer._require_writer()
        event_id = writer._event("register", pin, {"schema": SCHEMA_BUILD_RESUME,
            "reservation_id": receipt["reservation_id"], "claim_commit": receipt["claim_commit"], "token": receipt["token"],
            "previous_owner": {key: receipt[key] for key in OWNER_FIELDS}, "previous_receipt_sha256": sha256_hex(canon(receipt)),
            "build_attempt": len(resumes) + 2, "created_at": time.time(), **owner, **extra}, "agent:paid-launch")
        commit = _commit(writer)
        with writer.reader() as reader:
            resumed_snapshot = write_snapshot(reader, pin, out_dir=output)
    resumed = {**receipt, **owner, "resume_commit": commit, "resume_event": event_id, "snapshot_digest": resumed_snapshot.digest}
    if "execution_source_commit" in extra:
        resumed["execution_source_commit"] = source_commit
    _save_receipt(output / ".paid-launch.json", resumed)
    paused.unlink()
    _verify_committed_claim(binding, canon(resumed))
    return resumed, resumed_snapshot


def check_answer_continuation(continuation: Mapping[str, Any], receipt: Mapping[str, Any],
                              source_commit: str) -> dict[str, Any]:
    """Bind an explicit user-authorized answering continuation to one ready receipt and source.

    Args:
        continuation: Operator-supplied record with verbatim authorization, authority,
            claim, readiness commit, previous receipt hash, old checkout and corrected source.
        receipt: Preserved current ready receipt; its reservation, claim and readiness stay.
        source_commit: Validated execution checkout's exact Git commit.

    Returns:
        A detached authorization record suitable for the answering resume event.

    Raises:
        AuthorityError: The explicit authorization or any bound identity is absent or differs.
    """
    from .binding import load_binding

    fields = {"schema", "request_id", "authorization", "authority_uuid", "reservation_id", "claim_commit",
              "execution_commit", "previous_receipt_sha256", "previous_source_commit", "source_commit",
              "previous_source_dir"}
    if (not isinstance(continuation, Mapping) or set(continuation) != fields
            or continuation.get("schema") != SCHEMA_ANSWER_CONTINUATION
            or not isinstance(continuation.get("authorization"), str) or not continuation["authorization"].strip()
            or not isinstance(continuation.get("request_id"), str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", continuation["request_id"]) is None
            or continuation.get("authority_uuid") != load_binding().authority_uuid
            or any(continuation.get(key) != receipt.get(key) for key in ("reservation_id", "claim_commit", "execution_commit"))
            or continuation.get("previous_receipt_sha256") != sha256_hex(canon(receipt))
            or continuation.get("previous_source_commit") != receipt.get("execution_source_commit", receipt.get("source_commit"))
            or continuation.get("source_commit") != source_commit
            or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
            or not isinstance(continuation.get("previous_source_dir"), str)
            or not Path(continuation["previous_source_dir"]).is_absolute()):
        raise AuthorityError("answering continuation requires explicit authorization bound to this ready receipt, authority and source")
    return dict(continuation)


def answering_artifacts(out_dir: Path, databases: Iterable[str]) -> list[str]:
    """Return the answering outputs that prove a run started answering questions."""
    names = [f"{db}.json" for db in databases] + [".resume_health.json"]
    return sorted(name for name in names if (Path(out_dir) / name).exists())


def resume_answering_claim(receipt: Mapping[str, Any], *, pin: str, out_dir: Path, snapshot: Any,
                           inputs: Mapping[str, Any], knowledge: Mapping[str, Any], source_commit: str,
                           namespace: str, source_dir: Path,
                           continuation: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    """Explicitly transfer a stopped ready baseline's owner before any question was answered.

    The one uncounted reservation, claim, committed readiness, ready snapshot and provider
    assignment stay. Only a receipt-bound user authorization admits it; a live owner,
    unknown workers, any saved answer or a changed claim refuses. The previous
    pre-answering records move into ``.answer_history/`` so the corrected source
    prepares its own manifest and live preflight.
    """
    from .authority import Authority
    from .binding import load_binding

    if inputs.get("question_set") != EVOLUTION_SET.digest:
        raise AuthorityError("new baseline continuation requires Mini-Dev 500")
    output = Path(out_dir).resolve()
    continuation = check_answer_continuation(continuation, receipt, source_commit)
    if (receipt.get("phase") != "ready" or receipt.get("fresh_knowledge") is not True or receipt.get("requires_ready") is not True
            or receipt.get("out_dir") != str(output) or receipt.get("release") != pin
            or receipt.get("namespace") != namespace or receipt.get("inputs") != dict(inputs)
            or receipt.get("knowledge") != {key: knowledge.get(key) for key in ("root", "fingerprints")}
            or snapshot.authority_commit != receipt.get("execution_commit")
            or snapshot.digest != receipt.get("snapshot_digest")):
        raise AuthorityError("answering continuation differs from its committed ready fresh baseline")

    def refuse_saved_answers() -> None:
        started = answering_artifacts(output, inputs["db_fingerprints"])
        if started:
            raise AuthorityError("answering already saved results (" + ", ".join(started) + "); inspect and follow that execution")

    refuse_saved_answers()
    binding = load_binding()
    _verify_committed_claim(binding, canon(receipt))
    _assert_stopped_build(receipt, source_dir)
    old_source = Path(continuation["previous_source_dir"])
    previous_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=old_source, capture_output=True,
                                    text=True, check=True, timeout=30).stdout.strip()
    if previous_commit != continuation["previous_source_commit"]:
        raise AuthorityError("preserved answering checkout differs from its previous source pin")
    if old_source.resolve() != source_dir.resolve():
        _assert_stopped_build(receipt, old_source)
    refuse_saved_answers()  # A worker that flushed while the inventory ran is caught here.
    owner = {"owner_host": socket.gethostname(), "owner_pid": os.getpid(), "owner_started": _process(os.getpid())[1]}
    previous = sha256_hex(canon(receipt))
    with Authority.write(binding=binding) as writer:
        reservations = [event["payload"] for event in writer.events("eval-reserve")
                        if event["payload"].get("id") == receipt["reservation_id"]]
        readies = [event for event in writer.events("eval-ready") if event["payload"].get("reservation_id") == receipt["reservation_id"]]
        resumes = [event for event in writer.events("register") if event["payload"].get("schema") == SCHEMA_ANSWER_RESUME
                   and event["payload"].get("reservation_id") == receipt["reservation_id"]]
        if (len(reservations) != 1 or writer.head().generation != reservations[0].get("generation")
                or reservations[0].get("counted") is not False or len(readies) != 1
                or (resumes[-1]["object_id"] if resumes else None) != receipt.get("answer_resume_event")):
            raise AuthorityError("ready claim already advanced or its generation changed; follow the existing execution")
        history = output / ".answer_history" / previous
        history.mkdir(parents=True, exist_ok=True)
        retained = {}
        for name in PRE_ANSWERING_RECORDS:
            source = output / name
            if not source.is_file():
                continue
            body = source.read_bytes()
            target = history / name
            if target.exists():
                if target.read_bytes() != body:
                    raise AuthorityError("retained pre-answering record differs; preserve both records")
            else:
                with target.open("xb") as stream:
                    stream.write(body)
                    stream.flush()
                    os.fsync(stream.fileno())
            retained[name] = sha256_hex(body)
        receipt_copy = history / ".paid-launch.json"
        if not receipt_copy.exists():
            with receipt_copy.open("x", encoding="utf-8") as stream:
                stream.write(canon(receipt) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        elif json.loads(receipt_copy.read_text(encoding="utf-8")) != dict(receipt):
            raise AuthorityError("retained ready receipt differs; preserve both records")
        refuse_saved_answers()
        writer._require_writer()
        event_id = writer._event("register", pin, {"schema": SCHEMA_ANSWER_RESUME,
            "reservation_id": receipt["reservation_id"], "claim_commit": receipt["claim_commit"], "token": receipt["token"],
            "execution_commit": receipt["execution_commit"], "previous_owner": {key: receipt[key] for key in OWNER_FIELDS},
            "previous_receipt_sha256": previous, "answer_attempt": len(resumes) + 2, "retained_records": retained,
            "created_at": time.time(), **owner, "execution_source_commit": source_commit, "continuation": continuation},
            "agent:paid-launch")
        commit = _commit(writer)
    resumed = {**receipt, **owner, "answer_resume_commit": commit, "answer_resume_event": event_id,
               "execution_source_commit": source_commit}
    _save_receipt(output / ".paid-launch.json", resumed)
    for name in retained:
        (output / name).unlink()
    _verify_committed_claim(binding, canon(resumed))
    return resumed, snapshot


def finalize_knowledge(receipt: Mapping[str, Any], *, pin: str, knowledge: Mapping[str, Any],
                       inputs: Mapping[str, Any], out_dir: Path) -> tuple[dict[str, Any], Any]:
    """Commit built knowledge, revalidating the same claim after CAS contention.

    Publication retries never rebuild knowledge or start answering. Each attempt
    checks the live owner, original admission and completed workspace fingerprints.
    """
    from .authority import Authority
    from .binding import load_binding
    from .snapshot import write_snapshot
    from rubiksql_hb.embedding import EMBEDDING_MODEL
    from rubiksql_hb.knowledge_identity import knowledge_fingerprints

    if receipt.get("phase") != "claimed" or not receipt.get("requires_ready") or not _owned_process(receipt):
        raise AuthorityError("final knowledge requires the live original claimed build")
    if (pin != receipt.get("release") and not pin.startswith("p")) or Path(out_dir).resolve() != Path(receipt["out_dir"]):
        raise AuthorityError("final knowledge changed the claimed pin or output")
    binding = load_binding()
    _verify_committed_claim(binding, canon(receipt))
    _checked_build_plan(knowledge, inputs, receipt["release"], fresh=False)
    if knowledge.get("root") != receipt.get("knowledge", {}).get("root"):
        raise AuthorityError("final knowledge changed the claimed root")
    source_commit = receipt.get("resume_commit", receipt["claim_commit"])
    with Authority.read(binding=binding, commit=source_commit) as reader:
        context = _admission_context(reader, pin, receipt["release"])
        if context["release"] != receipt["release"]:
            raise AuthorityError("final knowledge changed the claimed release")
        source_events = reader.events()

    def finalize(writer: Any) -> tuple[str, Any]:
        if not _owned_process(receipt):
            raise AuthorityError("final knowledge requires the live original claimed build")
        _check_admission_context(writer, commit=source_commit, context=context,
                                 events=source_events, pin=pin, release=receipt["release"])
        fingerprints = knowledge_fingerprints(Path(knowledge["root"]), inputs["db_fingerprints"], embedding_model=EMBEDDING_MODEL)
        if fingerprints != knowledge.get("fingerprints"):
            raise AuthorityError("final knowledge fingerprints do not match the completed workspaces")
        reservations = [event["payload"] for event in writer.events("eval-reserve")
                        if event["payload"].get("id") == receipt.get("reservation_id")]
        claims = reservation_claims(writer, reservations[0], receipt.get("reservation_commit")) if len(reservations) == 1 else []
        if (len(reservations) != 1 or len(claims) != 1 or claims[0].get("token") != receipt.get("token")
                or writer.head().generation != reservations[0].get("generation")
                or reservations[0].get("inputs") != dict(inputs)):
            raise AuthorityError("final knowledge differs from the original current-generation claim")
        if any(event["payload"].get("reservation_id") == receipt.get("reservation_id") for event in writer.events("eval-ready")):
            raise AuthorityError("this claim already committed its final knowledge; follow its execution")
        writer._require_writer()
        writer._event("eval-ready", pin, {"schema": "rubiksql.paid-ready/1", "reservation_id": receipt["reservation_id"],
            "claim_commit": receipt["claim_commit"], "token": receipt["token"], "knowledge": dict(knowledge), "inputs": dict(inputs),
            "created_at": time.time()}, "agent:paid-launch")
        commit = _commit(writer)
        with writer.reader() as reader:
            snapshot = write_snapshot(reader, pin, out_dir=Path(out_dir))
        return commit, snapshot

    commit, snapshot = Authority.transact(finalize, binding=binding)
    ready = {**receipt, "phase": "ready", "execution_commit": commit, "snapshot_digest": snapshot.digest,
             "knowledge": {key: knowledge[key] for key in ("root", "fingerprints")}}
    _save_receipt(Path(out_dir) / ".paid-launch.json", ready)
    _verify_committed_claim(binding, canon(ready))
    return ready, snapshot
