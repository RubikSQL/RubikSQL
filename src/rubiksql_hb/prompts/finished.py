"""Register complete, previously uncounted evidence without rewriting its files."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

from . import evolve as EV
from .canon import canon, sha256_hex
from .types import AuthorityError


def _prompt_execution_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """Compare complete prompt semantics apart from later publication attribution.

    The proposal association is assigned by publication, after an uncounted local
    snapshot can exist. Its manifest fingerprint is derived from that association;
    compare every actual manifest/capsule/trust field instead of that redundant hash.
    """
    from .authority import semantic_row

    result = semantic_row(row)
    result["manifest"]["meta"].pop("proposal", None)
    result.get("fingerprints", {}).pop("manifest", None)
    return result


def run_identity(run: EV.RunView) -> dict[str, Any]:
    """Return the immutable files that identify a measurement."""
    return {"manifest_sha256": run.manifest_sha, "result_files": run.result_files}


def _retained_parent(record: Mapping[str, Any], child: EV.RunView, parent: EV.RunView) -> bool:
    """Match a moved pair by producer-retained file hashes, never a new path claim."""
    if Path(str(record.get("parent_run") or "")).resolve() == parent.path.resolve():
        return True
    if record.get("parent") == run_identity(parent):
        return True
    # G12's exact-parent replay retained both original parent file hashes on
    # every consumed candidate. These hashes survive transport unchanged.
    if record.get("candidate_pool") != "exact-parent":
        return False
    for row in child.rows.values():
        debug = row.get("candidates_debug") or []
        sources = row.get("saved_candidate_sources")
        if sources is None:
            sources = [candidate.get("saved_candidate_source") or {} for candidate in debug]
        if len(debug) != 3 or not isinstance(sources, list) or len(sources) != 3:
            return False
        for slot, (candidate, source) in enumerate(zip(debug, sources), 1):
            result = row["db_id"] + ".json"
            digest = parent.result_files.get(result) or parent.result_files.get("pass1/" + result)
            if (not digest or source.get("run") != record.get("parent_run") or source.get("slot") != slot
                    or source.get("manifest_sha256") != parent.manifest_sha
                    or source.get("result_sha256") != digest
                    or source.get("cache_source") != candidate.get("cache_source")):
                return False
    return True


def verify_registered_run(child: EV.RunView, parent: EV.RunView, evidence: Mapping[str, Any], *, store: Any) -> None:
    """Deep-check finished evidence against its proposal, source and actual knowledge.

    Args:
        child: Unchanged, complete measured run.
        parent: Exact paired control selected at registration.
        evidence: Registration's immutable expected identities.
        store: Current verified authority history, including the proposal.

    Raises:
        EV.EvolveError: Scope, configuration, snapshot, source, knowledge or pairing differs.
        ValueError: Recorded source cannot reconstruct its role contracts.
    """
    from ..implementation_provenance import verify_role_records
    from .. import cache_lineage
    from ..knowledge_identity import knowledge_fingerprints, recorded_knowledge_path
    from ..lab_ingest import run_snapshot
    from ..llm_runtime import EMBEDDING_MODEL
    from .authority import semantic_row
    from .entries import KNOBS
    from .paid_launch import evaluation_configuration
    from .runtime import load_release
    from .store import Store

    if store is None:
        raise EV.EvolveError("finished registration deep verification requires authority history")
    body = store.proposal(str(evidence["proposal"])).data
    plan = EV.plan_view(body)
    EV.require_evolution_run(child)
    EV.require_evolution_run(parent)
    for run in (child, parent):
        if (len(EV.read_rows(run.path)) != EV.EVOLUTION_SET.count
                or any(EV.M.EvalRow(row).ex not in (0.0, 1.0) for row in run.rows.values())):
            raise EV.EvolveError("finished run requires exactly 500 distinct, binary-scored question rows")
    if EV._coverage(parent, child)["status"] != "full":
        raise EV.EvolveError("finished run and parent have different complete question or gold identities")
    questions = json.loads((Path(__file__).resolve().parents[3] / EV.EVOLUTION_SET.path).read_bytes())
    databases_by_question = {f"Q{int(row['question_id']):05d}": row["db_id"] for row in questions}
    if any(run.rows[qid].get("db_id") != database for run in (child, parent)
           for qid, database in databases_by_question.items()):
        raise EV.EvolveError("finished run question/database scope differs from Mini-Dev 500")
    if (evidence.get("schema") != "rubiksql.finished-registration/1"
            or evidence.get("child") != run_identity(child) or evidence.get("parent") != run_identity(parent)
            or evidence.get("plan_digest") != plan.plan_digest
            or child.release != body["release"] or parent.release != plan.base_parent
            or child.ns != plan.namespace or parent.ns != plan.namespace
            or child.identity != parent.identity
            or abs(child.created_at - parent.created_at) > plan.max_age_days * 86400
            or child.manifest.get("paid_launch") or (child.path / ".paid-launch.json").exists()):
        raise EV.EvolveError("finished run scope, parent pairing or original uncounted identity differs")
    if child.release_problem(store=store) or parent.release_problem(store=store):
        raise EV.EvolveError("finished run or parent has mixed, diagnostic or changed release rows")
    snapshot = run_snapshot(child.path, child.manifest)
    if snapshot.digest != evidence.get("snapshot_digest") or snapshot.release != body["release"]:
        raise EV.EvolveError("finished run snapshot digest or release differs")
    measured_store = Store(package=snapshot.path)
    measured = load_release(child.release, stores=measured_store, allow_retired=True)
    published = load_release(child.release, stores=store, allow_retired=True)
    fields = ("refs", "entries", "rules", "content_refs")
    if any(measured.manifest.get(key) != published.manifest.get(key) for key in fields):
        raise EV.EvolveError("finished snapshot closure differs from published proposal")
    if any(_prompt_execution_row(measured.row(name)) != _prompt_execution_row(published.row(name)) for name in measured.refs):
        raise EV.EvolveError("finished snapshot prompt content differs from published proposal")
    for entity, rows in measured.typed_rows.items():
        for object_id, row in rows.items():
            if semantic_row(row) != semantic_row(published.typed_rows.get(entity, {}).get(object_id, {})):
                raise EV.EvolveError("finished snapshot typed content differs from published proposal")
    # A local hypothetical snapshot may have a different publication timestamp,
    # but its scientific plan must be exactly the one now being registered.
    declared = child.manifest.get("proposal") or measured.manifest.get("proposal")
    if declared is not None:
        local = measured_store.proposal(str(declared)).data
        if local.get("plan_digest") != plan.plan_digest or local.get("release") != body["release"]:
            raise EV.EvolveError("finished run was measured under a different proposal plan")
    if child.manifest.get("plan_digest") not in (None, plan.plan_digest):
        raise EV.EvolveError("finished run plan digest differs")
    configuration = evaluation_configuration(snapshot)
    identity = child.identity
    judge = identity.get("judge")
    if judge == KNOBS["judge_model"].default:
        judge = measured.knob_settings()["judge_model"]
    actual = {"candidates": identity.get("candidates"), "judge": judge}
    if (actual != configuration or evidence.get("configuration") != configuration
            or identity.get("knobs_env") or identity.get("saved_candidates") or identity.get("no_gold")):
        raise EV.EvolveError("finished run configuration differs from counted configuration")
    knobs = child.manifest.get("knobs")
    sources = measured.knob_sources()
    expected_knobs = {name: {"value": list(value) if isinstance(value, tuple) else value, "source": sources[name]}
        for name, value in ((name, measured.knob(name)) for name in sources)
        if not (name == "examples_inline_sql" and value is False and sources[name] == "default")}
    if knobs != expected_knobs:
        raise EV.EvolveError("finished run knob configuration differs from its measured snapshot")
    implementation = child.manifest.get("implementation") or {}
    records = EV._implementation_records(implementation.get("role_digests"))
    if not records or implementation.get("commit") != evidence.get("source_commit"):
        raise EV.EvolveError("finished run has no matching measured source and role records")
    verify_role_records(snapshot, implementation, records)
    contexts = EV._implementation_contexts(child)
    recorded_knowledge_path(child.manifest.get("knowledge_root"))
    root = Path(str(evidence.get("knowledge_root") or child.manifest["knowledge_root"]))
    fingerprints = child.manifest.get("knowledge_fingerprints")
    databases = {str(row["db_id"]) for row in child.rows.values()}
    if (not root.is_absolute() or str(root.resolve()) != str(root)
            or not isinstance(fingerprints, Mapping) or set(fingerprints) != databases
            or fingerprints != evidence.get("knowledge_fingerprints")
            or knowledge_fingerprints(root, sorted(databases), embedding_model=EMBEDDING_MODEL) != fingerprints):
        raise EV.EvolveError("finished run knowledge fingerprints differ from actual knowledge")
    for row in child.rows.values():
        units = EV._role_units(row)
        if (row.get("final_score_source") != "host-reexecution" or not units
                or len(row.get("candidates") or []) != 3 or len(row.get("candidates_debug") or []) != 3):
            raise EV.EvolveError("finished run requires host scores and complete role source proofs")
        for unit in units.values():
            proof = unit.get("source_proof") or {}
            if (not cache_lineage.valid_descriptor(proof, attributed=True)
                    or not unit.get("source_attributed") or proof.get("content_sha256") != unit.get("content_sha")
                    or EV._unit_implementation(unit, contexts) is None
                    or proof.get("identity", {}).get("knowledge_fingerprint") != fingerprints[row["db_id"]]):
                raise EV.EvolveError("finished run role proof differs from recorded knowledge or source")
    metadata = child.path / "uncounted.json"
    if metadata.is_file():
        record = json.loads(metadata.read_bytes())
        if (record.get("release") != child.release or record.get("snapshot_digest") != snapshot.digest
                or record.get("source") != implementation["commit"]
                or not _retained_parent(record, child, parent)
                or {key: record.get(key) for key in ("candidates", "judge")} != configuration):
            raise EV.EvolveError("original uncounted execution metadata differs from registration")
        if record.get("schema") == "rubiksql.g12-uncounted/1":
            state = json.loads((child.path.parents[2] / "evidence" / "uncounted-state.json").read_bytes())
            if (state.get("phase") != "complete" or any(state.get(key) != value for key, value in record.items())
                    or not state.get("exits") or any(state["exits"].values())):
                raise EV.EvolveError("original uncounted launcher has not completed successfully")


def stage_finished(writer: Any, pin: str, run_dir: Path, parent_run: Path, *, snapshot_digest: str,
                   source_commit: str, actor: str, budget_cap: int | None = None,
                   knowledge_root: Path | None = None) -> Mapping[str, Any]:
    """Charge one finished run and record its outcome in the caller's batch.

    Args:
        writer: Active verified authority transaction.
        pin: Published proposal, possibly staged earlier in this transaction.
        run_dir: Original complete uncounted run; never modified here.
        parent_run: Original complete paired parent.
        snapshot_digest: Expected measured snapshot digest, supplied by the operator.
        source_commit: Expected measured source commit, supplied by the operator.
        actor: Recorded operator identity.
        budget_cap: Explicit generation cap extension, from ten through twenty.
        knowledge_root: Explicit relocated child knowledge copy. Its complete
            fingerprints must match the original manifest and every role proof.

    Returns:
        Full outcome with explicit post-execution registration provenance.

    Raises:
        AuthorityError: Duplicate run, exhausted cap or changed generation.
        EV.EvolveError: Any immutable execution evidence fails deep verification.
    """
    from .paid_launch import counted_attempts, generation_budget, require_evolution_regime
    from . import views

    child, parent = EV.read_run(run_dir), EV.read_run(parent_run)
    with writer.reader() as reader, views.history_store(reader, require_export=False) as store:
        head = reader.head()
        body = reader.proposal(pin)
        if EV.plan_view(body).base_parent != head.generation:
            raise AuthorityError("finished registration belongs to a different generation")
        require_evolution_regime(reader)
        from ..lab_ingest import run_snapshot
        from .paid_launch import evaluation_configuration

        configuration = evaluation_configuration(run_snapshot(child.path, child.manifest))
        evidence = {"schema": "rubiksql.finished-registration/1", "proposal": pin,
                    "plan_digest": body["plan_digest"], "child": run_identity(child), "parent": run_identity(parent),
                    "snapshot_digest": snapshot_digest, "source_commit": source_commit,
                    "configuration": configuration, "knowledge_fingerprints": child.manifest.get("knowledge_fingerprints")}
        if knowledge_root is not None:
            evidence["knowledge_root"] = str(knowledge_root.resolve())
        reservation_id = "o" + sha256_hex(canon(evidence))[:16]
        evidence["reservation_id"] = reservation_id
        verify_registered_run(child, parent, evidence, store=store)
        for event in reader.events("eval-reserve"):
            held = event["payload"]
            if (held.get("id") == reservation_id or held.get("run") == child.name
                    or held.get("out_dir") == str(child.path.resolve())
                    or held.get("registration", {}).get("child") == evidence["child"]):
                raise AuthorityError("finished run was already registered or reserved")
        blocker = EV.release_measurement_blocker(child.release, head.generation,
            attachments=(event["payload"] for event in reader.events("eval-attach")),
            outcomes=(event["payload"] for event in reader.events("outcome")),
            reservations=(event["payload"] for event in reader.events("eval-reserve")))
        if blocker is not None:
            raise AuthorityError("finished release was already measured or reserved")
        counted = counted_attempts(reader.events(), head.generation)
        cap = generation_budget(reader, budget_cap)
        if len(counted) >= cap:
            raise AuthorityError("generation has exhausted its counted evaluation budget")
        reservation = {"schema": "rubiksql.paid-reservation/1", "id": reservation_id,
            "admission": "finished/1", "proposal": pin, "release": child.release,
            "generation": head.generation, "epoch": reader.epoch()["epoch"], "counted": True,
            "run": child.name, "out_dir": str(child.path.resolve()), "question_set": child.identity["question_set"],
            "parent": evidence["parent"], "registration": evidence,
            "budget_cap": cap,
            "attempt": {"used": len(counted), "counted": sorted(str(row["id"]) for row in counted), "allow_over_cap": False},
            "created_at": time.time()}
    writer._event("eval-reserve", pin, reservation, actor)
    result = writer.record_outcome(pin, child.path, parent=parent.path, expected=writer.head(), actor=actor,
                                   auto_promote=False, registration=evidence)
    if run_identity(EV.read_run(child.path)) != evidence["child"] or run_identity(EV.read_run(parent.path)) != evidence["parent"]:
        raise EV.EvolveError("finished run or paired parent changed during registration")
    if result.get("pairing", {}).get("status") != "paired" or "attempt_event" not in result:
        raise EV.EvolveError("finished registration did not produce a counted paired outcome: "
                            + str(result.get("verdict_reason") or result.get("pairing")))
    return result
