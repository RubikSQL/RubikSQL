"""Attempt-local recovery of executed judge decisions before candidate fallback."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from . import cache_lineage, judge_majority
from .metrics.official_ex import result_signature
from .submission_gate import _successful
from .sql_action import rank_publication_enabled

POOL_ROLE = "judge_checkpoint_pool"
PASS_ROLE = "judge_checkpoint_pass:"


def captured(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retain checkpoint notes without retaining another copy of every live transcript.

    Args:
        records (list[dict[str, Any]]): One attempt's detached capture snapshot.

    Returns:
        list[dict[str, Any]]: Pool identity and completed-pass notes only.
    """
    return [record for record in records if (record.get("execution") or {}).get("role") == POOL_ROLE
            or str((record.get("execution") or {}).get("role", "")).startswith(PASS_ROLE)]


def pool_identity(question: str, db: str, release: str, fingerprint: str,
                  candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Bind recovery to the current execution, knowledge and exact ordered pool.

    Args:
        question (str): Canonical question and hint.
        db (str): Database identity.
        release (str): Pinned prompt release.
        fingerprint (str): Pinned knowledge fingerprint.
        candidates (Sequence[dict[str, Any]]): Full executed candidates in slot order.

    Returns:
        dict[str, Any]: Execution identity and SHA-256 of the question and pool.
    """
    origin = cache_lineage.identity(db=db, role="judge", slot=None, release=release,
                                    knowledge_fingerprint=fingerprint)
    payload = [question, [{"sql": c.get("sql"), "error": c.get("error"),
                           "result": result_signature(c)} for c in candidates]]
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                      default=str).encode()).hexdigest()
    return {key: origin[key] for key in ("run", "db", "qid", "release", "knowledge_fingerprint")} | {
        "pool_sha256": digest, "candidate_sql": [c.get("sql") or "" for c in candidates]}


def publish(identity: dict[str, Any], index: int, chosen: dict[str, Any], stage: str) -> None:
    """Checkpoint an executed pass before an optional stage or cleanup can fail.

    Args:
        identity (dict[str, Any]): Identity from ``pool_identity``.
        index (int): Zero-based pass index.
        chosen (dict[str, Any]): Latest executed decision including finished hook records.
        stage (str): Last completed pipeline stage.
    """
    mode = str((chosen.get("judge_debug") or {}).get("mode", ""))
    if judge_majority.answered(chosen) and _successful(chosen) and not mode.startswith("fallback"):
        result = {key: value for key, value in chosen.items() if key not in ("candidates", "candidates_debug")}
        cache_lineage.note(PASS_ROLE + str(index), {"identity": identity, "pass": index + 1,
                                                   "stage": stage, "result": result})


def recover(pred: dict[str, Any], snapshots: Sequence[list[dict[str, Any]]]) -> dict[str, Any]:
    """Use completed pass checkpoints only when no final decision survived.

    Retry snapshots are scoped to one evaluator question. Only checkpoints matching
    the latest ready pool are eligible; a later unfinished pass cannot erase a prior
    completed stage. Each pass contributes once, with ties resolved by pass order.
    Recovery is explicitly incomplete and is never reported as a completed majority.

    Args:
        pred (dict[str, Any]): Prediction about to enter candidate fallback.
        snapshots (Sequence[list[dict[str, Any]]]): Chronological attempt captures.

    Returns:
        dict[str, Any]: Recovered prediction or the original prediction if ineligible.
    """
    if pred.get("judge_majority") or ((pred.get("sql") or "").strip() and pred.get("source") != "tail_judge"):
        return pred
    identity = {}
    checkpoints = []
    for records in snapshots:
        current = cache_lineage.noted(records, POOL_ROLE)
        if current:
            identity = current
        checkpoints.extend(record for record in records
                           if str((record.get("execution") or {}).get("role", "")).startswith(PASS_ROLE))
    if not identity or any(not identity.get(key) for key in
                           ("run", "db", "qid", "release", "knowledge_fingerprint", "pool_sha256")):
        return pred
    if pred.get("candidates") is not None and [c.get("sql") or "" for c in pred["candidates"]] != identity["candidate_sql"]:
        return pred
    by_pass = {}
    for checkpoint in checkpoints:
        result = checkpoint.get("result")
        index = checkpoint.get("pass")
        if (checkpoint.get("identity") == identity and type(index) is int and index > 0
                and judge_majority.answered(result) and _successful(result)):
            previous = by_pass.get(index)
            status = lambda item: ((item.get("result") or {}).get("rank_contract") or {}).get("after", {}).get("status")
            if not (rank_publication_enabled() and previous and status(previous) == "verified" and status(checkpoint) == "missing"):
                by_pass[index] = checkpoint
    if not by_pass:
        return pred
    completed = [by_pass[index] for index in sorted(by_pass)]
    chosen, groups = judge_majority.majority([item["result"] for item in completed], result_signature)
    status = lambda item: ((item.get("result") or {}).get("rank_contract") or {}).get("after", {}).get("status")
    if rank_publication_enabled() and status(completed[chosen]) == "missing":
        eligible = [index for index, item in enumerate(completed) if status(item) == "verified"]
        if eligible:
            winner, _ = judge_majority.majority([completed[index]["result"] for index in eligible], result_signature)
            chosen = eligible[winner]
    result = completed[chosen]["result"]
    recovery = {"v": 1, "identity": identity, "chosen_pass": completed[chosen]["pass"],
                "groups": [[completed[index - 1]["pass"] for index in group] for group in groups],
                "checkpoints": [{key: item[key] for key in ("pass", "stage")} for item in completed],
                "pipeline_complete": all(item["stage"] == "complete" for item in completed)}
    review = result.get("reviewer_debug") or {"active": False, "skipped": "checkpoint-before-review"}
    return {**pred, **result, "reviewer_debug": review, "source": "tail_judge", "judge_checkpoint_recovery": recovery,
            "tail_judge_recovery": pred.get("tail_judge_recovery") or
            {"policy": "completed_judge_checkpoints", "stage": "incomplete_majority"}}
