"""Score candidate SQL and retain bounded evidence for each evaluated answer.

Full execution rows decide official EX and distinct-row differences. Each parent row
then keeps at most 64 positional head/tail rows. A versioned QA owns the fixed gold
snapshot; Candidate, Judge, and final-answer snapshots remain independent. Scoring
keeps only bounded lab snapshots and never calculates a score from a preview.
The shared reference cache retains complete gold results outside the lab.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import heavenbase as hb

from .lab_ingest import _lab_write, refresh_experiment_scores
from .lab_identity import database_path, qa_for_trace, qa_gold_lock, validate_database_version
from .lab_schema import CandidateEntity, ExperimentEntity, JudgeEntity, QAEntity, TraceEntity
from .metrics import official_ex_match
from .metrics.official_ex import row_tuple
from .model_render import table_row_indices
from .sql_host import query_readonly
from .gold import execute_gold

#: Seconds one SQL may run. Model-written SQL includes accidental cross joins that never
#: finish; the cap turns those into an unmeasured row rather than a stalled pass.
SCORE_TIMEOUT_SEC = 120.0

#: Substring identifying the read guard's own cancellation, as opposed to a SQL error.
#: A cancelled query proves nothing about the answer, so it must not score zero.
_TIMEOUT_MARK = "execution limit"

#: Bounded stored row previews; full result sets still decide EX and the difference.
RESULT_ROW_LIMIT = 64

RESULT_DIGEST_ALGORITHM = "sql-result-v1"
SCORING_SOURCES = frozenset({"candidate_scoring_reexecution", "gold_scoring_reexecution",
                             "judge_scoring_reexecution", "final_scoring_reexecution"})


def _execute(url: str, sql: str) -> tuple[dict[str, Any], str]:
    """Execute one SQL read-only and classify its full result.

    Args:
        url: SQLite URL of the database to run against.
        sql: The SQL to execute.

    Returns:
        tuple: The raw query result and ``"ok"``, ``"error"`` or ``"timeout"``.
        An empty or rejected SQL is an error; only a cancelled query times out.
    """
    if not (sql or "").strip():
        return {"error": "empty SQL"}, "error"
    result = query_readonly(url, sql, timeout_sec=SCORE_TIMEOUT_SEC)
    if "error" in result:
        return result, "timeout" if _TIMEOUT_MARK in str(result["error"]) else "error"
    return result, "ok"


def _sql_hash(sql: str) -> str:
    """Bind a stored snapshot to the exact SQL text that produced it."""
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    """Represent every SQLite cell in JSON without shortening or losing special values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        label = "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
        return {"$sqlite": "float", "value": label}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"$sqlite": "blob", "base64": base64.b64encode(bytes(value)).decode("ascii")}
    raise TypeError(f"unsupported SQLite result value: {type(value).__name__}")


def _json_row(row: Any) -> list[Any]:
    return [_json_value(value) for value in row_tuple(row)]


def _truncation(total: int, retained: int) -> dict[str, Any] | None:
    if total <= retained:
        return None
    head = (retained + 1) // 2
    return {"strategy": "head_tail", "head_row_count": head,
            "tail_row_count": retained - head, "omitted_row_count": total - retained}


def _digest_chunk(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _ordered_cell(value: Any) -> list[Any]:
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["text", value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ["blob", base64.b64encode(bytes(value)).decode("ascii")]
    raise TypeError(f"unsupported SQLite result value: {type(value).__name__}")


def _ex_cell(value: Any) -> list[Any]:
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["number", str(int(value)), "1"]
    if isinstance(value, int):
        return ["number", str(value), "1"]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite numeric result is not digest-comparable")
        numerator, denominator = value.as_integer_ratio()
        return ["number", str(numerator), str(denominator)]
    if isinstance(value, str):
        return ["text", value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ["blob", base64.b64encode(bytes(value)).decode("ascii")]
    raise TypeError(f"unsupported SQLite result value: {type(value).__name__}")


def _result_digest(columns: list[str], rows: list[Any]) -> dict[str, Any]:
    """Digest complete ordered rows and the official EX distinct-row set.

    Numeric EX values use exact integer ratios, matching Python equality without
    coercing large integers through float. A nonfinite value makes the EX set digest
    unknown; the ordered digest still records its exact typed representation.
    """
    ordered = hashlib.sha256()
    _digest_chunk(ordered, _encoded([RESULT_DIGEST_ALGORITHM, "ordered", columns]))
    ex_rows: set[bytes] = set()
    comparable = True
    for row in rows:
        values = row_tuple(row)
        _digest_chunk(ordered, _encoded([_ordered_cell(value) for value in values]))
        if comparable:
            try:
                ex_rows.add(_encoded([_ex_cell(value) for value in values]))
            except ValueError:
                comparable = False
                ex_rows.clear()
    ex_hash = None
    if comparable:
        ex_digest = hashlib.sha256()
        _digest_chunk(ex_digest, _encoded([RESULT_DIGEST_ALGORITHM, "official_ex_distinct_rows"]))
        for row in sorted(ex_rows):
            _digest_chunk(ex_digest, row)
        ex_hash = ex_digest.hexdigest()
    return {
        "algorithm": RESULT_DIGEST_ALGORITHM,
        "ordered_sha256": ordered.hexdigest(),
        "ex_set_sha256": ex_hash,
        "ex_comparable": comparable,
        "reason": None if comparable else "nonfinite_numeric_value",
    }


def _snapshot(
    result: dict[str, Any], status: str, sql: str, db_id: str,
    database_version: str, *, source: str, sql_source: dict[str, Any],
    executed_at: str | None = None, copied_from: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a bounded positional snapshot bound to exact SQL and database bytes."""
    rows = (result.get("rows") or []) if status == "ok" else None
    indices = table_row_indices(len(rows), RESULT_ROW_LIMIT) if rows is not None else None
    snapshot = {
        "source": source,
        "executed_at": executed_at or datetime.now(timezone.utc).isoformat(),
        "db_id": db_id,
        "database_version": database_version,
        "sql": sql,
        "sql_source": sql_source,
        "sql_sha256": _sql_hash(sql),
        "status": status,
        "columns": list(result["columns"]) if status == "ok" else None,
        "rows": [_json_row(rows[index]) for index in indices] if indices is not None else None,
        "row_indices": list(indices) if indices is not None else None,
        "row_count": len(rows) if rows is not None else None,
        "truncated": len(rows) > RESULT_ROW_LIMIT if rows is not None else None,
        "truncation": _truncation(len(rows), len(indices)) if indices is not None else None,
        "error": str(result["error"]) if status != "ok" else None,
    }
    if rows is not None:
        snapshot["result_digest"] = _result_digest(snapshot["columns"], rows)
    if copied_from is not None:
        snapshot["copied_from"] = copied_from
    return snapshot


def _difference(pred_rows: list[Any], gold_rows: list[Any], gold_sql: str) -> dict[str, Any]:
    """Summarize complete result sets using official EX's distinct positional rows."""
    pred = {row_tuple(row) for row in pred_rows}
    gold = {row_tuple(row) for row in gold_rows}
    extra, missing = pred - gold, gold - pred

    def sample(rows: list[Any], wanted: set[tuple]) -> tuple[list[list[Any]], list[int]]:
        """Index first-seen distinct difference rows, not original result positions."""
        found: set[tuple] = set()
        distinct: list[tuple] = []
        for row in rows:
            values = row_tuple(row)
            if values in wanted and values not in found:
                distinct.append(values)
                found.add(values)
        indices = table_row_indices(len(distinct), RESULT_ROW_LIMIT)
        return [_json_row(distinct[index]) for index in indices], list(indices)

    extra_rows, extra_indices = sample(pred_rows, extra)
    missing_rows, missing_indices = sample(gold_rows, missing)

    return {
        "semantics": "official_ex_distinct_rows",
        "row_indices_scope": "distinct_difference_order",
        "gold_sql_sha256": _sql_hash(gold_sql),
        "extra_count": len(extra),
        "missing_count": len(missing),
        "extra_rows": extra_rows,
        "missing_rows": missing_rows,
        "extra_row_indices": extra_indices,
        "missing_row_indices": missing_indices,
        "extra_truncated": len(extra) > RESULT_ROW_LIMIT,
        "missing_truncated": len(missing) > RESULT_ROW_LIMIT,
        "extra_truncation": _truncation(len(extra), len(extra_indices)),
        "missing_truncation": _truncation(len(missing), len(missing_indices)),
    }


def _as_row(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else value.to_dict()


def _bound_snapshot(snapshot: Any, sql: str, db_id: str, database_version: str, *,
                    sql_source: dict[str, Any], native_source: str, gold_sql: str | None = None,
                    allow_copy: bool = False) -> bool:
    if not isinstance(snapshot, dict):
        return False
    source = snapshot.get("source")
    copied = snapshot.get("copied_from")
    if source == native_source:
        if copied is not None:
            return False
    elif (not allow_copy or source not in SCORING_SOURCES or not isinstance(copied, dict)
          or copied.get("role") != source.split("_", 1)[0]
          or not isinstance(copied.get("id"), str) or not copied["id"]):
        return False
    return (snapshot.get("sql") == sql
            and snapshot.get("sql_sha256") == _sql_hash(sql)
            and snapshot.get("db_id") == db_id
            and snapshot.get("database_version") == database_version
            and snapshot.get("sql_source") == sql_source
            and (gold_sql is None or snapshot.get("gold_sql_sha256") == _sql_hash(gold_sql))
            and snapshot.get("status") in ("ok", "error", "timeout")
            and (snapshot.get("status") != "ok" or
                 isinstance(snapshot.get("result_digest"), dict) and
                 snapshot["result_digest"].get("algorithm") == RESULT_DIGEST_ALGORITHM))


def _bound_answer_snapshot(snapshot: Any, sql: str, db_id: str, database_version: str, *,
                           sql_source: dict[str, Any], native_source: str, gold_sql: str,
                           allow_copy: bool = False) -> bool:
    return (_bound_snapshot(snapshot, sql, db_id, database_version, sql_source=sql_source,
                            native_source=native_source, gold_sql=gold_sql, allow_copy=allow_copy)
            and (snapshot["status"] != "ok" or isinstance(snapshot.get("difference"), dict)
                 or (snapshot.get("gold_execution") or {}).get("status") == "gold_unscorable"))


def _publish_gold(
    ws: hb.HeavenBase, lab_dir: str | os.PathLike[str], qa: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Fix the first successful QA gold snapshot; report later EX disagreement."""
    qa_id = str(qa["object_id"])
    with qa_gold_lock(lab_dir, qa_id):
        current_raw = ws.get(qa_id, QAEntity)
        if current_raw is None:
            raise ValueError(f"QA {qa_id} disappeared before gold publication")
        current = _as_row(current_raw)
        if current.get("gold_sql") != snapshot["sql"] or current.get("db_id") != snapshot["db_id"] \
                or current.get("database_version") != snapshot["database_version"]:
            raise ValueError(f"QA {qa_id} definition changed before gold publication")
        previous = current.get("gold_execution_result")
        if previous is not None and not _bound_snapshot(previous, snapshot["sql"],
                                                        snapshot["db_id"], snapshot["database_version"],
                                                        sql_source={"kind": "qa_gold", "qa_id": qa_id},
                                                        native_source="gold_scoring_reexecution"):
            raise ValueError(f"QA {qa_id} gold snapshot has a conflicting SQL/database binding")
        result = {"status": "new", "current_digest": snapshot.get("result_digest"),
                  "fixed_digest": previous.get("result_digest") if isinstance(previous, dict) else None,
                  "source": snapshot["source"], "executed_at": snapshot["executed_at"]}
        if snapshot["status"] != "ok":
            if previous is None:
                ws.set(qa_id, {"gold_execution_result": snapshot}, entity=QAEntity)
            result["status"] = "gold_unscorable" if snapshot.get("gold_execution") else "failed"
            result["gold_execution"] = snapshot.get("gold_execution")
            result["error"] = snapshot["error"]
            if previous is not None and (previous.get("status") == "ok"
                    or (previous.get("gold_execution") or {}).get("key") != (snapshot.get("gold_execution") or {}).get("key")):
                result["status"] = "conflict"
            return result
        if previous is None or previous["status"] != "ok":
            fields = {"gold_execution_result": snapshot}
            if current.get("gold_rows") is None:
                fields["gold_rows"] = snapshot["row_count"]
            if current.get("empty_result") is None:
                fields["empty_result"] = snapshot["row_count"] == 0
            ws.set(qa_id, fields, entity=QAEntity)
            if not snapshot["result_digest"]["ex_comparable"]:
                result["status"] = "unknown"
            if current.get("consistent_result") is False:
                result["status"] = "conflict"
                result["reason"] = "QA previously observed an EX gold conflict"
            return result
        old_digest, new_digest = previous.get("result_digest"), snapshot.get("result_digest")
        if not isinstance(old_digest, dict) or old_digest.get("algorithm") != RESULT_DIGEST_ALGORITHM \
                or not old_digest.get("ex_comparable") or not new_digest.get("ex_comparable"):
            result["status"] = "unknown"
            if current.get("consistent_result") is not False:
                ws.set(qa_id, {"consistent_result": None}, entity=QAEntity)
        elif old_digest["ex_set_sha256"] != new_digest["ex_set_sha256"]:
            result["status"] = "conflict"
            ws.set(qa_id, {"consistent_result": False}, entity=QAEntity)
        else:
            result["status"] = ("matching" if old_digest["ordered_sha256"] == new_digest["ordered_sha256"]
                                else "ordered_variation")
            if current.get("consistent_result") is None:
                ws.set(qa_id, {"consistent_result": True}, entity=QAEntity)
        if current.get("consistent_result") is False:
            result["status"] = "conflict"
            result["reason"] = "QA previously observed an EX gold conflict"
        return result


def gold_consistency_problem(ws: hb.HeavenBase, experiment: str) -> str | None:
    """Return the first missing or unusable gold comparison for an experiment."""
    raw = ws.get(experiment, ExperimentEntity)
    if raw is None:
        return f"experiment {experiment} does not exist"
    row = _as_row(raw)
    provenance = row.get("provenance") or {}
    comparisons = provenance.get("gold_consistency") or {}
    traces = ws.query(TraceEntity).where({"exp_id": {"$eq": experiment}}).execute().rows()
    if not traces:
        return f"experiment {experiment} has no bound question traces"
    declared = row.get("n_questions")
    if type(declared) is not int or declared <= 0 or declared != len(traces):
        return f"experiment {experiment} declares {declared!r} questions but has {len(traces)} bound traces"
    for trace in traces:
        qa_id = trace.get("qa_id")
        comparison = comparisons.get(qa_id) if isinstance(comparisons, dict) else None
        status = comparison.get("status") if isinstance(comparison, dict) else None
        if status not in ("new", "matching", "ordered_variation", "gold_unscorable"):
            return f"{trace['qid']} gold evidence {status or 'missing'} for QA {qa_id}"
        qa = qa_for_trace(ws, trace)
        if status == "gold_unscorable":
            fixed = (qa.get("gold_execution_result") or {}).get("gold_execution") or {}
            retained = comparison.get("gold_execution") or {}
            if fixed.get("status") != "gold_unscorable" or not fixed.get("key") or any(
                    fixed.get(key) != retained.get(key) for key in ("key", "status", "result_sha256")):
                return f"{trace['qid']} gold-unscorable evidence differs from QA {qa_id}"
        if qa.get("consistent_result") is False:
            return f"{trace['qid']} QA {qa_id} has an EX gold conflict"
    return None


@_lab_write
def score_candidates(
    ws: hb.HeavenBase,
    db_dir: str | os.PathLike[str],
    *,
    exp_id: str | None = None,
    db_id: str | None = None,
    rescore: bool = False,
    workers: int = 4,
) -> dict[str, int]:
    """Score candidates and complete QA, Judge, and final bounded snapshots.

    Args:
        ws: Open v0 lab workspace.
        db_dir: Raw BIRD SQLite database directory.
        exp_id: Optional experiment filter.
        db_id: Optional database filter.
        rescore: Explicitly re-execute already measured candidate and answer SQL.
        workers: Number of question workers.

    Returns:
        dict[str, int]: Candidate scored/correct/timeout/no_gold/skipped counts.

    Raises:
        ValueError: A Trace lacks its exact QA or SQL/database binding.
    """
    from .lab_store import _lab_directory

    lab_dir = _lab_directory(ws)
    trace_query = ws.query(TraceEntity)
    if exp_id:
        trace_query = trace_query.where({"exp_id": {"$eq": exp_id}})
    if db_id:
        trace_query = trace_query.where({"db_id": {"$eq": db_id}})
    traces = trace_query.execute().rows()
    candidate_query = ws.query(CandidateEntity)
    judge_query = ws.query(JudgeEntity)
    if exp_id:
        candidate_query = candidate_query.where({"exp_id": {"$eq": exp_id}})
        judge_query = judge_query.where({"exp_id": {"$eq": exp_id}})
    if db_id:
        candidate_query = candidate_query.where({"db_id": {"$eq": db_id}})
        judge_query = judge_query.where({"db_id": {"$eq": db_id}})
    by_trace: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in candidate_query.execute().rows():
        by_trace.setdefault((str(candidate["exp_id"]), str(candidate["qid"])), []).append(candidate)
    judges = {(str(row["exp_id"]), str(row["qid"])): row for row in judge_query.execute().rows()}
    trace_keys = {(str(row["exp_id"]), str(row["qid"])) for row in traces}
    orphaned = (set(by_trace) | set(judges)) - trace_keys
    if orphaned:
        raise ValueError(f"candidate or judge has no bound Trace: {sorted(orphaned)[0]}")
    counts = {"scored": 0, "correct": 0, "timeout": 0, "no_gold": 0, "skipped": 0}
    versions: dict[str, str] = {}
    urls: dict[str, str] = {}
    work = []
    for trace in traces:
        key = str(trace["exp_id"]), str(trace["qid"])
        qa = qa_for_trace(ws, trace)
        name, version = str(trace["db_id"]), str(qa["database_version"])
        if trace.get("gold_sql") != qa["gold_sql"]:
            raise ValueError(f"{key} gold SQL differs from bound QA {qa['object_id']}")
        if name in versions and versions[name] != version:
            raise ValueError(f"{name} has multiple database versions in one scoring pass")
        if name not in urls:
            path = database_path(db_dir, name)
            versions[name] = validate_database_version(path, version)
            urls[name] = "sqlite:///" + os.fspath(path)
        candidates = by_trace.get(key, [])
        if any(row.get("db_id") != name for row in candidates):
            raise ValueError(f"{key} candidate database differs from its Trace")
        pending = [row for row in candidates if rescore or row.get("scores") is None
                   or not _bound_answer_snapshot(row.get("execution_result"), str(row.get("sql") or ""), name, version,
                                                 sql_source={"kind": "candidate", "candidate_id": str(row["object_id"])},
                                                 native_source="candidate_scoring_reexecution",
                                                 gold_sql=str(qa["gold_sql"]))]
        counts["skipped"] += len(candidates) - len(pending)
        judge = judges.get(key)
        if judge is not None and judge.get("db_id") != name:
            raise ValueError(f"{key} Judge database differs from its Trace")
        judge_sql = str(judge.get("sql") or "") if judge else ""
        if judge_sql and not isinstance(judge.get("sql_source"), dict):
            raise ValueError(f"{key} Judge SQL lacks confirmed source provenance")
        judge_needed = bool(judge_sql and judge.get("sql_source") and
                            (rescore or not _bound_answer_snapshot(judge.get("execution_result"),
                                                                   judge_sql, name, version,
                                                                   sql_source=judge["sql_source"],
                                                                   native_source="judge_scoring_reexecution",
                                                                   gold_sql=str(qa["gold_sql"]), allow_copy=True)))
        final_sql = str(trace.get("pred_sql") or "")
        final_needed = rescore or not _bound_answer_snapshot(trace.get("pred_execution_result"),
                                                              final_sql, name, version,
                                                              sql_source={"kind": "trace_final", "trace_id": str(trace["object_id"])},
                                                              native_source="final_scoring_reexecution",
                                                              gold_sql=str(qa["gold_sql"]), allow_copy=True)
        stored_gold = qa.get("gold_execution_result")
        experiment_row = ws.get(key[0], ExperimentEntity)
        comparisons = ((_as_row(experiment_row).get("provenance") or {}).get("gold_consistency") or {}) \
            if experiment_row is not None else {}
        missing_comparison = str(qa["object_id"]) not in comparisons
        gold_incomplete = (not _bound_snapshot(stored_gold, str(qa["gold_sql"]), name, version,
                                               sql_source={"kind": "qa_gold", "qa_id": str(qa["object_id"])},
                                               native_source="gold_scoring_reexecution")
                           or stored_gold["status"] != "ok" and
                           (stored_gold.get("gold_execution") or {}).get("status") != "gold_unscorable")
        if pending or judge_needed or final_needed or gold_incomplete or missing_comparison:
            work.append((trace, qa, pending, judge if judge_needed else None, final_needed,
                         urls[name], version))

    def _measure(item: tuple) -> tuple:
        trace, qa, pending, judge, final_needed, url, version = item
        name = str(trace["db_id"])
        gold_sql = str(qa["gold_sql"])
        gold_result = execute_gold(url, gold_sql)
        retained = gold_result["gold_execution"]
        observed = (trace.get("result_attempts") or {}).get("gold_execution")
        if observed and any(observed.get(key) != retained.get(key) for key in ("key", "status", "result_sha256")):
            raise ValueError(f"{trace['qid']} retained gold differs from the evaluation reference")
        gold_status = ("timeout" if retained["failure_kind"] == "timeout" else "error") if "error" in gold_result else "ok"
        gold_snapshot = _snapshot(gold_result, gold_status, gold_sql, name, version,
                                  source="gold_scoring_reexecution",
                                  sql_source={"kind": "qa_gold", "qa_id": str(qa["object_id"])})
        gold_snapshot["gold_execution"] = retained
        # Even identical answer text must execute under the answer budget.
        executed: dict[str, tuple[dict[str, Any], str, dict[str, Any], dict[str, str]]] = {}
        candidate_patches = []
        for candidate in pending:
            sql = str(candidate.get("sql") or "")
            result, status = _execute(url, sql)
            snapshot = _snapshot(result, status, sql, name, version,
                                  source="candidate_scoring_reexecution",
                                  sql_source={"kind": "candidate", "candidate_id": str(candidate["object_id"])})
            snapshot["gold_sql_sha256"] = _sql_hash(gold_sql)
            snapshot["gold_execution"] = retained
            snapshot["difference"] = (_difference(result["rows"], gold_result["rows"], gold_sql)
                                      if status == gold_status == "ok" else None)
            if gold_status != "ok":
                scores, outcome = {"ex": 0.0}, "no_gold"
            elif status == "timeout":
                scores, outcome = None, "timeout"
            elif status == "error":
                scores, outcome = {"ex": 0.0}, "scored"
            else:
                scores = {"ex": float(official_ex_match(result["rows"], gold_result["rows"]))}
                outcome = "scored"
            candidate_patches.append((str(candidate["object_id"]), scores, snapshot, outcome))
            executed.setdefault(sql, (result, status, snapshot,
                                       {"role": "candidate", "id": str(candidate["object_id"])}))

        def answer_snapshot(sql: str, source: str, sql_source: dict[str, Any]) -> dict[str, Any]:
            if sql in executed:
                result, status, original, copied_from = executed[sql]
                snapshot = _snapshot(result, status, sql, name, version,
                                     source=original["source"], sql_source=sql_source,
                                     executed_at=original["executed_at"], copied_from=copied_from)
            else:
                result, status = _execute(url, sql)
                snapshot = _snapshot(result, status, sql, name, version,
                                     source=source, sql_source=sql_source)
                executed[sql] = result, status, snapshot, {"role": source.split("_")[0],
                                                            "id": str(trace["object_id"])}
            snapshot["gold_sql_sha256"] = _sql_hash(gold_sql)
            snapshot["gold_execution"] = retained
            snapshot["difference"] = (_difference(result["rows"], gold_result["rows"], gold_sql)
                                      if status == gold_status == "ok" else None)
            return snapshot

        judge_snapshot = None
        if judge is not None:
            judge_snapshot = answer_snapshot(str(judge["sql"]), "judge_scoring_reexecution",
                                              judge["sql_source"])
        final_snapshot = None
        if final_needed:
            final_snapshot = answer_snapshot(str(trace.get("pred_sql") or ""),
                                             "final_scoring_reexecution",
                                             {"kind": "trace_final", "trace_id": str(trace["object_id"])})
        return trace, qa, gold_snapshot, candidate_patches, judge_snapshot, final_snapshot

    updated_experiments: set[str] = set()
    by_database: dict[str, list[tuple]] = {}
    for item in work:
        by_database.setdefault(str(item[0]["db_id"]), []).append(item)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for name, batch in by_database.items():
            # Full rows die inside _measure. Only bounded snapshots remain while the
            # source bytes are checked again, before any measured row is published.
            measured = list(pool.map(_measure, batch))
            validate_database_version(database_path(db_dir, name), versions[name])
            for trace, qa, gold_snapshot, candidate_patches, judge_snapshot, final_snapshot in measured:
                comparison = _publish_gold(ws, lab_dir, qa, gold_snapshot)
                experiment = str(trace["exp_id"])
                with qa_gold_lock(lab_dir, f"experiment:{experiment}"):
                    raw_experiment = ws.get(experiment, ExperimentEntity)
                    if raw_experiment is None:
                        raise ValueError(f"experiment {experiment} disappeared before gold comparison publication")
                    provenance = dict(_as_row(raw_experiment).get("provenance") or {})
                    gold_consistency = dict(provenance.get("gold_consistency") or {})
                    gold_consistency[str(qa["object_id"])] = comparison
                    provenance["gold_consistency"] = gold_consistency
                    ws.set(experiment, {"provenance": provenance}, entity=ExperimentEntity)
                for candidate_id, scores, snapshot, outcome in candidate_patches:
                    ws.set(candidate_id, {"scores": scores, "execution_result": snapshot}, entity=CandidateEntity)
                    if outcome == "scored":
                        counts["scored"] += 1
                        counts["correct"] += int(scores["ex"] >= 1.0)
                    else:
                        counts[outcome] += 1
                if judge_snapshot is not None:
                    ws.set(str(trace["object_id"]), {"execution_result": judge_snapshot}, entity=JudgeEntity)
                if final_snapshot is not None:
                    ws.set(str(trace["object_id"]), {"pred_execution_result": final_snapshot}, entity=TraceEntity)
                updated_experiments.add(experiment)
    for experiment in sorted(updated_experiments):
        if ws.get(experiment, ExperimentEntity) is not None:
            refresh_experiment_scores(ws, experiment)
    return counts


def main(argv: list[str] | None = None) -> None:
    """CLI entry: score stored candidates and report the pool's ceiling.

    Args:
        argv: Argument list, or ``None`` for ``sys.argv``.

    Returns:
        None: Prints the outcome counts.
    """
    import argparse

    from .lab_schema import open_lab_workspace
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, help="Assert the bound evolution database or its directory")
    parser.add_argument("--lab-dir", help=argparse.SUPPRESS)
    parser.add_argument("--db-dir", required=True,
                        help="Directory holding the raw BIRD SQLite databases")
    parser.add_argument("--exp", default="", help="Score one experiment only")
    parser.add_argument("--db", default="", help="Score one database only")
    parser.add_argument("--rescore", action="store_true",
                        help="Re-measure candidates that already carry a score")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent executions")
    args = parser.parse_args(argv)
    if args.lab_dir is not None:
        parser.error("--lab-dir is retired; --workspace may only name the bound evolution workspace")

    counts = score_candidates(
        open_lab_workspace(args.workspace), args.db_dir,
        exp_id=args.exp or None, db_id=args.db or None,
        rescore=args.rescore, workers=args.workers,
    )
    print(f"scored {counts['scored']} candidates: {counts['correct']} match gold, "
          f"{counts['timeout']} timed out, {counts['no_gold']} have unusable gold, "
          f"{counts['skipped']} already scored")


if __name__ == "__main__":
    main()
