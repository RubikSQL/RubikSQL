"""Bounded, verifiable delivery of evolution evidence, not a claim that an agent understood it.

The lab builds immutable text for one row's history or one parent's question. The driver
fetches it independently and compares the actual completed CLI tool results byte for byte.
Capture completeness (what survived the original run) is distinct from delivery coverage.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

PAGE_CHARS = 16384
SCHEMA = "rubiksql.read-evidence/2"
OVERVIEW_CONTRACT = "rubiksql.experiment-overview/2"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _record_text(record: Mapping[str, Any]) -> str:
    # Chapter keys are intentionally ordered; nested values retain canonical encoding.
    return "{" + ",".join(_json(key) + ":" + _json(value) for key, value in record.items()) + "}\n"


def _canonical_id(kind: str, id: str) -> str:
    return _json(json.loads(id)) if kind == "question" else id


def _qa_for_trace(rows: Callable[[str, dict[str, Any]], list[dict[str, Any]]],
                  trace: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the exact QA version recorded by a trace."""
    from .lab_identity import qa_object_id

    qa_id = trace.get("qa_id")
    if not isinstance(qa_id, str) or not qa_id:
        raise ValueError(f"trace {trace.get('object_id')!r} has no versioned qa_id")
    found = rows("qa", {"object_id": qa_id})
    if len(found) != 1 or found[0].get("name") != trace.get("qid") or found[0].get("db_id") != trace.get("db_id"):
        raise ValueError(f"trace {trace.get('object_id')!r} has no matching versioned QA {qa_id!r}")
    qa = found[0]
    if qa_id != qa_object_id(qa.get("dataset_version"), qa.get("database_version"), qa["name"]):
        raise ValueError(f"trace {trace.get('object_id')!r} has an invalid QA version identity {qa_id!r}")
    return qa


def evidence_source(ws: Any, kind: str, id: str) -> dict[str, Any]:
    """Build full evidence from trusted lab rows, including original payloads and capture gaps.

    Args:
        ws: An open lab workspace.
        kind: ``row_history``, ``question`` or ``experiment_overview``.
        id: Prompt name or typed closure key, JSON ``[parent_experiment, qid]``, or
            experiment id respectively. Typed keys are ``tool:<name>``, ``rule:<title>``,
            ``parameters:knobs``, ``tools:loader`` and ``rules:corpus``.

    Returns:
        The complete source text, identity, digest and independent capture status.

    Raises:
        ValueError: The source does not exist, its identity is invalid or rows are inconsistent.
    """
    from .lab_schema import lab_query_entities

    entities = lab_query_entities(ws)
    id = _canonical_id(kind, id)

    def rows(alias: str, where: dict[str, Any]) -> list[dict[str, Any]]:
        found = ws.query(entities.get(alias, alias)).where(where).execute().rows()
        return sorted(found, key=lambda row: str(row.get("object_id", "")))

    gaps: list[str] = []
    capture_scope = "stored_row_versions_and_proposals" if kind == "row_history" else "saved_agent_conversations"
    unrecorded_attempts: Any = None
    records = None
    if kind == "row_history" and ":" in id:
        from .prompts.authority import Authority
        from .prompts.store import PromptStoreError
        from .prompts.views import history_store

        binding = getattr(ws, "_rubiksql_authority_binding", None)
        if binding is None:
            raise PromptStoreError("history reads require the bound workspace adapter")
        options = {"commit": ws._rubiksql_authority_commit} if getattr(ws, "_rubiksql_authority_commit", None) else {}
        with Authority.read(binding=binding, **options) as reader:
            with history_store(reader) as store:
                body = _typed_row_history(store, id)
        capture_scope = "typed_versions_releases_proposals_and_events"
    elif kind == "row_history":
        from .lab_store import _history_rows

        versions = rows("sys-prompt", {"name": id})
        inherited = "candidate" if not versions and re.fullmatch(r"candidate\.s[1-3]", id) else None
        if inherited:
            versions = rows("sys-prompt", {"name": inherited})
        if not versions:
            raise ValueError(f"no recorded versions for {id!r}")
        # Capsules are executable artifacts. All authored content and provenance live in manifest.
        versions = [{key: row.get(key) for key in ("object_id", "name", "version", "status", "manifest", "fingerprints")} for row in versions]
        changed_name = inherited or id
        proposals = [row for row in _history_rows(ws, "proposal") if changed_name in row["row_names"]]
        body = {"name": id, "inherited_from": inherited, "versions": versions, "proposals": proposals}
    elif kind == "experiment_overview":
        capture_scope = "all_scored_wrong_question_answer_cards"
        records = _experiment_overview_records(rows, id, gaps)
        body = {}
    elif kind == "question":
        identity = json.loads(id)
        if not isinstance(identity, list) or len(identity) != 2 or not all(isinstance(item, str) and item for item in identity):
            raise ValueError("question id must be JSON [parent_experiment, qid]")
        experiment, qid = identity
        where = {"exp_id": experiment, "qid": qid}
        traces = rows("trace", where)
        if len(traces) != 1:
            raise ValueError(f"expected one parent trace for {identity!r}")
        qa = _qa_for_trace(rows, traces[0])
        candidates, judges = rows("candidate", where), rows("judge", where)
        sessions = rows("session", {"state.exp_id": experiment, "state.qid": qid})
        expected = set(traces[0].get("session_ids") or [])
        unrecorded_attempts = (traces[0].get("result_attempts") or {}).get("unrecorded_results")
        expected.update(row["session_id"] for row in [*candidates, *judges] if row.get("session_id"))
        held = {row["object_id"] for row in sessions}
        gaps.extend(f"session_missing:{sid}" for sid in sorted(expected - held))
        if any(not row.get("session_id") for row in [*candidates, *judges]):
            gaps.append("candidate_or_judge_session_link_missing")
        conversations = []
        for session in sessions:
            sid = session["object_id"]
            messages = sorted(rows("message", {"session_id": sid}), key=lambda row: (row.get("idx") if type(row.get("idx")) is int else -1, str(row.get("object_id", ""))))
            if [row["object_id"] for row in messages] != list(session.get("message_ids") or []):
                gaps.append(f"message_roster_mismatch:{sid}")
            if [row["idx"] for row in messages] != list(range(len(messages))) or any(row.get("payload") is None for row in messages):
                gaps.append(f"message_payload_or_order_missing:{sid}")
            coverage = (session.get("state") or {}).get("coverage") or {}
            if coverage.get("status") != "complete":
                gaps.append(f"session_capture_{coverage.get('status', 'unknown')}:{sid}")
            conversations.append({"session": session, "messages": [{"object_id": row["object_id"], "idx": row["idx"], "payload": row.get("payload")} for row in messages]})
        body = {"experiment": experiment, "qid": qid, "trace": traces[0], "qa": qa,
                "candidates": candidates, "judges": judges, "conversations": conversations}
    else:
        raise ValueError(f"unknown evidence kind {kind!r}; expected row_history, question or experiment_overview")
    unknowns: list[str] = []
    if kind == "question":
        records = _question_records(body, gaps, unknowns)
    text = "".join(_record_text(record) for record in records) if records is not None else _json(body)
    capture = {"status": "partial" if gaps else "complete", "scope": capture_scope, "gaps": sorted(set(gaps)),
               "unrecorded_attempts": unrecorded_attempts, "integrity_unknowns": sorted(set(unknowns))}
    return {"schema": SCHEMA, "kind": kind, "id": id, "source_id": _json([kind, id]),
            "sha256": _source_digest(text, capture), "total": len(text), "text": text, "source_capture": capture}


def _typed_row_history(store: Any, name: str) -> dict[str, Any]:
    """Complete typed-key history from one verified, seeded authority view."""
    from .prompts.entity_model import RULE_ENTITY, closure_view
    from .prompts.views import history_events, typed_versions

    scope = store.v0_scope()
    if scope is None:
        raise ValueError("history evidence requires a verified seeded authority epoch")
    section_key = {"tools:loader": "loader", "rules:corpus": "rules"}.get(name)
    versions = ([{"row": dict(row), "releases": carriers} for row, carriers in typed_versions(store, name)]
                if section_key is None else [])
    index = store.typed_index()
    events = history_events(store)
    published = {}
    for order, event in enumerate(events):
        payload = event.get("payload") or {}
        release = payload.get("release") or (payload.get("body") or {}).get("release")
        if event.get("kind") == "publish" and isinstance(release, str):
            published.setdefault(release, order)
    ordered = sorted(scope.releases, key=lambda rid: (rid != scope.root, published.get(rid, len(events)), rid))
    releases, absent, sections = [], [], {}
    for rid in ordered:
        document = store.release(rid).data
        digest = closure_view(document, index).get(name)
        if digest is None:
            absent.append(rid)
            continue
        releases.append(document)
        if section_key is not None:
            content = document["content_refs"][section_key]
            if section_key == "rules":
                content = {"corpus": content["corpus"],
                           "titles": [index[RULE_ENTITY][ref["object_id"]]["title"] for ref in content["items"]],
                           "knowledge_digest": content["knowledge_digest"]}
            version = sections.setdefault(digest, {"digest": digest, "version": digest[:16],
                                                   "content": content, "releases": []})
            version["releases"].append(rid)
    if section_key is not None:
        versions = list(sections.values())
    proposals = []
    for pid in sorted(scope.proposals):
        proposal = store.proposal(pid).data
        if name in (proposal.get("changes") or {}):
            proposals.append({**proposal, "object_id": pid})
    # A removal's child no longer carries the key, but its publication is still this key's history.
    relevant = {document["id"] for document in releases} | {proposal["release"] for proposal in proposals}
    receipts = []
    for event in events:
        payload = event.get("payload") or {}
        release = payload.get("release") or (payload.get("body") or {}).get("release")
        if event.get("kind") in {"publish", "gate", "outcome", "eval-attach"} and release is not None and release not in relevant:
            continue
        receipts.append(event)
    return {"format": "rubiksql.typed-row-history/1", "name": name, "inherited_from": None,
            "versions": versions, "releases": releases, "absent_from": absent,
            "proposals": proposals, "events": receipts}


def _source_digest(text: str, capture: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json({"text": text, "source_capture": capture}).encode("utf-8")).hexdigest()


_EXECUTION_SOURCES = frozenset({"gold_scoring_reexecution", "candidate_scoring_reexecution",
                                "judge_scoring_reexecution", "final_scoring_reexecution"})


def _bound_execution(value: Any, sql: Any, db_id: Any, database_version: Any, *,
                     role: str, gold_sql: Any = None, sql_source: Any = None) -> bool:
    """Reject a retained measurement attached to a different SQL or database."""
    if not isinstance(value, Mapping):
        return False
    difference = value.get("difference")
    if difference is not None and not isinstance(difference, Mapping):
        return False
    copied = value.get("copied_from")
    if copied is not None and (not isinstance(copied, Mapping) or copied.get("role") not in ("gold", "candidate", "judge", "final")
                               or not isinstance(copied.get("id"), str) or not copied["id"]):
        return False
    predecessors = {"gold": (), "candidate": (), "judge": ("gold", "candidate"),
                    "final": ("gold", "candidate", "judge")}
    if role not in predecessors:
        return False
    expected_source = f"{role}_scoring_reexecution" if copied is None else f"{copied['role']}_scoring_reexecution"
    if value.get("source") != expected_source or copied is not None and copied["role"] not in predecessors[role]:
        return False
    return (isinstance(sql, str) and isinstance(database_version, str) and bool(database_version)
            and isinstance(value.get("source"), str) and value["source"] in _EXECUTION_SOURCES and value.get("db_id") == db_id
            and value.get("database_version") == database_version and value.get("sql") == sql
            and isinstance(value.get("sql_source"), Mapping)
            and (sql_source is None or value["sql_source"] == sql_source)
            and value.get("sql_sha256") == hashlib.sha256(sql.encode("utf-8")).hexdigest()
            and (gold_sql is None or difference is None or isinstance(gold_sql, str)
                 and difference.get("gold_sql_sha256") == hashlib.sha256(gold_sql.encode("utf-8")).hexdigest()))


def _experiment_overview_records(rows: Callable[[str, dict[str, Any]], list[dict[str, Any]]],
                                 experiment: str, gaps: list[str]) -> list[dict[str, Any]]:
    """Index every scored-wrong trace without copying question-level evidence."""
    experiments = rows("experiment", {"object_id": experiment})
    traces = sorted(rows("trace", {"exp_id": experiment}), key=lambda row: str(row.get("qid", "")))
    if len(experiments) != 1 or not traces:
        raise ValueError(f"expected one experiment and recorded traces for {experiment!r}")
    qids = [row.get("qid") for row in traces]
    if any(not isinstance(qid, str) or not qid for qid in qids) or len(set(qids)) != len(qids):
        raise ValueError("experiment traces have missing or duplicate question identities")
    qa_by_qid = {trace["qid"]: _qa_for_trace(rows, trace) for trace in traces}

    def ex(row: Mapping[str, Any]) -> Any:
        scores = row.get("scores")
        if not isinstance(scores, Mapping):
            return None
        value = scores.get("ex", scores.get("exd"))
        return value if type(value) in (int, float) else None

    def nonnegative_count(value: Any) -> int | None:
        return value if type(value) is int and value >= 0 else None

    def excerpt(value: Any, limit: int) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = " ".join(value.split())
        return normalized if len(normalized) <= limit else normalized[:limit - 1] + "…"

    def execution_counts(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        difference = value.get("difference")
        return {"status": excerpt(value.get("status"), 32), "row_count": nonnegative_count(value.get("row_count")),
                "extra_count": nonnegative_count(difference.get("extra_count")) if isinstance(difference, Mapping) else None,
                "missing_count": nonnegative_count(difference.get("missing_count")) if isinstance(difference, Mapping) else None}

    wrong = [trace for trace in traces if ex(trace) == 0]
    expected_questions = experiments[0].get("n_questions")
    if type(expected_questions) is not int or expected_questions < 0:
        gaps.append("experiment_question_count_unrecorded_or_invalid")
    elif expected_questions != len(traces):
        gaps.append("experiment_trace_roster_mismatch")
    header = {"record": "experiment_overview", "contract": OVERVIEW_CONTRACT,
              "experiment": experiment, "traces_total": len(traces),
              "wrong_qids": [trace["qid"] for trace in wrong],
              "scored_correct_qids": [trace["qid"] for trace in traces if ex(trace) == 1],
              "unscored_qids": [trace["qid"] for trace in traces if ex(trace) not in (0, 1)],
              "wrong_by_db": dict(sorted(Counter(str(trace.get("db_id")) for trace in wrong).items())),
              "scope": "Measured EX=0 only; question evidence holds complete saved facts and conversations."}
    candidates = rows("candidate", {"exp_id": experiment})
    judges = rows("judge", {"exp_id": experiment})
    session_rows = rows("session", {"state.exp_id": experiment})
    sessions = {session["object_id"]: session for session in session_rows}
    if len(sessions) != len(session_rows):
        gaps.append("session_identity_duplicate")
    trace_qids = set(qids)
    if any(row.get("qid") not in trace_qids for row in [*candidates, *judges]):
        gaps.append("candidate_or_judge_trace_missing")
    records = [header]
    for trace in wrong:
        qid = trace["qid"]
        qa = qa_by_qid[qid]
        question = trace.get("question") if trace.get("question") is not None else qa.get("question")
        normalized_question = " ".join(question.split()) if isinstance(question, str) else ""
        reasons = trace.get("reasons")
        if reasons is not None and (not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons)):
            gaps.append(f"trace_reasons_invalid:{qid}")
            reasons = None
        attempts = trace.get("result_attempts")
        if attempts is not None and not isinstance(attempts, Mapping):
            gaps.append(f"result_attempts_invalid:{qid}")
            attempts = None
        failures = (attempts or {}).get("failures") or []
        if not isinstance(failures, list):
            gaps.append(f"result_attempt_failures_invalid:{qid}")
            failures = []
        if any(not isinstance(failure, Mapping) for failure in failures):
            gaps.append(f"result_attempt_failure_invalid:{qid}")
        if any(isinstance(failure, Mapping) and failure.get("error") is not None
               and not isinstance(failure.get("error"), str) for failure in failures):
            gaps.append(f"result_attempt_error_invalid:{qid}")
        attempt_errors = [excerpt(failure.get("error"), 120) for failure in failures
                          if isinstance(failure, Mapping) and isinstance(failure.get("error"), str)]
        reviewer = trace.get("reviewer")
        reviewer = reviewer if isinstance(reviewer, Mapping) else {}
        card = {"record": "question_card", "qid": qid, "db_id": trace.get("db_id"),
                "question": excerpt(question, 200), "question_truncated": len(normalized_question) > 200,
                "question_source": "trace.question" if trace.get("question") is not None else "qa.question",
                "scores": {"ex": ex(trace)},
                "reasons": [excerpt(reason, 120) for reason in reasons[:4]] if reasons is not None else None,
                "reasons_omitted": max(0, len(reasons) - 4) if reasons is not None else 0,
                "result_attempt_errors": attempt_errors[:2],
                "result_attempt_errors_omitted": max(0, len(attempt_errors) - 2),
                "candidates": [], "judges": [],
                "reviewer": {"active": reviewer.get("active") if type(reviewer.get("active")) is bool else None,
                             "changed": reviewer.get("changed") if type(reviewer.get("changed")) is bool else None},
                "final": {"ex": ex(trace), "row_count": nonnegative_count(trace.get("pred_rows")),
                          "error": excerpt(trace.get("error"), 120),
                          "sql_present": isinstance(trace.get("pred_sql"), str) and bool(trace["pred_sql"])},
                "recorded_session_coverage": "complete",
                "result_details": {"kind": "question", "id": _json([experiment, qid]), "offset": 0}}
        if not normalized_question:
            gaps.append(f"question_unrecorded:{qid}")
        session_ids = trace.get("session_ids")
        if not isinstance(session_ids, list) or any(not isinstance(sid, str) for sid in session_ids):
            card["recorded_session_coverage"] = "unknown"
            session_ids = []
        elif not session_ids:
            card["recorded_session_coverage"] = "unknown"
        elif len(set(session_ids)) != len(session_ids):
            card["recorded_session_coverage"] = "partial"
        for sid in session_ids:
            session = sessions.get(sid)
            if session is None:
                card["recorded_session_coverage"] = "partial"
            else:
                state = session.get("state")
                state = state if isinstance(state, Mapping) else {}
                if state.get("qid") != qid or state.get("db_id") != trace.get("db_id"):
                    card["recorded_session_coverage"] = "partial"
                coverage = state.get("coverage")
                coverage_status = coverage.get("status") if isinstance(coverage, Mapping) else None
                if coverage_status == "partial":
                    card["recorded_session_coverage"] = "partial"
                elif coverage_status != "complete" and card["recorded_session_coverage"] == "complete":
                    card["recorded_session_coverage"] = "unknown"

        def final_session(row: Mapping[str, Any], role: str) -> Mapping[str, Any] | None:
            sid = row.get("session_id")
            session = sessions.get(sid) if isinstance(sid, str) else None
            if session is None:
                card["recorded_session_coverage"] = "partial"
                return None
            state = session.get("state")
            state = state if isinstance(state, Mapping) else {}
            if (state.get("qid") != qid or state.get("db_id") != trace.get("db_id")
                    or (state.get("role") or session.get("agent_ref")) != role or state.get("prior_attempt")
                    or role == "candidate" and state.get("slot") != row.get("slot")):
                card["recorded_session_coverage"] = "partial"
                return None
            coverage = state.get("coverage")
            coverage_status = coverage.get("status") if isinstance(coverage, Mapping) else None
            if coverage_status == "partial":
                card["recorded_session_coverage"] = "partial"
            elif coverage_status != "complete" and card["recorded_session_coverage"] == "complete":
                card["recorded_session_coverage"] = "unknown"
            return session

        effective_gold = qa.get("gold_sql")
        version = qa.get("database_version")
        final_execution = trace.get("pred_execution_result")
        if final_execution is not None and not _bound_execution(final_execution, trace.get("pred_sql"), trace.get("db_id"), version,
                                                                 role="final", gold_sql=effective_gold,
                                                                 sql_source={"kind": "trace_final", "trace_id": trace["object_id"]}):
            gaps.append(f"final_execution_binding_mismatch:{qid}")
            final_execution = None
        card["final"]["execution_result"] = execution_counts(final_execution)
        children = sorted([row for row in candidates if row.get("qid") == qid], key=lambda row: _slot_key(str(row.get("slot", ""))))
        count = trace.get("n_candidates")
        # A saved count is not an allocation budget. Above the actual rows plus one
        # omitted slot, coverage is already impossible: retain the count without
        # expanding an unbounded synthetic roster from a malformed stored value.
        bounded_count = type(count) is int and 0 <= count <= len(children) + 1
        expected = [f"c{index}" for index in range(1, count + 1)] if bounded_count else []
        held = [row.get("slot") for row in children]
        if not bounded_count or sorted(map(str, held)) != sorted(expected):
            gaps.append(f"candidate_roster_mismatch:{qid}")
        card["expected_candidate_count"] = count
        card["expected_slots"] = expected if bounded_count else None
        for row in children:
            final_session(row, "candidate")
            if row.get("db_id") != trace.get("db_id"):
                gaps.append(f"candidate_database_mismatch:{qid}:{row.get('slot')}")
            execution = row.get("execution_result")
            if execution is not None and not _bound_execution(execution, row.get("sql"), row.get("db_id"), version,
                                                               role="candidate", gold_sql=effective_gold,
                                                               sql_source={"kind": "candidate", "candidate_id": row["object_id"]}):
                gaps.append(f"candidate_execution_binding_mismatch:{qid}:{row.get('slot')}")
                execution = None
            card["candidates"].append({"slot": row.get("slot"), "ex": ex(row),
                                       "row_count": nonnegative_count(row.get("row_count")),
                                       "error": excerpt(row.get("error"), 120),
                                       "sql_present": isinstance(row.get("sql"), str) and bool(row["sql"]),
                                       "execution_result": execution_counts(execution)})
        for slot in expected:
            if slot not in held:
                card["candidates"].append({"slot": slot, "ex": None, "row_count": None,
                                           "error": None, "sql_present": False,
                                           "execution_result": None, "missing": "candidate_row_unrecorded"})
        card["candidates"].sort(key=lambda row: _slot_key(str(row["slot"])))
        deciders = [row for row in judges if row.get("qid") == qid]
        if len(deciders) != (1 if type(count) is int and count > 1 else 0):
            gaps.append(f"judge_roster_mismatch:{qid}")
        for row in deciders:
            session = final_session(row, "judge")
            own_sql = row.get("sql")
            sql_source: Any = row.get("sql_source")
            state = (session or {}).get("state")
            state = state if isinstance(state, Mapping) else {}
            if own_sql is not None and (not isinstance(own_sql, str) or not isinstance(sql_source, Mapping)):
                gaps.append(f"judge_sql_source_missing:{qid}")
                own_sql = None
            if row.get("db_id") != trace.get("db_id"):
                gaps.append(f"judge_database_mismatch:{qid}")
            judge_execution = row.get("execution_result")
            if judge_execution is not None and not _bound_execution(judge_execution, own_sql, row.get("db_id"), version,
                                                                     role="judge", gold_sql=effective_gold, sql_source=sql_source):
                gaps.append(f"judge_execution_binding_mismatch:{qid}")
                judge_execution = None
            context = state.get("context")
            context = context if isinstance(context, Mapping) else {}
            card["judges"].append({"picked": excerpt(row.get("picked"), 40), "mode": excerpt(row.get("mode"), 40),
                                   "fallback": row.get("fallback") if type(row.get("fallback")) is bool else None,
                                   "ex": ex(row), "row_count": nonnegative_count(row.get("row_count")),
                                   "error": excerpt(context.get("error"), 120),
                                   "sql_present": own_sql is not None,
                                   "execution_result": execution_counts(judge_execution)})
        if not deciders:
            card["judge_status"] = "disabled" if len(expected) == 1 else "unrecorded"
        card["final"]["origin"] = ("reviewer" if reviewer.get("changed") is True else
                                     "fallback" if len(deciders) == 1 and deciders[0].get("fallback") is True else
                                     "judge" if len(deciders) == 1 and deciders[0].get("fallback") is False and reviewer.get("changed") is False else
                                     "single_candidate" if len(expected) == 1 and reviewer.get("changed") is False else "unknown")
        records.append(card)
    return records


def _facts(row: Mapping[str, Any], omit: Sequence[str] = ()) -> dict[str, Any]:
    # HB storage/identity columns are wrappers, not evaluation facts or authored payload.
    return {key: value for key, value in row.items() if key not in {"object_id", "exp_id", "qid", "db_id", *omit}
            and not key.startswith("_")}


def _slot_key(slot: str) -> tuple[int, str]:
    match = re.fullmatch(r"c(\d+)", slot)
    return (int(match[1]), "") if match else (2 ** 31, slot)


def _session_position(session: Mapping[str, Any]) -> tuple[Any, ...]:
    state = session.get("state") or {}
    role = state.get("role") or session.get("agent_ref") or "unknown"
    slot = str(state.get("slot") or "").split(".attempt", 1)[0] if role == "candidate" else ""
    attempt = state.get("attempt")
    prior = bool(state.get("prior_attempt"))
    return ({"candidate": 0, "judge": 1, "reviewer": 2}.get(role, 3), _slot_key(slot),
            0 if prior else 1, attempt if prior and type(attempt) is int else 2 ** 31, str(session["object_id"]))


def _overview_records(overview: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Keep the first screen factual and compact; move no facts outside required evidence."""
    core = {key: overview[key] for key in ("record", "experiment", "qid", "db_id", "question", "gold_sql")}
    details = []

    def split(value: Mapping[str, Any], fields: Sequence[str], path: list[Any]) -> dict[str, Any]:
        shown = {key: value[key] for key in fields if key in value}
        rest = {key: child for key, child in value.items() if key not in fields}
        if rest:
            details.append({"record": "facts", "id": f"facts-{len(details) + 1:04d}", "path": path, "value": rest})
        return shown

    core["result"] = split(overview["result"], ("pred_sql", "scores", "reasons"), ["result"])
    core["candidates"] = [split(row, ("slot", "sql", "scores", "selected_by_judge"), ["candidates", index])
                          for index, row in enumerate(overview["candidates"])]
    core["judges"] = [split(row, ("picked", "mode", "fallback", "scores"), ["judges", index])
                      for index, row in enumerate(overview["judges"])]
    reviewer = overview.get("reviewer")
    core["reviewer"] = split(reviewer, ("active", "changed", "skipped"), ["reviewer"]) if isinstance(reviewer, Mapping) else reviewer
    core["qa_differences"] = overview["qa_differences"]
    rest = {key: value for key, value in overview.items() if key not in core}
    if rest:
        details.append({"record": "facts", "id": f"facts-{len(details) + 1:04d}", "path": [], "value": rest})
    core["detail_records"] = [record["id"] for record in details]
    return [core, *details]


def _question_records(body: Mapping[str, Any], gaps: list[str], unknowns: list[str]) -> list[dict[str, Any]]:
    trace, qa = body["trace"], body["qa"]
    candidates = sorted(body["candidates"], key=lambda row: _slot_key(str(row.get("slot", ""))))
    judges, conversations = body["judges"], sorted(body["conversations"], key=lambda item: _session_position(item["session"]))
    by_session = {item["session"]["object_id"]: item["session"] for item in conversations}
    identity_differences = []
    gold_execution = qa.get("gold_execution_result")
    effective_gold = qa.get("gold_sql")
    version = qa.get("database_version")
    if gold_execution is not None and not _bound_execution(gold_execution, effective_gold, trace.get("db_id"), version,
                                                           role="gold", sql_source={"kind": "qa_gold", "qa_id": qa["object_id"]}):
        gaps.append("gold_execution_binding_mismatch")
        gold_execution = None
    final_execution = trace.get("pred_execution_result")
    if final_execution is not None and not _bound_execution(final_execution, trace.get("pred_sql"), trace.get("db_id"), version,
                                                             role="final", gold_sql=effective_gold,
                                                             sql_source={"kind": "trace_final", "trace_id": trace["object_id"]}):
        gaps.append("final_execution_binding_mismatch")
        final_execution = None
    candidate_executions = {}
    for candidate in candidates:
        execution = candidate.get("execution_result")
        if execution is not None and not _bound_execution(execution, candidate.get("sql"), candidate.get("db_id"), version,
                                                           role="candidate", gold_sql=effective_gold,
                                                           sql_source={"kind": "candidate", "candidate_id": candidate["object_id"]}):
            gaps.append(f"candidate_execution_binding_mismatch:{candidate.get('slot')}")
            execution = None
        candidate_executions[candidate.get("object_id")] = execution
    judge_executions = {}
    for judge in judges:
        execution = judge.get("execution_result")
        if execution is not None and not _bound_execution(execution, judge.get("sql"), judge.get("db_id"), version,
                                                           role="judge", gold_sql=effective_gold, sql_source=judge.get("sql_source")):
            gaps.append("judge_execution_binding_mismatch")
            execution = None
        if judge.get("sql") is not None and not isinstance(judge.get("sql_source"), Mapping):
            gaps.append("judge_sql_source_missing")
        judge_executions[judge.get("object_id")] = execution
    for role, children in (("candidate", candidates), ("judge", judges)):
        for child in children:
            if child.get("db_id") != trace.get("db_id"):
                identity_differences.append({"entity": role, "object_id": child.get("object_id"), "field": "db_id",
                                             "trace": trace.get("db_id"), "recorded": child.get("db_id")})
                gaps.append(f"{role}_database_mismatch:{child.get('object_id')}")
            linked = by_session.get(child.get("session_id"))
            if linked is None:
                continue  # The expected-session check already reports the missing link.
            state = linked.get("state") or {}
            if ((state.get("role") or linked.get("agent_ref")) != role or state.get("prior_attempt")
                    or role == "candidate" and state.get("slot") != child.get("slot")):
                gaps.append(f"{role}_session_link_mismatch:{child.get('session_id')}")
    if len(candidates) != trace.get("n_candidates") or not candidates:
        gaps.append("candidate_roster_missing")
    if len({row.get("slot") for row in candidates}) != len(candidates):
        gaps.append("candidate_slot_duplicate")
    from .question import question_with_hint

    qa_question = question_with_hint(qa["question"], qa.get("hint"))
    qa_values = {"question": qa_question, "gold_sql": qa.get("gold_sql"), "gold_rows": qa.get("gold_rows"), "db_id": qa.get("db_id")}
    differences = {key: {"run": trace.get(key), "qa": value} for key, value in qa_values.items()
                   if value is not None and trace.get(key) != value}
    picked = judges[0].get("picked") if len(judges) == 1 else None
    qa_facts = _facts(qa, ("name", "question", "gold_sql", "gold_rows"))
    qa_facts["gold_execution_result"] = gold_execution
    trace_facts = _facts(trace, ("question", "gold_sql", "session_ids", "reviewer", "result_attempts", "gold_execution_result"))
    trace_facts["pred_execution_result"] = final_execution
    overview = {"record": "overview", "experiment": body["experiment"], "qid": body["qid"], "db_id": trace.get("db_id"),
                "question": trace.get("question"), "gold_sql": trace.get("gold_sql"),
                "qa": qa_facts, "qa_differences": differences,
                "result": trace_facts,
                "candidates": [{**_facts(row), "execution_result": candidate_executions[row.get("object_id")],
                                "selected_by_judge": row.get("slot") in picked.split(",") if picked else None} for row in candidates],
                "judges": [{**_facts(row), "execution_result": judge_executions[row.get("object_id")]} for row in judges],
                "reviewer": trace.get("reviewer"), "result_attempts": trace.get("result_attempts")}
    if identity_differences:
        overview["identity_differences"] = identity_differences
    records = _overview_records(overview)
    roles: dict[str, list[Mapping[str, Any]]] = {}
    attempts: set[tuple[str, str, int]] = set()
    for conversation in conversations:
        session, messages = conversation["session"], conversation["messages"]
        sid, state = session["object_id"], session.get("state") or {}
        role = state.get("role") or session.get("agent_ref") or "unknown"
        roles.setdefault(role, []).append(session)
        if role not in ("candidate", "judge", "reviewer"):
            gaps.append(f"session_role_unknown:{sid}")
        if state.get("exp_id") != body["experiment"] or state.get("qid") != body["qid"]:
            gaps.append(f"session_identity_mismatch:{sid}")
        if state.get("db_id") != trace.get("db_id"):
            gaps.append(f"session_database_mismatch:{sid}")
        if state.get("role") and session.get("agent_ref") and state["role"] != session["agent_ref"]:
            gaps.append(f"session_role_mismatch:{sid}")
        slot = str(state.get("slot") or "").split(".attempt", 1)[0] if role == "candidate" else ""
        if role == "candidate" and not re.fullmatch(r"c\d+", slot):
            gaps.append(f"candidate_slot_unknown:{sid}")
        if state.get("prior_attempt"):
            parent = by_session.get(state.get("parent_session_id"))
            if state.get("parent_session_id") and (parent is None or _session_position(parent)[:2] != _session_position(session)[:2]
                                                   or (parent.get("state") or {}).get("prior_attempt")):
                gaps.append(f"attempt_parent_mismatch:{sid}")
            if type(state.get("attempt")) is int:
                attempt_key = role, slot, state["attempt"]
                if attempt_key in attempts:
                    gaps.append(f"attempt_order_duplicate:{sid}")
                attempts.add(attempt_key)
        roster = list(session.get("message_ids") or [])
        if len(set(roster)) != len(roster) or len({row["object_id"] for row in messages}) != len(messages):
            gaps.append(f"message_duplicate:{sid}")
        if any(not isinstance(row.get("payload"), Mapping) for row in messages):
            gaps.append(f"message_payload_missing:{sid}")
        if any(type(row.get("idx")) is not int for row in messages):
            gaps.append(f"message_index_invalid:{sid}")
        payloads = [row.get("payload") for row in messages]
        digests = [hashlib.sha256(_json(payload).encode("utf-8")).hexdigest() for payload in payloads]
        for key, actual in (("messages_sha256", hashlib.sha256(_json(payloads).encode("utf-8")).hexdigest()), ("message_sha256", digests)):
            if state.get(key) is None:
                unknowns.append(f"{key}_unrecorded:{sid}")
            elif state[key] != actual:
                gaps.append(f"{key}_mismatch:{sid}")
        if state.get("prior_attempt") and (type(state.get("attempt")) is not int or state["attempt"] < 1):
            unknowns.append(f"attempt_order_unrecorded:{sid}")
        header = {"record": "session", "session_id": sid, "role": role,
                  "slot": slot,
                  "status": session.get("status"), "usage": session.get("usage"),
                  "attempt_order": ("recorded" if type(state.get("attempt")) is int and state["attempt"] >= 1 else "unknown") if state.get("prior_attempt") else "final",
                  "state": _facts(state, ("role", "slot", "message_sha256", "messages_sha256")),
                  "message_count": len(messages)}
        if state.get("slot") != slot:
            header["stored_slot"] = state.get("slot")
        if state.get("db_id") != trace.get("db_id"):
            header["state"]["db_id"] = state.get("db_id")
        summary = session.get("summary")
        if summary and summary not in [trace.get("pred_sql"), *(row.get("sql") for row in candidates)]:
            header["summary"] = summary
        records.append(header)
        records.extend({"record": "message", **row} for row in messages)
    # Optional roles are explicit, and a retry-only role does not masquerade as a final session.
    for role in ("candidate", "judge", "reviewer"):
        held = roles.get(role, [])
        final = [session for session in held if not (session.get("state") or {}).get("prior_attempt")]
        if final:
            continue
        status, reason = "missing", "expected final conversation not recorded"
        if role == "judge" and trace.get("n_candidates") == 1 and not judges:
            status, reason = "disabled", "single-candidate path has no judge"
        elif role == "reviewer":
            reviewer = trace.get("reviewer")
            if isinstance(reviewer, Mapping) and reviewer.get("skipped"):
                status, reason = "skipped", str(reviewer["skipped"])
            elif isinstance(reviewer, Mapping) and reviewer.get("active") is False:
                status, reason = "disabled", "reviewer explicitly inactive"
            elif not isinstance(reviewer, Mapping) or reviewer.get("active") is None:
                status, reason = "unknown", "reviewer activation was not recorded"
        if status in ("missing", "unknown"):
            gaps.append(f"{role}_session_{status}")
        marker = {"record": "role_status", "role": role, "status": status, "reason": reason, "recorded_retries": len(held)}
        # Insert at that role's chapter boundary, even when there is no session header.
        rank = {"candidate": 0, "judge": 1, "reviewer": 2}[role]
        index = next((i for i, item in enumerate(records) if item.get("record") in ("session", "role_status")
                      and {"candidate": 0, "judge": 1, "reviewer": 2}.get(item.get("role"), 3) > rank), len(records))
        records.insert(index, marker)
    if len(judges) > 1:
        gaps.append("judge_roster_duplicate")
    elif trace.get("n_candidates", 0) > 1 and not judges:
        gaps.append("judge_result_missing")
    return records


def _page_layout(source: Mapping[str, Any]) -> list[tuple[int, int, list[dict[str, Any]]]]:
    text = source["text"]
    if source["schema"] != SCHEMA:
        raise ValueError(f"unsupported evidence schema {source['schema']!r}")
    # JSONL uses LF only; U+2028/U+2029 inside an original payload are not delimiters.
    parts = text.split("\n")
    lines = ([part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
             if source["kind"] in ("question", "experiment_overview") else [text])
    units = []
    cursor = 0
    context: dict[str, Any] = {}
    for line in lines:
        record = json.loads(line)
        kind = record.get("record", source["kind"])
        if kind == "session":
            context = {key: record[key] for key in ("role", "slot", "session_id")}
        elif kind != "message":
            context = {}
        identity = {"record": kind, **context}
        if kind == "message":
            identity.update(message_id=record["object_id"], idx=record["idx"])
        elif kind == "role_status":
            identity["role"] = record["role"]
        elif kind == "facts":
            identity.update(record_id=record["id"], path=record["path"])
        elif kind == "question_card":
            identity["qid"] = record["qid"]
        units.append((cursor, cursor + len(line), identity))
        cursor += len(line)
    pages: list[tuple[int, int, list[dict[str, Any]]]] = []
    start = end = 0
    segments: list[dict[str, Any]] = []
    for left, right, identity in units:
        size = right - left
        if segments and size <= PAGE_CHARS and end - start + size > PAGE_CHARS:
            pages.append((start, end, segments))
            segments = []
        if size > PAGE_CHARS:
            position = left
            while position < right:
                if not segments:
                    start = position
                stop = min(right, position + PAGE_CHARS - (position - start))
                segment = {**identity, "start": position - left, "end": stop - left, "total": size,
                           "continued": position > left, "continues": stop < right}
                segments.append(segment)
                end = position = stop
                if end - start == PAGE_CHARS:
                    pages.append((start, end, segments))
                    segments = []
            continue
        if not segments:
            start = left
        end = right
        segments.append({**identity, "start": 0, "end": size, "total": size, "continued": False, "continues": False})
        if end - start == PAGE_CHARS:
            pages.append((start, end, segments))
            segments = []
    if segments:
        pages.append((start, end, segments))
    return pages


def evidence_offsets(source: Mapping[str, Any]) -> list[int]:
    """Return canonical page starts; callers must not guess fixed character increments.

    Args:
        source: Complete source under the current schema.

    Returns:
        Character offsets of all message-boundary pages, in delivery order.

    Raises:
        ValueError: The schema or source text is invalid.
    """
    return [start for start, _, _ in _page_layout(source)]


def evidence_page(source: Mapping[str, Any], offset: int = 0, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Return one original-text page with an explicit continuation and record boundaries.

    Args:
        source: A complete trusted source from :func:`evidence_source`.
        offset: Zero for the first page, then exactly the preceding page's ``next.offset``.
        expected_sha256: Digest from the preceding page; a changed source is refused.

    Returns:
        Identity, capture status, exact text slice, segment ranges and next tool arguments.
        Consecutive complete messages use a compact ``message_range`` with session and
        first/last indices; every message's identity and payload remain in the text.
        Segment ranges count characters in the serialized record (including its newline),
        not characters in the decoded message payload. Oversized records alone are split.

    Raises:
        ValueError: Offset is not a page boundary inside the source.
    """
    if expected_sha256 is not None and expected_sha256 != source["sha256"]:
        raise ValueError("evidence source changed; restart at offset 0")
    layout = _page_layout(source)
    index = next((i for i, (start, _, _) in enumerate(layout) if type(offset) is int and start == offset), None)
    if index is None:
        raise ValueError("offset must be a canonical page boundary; use the returned next arguments")
    return _page_at(source, layout, index)


def _compact_segments(segments: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Describe complete message runs once; keep every partial-message boundary explicit."""
    result: list[dict[str, Any]] = []
    for segment in segments:
        if segment["record"] != "message" or segment["continued"] or segment["continues"]:
            result.append(dict(segment))
            continue
        identity = {key: segment[key] for key in ("role", "slot", "session_id")}
        previous = result[-1] if result else {}
        if (previous.get("record") == "message_range" and all(previous[key] == value for key, value in identity.items())
                and type(previous["last_idx"]) is int and type(segment["idx"]) is int and previous["last_idx"] + 1 == segment["idx"]):
            previous["last_idx"] = segment["idx"]
            previous["count"] += 1
        else:
            result.append({"record": "message_range", **identity, "first_idx": segment["idx"], "last_idx": segment["idx"], "count": 1})
    return result


def _page_at(source: Mapping[str, Any], layout: Sequence[tuple[int, int, list[dict[str, Any]]]], index: int) -> dict[str, Any]:
    offset, end, segments = layout[index]
    page = {**source, "offset": offset, "end": end, "text": source["text"][offset:end]}
    next_offset = layout[index + 1][0] if index + 1 < len(layout) else None
    page.update(page_index=index, page_count=len(layout), segments=_compact_segments(segments), end_of_evidence=next_offset is None,
                next=None if next_offset is None else {"kind": source["kind"], "id": source["id"], "offset": next_offset,
                                                     "expected_sha256": source["sha256"]})
    return page


def evidence_pages(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compile all canonical pages with one parse of the source text.

    Args:
        source: Complete trusted evidence source.

    Returns:
        Ordered page envelopes. Callers caching these must not expose mutable references.
    """
    layout = _page_layout(source)
    return [_page_at(source, layout, index) for index in range(len(layout))]


def assemble_evidence(pages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate a complete ordered page chain and recover its immutable source.

    Args:
        pages: Actual page envelopes, beginning at zero and ending at the source's end.

    Returns:
        The original source, including complete text and capture status.

    Raises:
        ValueError: Pages are missing, altered, mixed across sources, or use a changed source.
    """
    if not pages:
        raise ValueError("evidence page chain is empty")
    keys = ("schema", "kind", "id", "source_id", "sha256", "total", "source_capture")
    try:
        source = {key: pages[0][key] for key in keys}
        text = "".join(page["text"] for page in pages)
        source["text"] = text
        if len(text) != source["total"] or _source_digest(text, source["source_capture"]) != source["sha256"]:
            raise ValueError("evidence source length or digest changed")
        if list(pages) != evidence_pages(source):
            raise ValueError("evidence pages do not match the canonical complete chain")
    except (KeyError, TypeError) as exc:
        raise ValueError("malformed evidence page chain") from exc
    return source


def required_sources(changes: Mapping[str, Any], proposal: Mapping[str, Any], experiment: str) -> list[tuple[str, str]]:
    """Derive required history from the actual merged diff and questions cited by the proposal.

    Args:
        changes: The canonical proposal diff, including inherited merge changes.
        proposal: The input proposal with hypothesis evidence and rules ``dossier_qids``.
        experiment: Exact parent experiment identity.

    Returns:
        Deterministically ordered evidence source keys.

    Raises:
        ValueError: Evidence is malformed or the proposal cites no parent question.
    """
    qids: set[str] = set()

    def add(value: Any, where: str) -> None:
        if not isinstance(value, (list, tuple)) or any(not isinstance(qid, str) or not re.fullmatch(r"Q\d{5}", qid) for qid in value):
            raise ValueError(f"{where} must list parent question IDs such as Q00001")
        qids.update(value)

    hypothesis = proposal.get("hypothesis")
    if not isinstance(hypothesis, Mapping):
        raise ValueError("hypothesis must contain structured evidence with parent question IDs")
    evidence = hypothesis.get("evidence")
    if not isinstance(evidence, list) or any(not isinstance(item, Mapping) or "qids" not in item for item in evidence):
        raise ValueError("hypothesis.evidence must be a list of objects containing qids")
    for item in evidence:
        add(item["qids"], "hypothesis.evidence.qids")

    def cited(value: Any) -> None:
        if isinstance(value, Mapping):
            if "dossier_qids" in value:
                add(value["dossier_qids"], "dossier_qids")
            for child in value.values():
                cited(child)
        elif isinstance(value, list):
            for child in value:
                cited(child)

    cited(proposal.get("changes") or [])
    if not qids:
        raise ValueError("the proposal must cite at least one parent question in evidence.qids or dossier_qids")
    return [("row_history", name) for name in sorted(changes)] + [("question", _json([experiment, qid])) for qid in sorted(qids)]


def store_history_problems(store: Any, sources: Sequence[Mapping[str, Any]]) -> list[str]:
    """Compare delivered lab history with exact rows in the verified current epoch.

    Args:
        store: Verified authority or snapshot history mapping supplied by the caller.
        sources: Independently fetched evidence sources.

    Returns:
        list[str]: Missing versions/proposals or differences in authored content.
    """
    scope = store.v0_scope()
    if scope is None:
        raise ValueError("history evidence requires a verified seeded authority epoch")
    proposals = {pid: store.proposal(pid).data for pid in scope.proposals}
    problems = []
    for source in sources:
        if source.get("kind") != "row_history":
            continue
        name = str(source["id"])
        body = json.loads(str(source["text"]))
        if ":" in name:
            expected = _typed_row_history(store, name)
            for field, value in expected.items():
                if body.get(field) != value:
                    problems.append(f"{name}: lab typed history {field} differs from the canonical store")
            if set(body) != set(expected):
                problems.append(f"{name}: lab typed history fields differ from the canonical store")
            continue
        versions = store.v0_versions(scope, name)
        inherited = body.get("inherited_from")
        history_name = name
        if inherited is not None:
            if (inherited != "candidate" or not re.fullmatch(r"candidate\.s[1-3]", name)
                    or versions or body.get("name") != name):
                problems.append(f"{name}: invalid shared-candidate history inheritance")
                continue
            history_name = "candidate"
            versions = store.v0_versions(scope, history_name)
            if not versions or any(row.get("name") != history_name for row in body["versions"]):
                problems.append(f"{name}: inherited history source differs from shared candidate")
                continue
            if (len({row["version"] for row in body["versions"]}) != len(body["versions"])
                    or len({row["object_id"] for row in body["proposals"]}) != len(body["proposals"])):
                problems.append(f"{name}: inherited history roster contains duplicates")
                continue
        elif not versions and re.fullmatch(r"candidate\.s[1-3]", name):
            problems.append(f"{name}: missing shared-candidate history inheritance")
            continue
        manifests = {row["version"]: row["manifest"] for row in body["versions"]}
        if set(manifests) != versions:
            problems.append(f"{name}: lab version roster differs from the canonical store")
        for version in sorted(versions & set(manifests)):
            expected = store.object(history_name, version).data["row"]["manifest"]
            if manifests[version] != expected:
                problems.append(f"{name}:{version}: lab authored content differs from the canonical store")
        relevant = {pid: proposal for pid, proposal in proposals.items() if history_name in (proposal.get("changes") or {})}
        held = {str(row["object_id"]): row for row in body["proposals"]}
        if set(held) != set(relevant):
            problems.append(f"{name}: lab proposal roster differs from the canonical store")
        for pid in sorted(relevant.keys() & held.keys()):
            original, projected = relevant[pid], held[pid]
            if any(projected.get(key) != original.get(key) for key in ("parents", "release", "changes", "reason", "hypothesis", "author")):
                problems.append(f"{pid}: lab proposal content differs from the canonical store")
    return problems


def _tool_payload(value: Any) -> dict[str, Any] | None:
    """Decode a complete tool text result; structured metadata alone proves no delivery."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    if isinstance(value, Mapping):
        if value.get("isError") or value.get("is_error"):
            return None
        if "content" in value:
            return _tool_payload(value["content"])
        return dict(value) if "structuredContent" not in value else None
    if isinstance(value, list):
        texts = [block.get("text", "") for block in value if isinstance(block, Mapping) and block.get("type") == "text"]
        if len(texts) == 1:
            return _tool_payload(texts[0])
    return None


def _cli_tool_receipts(log: Path) -> list[dict[str, Any]]:
    """Pair completed lab tool calls and textual returns in a single native CLI stream."""
    calls: dict[str, tuple[str, Mapping[str, Any]]] = {}
    duplicate_calls: set[str] = set()
    returned: set[str] = set()
    duplicate_returns: set[str] = set()
    receipts: list[dict[str, Any]] = []
    claude_receipts: list[tuple[str, dict[str, Any]]] = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, Mapping):
            continue
        item = event.get("item") or {}
        if (event.get("type") == "item.completed" and item.get("type", item.get("item_type")) == "mcp_tool_call"
                and item.get("server") == "lab" and item.get("status") == "completed" and not item.get("error")):
            tool = item.get("tool")
            if tool in ("read_evidence", "run_analysis_sql"):
                arguments = item.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except ValueError:
                        continue
                result = _tool_payload(item.get("result"))
                if isinstance(arguments, Mapping) and result is not None:
                    receipts.append({"tool": tool, "arguments": dict(arguments), "result": result,
                                     "receipt_id": item.get("id")})
        message = event.get("message") or {}
        if event.get("type") not in ("assistant", "user") or not isinstance(message, Mapping):
            continue
        for block in message.get("content") or []:
            if not isinstance(block, Mapping):
                continue
            if block.get("type") == "tool_use" and block.get("name") in ("mcp__lab__read_evidence", "mcp__lab__run_analysis_sql"):
                call_id = str(block.get("id"))
                if call_id in calls:
                    duplicate_calls.add(call_id)
                arguments = block.get("input") or {}
                if isinstance(arguments, Mapping):
                    calls[call_id] = (block["name"].removeprefix("mcp__lab__"), dict(arguments))
            elif block.get("type") == "tool_result" and not block.get("is_error"):
                call_id = str(block.get("tool_use_id"))
                if call_id not in calls:
                    continue
                if call_id in returned:
                    duplicate_returns.add(call_id)
                returned.add(call_id)
                result = _tool_payload(block.get("content"))
                if result is not None:
                    tool, arguments = calls[call_id]
                    claude_receipts.append((call_id, {"tool": tool, "arguments": arguments, "result": result,
                                                      "receipt_id": call_id}))
    return receipts + [receipt for call_id, receipt in claude_receipts if call_id not in duplicate_calls | duplicate_returns]


def delivered_pages(log: Path) -> list[dict[str, Any]]:
    """Read completed saved-evidence pages from one CLI stream, never agent prose.

    Args:
        log: A single Codex JSONL or Claude stream-json log.

    Returns:
        Decoded pages whose actual tool call and successful completion are both recorded.
    """
    pages = []
    for receipt in _cli_tool_receipts(log):
        page = receipt["result"]
        if (receipt["tool"] == "read_evidence" and page.get("schema") == SCHEMA
                and page.get("kind") in ("question", "experiment_overview", "row_history", "analysis_report")
                and _matches_call(page, receipt["arguments"])):
            pages.append(page)
    return pages


def diagnostic_receipts(log: Path) -> list[dict[str, Any]]:
    """Return only paired execution and temporary-page replies actually delivered in this round."""
    accepted = []
    for receipt in _cli_tool_receipts(log):
        arguments, result = receipt["arguments"], receipt["result"]
        if receipt["tool"] == "run_analysis_sql" and _matches_diagnostic_run(result, arguments):
            accepted.append(receipt)
        elif (receipt["tool"] == "read_evidence" and result.get("schema") == SCHEMA and result.get("kind") == "analysis_result"
              and _matches_call(result, arguments)):
            accepted.append(receipt)
    return accepted


def _matches_diagnostic_run(result: Mapping[str, Any], arguments: Mapping[str, Any]) -> bool:
    if (result.get("schema") != "rubiksql.analysis-run/1"
            or result.get("status") not in ("succeeded", "failed", "timed_out", "resource_exceeded", "running", "interrupted", "expired")
            or not isinstance(result.get("sql_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", result["sql_sha256"]) is None):
        return False
    if not all(result.get(key) == arguments.get(key) for key in ("request_id", "experiment", "qid", "source")):
        return False
    if arguments.get("source") == "custom" and (not isinstance(arguments.get("sql"), str)
                                                 or result.get("sql_sha256") != hashlib.sha256(arguments["sql"].encode("utf-8")).hexdigest()):
        return False
    if arguments.get("source") != "custom" and arguments.get("sql") is not None:
        return False
    page = result.get("page")
    return (result.get("status") != "succeeded" and page is None or
            result.get("status") == "succeeded" and isinstance(page, Mapping) and page.get("schema") == SCHEMA
            and page.get("kind") == "analysis_result" and page.get("offset") == 0 and page.get("id") == result.get("result_id"))


def _matches_call(page: Mapping[str, Any], arguments: Any) -> bool:
    if not isinstance(arguments, Mapping):
        return False
    try:
        return (page["kind"] == arguments.get("kind") and page["id"] == _canonical_id(page["kind"], arguments.get("id"))
                and page["offset"] == arguments.get("offset", 0)
                and (arguments.get("expected_sha256") is None or arguments["expected_sha256"] == page["sha256"])
                and (page["offset"] == 0 or arguments.get("expected_sha256") == page["sha256"]))
    except (TypeError, ValueError, KeyError):
        return False


def reading_coverage(sources: Sequence[Mapping[str, Any]], pages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare delivered originals against independent trusted sources, preserving capture gaps.

    Args:
        sources: Full sources independently fetched by the driver.
        pages: Actual completed tool results in this round's CLI log.

    Returns:
        Required-source coverage and ``passed``. Complete delivery does not prove understanding.
    """
    reports = []
    for source in sources:
        required = {page["offset"]: page for page in evidence_pages(source)}
        matches = {int(page["offset"]) for page in pages if type(page.get("offset")) is int and page.get("source_id") == source["source_id"]
                   and page == required.get(page["offset"])}
        missing = sorted(set(required) - matches)
        reports.append({key: source[key] for key in ("kind", "id", "source_id", "sha256", "total", "source_capture")} | {
            "delivered": sum(required[offset]["end"] - offset for offset in matches), "missing_offsets": missing,
            "delivery_complete": not missing})
    return {"schema": "rubiksql.reading-coverage/1", "passed": bool(reports) and all(row["delivery_complete"] and row["source_capture"]["status"] == "complete" for row in reports),
            "meaning": "exact saved evidence delivered in this round; not proof of understanding", "sources": reports}
