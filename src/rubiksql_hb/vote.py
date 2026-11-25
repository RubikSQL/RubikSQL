"""
nl2sql_vote — self-consistency ensemble with an agentic judge model.

Opens ONE MCP session for the entire vote pipeline:
  1. N sub-agents run concurrently on that session (candidates)
  2. Judge runs on the same session — no second MCP startup

The judge runs an agentic loop (one prompt covers complete and truncated result states) and ALWAYS
finalizes its answer via submit_final_sql, so the scored rows come from SQL execution
— never the model's transcribed rows. Tools: run_sql, submit_final_sql, get_rules,
get_examples, get_col_info / get_tab_info / get_db_info, search_columns / search_values,
save_sql_as_table / list_temp_tables. The top relevant business rules are also pre-injected
into the judge's prompt (it can call get_rules for more).

Usage:
    from rubiksql_hb.vote import nl2sql_vote
    result = nl2sql_vote("total revenue", "customers_demo")
"""

from contextlib import nullcontext
from copy import deepcopy
from functools import partial
import hashlib
import inspect
import json
import math
import os
import threading
import re as _re
import sys
import asyncio
import time
from collections import deque
import uuid

from mcp import ClientSession
from .experiment_gateway import GatewayConfigurationError, GatewayUnavailableError
from . import hint_conflicts
from . import llm_runtime as _llm_runtime_module
from .llm_runtime import (TRANSPORT_PROTOCOL, achat_turn, bind_retry_horizon, build_llm, merge_usage,
                          selected_transport_protocol)
from .model_render import (
    MODEL_PROTOCOL,
    fenced_sql,
    model_tool_content,
    render_rules,
    table_display as _table_display,
)
from .prompts import RenderRecorder
from .prompts.canon import canon, sha256_hex
from .sql_action import SQL_ACTIONS, rank_request as _rank_request, publish_rank_result, rank_publication_enabled
from .turn_memo import RunClock, TurnMemo
from . import cache_lineage
from . import deadline_events
from . import phase_budget
from . import judge_majority
from . import judge_checkpoints
from . import session_timing
from . import tool_contract
from .metrics.official_ex import result_signature as _result_sig
from .consensus_guard import IMPLEMENTATION_VERSION as CONSENSUS_GUARD_IMPLEMENTATION_VERSION
from .consensus_guard import STRICT_ADDED_TABLES_VERSION as CONSENSUS_GUARD_STRICT_ADDED_TABLES_VERSION
from .consensus_guard import ANSWERLESS_FALLBACK_VERSION, DEFECT_FALLBACK_VERSION
from .consensus_guard import (
    apply_answerless_fallback, apply_consensus_guard, apply_defect_fallback, plurality_slot, result_groups,
)
from .count_weight import IMPLEMENTATION_VERSION as COUNT_WEIGHT_IMPLEMENTATION_VERSION, apply_count_weight
from .entity_percentage_count import IMPLEMENTATION_VERSION as ENTITY_PERCENTAGE_COUNT_IMPLEMENTATION_VERSION, apply_entity_percentage_count
from .selection_normalization import IMPLEMENTATION_VERSION as SELECTION_NORMALIZATION_IMPLEMENTATION_VERSION, apply_selection_normalization
from .submission_gate import IMPLEMENTATION_VERSION as SUBMISSION_GATE_IMPLEMENTATION_VERSION, apply_submission_gate
from .submission_gate import HINT_AGGREGATE_VERSION as SUBMISSION_GATE_HINT_AGGREGATE_VERSION

# The shared per-db MCP session machinery now lives in nl2sql (used by both single and
# vote modes); import it here. _PROMPTS is the process's pinned prompt release: the judge
# and reviewer texts below are its surfaces, exactly as the candidate's are.
from .nl2sql import (
    _DeadSessionCandidatesError,
    _SharedSessionDeadline,
    CANDIDATE_CUT_ERROR,
    _detach_close_session,
    tool_input_schema,
    _cache_get, _cache_set, _db_knowledge_fingerprint,
    _call_mcp, _raw_call, _strip_db_id, _log, _normalize_tool_name,
    _LOG_META_PREFIX, _tool_result_content, _trace, _trace_tool_call, hint_conflict_frame,
    _register_tool_props, _parse_tool_args, _normalize_tool_args, _replayable_calls,
    _gather_in_session,
    _gather_mixed_in_session,
    _get_shared_session, _DEAD_SESSION_EXCS, _SessionSubmissionRejected,
    server_scope, server_slot,
    _SESSION_ID, _QUESTION_HINTS, _ORIGINAL_QUESTION,
    _PROVIDERS,
    _RuleTriggerState,
    _async_synonym_context,
    _LEGACY_CACHE_IDENTITY, _PROMPTS, _recorder_for,
    _TURN_STORE, _replayed_turn, _turn_lookup_allowed,
)
from .question import question_hints, question_with_hint
from .knowledge_failures import degraded_knowledge_result, degraded_record

# ── display config ────────────────────────────────────────────────────────────

# Shares RUBIKSQL_RUN_SQL_MAX_ROWS with the candidate side, because the two views are of the
# same rows and a reader comparing them should not have to know they were cut at different
# places: raising the candidates' window while the judge keeps a 64-row one would leave the
# judge deciding on strictly less than the candidates saw. Both default to 64, so an
# env-less run is byte-identical.
#
# The execution identity also keys an env override: a small initial preview can be unchanged
# while later SQL tool results differ under the new cap. The reviewer shares that contract.
_JUDGE_MAX_ROWS  = _PROMPTS.knob("run_sql_max_rows")   # config.knobs; RUBIKSQL_RUN_SQL_MAX_ROWS overrides
_JUDGE_MAX_TURNS = 200  # runaway guard: cap total judge turns
_JUDGE_MAX_NUDGES = 2   # B1: times we re-prompt a judge that stopped without submitting
# §13 reviewer layer (judge-external): sees question+hint, the chosen SQL and its
# truncated result table; repairs OBJECTIVE defects only. Checklist priorities follow
# the §12 error-cause distribution (A shape 64-66%, defensive guards, hint fidelity,
# grain, column source); with gold_suspect≈47% the hard rule is conservatism — no
# taste rewrites.
# The v2 scalpel is the PRODUCTION DEFAULT (user ruling 07-21, after the double-reproduction
# gate REV2C +2 / REV2C-R +4 and the substrate gate REV2-DNP +2): an unset env runs it.
# RUBIKSQL_REVIEWER=0 is the one supported override — it disables the layer for
# reviewer-free reproduction outside the official pipeline, which clears the override.
_REVIEWER = _PROMPTS.knob("reviewer")   # config.knobs; RUBIKSQL_REVIEWER=0 still turns it off
# seed env-overridable for independent replication runs (M10 double-reproduction
# precedent): a different seed resamples every reviewer decision, testing whether
# marginal calls (the Q00027/Q00036 simultaneity) survive; default byte-identical.
# run_official_pipeline scrubs it (_REV_ENV) so production always runs the default.
_REVIEWER_SEED = int(os.environ.get("RUBIKSQL_REVIEWER_SEED", "45") or "45")

# v2 (§13 4×2 matrix verdict): the retired v1 checklist's discretionary checks
# (grain/DISTINCT, hint arithmetic, column source) produced the stable misfire clusters
# (Q00281 4/4, Q01227 3/4) while ALL cross-substrate-stable fixes came from three rules.
# v2 keeps those three, demands a cited objective signature before any change, and
# hard-forbids the misfire territories. Rule 4 (decoded-label, 07-30) is the reviewer-side
# port of the E1 raw-vs-decoded verdict: unlike E1 it strips unilaterally with no raw
# sibling in evidence, so it ships with its own preregistered audit (see the
# decoded-label gate comment in _v2_risk_signal) instead of inheriting E1's 0/54 record.


# Imperative ranking request ("Rank <entities> by <measure>"): the question asks for the ranking
# itself, so a final whose SELECT list lacks the measure or the RANK() column needs the reviewer's
# projection-only signature (b) even when its result shape carries no other signal (a one-column
# ordered list). Keyed on the imperative wording alone: rank as a stored attribute or a position
# ("ranked first", "the rank of", "highest rank", "top 5 ranked") never matches.
def _v2_risk_signal(chosen: dict, question: str | None = None) -> str:
    """v2 scope gate: review only rows carrying a mechanical defect signature.
    Empty/NULL results, wide outputs and UNION'd slot rows are where every
    cross-substrate-stable v1 fix lived; everything else is pure misfire exposure.

    Args:
        chosen (dict): The judge's final record (SQL, rows, row count, judge debug).
        question (str | None): The question with its hint. Only the imperative ranking
            request reads it, at the lowest precedence, so every result-shape signal keeps
            its name.

    Returns:
        str: The signal name, or ``""`` when the row carries none.
    """
    rows = chosen.get("rows") or []
    if chosen.get("row_count", len(rows)) == 0:
        return "empty-result"
    first = rows[0] if rows else []
    # runtime rows are DICTS (run_sql tool JSON); tests/history used lists — accept both.
    # (the shape signals were silently dead for dict rows until 07-23: Q00236 anatomy)
    vals = (list(first.values()) if isinstance(first, dict)
            else list(first) if isinstance(first, (list, tuple)) else [first])
    if any(v is None for v in vals):
        return "null-cells"
    if len(vals) >= 3:
        return "wide-output"
    # two-column outputs: the column-ORDER convention territory (Q00050 anatomy — the
    # reference follows the question's phrasing order; a paraphrase-order SELECT scores 0
    # under official EX). The prompt only allows a reorder with the exact signature.
    if len(vals) == 2:
        return "two-col"
    # tall shape (A2): numbered slot columns UNION'd into extra rows (Q00036 anatomy);
    # legit multi-entity UNIONs pass through — the reviewer prompt demands the
    # same-entity-numbered-slots signature before it may rewrite.
    if _re.search(r"\bUNION\b", chosen.get("sql") or "", _re.IGNORECASE):
        return "union-tall"
    # defensive guard (D1): unrequested IS NOT NULL / != '' / > 0 filters (Q00037/Q00050
    # anatomy). Gate admits ALL guard carriers — the legit NULLs-in-ASC exception is
    # adjudicated by the prompt (rule 2 + _REVIEWER_COMMON), not excluded here; v1's
    # guard-removal record across full exposure was zero-misfire.
    if _re.search(r"IS\s+NOT\s+NULL|<>\s*''|!=\s*''|>\s*0(?!\.)\b",
                  chosen.get("sql") or "", _re.IGNORECASE):
        return "guard-suspect"
    # divide-first percentage (X / Y * 100): the arrangement-reorder rule's territory —
    # scalar percentage outputs (1x1 result) carry no shape signal, so without this the
    # rule can never reach them (coverage gap found 07-24; net new exposure ~2/498).
    if _re.search(r"/\s*(?:NULLIF\s*\()?[A-Za-z_\"'(](?:[^*;+-]|\(\s*\*\s*\)){0,120}?\*\s*100(\.0)?\b",
                  chosen.get("sql") or "", _re.IGNORECASE):
        return "arrangement-suspect"
    # decoded-label projection (E4, 07-30): a CASE branch or IIF emitting string-literal
    # labels — the value-format misselect family's syntactic trigger (largest cluster,
    # 19/131 in diag_20260724; 10 of its cases reached review through no other signal).
    # Full-run base rates (e2_run 2x1534): fires 120/3068 final SQLs (3.9%), only 9/2230
    # correct answers (0.40%); marginal un-gated exposure 27 rows. The gate is deliberately
    # coarse — rule 4's three-part signature does the narrowing. Preregistered audit: on
    # the next full eval, decoded-label edits must fix:break >= 2:1 (named at-risk correct
    # rows Q00091/Q00473 must survive) or the rule AND this gate revert together.
    if (_re.search(r"\bTHEN\s*'[^']", chosen.get("sql") or "", _re.IGNORECASE)
            or _re.search(r"\bIIF\s*\([^;]{0,200}?'[^']*'\s*,\s*'[^']*'\s*\)",
                          chosen.get("sql") or "", _re.IGNORECASE)):
        return "decoded-label"
    if (chosen.get("judge_debug") or {}).get("mode") == "truncated":
        return "judge-truncated"
    if _rank_request(question):
        return "rank-request"
    return ""


def _candidate_projection(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Project THIS run's candidate pool into a judge result's ``candidates`` and ``candidates_debug``.

    The summary is what the judge result keeps of each candidate; the debug records carry
    the candidate's full telemetry, stored in the result JSON only and never shown to the
    judge. A candidate's own replay flag and render identity travel with it (the
    projection used to drop both, so a replayed candidate looked freshly generated under
    the current release). A fresh judge and a judge cache hit share this projection: the
    hit replays the judge's decision, but the pool it describes is the one this run made,
    whose identities carry the divergence causes the outcome reads.

    Args:
        candidates (list[dict]): The pool as `_gather_in_session` returned it.

    Returns:
        tuple[list[dict], list[dict]]: ``(candidates, candidates_debug)`` in pool order.
    """
    summary = [
        {"sql": c.get("sql"), "row_count": c.get("row_count", len(c.get("rows", []))),
         "error": c.get("error"),
         **({"provider_failure": True} if c.get("provider_failure") else {}),
         **({"saved_candidate_source": c["saved_candidate_source"]} if "saved_candidate_source" in c else {})}
        for c in candidates
    ]
    debug = [
        {"sql": c.get("sql"), "tool_calls": c.get("tool_calls"), "messages": c.get("messages"),
         "sql_actions": c.get("sql_actions"), "protocol_events": c.get("protocol_events"),
         "usage": c.get("usage"), "duration_sec": c.get("duration_sec"),
         "timing": c.get("timing"),
         "reasoning": c.get("reasoning"), "error": c.get("error"),
         **{key: c[key] for key in ("_exc_type", "_dead_session", "_mcp_error_code", "provider_failure") if key in c},
         "transcript_status": c.get("transcript_status"),
         "capture_reason": c.get("capture_reason"), "last_checkpoint_reason": c.get("last_checkpoint_reason"),
         "recording_error": c.get("recording_error"), "cache_source_error": c.get("cache_source_error"),
         **({"rank_contract": c["rank_contract"]} if "rank_contract" in c else {}),
         "attempts": c.get("attempts", []),
         "route_debug": c.get("route_debug"),
         "from_cache": bool(c.get("from_cache")),
         "cache_source": c.get("cache_source"), "execution": c.get("execution"),
         **({"cache_key": c["cache_key"]} if "cache_key" in c else {}),
         **({"memo_base_key": c["memo_base_key"]} if "memo_base_key" in c else {}),
         **({"implementation_context": c["implementation_context"]} if "implementation_context" in c else {}),
         **({"consumer_implementation_context": c["consumer_implementation_context"]} if "consumer_implementation_context" in c else {}),
         **({"saved_candidate_source": c["saved_candidate_source"]} if "saved_candidate_source" in c else {}),
         "prompt_identity": c.get("prompt_identity")}
        for c in candidates
    ]
    return summary, debug


def _replayed(result: dict, *, role: str = "judge") -> dict:
    """Return one cached vote result with its cost counters cleared.

    A cache hit makes no API call, so the stored token counts and wall times belong to
    the run that first paid for them. Replaying them would bill this run for resources
    it never spent, inverting the arm-vs-arm cost comparison those counters exist to
    support. Every agent record in the result — the judge's and each candidate's — is
    marked replayed and left unmeasured instead. The judge cache-hit path then replaces
    ``candidates`` and ``candidates_debug`` with this run's own pool
    (`_candidate_projection`), so the cleared candidate records only reach a caller
    that keeps the stored pool.

    Args:
        result (dict): Cached vote result, not mutated.

    Returns:
        dict: Shallow copy whose ``judge_debug`` and ``candidates_debug`` entries carry
        empty usage, no duration, ``from_cache``, and a ``prompt_identity`` (the stored
        snapshot, or the legacy-cache marker when the entry predates it).
    """
    def _clear(record: object, *, capture: bool = False, seat: str = role, slot: int | None = None) -> object:
        if not isinstance(record, dict):
            return record
        # An agent record written before the render recorder existed carries no identity;
        # it is marked as legacy rather than credited to the release replaying it.
        identity = record.get("prompt_identity") or dict(_LEGACY_CACHE_IDENTITY)
        prepared = {**record, "prompt_identity": identity,
                    **({"attempts": [_clear(attempt, seat=seat, slot=slot) for attempt in record["attempts"]]}
                       if record.get("attempts") else {})}
        return cache_lineage.replay(prepared, role=seat, slot=slot, capture=capture)

    replayed = dict(result)
    if "judge_debug" in replayed:
        replayed["judge_debug"] = _clear(replayed["judge_debug"], capture=True)
    if isinstance(replayed.get("candidates_debug"), list):
        replayed["candidates_debug"] = [_clear(d, seat="candidate", slot=index + 1) for index, d in enumerate(replayed["candidates_debug"])]
    return replayed


def _tool_result_error(result: object) -> object:
    """Return the first error a judge tool result carries, or None.

    Dict results (run_sql, submit_final_sql, …) hold ``error``
    directly — the shape the old inline extraction covered. List results (get_rules,
    get_examples, query_knowledge) carry error ROWS, which that extraction silently
    dropped: a degraded ``get_rules`` answers with exactly one such row
    (``knowledge.RULES_RETRIEVAL_FAILED_ERROR``), and since the judge's transcript is
    never stored, ``judge_debug.tool_calls[].error`` is the only place that failure can
    reach the finished row for eval_bird's ``degraded_retrieval`` scan.

    Args:
        result (object): Whatever the tool returned — dict, list of rows, or scalar.

    Returns:
        object: The error value (usually str), or None when the result carries none.
    """
    if isinstance(result, dict):
        return result.get("error")
    if isinstance(result, list):
        return next((row.get("error") for row in result
                     if isinstance(row, dict) and row.get("error")), None)
    return None


#: Knob variables that change what the judge's and the reviewer's tools return, so an override enters their shared
#: execution identity, written only when set (the default keys keep their bytes; the release's own knob values move the
#: ``prompt`` term instead). The glyph floor decides which values search_values returns; the row cap limits SQL
#: tool results even when the initial preview is unchanged. The synonym floor needs no entry: it reaches the judge
#: through the synonym block of its user message, which the key hashes.
_JUDGE_KEY_ENV = ("RUBIKSQL_SYMBOL_NEAR", "RUBIKSQL_RUN_SQL_MAX_ROWS")


def _judge_execution_identity(
    active_tools: frozenset[str],
    *,
    judge_provider: str = "",
    arm_tag: str = "",
    route_tag: str = "",
    prompt: str = "",
    ns: str = "",
    implementation: str,
) -> str:
    """Return the shared judge/reviewer cache identity for runtime semantics.

    ``judge_style`` and ``judge_gate`` keep the values the default judge route wrote: the judge has
    no route since its grain route was deleted (2026-09-25), and the two terms retain those default
    values. The variables of :data:`_JUDGE_KEY_ENV` are read here, each written
    under its lowercased name only when set.

    Args:
        active_tools (frozenset[str]): Capabilities the served registry exposes to this run.
        judge_provider (str): Provider name of the judge model.
        arm_tag (str): ``RUBIKSQL_CACHE_ARM_TAG`` namespace, empty when unset.
        route_tag (str): Candidate-route signature of a mixed pool, empty for a plain pool.
        prompt (str): ``_PROMPTS.key_extra(role, None)``: the keyspace of this role's prompt
            closure when it differs from r0's, else empty. Written only when non-empty, so r0 and
            every refactor-only release keep today's bytes.
        ns (str): ``RUBIKSQL_CACHE_NS``, the sampling universe; written only when set.
        implementation (str): This role's declared implementation digest, retained
            in both whole and memo identity.

    Returns:
        str: Canonical JSON (sorted keys, compact separators) hashed into the judge and reviewer keys.
    """
    identity = {
        "model_protocol": MODEL_PROTOCOL,
        "transport_protocol": selected_transport_protocol(base_protocol=TRANSPORT_PROTOCOL),
        "judge_provider": judge_provider,
        "judge_style": None,
        "capabilities": sorted(active_tools),
        "native_tools": sorted(active_tools - SQL_ACTIONS),
        "sql_actions": sorted(active_tools & SQL_ACTIONS),
        "arm": arm_tag,
        "candidate_routes": route_tag,
        "judge_gate": "",
        "implementation": implementation,
    }
    if prompt:
        identity["prompt"] = prompt
    if ns:
        identity["ns"] = ns
    if _PROMPTS.has("shared.hint_conflict"):
        # The frame is appended after the key's user message, so the detector's version is keyed here; a release
        # without the row keeps today's bytes.
        identity["hint_conflicts"] = hint_conflicts.IMPLEMENTATION_VERSION
    for variable in _JUDGE_KEY_ENV:
        value = os.environ.get(variable)
        if value is not None:
            identity[variable.lower()] = value
    return json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


async def _judge_runtime_tools(
    session: ClientSession,
    active_tools: frozenset[str],
    *,
    role: str = "judge",
) -> tuple[list[object], frozenset[str]]:
    """Read the served tool registry and return its intersection with the judge's tool set."""
    tools = list((await session.list_tools()).tools)
    _register_tool_props(tools)
    tool_contract.assert_live_contract(
        tools, role,
        capabilities=active_tools,
    )
    effective = frozenset(tool.name for tool in tools if tool.name in active_tools)
    return tools, effective


async def _review_final(chosen, question, db_id, session, judge_model, verbose,
                        judge_provider, reviewer_tools, *,
                        arm_tag: str = "", route_tag: str = "",
                        knowledge_fingerprint: str = "", pass_tag: str = ""):
    """Reviewer pass over the judge's final answer (keeps the original on ordinary failures).

    The cache-hit path must not touch the session or the LLM: a cached verdict has to
    survive a dead MCP session, and re-rolling it would flip borderline verdicts between
    otherwise-identical arms. The capability term comes from the reviewer's pinned native
    selection. The registry is read inside _run_judge only on a cache miss; the judge's
    recorded capabilities cannot grant or suppress a reviewer-only callable. ``pass_tag``
    (``|pass=<index>``, empty for pass 0) keeps a later judge pass's reviewer from replaying
    another pass's review of the same SQL."""
    effective_capabilities = frozenset(reviewer_tools)
    contract = tool_contract.role_contract("reviewer", capabilities=effective_capabilities)
    execution_identity = _judge_execution_identity(
        effective_capabilities,
        judge_provider=judge_provider,
        arm_tag=arm_tag,
        route_tag=route_tag,
        prompt=_PROMPTS.key_extra("reviewer", None),
        ns=os.environ.get("RUBIKSQL_CACHE_NS", ""),
        implementation=contract["digest"],
    )
    rows = chosen.get("rows", [])
    preview, _trunc = _table_display(
        rows,
        _JUDGE_MAX_ROWS,
        columns=chosen.get("columns"),
    )
    # The reviewer is its own agent run with its own render identity: the user frame and
    # the system message render inside its recorder here, and _run_judge_inner rebinds the
    # same recorder for the loop. The preview and the fenced SQL stay Python data formatting.
    recorder = _recorder_for("reviewer", None)
    with recorder:
        user_msg = _PROMPTS.surface(
            "reviewer", "user",
            question=question,
            sql=fenced_sql(chosen.get("sql") or ""),
            row_count=int(chosen.get("row_count", len(rows))),
            preview=preview,
        )
        # The system surface is rendered here rather than read from _REVIEWER_SYSTEM so the
        # run's fired set names its row; the memoised render is the same object, same bytes.
        system = _PROMPTS.surface("reviewer", "system")
    sh = hashlib.sha256(system.encode()).hexdigest()[:16]
    ck = hashlib.sha256(
        f"reviewer|{judge_model}|{_REVIEWER_SEED}|{sh}|{user_msg}|"
        f"{execution_identity}|kb={knowledge_fingerprint}{pass_tag}".encode()
    ).hexdigest()
    # Turn memo base key: the same key with the prompt term left out (turn_memo.py), so a
    # reviewer under a changed release replays the turns of the parent's reviewer.
    base_identity = _judge_execution_identity(
        effective_capabilities,
        judge_provider=judge_provider,
        arm_tag=arm_tag,
        route_tag=route_tag,
        ns=os.environ.get("RUBIKSQL_CACHE_NS", ""),
        implementation=contract["digest"],
    )
    turn_base_key = hashlib.sha256(
        f"reviewer|{judge_model}|{_REVIEWER_SEED}|{sh}|{user_msg}|"
        f"{base_identity}|kb={knowledge_fingerprint}{pass_tag}".encode()
    ).hexdigest()
    cand = [{"sql": chosen.get("sql") or "", "rows": rows,
             "row_count": chosen.get("row_count", len(rows))}]
    # _run_judge only ever WROTE this key; without a read here every run re-rolled the
    # reviewer LLM call, so borderline verdicts flipped between otherwise-identical arms.
    reviewed = _cache_get(ck)
    fresh = reviewed is None
    if fresh:
        reviewed = await _run_judge(
            cand, user_msg, ck, db_id, session, judge_model, _REVIEWER_SEED, verbose,
            any_truncated=True, judge_provider=judge_provider, question=question,
            system=system, active_tools=effective_capabilities,
            recorder=recorder,
            turn_base_key=turn_base_key, turn_role="reviewer",
            turn_lookup=_turn_lookup_allowed("reviewer", None))
    else:
        reviewed = _replayed(reviewed, role="reviewer")
        reviewed["judge_debug"] = {
            **(reviewed.get("judge_debug") or {}),
            "consumer_implementation_context": {"context": contract["context"], "digest": contract["digest"]},
        }
    changed = (" ".join((reviewed.get("sql") or "").split())
               != " ".join((chosen.get("sql") or "").split()))
    reviewed = dict(reviewed)
    # The reviewer ran through _run_judge_inner, so its snapshot sits on the judge-shaped
    # record it returned (and, on a cache hit, on the stored one). A stored record without
    # a snapshot predates the recorder and is marked legacy, never credited to this release.
    identity = (reviewed.get("judge_debug") or {}).get("prompt_identity")
    if identity is None:
        identity = recorder.snapshot() if fresh else dict(_LEGACY_CACHE_IDENTITY)
    reviewed["reviewer_debug"] = {
        "active": True, "changed": changed,
        "original_sql": chosen.get("sql") or "",
        "system_hash": sh, "v": 2,
        "prompt_identity": identity,
        "route_debug": reviewed.get("judge_debug"),
    }
    if not changed:
        # keep the judge's richer execution record when nothing changed
        merged = dict(chosen)
        merged["reviewer_debug"] = reviewed["reviewer_debug"]
        return merged
    # record hygiene: the reviewer runs through _run_judge, so every judge-shaped key on
    # `reviewed` describes the REVIEWER — a different agent, prompted with one already
    # chosen SQL and its own system prompt. Returning those as the judge's would file the
    # reviewer's mode, tool calls, tokens and transcript under the judge, and since the
    # reviewer always runs with any_truncated=True it would report every rewritten
    # question as a truncated judge. Keep the judge's own record; move the reviewer's
    # into reviewer_debug, where its own name is on it.
    if "judge_debug" in chosen:
        reviewed["judge_debug"] = chosen["judge_debug"]
    else:
        reviewed.pop("judge_debug", None)
    for key in ("candidates", "candidates_debug"):
        # _run_judge only saw the single chosen SQL, so its pool and transcripts would
        # otherwise overwrite the real ones.
        if key in chosen:
            reviewed[key] = chosen[key]
    return reviewed

# B1 nudge — appended as a user turn when the judge ends without calling
# submit_final_sql. Only its submitted rows are scored, so a JSON/text-only
# decision forces a blind best-candidate fallback; this asks it to actually submit.
_JUDGE_SUBMIT_NUDGE = _PROMPTS.surface("judge", "nudge.submit")


#: Fraction of the candidate pool that must survive before the judge runs. Below it, the
#: failed seeds are re-run once (see nl2sql_vote). 0.6 = 2-of-3 on the shipped pool of three
#: (run_official_pipeline.DEFAULT_CANDIDATES), 3-of-5 on the five-candidate runs measured here.
#: Measured on the full 1534-question dev run with five candidates: 5 ok 97.7% / 4 ok 1.5% /
#: 3 ok 0.3% / 2-or-fewer 0.46%. So the gate fired on 7 questions and left the other 99.5%
#: untouched — the point is to protect the fragile tail, not to chase a full pool everywhere.
_CANDIDATE_QUORUM = float(os.environ.get("RUBIKSQL_CANDIDATE_QUORUM", "0.6") or "0.6")
#: A quorum top-up that would start fewer than this many seconds before the candidate cap
#: (`phase_budget.VotePhases.candidate_deadline`) is skipped: in C (G28) every top-up that started
#: that late was cancelled in flight at the deadline (9 questions, all wrong).
_TOPUP_MIN_LEFT_SEC = float(os.environ.get("RUBIKSQL_TOPUP_MIN_LEFT_SEC", "300") or "0")
# G31 retired the early straggler cut (cut stragglers at the vote budget minus a judge reserve once
# 4 of 6 had finished): the judge phase has its own budget, so candidates are cut only at theirs.
for _retired in phase_budget.RETIRED_ENV:
    if os.environ.get(_retired):
        print(f"[LOG ONLY] [phase-budget] {_retired} is retired and ignored: candidates are cut only at "
              f"{phase_budget.CANDIDATE_BUDGET_ENV}, and the judge has {phase_budget.JUDGE_BUDGET_ENV}",
              file=sys.stderr, flush=True)


_SQL_STRING_LITERAL = _re.compile(r"'(?:[^']|'')*'")


def _novel_literal_rewrite(final_sql: str, candidate_sqls: list[str]) -> list[str]:
    """String literals the final SQL retypes where it otherwise equals a candidate.

    Q00516 (8/5 census): every candidate wrote 'Volkan Baǵa'; the judge submitted the
    byte-identical SQL with 'Volkan Başa' and scored zero against a 100%-correct pool.
    Retyping corrupts exactly the unusual characters a judge cannot spell from memory.
    The guard fires only when the final SQL matches some candidate byte-for-byte once
    string literals are masked — a genuinely rewritten query is never touched.

    Args:
        final_sql: The SQL the judge is about to finalize.
        candidate_sqls: The candidate pool's SQLs.

    Returns:
        Literals present in the final SQL but in no candidate, or ``[]`` when the
        guard does not apply.
    """
    def skeleton(sql: str) -> str:
        masked = _SQL_STRING_LITERAL.sub("'\x00'", sql or "")
        return _re.sub(r"\s+", " ", masked).strip().lower()

    fin_skeleton = skeleton(final_sql)
    if "'\x00'" not in fin_skeleton:
        return []
    pool = [c for c in candidate_sqls if c and c.strip()]
    if not any(skeleton(c) == fin_skeleton for c in pool):
        return []
    pool_literals = {lit for c in pool for lit in _SQL_STRING_LITERAL.findall(c)}
    return [lit for lit in _SQL_STRING_LITERAL.findall(final_sql or "")
            if lit not in pool_literals]


# The receipt the guard hands back is the judge.note.literal_guard surface (slot novel), rendered
# where the bounce fires in _run_judge_inner.


def _quorum(pool: int) -> int:
    """Smallest surviving-candidate count that still lets the judge run without a top-up.

    Args:
        pool (int): How many candidates were launched.

    Returns:
        int: ``ceil(pool * _CANDIDATE_QUORUM)``, floored at 1 — a single survivor is always
        enough to produce an answer, so the gate never demands more than the pool can give.
    """
    return max(1, math.ceil(pool * _CANDIDATE_QUORUM))


# ── judge tools ───────────────────────────────────────────────────────────────
# Same tool set for both modes; judge decides whether to use them.

# query_knowledge is UNCONDITIONAL (promoted 07-26, no arm): the exact-filter/enumerate path
# into the knowledge base, the judge's only non-top-k channel (candidate side in nl2sql.py,
# tool registration in server.py — all three must stay in lockstep).
#
# expand_examples is UNCONDITIONAL (adopted 07-26, no arm): get_examples hands the judge a
# menu of id + question + sql_length with the SQL withheld; each id is a session-local E-id.
# The judge —
# unlike a candidate — had no way to expand a menu entry into its actual SQL. It was comparing
# candidate SQL against question text alone while the strongest available evidence sat one call
# away. That is a capability gap, not a hypothesis worth an arm.
#
# expand_synonyms is deliberately NOT given to the judge. The production knowledge state carries
# synonym=0 (verified across all 11 workspaces) so the tool is a no-op there, and the S1 arm that
# did populate synonyms measured +1.21pp alone but -1.00pp on top of the two-stage examples now
# in production — same gain path (user wording <-> schema entity), already spent. Handing the
# judge a proven no-op would cost a prompt line and a tool slot for nothing.

_JUDGE_TOOLS = tool_contract._selection("judge", "default", None, None)
_REVIEWER_TOOLS = tool_contract._selection("reviewer", "default", None, None)


# ── judge prompts ─────────────────────────────────────────────────────────────

# ── Shared judge rules reused by the reviewer ──────────────────────────────────
# The reviewer consumes its whole current role document.
_REVIEWER_SYSTEM = _PROMPTS.surface("reviewer", "system")


# ── Judge prompt (2026-08-06 consolidation) ─────────────────────────────────
# 08-06 用户裁决: complete/truncated 双头合一(免闸门直接生产)。截断态由渲染层的显式
# "truncation" JSON 对象自证,条件句教判官认标记,比按模式二选一的宣告更可操作;
# any_truncated 保留为遥测(judge_debug.mode / 日志),不再选提示词。
_JUDGE_SYSTEM = _PROMPTS.surface("judge", "system")

# 阶段二归并: 原 42 段/30.7K 中同主题散布 6-7 处(DISTINCT×6、LIMIT×7、原生值×6),按
# 决策类型归并为单一处;信息无损(多agent审计门),关键实测措辞逐句保留(J1 +1.55pp 主句、
# hint-verbatim、知识优先级、投影倾斜、E1 弹条、就地取列 99.6% 统计)。两版差异收敛为
# 头部两行;正文单一字面量。旧两代知识优先级矛盾(examples>rules vs rules>examples)按
# 08-06 误选解剖的实证定序: hint映射 > 库规则 > 示例,并保留"规则永不压题面"句。
# The current judge body is its complete role document.
_JUDGE_BODY = _JUDGE_SYSTEM

_JUDGE_COMMON = _JUDGE_BODY   # 身份指纹与钉测沿用此名


# ── judge implementation ──────────────────────────────────────────────────────


def _parse_json(content: str) -> dict:
    text = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()
    for pattern in [r"```(?:json)?\s*(\{.*?\})\s*```", r"\{.*\}"]:
        m = _re.search(pattern, text, _re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1) if "```" in pattern else m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _extract_judge_sql(content: str, candidates: list[dict]) -> tuple[str, str]:
    """B2: recover the judge's chosen SQL when it answered in text instead of calling
    submit_final_sql. Tries, in order of reliability:
      1. JSON {"sql": ...}                     — the documented decision format
      2. a ```sql … ``` fenced block            — judge wrote/corrected SQL inline
      3. "accept/choose/use candidate N"        — judge pointed at a candidate (the
         main lazy form: 'all candidates identical, accept candidate 2'); 1-indexed,
         maps to candidates[N-1] (matching the 'Candidate N' numbering in user_msg)
    Returns (sql, source). Empty sql ('', '') means nothing extractable → Case 3.
    Explicit SQL wins over a candidate pointer: if the judge wrote SQL, that exact
    SQL is its intent; the pointer is only used when no SQL text is present."""
    # 1. JSON sql
    parsed = _parse_json(content)
    if parsed.get("sql"):
        return parsed["sql"], "json"
    # 2. exactly one fenced SQL block — variable-length backtick/tilde fences,
    # matching exactly. Multiple blocks are an ambiguous comparison, not a final
    # answer; in that case only an explicit candidate pointer below may recover.
    fenced: list[tuple[int, str]] = []
    for pattern in (
        (
            r"^(?P<fence>`{3,}|~{3,})sql[ \t]*\r?\n"
            r"(?P<body>.*?)(?:\r?\n)(?P=fence)[ \t]*$"
        ),
        (
            r"^(?P<fence>`{3,}|~{3,})[ \t]*\r?\n"
            r"(?P<body>[ \t]*(?:SELECT|WITH)\b.*?)(?:\r?\n)"
            r"(?P=fence)[ \t]*$"
        ),
    ):
        fenced.extend(
            (match.start(), match.group("body").strip())
            for match in _re.finditer(
                pattern,
                content,
                _re.I | _re.MULTILINE | _re.DOTALL,
            )
            if match.group("body").strip()
        )
    if len(fenced) == 1:
        return fenced[0][1], "fenced"
    # 3. explicit acceptance of a numbered candidate
    m = _re.search(
        r"(?:accept|choose|select|pick|use|go with|best(?:\s+(?:one|candidate))?\s+is|"
        r"correct(?:\s+(?:one|candidate))?\s+is)\D{0,20}?candidate\s*#?\s*(\d+)",
        content, _re.I)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(candidates):
            c = candidates[idx]
            if not c.get("error") and c.get("sql"):
                return c["sql"], f"candidate_{idx + 1}"
    return "", ""


#: Concurrent judge pipelines per question; 1 is the single-pass vote (`judge_majority`).
_JUDGE_PASSES = _llm_runtime_module.JUDGE_PASSES


def _make_judge_gate(raw: str | None, passes: int = 1) -> threading.BoundedSemaphore | None:
    """Build the optional per-process judge/reviewer session cap.

    Unset or empty disables the gate. Explicit values retain the original integer
    parsing: positive integers set the cap, zero/negative integers clamp to one,
    and malformed non-empty values raise ValueError during module initialization.
    The cap is per judge pass: N concurrent passes admit N times as many sessions.
    A pass holds a permit only for its own judge or reviewer session and never waits
    on a sibling pass while holding it, so the scaled cap cannot deadlock.
    """
    return None if raw in (None, "") else threading.BoundedSemaphore(max(1, int(raw)) * passes)


#: The retired OpenRouter account's eight-session provider throttle is opt-in now.
#: Question workers bound evaluation fan-out; judge/reviewer sessions own their
#: state and do not require this gate for HB/DB handle or SQL execution safety.
_JUDGE_GATE = _make_judge_gate(os.environ.get("RUBIKSQL_JUDGE_MAX_CONCURRENCY"), _JUDGE_PASSES)

#: Optional per-process request pacing on top of the concurrency gate. Unset,
#: empty or 0 RUBIKSQL_JUDGE_RPM disables pacing; positive values charge each turn.
#: Historical 08-27 rationale on the retired OpenRouter account: the gate bounds how many judge
#: sessions run at once; it does not bound how many the lane admits per minute, and
#: the provider's then-new-account tier throttled exactly that (run 9: 46 429s, 21
#: questions burned their whole 30-minute budgets waiting out a saturated lane and
#: shipped single-candidate fallbacks - 9 of the 14 same-question losses against the
#: DashScope baseline, and the partial sweep alone bought 7 of them back). One
#: session made 1-3 provider calls, so admissions were paced a notch under the
#: measured request ceiling. Probed 08-27 on the live OpenRouter account (the
#: DashScope quota is unmeasured): 45 req/min drew 60% 429s, 30/min sat on the line
#: (5%), 20/min was near-clean and 12/min spotless - that account tier was
#: ~30 requests/min. 12 SESSIONS/min was chosen to keep the request rate
#: in the clean band at the typical 1-2 calls per session and admit a full
#: 1534-question run's judges in ~2 hours. This retired quota no longer sets the default.
def _parse_rpm(raw: str | None) -> float:
    """Parse the per-process RPM knob: unset, empty or 0 disables judge pacing.

    Malformed explicit values retain the conservative 12 RPM fallback. An invalid
    limit must not silently become unlimited merely because the unset default is
    off. Positive finite values are capped at 600; nan, infinities and negatives
    use the fallback (inf once silently disabled pacing and nan crashed the
    empty-window read).
    """
    try:
        value = float(raw if raw not in (None, "") else "0")
    except (TypeError, ValueError):
        return 12.0
    if value != value or value in (float("inf"), float("-inf")) or value < 0:
        return 12.0
    return min(value, 600.0)


#: RUBIKSQL_JUDGE_RPM is per judge pass: N concurrent passes share one window N times as wide.
_JUDGE_RPM = _parse_rpm(os.environ.get("RUBIKSQL_JUDGE_RPM")) * _JUDGE_PASSES
#: Window length; module-level so tests can shrink it and exercise real waits.
_PACE_WINDOW_S = 60.0
_PACE_LOCK = threading.Lock()
_PACE_COND = threading.Condition(_PACE_LOCK)
_PACE_ADMISSIONS: "deque[float]" = deque()
_PACE_TICKET_NEXT = 0
_PACE_TICKET_SERVING = 0


def _pace_slot(charge: bool) -> None:
    """Block FIFO until the sliding window has room; append only when ``charge``.

    08-27 audit closures, both: the window counts REQUESTS, not sessions (charge
    happens per judge turn via achat_turn's hook; admission merely waits for
    headroom), and waiters are served strictly in arrival order - the old
    poll-under-a-plain-lock scheme had no fairness contract, so under sustained
    arrival a waiter could starve behind newcomers indefinitely. A ticket per
    waiter and a condition variable make overtaking impossible.
    """
    global _PACE_TICKET_NEXT, _PACE_TICKET_SERVING
    if _JUDGE_RPM <= 0:
        return
    with _PACE_COND:
        my_ticket = _PACE_TICKET_NEXT
        _PACE_TICKET_NEXT += 1
        while True:
            now = time.monotonic()
            while _PACE_ADMISSIONS and now - _PACE_ADMISSIONS[0] > _PACE_WINDOW_S:
                _PACE_ADMISSIONS.popleft()
            if _PACE_TICKET_SERVING == my_ticket and len(_PACE_ADMISSIONS) < _JUDGE_RPM:
                if charge:
                    _PACE_ADMISSIONS.append(now)
                _PACE_TICKET_SERVING += 1
                _PACE_COND.notify_all()
                return
            if _PACE_TICKET_SERVING == my_ticket:
                wait = _PACE_WINDOW_S - (now - _PACE_ADMISSIONS[0]) + 0.05
                _PACE_COND.wait(max(0.05, min(wait, 5.0)))
            else:
                _PACE_COND.wait()


def _pace_judge_admission() -> None:
    """Session admission: wait for lane headroom without spending the window."""
    _pace_slot(charge=False)


def _pace_judge_turn_charge() -> None:
    """One judge REQUEST pays the lane window (installed as achat_turn's pacer hook)."""
    _pace_slot(charge=True)


_llm_runtime_module._JUDGE_TURN_PACER = _pace_judge_turn_charge


async def _acquire_gate(gate: threading.BoundedSemaphore) -> None:
    """Acquire an explicit session gate WITHOUT blocking the event loop.

    Every judge of one database runs on that database's single shared session loop
    (nl2sql._SharedSession) - a thread-blocking acquire executed on that loop would
    freeze the loop itself, so permit holders could never resume to release and
    the next waiter would deadlock the whole database (audited 08-27,
    three independent reviews converged on vote.py's old blocking acquire). The
    acquire therefore runs on a dedicated thread; if this task is cancelled while
    waiting, the orphaned acquire hands its permit straight back.
    """
    loop = asyncio.get_running_loop()
    got: "asyncio.Future[bool]" = loop.create_future()

    def _grab() -> None:
        gate.acquire()

        def _deliver() -> None:
            if got.cancelled():
                gate.release()      # 等待者已被取消: 坑原路奉还
            else:
                got.set_result(True)

        try:
            loop.call_soon_threadsafe(_deliver)
        except RuntimeError:               # 循环已关闭: 无人认领, 直接还坑
            gate.release()

    threading.Thread(target=_grab, daemon=True, name="judge-gate-wait").start()
    try:
        await got
    except asyncio.CancelledError:
        # Delivery can win just before cancellation resumes this task. In that
        # case _deliver already handed ownership here, but _run_judge has not
        # entered its release-finally yet.
        if got.done() and not got.cancelled():
            gate.release()
        raise


def _judge_record(evidence: dict, *, interrupted: bool, persist: bool = True) -> dict:
    """Snapshot this attempt independently of whether its answer can be cached."""
    record = dict(evidence.get("debug", {}))
    messages = deepcopy(evidence.get("messages", []))
    memo = evidence.get("memo")
    reasoning = memo.thinks() if memo is not None else []
    for message, think in zip((m for m in messages if m.get("role") == "assistant"), reasoning):
        if think and not message.get("reasoning_content"):
            message["reasoning_content"] = think
    record.update(messages=messages, reasoning=reasoning,
                  transcript_status="interrupted" if interrupted else "complete")
    recorder = evidence.get("recorder")
    if recorder is not None:
        if memo is not None:
            recorder.set_replay(**memo.replay_fields(evidence["turn"]))
        record["prompt_identity"] = recorder.snapshot()
    if interrupted:
        record["mode"] = "interrupted"
        if "clock" in evidence:
            record["duration_sec"] = round(evidence["clock"].real_elapsed(), 3)
    record["execution"] = evidence.get("execution", {})
    collector = evidence.get("timing")
    timing = collector.snapshot() if collector is not None else None
    if timing is not None:
        record["timing"] = timing
    if not persist:
        return record
    # The audit transcript annotates reasoning; the memo must replay the exact bytes
    # originally sent to the model, before those audit-only annotations.
    source = cache_lineage.retain_source(_TURN_STORE, record, payload={**record, "memo_messages": deepcopy(evidence.get("messages", []))})
    try:
        if memo is not None and memo.write_episode(
            evidence["cache_key"], evidence["messages"], evidence["release"],
            **({"source": source} if source is not None else {}),
        ):
            record["episode_ref"] = {
                "key": evidence["cache_key"], "namespace": evidence["namespace"],
                "release": evidence["release"], "messages_sha256": sha256_hex(canon(messages)),
            }
    except Exception as error:
        record["recording_error"] = str(error)
    return record


async def _run_judge(*args, **kwargs):
    """Apply optional session/request limits to every judge and reviewer.

    Both waits below - the gate and the admission pacer - are thread-blocking by
    nature and MUST NOT run on the shared per-database loop (see `_acquire_gate`);
    each waits off-loop so the loop keeps serving the permit
    holders' awaits while this session queues.
    """
    with session_timing.scope(kwargs.get("turn_role", "judge")) as timing:
        gate = _JUDGE_GATE
        acquired = False
        try:
            if gate is not None:
                with session_timing.measure("gate", "judge_concurrency"):
                    await _acquire_gate(gate)
                acquired = True
            with session_timing.measure("gate", "judge_admission"):
                await _llm_runtime_module.pace_judge(_pace_judge_admission)
            result = await _run_judge_inner(*args, **kwargs)
            if isinstance(result, dict) and "judge_debug" in result:
                result["judge_debug"]["timing"] = timing.snapshot()
            return result
        except (Exception, asyncio.CancelledError) as exc:
            exc.judge_debug = {**getattr(exc, "judge_debug", {}), "timing": timing.snapshot()}
            raise
        finally:
            if acquired:
                gate.release()


async def _run_judge_inner(*args, **kwargs):
    """Retain one admitted attempt, including a failure before its first reply."""
    # AttemptCapture can invoke its observer with the waiting thread's ContextVars.
    # Keep the owner's collector explicitly instead of resolving it in the callback.
    evidence: dict = {"timing": session_timing.current()}
    try:
        with tool_contract.dispatch_context(kwargs.get("turn_role", "judge"), capabilities=kwargs.get("active_tools")):
            result = await _run_judge_loop(*args, **kwargs, evidence=evidence)
    except (Exception, asyncio.CancelledError) as exc:
        # Keep the original exception type for the existing dead-session retry.
        exc.judge_debug = {
            **evidence.get("debug", {}), "messages": deepcopy(evidence.get("messages", [])),
            "execution": evidence.get("execution", {}), "transcript_status": "interrupted",
            "error": str(exc) or type(exc).__name__,
            "capture_reason": "async_cancel" if isinstance(exc, asyncio.CancelledError) else "agent_error",
        }
        try:
            exc.judge_debug.update(_judge_record(evidence, interrupted=True))
        except Exception as record_error:
            exc.judge_debug["recording_error"] = str(record_error)
        if exc.judge_debug.get("recording_error"):
            exc.add_note(f"Judge transcript metadata failed: {exc.judge_debug['recording_error']}")
        cache_lineage.finish(evidence.get("observer"), exc.judge_debug)
        raise
    evidence["debug"] = {**result["judge_debug"], "sql": result.get("sql")}
    try:
        result["judge_debug"].update(_judge_record(evidence, interrupted=False))
    finally:
        cache_lineage.finish(evidence.get("observer"), result["judge_debug"])
    if evidence.get("cache_result"):
        _cache_set(evidence["cache_key"], result)
    return result


async def _run_judge_loop(
    candidates:    list[dict],
    user_msg:      str,
    cache_key:     str,
    db_id:         str,
    session:       ClientSession,
    judge_model:   str,
    judge_seed:    int,
    verbose:       bool,
    any_truncated: bool,
    judge_provider: str = "qwen",
    question:      str = "",
    preinjected_titles: set[str] | None = None,
    preinjected_ids: set[str] | None = None,
    preinjected_session_id: str | None = None,
    system:        str | None = None,
    active_tools:  frozenset[str] | None = None,
    runtime_tools: list[object] | None = None,
    recorder:      RenderRecorder | None = None,
    turn_base_key: str | None = None,
    turn_lookup:   bool = False,
    turn_role:     str = "judge",
    evidence:      dict | None = None,
) -> dict:
    """Unified judge loop under ONE system prompt (_JUDGE_SYSTEM).

    The judge explores with run_sql, then finalizes via submit_final_sql — an unchanged
    candidate's SQL directly, or its own SQL after run_sql verification. The submit
    execution result is the scored answer (never the model's transcribed rows). Loop
    ends on successful submit; a bare 'stop' with JSON is a non-compliant fallback that
    re-executes the chosen SQL. ``any_truncated`` records whether any candidate preview
    was cut (a rendering fact the prompt teaches via the "truncation" JSON marker); it
    only labels telemetry (judge_debug.mode, logs) and no longer selects a prompt.
    ``recorder`` is the run's render recorder — the judge's, created by ``_vote_async``
    before the user message was rendered, or the reviewer's from ``_review_final``; its
    snapshot lands in ``route_debug["prompt_identity"]`` on every outcome. A direct caller
    that passes none gets a fresh judge recorder. ``turn_base_key`` is the run's cache key
    without its prompt term, under which the turn memo records this transcript (an episode,
    written next to the cache row) and, when ``turn_lookup`` is set, replays the recorded
    turns of a sibling run whose request bytes match (turn_memo.py); a direct caller that
    passes none records under ``cache_key`` and never replays. ``turn_role`` names the seat
    (``judge`` or ``reviewer``) for divergence attribution.
    """
    # The judge is its own agent run: fresh session_id (its own server-side session file for
    # any intermediates IT saves). Candidate final SQLs are already self-contained base-table
    # SQL, so the judge re-executes them with no intermediates. The session file is dropped by
    # the caller (_vote_async) after this returns. The render recorder is bound on the same
    # task context, so the rule-trigger frame below counts on it.
    evidence = {} if evidence is None else evidence
    evidence.update(cache_key=cache_key, namespace=os.environ.get("RUBIKSQL_CACHE_NS", ""),
                    release=_PROMPTS.id, debug={"model": judge_model, "provider": judge_provider})
    active_tools = tool_contract._selection(turn_role, "default", None, active_tools)
    contract = tool_contract.role_contract(turn_role, capabilities=active_tools)
    implementation_context = {"context": contract["context"], "digest": contract["digest"]}
    evidence["debug"]["implementation_context"] = implementation_context
    evidence["debug"]["memo_base_key"] = turn_base_key or cache_key
    evidence["execution"] = cache_lineage.identity(db=db_id, role=turn_role, slot=None,
                                                  release=_PROMPTS.id, knowledge_fingerprint=_db_knowledge_fingerprint(db_id))
    evidence["observer"] = cache_lineage.observe(evidence["execution"],
                                                lambda: _judge_record(evidence, interrupted=True, persist=False))
    _SESSION_ID.set(preinjected_session_id or uuid.uuid4().hex)
    _QUESTION_HINTS.set(question_hints(question))
    _ORIGINAL_QUESTION.set(question)
    if recorder is None:
        recorder = _recorder_for("judge", None)
    recorder.bind()
    evidence["recorder"] = recorder

    if runtime_tools is None:
        all_tools, effective_capabilities = await _judge_runtime_tools(
            session,
            active_tools,
            role=turn_role,
        )
    else:
        all_tools = list(runtime_tools)
        _register_tool_props(all_tools)
        tool_contract.assert_live_contract(
            all_tools, turn_role,
            capabilities=active_tools,
        )
        effective_capabilities = frozenset(
            tool.name for tool in all_tools if tool.name in active_tools
        )
    # sql_native_v2: SQL tools are model-visible native tools again, judge side included —
    # the 7/27-era judge ran SQL natively, and the text era measured a 15.2% turn
    # rejection rate on this seat.
    native_tool_names = effective_capabilities
    tools = [
        {"type": "function", "function": {
            "name": t.name,
            "description": t.description or "",
            "parameters": _strip_db_id(tool_input_schema(t)),
        }}
        for t in all_tools if t.name in native_tool_names
    ]

    if system is None:
        # Rendered through the release inside the bound recorder, so the fired set names the row.
        system = _PROMPTS.surface("judge", "system")
    rule_triggers = _RuleTriggerState(session, db_id)
    if preinjected_titles:
        # The baseline rules already sit in user_msg verbatim; mark them served so the
        # runtime hook and model-requested get_rules return only NEW rules.
        rule_triggers.triggered_titles.update(preinjected_titles)
    if preinjected_ids:
        # Same knowledge session now — id-level exclusion works too (belt over titles).
        rule_triggers.triggered_ids.update(preinjected_ids)
    with tool_contract.dispatch_context(turn_role, capabilities=active_tools):
        user_msg += await rule_triggers.match(question) + hint_conflict_frame(question)
    recorder.message("system", system)
    recorder.message("user", user_msg)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_msg},
    ]
    evidence["messages"] = messages
    evidence["debug"].update(user_msg=user_msg,
                             system_hash=hashlib.sha256(system.encode()).hexdigest()[:16])
    _log(
        verbose,
        "说明：带此前缀的行以及 ▶/◀ 箭头都是额外显示，真实模型上下文中没有；"
        "箭头后的正文或结构化字段才是模型实际收发的内容。",
    )
    _log(verbose, f"judge system ({len(system)} chars)")
    _trace(verbose, "▶", system)
    _log(verbose, f"judge user ({len(user_msg)} chars)")
    _trace(verbose, "▶", user_msg)

    last_submit_result = None
    last_submit_sql    = None
    last_content       = ""
    last_reasoning     = ""
    turn               = 0
    nudges             = 0   # B1: count of submit nudges issued
    literal_guard_warned = False   # retyped-literal bounce fires at most once per vote
    terminal_salvage_allowed = False
    judge_tool_calls: list[dict] = []
    sql_actions: list[dict] = []
    protocol_events: list[dict] = []
    judge_usage: dict = {}
    degraded_retrieval = rule_triggers.degraded_retrieval
    if degraded_retrieval:
        evidence["debug"]["degraded_retrieval"] = True
    # Logical clock and turn memo (turn_memo.py). The judge loop has no clock-driven
    # intervention, so the clock only stamps the recorded turns and feeds duration_sec.
    clock = RunClock(time.monotonic)
    memo = TurnMemo(
        _TURN_STORE,
        base_key=turn_base_key or cache_key,
        ns=evidence["namespace"],
        clock=clock,
        rebuild=_replayed_turn,
        lookup=turn_lookup,
        recorder=recorder,
        release=_PROMPTS,
        role=turn_role,
        route=None,
        validate_calls=partial(_replayable_calls, tools=native_tool_names),
    )
    evidence.update(memo=memo, clock=clock, turn=turn)
    evidence["debug"].update(
        usage=judge_usage, tool_calls=judge_tool_calls, sql_actions=sql_actions,
        protocol_events=protocol_events, model_protocol=MODEL_PROTOCOL,
        transport_protocol=selected_transport_protocol(base_protocol=TRANSPORT_PROTOCOL),
        effective_capabilities=sorted(effective_capabilities),
        sql_timeout_sec=os.environ.get("RUBIKSQL_SQL_TIMEOUT_SEC", "1200"),
    )

    async def _dispatch_judge_tool(
        tool_name: str,
        inputs: dict,
        *,
        source: str,
    ) -> object:
        """Run one accepted judge capability, the rule-trigger bookkeeping included."""
        nonlocal degraded_retrieval
        if tool_name not in effective_capabilities:
            return {"error": f"{tool_name} is not enabled for this judge runtime"}
        normalized = _normalize_tool_args(tool_name, inputs)
        tool_result = await _call_mcp(
            session,
            tool_name,
            normalized,
            db_id,
        )
        if tool_name == "get_rules":
            tool_result = rule_triggers.without_triggered(tool_result)
            rule_triggers.mark_served(tool_result)
        if degraded_knowledge_result(tool_name, tool_result):
            degraded_retrieval = True
            evidence["debug"]["degraded_retrieval"] = True
        if source == "native_tool":
            judge_tool_calls.append({
                "name": tool_name,
                "args": normalized,
                "source": source,
                "error": _tool_result_error(tool_result),
            })
        return tool_result

    with tool_contract.dispatch_context(turn_role, capabilities=active_tools), nullcontext(build_llm(
        judge_provider,
        judge_model,
        role="judge",
        seed=None if judge_provider == "claude" else judge_seed,
    )) as llm:
        while True:
            turn += 1
            evidence["turn"] = turn
            if turn > _JUDGE_MAX_TURNS:
                print(
                    f"{_LOG_META_PREFIX} [judge] turn cap ({_JUDGE_MAX_TURNS}) reached "
                    "without a submit — falling through to the fallback path.",
                    file=sys.stderr,
                    flush=True,
                )
                break
            _log(verbose, "─" * 60)
            _log(verbose, f"[Judge Turn {turn}] → {judge_model}")

            # A recorded reply whose request bytes match is replayed; otherwise the provider
            # is called. `achat_turn` is read off the module at call time (patched transports).
            response = await memo.turn(lambda: achat_turn(llm, messages, tools=tools), messages)
            merge_usage(judge_usage, response.usage)
            messages.append(response.message)

            last_reasoning = response.think
            last_content   = response.text
            native_calls   = list(response.tool_calls)
            terminal_salvage_allowed = False
            if last_content:
                _trace(verbose, "◀", last_content)
            for tc in response.tool_calls:
                function = tc.get("function") or {}
                _trace_tool_call(
                    verbose,
                    str(function.get("name") or ""),
                    str(function.get("arguments") or "{}"),
                )

            # Native transport: the assistant text is pure prose (SQL rides tool calls),
            # so the rule-trigger matcher reads it directly.
            triggered_block = await rule_triggers.match(last_content or "")
            if rule_triggers.degraded_retrieval:
                degraded_retrieval = True
                evidence["debug"]["degraded_retrieval"] = True

            if not native_calls:
                protocol_events.append({
                    "turn": turn,
                    "status": "no_action",
                    "native_tools": [],
                })
                # B1: stopped without submitting → nudge it to submit before we fall
                # through to text-extraction (Case 2) / blind fallback (Case 3).
                if last_submit_result is None and nudges < _JUDGE_MAX_NUDGES:
                    nudges += 1
                    nudge_content = _PROMPTS.surface("judge", "nudge.submit") + triggered_block
                    messages.append({
                        "role": "user",
                        "content": nudge_content,
                    })
                    _trace(verbose, "▶", nudge_content)
                    continue
                terminal_salvage_allowed = True
                break

            if native_calls:
                ran_sql = False
                for tc in native_calls:
                    function = tc.get("function") or {}
                    raw_name = str(function.get("name") or "")
                    inputs = _parse_tool_args(
                        raw_name,
                        str(function.get("arguments") or "{}"),
                    )
                    tool_name = _normalize_tool_name(
                        raw_name,
                        native_tool_names,
                    )
                    call_id = str(tc.get("id") or "")
                    if last_submit_result is not None:
                        # The submit already finalized this turn; later calls stay
                        # unexecuted but every tool_call still gets a reply so the
                        # transcript honors the provider contract.
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": "final SQL already accepted this turn; "
                                       "call not executed",
                        })
                        continue
                    if tool_name not in SQL_ACTIONS:
                        tool_result = await _dispatch_judge_tool(
                            tool_name,
                            inputs,
                            source="native_tool",
                        )
                        content = _tool_result_content(
                            verbose, tool_name, tool_result, inputs,
                        )
                        messages.append({
                            "role":         "tool",
                            "tool_call_id": call_id,
                            "content":      content,
                        })
                        continue
                    # SQL action semantics — text-era logic verbatim, only the envelope
                    # moved (a role:"tool" reply instead of a user message).
                    ran_sql = True
                    tool_result = await _dispatch_judge_tool(
                        tool_name,
                        inputs,
                        source="sql_action",
                    )
                    if (
                        tool_name == "submit_final_sql"
                        and isinstance(tool_result, dict)
                        and "error" not in tool_result
                    ):
                        # _final_sql is exactly what submit executed. Keep it out of the
                        # model-facing result body while retaining it as the final answer.
                        submitted = (
                            tool_result.pop("_final_sql", inputs.get("sql", ""))
                            or ""
                        )
                        novel = ([] if literal_guard_warned else _novel_literal_rewrite(
                            submitted, [c.get("sql") or "" for c in candidates]))
                        if novel:
                            # Bounce ONCE: the model may genuinely have verified its
                            # spelling via search_values, so a warned resubmit — even
                            # unchanged — is accepted.
                            literal_guard_warned = True
                            sql_actions.append({
                                "turn": turn,
                                "name": tool_name,
                                "sql": str(inputs.get("sql") or ""),
                                "status": "literal_guard",
                                "error": None,
                            })
                            note = _PROMPTS.surface("judge", "note.literal_guard",
                                                    novel=", ".join(novel))
                            messages.append({
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": note,
                            })
                            _trace(verbose, "▶", note)
                            continue
                        if submitted.strip():
                            last_submit_sql = submitted
                            last_submit_result = tool_result
                    action_error = (
                        tool_result.get("error")
                        if isinstance(tool_result, dict)
                        else None
                    )
                    sql_actions.append({
                        "turn": turn,
                        "name": tool_name,
                        # Non-object arguments never execute; record no SQL for them.
                        "sql": str(inputs.get("sql") or "") if isinstance(inputs, dict) else "",
                        "status": "error" if action_error else "ok",
                        "error": action_error,
                    })
                    content = model_tool_content(
                        tool_name,
                        tool_result,
                        inputs,
                        max_rows=_JUDGE_MAX_ROWS,
                    )
                    _log(verbose, f"result: {tool_name} ({len(content)} chars)")
                    _trace(verbose, "▶", content)
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": call_id,
                        "content":      content,
                    })
                protocol_events.append({
                    "turn": turn,
                    "status": "ok" if ran_sql else "native",
                    "native_tools": [
                        str((tc.get("function") or {}).get("name") or "")
                        for tc in native_calls
                    ],
                })
                if last_submit_result is not None:
                    # Accepted submission ends the run — a hook block appended now would
                    # sit unread in the transcript (and once did, right after the final
                    # submit result). Matching already happened; delivery is what stops.
                    break
                if triggered_block:
                    messages.append({"role": "user", "content": triggered_block})
                    _trace(verbose, "▶", triggered_block)
                continue

    # ── build result ──────────────────────────────────────────────────────────
    recorder.set_replay(**memo.replay_fields(turn))
    route_debug = {
        "provider": judge_provider,
        "memo_base_key": turn_base_key or cache_key,
        "implementation_context": implementation_context,
        "system_hash": hashlib.sha256(system.encode()).hexdigest()[:16],
        # Every outcome below spreads route_debug into judge_debug, so the run's render
        # identity reaches the row (and the cache) whichever way the judge ended.
        "prompt_identity": recorder.snapshot(),
        "model_protocol": MODEL_PROTOCOL,
        "transport_protocol": selected_transport_protocol(base_protocol=TRANSPORT_PROTOCOL),
        "effective_capabilities": sorted(effective_capabilities),
        "sql_timeout_sec": os.environ.get("RUBIKSQL_SQL_TIMEOUT_SEC", "1200"),
        "tool_calls": judge_tool_calls,
        "sql_actions": sql_actions,
        "protocol_events": protocol_events,
        "usage": judge_usage,
        # LLM-phase wall time only: Case 2/3 re-execution below is not included.
        "duration_sec": round(clock.real_elapsed(), 3),
        # Present only when this role consumed a failed knowledge result. Part of
        # the cached judge record on purpose — a replayed degraded judgment is still a
        # degraded answer; _vote_async lifts it to the row-level flag eval_bird reads.
        **({"degraded_retrieval": True} if degraded_retrieval else {}),
    }
    candidates_summary, candidates_debug = _candidate_projection(candidates)

    # Case 1: judge called submit_final_sql successfully
    if last_submit_result is not None:
        rows = last_submit_result.get("rows", [])
        result = {
            "sql":        last_submit_sql or "",
            "rows":       rows,
            "columns":    last_submit_result.get("columns")
                          or (list(rows[0].keys())
                              if rows and isinstance(rows[0], dict) else []),
            "row_count":  last_submit_result.get("row_count", len(rows)),
            "final":      True,
            "reasoning":  last_reasoning or last_content or "judge submitted SQL",
            "candidates": candidates_summary,
            "candidates_debug": candidates_debug,
            "judge_debug": {
                **route_debug,
                "model":             judge_model,
                "mode":              "truncated" if any_truncated else "complete_submit",
                "reasoning_content": last_reasoning,
                "raw_content":       last_content,
                "user_msg":          user_msg,
            },
        }
        evidence["cache_result"] = True
        return result

    # Case 2: judge ended with text instead of submitting (B1 nudge already exhausted).
    # B2: recover its chosen SQL from the text — JSON {"sql"}, a ```sql block, or an
    # explicit "accept candidate N" pointer — and re-execute it. NEVER score the model's
    # transcribed rows (they can carry copied-result errors); scored rows always come
    # from execution. Applies in BOTH modes (extraction beats Case 3's blind fallback). If
    # nothing is extractable or it fails to execute, fall through to Case 3.
    if terminal_salvage_allowed and last_content:
        parsed = _parse_json(last_content)
        psql, psrc = _extract_judge_sql(last_content, candidates)
        if psql:
            # Terminal recovery is deliberately broader than the interactive protocol:
            # JSON, an untagged fence, or a candidate pointer can be recovered only after
            # nudges are exhausted. Execution still goes through the same dispatcher as an
            # action the judge called.
            rerun = await _dispatch_judge_tool(
                "submit_final_sql",
                {"sql": psql},
                source=f"terminal_{psrc}",
            )
            recovery_error = (
                rerun.get("error")
                if isinstance(rerun, dict)
                else None
            )
            sql_actions.append({
                "turn": turn,
                "name": "submit_final_sql",
                "sql": psql,
                "source": f"terminal_{psrc}",
                "status": "error" if recovery_error else "ok",
                "error": recovery_error,
            })
            if isinstance(rerun, dict) and "error" not in rerun:
                final_sql = rerun.pop("_final_sql", psql)
                rows = rerun.get("rows", [])
                if rows and not isinstance(rows[0], dict):
                    rows = [dict(enumerate(r)) if isinstance(r, (list, tuple)) else {"value": r} for r in rows]
                result = {
                    "sql":        final_sql,
                    "rows":       rows,
                    "columns":    rerun.get("columns")
                                  or (list(rows[0].keys())
                                      if rows and isinstance(rows[0], dict) else []),
                    "row_count":  rerun.get("row_count", len(rows)),
                    "final":      True,
                    "reasoning":  (parsed.get("reasoning", "") if parsed else "") or last_reasoning,
                    "candidates": candidates_summary,
                    "candidates_debug": candidates_debug,
                    "judge_debug": {
                        **route_debug,
                        "model":             judge_model,
                        "mode":              f"reexec_{psrc}",
                        "reasoning_content": last_reasoning,
                        "raw_content":       last_content,
                        "user_msg":          user_msg,
                    },
                }
                evidence["cache_result"] = True
                return result
        # no usable SQL, or it failed to execute → fall through to Case 3 (the plurality candidate)

    # Case 3: fallback — no usable judge result (no submit, and Case 2's SQL was missing
    # or failed). Re-execute the plurality candidate's SQL: the first slot of the largest
    # result group, ties going to the group holding the lowest slot (`plurality_slot`; a
    # unique plurality was correct 590/590 in G24), so the scored rows come from execution
    # in BOTH modes (truncated candidate rows are incomplete; complete ones may still be
    # display-capped). Only if that candidate has no SQL, or its SQL won't execute, do we
    # fall back to its in-memory rows (still candidate rows, never hand-typed).
    print(
        f"{_LOG_META_PREFIX} [judge] no usable judge result — falling back to the "
        "plurality candidate.",
        file=sys.stderr,
        flush=True,
    )
    _log(verbose, "[judge] No usable result — falling back to the plurality candidate")
    plurality = plurality_slot(candidates, _result_sig)
    fallback = candidates[plurality - 1] if plurality else candidates[0]
    fallback_sql = fallback.get("sql", "")

    if fallback_sql:
        rerun = await _dispatch_judge_tool(
            "submit_final_sql",
            {"sql": fallback_sql},
            source="fallback_candidate",
        )
        fallback_error = (
            rerun.get("error")
            if isinstance(rerun, dict)
            else "fallback submission returned a non-dict result"
        )
        sql_actions.append({
            "turn": turn,
            "name": "submit_final_sql",
            "sql": fallback_sql,
            "source": "fallback_candidate",
            "status": "error" if fallback_error else "ok",
            "error": fallback_error,
        })
        if isinstance(rerun, dict) and "error" not in rerun:
            final_sql = rerun.pop("_final_sql", fallback_sql)
            rows = rerun.get("rows", [])
            if rows and not isinstance(rows[0], dict):
                rows = [dict(enumerate(r)) if isinstance(r, (list, tuple)) else {"value": r} for r in rows]
            return {
                "sql":        final_sql,
                "rows":       rows,
                "columns":    rerun.get("columns")
                              or (list(rows[0].keys())
                                  if rows and isinstance(rows[0], dict) else []),
                "row_count":  rerun.get("row_count", len(rows)),
                "final":      True,
                "reasoning":  "judge_fallback: re-executed the plurality candidate SQL",
                "candidates": candidates_summary,
                "candidates_debug": candidates_debug,
                "judge_debug": {
                    **route_debug,
                    # keep the judge's actual output even on fallback — without it the
                    # WHY (no submit + no extractable SQL) can't be diagnosed post-hoc
                    "model":             judge_model,
                    "mode":              "fallback_rerun",
                    "reasoning_content": last_reasoning,
                    "raw_content":       last_content,
                    "user_msg":          user_msg,
                },
            }

    rows = fallback.get("rows", [])
    return {
        "sql":        fallback_sql,
        "rows":       rows,
        "columns":    list(rows[0].keys()) if rows else [],
        "row_count":  fallback.get("row_count", len(rows)),
        "final":      True,
        "reasoning":  "judge_fallback: judge did not produce a usable result",
        "candidates": candidates_summary,
        "candidates_debug": candidates_debug,
        "judge_debug": {
            **route_debug,
            "model":             judge_model,
            "mode":              "fallback",
            "reasoning_content": last_reasoning,
            "raw_content":       last_content,
            "user_msg":          user_msg,
        },
    }


def candidate_seeds(sub_agents: list[tuple[str, str] | tuple[str, str, dict]]) -> list[int]:
    """Assign seeds in slot order, counting each distinct provider/model/options route.

    Args:
        sub_agents: Ordered candidate routes, with optional route options.

    Returns:
        list[int]: One seed per slot, starting at one independently for each route.
            A three-plus-three model mix receives ``[1, 2, 3, 1, 2, 3]``.
    """
    counts: dict[tuple[str, str, str], int] = {}
    seeds: list[int] = []
    for spec in sub_agents:
        options = spec[2] if len(spec) > 2 else {}
        route = (spec[0], spec[1], json.dumps(options, sort_keys=True))
        counts[route] = counts.get(route, 0) + 1
        seeds.append(counts[route])
    return seeds


#: Capture role of a vote's latest completed state (:func:`_publish_decision`).
_DECISION_ROLE = "vote_decision"


def _publish_decision(decision: dict, chosen: dict | None = None, hook: str | None = None) -> None:
    """Publish this vote's latest completed state to its evaluation attempt.

    A recovery deadline can stop the vote between stages, and the evaluator then submits
    the latest completed decision (``eval_bird._keep_recovery_decision``). The attempt
    capture already holds the judge's and reviewer's own records, so the state holds only
    what they lack: baseline rule health, the SQL after the review and after each finished
    post-review hook, the reviewer's record without its transcript, and the finished hooks'
    records. Rows and caches never read it.

    Args:
        decision (dict): This vote's accumulated state, updated in place.
        chosen (dict | None): The answer after its latest completed stage; ``None``
            publishes baseline rule health only.
        hook (str | None): The post-review hook that just finished; ``None`` for the review step.
    """
    if chosen is not None:
        decision["sql"] = chosen.get("sql")
        if hook is None:
            review = chosen.get("reviewer_debug")
            decision["reviewer"] = ({key: value for key, value in review.items() if key != "route_debug"}
                                    if isinstance(review, dict) else None)
        else:
            decision.setdefault("hooks", {})[hook] = chosen["judge_debug"][hook]
    cache_lineage.note(_DECISION_ROLE, decision)


def _published_decision(records: list[dict]) -> dict:
    """Return the latest state a vote published in an attempt snapshot, else ``{}``."""
    return cache_lineage.noted(records, _DECISION_ROLE)


async def _vote_pool_and_judge(
    question:       str,
    db_id:          str,
    n:              int,
    provider:       str,
    model:          str,
    judge_model:    str,
    judge_seed:     int,
    verbose:        bool,
    disabled_tools: set[str] | None,
    session:        ClientSession,
    judge_provider: str = "qwen",
    sub_agents:     list[tuple[str, str] | tuple[str, str, dict]] | None = None,
    saved_candidates: list[dict] | None = None,
) -> dict:
    """Single-session vote: sub-agents + judge share one MCP session.

    session is a shared, already-initialised ClientSession owned by _SharedSession.
    Multiple concurrent calls to _vote_async safely share the same session.
    sub_agents: optional list of (provider, model) pairs — overrides n/provider/model.
    saved_candidates: validated recorded pool; replaces generation and quorum top-up.
    Candidates run until every seat finished or the bound vote's candidate cap
    (`phase_budget.VotePhases.candidate_deadline`; `nl2sql._gather_candidates`). A quorum top-up
    re-runs only seats that failed on their own, never seats the cap cut; it is skipped when fewer
    than ``RUBIKSQL_TOPUP_MIN_LEFT_SEC`` (300 s) remain before the cap, and otherwise ends at the
    same cap. The pool is then final and the judge phase's clock starts.
    The pool is judged by RUBIKSQL_JUDGE_PASSES concurrent pipelines (`_judge_pass`) and the
    answer is their majority (`judge_majority.decide`); one pass returns its answer unchanged.
    """
    # Knowledge identity for the judge/reviewer cache keys: the candidate side already keys
    # on it (nl2sql._cache_key); without it two arms with byte-identical candidate sets
    # replayed each other's judge runs, including get_examples retrievals from the other
    # arm's example pool. Ported from codex/bird-final-ckpt-20260905 (9e225c9).
    initial_fingerprint = _db_knowledge_fingerprint(db_id)

    # ── step 1: N sub-agents in parallel ─────────────────────────────
    # In mixed mode, n is overridden by len(sub_agents).
    n = len(sub_agents) if sub_agents else n
    phases = phase_budget.current()
    cap_at = phases.candidate_deadline if phases is not None else None

    # Each candidate/judge run owns a per-run server-side session file (its intermediate
    # tables), set up + torn down inside _run_with_session / _run_judge — nothing to clean
    # up here.
    if saved_candidates is not None:
        candidates = deepcopy(saved_candidates)
    elif sub_agents:
        # Each distinct route gets its own seed counter. This lets an old/new 3+3 mix
        # reuse seeds 1–3 from both pure arms instead of assigning 1–6 by provider.
        _seeds = candidate_seeds(sub_agents)
        candidates = await _gather_mixed_in_session(
            question, db_id, sub_agents, verbose, session,
            seeds=_seeds,
            disabled_tools=disabled_tools,
            cap_at=cap_at,
        )
    else:
        candidates = await _gather_in_session(
            [question] * n, db_id, provider, model, verbose, session,
            seeds=list(range(1, n + 1)),
            disabled_tools=disabled_tools,
            slots=list(range(1, n + 1)),
            cap_at=cap_at,
        )

    valid = [c for c in candidates if not c.get("error")]
    _log(verbose, f"[vote] {len(valid)}/{len(candidates)} candidates succeeded")

    # A thin pool is the judge's worst input: it picks BETWEEN candidates, so with one or two
    # left there is nothing to pick between and a single bad generation becomes the answer.
    # Re-run just the failed seeds once when the pool falls below _CANDIDATE_QUORUM. Above it,
    # do nothing — measured on the full 1534: 97.7% of questions get all 5, 1.5% get 4, and
    # only 0.46% (7 questions) fall under the quorum, so this buys the fragile tail without
    # putting a second generation round on the other 99.5%.
    #
    # Only the FAILED seeds re-run; the ones that already succeeded are kept as-is (they are
    # cached anyway — the cache key is per (question, seed), so a survivor would replay for
    # free while a failure was never written: `_cache_set` only runs on the success path).
    # Seats the candidate cap cut are never re-run: the candidate budget is spent.
    failed_ix = [i for i, c in enumerate(candidates)
                 if c.get("error") and not str(c["error"]).startswith(CANDIDATE_CUT_ERROR)]
    cap_left = cap_at - time.monotonic() if cap_at is not None else None
    under_quorum = saved_candidates is None and bool(candidates) and len(valid) < _quorum(len(candidates))
    if under_quorum and (not failed_ix or (cap_left is not None and cap_left < _TOPUP_MIN_LEFT_SEC)):
        deadline_events.note("topup_skipped")
        reason = ("only seats cut by the candidate cap failed" if not failed_ix
                  else f"{cap_left:.0f}s left before the candidate cap")
        print(
            f"{_LOG_META_PREFIX} [candidate-quorum] skipped: {len(valid)}/{len(candidates)} "
            f"succeeded, {reason} — judging the pool as it is",
            flush=True,
        )
    elif under_quorum:
        deadline_events.note("topup_ran")
        _log(verbose, f"[vote] pool under quorum ({len(valid)}/{len(candidates)}) — "
                      f"re-running seeds {failed_ix}")
        print(
            f"{_LOG_META_PREFIX} [candidate-quorum] "
            f"{len(valid)}/{len(candidates)} succeeded — retrying "
            f"{len(failed_ix)} failed candidate(s) once",
            flush=True,
        )
        try:
            if sub_agents:
                retry = await _gather_mixed_in_session(
                    question, db_id, [sub_agents[i] for i in failed_ix], verbose, session,
                    seeds=[_seeds[i] for i in failed_ix],
                    disabled_tools=disabled_tools,
                    slots=[i + 1 for i in failed_ix],
                    cap_at=cap_at, topup=True,
                )
            else:
                retry = await _gather_in_session(
                    [question] * len(failed_ix), db_id, provider, model, verbose, session,
                    seeds=[i + 1 for i in failed_ix],
                    disabled_tools=disabled_tools,
                    slots=[i + 1 for i in failed_ix],
                    cap_at=cap_at, topup=True,
                )
            for slot, fresh in zip(failed_ix, retry):
                previous = candidates[slot]
                candidates[slot] = {
                    **fresh,
                    "attempts": [*previous.get("attempts", []),
                                 {k: v for k, v in previous.items() if k != "attempts"},
                                 *fresh.get("attempts", [])],
                }
        except (tool_contract.ToolContractDrift, GatewayConfigurationError, GatewayUnavailableError):
            raise
        except Exception as exc:                       # noqa: BLE001 — a failed top-up must not
            print(
                f"{_LOG_META_PREFIX} [candidate-quorum] retry failed "
                f"({type(exc).__name__}) — continuing with {len(valid)} candidate(s)",
                flush=True,
            )
        valid = [c for c in candidates if not c.get("error")]
        _log(verbose, f"[vote] after top-up: {len(valid)}/{len(candidates)}")

    # The pool is final: the judge phase's clock starts now.
    phase_budget.end_candidates()

    if not valid:
        errors = "; ".join(c.get("error", "unknown") for c in candidates[:2])
        # Gather has consumed the exceptions. Keep a terminal failure typed through the
        # quorum retry and projection so SharedSession.run and the evaluator can retire
        # this server. A saved pool's historical failure says nothing about this session.
        dead_session = saved_candidates is None and any(
            record.get("_dead_session")
            for candidate in candidates
            for record in (candidate, *candidate.get("attempts", []))
        )
        error_type = _DeadSessionCandidatesError if dead_session else RuntimeError
        exc = error_type(f"All {len(candidates)} candidates failed: {errors}")
        exc.provider_failure = any(c.get("provider_failure") for c in candidates)
        summary, debug = _candidate_projection(candidates)
        exc.vote_record = {"candidates": summary, "candidates_debug": debug}
        raise exc

    # ── step 2–3: N independent judge pipelines over the one pool ──────
    # One pass is the single-pass vote, run inline. Several run concurrently, each in its
    # own task, so a pass's session id, render recorder and question context stay its own.
    # A failed pass fails the vote as a single judge failure does: the question's retry
    # replays the passes that finished from their own cache keys.
    sessions = {"s": session}
    judge = partial(
        _judge_pass, question, db_id, candidates,
        judge_model=judge_model, judge_seed=judge_seed, verbose=verbose,
        judge_provider=judge_provider, sub_agents=sub_agents,
        initial_fingerprint=initial_fingerprint, sessions=sessions,
    )
    async def timed(index: int) -> dict:
        with phase_budget.track("judge_passes", **{"pass": index}):
            return await judge(pass_index=index)

    if _JUDGE_PASSES == 1:
        return await timed(0)
    outcomes = await asyncio.gather(*(timed(index) for index in range(_JUDGE_PASSES)),
                                    return_exceptions=True)
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    if failures:
        raise next((failure for failure in failures if isinstance(failure, _FATAL_PASS_ERRORS)), failures[0])
    return judge_majority.decide(outcomes, _result_sig, question=question)


async def _vote_async(*args, phases: phase_budget.VotePhases | None = None, events: dict | None = None,
                      **kwargs) -> dict:
    """`_vote_pool_and_judge` with its phase clock, its attempt's event dict and its turn-retry horizon bound.

    The phase clock starts here, when the vote starts answering: the candidate phase, or for a saved pool
    the judge phase at once. The turn-retry horizon (`llm_runtime.bind_retry_horizon`) is the current
    phase's deadline (`phase_budget.VotePhases.horizon`).

    Args:
        *args: `_vote_pool_and_judge`'s positional arguments.
        phases (phase_budget.VotePhases | None): The vote's phases; ``None`` runs unbudgeted.
        events (dict | None): The attempt's deadline event dict (`deadline_events.bind`).
        **kwargs: `_vote_pool_and_judge`'s keyword arguments.

    Returns:
        dict: The vote's answer.
    """
    phases = phases if phases is not None else phase_budget.VotePhases(None)
    phases.start(judge_only=kwargs.get("saved_candidates") is not None)
    try:
        with deadline_events.bind(events), phase_budget.bind(phases), bind_retry_horizon(phases.horizon):
            return await _vote_pool_and_judge(*args, **kwargs)
    finally:
        phases.finish()


# Introspection only (calls are unchanged): expose the wrapped implementation's parameters and defaults plus the
# wrapper's own keyword-only ones, so ``inspect.signature(_vote_async)`` shows e.g. the ``judge_provider`` default.
_vote_async.__signature__ = inspect.signature(_vote_pool_and_judge).replace(parameters=[
    *inspect.signature(_vote_pool_and_judge).parameters.values(),
    inspect.Parameter("phases", inspect.Parameter.KEYWORD_ONLY, default=None, annotation="phase_budget.VotePhases | None"),
    inspect.Parameter("events", inspect.Parameter.KEYWORD_ONLY, default=None, annotation="dict | None"),
])


#: Pass failures that end the evaluation rather than the question; they win over ordinary ones.
_FATAL_PASS_ERRORS = (tool_contract.ToolContractDrift, GatewayConfigurationError, GatewayUnavailableError,
                      asyncio.CancelledError)


async def _judge_pass(
    question:       str,
    db_id:          str,
    candidates:     list[dict],
    *,
    judge_model:    str,
    judge_seed:     int,
    verbose:        bool,
    judge_provider: str,
    sub_agents:     list[tuple[str, str] | tuple[str, str, dict]] | None,
    initial_fingerprint: str,
    sessions:       dict,
    pass_index:     int = 0,
) -> dict:
    """Run one judge pipeline over a ready candidate pool.

    The pipeline is the vote's per-pass path: baseline rule injection, the judge, the
    reviewer, the consensus guard and its answerless fallback, then the post-review hooks.
    Pass 0 keeps the single-pass cache keys and publishes the legacy recovery decision.
    Every later pass adds ``|pass=<index>`` to its judge and reviewer keys (whole cache and
    turn memo alike), so it never replays another pass's response. Every pass publishes
    its executed decision before cleanup and after each completed downstream stage.

    Args:
        question (str): Canonical question with its hint.
        db_id (str): Database the pool answers.
        candidates (list[dict]): The pool in slot order, shared read-only by every pass.
        judge_model (str): Judge and reviewer model.
        judge_seed (int): Judge seed, the same for every pass.
        verbose (bool): Mirror model-facing payloads.
        judge_provider (str): Judge and reviewer provider.
        sub_agents (list | None): Candidate routes; a routed pool keys the judge on them.
        initial_fingerprint (str): The database's knowledge fingerprint at vote start.
        sessions (dict): The vote's shared session cell, ``{"s": ClientSession}``.
        pass_index (int): 0-based pass number.

    Returns:
        dict: The pass's final answer, shaped as ``nl2sql_vote`` documents it.
    """
    session = sessions["s"]
    pass_tag = f"|pass={pass_index}" if pass_index else ""
    checkpoint_identity = judge_checkpoints.pool_identity(question, db_id, _PROMPTS.id,
                                                          initial_fingerprint, candidates)
    cache_lineage.note(judge_checkpoints.POOL_ROLE, checkpoint_identity)
    verified_rank_stages = []

    def checkpoint(chosen, stage):
        chosen = publish_rank_result(question, chosen, [], enabled=False)
        if ((chosen.get("rank_contract") or {}).get("after") or {}).get("status") == "verified":
            verified_rank_stages.append(deepcopy(chosen))
        if decision.get("baseline_rules_error"):
            chosen = {**chosen, "judge_debug": {**(chosen.get("judge_debug") or {}),
                                                "baseline_rules_error": decision["baseline_rules_error"]}}
        if decision.get("baseline_degraded"):
            chosen = {**chosen, "degraded_retrieval": True}
        judge_checkpoints.publish(checkpoint_identity, pass_index, chosen, stage)

    def publish(decision, chosen=None, hook=None):
        if pass_index == 0:
            _publish_decision(decision, chosen, hook)
        if chosen is not None:
            checkpoint(chosen, hook or "review")
    # ── step 2: build judge input ─────────────────────────────────────
    # E2 聚类信号,转正为默认行为(07-24,用户裁定,无开关):候选按执行结果签名分簇,
    # 判官头部标注 result-group 与簇大小。误选题 52-55% 的正解藏在少数簇——把隐藏结构
    # 变成可见信息;措辞为情报而非指令(净 +3~5,对照=噪声,从众化未观测)。
    _cluster: dict[int, tuple[str, int]] = {}
    for gi, g in enumerate(result_groups(candidates, _result_sig)):
        label = chr(ord("A") + gi)
        for i in g:
            _cluster[i] = (label, len(g))

    # The judge's first user message is the judge.user surface: the question with its
    # synonym notes, every candidate numbered in pool order with its result-group note (so
    # _extract_judge_sql's "candidate N" pointers keep working), the baseline rules and the
    # closing instruction. Clustering, fenced SQL and the result previews are the data this
    # loop hands it; the frame around them is the release's.
    judge_candidates: list[dict] = []
    any_truncated = False
    for i, c in enumerate(candidates, 1):
        if c.get("error"):
            judge_candidates.append({"error": str(c["error"]), "group_size": 0, "group_label": "",
                                     "sql_fenced": "", "n_rows": 0, "preview": ""})
            continue
        rows = c.get("rows", [])
        preview, truncated = _table_display(
            rows,
            _JUDGE_MAX_ROWS,
            columns=c.get("columns"),
        )
        any_truncated = any_truncated or truncated
        label, size = _cluster[i]
        judge_candidates.append({"error": "", "group_size": size, "group_label": label,
                                 "sql_fenced": fenced_sql(c.get("sql", "")),
                                 "n_rows": int(c.get("row_count", len(rows))),
                                 "preview": preview})
    # One render recorder for the judge's run, created before its user message exists so
    # the frame renders below are counted; _run_judge_inner rebinds it inside its task.
    judge_recorder = _recorder_for("judge", None)

    # Baseline rule retrieval may embed the question. Validate its full answering
    # contract before that spend, outside the optional retrieval error handling.
    # A dead session is not replaced here. This coroutine runs on that session's own loop, so
    # stopping it would stop the loop under every vote it serves, and a replacement server's
    # ClientSession belongs to another loop. The error goes to the question boundary instead:
    # SharedSession.run marks the session dead, eval_bird's retire_dead_session swaps exactly
    # that server, and the retry replays every finished candidate and judge pass from cache.
    try:
        initial_judge_tools, initial_judge_capabilities = await _judge_runtime_tools(
            session, _JUDGE_TOOLS,
        )
    except _DEAD_SESSION_EXCS as error:
        error.judge_debug = {"error": str(error), "messages": [], "transcript_status": "interrupted",
                             "prompt_identity": judge_recorder.snapshot()}
        _log(verbose, "\n[judge] MCP preflight session died — leaving its replacement to the question retry")
        raise

    # Fetch relevant business rules and inject into judge context
    rules_text = ""
    injected_rule_titles: set[str] = set()
    injected_rule_ids: set[str] = set()
    baseline_session_handoff = False
    # Both judge system prompts assert "The top-15 baseline rules for this question are already
    # injected ABOVE in full", so a failure here tells the judge rules exist that do not. Stay
    # non-fatal (rules are optional) but never silent. Key the alarm on the error-dict shape,
    # not on emptiness — a database can legitimately retrieve zero rules.
    # The error text is also kept for the row (judge_debug.baseline_rules_error): this is a
    # get_rules the harness makes on the judge's behalf, so its failure must be able to set
    # eval_bird's ``degraded_retrieval`` flag like any get_rules the model calls itself.
    baseline_rules_error = None
    baseline_degraded = False
    baseline_session_id = uuid.uuid4().hex
    baseline_session_token = _SESSION_ID.set(baseline_session_id)
    try:
        # This retrieval runs in the knowledge session the judge itself will inherit
        # (baseline_session_id is handed to _run_judge on success), so the [R01..R15]
        # ids rendered below stay REAL, referable ids for the whole judge run — later
        # get_rules/hook retrievals continue the same numbering instead of colliding.
        with tool_contract.dispatch_context("judge", capabilities=initial_judge_capabilities):
            rules_result = await _raw_call(
                session,
                "get_rules",
                # The release's judge_baseline_rules knob, 15 by default (user ruling 08-06: 20
                # diluted; rule-stack dilution is a measured failure family). Both judge prompts
                # promise "The top-15 baseline rules ... injected ABOVE in full" and the number
                # here must keep that promise true: a release that moves the knob edits the text.
                #
                # It was not always true. This comment used to claim the call "has always put 20
                # rules in front of the judge" on the theory that the old ``k`` meant per-branch
                # width and returned ``2 * k``; the traces say otherwise — the 7/27 run delivered
                # exactly 15 rules to the judge in all 1480 of its judged questions, against a
                # prompt that told the model it had 20. Today's runs deliver 20 in all of them.
                # So the promise was false before the k-semantics fix and is true after it, which
                # is the opposite of what the old note asserted.
                {"question": question, "k": _PROMPTS.knob("judge_baseline_rules"), "session_id": baseline_session_id},
                db_id,
            )
        baseline_degraded = degraded_knowledge_result("get_rules", rules_result)
        if isinstance(rules_result, list) and rules_result:
            if isinstance(rules_result[0], dict) and "error" in rules_result[0]:
                baseline_rules_error = rules_result[0]["error"]
                print(
                    f"{_LOG_META_PREFIX} [judge] rule injection FAILED for db={db_id!r} "
                    f"({rules_result[0]['error']}) — judging without business rules while "
                    "the prompt states they were injected.",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                with judge_recorder:
                    rules_text = render_rules(
                        rules_result,
                        heading=_PROMPTS.surface("shared", "rules_heading.baseline"),
                        explain_scope_conflict=True,
                        include_ids=True,
                        fields=("scope", "desc"),
                    )
                baseline_session_handoff = True
                # Cross-session dedup seed: the runtime rule hook and later get_rules
                # calls must not re-serve these 15. Ids are session-local (this was a
                # throwaway session) — titles are the stable identity.
                injected_rule_titles = {
                    str(r.get("title") or "") for r in rules_result
                    if isinstance(r, dict)
                } - {""}
                injected_rule_ids = {
                    str(r.get("virtual_id") or "") for r in rules_result
                    if isinstance(r, dict)
                } - {""}
    except (tool_contract.ToolContractDrift, GatewayConfigurationError):
        # Preserve fatal contract/configuration errors at the baseline-retrieval boundary.
        raise
    except Exception as e:
        print(
            f"{_LOG_META_PREFIX} [judge] rule injection FAILED for db={db_id!r} "
            f"({type(e).__name__}: {e}) — judging without business rules while the "
            "prompt states they were injected.",
            file=sys.stderr,
            flush=True,
        )
    finally:
        _SESSION_ID.reset(baseline_session_token)
        if not baseline_session_handoff and not _detach_close_session(session, baseline_session_id):
            # Injection failed — nothing references these ids; drop the session.
            try:
                await session.call_tool("close_session", {"session_id": baseline_session_id})
            except Exception:
                pass
    # A recovery deadline can stop this vote after any later stage. Its latest completed
    # state, a failed baseline included, goes to the attempt capture (_publish_decision).
    decision: dict = {}
    if baseline_rules_error is not None:
        decision["baseline_rules_error"] = baseline_rules_error
    if baseline_degraded:
        decision["baseline_degraded"] = True
    if decision:
        publish(decision)

    # The judge needs the vocabulary notes at least as much as the candidates do: its whole job
    # is deciding which candidate read the question right, and "did this one bind `charter
    # schools` to the correct column" is exactly that decision. Same block, same source, same
    # reference-not-instruction wording — one implementation so the two sides cannot drift.
    with judge_recorder:
        user_msg = _PROMPTS.surface(
            "judge", "user",
            question=question,
            synonyms=await _async_synonym_context(db_id, question),
            candidates=judge_candidates,
            rules=rules_text,
        )
        # _JUDGE_SYSTEM already embeds _JUDGE_COMMON. Rendered through the release here, so
        # the judge's fired set names the row that provided it (the render is memoised).
        judge_system = _PROMPTS.surface("judge", "system")
    judge_system_hash = hashlib.sha256(judge_system.encode()).hexdigest()[:16]
    # Prompt-release and sampling-universe terms of the judge key: absent under r0 and an
    # unset RUBIKSQL_CACHE_NS, so today's keys hold; see _judge_execution_identity.
    _judge_prompt_extra = _PROMPTS.key_extra("judge", None)
    _cache_ns = os.environ.get("RUBIKSQL_CACHE_NS", "")
    # Opt-in A/B-arm namespace (RUBIKSQL_CACHE_ARM_TAG). The judge/reviewer keys now also
    # carry the knowledge fingerprint (kb=), so arms that differ in rules/examples no longer
    # replay each other's judge run; the tag remains for arms whose knowledge is identical
    # but whose judge-side treatment differs. Unset (default) leaves keys unchanged.
    _arm = os.environ.get("RUBIKSQL_CACHE_ARM_TAG", "")
    _arm_tag = f"|arm={_arm}" if _arm else ""
    # A judge-side knowledge-access arm would be tagged here (belt-and-braces over
    # _judge_system_hash, which sees the prompt lines but not the TOOL SET). None are in flight:
    # query_knowledge and expand_examples are unconditional production now, so there is nothing
    # left to namespace apart from.
    _route_tag = ""
    if sub_agents and any(len(spec) > 2 and spec[2] for spec in sub_agents):
        signature = [
            [spec[0], spec[1], spec[2] if len(spec) > 2 else {}]
            for spec in sub_agents
        ]
        _route_tag = "|candidate_routes=" + json.dumps(
            signature, sort_keys=True, separators=(",", ":")
        )
    # ── step 3: judge — one run (same session, retry once on ANY dead-session error).
    # The session lives in a mutable cell because that retry REBUILDS it: the reviewer
    # which runs afterwards must start from the REBUILT session, and re-reading the stale
    # `session` param would evict the healthy replacement.
    _cur_session = sessions

    async def _judge_once(judge_session, seed_k: int) -> dict:
        contract = tool_contract.role_contract("judge", capabilities=_JUDGE_TOOLS)
        if judge_session is session:
            runtime_tools, effective_capabilities = initial_judge_tools, initial_judge_capabilities
        else:
            runtime_tools, effective_capabilities = await _judge_runtime_tools(
                judge_session, _JUDGE_TOOLS,
            )
        execution_identity = _judge_execution_identity(
            effective_capabilities,
            judge_provider=judge_provider,
            arm_tag=_arm_tag,
            route_tag=_route_tag,
            prompt=_judge_prompt_extra,
            ns=_cache_ns,
            implementation=contract["digest"],
        )
        ck = hashlib.sha256(
            f"judge|{judge_model}|{seed_k}|{judge_system_hash}|{user_msg}"
            f"|{execution_identity}|kb={initial_fingerprint}{pass_tag}".encode()
        ).hexdigest()
        # Turn memo base key: the same key with the prompt term left out (turn_memo.py), so
        # a judge under a changed release replays the turns of the parent's judge.
        base_identity = _judge_execution_identity(
            effective_capabilities,
            judge_provider=judge_provider,
            arm_tag=_arm_tag,
            route_tag=_route_tag,
            ns=_cache_ns,
            implementation=contract["digest"],
        )
        turn_base_key = hashlib.sha256(
            f"judge|{judge_model}|{seed_k}|{judge_system_hash}|{user_msg}"
            f"|{base_identity}|kb={initial_fingerprint}{pass_tag}".encode()
        ).hexdigest()
        cached = _cache_get(ck)
        if cached is not None:
            _log(verbose, "\n[judge] cache hit")
            # The judge's decision replays from the run that first wrote this key; the
            # pool is THIS run's. The key hashes the candidates' SQL and results, not
            # their identities, so a pool that diverged on a candidate-only prompt change
            # and resubmitted the same SQL hits here — its own snapshots (release, replay
            # cause) must reach the row, or the outcome reads the question as undiverged.
            replayed = _replayed(cached)
            replayed["judge_debug"] = {
                **(replayed.get("judge_debug") or {}),
                "consumer_implementation_context": {"context": contract["context"], "digest": contract["digest"]},
            }
            replayed["candidates"], replayed["candidates_debug"] = _candidate_projection(candidates)
            return replayed
        _log(
            verbose,
            f"\n[judge] mode={'agentic-truncated' if any_truncated else 'agentic-complete'} "
            f"— {judge_model} seed={seed_k} — {len(user_msg)} chars",
        )
        return await _run_judge(
            candidates,
            user_msg,
            ck,
            db_id,
            judge_session,
            judge_model,
            seed_k,
            verbose,
            any_truncated=any_truncated,
            judge_provider=judge_provider,
            question=question,
            preinjected_titles=injected_rule_titles,
            preinjected_ids=injected_rule_ids,
            preinjected_session_id=(baseline_session_id if baseline_session_handoff else None),
            system=judge_system,
            active_tools=effective_capabilities,
            runtime_tools=runtime_tools,
            recorder=judge_recorder,
            turn_base_key=turn_base_key,
            turn_lookup=_turn_lookup_allowed("judge", None),
        )

    async def _one_judge(seed_k: int) -> dict:
        judge_session = _cur_session["s"]
        try:
            # A dead session propagates to the question boundary (see the preflight above).
            chosen = await _judge_once(judge_session, seed_k)
            checkpoint(chosen, "judge")
            return chosen
        finally:
            sid = _SESSION_ID.get()   # the session_id _run_judge set in this (shared) task context
            if sid and not _detach_close_session(judge_session, sid):
                try:
                    with session_timing.measure("tool", "close_session"):
                        await judge_session.call_tool("close_session", {"session_id": sid})
                except Exception:
                    pass

    async def _maybe_review(chosen):
        if not (_REVIEWER and isinstance(chosen, dict) and (chosen.get("sql") or "").strip()):
            return chosen
        if not _v2_risk_signal(chosen, question):
            chosen = dict(chosen)
            chosen["reviewer_debug"] = {"active": False, "changed": False,
                                        "skipped": "no-risk-signal", "v": 2}
            return chosen
        try:
            with phase_budget.track("reviewer", **{"pass": pass_index}):
                reviewed = await _review_final(chosen, question, db_id, _cur_session["s"], judge_model,
                                               verbose, judge_provider, _REVIEWER_TOOLS,
                                               arm_tag=_arm_tag, route_tag=_route_tag,
                                               knowledge_fingerprint=initial_fingerprint,
                                               pass_tag=pass_tag)
            checkpoint(reviewed, "review")
            return reviewed
        except (tool_contract.ToolContractDrift, GatewayConfigurationError, GatewayUnavailableError):
            raise
        except Exception as exc:   # ordinary reviewer failures keep the judge answer
            _log(verbose, f"\n[reviewer] failed ({exc}) — keeping judge answer")
            chosen = dict(chosen)
            chosen["reviewer_debug"] = {"active": True, "changed": False,
                                        "error": str(exc)[:300],
                                        "route_debug": getattr(exc, "judge_debug", {
                                            "error": str(exc), "messages": [],
                                            "transcript_status": "interrupted",
                                        })}
            return chosen
        finally:
            # _review_final calls _run_judge a SECOND time, which sets a fresh _SESSION_ID —
            # but _one_judge's own finally has already run by now, so without this the
            # reviewer's session file is never dropped (one leftover per reviewed question).
            sid = _SESSION_ID.get()
            if sid and not _detach_close_session(_cur_session["s"], sid):
                try:
                    with session_timing.measure("tool", "close_session"):
                        await _cur_session["s"].call_tool("close_session", {"session_id": sid})
                except Exception:
                    pass

    try:
        chosen = await _maybe_review(await _one_judge(judge_seed))
    except Exception as exc:
        summary, debug = _candidate_projection(candidates)
        exc.vote_record = {
            "candidates": summary, "candidates_debug": debug,
            "judge_debug": getattr(exc, "judge_debug", {
                "error": str(exc), "messages": [], "transcript_status": "interrupted",
                "prompt_identity": judge_recorder.snapshot(),
            }),
        }
        raise
    if not isinstance(chosen, dict):
        return chosen
    publish(decision, chosen)

    async def _hook_execute(sql: str) -> dict:
        # Post-review hooks control the reviewer's final, so their probes dispatch as the reviewer.
        with tool_contract.dispatch_context("reviewer"):
            return await _raw_call(_cur_session["s"], "submit_final_sql", {"sql": sql}, db_id)

    # Post-review hooks run in KNOBS declaration order: the consensus guard first reverts a
    # judge departure from the candidate consensus, its answerless fallback replaces an answerless
    # final and its defect fallback an inert superlative or a non-constant bare aggregate, then the
    # gate, count-weight, selection and entity-percentage rules apply.
    if _PROMPTS.knob("consensus_guard"):
        pre_guard_sql = chosen.get("sql")
        strict_added_tables = _PROMPTS.knob("consensus_guard_strict_added_tables")
        try:
            guarded, guard_debug = await apply_consensus_guard(
                question, chosen, candidates, _hook_execute, _result_sig,
                strict_added_tables=strict_added_tables,
            )
            chosen = guarded
        except Exception as exc:
            # This optional step must never discard the confirmed answer.
            version = (CONSENSUS_GUARD_STRICT_ADDED_TABLES_VERSION if strict_added_tables
                       else CONSENSUS_GUARD_IMPLEMENTATION_VERSION)
            guard_debug = {"v": 1, "implementation_version": version,
                           "pre_guard_sql": pre_guard_sql, "fired": False, "features": [],
                           "group_label": None, "group_size": 0, "slot": None, "skipped": "error",
                           "errors": [{"feature": "guard", "type": type(exc).__name__}]}
        judge_debug = chosen.get("judge_debug")
        chosen = {**chosen, "judge_debug": {
            **(judge_debug if isinstance(judge_debug, dict) else {}),
            "consensus_guard": guard_debug,
        }}
        publish(decision, chosen, "consensus_guard")
        # The guard's answerless fallback: an empty, all-NULL or single-zero final gives way to the
        # largest agreeing group of at least two answered candidates, before the remaining hooks.
        pre_fallback_sql = chosen.get("sql")
        try:
            chosen, fallback_debug = apply_answerless_fallback(chosen, candidates, _result_sig)
        except Exception as exc:
            # This optional step must never discard the confirmed answer.
            fallback_debug = {"v": 1, "implementation_version": ANSWERLESS_FALLBACK_VERSION,
                              "pre_fallback_sql": pre_fallback_sql, "fired": False, "from_slot": None,
                              "to_slot": None, "group_size": 0, "skipped": "error",
                              "errors": [{"feature": "answerless_fallback", "type": type(exc).__name__}]}
        chosen = {**chosen, "judge_debug": {**chosen["judge_debug"], "answerless_fallback": fallback_debug}}
        publish(decision, chosen, "answerless_fallback")
        # The defect fallback: a final that aggregates without GROUP BY next to a non-constant bare
        # column (wrong under any reading), or an aggregate-only final whose ORDER BY and LIMIT rank
        # one aggregated row (an inert superlative) that no unflagged candidate's result supports,
        # gives way to the largest answered group whose representative is neither.
        pre_fallback_sql = chosen.get("sql")
        try:
            chosen, defect_debug = await apply_defect_fallback(chosen, candidates, _result_sig, _hook_execute)
        except Exception as exc:
            # This optional step must never discard the confirmed answer.
            defect_debug = {"v": 1, "implementation_version": DEFECT_FALLBACK_VERSION,
                            "pre_fallback_sql": pre_fallback_sql, "fired": False, "shape": None, "distinct": None,
                            "from_slot": None, "to_slot": None, "group_size": 0, "flagged_slots": [],
                            "skipped": "error", "errors": [{"feature": "defect_fallback", "type": type(exc).__name__}]}
        chosen = {**chosen, "judge_debug": {**chosen["judge_debug"], "defect_fallback": defect_debug}}
        publish(decision, chosen, "defect_fallback")
    if _PROMPTS.knob("submission_gate"):
        pre_gate_sql = chosen.get("sql")
        hint_aggregate_fidelity = _PROMPTS.knob("submission_gate_hint_aggregate")
        try:
            gated, gate_debug = await apply_submission_gate(
                question, chosen, candidates, _hook_execute, _result_sig,
                left_inner_min_support_margin=_PROMPTS.knob("submission_gate_left_inner_min_support_margin"),
                hint_aggregate_fidelity=hint_aggregate_fidelity,
            )
            chosen = gated
        except Exception as exc:
            # This optional step must never discard the confirmed answer.
            version = (SUBMISSION_GATE_HINT_AGGREGATE_VERSION if hint_aggregate_fidelity
                       else SUBMISSION_GATE_IMPLEMENTATION_VERSION)
            gate_debug = {"v": 1, "implementation_version": version,
                          "pre_gate_sql": pre_gate_sql, "fired": [],
                          "errors": [{"rule": "gate", "type": type(exc).__name__}]}
        judge_debug = chosen.get("judge_debug")
        chosen = {**chosen, "judge_debug": {
            **(judge_debug if isinstance(judge_debug, dict) else {}),
            "submission_gate": gate_debug,
        }}
        publish(decision, chosen, "submission_gate")
    if _PROMPTS.knob("count_weight"):
        pre_hook_sql = chosen.get("sql")
        try:
            weighted, weight_debug = await apply_count_weight(
                question, chosen, candidates, _hook_execute, _result_sig,
            )
            chosen = weighted
        except Exception as exc:
            # This optional step must never discard the confirmed answer.
            weight_debug = {"v": 1, "implementation_version": COUNT_WEIGHT_IMPLEMENTATION_VERSION,
                            "pre_hook_sql": pre_hook_sql, "fired": False, "skipped": "error",
                            "errors": [{"feature": "count_weight", "type": type(exc).__name__}]}
        judge_debug = chosen.get("judge_debug")
        chosen = {**chosen, "judge_debug": {
            **(judge_debug if isinstance(judge_debug, dict) else {}),
            "count_weight": weight_debug,
        }}
        publish(decision, chosen, "count_weight")
    if _PROMPTS.knob("selection_normalization"):
        pre_hook_sql = chosen.get("sql")
        try:
            normalized, normalization_debug = await apply_selection_normalization(
                question, chosen, _hook_execute, _result_sig,
            )
            chosen = normalized
        except Exception as exc:
            # This optional step must never discard the confirmed answer.
            normalization_debug = {"v": 1, "implementation_version": SELECTION_NORMALIZATION_IMPLEMENTATION_VERSION,
                                   "pre_hook_sql": pre_hook_sql, "fired": False, "skipped": "error",
                                   "errors": [{"feature": "selection_normalization", "type": type(exc).__name__}]}
        judge_debug = chosen.get("judge_debug")
        chosen = {**chosen, "judge_debug": {
            **(judge_debug if isinstance(judge_debug, dict) else {}),
            "selection_normalization": normalization_debug,
        }}
        publish(decision, chosen, "selection_normalization")
    if _PROMPTS.knob("entity_percentage_count"):
        pre_hook_sql = chosen.get("sql")
        try:
            counted, percentage_debug = await apply_entity_percentage_count(
                question, chosen, _hook_execute, _result_sig,
            )
            chosen = counted
        except Exception as exc:
            # This optional step must never discard the confirmed answer.
            percentage_debug = {"v": 1, "implementation_version": ENTITY_PERCENTAGE_COUNT_IMPLEMENTATION_VERSION,
                                "pre_hook_sql": pre_hook_sql, "fired": False, "skipped": "error",
                                "errors": [{"feature": "entity_percentage_count", "type": type(exc).__name__}]}
        judge_debug = chosen.get("judge_debug")
        chosen = {**chosen, "judge_debug": {
            **(judge_debug if isinstance(judge_debug, dict) else {}),
            "entity_percentage_count": percentage_debug,
        }}
        publish(decision, chosen, "entity_percentage_count")
    if baseline_rules_error is not None:
        # Attached AFTER the judge cache boundary on purpose: the baseline fetch runs
        # fresh every call (never cached), so this records THIS run's retrieval health —
        # a healthy replay of a healthy prior judge run stays unmarked, and a degraded
        # baseline marks the row even when the judge itself replayed.
        judge_debug = chosen.get("judge_debug")
        chosen = {**chosen, "judge_debug": {
            **(judge_debug if isinstance(judge_debug, dict) else {}),
            "baseline_rules_error": baseline_rules_error,
        }}
    # Row-level degraded flag: any consumer of failed knowledge taints the answer —
    # a candidate (its loop set the flag on its own record), the judge's baseline
    # injection (the harness-made get_rules above), the judge's own tool calls, or the
    # reviewer's (whose judge_debug lives under reviewer_debug.route_debug). eval_bird
    # copies this single field onto the row; the layer-2 sweep re-answers it fresh.
    reviewer_route = (chosen.get("reviewer_debug") or {}).get("route_debug") or {}
    degraded = (
        baseline_degraded
        or any(isinstance(c, dict) and degraded_record(c) for c in candidates)
        or degraded_record(chosen.get("judge_debug") or {})
        or bool(isinstance(reviewer_route, dict)
                and degraded_record(reviewer_route))
    )
    if degraded:
        chosen = {**chosen, "degraded_retrieval": True}
    ordered = [candidates[i - 1] for group in result_groups(candidates, _result_sig) for i in group]
    chosen = publish_rank_result(question, chosen, [*reversed(verified_rank_stages), *ordered], enabled=rank_publication_enabled())
    checkpoint(chosen, "complete")
    return chosen


# ── public API ────────────────────────────────────────────────────────────────

def nl2sql_vote(
    question:       str,
    db_id:          str,
    n:              int  = 3,
    provider:       str  = "deepseek",
    judge_model:    str  = "qwen3.7-max",
    judge_provider: str  = "qwen",
    judge_seed:     int  = 42,
    verbose:        bool = False,
    disabled_tools: set[str] | None = None,
    sub_agents:     list[tuple[str, str] | tuple[str, str, dict]] | None = None,
    timeout_s:      float | None = None,
    hint:           str | list[str] | tuple[str, ...] | None = None,
    saved_candidates: list[dict] | None = None,
    deadline: _SharedSessionDeadline | None = None,
    events: dict | None = None,
    phases: phase_budget.VotePhases | None = None,
) -> dict:
    """Run nl2sql n times in parallel, then judge and synthesise with an agentic model.

    All sub-agents and the judge share ONE MCP session (single server startup).

    The judge runs the agentic submit loop in BOTH modes and always finalises through
    ``submit_final_sql``, so the returned rows are that call's execution result — never the
    judge's transcribed context. The two modes differ only in system prompt and tool set: the
    truncated mode tells the judge that some candidate output was cut and must be re-queried.
    A judge that answers with JSON instead of submitting is the non-compliant path; its SQL is
    recovered and re-executed, and failing that the plurality candidate's SQL is.

    Args:
        question:       Natural-language question.
        db_id:          Registered database name.
        n:              Number of nl2sql candidates to generate (default 3, the pipeline's count).
                        Ignored when sub_agents is provided.
        provider:       Provider for the n candidate calls (default "deepseek"). Ignored
                        when sub_agents is provided.
        judge_model:    Model used for judging/synthesising (default "qwen3.7-max", the judge of a
                        release that does not set its ``judge_model`` knob; the pipeline and the eval
                        entry points pass the pinned release's, ``rubiksql_hb.judge_policy``). The
                        reviewer runs on the same provider and model.
        judge_provider: Provider for the judge call (default "qwen", DashScope).
        judge_seed:     Seed for judge call (default 42).
        verbose:        Mirror complete model-facing payloads with ``▶``/``◀`` and label
                        all display-only diagnostics as absent from model context.
        disabled_tools: Set of optional tool names to hide from the sub-agents.
        sub_agents:     Optional list of ``(provider, model)`` or ``(provider, model, opts)``
                        tuples for mixed-provider ensembles; ``opts`` carries per-candidate
                        overrides such as ``{"t": 0.7}`` for temperature. Overrides n and
                        provider. eval_bird always passes the 3-tuple form.
        timeout_s:      Wall-clock ceiling for the whole vote; None means no explicit budget.
                        A budgeted vote answers in two phases (`rubiksql_hb.phase_budget`):
                        candidates until all finish or ``RUBIKSQL_CANDIDATE_BUDGET_SEC``
                        (2000 s), then the judge passes, reviewer and hooks within
                        ``RUBIKSQL_JUDGE_BUDGET_SEC`` (2400 s) of the candidate phase's end; a
                        saved pool starts with its judge phase. A ceiling below the two budgets'
                        sum scales the candidate budget down in proportion. The vote is stopped
                        at its phase deadline, read again while it runs.
        hint:           Optional evidence/preference text, canonically appended to every
                        candidate/judge/retrieval semantic question.
        saved_candidates: Validated recorded candidate objects with SQL, execution results,
                        transcripts and source provenance. A copied pool replaces candidate
                        generation and quorum retries; judging, review and hooks still run.
                        The caller validates release, question and source identity before use.
        deadline: Optional absolute shared-session deadline created before acquisition.
                        It cancels the vote once at expiry; the stopped vote then starts no
                        model, pacing, synonym or tool work and detaches its session cleanup.
                        Its owner joins the worker until ``deadline.join_by``; completion
                        preserves exceptions raised while unwinding.
        events: Optional dict the deadline mechanics count into (`rubiksql_hb.deadline_events`);
                        the caller keeps it, so the counts survive a cancelled vote.
        phases: Optional phase clock built from the same ``timeout_s``
                        (`phase_budget.VotePhases`); the caller keeps it and reads the vote's
                        phase timing record from it (`VotePhases.snapshot`), which survives a
                        vote stopped at its deadline. Built from ``timeout_s`` when omitted.

    Returns:
        dict: ``{sql, rows, columns, row_count, final, reasoning, candidates,
        candidates_debug, judge_debug}``, plus ``reviewer_debug`` whenever the reviewer pass
        runs. ``judge_debug.mode`` records which path produced the answer
        (``submit`` / ``reexec_*`` / ``fallback_rerun``).
        When enabled, ``judge_debug.consensus_guard`` records the consensus guard's
        implementation version, input SQL, consensus group, fired features and submitted
        candidate slot, ``judge_debug.answerless_fallback`` records the guard's answerless
        fallback (version, input SQL, fired, from/to slot and group size), and ``judge_debug.submission_gate`` records the gate's
        implementation version, input SQL, rule firings and optional errors.
        ``judge_debug.count_weight`` records its implementation version, input and variant
        SQL, acceptance or skip reason, and optional errors when that hook is enabled.
        ``judge_debug.selection_normalization`` also records the ordered operations and
        their individual acceptance or skip reasons when enabled.
        ``judge_debug.entity_percentage_count`` records the input and variant SQL,
        implementation version, acceptance or skip reason, and optional errors when enabled.
        With ``RUBIKSQL_JUDGE_PASSES`` above one, N such pipelines judge the pool
        concurrently and the result is the majority pass's, plus ``judge_passes`` (every
        pass's final fields and records except the rows and the shared pool) and
        ``judge_majority`` (``passes``, ``chosen_pass``, ``groups``, ``judge_sources``),
        see `rubiksql_hb.judge_majority`.

    Raises:
        ValueError: ``provider`` (or ``sub_agents[0][0]``) or ``judge_provider`` is not a
            registered provider.
    """
    sub_provider = provider if not sub_agents else sub_agents[0][0]
    cfg = _PROVIDERS.get(sub_provider)
    if cfg is None:
        raise ValueError(f"Unknown provider '{sub_provider}'. Choose from: {list(_PROVIDERS)}")

    if judge_provider not in _PROVIDERS:
        raise ValueError(f"Unknown judge_provider '{judge_provider}'. Choose from: {list(_PROVIDERS)}")

    question = question_with_hint(question, hint)
    if verbose:
        if saved_candidates is not None:
            _log(verbose, f"[vote] Reusing {len(saved_candidates)} saved candidates "
                          f"for '{question[:60]}' on {db_id}")
        elif sub_agents:
            desc = ", ".join(
                f"{spec[0]}:{spec[1]}"
                + (":" + ":".join(f"{key}={value}" for key, value in spec[2].items())
                   if len(spec) > 2 and spec[2] else "")
                for spec in sub_agents
            )
            _log(
                verbose,
                f"[vote] Mixed sub-agents [{desc}] for '{question[:60]}' on {db_id}",
            )
        else:
            _log(
                verbose,
                f"[vote] Generating {n} candidates for '{question[:60]}' on {db_id}",
            )

    # Get (or lazily start) the shared MCP session for this db_id's question slot
    # (`server_slot`): candidates, every judge pass, the reviewer and each close_session run
    # inside this one `run`, so all of the question's work meets one MCP server.
    # timeout_s → real in-loop cancellation of the whole vote coroutine at its current phase
    # deadline: a hung API/MCP read is unwound instead of blocking the caller's thread forever.
    phases = phases if phases is not None else phase_budget.VotePhases(timeout_s)
    with server_scope(server_slot(question)):
        while True:
            shared = _get_shared_session(db_id)
            try:
                return shared.run(_vote_async(
                    question, db_id, n,
                    sub_provider, cfg["default_model"],
                    judge_model, judge_seed, verbose, disabled_tools,
                    session=shared.session,
                    judge_provider=judge_provider,
                    sub_agents=sub_agents,
                    saved_candidates=saved_candidates,
                    phases=phases,
                    events=events,
                ), **({"deadline": deadline} if deadline is not None else
                      {"until": phases.deadline} if phases.budget_sec is not None else {}))
            except _SessionSubmissionRejected:
                continue
