"""Lab analytics entities — experiments, per-question traces, and the QA ground truth.

The lab is an adapter over the single package-bound control database. Content, history,
experiments and conversations share its unchanged workspace id and binding. It remains
separate from per-database knowledge stores: its rows span databases and experiments,
and none joins ``workspace.ALL_ENTITIES`` or the benchmark MCP surface.

The workspace is empty at build time by design: ``QAEntity`` fills from the official
question file, and experiment/trace rows arrive after eval runs through the bound
Authority writer. MCP serves only readers. Rows are always written whole: ``upsert``
replaces the entire row, so use ``set`` to patch an existing row.

Transcripts are HeavenBase's own rows: the workspace enables the ``agent`` extension, and
ingest writes one ``agent-session`` per agent run of a question — each candidate slot, the
judge and the reviewer — with its ordered ``agent-message`` rows, every message as it was sent
or received (tool calls and tool results included) in ``payload``. A session's ``agent_ref``
is the role; its ``state`` names the experiment, question, database, slot, release, replay
record and the source the messages came from. ``CandidateEntity.session_id`` and
``JudgeEntity.session_id`` point at their sessions and ``TraceEntity.session_ids`` lists every
session of the question; read a conversation with a query on ``agent-message.session_id``,
ordered by ``idx``. The extension resolves its entity classes only once a workspace enabled
it, so :func:`transcript_entities` hands them out after :func:`open_lab_workspace`. An
experiment recorded without transcripts (``meta_ops record``) has no session or message rows:
those links are null, ``ExperimentEntity.provenance`` names ``transcripts: run_directory`` and
the run directory of ``result_files`` keeps every conversation.

Prompt documents are canonical ``sys-prompt`` rows. Tools, general rules and parameter sets are the
typed version entities ``ToolEntity``, ``GeneralRuleEntity`` and ``KnobSetEntity``: their canonical
rows (``format`` set) are the immutable versions a ``rubiksql.closure/2`` release references exactly.
GeneralRule and KnobSet rows with an empty ``format`` are historical projections of the immutable
``rules.general`` and ``config.knobs`` carriers, with source_prompt_id linking that source.
query_lab reads the three as business entities (:func:`rubiksql_hb.lab_store.query_typed`): the
current release's exact selection by default, an explicit release, a role/route tool selection, or
every stored version; a release published before the entity-model migration has its pinned
carriers converted for the read, never written. Releases and proposals remain unchanged
canonical rows; query_lab computes convenience lifecycle fields at query time. Every
evolution-loop iteration and loop decision is an ``agent-session`` of the proposer (``agent_ref`` ``evolver`` or
``evolver-decision``, with its transcript's ``agent-message`` rows when the loop kept one). The experiment row names
the knowledge root its run read and that root's ledgers (``knowledge_root``, ``knowledge_build``,
:func:`rubiksql_hb.lab_ingest.record_knowledge`).

Navigation — every pair of the five evaluation entities is ONE hop:

* Point lookups are O(1) computed-key ``get``s (~1ms measured): trace and judge SHARE
  one ``object_id`` (both ``hash_id(exp_id, qid)``); candidate ids derive as
  ``hash_id(exp_id, qid, "c1".."cN")`` with N = ``trace.n_candidates``; every
  per-question row reaches its experiment with ``get(exp_id)`` and its exact,
  versioned QA through ``Trace.qa_id``. ``qid`` remains the public question number.
* Set lookups are one filtered ``query`` on ``exp_id`` or ``qid``; experiment<->qa
  reads the trace table as the relation. Unindexed scans are fine at the current
  scale; if the lab grows ~100x, add ``.index()`` to ``exp_id``/``qid`` — never
  id-list fields (derivable state that drifts).

Generation direction: candidates and the judge are the primary records; the trace is
their aggregate plus the run's outcome. ``n_candidates``/``tool_calls``/``n_messages``
sum over the children (repaired runs leave orphaned transcripts that break the
equality), while ``scores``/``reasons``/``pred_sql``/``error`` come from the eval
harness ONLY — judge selection is unrecorded, and fallback/reexec/reviewer SQL can
equal no candidate's. Batch ingest writes everything in one pass from the artifacts;
live writers go children first, trace last, outcome fields from the final
submission/scoring step.

Two cross-run comparison hazards the fields themselves record:

* **Telemetry is per-run-vintage.** ``usage``/``duration_sec``/``n_turns`` are ``None``
  for every run whose artifacts predate their capture, so an average over mixed
  vintages must filter on presence rather than treat ``None`` as zero.
* **Replays cost nothing and say so.** An agent whose result came from the LLM cache
  made no API call, so its telemetry is ``None`` and its keywords carry ``from_cache``.
  A run built mostly from cache therefore reports far less token spend than a cold run
  of the same arm — correctly, because it spent less. Compare cost only across runs of
  the same cache warmth, and use the facet to tell "replayed" from "never recorded".
* **Some old rows lost their telemetry.** Runs saved before 2026-07-31 let the reviewer's
  record overwrite the judge's whenever it rewrote a submission, so for those questions
  the judge's tool calls, tokens and transcripts no longer exist anywhere. Their trace and
  judge rows carry the ``telemetry_lost`` facet: counts on them are floors, not totals.
* **Tool counts span both protocols.** ``tool_calls`` merges provider-native calls with
  plain-text SQL actions, the protocol the SQL verbs moved to on 2026-07-29. Counting one
  list alone would report a run that executed queries and submitted an answer as never
  having touched SQL.
* **Tool vocabularies drift.** Tool names are stored exactly as the run emitted them,
  including ones the code no longer serves (``get_rule_context``, ``expand_synonyms``).
  ``ExperimentEntity.tool_vocab``/``legacy_tools`` make the drift queryable: compare two
  runs' vocabularies before comparing their per-tool counts, because a tool missing from
  one run's counts may mean "never called" or "did not exist yet".

Storage is sqlite-only: no field here carries an embedding, so the workspace mounts no
vector backend. Added fields must use ``.optional()`` when older stored rows have no
value; defaulted new fields can cause ``SchemaVersionMismatchError`` on historical reads.
Even optional fields change complete authority export schemas and row bytes. Extend an
existing JSON payload or migrate the authority export before adding canonical evidence fields.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import heavenbase as hb

if TYPE_CHECKING:
    from heavenbase.workspace import WorkspaceSpec
    from .prompts.types import AuthorityBinding, GitAuthorityBinding

from .prompts.control_schema import GeneralRuleEntity, KnobSetEntity, ProposalEntity, ReleaseEntity, ToolEntity

#: BIRD's official difficulty labels, in ascending order.
LAB_DIFFICULTIES = ("simple", "moderate", "challenging")


def sql_length(sql: str | None = None) -> int:
    """Return the character length of one SQL string.

    Args:
        sql: SQL text. ``None`` counts as empty.

    Returns:
        int: ``len(sql)``, or 0 when the value is empty.
    """
    return len(sql or "")


class QAEntity(hb.Entity):
    """One question version and its sole shared gold-result snapshot.

    ``object_id`` derives from question-file version, raw database version and
    public ``name``. Experiments link through ``Trace.qa_id``.
    """
    name       = hb.field(hb.ShortText,  desc="Public question id, e.g. Q00371").required()
    dataset_version = hb.field(hb.ShortText, desc="SHA-256 content identity of the question file").required()
    database_version = hb.field(hb.ShortText, desc="SHA-256 content identity of the raw SQLite database").required()
    db_id      = hb.field(hb.ShortText,  desc="Database the question targets").required()
    question   = hb.field(hb.MediumText, desc="Natural language question").required()
    hint       = hb.field(hb.MediumText, desc="BIRD evidence text, empty when none").default("")
    gold_sql   = hb.field(hb.LongText,   desc="Official gold SQL").required()
    difficulty = hb.field(hb.Categorical[LAB_DIFFICULTIES],
                          desc="Official BIRD label: simple, moderate or challenging").optional()
    tags       = hb.field(hb.Array[hb.ShortText],
                          desc="Analysis facets, e.g. has_groupby or dirty_gold").default([])
    sql_len    = hb.field(hb.Integer, desc="Character length of gold_sql") \
                    .compute(sql_length, inputs=["gold_sql"])
    gold_rows  = hb.field(hb.Integer, desc="Row count of one gold execution, null until probed").optional()
    # Boolean is three-valued in HeavenBase; None on these two means "not yet probed".
    empty_result      = hb.field(hb.Boolean, desc="Gold SQL returns zero rows, null until probed").optional()
    consistent_result = hb.field(hb.Boolean, desc="Repeated gold executions agree, null until probed").optional()
    gold_execution_result = hb.field(hb.Json,
        desc="Shared bounded gold SQL execution snapshot with exact source, time, database version and full-result digests; null until measured").optional()


class ExperimentEntity(hb.Entity):
    """One eval run: its launch parameters and aggregate outcome.

    Its object_id is the ``exp_id`` every trace/candidate/judge row carries — query
    them by ``exp_id`` to enumerate the run.
    """
    name        = hb.field(hb.ShortText, desc="Run label, e.g. official_sim_dap_20260727; used as id seed").required()
    ts          = hb.field(hb.Timestamp, desc="Epoch-ms time of the last artifact write: run end, or the latest repair").optional()
    code_commit = hb.field(hb.ShortText, desc="Git SHA the run was built from").default("")
    prompt_fp   = hb.field(hb.ShortText, desc="Prompt fingerprint printed at launch").default("")
    queries     = hb.field(hb.ShortText, desc="Question file the run answered").default("")
    n_questions = hb.field(hb.Integer,   desc="Questions answered").default(0)
    parameters  = hb.field(hb.Json,      desc="Launch parameters: CLI args plus RUBIKSQL_* env flags").default({})
    # avg_, not ave_: the eval summaries these aggregates mirror spell it avg_ex / avg_bf.
    avg_scores  = hb.field(hb.Json,      desc="Mean per-metric scores over traces, e.g. {'ex': 0.77}. "
                                              "Rows written before the 2026-08 metric rename spell the "
                                              "primary metric 'exd' (and may carry the retired 'exu'); "
                                              "new writes use 'ex' only — read both keys when spanning "
                                              "that boundary, never rewrite stored rows").default({})
    avg_tool_calls = hb.field(hb.Json,   desc="Mean tool-call counts per trace, per tool plus 'total'").default({})
    sum_tool_calls = hb.field(hb.Json,   desc="Total tool-call counts over the run, per tool plus 'total'").default({})
    # Token and vocabulary aggregates, .optional() because they postdate the first rows.
    # None means the run predates them; {} / [] would claim the run measured zero.
    avg_usage   = hb.field(hb.Json,      desc="Mean token counters per trace, per counter plus "
                                              "'total'; null for runs recorded before usage "
                                              "was persisted").optional()
    sum_usage   = hb.field(hb.Json,      desc="Total token counters over the run, per counter "
                                              "plus 'total'; null when unrecorded").optional()
    tool_vocab  = hb.field(hb.Array[hb.ShortText],
                           desc="Tool names this run actually called, sorted — compare two "
                                "runs' vocabularies BEFORE comparing their tool counts").optional()
    legacy_tools = hb.field(hb.Array[hb.ShortText],
                            desc="Subset of tool_vocab the current code no longer serves; a "
                                 "non-empty value means per-tool counts are NOT comparable "
                                 "with a newer run's").optional()
    # Exact release attribution comes from a verified per-run authority snapshot.
    prompt_release = hb.field(hb.ShortText,
                              desc="Exact release id every row was answered under (r<16hex>); "
                                   "'mixed' when verified source releases differ — see "
                                   "prompt_releases for the split").optional()
    prompt_releases = hb.field(hb.Json,
                               desc="Distribution of per-row prompt_release values, release id to "
                                    "row count ('unrecorded' for rows without the field)").optional()
    prompt_proposal = hb.field(hb.ShortText,
                               desc="Proposal id (p<16hex>) when the pin was a proposal, from the "
                                    "run manifest; empty when the pin was a release").optional()
    prompt_keyspace = hb.field(hb.ShortText,
                               desc="Cache keyspace label recorded by the pinned run manifest").optional()
    authority_revision = hb.field(hb.Json,
                                  desc="Authority event count and last event id recorded by the verified run snapshot; "
                                       "execution evidence outside scientific run identity").optional()
    cache_ns    = hb.field(hb.ShortText,
                           desc="RUBIKSQL_CACHE_NS sampling universe of the run, from the manifest; "
                                "empty for the default universe").optional()
    run_identity = hb.field(hb.Json,
                            desc="One opaque recorded identity: a nonempty .pipeline_state.json run_identity "
                                 "takes precedence; otherwise this run's .prompt_manifest.json identity, the pairing "
                                 "identity (evolution question_set, model configuration and environment knob "
                                 "overrides). In a non-diagnostic manifest identity, judge records the judge_model knob "
                                 "default (qwen:qwen3.7-max) whenever the judge is the pinned release's own, as "
                                 "reviewer_enabled records the reviewer default; read the actual judge from the "
                                 "release's judge_model (prompt_release, then its release or knob_set row) or from each "
                                 "judge row's model. A diagnostic or pipeline-state identity names the judge as run. "
                                 "Empty or invalid identities are absent. The schemas are never merged; null if neither "
                                 "is available. provenance.run_identity_source names the selected file and its "
                                 "SHA256.").optional()
    # Where the raw result JSON stays, and what the evolution loop knows about the run. All
    # .optional(): runs ingested before these fields existed have none, and a run the loop did
    # not launch has no iteration, parent experiment or verdict. The run's release and proposal
    # are prompt_release and prompt_proposal above.
    result_files = hb.field(hb.Json,
                            desc="The raw result JSON the rows came from, which stays where the "
                                 "run wrote it: {'dir': absolute run directory, 'files': {path "
                                 "relative to dir: sha256 hex}}").optional()
    provenance = hb.field(hb.Json,
                          desc="Source records for this ingest: result/manifest/cache file paths and SHA256; "
                               "run_identity_source identifies the exact sidecar bytes selected for run_identity, "
                               "plus the exact per-session record and message digests in agent-session.state.provenance. "
                               "transcripts says where the run's conversations are kept: lab (agent-session rows) or "
                               "run_directory (recorded without transcripts). "
                               "A missing optional source is recorded as unavailable, never reconstructed.").optional()
    analysis_coverage = hb.field(hb.Json,
                                 desc="Transcript completeness by role: complete, partial and missing session counts, "
                                      "with missing reasons. Complete describes captured messages, not a successful answer. "
                                      "failure_attempts separately counts observed evaluation failures and unmeasured older "
                                      "results; it is not an HTTP-call or billing count. "
                                      "Null means an older ingest did not measure coverage.").optional()
    iteration   = hb.field(hb.Integer,
                           desc="Analysis iteration that launched the run; null for a run the "
                                "analysis did not launch").optional()
    parent_experiment = hb.field(hb.ShortText,
                                 desc="Experiment id of the run this one is paired with (a child's "
                                      "control); null for a run evaluated on its own").optional()
    verdict     = hb.field(hb.ShortText,
                           desc="Promotion verdict of the outcome that paired this run as a child: "
                                "promotable, adoptable, not-promotable or descriptive; null until an outcome "
                                "exists").optional()
    gain_pp     = hb.field(hb.Float,
                           desc="EX gain of that outcome, child minus parent, in percentage points; "
                                "null until an outcome exists").optional()
    # Aggregates of this run's rows, recomputed from the lab by lab_ingest.refresh_experiment_scores
    # (after ingest, and after candidate scoring fills CandidateEntity.scores).
    per_db_scores = hb.field(hb.Json,
                             desc="Per database: {db_id: {'questions', 'correct', 'accuracy'}} over "
                                  "the scored traces (accuracy is a fraction)").optional()
    per_difficulty_scores = hb.field(hb.Json,
                                     desc="Per BIRD difficulty of the QA row ('unknown' without "
                                          "one): {'questions', 'correct', 'accuracy'}").optional()
    per_slot_scores = hb.field(hb.Json,
                               desc="Per candidate slot (c1..cN): {'questions' (candidate runs), "
                                    "'scored', 'correct', 'accuracy' (the candidate's own EX, null "
                                    "until candidates are scored), 'picked' (how often the judge "
                                    "picked the slot), 'picked_correct', 'picked_accuracy'}").optional()
    # What the run cost and the knowledge it read. All .optional(): runs ingested before these fields existed have
    # none, and a root is recorded only by lab_ingest.record_knowledge.
    rate_limits = hb.field(hb.Json,
                           desc="The provider's answers during the run, per lane (chat, judge, embedding, ...): "
                                "{requests, rate_limited (HTTP 429), overloaded (5xx, timeout or lost connection)}, "
                                "summed over every record in the run's .rate_limits/ (every launch of a resumed run); "
                                "null when the run kept no records").optional()
    usage_by_role = hb.field(hb.Json,
                             desc="Token counters per agent role over the run's sessions, the reviewer included: "
                                  "{role: {sessions, measured (the sessions that recorded usage), tokens: {counter: "
                                  "total}}}; tokens only, because the repository keeps no price table; null when no "
                                  "session recorded usage").optional()
    cost = hb.field(hb.Json,
                    desc="Recorded monetary cost only: amount/currency when every session has a compatible recorded "
                         "amount, otherwise null with unknown_reason; totals_by_currency retain known subtotals, "
                         "recorded/missing count coverage. No inferred price table or unknown-as-zero.").optional()
    knowledge_root = hb.field(hb.Json,
                              desc="The knowledge root the run read: {path, status (known or unknown), id (k<16 hex> "
                                   "only for completed observed ledgers; equal known ids mean equal recorded knowledge), "
                                   "databases, unledgered (workspaces without a ledger), unknown (reasons identity cannot "
                                   "be established)}; null until recorded").optional()
    knowledge_build = hb.field(hb.Json,
                               desc="That root's ledgers when the run was evaluated, and its last build: {stages: {db: "
                                    "{stage: {status, version, release, output_digest, input_fingerprint, started_at, "
                                    "finished_at, action, reasons, inputs_changed, missing}}}, record, mode, passes, "
                                    "elapsed_sec, rate_limits, unfinished, source, warnings (unavailable optional build "
                                    "or source records)}; null until recorded").optional()
    summary     = hb.field(hb.LongText,  desc="Prose conclusions for this run").default("")


class TraceEntity(hb.Entity):
    """One question's answering trace: aggregate of its candidate/judge rows + final outcome.

    Aggregates (``n_candidates``/``tool_calls``/``n_messages``) sum over the children;
    ``scores``/``reasons``/``pred_sql``/``error`` are eval-harness outcomes no child
    carries. Its judge shares this row's object_id (direct ``get``); candidate ids are
    ``hash_id(exp_id, qid, "c1".."cN")``.
    """
    # Computed id makes agent-driven MCP upserts idempotent: re-writing the same
    # (experiment, question) pair overwrites the row instead of duplicating it.
    object_id  = hb.field(hb.Identifier, desc="Stable id derived from exp_id and qid") \
                    .compute(hb.utils.hash_id, inputs=["exp_id", "qid"])
    exp_id     = hb.field(hb.Identifier[ExperimentEntity],
                          desc="Experiment this trace belongs to; equals that row's object_id").required()
    qid        = hb.field(hb.Identifier,
                          desc="Public question id, e.g. Q00371; part of this trace's stable key").required()
    qa_id      = hb.field(hb.Identifier[QAEntity],
                          desc="Exact versioned QA this experiment answered").required()
    db_id      = hb.field(hb.ShortText, desc="Database the question targets").required()
    keywords   = hb.field(hb.Array[hb.ShortText],
                          desc="Trace facets, e.g. has_groupby or has_run_sql_base").default([])
    tool_calls = hb.field(hb.Json, desc="Recorded tool-call counts over candidates, judge, reviewer and prior attempts, name to count").default({})
    n_messages = hb.field(hb.Integer,
                          desc="Captured messages over every candidate, judge, reviewer and prior attempt, "
                               "including orphan candidate transcripts from repaired artifacts. A lower bound "
                               "when session coverage is partial or missing; older ingests counted only candidates.").default(0)
    n_candidates = hb.field(hb.Integer, desc="Candidate runs under this trace").default(0)
    scores     = hb.field(hb.Json, desc="Final per-metric scores of the submitted SQL, e.g. "
                                        "{'ex': 1}; a metric is ABSENT when unmeasured "
                                        "(no_gold, score_errors) — never a placeholder 0. Rows "
                                        "written before the 2026-08 metric rename spell the primary "
                                        "metric 'exd'; new writes use 'ex' only — read both keys "
                                        "when spanning that boundary").default({})
    reasons    = hb.field(hb.Array[hb.ShortText],
                          desc="Eval diagnosis tags, e.g. correct or no_result").default([])
    pred_sql   = hb.field(hb.LongText, desc="Final submitted SQL").default("")
    error      = hb.field(hb.MediumText, desc="Pipeline error when the trace failed, empty otherwise").default("")
    result_attempts = hb.field(hb.Json,
                                desc="Compact evaluation failure diagnostics under failures, including original errors, "
                                     "tracebacks, observed wait boundaries, coverage and raw record pointers. "
                                     "unrecorded_results names raw rows predating failure capture; these are unknown, "
                                     "not zero failures. No inferred agent role, HTTP call or duplicate transcript. "
                                     "gold_execution retains this run's reference identity, CPU budget, timings, "
                                     "result digest and ok/gold_unscorable status when recorded. "
                                     "Null means an older ingest did not project this history.").optional()
    # Cost telemetry, .optional() because runs recorded before it existed have none.
    usage      = hb.field(hb.Json, desc="Token counters summed over candidates and judge, e.g. "
                                        "{'prompt_tokens': ...}; null when the run predates "
                                        "usage capture").optional()
    duration_sec = hb.field(hb.Float, desc="Wall seconds for the whole question, retries "
                                           "included; null when unrecorded").optional()
    reviewer   = hb.field(hb.Json, desc="The reviewer's record: active, changed, skipped "
                                        "(reason) or error, plus its prompt_release and "
                                        "prompt_replay; null when the row carries no "
                                        "reviewer_debug (reviewer off, or a pre-reviewer run)").optional()
    # The run's original observation remains distinct from later scoring snapshots.
    question   = hb.field(hb.MediumText, desc="The question as the run asked it, the hint "
                                              "included; null when unrecorded").optional()
    gold_sql   = hb.field(hb.LongText, desc="The gold SQL the run scored against; null when "
                                            "unrecorded (compare QAEntity.gold_sql)").optional()
    gold_rows  = hb.field(hb.Integer, desc="Rows the gold SQL returned in the run; null when "
                                           "unrecorded").optional()
    pred_rows  = hb.field(hb.Integer, desc="Rows the submitted SQL returned in the run; null "
                                           "when unrecorded").optional()
    gold_preview = hb.field(hb.Json, desc="The first rows of the gold result (at most 50, as "
                                          "eval_bird keeps them); null when unrecorded").optional()
    pred_execution_result = hb.field(hb.Json,
        desc="Bounded snapshot of the final pred_sql after reviewer changes or fallback; null until measured").optional()
    pred_preview = hb.field(hb.Json, desc="The first rows of the submitted result (at most 50); "
                                          "null when unrecorded").optional()
    session_ids = hb.field(hb.Array[hb.ShortText],
                           desc="Every agent-session of the question, in order: the candidates "
                                "by slot, then the judge, then the reviewer; null for rows "
                                "ingested before sessions existed or recorded without transcripts "
                                "(the run directory keeps them)").optional()


class CandidateEntity(hb.Entity):
    """One candidate agent run inside one trace.

    Reach its trace or judge via ``hash_id(exp_id, qid)``, its experiment via
    ``get(exp_id)``, its exact question via ``Trace.qa_id``.
    """
    object_id  = hb.field(hb.Identifier, desc="Stable id derived from exp_id, qid and slot") \
                    .compute(hb.utils.hash_id, inputs=["exp_id", "qid", "slot"])
    exp_id     = hb.field(hb.Identifier[ExperimentEntity],
                          desc="Experiment this run belongs to; equals that row's object_id").required()
    qid        = hb.field(hb.Identifier,
                          desc="Public question id; resolve versioned QA through the trace").required()
    slot       = hb.field(hb.ShortText, desc="Candidate slot, c1..cN in launch order").required()
    db_id      = hb.field(hb.ShortText, desc="Database the question targets").required()
    model      = hb.field(hb.ShortText, desc="Provider or model route of this candidate").default("")
    sql        = hb.field(hb.LongText,  desc="This candidate's submitted SQL").default("")
    row_count  = hb.field(hb.Integer,   desc="Rows its SQL returned, null when it errored").optional()
    error      = hb.field(hb.MediumText, desc="Execution or pipeline error, empty otherwise").default("")
    tool_calls = hb.field(hb.Json, desc="Tool-call counts, name to count").default({})
    n_messages = hb.field(hb.Integer, desc="Messages in this candidate's transcript").default(0)
    keywords   = hb.field(hb.Array[hb.ShortText], desc="Run facets over this candidate's own SQL").default([])
    # .optional(), not .default({}): added after rows existed (the module docstring's
    # schema-evolution rule), and None is the meaning — this candidate's SQL was never
    # scored. Filled by counterfactual scoring passes via `set`, never by lab_ingest.
    scores     = hb.field(hb.Json, desc="Per-metric scores of this candidate's own SQL, "
                                        "null until a counterfactual pass measures it. Rows scored "
                                        "before the 2026-08 metric rename spell the primary metric "
                                        "'exd'; new writes use 'ex' only — read both keys when "
                                        "spanning that boundary").optional()
    execution_result = hb.field(
        hb.Json,
        desc="Candidate SQL result from a later scoring re-execution, not its original "
             "agent tool observation: source, timestamp, database and SQL hash, status, "
             "columns, up to 64 head/tail positional rows, original row indices, full row count and truncation flag. "
             "Its difference has full distinct-row counts and up to 64 extra/missing "
             "row samples under official EX semantics when gold ran.",
    ).optional()
    usage      = hb.field(hb.Json, desc="Token counters for this candidate's own run; null "
                                        "when the run predates usage capture").optional()
    duration_sec = hb.field(hb.Float, desc="Wall seconds this candidate spent; null when "
                                           "unrecorded").optional()
    n_turns    = hb.field(hb.Integer, desc="Model turns this candidate took, from its protocol "
                                           "events; null when unrecorded").optional()
    prompt_release = hb.field(hb.ShortText,
                              desc="Release id this candidate's texts came from, read from its "
                                   "prompt_identity snapshot; null when the run predates the "
                                   "snapshot or the answer replayed from a legacy cache entry").optional()
    prompt_replay = hb.field(hb.Json,
                             desc="Turn-replay record of the snapshot: turns, replayed, "
                                  "diverged_at, cause (initial | prompt:<entry> | tool | clock | "
                                  "unknown), source; null when unrecorded").optional()
    session_id = hb.field(hb.Identifier,
                          desc="The agent-session holding this candidate's conversation; null "
                               "for rows ingested before sessions existed or recorded without "
                               "transcripts (the run directory keeps them)").optional()


class JudgeEntity(hb.Entity):
    """The judge run inside one trace; absent for single-candidate experiments."""
    # Same id inputs as TraceEntity on purpose: the judge row shares its trace's id string,
    # so get(trace_id, JudgeEntity) reaches it without a query. Ids only need to be unique
    # within one entity type, and each type stores in its own table.
    object_id  = hb.field(hb.Identifier, desc="Stable id derived from exp_id and qid") \
                    .compute(hb.utils.hash_id, inputs=["exp_id", "qid"])
    exp_id     = hb.field(hb.Identifier[ExperimentEntity],
                          desc="Experiment this run belongs to; equals that row's object_id").required()
    qid        = hb.field(hb.Identifier,
                          desc="Public question id; resolve versioned QA through the trace").required()
    db_id      = hb.field(hb.ShortText, desc="Database the question targets").required()
    model      = hb.field(hb.ShortText, desc="Provider or model of the judge").default("")
    # ShortText, not Categorical: vote.py's mode vocabulary grows (complete_submit,
    # truncated, reexec_*, fallback_*) and a locked enum would reject the new labels —
    # which are exactly the diagnostically interesting rows.
    mode       = hb.field(hb.ShortText,
                          desc="How the judge ended: complete_submit, truncated, reexec_* "
                               "or fallback_*; empty when unknown").default("")
    fallback   = hb.field(hb.Boolean, desc="Fallback selection was used, null until derived").optional()
    picked     = hb.field(hb.ShortText,
                          desc="Candidate slot(s) whose SQL the judge chose, e.g. 'c2' or 'c1,c3' "
                               "when slots wrote the same SQL; empty when the judge chose no "
                               "candidate's SQL (its own, or a fallback) or the run predates the "
                               "field").default("")
    sql        = hb.field(hb.LongText,
                          desc="Judge's confirmed SQL choice before any reviewer rewrite; null if no reliable choice").optional()
    sql_source = hb.field(hb.Json,
                          desc="Auditable source of the judge's confirmed SQL; null if no reliable source").optional()
    execution_result = hb.field(hb.Json,
                          desc="Independent bounded snapshot of judge SQL, null until measured").optional()
    tool_calls = hb.field(hb.Json, desc="Tool-call counts, name to count").default({})
    n_messages = hb.field(hb.Integer,
                          desc="Messages in the judge transcript when the whole transcript is "
                               "known (its session's source is llm_episode or judge_messages). Null otherwise: the "
                               "eval harness persists the judge's tool calls and content but not "
                               "its transcript, so this is unknown rather than zero — do not read "
                               "a null as a judge that ran with no context").optional()
    keywords   = hb.field(hb.Array[hb.ShortText], desc="Run facets, e.g. judge_misselect").default([])
    scores     = hb.field(hb.Json, desc="Per-metric scores of the SQL this judge CHOSE: equals the "
                                        "trace scores when the judge submitted its own pick and the "
                                        "reviewer left it alone. Null — meaning unmeasured, not "
                                        "zero — whenever the judge did not choose the submission "
                                        "(fallback) or the reviewer rewrote it; such rows belong "
                                        "outside judge-quality averages, not in them as mistakes").optional()
    usage      = hb.field(hb.Json, desc="Token counters for the judge run; null when the run "
                                        "predates usage capture").optional()
    duration_sec = hb.field(hb.Float, desc="Wall seconds of the judge's LLM phase, excluding "
                                           "fallback re-execution; null when unrecorded").optional()
    n_turns    = hb.field(hb.Integer, desc="Model turns the judge took, from its protocol "
                                           "events; null when unrecorded").optional()
    prompt_release = hb.field(hb.ShortText,
                              desc="Release id the judge's texts came from, read from its "
                                   "prompt_identity snapshot; null when unrecorded or replayed "
                                   "from a legacy cache entry").optional()
    prompt_replay = hb.field(hb.Json,
                             desc="Turn-replay record of the judge's snapshot (see "
                                  "CandidateEntity.prompt_replay); null when unrecorded").optional()
    session_id = hb.field(hb.Identifier,
                          desc="The agent-session holding the judge's conversation; null for rows "
                               "ingested before sessions existed or recorded without transcripts "
                               "(the run directory keeps them)").optional()


#: Analytics entities, canonical releases/proposals and the typed Tool, rule and parameter-set versions.
#: Control.open registers them in the sole bound workspace. The transcripts and the
#: evolver sessions are HeavenBase's own ``agent-session`` / ``agent-message`` rows, registered by the ``agent``
#: extension (:func:`transcript_entities`), and the prompt versions its ``sys-prompt`` rows, registered by the
#: ``prompt`` extension every workspace loads.
LAB_ANALYTICS_ENTITIES: tuple[type, ...] = (QAEntity, ExperimentEntity, TraceEntity, CandidateEntity, JudgeEntity)
LAB_ENTITIES: tuple[type, ...] = (*LAB_ANALYTICS_ENTITIES, ReleaseEntity, ProposalEntity, ToolEntity, GeneralRuleEntity,
                                  KnobSetEntity)

#: Model-facing entity names for the lab query surface, mirroring
#: ``tools.knowledge.QUERY_ENTITIES`` for the knowledge surface; the transcript rows join
#: them through :func:`lab_query_entities`. ``sys-prompt`` stays off it on purpose: a whole prompt row carries its
#: executable capsule (about 100 KB for a role document), so its versions are read with the native ``query`` and a
#: ``select``. query_lab serves ``release`` and ``proposal`` with query-time history facts, and ``tool``,
#: ``general_rule`` and ``knob_set`` as business entities (:func:`rubiksql_hb.lab_store.query_typed`).
LAB_QUERY_ENTITIES: dict[str, type[hb.Entity]] = {
    "qa":         QAEntity,
    "experiment": ExperimentEntity,
    "trace":      TraceEntity,
    "candidate":  CandidateEntity,
    "judge":      JudgeEntity,
    "release":    ReleaseEntity,
    "proposal":   ProposalEntity,
    "tool":       ToolEntity,
    "general_rule": GeneralRuleEntity,
    "knob_set":   KnobSetEntity,
}


def transcript_entities(ws: hb.HeavenBase | None = None) -> dict[str, type[hb.Entity]]:
    """Return the agent extension's transcript entities by their lab query names.

    HeavenBase resolves ``heavenbase.extensions.agent``'s classes through the extension
    registry, which knows them once a workspace enabled the extension (or the machine home
    indexed it), so call this after :func:`open_lab_workspace`.

    Args:
        ws: Live workspace whose registered classes own query and mutation hooks.
            Omission returns extension classes for static schema or payload construction.

    Returns:
        dict[str, type[hb.Entity]]: ``{"session": agent-session, "message": agent-message}``.
    """
    if ws is not None:
        return {"session": ws.entities["agent-session"], "message": ws.entities["agent-message"]}
    from heavenbase.extensions.agent import Message, Session

    return {"session": Session, "message": Message}


def canonical_evidence_entities() -> dict[str, type[hb.Entity]]:
    """Return static native schemas for canonical evaluation and conversation rows.

    Returns:
        Entity identifiers to classes. Ordinary imports do not open a workspace,
        select a Context or change the machine's module registry.
    """
    from heavenbase.extensions.agent.message import Message
    from heavenbase.extensions.agent.session import Session

    return {"q-a-entity": QAEntity, "experiment-entity": ExperimentEntity,
            "trace-entity": TraceEntity, "candidate-entity": CandidateEntity,
            "judge-entity": JudgeEntity, "agent-session": Session, "agent-message": Message}


def lab_query_entities(ws: hb.HeavenBase | None = None) -> dict[str, type[hb.Entity]]:
    """Return the whole lab query surface: :data:`LAB_QUERY_ENTITIES` and the transcripts.

    Args:
        ws: Live workspace for runtime queries. Omission returns schema declarations.

    Returns:
        dict[str, type[hb.Entity]]: The lab entities of :data:`LAB_QUERY_ENTITIES`, then ``session``
        and ``message``; call after :func:`open_lab_workspace` (see :func:`transcript_entities`).
    """
    declared = {**LAB_QUERY_ENTITIES, **transcript_entities(ws)}
    return {alias: ws.entities[cls.schema().entity_id] for alias, cls in declared.items()} if ws is not None else declared


def lab_binding(db: str | os.PathLike[str] | None = None, *,
                binding: AuthorityBinding | GitAuthorityBinding | None = None) -> AuthorityBinding | GitAuthorityBinding:
    """Resolve one exact workspace selector against its ordinary binding.

    Args:
        db: Bound database or Git projection path assertion. Omission uses the package binding.
        binding: Explicit identity for a scratch test or restored authority.

    Returns:
        The unchanged selected binding.

    Raises:
        BindingError: The selector identifies another database. No environment,
            HOME or historical lab directory supplies an alternate authority.
    """
    from .prompts.binding import BindingError, load_binding
    from .prompts.types import GitAuthorityBinding

    selected = binding if binding is not None else load_binding()
    if isinstance(selected, GitAuthorityBinding):
        # Git supplies the authority. The opener checks an optional physical
        # cache assertion after resolving the requested immutable commit.
        return selected
    if db is not None:
        path = Path(db).expanduser().resolve()
        if path.suffix != ".db":
            path = path / selected.db_path.name
        if path != selected.db_path:
            raise BindingError(f"workspace selector {path} differs from bound database {selected.db_path}")
    return selected


def open_lab_workspace(db: str | os.PathLike[str] | None = None, *,
                       binding: AuthorityBinding | GitAuthorityBinding | None = None,
                       context: hb.Context | None = None,
                       create: bool = False, commit: str | None = None) -> hb.HeavenBase:
    """Open the lab adapter over the sole canonical control workspace.

    Args:
        db: Bound database or directory assertion; omission uses the package binding.
        binding: Explicit scratch or restored identity when supplied by the caller.
        context: HeavenBase machine context, or the process context.
        create: Allow explicit initialization. False binds existing storage read-only
            without schema publication, and refuses missing or incompatible tables.
        commit: Exact reachable Git authority commit. Omission captures the current remote head once.

    Returns:
        The canonical Control workspace, with all content and evidence schemas.

    Raises:
        BindingError: The explicit selector differs from the binding.
        PromptControlError: The database or established canonical storage is absent
            or incompatible when creation is disabled.
    """
    from .prompts.control import Control
    from .prompts.binding import BindingError
    from .prompts.types import GitAuthorityBinding

    selected = lab_binding(db, binding=binding)
    if isinstance(selected, GitAuthorityBinding):
        from .prompts.authority import Authority

        if create:
            raise BindingError("Git authority initialization and writes require an Authority.write transaction")
        control = Authority.project(binding=selected, context=context, commit=commit)
        ws = control.ws
        if db is not None:
            asserted = Path(db).expanduser().resolve()
            if asserted.suffix != ".db":
                asserted /= control.path.name
            if asserted != control.path.resolve():
                ws.close()
                raise BindingError("workspace selector differs from the pinned Git projection database")
        return ws
    if commit is not None:
        raise BindingError("an authority commit selector requires a Git binding")
    ws = Control.open(selected.db_path, context=context, create=create, read_only=not create).ws
    setattr(ws, "_rubiksql_authority_binding", selected)
    return ws


def register_lab_workspace(db: str | os.PathLike[str] | None = None, *,
                           binding: AuthorityBinding | GitAuthorityBinding | None = None,
                           context: hb.Context | None = None) -> WorkspaceSpec:
    """Bootstrap and register a bound read-only facade in the dashboard's Context.

    The dashboard's AI assistant then serves ``list_entities`` / ``entity_schema`` /
    ``sample_rows`` / ``column_stats`` / ``query`` / ``search_workspace`` over these
    entities with no RubikSQL-specific code. The manifest strips compute hooks for the
    same reason :func:`rubiksql_hb.workspace.register_workspace` does: a native dashboard
    process has not registered application callables such as :func:`sql_length`.
    Each serving process must call this before opening the native dashboard in the
    same Context. The saved runtime manifest cannot reconstruct the guarded backend
    in an unbootstrapped standalone ``hb dashboard`` process.

    Args:
        db: Bound database or directory assertion.
        binding: Explicit scratch or restored identity.
        context: HeavenBase machine context. Defaults to the process context.

    Returns:
        The durable :class:`heavenbase.workspace.WorkspaceSpec`.

    Raises:
        BindingError: The path or stored authority identity differs from the binding.
        PromptControlError: Existing storage or the opened backend path is incompatible.
    """
    from heavenbase.workspace import WorkspaceSpec
    from heavenbase.utils.files import file_uri_to_path
    from heavenbase.context import context_or_default
    from .prompts.authority import Authority
    from .prompts.binding import BindingError
    from .prompts.control import PromptControlError
    from .prompts.readonly import ReadOnlySQLiteBackend
    from .prompts.types import GitAuthorityBinding
    from .workspace import _registered_live_workspace, _registry_manifest

    selected = lab_binding(db, binding=binding)
    context = context_or_default(context)
    if not isinstance(selected, GitAuthorityBinding):
        with Authority.read(binding=selected) as reader:
            verified = reader.verify(require_export=False)
            if not verified.ok or not verified.bound:
                raise BindingError(f"dashboard authority verification refused: {verified.violations}")
    temporary = open_lab_workspace(db, binding=selected, context=context, create=False)
    projection_path = Path(temporary._rubiksql_authority_db).resolve()
    authority_commit = getattr(temporary, "_rubiksql_authority_commit", None)
    registered = None
    try:
        manifest = _registry_manifest(temporary)
        with _registered_live_workspace(temporary.context, manifest["id"]) as existing:
            if existing is not None:
                held = existing.backends.get("main")
                existing_path = (Path(getattr(existing, "_rubiksql_authority_db", "")).resolve()
                                 if isinstance(selected, GitAuthorityBinding) else projection_path)
                if (existing.id != manifest["id"] or not isinstance(held, ReadOnlySQLiteBackend)
                        or existing.context.authority_token != temporary.context.authority_token
                        or getattr(existing, "_rubiksql_authority_binding", None) != selected
                        or getattr(existing, "_rubiksql_authority_commit", None) != authority_commit
                        or held.db is None or Path(file_uri_to_path(held.db.spec.database)).resolve() != existing_path):
                    raise PromptControlError("existing dashboard facade differs from the guarded bound database")
                physical = held.db.execute("PRAGMA database_list", readonly=True).first() or {}
                if physical.get("name") != "main" or Path(physical.get("file", "")).resolve() != existing_path:
                    raise PromptControlError("existing dashboard facade differs from the guarded bound database")
                if isinstance(selected, GitAuthorityBinding):
                    return WorkspaceSpec.load(existing.id, context=existing.context)
            backend = ReadOnlySQLiteBackend("main", projection_path.as_uri(), ws_id=manifest["id"],
                                            resolver=temporary.module_resolver, config=temporary.context.config)
            registered = hb.HeavenBase.from_manifest(manifest, backends={"main": backend},
                                                    context=temporary.context, detached=False)
            backend.finish_binding()
            physical = backend.db.execute("PRAGMA database_list", readonly=True).first() or {}
            if (registered.backends.get("main") is not backend or physical.get("name") != "main"
                    or Path(physical.get("file", "")).resolve() != projection_path):
                raise PromptControlError("dashboard backend differs from the bound database")
            registered._rubiksql_authority_binding = selected
            registered._rubiksql_authority_db = str(projection_path)
            registered._rubiksql_lab_dir = str(projection_path.parent)
            if authority_commit is not None:
                registered._rubiksql_authority_commit = authority_commit
                registered._rubiksql_authority_carrier = temporary

                def close() -> None:
                    registered.release()
                    try:
                        backend.db.dispose()
                    finally:
                        temporary.close()

                registered.close = close
            return WorkspaceSpec.load(registered.id, context=registered.context)
    except BaseException:
        if registered is not None:
            registered.close() if isinstance(selected, GitAuthorityBinding) and hasattr(registered, "close") else registered.release()
        raise
    finally:
        if (isinstance(selected, GitAuthorityBinding)
                and (registered is None or getattr(registered, "_rubiksql_authority_carrier", None) is not temporary)):
            temporary.close()
        else:
            temporary.release()
