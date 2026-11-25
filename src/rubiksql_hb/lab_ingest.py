"""Backfill the lab workspace from official question files and saved eval runs.

Two ingestion paths, both idempotent because every lab entity derives a stable row id
from its natural key (see :mod:`rubiksql_hb.lab_schema`):

* :func:`ingest_qa` — the exact question file and recorded raw database content
  versions become immutable ``QAEntity`` rows. Another input version coexists.
* :func:`ingest_run` — one saved eval run directory (the parent of ``pass1/``, or the
  directory an evolution evaluation writes its per-database JSONs into) becomes one
  ``ExperimentEntity`` row plus per-question ``TraceEntity`` / ``CandidateEntity`` /
  ``JudgeEntity`` rows, aggregated from the per-database result JSONs that
  ``scripts/eval_bird.py`` writes, and one HeavenBase ``agent-session`` per agent run
  with its ordered ``agent-message`` rows (below).

Every row is written whole through ``upsert_many``: ``upsert`` REPLACES the entire row,
so this module never writes partial rows — later patches go through ``set``
(:func:`record_outcome`, :func:`refresh_experiment_scores`, the candidate scores of
:mod:`rubiksql_hb.lab_score`).

Telemetry (``usage``, ``duration_sec``, ``n_turns``) is read straight from the artifacts
and stays ``None`` for runs saved before the harness recorded it, so an older run reads
as unmeasured rather than free. Tool names are counted verbatim, including retired ones;
:data:`LEGACY_TOOLS` names them and every experiment row records the vocabulary it
actually used, because per-tool counts are only comparable within one vocabulary.

The prompt release (src/rubiksql_hb/prompts) is read the same way: every candidate, judge
and reviewer row takes its ``prompt_release`` / ``prompt_replay`` from its own
``prompt_identity`` snapshot, the experiment row resolves its release from the per-row
``prompt_release`` fields and the run's ``.prompt_manifest.json`` (``"mixed"`` when they
disagree, with the distribution in ``prompt_releases``), and the pipeline's
``run_identity`` keeps a nonempty identity from ``.pipeline_state.json`` first, otherwise the
same run's prompt-manifest ``identity``; their schemas are never merged. Its selected source's
path and byte digest are recorded in ``provenance.run_identity_source``. The console ``[prompt-release]``
banner is a cross-check only — the last one is compared and a disagreement is printed,
never written.

Transcripts: every candidate, the judge and (when it ran) the reviewer of a question get one
``agent-session`` with deterministic row ids — ``hash_id(exp_id, qid, role, slot)`` for the
session, ``hash_id(exp_id, qid, role, slot, idx)`` for its messages — so re-ingesting a run
rewrites the same rows. A candidate's messages are its ``candidates_debug[].messages``. The
judge's and reviewer's own ``messages`` records take precedence, with ``reasoning`` restored
per assistant turn. Cache-only records use the ``llm_episode`` table of the run's LLM cache
(``cache_db``): the episode of the run's namespace whose system message and
first user message hash to the digests the agent's record names (``system_hash``, its
``user_msg`` or ``prompt_identity.msg``), preferring one written under the agent's own
release; an exact ``episode_ref`` also verifies the cache key and message digest. Without
that reference a digest-only match is explicitly partial because it cannot prove the attempt.
Each assistant turn carries its reasoning, which the table keeps apart
(``think_z``), as its ``reasoning_content``. Without one the session keeps what the result
row holds — the first user message and the final reply with its reasoning, with the
recorded tool calls and SQL actions in its ``state`` — and
``state.source`` says which of ``candidates_debug``, ``llm_episode``, ``judge_debug``,
``reviewer_debug`` or ``none`` the messages came from. A message's ``content_text`` and a
session's ``summary`` hold at most :data:`CATALOG_DESC_MAX` characters, what HeavenBase's
catalog projection of them holds: a longer text is cut with a closing note, and the
message's ``payload`` keeps every character. With ``transcripts=False`` (``meta_ops record``)
the sessions are only measured: no ``agent-session`` or ``agent-message`` row is written, the
candidate, judge and trace session links stay null and ``provenance.transcripts`` is
``run_directory``. The run directory keeps every conversation; ``result_files`` names its
result files, and a trace's ``qid`` is the ``id`` of its record there.

The raw result JSON and the LLM cache stay where they are: ``ExperimentEntity.result_files``
names the result files with their sha256, the digests an outcome pairs runs by. The
experiment row also carries what the evolution loop knows about the run (``iteration``,
``parent_experiment``, and, once an outcome paired it, ``verdict`` and ``gain_pp``) and the
aggregates the loop's proposer reads first (:func:`refresh_experiment_scores`). What the run
cost comes with it: the provider's 429 and 5xx/timeout answers per lane from the run's
``.rate_limits/`` records (``rate_limits``) and the token counters per agent role over its
sessions (``usage_by_role``). Explicit monetary records are kept in ``cost``; missing amounts
or currencies stay unknown, never zero, and no price table is inferred. The knowledge
root a run read is the loop's to name: :func:`record_knowledge` patches ``knowledge_root`` and
``knowledge_build`` from that root's ledgers, and a re-ingest keeps them.

The ``refresh`` command rebuilds only the GeneralRule and KnobSet derived views from
the bound authority. ``knowledge`` records a run's knowledge root.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
import sqlite3
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from copy import deepcopy
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable

import heavenbase as hb

from . import rate_limits
from .lab_identity import content_version, qa_for_trace, qa_gold_lock, qa_object_id, question_file
from .question import question_with_hint
from .prompts import PromptStoreError, load_release
from .prompts.store import Store
from .prompts.evolve import same_run_release
from .prompts import knowledge as K
from .prompts import metrics as M
from .prompts.canon import canon, sha256_hex
from .lab_schema import (
    LAB_DIFFICULTIES,
    CandidateEntity,
    ExperimentEntity,
    JudgeEntity,
    QAEntity,
    TraceEntity,
    transcript_entities,
)

#: SQL-shape facets, tag name to detector, applied to gold and predicted SQL alike.
#: Word-boundary regexes over the raw text — a mention inside a string literal can
#: mis-tag, which is acceptable noise for stopping-power queries.
SQL_TAG_PATTERNS: dict[str, re.Pattern[str]] = {
    "has_groupby":  re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE),
    "has_orderby":  re.compile(r"\bORDER\s+BY\b", re.IGNORECASE),
    "has_join":     re.compile(r"\bJOIN\b", re.IGNORECASE),
    "has_cte":      re.compile(r"\bWITH\b\s+\S+\s+AS\s*\(", re.IGNORECASE),
    "has_window":   re.compile(r"\bOVER\s*\(", re.IGNORECASE),
    "has_distinct": re.compile(r"\bDISTINCT\b", re.IGNORECASE),
    "has_limit":    re.compile(r"\bLIMIT\b", re.IGNORECASE),
    "has_subquery": re.compile(r"\(\s*SELECT\b", re.IGNORECASE),
}

#: Score record keys eval_bird writes, metric name to per-question field.
SCORE_FIELDS: dict[str, str] = {
    "ex": "ex_score",
    "sf1": "sf1_score", "sf2": "sf2_score", "bf": "bf_score",
}

#: Per-metric fallback fields for run directories written before the 2026-08 metric
#: rename, which spell the primary EX score ``exd_score`` (and its score_errors label
#: ``exd``). Ingest READS the old spelling but only ever WRITES the new one; the
#: retired ``exu`` variant is not ingested at all.
LEGACY_SCORE_FIELDS: dict[str, str] = {"ex": "exd_score"}

#: Tool names saved runs contain that the current code no longer serves, each with why
#: it disappeared. Counts for these tools exist only in older runs, so a per-tool
#: comparison across that boundary reads a removal as a behaviour change unless the
#: reader knows better — which is what ``ExperimentEntity.legacy_tools`` records.
#: Extend this when a tool is renamed or retired; nothing else needs to change.
LEGACY_TOOLS: dict[str, str] = {
    "get_rule_context": "superseded by get_rules; absent from the current tool surface",
    "expand_synonyms":  "dropped from the candidate surface 2026-07-26; still an MCP tool",
}

#: The whole line after the tag is the identity — eval_bird prints multiple components,
#: e.g. ``_SYSTEM=66d6739b _JUDGE_COMMON=303011e2 grain_gate=v6``.
_FINGERPRINT_RE = re.compile(r"\[prompt-fingerprint\]\s*(\S.*?)\s*$")
#: The ``[prompt-release]`` banner eval_bird prints under the fingerprint. Ingest reads the
#: LAST one in the logs and uses it as a cross-check only: the per-row ``prompt_release``
#: fields and the run's ``.prompt_manifest.json`` are the record. The tag must be followed
#: by ``release=`` — eval_bird prints requeue, stranded and store lines under the same tag,
#: and those carry no release field.
_RELEASE_BANNER_RE = re.compile(r"\[prompt-release\]\s*(release=\S.*?)\s*$")
#: The run manifest eval_bird writes (schema ``rubiksql.prompt-manifest/1``), in the eval
#: pass directory and mirrored by the pipeline into the directory that holds ``pass1/``.
PROMPT_MANIFEST_NAME = ".prompt_manifest.json"
#: Key of the distribution bucket for rows that carry no ``prompt_release`` field.
UNRECORDED_RELEASE = "unrecorded"
#: Default evolution input; historical ingestion selects its recorded content digest.
V0_QUESTIONS = Path(__file__).resolve().parents[2] / "resources" / "official_input" / "minidev_500_20240627.json"
#: Where a session's messages came from (``agent-session.state.source``).
SESSION_SOURCES = ("candidates_debug", "llm_episode", "judge_messages", "reviewer_messages", "judge_debug", "reviewer_debug", "none")
#: The longest ``content_text`` of an ``agent-message`` and ``summary`` of an ``agent-session``: HeavenBase projects both
#: into the catalog's ``desc``, a MediumText, and refuses a longer value. A longer text is cut to this length with a
#: closing note; the message's ``payload`` keeps every character, and a session's whole final SQL is its candidate's
#: ``sql`` or its trace's ``pred_sql``.
CATALOG_DESC_MAX = hb.MediumText.default_max_chars


def _lab_write(function: Callable[..., Any]) -> Callable[..., Any]:
    """Keep direct lab writer calls in the same reentrant ingestion batch."""
    @wraps(function)
    def locked(ws: hb.HeavenBase, *args: Any, **kwargs: Any) -> Any:
        from .lab_store import lab_write

        with lab_write(ws, operation=function.__name__) as writable:
            return function(writable, *args, **kwargs)
    return locked


def run_snapshot(run_dir: str | os.PathLike[str], manifest: Mapping[str, Any] | None = None) -> Any:
    """Verify a run's recorded snapshot without consulting the authority.

    Args:
        run_dir: Run directory holding its manifest or pass1 prompt manifest.
        manifest: Already-read prompt manifest, so its provenance stays byte exact.

    Returns:
        VerifiedSnapshot with the exact recorded release and digest.

    Raises:
        PromptStoreError: Snapshot evidence is missing, incomplete or disagrees.
        SnapshotError: The recorded snapshot or package binding is invalid.
    """
    from .prompts.snapshot import verify_snapshot

    root = Path(run_dir).expanduser().resolve()
    prompt = manifest if manifest is not None else _read_prompt_manifest(str(root))[0]
    record = prompt.get("prompt_snapshot")
    if record is None:
        body, _source = _read_json_source(root / "manifest.json")
        record = (body or {}).get("prompt_snapshot")
    if not isinstance(record, Mapping) or any(not isinstance(record.get(key), str) or not record[key] for key in ("path", "release", "digest")):
        raise PromptStoreError(f"{root}: run lacks exact prompt_snapshot path, release and digest")
    source = Path(record["path"])
    # A child kit preserves the original manifest bytes while carrying its own
    # content-addressed snapshot. Prefer that verified local copy even when the
    # original host's absolute path still exists, so imported runs are independent.
    local = root / ".prompt_snapshot" / record["digest"]
    if re.fullmatch(r"[0-9a-f]{64}", record["digest"]) and local.exists():
        source = local
    elif not source.is_absolute():
        source = root / source
    return verify_snapshot(source, expected_release=record["release"], expected_digest=record["digest"],
                           expected_authority_commit=record.get("authority_commit"))


def derive_sql_tags(sql: str | None = None) -> list[str]:
    """Return the SQL-shape facet tags present in one SQL string.

    Args:
        sql: SQL text. ``None`` counts as empty.

    Returns:
        list[str]: Matching :data:`SQL_TAG_PATTERNS` names, in declaration order.
    """
    text = sql or ""
    return [tag for tag, pattern in SQL_TAG_PATTERNS.items() if pattern.search(text)]


def qid_str(question_id: int | str) -> str:
    """Return the canonical question id string for one official ``question_id``.

    Args:
        question_id: Official integer id, or an already-formatted ``Q00042`` string.

    Returns:
        str: ``Q%05d`` — the internal id format eval artifacts use throughout.
    """
    text = str(question_id)
    if text.startswith("Q"):
        return text
    return f"Q{int(text):05d}"


@_lab_write
def ingest_qa(ws: hb.HeavenBase, questions_path: str | os.PathLike[str], *,
              database_versions: Mapping[str, str], expected_dataset_version: str | None = None) -> int:
    """Register an exact question file under its raw database content versions.

    Existing versioned rows are checked and left untouched, including measured
    gold snapshots and probe fields. A changed definition needs new input bytes.

    Args:
        ws: Open lab workspace.
        questions_path: Official question JSON file actually used for this run.
        database_versions: Recorded content SHA-256 for every database in the file.
        expected_dataset_version: Optional version already verified from a run,
            preventing a file change between binding and registration.

    Returns:
        int: Newly registered QA rows.

    Raises:
        ValueError: A missing or conflicting input identity or duplicate question.
    """
    records, dataset_version, _source = question_file(questions_path)
    if expected_dataset_version is not None and dataset_version != content_version(expected_dataset_version):
        raise ValueError(f"question file {questions_path} changed since run binding")
    versions = {str(db): content_version(version) for db, version in database_versions.items()}
    definitions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in sorted(records, key=lambda item: qid_str(item["question_id"])):
        qid = qid_str(rec["question_id"])
        if qid in seen:
            raise ValueError(f"question file {questions_path} repeats {qid}")
        seen.add(qid)
        db_id = str(rec["db_id"])
        if db_id not in versions:
            raise ValueError(f"question {qid} database {db_id} has no content fingerprint")
        qa_id = qa_object_id(dataset_version, versions[db_id], qid)
        difficulty = rec.get("difficulty")
        row = {
            "object_id": qa_id,
            "name":      qid,
            "dataset_version": dataset_version,
            "database_version": versions[db_id],
            "db_id":     db_id,
            "question":  rec["question"],
            "hint":      rec.get("evidence") or "",
            "gold_sql":  rec["SQL"],
            "difficulty": difficulty if difficulty in LAB_DIFFICULTIES else None,
            "tags":      derive_sql_tags(rec["SQL"]),
        }
        definitions.append(row)
    lab_dir = getattr(ws, "_rubiksql_lab_dir", None)
    if not lab_dir:
        raise ValueError("QA registration requires a workspace opened by open_lab_workspace")
    # Hold every affected QA lock through the batch insert. Scoring takes the
    # same per-QA lock before setting gold; a second concurrent ingest therefore
    # sees the completed row instead of replacing its measured snapshot.
    with ExitStack() as locks:
        for row in sorted(definitions, key=lambda item: item["object_id"]):
            locks.enter_context(qa_gold_lock(lab_dir, row["object_id"]))
        rows: list[dict[str, Any]] = []
        for row in definitions:
            qa_id = row["object_id"]
            before = ws.get(qa_id, QAEntity)
            if before is not None:
                saved = before if isinstance(before, dict) else before.to_dict()
                if any(saved.get(key) != value for key, value in row.items()):
                    raise ValueError(f"existing QA {qa_id} conflicts with question file {questions_path}")
                continue
            rows.append(row)
        if rows:
            ws.upsert_many(QAEntity, rows, catalog_mode="merge")
    return len(rows)


def _tool_counts(calls: list[dict[str, Any]] | None = None) -> dict[str, int]:
    """Return per-tool call counts for one recorded tool-call list.

    Args:
        calls: Ordered ``{"name": ..., ...}`` records, or ``None``.

    Returns:
        dict[str, int]: Tool name to occurrence count; unnamed entries count as ``""``.
    """
    counts: dict[str, int] = {}
    for call in calls or []:
        name = str(call.get("name") or "")
        counts[name] = counts.get(name, 0) + 1
    return counts


def _agent_tool_counts(debug: dict[str, Any] | None = None) -> dict[str, int]:
    """Return one agent's tool usage across BOTH invocation protocols.

    A tool reaches the harness either as a provider-native tool call (recorded in
    ``tool_calls``) or as a plain-text SQL action (recorded in ``sql_actions``, the
    protocol the SQL verbs moved to on 2026-07-29). Which protocol carried a call is a
    transport detail; counting only the native list would report a run that executed
    eleven queries and submitted an answer as having touched no SQL at all.

    Args:
        debug: One agent's debug record, or ``None``.

    Returns:
        dict[str, int]: Tool name to total occurrences over both lists.
    """
    record = debug or {}
    counts = _tool_counts(record.get("tool_calls"))
    return _merge_counts(counts, _tool_counts(record.get("sql_actions")))


def _merge_counts(target: dict[str, int], extra: dict[str, int]) -> dict[str, int]:
    """Add ``extra``'s per-tool counts into ``target`` and return it."""
    for name, count in extra.items():
        target[name] = target.get(name, 0) + count
    return target


def merge_usage(target: dict[str, Any], extra: Any) -> dict[str, Any]:
    """Add one agent's numeric token counters into ``target`` and return it.

    Nested numeric breakdowns are summed recursively, including provider cache read/write counters.
    Non-numeric metadata is dropped; monetary values remain outside token totals.

    Args:
        target: Running totals, mutated in place.
        extra: One agent's recorded ``usage`` mapping, or any non-mapping to ignore.

    Returns:
        dict[str, Any]: ``target``.
    """
    if not isinstance(extra, dict):
        return target
    for key, value in extra.items():
        if key in ("cost", "amount"):
            continue  # monetary values belong to cost, never to token sums
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            target[key] = target.get(key, 0) + value
        elif isinstance(value, dict):
            nested = merge_usage(target[key] if isinstance(target.get(key), dict) else {}, value)
            if nested:
                target[key] = nested
    return target


def _turn_count(events: Any) -> int | None:
    """Return how many model turns one recorded ``protocol_events`` list describes.

    Args:
        events: The agent's ``protocol_events``, or any non-list to report as unknown.

    Returns:
        int | None: Highest recorded turn number, or ``None`` when the run predates
        protocol events (older artifacts drop them for candidates).
    """
    if not isinstance(events, list) or not events:
        return None
    turns = [event.get("turn") for event in events if isinstance(event, dict)]
    numbers = [int(turn) for turn in turns if isinstance(turn, int)]
    return max(numbers) if numbers else len(events)


def _protocol_tags(events: Any) -> list[str]:
    """Return facet tags for the non-clean protocol statuses one agent hit.

    A run whose turns all parsed cleanly gets no tag; anything else is worth being
    able to query for, so each distinct irregular status becomes ``proto_<status>``.

    Args:
        events: The agent's ``protocol_events``, or any non-list for no tags.

    Returns:
        list[str]: Sorted distinct tags, e.g. ``["proto_incomplete"]``.
    """
    if not isinstance(events, list):
        return []
    clean = {"ok", "native"}
    statuses = {str(event.get("status") or "") for event in events
                if isinstance(event, dict)}
    return sorted(f"proto_{status}" for status in statuses
                  if status and status not in clean)


def _find_upward(run_dir: str, filename: str, levels: int = 3) -> str | None:
    """Return the closest ``filename`` in ``run_dir`` or up to ``levels`` parents."""
    current = os.path.abspath(run_dir)
    for _ in range(levels + 1):
        candidate = os.path.join(current, filename)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def _run_log_sources(run_dir: str, levels: int) -> list[list[Path]]:
    """Group this run's console paths by proximity, without scanning unrelated parents."""
    run = Path(run_dir).resolve()
    sources = [sorted(run.glob("*.log"))]
    if run.name == "pass1" and levels > 0:
        run, levels = run.parent, levels - 1
        sources.append(sorted(run.glob("*.log")))
    if levels >= 1:
        sources.append([run.with_name(run.name + ".log")])
    if levels >= 2 and run.parent.name == "runs":
        sources.append([run.parent.parent / (run.name + ".log")])
    return sources


def _prompt_fingerprint(run_dir: str, levels: int = 3) -> str:
    """Return the first fingerprint in this run's owned console log, or ``""``.

    Uses the same scoped paths as :func:`_prompt_release_banner`. Multiple logs at
    the nearest supported location must agree on their first recorded fingerprint.
    """
    for paths in _run_log_sources(run_dir, levels):
        fingerprints: set[str] = set()
        for log_path in paths:
            try:
                with open(log_path, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        match = _FINGERPRINT_RE.search(line)
                        if match:
                            fingerprints.add(match.group(1))
                            break
            except OSError:
                continue
        if fingerprints:
            return next(iter(fingerprints))[:255] if len(fingerprints) == 1 else ""  # ShortText limit
    return ""


def _prompt_release_banner(run_dir: str, levels: int = 3) -> dict[str, str]:
    """Return the last banner from console logs belonging to this run.

    A resumed log prints one banner per launch. Logs inside the run directory belong
    to it; outside it, only the exact ``<run>.log`` path is accepted. The older
    ``<root>/runs/<arm>`` layout keeps its named log at ``<root>/<arm>.log``.
    Unrelated sibling logs and conflicting logs never supply a guessed banner.

    Args:
        run_dir: Run directory or its ``pass1`` result directory.
        levels: Maximum parent distance for the supported named-log layouts.

    Returns:
        dict[str, str]: ``k=v`` fields of the last banner, e.g. ``{"release": "r…"}``;
        empty when no owned log carries one or their last banners disagree.
    """
    for paths in _run_log_sources(run_dir, levels):
        banners: list[dict[str, str]] = []
        for log_path in paths:
            last = ""
            try:
                with open(log_path, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        match = _RELEASE_BANNER_RE.search(line)
                        if match:
                            last = match.group(1)
            except OSError:
                continue
            if last:
                banners.append(dict(token.split("=", 1) for token in last.split() if "=" in token))
        if banners:
            return banners[0] if all(banner == banners[0] for banner in banners) else {}
    return {}


def _read_json_source(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Read one JSON mapping and evidence from the same bytes; invalid optional files return neither."""
    source = path.resolve()
    try:
        raw = source.read_bytes()
        loaded = json.loads(raw)
    except (OSError, ValueError):
        return None, None
    if not isinstance(loaded, dict):
        return None, None
    return loaded, {"path": str(source), "available": True, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _read_prompt_manifest(run_dir: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return the run's ``.prompt_manifest.json``, from ``run_dir`` or its ``pass1/``.

    Args:
        run_dir: Run directory (the parent of ``pass1/``).

    Returns:
        tuple[dict[str, Any], dict[str, Any] | None]: The manifest and evidence from the same bytes,
            or ``({}, None)`` when absent or unreadable. The run root takes precedence over ``pass1/``.
    """
    for directory in (run_dir, os.path.join(run_dir, "pass1")):
        loaded, source = _read_json_source(Path(directory) / PROMPT_MANIFEST_NAME)
        if loaded is not None:
            return loaded, source
    return {}, None


def _read_run_identity(run_dir: str, manifest: Mapping[str, Any],
                       manifest_source: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Select one nonempty recorded identity and the evidence for its exact source bytes.

    The run root's pipeline identity wins. Missing, malformed, non-mapping or empty pipeline
    identities fall back to the already-read manifest identity. Values remain opaque dictionaries:
    the two schemas are never combined, and no field is inferred from another run or its parent.

    Args:
        run_dir: Run directory.
        manifest: This run's selected prompt manifest.
        manifest_source: Evidence captured when that manifest was read.

    Returns:
        tuple[dict[str, Any] | None, dict[str, Any] | None]: The selected identity and its source;
            both ``None`` when neither contains a nonempty identity mapping.
    """
    state, source = _read_json_source(Path(run_dir) / ".pipeline_state.json")
    identity = state.get("run_identity") if isinstance(state, dict) else None
    if isinstance(identity, dict) and identity:
        return dict(identity), source
    identity = manifest.get("identity")
    if isinstance(identity, dict) and identity:
        return dict(identity), manifest_source
    return None, None


def recorded_run_inputs(run_dir: str | os.PathLike[str]) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Read the run's verified question and database identities.

    Evolution runs write ``.evaluation_inputs.json`` after checking the database
    bytes again at the end. Official pipeline runs carry the map in their own
    pipeline state. Neither source is inferred from a current path or mtime.

    Args:
        run_dir: Directory holding this run's saved results.

    Returns:
        tuple: Dataset version, normalized database versions, source evidence.

    Raises:
        ValueError: The record is absent, incomplete, or unverified.
    """
    root = Path(run_dir)
    sidecar_path = root / ".evaluation_inputs.json"
    sidecar, source = _read_json_source(sidecar_path)
    if sidecar is None and sidecar_path.exists():
        raise ValueError(f"run {root} has an unreadable evaluation input identity")
    if sidecar is not None:
        if sidecar.get("schema") != "rubiksql.evaluation-inputs/1" or sidecar.get("verified_after") is not True:
            raise ValueError(f"{root}/.evaluation_inputs.json has no verified final input identity")
        question_set = sidecar.get("question_set")
        fingerprints = sidecar.get("db_fingerprints")
    else:
        state, source = _read_json_source(root / ".pipeline_state.json")
        if state is None:
            raise ValueError(f"run {root} has no recorded question and database fingerprints")
        identity = state.get("run_identity")
        question_set = identity.get("question_set") if isinstance(identity, Mapping) else None
        fingerprints = state.get("db_fingerprints")
    if not isinstance(fingerprints, Mapping) or not fingerprints:
        raise ValueError(f"run {root} has no recorded database fingerprints")
    try:
        dataset_version = content_version(question_set)
        versions = {str(db): content_version(value) for db, value in fingerprints.items()}
    except ValueError as exc:
        raise ValueError(f"run {root} has invalid input identity: {exc}") from exc
    return dataset_version, versions, source or {}


def _prompt_identity_fields(holder: dict[str, Any] | None = None) -> tuple[str | None, dict | None]:
    """Return ``(prompt_release, prompt_replay)`` from one agent record's snapshot.

    A record without a ``prompt_identity`` snapshot predates release recording; one whose
    snapshot marks ``provenance: legacy-cache`` replayed a cache entry written before it —
    both read as unrecorded (``None``), never as the current release.

    Args:
        holder: A candidate/judge/reviewer debug record, or ``None``.

    Returns:
        tuple[str | None, dict | None]: The release id and the replay record, each
        ``None`` when unrecorded.
    """
    identity = (holder or {}).get("prompt_identity")
    if not isinstance(identity, dict):
        own = (holder or {}).get("route_debug")
        identity = own.get("prompt_identity") if isinstance(own, Mapping) else None
    if not isinstance(identity, dict):
        return None, None
    release = identity.get("release")
    replay = identity.get("replay")
    return (str(release) if release else None,
            dict(replay) if isinstance(replay, dict) else None)


def _reviewer_record(debug: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return the trace's reviewer record from the row's ``reviewer_debug``.

    Args:
        debug: The row's ``reviewer_debug`` mapping, or ``None``/empty when absent.

    Returns:
        dict[str, Any] | None: ``active``, ``changed``, ``skipped`` (reason or ``None``),
        ``error`` (message or ``None``), ``prompt_release`` and ``prompt_replay``; ``None``
        when the row carries no reviewer record.
    """
    if not debug:
        return None
    release, replay = _prompt_identity_fields(debug)
    return {
        "active":  bool(debug.get("active")),
        "changed": bool(debug.get("changed")),
        "skipped": str(debug["skipped"]) if debug.get("skipped") else None,
        "error":   str(debug["error"]) if debug.get("error") else None,
        "prompt_release": release,
        "prompt_replay": replay,
    }


@lru_cache(maxsize=256)
def _same_release(a: str, b: str, *, stores: Store | None = None) -> bool:
    """Compare exact scientific releases, including their answering bundle."""
    if a == b:
        return True
    return same_run_release(load_release(a, stores=stores, allow_retired=True), b, store=stores)


def _experiment_release(recs: Sequence[dict[str, Any]],
                        manifest: dict[str, Any], *, stores: Store | None = None) -> tuple[str | None, dict[str, int] | None]:
    """Resolve the experiment's release from its rows, with the manifest as the declared value.

    Args:
        recs: Every per-question eval record of the run.
        manifest: The run's prompt manifest (``{}`` when absent).

    Returns:
        tuple[str | None, dict[str, int] | None]: ``(prompt_release, prompt_releases)`` —
        the one release every row was answered under (and the manifest declares, when it
        declares one); ``"mixed"`` when rows
        disagree with each other or with the manifest, or the manifest says the launch
        accepted a mixed directory; the manifest's release when no row records one;
        ``None`` when neither does. Releases with the same prompts and knowledge (a refactor-only
        release beside its parent) are one release: the declared id
        when it is one of them, else the first. The distribution counts rows per release id,
        ``"unrecorded"`` for rows without the field, or is ``None`` for a run with no rows.
    """
    counts: Counter[str] = Counter()
    for rec in recs:
        value = rec.get("prompt_release")
        counts[str(value) if value else UNRECORDED_RELEASE] += 1
    recorded = {key for key in counts if key != UNRECORDED_RELEASE}
    if UNRECORDED_RELEASE in counts:
        raise PromptStoreError("every result row must record an exact release; unrecorded attribution is removed")
    if stores is not None:
        for rid in recorded:
            load_release(rid, stores=stores, allow_retired=True)
    declared = manifest.get("release") if isinstance(manifest.get("release"), str) else None
    # Matching prompts and knowledge are one release for the lab, exactly as
    # eval_bird's launch gate counts their rows as done.
    classes: list[str] = []
    for rid in sorted(recorded):
        if not any(_same_release(rid, seen, stores=stores) for seen in classes):
            classes.append(rid)
    declared_matches = bool(declared) and any(_same_release(declared, seen, stores=stores) for seen in classes)
    if manifest.get("mixed") is True or len(classes) > 1 \
            or (len(classes) == 1 and declared and not declared_matches):
        release: str | None = "mixed"
    elif classes:
        release = declared if declared_matches else classes[0]
    else:
        release = declared
    return release, (dict(counts) or None)


def _pair_debug(candidates: list[dict], cdebug: list[dict]) -> list[dict]:
    """Return the debug transcript describing each candidate, by index or by SQL.

    Repaired runs keep every candidate summary but only the reissued transcript, so a
    positional zip attaches the surviving transcript to the wrong SQL — 54/1534 records
    in the shipped dap run, e.g. Q00001's lone transcript belongs to slot 4, not slot 1.
    Equal lengths pair by index (the writer emits both lists in lockstep); otherwise
    each transcript claims the first unclaimed candidate with identical ``sql``, and
    candidates left without a transcript pair with ``{}``.

    Args:
        candidates: ``candidates`` summaries of one eval record.
        cdebug: ``candidates_debug`` transcripts of the same record.

    Returns:
        list[dict]: One (possibly empty) debug mapping per candidate, index-aligned.
    """
    if len(cdebug) == len(candidates):
        return [debug or {} for debug in cdebug]
    paired: list[dict] = [{} for _ in candidates]
    for debug in cdebug:
        sql = (debug or {}).get("sql")
        for i, cand in enumerate(candidates):
            if not paired[i] and cand.get("sql") == sql:
                paired[i] = debug or {}
                break
    return paired


def _digest16(text: str) -> str:
    """First 16 hex of a text's sha256: how ``system_hash`` and ``prompt_identity.msg`` name a message."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _first_text(messages: Sequence[Any], role: str) -> str | None:
    """Text of the first message of ``role`` (``agent-message``'s ``content_text``), or ``None`` when there is none."""
    wrap = transcript_entities()["message"].wrap
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == role:
            return wrap(message).content_text
    return None


def _episode_ref(record: Mapping[str, Any], identity_holder: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    """The digests that name a judge's or reviewer's conversation, and the release it ran under.

    Args:
        record: The agent's own record: ``system_hash`` and, for a judge (or a reviewer that
            rewrote the answer), the first user message ``user_msg``, whose own digest wins.
        identity_holder: The record carrying its ``prompt_identity`` snapshot, whose ``msg``
            holds the digests of the system and first user message as sent.

    Returns:
        tuple[str | None, str | None, str | None]: ``(system digest, first-user digest,
        release)``, each ``None`` when unrecorded.
    """
    identity = record.get("prompt_identity") or identity_holder.get("prompt_identity")
    identity = identity if isinstance(identity, Mapping) else {}
    msg = identity.get("msg") if isinstance(identity.get("msg"), Mapping) else {}
    system = record.get("system_hash") or identity_holder.get("system_hash") or msg.get("system")
    user = _digest16(str(record["user_msg"])) if record.get("user_msg") else msg.get("user")
    release = identity.get("release")
    return (str(system) if system else None, str(user) if user else None, str(release) if release else None)


def _with_reasoning(messages: list[Any], thinks: Sequence[Any]) -> list[Any]:
    """An episode's messages with each assistant turn's reasoning in it, as a candidate's transcript carries its own.

    ``llm_episode`` keeps the reasoning apart from the messages (``think_z``: one text per assistant message, in order);
    the k-th non-empty text becomes the k-th assistant message's ``reasoning_content``.
    """
    out: list[Any] = []
    turn = 0
    for message in messages:
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            think = thinks[turn] if turn < len(thinks) else ""
            turn += 1
            if think and not message.get("reasoning_content"):
                message = {**message, "reasoning_content": str(think)}
        out.append(message)
    return out


def _file_evidence(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Identify one source by its bytes; an absent optional source stays explicitly absent."""
    source = Path(path).resolve()
    try:
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    except FileNotFoundError:
        return {"path": str(source), "available": False, "sha256": None}
    return {"path": str(source), "available": True, "sha256": digest, "bytes": source.stat().st_size}


def _recorded_cost(record: Mapping[str, Any], usage: Any) -> dict[str, Any]:
    """Keep an explicit monetary record, never infer a currency or token price."""
    raw, source = record.get("cost"), "artifact.cost"
    if raw is None and isinstance(usage, Mapping):
        raw, source = usage.get("cost"), "artifact.usage.cost"
    if raw is None:
        return {"amount": None, "currency": None, "source": None,
                "unknown_reason": "provider_cost_unrecorded"}
    amount = raw.get("amount") if isinstance(raw, Mapping) else raw
    currency = raw.get("currency") if isinstance(raw, Mapping) else None
    valid_amount = isinstance(amount, (int, float)) and not isinstance(amount, bool) and math.isfinite(amount) and amount >= 0
    valid_currency = isinstance(currency, str) and bool(currency.strip())
    return {"amount": amount if valid_amount else None,
            "currency": currency.strip() if valid_currency else None, "source": source,
            "unknown_reason": None if valid_amount and valid_currency else "cost_amount_or_currency_unrecorded",
            "record": dict(raw) if isinstance(raw, Mapping) else raw}


def _transcript_coverage(messages: Sequence[Any], source: str) -> dict[str, Any]:
    """Distinguish recorded full transcripts from debug excerpts and absent conversations."""
    if not messages:
        return {"status": "missing", "missing": ["transcript_unrecorded"]}
    if source not in ("candidates_debug", "llm_episode", "judge_messages", "reviewer_messages"):
        return {"status": "partial", "missing": ["full_transcript_unrecorded"]}
    roles = {str(item.get("role")) for item in messages if isinstance(item, Mapping)}
    missing = [f"{role}_message_unrecorded" for role in ("system", "user", "assistant") if role not in roles]
    calls = {str(call["id"]) for item in messages if isinstance(item, Mapping)
             for call in item.get("tool_calls") or [] if isinstance(call, Mapping) and call.get("id")}
    results = {str(item["tool_call_id"]) for item in messages if isinstance(item, Mapping) and item.get("tool_call_id")}
    missing.extend(f"tool_result_unrecorded:{call}" for call in sorted(calls - results))
    return {"status": "partial" if missing else "complete", "missing": missing}


class _Episodes:
    """The judges' and reviewers' whole conversations of one run, from the LLM cache's ``llm_episode`` table.

    The table (:mod:`rubiksql_hb.turn_memo`) keys an episode by its cache key, which a lab
    reader cannot recompute; an episode is found instead by the digests of its system
    message and first user message, which every judge and reviewer record names, inside the
    run's namespace. Each assistant turn carries its reasoning (``think_z``) as its
    ``reasoning_content``. Only episodes whose first user message some record of the run
    names are kept in memory. Read-only and best effort: a missing, unreadable or table-less
    cache finds nothing, and the sessions fall back to what the result rows hold; an
    unreadable ``think_z`` leaves the turns without their reasoning.

    Args:
        cache_db: The LLM cache, or ``None``.
        ns: The run's cache namespace (the manifest's ``ns``; ``""`` by default).
        users: First-user-message digests the run's records name.
    """

    def __init__(self, cache_db: str | os.PathLike[str] | None, ns: str, users: set[str]) -> None:
        self._by_user: dict[str, list[tuple[int, str, str, list[Any]]]] = {}
        self._sources: dict[int, dict[str, Any]] = {}
        if cache_db is None or not users or not os.path.isfile(cache_db):
            return
        try:
            conn = sqlite3.connect(Path(cache_db).resolve().as_uri() + "?mode=ro", uri=True)
        except sqlite3.Error:
            return
        try:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'llm_episode'").fetchone() is None:
                return
            for rowid, key, blob, think_blob, release in conn.execute("SELECT rowid, key, messages_z, think_z, release FROM llm_episode WHERE ns = ? ORDER BY rowid", (ns,)):
                try:
                    messages = json.loads(zlib.decompress(blob).decode("utf-8"))
                except (zlib.error, ValueError, UnicodeDecodeError):
                    continue
                user = _first_text(messages, "user") if isinstance(messages, list) else None
                if user is None or _digest16(user) not in users:
                    continue
                try:
                    thinks = json.loads(zlib.decompress(think_blob).decode("utf-8")) if think_blob else []
                except (zlib.error, ValueError, UnicodeDecodeError, TypeError):
                    thinks = []
                system = _first_text(messages, "system")
                messages = _with_reasoning(messages, thinks if isinstance(thinks, list) else [])
                self._sources[id(messages)] = {"path": str(Path(cache_db).resolve()), "table": "llm_episode",
                                               "key": str(key), "namespace": ns, "release": str(release or ""),
                                               "messages_sha256": sha256_hex(canon(messages))}
                self._by_user.setdefault(_digest16(user), []).append((int(rowid), _digest16(system) if system is not None else "", str(release or ""),
                                                                      messages))
        except sqlite3.Error:
            return
        finally:
            conn.close()

    def get(self, system: str | None, user: str | None, release: str | None,
            reference: Mapping[str, Any] | None = None) -> list[Any] | None:
        """The conversation whose first user message (and system message, when named) the digests name.

        Args:
            system: The system message's digest, or ``None`` to match on the first user message alone.
            user: The first user message's digest.
            release: The release the agent ran under; another release cannot stand in.
            reference: Exact key/namespace/release/messages_sha256 from the result. A key
                can be overwritten by another attempt, so its content digest must match too.

        Returns:
            list[Any] | None: The messages, or ``None`` when no episode matches.
        """
        matches = [item for item in self._by_user.get(user or "", []) if not system or item[1] == system]
        if not matches:
            return None
        own = [item for item in matches if not release or item[2] == release]
        if reference is not None:
            if not isinstance(reference, Mapping):
                return None
            own = [item for item in own if all(reference.get(key) == self._sources[id(item[3])].get(key)
                                              for key in ("key", "namespace", "release", "messages_sha256"))]
        for item in own:
            self._sources[id(item[3])]["identity_verified"] = reference is not None
        return max(own, key=lambda item: item[0])[3] if own else None

    def provenance(self, messages: list[Any]) -> dict[str, Any] | None:
        """The exact cache row whose decoded messages were selected, with their content digest."""
        source = self._sources.get(id(messages))
        return dict(source) if source is not None else None


def _recorded_turns(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The judge's or reviewer's first user message and final reply, as its result-row record keeps them."""
    messages: list[dict[str, Any]] = []
    if record.get("user_msg"):
        messages.append({"role": "user", "content": str(record["user_msg"])})
    if record.get("raw_content") or record.get("reasoning_content"):
        reply: dict[str, Any] = {"role": "assistant", "content": str(record.get("raw_content") or "")}
        if record.get("reasoning_content"):
            reply["reasoning_content"] = str(record["reasoning_content"])
        messages.append(reply)
    return messages


def cut_text(text: str, limit: int, rest: str) -> str:
    """Return ``text`` whole when it fits in ``limit`` characters, else cut to fit with a closing note that says where the rest is.

    Args:
        text (str): The text, e.g. a message's ``content_text`` or a session's ``summary``.
        limit (int): The most characters the result may hold (:data:`CATALOG_DESC_MAX` for a catalog-projected field).
        rest (str): Where the whole text is, for the closing note.

    Returns:
        str: ``text``, or its head followed by ``…[cut to <limit> of <n> characters; <rest>]``, ``limit`` characters in all.
    """
    if len(text) <= limit:
        return text
    note = f" …[cut to {limit} of {len(text)} characters; {rest}]"
    return text[: limit - len(note)] + note


def _session(exp_id: str, qid: str, db_id: str, role: str, slot: str, messages: Sequence[Any], *,
             source: str, identity_holder: Mapping[str, Any], model: str, usage: Any,
             summary: str, status: str, extra: Mapping[str, Any] | None = None) -> tuple[dict, list[dict]]:
    """Build one ``agent-session`` row and its ``agent-message`` rows, with deterministic ids.

    Args:
        exp_id: Experiment id.
        qid: Question id.
        db_id: Database id.
        role: ``candidate``, ``judge`` or ``reviewer`` (the session's ``agent_ref``).
        slot: ``c1``..``cN`` for a candidate, ``""`` otherwise.
        messages: The conversation, in order, each message exactly as sent or received.
        source: One of :data:`SESSION_SOURCES`.
        identity_holder: The record carrying the agent's ``prompt_identity`` snapshot.
        model: Model label of the agent.
        usage: The agent's usage record; numeric counters are kept.
        summary: One-line summary, the agent's final SQL; cut to :data:`CATALOG_DESC_MAX` characters.
        status: ``done``, or ``error`` for a candidate or a reviewer that failed.
        extra: More ``state`` fields (the recorded tool calls of a partial record).

    Returns:
        tuple[dict, list[dict]]: The session row and its message rows. A message's ``content_text`` is cut to
        :data:`CATALOG_DESC_MAX` characters; ``payload`` stays whole.
    """
    session_id = hb.utils.hash_id(exp_id, qid, role, slot)
    wrap = transcript_entities()["message"].wrap
    rows = []
    for idx, message in enumerate(messages):
        # HeavenBase's own projection (role, content_text, tool_call_count), under this
        # module's deterministic id instead of wrap's content hash.
        row = wrap(dict(message) if isinstance(message, Mapping) else {"content": message}, session_id=session_id, idx=idx).to_dict()
        rows.append({**row, "object_id": hb.utils.hash_id(exp_id, qid, role, slot, idx),
                     "content_text": cut_text(row["content_text"], CATALOG_DESC_MAX, "the whole message is payload")})
    release, replay = _prompt_identity_fields(dict(identity_holder))
    coverage = _transcript_coverage(messages, source)
    if identity_holder.get("transcript_status") == "interrupted":
        coverage = {"status": "partial", "missing": [*coverage["missing"], "agent_attempt_interrupted"]}
    if source == "llm_episode" and not ((extra or {}).get("episode") or {}).get("identity_verified"):
        coverage = {"status": "partial", "missing": [*coverage["missing"], "exact_attempt_identity_unrecorded"]}
    session = {
        "object_id": session_id,
        "title": " ".join(part for part in (exp_id, qid, role, slot) if part),
        "summary": cut_text(summary, CATALOG_DESC_MAX, "the whole SQL is the candidate's sql or the trace's pred_sql"),
        "status": status,
        "agent_ref": role,
        "message_ids": [row["object_id"] for row in rows],
        "turn_count": sum(1 for row in rows if row["role"] == "user"),
        "tool_call_count": sum(row["tool_call_count"] for row in rows),
        "state": {"exp_id": exp_id, "qid": qid, "db_id": db_id, "role": role, "slot": slot, "release": release,
                  "replay": replay, "source": source, "model": model,
                  "cache_source": identity_holder.get("cache_source"), "execution": identity_holder.get("execution"),
                  "coverage": coverage, "cost": _recorded_cost(identity_holder, usage),
                  "context": {key: identity_holder[key] for key in ("pool_guard_fallback", "transcript_status", "capture_reason", "last_checkpoint_reason", "recording_error", "cache_source_error", "error") if key in identity_holder},
                  "messages_sha256": sha256_hex(canon(list(messages))),
                  "message_sha256": [sha256_hex(canon(message)) for message in messages], **dict(extra or {})},
        "usage": merge_usage({}, usage),
    }
    return session, rows


def _judge_sql(rec: Mapping[str, Any]) -> str:
    """The SQL the judge chose: the reviewer's ``original_sql`` when the reviewer ran, else the submitted SQL."""
    reviewer = rec.get("reviewer_debug") if isinstance(rec.get("reviewer_debug"), Mapping) else {}
    if reviewer.get("active") and reviewer.get("original_sql") is not None:
        return str(reviewer.get("original_sql") or "")
    return str(rec.get("pred_sql") or "")


def _judge_choice(rec: Mapping[str, Any], *, session_id: str) -> tuple[str | None, dict[str, Any] | None]:
    """Return an auditable judge choice before reviewer edits, if one survives."""
    debug = rec.get("judge_debug") if isinstance(rec.get("judge_debug"), Mapping) else {}
    reviewer = rec.get("reviewer_debug") if isinstance(rec.get("reviewer_debug"), Mapping) else {}
    mode = str(debug.get("mode") or "")
    if mode.startswith("fallback") or debug.get("transcript_status") in ("not_run", "interrupted"):
        return None, None
    expected = (str(reviewer["original_sql"]) if reviewer.get("active") and reviewer.get("original_sql") is not None
                else str(rec.get("pred_sql") or ""))
    actions = debug.get("sql_actions") if isinstance(debug.get("sql_actions"), list) else []
    for index in range(len(actions) - 1, -1, -1):
        action = actions[index]
        if (not isinstance(action, Mapping) or action.get("name") != "submit_final_sql"
                or action.get("status") != "ok" or action.get("source") == "fallback_candidate"):
            continue
        sql = action.get("sql")
        if isinstance(sql, str) and sql.strip() and sql == expected:
            return sql, {"kind": "sql_action", "session_id": session_id,
                         "record_pointer": f"/judge_debug/sql_actions/{index}", "status": "ok"}
    if reviewer.get("active") and isinstance(reviewer.get("original_sql"), str) and expected.strip():
        return expected, {"kind": "reviewer_original_sql", "session_id": session_id,
                          "record_pointer": "/reviewer_debug/original_sql"}
    return None, None


def _prior_attempts(exp_id: str, qid: str, db_id: str, role: str, slot: str, holder: Mapping[str, Any],
                    parent: str | None) -> tuple[list[dict], list[dict]]:
    """Keep failed attempts a later retry replaced as separate sessions of the same role."""
    sessions, rows = [], []
    for number, attempt in enumerate(holder.get("attempts") or [], 1):
        if not isinstance(attempt, Mapping):
            raise ValueError(f"{exp_id} {qid} {role}: attempt {number} is not a record")
        messages = _with_reasoning(attempt.get("messages") or [], attempt.get("reasoning") or [])
        source = ("candidates_debug" if role == "candidate" else f"{role}_messages") if messages else "none"
        session, message_rows = _session(
            exp_id, qid, db_id, role, f"{slot}.attempt{number}" if slot else f"attempt{number}", messages,
            source=source, identity_holder=attempt, model=str(attempt.get("model") or ""), usage=attempt.get("usage"),
            summary=str(attempt.get("error") or attempt.get("sql") or ""), status="error" if attempt.get("error") else "done",
            extra={"attempt": number, "prior_attempt": True, "parent_session_id": parent, "recorded_tool_calls": attempt.get("tool_calls") or [],
                   "recorded_sql_actions": attempt.get("sql_actions") or [], "error": attempt.get("error")})
        sessions.append(session)
        rows.extend(message_rows)
    return sessions, rows


def _picked(rec: Mapping[str, Any]) -> str:
    """Slots whose SQL the judge chose (``metrics.slot_stats``' rule on the judge's own SQL), e.g. ``c1,c3``."""
    stats = M.slot_stats([{**rec, "pred_sql": _judge_sql(rec)}])
    return ",".join(f"c{slot}" for slot, bucket in stats.items() if bucket["picked"])


def _failure_diagnostics(rec: Mapping[str, Any]) -> dict[str, Any]:
    """Project observed evaluation failures, keeping old result rows' unmeasured scope explicit."""
    failures, unrecorded = [], []
    fields = ("kind", "attempt", "scope", "stage", "source", "error", "error_type", "error_tb",
              "duration_sec", "failure_kind", "coverage")

    def visit(row: Mapping[str, Any], pointer: str) -> None:
        if row.get("failure_attempts_recorded") is not True:
            unrecorded.append(pointer or "/")
        for index, attempt in enumerate(row.get("result_attempts") or []):
            path = f"{pointer}/result_attempts/{index}"
            if not isinstance(attempt, Mapping):
                raise ValueError(f"result attempt {path} is not a record")
            if attempt.get("kind") == "failure":
                failures.append({**deepcopy({key: attempt[key] for key in fields if key in attempt}), "record_pointer": path})
            else:
                visit(attempt, path)

    visit(rec, "")
    diagnostics = {"failures": failures, "unrecorded_results": unrecorded}
    if rec.get("gold_execution") is not None:
        diagnostics["gold_execution"] = deepcopy(rec["gold_execution"])
    return diagnostics


def _trace_rows(
    exp_id: str,
    rec: dict[str, Any],
    models: Sequence[str] | None = None,
    episodes: _Episodes | None = None,
) -> tuple[dict, list[dict], dict | None, list[dict], list[dict]]:
    """Build the trace, candidate, judge, session and message rows for one per-question eval record.

    Args:
        exp_id: Experiment row id the rows belong to.
        rec: One entry of a ``pass1/<db>.json`` ``results`` list.
        models: Launch-order candidate labels for the ``model`` field, or ``None``.
            The artifacts themselves carry no per-candidate provider (their
            ``route_debug`` holds only style/timeout keys), so the caller supplies
            what the launch command knew.
        episodes: The run's judge and reviewer conversations from its LLM cache, or
            ``None`` to keep what the result row holds.

    Returns:
        ``(trace_row, candidate_rows, judge_row_or_None, session_rows, message_rows)``.
    """
    qid = str(rec.get("id") or "")
    db_id = str(rec.get("db_id") or "")
    pred_sql = rec.get("pred_sql") or ""
    reasons = [str(r) for r in rec.get("reasons") or []]
    # A no_gold record (hidden-test simulation) carries 0.0 in EVERY score field as a
    # placeholder, and score_errors names metrics whose computation crashed — recording
    # either as a real 0 would poison experiment averages. Unscored means absent.
    score_errors = {"ex" if str(metric) == "exd" else str(metric)
                    for metric in rec.get("score_errors") or []}
    if "no_gold" in reasons:
        scores: dict[str, Any] = {}
    else:
        scores = {}
        for metric, field in SCORE_FIELDS.items():
            value = rec.get(field)
            if value is None and metric in LEGACY_SCORE_FIELDS:
                value = rec.get(LEGACY_SCORE_FIELDS[metric])
            if value is not None and metric not in score_errors:
                scores[metric] = value

    candidates = rec.get("candidates") or []
    cdebug = rec.get("candidates_debug") or []
    paired = _pair_debug(candidates, cdebug)
    trace_calls: dict[str, int] = {}
    trace_usage: dict[str, Any] = {}
    n_messages = 0
    # Trace aggregates cover EVERY transcript, paired or orphaned: an orphan whose
    # candidate summary was discarded by a repair is still tool usage this trace spent.
    for debug in cdebug:
        _merge_counts(trace_calls, _agent_tool_counts(debug))
        merge_usage(trace_usage, (debug or {}).get("usage"))
        n_messages += len((debug or {}).get("messages") or [])
    candidate_rows = []
    session_rows: list[dict] = []
    message_rows: list[dict] = []
    for i, cand in enumerate(candidates):
        calls = _agent_tool_counts(paired[i])
        messages = _with_reasoning(paired[i].get("messages") or [], paired[i].get("reasoning") or [])
        # Normalized, not copied verbatim: `usage` means the same numeric counters on
        # every row that has it, so a candidate's is comparable with its trace's sum.
        usage = merge_usage({}, paired[i].get("usage"))
        model = models[i] if models and i < len(models) \
            else str((paired[i].get("route_debug") or {}).get("provider") or "")
        session, rows = _session(exp_id, qid, db_id, "candidate", f"c{i + 1}", messages,
                                 source="candidates_debug" if messages else "none",
                                 identity_holder=paired[i], model=model, usage=paired[i].get("usage"),
                                 summary=str(cand.get("sql") or ""),
                                 status="error" if cand.get("error") else "done")
        session_rows.append(session)
        message_rows.extend(rows)
        attempts, attempt_messages = _prior_attempts(exp_id, qid, db_id, "candidate", f"c{i + 1}", paired[i], session["object_id"])
        session_rows.extend(attempts)
        message_rows.extend(attempt_messages)
        candidate_rows.append({
            "exp_id":     exp_id,
            "qid":        qid,
            "slot":       f"c{i + 1}",
            "db_id":      db_id,
            "model":      model,
            "sql":        cand.get("sql") or "",
            # An errored candidate never executed anything: vote.py hardcodes 0 in the
            # failure record, which is indistinguishable from a query that genuinely
            # returned no rows. Store None — unmeasured — so `row_count = 0` keeps
            # meaning "empty result" and errored candidates stay findable by `error`.
            "row_count":  None if str(cand.get("error") or "") else cand.get("row_count"),
            "error":      str(cand.get("error") or ""),
            "tool_calls": calls,
            "n_messages": len(messages),
            "keywords":   derive_sql_tags(cand.get("sql"))
                          + _protocol_tags(paired[i].get("protocol_events"))
                          + (["from_cache"] if paired[i].get("from_cache") else []),
            "usage":      usage or None,
            "duration_sec": paired[i].get("duration_sec"),
            "n_turns":    _turn_count(paired[i].get("protocol_events")),
            # Which release rendered this candidate's texts, and how far it replayed —
            # from its own snapshot, never from the row (a legacy cache hit is None).
            "prompt_release": _prompt_identity_fields(paired[i])[0],
            "prompt_replay": _prompt_identity_fields(paired[i])[1],
            "session_id": session["object_id"],
        })

    # A repaired pool can be smaller than the failed pool (e.g. the pool-guard's
    # extra fallback slot). Its old-only slots own prior sessions, not current
    # CandidateEntity rows or SQL that candidate scoring should execute.
    for i, debug in enumerate(cdebug):
        if i >= len(candidates) and (debug or {}).get("transcript_status") == "not_run":
            attempts, attempt_messages = _prior_attempts(exp_id, qid, db_id, "candidate", f"c{i + 1}", debug, None)
            session_rows.extend(attempts)
            message_rows.extend(attempt_messages)

    judge_row = None
    jdebug = rec.get("judge_debug") or {}
    if jdebug and jdebug.get("transcript_status") != "not_run":
        # Runs saved before 2026-07-31 let the reviewer's record overwrite the judge's on
        # a rewrite, keeping only the judge's mode under this key. Its presence marks
        # that vintage: everything else in `judge_debug` there belongs to the reviewer
        # and is not recoverable, so it is dropped rather than misattributed.
        stale_mode = (rec.get("reviewer_debug") or {}).get("judge_debug_original_mode")
        overwritten = bool(stale_mode)
        judge_calls = {} if overwritten else _agent_tool_counts(jdebug)
        _merge_counts(trace_calls, judge_calls)
        mode = str(stale_mode or jdebug.get("mode") or "")
        # fallback is derivable only at the extremes: fallback_* means the fallback
        # selection ran, complete_submit means it did not; truncated/reexec_* leave
        # whether extraction fell back unrecorded, so None.
        fallback = True if mode.startswith("fallback") \
            else (False if mode == "complete_submit" else None)
        # The trace scores are the judge's score only when the judge actually chose the
        # submitted SQL. It did not when the fallback ran — vote.py then re-executes the
        # first non-failed candidate, whose SQL can score anything, including a 1 the
        # judge had no part in. Scoring that as the judge's would read as a correct
        # selection by a judge that produced nothing. A reviewer rewrite breaks the
        # identity the same way, and an unscored trace has nothing to transfer.
        # None here means UNMEASURED, never zero: these rows must leave the denominator
        # of judge-quality averages rather than count as judge mistakes.
        reviewer_changed = bool((rec.get("reviewer_debug") or {}).get("changed"))
        interrupted = bool(jdebug.get("error") or jdebug.get("transcript_status") == "interrupted"
                           or mode == "interrupted" or rec.get("pool_guard_fallback"))
        judge_scores = scores if scores and not reviewer_changed and not fallback and not interrupted else None
        judge_usage = {} if overwritten else merge_usage({}, jdebug.get("usage"))
        merge_usage(trace_usage, judge_usage)
        # The overwritten vintage's judge_debug describes the reviewer, so its snapshot
        # (if any) is the reviewer's and is not attributed to the judge.
        judge_release, judge_replay = (None, None) if overwritten else _prompt_identity_fields(jdebug)
        # The judge's whole conversation is in the run's LLM cache when its episode is
        # there; otherwise the session keeps what the row holds. The overwritten vintage
        # holds nothing of the judge's own.
        judge_messages: list[Any] = []
        judge_source = "none"
        judge_extra: dict[str, Any] = {}
        if not overwritten:
            episode = episodes.get(*_episode_ref(jdebug, jdebug), reference=jdebug.get("episode_ref")) if episodes is not None else None
            if isinstance(jdebug.get("messages"), list) and jdebug["messages"]:
                judge_messages, judge_source = _with_reasoning(jdebug["messages"], jdebug.get("reasoning") or []), "judge_messages"
            elif episode is not None:
                judge_messages, judge_source = episode, "llm_episode"
                judge_extra["episode"] = episodes.provenance(episode)
            else:
                judge_messages = _recorded_turns(jdebug)
                judge_source = "judge_debug" if judge_messages else "none"
                judge_extra = {"recorded_tool_calls": list(jdebug.get("tool_calls") or []),
                               "recorded_sql_actions": list(jdebug.get("sql_actions") or [])}
        session, rows = _session(
            exp_id, qid, db_id, "judge", "", judge_messages, source=judge_source,
            identity_holder={} if overwritten else jdebug,
            model="" if overwritten else str(jdebug.get("model") or ""),
            usage=judge_usage, summary="" if interrupted else _judge_sql(rec),
            status="error" if jdebug.get("error") or jdebug.get("transcript_status") == "interrupted" else "done",
            extra=judge_extra)
        session_rows.append(session)
        message_rows.extend(rows)
        attempts, attempt_messages = _prior_attempts(exp_id, qid, db_id, "judge", "", jdebug, session["object_id"])
        session_rows.extend(attempts)
        message_rows.extend(attempt_messages)
        judge_sql, judge_sql_source = ((None, None) if overwritten else
                                         _judge_choice(rec, session_id=session["object_id"]))
        judge_row = {
            "exp_id":     exp_id,
            "qid":        qid,
            "db_id":      db_id,
            "model":      "" if overwritten else str(jdebug.get("model") or ""),
            "mode":       mode,
            "fallback":   fallback,
            "picked":     "" if overwritten or interrupted else _picked(rec),
            "sql":        judge_sql,
            "sql_source": judge_sql_source,
            "tool_calls": judge_calls,
            # None, not 0, unless the whole conversation came from the LLM cache:
            # eval_bird persists the judge's tool_calls and content but never its
            # transcript, so the count is otherwise unknown. A 0 would read as "the
            # judge ran with an empty context" and make it look 17x cheaper than a
            # candidate in any messages-per-agent comparison.
            "n_messages": len(judge_messages) if judge_source in ("llm_episode", "judge_messages") else None,
            # `telemetry_lost` marks the vintage whose judge record the reviewer
            # overwrote: only the mode survived, so every other measure here is absent
            # by necessity rather than by the judge having done nothing.
            "keywords":   (["telemetry_lost"] if overwritten else
                           _protocol_tags(jdebug.get("protocol_events"))
                           + (["from_cache"] if jdebug.get("from_cache") else [])),
            "scores":     judge_scores,
            "usage":      judge_usage or None,
            "duration_sec": None if overwritten else jdebug.get("duration_sec"),
            "n_turns":    None if overwritten else _turn_count(jdebug.get("protocol_events")),
            "prompt_release": judge_release,
            "prompt_replay": judge_replay,
            "session_id": session["object_id"],
        }

    reviewer = rec.get("reviewer_debug") if isinstance(rec.get("reviewer_debug"), dict) else {}
    if reviewer.get("active"):
        # The reviewer's own record sits under route_debug when it rewrote the answer; the
        # overwritten vintage filed it as the judge's, and it is not recovered here.
        own = reviewer.get("route_debug") if isinstance(reviewer.get("route_debug"), dict) else {}
        merge_usage(trace_usage, own.get("usage"))
        _merge_counts(trace_calls, _agent_tool_counts(own))
        overwritten = bool(reviewer.get("judge_debug_original_mode"))
        reviewer_messages: list[Any] = []
        reviewer_source = "none"
        reviewer_extra: dict[str, Any] = {"cost": _recorded_cost(own, own.get("usage"))}
        if not overwritten:
            episode = episodes.get(*_episode_ref(own, reviewer), reference=own.get("episode_ref")) if episodes is not None else None
            direct = own.get("messages") or reviewer.get("messages")
            if isinstance(direct, list) and direct:
                reviewer_messages, reviewer_source = _with_reasoning(direct, own.get("reasoning") or reviewer.get("reasoning") or []), "reviewer_messages"
            elif episode is not None:
                reviewer_messages, reviewer_source = episode, "llm_episode"
                reviewer_extra["episode"] = episodes.provenance(episode)
            else:
                reviewer_messages = _recorded_turns(own)
                reviewer_source = "reviewer_debug" if reviewer_messages else "none"
                reviewer_extra.update(recorded_tool_calls=list(own.get("tool_calls") or []),
                                      recorded_sql_actions=list(own.get("sql_actions") or []))
        session, rows = _session(
            exp_id, qid, db_id, "reviewer", "", reviewer_messages, source=reviewer_source,
            identity_holder={**reviewer, "prompt_identity": own.get("prompt_identity") or reviewer.get("prompt_identity"),
                             "cache_source": own.get("cache_source"), "execution": own.get("execution"),
                             **{key: own[key] for key in ("transcript_status", "capture_reason", "last_checkpoint_reason", "recording_error", "cache_source_error") if key in own}},
            model=str(own.get("model") or ""), usage=own.get("usage"),
            summary=pred_sql, status="error" if reviewer.get("error") else "done",
            extra=reviewer_extra)
        session_rows.append(session)
        message_rows.extend(rows)
        attempts, attempt_messages = _prior_attempts(exp_id, qid, db_id, "reviewer", "", own, session["object_id"])
        session_rows.extend(attempts)
        message_rows.extend(attempt_messages)

    if jdebug.get("transcript_status") == "not_run":
        attempts, attempt_messages = _prior_attempts(exp_id, qid, db_id, "judge", "", jdebug, None)
        session_rows.extend(attempts)
        message_rows.extend(attempt_messages)

    if not reviewer.get("active"):
        own = reviewer.get("route_debug") if isinstance(reviewer.get("route_debug"), dict) else {}
        attempts, attempt_messages = _prior_attempts(exp_id, qid, db_id, "reviewer", "", own, None)
        session_rows.extend(attempts)
        message_rows.extend(attempt_messages)

    for session in session_rows:
        if session["state"].get("prior_attempt"):
            merge_usage(trace_usage, session["usage"])
            _merge_counts(trace_calls, _agent_tool_counts({"tool_calls": session["state"]["recorded_tool_calls"],
                                                         "sql_actions": session["state"]["recorded_sql_actions"]}))
    # Count every persisted agent attempt, plus any old orphan candidate transcript
    # whose summary was removed by a repair. Missing conversations remain a lower bound.
    paired_messages = sum(len(debug.get("messages") or []) for debug in paired)
    n_messages += sum(len(session["message_ids"]) for session in session_rows) - paired_messages

    keywords = derive_sql_tags(pred_sql)
    keywords += [f"has_{name}" for name in sorted(trace_calls) if name]
    # Same vintage caveat as the judge row: on those records the transcripts the reviewer
    # replaced are gone, so this trace's tool_calls/n_messages cover only what survived.
    # The counts are low, not wrong-by-a-little — say so rather than let a cost average
    # quietly read these questions as cheap.
    if (rec.get("reviewer_debug") or {}).get("judge_debug_original_mode"):
        keywords.append("telemetry_lost")
    # The single-agent path has no children to sum, and records its usage on the
    # record itself; the vote path leaves that key absent and is summed above.
    if not trace_usage:
        merge_usage(trace_usage, rec.get("usage"))
    trace_row = {
        "exp_id":       exp_id,
        "qid":          qid,
        "db_id":        db_id,
        "keywords":     keywords,
        "tool_calls":   trace_calls,
        "n_messages":   n_messages,
        "n_candidates": len(candidates),
        "scores":       scores,
        "reasons":      reasons,
        "pred_sql":     pred_sql,
        "error":        str(rec.get("error") or ""),
        "result_attempts": _failure_diagnostics(rec),
        "usage":        trace_usage or None,
        "duration_sec": rec.get("duration_sec"),
        "reviewer":     _reviewer_record(rec.get("reviewer_debug")),
        "question":     str(rec["question"]) if rec.get("question") is not None else None,
        "gold_sql":     str(rec["gold_sql"]) if rec.get("gold_sql") is not None else None,
        "gold_rows":    rec.get("gold_rows") if isinstance(rec.get("gold_rows"), int) else None,
        "pred_rows":    rec.get("pred_rows") if isinstance(rec.get("pred_rows"), int) else None,
        "gold_preview": rec.get("gold_rows_preview") if isinstance(rec.get("gold_rows_preview"), list) else None,
        "pred_preview": rec.get("pred_rows_preview") if isinstance(rec.get("pred_rows_preview"), list) else None,
        "session_ids":  [session["object_id"] for session in session_rows],
    }
    return trace_row, candidate_rows, judge_row, session_rows, message_rows


def _check_rows(cls: type[hb.Entity], rows: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a row HeavenBase would refuse to write: each value checked by its field's logical type, as ``upsert`` checks it.

    Computed fields and the catalog projection are HeavenBase's to fill; the transcripts' projected texts already fit
    (:func:`_session`).

    Args:
        cls: The entity the rows are written as.
        rows: The rows as ``upsert_many`` receives them.

    Raises:
        ValueError: For a value its field's type refuses, naming the entity, the row and the field.
    """
    schema = cls.schema()
    for row in rows:
        for name, value in row.items():
            spec = schema.fields.get(name)
            if spec is None or (value is None and not spec.required):
                continue
            try:
                spec.dtype.validate(value)
            except (TypeError, ValueError) as error:
                label = row.get(schema.pk_name) or " ".join(str(row[key]) for key in ("exp_id", "qid", "slot") if row.get(key))
                raise ValueError(f"{schema.entity_id} row {label}: field {name}: {error}") from error


def _purge_experiment(ws: hb.HeavenBase, exp_id: str) -> int:
    """Delete every trace, candidate, judge, session and message row of one experiment.

    Upsert alone cannot reconcile a re-ingest of a run that SHRANK: a question repaired
    from five candidates plus judge down to one candidate would keep four stale
    candidate rows and a stale judge row (the shipped dap run contains exactly this —
    Q00412/Q00413/Q00416/Q00417), and a judge whose episode left the cache would keep the
    messages of the longer conversation. The experiment row itself needs no purge; it is
    rewritten whole under the same id.

    Args:
        ws: Open lab workspace.
        exp_id: Experiment row id whose child rows should go.

    Returns:
        int: Rows deleted.
    """
    transcripts = transcript_entities(ws)
    session, message = transcripts["session"], transcripts["message"]
    targets = [
        (cls, row["object_id"])
        for cls in (TraceEntity, CandidateEntity, JudgeEntity)
        for row in ws.query(cls).where({"exp_id": {"$eq": exp_id}}).execute().rows()
    ]
    for row in ws.query(session).where({"state.exp_id": {"$eq": exp_id}}).execute().rows():
        targets.extend((message, message_id) for message_id in row.get("message_ids") or [])
        targets.append((session, row["object_id"]))
    if targets:
        ws.delete_many(targets)
    return len(targets)


def result_directory(run_dir: str | os.PathLike[str]) -> str:
    """Return the directory holding a run's per-database result JSONs.

    Args:
        run_dir: A saved run: the parent of ``pass1/`` (a pipeline or ``eval_bird`` run) or
            the directory the per-database JSONs sit in (an evolution evaluation).

    Returns:
        str: ``<run_dir>/pass1`` when it exists, else ``run_dir`` itself — the layout
        :func:`rubiksql_hb.prompts.corpus.read_rows` reads as well.
    """
    pass1 = os.path.join(os.fspath(run_dir), "pass1")
    return pass1 if os.path.isdir(pass1) else os.fspath(run_dir)


def _result_payloads(run_dir: str | os.PathLike[str]) -> list[tuple[str, bytes, dict[str, Any]]]:
    """Every result file of a run: ``(path relative to run_dir, bytes, parsed JSON)``, sorted.

    A result file is a ``*.json`` directly in :func:`result_directory` that holds a
    ``results`` list; dot-files (``.prompt_manifest.json``, ``.resume_health.json``) and
    other JSON are sidecars.
    """
    root = os.fspath(run_dir)
    directory = result_directory(root)
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        if os.path.basename(path).startswith("."):
            continue
        with open(path, "rb") as fh:
            raw = fh.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            out.append((os.path.relpath(path, root).replace(os.sep, "/"), raw, payload))
    return out


def run_result_files(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Return where a run's raw result JSON lives, with each file's sha256.

    The digests are the ones ``prompts outcome`` pairs runs by (``RunView.result_files``),
    so an experiment row names exactly the files an outcome counted.

    Args:
        run_dir: A saved run (see :func:`result_directory`).

    Returns:
        dict[str, Any]: ``{"dir": absolute run directory, "files": {relative path: sha256}}``.
    """
    return _result_files(run_dir, _result_payloads(run_dir))


def _result_files(run_dir: str | os.PathLike[str], payloads: Sequence[tuple[str, bytes, Any]]) -> dict[str, Any]:
    return {"dir": os.path.abspath(os.fspath(run_dir)),
            "files": {relative: hashlib.sha256(raw).hexdigest() for relative, raw, _payload in payloads}}


def _ex(scores: Any) -> float | None:
    """The EX score in a scores record (``ex``, or the pre-rename ``exd``), or ``None`` when unmeasured."""
    if not isinstance(scores, Mapping):
        return None
    value = scores.get("ex", scores.get("exd"))
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _bucket(questions: int, correct: int) -> dict[str, Any]:
    return {"questions": questions, "correct": correct, "accuracy": round(correct / questions, 4) if questions else None}


@_lab_write
def refresh_experiment_scores(ws: hb.HeavenBase, exp_id: str) -> dict[str, Any]:
    """Recompute an experiment's per-database, per-difficulty and per-slot aggregates from its rows.

    ``per_db_scores`` and ``per_difficulty_scores`` count the scored traces (a trace whose
    ``scores`` carry EX) and the correct ones, the difficulty read from the question's QA row
    (``unknown`` when the versioned QA has no label). ``per_slot_scores`` counts, per candidate slot, the candidate
    runs, the ones whose own SQL is scored (``CandidateEntity.scores``, filled by
    :mod:`rubiksql_hb.lab_score`) and the correct ones, and how often the judge picked the
    slot (``JudgeEntity.picked``) with the EX of those picks: the slot's own score where the
    candidates are scored, else the judge row's score of the SQL it chose. Rates are
    fractions, ``None`` over an empty denominator. The row is patched with ``set``.

    Args:
        ws: Open lab workspace.
        exp_id: Experiment row id.

    Returns:
        dict[str, Any]: The three fields as written.
    """
    rows = {cls: ws.query(cls).where({"exp_id": {"$eq": exp_id}}).execute().rows()
            for cls in (TraceEntity, CandidateEntity, JudgeEntity)}
    difficulty = {str(trace["qid"]): qa_for_trace(ws, trace).get("difficulty")
                  for trace in rows[TraceEntity]}
    per_db: dict[str, list[int]] = {}
    per_difficulty: dict[str, list[int]] = {}
    for trace in rows[TraceEntity]:
        ex = _ex(trace.get("scores"))
        if ex is None:
            continue
        for key, table in ((str(trace["db_id"]), per_db), (str(difficulty.get(str(trace["qid"])) or "unknown"), per_difficulty)):
            bucket = table.setdefault(key, [0, 0])
            bucket[0] += 1
            bucket[1] += int(ex >= 1.0)
    own: dict[tuple[str, str], float | None] = {}
    slots: dict[str, dict[str, int]] = {}
    for candidate in rows[CandidateEntity]:
        slot, ex = str(candidate["slot"]), _ex(candidate.get("scores"))
        own[(str(candidate["qid"]), slot)] = ex
        bucket = slots.setdefault(slot, {"questions": 0, "scored": 0, "correct": 0, "picked": 0, "picked_scored": 0, "picked_correct": 0})
        bucket["questions"] += 1
        if ex is not None:
            bucket["scored"] += 1
            bucket["correct"] += int(ex >= 1.0)
    for judge in rows[JudgeEntity]:
        for slot in [part for part in str(judge.get("picked") or "").split(",") if part]:
            bucket = slots.setdefault(slot, {"questions": 0, "scored": 0, "correct": 0, "picked": 0, "picked_scored": 0, "picked_correct": 0})
            bucket["picked"] += 1
            ex = own.get((str(judge["qid"]), slot))
            ex = ex if ex is not None else _ex(judge.get("scores"))
            if ex is not None:
                bucket["picked_scored"] += 1
                bucket["picked_correct"] += int(ex >= 1.0)
    fields = {
        "per_db_scores": {key: _bucket(*value) for key, value in sorted(per_db.items())},
        "per_difficulty_scores": {key: _bucket(*value) for key, value in sorted(per_difficulty.items())},
        "per_slot_scores": {
            slot: {**bucket, "accuracy": round(bucket["correct"] / bucket["scored"], 4) if bucket["scored"] else None,
                   "picked_accuracy": round(bucket["picked_correct"] / bucket["picked_scored"], 4) if bucket["picked_scored"] else None}
            for slot, bucket in sorted(slots.items(), key=lambda item: (len(item[0]), item[0]))
        },
    }
    ws.set(exp_id, fields, entity=ExperimentEntity)
    return fields


@_lab_write
def record_outcome(ws: hb.HeavenBase, exp_id: str, *, verdict: str | None, gain_pp: float | None) -> None:
    """Patch the verdict and gain of the outcome that paired an experiment as a child.

    Args:
        ws: Open lab workspace.
        exp_id: The child run's experiment row id.
        verdict: ``promotable``, ``adoptable``, ``not-promotable`` or ``descriptive``.
        gain_pp: EX gain, child minus parent, in percentage points; ``None`` when unpaired.
    """
    ws.set(exp_id, {"verdict": verdict, "gain_pp": None if gain_pp is None else float(gain_pp)}, entity=ExperimentEntity)


def _episode_users(records: Sequence[Mapping[str, Any]]) -> set[str]:
    """The first-user-message digests of every judge and reviewer conversation the records name."""
    users: set[str] = set()
    for rec in records:
        judge = rec.get("judge_debug") if isinstance(rec.get("judge_debug"), Mapping) else {}
        reviewer = rec.get("reviewer_debug") if isinstance(rec.get("reviewer_debug"), Mapping) else {}
        own = reviewer.get("route_debug") if isinstance(reviewer.get("route_debug"), Mapping) else {}
        for record, holder in ((judge, judge), (own, reviewer)):
            user = _episode_ref(record, holder)[1]
            if user:
                users.add(user)
    return users


def _session_sources(session_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Per role, how many sessions took their messages from each source (``state.source``)."""
    out: dict[str, dict[str, int]] = {}
    for session in session_rows:
        state = session.get("state") or {}
        role = out.setdefault(str(state.get("role") or ""), {})
        role[str(state.get("source") or "none")] = role.get(str(state.get("source") or "none"), 0) + 1
    return out


def _analysis_coverage(session_rows: Sequence[Mapping[str, Any]], trace_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count captured conversations separately from observed failures and unmeasured older results."""
    roles: dict[str, Any] = {}
    for session in session_rows:
        state = session["state"]
        bucket = roles.setdefault(state["role"], {"complete": 0, "partial": 0, "missing": 0, "reasons": {}})
        coverage = state["coverage"]
        bucket[coverage["status"]] += 1
        for reason in coverage["missing"]:
            bucket["reasons"][reason] = bucket["reasons"].get(reason, 0) + 1
    failures: dict[str, Any] = {"observed": 0, "partial": 0, "missing": 0, "reasons": {}, "unrecorded_results": 0,
                                "scope": "observed_evaluation_failures_not_http_calls"}
    for trace in trace_rows:
        diagnostics = trace["result_attempts"]
        failures["unrecorded_results"] += len(diagnostics["unrecorded_results"])
        for attempt in diagnostics["failures"]:
            coverage = attempt["coverage"]
            failures["observed"] += 1
            failures[coverage["status"]] += 1
            for reason in coverage["missing"]:
                failures["reasons"][reason] = failures["reasons"].get(reason, 0) + 1
    return {"sessions": len(session_rows), "roles": roles, "failure_attempts": failures}


def _cost_summary(session_rows: Sequence[Mapping[str, Any]], failure_attempts: Mapping[str, Any]) -> dict[str, Any]:
    """Sum only explicit compatible monetary records; unknown sessions keep the total unknown."""
    totals: dict[str, float] = {}
    recorded = 0
    for session in session_rows:
        cost = session["state"]["cost"]
        if cost["unknown_reason"] is None:
            currency = cost["currency"]
            totals[currency] = totals.get(currency, 0.0) + cost["amount"]
            recorded += 1
    missing = len(session_rows) - recorded
    unrecorded_failures = failure_attempts["partial"] + failure_attempts["missing"]
    unrecorded_results = failure_attempts["unrecorded_results"]
    known = bool(recorded) and not missing and len(totals) == 1 and not unrecorded_failures and not unrecorded_results
    currency = next(iter(totals)) if known else None
    return {"amount": totals[currency] if known else None, "currency": currency,
            "source": "recorded_session_costs", "recorded": recorded, "missing": missing,
            "totals_by_currency": totals,
            "unrecorded_failure_attempts": unrecorded_failures, "unrecorded_results": unrecorded_results,
            "unknown_reason": (None if known else "failed_attempt_cost_unrecorded" if unrecorded_failures
                               else "failure_recording_unavailable" if unrecorded_results
                               else "session_cost_unrecorded" if missing or not recorded else "mixed_currencies")}


def _run_provenance(run_dir: str | os.PathLike[str], cache_db: str | os.PathLike[str] | None,
                    payloads: Sequence[tuple[str, bytes, Any]], identity_source: dict[str, Any] | None) -> dict[str, Any]:
    """Name raw result and telemetry inputs without silently attributing a parent's run metadata."""
    root = Path(run_dir).resolve()
    sidecars = {root / ".pipeline_state.json", root / ".evaluation_inputs.json", root / PROMPT_MANIFEST_NAME,
                Path(result_directory(root)) / PROMPT_MANIFEST_NAME}
    rate_dir = Path(result_directory(root)) / rate_limits.RECORD_DIR
    if rate_dir.is_dir():
        sidecars.update(path for path in rate_dir.iterdir() if path.is_file())
    return {"results": [{"path": str(root / relative), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                        for relative, raw, _payload in payloads],
            "sidecars": [identity_source if identity_source and str(path.resolve()) == identity_source["path"]
                         else _file_evidence(path) for path in sorted(sidecars)],
            "run_identity_source": identity_source,
            "cache": _file_evidence(cache_db) if cache_db is not None else {"available": False, "reason": "not_supplied"},
            "cache_wal": _file_evidence(str(cache_db) + "-wal") if cache_db is not None else None}


def experiment_session_sources(ws: hb.HeavenBase, exp_id: str) -> dict[str, dict[str, int]]:
    """Per role, how many of an experiment's sessions took their messages from each source (``state.source``).

    Args:
        ws: Open lab workspace.
        exp_id: Experiment row id.

    Returns:
        dict[str, dict[str, int]]: Role to source to session count, as :func:`ingest_run` reports it.
    """
    session = transcript_entities(ws)["session"]
    return _session_sources(ws.query(session).where({"state.exp_id": {"$eq": exp_id}}).execute().rows())


def run_rate_limits(run_dir: str | os.PathLike[str]) -> dict[str, dict[str, int]] | None:
    """Return the provider's answers during a run, per lane, from its ``.rate_limits/`` records.

    eval_bird and the MCP servers it starts write one record per process into ``<result directory>/.rate_limits/``
    (:mod:`rubiksql_hb.rate_limits`); every record counts, so a resumed run sums every launch.

    Args:
        run_dir (str | os.PathLike[str]): A saved run (see :func:`result_directory`).

    Returns:
        dict[str, dict[str, int]] | None: Lane to ``{"requests", "rate_limited", "overloaded"}``, or ``None`` for a run
            that kept no record directory.
    """
    directory = os.path.join(result_directory(run_dir), rate_limits.RECORD_DIR)
    return rate_limits.read_records(directory) if os.path.isdir(directory) else None


def _usage_by_role(session_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Token counters per agent role over a run's sessions: ``{role: {sessions, measured, tokens}}``, ``None`` when none measured."""
    out: dict[str, dict[str, Any]] = {}
    for session in session_rows:
        bucket = out.setdefault(str((session.get("state") or {}).get("role") or ""), {"sessions": 0, "measured": 0, "tokens": {}})
        bucket["sessions"] += 1
        if session.get("usage"):
            bucket["measured"] += 1
            merge_usage(bucket["tokens"], session["usage"])
    return {role: bucket for role, bucket in sorted(out.items())} if any(bucket["measured"] for bucket in out.values()) else None


def _json_file(path: Path) -> dict[str, Any] | None:
    """A JSON object, or ``None`` only when absent; an unreadable/malformed file remains an observation error."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: cannot read JSON: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _ledger_stages(root: Path) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    """The observed ledgers, unledgered workspaces and explicit reasons the knowledge cannot be identified."""
    stages: dict[str, dict[str, Any]] = {}
    unledgered: list[str] = []
    unknown: list[str] = []
    try:
        directories = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError as error:
        return {}, [], [f"{root}: cannot read knowledge root: {error}"]
    for ws_dir in directories:
        path = ws_dir / K.LEDGER_FILE
        try:
            ledger = _json_file(path)
        except ValueError as error:
            unknown.append(str(error))
            continue
        if ledger is None:
            if not (ws_dir / f"{ws_dir.name}.db").is_file():
                continue
            unledgered.append(ws_dir.name)
            unknown.append(f"{path}: workspace has no knowledge ledger")
            continue
        if (ledger.get("schema") not in (K.LEDGER_SCHEMA, "rubiksql.knowledge-ledger/1") or ledger.get("db_id") != ws_dir.name
                or not isinstance(ledger.get("stages"), dict) or not ledger["stages"]):
            unknown.append(f"{path}: ledger schema, database or stages are invalid")
            continue
        stages[ws_dir.name] = {}
        for name, record in ledger["stages"].items():
            if not isinstance(record, dict):
                unknown.append(f"{path}: {name} is not a stage record")
                continue
            stages[ws_dir.name][name] = record
            if record.get("status") != "done" or not record.get("input_fingerprint") or not record.get("output_digest"):
                unknown.append(f"{path}: {name} has no completed input/output identity")
    if not stages:
        unknown.append(f"{root}: no knowledge stages were observed")
    return stages, unledgered, unknown


def knowledge_fields(root: str | os.PathLike[str], *, build_record: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Return a knowledge root's ``knowledge_root`` and ``knowledge_build`` fields, read from its files.

    Each workspace ``<root>/<db>`` keeps its knowledge ledger (``prompts.knowledge.LEDGER_FILE``): per stage what it was
    last built from and what it wrote. Per database and stage the fields keep its ``status``, stage ``version``,
    ``release``, ``output_digest``, ``input_fingerprint`` and times; ``inputs_changed``, the inputs whose digests differ
    from the ledger of the root the loop copied this one from (``<root>/../.loop_copy.json``, ``source``), which is
    what the proposal changed in the knowledge (``None`` without a copy record); and, from the last build's record
    (``rubiksql.knowledge-plan/1``), the ``action`` the build took with its ``reasons`` and what a stopped build left
    ``missing``. Only a complete observed ledger set receives an ``id``: two ``status=known`` roots with equal ids
    carry the same recorded knowledge. Missing, unreadable or unfinished ledgers give ``status=unknown``, ``id=None``
    and explicit ``unknown`` reasons, rather than an identity for fabricated empty knowledge.

    Args:
        root (str | os.PathLike[str]): The knowledge root: one workspace directory per database.
        build_record (str | os.PathLike[str] | None): The last build's record; by default the loop's layout,
            ``<root>/../knowledge_build.json`` (``loop.ScriptEvaluator.build_knowledge``), when that file exists.

    Returns:
        dict[str, Any]: ``{"knowledge_root": {path, status, id, databases, unledgered, unknown}, "knowledge_build":
        {stages, record, mode, release, passes, elapsed_sec, rate_limits, unfinished, source, warnings}}``.
    """
    path = Path(root).resolve()
    stages, unledgered, unknown = _ledger_stages(path)
    warnings: list[str] = []
    record_path = Path(build_record) if build_record is not None else path.parent / "knowledge_build.json"
    try:
        record = _json_file(record_path)
        if record is not None and record.get("schema") != K.PLAN_SCHEMA:
            raise ValueError(f"{record_path}: expected {K.PLAN_SCHEMA}")
        if record is None and build_record is not None:
            warnings.append(f"{record_path}: requested build record is missing")
    except ValueError as error:
        warnings.append(str(error))
        record = None
    try:
        copy = _json_file(path.parent / ".loop_copy.json") or {}
    except ValueError as error:
        warnings.append(str(error))
        copy = {}
    source = Path(str(copy["source"])) if copy.get("source") else None
    source_stages: dict[str, dict[str, Any]] = {}
    if source is not None:
        source_stages, _unledgered, problems = _ledger_stages(source)
        warnings.extend(problems)
    decided = (record or {}).get("stages") or {}
    missing = {(str(item.get("stage")), str(item.get("db"))): item.get("missing") for item in (record or {}).get("unfinished") or [] if isinstance(item, Mapping)}
    out: dict[str, dict[str, Any]] = {}
    for db, by_stage in stages.items():
        for stage, entry in by_stage.items():
            before = ((source_stages.get(db) or {}).get(stage) or {}).get("inputs") if source is not None else None
            inputs = entry.get("inputs") if isinstance(entry.get("inputs"), dict) else {}
            decision = (decided.get(stage) or {}).get(db) or {}
            out.setdefault(db, {})[stage] = {
                "record": dict(entry),
                **{key: entry.get(key) for key in ("status", "version", "release", "output_digest", "input_fingerprint", "started_at", "finished_at")},
                "inputs_changed": None if before is None else sorted(name for name in set(before) | set(inputs) if before.get(name) != inputs.get(name)),
                "action": decision.get("action"),
                "reasons": list(decision.get("reasons") or []) if decision else None,
                "missing": missing.get((stage, db)),
            }
    identity = {db: {stage: [entry.get("status"), entry.get("input_fingerprint"), entry.get("output_digest")] for stage, entry in sorted(by_stage.items())}
                for db, by_stage in sorted(stages.items())}
    return {
        "knowledge_root": {"path": str(path), "status": "unknown" if unknown else "known",
                           "id": None if unknown else "k" + sha256_hex(canon(identity))[:16],
                           "databases": sorted(stages), "unledgered": unledgered, "unknown": unknown},
        "knowledge_build": {"stages": out, "record": str(record_path) if record is not None else None,
                            **{key: (record or {}).get(key) for key in ("mode", "release", "passes", "elapsed_sec", "rate_limits", "unfinished")},
                            "source": str(source) if source is not None else None, "warnings": warnings,
                            "cost": _recorded_cost(record or {}, (record or {}).get("usage")),
                            "provenance": {"ledgers": [_file_evidence(path / db / K.LEDGER_FILE) for db in sorted(stages)],
                                           "build_record": _file_evidence(record_path)}},
    }


@_lab_write
def record_knowledge(ws: hb.HeavenBase, exp_id: str, root: str | os.PathLike[str], *, build_record: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Patch an experiment with the knowledge root its run read, as the root's ledgers stand now.

    Call it after the run has been evaluated: the ledgers then describe the
    knowledge the run read. A later re-ingest of the run keeps these fields.

    Args:
        ws (hb.HeavenBase): Open lab workspace.
        exp_id (str): The experiment row id.
        root (str | os.PathLike[str]): The knowledge root the run read (``eval_evolution.py --dbs-dir``).
        build_record (str | os.PathLike[str] | None): The last build's record (:func:`knowledge_fields`).

    Returns:
        dict[str, Any]: The two fields as written.

    Raises:
        KeyError: When the lab holds no experiment ``exp_id``.
        ValueError: When an unknown observation would overwrite a previously identified knowledge snapshot.
    """
    experiment = ws.get(exp_id, ExperimentEntity)
    if not experiment:
        raise KeyError(f"the lab holds no experiment {exp_id!r}; ingest its run first")
    fields = knowledge_fields(root, build_record=build_record)
    previous = experiment.get("knowledge_root") or {}
    if fields["knowledge_root"]["status"] == "unknown" and previous.get("id"):
        raise ValueError(f"{exp_id}: cannot replace recorded knowledge {previous['id']} with unknown knowledge: "
                         + "; ".join(fields["knowledge_root"]["unknown"]))
    ws.set(exp_id, fields, entity=ExperimentEntity)
    return fields


#: Rows per ``upsert_many`` call. HeavenBase's upsert deletes a batch's ids in one statement, and SQLite builds bind
#: at most 32766 variables (CPython's Windows build); a complete evolution run holds more conversation messages than that.
_UPSERT_CHUNK = 2000


def ingest_run(
    ws: hb.HeavenBase,
    run_dir: str | os.PathLike[str],
    name: str,
    *,
    questions_path: str | os.PathLike[str],
    db_fingerprints: Mapping[str, str],
    parameters: dict[str, Any] | None = None,
    summary: str = "",
    queries: str = "",
    models: Sequence[str] | None = None,
    cache_db: str | os.PathLike[str] | None = None,
    iteration: int | None = None,
    parent_experiment: str | None = None,
    verdict: str | None = None,
    gain_pp: float | None = None,
    transcripts: bool = True,
) -> dict[str, Any]:
    """Ingest a verified run under the lab-wide lock; preserve its exact sources.

    Args:
        ws: Open lab workspace.
        run_dir: Run artifact directory with a verified snapshot record.
        name: Explicit experiment id.
        questions_path: Exact evaluated question file.
        db_fingerprints: Verified raw database fingerprints.
        parameters: Recorded launch parameters.
        summary: Local conclusions.
        queries: Question file label.
        models: Recorded candidate model labels.
        cache_db: Recorded episode cache.
        iteration: Optional analysis iteration label.
        parent_experiment: Paired control experiment.
        verdict: Recorded outcome verdict.
        gain_pp: Recorded host-scored EX gain.
        transcripts: Write each agent run's ``agent-session`` and ``agent-message`` rows. ``False``
            writes neither and leaves the session links null; the run directory keeps the conversations.

    Returns:
        Written entity counts and the written sessions' transcript sources.
    """
    from .lab_store import lab_write
    from .prompts.views import history_store

    payloads = _result_payloads(run_dir)
    if not payloads:
        raise FileNotFoundError(f"no result JSON in {os.fspath(run_dir)} or its pass1/")
    manifest, source = _read_prompt_manifest(os.fspath(run_dir))
    snapshot = run_snapshot(run_dir, manifest)
    with lab_write(ws, operation="ingest-run") as writable:
        with history_store(snapshot=snapshot) as stores:
            return _ingest_run(writable, run_dir, name, questions_path=questions_path,
                db_fingerprints=db_fingerprints, parameters=parameters, summary=summary,
                queries=queries, models=models, cache_db=cache_db, iteration=iteration,
                parent_experiment=parent_experiment, verdict=verdict, gain_pp=gain_pp,
                transcripts=transcripts, prompt_store=stores, prompt_snapshot=snapshot,
                prompt_manifest=manifest, prompt_manifest_source=source, recorded_payloads=payloads)


@_lab_write
def _ingest_run(
    ws: hb.HeavenBase,
    run_dir: str | os.PathLike[str],
    name: str,
    *,
    questions_path: str | os.PathLike[str],
    db_fingerprints: Mapping[str, str],
    parameters: dict[str, Any] | None = None,
    summary: str = "",
    queries: str = "",
    models: Sequence[str] | None = None,
    cache_db: str | os.PathLike[str] | None = None,
    iteration: int | None = None,
    parent_experiment: str | None = None,
    verdict: str | None = None,
    gain_pp: float | None = None,
    transcripts: bool = True,
    prompt_store: Store | None = None,
    prompt_snapshot: Any = None,
    prompt_manifest: Mapping[str, Any] | None = None,
    prompt_manifest_source: dict[str, Any] | None = None,
    recorded_payloads: Sequence[tuple[str, bytes, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Backfill one saved eval run into the lab workspace.

    Re-ingesting under the same ``name`` replaces the experiment exactly: existing
    trace/candidate/judge rows and sessions with their messages are purged first, so rows
    from a superseded state of the run cannot linger. Every new row is checked first
    (:func:`_check_rows`), so a value HeavenBase would refuse stops the ingest before the
    purge, with the previous ingest intact. The aggregates are computed last
    (:func:`refresh_experiment_scores`); a later candidate-scoring pass refreshes them.

    Args:
        ws: Open lab workspace (:func:`rubiksql_hb.lab_schema.open_lab_workspace`, which
            enables the agent extension the transcripts need).
        run_dir: The run: the directory containing ``pass1/`` — e.g.
            ``eval_results/official_sim_dap_20260727/runs/dap`` — or the directory holding
            the per-database JSONs, as ``scripts/eval_evolution.py`` writes them.
        name: Experiment row id and label. Explicit on purpose: run directories end in
            generic basenames such as ``dap``, and a guessed label would name every
            checkpoint the same way.
        questions_path: Exact question JSON file used by this evaluation.
        db_fingerprints: Verified content fingerprints of its raw SQLite files.
        parameters: Launch parameters to record verbatim, or ``None``. The eval pipeline
            does not persist its CLI args, so callers pass what they know.
        summary: Prose conclusions to store on the experiment row.
        queries: Question-file label, e.g. ``dev_20240627.json``.
        models: Launch-order candidate labels (the ``--candidates`` routes) for the
            per-candidate ``model`` field, or ``None`` to leave it empty — the saved
            artifacts do not record per-candidate providers.
        cache_db: The LLM cache the run wrote, whose ``llm_episode`` table holds the judges'
            and reviewers' whole conversations; ``None`` keeps what the result rows hold.
        iteration: The recorded analysis iteration that launched the run.
        parent_experiment: The experiment id of the run a child is paired with.
        verdict: The verdict of the outcome that paired the run as a child.
        gain_pp: That outcome's EX gain in percentage points.
        transcripts: Write the sessions and their messages. ``False`` still measures them for
            ``n_messages``, ``analysis_coverage``, ``cost`` and ``usage_by_role``, but writes neither,
            leaves ``session_id``/``session_ids`` null and records ``provenance.transcripts`` as
            ``run_directory`` (``lab`` otherwise).

        Each recorded analysis or outcome fact is ``None`` when not given: a re-ingest then keeps
        the experiment's stored value — only the caller knows them, so a re-ingest from the
        command line must not erase them — and a new experiment has none. The knowledge
        fields :func:`record_knowledge` wrote are kept the same way.

    Returns:
        dict[str, Any]: Row counts written per entity — ``{"traces", "candidates",
        "judges", "sessions", "messages", "experiments": 1}`` — and ``session_sources``,
        per role the written sessions whose messages came from each source.

    Raises:
        FileNotFoundError: ``run_dir`` holds no per-database result JSON, neither in
            ``pass1/`` nor directly.
        ValueError: A row holds a value its field refuses; nothing was deleted or written.
    """
    payloads = recorded_payloads
    if not payloads:
        raise FileNotFoundError(
            f"no result JSON in {os.fspath(run_dir)} or its pass1/ — pass the directory that "
            f"holds the per-database result JSONs or their pass1/, e.g. eval_results/<run>/runs/dap")

    if prompt_manifest is None or prompt_store is None or prompt_snapshot is None:
        raise PromptStoreError("private ingestion requires the verified run snapshot and original manifest bytes")
    manifest, manifest_source = dict(prompt_manifest), prompt_manifest_source
    run_identity, identity_source = _read_run_identity(os.fspath(run_dir), manifest, manifest_source)
    launch = manifest.get("paid_launch")
    diagnostic = bool(manifest.get("diagnostic") or manifest.get("promotion_eligible") is False
                      or (manifest.get("identity") or {}).get("diagnostic_experiment")
                      or (isinstance(launch, Mapping) and launch.get("diagnostic")))
    questions, dataset_version, questions_source = question_file(questions_path)
    recorded_dataset, recorded_dbs, db_source = recorded_run_inputs(run_dir)
    if recorded_dataset != dataset_version:
        raise ValueError(f"run {name} question file fingerprint mismatch: recorded {recorded_dataset}, actual {dataset_version}")
    if run_identity is not None and run_identity.get("question_set") is not None \
            and run_identity.get("question_set") != dataset_version:
        raise ValueError(f"run {name} identity.question_set does not match the actual question file")
    if manifest.get("identity") and manifest["identity"].get("question_set") is not None \
            and manifest["identity"].get("question_set") != dataset_version:
        raise ValueError(f"run {name} manifest question_set does not match the actual question file")
    supplied_dbs = {str(db): content_version(value) for db, value in db_fingerprints.items()}
    question_by_qid: dict[str, dict[str, Any]] = {}
    for item in questions:
        qid = qid_str(item["question_id"])
        if qid in question_by_qid:
            raise ValueError(f"question file {questions_path} repeats {qid}")
        db = str(item["db_id"])
        if db not in recorded_dbs or db not in supplied_dbs:
            raise ValueError(f"run {name} lacks a verified database fingerprint for {db}")
        if recorded_dbs[db] != supplied_dbs[db]:
            raise ValueError(f"run {name} database fingerprint conflict for {db}")
        question_by_qid[qid] = item
    records = [rec for _relative, _raw, payload in payloads for rec in payload.get("results") or []]
    seen_results: set[str] = set()
    for rec in records:
        qid = str(rec.get("id") or "")
        if qid in seen_results:
            raise ValueError(f"run {name} has duplicate result {qid!r}")
        seen_results.add(qid)
        item = question_by_qid.get(qid)
        if item is None:
            raise ValueError(f"run {name} result {qid!r} is absent from {questions_path}")
        expected = question_with_hint(str(item["question"]), item.get("evidence"))
        if rec.get("question") != expected:
            raise ValueError(f"run {name} {qid} question (including hint) differs from {questions_path}")
        if rec.get("gold_sql") != item["SQL"]:
            raise ValueError(f"run {name} {qid} gold SQL differs from {questions_path}")
        if rec.get("db_id") != item["db_id"]:
            raise ValueError(f"run {name} {qid} database differs from {questions_path}")
    result_hashes = {relative: hashlib.sha256(raw).hexdigest() for relative, raw, _payload in payloads}
    record_sources = {id(rec): {"path": str(Path(run_dir).resolve() / relative),
                               "sha256": result_hashes[relative], "pointer": f"/results/{idx}"}
                      for relative, raw, payload in payloads for idx, rec in enumerate(payload.get("results") or [])}
    episodes = _Episodes(cache_db, str(manifest.get("ns") or ""), _episode_users(records)) if cache_db is not None else None

    trace_rows: list[dict] = []
    candidate_rows: list[dict] = []
    judge_rows: list[dict] = []
    session_rows: list[dict] = []
    message_rows: list[dict] = []
    for rec in records:
        trace, cands, judge, sessions, messages = _trace_rows(name, rec, models, episodes)
        item = question_by_qid[trace["qid"]]
        trace["qa_id"] = qa_object_id(dataset_version, supplied_dbs[str(item["db_id"])], trace["qid"])
        record_digest = sha256_hex(canon(rec))
        for session in sessions:
            session["state"]["provenance"] = {**record_sources[id(rec)], "record_sha256": record_digest}
        trace_rows.append(trace)
        candidate_rows.extend(cands)
        if judge is not None:
            judge_rows.append(judge)
        session_rows.extend(sessions)
        message_rows.extend(messages)

    # Re-ingest replaces raw run observations, but scoring may already have
    # measured later executions. Retain those bounded measurements only when the
    # exact QA version and SQL still match; never transfer one version's result
    # onto another question or database.
    prior_traces = {str(row["qid"]): row for row in ws.query(TraceEntity)
                    .where({"exp_id": {"$eq": name}}).execute().rows()}
    prior_candidates = {(str(row["qid"]), str(row["slot"])): row for row in ws.query(CandidateEntity)
                        .where({"exp_id": {"$eq": name}}).execute().rows()}
    prior_judges = {str(row["qid"]): row for row in ws.query(JudgeEntity)
                    .where({"exp_id": {"$eq": name}}).execute().rows()}
    stable_qids = {str(row["qid"]) for row in trace_rows
                   if (old := prior_traces.get(str(row["qid"]))) is not None
                   and old.get("qa_id") == row["qa_id"]}
    stable_gold = (set(prior_traces) == {str(row["qid"]) for row in trace_rows}
                   and all(row["qid"] in stable_qids
                           and prior_traces[str(row["qid"])].get("gold_sql") == row.get("gold_sql")
                           for row in trace_rows))
    for trace in trace_rows:
        old = prior_traces.get(str(trace["qid"]))
        if old is not None and trace["qid"] in stable_qids and old.get("pred_sql") == trace["pred_sql"]:
            trace["pred_execution_result"] = old.get("pred_execution_result")
    for candidate in candidate_rows:
        old = prior_candidates.get((str(candidate["qid"]), str(candidate["slot"])))
        if old is not None and candidate["qid"] in stable_qids and old.get("sql") == candidate["sql"]:
            candidate["scores"] = old.get("scores")
            candidate["execution_result"] = old.get("execution_result")
    for judge in judge_rows:
        old = prior_judges.get(str(judge["qid"]))
        if (old is not None and judge["qid"] in stable_qids and old.get("sql") == judge.get("sql")
                and old.get("sql_source") == judge.get("sql_source")):
            judge["execution_result"] = old.get("execution_result")

    # The prompt release: per-row fields and the run manifest are the record; the last
    # console banner is read only to cross-check them (a disagreement is said out loud,
    # never resolved in the banner's favour).
    prompt_release, prompt_releases = _experiment_release(records, manifest, stores=prompt_store)
    banner = _prompt_release_banner(os.fspath(run_dir))
    if banner.get("release") and prompt_release not in (None, "mixed") \
            and banner["release"] != prompt_release:
        print(f"[lab-ingest] {name}: the last [prompt-release] banner names "
              f"{banner['release']} but the rows/manifest name {prompt_release}; the rows are "
              f"the record, the banner a cross-check", flush=True)

    sum_calls: dict[str, int] = {}
    for trace in trace_rows:
        _merge_counts(sum_calls, trace["tool_calls"])
    # Vocabulary is taken BEFORE the synthetic 'total' key joins the counts, and drops
    # the unnamed-call bucket that a malformed tool record lands in.
    tool_vocab = sorted(name for name in sum_calls if name)
    sum_calls["total"] = sum(sum_calls.values())
    n_traces = len(trace_rows)
    avg_calls = {name_: round(count / n_traces, 3) for name_, count in sum_calls.items()} \
        if n_traces else {}

    # Usage is summed only over the traces that recorded it: dividing by every trace
    # would report a run whose artifacts predate usage capture as cheap rather than
    # unmeasured. None on both fields when no trace carried any.
    sum_usage: dict[str, Any] = {}
    n_usage = 0
    for trace in trace_rows:
        if trace["usage"]:
            merge_usage(sum_usage, trace["usage"])
            n_usage += 1
    if sum_usage:
        # Some historical roles omit total_tokens. Derive from the complete
        # input/output counters, never from a partial total or cache aliases.
        sum_usage["total"] = (sum_usage.get("prompt_tokens", 0) + sum_usage.get("completion_tokens", 0)
                              if "prompt_tokens" in sum_usage or "completion_tokens" in sum_usage
                              else sum_usage.get("total_tokens", 0))

        def average(counters: dict[str, Any]) -> dict[str, Any]:
            return {key: average(value) if isinstance(value, dict) else round(value / n_usage, 3)
                    for key, value in counters.items()}

        avg_usage: dict[str, Any] | None = average(sum_usage)
    else:
        sum_usage, avg_usage = {}, None

    avg_scores: dict[str, float] = {}
    for metric in SCORE_FIELDS:
        values = [t["scores"][metric] for t in trace_rows if metric in t["scores"]]
        if values:
            avg_scores[metric] = round(sum(values) / len(values), 4)

    implementation = manifest.get("implementation") or {}
    code_commit = (implementation.get("commit") or "") if isinstance(implementation, dict) else ""
    commit_file = _find_upward(os.fspath(run_dir), "CODE_COMMIT.txt") if not code_commit else None
    if commit_file:
        with open(commit_file, encoding="utf-8") as fh:
            code_commit = fh.read().strip()[:40]

    # The analysis facts about the run are the caller's to give; a re-ingest that does not name one keeps the stored value,
    # and the knowledge root it read (record_knowledge) is kept whole.
    stored = ws.get(name, ExperimentEntity) or {}
    stored = stored if isinstance(stored, dict) else stored.to_dict()
    paired_parent = parent_experiment or stored.get("parent_experiment")
    if paired_parent:
        parent_row = ws.get(paired_parent, ExperimentEntity)
        parent_row = parent_row if isinstance(parent_row, dict) or parent_row is None else parent_row.to_dict()
        parent_inputs = ((parent_row or {}).get("provenance") or {}).get("inputs") or {}
        if diagnostic or ((parent_row or {}).get("provenance") or {}).get("diagnostic"):
            raise ValueError("diagnostic measurements cannot be paired in the lab")
        if parent_inputs.get("dataset_version") != dataset_version:
            raise ValueError("lab pairing requires the same content-pinned question set as its parent")
    facts = {"iteration": iteration, "parent_experiment": parent_experiment, "verdict": verdict, "gain_pp": None if gain_pp is None else float(gain_pp),
             "knowledge_root": None, "knowledge_build": None}
    analysis_coverage = _analysis_coverage(session_rows, trace_rows)
    experiment = {
        "object_id":      name,
        "name":           name,
        "ts":             int(max(os.path.getmtime(os.path.join(os.fspath(run_dir), relative)) for relative, _raw, _payload in payloads) * 1000),
        "code_commit":    code_commit,
        "prompt_fp":      _prompt_fingerprint(os.fspath(run_dir)),
        "queries":        queries or str(Path(questions_path).resolve()),
        "n_questions":    n_traces,
        "parameters":     parameters or {},
        "avg_scores":     avg_scores,
        "avg_tool_calls": avg_calls,
        "sum_tool_calls": sum_calls,
        "avg_usage":      avg_usage,
        "sum_usage":      sum_usage or None,
        "tool_vocab":     tool_vocab,
        "legacy_tools":   [name for name in tool_vocab if name in LEGACY_TOOLS],
        "prompt_release": prompt_release,
        "authority_revision": {"event_count": prompt_snapshot.revision.event_count,
                               "latest_event_id": prompt_snapshot.revision.latest_event_id},
        "prompt_releases": prompt_releases,
        "prompt_proposal": (str(manifest["proposal"]) if manifest.get("proposal") else None),
        "prompt_keyspace": (str(manifest["keyspace"]) if manifest.get("keyspace") else None),
        "cache_ns":       (str(manifest.get("ns") or "") if manifest else None),
        "run_identity":   run_identity,
        "result_files":   _result_files(run_dir, payloads),
        "provenance":     {**_run_provenance(run_dir, cache_db, payloads, identity_source),
                            **({"diagnostic": True} if diagnostic else {}),
                            "transcripts": "lab" if transcripts else "run_directory",
                            "prompt_snapshot": {"path": str(prompt_snapshot.path), "digest": prompt_snapshot.digest,
                                                "release": prompt_snapshot.release,
                                                "authority_revision": {"event_count": prompt_snapshot.revision.event_count,
                                                                       "latest_event_id": prompt_snapshot.revision.latest_event_id}},
                            "inputs": {"dataset_version": dataset_version,
                                       "db_fingerprints": {db: supplied_dbs[db] for db in sorted({str(item['db_id']) for item in question_by_qid.values()})},
                                       "questions_source": questions_source,
                                       "database_source": db_source}},
        "analysis_coverage": analysis_coverage,
        "cost":           _cost_summary(session_rows, analysis_coverage["failure_attempts"]),
        "rate_limits":    run_rate_limits(run_dir),
        "usage_by_role":  _usage_by_role(session_rows),
        **{key: value if value is not None else stored.get(key) for key, value in facts.items()},
        "summary":        summary,
    }
    prior_provenance = stored.get("provenance") if isinstance(stored.get("provenance"), Mapping) else {}
    if stable_gold and prior_provenance.get("gold_consistency") is not None:
        experiment["provenance"]["gold_consistency"] = prior_provenance["gold_consistency"]
    if not transcripts:
        # Measured above; the run directory keeps the conversations, so no session row exists to link to.
        for row in (*candidate_rows, *judge_rows):
            row["session_id"] = None
        for trace in trace_rows:
            trace["session_ids"] = None
        session_rows, message_rows = [], []
    native = transcript_entities(ws)
    batches = ((TraceEntity, trace_rows), (CandidateEntity, candidate_rows), (JudgeEntity, judge_rows),
               (native["message"], message_rows), (native["session"], session_rows))
    # Checked before the purge: a refused value must neither leave the experiment half written nor cost the rows of
    # its previous ingest.
    for cls, rows in (*batches, (ExperimentEntity, [experiment])):
        _check_rows(cls, rows)
    ingest_qa(ws, questions_path, database_versions=supplied_dbs,
              expected_dataset_version=dataset_version)
    _purge_experiment(ws, name)
    # The purge removed every row these batches write, so their catalog projections are new:
    # "fresh" skips HeavenBase's per-row lookup of an existing one (thousands of messages).
    for cls, rows in batches:
        for start in range(0, len(rows), _UPSERT_CHUNK):
            ws.upsert_many(cls, rows[start:start + _UPSERT_CHUNK], catalog_mode="fresh")
    ws.upsert(ExperimentEntity, experiment)
    refresh_experiment_scores(ws, name)
    return {"traces": n_traces, "candidates": len(candidate_rows),
            "judges": len(judge_rows), "sessions": len(session_rows),
            "messages": len(message_rows), "experiments": 1,
            "session_sources": _session_sources(session_rows)}


def main(argv: list[str] | None = None) -> None:
    """CLI entry: ``lab_ingest qa|run|refresh|knowledge|register`` on the bound workspace.

    Args:
        argv: Argument list, or ``None`` for ``sys.argv``.

    Returns:
        None: Prints the outcome of the chosen subcommand.
    """
    import argparse

    from .lab_schema import register_lab_workspace
    from .prompts.runner import Lab

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-binding", type=Path, help="explicit immutable authority file")
    parser.add_argument("--workspace", type=Path, help="Assert the bound evolution database or its directory")
    parser.add_argument("--lab-dir", help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    qa = sub.add_parser("qa", help="Register one content-versioned question file")
    qa.add_argument("--queries", default=str(V0_QUESTIONS),
                    help="Exact question JSON file (default: the Mini-Dev 500 evolution input)")
    qa.add_argument("--db-dir", required=True, help="Raw SQLite directory for content fingerprint registration")

    run = sub.add_parser("run", help="Backfill one saved eval run")
    run.add_argument("--run-dir", required=True,
                     help="The run: the directory containing pass1/, or the one holding the "
                          "per-database result JSONs")
    run.add_argument("--name", required=True, help="Experiment label, e.g. dap_20260727")
    run.add_argument("--params-json", default="", help="Launch parameters as a JSON object")
    run.add_argument("--summary", default="", help="Prose conclusions for the run")
    run.add_argument("--queries", default=str(V0_QUESTIONS), help="Exact question JSON file the run answered")
    run.add_argument("--db-dir", required=True, help="Raw SQLite directory; source content is reverified before ingest")
    run.add_argument("--models", default="",
                     help="Comma-separated launch-order candidate labels, e.g. "
                          "qwen:qwen3.7-plus,qwen:qwen3.7-plus")
    run.add_argument("--cache-db", default="",
                     help="The LLM cache the run wrote: the judges' and reviewers' whole "
                          "conversations come from its llm_episode table")

    sub.add_parser("refresh", help="Refresh the authority's derived GeneralRule and KnobSet views")
    sub.add_parser("store", help=argparse.SUPPRESS)

    knowledge = sub.add_parser("knowledge", help="Record the knowledge root an ingested run read")
    knowledge.add_argument("--name", required=True, help="The experiment the run was ingested as")
    knowledge.add_argument("--root", required=True, help="The knowledge root the run read (its --dbs-dir)")
    knowledge.add_argument("--build-record", default="",
                           help="The root's last build record (default: <root>/../knowledge_build.json when present)")

    sub.add_parser("register",
                   help="Register the lab workspace in HeavenBase's workspace registry; serve it with "
                        "scripts/experiment_dashboard.py --workspace-only")

    args = parser.parse_args(argv)
    if args.authority_binding is not None:
        from .prompts.binding import select_binding
        select_binding(args.authority_binding)
    if args.lab_dir is not None:
        parser.error("--lab-dir is retired; --workspace may only name the bound evolution workspace")
    if args.command == "store":
        parser.error("store projection is retired; use refresh for the derived GeneralRule and KnobSet views")
    if args.command == "register":
        spec = register_lab_workspace(args.workspace)
        print(f"registered evolution workspace {spec.id!r}; serve it with: "
              "uv run python scripts/experiment_dashboard.py --workspace-only")
        return
    lab = Lab(args.workspace)
    if args.command == "refresh":
        print(json.dumps(lab.refresh(), indent=1, sort_keys=True))
        return

    with lab.ingestion_lock(operation=f"lab-{args.command}") as ws:
        if args.command == "knowledge":
            fields = record_knowledge(ws, args.name, args.root, build_record=args.build_record or None)
            root = fields["knowledge_root"]
            print(f"recorded {root['status']} knowledge root {root['id']} ({root['path']}, {len(root['databases'])} databases) on {args.name}")
            return
        if args.command == "qa":
            from .lab_identity import database_content_version, database_path

            records, _version, _source = question_file(args.queries)
            versions = {db: database_content_version(database_path(args.db_dir, db))
                        for db in sorted({str(rec["db_id"]) for rec in records})}
            count = ingest_qa(ws, args.queries, database_versions=versions)
            print(f"wrote {count} new or changed QA rows of the {len(records)} in {args.queries} into {lab.path}")
        else:
            from .lab_identity import database_path, validate_database_version

            _dataset, versions, _source = recorded_run_inputs(args.run_dir)
            records, _version, _questions_source = question_file(args.queries)
            for db in sorted({str(rec["db_id"]) for rec in records}):
                if db not in versions:
                    raise ValueError(f"run has no fingerprint for {db}")
                validate_database_version(database_path(args.db_dir, db), versions[db])
            counts = ingest_run(
                ws, args.run_dir, args.name,
                questions_path=args.queries, db_fingerprints=versions,
                parameters=json.loads(args.params_json) if args.params_json else None,
                summary=args.summary,
                models=[m.strip() for m in args.models.split(",") if m.strip()] or None,
                cache_db=args.cache_db or None,
            )
            print(f"ingested {counts} from {args.run_dir} into {lab.path}")


if __name__ == "__main__":
    main()
