"""The knowledge stages a release feeds, and the stages a child release rebuilds on its parent's knowledge.

The pipeline's knowledge build (``scripts/run_official_pipeline.py --stop-after-examples``) runs eleven stages per
database into that database's workspace, in this order:

========================  =====================================================  ==============================
stage                     writes                                                 release rows it reads
========================  =====================================================  ==============================
``setup``                 tables, columns, enum values, the column docs          —
``profiling``             every column's ``profile_desc``                        ``gen.profile.columns``
``table_profiles``        every table's ``profile_desc``                         ``gen.profile.table``
``db_profile``            the database overview                                  ``gen.profile.db``
``general_rules``         the global rules (the corpus)                          ``rules.general``
``hint_rules``            the database's own rules, mined from its hints         ``gen.rules_hints``
``synonyms``              the database's synonyms, mined from its hints          ``gen.synonyms``
``examples``              the examples transferred from the train pool           ``gen.examples.transfer``
``source_gaps``           SQL-first examples at checked primary source gaps      packaged SQL-first document
``source_gap_check``      removes invalid supplement examples using data_check  —
``data_check``            measured notes on documented claims the stored         —
                          values refute (column and table texts, one rule
                          each), and deletes the transferred examples that
                          act on them
========================  =====================================================  ==============================

:data:`STAGES` declares, for each stage, its invalidating prerequisites (``upstream``), additional ordering-only
prerequisites (``after``) and the release rows it reads; no stage reads a knob, so a proposal that moves only knobs
rebuilds nothing. The scripts' knowledge ledger
(``scripts/knowledge_build.py``) fingerprints every other input of a stage — the database file, the questions, the
models and their request parameters, the stage's code version — and records in each workspace what the stage last built
from; this module is the part the package needs. :func:`release_changes` names the stages a child release rebuilds on
knowledge built with its parent, each with the reason, without a model call: META reads it to decide
whether a proposal's arms share one knowledge root (a prompt or knob proposal) or the child needs its own, rebuilt
incrementally from a copy of the parent's (a rules or generator proposal).

The general rule corpus is read by the ``general_rules`` stage alone. The hint-rule miner also compares its proposals
with its pinned ``rules.general`` corpus (it drops a near-verbatim restatement of a general rule), but ``hint_rules``
does not list the corpus: a corpus change re-injects the rules and re-mines nothing (the user's decision of
2026-09-26), so the rules a database mined stay those mined beside the corpus they were built with.

Example transfer includes column and table profiles in its schema text, but profile wording does not invalidate
transferred examples (the user's decision of 2026-10-20). A profile-only child reuses its parent's examples; a later
transfer triggered by another input uses the current profiles, waiting for any profiling and table-profile runs in
the same build to finish. Schema and official descriptions still invalidate examples.

The data check reads no release row: it is deterministic SQL over the database file, so it follows the schema and docs it
reads, a new example transfer (which it filters) and its own code version. It annotates the profiles but does not read
them: it waits for them, and a profile rewrite that drops its notes changes its output, which reruns it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from . import entries as E
from .canon import canon, sha256_hex

__all__ = [
    "ACTIONS",
    "DOCUMENTS",
    "EXIT_INCOMPLETE",
    "LEDGER_FILE",
    "LEDGER_SCHEMA",
    "NO_RECORD",
    "PLAN_SCHEMA",
    "STAGES",
    "STAGE_NAMES",
    "UNRECORDED_WORKSPACE",
    "KnowledgeStage",
    "corpus_rules_digest",
    "describe_changes",
    "plan_refusals",
    "plan_runs_nothing",
    "plan_summary",
    "plan_unledgered",
    "release_changes",
    "release_inputs",
    "stage",
]

#: The knowledge ledger inside each workspace directory, and its schema: ``scripts/knowledge_build.py`` writes it (its
#: ``LEDGER_FILE`` and ``LEDGER_SCHEMA`` are these), and the lab reads it (``lab_ingest.knowledge_fields``). Per stage it
#: records ``{status, version, inputs, input_fingerprint, output_digest, generated_rows, release, started_at, finished_at}``.
#: ``generated_rows`` maps owned semantic row identities to digests; null means legacy output remains unverified.
LEDGER_FILE = "knowledge_ledger.json"
LEDGER_SCHEMA = "rubiksql.knowledge-ledger/2"
#: The record of a knowledge plan or build (``run_official_pipeline.py --plan-json``): ``{schema, mode, release, root,
#: queries, stages: {stage: {db: {action, reasons}}}, summary: {stage: {action: [dbs]}}}``. ``mode`` is ``plan``, ``build``
#: (the decisions a finished build took) or ``incomplete`` (a build that stopped ``[knowledge-incomplete]``); the last
#: two also carry ``passes`` (the passes the build ran), ``elapsed_sec`` and ``rate_limits`` (stage to lane to the
#: provider's ``{requests, rate_limited, overloaded}`` over every pass: 429s, and 5xx or timeouts), and ``incomplete``
#: carries ``unfinished``, the stage runs still unfinished, each ``{stage, db, missing}`` (``missing`` what the stage
#: said it lacks, or why it was held).
PLAN_SCHEMA = "rubiksql.knowledge-plan/1"
#: ``run_official_pipeline.py``'s exit status when its knowledge build stopped ``[knowledge-incomplete]``: stage runs the
#: provider kept leaving unfinished after the last pass. With ``--plan-json`` the pipeline has then written the
#: ``incomplete`` record, and the same command resumes exactly those runs.
EXIT_INCOMPLETE = 3
#: A plan's decisions per stage and database, in the order a plan lists them: ``refuse`` (the workspace cannot follow a
#: changed input in place), ``run``, ``resume`` (an interrupted run of the same inputs), ``after`` (a plan only: an
#: upstream stage runs, and this one runs if the output it reads changes) and ``skip``.
ACTIONS = ("refuse", "run", "resume", "after", "skip")
#: The reason of a stage's first run in a workspace: its ledger holds no record of it.
NO_RECORD = "no ledger record"
#: The reason setup refuses a workspace that exists without a record of its setup — built before the ledger, by another
#: tool (``setup_dbs.py`` alone), or copied without its ledger: the build cannot vouch for what it holds, and a
#: registration it did not make cannot be redone in place.
UNRECORDED_WORKSPACE = ("the workspace exists but its ledger holds no record of its setup (built before the knowledge ledger, by another "
                        "tool, or copied without its ledger): this build cannot vouch for what it holds")


@dataclass(frozen=True)
class KnowledgeStage:
    """One stage of the knowledge build and the release rows it reads.

    Attributes:
        name (str): The stage, as the ledger and the pipeline name it.
        upstream (tuple[str, ...]): Prerequisite stages whose output facets can invalidate it.
        documents (tuple[str, ...]): The generator documents (``gen.*`` entries) it renders.
        corpus (bool): Whether it reads the general rule corpus (``rules.general``).
        knobs (tuple[str, ...]): The ``config.knobs`` values it reads.
        after (tuple[str, ...]): Additional prerequisites to wait for when this stage runs; their outputs do not invalidate it.
    """

    name: str
    upstream: tuple[str, ...] = ()
    documents: tuple[str, ...] = ()
    corpus: bool = False
    knobs: tuple[str, ...] = ()
    after: tuple[str, ...] = ()


#: The knowledge build's stages, in the order the pipeline runs them.
STAGES: tuple[KnowledgeStage, ...] = (
    KnowledgeStage("setup"),
    KnowledgeStage("profiling", ("setup",), ("gen.profile.columns",)),
    KnowledgeStage("table_profiles", ("setup", "profiling"), ("gen.profile.table",)),
    KnowledgeStage("db_profile", ("setup", "table_profiles"), ("gen.profile.db",)),
    KnowledgeStage("general_rules", corpus=True),
    KnowledgeStage("hint_rules", documents=("gen.rules_hints",)),
    KnowledgeStage("synonyms", ("setup",), ("gen.synonyms",)),
    KnowledgeStage("examples", ("setup",), ("gen.examples.transfer",), after=("profiling", "table_profiles")),
    KnowledgeStage("data_check", ("setup", "examples"), after=("profiling", "table_profiles")),
    KnowledgeStage("source_gaps", ("setup", "examples", "data_check")),
    KnowledgeStage("source_gap_check", ("setup", "source_gaps")),
)
#: The stage names, in order.
STAGE_NAMES: tuple[str, ...] = tuple(item.name for item in STAGES)
#: The generator documents a knowledge build renders, in stage order.
DOCUMENTS: tuple[str, ...] = tuple(document for item in STAGES for document in item.documents)

_BY_NAME = {item.name: item for item in STAGES}


def stage(name: str) -> KnowledgeStage:
    """The declaration of one stage.

    Args:
        name (str): A name of :data:`STAGE_NAMES`.

    Returns:
        KnowledgeStage: Its declaration.

    Raises:
        KeyError: If no stage has that name.
    """
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown knowledge stage {name!r}; the stages are {list(STAGE_NAMES)}") from None


def corpus_rules_digest(release: Any) -> str:
    """The identity of the rules a release's corpus writes into a workspace: ``rules.rules_digest`` of its records.

    Only the records reach a workspace, so provenance or generation metadata that changes alone changes nothing a
    stage writes. A ``rubiksql.closure/2`` release's corpus is reconstructed exactly from its rule versions
    (``Release.rules_general``), whose ``knowledge_digest`` the loader has verified equals this value, so the
    representation change itself, a new rule row id or a new release id rebuilds nothing.

    Args:
        release (Release): The loaded release.

    Returns:
        str: ``sha256:<hex>``, or ``absent`` for a release published before the corpus moved into the release.
    """
    from .rules import rules_digest

    try:
        corpus = release.rules_general()
    except KeyError:
        return "absent"
    return rules_digest(corpus.get("rules") or [])


def release_inputs(release: Any, item: KnowledgeStage) -> dict[str, str]:
    """The release rows one stage reads, each by what identifies the content the stage sees.

    Args:
        release (Release): The loaded release.
        item (KnowledgeStage): The stage.

    Returns:
        dict[str, str]: ``gen.*`` document name to its content key (``absent`` when the release lacks it),
            ``rules.general`` to :func:`corpus_rules_digest`, ``knob:<name>`` to the sha256 of the knob's value.
    """
    inputs = {document: str(release.refs.get(document) or "absent") for document in item.documents}
    if item.corpus:
        inputs[E.RULES_ENTRY] = corpus_rules_digest(release)
    for name in item.knobs:
        inputs[f"knob:{name}"] = "sha256:" + sha256_hex(canon(release.knob(name)))
    return inputs


def release_changes(parent: Any, child: Any) -> dict[str, list[str]]:
    """The stages a child release rebuilds on knowledge built with its parent, each with why.

    A stage rebuilds when a release row it reads differs between the two releases, and follows when an upstream stage
    rebuilds (its outputs may change); every other input — the database, the questions, the models, the code — is the
    same for two releases built by one checkout on one knowledge root. Ordering-only prerequisites (``after``) do not
    propagate rebuilds.

    Args:
        parent (Release): The release the knowledge was built with.
        child (Release): The release to build.

    Returns:
        dict[str, list[str]]: Stage name to its reasons, in stage order: the release rows that differ (``gen.…``,
            ``rules.general``, ``knob:…``) and ``after <stage>`` for an upstream stage that rebuilds. Empty when the
            two releases build the same knowledge.
    """
    changes: dict[str, list[str]] = {}
    for item in STAGES:
        before, after = release_inputs(parent, item), release_inputs(child, item)
        reasons = [name for name in sorted(set(before) | set(after)) if before.get(name) != after.get(name)]
        reasons += [f"after {name}" for name in item.upstream if name in changes]
        if reasons:
            changes[item.name] = reasons
    return changes


def describe_changes(changes: Mapping[str, list[str]]) -> str:
    """One line for a report: which stages rebuild and why.

    Args:
        changes (Mapping[str, list[str]]): A :func:`release_changes` result.

    Returns:
        str: E.g. ``general_rules (rules.general changed)``, or ``nothing to rebuild``.
    """
    if not changes:
        return "nothing to rebuild"
    parts = []
    for name, reasons in changes.items():
        changed = [reason for reason in reasons if not reason.startswith("after ")]
        follows = [reason for reason in reasons if reason.startswith("after ")]
        why = f"{', '.join(changed)} changed" if changed else ", ".join(follows)
        parts.append(f"{name} ({why})")
    return "; ".join(parts)


def plan_summary(record: Mapping[str, Any] | None) -> str:
    """One line for a report: what a knowledge plan or build runs, stage by stage.

    Args:
        record (Mapping[str, Any] | None): A :data:`PLAN_SCHEMA` record.

    Returns:
        str: E.g. ``general_rules run on 11 database(s)``, or ``nothing to rebuild``.
    """
    summary = (record or {}).get("summary") or {}
    order = {name: index for index, name in enumerate(STAGE_NAMES)}
    parts = []
    for name in sorted(summary, key=lambda name: (order.get(name, len(order)), name)):
        for action in ACTIONS[:-1]:
            if summary[name].get(action):
                parts.append(f"{name} {action} on {len(summary[name][action])} database(s)")
    return "; ".join(parts) or "nothing to rebuild"


def plan_runs_nothing(record: Mapping[str, Any]) -> bool:
    """Whether a knowledge plan skips every stage of every database: the root carries the release's knowledge.

    Args:
        record (Mapping[str, Any]): A :data:`PLAN_SCHEMA` record.

    Returns:
        bool: ``True`` when no decision is other than ``skip``.
    """
    return all(set(actions) <= {"skip"} for actions in (record.get("summary") or {}).values())


def plan_unledgered(record: Mapping[str, Any]) -> bool:
    """Whether no workspace of a knowledge plan's root carries a ledger: every decision is a first run
    (:data:`NO_RECORD`) or setup's refusal of a workspace that exists without one (:data:`UNRECORDED_WORKSPACE`).

    Such a root is empty (a first build would make every stage, paid) or was built before the incremental build, by
    another tool, or copied without its ledgers (the build refuses it at setup).

    Args:
        record (Mapping[str, Any]): A :data:`PLAN_SCHEMA` record.

    Returns:
        bool: ``True`` for such a root (and ``False`` for a plan with no decision at all).
    """
    decisions = [item for by_db in (record.get("stages") or {}).values() for item in by_db.values()]
    return bool(decisions) and all(item.get("reasons") in ([NO_RECORD], [UNRECORDED_WORKSPACE]) for item in decisions)


def plan_refusals(record: Mapping[str, Any]) -> list[str]:
    """The refusals of a knowledge plan: the workspaces the build cannot bring in line in place, each with why.

    Args:
        record (Mapping[str, Any]): A :data:`PLAN_SCHEMA` record.

    Returns:
        list[str]: ``<db> <stage>: <reason>``, in stage order; empty when the build refuses nothing.
    """
    order = {name: index for index, name in enumerate(STAGE_NAMES)}
    stages = record.get("stages") or {}
    return [f"{db} {name}: {'; '.join(item.get('reasons') or [])}"
            for name in sorted(stages, key=lambda name: (order.get(name, len(order)), name))
            for db, item in sorted(stages[name].items()) if item.get("action") == "refuse"]
