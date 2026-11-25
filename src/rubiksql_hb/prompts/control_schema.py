"""Canonical content/history entities of the evolution workspace.

Content and intent are separate, immutable identities: a :class:`ReleaseEntity` is a
content key over its references, a :class:`ProposalEntity` is an author's hypothesis and plan
pinned to one release, and :class:`PromptEventEntity` rows are appended, never edited. No row
of the control workspace carries mutable lifecycle state; the lifecycle (active, superseded,
pinned, evaluated, rejected, ...) is derived from the authority events. The verified snapshot view
(:mod:`view_ws`) registers none of these entities; it carries only the BASELINE closure's
``sys-prompt`` rows.

Prompts are native ``sys-prompt`` rows. :class:`ToolEntity`, canonical :class:`GeneralRuleEntity`
versions (``format`` ``rubiksql.rule/1``) and canonical :class:`KnobSetEntity` parameter sets
(``format`` ``rubiksql.parameters/1``) are the authoritative typed versions a
``rubiksql.closure/2`` release references exactly (:mod:`entity_model`). GeneralRule and KnobSet
rows with a null ``format`` are historical projections of the immutable ``rules.general`` and
``config.knobs`` prompt carriers; only that projection is rebuilt by the lab refresh, and its
projection-only columns are empty on canonical versions. HeavenBase cannot relax a stored
required column, so those columns keep their required flag.

The lab uses these same canonical rows. Optional release/proposal convenience fields
remain declared for filter validation and historical schema compatibility; current canonical
writers leave them null, except ``ReleaseEntity.content_refs``, which holds the typed
references of a closure/2 release and is part of its identity.
"""
from __future__ import annotations

import heavenbase as hb

__all__ = [
    "CONTROL_ENTITIES",
    "EVENT_KINDS",
    "GeneralRuleEntity",
    "KnobSetEntity",
    "PromptEventEntity",
    "ProposalEntity",
    "ReleaseEntity",
    "ToolEntity",
    "TYPED_ENTITIES",
]

#: The event kinds :meth:`control.Control.record_event` accepts.
EVENT_KINDS = ("draft", "gate", "publish", "eval-reserve", "eval-claim", "eval-ready", "eval-attach", "outcome", "decision", "register", "authority", "export",
               "migration", "workspace-change", "toolkit-retirement")


class ReleaseEntity(hb.Entity):
    """One content-addressed combination release; its object_id is the release id (``r`` + 16 hex).

    Create-only: the refs decide the id, so the same closure published twice is the same row.
    A ``rubiksql.closure/1`` release lists every closure row, data carriers included, in ``refs``.
    A ``rubiksql.closure/2`` release lists only native prompt rows in ``refs`` and carries its exact
    Tool, Rule and parameter-set versions in ``content_refs`` (:mod:`entity_model`).
    """

    refs = hb.field(hb.Json, desc="Row name -> full 64-hex content key of every sys-prompt row in the closure").required()
    entries = hb.field(hb.Json, desc="Sorted names of the rows that are code-facing entries").required()
    rules = hb.field(hb.ShortText, desc="Closure rules version: rubiksql.closure/1 (prompt carriers) or "
                                        "rubiksql.closure/2 (typed tool, rule and parameter references)").required()
    reason = hb.field(hb.LongText, desc="Why the release exists").default("")
    created_at = hb.field(hb.Float, desc="Epoch seconds when this workspace recorded the canonical release").default(0.0)
    # Query-time alias fields (rubiksql_hb.lab_store); current canonical writers leave them null.
    origin = hb.field(hb.ShortText, desc="v0 (the restarted root), cutover, proposal or state (adopted current release); "
                                        "query-time alias").optional()
    parents = hb.field(hb.Array[hb.ShortText],
                       desc="Parent release ids of the proposals that produced it, base parents first; empty for v0 or "
                            "adopted state; "
                            "query-time alias").optional()
    proposals = hb.field(hb.Array[hb.ShortText], desc="The proposals that produced it, oldest first; query-time alias").optional()
    row_versions = hb.field(hb.Array[hb.ShortText],
                            desc="Every row of its closure as name:version, sorted; a closure/2 release adds "
                                 "tool:<name>:<version>, rule:<title>:<version>, parameters:knobs:<version>, "
                                 "rules:corpus:<16 hex> and tools:loader:<16 hex>; filter one with $contains; "
                                 "query-time alias").optional()
    content_refs = hb.field(hb.Json,
                            desc="closure/2: the canonical typed references, part of the release id: {schema "
                                 "rubiksql.release-refs/1, tools: [{entity, object_id, version, digest}] in serve order, "
                                 "loader: {modules, serializer} module Capsule pins, parameters: one knob-set-entity "
                                 "reference, rules: {items: ordered general-rule-entity references, corpus: corpus-wide "
                                 "metadata and layout, knowledge_digest}}. closure/1: null in storage; query_lab computes "
                                 "each closure row name -> {entity, object_ids, version, content_key} at read time").optional()
    status = hb.field(hb.ShortText,
                      desc="active (the BASELINE now), retired, superseded (a BASELINE before) or published (never "
                           "BASELINE); query-time alias").optional()
    baseline = hb.field(hb.Json,
                        desc="The pointer decisions (cutover, entity-model, promote, rollback) and current-state anchors that moved "
                             "BASELINE to or from it, oldest first: [{kind, from, to, decision, proposal, actor, at}]; "
                             "query-time alias").optional()
    knobs = hb.field(hb.Json,
                     desc="Its knob table: every knob entries.KNOBS declares, {value, source (release: its config.knobs "
                          "row or closure/2 parameter set; default: the code's default), kind, minimum, maximum, "
                          "default, env (the variable that "
                          "overrides it), roles (the keyspaces a non-default value moves), effect}; query-time alias").optional()
    store_commit = hb.field(hb.ShortText,
                            desc="Historical compatibility field; query-time aliases return an empty string").optional()


class ProposalEntity(hb.Entity):
    """One proposal: intent pinned to a release; its object_id is ``p`` + 16 hex of the canonical body.

    Create-only. Several proposals may point at the same release (two agents proposing the same
    change, a rollback reproducing an ancestor); their hypotheses and outcomes never overwrite
    each other.
    """

    parents = hb.field(hb.Json, desc="[{release, role: base|merge}] the proposal derives from").required()
    release = hb.field(hb.ShortText, desc="Release id the proposal pins").required()
    changes = hb.field(hb.Json, desc="Derived closure diff: prompt row name -> {from, to} content keys; for typed "
                                     "versions tool:<name>, rule:<title>, parameters:knobs, rules:corpus and "
                                     "tools:loader -> {from, to} digests").default({})
    reason = hb.field(hb.LongText, desc="One-sentence summary of the proposal").default("")
    hypothesis = hb.field(hb.Json, desc="claim, mechanism, expected, evidence, risks, falsifier").optional()
    hypothesis_digest = hb.field(hb.ShortText, desc="sha256 of the canonical hypothesis, part of the id").default("")
    # Named analysis_plan rather than the design's ``plan``: sqlite treats PLAN as a reserved word and HB warns on it.
    analysis_plan = hb.field(hb.Json, desc="Frozen analysis plan: metric, exposure, control").optional()
    plan_digest = hb.field(hb.ShortText, desc="sha256 of the canonical plan, part of the id").default("")
    author = hb.field(hb.ShortText, desc="Author label, e.g. agent:claude/evolver-3").default("")
    origin = hb.field(hb.ShortText, desc="cutover, entity-model (the representation-only typed "
                                         "migration), agent or human").default("human")
    gate = hb.field(hb.Json, desc="Summary of the offline gate result: passed, checks (name to status) and hb; in the "
                                  "lab also warnings (the checks that warned) and change_ratio when the loop recorded "
                                  "it").default({})
    created_at = hb.field(hb.Float, desc="Epoch seconds when the proposal was recorded").default(0.0)
    # Query-time alias fields (rubiksql_hb.lab_store); current canonical writers leave them null.
    base = hb.field(hb.ShortText, desc="Its base parent release; query-time alias").optional()
    generation = hb.field(hb.ShortText,
                          desc="The BASELINE its evaluation ran under (its chosen outcome's generation), else its base "
                               "parent; query-time alias").optional()
    row_names = hb.field(hb.Array[hb.ShortText],
                         desc="Every row it added, changed or removed, merge parents' included (typed keys tool:<name>, "
                              "rule:<title>, parameters:knobs, rules:corpus, tools:loader); filter one with $contains; "
                              "query-time alias").optional()
    row_versions = hb.field(hb.Array[hb.ShortText],
                            desc="The row versions it introduced as name:version (a removal has none); query-time "
                                 "alias").optional()
    content_refs = hb.field(hb.Json,
                            desc="The complete pinned release's content references, as ReleaseEntity.content_refs; "
                                 "changes and row_names identify which of these the proposal changed; query-time alias").optional()
    kinds = hb.field(hb.Array[hb.ShortText], desc="The kinds of row it changed: prompt, rules, knobs, generator, tool; query-time "
                                                  "alias").optional()
    target = hb.field(hb.ShortText, desc="The scientific target selected by the author; null when not supplied. "
                                         "Query-time alias").optional()
    verdict = hb.field(hb.ShortText,
                       desc="Its chosen outcome's verdict: promotable, adoptable, not-promotable or descriptive; null when never "
                            "evaluated; query-time alias").optional()
    gain_pp = hb.field(hb.Float, desc="EX gain of that outcome, child minus parent, in percentage points; query-time "
                                      "alias").optional()
    p_value = hb.field(hb.Float, desc="The one-sided McNemar p of that outcome, information only; query-time alias").optional()
    outcomes = hb.field(hb.Json,
                        desc="Every outcome, oldest first: [{id, verdict, verdict_reason, gain_pp, p, per_db (database "
                             "to effect_pp), integrity, source_verification, generation, run, parent_run, created_at, file}]; "
                             "source_verification is the original outcome's mode (immutable/1 for strict cache provenance), "
                             "null when the original did not record it; query-time alias").optional()
    experiments = hb.field(hb.Array[hb.ShortText],
                           desc="The lab experiment ids of its evaluated child runs (its outcomes' runs); query-time "
                                "alias").optional()
    decision = hb.field(hb.ShortText,
                        desc="What became of it: promoted, rolled-back, rejected (by a person or by the promotion rule), "
                             "abandoned, not-evaluated (a knowledge-abandoned arm included) or undecided (promotable/adoptable, no "
                             "promotion recorded); cutover or migrated (the representation-only entity-model "
                             "migration) for the unmeasured pre-baseline pointer decisions; query-time alias").optional()
    decided_by = hb.field(hb.ShortText,
                          desc="Who decided: the decision's actor (auto:outcome for an automatic promotion), rule for a "
                               "not-promotable verdict, agent or loop for a knowledge decision; query-time alias").optional()
    decided_at = hb.field(hb.Float, desc="Epoch seconds of that decision or verdict; query-time alias").optional()
    decision_reason = hb.field(hb.LongText, desc="Why, as the decision, the verdict or the knowledge decision says; query-time "
                                                 "alias").optional()
    files = hb.field(hb.Json,
                     desc="Proposal export path and related native event IDs: {proposal, events}; query-time alias").optional()
    store_commit = hb.field(hb.ShortText,
                            desc="Historical compatibility field; query-time aliases return an empty string").optional()


class ToolEntity(hb.Entity):
    """One immutable answering Tool version: interface, permissions and exact implementation pin.

    Create-only. ``name`` is the stable logical tool name; ``object_id`` is ``t`` + ``version``, and
    ``version`` is the first 16 hex of ``digest``, the sha256 of the canonical definition plus the
    implementation pin (:func:`rubiksql_hb.hb_tool_entities.tool_row`). A release selects versions by
    typed reference; the physical row count is history, not a release's callable inventory. Source
    text stays in the referenced inert module Capsule.
    """

    format = hb.field(hb.ShortText, desc="Row format, rubiksql.tool/1").required()
    name = hb.field(hb.ShortText, desc="Stable logical tool name the model or loop calls").required()
    version = hb.field(hb.ShortText, desc="First 16 hex of digest; object_id is t + version").required()
    digest = hb.field(hb.ShortText, desc="sha256 of the canonical definition and implementation pin").required()
    description = hb.field(hb.LongText, desc="Model-facing tool description").required()
    input_schema = hb.field(hb.Json, desc="Direct JSON input schema").required()
    output_schema = hb.field(hb.Json, desc="Direct JSON output schema; null when undeclared").optional()
    serve_order = hb.field(hb.Integer, desc="The definition's unique nonnegative serve order").required()
    visible_to = hb.field(hb.Array[hb.ShortText], desc="Roles whose model may call it: candidate, judge, reviewer").required()
    loop_visible_to = hb.field(hb.Array[hb.ShortText], desc="Roles whose framework loop may call it").required()
    routes = hb.field(hb.Json, desc="Role -> permitted routes; an omitted role permits every route").required()
    required_for = hb.field(hb.Array[hb.ShortText], desc="Roles that cannot disable it; a subset of visible_to").required()
    module = hb.field(hb.ShortText, desc="Implementation module, rubiksql_hb.tools.*").required()
    qualname = hb.field(hb.ShortText, desc="Top-level implementation function name").required()
    implementation = hb.field(hb.Json,
                              desc="Exact pin: capsule_id and source_hash of the module Capsule, closure [{module, sha256}], "
                                   "serializer_id, serializer_digest and knob_defaults").required()


class GeneralRuleEntity(hb.Entity):
    """One global rule: a canonical immutable rule version, or a historical corpus projection.

    A canonical version (``format`` ``rubiksql.rule/1``) has ``object_id`` ``g`` + ``version``, where
    ``digest`` hashes its six content fields, its own provenance and evolution records and its JSON
    layout; a closure/2 release orders these references and owns the corpus-wide metadata. A row with a
    null ``format`` is the historical projection of one rule of one ``rules.general`` corpus, keyed by
    the corpus content key and title and rebuilt from that immutable source. Projection-only columns are
    empty on canonical versions. No model is executed here.
    """

    title = hb.field(hb.ShortText, desc="Canonical rule title; the stable logical rule name").required()
    scope = hb.field(hb.ShortText, desc="Canonical rule scope; always global").required()
    desc = hb.field(hb.LongText, desc="The complete rule instruction").required()
    example = hb.field(hb.LongText, desc="The complete rule example").required()
    tags = hb.field(hb.Array[hb.ShortText], desc="Canonical rule tags").required()
    triggers = hb.field(hb.Array[hb.ShortText], desc="Canonical literal trigger phrases").required()
    corpus_version = hb.field(hb.ShortText, desc="Projection: the rules.general version, first 16 hex of content_key; "
                                                 "empty on canonical versions").required()
    content_key = hb.field(hb.ShortText, desc="Projection: full 64-hex content key of the containing rules.general corpus; "
                                              "empty on canonical versions").required()
    source_prompt_id = hb.field(hb.ShortText, desc="Projection: the sys-prompt object_id holding the complete source corpus; "
                                                   "empty on canonical versions").required()
    status = hb.field(hb.ShortText, desc="Projection: the source version's lifecycle (active, revoked, dropped, superseded or "
                                         "unpromoted); empty on canonical versions, whose lifecycle derives from releases").required()
    provenance = hb.field(hb.Json, desc="This title's original train provenance record; null when absent").optional()
    evolution = hb.field(hb.Json, desc="This title's dev_evolved_rules record; null when absent").optional()
    adjudications = hb.field(hb.Json, desc="Projection: original dev_evolution_adjudications naming this title; empty on "
                                           "canonical versions, whose adjudications are release corpus metadata").required()
    amendments = hb.field(hb.Json, desc="Projection: original generation.dev_measurement_amendments naming this title; empty "
                                        "on canonical versions").required()
    releases = hb.field(hb.Array[hb.ShortText], desc="Projection: all release ids that carry this exact corpus version; "
                                                     "empty on canonical versions").required()
    store_commit = hb.field(hb.ShortText, desc="Historical compatibility field; always an empty string").required()
    format = hb.field(hb.ShortText, desc="rubiksql.rule/1 for a canonical rule version; null for a historical "
                                         "projection row").optional()
    version = hb.field(hb.ShortText, desc="Canonical version: first 16 hex of digest; null for a projection row").optional()
    digest = hb.field(hb.ShortText, desc="Canonical version: sha256 of the rule content, records and layout; null for a "
                                         "projection row").optional()
    layout = hb.field(hb.Json, desc="Canonical version: original JSON member order of the rule, its provenance and its "
                                    "evolution record; null for a projection row").optional()


class KnobSetEntity(hb.Entity):
    """One parameter set: a canonical immutable version, or a historical ``config.knobs`` projection.

    A canonical version (``format`` ``rubiksql.parameters/1``) has ``object_id`` ``k`` + ``version`` and
    holds exactly the explicit parameter values of one release; knob types, ranges and defaults remain
    code declarations (:data:`rubiksql_hb.prompts.entries.KNOBS`). A row with a null ``format`` is the
    historical projection of one ``config.knobs`` prompt carrier.
    """

    name = hb.field(hb.ShortText, desc="knobs for a canonical parameter set; config.knobs for a projection row").required()
    version = hb.field(hb.ShortText, desc="Canonical: first 16 hex of digest; projection: the config.knobs version").required()
    content_key = hb.field(hb.ShortText, desc="Projection: full 64-hex content key of the source parameter set; empty on "
                                              "canonical versions").required()
    source_prompt_id = hb.field(hb.ShortText, desc="Projection: the sys-prompt object_id holding the source JSON; empty on "
                                                   "canonical versions").required()
    status = hb.field(hb.ShortText, desc="Projection: the source version's lifecycle (active, revoked, dropped, superseded or "
                                         "unpromoted); empty on canonical versions").required()
    values = hb.field(hb.Json, desc="The exact explicit parameter values; absent parameters use code defaults").required()
    declarations = hb.field(hb.Json,
                             desc="Projection: each declared knob's value, source, kind, minimum, maximum, default, env, roles "
                                  "and effect for the projecting code version; empty on canonical versions").required()
    releases = hb.field(hb.Array[hb.ShortText], desc="Projection: all release ids that carry this exact parameter version; "
                                                     "empty on canonical versions").required()
    store_commit = hb.field(hb.ShortText, desc="Historical compatibility field; always an empty string").required()
    format = hb.field(hb.ShortText, desc="rubiksql.parameters/1 for a canonical parameter set; null for a projection "
                                         "row").optional()
    digest = hb.field(hb.ShortText, desc="Canonical: sha256 of the parameter set; null for a projection row").optional()


class PromptEventEntity(hb.Entity):
    """One immutable publication, authority, accounting, decision or export event."""

    kind = hb.field(hb.ShortText, desc="Event kind, one of EVENT_KINDS").required()
    subject = hb.field(hb.ShortText, desc="name:version, release id or proposal id the event is about").required()
    actor = hb.field(hb.ShortText, desc="Who caused the event").default("")
    ts = hb.field(hb.Float, desc="Epoch seconds").default(0.0)
    payload = hb.field(hb.Json, desc="Event details").default({})


#: Entities registered in the control workspace.
CONTROL_ENTITIES = (ReleaseEntity, ProposalEntity, PromptEventEntity, ToolEntity)

#: The typed version entities a closure/2 release references, in reference-section order.
TYPED_ENTITIES = (ToolEntity, GeneralRuleEntity, KnobSetEntity)
