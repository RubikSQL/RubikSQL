"""Current role, event, generator and data-row declarations for HB releases.

Every role uses one document; optional candidate slots and tools.answering share
the same current vocabulary. Tools are excluded from model prompt keyspaces.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

__all__ = [
    "CANDIDATE_SLOTS",
    "CLOSURE_RULES",
    "DROPPED_ENTRIES",
    "ENTRIES",
    "GENERATORS",
    "KEYSPACES",
    "KNOBS",
    "KNOBS_ENTRY",
    "RELEASE_REQUIRED",
    "ROLE_DOCUMENTS",
    "ROUTES",
    "RULES_ENTRY",
    "SURFACES",
    "Contract",
    "Entry",
    "Knob",
    "Surface",
    "closure",
    "fired_surfaces",
    "knobs_text",
    "required_surfaces",
    "route_label",
    "slot_entry",
    "surfaces_of",
]

#: Version of the closure rules a release records. The keyspace hashes surface texts, not this string.
CLOSURE_RULES = "rubiksql.closure/1"
#: Prefix of the r0 layout's row names.
#: The one-document layout's role documents; each provides its role's system and first user message.
ROLE_DOCUMENTS = ("candidate", "judge", "reviewer")
#: Candidate slots a release may give a document of their own (``candidate.s1`` … ``candidate.s16``), counted from 1
#: in the order of the candidates spec. The judge lists at most 16 candidates, so a larger pool gains nothing more.
CANDIDATE_SLOTS = 16
#: The row holding the evolvable pipeline settings, and the row holding the general rule corpus.
KNOBS_ENTRY = "config.knobs"
RULES_ENTRY = "rules.general"

#: Routes per role; ``None`` is the default route.
ROUTES: Mapping[str, tuple[str | None, ...]] = {
    "candidate": (None, "grain"),
    "judge": (None,),
    "reviewer": (None,),
}

#: The four keyspaces, as ``(role, route)``.
KEYSPACES: tuple[tuple[str, str | None], ...] = tuple((role, route) for role, routes in ROUTES.items() for route in routes)


def route_label(role: str, route: str | None, slot: int | None = None) -> str:
    """Spell a keyspace as ``role`` or ``role/route``, and a candidate slot's own keyspace with ``[s<k>]`` after it.

    Args:
        role (str): ``candidate``, ``judge`` or ``reviewer``.
        route (str | None): Route name, or ``None`` for the default route.
        slot (int | None): The candidate slot whose document gives it a keyspace of its own.

    Returns:
        str: The label used in banners and reports, e.g. ``candidate/grain`` or ``candidate[s2]``.
    """
    label = role if route is None else f"{role}/{route}"
    return label if slot is None else f"{label}[s{slot}]"


@dataclass(frozen=True)
class Contract:
    """A check on the rendered text of a surface without parameters.

    Attributes:
        name (str): Short identifier printed with a violation.
        check (Callable[[str], str | None]): Returns a violation message, or ``None`` when satisfied.
        severity (str): ``"error"`` fails the load; ``"warning"`` is collected on the release and shown in the banner.
    """

    name: str
    check: Callable[[str], str | None]
    severity: str = "error"


@dataclass(frozen=True)
class Surface:
    """One model-visible text the code asks for.

    Attributes:
        id (str): ``<owner>.<name>``, e.g. ``judge.system``, ``candidate.nudge.time``, ``shared.synonyms``.
        params (Mapping[str, Any]): Parameter types in the spec type language (name -> type); the call site
            passes exactly these.
        visible (frozenset[tuple[str, str | None]]): Keyspaces whose closure contains this surface.
        contracts (tuple[Contract, ...]): Checks run on the rendered text when the surface has no parameters.
        fallback (str | None): The document the surface renders when no row of the release provides it (the r0
            layout's candidate first user message); it is then also the surface's identity text, the same text
            the one-document layout writes as the candidate document's second message. Every surface without a
            fallback must be provided by a row of every release, unless it is optional.
        optional (bool): A release may leave the surface out: a candidate slot's message (the slot then renders the
            shared candidate surface), the knobs and the rule corpus (defaults, or no corpus row), a generator
            document's message (the generator refuses a release without it).
        slot (int | None): The candidate slot whose document provides the surface, for ``candidate.s<k>.*``.
    """

    id: str
    params: Mapping[str, Any] = field(default_factory=dict)
    visible: frozenset[tuple[str, str | None]] = frozenset()
    contracts: tuple[Contract, ...] = ()
    fallback: str | None = None
    optional: bool = False
    slot: int | None = None

    @property
    def owner(self) -> str:
        """The first segment of the id: a role, or ``shared``.

        Returns:
            str: ``candidate``, ``judge``, ``reviewer`` or ``shared``.
        """
        return self.id.split(".", 1)[0]


@dataclass(frozen=True)
class Entry:
    """One current row name the code knows.

    Attributes:
        name (str): Current role, event, generator or data-row name.
        surfaces (tuple[str, ...]): Provided surface ids in message order.
            tools.answering provides none and joins no prompt keyspace.
    """

    name: str
    surfaces: tuple[str, ...]

    @property
    def params(self) -> dict[str, Any]:
        """Parameter types of the row: its surfaces' parameters in order (none for a fragment).

        Returns:
            dict[str, Any]: Name to type.
        """
        merged: dict[str, Any] = {}
        for surface in self.surfaces:
            merged.update(SURFACES[surface].params)
        return merged

    @property
    def visible(self) -> frozenset[tuple[str, str | None]]:
        """Keyspaces the row reaches: the union of its surfaces' (data rows reach none).

        Returns:
            frozenset[tuple[str, str | None]]: ``(role, route)`` pairs; empty for an optional entry, whose surfaces join
                no keyspace (a slot document reaches its slot's candidate keys through ``Release.keyspace``).
        """
        ids = self.surfaces
        return frozenset(pair for surface in ids for pair in SURFACES[surface].visible)

    @property
    def optional(self) -> bool:
        """Whether a release may leave the entry out: every surface it provides is optional.

        Returns:
            bool: ``True`` for a slot document, ``config.knobs``, ``rules.general`` and a generator document.
        """
        return self.name == "tools.answering" or (bool(self.surfaces) and all(SURFACES[surface].optional for surface in self.surfaces))

    @property
    def slot(self) -> int | None:
        """The candidate slot a slot document serves.

        Returns:
            int | None: ``k`` for ``candidate.s<k>``, else ``None``.
        """
        return SURFACES[self.surfaces[0]].slot if self.surfaces else None


def _all(*roles: str) -> frozenset[tuple[str, str | None]]:
    return frozenset((role, route) for role in roles for route in ROUTES[role])


def _only(*pairs: tuple[str, str | None]) -> frozenset[tuple[str, str | None]]:
    return frozenset(pairs)


_GRAIN = _only(("candidate", "grain"))
_EVERYWHERE = _all("candidate", "judge", "reviewer")

_CANDIDATE_RECORD = {"record": {"error": "str", "group_size": "int", "group_label": "str", "sql_fenced": "str", "n_rows": "int", "preview": "str"}}
_HIT_RECORD = {"record": {"syn": "str", "target": "str", "near": "bool"}}


def _must_not_contain(*needles: str) -> Callable[[str], str | None]:
    return lambda text: next((f"text mentions {needle!r}" for needle in needles if needle in text), None)


def _must_contain(needle: str) -> Callable[[str], str | None]:
    return lambda text: None if needle in text else f"text lacks {needle!r}"


#: The generator documents, one per distinct LLM call of the knowledge generators under ``scripts/``, each with the
#: parameter types of its system message and of its user message. The documents hold every sentence the call
#: sends; the data it carries — rows read from a database, hints, rules, demo examples — arrives through these
#: slots already formatted by the generator, one line or one block per item.
GENERATORS: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {
    # rulegen_general.py: mine database-agnostic rules from the BIRD train set
    "rules_general.hint_map": ({}, {"theme": "str", "n_dbs": "int", "hints": "str"}),
    "rules_general.map": ({}, {"pattern": "str", "n_examples": "int", "n_dbs": "int", "examples": "str"}),
    "rules_general.reduce": ({}, {"existing": "str", "candidates": "str", "max_rules": "int"}),
    # rulegen_judge.py: induce per-database rules from a pass-1 run, the judge's pick as pseudo-gold
    "rules_judge.map": ({}, {"db_id": "str", "questions": "str"}),
    "rules_judge.reduce": ({}, {"db_id": "str", "existing": "str", "candidates": "str", "max_rules": "int"}),
    # rulegen_hints.py: per-database rules from the database's own hints (a chunk of them, or all)
    "rules_hints": ({"whole_corpus": "bool"}, {"db_id": "str", "hints": "str", "gold": "str", "max_rules": "int"}),
    # rulegen_synonyms.py: user phrase -> table.column mined from the evidence notes
    "synonyms": ({}, {"db_id": "str", "schema": "str", "notes": "str"}),
    # propose_rule_triggers.py: literal trigger phrases for one curated rule
    "rule_triggers": ({}, {"rule": "str"}),
    # profile_tables.py: one table's profile, then the database overview
    "profile.table": ({}, {"db_map": "str", "agreement": "str", "detail": "str"}),
    "profile.db": ({}, {"db_map": "str", "agreement": "str", "profiles": "str"}),
    # setup_profiling.py: one column's interpretation slots, with the repair turn when ``violation`` is set
    "profile.columns": ({}, {"profile": "str", "columns": "str", "violation": "str"}),
    # inject_examples.py: rewrite one source example for the target database
    "examples.transfer": ({"examples": "str"}, {"query": "str", "schema": "str"}),
}

#: Every surface, keyed by id. Descriptions of what each text is live in the rows themselves
#: (``description``); this table only fixes the code-facing contract.
SURFACES: Mapping[str, Surface] = {
    surface.id: surface
    for surface in (
        Surface("candidate.system", {"max_rows": "int"}, _all("candidate")),
        Surface("candidate.user", {"question": "str", "synonyms": "str", "rule_hooks": "str"}, _all("candidate")),
        Surface("candidate.route.grain", {}, _only(("candidate", "grain"))),
        Surface("candidate.note.final_check", {}, _all("candidate")),
        Surface("candidate.nudge.time", {"mins": "int"}, _all("candidate")),
        Surface("candidate.nudge.submit", {}, _all("candidate")),
        Surface("candidate.nudge.rank_output", {}, _all("candidate"), optional=True),
        Surface("candidate.note.result_shape", {"limit": "int", "first": "int", "second": "int"}, _all("candidate"), optional=True),
        Surface("candidate.nudge.converge", {"n": "int"}, _all("candidate")),
        Surface("candidate.note.empty", {"n": "int"}, _all("candidate")),
        Surface("candidate.note.repeat_query", {"n": "int"}, _all("candidate")),
        Surface("shared.synonyms", {"hits": {"list": _HIT_RECORD}}, _all("candidate", "judge")),
        Surface("shared.rule_hooks", {"body": "str"}, _EVERYWHERE),
        Surface("shared.hint_conflict", {"kind": "str", "hint_literal": "str", "question_literal": "str"}, _EVERYWHERE, optional=True),
        Surface("shared.count_column", {"column": "str"}, _EVERYWHERE, optional=True),
        Surface("shared.rules_heading.triggered", {}, _EVERYWHERE),
        Surface("shared.rules_heading.baseline", {}, _all("judge")),
        Surface("shared.grain.missing_audit", {}, _GRAIN),
        Surface("shared.grain.percentage_scale", {}, _GRAIN),
        Surface("shared.grain.entity_fanout", {}, _GRAIN),
        Surface("shared.grain.entity_note", {"table_name": "str", "expression": "str", "pks": "str"}, _GRAIN),
        Surface(
            "judge.system",
            {},
            _all("judge"),
            contracts=(
                # r0 keeps this defect on purpose; a release that fixes it silences the warning.
                Contract("judge.system.hard_coded_64_rows", _must_not_contain("64 rows"), "warning"),
                # r0's two hard thresholds count five candidates, the pool it was written for; the pipeline's default
                # (run_official_pipeline.DEFAULT_CANDIDATES) and every evolution evaluation run three since 2026-09-26.
                # A release that restates them silences the warning.
                Contract("judge.system.hard_coded_five_candidates", _must_not_contain("five non-failed candidates", "4-of-5"), "warning"),
            ),
        ),
        Surface("judge.user", {"question": "str", "synonyms": "str", "candidates": {"list": _CANDIDATE_RECORD}, "rules": "str"}, _all("judge")),
        Surface("judge.nudge.submit", {}, _all("judge", "reviewer")),
        Surface("judge.note.literal_guard", {"novel": "str"}, _all("judge", "reviewer")),
        Surface(
            "reviewer.system",
            {},
            _all("reviewer"),
            contracts=(
                # r0 concatenates core and common without a separator; also kept on purpose.
                Contract("reviewer.system.missing_separator", _must_not_contain("into SQL.## Shared rules"), "warning"),
                # vote.py's import-time guard: the NO_COLINFO path drops the tool without rewriting this text.
                Contract("reviewer.system.get_col_info_free", _must_not_contain("get_col_info")),
            ),
        ),
        Surface("reviewer.user", {"question": "str", "sql": "str", "row_count": "int", "preview": "str"}, _all("reviewer")),
        *(
            Surface(f"candidate.s{slot}.{message}", params, optional=True, slot=slot)
            for slot in range(1, CANDIDATE_SLOTS + 1)
            for message, params in (("system", {"max_rows": "int"}), ("user", {"question": "str", "synonyms": "str", "rule_hooks": "str"}))
        ),
        Surface(KNOBS_ENTRY, optional=True),
        Surface(RULES_ENTRY, optional=True),
        *(
            Surface(f"gen.{generator}.{message}", params, optional=True)
            for generator, messages in GENERATORS.items()
            for message, params in zip(("system", "user"), messages)
        ),
    )
}

#: Surfaces an event-time row provides one-to-one, under its own id.
_EXTRAS = tuple(sid for sid, surface in SURFACES.items() if not surface.optional and not sid.endswith((".system", ".user")))

#: Every row name the code knows, keyed by name: the current document layout.
ENTRIES: Mapping[str, Entry] = {
    entry.name: entry
    for entry in (
        *(Entry(role, (f"{role}.system", f"{role}.user")) for role in ROLE_DOCUMENTS),
        *(Entry(surface, (surface,)) for surface in _EXTRAS),
        Entry("candidate.nudge.rank_output", ("candidate.nudge.rank_output",)),
        *(Entry(name, (name,)) for name in ("candidate.note.result_shape", "shared.hint_conflict", "shared.count_column")),
        *(Entry(f"candidate.s{slot}", (f"candidate.s{slot}.system", f"candidate.s{slot}.user")) for slot in range(1, CANDIDATE_SLOTS + 1)),
        Entry("tools.answering", ()),
        Entry(KNOBS_ENTRY, (KNOBS_ENTRY,)),
        Entry(RULES_ENTRY, (RULES_ENTRY,)),
        *(Entry(f"gen.{generator}", (f"gen.{generator}.system", f"gen.{generator}.user")) for generator in GENERATORS),
    )
}
#: Entry names the code no longer renders but retained releases may still list; none in the current epoch.
DROPPED_ENTRIES: frozenset[str] = frozenset()

#: Optional entries every release proposed from now on carries (the gate's ``required_rows`` check): the knobs,
#: the rule corpus and one document per generator call. Slot documents stay optional in every release.
RELEASE_REQUIRED: tuple[str, ...] = (KNOBS_ENTRY, RULES_ENTRY, *(f"gen.{generator}" for generator in GENERATORS))

# ---------------------------------------------------------------------------------------
# Knobs


@dataclass(frozen=True)
class Knob:
    """One evolvable pipeline setting of the ``config.knobs`` row.

    Attributes:
        name (str): The key in the row's JSON object, e.g. ``max_turns``.
        kind (str): ``int``, ``float``, ``bool``, ``ints`` (a non-empty, strictly ascending list of ints) or
            ``choice`` (one of the strings ``choices`` lists).
        default (Any): The value the code ran with before the knobs moved into the release, and the value of a knob
            a release does not carry. It never changes: a new value belongs to a release's ``config.knobs``.
        minimum (float | None): The smallest allowed value (of each element, for ``ints``).
        maximum (float | None): The largest allowed value (of each element, for ``ints``).
        visible (frozenset[tuple[str, str | None]]): The keyspaces whose cache keys a value other than the default
            moves: the roles whose runs the knob changes.
        env (str | None): The environment variable that still overrides the release value, for experiments.
        parse (Callable[[str], Any] | None): How the code has always read that variable (an unparsable value raises
            exactly as it did).
        effect (str): What the knob controls, one line; the dossier's knob table prints it.
        post_review_answer_hook (tuple[str, str] | None): A deterministic answer hook after review, declared
            as its ``judge_debug`` record key and that record's pre-hook producer SQL field. Hooks run in
            :data:`KNOBS` declaration order. Attribution retains reviewer turn and answer credit; a skipped
            reviewer uses the judge's producer SQL check, read from the first recorded hook.
        choices (tuple[str, ...]): The values a ``choice`` knob allows; empty for every other kind.
        selects_model (bool): The value is the model of the roles the knob is visible to. Their whole and
            turn-memo base cache keys hash the model, so after a change no unit of those roles can replay its
            parent's turns; an outcome settles such a unit's missing replay source as the knob's divergence
            (``evolve._settled_cause``), as it settles a knowledge change's.
    """

    name: str
    kind: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    visible: frozenset[tuple[str, str | None]] = frozenset()
    env: str | None = None
    parse: Callable[[str], Any] | None = None
    effect: str = ""
    post_review_answer_hook: tuple[str, str] | None = None
    choices: tuple[str, ...] = ()
    selects_model: bool = False

    def normalize(self, value: Any) -> Any:
        """The value as the code uses it: a ``float`` for a float knob, a tuple for ``ints``.

        Args:
            value (Any): A value that passed :meth:`violation`.

        Returns:
            Any: The normalized value.
        """
        if self.kind == "float":
            return float(value)
        if self.kind == "ints":
            return tuple(int(item) for item in value)
        return value

    def violation(self, value: Any) -> str | None:
        """Check a release value against the knob's type and range.

        Args:
            value (Any): The JSON value the release carries.

        Returns:
            str | None: What is wrong with it, or ``None`` when it is allowed.
        """
        numbers: list[Any]
        if self.kind == "bool":
            return None if isinstance(value, bool) else f"{self.name} must be true or false, got {value!r}"
        if self.kind == "choice":
            return None if isinstance(value, str) and value in self.choices else f"{self.name} must be one of {list(self.choices)}, got {value!r}"
        if self.kind == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                return f"{self.name} must be an integer, got {value!r}"
            numbers = [value]
        elif self.kind == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return f"{self.name} must be a number, got {value!r}"
            if not math.isfinite(value):   # JSON's NaN compares false with every bound, so the range alone admits it
                return f"{self.name} must be a finite number, got {value!r}"
            numbers = [value]
        elif self.kind == "ints":
            if not isinstance(value, list) or not value or len(value) > _KNOB_LIST_MAX or any(not isinstance(item, int) or isinstance(item, bool) for item in value):
                return f"{self.name} must be a list of 1 to {_KNOB_LIST_MAX} integers, got {value!r}"
            if any(later <= earlier for earlier, later in zip(value, value[1:])):
                return f"{self.name} must be strictly ascending, got {value!r}"
            numbers = list(value)
        else:
            raise ValueError(f"knob {self.name}: unknown kind {self.kind!r}")
        if (self.minimum is not None and min(numbers) < self.minimum) or (self.maximum is not None and max(numbers) > self.maximum):
            return f"{self.name}={value!r} is outside its range [{self.minimum}, {self.maximum}]"
        return None


#: The longest list an ``ints`` knob may hold.
_KNOB_LIST_MAX = 8


def _env_flag(raw: str) -> bool:
    """An on/off variable read as the loops always read it: anything but ``0`` is on."""
    return raw != "0"


#: The values the ``judge_model`` knob allows, its default first (user decisions 2026-09-26 and 2026-10-23).
_JUDGE_MODELS = ("qwen:qwen3.7-max", "qwen:qwen3.8-max")

#: The evolvable pipeline settings, keyed by name. Each default is the value the code used before the knobs moved
#: into the release, so a release that carries the defaults behaves, keys and fingerprints exactly as before
#: (``symbol_near`` joined on 2026-09-27 with the 0.40 ``tools.schema`` had held as a constant;
#: ``judge_model`` joined on 2026-10-23 with the judge every earlier run used);
#: infrastructure settings (timeouts, concurrency, rate limits, paths, providers and models) are not knobs, except
#: the judge's model, which the user made a release setting so a model change is measured and adopted like any
#: other change. Some knobs are also stated in prompt text — the judge document says ``top-15`` and ``64 rows``,
#: the candidate and judge documents recommend ``top_k=12`` — and a proposal that moves such a knob edits that text
#: with it.
KNOBS: Mapping[str, Knob] = {
    knob.name: knob
    for knob in (
        Knob("run_sql_max_rows", "int", 64, 1, 100_000, _EVERYWHERE, "RUBIKSQL_RUN_SQL_MAX_ROWS", int,
             "rows a run_sql / submit result shows the candidate, and a candidate preview shows the judge and the reviewer"),
        Knob("max_turns", "int", 200, 1, 10_000, _all("candidate"), "RUBIKSQL_MAX_TURNS", int,
             "the candidate loop's runaway guard: model turns before it gives up"),
        Knob("max_submit_nudges", "int", 2, 0, 20, _all("candidate"), "RUBIKSQL_MAX_SUBMIT_NUDGES", int,
             "times a candidate that stops without submitting is re-prompted"),
        Knob("time_nudge_sec", "int", 1200, 0, 86_400, _all("candidate"), "RUBIKSQL_TIME_NUDGE_SEC", lambda raw: int(raw or "0"),
             "seconds after which a candidate is told to submit its best SQL now; 0 turns the nudge off"),
        Knob("loop_guard", "bool", True, visible=_all("candidate"), env="RUBIKSQL_LOOP_GUARD", parse=_env_flag,
             effect="the convergence nudge at the tool-call tiers and the repeated-query note"),
        Knob("loop_guard_tiers", "ints", [50, 80], 1, 10_000, _all("candidate"), "RUBIKSQL_LOOP_GUARD_TIERS", lambda raw: [int(item) for item in raw.split(",")],
             "tool-call counts at which the convergence nudge fires"),
        Knob("empty_probe", "bool", True, visible=_all("candidate"), env="RUBIKSQL_EMPTY_PROBE", parse=_env_flag,
             effect="the note that asks which conjunct empties a result"),
        Knob("empty_probe_at", "int", 2, 1, 1_000, _all("candidate"), "RUBIKSQL_EMPTY_PROBE_AT", int,
             "the empty run_sql result on which the empty-result note fires"),
        # An empty RUBIKSQL_SYNONYM_NEAR read as 0.65 until BASELINE moved the floor (2026-09-27); it would now override
        # the release with text-embedding-v3's floor, so an empty variable is refused, as an empty RUBIKSQL_SYMBOL_NEAR is.
        Knob("synonym_near", "float", 0.65, 0.0, 1.0, _all("candidate", "judge"), "RUBIKSQL_SYNONYM_NEAR", float,
             "cosine floor of the semantic synonym pass; 0 turns it off"),
        Knob("symbol_near", "float", 0.40, 0.0, 1.0, _EVERYWHERE, "RUBIKSQL_SYMBOL_NEAR", float,
             "cosine floor at which a glyph-only enum value ('=', '#', '∞') enters search_values results, scored by cosine alone"),
        # No environment override: a non-diagnostic run's judge is exactly this value (judge_policy.require_judge).
        Knob("judge_model", "choice", _JUDGE_MODELS[0], visible=_all("judge", "reviewer"),
             effect=f"the judge's model, which the reviewer follows: {' or '.join(_JUDGE_MODELS)}",
             choices=_JUDGE_MODELS, selects_model=True),
        Knob("reviewer", "bool", True, visible=_all("reviewer"), env="RUBIKSQL_REVIEWER", parse=lambda raw: (raw or "2") == "2",
             effect="the reviewer pass over the judge's answer (the variable keeps its old spelling: 2 on, anything else off)"),
        # Post-review answer hooks run in this declaration order (vote._vote_async), and attribution
        # reads the first recorded hook's pre-hook SQL as the producer SQL (evolve._post_review_answer).
        Knob("consensus_guard", "bool", False, visible=_all("reviewer"),
             effect="deterministic post-review return to the unique largest agreeing candidate group when the final "
                    "departs from it by adding a table or a de-duplication step",
             post_review_answer_hook=("consensus_guard", "pre_guard_sql")),
        Knob("consensus_guard_strict_added_tables", "bool", False, visible=_all("reviewer"),
             effect="when consensus_guard is enabled, bypass its hint-column exemption for added tables",
             post_review_answer_hook=("consensus_guard", "pre_guard_sql")),
        Knob("submission_gate", "bool", False, visible=_all("reviewer"),
             effect="deterministic post-review submission rules",
             post_review_answer_hook=("submission_gate", "pre_gate_sql")),
        Knob("submission_gate_left_inner_min_support_margin", "int", 1, minimum=1, visible=_all("reviewer"),
             effect="minimum successful INNER candidate support minus chosen-result support for LEFT-to-INNER submission",
             post_review_answer_hook=("submission_gate", "pre_gate_sql")),
        Knob("submission_gate_hint_aggregate", "bool", False, visible=_all("reviewer"),
             effect="prefer an otherwise identical saved candidate preserving a directly hinted SUM or AVG",
             post_review_answer_hook=("submission_gate", "pre_gate_sql")),
        Knob("count_weight", "bool", False, visible=_all("reviewer"),
             effect="deterministic post-review uncued joined-row counts with a candidate-result veto",
             post_review_answer_hook=("count_weight", "pre_hook_sql")),
        Knob("selection_normalization", "bool", False, visible=_all("reviewer"),
             effect="deterministic post-review bare selection casts and uncued single-row selection normalization",
             post_review_answer_hook=("selection_normalization", "pre_hook_sql")),
        Knob("entity_percentage_count", "bool", False, visible=_all("reviewer"),
             effect="deterministic post-review entity percentages with a row-count denominator",
             post_review_answer_hook=("entity_percentage_count", "pre_hook_sql")),
        Knob("rules_top_k", "int", 20, 1, 500, _EVERYWHERE,
             effect="rules get_rules returns when the caller passes no k"),
        Knob("examples_top_k", "int", 12, 1, 500, _EVERYWHERE,
             effect="menu rows get_examples returns when the caller passes no top_k (the documents recommend top_k=12)"),
        Knob("examples_inline_sql", "bool", False, visible=_EVERYWHERE,
             effect="return retained example SQL instead of sql_length in get_examples, at the unchanged retrieval width"),
        Knob("judge_baseline_rules", "int", 15, 1, 100, _all("judge"),
             effect="baseline rules injected into the judge's first message (the judge document says top-15)"),
    )
}


def knobs_text(values: Mapping[str, Any]) -> str:
    """The one text a ``config.knobs`` row may hold for a set of values: sorted keys, two-space indent.

    Args:
        values (Mapping[str, Any]): Knob name to JSON value.

    Returns:
        str: ``json.dumps(values, indent=2, sort_keys=True, ensure_ascii=False)``; the loader refuses any other
            spelling of the same values, so equal knobs always have the same content key.
    """
    return json.dumps(dict(values), indent=2, sort_keys=True, ensure_ascii=False)


def closure(role: str, route: str | None) -> tuple[str, ...]:
    """Surfaces visible in one keyspace, derived mechanically from the visibility declarations.

    Args:
        role (str): ``candidate``, ``judge`` or ``reviewer``.
        route (str | None): Route name, or ``None`` for the default route.

    Returns:
        tuple[str, ...]: Sorted surface ids.

    Raises:
        KeyError: If ``(role, route)`` is not one of the four keyspaces.
    """
    if (role, route) not in KEYSPACES:
        raise KeyError(f"unknown keyspace {route_label(role, route)!r}; known: {[route_label(*pair) for pair in KEYSPACES]}")
    return tuple(sorted(surface.id for surface in SURFACES.values() if (role, route) in surface.visible))


def required_surfaces(surfaces: Iterable[Surface] = SURFACES.values()) -> tuple[str, ...]:
    """Ids of the surfaces a row of every release must provide: all but those the code's fallback renders and the optional ones.

    Args:
        surfaces (Iterable[Surface]): Surface table to filter; the module table by default.

    Returns:
        tuple[str, ...]: Sorted ids.
    """
    return tuple(sorted(surface.id for surface in surfaces if surface.fallback is None and not surface.optional))


def slot_entry(slot: int) -> str:
    """The name of a candidate slot's document.

    Args:
        slot (int): The slot, counted from 1 in the order of the candidates spec.

    Returns:
        str: ``candidate.s<slot>``; a release carries it only for slots up to :data:`CANDIDATE_SLOTS`.
    """
    return f"candidate.s{slot}"


def surfaces_of(names: Iterable[str]) -> frozenset[str]:
    """The surfaces a set of row names provide; names the table does not know provide none.

    Consumers that read ``fired`` sets or change sets compare surfaces, so a run recorded under one
    layout and a release of the other meet on the same vocabulary.

    Args:
        names (Iterable[str]): Row names (entries of any layout) or surface ids.

    Returns:
        frozenset[str]: Surface ids.
    """
    found: set[str] = set()
    for name in names:
        entry = ENTRIES.get(name)
        if entry is not None:
            found.update(entry.surfaces)
        elif name in SURFACES:
            found.add(name)
    return frozenset(found)


def fired_surfaces(names: Iterable[str]) -> frozenset[str]:
    """The surfaces a run rendered, read from the row names its ``fired`` set records.

    A run renders its role's system message and its first user message together, but a surface the code's
    fallback renders records no row — the r0 layout's candidate first user message. A fired set whose rows
    provide ``<role>.system`` therefore also rendered that role's fallback surfaces, so a change to such a
    surface reaches the runs recorded under either layout. Change sets are read with :func:`surfaces_of`.

    Args:
        names (Iterable[str]): Row names of a ``fired`` set (entries of any layout) or surface ids.

    Returns:
        frozenset[str]: Surface ids, the fallback surfaces the role's system message implies included.
    """
    found = set(surfaces_of(names))
    found.update(surface.id for surface in SURFACES.values() if surface.fallback is not None and f"{surface.owner}.system" in found)
    return frozenset(found)
