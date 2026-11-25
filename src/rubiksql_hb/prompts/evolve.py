"""Scientific proposal, pairing, source-integrity and native authority accounting.

Local drafting uses a detached projection. Attempts, outcomes and decisions are
written only by Authority under its exclusive instance-bound lock; execution
consumes immutable snapshots. The host owns gold execution and EX scoring.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from . import entity_model as EM
from . import entries as E
from . import lineage as L
from . import metrics as M
from . import rules as rules_module
from .. import cache_lineage
from ..evaluation_questions import EVOLUTION_SET, QUESTION_SETS
from ..knowledge_failures import degraded_record
from .canon import canon, sha256_hex
from .corpus import read_rows
from .gate import PLAN_METRICS, GateReport, changed_surfaces, keyspace_moves, run_gate
from .policy import PROPOSAL_RE, RELEASE_RE, VERSION_RE, PolicyError, check_name
from .rules import RuleCorpusError, RulesChange, corpus_text
from .store import (
    POINTER_KINDS,
    SCHEMA_DECISION,
    SCHEMA_PROPOSAL,
    SCHEMA_RELEASE,
    PromptStoreError,
    Store,
    _bytes,
    proposal_id,
    release_document_id,
    release_id,
    write_if_absent,
)

__all__ = [
    "AUTO_ACTOR",
    "CAUSE_IMPLEMENTATION",
    "CAUSE_KNOB",
    "CAUSE_KNOWLEDGE",
    "CAUSE_NO_SOURCE",
    "DEFAULT_PLAN",
    "EXPOSURES",
    "HARNESS_MARKERS",
    "HUMAN_ONLY_VERBS",
    "INFO_METRICS",
    "INTEGRITY_MAX_DILUTION_RATE",
    "INTEGRITY_MAX_VIOLATION_RATE",
    "KNOWLEDGE_SURFACES",
    "MAX_ATTEMPTS_PER_GENERATION",
    "PROMOTE_MAX_P",
    "PROMOTE_MIN_GAIN_PP",
    "PROMOTE_QUESTION_SETS",
    "SCHEMA_ATTEMPT",
    "SCHEMA_DEEP_VERIFY",
    "SCHEMA_OUTCOME",
    "SCHEMA_RUN_MANIFEST",
    "VERDICTS",
    "DocChange",
    "EvolveError",
    "KnobsChange",
    "OutcomeComputation",
    "Proposal",
    "ProposeResult",
    "RemoveChange",
    "RulesChange",
    "RunView",
    "SpecChange",
    "TextChange",
    "ToolsChange",
    "attempt_marker_path",
    "changed_knobs",
    "common_ancestor",
    "deep_verify_outcome",
    "ensure_human",
    "find_parent_run",
    "format_gain",
    "format_p",
    "generation_attempts",
    "information_drift",
    "integrity_state",
    "knowledge_differences",
    "latest_verdict",
    "merge_refs",
    "merge_sections",
    "misfiled_outcomes",
    "authority_outcome",
    "validate_proposal",
    "changed_kinds",
    "outcome_id",
    "outcomes_of",
    "release_measurement_blocker",
    "plan_view",
    "promotion_rule",
    "promotion_veto",
    "promotion_verdict",
    "propose",
    "question_set_record",
    "read_run",
    "replays_parent_unit",
    "resolve_parent",
    "same_run_release",
    "store_lock",
    "verdict_gate",
    "verify_outcome",
]

SCHEMA_OUTCOME = "rubiksql.outcome/1"
SCHEMA_RUN_MANIFEST = "rubiksql.prompt-manifest/1"
SCHEMA_DEEP_VERIFY = "rubiksql.deep-verify/1"
SCHEMA_ATTEMPT = "rubiksql.attempt/1"
RUN_MANIFEST_NAME = ".prompt_manifest.json"
#: Store directory of the attempt markers: ``attempts/<generation>/<outcome id>.json``, one per evaluation.
ATTEMPTS_DIR = "attempts"
#: Exposure predicates a plan may name: questions whose child units diverged on the prompt, or whose
#: units (either arm) rendered an entry the proposal changed.
EXPOSURES = ("diverged:prompt", "fired:changed")
#: Replay cause for an unavailable source or an exhausted matching interrupted source. The memo's source
#: need not be the paired parent: a system-prompt change moves the base key, and a retry can replay an
#: earlier attempt under the child's key. The pairing settles it from the recorder's first-request
#: digests (``msg``): a differing first request is parent-relative ``initial``; the memo's ``diverged_at``
#: remains its source-local first miss. An identical first request leaves the unit *unverified*:
#: neither prompt-diverged nor eligible for the integrity check. The pairing also uses this bucket for a
#: missing cause without a source and counters proving a complete replay; live calls can record no cause.
CAUSE_NO_SOURCE = "no-source"
#: Settled cause of a child unit that could not replay because the proposal changed the knowledge: the rule corpus
#: or a generator document (:data:`KNOWLEDGE_SURFACES`). They move every database's knowledge fingerprint, which the
#: turn memo's base key hashes, so no unit replays against its parent — the proposal's own effect, like the
#: ``initial`` divergence of a system-prompt change, never noise.
CAUSE_KNOWLEDGE = "knowledge"
#: The surfaces whose change moves the knowledge a build writes, and so every knowledge fingerprint.
KNOWLEDGE_SURFACES = frozenset(sid for sid in E.SURFACES if sid == E.RULES_ENTRY or sid.startswith("gen."))
#: Settled cause of a child unit a knob change moved (:func:`_settled_cause`). A knob value other than the default joins
#: the keyspaces of the roles it affects but no text a unit renders, so the child replays its parent's turns until the
#: knob acts: a nudge that fires at another time (``clock``), a note that appears or does not (``unknown``), a tool
#: result of another size (``tool``), a turn cap or nudge budget that ends the run where the parent's went on, or a
#: reviewer that no longer runs (an undiverged unit whose final SQL changed). In the roles the changed knobs affect,
#: those are the proposal's own effect, like a system-prompt change's ``initial`` divergences, never noise. A knob
#: that selects its roles' model (``entries.Knob.selects_model``, the judge's ``judge_model``) moves their turn-memo
#: base key instead, so none of their units replays at all: their ``no-source`` units are the knob's effect too, as a
#: knowledge change's are :data:`CAUSE_KNOWLEDGE`.
CAUSE_KNOB = "knob"

#: A proven change in the paired unit's declared execution context (``roles/1`` outcomes only).
CAUSE_IMPLEMENTATION = "implementation"
_IMPLEMENTATION_ATTRIBUTION = "roles/1"
#: Plan fields every proposal freezes; the author's plan is merged over them (``gate.PLAN_FIELDS`` is the whole set).
DEFAULT_PLAN: Mapping[str, Any] = {
    "metric": "ex",
    "exposure": "diverged:prompt",
    "control": {"namespace": "", "max_age_days": 14},
}
#: Promotion rule (policy, not a plan field): the smallest EX gain, child minus parent in absolute percentage points
#: over the complete evolution question set, that promotes: 0.5pp requires 3 net questions of the 500 (2 is 0.40pp). No
#: significance test guards it (the user's decision of 2026-09-25); the p-value remains descriptive.
PROMOTE_MIN_GAIN_PP = 0.5
#: Promotion rule: the largest exact one-sided McNemar p-value (the child wins more discordant pairs than a fair coin
#: would give it) that promotes, or ``None``: no significance test. The rule in force has none; every outcome still
#: records its p-value, as information. An outcome keeps the threshold it recorded and verifies under it (outcomes of
#: the first rule record 0.05).
PROMOTE_MAX_P: float | None = None
#: The evolution question set's ``sha256:``, as ``eval_bird`` records it in ``identity.question_set``: the 500 questions
#: of the official 2024-06-27 Mini-Dev input.
EVOLUTION_QUESTION_SET = EVOLUTION_SET.digest
#: Promotion rule: the full question sets, by the ``identity.question_set`` a run manifest records (``sha256:`` of the
#: queries file, ``scripts/eval_bird.py``), with the number of questions each holds. A pairing takes a verdict only when
#: both runs scored every question their manifests record (``questions``) and the queries file is one of these: the
#: evolution question set (:data:`EVOLUTION_QUESTION_SET`, 500 questions) and nothing else. A run over part of it
#: (``--db``, ``--ids``, ``--limit``) or over another queries file (the 1534-question dev, the archived 330, a hand-made
#: file) is ``descriptive`` (``partial run``): a pilot, never a promotion.
PROMOTE_QUESTION_SETS: Mapping[str, int] = {EVOLUTION_QUESTION_SET: EVOLUTION_SET.count}
#: Spend guard: evaluations a BASELINE generation takes before :func:`outcome` refuses to compute another verdict
#: (``over cap``: nothing is computed or recorded) unless a person passes ``allow_over_cap``. Every run pair that takes
#: a verdict counts once, when :func:`outcome` first computes it — a dry run included — by its marker under
#: :data:`ATTEMPTS_DIR`; descriptive pairs do not count, and a promotion opens a new generation. It binds what
#: ``prompts outcome`` computes in the checkout that holds the markers; b and c read off the run directories by other
#: means escape it.
MAX_ATTEMPTS_PER_GENERATION = 10
#: Actor of the ``promote`` decision :func:`outcome` writes when a promotable outcome promotes.
AUTO_ACTOR = "auto:outcome"
#: Metrics every paired outcome reports beside EX, as information: they never decide a verdict.
INFO_METRICS = tuple(name for name in M.METRICS if name != "ex")
#: Environment markers of known agent harnesses; a human-only verb refuses when one is present, and a decision's default
#: actor names the first one (``agent:<marker>``).
HARNESS_MARKERS = ("RUBIKSQL_AGENT_HARNESS", "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CODEX_SANDBOX", "CODEX_THREAD_ID", "CURSOR_AGENT", "GEMINI_CLI")
#: Setting this variable to ``1`` bypasses the harness check; it exists for a person running under an agent harness.

#: Verbs only a human runs; promoting against the verdict (``promote --override-verdict``), computing past the attempt
#: cap (``outcome --allow-over-cap``) and launching the paid evolution loop (``evolve-loop`` without ``--no-eval``) are
#: human-only too. ``v0-init`` moves BASELINE to a new parentless root, so it is a person's decision as well.
HUMAN_ONLY_VERBS = ("rollback", "reject", "retire", "unretire", "revoke", "unrevoke", "register", "unregister", "v0-init")
#: Fraction of undiverged units whose final SQL may differ before the outcome is ``invalid``.
INTEGRITY_MAX_VIOLATION_RATE = 0.01
#: Fraction of paired units that may be diluted before the outcome is ``invalid`` (non-prompt divergences:
#: ``tool``, ``clock``, ``unknown``) or ``unverifiable`` (``no-source`` units nothing could be replayed against).
INTEGRITY_MAX_DILUTION_RATE = 0.01
#: Text of the deep-verification problem that a recorded verification may stand in for (:func:`promote`).
_NOT_ON_THIS_MACHINE = "is not on this machine"
_DAY = 86400.0
_ROW_REF_RE = re.compile(r"^(?P<name>[a-z][a-z0-9._-]*):(?P<version>[0-9a-f]{16})$")
_OUTCOME_ID_RE = re.compile(r"^o[0-9a-f]{16}$")
#: The closure key a typed op is recorded under (:func:`entity_model.closure_view`): ``op: knobs`` moves the parameter
#: set, ``op: tools`` the tool section with its loader pins; :func:`entity_model.change_kind` classifies both.
_TYPED_OPS = {"knobs": "parameters:knobs", "tools": "tools:loader"}
#: The typed sections of a closure/2 release that a merge combines as a whole.
_TYPED_SECTIONS = ("tools", "parameters", "rules")
#: How a closure/2 release changes what a closure/1 release carried in a prompt carrier.
_CARRIER_OPS = {
    E.KNOBS_ENTRY: f"{E.KNOBS_ENTRY}: a {EM.CLOSURE_V2} release keeps its knobs in a typed parameter set, not a text row; "
                   "change them with {\"op\": \"knobs\", \"set\": {\"<knob>\": <value>}, \"reason\": \"...\"}",
    E.RULES_ENTRY: f"{E.RULES_ENTRY}: a {EM.CLOSURE_V2} release keeps its rule corpus as typed rule versions, not a text row; "
                   f"change it with one {{\"op\": \"rules\", \"add\": [...], \"replace\": [...], \"remove\": [...]}}",
    "tools.answering": f"tools.answering: a {EM.CLOSURE_V2} release keeps its tools as typed Tool versions, not a text row; "
                       "change them with {\"op\": \"tools\", \"reason\": \"...\"} and --source-commit or --tool-source",
}


class EvolveError(RuntimeError):
    """An author-side or decision step was refused; the message names the step and the rule."""


# ---------------------------------------------------------------------------------------
# Guards and locks


def ensure_human(verb: str, env: Mapping[str, str] | None = None) -> None:
    """Refuse a human-only verb under a known agent harness.

    Args:
        verb (str): The verb being run.
        env (Mapping[str, str] | None): Environment to inspect; ``os.environ`` by default.

    Returns:
        None: This function does not return a value.

    Raises:
        EvolveError: If a harness marker is set and ``RUBIKSQL_PROMPT_HUMAN`` is not ``1``. The message does not
            name the override: the check exists to stop an agent from running a decision verb by habit, not to be
            advertised to it.
    """
    present = _harness_markers(env)
    if present:
        raise EvolveError(
            f"{verb} is a human-only verb and an agent harness marker is set ({', '.join(present)}); "
            "a person runs it in their own shell"
        )


def _harness_markers(env: Mapping[str, str] | None) -> list[str]:
    """The agent harness markers set in ``env`` (``os.environ`` by default); none when a person set the override."""
    environment = env if env is not None else os.environ
    return [marker for marker in HARNESS_MARKERS if environment.get(marker)]


def _lock_path(package: Path) -> Path:
    # Keep the established lock namespace so old and renamed clients serialize the same store.
    return Path(tempfile.gettempdir()) / f"rubiksql-prompts-{sha256_hex(str(Path(package).resolve()))[:16]}.lock"


@contextlib.contextmanager
def store_lock(package: Path) -> Iterator[None]:
    """Hold an exclusive file lock for one package store while checking and writing.

    Args:
        package (Path): The package store directory; the lock file lives in the temp directory, keyed by path.

    Yields:
        None: While the lock is held.
    """
    path = _lock_path(package)
    with open(path, "a+b") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            unlock = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)  # noqa: E731
        except ImportError:  # pragma: no cover - Windows
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            unlock = lambda: (handle.seek(0), msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1))  # noqa: E731
        try:
            yield
        finally:
            unlock()


# ---------------------------------------------------------------------------------------
# Proposal structures


@dataclass(frozen=True)
class TextChange:
    """A new version of a text row.

    Attributes:
        name (str): Row name (an existing name, or a new part's name).
        text (str): Human-readable text with single-brace slots.
        slots (tuple[str, ...]): Declared slots.
        reason (str): Why this version exists (``manifest.reason``).
        description (str): Row description; the parent row's when empty.
    """

    name: str
    text: str
    slots: tuple[str, ...]
    reason: str
    description: str = ""
    op: str = field(default="text", init=False)


@dataclass(frozen=True)
class SpecChange:
    """A new version of a spec row.

    Attributes:
        name (str): Row name.
        spec (Any): ``rubiksql.spec/1`` JSON.
        parts (Mapping[str, str]): Slot to row name.
        params (Mapping[str, Any]): Parameter types.
        reason (str): Why this version exists.
        description (str): Row description; the parent row's when empty.
    """

    name: str
    spec: Any
    parts: Mapping[str, str]
    params: Mapping[str, Any]
    reason: str
    description: str = ""
    op: str = field(default="spec", init=False)


@dataclass(frozen=True)
class DocChange:
    """A new version of a doc row: one document with its parameter types.

    Attributes:
        name (str): Row name, e.g. ``judge``.
        text (str): The document (``rubiksql.doc/1``).
        params (Mapping[str, Any]): Parameter types.
        reason (str): Why this version exists.
        description (str): Row description; the parent row's when empty.
    """

    name: str
    text: str
    params: Mapping[str, Any]
    reason: str
    description: str = ""
    op: str = field(default="doc", init=False)


@dataclass(frozen=True)
class RemoveChange:
    """An entry the child release no longer carries; parts only it reached leave the closure with it.

    Attributes:
        name (str): Complete document entry name, for example ``judge``.
        reason (str): Why the entry goes (optional; the proposal's reason covers a layout change).
    """

    name: str
    reason: str = ""
    op: str = field(default="remove", init=False)


@dataclass(frozen=True)
class KnobsChange:
    """New explicit knob values of a closure/2 release's parameter set (``op: knobs``).

    The child's parameter set is the parent's explicit values with these set over them; every other explicit
    value stays, and a knob neither lists keeps its code default.

    Attributes:
        values (Mapping[str, Any]): Knob name to its new explicit value (the authoring form's ``set``).
        reason (str): Why the parameter set changes.
    """

    values: Mapping[str, Any]
    reason: str
    name: str = field(default=_TYPED_OPS["knobs"], init=False)
    op: str = field(default="knobs", init=False)


@dataclass(frozen=True)
class ToolsChange:
    """A new typed tool inventory of a closure/2 release (``op: tools``).

    The Tool versions come from the authoring input the authority builds (``--source-commit`` or
    ``--tool-source``) and replace the parent's tool section and loader pins as a whole.

    Attributes:
        reason (str): Why the tools change.
    """

    reason: str
    name: str = field(default=_TYPED_OPS["tools"], init=False)
    op: str = field(default="tools", init=False)


@dataclass(frozen=True)
class Proposal:
    """One proposal as authored: intent plus the changes.

    Attributes:
        parents (tuple[Mapping[str, str], ...]): ``{"release": <baseline|r…|p…>, "role": base|merge}``; exactly one base.
        author (str): Author label, e.g. ``agent:claude/evolver-3``.
        reason (str): One-sentence summary.
        changes (tuple[TextChange | SpecChange | DocChange | RemoveChange | RulesChange | KnobsChange | ToolsChange, ...]):
            The new row versions, removed entries, at most one change to the general rule corpus
            (:class:`rules.RulesChange`) and, on a closure/2 base, at most one parameter-set change
            (:class:`KnobsChange`) and one tool change (:class:`ToolsChange`).
        hypothesis (Mapping[str, Any] | None): claim, mechanism, expected, evidence, risks, falsifier.
        plan (Mapping[str, Any] | None): Analysis plan; merged over :data:`DEFAULT_PLAN`.
        origin (str): ``agent`` or ``human``.
        revert_of (str | None): Proposal this one undoes, for the ``↩`` marker.
        created_at (float | None): Epoch seconds; ``time.time()`` at propose when ``None``.
    """

    parents: tuple[Mapping[str, str], ...]
    author: str
    reason: str
    changes: tuple[TextChange | SpecChange | DocChange | RemoveChange | RulesChange | KnobsChange | ToolsChange, ...]
    hypothesis: Mapping[str, Any] | None = None
    plan: Mapping[str, Any] | None = None
    origin: str = "agent"
    revert_of: str | None = None
    created_at: float | None = None

    @classmethod
    def from_json(cls, body: Mapping[str, Any], *, base_dir: Path | None = None) -> "Proposal":
        """Parse the authoring form (``rubiksql.proposal/1`` with ``changes`` as a list of ops).

        The ops are ``{"op": "text", "name", "text" | "text_file", "slots", "reason"}``,
        ``{"op": "spec", "name", "spec" | "spec_file", "parts", "params", "reason"}``,
        ``{"op": "doc", "name", "text" | "text_file", "params", "reason"}``,
        ``{"op": "remove", "name"}`` and ``{"op": "rules", "reason", "add", "replace", "remove"}`` (the general rule
        corpus, :class:`rules.RulesChange`, whose ``name`` may be omitted); ``description`` is optional on the first
        three. A closure/2 base also takes ``{"op": "knobs", "set": {knob: value, ...}, "reason"}``
        (:class:`KnobsChange`) and ``{"op": "tools", "reason"}`` (:class:`ToolsChange`), whose ``name`` may be omitted
        and is otherwise the typed key they are recorded under; :func:`propose` refuses either on a closure/1 base and
        a text change of a prompt carrier on a closure/2 base.

        Args:
            body (Mapping[str, Any]): The proposal document.
            base_dir (Path | None): Directory ``text_file`` / ``spec_file`` paths resolve against.

        Returns:
            Proposal: The parsed proposal.

        Raises:
            EvolveError: If the schema tag, a change op or a referenced file is wrong.
        """
        if body.get("schema") != SCHEMA_PROPOSAL:
            raise EvolveError(f"proposal: expected schema {SCHEMA_PROPOSAL!r}, got {body.get('schema')!r}")
        base = Path(base_dir) if base_dir is not None else Path.cwd()
        changes: list[TextChange | SpecChange | DocChange | RemoveChange | RulesChange | KnobsChange | ToolsChange] = []
        for index, item in enumerate(body.get("changes") or []):
            if not isinstance(item, Mapping) or ("name" not in item and item.get("op") not in ("rules", *_TYPED_OPS)):
                raise EvolveError(f"proposal.changes[{index}]: each change needs op and name")
            op = item.get("op")
            name = str(item["name"]) if "name" in item else _TYPED_OPS.get(str(op), E.RULES_ENTRY)
            reason = str(item.get("reason") or "")
            description = str(item.get("description") or "")
            if op == "text":
                text = item.get("text")
                if text is None and item.get("text_file"):
                    text = (base / str(item["text_file"])).read_text(encoding="utf-8")
                if not isinstance(text, str):
                    raise EvolveError(f"proposal.changes[{index}] ({name}): a text change needs text or text_file")
                changes.append(TextChange(name, text, tuple(str(slot) for slot in item.get("slots") or []), reason, description))
            elif op == "spec":
                spec = item.get("spec")
                if spec is None and item.get("spec_file"):
                    spec = json.loads((base / str(item["spec_file"])).read_text(encoding="utf-8"))
                if spec is None:
                    raise EvolveError(f"proposal.changes[{index}] ({name}): a spec change needs spec or spec_file")
                changes.append(SpecChange(name, spec, dict(item.get("parts") or {}), dict(item.get("params") or {}), reason, description))
            elif op == "doc":
                text = item.get("text")
                if text is None and item.get("text_file"):
                    text = (base / str(item["text_file"])).read_text(encoding="utf-8")
                if not isinstance(text, str):
                    raise EvolveError(f"proposal.changes[{index}] ({name}): a doc change needs text or text_file")
                if not isinstance(item.get("params"), Mapping):
                    raise EvolveError(f"proposal.changes[{index}] ({name}): a doc change needs params (parameter name to type)")
                changes.append(DocChange(name, text, dict(item["params"]), reason, description))
            elif op == "remove":
                changes.append(RemoveChange(name, reason))
            elif op == "rules":
                try:
                    changes.append(RulesChange.from_json(item, f"proposal.changes[{index}]"))
                except RuleCorpusError as error:
                    raise EvolveError(str(error)) from error
            elif op in _TYPED_OPS:
                known = {"op", "name", "reason", *(("set",) if op == "knobs" else ())}
                where = f"proposal.changes[{index}] (op {op})"
                if sorted(set(item) - known) or name != _TYPED_OPS[op]:
                    raise EvolveError(f"{where}: holds exactly {sorted(known)}, with name omitted or {_TYPED_OPS[op]!r}")
                if not reason.strip():
                    raise EvolveError(f"{where}: a reason is required")
                if op == "knobs":
                    if not isinstance(item.get("set"), Mapping) or not item["set"]:
                        raise EvolveError(f"{where}: set must be a non-empty object of knob name to value")
                    changes.append(KnobsChange(dict(item["set"]), reason))
                else:
                    changes.append(ToolsChange(reason))
            else:
                raise EvolveError(f"proposal.changes[{index}] ({name}): op must be 'text', 'spec', 'doc', 'remove', 'rules', "
                                  f"'knobs' or 'tools', got {op!r}")
        parents = tuple(dict(parent) for parent in body.get("parents") or [{"release": "baseline", "role": "base"}])
        created = body.get("created_at")
        return cls(
            parents,
            str(body.get("author") or ""),
            str(body.get("reason") or ""),
            tuple(changes),
            body.get("hypothesis"),
            body.get("plan"),
            str(body.get("origin") or "agent"),
            body.get("revert_of"),
            float(created) if isinstance(created, (int, float)) else None,
        )

    def frozen_plan(self) -> dict[str, Any]:
        """The plan as frozen into the proposal: the author's fields over :data:`DEFAULT_PLAN`.

        Returns:
            dict[str, Any]: ``metric``, ``exposure`` and ``control``, plus any other field the author wrote, which the
                gate then refuses (``gate.PLAN_FIELDS`` is the whole plan).
        """
        plan: dict[str, Any] = json.loads(json.dumps(DEFAULT_PLAN))
        for key, value in dict(self.plan or {}).items():
            if key == "control" and isinstance(value, Mapping):
                plan["control"].update(value)
            else:
                plan[key] = value
        return plan


# ---------------------------------------------------------------------------------------
# Parents, ancestry, merge


def resolve_parent(ref: str, store: Store) -> str:
    """Turn a parent reference into a release id.

    Args:
        ref (str): ``baseline``, ``r<16hex>`` or ``p<16hex>`` (a pinned proposal, whose release is used).
        store (Store): The store.

    Returns:
        str: The release id.

    Raises:
        EvolveError: If the reference has another shape.
        PromptStoreError: If the proposal or pointer cannot be read.
    """
    if ref == "baseline":
        return store.baseline().release
    if RELEASE_RE.fullmatch(ref):
        return ref
    if PROPOSAL_RE.fullmatch(ref):
        body = store.proposal(ref).data
        return str(body["release"])
    raise EvolveError(f"parent {ref!r} must be 'baseline', a release id (r<16hex>) or a pinned proposal id (p<16hex>)")


def _parent_map(store: Store) -> dict[str, set[str]]:
    """First-publication ancestry; later re-proposals never add release edges."""
    from .store import publication_ancestry

    releases = {path.stem: store.release(path.stem).data for path in (store.package / "releases").glob("r*.json")
                if RELEASE_RE.fullmatch(path.stem)}
    proposals = {path.stem: store.proposal(path.stem).data for path in (store.package / "proposals").glob("p*.json")
                 if PROPOSAL_RE.fullmatch(path.stem)}
    return publication_ancestry(releases, proposals)


def common_ancestor(store: Store, base: str, merge: str) -> str | None:
    """The nearest release both parents descend from, through the store's proposals.

    Args:
        store (Store): The store.
        base (str): Base parent release id.
        merge (str): Merge parent release id.

    Returns:
        str | None: The ancestor release id (a parent itself when one descends from the other), or ``None``.
    """
    parents = _parent_map(store)

    def ancestry(rid: str) -> list[str]:
        order, seen, queue = [], set(), [rid]
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            order.append(current)
            queue.extend(sorted(parents.get(current, ())))
        return order

    merge_side = set(ancestry(merge))
    for candidate in ancestry(base):
        if candidate in merge_side:
            return candidate
    return None


def merge_refs(base: Mapping[str, str], merge: Mapping[str, str], ancestor: Mapping[str, str] | None) -> dict[str, str]:
    """Three-way merge of two row sets against their common ancestor.

    Args:
        base (Mapping[str, str]): Row name to content key of the base parent.
        merge (Mapping[str, str]): Row name to content key of the merge parent.
        ancestor (Mapping[str, str] | None): Row name to content key of the common ancestor; ``None`` when unknown.

    Returns:
        dict[str, str]: The merged row set.

    Raises:
        EvolveError: If both sides changed the same name differently (or differ without an ancestor).
    """
    merged: dict[str, str] = {}
    conflicts: list[str] = []
    for name in sorted(set(base) | set(merge)):
        b, m = base.get(name), merge.get(name)
        a = ancestor.get(name) if ancestor is not None else None
        if b == m:
            if b is not None:
                merged[name] = b
        elif ancestor is None:
            conflicts.append(name)
        elif b == a:
            if m is not None:
                merged[name] = m
        elif m == a:
            if b is not None:
                merged[name] = b
        else:
            conflicts.append(name)
    if conflicts:
        raise EvolveError("merge conflict: both parents changed " + ", ".join(conflicts) + "; resolve by proposing the merged version explicitly in changes")
    return merged


def _typed_sections(release: Any) -> dict[str, Any]:
    """The typed sections of a loaded closure/2 release: tools with their loader pins, parameters and rules."""
    refs = release.content_refs
    return {"tools": {"tools": list(refs["tools"]), "loader": dict(refs["loader"])},
            "parameters": dict(refs["parameters"]), "rules": dict(refs["rules"])}


def merge_sections(base: Mapping[str, Any], merge: Mapping[str, Any], ancestor: Mapping[str, Any] | None) -> dict[str, Any]:
    """Three-way merge of two closure/2 releases' typed sections against their common ancestor's.

    A section is merged as a whole: the tools with their loader pins, the parameter set, and the rules section with
    its ordered items and corpus metadata. A section only one side changed takes that side's value.

    Args:
        base (Mapping[str, Any]): ``{"tools", "parameters", "rules"}`` of the base parent.
        merge (Mapping[str, Any]): The same sections of the merge parent.
        ancestor (Mapping[str, Any] | None): The same sections of the common ancestor; ``None`` when unknown or not a
            closure/2 release.

    Returns:
        dict[str, Any]: The merged sections.

    Raises:
        EvolveError: If both sides changed a section differently relative to the ancestor (or differ without one).
    """
    merged: dict[str, Any] = {}
    conflicts: list[str] = []
    for name in _TYPED_SECTIONS:
        mine, theirs = base[name], merge[name]
        common = ancestor[name] if ancestor is not None else None
        if mine == theirs or (ancestor is not None and theirs == common):
            merged[name] = mine
        elif ancestor is not None and mine == common:
            merged[name] = theirs
        else:
            conflicts.append(name)
    if conflicts:
        raise EvolveError("merge conflict: both parents changed the typed " + ", ".join(conflicts) + " section(s) differently; "
                          "resolve by proposing the merged content explicitly in changes")
    return merged


# ---------------------------------------------------------------------------------------
# propose


@dataclass
class ProposeResult:
    """What :func:`propose` built and checked.

    Attributes:
        id (str): Proposal id (``p`` + 16 hex of the canonical body).
        body (dict[str, Any]): The canonical proposal body, with ``id``.
        release (str): The child release id.
        refs (dict[str, str]): The child's row name to content key.
        entries (list[str]): The child's entries.
        drafts (list[Any]): ``TextDraft`` / ``SpecDraft`` of the changed rows, in publication order.
        report (GateReport): The gate report.
        base (Any): The loaded base parent release.
        child (Any): The loaded child release (from the scratch overlay).
        store (Store): The store the child loads from (package layer plus the scratch overlay).
        scratch_dir (Path): The scratch overlay directory.
        tool_rows (tuple[Mapping[str, Any], ...]): Canonical Tool versions a closure/2 child adds (none it reuses).
        rule_rows (tuple[Mapping[str, Any], ...]): Canonical rule versions a closure/2 child adds.
        parameter_rows (tuple[Mapping[str, Any], ...]): The canonical parameter set a closure/2 child adds, if new.
        capsule_rows (tuple[Mapping[str, Any], ...]): Module and serializer source Capsules a closure/2 child's new
            tools pin that the store does not hold yet.
    """

    id: str
    body: dict[str, Any]
    release: str
    refs: dict[str, str]
    entries: list[str]
    drafts: list[Any]
    report: GateReport
    base: Any
    child: Any
    store: Store
    scratch_dir: Path
    tool_rows: tuple[Mapping[str, Any], ...] = ()
    rule_rows: tuple[Mapping[str, Any], ...] = ()
    parameter_rows: tuple[Mapping[str, Any], ...] = ()
    capsule_rows: tuple[Mapping[str, Any], ...] = ()
    _keep: Any = None


def _role_of(name: str) -> str:
    parts = name.split(".")
    return parts[0] if parts[0] in (*E.ROLE_DOCUMENTS, "shared", "gen", "config", "rules") else "shared"


def _tags(name: str, kind: str, origin: str) -> list[str]:
    tags = ["rubiksql", f"role:{_role_of(name)}", f"kind:{kind}"]
    if name in E.ENTRIES:
        tags.append("entry")
    if name in E.ROLE_DOCUMENTS or (name in E.ENTRIES and E.ENTRIES[name].slot is not None):
        tags.append("body")
    tags.append(f"origin:{origin}")
    return tags


def _draft_for(change: TextChange | SpecChange | DocChange, proposal: Proposal, parent_row: Mapping[str, Any] | None) -> Any:
    from .control import DocDraft, SpecDraft, TextDraft

    description = change.description or (str(parent_row.get("description") or "") if parent_row else "") or change.reason
    parent_key = parent_row["manifest"]["meta"]["content_key"] if parent_row else None
    if isinstance(change, TextChange):
        return TextDraft(change.name, change.text, list(change.slots), description, _tags(change.name, "text", proposal.origin), change.reason, origin=proposal.origin, author=proposal.author, parent=parent_key)
    if isinstance(change, DocChange):
        return DocDraft(change.name, change.text, dict(change.params), description, _tags(change.name, "doc", proposal.origin), change.reason, origin=proposal.origin, author=proposal.author, parent=parent_key)
    return SpecDraft(change.name, change.spec, dict(change.parts), dict(change.params), description, _tags(change.name, "spec", proposal.origin), change.reason, origin=proposal.origin, author=proposal.author, parent=parent_key)


def _order_drafts(drafts: Sequence[Any]) -> list[Any]:
    """Text and doc drafts first (they reference no rows), then spec drafts after every draft part they reference."""
    by_name = {draft.name: draft for draft in drafts}
    ordered: list[Any] = [draft for draft in drafts if draft.kind in ("text", "doc")]
    done = {draft.name for draft in ordered}
    pending = [draft for draft in drafts if draft.kind == "spec"]
    while pending:
        progress = False
        for draft in list(pending):
            if all(target not in by_name or target in done for target in draft.parts.values()):
                ordered.append(draft)
                done.add(draft.name)
                pending.remove(draft)
                progress = True
        if not progress:
            raise EvolveError("spec changes reference each other in a cycle: " + ", ".join(draft.name for draft in pending))
    return ordered


def _closure(refs: Mapping[str, str], rows: Mapping[str, Mapping[str, Any]], entries: Iterable[str]) -> dict[str, str]:
    reachable: dict[str, str] = {}
    stack = list(entries)
    while stack:
        name = stack.pop()
        if name in reachable:
            continue
        if name not in refs:
            raise EvolveError(f"row {name!r} is referenced but has no version in the child release")
        reachable[name] = refs[name]
        meta = rows[name]["manifest"]["meta"]
        if meta["kind"] == "spec":
            stack.extend(meta["parts"].values())
    return reachable


def _object_doc(store: Store, name: str, key: str) -> Mapping[str, Any]:
    return store.object(name, key[:16]).data


def _hb_adapter(renderer: Any):
    def hb_render(row: Mapping[str, Any], part_rows: Mapping[str, Mapping[str, Any]], params: Mapping[str, Any]) -> str:
        return renderer.render(row["name"], **params)

    return hb_render


def propose(
    proposal: Proposal,
    store: Store,
    *,
    scratch_control: Any | None = None,
    hb: bool = True,
    corpus: Any | None = None,
    root_run: Any | None = None,
    now: float | None = None,
    tools: Any | None = None,
) -> ProposeResult:
    """Derive the child release from a proposal, publish its new rows into a scratch workspace and gate it.

    Nothing is written to the real control workspace or the store. The targets of the proposal
    are the rows whose content differs from the base parent (derived by diff, never declared);
    a merge parent is three-way merged against the common ancestor and a same-name edit on both
    sides is a conflict.

    A closure/2 base (``rubiksql.closure/2``) produces a closure/2 child: prompt changes work as above; an
    ``op: rules`` change rebuilds the corpus the base's rule versions reconstruct and partitions the child corpus into
    rule versions, so an unchanged rule keeps its version; an ``op: knobs`` change sets knob values over the parent's
    explicit ones in a new parameter set; an ``op: tools`` change takes the tool section ``tools`` carries. Merges
    combine the typed sections as a whole (:func:`merge_sections`), and the proposal's ``changes`` are
    :func:`entity_model.closure_changes`. The new typed rows join the scratch store's ``native/<entity>.json`` so the
    child loads, and the result names them for the publication batch.

    Args:
        proposal (Proposal): The proposal.
        store (Store): The store the parents load from.
        scratch_control (Control | None): A control workspace to publish the drafts into for HeavenBase
            reference execution; a temporary one is opened when ``None`` and ``hb`` is true.
        hb (bool): Run the HeavenBase half of the matrix dual run; skipped (and reported so) when false.
        corpus (Corpus | None): Real-input corpus for the ``k of N`` check.
        root_run (RootRun | None): Root run for the cost estimate.
        now (float | None): ``created_at`` when the proposal carries none.
        tools (TypedTools | None): Canonical Tool versions an ``op: tools`` change of a closure/2 base publishes
            (:func:`hb_tool_entities.build_tool_rows`), built by the authority from its authoring input.

    Returns:
        ProposeResult: The gated child; ``report.passed`` tells whether publication may follow.

    Raises:
        EvolveError: On a malformed proposal, an unresolvable parent, a merge conflict, a row the change cannot
            build on, a typed op on a closure/1 base, a prompt-carrier change on a closure/2 base, tool input
            without its ``op: tools`` change (or the reverse), or a closure/1 base in a store that holds an
            entity-model decision (a retained historical release no new content builds on).
        PromptStoreError: If BASELINE names a v0 root whose V0.json marker is absent.
    """
    store.require_seeded_v0()
    from .control import Control, PromptControlError, export_objects
    from .runtime import load_release

    bases = [parent for parent in proposal.parents if parent.get("role", "base") == "base"]
    merges = [parent for parent in proposal.parents if parent.get("role") == "merge"]
    if len(bases) != 1:
        raise EvolveError(f"a proposal needs exactly one base parent, got {len(bases)}")
    base_id = resolve_parent(str(bases[0]["release"]), store)
    base = load_release(base_id, stores=store, allow_retired=True)
    typed = base.rules == EM.CLOSURE_V2
    migrated = [str(decision.get("id")) for decision in store.decisions() if decision.get("kind") == EM.MIGRATION_KIND] if not typed else []
    if migrated:
        raise EvolveError(f"{base_id} is a retained historical {E.CLOSURE_RULES} release: the store migrated to typed tool, rule and "
                          f"parameter-set versions (entity-model decision {migrated[0]}), so new content builds on a {EM.CLOSURE_V2} "
                          "release; return to one with authority rollback <typed release>")
    refs: dict[str, str] = dict(base.refs)
    sections = _typed_sections(base) if typed else None
    resolved_parents = [{"release": base_id, "role": "base"}]
    for parent in merges:
        merge_id = resolve_parent(str(parent["release"]), store)
        merge_release = load_release(merge_id, stores=store, allow_retired=True)
        if (merge_release.rules == EM.CLOSURE_V2) != typed:
            raise EvolveError(f"merge parent {merge_id} is a {merge_release.rules} release and the base {base_id} a {base.rules} "
                              "release; merge parents of one release format")
        ancestor_id = common_ancestor(store, base_id, merge_id)
        ancestor = load_release(ancestor_id, stores=store, allow_retired=True) if ancestor_id else None
        refs = merge_refs(refs, merge_release.refs, ancestor.refs if ancestor is not None else None)
        if sections is not None:
            common = _typed_sections(ancestor) if ancestor is not None and ancestor.rules == EM.CLOSURE_V2 else None
            sections = merge_sections(sections, _typed_sections(merge_release), common)
        resolved_parents.append({"release": merge_id, "role": "merge"})

    rows: dict[str, Mapping[str, Any]] = {name: _object_doc(store, name, key)["row"] for name, key in refs.items()}
    drafts = []
    removed: set[str] = set()
    typed_changes: dict[str, RulesChange | KnobsChange | ToolsChange] = {}
    if sum(change.name == E.RULES_ENTRY for change in proposal.changes) > 1:
        raise EvolveError(f"{E.RULES_ENTRY}: a proposal changes the rule corpus once (one op rules, or one text change)")
    for change in proposal.changes:
        problem = _format_problem(change.op, change.name, typed, base_id)
        if problem:
            raise EvolveError(problem)
        if isinstance(change, (KnobsChange, ToolsChange)) or (typed and isinstance(change, RulesChange)):
            if change.op in typed_changes:
                raise EvolveError(f"op {change.op}: a proposal changes that typed section once")
            typed_changes[change.op] = change
            continue
        if isinstance(change, RulesChange):
            change = _rules_text_change(change, rows.get(E.RULES_ENTRY), base_id)
        try:
            check_name(change.name)
        except PolicyError as error:
            raise EvolveError(str(error)) from error
        if isinstance(change, RemoveChange):
            if change.name not in refs or change.name not in base.entries:
                raise EvolveError(f"{change.name}: only an entry of the base release can be removed")
            removed.add(change.name)
            continue
        # the kind may change with the text: a row whose text gains messages or blocks becomes a doc row,
        # a composition may become one document; the child's load checks the entry's contract either way
        drafts.append(_draft_for(change, proposal, rows.get(change.name)))
    clash = sorted(removed & {draft.name for draft in drafts})
    if clash:
        raise EvolveError(f"{clash}: a proposal either removes an entry or gives it a new version, not both")
    drafts = _order_drafts(drafts)
    draft_versions = {draft.name: draft.version for draft in drafts}
    for draft in drafts:
        if draft.kind == "spec":
            versions = {}
            for slot, target in draft.parts.items():
                if target in draft_versions:
                    versions[slot] = draft_versions[target]
                elif target in refs:
                    versions[slot] = refs[target][:16]
                else:
                    raise EvolveError(f"{draft.name}: part {slot!r} names {target!r}, which is neither a row of the parent release nor a change of this proposal")
            object.__setattr__(draft, "part_versions", versions)
    known = store.typed_index() if typed else {}
    if typed:
        content_refs, added, capsules = _typed_child(base_id, sections, typed_changes, tools, known, store)
    elif tools is not None:
        raise EvolveError(f"tool authoring input builds typed Tool versions, and {base_id} is a {base.rules} release; "
                          "change its tools.answering row with op text")
    else:
        content_refs, added, capsules = None, {}, []

    keep = None
    control = scratch_control
    if control is None and hb:
        keep = tempfile.TemporaryDirectory(prefix="rubiksql-prompts-scratch-")
        control = Control.open(Path(keep.name) / "control")
    scratch = tempfile.TemporaryDirectory(prefix="rubiksql-prompts-propose-")
    scratch_dir = Path(scratch.name) / "overlay"
    scratch_dir.mkdir()
    for entity, new in (*added.items(), ("sys-capsule", capsules)):
        if new:  # the base store's rows plus the new ones, so the child's typed closure resolves from the overlay
            held = {str(row["object_id"]): row for row in store.native(entity)}
            held.update({str(row["object_id"]): row for row in new})
            path = scratch_dir / "native" / f"{entity}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_bytes([held[key] for key in sorted(held)]))
    new_rows: dict[str, Mapping[str, Any]] = {}
    if control is not None:
        try:
            control.import_objects([_object_doc(store, name, key) for name, key in sorted(refs.items())], actor=proposal.author or "propose")
            for draft in drafts:
                outcome_ = control.publish(draft, actor=proposal.author or "propose")
                new_rows[draft.name] = outcome_.row
        except PromptControlError as error:
            raise EvolveError(f"propose: {error}") from error
        export_objects(scratch_dir, [new_rows[draft.name] for draft in drafts if not _in_store(store, draft.name, draft.version)])
    else:
        for draft in drafts:
            row = _synthetic_row(draft)
            new_rows[draft.name] = row
            if not _in_store(store, draft.name, draft.version):
                write_if_absent(scratch_dir / "objects" / draft.name / f"{draft.version}.json", {"schema": "rubiksql.object/1", "row": row, "export": {"content_key": draft.content_key, "name": draft.name, "version": draft.version, "control_object_id": ""}})
    for draft in drafts:
        refs[draft.name] = draft.content_key
        rows[draft.name] = new_rows[draft.name]
    entries = sorted(name for name in refs if name in E.ENTRIES and name not in removed)
    closure = _closure(refs, rows, entries)
    root = store.v0_root()
    epoch = store.release(root).data["epoch"] if root is not None else None
    if typed:
        release_doc = {"schema": EM.SCHEMA_RELEASE_V2, "refs": closure, "entries": entries, "rules": EM.CLOSURE_V2,
                       "content_refs": content_refs, "origin": "proposal", **({"epoch": epoch} if epoch is not None else {})}
        try:
            EM.check_release_document(release_doc)
        except EM.EntityModelError as error:
            raise EvolveError(f"propose: {error}") from error
        rid = release_document_id(release_doc)
        release_doc = {"schema": EM.SCHEMA_RELEASE_V2, "id": rid,
                       **{key: value for key, value in release_doc.items() if key != "schema"}}
    else:
        rid = release_id(closure, entries, E.CLOSURE_RULES, epoch=epoch)
        release_doc = {"schema": SCHEMA_RELEASE, "id": rid, "refs": closure, "entries": entries,
                       "rules": E.CLOSURE_RULES, "origin": "proposal"}
        if epoch is not None:
            release_doc["epoch"] = epoch
    if not _release_in_store(store, rid):
        write_if_absent(scratch_dir / "releases" / f"{rid}.json", release_doc)
    for path in store.package.rglob("*"):
        if path.is_file():
            destination = scratch_dir / path.relative_to(store.package)
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(path.read_bytes())
    child_store = Store(package=scratch_dir)
    child = load_release(rid, stores=child_store, allow_retired=True)

    if typed:
        try:
            changes = EM.closure_changes(store.release(base_id).data, child_store.release(rid).data, child_store.typed_index())
        except EM.EntityModelError as error:
            raise EvolveError(f"propose: {error}") from error
    else:
        changes = {name: {"from": (base.refs.get(name) or None), "to": closure.get(name)} for name in sorted(set(base.refs) | set(closure)) if base.refs.get(name) != closure.get(name)}
    plan = proposal.frozen_plan()
    hypothesis = json.loads(json.dumps(proposal.hypothesis)) if proposal.hypothesis is not None else None
    body: dict[str, Any] = {
        "schema": SCHEMA_PROPOSAL,
        "parents": resolved_parents,
        "release": rid,
        "changes": changes,
        "reason": proposal.reason,
        "hypothesis": hypothesis,
        "hypothesis_digest": sha256_hex(canon(hypothesis)) if hypothesis is not None else "",
        "plan": plan,
        "plan_digest": sha256_hex(canon(plan)),
        "author": proposal.author,
        "origin": proposal.origin,
        "created_at": float(proposal.created_at if proposal.created_at is not None else (now if now is not None else time.time())),
    }
    if proposal.revert_of:
        body["revert_of"] = proposal.revert_of
    pid = proposal_id(body)
    body["id"] = pid
    for draft in drafts:
        object.__setattr__(draft, "proposal", pid)

    hb_render = _hb_adapter(_renderer_for(control, child)) if control is not None else None
    report = run_gate(child, parent=base, store=store, proposal=body, hb_render=hb_render, corpus=corpus, root_run=root_run)
    return ProposeResult(pid, body, rid, dict(closure), entries, drafts, report, base, child, child_store, scratch_dir,
                         tool_rows=tuple(added.get(EM.TOOL_ENTITY, ())), rule_rows=tuple(added.get(EM.RULE_ENTITY, ())),
                         parameter_rows=tuple(added.get(EM.PARAMETER_ENTITY, ())), capsule_rows=tuple(capsules),
                         _keep=(scratch, keep, control))


def _format_problem(op: str, name: str, typed: bool, base_id: str) -> str:
    """Why a change does not fit its base release's format, or ``""``.

    A closure/1 base carries its knobs, rule corpus and tools in prompt carriers, which only text (and ``op: rules``)
    changes edit; a closure/2 base carries typed versions, which only the typed ops change.
    """
    if op in _TYPED_OPS and not typed:
        carrier = E.KNOBS_ENTRY if op == "knobs" else "tools.answering"
        return (f"op {op}: {base_id} is a {E.CLOSURE_RULES} release, which carries its {op} in the {carrier} text row; "
                f"change that row with op text, or propose on a {EM.CLOSURE_V2} release")
    if typed and name in EM.CARRIERS and op != "rules":
        return _CARRIER_OPS[name]
    return ""


def _typed_child(base_id: str, sections: Mapping[str, Any], changes: Mapping[str, Any], tools: Any,
                 known: Mapping[str, Mapping[str, Mapping[str, Any]]], store: Store
                 ) -> tuple[dict[str, Any], dict[str, list[Mapping[str, Any]]], list[Mapping[str, Any]]]:
    """Assemble a closure/2 child's typed block from the (merged) parent sections and the typed ops.

    Args:
        base_id (str): The base parent release, recorded as a new rule's ``parent_release``.
        sections (Mapping[str, Any]): The parent's (merged) typed sections (:func:`merge_sections`).
        changes (Mapping[str, Any]): ``rules``, ``knobs`` and ``tools`` ops of the proposal, by op.
        tools (TypedTools | None): The built Tool versions; required exactly when ``tools`` is changed.
        known (Mapping[str, Mapping[str, Mapping[str, Any]]]): The store's canonical rows by entity and id.
        store (Store): The base store, whose native Capsules decide which source Capsules are new.

    Returns:
        ``(content_refs, entity id -> rows the store does not hold, new source Capsules)``.

    Raises:
        EvolveError: Tool input and the ``op: tools`` change disagree, or a knob, corpus or block is invalid.
    """
    if ("tools" in changes) != (tools is not None):
        raise EvolveError("op tools: a tool change needs exactly one op tools change with its authoring input "
                          "(--source-commit or --tool-source), and authoring input needs the op tools change")
    added: dict[str, list[Mapping[str, Any]]] = {entity: [] for entity in EM.TYPED_ENTITY_IDS}
    capsules: list[Mapping[str, Any]] = []
    tool_section, parameters, rules_section = sections["tools"], sections["parameters"], sections["rules"]
    if tools is not None:
        tool_section = {"tools": list(tools.content["tools"]), "loader": dict(tools.content["loader"])}
        added[EM.TOOL_ENTITY] = [row for row in tools.tool_rows if row["object_id"] not in known.get(EM.TOOL_ENTITY, {})]
        held = {str(row["object_id"]) for row in store.native("sys-capsule")}
        capsules = [row for row in tools.capsule_rows if row["object_id"] not in held]
    if "knobs" in changes:
        current = known.get(EM.PARAMETER_ENTITY, {}).get(parameters["object_id"])
        if current is None:
            raise EvolveError(f"op knobs: the parent parameter set {parameters['object_id']} is not in the store")
        try:
            row = EM.parameter_row({**EM.verify_parameter_row(current), **changes["knobs"].values})
        except EM.EntityModelError as error:
            raise EvolveError(f"op knobs: {error}") from error
        parameters = EM.typed_ref(EM.PARAMETER_ENTITY, row)
        if row["object_id"] not in known.get(EM.PARAMETER_ENTITY, {}):
            added[EM.PARAMETER_ENTITY].append(row)
    if "rules" in changes:
        try:
            corpus = changes["rules"].corpus_after(EM.reconstruct_corpus(rules_section, known.get(EM.RULE_ENTITY, {})), base_id)
            rule_rows, rules_section = EM.partition_corpus(corpus)
        except (RuleCorpusError, EM.EntityModelError) as error:
            raise EvolveError(f"{E.RULES_ENTRY}: {error}") from error
        added[EM.RULE_ENTITY] = [row for row in rule_rows if row["object_id"] not in known.get(EM.RULE_ENTITY, {})]
    try:
        content_refs = EM.check_release_refs({"schema": EM.SCHEMA_RELEASE_REFS, **tool_section, "parameters": parameters,
                                              "rules": rules_section})
    except EM.EntityModelError as error:
        raise EvolveError(f"propose: {error}") from error
    return content_refs, added, capsules


def _rules_text_change(change: RulesChange, row: Mapping[str, Any] | None, parent_release: str) -> TextChange:
    """The ``rules.general`` text version an ``op: rules`` change makes of the (merged) parent's corpus."""
    if row is None:
        raise EvolveError(f"op rules: {parent_release} carries no {E.RULES_ENTRY}; propose on a release that carries the corpus")
    try:
        corpus = change.corpus_after(json.loads(row["manifest"]["meta"]["text"]), parent_release)
    except RuleCorpusError as error:
        raise EvolveError(f"{E.RULES_ENTRY}: {error}") from error
    return TextChange(E.RULES_ENTRY, corpus_text(corpus), (), change.reason)


def _renderer_for(control: Any, release: Any) -> Any:
    from .control import HbRenderer

    return HbRenderer.for_release(control.ws, release)


def _in_store(store: Store, name: str, version: str) -> bool:
    try:
        store.object(name, version)
    except PromptStoreError:
        return False
    return True


def _release_in_store(store: Store, rid: str) -> bool:
    try:
        store.release(rid)
    except PromptStoreError:
        return False
    return True


def _synthetic_row(draft: Any) -> dict[str, Any]:
    """An HB-shaped row for a draft without HeavenBase (``hb=False``): the fields the loader verifies."""
    from .canon import escape, template_code
    from .doc import DocError, compile_document
    from .spec import compile_spec

    meta = draft.meta
    meta["content_key"] = draft.content_key
    if draft.kind == "text":
        source, qualname = template_code(draft.name, escape(draft.text, list(draft.slots)), []), "render"
    elif draft.kind == "doc":
        try:
            source, qualname = compile_document(draft.name, draft.text, dict(draft.params)), "compose"
        except DocError as error:
            raise EvolveError(f"{draft.name}: {error}") from error
    else:
        source, qualname = compile_spec(draft.name, draft.spec, dict(draft.parts), dict(draft.params)), "compose"
    digest = sha256_hex(source)
    object_id = f"scratch:{draft.name}:{draft.version}"
    manifest = {"object_id": object_id, "name": draft.name, "version": draft.version, "description": draft.description, "tags": list(draft.tags), "dicts": [draft.name], "main_lang": "en", "elicit": "none", "meta": meta, "reason": draft.reason}
    capsule = {
        "name": draft.name,
        "namespace": "prompt",
        "version": draft.version,
        "restore_layers": [{"type": "source", "priority": 10, "source": source, "source_hash": digest, "qualname": qualname, "filename": "<capsule-code>", "global_refs": {}}],
        "source": {"ok": True, "source": source, "source_hash": digest, "filename": "<capsule-code>", "qualname": qualname, "doc": {"summary": "", "params": {}, "returns": "", "raw": ""}},
        "trust": {"local_only": True, "allow_binary_payload": False, "verify_source_hash": True},
        "fingerprints": {"source_hash": digest, "signature_hash": "scratch", "restore_hash": "scratch"},
        "status": "active",
    }
    return {"object_id": object_id, "name": draft.name, "version": draft.version, "status": "active", "description": draft.description, "tags": list(draft.tags), "manifest": manifest, "capsule": capsule, "dicts": [draft.name], "main_lang": "en", "fingerprints": {"manifest": "scratch", "capsule": "scratch"}, "created_at": 0.0, "updated_at": 0.0}


TARGETS = ("prompt", "rules", "knobs", "generator", "tool")
TARGET_KINDS = {kind: frozenset({kind}) for kind in TARGETS}
TARGET_KINDS.update(knobs=frozenset({"knobs", "prompt"}), any=frozenset(TARGETS))
_DOCUMENT_SUFFIXES = (".md", ".txt", ".json")


class ProposalError(ValueError):
    """A local scientific proposal draft violates the authoritative authoring contract."""


def _row_kind(name: str) -> str:
    """The kind of row or typed key a name is (one of :data:`TARGETS`, :func:`entity_model.change_kind`)."""
    return EM.change_kind(name)


def changed_kinds(changes: Iterable[Any]) -> list[str]:
    """The kinds of row changes or row names touch, sorted: one proposal may change several kinds together.

    Args:
        changes (Iterable[Any]): A proposal's ``changes`` (an ``op: rules`` change is the rule corpus, ``op: knobs``
            the knobs and ``op: tools`` the tools) or the prompt names and typed keys a child release changes
            (:func:`entity_model.closure_changes`).

    Returns:
        list[str]: The kinds, a subset of :data:`TARGETS`, in its order.
    """
    by_op = {"rules": "rules", "knobs": "knobs", "tools": "tool"}
    kinds = {by_op[change["op"]] if isinstance(change, Mapping) and change.get("op") in by_op
             else _row_kind(str(change.get("name") or "") if isinstance(change, Mapping) else str(change)) for change in changes}
    return [kind for kind in TARGETS if kind in kinds]


def _kinds_problem(kinds: Sequence[str], target: str) -> str:
    """Why rows of ``kinds`` break the loop's ``target``, or ``""``."""
    if target not in TARGET_KINDS:
        return f"unknown evolution target {target!r}"
    allowed = TARGET_KINDS[target]
    outside = [kind for kind in kinds if kind not in allowed]
    if outside:
        return f"the loop's target is {target}, which changes {sorted(allowed)} rows; this proposal also changes {outside} rows"
    if target != "any" and target not in kinds:
        return f"the loop's target is {target}, this proposal changes no {target} row (only {list(kinds)})"
    return ""


def _prompt_names(release: Any) -> list[str]:
    """Prompt rows a proposal may change: the release's role documents and event-time texts, and the evaluated slots."""
    names = [name for name in release.entries if name in E.ENTRIES and E.ENTRIES[name].surfaces and not E.ENTRIES[name].optional]
    return names + [E.slot_entry(slot) for slot in range(1, 3 + 1)]


def _generator_names(release: Any) -> list[str]:
    """Generator documents a proposal may change: the ones a knowledge rebuild sends (:data:`tuple(name for name in E.ENTRIES if name.startswith('gen.'))`)."""
    return [name for name in tuple(name for name in E.ENTRIES if name.startswith('gen.')) if name in release.entries and name in E.ENTRIES]


def _check_change(change: Mapping[str, Any], release: Any) -> None:
    """Refuse a change of a row the loop does not open to proposals, or with an op that row does not take.

    The release format decides how knobs, rules and tools change: a closure/1 release through its prompt carriers
    (``config.knobs`` and ``tools.answering`` with op text, ``rules.general`` with op rules), a closure/2 release
    through the typed ops (``knobs``, ``rules``, ``tools``). As :meth:`Proposal.from_json` requires, op rules names
    ``rules.general`` and op knobs/tools name ``parameters:knobs``/``tools:loader``, or omit the name.
    """
    name, op = str(change["name"]), str(change["op"])
    typed = getattr(release, "rules", None) == EM.CLOSURE_V2
    problem = _format_problem(op, name, typed, str(getattr(release, "id", "the base release")))
    if problem:
        raise ProposalError(problem)
    kind = "rules" if op == "rules" else _row_kind(name)
    if op in _TYPED_OPS:
        if name != _TYPED_OPS[op]:
            raise ProposalError(f"{name}: op {op} names {_TYPED_OPS[op]!r} or omits name")
        return
    slot = E.ENTRIES[name].slot if name in E.ENTRIES else None
    if name == "tools.answering":
        if op != "text" or list(change.get("slots") or []):
            raise ProposalError("tools.answering changes with op text and no slots")
        return
    if kind == "rules":
        if op != "rules" or name != E.RULES_ENTRY:
            raise ProposalError(f"{name}: the rule corpus changes with one change {{\"op\": \"rules\", \"name\": \"{E.RULES_ENTRY}\", \"add\": [...]}}")
        return
    if name == E.KNOBS_ENTRY:
        if op != "text" or list(change.get("slots") or []):
            raise ProposalError(f"{name}: the knob row changes with op text and no slots")
        return
    allowed = _generator_names(release) if kind == "generator" else _prompt_names(release)
    if name not in allowed:
        raise ProposalError(f"{name}: not a row a proposal may change; the {kind} rows are {allowed}")
    if op == "remove":
        if slot is None or name not in release.entries:
            raise ProposalError(f"{name}: only a candidate slot document the release carries may be removed")
    elif op not in ("doc", "text"):
        raise ProposalError(f"{name}: op {op!r} is not open to the loop; use doc or text (remove only for a slot document)")


def _dossier_file(dossier: Path, value: Any, where: str) -> None:
    """Refuse a ``text_file`` outside the dossier, missing, or not a document."""
    if not isinstance(value, str) or not value.strip():
        raise ProposalError(f"{where}: text_file must be a path relative to the dossier")
    relative = Path(value)
    if relative.is_absolute():
        raise ProposalError(f"{where}: text_file {value!r} is absolute; write paths relative to the dossier")
    path = (dossier / relative).resolve()
    if not path.is_relative_to(dossier.resolve()):
        raise ProposalError(f"{where}: text_file {value!r} leaves the dossier")
    if path.suffix.lower() not in _DOCUMENT_SUFFIXES:
        raise ProposalError(f"{where}: text_file {value!r} must be a document ({', '.join(_DOCUMENT_SUFFIXES)})")
    if not path.is_file():
        raise ProposalError(f"{where}: text_file {value!r} does not exist")


def _knob_spelling(change: Mapping[str, Any], dossier: Path) -> dict[str, Any]:
    """A knob change with its values in the one spelling the loader accepts (:func:`entries.knobs_text`), as text.

    The agent writes the values and the loop the spelling, so a trailing newline, another indent or another key order
    is no refusal; the values themselves are the gate's to check.
    """
    text = change.get("text")
    if text is None and change.get("text_file"):
        text = (dossier / str(change["text_file"])).read_text(encoding="utf-8")
    try:
        values = json.loads(text) if isinstance(text, str) else None
    except ValueError as error:
        raise ProposalError(f"{E.KNOBS_ENTRY}: the knob row is not JSON: {error}") from error
    if not isinstance(values, dict):
        raise ProposalError(f"{E.KNOBS_ENTRY}: the knob row is one JSON object of knob values")
    return {**{key: value for key, value in change.items() if key != "text_file"}, "text": E.knobs_text(values)}


def _check_rule_sources(changes: Sequence[Mapping[str, Any]], failures: Iterable[str], database_titles: Mapping[str, str]) -> None:
    """Refuse a rule an ``op: rules`` change brings that cites a question outside the parent run's failures, or takes the
    title of a database's own rule (the knowledge build would rename that rule ``<title>_<db_id>``); :class:`rules.RulesChange`
    refuses a malformed entry itself."""
    wrong = set(failures)
    for index, change in enumerate(changes):
        if change.get("op") != "rules":
            continue
        for key in ("add", "replace"):
            entries = change.get(key)
            for position, entry in enumerate(entries if isinstance(entries, list) else []):
                if not isinstance(entry, Mapping):
                    continue
                where = f"changes[{index}].{key}[{position}]"
                cited = entry.get("dossier_qids")
                try:
                    qids = {rules_module.qid(value) for value in cited} if isinstance(cited, list) else set()
                except rules_module.RuleCorpusError as error:
                    raise ProposalError(f"{where}.dossier_qids: {error}") from error
                outside = sorted(qids - wrong)
                if outside:
                    raise ProposalError(f"{where}.dossier_qids: {outside} are not failures of the parent experiment (its traces with EX 0); name the failures the rule came from")
                record = entry.get("rule") if key == "replace" else entry
                title = record.get("title") if isinstance(record, Mapping) else None
                if isinstance(title, str) and title in database_titles:
                    raise ProposalError(f"{where}: {title!r} is the title of a rule {database_titles[title]} keeps for itself; the knowledge build would rename that rule "
                                        "<title>_<db_id> to make room for yours, a change of the arm's knowledge beyond this proposal, so give the rule another title")


def validate_proposal(dossier: Path, *, target: str, release: Any, pool: Iterable[str], author: str, namespace: str = "", now: float | None = None, database_titles: Mapping[str, str] | None = None,
                      failures: Iterable[str] = ()) -> tuple[dict[str, Any], list[str]]:
    """Read the agent's ``proposal.json`` and apply the loop's rules to it; the gate checks everything else.

    The loop's rules: the schema tag; a non-empty ``changes`` list whose rows are rows a proposal may change, of kinds the
    loop's target allows (:data:`TARGET_KINDS`: under ``any`` one proposal may change prompt, rules, knobs and generator
    rows together; under a named target at least one row is of that kind; an ``op: rules`` change without a name changes
    ``rules.general``, as :class:`rules.RulesChange` reads it), in the form the release's format takes (a closure/2
    BASELINE changes knobs with ``op: knobs``, whose ``set`` names declared knobs with valid values, and tools with
    ``op: tools``, never through a prompt carrier; a closure/1 BASELINE takes neither typed op); every ``text_file`` a
    document inside the dossier; for an
    ``op: rules`` change, ``dossier_qids`` that are failures of the parent experiment and no rule title one of the
    evaluation databases' own rules has (the knowledge build would rename that rule ``<title>_<db_id>``, a change of the
    arm's knowledge beyond the proposal); the base parent BASELINE and any merge parent a proposal of the pool. The loop
    then sets what the agent does not choose: ``author``, ``origin`` (``agent``), the frozen default ``plan`` (its
    namespace the run universe), ``created_at`` and the spelling of a knob row (:func:`entries.knobs_text`). Reasons, the
    hypothesis — its ``prior`` included, which the gate requires of an author ending in ``gate.LOOP_AUTHOR_SUFFIX`` —,
    the documents' grammar and parameters, and the knob values are the gate's to check, all in one report.

    Args:
        dossier (Path): The dossier holding ``proposal.json``.
        target (str): The loop's target, one of :data:`TARGET_CHOICES`.
        release (Release): BASELINE.
        pool (Iterable[str]): Proposal ids of this generation a proposal may merge.
        author (str): The author label the loop records.
        namespace (str): The cache namespace the evaluations run in (the plan's ``control.namespace``).
        now (float | None): ``created_at``; ``time.time()`` by default.
        database_titles (Mapping[str, str] | None): Titles of the evaluation databases' own rules, each to a database
            that has it; none checked when ``None``.
        failures (Iterable[str]): The question ids the parent run got wrong: the only ones a new rule may cite.

    Returns:
        tuple[dict[str, Any], list[str]]: The proposal body to hand to ``Proposal.from_json`` and the kinds of row it
            changes (:func:`changed_kinds`).

    Raises:
        ProposalError: When ``proposal.json`` is missing or malformed or breaks a rule of the loop.
    """
    path = Path(dossier)
    if path.is_file():
        dossier = path.parent
    else:
        path = dossier / "proposal.json"
    if not path.is_file():
        raise ProposalError("no proposal.json in the dossier")
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ProposalError(f"proposal.json is not JSON: {error}") from error
    if not isinstance(body, dict):
        raise ProposalError("proposal.json must hold one JSON object")
    if body.get("schema") != SCHEMA_PROPOSAL:
        raise ProposalError(f"proposal.json: schema must be {SCHEMA_PROPOSAL!r}")
    changes = body.get("changes")
    if not isinstance(changes, list) or not changes:
        raise ProposalError("proposal.json: changes must be a non-empty list")
    # an op rules change names rules.general or nothing (rules.RulesChange), an op knobs or tools change its typed key or
    # nothing; the loop's checks below read the name
    defaults = {"rules": E.RULES_ENTRY, **_TYPED_OPS}
    changes = [{"name": defaults[change["op"]], **change}
               if isinstance(change, Mapping) and change.get("op") in tuple(defaults) and "name" not in change else change
               for change in changes]
    for index, change in enumerate(changes):
        if not isinstance(change, Mapping) or not isinstance(change.get("name"), str) or not isinstance(change.get("op"), str):
            raise ProposalError(f"changes[{index}]: every change is an object with op and name")
        if "spec_file" in change or change.get("op") == "spec":
            raise ProposalError(f"changes[{index}] ({change['name']}): spec changes are not open to the loop; publish documents")
        if "text_file" in change:
            _dossier_file(dossier, change["text_file"], f"changes[{index}] ({change['name']})")
    kinds = changed_kinds(changes)
    problem = _kinds_problem(kinds, target)
    if problem:
        raise ProposalError(problem)
    for index, change in enumerate(changes):
        _check_change(change, release)
        if change["op"] == "knobs":
            values = change.get("set")
            if not isinstance(values, Mapping) or not values:
                raise ProposalError(f"changes[{index}] (op knobs): set must be a non-empty object of knob name to value")
            try:
                EM.parameter_values(values)
            except EM.EntityModelError as error:
                raise ProposalError(f"changes[{index}] (op knobs): {error}") from error
    if "rules" in kinds:
        _check_rule_sources(changes, failures, database_titles or {})
    parents = body.get("parents") or [{"release": "baseline", "role": "base"}]
    if not isinstance(parents, list) or not all(isinstance(parent, Mapping) for parent in parents):
        raise ProposalError("parents must be a list of {release, role} objects")
    bases = [parent for parent in parents if parent.get("role", "base") == "base"]
    if len(bases) != 1 or bases[0].get("release") not in ("baseline", release.id):
        raise ProposalError(f"the base parent is BASELINE: {{\"release\": \"baseline\", \"role\": \"base\"}}; got {parents}")
    merges = [str(parent.get("release")) for parent in parents if parent.get("role") == "merge"]
    known = set(pool)
    outside = [pid for pid in merges if pid not in known]
    if outside:
        raise ProposalError(f"merge parents must be proposals of this generation's pool (authority history pool); {outside} are not")
    plan = json.loads(json.dumps(DEFAULT_PLAN))
    plan["control"]["namespace"] = namespace
    created_at = now if now is not None else body.get("created_at", time.time())
    if isinstance(created_at, bool) or not isinstance(created_at, (int, float)) or not math.isfinite(created_at):
        raise ProposalError("proposal.json: created_at must be a finite numeric timestamp")
    normalized = {
        **body,
        "changes": [_knob_spelling(change, dossier) if change["name"] == E.KNOBS_ENTRY else change for change in changes],
        "parents": [{"release": "baseline", "role": "base"}, *({"release": pid, "role": "merge"} for pid in merges)],
        "author": author,
        "origin": "agent",
        "plan": plan,
        "created_at": float(created_at),
    }
    return normalized, kinds


# ---------------------------------------------------------------------------------------
# Authority publication adapters


def _gate_summary(report: GateReport) -> dict[str, Any]:
    """The gate record stored with a proposal: pass/fail per check and whether HeavenBase actually ran."""
    matrix = next((check for check in report.checks if check.name == "matrix_dual_run"), None)
    hb = str(matrix.data.get("hb")) if matrix is not None and "hb" in matrix.data else "unknown"
    return {"passed": report.passed, "checks": {check.name: check.status for check in report.checks}, "hb": hb}


@dataclass
class RunView:
    """One evaluated run directory as :func:`outcome` reads it.

    Attributes:
        path (Path): Run directory.
        name (str): Directory name.
        manifest (dict[str, Any]): The ``.prompt_manifest.json`` document (``{}`` when absent).
        manifest_sha (str): sha256 of the manifest bytes (``""`` when absent).
        rows (dict[str, Mapping[str, Any]]): Eval rows by question id.
        result_files (dict[str, str]): Result file (relative) to sha256.
        created_at (float): Manifest ``created_at`` or the manifest/directory mtime.
    """

    path: Path
    name: str
    manifest: dict[str, Any]
    manifest_sha: str
    rows: dict[str, Mapping[str, Any]]
    result_files: dict[str, str]
    created_at: float

    @property
    def release(self) -> str:
        """The release the run declares.

        Returns:
            str: ``manifest.release`` or the rows' ``prompt_release`` when the manifest is absent.
        """
        declared = self.manifest.get("release")
        if isinstance(declared, str):
            return declared
        seen = {str(row.get("prompt_release")) for row in self.rows.values() if row.get("prompt_release")}
        return seen.pop() if len(seen) == 1 else ("mixed" if seen else "")

    @property
    def diagnostic(self) -> bool:
        """Whether any retained run provenance marks a descriptive measurement."""
        launch = self.manifest.get("paid_launch")
        return bool(self.manifest.get("diagnostic") or self.manifest.get("promotion_eligible") is False
                    or self.identity.get("diagnostic_experiment")
                    or (isinstance(launch, Mapping) and launch.get("diagnostic")))

    def release_problem(self, *, store: Store | None = None) -> str:
        """Why the current rows cannot represent the run's declared release, or ``""``.

        Every current result row must record its exact release. Refactor-only releases
        with the same prompts and knowledge rows qualify; historical ``result_attempts`` never decide admission.

        Args:
            store (Store | None): Store layers used to compare releases; the runtime's default when absent.

        Returns:
            str: The first inconsistent row, a mixed-manifest reason, or ``""`` when the run is consistent.
        """
        from .runtime import load_release

        if self.diagnostic:
            return "diagnostic measurements are descriptive only and cannot take an outcome or serve as a promotion parent"
        if self.manifest.get("mixed"):
            releases = self.manifest.get("releases") or {}
            return f"its manifest records rows of several prompt releases ({', '.join(f'{key}: {value}' for key, value in sorted(releases.items())) or 'mixed'}); a mixed directory takes no outcome, re-run it under one release"
        declared = self.release
        equivalent: dict[str | None, bool] = {declared: True}
        release = None
        for qid, row in self.rows.items():
            recorded = row.get("prompt_release")
            if not recorded:
                return f"its current row {qid} has no recorded prompt release"
            row_release = str(recorded) if recorded is not None else None
            if row_release not in equivalent:
                if release is None:
                    try:
                        release = load_release(declared, stores=store, allow_retired=True)
                    except PromptStoreError:
                        return f"its declared release {declared!r} cannot be loaded to verify current row {qid}'s prompt release {row_release!r}"
                equivalent[row_release] = same_run_release(release, row_release, store=store)
            if not equivalent[row_release]:
                return f"its current row {qid} has prompt release {row_release!r}, which differs from the run's declared release {declared!r}"
        return ""

    @property
    def ns(self) -> str:
        """The cache namespace (sampling universe) of the run.

        Returns:
            str: ``manifest.ns`` or ``""``.
        """
        return str(self.manifest.get("ns") or "")

    @property
    def identity(self) -> Mapping[str, Any]:
        """Every non-prompt identity field of the run.

        Returns:
            Mapping[str, Any]: ``manifest.identity`` or ``{}``.
        """
        value = self.manifest.get("identity")
        return value if isinstance(value, Mapping) else {}

    @property
    def questions(self) -> dict[str, Any] | None:
        """The questions of the whole queries file the run was launched with, as the manifest records them.

        Returns:
            dict[str, Any] | None: ``{"count": int, "ids_sha256": str}`` (:func:`question_set_record`), or ``None``
                when the manifest records none (a run made before ``eval_bird`` recorded it) or a malformed one.
        """
        value = self.manifest.get("questions")
        if not isinstance(value, Mapping) or set(value) != {"count", "ids_sha256"}:
            return None
        count, digest = value["count"], value["ids_sha256"]
        if not isinstance(count, int) or isinstance(count, bool) or not isinstance(digest, str):
            return None
        return {"count": count, "ids_sha256": digest}


def read_run(run_dir: Path) -> RunView:
    """Read a run directory: manifest, rows and result-file digests.

    Args:
        run_dir (Path): Directory holding ``.prompt_manifest.json`` and result files (``<db>.json`` or ``pass1/``).

    Returns:
        RunView: The run.

    Raises:
        PromptStoreError: If the directory holds no result files.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / RUN_MANIFEST_NAME
    manifest: dict[str, Any] = {}
    manifest_sha = ""
    if manifest_path.is_file():
        raw = manifest_path.read_bytes()
        manifest_sha = sha256_hex(raw)
        loaded = json.loads(raw.decode("utf-8"))
        manifest = dict(loaded) if isinstance(loaded, Mapping) else {}
    rows = {str(row.get("id")): row for row in read_rows(run_dir)}
    directory = run_dir / "pass1" if (run_dir / "pass1").is_dir() else run_dir
    digests = {}
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if isinstance(data, Mapping) and isinstance(data.get("results"), list):
            digests[path.relative_to(run_dir).as_posix()] = sha256_hex(path.read_bytes())
    created = manifest.get("created_at")
    if not isinstance(created, (int, float)):
        created = manifest_path.stat().st_mtime if manifest_path.is_file() else run_dir.stat().st_mtime
    return RunView(run_dir, run_dir.name, manifest, manifest_sha, rows, digests, float(created))


def require_evolution_run(run: RunView) -> None:
    """Require a complete, scored, nondiagnostic control of the current content-pinned set.

    Args:
        run: Immutable evaluation evidence to use for new evolution work.

    Raises:
        EvolveError: The input bytes, question roster, or scores differ from Mini-Dev 500.
    """
    path = Path(__file__).resolve().parents[3] / EVOLUTION_SET.path
    raw = path.read_bytes()
    ids = {f"Q{int(row.get('question_id', row.get('id'))):05d}" for row in json.loads(raw)}
    if (run.diagnostic or "sha256:" + sha256_hex(raw) != EVOLUTION_QUESTION_SET
            or run.identity.get("question_set") != EVOLUTION_QUESTION_SET
            or len(ids) != EVOLUTION_SET.count or run.questions != question_set_record(ids)
            or set(run.rows) != ids or any(M.EvalRow(row).ex is None for row in run.rows.values())):
        raise EvolveError("new evolution work requires a complete scored Mini-Dev 500 control")


def find_parent_run(child: RunView, parent_release: str, *, namespace: str, max_age_days: float, search: Iterable[Path] = (), same: Callable[[str], bool] | None = None, store: Store | None = None) -> RunView | None:
    """Find the most recent run of the parent release in the child's universe that scored every question the child did.

    A candidate must share the child's namespace and identity, be no older than the drift window and no newer
    than a day (the window when that is shorter; ``--parent-run`` accepts the window either way), and cover the
    child's scored questions: a smoke run of the parent (``--limit``, ``--ids``, ``--db``) is never chosen as the
    control of a run over the full question set, however recent it is. Its current row releases must also agree
    with its declared release (:meth:`RunView.release_problem`).

    Args:
        child (RunView): The evaluated child run.
        parent_release (str): The base parent release id.
        namespace (str): The plan's ``control.namespace``.
        max_age_days (float): The drift window.
        search (Iterable[Path]): Directories whose subdirectories are candidate runs; the child's parent
            directory by default.
        same (Callable[[str], bool] | None): Whether a run's release sends the same prompts as the base parent
            (``runtime.same_prompts``), so a run under a refactor-only sibling of the parent — a sibling with the same prompts, knowledge and tools — qualifies as its control; id equality when ``None``.
        store (Store | None): Store layers used to check the candidate runs' current row releases.

    Returns:
        RunView | None: The paired parent run, or ``None`` when nothing qualifies.
    """
    if child.release_problem(store=store):
        return None
    roots = list(search) or [child.path.parent]
    is_same = same or (lambda release: release == parent_release)
    child_scored = _scored(child)
    best: RunView | None = None
    for root in roots:
        for path in sorted(Path(root).iterdir()) if Path(root).is_dir() else []:
            if not path.is_dir() or path.resolve() == child.path.resolve() or not (path / RUN_MANIFEST_NAME).is_file():
                continue
            try:
                candidate = read_run(path)
            except PromptStoreError:
                continue
            if not is_same(candidate.release) or candidate.ns != namespace or candidate.identity != child.identity:
                continue
            if candidate.release_problem(store=store):
                continue
            if child.created_at - candidate.created_at > max_age_days * _DAY or candidate.created_at > child.created_at + min(_DAY, max_age_days * _DAY):
                continue
            if not child_scored <= _scored(candidate):
                continue
            if best is None or candidate.created_at > best.created_at:
                best = candidate
    return best


def _sql_digest(sql: Any) -> str:
    text = re.sub(r"\s+", " ", str(sql or "")).strip().rstrip(";").strip()
    return sha256_hex(text)[:16] if text else ""


#: Divergence buckets ``integrity_detail.divergences`` counts child units by: ``prompt`` aggregates every
#: ``prompt:<entry>`` cause, ``none`` is a unit that replayed to the end, ``other`` an unrecognised cause string.
CAUSE_BUCKETS = ("prompt", "initial", CAUSE_KNOWLEDGE, CAUSE_KNOB, "tool", "clock", "unknown", CAUSE_NO_SOURCE, "none", "other")
_NOISE_BUCKETS = ("tool", "clock", "unknown", "other")


def _cause_bucket(cause: Any, *, implementation_attribution: str | None = None) -> str:
    if cause is None:
        return "none"
    text = str(cause)
    if text.startswith("prompt"):
        return "prompt"
    if text == CAUSE_IMPLEMENTATION and implementation_attribution == _IMPLEMENTATION_ATTRIBUTION:
        return text
    if text in ("initial", CAUSE_KNOWLEDGE, CAUSE_KNOB, "tool", "clock", "unknown", CAUSE_NO_SOURCE):
        return text
    return "other"


def _source_attributed(origin: Mapping[str, Any], row: Mapping[str, Any], role: str, index: int, release: Any) -> bool:
    """Whether a retained producer proof names exactly this row's database, question, role seat and render release.

    Attribution only: ``source_compatible`` additionally requires a transcript that was not interrupted,
    the only kind a child may claim to have replayed (:func:`_verified_parent_source`).
    """
    return (origin.get("db") == row.get("db_id")
            and origin.get("qid") == str(row.get("qid", row.get("id", "")))
            and origin.get("role") == role
            and origin.get("slot") == (index + 1 if role == "candidate" else None)
            and origin.get("release") == release)


def _units(row: Mapping[str, Any], *, implementation_attribution: str | None = None) -> dict[str, dict[str, Any]]:
    if implementation_attribution == _IMPLEMENTATION_ATTRIBUTION:
        return _role_units(row)
    units: dict[str, dict[str, Any]] = {}
    candidates = row.get("candidates") if isinstance(row.get("candidates"), list) else []
    for index, debug in enumerate(row.get("candidates_debug") or []):
        if not isinstance(debug, Mapping) or debug.get("transcript_status") == "not_run":
            continue
        identity = debug.get("prompt_identity") if isinstance(debug.get("prompt_identity"), Mapping) else {}
        replay = identity.get("replay") if isinstance(identity.get("replay"), Mapping) else {}
        summary = candidates[index] if index < len(candidates) and isinstance(candidates[index], Mapping) else {}
        sql = debug.get("sql") if debug.get("sql") is not None else summary.get("sql")
        units[f"c{index}"] = {"role": "candidate", "index": index, "sql": _sql_digest(sql), "diverged_at": replay.get("diverged_at"), "cause": replay.get("cause"), "source": replay.get("source"), "turns": replay.get("turns"), "replayed": replay.get("replayed"), "fired": sorted(identity.get("fired") or {}) if isinstance(identity.get("fired"), Mapping) else [], "msg": dict(identity.get("msg") or {}) if isinstance(identity.get("msg"), Mapping) else {}}
    judge = row.get("judge_debug") if isinstance(row.get("judge_debug"), Mapping) else {}
    if judge and judge.get("transcript_status") != "not_run":
        identity = judge.get("prompt_identity") if isinstance(judge.get("prompt_identity"), Mapping) else {}
        replay = identity.get("replay") if isinstance(identity.get("replay"), Mapping) else {}
        units["j"] = {"role": "judge", "index": -1, "sql": _sql_digest(row.get("pred_sql")), "diverged_at": replay.get("diverged_at"), "cause": replay.get("cause"), "source": replay.get("source"), "turns": replay.get("turns"), "replayed": replay.get("replayed"), "fired": sorted(identity.get("fired") or {}) if isinstance(identity.get("fired"), Mapping) else [], "msg": dict(identity.get("msg") or {}) if isinstance(identity.get("msg"), Mapping) else {}}
    reviewer = row.get("reviewer_debug") if isinstance(row.get("reviewer_debug"), Mapping) else {}
    review = reviewer.get("route_debug") if isinstance(reviewer.get("route_debug"), Mapping) else {}
    reviewed_degraded = (reviewer.get("active") is not False and review.get("transcript_status") != "not_run"
                         and degraded_record(review))
    for label, unit in units.items():
        debug = judge if label == "j" else row["candidates_debug"][unit["index"]]
        identity = debug.get("prompt_identity") if isinstance(debug.get("prompt_identity"), Mapping) else {}
        replay = identity.get("replay") if isinstance(identity.get("replay"), Mapping) else {}
        proof = debug.get("cache_source")
        origin = proof.get("identity", {}) if isinstance(proof, Mapping) else {}
        if not isinstance(origin, Mapping):
            origin = {}
        attributed = _source_attributed(origin, row, unit["role"], unit["index"], identity.get("release"))
        # The original policy's judge unit carries the post-review SQL. Its
        # retrieval health must include that reviewer; roles/1 splits them below.
        unit.update(degraded_retrieval=degraded_record(debug) or (label == "j" and reviewed_degraded),
                    source_proof=proof, sources=replay.get("sources"),
                    cache_key=debug.get("cache_key"), memo_base_key=debug.get("memo_base_key"),
                    content_sha=cache_lineage.content_sha(debug),
                    source_indices=[i for i, message in enumerate(debug.get("messages") or [])
                                    if isinstance(message, Mapping) and message.get("role") == "assistant"],
                    source_attributed=attributed,
                    source_compatible=attributed and debug.get("transcript_status") != "interrupted")
    return units


def _role_unit(debug: Mapping[str, Any], row: Mapping[str, Any], role: str, index: int, sql: Any) -> dict[str, Any]:
    """Project a retained role transcript without changing its immutable producer record.

    ``source_attributed`` means the proof's producer identity names this unit's database, question, role seat and
    release (descriptor validity and the content hash are separate checks); ``source_compatible`` additionally
    requires a transcript that was not interrupted, the only kind a child may claim to replay.
    """
    identity = debug.get("prompt_identity") if isinstance(debug.get("prompt_identity"), Mapping) else {}
    replay = identity.get("replay") if isinstance(identity.get("replay"), Mapping) else {}
    proof = debug.get("cache_source")
    origin = proof.get("identity") if isinstance(proof, Mapping) else {}
    origin = origin if isinstance(origin, Mapping) else {}
    attributed = _source_attributed(origin, row, role, index, identity.get("release"))
    return {
        "role": role, "index": index, "sql": _sql_digest(sql),
        "degraded_retrieval": degraded_record(debug),
        "cache_key": debug.get("cache_key"), "memo_base_key": debug.get("memo_base_key"),
        "db": row.get("db_id"), "qid": str(row.get("qid", row.get("id", ""))),
        "transcript_status": debug.get("transcript_status"), "whole_replay": bool(debug.get("from_cache")),
        "diverged_at": replay.get("diverged_at"), "cause": replay.get("cause"),
        "source": replay.get("source"), "turns": replay.get("turns"), "replayed": replay.get("replayed"),
        "fired": sorted(identity.get("fired") or {}) if isinstance(identity.get("fired"), Mapping) else [],
        "msg": dict(identity.get("msg") or {}) if isinstance(identity.get("msg"), Mapping) else {},
        "source_proof": proof, "sources": replay.get("sources"), "content_sha": cache_lineage.content_sha(debug),
        "source_indices": [i for i, message in enumerate(debug.get("messages") or [])
                           if isinstance(message, Mapping) and message.get("role") == "assistant"],
        "source_attributed": attributed,
        "source_compatible": attributed and debug.get("transcript_status") != "interrupted",
        "implementation_context": debug.get("consumer_implementation_context", debug.get("implementation_context")),
        "route": identity.get("route") or "default",
        "route_known": "route" in identity,
        "capabilities": debug.get("effective_capabilities"),
    }


def _role_units(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Read candidate, pre-review judge and separate reviewer units under ``roles/1``."""
    units = _units(row)
    for label, unit in tuple(units.items()):
        debug = row["judge_debug"] if label == "j" else row["candidates_debug"][unit["index"]]
        units[label] = _role_unit(debug, row, unit["role"], unit["index"], debug.get("sql"))
        if label != "j":
            units[label]["sql"] = unit["sql"]
    reviewer = row.get("reviewer_debug") if isinstance(row.get("reviewer_debug"), Mapping) else {}
    review_debug = reviewer.get("route_debug") if isinstance(reviewer.get("route_debug"), Mapping) else {}
    judge_debug = row.get("judge_debug") if isinstance(row.get("judge_debug"), Mapping) else {}
    answer_hooks = {}
    for name, knob in E.KNOBS.items():
        if knob.post_review_answer_hook is not None:
            record_key, producer_sql_key = knob.post_review_answer_hook
            record = judge_debug.get(record_key)
            if isinstance(record, Mapping):
                answer_hooks[name] = _sql_digest(record.get(producer_sql_key))
    has_reviewer = bool(review_debug and reviewer.get("active") is not False
                        and review_debug.get("transcript_status") != "not_run")
    if "j" in units:
        judge = row["judge_debug"]
        sql = judge.get("sql")
        if sql is None:
            sql = reviewer.get("original_sql")
        if sql is None and (not reviewer.get("active") or reviewer.get("error") or reviewer.get("skipped")):
            sql = row.get("pred_sql")
        if answer_hooks and not has_reviewer:
            # A skipped reviewer leaves the judge as the final-answer unit.
            # A declared answer hook can change its answer without changing its turns.
            sql = row.get("pred_sql")
            units["j"]["post_review_pre_sql"] = answer_hooks
        units["j"]["sql"] = _sql_digest(sql)
        units["j"]["answer_known"] = sql is not None
    if has_reviewer:
        # The post-review hook controls this role's final answer, while the
        # immutable reviewer transcript retains its pre-hook submission.
        reviewer_sql = row.get("pred_sql") if answer_hooks else review_debug.get("sql", row.get("pred_sql"))
        units["r"] = _role_unit(review_debug, row, "reviewer", -1,
                                reviewer_sql)
    return units


def _implementation_records(value: Any) -> list[dict[str, Any]]:
    """Read the declared role context/digest records from a manifest."""
    if not isinstance(value, Mapping):
        return []
    candidates = value.get("candidate")
    values = [*(candidates if isinstance(candidates, list) else []), value.get("judge"), value.get("reviewer")]
    records = []
    for record in values:
        if not isinstance(record, Mapping) or set(record) != {"context", "digest"}:
            continue
        context, digest = record["context"], record["digest"]
        if (not isinstance(context, Mapping) or context.get("role") not in ("candidate", "judge", "reviewer")
                or not isinstance(context.get("tools"), list) or not all(isinstance(name, str) for name in context["tools"])
                or len(set(context["tools"])) != len(context["tools"])
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            continue
        expected = {"role", "route", "tools"} if context["role"] == "candidate" else {"role", "tools"}
        if set(context) != expected or (context["role"] == "candidate" and not isinstance(context.get("route"), str)):
            continue
        records.append({"context": dict(context), "digest": digest})
    return records


def _implementation_contexts(run: RunView | None) -> dict[str, Any] | None:
    """Bind declared contexts to this immutable manifest, never to the current checkout."""
    if run is None:
        return None
    block = run.manifest.get("implementation")
    if isinstance(block, Mapping) and "role_digests" in block:
        return {"manifest_sha256": run.manifest_sha, "source": "manifest", "role_digests": block["role_digests"]}
    return {"manifest_sha256": run.manifest_sha, "source": "unknown", "role_digests": None}


def _unit_implementation(unit: Mapping[str, Any], record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Resolve only a context observed by the unit and declared by its run."""
    if not record:
        return None
    candidates = [item for item in _implementation_records(record.get("role_digests"))
                  if item["context"]["role"] == unit.get("role")]
    capabilities = unit.get("capabilities")
    if isinstance(capabilities, list):
        candidates = [item for item in candidates if set(item["context"]["tools"]) == set(capabilities)]
    observed = unit.get("implementation_context")
    if observed is not None:
        if (unit.get("role") == "candidate" and unit.get("route_known") and isinstance(observed, Mapping)
                and isinstance(observed.get("context"), Mapping) and observed["context"].get("route") != unit.get("route")):
            return None
        matched = list({canon(item): item for item in candidates
                        if isinstance(observed, Mapping) and item["context"] == observed.get("context")}.values())
        return matched[0] if len(matched) == 1 and matched[0] == observed else None
    return None


def _implementation_change(parent_unit: Mapping[str, Any], child_unit: Mapping[str, Any], contexts: Mapping[str, Any]) -> dict[str, Any] | None:
    parent = _unit_implementation(parent_unit, contexts.get("parent"))
    child = _unit_implementation(child_unit, contexts.get("child"))
    if (parent is None or child is None or parent["digest"] == child["digest"]
            or parent["context"]["role"] != child["context"]["role"]
            or parent["context"].get("route") != child["context"].get("route")):
        return None
    return {"parent": parent, "child": child,
            "parent_manifest_sha256": contexts["parent"]["manifest_sha256"],
            "child_manifest_sha256": contexts["child"]["manifest_sha256"]}


def _implementation_changes(parent: RunView | None, child: RunView, contexts: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Record the distinct proven differences of counterpart units' execution contexts."""
    changes = {}
    if parent is not None:
        for qid in sorted(set(parent.rows) & set(child.rows)):
            punits, cunits = _role_units(parent.rows[qid]), _role_units(child.rows[qid])
            for label in sorted(set(punits) & set(cunits)):
                change = _implementation_change(punits[label], cunits[label], contexts)
                if change is not None:
                    changes[canon(change)] = change
    return [changes[key] for key in sorted(changes)]


def _verified_parent_source(parent_unit: Mapping[str, Any], child_unit: Mapping[str, Any]) -> bool:
    """Prove every replayed turn refers to the immutable content actually retained by the parent."""
    proof, sources = parent_unit.get("source_proof"), child_unit.get("sources")
    if not cache_lineage.valid_descriptor(proof, attributed=True) or not parent_unit.get("source_compatible"):
        return False
    if proof["content_sha256"] != parent_unit.get("content_sha") or not isinstance(sources, list):
        return False
    if len(sources) != child_unit.get("replayed") or not sources:
        return False
    indices = []
    for item in sources:
        if not isinstance(item, Mapping) or not cache_lineage.valid_descriptor(item.get("source"), attributed=True):
            return False
        if item["source"] != proof or type(item.get("idx")) is not int:
            return False
        indices.append(item["idx"])
    return indices == parent_unit.get("source_indices")


def replays_parent_unit(parent_unit: Mapping[str, Any], record: Mapping[str, Any], *, slot: int) -> bool:
    """Whether a child replaying a cached candidate record whole would pass the strict parent-source check.

    The pre-launch pool check of a paired child asks this of the ``llm_cache`` row stored under
    each parent candidate's key, before any child runs; ``scripts/repair_parent_candidate_cache.py``
    uses the same immutable-source verification after restoring those rows.

    Args:
        parent_unit (Mapping[str, Any]): The parent's candidate unit, as `_units` reads it from the parent row.
        record (Mapping[str, Any]): The ``llm_cache`` record stored under that candidate's cache key.
        slot (int): The candidate's position in the pool, counted from 1.

    Returns:
        bool: True when a whole-cache replay of ``record`` names the parent's own immutable source at
            every assistant turn (`_verified_parent_source`).
    """
    replay = cache_lineage.replay(record, role="candidate", slot=slot)["prompt_identity"].get("replay") or {}
    return _verified_parent_source(parent_unit, {"sources": replay.get("sources"), "replayed": replay.get("replayed")})


def _implementation_replay_sources(child_unit: Mapping[str, Any], parent_run: str | None) -> list[Mapping[str, Any]] | None:
    """Validate complete replay descriptors from this changed unit's non-parent producers."""
    source, turns, replayed = child_unit.get("source"), child_unit.get("turns"), child_unit.get("replayed")
    sources = child_unit.get("sources")
    if (parent_run is None or not isinstance(source, str) or not source.strip()
            or type(turns) is not int or turns < 1 or type(replayed) is not int or replayed != turns
            or child_unit.get("diverged_at") is not None or child_unit.get("transcript_status") == "interrupted"
            or not isinstance(sources, list) or len(sources) != replayed):
        return None
    descriptors, indices = [], []
    slot = child_unit.get("index", -1) + 1 if child_unit.get("role") == "candidate" else None
    for item in sources:
        if not isinstance(item, Mapping) or not cache_lineage.valid_descriptor(item.get("source"), attributed=True):
            return None
        proof, origin = item["source"], item["source"]["identity"]
        if (origin["run"] == parent_run or origin["db"] != child_unit.get("db")
                or origin["qid"] != child_unit.get("qid") or origin["role"] != child_unit.get("role")
                or origin.get("slot") != slot or type(item.get("idx")) is not int):
            return None
        if child_unit.get("whole_replay") and (proof["content_sha256"] != child_unit.get("content_sha")
                                               or source != cache_lineage.REF + proof["id"]):
            return None
        descriptors.append(proof)
        indices.append(item["idx"])
    return descriptors if indices == child_unit.get("source_indices") else None


def _sibling_replay_problems(parent: RunView, child: RunView, table: Iterable[Mapping[str, Any]]) -> list[str]:
    """Require a clean sibling producer's manifest to declare the replayed child's role digest."""
    problems, manifests = [], {}
    for entry in table:
        change = entry.get("implementation_change")
        if entry.get("cause") != CAUSE_IMPLEMENTATION or not isinstance(change, Mapping):
            continue
        unit = _role_units(child.rows[entry["qid"]])[entry["unit"]]
        sources = _implementation_replay_sources(unit, parent.name)
        if sources is None:
            continue  # Fresh sampling has no replay producer to validate.
        for producer in sorted({proof["identity"]["run"] for proof in sources} - {child.name}):
            if producer not in manifests:
                path = child.path.parent / producer / RUN_MANIFEST_NAME
                if Path(producer).name != producer:
                    manifests[producer] = ([], False, "unknown", "is not a sibling run name")
                elif not path.is_file():
                    manifests[producer] = ([], False, "unknown", f"manifest {path} {_NOT_ON_THIS_MACHINE}")
                else:
                    try:
                        manifest = json.loads(path.read_text(encoding="utf-8"))
                        block = manifest.get("implementation") if isinstance(manifest, Mapping) else None
                        records = _implementation_records(block.get("role_digests") if isinstance(block, Mapping) else None)
                        clean = isinstance(block, Mapping) and block.get("clean") is True
                        commit = block.get("commit", "unknown") if isinstance(block, Mapping) else "unknown"
                        manifests[producer] = (records, clean, commit, "does not declare the child's role digest")
                    except (OSError, ValueError) as error:
                        manifests[producer] = ([], False, "unknown", f"manifest cannot be read: {error}")
            records, clean, commit, reason = manifests[producer]
            if change.get("child") in records:
                if clean:
                    continue
                reason = "implementation.clean is not true"
            problems.append(f"sibling replay producer {producer!r} for {entry['qid']}/{entry['unit']} (commit {commit!r}) {reason}")
    return problems


def _settled_cause(parent_unit: Mapping[str, Any], child_unit: Mapping[str, Any], knowledge: bool = False, knob_turns: bool = False, knob_answer: bool = False, *, knob_model: bool = False, strict_sources: bool = False, implementation: bool = False, implementation_attribution: str | None = None, parent_run: str | None = None) -> Any:
    """Verify a missing replay cause, then settle it by the first-request digests and changed surfaces.

    A missing cause requires a nonempty source, positive integer turns, the same integer replayed count,
    and no divergence turn. New outcomes also require ``strict_sources``: each turn must identify the
    same immutable producer and transcript retained by the actual parent unit. Legacy verification keeps
    the earlier counter-only algorithm so recorded outcomes do not change. Otherwise it starts as ``no-source``: live calls with
    lookup disabled also leave the cause empty. Explicit recorded causes remain authoritative.
    The turn memo writes ``no-source`` when no source is available or an interrupted source's exact
    prefix is exhausted. When the parent counterpart exists and either first-request digest
    (``msg.system``/``msg.user``) differs, the first request itself changed, which is the design's
    ``initial`` divergence; only an identical first
    request leaves the unit genuinely unverified. Missing digests on either side settle nothing. When the
    proposal changed the knowledge (``knowledge``), no unit could replay — the base key hashes the knowledge
    fingerprint, and a proposal that moves no keyspace looks nothing up, which leaves no cause at all — so an
    unsettled ``no-source`` and a missing cause are the proposal's :data:`CAUSE_KNOWLEDGE` divergence. When it
    changed a knob that selects the unit's model (``knob_model``), the base key hashes that model, so an unsettled
    ``no-source`` is likewise the knob's (:data:`CAUSE_KNOB`). When it
    changed a knob that acts on the unit's turns (``knob_turns``), a ``tool``, ``clock``, ``unknown`` or unrecognised
    divergence is the knob's (:data:`CAUSE_KNOB`), and when it changed a knob that can change the unit's answer
    (``knob_answer``: the turns, or the reviewer after the judge) so is a unit that diverged nowhere yet ended on
    another final SQL.

    Under ``roles/1``, a proven counterpart implementation difference settles fresh ``no-source`` and
    non-prompt divergences, or an otherwise undiverged changed answer, after prompt/knowledge/knob causes.
    A complete replay proven to be the parent's retained transcript keeps its existing settlement.
    A complete replay from the child's own run or a sibling settles as implementation when every attributed
    source names this unit and a non-parent producer; a whole replay must also match the retained content hash.
    Missing or mismatched replay sources stay unverified. Deep verification additionally requires each sibling
    producer's manifest to declare the child's role digest and ``implementation.clean`` to be exactly true.
    Equal or unknown digests use the earlier order.

    Args:
        parent_unit (Mapping[str, Any]): The parent's unit record from :func:`_units` (empty when absent).
        child_unit (Mapping[str, Any]): The child's unit record.
        knowledge (bool): The pairing's changed surfaces include a :data:`KNOWLEDGE_SURFACES` one.
        knob_turns (bool): A changed knob acts on this unit's turns (:func:`_knob_roles`).
        knob_answer (bool): A changed knob can change this unit's final SQL (:func:`_knob_roles`).
        knob_model (bool): A changed knob selects this unit's model (:func:`_knob_model_roles`).
        strict_sources (bool): Require immutable parent source proof; false only for historical verification.
        implementation (bool): The counterpart contexts have a proven differing role digest.
        implementation_attribution (str | None): ``roles/1`` selects separate role attribution.
        parent_run (str | None): Paired parent run name, required to validate non-parent implementation replays.

    Returns:
        Any: ``"initial"`` for a settled first-request change, ``"knowledge"`` for a unit a knowledge change kept
            from replaying, ``"knob"`` for a unit a knob change moved, ``"no-source"`` for an unverified
            missing cause, ``"implementation"`` for a proven implementation effect, else the recorded cause.
    """
    cause = child_unit.get("cause")
    if cause is None:
        source, turns, replayed = child_unit.get("source"), child_unit.get("turns"), child_unit.get("replayed")
        completed = (isinstance(source, str) and bool(source.strip())
                     and type(turns) is int and turns > 0 and type(replayed) is int and replayed == turns
                     and child_unit.get("diverged_at") is None)
        if strict_sources:
            completed = completed and _verified_parent_source(parent_unit, child_unit)
        if not completed:
            cause = CAUSE_NO_SOURCE
    if not parent_unit:
        return cause
    bucket = _cause_bucket(cause, implementation_attribution=implementation_attribution)
    if bucket == CAUSE_NO_SOURCE:
        parent_msg, child_msg = parent_unit.get("msg") or {}, child_unit.get("msg") or {}
        if parent_msg and child_msg and parent_msg != child_msg:
            return "initial"
    if knowledge and bucket in ("none", CAUSE_NO_SOURCE):
        return CAUSE_KNOWLEDGE
    if knob_model and bucket == CAUSE_NO_SOURCE:
        return CAUSE_KNOB
    if knob_turns and bucket in _NOISE_BUCKETS:
        return CAUSE_KNOB
    if knob_answer and bucket == "none" and child_unit.get("sql") != parent_unit.get("sql"):
        return CAUSE_KNOB
    if implementation and implementation_attribution == _IMPLEMENTATION_ATTRIBUTION:
        if ((child_unit.get("answer_known") is False or parent_unit.get("answer_known") is False)
                and bucket not in ("prompt", "initial", CAUSE_KNOWLEDGE, CAUSE_KNOB)):
            return CAUSE_NO_SOURCE
        turns, replayed = child_unit.get("turns"), child_unit.get("replayed")
        claimed_replay = (type(turns) is int and turns > 0 and type(replayed) is int and replayed == turns
                          and child_unit.get("diverged_at") is None)
        if claimed_replay:
            if bucket in ("prompt", "initial", CAUSE_KNOWLEDGE, CAUSE_KNOB):
                return cause
            # A replay proven to be the parent's own retained transcript keeps the
            # existing settlement; only self or sibling replays can be implementation.
            if _verified_parent_source(parent_unit, child_unit):
                return cause
            if _implementation_replay_sources(child_unit, parent_run) is not None:
                return CAUSE_IMPLEMENTATION
            return CAUSE_NO_SOURCE
        if bucket in (CAUSE_NO_SOURCE, *_NOISE_BUCKETS):
            return CAUSE_IMPLEMENTATION
        if bucket == "none" and child_unit.get("sql") != parent_unit.get("sql"):
            return CAUSE_IMPLEMENTATION
    return cause


def _knob_roles(knobs: Iterable[str], *, implementation_attribution: str | None = None) -> tuple[frozenset[str], frozenset[str]]:
    """Map changed knobs to the unit roles whose turns and answers they can change.

    Under ``roles/1``, candidate, judge and reviewer knobs reach their own unit's turns and answer.
    A post-review answer hook (visible to the reviewer) therefore keeps reviewer turn and answer credit,
    including reviewer noise; when the reviewer is skipped, the hook has a separate, per-unit producer
    answer check.
    With no policy marker, the reviewer shares the judge's final answer, so reviewer knobs reach
    the judge's answer without changing the judge's turns.

    Args:
        knobs (Iterable[str]): Changed knob names (:data:`entries.KNOBS`); unknown ones reach nothing.
        implementation_attribution (str | None): ``roles/1`` maps the reviewer to its own unit;
            ``None`` keeps the original candidate/judge unit mapping.

    Returns:
        tuple[frozenset[str], frozenset[str]]: ``(turns, answers)`` sets of affected unit role names.
    """
    roles = {role for name in knobs if name in E.KNOBS for role, _route in E.KNOBS[name].visible}
    if implementation_attribution == _IMPLEMENTATION_ATTRIBUTION:
        return frozenset(roles), frozenset(roles)
    turns = frozenset(roles - {"reviewer"})
    return turns, turns | (frozenset({"judge"}) if "reviewer" in roles else frozenset())


def _knob_model_roles(knobs: Iterable[str]) -> frozenset[str]:
    """The unit roles whose model a changed knob selects (:attr:`entries.Knob.selects_model`).

    Their whole and turn-memo base keys hash the model, so none of their child units can replay the parent's turns.
    A role without units of its own under the policy in force (the reviewer before ``roles/1``) matches no unit.

    Args:
        knobs (Iterable[str]): Changed knob names (:data:`entries.KNOBS`); unknown ones select nothing.

    Returns:
        frozenset[str]: Unit role names, e.g. ``{"judge", "reviewer"}`` for ``judge_model``.
    """
    return frozenset(role for name in knobs if name in E.KNOBS and E.KNOBS[name].selects_model
                     for role, _route in E.KNOBS[name].visible)


def _post_review_answer(parent_unit: Mapping[str, Any], child_unit: Mapping[str, Any], knobs: Iterable[str]) -> bool:
    """Allow a skipped reviewer's declared hook effect only over unchanged producer SQL.

    The producer SQL is the pre-hook SQL of the first recorded hook in :data:`entries.KNOBS` order
    (guard, gate, count-weight, selection-normalization, entity-percentage-count), else the unit's SQL, so composed
    hooks share one producer check even when only the last hook's knob changes.
    """
    if child_unit.get("role") != "judge":
        return False
    parent_hooks = parent_unit.get("post_review_pre_sql", {})
    child_hooks = child_unit.get("post_review_pre_sql", {})

    def producer(unit: Mapping[str, Any], hooks: Mapping[str, Any]) -> Any:
        # Hooks run in KNOBS order, so the first recorded hook saw the judge's own SQL.
        return next((hooks[name] for name in E.KNOBS if name in hooks), unit.get("sql"))

    for name in knobs:
        if name not in E.KNOBS or E.KNOBS[name].post_review_answer_hook is None:
            continue
        if name not in parent_hooks and name not in child_hooks:
            continue
        before, after = producer(parent_unit, parent_hooks), producer(child_unit, child_hooks)
        if before and after and before == after:
            return True
    return False


def _corpus_identity(release: Any) -> tuple[str, str] | None:
    """A release's rule corpus as knowledge reads it, whatever carries it: ``(rules digest, corpus text)``, or ``None``."""
    from .knowledge import corpus_rules_digest

    try:
        corpus = release.rules_general()
    except KeyError:
        return None
    return corpus_rules_digest(release), corpus_text(corpus)


def knowledge_differences(release: Any, other: Any) -> list[str]:
    """The knowledge rows — the rule corpus and the generator documents — two releases both carry with different texts.

    They join no keyspace, so two releases that send the same prompts may still differ here, and the runs made for them
    read different knowledge (the loop evaluates a rules or generator proposal on knowledge prepared for it). The rule
    corpus compares as knowledge reads it (:func:`knowledge.corpus_rules_digest` and the corpus text), whether a
    ``rules.general`` row or typed rule versions carry it, so a closure/1 release and its representation-equivalent
    closure/2 release do not differ and a typed corpus is never equal for want of a carrier. A row only one of them
    carries — a release published before the rows moved into the release — is no difference.

    Args:
        release (Release): One release.
        other (Release): The other.

    Returns:
        list[str]: The row names that differ, sorted; empty when none does.
    """
    names = [name for name in set(release.refs) & set(other.refs)
             if name.startswith("gen.") and release.refs[name] != other.refs[name]]
    mine, theirs = _corpus_identity(release), _corpus_identity(other)
    if mine is not None and theirs is not None and mine != theirs:
        names.append(E.RULES_ENTRY)
    return sorted(names)


def _tool_identity(release: Any) -> tuple[tuple[str, str], ...] | None:
    """The executable tool identity a release serves: ``(name, implementation version)`` in serve order.

    A closure/2 release derives it from its Tool versions, a closure/1 release from its schema2 ``tools.answering``
    callable versions (equal by construction, :func:`hb_tool_entities.implementation_version`); a release without tools
    serves none, and a historical schema1 bundle has no comparable identity (``None``).
    """
    from ..hb_tool_entities import BUNDLE_SCHEMA, implementation_version, tool_definition

    if getattr(release, "rules", None) == EM.CLOSURE_V2:
        rows = release.typed_rows.get(EM.TOOL_ENTITY, {})
        served = [rows[ref["object_id"]] for ref in release.content_refs["tools"]]
        return tuple((row["name"], implementation_version(tool_definition(row), row["implementation"])) for row in served)
    if "tools.answering" not in release.refs:
        return ()
    payload = json.loads(release.row("tools.answering")["manifest"]["meta"]["text"])
    if payload.get("schema") != BUNDLE_SCHEMA:
        return None
    return tuple((str(ref["name"]), str(ref["version"])) for ref in payload["tools"])


def same_run_release(release: Any, other: Any, *, store: Store | None = None) -> bool:
    """Whether another release measures the same prompts, knowledge and tools as a loaded release.

    Two closure/1 releases serve the same tools when they pin the same ``tools.answering`` row; otherwise the tool
    definitions and implementation versions are compared (:func:`_tool_identity`), so a release and its
    representation-equivalent closure/2 release serve the same tools.

    Args:
        release (Release): The loaded reference release.
        other (Any): Another release or proposal id; unknown releases do not qualify.
        store (Store | None): Store layers used to load the other release; the runtime's default when absent.

    Returns:
        bool: Whether the ids agree, or the prompt keyspaces, shared knowledge rows and executable tools agree.
    """
    from .runtime import load_release, same_prompts

    if other == release.id:
        return True
    if not same_prompts(release, other, stores=store):
        return False
    try:
        candidate = load_release(other, stores=store, allow_retired=True)
    except PromptStoreError:
        return False
    if knowledge_differences(release, candidate):
        return False
    if EM.CLOSURE_V2 not in (getattr(release, "rules", None), getattr(candidate, "rules", None)):
        return release.refs.get("tools.answering") == candidate.refs.get("tools.answering")
    mine = _tool_identity(release)
    return mine is not None and mine == _tool_identity(candidate)


def changed_knobs(release: Any, parent: Any) -> frozenset[str]:
    """Knobs whose value differs between two releases, whatever the environment sets (:meth:`runtime.Release.knob_settings`).

    Args:
        release (Release): The child.
        parent (Release): The parent.

    Returns:
        frozenset[str]: Knob names.
    """
    mine, theirs = release.knob_settings(), parent.knob_settings()
    return frozenset(name for name in mine if mine[name] != theirs.get(name))


def _pair_rows(parent: RunView, child: RunView, metric: M.Metric, ctx: M.ExposureContext, *, strict_sources: bool = False, implementation_attribution: str | None = None, implementation_contexts: Mapping[str, Any] | None = None, retrieval_integrity: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any], set[str]]:
    """Build the recomputable pairing table of two runs.

    One ``q`` row per shared question carries the metric values of both arms (``None`` when neither arm
    exposes the row, so the table reproduces :func:`metrics.paired` exactly); one row per agent unit
    carries the final-SQL digests and the child's divergence turn and cause.

    The integrity detail counts every child unit by cause bucket (:data:`CAUSE_BUCKETS`) and, over the
    *paired* units (a child unit with a parent counterpart): ``undiverged`` units whose final SQL must
    match the parent's (``violations`` when it does not), ``noise`` units that diverged for a non-prompt
    reason (they dilute the pairing without biasing it) and ``unverified`` ``no-source`` units (a complete
    source replay is unproven, so neither a prompt divergence nor an integrity check can be claimed). A
    ``no-source`` unit whose first-request digests differ from the parent's is settled as ``initial``
    (:func:`_settled_cause`); the table keeps the memo's ``recorded_cause`` beside it. When the changed surfaces
    include the rule corpus or a generator document, every child unit that did not diverge otherwise is settled
    as a ``knowledge`` divergence, since none can replay. A judge unit is compared only when no unit of its
    question diverged or went unverified. When the changed knobs (``ctx.changed_knobs``) act on a unit's role, its
    non-prompt divergences, and in the roles whose answer they can change an undiverged unit whose final SQL changed,
    are settled as a ``knob`` divergence (:func:`_settled_cause`); so are the ``no-source`` units of a role whose
    model a changed knob selects (:func:`_knob_model_roles`).

    Returns:
        tuple[list[dict[str, Any]], dict[str, Any], set[str]]: The table, the integrity detail, and the qids
            whose child units diverged on the prompt (``prompt:*``, ``initial``, ``knowledge`` or ``knob``),
            also including declared implementation effects under ``roles/1``. That policy counts a separate
            reviewer unit and compares the judge's retained pre-review SQL; no other role's implementation
            difference excuses its noise or a missing immutable replay source.
    """
    knowledge = bool(E.surfaces_of(ctx.changed_surfaces) & KNOWLEDGE_SURFACES)
    role_policy = implementation_attribution == _IMPLEMENTATION_ATTRIBUTION
    knob_turns, knob_answers = _knob_roles(ctx.changed_knobs, implementation_attribution=implementation_attribution)
    knob_models = _knob_model_roles(ctx.changed_knobs)
    contexts = implementation_contexts if implementation_contexts is not None else ({"parent": _implementation_contexts(parent), "child": _implementation_contexts(child)} if role_policy else {})
    table: list[dict[str, Any]] = []
    divergences = {bucket: 0 for bucket in CAUSE_BUCKETS}
    if role_policy:
        divergences[CAUSE_IMPLEMENTATION] = 0
    detail: dict[str, Any] = {"paired_units": 0, "undiverged": 0, "violations": 0, "rate": 0.0, "noise": 0, "noise_rate": 0.0, "unverified": 0, "unverified_rate": 0.0, "divergences": divergences}
    if retrieval_integrity:
        detail["degraded_retrieval"] = {"parent": 0, "child": 0, "units": 0, "total": 0, "rate": 0.0}
    prompt_diverged: set[str] = set()
    for qid in sorted(set(parent.rows) & set(child.rows)):
        prow, crow = parent.rows[qid], child.rows[qid]
        db = str(crow.get("db_id") or prow.get("db_id") or "")
        p_eval, c_eval = M.EvalRow(prow), M.EvalRow(crow)
        if metric.exposed(p_eval, ctx) or metric.exposed(c_eval, ctx):
            pv, cv = metric.value(p_eval), metric.value(c_eval)
        else:
            pv = cv = None
        table.append({"qid": qid, "db": db, "unit": "q", "parent": pv, "child": cv, "diverged_at": None, "cause": None})
        punits, cunits = _units(prow, implementation_attribution=implementation_attribution), _units(crow, implementation_attribution=implementation_attribution)
        settled = {label: _settled_cause(punits.get(label, {}), unit, knowledge, unit["role"] in knob_turns,
                                       unit["role"] in knob_answers or (role_policy and _post_review_answer(punits.get(label, {}), unit, ctx.changed_knobs)),
                                       knob_model=unit["role"] in knob_models,
                                       strict_sources=strict_sources, implementation=bool(role_policy and _implementation_change(punits.get(label, {}), unit, contexts)),
                                       implementation_attribution=implementation_attribution,
                                       # A nameless view cannot prove a non-parent producer; no self/sibling credit.
                                       parent_run=getattr(parent, "name", None)) for label, unit in cunits.items()}
        for label in sorted(set(punits) | set(cunits), key=lambda item: (item != "j", item)):
            pu, cu = punits.get(label, {}), cunits.get(label, {})
            cause = settled.get(label, cu.get("cause"))
            bucket = _cause_bucket(cause, implementation_attribution=implementation_attribution)
            if bucket in ("prompt", "initial", CAUSE_KNOWLEDGE, CAUSE_KNOB) or (role_policy and bucket == CAUSE_IMPLEMENTATION):
                prompt_diverged.add(qid)
            entry = {"qid": qid, "db": db, "unit": label, "parent": pu.get("sql", ""), "child": cu.get("sql", ""), "diverged_at": cu.get("diverged_at"), "cause": cause}
            if retrieval_integrity:
                entry.update(parent_degraded=bool(pu.get("degraded_retrieval")),
                             child_degraded=bool(cu.get("degraded_retrieval")))
            if role_policy:
                entry.update(paired=label in punits and label in cunits, child_present=label in cunits)
                change = _implementation_change(pu, cu, contexts)
                if change is not None:
                    entry["implementation_change"] = change
            if cause != cu.get("cause"):
                entry["recorded_cause"] = cu.get("cause")
            table.append(entry)
            _count_integrity_row(detail, entry, settled, paired=label in punits and label in cunits,
                                 child_present=label in cunits, implementation_attribution=implementation_attribution)
    _integrity_rates(detail)
    return table, detail, prompt_diverged


def _upstream_diverged(label: str, causes: Mapping[str, Any]) -> bool:
    """Exclude answer equality only for a role whose own upstream inputs diverged."""
    if label not in ("j", "r"):
        return False
    return any(cause is not None for unit, cause in causes.items()
               if unit.startswith("c") or (label == "r" and unit == "j"))


def _count_integrity_row(detail: dict[str, Any], row: Mapping[str, Any], causes: Mapping[str, Any], *, paired: bool, child_present: bool, implementation_attribution: str | None = None) -> None:
    """Count one unit's replay integrity with the policy's upstream answer exclusion."""
    bucket = _cause_bucket(row.get("cause"), implementation_attribution=implementation_attribution)
    if "degraded_retrieval" in detail:
        degraded = detail["degraded_retrieval"]
        parent, child = bool(row.get("parent_degraded")), bool(row.get("child_degraded"))
        degraded["parent"] += parent
        degraded["child"] += child
        degraded["units"] += parent or child
        degraded["total"] += 1
    if child_present:
        detail["divergences"][bucket] += 1
    if not paired:
        return
    detail["paired_units"] += 1
    if bucket == CAUSE_NO_SOURCE:
        detail["unverified"] += 1
    elif bucket in _NOISE_BUCKETS:
        detail["noise"] += 1
    elif bucket == "none":
        label = str(row.get("unit"))
        upstream = (_upstream_diverged(label, causes) if implementation_attribution == _IMPLEMENTATION_ATTRIBUTION
                    else label == "j" and any(_cause_bucket(cause) != "none" for cause in causes.values()))
        if not upstream:
            detail["undiverged"] += 1
            detail["violations"] += row.get("parent") != row.get("child")


def _integrity_rates(detail: dict[str, Any]) -> None:
    """Finish counts under the policies actually recorded in this outcome."""
    if detail["undiverged"]:
        detail["rate"] = round(detail["violations"] / detail["undiverged"], 6)
    if detail["paired_units"]:
        detail["noise_rate"] = round(detail["noise"] / detail["paired_units"], 6)
        detail["unverified_rate"] = round(detail["unverified"] / detail["paired_units"], 6)
    degraded = detail.get("degraded_retrieval")
    if degraded and degraded["total"]:
        degraded["rate"] = round(degraded["units"] / degraded["total"], 6)


def integrity_state(detail: Mapping[str, Any]) -> tuple[str, str]:
    """Classify a pairing's replay integrity from its detail counts.

    Args:
        detail (Mapping[str, Any]): ``integrity_detail`` as :func:`_pair_rows` builds it.

    Returns:
        tuple[str, str]: ``("ok", "")``; ``("invalid", why)`` when undiverged units changed their final SQL beyond
            :data:`INTEGRITY_MAX_VIOLATION_RATE` or non-prompt divergences exceed :data:`INTEGRITY_MAX_DILUTION_RATE`
            of the paired units, or failed retrievals exceed that share of agent units;
            ``("unverifiable", why)`` when ``no-source`` units exceed that share.
    """
    degraded = detail.get("degraded_retrieval") or {}
    degraded_rate = float(degraded.get("rate") or 0.0)
    if degraded_rate > INTEGRITY_MAX_DILUTION_RATE:
        return "invalid", (f"{degraded.get('units', 0)} of {degraded.get('total', 0)} agent units consumed failed knowledge retrieval "
                           f"(parent={degraded.get('parent', 0)}, child={degraded.get('child', 0)}; "
                           f"{degraded_rate:.2%} > {INTEGRITY_MAX_DILUTION_RATE:.0%})")
    rate = float(detail.get("rate") or 0.0)
    if rate > INTEGRITY_MAX_VIOLATION_RATE:
        return "invalid", f"{detail.get('violations', 0)} of {detail.get('undiverged', 0)} undiverged units changed their final SQL ({rate:.2%} > {INTEGRITY_MAX_VIOLATION_RATE:.0%})"
    noise_rate = float(detail.get("noise_rate") or 0.0)
    if noise_rate > INTEGRITY_MAX_DILUTION_RATE:
        return "invalid", f"{detail.get('noise', 0)} of {detail.get('paired_units', 0)} paired units diverged for a non-prompt reason (tool/clock/unknown: {noise_rate:.2%} > {INTEGRITY_MAX_DILUTION_RATE:.0%})"
    unverified_rate = float(detail.get("unverified_rate") or 0.0)
    if unverified_rate > INTEGRITY_MAX_DILUTION_RATE:
        return "unverifiable", f"{detail.get('unverified', 0)} of {detail.get('paired_units', 0)} paired units had no source episode to replay against ({CAUSE_NO_SOURCE}: {unverified_rate:.2%} > {INTEGRITY_MAX_DILUTION_RATE:.0%})"
    return "ok", ""


def _exposed_qids(parent: RunView, child: RunView, exposure: str, changed: frozenset[str], prompt_diverged: set[str], *, implementation_attribution: str | None = None) -> set[str]:
    if exposure == "diverged:prompt":
        return set(prompt_diverged)
    exposed: set[str] = set()
    for qid in set(parent.rows) & set(child.rows):
        fired: set[str] = set()
        for units in (_units(parent.rows[qid], implementation_attribution=implementation_attribution), _units(child.rows[qid], implementation_attribution=implementation_attribution)):
            for unit in units.values():
                fired.update(unit.get("fired") or [])
        if E.fired_surfaces(fired) & E.surfaces_of(changed):
            exposed.add(qid)
    return exposed


def _scored(run: RunView) -> set[str]:
    """Questions the run scored (a row with an EX value)."""
    return {qid for qid, row in run.rows.items() if M.EvalRow(row).ex is not None}


def question_set_record(ids: Iterable[str]) -> dict[str, Any]:
    """The ``questions`` record of a run manifest: how many distinct question ids there are and the digest of them.

    ``eval_bird`` records it for every question of the queries file before ``--db``, ``--ids`` or ``--limit``
    subset the run; :func:`outcome` computes it over the questions a run scored, so a run that scored exactly its
    question set's questions gives back its manifest's record.

    Args:
        ids (Iterable[str]): Question ids (duplicates count once).

    Returns:
        dict[str, Any]: ``{"count": int, "ids_sha256": "sha256:<hex of the sorted ids joined by newlines>"}``.
    """
    unique = sorted({str(qid) for qid in ids})
    return {"count": len(unique), "ids_sha256": "sha256:" + sha256_hex("\n".join(unique))}


def _coverage(parent: RunView, child: RunView) -> dict[str, Any]:
    """Which questions each run scored, against the question list its manifest records (:func:`verdict_gate` reads it).

    ``status`` is ``full`` when both runs scored exactly the questions their manifests record for their question
    set, ``partial`` otherwise (a subset run, extra rows, or a manifest that records no question list).
    """
    parent_set, child_set = _scored(parent), _scored(child)
    missing, extra = sorted(parent_set - child_set), sorted(child_set - parent_set)
    recorded = {"parent": parent.questions, "child": child.questions}
    complete = {"parent": recorded["parent"] == question_set_record(parent_set), "child": recorded["child"] == question_set_record(child_set)}
    status = "full" if complete["parent"] and complete["child"] and not missing and not extra else "partial"
    result = {
        "status": status,
        "question_set": str(child.identity.get("question_set") or ""),
        "questions": {arm: (record["count"] if record is not None else None) for arm, record in recorded.items()},
        "complete": complete,
        "parent": len(parent_set),
        "child": len(child_set),
        "missing": len(missing),
        "extra": len(extra),
        "examples": (missing + extra)[:10],
    }
    gold_rows = {qid for run in (parent, child) for qid, row in run.rows.items() if row.get("gold_execution")}
    if gold_rows:
        mismatches = []
        unscorable = []
        for qid in sorted(gold_rows):
            left = parent.rows.get(qid, {}).get("gold_execution") or {}
            right = child.rows.get(qid, {}).get("gold_execution") or {}
            if not left.get("key") or any(left.get(key) != right.get(key)
                                          for key in ("key", "status", "result_sha256")):
                mismatches.append(qid)
            elif left.get("status") == "gold_unscorable":
                unscorable.append(qid)
        result["gold"] = {"unscorable": unscorable, "unscorable_count": len(unscorable),
                          "policy": "zero in both arms; fixed question-set denominator",
                          "mismatches": mismatches}
        if mismatches:
            result["status"] = "partial"
    return result


def _provenance(child: RunView, pid: str, plan_digest: Any, registration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Whether the run was started under this proposal's pin, so its frozen plan predates the results."""
    if registration is not None:
        if (registration.get("proposal") != pid or registration.get("plan_digest") != plan_digest
                or registration.get("child") != {"manifest_sha256": child.manifest_sha, "result_files": child.result_files}):
            raise EvolveError("finished registration differs from immutable child evidence")
        return {"status": "registered", "proposal": pid, "plan_digest": plan_digest,
                "reason": "registered after execution", "registration": registration["reservation_id"]}
    declared_pid = child.manifest.get("proposal")
    declared_digest = child.manifest.get("plan_digest")
    record = {"status": "registered", "proposal": declared_pid, "plan_digest": declared_digest, "reason": ""}
    if declared_pid is None:
        record.update(status="unregistered", reason="the run was pinned by release id, not by this proposal (manifest.proposal is null): its plan was not frozen before the results existed")
    elif declared_pid != pid:
        record.update(status="unregistered", reason=f"the run was started under proposal {declared_pid}, not {pid}")
    elif declared_digest != plan_digest:
        record.update(status="unregistered", reason=f"the run's plan_digest {declared_digest} is not this proposal's {plan_digest}")
    return record


def _outcome_files(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Every ``rubiksql.outcome/1`` document under one proposal's outcome directory, by file name."""
    docs = []
    for path in sorted(directory.glob("o*.json")) if directory.is_dir() else []:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, Mapping) and data.get("schema") == SCHEMA_OUTCOME:
            docs.append((path, dict(data)))
    return docs


def _proposal_release(store: Store, pid: str) -> str | None:
    try:
        release = store.proposal(pid).data.get("release")
    except PromptStoreError:
        return None
    return str(release) if isinstance(release, str) else None


def outcomes_of(store: Store, pid: str) -> list[dict[str, Any]]:
    """Read exact outcomes of a proposal or an adopted current release, oldest first.

    A document under ``outcomes/<pid>/`` counts only when its ``proposal`` is ``pid`` and its ``release``
    is the proposal's release; a file copied in from another proposal's directory is ignored here and
    reported by :func:`misfiled_outcomes`. An adopted release uses ``target`` instead of a proposal
    and records only an unpaired descriptive baseline.

    Args:
        store (Store): The store.
        pid (str): Proposal id or exact adopted release id.

    Returns:
        list[dict[str, Any]]: Outcome documents sorted by ``created_at``.
    """
    release = _proposal_release(store, pid)
    anchor = _adopted_state(store, pid)
    docs = [doc for _path, doc in _outcome_files(store.package / "outcomes" / pid)
            if ((doc.get("proposal") == pid and doc.get("release") == release)
                or (doc.get("proposal") is None and doc.get("target") == doc.get("release") == pid
                    and ((anchor is not None and doc.get("authority_state") == anchor)
                         or _baseline_decision_valid(store, doc))))]
    return sorted(docs, key=_outcome_order)


def _baseline_decision_valid(store: Store, document: Mapping[str, Any]) -> bool:
    """Check a control's immutable pointer anchor against recorded history."""
    return any(decision.get("id") == document.get("baseline_decision")
               and decision.get("kind") in POINTER_KINDS
               and decision.get("release") == document.get("target") for decision in store.decisions())


def _adopted_state(store: Store, rid: str) -> str | None:
    if RELEASE_RE.fullmatch(rid) is None:
        return None
    try:
        release = store.release(rid).data
    except PromptStoreError:
        return None
    anchor = release.get("authority_state")
    if release.get("origin") == "v0" and isinstance(anchor, str):
        from .fresh_epoch import validate_native_admission
        from .store import TABLES

        event = validate_native_admission({entity: store.native(entity) for entity in TABLES})
        return anchor if event is not None and event["object_id"] == anchor and event["payload"]["release"] == rid else None
    return anchor if release.get("origin") == "state" and isinstance(anchor, str) else None


def _outcome_order(doc: Mapping[str, Any]) -> tuple[float, str]:
    return float(doc.get("created_at") or 0.0), str(doc.get("id"))


def misfiled_outcomes(store: Store) -> list[str]:
    """Outcome files whose directory, ``proposal`` field or ``release`` field disagree with each other or the store.

    Args:
        store (Store): The store.

    Returns:
        list[str]: One line per offending file (path relative to the package store, and why), in path order.
    """
    problems: list[str] = []
    root = store.package / "outcomes"
    for directory in sorted(root.iterdir()) if root.is_dir() else []:
        if not directory.is_dir() or not (PROPOSAL_RE.fullmatch(directory.name) or RELEASE_RE.fullmatch(directory.name)):
            continue
        anchor = _adopted_state(store, directory.name)
        release = _proposal_release(store, directory.name)
        for path, doc in _outcome_files(directory):
            relative = path.relative_to(store.package).as_posix()
            if RELEASE_RE.fullmatch(directory.name):
                if _baseline_decision_valid(store, doc):
                    if doc.get("proposal") is not None or not doc.get("target") == doc.get("release") == directory.name:
                        problems.append(f"{relative}: control target differs from its pointer decision")
                elif anchor is None:
                    problems.append(f"{relative}: release {directory.name} has no current-state anchor")
                elif doc.get("proposal") is not None or doc.get("target") != directory.name:
                    problems.append(f"{relative}: target is not the adopted release {directory.name}")
                elif doc.get("authority_state") != anchor:
                    problems.append(f"{relative}: current-state admission differs from release {directory.name}")
                elif doc.get("release") != directory.name:
                    problems.append(f"{relative}: names release {doc.get('release')}, the adopted target is {directory.name}")
            elif release is None:
                problems.append(f"{relative}: proposal {directory.name} is not in the store")
            elif doc.get("proposal") != directory.name:
                problems.append(f"{relative}: names proposal {doc.get('proposal')} but lies under {directory.name}")
            elif doc.get("release") != release:
                problems.append(f"{relative}: names release {doc.get('release')}, the proposal's release is {release}")
    return problems


#: Verdicts a paired outcome carries; ``descriptive`` carries none.
VERDICTS = ("promotable", "adoptable", "not-promotable")


def attempt_marker_path(store: Store, generation: str, oid: str) -> Path:
    """Where the marker that counts one evaluation toward a generation's attempt cap lives.

    Args:
        store (Store): The store.
        generation (str): The BASELINE release id the evaluation was computed under.
        oid (str): The outcome id of the run pair (:func:`outcome_id`).

    Returns:
        Path: ``<package>/attempts/<generation>/<oid>.json``.
    """
    return store.package / ATTEMPTS_DIR / generation / f"{oid}.json"


def _read_marker(path: Path) -> dict[str, Any]:
    """An attempt marker's document, ``{}`` when it is not a JSON object (the file still counts, by its name)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def _markers(store: Store, generation: str) -> list[dict[str, Any]]:
    """The attempt markers one generation holds, each with its file name as ``id``."""
    directory = store.package / ATTEMPTS_DIR / generation
    if not RELEASE_RE.fullmatch(generation) or not directory.is_dir():
        return []
    return [{**_read_marker(path), "id": path.stem} for path in sorted(directory.glob("o*.json")) if _OUTCOME_ID_RE.fullmatch(path.stem)]


def generation_attempts(store: Store, generation: str, *, question_set: str | None = None) -> list[dict[str, Any]]:
    """The evaluations one BASELINE generation has counted, oldest first: what the attempt cap counts.

    Every run pair :func:`outcome` computed a verdict for under ``generation`` — recorded, or only shown by a dry
    run — left a marker ``attempts/<generation>/<outcome id>.json`` and counts once; a paired outcome recorded
    without one (a verdict-carrying document of the generation, including ``adoptable``) counts too.
    A ``descriptive`` outcome, whose run could not be paired, does not. Documents misfiled under another proposal's
    directory are skipped (:func:`misfiled_outcomes` reports them). The count is the store's: two checkouts that each
    count evaluations and are merged afterwards hold more than either counted, so the cap binds per checkout.

    Args:
        store (Store): The store.
        generation (str): The BASELINE release id the generation is named after.

    Returns:
        list[dict[str, Any]]: Markers and outcome documents, one per evaluation, each with ``id``, in the order they
            were counted.
    """
    scope = store.v0_scope()
    if scope is not None and generation not in scope.releases:
        raise PromptStoreError(f"generation {generation} lies outside v0 history rooted at {scope.root}")
    directory = store.package / "reservations"
    reserved = [_read_marker(path) for path in sorted(directory.glob("o*.json"))] if directory.is_dir() else []
    from .paid_launch import attempt_question_set

    question_set = question_set or next(iter(PROMOTE_QUESTION_SETS))
    reserved = [row for row in reserved if row.get("generation") == generation and row.get("counted") is True
                and attempt_question_set(row) == question_set]
    ids = {row["id"] for row in reserved}
    attached = [row for row in _markers(store, generation) if row.get("reservation_id") not in ids
                and attempt_question_set(row) == question_set]
    return sorted([*reserved, *attached], key=_attempt_rank)



def _attempt(store: Store, generation: str, allow_over_cap: bool, *, exclude: str) -> dict[str, Any]:
    """The ``attempt`` record of a new evaluation: the other evaluations the generation holds, by id, and the override.

    Naming what was counted, rather than a position in time, keeps every recorded count verifiable after checkouts
    are merged or a clock is set back (:func:`_attempt_problems`); ``exclude`` is the evaluation's own id.
    """
    counted = sorted(str(record["id"]) for record in generation_attempts(store, generation) if record["id"] != exclude)
    return {"used": len(counted), "counted": counted, "allow_over_cap": bool(allow_over_cap)}


def _child_markers(store: Store, pid: str, child: RunView) -> list[dict[str, Any]]:
    """The attempt markers, under any generation, of pairs of this proposal whose child run is ``child``."""
    root = store.package / ATTEMPTS_DIR
    found = []
    for directory in sorted(root.iterdir()) if root.is_dir() else []:
        for record in _markers(store, directory.name) if directory.is_dir() else []:
            pair = record.get("run_pair") if isinstance(record.get("run_pair"), Mapping) else {}
            if record.get("proposal") == pid and pair.get("child") == {"manifest_sha256": child.manifest_sha, "result_files": child.result_files}:
                found.append(record)
    return found


def release_measurement_blocker(release: str, generation: str, *, attachments: Iterable[Mapping[str, Any]],
                                outcomes: Iterable[Mapping[str, Any]], reservations: Iterable[Mapping[str, Any]],
                                exclude_reservation: str | None = None,
                                question_set: str = EVOLUTION_QUESTION_SET) -> Mapping[str, Any] | None:
    """Find an earlier measurement that prevents another counted attempt of a release.

    Args:
        release: Content-addressed release to measure.
        generation: BASELINE generation in which to check earlier attempts.
        attachments: ``eval-attach`` payloads (or their exported attempt markers).
        outcomes: Recorded outcome documents, carrying replay integrity.
        reservations: ``eval-reserve`` payloads (or their exported reservations).
        exclude_reservation: The current outcome's already validated reservation, if any.

    Returns:
        Mapping[str, Any] | None: A blocking attachment or unfinished counted reservation,
        or ``None`` if every earlier attachment has an invalid/unverifiable outcome and
        every earlier counted reservation is attached. Missing outcomes fail closed.
    """
    from .paid_launch import attempt_question_set

    attached = [row for row in attachments if row.get("release") == release and row.get("generation") == generation
                and attempt_question_set(row) == question_set]
    documents = list(outcomes)
    for row in attached:
        matches = [doc for doc in documents if doc.get("id") == row.get("outcome", row.get("id"))
                   and (doc.get("release"), doc.get("generation"), doc.get("proposal"))
                   == (release, generation, row.get("proposal"))]
        if not matches or any(doc.get("integrity") not in {"unverifiable", "invalid"} for doc in matches):
            return row
    attached_ids = {row.get("reservation_id") for row in attached if row.get("reservation_id")}
    return next((row for row in reservations if row.get("counted") is True
                 and row.get("release") == release and row.get("generation") == generation
                 and attempt_question_set(row) == question_set
                 and row.get("id") != exclude_reservation and row.get("id") not in attached_ids), None)


def _authority_count_attempt(authority: Any, reader: Any, generation: str, oid: str, *, pid: str, rid: str,
                             child: RunView, parent: RunView, allow_over_cap: bool, dry_run: bool,
                             cap: int, now: float, actor: str,
                             registration: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], str]:
    """Reserve one create-only native attempt before any statistic is computed."""
    if child.diagnostic or parent.diagnostic:
        raise EvolveError("diagnostic measurements cannot consume a promotion attempt")
    pair = {"child": {"manifest_sha256": child.manifest_sha, "result_files": child.result_files},
            "parent": {"manifest_sha256": parent.manifest_sha, "result_files": parent.result_files}}
    from .paid_launch import counted_attempts

    launch = child.manifest.get("paid_launch") or {}
    if not isinstance(launch, Mapping):
        raise EvolveError("outcome paid launch evidence is malformed")
    reservation_id = registration["reservation_id"] if registration is not None else launch.get("reservation_id") if isinstance(launch, Mapping) else None
    reservations = [event["payload"] for event in reader.events("eval-reserve")
                    if event["payload"].get("id") == reservation_id]
    reservation = reservations[0] if len(reservations) == 1 else None
    events = reader.events("eval-attach")
    for event in events:
        record = event["payload"]
        if record.get("id") == oid:
            if (record.get("run_pair"), record.get("proposal"), record.get("release"), record.get("generation")) != (pair, pid, rid, generation):
                raise EvolveError("existing attempt has different complete run-pair semantics")
            return dict(record["attempt"]), str(event["object_id"])
        if record.get("proposal") == pid and record.get("run_pair", {}).get("child") == pair["child"]:
            raise EvolveError("child run already took a verdict with a different control")
    counted = sorted(str(record["id"]) for record in counted_attempts(reader.events(), generation,
                                                                    question_set=child.identity["question_set"]))
    if reservation is not None:
        if (reservation.get("proposal"), reservation.get("release"), reservation.get("generation"), reservation.get("run")) != (pid, rid, generation, child.name):
            raise EvolveError("outcome differs from its paid reservation")
        from .paid_launch import reservation_claims

        claims = reservation_claims(reader, reservation)
        finished = registration is not None and reservation.get("registration") == registration
        if ((not finished and (len(claims) != 1 or claims[0].get("token") != launch.get("token")))
                or reservation.get("parent", {}).get("manifest_sha256") != parent.manifest_sha
                or reservation.get("parent", {}).get("result_files") != parent.result_files):
            raise EvolveError("outcome has no matching one-use claim and immutable parent")
        attempt = dict(reservation["attempt"])
    elif reader.git_binding is not None and not dry_run:
        raise EvolveError("Git child outcomes require their committed pre-launch paid reservation")
    elif len(counted) >= cap and not allow_over_cap:
        raise EvolveError(f"over cap: generation {generation} already took {len(counted)} evaluations, cap {cap}; no statistic computed")
    else:
        attempt = {"used": len(counted), "counted": counted, "allow_over_cap": bool(allow_over_cap)}
    blocker = release_measurement_blocker(rid, generation, attachments=(event["payload"] for event in events),
        outcomes=(event["payload"] for event in reader.events("outcome")),
        reservations=(event["payload"] for event in reader.events("eval-reserve")),
        exclude_reservation=reservation_id if reservation is not None else None,
        question_set=child.identity["question_set"])
    if blocker is not None:
        if blocker.get("counted") is True:
            raise EvolveError("release has an unfinished paid reservation in this generation; follow that execution")
        raise EvolveError("release was already measured in this generation")
    payload = {"schema": SCHEMA_ATTEMPT, "id": oid, "outcome": oid, "proposal": pid, "release": rid,
               "question_set": child.identity.get("question_set"),
               "generation": generation, "run": child.name, "parent_run": parent.name, "run_pair": pair,
               "attempt": attempt, "prior_counted_ids": attempt["counted"], "used": attempt["used"], "cap": cap,
               "dry_run": bool(dry_run), "allow_over_cap": bool(allow_over_cap), "created_at": now,
               **({"reservation_id": reservation_id} if reservation is not None else {})}
    event_id = authority._event("eval-attach", pid, payload, actor) if not dry_run else ""
    return attempt, event_id


def _attempt_rank(doc: Mapping[str, Any]) -> tuple[int, float, str]:
    """Order of two evaluations of one generation: a later one counted every earlier one of its checkout."""
    attempt = doc.get("attempt") if isinstance(doc.get("attempt"), Mapping) else {}
    used = attempt.get("used")
    return (used if isinstance(used, int) and not isinstance(used, bool) else 0, *_outcome_order(doc))


def latest_verdict(docs: Iterable[Mapping[str, Any]], generation: str) -> dict[str, Any] | None:
    """The latest verdict among a proposal's outcomes: the one :func:`promote` acts on.

    Only an outcome computed under ``generation`` (the BASELINE a promotion would replace) that carries a verdict
    counts; a ``descriptive`` outcome carries none, and one computed under another BASELINE measured another move.
    The newest is the one with the larger recorded attempt count — every evaluation counts the generation's
    earlier ones when it is first computed, so the order survives a clock set back — then the later
    ``created_at`` (two checkouts merged).

    Args:
        docs (Iterable[Mapping[str, Any]]): Outcome documents of one proposal.
        generation (str): The current BASELINE release id.

    Returns:
        dict[str, Any] | None: The outcome, or ``None`` when no outcome under ``generation`` carries a verdict.
    """
    candidates = [dict(doc) for doc in docs if doc.get("generation") == generation and doc.get("verdict") in VERDICTS]
    return max(candidates, key=_attempt_rank) if candidates else None


def _paired_stats(metric: Any, parent_rows: list[Mapping[str, Any]], child_rows: list[Mapping[str, Any]], ctx: M.ExposureContext) -> dict[str, Any]:
    result = M.paired(metric, parent_rows, child_rows, ctx)
    return {"n": result.n, "b": result.b, "c": result.c, "effect_pp": round(result.effect_pp, 4)}


def _q_pairs(table: Iterable[Mapping[str, Any]]) -> tuple[int, int, int, float, float]:
    """``(b, c, n, parent_sum, child_sum)`` over the ``q`` rows of a pairing table that both arms value."""
    b = c = n = 0
    parent_sum = child_sum = 0.0
    for row in table:
        if row.get("unit") != "q" or row.get("parent") is None or row.get("child") is None:
            continue
        n += 1
        parent_sum += float(row["parent"])
        child_sum += float(row["child"])
        if row["parent"] > row["child"]:
            b += 1
        elif row["child"] > row["parent"]:
            c += 1
    return b, c, n, parent_sum, child_sum


def _per_db(table: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Discordant counts and net effect of the ``q`` rows both arms value, per database."""
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for row in table:
        if row.get("unit") == "q" and row.get("parent") is not None and row.get("child") is not None:
            buckets.setdefault(str(row.get("db") or ""), []).append(row)
    result = {}
    for db, rows in sorted(buckets.items()):
        b, c, n, _parent, _child = _q_pairs(rows)
        result[db] = {"n": n, "b": b, "c": c, "effect_pp": round(M.effect_pp(b, c, n), 4)}
    return result


def _exposed_stats(parent: RunView, child: RunView, metric: M.Metric, exposure: str, changed: frozenset[str], prompt_diverged: set[str], ctx: M.ExposureContext, *, implementation_attribution: str | None = None) -> dict[str, Any]:
    """The plan's exposure slice of a pairing: the metric paired on the exposed questions, and how many there are."""
    exposed = _exposed_qids(parent, child, exposure, changed, prompt_diverged, implementation_attribution=implementation_attribution)
    shared = sorted(exposed & set(parent.rows) & set(child.rows))
    return {**_paired_stats(metric, [parent.rows[qid] for qid in shared], [child.rows[qid] for qid in shared], ctx), "questions": len(exposed)}


def _info_metrics(parent_rows: list[Mapping[str, Any]], child_rows: list[Mapping[str, Any]], ctx: M.ExposureContext) -> dict[str, dict[str, Any]]:
    """Every :data:`INFO_METRICS` metric paired on the shared questions: reported beside EX, never a verdict."""
    return {name: _paired_stats(M.resolve_metric(name), parent_rows, child_rows, ctx) for name in INFO_METRICS}


@dataclass(frozen=True)
class PlanView:
    """The frozen plan of a proposal as :func:`outcome` and :func:`verify_outcome` read it.

    Fields of the retired holdout/alpha design that an older proposal still carries (``alpha``,
    ``eval_on``, ``direction``, ...) are ignored: the promotion rule is policy (:func:`promotion_rule`),
    not plan.

    Attributes:
        metric (str): Registry name of the metric the promotion rule reads (``ex``).
        exposure (str): One of :data:`EXPOSURES`: the slice reported beside the full comparison.
        namespace (str): ``control.namespace`` (the sampling universe).
        max_age_days (float): ``control.max_age_days`` (the drift window).
        base_parent (str | None): The base parent release id.
        plan_digest (str | None): ``plan_digest`` frozen into the proposal.
    """

    metric: str
    exposure: str
    namespace: str
    max_age_days: float
    base_parent: str | None
    plan_digest: str | None


def plan_view(body: Mapping[str, Any]) -> PlanView:
    """Read the plan out of a pinned proposal body.

    Args:
        body (Mapping[str, Any]): The proposal document.

    Returns:
        PlanView: The plan fields with the defaults :func:`outcome` applies.
    """
    plan = dict(body.get("plan") or {})
    control = dict(plan.get("control") or {})
    base_parent = next((parent["release"] for parent in body.get("parents") or [] if isinstance(parent, Mapping) and parent.get("role", "base") == "base"), None)
    digest = body.get("plan_digest")
    return PlanView(
        str(plan.get("metric") or "ex"),
        str(plan.get("exposure") or "diverged:prompt"),
        str(control.get("namespace") or ""),
        float(control.get("max_age_days") or 14),
        str(base_parent) if isinstance(base_parent, str) else None,
        str(digest) if isinstance(digest, str) else None,
    )


def promotion_rule() -> dict[str, Any]:
    """The promotion rule in force, as every outcome records it under ``rule``.

    Read from the policy constants at call time, so a reviewed change of :data:`PROMOTE_MIN_GAIN_PP`,
    :data:`PROMOTE_MAX_P`, :data:`PROMOTE_QUESTION_SETS` or :data:`MAX_ATTEMPTS_PER_GENERATION` applies to the
    next outcome while every recorded outcome keeps verifying under the rule it was computed with.

    Args:
        None.

    Returns:
        dict[str, Any]: Gain, significance and attempt limits, full question sets, and ``adopt_no_harm``.
            The latter permits explicit adoption below the promotion margin on the complete recorded evolution set.
            ``max_p`` is ``None`` under the rule in force (no significance test).
    """
    return {
        "min_gain_pp": float(PROMOTE_MIN_GAIN_PP),
        "max_p": None if PROMOTE_MAX_P is None else float(PROMOTE_MAX_P),
        "max_attempts": int(MAX_ATTEMPTS_PER_GENERATION),
        "question_sets": {str(name): int(count) for name, count in sorted(PROMOTE_QUESTION_SETS.items())},
        "adopt_no_harm": True,
    }


def _recorded_rule(document: Mapping[str, Any]) -> dict[str, Any]:
    """The rule an outcome records, over the policy in force for any field it lacks."""
    recorded = document.get("rule")
    # Historical outcomes retain their original verdict, including net-zero rejections.
    return {**promotion_rule(), "adopt_no_harm": False, **(recorded if isinstance(recorded, Mapping) else {})}


def _short_set(question_set: str) -> str:
    return question_set if len(question_set) <= 23 else question_set[:23] + "…"


def verdict_gate(document: Mapping[str, Any]) -> str:
    """Why an outcome is ``descriptive`` (its run is not a legal pair), or ``""`` when the promotion rule applies.

    Checked in order, each from the recorded document so :func:`verify_outcome` reaches the same answer:
    the proposal's release must not already be the BASELINE the outcome was computed under (``already
    BASELINE``: re-measuring a promoted proposal promotes nothing and spends none of the new generation's
    attempts), the run must be registered to the proposal (``provenance``: its manifest names the proposal
    and its frozen plan digest), paired with a run of the base release in the same universe within the
    drift window (``pairing``), and cover the complete question set (``coverage``): each run scored exactly
    the questions its manifest records for the queries file it was launched with, and that queries file is a
    full question set of the recorded rule (``rule.question_sets``). A run over part of the full set
    (``--db``, ``--ids``, ``--limit``), over another queries file, or whose manifest records no question list
    is a ``partial run``: a descriptive pilot.

    Args:
        document (Mapping[str, Any]): An outcome document (``stats`` and ``verdict`` may still be absent).

    Returns:
        str: The reason, starting with ``already BASELINE``, ``unregistered run``, ``unpaired`` or ``partial run``;
            empty when the pairing is legal.
    """
    if document.get("release") and document.get("release") == document.get("generation"):
        return f"already BASELINE: the proposal's release {document.get('release')} is the BASELINE this outcome was computed under; there is nothing to promote"
    provenance = document.get("provenance") or {}
    if provenance.get("status") != "registered":
        return f"unregistered run: {provenance.get('reason') or 'the manifest does not name this proposal'}"
    pairing = document.get("pairing") or {}
    if pairing.get("status") != "paired":
        return f"unpaired: {pairing.get('reason') or 'no parent run'}"
    coverage = document.get("coverage") or {}
    if coverage.get("status") != "full":
        return "partial run: " + _partial_text(coverage) + "; a verdict needs every question of the full question set in both runs"
    sets = _recorded_rule(document).get("question_sets")
    sets = sets if isinstance(sets, Mapping) else {}
    question_set = str(coverage.get("question_set") or "")
    recorded = (coverage.get("questions") or {}).get("child")
    if question_set not in sets:
        full = ", ".join(f"{_short_set(str(name))} ({count} questions)" for name, count in sorted(sets.items())) or "none"
        return f"partial run: question set {_short_set(question_set) or '(unrecorded)'} ({recorded} questions) is not a full question set of the promotion rule ({full}); a verdict needs every question of one"
    if recorded != sets[question_set]:
        return f"partial run: the manifests record {recorded} questions for question set {_short_set(question_set)}, the promotion rule's full set holds {sets[question_set]}"
    return ""


def _partial_text(coverage: Mapping[str, Any]) -> str:
    """What the runs of a ``partial`` coverage missed, arm by arm."""
    questions = coverage.get("questions") if isinstance(coverage.get("questions"), Mapping) else {}
    complete = coverage.get("complete") if isinstance(coverage.get("complete"), Mapping) else {}
    parts = []
    for arm in ("child", "parent"):
        if questions.get(arm) is None:
            parts.append(f"the {arm} run's manifest records no question list (a run made before eval_bird recorded one)")
        elif not complete.get(arm):
            parts.append(f"the {arm} scored {coverage.get(arm)} questions, not the {questions[arm]} of its question set")
    if not parts:
        parts.append(f"the runs scored different questions ({coverage.get('missing')} missing, {coverage.get('extra')} extra)")
    return "; ".join(parts)


def _stale_base_reason(base_parent: Any, release: Any, baseline: Any, pid: Any) -> str:
    """The C8 rule's explanation: an evaluation against another base measured a different move than promoting now."""
    return (
        f"the proposal measured {base_parent} -> {release}, but BASELINE is {baseline}; promoting it would revert what "
        f"{baseline} changed since {base_parent}. Re-propose with parents=[{{\"release\": \"baseline\", \"role\": \"base\"}}, "
        f"{{\"release\": \"{pid}\", \"role\": \"merge\"}}] so the merge carries BASELINE's rows"
    )


#: Float slack of the gain comparison, so a gain equal to the threshold (1 of 200 questions is 0.5pp) promotes.
_GAIN_EPSILON = 1e-9


def _apart(value: float, bound: float | None, spec: str, digits: int) -> str:
    """``value`` formatted by ``spec`` (a template such as ``"+.{}f"``) with ``digits`` digits, or with as many more (up
    to 12) as it takes to print on its own side of ``bound``; a value equal to ``bound``, or any value when there is no
    bound, prints as it is."""
    if bound is None:
        return format(value, spec.format(digits))
    for count in range(digits, 13):
        text = format(value, spec.format(count))
        shown = float(text)
        if value == bound or (shown != bound and (shown < bound) == (value < bound)):
            return text
    return format(value, spec.format(12))


def _number(value: Any) -> bool:
    """Whether a recorded value is a JSON number (``True`` and ``False`` are not)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _rule_value(rule: Mapping[str, Any] | None, key: str) -> float | None:
    """A threshold of a recorded rule, or of the rule in force when the record lacks it or holds no number; ``None``
    when the rule sets none (``max_p`` of a rule without a significance test)."""
    if isinstance(rule, Mapping) and key in rule and (rule[key] is None or _number(rule[key])):
        value = rule[key]
    else:
        value = promotion_rule()[key]
    return None if value is None else float(value)


def format_p(p: float, rule: Mapping[str, Any] | None = None) -> str:
    """A one-sided McNemar p-value as outcome reasons and lines print it.

    Three significant digits, or more when three would print it equal to, or on the other side of, the rule's
    ``max_p``: no line reads ``p 0.05`` beside ``not significant`` (p = 0.050017 prints ``0.05002`` under a rule of
    ``max_p`` 0.05). A rule without a significance test (the rule in force) prints three significant digits.

    Args:
        p (float): The p-value.
        rule (Mapping[str, Any] | None): The rule the outcome recorded; the rule in force when ``None``.

    Returns:
        str: The digits, e.g. ``0.00391``, ``0.242``, ``0.05002``.
    """
    return _apart(float(p), _rule_value(rule, "max_p"), ".{}g", 3)


def format_gain(gain: float, rule: Mapping[str, Any] | None = None) -> str:
    """An EX gain in percentage points as outcome reasons and lines print it.

    Signed with two decimals, or more when two would print it equal to, or on the other side of, the rule's
    ``min_gain_pp``: 8 net wins of 1605 questions (+0.4984pp) print ``+0.498``, not ``+0.50``.

    Args:
        gain (float): The gain, child minus parent, in percentage points.
        rule (Mapping[str, Any] | None): The rule the outcome recorded; the rule in force when ``None``.

    Returns:
        str: The signed digits, e.g. ``+20.00``, ``+0.498``, ``-1.30``.
    """
    return _apart(float(gain), _rule_value(rule, "min_gain_pp"), "+.{}f", 2)


def promotion_verdict(document: Mapping[str, Any]) -> tuple[str, str]:
    """Apply the promotion rule to an outcome document: the verdict and its reason.

    A run that is not a legal pair is ``descriptive`` (:func:`verdict_gate`). A legally paired outcome is
    ``not-promotable`` when, checked in this order, the generation already took ``rule.max_attempts``
    evaluations and ``attempt.allow_over_cap`` is false (``over cap``: :func:`outcome` refuses such a pair before
    computing it, so only a document written past the cap without the override shows it), replay integrity
    is not ``ok`` (``integrity invalid`` / ``integrity unverifiable``), the proposal's base parent is not the
    generation's BASELINE (``stale base``), the EX gain is below ``rule.min_gain_pp`` (``gain below
    threshold``), or — only under a rule that records a ``max_p`` (the first rule's 0.05) — the exact one-sided
    McNemar p-value exceeds it (``not significant``); otherwise it is ``promotable``. A rule recording
    ``adopt_no_harm: True`` instead returns ``adoptable`` for a nonnegative net below the promotion margin,
    only on the complete paired recorded evolution set and after all integrity, budget and significance checks pass.
    Older rules without that field retain their original verdicts. The rule in force records
    ``max_p: None``: its p-value is information and decides nothing. The gain and the p-value are recomputed
    from the recorded discordant counts, and the thresholds are the ones the outcome recorded, so
    :func:`verify_outcome` reaches the same answer after the policy changes.

    Args:
        document (Mapping[str, Any]): An outcome document with ``provenance``, ``pairing``, ``coverage``,
            ``integrity_detail``, ``proposal``, ``release``, ``base_parent``, ``generation``, ``stats``, ``rule``
            and ``attempt``.

    Returns:
        tuple[str, str]: ``("promotable", "")``, ``("adoptable", reason)``,
            ``("not-promotable", reason)`` or ``("descriptive", reason)``.
    """
    reason = verdict_gate(document)
    if reason:
        return "descriptive", reason
    rule = _recorded_rule(document)
    attempt = document.get("attempt") if isinstance(document.get("attempt"), Mapping) else {}
    used, cap = int(attempt.get("used") or 0), int(rule["max_attempts"])
    if used >= cap and not attempt.get("allow_over_cap"):
        return "not-promotable", f"over cap: generation {document.get('generation')} already took {used} evaluation(s), the cap is {cap}, and the outcome was not computed with allow_over_cap"
    state, why = integrity_state(document.get("integrity_detail") or {})
    if state != "ok":
        return "not-promotable", f"integrity {state}: {why}"
    if document.get("base_parent") != document.get("generation"):
        return "not-promotable", "stale base: " + _stale_base_reason(document.get("base_parent"), document.get("release"), document.get("generation"), document.get("proposal"))
    stats = document.get("stats") if isinstance(document.get("stats"), Mapping) else {}
    b, c, n = int(stats.get("b") or 0), int(stats.get("c") or 0), int(stats.get("n") or 0)
    gain, min_gain = M.effect_pp(b, c, n), float(rule["min_gain_pp"])
    below_margin = ""
    if c - b < 2:
        below_margin = f"net gain below threshold: {c - b} questions < 2"
    elif gain < min_gain - _GAIN_EPSILON:
        below_margin = f"gain below threshold: EX {format_gain(gain, rule)}pp < {min_gain:g}pp (c={c}, b={b}; n={n})"
    question_set = document["coverage"].get("question_set")
    adoptable = (rule["adopt_no_harm"] is True and c - b >= 0
                 and (question_set, n) in {(QUESTION_SETS["330"].digest, 330),
                                           (QUESTION_SETS["minidev500"].digest, 500)}
                 and question_set in rule["question_sets"] and n == rule["question_sets"][question_set])
    if below_margin and not adoptable:
        return "not-promotable", below_margin
    if rule["max_p"] is not None:
        p, max_p = M.mcnemar_exact(b, c).p, float(rule["max_p"])
        if p > max_p:
            return "not-promotable", f"not significant: one-sided McNemar p {format_p(p, rule)} > {max_p:g} (c={c}, b={b})"
    if below_margin:
        scope = "fixed 330" if question_set == QUESTION_SETS["330"].digest else "Mini-Dev 500"
        return "adoptable", f"no harm: net {c - b} questions over the paired {scope}; {below_margin}; explicit adoption rationale required"
    return "promotable", ""


_OutcomeStatistics = tuple[list[dict[str, Any]], dict[str, Any], str, str, dict[str, Any]]


@dataclass
class OutcomeComputation:
    """Retain pure scoring work for one command's conflict retries only.

    Each attempt still reloads and validates both runs, resolves the releases,
    pairs the control and reserves accounting before calling ``statistics``.
    The key includes the actual decoded rows as well as file hashes, so a file
    changed between the separate row/hash reads cannot authorize stale scoring.
    No authority events, attempts, verdicts or decisions are cached here.
    """

    _key: str | None = field(default=None, init=False)
    _result: _OutcomeStatistics | None = field(default=None, init=False)

    def statistics(self, parent: RunView | None, child: RunView, metric: M.Metric, plan: PlanView,
                   changed: frozenset[str], knobs: frozenset[str], *,
                   implementation_attribution: str | None,
                   implementation_contexts: Mapping[str, Any] | None) -> _OutcomeStatistics:
        """Reuse statistics only after exact scoring-input equality; refuse changes."""
        def run_inputs(run: RunView | None) -> Any:
            return None if run is None else {"path": str(run.path.resolve()), "manifest": run.manifest,
                "manifest_sha": run.manifest_sha, "files": run.result_files, "rows": run.rows,
                "created_at": run.created_at}

        key = sha256_hex(canon({"parent": run_inputs(parent), "child": run_inputs(child),
            "plan": asdict(plan), "metric": metric.name, "changed": sorted(changed), "knobs": sorted(knobs),
            "attribution": implementation_attribution, "contexts": implementation_contexts, "strict_sources": True}))
        if self._key is not None and self._key != key:
            raise EvolveError("outcome scoring inputs changed during authority publication; check the runs again")
        if self._key is None:
            self._result = _outcome_statistics(parent, child, metric, plan, changed, knobs, strict_sources=True,
                implementation_attribution=implementation_attribution, implementation_contexts=implementation_contexts)
            self._key = key
        # The outcome document is mutable; never let staging change retained work.
        from copy import deepcopy

        assert self._result is not None
        return deepcopy(self._result)


def authority_outcome(
    authority: Any,
    reader: Any,
    pid: str,
    run_dir: Path,
    store: Store,
    *,
    parent_run_dir: Path | None = None,
    search: Iterable[Path] = (),
    decision_rationale: str = "",
    now: float | None = None,
    write: bool = True,
    actor: str = "authority:outcome",
    allow_over_cap: bool = False,
    env: Mapping[str, str] | None = None,
    computation: OutcomeComputation | None = None,
    registration: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Read and pair runs, reserve native quota, then compute and record EX.

    Args:
        authority: Writer holding the instance-bound authority lock.
        reader: Consistent pre-batch authority view.
        pid: Complete published proposal identity, or the exact adopted current release for an unpaired baseline.
        run_dir: Child result directory with immutable run source proofs.
        store: Detached authority projection used for exact release comparison.
        parent_run_dir: Explicit parent, or search for a matching control.
        search: Additional parent-run roots.
        decision_rationale: Text saved verbatim in the outcome.
        now: Record time, current time by default.
        write: Save outcome; false still reserves a verdict-taking attempt.
        actor: Event actor.
        allow_over_cap: Human-authorized spend override.
        env: Harness environment for the override guard.
        computation: Command-owned pure statistics retained across CAS conflicts.

    Returns:
        Complete reproducible outcome document. The writer owns promotion.

    Raises:
        EvolveError: Invalid run identity/source, changed pairing, repeated
            measurement, unauthorized override or exhausted generation quota.
    """
    store.require_seeded_v0()
    if allow_over_cap:
        ensure_human("outcome --allow-over-cap", env)
    adopted = pid == reader.head().release
    if adopted:
        if parent_run_dir is not None or tuple(search):
            raise EvolveError("an adopted current-release baseline is descriptive and accepts no paired parent")
        rid = pid
        plan = PlanView("ex", "diverged:prompt", f"v0-{reader.epoch()['release']}", 14.0, None, None)
    else:
        body = reader.proposal(pid)
        rid = str(body["release"])
        plan = plan_view(body)
    if plan.metric not in PLAN_METRICS:
        raise EvolveError(f"proposal {pid}: its plan names metric {plan.metric!r}; the promotion rule reads {list(PLAN_METRICS)}")
    metric = M.resolve_metric(plan.metric)

    child = read_run(Path(run_dir))
    if adopted and not child.diagnostic:
        require_evolution_run(child)
    if adopted and child.manifest.get("proposal") is not None:
        raise EvolveError("an adopted current-release baseline cannot claim a proposal pin")
    if child.release not in (rid, pid):
        raise EvolveError(f"run {child.name} declares release {child.release!r}, not the proposal's {rid} (or the pin {pid})")
    if problem := child.release_problem(store=store):
        raise EvolveError(f"run {child.name}: {problem}")
    if child.ns != plan.namespace:
        raise EvolveError(f"run {child.name}: cache namespace {child.ns!r} differs from the plan's {plan.namespace!r}")
    provenance = _provenance(child, pid, plan.plan_digest, registration)

    from .runtime import load_release

    # The base parent, loaded once: pairing accepts a control run under any release that sends the same
    # prompts as the base parent (a refactor-only sibling) and carries the same knowledge rows (a rules or
    # generator proposal moves no keyspace, and its runs read other knowledge), and the exposure context
    # compares against it.
    parent_release = None
    if plan.base_parent:
        try:
            parent_release = load_release(plan.base_parent, stores=store, allow_retired=True)
        except PromptStoreError:
            parent_release = None

    siblings: dict[str, bool] = {}

    def same_parent(release_id: Any) -> bool:
        if release_id == plan.base_parent:
            return True
        if parent_release is None or not isinstance(release_id, str) or not same_run_release(parent_release, release_id, store=store):
            return False
        if release_id not in siblings:
            try:
                siblings[release_id] = not knowledge_differences(parent_release, load_release(release_id, stores=store, allow_retired=True))
            except PromptStoreError:
                siblings[release_id] = False
        return siblings[release_id]

    parent: RunView | None = None
    pairing: dict[str, Any] = {"status": "unpaired", "reason": "no parent run given or found"}
    if provenance["status"] != "registered":
        pairing = {"status": "unpaired", "reason": f"not paired: {provenance['reason']}"}
    elif parent_run_dir is not None:
        parent = read_run(Path(parent_run_dir))
        problems = []
        if problem := parent.release_problem(store=store):
            problems.append(f"parent run {parent.name}: {problem}")
        if not same_parent(parent.release):
            problems.append(f"parent run declares release {parent.release!r}, the base parent is {plan.base_parent}")
        if parent.ns != plan.namespace:
            problems.append(f"parent namespace {parent.ns!r} differs from the plan's {plan.namespace!r}")
        if parent.identity != child.identity:
            problems.append("manifest identities differ outside the prompt fields")
        age_days = (child.created_at - parent.created_at) / _DAY
        if age_days > plan.max_age_days:
            problems.append(f"parent run is {age_days:.1f} days older than the child; the drift window is {plan.max_age_days:g} days")
        elif -age_days > plan.max_age_days:
            problems.append(f"parent run is {-age_days:.1f} days newer than the child; the drift window is {plan.max_age_days:g} days either way")
        if problems:
            pairing = {"status": "unpaired", "reason": "; ".join(problems), "parent": parent.name}
            parent = None
        else:
            pairing = {"status": "paired", "parent": parent.name, "age_days": round(age_days, 3)}
    elif plan.base_parent:
        parent = find_parent_run(child, plan.base_parent, namespace=plan.namespace, max_age_days=plan.max_age_days, search=search, same=same_parent, store=store)
        if parent is not None:
            pairing = {"status": "paired", "parent": parent.name, "age_days": round((child.created_at - parent.created_at) / _DAY, 3)}

    try:
        child_release = load_release(rid, stores=store, allow_retired=True)
        changed = changed_surfaces(child_release, parent_release) if parent_release is not None else frozenset()
        knobs = changed_knobs(child_release, parent_release) if parent_release is not None else frozenset()
    except PromptStoreError:
        changed = knobs = frozenset()
    coverage = _coverage(parent, child) if parent is not None else None
    child_implementation = child.manifest.get("implementation")
    attribution = _IMPLEMENTATION_ATTRIBUTION if isinstance(child_implementation, Mapping) and "role_digests" in child_implementation else None
    contexts = {"child": _implementation_contexts(child), "parent": _implementation_contexts(parent)} if attribution else None
    parent_sha = parent.manifest_sha if parent else None
    ts = float(now if now is not None else time.time())
    oid = outcome_id(pid, child.manifest_sha, child.result_files, parent_sha, parent.result_files if parent else None)
    rule = promotion_rule()
    launch = child.manifest.get("paid_launch") or {}
    if not isinstance(launch, Mapping):
        raise EvolveError("outcome paid launch evidence is malformed")
    reservation_id = registration.get("reservation_id") if registration is not None else launch.get("reservation_id")
    reserved = [event["payload"] for event in reader.events("eval-reserve")
                if event["payload"].get("id") == reservation_id]
    if len(reserved) == 1 and "budget_cap" in reserved[0]:
        cap = reserved[0]["budget_cap"]
        if type(cap) is not int or not 0 <= cap <= 20:
            raise EvolveError("outcome reservation has an invalid bounded generation cap")
        rule["max_attempts"] = cap
    with contextlib.nullcontext():
        generation = reader.head().generation
        # a pair that takes a verdict is decided, and counted, before any statistic exists
        takes_verdict = not verdict_gate({"release": rid, "generation": generation, "provenance": provenance, "pairing": pairing, "coverage": coverage, "rule": rule})
        existing = [event["payload"] for event in reader.events("outcome") if event["payload"].get("id") == oid]
        if existing:
            return existing[0]
        supersedes = _earlier_outcomes(store, pid, child, parent_sha, takes_verdict=takes_verdict)
        marker = None
        if takes_verdict and parent is not None:
            attempt, marker = _authority_count_attempt(authority, reader, generation, oid, pid=pid, rid=rid, child=child, parent=parent, allow_over_cap=allow_over_cap, dry_run=not write, cap=int(rule["max_attempts"]), now=ts, actor=actor, registration=registration)
        else:
            attempt = _attempt(store, generation, allow_over_cap, exclude=oid)
        table, integrity_detail, integrity, integrity_reason, stats = (computation or OutcomeComputation()).statistics(
            parent, child, metric, plan, changed, knobs,
            implementation_attribution=attribution, implementation_contexts=contexts)
        document: dict[str, Any] = {
            "schema": SCHEMA_OUTCOME,
            "source_verification": "immutable/1",
            "id": oid,
            "proposal": None if adopted else pid,
            **({"target": rid, **({"authority_state": reader.current_state()["id"]}
                 if reader.source_admission(rid) is not None else {"baseline_decision": reader.head().decision}),
                **({"evolution_regime": {"question_set": EVOLUTION_QUESTION_SET,
                                        "previous_question_set": QUESTION_SETS["330"].digest,
                                        "questions": child.questions}}
                   if not child.diagnostic and not any(event["payload"].get("evolution_regime")
                                                       for event in reader.events("outcome")) else {})}
               if adopted else {}),
            "release": rid,
            "base_parent": plan.base_parent,
            "generation": generation,
            "run": child.name,
            "run_dir": str(child.path),
            "parent_run": parent.name if parent else "",
            "parent_run_dir": str(parent.path) if parent else None,
            "manifests": {"child": {"path": RUN_MANIFEST_NAME, "sha256": child.manifest_sha}, "parent": {"path": RUN_MANIFEST_NAME, "sha256": parent.manifest_sha} if parent else None},
            "implementations": {"child": child.manifest.get("implementation"), "parent": parent.manifest.get("implementation") if parent else None},
            "result_files": {"child": child.result_files, "parent": parent.result_files if parent else None},
            "identity": {"child": sha256_hex(canon(child.identity))[:16], "parent": sha256_hex(canon(parent.identity))[:16] if parent else None},
            "plan_digest": plan.plan_digest,
            "metric": plan.metric,
            "provenance": provenance,
            "pairing": pairing,
            "coverage": coverage,
            "integrity": integrity,
            "integrity_reason": integrity_reason,
            "integrity_detail": integrity_detail,
            "table": table,
            "changed_surfaces": sorted(changed),
            "stats": stats,
            "rule": rule,
            "attempt": attempt,
            "supersedes": supersedes,
            "decision_rationale": decision_rationale,
            "created_at": ts,
        }
        if knobs:  # absent otherwise, so an outcome of a proposal that moves no knob is written as before
            document["changed_knobs"] = sorted(knobs)
        if attribution:
            document.update(implementation_attribution=attribution, implementation_contexts=contexts,
                            changed_implementations=_implementation_changes(parent, child, contexts))
        verdict, reason = promotion_verdict(document)
        document.update({"verdict": verdict, "verdict_reason": reason})
        if marker is not None:
            document["attempt_event"] = marker
        if registration is not None:
            document["registration"] = dict(registration)
        if write:
            authority._event("outcome", pid, document, actor)
    return document


def _earlier_outcomes(store: Store, pid: str, child: RunView, parent_sha: str | None, *, takes_verdict: bool) -> list[str]:
    """Check a new outcome against the earlier ones on the same child run; return the unpaired ones it supersedes.

    Raises:
        EvolveError: When the same run pair already has an outcome, or when the new outcome would take a verdict and
            the child run already has one: a run takes one verdict, whichever control run it is paired with, so
            re-pairing a run with other controls cannot shop for a better verdict.
    """
    supersedes: list[str] = []
    for doc in outcomes_of(store, pid):
        manifests = doc.get("manifests") or {}
        if (manifests.get("child") or {}).get("sha256") != child.manifest_sha or (doc.get("result_files") or {}).get("child") != child.result_files:
            continue
        earlier_parent = (manifests.get("parent") or {}).get("sha256")
        if earlier_parent == parent_sha:
            raise EvolveError(f"run {child.name} already has outcome {doc.get('id')} on proposal {pid}")
        if takes_verdict and doc.get("verdict") in VERDICTS:
            raise EvolveError(f"run {child.name} already has a verdict on proposal {pid} (outcome {doc.get('id')}, {doc.get('verdict')}, paired with {doc.get('parent_run') or '-'}); a run takes one verdict, whichever control run it is paired with")
        if parent_sha is not None and earlier_parent is None:
            supersedes.append(str(doc.get("id")))
    return supersedes


def _outcome_statistics(parent: RunView | None, child: RunView, metric: M.Metric, plan: PlanView, changed: frozenset[str], knobs: frozenset[str] = frozenset(), *, strict_sources: bool = False, implementation_attribution: str | None = None, implementation_contexts: Mapping[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any], str, str, dict[str, Any]]:
    """The pairing table, integrity detail and state, and statistics of an outcome; a child alone gets its mean."""
    ctx = M.ExposureContext(changed, changed_knobs=knobs)
    stats: dict[str, Any] = {"metric": plan.metric}
    if parent is not None:
        table, integrity_detail, prompt_diverged = _pair_rows(parent, child, metric, ctx, strict_sources=strict_sources,
                                                            implementation_attribution=implementation_attribution, implementation_contexts=implementation_contexts)
        shared = sorted(set(parent.rows) & set(child.rows))
        parent_rows = [parent.rows[qid] for qid in shared]
        child_rows = [child.rows[qid] for qid in shared]
        integrity, integrity_reason = integrity_state(integrity_detail)
        b, c, n, parent_sum, child_sum = _q_pairs(table)
        stats.update({
            "n": n,
            "b": b,
            "c": c,
            "parent_mean": round(parent_sum / n, 6) if n else None,
            "child_mean": round(child_sum / n, 6) if n else None,
            "gain_pp": round(M.effect_pp(b, c, n), 4),
            "p": M.mcnemar_exact(b, c).p,
            "exposed": _exposed_stats(parent, child, metric, plan.exposure, changed, prompt_diverged, ctx, implementation_attribution=implementation_attribution),
            "per_db": _per_db(table),
            "metrics": _info_metrics(parent_rows, child_rows, ctx),
            "slots": {"parent": M.slot_stats(parent_rows), "child": M.slot_stats(child_rows)},
        })
    else:
        table, integrity_detail = [], {"paired_units": 0, "undiverged": 0, "violations": 0, "rate": 0.0, "noise": 0, "noise_rate": 0.0, "unverified": 0, "unverified_rate": 0.0, "divergences": {}}
        integrity, integrity_reason = "unpaired", ""
        child_only = M.compute(metric, list(child.rows.values()), ctx)
        stats.update({"n": child_only.n_valued, "mean": child_only.mean, "b": None, "c": None, "parent_mean": None, "child_mean": None, "gain_pp": None, "p": None, "exposed": None, "per_db": {}, "metrics": {}, "slots": {"child": M.slot_stats(child.rows.values())}})
    return table, integrity_detail, integrity, integrity_reason, stats


def _decisions_in_order(store: Store) -> list[Mapping[str, Any]]:
    """Decision documents by ``created_at``, then id: the order the lineage reads them in."""
    return sorted(store.decisions(), key=lambda decision: (float(decision.get("created_at") or 0.0), str(decision.get("id", ""))))


def promotion_veto(store: Store, rid: str) -> dict[str, Any] | None:
    """The person's decision that bars a release from promotion by the rule, or ``None``.

    A ``reject`` of any proposal whose release is ``rid``, or a ``rollback`` that demoted ``rid`` (its
    ``previous``), bars ``rid`` until a later ``promote --override-verdict`` of it. Both verbs are human-only,
    so neither :func:`outcome`'s automatic promotion nor an agent running :func:`promote` can overturn them;
    the bar is on the release, so re-proposing the same rows under a new proposal id does not lift it.

    Args:
        store (Store): The store.
        rid (str): Release id.

    Returns:
        dict[str, Any] | None: The barring ``reject`` or ``rollback`` decision, or ``None``.
    """
    veto: Mapping[str, Any] | None = None
    for decision in _decisions_in_order(store):
        kind = decision.get("kind")
        if kind == "reject" and _proposal_release(store, str(decision.get("target"))) == rid:
            veto = decision
        elif kind == "rollback" and decision.get("previous") == rid:
            veto = decision
        elif kind == "promote" and decision.get("release") == rid and decision.get("override_verdict"):
            veto = None
    return dict(veto) if veto is not None else None


def _withdrawal(store: Store, pid: str) -> dict[str, Any] | None:
    """The ``abandon`` decision that withdrew a proposal, when no later decision set the proposal's status again."""
    latest: Mapping[str, Any] | None = None
    for decision in _decisions_in_order(store):
        demoted = decision.get("demoted") if decision.get("kind") == "rollback" else None
        if (decision.get("target") == pid and decision.get("kind") in ("promote", "reject", "abandon")) or (isinstance(demoted, list) and pid in demoted):
            latest = decision
    return dict(latest) if latest is not None and latest.get("kind") == "abandon" else None


def _revoked_rows(store: Store, rid: str) -> list[str]:
    """``name:version`` of every row of a release's closure that a person revoked (such a release never loads)."""
    try:
        refs = store.release(rid).data.get("refs") or {}
    except PromptStoreError:
        return []
    revoked = store.revoked()
    return sorted(f"{name}:{str(key)[:16]}" for name, key in refs.items() if key in revoked)


def _bar_text(rid: str, pid: str, veto: Mapping[str, Any] | None, withdrawal: Mapping[str, Any] | None) -> str:
    """Why a person's decision keeps a proposal from promotion by the rule, or ``""``."""
    if veto is not None:
        done = "rejected" if veto.get("kind") == "reject" else "rolled back"
        return f"release {rid} was {done} by {veto.get('actor') or '?'} in decision {veto.get('id')} ({veto.get('reason')}); only a person promotes it again, with promote --override-verdict"
    if withdrawal is not None:
        return f"proposal {pid} was abandoned by {withdrawal.get('actor') or '?'} in decision {withdrawal.get('id')}; re-propose it to have it promoted"
    return ""


def outcome_id(pid: str, child_manifest_sha: str, child_files: Mapping[str, str], parent_manifest_sha: str | None, parent_files: Mapping[str, str] | None) -> str:
    """The outcome id: a content key over the run pair, so the same evaluation attaches once.

    Args:
        pid (str): Proposal id.
        child_manifest_sha (str): sha256 of the child run's manifest bytes.
        child_files (Mapping[str, str]): Child result file to sha256.
        parent_manifest_sha (str | None): sha256 of the parent run's manifest bytes, ``None`` when unpaired.
        parent_files (Mapping[str, str] | None): Parent result file to sha256, ``None`` when unpaired.

    Returns:
        str: ``"o"`` + 16 hex.
    """
    return "o" + sha256_hex(canon({"proposal": pid, "child": child_manifest_sha, "child_files": dict(child_files), "parent": parent_manifest_sha, "parent_files": dict(parent_files) if parent_files is not None else None}))[:16]


def _rule_problems(document: Mapping[str, Any]) -> list[str]:
    """Shape of the recorded ``rule`` and ``attempt``: the inputs :func:`promotion_verdict` trusts."""
    problems = []
    rule = document.get("rule")
    numeric = ("min_gain_pp", "max_attempts")
    required = {*numeric, "max_p", "question_sets"}
    if (not isinstance(rule, Mapping) or set(rule) not in (required, required | {"adopt_no_harm"})
            or not all(_number(rule[key]) for key in numeric)
            or not (rule["max_p"] is None or _number(rule["max_p"]))
            or not isinstance(rule.get("adopt_no_harm", False), bool)):
        problems.append(f"rule {rule!r} is not {{min_gain_pp, max_p, max_attempts, question_sets}} with numeric thresholds and optional boolean adopt_no_harm (max_p null: no significance test)")
    elif not isinstance(rule["question_sets"], Mapping) or not all(isinstance(name, str) and isinstance(count, int) and not isinstance(count, bool) and count > 0 for name, count in rule["question_sets"].items()):
        problems.append(f"rule.question_sets {rule['question_sets']!r} is not a mapping of question-set sha to a positive question count")
    attempt = document.get("attempt")
    counted = attempt.get("counted") if isinstance(attempt, Mapping) else None
    if (
        not isinstance(attempt, Mapping)
        or set(attempt) != {"used", "counted", "allow_over_cap"}
        or not isinstance(attempt.get("used"), int)
        or isinstance(attempt.get("used"), bool)
        or not isinstance(counted, list)
        or not all(isinstance(oid, str) for oid in counted)
        or len(set(counted)) != len(counted)
        or attempt["used"] != len(counted)
        or document.get("id") in counted
        or not isinstance(attempt.get("allow_over_cap"), bool)
    ):
        problems.append(f"attempt {attempt!r} is not {{used: the number of counted ids, counted: [distinct ids of other outcomes], allow_over_cap: bool}}")
    return problems


def _role_attribution_problems(document: Mapping[str, Any]) -> list[str]:
    """Check the recorded context bindings and recount integrity under ``roles/1``."""
    problems = []
    contexts = document.get("implementation_contexts")
    if not isinstance(contexts, Mapping) or set(contexts) != {"parent", "child"}:
        return ["implementation contexts do not record both run arms"]
    manifests = document.get("manifests") if isinstance(document.get("manifests"), Mapping) else {}
    implementations = document.get("implementations") if isinstance(document.get("implementations"), Mapping) else {}
    for arm in ("parent", "child"):
        record = contexts.get(arm)
        manifest = manifests.get(arm)
        if manifest is None and record is None:
            continue
        if (not isinstance(record, Mapping) or set(record) != {"manifest_sha256", "source", "role_digests"}
                or not isinstance(manifest, Mapping) or record.get("manifest_sha256") != manifest.get("sha256")):
            problems.append(f"{arm} implementation contexts do not bind the recorded manifest")
            continue
        source = record.get("source")
        block = implementations.get(arm)
        if source == "manifest":
            if not isinstance(block, Mapping) or block.get("role_digests") != record.get("role_digests"):
                problems.append(f"{arm} implementation contexts differ from the copied implementation block")
        elif (source != "unknown" or record.get("role_digests") is not None
              or isinstance(block, Mapping) and "role_digests" in block):
            problems.append(f"{arm} implementation context source is invalid")
        if source != "unknown":
            declared = record.get("role_digests")
            records = _implementation_records(declared)
            expected_count = len(declared["candidate"]) + 2 if isinstance(declared, Mapping) and isinstance(declared.get("candidate"), list) else 0
            if not records or len(records) != expected_count or len({canon(item["context"]) for item in records}) != len(records):
                problems.append(f"{arm} implementation contexts contain invalid or ambiguous role digests")
    if any(record is not None and not isinstance(record, Mapping) for record in contexts.values()):
        return problems
    changes = document.get("changed_implementations")
    if not isinstance(changes, list):
        problems.append("changed implementations is not a list")
        changes = []
    for change in changes:
        if not isinstance(change, Mapping) or set(change) != {"parent", "child", "parent_manifest_sha256", "child_manifest_sha256"}:
            problems.append("changed implementation record is malformed")
            continue
        parent, child = change.get("parent"), change.get("child")
        if (parent not in _implementation_records((contexts.get("parent") or {}).get("role_digests"))
                or child not in _implementation_records((contexts.get("child") or {}).get("role_digests"))):
            problems.append("changed implementation lacks a declared context binding")
            continue
        if (parent["context"]["role"] != child["context"]["role"] or parent["context"].get("route") != child["context"].get("route")
                or parent["digest"] == child["digest"]
                or any(change.get(f"{arm}_manifest_sha256") != (contexts.get(arm) or {}).get("manifest_sha256") for arm in ("parent", "child"))):
            problems.append("changed implementation does not prove a counterpart digest difference")
    if (document.get("pairing") or {}).get("status") != "paired":
        return problems
    table = document.get("table") or []
    rows = [row for row in table if isinstance(row, Mapping) and row.get("unit") != "q"]
    if any(type(row.get("paired")) is not bool or type(row.get("child_present")) is not bool or row["paired"] and not row["child_present"] for row in rows):
        return problems + ["role attribution table lacks valid unit presence records"]
    divergences = {bucket: 0 for bucket in (*CAUSE_BUCKETS, CAUSE_IMPLEMENTATION)}
    detail: dict[str, Any] = {"paired_units": 0, "undiverged": 0, "violations": 0, "rate": 0.0,
                              "noise": 0, "noise_rate": 0.0, "unverified": 0, "unverified_rate": 0.0,
                              "divergences": divergences}
    # Old immutable outcomes retain their recorded counting policy. Only new
    # computations add retrieval evidence; verification never reads current run files.
    if "degraded_retrieval" in (document.get("integrity_detail") or {}):
        detail["degraded_retrieval"] = {"parent": 0, "child": 0, "units": 0, "total": 0, "rate": 0.0}
        if any(type(row.get(key)) is not bool for row in rows for key in ("parent_degraded", "child_degraded")):
            problems.append("retrieval integrity table lacks valid per-arm degradation records")
    by_question: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["child_present"]:
            by_question.setdefault(str(row.get("qid")), {})[str(row.get("unit"))] = row.get("cause")
    for row in rows:
        bucket = _cause_bucket(row.get("cause"), implementation_attribution=_IMPLEMENTATION_ATTRIBUTION)
        change = row.get("implementation_change")
        role = "candidate" if str(row.get("unit")).startswith("c") else {"j": "judge", "r": "reviewer"}.get(row.get("unit"))
        if change is not None and (not isinstance(change, Mapping) or change not in changes
                                   or not isinstance(change.get("child"), Mapping)
                                   or not isinstance(change["child"].get("context"), Mapping)
                                   or change["child"]["context"].get("role") != role):
            problems.append("table implementation change is not a recorded difference of that role")
        if bucket == CAUSE_IMPLEMENTATION and (change is None or not row["paired"]):
            problems.append("implementation cause lacks a paired context difference")
        _count_integrity_row(detail, row, by_question.get(str(row.get("qid")), {}),
                             paired=row["paired"], child_present=row["child_present"],
                             implementation_attribution=_IMPLEMENTATION_ATTRIBUTION)
    _integrity_rates(detail)
    if detail != document.get("integrity_detail"):
        problems.append("role attribution integrity counts do not follow from the table")
    return problems


def verify_outcome(document: Mapping[str, Any], *, plan: PlanView | None = None) -> list[str]:
    """Recompute an outcome's id, statistics, integrity state and verdict from the document alone.

    Every decision :func:`outcome` took is re-derived: the id from the run pair; the discordant pairs, the arm
    means and the EX gain from the pairing table; the one-sided McNemar p-value from those pairs; the integrity
    state from ``integrity_detail``; and the verdict with its reason under the rule the outcome recorded
    (:func:`promotion_verdict`). With the proposal's plan, the recorded metric and base parent must be the plan's.
    What the runs alone can confirm needs them: :func:`deep_verify_outcome` rebuilds the pairing, and
    :func:`information_drift` the information slices, which decide nothing.

    Args:
        document (Mapping[str, Any]): An outcome document.
        plan (PlanView | None): The proposal's frozen plan (:func:`plan_view`), when the caller has the store.

    Returns:
        list[str]: Problems found; empty when the document reproduces. A document that records no ``rule`` at all
            was written before the promotion rule (the holdout/alpha design) and gets one line saying so: its
            verdict is not a rule-A verdict, is not recomputed and carries no promotion.
    """
    if "rule" not in document:
        return [f"recorded before rule A (the outcome records no rule): its verdict {document.get('verdict')!r} is not recomputed and carries no promotion"]
    problems: list[str] = []
    manifests = document.get("manifests") if isinstance(document.get("manifests"), Mapping) else {}
    if document.get("source_verification") not in (None, "immutable/1"):
        problems.append("unknown source verification policy")
    attribution = document.get("implementation_attribution")
    if attribution not in (None, _IMPLEMENTATION_ATTRIBUTION):
        problems.append("unknown implementation attribution policy")
    if attribution == _IMPLEMENTATION_ATTRIBUTION:
        problems.extend(_role_attribution_problems(document))
    files = document.get("result_files") if isinstance(document.get("result_files"), Mapping) else {}
    parent_manifest = manifests.get("parent") if isinstance(manifests.get("parent"), Mapping) else None
    expected_id = outcome_id(str(document.get("proposal") or document.get("target")), str((manifests.get("child") or {}).get("sha256", "")), dict(files.get("child") or {}), parent_manifest.get("sha256") if parent_manifest else None, files.get("parent") if isinstance(files.get("parent"), Mapping) else None)
    if expected_id != document.get("id"):
        problems.append("id does not hash the run pair")
    if "evolution_regime" in document:
        regime = document["evolution_regime"]
        if (document.get("target") is None or document.get("proposal") is not None
                or regime.get("question_set") != QUESTION_SETS["minidev500"].digest
                or regime.get("previous_question_set") != QUESTION_SETS["330"].digest
                or (regime.get("questions") or {}).get("count") != 500
                or (document.get("stats") or {}).get("n") != 500):
            problems.append("invalid Mini-Dev 500 regime control boundary")
    if document.get("target") is not None and (
            document.get("proposal") is not None or document.get("target") != document.get("release")
            or document.get("release") != document.get("generation")
            or not isinstance(document.get("authority_state", document.get("baseline_decision")), str)
            or document.get("base_parent") is not None or parent_manifest is not None
            or document.get("parent_run_dir") is not None
            or (document.get("pairing") or {}).get("status") != "unpaired"
            or (document.get("provenance") or {}).get("status") != "unregistered"
            or document.get("verdict") != "descriptive"):
        problems.append("current-release baseline must remain an unpaired descriptive outcome")
    problems.extend(_rule_problems(document))
    stats = document.get("stats") if isinstance(document.get("stats"), Mapping) else {}
    if plan is not None:
        if stats.get("metric") != plan.metric or document.get("metric") != plan.metric:
            problems.append(f"metric {stats.get('metric')!r} is not the plan's {plan.metric!r}")
        if document.get("base_parent") != plan.base_parent:
            problems.append(f"base_parent {document.get('base_parent')!r} is not the proposal's {plan.base_parent!r}")
    if (document.get("pairing") or {}).get("status") == "paired":
        detail = document.get("integrity_detail") or {}
        if "degraded_retrieval" in detail:
            units = [row for row in document.get("table") or [] if row.get("unit") != "q"]
            parent = sum(row.get("parent_degraded") is True for row in units)
            child = sum(row.get("child_degraded") is True for row in units)
            degraded = sum(bool(row.get("parent_degraded") or row.get("child_degraded")) for row in units)
            expected = {"parent": parent, "child": child, "units": degraded, "total": len(units),
                        "rate": round(degraded / len(units), 6) if units else 0.0}
            if (any(type(row.get(key)) is not bool for row in units for key in ("parent_degraded", "child_degraded"))
                    or expected != detail["degraded_retrieval"]):
                problems.append("retrieval integrity counts do not follow from the table")
        state, _why = integrity_state(document.get("integrity_detail") or {})
        if state != document.get("integrity"):
            problems.append(f"integrity recomputed as {state!r}, recorded {document.get('integrity')!r}")
        b, c, n, parent_sum, child_sum = _q_pairs(document.get("table") or [])
        if (b, c, n) != (stats.get("b"), stats.get("c"), stats.get("n")):
            problems.append(f"pairs recomputed as n={n} b={b} c={c}, recorded n={stats.get('n')} b={stats.get('b')} c={stats.get('c')}")
        if abs(M.effect_pp(b, c, n) - float(stats.get("gain_pp") or 0.0)) > 1e-3:
            problems.append("gain_pp does not follow from the pairs")
        if n and (stats.get("parent_mean") is None or stats.get("child_mean") is None or abs(parent_sum / n - float(stats["parent_mean"])) > 1e-6 or abs(child_sum / n - float(stats["child_mean"])) > 1e-6):
            problems.append("arm means do not follow from the table")
        expected_p = M.mcnemar_exact(b, c).p
        if stats.get("p") is None or abs(expected_p - float(stats["p"])) > 1e-9:
            problems.append(f"p recomputed as {expected_p}, recorded {stats.get('p')}")
    elif document.get("integrity") != "unpaired":
        problems.append(f"integrity {document.get('integrity')!r} on an unpaired outcome")
    try:
        verdict, reason = promotion_verdict(document)
    except (TypeError, ValueError) as error:  # a malformed count or threshold; the shape checks above name it
        return problems + [f"verdict cannot be recomputed: {error}"]
    if str(document.get("verdict_reason") or "") != reason:
        problems.append(f"verdict_reason recomputed as {reason!r}, recorded {document.get('verdict_reason')!r}")
    if verdict != document.get("verdict"):
        problems.append(f"verdict recomputed as {verdict!r}, recorded {document.get('verdict')!r}")
    return problems


def _run_pair(document: Mapping[str, Any]) -> dict[str, Any]:
    """The hashes that identify an outcome's run pair, as the deep-verification record stores them."""
    manifests = document.get("manifests") if isinstance(document.get("manifests"), Mapping) else {}
    files = document.get("result_files") if isinstance(document.get("result_files"), Mapping) else {}
    parent = manifests.get("parent") if isinstance(manifests.get("parent"), Mapping) else None
    return {
        "child": {"manifest_sha256": (manifests.get("child") or {}).get("sha256"), "result_files": dict(files.get("child") or {})},
        "parent": {"manifest_sha256": parent.get("sha256"), "result_files": dict(files.get("parent") or {})} if parent else None,
    }


def _attempt_problems(store: Store, document: Mapping[str, Any]) -> list[str]:
    """Whether the evaluations ``attempt.counted`` names are the generation's, and a paired outcome's own marker agrees.

    Every counted id must be an evaluation of the same generation in the store (a marker, or a paired outcome
    recorded without one). Nothing is reordered by time, so checkouts merged after each counted evaluations, or a
    clock set back, leave the count verifiable; a counted evaluation that is gone (a write-once file deleted) is a
    problem. A paired outcome must have been counted: its marker exists and records the same attempt and run pair.
    """
    problems = []
    generation = str(document.get("generation"))
    attempt = document.get("attempt") if isinstance(document.get("attempt"), Mapping) else {}
    counted = attempt.get("counted") if isinstance(attempt.get("counted"), list) else []
    question_set = (document.get("coverage") or {}).get("question_set")
    if question_set is None:
        question_set = next(iter(_recorded_rule(document)["question_sets"]))
    held = {str(record["id"]) for record in generation_attempts(store, generation, question_set=question_set)}
    missing = sorted(str(oid) for oid in counted if oid not in held)
    if missing:
        problems.append(f"attempt count: the outcome counted {', '.join(missing)}, which generation {generation} does not hold as evaluation(s)")
    registration = document.get("registration")
    if isinstance(registration, Mapping):
        reservation_id = str(registration.get("reservation_id", ""))
        if not _OUTCOME_ID_RE.fullmatch(reservation_id):
            problems.append("finished registration has no valid reservation identity")
        else:
            path = store.package / "reservations" / (reservation_id + ".json")
            held_registration = _read_marker(path) if path.is_file() else {}
            if (held_registration.get("registration") != registration or held_registration.get("counted") is not True
                    or held_registration.get("generation") != generation
                    or held_registration.get("attempt") != attempt
                    or held_registration.get("budget_cap") != _recorded_rule(document)["max_attempts"]
                    or held_registration.get("proposal") != document.get("proposal")
                    or held_registration.get("release") != document.get("release")):
                problems.append("finished registration differs from its counted authority reservation")
    if document.get("verdict") in VERDICTS and RELEASE_RE.fullmatch(generation) and _OUTCOME_ID_RE.fullmatch(str(document.get("id"))):
        path = attempt_marker_path(store, generation, str(document.get("id")))
        relative = path.relative_to(store.package).as_posix()
        if not path.is_file():
            problems.append(f"attempt marker {relative} is missing: the evaluation was never counted toward the cap")
        else:
            marker = _read_marker(path)
            if marker.get("attempt") != document.get("attempt") or marker.get("run_pair") != _run_pair(document):
                problems.append(f"attempt marker {relative} records another attempt or run pair than the outcome")
            if marker.get("cap") != _recorded_rule(document)["max_attempts"]:
                problems.append(f"attempt marker {relative} records another generation budget cap")
            reservation_id = str(marker.get("reservation_id", ""))
            if _OUTCOME_ID_RE.fullmatch(reservation_id):
                reservation_path = store.package / "reservations" / (reservation_id + ".json")
                reservation = _read_marker(reservation_path) if reservation_path.is_file() else {}
                if "budget_cap" in reservation and reservation["budget_cap"] != _recorded_rule(document)["max_attempts"]:
                    problems.append("outcome generation budget cap differs from its immutable reservation")
    return problems


def deep_verify_outcome(document: Mapping[str, Any], *, store: Store | None = None, plan: PlanView | None = None) -> list[str]:
    """Re-read the run directories an outcome cites and check every recorded hash, the pairing and the attempt count.

    Args:
        document (Mapping[str, Any]): An outcome document.
        store (Store | None): When given, the proposal's plan is read from it (unless ``plan`` is passed), every
            evaluation ``attempt.counted`` names must be the generation's in the store, and a paired outcome's own
            attempt marker must exist and agree (:func:`_attempt_problems`).
        plan (PlanView | None): The proposal's plan, when the caller already has it.

    Only what decides the verdict is checked — manifests, current row releases, result files, provenance, pairing, table, integrity
    detail, coverage, per-database counts (a function of the table) and the attempt count — so the check a
    promotion needs never depends on the metric registry; the information slices are :func:`information_drift`'s.

    Returns:
        list[str]: Problems found; empty when all of it reproduces. A missing run directory is a problem naming
            the directory (deep verification needs the machine that holds the runs, or a recorded verification,
            see :func:`recorded_deep_verification`).
    """
    plan = _plan_for(document, store, plan)
    problems = verify_outcome(document, plan=plan)
    if "rule" not in document:  # written before rule A: verify_outcome said so, and nothing of it is rebuilt
        return problems
    if store is not None:
        problems.extend(_attempt_problems(store, document))
    child_dir = Path(str(document.get("run_dir") or ""))
    if not child_dir.is_dir():
        return problems + [f"child run directory {child_dir} {_NOT_ON_THIS_MACHINE}"]
    child = read_run(child_dir)
    if problem := child.release_problem(store=store):
        problems.append(f"child run {child.name}: {problem}")
    manifests = document.get("manifests") if isinstance(document.get("manifests"), Mapping) else {}
    files = document.get("result_files") if isinstance(document.get("result_files"), Mapping) else {}
    if child.manifest_sha != (manifests.get("child") or {}).get("sha256"):
        problems.append("child manifest sha differs")
    if child.result_files != files.get("child"):
        problems.append("child result files differ")
    attribution = document.get("implementation_attribution")
    if attribution == _IMPLEMENTATION_ATTRIBUTION:
        implementations = document.get("implementations") if isinstance(document.get("implementations"), Mapping) else {}
        recorded_contexts = document.get("implementation_contexts") if isinstance(document.get("implementation_contexts"), Mapping) else {}
        if implementations.get("child") != child.manifest.get("implementation"):
            problems.append("child implementation block differs from its hashed manifest")
        try:
            child_contexts = _implementation_contexts(child)
        except (RuntimeError, ValueError) as error:
            return problems + [f"child implementation contexts cannot be rebuilt: {error}"]
        if recorded_contexts.get("child") != child_contexts:
            problems.append("child implementation contexts differ from its hashed manifest binding")
    if plan is not None:
        try:
            if _provenance(child, str(document.get("proposal")), plan.plan_digest, document.get("registration")) != document.get("provenance"):
                problems.append("provenance rebuilt from the child manifest differs")
        except (EvolveError, KeyError, TypeError) as error:
            problems.append("finished registration provenance differs: " + str(error))
    parent_dir = document.get("parent_run_dir")
    if (document.get("pairing") or {}).get("status") != "paired" or not parent_dir:
        return problems
    parent_path = Path(str(parent_dir))
    if not parent_path.is_dir():
        return problems + [f"parent run directory {parent_path} {_NOT_ON_THIS_MACHINE}"]
    parent = read_run(parent_path)
    if document.get("registration") is not None:
        from .finished import verify_registered_run

        try:
            verify_registered_run(child, parent, document["registration"], store=store)
        except (EvolveError, ValueError, OSError) as error:
            problems.append("finished registration verification failed: " + str(error))
    if problem := parent.release_problem(store=store):
        problems.append(f"parent run {parent.name}: {problem}")
    if parent.manifest_sha != (manifests.get("parent") or {}).get("sha256"):
        problems.append("parent manifest sha differs")
    if parent.result_files != files.get("parent"):
        problems.append("parent result files differ")
    contexts = None
    if attribution == _IMPLEMENTATION_ATTRIBUTION:
        if implementations.get("parent") != parent.manifest.get("implementation"):
            problems.append("parent implementation block differs from its hashed manifest")
        try:
            contexts = {"child": child_contexts, "parent": _implementation_contexts(parent)}
        except (RuntimeError, ValueError) as error:
            return problems + [f"parent implementation contexts cannot be rebuilt: {error}"]
        if recorded_contexts.get("parent") != contexts["parent"]:
            problems.append("parent implementation contexts differ from its hashed manifest binding")
        if _implementation_changes(parent, child, contexts) != document.get("changed_implementations"):
            problems.append("changed implementations rebuilt from the hashed manifests and unit contexts differ")
    if parent.identity != child.identity or parent.ns != child.ns:
        problems.append("the paired runs differ in identity or namespace")
    if plan is not None and abs(child.created_at - parent.created_at) / _DAY > plan.max_age_days:
        problems.append(f"the paired runs are further apart than the drift window of {plan.max_age_days:g} days")
    metric = M.resolve_metric(str(document.get("metric") or "ex"))
    ctx = _document_context(document)
    rebuilt, detail, _diverged = _pair_rows(parent, child, metric, ctx, strict_sources=document.get("source_verification") == "immutable/1",
                                          implementation_attribution=attribution, implementation_contexts=contexts,
                                          retrieval_integrity="degraded_retrieval" in (document.get("integrity_detail") or {}))
    if attribution == _IMPLEMENTATION_ATTRIBUTION:
        problems.extend(_sibling_replay_problems(parent, child, rebuilt))
    recorded = [(row["qid"], row["unit"], row["parent"], row["child"], row.get("diverged_at"), row.get("cause")) for row in document.get("table") or []]
    if ((rebuilt != document.get("table") if attribution == _IMPLEMENTATION_ATTRIBUTION or "degraded_retrieval" in detail else
         [(row["qid"], row["unit"], row["parent"], row["child"], row["diverged_at"], row["cause"]) for row in rebuilt] != recorded)):
        problems.append(f"pairing table rebuilt from the runs differs ({len(rebuilt)} rows rebuilt, {len(recorded)} recorded)")
    if detail != document.get("integrity_detail"):
        problems.append("integrity detail rebuilt from the runs differs")
    if _coverage(parent, child) != document.get("coverage"):
        problems.append("question coverage rebuilt from the runs differs")
    stats = document.get("stats") if isinstance(document.get("stats"), Mapping) else {}
    if _per_db(rebuilt) != stats.get("per_db"):
        problems.append("per-database counts rebuilt from the runs differ")
    return problems


def _document_context(document: Mapping[str, Any]) -> M.ExposureContext:
    """The context an outcome was computed in: its changed surfaces (an older outcome names changed entries) and knobs."""
    return M.ExposureContext(
        frozenset(str(name) for name in document.get("changed_surfaces") or document.get("changed_entries") or []),
        changed_knobs=frozenset(str(name) for name in document.get("changed_knobs") or []),
    )


def _plan_for(document: Mapping[str, Any], store: Store | None, plan: PlanView | None) -> PlanView | None:
    """The plan a check reads: the one given, else the proposal's from the store, else ``None``."""
    if plan is not None or store is None:
        return plan
    try:
        return plan_view(store.proposal(str(document.get("proposal"))).data)
    except PromptStoreError:
        return None


def information_drift(document: Mapping[str, Any], *, store: Store | None = None, plan: PlanView | None = None) -> list[str]:
    """The information slices of a paired outcome that no longer rebuild from its run directories.

    The exposure slice (under the plan's exposure predicate), every other metric the outcome recorded that the
    registry still resolves and the per-slot statistics are rebuilt and compared. They decide no verdict: a metric
    registered since the outcome was computed is not compared, one no longer registered is skipped, and a changed
    definition shows here as drift, which ``verify --deep`` reports and :func:`promote` never reads.

    Args:
        document (Mapping[str, Any]): An outcome document.
        store (Store | None): Where the proposal's plan is read from, unless ``plan`` is given.
        plan (PlanView | None): The proposal's plan; without one the exposure slice is not compared.

    Returns:
        list[str]: One line per slice that differs; empty for an outcome that is unpaired or whose runs are not on
            this machine.
    """
    plan = _plan_for(document, store, plan)
    child_dir, parent_dir = Path(str(document.get("run_dir") or "")), Path(str(document.get("parent_run_dir") or ""))
    if "rule" not in document or (document.get("pairing") or {}).get("status") != "paired" or not document.get("parent_run_dir") or not child_dir.is_dir() or not parent_dir.is_dir():
        return []
    child, parent = read_run(child_dir), read_run(parent_dir)
    metric = M.resolve_metric(str(document.get("metric") or "ex"))
    ctx = _document_context(document)
    changed = ctx.changed_surfaces
    attribution = document.get("implementation_attribution")
    _table, _detail, diverged = _pair_rows(parent, child, metric, ctx, strict_sources=document.get("source_verification") == "immutable/1" if attribution else False,
                                         implementation_attribution=attribution)
    stats = document.get("stats") if isinstance(document.get("stats"), Mapping) else {}
    drift = []
    if plan is not None and _exposed_stats(parent, child, metric, plan.exposure, changed, diverged, ctx, implementation_attribution=attribution) != stats.get("exposed"):
        drift.append(f"exposure slice ({plan.exposure}) rebuilt from the runs differs")
    shared = sorted(set(parent.rows) & set(child.rows))
    parent_rows, child_rows = [parent.rows[qid] for qid in shared], [child.rows[qid] for qid in shared]
    recorded = stats.get("metrics") if isinstance(stats.get("metrics"), Mapping) else {}
    for name in sorted(recorded):
        try:
            rebuilt = _paired_stats(M.resolve_metric(str(name)), parent_rows, child_rows, ctx)
        except KeyError:
            continue
        if rebuilt != recorded[name]:
            drift.append(f"information metric {name} rebuilt from the runs differs")
    if "slots" in stats and stats["slots"] != {"parent": M.slot_stats(parent_rows), "child": M.slot_stats(child_rows)}:
        drift.append("per-slot statistics rebuilt from the runs differ")
    return drift


# ---------------------------------------------------------------------------------------
# Decisions


def _actor(actor: str | None, env: Mapping[str, str] | None = None) -> str:
    """Who a decision records: the given actor; else ``agent:<marker>`` when an agent harness runs the verb, so an
    agent's decision is never filed under the person whose shell it runs in; else the person (``$USER``).

    Args:
        actor (str | None): The explicit actor (``--actor``).
        env (Mapping[str, str] | None): Environment to read the harness markers and ``USER`` from; ``os.environ`` by
            default. ``RUBIKSQL_PROMPT_HUMAN=1`` marks a person under a harness, who is recorded as ``$USER``.

    Returns:
        str: The actor label.
    """
    if actor:
        return actor
    present = _harness_markers(env)
    if present:
        return f"agent:{present[0]}"
    environment = env if env is not None else os.environ
    return environment.get("USER") or "human"


def _row_ref(ref: str, store: Store) -> tuple[str, str, str]:
    """``name:version`` (16 hex, or a unique prefix of at least 6) to ``(name, version, content_key)``.

    Only prompt row versions are revocable; a typed rule, tool or parameter-set version (``rule:<title>@<version>``,
    ``tool:<name>@…``, ``parameters:knobs@…``) is refused, since a release pins it by digest: retire the release instead.
    """
    if "@" in ref or ref.split(":", 1)[0] in {"rule", "rules", "tool", "tools", "parameters"}:
        raise EvolveError(f"{ref!r}: typed rule, tool and parameter-set versions are not revocable; revoke and unrevoke "
                          "name a prompt row version (name:<version>), and a release that pins a typed version is retired instead")
    if ":" not in ref:
        raise EvolveError(f"{ref!r}: expected name:<version>")
    name, prefix = ref.rsplit(":", 1)
    try:
        check_name(name)
    except PolicyError as error:
        raise EvolveError(str(error)) from error
    if len(prefix) < 6 or not all(char in "0123456789abcdef" for char in prefix):
        raise EvolveError(f"{ref!r}: the version prefix must be at least 6 hex digits")
    directory = store.package / "objects" / name
    matches = sorted(path.stem for path in directory.glob(f"{prefix}*.json")) if directory.is_dir() else []
    matches = [stem for stem in matches if VERSION_RE.fullmatch(stem)]
    if len(matches) != 1:
        raise EvolveError(f"{ref!r} matches {len(matches)} versions in the store: {matches}")
    key = store.object(name, matches[0]).data["row"]["manifest"]["meta"]["content_key"]
    return name, matches[0], key
