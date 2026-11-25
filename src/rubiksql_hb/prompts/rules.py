"""Author and materialize a release's general rule corpus.

Explicit rule changes retain structured, append-only provenance and content-pinned
adjudication records. Comparative shares are advisory under the current policy;
the complete paired evolution outcome decides adoption. Structural consistency,
current-input membership and model-facing leak checks remain blocking. Historical
records retain their original question-set pins and text.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import entries as E
from .canon import canon, sha256_hex
from .policy import RELEASE_RE

__all__ = [
    "ADJUDICATIONS",
    "ADJUDICATION_FIELDS",
    "ADJUDICATION_QUESTION_SET",
    "APPLIED_FILE",
    "APPLY_STAGE_VERSION",
    "ADJUDICATION_QUESTIONS_FILE",
    "DOSSIER_INSTRUCTIONS_FILE",
    "EVOLVED",
    "EVOLVED_FIELDS",
    "EVOLVED_NOTE",
    "EVOLVED_STATUS",
    "MANAGED_GENERATION",
    "RULE_FIELDS",
    "SCHEMA_APPLIED",
    "Adjudication",
    "AppliedRules",
    "RuleCorpusError",
    "RulesChange",
    "WorkspaceRules",
    "apply_rules",
    "change_problems",
    "change_summary",
    "corpus_lines",
    "corpus_problems",
    "corpus_text",
    "evolved_origins",
    "knowledge_workspaces",
    "qid",
    "rule_record",
    "rules_digest",
    "rules_problems",
    "shared_triggers_after",
]

#: Current evolution input. Historical adjudications retain their own content pins;
#: their measured shares are advisory, while structural provenance remains checked.
ADJUDICATION_QUESTION_SET = "sha256:cfedb9026a9fc62760baaab581e933eda466b64a1f98400eeb0f19ae594e9b87"
ADJUDICATION_QUESTIONS_FILE = ("resources", "official_input", "evolve_800_20261027.json")
#: The contract of a rules change the evolution loop's dossier points at, relative to the checkout root.
#: The provenance lists of the evolution loop: the rules it added, and the conflicts they won.
EVOLVED = "dev_evolved_rules"
ADJUDICATIONS = "dev_evolution_adjudications"
#: The fields of a rule record, in the order the corpus writes them.
RULE_FIELDS = ("scope", "title", "desc", "tags", "triggers", "example")
#: The fields of a ``dev_evolved_rules`` entry.
EVOLVED_FIELDS = ("title", "parent_release", "dossier_qids")
#: The fields of a ``dev_evolution_adjudications`` record: the brief's seven, after the release they were measured on.
ADJUDICATION_FIELDS = ("parent_release", "new_rule", "existing_rule", "question_set", "applicable_qids", "follows_new_qids", "follows_existing_qids", "method")
#: ``generation.status`` of a corpus that holds dev-evolved rules.
EVOLVED_STATUS = "train_grounded_plus_dev_evolved"
#: ``generation.public_dev_used_note`` of a corpus that holds dev-evolved rules (``public_dev_used`` is then true).
EVOLVED_NOTE = (
    "Scope of the flag above: the whole corpus, which since the evolution loop holds rules written from public-dev "
    "failures. They are listed in provenance.dev_evolved_rules, each with the release it was proposed on and the "
    "questions of the dossier it came from. Every replacement or removal is recorded in "
    "provenance.dev_evolution_adjudications with its content-pinned question set, method and advisory shares; "
    "an unmeasured share is recorded as unmeasured, and the full paired outcome decides adoption. "
    "The other rules are the train-grounded corpus as it was: a provenance record under a "
    "rule's title says public_dev_used false for that rule, and the aggregate dev measurements that amended two of "
    "them are listed in dev_measurement_amendments. The gate audits every added rule for dev question text, "
    "question ids, gold SQL and gold values."
)
#: The ``generation`` fields a rules change writes; a change leaves every other one as its parent had it.
#: ``shared_triggers`` follows the rules (:func:`shared_triggers_after`): a change only drops the owners that left.
MANAGED_GENERATION = ("rule_count", "public_dev_used", "public_dev_used_note", "status", "dev_evolved_rule_count", "shared_triggers")
#: The record a knowledge workspace keeps of the corpus :func:`apply_rules` last materialised into it.
APPLIED_FILE = "rules_applied.json"
SCHEMA_APPLIED = "rubiksql.rules-applied/1"
#: Version of :func:`apply_rules`' materialisation in the knowledge ledger (``scripts/knowledge_build.py``), where it is
#: the ``general_rules`` stage. Bump it by hand whenever a change would make it write other rule rows from the same
#: corpus.
APPLY_STAGE_VERSION = 1

_TITLE_RE = re.compile(r"[a-z][a-z0-9_]{0,254}")
_QID_RE = re.compile(r"Q\d{5}")


class RuleCorpusError(ValueError):
    """A rule corpus, a rules change or a knowledge workspace that breaks the rule-evolution contract."""


# ---------------------------------------------------------------------------------------
# Records


def corpus_text(corpus: Mapping[str, Any]) -> str:
    """The one text a ``rules.general`` row may hold for a corpus: the spelling of ``general_rules.json``.

    Args:
        corpus (Mapping[str, Any]): ``{"version", "generation", "rules", "provenance"}``.

    Returns:
        str: ``json.dumps(corpus, ensure_ascii=False, indent=2)``, which the runtime loader requires.
    """
    return json.dumps(corpus, ensure_ascii=False, indent=2)


def qid(value: Any) -> str:
    """A question id in the pipeline's spelling, ``Q%05d``.

    Args:
        value (Any): A BIRD ``question_id`` (an int from 0 to 99999, as the dev file writes it) or a ``Q%05d`` string,
            as run rows and dossiers write it.

    Returns:
        str: The ``Q%05d`` spelling.

    Raises:
        RuleCorpusError: If the value is neither.
    """
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 99999:
        return f"Q{value:05d}"
    if isinstance(value, str) and _QID_RE.fullmatch(value):
        return value
    raise RuleCorpusError(f"{value!r} is not a question id (an int question_id or a Q%05d string)")


def _qids(value: Any, what: str, *, stored: bool = False) -> list[str]:
    """A list of question ids, each once; sorted as the corpus stores it (``stored`` also refuses another order)."""
    if not isinstance(value, list):
        raise RuleCorpusError(f"{what} must be a list of question ids")
    ids = [qid(item) for item in value]
    repeated = sorted({item for item in ids if ids.count(item) > 1})
    if repeated:
        raise RuleCorpusError(f"{what} lists {', '.join(repeated)} more than once")
    if stored and ids != sorted(ids):
        raise RuleCorpusError(f"{what} must be sorted")
    return sorted(ids)


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _rule_problems(rule: Any, what: str) -> list[str]:
    """What keeps one rule record from the corpus: its fields, their types, a global scope and a title identity."""
    if not isinstance(rule, Mapping):
        return [f"{what}: a rule is a JSON object"]
    problems: list[str] = []
    if set(rule) != set(RULE_FIELDS):
        problems.append(f"{what}: a rule has exactly the fields {list(RULE_FIELDS)}, got {sorted(rule)}")
    title = rule.get("title")
    if not isinstance(title, str) or not _TITLE_RE.fullmatch(title):
        problems.append(f"{what}: title {title!r} is not a lower snake_case identifier")
    if rule.get("scope") != "global":
        problems.append(f"{what}: the general corpus holds global rules only, got scope {rule.get('scope')!r}")
    for key in ("desc", "example"):
        if not isinstance(rule.get(key), str):
            problems.append(f"{what}: {key} must be a string")
    if isinstance(rule.get("desc"), str) and not rule["desc"].strip():
        problems.append(f"{what}: desc is empty")
    for key in ("tags", "triggers"):
        if not _string_list(rule.get(key)):
            problems.append(f"{what}: {key} must be a list of strings")
    return problems


def rule_record(value: Any, what: str) -> dict[str, Any]:
    """One rule record as the corpus writes it, fields in :data:`RULE_FIELDS` order.

    Args:
        value (Any): The authored record: title, desc, example, scope, tags and triggers.
        what (str): Where it came from, for messages.

    Returns:
        dict[str, Any]: The record.

    Raises:
        RuleCorpusError: If a field is missing, extra or mistyped, the scope is not ``global`` or the title is not a
            lower snake_case identifier.
    """
    problems = _rule_problems(value, what)
    if problems:
        raise RuleCorpusError("; ".join(problems))
    return {key: value[key] for key in RULE_FIELDS}


def _labelled(provenance: Mapping[str, Any], key: str) -> list[Any]:
    value = provenance.get(key, [])
    return list(value) if isinstance(value, list) else []


def _records(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """The per-title provenance records: every provenance key but the labelled lists."""
    return {title: record for title, record in provenance.items() if title not in (EVOLVED, ADJUDICATIONS)}


@dataclass(frozen=True)
class Adjudication:
    """One measured conflict between a rule a proposal brings and a rule already in the corpus.

    Over the Mini-Dev 500 evolution questions where the two rules' shared condition applies (``applicable``), the gold
    SQLs that follow the new rule's prescription (``follows_new``) and those that follow the existing rule's
    (``follows_existing``); the rule with the larger share stays. The proposer reads the gold SQL and classifies;
    no pipeline call makes the measurement.

    Attributes:
        new_rule (str): Title of the rule the proposal brings.
        existing_rule (str): Title of the conflicting rule already in the corpus (the same title for an edit).
        question_set (str): The sha256 of the questions counted over, :data:`ADJUDICATION_QUESTION_SET`.
        applicable (tuple[str, ...]): The questions where the shared condition applies, ``Q%05d``, sorted.
        follows_new (tuple[str, ...]): The applicable questions whose gold SQL follows the new rule.
        follows_existing (tuple[str, ...]): The applicable questions whose gold SQL follows the existing rule.
        method (str): How the questions were selected and classified.
    """

    new_rule: str
    existing_rule: str
    question_set: str
    applicable: tuple[str, ...]
    follows_new: tuple[str, ...]
    follows_existing: tuple[str, ...]
    method: str

    @classmethod
    def from_json(cls, value: Any, what: str) -> "Adjudication":
        """Parse the authored record ``{new_rule, existing_rule, question_set, applicable_qids, follows_new_qids,
        follows_existing_qids, method}``; question ids may be ints or ``Q%05d`` strings.

        Args:
            value (Any): The authored record.
            what (str): Where it came from, for messages.

        Returns:
            Adjudication: The parsed record, question ids in ``Q%05d`` spelling and sorted.

        Raises:
            RuleCorpusError: If a field is missing, extra or malformed.
        """
        fields = ADJUDICATION_FIELDS[1:]
        if not isinstance(value, Mapping) or set(value) != set(fields):
            got = sorted(value) if isinstance(value, Mapping) else type(value).__name__
            raise RuleCorpusError(f"{what}: an adjudication has exactly the fields {list(fields)}, got {got}")
        for key in ("new_rule", "existing_rule"):
            if not isinstance(value[key], str) or not _TITLE_RE.fullmatch(value[key]):
                raise RuleCorpusError(f"{what}.{key}: {value[key]!r} is not a rule title")
        if not isinstance(value["method"], str) or not value["method"].strip():
            raise RuleCorpusError(f"{what}.method: say how the questions were selected and classified")
        if not isinstance(value["question_set"], str):
            raise RuleCorpusError(f"{what}.question_set must be the sha256:<hex> of the evolution questions")
        return cls(
            value["new_rule"],
            value["existing_rule"],
            value["question_set"],
            tuple(_qids(value["applicable_qids"], f"{what}.applicable_qids")),
            tuple(_qids(value["follows_new_qids"], f"{what}.follows_new_qids")),
            tuple(_qids(value["follows_existing_qids"], f"{what}.follows_existing_qids")),
            value["method"],
        )

    def record(self, parent_release: str) -> dict[str, Any]:
        """The record ``provenance.dev_evolution_adjudications`` keeps.

        Args:
            parent_release (str): The release whose corpus held the existing rule.

        Returns:
            dict[str, Any]: The fields of :data:`ADJUDICATION_FIELDS`.
        """
        return {
            "parent_release": parent_release,
            "new_rule": self.new_rule,
            "existing_rule": self.existing_rule,
            "question_set": self.question_set,
            "applicable_qids": list(self.applicable),
            "follows_new_qids": list(self.follows_new),
            "follows_existing_qids": list(self.follows_existing),
            "method": self.method,
        }


def _adjudication_problems(record: Any, what: str) -> list[str]:
    """Check an evidence record's structure; its relative shares do not authorize or veto a rule."""
    if not isinstance(record, Mapping) or set(record) != set(ADJUDICATION_FIELDS):
        got = sorted(record) if isinstance(record, Mapping) else type(record).__name__
        return [f"{what}: an adjudication has exactly the fields {list(ADJUDICATION_FIELDS)}, got {got}"]
    problems: list[str] = []
    if not isinstance(record["parent_release"], str) or not RELEASE_RE.fullmatch(record["parent_release"]):
        problems.append(f"{what}: parent_release {record['parent_release']!r} is not a release id")
    for key in ("new_rule", "existing_rule"):
        if not isinstance(record[key], str) or not _TITLE_RE.fullmatch(record[key]):
            problems.append(f"{what}: {key} {record[key]!r} is not a rule title")
    if not isinstance(record["question_set"], str) or re.fullmatch(r"sha256:[0-9a-f]{64}", record["question_set"]) is None:
        problems.append(f"{what}: question_set must be a content-pinned sha256 digest")
    if not isinstance(record["method"], str) or not record["method"].strip():
        problems.append(f"{what}: method is empty")
    try:
        applicable = set(_qids(record["applicable_qids"], f"{what}.applicable_qids", stored=True))
        new = set(_qids(record["follows_new_qids"], f"{what}.follows_new_qids", stored=True))
        existing = set(_qids(record["follows_existing_qids"], f"{what}.follows_existing_qids", stored=True))
    except RuleCorpusError as error:
        return problems + [str(error)]
    if not new <= applicable or not existing <= applicable:
        problems.append(f"{what}: the questions following either rule must be applicable ones; outside: {sorted((new | existing) - applicable)[:5]}")
    if new & existing:
        problems.append(f"{what}: a gold SQL follows one of two conflicting prescriptions, not both: {sorted(new & existing)[:5]}")
    return problems


def shared_triggers_after(declared: Any, rules: Sequence[Any]) -> Any:
    """``generation.shared_triggers`` once a change has moved the rules: the reviewed sharing that still holds.

    The builder records a trigger phrase several rules carry, as written, with the sorted titles that carry it
    (``rulegen._report_shared_triggers``), so each sharing was reviewed. A rules change keeps a phrase's owners that are
    still in the corpus and still carry the phrase, and drops the phrase once fewer than two do; it never declares a new
    sharing (the audit refuses a trigger another rule owns).

    Args:
        declared (Any): The parent's ``shared_triggers``; anything but a mapping is returned as it is.
        rules (Sequence[Any]): The child's rule records.

    Returns:
        Any: The phrases still shared, each with its remaining owners sorted.
    """
    if not isinstance(declared, Mapping):
        return declared
    triggers = {rule.get("title"): set(rule.get("triggers") or []) for rule in rules if isinstance(rule, Mapping)}
    kept: dict[str, list[str]] = {}
    for phrase, titles in declared.items():
        owners = sorted(title for title in (titles if isinstance(titles, list) else []) if phrase in triggers.get(title, set()))
        if len(owners) > 1:
            kept[phrase] = owners
    return kept


# ---------------------------------------------------------------------------------------
# The proposal op


@dataclass(frozen=True)
class RulesChange:
    """A proposal's change to the general rule corpus (``op: rules``): rules added, and conflicts resolved.

    Authoring form::

        {"op": "rules", "reason": "<why the corpus changes>",
         "add": [{"title", "desc", "example", "scope": "global", "tags", "triggers", "dossier_qids"}],
         "replace": [{"title": <existing title>, "rule": {<full record>}, "dossier_qids", "adjudication": {...}}],
         "remove": [{"title": <existing title>, "adjudication": {...}}]}

    ``name`` may be omitted and is otherwise ``rules.general``. ``replace`` puts the new record in the existing
    rule's place (its title may stay, for an edit, or change); ``remove`` drops the existing rule for a rule this
    change adds, which its adjudication names as ``new_rule``. Both keep an adjudication record whose shares are advisory; nothing
    else touches an existing rule.

    Attributes:
        reason (str): The reason of the new ``rules.general`` version.
        add (tuple[tuple[dict[str, Any], tuple[str, ...]], ...]): New rule records with their dossier question ids.
        replace (tuple[tuple[str, dict[str, Any], tuple[str, ...], Adjudication], ...]): ``(existing title, new record,
            dossier question ids, adjudication)``.
        remove (tuple[tuple[str, Adjudication], ...]): ``(existing title, adjudication)``.
    """

    reason: str
    add: tuple[tuple[dict[str, Any], tuple[str, ...]], ...] = ()
    replace: tuple[tuple[str, dict[str, Any], tuple[str, ...], Adjudication], ...] = ()
    remove: tuple[tuple[str, Adjudication], ...] = ()
    name: str = field(default=E.RULES_ENTRY, init=False)
    op: str = field(default="rules", init=False)

    @classmethod
    def from_json(cls, item: Mapping[str, Any], what: str = "rules") -> "RulesChange":
        """Parse the authoring form of one ``op: rules`` change.

        Args:
            item (Mapping[str, Any]): The change as the proposal file writes it.
            what (str): Where it came from, for messages.

        Returns:
            RulesChange: The parsed change.

        Raises:
            RuleCorpusError: If the name is not ``rules.general``, a key is unknown, the change is empty, a record is
                malformed or a dossier question list is empty.
        """
        known = {"op", "name", "reason", "add", "replace", "remove", "description"}
        unknown = sorted(set(item) - known)
        if unknown:
            raise RuleCorpusError(f"{what}: unknown key(s) {unknown}; an op rules change holds {sorted(known - {'description'})}")
        if item.get("name", E.RULES_ENTRY) != E.RULES_ENTRY:
            raise RuleCorpusError(f"{what}: op rules changes {E.RULES_ENTRY!r}, not {item.get('name')!r}")
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise RuleCorpusError(f"{what}: a reason is required (the new {E.RULES_ENTRY} version's reason)")

        def entries(key: str) -> list[Mapping[str, Any]]:
            value = item.get(key, [])
            if not isinstance(value, list) or any(not isinstance(entry, Mapping) for entry in value):
                raise RuleCorpusError(f"{what}.{key} must be a list of objects")
            return value

        def dossier(entry: Mapping[str, Any], where: str) -> tuple[str, ...]:
            ids = _qids(entry.get("dossier_qids"), f"{where}.dossier_qids")
            if not ids:
                raise RuleCorpusError(f"{where}.dossier_qids: name the dossier questions the rule came from")
            return tuple(ids)

        def existing(entry: Mapping[str, Any], keys: set[str], where: str) -> str:
            if set(entry) != keys:
                raise RuleCorpusError(f"{where}: an entry has exactly {sorted(keys)}, got {sorted(entry)}")
            if not isinstance(entry["title"], str) or not _TITLE_RE.fullmatch(entry["title"]):
                raise RuleCorpusError(f"{where}.title: {entry['title']!r} is not a rule title")
            return entry["title"]

        add = []
        for index, entry in enumerate(entries("add")):
            where = f"{what}.add[{index}]"
            add.append((rule_record({key: value for key, value in entry.items() if key != "dossier_qids"}, where), dossier(entry, where)))
        replace = []
        for index, entry in enumerate(entries("replace")):
            where = f"{what}.replace[{index}]"
            title = existing(entry, {"title", "rule", "dossier_qids", "adjudication"}, where)
            replace.append((title, rule_record(entry["rule"], f"{where}.rule"), dossier(entry, where), Adjudication.from_json(entry["adjudication"], f"{where}.adjudication")))
        remove = []
        for index, entry in enumerate(entries("remove")):
            where = f"{what}.remove[{index}]"
            title = existing(entry, {"title", "adjudication"}, where)
            remove.append((title, Adjudication.from_json(entry["adjudication"], f"{where}.adjudication")))
        if not (add or replace or remove):
            raise RuleCorpusError(f"{what}: add, replace or remove at least one rule")
        return cls(reason, tuple(add), tuple(replace), tuple(remove))

    def corpus_after(self, corpus: Mapping[str, Any], parent_release: str) -> dict[str, Any]:
        """The child corpus: the parent's with this change, its provenance entries and its flags.

        Replaced rules keep their position, removed rules leave with their per-title provenance record and their
        evolved entry, added rules follow the parent's in order. Every added or replacing rule gets a
        ``dev_evolved_rules`` entry naming ``parent_release`` and its dossier questions, every adjudication a
        ``dev_evolution_adjudications`` record, and ``generation`` its rule count, the dev-evolved flags and the
        declared trigger sharing that still holds (:func:`shared_triggers_after`).

        Args:
            corpus (Mapping[str, Any]): The parent's corpus (``Release.rules_general()``).
            parent_release (str): The release the change is proposed on.

        Returns:
            dict[str, Any]: The child corpus; :func:`corpus_text` spells it.

        Raises:
            RuleCorpusError: If an added title is already in the corpus, a replaced or removed title is not, a title
                repeats within the change, or an adjudication names other rules than the change resolves, contains malformed evidence or lists inconsistent questions.
        """
        rules = [dict(rule) for rule in corpus.get("rules") or []]
        titles = [rule.get("title") for rule in rules]
        provenance = dict(corpus.get("provenance") or {})
        records = _records(provenance)
        evolved = _labelled(provenance, EVOLVED)
        adjudications = _labelled(provenance, ADJUDICATIONS)
        brought = [record["title"] for record, _ids in self.add] + [record["title"] for _title, record, _ids, _adj in self.replace]
        resolved = [title for title, *_rest in self.replace] + [title for title, _adj in self.remove]
        for label, names in (("brings", brought), ("resolves", resolved)):
            repeated = sorted({name for name in names if names.count(name) > 1})
            if repeated:
                raise RuleCorpusError(f"the change {label} {', '.join(repeated)} more than once")
        for title in resolved:
            if title not in titles:
                raise RuleCorpusError(f"{title}: not a rule of {parent_release}'s corpus; only a rule already there is replaced or removed")

        def entry(title: str, ids: tuple[str, ...]) -> dict[str, Any]:
            return {"title": title, "parent_release": parent_release, "dossier_qids": list(ids)}

        for label, adjudication in [*((f"replace {old}", adj) for old, _record, _ids, adj in self.replace), *((f"remove {old}", adj) for old, adj in self.remove)]:
            problems = _adjudication_problems(adjudication.record(parent_release), label)
            if problems:
                raise RuleCorpusError("; ".join(problems))
        for old, record, ids, adjudication in self.replace:
            if (adjudication.existing_rule, adjudication.new_rule) != (old, record["title"]):
                raise RuleCorpusError(f"replace {old}: its adjudication must name existing_rule {old!r} and new_rule {record['title']!r}")
            if record["title"] != old and (record["title"] in titles or record["title"] in resolved):
                raise RuleCorpusError(f"replace {old}: {record['title']} is already a rule of the corpus")
            index = titles.index(old)
            rules[index], titles[index] = record, record["title"]
            records.pop(old, None)
            evolved = [item for item in evolved if item.get("title") != old]
            evolved.append(entry(record["title"], ids))
            adjudications.append(adjudication.record(parent_release))
        for old, adjudication in self.remove:
            if adjudication.existing_rule != old or adjudication.new_rule not in brought or adjudication.new_rule == old:
                raise RuleCorpusError(f"remove {old}: its adjudication must name existing_rule {old!r} and, as new_rule, a rule this change adds or puts in place")
            index = titles.index(old)
            del rules[index], titles[index]
            records.pop(old, None)
            evolved = [item for item in evolved if item.get("title") != old]
            adjudications.append(adjudication.record(parent_release))
        for record, ids in self.add:
            if record["title"] in titles:
                raise RuleCorpusError(f"add {record['title']}: already a rule of the corpus; a conflicting rule is replaced or removed through an adjudication, never re-added")
            rules.append(record)
            titles.append(record["title"])
            evolved.append(entry(record["title"], ids))

        generation: dict[str, Any] = {}
        for key, value in (corpus.get("generation") or {}).items():
            if key != "dev_evolved_rule_count":
                generation[key] = value
            if key == "rule_count":
                generation["rule_count"] = len(rules)
                generation["dev_evolved_rule_count"] = len(evolved)
        generation.setdefault("rule_count", len(rules))
        generation.setdefault("dev_evolved_rule_count", len(evolved))
        generation.update(public_dev_used=True, public_dev_used_note=EVOLVED_NOTE, status=EVOLVED_STATUS)
        if "shared_triggers" in generation:
            generation["shared_triggers"] = shared_triggers_after(generation["shared_triggers"], rules)
        return {
            "version": corpus.get("version"),
            "generation": generation,
            "rules": rules,
            "provenance": {**records, EVOLVED: evolved, ADJUDICATIONS: adjudications},
        }


# ---------------------------------------------------------------------------------------
# What the gate and the guards hold a corpus to


def corpus_problems(corpus: Any) -> list[str]:
    """What keeps a rule corpus from shipping, on its own: rule records, labelled provenance and the flags.

    Every rule is a global record of :data:`RULE_FIELDS` under a unique title. Provenance holds one record per
    train-grounded rule under its title (each saying ``public_dev_used`` false) and, once the evolution loop has
    added a rule, the two labelled lists: ``dev_evolved_rules`` — one entry per dev-evolved rule of the corpus, none
    of them with a per-title record — and ``dev_evolution_adjudications``, each a conflict the new rule won over the
    registered questions. ``generation.rule_count`` counts the rules, and ``public_dev_used`` is true exactly
    when the corpus holds a dev-evolved rule, with :data:`EVOLVED_STATUS`, ``dev_evolved_rule_count`` and a note that
    names both lists. A rule with neither a record nor an entry is allowed here (the corpus has two, measured ones);
    :func:`change_problems` refuses a change that adds one.

    Args:
        corpus (Any): The parsed corpus.

    Returns:
        list[str]: One line per problem; empty when the corpus is consistent.
    """
    if not isinstance(corpus, Mapping):
        return ["the corpus is a JSON object"]
    rules, provenance, generation = corpus.get("rules"), corpus.get("provenance"), corpus.get("generation")
    if not isinstance(rules, list) or not isinstance(provenance, Mapping) or not isinstance(generation, Mapping):
        return ["the corpus holds a rules list and provenance and generation objects"]
    problems: list[str] = []
    titles: list[str] = []
    for index, rule in enumerate(rules):
        problems.extend(_rule_problems(rule, f"rules[{index}]"))
        if isinstance(rule, Mapping) and isinstance(rule.get("title"), str):
            titles.append(rule["title"])
    repeated = sorted({title for title in titles if titles.count(title) > 1})
    if repeated:
        problems.append(f"titles repeat: {repeated}")
    records = _records(provenance)
    orphans = sorted(set(records) - set(titles))
    if orphans:
        problems.append(f"provenance records name no rule of the corpus: {orphans}")
    for title, record in records.items():
        validation = record.get("validation") if isinstance(record, Mapping) else None
        if not isinstance(validation, Mapping) or validation.get("public_dev_used") is not False:
            problems.append(f"provenance.{title}: a per-title record is a train-grounded rule's and says validation.public_dev_used false")
    for key in (EVOLVED, ADJUDICATIONS):
        if key in provenance and not isinstance(provenance[key], list):
            problems.append(f"provenance.{key} must be a list")
    evolved = _labelled(provenance, EVOLVED)
    seen: set[str] = set()
    for index, item in enumerate(evolved):
        where = f"provenance.{EVOLVED}[{index}]"
        if not isinstance(item, Mapping) or set(item) != set(EVOLVED_FIELDS):
            problems.append(f"{where}: an entry has exactly the fields {list(EVOLVED_FIELDS)}")
            continue
        title = item["title"]
        if title not in titles:
            problems.append(f"{where}: {title!r} is not a rule of the corpus")
        if title in seen:
            problems.append(f"{where}: {title} is listed twice")
        seen.add(title)
        if title in records:
            problems.append(f"{where}: {title} has a train provenance record too; a rule is train-grounded or dev-evolved")
        if not isinstance(item["parent_release"], str) or not RELEASE_RE.fullmatch(item["parent_release"]):
            problems.append(f"{where}: parent_release {item['parent_release']!r} is not a release id")
        try:
            if not _qids(item["dossier_qids"], f"{where}.dossier_qids", stored=True):
                problems.append(f"{where}: dossier_qids is empty")
        except RuleCorpusError as error:
            problems.append(str(error))
    adjudications = _labelled(provenance, ADJUDICATIONS)
    for index, record in enumerate(adjudications):
        problems.extend(_adjudication_problems(record, f"provenance.{ADJUDICATIONS}[{index}]"))
    if generation.get("rule_count") != len(rules):
        problems.append(f"generation.rule_count is {generation.get('rule_count')!r}; the corpus holds {len(rules)} rules")
    if evolved:
        if generation.get("public_dev_used") is not True:
            problems.append("generation.public_dev_used must be true: the corpus holds dev-evolved rules")
        if generation.get("status") != EVOLVED_STATUS:
            problems.append(f"generation.status must be {EVOLVED_STATUS!r} once the corpus holds dev-evolved rules")
        if generation.get("dev_evolved_rule_count") != len(evolved):
            problems.append(f"generation.dev_evolved_rule_count is {generation.get('dev_evolved_rule_count')!r}; {len(evolved)} rules are dev-evolved")
        note = generation.get("public_dev_used_note")
        if not isinstance(note, str) or EVOLVED not in note or ADJUDICATIONS not in note:
            problems.append(f"generation.public_dev_used_note must name provenance.{EVOLVED} and provenance.{ADJUDICATIONS}")
    else:
        if generation.get("public_dev_used") is not False:
            problems.append("generation.public_dev_used must be false: no rule is labelled dev-evolved")
        if adjudications:
            problems.append("adjudications without a dev-evolved rule: every conflict's winner is a dev-evolved rule")
        if "dev_evolved_rule_count" in generation:
            problems.append("generation.dev_evolved_rule_count without a dev-evolved rule")
    return problems


def _by_title(corpus: Mapping[str, Any]) -> dict[str, Any]:
    return {rule.get("title"): rule for rule in corpus.get("rules") or [] if isinstance(rule, Mapping)}


def _evolved_by_title(corpus: Mapping[str, Any]) -> dict[str, Any]:
    return {item.get("title"): item for item in _labelled(corpus.get("provenance") or {}, EVOLVED) if isinstance(item, Mapping)}


def change_problems(parent: Mapping[str, Any], child: Mapping[str, Any], *, question_ids: Iterable[str] | None = None) -> list[str]:
    """What keeps a child corpus from following its parent's under the rule-evolution contract.

    A rule of the parent stays byte for byte, with its provenance record and its evolved entry, unless an
    adjudication the child appends names it as ``existing_rule``: then it is removed for a rule the child brings, or
    edited in place (``new_rule`` is its own title) and so becomes dev-evolved. Every rule new in the child carries a
    new ``dev_evolved_rules`` entry and no per-title record; the recorded adjudications are append-only, each new one
    resolves exactly one parent rule for a rule the child brings; ``version`` and every ``generation`` field but
    :data:`MANAGED_GENERATION` stay as the parent has them, and ``shared_triggers`` is the parent's with the owners that
    left dropped (:func:`shared_triggers_after`). With ``question_ids``, every question a new entry or
    adjudication names must be one of them.

    Args:
        parent (Mapping[str, Any]): The parent's corpus.
        child (Mapping[str, Any]): The child's corpus.
        question_ids (Iterable[str] | None): The ``Q%05d`` ids of the Mini-Dev 500 evolution questions.

    Returns:
        list[str]: One line per problem; empty when the change follows the contract.
    """
    problems: list[str] = []
    before, after = _by_title(parent), _by_title(child)
    parent_provenance, child_provenance = parent.get("provenance") or {}, child.get("provenance") or {}
    records_before, records_after = _records(parent_provenance), _records(child_provenance)
    evolved_before, evolved_after = _evolved_by_title(parent), _evolved_by_title(child)
    recorded, adjudications = _labelled(parent_provenance, ADJUDICATIONS), _labelled(child_provenance, ADJUDICATIONS)
    if adjudications[: len(recorded)] != recorded:
        problems.append(f"provenance.{ADJUDICATIONS}: the child rewrites or drops adjudications its parent recorded; they are append-only")
    appended = [record for record in adjudications[len(recorded):] if isinstance(record, Mapping)]
    resolving: dict[Any, list[Mapping[str, Any]]] = {}
    for record in appended:
        resolving.setdefault(record.get("existing_rule"), []).append(record)
    brought = {title for title in after if title not in before or after[title] != before[title]}
    for title, rule in before.items():
        if after.get(title) == rule:
            if records_after.get(title) != records_before.get(title):
                problems.append(f"{title}: an unchanged rule keeps its provenance record as it was")
            if evolved_after.get(title) != evolved_before.get(title):
                problems.append(f"{title}: an unchanged rule keeps its dev_evolved_rules entry as it was")
            if title in resolving:
                problems.append(f"{title}: an adjudication names it as the existing rule, but the rule is unchanged")
            continue
        edited = title in after
        winners = resolving.get(title, [])
        if not winners:
            problems.append(f"{title}: {'edited' if edited else 'removed'} without an adjudication; a rule already in the corpus leaves or changes only through a measured conflict it lost")
            continue
        if len(winners) > 1:
            problems.append(f"{title}: resolved by {len(winners)} adjudications; one conflict, one record")
        new_rule = winners[0].get("new_rule")
        if edited and new_rule != title:
            problems.append(f"{title}: edited in place, so its adjudication's new_rule is {title!r}, not {new_rule!r}")
        if not edited and new_rule == title:
            problems.append(f"{title}: removed, so its adjudication's new_rule is the rule that replaces it, not {title!r} itself")
        if edited and (title in records_after or title not in evolved_after):
            problems.append(f"{title}: an edited rule is dev-evolved: a provenance.{EVOLVED} entry and no train provenance record")
    for title in after:
        if title in before:
            continue
        if title in records_after:
            problems.append(f"{title}: a rule the change adds has no train provenance record; it is labelled in provenance.{EVOLVED}")
        if title not in evolved_after:
            problems.append(f"{title}: added without a provenance.{EVOLVED} entry (an unlabelled rule; dev-derived text is only accepted labelled)")
    for record in appended:
        existing_rule, new_rule = record.get("existing_rule"), record.get("new_rule")
        if record.get("question_set") != ADJUDICATION_QUESTION_SET:
            problems.append(f"adjudication {existing_rule} -> {new_rule}: new records name the current evolution input {ADJUDICATION_QUESTION_SET}")
        if existing_rule not in before:
            problems.append(f"adjudication {existing_rule} -> {new_rule}: {existing_rule!r} is not a rule of the parent's corpus")
        if new_rule not in brought:
            problems.append(f"adjudication {existing_rule} -> {new_rule}: {new_rule!r} is not a rule this change brings")
    parent_generation, child_generation = parent.get("generation") or {}, child.get("generation") or {}
    for key in sorted(set(parent_generation) | set(child_generation)):
        if key not in MANAGED_GENERATION and parent_generation.get(key) != child_generation.get(key):
            problems.append(f"generation.{key}: a rules change leaves it as the parent has it (it writes only {list(MANAGED_GENERATION)})")
    if "shared_triggers" in parent_generation or "shared_triggers" in child_generation:
        expected = shared_triggers_after(parent_generation.get("shared_triggers"), child.get("rules") or [])
        if child_generation.get("shared_triggers") != expected:
            problems.append(f"generation.shared_triggers: a rules change keeps the declared sharing that still holds and declares none, so it is {expected!r}")
    if parent.get("version") != child.get("version"):
        problems.append(f"version: a rules change keeps {parent.get('version')!r}")
    if question_ids is not None:
        known = set(question_ids)
        named: set[str] = set()
        for title in after:
            item = evolved_after.get(title)
            if item is not None and item != evolved_before.get(title) and isinstance(item.get("dossier_qids"), list):
                named.update(str(value) for value in item["dossier_qids"])
        for record in appended:
            for key in ("applicable_qids", "follows_new_qids", "follows_existing_qids"):
                if isinstance(record.get(key), list):
                    named.update(str(value) for value in record[key])
        outside = sorted(named - known)
        if outside:
            problems.append(f"question ids outside the supplied evolution questions: {outside[:8]}")
    return problems


def change_summary(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    """What a child corpus changed against its parent, for the gate's report.

    Args:
        parent (Mapping[str, Any]): The parent's corpus.
        child (Mapping[str, Any]): The child's corpus.

    Returns:
        dict[str, Any]: ``added`` (titles), ``edited`` and ``removed`` (titles), ``adjudications`` (``existing``,
            ``new``, ``follows_new``, ``follows_existing``, ``applicable`` for each appended record), ``rules`` (the
            child's count) and ``dev_evolved`` (its dev-evolved count).
    """
    before, after = _by_title(parent), _by_title(child)
    recorded = _labelled(parent.get("provenance") or {}, ADJUDICATIONS)
    appended = _labelled(child.get("provenance") or {}, ADJUDICATIONS)[len(recorded):]
    return {
        "added": [title for title in after if title not in before],
        "edited": [title for title in after if title in before and after[title] != before[title]],
        "removed": [title for title in before if title not in after],
        "adjudications": [
            {
                "existing": record.get("existing_rule"),
                "new": record.get("new_rule"),
                "follows_new": len(record.get("follows_new_qids") or []),
                "follows_existing": len(record.get("follows_existing_qids") or []),
                "applicable": len(record.get("applicable_qids") or []),
            }
            for record in appended
            if isinstance(record, Mapping)
        ],
        "rules": len(after),
        "dev_evolved": len(_evolved_by_title(child)),
    }


def rules_digest(rules: Iterable[Mapping[str, Any]]) -> str:
    """One identity for a set of rules as a workspace stores them: the records, sorted by title.

    Args:
        rules (Iterable[Mapping[str, Any]]): Rule records (corpus rules, or a workspace's global rules).

    Returns:
        str: ``sha256:`` + the hex digest of their canonical JSON; equal exactly when the records are.
    """
    records = sorted(({key: rule.get(key) for key in RULE_FIELDS} for rule in rules), key=lambda record: str(record["title"]))
    return "sha256:" + sha256_hex(canon(records))


def evolved_origins(store: Any, corpus: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The proposal and the outcome that brought each dev-evolved rule of a corpus into BASELINE.

    An entry names the release its rule was proposed on; the first ``promote`` decision whose release carries the
    same entry introduced it, and names the proposal (``target``) and the outcome it promoted on (``outcome``,
    ``None`` for a person's override without one).

    Args:
        store (Store): The store whose decisions and releases are read.
        corpus (Mapping[str, Any]): The corpus whose dev-evolved rules are looked up.

    Returns:
        dict[str, dict[str, Any]]: Title to ``{"parent_release", "proposal", "outcome"}``; ``proposal`` and
            ``outcome`` are ``None`` for a rule no promotion has carried yet.
    """
    wanted = {(item["title"], item["parent_release"]) for item in _evolved_by_title(corpus).values()}
    origins: dict[str, dict[str, Any]] = {title: {"parent_release": release, "proposal": None, "outcome": None} for title, release in wanted}
    corpora: dict[str, dict[str, Any] | None] = {}
    decisions = sorted((decision for decision in store.decisions() if decision.get("kind") == "promote"), key=lambda decision: (float(decision.get("created_at") or 0.0), str(decision.get("id"))))
    for decision in decisions:
        rid = decision.get("release")
        if not isinstance(rid, str):
            continue
        if rid not in corpora:
            corpora[rid] = _release_corpus(store, rid)
        carried = corpora[rid]
        if carried is None:
            continue
        for title, item in _evolved_by_title(carried).items():
            origin = origins.get(title)
            if origin is not None and origin["proposal"] is None and item.get("parent_release") == origin["parent_release"]:
                origin.update(proposal=decision.get("target"), outcome=decision.get("outcome"))
    return origins


def _release_corpus(store: Any, rid: str) -> dict[str, Any] | None:
    """The corpus a stored release carries, or ``None`` when it carries none or cannot be read.

    A closure/1 release carries its ``rules.general`` row; a closure/2 release's corpus is reconstructed exactly
    from its ordered rule versions in the store and its corpus metadata (:func:`entity_model.reconstruct_corpus`).
    """
    try:
        from .entity_model import CLOSURE_V2, RULE_ENTITY, reconstruct_corpus

        release = store.release(rid).data
        if release.get("rules") == CLOSURE_V2:
            return reconstruct_corpus(release["content_refs"]["rules"], store.typed_rows(RULE_ENTITY), policy=False)
        key = release["refs"].get(E.RULES_ENTRY)
        if not key:
            return None
        return json.loads(store.object(E.RULES_ENTRY, key[:16]).data["row"]["manifest"]["meta"]["text"])
    except Exception:  # noqa: BLE001 - an unreadable release names no origin; verify reports the store itself
        return None


# ---------------------------------------------------------------------------------------
# The corpus of a release


def _corpus_of(release: Any) -> tuple[str, dict[str, Any]]:
    """The id and the parsed corpus of a loaded release, or of a ``baseline``/``r…``/``p…`` reference.

    ``Release.rules_general`` reads a closure/1 ``rules.general`` row or reconstructs a closure/2 release's corpus
    from its rule versions, so both formats yield the same ordered corpus.
    """
    if isinstance(release, str):
        from .runtime import current, load_release

        release = current() if release == "baseline" else load_release(release)
    try:
        corpus = release.rules_general()
    except KeyError as error:
        raise RuleCorpusError(f"{error.args[0] if error.args else error}; only a release that carries the corpus is materialised") from error
    return str(release.id), corpus


def pinned_corpus(release: Any = None) -> dict[str, Any]:
    """Read a detached rule corpus from the pinned release, without a Git mirror.

    Args:
        release (Release | str | None): Exact loaded release or release reference.
            Omitted means the process's verified execution snapshot.

    Returns:
        dict[str, Any]: The complete ordered corpus, provenance and generation data: a closure/1 release's
            ``rules.general`` row, or a closure/2 release's exact reconstruction from its rule versions.

    Raises:
        RuleCorpusError: The release has no corpus or its payload is malformed.
    """
    from .runtime import current

    rid, corpus = _corpus_of(current() if release is None else release)
    if not isinstance(corpus, dict) or not isinstance(corpus.get("rules"), list):
        raise RuleCorpusError(f"{rid} has a malformed rules.general corpus")
    return json.loads(json.dumps(corpus, ensure_ascii=False))


def corpus_lines(release: Any, store: Any) -> list[str]:
    """The corpus of a release as ``prompts rules`` prints it: its make-up, each dev-evolved rule's origin, the
    adjudications, and any labelling problem.

    Args:
        release (Release): The loaded release.
        store (Store): The store whose decisions name the origins.

    Returns:
        list[str]: Lines to print.

    Raises:
        RuleCorpusError: If the release carries no corpus.
    """
    rid, corpus = _corpus_of(release)
    provenance = corpus.get("provenance") or {}
    titles = list(_by_title(corpus))
    records, evolved = _records(provenance), _evolved_by_title(corpus)
    adjudications = _labelled(provenance, ADJUDICATIONS)
    measured = [title for title in titles if title not in records and title not in evolved]
    lines = [f"{E.RULES_ENTRY} of {rid}: {len(titles)} rules — {len(records)} train-grounded (a provenance record each), {len(measured)} measured without a record, {len(evolved)} dev-evolved; {len(adjudications)} adjudication(s)"]
    origins = evolved_origins(store, corpus) if evolved else {}
    for title, item in evolved.items():
        origin = origins.get(title) or {}
        promoted = f"promoted by {origin['proposal']}" + (f" on outcome {origin['outcome']}" if origin.get("outcome") else " (no outcome)") if origin.get("proposal") else "not promoted yet"
        ids = list(item.get("dossier_qids") or [])
        lines.append(f"  evolved {title}: proposed on {item.get('parent_release')}, from {len(ids)} dossier question(s) {', '.join(ids[:6])}{', …' if len(ids) > 6 else ''}; {promoted}")
    for record in adjudications:
        if isinstance(record, Mapping):
            lines.append(
                f"  adjudicated {record.get('new_rule')} over {record.get('existing_rule')} on {record.get('parent_release')}: "
                f"{len(record.get('follows_new_qids') or [])}:{len(record.get('follows_existing_qids') or [])} of {len(record.get('applicable_qids') or [])} dev questions — {record.get('method')}"
            )
    problems = corpus_problems(corpus)
    lines.extend(f"  PROBLEM {problem}" for problem in problems)
    if not problems:
        lines.append("  labelled provenance consistent")
    return lines


# ---------------------------------------------------------------------------------------
# Materialisation into the knowledge workspaces


def knowledge_workspaces(dbs_dir: str | os.PathLike[str], dbs: Sequence[str] | None = None) -> list[Path]:
    """The knowledge workspaces under a root: every ``<db>/`` holding ``<db>.db``, or the named ones.

    Args:
        dbs_dir (str | os.PathLike[str]): The root, one directory per database (``$RUBIKSQL_BIRD_DBS_DIR``).
        dbs (Sequence[str] | None): Databases to take; every workspace under the root when ``None``.

    Returns:
        list[Path]: Workspace directories, sorted by database when ``dbs`` is ``None``, else in its order.

    Raises:
        RuleCorpusError: If the root holds no workspace, or a named database has none.
    """
    root = Path(dbs_dir)
    if dbs is not None:
        missing = [db for db in dbs if not (root / db / f"{db}.db").is_file()]
        if missing:
            raise RuleCorpusError(f"no knowledge workspace under {root} for {', '.join(missing)}")
        return [root / db for db in dbs]
    found = sorted(path for path in root.iterdir() if path.is_dir() and (path / f"{path.name}.db").is_file()) if root.is_dir() else []
    if not found:
        raise RuleCorpusError(f"no knowledge workspace under {root}")
    return found


def _target(rid: str, corpus: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The corpus's rules by title, each checked as a record a workspace may store."""
    rules = corpus.get("rules") or []
    problems = [problem for index, rule in enumerate(rules) for problem in _rule_problems(rule, f"{rid} rules[{index}]")]
    titles = [rule.get("title") for rule in rules if isinstance(rule, Mapping)]
    repeated = sorted({str(title) for title in titles if titles.count(title) > 1})
    if repeated:
        problems.append(f"{rid}: titles repeat: {repeated}")
    if problems:
        raise RuleCorpusError("; ".join(problems))
    return {rule["title"]: {key: rule[key] for key in RULE_FIELDS} for rule in rules}


@dataclass(frozen=True)
class WorkspaceRules:
    """The rules one knowledge workspace stores, read from its SQLite file without opening HeavenBase.

    Attributes:
        path (Path): The workspace directory, ``<dbs_dir>/<db_id>``.
        db_id (str): The database, the directory's name.
        rules (Mapping[str, Mapping[str, Any]]): The global rules, title to record (:data:`RULE_FIELDS`).
        object_ids (Mapping[str, str]): The global rules' object ids, by title.
        database_titles (frozenset[str]): Titles of the database's own rules (``scope`` ``database``), which no
            corpus change touches.
        unvectored (tuple[str, ...]): Titles of rows without both inline vectors, which rule retrieval cannot rank.
        record (Mapping[str, Any] | None): The :data:`APPLIED_FILE` record, or ``None``.
    """

    path: Path
    db_id: str
    rules: Mapping[str, Mapping[str, Any]]
    object_ids: Mapping[str, str]
    database_titles: frozenset[str]
    unvectored: tuple[str, ...]
    record: Mapping[str, Any] | None

    @property
    def digest(self) -> str:
        """The :func:`rules_digest` of the global rules.

        Args:
            None.

        Returns:
            str: ``sha256:<hex>``.
        """
        return rules_digest(self.rules.values())

    @classmethod
    def read(cls, path: str | os.PathLike[str]) -> "WorkspaceRules":
        """Read one workspace's rules read-only.

        Args:
            path (str | os.PathLike[str]): The workspace directory; its name is the database id and its SQLite file
                ``<db_id>.db``.

        Returns:
            WorkspaceRules: The stored rules; a workspace without a rule table stores none.

        Raises:
            RuleCorpusError: If the directory holds no ``<db_id>.db``: materialising rules builds no knowledge.
        """
        root = Path(path)
        db_id = root.name
        database = root / f"{db_id}.db"
        if not database.is_file():
            raise RuleCorpusError(f"no knowledge workspace at {root}: {database.name} is missing (build knowledge first; rules are applied to a built workspace)")
        rules: dict[str, dict[str, Any]] = {}
        object_ids: dict[str, str] = {}
        own: set[str] = set()
        unvectored: list[str] = []
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        try:
            if connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'rule_entity'").fetchone():
                rows = connection.execute(
                    'SELECT object_id, scope, title, "desc", tags, triggers, example, '
                    "title_vec IS NOT NULL AND title_vec NOT IN ('', '[]') AND desc_vec IS NOT NULL AND desc_vec NOT IN ('', '[]') "
                    "FROM rule_entity WHERE db_id = ? ORDER BY object_id",
                    (db_id,),
                ).fetchall()
                for object_id, scope, title, desc, tags, triggers, example, vectored in rows:
                    if not vectored:
                        unvectored.append(str(title))
                    if scope != "global":
                        own.add(str(title))
                        continue
                    object_ids[str(title)] = str(object_id)
                    rules[str(title)] = {
                        "scope": scope,
                        "title": title,
                        "desc": desc,
                        "tags": json.loads(tags) if tags else [],
                        "triggers": json.loads(triggers) if triggers else [],
                        "example": example or "",
                    }
        finally:
            connection.close()
        record_path = root / APPLIED_FILE
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else None
        return cls(root, db_id, rules, object_ids, frozenset(own), tuple(sorted(unvectored)), record)

    def vectors(self, title: str) -> tuple[list[float], list[float]] | None:
        """The stored inline vectors of one of the workspace's global rules.

        Args:
            title (str): A title of :attr:`rules`.

        Returns:
            tuple[list[float], list[float]] | None: ``(title_vec, desc_vec)``, or ``None`` when the row is gone or lacks
                either vector.
        """
        object_id = self.object_ids.get(title)
        if object_id is None or title in self.unvectored:
            return None
        connection = sqlite3.connect(f"file:{self.path / f'{self.db_id}.db'}?mode=ro", uri=True)
        try:
            row = connection.execute("SELECT title_vec, desc_vec FROM rule_entity WHERE object_id = ?", (object_id,)).fetchone()
        finally:
            connection.close()
        if row is None or not row[0] or not row[1]:
            return None
        title_vec, desc_vec = (json.loads(value) if isinstance(value, str) else value for value in row)
        return (title_vec, desc_vec) if title_vec and desc_vec else None

    def plan(self, target: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], list[str], list[str]]:
        """What materialising a corpus changes here.

        Args:
            target (Mapping[str, Mapping[str, Any]]): The corpus's rules by title.

        Returns:
            tuple[list[str], list[str], list[str]]: Titles to add, to rewrite (stored with other fields, or without both
                vectors) and to delete (global rules the corpus does not carry).
        """
        add = [title for title in target if title not in self.rules]
        rewrite = [title for title in target if title in self.rules and (dict(self.rules[title]) != dict(target[title]) or title in self.unvectored)]
        delete = [title for title in self.rules if title not in target]
        return add, rewrite, delete


@dataclass(frozen=True)
class AppliedRules:
    """What :func:`apply_rules` did to one workspace.

    Attributes:
        db_id (str): The database.
        path (Path): The workspace directory.
        release (str): The release whose corpus the workspace now carries.
        digest (str): The :func:`rules_digest` of that corpus's rules.
        added (tuple[str, ...]): Rules written that the workspace lacked.
        rewritten (tuple[str, ...]): Rules rewritten because a stored field differed.
        deleted (tuple[str, ...]): Global rules deleted because the corpus does not carry them.
        unchanged (int): Rules already stored as the corpus has them.
    """

    db_id: str
    path: Path
    release: str
    digest: str
    added: tuple[str, ...]
    rewritten: tuple[str, ...]
    deleted: tuple[str, ...]
    unchanged: int

    @property
    def changed(self) -> bool:
        """Whether a rule row was written or deleted.

        Args:
            None.

        Returns:
            bool: ``True`` when the knowledge fingerprint of the workspace moved.
        """
        return bool(self.added or self.rewritten or self.deleted)


def _write_record(state: WorkspaceRules, rid: str, digest: str, count: int) -> None:
    """Write the workspace's :data:`APPLIED_FILE` atomically (a temporary file, then a rename)."""
    record = {"schema": SCHEMA_APPLIED, "db_id": state.db_id, "release": rid, "rules_digest": digest, "rules": count}
    if state.record == record:
        return
    temporary = state.path / f"{APPLIED_FILE}.tmp"
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, state.path / APPLIED_FILE)


def _stored_vectors(titles: Iterable[str], target: Mapping[str, Mapping[str, Any]], states: Sequence[WorkspaceRules]) -> dict[str, tuple[list[float], list[float]]]:
    """Vectors a workspace already stores for a rule whose embedded texts the corpus leaves unchanged.

    A rule's vectors embed its title and its ``example`` + ``desc`` (:func:`rubiksql_hb.embedding.embed_title`,
    :func:`~rubiksql_hb.embedding.embed_rule_desc`); a rule rewritten only in its tags, triggers or scope, or added to a
    workspace while another of the same root stores it, keeps the stored pair instead of being embedded again.
    """
    found: dict[str, tuple[list[float], list[float]]] = {}
    for title in titles:
        wanted = target[title]
        for state in states:
            stored = state.rules.get(title)
            if stored is None or stored.get("desc") != wanted["desc"] or (stored.get("example") or "") != (wanted["example"] or ""):
                continue
            pair = state.vectors(title)
            if pair is not None:
                found[title] = pair
                break
    return found


def apply_rules(release: Any, workspaces: Iterable[str | os.PathLike[str]], *, workers: int = 4, embed_workers: int = 1) -> list[AppliedRules]:
    """Materialise a release's general rule corpus into knowledge workspaces.

    Each workspace's global rules become exactly the corpus's: a rule it lacks is added, a rule stored with another
    field (or without both vectors) is rewritten, a global rule the corpus does not carry is deleted. The database's own rules (``scope``
    ``database``) are never touched, and a corpus title that is one of them refuses the whole call before anything is
    written. Rows go in whole — HeavenBase's upsert replaces a row — with both vectors supplied here: a rule whose title
    and ``example`` + ``desc`` some workspace of the call already stores with its vectors keeps that pair, and every
    other rule is embedded once for every workspace, through the DashScope embedding of :mod:`rubiksql_hb.embedding`
    (the offline suite stubs it), ``embed_workers`` calls at a time. A workspace already carrying the corpus is not
    opened. Every workspace records the release and the rules digest it now carries in :data:`APPLIED_FILE`. The
    knowledge fingerprint of a changed workspace moves with its rule rows, so the candidate and judge cache keys follow,
    and applying the parent's corpus again restores the fingerprint byte for byte.

    Args:
        release (Release | str): The loaded release, or ``baseline``, ``r<16hex>`` or ``p<16hex>`` to load.
        workspaces (Iterable[str | os.PathLike[str]]): Workspace directories, ``<dbs_dir>/<db_id>``.
        workers (int): Workspaces written at once (at least 1).
        embed_workers (int): Embedding calls in flight at once (at least 1).

    Returns:
        list[AppliedRules]: One per workspace, in order.

    Raises:
        RuleCorpusError: If the release carries no corpus, a corpus rule is malformed, a workspace is missing or listed
            twice, a corpus title is one of a database's own rules, or a workspace does not read back as the corpus after
            writing. Writes require this release's running ``general_rules`` pipeline ledger record.
    """
    rid, corpus = _corpus_of(release)
    target = _target(rid, corpus)
    digest = rules_digest(target.values())
    states = [WorkspaceRules.read(path) for path in workspaces]
    for state in states:
        try:
            ledger = json.loads((state.path / "knowledge_ledger.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            ledger = {}
        record = (ledger.get("stages") or {}).get("general_rules") or {}
        from .knowledge import LEDGER_SCHEMA
        if (ledger.get("schema") != LEDGER_SCHEMA or ledger.get("db_id") != state.db_id
                or record.get("status") != "running" or record.get("release") != rid):
            raise RuleCorpusError("release corpus writes require a running general_rules pipeline stage; "
                                  "use run_official_pipeline.py --rebuild general_rules")
    paths = [state.path.resolve() for state in states]
    if len(set(paths)) != len(paths):
        raise RuleCorpusError(f"a workspace is listed more than once: {sorted({str(path) for path in paths if paths.count(path) > 1})}")
    collisions = [f"{state.db_id}: {title}" for state in states for title in target if title in state.database_titles]
    if collisions:
        raise RuleCorpusError(f"{rid}: corpus rule(s) share a title with a database's own rule, whose row the write would replace: {collisions}; rename the corpus rule")
    plans = [(state, *state.plan(target)) for state in states]
    needed = sorted({title for _state, add, rewrite, _delete in plans for title in (*add, *rewrite)})
    vectors = _stored_vectors(needed, target, states)
    missing = [title for title in needed if title not in vectors]
    if missing:
        from ..embedding import embed_many, rule_with_example, title_text

        texts = [text for title in missing for text in (title_text(title), rule_with_example(target[title]["desc"], target[title]["example"]))]
        embedded = embed_many(texts, workers=embed_workers)
        for index, title in enumerate(missing):
            vectors[title] = (embedded[2 * index], embedded[2 * index + 1])

    def write(plan: tuple[WorkspaceRules, list[str], list[str], list[str]]) -> AppliedRules:
        state, add, rewrite, delete = plan
        if add or rewrite or delete:
            from ..schema import RuleEntity
            from ..workspace import open_workspace

            workspace = open_workspace(state.path, entities=(RuleEntity,))
            if delete:
                workspace.delete_many([(RuleEntity, state.object_ids[title]) for title in delete])
            rows = []
            for title in (*add, *rewrite):
                row = RuleEntity.from_dict({"db_id": state.db_id, **target[title]}).to_dict()
                row["title_vec"], row["desc_vec"] = vectors[title]
                rows.append(row)
            if rows:
                workspace.upsert_many(RuleEntity, rows)
            after = WorkspaceRules.read(state.path)
            if after.digest != digest or set(after.unvectored) & set(target):
                raise RuleCorpusError(f"{state.db_id}: after writing, the workspace's global rules are not {rid}'s ({len(after.rules)} stored, {len(target)} in the corpus, {len(set(after.unvectored) & set(target))} without vectors)")
            state = after
        _write_record(state, rid, digest, len(target))
        return AppliedRules(state.db_id, state.path, rid, digest, tuple(add), tuple(rewrite), tuple(delete), len(target) - len(add) - len(rewrite))

    pending = [plan for plan in plans if any(plan[1:])]
    results: dict[int, AppliedRules] = {}
    if len(pending) > 1 and workers > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = {index: pool.submit(write, plan) for index, plan in enumerate(plans) if any(plan[1:])}
            for index, future in futures.items():
                results[index] = future.result()
    for index, plan in enumerate(plans):
        if index not in results:
            results[index] = write(plan)
    return [results[index] for index in range(len(plans))]


def rules_problems(release: Any, workspaces: Iterable[str | os.PathLike[str]]) -> list[str]:
    """Check, read-only, that knowledge workspaces carry a release's general rule corpus.

    A workspace carries the corpus when its global rules are exactly the corpus's rules, every one of them with both
    inline vectors, and no corpus title is one of the database's own rules. Nothing is written and HeavenBase is not
    opened, so an evaluation can check its workspaces before it spends.

    Args:
        release (Release | str): The loaded release, or ``baseline``, ``r<16hex>`` or ``p<16hex>`` to load.
        workspaces (Iterable[str | os.PathLike[str]]): Workspace directories, ``<dbs_dir>/<db_id>``.

    Returns:
        list[str]: One line per workspace that does not carry the corpus; empty when all do.

    Raises:
        RuleCorpusError: If the release carries no corpus or a corpus rule is malformed.
    """
    rid, corpus = _corpus_of(release)
    target = _target(rid, corpus)
    problems: list[str] = []
    for path in workspaces:
        try:
            state = WorkspaceRules.read(path)
        except RuleCorpusError as error:
            problems.append(str(error))
            continue
        add, rewrite, delete = state.plan(target)
        parts = [f"{label} {len(titles)} ({', '.join(titles[:4])}{', …' if len(titles) > 4 else ''})" for label, titles in (("lacks", add), ("stores differently", rewrite), ("also stores", delete)) if titles]
        clash = sorted(set(target) & state.database_titles)
        if clash:
            parts.append(f"has database rules titled like corpus rules {clash}")
        unvectored = sorted(set(state.unvectored) & set(target))
        if unvectored:
            parts.append(f"stores {len(unvectored)} corpus rule(s) without vectors")
        if parts:
            problems.append(f"{state.db_id}: " + "; ".join(parts) + f" — it does not carry {rid}'s general rules")
    return problems
