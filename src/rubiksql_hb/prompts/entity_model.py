"""Typed entity-model contract: canonical Tool, Rule and parameter-set versions in closure/2 releases.

This module is the single owner of the typed reference format, the canonical identity of rule and
parameter-set versions, the exact partition and reconstruction of a rule corpus, and the closure/2
release reference block. It imports no HeavenBase, provider or runtime module, so the authority,
the inert store projection, the snapshot verifier and the runtime loader share it. The Tool row
contract lives with the other answering-tool identities in :mod:`rubiksql_hb.hb_tool_entities`.

A typed reference is exactly ``{"entity", "object_id", "version", "digest"}``: the stored entity
id, the row identity, the first 16 hex of the digest and the full 64-hex canonical digest. Every
digest hashes canonical JSON (:func:`canon`) of a body that starts with the row format, so two
formats can never collide. A closure/2 release (``rules`` ``rubiksql.closure/2``) keeps native
prompt references in ``refs`` and carries its typed references in ``content_refs``:

``{"schema": "rubiksql.release-refs/1", "tools": [tool refs in serve order],
"loader": {"modules": [...], "serializer": {...}}, "parameters": knob-set ref,
"rules": {"items": [rule refs in corpus order], "corpus": {...}, "knowledge_digest": "sha256:..."}}``.

HeavenBase stores Json fields with sorted member names; JSON object member order is recorded
explicitly as order skeletons (lists survive storage), owned by the row or release that owns the
values, so a corpus reconstructs to the exact original bytes.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from . import entries as E
from . import rules as R
from .canon import canon, sha256_hex

__all__ = [
    "CARRIERS",
    "CLOSURE_V2",
    "FORMAT_PARAMETERS",
    "FORMAT_RULE",
    "FORMAT_TOOL",
    "MIGRATION_KIND",
    "PARAMETER_ENTITY",
    "PARAMETER_NAME",
    "RULE_ENTITY",
    "SCHEMA_ENTITY_GATE",
    "SCHEMA_RELEASE_REFS",
    "SCHEMA_RELEASE_V2",
    "TOOL_ENTITY",
    "TYPED_ENTITY_IDS",
    "PARAMETER_PLACEHOLDERS",
    "RULE_PLACEHOLDERS",
    "EntityModelError",
    "TypedClosure",
    "apply_order",
    "change_kind",
    "check_ref",
    "check_release_document",
    "check_release_refs",
    "closure_changes",
    "closure_view",
    "corpus_from_text",
    "is_canonical",
    "legacy_texts",
    "order_of",
    "parameter_row",
    "parameter_values",
    "partition_corpus",
    "reconstruct_corpus",
    "resolve_closure",
    "rule_row",
    "stored_json",
    "typed_ref",
    "values_from_text",
    "verify_parameter_row",
    "verify_rule_row",
]

#: Row format of a canonical Tool version (``tool-entity``).
FORMAT_TOOL = "rubiksql.tool/1"
#: Row format of a canonical rule version (``general-rule-entity`` with ``format`` set).
FORMAT_RULE = "rubiksql.rule/1"
#: Row format of a canonical parameter set (``knob-set-entity`` with ``format`` set).
FORMAT_PARAMETERS = "rubiksql.parameters/1"
#: Release document schema of a closure/2 release (store projection and snapshots).
SCHEMA_RELEASE_V2 = "rubiksql.release/2"
#: Closure rules of a release whose tools, rules and parameters are typed references.
CLOSURE_V2 = "rubiksql.closure/2"
#: Schema of a closure/2 release's ``content_refs`` block.
SCHEMA_RELEASE_REFS = "rubiksql.release-refs/1"
#: Gate receipt schema of the representation-only entity-model migration.
SCHEMA_ENTITY_GATE = "rubiksql.entity-model-gate/1"
#: Decision kind and proposal origin of the representation-only entity-model migration.
MIGRATION_KIND = "entity-model"
#: Logical name of the one canonical parameter set a release references.
PARAMETER_NAME = "knobs"

TOOL_ENTITY = "tool-entity"
RULE_ENTITY = "general-rule-entity"
PARAMETER_ENTITY = "knob-set-entity"
#: Typed entity ids in reference-section order.
TYPED_ENTITY_IDS = (TOOL_ENTITY, RULE_ENTITY, PARAMETER_ENTITY)
#: The historical prompt carriers a closure/2 release never references.
CARRIERS = (E.RULES_ENTRY, E.KNOBS_ENTRY, "tools.answering")

_PREFIX = {TOOL_ENTITY: "t", RULE_ENTITY: "g", PARAMETER_ENTITY: "k"}
_FORMAT = {TOOL_ENTITY: FORMAT_TOOL, RULE_ENTITY: FORMAT_RULE, PARAMETER_ENTITY: FORMAT_PARAMETERS}
_DIGEST = re.compile(r"[0-9a-f]{64}")
_KNOWLEDGE = re.compile(r"sha256:[0-9a-f]{64}")
_REF_KEYS = frozenset({"entity", "object_id", "version", "digest"})
_LABELS = (R.EVOLVED, R.ADJUDICATIONS)
_CORPUS_KEYS = frozenset({"version", "generation", "rules", "provenance"})
#: Projection-only columns a canonical rule version stores empty (HB keeps them NOT NULL).
RULE_PLACEHOLDERS: Mapping[str, Any] = {"corpus_version": "", "content_key": "", "source_prompt_id": "", "status": "",
                                        "adjudications": [], "amendments": [], "releases": [], "store_commit": ""}
#: Projection-only columns a canonical parameter set stores empty.
PARAMETER_PLACEHOLDERS: Mapping[str, Any] = {"content_key": "", "source_prompt_id": "", "status": "",
                                             "declarations": {}, "releases": [], "store_commit": ""}


class EntityModelError(ValueError):
    """A typed row, reference, corpus partition or closure/2 release is invalid."""


# ---------------------------------------------------------------------------------------
# References


def _digest(value: Any) -> str:
    return sha256_hex(canon(value))


def typed_ref(entity: str, row: Mapping[str, Any]) -> dict[str, str]:
    """Return the exact typed reference of one canonical row.

    Args:
        entity: Stored entity id (``tool-entity``, ``general-rule-entity`` or ``knob-set-entity``).
        row: The canonical row.

    Returns:
        ``{"entity", "object_id", "version", "digest"}``.
    """
    return check_ref({"entity": entity, "object_id": row["object_id"], "version": row["version"], "digest": row["digest"]},
                     entity)


def check_ref(value: Any, entity: str | None = None) -> dict[str, str]:
    """Validate one typed reference's exact shape and internal identity.

    Args:
        value: Candidate reference.
        entity: Required stored entity id, or None for any typed entity.

    Returns:
        A detached copy of the reference.

    Raises:
        EntityModelError: Missing/extra keys, unknown entity, or inconsistent id, version and digest.
    """
    if not isinstance(value, Mapping) or set(value) != _REF_KEYS:
        raise EntityModelError(f"typed reference must have exactly {sorted(_REF_KEYS)}")
    ref = {key: value[key] for key in sorted(_REF_KEYS)}
    if ref["entity"] not in _PREFIX or (entity is not None and ref["entity"] != entity):
        raise EntityModelError(f"typed reference names entity {ref['entity']!r}, expected {entity or 'a typed entity'}")
    digest = ref["digest"]
    if (not isinstance(digest, str) or not _DIGEST.fullmatch(digest) or ref["version"] != digest[:16]
            or ref["object_id"] != _PREFIX[ref["entity"]] + digest[:16]):
        raise EntityModelError(f"typed reference {ref['object_id']!r} has an inconsistent id, version or digest")
    return ref


def is_canonical(entity: str, row: Mapping[str, Any]) -> bool:
    """Classify a stored GeneralRule/KnobSet/Tool row.

    Args:
        entity: Stored entity id.
        row: Stored row.

    Returns:
        True for a canonical version, False for a historical projection row (empty ``format``).

    Raises:
        EntityModelError: The row names an unknown format.
    """
    value = row.get("format")
    if value in (None, "") and entity != TOOL_ENTITY:
        return False
    if value != _FORMAT[entity]:
        raise EntityModelError(f"{entity} row {row.get('object_id')!r} has unknown format {value!r}")
    return True


# ---------------------------------------------------------------------------------------
# JSON member order


def order_of(value: Any) -> Any:
    """Record the JSON object member order of a value as a storage-stable skeleton.

    Args:
        value: Parsed JSON value.

    Returns:
        None for a scalar or for a list holding no object; ``{"keys": [...], "children": {...}}`` for
        an object (children only for members whose own skeleton is not None); ``{"items": [...]}`` for
        a list holding at least one object.
    """
    if isinstance(value, Mapping):
        children = {key: skeleton for key, item in value.items() if (skeleton := order_of(item)) is not None}
        return {"keys": list(value), **({"children": children} if children else {})}
    if isinstance(value, (list, tuple)):
        items = [order_of(item) for item in value]
        return {"items": items} if any(item is not None for item in items) else None
    return None


def apply_order(value: Any, skeleton: Any) -> Any:
    """Restore the recorded member order of a value read back from sorted storage.

    Args:
        value: Parsed JSON value with the same members.
        skeleton: :func:`order_of` result, or None to keep the value as stored.

    Returns:
        A detached value whose objects follow the skeleton's member order.

    Raises:
        EntityModelError: The skeleton is malformed or does not describe exactly this value.
    """
    if skeleton is None:
        return deepcopy(value)
    if not isinstance(skeleton, Mapping):
        raise EntityModelError("JSON layout skeleton must be an object")
    if set(skeleton) <= {"keys", "children"} and "keys" in skeleton:
        keys, children = skeleton["keys"], skeleton.get("children", {})
        if (not isinstance(value, Mapping) or not isinstance(keys, list) or not isinstance(children, Mapping)
                or len(set(keys)) != len(keys) or set(keys) != set(value) or not set(children) <= set(keys)):
            raise EntityModelError("JSON layout does not describe the stored object members")
        return {key: apply_order(value[key], children.get(key)) for key in keys}
    if set(skeleton) == {"items"}:
        items = skeleton["items"]
        if not isinstance(value, list) or not isinstance(items, list) or len(items) != len(value):
            raise EntityModelError("JSON layout does not describe the stored list")
        return [apply_order(item, layout) for item, layout in zip(value, items)]
    raise EntityModelError("JSON layout skeleton has unknown members")


# ---------------------------------------------------------------------------------------
# Rules


def _rule_content(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"format": FORMAT_RULE, "rule": {field: row[field] for field in R.RULE_FIELDS},
            "provenance": row.get("provenance"), "evolution": row.get("evolution"), "layout": row.get("layout")}


def rule_row(rule: Mapping[str, Any], *, provenance: Mapping[str, Any] | None, evolution: Mapping[str, Any] | None,
             layout: Mapping[str, Any], policy: bool = True) -> dict[str, Any]:
    """Build one canonical rule version.

    Args:
        rule: The six rule fields (any member order; the layout records the original one).
        provenance: The title's own train provenance record, or None.
        evolution: The title's own ``dev_evolved_rules`` entry, or None.
        layout: ``{"rule": [field order], "provenance": skeleton, "evolution": skeleton}``.
        policy: Also apply the current code's rule-field policy (see :func:`verify_rule_row`).

    Returns:
        A complete ``general-rule-entity`` row whose projection-only columns are empty.

    Raises:
        EntityModelError: The rule, records or layout are invalid.
    """
    problems = R._rule_problems(rule, "rule") if policy else []
    if problems:
        raise EntityModelError("; ".join(problems))
    row: dict[str, Any] = {**{field: deepcopy(rule[field]) for field in R.RULE_FIELDS},
                           "provenance": deepcopy(dict(provenance)) if provenance is not None else None,
                           "evolution": deepcopy(dict(evolution)) if evolution is not None else None,
                           "layout": deepcopy(dict(layout)), "format": FORMAT_RULE}
    digest = _digest(_rule_content(row))
    row.update({"object_id": _PREFIX[RULE_ENTITY] + digest[:16], "version": digest[:16], "digest": digest,
                **deepcopy(dict(RULE_PLACEHOLDERS))})
    verify_rule_row(row, policy=policy)
    return row


def verify_rule_row(row: Mapping[str, Any], *, policy: bool = True) -> None:
    """Verify a canonical rule version's content, layout, identity and empty projection columns.

    Integrity (format, layout, identity, placeholders) is always checked. ``policy`` adds the current
    code's rule-field rules; stored-history verifiers pass ``policy=False`` so a later code change
    cannot invalidate a retained version, while publication and runtime load keep the default.

    Args:
        row: Stored or candidate ``general-rule-entity`` row.
        policy: Also apply the current code's rule-field policy.

    Raises:
        EntityModelError: The row is not an exact canonical rule version.
    """
    try:
        if row.get("format") != FORMAT_RULE:
            raise EntityModelError(f"rule {row.get('object_id')!r} is not a {FORMAT_RULE} version")
        content = {field: row[field] for field in R.RULE_FIELDS}
        problems = R._rule_problems(content, f"rule {row.get('object_id')!r}") if policy else []
        if problems:
            raise EntityModelError("; ".join(problems))
        layout = row.get("layout")
        if (not isinstance(layout, Mapping) or set(layout) != {"rule", "provenance", "evolution"}
                or not isinstance(layout["rule"], list) or sorted(layout["rule"]) != sorted(R.RULE_FIELDS)):
            raise EntityModelError(f"rule {row.get('object_id')!r} has an invalid layout")
        for key in ("provenance", "evolution"):
            if row.get(key) is not None:
                if not isinstance(row[key], Mapping):
                    raise EntityModelError(f"rule {row.get('object_id')!r} {key} must be an object")
                # The skeleton must be exactly the one order_of records, so one record has one row identity.
                if layout[key] != order_of(apply_order(row[key], layout[key])):
                    raise EntityModelError(f"rule {row.get('object_id')!r} {key} layout is not canonical")
            elif layout[key] is not None:
                raise EntityModelError(f"rule {row.get('object_id')!r} has a layout for an absent {key}")
        if row.get("evolution") is not None and (set(row["evolution"]) != set(R.EVOLVED_FIELDS)
                                                 or row["evolution"].get("title") != row["title"]):
            raise EntityModelError(f"rule {row.get('object_id')!r} evolution record belongs to another title")
        digest = _digest(_rule_content(row))
        if (row.get("digest") != digest or row.get("version") != digest[:16]
                or row.get("object_id") != _PREFIX[RULE_ENTITY] + digest[:16]):
            raise EntityModelError(f"rule {row.get('object_id')!r} content identity differs")
        if any(row.get(key) != value for key, value in RULE_PLACEHOLDERS.items()):
            raise EntityModelError(f"rule {row.get('object_id')!r} carries projection-only values")
    except (KeyError, TypeError) as error:
        raise EntityModelError(f"rule {row.get('object_id')!r} is malformed: {error}") from error


def partition_corpus(corpus: Mapping[str, Any], *, policy: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Partition a whole rule corpus into rule versions and a release rules section without loss.

    Rule content, the title's own provenance record and its ``dev_evolved_rules`` entry belong to the
    rule version. ``version``, ``generation``, every other provenance member (adjudications, future or
    unmatched keys), the evolved-entry order and all member orders belong to the release section.

    Args:
        corpus: Parsed corpus with exactly ``version``, ``generation``, ``rules`` and ``provenance``.
        policy: Also apply the current corpus and rule-field policy; reads of retained history pass
            ``policy=False`` (integrity and the reconstruction proof only).

    Returns:
        ``(rule rows in corpus order, {"items", "corpus", "knowledge_digest"})``; reconstruction is
        proven equal, including the exact ``indent=2`` spelling, before returning.

    Raises:
        EntityModelError: The corpus is invalid or cannot be partitioned without loss.
    """
    if not isinstance(corpus, Mapping) or set(corpus) != _CORPUS_KEYS:
        raise EntityModelError("a rule corpus has exactly version, generation, rules and provenance")
    problems = R.corpus_problems(corpus) if policy else []
    if problems:
        raise EntityModelError("rule corpus is invalid: " + "; ".join(problems))
    rows, section = _partition(corpus, policy=policy)
    # Prove reconstruction from rows and metadata as sorted Json storage returns them.
    stored = {row["object_id"]: stored_json(row) for row in rows}
    rebuilt = reconstruct_corpus(stored_json(section), stored, policy=policy)
    if rebuilt != corpus or R.corpus_text(rebuilt) != R.corpus_text(corpus):
        raise EntityModelError("rule corpus partition does not reconstruct the source corpus exactly")
    return rows, section


def _partition(corpus: Mapping[str, Any], *, policy: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the one canonical partition of a corpus (no reconstruction proof)."""
    if not isinstance(corpus, Mapping) or set(corpus) != _CORPUS_KEYS:
        raise EntityModelError("a rule corpus has exactly version, generation, rules and provenance")
    if not isinstance(corpus["version"], str):
        raise EntityModelError("rule corpus version must be a string")
    provenance = corpus["provenance"]
    titles = [rule["title"] for rule in corpus["rules"]]
    evolved = provenance.get(R.EVOLVED) if R.EVOLVED in provenance else None
    entries = {item["title"]: item for item in evolved or []}
    owned = {title for title in titles if title in provenance and title not in _LABELS}
    rows = []
    for rule in corpus["rules"]:
        title = rule["title"]
        record = provenance[title] if title in owned else None
        entry = entries.get(title)
        rows.append(rule_row(rule, provenance=record, evolution=entry, policy=policy,
                             layout={"rule": list(rule), "provenance": order_of(record) if record is not None else None,
                                     "evolution": order_of(entry) if entry is not None else None}))
    shared = {key: deepcopy(value) for key, value in provenance.items() if key not in owned and key != R.EVOLVED}
    section = {"items": [typed_ref(RULE_ENTITY, row) for row in rows],
               "corpus": {"version": corpus["version"], "generation": deepcopy(corpus["generation"]),
                          "provenance": shared,
                          "evolved": [item["title"] for item in evolved] if evolved is not None else None,
                          "layout": {"keys": list(corpus), "generation": order_of(corpus["generation"]),
                                     "provenance": list(provenance),
                                     "shared": {key: skeleton for key, value in shared.items()
                                                if (skeleton := order_of(value)) is not None}}},
               "knowledge_digest": R.rules_digest(corpus["rules"])}
    return rows, section


def stored_json(value: Any) -> Any:
    """Return a value as sorted-member Json storage hands it back.

    Args:
        value: JSON-compatible value.

    Returns:
        A detached copy whose objects have sorted member order, as HeavenBase Json fields store them.
    """
    return json.loads(canon(value))


def _check_rules_section(section: Any) -> None:
    if not isinstance(section, Mapping) or set(section) != {"items", "corpus", "knowledge_digest"}:
        raise EntityModelError("release rules section has exactly items, corpus and knowledge_digest")
    if not isinstance(section["items"], list) or not section["items"]:
        raise EntityModelError("release rules section needs at least one rule reference")
    ids = [check_ref(item, RULE_ENTITY)["object_id"] for item in section["items"]]
    if len(set(ids)) != len(ids):
        raise EntityModelError("release rules section repeats a rule version")
    if not isinstance(section["knowledge_digest"], str) or not _KNOWLEDGE.fullmatch(section["knowledge_digest"]):
        raise EntityModelError("release rules knowledge_digest must be sha256:<64 hex>")
    corpus = section["corpus"]
    if (not isinstance(corpus, Mapping) or set(corpus) != {"version", "generation", "provenance", "evolved", "layout"}
            or not isinstance(corpus["version"], str) or not isinstance(corpus["generation"], Mapping)
            or not isinstance(corpus["provenance"], Mapping)
            or corpus["evolved"] is not None and not (isinstance(corpus["evolved"], list)
                                                      and all(isinstance(title, str) for title in corpus["evolved"]))):
        raise EntityModelError("release rules corpus metadata is malformed")
    layout = corpus["layout"]
    if (not isinstance(layout, Mapping) or set(layout) != {"keys", "generation", "provenance", "shared"}
            or not isinstance(layout["keys"], list) or set(layout["keys"]) != _CORPUS_KEYS or len(layout["keys"]) != 4
            or not isinstance(layout["provenance"], list) or not isinstance(layout["shared"], Mapping)
            or not set(layout["shared"]) <= set(corpus["provenance"])):
        raise EntityModelError("release rules corpus layout is malformed")


def reconstruct_corpus(section: Mapping[str, Any], rows: Mapping[str, Mapping[str, Any]], *,
                       policy: bool = True) -> dict[str, Any]:
    """Deterministically rebuild the whole corpus a release rules section selects.

    Args:
        section: ``content_refs["rules"]`` of a closure/2 release.
        rows: Canonical rule rows by object id (at least every referenced row).
        policy: Apply the current rule-field policy to each row (see :func:`verify_rule_row`).

    Returns:
        The corpus with its original member order; ``rules.corpus_text`` of it is the source spelling.

    Raises:
        EntityModelError: A reference, row, layout or knowledge digest differs, or the section and rows
            are not the one canonical partition of the corpus they rebuild.
    """
    _check_rules_section(section)
    meta, layout = section["corpus"], section["corpus"]["layout"]
    rules, records, evolution = [], {}, {}
    for ref in section["items"]:
        row = rows.get(ref["object_id"])
        if row is None:
            raise EntityModelError(f"release rule reference {ref['object_id']} is absent")
        verify_rule_row(row, policy=policy)
        if typed_ref(RULE_ENTITY, row) != check_ref(ref, RULE_ENTITY):
            raise EntityModelError(f"release rule reference {ref['object_id']} differs from its stored row")
        title = row["title"]
        if title in records or title in evolution or any(rule["title"] == title for rule in rules):
            raise EntityModelError(f"release rules repeat the title {title!r}")
        rules.append({field: deepcopy(row[field]) for field in row["layout"]["rule"]})
        if row.get("provenance") is not None:
            records[title] = apply_order(row["provenance"], row["layout"]["provenance"])
        if row.get("evolution") is not None:
            evolution[title] = apply_order(row["evolution"], row["layout"]["evolution"])
    shared = {key: apply_order(value, layout["shared"].get(key)) for key, value in meta["provenance"].items()}
    if set(records) & set(shared) or R.EVOLVED in shared:
        raise EntityModelError("release rules assign one provenance member to two owners")
    values = {**shared, **records}
    if meta["evolved"] is not None:
        if sorted(meta["evolved"]) != sorted(evolution) or len(set(meta["evolved"])) != len(meta["evolved"]):
            raise EntityModelError("release evolved-rule order differs from its rule versions")
        values[R.EVOLVED] = [evolution[title] for title in meta["evolved"]]
    elif evolution:
        raise EntityModelError("release rules carry evolution records without an evolved list")
    order = layout["provenance"]
    if len(set(order)) != len(order) or set(order) != set(values):
        raise EntityModelError("release rules provenance layout differs from its members")
    built = {"version": meta["version"], "generation": apply_order(meta["generation"], layout["generation"]),
             "rules": rules, "provenance": {key: values[key] for key in order}}
    corpus = {key: built[key] for key in layout["keys"]}
    if R.rules_digest(rules) != section["knowledge_digest"]:
        raise EntityModelError("release rules knowledge digest differs from its rule versions")
    # One corpus has exactly one representation: the rows and section must be its canonical partition.
    _, canonical = _partition(corpus, policy=False)
    if canon(canonical) != canon(section):
        raise EntityModelError("release rules are not the canonical partition of the corpus they rebuild")
    return corpus


# ---------------------------------------------------------------------------------------
# Parameters


def parameter_values(values: Any) -> dict[str, Any]:
    """Validate an explicit parameter payload exactly as the runtime validates ``config.knobs``.

    Args:
        values: Knob name to JSON value; omitted knobs keep their code defaults.

    Returns:
        A detached copy in sorted key order.

    Raises:
        EntityModelError: Not an object, undeclared knob, or a value outside its type or range.
    """
    if not isinstance(values, Mapping):
        raise EntityModelError("parameter values must be a JSON object")
    unknown = sorted(set(values) - set(E.KNOBS))
    if unknown:
        raise EntityModelError(f"unknown knob(s) {unknown}; the code declares {sorted(E.KNOBS)}")
    problems = [problem for name, value in sorted(values.items()) if (problem := E.KNOBS[name].violation(value)) is not None]
    if problems:
        raise EntityModelError("; ".join(problems))
    return {name: deepcopy(values[name]) for name in sorted(values)}


def _parameter_digest(values: Mapping[str, Any]) -> str:
    return _digest({"format": FORMAT_PARAMETERS, "name": PARAMETER_NAME, "values": dict(values)})


def parameter_row(values: Mapping[str, Any], *, policy: bool = True) -> dict[str, Any]:
    """Build one canonical parameter set.

    Args:
        values: Explicit knob values (validated by :func:`parameter_values`).
        policy: Validate the values against the current knob declarations; reads of retained history
            pass ``policy=False`` (see :func:`verify_parameter_row`).

    Returns:
        A complete ``knob-set-entity`` row whose projection-only columns are empty.
    """
    if not policy and (not isinstance(values, Mapping) or any(not isinstance(name, str) for name in values)):
        raise EntityModelError("parameter values must be a JSON object")
    payload = parameter_values(values) if policy else {name: deepcopy(values[name]) for name in sorted(values)}
    digest = _parameter_digest(payload)
    row = {"object_id": _PREFIX[PARAMETER_ENTITY] + digest[:16], "format": FORMAT_PARAMETERS, "name": PARAMETER_NAME,
           "version": digest[:16], "digest": digest, "values": payload, **deepcopy(dict(PARAMETER_PLACEHOLDERS))}
    verify_parameter_row(row, policy=policy)
    return row


def verify_parameter_row(row: Mapping[str, Any], *, policy: bool = True) -> dict[str, Any]:
    """Verify a canonical parameter set and return its explicit values.

    Integrity (format, name, identity, placeholders, a JSON object of named values) is always
    checked. ``policy`` adds the current knob declarations (:func:`parameter_values`); stored-history
    verifiers pass ``policy=False`` so removing or narrowing a knob later cannot invalidate a retained
    version, while publication and runtime load keep the default.

    Args:
        row: Stored or candidate ``knob-set-entity`` row.
        policy: Also validate the values against the current knob declarations.

    Returns:
        The explicit values in sorted key order.

    Raises:
        EntityModelError: The row is not an exact canonical parameter set.
    """
    if row.get("format") != FORMAT_PARAMETERS or row.get("name") != PARAMETER_NAME:
        raise EntityModelError(f"parameter set {row.get('object_id')!r} is not a {FORMAT_PARAMETERS} version")
    stored = row.get("values")
    if not isinstance(stored, Mapping) or any(not isinstance(name, str) for name in stored):
        raise EntityModelError(f"parameter set {row.get('object_id')!r} values must be a JSON object")
    values = parameter_values(stored) if policy else {name: deepcopy(stored[name]) for name in sorted(stored)}
    digest = _parameter_digest(values)
    if (row.get("digest") != digest or row.get("version") != digest[:16]
            or row.get("object_id") != _PREFIX[PARAMETER_ENTITY] + digest[:16]):
        raise EntityModelError(f"parameter set {row.get('object_id')!r} content identity differs")
    if any(row.get(key) != value for key, value in PARAMETER_PLACEHOLDERS.items()):
        raise EntityModelError(f"parameter set {row.get('object_id')!r} carries projection-only values")
    return values


# ---------------------------------------------------------------------------------------
# closure/2 releases


def check_release_refs(content_refs: Any) -> dict[str, Any]:
    """Validate the shape of a closure/2 ``content_refs`` block without resolving rows.

    Args:
        content_refs: Candidate block.

    Returns:
        A detached copy.

    Raises:
        EntityModelError: Wrong schema, members or reference shapes, or repeated tool versions.
    """
    if not isinstance(content_refs, Mapping) or set(content_refs) != {"schema", "tools", "loader", "parameters", "rules"}:
        raise EntityModelError("closure/2 content_refs has exactly schema, tools, loader, parameters and rules")
    if content_refs["schema"] != SCHEMA_RELEASE_REFS:
        raise EntityModelError(f"unknown typed reference schema {content_refs['schema']!r}; expected {SCHEMA_RELEASE_REFS}")
    tools = content_refs["tools"]
    if not isinstance(tools, list) or not tools:
        raise EntityModelError("closure/2 release needs at least one tool reference")
    ids = [check_ref(item, TOOL_ENTITY)["object_id"] for item in tools]
    if len(set(ids)) != len(ids):
        raise EntityModelError("closure/2 release repeats a tool version")
    loader = content_refs["loader"]
    if (not isinstance(loader, Mapping) or set(loader) != {"modules", "serializer"} or not isinstance(loader["modules"], list)
            or not loader["modules"] or any(not isinstance(item, Mapping) or set(item) != {"module", "capsule_id", "sha256", "row_digest"}
                                            for item in loader["modules"])
            or not isinstance(loader["serializer"], Mapping)
            or set(loader["serializer"]) != {"capsule_id", "source_digest", "row_digest"}):
        raise EntityModelError("closure/2 loader pins must list module and serializer Capsule references")
    check_ref(content_refs["parameters"], PARAMETER_ENTITY)
    _check_rules_section(content_refs["rules"])
    return deepcopy(dict(content_refs))


def check_release_document(document: Mapping[str, Any]) -> None:
    """Check the release-level invariants of a closure/2 release document or row.

    Args:
        document: Release file body or native row with ``refs``, ``entries``, ``rules``, ``content_refs``.

    Raises:
        EntityModelError: A prompt carrier is referenced, an entry is outside refs, or the block is invalid.
    """
    if document.get("rules") != CLOSURE_V2:
        raise EntityModelError("not a closure/2 release")
    refs, entries = document.get("refs"), document.get("entries")
    if not isinstance(refs, Mapping) or not refs or not isinstance(entries, list) or not entries:
        raise EntityModelError("closure/2 release needs prompt refs and entries")
    carried = sorted(set(CARRIERS) & (set(refs) | set(entries)))
    if carried:
        raise EntityModelError(f"closure/2 release cannot reference prompt carriers {carried}")
    if any(name not in refs for name in entries) or len(set(entries)) != len(entries):
        raise EntityModelError("closure/2 release entries must be distinct prompt refs")
    check_release_refs(document.get("content_refs"))


@dataclass(frozen=True)
class TypedClosure:
    """A resolved, verified closure/2 typed content.

    Attributes:
        tool_rows: Referenced Tool rows in release (serve) order.
        rule_rows: Referenced rule rows in corpus order.
        parameter_row: The referenced parameter set.
        corpus: The reconstructed corpus.
        corpus_text: ``rules.corpus_text(corpus)``, the exact legacy spelling.
        values: Explicit parameter values.
    """

    tool_rows: tuple[Mapping[str, Any], ...]
    rule_rows: tuple[Mapping[str, Any], ...]
    parameter_row: Mapping[str, Any]
    corpus: Mapping[str, Any]
    corpus_text: str
    values: Mapping[str, Any]


def resolve_closure(content_refs: Mapping[str, Any], rows: Mapping[str, Mapping[str, Mapping[str, Any]]], *,
                    capsule_rows: Mapping[str, Mapping[str, Any]], policy: bool = True) -> TypedClosure:
    """Resolve and verify every typed reference of a closure/2 release.

    Args:
        content_refs: The release's typed block.
        rows: Entity id -> object id -> canonical row (extra rows are ignored).
        capsule_rows: Native module/serializer Capsule rows by id.
        policy: Also apply the current rule, corpus and knob policy. Publication and runtime load of
            the selected release keep the default; authority and snapshot verification of stored
            history pass ``policy=False`` (integrity only, like closure/1 carriers today).

    Returns:
        The verified typed closure. Tools are validated by deterministic rebuild from their pinned
        module sources (:func:`rubiksql_hb.hb_tool_entities.validate_typed_tools`).

    Raises:
        EntityModelError: Any reference, stored entity, row, order, digest or closure differs.
    """
    from ..hb_tool_entities import validate_typed_tools

    block = check_release_refs(content_refs)

    def resolve(entity: str, ref: Mapping[str, Any]) -> Mapping[str, Any]:
        row = rows.get(entity, {}).get(ref["object_id"])
        if row is None:
            raise EntityModelError(f"{entity} {ref['object_id']} referenced by the release is absent")
        if not is_canonical(entity, row) or typed_ref(entity, row) != check_ref(ref, entity):
            raise EntityModelError(f"{entity} {ref['object_id']} differs from its typed reference")
        return row

    tools = tuple(resolve(TOOL_ENTITY, ref) for ref in block["tools"])
    try:
        validate_typed_tools(block, tool_rows={row["object_id"]: row for row in tools}, capsule_rows=capsule_rows)
    except ValueError as error:
        raise EntityModelError(f"closure/2 tools are invalid: {error}") from error
    rule_rows = tuple(resolve(RULE_ENTITY, ref) for ref in block["rules"]["items"])
    corpus = reconstruct_corpus(block["rules"], {row["object_id"]: row for row in rule_rows}, policy=policy)
    problems = R.corpus_problems(corpus) if policy else []
    if problems:
        raise EntityModelError("closure/2 rule corpus is invalid: " + "; ".join(problems))
    parameter = resolve(PARAMETER_ENTITY, block["parameters"])
    values = verify_parameter_row(parameter, policy=policy)
    return TypedClosure(tools, rule_rows, parameter, corpus, R.corpus_text(corpus), values)


def closure_view(document: Mapping[str, Any], rows: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, str]:
    """Flatten one release's closure into comparable keys.

    Prompt rows keep their names and content keys (closure/1 data carriers included). A closure/2
    release adds ``tool:<name>``, ``rule:<title>`` and ``parameters:knobs`` (row digests),
    ``rules:corpus`` (corpus metadata, rule order and knowledge digest) and ``tools:loader``
    (module and serializer pins). Typed keys contain ``:``, which no prompt name may contain.

    Args:
        document: Release file body or native row.
        rows: Entity id -> object id -> canonical row, for names and titles.

    Returns:
        Key -> 64-hex content key or digest.

    Raises:
        EntityModelError: A typed reference cannot be resolved to its row.
    """
    view = {str(name): str(key) for name, key in (document.get("refs") or {}).items()}
    if document.get("rules") != CLOSURE_V2:
        return view
    refs = check_release_refs(document.get("content_refs"))

    def row(entity: str, ref: Mapping[str, Any]) -> Mapping[str, Any]:
        found = rows.get(entity, {}).get(ref["object_id"])
        if found is None:
            raise EntityModelError(f"{entity} {ref['object_id']} referenced by the release is absent")
        return found

    for ref in refs["tools"]:
        view[f"tool:{row(TOOL_ENTITY, ref)['name']}"] = ref["digest"]
    titles = []
    for ref in refs["rules"]["items"]:
        title = row(RULE_ENTITY, ref)["title"]
        titles.append(title)
        view[f"rule:{title}"] = ref["digest"]
    view["rules:corpus"] = _digest({"corpus": refs["rules"]["corpus"], "titles": titles,
                                    "knowledge_digest": refs["rules"]["knowledge_digest"]})
    view["parameters:knobs"] = refs["parameters"]["digest"]
    view["tools:loader"] = _digest(refs["loader"])
    return view


def closure_changes(base: Mapping[str, Any], child: Mapping[str, Any],
                    rows: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, dict[str, str | None]]:
    """Derive a proposal's exact ``changes`` from its base and child releases.

    Args:
        base: Base release body or row.
        child: Child release body or row.
        rows: Entity id -> object id -> canonical row.

    Returns:
        Key -> ``{"from", "to"}`` for every differing :func:`closure_view` key.
    """
    before, after = closure_view(base, rows), closure_view(child, rows)
    return {key: {"from": before.get(key), "to": after.get(key)}
            for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)}


def change_kind(key: str) -> str:
    """Classify one :func:`closure_view` key as a proposal target kind.

    Args:
        key: Prompt name or typed key.

    Returns:
        ``rules``, ``knobs``, ``tool``, ``generator`` or ``prompt``.
    """
    if key == E.RULES_ENTRY or key.startswith("rule:") or key == "rules:corpus":
        return "rules"
    if key == E.KNOBS_ENTRY or key == "parameters:knobs":
        return "knobs"
    if key == "tools.answering" or key.startswith("tool:") or key == "tools:loader":
        return "tool"
    return "generator" if key.startswith("gen.") else "prompt"


def legacy_texts(closure: TypedClosure) -> dict[str, str]:
    """Return the exact legacy data-surface texts of a typed closure.

    Args:
        closure: Verified typed closure.

    Returns:
        ``{"config.knobs": knobs_text(values), "rules.general": corpus text}``; a closure/2 release
        keeps the same data-surface identities as the closure/1 release it represents.
    """
    return {E.KNOBS_ENTRY: E.knobs_text(closure.values), E.RULES_ENTRY: closure.corpus_text}


def corpus_from_text(text: str) -> dict[str, Any]:
    """Parse a legacy ``rules.general`` text that must already use the exact corpus spelling.

    Args:
        text: Carrier text.

    Returns:
        The member-order-preserving corpus.

    Raises:
        EntityModelError: The text is not the canonical ``indent=2`` corpus spelling.
    """
    corpus = json.loads(text)
    if not isinstance(corpus, dict) or R.corpus_text(corpus) != text:
        raise EntityModelError("rules.general text is not the exact corpus spelling")
    return corpus


def values_from_text(text: str, *, policy: bool = True) -> dict[str, Any]:
    """Parse a legacy ``config.knobs`` text that must already use the exact knob spelling.

    Args:
        text: Carrier text.
        policy: Validate against the current knob declarations; reads of retained history pass
            ``policy=False`` (a JSON object in the exact spelling only).

    Returns:
        Validated explicit values.

    Raises:
        EntityModelError: Invalid knobs or a different spelling.
    """
    values = json.loads(text)
    if policy:
        values = parameter_values(values)
    elif not isinstance(values, dict):
        raise EntityModelError("config.knobs text is not a JSON object")
    if E.knobs_text(values) != text:
        raise EntityModelError("config.knobs text is not the exact knob spelling")
    return values
