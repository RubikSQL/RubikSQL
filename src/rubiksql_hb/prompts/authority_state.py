"""One-time removal of refresh bookkeeping and independent current-state admission.

Only ``migration_preimage`` understands the retired decision sequence. Normal
readers verify the compact state admission and its exact immutable typed closure.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import re
import time
from typing import Any, Mapping

from .canon import canon, sha256_hex
from .types import AuthorityError, AuthorityHead, AuthorityRevision

SCHEMA = "rubiksql.authority-state/1"
OPERATION = "retire-refresh"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_EXPECTED_FIELDS = {"release", "generation", "decision", "revision", "exported", "export_commit"}
_FIELDS = {"schema", "operation", "phase", "request_id", "actor", "user_instruction", "expected",
           "epoch", "release", "generation", "namespace", "budget", "content_digest", "source_admission",
           "preserved", "backup", "removed", "preimage_digest", "metadata_updates"}
_METADATA_FIELDS = {"parents", "proposals", "origin", "row_versions", "status", "baseline", "knobs", "store_commit"}


@dataclass(frozen=True)
class RetirementPlan:
    """Exact checked preimage and native deletion/metadata batch; no new content."""

    expected: AuthorityHead
    preimage_digest: str
    deletions: Mapping[str, tuple[str, ...]]
    release_updates: Mapping[str, Mapping[str, Any]]
    anchor_payload: Mapping[str, Any]


def _hash(value: Any) -> str:
    return sha256_hex(canon(value))


def _content(row: Mapping[str, Any]) -> dict[str, Any]:
    return {name: row[name] for name in ("refs", "entries", "rules", "content_refs")}


def _order(event: Mapping[str, Any]) -> tuple[float, str]:
    return float(event["ts"]), str(event["object_id"])


def _event_identity(event: Mapping[str, Any]) -> None:
    if event.get("object_id") != "e" + _hash({key: value for key, value in event.items() if key != "object_id"})[:16]:
        raise AuthorityError("authority state contains an altered event")


def _selected_rows(reader: Any, release: Mapping[str, Any]) -> tuple[list[Any], Mapping[str, Any]]:
    from . import entity_model as M
    from .integrity import check_prompt_row

    indexed = {(row["name"], row["manifest"]["meta"]["content_key"]): row for row in reader.rows("sys-prompt")}
    try:
        prompts = [indexed[(name, key)] for name, key in sorted(release["refs"].items())]
        for row in prompts:
            check_prompt_row(row)
        resolved = M.resolve_closure(release["content_refs"], reader.typed_index(), policy=False,
                                     capsule_rows={row["object_id"]: row for row in reader.rows("sys-capsule")})
    except (KeyError, ValueError, TypeError) as error:
        raise AuthorityError("authority state typed closure is incomplete or altered") from error
    return prompts, resolved


def _preserved(reader: Any, release: Mapping[str, Any], *, operational: bool = True) -> dict[str, str]:
    from . import entity_model as M

    prompts, _resolved = _selected_rows(reader, release)
    ids: dict[str, set[str]] = {entity: set() for entity in M.TYPED_ENTITY_IDS}
    refs = release["content_refs"]
    for reference in [*refs["tools"], refs["parameters"], *refs["rules"]["items"]]:
        ids[reference["entity"]].add(reference["object_id"])
    typed = {entity: [row for row in reader.rows(entity) if row["object_id"] in ids[entity]] for entity in ids}
    namespace = f"v0-{reader.epoch()['release']}"
    return {"release_content_sha256": _hash(_content(release)), "prompt_rows_sha256": _hash(prompts),
            "typed_rows_sha256": _hash(typed),
            "operational_sha256": _hash(reader.operational_state() if operational and hasattr(reader, "operational_state") else {}),
            "budget_sha256": _hash({"used": 0, "cap": 10}), "namespace_sha256": _hash(namespace)}


def _projection_rows(reader: Any) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """Read obsolete descriptive projections only for migration/reference fences.

    Ordinary version readers intentionally exclude these noncanonical rows. A
    sealed version-only snapshot has no such physical rows to inspect.
    """
    from heavenbase.utils.serialize import loads_json
    from . import entity_model as M
    from .store import TABLES

    if not hasattr(reader, "connection"):
        return {}
    result = {}
    for entity in ("general-rule-entity", "knob-set-entity"):
        table = TABLES[entity]
        columns = {row[1] for row in reader.connection.execute(f'PRAGMA table_info("{table}")')}
        if not columns:
            continue
        schema = reader._schemas[entity]
        rows = []
        for native in reader.connection.execute(f'SELECT * FROM "{table}" ORDER BY object_id'):
            row = dict(native)
            for name, field in schema.fields.items():
                if row.get(name) is None:
                    continue
                value = row[name]
                if type(field.dtype).identifier in {"json", "array"}:
                    value = loads_json(value)
                row[name] = field.dtype.decode(value)
            if not M.is_canonical(entity, row):
                rows.append(row)
        result[entity] = tuple(rows)
    return result


def _preimage_digest(reader: Any, head: AuthorityHead) -> str:
    from .store import TABLES

    return _hash({"expected": asdict(head), "epoch": reader.epoch(),
                  "native": {entity: reader.rows(entity) for entity in TABLES},
                  "projections": _projection_rows(reader),
                  "operational": reader.operational_state()})


def _idle(reader: Any) -> None:
    from .workspace_migration import pending_migration

    if (reader.pending_publications() or reader.pending_tool_registrations() or pending_migration(reader)
            or reader.events("eval-reserve") or reader.events("eval-claim")
            or reader.events("eval-attach") or reader.events("outcome")
            or reader.workspace_rows("experiment-entity") or reader.workspace_rows("trace-entity")
            or reader.workspace_rows("candidate-entity") or reader.workspace_rows("judge-entity")
            or reader.workspace_rows("agent-session") or reader.workspace_rows("agent-message")):
        raise AuthorityError("refresh retirement refuses pending, retained measured, or in-flight activity")


def _old_export(reader: Any) -> tuple[bool, str | None, str | None]:
    """Check an existing export receipt or a verified restored-copy export anchor."""
    from .authority import _export_format, _read_export
    from .store import TABLES, _json_value
    from .workspace_state import EVIDENCE_TABLES, decode_evidence_rows

    anchor = getattr(reader, "export_anchor", None)
    if anchor is not None:
        head, files, commit = _read_export(anchor)
        number = _export_format(head["schema"])
        if (number not in {3, 4}
                or any(json.loads(files[f"native/{entity}.json"]) != _json_value(reader.rows(entity)) for entity in TABLES)
                or any(decode_evidence_rows(entity, files, row_files=number == 4) != _json_value(reader.workspace_rows(entity))
                       for entity in EVIDENCE_TABLES)):
            raise AuthorityError("refresh retirement backup differs from the complete native preimage")
        selected = reader.events("decision")[-1]
        if ({key: head.get(key) for key in ("release", "generation", "decision")} != {
                "release": selected["payload"]["to"], "generation": selected["payload"]["to"],
                "decision": selected["object_id"]} or head.get("covered_revision") != {
                    "event_count": len(reader.events()), "latest_event_id": reader.events()[-1]["object_id"]}):
            raise AuthorityError("refresh retirement backup pointer or complete revision differs")
        return True, commit, sha256_hex((anchor / "HEAD.json").read_bytes())
    fresh, commit = reader._export_state()
    exports = reader.events("export")
    return fresh, commit, exports[-1]["payload"].get("head_sha256") if exports else None


def _publication(reader: Any, pid: str) -> Mapping[str, Any]:
    """Strict migration-only interpretation of the exact old complete publication."""
    from .authority import _batch_digest, _prepared_batch, _proposal_row, semantic_row
    from .store import proposal_id

    complete = [event for event in reader.events("publish") if event["subject"] == pid
                and isinstance(event["payload"].get("body"), Mapping)]
    prepared = [event for event in reader.events("publish") if event["subject"] == pid
                and event["payload"].get("phase") == "prepared"]
    if len(complete) != 1 or len(prepared) != 1:
        raise AuthorityError("refresh retirement requires an exact completed publication and preimage")
    body = complete[0]["payload"]["body"]
    row = reader.row("proposal-entity", pid)
    batch = _prepared_batch(prepared[0]["payload"])
    if (body.get("id") != pid or proposal_id(body) != pid or batch.proposal != body
            or prepared[0]["payload"].get("batch_digest") != _batch_digest(batch)
            or complete[0]["payload"].get("batch_digest") != _batch_digest(batch)
            or semantic_row(row) != semantic_row(_proposal_row(body, row.get("gate") or {}))):
        raise AuthorityError("refresh retirement publication preimage differs")
    for entity, candidates in prepared[0]["payload"].get("candidates", {}).items():
        held = {item["object_id"]: item for item in reader.rows(entity)}
        for candidate in candidates:
            # Deleted historical Toolkits are verified against their exact retained
            # cutover preimage by _toolkit_preimage; every other candidate must exist.
            if entity == "sys-toolkit" and candidate["object_id"] not in held:
                continue
            if (candidate["object_id"] not in held
                    or semantic_row(candidate) != semantic_row(held[candidate["object_id"]])):
                raise AuthorityError("refresh retirement original native publication candidate differs")
    return body


def migration_preimage(reader: Any, expected: AuthorityHead | None = None) -> AuthorityHead:
    """Derive only the exact unmeasured refresh-dependent typed preimage.

    Args:
        reader: One stable native read transaction; normal ``head`` is not called.
        expected: Optional full optimistic revision and export facts.

    Returns:
        AuthorityHead: Verified original head, including its complete revision.

    Raises:
        AuthorityError: Any chain, content, export, revision or activity differs.
    """
    from . import entity_model as M
    from .integrity import check_prompt_row
    from .store import release_document_id

    events = reader.events()
    for event in events:
        _event_identity(event)
    if any(event["payload"].get("schema") == SCHEMA for event in events):
        raise AuthorityError("refresh retirement already has a current-state admission")
    _idle(reader)
    moves = reader.events("decision")
    if [event["payload"].get("kind") for event in moves] != ["cutover", "refresh-tools", M.MIGRATION_KIND, "refresh-tools"]:
        raise AuthorityError("refresh retirement requires the exact cutover, refresh, entity-model, typed-refresh chain")
    if (len({event["payload"].get("to") for event in moves}) != 4
            or len({event["payload"].get("target") for event in moves}) != 4):
        raise AuthorityError("refresh retirement requires distinct original releases and producers")
    previous, current = None, reader.epoch()["release"]
    for event in moves:
        payload = event["payload"]
        body = _publication(reader, str(payload["target"]))
        if (payload.get("schema") != "rubiksql.decision/1" or payload.get("previous_decision") != previous
                or payload.get("from") != current or payload.get("generation") != payload.get("to")
                or not str(payload.get("user_instruction", "")).strip()
                or body.get("origin") != payload["kind"] or body.get("release") != payload.get("to")
                or body.get("parents") != [{"release": current, "role": "base"}]
                or event["subject"] != body["id"]):
            raise AuthorityError("refresh retirement original decision or publication chain differs")
        previous, current = event["object_id"], payload["to"]
    _check_original_content(reader, moves)
    for row in reader.rows("sys-prompt"):
        check_prompt_row(row)
    for row in reader.rows("release-entity"):
        if release_document_id(row, epoch=reader.epoch()["epoch"]) != row["object_id"]:
            raise AuthorityError("refresh retirement original release identity differs")
    release = reader.row("release-entity", str(current))
    if release["rules"] != M.CLOSURE_V2:
        raise AuthorityError("refresh retirement adopts only the already-selected typed release")
    _selected_rows(reader, release)
    _source_admission(reader, moves[-1])
    _toolkit_preimage(reader, moves)
    fresh, commit, head_hash = _old_export(reader)
    if not fresh or not isinstance(commit, str) or not _COMMIT.fullmatch(commit) or not _valid_hash(head_hash):
        raise AuthorityError("refresh retirement requires an intact complete exported backup")
    head = AuthorityHead(str(current), str(current), str(previous),
                         AuthorityRevision(len(events), events[-1]["object_id"] if events else None), True, commit)
    if expected is not None and expected != head:
        raise AuthorityError("refresh retirement expected full preimage revision is stale")
    return head


def _source_admission(reader: Any, decision: Mapping[str, Any]) -> dict[str, Any]:
    from ..hb_tool_entities import digest, typed_bundle

    payload = decision["payload"]
    gates = [event for event in reader.events("gate") if event["subject"] == payload["target"]]
    parents = [event for event in reader.events("gate") if event["subject"] == reader.events("decision")[-2]["subject"]]
    if len(gates) != 1 or len(parents) != 1:
        raise AuthorityError("refresh retirement requires exact current and parent source admissions")
    gate = gates[0]["payload"]
    parent = reader.row("release-entity", payload["from"])
    parent_gate = parents[0]["payload"]
    if parent_gate.get("schema") == "rubiksql.entity-model-gate/1":
        source_gates = [event["payload"] for event in reader.events("gate")
                        if event["payload"].get("schema") == "rubiksql.tool-gate/1"
                        and event["payload"].get("bundle_digest") == parent_gate.get("legacy_bundle_digest")
                        and _order(event) < _order(parents[0])]
        if len(source_gates) != 1:
            raise AuthorityError("refresh retirement typed parent lacks its exact original source admission")
        source = source_gates[0].get("source_commit")
    else:
        source = parent_gate.get("source_commit")
    admission = {"gate": deepcopy(gate), "parent_bundle_digest": digest(typed_bundle(parent["content_refs"])),
                 "parent_source_commit": source}
    _check_admission(reader, reader.row("release-entity", payload["to"]), admission)
    if gate.get("source_commit") != payload.get("deployed_commit"):
        raise AuthorityError("refresh retirement current source differs from its original decision")
    return admission


def _check_original_content(reader: Any, moves: Any) -> None:
    """Migration-only recomputation of original content changes and gate proofs."""
    from ..hb_tool_entities import (_validate_legacy_answering, digest, typed_bundle, validate_answering,
                                    validate_tool_receipts)
    from . import entity_model as M
    from .authority import _check_entity_gate, _check_entity_migration

    capsules = {row["object_id"]: row for row in reader.rows("sys-capsule")}
    for index, event in enumerate(moves):
        payload = event["payload"]
        proposal = _publication(reader, payload["target"])
        child = reader.row("release-entity", payload["to"])
        parent = reader.row("release-entity", payload["from"])
        gates = [item for item in reader.events("gate") if item["subject"] == payload["target"]]
        if len(gates) != 1:
            raise AuthorityError("refresh retirement original publication lacks its unique gate")
        gate = gates[0]["payload"]
        if index == 2:
            check = _check_entity_migration(reader, proposal, child, payload["user_instruction"], policy=False)
            _check_entity_gate(reader, payload["target"], check, payload.get("deployed_commit"))
            continue
        if index == 0:
            if (len(parent["refs"]) != 34 or child["rules"] != parent["rules"]
                    or child["entries"] != sorted([*parent["entries"], "tools.answering"])
                    or child["refs"] != {
                    **parent["refs"], "tools.answering": child["refs"].get("tools.answering")}
                    or proposal["changes"] != {"tools.answering": {"from": None, "to": child["refs"]["tools.answering"]}}):
                raise AuthorityError("refresh retirement original cutover content differs")
            bundle = reader.answering(payload["to"])
            prepared = next(item for item in reader.events("publish") if item["subject"] == proposal["id"]
                            and item["payload"].get("phase") == "prepared")
            _validate_legacy_answering(bundle, capsule_rows=capsules, toolkit_rows={
                row["object_id"]: row for row in prepared["payload"]["candidates"].get("sys-toolkit", [])})
            parent_digest = None
        elif index == 1:
            if (child["rules"] != parent["rules"] or child["entries"] != parent["entries"]
                    or child["refs"] != {**parent["refs"], "tools.answering": child["refs"].get("tools.answering")}
                    or proposal["changes"] != {"tools.answering": {"from": parent["refs"]["tools.answering"],
                                                                   "to": child["refs"]["tools.answering"]}}):
                raise AuthorityError("refresh retirement original carrier refresh changed unrelated content")
            bundle = reader.answering(payload["to"])
            validate_answering(bundle, capsule_rows=capsules)
            parent_digest = digest(reader.answering(payload["from"]))
        else:
            if (child["refs"] != parent["refs"] or child["entries"] != parent["entries"] or child["rules"] != parent["rules"]
                    or {key: value for key, value in child["content_refs"].items() if key not in {"tools", "loader"}} != {
                        key: value for key, value in parent["content_refs"].items() if key not in {"tools", "loader"}}
                    or proposal["changes"] != M.closure_changes(parent, child, reader.typed_index())):
                raise AuthorityError("refresh retirement original typed refresh changed unrelated content")
            bundle = typed_bundle(child["content_refs"])
            parent_digest = digest(typed_bundle(parent["content_refs"]))
        try:
            validate_tool_receipts(bundle, gate, parent_bundle=parent_digest, launch=False, tiers=gate.get("tiers", {}))
        except (ValueError, KeyError, TypeError) as error:
            raise AuthorityError(f"refresh retirement original source admission is invalid: {error}") from error
        if gate.get("source_commit") != payload.get("deployed_commit"):
            raise AuthorityError("refresh retirement original source gate differs from the decision")


def _check_admission(reader: Any, release: Mapping[str, Any], admission: Mapping[str, Any]) -> None:
    from ..hb_tool_entities import typed_bundle, validate_tool_receipts

    if (not isinstance(admission, Mapping) or set(admission) != {"gate", "parent_bundle_digest", "parent_source_commit"}
            or not _valid_hash(admission.get("parent_bundle_digest"))
            or not isinstance(admission.get("parent_source_commit"), str)
            or not _COMMIT.fullmatch(admission["parent_source_commit"])):
        raise AuthorityError("authority state source admission parent identity is invalid")
    gate = admission["gate"]
    if not isinstance(gate, Mapping):
        raise AuthorityError("authority state source gate is not a record")
    try:
        validate_tool_receipts(typed_bundle(release["content_refs"]), gate,
                               parent_bundle=admission["parent_bundle_digest"], launch=True,
                               tiers=gate.get("tiers", {}))
        if gate.get("mode") == "git" and gate.get("audit_implementation_commit") != admission["parent_source_commit"]:
            raise ValueError("current Git audit did not execute from its exact original source parent")
        if "scientific_gate" in gate and (not isinstance(gate["scientific_gate"], Mapping)
                                          or gate["scientific_gate"].get("passed") is not True):
            raise ValueError("current scientific gate did not pass")
    except (ValueError, KeyError, TypeError) as error:
        raise AuthorityError(f"authority state source admission is invalid: {error}") from error


def _toolkit_preimage(reader: Any, moves: Any) -> Mapping[str, Any] | None:
    from .toolkit_retirement import _archived, verify_detached

    receipts = reader.events("toolkit-retirement")
    if not receipts:
        return None
    if len(receipts) != 1:
        raise AuthorityError("refresh retirement requires a unique historical Toolkit receipt")
    event, = receipts
    body = event["payload"]
    prior = [item for item in reader.events() if _order(item) < _order(event)]
    decisions = [item for item in moves if _order(item) < _order(event)]
    current = decisions[-1] if decisions else None
    if (not current or body.get("current") != {"release": current["payload"]["to"],
            "generation": current["payload"]["to"], "decision": current["object_id"]}
            or body.get("epoch") != reader.epoch() or body.get("revision_before") != {
                "event_count": len(prior), "latest_event_id": prior[-1]["object_id"] if prior else None}):
        raise AuthorityError("refresh retirement original Toolkit receipt preimage differs")
    detached = {"archived": body["archived"], "deleted_toolkits": body["deleted_toolkits"]}
    archived = _archived(reader.rows("release-entity"), reader.rows("sys-prompt"))
    verify_detached(detached, archived, {row["object_id"]: row for row in reader.rows("sys-capsule")},
                    {row["object_id"]: row for row in reader.rows("sys-toolkit")}, reader.events())
    return deepcopy(detached)


def _references(value: Any, ids: set[str]) -> bool:
    if isinstance(value, Mapping):
        return any(_references(key, ids) or _references(item, ids) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return any(_references(item, ids) for item in value)
    return isinstance(value, str) and (value in ids or any(
        value in {f"releases/{identifier}.json", f"proposals/{identifier}.json",
                  f"decisions/{identifier}.json", f"attestations/{identifier}-gate.json"} for identifier in ids))


def _tokens(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return set().union(*(_tokens(key) | _tokens(item) for key, item in value.items())) if value else set()
    if isinstance(value, (tuple, list)):
        return set().union(*map(_tokens, value)) if value else set()
    return {value} if isinstance(value, str) else set()


def _prune_owned(reader: Any, proposals: set[str], deletions: dict[str, set[str]],
                 release: Mapping[str, Any], admission: Mapping[str, Any], detached: Any, *,
                 release_updates: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    """Prune only exact publication-owned rows unreachable from all retained roots."""
    from .authority import semantic_row
    from .store import TABLES
    from .workspace_state import EVIDENCE_TABLES

    eligible: dict[tuple[str, str], Mapping[str, Any]] = {}
    for event in reader.events("publish"):
        if event["subject"] not in proposals or event["payload"].get("phase") != "prepared":
            continue
        for entity, candidates in event["payload"]["candidates"].items():
            if entity not in TABLES or entity in {"release-entity", "proposal-entity", "prompt-event-entity"}:
                continue
            held = {row["object_id"]: row for row in reader.rows(entity)}
            for candidate in candidates:
                identifier = candidate["object_id"]
                if identifier in held:
                    if semantic_row(candidate) != semantic_row(held[identifier]):
                        raise AuthorityError("refresh retirement publication-owned immutable row differs")
                    eligible[(entity, identifier)] = held[identifier]
    aliases: dict[str, set[tuple[str, str]]] = {}
    for key, row in eligible.items():
        identifiers = {key[1], f"tools/capsules/{key[1]}.json"}
        if key[0] == "sys-prompt":
            content_key = row["manifest"]["meta"]["content_key"]
            identifiers.update({content_key, f'{row["name"]}:{content_key[:16]}',
                                f'objects/{row["name"]}/{content_key[:16]}.json'})
        for identifier in identifiers:
            aliases.setdefault(identifier, set()).add(key)
    import_events: dict[str, Mapping[str, Any]] = {}
    roots = [admission, detached]
    for entity in TABLES:
        for row in reader.rows(entity):
            if row["object_id"] in deletions.get(entity, set()) or (entity, row["object_id"]) in eligible:
                continue
            if entity == "prompt-event-entity" and row["kind"] == "import" and _tokens(row) & aliases.keys():
                import_events[row["object_id"]] = row
                continue
            if entity == "release-entity":
                row = (release_updates or {}).get(row["object_id"], row)
                if row["object_id"] == release["object_id"]:
                    row = release
            roots.append(row)
    roots.extend(row for entity in EVIDENCE_TABLES for row in reader.workspace_rows(entity))
    roots.extend(row for entity, rows in _projection_rows(reader).items() for row in rows
                 if row["object_id"] not in deletions.get(entity, set()))
    live: set[tuple[str, str]] = set()
    frontier = _tokens(roots)
    while frontier:
        reachable = set().union(*(aliases.get(token, set()) for token in frontier)) - live
        if not reachable:
            break
        live.update(reachable)
        frontier = _tokens([eligible[key] for key in reachable])
    dead = set(eligible) - live
    for entity, identifier in dead:
        deletions.setdefault(entity, set()).add(identifier)
    dead_aliases = {alias for alias, keys in aliases.items() if keys <= dead}
    for identifier, event in import_events.items():
        if _tokens(event) & dead_aliases:
            deletions["prompt-event-entity"].add(identifier)


def _clear_metadata(reader: Any, current: str, deleted_ids: set[str]) -> tuple[dict[str, Mapping[str, Any]], list[Any]]:
    """Detach only descriptive fields naming removed history and current producers."""
    updates: dict[str, Mapping[str, Any]] = {}
    roster = []
    for row in reader.rows("release-entity"):
        if row["object_id"] in deleted_ids:
            continue
        fields = {key for key in _METADATA_FIELDS if key in row and _references(row[key], deleted_ids)}
        if row["object_id"] == current:
            fields.update(key for key in ("parents", "proposals", "origin") if row.get(key) is not None)
        if fields:
            updated = {**row, **{key: None for key in fields}}
            updates[row["object_id"]] = updated
            roster.append({"entity": "release-entity", "object_id": row["object_id"],
                "fields": sorted(fields), "before_sha256": _hash({key: row[key] for key in sorted(fields)}),
                "after_sha256": _hash({key: updated[key] for key in sorted(fields)})})
    return updates, sorted(roster, key=lambda item: item["object_id"])


def _obsolete_projection_events(reader: Any, obsolete: set[str]) -> set[str]:
    """Select only unmeasured derived-view receipts owned by deleted releases."""
    from .workspace_state import EVIDENCE_TABLES

    selected = set()
    for event in reader.events("workspace-change"):
        payload = event["payload"]
        if (event["subject"] not in obsolete or payload.get("schema") != "rubiksql.workspace-change/1"
                or payload.get("operation") != "refresh-views"):
            continue
        counts, digests = payload.get("counts"), payload.get("entity_digests")
        if (set(payload) != {"schema", "operation", "workspace_digest", "counts", "entity_digests", "schemas_digest"}
                or not isinstance(counts, Mapping) or set(counts) != set(EVIDENCE_TABLES)
                or any(type(value) is not int or value < 0 for value in counts.values())
                or any(counts[entity] for entity in EVIDENCE_TABLES if entity != "q-a-entity")
                or not isinstance(digests, Mapping) or set(digests) != set(EVIDENCE_TABLES)
                or not all(_valid_hash(value) for value in digests.values())
                or not _valid_hash(payload["workspace_digest"]) or not _valid_hash(payload["schemas_digest"])):
            raise AuthorityError("refresh retirement obsolete projection receipt is malformed or measured")
        selected.add(event["object_id"])
    return selected


def plan_retirement(reader: Any, expected: AuthorityHead, *, request_id: str,
                    user_instruction: str, actor: str) -> RetirementPlan:
    """Plan atomic deletion and admission of the unchanged selected typed release.

    Args:
        reader: Stable native preimage read transaction.
        expected: Exact original head, revision and exported backup.
        request_id: Explicit unique cleanup request identity.
        user_instruction: Verbatim authorization for this cleanup.
        actor: Identity responsible for the authorized request.

    Returns:
        RetirementPlan: Detached exact row ids/hashes and the standalone admission.

    Raises:
        AuthorityError: Stale preimage, pending activity, missing proof or retained reference.
    """
    from .store import TABLES

    if any(not isinstance(value, str) or not value.strip() for value in (request_id, user_instruction, actor)):
        raise AuthorityError("refresh retirement requires request identity, actor and verbatim user instruction")
    head = migration_preimage(reader, expected)
    moves = reader.events("decision")
    dependent = moves[1:]
    proposals = {event["payload"]["target"] for event in dependent}
    obsolete = {event["payload"]["to"] for event in dependent} - {head.release}
    deletions: dict[str, set[str]] = {"proposal-entity": proposals, "release-entity": obsolete,
                                    "prompt-event-entity": {event["object_id"] for event in dependent}}
    ids = proposals | obsolete
    for event in reader.events():
        if event["kind"] in {"publish", "gate"} and event["subject"] in proposals:
            deletions["prompt-event-entity"].add(event["object_id"])
    deletions["prompt-event-entity"].update(_obsolete_projection_events(reader, obsolete))
    detached = _toolkit_preimage(reader, moves)
    if detached is not None:
        deletions["prompt-event-entity"].add(reader.events("toolkit-retirement")[0]["object_id"])
    release = deepcopy(reader.row("release-entity", head.release))
    for key in ("parents", "proposals", "origin"):
        if key in release:
            release[key] = None
    if "proposal" in release:
        release.pop("proposal")
    admission = _source_admission(reader, moves[-1])
    projections = _projection_rows(reader)
    for entity, rows in projections.items():
        for row in rows:
            if _references(row, ids | deletions["prompt-event-entity"]):
                deletions.setdefault(entity, set()).add(row["object_id"])
    # Exports covering retired producers are invalid and must not protect their
    # dead candidate rows merely through old file-map entries.
    for event in reader.events("export"):
        if _references(event, ids | deletions["prompt-event-entity"]):
            deletions["prompt-event-entity"].add(event["object_id"])
    roots, _metadata = _clear_metadata(reader, head.release, set().union(*deletions.values()))
    release = roots.get(head.release, release)
    _prune_owned(reader, proposals, deletions, release, admission, detached, release_updates=roots)
    # An export that covers any removed event cannot remain a live certification.
    while True:
        prior = set(deletions["prompt-event-entity"])
        for event in reader.events("export"):
            if _references(event, ids | prior):
                deletions["prompt-event-entity"].add(event["object_id"])
        if prior == deletions["prompt-event-entity"]:
            break
    deleted_ids = set().union(*deletions.values())
    updates, metadata_updates = _clear_metadata(reader, head.release, deleted_ids)
    release = updates.get(head.release, release)
    for entity in TABLES:
        for row in (*reader.rows(entity), *projections.get(entity, ())):
            if row["object_id"] in deletions.get(entity, set()):
                continue
            retained = updates.get(row["object_id"], row) if entity == "release-entity" else row
            if _references(retained, deleted_ids):
                raise AuthorityError(f"refresh retirement retained {entity} row references removed history")
    digest = _preimage_digest(reader, head)
    _fresh, commit, head_hash = _old_export(reader)
    raw_rows = {(entity, row["object_id"]): row for entity, rows in projections.items() for row in rows}
    removed = {entity: [{"object_id": identifier, "row_sha256": _hash(
                            raw_rows[(entity, identifier)] if (entity, identifier) in raw_rows else reader.row(entity, identifier))}
                        for identifier in sorted(identifiers)] for entity, identifiers in sorted(deletions.items())}
    payload = {"schema": SCHEMA, "operation": OPERATION, "phase": "completed", "request_id": request_id,
               "actor": actor, "user_instruction": user_instruction, "expected": asdict(head),
               "epoch": deepcopy(reader.epoch()), "release": head.release, "generation": head.generation,
               "namespace": f"v0-{reader.epoch()['release']}", "budget": {"used": 0, "cap": 10},
               "content_digest": _hash(_content(release)), "source_admission": admission,
               "preserved": _preserved(reader, release), "backup": {"export_commit": commit, "head_sha256": head_hash},
               "removed": removed, "preimage_digest": digest, "metadata_updates": metadata_updates,
               **({"toolkit_retirement": detached} if detached is not None else {})}
    return RetirementPlan(head, digest, {entity: tuple(sorted(values)) for entity, values in deletions.items()},
                          updates, payload)


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


def validate_current_state(reader: Any) -> Mapping[str, Any] | None:
    """Verify a standalone completed state admission without retired-history readers.

    Args:
        reader: Stable native authority or snapshot read view.

    Returns:
        Mapping[str, Any] | None: Exact verified event, or None if no admission exists.

    Raises:
        AuthorityError: Altered proof, missing source/content, dangling references or duplicate admission.
    """
    from . import entity_model as M
    from .store import TABLES, release_document_id

    anchors = [event for event in reader.events("register") if event["payload"].get("schema") == SCHEMA]
    if not anchors:
        return None
    if len(anchors) != 1:
        raise AuthorityError("authority state needs one completed admission")
    if any(item["kind"] == "decision" and item["payload"].get("kind") == "refresh-tools"
           or item["kind"] == "publish" and item["payload"].get("body", {}).get("origin") == "refresh-tools"
           for item in reader.events()):
        raise AuthorityError("authority state still contains obsolete refresh records")
    event, = anchors
    _event_identity(event)
    body = event["payload"]
    if (set(body) not in (_FIELDS, _FIELDS | {"toolkit_retirement"}) or body.get("operation") != OPERATION
            or body.get("phase") != "completed" or event["subject"] != body.get("release")
            or event["actor"] != body.get("actor")
            or any(not isinstance(body.get(key), str) or not body[key].strip()
                   for key in ("request_id", "actor", "user_instruction"))
            or body.get("epoch") != reader.epoch() or body.get("generation") != body.get("release")
            or body.get("namespace") != f"v0-{reader.epoch()['release']}"
            or body.get("budget") != {"used": 0, "cap": 10} or not _valid_hash(body.get("preimage_digest"))):
        raise AuthorityError("authority state authorization or preserved identity differs")
    expected = body["expected"]
    backup = body["backup"]
    if (not isinstance(expected, Mapping) or set(expected) not in (_EXPECTED_FIELDS, _EXPECTED_FIELDS | {"authority_commit"})
            or expected.get("authority_commit") is not None and not _COMMIT.fullmatch(str(expected["authority_commit"]))
            or expected["release"] != body["release"] or expected["generation"] != body["generation"]
            or not isinstance(expected["decision"], str) or not expected["decision"]
            or not isinstance(backup, Mapping)
            or expected["exported"] is not True or expected["export_commit"] != backup.get("export_commit")
            or not isinstance(expected.get("revision"), Mapping)
            or set(expected["revision"]) != {"event_count", "latest_event_id"}
            or type(expected["revision"]["event_count"]) is not int or expected["revision"]["event_count"] < 1
            or not isinstance(expected["revision"]["latest_event_id"], str) or not expected["revision"]["latest_event_id"]
            or set(backup) != {"export_commit", "head_sha256"}
            or not isinstance(backup["export_commit"], str) or not _COMMIT.fullmatch(backup["export_commit"])
            or not _valid_hash(backup["head_sha256"])):
        raise AuthorityError("authority state prior expected revision or backup identity differs")
    release = reader.row("release-entity", body["release"])
    if (release["rules"] != M.CLOSURE_V2 or release_document_id(release, epoch=reader.epoch()["epoch"]) != body["release"]
            or body["content_digest"] != _hash(_content(release))
            or any(release.get(key) for key in ("parents", "proposal", "proposals", "origin"))):
        raise AuthorityError("authority state adopted release content or detached metadata differs")
    _check_admission(reader, release, body["source_admission"])
    preserved = body["preserved"]
    actual = _preserved(reader, release, operational=False)
    if (not isinstance(preserved, Mapping) or set(preserved) != set(actual) or not all(_valid_hash(value) for value in preserved.values())
            or any(preserved[key] != actual[key] for key in actual if key != "operational_sha256")):
        raise AuthorityError("authority state preserved immutable content differs")
    removed = body["removed"]
    if (not isinstance(removed, Mapping) or not {"proposal-entity", "release-entity", "prompt-event-entity"} <= set(removed)
            or set(removed) - set(TABLES)):
        raise AuthorityError("authority state deleted native entity roster differs")
    deleted: dict[str, set[str]] = {}
    projections = _projection_rows(reader)
    for entity, records in removed.items():
        if (not isinstance(records, list) or not records
                or any(not isinstance(item, Mapping) or set(item) != {"object_id", "row_sha256"}
                       or not isinstance(item["object_id"], str) or not _valid_hash(item["row_sha256"]) for item in records)
                or [item["object_id"] for item in records] != sorted({item["object_id"] for item in records})):
            raise AuthorityError("authority state deleted row identity/hash roster differs")
        deleted[entity] = {item["object_id"] for item in records}
    readmitted = _catalog_readmitted(reader, event) if {M.TOOL_ENTITY, "sys-capsule"} & set(deleted) else {}
    deleted_ids: set[str] = set()
    for entity, identifiers in deleted.items():
        ids = identifiers - readmitted.get(entity, set())
        if any(row["object_id"] in ids for row in (*reader.rows(entity), *projections.get(entity, ()))):
            raise AuthorityError("authority state deleted row reappeared")
        deleted_ids.update(ids)
    metadata = body["metadata_updates"]
    if (not isinstance(metadata, list) or any(not isinstance(item, Mapping)
            or not isinstance(item.get("object_id"), str) for item in metadata)
            or [item["object_id"] for item in metadata] != sorted({item["object_id"] for item in metadata})):
        raise AuthorityError("authority state metadata update roster differs")
    for item in metadata:
        if (not isinstance(item, Mapping) or set(item) != {"entity", "object_id", "fields", "before_sha256", "after_sha256"}
                or item["entity"] != "release-entity" or not isinstance(item["object_id"], str)
                or not isinstance(item["fields"], list) or not item["fields"]
                or any(not isinstance(name, str) for name in item["fields"])
                or item["fields"] != sorted(set(item["fields"])) or set(item["fields"]) - _METADATA_FIELDS
                or not _valid_hash(item["before_sha256"]) or not _valid_hash(item["after_sha256"])):
            raise AuthorityError("authority state metadata before/after proof differs")
        row = reader.row("release-entity", item["object_id"])
        fields = {name: row.get(name) for name in item["fields"]}
        if any(value is not None for value in fields.values()) or _hash(fields) != item["after_sha256"]:
            raise AuthorityError("authority state cleared native metadata reappeared or changed")
    for entity in TABLES:
        for row in (*reader.rows(entity), *projections.get(entity, ())):
            if row["object_id"] != event["object_id"] and _references(row, deleted_ids):
                raise AuthorityError("authority state retained row has a dangling removed-history reference")
    prior = [item for item in reader.events() if _order(item) < _order(event)]
    if any(item["kind"] in {"eval-attach", "outcome"} for item in prior):
        raise AuthorityError("authority state adopted an already-measured generation")
    if "toolkit_retirement" in body:
        from .toolkit_retirement import _archived, verify_detached

        verify_detached(body["toolkit_retirement"], _archived(reader.rows("release-entity"), reader.rows("sys-prompt")),
                        {row["object_id"]: row for row in reader.rows("sys-capsule")},
                        {row["object_id"]: row for row in reader.rows("sys-toolkit")}, reader.events())
    return event


class _NativeStateRead:
    """Version-only detached adapter shared by export/snapshot integrity readers."""

    def __init__(self, native: Mapping[str, Any]) -> None:
        self.native = native

    def rows(self, entity: str) -> tuple[Mapping[str, Any], ...]:
        return tuple(sorted(self.native.get(entity, ()), key=lambda row: row["object_id"]))

    def row(self, entity: str, identifier: str) -> Mapping[str, Any]:
        found = [row for row in self.rows(entity) if row["object_id"] == identifier]
        if len(found) != 1:
            raise AuthorityError(f"authority state {entity} {identifier} is absent or duplicated")
        return found[0]

    def events(self, kind: str | None = None) -> tuple[Mapping[str, Any], ...]:
        return tuple(event for event in sorted(self.rows("prompt-event-entity"), key=_order)
                     if kind is None or event["kind"] == kind)

    def epoch(self) -> Mapping[str, Any]:
        markers = [event for event in self.events("authority") if event["payload"].get("operation") == "mark"]
        if len(markers) != 1:
            raise AuthorityError("authority state detached rows lack an exact epoch marker")
        return markers[0]["payload"]["epoch"]

    def typed_index(self) -> Mapping[str, Mapping[str, Mapping[str, Any]]]:
        from .store import TYPED_TABLES

        return {entity: {row["object_id"]: row for row in self.rows(entity)} for entity in TYPED_TABLES}


class _AdmittedCatalogRead(_NativeStateRead):
    """Provide the already checked source admission without recursively reading head."""

    def __init__(self, reader: Any, event: Mapping[str, Any]) -> None:
        from .store import TABLES

        super().__init__({entity: reader.rows(entity) for entity in TABLES})
        self.admission = {**deepcopy(event["payload"]), "id": event["object_id"]}

    def current_state(self) -> Mapping[str, Any]:
        return self.admission


def _catalog_readmitted(reader: Any, event: Mapping[str, Any]) -> dict[str, set[str]]:
    """Accept content reuse only from a fully verified later publication request.

    The cleanup's removed roster proves physical deletion at admission time. A
    later source-audited catalog may reconstruct the same immutable Tool or
    Capsule. Its complete chain, gates, staged closure and native row equality
    must pass before those identities can be referenced again. A verified
    pending preparation also admits its partially written rows for exact resume.
    Deleted provenance records never receive this exception.
    """
    from ..tools_catalog import PUBLICATION_SCHEMA, catalog_history
    from . import entity_model as M

    if not any(item["payload"].get("schema") == PUBLICATION_SCHEMA and _order(item) > _order(event)
               for item in reader.events("register")):
        return {}
    try:
        publications, _deployments, pending = catalog_history(_AdmittedCatalogRead(reader, event), allow_pending=True)
    except (KeyError, TypeError, ValueError) as error:
        raise AuthorityError("authority state catalog readmission integrity refused") from error
    preparations = [record["prepared"] for record in publications.values()] + list(pending.values())
    admitted = {M.TOOL_ENTITY: set(), "sys-capsule": set()}
    for prepared in preparations:
        if prepared["payload"]["schema"] != PUBLICATION_SCHEMA or _order(prepared) <= _order(event):
            continue
        for entity, key in ((M.TOOL_ENTITY, "tool_rows"), ("sys-capsule", "capsule_rows")):
            admitted[entity].update(row["object_id"] for row in prepared["payload"][key])
    return admitted


def validate_native_state(native: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Verify a sealed snapshot's current-state admission using only native rows.

    Args:
        native: Complete detached version entity sets, with no evidence rows.

    Returns:
        Mapping[str, Any] | None: Verified admission event, absent on ordinary history.

    Raises:
        AuthorityError: The immutable closure or standalone admission is altered.
    """
    return validate_current_state(_NativeStateRead(native))


def apply_retirement(writer: Any, plan: RetirementPlan) -> Mapping[str, Any]:
    """Commit checked native deletions, metadata detachment and admission atomically.

    Args:
        writer: Already instance-fenced Authority writer holding its process lock.
        plan: Exact detached result of :func:`plan_retirement`.

    Returns:
        Mapping[str, Any]: Admission event and count of deleted native rows.

    Raises:
        AuthorityError: Stale preimage, active process/run, changed plan or failed postimage.
        Exception: Native transaction errors propagate after automatic rollback.
    """
    from .authority import Authority, AuthorityRead
    from .store import TABLES
    from .toolkit_retirement import active_processes

    if Authority.active_writer(writer.path) is not writer:
        raise AuthorityError("refresh retirement requires the active instance-bound writer lock")
    if active_processes():
        raise AuthorityError("refresh retirement requires idle evaluation, MCP, Lab and Dashboard processes")
    runs = writer.path.parent.parent / "runs"
    if runs.exists() and any(runs.rglob(".gateway-experiment.json")):
        raise AuthorityError("refresh retirement refuses an allocated run before manifest creation")
    for manifest in runs.rglob(".prompt_manifest.json") if runs.exists() else ():
        if any(manifest.parent.rglob(".gateway-experiment.json")):
            raise AuthorityError("refresh retirement refuses an allocated run")
        for path in manifest.parent.rglob("*.json"):
            document = json.loads(path.read_bytes())
            if isinstance(document, Mapping) and document.get("results"):
                raise AuthorityError("refresh retirement refuses measured or in-flight run evidence")
    with writer.reader() as reader:
        fresh = plan_retirement(reader, plan.expected, request_id=plan.anchor_payload["request_id"],
                                user_instruction=plan.anchor_payload["user_instruction"], actor=plan.anchor_payload["actor"])
        if fresh != plan:
            raise AuthorityError("refresh retirement plan changed from its checked exact preimage")
    database = writer.control.ws.backends.get("main").db
    with database.tx() as transaction:
        # The pinned HB database owns this physical transaction. Its DBAPI read view
        # is used only to validate uncommitted rows; all writes use Database.execute.
        connection = transaction._conn.connection.driver_connection
        if not connection.in_transaction:
            transaction.execute("BEGIN IMMEDIATE")
        original_factory = connection.row_factory
        import sqlite3

        connection.row_factory = sqlite3.Row
        try:
            reader = AuthorityRead(connection, writer.path, writer.binding)
            if _preimage_digest(reader, plan.expected) != plan.preimage_digest:
                raise AuthorityError("refresh retirement preimage changed inside the native transaction")
            for entity, ids in plan.deletions.items():
                for identifier in ids:
                    transaction.execute(f'DELETE FROM "{TABLES[entity]}" WHERE object_id = :id', {"id": identifier})
            for identifier, row in plan.release_updates.items():
                names = next(item["fields"] for item in plan.anchor_payload["metadata_updates"]
                             if item["object_id"] == identifier)
                fields = {key: row[key] for key in names}
                if fields:
                    assignments = ", ".join(f'"{key}" = :{key}' for key in fields)
                    transaction.execute(f'UPDATE "{TABLES["release-entity"]}" SET {assignments} WHERE object_id = :id',
                                        {**fields, "id": identifier})
            ts = max(time.time(), max(float(item["ts"]) for item in reader.events()) + 0.000001)
            body = {"kind": "register", "subject": plan.anchor_payload["release"], "payload": plan.anchor_payload,
                    "actor": plan.anchor_payload["actor"], "ts": ts}
            event = {"object_id": "e" + _hash(body)[:16], **body}
            transaction.execute(f'INSERT INTO "{TABLES["prompt-event-entity"]}" '
                                '(object_id,kind,subject,payload,actor,ts) VALUES (:id,:kind,:subject,:payload,:actor,:ts)',
                                {"id": event["object_id"], **body, "payload": canon(body["payload"])})
            validate_current_state(reader)
            checked = reader.verify(require_export=False)
            if not checked.ok:
                raise AuthorityError("refresh retirement postimage failed verification: " + "; ".join(checked.violations))
        finally:
            connection.row_factory = original_factory
    writer._batch_dirty = True
    return {"event": event["object_id"], "release": plan.expected.release,
            "deleted_count": sum(len(ids) for ids in plan.deletions.values())}
