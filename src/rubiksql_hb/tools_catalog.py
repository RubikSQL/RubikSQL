"""Immutable catalog registrations, independent consumer pins, and inert verification.

The catalog shares native authority rows with BIRD. Its register events never
create a BIRD proposal, release, or pointer decision.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import re
from typing import Any

from .hb_tool_entities import digest, typed_bundle, validate_typed_tools
from .prompts import entity_model as M
from .prompts.canon import canon, sha256_hex
from .prompts.types import AuthorityError
from .tool_definitions import select_definitions

PUBLICATION_SCHEMA = "rubiksql.tool-publication/1"
DEPLOYMENT_SCHEMA = "rubiksql.tool-deployment/1"
CONSUMERS = ("assistant", "meta")
_SCHEMAS = {PUBLICATION_SCHEMA, DEPLOYMENT_SCHEMA}
_PUB_FIELDS = {"schema", "id", "request", "content", "content_digest", "source", "gate", "parent_provenance"}
_DEP_FIELDS = {"schema", "id", "request", "consumer", "publication_id", "publication_digest", "tools", "previous_deployment"}


def registration_id(document: Mapping[str, Any]) -> str:
    """Return the exact immutable identity of a registration document.

    Args:
        document (Mapping[str, Any]): Publication or deployment semantic fields.

    Returns:
        str: Content addressed publication or deployment identity.

    Raises:
        AuthorityError: The schema is unsupported.
    """
    if document.get("schema") not in _SCHEMAS:
        raise AuthorityError("unsupported tool registration schema")
    prefix = "tp" if document["schema"] == PUBLICATION_SCHEMA else "td"
    return prefix + digest({key: value for key, value in document.items() if key != "id"})[:32]


def _document(payload: Mapping[str, Any]) -> dict[str, Any]:
    fields = _PUB_FIELDS if payload.get("schema") == PUBLICATION_SCHEMA else _DEP_FIELDS
    return {key: deepcopy(payload[key]) for key in fields if key in payload}


def _check_document_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema = payload.get("schema")
    fields = _PUB_FIELDS if schema == PUBLICATION_SCHEMA else _DEP_FIELDS
    extra = {"phase", "tool_rows", "capsule_rows"} if payload.get("phase") == "prepared" else {"phase", "prepared_event"}
    if schema not in _SCHEMAS or set(payload) != fields | extra:
        raise AuthorityError("tool registration has missing or unsupported fields")
    if payload.get("phase") not in {"prepared", "completed"}:
        raise AuthorityError("tool registration has unsupported phase")
    doc = _document(payload)
    if doc["id"] != registration_id(doc):
        raise AuthorityError("tool registration content identity differs")
    request = doc["request"]
    if (not isinstance(request, Mapping) or not isinstance(request.get("request_id"), str)
            or not request["request_id"].strip() or not isinstance(request.get("actor"), str)
            or not request["actor"].strip() or not isinstance(request.get("expected"), Mapping)):
        raise AuthorityError("tool registration lacks exact request facts")
    expected = request["expected"]
    if (set(expected) != {"release", "generation", "decision", "revision"}
            or not isinstance(expected["revision"], Mapping)
            or set(expected["revision"]) != {"event_count", "latest_event_id"}
            or not isinstance(expected["revision"]["event_count"], int)
            or isinstance(expected["revision"]["event_count"], bool) or expected["revision"]["event_count"] < 0):
        raise AuthorityError("tool registration expected facts are malformed")
    if (any(not isinstance(expected[key], str) or not expected[key] for key in ("release", "generation"))
            or expected["decision"] is not None and not isinstance(expected["decision"], str)
            or expected["revision"]["latest_event_id"] is not None
            and not isinstance(expected["revision"]["latest_event_id"], str)):
        raise AuthorityError("tool registration expected selection is malformed")
    if schema == PUBLICATION_SCHEMA:
        if (set(request) != {"operation", "request_id", "actor", "expected", "content_digest", "receipts_sha256", "source"}
                or request.get("operation") != "publish-tools" or set(doc["content"]) != {"tools", "loader"}):
            raise AuthorityError("tool publication carries unsupported content")
        if doc["content_digest"] != digest(doc["content"]) or doc["content_digest"] != request["content_digest"]:
            raise AuthorityError("tool publication content digest differs")
        if (not isinstance(request.get("receipts_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", request["receipts_sha256"]) is None):
            raise AuthorityError("tool publication lacks its exact input receipt digest")
        source = doc["source"]
        if (not isinstance(source, Mapping) or set(source) != {"mode", "source_commit", "deployed_commit"}
                or source.get("mode") not in {"git", "hb"}
                or (source["mode"] == "git") != bool(source.get("source_commit"))
                or not isinstance(source.get("deployed_commit"), str)
                or re.fullmatch(r"[0-9a-f]{40}", source["deployed_commit"]) is None
                or source["source_commit"] is not None and re.fullmatch(r"[0-9a-f]{40}", str(source["source_commit"])) is None
                or source != request["source"]):
            raise AuthorityError("tool publication source provenance is malformed")
        from .tool_receipts import validate_catalog_receipts

        validate_catalog_receipts(typed_bundle(doc["content"]), doc["gate"], parent_provenance=doc["parent_provenance"])
        if (doc["gate"].get("mode") != source["mode"]
                or doc["gate"].get("source_commit") != source["source_commit"]
                or doc["gate"].get("audit_implementation_commit") != source["deployed_commit"]):
            raise AuthorityError("tool publication source and audit differ")
    elif (set(request) != {"operation", "request_id", "actor", "expected", "consumer", "publication_id", "previous_deployment"}
          or request.get("operation") != "deploy-tools" or doc["consumer"] not in CONSUMERS
          or not isinstance(doc["publication_id"], str) or not isinstance(doc["tools"], list)
          or not doc["tools"] or (doc["previous_deployment"] is not None
                                and not isinstance(doc["previous_deployment"], str))):
        raise AuthorityError("tool deployment consumer or pin is invalid")
    elif any(doc[key] != request[key] for key in ("consumer", "publication_id", "previous_deployment")):
        raise AuthorityError("tool deployment differs from its exact request")
    return doc


def _check_document(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return _check_document_fields(payload)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise AuthorityError(f"tool registration integrity refused: {error}") from error


def _check_event(event: Mapping[str, Any]) -> None:
    body = {key: value for key, value in event.items() if key != "object_id"}
    if (event.get("kind") != "register"
            or event.get("object_id") != "e" + sha256_hex(canon(body))[:16]
            or event.get("subject") != event.get("payload", {}).get("id")):
        raise AuthorityError("tool registration event identity differs")


def _record(prepared: Mapping[str, Any], completed: Mapping[str, Any]) -> dict[str, Any]:
    _check_event(prepared)
    _check_event(completed)
    prior, doc = _check_document(prepared["payload"]), _check_document(completed["payload"])
    if (prior != doc or completed["payload"]["phase"] != "completed"
            or prepared["payload"]["phase"] != "prepared"
            or completed["payload"]["prepared_event"] != prepared["object_id"]
            or completed["actor"] != doc["request"]["actor"] or prepared["actor"] != completed["actor"]):
        raise AuthorityError("tool registration completion differs from its prepared request")
    return {**deepcopy(completed["payload"]), "event": deepcopy(completed), "prepared": deepcopy(prepared)}


def selected_refs(content: Mapping[str, Any], definitions: Sequence[Mapping[str, Any]], consumer: str) -> list[dict[str, Any]]:
    """Select ordered catalog references using their consumer permissions.

    Args:
        content (Mapping[str, Any]): Verified complete catalog content.
        definitions (Sequence[Mapping[str, Any]]): Its verified callable definitions.
        consumer (str): ``assistant`` or ``meta``.

    Returns:
        list[dict[str, Any]]: Exact permitted Tool references in serve order.

    Raises:
        AuthorityError: The consumer is unsupported or has no permitted tools.
    """
    if consumer not in CONSUMERS:
        raise AuthorityError("independent tools deploy only to assistant or meta")
    names = {row["name"] for row in select_definitions(definitions, consumer)}
    selected = [deepcopy(ref) for ref, definition in zip(content["tools"], definitions, strict=True)
                if definition["name"] in names]
    if not selected:
        raise AuthorityError("tool deployment has no permitted consumer tools")
    return selected


def validate_resolved_tools(resolved: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Verify a detached completed deployment before executable imports.

    Args:
        resolved (Mapping[str, Any]): Exact deployment/publication and object-id
            maps of Tool and Capsule rows returned by the authority reader.

    Returns:
        tuple[dict[str, Any], ...]: Complete catalog definitions in serve order.

    Raises:
        AuthorityError: Event, source, audit, row or role-selection integrity differs.
    """
    try:
        publication, deployment = resolved["publication"], resolved["deployment"]
        for record in (publication, deployment):
            verified = _record(record["prepared"], record["event"])
            if verified != record:
                raise AuthorityError("detached tool registration differs from its events")
        if publication["schema"] != PUBLICATION_SCHEMA or deployment["schema"] != DEPLOYMENT_SCHEMA:
            raise AuthorityError("detached tool registration has the wrong schema")
        definitions = validate_typed_tools(publication["content"], tool_rows=resolved["tool_rows"],
                                           capsule_rows=resolved["capsule_rows"])
        source_refs = {ref["capsule_id"] for ref in publication["content"]["loader"]["modules"]}
        source_refs.add(publication["content"]["loader"]["serializer"]["capsule_id"])
        if (set(resolved["tool_rows"]) != {ref["object_id"] for ref in publication["content"]["tools"]}
                or set(resolved["capsule_rows"]) != source_refs):
            raise AuthorityError("detached catalog includes unreferenced or incomplete native rows")
        if (deployment["publication_id"] != publication["id"]
                or deployment["publication_digest"] != publication["content_digest"]
                or deployment["tools"] != selected_refs(publication["content"], definitions, deployment["consumer"])):
            raise AuthorityError("tool deployment does not select its exact permitted publication")
        return definitions
    except (KeyError, TypeError, ValueError) as error:
        raise AuthorityError(f"detached tool deployment integrity refused: {error}") from error


def catalog_parent(reader: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Resolve the actual catalog parent or the initial admitted BIRD tool closure.

    Args:
        reader (Any): Stable native authority read transaction.

    Returns:
        tuple[dict[str, Any], dict[str, Any], str]: Parent provenance, typed
            content and exact source commit of its admitting audit.

    Raises:
        AuthorityError: No verified source admission identifies the parent.
    """
    publications, _deployments, pending = catalog_history(reader, allow_pending=True)
    if pending:
        raise AuthorityError("complete the exact prepared tool registration first")
    if publications:
        publication = next(reversed(publications.values()))
        commit = publication["source"]["source_commit"] or publication["source"]["deployed_commit"]
        return ({"kind": "catalog", "publication_id": publication["id"],
                 "content_digest": publication["content_digest"], "source_commit": commit},
                deepcopy(publication["content"]), commit)
    release = reader.row("release-entity", reader.head().release)
    if release["rules"] != M.CLOSURE_V2:
        raise AuthorityError("tool catalog requires an admitted typed BIRD tool parent")
    from .prompts.authority import Authority

    content = {key: deepcopy(release["content_refs"][key]) for key in ("tools", "loader")}
    commit = Authority._typed_parent_commit(reader, digest(typed_bundle(content)))
    return ({"kind": "bird", "release": release["object_id"], "content_digest": digest(content),
             "source_commit": commit}, content, commit)


def catalog_history(reader: Any, *, allow_pending: bool = False) -> tuple[dict, dict, dict]:
    """Verify catalog publication and consumer chains in one native transaction.

    Args:
        reader (Any): Native authority reader with stable events and rows.
        allow_pending (bool): Validate prepared rows but permit their exact
            interrupted completion for writer recovery.

    Returns:
        tuple[dict, dict, dict]: Ordered completed publications, deployments,
            and pending prepared events, each indexed by immutable registration id.

    Raises:
        AuthorityError: Malformed content, provenance, chain, request reuse, or
            incomplete registration when pending recovery is not requested.
    """
    from .prompts.authority import Authority, _revision, semantic_row
    from .prompts.store import POINTER_KINDS

    publications, deployments, prepared, pending, requests, latest = {}, {}, {}, {}, {}, {}
    tools = {row["object_id"]: row for row in reader.rows(M.TOOL_ENTITY)}
    capsules = {row["object_id"]: row for row in reader.rows("sys-capsule")}
    prior_events = []
    selection = {"release": str(reader.epoch()["release"]), "generation": str(reader.epoch()["release"]), "decision": None}
    for event in reader.events():
        payload = event["payload"]
        if event["kind"] != "register" or payload.get("schema") not in _SCHEMAS:
            if event["kind"] == "decision" and payload.get("kind") in POINTER_KINDS:
                selection = {"release": payload["to"], "generation": payload["generation"], "decision": event["object_id"]}
            elif event["kind"] == "register" and payload.get("schema") in {"rubiksql.authority-state/1", "rubiksql.fresh-epoch/1"}:
                selection = {"release": payload["release"], "generation": payload["generation"], "decision": event["object_id"]}
            prior_events.append(event)
            continue
        _check_event(event)
        doc = _check_document(payload)
        identifier, request = doc["id"], doc["request"]
        if payload["phase"] == "prepared":
            if pending or identifier in prepared or request["request_id"] in requests:
                raise AuthorityError("tool registration duplicates or interrupts another exact request")
            expected = request["expected"]
            if any(expected[key] != value for key, value in selection.items()):
                raise AuthorityError("tool registration expected BIRD selection differs from its preparation")
            if expected.get("revision") != {"event_count": _revision(prior_events).event_count,
                                             "latest_event_id": _revision(prior_events).latest_event_id}:
                raise AuthorityError("tool registration expected revision does not bind its preparation")
            if event["actor"] != request["actor"]:
                raise AuthorityError("tool registration actor differs from its request")
            if doc["schema"] == PUBLICATION_SCHEMA:
                staged_tools = {row["object_id"]: row for row in payload["tool_rows"]}
                staged_capsules = {row["object_id"]: row for row in payload["capsule_rows"]}
                if len(staged_tools) != len(payload["tool_rows"]) or len(staged_capsules) != len(payload["capsule_rows"]):
                    raise AuthorityError("prepared catalog has duplicate rows")
                validate_typed_tools(doc["content"], tool_rows=staged_tools, capsule_rows=staged_capsules)
                refs = {ref["object_id"] for ref in doc["content"]["tools"]}
                source_refs = {ref["capsule_id"] for ref in doc["content"]["loader"]["modules"]}
                source_refs.add(doc["content"]["loader"]["serializer"]["capsule_id"])
                if set(staged_tools) != refs or set(staged_capsules) != source_refs:
                    raise AuthorityError("prepared catalog has unreferenced or incomplete rows")
                for held, staged in ((tools, staged_tools), (capsules, staged_capsules)):
                    if any(key in held and semantic_row(held[key]) != semantic_row(row) for key, row in staged.items()):
                        raise AuthorityError("prepared catalog collides with an immutable row")
                provenance = doc["parent_provenance"]
                if publications:
                    parent = next(reversed(publications.values()))
                    expected_parent = {"kind": "catalog", "publication_id": parent["id"],
                                       "content_digest": parent["content_digest"],
                                       "source_commit": parent["source"]["source_commit"] or parent["source"]["deployed_commit"]}
                    parent_content = parent["content"]
                else:
                    release = reader.row("release-entity", str(provenance.get("release")))
                    if release["rules"] != M.CLOSURE_V2 or release["object_id"] != expected.get("release"):
                        raise AuthorityError("initial catalog parent is not the prepared BIRD head")
                    parent_content = {key: release["content_refs"][key] for key in ("tools", "loader")}
                    expected_parent = {"kind": "bird", "release": release["object_id"],
                                       "content_digest": digest(parent_content),
                                       "source_commit": Authority._typed_parent_commit(reader, digest(typed_bundle(parent_content)))}
                if provenance != expected_parent or doc["gate"].get("parent_bundle_digest") != digest(typed_bundle(parent_content)):
                    raise AuthorityError("catalog audit does not bind the actual source parent")
            else:
                if payload["tool_rows"] or payload["capsule_rows"]:
                    raise AuthorityError("deployment preparation cannot create native versions")
                publication = publications.get(doc["publication_id"])
                if publication is None or doc["previous_deployment"] != latest.get(doc["consumer"]):
                    raise AuthorityError("tool deployment publication or previous consumer pin is stale")
                definitions = validate_typed_tools(publication["content"], tool_rows=tools, capsule_rows=capsules)
                if (doc["publication_digest"] != publication["content_digest"]
                        or doc["tools"] != selected_refs(publication["content"], definitions, doc["consumer"])):
                    raise AuthorityError("tool deployment exact role selection differs")
            prepared[identifier], pending[identifier] = event, event
            requests[request["request_id"]] = identifier
        else:
            if set(pending) != {identifier}:
                raise AuthorityError("tool registration has no unique pending preparation")
            if prior_events[-1]["object_id"] != pending[identifier]["object_id"]:
                raise AuthorityError("another event interrupted the prepared tool registration")
            record = _record(pending.pop(identifier), event)
            if doc["schema"] == PUBLICATION_SCHEMA:
                validate_typed_tools(doc["content"], tool_rows=tools, capsule_rows=capsules)
                publications[identifier] = record
            else:
                deployments[identifier] = record
                latest[doc["consumer"]] = identifier
        prior_events.append(event)
    if pending and not allow_pending:
        raise AuthorityError("tool registration " + next(iter(pending)) + " is incomplete")
    return publications, deployments, pending
