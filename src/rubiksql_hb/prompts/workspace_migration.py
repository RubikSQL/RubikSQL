"""Canonical evidence validation and retained historical migration receipts."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .canon import sha256_hex

if TYPE_CHECKING:
    from .authority import AuthorityRead

SCHEMA_MIGRATION = "rubiksql.lab-migration/1"
#: Rows that passed every per-row field contract in this process, by entity class and
#: full-row content digest. Field validation is a pure function of both, so any changed
#: row is checked again; duplicates and references are checked per call.
_VALID_ROWS: set[tuple[type, str]] = set()
_VALID_ROWS_LIMIT = 200_000


def _digest(value: Any) -> str:
    return sha256_hex(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def _validate_rows(rows: Mapping[str, Sequence[Mapping[str, Any]]], entities: Mapping[str, type]) -> None:
    from .authority import AuthorityError

    for entity, values in rows.items():
        entity_cls = entities[entity]
        schema = entity_cls.schema()
        seen: set[str] = set()
        for row in values:
            if set(row) != set(schema.fields):
                raise AuthorityError(f"lab migration {entity} row does not contain its complete canonical fields")
            object_id = row["object_id"]
            if not isinstance(object_id, str) or not object_id.strip() or object_id in seen:
                raise AuthorityError(f"lab migration {entity} has an empty or duplicate object id")
            seen.add(object_id)
            try:
                content = (entity_cls, _digest(row))
            except (TypeError, ValueError):
                content = None
            if content in _VALID_ROWS:
                continue
            for name, field in schema.fields.items():
                value = row[name]
                if value is None:
                    if field.required or entities[entity]._hb_specs[name].computes:
                        raise AuthorityError(f"lab migration {entity}.{name} is null but cannot be preserved on insertion")
                    continue
                try:
                    normalized = field.dtype.validate(value)
                    field.dtype.encode(normalized)
                    # A value returned unchanged serializes identically once its whole row does.
                    exact = (content is not None and normalized is value) or _digest(value) == _digest(normalized)
                except (TypeError, ValueError, OverflowError):
                    raise AuthorityError(f"lab migration {entity}.{name} fails its canonical field contract") from None
                if not exact:
                    raise AuthorityError(f"lab migration {entity}.{name} would change during native insertion")
            if content is None:
                raise AuthorityError(f"lab migration {entity} row is not finite canonical JSON")
            if len(_VALID_ROWS) >= _VALID_ROWS_LIMIT:
                _VALID_ROWS.clear()
            _VALID_ROWS.add(content)


def _validate_references(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    from heavenbase.utils import hash_id

    from ..lab_identity import qa_object_id
    from .authority import AuthorityError

    indexed = {entity: {row["object_id"]: row for row in values} for entity, values in rows.items()}
    questions = indexed["q-a-entity"]
    traces, sessions, messages = indexed["trace-entity"], indexed["agent-session"], indexed["agent-message"]

    def require(entity: str, object_id: Any, context: str) -> Mapping[str, Any]:
        if not isinstance(object_id, str) or object_id not in indexed[entity]:
            raise AuthorityError(f"lab migration {context} references a missing {entity}")
        return indexed[entity][object_id]

    trace_ids: dict[tuple[Any, Any], str] = {}

    def trace_identity(exp_id: Any, qid: Any) -> str:
        # hash_id is pure: a trace, its candidates and its judge share one derivation.
        if (exp_id, qid) not in trace_ids:
            trace_ids[exp_id, qid] = hash_id(exp_id, qid)
        return trace_ids[exp_id, qid]

    for row in sessions.values():
        if (not isinstance(row["state"], Mapping) or not isinstance(row["message_ids"], list)
                or any(not isinstance(value, str) for value in row["message_ids"])):
            raise AuthorityError("lab migration session state or message references are malformed")
    for row in indexed["experiment-entity"].values():
        if row["parent_experiment"] not in (None, ""):
            require("experiment-entity", row["parent_experiment"], "experiment")
    for row in questions.values():
        try:
            valid = row["object_id"] == qa_object_id(row["dataset_version"], row["database_version"], row["name"])
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise AuthorityError("lab migration QA identity differs from its exact dataset/database/question versions")
    for row in traces.values():
        require("experiment-entity", row["exp_id"], "trace")
        qa = require("q-a-entity", row["qa_id"], "trace")
        if (row["object_id"] != trace_identity(row["exp_id"], row["qid"])
                or (row["qid"], row["db_id"]) != (qa["name"], qa["db_id"])):
            raise AuthorityError("lab migration trace identity or exact QA reference differs")
        session_ids = row["session_ids"]
        if session_ids is not None:
            if len(set(session_ids)) != len(session_ids):
                raise AuthorityError("lab migration trace repeats a session reference")
            for session_id in session_ids:
                session = require("agent-session", session_id, "trace")
                state = session["state"]
                if (state.get("exp_id"), state.get("qid"), state.get("db_id")) != (row["exp_id"], row["qid"], row["db_id"]):
                    raise AuthorityError("lab migration trace and referenced session disagree")
    for entity in ("candidate-entity", "judge-entity"):
        for row in indexed[entity].values():
            require("experiment-entity", row["exp_id"], entity)
            trace_id = trace_identity(row["exp_id"], row["qid"])
            trace = require("trace-entity", trace_id, entity)
            stable_id = hash_id(row["exp_id"], row["qid"], row["slot"]) if entity == "candidate-entity" else trace_id
            if row["object_id"] != stable_id or row["db_id"] != trace["db_id"]:
                raise AuthorityError(f"lab migration {entity} identity or trace reference differs")
            if row["session_id"] is not None:
                session = require("agent-session", row["session_id"], entity)
                state = session["state"]
                if ((state.get("exp_id"), state.get("qid"), state.get("db_id")) != (row["exp_id"], row["qid"], row["db_id"])
                        or session["agent_ref"] != ("candidate" if entity == "candidate-entity" else "judge")
                        or entity == "candidate-entity" and state.get("slot") != row["slot"]):
                    raise AuthorityError(f"lab migration {entity} and referenced session disagree")
    for row in sessions.values():
        state = row["state"]
        if state.get("exp_id") is not None:
            require("experiment-entity", state["exp_id"], "session")
        if state.get("parent_session_id") is not None:
            require("agent-session", state["parent_session_id"], "session")
        ids = row["message_ids"]
        if any(not isinstance(value, str) for value in ids) or len(set(ids)) != len(ids):
            raise AuthorityError("lab migration session has invalid or duplicate message references")
        for idx, message_id in enumerate(ids):
            message = require("agent-message", message_id, "session")
            if (message["session_id"], message["idx"]) != (row["object_id"], idx):
                raise AuthorityError("lab migration ordered session/message reference differs")
    for row in messages.values():
        session = require("agent-session", row["session_id"], "message")
        if (not isinstance(row["idx"], int) or isinstance(row["idx"], bool) or row["idx"] < 0
                or row["idx"] >= len(session["message_ids"]) or session["message_ids"][row["idx"]] != row["object_id"]
                or not isinstance(row["payload"], Mapping)):
            raise AuthorityError("lab migration message is absent from its session's ordered references")


def validate_evidence(rows_by_entity: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """Validate complete canonical evidence fields, identities and references.

    Args:
        rows_by_entity: All seven canonical evidence sets, including empty sets.

    Raises:
        AuthorityError: Missing or extra entities, malformed native fields, invalid
            stable IDs or a dangling or inconsistent evidence reference. Errors
            name entity/field contracts without including question or gold text.
    """
    from ..lab_schema import canonical_evidence_entities
    from .authority import AuthorityError

    entities = canonical_evidence_entities()
    if set(rows_by_entity) != set(entities):
        raise AuthorityError("canonical evidence validation requires exactly the seven evidence entity sets")
    _validate_rows(rows_by_entity, entities)
    _validate_references(rows_by_entity)


def pending_migration(reader: AuthorityRead) -> Mapping[str, Any] | None:
    """Return the sole exact prepared lab import that lacks a completion receipt.

    Args:
        reader: Stable authority view, including operational migration events.

    Returns:
        The full metadata preparation event, or ``None`` when no import is pending.

    Raises:
        AuthorityError: Unknown, inconsistent or competing migration checkpoints.
    """
    from .authority import AuthorityError

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for event in reader.events("migration"):
        payload = event["payload"]
        if payload.get("schema") != SCHEMA_MIGRATION or payload.get("phase") not in {"prepared", "complete"}:
            raise AuthorityError("lab migration has an unknown durable request receipt")
        request = {key: payload.get(key) for key in ("schema", "source_path", "source_digest", "user_instruction", "actor")}
        if (not isinstance(request["user_instruction"], str) or not request["user_instruction"].strip()
                or not isinstance(request["actor"], str) or not request["actor"].strip()
                or event["actor"] != request["actor"] or event["subject"] != _digest(request)):
            raise AuthorityError("lab migration durable request identity or authorization differs")
        groups.setdefault(event["subject"], []).append(event)
    pending = []
    for events in groups.values():
        prepared = [event for event in events if event["payload"]["phase"] == "prepared"]
        completed = [event for event in events if event["payload"]["phase"] == "complete"]
        if len(prepared) != 1 or len(completed) > 1 or completed and events.index(completed[0]) < events.index(prepared[0]):
            raise AuthorityError("lab migration has inconsistent durable request receipts")
        if not completed:
            pending.append(prepared[0])
    if len(pending) > 1:
        raise AuthorityError("lab migration history has multiple incomplete requests")
    return pending[0] if pending else None
