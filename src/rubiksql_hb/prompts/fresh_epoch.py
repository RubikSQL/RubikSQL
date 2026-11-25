"""Source admission for one explicitly initialized, parentless Git epoch."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import re
from typing import Any

from .canon import canon, sha256_hex
from .types import AuthorityError, GitAuthorityBinding

SCHEMA = "rubiksql.fresh-epoch/1"
_FIELDS = {"schema", "operation", "phase", "release", "generation", "epoch", "namespace", "budget",
           "request_id", "user_instruction", "seed", "content_digest", "source_admission", "actor"}


def content_digest(release: Mapping[str, Any]) -> str:
    """Hash the exact prompt and typed references of a seeded release."""
    return sha256_hex(canon({key: release.get(key) for key in ("refs", "entries", "rules", "content_refs")}))


def validate_admission(reader: Any) -> Mapping[str, Any] | None:
    """Verify a fresh root's immutable authorization and real tool receipts.

    Args:
        reader: One native authority read view, or the native export adapter.

    Returns:
        Its unique seed admission event, or None for an older epoch.

    Raises:
        AuthorityError: The marker, seed, source evidence or root identity differs.
    """
    from ..tool_receipts import validate_tool_receipts
    from ..hb_tool_entities import typed_bundle
    from .entity_model import CLOSURE_V2
    from .store import release_document_id

    events = reader.events()
    admissions = [event for event in events if event["kind"] == "register" and event["payload"].get("schema") == SCHEMA]
    if not admissions:
        return None
    if len(admissions) != 1:
        raise AuthorityError("fresh epoch requires exactly one source admission")
    event, = admissions
    body = event["payload"]
    marker = reader.marker()
    if (set(body) != _FIELDS or body.get("operation") != "initialize" or body.get("phase") != "completed"
            or event["subject"] != body.get("release") or event["actor"] != body.get("actor")
            or marker.get("schema") != "rubiksql.authority/2" or marker.get("operation") != "mark"
            or body.get("epoch") != reader.epoch() or body.get("release") != reader.epoch()["release"]
            or body.get("generation") != body.get("release")
            or body.get("namespace") != "v0-" + str(body.get("release"))
            or body.get("budget") != {"used": 0, "cap": 10}
            or any(not isinstance(body.get(key), str) or not body[key].strip()
                   for key in ("request_id", "user_instruction", "actor"))
            or any(body.get(key) != marker.get(key) for key in ("request_id", "user_instruction", "seed"))):
        raise AuthorityError("fresh epoch admission differs from its exact request or marker")
    seed = body["seed"]
    if (not isinstance(seed, Mapping) or not isinstance(seed.get("source_manifest"), Mapping)
            or seed.get("source_scope_digest") != sha256_hex(canon(seed["source_manifest"]))
            or any(not isinstance(seed.get(key), str) or re.fullmatch(r"[0-9a-f]{40}", seed[key]) is None
                   for key in ("source_commit", "execution_source_commit"))
            or seed.get("source_commit") != body["epoch"]["source_commit"]):
        raise AuthorityError("fresh epoch seed source identity differs")
    release = reader.row("release-entity", body["release"])
    if (release["rules"] != CLOSURE_V2
            or release_document_id(release, epoch=body["epoch"]["epoch"]) != body["release"]
            or body["content_digest"] != content_digest(release)
            or seed.get("release") != body["release"] or seed.get("epoch") != body["epoch"]
            or seed.get("content_digest") != body["content_digest"]):
        raise AuthorityError("fresh epoch root content identity differs")
    definition = {key: release.get(key) for key in ("refs", "entries", "rules", "content_refs")}
    expected_seed = sha256_hex(canon({"source_commit": seed["source_commit"], "epoch": body["epoch"],
                                     "release": definition, "source_scope_digest": seed["source_scope_digest"]}))
    if seed.get("seed_digest") != expected_seed:
        raise AuthorityError("fresh epoch seed digest differs from its exact definitions")
    proof = body["source_admission"]
    if not isinstance(proof, Mapping) or set(proof) != {"gate", "parent_bundle_digest"} or proof["parent_bundle_digest"] is not None:
        raise AuthorityError("fresh root source admission requires a parentless tool gate")
    gate = proof["gate"]
    validate_tool_receipts(typed_bundle(release["content_refs"]), gate, parent_bundle=None, launch=False)
    if gate.get("source_commit") != seed["execution_source_commit"]:
        raise AuthorityError("fresh root gate differs from its reviewed executing source")
    for key in ("review", "invariants"):
        report = gate[key]["report"]
        if (any(report.get(field) != seed.get(field) for field in ("seed_digest", "source_scope_digest"))
                or report.get("source_commit") != seed["execution_source_commit"]):
            raise AuthorityError("fresh root evidence differs from its exact preview scope")
    before = [item for item in events if (item["ts"], item["object_id"]) < (event["ts"], event["object_id"])]
    if (any(item["kind"] not in {"publish", "authority", "gate"} for item in before)
            or any(item["kind"] == "publish" and item["payload"].get("body") is not None for item in before)
            or any(item["kind"] == "register" and item["payload"].get("schema") == "rubiksql.authority-state/1"
                   for item in events)):
        raise AuthorityError("fresh epoch imports prior experiment or proposal history")
    return event


def admit_seed(writer: Any, seed: Any, gate: Mapping[str, Any], *, request_id: str,
               user_instruction: str) -> str:
    """Admit freshly materialized definitions without importing prior history."""
    from ..hb_tool_entities import typed_bundle
    from ..tool_receipts import scan_tool_texts, validate_tool_receipts
    from .workspace_state import EVIDENCE_TABLES

    writer._require_writer()
    if not isinstance(writer.binding, GitAuthorityBinding):
        raise AuthorityError("fresh initialization requires the explicit schema-2 Git binding")
    validate_tool_receipts(typed_bundle(seed.release["content_refs"]), gate, parent_bundle=None, launch=False)
    with writer.reader() as reader:
        if reader.rows("proposal-entity") or any(reader.workspace_rows(entity) for entity in EVIDENCE_TABLES):
            raise AuthorityError("fresh epoch materialization requires empty experiment and QA evidence")
        if any(reader.events(kind) for kind in ("eval-reserve", "eval-claim", "eval-ready", "eval-attach", "outcome", "decision")):
            raise AuthorityError("fresh epoch materialization refuses historical accounting")
    actor = "agent:META/fresh-epoch"
    body = {"schema": SCHEMA, "operation": "initialize", "phase": "completed", "release": seed.epoch["release"],
            "generation": seed.epoch["release"], "epoch": dict(seed.epoch), "namespace": "v0-" + seed.epoch["release"],
            "budget": {"used": 0, "cap": 10}, "request_id": request_id, "user_instruction": user_instruction,
            "seed": deepcopy(dict(seed.identity)), "content_digest": content_digest(seed.release),
            "source_admission": {"gate": deepcopy(dict(gate)), "parent_bundle_digest": None}, "actor": actor}
    scan_tool_texts([canon(body)])
    writer._event("gate", seed.epoch["release"], dict(gate), actor)
    result = writer._event("register", seed.epoch["release"], body, actor)
    with writer.reader() as reader:
        validate_admission(reader)
    return result


def publisher_gate(seed: Any, receipts: Mapping[str, Any], *, checkout: Path) -> Mapping[str, Any]:
    """Run the normal merged-source publisher audit over a reviewed original seed."""
    from ..hb_tool_entities import typed_bundle
    from ..tool_receipts import publish_tool_gate

    built = seed.answering
    capsules = {row["object_id"]: row for row in built.capsule_rows}
    sources = {ref["module"]: capsules[ref["capsule_id"]]["manifest"]["source"]["source"]
               for ref in built.content["loader"]["modules"]}
    serializer = capsules[built.content["loader"]["serializer"]["capsule_id"]]["manifest"]["source"]["source"]
    for key in ("review", "invariants"):
        report = receipts.get(key, {}).get("report")
        if (not isinstance(report, Mapping) or any(report.get(field) != seed.identity.get(field)
                for field in ("seed_digest", "source_scope_digest"))
                or report.get("source_commit") != seed.identity["execution_source_commit"]):
            raise AuthorityError("root review and invariant receipts must bind the exact preview scope")
    return publish_tool_gate(checkout, typed_bundle(built.content), module_sources=sources,
        parent_payload=None, parent_sources=None, parent_contracts=None, parent_source_commit=seed.identity["source_commit"],
        receipts=receipts, mode="git", contracts=list(built.definitions), serializer_source=serializer,
        tiers={}, root_seed_commit=seed.identity["source_commit"])


def validate_native_admission(native: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Apply the same admission verifier to the closed export's native rows."""
    from .authority_state import _NativeStateRead

    class SeedRead(_NativeStateRead):
        def marker(self) -> Mapping[str, Any]:
            marks = [event for event in self.events("authority") if event["payload"].get("operation") == "mark"]
            if len(marks) != 1:
                raise AuthorityError("fresh native rows lack their unique marker")
            return marks[0]["payload"]

    return validate_admission(SeedRead(native))
