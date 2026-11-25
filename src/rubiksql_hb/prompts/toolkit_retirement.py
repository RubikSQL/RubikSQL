"""Audit-only schema1 history and retained four-row Toolkit deletion receipts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Mapping, Sequence

from .canon import canon, sha256_hex

KIND = "toolkit-retirement"
SCHEMA = "rubiksql.toolkit-retirement/1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _archived(releases: Sequence[Mapping[str, Any]], prompts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    from ..hb_tool_entities import LEGACY_BUNDLE_SCHEMA, canonical, digest

    indexed = {(row["name"], row["manifest"]["meta"]["content_key"]): row for row in prompts}
    found = []
    for release in releases:
        key = release["refs"].get("tools.answering")
        if key is None:
            continue
        row = indexed[("tools.answering", key)]
        text = row["manifest"]["meta"]["text"]
        payload = json.loads(text)
        if payload.get("schema") != LEGACY_BUNDLE_SCHEMA:
            continue
        if canonical(payload) != text:
            raise ValueError("archived Toolkit carrier is not canonical JSON")
        found.append({"release": release.get("object_id", release.get("id")), "carrier": key,
                      "bundle_digest": digest(payload), "modules": payload["modules"],
                      "serializer": payload["serializer"], "toolkit_refs": payload["toolkits"]})
    return found


def verify_history(releases: Sequence[Mapping[str, Any]], prompts: Sequence[Mapping[str, Any]],
                   capsules: Mapping[str, Mapping[str, Any]], toolkits: Mapping[str, Mapping[str, Any]],
                   events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Verify archived preimages or the exact receipt for deleted Toolkit rows.

    This never restores or executes a Toolkit. Original completed publication
    candidates remain opaque audit bytes and supply the deleted full-row hashes.
    """
    from ..hb_tool_entities import _semantic, _validate_legacy_answering, digest, row_digest
    from .entity_model import CLOSURE_V2
    from .store import POINTER_KINDS

    archived = _archived(releases, prompts)
    anchors = [event for event in events if event["kind"] == "register"
               and event["payload"].get("schema") == "rubiksql.authority-state/1"
               and "toolkit_retirement" in event["payload"]]
    if anchors:
        if len(anchors) != 1 or any(event["kind"] == KIND for event in events):
            raise ValueError("Toolkit retirement has duplicate historical and detached admission proofs")
        verify_detached(anchors[0]["payload"]["toolkit_retirement"], archived, capsules, toolkits, events)
        return anchors[0]
    receipts = [event for event in events if event["kind"] == KIND]
    if not receipts:
        # Archives without a deletion receipt must retain their original Toolkit rows.
        indexed = {(row["name"], row["manifest"]["meta"]["content_key"]): row for row in prompts}
        for old in archived:
            payload = json.loads(indexed[("tools.answering", old["carrier"])]["manifest"]["meta"]["text"])
            _validate_legacy_answering(payload, capsule_rows=capsules, toolkit_rows=toolkits)
        return None
    if len(receipts) != 1 or len(archived) != 1 or toolkits:
        raise ValueError("Toolkit retirement needs one archived closure, one receipt and an empty Toolkit set")
    event = receipts[0]
    body = event["payload"]
    fields = {"schema", "user_instruction", "current", "epoch", "revision_before", "export_before",
              "archived", "deleted_toolkits"}
    if (set(body) != fields or body["schema"] != SCHEMA
            or not isinstance(body["user_instruction"], str) or not body["user_instruction"].strip()
            or body["archived"] != archived[0]):
        raise ValueError("Toolkit retirement receipt differs from its original archived closure")
    before = [item for item in events if (float(item["ts"]), str(item["object_id"]))
              < (float(event["ts"]), str(event["object_id"]))]
    marks = [item for item in before if item["kind"] == "authority" and item["payload"].get("operation") == "mark"]
    exports = [item for item in before if item["kind"] == "export"]
    if len(marks) != 1 or not exports or body["epoch"] != marks[0]["payload"]["epoch"]:
        raise ValueError("Toolkit retirement has no exact prior epoch/export")
    epoch = body["epoch"]
    moves = [item for item in before if (item["kind"] == "decision" and item["payload"].get("kind") in POINTER_KINDS)
             or (item["kind"] == "register" and item["payload"].get("schema") == "rubiksql.authority-state/1")]
    selected_release = (moves[-1]["payload"].get("to", moves[-1]["payload"].get("release")) if moves else epoch["release"])
    current = {"release": selected_release, "generation": selected_release,
               "decision": moves[-1]["object_id"] if moves else None}
    expected_revision = {"event_count": len(before), "latest_event_id": before[-1]["object_id"] if before else None}
    selected = next((row for row in releases if row.get("object_id", row.get("id")) == current["release"]), None)
    if (body["current"] != current or event["subject"] != current["release"]
            or body["revision_before"] != expected_revision
            or body["export_before"] != exports[-1]["payload"]["export_commit"]
            or selected is None or selected["rules"] != CLOSURE_V2
            or any(item["kind"] in {"eval-attach", "outcome"} for item in before)):
        raise ValueError("Toolkit retirement changed its unmeasured typed head or prior revision")
    original = archived[0]
    references = {ref["toolkit_id"]: ref for ref in original["toolkit_refs"]}
    removed = body["deleted_toolkits"]
    if (not isinstance(removed, list) or len(removed) != 4 or len(references) != 4
            or any(not isinstance(row, Mapping) or set(row) != {"object_id", "row_digest", "row_sha256"}
                   or not isinstance(row["row_sha256"], str) or not _HEX.fullmatch(row["row_sha256"]) for row in removed)
            or [row["object_id"] for row in removed] != sorted(references)):
        raise ValueError("Toolkit retirement does not identify exactly four deleted rows")
    publications = [item for item in before if item["kind"] == "publish"
                    and (item["payload"].get("body") or {}).get("release") == original["release"]]
    if len(publications) != 1:
        raise ValueError("Toolkit retirement has no unique original publication")
    prepared = [item for item in before if item["kind"] == "publish" and item["subject"] == publications[0]["subject"]
                and item["payload"].get("phase") == "prepared"]
    if len(prepared) != 1:
        raise ValueError("Toolkit retirement has no complete original prepared preimage")
    candidates = {row["object_id"]: row for row in prepared[0]["payload"]["candidates"].get("sys-toolkit", [])}
    if set(candidates) != set(references):
        raise ValueError("Toolkit retirement original Toolkit roster differs")
    for deleted in removed:
        row = candidates[deleted["object_id"]]
        if (deleted["row_sha256"] != sha256_hex(canon(row))
                or deleted["row_digest"] != references[deleted["object_id"]]["digest"]
                or digest(_semantic(row)) != deleted["row_digest"]):
            raise ValueError("Toolkit retirement deleted row hash differs from its original audit bytes")
    for reference in [*original["modules"], original["serializer"]]:
        capsule = capsules[reference["capsule_id"]]
        if row_digest(capsule, capsule=True) != reference["row_digest"]:
            raise ValueError("Toolkit retirement archived source Capsule differs")
    return event


def verify_detached(proof: Mapping[str, Any], archived: Sequence[Mapping[str, Any]],
                    capsules: Mapping[str, Mapping[str, Any]], toolkits: Mapping[str, Mapping[str, Any]],
                    events: Sequence[Mapping[str, Any]]) -> None:
    """Verify deleted Toolkit hashes against retained cutover bytes and source rows.

    The new state admission carries only the archive identity and deleted row
    hashes. It has no historical current pointer or revision to reinterpret.

    Args:
        proof: Detached ``archived`` and ``deleted_toolkits`` fields.
        archived: Canonical archived schema1 carrier descriptions.
        capsules: Retained source Capsule rows.
        toolkits: Live Toolkit rows; must be empty after retirement.
        events: Retained canonical events, including the original cutover preimage.

    Returns:
        None.

    Raises:
        ValueError: Any carrier, source row, cutover preimage or deleted hash differs.
    """
    from ..hb_tool_entities import _semantic, digest, row_digest

    if (set(proof) != {"archived", "deleted_toolkits"} or len(archived) != 1
            or proof["archived"] != archived[0] or toolkits):
        raise ValueError("detached Toolkit retirement archive or live Toolkit set differs")
    original = archived[0]
    references = {ref["toolkit_id"]: ref for ref in original["toolkit_refs"]}
    removed = proof["deleted_toolkits"]
    if (not isinstance(removed, list) or len(removed) != 4 or len(references) != 4
            or any(not isinstance(row, Mapping) or set(row) != {"object_id", "row_digest", "row_sha256"}
                   or not isinstance(row["row_sha256"], str) or not _HEX.fullmatch(row["row_sha256"]) for row in removed)
            or [row["object_id"] for row in removed] != sorted(references)):
        raise ValueError("detached Toolkit retirement does not identify exactly four deleted hashes")
    publications = [item for item in events if item["kind"] == "publish"
                    and (item["payload"].get("body") or {}).get("release") == original["release"]]
    prepared = [item for item in events if item["kind"] == "publish" and len(publications) == 1
                and item["subject"] == publications[0]["subject"] and item["payload"].get("phase") == "prepared"]
    if len(publications) != 1 or len(prepared) != 1:
        raise ValueError("detached Toolkit retirement has no exact retained cutover preimage")
    candidates = {row["object_id"]: row for row in prepared[0]["payload"]["candidates"].get("sys-toolkit", [])}
    if set(candidates) != set(references):
        raise ValueError("detached Toolkit retirement cutover Toolkit roster differs")
    for deleted in removed:
        row = candidates[deleted["object_id"]]
        if (deleted["row_sha256"] != sha256_hex(canon(row))
                or deleted["row_digest"] != references[deleted["object_id"]]["digest"]
                or digest(_semantic(row)) != deleted["row_digest"]):
            raise ValueError("detached Toolkit retirement deleted hash differs from retained cutover bytes")
    for reference in [*original["modules"], original["serializer"]]:
        capsule = capsules.get(reference["capsule_id"])
        if capsule is None or row_digest(capsule, capsule=True) != reference["row_digest"]:
            raise ValueError("detached Toolkit retirement archived source Capsule differs")


def active_processes() -> list[int]:
    """Return project application process ids; do not stop or print their arguments."""
    programs = {"eval_evolution.py", "eval_parallel.py", "eval_bird.py", "knowledge_build.py",
                "run_official_pipeline.py", "experiment_dashboard.py", "lab_server.py", "mcp_server.py"}
    modules = {"rubiksql_hb.server", "rubiksql_hb.lab_server"}
    result = subprocess.run(["ps", "-eo", "pid,args"], check=True, capture_output=True, text=True, timeout=10)
    found = []
    for line in result.stdout.splitlines()[1:]:
        pid, separator, arguments = line.strip().partition(" ")
        if not separator or not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            argv = shlex.split(arguments)
        except ValueError:
            continue
        if any(Path(value).name in programs for value in argv) or any(
                value == "-m" and index + 1 < len(argv) and argv[index + 1] in modules for index, value in enumerate(argv)):
            found.append(int(pid))
    return found
