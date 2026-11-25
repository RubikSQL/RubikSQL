"""Publication and launch receipts for native answering-code changes."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

from .hb_tool_entities import BUNDLE_SCHEMA, TYPED_BUNDLE_SCHEMA, canonical, digest, source_digest

_KEY_LIKE = re.compile(r"(?:sk-[A-Za-z0-9_-]{16,}|(?:api[_-]?key|access[_-]?token|secret)\s*[=:]\s*[\"']?[A-Za-z0-9_-]{24,})", re.I)
# The trusted parent compares every added source line with the base evolution set and all 1534 records.
# Profiling0.107s/check against9,726 added lines suggests about17 minutes before
# extra block checks. Keep the exact scope under a finite30-minute deadline.
_AUDIT_TIMEOUT_SECONDS = 1800


def scan_tool_texts(texts: Iterable[str], *, secret_values: Iterable[str] = ()) -> None:
    """Refuse candidate/export secrets without reporting matched text.

    Args:
        texts (Iterable[str]): Module, contract and receipt text.
        secret_values (Iterable[str]): Live configured secret values held only
            in memory; empty values are ignored.

    Returns:
        None.

    Raises:
        ValueError: A candidate carries a configured secret or key-like text.
    """
    values = tuple(value for value in secret_values if value)
    for text in texts:
        if _KEY_LIKE.search(text) or any(value in text for value in values):
            raise ValueError("tool candidate secret scan refused; matched text is redacted")


def _receipt(value: Any, kind: str, bundle: str, *, resolve_path: bool = False) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"missing {kind} receipt")
    receipt = dict(value)
    report = receipt.get("report")
    if report is None and receipt.get("path") and resolve_path:
        path = Path(receipt["path"])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing {kind} report")
        report = json.loads(path.read_text(encoding="utf-8"))
    if report is None and receipt.get("path"):
        raise ValueError(f"published {kind} receipt requires inline report evidence")
    if not isinstance(report, Mapping) or receipt.get("report_sha256") != digest(report):
        raise ValueError(f"stale or malformed {kind} report digest")
    if report.get("bundle_digest") != bundle:
        raise ValueError(f"{kind} receipt is bound to another bundle")
    if kind in {"review", "invariant"}:
        if (not isinstance(report.get("report_text"), str)
                or report.get("report_text_sha256") != source_digest(report["report_text"])):
            raise ValueError(f"{kind} receipt lacks its exact hashed evidence")
    return dict(report)


def validate_tool_receipts(payload: Mapping[str, Any], gate: Mapping[str, Any], *,
                           parent_bundle: str | None, launch: bool,
                           tiers: Mapping[str, str] | None = None) -> None:
    """Recheck immutable audit, independent review and active-bundle invariants.

    Args:
        payload (Mapping[str, Any]): Exact tools.answering payload.
        gate (Mapping[str, Any]): Authority gate event payload.
        parent_bundle (str | None): Exact parent's bundle digest, null for R1.
        launch (bool): Whether validating launch rather than publication. Both
            apply the same integrity requirements.
        tiers (Mapping[str, str] | None): Actual inherited execution tier map
            from authority configuration or the verified launch snapshot.

    Returns:
        None.

    Raises:
        ValueError: A receipt is absent, stale, self-reviewed or failed.
    """
    bundle = digest(payload)
    if (gate.get("schema") != "rubiksql.tool-gate/1" or gate.get("bundle_digest") != bundle
            or "parent_bundle_digest" not in gate
            or gate.get("parent_bundle_digest") != parent_bundle or gate.get("mode") not in {"git", "hb"}):
        raise ValueError("tool gate does not bind the exact candidate and parent")
    if gate["mode"] == "git" and not gate.get("source_commit"):
        raise ValueError("Git tool gate lacks source commit")
    if not gate.get("audit_implementation_commit") or gate.get("audit_exit") != 0:
        raise ValueError("tool gate has no successful publisher audit")
    audit = _receipt(gate.get("audit"), "audit", bundle)
    if (audit.get("exit_code") != 0 or "parent_bundle_digest" not in audit or audit.get("parent_bundle_digest") != parent_bundle
            or audit.get("implementation_commit") != gate["audit_implementation_commit"]
            or audit.get("mode") != gate["mode"] or audit.get("source_commit") != gate.get("source_commit")
            or gate.get("audit_report_sha256") != digest(audit)):
        raise ValueError("tool audit receipt is stale")
    if (not isinstance(audit.get("report"), str)
            or audit.get("report_digest") != source_digest(audit["report"])):
        raise ValueError("tool audit evidence hash is stale")
    review = _receipt(gate.get("review"), "review", bundle)
    if "parent_bundle_digest" not in review or review["parent_bundle_digest"] != parent_bundle:
        raise ValueError("tool review receipt is bound to another or missing parent")
    if (review.get("model") != "gpt-6.1-sol" or review.get("reasoning_effort") != "ultra"
            or review.get("verdict") != "accepted" or not review.get("session_id")
            or not gate.get("author_session") or review["session_id"] == gate["author_session"]):
        raise ValueError("tool review must be accepted by an independent designated-model session")
    invariant = _receipt(gate.get("invariants"), "invariant", bundle)
    if "parent_bundle_digest" not in invariant or invariant["parent_bundle_digest"] != parent_bundle:
        raise ValueError("tool invariant receipt is bound to another or missing parent")
    if (invariant.get("exit_code") != 0 or invariant.get("bundle_active") is not True
            or not invariant.get("command") or invariant.get("transport_attempts") != 0
            or invariant.get("allocation_attempts") != 0 or invariant.get("credential_free") is not True):
        raise ValueError("tool invariant receipt must validate the active candidate offline")
    if gate.get("mode") == "hb":
        changed = gate.get("changed_modules")
        if not isinstance(changed, list):
            raise ValueError("direct HB gate lacks its changed-module set")
        if any((tiers or {}).get(module) == "host-verified" for module in changed):
            raise ValueError("direct HB authoring is unavailable for changed host-verified modules")
    scan_tool_texts([canonical(payload), canonical(gate)])


def validate_catalog_receipts(payload: Mapping[str, Any], gate: Mapping[str, Any], *,
                              parent_provenance: Mapping[str, Any]) -> None:
    """Verify catalog audit and review receipts against their actual source parent.

    Args:
        payload (Mapping[str, Any]): Exact complete typed catalog bundle.
        gate (Mapping[str, Any]): Immutable catalog gate; no BIRD admission is implied.
        parent_provenance (Mapping[str, Any]): Exact catalog publication or initial
            admitted BIRD Tool source identity.

    Returns:
        None: The complete evidence is verified without executing source.

    Raises:
        ValueError: Audit, independent review, invariant or parent evidence differs.
    """
    parent_fields = {"kind", "content_digest", "source_commit",
                     "release" if parent_provenance.get("kind") == "bird" else "publication_id"}
    if (gate.get("schema") != "rubiksql.catalog-gate/1" or gate.get("parent_provenance") != parent_provenance
            or parent_provenance.get("kind") not in {"bird", "catalog"} or set(parent_provenance) != parent_fields
            or not isinstance(parent_provenance.get("source_commit"), str)
            or re.fullmatch(r"[0-9a-f]{40}", parent_provenance["source_commit"]) is None
            or not isinstance(parent_provenance.get("content_digest"), str)
            or re.fullmatch(r"[0-9a-f]{64}", parent_provenance["content_digest"]) is None
            or not isinstance(parent_provenance.get("release", parent_provenance.get("publication_id")), str)
            or not parent_provenance.get("release", parent_provenance.get("publication_id"))):
        raise ValueError("catalog gate does not bind actual parent source provenance")
    audit = gate.get("audit", {}).get("report", {})
    if (audit.get("scope") != "catalog" or audit.get("parent_provenance") != parent_provenance
            or audit.get("parent_source_commit") != parent_provenance["source_commit"]):
        raise ValueError("catalog audit parent source evidence differs")
    # The shared receipt rules validate exact source/review evidence. Catalog
    # schema remains distinct in every persisted document and never admits BIRD.
    validate_tool_receipts(payload, {**gate, "schema": "rubiksql.tool-gate/1"},
                           parent_bundle=gate.get("parent_bundle_digest"), launch=False,
                           tiers=gate.get("tiers"))


def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=checkout, capture_output=True, text=True, check=True, timeout=60).stdout.strip()


def _audit_diagnostic(text: str, secret_values: Iterable[str]) -> str:
    """Retain auditor findings while redacting configured and key-like secrets."""
    for value in sorted(secret_values, key=len, reverse=True):
        text = text.replace(value, "[REDACTED]")
    return _KEY_LIKE.sub("[REDACTED]", text)


def publish_tool_gate(checkout: Path, payload: Mapping[str, Any], *, module_sources: Mapping[str, str],
                      parent_payload: Mapping[str, Any] | None, parent_sources: Mapping[str, str] | None,
                      parent_source_commit: str, receipts: Mapping[str, Any], mode: str = "git",
                      secret_values: Iterable[str] = (), tiers: Mapping[str, str] | None = None,
                      contracts: Any = None, parent_contracts: Any = None,
                      serializer_source: str | None = None,
                      catalog_parent: Mapping[str, Any] | None = None,
                      root_seed_commit: str | None = None) -> dict:
    """Run the publisher's audit and bind supplied independent validation receipts.

    Args:
        checkout (Path): Deployed source checkout; Git authoring must be clean
            and merged into its lane's meta/main.
        payload (Mapping[str, Any]): Candidate reference payload.
        module_sources (Mapping[str, str]): Exact candidate module sources.
        parent_payload (Mapping[str, Any] | None): Parent reference payload.
        parent_sources (Mapping[str, str] | None): Exact parent module sources.
        parent_source_commit (str): Recorded parent's Git source commit, or
            ccc996fe for R1.
        receipts (Mapping[str, Any]): Author session, review and invariants.
        mode (str): git for merged checkout source; hb for direct reviewed text.
        secret_values (Iterable[str]): Configured secret values, in memory only.
        tiers (Mapping[str, str] | None): Inherited host-verified tier map.
        contracts (Any): Complete reviewed candidate definitions, including
            visibility. Git mode checks them against committed answering_tools.json.
        parent_contracts (Any): Complete exact parent definitions for HB audit.
        serializer_source (str | None): Native serializer text, secret-scanned
            with candidate source before publication.
        catalog_parent (Mapping[str, Any] | None): Actual catalog parent source
            provenance; selects a complete direct-text audit with the committed
            deployed auditor and creates a distinct catalog gate.
        root_seed_commit (str | None): Original committed definitions for a
            parentless source seed. Requires Git mode, no parent payload and
            exact source equality with that ancestor. The executing source must
            still satisfy the normal clean, merged main/meta publication gate.

    Returns:
        dict: Successful tool-gate payload ready for create-only authority write.

    Raises:
        ValueError: Source, audit or immutable receipt verification fails.
    """
    if mode not in {"git", "hb"}:
        raise ValueError("tool authoring mode must be git or hb")
    secret_values = tuple(value for value in secret_values if value)
    scan_tool_texts([*module_sources.values(), serializer_source or "", canonical(contracts),
                     canonical(payload), canonical(receipts)], secret_values=secret_values)
    parent_digest = digest(parent_payload) if parent_payload else None
    # Paths are accepted only as authoring inputs. Publication embeds detached,
    # hash-checked evidence so export, restore and launch need no scratch files.
    durable_receipts = {"author_session": receipts.get("author_session")}
    for key, kind in (("review", "review"), ("invariants", "invariant")):
        report = _receipt(receipts.get(key), kind, digest(payload), resolve_path=True)
        durable_receipts[key] = {"report": deepcopy(report), "report_sha256": digest(report)}
    scan_tool_texts([canonical(durable_receipts)], secret_values=secret_values)
    head = _git(checkout, "rev-parse", "HEAD")
    if _git(checkout, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("tool publication requires a clean source checkout")
    if root_seed_commit is not None:
        if (mode != "git" or catalog_parent is not None or parent_payload is not None
                or parent_sources is not None or parent_contracts is not None
                or re.fullmatch(r"[0-9a-f]{40}", root_seed_commit) is None
                or parent_source_commit != root_seed_commit):
            raise ValueError("root tool admission requires exact original Git definitions and no parent")
        subprocess.run(["git", "merge-base", "--is-ancestor", root_seed_commit, head], cwd=checkout,
                       check=True, capture_output=True, timeout=60)
    if catalog_parent is None:
        merged = any(subprocess.run(["git", "merge-base", "--is-ancestor", head, lane], cwd=checkout,
                                   check=False, capture_output=True, timeout=60).returncode == 0
                     for lane in ("main", "meta/main"))
        if not merged:
            raise ValueError("tool publication requires source merged into main or meta/main")
    audit_commit = head
    temporary = tempfile.TemporaryDirectory(prefix="rubiksql-tool-audit-")
    with temporary as scratch:
        scratch_path = Path(scratch)
        if mode == "git":
            for module, source in module_sources.items():
                path = "src/" + module.replace(".", "/") + ("/__init__.py" if module == "rubiksql_hb.tools" else ".py")
                committed = subprocess.run(["git", "show", head + ":" + path], cwd=checkout, capture_output=True,
                                           check=True, timeout=60).stdout.decode("utf-8")
                if committed != source:
                    raise ValueError("module text differs from the committed source")
                if root_seed_commit is not None:
                    original = subprocess.run(["git", "show", root_seed_commit + ":" + path], cwd=checkout,
                                              capture_output=True, check=True, timeout=60).stdout.decode("utf-8")
                    if original != source:
                        raise ValueError("root module text differs from its original seed definitions")
            if payload.get("schema") in (BUNDLE_SCHEMA, TYPED_BUNDLE_SCHEMA):
                from .tool_definitions import validate_definitions

                if contracts is None:
                    raise ValueError("Git tool publication requires complete callable definitions")
                committed = subprocess.run(["git", "show", head + ":src/rubiksql_hb/answering_tools.json"],
                                           cwd=checkout, capture_output=True, check=True, timeout=60).stdout.decode("utf-8")
                committed_definitions = json.loads(committed)
                if catalog_parent is None:
                    from .tool_definitions import ANSWERING_ROLES

                    committed_definitions = [record for record in committed_definitions
                                             if any(role.split(":", 1)[0] in ANSWERING_ROLES
                                                    for role in [*record["visible_to"], *record["loop_visible_to"]])
                                             or not record["visible_to"] and not record["loop_visible_to"]]
                if canonical(validate_definitions(committed_definitions, allowed_modules=module_sources)) != canonical(contracts):
                    raise ValueError("callable definitions differ from the committed source")
                if root_seed_commit is not None:
                    original = subprocess.run(["git", "show", root_seed_commit + ":src/rubiksql_hb/answering_tools.json"],
                                              cwd=checkout, capture_output=True, check=True, timeout=60).stdout.decode("utf-8")
                    if json.loads(original) != json.loads(committed):
                        raise ValueError("root callable definitions differ from their original seed")
            # The trusted base implementation owns its complete 14-file import
            # closure. Always using it is the simplest stronger form of the rule.
            if catalog_parent is None:
                audit_commit = _git(checkout, "rev-parse", parent_source_commit)
                audit_root = scratch_path / "base"
                subprocess.run(["git", "worktree", "add", "--detach", str(audit_root), audit_commit], cwd=checkout,
                               check=True, capture_output=True, timeout=60)
                command = [sys.executable, str(audit_root / "scripts/audit_tool_change.py"), "--base", audit_commit, "--head", head]
        if mode == "hb" or catalog_parent is not None:
            if not parent_sources or contracts is None or parent_contracts is None:
                raise ValueError("direct HB authoring needs exact parent source and complete contracts")
            # HB candidates need the current host audit's direct-text interface.
            # Execute a detached committed closure, never mutable checkout bytes.
            audit_root = scratch_path / "host"
            subprocess.run(["git", "worktree", "add", "--detach", str(audit_root), audit_commit], cwd=checkout,
                           check=True, capture_output=True, timeout=60)
            parent_file, candidate_file = scratch_path / "parent.json", scratch_path / "candidate.json"
            parent_file.write_text(canonical({"modules": parent_sources, "contracts": parent_contracts}), encoding="utf-8")
            candidate_file.write_text(canonical({"modules": module_sources, "contracts": contracts}), encoding="utf-8")
            command = [sys.executable, str(audit_root / "scripts/audit_tool_change.py"), "--base", parent_source_commit,
                       "--hb-parent", str(parent_file), "--hb-candidate", str(candidate_file)]
        environment = {key: value for key, value in os.environ.items()
                       if not (key.endswith("API_KEY") or key.startswith("RUBIKSQL_GATEWAY"))}
        environment["PYTHONPATH"] = str(audit_root / "src") + os.pathsep + str(audit_root / "scripts")
        environment.pop("RUBIKSQL_TOOL_BUNDLE", None)
        environment.pop("RUBIKSQL_PROMPT_SNAPSHOT", None)
        try:
            result = subprocess.run(command, cwd=audit_root, env=environment, capture_output=True, text=True,
                                    timeout=_AUDIT_TIMEOUT_SECONDS)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(audit_root)], cwd=checkout,
                           capture_output=True, check=True, timeout=60)
        if result.returncode:
            stdout = _audit_diagnostic(result.stdout, secret_values)
            stderr = _audit_diagnostic(result.stderr, secret_values)
            raise ValueError(f"publisher answering-tool leak audit refused (exit {result.returncode})\n"
                             f"audit stdout:\n{stdout}\naudit stderr:\n{stderr}")
        audit = {"bundle_digest": digest(payload), "parent_bundle_digest": parent_digest,
                 "implementation_commit": audit_commit, "exit_code": 0,
                 "mode": mode, "source_commit": head if mode == "git" else None,
                 "report_digest": source_digest(result.stdout), "report": result.stdout}
        if catalog_parent is not None:
            audit.update(scope="catalog", parent_provenance=deepcopy(dict(catalog_parent)),
                         parent_source_commit=parent_source_commit)
        gate = {"schema": "rubiksql.tool-gate/1", "bundle_digest": digest(payload), "parent_bundle_digest": parent_digest,
                "mode": mode, "source_commit": head if mode == "git" else None,
                "audit_implementation_commit": audit_commit, "audit_report_sha256": digest(audit), "audit_exit": 0,
                "audit": {"report": audit, "report_sha256": digest(audit)}, "tiers": dict(tiers or {}),
                "changed_modules": sorted(module for module, source in module_sources.items()
                                          if (parent_sources or {}).get(module) != source),
                **durable_receipts}
        if catalog_parent is None:
            validate_tool_receipts(payload, gate, parent_bundle=parent_digest, launch=False, tiers=tiers)
        else:
            gate.update(schema="rubiksql.catalog-gate/1", parent_provenance=deepcopy(dict(catalog_parent)))
            validate_catalog_receipts(payload, gate, parent_provenance=catalog_parent)
        scan_tool_texts([canonical(gate)], secret_values=secret_values)
        return gate
