"""Bound workspace batches, historical rule/knob projections and query-time history facts.

GeneralRule and KnobSet rows with an empty ``format`` are the historical projection of the
immutable ``rules.general``/``config.knobs`` prompt carriers; :func:`refresh_views` maintains only
those. Canonical typed versions (non-empty ``format``) are authority rows written by publication
and are never written, updated or deleted here. Projection rows are not exported: they are a
deterministic function of the authority and are rebuilt by a refresh. Before the entity-model
migration they follow the live head; once an ``entity-model`` decision exists they are frozen as
of the first one, so their ``status`` and ``releases`` describe the state at migration time.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import heavenbase as hb

from .prompts import entity_model as M
from .prompts import entries as E
from .prompts import evolve as EV
from .prompts import rules as R
from .prompts.control import PromptControlError, check_row_integrity
from .prompts.control_schema import GeneralRuleEntity, KnobSetEntity, ProposalEntity, ReleaseEntity
from .prompts.store import POINTER_KINDS, PromptStoreError

if TYPE_CHECKING:
    from .prompts.authority import Authority, AuthorityRead

__all__ = ["DECISIONS", "PROMPT_STATUSES", "TYPED_ALIASES", "lab_lock", "lab_write", "query_history", "query_typed",
           "refresh_views"]

PROMPT_STATUSES = ("active", "revoked", "dropped", "superseded", "unpromoted")
DECISIONS = ("cutover", "migrated", "promoted", "rolled-back", "rejected", "abandoned", "not-evaluated",
             "undecided")
_DECISION_WORDS = {"cutover": "cutover", M.MIGRATION_KIND: "migrated", "promote": "promoted",
                   "rollback": "rolled-back", "reject": "rejected", "abandon": "abandoned"}
#: Pointer decisions whose target is the proposal that produced the new release.
_PROPOSAL_MOVES = frozenset({"cutover", M.MIGRATION_KIND, "promote"})
_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_LOCK_DEPTH = threading.local()
_WRITE_STATE = threading.local()


@contextlib.contextmanager
def lab_lock(directory: str | Path) -> Iterator[None]:
    """Serialize ingestion and derived-view refresh across threads and processes.

    Args:
        directory: The sole bound workspace directory.

    Yields:
        None while the lab-wide lock is held. Nested calls on the same thread reuse
        its file lock, so opening and helper-level writes can share one batch.

    Raises:
        OSError: The lock directory or file cannot be opened.
    """
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _LOCKS_GUARD:
        lock = _LOCKS.setdefault(root, threading.RLock())
    with lock:
        depths = getattr(_LOCK_DEPTH, "depths", None)
        if depths is None:
            depths = _LOCK_DEPTH.depths = {}
        if depths.get(root, 0):
            depths[root] += 1
            try:
                yield
            finally:
                depths[root] -= 1
            return
        with (root / ".ingestion.lock").open("a+b") as stream:
            if os.name == "nt":
                import msvcrt

                if stream.seek(0, os.SEEK_END) == 0:
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            depths[root] = 1
            try:
                yield
            finally:
                del depths[root]
                if os.name == "nt":
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _lab_directory(ws: hb.HeavenBase) -> Path:
    root = getattr(ws, "_rubiksql_lab_dir", None)
    if not root:
        raise PromptStoreError("lab writes require a workspace opened by open_lab_workspace")
    return Path(root)


def _epoch(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _created_order(doc: Mapping[str, Any]) -> tuple[float, str]:
    return _epoch(doc.get("created_at")), str(doc.get("id"))


def _base_release(body: Mapping[str, Any]) -> str | None:
    return next((str(parent["release"]) for parent in body.get("parents") or []
                 if isinstance(parent, Mapping) and parent.get("role", "base") == "base" and parent.get("release")), None)


def _knob_table(values: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: {"value": values.get(name, knob.default), "default": knob.default,
                   "source": "release" if name in values else "default", "kind": knob.kind,
                   "minimum": knob.minimum, "maximum": knob.maximum, "effect": knob.effect,
                   "env": knob.env, "roles": sorted(E.route_label(role, route) for role, route in knob.visible)}
            for name, knob in sorted(E.KNOBS.items())}


def _changed_kinds(changes: Mapping[str, Any]) -> list[str]:
    return sorted({M.change_kind(name) for name in changes})


def _outcome_summary(doc: Mapping[str, Any], file: str) -> dict[str, Any]:
    stats = doc.get("stats") or {}
    return {"id": doc.get("id"), "verdict": doc.get("verdict"), "verdict_reason": doc.get("verdict_reason"),
            "gain_pp": stats.get("gain_pp"), "p": stats.get("p"),
            "per_db": {db: (bucket or {}).get("effect_pp") for db, bucket in (stats.get("per_db") or {}).items()},
            "integrity": doc.get("integrity"), "source_verification": doc.get("source_verification"),
            "generation": doc.get("generation"), "run": doc.get("run") or None, "parent_run": doc.get("parent_run") or None,
            "created_at": doc.get("created_at"), "file": file, "document": dict(doc)}


class _Reading:
    """Native content and event facts from one committed authority read transaction.

    Args:
        reader: Committed authority view.
        as_of: Optional decision event at which the facts are cut off. The baseline is that pointer
            decision's ``from`` release, and only events strictly before it count (decisions, pointer
            moves, revocations, retirements, outcomes); releases, proposals and prompt rows first
            published at or after it are ignored. ``None`` reads the live head and every fact.
    """

    def __init__(self, reader: AuthorityRead, *, as_of: Mapping[str, Any] | None = None) -> None:
        head = reader.head()
        self.current = head.release
        self.baseline = str(as_of["payload"]["from"]) if as_of is not None else head.release
        self.revision = {"event_count": head.revision.event_count, "latest_event_id": head.revision.latest_event_id}
        events = reader.events()
        cut = len(events) if as_of is None else next(index for index, event in enumerate(events)
                                                     if event["object_id"] == as_of["object_id"])
        self.events = events[:cut]

        def published(selected: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str], set[str]]:
            publications = [event for event in selected if event["kind"] == "publish"]
            return ({str(event["payload"]["release"]) for event in publications if event["payload"].get("release")},
                    {str(event["subject"]) for event in publications},
                    {str(row["object_id"]) for event in publications for row in (event["payload"].get("batch") or {}).get("prompt_rows", ())}
                    | {str(pid) for event in publications for pid in event["payload"].get("prompt_ids", ())})

        # Only identities first published at or after the cut-off are excluded (the seeded root has no publication).
        late = [after - before for after, before in zip(published(events[cut:]), published(self.events))]
        epoch = reader.epoch()
        self.releases = {str(row["object_id"]): {**dict(row), "id": str(row["object_id"]),
                         "origin": "v0" if row["object_id"] == epoch["release"] else "proposal"}
                         for row in reader.rows("release-entity") if str(row["object_id"]) not in late[0]}
        held = {(name, key) for release in self.releases.values() for name, key in release["refs"].items()}
        self.objects = {(row["name"], row["version"]): {"row": dict(row)} for row in reader.rows("sys-prompt")
                        if str(row["object_id"]) not in late[2]
                        or (row["name"], row["manifest"]["meta"]["content_key"]) in held}
        self.typed = reader.typed_index()
        self.proposals = {str(row["object_id"]): dict(reader.proposal(str(row["object_id"])))
                          for row in reader.rows("proposal-entity") if str(row["object_id"]) not in late[1]}
        self.decisions = sorted(({**event["payload"], "id": event["object_id"], "actor": event["actor"],
                                "created_at": event["ts"], "previous": event["payload"].get("from"),
                                "release": event["payload"].get("to", event["payload"].get("release"))}
                               for event in self.events if event["kind"] == "decision"), key=_created_order)
        self.moves = [decision for decision in self.decisions if decision.get("kind") in POINTER_KINDS]
        for event in self.events:
            state = event["payload"]
            if (event["kind"] == "register" and state.get("schema") == "rubiksql.authority-state/1"
                    and state.get("operation") == "retire-refresh" and state.get("phase") == "completed"):
                self.releases[state["release"]].update(origin="state", authority_state=event["object_id"])
                self.moves.append({"kind": "state", "release": state["release"], "id": event["object_id"],
                                   "actor": event["actor"], "created_at": event["ts"]})
        self.moves.sort(key=_created_order)
        self.ever = {self.baseline} | {str(move["release"]) for move in self.moves if move.get("release")} | {str(move["previous"]) for move in self.moves if move.get("previous")}
        self.ever_rows = {(name, key) for rid in self.ever for name, key in self.releases[rid]["refs"].items()}
        self.baseline_refs = self.releases[self.baseline]["refs"]
        self.revoked: set[str] = set()
        self.retired: set[str] = set()
        for decision in self.decisions:
            kind = decision["kind"]
            if kind in ("revoke", "unrevoke"):
                key = str(decision["content_key"])
                self.revoked.add(key) if kind == "revoke" else self.revoked.discard(key)
            elif kind in ("retire", "unretire"):
                target = str(decision["target"])
                self.retired.add(target) if kind == "retire" else self.retired.discard(target)
        self.outcomes = {pid: sorted((dict(event["payload"]) for event in self.events if event["kind"] == "outcome"
                                     and event["payload"].get("proposal") == pid), key=_created_order)
                         for pid in self.proposals}
        self.producers: dict[str, list[str]] = {}
        for pid, body in self.proposals.items():
            self.producers.setdefault(str(body["release"]), []).append(pid)
            if body.get("origin") == "cutover" and self.releases[str(body["release"])].get("origin") != "state":
                self.releases[str(body["release"])]["origin"] = "cutover"
        for event in self.events:
            if event["kind"] != "publish" or event["subject"] not in self.proposals:
                continue
            body = self.proposals[event["subject"]]
            release = self.releases[body["release"]]
            if release["origin"] not in {"v0", "state"} and "proposal" not in release:
                release.update(proposal=body["id"], parents=body["parents"])
        self.gates = {pid: dict(reader.row("proposal-entity", pid).get("gate") or {}) for pid in self.proposals}
        self.corpora: dict[str, dict[str, Any]] = {}
        self.knob_sets: dict[str, dict[str, Any]] = {}
        for (name, version), document in self.objects.items():
            if name not in (E.RULES_ENTRY, E.KNOBS_ENTRY):
                continue
            try:
                check_row_integrity(document["row"])
            except PromptControlError as error:
                raise PromptStoreError(f"{name}:{version}: {error}") from error
            text = document["row"]["manifest"]["meta"]["text"]
            # Retained carriers keep their exact identity and layout, independent of the
            # current authoring policy, just like the authority's policy=False typed reads.
            if name == E.RULES_ENTRY:
                corpus = M.corpus_from_text(text)
                M.partition_corpus(corpus, policy=False)
                self.corpora[version] = corpus
            else:
                self.knob_sets[version] = M.values_from_text(text, policy=False)

    def version_status(self, name: str, key: str) -> str:
        if self.baseline_refs.get(name) == key:
            return "active"
        if key in self.revoked:
            return "revoked"
        if name in E.DROPPED_ENTRIES:
            return "dropped"
        return "superseded" if (name, key) in self.ever_rows else "unpromoted"

    def _content_source(self, name: str, version: str) -> dict[str, Any]:
        from heavenbase.extensions.prompt.name import prompt_row_id

        key = self.objects[(name, version)]["row"]["manifest"]["meta"]["content_key"]
        return {"content_key": key, "source_prompt_id": prompt_row_id(name, version),
                "status": self.version_status(name, key), "store_commit": "",
                "releases": sorted(rid for rid, release in self.releases.items() if release["refs"].get(name) == key)}

    def general_rule_rows(self) -> list[dict[str, Any]]:
        rows = []
        for version, corpus in self.corpora.items():
            source = self._content_source(E.RULES_ENTRY, version)
            provenance = corpus["provenance"]
            evolved = {record["title"]: record for record in provenance.get(R.EVOLVED, [])}
            for rule in corpus["rules"]:
                title = rule["title"]
                rows.append({"object_id": hb.utils.hash_id("general-rule", source["content_key"], title),
                    **{field: rule[field] for field in R.RULE_FIELDS}, "corpus_version": version, **source,
                    "provenance": provenance.get(title), "evolution": evolved.get(title),
                    "adjudications": [record for record in provenance.get(R.ADJUDICATIONS, []) if title in (record["new_rule"], record["existing_rule"])],
                    "amendments": [record for record in corpus["generation"].get("dev_measurement_amendments", []) if isinstance(record, Mapping) and record.get("rule") == title]})
        return rows

    def knob_set_rows(self) -> list[dict[str, Any]]:
        return [{"object_id": hb.utils.hash_id("knob-set", (source := self._content_source(E.KNOBS_ENTRY, version))["content_key"]),
                 "name": E.KNOBS_ENTRY, "version": version, **source, "values": values, "declarations": _knob_table(values)}
                for version, values in self.knob_sets.items()]

    def content_refs(self, refs: Mapping[str, str]) -> dict[str, dict[str, Any]]:
        from heavenbase.extensions.prompt.name import prompt_row_id

        result = {}
        for name, key in sorted(refs.items()):
            version = key[:16]
            if name == E.RULES_ENTRY:
                entity, ids = "general-rule-entity", [hb.utils.hash_id("general-rule", key, rule["title"]) for rule in self.corpora[version]["rules"]]
            elif name == E.KNOBS_ENTRY:
                entity, ids = "knob-set-entity", [hb.utils.hash_id("knob-set", key)]
            else:
                entity, ids = "sys-prompt", [prompt_row_id(name, version)]
            result[name] = {"entity": entity, "object_ids": ids, "version": version, "content_key": key}
            if name == "tools.answering":
                bundle = json.loads(self.objects[(name, version)]["row"]["manifest"]["meta"]["text"])
                result[name].update({"tools": bundle.get("tools", []), "modules": bundle["modules"],
                                     "serializer": bundle["serializer"]})
                if bundle.get("toolkits"):
                    result[name]["toolkits"] = bundle["toolkits"]
        return result

    def release_references(self, rid: str) -> dict[str, Any]:
        """A closure/2 release's stored typed block verbatim; closure/1 references computed from its refs."""
        data = self.releases[rid]
        return dict(data["content_refs"]) if data["rules"] == M.CLOSURE_V2 else self.content_refs(data["refs"])

    def parameter_values(self, rid: str) -> Mapping[str, Any]:
        """Explicit knob values: the closure/2 parameter set, else the closure/1 ``config.knobs`` text."""
        data = self.releases[rid]
        if data["rules"] == M.CLOSURE_V2:
            return self.typed[M.PARAMETER_ENTITY][data["content_refs"]["parameters"]["object_id"]]["values"]
        return self.knob_sets.get(data["refs"].get(E.KNOBS_ENTRY, "")[:16], {})

    def baseline_moves(self, rid: str) -> list[dict[str, Any]]:
        return [{"kind": move["kind"], "from": move.get("previous"), "to": move.get("release"), "decision": move["id"],
                 "proposal": move.get("proposal", move.get("target") if move["kind"] in _PROPOSAL_MOVES else None),
                 "actor": move.get("actor"), "at": move.get("created_at")}
                for move in self.moves if rid in (move.get("release"), move.get("previous"))]

    def release_rows(self) -> list[dict[str, Any]]:
        rows = []
        for rid, data in self.releases.items():
            refs = dict(data["refs"])
            pids = [] if data.get("origin") == "state" else self.producers.get(rid, [])
            parents = list(dict.fromkeys(str(parent["release"]) for parent in data.get("parents") or []))
            publication = self.proposals.get(data.get("proposal"), {})
            status = "active" if rid == self.baseline else "retired" if rid in self.retired else "superseded" if rid in self.ever else "published"
            # Closure keys share the proposal-changes vocabulary; closure/1 yields exactly its prompt refs.
            closure = M.closure_view(data, self.typed)
            rows.append({"object_id": rid, "refs": refs, "entries": sorted(data["entries"]), "rules": data["rules"],
                "reason": str(data.get("reason") or publication.get("reason") or ""),
                "created_at": _epoch(publication.get("created_at")),
                "origin": data.get("origin", ""), "parents": parents, "proposals": pids,
                **({"authority_state": data["authority_state"]} if data.get("origin") == "state" else {}),
                "row_versions": sorted(f"{key}:{value[:16]}" for key, value in closure.items()),
                "content_refs": self.release_references(rid),
                "status": status, "baseline": self.baseline_moves(rid), "knobs": _knob_table(self.parameter_values(rid)),
                "store_commit": ""})
        return rows

    def proposal_rows(self) -> list[dict[str, Any]]:
        decided = {}
        for decision in self.decisions:
            if decision.get("kind") not in _DECISION_WORDS:
                continue
            targets = [decision.get("proposal", decision.get("target"))]
            if decision["kind"] in _PROPOSAL_MOVES and not any(pid in self.proposals for pid in targets):
                targets = self.producers.get(str(decision.get("release")), [])
            if decision["kind"] == "rollback":
                targets = decision.get("demoted") or self.producers.get(str(decision.get("previous")), [])
            for pid in targets:
                if pid in self.proposals:
                    decided[pid] = decision
        rows = []
        for pid, body in self.proposals.items():
            base, docs = _base_release(body), self.outcomes[pid]
            verdicts = [doc for doc in docs if doc.get("verdict") in EV.VERDICTS]
            chosen = EV.latest_verdict(verdicts, base or "") or max(verdicts or docs, key=_created_order, default={})
            stats, verdict, decision = chosen.get("stats") or {}, chosen.get("verdict"), decided.get(pid)
            if decision:
                word, by, at, why = _DECISION_WORDS[decision["kind"]], decision.get("actor"), decision.get("created_at"), decision.get("reason")
            elif verdict == "not-promotable":
                word, by, at, why = "rejected", "rule", chosen.get("created_at"), chosen.get("verdict_reason")
            else:
                word, by, at, why = ("undecided" if verdict in ("promotable", "adoptable") else "not-evaluated"), None, chosen.get("created_at"), chosen.get("verdict_reason")
            changes = dict(body.get("changes") or {})
            related = [event for event in self.events if event["subject"] in (pid, body["release"]) or event["payload"].get("proposal") == pid]
            rows.append({"object_id": pid, "parents": body.get("parents") or [], "release": body["release"], "changes": changes,
                "reason": body.get("reason", ""), "hypothesis": body.get("hypothesis"), "hypothesis_digest": body.get("hypothesis_digest", ""),
                "analysis_plan": body.get("plan"), "plan_digest": body.get("plan_digest", ""), "author": body.get("author", ""),
                "origin": body.get("origin", ""), "gate": {**self.gates[pid], "receipts": [event["payload"] for event in related if event["kind"] == "gate"]},
                "created_at": _epoch(body.get("created_at")), "base": base, "generation": chosen.get("generation") or base,
                "row_names": sorted(changes), "row_versions": sorted(f"{name}:{change['to'][:16]}" for name, change in changes.items() if change.get("to")),
                "content_refs": self.release_references(str(body["release"])), "kinds": _changed_kinds(changes),
                "verdict": verdict, "gain_pp": stats.get("gain_pp"), "p_value": stats.get("p"),
                "outcomes": [_outcome_summary(doc, f"outcomes/{pid}/{doc['id']}.json") for doc in docs],
                "experiments": list(dict.fromkeys(str(doc["run"]) for doc in docs if doc.get("run"))),
                "decision": word, "decided_by": by, "decided_at": _epoch(at) if at is not None else None, "decision_reason": why,
                "files": {"proposal": f"proposals/{pid}.json", "events": [event["object_id"] for event in related]},
                "store_commit": ""})
        return rows


def _differs(before: Mapping[str, Any] | None, row: Mapping[str, Any]) -> bool:
    return before is None or any(before.get(key) != value for key, value in row.items())


def _sync(ws: hb.HeavenBase, cls: type, rows: Sequence[Mapping[str, Any]], existing: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    held = {str(row["object_id"]): row for row in existing}
    stale = sorted(set(held) - {str(row["object_id"]) for row in rows})
    changed = [row for row in rows if _differs(held.get(str(row["object_id"])), row)]
    if stale:
        ws.delete_many([(cls, object_id) for object_id in stale])
    if changed:
        ws.upsert_many(cls, list(changed), catalog_mode="merge" if held else "fresh")
    return len(changed), len(stale)


def refresh_views(ws: hb.HeavenBase, reader: AuthorityRead) -> dict[str, Any]:
    """Synchronize only the historical GeneralRule and KnobSet projection rows from the prompt carriers.

    Comparison, update and deletion consider only projection rows (an empty ``format``); a
    canonical rule or parameter-set version is never written, updated or deleted. Once an
    ``entity-model`` decision exists the projection is frozen as of the first such decision: it is
    rebuilt from the authority cut off there (baseline = that decision's ``from`` release; only
    earlier events, releases, proposals and carrier rows count), so its ``status`` and
    ``releases`` stay as they were at migration time (the migrated carriers remain ``active``) and
    later content never reaches it. A live lab refreshed just before migrating therefore writes
    nothing, while a restored or stale workspace gets exactly those rows back; the old refresh is
    never a repair path for new content.

    Args:
        ws: The writer's bound canonical workspace.
        reader: Its live AuthorityRead under the same Authority.write batch.

    Returns:
        Current release (``baseline``), authority revision, projection row counts, changes,
        ``frozen``, ``as_of`` (the cut-off decision id or null) and timing. Canonical prompts, typed
        versions, capsules, toolkits, releases, proposals and evidence are untouched.

    Raises:
        PromptStoreError: The reader is not the active writer, the path differs,
            or source content is invalid.
    """
    from .prompts.authority import AuthorityRead

    if not isinstance(reader, AuthorityRead) or not reader.writer_lock_held:
        raise PromptStoreError("refresh_views requires an active Authority.write reader")
    if Path(getattr(ws, "_rubiksql_authority_db", "")).resolve() != reader.path:
        raise PromptStoreError("refresh_views workspace differs from its bound reader")
    started = time.perf_counter()
    with lab_lock(_lab_directory(ws)):
        migration = next((event for event in reader.events("decision")
                          if event["payload"].get("kind") == M.MIGRATION_KIND), None)
        reading = _Reading(reader, as_of=migration)
        held = {kind: [row for row in ws.query(cls).execute().rows() if not row.get("format")]
                for kind, cls in (("general_rules", GeneralRuleEntity), ("knob_sets", KnobSetEntity))}
        batches = (("general_rules", GeneralRuleEntity, reading.general_rule_rows()),
                   ("knob_sets", KnobSetEntity, reading.knob_set_rows()))
        read = time.perf_counter() - started
        counts, written, deleted = {}, {}, {}
        for kind, cls, rows in batches:
            counts[kind] = len(rows)
            written[kind], deleted[kind] = _sync(ws, cls, rows, held[kind])
        return {"authority": str(reader.path), "baseline": reading.current, "authority_revision": reading.revision,
                "counts": counts, "written": written, "deleted": deleted, "frozen": migration is not None,
                "as_of": migration["object_id"] if migration is not None else None,
                "seconds": {"read": round(read, 3), "write": round(time.perf_counter() - started - read, 3)}}


@contextlib.contextmanager
def _evidence_transaction(ws: hb.HeavenBase) -> Iterator[None]:
    """Commit or roll back the evidence and derived views as one SQLite batch.

    The pinned HB backend opens a Database.tx for every entity write, while its
    Database does not support nested transactions. Reuse that one facade's outer
    transaction within this scope; no other writer or database facade is changed.
    """
    database = ws.backends.get("main").db
    original_tx = database.tx
    had_override = "tx" in vars(database)
    with original_tx():
        database.tx = lambda readonly=False: contextlib.nullcontext(database)
        try:
            yield
            if database._tx_failed:
                raise PromptStoreError("evidence batch database execution failed; the batch was rolled back")
        finally:
            if had_override:
                database.tx = original_tx
            else:
                del database.tx


@contextlib.contextmanager
def lab_write(ws: hb.HeavenBase, operation: str) -> Iterator[hb.HeavenBase]:
    """Write one complete evidence batch through the bound Authority writer.

    Evidence and derived views commit together or roll back on callback/validation
    failure. The event and complete export follow that commit. An export interruption
    leaves valid evidence that a subsequent bound writer can export again.

    Args:
        ws: Workspace opened through the canonical Control/Lab adapter.
        operation: Domain operation recorded once after a successful outer batch.

    Yields:
        The active writer's canonical workspace. Nested helpers on this thread and
        path reuse it, including their authority and ingestion locks.

    Raises:
        PromptStoreError: The workspace has no explicit binding or its path differs.
        AuthorityError: The live binding, source integrity or writer fence fails.
    """
    from .lab_schema import canonical_evidence_entities, lab_binding
    from .prompts.authority import Authority, default_export_dir, secret_inputs
    from .prompts.types import GitAuthorityBinding
    from .prompts.workspace_migration import validate_evidence

    binding = getattr(ws, "_rubiksql_authority_binding", None)
    if binding is None:
        raise PromptStoreError("lab writes require a workspace with an explicit authority binding")
    source_path = Path(getattr(ws, "_rubiksql_authority_db", "")).resolve()
    is_git = isinstance(binding, GitAuthorityBinding)
    binding = lab_binding(None if is_git else source_path, binding=binding)
    commit = getattr(ws, "_rubiksql_authority_commit", None)
    if is_git and (not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) is None):
        raise PromptStoreError("lab writes require the projection's exact Git authority commit")

    def check_writer(writer: Authority) -> None:
        if (getattr(writer, "git_binding", None) or writer.binding) != binding:
            raise PromptStoreError("active writer differs from the workspace binding")
        if is_git and commit != getattr(writer.control.ws, "_rubiksql_authority_commit", None):
            raise PromptStoreError("active writer differs from the workspace's pinned Git authority commit")

    key = binding if is_git else binding.db_path
    active: dict[Any, Authority] = getattr(_WRITE_STATE, "active", {})
    if key in active:
        check_writer(active[key])
        yield active[key].control.ws
        return
    owner = Authority.active_writer(binding=binding) if is_git else Authority.active_writer(source_path)
    if owner is not None:
        check_writer(owner)
    options = {}
    if is_git and owner is None:
        options["expected_commit"] = commit
    writer_context = contextlib.nullcontext(owner) if owner is not None else Authority.write(binding=binding, **options)
    with writer_context as writer:
        active = _WRITE_STATE.active = {**active, key: writer}
        try:
            with lab_lock(writer.path.parent):
                with _evidence_transaction(writer.control.ws):
                    yield writer.control.ws
                    with writer.reader() as reader:
                        refresh_views(writer.control.ws, reader)
                    validate_evidence({entity: writer.control.ws.query(writer.control.ws.entities[entity]).execute().rows()
                                       for entity in canonical_evidence_entities()})
                writer.record_workspace_change(operation)
                if is_git:
                    writer.commit()
                else:
                    writer.export(export_dir=default_export_dir(binding), secret_inputs=secret_inputs())
        finally:
            _WRITE_STATE.active = {path: held for path, held in active.items() if path != key}


def _history_rows(ws: hb.HeavenBase, entity: str) -> list[dict[str, Any]]:
    """Return complete query-time history facts for bounded queries and evidence."""
    from .prompts.authority import Authority

    if entity not in ("release", "proposal"):
        raise ValueError(f"unknown history alias {entity!r}")
    binding = getattr(ws, "_rubiksql_authority_binding", None)
    if binding is None:
        raise PromptStoreError("history reads require the bound workspace adapter")
    options = {"commit": ws._rubiksql_authority_commit} if getattr(ws, "_rubiksql_authority_commit", None) else {}
    with Authority.read(binding=binding, **options) as reader:
        verified = reader.verify(require_export=False)
        if not verified.ok:
            raise PromptStoreError(f"authority history failed verification: {verified.violations}")
        reading = _Reading(reader)
        return reading.release_rows() if entity == "release" else reading.proposal_rows()


def query_history(ws: hb.HeavenBase, entity: str, where: Mapping[str, Any] | None = None,
                  limit: int = 50) -> list[dict[str, Any]]:
    """Read release/proposal lifecycle facts without replacing canonical rows.

    Args:
        ws: Canonical workspace with an explicit authority binding.
        entity: ``release`` or ``proposal``.
        where: The same typed exact filters query_lab accepts.
        limit: Result cap, clamped to query_lab's published range.

    Returns:
        Query-time enriched rows with object_id exposed as entity_id.

    Raises:
        ValueError: Unknown fields, operations or entity alias.
        PromptStoreError: The workspace has no binding or authority verification fails.
    """
    from heavenbase.execution.handlers.filters import normalize_filter_value
    from heavenbase.execution.handlers.predicates import Predicate
    from heavenbase.execution.handlers.seed import filter_ops
    from heavenbase.filters.spec import And, Filter, Not, Or
    from heavenbase.query.json import query_expr
    from heavenbase.types.descriptor import coerce_type
    from .knowledge_host import QUERY_LIMIT_MAX, _normalize_where
    from .query_policy import refuse_code_reduce

    refuse_code_reduce(where)
    rows = _history_rows(ws, entity)
    cls = ReleaseEntity if entity == "release" else ProposalEntity
    resolver, config = ws.module_resolver, ws.context.config
    expression = query_expr(_normalize_where(cls, dict(where)) if where else None, cls.schema(), resolver=resolver)

    def compile_match(node: Any) -> Any:
        if node is None:
            return lambda row: True
        if isinstance(node, (And, Or)):
            children = [compile_match(child) for child in node.clauses]
            combine = all if isinstance(node, And) else any
            return lambda row: combine(match(row) for match in children)
        if isinstance(node, Not):
            match = compile_match(node.inner)
            return lambda row: not match(row)
        if isinstance(node, Filter):
            field = cls.schema().fields[node.field]
            dtype = coerce_type(field.dtype)
            if node.op not in filter_ops(dtype, resolver=resolver):
                raise ValueError(f"operation {node.op!r} is unsupported for {entity}.{node.field} ({dtype.identifier})")
            value = normalize_filter_value(dtype, node.op, node.value, resolver=resolver)
            predicate = Predicate.load(node.op, dtype, resolver=resolver)
            prepared = predicate.compile(value, resolver=resolver, config=config)
            return lambda row: predicate.match(row.get(node.field), prepared)
        raise TypeError(f"unsupported history filter {type(node).__name__}")

    match = compile_match(expression)
    selected = [row for row in rows if match(row)][:max(1, min(int(limit), QUERY_LIMIT_MAX))]
    return [{**{key: value for key, value in row.items() if key != "object_id"}, "entity_id": row["object_id"]}
            for row in selected]


#: query_lab aliases of the typed version entities that :func:`query_typed` reads as business entities.
TYPED_ALIASES = {"tool": "tool-entity", "general_rule": "general-rule-entity", "knob_set": "knob-set-entity"}


def _where_match(ws: hb.HeavenBase, cls: Any, where: Mapping[str, Any] | None) -> Any:
    """Compile one exact query_lab filter over an entity schema with query_history's HB predicates."""
    from heavenbase.execution.handlers.filters import normalize_filter_value
    from heavenbase.execution.handlers.predicates import Predicate
    from heavenbase.execution.handlers.seed import filter_ops
    from heavenbase.filters.spec import And, Filter, Not, Or
    from heavenbase.query.json import query_expr
    from heavenbase.types.descriptor import coerce_type
    from .knowledge_host import _normalize_where

    schema, resolver, config = cls.schema(), ws.module_resolver, ws.context.config
    expression = query_expr(_normalize_where(cls, dict(where)) if where else None, schema, resolver=resolver)

    def compile_match(node: Any) -> Any:
        if node is None:
            return lambda row: True
        if isinstance(node, (And, Or)):
            children = [compile_match(child) for child in node.clauses]
            combine = all if isinstance(node, And) else any
            return lambda row: combine(match(row) for match in children)
        if isinstance(node, Not):
            match = compile_match(node.inner)
            return lambda row: not match(row)
        if isinstance(node, Filter):
            dtype = coerce_type(schema.fields[node.field].dtype)
            if node.op not in filter_ops(dtype, resolver=resolver):
                raise ValueError(f"operation {node.op!r} is unsupported for {schema.entity_id}.{node.field} ({dtype.identifier})")
            value = normalize_filter_value(dtype, node.op, node.value, resolver=resolver)
            predicate = Predicate.load(node.op, dtype, resolver=resolver)
            prepared = predicate.compile(value, resolver=resolver, config=config)
            return lambda row: predicate.match(row.get(node.field), prepared)
        raise TypeError(f"unsupported typed version filter {type(node).__name__}")

    return compile_match(expression)


def _typed_selection(reader: AuthorityRead, entity: str, release: str,
                     canonical: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the exact typed rows one release references, in release order."""
    from .hb_tool_entities import BUNDLE_SCHEMA, convert_answering
    from .prompts import entity_model as M

    row = reader.row("release-entity", release)
    if row["rules"] == M.CLOSURE_V2:
        refs = M.check_release_refs(row["content_refs"])
        wanted = (refs["tools"] if entity == M.TOOL_ENTITY else refs["rules"]["items"] if entity == M.RULE_ENTITY
                  else [refs["parameters"]])
        selected = []
        for ref in wanted:
            stored = canonical.get(ref["object_id"])
            if stored is None or M.typed_ref(entity, stored) != M.check_ref(ref, entity):
                raise M.EntityModelError(f"release {release} references {entity} {ref['object_id']}, which is absent or differs")
            selected.append(dict(stored))
        return selected
    if row["rules"] != E.CLOSURE_RULES:
        raise M.EntityModelError(f"release {release} has unknown closure rules {row['rules']!r}")
    # A closure/1 release stores no typed rows: convert its pinned carriers exactly as the
    # entity-model migration would, for this read only, without current rule or knob policy
    # (retained history stays readable as recorded, like the authority's policy=False checks).
    carrier = "tools.answering" if entity == M.TOOL_ENTITY else E.RULES_ENTRY if entity == M.RULE_ENTITY else E.KNOBS_ENTRY
    key = row["refs"].get(carrier)
    if key is None:
        return []
    if entity == M.TOOL_ENTITY:
        payload = reader.answering(release)
        if payload.get("schema") != BUNDLE_SCHEMA:
            raise M.EntityModelError(f"release {release} pins a historical {payload.get('schema')} Toolkit bundle; "
                                     "read its sys-toolkit and sys-capsule rows")
        capsules = {capsule["object_id"]: capsule for capsule in reader.rows("sys-capsule")}
        return [M.stored_json(tool) for tool in convert_answering(payload, capsules).tool_rows]
    texts = [prompt["manifest"]["meta"]["text"] for prompt in reader.rows("sys-prompt")
             if prompt["name"] == carrier and prompt["manifest"]["meta"]["content_key"] == key]
    if len(texts) != 1:
        raise PromptStoreError(f"release {release} has no exact {carrier} row")
    if entity == M.RULE_ENTITY:
        return [M.stored_json(rule) for rule in M.partition_corpus(M.corpus_from_text(texts[0]), policy=False)[0]]
    return [M.stored_json(M.parameter_row(M.values_from_text(texts[0], policy=False), policy=False))]


def query_typed(ws: hb.HeavenBase, entity: str, where: Mapping[str, Any] | None = None, limit: int | None = 50, *,
                release: str | None = None, role: str | None = None, route: str = "default",
                history: bool = False) -> list[dict[str, Any]]:
    """Read Tool, rule and parameter-set versions as business entities, by default the current release's.

    Nothing is written. A closure/2 release selects its exact stored canonical rows. A closure/1
    release (published before the entity-model migration) stores no typed rows: its selection is
    the deterministic conversion of its pinned ``tools.answering``, ``rules.general`` and
    ``config.knobs`` carriers (the rows that migration stores), computed for this read only.

    Args:
        ws: Canonical workspace with an explicit authority binding.
        entity: ``tool``, ``general_rule`` or ``knob_set``.
        where: The same exact filters query_lab accepts, over the entity's stored fields.
        limit: Result cap clamped into 1..200, or None for every match (dashboard paging).
        release: ``baseline`` (the default), an exact release id or a published proposal id.
        role: Tools only: candidate, judge or reviewer; keeps the versions that role's model may call.
        route: Tools with ``role``: the exact route, ``default`` unless the role runs another route.
        history: List every stored version instead of one release's selection: canonical versions
            and, for rules and parameter sets, historical projection rows (empty ``format``).

    Returns:
        Stored fields only, with object_id exposed as entity_id. A release selection keeps release
        order (tools in serve order, rules in corpus order, the one parameter set); history is
        ordered by logical name and object id.

    Raises:
        ValueError: Unknown alias, field, operation, role or route; a selector combined with
            history; a role for rules or parameters; unresolvable typed references or carriers.
        PromptStoreError: The workspace has no binding or authority verification fails.
    """
    from .hb_tool_entities import tool_definition
    from .knowledge_host import QUERY_LIMIT_MAX
    from .prompts import entity_model as M
    from .prompts.authority import Authority
    from .query_policy import refuse_code_reduce
    from .tool_definitions import select_definitions

    alias = (entity or "").strip().lower()
    if alias not in TYPED_ALIASES:
        raise ValueError(f"unknown typed version alias {entity!r}; valid names are: {', '.join(TYPED_ALIASES)}")
    if (release is not None and not isinstance(release, str) or role is not None and not isinstance(role, str)
            or not isinstance(route, str) or not isinstance(history, bool)):
        raise ValueError("release and role are strings or null, route is a string and history is a boolean")
    if history and (release is not None or role is not None or route != "default"):
        raise ValueError("history lists every stored version; it takes no release, role or route")
    if alias != "tool" and (role is not None or route != "default"):
        raise ValueError("role and route select tools only")
    if role is None and route != "default":
        raise ValueError("route selects the tools of a role; name the role")
    if role is not None:
        select_definitions((), role, route)  # refuses an unknown role or route before any read
    refuse_code_reduce(where)
    entity_id = TYPED_ALIASES[alias]
    cls = ws.entities[entity_id]
    match = _where_match(ws, cls, where)
    binding = getattr(ws, "_rubiksql_authority_binding", None)
    if binding is None:
        raise PromptStoreError("typed version reads require the bound workspace adapter")
    options = {"commit": ws._rubiksql_authority_commit} if getattr(ws, "_rubiksql_authority_commit", None) else {}
    with Authority.read(binding=binding, **options) as reader:
        verified = reader.verify(require_export=False)
        if not verified.ok:
            raise PromptStoreError(f"authority typed read failed verification: {verified.violations}")
        canonical = {str(row["object_id"]): dict(row) for row in reader.rows(entity_id) if M.is_canonical(entity_id, row)}
        if history:
            fields = set(cls.schema().fields)
            projections = [] if entity_id == M.TOOL_ENTITY else [
                {key: value for key, value in row.items() if key in fields}
                for row in ws.query(cls).execute().rows() if not M.is_canonical(entity_id, row)]
            rows = sorted([*canonical.values(), *projections],
                          key=lambda row: (str(row.get("title") or row.get("name") or ""), str(row["object_id"])))
        else:
            rows = _typed_selection(reader, entity_id, reader.resolve(release or "baseline"), canonical)
            if role is not None:
                definitions = [tool_definition(row) for row in rows]
                permitted = {tool["name"] for tool in select_definitions(definitions, role, route)}
                rows = [row for row in rows if row["name"] in permitted]
    selected = [row for row in rows if match(row)]
    if limit is not None:
        selected = selected[:max(1, min(int(limit), QUERY_LIMIT_MAX))]
    return [{**{key: value for key, value in row.items() if key != "object_id"}, "entity_id": row["object_id"]}
            for row in selected]
