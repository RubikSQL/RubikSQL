"""Lifecycle derived from the store and git, display version numbers, the proposal tree, and the baseline pins.

Nothing here is a mutable status field: a release is *active* because ``BASELINE.json``
points at it, *superseded* because some earlier commit of ``BASELINE.json`` did, a proposal is
*pinned* because its file exists, *evaluated* because an outcome does, and the rest follows
from the decision files, whose ``actor`` tells a promotion :func:`evolve.outcome` performed
(``auto:outcome``) from one a person made. ``vN`` numbers are display-only and come from the
commit that first added each version file. ``BASELINE_PINS`` lists the test nodes whose asserted
values depend on the BASELINE bytes, so a promotion can print what must move with a new
baseline; it is empty, because the offline suite must survive an unattended promotion.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .policy import PROPOSAL_RE, RELEASE_RE
from .store import DECISION_KINDS, POINTER_KINDS, PromptStoreError, Store

__all__ = [
    "BASELINE_PINS",
    "DECISION_KINDS",
    "Lineage",
    "Pin",
    "ProposalState",
    "ReleaseState",
    "TreeNode",
    "baseline_history",
    "derive",
    "git_output",
    "proposal_tree",
    "render_tree",
    "version_numbers",
]

# ---------------------------------------------------------------------------------------
# Baseline pins


@dataclass(frozen=True)
class Pin:
    """One existing test node whose asserted value depends on the BASELINE bytes.

    Attributes:
        file (str): Test file relative to the repository root.
        node (str): Top-level function or constant name in that file.
        note (str): What the node pins.
    """

    file: str
    node: str
    note: str


#: Test nodes whose asserted values depend on the BASELINE bytes, printed by every promotion as the pins to move with
#: the new baseline. Empty since 2026-09-26, by rule: an unattended promotion by the evolution loop must leave the
#: offline suite green, so no test names the release ``BASELINE.json`` points at, or asserts a value only that
#: release's bytes produce. A test reads the pointer and asserts what holds for any BASELINE, compares what the code
#: sends with the loaded release's own render, or states a fact about an immutable release by its id — root for the
#: role documents' wording, fingerprints, cache keys and golden bytes, ``r114f86603b144568`` for the generator
#: documents' (docs/architecture.md, *Tests and the pointer*). The table and the report line stay;
#: tests/test_prompt_lineage.py holds the table empty, so listing a node here is a deliberate break of that rule,
#: printed by every promotion until the node is rewritten.
BASELINE_PINS: tuple[Pin, ...] = ()


# ---------------------------------------------------------------------------------------
# git helpers


def git_output(repo_root: Path, *args: str) -> str:
    """Run one git command and return its stdout.

    Args:
        repo_root (Path): Working directory for git.
        *args (str): Arguments after ``git``.

    Returns:
        str: Standard output; empty when git fails or the directory is not a checkout.
    """
    try:
        completed = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=False)
    except (OSError, ValueError):
        return ""
    return completed.stdout if completed.returncode == 0 else ""


def _repo_root(store_dir: Path) -> Path | None:
    top = git_output(store_dir, "rev-parse", "--show-toplevel").strip()
    return Path(top) if top else None


def _store_rel(store_dir: Path, repo_root: Path) -> str:
    return Path(store_dir).resolve().relative_to(repo_root.resolve()).as_posix()


def baseline_history(store_dir: Path) -> list[str]:
    """Release ids ``BASELINE.json`` has pointed at, oldest commit first, from git history.

    Args:
        store_dir (Path): The package store directory inside a checkout.

    Returns:
        list[str]: Release ids in commit order (oldest first, duplicates kept); empty without git history.
    """
    store = Store(package=Path(store_dir))
    scope = store.v0_scope()
    history = [scope.root] if scope is not None else []
    states = [{"kind": "state", "release": event["payload"]["release"], "id": event["object_id"], "created_at": event["ts"]}
              for event in store.native("prompt-event-entity") if event["kind"] == "register"
              and event["payload"].get("schema") == "rubiksql.authority-state/1"
              and event["payload"].get("operation") == "retire-refresh"
              and event["payload"].get("phase") == "completed"]
    for decision in sorted([*store.decisions(), *states], key=lambda item: (float(item.get("created_at") or 0), item.get("id", ""))):
        if decision.get("kind") not in POINTER_KINDS | {"state"}:
            continue
        release = decision.get("to") or decision.get("release")
        if release is None:
            target = decision.get("target")
            release = store.proposal(target).data["release"] if isinstance(target, str) and PROPOSAL_RE.fullmatch(target) else target
        if isinstance(release, str) and RELEASE_RE.fullmatch(release):
            history.append(release)
    return history


def version_numbers(store_dir: Path, name: str) -> dict[str, int]:
    """Display ``vN`` numbers of a row name, by the commit that first added each version file.

    Args:
        store_dir (Path): The package store directory.
        name (str): Row name.

    Returns:
        dict[str, int]: 16-hex version to ``N`` (1-based). Committed versions come first in commit order;
            versions present on disk but never committed follow in file-name order.
    """
    store_dir = Path(store_dir)
    directory = store_dir / "objects" / name
    ordered: list[str] = []
    fixture_store = Store(package=store_dir)
    fixture_scope = fixture_store.v0_scope()
    root_key = fixture_store.release(fixture_scope.root).data["refs"].get(name) if fixture_scope is not None else None
    if root_key is not None:
        ordered.append(root_key[:16])
    if directory.is_dir():
        def created(path: Path) -> tuple[float, str]:
            row = json.loads(path.read_text(encoding="utf-8"))["row"]
            return float(row.get("created_at") or 0), path.stem

        for path in sorted(directory.glob("*.json"), key=created):
            if path.stem not in ordered:
                ordered.append(path.stem)
    store = Store(package=store_dir)
    scope = store.v0_scope()
    if scope is not None:
        visible = {key[:16] for rid in scope.releases
                   for row, key in store.release(rid).data["refs"].items() if row == name}
        ordered = [version for version in ordered if version in visible]
    return {version: index for index, version in enumerate(ordered, 1)}


# ---------------------------------------------------------------------------------------
# Lifecycle


@dataclass
class ReleaseState:
    """Derived state of one release."""

    id: str
    active: bool = False
    superseded: bool = False
    retired: bool = False
    proposals: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """One word for the release.

        Returns:
            str: ``active``, ``retired``, ``superseded`` or ``published``.
        """
        if self.active:
            return "active"
        if self.retired:
            return "retired"
        if self.superseded:
            return "superseded"
        return "published"


#: Decision kinds that set a proposal's status word, with that word.
_STATUS_OF = {"entity-model": "migrated", "promote": "promoted", "rollback": "rolled-back",
              "reject": "rejected", "abandon": "abandoned"}


@dataclass
class ProposalState:
    """Derived state of one proposal.

    Attributes:
        decisions (tuple[str, ...]): Kinds of the decisions about the proposal, oldest first.
        actors (tuple[str, ...]): The actor of each of those decisions, in the same order.
    """

    id: str
    release: str
    parents: tuple[Mapping[str, Any], ...]
    body: Mapping[str, Any]
    pinned: bool = True
    evaluated: bool = False
    outcomes: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    draft: bool = False
    actors: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        """One word for the proposal, the latest decision (by ``created_at``) winning.

        Returns:
            str: ``draft``, ``migrated``, ``promoted``, ``rolled-back``, ``rejected``, ``abandoned``,
                ``evaluated`` or ``pinned``.
        """
        if self.draft and not self.pinned:
            return "draft"
        for kind in reversed(self.decisions):
            if kind in _STATUS_OF:
                return _STATUS_OF[kind]
        return "evaluated" if self.evaluated else "pinned"

    @property
    def decided_by(self) -> str:
        """Who made the decision behind the status word.

        Returns:
            str: The actor of the latest status-setting decision (``auto:outcome`` for a promotion ``outcome``
                performed), or ``""`` when no decision sets the status.
        """
        for kind, actor in zip(reversed(self.decisions), reversed(self.actors)):
            if kind in _STATUS_OF:
                return actor
        return ""


@dataclass
class Lineage:
    """Everything derived from the store and git for the tree, the log and the status line.

    Attributes:
        active (str | None): The BASELINE release.
        releases (dict[str, ReleaseState]): Every release in the store.
        proposals (dict[str, ProposalState]): Every proposal in the store (plus drafts the caller names).
        revoked (Mapping[str, str]): Content key to decision id.
        decisions (tuple[Mapping[str, Any], ...]): Decision documents in file order.
        history (tuple[str, ...]): BASELINE history from git, oldest first.
    """

    active: str | None
    releases: dict[str, ReleaseState]
    proposals: dict[str, ProposalState]
    revoked: Mapping[str, str]
    decisions: tuple[Mapping[str, Any], ...]
    history: tuple[str, ...]


def _ids(directory: Path, pattern: str) -> list[str]:
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob(pattern))


def derive(store: Store, *, drafts: Iterable[Mapping[str, Any]] = ()) -> Lineage:
    """Derive the lifecycle of every release and proposal from the package store and git.

    Args:
        store (Store): The store; only its package layer carries integrity files.
        drafts (Iterable[Mapping[str, Any]]): Proposal bodies that exist only in the control workspace
            (``{"id", "release", "parents", ...}``); they appear as ``draft`` unless a pinned file exists.

    Returns:
        Lineage: The derived states.

    Raises:
        PromptStoreError: If a release, proposal or decision file is malformed.
    """
    package = store.package
    scope = store.v0_scope()
    try:
        active = store.baseline().release
    except PromptStoreError:
        active = None
    history = tuple(baseline_history(package))
    releases: dict[str, ReleaseState] = {}
    for rid in _ids(package / "releases", "r*.json"):
        if RELEASE_RE.fullmatch(rid) and (scope is None or rid in scope.releases):
            releases[rid] = ReleaseState(rid, active=(rid == active), superseded=(rid in history and rid != active), retired=store.is_retired(rid))
    for rid in history:
        releases.setdefault(rid, ReleaseState(rid, active=(rid == active), superseded=(rid != active)))
    proposals: dict[str, ProposalState] = {}
    for pid in _ids(package / "proposals", "p*.json"):
        if not PROPOSAL_RE.fullmatch(pid) or (scope is not None and pid not in scope.proposals):
            continue
        body = store.proposal(pid).data
        release = body.get("release")
        if not isinstance(release, str) or not RELEASE_RE.fullmatch(release):
            raise PromptStoreError(f"proposals/{pid}.json: release {release!r} is not a release id")
        outcomes = tuple(_ids(package / "outcomes" / pid, "*.json")) + tuple(path.stem for path in sorted((package / "outcomes").glob(f"{pid}-*.json")))
        proposals[pid] = ProposalState(pid, release, tuple(body.get("parents") or ()), body, pinned=True, evaluated=bool(outcomes), outcomes=outcomes)
        releases.setdefault(release, ReleaseState(release, active=(release == active), superseded=(release in history and release != active))).proposals.append(pid)
    for body in drafts:
        pid = str(body.get("id"))
        if scope is not None and (not body.get("parents") or any(parent.get("release") not in scope.releases for parent in body["parents"])):
            continue
        if pid in proposals:
            continue
        release = str(body.get("release", ""))
        proposals[pid] = ProposalState(pid, release, tuple(body.get("parents") or ()), dict(body), pinned=False, draft=True)
    # Decision files are content-addressed, so the store lists them by name; the lifecycle
    # rule is "the latest decision wins", which needs their creation order.
    decisions = tuple(sorted((d for d in store.decisions()
                              if scope is None or d.get("target") in scope.releases | scope.proposals),
                             key=lambda d: (float(d.get("created_at") or 0.0), str(d.get("id", "")))))
    for decision in decisions:
        kind = decision.get("kind")
        target = decision.get("target")
        if kind not in DECISION_KINDS or not isinstance(target, str):
            raise PromptStoreError(f"decision {decision.get('id', '?')}: needs kind in {list(DECISION_KINDS)} and a target id")
        actor = str(decision.get("actor") or "")
        if target in proposals:
            proposals[target].decisions = proposals[target].decisions + (kind,)
            proposals[target].actors = proposals[target].actors + (actor,)
        if kind in POINTER_KINDS and kind != "rollback" and target in proposals:
            releases.setdefault(proposals[target].release, ReleaseState(proposals[target].release))
        if kind == "rollback":
            # a rollback targets a release; the proposals whose promotion produced the BASELINE it
            # undid are listed on the decision (``demoted``) and show ``rolled-back`` from here on
            for pid in decision.get("demoted") or []:
                if isinstance(pid, str) and pid in proposals:
                    proposals[pid].decisions = proposals[pid].decisions + ("rollback",)
                    proposals[pid].actors = proposals[pid].actors + (actor,)
    return Lineage(active, releases, proposals, store.revoked(), decisions, history)


# ---------------------------------------------------------------------------------------
# Proposal tree


@dataclass
class TreeNode:
    """One node of the proposal tree.

    Attributes:
        id (str): Release id for a root, proposal id otherwise.
        kind (str): ``"release"`` or ``"proposal"``.
        release (str): The release the node stands for or produces.
        parents (tuple[str, ...]): Parent release ids (``base`` first, then ``merge`` parents).
        status (str): Derived status word.
        decided_by (str): Who promoted the proposal (``auto:outcome`` or a person), shown after ``promoted``; empty
            for every other status.
        same_as (str | None): ``≡`` marker: an earlier node's release this node reproduces byte-for-byte.
        revert_of (str | None): ``↩`` marker: the proposal whose change this node undoes.
        children (list[TreeNode]): Proposals based on this node's release, in creation order.
        body (Mapping[str, Any]): The proposal body (empty for a release root).
    """

    id: str
    kind: str
    release: str
    parents: tuple[str, ...]
    status: str
    decided_by: str = ""
    same_as: str | None = None
    revert_of: str | None = None
    children: list["TreeNode"] = field(default_factory=list)
    body: Mapping[str, Any] = field(default_factory=dict)


def _base_parent(state: ProposalState) -> str | None:
    for parent in state.parents:
        if isinstance(parent, Mapping) and parent.get("role", "base") == "base" and isinstance(parent.get("release"), str):
            return parent["release"]
    return None


def _creation_key(state: ProposalState) -> tuple[float, str]:
    created = state.body.get("created_at")
    return (float(created) if isinstance(created, (int, float)) else float("inf"), state.id)


def proposal_tree(lineage: Lineage) -> list[TreeNode]:
    """Build the proposal tree with ``≡`` and ``↩`` markers.

    Roots are the releases that no proposal produced (the migration baseline and imported
    releases). A proposal hangs under the first node whose release is its base parent. A node
    whose release an earlier node already carries is marked ``same_as`` that node's release; a
    proposal whose release equals its base parent's own base parent is marked ``revert_of`` that
    parent proposal (an explicit ``revert_of`` field in the body wins).

    Args:
        lineage (Lineage): Output of :func:`derive`.

    Returns:
        list[TreeNode]: Root nodes, the active release first.
    """
    produced = {state.release for state in lineage.proposals.values()}
    anchored = set(lineage.history) | ({lineage.active} if lineage.active else set())
    roots: list[TreeNode] = []
    by_release: dict[str, TreeNode] = {}
    root_ids = sorted(lineage.releases, key=lambda rid: (rid != lineage.active, rid))
    for rid in root_ids:
        # A release some proposal produced is not a root, unless it is (or was) the baseline: a
        # revert proposal reproduces root's bytes without becoming root's origin.
        if rid in produced and rid not in anchored:
            continue
        node = TreeNode(rid, "release", rid, (), lineage.releases[rid].status)
        roots.append(node)
        by_release.setdefault(rid, node)
    ordered = sorted(lineage.proposals.values(), key=_creation_key)
    pending = list(ordered)
    placed: dict[str, TreeNode] = {}
    progress = True
    while pending and progress:
        progress = False
        for state in list(pending):
            base = _base_parent(state)
            parent_node = by_release.get(base) if base is not None else None
            if parent_node is None and base is not None and base in lineage.releases and (base not in produced or base in anchored):
                parent_node = TreeNode(base, "release", base, (), lineage.releases[base].status)
                roots.append(parent_node)
                by_release[base] = parent_node
            if parent_node is None and base is not None:
                continue
            node = TreeNode(
                state.id,
                "proposal",
                state.release,
                tuple(parent["release"] for parent in state.parents if isinstance(parent, Mapping) and isinstance(parent.get("release"), str)),
                state.status,
                decided_by=state.decided_by if state.status == "promoted" else "",
                body=state.body,
            )
            earlier = by_release.get(state.release)
            if earlier is not None and earlier.release == state.release:
                node.same_as = earlier.release
            explicit = state.body.get("revert_of")
            if isinstance(explicit, str):
                node.revert_of = explicit
            elif parent_node is not None and parent_node.kind == "proposal":
                grand = _base_parent(lineage.proposals[parent_node.id]) if parent_node.id in lineage.proposals else None
                if grand is not None and grand == state.release:
                    node.revert_of = parent_node.id
            if parent_node is not None:
                parent_node.children.append(node)
            else:
                roots.append(node)
            by_release.setdefault(state.release, node)
            placed[state.id] = node
            pending.remove(state)
            progress = True
    for state in pending:
        node = TreeNode(state.id, "proposal", state.release, tuple(), state.status, body=state.body)
        roots.append(node)
    return roots


def render_tree(roots: Sequence[TreeNode], *, label: Any = None) -> str:
    """Render the tree as indented text with ``├─``/``└─`` connectors and the two markers.

    Args:
        roots (Sequence[TreeNode]): Output of :func:`proposal_tree`.
        label (Callable[[TreeNode], str] | None): Extra text per node (claims, effects); none by default.

    Returns:
        str: One line per node.
    """
    lines: list[str] = []

    def describe(node: TreeNode) -> str:
        text = f"{node.id} {'★ ' if node.status == 'active' else ''}{node.status}" + (f" by {node.decided_by}" if node.decided_by else "")
        if node.same_as:
            text += f"  ≡ {node.same_as}"
        if node.revert_of:
            text += f"  ↩ revert of {node.revert_of}"
        if label is not None:
            extra = label(node)
            if extra:
                text += f"  {extra}"
        return text

    def walk(node: TreeNode, prefix: str, last: bool, top: bool) -> None:
        if top:
            lines.append(describe(node))
            child_prefix = ""
        else:
            lines.append(f"{prefix}{'└─ ' if last else '├─ '}{describe(node)}")
            child_prefix = prefix + ("   " if last else "│  ")
        for index, child in enumerate(node.children):
            walk(child, child_prefix, index == len(node.children) - 1, False)

    for root in roots:
        walk(root, "", True, True)
    return "\n".join(lines)
