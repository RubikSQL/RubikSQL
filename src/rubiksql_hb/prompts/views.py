"""Read-side history: status, log, show, diff, release-diff, render blame, tree and report.

Every view works from a :class:`store.Store` alone (no control workspace, no HeavenBase): rows
are read from the object files, releases through :func:`runtime.load_release`, lifecycle and
display version numbers from :mod:`lineage`, statistics from the outcome documents. The
reference syntax accepts exactly two forms — ``name:<hex prefix of at least 6 digits>`` and
``name@<baseline|r…|p…>`` — because ``vN`` is display-only and a bare name would silently mean
"whatever is active". The typed versions a ``rubiksql.closure/2`` release references are named by
their typed keys ``tool:<name>``, ``rule:<title>`` and ``parameters:knobs`` in the same two forms
(``tool:run_sql@baseline``, ``rule:<title>:<hex prefix>``); ``log`` takes the bare key. Every
function returns lines (or one string) so the CLI, the assistant tools and the HTML report share
one rendering.
"""
from __future__ import annotations

import difflib
import html
import json
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping, Sequence

if TYPE_CHECKING:
    from .authority import AuthorityRead
    from .types import VerifiedSnapshot

from . import entries as E
from . import lineage as L
from .canon import canon, sha256_hex
from .cells import all_cells
from .doc import MESSAGE_SEPARATOR, DocError, document_of, parse_document
from .entity_model import CLOSURE_V2, PARAMETER_ENTITY, RULE_ENTITY, TOOL_ENTITY, EntityModelError, closure_changes, closure_view
from .gate import CHANGE_RATIO_WARN, RENDER_CONTRACTS, RootRun, change_stat, changed_surfaces, check_change_ratio, estimate_cost, keyspace_moves, row_text
from .matrix import Cell
from .policy import PROPOSAL_RE, RELEASE_RE, VERSION_RE, PolicyError, check_name
from .store import PromptStoreError, Store

__all__ = [
    "RowRef",
    "blame_lines",
    "checkout_line",
    "diff_lines",
    "draft_text",
    "history_store",
    "load",
    "log_lines",
    "outcome_line",
    "release_diff_lines",
    "releases_with",
    "report_html",
    "resolve_ref",
    "show_lines",
    "spec_diff",
    "status_lines",
    "tree_lines",
    "typed_lines",
    "typed_versions",
    "word_diff",
]

_TOKEN_RE = re.compile(r"\S+\s*|\s+")
_SHORT = "rubiksql."
#: Typed history key prefixes (``tool:<name>``, ``rule:<title>``, ``parameters:knobs``) and their stored entities.
_TYPED_KEYS = {"tool": TOOL_ENTITY, "rule": RULE_ENTITY, "parameters": PARAMETER_ENTITY}

_CSS = """
body{font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;color:#1b1b1b;background:#fff}
h1,h2{font-weight:600} pre{font:12px/1.4 ui-monospace,Menlo,monospace;background:#f6f7f9;padding:10px;overflow:auto;border-radius:6px}
table{border-collapse:collapse;margin:8px 0} td,th{border:1px solid #d9dde3;padding:3px 8px;font-size:13px;text-align:left;vertical-align:top}
th{background:#eef1f5} .ins{background:#dcf5dc} .del{background:#f9d7d7;text-decoration:line-through}
.heat td.same{background:#eef7ee} .heat td.changed{background:#f7d9c4} .heat td.absent{background:#eee;color:#888}
.side{display:grid;grid-template-columns:1fr 1fr;gap:12px} .side pre{white-space:pre-wrap}
.status{font-variant:small-caps;color:#555}
"""


@contextmanager
def history_store(reader: AuthorityRead | None = None, *, snapshot: VerifiedSnapshot | None = None,
                  require_export: bool = True) -> Iterator[Store]:
    """Read history from one authority transaction or an exact verified snapshot.

    Args:
        reader (AuthorityRead | None): Already-open read-only authority transaction.
            When omitted, open the package binding's authority read-only.
        snapshot (VerifiedSnapshot | None): Explicit execution snapshot to read instead.
        require_export (bool): Require a fresh export. A writer's already-held
            transaction may pass False for the verified intermediate batch.

    Yields:
        Store: Detached store-layout view, valid until context exit. No authority or
            native registry row is written; temporary files contain only verified data.

    Raises:
        PromptStoreError: The authority, snapshot, or exported file layout is invalid.
    """
    if not require_export:
        from .authority import AuthorityRead

        if snapshot is not None or not isinstance(reader, AuthorityRead) or not reader.writer_lock_held:
            raise PromptStoreError("intermediate history requires an active locked authority writer reader")
    if reader is not None and snapshot is not None:
        raise PromptStoreError("history needs an authority reader or a snapshot, not both")
    if reader is None and snapshot is None and os.environ.get("RUBIKSQL_PROMPT_SNAPSHOT"):
        from .snapshot import verify_snapshot

        snapshot = verify_snapshot(Path(os.environ["RUBIKSQL_PROMPT_SNAPSHOT"]))
    if snapshot is not None:
        from .snapshot import verify_snapshot

        verified = verify_snapshot(snapshot.path, expected_digest=snapshot.digest)
        store = Store(package=verified.path)
        store.history_head = None
        store.history_snapshot = verified
        store.history_events = _events_at(verified.path)
        yield store
        return
    if reader is None:
        from .authority import Authority
        from .binding import load_binding

        binding = load_binding()
        with Authority.read(binding=binding) as opened:
            with history_store(opened, require_export=require_export) as store:
                yield store
        return
    verified = reader.verify(require_export=require_export)
    if not verified.ok:
        raise PromptStoreError(f"authority history failed verification: {verified.violations}")
    files = reader.store_files()
    with tempfile.TemporaryDirectory(prefix="rubiksql-history-") as temporary:
        root = Path(temporary)
        for relative, content in files.items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise PromptStoreError(f"invalid history file path {relative!r}")
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        store = Store(package=root)
        store.history_head = reader.head()
        store.history_events = tuple(reader.events())
        yield store


def _events_at(root: Path) -> tuple[Mapping[str, Any], ...]:
    path = root / "events.jsonl"
    if not path.is_file():
        return ()
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)


def native_rows(store: Store, name: str | None = None) -> tuple[Mapping[str, Any], ...]:
    """Read complete native callable, module, serializer and historical toolkit rows.

    Args:
        store (Store): Verified authority mapping or snapshot.
        name (str | None): Exact native name, or all available native rows.

    Returns:
        tuple[Mapping[str, Any], ...]: Deterministically ordered complete native rows.
            Both authority views and run snapshots carry the complete epoch history.
            Executable bundle files remain limited to the selected pin.
    """
    rows = []
    for entity in ("sys-capsule", "sys-toolkit"):
        path = store.package / "native" / f"{entity}.json"
        if path.is_file():
            rows.extend(row for row in json.loads(path.read_text(encoding="utf-8"))
                        if row.get("namespace") in {"rubiksql.answering", "rubiksql.answering.module", "rubiksql.answering.tool"})
    publications = [event for event in getattr(store, "history_events", _events_at(store.package)) if event.get("kind") == "publish"]

    def introduced(row: Mapping[str, Any]) -> int:
        identity = str(row.get("object_id"))
        for index, event in enumerate(publications):
            if identity in json.dumps(event.get("payload") or {}, ensure_ascii=False):
                return index
        return len(publications)

    return tuple(sorted((row for row in rows if name is None or row.get("name") == name),
                        key=lambda row: (introduced(row), str(row.get("object_id")))))


def _native_carriers(store: Store, object_id: str) -> list[str]:
    def contains(value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(contains(item) for item in value.values())
        if isinstance(value, list):
            return any(contains(item) for item in value)
        return value == object_id

    found = []
    scope = store.v0_scope()
    for rid in sorted(scope.releases if scope is not None else ()):
        document = store.release(rid).data
        if document.get("rules") == CLOSURE_V2:
            # A typed release pins its module and serializer Capsules in its loader references.
            if contains(document["content_refs"]):
                found.append(rid)
            continue
        refs = document["refs"]
        key = refs.get("tools.answering")
        if key:
            row = store.object("tools.answering", key[:16]).data["row"]
            if contains(json.loads(row["manifest"]["meta"]["text"])):
                found.append(rid)
    return found


def history_events(store: Store, *, name: str | None = None) -> tuple[Mapping[str, Any], ...]:
    """Return complete authority receipts relevant to a row, or every receipt.

    Args:
        store (Store): Authority or snapshot store view.
        name (str | None): Row to filter publications by; global authority/export
            receipts and decisions remain visible.

    Returns:
        tuple[Mapping[str, Any], ...]: Ordered event records, with no shortened payload.
    """
    events = getattr(store, "history_events", _events_at(store.package))
    relevant = set()
    if name is not None and _typed_key(name) is not None:
        relevant = {rid for _row, carriers in typed_versions(store, name) for rid in carriers}
    elif name is not None:
        relevant = {rid for version, row in _versions(store, name).items()
                    for rid in releases_with(store, name, row["manifest"]["meta"]["content_key"])}
        relevant.update(rid for row in native_rows(store, name) for rid in _native_carriers(store, str(row["object_id"])))
    lines = []
    for event in events:
        payload = event.get("payload") or {}
        if name is not None and event.get("kind") in {"publish", "gate", "outcome", "eval-attach"}:
            release = payload.get("release") or payload.get("body", {}).get("release")
            if release is not None and release not in relevant:
                continue
        lines.append(event)
    return tuple(lines)


def _typed_key(text: str) -> tuple[str, str] | None:
    """``(entity, logical name)`` of a typed history key, or None when the text names a row."""
    prefix, separator, name = text.partition(":")
    if not separator or prefix not in _TYPED_KEYS:
        return None
    if not name or ":" in name or "@" in name:
        raise PromptStoreError(f"{text!r}: a typed key is tool:<name>, rule:<title> or parameters:knobs")
    return _TYPED_KEYS[prefix], name


def _typed_members(store: Store, document: Mapping[str, Any]) -> dict[str, tuple[str, Mapping[str, Any]]]:
    """Typed key -> (entity, referenced canonical row) of one release document, in reference order; {} for closure/1."""
    if document.get("rules") != CLOSURE_V2:
        return {}
    index = store.typed_index()
    content = document["content_refs"]
    members = {}
    for entity, ref in (*((TOOL_ENTITY, ref) for ref in content["tools"]), *((RULE_ENTITY, ref) for ref in content["rules"]["items"]),
                        (PARAMETER_ENTITY, content["parameters"])):
        row = index[entity].get(ref["object_id"])
        if row is None:
            raise PromptStoreError(f"{entity} {ref['object_id']} referenced by release {document.get('id')} is absent")
        label = {TOOL_ENTITY: "tool", RULE_ENTITY: "rule", PARAMETER_ENTITY: "parameters"}[entity]
        members[f"{label}:{row['title'] if entity == RULE_ENTITY else row['name']}"] = (entity, row)
    return members


def _release_order(store: Store) -> list[str]:
    """In-scope release ids (every release without a v0 marker), the root first, then by first publication."""
    scope = store.v0_scope()
    directory = store.package / "releases"
    if scope is not None:
        rids = set(scope.releases)
    else:
        rids = {path.stem for path in directory.glob("r*.json") if RELEASE_RE.fullmatch(path.stem)} if directory.is_dir() else set()
    published: dict[str, int] = {}
    for index, event in enumerate(getattr(store, "history_events", _events_at(store.package))):
        payload = event.get("payload") or {}
        release = payload.get("release") or (payload.get("body") or {}).get("release") if event.get("kind") == "publish" else None
        if isinstance(release, str):
            published.setdefault(release, index)
    root = scope.root if scope is not None else None
    return sorted(rids, key=lambda rid: (rid != root, published.get(rid, len(published)), rid))


def typed_versions(store: Store, key: str) -> list[tuple[Mapping[str, Any], list[str]]]:
    """The versions of one typed key that in-scope releases reference, newest last.

    Args:
        store (Store): Verified authority mapping or snapshot.
        key (str): ``tool:<name>``, ``rule:<title>`` or ``parameters:knobs``.

    Returns:
        list[tuple[Mapping[str, Any], list[str]]]: ``(canonical row, release ids carrying it)``, ordered by the
            first release (root first, then publication order) that references each version.

    Raises:
        PromptStoreError: The key is malformed or a referenced row is absent.
    """
    if _typed_key(key) is None:
        raise PromptStoreError(f"{key!r} is not a typed key (tool:<name>, rule:<title> or parameters:knobs)")
    rows: dict[str, Mapping[str, Any]] = {}
    carriers: dict[str, list[str]] = {}
    for rid in _release_order(store):
        member = _typed_members(store, store.release(rid).data).get(key)
        if member is not None:
            rows.setdefault(member[1]["object_id"], member[1])
            carriers.setdefault(member[1]["object_id"], []).append(rid)
    return [(rows[object_id], carriers[object_id]) for object_id in carriers]


def typed_lines(store: Store, release: Any) -> list[str]:
    """List the typed versions a closure/2 release references, for ``prompts ls``.

    Args:
        store (Store): The store holding the release.
        release (Release): A loaded release.

    Returns:
        list[str]: ``<typed key>  <entity>  <version>`` per reference in release order (tools in serve order,
            rules in corpus order, then the parameter set); empty for a closure/1 release.
    """
    members = _typed_members(store, store.release(release.id).data)
    width = max((len(key) for key in members), default=4)
    return [f"{key.ljust(width)}  {entity:19}  {row['version']}" for key, (entity, row) in members.items()]


def _typed_log_lines(store: Store, key: str) -> list[str]:
    versions = typed_versions(store, key)
    if not versions:
        raise PromptStoreError(f"{key!r} has no version referenced by a release of this history")
    try:
        active = store.baseline().release
    except PromptStoreError:
        active = None
    lines = [f"{key}  {_typed_key(key)[0]} · {len(versions)} version{'' if len(versions) == 1 else 's'} (newest last)"]
    for number, (row, carriers) in enumerate(versions, 1):
        star = f"  ★ in BASELINE {active}" if active in carriers else ""
        lines.append(f"  v{number}  {row['version']}  {row['object_id']}  releases {', '.join(carriers)}{star}")
    return lines + event_lines(store, name=key)


def _typed_by_digest(index: Mapping[str, Mapping[str, Mapping[str, Any]]], key: str, digest: Any) -> Mapping[str, Any] | None:
    typed = _typed_key(key)
    if typed is None or not isinstance(digest, str):
        return None
    return next((row for row in index[typed[0]].values() if row.get("digest") == digest), None)


def event_lines(store: Store, *, name: str | None = None) -> list[str]:
    """Render complete receipt records as canonical JSON lines.

    Args:
        store (Store): Verified authority or snapshot store view.
        name (str | None): Optional row whose publications should be selected.

    Returns:
        list[str]: One complete canonical JSON record per matching event.
    """
    return ["event " + json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for event in history_events(store, name=name)]


def _short(name: str) -> str:
    return name[len(_SHORT):] if name.startswith(_SHORT) else name


def _fp8(text: str) -> str:
    return sha256_hex(text)[:8]


def load(store: Store, ref: str, *, allow_retired: bool = True) -> Any:
    """Load a release by ``baseline``, release id or proposal id.

    Args:
        store (Store): The store.
        ref (str): ``baseline``, ``r<16hex>`` or ``p<16hex>``.
        allow_retired (bool): Load retired releases too (views show history).

    Returns:
        Release: The verified release.

    Raises:
        PromptStoreError: If the reference has another shape or the load fails.
    """
    from .runtime import load_release

    if ref == "baseline":
        ref = store.baseline().release
    if not (RELEASE_RE.fullmatch(ref) or PROPOSAL_RE.fullmatch(ref)):
        raise PromptStoreError(f"{ref!r} is not 'baseline', a release id (r<16hex>) or a proposal id (p<16hex>)")
    return load_release(ref, stores=store, allow_retired=allow_retired)


def checkout_line(repo: Path) -> str:
    """``checkout <branch>@<sha> (clean|dirty)`` for the checkout that holds the store.

    Args:
        repo (Path): Repository root.

    Returns:
        str: The line; ``checkout ? (not a git checkout)`` outside git.
    """
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        branch = subprocess.run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "checkout ? (not a git checkout)"
    return f"checkout {branch}@{head} ({'dirty' if dirty else 'clean'})"


# ---------------------------------------------------------------------------------------
# References


@dataclass(frozen=True)
class RowRef:
    """One resolved row reference.

    Attributes:
        name (str): Row name.
        version (str): 16-hex version.
        content_key (str): Full content key.
        row (Mapping[str, Any]): The exported HB row.
        release (Any | None): The release the reference named (``name@…``), or a release carrying the version.
        pin (str | None): The ``@`` part as typed, or ``None`` for ``name:prefix``.
    """

    name: str
    version: str
    content_key: str
    row: Mapping[str, Any]
    release: Any | None
    pin: str | None


def releases_with(store: Store, name: str, content_key: str) -> list[str]:
    """Release ids of the package store whose refs carry a row version.

    Args:
        store (Store): The store.
        name (str): Row name.
        content_key (str): Full content key.

    Returns:
        list[str]: Release ids, the active BASELINE first when it is one of them.
    """
    directory = store.package / "releases"
    scope = store.v0_scope()
    found = []
    for path in sorted(directory.glob("r*.json")) if directory.is_dir() else []:
        if not RELEASE_RE.fullmatch(path.stem) or (scope is not None and path.stem not in scope.releases):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, Mapping) and (data.get("refs") or {}).get(name) == content_key:
            found.append(path.stem)
    try:
        active = store.baseline().release
    except PromptStoreError:
        active = None
    return sorted(found, key=lambda rid: (rid != active, rid))


def resolve_ref(ref: str, store: Store) -> RowRef:
    """Resolve ``name@<baseline|r…|p…>`` or ``name:<hex prefix ≥ 6>``; nothing else.

    Args:
        ref (str): The reference.
        store (Store): The store.

    Returns:
        RowRef: The row, and the release it was named through (or one carrying it, for ``name:prefix``).

    Raises:
        PromptStoreError: On any other syntax, an unknown name, an ambiguous prefix, or a row absent from the release.
    """
    if ref.partition(":")[0] in _TYPED_KEYS:
        return _resolve_typed(ref, store)
    if "@" in ref:
        name, pin = ref.split("@", 1)
        try:
            check_name(name)
        except PolicyError as error:
            raise PromptStoreError(str(error)) from error
        release = load(store, pin)
        if name not in release.refs:
            matches = [row for row in native_rows(store, name) if release.id in _native_carriers(store, str(row["object_id"]))]
            if len(matches) == 1:
                row = matches[0]
                return RowRef(name, str(row["version"]), sha256_hex(canon(row)), row, release, pin)
            raise PromptStoreError(f"{name!r} is not a row of {release.id}")
        key = release.refs[name]
        return RowRef(name, key[:16], key, release.row(name), release, pin)
    if ":" in ref:
        name, prefix = ref.rsplit(":", 1)
        try:
            check_name(name)
        except PolicyError as error:
            raise PromptStoreError(str(error)) from error
        if len(prefix) < 6 or not all(char in "0123456789abcdef" for char in prefix):
            raise PromptStoreError(f"{ref!r}: the version prefix must be at least 6 hex digits")
        native = [row for row in native_rows(store, name) if str(row.get("version", "")).startswith(prefix)]
        if native:
            if len(native) != 1:
                raise PromptStoreError(f"{ref!r} matches {len(native)} native versions")
            row = native[0]
            carriers = _native_carriers(store, str(row["object_id"]))
            release = load(store, carriers[0]) if carriers else None
            return RowRef(name, str(row["version"]), sha256_hex(canon(row)), row, release, None)
        matches = sorted({path.stem for layer in store.layers for path in (layer.path / "objects" / name).glob(f"{prefix}*.json") if VERSION_RE.fullmatch(path.stem)})
        if len(matches) != 1:
            raise PromptStoreError(f"{ref!r} matches {len(matches)} versions: {matches}")
        row = store.object(name, matches[0]).data["row"]
        key = row["manifest"]["meta"]["content_key"]
        carriers = releases_with(store, name, key)
        release = load(store, carriers[0]) if carriers else None
        return RowRef(name, matches[0], key, row, release, None)
    raise PromptStoreError(f"{ref!r}: a reference is name@<baseline|r…|p…> or name:<hex prefix of at least 6 digits>; vN is display-only")


def _resolve_typed(ref: str, store: Store) -> RowRef:
    """Resolve ``<typed key>@<baseline|r…|p…>`` or ``<typed key>:<hex prefix ≥ 6>`` to a canonical typed row."""
    if "@" in ref:
        key, pin = ref.split("@", 1)
        _typed_key(key)
        release = load(store, pin)
        member = _typed_members(store, store.release(release.id).data).get(key)
        if member is None:
            raise PromptStoreError(f"{key!r} is not a typed reference of {release.id}")
        row = member[1]
        return RowRef(key, str(row["version"]), str(row["digest"]), row, release, pin)
    key, _separator, prefix = ref.rpartition(":")
    if _typed_key(key) is None:
        raise PromptStoreError(f"{ref!r}: a typed reference is <key>@<baseline|r…|p…> or <key>:<hex prefix of at least 6 digits>")
    if len(prefix) < 6 or not all(char in "0123456789abcdef" for char in prefix):
        raise PromptStoreError(f"{ref!r}: the version prefix must be at least 6 hex digits")
    matches = [(row, carriers) for row, carriers in typed_versions(store, key) if str(row["version"]).startswith(prefix)]
    if len(matches) != 1:
        raise PromptStoreError(f"{ref!r} matches {len(matches)} typed versions")
    row, carriers = matches[0]
    try:
        active = store.baseline().release
    except PromptStoreError:
        active = None
    return RowRef(key, str(row["version"]), str(row["digest"]), row, load(store, active if active in carriers else carriers[-1]), None)


# ---------------------------------------------------------------------------------------
# Row history helpers


def _versions(store: Store, name: str) -> dict[str, Mapping[str, Any]]:
    """Every stored version of a name, by 16-hex version."""
    versions: dict[str, Mapping[str, Any]] = {}
    scope = store.v0_scope()
    allowed = store.v0_versions(scope, name) if scope is not None else None
    for layer in store.layers:
        directory = layer.path / "objects" / name
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
            if VERSION_RE.fullmatch(path.stem) and path.stem not in versions and (allowed is None or path.stem in allowed):
                row = json.loads(path.read_text(encoding="utf-8"))["row"]
                versions[path.stem] = row
    return versions


def _version_dates(store_dir: Path, name: str) -> dict[str, str]:
    """Version to the short date of the commit that first added its file (``uncommitted`` otherwise)."""
    from datetime import datetime, timezone

    return {path.stem: datetime.fromtimestamp(float(json.loads(path.read_text(encoding="utf-8"))["row"].get("created_at") or 0), timezone.utc).date().isoformat()
            for path in sorted((store_dir / "objects" / name).glob("*.json"))}


def _proposals_introducing(store: Store, lineage: L.Lineage, name: str) -> dict[str, list[str]]:
    """Content key to the proposals whose derived changes introduce it for ``name``."""
    by_key: dict[str, list[str]] = {}
    for pid, state in sorted(lineage.proposals.items(), key=lambda item: float(item[1].body.get("created_at") or 0.0)):
        change = (state.body.get("changes") or {}).get(name)
        if isinstance(change, Mapping) and isinstance(change.get("to"), str):
            by_key.setdefault(change["to"], []).append(pid)
    return by_key


def _shape(meta: Mapping[str, Any]) -> str:
    """``slots: …`` for a text row, ``params: …`` for a doc row, ``params: … · parts: N`` for a spec row."""
    if meta["kind"] == "text":
        return f"slots: {', '.join(meta['slots']) or '-'}"
    if meta["kind"] == "doc":
        return f"params: {', '.join(meta['params']) or '-'}"
    return f"params: {', '.join(meta['params']) or '-'} · parts: {len(meta['parts'])}"


def _author(row: Mapping[str, Any]) -> str:
    meta = row["manifest"]["meta"]
    if meta.get("author"):
        return str(meta["author"])
    return str(meta.get("origin") or "?")


def _entry_label(name: str) -> str:
    """``entry``, ``dropped`` (an entry the code no longer renders, :data:`entries.DROPPED_ENTRIES`) or ``part``."""
    if name in E.ENTRIES:
        return "entry"
    return "dropped" if name in E.DROPPED_ENTRIES else "part"


def _visibility(name: str) -> str:
    if name == "tools.answering":
        return "tool implementation"
    entry = E.ENTRIES.get(name)
    if entry is None:
        return "-" if name in E.DROPPED_ENTRIES else "part"  # a dropped entry joins no keyspace
    if entry.slot is not None:
        return f"candidate/* slot {entry.slot}"
    if name == E.KNOBS_ENTRY:
        return "per knob, when moved"  # a knob joins its roles' keyspaces only with a value other than its default
    if entry.optional:
        return "knowledge"  # the rule corpus and the generators change what a build writes: the knowledge fingerprint
    labels = []
    for role, routes in E.ROUTES.items():
        mine = [route for route in routes if (role, route) in entry.visible]
        if len(mine) == len(routes):
            labels.append(f"{role}/*")
        else:
            labels.extend(E.route_label(role, route) for route in mine)
    return ", ".join(labels) or "-"


def _verdict_text(doc: Mapping[str, Any]) -> str:
    """The verdict word with the tag of its reason, e.g. ``not-promotable [gain below threshold]``."""
    reason = str(doc.get("verdict_reason") or "")
    return str(doc.get("verdict")) + (f" [{reason.split(':', 1)[0]}]" if reason else "")


def _gain_p(doc: Mapping[str, Any]) -> tuple[str, str]:
    """The EX gain and p-value of a paired outcome as its lines print them: on the side of the recorded thresholds
    their verdict says (:func:`evolve.format_gain`, :func:`evolve.format_p`)."""
    from .evolve import format_gain, format_p

    stats = doc.get("stats") or {}
    rule = doc.get("rule") if isinstance(doc.get("rule"), Mapping) else None
    return format_gain(float(stats["gain_pp"]), rule), format_p(float(stats.get("p") if stats.get("p") is not None else 1.0), rule)


def _ex_arms(stats: Mapping[str, Any]) -> str:
    """``EX 72.00% -> 73.50% `` from the arm means of a paired outcome, or ``EX `` when they are absent."""
    if stats.get("parent_mean") is None or stats.get("child_mean") is None:
        return "EX "
    return f"EX {100 * float(stats['parent_mean']):.2f}% -> {100 * float(stats['child_mean']):.2f}% "


def outcome_line(doc: Mapping[str, Any]) -> str:
    """One line for an outcome document, in the log's format.

    Args:
        doc (Mapping[str, Any]): An outcome document.

    Returns:
        str: ``EX 72.12% -> 73.64% (+1.52pp; c=12, b=7; n=330; exposed 16 questions) p 0.18 · attempt 1 → promotable ·
            integrity ok (…)``; a run that could not be paired shows its descriptive mean instead. The p-value is
            information under the rule in force, which has no significance test.
    """
    stats = doc.get("stats") or {}
    detail = doc.get("integrity_detail") or {}
    divergences = detail.get("divergences") or {}
    noise = ", ".join(f"{count} {cause}" for cause, count in sorted(divergences.items()) if count and cause in ("tool", "clock", "unknown", "no-source"))
    integrity = f"integrity {doc.get('integrity')}" + (f" ({noise} divergences)" if noise else "")
    verdict = _verdict_text(doc)
    if "rule" not in doc:
        return f"recorded before rule A: verdict {doc.get('verdict')} (not recomputed, carries no promotion) · run {doc.get('run')}"
    if stats.get("gain_pp") is None:
        mean = stats.get("mean")
        head = f"descriptive (n={stats.get('n')}" + (f", mean {mean:.4f}" if isinstance(mean, float) else "") + ")"
        return f"{head} → {verdict} · {integrity} · run {doc.get('run')}"
    exposed = (stats.get("exposed") or {}).get("questions", 0)
    attempt = int((doc.get("attempt") or {}).get("used") or 0) + 1
    gain, p = _gain_p(doc)
    return f"{_ex_arms(stats)}({gain}pp; c={stats.get('c')}, b={stats.get('b')}; n={stats.get('n')}; exposed {exposed} questions) p {p} · attempt {attempt} → {verdict} · {integrity}"


def _outcome_docs(store: Store, pid: str) -> list[dict[str, Any]]:
    from .evolve import outcomes_of

    return outcomes_of(store, pid)


def log_lines(store: Store, name: str) -> list[str]:
    """The version history of one name: header, one block per version, the pre-root pointer.

    A typed key (``tool:<name>``, ``rule:<title>``, ``parameters:knobs``) lists the versions that releases of
    this history reference, newest last, with the releases carrying each.

    Args:
        store (Store): The store.
        name (str): Row name or typed key.

    Returns:
        list[str]: Lines.

    Raises:
        PromptStoreError: If the name has no stored version.
    """
    if _typed_key(name) is not None:
        return _typed_log_lines(store, name)
    try:
        check_name(name)
    except PolicyError as error:
        raise PromptStoreError(str(error)) from error
    versions = _versions(store, name)
    if not versions:
        native = native_rows(store, name)
        if native:
            return [f"native {name} · {len(native)} versions"] + [json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in native] + event_lines(store, name=name)
        raise PromptStoreError(f"{name!r} has no version in the store")
    lineage = L.derive(store)
    numbers = L.version_numbers(store.package, name)
    dates = _version_dates(store.package, name)
    if (scope := store.v0_scope()) is not None:
        root_key = store.release(scope.root).data["refs"].get(name)
        if root_key is not None:
            dates[root_key[:16]] = "v0 seed"
    introducing = _proposals_introducing(store, lineage, name)
    active_key = None
    if lineage.active:
        try:
            active_key = load(store, lineage.active).refs.get(name)
        except PromptStoreError:
            active_key = None
    sample = next(iter(versions.values()))
    meta = sample["manifest"]["meta"]
    kind = meta["kind"]
    shape = _shape(meta)
    baseline_v = next((f"v{numbers.get(version, 0)}" for version, row in versions.items() if row["manifest"]["meta"]["content_key"] == active_key), "-")
    lines = [f"{name}  {kind} · {_entry_label(name)} · {_visibility(name)} · {shape} · baseline {baseline_v}"]
    ordered = sorted(versions.items(), key=lambda item: -numbers.get(item[0], 0))
    for version, row in ordered:
        key = row["manifest"]["meta"]["content_key"]
        pids = introducing.get(key, [])
        pid_text = pids[0] if pids else str(row["manifest"]["meta"].get("proposal") or "—")
        proposal_state = lineage.proposals.get(pids[0]) if pids else None
        state = "" if proposal_state is None else proposal_state.status + (f" by {proposal_state.decided_by}" if proposal_state.status == "promoted" and proposal_state.decided_by else "")
        star = f"  ★ in BASELINE {lineage.active}" if key == active_key else ""
        lines.append(f"  v{numbers.get(version, 0)}  {version}  {dates.get(version, 'uncommitted')}  {_author(row)}  {('proposal ' + pid_text) if pids else pid_text}  {state}{star}".rstrip())
        lines.append(f"      reason   {row['manifest'].get('reason', '')}")
        for pid in pids:
            others = [other for other in (lineage.proposals[pid].body.get("changes") or {}) if other != name] if pid in lineage.proposals else []
            if others:
                lines.append(f"      changed with {len(others)} other name(s) in {pid}: {', '.join(_short(other) for other in others)} — the effect is not attributed to {_short(name)} alone")
            for doc in _outcome_docs(store, pid):
                lines.append(f"      outcome  {outcome_line(doc)}")
    lines.extend(event_lines(store, name=name))
    return lines


# ---------------------------------------------------------------------------------------
# show


def show_lines(store: Store, ref: str, *, spec: bool = False, compiled: bool = False) -> list[str]:
    """Print one row: header, description, reason, text, document or spec, optionally the capsule source.

    A version a v0 release carries reads as ``prompts log`` and the lab show it (v0 root content
    with its original native lineage); every version reads exactly as stored.

    Args:
        store (Store): The store.
        ref (str): A row reference (see :func:`resolve_ref`).
        spec (bool): Print the spec JSON of a spec row (default when ``compiled`` is not asked), or the spec a doc
            row derives from its document after the document.
        compiled (bool): Print the capsule source.

    Returns:
        list[str]: Lines.
    """
    resolved = resolve_ref(ref, store)
    typed = _typed_key(resolved.name)
    if typed is not None:
        carriers = next((found for row, found in typed_versions(store, resolved.name)
                         if row["object_id"] == resolved.row["object_id"]), [])
        return [f"{resolved.name}:{resolved.version}  {typed[0]}  {resolved.row['object_id']}  releases {', '.join(carriers) or '-'}",
                json.dumps(resolved.row, ensure_ascii=False, sort_keys=True, indent=2)] + event_lines(store, name=resolved.name)
    if "capsule" not in resolved.row:
        return [f"native {resolved.name}:{resolved.version}", json.dumps(resolved.row, ensure_ascii=False, sort_keys=True, indent=2)] + event_lines(store, name=resolved.name)
    row = resolved.row
    meta = row["manifest"]["meta"]
    numbers = L.version_numbers(store.package, resolved.name)
    header = f"{resolved.name}:{resolved.version}  v{numbers.get(resolved.version, 0)}  {meta['kind']}  {_entry_label(resolved.name)}  tags {', '.join(row.get('tags') or [])}"
    if resolved.release is not None:
        header += f"  · in {resolved.release.id}"
    lines = [header, f"description  {row.get('description', '')}", f"reason       {row['manifest'].get('reason', '')}", f"author       {_author(row)}"]
    if meta["kind"] == "text":
        lines.append(f"slots        {list(meta['slots'])}")
        lines.append("---")
        lines.append(meta["text"].rstrip("\n") if meta["text"].endswith("\n") else meta["text"])
    elif meta["kind"] == "doc":
        lines.append(f"params       {json.dumps(meta['params'], ensure_ascii=False)}")
        lines.append("--- document")
        lines.extend(meta["text"].split("\n"))
        if spec:
            lines.append("--- derived spec")
            lines.extend(json.dumps(parse_document(meta["text"], meta["params"]).spec, ensure_ascii=False, indent=1).splitlines())
    else:
        lines.append(f"params       {json.dumps(meta['params'], ensure_ascii=False)}")
        lines.append("parts        " + json.dumps(meta["parts"], ensure_ascii=False, indent=1).replace("\n", "\n             "))
        if spec or not compiled:
            lines.append("--- spec")
            lines.extend(json.dumps(meta["spec"], ensure_ascii=False, indent=1).splitlines())
    if compiled:
        lines.append("--- capsule source")
        lines.extend(row["capsule"]["restore_layers"][0]["source"].rstrip("\n").splitlines())
    gate_path = store.package / "attestations"
    if resolved.release is not None and resolved.release.proposal and (gate_path / f"{resolved.release.proposal}-gate.json").is_file():
        gate = json.loads((gate_path / f"{resolved.release.proposal}-gate.json").read_text(encoding="utf-8")).get("gate") or {}
        lines.append("gate         " + ("PASS" if gate.get("passed") else "FAIL") + " · " + " ".join(f"{name}:{status}" for name, status in sorted((gate.get("checks") or {}).items())))
    lines.extend(event_lines(store, name=resolved.name))
    return lines


def _publish_op(text: str, params: Mapping[str, Any]) -> str:
    """The proposal change that publishes a drafted text: ``op doc`` for a document with tags, ``op text`` otherwise."""
    try:
        tagged = parse_document(text, params).tagged
    except DocError:
        tagged = False   # a text row's construct no document writes, such as a !r slot
    return "op doc" if tagged else f"op text, slots {list(params)}"


def draft_text(release: Any, name: str, *, spec: bool = False) -> tuple[str, str]:
    """The editable form of a row, or of an entry the release lacks, with a one-line note on what it is.

    A text or doc row gives its ``meta.text``. A spec row gives its normal form printed as a document
    (:func:`doc.document_of`) or its spec JSON with ``spec``. A known entry the release does not carry, such
    as a role document of the one-document layout under root, is assembled from the identity texts of the
    surfaces it would provide, joined by ``{{message}}`` lines: exactly the document that re-cuts those texts
    into one row, so the documents of a refactor are generated, never retyped. A candidate slot's document the
    release lacks is drafted from the shared candidate document, which that slot renders today, a missing
    ``config.knobs`` from the knobs' defaults and a missing ``rules.general`` from the checkout's committed
    corpus (``resources/generated_rules/general_rules.json``), which is what such a release's pipeline reads. A
    ``rubiksql.closure/2`` release drafts ``config.knobs`` from its parameter set's explicit values and
    ``rules.general`` from the corpus its rule versions reconstruct, never from code defaults. The
    note of a printed or assembled text names the change that publishes it: ``op doc`` when it has tags, ``op
    text`` when it has none (a doc row needs a tag).

    Args:
        release (Release): The release to draft from.
        name (str): Row or entry name.
        spec (bool): Give a spec row's JSON instead of its document.

    Returns:
        tuple[str, str]: The text and a note naming its kind and declarations.

    Raises:
        PromptStoreError: If the name is neither a row of the release nor an entry whose surfaces it provides.
    """
    if name in release.refs:
        meta = release.row(name)["manifest"]["meta"]
        if meta["kind"] == "text":
            return meta["text"], f"text; slots {list(meta['slots'])}"
        if meta["kind"] == "doc":
            return meta["text"], f"doc; params {json.dumps(meta['params'], ensure_ascii=False)}"
        if spec:
            return json.dumps(meta["spec"], ensure_ascii=False, indent=1) + "\n", f"spec; params {list(meta['params'])} parts {list(meta['parts'])}"
        composer = release.renderable(name)
        try:
            text = document_of(composer.normal_form(), composer.params)
        except DocError as error:
            raise PromptStoreError(f"{name}: {error}; draft its spec JSON with --spec") from error
        return text, f"spec printed as a document; params {json.dumps(dict(composer.params), ensure_ascii=False)}; publish it with {_publish_op(text, composer.params)}"
    known = E.ENTRIES.get(name)
    if known is None or not known.surfaces:
        raise PromptStoreError(f"{name!r} is not a row of {release.id}")
    typed = release.rules == CLOSURE_V2
    if name == E.KNOBS_ENTRY and typed:
        return E.knobs_text(release.parameter_values()), f"text; the explicit values of the parameter set {release.id} pins; change them with op knobs"
    if name == E.KNOBS_ENTRY:
        return E.knobs_text({knob.name: knob.default for knob in E.KNOBS.values()}), f"text; the defaults of every knob, which {release.id} runs with; publish it with op text, slots []"
    if name == E.RULES_ENTRY and typed:
        from .rules import corpus_text

        return corpus_text(release.rules_general()), f"text; the corpus reconstructed from the rule versions {release.id} pins; change it with op rules"
    if name == E.RULES_ENTRY:
        raise PromptStoreError(f"{release.id} carries no {name!r}; every authority release must carry its own corpus")
    identity = release.surface_identity()
    if known.slot is not None:
        text, note = draft_text(release, "candidate")
        return text, f"slot {known.slot} renders the shared candidate document under {release.id}; publishing this text as {name} gives the slot its own ({note})"
    missing = [surface for surface in known.surfaces if surface not in identity]
    if missing:
        raise PromptStoreError(f"{release.id} provides no {missing}; nothing to assemble {name!r} from")
    undocumented = [surface for surface in known.surfaces if identity[surface].startswith(MESSAGE_SEPARATOR)]
    if undocumented:
        raise PromptStoreError(f"{release.id}'s {undocumented} use a construct no document can write; rewrite them before re-cutting {name!r}")
    text = "\n{{message}}\n".join(identity[surface] for surface in known.surfaces)
    return text, f"assembled from the {', '.join(known.surfaces)} surfaces of {release.id}; params {json.dumps(known.params, ensure_ascii=False)}; publish it with {_publish_op(text, known.params)}"


def _anchor_of(store: Store) -> str:
    root = store.v0_root()
    if root is None:
        raise PromptStoreError("history has no root")
    return root


# ---------------------------------------------------------------------------------------
# diff


def word_diff(a: str, b: str, *, context: int = 12) -> str:
    """Word-level diff with ``{+inserted+}`` and ``[-deleted-]`` markers and elided unchanged runs.

    Args:
        a (str): Old text.
        b (str): New text.
        context (int): Tokens (words and whitespace runs) kept on each side of an unchanged run.

    Returns:
        str: The marked text; unchanged runs longer than ``2 * context`` collapse to ``… … …``.
    """
    ta, tb = _TOKEN_RE.findall(a), _TOKEN_RE.findall(b)
    out: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if tag == "equal":
            segment = ta[i1:i2]
            if len(segment) > 2 * context + 1:
                out.append("".join(segment[:context]) + " … " + "".join(segment[-context:]))
            else:
                out.append("".join(segment))
        else:
            if tag in ("delete", "replace"):
                out.append("[-" + "".join(ta[i1:i2]) + "-]")
            if tag in ("insert", "replace"):
                out.append("{+" + "".join(tb[j1:j2]) + "+}")
    return "".join(out)


def spec_diff(a: Any, b: Any, path: str = "$") -> list[tuple[str, Any, Any]]:
    """Structural differences between two spec trees.

    Args:
        a (Any): Old spec (JSON value).
        b (Any): New spec.
        path (str): Path of the current node.

    Returns:
        list[tuple[str, Any, Any]]: ``(path, old, new)`` for every differing node; a missing side is ``None``.
    """
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a) != set(b):
            return [(path, a, b)]
        out: list[tuple[str, Any, Any]] = []
        for key in sorted(a):
            out.extend(spec_diff(a[key], b[key], f"{path}.{key}"))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [(path, a, b)]
        out = []
        for index, (left, right) in enumerate(zip(a, b)):
            out.extend(spec_diff(left, right, f"{path}[{index}]"))
        return out
    return [] if a == b else [(path, a, b)]


def _unified(a: str, b: str, label_a: str, label_b: str, limit: int = 200) -> list[str]:
    lines = list(difflib.unified_diff(a.splitlines(True), b.splitlines(True), label_a, label_b, n=2))
    if len(lines) > limit:
        lines = lines[:limit] + [f"... {len(lines) - limit} more lines\n"]
    return [line.rstrip("\n") for line in lines]


def _reaching_entries(release: Any, name: str) -> list[str]:
    from .spec import Composer

    if name in release.entries:
        return [name]
    reaching = []
    for entry in release.entries:
        stack, seen = [release.renderable(entry)], set()
        while stack:
            item = stack.pop()
            if item.name in seen:
                continue
            seen.add(item.name)
            if item.name == name:
                reaching.append(entry)
                break
            if isinstance(item, Composer):
                stack.extend(item.parts.values())
    return reaching


def _cells_for(store: Store, release: Any, entries: Iterable[str]) -> list[Cell]:
    wanted = set(entries)
    return [cell for cell in all_cells(store) if cell.entry in wanted and release.has(cell.entry)]


def _render(release: Any, cell: Cell) -> str:
    return release.text(cell.entry, **cell.params)


def _contract_lines(release: Any, entry: str, cells: Sequence[Cell]) -> str:
    results = []
    known = E.ENTRIES.get(entry)
    surfaces = known.surfaces if known is not None else ()
    for surface in surfaces:
        declared = E.SURFACES[surface]
        if declared.params or not declared.contracts:
            continue
        text = release.surface(*surface.split(".", 1))
        for contract in declared.contracts:
            violation = contract.check(text)
            results.append(f"{contract.name} {'PASS' if violation is None else ('WARN' if contract.severity == 'warning' else 'FAIL')}")
    for contract in RENDER_CONTRACTS:
        if contract.surface not in surfaces:
            continue
        index = surfaces.index(contract.surface)
        failed = [cell.name for cell in cells if cell.entry == entry and contract.check(cell.params, release.messages(entry, **cell.params)[index]) is not None]
        results.append(f"{contract.name} {'FAIL @ ' + ','.join(failed) if failed else 'PASS'}")
    return " · ".join(results) if results else "none declared"


def _coverage_line(store: Store, release: Any, cells: Sequence[Cell]) -> str:
    """Structural coverage of the spec rows the given cells' entries reach, under those cells alone."""
    from .matrix import render_cells
    from .spec import Coverage

    entries = {cell.entry for cell in cells}
    reached: set[str] = set()
    for entry in entries:
        for composer in release.composers():
            if entry in _reaching_entries(release, composer.name):
                reached.add(composer.name)
    composers = [composer for composer in release.composers() if composer.name in reached]
    if not composers:
        return "no structural obligations (text rows only)"
    coverage = Coverage()
    renders = render_cells(release, cells, coverage)
    missing = coverage.missing(composers)
    obligations = sum(len(composer.spec.branch_points()) for composer in composers)
    if not missing:
        return f"complete ({obligations} structural obligations under {len(renders)} cell renders)"
    return f"{len(missing)} gap(s) of {obligations}: " + " | ".join(missing[:3])


def _keyspace_line(release_a: Any, release_b: Any) -> str:
    moves = keyspace_moves(release_b, release_a)
    labels = [E.route_label(*pair) for pair in E.KEYSPACES]
    moved = [label for label in labels if label in moves]
    same = [label for label in labels if label not in moves]
    vs_root = release_b.moved_keyspaces()
    text = (", ".join(moved) + " move" if moved else "no keyspace moves") + (f" · unchanged: {', '.join(same)}" if moved and same else "")
    return text + (f" · vs root: {', '.join(vs_root)} moved" if vs_root else " · vs root: all equal (prompt field absent)")


def diff_lines(
    store: Store,
    ref_a: str,
    ref_b: str,
    *,
    words: bool = False,
    cell: str | None = None,
    real: int = 0,
    corpus: Any | None = None,
    root_run: RootRun | None = None,
) -> list[str]:
    """Diff two versions of a row: change ratio, text or spec, matrix cells, coverage, contracts, keyspaces, corpus and cost.

    Args:
        store (Store): The store.
        ref_a (str): Old reference.
        ref_b (str): New reference.
        words (bool): Word-level diff of text rows instead of a unified diff.
        cell (str | None): Also print the unified diff of one rendered matrix cell.
        real (int): Also print up to this many real-corpus sample diffs (needs ``corpus``).
        corpus (Corpus | None): A locally built corpus.
        root_run (RootRun | None): A root run for the exposure and cost lines.

    Returns:
        list[str]: Lines.

    Raises:
        PromptStoreError: If a reference does not resolve or the two rows have different names.
    """
    a, b = resolve_ref(ref_a, store), resolve_ref(ref_b, store)
    if a.name != b.name:
        raise PromptStoreError(f"diff compares two versions of one name; got {a.name} and {b.name}")
    if "capsule" not in a.row or "capsule" not in b.row:
        before = json.dumps(a.row, ensure_ascii=False, sort_keys=True, indent=2)
        after = json.dumps(b.row, ensure_ascii=False, sort_keys=True, indent=2)
        change = change_stat(before, after)
        label = "typed" if _typed_key(a.name) is not None else "native"
        lines = [f"{label} {a.name} {a.version} -> {b.version} · +{change.lines_added}/-{change.lines_removed} · change {change.ratio:.1%}"]
        return lines + ([word_diff(before, after)] if words else _unified(before, after, ref_a, ref_b))
    numbers = L.version_numbers(store.package, a.name)
    row_a, row_b = a.row, b.row
    meta_a, meta_b = row_a["manifest"]["meta"], row_b["manifest"]["meta"]
    pid_b = f" · {b.release.proposal}" if b.release is not None and b.release.proposal else ""
    lines = [
        f"--- {a.name}:{a.version}  v{numbers.get(a.version, 0)}  {meta_a['kind']}  {_author(row_a)}",
        f"+++ {b.name}:{b.version}  v{numbers.get(b.version, 0)}  {meta_b['kind']}  {_author(row_b)}{pid_b}",
    ]
    if a.version == b.version:
        lines.append("identical versions")
        return lines
    if row_a["manifest"].get("reason") != row_b["manifest"].get("reason"):
        lines.append(f"reason    {row_b['manifest'].get('reason', '')}")
    stat = change_stat(row_text(meta_a), row_text(meta_b))
    lines.append(f"change    {stat.summary()} of the characters" + (f" · WARN more than {CHANGE_RATIO_WARN:.0%}: one proposal should test one idea" if stat.ratio > CHANGE_RATIO_WARN else ""))
    if meta_a["kind"] in ("text", "doc") and meta_b["kind"] in ("text", "doc"):
        if meta_a["kind"] != meta_b["kind"]:
            lines.append(f"kind      {meta_a['kind']} → {meta_b['kind']}")
        declared_a = list(meta_a["slots"] if meta_a["kind"] == "text" else meta_a["params"])
        declared_b = list(meta_b["slots"] if meta_b["kind"] == "text" else meta_b["params"])
        label = "slots" if meta_a["kind"] == meta_b["kind"] == "text" else "params"
        declared = f"{', '.join(declared_a) or '-'} (unchanged)" if declared_a == declared_b else f"{', '.join(declared_a) or '-'} → {', '.join(declared_b) or '-'}"
        size_a, size_b = len(meta_a["text"].encode("utf-8")), len(meta_b["text"].encode("utf-8"))
        lines.append(f"{label:<9} {declared}    bytes {size_a} → {size_b} ({size_b - size_a:+d})")
        if words:
            lines.extend("  " + line for line in word_diff(meta_a["text"], meta_b["text"]).splitlines())
        else:
            lines.extend(_unified(meta_a["text"], meta_b["text"], f"{a.name}:{a.version}", f"{b.name}:{b.version}"))
    elif meta_a["kind"] == "spec" and meta_b["kind"] == "spec":
        parts_line = []
        for slot in sorted(set(meta_a["parts"]) | set(meta_b["parts"])):
            before, after = meta_a["parts"].get(slot), meta_b["parts"].get(slot)
            parts_line.append(slot if before == after else (f"+{slot}" if before is None else (f"-{slot}" if after is None else f"{slot}({_short(before)}→{_short(after)})")))
        lines.append("parts   = " + " ".join(parts_line))
        if meta_a["params"] != meta_b["params"]:
            lines.append(f"params  {json.dumps(meta_a['params'], ensure_ascii=False)} → {json.dumps(meta_b['params'], ensure_ascii=False)}")
        for path, old, new in spec_diff(meta_a["spec"], meta_b["spec"]):
            lines.append(f"@@ spec {path} @@")
            lines.append("- " + (canon(old)[:240] if old is not None else "(absent)"))
            lines.append("+ " + (canon(new)[:240] if new is not None else "(absent)"))
    else:
        lines.append(f"kind      {meta_a['kind']} → {meta_b['kind']}")
    if a.release is None or b.release is None:
        lines.append("matrix    (no release carries " + ("both versions" if a.release is None and b.release is None else "one version") + "; use name@<release> for cell, coverage and keyspace lines)")
        return lines
    entries_b = _reaching_entries(b.release, b.name)
    entries_a = _reaching_entries(a.release, a.name)
    cells_b = _cells_for(store, b.release, entries_b)
    cells_a = {(item.entry, item.name): item for item in _cells_for(store, a.release, entries_a)}
    marks = []
    changed_cells = 0
    for item in cells_b:
        text_b = _render(b.release, item)
        other = cells_a.get((item.entry, item.name))
        if other is None:
            marks.append(f"  + {item.entry}/{item.name}  {sha256_hex(text_b)[:12]}")
            continue
        text_a = _render(a.release, other)
        if text_a == text_b:
            marks.append(f"  = {item.entry}/{item.name}  {sha256_hex(text_a)[:12]}")
        else:
            changed_cells += 1
            marks.append(f"  ~ {item.entry}/{item.name}  {sha256_hex(text_a)[:12]} → {sha256_hex(text_b)[:12]}")
    code = sum(1 for item in cells_b if item.origin == "code")
    lines.append(f"@@ render matrix ({len(cells_b)} cells: {code} code + {len(cells_b) - code} data; {changed_cells} change) @@")
    lines.extend(marks)
    lines.append(f"coverage   {_coverage_line(store, b.release, cells_b)}")
    lines.append("contracts  " + " · ".join(f"{_short(entry)}: {_contract_lines(b.release, entry, cells_b)}" for entry in entries_b))
    if cell:
        matches = [item for item in cells_b if item.name == cell]
        if not matches:
            lines.append(f"cell {cell!r}: not a cell of {', '.join(entries_b)}")
        for item in matches:
            other = cells_a.get((item.entry, item.name))
            lines.append(f"@@ cell {item.entry}/{item.name} @@")
            lines.extend(_unified(_render(a.release, other) if other else "", _render(b.release, item), a.release.id, b.release.id))
    if corpus is not None:
        from .corpus import stats

        result = stats(corpus, a.release, b.release, samples=max(real, 0))
        # the corpus counts samples by their leading surface, whichever layout's row renders it
        wanted = E.surfaces_of(set(entries_a) | set(entries_b))
        relevant = {surface: counts for surface, counts in result.per_entry.items() if surface in wanted}
        total_n = sum(n for n, _k in relevant.values())
        total_k = sum(k for _n, k in relevant.values())
        lines.append(f"corpus     {total_k} / {total_n} real inputs of {', '.join(_short(entry) for entry in sorted(relevant)) or _short(b.name)} change" + (f" · --real {real} samples below" if real else " · --real K for samples"))
        for sample in result.samples[:real]:
            lines.append(f"@@ real {sample.sample.qid} {sample.sample.role}/{sample.sample.message} @@")
            lines.extend(line.rstrip("\n") for line in sample.diff.splitlines())
    lines.append(f"keyspace   {_keyspace_line(a.release, b.release)}")
    if root_run is not None:
        estimate = estimate_cost(b.release, a.release, root_run)
        fired = sum(1 for unit in root_run.units if E.fired_surfaces(unit.fired) & E.surfaces_of(entries_b))
        by_role = {}
        for unit in root_run.units:
            by_role[unit.role] = by_role.get(unit.role, 0) + 1
        lines.append(f"exposure   root {root_run.name}: {_short(b.name)} fired in {fired} of {len(root_run.units)} units ({', '.join(f'{count} {role}' for role, count in sorted(by_role.items()))}) → exposed units diverge at their first changed render; the rest replay turn by turn")
        lines.append(f"cost       {estimate.summary()}")
    return lines


# ---------------------------------------------------------------------------------------
# release-diff


def _fingerprints(release: Any) -> dict[str, str]:
    """The 8-hex fingerprints release-diff compares, for the surfaces the release provides.

    ``_SYSTEM`` (the candidate system message at the default 64-row cap), ``_JUDGE_COMMON`` and ``grain_prompt`` are
    the fields of the ``[prompt-fingerprint]`` banner; ``_JUDGE_COMMON`` follows ``vote._JUDGE_COMMON``: the judge body
    (:meth:`runtime.Release.fragment`), or the whole judge system message once a release changed that text and no
    separable body exists. ``_JUDGE_SYSTEM`` and ``_REVIEWER_SYSTEM`` are the whole judge and reviewer system messages.
    """
    fields: dict[str, str] = {}
    if release.has_surface("candidate", "system"):
        fields["_SYSTEM"] = _fp8(release.surface("candidate", "system", max_rows=64))
    if release.has_surface("judge", "system"):
        fields["_JUDGE_COMMON"] = _fp8(release.surface("judge", "system"))
        fields["_JUDGE_SYSTEM"] = _fp8(release.surface("judge", "system"))
    if release.has_surface("candidate", "route.grain"):
        fields["grain_prompt"] = _fp8(release.surface("candidate", "route.grain"))
    if release.has_surface("reviewer", "system"):
        fields["_REVIEWER_SYSTEM"] = _fp8(release.surface("reviewer", "system"))
    return fields


def _group_unchanged(names: Iterable[str], changed: Iterable[str]) -> str:
    """``owner.*`` for an owner (role or ``shared``) with no changed surface, ``owner.* except a, b`` otherwise."""
    changed_set = set(changed)
    by_owner: dict[str, list[str]] = {}
    for name in names:
        by_owner.setdefault(name.split(".", 1)[0], []).append(name)
    out = []
    for owner, members in sorted(by_owner.items()):
        moved = sorted(name for name in members if name in changed_set)
        if not moved:
            out.append(f"{owner}.*")
        elif len(moved) < len(members):
            out.append(f"{owner}.* except " + ", ".join(name.split(".", 1)[1] for name in moved))
    return ", ".join(out) or "none"


def _render_for(release: Any, name: str, params: Mapping[str, Any]) -> str:
    """Render an entry, or a surface id through whichever row provides it, with a cell's arguments."""
    if release.has(name):
        return release.text(name, **params)
    surface = E.SURFACES.get(name)
    if surface is None:
        raise PromptStoreError(f"{name!r} is neither an entry of {release.id} nor a surface id")
    return release.surface(*name.split(".", 1), **{key: params[key] for key in surface.params})


def release_diff_lines(
    store: Store,
    ref_a: str,
    ref_b: str,
    *,
    render: bool = False,
    entry: str | None = None,
    cell: str | None = None,
    real: int = 0,
    corpus: Any | None = None,
) -> list[str]:
    """Compare two releases: changed names, change ratio, moved surfaces, keyspaces, fingerprints, optionally a rendered diff.

    Args:
        store (Store): The store.
        ref_a (str): ``baseline``, release or proposal id.
        ref_b (str): The other one.
        render (bool): Print the unified diff of one rendered entry or surface (``entry`` required).
        entry (str | None): Entry or surface id to render; a surface id renders under both releases whichever rows
            provide it, so two layouts compare.
        cell (str | None): Matrix cell whose arguments render a parameterised entry.
        real (int): Print up to this many real-corpus sample diffs (needs ``corpus``).
        corpus (Corpus | None): A locally built corpus.

    Returns:
        list[str]: Lines.
    """
    a, b = load(store, ref_a), load(store, ref_b)
    # Typed releases also list their typed keys (tool:, rule:, parameters:, rules:corpus, tools:loader); for two
    # closure/1 releases the closure view is exactly the prompt refs.
    document_a, document_b, index = store.release(a.id).data, store.release(b.id).data, store.typed_index()
    try:
        names = sorted(set(closure_view(document_a, index)) | set(closure_view(document_b, index)))
        changes = closure_changes(document_a, document_b, index)
    except EntityModelError as error:
        raise PromptStoreError(f"release-diff cannot resolve a typed reference: {error}") from error
    lines = [f"names changed {len(changes)}/{len(names)}" + ("" if changes else "   (identical closures)")]
    for name, change in changes.items():
        before, after = change["from"], change["to"]
        lines.append(f"  {name}  {before[:16] if before else '(absent)'} → {after[:16] if after else '(absent)'}")
    ratio = check_change_ratio(b, a)
    lines.append("change ratio          " + ("WARN " if ratio.status == "warn" else "") + ratio.detail)
    moved_surfaces = sorted(changed_surfaces(b, a))
    lines.append(f"surfaces moved        {', '.join(moved_surfaces) or 'none'}      unchanged: {_group_unchanged(set(a.surface_identity()) | set(b.surface_identity()), moved_surfaces)}")
    moves = keyspace_moves(b, a)
    labels = [E.route_label(*pair) for pair in E.KEYSPACES]
    anchor = store.v0_root() or _anchor_of(store)
    lines.append(f"keyspace              {', '.join(label for label in labels if label in moves) or 'none'} → prompt field present · others = {a.id if a.id != anchor else 'v0'}")
    fa, fb = _fingerprints(a), _fingerprints(b)
    lines.append("fingerprints          " + " · ".join(f"{key} {fa.get(key, '-')} → {fb[key]}" if fa.get(key) != fb.get(key) else f"{key} unchanged" for key in ("_SYSTEM", "_JUDGE_COMMON", "_JUDGE_SYSTEM", "_REVIEWER_SYSTEM", "grain_prompt") if key in fb))
    if render:
        if not entry:
            raise PromptStoreError("--render needs --entry")
        params: Mapping[str, Any] = {}
        # a surface id takes the cells of the row that provides it in the newer release
        provided = b.provider(entry) if entry in E.SURFACES else None
        cell_entry = provided[0] if provided is not None else entry
        entry_cells = all_cells(store, entry=cell_entry)
        if cell:
            match = [item for item in entry_cells if item.name == cell]
            if not match:
                raise PromptStoreError(f"entry {cell_entry!r} has no cell {cell!r}")
            params = match[0].params
        elif E.ENTRIES.get(cell_entry) is not None and E.ENTRIES[cell_entry].params and entry_cells:
            cell = entry_cells[0].name  # a parameterised entry renders with its first matrix cell
            params = entry_cells[0].params
        present = [release for release in (a, b) if release.has(entry) or (entry in E.SURFACES and release.has_surface(*entry.split(".", 1)))]
        if len(present) < 2:
            lines.append(f"@@ {entry} (rendered) @@ absent from {'both' if not present else (b.id if present[0] is a else a.id)}")
        else:
            lines.append(f"@@ {entry} (rendered{', cell ' + cell if cell else ''}) @@")
            lines.extend(_unified(_render_for(a, entry, params), _render_for(b, entry, params), a.id, b.id))
    if corpus is not None:
        from .corpus import stats

        result = stats(corpus, a, b, samples=max(real, 0))
        lines.append(f"corpus                {result.summary()}")
        for sample in result.samples[:real]:
            lines.append(f"@@ real {sample.sample.qid} {sample.sample.role}/{sample.sample.message} @@")
            lines.extend(line.rstrip("\n") for line in sample.diff.splitlines())
    return lines


# ---------------------------------------------------------------------------------------
# render --trace (blame)


def blame_lines(release: Any, entry: str, params: Mapping[str, Any], *, numbers: Mapping[str, int] | None = None, width: int = 72) -> list[str]:
    """Attribute every output line of a render to the row that produced most of it, without sentinels.

    Args:
        release (Release): The release.
        entry (str): Entry name.
        params (Mapping[str, Any]): Render arguments.
        numbers (Mapping[str, int] | None): ``name:version`` to display ``vN``; ``v?`` when absent.
        width (int): Column at which the attribution starts.

    Returns:
        list[str]: ``<output line> │ <row> vN <version> <origin>`` per line, bytes identical to the render; a
            document's message boundary prints as a ``{{message}}`` line. The boundaries come from the rendered
            messages, not from U+001E in the whole render, so an argument that contains U+001E stays text.
    """
    spans = release.trace(entry, **params)   # over the whole render: the messages joined by one U+001E each
    lines = []
    position = 0
    for index, message in enumerate(release.messages(entry, **params)):
        if index:
            lines.append("{{message}}".ljust(width) + "│ message boundary")
            position += len(MESSAGE_SEPARATOR)
        for raw in re.split("(?<=\n)", message):
            if raw:
                lines.extend(_blame_line(raw, position, spans, release, numbers, width))
                position += len(raw)
    return lines


def _blame_line(raw: str, start: int, spans: Sequence[Any], release: Any, numbers: Mapping[str, int] | None, width: int) -> list[str]:
    """One blame line: the output line, then the row that produced most of its characters."""
    end = start + len(raw)
    best, best_overlap = None, -1
    for span in spans:
        overlap = min(end, span.end) - max(start, span.start)
        if overlap > best_overlap:
            best, best_overlap = span, overlap
    shown = raw.rstrip("\n")
    if len(shown) > width - 2:
        shown = shown[: width - 3] + "…"
    origin = release.row(best.name)["manifest"]["meta"].get("origin", "?") if best is not None else "?"
    version_number = f"v{numbers.get(f'{best.name}:{best.version}', '?')}" if best is not None and numbers is not None else "v?"
    attribution = f"{_short(best.name)}  {version_number} {best.version[:4]}…  {origin}" if best is not None else "?"
    return [f"{shown.ljust(width)}│ {attribution}"]


# ---------------------------------------------------------------------------------------
# status and tree


def status_lines(store: Store, *, repo: Path | None = None, control_path: Path | None = None, control_drafts: int | None = None, view_state: str = "unknown") -> list[str]:
    """The three status lines: checkout and store, generation (attempts against the cap), control and view.

    Args:
        store (Store): The store.
        repo (Path | None): Repository root for the checkout line.
        control_path (Path | None): Control database path.
        control_drafts (int | None): Proposals in the control workspace that are not pinned; ``None`` when unknown.
        view_state (str): ``registered``, ``not registered`` or ``unknown``.

    Returns:
        list[str]: Three lines.
    """
    store.require_seeded_v0()
    package = store.package
    scope = store.v0_scope()
    first = [checkout_line(repo) if repo is not None else "checkout -"]
    try:
        pointer = store.baseline()
        anchor_id = scope.root if scope is not None else _anchor_of(store)
        anchor_name = "v0"
        anchor = f"= {anchor_name}" if pointer.release == anchor_id else f"({anchor_name} {anchor_id or 'unset'})"
        first.append(f"BASELINE {pointer.release} {anchor}")
        active: str | None = pointer.release
    except PromptStoreError as error:
        first.append(f"BASELINE absent ({str(error).splitlines()[0][:60]})")
        active = None

    def count(directory: str, pattern: str) -> int:
        path = package / directory
        return len([item for item in path.glob(pattern) if item.suffix == ".json"]) if path.is_dir() else 0

    if scope is None:
        counts = (count('objects', '*/*.json'), count('releases', 'r*.json'), count('proposals', 'p*.json'),
                  count('outcomes', '*/o*.json'), count('decisions', 'd*.json'))
    else:
        from . import evolve as EV

        objects = {(name, key[:16]) for rid in scope.releases
                   for name, key in store.release(rid).data["refs"].items()}
        decisions = [decision for decision in store.decisions()
                     if decision.get("target") in scope.releases | scope.proposals]
        counts = (len(objects), len(scope.releases), len(scope.proposals),
                  sum(len(EV.outcomes_of(store, pid)) for pid in scope.proposals), len(decisions))
    first.append(f"store {counts[0]} objects / {counts[1]} releases / {counts[2]} proposals / {counts[3]} outcomes / {counts[4]} decisions")
    lines = [" · ".join(first)]
    if active:
        lines.append(f"generation {active[:7]}… {_attempts_text(store, active)}")
    else:
        lines.append("generation - (no BASELINE)")
    control_text = f"control {control_path} ({'exists' if control_path is not None and control_path.exists() else 'absent'}" + (f", drafts {control_drafts}" if control_drafts is not None else "") + ")" if control_path is not None else "control -"
    from .view_ws import VIEW_WS_ID

    lines.append(f"{control_text} · view {VIEW_WS_ID} {view_state}")
    head = getattr(store, "history_head", None)
    if head is not None:
        lines.append(f"authority revision {head.revision.event_count}:{head.revision.latest_event_id} · export {'fresh' if head.exported else 'stale'} · commit {head.export_commit or '-'}")
    snapshot = getattr(store, "history_snapshot", None)
    if snapshot is not None:
        exports = [event for event in history_events(store) if event.get("kind") == "export"]
        receipt = exports[-1].get("payload", {}) if exports else {}
        lines.append(f"snapshot {snapshot.digest} · authority revision {snapshot.revision.event_count}:{snapshot.revision.latest_event_id} · export receipt {receipt.get('export_commit') or receipt.get('commit') or 'not recorded'}")
    return lines


def _attempts_text(store: Store, generation: str) -> str:
    """``attempts 3/10``: the evaluations the generation counted against the cap (:data:`evolve.MAX_ATTEMPTS_PER_GENERATION`)."""
    from . import evolve as EV

    return f"attempts {len(EV.generation_attempts(store, generation))}/{EV.MAX_ATTEMPTS_PER_GENERATION}"


def _claim(body: Mapping[str, Any]) -> str:
    hypothesis = body.get("hypothesis")
    claim = str(hypothesis.get("claim") or "") if isinstance(hypothesis, Mapping) else ""
    return f'"{claim[:72]}"' if claim else str(body.get("reason") or "")[:72]


def _tree_label(store: Store, lineage: L.Lineage) -> Callable[[L.TreeNode], str]:
    def label(node: L.TreeNode) -> str:
        if node.kind == "release":
            release_doc = store.release(node.release).data if _has_release(store, node.release) else {}
            bits = [str(release_doc.get("origin") or "imported")]
            if release_doc.get("reason"):
                bits.append(str(release_doc["reason"])[:60])
            return "  ".join(bits)
        body = node.body
        changed = ", ".join(_short(name) for name in sorted(body.get("changes") or {})) or "no content change"
        bits = [str(body.get("author") or "?"), changed, _claim(body)]
        docs = _outcome_docs(store, node.id)
        if docs:
            latest = docs[-1]
            stats = latest.get("stats") or {}
            if "rule" not in latest:
                bits.append(f"recorded before rule A: {latest.get('verdict')} · run {latest.get('run')}")
            elif stats.get("gain_pp") is not None:
                exposed = stats.get("exposed") or {}
                gain, p = _gain_p(latest)
                bits.append(f"EX {gain}pp (c={stats.get('c')}, b={stats.get('b')}; exposed {exposed.get('questions', 0)}/{stats.get('n')}) p {p} → {_verdict_text(latest)} · integrity {latest.get('integrity')} · control {latest.get('parent_run') or '-'} (age {float((latest.get('pairing') or {}).get('age_days') or 0):.0f}d)")
            else:
                bits.append(f"descriptive only · run {latest.get('run')}")
        elif node.status == "pinned":
            bits.append("eval pending")
        return "  ".join(bits)

    return label


def _has_release(store: Store, rid: str) -> bool:
    try:
        store.release(rid)
    except PromptStoreError:
        return False
    return True


def tree_lines(store: Store, *, repo: Path | None = None, drafts: Iterable[Mapping[str, Any]] = ()) -> list[str]:
    """The proposal tree with verdicts, EX gains, p-values, integrity and control age per node, and who promoted.

    Args:
        store (Store): The store.
        repo (Path | None): Repository root for the header's checkout.
        drafts (Iterable[Mapping[str, Any]]): Control-only proposal bodies to show as drafts.

    Returns:
        list[str]: Header line plus one line per node.
    """
    lineage = L.derive(store, drafts=drafts)
    header = [checkout_line(repo) if repo is not None else "checkout -"]
    if lineage.active:
        header.append(f"BASELINE {lineage.active}")
        header.append(f"generation {_attempts_text(store, lineage.active)}")
    else:
        header.append("BASELINE absent")
    roots = L.proposal_tree(lineage)
    body = L.render_tree(roots, label=_tree_label(store, lineage))
    return [" · ".join(header)] + body.splitlines() + event_lines(store)


# ---------------------------------------------------------------------------------------
# verify


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _word_diff_html(a: str, b: str) -> str:
    ta, tb = _TOKEN_RE.findall(a), _TOKEN_RE.findall(b)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if tag == "equal":
            out.append(_esc("".join(ta[i1:i2])))
        else:
            if tag in ("delete", "replace"):
                out.append(f"<span class='del'>{_esc(''.join(ta[i1:i2]))}</span>")
            if tag in ("insert", "replace"):
                out.append(f"<span class='ins'>{_esc(''.join(tb[j1:j2]))}</span>")
    return "".join(out)


def report_html(store: Store, *, repo: Path | None = None, title: str = "RubikSQL prompt report") -> str:
    """A self-contained HTML report: tree, entry timelines, side-by-side diffs, matrix heatmap, metrics.

    Args:
        store (Store): The store.
        repo (Path | None): Repository root for the header.
        title (str): Page title.

    Returns:
        str: The HTML document (inline CSS, no scripts, no external resources).
    """
    from .evolve import VERDICTS, outcomes_of

    lineage = L.derive(store)
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{_esc(title)}</title><style>{_CSS}</style></head><body>"]
    parts.append(f"<h1>{_esc(title)}</h1><p class='status'>{_esc(' · '.join(tree_lines(store, repo=repo)[:1]))}</p>")
    parts.append("<h2>Proposal tree</h2><pre>" + _esc("\n".join(tree_lines(store, repo=repo)[1:])) + "</pre>")

    baseline = load(store, lineage.active) if lineage.active else None
    proposals = sorted(lineage.proposals.values(), key=lambda state: float(state.body.get("created_at") or 0.0))
    releases: dict[str, Any] = {}
    for state in proposals:
        try:
            releases[state.id] = load(store, state.release)
        except PromptStoreError:
            continue

    parts.append("<h2>Entry timelines</h2><table><tr><th>entry</th><th>versions (oldest first)</th></tr>")
    for name in sorted(E.ENTRIES):
        versions = _versions(store, name)
        if not versions:
            continue
        numbers = L.version_numbers(store.package, name)
        cells = []
        for version, row in sorted(versions.items(), key=lambda item: numbers.get(item[0], 0)):
            key = row["manifest"]["meta"]["content_key"]
            mark = " ★" if baseline is not None and baseline.refs.get(name) == key else ""
            cells.append(f"v{numbers.get(version, 0)} {version[:8]}{mark} <small>{_esc(_author(row))}</small>")
        parts.append(f"<tr><td>{_esc(_short(name))}</td><td>{' → '.join(cells)}</td></tr>")
    parts.append("</table>")

    parts.append("<h2>Changed rows, side by side</h2>")
    index = store.typed_index()
    for state in proposals:
        release = releases.get(state.id)
        if release is None:
            continue
        parts.append(f"<h3>{_esc(state.id)} <span class='status'>{_esc(state.status)}</span> — {_esc(state.body.get('reason') or '')}</h3>")
        base_id = next((parent.get("release") for parent in state.parents if isinstance(parent, Mapping) and parent.get("role", "base") == "base"), None)
        try:
            parent = load(store, base_id) if isinstance(base_id, str) else None
        except PromptStoreError:
            parent = None
        for name, change in sorted((state.body.get("changes") or {}).items()):
            before = parent.row(name)["manifest"]["meta"] if parent is not None and name in parent.refs else None
            after = release.row(name)["manifest"]["meta"] if name in release.refs else None
            parts.append(f"<h4>{_esc(name)} <small>{_esc((change or {}).get('from') or '∅')[:16]} → {_esc((change or {}).get('to') or '∅')[:16]}</small></h4>")
            typed = [_typed_by_digest(index, name, (change or {}).get(side)) for side in ("from", "to")]
            if any(row is not None for row in typed):
                left, right = (json.dumps(row, ensure_ascii=False, sort_keys=True, indent=1) if row is not None else "(absent)" for row in typed)
                parts.append(f"<div class='side'><pre>{_esc(left)}</pre><pre>{_esc(right)}</pre></div>")
            elif before is not None and after is not None and before["kind"] in ("text", "doc") and after["kind"] in ("text", "doc"):
                parts.append(f"<div class='side'><pre>{_esc(before['text'])}</pre><pre>{_word_diff_html(before['text'], after['text'])}</pre></div>")
            else:
                left = (json.dumps(before["spec"], ensure_ascii=False, indent=1) if before["kind"] == "spec" else before["text"]) if before else "(absent)"
                right = (json.dumps(after["spec"], ensure_ascii=False, indent=1) if after["kind"] == "spec" else after["text"]) if after else "(absent)"
                parts.append(f"<div class='side'><pre>{_esc(left)}</pre><pre>{_esc(right)}</pre></div>")

    parts.append("<h2>Render matrix heatmap</h2>")
    if baseline is not None and releases:
        cells = [cell for cell in all_cells(store) if baseline.has(cell.entry)]
        parts.append("<table class='heat'><tr><th>proposal</th>" + "".join(f"<th title='{_esc(cell.entry)}'>{_esc(_short(cell.entry))}/{_esc(cell.name)}</th>" for cell in cells) + "</tr>")
        for state in proposals:
            release = releases.get(state.id)
            if release is None:
                continue
            row_cells = []
            for cell in cells:
                if not release.has(cell.entry):
                    row_cells.append("<td class='absent'>–</td>")
                    continue
                same = release.text(cell.entry, **cell.params) == baseline.text(cell.entry, **cell.params)
                row_cells.append(f"<td class='{'same' if same else 'changed'}'>{'=' if same else '~'}</td>")
            parts.append(f"<tr><td>{_esc(state.id)}</td>{''.join(row_cells)}</tr>")
        parts.append("</table>")
    else:
        parts.append("<p>No proposals yet.</p>")

    parts.append("<h2>Exposure slices and metrics</h2><table><tr><th>proposal</th><th>outcome</th><th>metric</th><th>n</th><th>b</th><th>c</th><th>gain pp</th><th>p</th><th>attempt</th><th>exposed</th><th>integrity</th><th>verdict</th><th>reason</th></tr>")
    for state in proposals:
        for doc in outcomes_of(store, state.id):
            stats = doc.get("stats") or {}
            exposed = stats.get("exposed") or {}
            gain, p = _gain_p(doc) if stats.get("gain_pp") is not None else ("-", "-")
            counted = doc.get("verdict") in VERDICTS  # a descriptive outcome spends no attempt
            parts.append(
                "<tr>" + "".join(f"<td>{_esc(value)}</td>" for value in (
                    state.id, doc.get("id"), doc.get("metric"), stats.get("n"), stats.get("b"), stats.get("c"),
                    gain,
                    p if stats.get("p") is not None else "-",
                    int((doc.get("attempt") or {}).get("used") or 0) + 1 if counted else "-",
                    f"{exposed.get('questions', 0)} q, {float(exposed.get('effect_pp') or 0.0):+.2f}pp" if exposed else "-",
                    doc.get("integrity"), doc.get("verdict"), doc.get("verdict_reason") or "",
                )) + "</tr>"
            )
    parts.append("</table>")
    if native_rows(store):
        parts.append("<section><h2>Native tool history</h2>")
        for row in native_rows(store):
            label = f"{row.get('namespace')} / {row.get('name')}:{row.get('version')}"
            parts.append(f"<details><summary>{_esc(label)}</summary><pre>{_esc(json.dumps(row, ensure_ascii=False, sort_keys=True, indent=2))}</pre></details>")
        parts.append("</section>")
    parts.append("</body></html>")
    return "\n".join(parts)
