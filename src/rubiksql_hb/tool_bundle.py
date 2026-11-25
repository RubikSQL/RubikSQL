"""Verified snapshot imports and inert answering contracts.

This module belongs to the package bootstrap graph and imports no HB, prompt
runtime, provider or bundle implementation. The finder owns only pinned modules.
A ``rubiksql.closure/2`` release's bundle derives from its typed Tool references
(``rubiksql.answering-bundle/3``); historical releases keep their ``tools.answering``
carrier. Neither format falls back to the other.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import importlib.abc
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, TYPE_CHECKING

from .hb_tool_entities import (
    BUNDLE_SCHEMA, TYPED_BUNDLE_SCHEMA, answering_definitions, canonical, digest, implementation_version, source_digest,
    tool_definition, typed_bundle, validate_answering, validate_typed_tools,
)

_PROCESS_BUNDLE: SnapshotBundle | None = None
_PROCESS_PATH: Path | None = None
_DEPLOYMENT_BUNDLE: SnapshotBundle | None = None
_BUNDLE_LOCK = threading.RLock()

if TYPE_CHECKING:
    from .prompts.types import AuthorityBinding, GitAuthorityBinding, VerifiedSnapshot
    from .prompts.runtime import Release


class _UnpinnedBundleFinder(importlib.abc.MetaPathFinder):
    """Refuse answering imports until the verified finder precedes this guard."""

    _rubiksql_unpinned_guard = True

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname == "rubiksql_hb.tool_declarations" and any(isinstance(finder, BundleFinder) for finder in sys.meta_path):
            return None
        if (fullname == "rubiksql_hb.tool_declarations" or fullname == "rubiksql_hb.tools"
                or fullname.startswith("rubiksql_hb.tools.")):
            raise ModuleNotFoundError(f"answering module {fullname} requires a verified bundle importer")
        return None


def install_unpinned_import_guard() -> None:
    """Keep direct library imports from falling through to checkout answering code.

    Source readers and host services remain importable without an execution pin.
    Installing a verified BundleFinder puts it before this final refusal boundary.
    """
    if not any(getattr(finder, "_rubiksql_unpinned_guard", False) for finder in sys.meta_path):
        sys.meta_path.insert(0, _UnpinnedBundleFinder())


def validate_snapshot_tools(payload: Mapping[str, Any], capsule_rows: Mapping[str, Mapping[str, Any]],
                            toolkit_rows: Mapping[str, Mapping[str, Any]], module_sources: Mapping[str, str]) -> None:
    """Verify inert native rows and the exact execution module source map.

    Args:
        payload (Mapping[str, Any]): Decoded tools.answering reference row.
        capsule_rows (Mapping[str, Mapping[str, Any]]): Native Capsule id map.
        toolkit_rows (Mapping[str, Mapping[str, Any]]): Native Toolkit id map.
        module_sources (Mapping[str, str]): Exact snapshot module text map.

    Returns:
        None.

    Raises:
        ValueError: A module or native reference differs from the pinned closure.
    """
    validate_answering(payload, capsule_rows=capsule_rows, toolkit_rows=toolkit_rows)
    _check_module_sources(payload, capsule_rows, module_sources)


def validate_typed_snapshot_tools(content_refs: Mapping[str, Any], tool_rows: Mapping[str, Mapping[str, Any]],
                                  capsule_rows: Mapping[str, Mapping[str, Any]],
                                  module_sources: Mapping[str, str]) -> tuple[dict, ...]:
    """Verify a closure/2 release's Tool rows, source Capsules and exact execution module source map.

    Args:
        content_refs (Mapping[str, Any]): The release's typed block (``tools`` and ``loader`` are read).
        tool_rows (Mapping[str, Mapping[str, Any]]): Tool rows by object id.
        capsule_rows (Mapping[str, Mapping[str, Any]]): Module and serializer Capsules by native id.
        module_sources (Mapping[str, str]): Exact snapshot module text map.

    Returns:
        tuple[dict, ...]: Validated definitions in serve order.

    Raises:
        ValueError: A Tool row, source pin, Capsule or module file differs from the pinned closure.
    """
    definitions = validate_typed_tools(content_refs, tool_rows=tool_rows, capsule_rows=capsule_rows)
    _check_module_sources(typed_bundle(content_refs), capsule_rows, module_sources)
    return definitions


def _check_module_sources(payload: Mapping[str, Any], capsule_rows: Mapping[str, Mapping[str, Any]],
                          module_sources: Mapping[str, str]) -> None:
    if set(module_sources) != {item["module"] for item in payload["modules"]}:
        raise ValueError("snapshot has an undeclared or missing answering module")
    for reference in payload["modules"]:
        source = module_sources[reference["module"]]
        if (source_digest(source) != reference["sha256"]
                or capsule_rows[reference["capsule_id"]]["manifest"]["source"]["source"] != source):
            raise ValueError("snapshot module source differs from its native capsule")


@dataclass(frozen=True)
class SnapshotBundle:
    """Detached verified bundle metadata and source, fixed for one process.

    A typed bundle (``rubiksql.answering-bundle/3``, closure/2) holds the release's Tool rows in serve order and
    only its module and serializer Capsules; its per-tool digests are the Tools' executable identities, equal to
    the callable Capsule versions of the historical format, so role digests survive the representation change.
    """

    snapshot: Any
    payload: Mapping[str, Any]
    capsule_rows: Mapping[str, Mapping[str, Any]]
    toolkit_rows: Mapping[str, Mapping[str, Any]]
    sources: Mapping[str, str]
    tool_rows: tuple[Mapping[str, Any], ...] = ()

    @property
    def definitions(self) -> tuple[Mapping[str, Any], ...]:
        if self.payload.get("schema") == TYPED_BUNDLE_SCHEMA:
            from .tool_definitions import validate_definitions

            return validate_definitions([tool_definition(row) for row in self.tool_rows],
                                        allowed_modules=[ref["module"] for ref in self.payload["modules"]])
        return answering_definitions(self.payload, self.capsule_rows, toolkit_rows=self.toolkit_rows)

    @property
    def contracts(self) -> tuple[Mapping[str, Any], ...]:
        return tuple({key: deepcopy(record[key]) for key in ("name", "description", "input_schema", "output_schema")}
                     for record in self.definitions)

    @property
    def tool_digests(self) -> dict[str, str]:
        if self.payload.get("schema") == TYPED_BUNDLE_SCHEMA:
            return {row["name"]: implementation_version(tool_definition(row), row["implementation"])
                    for row in self.tool_rows}
        if self.payload.get("schema") == BUNDLE_SCHEMA:
            return {ref["name"]: ref["version"] for ref in self.payload["tools"]}
        raise ValueError("schema1 Toolkit execution is permanently retired")


@dataclass(frozen=True)
class DeploymentSnapshot:
    """Verified independent consumer identity used only by the process importer."""

    deployment_id: str
    publication_id: str
    role: str
    digest: str
    path: Path
    tiers: Mapping[str, str]


def selected_tool_content(content: Mapping[str, Any], tool_rows: Mapping[str, Mapping[str, Any]],
                          definitions: Sequence[Mapping[str, Any]]) -> dict:
    """Project verified catalog refs onto one selected source closure.

    Args:
        content: Full verified typed catalog content.
        tool_rows: Exact native Tool rows indexed by object id.
        definitions: Ordered definitions already selected for the consumer.

    Returns:
        dict: Selected refs with their transitive source pins, empty package and
        unchanged serializer. Existing Tool versions are never reconstructed here.

    Raises:
        ValueError: Selection is absent, reordered or lacks its declared closure.
    """
    selected = {record["name"] for record in definitions}
    refs = [ref for ref in content["tools"] if tool_rows[ref["object_id"]]["name"] in selected]
    if [tool_rows[ref["object_id"]]["name"] for ref in refs] != [record["name"] for record in definitions]:
        raise ValueError("selected tools must follow exact catalog serve order")
    modules = {"rubiksql_hb.tools"}
    for ref in refs:
        modules.update(item["module"] for item in tool_rows[ref["object_id"]]["implementation"]["closure"])
    pinned = [ref for ref in content["loader"]["modules"] if ref["module"] in modules]
    if {ref["module"] for ref in pinned} != modules:
        raise ValueError("selected tool closure is missing a module pin")
    return {"tools": deepcopy(refs), "loader": {"modules": deepcopy(pinned),
                                                "serializer": deepcopy(content["loader"]["serializer"])}}


def install_deployment_bundle(resolved: Mapping[str, Any], role: str) -> SnapshotBundle:
    """Verify an independent deployment inertly, then install only its consumer scope.

    Args:
        resolved: Detached verified deployment/publication records and exact native
            Tool/Capsule maps from ``AuthorityRead.resolve_tools``.
        role: Trusted ``assistant`` or ``meta`` consumer.

    Returns:
        SnapshotBundle: Process-fixed selected implementation pin.

    Raises:
        ValueError: Publication, gate, source or selected refs are inconsistent.
        ImportError: Another process pin exists or tool modules were already loaded.
    """
    from .tool_definitions import select_definitions
    from .tools_catalog import validate_resolved_tools
    global _DEPLOYMENT_BUNDLE

    if role not in {"assistant", "meta"}:
        raise ValueError("independent tool deployments serve only assistant or meta")
    definitions = validate_resolved_tools(resolved)
    deployment, publication = resolved["deployment"], resolved["publication"]
    if deployment["consumer"] != role:
        raise ValueError("tool deployment belongs to another consumer")
    identity = digest({"deployment": deployment, "publication": publication["id"], "role": role})
    with _BUNDLE_LOCK:
        if _DEPLOYMENT_BUNDLE is not None:
            if _DEPLOYMENT_BUNDLE.snapshot.digest != identity:
                raise ImportError("process tool deployment pin is already fixed")
            return _DEPLOYMENT_BUNDLE
        selected = select_definitions(definitions, role)
        content = selected_tool_content(publication["content"], resolved["tool_rows"], selected)
        if content["tools"] != deployment["tools"]:
            raise ValueError("deployment tool selection differs from its catalog permissions")
        capsules = {ref["capsule_id"]: deepcopy(resolved["capsule_rows"][ref["capsule_id"]])
                    for ref in [*content["loader"]["modules"], content["loader"]["serializer"]]}
        rows = {ref["object_id"]: deepcopy(resolved["tool_rows"][ref["object_id"]]) for ref in content["tools"]}
        validate_typed_tools(content, tool_rows=rows, capsule_rows=capsules)
        sources = {ref["module"]: capsules[ref["capsule_id"]]["manifest"]["source"]["source"]
                   for ref in content["loader"]["modules"]}
        snapshot = DeploymentSnapshot(deployment["id"], publication["id"], role, identity,
                                      Path("/__rubiksql_tool_deployments__") / deployment["id"], {})
        bundle = SnapshotBundle(snapshot, typed_bundle(content), capsules, {}, sources,
                                tuple(rows[ref["object_id"]] for ref in content["tools"]))
        if any(isinstance(finder, BundleFinder) for finder in sys.meta_path) or _PROCESS_BUNDLE is not None:
            raise ImportError("process bundle pin is already fixed")
        if any(name in sys.modules for name in sources):
            raise ImportError("deployment importer must precede tool module imports")
        sys.meta_path.insert(0, BundleFinder(bundle))
        _DEPLOYMENT_BUNDLE = bundle
        return bundle


def deployment_bundle(role: str) -> SnapshotBundle:
    """Return the fixed independently deployed consumer pin without authority reads.

    Args:
        role: Trusted consumer whose pin is required.

    Returns:
        SnapshotBundle: Existing independent deployment for this process.

    Raises:
        ValueError: No matching independent deployment has been installed.
    """
    if _DEPLOYMENT_BUNDLE is None or _DEPLOYMENT_BUNDLE.snapshot.role != role:
        raise ValueError("consumer requires a matching fixed tool deployment")
    return _DEPLOYMENT_BUNDLE


def _module_path(module: str) -> str:
    return module.replace(".", "/") + ("/__init__.py" if module == "rubiksql_hb.tools" else ".py")


def read_bundle(snapshot: VerifiedSnapshot) -> SnapshotBundle:
    """Read a bundle after full snapshot verification, without importing tools.

    A closure/2 release's bundle is built from its typed references: the payload is
    :func:`hb_tool_entities.typed_bundle` of its ``content_refs``, the Tool rows come from
    ``native/tool-entity.json``, the module and serializer Capsules from ``tools/capsules`` and the
    sources from ``tools/modules``; no ``tools.answering`` carrier is read or fabricated.

    Args:
        snapshot (VerifiedSnapshot): Exact per-run execution snapshot.

    Returns:
        SnapshotBundle: Fixed native rows, contracts and execution sources.

    Raises:
        ValueError: tools.answering (closure/1) or a typed Tool reference (closure/2), or its closure, is
            absent or inconsistent.
    """
    from .prompts.entity_model import CLOSURE_V2
    from .prompts.snapshot import verify_snapshot

    snapshot = verify_snapshot(snapshot.path, expected_release=snapshot.release, expected_digest=snapshot.digest)
    release = json.loads((snapshot.path / "releases" / (snapshot.release + ".json")).read_text())
    if release.get("rules") == CLOSURE_V2:
        return _read_typed_bundle(snapshot, release)
    reference = release.get("refs", {}).get("tools.answering")
    if not isinstance(reference, str):
        raise ValueError("pinned release has no tools.answering bundle")
    envelope = json.loads((snapshot.path / "objects" / "tools.answering" / (reference[:16] + ".json")).read_text())
    row = envelope.get("row", envelope)
    text = row["manifest"]["meta"]["text"]
    payload = json.loads(text)
    if payload.get("schema") == "rubiksql.answering-bundle/1":
        raise ValueError("schema1 Toolkit execution is permanently retired")
    if text != canonical(payload):
        raise ValueError("tools.answering must use canonical JSON text")
    capsules = {item["capsule_id"]: json.loads((snapshot.path / "tools/capsules" / (item["capsule_id"] + ".json")).read_text())
                for item in [*payload["modules"], payload["serializer"], *payload.get("tools", [])]}
    sources = {item["module"]: (snapshot.path / "tools/modules" / _module_path(item["module"])).read_bytes().decode("utf-8")
               for item in payload["modules"]}
    validate_snapshot_tools(payload, capsules, {}, sources)
    if snapshot.bundle_digest != digest(payload):
        raise ValueError("snapshot bundle digest differs from pinned tools row")
    return SnapshotBundle(snapshot, payload, capsules, {}, sources)


def _read_typed_bundle(snapshot: VerifiedSnapshot, release: Mapping[str, Any]) -> SnapshotBundle:
    """Build a verified closure/2 bundle from the release's typed Tool references and pinned source files."""
    from .prompts.entity_model import TOOL_ENTITY, check_release_document

    check_release_document(release)
    content_refs = release["content_refs"]
    payload = typed_bundle(content_refs)
    stored = {row["object_id"]: row for row in json.loads((snapshot.path / "native" / f"{TOOL_ENTITY}.json").read_bytes())}
    missing = sorted(ref["object_id"] for ref in content_refs["tools"] if ref["object_id"] not in stored)
    if missing:
        raise ValueError(f"pinned release references absent Tool rows {missing}")
    tool_rows = tuple(stored[ref["object_id"]] for ref in content_refs["tools"])
    tools = snapshot.path / "tools"
    capsules = {item["capsule_id"]: json.loads((tools / "capsules" / f"{item['capsule_id']}.json").read_bytes())
                for item in [*payload["modules"], payload["serializer"]]}
    sources = {item["module"]: (tools / "modules" / _module_path(item["module"])).read_bytes().decode("utf-8")
               for item in payload["modules"]}
    validate_typed_snapshot_tools(content_refs, {row["object_id"]: row for row in tool_rows}, capsules, sources)
    if snapshot.bundle_digest != digest(payload):
        raise ValueError("snapshot bundle digest differs from the pinned typed Tool references")
    return SnapshotBundle(snapshot, payload, capsules, {}, sources, tool_rows)


def process_bundle() -> SnapshotBundle:
    """Read the process's verified pin; an existing importer keeps its fixed pin.

    Args:
        None.

    Returns:
        SnapshotBundle: Detached verified metadata.

    Raises:
        ValueError: The snapshot is missing or changed in this process.
    """
    from .prompts.snapshot import verify_snapshot
    global _PROCESS_BUNDLE, _PROCESS_PATH

    with _BUNDLE_LOCK:
        if _DEPLOYMENT_BUNDLE is not None:
            raise ValueError("a pinned process cannot change its independent deployment to answering")
        path = os.environ.get("RUBIKSQL_PROMPT_SNAPSHOT")
        if not path:
            raise ValueError("answering contracts require RUBIKSQL_PROMPT_SNAPSHOT")
        fixed_path = Path(path).resolve()
        if _PROCESS_BUNDLE is not None:
            if fixed_path != _PROCESS_PATH:
                raise ValueError("a pinned process cannot change answering snapshot")
            return _PROCESS_BUNDLE
        snapshot = verify_snapshot(fixed_path)
        for finder in sys.meta_path:
            if isinstance(finder, BundleFinder):
                if finder.bundle.snapshot.digest != snapshot.digest:
                    raise ValueError("a pinned process cannot change answering snapshot")
                _PROCESS_BUNDLE, _PROCESS_PATH = finder.bundle, fixed_path
                return finder.bundle
        _PROCESS_BUNDLE, _PROCESS_PATH = read_bundle(snapshot), fixed_path
        return _PROCESS_BUNDLE


def snapshot_contracts(snapshot: VerifiedSnapshot) -> tuple[Mapping[str, Any], ...]:
    """Read canonical declarations from the pinned native module/toolkit rows.

    Args:
        snapshot (VerifiedSnapshot): Verified per-run pin.

    Returns:
        tuple[Mapping[str, Any], ...]: Contracts in served order, without imports.
    """
    return read_bundle(snapshot).contracts


def snapshot_role_contract(snapshot: VerifiedSnapshot, context: Mapping[str, Any]) -> dict:
    """Compute one recorded role key from a separate pin without importing tools.

    Args:
        snapshot (VerifiedSnapshot): Parent or candidate's verified native closure.
        context (Mapping[str, Any]): Exact role/route/ordered tools context.

    Returns:
        dict: Canonical context and digest under the trusted current host's
            manual loop/helper/renderer counters. Historical host revisions
            must be checked for compatibility by the provenance caller.

    Raises:
        ValueError: The context is malformed or violates pinned callable permissions.
    """
    from . import tool_contract as contract

    bundle = read_bundle(snapshot)
    role = context.get("role")
    expected_keys = {"role", "route", "tools"} if role == "candidate" else {"role", "tools"}
    tools = context.get("tools")
    route = context.get("route", "default")
    if (role not in {"candidate", "judge", "reviewer"} or set(context) != expected_keys
            or not isinstance(route, str) or not isinstance(tools, list)
            or any(not isinstance(name, str) for name in tools) or len(set(tools)) != len(tools)):
        raise ValueError("recorded answering context is malformed")
    from .tool_definitions import select_definitions

    permitted = select_definitions(bundle.definitions, role, route)
    ordered = [record["name"] for record in permitted if record["name"] in tools]
    if ordered != tools:
        raise ValueError("recorded answering context contains unauthorized or unordered tools")
    versions = {name: digest({"tool": value, "host_helpers": contract.TOOL_VERSIONS.get(name, "1")})
                for name, value in bundle.tool_digests.items()}
    try:
        payload = contract._payload(role, route, frozenset(tools), list(bundle.contracts), versions=versions,
                                    shared_loops=contract.SHARED_LOOP_VERSIONS, loops=contract.LOOP_VERSIONS,
                                    loop_tools=contract.LOOP_TOOLS, adapters=contract.ADAPTER_VERSIONS,
                                    results=contract.result_versions(bundle), protocol=contract.MODEL_PROTOCOL,
                                    loop_hooks=contract.LOOP_HOOKS)
    except KeyError as error:
        raise ValueError("recorded answering context has no compatible host identity") from error
    result = contract._record(payload)
    if result["context"] != dict(context):
        raise ValueError("recorded answering tool order differs from its pinned declaration")
    return result


def expected_live_contract(release: Release, snapshot: VerifiedSnapshot) -> tuple[Mapping[str, Any], ...]:
    """Apply effective release knob defaults to detached canonical schemas.

    Args:
        release (Release): Pinned release whose effective knobs govern serving.
        snapshot (VerifiedSnapshot): Matching execution snapshot.

    Returns:
        tuple[Mapping[str, Any], ...]: Expected live contracts for parity checks.
    """
    records = deepcopy(snapshot_contracts(snapshot))
    for record in records:
        for tool, parameter, knob in (("get_rules", "k", "rules_top_k"), ("get_examples", "top_k", "examples_top_k")):
            if record["name"] == tool:
                record["input_schema"]["properties"][parameter]["default"] = release.knob(knob)
    return records


class BundleFinder(importlib.abc.MetaPathFinder, importlib.abc.InspectLoader):
    """Load only verified pinned bytes under the package's real module names."""

    def __init__(self, bundle: SnapshotBundle) -> None:
        self.bundle = bundle
        self.origins: dict[str, str] = {}
        self.bytes: dict[str, bytes] = {}
        for name, source in bundle.sources.items():
            tier = bundle.snapshot.tiers.get(name, "hb-executed")
            if tier not in {"hb-executed", "host-verified"}:
                raise ValueError(f"unknown answering execution tier for {name}")
            path = bundle.snapshot.path / "tools/modules" / _module_path(name)
            expected = source.encode("utf-8")
            if tier == "host-verified":
                path = Path(__file__).resolve().parent.parent / _module_path(name)
                if path.read_bytes() != expected:
                    raise ValueError(f"host-verified module {name} differs from its pin")
            self.origins[name] = str(path)
            self.bytes[name] = expected

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        owned = fullname in self.bytes or fullname == "rubiksql_hb.tools" or fullname.startswith("rubiksql_hb.tools.")
        if not owned:
            return None
        if fullname not in self.bytes:
            raise ModuleNotFoundError(f"unlisted pinned bundle module {fullname}")
        return importlib.util.spec_from_file_location(fullname, self.origins[fullname], loader=self,
            submodule_search_locations=[] if fullname == "rubiksql_hb.tools" else None)

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: Any) -> None:
        name = module.__name__
        if self.bundle.snapshot.tiers.get(name) == "host-verified" and Path(self.origins[name]).read_bytes() != self.bytes[name]:
            raise ImportError(f"host-verified module {name} changed after pinning")
        module.__file__ = self.origins[name]
        module.__loader__ = self
        exec(compile(self.bytes[name], self.origins[name], "exec", dont_inherit=True), module.__dict__)

    def get_source(self, fullname: str) -> str:
        return self.bytes[fullname].decode("utf-8")

    def get_code(self, fullname: str) -> Any:
        return compile(self.bytes[fullname], self.origins[fullname], "exec", dont_inherit=True)

    def is_package(self, fullname: str) -> bool:
        return fullname == "rubiksql_hb.tools"


def install_bundle_importer(snapshot: VerifiedSnapshot) -> BundleFinder:
    """Install one fixed pin before any owned module is loaded.

    Args:
        snapshot (VerifiedSnapshot): Verified execution snapshot.

    Returns:
        BundleFinder: Installed finder with recorded module origins.

    Raises:
        ImportError: A disk module is already loaded, or an existing pin differs.
    """
    global _PROCESS_BUNDLE, _PROCESS_PATH
    with _BUNDLE_LOCK:
        if _DEPLOYMENT_BUNDLE is not None:
            raise ImportError("process tool deployment pin is already fixed")
        bundle = read_bundle(snapshot)
        if _PROCESS_BUNDLE is not None and _PROCESS_BUNDLE.snapshot.digest != snapshot.digest:
            raise ImportError("client metadata pin is already fixed")
        for existing in sys.meta_path:
            if isinstance(existing, BundleFinder):
                if existing.bundle.snapshot.digest != snapshot.digest:
                    raise ImportError("process bundle pin is already fixed")
                return existing
        if any(name in sys.modules for name in bundle.sources):
            raise ImportError("bundle importer must be installed before tool module imports")
        finder = BundleFinder(bundle)
        sys.meta_path.insert(0, finder)
        _PROCESS_BUNDLE, _PROCESS_PATH = bundle, snapshot.path.resolve()
        return finder


def bootstrap_head_bundle(*, binding: AuthorityBinding | GitAuthorityBinding, into: Path | None = None) -> VerifiedSnapshot:
    """Pin the current authority once before importing an interactive consumer.

    Args:
        binding (AuthorityBinding | GitAuthorityBinding): Explicit authority binding.
        into (Path | None): Head snapshot cache parent. Defaults to the native authority's
            snapshot directory or the Git consumer's user-owned cache. Inherited pins
            require neither a cache directory nor an authority read.

    Returns:
        VerifiedSnapshot: Pin fixed in environment and importer for this process.
    """
    from .prompts.snapshot import head_snapshot, snapshot_env, verify_snapshot
    from .prompts.types import AuthorityBinding

    inherited = os.environ.get("RUBIKSQL_PROMPT_SNAPSHOT")
    if inherited:
        snapshot = verify_snapshot(Path(inherited))
    else:
        if into is None:
            into = (binding.db_path.parent.parent / "snapshots" if isinstance(binding, AuthorityBinding)
                    else Path.home() / ".cache" / "rubiksql-hb" / "snapshots")
        snapshot = head_snapshot(binding, into=into)
    os.environ.update(snapshot_env(snapshot, bundle=True))
    install_bundle_importer(snapshot)
    return snapshot


def assert_startup_parity(functions: Sequence[Any], served: Sequence[Any], release: Release) -> None:
    """Inspect raw callable and real served defaults before metadata normalization.

    Args:
        functions (Sequence[Any]): Imported pinned served functions.
        served (Sequence[Any]): Actual native Toolkit Tool descriptors.
        release (Release): Pinned effective knob values.

    Returns:
        None.

    Raises:
        RuntimeError: Callable, served schema, serializer or knob parity fails.
    """
    from . import tool_contract
    from .hb_tool_entities import _normalized_source
    from heavenbase.toolkit.serializers import jstr_serializer

    bundle = process_bundle()
    by_name = {fn.__name__: fn for fn in functions}
    schemas = {tool.name: tool.input_schema for tool in served}
    for name, parameter, knob in (("get_rules", "k", "rules_top_k"), ("get_examples", "top_k", "examples_top_k")):
        value = release.knob(knob)
        if (inspect.signature(by_name[name]).parameters[parameter].default != value
                or schemas[name]["properties"][parameter].get("default") != value):
            raise RuntimeError(f"raw default parity failed for {name}.{parameter}")
    values_module = sys.modules.get(by_name["search_values"].__module__)
    if values_module is None or values_module._SYMBOL_NEAR != release.knob("symbol_near"):
        raise RuntimeError("raw default parity failed for search_values.symbol_near")
    examples_module = sys.modules.get(by_name["get_examples"].__module__)
    # Historical menu-only pins have no flag and remain valid at the default OFF.
    # Refuse an ON release with an old pin instead of silently measuring a no-op.
    if getattr(examples_module, "_EXAMPLES_INLINE_SQL", False) is not release.knob("examples_inline_sql"):
        raise RuntimeError("raw default parity failed for get_examples.examples_inline_sql")
    serializer_digest = source_digest(_normalized_source(inspect.getsource(jstr_serializer)))
    if serializer_digest != bundle.payload["serializer"]["source_digest"]:
        raise RuntimeError("served serializer differs from the pinned wheel source")
    observed = tool_contract.reflect_tools(functions)
    if canonical(observed) != canonical(bundle.contracts):
        raise RuntimeError("pinned callable contracts differ from recorded bundle declarations")
