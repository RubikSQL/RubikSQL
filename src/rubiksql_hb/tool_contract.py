"""Declared answering-role metadata and explicit implementation identities.

Direct reflection never captures Capsules, builds a toolkit, opens a workspace,
or probes MCP. Declarations and explicit versions are frozen in process memory
on first use; every resolved context carries its digest in its cache keys.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import ast
from functools import lru_cache
import inspect
import json
import threading
from typing import Any, get_type_hints

from .model_render import MODEL_PROTOCOL
from .prompts.canon import canon, sha256_hex
from .prompts.entries import KNOBS
from .submission_gate import IMPLEMENTATION_VERSION as SUBMISSION_GATE_IMPLEMENTATION_VERSION

# These manual versions identify host helpers delegated to by a pinned tool.
# Bundle implementation and contract identities come from native snapshot rows.
# Registration, session cleanup, and knowledge writers outside answering loops
# are excluded. New callables inherit the generic helper identity; these entries
# version existing host helpers without constraining the open callable family.
TOOL_VERSIONS = {
    "get_db_info": "1", "get_tab_info": "1",
    "get_col_info": "1", "search_columns": "1", "search_values": "1",
    "run_sql": "1", "run_sql_base": "1", "save_sql_as_table": "1",
    "list_temp_tables": "1", "submit_final_sql": "1",
    "get_examples": "2", "expand_examples": "2", "query_knowledge": "3",
    "query_spec": "3", "get_rules": "1", "match_rule_triggers": "1",
    "audit_measure_grain": "1",
}

# _run_openai_compat, _exc_to_result and their shared helpers bump candidate
# below, covering both routes. _run_judge and _run_judge_loop bump judge below,
# covering the judge and reviewer.
SHARED_LOOP_VERSIONS = {"candidate": "4", "judge": "3"}
# Route hooks and reviewer risk/selection logic retain independent versions.
# Helpers shared across these two loop families bump both shared versions.
LOOP_VERSIONS = {"candidate": {"default": "1", "grain": "1"}, "judge": "1", "reviewer": "2"}
# These calls bypass the model's capability selection. Judge baseline retrieval
# uses get_rules even when its selected model capabilities omit that tool.
LOOP_TOOLS = {"candidate": ("match_rule_triggers",),
              "judge": ("get_rules", "match_rule_triggers"),
              "reviewer": ("match_rule_triggers", "submit_final_sql")}
# Internal hooks execute outside the served MCP toolkit. Their identity belongs
# to the role whose final they control.
LOOP_HOOKS = {
    "reviewer": ({"name": "submission_gate", "version": SUBMISSION_GATE_IMPLEMENTATION_VERSION},),
}
# "native" covers shared protocol projection. A role-local projection helper
# change belongs to its consuming loop versions, just like other loop helpers.
ADAPTER_VERSIONS = {"schema": "2", "native": "1", "arguments": "4"}
# Includes JSON result channels and render_rules for loop-called rule tools.
# A tool-specific renderer change affects its selected and loop consumers.
RESULT_VERSIONS = {
    "get_db_info": "1", "get_tab_info": "1",
    "get_col_info": "1", "search_columns": "1", "search_values": "1",
    "run_sql": "1", "run_sql_base": "1", "save_sql_as_table": "1",
    "list_temp_tables": "1", "submit_final_sql": "1",
    "get_examples": "1", "expand_examples": "1", "query_knowledge": "1",
    "query_spec": "1", "get_rules": "1", "match_rule_triggers": "1",
    "audit_measure_grain": "1",
}

_KNOB_DEFAULTS = {("get_rules", "k"): "rules_top_k", ("get_examples", "top_k"): "examples_top_k"}
_DECLARATION_PIN: str | None = None
_DISPATCH_CONTEXT: ContextVar[tuple[str, str, frozenset[str]] | None] = ContextVar("answering_dispatch", default=None)
#: Name sets derived from the process's pinned bundle, keyed by what they were derived from and held with the
#: bundle object that produced them. ``SnapshotBundle.definitions`` re-validates every Tool row on each access;
#: the answering loop checks permissions and argument names on every tool call and model turn, on its event loop,
#: so recomputing them there cost every session sharing the loop. A different bundle object recomputes.
_DERIVED: dict[tuple[Any, ...], tuple[Any, frozenset[str]]] = {}
_DERIVED_LOCK = threading.Lock()


def _bundle_derived(key: tuple[Any, ...], compute: Callable[[Any], frozenset[str]]) -> frozenset[str]:
    """Return ``compute(bundle)`` for the process bundle, computed once per bundle and ``key``.

    Args:
        key (tuple[Any, ...]): Hashable description of the derivation.
        compute (Callable[[Any], frozenset[str]]): Derives an immutable name set from the bundle.

    Returns:
        frozenset[str]: The derived names. A failed derivation raises and is not remembered.
    """
    from .tool_bundle import process_bundle

    bundle = process_bundle()
    with _DERIVED_LOCK:
        held = _DERIVED.get(key)
    if held is not None and held[0] is bundle:
        return held[1]
    value = compute(bundle)
    with _DERIVED_LOCK:
        _DERIVED[key] = (bundle, value)
    return value


def result_versions(bundle: Any) -> dict[str, str]:
    """Derive result identities for every callable, including newly published names.

    Args:
        bundle (SnapshotBundle): Verified native callable and serializer closure.

    Returns:
        dict[str, str]: Generic renderer content plus existing host overrides.
    """
    from . import model_render

    renderer = sha256_hex(inspect.getsource(model_render))
    return {record["name"]: sha256_hex(canon({"serializer": bundle.payload["serializer"]["source_digest"],
            "renderer": renderer, "host_version": RESULT_VERSIONS.get(record["name"], "1")}))
            for record in bundle.definitions}


class ToolContractDrift(RuntimeError):
    """Refuse an evaluation launch whose live answering registry has drifted.

    Callers must propagate this failure without retry, quorum top-up or fallback.
    """


def _input_schema(schema: dict) -> dict:
    # Match nl2sql._strip_db_id, including empty properties/required fields.
    injected = {"db_id", "session_id", "original_question"}
    return {
        **schema,
        "properties": {key: value for key, value in schema.get("properties", {}).items() if key not in injected},
        "required": [key for key in schema.get("required", []) if key not in injected],
    }


def _metadata(name: str, description: str | None, input_schema: dict, output_schema: dict | None) -> dict:
    schema = json.loads(canon(input_schema or {}))
    for (tool, parameter), knob in _KNOB_DEFAULTS.items():
        parameter_schema = schema.get("properties", {}).get(parameter, {})
        if name == tool and "default" in parameter_schema:
            parameter_schema["default"] = KNOBS[knob].default
    return {"name": name, "description": description or "", "input_schema": _input_schema(schema),
            "output_schema": json.loads(canon(output_schema))}


def reflect_tools(functions: Iterable[Callable[..., Any]]) -> list[dict]:
    """Project callables into served metadata, with committed knob defaults.

    Args:
        functions (Iterable[Callable[..., Any]]): Declarations in served order.
            No function executes.

    Returns:
        list[dict]: Name, description and stripped input/output schema records.
    """
    from heavenbase.capsule.docstring import parse_docstring
    from heavenbase.capsule.signature import inspect_callable_signature
    from heavenbase.toolkit.mcp import _mcp_output_schema

    records = []
    for fn in functions:
        raw_doc = inspect.getdoc(fn) or ""
        doc = parse_docstring(raw_doc)
        signature = inspect_callable_signature(fn)
        # capsule.capture merges descriptions into parameter metadata, not into
        # input_schema.properties. Preserve that exact projection.
        for parameter in signature.parameters:
            if parameter["name"] in doc["params"]:
                parameter["description"] = doc["params"][parameter["name"]]
        records.append(_metadata(fn.__name__, doc["summary"] or raw_doc, signature.input_schema, _mcp_output_schema(signature.output_schema)))
    return records


@lru_cache(maxsize=1)
def _declaration_json() -> str:
    from .tool_bundle import process_bundle
    global _DECLARATION_PIN

    bundle = process_bundle()
    _DECLARATION_PIN = bundle.snapshot.digest
    return canon(bundle.contracts)


def declaration_tools() -> list[dict]:
    """Return a detached copy of the immutable in-process metadata projection.

    Args:
        None.

    Returns:
        list[dict]: Project metadata in served order; versions resolve later.
    """
    return json.loads(_declaration_json())


@lru_cache(maxsize=128)
def _argument_adapter(name: str) -> Any:
    """Load the verified executor's signature without invoking its tool body."""
    from fastmcp.utilities.types import get_cached_typeadapter
    from .tool_bundle import install_bundle_importer, process_bundle

    # Answering clients inherit the snapshot, not the MCP process's importer.
    # Resolve callable types only through that same verified pin, never checkout
    # executors or a JSON Schema approximation of FastMCP's lax validation.
    bundle = process_bundle()
    install_bundle_importer(bundle.snapshot)
    from .tool_declarations import project_tool_functions

    declaration = next(record for record in bundle.definitions if record["name"] == name)
    function, = project_tool_functions(selection=(declaration,))

    def validate_only(*_args: Any, **_kwargs: Any) -> None:
        pass

    # Match HeavenBase's MCP wrapper: resolved hints, original signature and an
    # unconstrained return. JSON Schema loses callable types and lax coercions.
    signature = inspect.signature(function)
    parameters = [parameter for parameter in signature.parameters.values()
                  if parameter.name not in {"db_id", "session_id", "original_question"}]
    try:
        annotations = dict(get_type_hints(function))
    except Exception:
        # HeavenBase uses these raw hints when forward references cannot resolve.
        annotations = dict(getattr(function, "__annotations__", {}) or {})
    validate_only.__name__ = function.__name__
    setattr(validate_only, "__signature__", signature.replace(parameters=parameters, return_annotation=Any))
    validate_only.__annotations__ = {parameter.name: annotations.get(parameter.name, Any) for parameter in parameters}
    validate_only.__annotations__["return"] = Any
    return get_cached_typeadapter(validate_only)


def argument_error(name: str, inputs: object) -> str | None:
    """Validate normalized model arguments with the pinned executor's lax adapter.

    Args:
        name (str): Normalized callable name.
        inputs (object): Decoded model arguments, before host-injected fields.

    Returns:
        str | None: An ordinary invalid-arguments error, without argument values,
            or ``None`` when FastMCP accepts the model arguments. Validation never
            invokes the tool body or mutates the supplied arguments.
    """
    from pydantic import ValidationError

    declared = _bundle_derived(("declared",), lambda bundle: frozenset(record["name"] for record in bundle.definitions))
    if name not in declared:
        return f"Invalid arguments for unknown tool: {name}"
    if not isinstance(inputs, dict):
        return f"Invalid arguments for {name}: arguments must be an object"
    # The host discards caller-supplied context fields and injects trusted values.
    arguments = {key: value for key, value in inputs.items() if key not in {"db_id", "session_id", "original_question"}}
    try:
        _argument_adapter(name).validate_python(arguments, strict=False)
    except ValidationError as exc:
        error = exc.errors(include_input=False, include_context=False, include_url=False)[0]
        path = ".".join(str(part) for part in error["loc"]) or "arguments"
        return f"Invalid arguments for {name}: {path} violates {error['type']}"
    return None


@lru_cache(maxsize=1)
def _implementation_json() -> str:
    # Snapshot every input together. An unseen context later in this process
    # still uses the same versions and metadata as its first context.
    from .tool_bundle import process_bundle

    bundle = process_bundle()
    versions = {name: sha256_hex(canon({"tool": value, "host_helpers": TOOL_VERSIONS.get(name, "1")}))
                for name, value in bundle.tool_digests.items()}
    return canon({"tools": declaration_tools(), "tool_versions": versions,
                  "shared_loop_versions": SHARED_LOOP_VERSIONS,
                  "loop_versions": LOOP_VERSIONS, "loop_tools": LOOP_TOOLS,
                  "loop_hooks": LOOP_HOOKS,
                  "adapter_versions": ADAPTER_VERSIONS, "result_versions": result_versions(bundle),
                  "model_protocol": MODEL_PROTOCOL})


def _selection(role: str, route: str, disabled: Iterable[str] | None, capabilities: Iterable[str] | None) -> frozenset[str]:
    disabled = frozenset(disabled or ())
    capabilities = None if capabilities is None else frozenset(capabilities)
    if role == "candidate" and route != "default":
        # A styled route falls back to default while its requirements are unmet, which can change in-process.
        return _resolve_selection(role, route, disabled, capabilities)
    key = ("selection", role, route, disabled, capabilities)
    return _bundle_derived(key, lambda bundle: _resolve_selection(role, route, disabled, capabilities, bundle))


def _resolve_selection(role: str, route: str, disabled: Iterable[str] | None, capabilities: Iterable[str] | None,
                       bundle: Any = None) -> frozenset[str]:
    from .tool_bundle import process_bundle
    from .tool_definitions import select_definitions

    records = (bundle if bundle is not None else process_bundle()).definitions
    if role not in {"candidate", "judge", "reviewer"}:
        raise ValueError(f"unknown answering role {role!r}")
    if role == "candidate":
        if capabilities is not None:
            raise ValueError("candidate capabilities must be resolved by its route selector")
        effective_route = route
        if route != "default":
            from . import routes
            try:
                routes.get(route).require()
            except RuntimeError:
                effective_route = "default"
        selected = select_definitions(records, role, effective_route)
        return frozenset(record["name"] for record in selected
                         if record["name"] not in (disabled or ()) or role in record.get("required_for", ()))
    if route != "default" or disabled:
        raise ValueError(f"{role} has no candidate route or disabled-tool selector")
    permitted = frozenset(record["name"] for record in select_definitions(records, role, route))
    active = permitted if capabilities is None else frozenset(capabilities)
    denied = active - permitted
    if denied:
        raise ValueError(f"{role}/{route} selects unauthorized answering tools: {sorted(denied)}")
    return active


def loop_selection(role: str, route: str = "default") -> frozenset[str]:
    """Return the pinned tools permitted for explicit host loop calls.

    Args:
        role (str): Trusted runtime seat.
        route (str): Trusted candidate route.

    Returns:
        frozenset[str]: Independently declared host-loop callable names.
    """
    from .tool_definitions import select_definitions

    return _bundle_derived(("loop", role, route), lambda bundle: frozenset(
        record["name"] for record in select_definitions(bundle.definitions, role, route, calling="loop")))


@contextmanager
def dispatch_context(role: str, route: str = "default", disabled: Iterable[str] | None = None, *,
                     capabilities: Iterable[str] | None = None):
    """Bind trusted model permissions for one run and restore the caller's seat.

    Args:
        role (str): Host-selected runtime seat.
        route (str): Host-selected candidate route.
        disabled (Iterable[str] | None): Candidate exclusions.
        capabilities (Iterable[str] | None): Authorized judge/reviewer subset.

    Yields:
        None.
    """
    token = _DISPATCH_CONTEXT.set((role, route, _selection(role, route, disabled, capabilities)))
    try:
        yield
    finally:
        _DISPATCH_CONTEXT.reset(token)


def dispatch_allowed(name: str, *, calling: str = "runtime") -> bool:
    """Check a callable before hooks, native dispatch, or SQL-proof reuse.

    Args:
        name (str): Requested callable name.
        calling (str): ``model`` restricts calls to exposed tools; ``runtime``
            also permits explicitly declared host-loop calls.

    Returns:
        bool: Whether the pinned role/route permits the call.
    """
    held = _DISPATCH_CONTEXT.get()
    role, route, active = held if held is not None else ("candidate", "default", _selection("candidate", "default", None, None))
    return name in active or (calling == "runtime" and name in loop_selection(role, route))


@lru_cache(maxsize=None)
def callable_parameters(name: str) -> frozenset[str]:
    """Read parameter names from the fixed verified source without tool imports.

    Args:
        name (str): Declared callable name.

    Returns:
        frozenset[str]: Python parameter names, including host-injected ones.

    Raises:
        ValueError: The name or locator is absent from the verified closure.
    """
    from .tool_bundle import process_bundle

    bundle = process_bundle()
    record = next((record for record in bundle.definitions if record["name"] == name), None)
    if record is None:
        raise ValueError(f"unknown pinned callable {name!r}")
    node: Any = ast.parse(bundle.sources[record["module"]])
    for part in record["qualname"].split("."):
        node = next((child for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                     and child.name == part), None)
        if node is None:
            raise ValueError(f"invalid pinned callable locator for {name!r}")
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError(f"invalid pinned callable locator for {name!r}")
    return frozenset(argument.arg for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs])


def _payload(role: str, route: str, active: frozenset[str], declarations: list[dict], *, versions: dict,
             shared_loops: dict, loops: dict, loop_tools: dict, adapters: dict, results: dict, protocol: str,
             loop_hooks: dict) -> dict:
    called = frozenset(loop_tools[role])
    missing = (active | called) - {record["name"] for record in declarations}
    if missing:
        raise ValueError(f"{role}/{route} selects undeclared tools: {sorted(missing)}")
    unversioned = (active | called) - (versions.keys() & results.keys())
    if unversioned:
        raise ValueError(f"{role}/{route} selects unversioned answering tools: {sorted(unversioned)}")
    tools = [{**record, "version": versions[record["name"]]} for record in declarations if record["name"] in active]
    by_name = {record["name"]: record for record in declarations}
    hooks = [{**by_name[name], "version": versions[name], "result_version": results[name]}
             for name in loop_tools[role]]
    return {
        "role": role,
        **({"route": route} if role == "candidate" else {}),
        "shared_loop_version": shared_loops["candidate" if role == "candidate" else "judge"],
        "loop_version": loops[role][route] if role == "candidate" else loops[role],
        "model_protocol": protocol,
        "adapters": {**adapters, "results": {record["name"]: results[record["name"]] for record in tools}},
        "tools": tools,
        "loop_tools": hooks,
        **({"loop_hooks": loop_hooks[role]} if loop_hooks.get(role) else {}),
    }


def _record(payload: dict) -> dict:
    return {
        "context": {"role": payload["role"], **({"route": payload["route"]} if "route" in payload else {}),
                    "tools": [record["name"] for record in payload["tools"]]},
        "digest": sha256_hex(canon(payload)),
    }


@lru_cache(maxsize=None)
def _role_contract_json(role: str, route: str, active: frozenset[str]) -> str:
    frozen = json.loads(_implementation_json())
    payload = _payload(role, route, active, frozen["tools"], versions=frozen["tool_versions"],
                       shared_loops=frozen["shared_loop_versions"], loops=frozen["loop_versions"],
                       loop_tools=frozen["loop_tools"],
                       adapters=frozen["adapter_versions"], results=frozen["result_versions"],
                       protocol=frozen["model_protocol"], loop_hooks=frozen["loop_hooks"])
    return canon(_record(payload))


def assert_selection_available(role: str, route: str = "default", disabled: Iterable[str] | None = None, *,
                               capabilities: Iterable[str] | None = None) -> None:
    """Refuse launching a selection outside the independently pinned permissions.

    Args:
        role (str): candidate, judge or reviewer.
        route (str): Candidate route, or default otherwise.
        disabled (Iterable[str] | None): Candidate exclusions.
        capabilities (Iterable[str] | None): Explicit effective child selection,
            including a candidate launch whose child environment differs.

    Returns:
        None.

    Raises:
        ToolContractDrift: A selected name is not permitted by its native row.
    """
    from .tool_bundle import process_bundle

    bundle = process_bundle()
    from .tool_definitions import select_definitions

    permitted = frozenset(record["name"] for record in select_definitions(bundle.definitions, role, route))
    if role == "candidate" and capabilities is not None:
        active = frozenset(capabilities)
        if active.intersection(disabled or ()):
            raise ValueError("explicit candidate selection includes disabled tools")
    else:
        if role == "candidate" and route != "default":
            from . import routes
            routes.get(route).require()
        active = _selection(role, route, disabled, capabilities)
    denied = active - permitted
    required = frozenset(record["name"] for record in select_definitions(bundle.definitions, role, route)
                         if role in record.get("required_for", ()))
    if denied or (role == "candidate" and not required <= active) or not frozenset(LOOP_TOOLS[role]) <= (active | loop_selection(role, route)):
        raise ToolContractDrift(f"pinned bundle denies {role}/{route} selection: {sorted(denied)}")


def role_contract(role: str, route: str = "default", disabled: Iterable[str] | None = None, *, capabilities: Iterable[str] | None = None) -> dict:
    """Resolve the immutable process contract for one answering context.

    Args:
        role (str): ``candidate``, ``judge`` or ``reviewer``.
        route (str): Candidate route name; ``default`` for an unstyled candidate.
        disabled (Iterable[str] | None): Candidate exclusions, subject to
            required/route protections.
        capabilities (Iterable[str] | None): Explicit judge/reviewer capability
            selection, when recorded.

    Returns:
        dict: Detached ``context`` and ``digest``. The digest always enters this
            role's whole and memo cache keys, independently of other roles.

    Raises:
        ValueError: The role, route, or selected declaration is invalid.
    """
    from .tool_bundle import process_bundle

    if _DECLARATION_PIN is not None and process_bundle().snapshot.digest != _DECLARATION_PIN:
        raise ToolContractDrift("answering client snapshot pin changed in process")
    disabled = frozenset(disabled or ())
    capabilities = None if capabilities is None else frozenset(capabilities)
    active = _selection(role, route, disabled, capabilities)
    return json.loads(_role_contract_json(role, route, active))


def role_payload(role: str, route: str = "default", disabled: Iterable[str] | None = None, *,
                 capabilities: Iterable[str] | None = None) -> dict:
    """Return the full fixed role identity payload for host consumers.

    Args:
        role (str): candidate, judge or reviewer.
        route (str): Candidate route; default otherwise.
        disabled (Iterable[str] | None): Candidate exclusions.
        capabilities (Iterable[str] | None): Explicit judge/reviewer selection.

    Returns:
        dict: Tool closure digests plus host loop/result/adapter identities.
    """
    role_contract(role, route, disabled, capabilities=capabilities)
    active = _selection(role, route, disabled, capabilities)
    frozen = json.loads(_implementation_json())
    return _payload(role, route, active, frozen["tools"], versions=frozen["tool_versions"],
                    shared_loops=frozen["shared_loop_versions"], loops=frozen["loop_versions"],
                    loop_tools=frozen["loop_tools"], adapters=frozen["adapter_versions"],
                    results=frozen["result_versions"], protocol=frozen["model_protocol"], loop_hooks=frozen["loop_hooks"])


def assert_live_contract(tools: Sequence[object], role: str, route: str = "default", disabled: Iterable[str] | None = None,
                         *, capabilities: Iterable[str] | None = None) -> None:
    """Refuse live answering metadata drift before a model call.

    Args:
        tools (Sequence[object]): Served registry descriptors in received order. Generic and
            harness-only entries are excluded by the resolved answering context.
        role (str): Answering role whose declared selection is expected.
        route (str): Candidate route, or ``default``.
        disabled (Iterable[str] | None): Candidate exclusions.
        capabilities (Iterable[str] | None): Declared judge/reviewer selection.

    Returns:
        None.

    Raises:
        ToolContractDrift: Live order, description or schema differs. This is a
            fatal launch failure; callers must not retry or return a fallback.
    """
    active = _selection(role, route, disabled, capabilities)
    frozen = json.loads(_implementation_json())
    answering = active | frozenset(frozen["loop_tools"][role])
    expected = [record for record in frozen["tools"] if record["name"] in answering]
    missing = answering - {record["name"] for record in expected}
    if missing:
        raise ToolContractDrift(f"{role}/{route} selects undeclared tools: {sorted(missing)}")
    observed = []
    for tool in tools:
        if tool.name not in answering:
            continue
        schema = getattr(tool, "input_schema", None)
        if schema is None:
            schema = getattr(tool, "inputSchema", {})
        output = getattr(tool, "output_schema", None)
        if output is None:
            output = getattr(tool, "outputSchema", None)
        observed.append(_metadata(tool.name, getattr(tool, "description", None), schema, output))
    for position in range(max(len(expected), len(observed))):
        declared = expected[position] if position < len(expected) else None
        served = observed[position] if position < len(observed) else None
        if declared is None or served is None or declared["name"] != served["name"]:
            expected_name = declared["name"] if declared is not None else "<end>"
            observed_name = served["name"] if served is not None else "<end>"
            detail = f"position {position}: expected tool {expected_name!r}, received {observed_name!r}"
        else:
            field = next((field for field in ("description", "input_schema", "output_schema")
                          if declared[field] != served[field]), None)
            if field is None:
                continue
            detail = f"tool {declared['name']!r} at position {position}: {field} differs"
        raise ToolContractDrift(f"live MCP contract differs from declared {role}/{route}: {detail}")
