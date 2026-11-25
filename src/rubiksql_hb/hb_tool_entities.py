"""Inert answering-tool identities: canonical Tool versions, source Capsules and bundles.

New-format releases pin canonical ``tool-entity`` rows plus inert module source Capsules
(:func:`build_tool_rows`, ``rubiksql.answering-bundle/3``). Historical releases keep their
``tools.answering`` carrier with callable Capsules (schema2) or Toolkits (schema1), which stay
readable as recorded. The execution verifier uses only the standard library. No native source
restore, hidden registry, provider construction or application tool import occurs here.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.util
from io import StringIO
import json
from pathlib import Path
import re
import subprocess
import textwrap
import tokenize
import zipfile
from typing import Any

NAMESPACE = "rubiksql.answering"
MODULE_NAMESPACE = NAMESPACE + ".module"
TOOL_NAMESPACE = NAMESPACE + ".tool"
BUNDLE_SCHEMA = "rubiksql.answering-bundle/2"
LEGACY_BUNDLE_SCHEMA = "rubiksql.answering-bundle/1"
#: The executable bundle document of a closure/2 release, derived from its typed Tool references.
TYPED_BUNDLE_SCHEMA = "rubiksql.answering-bundle/3"
MODULE_FORMAT = "rubiksql.module-source/2"
LEGACY_MODULES = ("rubiksql_hb.tool_declarations", "rubiksql_hb.tools",
           "rubiksql_hb.tools.grain", "rubiksql_hb.tools.knowledge",
           "rubiksql_hb.tools.schema", "rubiksql_hb.tools.sql", "rubiksql_hb.tools.template")
# Historical source maps remain valid; current Git authoring discovers the tools directory.
MODULES = LEGACY_MODULES[1:]
_TOOL_MODULE = re.compile(r"^rubiksql_hb\.tools\.[A-Za-z_][A-Za-z0-9_]*$")
SERVED_ORDER = ("register_database", "get_db_info", "get_tab_info", "get_col_info", "search_columns", "search_values",
                "run_sql", "run_sql_base", "save_sql_as_table", "list_temp_tables", "submit_final_sql", "close_session",
                "get_examples", "save_example", "save_synonym", "expand_synonyms", "expand_examples",
                "query_knowledge", "query_spec", "save_rule", "update_rule", "get_rules", "match_rule_triggers",
                "check_functional_dependency", "audit_measure_grain", "render_template")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_KNOB_DEFAULTS = {("get_rules", "k"): "rules_top_k", ("get_examples", "top_k"): "examples_top_k"}
_SERVER_REEXPORTS = {
    **{name: "rubiksql_hb.tools.schema" for name in (
        "register_database", "get_db_info", "get_tab_info", "get_col_info", "search_columns", "search_values",
        "check_functional_dependency")},
    **{name: "rubiksql_hb.tools.sql" for name in (
        "run_sql", "run_sql_base", "save_sql_as_table", "list_temp_tables", "submit_final_sql", "close_session")},
    **{name: "rubiksql_hb.tools.knowledge" for name in (
        "get_examples", "save_example", "save_synonym", "expand_synonyms", "expand_examples", "query_knowledge",
        "query_spec", "save_rule", "update_rule", "get_rules", "match_rule_triggers")},
    "audit_measure_grain": "rubiksql_hb.tools.grain", "render_template": "rubiksql_hb.tools.template",
    **{name: "rubiksql_hb.tools.schema" for name in ("db_info", "tab_info", "col_info", "fuzzy_enum", "fuzzy_enum_values", "_fd_check")},
    **{name: "rubiksql_hb.tools.knowledge" for name in ("read_examples", "resolve_synonyms", "_get_rules", "_match_rule_triggers", "_SPEC_HIDDEN_IDS")},
    **{name: "rubiksql_hb.tools.grain" for name in ("_audit_measure_grain", "_grain_alias_list", "_split_outside_quotes")},
    "render_sql_template": "rubiksql_hb.tools.template",
}
# Only historical pinned aggregate implementations call these server aliases.
# Current tools call their private helpers directly; authoring still resolves an
# explicit borrowed alias to the actual source owner in either representation.
_SPLIT_SERVER_REEXPORTS = {
    "db_info": "get_db_info", "tab_info": "get_tab_info", "col_info": "get_col_info",
    "fuzzy_enum": "search_columns", "fuzzy_enum_values": "search_values", "_fd_check": "check_functional_dependency",
    "read_examples": "get_examples", "resolve_synonyms": "expand_synonyms", "_get_rules": "get_rules",
    "_match_rule_triggers": "match_rule_triggers", "_SPEC_HIDDEN_IDS": "query_spec",
    "_audit_measure_grain": "audit_measure_grain", "_grain_alias_list": "audit_measure_grain",
    "_split_outside_quotes": "audit_measure_grain", "render_sql_template": "render_template",
}
# These composition-root services remain in the host's manual implementation
# identity. Unknown server exports cannot silently be classified as host code.
_SERVER_HOST_EXPORTS = {
    "_register_db", "_ws_for", "_DB_REGISTRY", "_persist_registry", "_SEARCH_COLUMNS_PROFILE", "_db_url",
    "exec_session", "exec_sql", "save_temp", "list_session_tables", "_close_session", "sweep_sessions",
    "knowledge_sessions", "_submit_sql", "add_example", "add_synonym", "_update_rule", "normalize_rule_scope",
    "_save_rule", "_query_knowledge", "refuse_code_reduce", "code_reduce_disabled", "_query_frame",
    "_QUERY_ENTITIES", "_QUERY_LIMIT_MAX", "_RULES_TOP_K", "_EXAMPLES_TOP_K", "_workspace",
    "_retrieval_open_kwargs", "_prompt_release", "check_retrieval_version", "open_workspace",
}
# Native/safety and persisted-format contracts remain trusted host code. Project
# read-tool behavior and diagnostic helpers must be included in source Capsules.
_READ_HOST_IMPORTS = {
    "rubiksql_hb.tool_context": frozenset({"current_tool_context"}),
}
_META_HOST_IMPORTS = {
    **_READ_HOST_IMPORTS,
    "rubiksql_hb.sql_host": frozenset({"_dedupe_cols", "guarded_readonly_cursor"}),
    "rubiksql_hb.query_policy": frozenset({"refuse_code_reduce"}),
    "rubiksql_hb.hb_tool_entities": frozenset({"BUNDLE_SCHEMA", "convert_answering", "tool_definition"}),
    "rubiksql_hb.tool_definitions": frozenset({"select_definitions"}),
    "rubiksql_hb.prompts.entity_model": frozenset({
        "CLOSURE_V2", "EntityModelError", "MIGRATION_KIND", "PARAMETER_ENTITY", "RULE_ENTITY", "TOOL_ENTITY",
        "change_kind", "check_ref", "check_release_refs", "closure_view", "corpus_from_text", "is_canonical",
        "parameter_row", "partition_corpus", "stored_json", "typed_ref", "values_from_text"}),
    "rubiksql_hb.prompts.entries": frozenset({"CLOSURE_RULES", "DROPPED_ENTRIES", "KNOBS", "KNOBS_ENTRY", "RULES_ENTRY"}),
}


def canonical(value: Any) -> str:
    """Encode a detached JSON value with the project's canonical identity format.

    Args:
        value (Any): JSON-compatible value.

    Returns:
        str: Sorted, compact UTF-8 JSON text.
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    """Hash canonical JSON.

    Args:
        value (Any): JSON-compatible semantic content.

    Returns:
        str: SHA-256 hexadecimal digest.
    """
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def source_digest(source: str) -> str:
    """Hash exact UTF-8 module bytes.

    Args:
        source (str): Complete module source, including whitespace.

    Returns:
        str: SHA-256 hexadecimal digest.
    """
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _native_id(kind: str, namespace: str, name: str, version: str) -> str:
    # Vendored HB hash_id uses tagged tuples and the serializer's default JSON
    # separators, then a decimal MD5 id of width42. P0 verifies this native seam.
    value = {"__obj_type__": "tuple", "__obj_data__": [kind, namespace, name, version]}
    data = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return str(int(hashlib.md5(data).hexdigest(), 16)).zfill(42)


def _native_digest(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _semantic(value: Any) -> Any:
    result = deepcopy(dict(value))
    for current in (result, result.get("manifest"),
                    (result.get("manifest") or {}).get("manifest")):
        if isinstance(current, dict):
            current.pop("created_at", None)
            current.pop("updated_at", None)
    return result


def row_digest(row: Mapping[str, Any], *, capsule: bool) -> str:
    """Hash full native row semantics without incidental capture locations/times.

    Args:
        row (Mapping[str, Any]): Complete Capsule or Toolkit row.
        capsule (bool): Whether this is a Capsule; used to validate its shape.

    Returns:
        str: Canonical semantic SHA-256.

    Raises:
        ValueError: Native row shape or mirrored fields are inconsistent.
    """
    (_check_capsule_integrity if capsule else _check_toolkit_integrity)(row)
    return digest(_semantic(row))


def _check_declarations(source: str, implementations: Mapping[str, str]) -> None:
    """Keep the declaration factory inert because it is outside tool closures."""
    module = "rubiksql_hb.tool_declarations"
    tree = ast.parse(source, filename=module)
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body.pop(0)
    imports: dict[str, str] = {}
    factory = None
    type_imports = {"collections.abc": {"Callable"}, "typing": {"Any"}, "__future__": {"annotations"}}
    for node in body:
        if isinstance(node, ast.ImportFrom) and factory is None:
            base = importlib.util.resolve_name("." * node.level + (node.module or ""), "rubiksql_hb") if node.level else node.module
            for alias in node.names:
                if alias.asname is not None or alias.name in imports:
                    raise ValueError("tool_declarations must use direct, unique callable imports")
                if implementations.get(alias.name) != base and alias.name not in type_imports.get(base, set()):
                    raise ValueError("tool_declarations contains an undeclared callable or executable import")
                imports[alias.name] = base
        elif isinstance(node, ast.FunctionDef) and node.name == "project_tool_functions" and factory is None:
            factory = node
        else:
            raise ValueError("tool_declarations must contain only imports and its inert ordered factory")
    if factory is None or any(imports.get(name) != implementations[name] for name in SERVED_ORDER):
        raise ValueError("tool_declarations must import all 26 recorded implementations directly")
    arguments = factory.args
    if (factory.decorator_list or getattr(factory, "type_params", []) or arguments.posonlyargs or arguments.args
            or arguments.vararg or arguments.kwonlyargs or arguments.kwarg or arguments.defaults or arguments.kw_defaults):
        raise ValueError("tool_declarations factory must be undecorated and take no arguments")
    if factory.returns is not None:
        annotation = ast.parse("tuple[Callable[..., Any], ...]", mode="eval").body
        if (ast.dump(factory.returns, include_attributes=False) != ast.dump(annotation, include_attributes=False)
                or imports.get("Callable") != "collections.abc" or imports.get("Any") != "typing"):
            raise ValueError("tool_declarations factory has an executable or unsupported return annotation")
    statements = list(factory.body)
    if (statements and isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant)
            and isinstance(statements[0].value.value, str)):
        statements.pop(0)
    if (len(statements) != 1 or not isinstance(statements[0], ast.Return)
            or not isinstance(statements[0].value, ast.Tuple)
            or any(not isinstance(item, ast.Name) for item in statements[0].value.elts)
            or tuple(item.id for item in statements[0].value.elts) != SERVED_ORDER):
        raise ValueError("tool_declarations factory must only return the exact ordered 26 callables")


def _imports(module: str, source: str, listed: set[str], *,
             server_reexports: Mapping[str, str] = _SERVER_REEXPORTS) -> set[str]:
    dependencies: set[str] = set()
    server_aliases: set[tuple[str, ...]] = set()
    package = module if module == "rubiksql_hb.tools" else module.rpartition(".")[0]
    try:
        tree = ast.parse(source, filename=module)
    except (SyntaxError, TypeError, ValueError) as error:
        raise ValueError(f"bundle module {module} has invalid Python source") from error

    def server_export(name: str) -> None:
        if name in server_reexports:
            target = server_reexports[name]
            if target not in listed:
                raise ValueError(f"undeclared server reexport source {target!r} in bundle module {module}")
            dependencies.add(target)
        elif name not in _SERVER_HOST_EXPORTS:
            raise ValueError(f"undeclared server reexport {name!r} in bundle module {module}")

    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Name) and node.id == "__import__":
            raise ValueError(f"dynamic import is not declared in bundle module {module}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.search(
                r"(?:^|[/\\])(?:resources|eval_results|runs|results)[/\\]", node.value):
            raise ValueError(f"bundle module {module} may not read gold/resource/run paths")
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            for alias in node.names:
                if alias.name == "rubiksql_hb.server":
                    server_aliases.add((alias.asname,) if alias.asname else ("rubiksql_hb", "server"))
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = importlib.util.resolve_name("." * node.level + base, package)
            if base == "builtins" and any(alias.name == "__import__" for alias in node.names):
                raise ValueError(f"dynamic import is not declared in bundle module {module}")
            names = [base, *(base + "." + alias.name for alias in node.names)]
            if base == "rubiksql_hb.server":
                for alias in node.names:
                    server_export(alias.name)
            elif base == "rubiksql_hb":
                server_aliases.update((alias.asname or alias.name,) for alias in node.names if alias.name == "server")
        elif isinstance(node, ast.Call):
            is_dynamic = (isinstance(node.func, ast.Name) and node.func.id == "__import__") or (
                isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
            if is_dynamic:
                raise ValueError(f"dynamic import is not declared in bundle module {module}")
        for name in names:
            if name == "importlib" or name.startswith("importlib."):
                raise ValueError(f"dynamic import machinery is not declared in bundle module {module}")
            if name in {"rubiksql_hb.lab_server", "rubiksql_hb.lab_ingest", "rubiksql_hb.lab_score",
                        "rubiksql_hb.lab_sql", "rubiksql_hb.lab_probe", "rubiksql_hb.prompts.authority"}:
                raise ValueError(f"bundle module {module} cannot import host scoring/gold authority")
            if name in listed:
                dependencies.add(name)
            elif name.startswith("rubiksql_hb.tools."):
                # from package import symbol is valid only when its base is a
                # listed implementation module; symbols themselves are not modules.
                base = name.rpartition(".")[0]
                if base not in listed or base == "rubiksql_hb.tools":
                    raise ValueError(f"undeclared bundle import {name!r} in {module}")
    if server_aliases:
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        roots = {alias[0] for alias in server_aliases}
        for node in ast.walk(tree):
            if isinstance(node, ast.arg) and node.arg in roots or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in roots:
                raise ValueError(f"server module alias is rebound in bundle module {module}")
            if not isinstance(node, ast.Name) or node.id not in roots:
                continue
            if not isinstance(node.ctx, ast.Load):
                raise ValueError(f"server module alias is rebound in bundle module {module}")
            parts = [node.id]
            current = node
            while isinstance(parents.get(current), ast.Attribute) and parents[current].value is current:
                current = parents[current]
                parts.append(current.attr)
            matching = [alias for alias in server_aliases if tuple(parts[:len(alias)]) == alias and len(parts) > len(alias)]
            if len(matching) != 1:
                raise ValueError(f"dynamic or unresolved server module alias in bundle module {module}")
            server_export(parts[len(matching[0])])
    dependencies.discard(module)
    dependencies.discard("rubiksql_hb.tool_declarations")
    return dependencies


def _closure(module: str, graph: Mapping[str, set[str]]) -> list[str]:
    reached = {module}
    pending = [module]
    while pending:
        for item in graph[pending.pop()]:
            if item not in reached:
                reached.add(item)
                pending.append(item)
    return sorted(reached)


def _check_role_imports(records: Sequence[Mapping[str, Any]], sources: Mapping[str, str],
                        graph: Mapping[str, set[str]]) -> None:
    """Keep read-only host seams outside every answering execution closure."""
    from .tool_definitions import ANSWERING_ROLES

    for record in records:
        roles = set(record["visible_to"]) | set(record["loop_visible_to"])
        answering = not roles or bool(roles.intersection(ANSWERING_ROLES))
        read_roles = roles.intersection({"assistant", "meta"})
        if answering and read_roles:
            raise ValueError("answering and independent read consumers cannot share an implementation context")
        permitted = _META_HOST_IMPORTS if read_roles == {"meta"} else _READ_HOST_IMPORTS
        for module in _closure(record["module"], graph):
            package = module.rpartition(".")[0]
            for node in ast.walk(ast.parse(sources[module], filename=module)):
                if isinstance(node, ast.Import):
                    imports = [(alias.name, None) for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    base = importlib.util.resolve_name("." * node.level + (node.module or ""), package) if node.level else node.module
                    imports = [(base, alias.name) for alias in node.names]
                else:
                    continue
                for base, name in imports:
                    if answering:
                        if base == "rubiksql_hb.tool_context" or base == "rubiksql_hb" and name == "tool_context":
                            raise ValueError("answering source cannot bind independent read tool contexts")
                    elif base and base.startswith("rubiksql_hb"):
                        if base in sources or base.startswith("rubiksql_hb.tools."):
                            continue
                        if name is None or name not in permitted.get(base, ()):
                            raise ValueError(f"read tool source {module} imports an undeclared host capability")


def _capsule(module: str, source: str, dependencies: Sequence[str], *, serializer: bool = False,
             module_format: str | None = None) -> dict:
    hashed_source = source_digest(source)
    version = hashed_source if module_format is None else digest({"module_format": module_format, "source_hash": hashed_source})
    namespace = NAMESPACE if serializer else MODULE_NAMESPACE
    name = "__jstr_serializer__" if serializer else module
    layer = {"type": "rubiksql-module", "module": module, "source": source, "source_hash": hashed_source}
    capabilities = {"mcp_tool": {"ok": False, "reasons": ["project snapshot importer required"]},
                    "registry_persistence": {"ok": True, "reasons": []}}
    manifest = {"id": _native_id("capsule", namespace, name, version), "name": name, "namespace": namespace,
                "version": version, "kind": "python-function" if serializer else "python-module", "status": "active",
                "description": "Pinned native JSON serializer identity" if serializer else f"Pinned module {module}",
                "tags": ["rubiksql", "serializer" if serializer else "module"], "identity": {"module": module},
                "signature": {}, "input_schema": {}, "output_schema": {}, "restore_layers": [layer],
                "dependencies": {"modules": list(dependencies)}, "capabilities": capabilities, "runtime": {},
                "source": {"module": module, "source": source, "source_hash": hashed_source},
                "provenance": {} if module_format is None else {"module_format": module_format}, "trust": {},
                "fingerprints": {"source_hash": hashed_source, "signature_hash": _native_digest({}), "restore_hash": _native_digest([layer])},
                "created_at": 0.0, "updated_at": 0.0}
    return {"object_id": manifest["id"], **{key: deepcopy(manifest[key]) for key in (
        "name", "namespace", "version", "status", "description", "tags", "capabilities", "fingerprints")},
        "manifest": manifest, "created_at": 0.0, "updated_at": 0.0, "last_reason": "answering bundle publication"}


def _check_capsule_integrity(row: Mapping[str, Any]) -> None:
    row = _mapping(row, "Capsule row")
    manifest = _mapping(row.get("manifest"), "Capsule manifest")
    if manifest.get("namespace") == TOOL_NAMESPACE:
        _check_function_capsule(row)
        return
    source = _mapping(manifest.get("source"), "Capsule source")
    if not isinstance(source.get("source"), str) or not isinstance(source.get("module"), str):
        raise ValueError("Capsule must carry complete UTF-8 module source")
    hashed_source = source_digest(source["source"])
    module_format = _mapping(manifest.get("provenance"), "Capsule provenance").get("module_format")
    if module_format not in {None, MODULE_FORMAT}:
        raise ValueError("Capsule has an unsupported module format")
    version = hashed_source if module_format is None else digest({"module_format": module_format, "source_hash": hashed_source})
    layer = {"type": "rubiksql-module", "module": source["module"], "source": source["source"], "source_hash": hashed_source}
    if (manifest.get("version") != version or source.get("source_hash") != hashed_source
            or manifest.get("restore_layers") != [layer]
            or manifest.get("signature") != {} or manifest.get("input_schema") != {} or manifest.get("output_schema") != {}
            or manifest.get("identity") != {"module": source["module"]}
            or manifest.get("fingerprints") != {"source_hash": hashed_source, "signature_hash": _native_digest({}), "restore_hash": _native_digest([layer])}
            or manifest.get("status") != "active"
            or manifest.get("capabilities", {}).get("mcp_tool", {}).get("ok") is not False
            or manifest.get("id") != _native_id("capsule", manifest.get("namespace"), manifest.get("name"), version)
            or row.get("object_id") != manifest.get("id")
            or any(row.get(key) != manifest.get(key) for key in (
                "name", "namespace", "version", "status", "description", "tags", "capabilities", "fingerprints"))):
        raise ValueError("Capsule has inconsistent module identity or mirrored native fields")


def _check_toolkit_integrity(row: Mapping[str, Any]) -> None:
    row = _mapping(row, "Toolkit row")
    manifest = _mapping(row.get("manifest"), "Toolkit manifest")
    for key in ("capsule_refs", "loop_capsule_refs"):
        refs = manifest.get(key, [])
        if not isinstance(refs, list) or any(not isinstance(item, Mapping) for item in refs):
            raise ValueError(f"Toolkit {key} must be a list of refs")
        for item in refs:
            metadata = _mapping(item.get("metadata"), "Toolkit ref metadata")
            _mapping(metadata.get("contract"), "Toolkit contract")
            _mapping(metadata.get("context"), "Toolkit context")
    expected_version = digest({"capsule_refs": manifest["capsule_refs"],
                               "loop_capsule_refs": manifest.get("loop_capsule_refs", [])})
    if (manifest.get("version") != expected_version or manifest.get("namespace") != NAMESPACE
            or manifest.get("id") != _native_id("toolkit", NAMESPACE, manifest.get("name"), expected_version)
            or row.get("object_id") != manifest.get("id") or manifest.get("status") != "active"
            or any(row.get(key) != manifest.get(key) for key in (
                "name", "namespace", "version", "status", "description", "capsule_refs", "serving", "capabilities"))):
        raise ValueError("Toolkit has inconsistent version or mirrored native fields")
    nested = manifest.get("manifest")
    if nested is not None and _semantic(nested) != _semantic({key: value for key, value in manifest.items() if key != "manifest"}):
        raise ValueError("Toolkit nested manifest differs")


@dataclass(frozen=True)
class AnsweringRows:
    """Detached publication input; no workspace or persisted callable is opened."""

    payload: Mapping[str, Any]
    capsule_rows: tuple[Mapping[str, Any], ...]
    toolkit_rows: tuple[Mapping[str, Any], ...]


def snapshot_files(payload: Mapping[str, Any], capsule_rows: Mapping[str, Mapping[str, Any]],
                   toolkit_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, bytes]:
    """Serialize inert recovery rows and exact source files; this grants no execution permission.

    Args:
        payload (Mapping[str, Any]): Validated tools.answering data.
        capsule_rows (Mapping[str, Mapping[str, Any]]): Native Capsule id map.
        toolkit_rows (Mapping[str, Mapping[str, Any]]): Native Toolkit id map.

    Returns:
        dict[str, bytes]: K2 relative POSIX snapshot paths to immutable bytes.
    """
    if payload.get("schema") == LEGACY_BUNDLE_SCHEMA:
        _validate_legacy_answering(payload, capsule_rows=capsule_rows, toolkit_rows=toolkit_rows)
    else:
        validate_answering(payload, capsule_rows=capsule_rows)
    files = {}
    for ref in [*payload["modules"], payload["serializer"], *payload.get("tools", [])]:
        row = capsule_rows[ref["capsule_id"]]
        files[f"tools/capsules/{row['object_id']}.json"] = canonical(row).encode("utf-8")
    for reference in payload.get("toolkits", []):
        row = toolkit_rows[reference["toolkit_id"]]
        files[f"tools/toolkits/{row['object_id']}.json"] = canonical(row).encode("utf-8")
    for ref in payload["modules"]:
        module = ref["module"]
        path = module.replace(".", "/") + ("/__init__.py" if module == "rubiksql_hb.tools" else ".py")
        files["tools/modules/" + path] = capsule_rows[ref["capsule_id"]]["manifest"]["source"]["source"].encode("utf-8")
    return files


def _checkout_tool_input(checkout: Path, definitions: Sequence[Mapping[str, Any]] | None = None) -> tuple[dict, Any, str]:
    """Read complete inert source, definitions and the vendored serializer.

    Args:
        checkout (Path): Explicit clean authoring checkout.
        definitions (Sequence[Mapping[str, Any]] | None): Complete current
            callable metadata, bypassing the initial Git contract seed.

    Returns:
        tuple: Source text by module, complete definitions and serializer source.
    """
    source_root = checkout / "src"
    sources = {}
    for path in sorted((source_root / "rubiksql_hb/tools").glob("*.py")):
        if path.is_symlink():
            raise ValueError("answering authoring source must be a regular checkout file")
        module = "rubiksql_hb.tools" + ("" if path.stem == "__init__" else "." + path.stem)
        sources[module] = path.read_bytes().decode("utf-8")
    if definitions is None:
        definitions = json.loads((source_root / "rubiksql_hb/answering_tools.json").read_text(encoding="utf-8"))
    wheels = list((checkout / "vendor").glob("heavenbase-*.whl"))
    if len(wheels) != 1:
        raise ValueError("authoring requires the one repository-owned HB wheel")
    with zipfile.ZipFile(wheels[0]) as wheel:
        serializer_module = wheel.read("heavenbase/toolkit/serializers.py").decode("utf-8")
    node = next(node for node in ast.parse(serializer_module).body
                if isinstance(node, ast.FunctionDef) and node.name == "jstr_serializer")
    serializer = "\n".join(serializer_module.splitlines()[node.lineno - 1:node.end_lineno]) + "\n"
    return sources, definitions, serializer


def answering_selection(definitions: Sequence[Mapping[str, Any]]) -> tuple[dict, ...]:
    """Select answering and existing host-only definitions before source loading.

    Args:
        definitions (Sequence[Mapping[str, Any]]): Complete verified catalog metadata.

    Returns:
        tuple[dict, ...]: Ordered definitions with answering model/loop visibility,
        plus definitions with neither visibility set populated.
    """
    from .tool_definitions import ANSWERING_ROLES, validate_definitions

    roles = frozenset(ANSWERING_ROLES)
    return tuple(dict(record) for record in validate_definitions(definitions)
                 if roles.intersection((*record["visible_to"], *record["loop_visible_to"]))
                 or not record["visible_to"] and not record["loop_visible_to"])


def _selected_sources(sources: Mapping[str, str], definitions: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Keep exactly selected transitive source dependencies and the empty package."""
    sources = _mapping(sources, "tool module sources")
    if sources.get("rubiksql_hb.tools") != "":
        raise ValueError("tool sources require the empty tools package init")
    implementations = {record["name"]: record["module"] for record in definitions}
    reexports = dict(_SERVER_REEXPORTS)
    reexports.update({name: implementations[owner] for name, owner in _SPLIT_SERVER_REEXPORTS.items()
                     if owner in implementations})
    reexports.update(implementations)
    reached = {"rubiksql_hb.tools"}
    pending = list(implementations.values())
    while pending:
        module = pending.pop()
        if module in reached:
            continue
        if module not in sources:
            raise ValueError(f"selected tool source is absent: {module}")
        reached.add(module)
        pending.extend(_imports(module, sources[module], set(sources), server_reexports=reexports) - reached)
    return {module: sources[module] for module in sorted(reached)}


def checkout_answering_rows(checkout: Path, *, definitions: Sequence[Mapping[str, Any]] | None = None) -> AnsweringRows:
    """Prepare Git answering source without importing independent consumers.

    Args:
        checkout (Path): Explicit clean authoring checkout.
        definitions (Sequence[Mapping[str, Any]] | None): Complete catalog metadata;
            null reads the committed authoring definitions.

    Returns:
        AnsweringRows: Answering and host-only records with their exact source closure.
    """
    sources, definitions, serializer = _checkout_tool_input(checkout, definitions)
    selected = answering_selection(definitions)
    return build_answering_rows(_selected_sources(sources, selected), definitions=selected, serializer_source=serializer)


def _direct_tool_input(tool_source: Path) -> tuple[Mapping[str, str], Any, str]:
    path = Path(tool_source)
    if path.is_dir():
        path = path / "answering.json"
    if path.is_symlink():
        raise ValueError("direct answering input cannot be a symlink")
    body = _mapping(json.loads(path.read_text(encoding="utf-8")), "direct answering input")
    if set(body) != {"module_sources", "definitions", "serializer_source"}:
        raise ValueError("direct answering input needs module_sources, definitions and serializer_source")
    return body["module_sources"], body["definitions"], body["serializer_source"]


def _check_source_commit(checkout: Path, source_commit: str | None) -> None:
    if source_commit:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, capture_output=True,
                              text=True, check=True, timeout=30).stdout.strip()
        requested = subprocess.run(["git", "rev-parse", source_commit], cwd=checkout, capture_output=True,
                                   text=True, check=True, timeout=30).stdout.strip()
        if head != requested:
            raise ValueError("authoring source commit must equal the deployed checkout")


def prepare_catalog_rows(source_commit: str | None = None, tool_source: Path | None = None) -> TypedTools:
    """Build the complete catalog through the same inert closure proof as BIRD.

    Args:
        source_commit (str | None): Exact deployed Git HEAD when provided.
        tool_source (Path | None): Direct JSON source input, or null for Git authoring.

    Returns:
        TypedTools: Complete ordered Tool versions and source/serializer Capsules.

    Raises:
        ValueError: Source identity, definitions or exact closure proof is invalid.
    """
    checkout = Path(__file__).resolve().parents[2]
    _check_source_commit(checkout, source_commit)
    sources, definitions, serializer = (_direct_tool_input(tool_source) if tool_source is not None
                                        else _checkout_tool_input(checkout))
    return build_tool_rows(sources, definitions=definitions, serializer_source=serializer)


def prepare_answering_rows(source_commit: str | None = None, tool_source: Path | None = None) -> AnsweringRows:
    """Prepare exact authoring input for the authority CLI without executing it.

    Args:
        source_commit (str | None): Git source commit, required to equal deployed
            HEAD when supplied. Publication performs the merged/clean checks.
        tool_source (Path | None): Direct JSON with module_sources, definitions
            and serializer_source. Null uses the checkout.

    Returns:
        AnsweringRows: Complete detached native publication closure.

    Raises:
        ValueError: Source commit or direct input shape is inconsistent.
    """
    checkout = Path(__file__).resolve().parents[2]
    if tool_source is not None:
        sources, definitions, serializer = _direct_tool_input(tool_source)
        selected = answering_selection(definitions)
        return build_answering_rows(_selected_sources(sources, selected), definitions=selected, serializer_source=serializer)
    _check_source_commit(checkout, source_commit)
    return checkout_answering_rows(checkout)


def default_selections(definitions: Sequence[Mapping[str, Any]] | None = None) -> tuple[dict, ...]:
    """Derive the four conventional answering contexts from callable metadata.

    Args:
        definitions (Sequence[Mapping[str, Any]] | None): Verified callable
            metadata, or the Git bootstrap definitions when null.

    Returns:
        tuple[dict, ...]: Detached role/route projections. These are views, never
            persisted Toolkit membership or capability allowlists.
    """
    from .tool_definitions import select_definitions

    if definitions is None:
        definitions = json.loads(Path(__file__).with_name("answering_tools.json").read_text(encoding="utf-8"))
    return tuple({"role": role, **({"route": route, "disabled": []} if role == "candidate" else {}),
                  "tools": [record["name"] for record in select_definitions(definitions, role, route)],
                  "loop_tools": [record["name"] for record in select_definitions(definitions, role, route, calling="loop")]}
                 for role, route in (("candidate", "default"), ("candidate", "grain"), ("judge", "default"), ("reviewer", "default")))


def _build_legacy_answering_rows(module_sources: Mapping[str, str], *, contracts: Sequence[Mapping[str, Any]],
                         selections: Sequence[Mapping[str, Any]], serializer_source: str) -> AnsweringRows:
    """Build native immutable rows without importing or executing their source.

    Args:
        module_sources (Mapping[str, str]): Complete explicit module map.
        contracts (Sequence[Mapping[str, Any]]): Metadata in exact served order.
        selections (Sequence[Mapping[str, Any]]): Role/route, ordered selected and
            loop tool names, with optional disabled/capabilities selectors.
        serializer_source (str): Pinned wheel serializer source.

    Returns:
        AnsweringRows: Content-addressed Capsule/Toolkit rows and reference payload.

    Raises:
        ValueError: Modules, imports, contracts or selections are ambiguous.
    """
    if set(module_sources) != set(LEGACY_MODULES) or module_sources["rubiksql_hb.tools"] != "":
        raise ValueError("answering bundle requires the exact module map and empty package init")
    from .tool_receipts import scan_tool_texts

    scan_tool_texts([*module_sources.values(), serializer_source, canonical(contracts)])
    trees = {}
    for module, source in module_sources.items():
        try:
            trees[module] = ast.parse(source, filename=module)
        except (SyntaxError, TypeError, ValueError) as error:
            raise ValueError(f"bundle module {module} has invalid Python source") from error
    by_name: dict[str, dict] = {}
    implementations: dict[str, str] = {}
    for record in contracts:
        record = deepcopy(dict(record))
        if (not all(key in record for key in ("name", "description", "input_schema", "output_schema"))
                or not isinstance(record["name"], str) or not isinstance(record["description"], str)
                or not isinstance(record["input_schema"], Mapping)
                or record["output_schema"] is not None and not isinstance(record["output_schema"], Mapping)):
            raise ValueError("served contract must carry valid name, description and schemas")
        name = record["name"]
        if name in by_name:
            raise ValueError(f"duplicate served tool name {name!r}")
        matching = [module for module, tree in trees.items()
                    if module != "rubiksql_hb.tool_declarations" and any(
                        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
                        for node in tree.body)]
        if len(matching) != 1:
            raise ValueError(f"tool {name!r} needs exactly one defining module")
        implementations[name] = matching[0]
        by_name[name] = {key: record[key] for key in ("name", "description", "input_schema", "output_schema")}
    if tuple(by_name) != SERVED_ORDER:
        raise ValueError("served contract order differs from the 26-tool declaration")
    _check_declarations(module_sources["rubiksql_hb.tool_declarations"], implementations)
    server_reexports = {**_SERVER_REEXPORTS, **implementations}
    graph = {name: _imports(name, text, set(module_sources), server_reexports=server_reexports)
             for name, text in module_sources.items()}
    modules = {name: _capsule(name, text, sorted(graph[name])) for name, text in module_sources.items()}
    serializer = _capsule("heavenbase.toolkit.serializers", serializer_source, (), serializer=True)
    for module, row in modules.items():
        row["manifest"]["provenance"]["contracts"] = [deepcopy(by_name[name]) for name in SERVED_ORDER
                                                        if implementations[name] == module]
    toolkits = []
    selected_refs = []
    seen_selections: set[str] = set()
    for supplied in selections:
        context = {key: deepcopy(value) for key, value in supplied.items() if key not in {"tools", "loop_tools"}}
        if context.get("role") == "candidate":
            context.setdefault("route", "default")
            context["disabled"] = sorted(context.get("disabled", []))
        elif context.get("role") not in {"judge", "reviewer"}:
            raise ValueError("unknown answering role")
        label = canonical(context)
        if label in seen_selections:
            raise ValueError("duplicate answering selection")
        seen_selections.add(label)
        groups = {}
        for calling, key in (("selected", "tools"), ("loop", "loop_tools")):
            refs = []
            names = [item["name"] if isinstance(item, Mapping) else item for item in supplied.get(key, [])]
            if len(set(names)) != len(names) or any(name not in by_name for name in names):
                raise ValueError("unknown or duplicate selected tool")
            for name in names:
                module = implementations[name]
                closure = [{"module": item, "sha256": source_digest(module_sources[item])} for item in _closure(module, graph)]
                contract = by_name[name]
                tool_digest = digest({"contract": contract, "closure": closure})
                module_row = modules[module]
                metadata = {"context": context, "contract": contract, "module": module, "qualname": name,
                            "closure": closure, "tool_digest": tool_digest, "serializer_id": serializer["object_id"],
                            "serializer_digest": source_digest(_normalized_source(serializer_source)), "calling": calling,
                            "knob_defaults": {parameter: knob for (tool, parameter), knob in _KNOB_DEFAULTS.items() if tool == name}}
                refs.append({"name": name, "alias": name, "capsule_id": module_row["object_id"], "version": module_row["version"],
                             "serializer_capsule_id": serializer["object_id"], "serializer_name": serializer["name"],
                             "serializer_version": serializer["version"], "description": contract["description"],
                             "input_schema": contract["input_schema"], "output_schema": contract["output_schema"] or {},
                             "annotations": {}, "required": True, "metadata": metadata})
            groups[key] = refs
        version = digest({"capsule_refs": groups["tools"], "loop_capsule_refs": groups["loop_tools"]})
        name = "candidate." + context["route"] if context["role"] == "candidate" else context["role"]
        if context.get("disabled") or context.get("capabilities"):
            name += "." + digest(context)[:16]
        manifest = {"id": _native_id("toolkit", NAMESPACE, name, version), "name": name, "namespace": NAMESPACE,
                    "version": version, "status": "active", "description": f"Pinned {name} answering selection",
                    "capsule_refs": groups["tools"], "loop_capsule_refs": groups["loop_tools"], "serving": {},
                    "capabilities": {}, "created_at": 0.0, "updated_at": 0.0}
        manifest["manifest"] = deepcopy(manifest)
        row = {"object_id": manifest["id"], **{key: deepcopy(manifest[key]) for key in (
            "name", "namespace", "version", "status", "description", "capsule_refs", "serving", "capabilities")},
            "manifest": manifest, "created_at": 0.0, "updated_at": 0.0, "last_reason": "answering bundle publication"}
        toolkits.append(row)
        selected_refs.append({"selection": context, "toolkit_id": row["object_id"], "digest": row_digest(row, capsule=False)})
    payload = {"schema": LEGACY_BUNDLE_SCHEMA, "modules": [{"module": name, "capsule_id": row["object_id"],
                "sha256": row["version"], "row_digest": row_digest(row, capsule=True)} for name, row in sorted(modules.items())],
               "serializer": {"capsule_id": serializer["object_id"], "source_digest": source_digest(_normalized_source(serializer_source)),
                              "row_digest": row_digest(serializer, capsule=True)},
               "toolkits": sorted(selected_refs, key=lambda item: canonical(item["selection"])), "served_order": list(SERVED_ORDER)}
    return AnsweringRows(deepcopy(payload), tuple([*modules.values(), serializer]), tuple(toolkits))


def _validate_legacy_answering(payload: Mapping[str, Any], *, capsule_rows: Mapping[str, Mapping[str, Any]],
                       toolkit_rows: Mapping[str, Mapping[str, Any]]) -> None:
    """Audit an existing schema1 cleanup preimage as inert rows; never execute or serve it.

    Args:
        payload (Mapping[str, Any]): Decoded tools.answering data.
        capsule_rows (Mapping[str, Mapping[str, Any]]): Rows indexed by native id.
        toolkit_rows (Mapping[str, Mapping[str, Any]]): Rows indexed by native id.

    Returns:
        None: All references, native mirrors and tool closures match.

    Raises:
        ValueError: Any referenced row or semantic digest is inconsistent.
    """
    if payload.get("schema") != LEGACY_BUNDLE_SCHEMA or payload.get("served_order") != list(SERVED_ORDER):
        raise ValueError("invalid answering payload schema or served order")
    try:
        sources = {}
        for reference in payload["modules"]:
            row = capsule_rows[reference["capsule_id"]]
            if (row_digest(row, capsule=True) != reference["row_digest"] or row["version"] != reference["sha256"]
                    or row["namespace"] != MODULE_NAMESPACE or row["name"] != reference["module"]
                    or reference["module"] in sources):
                raise ValueError("module capsule reference differs")
            sources[reference["module"]] = row["manifest"]["source"]["source"]
        serializer_ref = payload["serializer"]
        serializer = capsule_rows[serializer_ref["capsule_id"]]
        serializer_source = serializer["manifest"]["source"]["source"]
        if (row_digest(serializer, capsule=True) != serializer_ref["row_digest"]
                or source_digest(_normalized_source(serializer_source)) != serializer_ref["source_digest"]):
            raise ValueError("serializer reference differs")
        contracts: dict[str, dict] = {}
        selections = []
        for reference in payload["toolkits"]:
            row = toolkit_rows[reference["toolkit_id"]]
            if row_digest(row, capsule=False) != reference["digest"]:
                raise ValueError("toolkit reference differs")
            selected = row["capsule_refs"]
            loop = row["manifest"].get("loop_capsule_refs", [])
            for ref in [*selected, *loop]:
                name = ref["metadata"]["contract"]["name"]
                if name in contracts and contracts[name] != ref["metadata"]["contract"]:
                    raise ValueError("two contracts under one served name")
                contracts[name] = ref["metadata"]["contract"]
            selections.append({**reference["selection"], "tools": [ref["alias"] for ref in selected],
                               "loop_tools": [ref["alias"] for ref in loop]})
        # Management tools may not occur in a role toolkit. Their contracts are
        # stored on the module manifest to preserve the full startup registry.
        for reference in payload["modules"]:
            row = capsule_rows[reference["capsule_id"]]
            for record in row.get("manifest", {}).get("provenance", {}).get("contracts", []):
                name = record["name"]
                if name in contracts and contracts[name] != record:
                    raise ValueError("module contract conflicts with selected contract")
                contracts[name] = record
        if set(contracts) != set(SERVED_ORDER):
            raise ValueError("missing served management contracts")
        rebuilt = _build_legacy_answering_rows(sources, contracts=[contracts[name] for name in SERVED_ORDER],
                                      selections=selections, serializer_source=serializer_source)
        if canonical(rebuilt.payload) != canonical(payload):
            raise ValueError("answering references do not reproduce their source/contract closure")
        for row in rebuilt.capsule_rows:
            if _semantic(row) != _semantic(capsule_rows[row["object_id"]]):
                raise ValueError("module capsule semantics differ")
        for row in rebuilt.toolkit_rows:
            if _semantic(row) != _semantic(toolkit_rows[row["object_id"]]):
                raise ValueError("toolkit semantics differ")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("answering bundle has missing or malformed references") from error


def _bootstrap_definitions(module_sources: Mapping[str, str], contracts: Sequence[Mapping[str, Any]],
                           selections: Sequence[Mapping[str, Any]]) -> tuple[dict, ...]:
    """Upgrade contract seeds once; normal runtime permissions live on tool rows."""
    from .tool_definitions import ROLES, validate_definitions

    trees = {module: ast.parse(source, filename=module) for module, source in module_sources.items()}
    known_routes = {role: {item.get("route", "default") for item in selections if item.get("role") == role} for role in ROLES}
    records = []
    for order, contract in enumerate(contracts):
        if "visible_to" in contract:
            records.append(deepcopy(dict(contract)))
            continue
        if set(contract) != {"name", "description", "input_schema", "output_schema"}:
            raise ValueError("served contract must carry name, description and schemas")
        name = contract["name"]
        matching = [module for module, tree in trees.items() if any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name for node in tree.body)]
        if len(matching) != 1:
            raise ValueError(f"tool {name!r} needs exactly one defining module")
        visibility = {key: {role: {item.get("route", "default") for item in selections
                                  if item.get("role") == role and name in item.get(key, [])} for role in ROLES}
                      for key in ("tools", "loop_tools")}
        routes = {}
        for role in ROLES:
            permitted = visibility["tools"][role] | visibility["loop_tools"][role]
            if permitted and permitted != known_routes[role]:
                routes[role] = sorted(permitted)
        records.append({**deepcopy(dict(contract)), "module": matching[0], "qualname": name, "order": order,
                        "visible_to": [role for role in ROLES if visibility["tools"][role]],
                        "loop_visible_to": [role for role in ROLES if visibility["loop_tools"][role]], "routes": routes,
                        "required_for": ["candidate"] if visibility["tools"]["candidate"] and name in {
                            "get_db_info", "get_rules", "get_examples", "submit_final_sql", "audit_measure_grain"} else []})
    return validate_definitions(records, allowed_modules=module_sources)


def _function_capsule(definition: Mapping[str, Any], modules: Mapping[str, Mapping[str, Any]],
                      closure: Sequence[Mapping[str, Any]], serializer: Mapping[str, Any]) -> dict:
    module, name = definition["module"], definition["name"]
    module_row = modules[module]
    source = {"module": module, "qualname": definition["qualname"], "capsule_id": module_row["object_id"],
              "source_hash": module_row["manifest"]["source"]["source_hash"]}
    metadata = {"tool": deepcopy(dict(definition)), "closure": deepcopy(list(closure)),
                "serializer_id": serializer["object_id"],
                "serializer_digest": source_digest(_normalized_source(serializer["manifest"]["source"]["source"])),
                "knob_defaults": {parameter: knob for (tool, parameter), knob in _KNOB_DEFAULTS.items() if tool == name}}
    version = digest({"source": source, **metadata})
    signature = {"input_schema": definition["input_schema"], "output_schema": definition["output_schema"]}
    capabilities = {"mcp_tool": {"ok": False, "reasons": ["project snapshot importer required"]},
                    "registry_persistence": {"ok": True, "reasons": []}}
    manifest = {"id": _native_id("capsule", TOOL_NAMESPACE, name, version), "name": name, "namespace": TOOL_NAMESPACE,
                "version": version, "kind": "python-function", "status": "active", "description": definition["description"],
                "tags": ["rubiksql", "tool"], "identity": {"module": module, "qualname": definition["qualname"]},
                "signature": deepcopy(signature), "input_schema": deepcopy(definition["input_schema"]),
                "output_schema": deepcopy(definition["output_schema"]), "restore_layers": [],
                "dependencies": {"modules": [item["module"] for item in closure]}, "capabilities": capabilities,
                "runtime": {"executor": "rubiksql-snapshot-importer"}, "source": source, "provenance": metadata, "trust": {},
                "fingerprints": {"source_hash": source["source_hash"], "signature_hash": _native_digest(signature),
                                 "restore_hash": _native_digest([])}, "created_at": 0.0, "updated_at": 0.0}
    return {"object_id": manifest["id"], **{key: deepcopy(manifest[key]) for key in (
        "name", "namespace", "version", "status", "description", "tags", "capabilities", "fingerprints")},
        "manifest": manifest, "created_at": 0.0, "updated_at": 0.0, "last_reason": "answering tool publication"}


def _check_function_capsule(row: Mapping[str, Any]) -> None:
    from .tool_definitions import validate_definitions

    manifest = _mapping(row.get("manifest"), "Tool Capsule manifest")
    metadata = _mapping(manifest.get("provenance"), "Tool Capsule provenance")
    definition = validate_definitions([_mapping(metadata.get("tool"), "Tool Capsule definition")])[0]
    source = _mapping(manifest.get("source"), "Tool Capsule source reference")
    closure = metadata.get("closure")
    if (not isinstance(closure, list) or not closure or any(not isinstance(item, Mapping)
            or set(item) != {"module", "sha256"} or not isinstance(item.get("module"), str)
            or not isinstance(item.get("sha256"), str) or not _DIGEST.fullmatch(item["sha256"]) for item in closure)
            or [item["module"] for item in closure] != sorted({item["module"] for item in closure})
            or source.get("module") != definition["module"] or source.get("qualname") != definition["qualname"]
            or not isinstance(source.get("capsule_id"), str)
            or not isinstance(source.get("source_hash"), str) or not _DIGEST.fullmatch(source["source_hash"])
            or {"module": source["module"], "sha256": source["source_hash"]} not in closure):
        raise ValueError("Tool Capsule has invalid source reference or module closure")
    signature = {"input_schema": definition["input_schema"], "output_schema": definition["output_schema"]}
    version = digest({"source": dict(source), **dict(metadata)})
    if (manifest.get("version") != version or manifest.get("name") != definition["name"]
            or manifest.get("kind") != "python-function" or manifest.get("status") != "active"
            or manifest.get("identity") != {"module": definition["module"], "qualname": definition["qualname"]}
            or manifest.get("description") != definition["description"] or manifest.get("signature") != signature
            or manifest.get("input_schema") != definition["input_schema"] or manifest.get("output_schema") != definition["output_schema"]
            or manifest.get("restore_layers") != [] or manifest.get("dependencies") != {"modules": [item["module"] for item in closure]}
            or manifest.get("fingerprints") != {"source_hash": source["source_hash"], "signature_hash": _native_digest(signature),
                                               "restore_hash": _native_digest([])}
            or manifest.get("capabilities", {}).get("mcp_tool", {}).get("ok") is not False
            or manifest.get("id") != _native_id("capsule", TOOL_NAMESPACE, definition["name"], version)
            or row.get("object_id") != manifest.get("id") or any(row.get(key) != manifest.get(key) for key in (
                "name", "namespace", "version", "status", "description", "tags", "capabilities", "fingerprints"))):
        raise ValueError("Tool Capsule has inconsistent callable identity or mirrored native fields")


def _callable_closure(module_sources: Mapping[str, str], definitions: Sequence[Mapping[str, Any]],
                      serializer_source: str) -> tuple[tuple[dict, ...], dict[str, dict], dict, dict[str, list[dict]]]:
    """Validate authoring input and derive module Capsules and each callable's source closure.

    Returns:
        ``(definitions in serve order, module Capsules by module, serializer Capsule,
        tool name -> sorted [{module, sha256}] closure)``.
    """
    from .tool_definitions import validate_definitions
    from .tool_receipts import scan_tool_texts

    module_sources = _mapping(module_sources, "answering module sources")
    if (module_sources.get("rubiksql_hb.tools") != ""
            or any(not isinstance(module, str) or module != "rubiksql_hb.tools" and not _TOOL_MODULE.fullmatch(module)
                   for module in module_sources)
            or any(not isinstance(source, str) for source in module_sources.values()) or not isinstance(serializer_source, str)):
        raise ValueError("answering bundle requires tools source modules and empty package init")
    try:
        trees = {module: ast.parse(source, filename=module) for module, source in module_sources.items()}
    except (SyntaxError, TypeError, ValueError) as error:
        raise ValueError("bundle module has invalid Python source") from error
    records = validate_definitions(definitions, allowed_modules=module_sources)
    if not records:
        raise ValueError("answering bundle needs at least one callable definition")
    scan_tool_texts([*module_sources.values(), serializer_source, canonical(records)])
    implementations = {}
    for record in records:
        matching = [node for node in trees[record["module"]].body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == record["qualname"]]
        if len(matching) != 1:
            raise ValueError(f"tool {record['name']!r} needs exactly one defining callable at its locator")
        implementations[record["name"]] = record["module"]
    reexports = dict(_SERVER_REEXPORTS)
    for name, owner in _SPLIT_SERVER_REEXPORTS.items():
        if owner in implementations:
            reexports[name] = implementations[owner]
    reexports.update(implementations)
    graph = {module: _imports(module, source, set(module_sources), server_reexports=reexports)
             for module, source in module_sources.items()}
    _check_role_imports(records, module_sources, graph)
    modules = {module: _capsule(module, source, sorted(graph[module]), module_format=MODULE_FORMAT)
               for module, source in module_sources.items()}
    serializer = _capsule("heavenbase.toolkit.serializers", serializer_source, (), serializer=True)
    closures = {record["name"]: [{"module": module, "sha256": source_digest(module_sources[module])}
                                 for module in _closure(record["module"], graph)] for record in records}
    return tuple(dict(record) for record in records), modules, serializer, closures


def build_answering_rows(module_sources: Mapping[str, str], *, definitions: Sequence[Mapping[str, Any]],
                         serializer_source: str) -> AnsweringRows:
    """Build one native Capsule per callable, without executing authoring source.

    This is the historical ``rubiksql.answering-bundle/2`` representation; new-format releases
    use :func:`build_tool_rows`.

    Args:
        module_sources (Mapping[str, str]): Exact allowed module-source closure.
        definitions (Sequence[Mapping[str, Any]]): Complete tool metadata. No
            Toolkit membership is persisted.
        serializer_source (str): Exact repository-owned serializer source.

    Returns:
        AnsweringRows: Native source, serializer and function Capsules, without
            persisted Toolkits. Callable metadata controls consumer selection.

    Raises:
        ValueError: Source, locator, contracts or permissions are inconsistent.
    """
    records, modules, serializer, closures = _callable_closure(module_sources, definitions, serializer_source)
    tools = [_function_capsule(record, modules, closures[record["name"]], serializer) for record in records]
    payload = {"schema": BUNDLE_SCHEMA,
               "modules": [{"module": module, "capsule_id": row["object_id"], "sha256": row["manifest"]["source"]["source_hash"],
                            "row_digest": row_digest(row, capsule=True)} for module, row in sorted(modules.items())],
               "serializer": {"capsule_id": serializer["object_id"],
                              "source_digest": source_digest(_normalized_source(serializer_source)),
                              "row_digest": row_digest(serializer, capsule=True)},
               "tools": [{"name": row["name"], "capsule_id": row["object_id"], "version": row["version"],
                          "row_digest": row_digest(row, capsule=True)} for row in tools]}
    return AnsweringRows(deepcopy(payload), tuple([*modules.values(), serializer, *tools]), ())


def answering_definitions(payload: Mapping[str, Any], capsule_rows: Mapping[str, Mapping[str, Any]], *,
                          toolkit_rows: Mapping[str, Mapping[str, Any]] | None = None) -> tuple[dict, ...]:
    """Read ordered metadata from verified current or immutable legacy rows.

    Args:
        payload (Mapping[str, Any]): Native answering reference payload.
        capsule_rows (Mapping[str, Mapping[str, Any]]): Rows indexed by native id.
        toolkit_rows (Mapping[str, Mapping[str, Any]] | None): Required only for
            schema1 historical selections; current bundles have no Toolkit rows.

    Returns:
        tuple[dict, ...]: Detached complete callable definitions in serve order.

    Raises:
        ValueError: Payload schema, definition shape or references are malformed.
    """
    from .tool_definitions import validate_definitions

    try:
        if payload.get("schema") == BUNDLE_SCHEMA:
            records = [capsule_rows[ref["capsule_id"]]["manifest"]["provenance"]["tool"] for ref in payload["tools"]]
            normalized = validate_definitions(records, allowed_modules=[ref["module"] for ref in payload["modules"]])
            if [record["name"] for record in normalized] != [ref["name"] for ref in payload["tools"]]:
                raise ValueError("tool reference order differs from callable metadata")
            return normalized
        if payload.get("schema") == LEGACY_BUNDLE_SCHEMA:
            raise ValueError("schema1 Toolkit execution is permanently retired")
        raise ValueError("invalid answering payload schema")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("answering definitions have missing or malformed references") from error


def validate_answering(payload: Mapping[str, Any], *, capsule_rows: Mapping[str, Mapping[str, Any]],
                       toolkit_rows: Mapping[str, Mapping[str, Any]] | None = None) -> None:
    """Verify immutable native source and callable rows without executing them.

    Args:
        payload (Mapping[str, Any]): Current or historical answering payload.
        capsule_rows (Mapping[str, Mapping[str, Any]]): Native Capsule id map.
        toolkit_rows (Mapping[str, Mapping[str, Any]] | None): Historical Toolkit
            id map; current schema derives membership from callable metadata.

    Returns:
        None: Exact references, source closures and callable identities match.

    Raises:
        ValueError: Native rows, permissions or source closure are inconsistent.
    """
    if payload.get("schema") == LEGACY_BUNDLE_SCHEMA:
        raise ValueError("schema1 Toolkit execution is permanently retired")
    if payload.get("schema") != BUNDLE_SCHEMA or set(payload) != {"schema", "modules", "serializer", "tools"}:
        raise ValueError("invalid answering payload schema")
    try:
        sources = {}
        for reference in payload["modules"]:
            row = capsule_rows[reference["capsule_id"]]
            if (row_digest(row, capsule=True) != reference["row_digest"] or row["manifest"]["source"]["source_hash"] != reference["sha256"]
                    or row["namespace"] != MODULE_NAMESPACE or row["name"] != reference["module"] or reference["module"] in sources):
                raise ValueError("module capsule reference differs")
            sources[reference["module"]] = row["manifest"]["source"]["source"]
        serializer_ref = payload["serializer"]
        serializer = capsule_rows[serializer_ref["capsule_id"]]
        serializer_source = serializer["manifest"]["source"]["source"]
        if (row_digest(serializer, capsule=True) != serializer_ref["row_digest"] or serializer["namespace"] != NAMESPACE
                or serializer["name"] != "__jstr_serializer__"
                or source_digest(_normalized_source(serializer_source)) != serializer_ref["source_digest"]):
            raise ValueError("serializer reference differs")
        for reference in payload["tools"]:
            row = capsule_rows[reference["capsule_id"]]
            if (row_digest(row, capsule=True) != reference["row_digest"] or row["namespace"] != TOOL_NAMESPACE
                    or row["name"] != reference["name"] or row["version"] != reference["version"]):
                raise ValueError("tool capsule reference differs")
        definitions = answering_definitions(payload, capsule_rows)
        rebuilt = build_answering_rows(sources, definitions=definitions, serializer_source=serializer_source)
        if canonical(rebuilt.payload) != canonical(payload):
            raise ValueError("answering references do not reproduce their source/definition closure")
        for row in rebuilt.capsule_rows:
            if _semantic(row) != _semantic(capsule_rows[row["object_id"]]):
                raise ValueError("answering capsule semantics differ")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("answering bundle has missing or malformed references") from error


# ---------------------------------------------------------------------------------------
# Canonical Tool versions (``tool-entity``) and the typed answering bundle


_IMPLEMENTATION_KEYS = frozenset({"capsule_id", "source_hash", "closure", "serializer_id", "serializer_digest", "knob_defaults"})
_TOOL_COLUMNS = ("name", "description", "input_schema", "output_schema", "module", "qualname", "visible_to",
                 "loop_visible_to", "routes", "required_for")


@dataclass(frozen=True)
class TypedTools:
    """Detached typed tool publication input; no workspace or callable is opened.

    Attributes:
        content: ``{"tools": [typed refs in serve order], "loader": {"modules", "serializer"}}``.
        capsule_rows: Inert module source Capsules and the serializer Capsule.
        tool_rows: Canonical ``tool-entity`` rows in serve order.
        definitions: Validated definitions in serve order.
    """

    content: Mapping[str, Any]
    capsule_rows: tuple[Mapping[str, Any], ...]
    tool_rows: tuple[Mapping[str, Any], ...]
    definitions: tuple[Mapping[str, Any], ...]


def tool_definition(row: Mapping[str, Any]) -> dict:
    """Return the validated callable definition a Tool row stores.

    Args:
        row (Mapping[str, Any]): ``tool-entity`` row.

    Returns:
        dict: The eleven-field definition (``order`` from ``serve_order``).

    Raises:
        ValueError: Missing or invalid definition fields.
    """
    from .tool_definitions import validate_definitions

    try:
        record = {**{key: deepcopy(row[key]) for key in _TOOL_COLUMNS}, "order": row["serve_order"]}
    except (KeyError, TypeError) as error:
        raise ValueError("Tool row lacks a definition field") from error
    record["visible_to"], record["loop_visible_to"] = list(record["visible_to"]), list(record["loop_visible_to"])
    record["required_for"] = list(record["required_for"])
    return dict(validate_definitions([record])[0])


def implementation_version(definition: Mapping[str, Any], implementation: Mapping[str, Any]) -> str:
    """Return a tool's executable identity, equal to its schema2 callable Capsule version.

    Role implementation digests and answering cache keys hash this value, so an unchanged
    callable keeps its identity across the representation change.

    Args:
        definition (Mapping[str, Any]): Validated definition.
        implementation (Mapping[str, Any]): Exact implementation pin.

    Returns:
        str: 64-hex digest.
    """
    source = {"module": definition["module"], "qualname": definition["qualname"],
              "capsule_id": implementation["capsule_id"], "source_hash": implementation["source_hash"]}
    return digest({"source": source, "tool": dict(definition), "closure": list(implementation["closure"]),
                   "serializer_id": implementation["serializer_id"], "serializer_digest": implementation["serializer_digest"],
                   "knob_defaults": dict(implementation["knob_defaults"])})


def _check_implementation(value: Any) -> dict:
    implementation = dict(_mapping(value, "Tool implementation"))
    closure = implementation.get("closure")
    if (set(implementation) != _IMPLEMENTATION_KEYS or not isinstance(implementation["capsule_id"], str)
            or not isinstance(implementation["source_hash"], str) or not _DIGEST.fullmatch(implementation["source_hash"])
            or not isinstance(implementation["serializer_id"], str)
            or not isinstance(implementation["serializer_digest"], str) or not _DIGEST.fullmatch(implementation["serializer_digest"])
            or not isinstance(implementation["knob_defaults"], Mapping)
            or not isinstance(closure, list) or not closure
            or any(not isinstance(item, Mapping) or set(item) != {"module", "sha256"} or not isinstance(item["module"], str)
                   or not isinstance(item["sha256"], str) or not _DIGEST.fullmatch(item["sha256"]) for item in closure)
            or [item["module"] for item in closure] != sorted({item["module"] for item in closure})):
        raise ValueError("Tool implementation pin is malformed")
    return deepcopy(implementation)


def tool_row(definition: Mapping[str, Any], implementation: Mapping[str, Any]) -> dict:
    """Build one canonical Tool version.

    Args:
        definition (Mapping[str, Any]): Complete callable definition.
        implementation (Mapping[str, Any]): Exact pin of its module Capsule, closure and serializer.

    Returns:
        dict: A complete ``tool-entity`` row; ``object_id`` is ``t`` + its 16-hex version.
    """
    from .tool_definitions import validate_definitions
    from .prompts.entity_model import FORMAT_TOOL

    record = dict(validate_definitions([definition])[0])
    pin = _check_implementation(implementation)
    if {"module": record["module"], "sha256": pin["source_hash"]} not in pin["closure"]:
        raise ValueError("Tool implementation closure lacks its defining module")
    value = digest({"format": FORMAT_TOOL, "definition": record, "implementation": pin})
    return {"object_id": "t" + value[:16], "format": FORMAT_TOOL, "version": value[:16], "digest": value,
            **{key: deepcopy(record[key]) for key in _TOOL_COLUMNS}, "serve_order": record["order"],
            "implementation": pin}


def verify_tool_row(row: Mapping[str, Any]) -> dict:
    """Verify one canonical Tool version and return its definition.

    Args:
        row (Mapping[str, Any]): Stored or candidate ``tool-entity`` row.

    Returns:
        dict: The validated definition.

    Raises:
        ValueError: Format, definition, implementation pin or content identity differs.
    """
    from .prompts.entity_model import FORMAT_TOOL

    if row.get("format") != FORMAT_TOOL:
        raise ValueError(f"tool {row.get('object_id')!r} is not a {FORMAT_TOOL} version")
    definition = tool_definition(row)
    rebuilt = tool_row(definition, row.get("implementation"))
    if any(rebuilt[key] != row.get(key) for key in ("object_id", "version", "digest")):
        raise ValueError(f"tool {row.get('object_id')!r} content identity differs")
    return definition


def build_tool_rows(module_sources: Mapping[str, str], *, definitions: Sequence[Mapping[str, Any]],
                    serializer_source: str) -> TypedTools:
    """Build canonical Tool versions and inert source Capsules without executing authoring source.

    The module and serializer Capsules are byte-identical to the schema2 builder's, so unchanged
    source reuses existing Capsule rows; no callable Capsule or Toolkit is created.

    Args:
        module_sources (Mapping[str, str]): Exact allowed module-source closure.
        definitions (Sequence[Mapping[str, Any]]): Complete tool metadata.
        serializer_source (str): Exact repository-owned serializer source.

    Returns:
        TypedTools: Typed release content, source Capsules and Tool rows.

    Raises:
        ValueError: Source, locator, definitions or permissions are inconsistent.
    """
    records, modules, serializer, closures = _callable_closure(module_sources, definitions, serializer_source)
    serializer_digest = source_digest(_normalized_source(serializer_source))
    rows = []
    for record in records:
        module_row = modules[record["module"]]
        rows.append(tool_row(record, {
            "capsule_id": module_row["object_id"], "source_hash": module_row["manifest"]["source"]["source_hash"],
            "closure": closures[record["name"]], "serializer_id": serializer["object_id"],
            "serializer_digest": serializer_digest,
            "knob_defaults": {parameter: knob for (tool, parameter), knob in _KNOB_DEFAULTS.items() if tool == record["name"]}}))
    from .prompts.entity_model import TOOL_ENTITY, typed_ref

    loader = {"modules": [{"module": module, "capsule_id": row["object_id"], "sha256": row["manifest"]["source"]["source_hash"],
                           "row_digest": row_digest(row, capsule=True)} for module, row in sorted(modules.items())],
              "serializer": {"capsule_id": serializer["object_id"], "source_digest": serializer_digest,
                             "row_digest": row_digest(serializer, capsule=True)}}
    content = {"tools": [typed_ref(TOOL_ENTITY, row) for row in rows], "loader": loader}
    return TypedTools(deepcopy(content), tuple([*modules.values(), serializer]), tuple(rows), records)


def typed_bundle(content_refs: Mapping[str, Any]) -> dict:
    """Return the executable answering bundle document of a closure/2 release.

    Receipts, snapshot manifests and launch checks bind :func:`digest` of this document.

    Args:
        content_refs (Mapping[str, Any]): The release's typed block (``tools`` and ``loader`` are read).

    Returns:
        dict: ``{"schema": "rubiksql.answering-bundle/3", "modules", "serializer", "tools"}``.
    """
    return {"schema": TYPED_BUNDLE_SCHEMA, "modules": deepcopy(list(content_refs["loader"]["modules"])),
            "serializer": deepcopy(dict(content_refs["loader"]["serializer"])), "tools": deepcopy(list(content_refs["tools"]))}


def validate_typed_tools(content_refs: Mapping[str, Any], *, tool_rows: Mapping[str, Mapping[str, Any]],
                         capsule_rows: Mapping[str, Mapping[str, Any]]) -> tuple[dict, ...]:
    """Verify a closure/2 release's Tool versions and source pins by deterministic rebuild.

    Args:
        content_refs (Mapping[str, Any]): Typed block with ``tools`` and ``loader``.
        tool_rows (Mapping[str, Mapping[str, Any]]): Tool rows by object id.
        capsule_rows (Mapping[str, Mapping[str, Any]]): Native Capsule rows by id.

    Returns:
        tuple[dict, ...]: Validated definitions in serve order.

    Raises:
        ValueError: A reference, row, source pin, closure or Capsule differs.
    """
    from .prompts.entity_model import TOOL_ENTITY, check_ref, typed_ref

    try:
        loader = content_refs["loader"]
        sources = {}
        for reference in loader["modules"]:
            row = capsule_rows[reference["capsule_id"]]
            if (row_digest(row, capsule=True) != reference["row_digest"] or row["manifest"]["source"]["source_hash"] != reference["sha256"]
                    or row["namespace"] != MODULE_NAMESPACE or row["name"] != reference["module"] or reference["module"] in sources):
                raise ValueError("module capsule reference differs")
            sources[reference["module"]] = row["manifest"]["source"]["source"]
        serializer = capsule_rows[loader["serializer"]["capsule_id"]]
        serializer_source = serializer["manifest"]["source"]["source"]
        if (row_digest(serializer, capsule=True) != loader["serializer"]["row_digest"] or serializer["namespace"] != NAMESPACE
                or serializer["name"] != "__jstr_serializer__"
                or source_digest(_normalized_source(serializer_source)) != loader["serializer"]["source_digest"]):
            raise ValueError("serializer reference differs")
        definitions = []
        for reference in content_refs["tools"]:
            expected = check_ref(reference, TOOL_ENTITY)
            row = tool_rows[expected["object_id"]]
            definitions.append(verify_tool_row(row))
            if typed_ref(TOOL_ENTITY, row) != expected:
                raise ValueError("tool reference differs from its stored row")
        rebuilt = build_tool_rows(sources, definitions=definitions, serializer_source=serializer_source)
        if canonical(rebuilt.content) != canonical({"tools": list(content_refs["tools"]), "loader": loader}):
            raise ValueError("typed tools do not reproduce their source/definition closure or serve order")
        for row in rebuilt.capsule_rows:
            if _semantic(row) != _semantic(capsule_rows[row["object_id"]]):
                raise ValueError("typed tool source capsule semantics differ")
        return rebuilt.definitions
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("typed tools have missing or malformed references") from error


def convert_answering(payload: Mapping[str, Any], capsule_rows: Mapping[str, Mapping[str, Any]]) -> TypedTools:
    """Convert a pinned schema2 answering bundle into equivalent canonical Tool versions.

    The conversion reads the release's actual pinned Capsules, never Git defaults. It proves the
    module and serializer pins are identical and every Tool's executable identity equals the
    callable Capsule version it replaces.

    Args:
        payload (Mapping[str, Any]): Decoded ``tools.answering`` schema2 payload.
        capsule_rows (Mapping[str, Mapping[str, Any]]): Native Capsule rows by id.

    Returns:
        TypedTools: Equivalent typed content; its source Capsules already exist.

    Raises:
        ValueError: The bundle is not schema2, fails validation or is not representation-equivalent.
    """
    if payload.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("entity-model conversion requires a rubiksql.answering-bundle/2 pin")
    validate_answering(payload, capsule_rows=capsule_rows)
    definitions = answering_definitions(payload, capsule_rows)
    sources = {ref["module"]: capsule_rows[ref["capsule_id"]]["manifest"]["source"]["source"] for ref in payload["modules"]}
    serializer_source = capsule_rows[payload["serializer"]["capsule_id"]]["manifest"]["source"]["source"]
    typed = build_tool_rows(sources, definitions=definitions, serializer_source=serializer_source)
    if canonical(typed.content["loader"]) != canonical({"modules": payload["modules"], "serializer": payload["serializer"]}):
        raise ValueError("typed tool loader differs from the pinned module and serializer Capsules")
    legacy = {ref["name"]: ref["version"] for ref in payload["tools"]}
    if [row["name"] for row in typed.tool_rows] != [ref["name"] for ref in payload["tools"]]:
        raise ValueError("typed tool order differs from the pinned bundle")
    for row, definition in zip(typed.tool_rows, typed.definitions):
        if implementation_version(definition, row["implementation"]) != legacy[row["name"]]:
            raise ValueError(f"tool {row['name']} executable identity differs from its pinned callable Capsule")
    return typed


def typed_snapshot_files(content_refs: Mapping[str, Any], capsule_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, bytes]:
    """Materialize a closure/2 release's inert execution closure for a snapshot.

    Args:
        content_refs (Mapping[str, Any]): The release's typed block.
        capsule_rows (Mapping[str, Mapping[str, Any]]): Native Capsule rows by id.

    Returns:
        dict[str, bytes]: ``tools/capsules/<id>.json`` for module and serializer Capsules and
            ``tools/modules/...`` source files; Tool rows travel in ``native/tool-entity.json``.
    """
    loader = content_refs["loader"]
    files = {}
    for ref in [*loader["modules"], loader["serializer"]]:
        row = capsule_rows[ref["capsule_id"]]
        files[f"tools/capsules/{row['object_id']}.json"] = canonical(row).encode("utf-8")
    for ref in loader["modules"]:
        module = ref["module"]
        path = module.replace(".", "/") + ("/__init__.py" if module == "rubiksql_hb.tools" else ".py")
        files["tools/modules/" + path] = capsule_rows[ref["capsule_id"]]["manifest"]["source"]["source"].encode("utf-8")
    return files


def _assert_identical(existing: Mapping[str, Any] | None, row: Mapping[str, Any], *, capsule: bool) -> None:
    row_digest(row, capsule=capsule)
    if existing is not None and row_digest(existing, capsule=capsule) != row_digest(row, capsule=capsule):
        raise ValueError("native answering id already exists with different semantic content")


def validate_tool_receipts(payload: Mapping[str, Any], gate: Mapping[str, Any], *,
                           parent_bundle: str | None, launch: bool,
                           tiers: Mapping[str, str] | None = None) -> None:
    """Validate publication/launch receipts through the owning tool policy module.

    Args:
        payload (Mapping[str, Any]): Exact candidate reference payload.
        gate (Mapping[str, Any]): Immutable gate event payload.
        parent_bundle (str | None): Parent digest, null for R1.
        launch (bool): True for launch; checks remain identical at publication.
        tiers (Mapping[str, str] | None): Inherited host-verified module tiers.

    Returns:
        None.
    """
    from .tool_receipts import validate_tool_receipts as validate

    validate(payload, gate, parent_bundle=parent_bundle, launch=launch, tiers=tiers)


class _WriteOnceStore:
    """Create-only adapter for explicit native row stores, never hidden registries."""

    def __init__(self, store: Any, *, capsule: bool) -> None:
        self.store, self.capsule = store, capsule
        self.held = {row["object_id"]: row for row in store.rows()}
        self.written: set[str] = set()

    def get(self, object_id: str) -> dict | None:
        return self.held.get(object_id)

    def rows(self) -> list[dict]:
        return list(self.held.values())

    def upsert(self, row: dict) -> None:
        existing = self.get(row["object_id"])
        _assert_identical(existing, row, capsule=self.capsule)
        if existing is None:
            self.store.upsert(row)
            self.held[row["object_id"]] = deepcopy(row)
            self.written.add(row["object_id"])

    def set(self, row: dict) -> None:
        raise ValueError("native answering rows cannot be replaced")

    def delete(self, object_id: str) -> bool:
        raise ValueError("native answering rows cannot be deleted")


def _normalized_source(source: str) -> str:
    if not isinstance(source, str):
        raise ValueError("Answering Capsule has invalid captured Python source")
    source = textwrap.dedent(source)
    try:
        tree = ast.parse(source)
    except (SyntaxError, TypeError) as exc:
        raise ValueError("Answering Capsule has invalid captured Python source") from exc
    lines = source.splitlines(keepends=True)

    def position(row: int, column: int) -> tuple[int, int]:
        # AST columns count UTF-8 bytes; tokenize columns count characters.
        return row, len(lines[row - 1].encode("utf-8")[:column].decode("utf-8"))

    omitted = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) or not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            omitted.append((position(first.lineno, first.col_offset),
                            position(first.end_lineno, first.end_col_offset)))
    # HB capture removes decorators from the one captured function. AST is used
    # only for source spans; no interpreter-dependent AST fields enter identity.
    start = (1, 0)
    if len(tree.body) == 1 and isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        start = position(tree.body[0].lineno, tree.body[0].col_offset)
    try:
        tokens = list(tokenize.generate_tokens(StringIO(source).readline))
    except (SyntaxError, tokenize.TokenError) as exc:
        raise ValueError("Answering Capsule has invalid captured Python source") from exc
    normalized = []
    index = 0
    while index < len(tokens):
        item = tokens[index]
        index += 1
        kind = tokenize.tok_name[item.type]
        if item.start < start or kind in {"ENCODING", "COMMENT", "NL", "ENDMARKER"}:
            continue
        if kind in {"INDENT", "DEDENT"}:
            normalized.append((kind, ""))
            continue
        if any(begin <= item.start < end for begin, end in omitted):
            continue
        follows_docstring = any(item.start[0] == end[0] and item.start >= end
                                and not lines[end[0] - 1][end[1]:item.start[1]].strip()
                                for _, end in omitted)
        if follows_docstring and (kind == "NEWLINE" or item.string == ";"):
            continue
        if kind == "NEWLINE":
            # Suite boundaries matter; indentation width and line endings do not.
            normalized.append((kind, ""))
            continue
        if kind == "FSTRING_START":
            # Python 3.12 splits f-strings, while 3.11 emits one STRING token.
            # Join the exact source span so both interpreters record the same
            # literal without making their numeric token ids part of identity.
            depth = 1
            end = item.end
            while depth and index < len(tokens):
                part = tokens[index]
                index += 1
                label = tokenize.tok_name[part.type]
                depth += (label == "FSTRING_START") - (label == "FSTRING_END")
                end = part.end
            if depth:
                raise ValueError("Answering Capsule has invalid captured Python source")
            text = "".join(lines[item.start[0] - 1:end[0]])
            offset = sum(len(line) for line in lines[item.start[0] - 1:end[0] - 1]) + end[1]
            normalized.append(("token", text[item.start[1]:offset]))
            continue
        normalized.append(("token", item.string))
    return canonical(normalized)
