"""Inert callable metadata shared by native publication and answering consumers.

Definitions are JSON values, never Python callables or native restore recipes.
`visible_to` permits model calls; `loop_visible_to` permits host loop calls.
An omitted role in `routes` permits every route for that role.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
import keyword
import re
from typing import Any, Literal, TypedDict, cast

ANSWERING_ROLES = ("candidate", "judge", "reviewer")
ROLES = (*ANSWERING_ROLES, "assistant", "meta")
Calling = Literal["model", "loop"]
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class ToolDefinition(TypedDict):
    """Complete stable identity and permission metadata for one callable."""

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    module: str
    qualname: str
    order: int
    visible_to: list[str]
    loop_visible_to: list[str]
    routes: dict[str, list[str]]
    required_for: list[str]


def _roles(value: Any, field: str) -> list[str]:
    if (not isinstance(value, list) or any(not isinstance(role, str) or role not in ROLES for role in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"tool {field} must contain unique consumer roles")
    return [role for role in ROLES if role in value]


def validate_definitions(records: Sequence[Mapping[str, Any]], *,
                         allowed_modules: Iterable[str] | None = None) -> tuple[ToolDefinition, ...]:
    """Normalize complete per-callable definitions without loading their source.

    Args:
        records (Sequence[Mapping[str, Any]]): Name, contract, locator, order,
            model/loop roles, optional routes and required roles. The allowed
            source locator enforces the host implementation boundary.
        allowed_modules (Iterable[str] | None): Permitted implementation modules.
            Null validates metadata alone, for already verified row consumers.

    Returns:
        tuple[ToolDefinition, ...]: Detached definitions in unique numeric order.

    Raises:
        ValueError: Metadata, locator or permission fields are ambiguous.
    """
    modules = None if allowed_modules is None else frozenset(allowed_modules)
    if not isinstance(records, Sequence):
        raise ValueError("tool definitions must be a sequence")
    normalized: list[ToolDefinition] = []
    names: set[str] = set()
    orders: set[int] = set()
    fields = set(ToolDefinition.__annotations__)
    required = fields - {"routes", "required_for"}
    for value in records:
        if not isinstance(value, Mapping) or set(value) - fields or required - value.keys():
            raise ValueError("tool definition has missing or unknown fields")
        record = deepcopy(dict(value))
        name, module, qualname, order = (record.get(key) for key in ("name", "module", "qualname", "order"))
        if (not isinstance(name, str) or not _NAME.fullmatch(name) or keyword.iskeyword(name)
                or not isinstance(module, str) or not module.startswith("rubiksql_hb.tools.")
                or modules is not None and module not in modules or qualname != name
                or not isinstance(order, int) or isinstance(order, bool) or order < 0
                or name in names or order in orders):
            raise ValueError("tool definition needs a unique name/order and an allowed top-level callable locator")
        if (not isinstance(record.get("description"), str) or not isinstance(record.get("input_schema"), Mapping)
                or record.get("output_schema") is not None and not isinstance(record["output_schema"], Mapping)):
            raise ValueError("tool definition has invalid description or schemas")
        record["input_schema"] = deepcopy(dict(record["input_schema"]))
        if record["output_schema"] is not None:
            record["output_schema"] = deepcopy(dict(record["output_schema"]))
        for field in ("visible_to", "loop_visible_to", "required_for"):
            record[field] = _roles(record.get(field, []), field)
        if not set(record["required_for"]).issubset(record["visible_to"]):
            raise ValueError("tool required_for must be a subset of model visibility")
        routes = record.get("routes", {})
        if not isinstance(routes, Mapping):
            raise ValueError("tool routes must be a role-to-route mapping")
        permitted_roles = set(record["visible_to"]) | set(record["loop_visible_to"])
        for role, routes_for_role in routes.items():
            if (role not in permitted_roles or not isinstance(routes_for_role, list) or not routes_for_role
                    or any(not isinstance(route, str) or not _NAME.fullmatch(route) for route in routes_for_role)
                    or len(set(routes_for_role)) != len(routes_for_role)):
                raise ValueError("tool routes must contain unique routes for visible roles")
        record["routes"] = {role: sorted(routes[role]) for role in ROLES if role in routes}
        names.add(name)
        orders.add(order)
        normalized.append(cast(ToolDefinition, record))
    return tuple(sorted(normalized, key=lambda record: record["order"]))


def select_definitions(records: Sequence[Mapping[str, Any]], role: str, route: str = "default", *,
                       calling: Calling = "model") -> tuple[ToolDefinition, ...]:
    """Select ordered callables permitted for a consumer role and route.

    Args:
        records (Sequence[Mapping[str, Any]]): Verified per-callable definitions.
        role (str): candidate, judge, reviewer, assistant or meta.
        route (str): Exact route identity; default for ordinary contexts.
        calling (Calling): model checks visible_to; loop checks loop_visible_to.

    Returns:
        tuple[ToolDefinition, ...]: Detached permitted definitions in serve order.

    Raises:
        ValueError: Role, route, calling mode or definition shape is invalid.
    """
    if role not in ROLES or not isinstance(route, str) or not _NAME.fullmatch(route) or calling not in {"model", "loop"}:
        raise ValueError("invalid consumer role, route or calling mode")
    field = "visible_to" if calling == "model" else "loop_visible_to"
    return tuple(record for record in validate_definitions(records)
                 if role in record[field] and (role not in record["routes"] or route in record["routes"][role]))
