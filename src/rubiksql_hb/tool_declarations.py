"""Process-local bindings for the ordered callable definitions in the native pin."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import wraps
import importlib
import inspect
from typing import Any, TYPE_CHECKING

from .tool_bundle import deployment_bundle, process_bundle

if TYPE_CHECKING:
    from .tool_bundle import SnapshotBundle
    from .tool_context import ToolExecutionContext


def project_tool_functions(*, role: str | None = None, selection: Sequence[Mapping[str, Any]] | None = None,
                           context: ToolExecutionContext | None = None,
                           bundle: SnapshotBundle | None = None) -> tuple[Callable[..., Any], ...]:
    """Select pinned callables before imports and bind independent read capabilities.

    Args:
        role: Explicit consumer, or null for the combined answering MCP surface.
        selection: Optional exact ordered subset of the permitted definitions.
        context: Trusted assistant/META read capabilities; never model input.
        bundle: Verified installed pin; null selects the process's matching pin.

    Returns:
        tuple[Callable[..., Any], ...]: Functions loaded by the verified bundle
            importer. Native row metadata owns their selection and identity.

    Raises:
        ImportError: A pinned locator does not resolve to its declared function.
    """
    from .hb_tool_entities import answering_selection, canonical
    from .tool_definitions import ANSWERING_ROLES, select_definitions

    if role is not None and role not in {*ANSWERING_ROLES, "assistant", "meta"}:
        raise ValueError("unknown tool consumer")
    independent = role in {"assistant", "meta"}
    installed = deployment_bundle(role) if independent else process_bundle()
    if bundle is not None and bundle is not installed:
        raise ValueError("tool functions require the installed fixed bundle")
    bundle = installed
    if independent:
        if context is None or context.role != role:
            raise ValueError("independent tools require their matching trusted context")
        from .tool_context import bind_tool_context

        with bind_tool_context(context):
            pass
        permitted = select_definitions(bundle.definitions, role)
    else:
        if context is not None:
            raise ValueError("answering tools cannot bind read consumer contexts")
        permitted = answering_selection(bundle.definitions) if role is None else select_definitions(bundle.definitions, role)
    records = permitted if selection is None else tuple(selection)
    selected = {record["name"] for record in records}
    expected = tuple(record for record in permitted if record["name"] in selected)
    if canonical(records) != canonical(expected):
        raise ValueError("tool selection is unauthorized, altered or out of serve order")
    functions = []
    for record in records:
        function = importlib.import_module(record["module"])
        for part in record["qualname"].split("."):
            function = getattr(function, part)
        if not callable(function) or function.__name__ != record["name"]:
            raise ImportError(f"pinned callable {record['name']!r} has an invalid locator")
        if independent:
            functions.append(_bind_function(function, context))
        else:
            functions.append(function)
    return tuple(functions)


def _bind_function(function: Callable[..., Any], context: ToolExecutionContext) -> Callable[..., Any]:
    from .tool_context import bind_tool_context

    if inspect.iscoroutinefunction(function):
        @wraps(function)
        async def bound(*args: Any, **kwargs: Any) -> Any:
            with bind_tool_context(context):
                return await function(*args, **kwargs)
    else:
        @wraps(function)
        def bound(*args: Any, **kwargs: Any) -> Any:
            with bind_tool_context(context):
                return function(*args, **kwargs)
    return bound


def __getattr__(name: str) -> Callable[..., Any]:
    """Resolve a named pinned function for direct host consumers.

    Args:
        name (str): Published callable name.

    Returns:
        Callable[..., Any]: The function from the fixed verified pin.

    Raises:
        AttributeError: No declared callable has this name.
    """
    from .hb_tool_entities import answering_selection

    if name.startswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    for record in answering_selection(process_bundle().definitions):
        if record["name"] == name:
            function = getattr(importlib.import_module(record["module"]), record["qualname"])
            if not callable(function) or function.__name__ != name:
                raise ImportError(f"pinned callable {name!r} has an invalid locator")
            return function
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
