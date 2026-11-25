"""Route registry — one declarative Route bundle per heterogeneous NL→SQL mechanism.

## Architecture (one page)

A **Route** is the complete declaration of one experimental mechanism: a candidate
prompt style with its tools, gates and cache namespace. The judge has no route since
the user's order of 2026-09-25 deleted the grain route's judge half, which measured
inert. The three layers never mix responsibilities:

- **Trigger rules** live in ``router.py``: ``route_question`` decides — from
  question/hint content plus frozen artifacts only — WHICH route a benchmark
  question takes, and ``apply_route`` maps that decision onto the candidate
  specs. Only routes registered with ``routed=True`` are router-eligible.
- **Declarations** live HERE: the ``Route`` dataclass names the frozen prompt
  block, the extra MCP tools, the feature gate, the cache-key extras, and the
  per-run hook class of the candidate side. ``REGISTRY`` / ``register`` /
  ``get`` are the only lookup surface.
- **Frozen bytes** stay in their host module: ``nl2sql._GRAIN_STYLE_BLOCK``,
  the ``_GRAIN_*`` regexes and gate functions. The registry references those
  objects by identity (never copies), so the eval ``[prompt-fingerprint]``
  attribute paths and every cache key stay byte-identical.

The candidate loops in ``nl2sql.py`` know nothing about any specific style:
``style=None`` short-circuits to the byte-identical legacy path (no registry
lookup, no hooks); any other style resolves to a Route whose hook instance is
dispatched at fixed points — ``init`` / ``before_tool`` / ``after_tool`` /
``on_stop`` / ``route_debug``.

Import discipline: this module imports the host modules eagerly (their frozen
constants are field values here); the host modules import THIS module lazily at
their dispatch points, so module load order can never cycle.

## Adding a new mechanism — five steps

1. ``tools/<x>.py`` — implement the new MCP tool(s).
2. ``server.py`` — register the tool(s) with the MCP server.
3. ``routes.py`` — subclass ``CandidateHooks`` as needed, then ``register(Route(...))``
   with the frozen prompt block, tools, gate, and cache extras. No central
   loop edits, ever.
4. ``router.py`` — add the content-only trigger rule that emits the new style,
   and set ``routed=True`` on the Route so ``apply_route`` accepts it.
5. ``tests/`` — pin the Route declaration (block identity, cache extras, gates)
   and the trigger rule's positives plus negative controls.
"""

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass

from . import nl2sql as _nl2sql
from .nl2sql import (
    _call_mcp,
    _grain_entity_error,
    _grain_expected_entity,
    _grain_gate_correction,
    _grain_submit_error,
    _normalize_tool_args,
)

# ── hook bases (all no-op) ────────────────────────────────────────────────────


class CandidateHooks:
    """No-op per-run hook set for one styled candidate agent loop.

    One instance is created per candidate run (question × seed). The loop
    dispatches every hook unconditionally for styled routes, so each base
    method is a safe no-op; a mechanism overrides only the points it needs.
    ``style=None`` runs never construct hooks at all (legacy fast path).
    """

    gate_version: str | None = None   # surfaces as route_debug.grain_gate_version

    def __init__(self, route: "Route", question: str) -> None:
        self.route = route
        self.question = question

    async def init(self, session, db_id: str, known_tools: set[str]) -> None:
        """Prepare state from effective runtime capabilities before the loop starts."""

    async def before_tool(self, tool_name: str, inputs: dict) -> dict | None:
        """Intercept one tool call; a non-None result replaces the MCP call."""
        return None

    async def after_tool(self, tool_name: str, result: object) -> object:
        """Observe/enrich one tool result (state tracking, field injection)."""
        return result

    async def on_stop(self, last_base_sql: str | None) -> dict | None:
        """Gate a stop-without-submit; a returned correction re-prompts the model."""
        return None

    def route_debug(self, style_block: str) -> dict:
        """Build the ``route_debug`` dict attached to a styled candidate result."""
        return {
            "style": self.route.name,
            "style_prompt_hash": hashlib.sha256(style_block.encode()).hexdigest()[:16],
            "grain_gate_version": self.gate_version,
            "sql_timeout_sec": os.environ.get("RUBIKSQL_SQL_TIMEOUT_SEC", "1200"),
            "audit_timeout_sec": os.environ.get("RUBIKSQL_GRAIN_AUDIT_TIMEOUT_SEC", "30"),
        }


# ── grain hooks (wrap the frozen gate functions in nl2sql; never alter them) ──


class GrainCandidateHooks(CandidateHooks):
    """Grain candidate hooks: schema prefetch, audit tracking, submit/stop gates.

    Wraps the FROZEN gate functions living in ``nl2sql`` (submit gate, entity
    gate, bounded corrections, entity inference) around per-run audit/schema
    state. The prompt/regex/gate bytes themselves stay in the host module.
    """

    gate_version = _nl2sql._GRAIN_GATE_VERSION
    prefetch_error = "grain route could not prefetch schema for entity validation"

    def __init__(self, route: "Route", question: str) -> None:
        super().__init__(route, question)
        self.audit: dict | None = None
        self.schema: dict | None = None
        self.rejected: dict[str, int] = {}
        self._expected: dict | None = None
        self._entity_note: str | None = None
        self._entity_notes_sent = 0

    async def init(self, session, db_id: str, known_tools: set[str]) -> None:
        if "get_db_info" not in known_tools:
            raise RuntimeError(self.prefetch_error)
        await self._prefetch_schema(session, db_id)

    async def _prefetch_schema(self, session, db_id: str) -> None:
        schema = await _call_mcp(session, "get_db_info", {}, db_id)
        if not isinstance(schema, dict) or "error" in schema:
            raise RuntimeError(self.prefetch_error)
        self.schema = schema

    async def before_tool(self, tool_name: str, inputs: dict) -> dict | None:
        self._expected = None
        if tool_name == "submit_final_sql":
            normalized = _normalize_tool_args(tool_name, inputs)
            gate = _grain_submit_error(normalized.get("sql", ""), self.audit, self.question)
            if gate is not None:
                reason, correction = gate
                return _grain_gate_correction(self.rejected, reason, correction)
            return None
        if tool_name == "audit_measure_grain":
            self._expected = _grain_expected_entity(self.question, self.schema)
            self._entity_note = _grain_entity_error(
                inputs.get("source_sql", ""), self._expected,
                str(inputs.get("entity_key", "") or ""),
            )
        return None

    async def after_tool(self, tool_name: str, result: object) -> object:
        if (tool_name == "audit_measure_grain" and self._entity_note
                and isinstance(result, dict) and "error" not in result
                and self._entity_notes_sent < 3):
            # Advisory, and capped like the submit-gate corrections: the old hard
            # gate re-rejected the same conversation up to 6 times (81 repeat hits
            # in the 08-06/08-07 corpus); three statements of one opinion is enough.
            self._entity_notes_sent += 1
            result = {**result, "notes": [*result.get("notes", []), self._entity_note]}
        if (tool_name == "audit_measure_grain" and self._expected
                and isinstance(result, dict) and "error" not in result):
            result = {**result, "declared_answer_entity": self._expected.get("table", "")}
        if (tool_name == "audit_measure_grain" and isinstance(result, dict)
                and "error" not in result and result.get("audit_status") in {
                    "one_row_per_entity", "fanout_detected",
                }):
            if (self.audit is None
                    or result.get("audit_status") == "fanout_detected"):
                self.audit = result
        if (tool_name == "get_db_info" and isinstance(result, dict)
                and "error" not in result):
            self.schema = result
        return result

    async def on_stop(self, last_base_sql: str | None) -> dict | None:
        if not last_base_sql:
            return None
        gate = _grain_submit_error(last_base_sql, self.audit, self.question)
        if gate is None:
            return None
        reason, correction = gate
        _grain_gate_correction(self.rejected, reason, correction)
        return correction


# ── Route declaration ─────────────────────────────────────────────────────────


def _no_gate() -> None:
    """Feature gate for routes with no feature bit (always available)."""


def _no_extras(db_id: str, question: str) -> dict:
    """Cache extras for routes with no style-specific cache namespace."""
    return {}


@dataclass(frozen=True)
class Route:
    """Declarative bundle for one heterogeneous candidate mechanism.

    Every field references FROZEN objects owned by the host module (prompt
    block, tool-name set) or small pure callables defined in this module;
    a Route never carries copied prompt text or duplicated gate logic.

    Args:
        name: Style name — the key in ``REGISTRY`` and the ``style`` value in
            cache extras and ``route_debug``.
        candidate_block: Prompt suffix appended to the candidate system prompt
            (identity reference to the host module's frozen constant).
        candidate_tools: Extra MCP tools this route adds to the candidate tool
            set; also protected from ``disabled_tools`` removal.
        candidate_hooks: Hook class instantiated per candidate run.
        candidate_gate: Feature-bit check; raises RuntimeError (frozen text)
            when the bit is missing.
        cache_extras_fn: Style-specific cache-key extras beyond ``style``.
        routed: Whether ``router.apply_route`` may emit this style. Routes
            without trigger semantics stay manual-only.
    """

    name: str
    candidate_block: str
    candidate_tools: frozenset = frozenset()
    candidate_hooks: type[CandidateHooks] = CandidateHooks
    candidate_gate: Callable[[], None] = _no_gate
    cache_extras_fn: Callable[[str, str], dict] = _no_extras
    routed: bool = False

    def require(self) -> None:
        """Check this route's feature bit.

        Raises:
            RuntimeError: The feature bit is missing (frozen message, e.g.
                "style=grain requires RUBIKSQL_GRAIN_AUDIT=1").
        """
        self.candidate_gate()

    def enabled_candidate_tools(self) -> frozenset:
        """Return ``candidate_tools`` when the feature bit is satisfied, else empty.

        Mirrors the legacy behavior of tool listing: an unsatisfied feature bit
        silently withholds the route's tools instead of raising (only prompt
        construction raises).
        """
        try:
            self.require()
        except RuntimeError:
            return frozenset()
        from .tool_bundle import process_bundle
        from .tool_definitions import select_definitions

        records = process_bundle().definitions
        selected = {record["name"] for record in select_definitions(records, "candidate", self.name)}
        default = {record["name"] for record in select_definitions(records, "candidate")}
        return frozenset(selected - default)

    def cache_extras(self, db_id: str, question: str) -> dict:
        """Cache-key extras for one styled candidate run.

        Returns:
            dict: ``{"style": name}`` merged with the route's specific extras —
            byte-identical to the legacy per-style ``route_extras`` dicts.
        """
        return {"style": self.name, **self.cache_extras_fn(db_id, question)}


# ── registry ──────────────────────────────────────────────────────────────────

REGISTRY: dict[str, Route] = {}


def register(route: Route) -> Route:
    """Register ``route`` under its name (last registration wins) and return it."""
    REGISTRY[route.name] = route
    return route


def get(style: str) -> Route:
    """Look up one registered route by style name.

    Args:
        style: Style name emitted by the router or passed by a caller.

    Returns:
        Route: The registered route.

    Raises:
        ValueError: Unknown style (frozen text "Unknown candidate style {style!r}").
    """
    route = REGISTRY.get(style)
    if route is None:
        raise ValueError(f"Unknown candidate style {style!r}")
    return route


# ── grain feature gates / cache extras (values frozen in the host modules) ────


def _grain_candidate_gate() -> None:
    # Module-attribute read (not a captured value): tests monkeypatch
    # nl2sql._GRAIN_AUDIT, and the env snapshot lives there.
    if not _nl2sql._GRAIN_AUDIT:
        raise RuntimeError("style=grain requires RUBIKSQL_GRAIN_AUDIT=1")


def _grain_cache_extras(db_id: str, question: str) -> dict:
    return {
        "style_prompt": hashlib.sha256(
            _nl2sql._GRAIN_STYLE_BLOCK.encode()).hexdigest()[:12],
        "grain_gate": _nl2sql._GRAIN_GATE_VERSION,
        "sql_timeout": os.environ.get("RUBIKSQL_SQL_TIMEOUT_SEC", "1200"),
        "audit_timeout": os.environ.get("RUBIKSQL_GRAIN_AUDIT_TIMEOUT_SEC", "30"),
    }


# ── route definitions ─────────────────────────────────────────────────────────

# Logical-grain audit route: the candidate suite with the audit_measure_grain
# tool, submit/stop/entity gates, and its own cache namespace.
register(Route(
    name="grain",
    candidate_block=_nl2sql._GRAIN_STYLE_BLOCK,
    candidate_tools=_nl2sql._GRAIN_AUDIT_TOOLS,
    candidate_hooks=GrainCandidateHooks,
    candidate_gate=_grain_candidate_gate,
    cache_extras_fn=_grain_cache_extras,
    routed=True,
))
