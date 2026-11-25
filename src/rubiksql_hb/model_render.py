"""Deterministic model-facing text projections for structured tool results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
import math
import re
import unicodedata
from typing import Any

#: render_v6 drops ``tags`` from the rule blocks the model reads (see :func:`render_rules`).
#: Pre-v6 transcripts must not replay: they were produced by a model that read a tag line,
#: so replaying them would score the removal as a no-op — the exact trap the candidate-side
#: ``msg_shape`` comment in ``nl2sql`` describes.
#:
#: render_v7 serves the example menu as vertical records instead of positional JSON rows
#: (see :func:`_render_example_menu`), so a v6 transcript answers from a menu whose hints
#: were escaped ``\n`` inside a string. Same trap, same remedy: bump so it cannot replay.
MODEL_PROTOCOL = "sql_native_v2|render_v7"

KNOWLEDGE_RENDER_TOOLS = frozenset({
    "get_rules",
    "get_examples",
    "expand_examples",
    "save_rule",
    "update_rule",
    "query_knowledge",
})
_SQL_TABLE_TOOLS = frozenset({"run_sql", "run_sql_base", "submit_final_sql"})
_SQL_RECEIPT_TOOLS = frozenset({"save_sql_as_table", "audit_measure_grain"})
MODEL_RENDER_TOOLS = frozenset({
    *_SQL_TABLE_TOOLS,
    *_SQL_RECEIPT_TOOLS,
    *KNOWLEDGE_RENDER_TOOLS,
    "search_values",
    "search_columns",
    "get_col_info",
    "get_db_info",
    "get_tab_info",
})

__all__ = [
    "binary_json_default",
    "restore_sql_results",
    "KNOWLEDGE_RENDER_TOOLS",
    "MODEL_PROTOCOL",
    "MODEL_RENDER_TOOLS",
    "fenced_sql",
    "model_tool_content",
    "render_knowledge_result",
    "render_rules",
    "table_display",
    "table_row_indices",
]


def binary_json_default(value: object) -> object:
    """Encode binary SQL cells for JSON writes without changing other JSON values.

    Use as ``json.dumps(default=...)``. Unsupported nonbinary types still raise
    TypeError. The same tagged hex object is emitted by the pinned SQL tools.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"$type": "bytes", "hex": bytes(value).hex()}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _binary_cell(value: object) -> object:
    """Restore only a cell marker, never a row or arbitrary metadata object."""
    if isinstance(value, dict) and set(value) == {"$type", "hex"} and value["$type"] == "bytes" and isinstance(value["hex"], str):
        return bytes.fromhex(value["hex"])
    return value


def restore_sql_results(value: object) -> object:
    """Restore binary cells in decoded SQL result rows, including nested caches.

    Mapping rows keep their column names, even when those names spell marker
    keys. Metadata objects and database strings are never interpreted as cells.
    """
    if isinstance(value, list):
        return [restore_sql_results(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key != "rows" or not isinstance(item, list):
            result[key] = restore_sql_results(item)
            continue
        result[key] = [
            {column: _binary_cell(cell) for column, cell in row.items()} if isinstance(row, dict)
            else [_binary_cell(cell) for cell in row] if isinstance(row, (list, tuple))
            else _binary_cell(row)
            for row in item
        ]
    return result


@dataclass(frozen=True, slots=True)
class _TableView:
    """One JSON-row projection before channel-specific labels are added."""

    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    total: int
    shown: int
    truncated: bool
    head_count: int
    tail_count: int


def _json_value(value: object) -> object:
    """Normalize one value into strict, type-preserving JSON data."""
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return binary_json_default(value)
    if isinstance(value, float) and not math.isfinite(value):
        label = "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
        return {"$type": "float", "value": label}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _compact_json(value: object) -> str:
    """Serialize strict JSON deterministically without ASCII escaping."""
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _record_fragment(value: object) -> str:
    """Render an inline record fragment without allowing physical control lines."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "NULL"
    if isinstance(value, (Mapping, list, tuple)):
        payload: object = dict(value) if isinstance(value, Mapping) else list(value)
        return _compact_json(payload)
    text = str(value)
    out: list[str] = []
    for char in text:
        if char == "\r":
            out.append("\\r")
        elif char == "\n":
            out.append("\\n")
        elif char == "\t":
            out.append("\\t")
        elif unicodedata.category(char) == "Cc":
            out.append(f"\\u{ord(char):04X}")
        else:
            out.append(char)
    return "".join(out)


def _record_body(value: object) -> list[str]:
    """Render a long record field as indented, sentinel-safe physical lines."""
    if isinstance(value, (Mapping, list, tuple)):
        payload: object = dict(value) if isinstance(value, Mapping) else list(value)
        return ["  " + _compact_json(payload)]
    normalized = str(value if value is not None else "").replace("\r\n", "\n").replace("\r", "\n")
    return ["  " + _record_fragment(line) for line in normalized.split("\n")]


def _row_sequence(rows: object) -> list[object]:
    """Normalize a table input without mutating it or splitting scalar strings."""
    if rows is None:
        return []
    if isinstance(rows, list):
        return list(rows)
    if isinstance(rows, tuple):
        return list(rows)
    return [rows]


def _column_sequence(rows: list[object], columns: Sequence[object] | None) -> list[object]:
    """Return an explicit or first-seen table column order."""
    if columns is not None:
        return list(columns)
    seen: set[object] = set()
    inferred: list[object] = []
    for row in rows:
        if isinstance(row, Mapping):
            for key in row:
                if key not in seen:
                    seen.add(key)
                    inferred.append(key)
        elif isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray, memoryview)):
            for index in range(len(row)):
                if index not in seen:
                    seen.add(index)
                    inferred.append(index)
        elif "value" not in seen:
            seen.add("value")
            inferred.append("value")
    return inferred


def _row_values(row: object, columns: list[object]) -> list[object]:
    """Project one arbitrary row into an explicit column order."""
    if isinstance(row, Mapping):
        return [row.get(column) for column in columns]
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray, memoryview)):
        return [
            row[index] if index < len(row) else None
            for index, _column in enumerate(columns)
        ]
    return [row if index == 0 else None for index, _column in enumerate(columns)]


def _header_value(column: object) -> str:
    """Return one deterministic JSON-row column label."""
    if isinstance(column, str):
        return column
    normalized = _json_value(column)
    if normalized is None:
        return "null"
    if isinstance(normalized, bool):
        return "true" if normalized else "false"
    if isinstance(normalized, (int, float)):
        return str(normalized)
    return _compact_json(normalized)


def table_row_indices(total: int, max_rows: int) -> tuple[int, ...]:
    """Select zero-based source rows for a bounded head/tail table preview.

    When rows are omitted, the head gets the extra row for an odd limit. A
    non-positive limit selects no rows; an untruncated result selects every row.
    """
    count = max(0, int(total))
    limit = max(0, int(max_rows))
    if count <= limit:
        return tuple(range(count))
    head_count = (limit + 1) // 2
    tail_count = limit - head_count
    return (*range(head_count), *range(count - tail_count, count))


def _render_table(
    rows: object,
    *,
    max_rows: int,
    columns: Sequence[object] | None,
) -> _TableView:
    """Build a deterministic positional JSON-row view."""
    source = _row_sequence(rows)
    table_columns = _column_sequence(source, columns)
    total = len(source)
    limit = max(0, int(max_rows))
    truncated = total > limit
    selected = [source[index] for index in table_row_indices(total, limit)]
    top_count = (limit + 1) // 2 if truncated else total
    bottom_count = len(selected) - top_count
    return _TableView(
        columns=tuple(_header_value(column) for column in table_columns),
        rows=tuple(
            tuple(_json_value(value) for value in _row_values(row, table_columns))
            for row in selected
        ),
        total=total,
        shown=len(selected),
        truncated=truncated,
        head_count=top_count,
        tail_count=bottom_count,
    )


def _table_payload(
    view: _TableView,
    *,
    count: int | None = None,
    extras: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return one strict JSON result object with optional truncation metadata."""
    total = view.total if count is None else count
    payload: dict[str, object] = {
        "columns": list(view.columns),
        "rows": [list(row) for row in view.rows],
        "row_count": total,
    }
    if view.truncated or total > view.shown:
        payload["truncation"] = {
            "strategy": "head_tail" if view.truncated else "provided_excerpt",
            "head_row_count": view.head_count,
            "tail_row_count": view.tail_count,
            "omitted_row_count": max(0, total - view.shown),
        }
    if extras:
        payload.update({
            str(key): _json_value(value)
            for key, value in extras.items()
        })
    return payload


def _table_text(
    view: _TableView,
    *,
    count: int | None = None,
    extras: Mapping[str, object] | None = None,
) -> str:
    """Serialize one JSON-row view as compact valid JSON."""
    return json.dumps(
        _table_payload(view, count=count, extras=extras),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def table_display(
    rows: object,
    max_rows: int = 64,
    columns: Sequence[object] | None = None,
) -> tuple[str, bool]:
    """Render rows as deterministic, strict model-facing JSON.

    The first and last halves are shown when the row count exceeds ``max_rows``.
    Column names and positional row arrays preserve order without delimiter ambiguity.

    Args:
        rows: Structured result rows. Historical list/tuple/scalar rows are normalized.
        max_rows: Maximum number of real rows to display.
        columns: Optional explicit column order. Supplying it preserves an empty result's schema.

    Returns:
        tuple[str, bool]: Compact JSON result and whether rows were omitted.
    """
    view = _render_table(rows, max_rows=max_rows, columns=columns)
    return _table_text(view), view.truncated


def fenced_sql(sql: object) -> str:
    """Render SQL in a collision-free Markdown fence without changing its body.

    Args:
        sql: SQL text to display.

    Returns:
        str: A fenced ``sql`` block whose delimiter is longer than any matching run in the body.
    """
    body = "" if sql is None else str(sql)
    tick_run = max((len(match.group(0)) for match in re.finditer(r"`+", body)), default=0)
    tilde_run = max((len(match.group(0)) for match in re.finditer(r"~+", body)), default=0)
    marker, run = ("`", tick_run) if tick_run <= tilde_run else ("~", tilde_run)
    fence = marker * max(3, run + 1)
    separator = "" if body.endswith(("\n", "\r")) else "\n"
    return f"{fence}sql\n{body}{separator}{fence}"


def _dict_rows(rows: object) -> list[dict[str, Any]]:
    """Keep mapping rows and copy their outer mappings for safe projection."""
    return [dict(row) for row in _row_sequence(rows) if isinstance(row, Mapping)]


def _error_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split valid and error records without modifying either group."""
    return (
        [row for row in rows if not row.get("error")],
        [row for row in rows if row.get("error")],
    )


def _render_errors(rows: list[dict[str, Any]], heading: str = "Tool errors") -> str:
    """Render error rows while retaining any session-local handle."""
    lines = [heading]
    for row in rows:
        virtual_id = row.get("virtual_id")
        prefix = f"[{_record_fragment(virtual_id)}] " if virtual_id else "- "
        lines.append(prefix + _record_fragment(row.get("error") or "unknown error"))
    return "\n".join(lines)


def render_rules(
    rows: list[dict[str, Any]],
    heading: str = "Rules",
    explain_scope_conflict: bool = False,
    include_ids: bool = True,
    fields: tuple[str, ...] = ("scope", "desc", "example"),
) -> str:
    """Render rules as deterministic vertical record blocks.

    ``tags`` is deliberately absent. It stays on the entity and still earns its keep in
    retrieval — :func:`_text_score` weights it at 3.0, second only to the title — but the
    model gains nothing from reading it: the labels are generic construct words
    (``aggregate``, ``where``, ``ordering``) already implied by the desc and example that
    follow them. Measured at 6% of a 20-rule block, about 200 tokens per call. The judge is
    known to read retrieved rules literally and over-apply formulaic shapes, so a bare
    category label is likelier to invite template matching than to aid comprehension.

    Args:
        rows: Public rule projections.
        heading: Section heading placed before the record count.
        explain_scope_conflict: Add database-over-global guidance only for mixed scopes.
        include_ids: Use session-local R-ids; when false, use the non-actionable ``[Rule]`` label.
        fields: Ordered fields to include after each title.

    Returns:
        str: One indented block per rule, with no persistent ids or vector fields.
    """
    valid = [dict(row) for row in rows if isinstance(row, Mapping) and not row.get("error")]
    lines = [f"{heading} ({len(valid)})"]
    scopes = {
        str(row.get("scope") or "").strip()
        for row in valid
        if row.get("scope")
    }
    if explain_scope_conflict and len(scopes) >= 2:
        lines.append(
            "Scope priority: when rules with different scopes conflict, prefer "
            "scope: database (specific to the current database) over scope: global."
        )
    for row in valid:
        virtual_id = _record_fragment(row.get("virtual_id") or "?")
        label = virtual_id if include_ids else "Rule"
        lines.extend(["", f"[{label}] {_record_fragment(row.get('title') or '')}"])
        for field in fields:
            value = row.get(field)
            if value in (None, "", []):
                continue
            if field in {"desc", "example"}:
                lines.append(f"{field}:")
                lines.extend(_record_body(value))
            else:
                lines.append(f"{field}: {_record_fragment(value)}")
    return "\n".join(lines)


def _render_rule_result(
    tool_name: str,
    rows: list[dict[str, Any]],
    *,
    heading: str | None = None,
) -> str:
    """Render a rule result plus any error records."""
    valid, errors = _error_rows(rows)
    section = heading or {
        "save_rule": "Saved rules",
        "update_rule": "Updated rules",
    }.get(tool_name, "Rules")
    parts: list[str] = []
    if valid:
        parts.append(render_rules(
            valid,
            section,
            explain_scope_conflict=tool_name == "get_rules",
        ))
    if errors:
        parts.append(_render_errors(errors, "Knowledge errors"))
    return "\n\n".join(parts) if parts else f"{section} (0)"


def _question_lines(row: dict[str, Any]) -> list[str]:
    """Render an expanded example question without creating unowned record lines."""
    virtual_id = _record_fragment(row.get("virtual_id") or "?")
    question = str(row.get("question") or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = question.split("\n")
    return [
        f"[{virtual_id}] {_record_fragment(parts[0])}",
        *["  " + _record_fragment(part) for part in parts[1:]],
    ]


def _example_scope_lines(row: dict[str, Any]) -> list[str]:
    scope = row.get("example_scope")
    if not isinstance(scope, dict) or not scope:
        return []
    return ["  Example scope (this precedent only; do not narrow the current request): "
            + json.dumps(scope, ensure_ascii=False, sort_keys=True)]


def _render_expanded_examples(rows: list[dict[str, Any]]) -> str:
    """Render full example SQL retained behind session-local E-ids."""
    lines = [f"Expanded examples ({len(rows)})"]
    for row in rows:
        lines.append("")
        lines.extend(_question_lines(row))
        lines.extend(_example_scope_lines(row))
        if "sql" in row:
            lines.extend(["SQL:", fenced_sql(row.get("sql"))])
    return "\n".join(lines)


def _render_example_menu(rows: list[dict[str, Any]]) -> str:
    """Render the two-stage example menu as vertical records, one block per candidate.

    The menu was the last knowledge channel still served as positional JSON rows, so a model
    reading `get_rules`, `expand_examples` and this one in the same turn had to switch
    between a labelled block form and a form where each value is identified by its index in
    a separate `columns` list. Nothing is gained by the difference: the menu is around a
    dozen rows of three fields, so repeating the labels costs almost nothing, and it shares
    both its id space (`[E01]`) and its questions with `expand_examples`, which is already
    vertical.

    The questions are the concrete reason. Each carries its `[Hint]` on a second line, and a
    JSON row can only hold that as an escaped `\\n` inside a string — the same escaping
    problem that moved SQL out of function arguments and into text blocks. `_question_lines`
    puts the hint on its own indented line instead, so the menu reads the way the expanded
    form of the very same example will.

    Every row is rendered. The previous implementation borrowed the table renderer and with
    it a 64-row cap, which the menu can never reach: `get_examples` serves exactly the
    requested `top_k` (default 12 — the ask IS the behavior since the ×2 removal), and
    retained traces top out at 12. A cap that cannot fire is not a
    safeguard — it is a silent truncation waiting for the day someone raises `top_k`.
    """
    lines = [f"Examples ({len(rows)})"]
    for row in rows:
        lines.append("")
        lines.extend(_question_lines(row))
        lines.extend(_example_scope_lines(row))
        length = row.get("sql_length", len(str(row.get("sql") or "")))
        lines.append(f"  sql_length: {_record_fragment(length)}")
    return "\n".join(lines)


def _render_examples(tool_name: str, rows: list[dict[str, Any]]) -> str:
    """Render example menu/expansion rows and preserve id-associated errors.

    No row cap. Both forms are bounded by what the retrieval asked for — `get_examples`
    takes `top_k` (6 by default, doubled for the menu) and `expand_examples` renders the
    ids the model named — so a second limit here could only ever remove rows the model
    had already been promised, silently.
    """
    valid, errors = _error_rows(rows)
    expanded = tool_name == "expand_examples" or (
        tool_name in {"get_examples", "query_knowledge"}
        and any("sql" in row for row in valid)
    )
    parts: list[str] = []
    if valid:
        parts.append(
            _render_expanded_examples(valid)
            if expanded
            else _render_example_menu(valid)
        )
    elif not errors:
        parts.append(
            _render_expanded_examples([])
            if expanded
            else _render_example_menu([])
        )
    if errors:
        parts.append(_render_errors(errors, "Knowledge errors"))
    return "\n\n".join(parts)


def _render_records(rows: list[dict[str, Any]], entity: str) -> str:
    """Preserve the existing exact-query text shape for other knowledge entities."""
    lines = [f"Knowledge records: {entity or 'unknown'} ({len(rows)})"]
    for index, row in enumerate(rows, 1):
        label = row.get("virtual_id") or row.get("entity_id") or index
        lines.extend(["", f"[{_record_fragment(label)}]"])
        for key, value in row.items():
            if key in {"virtual_id", "entity_id"} or value in (None, "", []):
                continue
            lines.append(f"{key}: {_record_fragment(value)}")
    return "\n".join(lines)


def render_knowledge_result(
    tool_name: str,
    result: object,
    inputs: dict[str, Any] | None = None,
) -> str | None:
    """Render an existing knowledge channel, leaving other tools untouched.

    Takes no row cap: every knowledge channel is already bounded by its own retrieval
    parameter, and the ones that enumerate — `query_knowledge` promises the model "EVERY
    record matching an exact filter (up to 200)" — would be cut to a partial answer the
    model could not tell from a complete one.

    Args:
        tool_name: Normalized MCP tool name.
        result: Structured domain/server result.
        inputs: Normalized arguments, used to identify an exact-query entity.

    Returns:
        str | None: Model-facing knowledge text, or ``None`` for another tool.
    """
    if tool_name not in KNOWLEDGE_RENDER_TOOLS:
        return None
    rows = _dict_rows(result)
    if tool_name in {"get_rules", "save_rule", "update_rule"}:
        return _render_rule_result(tool_name, rows)
    if tool_name in {"get_examples", "expand_examples"}:
        return _render_examples(tool_name, rows)
    entity = str((inputs or {}).get("entity") or "")
    valid, errors = _error_rows(rows)
    parts: list[str] = []
    if entity == "rule":
        if valid or not errors:
            # render_rules([]) says "Rule records (0)" — a zero-match rule query must never
            # reach the model as a zero-byte tool message (indistinguishable from a crash).
            parts.append(render_rules(valid, "Rule records"))
    elif entity == "example":
        if valid or not errors:
            parts.append(_render_expanded_examples(valid))
    elif valid:
        parts.append(_render_records(valid, entity))
    elif not errors:
        parts.append(f"Knowledge records: {entity or 'unknown'} (0)")
    if errors:
        parts.append(_render_errors(errors, "Knowledge errors"))
    return "\n\n".join(parts)


def _render_search_values(result: object) -> str:
    """Render value matches with stable columns across vector and substring paths.

    Every match is rendered. `search_values` already bounds itself with `top_k`, and that
    parameter is exposed to the model — a second cap here would silently drop matches a
    model had explicitly asked for, which is the failure the caller cannot detect.
    """
    if not isinstance(result, Mapping):
        return _compact_json(result)
    if result.get("error"):
        return "ERROR: " + _record_fragment(result["error"])
    matches = _dict_rows(result.get("matches"))
    table_rows = [{
        "table_name": row.get("table_name"),
        "col_name": row.get("col_name"),
        "value": row.get("value"),
        "score": row.get("score"),
        "semantic_score": row.get("semantic_score") if "semantic_score" in row else None,
        "spelling_score": row.get("spelling_score") if "spelling_score" in row else None,
        "via": row.get("via", ""),
    } for row in matches]
    view = _render_table(
        table_rows,
        max_rows=len(table_rows),
        columns=("table_name", "col_name", "value", "score", "semantic_score", "spelling_score", "via"),
    )
    extras: dict[str, object] = {
        "strong_match": result.get("strong_match"),
        "best_score": result.get("best_score"),
    }
    if result.get("note"):
        extras["note"] = result["note"]
    return "Value matches\n" + _table_text(view, extras=extras)


def _render_search_columns(result: object) -> str:
    """Render column search hits without dropping stable lookup fields."""
    rows = _dict_rows(result)
    valid, errors = _error_rows(rows)
    include_profile = any("profile_desc" in row for row in valid)
    columns = [
        "object_id",
        "table_name",
        "column_name",
        "datatype_orig",
        "datatype_anno",
        "desc",
    ]
    if include_profile:
        columns.append("profile_desc")
    table_rows = [{
        "object_id": row.get("object_id"),
        "table_name": row.get("table_name"),
        "column_name": row.get("name"),
        "datatype_orig": row.get("datatype_orig"),
        "datatype_anno": row.get("datatype_anno"),
        "desc": row.get("desc"),
        **({"profile_desc": row.get("profile_desc")} if include_profile else {}),
    } for row in valid]
    view = _render_table(
        table_rows,
        max_rows=len(table_rows),
        columns=columns,
    )
    parts = ["Column matches\n" + _table_text(view)]
    if errors:
        parts.append(_render_errors(errors))
    return "\n\n".join(parts)


def _render_col_info(result: object) -> str:
    """Unify whole-table and single-column profile results into record blocks."""
    if not isinstance(result, Mapping):
        return _compact_json(result)
    if result.get("error"):
        return "ERROR: " + _record_fragment(result["error"])
    lines = [
        "WARNING: " + _record_fragment(warning)
        for warning in result.get("warnings") or []
    ]
    records: list[tuple[object, object, object]] = []
    if isinstance(result.get("columns"), list):
        table_name = result.get("table")
        records.extend(
            (table_name, row.get("name"), row.get("profile_desc"))
            for row in result["columns"]
            if isinstance(row, Mapping)
        )
    elif "name" in result:
        records.append((
            result.get("table_name") or result.get("table"),
            result.get("name"),
            result.get("profile_desc"),
        ))
    for table_name, column_name, profile_desc in records:
        if lines:
            lines.append("")
        lines.append(
            f"[{_record_fragment(table_name or '?')}.{_record_fragment(column_name or '?')}]"
        )
        lines.append("profile_desc:")
        lines.extend(_record_body(profile_desc))
    if not records:
        if lines:
            lines.append("")
        lines.append("No column profiles found.")
    return "\n".join(lines)


def _schema_rows(result: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Flatten one database/table schema without losing parent table identity."""
    source_tables = result.get("tables")
    if isinstance(source_tables, list):
        tables = [dict(row) for row in source_tables if isinstance(row, Mapping)]
    elif "table" in result:
        tables = [dict(result)]
    else:
        tables = []
    table_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    for table in tables:
        table_name = table.get("table")
        table_rows.append({
            "table_name": table_name,
            "n_rows": table.get("n_rows"),
            "pks": table.get("pks") or [],
            "desc": table.get("desc") or "",
        })
        for column in table.get("columns") or []:
            if not isinstance(column, Mapping):
                continue
            column_rows.append({
                "table_name": table_name,
                "column_name": column.get("name"),
                "type": column.get("type"),
                "is_pk": column.get("is_pk"),
                "fk_ref": column.get("fk_ref", ""),
                "desc": column.get("desc", ""),
            })
    return table_rows, column_rows


def _render_schema(tool_name: str, result: object) -> str:
    """Render nested database metadata as related JSON-row sections."""
    if not isinstance(result, Mapping):
        return _compact_json(result)
    if result.get("error"):
        return "ERROR: " + _record_fragment(result["error"])
    table_rows, column_rows = _schema_rows(result)
    table_view = _render_table(
        table_rows,
        max_rows=len(table_rows),
        columns=("table_name", "n_rows", "pks", "desc"),
    )
    column_view = _render_table(
        column_rows,
        max_rows=len(column_rows),
        columns=("table_name", "column_name", "type", "is_pk", "fk_ref", "desc"),
    )
    lines: list[str] = []
    if tool_name == "get_db_info":
        lines.append("Database: " + _record_fragment(result.get("db_id")))
        overview = str(result.get("overview") or "").strip()
        if overview:
            # Database-level overview (cross-table disambiguation, grain map) — plain
            # paragraph, no tag: it is part of the schema the model is already reading.
            # Absent overview keeps the pre-08-05 byte shape exactly.
            lines.extend(["", "Overview: " + _record_fragment(overview)])
        lines.extend(["", "Tables"])
    else:
        lines.append("Table")
    lines.extend([_table_text(table_view), "", "Columns", _table_text(column_view)])
    # Brief-schema footers (生产唯一形态): a pointer the model can act on beats a silent cut.
    if tool_name == "get_db_info":
        lines.extend(["", "(Full column docs: get_tab_info(table).)"])
    else:
        lines.extend(["", "(Column value profiles — stored formats, nulls, top values — "
                          "via get_col_info.)"])
    return "\n".join(lines)


def _sql_row_count(result: Mapping[str, Any], fallback: int) -> int:
    """Read a non-negative SQL row count, falling back to the projected row length."""
    try:
        return max(0, int(result.get("row_count", fallback)))
    except (TypeError, ValueError):
        return fallback


def _render_sql_result(
    tool_name: str,
    result: object,
    *,
    max_rows: int,
) -> str:
    """Render one SQL execution result inside its unique plain-text result tag."""
    tag = f"{tool_name}_result"
    if not isinstance(result, Mapping):
        body = _compact_json(result)
    elif result.get("error"):
        body = "ERROR: " + _record_fragment(result["error"])
    else:
        rows = _row_sequence(result.get("rows"))
        raw_columns = result.get("columns")
        columns = (
            list(raw_columns)
            if isinstance(raw_columns, Sequence) and not isinstance(raw_columns, (str, bytes))
            else None
        )
        view = _render_table(
            rows,
            max_rows=max_rows,
            columns=columns,
        )
        count = _sql_row_count(result, len(rows))
        notes = [
            result[note_key]
            for note_key in ("_note", "note")
            if result.get(note_key)
        ]
        extras: dict[str, object] = {}
        if notes:
            extras["notes"] = notes
        if result.get("final") is True:
            extras["final"] = True
        body = _table_text(view, count=count, extras=extras)
    return f"<{tag}>\n{body}\n</{tag}>"


def _render_sql_receipt(tool_name: str, result: object) -> str:
    """Render a non-tabular SQL action receipt inside its matching result tag."""
    tag = f"{tool_name}_result"
    if isinstance(result, Mapping) and result.get("error"):
        body = "ERROR: " + _record_fragment(result["error"])
    else:
        body = _compact_json(result)
    return f"<{tag}>\n{body}\n</{tag}>"


def model_tool_content(
    tool_name: str,
    result: object,
    inputs: dict[str, Any] | None = None,
    *,
    max_rows: int = 64,
) -> str:
    """Return the canonical model message content for any tool result.

    Migrated channels use deterministic JSON rows, vertical records, or fenced SQL.
    Tools outside the migration retain the prior ``json.dumps`` byte shape.

    Args:
        tool_name: Normalized MCP tool or SQL action name.
        result: Structured domain/server result.
        inputs: Normalized tool arguments, when an adapter needs them.
        max_rows: Maximum real rows shown by table adapters.

    Returns:
        str: Exact text to place in the model transcript.
    """
    if tool_name in _SQL_TABLE_TOOLS:
        return _render_sql_result(
            tool_name,
            result,
            max_rows=max_rows,
        )
    if tool_name in _SQL_RECEIPT_TOOLS:
        return _render_sql_receipt(tool_name, result)
    rendered = render_knowledge_result(tool_name, result, inputs)
    if rendered is not None:
        return rendered
    if tool_name == "search_values":
        return _render_search_values(result)
    if tool_name == "search_columns":
        return _render_search_columns(result)
    if tool_name == "get_col_info":
        return _render_col_info(result)
    if tool_name in {"get_db_info", "get_tab_info"}:
        return _render_schema(tool_name, result)
    return json.dumps(result, ensure_ascii=False)
