"""Logical-entity grain diagnostics for candidate SQL construction.

The model-provided source query is the one deliberate raw-SQL exception in this
module. It is treated as an opaque subquery only after the shared HeavenBase
read-only guard accepts it; execution then adds a physical read-only connection,
SQLite query-only mode, an allowlist authorizer, and the project progress guard.
Projected identifiers are schema-validated and quoted, and the witness limit is
bound rather than interpolated.
"""

from typing import Any
import os
import re

import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

from ..sql_host import query_readonly


_MAX_FIELDS = 8
_MAX_WITNESSES = 3
_MAX_SOURCE_ROWS = 2_000_000
_AUDIT_TIMEOUT_SEC = float(os.environ.get("RUBIKSQL_GRAIN_AUDIT_TIMEOUT_SEC", "30"))
_SQLITE_DIALECT = sqlite.dialect(paramstyle="named")
_ALIAS_PUNCT_RE = re.compile(r"[^0-9a-z]+")
_SOURCE_TERMINATOR_RE = re.compile(r"[;\s]+$")
_RAW_SOURCE_FORBIDDEN_RE = re.compile(
    r"\b(?:GROUP\s+BY|HAVING|LIMIT|UNION|INTERSECT|EXCEPT|SELECT\s+DISTINCT|"
    r"(?:COUNT|SUM|AVG|MIN|MAX|TOTAL|GROUP_CONCAT)\s*\(|OVER\s*\()",
    re.IGNORECASE,
)


def _raw_source_error(source_sql: str) -> str | None:
    """Reject sources that already hide row contributions through aggregation."""
    scrubbed = re.sub(r"/\*.*?\*/", " ", source_sql, flags=re.DOTALL)
    scrubbed = re.sub(r"--[^\r\n]*", " ", scrubbed)
    scrubbed = re.sub(r"'(?:''|[^'])*'", "''", scrubbed)
    scrubbed = re.sub(r'"(?:""|[^"])*"', '""', scrubbed)
    if _RAW_SOURCE_FORBIDDEN_RE.search(scrubbed):
        return (
            "source_sql must expose raw, unaggregated contribution rows; remove DISTINCT, "
            "aggregate/window functions, set operations, GROUP BY/HAVING, and LIMIT before auditing"
        )
    return None


def _compile(statement: sa.Executable) -> tuple[str, dict[str, Any]]:
    """Compile an owned SQLAlchemy statement for the guarded SQLite executor."""
    compiled = statement.compile(dialect=_SQLITE_DIALECT)
    return str(compiled), compiled.params


def _identifier_parts(name: str) -> list[str]:
    """Split a possibly quoted, possibly table-qualified identifier into its parts.

    Quoting is stripped per part, so `"T1"."answer key"` yields `['T1', 'answer key']`
    and a plain alias containing spaces survives intact. An apostrophe is a STRING
    delimiter in SQLite, never an identifier quote, so `O'Brien` stays one part.
    """
    parts: list[str] = []
    current: list[str] = []
    closing = ""
    for char in name.strip():
        if closing:
            if char == closing:
                closing = ""
            else:
                current.append(char)
        elif char in "\"`[":
            closing = "]" if char == "[" else char
        elif char == ".":
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def _alias_forms(name: str) -> list[tuple[str, str]]:
    """Yield (candidate spelling, tolerance reason) forms for one requested field.

    A table-qualified spelling deliberately gets NO fallback form: stripping the
    qualifier resolved `o.id` to whichever projection happened to be named `id` —
    a confidently wrong grain verdict where a plain error would have taught the
    model to pass the alias it actually projected. Qualified requests therefore
    fail resolution, and the error text says what to pass instead.
    """
    stripped = name.strip()
    forms = [(stripped, "whitespace trimmed" if stripped != name else "")]
    parts = _identifier_parts(name)
    if len(parts) == 1 and parts[0] != stripped:
        forms.append((parts[0], "quoting stripped"))
    return forms


def _resolve_fields(
    requested: list[str], columns: list[str], role: str
) -> tuple[list[str], list[tuple[str, str, str, str]], list[str]]:
    """Resolve requested output aliases leniently against a source query's projection.

    Models write the alias they *meant* rather than the alias they projected, so an exact
    hit is tried first and then progressively looser spellings: whitespace/quote
    stripping, case folding, and finally folding away punctuation when that lands on
    exactly one projected column. Table-qualified spellings are never resolved (see
    `_alias_forms`). Every inexact hit is reported back so the caller can tell the model
    what its input was read as; unresolved names are returned, not raised, because the
    caller decides whether they are fatal (entity keys) or skippable (measures).

    Returns:
        tuple: the resolved projected aliases, (requested, resolved, reason, role)
        tuples for the ones that needed tolerance, and the names that resolved to
        nothing.
    """
    if len(requested) > _MAX_FIELDS:
        raise ValueError(f"at most {_MAX_FIELDS} {role} fields are allowed")
    lookup = {name.casefold(): name for name in columns}
    folded: dict[str, list[str]] = {}
    for name in columns:
        key = _ALIAS_PUNCT_RE.sub("", name.casefold())
        if key:
            folded.setdefault(key, []).append(name)

    resolved: list[str] = []
    corrections: list[tuple[str, str, str, str]] = []
    missing: list[str] = []
    for name in requested:
        hit = reason = None
        for form, why in _alias_forms(name):
            if form.casefold() in lookup:
                hit = lookup[form.casefold()]
                reason = (f"{why} and case folded" if why else "case folded") if hit != form else why
                break
        if hit is None:
            parts = _identifier_parts(name)
            key = _ALIAS_PUNCT_RE.sub("", parts[0].casefold()) if len(parts) == 1 else ""
            matches = folded.get(key, []) if key else []
            if len(matches) == 1:
                hit, reason = matches[0], "punctuation and case folded"
        if hit is None:
            missing.append(name)
            continue
        if hit != name:
            corrections.append((name, hit, reason, role))
        resolved.append(hit)
    if len(set(name.casefold() for name in resolved)) != len(resolved):
        raise ValueError(f"{role} fields must be unique")
    return resolved, corrections, missing


def measure_grain(
    url: str,
    source_sql: str,
    entity_keys: list[str] | str,
    measures: list[str] | str | None = None,
) -> dict[str, Any]:
    """Audit row contributions relative to a query's logical entity grain.

    The source query must be an unaggregated, single `SELECT`/`WITH` statement and
    must project every entity key and optional measure under a unique alias. For a
    conditional measure, project `CASE WHEN <condition> THEN 1 END`; the audit then
    reports both matching rows and matching logical entities without choosing which
    interpretation is correct.

    Intake is deliberately lenient about spelling, never about the audit itself: a
    trailing statement terminator is stripped from `source_sql`, a lone alias may be
    passed instead of a list, blank entries are dropped, and a requested alias is
    matched to the projection after whitespace/quote stripping, case folding, and
    finally punctuation folding when that lands on exactly one projected column. A
    table-qualified spelling (`T.col`) is NOT resolved — the projected alias is the
    only reliable name for a source column. An entity key that stays unresolved is an
    error; a measure that stays unresolved is SKIPPED and named in `notes` as not
    audited, because the fanout verdict does not depend on it. Every tolerance is
    stated back in `notes` so the model learns what its input was read as rather than
    silently trusting a guess.

    Args:
        url (str): Registered SQLite URL for the database being audited.
        source_sql (str): Unaggregated source query over base tables.
        entity_keys (list[str] | str): Projected aliases that identify one logical
            entity; a bare string is read as a single-element list.
        measures (list[str] | str | None): Optional projected aliases whose non-null
            row and entity coverage should be compared; same string tolerance.

    Returns:
        dict[str, Any]: Row/entity counts, contribution quantiles, repeated-entity
        witnesses, per-measure coverage, and interpretation notes. Invalid or unsafe
        input is returned as an `error` field.
    """
    intake_notes: list[str] = []

    def _clean(value: list[str] | str | None, role: str) -> list[str]:
        items = [value] if isinstance(value, str) else list(value or [])
        kept = [str(name) for name in items if str(name).strip()]
        if len(kept) != len(items):
            intake_notes.append(
                f"Input tolerance: {len(items) - len(kept)} blank {role} entr"
                f"{'y was' if len(items) - len(kept) == 1 else 'ies were'} ignored."
            )
        return kept

    entity_keys = _clean(entity_keys, "entity key")
    if not entity_keys:
        return {"error": "entity_keys must contain at least one projected source column"}
    measures = _clean(measures, "measure")
    raw_source = (source_sql or "").strip()
    source = _SOURCE_TERMINATOR_RE.sub("", raw_source)
    shape_error = _raw_source_error(source)
    if shape_error:
        return {"error": shape_error}

    probe = query_readonly(
        url, f"SELECT * FROM ({source}) AS _grain_source LIMIT 0",
        timeout_sec=_AUDIT_TIMEOUT_SEC,
    )
    if "error" in probe:
        return probe
    bounded = query_readonly(
        url,
        f"SELECT COUNT(*) AS n FROM (SELECT 1 FROM ({source}) AS _grain_source "
        f"LIMIT {_MAX_SOURCE_ROWS + 1}) AS _grain_bounded",
        timeout_sec=_AUDIT_TIMEOUT_SEC,
    )
    if "error" in bounded:
        return bounded
    if bounded["rows"][0]["n"] > _MAX_SOURCE_ROWS:
        return {"error": (
            f"source_sql exceeds the {_MAX_SOURCE_ROWS:,}-row audit budget; narrow it to the "
            "question's population before auditing and avoid accidental Cartesian joins"
        )}
    try:
        keys, key_fixes, key_missing = _resolve_fields(entity_keys, probe["columns"], "entity key")
        resolved_measures, measure_fixes, skipped_measures = _resolve_fields(
            measures, probe["columns"], "measure")
    except ValueError as exc:
        return {"error": str(exc)}
    if key_missing:
        hint = (
            "; table-qualified spellings are not resolved — pass the projected output alias"
            if any(len(_identifier_parts(name)) > 1 for name in key_missing) else ""
        )
        return {"error": (
            f"unknown entity key field(s): {', '.join(key_missing)}; "
            f"source columns: {', '.join(probe['columns'])}{hint}"
        )}
    tolerance_notes = intake_notes + [
        f"Input tolerance: {role} {requested!r} was read as projected alias "
        f"{actual!r} ({why})."
        for requested, actual, why, role in (*key_fixes, *measure_fixes)
    ]
    tolerance_notes.extend(
        f"Measure {name!r} is NOT in the source projection and was NOT audited — "
        "project it in source_sql and re-run to audit it."
        for name in skipped_measures
    )
    if source != raw_source:
        tolerance_notes.append(
            "Input tolerance: a trailing statement terminator was stripped from source_sql."
        )

    source_cte = sa.text(source).columns(
        *(sa.column(name) for name in probe["columns"])
    ).cte("_grain_source")
    key_columns = [source_cte.c[name] for name in keys]
    contrib_columns = [*key_columns, sa.func.count().label("_grain_n")]
    for index, measure in enumerate(resolved_measures):
        column = source_cte.c[measure]
        contrib_columns.extend([
            sa.func.count(column).label(f"_grain_m{index}_rows"),
            sa.func.count(sa.distinct(column)).label(f"_grain_m{index}_values"),
        ])
    contribution = sa.select(*contrib_columns).group_by(*key_columns).cte("_grain_contrib")
    ranked = sa.select(
        contribution.c._grain_n,
        sa.func.row_number().over(order_by=contribution.c._grain_n).label("_grain_rank"),
        sa.func.count().over().label("_grain_total"),
    ).cte("_grain_ranked")

    summary_columns = [
        sa.select(sa.func.count()).select_from(source_cte).scalar_subquery().label("row_count"),
        sa.select(sa.func.count()).select_from(contribution).scalar_subquery().label("entity_count"),
        sa.select(sa.func.count()).select_from(contribution)
        .where(contribution.c._grain_n > 1).scalar_subquery().label("repeated_entity_count"),
        sa.select(sa.func.count()).select_from(source_cte)
        .where(sa.or_(*(column.is_(None) for column in key_columns)))
        .scalar_subquery().label("null_key_rows"),
        sa.select(sa.func.min(contribution.c._grain_n)).scalar_subquery().label("min_rows_per_entity"),
        sa.select(ranked.c._grain_n)
        .where(ranked.c._grain_rank == sa.cast((ranked.c._grain_total + 1) / 2, sa.Integer))
        .scalar_subquery().label("p50_rows_per_entity"),
        sa.select(ranked.c._grain_n)
        .where(ranked.c._grain_rank == sa.cast((ranked.c._grain_total * 95 + 99) / 100, sa.Integer))
        .scalar_subquery().label("p95_rows_per_entity"),
        sa.select(sa.func.max(contribution.c._grain_n)).scalar_subquery().label("max_rows_per_entity"),
    ]
    for index in range(len(resolved_measures)):
        rows = contribution.c[f"_grain_m{index}_rows"]
        values = contribution.c[f"_grain_m{index}_values"]
        summary_columns.extend([
            sa.select(sa.func.coalesce(sa.func.sum(rows), 0)).scalar_subquery()
            .label(f"_grain_m{index}_non_null_rows"),
            sa.select(sa.func.count()).select_from(contribution).where(rows > 0).scalar_subquery()
            .label(f"_grain_m{index}_non_null_entities"),
            sa.select(sa.func.count()).select_from(contribution).where(rows > 1).scalar_subquery()
            .label(f"_grain_m{index}_repeated_entities"),
            sa.select(sa.func.count()).select_from(contribution).where(values > 1).scalar_subquery()
            .label(f"_grain_m{index}_multi_value_entities"),
        ])
    summary_sql, summary_params = _compile(sa.select(*summary_columns))
    summary = query_readonly(
        url, summary_sql, summary_params, timeout_sec=_AUDIT_TIMEOUT_SEC
    )
    if "error" in summary:
        return summary
    stats = summary["rows"][0]

    witness = sa.select(
        *(contribution.c[name] for name in keys),
        contribution.c._grain_n.label("contribution_rows"),
    ).where(contribution.c._grain_n > 1).order_by(
        contribution.c._grain_n.desc()
    ).limit(_MAX_WITNESSES)
    witness_sql, witness_params = _compile(witness)
    witness_result = query_readonly(
        url, witness_sql, witness_params, timeout_sec=_AUDIT_TIMEOUT_SEC
    )
    if "error" in witness_result:
        return witness_result

    measure_stats: list[dict[str, Any]] = []
    for index, measure in enumerate(resolved_measures):
        prefix = f"_grain_m{index}_"
        measure_stats.append({
            "measure": measure,
            "non_null_rows": stats.pop(prefix + "non_null_rows"),
            "non_null_entities": stats.pop(prefix + "non_null_entities"),
            "repeated_entities": stats.pop(prefix + "repeated_entities"),
            "multi_value_entities": stats.pop(prefix + "multi_value_entities"),
        })

    row_count = int(stats["row_count"] or 0)
    entity_count = int(stats["entity_count"] or 0)
    fanout = row_count > entity_count
    notes = list(tolerance_notes)
    if fanout:
        notes.append(
            "The declared answer entity repeats in the source, so row-level and entity-level "
            "aggregates are not equivalent."
        )
    else:
        notes.append("The source contributes at most one row per logical entity.")
    if stats["null_key_rows"]:
        notes.append("Some source rows have NULL entity keys; verify whether they belong in the population.")
    for measure in measure_stats:
        if measure["repeated_entities"]:
            notes.append(
                f"Measure {measure['measure']!r} repeats across entity rows; its row count and entity count differ."
            )
        if measure["multi_value_entities"]:
            notes.append(
                f"Measure {measure['measure']!r} has multiple values within some entities; "
                "a within-entity aggregation stage may be required."
            )

    required_action = (
        "DIAGNOSTIC: the declared entity repeats. If the requested result counts entities or "
        "asks whether each entity ever satisfies a condition, use COUNT(DISTINCT ...), EXISTS, "
        "or a one-row-per-entity stage. If the requested result instead SUMs/AVGs event or "
        "measurement values (especially an explicit hint formula), repeated rows may be the "
        "intended grain; preserve that row-grain arithmetic."
        if fanout else
        "The source already contributes at most one row per declared answer entity."
    )
    return {
        "audit_status": "fanout_detected" if fanout else "one_row_per_entity",
        "fanout_detected": fanout,
        "required_action": required_action,
        "entity_keys": keys,
        **stats,
        "row_to_entity_ratio": round(row_count / entity_count, 3) if entity_count else None,
        "contribution_witnesses": witness_result["rows"],
        "measures": measure_stats,
        "notes": notes,
    }

import json


def _split_outside_quotes(text: str) -> list[str]:
    """Split on commas that sit outside quoted identifiers, so `"a,b"` stays one alias."""
    items: list[str] = []
    current: list[str] = []
    closing = ""
    for char in text:
        if closing:
            current.append(char)
            if char == closing:
                closing = ""
        elif char in "\"`[":
            current.append(char)
            closing = "]" if char == "[" else char
        elif char == ",":
            items.append("".join(current))
            current = []
        else:
            current.append(char)
    items.append("".join(current))
    return items


def _grain_alias_list(value: object) -> list[str]:
    """Read aliases written as a JSON array, a bracketed or comma-separated string, or one name.

    This tool's served schema (pinned in tests/mcp_contract_baseline.json) advertises
    `entity_key` and `measure_aliases` as plain strings, so models hand lists over in
    string clothing — whichever shape their tool-call habit suggests. The comma split is
    quote-aware (an alias literally containing a comma survives when quoted); per-item
    quoting is otherwise left alone — `tools.grain` resolves each name against the actual
    projection and reports what it did.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items: list[str] = [str(item) for item in value]
    else:
        text = str(value).strip()
        if not text:
            return []
        items = None
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            items = (
                [str(item) for item in parsed] if isinstance(parsed, list)
                else _split_outside_quotes(text[1:-1])
            )
        if items is None:
            items = _split_outside_quotes(text)
    return [item.strip() for item in items if str(item).strip()]


def audit_measure_grain(
    db_id: str,
    source_sql: str,
    entity_key: str,
    measure_aliases: str = "",
) -> dict:
    """Compare source rows with the logical entities they represent.

    Args:
        db_id: Registered database identifier, injected by the agent runtime.
        source_sql: Unaggregated single `SELECT`/`WITH` query that projects
            `answer_entity_id` and optional measures under unique aliases.
        entity_key: One projected output alias identifying the answer entity. Quoted
            or misspelled aliases are mapped back; a `table.column` spelling is NOT —
            pass the alias the projection defines.
        measure_aliases: Optional projected aliases, comma-separated or a JSON array.
            An alias missing from the projection is skipped and named in `notes`, not
            an error. For cohort conditions, project `CASE WHEN ... THEN 1 END`.

    Returns:
        dict: Exact row/entity contribution statistics, witnesses, and notes, or
        an `error` field when the database or input is invalid.
    """
    from ..server import _db_url

    url = _db_url(db_id)
    if not url:
        return {"error": f"db_id '{db_id}' not registered."}
    return measure_grain(
        url, source_sql, _grain_alias_list(entity_key), _grain_alias_list(measure_aliases)
    )
