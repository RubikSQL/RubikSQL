"""The render matrix: fixed argument cells that exercise every entry, in code and in the store.

The gate renders every cell with the interpreter and with HeavenBase and requires
byte-equality; structural coverage of every ``if``, ``each`` and ``where`` comes from the
union of cells. Code cells live here; data cells are append-only files under
``store/matrix/<entry>/<cell>.json`` with ``{"schema": "cell/1", "entry", "params"}``.
Several cells carry braces, backslashes and non-ASCII text on purpose: a value is inserted
once and never parsed again. Every root cell has a twin for the one-document layout
(:func:`one_document_cells`): the entry that provides the same surface, with the same
arguments, so both layouts are exercised by the same inputs. Every cell of the ``candidate``
document is rendered again by each candidate slot's document (:func:`slot_cells`), and each
generator document has cells for both arms of its blocks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from . import entries as E
from .canon import sha256_hex
from .spec import Coverage, RenderContext
from .store import SCHEMA_CELL, PromptStoreError, Store

__all__ = ["CODE_CELLS", "Cell", "CellRender", "cells", "data_cells", "one_document_cells", "render_cells", "slot_cells"]

_CELL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_TRICKY = "{braces} \\backslash \\n ünïcødé 中文 {{double}} {0.__class__} {x[0]}"


@dataclass(frozen=True)
class Cell:
    """One render-matrix cell: an entry plus the arguments to render it with."""

    entry: str
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    origin: str = "code"


@dataclass(frozen=True)
class CellRender:
    """The result of rendering one cell."""

    cell: Cell
    digest: str
    length: int


def _candidate(error: str = "", group_size: int = 1, group_label: str = "A", sql: str = "SELECT 1", n_rows: int = 1, preview: str = "| 1 |") -> dict[str, Any]:
    return {"error": error, "group_size": group_size, "group_label": group_label, "sql_fenced": f"```sql\n{sql}\n```", "n_rows": n_rows, "preview": preview}


def _static_cells() -> list[Cell]:
    return [Cell(entry.name, "static") for entry in E.ENTRIES.values() if not entry.params]


#: root entries whose cells also render a role document, and the arguments of its other message.
def slot_cells(matrix: Iterable[Cell]) -> tuple[Cell, ...]:
    """The cells of the ``candidate`` document once more for every candidate slot's document.

    A slot document takes the candidate document's arguments, so the gate renders each slot a release carries
    with every input the shared document is rendered with; a release without the slot skips its cells.

    Args:
        matrix (Iterable[Cell]): Cells; those of the ``candidate`` entry are copied.

    Returns:
        tuple[Cell, ...]: One copy per cell and slot, same names and origins.
    """
    candidate = [cell for cell in matrix if cell.entry == "candidate"]
    return tuple(Cell(E.slot_entry(slot), cell.name, dict(cell.params), cell.origin) for slot in range(1, E.CANDIDATE_SLOTS + 1) for cell in candidate)


#: Cells of the generator documents: realistic arguments, both arms of every block, and data that carries braces.
_GENERATOR_CELLS: tuple[Cell, ...] = (
    Cell("gen.rules_general.hint_map", "plain", {"theme": "ratio_percentage", "n_dbs": 3, "hints": "- (db=shop) ratio = SUM(a) / SUM(b)\n- (db=school) " + _TRICKY}),
    Cell("gen.rules_general.map", "plain", {"pattern": "['CAST_DIVISION', 'PCT_100']", "n_examples": 2, "n_dbs": 2, "examples": "- question: What is the ratio?\n  hint: ratio = a / b\n  gold SQL: SELECT 1\n\n- question: " + _TRICKY}),
    Cell("gen.rules_general.reduce", "plain", {"existing": "- Use ORDER BY ... LIMIT 1.", "candidates": "- [t] " + _TRICKY, "max_rules": 45}),
    Cell("gen.rules_judge.map", "plain", {"db_id": "california_schools", "questions": "- question: How many?\n  chosen SQL: SELECT 1\n    rejected: SELECT 2\n  judge reason: " + _TRICKY}),
    Cell("gen.rules_judge.reduce", "plain", {"db_id": "california_schools", "existing": "- A rule.", "candidates": "- [r] " + _TRICKY, "max_rules": 12}),
    Cell("gen.rules_hints", "whole", {"whole_corpus": True, "db_id": "california_schools", "hints": "- [1] (3×) charter refers to = 1", "gold": "", "max_rules": 12}),
    Cell("gen.rules_hints", "chunk_gold", {"whole_corpus": False, "db_id": "california_schools", "hints": "- [1] (1×) " + _TRICKY, "gold": "- Q: How many?\n  hint: h\n  gold SQL: SELECT 1", "max_rules": 8}),
    Cell("gen.synonyms", "plain", {"db_id": "california_schools", "schema": "schools.CDSCode\nschools.School Name", "notes": "- (4×) " + _TRICKY}),
    Cell("gen.rule_triggers", "plain", {"rule": '{"title": "percentage_of_value", "desc": "' + _TRICKY + '"}'}),
    Cell("gen.profile.table", "agreement", {"db_map": "# Database map (for contrast)\n- schools: CDSCode", "agreement": "# Measured column agreement for schools\n- CDSCode: schools vs frpm -> IDENTICAL COPIES", "detail": "# FOCUS table: schools\n- CDSCode: " + _TRICKY}),
    Cell("gen.profile.table", "plain", {"db_map": "# Database map (for contrast)\n- schools: CDSCode", "agreement": "", "detail": "# FOCUS table: schools"}),
    Cell("gen.profile.db", "agreement", {"db_map": "# Database map (for contrast)\n- schools: CDSCode", "agreement": "# Measured column agreement\n- CDSCode: schools vs frpm", "profiles": "- schools: " + _TRICKY}),
    Cell("gen.profile.db", "plain", {"db_map": "# Database map (for contrast)", "agreement": "", "profiles": "- schools: One row per school."}),
    Cell("gen.profile.columns", "first", {"profile": "Column `CDSCode` of schools: TEXT. " + _TRICKY, "columns": "CDSCode, School Name", "violation": ""}),
    Cell("gen.profile.columns", "repair", {"profile": "Column `CDSCode` of schools: TEXT.", "columns": "CDSCode", "violation": "the reply contains no JSON object"}),
    Cell("gen.examples.transfer", "demos", {"examples": '### Example 1\nOutput for target database "shop":\n{"question": "' + _TRICKY + '"}', "query": '{\n  "db_id": "a"\n}', "schema": "Target database: shop"}),
    Cell("gen.examples.transfer", "no_demos", {"examples": "", "query": '{"db_id": "a"}', "schema": "Target database: " + _TRICKY}),
)


_DOCUMENT_CELLS: tuple[Cell, ...] = tuple(
    [
        Cell("candidate", "max_rows_64", {"max_rows": 64}),
        Cell("candidate", "max_rows_200", {"max_rows": 200}),
        Cell("candidate.nudge.time", "mins_20", {"mins": 20}),
        Cell("candidate.nudge.time", "mins_0", {"mins": 0}),
        Cell("candidate.nudge.converge", "n_50", {"n": 50}),
        Cell("candidate.note.empty", "n_2", {"n": 2}),
        Cell("candidate.note.empty", "n_1", {"n": 1}),
        Cell("candidate.note.repeat_query", "n_3", {"n": 3}),
        Cell("candidate.note.result_shape", "limit_3", {"limit": 3, "first": 1, "second": 2}),
        Cell("shared.synonyms", "hits_none", {"hits": []}),
        Cell("shared.synonyms", "hits_one_exact", {"hits": [{"syn": "revenue", "target": "sales.amount", "near": False}]}),
        Cell(
            "shared.synonyms",
            "hits_many_mixed",
            {"hits": [{"syn": "revenue", "target": "sales.amount", "near": False}, {"syn": "client", "target": "customers.name", "near": True}, {"syn": "city", "target": "addresses.city", "near": True}]},
        ),
        Cell("shared.synonyms", "hits_braces_unicode", {"hits": [{"syn": _TRICKY, "target": _TRICKY, "near": True}]}),
        Cell("shared.rule_hooks", "body_plain", {"body": "- rule one\n- rule two"}),
        Cell("shared.rule_hooks", "body_braces_unicode", {"body": _TRICKY}),
        Cell("shared.hint_conflict", "braces_unicode", {"kind": "year", "hint_literal": _TRICKY, "question_literal": _TRICKY}),
        Cell("shared.count_column", "braces_unicode", {"column": _TRICKY}),
        Cell("shared.grain.entity_note", "plain", {"table_name": "schools", "expression": "COUNT(DISTINCT frpm.CDSCode)", "pks": "CDSCode"}),
        Cell("shared.grain.entity_note", "braces_unicode", {"table_name": _TRICKY, "expression": "it's {a} \\ 'quote'", "pks": "id, sub_id"}),
        Cell("judge", "pool_empty", {"question": "How many?", "synonyms": "", "candidates": [], "rules": ""}),
        Cell("judge", "pool_1ok", {"question": "How many?", "synonyms": "", "candidates": [_candidate()], "rules": ""}),
        Cell(
            "judge",
            "pool_5_unanimous",
            {"question": "How many?", "synonyms": "\n\n---\n- \"count\" refers to n", "candidates": [_candidate(group_size=5, group_label="A") for _ in range(5)], "rules": ""},
        ),
        Cell(
            "judge",
            "pool_3ok_2groups_1failed",
            {
                "question": "How many?",
                "synonyms": "",
                "candidates": [_candidate(group_size=2, group_label="A"), _candidate(error="timeout after 60s"), _candidate(group_size=2, group_label="A"), _candidate(group_label="B", sql="SELECT 2", preview="| 2 |")],
                "rules": "- keep the asked columns",
            },
        ),
        Cell("judge", "pool_2failed_1ok", {"question": "How many?", "synonyms": "", "candidates": [_candidate(error="syntax error"), _candidate(error="no such table"), _candidate()], "rules": ""}),
        Cell("judge", "pool_all_failed", {"question": "How many?", "synonyms": "", "candidates": [_candidate(error="e1"), _candidate(error="e2")], "rules": ""}),
        Cell(
            "judge",
            "pool_braces_unicode",
            {"question": _TRICKY, "synonyms": _TRICKY, "candidates": [_candidate(group_label=_TRICKY, sql=_TRICKY, preview=_TRICKY), _candidate(error=_TRICKY)], "rules": _TRICKY},
        ),
        Cell("judge.note.literal_guard", "novel_one", {"novel": "'Volkan Başa'"}),
        Cell("judge.note.literal_guard", "novel_braces", {"novel": _TRICKY}),
        Cell("reviewer", "plain", {"question": "How many?", "sql": "SELECT COUNT(*) FROM t", "row_count": 1, "preview": "| 3 |"}),
        Cell("reviewer", "braces_unicode", {"question": _TRICKY, "sql": _TRICKY, "row_count": 0, "preview": _TRICKY}),
    ]
)

_FRAME_CELLS: tuple[Cell, ...] = (
    Cell(
        "candidate",
        "frames_braces_unicode",
        {"max_rows": 64, "question": _TRICKY, "synonyms": "\n\n---\n## Reference: how other questions worded things\n- \"count\" refers to n", "rule_hooks": "\n\n" + _TRICKY},
    ),
)

#: Code-defined cells; every entry has at least one, parameterised entries have several.
CODE_CELLS: tuple[Cell, ...] = tuple(
    _static_cells()
    + list(_DOCUMENT_CELLS)
    + list(_FRAME_CELLS)
    + list(slot_cells(_DOCUMENT_CELLS + _FRAME_CELLS))
    + list(_GENERATOR_CELLS)
)


# Candidate cells exercise both the system and first-message frame.
CODE_CELLS = tuple(Cell(cell.entry, cell.name, {"question": "How many?", "synonyms": "", "rule_hooks": "", **cell.params}, cell.origin) if cell.entry == "candidate" or cell.entry.startswith("candidate.s") else cell for cell in CODE_CELLS)


def data_cells(store: Store) -> tuple[Cell, ...]:
    """Load the append-only data cells from the package store's ``matrix/`` directory.

    Args:
        store (Store): The store; only its package layer is read.

    Returns:
        tuple[Cell, ...]: Cells sorted by entry and name, with ``origin="data"``.

    Raises:
        PromptStoreError: If a cell file is malformed, names an unknown entry or an invalid cell name.
    """
    directory = store.package / "matrix"
    if not directory.is_dir():
        return ()
    found = []
    for path in sorted(directory.glob("*/*.json")):
        relative = f"matrix/{path.parent.name}/{path.name}"
        loaded = store.require(relative, f"cell {path.stem}", package_only=True)
        data = loaded.data
        if not isinstance(data, Mapping) or data.get("schema") != SCHEMA_CELL or not isinstance(data.get("params"), Mapping):
            raise PromptStoreError(f"{relative}: expected {{'schema': {SCHEMA_CELL!r}, 'entry', 'params': {{...}}}}")
        if data.get("entry") != path.parent.name or data["entry"] not in E.ENTRIES:
            raise PromptStoreError(f"{relative}: entry {data.get('entry')!r} must equal the directory name and be a known entry")
        if not _CELL_NAME_RE.fullmatch(path.stem):
            raise PromptStoreError(f"{relative}: cell name must match {_CELL_NAME_RE.pattern}")
        found.append(Cell(data["entry"], path.stem, dict(data["params"]), "data"))
    return tuple(found)


def cells(store: Store | None = None, entry: str | None = None) -> tuple[Cell, ...]:
    """Every matrix cell: code cells plus the store's data cells.

    Args:
        store (Store | None): Store whose data cells are included; ``None`` gives code cells only.
        entry (str | None): Restrict to one entry.

    Returns:
        tuple[Cell, ...]: Cells in table order.

    Raises:
        PromptStoreError: If a data cell repeats a code cell's ``(entry, name)``.
    """
    combined = list(CODE_CELLS)
    if store is not None:
        seen = {(cell.entry, cell.name) for cell in combined}
        for cell in data_cells(store):
            if (cell.entry, cell.name) in seen:
                raise PromptStoreError(f"matrix cell {cell.entry}/{cell.name} is defined both in code and in the store")
            combined.append(cell)
    if entry is not None:
        combined = [cell for cell in combined if cell.entry == entry]
    return tuple(combined)


def render_cells(release: Any, matrix: Iterable[Cell], coverage: Coverage | None = None) -> tuple[CellRender, ...]:
    """Render cells against a release, digesting each output and accumulating coverage.

    Args:
        release (Release): The release to render with.
        matrix (Iterable[Cell]): Cells to render; cells of entries the release lacks are skipped.
        coverage (Coverage | None): Structural coverage accumulator shared across cells.

    Returns:
        tuple[CellRender, ...]: One result per rendered cell with a 16-hex digest and the character length.
    """
    results = []
    for cell in matrix:
        if not release.has(cell.entry):
            continue
        ctx = RenderContext(coverage=coverage)
        chunks = release.renderable(cell.entry).render_chunks(ctx, cell.params)
        text = "".join(piece for piece, _src in chunks)
        results.append(CellRender(cell, sha256_hex(text)[:16], len(text)))
    return tuple(results)
