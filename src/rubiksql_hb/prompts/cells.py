"""Render-matrix coverage: the cells a release still lacks, and hints to add them.

``matrix.py`` (C2) owns the code cells and the loader of ``store/matrix`` data cells. This
module adds what the gate needs on top: extra code cells that push every root spec shape
through both arms of each ``if``, the zero and two-or-more iteration classes of each
``each`` and both outcomes of each ``where``; and a coverage report whose gaps carry a hint
naming the entry, the parameter and the value class that would close them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .matrix import Cell, CellRender, cells, render_cells
from .spec import Cmp, Composer, Concat, Coverage, Each, Eq, If, Join, ListLit, Not, Param, Part, Seq, Truthy
from .store import PromptStoreError, Store

__all__ = [
    "EXTRA_CODE_CELLS",
    "CoverageReport",
    "Gap",
    "all_cells",
    "coverage_report",
]

_TRICKY = "{braces} \\backslash \\n ünïcødé 中文 {{double}} {0.__class__} {x[0]}"


def _candidate(error: str = "", group_size: int = 1, group_label: str = "A", sql: str = "SELECT 1", n_rows: int = 1, preview: str = "| 1 |") -> dict[str, Any]:
    return {"error": error, "group_size": group_size, "group_label": group_label, "sql_fenced": f"```sql\n{sql}\n```", "n_rows": n_rows, "preview": preview}


#: Cells beyond ``matrix.CODE_CELLS``: loop truncation at ``max``, single-element loops, a
#: rules block with braces, a second exact-only synonym list, and the small-int edges of the
#: counters. Together with the matrix cells every root spec shape and every block of the
#: one-document layout reaches full coverage; each root cell has its one-document twin.
_EXTRA_CELLS: tuple[Cell, ...] = (
    Cell("candidate.nudge.time", "mins_1", {"mins": 1}),
    Cell("candidate.nudge.converge", "n_1", {"n": 1}),
    Cell("candidate.note.repeat_query", "n_1", {"n": 1}),
    Cell("shared.synonyms", "hits_two_exact", {"hits": [{"syn": "revenue", "target": "sales.amount", "near": False}, {"syn": "client", "target": "customers.name", "near": False}]}),
    Cell("shared.synonyms", "hits_two_near", {"hits": [{"syn": "revenue", "target": "sales.amount", "near": True}, {"syn": "client", "target": "customers.name", "near": True}]}),
    Cell("shared.rule_hooks", "body_empty", {"body": ""}),
    Cell("shared.grain.entity_note", "pks_composite", {"table_name": "orders", "expression": "COUNT(*)", "pks": "order_id, line_no"}),
    Cell(
        "judge",
        "pool_17_truncated",
        {"question": "How many?", "synonyms": "", "candidates": [_candidate(group_size=17, group_label="A", sql=f"SELECT {index}") for index in range(17)], "rules": ""},
    ),
    Cell(
        "judge",
        "pool_rules_braces",
        {"question": "How many?", "synonyms": "", "candidates": [_candidate(group_size=2, group_label="A"), _candidate(group_size=2, group_label="A")], "rules": _TRICKY},
    ),
    Cell("judge.note.literal_guard", "novel_two", {"novel": "'A', 'B'"}),
    Cell("reviewer", "many_rows", {"question": "Which?", "sql": "SELECT name FROM t", "row_count": 4096, "preview": "| a |\n| b |"}),
)
EXTRA_CODE_CELLS: tuple[Cell, ...] = _EXTRA_CELLS


def all_cells(store: Store | None = None, entry: str | None = None) -> tuple[Cell, ...]:
    """Every render-matrix cell the gate runs: matrix code cells, the extra code cells, the store's data cells.

    Args:
        store (Store | None): Store whose ``matrix/`` data cells are included; ``None`` gives code cells only.
        entry (str | None): Restrict to one entry.

    Returns:
        tuple[Cell, ...]: Cells in table order; matrix cells first, then the extra code cells, then data cells.

    Raises:
        PromptStoreError: If an extra code cell or a data cell repeats an ``(entry, name)`` already defined.
    """
    combined = list(cells(store))
    seen = {(cell.entry, cell.name) for cell in combined}
    for cell in EXTRA_CODE_CELLS:
        if (cell.entry, cell.name) in seen:
            raise PromptStoreError(f"matrix cell {cell.entry}/{cell.name} is defined twice")
        seen.add((cell.entry, cell.name))
        combined.append(cell)
    if entry is not None:
        combined = [cell for cell in combined if cell.entry == entry]
    return tuple(combined)


# ---------------------------------------------------------------------------------------
# Coverage with hints


@dataclass(frozen=True)
class Gap:
    """One unmet structural obligation with the hint that would close it.

    Attributes:
        row (str): Spec row whose node is uncovered.
        path (str): Node path inside that row.
        kind (str): ``"if"``, ``"each"`` or ``"where"``.
        outcome (str): The missing outcome: ``then``/``else``, ``0``/``2+``, ``true``/``false``.
        entries (tuple[str, ...]): Entries whose render reaches the row.
        driver (str): The parameter path the node tests or iterates, as written in the spec.
        hint (str): Human-readable instruction naming an entry, the parameter and the value class to add.
    """

    row: str
    path: str
    kind: str
    outcome: str
    entries: tuple[str, ...]
    driver: str
    hint: str

    def line(self) -> str:
        """Spell the gap the way ``Coverage.missing`` does, plus the hint.

        Returns:
            str: ``"<row> <path> <kind>:<outcome> — <hint>"``.
        """
        return f"{self.row} {self.path} {self.kind}:{self.outcome} — {self.hint}"


@dataclass(frozen=True)
class CoverageReport:
    """Structural coverage of a release's spec rows under a set of cells."""

    renders: tuple[CellRender, ...]
    gaps: tuple[Gap, ...] = field(default_factory=tuple)

    @property
    def complete(self) -> bool:
        """Whether every obligation of every spec row was met.

        Returns:
            bool: ``True`` when there are no gaps.
        """
        return not self.gaps


def _find_node(node: Any, path: str) -> Any:
    if getattr(node, "path", None) == path:
        return node
    if isinstance(node, (Seq, ListLit, Concat)):
        for item in node.items:
            found = _find_node(item, path)
            if found is not None:
                return found
    elif isinstance(node, Part):
        for _name, arg in node.args:
            found = _find_node(arg, path)
            if found is not None:
                return found
    elif isinstance(node, If):
        return _find_node(node.then, path) or _find_node(node.else_, path)
    elif isinstance(node, Join):
        return _find_node(node.items, path)
    elif isinstance(node, Each):
        return _find_node(node.body, path)
    return None


def _spell(param: Param) -> str:
    return ".".join(param.segments)


def _test_driver(test: Any) -> tuple[str, str, str]:
    """Return (param path, condition when true, condition when false) for a test."""
    if isinstance(test, Truthy):
        path = _spell(test.operand)
        return path, f"{path} truthy (non-empty / non-zero / true)", f"{path} falsy (empty string, 0, false or empty list)"
    if isinstance(test, Not):
        path, when_true, when_false = _test_driver(test.test)
        return path, when_false, when_true
    if isinstance(test, Eq):
        path = _spell(test.operand)
        return path, f"{path} == {test.scalar!r}", f"{path} != {test.scalar!r}"
    if isinstance(test, Cmp):
        path = _spell(test.operand)
        left = f"len({path})" if test.length else path
        ops = {"gt": (">", "<="), "ge": (">=", "<"), "lt": ("<", ">="), "le": ("<=", ">")}
        yes, no = ops[test.op]
        return path, f"{left} {yes} {test.value}", f"{left} {no} {test.value}"
    return "?", "the test holds", "the test fails"


def _entries_reaching(release: Any, row: str) -> tuple[str, ...]:
    reaching = []
    for entry in release.entries:
        stack = [release.renderable(entry)]
        seen = set()
        while stack:
            item = stack.pop()
            if item.name in seen:
                continue
            seen.add(item.name)
            if item.name == row:
                reaching.append(entry)
                break
            if isinstance(item, Composer):
                stack.extend(item.parts.values())
    return tuple(reaching)


def _gap(release: Any, composer: Composer, path: str, kind: str, outcome: str) -> Gap:
    node = _find_node(composer.spec.root, path)
    entries = _entries_reaching(release, composer.name)
    where = f"render {' or '.join(entries) if entries else composer.name}"
    if isinstance(node, If):
        driver, when_true, when_false = _test_driver(node.test)
        hint = f"{where} with {when_true if outcome == 'then' else when_false} (reaches the {outcome} arm at {path})"
    elif isinstance(node, Each) and kind == "where":
        driver, when_true, when_false = _test_driver(node.where)
        hint = f"{where} with an element of {_spell(node.source)} where {when_true if outcome == 'true' else when_false}"
    elif isinstance(node, Each):
        driver = _spell(node.source)
        if outcome == "0":
            hint = f"{where} with {driver} = [] (zero iterations at {path})"
        else:
            hint = f"{where} with {driver} holding at least 2 elements (max {node.max}) so the loop at {path} runs {outcome} times"
    else:
        driver = "?"
        hint = f"{where} so that {path} reaches {kind}:{outcome}"
    if composer.name not in entries and entries:
        hint += f"; the node lives in part {composer.name}, whose arguments the entry supplies"
    return Gap(composer.name, path, kind, outcome, entries, driver, hint)


def coverage_report(release: Any, matrix: Iterable[Cell] | None = None, *, store: Store | None = None) -> CoverageReport:
    """Render the matrix against a release and list every unmet structural obligation with a hint.

    Args:
        release (Release): The release under test.
        matrix (Iterable[Cell] | None): Cells to render; :func:`all_cells` of ``store`` when omitted.
        store (Store | None): Store whose data cells join the code cells when ``matrix`` is omitted.

    Returns:
        CoverageReport: Cell renders plus the gaps, each carrying the entry, parameter and value class to add.
    """
    cell_list = tuple(matrix) if matrix is not None else all_cells(store)
    coverage = Coverage()
    renders = render_cells(release, cell_list, coverage)
    composers = {composer.name: composer for composer in release.composers()}
    gaps = []
    for line in coverage.missing(composers.values()):
        row, path, kind_outcome = line.split(" ", 2)
        kind, outcome = kind_outcome.split(":", 1)
        gaps.append(_gap(release, composers[row], path, kind, outcome))
    return CoverageReport(renders, tuple(gaps))
