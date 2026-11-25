"""Detect stored count columns whose documented definition already applies a threshold.

A column documented as "number of <members> whose/with <measure> >= N" counts the members that
meet the threshold; a value description such as "N indicates ..." defines no such count. Comparing that count with N compares a head count with the measure, and an
entity "with members scoring >= N" is ``<column> > 0``. The detector reads the documented column
description that schema tools already return; it never reads data, a question or a reference.
"""
from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any

__all__ = ["IMPLEMENTATION_VERSION", "threshold_count_columns"]

IMPLEMENTATION_VERSION = "count-columns-2"

_COMPARATOR = (r"(?:>=|=>|\bat\s+least\b|\bgreater\s+(?:than\s+)?or\s+equal(?:\s+to)?\b|\bmore\s+than\b"
               r"|\babove\b|\bover\b)")
#: "number of <members> whose/with/that/who <measure> <comparator> N": a count of members meeting a threshold.
_THRESHOLD_COUNT = re.compile(
    r"\b(?:number|count)\s+of\s+(?:[\w-]+\s+){1,5}?(?:whose|with|that|who|having)\b[^.;,\n]{0,80}?"
    + _COMPARATOR + r"\s*\$?(\d[\d,]*(?:\.\d+)?)", re.I)
#: A value description ("more than 50 indicates a large firm") explains a value; it defines no count.
_VALUE_NOTE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:\w+\s+){0,2}?(?:indicates?|means?|refers?\s+to|is\s+considered)\b", re.I)


def _column_rows(tool_name: str, result: Mapping[str, Any]) -> list[tuple[str, str]]:
    if tool_name == "get_col_info":
        rows = result.get("columns") if isinstance(result.get("columns"), list) else [result]
        return [(str(row.get("name") or ""), str(row.get("profile_desc") or "")) for row in rows if isinstance(row, Mapping)]
    tables = result.get("tables") if isinstance(result.get("tables"), list) else [result]
    return [(str(column.get("name") or ""), str(column.get("desc") or ""))
            for table in tables if isinstance(table, Mapping)
            for column in table.get("columns") or [] if isinstance(column, Mapping)]


def threshold_count_columns(tool_name: str, result: object) -> list[str]:
    """Name the columns of a schema tool result whose description is a thresholded count.

    Args:
        tool_name (str): ``get_db_info``, ``get_tab_info`` or ``get_col_info``; any other tool yields nothing.
        result (object): The structured tool result, as the server returns it.

    Returns:
        list[str]: Column names in result order, each once; empty when nothing matches or the result is an error.
    """
    if tool_name not in {"get_db_info", "get_tab_info", "get_col_info"} or not isinstance(result, Mapping) or result.get("error"):
        return []
    names: list[str] = []
    for name, description in _column_rows(tool_name, result):
        first_line = description.strip().splitlines()[0] if description.strip() else ""
        text = description if tool_name != "get_col_info" else first_line
        match = _THRESHOLD_COUNT.search(text)
        if name and name not in names and match and not _VALUE_NOTE.match(text, match.start(1)):
            names.append(name)
    return names
