"""Where an answering MCP server's tool calls spend their time: session queue, worker admission or execution.

Client-side session timing (``rubiksql.session-timing/1``) sees one tool round trip. Under load that round trip mixes
three waits the server can separate: the FIFO session lock, admission to a tool-worker token
(``RUBIKSQL_TOOL_WORKERS``) and the tool body itself (SQL, retrieval, CPU). Each served call notes those three spans here
(:func:`observe`); the counts, sums and maxima per tool, with the peak tokens in use when a body started and the peak
calls already waiting when a call asked for a token, are kept in one JSON record per server process (``rubiksql.tool-timing/1``).

A server its parent may kill without warning keeps the record current instead of writing it at exit: when the
launching ``eval_bird`` names its rate-limit record directory (``RUBIKSQL_RATE_LIMIT_DIR``), :func:`keep_record` places
the record in that directory's :data:`RECORD_SUBDIR` and :func:`observe` rewrites it at most every
:data:`WRITE_INTERVAL_SEC` seconds. :func:`read_records` and :func:`summary` give a launch's ``[tool-timing]`` line.

Stdlib only; recording never changes a tool's result or fails its call.
"""
from __future__ import annotations

import glob
import json
import os
import threading
import time
from collections.abc import Collection, Mapping
from typing import Any

__all__ = ["RECORD_SUBDIR", "SCHEMA", "WRITE_INTERVAL_SEC", "keep_record", "observe", "read_records", "snapshot",
           "summary"]

SCHEMA = "rubiksql.tool-timing/1"
#: The subdirectory of the rate-limit record directory that holds these records (outside its ``*.json`` scan).
RECORD_SUBDIR = "tool_timing"
#: The shortest interval between two rewrites of a kept record.
WRITE_INTERVAL_SEC = 5.0
_SPANS = ("session_wait", "admission_wait", "execution")

_LOCK = threading.Lock()
_TOOLS: dict[str, dict[str, Any]] = {}
_PEAKS: dict[str, Any] = {"borrowed": 0, "waiting": 0, "limit": None}
_KEPT: dict[str, Any] | None = None
_LAST_WRITE = 0.0


def observe(tool: str, *, session_wait: float, admission_wait: float, execution: float,
            borrowed: int, waiting: int, limit: int) -> None:
    """Count one served call's three spans and the limiter load it met at admission.

    Args:
        tool (str): The tool name.
        session_wait (float): Seconds queued on the call's session lock (0 without a session id).
        admission_wait (float): Seconds from asking the limiter for a worker token to the body starting.
        execution (float): Seconds the body ran in its worker thread.
        borrowed (int): Tokens borrowed when the body started, its own included.
        waiting (int): Calls already waiting for a token when the call asked for one.
        limit (int): The limiter's total tokens.
    """
    global _LAST_WRITE
    with _LOCK:
        stats = _TOOLS.setdefault(tool, {"count": 0, **{span: {"sum": 0.0, "max": 0.0} for span in _SPANS}})
        stats["count"] += 1
        for span, seconds in zip(_SPANS, (session_wait, admission_wait, execution)):
            stats[span]["sum"] += seconds
            stats[span]["max"] = max(stats[span]["max"], seconds)
        _PEAKS["borrowed"] = max(_PEAKS["borrowed"], borrowed)
        _PEAKS["waiting"] = max(_PEAKS["waiting"], waiting)
        _PEAKS["limit"] = limit
        now = time.monotonic()
        due = _KEPT is not None and now - _LAST_WRITE >= WRITE_INTERVAL_SEC
        if due:
            _LAST_WRITE = now
    if due:
        _write_kept()


def snapshot() -> dict[str, Any]:
    """This process's counts: ``{"limit", "peak_borrowed", "peak_waiting", "tools"}`` (a copy)."""
    with _LOCK:
        return {"limit": _PEAKS["limit"], "peak_borrowed": _PEAKS["borrowed"], "peak_waiting": _PEAKS["waiting"],
                "tools": json.loads(json.dumps(_TOOLS))}


def keep_record(directory: str | os.PathLike[str], label: str) -> None:
    """Keep this process's record current under ``directory``/:data:`RECORD_SUBDIR` from now on.

    Args:
        directory (str | os.PathLike[str]): The parent's rate-limit record directory.
        label (str): What the process serves, e.g. ``mcp-<db_id>``; the file is ``<label>-<pid>.json``.
    """
    global _KEPT
    _KEPT = {"path": os.path.join(os.fspath(directory), RECORD_SUBDIR, f"{label}-{os.getpid()}.json"),
             "label": label, "started_at": time.time()}


def _write_kept() -> None:
    kept = _KEPT
    if kept is None:
        return
    record = {"schema": SCHEMA, "label": kept["label"], "pid": os.getpid(), "started_at": kept["started_at"],
              "updated_at": time.time(), **snapshot()}
    temporary = f"{kept['path']}.{threading.get_ident()}.tmp"
    try:
        os.makedirs(os.path.dirname(kept["path"]), exist_ok=True)
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
        os.replace(temporary, kept["path"])
    except OSError:
        pass   # the next due call writes again; timing must never fail the call it measures


def read_records(directory: str | os.PathLike[str], *, since: float = 0.0,
                 labels: Collection[str] | None = None) -> list[dict[str, Any]]:
    """The records under ``directory``/:data:`RECORD_SUBDIR` whose server started at or after ``since``.

    Args:
        directory (str | os.PathLike[str]): The rate-limit record directory; absent, it holds nothing.
        since (float): The launch epoch; earlier launches into the same directory left records of their own.
        labels (Collection[str] | None): Only these labels (e.g. ``mcp-<db_id>``); every record when ``None``.

    Returns:
        list[dict[str, Any]]: The readable records, in file-name order.
    """
    records = []
    for path in sorted(glob.glob(os.path.join(os.fspath(directory), RECORD_SUBDIR, "*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError):
            continue
        if (isinstance(record, dict) and record.get("schema") == SCHEMA
                and float(record.get("started_at") or 0.0) >= since
                and (labels is None or record.get("label") in labels)):
            records.append(record)
    return records


def summary(records: list[Mapping[str, Any]]) -> str:
    """One line over all tools of ``records``: calls, mean/max of each span, and the peak limiter load.

    Args:
        records (list[Mapping[str, Any]]): Records from :func:`read_records`.

    Returns:
        str: E.g. ``"1200 calls; session_wait mean 0.10s max 3.2s; ...; peak 64/64 borrowed, 30 waiting"``.
    """
    count, sums, maxima = 0, dict.fromkeys(_SPANS, 0.0), dict.fromkeys(_SPANS, 0.0)
    for record in records:
        for stats in (record.get("tools") or {}).values():
            count += int(stats.get("count", 0))
            for span in _SPANS:
                sums[span] += float(stats.get(span, {}).get("sum", 0.0))
                maxima[span] = max(maxima[span], float(stats.get(span, {}).get("max", 0.0)))
    if not count:
        return "no served tool calls recorded"
    spans = "; ".join(f"{span} mean {sums[span] / count:.2f}s max {maxima[span]:.1f}s" for span in _SPANS)
    borrowed = max(int(record.get("peak_borrowed") or 0) for record in records)
    waiting = max(int(record.get("peak_waiting") or 0) for record in records)
    limits = sorted({record.get("limit") for record in records if record.get("limit")})
    return f"{count} calls; {spans}; peak {borrowed} borrowed of {'/'.join(map(str, limits))}, {waiting} waiting"
