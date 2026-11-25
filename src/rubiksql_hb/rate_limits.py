"""Process-wide counts of provider requests, and of the rate-limit and overload answers among them, for the run logs.

Every transport that talks to a provider notes each request attempt under a lane name (``chat``, ``judge``,
``embedding``, ``profiling``, ``rulegen``, ``transfer``), each attempt the provider answered with a rate limit (HTTP 429,
or an error that says "rate limit": :func:`rate_limited`) and each attempt it left unanswered under load (an HTTP 5xx, a
timeout or a lost connection: :func:`overloaded`). The run logs print the counts as ``[rate-limit]`` lines — the
knowledge build after each stage, ``eval_bird`` at the end of a run, ``eval_parallel`` summed over its database processes
— so the concurrency settings (7 databases x 10 workers by default, for the knowledge build and for answering) can be
tuned from what the provider answered. A rate limit an SDK retries inside one call never reaches this module: the counts
are the attempts the pipeline's own retry ladders saw.

A process that runs as a child of another (an ``eval_bird`` under ``eval_parallel``, the example transfer under the
knowledge build) writes its counts into a record directory (:func:`write_record`, one JSON file per process under
:data:`RECORD_DIR`), and the parent sums the records of its own launch (:func:`read_records`). A child its parent may
kill without warning — the MCP servers of an ``eval_bird``, which answer the knowledge tools and embed their queries —
keeps its record current instead (:func:`keep_record`, rewritten at every count), in the directory the parent names in
:data:`RECORD_ENV`.

Stdlib only, so every script and module can import it without loading HeavenBase.
"""
from __future__ import annotations

import glob
import json
import os
import threading
import time
from collections.abc import Collection, Iterable, Mapping

__all__ = ["RECORD_DIR", "RECORD_ENV", "add", "keep_record", "merge", "overloaded", "rate_limited", "read_records", "request",
           "snapshot", "summary", "write_record"]

#: The record directory's name inside a run directory: a directory, so the per-database result scans (the ``*.json``
#: files of an eval out-dir) never meet it.
RECORD_DIR = ".rate_limits"
#: The variable a parent sets to its record directory for the children that keep their records current (eval_bird for
#: the MCP servers it launches).
RECORD_ENV = "RUBIKSQL_RATE_LIMIT_DIR"
#: The counts of one lane, in the order :data:`_COUNTS` keeps them.
_FIELDS = ("requests", "rate_limited", "overloaded")

_LOCK = threading.Lock()
#: Lane name to ``[requests, rate_limited, overloaded]``.
_COUNTS: dict[str, list[int]] = {}
#: The record :func:`keep_record` keeps current: ``(directory, label, started_at)``.
_KEPT: tuple[str, str, float] | None = None
_KEPT_LOCK = threading.Lock()


def _count(lane: str, field: int) -> None:
    with _LOCK:
        _COUNTS.setdefault(lane, [0] * len(_FIELDS))[field] += 1
    if _KEPT is not None:
        _write_kept()


def request(lane: str) -> None:
    """Count one request attempt.

    Args:
        lane (str): The transport, e.g. ``chat`` or ``embedding``.
    """
    _count(lane, 0)


def rate_limited(lane: str) -> None:
    """Count one attempt of ``lane`` the provider answered with a rate limit (the attempt itself is counted by :func:`request`).

    Args:
        lane (str): The transport the attempt went through.
    """
    _count(lane, 1)


def overloaded(lane: str) -> None:
    """Count one attempt of ``lane`` the provider left unanswered under load: an HTTP 5xx, a timeout or a lost connection.

    Args:
        lane (str): The transport the attempt went through (the attempt itself is counted by :func:`request`).
    """
    _count(lane, 2)


def _pairs(counts: Mapping[str, Mapping[str, int]]) -> Iterable[tuple[str, list[int]]]:
    """Each lane's counts as a list in :data:`_FIELDS` order; a record written before a field existed counts it 0."""
    for lane, pair in counts.items():
        yield lane, [int(pair.get(name) or 0) for name in _FIELDS]


def add(counts: Mapping[str, Mapping[str, int]]) -> None:
    """Add counts to this process's (a child process's, read back with :func:`read_records`).

    Args:
        counts (Mapping[str, Mapping[str, int]]): A :func:`snapshot`, :func:`merge` or :func:`read_records` result.
    """
    with _LOCK:
        for lane, values in _pairs(counts):
            into = _COUNTS.setdefault(lane, [0] * len(_FIELDS))
            for index, value in enumerate(values):
                into[index] += value


def snapshot(*, reset: bool = False) -> dict[str, dict[str, int]]:
    """The counts so far.

    Args:
        reset (bool): Start the counts again from zero (a stage that reports only its own requests).

    Returns:
        dict[str, dict[str, int]]: Lane to ``{"requests", "rate_limited", "overloaded"}``, sorted by lane.
    """
    with _LOCK:
        counts = {lane: dict(zip(_FIELDS, values)) for lane, values in sorted(_COUNTS.items())}
        if reset:
            _COUNTS.clear()
    return counts


def merge(snapshots: Iterable[Mapping[str, Mapping[str, int]]]) -> dict[str, dict[str, int]]:
    """Sum snapshots, lane by lane (the records several processes of one run wrote).

    Args:
        snapshots (Iterable[Mapping[str, Mapping[str, int]]]): Results of :func:`snapshot`.

    Returns:
        dict[str, dict[str, int]]: The summed counts, sorted by lane.
    """
    total: dict[str, list[int]] = {}
    for counts in snapshots:
        for lane, values in _pairs(counts):
            into = total.setdefault(lane, [0] * len(_FIELDS))
            for index, value in enumerate(values):
                into[index] += value
    return {lane: dict(zip(_FIELDS, values)) for lane, values in sorted(total.items())}


def summary(counts: Mapping[str, Mapping[str, int]]) -> str:
    """One line for the run log: the rate-limit share over every lane, then each lane's, with the overloads when any.

    Args:
        counts (Mapping[str, Mapping[str, int]]): A :func:`snapshot` or :func:`merge` result.

    Returns:
        str: E.g. ``429 on 3 of 1200 requests (0.25%): chat 2/900, judge 1/300``; with overloads, e.g. ``429 on 3 of 1200
            requests (0.25%), 5xx/timeout on 2 (0.17%): chat 2/900 (+2 5xx/timeout), judge 1/300``; ``no provider
            request`` when nothing was counted.
    """
    lanes = dict(_pairs(counts))
    requests, limited, over = (sum(values[index] for values in lanes.values()) for index in range(len(_FIELDS)))
    if not requests:
        return "no provider request"
    head = f"429 on {limited} of {requests} requests ({100.0 * limited / requests:.2f}%)"
    if over:
        head += f", 5xx/timeout on {over} ({100.0 * over / requests:.2f}%)"
    parts = ", ".join(f"{lane} {values[1]}/{values[0]}" + (f" (+{values[2]} 5xx/timeout)" if values[2] else "") for lane, values in lanes.items())
    return f"{head}: {parts}"


def write_record(directory: str | os.PathLike[str], label: str, started_at: float) -> str:
    """Write this process's counts into a record directory, for the parent process to sum.

    The file is written whole (a temporary file, then a rename), so a reader never meets half of one.

    Args:
        directory (str | os.PathLike[str]): The record directory (``<run dir>/<RECORD_DIR>``), created when absent.
        label (str): What the process did, e.g. its database; the file is ``<label>-<pid>.json``.
        started_at (float): When the process started its requests, the epoch :func:`read_records` filters on.

    Returns:
        str: The file written.

    Raises:
        OSError: The directory or the file cannot be written.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(os.fspath(directory), f"{label}-{os.getpid()}.json")
    record = {"label": label, "pid": os.getpid(), "started_at": started_at, "finished_at": time.time(), "counts": snapshot()}
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(record, handle)
    os.replace(temporary, path)
    return path


def keep_record(directory: str | os.PathLike[str], label: str) -> None:
    """Keep this process's record in ``directory`` current: rewritten at every count from now on.

    For a child its parent may kill without warning (an MCP server, which ``eval_bird`` kills at its exit), whose
    counts a record written at the end would lose. A write that fails is skipped; the next count writes again.

    Args:
        directory (str | os.PathLike[str]): The record directory (the parent's :data:`RECORD_ENV`).
        label (str): What the process does; the file is ``<label>-<pid>.json``.
    """
    global _KEPT
    _KEPT = (os.fspath(directory), label, time.time())


def _write_kept() -> None:
    with _KEPT_LOCK:
        if _KEPT is None:
            return
        directory, label, started_at = _KEPT
        try:
            write_record(directory, label, started_at)
        except OSError:
            pass   # the next count writes it again; a count must never fail the request it counts


def read_records(directory: str | os.PathLike[str], *, since: float = 0.0,
                 labels: Collection[str] | None = None) -> dict[str, dict[str, int]]:
    """Sum the counts of the records in a directory whose process started at or after ``since``.

    Args:
        directory (str | os.PathLike[str]): The record directory; absent, it holds nothing.
        since (float): The epoch the parent launched its children at: an earlier launch into the same directory left
            records of its own.
        labels (Collection[str] | None): Only the records of these labels (a parent's own children, e.g. the
            ``mcp-<db_id>`` servers of an eval_bird); every record when ``None``.

    Returns:
        dict[str, dict[str, int]]: Lane to ``{"requests", "rate_limited", "overloaded"}``; an unreadable record counts
            nothing.
    """
    records = []
    for path in sorted(glob.glob(os.path.join(os.fspath(directory), "*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError):
            continue
        if (isinstance(record, dict) and float(record.get("started_at") or 0.0) >= since
                and (labels is None or record.get("label") in labels)):
            records.append(record.get("counts") or {})
    return merge(records)
