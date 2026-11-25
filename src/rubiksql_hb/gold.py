"""Retained, content-bound reference execution shared by evaluation and lab scoring.

The first execution (including failure) is final for a database/SQL/executor/budget.
Unscorable references contribute zero in both arms of the fixed-size evaluation;
they are explicitly reported and never trigger answer retries.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import closing
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import sql_host
from .lab_identity import database_content_version

GOLD_CPU_TIMEOUT_SEC = 300.0
GOLD_POLICY = "retained-gold-zero/1"


@lru_cache(maxsize=64)
def _database_version(path: str, size: int, modified: int, changed: int) -> str:
    return database_content_version(path)


def _version(path: Path) -> str:
    wal = Path(f"{path}-wal")
    if wal.exists() and wal.stat().st_size:
        raise ValueError(f"gold database has a non-empty WAL: {path}")
    stat = path.stat()
    return _database_version(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _encode(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"$gold_blob": base64.b64encode(bytes(value)).decode("ascii")}
    raise TypeError(f"unsupported gold result value: {type(value).__name__}")


def execute_gold(url: str, sql: str) -> dict[str, Any]:
    """Execute or load a complete reference result under the fixed gold CPU budget.

    Args:
        url: On-disk SQLite URL; source bytes must be complete and immutable.
        sql: Exact reference SQL, never model-written answer SQL.

    Returns:
        Executor result plus ``gold_execution`` identity, status, budget, original
        CPU/wall timing and result digest. SQL failures are retained as
        ``gold_unscorable``. Full rows, including blobs, are retained losslessly.

    Raises:
        OSError: Source/cache cannot be read or written.
        ValueError: Source changed, has pending WAL writes, or cache is corrupt.
        sqlite3.Error: Cache storage/locking failed. Infrastructure errors never
            masquerade as an unscorable reference.

    Notes:
        The default cache is beside the source database, independent of HOME and
        release. ``RUBIKSQL_GOLD_CACHE_DIR`` may select one shared retained root.
        One SQLite transaction per content key serializes first execution across
        processes; crash rollback leaves an unpublished attempt safe to retry.
    """
    path = Path(sql_host._sqlite_path(url)).resolve()
    version = _version(path)
    binding = {"policy": GOLD_POLICY, "database_version": version,
               "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest(),
               "executor_sha256": hashlib.sha256(Path(sql_host.__file__).read_bytes()).hexdigest(),
               "sqlite_version": sqlite3.sqlite_version, "cpu_timeout_sec": GOLD_CPU_TIMEOUT_SEC}
    key = hashlib.sha256(json.dumps(binding, sort_keys=True).encode("utf-8")).hexdigest()
    root = Path(os.environ.get("RUBIKSQL_GOLD_CACHE_DIR") or path.parent / ".rubiksql-gold-v1")
    root.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(root / f"{key}.sqlite", timeout=3600)) as cache, cache:
        cache.execute("CREATE TABLE IF NOT EXISTS result (id INTEGER PRIMARY KEY CHECK(id=1), payload BLOB NOT NULL)")
        cache.execute("BEGIN IMMEDIATE")
        stored = cache.execute("SELECT payload FROM result WHERE id=1").fetchone()
        if stored:
            record = json.loads(gzip.decompress(stored[0]))
            for row in record.get("rows", []):
                for name, value in row.items():
                    if isinstance(value, dict):
                        row[name] = base64.b64decode(value["$gold_blob"])
            metadata = record["gold_execution"]
            if any(metadata.get(name) != value for name, value in binding.items()) or metadata.get("key") != key:
                raise ValueError(f"gold cache binding differs: {key}")
        else:
            cpu, wall = time.thread_time(), time.monotonic()
            record = sql_host.query_readonly(url, sql, timeout_sec=GOLD_CPU_TIMEOUT_SEC)
            metadata = {**binding, "key": key,
                        "status": "gold_unscorable" if "error" in record else "ok",
                        "failure_kind": ("timeout" if sql_host.is_timeout_error(record) else "error") if "error" in record else None,
                        "cpu_sec": time.thread_time() - cpu, "wall_sec": time.monotonic() - wall}
            payload = json.dumps(record, default=_encode, ensure_ascii=False).encode("utf-8")
            metadata["result_sha256"] = hashlib.sha256(payload).hexdigest()
            record["gold_execution"] = metadata
            if _version(path) != version:
                raise ValueError(f"gold database changed during execution: {path}")
            cache.execute("INSERT INTO result VALUES (1, ?)",
                          (gzip.compress(json.dumps(record, default=_encode, ensure_ascii=False).encode("utf-8")),))
        payload = json.dumps({k: v for k, v in record.items() if k != "gold_execution"},
                             default=_encode, ensure_ascii=False).encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != metadata["result_sha256"] or _version(path) != version:
            raise ValueError(f"gold cache result or database changed: {key}")
        return record


def probe_gold_results(url: str, sql: str, *, runs: int) -> Iterator[dict[str, Any]]:
    """Yield the retained reference, then fresh executions to measure consistency.

    A retained failure is terminal. Successful references may be repeated for
    diagnostics, without replacing the scoring result or its original timing.

    Args:
        url: On-disk SQLite URL of a complete, immutable source.
        sql: Exact reference SQL.
        runs: Total observations, including the retained reference; at least one.

    Yields:
        Executor results. Only the first carries retained ``gold_execution`` metadata.
        Every fresh execution uses the same gold CPU budget and a new connection.

    Raises:
        OSError: Source or retained cache cannot be read or written.
        ValueError: Source changed, has pending WAL writes, or cache is corrupt.
        sqlite3.Error: Retained cache storage or locking failed.
    """
    reference = execute_gold(url, sql)
    yield reference
    if "error" in reference:
        return
    path = Path(sql_host._sqlite_path(url)).resolve()
    for _ in range(1, max(1, runs)):
        result = sql_host.query_readonly(url, sql, timeout_sec=GOLD_CPU_TIMEOUT_SEC)
        if _version(path) != reference["gold_execution"]["database_version"]:
            raise ValueError(f"gold database changed during probe: {path}")
        yield result
        if "error" in result:
            return
