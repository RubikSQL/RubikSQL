"""Read-only canonical evidence rows and structural schema facts for recovery.

This serialization scope is separate from version history and answering snapshots. Canonical
Tool, rule and parameter-set versions are version sets (``store.TYPED_TABLES``) exported with
the other native versions; historical rule and knob projection rows are rebuilt by their owner
and never exported.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import sqlite3
from typing import Any

from .canon import sha256_hex
from .store import _bytes

EVIDENCE_TABLES = {
    "q-a-entity": "q_a_entity", "experiment-entity": "experiment_entity", "trace-entity": "trace_entity",
    "candidate-entity": "candidate_entity", "judge-entity": "judge_entity",
    "agent-session": "agent_session", "agent-message": "agent_message",
}

# A single logical row must remain transportable by the GitHub authority remote.
MAX_EVIDENCE_ROW_BYTES = 100 * 1024 * 1024


def _evidence_row_path(entity: str, row: Mapping[str, Any]) -> str:
    identifier = row.get("object_id")
    if entity not in EVIDENCE_TABLES or not isinstance(identifier, str) or not identifier:
        raise ValueError("export evidence needs a known entity and a nonempty string identity")
    digest = sha256_hex(identifier)
    return f"workspace/{entity}/{digest[:2]}/{digest}.json"


def encode_evidence_rows(entity: str, rows: Sequence[Mapping[str, Any]], *,
                         row_files: bool) -> dict[str, bytes]:
    """Serialize complete evidence without changing any logical row.

    Args:
        entity: One canonical evidence entity identifier.
        rows: Complete rows in their native object-id order.
        row_files: Use format-4 identity-addressed files; false retains formats 2–3.

    Returns:
        Canonical UTF-8 files. Empty format-4 tables have no row files.

    Raises:
        ValueError: The entity or identity is invalid, an identity repeats, or one
            row exceeds the remote's 100-MiB blob limit.
    """
    if entity not in EVIDENCE_TABLES:
        raise ValueError(f"unknown canonical evidence entity {entity!r}")
    if not row_files:
        return {f"workspace/{entity}.json": _bytes(rows)}
    files = {}
    for row in rows:
        path = _evidence_row_path(entity, row)
        if path in files:
            raise ValueError(f"export evidence repeats an identity of {entity}")
        payload = _bytes(row)
        if len(payload) > MAX_EVIDENCE_ROW_BYTES:
            raise ValueError(f"one {entity} row exceeds the 100-MiB export blob limit")
        files[path] = payload
    return files


def decode_evidence_rows(entity: str, files: Mapping[str, bytes], *,
                         row_files: bool) -> list[dict[str, Any]]:
    """Read complete evidence from a verified export's physical layout.

    Args:
        entity: One canonical evidence entity identifier.
        files: The export's digest-verified file map, without HEAD.json.
        row_files: Select format 4 rather than the historical table-file layout.

    Returns:
        Full rows in native object-id order. The caller verifies STATE's complete
        count, semantic digest, schema and overall file coverage.

    Raises:
        ValueError: A row file is oversized, noncanonical or names another identity.
        KeyError: A historical table file is absent.
    """
    if entity not in EVIDENCE_TABLES:
        raise ValueError(f"unknown canonical evidence entity {entity!r}")
    if not row_files:
        rows = json.loads(files[f"workspace/{entity}.json"])
        if not isinstance(rows, list):
            raise ValueError(f"export evidence of {entity} is not a row list")
        return rows
    rows = []
    prefix = f"workspace/{entity}/"
    for path, payload in files.items():
        if not path.startswith(prefix):
            continue
        if len(payload) > MAX_EVIDENCE_ROW_BYTES:
            raise ValueError(f"one {entity} row exceeds the 100-MiB export blob limit")
        row = json.loads(payload)
        if (not isinstance(row, dict) or _evidence_row_path(entity, row) != path
                or _bytes(row) != payload):
            raise ValueError(f"export evidence row path or canonical bytes differ for {entity}")
        rows.append(row)
    return sorted(rows, key=lambda row: row["object_id"])


def evidence_entities() -> Mapping[str, Any]:
    """Return static native canonical classes without opening a workspace."""
    from ..lab_schema import canonical_evidence_entities

    entities = canonical_evidence_entities()
    if set(entities) != set(EVIDENCE_TABLES):
        raise ValueError("canonical evidence entity declaration differs from recovery scope")
    return entities


def evidence_rows(connection: sqlite3.Connection, entity: str) -> tuple[Mapping[str, Any], ...]:
    """Decode complete native evidence rows from an already-open read transaction.

    Args:
        connection: Existing SQLite connection; no workspace is initialized.
        entity: One of the seven closed canonical evidence entity identifiers.

    Returns:
        Full decoded rows in object-id order. A historical missing table is empty.

    Raises:
        ValueError: Unknown entity, mismatched schema or invalid native encoding.
    """
    if entity not in EVIDENCE_TABLES:
        raise ValueError(f"unknown canonical evidence entity {entity!r}")
    table = EVIDENCE_TABLES[entity]
    columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
    if not columns:
        return ()
    schema = evidence_entities()[entity].schema()
    expected = set(schema.fields) | {"object_id"}
    if columns != expected:
        raise ValueError(f"canonical evidence schema differs for {entity}: missing {sorted(expected - columns)}, extra {sorted(columns - expected)}")
    from heavenbase.utils.serialize import loads_json

    cursor = connection.execute(f'SELECT * FROM "{table}" ORDER BY object_id')
    names = [column[0] for column in cursor.description]
    result = []
    for native in cursor:
        row = dict(zip(names, native))
        for name, field in schema.fields.items():
            if row.get(name) is None:
                continue
            identifier = type(field.dtype).identifier
            value = loads_json(row[name]) if identifier in {"json", "array"} else row[name]
            if identifier == "boolean":
                # Raw sqlite3 reads bypass HB's SQLAlchemy Boolean result codec.
                if type(value) is not int or value not in (0, 1):
                    raise ValueError(f"canonical evidence encoding differs for {entity}.{name}: expected SQLite Boolean 0 or 1")
                value = bool(value)
            row[name] = field.dtype.decode(value)
        result.append(row)
    return tuple(result)


def schema_facts(connection: sqlite3.Connection, table: str) -> Mapping[str, Any]:
    """Read exact canonical-table DDL, indexes, triggers and column contracts.

    Args:
        connection: Existing SQLite connection in the owner's transaction.
        table: Repository-owned table name, never an external SQL identifier.

    Returns:
        Detached structural facts, including an explicit absent-table marker.
    """
    return {"present": bool(connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()),
            "columns": [list(row) for row in connection.execute(f'PRAGMA table_info("{table}")')],
            "objects": [list(row) for row in connection.execute(
                "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE tbl_name=? ORDER BY type,name", (table,))]}
