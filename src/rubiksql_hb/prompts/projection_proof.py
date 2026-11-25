"""Content proofs for a previously verified, privately leased Git projection.

Proofs belong to the exact application/HB/schema identity in the cache seal. They
are accelerators, never authority: remote ancestry, committed export bytes and
the sealed database hash are checked before a persisted proof is admitted.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from .canon import canon, sha256_hex
from .store import TABLES, _canon_json
from .workspace_state import EVIDENCE_TABLES, schema_facts


def raw_state(reader: Any) -> Mapping[str, Any]:
    """Hash native encodings and DDL without decoding large JSON evidence bodies."""
    def compute() -> Mapping[str, Any]:
        state = {}
        for entity, table in {**TABLES, **EVIDENCE_TABLES}.items():
            schema = schema_facts(reader.connection, table)
            digest = hashlib.sha256(canon(schema).encode("utf-8"))
            if schema["present"]:
                for row in reader.connection.execute(f'SELECT * FROM "{table}" ORDER BY object_id'):
                    for value in row:
                        payload = value.encode("utf-8") if isinstance(value, str) else canon(value).encode("utf-8")
                        digest.update((type(value).__name__ + ":" + str(len(payload)) + ":").encode("ascii"))
                        digest.update(payload)
                    digest.update(b";")
            state[entity] = digest.hexdigest()
        return state
    return reader._retained(("projection_raw_state",), compute)


def capture(reader: Any, head: Mapping[str, Any], files: Mapping[str, bytes]) -> dict[str, Any]:
    """Capture only after full or incremental native verification has succeeded."""
    import json

    return {"raw": dict(raw_state(reader)),
            "rows": {entity: {row["object_id"]: sha256_hex(_canon_json(row)) for row in reader._held_rows(entity)}
                     for entity in TABLES},
            "export": {**head, "workspace_entities": json.loads(files["workspace/STATE.json"])["entities"]}}


def unchanged(reader: Any, proof: Mapping[str, Any], entity: str) -> bool:
    """Check the complete table encoding and schema against its verified anchor."""
    return raw_state(reader).get(entity) == proof["raw"].get(entity)


def verified_row(proof: Mapping[str, Any] | None, entity: str, row: Mapping[str, Any]) -> bool:
    """Reuse a row's semantic checks only for its exact complete content."""
    return bool(proof and proof["rows"].get(entity, {}).get(row["object_id"]) == sha256_hex(_canon_json(row)))
