"""Read-only loading and validation of contrast-atlas fanout artifacts.

A contrast atlas is the frozen offline fanout artifact written by
`scripts/build_contrast_atlas.py`: per-database declared join-edge evidence whose 1:N
fanout profiles (`declared_edges[].max_children`) are the grain router's
fanout evidence source. This module loads one artifact and validates its fixed
v1 envelope, identity, and canonical self-hash; the runtime loader never opens
the source SQLite database and never recomputes the live SQLite fingerprint
(`sqlite_content_fingerprint` exists for the OFFLINE builder and the arm-level
preflight, which do read the source file's bytes).

Envelope validation is a subset check: the slim production artifact carries
exactly the required keys, while older full experiment artifacts (extra
`soft_edges` / `paths` payloads, per-table role tags) keep validating — extra
keys are tolerated and still participate in the canonical self-hash.
"""

import concurrent.futures
import hashlib
import json
import os

# Fixed v1 literals. Builders, loaders, and --verify-only reject other literal
# versions instead of guessing.
_FORMAT_VERSION = "contrast-atlas-artifact-v1"
_SCHEMA_VERSION = 1
_ATLAS_VERSION = "v1"
# Production-required payload only (subset check): the grain router reads the
# `tables` key set and the `declared_edges` fanout fields. Old full artifacts
# additionally carry `soft_edges` / `paths`; those extra keys are tolerated.
_ENVELOPE_DICT_KEYS = ("tables",)
_ENVELOPE_LIST_KEYS = ("declared_edges",)
_ENVELOPE_REQUIRED_KEYS = ("sqlite_fingerprint", "built_at", "inputs", "content_sha256")


def _load_timeout_sec() -> float:
    """Read the atlas artifact load/parse budget at call time (env-tunable, default 5s)."""
    return float(os.environ.get("RUBIKSQL_ATLAS_LOAD_TIMEOUT_SEC", "5"))


def sqlite_content_fingerprint(path: str) -> str:
    """Content-stable fingerprint of one SQLite database file (streamed SHA-256).

    Hashes the file bytes in 1 MiB chunks through a plain read-only file
    handle: no SQLite connection is opened, nothing is written, and no
    -wal/-shm sidecars can appear. A touch or byte-exact copy (mtime moves,
    bytes do not) keeps the fingerprint; any content change — even one that
    preserves the file size — changes it. The single implementation shared by
    the offline builder, its `--verify-only` mode, and the arm-level preflight.

    Args:
        path (str): SQLite database file path.

    Returns:
        str: Hex SHA-256 digest of the file bytes.

    Raises:
        OSError: The file cannot be read.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_artifact_sha256(obj: dict) -> str:
    """Return the campaign-wide canonical self-hash of one artifact object.

    The single byte-level algorithm shared by every builder, loader, and
    `--verify-only` path: `built_at` is included; only the self-hash field
    itself is excluded.
    """
    payload = dict(obj)
    payload.pop("content_sha256", None)
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def atlas_path(db_id: str, artifact_dir: str) -> str:
    """Return the canonical versioned artifact path for one database."""
    return os.path.join(artifact_dir, "contrast_atlas", _ATLAS_VERSION, f"{db_id}.json")


def validate_atlas_envelope(atlas: object, db_id: str) -> dict:
    """Validate the fixed v1 envelope, identity, and canonical self-hash.

    Shared by the runtime loader and the offline builder's `--verify-only`
    path. A subset check: only the production-required keys are demanded, so
    slim artifacts and old full experiment artifacts (extra `soft_edges` /
    `paths` / role payloads) both pass. Raises ValueError on any mismatch;
    never repairs or guesses.
    """
    if not isinstance(atlas, dict):
        raise ValueError("contrast atlas artifact must be a JSON object")
    if atlas.get("format_version") != _FORMAT_VERSION:
        raise ValueError(
            f"unsupported format_version {atlas.get('format_version')!r}; expected {_FORMAT_VERSION!r}"
        )
    if atlas.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {atlas.get('schema_version')!r}; expected {_SCHEMA_VERSION}"
        )
    if atlas.get("atlas_version") != _ATLAS_VERSION:
        raise ValueError(
            f"unsupported atlas_version {atlas.get('atlas_version')!r}; expected {_ATLAS_VERSION!r}"
        )
    if atlas.get("db_id") != db_id:
        raise ValueError(f"artifact db_id {atlas.get('db_id')!r} does not match requested {db_id!r}")
    missing = [key for key in _ENVELOPE_REQUIRED_KEYS if key not in atlas]
    if missing:
        raise ValueError(f"artifact envelope is missing required key(s): {', '.join(missing)}")
    for key in _ENVELOPE_DICT_KEYS:
        if not isinstance(atlas.get(key), dict):
            raise ValueError(f"artifact envelope key {key!r} must be a JSON object")
    for key in _ENVELOPE_LIST_KEYS:
        if not isinstance(atlas.get(key), list):
            raise ValueError(f"artifact envelope key {key!r} must be a JSON array")
    recorded = atlas.get("content_sha256")
    computed = canonical_artifact_sha256(atlas)
    if recorded != computed:
        raise ValueError(
            f"artifact self-hash mismatch: recorded {str(recorded)[:16]}..., computed {computed[:16]}..."
        )
    return atlas


def load_contrast_atlas(db_id: str, artifact_dir: str) -> dict:
    """Load and validate one frozen contrast-atlas artifact (read-only).

    File read + JSON parse are bounded by `RUBIKSQL_ATLAS_LOAD_TIMEOUT_SEC`
    (default 5s). Never builds, writes, opens the source database, or
    recomputes the live SQLite fingerprint at runtime.

    Args:
        db_id (str): Database identifier the artifact must declare.
        artifact_dir (str): Strategy-artifact root containing `contrast_atlas/v1/`.

    Returns:
        dict: The validated atlas object.

    Raises:
        FileNotFoundError: The canonical artifact file does not exist.
        TimeoutError: Load/parse exceeded the configured budget.
        ValueError: The artifact fails envelope/identity/self-hash validation.
    """
    path = atlas_path(db_id, artifact_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"contrast atlas artifact missing: {path}; build it offline with scripts/build_contrast_atlas.py"
        )

    def _read() -> dict:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    budget = _load_timeout_sec()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        atlas = pool.submit(_read).result(timeout=budget)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"contrast atlas load exceeded the {budget}s budget: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"contrast atlas artifact is not valid JSON: {path}: {exc}") from None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return validate_atlas_envelope(atlas, db_id)
