"""Start the NL2SQL MCP server from the checked-in execution snapshot."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
ADOPTED = ROOT / "releases" / "adopted.json"
ARCHIVE = ROOT / "releases" / "adopted-snapshot.tar.gz"


def ensure_snapshot() -> Path:
    """Extract and verify the adopted immutable snapshot in the local runtime cache."""
    from rubiksql_hb.prompts.snapshot import verify_snapshot

    adopted = json.loads(ADOPTED.read_text(encoding="utf-8"))
    digest = adopted["digest"]
    cache = Path(os.environ.get("RUBIKSQL_SNAPSHOT_CACHE", ROOT / "runtime" / "snapshots"))
    target = cache / digest
    if target.is_dir():
        verify_snapshot(target, expected_digest=digest, expected_release=adopted["release"])
        return target

    archive_hash = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if archive_hash != adopted["archive_sha256"]:
        raise RuntimeError("adopted snapshot archive hash differs from releases/adopted.json")
    cache.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="rubik-snapshot-", dir=cache) as temporary:
        staging = Path(temporary)
        with tarfile.open(ARCHIVE, mode="r:gz") as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                if (relative.is_absolute() or ".." in relative.parts
                        or relative.parts[:2] != ("snapshot", digest)
                        or not (member.isfile() or member.isdir())):
                    raise RuntimeError(f"unsafe snapshot member: {member.name}")
                output = staging.joinpath(*relative.parts)
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, output.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
        staged = staging / "snapshot" / digest
        verify_snapshot(staged, expected_digest=digest, expected_release=adopted["release"])
        if not target.exists():
            os.replace(staged, target)
    verify_snapshot(target, expected_digest=digest, expected_release=adopted["release"])
    return target


def main() -> int:
    snapshot_path = ensure_snapshot()
    from rubiksql_hb.prompts.snapshot import snapshot_env, verify_snapshot
    from rubiksql_hb.tool_bundle import install_bundle_importer

    snapshot = verify_snapshot(snapshot_path)
    os.environ.update(snapshot_env(snapshot, bundle=True))
    os.environ.setdefault("RUBIKSQL_DATA_DIR", str(ROOT / "runtime" / "data"))
    install_bundle_importer(snapshot)
    print(f"[rubiksql] release={snapshot.release} snapshot={snapshot.digest}", file=sys.stderr)
    from rubiksql_hb.server import main as serve

    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
