"""Verify retained role identities with their recorded Git implementation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile
from typing import Any

from .prompts.canon import canon, sha256_hex
from .prompts.binding import binding_document, load_binding, write_binding
from .prompts.snapshot import VerifiedSnapshot, verify_snapshot


# Process-local successful proofs only. Each reuse still checks the actual
# snapshot and all Git source/dependency bytes before consulting this set.
_VERIFIED_ROLES: set[tuple[str, str, str, str]] = set()


def verify_role_records(snapshot: VerifiedSnapshot, implementation: Mapping[str, Any],
                        records: Sequence[Mapping[str, Any]]) -> None:
    """Recompute role records from a clean producer commit and verified native pin.

    Args:
        snapshot: Parent's already verified prompt and Tool closure.
        implementation: Producer manifest's commit, clean flag and artifact hashes.
        records: Declared role contexts and digests to verify.

    Returns:
        None: Every role record matches the producer's source and native pin.

    Raises:
        ValueError: Source is unavailable, provenance is malformed, dependency
            artifacts differ from Git, or any reconstructed role record differs.

    The historical package runs in a separate interpreter with its own source
    directory. It reads the snapshot without importing or executing pinned tools.
    No current-host version counter or renderer participates in its role keys.
    """
    commit = implementation.get("commit")
    if implementation.get("clean") is not True or not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("parent implementation requires a clean recorded full Git commit")
    checkout = Path(__file__).resolve().parents[2]

    def git(*args: str) -> bytes:
        return subprocess.run(["git", *args], cwd=checkout, capture_output=True, check=True, timeout=60).stdout

    try:
        if git("rev-parse", "--verify", commit + "^{commit}").decode().strip() != commit:
            raise ValueError("parent implementation commit does not resolve exactly")
        wheels = git("ls-tree", "--name-only", commit, "vendor/").decode().splitlines()
        wheels = [name for name in wheels if name.startswith("vendor/heavenbase-") and name.endswith(".whl")]
        if len(wheels) != 1:
            raise ValueError("parent implementation must record one vendored HeavenBase wheel")
        for key, name in (("uv_lock_sha256", "uv.lock"), ("wheel_sha256", wheels[0])):
            if implementation.get(key) != "sha256:" + sha256_hex(git("show", commit + ":" + name)):
                raise ValueError(f"parent implementation {key} differs from its Git commit")
        archive = git("archive", "--format=zip", commit, "src")
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"parent implementation source {commit} is unavailable; fetch its Git history") from error
    snapshot = verify_snapshot(snapshot.path, expected_release=snapshot.release, expected_digest=snapshot.digest,
                               expected_authority_commit=snapshot.authority_commit)
    selected = load_binding()
    proof = (snapshot.digest, sha256_hex(archive),
             sha256_hex(canon({"implementation": dict(implementation), "records": list(records)})),
             canon(binding_document(selected)))
    if proof in _VERIFIED_ROLES:
        return
    with tempfile.TemporaryDirectory(prefix="role-source-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(io.BytesIO(archive)) as source:
            for item in source.infolist():
                if not (root / item.filename).resolve().is_relative_to(root.resolve()):
                    raise ValueError("parent Git source archive contains an invalid path")
            source.extractall(root)
        probe = root / "verify_roles.py"
        # An isolated epoch's approved binding can differ from the historical
        # package default. Preserve the caller's selected authority in the probe.
        binding = write_binding(selected, root / "authority.json")
        probe.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent / 'src'))\n"
            "from rubiksql_hb.prompts.snapshot import verify_snapshot\n"
            "from rubiksql_hb.prompts.binding import select_binding\n"
            "from rubiksql_hb.tool_bundle import snapshot_role_contract\n"
            "request = json.load(sys.stdin)\n"
            "select_binding(Path(request['binding']))\n"
            "snapshot = verify_snapshot(Path(request['path']), expected_digest=request['digest'])\n"
            "print(json.dumps([snapshot_role_contract(snapshot, record['context']) for record in request['records']]))\n",
            encoding="utf-8")
        environment = {key: value for key, value in os.environ.items() if not key.startswith("RUBIKSQL_")}
        environment["PYTHONUTF8"] = "1"
        try:
            result = subprocess.run([sys.executable, "-I", "-X", "utf8", str(probe)], cwd=root, env=environment,
                input=json.dumps({"path": str(snapshot.path), "digest": snapshot.digest,
                                  "binding": str(binding), "records": list(records)}),
                capture_output=True, text=True, encoding="utf-8", check=True, timeout=120)
            rebuilt = json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise ValueError(f"parent implementation {commit} could not reconstruct its recorded role contracts") from error
    if rebuilt != list(records):
        raise ValueError("parent role digests differ from its verified snapshot and recorded Git implementation")
    if len(_VERIFIED_ROLES) >= 256:
        _VERIFIED_ROLES.clear()
    _VERIFIED_ROLES.add(proof)
