"""Secret gates and canonical copies of quiescent knowledge stores.

Chroma 1.5.9's HNSW writer serializes malloc capacity, not just initialized
records. Never copy that capacity: preserve live records and zero the remainder.
This module does not open a Chroma client, compute embeddings or alter a source.
"""
from __future__ import annotations

from contextlib import closing
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from multiprocessing import get_context
from pathlib import Path
import shutil
import sqlite3
import stat
import struct
import tempfile
from typing import Any, BinaryIO, Iterator

_HEADER = struct.Struct("<i6QiI3QdQ")
_BLOCK = 1 << 20


def _files(root: Path) -> list[Path]:
    def physical(path: Path) -> None:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"knowledge path is a link or reparse point: {path}")

    for path in (root, *root.parents):
        physical(path)
    if not root.is_dir():
        raise ValueError(f"knowledge root is missing: {root}")
    result = []
    for path in sorted(root.rglob("*")):
        physical(path)
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"knowledge path is not a regular file: {path}")
        result.append(path)
    return result


def _blocks(stream: BinaryIO, *, keep: int | None = None) -> Iterator[bytes]:
    remaining = keep
    for block in iter(lambda: stream.read(_BLOCK), b""):
        if remaining is not None:
            size = min(remaining, len(block))
            block = block[:size] + bytes(len(block) - size)
            remaining -= size
        yield block


def _scan_blocks(blocks: Iterator[bytes], name: str, secrets: Any) -> None:
    from .prompts.authority import scan_secrets

    overlap = max([256, *(len(value.encode("utf-8")) for value in secrets.values)])
    tail = b""
    for block in blocks:
        value = tail + block
        scan_secrets({name: value}, secrets)
        tail = value[-overlap:]


def scan_file(path: Path, secrets: Any) -> None:
    """Refuse configured secrets and key patterns in binary files, including block boundaries.

    Args:
        path: Regular artifact to read without modifying it.
        secrets: In-memory SecretInputs from the authority's credential owner.

    Raises:
        AuthorityError: A match is found. The diagnostic contains only the path.
        OSError: The file cannot be read completely.
    """
    with path.open("rb") as stream:
        _scan_blocks(_blocks(stream), str(path), secrets)


def _scan_knowledge(root: Path, secrets: Any) -> None:
    for path in _files(root):
        scan_file(path, secrets)


def scan_knowledge(root: Path) -> None:
    """Scan every file under a knowledge root; refuse unsafe paths or any secret hit.

    Args:
        root: Existing workspace or parent of per-database workspaces.

    Raises:
        AuthorityError: A file contains configured credentials or a key pattern.
        ValueError: The root is absent or contains links or special files.
        OSError: Any file cannot be scanned. No receipt can bypass a fresh scan.
    """
    from .prompts.authority import secret_inputs

    secrets = secret_inputs()
    root = Path(root).absolute()
    if os.name == "posix":
        # Closing ANY descriptor for an inode releases this process's POSIX
        # record locks on it, including SQLite's database and WAL-index locks.
        # Workspaces may already be open here. Read their bytes in a spawned
        # process so a scan cannot unlock the coordinator's live connections.
        # Credentials travel only through the private multiprocessing pipe.
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
            pool.submit(_scan_knowledge, root, secrets).result()
    else:
        _scan_knowledge(root, secrets)


def _native_ranges(root: Path, *, immutable: bool = False) -> tuple[dict[Path, int], set[Path]]:
    """Identify initialized prefixes from the pinned x64/arm64 HNSW v1 format."""
    keep: dict[Path, int] = {}
    orphans: set[Path] = set()
    for path in root.rglob("length.bin"):
        if path.parent.parent.name == "chroma_db" and (
                not (path.parent / "header.bin").is_file()
                or not (path.parent.parent / "chroma.sqlite3").is_file()):
            raise ValueError(f"incomplete HNSW store: {path}")
    for database in root.rglob("chroma.sqlite3"):
        uri = database.resolve().as_uri() + ("?mode=ro&immutable=1" if immutable else "?mode=ro")
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.execute("PRAGMA query_only=ON")
            live = dict(conn.execute('SELECT id, collection FROM segments WHERE scope=\'VECTOR\''))
            spaces = {}
            for identity, config, schema in conn.execute("SELECT id, config_json_str, schema_str FROM collections"):
                settings = json.loads(schema) if schema else {}
                vector = settings.get("keys", {}).get("#embedding", {}).get("float_list", {}).get("vector_index", {})
                legacy = json.loads(config) if config else {}
                spaces[identity] = vector.get("config", {}).get("space", legacy.get("hnsw", {}).get("space", "l2"))
        for header in database.parent.glob("*/header.bin"):
            if header.parent.name not in live:
                orphans.add(header.parent)
                continue
            raw = header.read_bytes()
            if len(raw) != _HEADER.size:
                raise ValueError(f"unsupported HNSW header: {header}")
            version, offset, capacity, count, stride, label, vector_offset, *_ = _HEADER.unpack(raw)
            if (version != 1 or offset != 0 or count > capacity or stride < 8
                    or not 0 <= vector_offset < label <= stride - 8):
                raise ValueError(f"invalid HNSW layout: {header}")
            space = spaces[live[header.parent.name]]
            if space not in {"cosine", "l2", "ip"}:
                raise ValueError(f"unsupported HNSW metric: {header}")
            for name, prefix, maximum in (
                ("length.bin", count * 4 if space == "cosine" else 0, capacity * 4),
                ("data_level0.bin", count * stride, capacity * stride),
            ):
                path = header.parent / name
                size = path.stat().st_size
                if size < prefix or size > maximum:
                    raise ValueError(f"invalid HNSW payload size: {path}")
                keep[path] = prefix
    return keep, orphans


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_knowledge(source: Path, destination: Path) -> None:
    """Create a scanned equivalent copy without ever copying unused native memory.

    Args:
        source: Quiescent root. Nonempty SQLite journals refuse the copy. Readers
            must leave it stable throughout the operation; it is never modified.
        destination: Absent path below a private existing directory. ACLs inherit
            from that directory; Unix files/directories are private too.

    Raises:
        AuthorityError: Any retained bytes contain a secret, before they are copied.
        ValueError: Layout is unsupported, journals exist, or source bytes change.
        OSError: Copy, final scan or atomic directory publication fails.

    Orphan HNSW segments are omitted. Live vectors, graph records, norms, SQLite
    bytes, training inputs and ledgers remain unchanged. Only unused capacity is
    zeroed; no embedding provider is called. Failed stages are removed.
    """
    from .prompts.authority import secret_inputs

    source, destination = Path(source).absolute(), Path(destination).absolute()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("knowledge copy requires an absent destination outside its source")
    files = _files(source)
    for path in files:
        if path.name.endswith(("-wal", "-journal")) and path.stat().st_size:
            raise ValueError(f"knowledge copy requires a quiescent SQLite snapshot: {path}")
    keep, orphans = _native_ranges(source, immutable=True)
    before = {path: _digest(path) for path in files}
    selected = []
    secrets = secret_inputs()
    for path in files:
        if path.name.endswith(("-wal", "-journal")):
            if path.stat().st_size:
                raise ValueError(f"knowledge copy requires a quiescent SQLite snapshot: {path}")
            continue
        if path.name.endswith("-shm") or any(parent in orphans for parent in path.parents):
            continue
        with path.open("rb") as stream:
            _scan_blocks(_blocks(stream, keep=keep.get(path)), str(path), secrets)
        selected.append(path)
    stage = Path(tempfile.mkdtemp(prefix=".knowledge-", dir=destination.parent))
    try:
        for path in selected:
            target = stage / path.relative_to(source)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with path.open("rb") as stream, target.open("xb") as output:
                if os.name != "nt":
                    target.chmod(0o600)
                for block in _blocks(stream, keep=keep.get(path)):
                    output.write(block)
        if files != _files(source) or before != {path: _digest(path) for path in files}:
            raise ValueError("knowledge source changed during copy")
        scan_knowledge(stage)
        stage.rename(destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def seal_knowledge(root: Path) -> None:
    """Zero unused HNSW capacity after writers finish, then scan the entire root.

    Args:
        root: Task-owned quiescent build output, never an active evaluation root.

    Raises:
        ValueError: A native layout is unsupported.
        AuthorityError: A secret remains anywhere in the output. It cannot be used.
        OSError: A file cannot be canonicalized or scanned.
    """
    root = Path(root).absolute()
    _files(root)
    keep, orphans = _native_ranges(root)
    # Windows may still map an orphan deleted by Chroma's collection rewrite.
    # It cannot be unlinked yet, but none of its graph payload is authoritative.
    # Clear its native buffers in place; the normal sweep reclaims it on reopen.
    for directory in orphans:
        for name in ("length.bin", "data_level0.bin", "link_lists.bin"):
            path = directory / name
            if path.is_file():
                keep[path] = 0
    for path, prefix in keep.items():
        with path.open("r+b") as stream:
            remaining = path.stat().st_size - prefix
            stream.seek(prefix)
            while remaining:
                size = min(remaining, _BLOCK)
                stream.write(bytes(size))
                remaining -= size
    scan_knowledge(root)
