"""Git is the committed authority; HeavenBase is a private, rebuildable view.

Each writer starts at one fetched remote commit, stages native changes locally,
then publishes exactly once with an explicit remote-ref lease. A failed lease
never merges or authorizes an execution effect. Replay-safe callers may discard
the writer and revalidate their entire operation in a fresh writer.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import errno
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from . import binding as B
from .canon import canon, sha256_hex
from .types import AuthorityBinding, AuthorityError, GitAuthorityBinding

if os.name == "nt":
    import ctypes
    from ctypes import wintypes
    import msvcrt

    class _Overlapped(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                    ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                    ("hEvent", wintypes.HANDLE)]

    _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _KERNEL32.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(_Overlapped)]
    _KERNEL32.LockFileEx.restype = wintypes.BOOL
    _KERNEL32.UnlockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                     wintypes.DWORD, ctypes.POINTER(_Overlapped)]
    _KERNEL32.UnlockFileEx.restype = wintypes.BOOL

_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_WRITERS = threading.local()
_READ_CACHE_THREAD_LOCK = threading.Lock()
_READ_CACHE_LOCKS: dict[Path, threading.Lock] = {}
_READ_CACHE_LEASES = threading.local()
_READ_CACHE_ACTIVE: set[Path] = set()
_READ_CACHE_READERS: dict[Path, int] = {}
_READ_CACHE_ENTRIES = 4
# Retain the current lab projection without rebuilding it on each authority read.
_READ_CACHE_BYTES = 2 * 1024 ** 3
_Result = TypeVar("_Result")
WRITE_ATTEMPTS = 32
#: No conflict retry starts after this long. A lost attempt discards its writer's
#: projection, so each retry restores again and repeats the whole operation.
WRITE_RETRY_SECONDS = 3600.0
#: Only read-only remote transport failures retry, at most twice.
_REMOTE_READ_RETRY_DELAYS = (1.0, 3.0)


def retry_conflicts(operation: Callable[[], _Result]) -> _Result:
    """Re-run a replay-safe operation after remote-head conflicts, within count and time bounds.

    Args:
        operation: Complete operation that opens and closes its own writer on
            every call, rechecking all preconditions. It must not perform an
            execution effect before confirmed publication. Nested operations
            must leave retry ownership to their outer writer.

    Returns:
        The result of the first successful operation.

    Raises:
        GitAuthorityConflict: Thirty-two attempts conflict, a conflict arrives
            after ``WRITE_RETRY_SECONDS`` since the first attempt started, or
            the ref did not move.
        AuthorityError: Any non-conflict failure, including unconfirmed pushes
            and failures after a committed checkpoint; these are never retried.
    """
    started = time.monotonic()
    for attempt in range(WRITE_ATTEMPTS):
        try:
            return operation()
        except GitAuthorityConflict as error:
            # Older publishers also classify an unchanged rejected push as a
            # conflict. It is not contention and must not be retried.
            if (attempt == WRITE_ATTEMPTS - 1 or error.expected == error.observed
                    or time.monotonic() - started >= WRITE_RETRY_SECONDS):
                raise
            time.sleep(random.uniform(0.02, min(1.0, 0.05 * 2 ** attempt)))
    raise AssertionError("unreachable retry exhaustion")


class _ReadCacheBusy(AuthorityError):
    """The shared read cache stayed locked past the wait limit; the reader restores privately."""


def _account_home() -> Path:
    """Resolve the token's account profile independently of task HOME/USERPROFILE."""
    if os.name != "nt":
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    # GetUserProfileDirectory uses the current token, unlike environment-backed
    # Path.home()/expanduser(). Execution arms deliberately replace both HOME
    # and USERPROFILE and must still share one account-private projection cache.
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    userenv = ctypes.WinDLL("userenv", use_last_error=True)
    advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi.OpenProcessToken.restype = wintypes.BOOL
    userenv.GetUserProfileDirectoryW.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    userenv.GetUserProfileDirectoryW.restype = wintypes.BOOL
    _KERNEL32.GetCurrentProcess.restype = wintypes.HANDLE
    _KERNEL32.CloseHandle.argtypes = [wintypes.HANDLE]
    _KERNEL32.CloseHandle.restype = wintypes.BOOL
    token = wintypes.HANDLE()
    if not advapi.OpenProcessToken(_KERNEL32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
        raise AuthorityError("Git projection cache cannot resolve the current account profile")
    try:
        size = wintypes.DWORD()
        userenv.GetUserProfileDirectoryW(token, None, ctypes.byref(size))
        if not size.value:
            raise AuthorityError("Git projection cache cannot resolve the current account profile")
        value = ctypes.create_unicode_buffer(size.value)
        if not userenv.GetUserProfileDirectoryW(token, value, ctypes.byref(size)):
            raise AuthorityError("Git projection cache cannot resolve the current account profile")
        return Path(value.value).resolve()
    finally:
        _KERNEL32.CloseHandle(token)


def default_authority_cache_dir() -> Path:
    """Return the explicitly inherited cache or the current account's stable cache.

    The caller still verifies ownership, private permissions and redirects before
    use. This resolver creates nothing and does not use a task's isolated HOME.

    Returns:
        Absolute inherited cache path or the current account's .rhb/authority-read.

    Raises:
        AuthorityError: The override is empty or relative, or the token's account
            profile cannot be resolved. There is no task-HOME fallback.
    """
    configured = os.environ.get("RUBIKSQL_AUTHORITY_CACHE_DIR")
    if configured is not None:
        if not configured or not Path(configured).is_absolute():
            raise AuthorityError("Git projection cache override must be an absolute private path")
        return Path(configured)
    # HB embeds its module artifact paths. Keep the stable account directory
    # short enough for Windows MAX_PATH. A seed rebuilds HB artifacts at its new path.
    return _account_home() / ".rhb" / "authority-read"


def _record_payload_path(path: str) -> str:
    """A RECORD path relative to its site-packages, in POSIX form, whatever prefix the installer wrote."""
    parts = path.replace("\\", "/").split("/")
    if "site-packages" in parts:
        parts = parts[len(parts) - parts[::-1].index("site-packages"):]
    return "/".join(parts)


def _heavenbase_build_identity() -> Mapping[str, str]:
    """Identify the installed wheel separately from permitted dirty source edits.

    Like the runtime contract, use installed wheel provenance; like the test
    runner, include RECORD evidence. Keep only HB payload records so installation
    paths in direct_url.json and generated bytecode do not split the same build.
    uv may omit archive hashes, in which case hash its local source wheel. Missing
    build evidence disables cache reuse through the ordinary recovery fallback.
    """
    import csv
    import importlib.metadata
    import io
    from urllib.parse import unquote, urlsplit
    from urllib.request import url2pathname

    try:
        distribution = importlib.metadata.distribution("heavenbase")
        source = json.loads(distribution.read_text("direct_url.json") or "{}")
        url = urlsplit(source["url"])
        wheel = unquote(url.path).rsplit("/", 1)[-1]
        if not wheel.startswith("heavenbase-") or not wheel.endswith(".whl"):
            raise ValueError("HB was not installed from a wheel")
        archive = source.get("archive_info", {})
        checksum = archive.get("hashes", {}).get("sha256")
        if checksum is None and archive.get("hash", "").startswith("sha256="):
            checksum = archive["hash"].removeprefix("sha256=")
        if url.scheme == "file" and url.netloc in {"", "localhost"}:
            path = Path(url2pathname(url.path))
            if path.is_file():
                actual = _database_digest(path)
                if checksum is not None and checksum != actual:
                    raise ValueError("HB wheel differs from installed provenance")
                checksum = actual
        if not isinstance(checksum, str) or re.fullmatch(r"[0-9a-f]{64}", checksum) is None:
            raise ValueError("HB wheel checksum is unavailable")
        rows = csv.reader(io.StringIO(distribution.read_text("RECORD") or ""))
        records = sorted([path, digest, size] for path, digest, size in
                         ((_record_payload_path(row[0]), row[1], row[2]) for row in rows if len(row) == 3)
                         if path.startswith("heavenbase/") and not path.endswith(".pyc") and digest)
        if not records:
            raise ValueError("HB installed payload records are unavailable")
        return {"version": distribution.version, "wheel": wheel, "wheel_sha256": checksum,
                "record_sha256": sha256_hex(canon(records))}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, importlib.metadata.PackageNotFoundError) as error:
        raise AuthorityError("Git projection cache requires installed HB wheel provenance") from error


def _read_cache_identity(head: Mapping[str, Any], commit: str, binding: GitAuthorityBinding) -> Mapping[str, Any]:
    """Include installed HB provenance and source/schema metadata, including dirty edits."""
    import heavenbase
    from .authority import _schemas
    from .workspace_state import evidence_entities

    sources = {}
    for name, root in (("rubiksql", Path(__file__).resolve().parents[1]),
                       ("heavenbase", Path(heavenbase.__file__).resolve().parent)):
        sources[name] = {path.relative_to(root).as_posix(): sha256_hex(path.read_bytes())
                         for path in sorted(root.rglob("*")) if path.suffix in {".py", ".yaml", ".yml"}}
    metadata = {name: {path: digest for path, digest in files.items() if Path(path).suffix in {".yaml", ".yml"}}
                for name, files in sources.items()}
    schemas = {**_schemas(), **{name: entity.schema() for name, entity in evidence_entities().items()}}
    return {"schema": "rubiksql.authority-projection/7", "binding": B.binding_document(binding),
            "commit": commit, "export_format": head["schema"], "head_sha256": sha256_hex(canon(head)),
            "code": sha256_hex(canon(sources)), "hb_build": _heavenbase_build_identity(),
            "metadata": sha256_hex(canon(metadata)),
            "native_schema": sha256_hex(canon({name: schema.to_dict() for name, schema in schemas.items()})),
            "python": sys.version, "sqlite": sqlite3.sqlite_version}


def _check_cache_path(directory: Path) -> None:
    """Refuse redirects without creating or changing a prospective source root."""
    for candidate in (*reversed(directory.parents), directory):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise AuthorityError("Git projection cache must not use a redirected directory")


def _check_cache_permissions(path: Path, *, recursive: bool = False) -> None:
    """Check private ownership/ACLs without repairing a seed source's permissions."""
    if os.name == "nt":
        code = (
            "$ErrorActionPreference='Stop';"
            "$identity=[Security.Principal.WindowsIdentity]::GetCurrent();"
            "$principal=[Security.Principal.WindowsPrincipal]$identity;"
            "if(-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)){exit 3};"
            "$paths=@(Get-Item -Force -LiteralPath $env:RUBIKSQL_AUTHORITY_CACHE_DIR);"
            + ("$paths+=@(Get-ChildItem -Force -Recurse -LiteralPath $env:RUBIKSQL_AUTHORITY_CACHE_DIR);" if recursive else "")
            + "foreach($path in $paths){"
            "$allowed=@($identity.User.Value,'S-1-5-18','S-1-5-32-544');"
            "$acl=Get-Acl -LiteralPath $path.FullName;"
            "if($acl.GetOwner([Security.Principal.SecurityIdentifier]).Value -notin $allowed){exit 4};"
            "$allowed+='S-1-3-4';"  # OWNER RIGHTS refers only to the already trusted owner.
            "foreach($rule in $acl.GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier])){"
            "if($rule.AccessControlType -eq 'Allow' -and $rule.IdentityReference.Value -notin $allowed){exit 5}}}"
        )
        try:
            checked = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", code],
                capture_output=True, text=True, timeout=15,
                env={**os.environ, "RUBIKSQL_AUTHORITY_CACHE_DIR": str(path)})
        except (OSError, subprocess.SubprocessError) as error:
            raise AuthorityError("Git projection cache ACL verification unavailable") from error
        if checked.returncode:
            raise AuthorityError("Git projection cache requires elevation and restricted private ACLs")
    else:
        for candidate in (path, *path.rglob("*")) if recursive else (path,):
            info = candidate.lstat()
            # HB seals artifact children 0444/0555. The private enclosing root
            # denies traversal; children must still be owned and not writable by peers.
            forbidden = 0o077 if candidate == path and stat.S_ISDIR(info.st_mode) else 0o022
            if info.st_uid != os.getuid() or info.st_mode & forbidden:
                raise AuthorityError("Git projection cache must be a private owned directory")


def _private_open(path: str, flags: int) -> int:
    """``open()`` opener: a new cache file gets no group or other access, whatever the umask."""
    return os.open(path, flags, 0o600)


def _write_private(path: Path, data: bytes) -> None:
    """Write a cache file with mode 0o600, independent of the process umask.

    POSIX masks a new file's mode with the umask and keeps a rewritten file's mode, so
    a group-writable umask (Ubuntu's 002) would otherwise leave the peer write access
    that :func:`_check_cache_permissions` refuses. Windows files keep the private
    root's inherited ACL; mode 0o600 never makes them read-only.
    """
    with open(path, "wb", opener=_private_open) as stream:
        if os.name != "nt":
            os.fchmod(stream.fileno(), 0o600)
        stream.write(data)


def _make_lease_private(path: Path) -> None:
    """Make an existing lease file owner-only when it is an owned regular file (POSIX).

    The check and the change use one descriptor, opened without following a symlink or
    waiting for a FIFO writer, so a path swapped after the check is never changed. An
    absent lease or a symlink is left alone. Another process may hold the lease, so it is
    never replaced or removed.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return
    except OSError as error:
        if error.errno == errno.ELOOP:  # O_NOFOLLOW refuses a symlink.
            return
        raise
    try:
        info = os.fstat(descriptor)
        if stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) != 0o600:
            os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def _database_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_cache_stamp(entry: Path) -> tuple[int, int, int, int] | None:
    """Detect a fill while waiting for exclusivity; never used as integrity evidence."""
    try:
        info = (entry / "complete.json").lstat()
    except OSError:
        return None
    return info.st_ino, info.st_mtime_ns, info.st_ctime_ns, info.st_size


def _read_cache_key(identity: Mapping[str, Any]) -> str:
    """One stable directory per runtime and binding, independent of remote advancement."""
    return sha256_hex(canon({key: value for key, value in identity.items() if key not in {"commit", "head_sha256"}}))


def _read_cache_slot(identity: Mapping[str, Any]) -> str:
    """Keep HB's content-addressed artifact paths below Windows MAX_PATH.

    The directory name is only a slot, never identity evidence. Its completion
    manifest retains the full key and identity, both checked before reuse.
    """
    return _read_cache_key(identity)[:16]


def _checkpoint_read_cache(path: Path) -> None:
    """Seal a closed projection without moving a DB away from its module artifacts."""
    with contextlib.closing(sqlite3.connect(path)) as connection:
        if connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]:
            raise AuthorityError("projection still has an active SQLite reader")
        connection.execute("PRAGMA journal_mode=DELETE")


@contextlib.contextmanager
def _read_cache_lock(directory: Path, *, blocking: bool = True, shared: bool = False) -> Iterator[None]:
    """Lease a HOME's projections through close, across threads and processes.

    Sealed native readers share the lease; restoration, HB activation and pruning
    require exclusive access. The OS releases this lock on process exit. Keep its file in the private
    cache, independent of each caller's TEMP directory; it is created owner-only, so a seed
    source's lease passes its permission check. Reentrant callers restore
    privately: an outer reader must stay pinned while a nested writer advances.
    """
    directory = directory.resolve()
    active = getattr(_READ_CACHE_LEASES, "active", set())
    if directory in active:
        raise _ReadCacheBusy("nested Git projection requires a private restore")
    with _READ_CACHE_THREAD_LOCK:
        lock = _READ_CACHE_LOCKS.setdefault(directory, threading.Lock())
    # Each OS lease has its own file handle, so shared leases also arbitrate
    # correctly between threads. Retain the local mutex for exclusive callers.
    if not shared and not lock.acquire(blocking=blocking):
        raise _ReadCacheBusy("Git read projection cache thread lock is busy")
    try:
        with open(directory / "read-cache.lock", "a+b", opener=_private_open) as handle:
            if os.name == "nt":
                native = msvcrt.get_osfhandle(handle.fileno())
                position = _Overlapped()  # Byte zero, also used by older msvcrt leases.
                flags = 1 | (0 if shared else 2)  # FAIL_IMMEDIATELY | EXCLUSIVE_LOCK
                deadline = time.monotonic() + 600
                while not _KERNEL32.LockFileEx(native, flags, 0, 1, 0, ctypes.byref(position)):
                    code = ctypes.get_last_error()
                    if code != 33:  # ERROR_LOCK_VIOLATION
                        raise ctypes.WinError(code)
                    if not blocking or time.monotonic() >= deadline:
                        raise _ReadCacheBusy("Git read projection cache lock is busy")
                    time.sleep(0.1)

                def unlock() -> None:
                    if not _KERNEL32.UnlockFileEx(native, 0, 1, 0, ctypes.byref(position)):
                        raise ctypes.WinError(ctypes.get_last_error())
            else:
                import fcntl

                try:
                    fcntl.flock(handle.fileno(), (fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
                                | (0 if blocking else fcntl.LOCK_NB))
                except BlockingIOError as error:
                    raise _ReadCacheBusy("Git read projection cache lock is busy") from error
                unlock = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            try:
                _READ_CACHE_LEASES.active = active | {directory}
                yield
            finally:
                _READ_CACHE_LEASES.active = active
                unlock()
    finally:
        if not shared:
            lock.release()


def _remove_read_cache_entry(path: Path) -> None:
    # Older cache layouts can already exceed MAX_PATH. Use extended addressing
    # only to remove them; HB and Git continue to receive ordinary short paths.
    # absolute() is lexical: resolving here would follow a redirected entry.
    if os.name == "nt":
        absolute = str(path.absolute())
        if not absolute.startswith("\\\\?\\"):
            path = Path("\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\") else "\\\\?\\" + absolute)
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        path.unlink()
    elif getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        path.rmdir()  # Remove a Windows junction itself, never its target.
    else:
        # HB seals directories too; POSIX unlink needs a writable parent.
        path.chmod(0o700)
        for directory, children, _files in os.walk(path, followlinks=False):
            for child in list(children):
                target = Path(directory) / child
                info = target.lstat()
                if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    children.remove(child)
                else:
                    target.chmod(0o700)

        def writable_remove(function: Any, target: str, error: Any) -> None:
            if not isinstance(error[1], PermissionError):
                raise error[1]
            info = os.lstat(target)
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise error[1]
            # HB seals artifact files read-only. This handles their Windows
            # attributes, after all owning contexts and SQLite handles closed.
            os.chmod(target, 0o700)
            function(target)

        shutil.rmtree(path, onerror=writable_remove)


def _prune_read_cache(directory: Path) -> None:
    """Bound closed projections; callers retain the HOME lease until pruning ends."""
    # Include superseded long-name entries so the new layout remains bounded.
    entries = sorted((path for path in directory.iterdir() if re.fullmatch(r"[0-9a-f]{16}|[0-9a-f]{64}", path.name)),
                     key=lambda path: (path.lstat().st_mtime_ns, path.name), reverse=True)
    size, kept = 0, 0
    for entry in entries:
        if entry in _READ_CACHE_ACTIVE:
            continue  # The owning operation will prune again after closing every handle.
        if len(entry.name) == 64:
            _remove_read_cache_entry(entry)
            continue  # Retired layouts may be too deep even to stat their files.
        info = entry.lstat()
        if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            _remove_read_cache_entry(entry)
            continue
        entry_size = sum(path.stat().st_size for path in entry.rglob("*") if path.is_file())
        if kept >= _READ_CACHE_ENTRIES or size + entry_size > _READ_CACHE_BYTES:
            _remove_read_cache_entry(entry)
        else:
            size += entry_size
            kept += 1


class GitAuthorityConflict(AuthorityError):
    """The lease failed; discard the candidate before revalidating domain work."""

    def __init__(self, expected: str | None, observed: str | None, candidate: str | None = None) -> None:
        self.expected, self.observed, self.candidate = expected, observed, candidate
        super().__init__(f"Git authority conflict: expected {expected}, observed {observed}; candidate {candidate}")


class GitPublicationUnconfirmed(AuthorityError):
    """Publication may have committed, but remote confirmation is unavailable."""

    def __init__(self, candidate: str) -> None:
        self.candidate = candidate
        super().__init__(f"Git authority publication is unconfirmed; reconcile candidate {candidate} before any execution effect")


class GitAuthorityCheckpointError(AuthorityError):
    """An operation failed after a confirmed checkpoint that remains committed."""

    def __init__(self, commit: str, error: Exception) -> None:
        self.commit = commit
        super().__init__(f"authority checkpoint {commit} remains committed; the later operation failed: {error}")
        for note in getattr(error, "__notes__", ()):
            self.add_note(note)


def _transient_remote_read(result: subprocess.CompletedProcess[str] | subprocess.TimeoutExpired) -> bool:
    """Retry recognized transport failures only; permanent diagnostics take precedence.

    Git runs with LC_ALL=C. Inspect stderr without exposing remote diagnostics in
    errors or logs; TimeoutExpired can carry bytes even when text=True.
    """
    stderr = result.stderr or ""
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    detail = stderr.lower()
    if any(marker in detail for marker in (
            "permission denied", "access denied", "authentication failed", "authentication required",
            "could not read username", "could not read password", "terminal prompts disabled", "repository not found",
            "does not appear to be a git repository", "host key verification failed", "host-key verification failed",
            "remote host identification has changed", "no matching host key type found",
            "bad owner or permissions", "couldn't find remote ref", "invalid ref", "ambiguous", "not our ref",
            "unadvertised object", "cannot lock ref", "stale info", "incorrect old value", "reference already exists",
            "[rejected]", "[remote rejected]")):
        return False
    if isinstance(result, subprocess.TimeoutExpired):
        return True
    return result.returncode in {1, 128, 255} and any(marker in detail for marker in (
        "connection reset", "connection refused", "kex_exchange_identification", "ssh_exchange_identification", "connection closed", "closed by remote host", "connection abort", "timed out",
        "connection timeout", "remote end hung up unexpectedly", "unexpected disconnect", "broken pipe",
        "temporary failure in name resolution", "could not resolve host", "network is unreachable", "no route to host",
        "early eof", "tls connection was non-properly terminated", "rpc failed", "http/2 stream",
        "expected flush after ref listing", "gnutls_handshake() failed", "ssl_read", "failed to connect to",
        "requested url returned error: 500", "requested url returned error: 502",
        "requested url returned error: 503", "requested url returned error: 504"))


def _git(directory: Path | None, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    # Transport identity cannot be rewritten by user pushurl/insteadOf settings.
    environment = dict(os.environ)
    for key in tuple(environment):
        if key.startswith("GIT_CONFIG_") or key in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                                                    "GIT_ALTERNATE_OBJECT_DIRECTORIES"}:
            environment.pop(key)
    # Classify transport failures and push rejections from untranslated messages,
    # including the receive-pack of a local remote on localized hosts.
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0", LC_ALL="C")
    command = ["git", "-c", f"core.hooksPath={os.devnull}", "-c", "core.fsmonitor=false",
               "-c", "gc.auto=0", "-c", "maintenance.auto=false",
               "-c", "core.sshCommand=ssh -o BatchMode=yes"]
    if directory is not None:
        command.extend(["-C", str(directory)])
    try:
        # Git also creates retained object-cache files. Set only the child's mask;
        # changing this process's umask would race with concurrent authority users.
        result = subprocess.run([*command, *arguments], capture_output=True, text=True, timeout=90, env=environment,
                                umask=0o077 if os.name == "posix" else -1)
    except (OSError, subprocess.SubprocessError) as error:
        raise AuthorityError(f"Git authority operation unavailable: {arguments[0]}") from error
    if check and result.returncode:
        raise AuthorityError(f"Git authority operation refused: {arguments[0]}")
    return result


def _git_remote_read(directory: Path | None, *arguments: str, check: bool = True,
                     sleep: Callable[[float], None] = time.sleep) -> subprocess.CompletedProcess[str]:
    """Retry read-only transport failures through the ordinary Git runner.

    Call sites use only ls-remote or fetch into disposable checkouts. The caller
    retains its lease during attempts and backoff. The configured worst case is
    (1 + 2 retries) * 90 seconds + 1 + 3 seconds = 274 seconds. This inherits
    subprocess.run's startup/cleanup limitations, not a strict elapsed deadline,
    just like any single Git call. Tests inject sleep only at this retry layer.
    """
    if arguments[0] not in {"ls-remote", "fetch"}:
        raise ValueError("Git remote read requires ls-remote or read-only fetch")
    delays = _REMOTE_READ_RETRY_DELAYS
    attempt = 0
    while True:
        try:
            result = _git(directory, *arguments, check=False)
        except AuthorityError as error:
            cause = error.__cause__
            if attempt == len(delays) or not isinstance(cause, subprocess.TimeoutExpired) or not _transient_remote_read(cause):
                raise
        else:
            if not result.returncode or attempt == len(delays) or not _transient_remote_read(result):
                if check and result.returncode:
                    raise AuthorityError(f"Git authority operation refused: {arguments[0]}")
                return result
        sleep(delays[attempt])
        attempt += 1


def _lease_rejection(pushed: subprocess.CompletedProcess[str], ref: str, expected: str | None) -> str | None:
    """Classify a rejected ``push --porcelain`` of ``ref`` that lost its compare-and-swap race.

    The client reports a failed lease check as ``[rejected] (stale info)``. A
    receive-pack that batches ref updates (git 2.53, for example) reports a lost
    server-side CAS as ``incorrect old value provided`` (``reference already
    exists`` for a create), and a ref lock that another update held past the
    lock timeout as ``reference already exists``, with nothing on stderr. Older
    receive-packs (git 2.50) name the ref lock on stderr or in the status reason:
    ``is at <x> but expected <expected>`` after losing the race, or ``Unable to
    create '<ref>.lock': File exists`` for the busy lock. Authentication, hook
    and transport failures match none of these.

    Returns:
        ``"lease"`` for a lost compare-and-swap, ``"lock"`` for a busy ref lock
        whose winning update may not be visible yet, or None for any other rejection.
    """
    statuses = [fields[2] for fields in (line.split("\t") for line in (pushed.stdout or "").splitlines())
                if len(fields) == 3 and fields[0] == "!" and fields[1].endswith(":" + ref)]
    if "[rejected] (stale info)" in statuses:
        return "lease"
    reasons = [status[len("[remote rejected] ("):-1] for status in statuses
               if status.startswith("[remote rejected] (") and status.endswith(")")]
    if not reasons:
        return None
    lost = "incorrect old value provided" if expected is not None else "reference already exists"
    mismatch = f"but expected {expected}" if expected is not None else "reference already exists"
    locked = f"cannot lock ref '{ref}': "
    details = [*reasons, *(pushed.stderr or "").splitlines()]
    if lost in reasons or any(locked in text and mismatch in text for text in details):
        return "lease"
    if any(locked in text and ".lock': File exists" in text for text in details):
        return "lock"
    if "reference already exists" in reasons:
        # An update carries an old value, so only the busy ref lock can make a
        # batched receive-pack report it as a create that found the ref.
        return "lock"
    return None


def _historical_binding(head: Mapping[str, Any]) -> AuthorityBinding:
    record = head["binding"]
    return AuthorityBinding(record["schema"], record["authority_uuid"], record["instance_id"], Path(record["db_path"]))


class _Projection:
    """Private capability for one verified remote export and its local DB."""

    def __init__(self, owner: GitAuthority, *, commit: str | None = None) -> None:
        self.owner = owner
        self.temporary = tempfile.TemporaryDirectory(prefix="o", dir=owner.cache_dir)
        self.root = Path(self.temporary.name).resolve()
        self.export = self.root / "export"
        self.path = self.root / "p" / "control.db"
        self.commit: str | None = commit
        self.head: Mapping[str, Any] | None = None
        self.metadata: Mapping[str, bytes] = {}
        self.context: Any = None
        self.closed = False
        self.read_only = False
        self._hb_ready = False
        self._cache_entry: Path | None = None
        self._cache_key: str | None = None
        self._cache_lease = contextlib.ExitStack()
        self._cache_locked = False
        self._cache_shared = False
        self._cache_miss: tuple[Path, tuple[int, int, int, int] | None] | None = None
        self._cache_seed: tuple[Mapping[str, Any], Path] | None = None
        self._seed_attempted = False
        self._seed_record: Mapping[str, str] | None = None
        self._retained_reader: Any = None
        self._reader_lease = contextlib.ExitStack()
        self.writer: Any = None
        self._verified_exports: dict[Path, tuple[Mapping[str, Any], Mapping[str, bytes], str, bytes]] = {}
        self._verified_reads: dict[tuple[Any, ...], Any] = {}
        self.proof: Mapping[str, Any] | None = None
        self._proof_files: Mapping[str, bytes] = {}
        self._sealed_proof: Mapping[str, Any] | None = None
        self.force_full = os.environ.get("RUBIKSQL_AUTHORITY_VERIFY") == "full"

    def read_export(self, directory: Path) -> tuple[Mapping[str, Any], Mapping[str, bytes], str]:
        """Reuse export validation only while every actual byte and Git identity agrees.

        This memo belongs to one operation, never a persisted completion marker.
        Exact byte comparison avoids repeated hashing, JSON decoding and canonical
        replay; file enumeration, path safety and Git cleanliness remain checked.
        """
        from .authority import _export_paths, _read_export

        directory = Path(directory).resolve()
        held = self._verified_exports.get(directory)
        if held is None:
            head, files, commit = _read_export(directory, _verified=None if self.proof is None else self.proof["export"])
            try:
                raw_head = (directory / "HEAD.json").read_bytes()
                if json.loads(raw_head) != head:
                    raise ValueError("authority export HEAD changed during verification")
            except (OSError, ValueError, KeyError) as error:
                raise AuthorityError(f"authority export refused: {error}") from error
            self._verified_exports[directory] = (copy.deepcopy(head), dict(files), commit, raw_head)
        else:
            head, files, commit, raw_head = held
            try:
                paths = _export_paths(directory)
                for relative, expected in {**files, "HEAD.json": raw_head}.items():
                    if paths[relative].read_bytes() != expected:
                        raise ValueError("verified export bytes or path changed")
                present = set(paths)
                if (present != set(files) | {"HEAD.json"}
                        or _git(directory, "rev-parse", "HEAD").stdout.strip() != commit
                        or _git(directory, "status", "--porcelain").stdout.strip()):
                    raise ValueError("verified export file set or Git identity changed")
            except (OSError, ValueError, KeyError) as error:
                raise AuthorityError(f"authority export refused: {error}") from error
        return copy.deepcopy(head), dict(files), commit

    def permits_control(self, path: Path) -> bool:
        return not self.closed and not self.read_only and Path(path).resolve() == self.path

    def owns_reader(self, reader: Any) -> bool:
        return (not self.closed and Path(reader.path).resolve() == self.path
                and (self.context is not None or self.read_only)
                and reader.binding == self.native_binding
                and reader.binding.authority_uuid == self.owner.binding.authority_uuid)

    @property
    def native_binding(self) -> AuthorityBinding | GitAuthorityBinding:
        """Use the real Git identity for a host-neutral projection."""
        if self.head is None or self.head["binding"]["schema"] == B.SCHEMA_GIT_BINDING:
            return self.owner.binding
        return _historical_binding(self.head)

    def clean(self, reader: Any) -> bool:
        """Compare complete covered state without invoking head() recursively."""
        from .authority import _revision
        from .store import TABLES

        if self.head is None:
            return False
        events = reader._held_events()  # Compared in place: only identities and counts are read.
        return (self.head["covered_revision"] == dataclasses.asdict(_revision(events))
                and self.head["covered_event_ids"] == [row["object_id"] for row in events]
                and self.head["rows_digest"] == reader._row_digest(TABLES)
                and self.head["version_rows_digest"] == reader._complete_version_digest(TABLES)
                and self.head["workspace_digest"] == sha256_hex(canon(reader.operational_state()))
                and self.head["workspace_schemas_digest"] == reader._workspace_schemas_digest(TABLES))

    def decorate(self, reader: Any) -> Any:
        reader._git_projection = self
        reader.git_binding = self.owner.binding
        reader.authority_commit = self.commit
        reader._ancillary_files = self.metadata
        reader.export_anchor = self.export if self.clean(reader) else None
        return reader

    def bootstrap(self, *, commit: str | None = None, read_cache: bool = False, shared_cache: bool = False,
                  expected_commit: str | None = None) -> None:
        """Fetch and verify one exact export, then restore or catch up its private projection.

        Args:
            commit: Exact reachable commit, or the remote ref as fetched.
            read_cache: Reuse or refill the HOME's retained projection under its lease.
            shared_cache: The caller only reads native rows through :meth:`reader`, never
                writes, activates HB or changes this projection. A catch-up then skips derived
                HB state, and the caller reads the view that verified the final state.
            expected_commit: Refuse any other selected commit before touching retained state.
        """
        if read_cache:
            try:
                self._cache_lease.enter_context(_read_cache_lock(self.owner.cache_dir, shared=shared_cache))
                self._cache_locked = True
                self._cache_shared = shared_cache
            except _ReadCacheBusy:
                read_cache = False
        # Capture after the lease: a queued writer must not decide from the head
        # fetched before another local operation published.
        captured = self.owner.remote_commit()
        if captured is None:
            raise AuthorityError("selected Git authority ref is missing; explicit initialization or import is required")
        if commit is not None and _COMMIT.fullmatch(commit) is None:
            raise AuthorityError("authority commit must be a full 40-hex Git identity")
        self.export.mkdir()
        branch = self.owner.binding.ref.removeprefix("refs/heads/")
        _git(self.export, "init", "-b", branch)
        if read_cache:
            transport = self.owner.cache_dir / ("g" + sha256_hex(canon(B.binding_document(self.owner.binding)))[:12])
            transport.mkdir(exist_ok=True, mode=0o700)
            _check_cache_path(transport)
            _check_cache_permissions(transport)
            with _read_cache_lock(transport):
                repository = transport / "repository.git"
                if not repository.exists():
                    _git(None, "init", "--bare", str(repository))
                    repository.chmod(0o700)
                _check_cache_path(repository)
                _check_cache_permissions(repository)
                _git_remote_read(repository, "fetch", "--no-tags", self.owner.binding.remote,
                                 "+" + self.owner.binding.ref + ":" + self.owner.binding.ref)
                # Keep verified remote objects in this HOME; each operation keeps
                # its own refs/index and commits. No operation prunes these objects.
                _write_private(self.export / ".git" / "objects" / "info" / "alternates",
                               ((repository / "objects").as_posix() + "\n").encode("utf-8"))
                _git(self.export, "fetch", "--no-tags", str(repository), self.owner.binding.ref)
        else:
            _git_remote_read(self.export, "fetch", "--no-tags", self.owner.binding.remote, self.owner.binding.ref)
        fetched = _git(self.export, "rev-parse", "FETCH_HEAD").stdout.strip()
        if fetched != captured:
            # Another writer published between the capture and this fetch. A head
            # view follows that newer head, never a rewound ref; writers still
            # publish with a lease on it, and an expected commit stays exact below.
            # A pinned view verifies its exact commit against the fetched history.
            if commit is None and _git(self.export, "merge-base", "--is-ancestor", captured, fetched,
                                       check=False).returncode:
                raise GitAuthorityConflict(captured, fetched)
            captured = fetched
        selected = commit or captured
        if selected != captured:
            exists = _git(self.export, "cat-file", "-t", selected, check=False)
            ancestor = _git(self.export, "merge-base", "--is-ancestor", selected, captured, check=False)
            if exists.returncode or exists.stdout.strip() != "commit" or ancestor.returncode:
                raise AuthorityError("pinned authority commit is not in the canonical remote history")
        _git(self.export, "checkout", "-B", branch, selected)
        if read_cache and not self.force_full:
            # The seal is admitted only after the exact runtime identity, private
            # paths and closed DB bytes agree. Never consult another HOME/cache.
            try:
                raw_head = json.loads((self.export / "HEAD.json").read_bytes())
                identity = _read_cache_identity(raw_head, selected, self.owner.binding)
                entry = self.owner.cache_dir / "entries" / _read_cache_slot(identity)
                self._validate_cache_directory(entry)
                complete = json.loads((entry / "complete.json").read_bytes())
                previous = complete["identity"]
                if (complete["key"] == _read_cache_key(identity) == _read_cache_key(previous)
                        and _database_digest(entry / "p" / "control.db") == complete["db_sha256"]
                        and complete.get("proof_sha256") == sha256_hex((entry / "proof.json").read_bytes())
                        and not _git(self.export, "merge-base", "--is-ancestor", previous["commit"], selected,
                                     check=False).returncode):
                    proof = json.loads((entry / "proof.json").read_bytes())
                    proved_head = {key: value for key, value in proof["export"].items() if key != "workspace_entities"}
                    if (previous["head_sha256"] == sha256_hex(canon(proved_head))
                            and proof.get("db_sha256") == complete["db_sha256"]):
                        self.proof = proof
            except (OSError, ValueError, KeyError, TypeError, AuthorityError):
                self.proof = None
        try:
            head, files, actual = self.read_export(self.export)
        except AuthorityError:
            if self.proof is None:
                raise
            # A broken accelerator cannot make a valid committed authority
            # unreadable. Retry structural verification without any old proof;
            # actual export damage still fails that complete check loudly.
            self.proof = None
            head, files, actual = self.read_export(self.export)
        self.owner._check_head(head)
        if actual != selected:
            raise AuthorityError("fetched authority checkout differs from its selected commit")
        self.commit, self.head = selected, head
        # Refuse stale writers from the freshly verified remote before taking
        # down the retained entry's completion record or touching its storage.
        if expected_commit is not None and selected != expected_commit:
            raise GitAuthorityConflict(expected_commit, selected)
        self.metadata = {"metadata/tool-tiers.json": files["metadata/tool-tiers.json"]}
        _write_private(self.root / "tool-tiers.json", self.metadata["metadata/tool-tiers.json"])
        if read_cache:
            while not self._restore_cached(files, native=shared_cache):
                # Never upgrade while holding a shared lock: two simultaneous
                # misses would deadlock. Recheck under the exclusive lease;
                # another process may have restored or advanced the slot.
                self._cache_lease.close()
                self._cache_locked = self._cache_shared = False
                assert self._cache_miss is not None
                entry, stamp = self._cache_miss
                deadline = time.monotonic() + 600
                try:
                    while True:
                        # Observe only marker metadata outside the lease. A fill
                        # lets ALL waiters rejoin shared readers, including those
                        # still waiting when the first warm reader enters.
                        shared = _read_cache_stamp(entry) != stamp
                        try:
                            self._cache_lease.enter_context(_read_cache_lock(
                                self.owner.cache_dir, shared=shared, blocking=shared))
                            if not shared and _read_cache_stamp(entry) != stamp:
                                # The fill may finish between the polling stat
                                # and exclusive acquisition. Join shared readers
                                # before opening any retained database handles.
                                self._cache_lease.close()
                                continue
                            self._cache_locked = True
                            self._cache_shared = shared
                            break
                        except _ReadCacheBusy:
                            if shared or (os.name == "nt" and time.monotonic() >= deadline):
                                raise
                            time.sleep(0.1)
                except _ReadCacheBusy:
                    self._restore()
                    break
        else:
            self._restore()

    def _restore(self) -> None:
        from .authority import restore_authority

        self._load_context()
        restore_authority(self.export, into=self.path.parent, context=self.context, _projection=self)
        self._hb_ready = True

    def _load_context(self, *, reuse: bool = False) -> None:
        """Initialize private HB services without reconstructing canonical rows."""
        from heavenbase.context import Context, load_bootstrap

        if self.context is None:
            # The system registry/config is not covered by the committed export.
            # Recreate it from the installed build before executing HB services;
            # keep content-addressed module objects beside their control DB.
            if not reuse:
                for suffix in ("", "-wal", "-shm", "-journal"):
                    (self.root / "h" / ("system.db" + suffix)).unlink(missing_ok=True)
            bootstrap = dataclasses.asdict(load_bootstrap(home_path=Path(self.temporary.name) / "absent-bootstrap.yaml"))
            self.context = Context.load(bootstrap={"version": 1, **bootstrap, "root": str(self.root / "h")})
            if not reuse:
                self.context.modules().setup()

    def _verify_cached(self, export: Path, files: Mapping[str, bytes], *, retain: bool = False,
                       base: bool = False) -> None:
        """Verify retained rows against committed bytes, independently of the local marker.

        Args:
            retain: Keep this verified view for a native caller's reads (see :meth:`bootstrap`).
                Only the final state qualifies: nothing writes the projection afterwards.
            base: Also require exact raw encodings before trusting sealed HB derived state.
        """
        with contextlib.ExitStack() as lease:
            reader = lease.enter_context(self.reader())
            if base and self.proof is not None:
                from .projection_proof import raw_state

                if raw_state(reader) != self.proof["raw"]:
                    self._hb_ready = False
            reader._ancillary_files = {"metadata/tool-tiers.json": files["metadata/tool-tiers.json"]}
            checked = reader.verify(export_anchor=export)
            if not checked.ok:
                raise AuthorityError("cached authority projection refused: " + "; ".join(checked.violations))
            if retain:
                # Keep the verified SQLite transaction and its decoded read memo
                # through the caller's read. The HOME lease pins its files.
                self._retained_reader = reader
                self._reader_lease = lease.pop_all()

    def _prepare_hb(self, export: Path, files: Mapping[str, bytes], *, derived: bool = True,
                    then: Callable[[Any], None] | None = None) -> bool:
        """Rebuild unexported HB state before its first use of retained storage.

        Native verification covers canonical rows, not translations, catalogs or
        sparse postings. Recreate auxiliary storage through its HB owner instead
        of allowing cached metadata to affect a writer or facade. Catalog repair
        is a bulk derived-state rebuild; unchanged canonical rows are not upserted.

        Args:
            derived: Also rebuild sparse postings and the catalog for HB consumers. A
                native catch-up passes False: it only writes exact canonical rows, keeps
                every other step and check, and leaves the projection unprepared, so the
                next HB use drops the partial derived state and rebuilds it completely.
            then: Changes applied in the same HB session once the prepared state has
                verified against ``files``; the caller verifies their result.
        """
        from heavenbase.execution.contracts import RowOp
        from heavenbase.strategies import InlineColumn, JsonField, SparseGramIndex
        from .control import Control
        from .store import TABLES, TYPED_TABLES
        from .workspace_state import EVIDENCE_TABLES

        if self._hb_ready and then is None:
            return True
        canonical = set(TABLES.values()) | set(EVIDENCE_TABLES.values())
        with contextlib.closing(sqlite3.connect(self.path)) as connection:
            if connection.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' LIMIT 1").fetchone():
                return False
            objects = connection.execute("SELECT type,name FROM sqlite_master WHERE type IN ('view','table')").fetchall()
            for kind, name in sorted(objects, key=lambda item: item[0] != "view"):
                if name not in canonical and not name.startswith("sqlite_"):
                    quoted = name.replace('"', '""')
                    connection.execute(f'DROP {kind.upper()} "{quoted}"')
            connection.commit()
            extras = {}
            for entity, table in TYPED_TABLES.items():
                expected = {row["object_id"] for row in json.loads(files[f"native/{entity}.json"])}
                extras[entity] = [row[0] for row in connection.execute(f'SELECT object_id FROM "{table}"')
                                  if row[0] not in expected]
        self.read_only = False
        self._load_context()
        control = Control.open(self.path, context=self.context, create=False, _projection=self)
        try:
            for entity, identifiers in extras.items():
                for object_id in identifiers:
                    control.ws.delete((control.ws.entities[entity], object_id))
            indexed = set()
            for entity in {*TABLES, *EVIDENCE_TABLES}:
                for binding in control.ws.storage_plans[entity].bindings:
                    if binding.strategy == SparseGramIndex.identifier:
                        indexed.add((binding.backend, entity))
                    elif binding.strategy not in {InlineColumn.identifier, JsonField.identifier}:
                        return False
            for backend_name, entity in sorted(indexed):
                backend = control.ws.backends.get(backend_name)
                adapter = backend.strategy_adapter(SparseGramIndex.identifier)
                if adapter is None:
                    return False
                if derived:
                    # Missing sparse metadata seeds a fingerprint without backfilling
                    # retained arrays. Rebuild postings through their public owner.
                    adapter.upsert([RowOp(entity, row["object_id"], row) for row in backend.scan(entity)])
            if derived and not control.ws.repair()["catalog"]["ok"]:
                raise AuthorityError("cached authority catalog reconstruction failed")
            if then is not None:
                # Committed HB writes are visible to the native verifier: check the
                # prepared state before changing it, without reopening HB.
                self._verify_cached(export, files)
                then(control)
        finally:
            control.close()
        if then is None:
            self._verify_cached(export, files)
        self._hb_ready = derived
        return True

    def _activate_hb(self) -> None:
        """Prepare a cached HB projection, falling back to committed recovery on damage."""
        if self._hb_ready:
            self._load_context(reuse=True)
            return
        try:
            if self._prepare_hb(self.export, self.read_export(self.export)[1]):
                return
        except Exception:
            # Includes HB artifact/registry failures, such as SourceUnavailable.
            # A subsequent full restore must still report its own failure.
            pass
        self._discard_state()
        self._restore()

    def _restore_incremental(self, base_export: Path, base: Mapping[str, bytes], target: Mapping[str, bytes], *,
                             base_head: Mapping[str, Any], derived: bool) -> bool:
        """Synchronize changed complete-export rows through HB; retain full final verification.

        Events alone cannot reconstruct mutable evidence. Schema changes and
        triggers use ordinary recovery, whose row loading precedes trigger DDL.
        Comparisons and integrity checks still scan the complete covered state.

        Args:
            derived: Prepare derived HB state for a consumer that may open HB (see
                :meth:`_prepare_hb`); a native reader needs canonical rows only and
                keeps the view that verified them.
        """
        from .authority import _RESTORE_BATCH_ROWS, _export_format
        from .store import TABLES
        from .workspace_state import EVIDENCE_TABLES, decode_evidence_rows

        if self.head is None:
            raise ValueError("incremental projection lacks a verified target head")
        base_format, target_format = _export_format(base_head["schema"]), _export_format(self.head["schema"])

        before = json.loads(base["workspace/STATE.json"])["entities"]
        after = json.loads(target["workspace/STATE.json"])["entities"]
        if ({entity: value["schema"] for entity, value in before.items()}
                != {entity: value["schema"] for entity, value in after.items()}):
            return False
        # A prepared HB catalog does not make canonical SQL triggers safe for
        # replaying row deltas. Recovery loads rows before reinstating triggers.
        with contextlib.closing(sqlite3.connect(self.path)) as connection:
            if connection.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' LIMIT 1").fetchone():
                return False
        # A sealed catalog already contains the retained identities. Catch-up
        # must merge into it even when the eventual consumer is native-only.
        derived = derived or self._hb_ready

        def synchronize(control: Any) -> None:
            for entity in {**TABLES, **EVIDENCE_TABLES}:
                if entity in TABLES:
                    relative = f"native/{entity}.json"
                    if base[relative] == target[relative]:
                        continue
                    old_rows, new_rows = json.loads(base[relative]), json.loads(target[relative])
                else:
                    if (before[entity]["count"], before[entity]["rows_digest"]) == (after[entity]["count"], after[entity]["rows_digest"]):
                        continue
                    old_rows = decode_evidence_rows(entity, base, row_files=base_format == 4)
                    new_rows = decode_evidence_rows(entity, target, row_files=target_format == 4)
                old = {row["object_id"]: row for row in old_rows}
                new = {row["object_id"]: row for row in new_rows}
                entity_cls = control.ws.entities[entity]
                for object_id in sorted(old.keys() - new.keys()):
                    control.ws.delete((entity_cls, object_id))
                # Changed rows replace their versions; new rows still refuse an existing identity.
                # A native catch-up starts from an emptied catalog, so its rows are all fresh.
                for overwrite in (True, False):
                    rows = [row for object_id, row in new.items() if (object_id in old) == overwrite and old.get(object_id) != row]
                    for start in range(0, len(rows), _RESTORE_BATCH_ROWS):
                        control.ws.upsert_many(entity_cls, rows[start:start + _RESTORE_BATCH_ROWS], overwrite=overwrite,
                                               catalog_mode="merge" if derived else "fresh")

        if self._hb_ready:
            from .control import Control

            self.read_only = False
            self._load_context(reuse=True)
            control = Control.open(self.path, context=self.context, create=False, _projection=self)
            try:
                synchronize(control)
            finally:
                control.close()
        elif not self._prepare_hb(base_export, base, derived=derived, then=synchronize):
            return False
        self.read_only = True
        self._verify_cached(self.export, target, retain=not derived)
        return True

    def _discard_state(self) -> None:
        """Close every HB handle before removing failed state (also on Windows)."""
        if self.context is not None:
            self.context.close()
            self.context = None
        self._hb_ready = False
        self.read_only = False
        try:
            info = self.root.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            return
        for path in (self.path.parent, self.root / "h"):
            if path.exists() or path.is_symlink():
                _remove_read_cache_entry(path)

    def _validate_cache_directory(self, entry: Path) -> None:
        """Refuse redirected artifacts and DB sidecars before trusting or opening them."""
        for path in (entry, *entry.rglob("*")):
            info = path.lstat()
            if (not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode))
                    or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
                    or not path.resolve().is_relative_to(entry)):
                raise ValueError("projection cache contains a redirected path")
        database = entry / "p" / "control.db"
        if any(database.with_name(database.name + suffix).exists() for suffix in ("-wal", "-shm", "-journal")):
            raise ValueError("projection cache was not cleanly closed")
        if not (entry / "h" / "artifacts" / "modules").is_dir():
            raise ValueError("projection cache lacks its private module artifacts")

    def _stage_read_cache_seed(self, identity: Mapping[str, Any]) -> None:
        """Copy the newest compatible sealed DB under its source lease, never its marker.

        Search the selected cache and explicitly opted-in source roots only.
        Cross-root shared leases are nonblocking to avoid lock cycles.
        A shared miss stages before waiting for the destination's exclusive lease,
        so copying itself can overlap other native readers of the source root.
        """
        if self._seed_attempted:
            return
        self._seed_attempted = True
        roots = (self.owner.cache_dir, *self.owner.seed_dirs)
        compatible = {key: value for key, value in identity.items() if key not in {"code", "commit", "head_sha256"}}
        destination = self.owner.cache_dir / "entries" / _read_cache_slot(identity)
        staged = Path(self.temporary.name) / "seed.db"
        with contextlib.ExitStack() as leases:
            candidates = []
            for root in dict.fromkeys(roots):
                try:
                    _check_cache_path(root)
                    root = root.resolve()
                    _check_cache_permissions(root)
                    if root != self.owner.cache_dir:
                        lock = root / "read-cache.lock"
                        _check_cache_path(lock)
                        _check_cache_permissions(lock)
                        leases.enter_context(_read_cache_lock(root, shared=True, blocking=False))
                    entries = root / "entries"
                    _check_cache_path(entries)
                    _check_cache_permissions(entries)
                    for entry in entries.iterdir():
                        if entry != destination and re.fullmatch(r"[0-9a-f]{16}", entry.name):
                            candidates.append((entry.lstat().st_mtime_ns, entry))
                except (AuthorityError, OSError):
                    continue  # An unavailable/untrusted source cannot prevent ordinary recovery.
            for _mtime, entry in sorted(candidates, key=lambda item: (item[0], str(item[1])), reverse=True):
                copying = False
                try:
                    self._validate_cache_directory(entry)
                    complete = json.loads((entry / "complete.json").read_bytes())
                    previous = complete["identity"]
                    if (not isinstance(previous, dict)
                            or {key: value for key, value in previous.items()
                                if key not in {"code", "commit", "head_sha256"}} != compatible
                            or complete["key"] != _read_cache_key(previous)
                            or entry.name != _read_cache_slot(previous)):
                        continue
                    _check_cache_permissions(entry, recursive=True)
                    commit = previous["commit"]
                    if (not isinstance(commit, str) or not _COMMIT.fullmatch(commit)
                            or _git(self.export, "merge-base", "--is-ancestor", commit, self.commit, check=False).returncode):
                        continue
                    if sum(path.stat().st_size for path in entry.rglob("*") if path.is_file()) > _READ_CACHE_BYTES:
                        continue
                    # No HB registry or artifact path is relocated. Current HB
                    # reconstructs those owners at the final destination below.
                    copying = True
                    shutil.copyfile(entry / "p" / "control.db", staged)
                    staged.chmod(0o600)
                    if _database_digest(staged) != complete["db_sha256"]:
                        raise ValueError("seed database digest differs")
                    self._cache_seed = (complete, staged)
                    return
                except Exception:
                    staged.unlink(missing_ok=True)
                    if copying:
                        # A failed/interrupted copy uses ordinary full recovery.
                        return
                    continue

    def _verify_cache_base(self, previous: Mapping[str, Any], identity: Mapping[str, Any],
                           files: Mapping[str, bytes], *, native: bool, seeded: bool = False) -> None:
        """Use the same Git/native proof for warm state and an untrusted seed copy."""
        commit = previous["commit"]
        if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
            raise ValueError("projection commit is invalid")
        if _git(self.export, "merge-base", "--is-ancestor", commit, self.commit, check=False).returncode:
            raise ValueError("projection commit is outside the selected history")
        ancestor = Path(self.temporary.name) / "cache-ancestor"
        try:
            if commit == self.commit:
                export, head, base = self.export, self.head, files
            else:
                _git(self.export, "worktree", "add", "--detach", str(ancestor), commit)
                export = ancestor
                head, base, actual = self.read_export(export)
                self.owner._check_head(head)
                if actual != commit or head["binding"] != self.head["binding"]:
                    raise ValueError("projection ancestor differs from its committed export")
            expected = {**identity, "commit": commit, "head_sha256": sha256_hex(canon(head))}
            if seeded:
                expected["code"] = previous["code"]
            if previous != expected:
                raise ValueError("projection head identity differs")
            self.read_only = True
            self._proof_files = base
            self._verify_cached(export, base, retain=native and not seeded and commit == self.commit, base=True)
            if commit != self.commit:
                if not self._restore_incremental(export, base, files, base_head=head, derived=seeded or not native):
                    raise ValueError("projection schema requires complete recovery")
            elif seeded and not self._prepare_hb(export, base):
                raise ValueError("seed schema requires complete recovery")
            if seeded:
                # Even native seeds rebuild registry/catalog/index owners and let
                # HB verify installed artifacts before their final native proof.
                self.read_only = True
                self._verify_cached(self.export, files, retain=native)
        finally:
            self._verified_exports.pop(ancestor.resolve(), None)
            if ancestor.exists():
                _git(self.export, "worktree", "remove", "--force", str(ancestor), check=False)

    def _restore_cached(self, files: Mapping[str, bytes], *, native: bool) -> bool:
        """Verify a leased entry; a shared miss must retry under exclusive access.

        Args:
            native: The caller only reads native rows (see :meth:`bootstrap`), also after a
                shared miss continues under exclusive access.
        """
        if self.head is None or self.commit is None or not self._cache_locked:
            raise AuthorityError("read cache requires a verified exact Git export")
        try:
            identity = _read_cache_identity(self.head, self.commit, self.owner.binding)
        except AuthorityError:
            # Without installed build provenance, use only the operation's
            # private restore. No retained entry has been selected or opened.
            self._cache_lease.close()
            self._cache_locked = self._cache_shared = False
            self._cache_seed = None
            self._restore()
            return True
        entries = self.owner.cache_dir / "entries"
        if not self._cache_shared:
            entries.mkdir(exist_ok=True, mode=0o700)
        entry = entries / _read_cache_slot(identity)
        self._cache_entry = None if self._cache_shared else entry
        self._cache_key = _read_cache_key(identity)
        self.root, self.path = entry, entry / "p" / "control.db"
        if self._cache_shared:
            self._cache_miss = (entry, _read_cache_stamp(entry))
        if not self._cache_shared:
            _READ_CACHE_ACTIVE.add(entry)
        existed = entry.exists() or entry.is_symlink()
        if existed:
            try:
                self._validate_cache_directory(entry)
                complete = json.loads((entry / "complete.json").read_bytes())
                previous = complete["identity"]
                if (not isinstance(previous, dict) or complete.get("key") != self._cache_key
                        or _read_cache_key(previous) != self._cache_key):
                    raise ValueError("projection identity differs")
                if self._cache_shared and previous != identity:
                    raise ValueError("projection requires exclusive recovery")
                if _database_digest(self.path) != complete["db_sha256"]:
                    raise ValueError("projection database digest differs")
                if self.proof is not None:
                    # Derived HB storage is reusable only when every sealed artifact
                    # agrees too; canonical proof reuse does not trust a catalog alone.
                    artifacts = complete.get("hb_files")
                    if artifacts is not None and self._hb_artifacts() == artifacts:
                        self._hb_ready = True
                if not self._cache_shared:
                    (entry / "complete.json").unlink()
                self._verify_cache_base(previous, identity, files, native=native)
                self._seed_record = complete.get("seed")
                if self._cache_shared:
                    with _READ_CACHE_THREAD_LOCK:
                        _READ_CACHE_READERS[entry] = _READ_CACHE_READERS.get(entry, 0) + 1
                        _READ_CACHE_ACTIVE.add(entry)
                        self._cache_entry = entry
                else:
                    _write_private(entry / "tool-tiers.json", self.metadata["metadata/tool-tiers.json"])
                return True
            except Exception:
                # Cached state is disposable, including HB-specific failures.
                # Shared readers may neither discard nor repair it. Full recovery
                # below remains outside this catch and fails loudly.
                if not self._cache_shared:
                    # A retained view must release its database before the entry is removed.
                    self._reader_lease.close()
                    self._retained_reader = None
                    self.proof, self._proof_files = None, {}
                    self._discard_state()
                    _remove_read_cache_entry(entry)
        if self._cache_shared:
            if not existed:
                self._stage_read_cache_seed(identity)
            self._reader_lease.close()
            self._retained_reader = None
            self._cache_entry = self._cache_key = None
            self.root = Path(self.temporary.name).resolve()
            self.path = self.root / "p" / "control.db"
            self.read_only = False
            return False
        entry.mkdir(mode=0o700)
        _write_private(entry / "tool-tiers.json", self.metadata["metadata/tool-tiers.json"])
        if not existed:
            self._stage_read_cache_seed(identity)
            if self._cache_seed is not None:
                complete, staged = self._cache_seed
                try:
                    self.path.parent.mkdir(mode=0o700)
                    staged.replace(self.path)
                    _check_cache_permissions(entry, recursive=True)
                    self._verify_cache_base(complete["identity"], identity, files, native=native, seeded=True)
                    self._seed_record = {"key": complete["key"], "commit": complete["identity"]["commit"]}
                    return True
                except Exception:
                    self._reader_lease.close()
                    self._retained_reader = None
                    self._discard_state()
        self._seed_record = None
        self._restore()
        self.read_only = True
        return True

    def _publish_read_cache(self, identity: Mapping[str, Any], entry: Path) -> None:
        """Seal closed state in place, recording successful seed provenance.

        On POSIX, SQLite and HB create the database and HB's home with the process
        umask. Before the marker admits the entry as a later seed source, one walk that
        neither follows nor changes a symlink makes the database owner-only and leaves no
        path with peer write access; HB's sealed 0o444/0o555 artifacts carry none and keep
        their modes.
        """
        _checkpoint_read_cache(self.path)
        if os.name != "nt":
            for path in entry.rglob("*"):
                mode = path.lstat().st_mode
                private = 0o600 if path == self.path else stat.S_IMODE(mode) & ~0o022
                if stat.S_IMODE(mode) != private and not stat.S_ISLNK(mode):
                    path.chmod(private)
        complete = {"key": _read_cache_key(identity), "identity": identity, "db_sha256": _database_digest(self.path)}
        if self._sealed_proof is not None:
            payload = canon({**self._sealed_proof, "db_sha256": complete["db_sha256"]}).encode("utf-8")
            _write_private(entry / "proof.json", payload)
            complete["proof_sha256"] = sha256_hex(payload)
        if self._hb_ready:
            complete["hb_files"] = self._hb_artifacts()
        if self._seed_record is not None:
            complete["seed"] = self._seed_record
        staged = entry / "complete.tmp"
        _write_private(staged, canon(complete).encode("utf-8"))
        staged.replace(entry / "complete.json")
        os.utime(entry, None)

    def _hb_artifacts(self) -> Mapping[str, str]:
        """Fingerprint derived files only after HB handles have closed."""
        return {path.relative_to(self.root).as_posix(): _database_digest(path)
                for folder in (self.root / "h", self.path.parent) for path in folder.rglob("*")
                if path.is_file() and path != self.path}

    def bootstrap_empty(self, tool_tiers: Mapping[str, str]) -> None:
        """Create one private seed workspace without an existing export or ECS identity."""
        from heavenbase.context import Context, load_bootstrap
        from .control import Control
        from .store import _bytes

        if self.context is not None or self.head is not None or self.closed:
            raise AuthorityError("Git projection was already initialized")
        tiers = dict(tool_tiers)
        if any(not isinstance(name, str) or not name or tier not in {"hb-executed", "host-verified"}
               for name, tier in tiers.items()):
            raise AuthorityError("fresh epoch tool tier policy is invalid")
        self.export.mkdir()
        _git(self.export, "init", "-b", self.owner.binding.ref.removeprefix("refs/heads/"))
        self.metadata = {"metadata/tool-tiers.json": _bytes(tiers)}
        _write_private(self.root / "tool-tiers.json", self.metadata["metadata/tool-tiers.json"])
        bootstrap = dataclasses.asdict(load_bootstrap(home_path=self.root / "absent-bootstrap.yaml"))
        self.context = Context.load(bootstrap={"version": 1, **bootstrap, "root": str(self.root / "h")})
        self.context.modules().setup()
        control = Control.open(self.path, context=self.context, _projection=self)
        control.close()

    @contextlib.contextmanager
    def reader(self) -> Iterator[Any]:
        from .authority import Authority

        if (self.context is None and not self.read_only) or self.closed:
            raise AuthorityError("Git projection is not an open private workspace")
        if self._retained_reader is not None:
            yield self._retained_reader
            return
        with Authority._read_native(self.path, binding=self.native_binding) as reader:
            self.decorate(reader)
            if self.writer is not None:
                reader._writer_owner = self.writer
                reader._writer_token = self.writer._writer_token
            yield reader

    def publish(self, secrets: Any) -> Any:
        from .authority import EXPORT_FORMAT, ExportResult
        from .store import _bytes
        from .authority import scan_secrets

        if self.read_only:
            raise AuthorityError("a read-only Git projection cannot publish")
        with self.reader() as reader:
            checked = reader.verify(require_export=False)
            if not checked.ok:
                raise AuthorityError("Git publication integrity refused: " + "; ".join(checked.violations))
            current = reader.head()
            files = dict(reader.workspace_files(tools_release=current.release, export_format=EXPORT_FORMAT))
            head = reader.covered_head(files, export_format=EXPORT_FORMAT)
            if self.head is not None and head["binding"] != self.head["binding"]:
                # No Git publication rebinds: a fresh epoch keeps its marker's binding
                # and an imported history the identity its committed head covers.
                raise AuthorityError("Git publication integrity refused: covered binding differs from the committed head")
            files["HEAD.json"] = _bytes(head)
            scan_secrets(files, secrets)
        # An absent-ref epoch has no prior export. Existing epochs must still
        # verify their committed file map before replacing any covered bytes.
        previous_files = self.read_export(self.export)[1] if self.commit is not None else {}
        self.owner._replace_files(self.export, files, previous=previous_files)
        self._verified_exports.pop(self.export.resolve(), None)
        previous = self.commit
        candidate = self.owner._commit(self.export, previous, f"Commit authority {current.release} at event {current.revision.event_count}")
        verified_head, _verified_files, confirmed = self.read_export(self.export)
        if confirmed != candidate or verified_head != head:
            raise AuthorityError("candidate Git authority commit differs from its staged export")
        self.owner._publish(self.export, expected=previous, candidate=candidate)
        # Record the durable checkpoint before any fallible caller-side work.
        self.commit, self.head = candidate, verified_head
        if self.writer is not None:
            self.writer._batch_dirty = False
            self.writer.control.ws._rubiksql_authority_commit = candidate
        # The leased projection stays in place. Seal it only after all writer
        # and HB handles close; never move artifacts or prune an active workspace.
        return ExportResult(candidate, sha256_hex(files["HEAD.json"]), current.revision, current.revision)

    def close(self) -> None:
        if self.closed:
            return
        if self._cache_entry is not None and not self._cache_shared and sys.exception() is None:
            from .projection_proof import capture

            # A failed or dirty stage must never acquire a verification seal.
            with contextlib.suppress(Exception):
                with self.reader() as reader:
                    checked = reader.verify(export_anchor=self.export)
                    if checked.ok:
                        self._sealed_proof = capture(reader, self.head, self.read_export(self.export)[1])
        self.closed = True
        pending = sys.exception()
        try:
            self._reader_lease.close()
            self._retained_reader = None
            if self.context is not None:
                self.context.close()
                self.context = None
            if self._cache_entry is not None and not self._cache_shared:
                if pending is not None:
                    # The writer's monitor and facade have already closed.
                    # Discard failed stages only after the HB context releases
                    # its handles, and surface removal failures on every OS.
                    _remove_read_cache_entry(self._cache_entry)
                else:
                    # Cache maintenance cannot turn a confirmed CAS into a failed
                    # publication. A missing marker forces full recovery next time.
                    with contextlib.suppress(Exception):
                        if self.head is not None and self.commit is not None:
                            identity = _read_cache_identity(self.head, self.commit, self.owner.binding)
                            if _read_cache_key(identity) == self._cache_key:
                                self._publish_read_cache(identity, self._cache_entry)
                    _READ_CACHE_ACTIVE.discard(self._cache_entry)
                    with contextlib.suppress(Exception):
                        _prune_read_cache(self._cache_entry.parent)
        finally:
            if self._cache_entry is not None:
                if self._cache_shared:
                    with _READ_CACHE_THREAD_LOCK:
                        readers = _READ_CACHE_READERS[self._cache_entry] - 1
                        if readers:
                            _READ_CACHE_READERS[self._cache_entry] = readers
                        else:
                            del _READ_CACHE_READERS[self._cache_entry]
                            _READ_CACHE_ACTIVE.discard(self._cache_entry)
                else:
                    _READ_CACHE_ACTIVE.discard(self._cache_entry)
            try:
                if Path(self.temporary.name).exists():
                    _remove_read_cache_entry(Path(self.temporary.name))
                self.temporary.cleanup()
            finally:
                self._verified_exports.clear()
                self._verified_reads.clear()
                self._cache_lease.close()


class GitAuthority:
    """Read, stage and commit the explicitly selected authority ref.

    Args:
        binding: Validated Git identity. Production accepts only the canonical
            GitHub remote; explicit local bare repositories support offline tests.
        cache_dir: Private parent for disposable projections and bounded read
            snapshots. Defaults to RUBIKSQL_AUTHORITY_CACHE_DIR when inherited,
            otherwise the current account's .rhb/authority-read directory. Task
            HOME and USERPROFILE isolation do not create new projection caches.
            Earlier compatible entries in this selected root may seed new entries.
            On POSIX the root and its existing owned lease file become owner-only.
        seed_dirs: Additional absolute private source roots, opted in explicitly.
            None reads RUBIKSQL_AUTHORITY_SEED_DIRS, an os.pathsep-separated list;
            absent or empty means no extra roots. An empty tuple ignores that
            environment setting. Neither the account cache nor an overridden
            RUBIKSQL_AUTHORITY_CACHE_DIR is searched implicitly. These roots are
            only read; their permissions are never repaired.

    Raises:
        AuthorityError: Invalid schema, UUID, remote, full ref, or cache path.
    """

    def __init__(self, binding: GitAuthorityBinding, *, cache_dir: Path | None = None,
                 seed_dirs: tuple[Path, ...] | None = None) -> None:
        try:
            valid_uuid = str(uuid.UUID(binding.authority_uuid)) == binding.authority_uuid
        except (AttributeError, TypeError, ValueError):
            valid_uuid = False
        if (not isinstance(binding, GitAuthorityBinding) or binding.schema != B.SCHEMA_GIT_BINDING
                or not valid_uuid or not B.valid_authority_ref(binding.ref)):
            raise AuthorityError("Git authority binding identity differs")
        if binding.remote not in B._REMOTES:
            # Local test fixtures are an explicit typed identity, never package configuration.
            local = Path(binding.remote)
            if (not local.is_absolute() or str(local.resolve()) != binding.remote or not local.is_dir()
                    or _git(local, "rev-parse", "--is-bare-repository").stdout.strip() != "true"):
                raise AuthorityError("Git authority remote is not the canonical repository or an explicit local bare fixture")
        self.binding = binding
        if seed_dirs is None:
            configured_seeds = os.environ.get("RUBIKSQL_AUTHORITY_SEED_DIRS", "")
            seed_dirs = tuple(Path(value) for value in configured_seeds.split(os.pathsep)) if configured_seeds else ()
        if any(not path.is_absolute() for path in seed_dirs):
            raise AuthorityError("Git projection seed roots must be absolute private paths")
        self.seed_dirs = tuple(dict.fromkeys(seed_dirs))
        selected = Path(cache_dir) if cache_dir is not None else default_authority_cache_dir()
        selected = selected.absolute()
        _check_cache_path(selected)
        self.cache_dir = selected.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name != "nt":
            if self.cache_dir.stat().st_uid != os.getuid():
                raise AuthorityError("Git projection cache must be a private owned directory")
            self.cache_dir.chmod(0o700)
            # Older code created the lease file with the process umask, but a seed source's
            # lease must deny peer write access. Repair only this selected root's own lease,
            # never a seed root's.
            _make_lease_private(self.cache_dir / "read-cache.lock")
        _check_cache_permissions(self.cache_dir)

    def remote_commit(self) -> str | None:
        """Read the exact configured remote ref, with no cached fallback.

        Returns:
            Full Git SHA or None when the remote ref does not exist.

        Raises:
            AuthorityError: Unavailable remote or malformed/ambiguous response.
        """
        output = _git_remote_read(None, "ls-remote", "--refs", self.binding.remote, self.binding.ref).stdout.strip()
        if not output:
            return None
        lines = output.splitlines()
        if len(lines) != 1:
            raise AuthorityError("canonical authority ref returned an ambiguous identity")
        fields = lines[0].split()
        if len(fields) != 2 or fields[1] != self.binding.ref or _COMMIT.fullmatch(fields[0]) is None:
            raise AuthorityError("canonical authority ref returned an invalid identity")
        return fields[0]

    def _check_head(self, head: Mapping[str, Any], *, source_export: bool = False) -> None:
        if (head.get("schema") not in {"rubiksql.authority-head/3", "rubiksql.authority-head/4"}
                or head.get("binding", {}).get("authority_uuid") != self.binding.authority_uuid):
            raise AuthorityError("Git authority requires a complete typed export (format 3 or 4) of its exact UUID")
        if (head["binding"].get("schema") == B.SCHEMA_GIT_BINDING
                and head["binding"] != B.binding_document(self.binding)):
            raise AuthorityError("covered Git authority binding differs from its selected remote identity")
        if head["binding"].get("schema") not in {B.SCHEMA_BINDING, B.SCHEMA_GIT_BINDING}:
            raise AuthorityError("covered authority binding schema is unsupported")
        if not source_export and "metadata/tool-tiers.json" not in head.get("files", {}):
            raise AuthorityError("Git authority requires its committed tool tier policy")

    @staticmethod
    def _replace_files(checkout: Path, files: Mapping[str, bytes], *,
                       previous: Mapping[str, bytes] | None = None) -> None:
        from .authority import _export_paths

        existing = _export_paths(checkout)
        tracked = _git(checkout, "ls-files").stdout.splitlines()
        for relative in set(tracked) - set(files):
            if relative in existing:
                existing[relative].unlink()
        for relative, value in files.items():
            name = PurePosixPath(relative)
            if (name.is_absolute() or relative != name.as_posix() or not name.parts
                    or any(part in {"..", ".git"} for part in name.parts) or "\\" in relative or ":" in relative):
                raise AuthorityError("authority file path escapes checkout")
            if previous is not None and previous.get(relative) == value:
                continue
            path = checkout / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative not in existing or existing[relative].read_bytes() != value:
                path.write_bytes(value)
        _git(checkout, "add", "--all")

    @staticmethod
    def _commit(checkout: Path, parent: str | None, message: str) -> str:
        if parent is not None and _git(checkout, "rev-parse", "HEAD").stdout.strip() != parent:
            raise AuthorityError("authority staged commit parent differs")
        changed = _git(checkout, "diff", "--cached", "--name-only").stdout.strip()
        if changed or parent is None:
            _git(checkout, "-c", "user.name=RubikSQL authority", "-c", "user.email=authority@rubiksql.local",
                 "-c", "commit.gpgSign=false", "commit", "-m", message)
        candidate = _git(checkout, "rev-parse", "HEAD").stdout.strip()
        if candidate != parent:
            parents = _git(checkout, "show", "-s", "--format=%P", candidate).stdout.strip().split()
            if parents != ([parent] if parent is not None else []):
                raise AuthorityError("authority publication must have exactly its captured remote parent")
        return candidate

    def _includes(self, checkout: Path, head: str, commit: str) -> bool:
        """Fetch the remote ref and report whether the history of ``head`` contains ``commit``.

        Raises:
            AuthorityError: The remote ref cannot be fetched.
        """
        _git_remote_read(checkout, "fetch", "--no-tags", self.binding.remote, self.binding.ref)
        return not _git(checkout, "merge-base", "--is-ancestor", commit, head, check=False).returncode

    def _confirm_unchanged(self, checkout: Path, head: str, commit: str) -> bool:
        """Report whether the advanced remote ``head`` contains a batch's unchanged ``commit``.

        The batch pushed nothing. The read-only fetch retries transient transport
        failures within a small bound without replaying this durable operation.

        Raises:
            AuthorityError: Confirmation failed permanently or exhausted its read
                retries; the batch must not be replayed automatically.
        """
        try:
            return self._includes(checkout, head, commit)
        except AuthorityError as error:
            raise AuthorityError(
                f"Git authority confirmation unavailable: nothing was pushed and {commit} was this writer's fetched "
                f"head, but fetching the advanced remote head {head} to prove that it contains {commit} failed; "
                "check that history before running the operation again") from error

    def _publish(self, checkout: Path, *, expected: str | None, candidate: str) -> None:
        configured = B.load_binding()
        if isinstance(configured, GitAuthorityBinding) and configured != self.binding:
            raise AuthorityError("configured Git authority identity differs from this publication")
        observed = self.remote_commit()
        if candidate == expected:
            # A batch that staged nothing pushes nothing: its unchanged commit is
            # durable while the remote head is that commit or contains it.
            if observed == candidate or (observed is not None and self._confirm_unchanged(checkout, observed, candidate)):
                return
            raise GitAuthorityConflict(expected, observed, candidate)
        if observed != expected:
            raise GitAuthorityConflict(expected, observed, candidate)
        pushed = None
        try:
            pushed = _git(checkout, "push", "--porcelain", f"--force-with-lease={self.binding.ref}:{expected or ''}",
                          self.binding.remote, f"{candidate}:{self.binding.ref}", check=False)
        except AuthorityError:
            # A transport failure can occur after the remote accepted the CAS.
            pass
        rejection = (_lease_rejection(pushed, self.binding.ref, expected)
                     if pushed is not None and pushed.returncode else None)
        try:
            confirmed = self.remote_commit()
            for delay in ((0.1, 0.2, 0.4, 0.8) if rejection == "lock" else ()):
                # The update that held the ref lock past receive-pack's timeout
                # can become visible only after this rejection.
                if confirmed != expected:
                    break
                time.sleep(delay)
                confirmed = self.remote_commit()
        except AuthorityError as error:
            if rejection is not None:
                # Git refused this exact update, so nothing was published; the
                # head it lost to is unknown here and the next attempt reads it.
                raise GitAuthorityConflict(expected, None, candidate) from error
            raise GitPublicationUnconfirmed(candidate) from error
        if confirmed != candidate:
            # Another writer may already have extended our accepted commit.
            # Establish reachability before classifying this as a lost lease;
            # replaying a possibly published reservation would duplicate effects.
            # A refused update never reached the remote and needs no such proof.
            try:
                if rejection is None and confirmed is not None and self._includes(checkout, confirmed, candidate):
                    return
            except AuthorityError as error:
                raise GitPublicationUnconfirmed(candidate) from error
            if pushed is None or pushed.returncode == 0:
                raise GitPublicationUnconfirmed(candidate)
            if confirmed != expected and rejection is None:
                # A moved ref alone does not turn authentication, hooks or a
                # transport failure into a retryable lease rejection. The client
                # lease check, the server-side CAS and its ref lock all can lose.
                raise AuthorityError("Git authority push failed without a confirmed lease conflict")
            raise GitAuthorityConflict(expected, confirmed, candidate)

    def _epoch_request(self, seed: Any, request_id: str, user_instruction: str) -> Mapping[str, Any]:
        """Check explicit isolation before any authority transport is consulted."""
        if not isinstance(request_id, str) or not request_id.strip():
            raise AuthorityError("fresh epoch initialization requires a nonempty request identity")
        if not isinstance(user_instruction, str) or not user_instruction.strip():
            raise AuthorityError("fresh epoch initialization requires the verbatim user instruction")
        identity = seed.identity
        if (not isinstance(identity, Mapping) or not isinstance(identity.get("source_commit"), str)
                or _COMMIT.fullmatch(identity["source_commit"]) is None):
            raise AuthorityError("fresh epoch seed source identity differs")
        if B.selected_binding_path() is None or B.load_binding() != self.binding:
            raise AuthorityError("fresh epoch initialization requires its explicit selected binding file")
        default = B.load_binding(B.PACKAGE_BINDING)
        if (self.binding.authority_uuid == default.authority_uuid
                or isinstance(default, GitAuthorityBinding) and self.binding.ref == default.ref):
            raise AuthorityError("fresh epoch must use a different ref and UUID from the package authority")
        if active_writer(self.binding) is not None:
            raise AuthorityError("fresh epoch initialization cannot nest inside an authority writer")
        return dict(identity)

    @contextlib.contextmanager
    def _seed_writer(self, projection: _Projection) -> Iterator[Any]:
        """Hold the ordinary writer capabilities while capturing fresh definitions."""
        from .authority import Authority, _LOCK_STATE, authority_lock
        from .control import Control

        control = Control.open(projection.path, context=projection.context, create=False, _projection=projection)
        try:
            writer = Authority(control, self.binding)
            writer._git_projection, writer.git_binding = projection, self.binding
            writer.control.ws._rubiksql_authority_binding = self.binding
            writer.control.ws._rubiksql_authority_db = str(projection.path)
            writer.control.ws._rubiksql_lab_dir = str(projection.path.parent)
            writer.control.ws._rubiksql_authority_commit = projection.commit
            projection.writer = writer
            writers = getattr(_WRITERS, "active", {})
            with authority_lock(projection.path):
                writer._writer_active = True
                writer._writer_thread = threading.get_ident()
                writer._writer_token = object()
                _WRITERS.active = {**writers, self.binding: writer}
                native_writers = getattr(_LOCK_STATE, "writers", {})
                _LOCK_STATE.writers = {**native_writers, projection.path: writer}
                try:
                    yield writer
                finally:
                    _WRITERS.active = writers
                    _LOCK_STATE.writers = native_writers
                    writer._writer_active = False
                    writer._writer_token = None
                    projection.writer = None
                    writer._close_read_memo()
        finally:
            pending = sys.exception()
            try:
                control.close()
            except Exception as error:
                if pending is not None:
                    pending.add_note(f"seed control cleanup also failed: {error}")
                else:
                    raise

    def preview_epoch(self, seed: Any, *, request_id: str, user_instruction: str, into: Path) -> Mapping[str, Any]:
        """Create a real local seed export and snapshot without authority transport.

        Args:
            seed: Source-built definitions exposing ``materialize_definitions``.
            request_id: Nonempty initialization identity retained in the marker.
            user_instruction: Verbatim authorization for the fresh seed.
            into: New private output directory for the export and sealed snapshot.

        Returns:
            Seed description, actual parentless local candidate commit, export
            path, snapshot path and snapshot digest. It grants no source admission.

        Raises:
            AuthorityError: Wrong selected binding, existing output or invalid seed.
        """
        from .authority import EXPORT_FORMAT, SecretInputs, _read_export, scan_secrets
        from .control import ensure_private_dir
        from .snapshot import write_snapshot
        from .store import _bytes

        self._epoch_request(seed, request_id, user_instruction)
        directory = Path(into).resolve()
        if directory.exists():
            raise AuthorityError("fresh epoch preview requires a new private output directory")
        ensure_private_dir(directory)
        projection = _Projection(self)
        try:
            projection.bootstrap_empty({})
            with self._seed_writer(projection) as writer:
                seed.materialize_definitions(writer, request_id=request_id, user_instruction=user_instruction)
                with writer.reader() as reader:
                    checked = reader.verify(require_export=False)
                    if not checked.ok:
                        raise AuthorityError("fresh epoch preview integrity refused: " + "; ".join(checked.violations))
                    files = dict(reader.workspace_files(tools_release=seed.release["id"], export_format=EXPORT_FORMAT))
                    head = reader.covered_head(files, export_format=EXPORT_FORMAT)
                    files["HEAD.json"] = _bytes(head)
                    scan_secrets(files, SecretInputs())
                self._replace_files(projection.export, files)
                candidate = self._commit(projection.export, None, "Preview fresh authority seed " + request_id)
                verified, _files, actual = _read_export(projection.export)
                self._check_head(verified)
                if actual != candidate or verified != head:
                    raise AuthorityError("fresh epoch preview commit differs from its staged export")
                projection.commit, projection.head = candidate, verified
                writer._batch_dirty = False
                writer.control.ws._rubiksql_authority_commit = candidate
                with writer.reader() as reader:
                    snapshot = write_snapshot(reader, seed.release["id"], out_dir=directory / "snapshot", tiers={})
                shutil.copytree(projection.export, directory / "export")
                _read_export(directory / "export")
                return {**seed.preview(), "candidate_local_commit": candidate, "export": str(directory / "export"),
                        "snapshot": str(snapshot.path), "snapshot_digest": snapshot.digest}
        finally:
            projection.close()

    def init_epoch(self, seed: Any, *, request_id: str, user_instruction: str,
                   gate: Mapping[str, Any] | Callable[[], Mapping[str, Any]],
                   receipts: Mapping[str, Any] | None = None) -> Any:
        """Initialize a separately selected empty Git epoch with one absent-ref CAS.

        Args:
            seed: Committed fresh seed with an exact ``identity`` mapping and a
                ``materialize(writer, request_id=..., user_instruction=..., gate=...)`` method.
            request_id: The authorized one-time epoch request identity.
            user_instruction: Verbatim authorization retained in the marker.
            gate: Complete validated source admission, or a factory called only
                when the selected ref is absent.
            receipts: Exact inline author/review/invariant inputs for comparing a
                repeated request without rerunning the source-audit factory.

        Returns:
            ExportResult after confirming the new ref, or the verified existing
            result when its marker, seed and admission match this exact request.

        Raises:
            AuthorityError: Wrong selected identity, request, seed, or existing epoch.
            GitAuthorityConflict: The initially absent ref was concurrently created.
            GitPublicationUnconfirmed: Reconcile its candidate before any retry.
        """
        from .authority import ExportResult, SecretInputs

        identity = self._epoch_request(seed, request_id, user_instruction)
        if not isinstance(gate, Mapping) and not callable(gate):
            raise AuthorityError("fresh epoch source admission must be a gate or an audit factory")
        observed = self.remote_commit()
        if observed is not None:
            with self.read(commit=observed) as reader:
                marker, state = reader.marker(), reader.current_state()
                if (marker.get("request_id") != request_id
                        or marker.get("user_instruction") != user_instruction
                        or marker.get("seed") != identity or state is None):
                    raise AuthorityError("existing fresh epoch differs from this exact initialization request")
                recorded = state["source_admission"]["gate"]
                if callable(gate):
                    if not isinstance(receipts, Mapping) or recorded.get("author_session") != receipts.get("author_session"):
                        raise AuthorityError("existing fresh epoch author receipt differs from this request")
                    for key in ("review", "invariants"):
                        receipt = receipts.get(key)
                        if (not isinstance(receipt, Mapping) or not isinstance(receipt.get("report"), Mapping)
                                or recorded.get(key) != {"report": dict(receipt["report"]),
                                                         "report_sha256": receipt.get("report_sha256")}):
                            raise AuthorityError("existing fresh epoch inline receipt differs from this request")
                elif recorded != dict(gate):
                    raise AuthorityError("existing fresh epoch source admission differs from this request")
                revision = reader.head().revision
                return ExportResult(observed, sha256_hex((reader.export_anchor / "HEAD.json").read_bytes()),
                                    revision, revision)
        admitted = gate() if callable(gate) else dict(gate)
        if not isinstance(admitted, Mapping) or not isinstance(admitted.get("tiers"), Mapping):
            raise AuthorityError("fresh epoch source admission tier identity differs")
        projection = _Projection(self)
        try:
            projection.bootstrap_empty(admitted["tiers"])
            with self._seed_writer(projection) as writer:
                seed.materialize(writer, request_id=request_id, user_instruction=user_instruction, gate=admitted)
                return projection.publish(SecretInputs())
        except Exception as error:
            if projection.commit is not None:
                raise GitAuthorityCheckpointError(projection.commit, error) from error
            raise
        finally:
            pending = sys.exception()
            try:
                projection.close()
            except Exception as cleanup_error:
                if projection.commit is not None:
                    cleanup_error = GitAuthorityCheckpointError(projection.commit, cleanup_error)
                if pending is not None:
                    pending.add_note(f"projection cleanup also failed: {cleanup_error}")
                else:
                    raise cleanup_error

    @contextlib.contextmanager
    def read(self, *, commit: str | None = None) -> Iterator[Any]:
        """Yield a verified native reader pinned to a reachable remote commit.

        Args:
            commit: Exact historical SHA, or the remote ref as fetched at entry
                (a ref that moved forward after its capture is followed).

        Yields:
            AuthorityRead whose authority_commit is immutable for this view.

        Raises:
            AuthorityError: Remote identity, export, history or native state differs.
        """
        projection = _Projection(self)
        try:
            projection.bootstrap(commit=commit, read_cache=True, shared_cache=True)
            # Both retained state and cold recovery were checked against this
            # exact committed export before bootstrap completed.
            with projection.reader() as reader:
                yield reader
        finally:
            projection.close()

    @contextlib.contextmanager
    def write(self, *, expected_commit: str | None = None) -> Iterator[Any]:
        """Stage a domain batch, then CAS commit before successful context exit.

        Args:
            expected_commit: Exact commit used to prepare the request.

        Yields:
            Existing Authority domain writer over a disposable private workspace.
            Call writer.commit() before any external execution effect.

        Raises:
            GitAuthorityConflict: A stale expected commit or remote CAS conflict.
            GitPublicationUnconfirmed: Push may have succeeded, but is unconfirmed.
            AuthorityError: Invalid complete authority state.
        """
        from .authority import Authority, _LOCK_STATE, authority_lock
        from .control import Control

        writers = getattr(_WRITERS, "active", {})
        existing = writers.get(self.binding)
        if existing is not None:
            if expected_commit is not None and expected_commit != existing._git_projection.commit:
                raise GitAuthorityConflict(expected_commit, existing._git_projection.commit)
            yield existing
            return
        projection = _Projection(self)
        control = None
        original_commit = None
        try:
            projection.bootstrap(read_cache=True, expected_commit=expected_commit)
            original_commit = projection.commit
            projection._activate_hb()
            projection.read_only = False
            control = Control.open(projection.path, context=projection.context, create=False, _projection=projection)
            # Opening a writer registers HB's declared schemas. Check complete
            # committed state again before exposing a cached writable workspace.
            with projection.reader() as reader:
                checked = reader.verify(export_anchor=projection.export)
                if not checked.ok:
                    raise AuthorityError("Git writer projection refused: " + "; ".join(checked.violations))
            writer = Authority(control, projection.native_binding)
            writer.path = projection.path
            writer._git_projection = projection
            writer.git_binding = self.binding
            writer.control.ws._rubiksql_authority_binding = self.binding
            writer.control.ws._rubiksql_authority_db = str(projection.path)
            writer.control.ws._rubiksql_lab_dir = str(projection.path.parent)
            writer.control.ws._rubiksql_authority_commit = projection.commit
            projection.writer = writer
            with authority_lock(projection.path):
                writer._writer_active = True
                writer._writer_thread = threading.get_ident()
                writer._writer_token = object()
                _WRITERS.active = {**writers, self.binding: writer}
                native_writers = getattr(_LOCK_STATE, "writers", {})
                _LOCK_STATE.writers = {**native_writers, projection.path: writer}
                try:
                    yield writer
                    with writer.reader() as reader:
                        dirty = not projection.clean(reader)
                    if dirty:
                        writer.commit()
                except Exception as error:
                    if projection.commit != original_commit:
                        raise GitAuthorityCheckpointError(str(projection.commit), error) from error
                    raise
                finally:
                    _WRITERS.active = writers
                    _LOCK_STATE.writers = native_writers
                    writer._writer_active = False
                    writer._writer_token = None
                    projection.writer = None
                    writer._close_read_memo()
        finally:
            pending = sys.exception()
            cleanup_error = None
            try:
                try:
                    if control is not None:
                        control.close()
                finally:
                    closing_error = sys.exception()
                    try:
                        projection.close()
                    except Exception as error:
                        if closing_error is None:
                            raise
                        closing_error.add_note(f"projection cleanup also failed: {error}")
            except Exception as error:
                cleanup_error = error
            if cleanup_error is not None:
                if original_commit is not None and projection.commit != original_commit:
                    cleanup_error = GitAuthorityCheckpointError(str(projection.commit), cleanup_error)
                if pending is not None:
                    pending.add_note(f"projection cleanup also failed: {cleanup_error}")
                else:
                    raise cleanup_error

    def project(self, *, context: Any = None, commit: str | None = None) -> Any:
        """Open a read-only HB facade of one verified commit.

        Args:
            context: Optional caller's HB context for the read-only facade. The
                restoration still uses its own private machine state.
            commit: Exact reachable historical commit; remote HEAD by default.

        Returns:
            Control with Git binding, commit and local path on its workspace.
            Close control.ws to release both facade and private projection.

        Raises:
            AuthorityError: Invalid source commit or complete projection.
        """
        from .control import Control

        projection = _Projection(self)
        try:
            projection.bootstrap(commit=commit, read_cache=True)
            projection._activate_hb()
            projection.read_only = True
            control = Control.open(projection.path, context=context or projection.context, create=False, read_only=True)
            ws = control.ws
            ws._rubiksql_authority_binding = self.binding
            ws._rubiksql_authority_commit = projection.commit
            ws._rubiksql_authority_projection = projection
            ws._rubiksql_authority_db = str(projection.path)
            ws._rubiksql_lab_dir = str(projection.path.parent)
            control._projection_owner = projection
            ws.close = control.close
            return control
        except BaseException:
            projection.close()
            raise

    def import_export(self, export: Path, *, expected_commit: str | None, user_instruction: str,
                      live_db: Path | None = None, tool_tiers: Path | Mapping[str, str] | None = None) -> Any:
        """Seed Git from an explicitly approved complete export, preserving live receipts.

        Args:
            export: Exact verified committed format-3 source export.
            expected_commit: Exact existing remote SHA, or None for an absent ref.
            user_instruction: Verbatim authorization for this authority cutover.
            live_db: Original stopped native DB. Production cutover requires it
                to preserve the final self-excluded certifying export event.
            tool_tiers: Exact operational tier policy, or the original DB's
                adjacent policy file at explicit cutover.

        Returns:
            ExportResult only after the exact remote commit is confirmed.

        Raises:
            AuthorityError: Incomplete source, changed UUID, missing instruction,
                stale receipt or missing original production database.
            GitAuthorityConflict: Remote ref differs; no merge or retry is made.
        """
        from .authority import Authority, EXPORT_FORMAT, ExportResult, _read_export, secret_inputs, scan_secrets
        from .store import _bytes

        if not user_instruction.strip():
            raise AuthorityError("Git authority import requires the exact nonempty user instruction")
        head, exported, source_commit = _read_export(export)
        self._check_head(head, source_export=True)
        source_head = head
        files = {**exported, "HEAD.json": (Path(export) / "HEAD.json").read_bytes()}
        if live_db is None and self.binding.remote in B._REMOTES:
            raise AuthorityError("production authority cutover requires the original DB to retain its final export receipt")
        if live_db is not None:
            with Authority.read(Path(live_db), binding=_historical_binding(head)) as reader:
                checked = reader.verify()
                receipts = reader.events("export")
                if (not checked.ok or not receipts or receipts[-1]["payload"].get("export_commit") != source_commit
                        or receipts[-1]["payload"].get("head_sha256") != sha256_hex(files["HEAD.json"])
                        or receipts[-1]["payload"].get("covered_revision") != head["covered_revision"]):
                    raise AuthorityError("original authority is not exactly certified by the cutover export")
                files = dict(reader.workspace_files(tools_release=reader.head().release, export_format=EXPORT_FORMAT))
                head = reader.covered_head(files, export_format=EXPORT_FORMAT)
                files["HEAD.json"] = _bytes(head)
                source_tiers = dict(reader.tool_tiers())
        else:
            source_tiers = json.loads(exported.get("metadata/tool-tiers.json", b"{}"))
        tiers = (dict(tool_tiers) if isinstance(tool_tiers, Mapping) else json.loads(tool_tiers.read_bytes())
                 if isinstance(tool_tiers, Path) else source_tiers)
        if tiers != source_tiers:
            raise AuthorityError("cutover tool tier policy differs from its original authority")
        if (not isinstance(tiers, dict) or any(not isinstance(name, str) or not name
                or tier not in {"hb-executed", "host-verified"} for name, tier in tiers.items())):
            raise AuthorityError("cutover tool tier policy is invalid")
        files["metadata/tool-tiers.json"] = _bytes(tiers)
        # Ancillary policy is digest-covered without adding a seventeenth set.
        mapped = {name: sha256_hex(raw) for name, raw in sorted(files.items()) if name != "HEAD.json"}
        head = {**head, "files": mapped, "file_map_digest": sha256_hex(canon(mapped))}
        files["HEAD.json"] = _bytes(head)
        secrets = secret_inputs()
        scan_secrets({**files, "cutover-instruction": user_instruction.encode("utf-8")}, secrets)
        revision = head["covered_revision"]
        from .types import AuthorityRevision

        covered = AuthorityRevision(revision["event_count"], revision["latest_event_id"])
        with tempfile.TemporaryDirectory(prefix="rubiksql-git-import-", dir=self.cache_dir) as temporary:
            checkout = Path(temporary)
            _git(checkout, "init", "-b", "authority/v0")
            observed = self.remote_commit()
            if observed != expected_commit:
                raise GitAuthorityConflict(expected_commit, observed)
            if expected_commit is not None:
                _git_remote_read(checkout, "fetch", "--no-tags", self.binding.remote, self.binding.ref)
                fetched = _git(checkout, "rev-parse", "FETCH_HEAD").stdout.strip()
                if fetched != expected_commit:
                    raise GitAuthorityConflict(expected_commit, fetched)
                _git(checkout, "checkout", "-B", "authority/v0", expected_commit)
                prior, _prior_files, _prior_commit = _read_export(checkout)
                self._check_head(prior, source_export=True)
                # Import cannot overwrite an evolved remote; ordinary write owns mutations.
                if prior != source_head or _prior_files != exported:
                    raise AuthorityError("authority import cannot replace an existing different committed state")
            self._replace_files(checkout, files)
            candidate = self._commit(checkout, expected_commit, "Adopt complete authority export: " + user_instruction)
            _read_export(checkout)
            self._publish(checkout, expected=expected_commit, candidate=candidate)
            return ExportResult(candidate, sha256_hex(files["HEAD.json"]), covered, covered)


def active_writer(binding: GitAuthorityBinding) -> Any | None:
    """Return only this thread's staged writer for the exact Git identity."""
    writer = getattr(_WRITERS, "active", {}).get(binding)
    return writer if writer is not None and writer._writer_active else None
