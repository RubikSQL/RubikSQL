"""Tell whether the process that wrote a host-count row still runs, without trusting a reusable pid.

A process killed without cleanup (``TerminateProcess`` on Windows, ``SIGKILL`` on POSIX) never runs ``atexit``, so its
published counts stay in the host broker until their heartbeat expires. Readers drop such rows at once by checking the
writer: every row's owner names the writer's pid and its **birth** (the process creation time), and a row is dead when
that pid no longer runs or now belongs to a process born at another time (Windows reuses pids quickly).

Births come from ``GetProcessTimes`` on Windows and the ``starttime`` field of ``/proc/<pid>/stat`` on Linux. Elsewhere
only the pid's existence is checked. Every doubt keeps the row: an unreadable process, an owner written without a
birth (pid reuse then looks alive) or an owner in an older format leaves the row to the heartbeat expiry.
"""
from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import threading
import time

#: A live owner is checked again after this many seconds; a dead one stays dead.
RECHECK_S = 2.0
#: Cached verdicts beyond this many are dropped (owners are per process, so this only bounds a long-lived reader).
CACHE_LIMIT = 4096
_ERROR_INVALID_PARAMETER = 87
_STILL_ACTIVE = 259
_kernel32 = None


def _windows_kernel():
    global _kernel32
    if _kernel32 is None:
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
        kernel.GetProcessTimes.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        _kernel32 = kernel
    return _kernel32


def _windows_birth(pid: int) -> str | None:
    import ctypes
    from ctypes import wintypes

    kernel = _windows_kernel()
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION; never a terminate right
    if not handle:
        error = ctypes.get_last_error()
        if error == _ERROR_INVALID_PARAMETER:
            return None  # no process has this pid
        raise ctypes.WinError(error)  # e.g. access denied: it runs, but its birth is unknown
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        if code.value != _STILL_ACTIVE:
            return None  # exited; another handle keeps its object (a process exiting with code 259 reads as running)
        created, exited, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel_time),
                                      ctypes.byref(user_time)):
            raise ctypes.WinError(ctypes.get_last_error())
        return str((created.dwHighDateTime << 32) | created.dwLowDateTime)
    finally:
        kernel.CloseHandle(handle)


def birth(pid: int) -> str | None:
    """Return a running process's birth identity.

    Args:
        pid: The process id.

    Returns:
        ``None`` when no process with this pid runs (exited, or a zombie on Linux); the creation time as a decimal
        string; or ``""`` when the process runs but this platform cannot read its birth.

    Raises:
        OSError: The process could not be inspected (for example access denied); its state is unknown.
    """
    if os.name == "nt":
        return _windows_birth(pid)
    stat = Path(f"/proc/{pid}/stat")
    if Path("/proc/self/stat").is_file():
        try:
            fields = stat.read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
        except FileNotFoundError:
            return None
        return None if fields[0] in ("Z", "X") else fields[19]
    try:
        os.kill(pid, 0)  # POSIX only: signal 0 checks existence; Windows never reaches this line
    except ProcessLookupError:
        return None
    except PermissionError:
        return ""
    return ""


def owner_tag(pid: int | None = None) -> str:
    """``<pid>-<birth>`` for a count row's owner; the birth is empty when it cannot be read."""
    pid = os.getpid() if pid is None else pid
    try:
        born = birth(pid) or ""
    except OSError:
        born = ""
    return f"{pid}-{born}"


def owner_dead(owner: str) -> bool:
    """Whether a ``<pid>-<birth>-<nonce>`` owner's process is provably gone.

    True only when the pid no longer runs, or runs with another recorded birth (a reused pid). Owners in another
    format, an empty birth (only pid existence is then checked) and processes that cannot be inspected count as alive.
    """
    parts = owner.split("-")
    if len(parts) != 3 or not parts[0].isdigit():
        return False
    recorded = parts[1]
    try:
        current = birth(int(parts[0]))
    except (OSError, ValueError, IndexError):
        return False
    if current is None:
        return True
    return bool(recorded and current and current != recorded)


class OwnerLiveness:
    """Cache :func:`owner_dead` verdicts for a reader that checks the same owners every second.

    Args:
        check: The verdict for one owner; tests pass a fake.
        clock: Monotonic clock.
        recheck: Seconds a live verdict is reused; a dead verdict is final.
    """

    def __init__(self, check: Callable[[str], bool] = owner_dead, *, clock: Callable[[], float] = time.monotonic,
                 recheck: float = RECHECK_S) -> None:
        self._check = check
        self._clock = clock
        self._recheck = recheck
        self._lock = threading.Lock()
        self._checked: dict[str, float | None] = {}  # owner -> time of its last live verdict, None once dead

    def dead(self, owner: str) -> bool:
        """Whether ``owner`` is provably gone (cached; see :func:`owner_dead`)."""
        now = self._clock()
        with self._lock:
            if owner in self._checked:
                last = self._checked[owner]
                if last is None:
                    return True
                if now - last < self._recheck:
                    return False
        gone = self._check(owner)
        with self._lock:
            if len(self._checked) >= CACHE_LIMIT:
                self._checked.clear()
            self._checked[owner] = None if gone else now
        return gone
