"""Stat identities for immutable file and directory verification memos."""
from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import threading

_WINDOWS_READER: Callable[[Path], tuple[int, ...]] | None = None
_WINDOWS_READER_LOCK = threading.Lock()


def _windows_reader() -> Callable[[Path], tuple[int, ...]]:
    """Bind native metadata readers once, including Windows metadata change time."""
    import ctypes
    from ctypes import wintypes

    class FileBasicInfo(ctypes.Structure):
        _fields_ = [
            ("creation", ctypes.c_longlong), ("access", ctypes.c_longlong),
            ("write", ctypes.c_longlong), ("change", ctypes.c_longlong),
            ("attributes", wintypes.DWORD),
        ]

    class FileInformation(ctypes.Structure):
        _fields_ = [
            ("attributes", wintypes.DWORD), ("creation", wintypes.FILETIME),
            ("access", wintypes.FILETIME), ("write", wintypes.FILETIME),
            ("volume", wintypes.DWORD), ("size_high", wintypes.DWORD),
            ("size_low", wintypes.DWORD), ("links", wintypes.DWORD),
            ("index_high", wintypes.DWORD), ("index_low", wintypes.DWORD),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    open_file = kernel.CreateFileW
    open_file.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                          wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    open_file.restype = wintypes.HANDLE
    read_basic = kernel.GetFileInformationByHandleEx
    read_basic.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    read_basic.restype = wintypes.BOOL
    read_file = kernel.GetFileInformationByHandle
    read_file.argtypes = [wintypes.HANDLE, ctypes.POINTER(FileInformation)]
    read_file.restype = wintypes.BOOL
    close_file = kernel.CloseHandle
    close_file.argtypes = [wintypes.HANDLE]
    close_file.restype = wintypes.BOOL
    invalid = ctypes.c_void_p(-1).value

    def read(path: Path) -> tuple[int, ...]:
        # Read attributes only; backup semantics also permits directory handles.
        handle = open_file(str(path), 0x80, 0x7, None, 3, 0x02000000, None)
        if handle == invalid:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            basic, info = FileBasicInfo(), FileInformation()
            if not read_basic(handle, 0, ctypes.byref(basic), ctypes.sizeof(basic)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not read_file(handle, ctypes.byref(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            return basic.change, info.volume, (info.index_high << 32) | info.index_low
        finally:
            close_file(handle)

    return read


def file_identity(path: Path) -> tuple[str | int, ...]:
    """Identify file or directory edits, replacement, permissions and Windows change time.

    Args:
        path: Existing file or directory. Callers that forbid symlinks must check them separately.

    Returns:
        tuple: Canonical path, device, inode, size, nanosecond modification/creation-or-change
        times and mode; Windows also includes native change time, volume and file index.

    Raises:
        OSError: The path or its native metadata cannot be read.
    """
    path = Path(path)
    metadata = path.stat()
    native: tuple[int, ...] = ()
    if os.name == "nt":
        global _WINDOWS_READER
        reader = _WINDOWS_READER
        if reader is None:
            with _WINDOWS_READER_LOCK:
                if _WINDOWS_READER is None:
                    _WINDOWS_READER = _windows_reader()
                reader = _WINDOWS_READER
        native = reader(path)
    # Like the gateway configuration memo, this assumes an attacker cannot forge
    # every OS identity field, including metadata change time and native file id.
    return (str(path.resolve()), metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns, metadata.st_mode, *native)
