"""Keep CPython's ``platform`` module off WMI in every RubikSQL process on Windows.

On Windows, CPython 3.12 answers ``platform.uname()``, ``platform.platform()``, ``platform.machine()``
and ``platform.processor()`` on first use through its private ``_wmi`` extension. ``_wmi.exec_query``
waits about one second for COM and 100 ms for the WMI connection, then returns while its helper thread
still holds a pointer into the caller's stack frame. CPython 3.13 copies that data into the thread;
3.12 does not. When WMI answers after the calling thread has exited, the helper reads freed stack
memory and the whole process dies with an access violation inside ``_wmi.pyd``.

Answering can reach that path whenever many threads make their first OpenAI SDK request at the same
moment: the SDK's ``platform_headers`` cache is still cold, every embedding call runs on its own
short-lived deadline thread, and each one asks ``platform``. On 2026-10-23 a judge-only child lost
seven MCP servers and four evaluators to access violations inside ``_wmi.pyd`` within 23 seconds
of its first rule-injection burst.

Replacing ``platform._wmi_query`` with a function that raises ``OSError`` sends every lookup down the
module's own non-WMI fallback (``sys.getwindowsversion``, the registry, ``ver`` and the ``PROCESSOR_*``
environment). The package ``__init__`` applies it before anything else is imported, so no thread of
a RubikSQL process can start a WMI query.
"""
from __future__ import annotations

import sys
from typing import NoReturn

__all__ = ["disable_wmi_queries"]


def _refuse_wmi_query(*_keys: str) -> NoReturn:
    """Stand in for ``platform._wmi_query``: always the "WMI unavailable" answer platform handles."""
    raise OSError("RubikSQL processes never query WMI; platform uses its non-WMI fallback")


def disable_wmi_queries() -> None:
    """Make CPython's ``platform`` module answer without WMI for the rest of this process.

    On Windows the module-level ``platform._wmi_query`` is replaced; its callers in the standard
    library already catch ``OSError`` and fall back to ``sys.getwindowsversion``, the registry,
    the ``ver`` command and the ``PROCESSOR_*`` environment. Elsewhere, and on interpreters whose
    ``platform`` has no WMI path, nothing changes. Calling it again is harmless.

    Args:
        None.

    Returns:
        None: This function does not return a value.
    """
    if sys.platform != "win32":
        return
    import platform

    if hasattr(platform, "_wmi_query"):
        platform._wmi_query = _refuse_wmi_query
