# No convenience re-exports: the public surface is the submodules themselves and every
# consumer imports explicitly (`from rubiksql_hb.schema import ...`). A partial name
# list here once masqueraded as the entity roster and drifted three entities behind.
import os as _os
import sys as _sys
from .platform_guard import disable_wmi_queries as _disable_wmi_queries

# First, before any import that could start a thread: CPython 3.12's WMI lookups can crash
# a process whose calling thread exits early (platform_guard.py).
_disable_wmi_queries()

from .tool_bundle import install_unpinned_import_guard as _install_unpinned_import_guard  # noqa: E402

_install_unpinned_import_guard()

if _os.environ.get("RUBIKSQL_TOOL_BUNDLE"):
    if _os.environ["RUBIKSQL_TOOL_BUNDLE"] != "1":
        raise ImportError("RUBIKSQL_TOOL_BUNDLE must be 1 when set")
    _snapshot_path = _os.environ.get("RUBIKSQL_PROMPT_SNAPSHOT")
    if not _snapshot_path:
        raise ImportError("bundle execution requires a verified RUBIKSQL_PROMPT_SNAPSHOT")
    from pathlib import Path as _Path
    from .prompts.binding import select_binding_from_argv as _select_binding_from_argv
    from .prompts.snapshot import verify_snapshot as _verify_snapshot
    from .tool_bundle import install_bundle_importer as _install_bundle_importer

    _select_binding_from_argv(_sys.argv[1:])
    _install_bundle_importer(_verify_snapshot(_Path(_snapshot_path)))
