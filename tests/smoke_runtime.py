"""Offline check for the public NL2SQL checkout and adopted tool snapshot."""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from mcp_server import ensure_snapshot

os.environ["RUBIKSQL_DATA_DIR"] = str(ROOT / "runtime" / "smoke-data")
snapshot_path = ensure_snapshot()

from rubiksql_hb.prompts.snapshot import snapshot_env, verify_snapshot
from rubiksql_hb.tool_bundle import install_bundle_importer

snapshot = verify_snapshot(snapshot_path)
os.environ.update(snapshot_env(snapshot, bundle=True))
install_bundle_importer(snapshot)

from rubiksql_hb.prompts import current
from rubiksql_hb.tool_declarations import project_tool_functions
from rubiksql_hb.nl2sql import nl2sql
from rubiksql_hb.vote import nl2sql_vote

release = current()
tools = {function.__name__: function for function in project_tool_functions()}
assert release.id == snapshot.release
assert len(tools) == 26
assert tools["render_template"]("SELECT {value}", {"value": 42}) == "SELECT 42"
assert callable(nl2sql) and callable(nl2sql_vote)

from rubiksql_hb.server import _mcp_toolkit

toolkit = _mcp_toolkit()
assert len(toolkit.tools) >= 26
print(f"release={release.id} tools={len(tools)} mcp_tools={len(toolkit.tools)}")
