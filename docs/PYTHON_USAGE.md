# Python usage from a source checkout

This example keeps the existing runtime protocol: verify the adopted snapshot, install its tool importer, register a database, then call `nl2sql` or `nl2sql_vote`. It requires valid embedding and chat credentials configured for the selected provider profile. Registration may make embedding calls; answering makes chat calls.

```python
import os
from pathlib import Path

from scripts.mcp_server import ensure_snapshot
from rubiksql_hb.prompts.snapshot import snapshot_env, verify_snapshot
from rubiksql_hb.tool_bundle import install_bundle_importer

data_dir = Path("runtime/data").resolve()
os.environ["RUBIKSQL_DATA_DIR"] = str(data_dir)

snapshot = verify_snapshot(ensure_snapshot())
os.environ.update(snapshot_env(snapshot, bundle=True))
install_bundle_importer(snapshot)

from rubiksql_hb.tool_declarations import project_tool_functions

tools = {fn.__name__: fn for fn in project_tool_functions()}
print(tools["register_database"]("shop", "sqlite:///C:/data/shop.sqlite"))

from rubiksql_hb.nl2sql import nl2sql

answer = nl2sql("How many orders were placed?", "shop", provider="qwen")
print(answer["sql"], answer["rows"])
```

Run the script from the repository root after `uv sync --frozen`. Replace the SQLite URL with your own SQLAlchemy database URL. `DASHSCOPE_API_KEY` supplies schema embeddings, but Python `nl2sql` also needs the existing private chat gateway pool; one DashScope environment key alone does not configure that route. To inspect credential configuration without exposing keys, use `uv run python -m rubiksql_hb.model_config status`; `--help` lists the supported setup commands. Do not commit API keys or local workspaces. The [demo](../examples/demo/README.md) creates its own SQLite database and exercises registration, schema inspection, and read-only SQL.

`nl2sql_vote(question, db_id, ...)` in `rubiksql_hb.vote` uses multiple candidate agents and a judge. Both calls require a registered database ID. The adopted snapshot fixes their tool and prompt versions; changing a tool's checkout source file alone does not change the pinned execution version.

The included offline smoke test verifies imports and tool startup only. It does not guarantee that every database dialect, credential profile, or model endpoint works on another machine.
