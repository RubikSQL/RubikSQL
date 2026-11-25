# RubikSQL demo

`demo.py` creates a small SQLite shop database with two customers and three orders. It registers that database with RubikSQL, reads the reflected schema, and runs a read-only join through a pinned RubikSQL tool. Each run has a new workspace under the checkout's ignored `runtime/demo/` directory; the script prints its path so you can inspect or remove it later. The verified execution snapshot is cached under `runtime/snapshots/`.

## Run it

From the repository root, with Python 3.11–3.13, `uv`, and `DASHSCOPE_API_KEY` set in the environment:

```bash
uv sync --frozen
uv run python examples/demo/demo.py
```

Registration embeds the demo's schema with DashScope `text-embedding-v4`, so this default demo makes paid embedding requests. It does not call a chat model. A successful run ends with two table names and two SQL result rows; exact row formatting depends on the installed HeavenBase version.

To let the model generate SQL too, first configure the existing private chat gateway file (set `RUBIKSQL_GATEWAY_POOL` if it is outside the default location), then run:

```bash
uv run python examples/demo/demo.py --ask --provider qwen
```

The gateway pool is separate from the single `DASHSCOPE_API_KEY` used for embeddings. `--ask` sends a real chat request and may take longer. `--model` overrides the provider's default model. See [Python usage](../../docs/PYTHON_USAGE.md) for the in-process API and [scope](../../docs/SCOPE.md) for the retained snapshot and release limitations.
