# RubikSQL

RubikSQL answers questions about a relational database with SQL. It registers the database schema in a HeavenBase knowledge workspace, retrieves schema and business context, and gives an agent tools to inspect the database and run read-only SQL. The Python API can generate one SQL answer or compare several candidate answers with a judge.

- **Bring your own database.** Register a SQLAlchemy database URL; the demo uses SQLite.
- **Inspect before answering.** The agent can read table and column metadata, rules, and saved question–SQL examples.
- **Use Python or MCP.** Call `nl2sql` from Python or expose the pinned tools to an MCP client.

This repository retains the NL2SQL runtime and its verified execution snapshot from RubikSQL-HB. Benchmark data, leaderboard scripts, experiment records, and self-evolution entry points are excluded. [Scope and remaining dependencies](docs/SCOPE.md) explains what remains.

## Demo

[The demo](examples/demo/README.md) creates a small shop database, registers it, reads its schema, and runs a join through RubikSQL. It uses a fresh local workspace for every run. From a checkout with `uv` and `DASHSCOPE_API_KEY` configured:

```bash
uv sync --frozen
uv run python examples/demo/demo.py
```

Output from a local run (warnings and the generated workspace path omitted):

```text
Registered 'shop': 2 tables, 5 columns, 2 enum values.
Schema: customers, orders
SQL rows: [{'name': 'Ada', 'total': 100}, {'name': 'Lin', 'total': 25}]
```

Registration makes DashScope embedding requests. The default demo runs a fixed, read-only SQL query and does not call a chat model. With an existing RubikSQL chat gateway pool, add `--ask --provider qwen` to generate SQL for the same database with a model. That optional path makes paid chat requests and requires the [chat configuration](#model-configuration) below.

## Quick start

You need Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/) on `PATH`. This is currently a source-checkout install; the repository includes its pinned HeavenBase wheel.

1. Clone this repository and install the locked environment:

   ```bash
   git clone https://github.com/RubikSQL/RubikSQL.git
   cd RubikSQL
   uv sync --frozen
   ```

2. Check the runtime without a key or database:

   ```bash
   uv run python tests/smoke_runtime.py
   ```

   It verifies the adopted snapshot, imports both Python answer functions, loads 26 pinned answering tools, and initializes the MCP toolkit. It makes no model request.

3. Set `DASHSCOPE_API_KEY` in your shell or secret manager, then run the [demo](examples/demo/README.md):

   ```bash
   uv run python examples/demo/demo.py
   ```

   On PowerShell 7, `$env:DASHSCOPE_API_KEY = Read-Host "DashScope API key" -MaskInput` accepts the key without echoing it. The demo writes only under ignored `runtime/`; it prints the workspace path so you can inspect or delete it after the run.

## Use it from Python

The main call is `nl2sql(question, db_id, provider="qwen")`. Register a database first, then bootstrap the verified snapshot before importing the answering modules. A complete in-process example is in [Python usage](docs/PYTHON_USAGE.md). For example, after the registration and bootstrap in that guide:

```python
from rubiksql_hb.nl2sql import nl2sql

answer = nl2sql("How much did each customer spend?", "shop", provider="qwen")
print(answer["sql"], answer["rows"])
```

`rubiksql_hb.vote.nl2sql_vote(question, db_id)` runs several candidates and a judge. Both paths require chat credentials and may incur model charges. They return the executed SQL and rows along with diagnostic data.

## Use it through MCP

Start the stdio server from the checkout:

```bash
uv run python scripts/mcp_server.py
```

Configure your MCP client to run that command in the repository directory. The server loads the verified snapshot and exposes database registration, schema inspection, knowledge retrieval, and SQL tools. Call `register_database` with a database ID and SQLAlchemy URL before asking the client to inspect or query that database. Set `RUBIKSQL_DATA_DIR` if you want its registry and knowledge workspace outside the default `runtime/data/` location. The MCP client supplies its own chat model; Python `nl2sql` uses RubikSQL's chat configuration.

## Model configuration

| Setting | Used for | When needed |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | DashScope `text-embedding-v4` for schema and knowledge vectors | Database registration and vector retrieval, unless a managed embedding key is configured |
| Private chat gateway file (`RUBIKSQL_GATEWAY_POOL` overrides its default path) | RubikSQL chat account selection | Python `nl2sql`, `nl2sql_vote`, or demo `--ask` |
| `RUBIKSQL_DATA_DIR` | Local database registry, knowledge workspace, and cache | Optional; defaults to the checkout's `runtime/data/` in the MCP launcher |

The single `DASHSCOPE_API_KEY` is sufficient for the default demo's embedding calls. It does **not** configure RubikSQL's Python chat route. The current chat runtime requires its existing gateway pool format; an official profile needs both Qwen and DeepSeek account pools even when a run selects Qwen. `uv run python -m rubiksql_hb.model_config status` reports configured accounts without printing keys. Keep credential files outside the repository. The demo and smoke test do not require another database server.

## Documentation and release status

- [Demo details](examples/demo/README.md)
- [Python usage](docs/PYTHON_USAGE.md)
- [Source scope and release review](docs/SCOPE.md)

The adopted snapshot contains prompt text and Python tool source. This repository has no selected RubikSQL license yet; the license and redistribution of the included patched HeavenBase wheel must be reviewed before public release. The [RubikSQL paper](https://arxiv.org/abs/2508.17590) describes the original research project.
