# RubikSQL

RubikSQL is an NL2SQL runtime built on HeavenBase. It gives an agent MCP tools for database schema inspection, read-only SQL execution, business-rule retrieval, and retrieval of prior question–SQL examples. The Python runtime can produce a single answer or run several candidates and a judge.

This code retains the existing NL2SQL architecture and an adopted, immutable execution snapshot from RubikSQL-HB. The self-evolution operators, BIRD submission/evaluation scripts, benchmark inputs, dashboards, and recorded experiment store are not part of this repository slice. See [scope and remaining dependencies](docs/SCOPE.md).

## Install from this checkout

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required for the pinned environment. The repository includes the exact HeavenBase wheel used by this runtime.

```powershell
uv sync --frozen
uv run python tests/smoke_runtime.py
```

The smoke test uses no model key or external database. It verifies the adopted snapshot, imports `nl2sql` and `nl2sql_vote`, loads the 26 pinned answering tools, and starts the MCP toolkit locally. Its generated files go under the ignored `runtime/` directory.

To serve MCP over stdio from the checkout:

```powershell
uv run python scripts/mcp_server.py
```

The launcher verifies `releases/adopted-snapshot.tar.gz`, extracts it into `runtime/snapshots/`, installs its pinned tool bundle, and starts the existing server. An MCP client can then call `register_database` with a database ID and SQLAlchemy URL before asking schema or SQL questions. Building the knowledge workspace and calling a chat model require separately configured embedding and chat credentials; this checkout does not contain any credentials or prepared user database.

The Python entry points are `rubiksql_hb.nl2sql.nl2sql(question, db_id)` and `rubiksql_hb.vote.nl2sql_vote(question, db_id)`. They expect a registered database and `RUBIKSQL_PROMPT_SNAPSHOT` pointing at the verified extracted snapshot. The MCP launcher performs that bootstrap for its own process. See [Python usage notes](docs/PYTHON_USAGE.md) for an example of the in-process bootstrap.

## Source and release status

The adopted snapshot contains Python tool source and prompt content by design. Its historical authority binding is retained to verify that snapshot; the original private authority service is not required for the local smoke test. This branch does not include the prompt-authoring store, so it does not provide a way to publish a new authority release.

No RubikSQL license has been selected in this repository yet. Licensing and the redistribution of the vendored HeavenBase wheel must be reviewed before a public release.

The original research description and citation are available in the existing [RubikSQL project](https://github.com/RubikSQL/RubikSQL) history and [paper](https://arxiv.org/abs/2508.17590).
