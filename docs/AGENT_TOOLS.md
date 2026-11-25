# Connect RubikSQL to CLI agents

RubikSQL already serves its tools over local stdio MCP. A CLI agent that supports MCP can start the server and call `register_database`, `get_db_info`, `get_rules`, `get_examples`, and `run_sql`. This is an explicit database-tool integration: RubikSQL does not capture chat history or inject context automatically as QuoteMem does.

## Prepare one checkout

In the RubikSQL checkout:

```bash
uv sync --frozen
uv run python scripts/check_mcp.py
```

The second command starts a separate server process, performs the MCP handshake, lists the tools, and calls `render_template`. It needs no database or model key. A successful result includes `mcp_tools=42 render=SELECT 42`.

For the client configurations below, replace `C:/path/RubikSQL` with the absolute path to this checkout. On macOS or Linux, use `.venv/bin/python` in place of `.venv/Scripts/python.exe`. The MCP server command is the virtual environment's Python followed by the checkout's `scripts/mcp_server.py`; it works from another current directory and does not require a globally installed RubikSQL command.

## One shared database workspace

All clients that launch this checkout use `runtime/data/` by default. To share a different local workspace, set the **same absolute `RUBIKSQL_DATA_DIR`** for each client's MCP server process. The database registry, reflected schema, rules, and saved examples then live in that one directory. Each client starts its own MCP process; this is shared on-disk state, not a single long-running service. Keep the directory accessible only to the intended local user, especially if a registered database URL contains credentials.

The MCP client's model supplies the chat reasoning. `DASHSCOPE_API_KEY` is still needed when `register_database` embeds schema text, unless the server has a managed embedding key. The Python `nl2sql` function's separate chat gateway pool is **not** needed merely to use MCP tools from an agent. Give the database account read-only permissions when practical; RubikSQL's SQL tool also applies its own read-only policy.

After adding the server in a client, ask it to call `register_database` with your database ID and SQLAlchemy URL, then `get_db_info` for that ID. For a demo, run `examples/demo/demo.py` first and point another MCP process at the printed `rubiksql-data` directory. Check that process with:

```bash
uv run python scripts/check_mcp.py --data-dir /absolute/path/to/rubiksql-data --db-id shop
```

The last command reads the schema from another process without making a model request.

## DeepSeek Harness

DeepSeek Harness uses its `@deepseek-ai/dsh-mcp-client` plugin to connect stdio MCP servers. In the profile you use (for example `web`), make sure a compatible version of that client package is installed. Check `dsh --version`; if the client is absent from that profile, install a version compatible with your DSH release using `dsh plugin --profile web add @deepseek-ai/dsh-mcp-client@<compatible-version>`. Then append an `insert` row to `$DSH_HOME/profiles/<profile>/cordis.patch.yml`. Replace the paths and keep the `id` and `serverName` unique:

```yaml
- insert:
    - id: mcp-rubiksql
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: rubiksql
        transport: stdio
        command: 'C:/path/RubikSQL/.venv/Scripts/python.exe'
        args: ['C:/path/RubikSQL/scripts/mcp_server.py']
        cwd: 'C:/path/RubikSQL'
        env:
          RUBIKSQL_DATA_DIR: 'C:/path/rubiksql-data'
          DASHSCOPE_API_KEY: !!js process.env.DASHSCOPE_API_KEY
        startupTimeoutMs: 120000
        toolCallTimeoutMs: 180000
```

Start DSH with `DASHSCOPE_API_KEY` in its environment if you will register a database. Restart the profile after editing the patch. The tools appear with names such as `mcp__rubiksql__get_db_info`. A DSH installation may use another profile name or supply the MCP client already; check its current profile and package version before adding a second instance. [DSH's MCP client reference](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/mcp/mcp-client/README.md) documents the row and timeout fields.

## Claude Code

In a PowerShell terminal, replace the paths and run:

```powershell
claude mcp add --transport stdio rubiksql -- "C:/path/RubikSQL/.venv/Scripts/python.exe" "C:/path/RubikSQL/scripts/mcp_server.py"
claude mcp get rubiksql
```

This uses Claude Code's default local scope. Start Claude Code with `DASHSCOPE_API_KEY` set if you intend to register a database; set `RUBIKSQL_DATA_DIR` in the same environment to select a shared workspace. Use `/mcp` inside Claude Code to check the connection. See [Claude Code's MCP guide](https://code.claude.com/docs/en/mcp) for other scopes and server management.

## Codex CLI

From a terminal with the desired environment variables:

```powershell
codex mcp add rubiksql -- "C:/path/RubikSQL/.venv/Scripts/python.exe" "C:/path/RubikSQL/scripts/mcp_server.py"
codex mcp list
```

Codex stores the MCP command in its user configuration. Set `RUBIKSQL_DATA_DIR` and, for registration, `DASHSCOPE_API_KEY` where Codex is launched; avoid putting a raw key in the command line or repository file. Use `/mcp` in a Codex session to inspect the tools. See [Codex MCP documentation](https://learn.chatgpt.com/docs/extend/mcp?surface=cli).

## OpenCode

Add a local server to your `opencode.jsonc`, replacing both absolute paths:

```jsonc
{
  "mcp": {
    "servers": {
      "rubiksql": {
        "type": "local",
        "command": [
          "C:/path/RubikSQL/.venv/Scripts/python.exe",
          "C:/path/RubikSQL/scripts/mcp_server.py"
        ],
        "environment": {
          "RUBIKSQL_DATA_DIR": "C:/path/rubiksql-data",
          "DASHSCOPE_API_KEY": "{env:DASHSCOPE_API_KEY}"
        },
        "timeout": { "startup": 120000, "catalog": 120000 }
      }
    }
  }
}
```

`{env:DASHSCOPE_API_KEY}` is OpenCode's environment substitution, not a literal key. Run `opencode mcp list` to inspect the connection. See [OpenCode's MCP guide](https://opencode.ai/docs/mcp-servers/).

## What was verified

On Windows, an independent MCP SDK client launched the checkout server, completed `initialize`, discovered 42 tools, and called `render_template`. It also read a database registered by a separate demo process from the same `RUBIKSQL_DATA_DIR`; two MCP clients read that workspace concurrently. DeepSeek Harness, Claude Code, and OpenCode executables were not installed on the verification machine, so their client-specific configuration examples follow the linked upstream documentation rather than a local end-to-end run. The Codex CLI command syntax was checked against the installed CLI help. No model-generated answer was tested through these clients.
