"""Check the checkout's stdio MCP server with an independent client process."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


ROOT = Path(__file__).resolve().parents[1]


async def check(data_dir: Path | None, db_id: str | None) -> None:
    environment = os.environ.copy()
    if data_dir is not None:
        environment["RUBIKSQL_DATA_DIR"] = str(data_dir.resolve())
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "scripts" / "mcp_server.py")],
        env=environment,
    )
    async with stdio_client(server) as (reader, writer):
        async with ClientSession(reader, writer, read_timeout_seconds=180) as session:
            await session.initialize()
            listed = await session.list_tools()
            names = {tool.name for tool in listed.tools}
            required = {"register_database", "get_db_info", "run_sql", "render_template"}
            if not required <= names:
                raise RuntimeError(f"missing MCP tools: {sorted(required - names)}")
            rendered = await session.call_tool(
                "render_template",
                {"template": "SELECT {value}", "params": {"value": 42}},
            )
            text = " ".join(getattr(part, "text", "") for part in rendered.content)
            if rendered.is_error or "SELECT 42" not in text:
                raise RuntimeError(f"MCP tool call failed: {text[:200]}")
            print(f"mcp_tools={len(names)} render=SELECT 42")
            if db_id:
                schema = await session.call_tool("get_db_info", {"db_id": db_id})
                summary = " ".join(getattr(part, "text", "") for part in schema.content)
                if schema.is_error or "No schema found" in summary or '"error"' in summary:
                    raise RuntimeError(f"MCP database lookup failed: {summary[:200]}")
                print(f"database={db_id} schema_available=True")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Optional shared RubikSQL workspace directory")
    parser.add_argument("--db-id", help="Also check a database registered in that workspace")
    args = parser.parse_args()
    if args.db_id and args.data_dir is None:
        parser.error("--db-id requires --data-dir")
    asyncio.run(check(args.data_dir, args.db_id))


if __name__ == "__main__":
    main()
