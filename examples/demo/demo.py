"""Run a small RubikSQL database demo in a fresh local workspace."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def create_shop_database(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
                amount INTEGER NOT NULL
            );
            INSERT INTO customers VALUES (1, 'Ada'), (2, 'Lin');
            INSERT INTO orders VALUES (101, 1, 40), (102, 1, 60), (103, 2, 25);
            """
        )


def run_demo(database: Path, *, ask: bool, provider: str, model: str | None) -> None:
    from scripts.mcp_server import ensure_snapshot
    from rubiksql_hb.prompts.snapshot import snapshot_env, verify_snapshot
    from rubiksql_hb.tool_bundle import install_bundle_importer

    snapshot = verify_snapshot(ensure_snapshot())
    os.environ.update(snapshot_env(snapshot, bundle=True))
    install_bundle_importer(snapshot)

    from rubiksql_hb.tool_declarations import project_tool_functions

    tools = {function.__name__: function for function in project_tool_functions()}
    database_url = f"sqlite:///{database.as_posix()}"
    print(tools["register_database"]("shop", database_url), flush=True)

    schema = tools["get_db_info"]("shop")
    if "error" in schema:
        raise RuntimeError(schema["error"])
    tables = sorted(table["table"] for table in schema["tables"])
    assert tables == ["customers", "orders"], tables
    print("Schema:", ", ".join(tables), flush=True)

    result = tools["run_sql"](
        "shop",
        "SELECT c.name, SUM(o.amount) AS total "
        "FROM customers AS c JOIN orders AS o USING (customer_id) "
        "GROUP BY c.name ORDER BY total DESC",
    )
    if "error" in result:
        raise RuntimeError(result["error"])
    print("SQL rows:", result["rows"], flush=True)
    assert result["row_count"] == 2, result

    if ask:
        from rubiksql_hb.nl2sql import nl2sql

        answer = nl2sql(
            "How much did each customer spend in total?",
            "shop",
            provider=provider,
            model=model,
        )
        print("Generated SQL:", answer["sql"], flush=True)
        print("Answer rows:", answer["rows"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ask", action="store_true", help="Also ask the model to generate SQL")
    parser.add_argument("--provider", default="qwen", help="Chat provider for --ask (default: qwen)")
    parser.add_argument("--model", help="Optional model override for --ask")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        run_demo(Path(os.environ["RUBIKSQL_DEMO_DATABASE"]), ask=args.ask, provider=args.provider, model=args.model)
        return

    if not os.environ.get("DASHSCOPE_API_KEY") and not os.environ.get("RUBIKSQL_GATEWAY_POOL"):
        parser.error("set DASHSCOPE_API_KEY for schema embeddings, or configure RUBIKSQL_GATEWAY_POOL")

    workdir = ROOT / "runtime" / "demo" / uuid4().hex
    workdir.mkdir(parents=True)
    database = workdir / "shop.sqlite"
    create_shop_database(database)
    environment = os.environ.copy()
    environment["RUBIKSQL_DATA_DIR"] = str(workdir / "rubiksql-data")
    environment["RUBIKSQL_DEMO_DATABASE"] = str(database)
    command = [sys.executable, str(Path(__file__).resolve()), "--_worker"]
    if args.ask:
        command += ["--ask", "--provider", args.provider]
        if args.model:
            command += ["--model", args.model]
    try:
        subprocess.run(command, env=environment, check=True)
    finally:
        print(f"Demo files: {workdir}", flush=True)


if __name__ == "__main__":
    main()
