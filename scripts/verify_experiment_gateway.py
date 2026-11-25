"""Bounded, explicitly requested Windows gateway verification; never an evaluation."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path
import secrets
import sys
import tempfile

import httpx

from rubiksql_hb.experiment_gateway import (
    DASHSCOPE_BASE_URL, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, GATEWAY_BASE_URL,
    GatewayConfigurationError, GatewayPool,
)
from rubiksql_hb import gateway_chat, llm_runtime

_DEFAULT_MAX_ACCOUNTS = 6
_GET_WALL_SECONDS = 30
_CHAT_WALL_SECONDS = 65
_TOKEN_FIELDS = frozenset({"prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens",
                           "reasoning_tokens", "cached_tokens"})
_OPERATIONS = [("hb", "qwen", "qwen3.7-plus"), ("hb", "qwen", "qwen3.7-max"),
               ("hb", "deepseek", DEEPSEEK_MODEL), ("generator", "qwen", "qwen3.7-plus"),
               ("generator", "deepseek", DEEPSEEK_MODEL)]
_PROFILE_POOLS = {"volcengine": ("volcengine",), "official": ("dashscope", "deepseek")}
_BASE_URLS = {"volcengine": GATEWAY_BASE_URL, "dashscope": DASHSCOPE_BASE_URL,
              "deepseek": DEEPSEEK_BASE_URL}


def _pool_name(profile: str, provider: str) -> str:
    """Resolve a model provider to the active profile's credential pool."""
    if provider not in {"qwen", "deepseek"} or profile not in _PROFILE_POOLS:
        raise GatewayConfigurationError("Verification requires a supported selected provider profile.")
    return "volcengine" if profile == "volcengine" else ("dashscope" if provider == "qwen" else "deepseek")


def _selected_lease(pool: GatewayPool, provider: str, identity: str, profile: str):
    """Refuse a mismatched profile or endpoint before any probe request."""
    if pool.profile != profile:
        raise GatewayConfigurationError("Verification profile changed before a probe started.")
    lease = pool.for_provider(provider).lease(identity)
    if lease.base_url.rstrip("/") != _BASE_URLS[_pool_name(profile, provider)].rstrip("/"):
        raise GatewayConfigurationError("Verification provider endpoint does not match the selected profile.")
    return lease


def _tokens(usage: dict) -> dict:
    """Retain only known scalar counters, never arbitrary provider-supplied names or text."""
    return {key: usage[key] for key in _TOKEN_FIELDS if type(usage.get(key)) is int and usage[key] >= 0}


async def _runtime_probe(provider: str, model: str, messages: list[dict]) -> dict:
    llm_runtime._MAX_RETRIES = 0
    llm_runtime._RATE_LIMIT_RETRIES = 0
    llm_runtime._CONN_RETRIES = 0
    llm_runtime._CHAT_DEADLINE_S = 45.0
    runtime = llm_runtime.build_llm(provider, model, role="candidate")
    args = {"max_tokens": 32, "timeout": 20.0, "max_retries": 0}
    if provider == "qwen":
        args["extra_body"] = {"enable_thinking": False}
    runtime.spec = runtime.spec.with_args(args)
    turn = await asyncio.wait_for(llm_runtime.achat_turn(runtime, messages=messages, system="Be brief.", tools=[]), timeout=60)
    return turn.usage


def _operation(kind: str, identity: str, provider: str, model: str | None, profile: str, output: str) -> None:
    """Child-process probe. Its only output is a safe result file; raw SDK output is discarded."""
    descriptor = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
    finally:
        os.close(descriptor)
    result = {"ok": False}
    try:
        os.environ["RUBIKSQL_GATEWAY_EXPERIMENT_ID"] = identity
        pool = GatewayPool.from_env()
        lease = _selected_lease(pool, provider, identity, profile)
        result["start_index"] = lease.index
        if kind == "models":
            # Read status/headers only: there is no reason to download or print a model catalog.
            with httpx.Client(follow_redirects=False, timeout=20.0) as client:
                with client.stream("GET", lease.base_url + "/models", headers={"Authorization": "Bearer " + lease.secret}) as response:
                    result.update(index=lease.index, http_status=response.status_code, ok=response.is_success)
        else:
            messages = [{"role": "user", "content": "Reply with OK."}]
            if kind == "hb":
                usage = asyncio.run(_runtime_probe(provider, model, messages))
            else:
                body = {"model": model, "messages": messages, "max_tokens": 32}
                body.update({"thinking": {"type": "disabled"}} if provider == "deepseek" else {"enable_thinking": False})
                usage = gateway_chat.complete(body, timeout=20).get("usage") or {}
            result.update(ok=True, end_index=_selected_lease(pool, provider, identity, profile).index,
                          tokens=_tokens(usage))
    except Exception as error:
        result.update(ok=False, error_type=type(error).__name__)
        status = getattr(error, "status_code", getattr(error, "code", None))
        if type(status) is int and 100 <= status <= 599:
            result["http_status"] = status
    Path(output).write_text(json.dumps(result) + "\n", encoding="utf-8")


def _bounded_operation(kind: str, identity: str, provider: str, model: str | None, profile: str) -> dict:
    """Enforce a process wall limit for both async SDK and synchronous socket operations."""
    limit = _GET_WALL_SECONDS if kind == "models" else _CHAT_WALL_SECONDS
    with tempfile.TemporaryDirectory(prefix="rubiksql-gateway-probe-") as temporary:
        result_path = Path(temporary) / "result.json"
        process = multiprocessing.get_context("spawn").Process(
            target=_operation, args=(kind, identity, provider, model, profile, str(result_path)),
        )
        process.start()
        process.join(limit)
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
            result = {"ok": False, "error_type": "ProbeWallTimeout", "checkpoint": "operation stopped at wall limit; allocation state retained"}
        elif not result_path.is_file():
            result = {"ok": False, "error_type": "ProbeWorkerFailed", "checkpoint": "allocation state retained"}
        else:
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = {"ok": False, "error_type": "ProbeResultUnavailable", "checkpoint": "allocation state retained"}
        process.close()
    return {**result, "wall_limit_seconds": limit}


def _overlaps(path: Path, protected: set[Path]) -> bool:
    """Check resolved paths and existing hard links before writing a report or private allocation state."""
    return path.resolve() in protected or (path.exists() and any(other.exists() and path.samefile(other) for other in protected))


def _allocation_paths(path: Path) -> set[Path]:
    """Reserve SQLite's database and sidecar names together before any writes."""
    return {Path(str(path) + suffix).resolve() for suffix in ("", "-wal", "-shm", "-journal")}


def _selected_accounts(status: dict) -> dict[str, list[dict]]:
    """Return eligible accounts for each pool used by the selected profile."""
    profile = status["profile"]
    return {
        name: [account for account in status["providers"][name]["accounts"] if account["status"] == "available"]
        for name in _PROFILE_POOLS[profile]
    }


@contextmanager
def _verification_pool(source: GatewayPool, state: Path, status: dict, eligible: dict[str, list[dict]]):
    """Copy only currently eligible credentials into a private, independent pool.

    The temporary directory inherits the configured credential directory's Windows
    ACL; POSIX modes are tightened explicitly. Source allocation is read-only.
    Environment overrides are scoped so the unified management CLI can call us.
    """
    try:
        configuration = json.loads(source.configuration_path.read_text(encoding="utf-8-sig"))
        if source.profile == "volcengine":
            snapshot_config = {
                "base_url": configuration["base_url"], "profile": "volcengine",
                "keys": [configuration["keys"][account["index"]] for account in eligible["volcengine"]],
            }
        else:
            snapshot_config = {
                "base_url": configuration["base_url"], "profile": "official",
                # The legacy field remains required by the file schema. No usable
                # production Volc credential belongs in an official verification.
                "keys": ["verification-inert-volc-" + secrets.token_hex(16)],
                "official": {
                    name: {"keys": [configuration["official"][name]["keys"][account["index"]]
                                    for account in eligible[name]]}
                    for name in _PROFILE_POOLS["official"]
                },
            }
        if source.status() != status:
            raise GatewayConfigurationError("Gateway eligibility changed before verification; retry after updates stop.")
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        raise GatewayConfigurationError("Cannot snapshot the current approved verification accounts.") from None
    with tempfile.TemporaryDirectory(prefix="gateway-verification-", dir=source.configuration_path.parent) as temporary:
        directory = Path(temporary)
        directory.chmod(0o700)
        snapshot = directory / "accounts.json"
        with os.fdopen(os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="utf-8") as stream:
            json.dump(snapshot_config, stream)
        snapshot.chmod(0o600)
        variables = {"RUBIKSQL_GATEWAY_POOL": str(snapshot), "RUBIKSQL_GATEWAY_STATE": str(state)}
        previous = {name: os.environ.get(name) for name in variables}
        os.environ.update(variables)
        try:
            pool = GatewayPool.from_env()
            if pool.profile != source.profile:
                raise GatewayConfigurationError("Verification snapshot did not retain the selected profile.")
            yield pool
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _source_indices(result: dict, accounts: list[dict]) -> dict:
    """Label every subset index separately from its original source account."""
    mapped = dict(result)
    for field in ("index", "start_index", "end_index"):
        index = mapped.pop(field, None)
        if type(index) is int and 0 <= index < len(accounts):
            mapped["probe_" + field] = index
            mapped["source_" + field] = accounts[index]["index"]
            mapped["source_" + field.removesuffix("index") + "credential_revision"] = accounts[index]["credential_revision"]
    return mapped


def main(argv: list[str] | None = None) -> int:
    """Run explicitly authorized bounded probes against a current private snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize the bounded model probes in this invocation")
    parser.add_argument("--state", type=Path, required=True, help="Dedicated verification allocation state, never the experiment state")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-accounts", type=int, default=_DEFAULT_MAX_ACCOUNTS,
                        help="Maximum eligible accounts allowed for this invocation (default: 6)")
    args = parser.parse_args(argv)
    if not args.live or sys.platform != "win32":
        parser.error("Live verification requires --live and Windows")
    source = GatewayPool.from_env()
    protected = {source.pool_path.resolve(), source.configuration_path.resolve()} | _allocation_paths(source.state_path)
    verification = _allocation_paths(args.state)
    managed = source.configuration_directory.resolve()
    if any(path.resolve() == managed or managed in path.resolve().parents for path in verification | {args.output}):
        parser.error("Verification files must stay outside the managed credential revision directory")
    if any(_overlaps(path, protected) for path in verification):
        parser.error("Use dedicated verification state to preserve the experiment allocation cursor")
    if _overlaps(args.output, protected | verification):
        parser.error("The report must not overwrite credentials or allocation state")
    if args.output.exists() or any(path.exists() for path in verification):
        parser.error("Use new verification state and report paths; previous verification evidence is preserved")
    status = source.status()
    profile = status["profile"]
    eligible = _selected_accounts(status)
    count = sum(len(accounts) for accounts in eligible.values())
    if args.max_accounts < 1 or any(not accounts for accounts in eligible.values()) or count > args.max_accounts:
        parser.error("Verification requires an eligible account in each selected pool and must stay within --max-accounts")
    report = {"schema": "rubiksql.gateway-verification/3", "cost": None,
              "scope": "model catalog access and five tiny chat operations; no evaluation or knowledge build",
              "credential_coverage": (
                  "Only selected provider accounts marked available are copied. "
                  "GET /models may be public and does not authenticate each account. "
                  "Chat checks the current verification lease, including credential rotation; "
                  "it does not authenticate every source account."
              ),
              "profile": profile,
              "source_configuration_revision": status["configuration_revision"],
              "source_accounts_by_provider": {name: status["providers"][name]["accounts"] for name in eligible},
              "max_accounts": args.max_accounts,
              "max_http_model_requests": count, "logical_chat_operations": len(_OPERATIONS),
              "max_http_chat_requests": sum(len(eligible[_pool_name(profile, provider)])
                                             for _, provider, _ in _OPERATIONS),
              "credential_attempts_per_chat": {name: len(accounts) for name, accounts in eligible.items()},
              "ordinary_chat_retries": 0,
              "production_allocation_used": False,
              "accounts": [], "chat": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with _verification_pool(source, args.state, status, eligible) as pool:
        report["probe_accounts"] = [
            {"provider_pool": name, "probe_index": ordinal, "source_index": account["index"],
             "source_credential_revision": account["credential_revision"]}
            for name, accounts in eligible.items() for ordinal, account in enumerate(accounts)
        ]
        # Reserve the report without overwriting even if another process won the path race.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
        for name, accounts in eligible.items():
            provider = "deepseek" if name == "deepseek" else "qwen"
            for ordinal in range(len(accounts)):
                identity = f"gateway-verification-{name}-account-{ordinal}"
                report["accounts"].append({
                    "operation": "model_catalog_access", "endpoint": "GET /models", "provider_pool": name,
                    **_source_indices(_bounded_operation("models", identity, provider, None, profile), accounts),
                })
                checkpoint()
        coverage = {}
        for name, accounts in eligible.items():
            observed = sorted({item["source_index"] for item in report["accounts"]
                               if item["provider_pool"] == name and "source_index" in item})
            coverage[name] = {"observed_source_indices": observed,
                              "complete": observed == [account["index"] for account in accounts]}
        report["account_coverage"] = {"by_provider": coverage,
                                      "complete": all(item["complete"] for item in coverage.values())}
        for transport, provider, model in _OPERATIONS:
            name = _pool_name(profile, provider)
            report["chat"].append({
                "transport": transport, "provider_pool": name, "model": model, "cost": None,
                **_source_indices(_bounded_operation(
                    transport, f"gateway-verification-{name}-chat", provider, model, profile,
                ), eligible[name]),
            })
            checkpoint()
        probe_status = pool.status()
        report["probe_allocation"] = {
            "profile": probe_status["profile"],
            "providers": {name: probe_status["providers"][name] for name in eligible},
        }
        report["ok"] = report["account_coverage"]["complete"] and all(item["ok"] for item in report["chat"]) and all(item["ok"] for item in report["accounts"])
        checkpoint()
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
