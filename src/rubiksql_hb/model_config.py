"""Inspect or update private model credentials while experiments are paused.

Run ``python -m rubiksql_hb.model_config --help`` for the command interface.
Updates preserve the account indexes and experiment checkpoints owned by
``GatewayPool``. Only ``verify --live`` calls the existing bounded Windows
verification tool, using its separate allocation state.
"""

from __future__ import annotations

import argparse
import getpass
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence
import warnings

from .experiment_gateway import COORDINATOR_ENV, GatewayConfigurationError, GatewayPool, GatewayUnavailableError

_KEY_BYTES = 16_384
_SECRET_COMMANDS = {"replace-chat-key", "add-chat-key", "set-embedding-key"}


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        # Subparsers inherit this class, so neither level treats a pasted --key
        # argument as an abbreviation for the private-file option.
        kwargs["allow_abbrev"] = False
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:
        # argparse normally echoes rejected argument values, including an accidentally
        # supplied secret. Only fixed guidance is safe here.
        self.exit(2, "Invalid model configuration command or arguments; use --help. Never pass a key as an argument.\n")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description=__doc__, prog="python -m rubiksql_hb.model_config")
    parser.add_argument("--pool", type=Path, help="Private approved pool configuration path")
    parser.add_argument("--state", type=Path, help="Durable allocation state path")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="Read redacted configuration and account status without allocation")
    verify = commands.add_parser("verify", help="Run the repository's bounded Windows verification only with --live")
    verify.add_argument("--live", action="store_true", help="Explicitly authorize bounded paid probes in this invocation")
    verify.add_argument("--verification-state", type=Path, required=True, help="Separate verification allocation state")
    verify.add_argument("--output", type=Path, required=True, help="Safe verification report path")
    verify.add_argument("--max-accounts", type=int, default=6, help="Maximum approved accounts the verification may inspect (default: 6)")
    select = commands.add_parser("select-profile", help="Select the persisted chat profile for new experiments")
    select.add_argument("--paused", action="store_true", help="Acknowledge that experiments are paused")
    select.add_argument("--profile", choices=("volcengine", "official"), required=True)
    rebind = commands.add_parser("rebind-experiment", help="Explicitly move an existing experiment to the selected profile")
    rebind.add_argument("--paused", action="store_true", help="Acknowledge that experiments are paused")
    rebind.add_argument("--experiment-id", required=True)
    paired = commands.add_parser("set-embedding-pool",
                                 help="Publish the embedding-only key pool every embedding request rotates over (replaces the previous one)")
    paired.add_argument("--paused", action="store_true", help="Acknowledge that experiments are paused")
    paired.add_argument("--key-file", type=Path,
                        help="Read a private JSON list/object, or one key per line; otherwise prompt without echo")
    paired.add_argument("--key-count", type=int, default=4, help="Keys to prompt for without --key-file (default: 4)")
    paired.add_argument("--bucket-id", action="append",
                        help="Explicit nonsecret quota bucket (account) id; one shared id or one ordered id per key")
    paired.add_argument("--requests-per-minute", type=int, default=1800)
    paired.add_argument("--tokens-per-minute", type=int, default=1_200_000)
    paired.add_argument("--chat-requests-per-minute", type=int, help="Optional explicit chat request pacing budget")
    paired.add_argument("--chat-tokens-per-minute", type=int, help="Optional explicit chat token pacing budget")
    scrub = commands.add_parser("scrub-replaced-embedding-keys",
                                help="Rewrite inactive snapshots and the bootstrap file so replaced embedding keys survive nowhere")
    scrub.add_argument("--paused", action="store_true", help="Acknowledge that experiments are paused")
    restored = commands.add_parser("restore-embedding-account", help="Restore a funded or repaired embedding account")
    restored.add_argument("--paused", action="store_true", help="Acknowledge that experiments are paused")
    restored.add_argument("--index", type=int, required=True)
    for name in ("replace-chat-key", "add-chat-key", "retire-chat-account", "restore-chat-account", "set-embedding-key"):
        command = commands.add_parser(name)
        command.add_argument("--paused", action="store_true", help="Acknowledge that experiments have been paused before this update")
        if name != "set-embedding-key":
            command.add_argument("--provider", choices=("volcengine", "dashscope", "deepseek"),
                                 default="volcengine", help="Account pool to maintain (default: volcengine)")
        if name in {"replace-chat-key", "retire-chat-account", "restore-chat-account"}:
            command.add_argument("--index", type=int, required=True, help="Existing stable chat account index")
        if name in _SECRET_COMMANDS:
            command.add_argument("--key-file", type=Path, help="Read the key from a private UTF-8 file; otherwise prompt without echo")
    return parser


def _read_key(path: Path | None) -> str:
    if path is None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("New key (hidden): ")
    else:
        with path.open("rb") as stream:
            data = stream.read(_KEY_BYTES + 1)
        if len(data) > _KEY_BYTES:
            raise ValueError("key input limit")
        key = data.decode("utf-8-sig").strip()
    if not key or len(key.encode("utf-8")) > _KEY_BYTES or any(character.isspace() for character in key):
        raise ValueError("invalid key input")
    return key


def _read_pool_keys(path: Path | None, count: int = 4) -> dict[str, Any]:
    """Read embedding keys privately; return no values through the command's status output.

    A file holds a JSON list, a JSON object with ``keys`` (and optional limits), or one key per line.
    """
    if path is None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            return {"keys": [getpass.getpass(f"Embedding key {index + 1} (hidden): ") for index in range(count)]}
    with path.open("rb") as stream:
        data = stream.read(64 * _KEY_BYTES + 4097)
    if len(data) > 64 * _KEY_BYTES + 4096:
        raise ValueError("pool input limit")
    text = data.decode("utf-8-sig")
    # One key per line only when no line could be JSON structure; anything JSON-shaped must parse as JSON.
    if any(character in text for character in '",[{'):
        body = json.loads(text)
    else:
        body = [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(body, list):
        return {"keys": body}
    if not isinstance(body, dict) or set(body) - {"keys", "bucket_ids", "requests_per_minute", "tokens_per_minute"}:
        raise ValueError("invalid private pool input")
    return body


def _verify(args: argparse.Namespace, env: dict[str, str]) -> int:
    if not args.live or args.max_accounts < 1:
        print("Verification requires --live and a positive --max-accounts; no probes were started.", file=sys.stderr)
        return 2
    try:
        verify = importlib.import_module("scripts.verify_experiment_gateway")
    except ImportError:
        print("The verification tool is unavailable; run this command from the repository checkout.", file=sys.stderr)
        return 2
    names = ("RUBIKSQL_GATEWAY_POOL", "RUBIKSQL_GATEWAY_STATE")
    previous = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            if name in env:
                os.environ[name] = env[name]
            else:
                os.environ.pop(name, None)
        return verify.main(["--live", "--state", str(args.verification_state), "--output", str(args.output),
                            "--max-accounts", str(args.max_accounts)])
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main(argv: Sequence[str] | None = None) -> int:
    """Run a redacted configuration command, or explicitly requested verification.

    Args:
        argv: Command arguments, excluding the executable; defaults to ``sys.argv``.

    Returns:
        Zero on success, two for refused configuration/input, or 130 on interruption.
        Parser errors use argparse's exit status two. Secret values and exception
        details are never written to stdout or stderr.
    """
    args = _parser().parse_args(argv)
    env = dict(os.environ)
    if any(name in env for name in COORDINATOR_ENV) and args.command != "status":
        print("Run model maintenance and live verification on the gateway coordinator host; the remote interface permits status only.", file=sys.stderr)
        return 2
    if args.command not in {"status", "verify"} and not args.paused:
        print("Updates require --paused after experiments have been paused.", file=sys.stderr)
        return 2
    if args.pool is not None:
        env["RUBIKSQL_GATEWAY_POOL"] = str(args.pool)
    if args.state is not None:
        env["RUBIKSQL_GATEWAY_STATE"] = str(args.state)
    try:
        if args.command == "verify":
            return _verify(args, env)
        pool = GatewayPool.from_env(env)
        result = {"operation": args.command}
        if args.command == "set-embedding-pool":
            private = _read_pool_keys(args.key_file, args.key_count)
            bucket_ids = args.bucket_id if args.bucket_id is not None else private.get("bucket_ids")
            if bucket_ids is not None and len(bucket_ids) == 1:
                bucket_ids = bucket_ids * len(private.get("keys", []))
            chat_limits = {name: value for name, value in {
                "requests_per_minute": args.chat_requests_per_minute,
                "tokens_per_minute": args.chat_tokens_per_minute}.items() if value is not None}
            pool.set_embedding_pool(private.get("keys", []), bucket_ids=bucket_ids,
                requests_per_minute=private.get("requests_per_minute", args.requests_per_minute),
                tokens_per_minute=private.get("tokens_per_minute", args.tokens_per_minute),
                chat_limits=chat_limits or None)
        elif args.command == "restore-embedding-account":
            pool.restore_embedding(args.index)
        elif args.command == "scrub-replaced-embedding-keys":
            result["rewritten_files"] = pool.scrub_replaced_embedding_keys()
        elif args.command in _SECRET_COMMANDS:
            key = _read_key(args.key_file)
            if args.command == "replace-chat-key":
                pool.replace_key(args.index, key, provider=args.provider)
            elif args.command == "add-chat-key":
                result["index"] = pool.add_key(key, provider=args.provider)
            else:
                pool.set_embedding_key(key)
        elif args.command == "retire-chat-account":
            pool.retire(args.index, provider=args.provider)
        elif args.command == "restore-chat-account":
            pool.restore(args.index, provider=args.provider)
        elif args.command == "select-profile":
            pool.select_profile(args.profile)
        elif args.command == "rebind-experiment":
            pool.rebind_experiment(args.experiment_id)
        if hasattr(args, "index"):
            result["index"] = args.index
        result["status"] = pool.status()
    except (GatewayConfigurationError, GatewayUnavailableError):
        print("Model configuration refused; check the approved pool, durable state and account index. Resume only after a successful update.", file=sys.stderr)
        return 2
    except (OSError, TypeError, ValueError, EOFError, getpass.GetPassWarning):
        print("Cannot read or update model configuration. Use a private UTF-8 key file (one key, one key per line, "
              "or valid JSON) or a terminal with hidden input.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Model configuration interrupted.", file=sys.stderr)
        return 130
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
