"""Share one private gateway checkpoint through an authenticated loopback tunnel."""

from __future__ import annotations

import argparse
from contextlib import closing
import ctypes
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import socket
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .experiment_gateway import (
    CredentialFailure, DASHSCOPE_BASE_URL, GATEWAY_BASE_URL, GROUP_ENV,
    GatewayConfigurationError, GatewayLease, GatewayPool, GatewayUnavailableError,
    ModelProvider, Profile, ProviderPool, current_group_id, group_context,
)

COORDINATOR_ENV = "RUBIKSQL_GATEWAY_COORDINATOR"
TOKEN_ENV = "RUBIKSQL_GATEWAY_COORDINATOR_TOKEN"
_LIMIT = 256 * 1024
_READ_ONLY = {"status", "experiment_binding", "binding_revision"}
_OPERATIONS = _READ_ONLY | {"database_groups", "lease", "embedding_lease", "rotate",
                          "block_embedding", "pace_chat", "pace_embedding"}


def _check_private_token(path: Path) -> None:
    """Refuse a token file readable by another ordinary account."""
    if os.name != "nt":
        info = path.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise GatewayConfigurationError("The coordinator token file must be private to its owner.")
        return
    script = (
        "$ErrorActionPreference='Stop';"
        "$allowed=@([Security.Principal.WindowsIdentity]::GetCurrent().User.Value,'S-1-5-18','S-1-5-32-544');"
        "$acl=Get-Acl -LiteralPath $env:RUBIKSQL_COORDINATOR_TOKEN_CHECK_PATH;"
        "$owner=([Security.Principal.NTAccount]$acl.Owner).Translate([Security.Principal.SecurityIdentifier]).Value;"
        "if($allowed -notcontains $owner){exit 1};"
        "foreach($rule in $acl.Access){"
        "if($rule.AccessControlType -eq 'Allow' -and "
        "$allowed -notcontains $rule.IdentityReference.Translate([Security.Principal.SecurityIdentifier]).Value){exit 1}};exit 0"
    )
    environment = {**os.environ, "RUBIKSQL_COORDINATOR_TOKEN_CHECK_PATH": str(path)}
    executable = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    try:
        checked = subprocess.run([str(executable), "-NoProfile", "-NonInteractive", "-Command", script],
                                 env=environment, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        raise GatewayConfigurationError("Cannot verify the private coordinator token ACL.") from None
    if checked.returncode:
        raise GatewayConfigurationError("The coordinator token file must have a private ACL.")


def _read_token(path: Path | str) -> str:
    try:
        file = Path(path)
        if file.is_symlink() or not file.is_file() or file.stat().st_size > 1024:
            raise ValueError
        _check_private_token(file)
        token = file.read_text(encoding="utf-8").strip()
        if not 32 <= len(token) <= 256 or not token.isascii() or any(character.isspace() for character in token):
            raise ValueError
        return token
    except (OSError, ValueError, TypeError):
        raise GatewayConfigurationError("Cannot read the private coordinator token file.") from None


def initialize_token(path: Path | str) -> None:
    """Create one private channel token without exposing it or replacing a file.

    Args:
        path: New token filename beneath an existing private directory.

    Raises:
        GatewayConfigurationError: The file exists or cannot be made private.
    """
    file = Path(path)
    created = False
    try:
        if not file.parent.is_dir() or file.parent.is_symlink():
            raise ValueError
        if os.name != "nt":
            _check_private_token(file.parent)
        descriptor = os.open(file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        created = True
        if os.name == "nt":
            script = (
                "$ErrorActionPreference='Stop';"
                "$sid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value;"
                # Windows Python's private temporary directories can materialize an explicit OWNER RIGHTS ACE.
                # Remove it on this newly created file; the strict verifier below still accepts only user/system/admin.
                "& icacls.exe $env:RUBIKSQL_COORDINATOR_TOKEN_CHECK_PATH '/inheritance:r' '/remove:g' '*S-1-3-4' '/grant:r' "
                "('*'+$sid+':F') '*S-1-5-18:F' | Out-Null;exit $LASTEXITCODE"
            )
            executable = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
            checked = subprocess.run([str(executable), "-NoProfile", "-NonInteractive", "-Command", script],
                env={**os.environ, "RUBIKSQL_COORDINATOR_TOKEN_CHECK_PATH": str(file)}, capture_output=True, timeout=10, check=False)
            if checked.returncode:
                raise ValueError
        _check_private_token(file)
        file.write_text(secrets.token_urlsafe(48) + "\n", encoding="ascii")
    except (OSError, ValueError, subprocess.SubprocessError, GatewayConfigurationError):
        if created:
            file.unlink(missing_ok=True)
        raise GatewayConfigurationError("Cannot create a new private coordinator token file; existing files are preserved.") from None


def _identity(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise GatewayConfigurationError("A stable coordinator experiment identity is required.")
    return value


def _lease_fields(lease: GatewayLease) -> dict[str, Any]:
    return {"experiment_id": lease.experiment_id, "group_id": lease.group_id,
            "index": lease.index, "credential_revision": lease.credential_revision}


def _encode_lease(lease: GatewayLease) -> dict[str, Any]:
    return {**_lease_fields(lease), "secret": lease.secret, "base_url": lease.base_url}


class GatewayCoordinator:
    """Resolve authenticated requests against one existing private checkpoint.

    Args:
        pool_path: Private Volcengine account configuration.
        state_path: The existing allocation and pacing SQLite database.
        token_file: Private bearer token file; no token is logged.

    Raises:
        GatewayConfigurationError: A required private input is unavailable.
    """

    def __init__(self, pool_path: Path | str, state_path: Path | str, token_file: Path | str) -> None:
        self.pool_path, self.state_path = Path(pool_path), Path(state_path)
        self._token = _read_token(token_file)
        if not self.state_path.is_file():
            raise GatewayConfigurationError("The coordinator requires an existing allocation checkpoint.")
        self._pool_lock = threading.Lock()
        self._current_pool: GatewayPool | None = None
        self._pool()

    def _pool(self) -> GatewayPool:
        with self._pool_lock:
            # Remote clients share request cursors until maintenance publishes a new snapshot.
            pool = self._current_pool
            metadata = None if pool is None else pool._configuration_metadata()
            if pool is None or (metadata is not None and metadata[0] != pool._configuration_revision):
                pool = GatewayPool(self.pool_path, self.state_path)
                if pool.profile != "volcengine" or not pool.paired:
                    raise GatewayConfigurationError("The coordinator requires the approved grouped Volcengine pool.")
                self._current_pool = pool
            return pool

    def authorized(self, authorization: str) -> bool:
        """Check the exact Bearer header without timing-sensitive equality."""
        try:
            return hmac.compare_digest(authorization.encode("utf-8"), ("Bearer " + self._token).encode("ascii"))
        except UnicodeError:
            return False

    def _claim(self, receipt: Any, identity: str) -> Mapping[str, Any]:
        from .prompts.paid_launch import verify_remote_claim

        if not isinstance(receipt, dict):
            raise GatewayConfigurationError("A verified committed launch claim is required.")
        reservation = verify_remote_claim(receipt)
        pin, output = reservation.get("pin"), reservation.get("out_dir")
        if not isinstance(pin, str) or not pin or not isinstance(output, str) or not output:
            raise GatewayConfigurationError("The coordinator experiment differs from its committed reservation.")
        expected = "rubiksql:" + pin + ":" + output
        if reservation.get("diagnostic") is True:
            namespace = reservation.get("namespace")
            if not isinstance(namespace, str) or not namespace or namespace != namespace.strip():
                raise GatewayConfigurationError("The committed diagnostic namespace is invalid.")
            expected = "rubiksql:diagnostic:" + namespace
        if identity != expected:
            raise GatewayConfigurationError("The coordinator experiment differs from its committed reservation.")
        databases = reservation.get("inputs", {}).get("db_fingerprints")
        if (not isinstance(databases, Mapping) or not 1 <= len(databases) <= 11
                or any(not isinstance(database, str) or not database or database != database.strip() for database in databases)):
            raise GatewayConfigurationError("The committed database scope is invalid.")
        if receipt.get("inputs", {}).get("db_fingerprints") != dict(databases):
            raise GatewayConfigurationError("The receipt database scope differs from its committed reservation.")
        return databases

    def _group(self, pool: GatewayPool, identity: str, group_id: Any, scope: Mapping[str, Any]) -> str:
        if not isinstance(group_id, str):
            raise GatewayConfigurationError("The coordinator requires an allocated database group.")
        with closing(sqlite3.connect(self.state_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)) as connection:
            connection.execute("PRAGMA query_only=ON")
            row = connection.execute("SELECT groups_json FROM gateway_database_groups WHERE experiment_id=?", (identity,)).fetchone()
        groups = [] if row is None else pool._read_database_groups(row[0])
        if (set(database for group in groups for database in group["database_ids"]) != set(scope)
                or not any(group["group_id"] == group_id for group in groups)):
            raise GatewayConfigurationError("The provider group differs from the committed database scope.")
        return group_id

    def dispatch(self, request: Mapping[str, Any]) -> Any:
        """Handle one validated JSON RPC without making any provider request.

        Args:
            request: Operation, pinned revision, arguments and optional claim.

        Returns:
            JSON-compatible status, allocation, private lease or pacing delay.

        Raises:
            GatewayConfigurationError: Invalid request, claim, scope or revision.
            GatewayUnavailableError: No chat account can serve the group, or its
                embedding account is blocked.
        """
        if (not isinstance(request, dict) or set(request) - {"operation", "revision", "arguments", "receipt"}
                or request.get("operation") not in _OPERATIONS or not isinstance(request.get("arguments"), dict)):
            raise GatewayConfigurationError("The coordinator request is invalid.")
        operation, arguments = request["operation"], request["arguments"]
        pool = self._pool()
        configuration_revision = pool.configuration_revision
        revision = request.get("revision")
        if (revision is None and operation != "status") or (revision is not None and
                (type(revision) is not int or revision != configuration_revision)):
            raise GatewayConfigurationError("The coordinator configuration revision changed; reconstruct the client.")
        if operation == "status":
            if arguments:
                raise GatewayConfigurationError("The coordinator request is invalid.")
            return {**pool.status(), "chat_pacing_enabled": pool.chat_pacing_enabled}
        identity = _identity(arguments.get("experiment_id"))
        if operation in _READ_ONLY:
            if set(arguments) != {"experiment_id"}:
                raise GatewayConfigurationError("The coordinator request is invalid.")
            return getattr(pool, operation)(identity)
        scope = self._claim(request.get("receipt"), identity)
        if operation == "database_groups":
            databases = arguments.get("database_ids")
            if (set(arguments) != {"experiment_id", "database_ids"} or not isinstance(databases, list)
                    or any(not isinstance(database, str) for database in databases)
                    or len(databases) != len(scope) or set(databases) != set(scope)):
                raise GatewayConfigurationError("The requested database scope differs from the committed reservation.")
            return pool.database_groups(identity, databases)
        group_id = self._group(pool, identity, arguments.get("group_id"), scope)
        allowed = {"experiment_id", "group_id"}
        if operation == "lease" and "advance" in arguments:
            allowed.add("advance")
        elif operation in {"rotate", "block_embedding"}:
            allowed |= {"index", "credential_revision", "reason"}
        elif operation in {"pace_chat", "pace_embedding"}:
            allowed |= {"index", "credential_revision", "model", "tokens"}
        if set(arguments) != allowed:
            raise GatewayConfigurationError("The coordinator request is invalid.")
        with group_context(group_id):
            if operation == "lease":
                advance = arguments.get("advance", True)
                if type(advance) is not bool:
                    raise GatewayConfigurationError("The coordinator lease cursor selection is invalid.")
                return _encode_lease(pool.for_provider("qwen").lease(identity, advance=advance))
            if operation == "embedding_lease":
                lease = pool.embedding_lease(identity)
                if lease is None:
                    raise GatewayConfigurationError("The grouped embedding account is unavailable.")
                return _encode_lease(lease)
            if operation == "rotate":
                return _encode_lease(pool.for_provider("qwen").rotate(identity, arguments["index"], arguments["reason"],
                    failed_revision=arguments["credential_revision"], group_id=group_id))
            if operation == "block_embedding":
                failed = GatewayLease(arguments["index"], "", DASHSCOPE_BASE_URL, identity,
                                      arguments["credential_revision"], group_id)
                return pool.block_embedding(failed, arguments["reason"])
            embedding = operation == "pace_embedding"
            index, credential_revision = arguments["index"], arguments["credential_revision"]
            if type(index) is not int or type(credential_revision) is not int:
                raise GatewayConfigurationError("The pacing account credential is invalid.")
            if embedding:
                # Embedding requests rotate over the whole pool: pace any currently available account.
                account_matches = any((index, credential_revision, "available")
                                      == (account["index"], account["credential_revision"], account["status"])
                                      for account in pool.status()["paired"].get("embedding_accounts", ()))
            else:
                # An in-flight request may retain the account its group just left.
                # Pace that exact credential; a rejection then rotates to the current lease.
                account_matches = any((index, credential_revision) == (account["index"], account["credential_revision"])
                                      for account in pool.status()["accounts"])
            if not account_matches:
                raise GatewayConfigurationError("The pacing lease has no matching account credential.")
            model, tokens = arguments["model"], arguments["tokens"]
            if not isinstance(model, str) or not model or type(tokens) is not int or tokens < 1:
                raise GatewayConfigurationError("The coordinator pacing reservation is invalid.")
            limits = pool._config["embedding"] if embedding else pool._config.get("chat_limits")
            if limits is None:
                return 0.0
            bucket = ("dashscope-embedding:" + limits["bucket_ids"][index] if embedding
                      else f"volcengine:account-{index}")
            from .pool_rates import reserve

            return reserve(self.state_path, bucket=bucket, model=model, tokens=tokens,
                requests_per_minute=limits.get("requests_per_minute"), tokens_per_minute=limits.get("tokens_per_minute"))


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _reply(self, status: int, value: Any) -> None:
        body = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        self.connection.settimeout(10)
        if self.path != "/rpc" or not self.server.coordinator.authorized(self.headers.get("Authorization", "")):
            self._reply(403, {"error": "coordinator_request_refused"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= _LIMIT or self.headers.get("Transfer-Encoding"):
                raise ValueError
            request = json.loads(self.rfile.read(length))
            result = self.server.coordinator.dispatch(request)
        except GatewayUnavailableError:
            self._reply(409, {"error": "approved_gateway_accounts_unavailable"})
        except Exception:
            self._reply(400, {"error": "coordinator_request_refused"})
        else:
            self._reply(200, {"result": result})


class GatewayCoordinatorServer(ThreadingHTTPServer):
    """Serve bounded JSON requests only on a literal loopback listener.

    Args:
        address: ``127.0.0.1`` or ``::1`` and the selected local port.
        coordinator: Owner of the existing private gateway checkpoint.
    """

    daemon_threads = True

    def __init__(self, address: tuple[str, int], coordinator: GatewayCoordinator) -> None:
        if address[0] not in {"127.0.0.1", "::1"}:
            raise GatewayConfigurationError("The gateway coordinator must bind literal loopback.")
        self.coordinator = coordinator
        if address[0] == "::1":
            self.address_family = socket.AF_INET6
        super().__init__(address, _Handler)

    def handle_error(self, request: Any, client_address: Any) -> None:
        # Broken tunnel sessions must not print request context or private data.
        pass


class RemoteProviderPool(ProviderPool):
    """Keep the provider model contract while routing allocations to its owner."""

    def lease(self, experiment_id: str | None = None, *, group_id: str | None = None,
              advance: bool = True) -> GatewayLease:
        return self.owner._lease("lease", experiment_id, group_id, advance=advance)

    def rotate(self, experiment_id: str, failed_index: int, reason: CredentialFailure,
               *, failed_revision: int | None = None, group_id: str | None = None) -> GatewayLease:
        return self.owner.rotate(experiment_id, failed_index, reason, failed_revision=failed_revision, group_id=group_id)

    def before_request(self, lease: GatewayLease, model: str, payload: Any) -> None:
        self.owner._pace_chat(lease, self.provider, model, payload)


class RemoteGatewayPool:
    """Use one coordinator exclusively; never open local credentials or checkpoints.

    Args:
        environ: Complete caller environment, including inherited paid receipt,
            loopback coordinator URL and the path of its private token file.

    Raises:
        GatewayConfigurationError: Endpoint, token or bootstrap is unavailable.
    """

    def __init__(self, environ: Mapping[str, str]) -> None:
        self.env = dict(environ)
        try:
            endpoint = urlsplit(self.env.get(COORDINATOR_ENV, ""))
            if (endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "::1"}
                    or endpoint.username is not None or endpoint.password is not None
                    or endpoint.path not in {"", "/"} or endpoint.query or endpoint.fragment
                    or endpoint.port is None):
                raise ValueError
            self.host, self.port = endpoint.hostname, endpoint.port
            token_file = Path(self.env[TOKEN_ENV])
            self._token = _read_token(token_file)
            self.token_file = token_file.resolve(strict=True)
        except (KeyError, ValueError, TypeError, OSError):
            raise GatewayConfigurationError("An explicit loopback coordinator and private token file are required.") from None
        self.experiment_id = self.env.get("RUBIKSQL_GATEWAY_EXPERIMENT_ID")
        self.state_path = Path("gateway-coordinator")
        self._configuration_revision: int | None = None
        status = self._rpc("status")
        if (not isinstance(status, dict) or status.get("profile") != "volcengine"
                or status.get("paired", {}).get("configured") is not True
                or type(status.get("configuration_revision")) is not int
                or type(status.get("chat_pacing_enabled")) is not bool):
            raise GatewayConfigurationError("The coordinator bootstrap is invalid.")
        self._configuration_revision = status["configuration_revision"]
        self._chat_pacing_enabled = status["chat_pacing_enabled"]

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> RemoteGatewayPool:
        """Load the explicit coordinator endpoint and private token file."""
        return cls(os.environ if environ is None else environ)

    @property
    def profile(self) -> Profile:
        return "volcengine"

    @property
    def paired(self) -> bool:
        return True

    @property
    def chat_pacing_enabled(self) -> bool:
        return self._chat_pacing_enabled

    def _rpc(self, operation: str, **arguments: Any) -> Any:
        receipt = None
        if operation not in _READ_ONLY:
            from .prompts.paid_launch import verify_claim

            receipt = verify_claim(environment=self.env)
            if not receipt:
                raise GatewayConfigurationError("The coordinator requires a claimed Git-authority launch.")
        body = json.dumps({"operation": operation, "revision": self._configuration_revision,
                           "arguments": arguments, "receipt": receipt}, separators=(",", ":")).encode("utf-8")
        if len(body) > _LIMIT:
            raise GatewayConfigurationError("The coordinator request exceeds its bounded JSON limit.")
        connection = http.client.HTTPConnection(self.host, self.port, timeout=10)
        try:
            connection.request("POST", "/rpc", body=body, headers={"Authorization": "Bearer " + self._token,
                                                                   "Content-Type": "application/json"})
            response = connection.getresponse()
            encoded = response.read(_LIMIT + 1)
            if len(encoded) > _LIMIT:
                raise ValueError
            result = json.loads(encoded)
            if (response.status == 409 and isinstance(result, dict)
                    and result.get("error") == "approved_gateway_accounts_unavailable"):
                raise GatewayUnavailableError(arguments.get("experiment_id", self.experiment_id or ""), self.state_path)
            if response.status != 200 or not isinstance(result, dict) or set(result) != {"result"}:
                raise ValueError
            return result["result"]
        except (OSError, ValueError, http.client.HTTPException):
            raise GatewayConfigurationError("The coordinator request was refused or interrupted; inspect its checkpoint before retrying.") from None
        finally:
            connection.close()

    def _effect(self, operation: str, **arguments: Any) -> Any:
        """Verify the caller claim and send one effect; interrupted calls never retry."""
        if operation not in _OPERATIONS - _READ_ONLY:
            raise GatewayConfigurationError("Unknown coordinator effect.")
        return self._rpc(operation, **arguments)

    def status(self) -> dict[str, Any]:
        """Read nonsecret status and refuse a changed configuration revision."""
        return self._rpc("status")

    def for_provider(self, provider: ModelProvider) -> RemoteProviderPool:
        """Return the Volcengine model-family adapter."""
        if provider not in {"qwen", "deepseek"}:
            raise GatewayConfigurationError("Unknown chat model provider.")
        return RemoteProviderPool(self, "volcengine", GATEWAY_BASE_URL)

    def lease_provider(
        self, provider: ModelProvider, experiment_id: str | None = None, *,
        group_id: str | None = None, advance: bool = True,
    ) -> tuple[RemoteProviderPool, GatewayLease]:
        """Select a route and obtain its coordinator-admitted lease.

        Arguments and failures follow :meth:`GatewayPool.lease_provider`.
        """
        route = self.for_provider(provider)
        return route, route.lease(experiment_id, group_id=group_id, advance=advance)

    def transport_identity(self, provider: ModelProvider) -> str:
        route = self.for_provider(provider)
        return f"volcengine:{provider}:{route.base_url}"

    def experiment_binding(self, experiment_id: str) -> tuple[Profile, int] | None:
        binding = self._rpc("experiment_binding", experiment_id=_identity(experiment_id))
        return tuple(binding) if binding is not None else None

    def binding_revision(self, experiment_id: str) -> int:
        return self._rpc("binding_revision", experiment_id=_identity(experiment_id))

    def database_groups(self, experiment_id: str, database_ids: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
        groups = self._effect("database_groups", experiment_id=_identity(experiment_id), database_ids=list(database_ids))
        return tuple({**group, "chat_indexes": tuple(group["chat_indexes"]),
                      "database_ids": tuple(group["database_ids"])} for group in groups)

    def _group_id(self, group_id: str | None = None) -> str:
        return _identity(group_id or current_group_id() or self.env.get(GROUP_ENV))

    def _decode_lease(self, value: Any, *, embedding: bool = False) -> GatewayLease:
        if (not isinstance(value, dict) or set(value) != {"index", "secret", "base_url", "experiment_id", "credential_revision", "group_id"}
                or type(value["index"]) is not int or value["index"] < 0
                or type(value["credential_revision"]) is not int or value["credential_revision"] < 1
                or not isinstance(value["secret"], str) or not value["secret"]
                or value["base_url"] != (DASHSCOPE_BASE_URL if embedding else GATEWAY_BASE_URL)
                or not isinstance(value["experiment_id"], str) or not isinstance(value["group_id"], str)):
            raise GatewayConfigurationError("The coordinator returned an invalid approved lease.")
        if not embedding:
            from .chat_keys import DeniedChatKey, denylist, refused
            if refused(value["secret"], f"coordinator chat index {value['index']}", denylist()):
                raise DeniedChatKey("The coordinator returned a retired chat credential; no request sent.")
        return GatewayLease(value["index"], value["secret"], value["base_url"], value["experiment_id"],
                            value["credential_revision"], value["group_id"])

    def _lease(self, operation: str, experiment_id: str | None, group_id: str | None = None,
               *, advance: bool = True) -> GatewayLease:
        arguments: dict[str, Any] = {"experiment_id": _identity(experiment_id or self.experiment_id),
                                     "group_id": self._group_id(group_id)}
        if operation == "lease":
            arguments["advance"] = advance
        return self._decode_lease(self._effect(operation, **arguments), embedding=operation == "embedding_lease")

    def lease(self, experiment_id: str | None = None) -> GatewayLease:
        return self._lease("lease", experiment_id)

    def embedding_lease(self, experiment_id: str | None = None) -> GatewayLease:
        return self._lease("embedding_lease", experiment_id)

    def rotate(self, experiment_id: str, failed_index: int, reason: CredentialFailure,
               *, failed_revision: int | None = None, group_id: str | None = None) -> GatewayLease:
        return self._decode_lease(self._effect("rotate", experiment_id=_identity(experiment_id), group_id=self._group_id(group_id),
            index=failed_index, credential_revision=failed_revision, reason=reason))

    def block_embedding(self, lease: GatewayLease, reason: CredentialFailure) -> None:
        self._effect("block_embedding", **_lease_fields(lease), reason=reason)

    def _pace(self, operation: str, lease: GatewayLease, model: str, payload: Any) -> None:
        from .pool_rates import token_reservation

        delay = self._effect(operation, **_lease_fields(lease), model=model, tokens=token_reservation(payload))
        if not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay < 0:
            raise GatewayConfigurationError("The coordinator returned an invalid reservation delay.")
        if delay:
            time.sleep(delay)

    def _pace_chat(self, lease: GatewayLease, provider: str, model: str, payload: Any) -> None:
        if provider != "volcengine":
            raise GatewayConfigurationError("Unknown coordinator provider.")
        self._pace("pace_chat", lease, model, payload)

    def pace_embedding(self, lease: GatewayLease, model: str, payload: Any) -> None:
        self._pace("pace_embedding", lease, model, payload)


def main() -> int:
    """Run an elevated Windows or ordinary private-host loopback coordinator."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--pool", type=Path, required=True)
    serve.add_argument("--state", type=Path, required=True)
    serve.add_argument("--token-file", type=Path, required=True)
    serve.add_argument("--authority-binding", type=Path, help="explicit immutable authority binding file")
    serve.add_argument("--host", choices=("127.0.0.1", "::1"), default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8890)
    token = commands.add_parser("init-token")
    token.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        if os.name == "nt" and not ctypes.windll.shell32.IsUserAnAdmin():
            raise GatewayConfigurationError("Windows gateway coordination requires an elevated administrator token.")
        if args.command == "init-token":
            initialize_token(args.token_file)
            print("Private coordinator token file created.")
            return 0
        if args.authority_binding is not None:
            from .prompts.binding import select_binding
            select_binding(args.authority_binding)
        coordinator = GatewayCoordinator(args.pool, args.state, args.token_file)
        with GatewayCoordinatorServer((args.host, args.port), coordinator) as server:
            host = f"[{args.host}]" if args.host == "::1" else args.host
            print(f"Gateway coordinator listening at http://{host}:{server.server_port}; approved grouped Volcengine pool.", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except Exception:
        print("Gateway coordinator could not start or continue; check private inputs and the loopback listener.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
