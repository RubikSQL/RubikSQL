"""Allocate gateway accounts without persisting their credentials in checkpoints."""

from __future__ import annotations

from contextlib import ExitStack, closing, contextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import threading
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterator, Literal, Mapping, Sequence
from uuid import uuid4

from .file_identity import file_identity
from .key_broker import BROKER_STATE_ENV, LocalKeyBroker, SqliteKeyBroker, broker_enabled, key_id

if TYPE_CHECKING:
    from .gateway_coordinator import RemoteGatewayPool


GATEWAY_BASE_URL = "https://st8tp3ajl0df3n8b8l8qu.apigateway-cn-beijing.volceapi.com/v1"
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_POOL_PATH = Path.home() / ".config" / "rubiksql-hb" / "credentials" / "experiment-gateway-20260928.json"
CredentialFailure = Literal["invalid", "exhausted"]
Profile = Literal["volcengine", "official"]
AccountProvider = Literal["volcengine", "dashscope", "deepseek"]
ModelProvider = Literal["qwen", "deepseek"]
_OperationPhase = Literal[
    "state_lock", "state_directory", "state_connect", "state_begin", "state_snapshot", "state_schema",
    "state_pool_read", "state_initialize", "state_migrate", "state_configuration",
    "state_event_schema", "state_validate", "state_body", "state_commit", "pace_tokens", "pace_reserve",
]
_OFFICIAL_URLS = {"dashscope": DASHSCOPE_BASE_URL, "deepseek": DEEPSEEK_BASE_URL}
#: The September DeepSeek id; cache and stage identities of builds made with it retain this id. Since early
#: November 2026 the gateway answers it with the October build (``deepseek-v4-1-flash-261017``), which the exact
#: served-model check refuses as a substitution, so new requests use :data:`DEEPSEEK_CANDIDATE_MODEL`.
DEEPSEEK_MODEL = "deepseek-v4.1-flash"
#: The October Volcengine route: DeepSeek candidates and, by default, knowledge building.
DEEPSEEK_CANDIDATE_MODEL = "volcengine/deepseek-v4.1-flash"
#: The dated build the gateway serves for :data:`DEEPSEEK_CANDIDATE_MODEL`, the only served id that route accepts.
DEEPSEEK_CANDIDATE_SERVED_MODEL = "deepseek-v4-1-flash-261017"
#: The model id each profile's DeepSeek route receives for :data:`DEEPSEEK_MODEL` (:meth:`ProviderPool.model_id`).
#: ``volcengine``: the gateway's own name, served there on 2026-09-29. ``official``: DeepSeek's API name for the same
#: flash model, from the 2026-09-26 probe (the API refused ``deepseek-v4.1-flash`` and accepted ``deepseek-flash``),
#: unverified since; official chat routes are currently prohibited by project policy.
DEEPSEEK_ROUTE_MODELS: Mapping[str, str] = MappingProxyType({"volcengine": DEEPSEEK_MODEL, "official": "deepseek-flash"})
GROUP_ENV = "RUBIKSQL_GATEWAY_GROUP_ID"
COORDINATOR_ENV = ("RUBIKSQL_GATEWAY_COORDINATOR", "RUBIKSQL_GATEWAY_COORDINATOR_TOKEN")
_GROUP: ContextVar[str | None] = ContextVar("rubiksql_provider_group", default=None)
_DATABASE_BATCH_SIZE = 11
#: The only scope an embedding pool's keys may carry: they serve ``text-embedding-v4`` and never a chat request.
EMBEDDING_SCOPE = "embedding-only"
#: The host broker's key pool names (``key_broker``); chat and embedding keys never share one.
CHAT_KEY_POOL = "volcengine-chat"
EMBEDDING_KEY_POOL = "dashscope-embedding"
_EMBEDDING_STATUSES = frozenset({"available", "invalid", "exhausted", "retired"})
#: Durable account statuses the host broker may hand out (see ``GatewayPool._round_robin``).
_BROKER_ELIGIBLE = frozenset({"available", "exhausted"})
_BUCKET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
#: Tables :mod:`.pool_rates` keeps in the allocation state file. They are not allocation state, so a file holding
#: only these (or no table at all) is a fresh, uninitialized allocation state.
_PACING_TABLES = frozenset({"gateway_request_pacing", "gateway_token_reservations"})
_PROCESS_POOLS: dict[tuple[Any, ...], GatewayPool] = {}
_PROCESS_POOLS_LOCK = threading.Lock()


def _reset_process_pools() -> None:
    global _PROCESS_POOLS_LOCK
    _PROCESS_POOLS.clear()
    _PROCESS_POOLS_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_process_pools)


def current_group_id() -> str | None:
    """Return the current context's group for an explicitly inherited child environment."""
    return _GROUP.get() or os.environ.get(GROUP_ENV)


@contextmanager
def group_context(group_id: str) -> Iterator[None]:
    """Select one durable database group for concurrent work in this context.

    Args:
        group_id: A group id returned by ``GatewayPool.database_groups``.

    Yields:
        A provider scope without changing the parent experiment or process environment.

    Raises:
        GatewayConfigurationError: The group identity is empty or malformed.
    """
    if not isinstance(group_id, str) or not group_id or group_id != group_id.strip():
        raise GatewayConfigurationError("A durable provider group identity is required.")
    token = _GROUP.set(group_id)
    try:
        yield
    finally:
        _GROUP.reset(token)


def provider_claim() -> None:
    """Verify a Git evolution launch claim before a provider request, when marked."""
    if "RUBIKSQL_PAID_LAUNCH" in os.environ:
        from .prompts.paid_launch import verify_claim

        verify_claim(environment=os.environ)


class GatewayConfigurationError(RuntimeError):
    """Refuse missing or inconsistent account configuration."""


def _diagnostic_message(message: str, phase: _OperationPhase,
                        error: OSError | ValueError | sqlite3.Error) -> str:
    """Append a controlled phase and exact integer codes without exception text."""
    details = [f"phase={phase}"]
    for name in ("errno", "winerror", "sqlite_errorcode"):
        value = getattr(error, name, None)
        if type(value) is int:
            details.append(f"{name}={value}")
    return message + " [" + "; ".join(details) + "]"


#: Evaluation processes preserve an account interruption instead of backfilling answers.
EXIT_PROVIDER_UNAVAILABLE = 75
EXIT_PROVIDER_CONFIGURATION = 78


class GatewayUnavailableError(RuntimeError):
    """Report a persisted, recoverable experiment checkpoint without credentials."""

    def __init__(self, experiment_id: str, state_path: Path) -> None:
        self.experiment_id = experiment_id
        self.state_path = state_path
        self.checkpoint = {
            "experiment_id": experiment_id,
            "status": "blocked",
            "reason": "approved_gateway_accounts_unavailable",
            "state_path": str(state_path),
        }
        super().__init__(
            "Provider accounts are unavailable for the selected profile; the experiment "
            "allocation checkpoint is preserved. Restore an account before resuming."
        )

    def __reduce__(self):
        """Preserve the recovery identity across spawned database workers."""
        return type(self), (self.experiment_id, self.state_path)


def require_deepseek_model(model: str, *, candidate: bool = False) -> str:
    """Require the default or an explicitly enabled October model.

    Args:
        model: A requested DeepSeek model id.
        candidate: Also permit the October Volcengine route for an explicit caller.

    Returns:
        The approved requested id, unchanged.

    Raises:
        GatewayConfigurationError: Any other id, including a route's own wire name.
    """
    if model != DEEPSEEK_MODEL and not (candidate and model == DEEPSEEK_CANDIDATE_MODEL):
        raise GatewayConfigurationError(f"New DeepSeek calls require model {DEEPSEEK_MODEL!r}.")
    return model


def knowledge_deepseek_model() -> str:
    """Resolve the explicitly selected knowledge-building model.

    The default is the pinned October Volcengine route, the one the gateway still
    serves under its requested identity. ``RUBIKSQL_KNOWLEDGE_DEEPSEEK_MODEL`` may
    name the September id instead, which keeps a September-built root's stage
    identities (fully cached stages make no request). It does not change answering
    model defaults.

    Returns:
        str: Approved model identifier; the October route when no override is set.

    Raises:
        GatewayConfigurationError: The override names an unsupported model.
    """
    model = os.environ.get("RUBIKSQL_KNOWLEDGE_DEEPSEEK_MODEL", DEEPSEEK_CANDIDATE_MODEL)
    return require_deepseek_model(model, candidate=True)


@dataclass(frozen=True, slots=True)
class GatewayLease:
    """Carry a single experiment's account; never log ``secret``."""

    index: int
    secret: str = field(repr=False)
    base_url: str
    experiment_id: str
    credential_revision: int = 1
    group_id: str | None = None


@dataclass
class _PoolProcessState:
    """Share rotation and verification caches, never mutable credential views."""

    lock: threading.RLock = field(default_factory=threading.RLock)
    snapshot_identity: tuple[Any, ...] | None = None
    schema_identity: tuple[Any, ...] | None = None
    cursors: dict[tuple[str, str], int] = field(default_factory=dict)
    paired_identities: set[str] = field(default_factory=set)
    embedding_broker: LocalKeyBroker = field(default_factory=LocalKeyBroker)


@dataclass(frozen=True, slots=True)
class ProviderPool:
    """Route one model family through the selected profile's account pool."""

    owner: GatewayPool = field(repr=False)
    provider: AccountProvider
    base_url: str

    def model_id(self, model: str) -> str:
        """Return the model id a request on this route sends for a project chat model id.

        Qwen ids are unchanged. The DeepSeek knowledge default follows
        :data:`DEEPSEEK_ROUTE_MODELS`; the October candidate id is Volcengine-only.

        Args:
            model: The project model id a caller requested.

        Returns:
            The provider-native id to put in the request's ``model`` field.

        Raises:
            GatewayConfigurationError: An unapproved DeepSeek id or provider route.
        """
        if model == DEEPSEEK_CANDIDATE_MODEL:
            if self.provider != "volcengine":
                raise GatewayConfigurationError("The October DeepSeek candidate requires the Volcengine route.")
            return model
        if not model.startswith("deepseek"):
            return model
        require_deepseek_model(model)
        if self.provider == "dashscope":
            raise GatewayConfigurationError("The DashScope chat route serves Qwen models only.")
        return DEEPSEEK_ROUTE_MODELS["volcengine" if self.provider == "volcengine" else "official"]

    def lease(self, experiment_id: str | None = None, *, group_id: str | None = None,
              advance: bool = True) -> GatewayLease:
        """Select this provider's next account; runtime construction only previews it.

        With the host key broker on (:func:`key_broker.broker_enabled`, the default), every Volcengine request
        takes the next available chat key of the whole pool in strict round-robin order shared by the host's
        processes; database groups no longer narrow the choice. With it off, the group rotation applies.

        Args:
            experiment_id: Stable experiment identity, or the configured default.
            group_id: Durable database group, or the current group context.
            advance: Consume the turn for a request. Runtime construction
                passes False because its request-local copy leases again before sending.

        Returns:
            The next available account and its credential revision.

        Raises:
            GatewayConfigurationError: Identity, state or cursor policy is invalid.
            GatewayUnavailableError: No eligible account remains.
        """
        if type(advance) is not bool:
            raise GatewayConfigurationError("The group chat cursor policy is invalid.")
        provider_claim()
        if self.provider == "volcengine":
            identity = self.owner._identity(experiment_id)
            selected = self.owner._group_id(group_id)
            if self.owner.file_keys is not None:
                self.owner.file_keys.check_window()
                self.owner._assert_snapshot()
                # Runtime construction must never reserve a wire request or cache a lent secret.
                return GatewayLease(0, "file-key-selected-at-send", GATEWAY_BASE_URL, identity, 1, selected)
            if broker_enabled():
                return self.owner._chat_round_robin(identity, selected, advance=advance)
            if self.owner._uses_pairs(identity):
                if selected is None:
                    raise GatewayConfigurationError("A grouped parent requires its durable database group; no parent account is allocated.")
                return self.owner._paired_lease(identity, selected, advance=advance)
            return self.owner._allocate(identity)
        return self.owner._allocate_official(self.provider, self.owner._identity(experiment_id))

    def rotate(
        self, experiment_id: str, failed_index: int, reason: CredentialFailure,
        *, failed_revision: int | None = None, group_id: str | None = None,
    ) -> GatewayLease:
        """Rotate only this provider after a classified credential failure."""
        if self.provider == "volcengine":
            identity = self.owner._identity(experiment_id)
            group = self.owner._group_id(group_id)
            if broker_enabled():
                self.owner._fail_chat(failed_index, reason, failed_revision)
                return self.owner._chat_round_robin(identity, group, advance=False)
            if self.owner._uses_pairs(identity):
                if group is None:
                    raise GatewayConfigurationError("A grouped parent requires its durable database group; no parent account is allocated.")
                return self.owner._paired_lease(identity, group,
                                               failed_index=failed_index, reason=reason,
                                               failed_revision=failed_revision, advance=False)
            return self.owner.rotate(identity, failed_index, reason, failed_revision=failed_revision)
        return self.owner._allocate_official(
            self.provider, self.owner._identity(experiment_id), failed_index, reason, failed_revision,
        )

    def before_request(self, lease: GatewayLease, model: str, payload: Any) -> None:
        """Verify a marked launch and pace this account/model if configured."""
        provider_claim()
        from .chat_keys import DeniedChatKey, refused
        if refused(lease.secret, f"chat index {lease.index}", self.owner._chat_denylist):
            raise DeniedChatKey("A denylisted chat credential is retired; no request sent.")
        self.owner._pace_chat(lease, self.provider, model, payload)


class GatewayPool:
    """Own account allocation shared by every process in an experiment.

    The bootstrap JSON contains the Volcengine ``base_url`` and an
    ordered nonempty ``keys`` list. Optional ``official`` entries hold separate
    DashScope and DeepSeek chat pools; ``profile`` selects their fixed routes.
    SQLite stores only a pool fingerprint, account indexes/statuses, ordinary and
    database batch cursors, experiment mappings, and configuration revision metadata.
    Explicit maintenance writes an immutable private snapshot before atomically
    publishing its SQLite pointer. Callers must pause work before maintenance and
    construct a new pool when resuming; stale instances fail closed.

    Args:
        pool_path: Private account file. No environment credential fallback.
        state_path: Durable SQLite state, defaulting beside the account file.
        experiment_id: Default stable experiment identity, when supplied.

    Raises:
        GatewayConfigurationError: The file, endpoint, or account pool is invalid.
    """

    def __init__(
        self,
        pool_path: Path | str,
        state_path: Path | str | None = None,
        *,
        experiment_id: str | None = None,
    ) -> None:
        self.pool_path = Path(pool_path)
        self.state_path = (
            Path(state_path) if state_path is not None
            else self.pool_path.with_suffix(".state.sqlite3")
        )
        self.experiment_id = experiment_id
        if self.pool_path.resolve() == self.state_path.resolve():
            raise GatewayConfigurationError("Gateway allocation state must be separate from the private account file.")
        self._configuration_revision = 0
        self._configuration_path = self.pool_path
        metadata = self._configuration_metadata()
        if metadata is not None:
            revision, snapshot, _ = metadata
            if snapshot is not None and self._uses_key_files(self.pool_path):
                raise GatewayConfigurationError("A lent file pool cannot adopt a managed inline credential snapshot.")
            self._configuration_revision = revision
            if snapshot is not None:
                self._configuration_path = Path(snapshot)
        config = self._read_configuration(self._configuration_path)
        self._adopt(config, self._configuration_path, self._configuration_revision)
        if metadata is not None and metadata[2] != self._configuration_fingerprint:
            raise GatewayConfigurationError("The private configuration does not match its published revision.")

    @property
    def configuration_path(self) -> Path:
        """Return the current private snapshot path for a separate verification state.

        This is an internal file handoff, not a public status field. A verifier can
        construct a pool from this path with independent allocation state without
        copying credentials into a report or moving the production cursor.
        """
        return self._configuration_path

    @property
    def configuration_revision(self) -> int:
        """Return the pinned revision after checking the current durable pointer."""
        if self.state_path.exists():
            with self._read_state(allow_uninitialized=True):
                pass
        else:
            self._assert_snapshot()
        return self._configuration_revision

    @property
    def profile(self) -> Profile:
        """Return the selected chat profile from the verified private snapshot."""
        return self._profile

    @property
    def chat_pacing_enabled(self) -> bool:
        """Whether every physical chat request needs the shared pacing gate."""
        return bool(self._config.get("chat_limits"))

    @property
    def configuration_directory(self) -> Path:
        """Return the private managed snapshot directory, never an output location."""
        return self.pool_path.parent / f".{self.pool_path.stem}.configurations"

    def _configuration_metadata(self) -> tuple[int, str | None, str] | None:
        if not self.state_path.exists():
            return None
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.state_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='gateway_configuration'",
            ).fetchone()
            if exists is None:
                return None
            row = connection.execute(
                "SELECT active_revision, snapshot, fingerprint FROM gateway_configuration WHERE id=1",
            ).fetchone()
            if row is None:
                raise GatewayConfigurationError("The gateway configuration pointer is incomplete.")
            if type(row[0]) is not int or row[0] < 0 or not isinstance(row[2], str):
                raise GatewayConfigurationError("The gateway configuration pointer is invalid.")
            if (row[0] == 0) != (row[1] is None):
                raise GatewayConfigurationError("The gateway configuration pointer is invalid.")
            if row[1] is not None:
                if not isinstance(row[1], str):
                    raise GatewayConfigurationError("The gateway configuration snapshot path is invalid.")
                snapshot = Path(row[1])
                if not snapshot.is_absolute() or not snapshot.resolve().is_relative_to(self.configuration_directory.resolve()):
                    raise GatewayConfigurationError("The gateway configuration snapshot path is invalid.")
            return row
        except (OSError, sqlite3.Error):
            raise GatewayConfigurationError("Cannot read durable gateway configuration metadata.") from None
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _uses_key_files(path: Path) -> bool:
        """Detect the immutable file-pool format without reading any referenced credential.

        Managed inline pools may have removed their bootstrap after publishing a snapshot,
        so a missing bootstrap retains the legacy metadata path.
        """
        try:
            config = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return False
        return isinstance(config, dict) and "key_files" in config

    @staticmethod
    def _read_configuration(path: Path) -> dict[str, Any]:
        try:
            config = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            raise GatewayConfigurationError("Cannot read the private gateway account file.") from None
        if not isinstance(config, dict) or config.get("base_url") != GATEWAY_BASE_URL:
            raise GatewayConfigurationError("The gateway endpoint is not the explicitly approved endpoint.")
        if "key_files" in config:
            from .lent_keys import validate_files
            paths = validate_files(config["key_files"])
            if "keys" in config or config.get("profile", "volcengine") != "volcengine" or config.get("official"):
                raise GatewayConfigurationError("File-backed lent chat keys require the exclusive approved Volcengine pool.")
            config["keys"] = [str(item) for item in paths]
            if "embedding_source" in config:
                try:
                    source_path = Path(config["embedding_source"])
                    if source_path.resolve().is_relative_to(paths[0].parent.resolve()):
                        raise ValueError
                    source = json.loads(source_path.read_text(encoding="utf-8-sig"))
                    if "key_files" in source:
                        raise ValueError
                    config["embedding"] = source["embedding"]
                except (OSError, ValueError, KeyError, TypeError):
                    raise GatewayConfigurationError("Cannot read the unchanged embedding source configuration.") from None
        keys = config.get("keys")
        if (
            not isinstance(keys, list) or not keys
            or any(not isinstance(key, str) or not key.strip() or key != key.strip() for key in keys)
            or len(set(keys)) != len(keys)
        ):
            raise GatewayConfigurationError("The gateway pool must contain distinct nonempty account keys.")
        embedding = config.get("embedding")
        paired = isinstance(embedding, dict) and "keys" in embedding
        if paired:
            GatewayPool._validate_embedding_pool(embedding)
        elif "embedding" in config and (
            not isinstance(embedding, dict) or set(embedding) != {"api_key"}
            or not isinstance(embedding.get("api_key"), str)
            or not embedding["api_key"].strip() or embedding["api_key"] != embedding["api_key"].strip()
        ):
            raise GatewayConfigurationError("The private embedding configuration must contain a nonempty account key only.")
        profile = config.get("profile", "volcengine")
        if profile not in ("volcengine", "official"):
            raise GatewayConfigurationError("The selected chat profile is invalid.")
        if paired and profile != "volcengine":
            raise GatewayConfigurationError("The embedding pool requires the approved Volcengine chat profile.")
        if "chat_limits" in config:
            limits = config["chat_limits"]
            if (not isinstance(limits, dict) or not limits
                    or set(limits) - {"requests_per_minute", "tokens_per_minute"}
                    or any(type(value) is not int or value < 1 for value in limits.values())):
                raise GatewayConfigurationError("Chat pacing requires explicitly configured positive limits.")
        official = config.get("official", {})
        if not isinstance(official, dict) or set(official) - set(_OFFICIAL_URLS):
            raise GatewayConfigurationError("The official chat account configuration is invalid.")
        for provider, value in official.items():
            if not isinstance(value, dict) or set(value) != {"keys"}:
                raise GatewayConfigurationError("The official chat account configuration is invalid.")
            provider_keys = value["keys"]
            if (not isinstance(provider_keys, list) or not provider_keys
                or any(not isinstance(key, str) or not key.strip() or key != key.strip() for key in provider_keys)
                or len(set(provider_keys)) != len(provider_keys)):
                raise GatewayConfigurationError("Official chat pools require distinct nonempty keys.")
        if profile == "official" and set(official) != set(_OFFICIAL_URLS):
            raise GatewayConfigurationError("Both official chat account pools are required for the official profile.")
        chat_keys = set(keys).union(*(value["keys"] for value in official.values()))
        if paired and chat_keys.intersection(embedding["keys"]):
            raise GatewayConfigurationError("Embedding-only credentials must not appear in any chat account pool.")
        return config

    @staticmethod
    def _validate_embedding_pool(embedding: Mapping[str, Any]) -> None:
        """Refuse an embedding pool that is not distinct keys with one quota bucket id each and positive limits.

        ``scope`` is optional when reading (snapshots published before 2026-10-29 lack it) and, when present,
        must be :data:`EMBEDDING_SCOPE`; :meth:`set_embedding_pool` always writes it.
        """
        required = {"keys", "bucket_ids", "requests_per_minute", "tokens_per_minute"}
        keys = embedding.get("keys")
        if (not required <= set(embedding) <= required | {"scope"}
                or embedding.get("scope", EMBEDDING_SCOPE) != EMBEDDING_SCOPE
                or not isinstance(keys, list) or not 1 <= len(keys) <= 64
                or any(not isinstance(key, str) or not key or key != key.strip()
                       or any(character.isspace() for character in key) for key in keys)
                or len(set(keys)) != len(keys)
                or not isinstance(embedding.get("bucket_ids"), list) or len(embedding["bucket_ids"]) != len(keys)
                or any(not isinstance(bucket, str) or _BUCKET_ID.fullmatch(bucket) is None
                       for bucket in embedding["bucket_ids"])
                or any(type(embedding.get(name)) is not int or embedding[name] < 1
                       for name in ("requests_per_minute", "tokens_per_minute"))):
            raise GatewayConfigurationError(
                "The private embedding pool requires distinct embedding-only keys, one quota bucket id per key and positive limits.")

    @property
    def paired(self) -> bool:
        """Whether a managed embedding-only key pool is configured beside the chat pool (database groups need it)."""
        return "keys" in (self._config.get("embedding") or {})

    @property
    def embedding_account_count(self) -> int:
        """How many keys embedding requests rotate over: the managed pool's size, else one."""
        return len(self._config["embedding"]["keys"]) if self.paired else 1

    @property
    def chat_account_count(self) -> int:
        """How many Volcengine chat keys the pool lists, retired ones included."""
        return len(self._keys)

    def _uses_pairs(self, identity: str) -> bool:
        """Restrict database group allocation to grouped parents and marked paid scopes."""
        if not self.paired:
            return False
        if "RUBIKSQL_PAID_LAUNCH" in os.environ:
            return True
        with self._local.lock:
            if identity in self._local.paired_identities:
                return True
        if not self.state_path.exists():
            return False
        try:
            with self._read_state(allow_uninitialized=True) as connection:
                paired = (self._table_exists(connection, "gateway_database_groups")
                          and connection.execute("SELECT 1 FROM gateway_database_groups WHERE experiment_id=?", (identity,)).fetchone() is not None)
            if paired:
                with self._local.lock:
                    self._local.paired_identities.add(identity)
            return paired
        except (OSError, sqlite3.Error):
            raise GatewayConfigurationError("Cannot read the durable provider group allocation.") from None

    @staticmethod
    def _group_id(group_id: str | None = None) -> str | None:
        value = group_id if group_id is not None else current_group_id()
        if value is not None and (not isinstance(value, str) or not value or value != value.strip()):
            raise GatewayConfigurationError("The provider group identity is invalid.")
        return value

    def database_groups(self, experiment_id: str, database_ids: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
        """Split available chat accounts evenly across a parent experiment's databases.

        Args:
            experiment_id: Existing canonical parent evaluation or build identity.
            database_ids: Distinct database ids in canonical caller order, at most eleven.

        Returns:
            One group per database with ``group_id``, ``chat_indexes`` and singleton
            ``database_ids``, or one legacy group when the embedding pool is absent.
            Embedding keys are not assigned to databases: every embedding request
            takes the next key of the whole embedding pool. All eligible chat indexes
            are distributed in rounds from a separate batch cursor, skipping
            unavailable indexes. Group sizes differ by at most one.
            Every new experiment advances that cursor by eleven modulo pool size,
            including smaller scopes. Retries return the current assignment without
            moving the cursor; they include account replacements (``rotate``).
            Existing single-account groups retain their original allocation.

        Raises:
            GatewayConfigurationError: Invalid scope or a changed durable assignment.
            GatewayUnavailableError: Too few chat accounts are eligible;
                no assignment or cursor change is committed.
        """
        identity = self._identity(experiment_id)
        databases = tuple(database_ids)
        if (not databases or len(databases) > _DATABASE_BATCH_SIZE
                or any(not isinstance(value, str) or not value or value != value.strip() for value in databases)
                or len(set(databases)) != len(databases)):
            raise GatewayConfigurationError("Provider groups require a distinct ordered scope of at most eleven databases.")
        status = self.status()
        if not self.paired:
            return ({"group_id": "legacy", "pair_index": None, "database_ids": databases},)
        # Refuse an initially undersized pool without creating allocation state.
        # Existing mappings remain readable when one of their accounts is blocked.
        if not status["initialized"] and sum(row["status"] == "available" for row in status["accounts"]) < len(databases):
            raise GatewayUnavailableError(identity, self.state_path)
        with self._transaction() as connection:
            existing = connection.execute("SELECT groups_json FROM gateway_database_groups WHERE experiment_id=?",
                                          (identity,)).fetchone()
            if existing is not None:
                groups = self._read_database_groups(existing[0])
                if tuple(group["database_ids"][0] for group in groups) != databases:
                    raise GatewayConfigurationError("The parent experiment's durable database groups differ.")
            else:
                if connection.execute("SELECT 1 FROM gateway_experiments WHERE experiment_id=?", (identity,)).fetchone():
                    raise GatewayConfigurationError("An existing experiment account cannot be reassigned to database groups.")
                available = {index for (index,) in connection.execute(
                    "SELECT key_index FROM gateway_accounts WHERE status='available'")}
                if len(available) < len(databases):
                    raise GatewayUnavailableError(identity, self.state_path)
                cursor = self._batch_cursor(connection)
                indexes = [(cursor + offset) % len(self._keys) for offset in range(len(self._keys))
                           if (cursor + offset) % len(self._keys) in available]
                groups = [{"group_id": f"db-{ordinal}", "chat_indexes": indexes[ordinal::len(databases)],
                           "database_ids": [database]}
                          for ordinal, database in enumerate(databases)]
                connection.execute("INSERT INTO gateway_database_groups VALUES (?,?)",
                                   (identity, json.dumps(groups, separators=(",", ":"))))
                connection.execute("UPDATE gateway_database_batch SET next_batch_index=? WHERE id=1",
                                   ((cursor + _DATABASE_BATCH_SIZE) % len(self._keys),))
            self._bind_experiment(connection, identity)
        with self._local.lock:
            self._local.paired_identities.add(identity)
        return tuple({**group, "chat_indexes": tuple(group["chat_indexes"]),
                      "database_ids": tuple(group["database_ids"])} for group in groups)

    def _read_database_groups(self, encoded: str) -> list[dict[str, Any]]:
        try:
            groups = json.loads(encoded)
        except (TypeError, ValueError):
            raise GatewayConfigurationError("The durable provider group assignment is inconsistent.") from None
        if isinstance(groups, list) and any(isinstance(item, dict) and "pair_index" in item for item in groups):
            raise GatewayConfigurationError("The durable provider groups use an unsupported assignment format; existing assignments cannot be reassigned.")
        # Read existing experiments as singleton sets, without expanding or
        # reassigning them. A subsequent rotation writes the canonical format.
        if isinstance(groups, list):
            for item in groups:
                if isinstance(item, dict) and "chat_index" in item and "chat_indexes" not in item:
                    item["chat_indexes"] = [item.pop("chat_index")]
        # Assignments made before 2026-10-29 carry the retired per-database ``embedding_index``;
        # it is read and kept, never used.
        if (not isinstance(groups, list) or not 1 <= len(groups) <= _DATABASE_BATCH_SIZE
                or any(not isinstance(item, dict)
                       or set(item) - {"embedding_index"} != {"group_id", "chat_indexes", "database_ids"}
                       or item["group_id"] != f"db-{ordinal}"
                       or not isinstance(item["chat_indexes"], list) or not item["chat_indexes"]
                       or any(type(index) is not int or not 0 <= index < len(self._keys) for index in item["chat_indexes"])
                       or len(set(item["chat_indexes"])) != len(item["chat_indexes"])
                       or ("embedding_index" in item and type(item["embedding_index"]) is not int)
                       or not isinstance(item["database_ids"], list) or len(item["database_ids"]) != 1
                       or not isinstance(item["database_ids"][0], str) or not item["database_ids"][0]
                       or item["database_ids"][0] != item["database_ids"][0].strip()
                       for ordinal, item in enumerate(groups))
                or len({item["database_ids"][0] for item in groups}) != len(groups)):
            raise GatewayConfigurationError("The durable provider group assignment is inconsistent.")
        return groups

    def _batch_cursor(self, connection: sqlite3.Connection) -> int:
        if not self._table_exists(connection, "gateway_database_batch"):
            return 0
        row = connection.execute("SELECT next_batch_index FROM gateway_database_batch WHERE id=1").fetchone()
        if row is None or type(row[0]) is not int or not 0 <= row[0] < len(self._keys):
            raise GatewayConfigurationError("The durable database batch cursor is inconsistent.")
        return row[0]

    @staticmethod
    def _create_pair_tables(connection: sqlite3.Connection, count: int) -> None:
        """Create the grouped-allocation tables and check the embedding accounts against a ``count``-key pool.

        Rows past ``count`` are kept as ``retired`` history (a pool may shrink), never deleted.
        """
        initialized = GatewayPool._table_exists(connection, "gateway_embedding_accounts")
        batch_initialized = GatewayPool._table_exists(connection, "gateway_database_batch")
        connection.execute("CREATE TABLE IF NOT EXISTS gateway_database_groups ("
                           "experiment_id TEXT PRIMARY KEY, groups_json TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS gateway_database_batch ("
                           "id INTEGER PRIMARY KEY CHECK(id=1),next_batch_index INTEGER NOT NULL)")
        if not batch_initialized:
            connection.execute("INSERT INTO gateway_database_batch VALUES (1,0)")
        connection.execute("CREATE TABLE IF NOT EXISTS gateway_embedding_accounts ("
                           "key_index INTEGER PRIMARY KEY,status TEXT NOT NULL,credential_revision INTEGER NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS gateway_embedding_account_events ("
                           "event_id INTEGER PRIMARY KEY,key_index INTEGER NOT NULL,credential_revision INTEGER NOT NULL,"
                           "action TEXT NOT NULL,configuration_revision INTEGER NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS gateway_group_checkpoints ("
                           "experiment_id TEXT NOT NULL,group_id TEXT NOT NULL,pair_index INTEGER NOT NULL,"
                           "status TEXT NOT NULL,PRIMARY KEY(experiment_id,group_id))")
        connection.execute("CREATE TABLE IF NOT EXISTS gateway_group_chat_cursors ("
                           "experiment_id TEXT NOT NULL,group_id TEXT NOT NULL,next_position INTEGER NOT NULL,"
                           "PRIMARY KEY(experiment_id,group_id))")
        if not initialized:
            connection.executemany("INSERT INTO gateway_embedding_accounts VALUES (?,'available',1)",
                                   ((index,) for index in range(count)))
        GatewayPool._embedding_rows(connection, count)

    @staticmethod
    def _embedding_rows(connection: sqlite3.Connection, count: int) -> dict[int, tuple[str, int]]:
        """Read the durable embedding accounts ``{index: (status, credential revision)}`` of a ``count``-key pool."""
        accounts = connection.execute("SELECT key_index,status,credential_revision FROM gateway_embedding_accounts ORDER BY key_index").fetchall()
        if ([row[0] for row in accounts] != list(range(len(accounts))) or len(accounts) < count
                or any(row[1] not in _EMBEDDING_STATUSES or type(row[2]) is not int or row[2] < 1 for row in accounts)
                or any(row[1] != "retired" for row in accounts[count:])):
            raise GatewayConfigurationError("The durable embedding account checkpoints are inconsistent.")
        return {index: (status, revision) for index, status, revision in accounts[:count]}

    def _paired_lease(self, identity: str, group_id: str, *,
                      failed_index: int | None = None, reason: CredentialFailure | None = None,
                      failed_revision: int | None = None, advance: bool = True) -> GatewayLease:
        """Lease a database group's chat account (the fallback when the host key broker is off)."""
        if not self.paired:
            raise GatewayConfigurationError("Separate chat and embedding pools are required.")
        if failed_index is not None and type(failed_index) is not int:
            raise GatewayConfigurationError("The failed group account index is invalid.")
        if failed_index is not None:
            self._index(failed_index)
        if failed_index is not None and reason not in ("invalid", "exhausted"):
            raise GatewayConfigurationError("Only invalid or exhausted group credentials may be blocked.")
        if failed_revision is not None and (type(failed_revision) is not int or failed_revision < 1):
            raise GatewayConfigurationError("The failed group credential revision is invalid.")
        if failed_index is None:
            with self._read_state() as connection:
                groups, group, accounts, seed = self._group_state(connection, identity, group_id)
                indexes = group["chat_indexes"]
                healthy = any(accounts[index][0] == "available" for index in indexes)
                repair = any(
                    accounts[index][0] != "available"
                    and self._replacement_chat_index(connection, groups, index, indexes,
                                                     allow_shared=not healthy) is not None
                    for index in indexes
                )
                checkpoint = connection.execute(
                    "SELECT status FROM gateway_group_checkpoints WHERE experiment_id=? AND group_id=?",
                    (identity, group_id),
                ).fetchone() if self._table_exists(connection, "gateway_group_checkpoints") else None
                repair = repair or (not healthy and checkpoint != ("blocked",)) or (healthy and checkpoint == ("blocked",))
            if not repair:
                return self._group_lease(identity, group_id, group, accounts, seed, advance=advance)
        with self._transaction() as connection:
            groups, group, accounts, seed = self._group_state(connection, identity, group_id)
            indexes = group["chat_indexes"]
            # A delayed failure of an account the group already left changes nothing.
            if failed_index is not None and failed_index in indexes:
                index = failed_index
                revision = connection.execute("SELECT credential_revision FROM gateway_accounts WHERE key_index=?", (index,)).fetchone()[0]
                if failed_revision is None and revision != 1:
                    raise GatewayConfigurationError("The failed group credential revision is required.")
                if failed_revision in (None, revision):
                    changed = connection.execute("UPDATE gateway_accounts SET status=? WHERE key_index=? AND status='available'",
                                                 (reason, index)).rowcount
                    if changed:
                        self._account_event(connection, index, reason, self._configuration_revision)
            accounts = {index: (status, revision) for index, status, revision in connection.execute(
                "SELECT key_index,status,credential_revision FROM gateway_accounts")}
            changed = False
            for position, index in enumerate(indexes):
                if accounts[index][0] == "available":
                    continue
                replacement = self._replacement_chat_index(
                    connection, groups, index, indexes,
                    allow_shared=not any(accounts[member][0] == "available" for member in indexes))
                if replacement is not None:
                    indexes[position] = replacement
                    changed = True
            if changed:
                connection.execute("UPDATE gateway_database_groups SET groups_json=? WHERE experiment_id=?",
                                   (json.dumps(groups, separators=(",", ":")), identity))
            available = [index for index in indexes if accounts[index][0] == "available"]
            chat_index = available[0] if available else indexes[0]
            connection.execute("INSERT INTO gateway_group_checkpoints VALUES (?,?,?,?) "
                               "ON CONFLICT(experiment_id,group_id) DO UPDATE SET "
                               "pair_index=excluded.pair_index,status=excluded.status "
                               "WHERE pair_index!=excluded.pair_index OR status!=excluded.status",
                               (identity, group_id, chat_index, "active" if available else "blocked"))
        return self._group_lease(identity, group_id, group, accounts, seed, advance=advance)

    def _group_state(
        self, connection: sqlite3.Connection, identity: str, group_id: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, tuple[str, int]], int]:
        row = connection.execute("SELECT groups_json FROM gateway_database_groups WHERE experiment_id=?",
                                 (identity,)).fetchone()
        groups = [] if row is None else self._read_database_groups(row[0])
        group = next((item for item in groups if item["group_id"] == group_id), None)
        if group is None:
            raise GatewayConfigurationError("The provider request has no exact durable database group.")
        binding = connection.execute(
            "SELECT profile FROM gateway_experiment_profiles WHERE experiment_id=?", (identity,),
        ).fetchone()
        if binding != (self.profile,):
            raise GatewayConfigurationError("The durable provider group's profile binding is inconsistent.")
        accounts = {index: ("retired" if index in self._denied_chat else status, revision)
                    for index, status, revision in connection.execute(
            "SELECT key_index,status,credential_revision FROM gateway_accounts")}
        if (set(accounts) != set(range(len(self._keys)))
                or any(status not in {"available", "invalid", "exhausted", "retired"}
                       or type(revision) is not int or revision < 1 for status, revision in accounts.values())):
            raise GatewayConfigurationError("Gateway allocation state has inconsistent accounts.")
        # Older executables may still update this table. Read it as an initial
        # seed only; never reset a live process's cursor or write it per request.
        cursor = connection.execute(
            "SELECT next_position FROM gateway_group_chat_cursors WHERE experiment_id=? AND group_id=?",
            (identity, group_id),
        ).fetchone() if self._table_exists(connection, "gateway_group_chat_cursors") else None
        seed = 0 if cursor is None else cursor[0]
        if type(seed) is not int or not 0 <= seed < len(group["chat_indexes"]):
            raise GatewayConfigurationError("The durable group chat cursor is inconsistent.")
        return groups, group, accounts, seed

    def _group_lease(
        self, identity: str, group_id: str, group: Mapping[str, Any],
        accounts: Mapping[int, tuple[str, int]], seed: int, *, advance: bool,
    ) -> GatewayLease:
        indexes = group["chat_indexes"]
        available = [position for position, index in enumerate(indexes) if accounts[index][0] == "available"]
        if not available:
            raise GatewayUnavailableError(identity, self.state_path)
        with self._local.lock:
            key = (identity, group_id)
            next_position = self._local.cursors.get(key, seed)
            position = min(available, key=lambda value: (value - next_position) % len(indexes))
            if advance:
                self._local.cursors[key] = (position + 1) % len(indexes)
            index = indexes[position]
        return self._chat_lease(index, GATEWAY_BASE_URL, identity, accounts[index][1], group_id)

    def _chat_lease(self, index: int, base_url: str, identity: str, revision: int,
                    group_id: str | None = None, *, keys: Sequence[str] | None = None) -> GatewayLease:
        """Build a chat lease, refusing any key the pool marks embedding-only.

        Every chat allocation path ends here, so a configuration that slipped an embedding-only key into a chat
        pool still cannot send a chat request with it.
        """
        secret = ("file-key-selected-at-send" if self.file_keys is not None and keys is None
                  else (self._keys if keys is None else keys)[index])
        if self.file_keys is None or keys is not None:
            from .chat_keys import DeniedChatKey, refused
            if refused(secret, f"chat index {index}", self._chat_denylist):
                raise DeniedChatKey("A denylisted chat credential is retired; no request sent.")
        if secret in self._embedding_only:
            raise GatewayConfigurationError("An embedding-only credential cannot serve a chat request.")
        return GatewayLease(index, secret, base_url, identity, revision, group_id)

    def _chat_accounts(self) -> dict[int, tuple[str, int]]:
        """The durable chat accounts ``{index: (status, credential revision)}``; all available before any state."""
        if not self.state_path.exists():
            self._assert_snapshot()
            return {index: ("available", 1) for index in range(len(self._keys))}
        with self._read_state(allow_uninitialized=True) as connection:
            if self._allocated(connection):
                return {index: (status, revision) for index, status, revision in connection.execute(
                    "SELECT key_index,status,credential_revision FROM gateway_accounts")}
        return {index: ("available", 1) for index in range(len(self._keys))}

    def _embedding_accounts(self) -> dict[int, tuple[str, int]]:
        """The durable embedding accounts ``{index: (status, credential revision)}``; all available before any state."""
        count = len(self._config["embedding"]["keys"])
        if self.state_path.exists():
            with self._read_state(allow_uninitialized=True) as connection:
                if self._table_exists(connection, "gateway_embedding_accounts"):
                    return self._embedding_rows(connection, count)
        else:
            self._assert_snapshot()
        return {index: ("available", 1) for index in range(count)}

    def _key_broker(self, pool: str) -> LocalKeyBroker | SqliteKeyBroker:
        """The host broker for this pool's state, or the process-local one for embeddings when the host broker is off."""
        if not broker_enabled():
            if pool == CHAT_KEY_POOL:
                raise GatewayConfigurationError("Chat requests use their database group while the host key broker is off.")
            return self._local.embedding_broker
        configured = os.environ.get(BROKER_STATE_ENV, "").strip()
        path = (Path(configured).expanduser() if configured
                else self.state_path.with_name(self.state_path.stem + ".broker.sqlite3"))
        return SqliteKeyBroker.for_path(path)

    def _round_robin(self, pool: str, keys: Sequence[str], accounts: Mapping[int, tuple[str, int]],
                     *, advance: bool) -> tuple[int, int] | None:
        """Take the next eligible key of ``pool`` from its broker: ``(index, credential revision)`` or ``None``.

        Retired and invalid accounts are never eligible. A durably ``exhausted`` account (recorded by the
        per-database fallback, or before the broker existed) is: daily quota comes back, so the broker's own
        30-minute and quota-day marks decide when it is skipped, exactly as for a key it marked itself.
        """
        ids = [key_id(index, secret) for index, secret in enumerate(keys)]
        eligible = {ids[index] for index, (status, _) in accounts.items()
                    if status in _BROKER_ELIGIBLE and index < len(ids)
                    and (pool == EMBEDDING_KEY_POOL or index not in self._denied_chat)}
        chosen = self._key_broker(pool).next(pool, ids, eligible, advance=advance)
        if chosen is None:
            return None
        index = ids.index(chosen)
        return index, accounts[index][1]

    def _chat_round_robin(self, identity: str, group_id: str | None, *, advance: bool) -> GatewayLease:
        """Lease the next available Volcengine chat key of the whole pool from the host broker."""
        if self.profile != "volcengine":
            raise GatewayConfigurationError("The host key broker serves the approved Volcengine chat pool only.")
        chosen = self._round_robin(CHAT_KEY_POOL, self._keys, self._chat_accounts(), advance=advance)
        if chosen is None:
            raise GatewayUnavailableError(identity, self.state_path)
        return self._chat_lease(chosen[0], GATEWAY_BASE_URL, identity, chosen[1], group_id)

    def _fail_chat(self, index: int, reason: CredentialFailure, failed_revision: int | None) -> None:
        """Record a broker-mode chat account failure: retire an invalid key, skip an exhausted one for a while.

        ``invalid`` (HTTP 401/403 and other account rejections) retires the account durably until maintenance
        restores it; a failure reported for an older credential revision changes nothing. ``exhausted`` (daily
        quota or balance) only marks the key in the host broker, which skips it for 30 minutes or until the
        quota day ends.
        """
        self._index(index)
        if reason not in ("invalid", "exhausted"):
            raise GatewayConfigurationError("Only invalid or exhausted credentials may be blocked.")
        if failed_revision is not None and (type(failed_revision) is not int or failed_revision < 1):
            raise GatewayConfigurationError("The failed credential revision is invalid.")
        if reason == "exhausted":
            self._key_broker(CHAT_KEY_POOL).mark_exhausted(CHAT_KEY_POOL, key_id(index, self._keys[index]))
            return
        with self._transaction() as connection:
            revision = connection.execute("SELECT credential_revision FROM gateway_accounts WHERE key_index=?",
                                          (index,)).fetchone()[0]
            if failed_revision in (None, revision) and connection.execute(
                    "UPDATE gateway_accounts SET status=? WHERE key_index=? AND status IN ('available','exhausted')",
                    (reason, index)).rowcount:
                self._account_event(connection, index, reason, self._configuration_revision)

    def _replacement_chat_index(self, connection: sqlite3.Connection, groups: Sequence[Mapping[str, Any]],
                                current: int, members: Sequence[int], *, allow_shared: bool) -> int | None:
        """Choose the chat account a database group moves to when its own is unavailable.

        Prefer an account unused by this experiment's groups, searching cyclically
        after ``current``. When there is no spare, keep using healthy group members;
        share another group's account only when no member remains available. Replacement
        does not reserve a new experiment batch or move either allocation cursor.
        """
        available = {index for (index,) in connection.execute(
            "SELECT key_index FROM gateway_accounts WHERE status='available'")}
        unused = available - {index for group in groups for index in group["chat_indexes"]}
        eligible = unused or (available - set(members) if allow_shared else set())
        return next((index for offset in range(1, len(self._keys))
                     if (index := (current + offset) % len(self._keys)) in eligible), None)

    def embedding_lease(self, experiment_id: str | None = None) -> GatewayLease | None:
        """Take the next embedding key of the managed embedding-only pool, in strict round-robin order.

        No key belongs to a database: each request takes the next available key of the whole pool, from the host
        key broker (shared by the host's processes) or, with the broker off, from this process's own rotation.

        Args:
            experiment_id: Canonical parent identity, or the configured default; only labels the lease.

        Returns:
            The next embedding lease, a configured legacy single-key lease, or ``None``
            when no embedding credential was configured.

        Raises:
            GatewayConfigurationError: The configuration or state is inconsistent.
            GatewayUnavailableError: Every embedding key is retired or skipped for daily exhaustion.
        """
        if not self.paired:
            self.status()
            key = (self._config.get("embedding") or {}).get("api_key")
            return GatewayLease(0, key, DASHSCOPE_BASE_URL, self.experiment_id or "legacy-embedding") if key else None
        identity = self._identity(experiment_id) if (experiment_id or self.experiment_id) else "embedding"
        keys = self._config["embedding"]["keys"]
        chosen = self._round_robin(EMBEDDING_KEY_POOL, keys, self._embedding_accounts(), advance=True)
        if chosen is None:
            raise GatewayUnavailableError(identity, self.state_path)
        return GatewayLease(chosen[0], keys[chosen[0]], DASHSCOPE_BASE_URL, identity, chosen[1], self._group_id())

    def block_embedding(self, lease: GatewayLease, reason: CredentialFailure) -> None:
        """Record an embedding key's account failure; the next request takes the next key.

        Args:
            lease: The failed request's lease and credential revision.
            reason: ``invalid`` retires the key durably (until :meth:`restore_embedding`); ``exhausted`` (daily
                quota or balance) skips it for 30 minutes or until the quota day ends. Throttling is neither.

        Returns:
            None: A delayed failure of a replaced credential revision leaves it available.

        Raises:
            GatewayConfigurationError: The lease or failure classification is invalid.
        """
        if not self.paired:
            raise GatewayConfigurationError("Legacy embedding credentials have no rotation policy.")
        keys = self._config["embedding"]["keys"]
        if type(lease.index) is not int or not 0 <= lease.index < len(keys) or reason not in ("invalid", "exhausted"):
            raise GatewayConfigurationError("The failed embedding account or its classification is invalid.")
        revision = lease.credential_revision
        if type(revision) is not int or revision < 1:
            raise GatewayConfigurationError("The failed embedding credential revision is invalid.")
        if reason == "exhausted":
            broker = self._key_broker(EMBEDDING_KEY_POOL)
            broker.mark_exhausted(EMBEDDING_KEY_POOL, key_id(lease.index, keys[lease.index]))
            return
        with self._transaction() as connection:
            self._create_pair_tables(connection, len(keys))
            current = connection.execute("SELECT credential_revision FROM gateway_embedding_accounts WHERE key_index=?",
                                         (lease.index,)).fetchone()[0]
            if current == revision and connection.execute(
                    "UPDATE gateway_embedding_accounts SET status=? WHERE key_index=? AND status IN ('available','exhausted')",
                    (reason, lease.index)).rowcount:
                connection.execute("INSERT INTO gateway_embedding_account_events "
                                   "(key_index,credential_revision,action,configuration_revision) VALUES (?,?,?,?)",
                                   (lease.index, revision, reason, self._configuration_revision))

    def _pace_chat(self, lease: GatewayLease, provider: str, model: str, payload: Any) -> None:
        limits = self._config.get("chat_limits")
        if limits is not None:
            self._pace(f"{provider}:account-{lease.index}", model, payload, limits)

    def pace_embedding(self, lease: GatewayLease, model: str, payload: Any) -> None:
        """Pace an embedding pool request using its configured quota bucket.

        Args:
            lease: The database group's fixed embedding account lease.
            model: Embedding model whose requests share the quota bucket.
            payload: Request content used to reserve tokens.

        Returns:
            None: The request waits until its account bucket permits it.

        Raises:
            GatewayConfigurationError: The shared reservation is refused.
        """
        provider_claim()
        if self.paired:
            config = self._config["embedding"]
            if type(lease.index) is not int or not 0 <= lease.index < len(config["bucket_ids"]):
                raise GatewayConfigurationError("The embedding lease names no configured account.")
            self._pace("dashscope-embedding:" + config["bucket_ids"][lease.index], model, payload, config)

    def _pace(self, bucket: str, model: str, payload: Any, limits: Mapping[str, Any]) -> None:
        from .pool_rates import token_reservation, wait

        if self._local.schema_identity is None:
            # Pacing keeps its tables in this file. Create the allocation state in its own single
            # transaction first, so no process ever finds a file holding pacing tables alone.
            with self._transaction():
                pass
        phase: _OperationPhase = "pace_tokens"
        try:
            tokens = token_reservation(payload)
            phase = "pace_reserve"
            wait(self.state_path, bucket=bucket, model=model, tokens=tokens,
                 requests_per_minute=limits.get("requests_per_minute"), tokens_per_minute=limits.get("tokens_per_minute"))
        except (OSError, ValueError, sqlite3.Error) as error:
            raise GatewayConfigurationError(_diagnostic_message(
                "The shared provider request reservation was refused.", phase, error)) from None

    def _adopt(self, config: dict[str, Any], path: Path, revision: int) -> None:
        self._config = config
        self._configuration_path = path
        self._configuration_revision = revision
        self._keys = tuple(config["keys"])
        from .chat_keys import denylist, refused
        self._chat_denylist = denylist()
        self._denied_chat = frozenset(index for index, secret in enumerate(self._keys)
                                     if "key_files" not in config and refused(secret, f"chat index {index}", self._chat_denylist))
        self._denied_official = {provider: frozenset(index for index, secret in enumerate(value["keys"])
                                                    if refused(secret, f"{provider} chat index {index}", self._chat_denylist))
                                 for provider, value in config.get("official", {}).items()}
        self.file_keys = None
        if "key_files" in config:
            from .lent_keys import LentKeyPool, validate_files
            self.file_keys = LentKeyPool(validate_files(config["key_files"]))
        self._embedding_only = frozenset((config.get("embedding") or {}).get("keys") or ())
        self._profile: Profile = config.get("profile", "volcengine")
        self._fingerprint = hashlib.sha256(
            json.dumps([GATEWAY_BASE_URL, config["keys"]], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._configuration_fingerprint = self._config_fingerprint(config)
        self._local = _PoolProcessState()

    @staticmethod
    def _config_fingerprint(config: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> GatewayPool | RemoteGatewayPool:
        """Load the home-based private file or explicit environment overrides.

        Args:
            environ: Environment mapping; defaults to the process environment.
                ``RUBIKSQL_GATEWAY_POOL`` overrides the default private file under
                ``~/.config/rubiksql-hb/credentials``;
                ``RUBIKSQL_GATEWAY_STATE`` overrides its adjacent state path;
                ``RUBIKSQL_GATEWAY_EXPERIMENT_ID`` supplies the stable identity.
                The coordinator URL and private token-file path select the shared
                cross-host pool together; partial configuration is refused.

        Returns:
            A pinned configuration view sharing process-local rotation and
            verification state with callers using the same paths and experiment.
            Maintenance on one view cannot update an existing runtime's pinned
            credentials. Each call checks the published revision; leases recheck
            durable state.

        Raises:
            GatewayConfigurationError: No configured pool can be loaded.
        """
        env = os.environ if environ is None else environ
        path = Path(env.get("RUBIKSQL_GATEWAY_POOL") or DEFAULT_POOL_PATH).expanduser()
        file_pool = cls._uses_key_files(path)
        if any(name in env for name in COORDINATOR_ENV) and not file_pool:
            if not all(env.get(name, "").strip() for name in COORDINATOR_ENV):
                raise GatewayConfigurationError("The gateway coordinator URL and private token file are both required.")
            from .gateway_coordinator import RemoteGatewayPool

            return RemoteGatewayPool.from_env(env)
        state = Path(env["RUBIKSQL_GATEWAY_STATE"]).expanduser() if env.get("RUBIKSQL_GATEWAY_STATE") else path.with_suffix(".state.sqlite3")
        if file_pool and env.get("RUBIKSQL_GATEWAY_STATE"):
            namespace = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
            state = state.with_name(f"{state.stem}.filekeys-{namespace}{state.suffix}")
        experiment = env.get("RUBIKSQL_GATEWAY_EXPERIMENT_ID")
        try:
            path, state = path.resolve(), state.resolve()
            key = (cls, os.getpid(), os.path.normcase(str(path)),
                   os.path.normcase(str(state)), experiment)
            with _PROCESS_POOLS_LOCK:
                pool = _PROCESS_POOLS.get(key)
                if pool is None:
                    pool = cls(path, state, experiment_id=experiment)
                    _PROCESS_POOLS[key] = pool
            # Read outside the factory lock: unrelated workers do not serialize
            # their SQLite snapshots or file checks through a process-wide gate.
            metadata = pool._configuration_metadata()
            if metadata is not None and metadata[0] != pool._configuration_revision:
                with _PROCESS_POOLS_LOCK:
                    current = _PROCESS_POOLS[key]
                    if current is pool:
                        current = cls(path, state, experiment_id=experiment)
                        _PROCESS_POOLS[key] = current
                    pool = current
                    metadata = pool._configuration_metadata()
            expected_snapshot = str(pool._configuration_path.resolve()) if pool._configuration_revision else None
            if ((metadata is None and pool._configuration_revision)
                    or (metadata is not None and metadata != (
                        pool._configuration_revision, expected_snapshot, pool._configuration_fingerprint))):
                raise GatewayConfigurationError("The private configuration does not match its published revision.")
            pool._assert_snapshot()
            # Configurations are replaced by _adopt, never mutated in place.
            # A shallow view shares only the process cache; maintenance adopts
            # new credentials on its own view and leaves old runtimes pinned.
            return copy(pool)
        except OSError as error:
            raise GatewayConfigurationError(_diagnostic_message(
                "Cannot read the private gateway configuration.", "state_connect", error)) from None

    def for_provider(self, provider: ModelProvider) -> ProviderPool:
        """Return the selected profile's fixed route for Qwen or DeepSeek.

        Args:
            provider: ``qwen`` or ``deepseek`` model family.

        Returns:
            A view with the provider endpoint and durable lease/rotate operations.

        Raises:
            GatewayConfigurationError: The family is unknown or state is stale.
        """
        if provider not in ("qwen", "deepseek"):
            raise GatewayConfigurationError("Unknown chat model provider.")
        _ = self.configuration_revision
        return self._provider(provider)

    def _provider(self, provider: ModelProvider) -> ProviderPool:
        """Select a route; its caller owns configuration admission before exposure."""
        if provider not in ("qwen", "deepseek"):
            raise GatewayConfigurationError("Unknown chat model provider.")
        if self.profile == "volcengine":
            return ProviderPool(self, "volcengine", GATEWAY_BASE_URL)
        account_provider: AccountProvider = "dashscope" if provider == "qwen" else "deepseek"
        return ProviderPool(self, account_provider, _OFFICIAL_URLS[account_provider])

    def lease_provider(
        self, provider: ModelProvider, experiment_id: str | None = None, *,
        group_id: str | None = None, advance: bool = True,
    ) -> tuple[ProviderPool, GatewayLease]:
        """Select and admit a chat route in the lease's existing state snapshot.

        Args:
            provider: ``qwen`` or ``deepseek`` model family.
            experiment_id: Stable identity, or the configured default.
            group_id: Durable database group, or the current group context.
            advance: Consume the group's turn; False only previews runtime setup.

        Returns:
            The selected route and its admitted account. No unverified route escapes.

        Raises:
            GatewayConfigurationError: The route, configuration or durable state is invalid.
            GatewayUnavailableError: No eligible account remains.
        """
        route = self._provider(provider)
        # lease() checks the pinned configuration and current account state in
        # one read transaction (or the existing allocation/repair transaction).
        # Standalone for_provider() retains its own admission boundary.
        lease = route.lease(experiment_id, group_id=group_id, advance=advance)
        return route, lease

    def transport_identity(self, provider: ModelProvider) -> str:
        """Return a nonsecret cache identity stable across credential rotations."""
        route = self.for_provider(provider)
        family = provider if self.profile == "volcengine" else route.provider
        return f"{self.profile}:{family}:{route.base_url}"

    def lease(self, experiment_id: str | None = None) -> GatewayLease:
        """Lease a database group's next account or an ordinary experiment's account.

        Args:
            experiment_id: Stable experiment ID, or the configured default ID.
                Missing/blank identities are rejected rather than invented.

        Returns:
            A group's next healthy member, advancing its process-local round-robin cursor,
            or an ordinary experiment's current account without advancing its cursor.
            Unavailable accounts are replaced under the policy in ``rotate``.

        Raises:
            GatewayConfigurationError: Identity or persistent state is invalid.
            GatewayUnavailableError: No chat account is available, or the group's
                embedding account is blocked. The assignment remains for recovery.
        """
        identity = self._identity(experiment_id)
        if self.file_keys is not None:
            return self._provider("qwen").lease(identity)
        if self._uses_pairs(identity):
            group = self._group_id()
            if group is None:
                raise GatewayConfigurationError("Grouped preflight requires durable database groups; no parent account is allocated.")
            return self._paired_lease(identity, group)
        if self.profile == "official":
            return self._lease_official_bundle(identity)
        return self._allocate(identity)

    def rotate(
        self, experiment_id: str, failed_index: int, reason: CredentialFailure,
        *, failed_revision: int | None = None,
    ) -> GatewayLease:
        """Retain a failed account and resume the experiment under its allocation policy.

        Args:
            experiment_id: The experiment whose request failed.
            failed_index: Account index actually used by that request.
            reason: ``invalid`` or ``exhausted``; never ordinary throttling.
            failed_revision: The revision from the failed request's lease. A late
                failure cannot retire a replacement credential. Omission is
                accepted only for an account still at its initial revision.

        Returns:
            The next usable account, without consuming a group's request turn.
            A failed group member is durably replaced by an unused account when
            available. Otherwise the group uses its healthy members, sharing a
            sibling's account only when none remain. Concurrent reports converge
            on the same membership; stale reports cannot retire a restored revision.

        Raises:
            GatewayConfigurationError: Input is invalid or the experiment was
                never allocated.
            GatewayUnavailableError: No chat account is available, or the group's
                embedding account is blocked.
        """
        if self.profile != "volcengine":
            raise GatewayConfigurationError("Select a model provider before rotating an official account.")
        if reason not in ("invalid", "exhausted"):
            raise GatewayConfigurationError("Only invalid or exhausted gateway accounts can be rotated.")
        self._index(failed_index)
        if failed_revision is not None and (type(failed_revision) is not int or failed_revision < 1):
            raise GatewayConfigurationError("The failed gateway credential revision is invalid.")
        identity = self._identity(experiment_id)
        if self._uses_pairs(identity):
            group = self._group_id()
            if group is None:
                raise GatewayConfigurationError("A grouped parent requires its durable database group; no parent account is allocated.")
            return self._paired_lease(identity, group, failed_index=failed_index,
                                      reason=reason, failed_revision=failed_revision, advance=False)
        return self._allocate(identity, failed_index, reason, failed_revision)

    def replace_key(self, index: int, secret: str, *, provider: AccountProvider = "volcengine") -> None:
        """Replace one account's credential during paused maintenance.

        Args:
            index: Stable index of the same account; indexes are never
                reassigned to a different account implicitly.
            secret: Replacement key supplied privately, never in CLI arguments.
            provider: Account pool to maintain; defaults to the legacy Volcengine pool.

        Raises:
            GatewayConfigurationError: Input, published state, or snapshot writing
                is invalid. Cursor and experiment mappings are always preserved.

        A changed key increases this account's credential revision and makes only
        this account available again. Supplying the identical key is a no-op;
        use ``restore`` for an unchanged credential whose quota was replenished.
        """
        self._refuse_embedding_only(secret)
        if provider != "volcengine":
            self._replace_official_key(provider, index, secret)
            return
        self._index(index)
        self._secret(secret)
        with self._transaction() as connection:
            if secret == self._keys[index]:
                return
            if secret in self._keys:
                raise GatewayConfigurationError("The credential already belongs to another pool index.")
            config = {**self._config, "keys": list(self._keys)}
            config["keys"][index] = secret
            snapshot = self._publish_configuration(connection, config)
            connection.execute(
                "UPDATE gateway_accounts SET status='available', credential_revision=credential_revision+1 WHERE key_index=?",
                (index,),
            )
            self._account_event(connection, index, "replace", snapshot[1])
        self._adopt(config, *snapshot)

    def add_key(self, secret: str, *, provider: AccountProvider = "volcengine") -> int:
        """Append one account without renumbering existing ones.

        Args:
            secret: New account credential, supplied privately.
            provider: Account pool to maintain; defaults to the legacy Volcengine pool.

        Returns:
            Its new stable index. Allocation cursor and prior mappings do not move.

        Raises:
            GatewayConfigurationError: The key is empty, duplicated, or the
                current configuration cannot be safely published.
        """
        self._refuse_embedding_only(secret)
        if provider != "volcengine":
            return self._add_official_key(provider, secret)
        self._secret(secret)
        with self._transaction() as connection:
            if secret in self._keys:
                raise GatewayConfigurationError("The credential already belongs to a pool index.")
            index = len(self._keys)
            config = {**self._config, "keys": [*self._keys, secret]}
            snapshot = self._publish_configuration(connection, config)
            connection.execute("INSERT INTO gateway_accounts VALUES (?, 'available', 1)", (index,))
            self._account_event(connection, index, "add", snapshot[1])
        self._adopt(config, *snapshot)
        return index

    def retire(self, index: int, *, provider: AccountProvider = "volcengine") -> None:
        """Disable an account without deleting its index or experimental history.

        Args:
            index: Index to exclude from future allocation. Database groups and
                ordinary experiments choose another eligible chat account.
            provider: Account pool containing the index.

        Raises:
            GatewayConfigurationError: The index or persistent state is invalid.
        """
        if provider != "volcengine":
            self._set_official_status(provider, index, "retired")
            return
        self._index(index)
        with self._transaction() as connection:
            changed = connection.execute(
                "UPDATE gateway_accounts SET status='retired', credential_revision=credential_revision+1 "
                "WHERE key_index=? AND status!='retired'", (index,),
            ).rowcount
            if changed:
                self._account_event(connection, index, "retire", self._configuration_revision)

    def set_embedding_key(self, secret: str) -> None:
        """Set the DashScope embedding credential in the same private configuration.

        Args:
            secret: Credential for the already approved DashScope embedding route.
                Endpoint, model and dimensions remain fixed by the runtime.

        Raises:
            GatewayConfigurationError: The key or current configuration is invalid.
        """
        self._secret(secret)
        if self.paired:
            raise GatewayConfigurationError("An embedding pool cannot be replaced by a single key.")
        with self._transaction() as connection:
            if (self._config.get("embedding") or {}).get("api_key") == secret:
                return
            config = {**self._config, "embedding": {"api_key": secret}}
            snapshot = self._publish_configuration(connection, config)
        self._adopt(config, *snapshot)

    def set_embedding_pool(self, keys: Sequence[str], *, bucket_ids: Sequence[str] | None = None,
                           requests_per_minute: int = 1800, tokens_per_minute: int = 1_200_000,
                           chat_limits: Mapping[str, int] | None = None) -> None:
        """Publish the embedding-only key pool every embedding request rotates over.

        The keys are marked :data:`EMBEDDING_SCOPE`: no chat path may lease them, and the published file is
        refused when one also appears in a chat pool. Replacing the pool keeps each index's durable history: an
        index whose key changed becomes available at a new credential revision, a new index starts at revision
        one, and indexes past the new pool's size are retired.

        Args:
            keys: Distinct private embedding credentials in their fixed rotation order.
            bucket_ids: One nonsecret quota bucket (account) id per key; a key's pacing is shared with every
                key of the same bucket. Omission conservatively publishes one shared Beijing bucket for all keys.
            requests_per_minute: Positive configured embedding request budget per bucket.
            tokens_per_minute: Positive configured embedding token budget per bucket.
            chat_limits: Optional explicit positive chat RPM/TPM limits. Omission
                preserves existing chat pacing configuration and request behavior.

        Returns:
            None: Chat account indexes and existing database assignments are preserved.

        Raises:
            GatewayConfigurationError: The pool or current private revision is invalid.
        """
        embedding = {"keys": list(keys), "bucket_ids": list(bucket_ids) if bucket_ids is not None
                     else ["beijing-shared"] * len(keys), "scope": EMBEDDING_SCOPE,
                     "requests_per_minute": requests_per_minute, "tokens_per_minute": tokens_per_minute}
        self._validate_embedding_pool(embedding)
        if self.profile != "volcengine":
            raise GatewayConfigurationError("The embedding pool requires the approved Volcengine profile.")
        if chat_limits is not None and (not chat_limits or set(chat_limits) - {"requests_per_minute", "tokens_per_minute"}
                                       or any(type(value) is not int or value < 1 for value in chat_limits.values())):
            raise GatewayConfigurationError("Chat pacing requires explicitly configured positive limits.")
        old_keys = (self._config.get("embedding") or {}).get("keys")
        config = {**self._config, "embedding": embedding}
        if chat_limits is not None:
            config["chat_limits"] = dict(chat_limits)
        with self._transaction() as connection:
            if config == self._config:
                return
            existing = self._table_exists(connection, "gateway_embedding_accounts")
            self._create_pair_tables(connection, len(old_keys) if old_keys is not None and existing else len(keys))
            snapshot = self._publish_configuration(connection, config)
            if old_keys is not None:
                rows = {index: status for index, status in connection.execute(
                    "SELECT key_index,status FROM gateway_embedding_accounts")}
                changes = []
                for index, key in enumerate(keys):
                    if index not in rows:
                        connection.execute("INSERT INTO gateway_embedding_accounts VALUES (?,'available',1)", (index,))
                        changes.append((index, "add"))
                    elif index >= len(old_keys) or old_keys[index] != key or rows[index] == "retired":
                        connection.execute("UPDATE gateway_embedding_accounts SET status='available',"
                                           "credential_revision=credential_revision+1 WHERE key_index=?", (index,))
                        changes.append((index, "replace"))
                for index, status in rows.items():
                    if index >= len(keys) and status != "retired":
                        connection.execute("UPDATE gateway_embedding_accounts SET status='retired',"
                                           "credential_revision=credential_revision+1 WHERE key_index=?", (index,))
                        changes.append((index, "remove"))
                for index, action in changes:
                    revision = connection.execute("SELECT credential_revision FROM gateway_embedding_accounts WHERE key_index=?", (index,)).fetchone()[0]
                    connection.execute("INSERT INTO gateway_embedding_account_events "
                                       "(key_index,credential_revision,action,configuration_revision) VALUES (?,?,?,?)",
                                       (index, revision, action, snapshot[1]))
        self._adopt(config, *snapshot)

    def scrub_replaced_embedding_keys(self) -> int:
        """Rewrite this pool's inactive private files so no replaced embedding key survives in them.

        A published revision keeps its predecessors' snapshots (and the bootstrap file) as history; after
        :meth:`set_embedding_pool` replaced the embedding keys, those files still hold the old ones. Each such
        file's ``embedding`` section is replaced by the current one, written atomically beside it. The active
        snapshot, chat keys, durable state and every revision pointer are unchanged.

        Returns:
            int: How many files were rewritten.

        Raises:
            GatewayConfigurationError: No embedding pool or published revision exists yet (the bootstrap file
                would still be active), or a private file cannot be read or replaced.
        """
        if not self.paired:
            raise GatewayConfigurationError("Only a managed embedding pool has replaced keys to scrub.")
        rewritten = 0
        with self._transaction():
            if self._configuration_revision < 1:
                raise GatewayConfigurationError("Publish the embedding pool before scrubbing its predecessors.")
            current = self._config["embedding"]
            active = self._configuration_path.resolve()
            for path in [self.pool_path, *sorted(self.configuration_directory.glob("config-*.json"))]:
                if not path.exists() or path.resolve() == active:
                    continue
                try:
                    config = json.loads(path.read_text(encoding="utf-8-sig"))
                except (OSError, ValueError):
                    raise GatewayConfigurationError("Cannot read an inactive private configuration file.") from None
                if not isinstance(config, dict) or "embedding" not in config or config["embedding"] == current:
                    continue
                descriptor, temporary_name = tempfile.mkstemp(prefix=".scrub-", suffix=".json", dir=path.parent)
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                        json.dump({**config, "embedding": current}, handle, ensure_ascii=False, indent=2)
                        handle.write("\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, path)
                except OSError:
                    raise GatewayConfigurationError("Cannot replace an inactive private configuration file.") from None
                finally:
                    temporary.unlink(missing_ok=True)
                rewritten += 1
        return rewritten

    def restore_embedding(self, index: int) -> None:
        """Restore a funded or repaired embedding account at its existing index.

        Args:
            index: Existing embedding pool index.

        Returns:
            None: The account is available at a new credential revision.

        Raises:
            GatewayConfigurationError: The embedding index or state is invalid.
        """
        if not self.paired or type(index) is not int or not 0 <= index < len(self._config["embedding"]["keys"]):
            raise GatewayConfigurationError("The embedding pool account index is invalid.")
        with self._transaction() as connection:
            self._create_pair_tables(connection, len(self._config["embedding"]["keys"]))
            connection.execute("UPDATE gateway_embedding_accounts SET status='available',credential_revision=credential_revision+1 WHERE key_index=?", (index,))
            revision = connection.execute("SELECT credential_revision FROM gateway_embedding_accounts WHERE key_index=?", (index,)).fetchone()[0]
            connection.execute("INSERT INTO gateway_embedding_account_events "
                               "(key_index,credential_revision,action,configuration_revision) VALUES (?,?,'restore',?)",
                               (index, revision, self._configuration_revision))

    def embedding_key(self) -> str | None:
        """Read the configured DashScope key, or ``None`` when it was never set.

        Returns:
            The private key for runtime use only. Missing optional configuration
            yields ``None``; malformed or stale configuration never falls back.

        Raises:
            GatewayConfigurationError: The published configuration changed or
                its current private snapshot cannot be verified.
        """
        if self.paired:
            lease = self.embedding_lease()
            return lease.secret if lease is not None else None
        self.status()
        return (self._config.get("embedding") or {}).get("api_key")

    def select_profile(self, profile: Profile) -> None:
        """Publish an explicit paused switch of the default chat profile.

        Existing experiment bindings remain on their original profile. Callers
        must start a new process after publication, then explicitly rebind an
        existing experiment if they intend to continue it on the new profile.
        """
        if profile not in ("volcengine", "official"):
            raise GatewayConfigurationError("The requested chat profile is invalid.")
        if profile == "official" and set(self._config.get("official", {})) != set(_OFFICIAL_URLS):
            raise GatewayConfigurationError("Both official chat account pools must be configured before selection.")
        with self._transaction() as connection:
            if profile == self.profile:
                return
            config = {**self._config, "profile": profile}
            snapshot = self._publish_configuration(connection, config)
        self._adopt(config, *snapshot)

    def rebind_experiment(self, experiment_id: str) -> None:
        """Explicitly bind an existing experiment to the selected profile.

        Previous account mappings remain as historical checkpoints. The new
        profile makes its own allocation on the next lease.
        """
        identity = self._identity(experiment_id)
        with self._transaction() as connection:
            self._create_profile_bindings(connection)
            known = connection.execute(
                "SELECT 1 FROM gateway_experiment_profiles WHERE experiment_id=?", (identity,),
            ).fetchone()
            if known is None:
                known = connection.execute(
                    "SELECT 1 FROM gateway_experiments WHERE experiment_id=?", (identity,),
                ).fetchone()
            if known is None and self._table_exists(connection, "gateway_provider_experiments"):
                known = connection.execute(
                    "SELECT 1 FROM gateway_provider_experiments WHERE experiment_id=? LIMIT 1", (identity,),
                ).fetchone()
            if known is None:
                raise GatewayConfigurationError("Cannot rebind an experiment without an allocation checkpoint.")
            connection.execute(
                "INSERT INTO gateway_experiment_profiles VALUES (?, ?, 2) "
                "ON CONFLICT(experiment_id) DO UPDATE SET "
                "profile=excluded.profile, revision=gateway_experiment_profiles.revision+1",
                (identity, self.profile),
            )

    def experiment_binding(self, experiment_id: str) -> tuple[Profile, int] | None:
        """Read an experiment's durable profile and explicit binding revision."""
        identity = self._identity(experiment_id)
        if not self.state_path.exists():
            self._assert_snapshot()
            return None
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.state_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            if not self._allocated(connection):
                self._assert_snapshot()
                return None
            self._validate_state(connection)
            if self._table_exists(connection, "gateway_experiment_profiles"):
                row = connection.execute(
                    "SELECT profile, revision FROM gateway_experiment_profiles WHERE experiment_id=?",
                    (identity,),
                ).fetchone()
                if row is not None:
                    return row
            old = connection.execute(
                "SELECT 1 FROM gateway_experiments WHERE experiment_id=?", (identity,),
            ).fetchone()
            return ("volcengine", 1) if old else None
        except (OSError, sqlite3.Error):
            raise GatewayConfigurationError("Cannot read the durable experiment profile binding.") from None
        finally:
            if connection is not None:
                connection.close()

    def binding_revision(self, experiment_id: str) -> int:
        """Return the binding revision, or zero before first allocation."""
        binding = self.experiment_binding(experiment_id)
        return binding[1] if binding is not None else 0

    def restore(self, index: int, *, provider: AccountProvider = "volcengine") -> None:
        """Explicitly make a repaired account eligible again.

        Call only after a maintainer has verified that this account's invalid or
        exhausted condition is resolved. This method makes no provider request,
        changes no account identity, and never runs automatically on resume.

        Args:
            index: Existing pool index whose failure has been resolved.
            provider: Account pool containing the index.

        Raises:
            GatewayConfigurationError: The index or persistent pool state is
                invalid. The cursor and every experiment mapping are preserved.
        """
        if type(index) is int and index in (self._denied_chat if provider == "volcengine" else self._denied_official.get(provider, ())):
            return  # Policy retirement cannot be undone by account maintenance.
        if provider != "volcengine":
            self._set_official_status(provider, index, "available")
            return
        self._index(index)
        with self._transaction() as connection:
            changed = connection.execute(
                "UPDATE gateway_accounts SET status='available', credential_revision=credential_revision+1 "
                "WHERE key_index=? AND status!='available'", (index,),
            ).rowcount
            if changed:
                self._account_event(connection, index, "restore", self._configuration_revision)

    def status(self) -> dict[str, Any]:
        """Read allocation eligibility without creating or modifying state.

        Args:
            None.

        Returns:
            Selected profile, provider readiness, each account index and status,
            chat count, separate ordinary and database batch cursors, the
            embedding-only pool's account readiness, whether the host key broker is on, and blocked experiment IDs. ``next_index`` keeps
            its ordinary Volcengine meaning. No credentials, fingerprints,
            or file paths are exposed. ``available`` means eligible for allocation;
            it does not claim the account has passed a live provider check.

        Raises:
            GatewayConfigurationError: Existing state cannot be read or does not
                belong to the configured ordered pool.
        """
        fresh = {
            "initialized": False, "next_index": 0, "configuration_revision": 0,
            "embedding_configured": self._config.get("embedding") is not None,
            "accounts": [{"index": index, "status": "available", "credential_revision": 1}
                         for index in range(len(self._keys))],
            "blocked_experiments": [],
        }
        if not self.state_path.exists():
            self._assert_snapshot()
            return self._decorate_status(fresh)
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.state_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            if not self._allocated(connection):
                self._assert_snapshot()
                return self._decorate_status(fresh)
            self._validate_state(connection)
            revision_column = (
                "credential_revision" if "credential_revision" in {
                    row[1] for row in connection.execute("PRAGMA table_info(gateway_accounts)")
                } else "1"
            )
            return self._decorate_status({
                "initialized": True,
                "configuration_revision": self._configuration_revision,
                "embedding_configured": self._config.get("embedding") is not None,
                "next_index": connection.execute("SELECT next_index FROM gateway_pool WHERE id=1").fetchone()[0],
                "accounts": [
                    {"index": index, "status": status, "credential_revision": revision}
                    for index, status, revision in connection.execute(
                        f"SELECT key_index, status, {revision_column} FROM gateway_accounts ORDER BY key_index",
                    )
                ],
                "blocked_experiments": [
                    identity for (identity,) in connection.execute(
                        "SELECT experiment_id FROM gateway_experiments WHERE status='blocked' ORDER BY experiment_id",
                    )
                ],
            }, connection)
        except (OSError, sqlite3.Error):
            raise GatewayConfigurationError("Cannot read durable gateway allocation status.") from None
        finally:
            if connection is not None:
                connection.close()

    def _decorate_status(
        self, status: dict[str, Any], connection: sqlite3.Connection | None = None,
    ) -> dict[str, Any]:
        for account in status["accounts"]:
            if account["index"] in self._denied_chat:
                account["status"] = "retired"
        status["chat_account_count"] = len(self._keys)
        status["next_batch_index"] = self._batch_cursor(connection) if connection is not None else 0
        volc = {
            "configured": True, "ready": any(row["status"] == "available" for row in status["accounts"]),
            "base_url": GATEWAY_BASE_URL, "next_index": status["next_index"],
            "chat_account_count": status["chat_account_count"], "next_batch_index": status["next_batch_index"],
            "accounts": status["accounts"], "blocked_experiments": status["blocked_experiments"],
        }
        providers = {"volcengine": volc}
        for provider in _OFFICIAL_URLS:
            providers[provider] = self._official_status(provider, connection)
        paired_status: dict[str, Any] = {"configured": self.paired,
                                       "chat_account_count": len(self._keys),
                                       "database_batch_size": _DATABASE_BATCH_SIZE,
                                       "next_batch_index": status["next_batch_index"],
                                       "embedding_account_count": self.embedding_account_count if self.paired
                                       else int(status["embedding_configured"]),
                                       "embedding_bucket_count": len(set(self._config["embedding"]["bucket_ids"])) if self.paired else 0,
                                       "key_broker": "host" if broker_enabled() else "off"}
        if self.paired:
            count = self.embedding_account_count
            paired_status["embedding_scope"] = EMBEDDING_SCOPE
            rows = {index: ("available", 1) for index in range(count)}
            if connection is not None and self._table_exists(connection, "gateway_embedding_accounts"):
                rows = self._embedding_rows(connection, count)
            paired_status["embedding_accounts"] = [{"index": index, "status": state, "credential_revision": revision}
                                                   for index, (state, revision) in sorted(rows.items())]
            paired_status["blocked_groups"] = (connection.execute(
                "SELECT count(*) FROM gateway_group_checkpoints WHERE status='blocked'").fetchone()[0]
                if connection is not None and self._table_exists(connection, "gateway_group_checkpoints") else 0)
        return {**status, "profile": self.profile, "providers": providers, "paired": paired_status}

    def _official_status(self, provider: str, connection: sqlite3.Connection | None) -> dict[str, Any]:
        keys = self._official_keys(provider)
        accounts = [{"index": index, "status": "available", "credential_revision": 1}
                    for index in range(len(keys))]
        cursor = 0
        blocked: list[str] = []
        if connection is not None and self._table_exists(connection, "gateway_provider_pool"):
            row = connection.execute(
                "SELECT next_index FROM gateway_provider_pool WHERE provider=?", (provider,),
            ).fetchone()
            if row is not None:
                cursor = row[0]
                accounts = [
                    {"index": index, "status": state, "credential_revision": revision}
                    for index, state, revision in connection.execute(
                        "SELECT key_index, status, credential_revision FROM gateway_provider_accounts "
                        "WHERE provider=? ORDER BY key_index", (provider,),
                    )
                ]
                blocked = [identity for (identity,) in connection.execute(
                    "SELECT experiment_id FROM gateway_provider_experiments "
                    "WHERE provider=? AND status='blocked' ORDER BY experiment_id", (provider,),
                )]
        for account in accounts:
            if account["index"] in self._denied_official.get(provider, ()):
                account["status"] = "retired"
        return {
            "configured": bool(keys), "ready": any(row["status"] == "available" for row in accounts),
            "base_url": _OFFICIAL_URLS[provider], "next_index": cursor,
            "accounts": accounts, "blocked_experiments": blocked,
        }

    def _identity(self, experiment_id: str | None) -> str:
        identity = self.experiment_id if experiment_id is None else experiment_id
        if not isinstance(identity, str) or not identity.strip() or identity != identity.strip():
            raise GatewayConfigurationError("A stable RUBIKSQL_GATEWAY_EXPERIMENT_ID is required before gateway calls.")
        return identity

    def _index(self, index: int) -> None:
        if type(index) is not int or not 0 <= index < len(self._keys):
            raise GatewayConfigurationError("The gateway account index is outside the configured pool.")

    def _official_keys(self, provider: str) -> tuple[str, ...]:
        if provider not in _OFFICIAL_URLS:
            raise GatewayConfigurationError("Unknown official chat account provider.")
        return tuple(self._config.get("official", {}).get(provider, {}).get("keys", ()))

    @staticmethod
    def _official_fingerprint(provider: str, keys: tuple[str, ...] | list[str]) -> str:
        return hashlib.sha256(json.dumps(
            [_OFFICIAL_URLS[provider], keys], ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()

    @staticmethod
    def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
        return connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,),
        ).fetchone() is not None

    @staticmethod
    def _allocated(connection: sqlite3.Connection) -> bool:
        """Whether the file holds allocation state, complete or not.

        A fresh file holds no table, or only :data:`_PACING_TABLES`: connecting creates the file before the
        first writer's schema commits, and processes before 2026-11-05 paced before any allocation. Any other
        table means allocation state exists; :meth:`_validate_state` then requires it whole, and
        :meth:`_initialize_state` never completes a partial one.
        """
        names = {name for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        return any(not name.startswith("sqlite_") and name not in _PACING_TABLES for name in names)

    @staticmethod
    def _create_profile_bindings(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS gateway_experiment_profiles ("
            "experiment_id TEXT PRIMARY KEY, profile TEXT NOT NULL "
            "CHECK(profile IN ('volcengine','official')), "
            "revision INTEGER NOT NULL CHECK(revision>=1))"
        )

    def _bind_experiment(self, connection: sqlite3.Connection, identity: str) -> None:
        row = connection.execute(
            "SELECT profile FROM gateway_experiment_profiles WHERE experiment_id=?", (identity,),
        ).fetchone()
        if row is None:
            old = connection.execute(
                "SELECT 1 FROM gateway_experiments WHERE experiment_id=?", (identity,),
            ).fetchone()
            bound = "volcengine" if old else self.profile
            connection.execute(
                "INSERT INTO gateway_experiment_profiles VALUES (?, ?, 1)", (identity, bound),
            )
        else:
            bound = row[0]
        if bound != self.profile:
            raise GatewayConfigurationError(
                "Experiment profile differs from the selected profile; explicitly rebind while paused."
            )

    @staticmethod
    def _create_official_tables(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS gateway_provider_pool ("
            "provider TEXT PRIMARY KEY CHECK(provider IN ('dashscope','deepseek')), "
            "fingerprint TEXT NOT NULL, next_index INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS gateway_provider_accounts ("
            "provider TEXT NOT NULL, key_index INTEGER NOT NULL, "
            "status TEXT NOT NULL CHECK(status IN ('available','invalid','exhausted','retired')), "
            "credential_revision INTEGER NOT NULL CHECK(credential_revision>=1), "
            "PRIMARY KEY(provider,key_index))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS gateway_provider_experiments ("
            "provider TEXT NOT NULL, experiment_id TEXT NOT NULL, key_index INTEGER, "
            "status TEXT NOT NULL CHECK(status IN ('active','blocked')), "
            "PRIMARY KEY(provider,experiment_id))"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS gateway_provider_account_events ("
            "event_id INTEGER PRIMARY KEY, provider TEXT NOT NULL, "
            "configuration_revision INTEGER NOT NULL, key_index INTEGER NOT NULL, "
            "credential_revision INTEGER NOT NULL, action TEXT NOT NULL, "
            "created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))"
        )

    def _ensure_official_state(self, connection: sqlite3.Connection, provider: str) -> None:
        keys = self._official_keys(provider)
        self._create_official_tables(connection)
        row = connection.execute(
            "SELECT fingerprint FROM gateway_provider_pool WHERE provider=?", (provider,),
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO gateway_provider_pool VALUES (?, ?, 0)",
                (provider, self._official_fingerprint(provider, keys)),
            )
            connection.executemany(
                "INSERT INTO gateway_provider_accounts VALUES (?, ?, 'available', 1)",
                ((provider, index) for index in range(len(keys))),
            )
        elif row[0] != self._official_fingerprint(provider, keys):
            raise GatewayConfigurationError("Official account state does not match its published pool.")
        self._retire_denied(connection)

    def _validate_official_state(self, connection: sqlite3.Connection) -> None:
        if not self._table_exists(connection, "gateway_provider_pool"):
            return
        for provider in _OFFICIAL_URLS:
            row = connection.execute(
                "SELECT fingerprint FROM gateway_provider_pool WHERE provider=?", (provider,),
            ).fetchone()
            if row is None:
                continue
            keys = self._official_keys(provider)
            indexes = connection.execute(
                "SELECT key_index FROM gateway_provider_accounts WHERE provider=? ORDER BY key_index",
                (provider,),
            ).fetchall()
            if row[0] != self._official_fingerprint(provider, keys) or indexes != [
                (index,) for index in range(len(keys))
            ]:
                raise GatewayConfigurationError("Official account state does not match its published pool.")

    def _refuse_embedding_only(self, secret: str) -> None:
        """Refuse to add a retired or embedding-only credential to any chat pool."""
        self._secret(secret)
        from .chat_keys import DeniedChatKey, refused
        if refused(secret, "chat maintenance input", self._chat_denylist):
            raise DeniedChatKey("A denylisted chat credential cannot join a chat account pool.")
        if secret in self._embedding_only:
            raise GatewayConfigurationError("An embedding-only credential cannot join a chat account pool.")

    @staticmethod
    def _secret(secret: str) -> None:
        if not isinstance(secret, str) or not secret.strip() or secret != secret.strip():
            raise GatewayConfigurationError("A nonempty private credential is required.")

    def _configuration_file_identity(self) -> tuple[Any, ...]:
        """Identify edits as well as replacements, including restored Windows mtimes."""
        return file_identity(self._configuration_path)

    def _assert_snapshot(self) -> None:
        try:
            identity = self._configuration_file_identity()
            if identity == self._local.snapshot_identity:
                return
            with self._local.lock:
                if identity == self._local.snapshot_identity:
                    return
                current = self._read_configuration(self._configuration_path)
                if (self._config_fingerprint(current) != self._configuration_fingerprint
                        or identity != self._configuration_file_identity()):
                    raise GatewayConfigurationError("The private configuration does not match its published revision.")
                self._local.snapshot_identity = identity
        except OSError as error:
            raise GatewayConfigurationError(_diagnostic_message(
                "Cannot read the private gateway account file.", "state_snapshot", error)) from None

    def _state_identity(self, connection: sqlite3.Connection) -> tuple[Any, ...]:
        stat = self.state_path.stat()
        return (os.getpid(), str(self.state_path.resolve()), stat.st_dev, stat.st_ino,
                connection.execute("PRAGMA schema_version").fetchone()[0])

    @contextmanager
    def _read_state(self, *, allow_uninitialized: bool = False) -> Iterator[sqlite3.Connection]:
        """Hold one consistent read snapshot only while its caller copies rows."""
        phase: _OperationPhase = "state_connect"
        try:
            with closing(sqlite3.connect(self.state_path.resolve().as_uri() + "?mode=ro",
                                         uri=True, timeout=30, isolation_level=None)) as connection:
                connection.execute("PRAGMA query_only=ON")
                connection.execute("BEGIN")
                phase = "state_validate"
                if allow_uninitialized and not self._allocated(connection):
                    # Opening the first writer creates a file before its schema
                    # commits; a rolled-back first allocation leaves one too, and
                    # older processes paced before allocating. Admission may
                    # initialize it under the writer lock, but must still verify
                    # its pinned private configuration.
                    self._assert_snapshot()
                else:
                    self._validate_state(connection)
                phase = "state_body"
                yield connection
        except (OSError, sqlite3.Error) as error:
            raise GatewayConfigurationError(_diagnostic_message(
                "Cannot access durable gateway allocation state; no account fallback is permitted.", phase, error)) from None

    @staticmethod
    def _create_accounts(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS gateway_accounts (key_index INTEGER PRIMARY KEY, "
            "status TEXT NOT NULL CHECK(status IN ('available','invalid','exhausted','retired')), "
            "credential_revision INTEGER NOT NULL CHECK(credential_revision>=1))"
        )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        from .pool_rates import begin_immediate, commit, state_lock

        with ExitStack() as locks:
            connection: sqlite3.Connection | None = None
            phase: _OperationPhase = "state_lock"
            try:
                locks.enter_context(state_lock(self.state_path))
                phase = "state_directory"
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                phase = "state_connect"
                connection = sqlite3.connect(self.state_path, timeout=30, isolation_level=None)
                phase = "state_begin"
                begin_immediate(connection)
                phase = "state_snapshot"
                self._assert_snapshot()
                phase = "state_schema"
                if self._local.schema_identity != self._state_identity(connection):
                    self._initialize_state(connection)
                phase = "state_validate"
                self._validate_state(connection)
                self._retire_denied(connection)
                phase = "state_body"
                yield connection
                schema_identity = self._state_identity(connection)
                phase = "state_commit"
                commit(connection)
                # Failed setup/commit must not publish a trusted schema cache.
                self._local.schema_identity = schema_identity
            except (OSError, sqlite3.Error) as error:
                raise GatewayConfigurationError(_diagnostic_message(
                    "Cannot access durable gateway allocation state; no account fallback is permitted.", phase, error)) from None
            finally:
                if connection is not None:
                    if connection.in_transaction:
                        connection.rollback()
                    connection.close()

    def _retire_denied(self, connection: sqlite3.Connection) -> None:
        # Preserve account indexes and configuration fingerprints in existing allocation state.
        connection.executemany("UPDATE gateway_accounts SET status='retired' WHERE key_index=? AND status!='retired'",
                               [(index,) for index in self._denied_chat])
        if self._table_exists(connection, "gateway_provider_accounts"):
            connection.executemany("UPDATE gateway_provider_accounts SET status='retired' "
                                   "WHERE provider=? AND key_index=? AND status!='retired'",
                                   [(provider, index) for provider, indexes in self._denied_official.items() for index in indexes])

    def _initialize_state(self, connection: sqlite3.Connection) -> None:
        """Create or upgrade the allocation schema inside the caller's single writer transaction.

        The ``gateway_pool`` row is written in the same transaction as every base table, so it marks a complete
        initialization. Allocation tables without that row are a partial state that is refused, never completed.
        """
        if self._allocated(connection) and (not self._table_exists(connection, "gateway_pool") or connection.execute(
                "SELECT 1 FROM gateway_pool WHERE id=1").fetchone() is None):
            raise GatewayConfigurationError("Gateway allocation state is incomplete; it is not a committed initialization.")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS gateway_pool ("
            "id INTEGER PRIMARY KEY CHECK(id=1), schema_version INTEGER NOT NULL, "
            "fingerprint TEXT NOT NULL, next_index INTEGER NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS gateway_experiments ("
            "experiment_id TEXT PRIMARY KEY, key_index INTEGER, "
            "status TEXT NOT NULL CHECK(status IN ('active','blocked')))"
        )
        row = connection.execute("SELECT schema_version, fingerprint FROM gateway_pool WHERE id=1").fetchone()
        if row is None:
            connection.execute("INSERT INTO gateway_pool VALUES (1, 2, ?, 0)", (self._fingerprint,))
            self._create_accounts(connection)
            connection.executemany(
                "INSERT INTO gateway_accounts VALUES (?, 'available', 1)",
                ((index,) for index in range(len(self._keys))),
            )
        elif row == (1, self._fingerprint):
            # Migrate only under the writer; read-only preflight leaves v1 intact.
            connection.execute("ALTER TABLE gateway_accounts RENAME TO gateway_accounts_v1")
            self._create_accounts(connection)
            connection.execute("INSERT INTO gateway_accounts SELECT key_index, status, 1 FROM gateway_accounts_v1")
            connection.execute("DROP TABLE gateway_accounts_v1")
            connection.execute("UPDATE gateway_pool SET schema_version=2 WHERE id=1")
        elif row[0] != 2:
            raise GatewayConfigurationError("Gateway allocation state does not match this configured ordered pool.")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS gateway_configuration (id INTEGER PRIMARY KEY CHECK(id=1), "
            "active_revision INTEGER NOT NULL, snapshot TEXT, fingerprint TEXT NOT NULL)"
        )
        if row is None or row[0] == 1:
            connection.execute(
                "INSERT INTO gateway_configuration VALUES (1, 0, NULL, ?)", (self._configuration_fingerprint,),
            )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS gateway_account_events (event_id INTEGER PRIMARY KEY, "
            "configuration_revision INTEGER NOT NULL, key_index INTEGER NOT NULL, "
            "credential_revision INTEGER NOT NULL, action TEXT NOT NULL, "
            "created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))"
        )
        self._create_profile_bindings(connection)
        if self.paired:
            self._create_pair_tables(connection, len(self._config["embedding"]["keys"]))

    def _validate_configuration(self, connection: sqlite3.Connection) -> None:
        row = connection.execute("SELECT schema_version, fingerprint FROM gateway_pool WHERE id=1").fetchone()
        if row is None or row[0] not in (1, 2):
            raise GatewayConfigurationError("Gateway allocation state does not match this configured ordered pool.")
        if row[0] == 2:
            configured = connection.execute(
                "SELECT active_revision, snapshot, fingerprint FROM gateway_configuration WHERE id=1",
            ).fetchone()
            if configured is None or configured[0] != self._configuration_revision:
                raise GatewayConfigurationError("Gateway configuration changed; resume paused work with a new process.")
            expected_snapshot = str(self._configuration_path.resolve()) if self._configuration_revision else None
            if configured[1:] != (expected_snapshot, self._configuration_fingerprint):
                raise GatewayConfigurationError("The private configuration does not match its published revision.")
        if row[1] != self._fingerprint:
            raise GatewayConfigurationError("Gateway allocation state does not match this configured ordered pool.")

    def _validate_state(self, connection: sqlite3.Connection) -> None:
        self._validate_configuration(connection)
        self._assert_snapshot()
        indexes = connection.execute("SELECT key_index FROM gateway_accounts ORDER BY key_index").fetchall()
        if indexes != [(index,) for index in range(len(self._keys))]:
            raise GatewayConfigurationError("Gateway allocation state has an inconsistent account index set.")
        self._batch_cursor(connection)
        self._validate_official_state(connection)

    def _publish_configuration(self, connection: sqlite3.Connection, config: dict[str, Any]) -> tuple[Path, int]:
        if self.file_keys is not None:
            raise GatewayConfigurationError("Lent file pools are immutable; maintain the original embedding pool separately.")
        revision = self._configuration_revision + 1
        directory = self.configuration_directory
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name != "nt":
            directory.chmod(0o700)
        path = (directory / f"config-{revision}-{uuid4().hex}.json").resolve()
        descriptor, temporary_name = tempfile.mkstemp(prefix=".pending-", suffix=".json", dir=directory)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(config, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            self._read_configuration(path)
            if os.name != "nt":
                directory_fd = os.open(directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        # A complete private file exists before the single SQLite commit publishes
        # it. A precommit crash leaves an unused file, never a half-updated pool.
        fingerprint = hashlib.sha256(
            json.dumps([GATEWAY_BASE_URL, config["keys"]], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        connection.execute("UPDATE gateway_pool SET fingerprint=? WHERE id=1", (fingerprint,))
        connection.execute(
            "UPDATE gateway_configuration SET active_revision=?, snapshot=?, fingerprint=? WHERE id=1",
            (revision, str(path), self._config_fingerprint(config)),
        )
        return path, revision

    @staticmethod
    def _account_event(connection: sqlite3.Connection, index: int, action: str, configuration_revision: int) -> None:
        revision = connection.execute(
            "SELECT credential_revision FROM gateway_accounts WHERE key_index=?", (index,),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO gateway_account_events (configuration_revision, key_index, credential_revision, action) "
            "VALUES (?, ?, ?, ?)", (configuration_revision, index, revision, action),
        )

    def _allocate(
        self, experiment_id: str, failed_index: int | None = None,
        reason: CredentialFailure | None = None,
        failed_revision: int | None = None,
    ) -> GatewayLease:
        if failed_index is None and self.state_path.exists():
            with self._read_state(allow_uninitialized=True) as connection:
                schema = (connection.execute("SELECT schema_version FROM gateway_pool WHERE id=1").fetchone()
                          if self._table_exists(connection, "gateway_pool") else None)
                binding = connection.execute(
                    "SELECT profile FROM gateway_experiment_profiles WHERE experiment_id=?", (experiment_id,),
                ).fetchone() if self._table_exists(connection, "gateway_experiment_profiles") else None
                if binding is not None and binding != (self.profile,):
                    raise GatewayConfigurationError(
                        "Experiment profile differs from the selected profile; explicitly rebind while paused."
                    )
                if schema == (2,) and binding is not None:
                    row = connection.execute(
                        "SELECT a.key_index,a.credential_revision FROM gateway_experiments e "
                        "JOIN gateway_accounts a ON a.key_index=e.key_index "
                        "WHERE e.experiment_id=? AND e.status='active' AND a.status='available'",
                        (experiment_id,),
                    ).fetchone()
                    if row is not None and row[0] not in self._denied_chat:
                        self._index(row[0])
                        return self._chat_lease(row[0], GATEWAY_BASE_URL, experiment_id, row[1])
        selected: int | None = None
        with self._transaction() as connection:
            self._bind_experiment(connection, experiment_id)
            row = connection.execute(
                "SELECT key_index FROM gateway_experiments WHERE experiment_id=?", (experiment_id,),
            ).fetchone()
            if failed_index is not None and row is None:
                raise GatewayConfigurationError("Cannot rotate a gateway account for an unallocated experiment.")
            current = None if row is None else row[0]
            if failed_index is not None:
                # A late response from the old account must not rotate the new
                # account a second time. Every process rereads this transaction.
                revision = connection.execute(
                    "SELECT credential_revision FROM gateway_accounts WHERE key_index=?", (failed_index,),
                ).fetchone()[0]
                if failed_revision is None and revision != 1:
                    raise GatewayConfigurationError("A failed credential revision is required after account maintenance.")
                if failed_revision in (None, revision):
                    changed = connection.execute(
                        "UPDATE gateway_accounts SET status=? WHERE key_index=? AND status='available'",
                        (reason, failed_index),
                    ).rowcount
                    if changed:
                        self._account_event(connection, failed_index, reason, self._configuration_revision)
            available = {
                index for (index,) in connection.execute(
                    "SELECT key_index FROM gateway_accounts WHERE status='available'",
                )
            }
            if current in available:
                selected = current
            elif available:
                cursor = connection.execute("SELECT next_index FROM gateway_pool WHERE id=1").fetchone()[0]
                start = cursor if current is None else current + 1
                selected = next(
                    index for offset in range(len(self._keys))
                    if (index := (start + offset) % len(self._keys)) in available
                )
                connection.execute(
                    "UPDATE gateway_pool SET next_index=? WHERE id=1", ((selected + 1) % len(self._keys),),
                )
            connection.execute(
                "INSERT INTO gateway_experiments VALUES (?, ?, ?) "
                "ON CONFLICT(experiment_id) DO UPDATE SET key_index=excluded.key_index, status=excluded.status",
                (experiment_id, selected if selected is not None else current, "active" if selected is not None else "blocked"),
            )
            selected_revision = None if selected is None else connection.execute(
                "SELECT credential_revision FROM gateway_accounts WHERE key_index=?", (selected,),
            ).fetchone()[0]
        if selected is None:
            raise GatewayUnavailableError(experiment_id, self.state_path)
        return self._chat_lease(selected, GATEWAY_BASE_URL, experiment_id, selected_revision)

    def _lease_official_bundle(self, experiment_id: str) -> GatewayLease:
        """Reserve both official routes for a preflight in one transaction."""
        with self._transaction() as connection:
            self._bind_experiment(connection, experiment_id)
            for provider in _OFFICIAL_URLS:
                self._ensure_official_state(connection, provider)
            unavailable = [provider for provider in _OFFICIAL_URLS if connection.execute(
                "SELECT 1 FROM gateway_provider_accounts WHERE provider=? AND status='available' LIMIT 1",
                (provider,),
            ).fetchone() is None]
            if unavailable:
                for provider in _OFFICIAL_URLS:
                    if provider in unavailable:
                        connection.execute(
                            "INSERT INTO gateway_provider_experiments VALUES (?, ?, NULL, 'blocked') "
                            "ON CONFLICT(provider,experiment_id) DO UPDATE SET status='blocked'",
                            (provider, experiment_id),
                        )
                    else:
                        connection.execute(
                            "INSERT INTO gateway_provider_experiments VALUES (?, ?, NULL, 'blocked') "
                            "ON CONFLICT(provider,experiment_id) DO NOTHING",
                            (provider, experiment_id),
                        )
                leases = None
            else:
                leases = {provider: self._allocate_official_tx(connection, provider, experiment_id)
                          for provider in _OFFICIAL_URLS}
        if leases is None or leases["dashscope"] is None or leases["deepseek"] is None:
            raise GatewayUnavailableError(experiment_id, self.state_path)
        return leases["dashscope"]

    def _allocate_official(
        self, provider: AccountProvider, experiment_id: str,
        failed_index: int | None = None, reason: CredentialFailure | None = None,
        failed_revision: int | None = None,
    ) -> GatewayLease:
        if self.profile != "official":
            raise GatewayConfigurationError("Official account allocation requires the official profile.")
        keys = self._official_keys(provider)
        if not keys:
            raise GatewayConfigurationError("The selected official chat pool has no configured accounts.")
        if failed_index is not None:
            if reason not in ("invalid", "exhausted"):
                raise GatewayConfigurationError("Only invalid or exhausted accounts can be rotated.")
            if type(failed_index) is not int or not 0 <= failed_index < len(keys):
                raise GatewayConfigurationError("The official account index is outside its configured pool.")
            if failed_revision is not None and (type(failed_revision) is not int or failed_revision < 1):
                raise GatewayConfigurationError("The failed credential revision is invalid.")
        with self._transaction() as connection:
            self._bind_experiment(connection, experiment_id)
            self._ensure_official_state(connection, provider)
            lease = self._allocate_official_tx(
                connection, provider, experiment_id, failed_index, reason, failed_revision,
            )
        if lease is None:
            raise GatewayUnavailableError(experiment_id, self.state_path)
        return lease

    def _allocate_official_tx(
        self, connection: sqlite3.Connection, provider: AccountProvider, experiment_id: str,
        failed_index: int | None = None, reason: CredentialFailure | None = None,
        failed_revision: int | None = None,
    ) -> GatewayLease | None:
        keys = self._official_keys(provider)
        row = connection.execute(
            "SELECT key_index FROM gateway_provider_experiments WHERE provider=? AND experiment_id=?",
            (provider, experiment_id),
        ).fetchone()
        if failed_index is not None and row is None:
            raise GatewayConfigurationError("Cannot rotate an unallocated official experiment account.")
        current = None if row is None else row[0]
        if failed_index is not None:
            revision = connection.execute(
                "SELECT credential_revision FROM gateway_provider_accounts WHERE provider=? AND key_index=?",
                (provider, failed_index),
            ).fetchone()[0]
            if failed_revision is None and revision != 1:
                raise GatewayConfigurationError("A failed credential revision is required after account maintenance.")
            if failed_revision in (None, revision):
                connection.execute(
                    "UPDATE gateway_provider_accounts SET status=? "
                    "WHERE provider=? AND key_index=? AND status='available'",
                    (reason, provider, failed_index),
                )
        available = {index for (index,) in connection.execute(
            "SELECT key_index FROM gateway_provider_accounts WHERE provider=? AND status='available'",
            (provider,),
        )}
        selected = None
        if current in available:
            selected = current
        elif available:
            cursor = connection.execute(
                "SELECT next_index FROM gateway_provider_pool WHERE provider=?", (provider,),
            ).fetchone()[0]
            start = cursor if current is None else current + 1
            selected = next(
                index for offset in range(len(keys))
                if (index := (start + offset) % len(keys)) in available
            )
            connection.execute(
                "UPDATE gateway_provider_pool SET next_index=? WHERE provider=?",
                ((selected + 1) % len(keys), provider),
            )
        connection.execute(
            "INSERT INTO gateway_provider_experiments VALUES (?, ?, ?, ?) "
            "ON CONFLICT(provider,experiment_id) DO UPDATE SET "
            "key_index=excluded.key_index, status=excluded.status",
            (provider, experiment_id, selected if selected is not None else current,
             "active" if selected is not None else "blocked"),
        )
        if selected is None:
            return None
        revision = connection.execute(
            "SELECT credential_revision FROM gateway_provider_accounts WHERE provider=? AND key_index=?",
            (provider, selected),
        ).fetchone()[0]
        return self._chat_lease(selected, _OFFICIAL_URLS[provider], experiment_id, revision, keys=keys)

    @staticmethod
    def _provider_event(
        connection: sqlite3.Connection, provider: str, index: int, action: str, configuration_revision: int,
    ) -> None:
        revision = connection.execute(
            "SELECT credential_revision FROM gateway_provider_accounts WHERE provider=? AND key_index=?",
            (provider, index),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO gateway_provider_account_events "
            "(provider, configuration_revision, key_index, credential_revision, action) "
            "VALUES (?, ?, ?, ?, ?)",
            (provider, configuration_revision, index, revision, action),
        )

    def _replace_official_key(self, provider: AccountProvider, index: int, secret: str) -> None:
        keys = self._official_keys(provider)
        if type(index) is not int or not 0 <= index < len(keys):
            raise GatewayConfigurationError("The official account index is outside its configured pool.")
        self._secret(secret)
        with self._transaction() as connection:
            self._ensure_official_state(connection, provider)
            if secret == keys[index]:
                return
            if secret in keys:
                raise GatewayConfigurationError("The credential already belongs to another official pool index.")
            updated = list(keys)
            updated[index] = secret
            official = {**self._config.get("official", {}), provider: {"keys": updated}}
            config = {**self._config, "official": official}
            snapshot = self._publish_configuration(connection, config)
            connection.execute(
                "UPDATE gateway_provider_pool SET fingerprint=? WHERE provider=?",
                (self._official_fingerprint(provider, updated), provider),
            )
            connection.execute(
                "UPDATE gateway_provider_accounts SET status='available', "
                "credential_revision=credential_revision+1 WHERE provider=? AND key_index=?",
                (provider, index),
            )
            self._provider_event(connection, provider, index, "replace", snapshot[1])
        self._adopt(config, *snapshot)

    def _add_official_key(self, provider: AccountProvider, secret: str) -> int:
        keys = self._official_keys(provider)
        self._secret(secret)
        with self._transaction() as connection:
            self._ensure_official_state(connection, provider)
            if secret in keys:
                raise GatewayConfigurationError("The credential already belongs to an official pool index.")
            index = len(keys)
            updated = [*keys, secret]
            official = {**self._config.get("official", {}), provider: {"keys": updated}}
            config = {**self._config, "official": official}
            snapshot = self._publish_configuration(connection, config)
            connection.execute(
                "UPDATE gateway_provider_pool SET fingerprint=? WHERE provider=?",
                (self._official_fingerprint(provider, updated), provider),
            )
            connection.execute(
                "INSERT INTO gateway_provider_accounts VALUES (?, ?, 'available', 1)",
                (provider, index),
            )
            self._provider_event(connection, provider, index, "add", snapshot[1])
        self._adopt(config, *snapshot)
        return index

    def _set_official_status(self, provider: AccountProvider, index: int, status: str) -> None:
        keys = self._official_keys(provider)
        if type(index) is not int or not 0 <= index < len(keys):
            raise GatewayConfigurationError("The official account index is outside its configured pool.")
        with self._transaction() as connection:
            self._ensure_official_state(connection, provider)
            changed = connection.execute(
                "UPDATE gateway_provider_accounts SET status=?, credential_revision=credential_revision+1 "
                "WHERE provider=? AND key_index=? AND status!=?",
                (status, provider, index, status),
            ).rowcount
            if changed:
                self._provider_event(connection, provider, index, status, self._configuration_revision)


def transport_identity(provider: ModelProvider, environ: Mapping[str, str] | None = None) -> str:
    """Load the private selected profile and return its nonsecret cache identity."""
    return GatewayPool.from_env(environ).transport_identity(provider)


def credential_failure(
    error: BaseException | None = None, *, status_code: int | None = None, body: Any = None,
) -> CredentialFailure | None:
    """Classify only affirmative account failures, without exposing the response.

    Args:
        error: An SDK/transport exception. Its status and structured body are read
            when explicit values are absent; it is never retained or logged.
        status_code: HTTP status, when known.
        body: Provider response body or error payload, when available.

    Returns:
        ``invalid`` for a refused credential or account: HTTP 401 or bare 403, rejected or
        expired authentication, denied access, a suspended or disabled account or key,
        or a model not activated for the account. ``exhausted`` for an account that
        cannot pay: HTTP 402, arrears, an insufficient balance, or spent credit or
        quota. ``None`` for plain throttling (including explicit per-minute quota
        wording without a separate account rejection), server errors
        (5xx) and unclassified failures. Account wording beyond the fixed provider
        codes counts only in a response with an HTTP status, never in local error
        text such as a file permission error.
    """
    seen: set[int] = set()
    current = error
    for _ in range(8):
        if current is not None:
            if id(current) in seen:
                break
            seen.add(id(current))
        status = status_code
        if status is None and current is not None:
            status = getattr(current, "status_code", None)
            if status is None:
                status = getattr(getattr(current, "response", None), "status_code", None)
        payload = body
        if payload is None and current is not None:
            payload = getattr(current, "body", None)
            if payload is None:
                payload = str(current)
        result = _classify_failure(status, payload)
        if result is not None:
            return result
        if current is None or status is not None and status >= 500:
            break
        current = current.__cause__ or current.__context__
        if current is None:
            break
        status_code, body = None, None
    return None


#: Explicit short-window throughput wording (see ``_classify_failure``).
_THROTTLE_WINDOW = re.compile(
    r"\b(?:tpm|rpm|qps|tps|rps)\b|\bper[ -]?(?:minute|second)\b"
    r"|\b(?:tokens?|requests?|queries?)/(?:min(?:ute)?|sec(?:ond)?)\b"
    r"|\b(?:minute|second)[ -](?:window|limit)\b|每分钟|每秒"
)
#: Billing wording remains an account failure even alongside throughput diagnostics.
_ACCOUNT_BILLING = re.compile(
    r"overdue|arrear|欠费|余额不足|payment[\s_-]*required"
    r"|insufficient[\s_-]*(?:account[\s_-]*)?(?:balance|funds|credits?)"
    r"|(?:credits?|balance|funds)[^\n.;]{0,40}?(?:insufficient|exhausted|depleted|used up|run out)"
    r"|(?:exhausted|depleted)[\s_-]*(?:credits?|balance|funds)"
)
_ACCOUNT_QUOTA = re.compile(
    r"insufficient[\s_-]*quota|(?:quota|allowance)[^\n.;]{0,40}?(?:exhausted|depleted|used up|run out)"
    r"|(?:exhausted|depleted)[\s_-]*quota"
    r"|(?:额度|配额)已?(?:用尽|用完|耗尽|不足)"
    # A daily (or user-set) allowance that resets: "daily quota exceeded", "exceeded your daily limit",
    # "today's quota has been reached", 当日/今日/每日 额度 已达上限.
    r"|\b(?:daily|today'?s|per[\s_-]*day)[\s_-]*(?:free[\s_-]*)?(?:quota|limit|allowance|usage|tokens?)"
    r"[^\n.;]{0,40}?(?:exceed|reach|hit|exhaust|deplet|used up|run out)"
    r"|(?:exceed|reach|hit)(?:ed|es)?[\s_-]*(?:the|your)?[\s_-]*(?:daily|today'?s)[\s_-]*(?:free[\s_-]*)?(?:quota|limit|allowance)"
    r"|(?:当日|今日|每日|日)(?:免费)?(?:额度|配额|限额|用量)[^\n。；]{0,12}?(?:用尽|用完|耗尽|超出|超过|达到上限|已达上限)"
)
#: Refusal wording of a provider answer: denied access, a suspended or disabled account or
#: key, or a model or service not activated or purchased for the account.
_ACCOUNT_DENIED = re.compile(
    r"access[^\n.;]{0,30}?(?:denied|forbidden)|permission[\s_-]*denied|\bforbidden\b"
    r"|(?:account|consumer)[^\n.;]{0,40}?(?:suspended|disabled|deactivated|frozen|banned|locked|terminated|forbidden)"
    r"|(?:suspended|disabled|deactivated)[\s_-]*(?:account|consumer)"
    r"|(?:api|access|secret)[\s_-]*key[^\n.;]{0,30}?(?:disabled|revoked|deactivated|suspended)"
    r"|not[\s_-]*(?:been[\s_-]*)?(?:activated|purchased|subscribed)|unpurchased"
    r"|(?:model|service)[\s_-]*not[\s_-]*open"
    r"|(?:账户|账号|帐户|帐号)[^\n。；]{0,12}?(?:冻结|停用|禁用|注销|停机)|未开通|未激活|拒绝访问|无权访问",
    re.MULTILINE,
)
#: Statuses that name an account-level rejection by themselves; 401 is classified first.
_ACCOUNT_STATUSES: Mapping[int, CredentialFailure] = MappingProxyType({402: "exhausted", 403: "invalid"})


def _classify_failure(status_code: int | None, body: Any) -> CredentialFailure | None:
    if status_code == 401:
        return "invalid"
    if status_code is not None and status_code >= 500:
        return None
    values: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            values.append(value.lower())
            if value.lstrip().startswith(("{", "[")):
                try:
                    decoded = json.loads(value)
                except ValueError:
                    pass
                else:
                    if isinstance(decoded, (dict, list)):
                        collect(decoded)
        elif isinstance(value, dict):
            for key in ("error", "code", "type", "message", "detail"):
                if key in value:
                    collect(value[key])
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(body)
    normalized = {re.sub(r"[^a-z]", "", value) for value in values}
    if normalized & {
        "invalidapikey", "invalidaccesskey", "authenticationerror", "authenticationfailed",
        "apikeyexpired", "keyexpired", "invalidtoken",
    }:
        return "invalid"
    text = "\n".join(values)
    if any(phrase in text for phrase in (
        "invalid api key", "incorrect api key", "api key is invalid", "api key has expired",
        "api key 无效", "无效的 api key",
    )):
        return "invalid"
    # Local errors such as Windows "Access is denied" name no provider account.
    answered = status_code is not None
    if answered and _ACCOUNT_BILLING.search(text):
        return "exhausted"
    if answered and _ACCOUNT_DENIED.search(text):
        return "invalid"
    # Some gateways use quota-shaped codes for a short-lived throughput bucket.
    # Explicit TPM/RPM/QPS evidence wins over those codes; retiring every account
    # would turn an ordinary refill delay into a persistent pool outage.
    if _THROTTLE_WINDOW.search(text):
        return _ACCOUNT_STATUSES.get(status_code)
    if normalized & {
        "insufficientquota", "insufficientbalance", "balanceinsufficient", "accountoverdue",
        "quotaexhausted", "billinghardlimitreached", "creditbalanceinsufficient",
        "accountbalanceinsufficient", "dailyquotaexceeded", "dailylimitexceeded", "setlimitexceeded",
    } or any(phrase in text for phrase in (
        "insufficient balance", "insufficient credit", "credit balance is too low",
        "account balance is insufficient", "account is overdue", "billing hard limit",
        "quota has been exhausted", "quota is exhausted",
        "check your plan and billing details", "余额不足", "账户欠费", "额度已用尽", "配额已耗尽",
    )) or answered and _ACCOUNT_QUOTA.search(text):
        return "exhausted"
    return _ACCOUNT_STATUSES.get(status_code)
