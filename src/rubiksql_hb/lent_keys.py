"""The owner's closed, time-limited Volcano file pool. No credential is serialized here."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import logging
import math
import os
from pathlib import Path
import random
import time
from typing import Any

from .key_broker import BROKER_STATE_ENV, QUOTA_TZ, HostCounter, SqliteKeyBroker, file_key_id, host_counter

KEY_DIR_ENV = "RUBIKSQL_LENT_KEY_DIR"
KEY_NAMES_ENV = "RUBIKSQL_LENT_KEY_NAMES"


def configured_key_dir() -> Path:
    """Return the explicit file-pool directory, defaulting to the current usable pool.

    Args:
        None.

    Returns:
        Path: Expanded absolute directory; no directory enumeration occurs.
    """
    return Path(os.environ.get(KEY_DIR_ENV) or Path.home() / ".config/rubiksql-hb/keys/volcano-usable-20261111").expanduser().absolute()


def valid_key_name(name: Any) -> bool:
    """Validate a key basename without allowing paths or control characters.

    Args:
        name (Any): Candidate filename from configuration or persisted marks.

    Returns:
        bool: True for a printable ASCII basename ending in `.key`.
    """
    return (isinstance(name, str) and name.endswith(".key") and len(name) > 4
            and name.isascii() and name.isprintable() and not any(c.isspace() for c in name)
            and not any(c in name for c in '/\\:') and name not in (".", ".."))


def configured_key_names(key_dir: Path | None = None) -> tuple[str, ...]:
    """Resolve an ordered filename override, or sorted key files in the selected directory.

    Args:
        key_dir: Explicit directory, otherwise :func:`configured_key_dir`.

    Returns:
        tuple[str, ...]: Distinct validated basenames in round-robin order.

    Raises:
        GatewayConfigurationError: The list is empty, duplicated or unsafe, or discovery fails.
    """
    raw = os.environ.get(KEY_NAMES_ENV)
    try:
        names = (tuple(name.strip() for name in raw.split(",")) if raw is not None else
                 tuple(sorted(path.name for path in (key_dir or configured_key_dir()).expanduser().glob("*.key")
                              if path.is_file())))
    except OSError:
        raise _error("Cannot enumerate the lent key directory; contents withheld.") from None
    if not names or len(set(names)) != len(names) or not all(valid_key_name(name) for name in names):
        raise _error("The lent key names must be distinct .key basenames in request order.")
    return names


#: Optional window end of the lent pool (ISO-8601 with a UTC offset); unset means no end. The cap controller
#: (``scripts/g28_capbal_host.py``) reads the same variable.
WINDOW_END_ENV = "RUBIKSQL_LENT_WINDOW_END"
WINDOW_END_FORMAT = "YYYY-MM-DDTHH:MM+HH:MM"
CAP_ENV = "RUBIKSQL_KEY_MAX_INFLIGHT"
CAP_FILE_ENV = "RUBIKSQL_KEY_CAP_FILE"
_LOGGER = logging.getLogger(__name__)


def _error(message: str) -> RuntimeError:
    from .experiment_gateway import GatewayConfigurationError
    return GatewayConfigurationError(message)


def parse_window_end(value: str) -> float:
    """Parse an ISO-8601 window end that names its UTC offset (:data:`WINDOW_END_FORMAT`).

    Args:
        value: The timestamp text; a trailing ``Z`` means UTC.

    Returns:
        The window end as a Unix timestamp.

    Raises:
        GatewayConfigurationError: The text is not ISO-8601 or carries no offset.
    """
    text = value.strip() if isinstance(value, str) else ""
    try:
        moment = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text)
    except ValueError:
        moment = None
    if moment is None or moment.utcoffset() is None:
        raise _error(f"{WINDOW_END_ENV} must be ISO-8601 with a UTC offset ({WINDOW_END_FORMAT}).")
    return moment.timestamp()


def window_end(environ: Mapping[str, str] | None = None) -> float | None:
    """The lent-key window end from :data:`WINDOW_END_ENV` when set and non-empty, else None (no end)."""
    env = os.environ if environ is None else environ
    raw = env.get(WINDOW_END_ENV, "")
    return parse_window_end(raw) if raw.strip() else None


def window_label(environ: Mapping[str, str] | None = None) -> str:
    """The configured window end for status output: its ISO time, ``none`` when unset, ``invalid`` when unusable."""
    try:
        end = window_end(environ)
    except RuntimeError:
        return "invalid"
    return "none" if end is None else datetime.fromtimestamp(end, QUOTA_TZ).isoformat()


def is_key_path(path: Path) -> bool:
    """Whether a path is inside an owner key directory, including through an output symlink."""
    expanded = path.expanduser().absolute()
    protected = (configured_key_dir(), Path.home() / ".config/aml-keys/volcano")
    return any(candidate == root or root in candidate.parents
               for candidate in (expanded, expanded.resolve())
               for directory in protected for root in (directory, directory.resolve())) or any(
        tuple(parent.parts[-3:]) == (".config", "aml-keys", "volcano")
        for candidate in (expanded, expanded.resolve()) for parent in candidate.parents)


def validate_files(value: Any) -> tuple[Path, ...]:
    """Validate an explicit ordered file manifest without reading any credentials.

    Args:
        value (Any): Nonempty JSON list of absolute key paths in one directory.

    Returns:
        tuple[Path, ...]: Distinct paths; symlinks and unsafe basenames are refused.

    Raises:
        GatewayConfigurationError: The file manifest or directory layout is unsafe.
    """
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise _error("The lent pool requires a nonempty ordered key file manifest.")
    paths = tuple(Path(item).expanduser() for item in value)
    parent = paths[0].parent
    if (len(set(paths)) != len(paths) or any(not valid_key_name(path.name) for path in paths)
            or any(not path.is_absolute() or path.parent != parent or path.is_symlink() for path in paths)
            or any(part.is_symlink() for part in (parent, *parent.parents))):
        raise _error("The lent pool file paths violate the approved directory and allow-list.")
    return paths


def make_pool(source: Path, destination: Path, key_dir: Path) -> None:
    """Write a nonsecret pool pointing to approved keys and the existing embedding configuration.

    Args:
        source: Existing private pool; only approved endpoint/profile and routing fields are copied.
        destination: New nonsecret JSON file; must differ from source and all key paths.
        key_dir: Explicit host directory containing the configured key basenames.

    Raises:
        GatewayConfigurationError: Source route or destination is unsafe.
    """
    from .experiment_gateway import GATEWAY_BASE_URL, GatewayPool
    directory = key_dir.expanduser().absolute().resolve()
    if (is_key_path(source) or is_key_path(destination)
            or any(path.resolve() == directory or directory in path.resolve().parents for path in (source, destination))):
        raise _error("Pool source and output must be outside the lent key directory.")
    source = GatewayPool(source).configuration_path
    try:
        original = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise _error("Cannot read source gateway routing configuration.") from None
    if (not isinstance(original, dict) or "key_files" in original
            or original.get("base_url") != GATEWAY_BASE_URL or original.get("profile", "volcengine") != "volcengine"):
        raise _error("The lent pool requires the approved Volcengine route.")
    paths = validate_files([str(key_dir.expanduser().absolute() / name) for name in configured_key_names(key_dir)])
    if destination.resolve() == source.resolve() or destination.resolve() in [path.resolve() for path in paths] or destination.parent.resolve() == paths[0].parent.resolve():
        raise _error("The generated pool must be separate from credentials and key files.")
    # Embedding credentials stay in their original private file, including its managed snapshot.
    config = {"base_url": original["base_url"], "profile": "volcengine", "key_files": [str(path) for path in paths]}
    if "embedding" in original:
        config["embedding_source"] = str(source.resolve())
    if "chat_limits" in original:
        config["chat_limits"] = original["chat_limits"]
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with destination.open("x", encoding="utf-8") as output:
        json.dump(config, output, indent=2)
        output.write("\n")


def retry_delay(headers: Any, now: float | None = None) -> float:
    """Parse Retry-After seconds or an HTTP date; otherwise use the owner's uniform 10–30 seconds."""
    value = (headers or {}).get("Retry-After")
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(value)
            seconds = date.replace(tzinfo=date.tzinfo or timezone.utc).timestamp() - (time.time() if now is None else now)
        except (TypeError, ValueError, OverflowError):
            seconds = float("nan")
    # Zero/negative delays must still avoid an immediate retry.
    return max(1.0, seconds) if math.isfinite(seconds) else random.uniform(10.0, 30.0)


def read_host_cap(cap_file: Path, default: str, *, maximum: int = 60) -> int:
    """Read the host's per-key cap: the live cap file when it exists, else ``default`` (``RUBIKSQL_KEY_MAX_INFLIGHT``).

    Args:
        cap_file: The live cap file (``RUBIKSQL_KEY_CAP_FILE``, or the broker path with a ``.cap`` suffix).
        default: The cap used while no cap file exists.
        maximum: Owner's explicit ceiling; direct broker clients retain 60.

    Returns:
        The cap, 0–maximum (0 pauses admission).

    Raises:
        GatewayConfigurationError: The value exceeds the owner's ceiling, or the file cannot be read.
    """
    try:
        value = cap_file.read_text(encoding="utf-8").strip() if cap_file.exists() else default
        cap = int(value)
        if not 0 <= cap <= maximum:
            raise ValueError
        return cap
    except (OSError, ValueError):
        raise _error(f"File-key host cap must be an integer from 0 to {maximum}.") from None


def read_key_file(paths: tuple[Path, ...], name: str) -> str:
    """Read one approved key file's bearer key, revalidating the whole allow-list first.

    Args:
        paths: All validated owner-approved file references (:func:`validate_files`).
        name: A basename from the explicit ordered file manifest.

    Returns:
        The key, held only in memory by the caller; it is never rendered.

    Raises:
        GatewayConfigurationError: The allow-list or directory changed, the file is missing or a symlink, or its
            contents are not one printable ASCII token. The message names neither the contents nor the key.
    """
    try:
        path = next(path for path in paths if path.name == name)
        # Revalidate before each read; O_NOFOLLOW also closes a final-component symlink race on Linux.
        validate_files([str(item) for item in paths])
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "r", encoding="utf-8-sig") as source:
            secret = source.read(4097).strip()
        if (not secret or len(secret) > 4096 or not secret.isascii() or not secret.isprintable()
                or any(character.isspace() for character in secret)):
            raise ValueError
    except (OSError, ValueError, RuntimeError, StopIteration):
        raise _error("Cannot load an approved lent key file; contents withheld.") from None
    from .chat_keys import DeniedChatKey, denylist, refused
    if refused(secret, path.name, denylist()):
        raise DeniedChatKey("A denylisted chat key is retired; contents withheld.")
    return secret


@dataclass(frozen=True)
class FileLease:
    """Own one physical request's broker reservation; never render its credential."""
    token: str
    file_name: str
    secret: str = field(repr=False)


class LentKeyPool:
    """Resolve live host caps and reserve secrets only for physical requests.

    Args:
        paths: All validated owner-approved file references.

    The optional window end is read once, from :data:`WINDOW_END_ENV`: unset means no end, an invalid value
    refuses the pool, and a process keeps the end it read at start.
    """

    def __init__(self, paths: tuple[Path, ...]) -> None:
        self.paths = validate_files([str(path) for path in paths])
        self.names = tuple(path.name for path in self.paths)
        default = paths[0].parent.parent / "volcano-host-broker.sqlite3"
        self.broker = SqliteKeyBroker.for_path(Path(os.environ.get(BROKER_STATE_ENV) or default).expanduser())
        self.cap_file = Path(os.environ.get(CAP_FILE_ENV) or self.broker.path.with_suffix(".cap")).expanduser()
        if any(is_key_path(path) or path.resolve() == self.paths[0].parent.resolve()
               or self.paths[0].parent.resolve() in path.resolve().parents
               for path in (self.broker.path, self.cap_file)):
            raise _error("Broker state and cap files must be outside the lent key directory.")
        self.default_cap = os.environ.get(CAP_ENV, "1")
        self.ends_at = window_end()
        self._last_notice = float("-inf")
        from .keyproxy import Endpoint
        #: The host's local key proxy (``RUBIKSQL_KEYPROXY_URL``), or None: then every request leases from the broker.
        self.proxy = Endpoint.from_env(paths[0].parent)
        if self.proxy is None:
            from .chat_keys import DeniedChatKey
            marks = []
            retired = {row["file_name"] for row in self.broker.stats()["keys"] if row["retired"]}
            for name in self.names:
                if name in retired:
                    continue
                try:
                    read_key_file(self.paths, name)
                except DeniedChatKey:
                    marks.append({"key_id": file_key_id(name), "file_name": name, "exhausted_until": 0, "retired": 1})
            if marks:
                self.broker.import_marks(marks)

    def check_window(self) -> None:
        """Refuse admission and sending at or after the configured window end; without one, never."""
        if self.ends_at is not None and time.time() >= self.ends_at:
            end = datetime.fromtimestamp(self.ends_at, QUOTA_TZ).isoformat()
            raise _error(f"Lent Volcano key window ended at {end}; provider calls stopped.")

    def cap(self) -> int:
        """Read the cap file on every admission (0 pauses); fail closed on invalid configuration."""
        return read_host_cap(self.cap_file, self.default_cap)

    def try_acquire(self, *, ttl: float) -> FileLease | None:
        """Attempt one atomic admission; a caller sleeps between unsuccessful attempts.

        Args:
            ttl: Request absolute deadline plus crash-recovery margin, in seconds.

        Returns:
            A reservation with its runtime-only credential, or None while waiting.
        """
        self.check_window()
        chosen = self.broker.acquire(self.names, cap=self.cap(), ttl=ttl)
        if chosen is None:
            rows = [row for row in self.broker.stats()["keys"] if row["file_name"] in self.names]
            if len(rows) == len(self.names) and all(row["retired"] for row in rows):
                raise _error("Every lent chat key is retired; no usable credential remains.")
            if time.monotonic() - self._last_notice >= 60:
                self._last_notice = time.monotonic()
                if self.broker.stall_notice():
                    _LOGGER.warning("[keypool-stalled] Lent Volcano pool waiting: broker busy or keys capped, cooling, exhausted or retired; no answer fabricated.")
            return None
        token, name = chosen
        try:
            secret = read_key_file(self.paths, name)
            self.check_window()
            return FileLease(token, name, secret)
        except RuntimeError as error:
            from .chat_keys import DeniedChatKey
            self.broker.finish(token, None)
            if isinstance(error, DeniedChatKey):
                self.broker.import_marks([{"key_id": file_key_id(name), "file_name": name,
                                           "exhausted_until": 0, "retired": 1}])
                return self.try_acquire(ttl=ttl)
            raise

    @property
    def waiting(self) -> HostCounter:
        """This process's count of admissions waiting for a slot, published to the pool's host broker."""
        return host_counter(self.broker, "waiting")

    def acquire(self, *, ttl: float) -> FileLease:
        """Wait without spinning for admission; caller deadlines and any window end remain effective."""
        from .llm_runtime import refuse_if_stopped
        waiting = False
        try:
            while True:
                refuse_if_stopped()
                lease = self.try_acquire(ttl=ttl)
                if lease is not None:
                    return lease
                if not waiting:
                    waiting = True
                    self.waiting.change(+1)
                time.sleep(random.uniform(0.5, 1.5))
        finally:
            if waiting:
                self.waiting.change(-1)

    def backoff(self, seconds: float) -> None:
        """Wait the full retry delay, checking the caller's stop flag and any window end each second."""
        from .llm_runtime import refuse_if_stopped
        while seconds > 0:
            refuse_if_stopped()
            self.check_window()
            chunk = min(seconds, 1.0)
            time.sleep(chunk)
            seconds -= chunk

    async def abackoff(self, seconds: float) -> None:
        """Apply the same full backoff without blocking the event loop or suppressing cancellation."""
        import asyncio
        from .llm_runtime import refuse_if_stopped
        while seconds > 0:
            refuse_if_stopped()
            self.check_window()
            chunk = min(seconds, 1.0)
            await asyncio.sleep(chunk)
            seconds -= chunk
