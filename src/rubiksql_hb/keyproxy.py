"""One process that owns the lent Volcano keys, forwards chat requests with them and admits questions globally.

Without the proxy every physical request leases its key from the host broker (:mod:`.key_broker`): two forced-sync
SQLite write transactions on one host-wide lock that every eval process shares, which under load cost seconds of
admission and release per request, and each host balanced its share of the keys separately. The proxy keeps the same
owner rules in one asyncio process's memory instead, and one proxy serves every runner of an experiment fleet:

* Clients post OpenAI-compatible chat requests to ``http://<proxy>:<port>/v1/chat/completions`` with the bearer token
  (``<state>/token``, mode 600, copied to the client runners). The proxy picks the next eligible key in the owner's
  fixed order (strict round robin), sends the body bytes unchanged to the approved Volcengine gateway with that key,
  and returns the provider's status, body, ``Content-Type`` and ``Retry-After`` unchanged. Only that gateway is ever
  called: no other upstream exists.
* A key is eligible while it is neither retired, paused nor cooling and has fewer than ``per_key`` requests in flight.
  ``per_key`` is re-read live from the cap file (an integer 0-70; 0 pauses new sends); one proxy for every host makes
  it the global per-key limit.
* HTTP 429 cools the key for ``Retry-After`` (uniform 10-30 s when absent), doubling with each consecutive 429 episode
  of that key up to 120 s, plus up to 10 % jitter, and never shorter than ``Retry-After``; the request moves on to the
  next eligible key. HTTP 403 pauses the key until the next 00:05 Asia/Shanghai after the request was sent. HTTP 401
  retires it permanently once :data:`BREAKER_WINDOW_S` passed without 401s from :data:`BREAKER_KEYS` distinct keys
  (the key is paused meanwhile); 401s from that many keys at once are a gateway fault, not invalid keys, and pause
  the keys for :data:`BREAKER_PAUSE_S` instead. Previously retired files are never opened;
  hash-denied keys become retired slots before any request. Cooling episodes, pauses and retirements are appended to ``<state>/marks.jsonl`` (fsync) and
  restored at every start.
* A request without an eligible key waits in one first-come first-served queue that releases wake. Its waiting budget
  is the request header ``x-rubiksql-keyproxy-wait-ms`` (at most ``max_wait``); upstream time never counts against it.
  Past the budget, after :data:`MAX_ATTEMPTS` physical sends, when every key is retired, or at once when every usable
  key is paused beyond the budget, the proxy answers 503 with a JSON ``error.code`` and ``Retry-After``. It never
  fabricates an answer.
* A client that disconnects while waiting or while its request is upstream cancels that request and frees its slot.
  Accepted connections use TCP keepalive and upstream reads time out after :data:`UPSTREAM_READ_S`, so a client or
  gateway cut off without a FIN cannot hold a slot for the rest of a run.
* A shutdown (``stop``, SIGTERM or SIGINT) refuses queued and new requests but keeps listening while in-flight
  requests drain (``/keyproxy/health`` says ``closing``); the process exits once nothing is in flight or ``--drain``
  seconds passed, so "stopped" means no paid request is still upstream.
* The question-start gate's reservations live here too (:mod:`.question_book`, ``/question/reserve|phase|release``),
  so admission rotates across waiting owners, against the pool's live capacity.
* Every upstream attempt appends one row to ``<state>/ledger/<Asia/Shanghai date>.jsonl``: key file name, status,
  waiting and upstream milliseconds, requested and served model, token usage and, for refusals, the sanitized provider
  error (:func:`.provider_errors.event`). No key value is ever logged, written or passed on a command line.

Run it on one runner as the user that owns ``~/.config/rubiksql-hb/keys/volcano-usable-20261111``, bound to its private address (or loopback)::

    python -m rubiksql_hb.keyproxy start --host 172.16.0.10   # supervised and detached; returns once health answers
    export RUBIKSQL_KEYPROXY_URL=http://172.16.0.10:18471 RUBIKSQL_KEYPROXY_TOKEN_FILE=<copy of the token>
    python -m rubiksql_hb.keyproxy status
    python -m rubiksql_hb.keyproxy stop                       # refuses queued requests, drains in-flight ones, exits

One proxy owns one state directory: ``serve`` holds ``<state>/keyproxy.lock`` for its whole life, so a second proxy
on the same state never starts beside one that is still draining. ``start`` runs ``serve`` under a supervisor that
restarts it after a crash (never after a graceful stop or a failed start).

With ``RUBIKSQL_KEYPROXY_URL`` set, the lent pool's chat requests (:mod:`.lent_chat`, :mod:`.gateway_chat`) go to the
proxy and never lease from the SQLite broker; an unreachable proxy is retried for ``RUBIKSQL_KEYPROXY_CONNECT_SEC``
and then fails the request (never a key outside the proxy). The question-start gate reserves through the proxy.
Embeddings stay on DashScope and never pass through the proxy.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import errno
import functools
import hashlib
import ipaddress
import json
import logging
import os
from pathlib import Path
import math
import random
import secrets
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

import httpx

from . import provider_errors
from .key_broker import QUOTA_TZ
from .question_book import WARMUP_S, QuestionBook, QuestionError
from .lent_keys import configured_key_dir, configured_key_names, valid_key_name, read_host_cap, read_key_file, retry_delay, validate_files, window_end

_LOGGER = logging.getLogger(__name__)

#: Base URL of the key proxy (``http://<loopback or private IP>:<port>``). When set, lent-pool chat and the
#: question-start gate go through the proxy.
URL_ENV = "RUBIKSQL_KEYPROXY_URL"
#: Optional proxy state directory; default :func:`default_state_dir` beside the lent key directory.
STATE_ENV = "RUBIKSQL_KEYPROXY_STATE"
#: Optional path of the bearer token file; default ``<state>/token``. Client runners name their copy here.
TOKEN_FILE_ENV = "RUBIKSQL_KEYPROXY_TOKEN_FILE"
#: Seconds a client lets one request wait inside the proxy for a key (default :data:`DEFAULT_WAIT_S`).
WAIT_ENV = "RUBIKSQL_KEYPROXY_WAIT_SEC"
#: Seconds a client keeps retrying a proxy it cannot connect to before the request fails (default
#: :data:`DEFAULT_CONNECT_S`). The request never falls back to a key outside the proxy.
CONNECT_ENV = "RUBIKSQL_KEYPROXY_CONNECT_SEC"
#: Request header: the in-proxy waiting budget in milliseconds. Response header: the milliseconds actually waited.
WAIT_HEADER = "x-rubiksql-keyproxy-wait-ms"
#: Response header: the approved file name of the key that answered (never its value).
KEY_HEADER = "x-rubiksql-keyproxy-key"
#: Response header: physical sends this request took (1 unless a 401/403/429 moved it to another key).
ATTEMPTS_HEADER = "x-rubiksql-keyproxy-attempts"
#: Request header: the requested model, so the proxy need not parse the body (it falls back to parsing).
MODEL_HEADER = "x-rubiksql-keyproxy-model"
DEFAULT_PORT = 18471
DEFAULT_WAIT_S = 600.0
DEFAULT_CONNECT_S = 120.0
DEFAULT_MAX_WAIT_S = 900.0
#: Seconds a shutdown waits for in-flight requests (``--drain``).
DEFAULT_DRAIN_S = 900.0
#: Per-key concurrency when neither ``--per-key`` nor ``RUBIKSQL_KEY_MAX_INFLIGHT`` is given: one host owns the keys.
DEFAULT_PER_KEY = 60
MAX_PER_KEY = 70
#: Physical sends of one request, as the lent SDK path allows (one invalid/quota sweep plus retries).
MAX_ATTEMPTS = 43
#: 429 cooling: base Retry-After (or uniform 10-30 s) doubled per consecutive episode, capped here.
MAX_COOLING_S = 120.0
#: How often the cap file is re-read and expired cooling/pauses re-checked for queued requests.
TICK_S = 0.5
#: Recent waiting times kept for the percentile summary of ``/keyproxy/stats``.
RECENT_WAITS = 5000
#: 401 breaker: 401s from this many distinct keys within :data:`BREAKER_WINDOW_S` are a gateway fault, not invalid
#: keys; nothing is retired and the keys pause for :data:`BREAKER_PAUSE_S`. A lone 401 retires its key once the
#: window passed without the breaker tripping (the key is paused meanwhile).
BREAKER_KEYS = 3
BREAKER_WINDOW_S = 60.0
BREAKER_PAUSE_S = 600.0
#: Longest silence on an upstream connection while its answer is awaited (non-streaming answers arrive whole); above
#: any chat deadline, it only frees slots held by a gateway connection that died without a FIN.
UPSTREAM_READ_S = 1800.0
#: A reused upstream connection that the gateway closed fails at once; such a failure within this many seconds of
#: sending is retried once on a fresh connection (the request never reached the provider).
STALE_RETRY_S = 2.0
#: Tokens shorter than this are refused (an empty token file must never authorize ``Bearer ``).
MIN_TOKEN_CHARS = 32
#: The lifetime lock ``serve`` holds in its state directory.
LOCK_NAME = "keyproxy.lock"
#: The supervisor restarts a ``serve`` that served at least this long and then died; a quicker exit is a failed start.
SUPERVISE_MIN_UP_S = 10.0
#: Crash restarts allowed within :data:`SUPERVISE_WINDOW_S` before the supervisor gives up.
SUPERVISE_RESTARTS = 5
SUPERVISE_WINDOW_S = 600.0
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_HTTP_LOCK = threading.Lock()


def private_host(host: str | None) -> bool:
    """Whether a proxy may listen on, or be reached at, ``host``: loopback or a private (VPC) IP literal.

    Public addresses, the unspecified ``0.0.0.0``/``::`` and DNS names other than ``localhost`` are refused, so the
    proxy is never exposed beyond the fleet's private network.
    """
    if host in _LOOPBACK:
        return True
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    return (address.is_loopback or address.is_private) and not (address.is_unspecified or address.is_link_local
                                                                or address.is_multicast)


def origin(url: str) -> str:
    """Validate a proxy URL before any token is sent to it: a plain ``http://host:port`` origin on a loopback or
    private IP (:func:`private_host`).

    Raises:
        GatewayConfigurationError: Anything else (a public or named host, a path, credentials, https).
    """
    raw = url.strip().rstrip("/")
    try:
        parsed = urllib.parse.urlsplit(raw)
        port = parsed.port
    except ValueError:
        parsed, port = None, None
    if (parsed is None or parsed.scheme != "http" or not private_host(parsed.hostname) or port is None
            or parsed.path or parsed.query or parsed.fragment or parsed.username or parsed.password):
        raise _error(f"{URL_ENV} must be an origin on a loopback or private IP, such as "
                     f"http://127.0.0.1:{DEFAULT_PORT}.")
    return raw


def default_state_dir(key_dir: Path) -> Path:
    """The proxy state directory beside, never inside, the lent key directory: ``<key dir>/../rubiksql-keyproxy``."""
    return key_dir.parent / "rubiksql-keyproxy"


def quota_day(now: float) -> str:
    """The Asia/Shanghai date of a POSIX time, the ledger's file name and the providers' quota day."""
    return datetime.fromtimestamp(now, QUOTA_TZ).date().isoformat()


def next_quota_reset(now: float) -> float:
    """The next 00:05 Asia/Shanghai after ``now``: when a key paused by a 403 is tried again."""
    local = datetime.fromtimestamp(now, QUOTA_TZ)
    reset = local.replace(hour=0, minute=5, second=0, microsecond=0)
    return (reset if reset.timestamp() > now else reset + timedelta(days=1)).timestamp()


def cooling_seconds(retry_after: str | None, strikes: int) -> float:
    """How long a 429 cools its key.

    Args:
        retry_after: The provider's ``Retry-After`` (seconds or an HTTP date), or None.
        strikes: The key's earlier consecutive 429 episodes (0 for the first).

    Returns:
        ``min(120, base * 2**strikes * jitter)`` with ``base`` the Retry-After (at least 1 s) or uniform 10-30 s and
        ``jitter`` uniform 1.0-1.1, and never less than a given Retry-After.
    """
    base = retry_delay({"Retry-After": retry_after} if retry_after is not None else {})
    delay = min(MAX_COOLING_S, base * 2 ** min(strikes, 8) * random.uniform(1.0, 1.1))
    return max(delay, base) if retry_after is not None else delay


# ----------------------------------------------------------------------------------------------- client side


@dataclass(frozen=True)
class Endpoint:
    """A client's view of the key proxy: where it listens and how to authenticate.

    Args:
        url: The proxy's origin, e.g. ``http://127.0.0.1:18471`` or ``http://172.16.0.10:18471``.
        token_file: The bearer token file the proxy wrote at its first start (or a client runner's copy).
        wait: Seconds a request may wait inside the proxy for a key (sent as :data:`WAIT_HEADER`).
        connect: Seconds a request keeps retrying a proxy it cannot connect to before it fails.
    """

    url: str
    token_file: Path
    wait: float = DEFAULT_WAIT_S
    connect: float = DEFAULT_CONNECT_S

    @classmethod
    def from_env(cls, key_dir: Path, environ: Mapping[str, str] | None = None) -> Endpoint | None:
        """The configured proxy, or None when :data:`URL_ENV` is unset or empty.

        Args:
            key_dir: The lent key directory; locates the default state directory (and token file).
            environ: Environment to read; the process environment by default.

        Raises:
            GatewayConfigurationError: The URL is not a plain ``http://host:port`` origin on a loopback or private
                IP (:func:`private_host`), or a budget is not a positive number.
        """
        env = os.environ if environ is None else environ
        raw = (env.get(URL_ENV) or "").strip().rstrip("/")
        if not raw:
            return None
        raw = origin(raw)
        budgets = []
        for name, default in ((WAIT_ENV, DEFAULT_WAIT_S), (CONNECT_ENV, DEFAULT_CONNECT_S)):
            try:
                value = float(env.get(name) or default)
            except ValueError:
                value = -1.0
            if not 0 < value < float("inf"):
                raise _error(f"{name} must be a positive number of seconds.")
            budgets.append(value)
        state = Path(env.get(STATE_ENV) or default_state_dir(key_dir)).expanduser()
        return cls(raw, Path(env.get(TOKEN_FILE_ENV) or state / "token").expanduser(), *budgets)

    @property
    def chat_url(self) -> str:
        """The proxy's chat completions route."""
        return self.url + "/v1/chat/completions"

    @functools.cached_property
    def token(self) -> str:
        """The local bearer token, read once from :attr:`token_file`.

        Raises:
            GatewayConfigurationError: The file is missing, empty or too short (start the proxy first, or copy its
                token to this runner and name the copy in :data:`TOKEN_FILE_ENV`), or on POSIX it is readable by
                the group or others (``chmod 600`` it).
        """
        try:
            value = self.token_file.read_text(encoding="utf-8").strip()
            mode = self.token_file.stat().st_mode
        except OSError:
            value, mode = "", 0
        if len(value) < MIN_TOKEN_CHARS:
            raise _error("The key proxy's token is missing; start the proxy first, or copy its token file to this "
                         f"runner and name it in {TOKEN_FILE_ENV}.")
        if os.name != "nt" and mode & 0o077:
            raise _error(f"The key proxy's token file {self.token_file} is readable by others; chmod 600 it.")
        return value

    def headers(self, model: str | None = None) -> dict[str, str]:
        """The authorization, waiting-budget and (optional) model headers of one proxied request."""
        headers = {"Authorization": "Bearer " + self.token, WAIT_HEADER: str(int(self.wait * 1000))}
        if model:
            headers[MODEL_HEADER] = model[:128]
        return headers

    def open(self, request: urllib.request.Request, *, timeout: float) -> Any:
        """Open a urllib request to the proxy directly, never through an environment HTTP proxy."""
        return _NO_PROXY.open(request, timeout=timeout)

    def health(self, *, timeout: float = 10.0) -> dict[str, Any]:
        """Read ``/keyproxy/health`` (no token needed; it carries counts and file names only).

        Retry transport failures up to three times. A single short read timeout must not
        terminate every database worker in a healthy, busy fleet. Paused/unavailable
        responses are returned unchanged, and persistent failures still propagate.

        Raises:
            OSError: The proxy is unreachable or answered with an error.
            ValueError: The answer is not a JSON object.
        """
        from .keyproxy_health import read_health
        return read_health(self.open, self.url, timeout=timeout)

    @functools.cached_property
    def _http(self) -> httpx.Client:
        return httpx.Client(trust_env=False, follow_redirects=False,
                            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64))

    def _client(self) -> httpx.Client:
        with _HTTP_LOCK:   # one shared client even when the first calls race
            return self._http

    def call(self, route: str, payload: Mapping[str, Any] | None = None, *, timeout: float = 10.0) -> dict[str, Any]:
        """POST a JSON body (or GET without one) to an authenticated proxy route over a kept-alive connection.

        Args:
            route: E.g. ``/question/reserve`` or ``/keyproxy/stats``.
            payload: The JSON body; ``None`` sends a GET.
            timeout: Seconds for the whole exchange.

        Returns:
            The JSON object the proxy answered.

        Raises:
            GatewayConfigurationError: The proxy refused this host's token (401) or the request itself (400): a
                configuration or version mismatch that retrying cannot fix.
            OSError: The proxy is unreachable or answered with another error (the status is in the message; no
                body text).
            ValueError: The answer is not a JSON object.
        """
        headers = {"Authorization": "Bearer " + self.token}
        try:
            if payload is None:
                response = self._client().get(self.url + route, headers=headers, timeout=timeout)
            else:
                response = self._client().post(self.url + route, json=dict(payload), headers=headers, timeout=timeout)
        except httpx.HTTPError as error:
            raise ConnectionError(f"key proxy {route} unreachable ({type(error).__name__})") from None
        if response.status_code == 401:
            raise _error(f"The key proxy at {self.url} refused this host's token; copy the proxy's current token "
                         f"file and name it in {TOKEN_FILE_ENV}.")
        if response.status_code == 400:
            code = ""
            with contextlib.suppress(ValueError, AttributeError, TypeError):
                code = str(response.json()["error"]["code"])[:64]
            raise _error(f"The key proxy refused {route} ({code or 'HTTP 400'}).")
        if response.status_code != 200:
            raise ConnectionError(f"key proxy {route} answered HTTP {response.status_code}")
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError(f"key proxy {route} must answer a JSON object")
        return result

    def check(self) -> dict[str, Any]:
        """One authenticated call, retried while the proxy cannot be connected to (at most :attr:`connect` s).

        A launch runs it once, so a missing proxy, a stale token copy or a wrong URL refuses the launch instead of
        holding every question at the gate.

        Returns:
            The proxy's health counts.

        Raises:
            GatewayConfigurationError: The token or URL is wrong, or the proxy stayed unreachable for the budget.
        """
        until, attempt = time.monotonic() + self.connect, 0
        while True:
            try:
                return self.call("/keyproxy/ping", timeout=10.0)
            except (OSError, ValueError) as error:
                now = time.monotonic()
                if now >= until:
                    raise _error(f"The key proxy at {self.url} is unreachable ({type(error).__name__}); start it "
                                 "first. No key outside the proxy is used.") from None
                time.sleep(min(connect_delay(attempt), until - now))
                attempt += 1


def connect_failed(error: BaseException) -> bool:
    """Whether a client error means the request never reached the proxy (refused, unroutable or connect timeout).

    Such a request is safe to send again: no key was used for it. A failure after the connection was made (a reset
    mid-request, a read timeout) is left to the caller's own retry policy.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (httpx.ConnectError, httpx.ConnectTimeout, ConnectionRefusedError)):
            return True
        if isinstance(current, urllib.error.URLError) and not isinstance(current, urllib.error.HTTPError):
            reason = current.reason
            if isinstance(reason, (ConnectionRefusedError, TimeoutError)) or (
                    isinstance(reason, OSError) and reason.errno in _UNREACHABLE):
                return True
        current = current.__cause__ or current.__context__
    return False


_UNREACHABLE = frozenset(getattr(errno, name) for name in (
    "ECONNREFUSED", "EHOSTUNREACH", "ENETUNREACH", "EHOSTDOWN", "ENETDOWN") if hasattr(errno, name))


def connect_delay(attempt: int) -> float:
    """Seconds before reconnect attempt ``attempt`` (0-based): 0.5 s doubling to 8 s, with up to 25 % jitter."""
    return min(8.0, 0.5 * 2 ** min(attempt, 4)) * random.uniform(1.0, 1.25)


def _error(message: str) -> RuntimeError:
    from .experiment_gateway import GatewayConfigurationError
    return GatewayConfigurationError(message)


# ----------------------------------------------------------------------------------------------- key state


@dataclass(eq=False)
class KeySlot:
    """One lent key's live state. ``secret`` is never rendered; everything else is file names and counts."""

    name: str
    secret: str = field(repr=False)
    in_flight: int = 0
    peak: int = 0
    cooling_until: float = 0.0
    paused_until: float = 0.0
    retired: bool = False
    strikes: int = 0
    #: POSIX time at which a 401 retires the key unless the 401 breaker trips first (0: none pending).
    retire_at: float = 0.0
    sent: int = 0
    upstream_s: float = 0.0
    statuses: Counter = field(default_factory=Counter)

    def usable(self, now: float) -> bool:
        """Neither retired nor paused (a cooling key is usable again soon)."""
        return not self.retired and self.paused_until <= now

    def available(self, now: float) -> bool:
        """Usable and not cooling: it counts toward the host's slot capacity."""
        return self.usable(now) and self.cooling_until <= now

    def summary(self, now: float) -> dict[str, Any]:
        """Counts and marks of this key; no key material."""
        answered = sum(count for status, count in self.statuses.items() if status != "none")
        return {"key": self.name, "state": self.state(now), "in_flight": self.in_flight, "peak": self.peak,
                "sent": self.sent, "retired": self.retired,
                "paused_until": self.paused_until if self.paused_until > now else None,
                "cooling_s": round(self.cooling_until - now, 1) if self.cooling_until > now else 0.0,
                "retire_at": self.retire_at or None, "strikes": self.strikes, "statuses": dict(self.statuses),
                "upstream_ms_mean": round(1000 * self.upstream_s / answered) if answered else None}

    def state(self, now: float) -> str:
        """``retired``, ``paused``, ``cooling`` or ``ready``."""
        if self.retired:
            return "retired"
        if self.paused_until > now:
            return "paused"
        return "cooling" if self.cooling_until > now else "ready"


class CapFile:
    """The host's per-key concurrency: the live cap file when it exists, else ``default``.

    The file is a plain integer 0-70 replaced atomically (as for the broker). It is re-read when its modification time
    changes, at most every :data:`TICK_S`. An unreadable moment keeps the last value; an invalid value pauses sending
    (0) until it is fixed, and is logged once.

    Args:
        path: The cap file.
        default: The cap while no file exists.
    """

    def __init__(self, path: Path, default: int) -> None:
        self.path = path
        self.default = default
        self._value = default
        self._stamp: float | None = None
        self._checked = float("-inf")
        self._warned = False

    def value(self) -> int:
        """The current per-key cap."""
        now = time.monotonic()
        if now - self._checked < TICK_S:
            return self._value
        self._checked = now
        try:
            stamp = self.path.stat().st_mtime_ns if self.path.exists() else None
        except OSError:
            return self._value
        if stamp == self._stamp:
            return self._value
        try:
            self._value = read_host_cap(self.path, str(self.default), maximum=MAX_PER_KEY)
            self._warned = False
        except RuntimeError:
            if self.path.exists():
                self._value = 0
                if not self._warned:
                    self._warned = True
                    _LOGGER.warning("[keyproxy] Cap file %s is not an integer 0-70; sending paused.", self.path)
            return self._value
        self._stamp = stamp
        return self._value


@dataclass
class Restored:
    """The key marks a starting proxy restores: retirements, unexpired 403 pauses and unexpired 429 cooling."""

    retired: set[str] = field(default_factory=set)
    paused: dict[str, float] = field(default_factory=dict)
    cooling: dict[str, tuple[float, int]] = field(default_factory=dict)


class Marks:
    """Durable key marks: an append-only ``marks.jsonl`` of ``{at, key, kind, until[, strikes]}`` rows.

    ``kind`` is ``retired`` (permanent), ``paused`` (a 403, until ``until``) or ``cooling`` (one 429 episode, until
    ``until``, with the key's consecutive ``strikes``). Rows name only approved file names.
    """

    KINDS = ("retired", "paused", "cooling")

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self, now: float) -> Restored:
        """The marks still in force at ``now``.

        A line that is not JSON (an append torn by a crash) is skipped and logged: losing one mark costs at most one
        more refusal from that key, while refusing to start would stop the whole fleet.

        Raises:
            ValueError: A JSON row is not a mark of an approved file name.
        """
        restored = Restored()
        if not self.path.exists():
            return restored
        torn = 0
        for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                torn += 1
                continue
            if not isinstance(row, dict) or not valid_key_name(row.get("key")) or row.get("kind") not in self.KINDS:
                raise ValueError("invalid key proxy mark")
            name = row["key"]
            if row["kind"] == "retired":
                restored.retired.add(name)
            elif row["kind"] == "paused" and float(row["until"]) > now:
                restored.paused[name] = max(restored.paused.get(name, 0.0), float(row["until"]))
            elif row["kind"] == "cooling":
                # The latest episode's strikes carry over; an expired one still remembers them until a success.
                until = float(row["until"])
                restored.cooling[name] = (until if until > now else 0.0, int(row.get("strikes") or 0))
        if torn:
            _LOGGER.warning("[keyproxy] Skipped %d unreadable line(s) of %s (a torn append).", torn, self.path)
        return restored

    def append(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Append marks durably (flush and fsync), starting on a new line after a torn last line."""
        if not rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.path.open("a+b") as output:
            output.seek(0, os.SEEK_END)
            text = "".join(json.dumps(dict(row), separators=(",", ":")) + "\n" for row in rows)
            if output.tell():
                output.seek(-1, os.SEEK_END)
                if output.read(1) != b"\n":
                    text = "\n" + text
            output.write(text.encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())


def broker_marks(broker_path: Path, now: float) -> list[dict[str, Any]]:
    """Read the host broker's retirements and unexpired 403 pauses, read-only, as proxy marks.

    Args:
        broker_path: The broker's SQLite file (``volcano-host-broker.sqlite3``); it is opened ``mode=ro``.
        now: POSIX seconds; pauses that already ended are skipped.

    Returns:
        Mark rows for :meth:`Marks.append`.
    """
    uri = broker_path.expanduser().resolve().as_uri() + "?mode=ro"
    with contextlib.closing(sqlite3.connect(uri, uri=True, timeout=5)) as connection:
        rows = connection.execute("SELECT file_name, exhausted_until, retired FROM broker_file_keys").fetchall()
    return marks_from_export([{"file_name": name, "exhausted_until": until, "retired": retired}
                              for name, until, retired in rows], now)


def marks_from_export(rows: Sequence[Mapping[str, Any]], now: float) -> list[dict[str, Any]]:
    """Convert ``key_broker export-marks`` rows (or broker rows) into proxy marks; unknown names are refused.

    Raises:
        ValueError: A row names a file outside the approved allow-list or has an invalid value.
    """
    marks = []
    for row in rows:
        name = row.get("file_name")
        if not valid_key_name(name) or row.get("retired") not in (0, 1, True, False):
            raise ValueError("Invalid file-key mark; only approved names are accepted.")
        until = float(row.get("exhausted_until") or 0)
        if row["retired"]:
            marks.append({"at": now, "key": name, "kind": "retired", "until": None})
        elif until > now:
            marks.append({"at": now, "key": name, "kind": "paused", "until": until})
    return marks


class Ledger:
    """One JSON row per upstream attempt or refusal, in a file per Asia/Shanghai date, line-buffered."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._day: str | None = None
        self._file: Any = None
        self._failed_at = float("-inf")

    def write(self, row: dict[str, Any]) -> None:
        """Append one row (its ``t`` decides the file).

        A write error (a full disk, a day file that cannot be opened) is logged at most once a minute and dropped:
        the answer the row describes was paid for and still goes to its client, and the in-memory usage counters
        keep counting.
        """
        try:
            day = quota_day(row["t"])
            if day != self._day:
                self.close()
                self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                self._file = (self.directory / f"{day}.jsonl").open("a", encoding="utf-8", buffering=1)
                self._day = day
            self._file.write(json.dumps(row, separators=(",", ":")) + "\n")
        except OSError as error:
            self.close()
            now = time.monotonic()
            if now - self._failed_at >= 60:
                self._failed_at = now
                _LOGGER.error("[keyproxy] Ledger write failed (%s); rows are dropped until it recovers.",
                              type(error).__name__)

    def close(self) -> None:
        """Close the current file."""
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None
            self._day = None


class KeyPool:
    """The host's keys and the queue of requests waiting for one, all on one event loop (no locks needed).

    Args:
        slots: The keys in the owner's fixed order.
        cap: The live per-key cap.
        marks: Durable pause/retirement store.
        clock: POSIX seconds (tests pass a fake clock).
    """

    def __init__(self, slots: Sequence[KeySlot], cap: CapFile, marks: Marks, clock: Callable[[], float] = time.time) -> None:
        self.slots = list(slots)
        self.cap = cap
        self.marks = marks
        self.clock = clock
        self.cursor = 0
        self.closing = False
        #: POSIX time of the last 2xx answer: a 401 followed by a success elsewhere is the key's own fault.
        self.last_success = float("-inf")
        #: POSIX time until which the last suspected gateway-wide 401 fault pauses the keys it hit.
        self.gateway_fault_until = 0.0
        self._waiters: deque[asyncio.Future] = deque()
        self._unauthorized: deque[tuple[float, str]] = deque()
        self._persist_failed_at = float("-inf")

    @property
    def waiting(self) -> int:
        """Requests queued for a key now (withdrawn and served requests leave the queue at once)."""
        return len(self._waiters)

    def _persist(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Append marks; a write error is logged (at most once a minute) and the in-memory state still applies."""
        try:
            self.marks.append(rows)
        except OSError as error:
            now = time.monotonic()
            if now - self._persist_failed_at >= 60:
                self._persist_failed_at = now
                _LOGGER.error("[keyproxy] Cannot persist key marks (%s); they hold until this proxy exits.",
                              type(error).__name__)

    def _pick(self, now: float, per_key: int) -> KeySlot | None:
        count = len(self.slots)
        for step in range(count):
            slot = self.slots[(self.cursor + step) % count]
            if slot.available(now) and slot.in_flight < per_key:
                self.cursor = (self.cursor + step + 1) % count
                slot.in_flight += 1
                slot.sent += 1
                slot.peak = max(slot.peak, slot.in_flight)
                return slot
        return None

    def dispatch(self) -> None:
        """Hand free slots to the queued requests, first come first served."""
        now, per_key = self.clock(), self.cap.value()
        while self._waiters:
            future = self._waiters[0]
            if future.done():
                self._waiters.popleft()
                continue
            slot = self._pick(now, per_key)
            if slot is None:
                return
            self._waiters.popleft()
            future.set_result(slot)

    def paused_for(self) -> float | None:
        """Seconds until some key is usable again when every key that is not retired is paused, else None."""
        now = self.clock()
        live = [slot for slot in self.slots if not slot.retired]
        if not live or any(slot.usable(now) for slot in live):
            return None
        return min(slot.paused_until for slot in live) - now

    async def acquire(self, deadline: float, gone: asyncio.Future | None = None) -> tuple[KeySlot | None, str]:
        """Take a slot of the next eligible key, waiting in the queue while none is free.

        Args:
            deadline: Loop time after which the request stops waiting.
            gone: Completes when the request's client disconnects (no polling).

        Returns:
            ``(slot, "ok")``, or ``(None, reason)`` with reason ``all_retired``, ``all_paused`` (every usable key is
            paused past the deadline: refused at once), ``shutdown``, ``wait_deadline`` or ``client_disconnected``. A
            cancelled caller never keeps a slot.
        """
        if self.closing:
            return None, "shutdown"
        if all(slot.retired for slot in self.slots):
            return None, "all_retired"
        loop = asyncio.get_running_loop()
        paused = self.paused_for()
        if paused is not None and paused > deadline - loop.time():
            return None, "all_paused"
        if not self.waiting:
            slot = self._pick(self.clock(), self.cap.value())
            if slot is not None:
                return slot, "ok"
        future = loop.create_future()
        self._waiters.append(future)
        try:
            reason = "wait_deadline"
            watched = {future} if gone is None else {future, gone}
            while not future.done():
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                await asyncio.wait(watched, timeout=remaining, return_when=asyncio.FIRST_COMPLETED)
                if not future.done() and gone is not None and gone.done():
                    reason = "client_disconnected"
                    break
        except BaseException:
            self._abandon(future)
            raise
        if future.done() and not future.cancelled():
            slot = future.result()
            return (slot, "ok") if slot is not None else (None, "shutdown")
        self._abandon(future)
        return None, reason

    def _abandon(self, future: asyncio.Future) -> None:
        """Withdraw a queued request; a slot handed to it in the meantime goes back to the queue."""
        if future.done() and not future.cancelled():
            slot = future.result()
            if slot is not None:
                slot.in_flight -= 1
                slot.sent -= 1
                self.dispatch()
            return
        future.cancel()
        with contextlib.suppress(ValueError):
            self._waiters.remove(future)

    def release(self, slot: KeySlot, status: int | None, retry_after: str | None, *,
                sent_at: float | None = None) -> str | None:
        """Free a slot and apply the owner's rule for the provider's answer.

        Args:
            slot: The slot taken by :meth:`acquire`.
            status: HTTP status, or None when no answer arrived (transport failure, disconnect).
            retry_after: The answer's ``Retry-After`` header.
            sent_at: POSIX time the request was sent (default now): a 403 pauses its key until the first 00:05
                Asia/Shanghai after it, so a refusal of yesterday's quota answered just past 00:05 costs no day.

        Returns:
            ``cooling``, ``paused``, ``retiring`` (a 401, retired once :data:`BREAKER_WINDOW_S` confirms it) or
            ``retired`` when the key's state changed, else None.
        """
        slot.in_flight -= 1
        try:
            now = self.clock()
            slot.statuses[str(status) if status is not None else "none"] += 1
            action = None
            if status == 429:
                delay = cooling_seconds(retry_after, slot.strikes)
                if slot.cooling_until <= now:
                    # One episode per cooling, however many in-flight requests report it; each episode is persisted.
                    slot.strikes += 1
                    slot.cooling_until = now + delay
                    self._persist([{"at": now, "key": slot.name, "kind": "cooling", "until": slot.cooling_until,
                                    "strikes": slot.strikes}])
                else:
                    slot.cooling_until = max(slot.cooling_until, now + delay)
                action = "cooling"
            elif status == 403:
                reset = next_quota_reset(now if sent_at is None else min(sent_at, now))
                if reset > slot.paused_until:   # one mark per pause, however many in-flight requests report it
                    slot.paused_until = reset
                    self._persist([{"at": now, "key": slot.name, "kind": "paused", "until": reset}])
                action = "paused"
            elif status == 401:
                action = "retired" if slot.retired else self._unauthorized_key(slot, now)
            elif status is not None and 200 <= status < 300:
                self.last_success = max(self.last_success, now)
                if slot.strikes:
                    slot.strikes = 0
                    self._persist([{"at": now, "key": slot.name, "kind": "cooling", "until": 0, "strikes": 0}])
            return action
        finally:
            self.dispatch()

    def _unauthorized_key(self, slot: KeySlot, now: float) -> str:
        """A 401: pause the key and retire it at :meth:`settle` unless a gateway-wide 401 fault is suspected."""
        self._unauthorized.append((now, slot.name))
        if slot.retire_at:
            return "retiring"
        if now < self.gateway_fault_until:
            until = self.gateway_fault_until
        else:
            until = slot.retire_at = now + BREAKER_WINDOW_S
        if until > slot.paused_until:
            slot.paused_until = until
            self._persist([{"at": now, "key": slot.name, "kind": "paused", "until": until}])
        return "retiring" if slot.retire_at else "paused"

    def settle(self) -> None:
        """Decide 401 retirements whose confirmation window passed (called by the proxy's ticker).

        A key is retired when a 2xx answer arrived after its 401 (the gateway accepts other keys) or when fewer
        than :data:`BREAKER_KEYS` distinct keys were refused with 401 in its window. Otherwise every key refused in
        that window is paused for :data:`BREAKER_PAUSE_S` and kept: a gateway-wide 401 fault never retires the pool.
        """
        now = self.clock()
        while self._unauthorized and self._unauthorized[0][0] < now - 3 * BREAKER_WINDOW_S:
            self._unauthorized.popleft()
        rows = []
        for slot in self.slots:
            if not slot.retire_at or slot.retire_at > now:
                continue
            since = slot.retire_at - BREAKER_WINDOW_S
            refused = {name for at, name in self._unauthorized if since <= at <= slot.retire_at}
            slot.retire_at = 0.0
            if self.last_success >= since or len(refused) < BREAKER_KEYS:
                slot.retired = True
                rows.append({"at": now, "key": slot.name, "kind": "retired", "until": None})
                continue
            if now >= self.gateway_fault_until:
                self.gateway_fault_until = now + BREAKER_PAUSE_S
                _LOGGER.error("[keyproxy] %d keys were refused with 401 within %g s and nothing succeeded: a "
                              "gateway fault is suspected, so no key is retired; they pause for %g s.",
                              len(refused), BREAKER_WINDOW_S, BREAKER_PAUSE_S)
            slot.paused_until = max(slot.paused_until, self.gateway_fault_until)
            rows.append({"at": now, "key": slot.name, "kind": "paused", "until": slot.paused_until})
        self._persist(rows)

    def close(self) -> None:
        """Stop handing out keys: queued requests get ``shutdown`` at once; requests already sending finish."""
        self.closing = True
        while self._waiters:
            future = self._waiters.popleft()
            if not future.done():
                future.set_result(None)

    def capacity(self) -> int:
        """Request slots now: available keys (neither retired, paused nor cooling) times the live per-key cap."""
        now = self.clock()
        return sum(1 for slot in self.slots if slot.available(now)) * self.cap.value()

    def summary(self) -> dict[str, Any]:
        """Pool-level counts for ``/keyproxy/health``."""
        now, per_key = self.clock(), self.cap.value()
        available = sum(1 for slot in self.slots if slot.available(now))
        return {"keys": len(self.slots), "per_key": per_key, "available_keys": available,
                "usable_keys": sum(1 for slot in self.slots if slot.usable(now)),
                "cooling": sum(1 for slot in self.slots if slot.usable(now) and slot.cooling_until > now),
                "paused": sum(1 for slot in self.slots if not slot.retired and slot.paused_until > now),
                "retiring": sum(1 for slot in self.slots if slot.retire_at),
                "retired": sum(1 for slot in self.slots if slot.retired),
                "gateway_401_fault": now < self.gateway_fault_until,
                "in_flight": sum(slot.in_flight for slot in self.slots), "waiting": self.waiting,
                "capacity": available * per_key}


def load_slots(paths: tuple[Path, ...], restored: Restored, *,
               retired: frozenset[str] = frozenset()) -> list[KeySlot]:
    """Read every approved key that is not retired, in the owner's order; retired keys are never opened.

    Args:
        paths: The validated approved key files.
        restored: The marks in force (:meth:`Marks.load`).
        retired: Files retired whatever the marks say: the owner's retirements (tests pass their own).

    Raises:
        GatewayConfigurationError: A key that is not retired cannot be read (its file name is not secret and is
            logged; its contents never are).
    """
    from .chat_keys import DeniedChatKey
    slots = []
    for path in validate_files([str(path) for path in paths]):
        name = path.name
        if name in restored.retired or name in retired:
            slots.append(KeySlot(name, "", retired=True))
            continue
        try:
            secret = read_key_file(paths, name)
        except DeniedChatKey:
            slots.append(KeySlot(name, "", retired=True))
            continue
        except RuntimeError:
            _LOGGER.error("[keyproxy] Cannot load approved key file %s; contents withheld.", name)
            raise
        cooling_until, strikes = restored.cooling.get(name, (0.0, 0))
        slots.append(KeySlot(name, secret, paused_until=restored.paused.get(name, 0.0), cooling_until=cooling_until,
                             strikes=strikes))
    return slots


# ----------------------------------------------------------------------------------------------- proxy


def _usage(content: bytes) -> tuple[dict[str, Any] | None, str | None, dict[str, int]]:
    """``(parsed body, served model, usage counts)`` of a provider answer; empty when it is not a JSON object."""
    try:
        parsed = json.loads(content)
    except (ValueError, UnicodeDecodeError):
        return None, None, {}
    if not isinstance(parsed, dict):
        return None, None, {}
    usage = parsed.get("usage") if isinstance(parsed.get("usage"), dict) else {}
    counts = {name: int(usage[name]) for name in ("prompt_tokens", "completion_tokens", "total_tokens")
              if isinstance(usage.get(name), int)}
    for parent, name, label in (("completion_tokens_details", "reasoning_tokens", "reasoning_tokens"),
                                ("prompt_tokens_details", "cached_tokens", "cached_tokens"),
                                ("prompt_tokens_details", "cache_creation_input_tokens", "cache_creation_input_tokens")):
        details = usage.get(parent)
        if isinstance(details, dict) and isinstance(details.get(name), int):
            counts[label] = int(details[name])
    model = parsed.get("model") if isinstance(parsed.get("model"), str) else None
    return parsed, model, counts


def _requested_model(body: bytes) -> str | None:
    try:
        model = json.loads(body).get("model")
    except (ValueError, UnicodeDecodeError, AttributeError):
        return None
    return model[:128] if isinstance(model, str) else None


class _Disconnected(Exception):
    """The client went away while its request was upstream."""


class _UpstreamError(Exception):
    """The gateway could not be reached or the exchange broke; the message is the error's type name only."""


#: Provider response headers returned to the client unchanged.
_PASSED_HEADERS = ("content-type", "retry-after")


@dataclass(frozen=True)
class _Answer:
    """One provider answer: its status, its passed-through headers (lower-case names) and its body bytes."""

    status: int
    headers: Mapping[str, str]
    content: bytes


class KeyProxy:
    """The proxy's request handling, telemetry and lifecycle, independent of the ASGI server that hosts it.

    Args:
        pool: The host's keys.
        token: The local bearer token clients must present.
        state_dir: Where the ledger lives.
        upstream: The chat completions URL; always the approved Volcengine gateway in production.
        transport: An httpx transport replacing the network (tests only); without it the proxy sends with aiohttp.
        max_wait: Upper bound of any request's in-proxy waiting budget, in seconds.
        ends_at: Optional lent window end (POSIX seconds); requests are refused from then on.
        book: The global question-start book; by default one over this pool's capacity.
        drain: Seconds a shutdown waits for in-flight requests before :attr:`on_shutdown` ends the server anyway.
    """

    def __init__(self, pool: KeyPool, token: str, state_dir: Path, *, upstream: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None, max_wait: float = DEFAULT_MAX_WAIT_S,
                 ends_at: float | None = None, book: QuestionBook | None = None,
                 drain: float = DEFAULT_DRAIN_S) -> None:
        from .experiment_gateway import GATEWAY_BASE_URL
        self.upstream = upstream or GATEWAY_BASE_URL.rstrip("/") + "/chat/completions"
        if "openrouter" in self.upstream.lower():
            raise _error("The key proxy never calls OpenRouter.")
        self.pool = pool
        self.state_dir = state_dir
        self.max_wait = max_wait
        self.ends_at = ends_at
        self.ledger = Ledger(state_dir / "ledger")
        self.book = book if book is not None else QuestionBook(pool.capacity)
        self.started_at = time.time()
        self.drain = drain
        #: Called once by the ticker when a shutdown has drained (nothing in flight, or ``drain`` passed): it ends
        #: the hosting server, which closes the listener only then.
        self.on_shutdown: Callable[[], None] | None = None
        self.draining_since: float | None = None
        if len(token) < MIN_TOKEN_CHARS:
            raise _error("The key proxy's token is missing or too short.")
        self._expected = ("Bearer " + token).encode()
        self._transport = transport
        self._clients: dict[str, Any] = {}
        self._ticker: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown_requested = False
        self._exited = False
        self._tick_failed_at = float("-inf")
        self._waits: deque[float] = deque(maxlen=RECENT_WAITS)
        self._requests: Counter = Counter()
        self._usage: dict[tuple[str, str], Counter] = {}
        self._restore_usage()

    def _restore_usage(self) -> None:
        """Seed today's token usage from today's ledger, so a restart keeps the day's budget count."""
        path = self.state_dir / "ledger" / f"{quota_day(time.time())}.jsonl"
        if not path.exists():
            return
        day = path.stem
        for model, counts in summarize_ledger(path)["usage"].items():
            self._usage[(day, model)] = Counter(counts)

    # -- lifecycle

    async def start(self) -> None:
        """Open one upstream session per usable key and start the queue ticker (cap file and expired cooling).

        The network path is aiohttp with one kept-alive connection pool per key (at most 64 connections; larger slot
        caps can queue at that pool). In the 2026-11-04 benchmark an httpx client saturated the proxy's core at about 200
        requests per second: httpcore scans its whole pool for every request, and a shared pool smaller than the
        requests in flight reconnected (a new TLS handshake upstream) for almost every request.
        """
        if self._transport is not None:
            for slot in self.pool.slots:
                self._clients[slot.name] = httpx.AsyncClient(transport=self._transport, trust_env=False,
                                                             follow_redirects=False, timeout=None)
        else:
            import aiohttp
            for slot in self.pool.slots:
                if not slot.retired:
                    self._clients[slot.name] = aiohttp.ClientSession(
                        connector=aiohttp.TCPConnector(limit=64, keepalive_timeout=30, ttl_dns_cache=300),
                        timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=UPSTREAM_READ_S),
                        trust_env=False)
        self._loop = asyncio.get_running_loop()
        self._ticker = asyncio.create_task(self._tick())

    async def _tick(self) -> None:
        """Every :data:`TICK_S`: wake queued requests (cap file, expired cooling), settle 401 retirements, expire
        silent question owners and end a drained shutdown. An error is logged (once a minute) and never stops it."""
        while True:
            await asyncio.sleep(TICK_S)
            try:
                self.pool.dispatch()
                self.pool.settle()
                self.book.expire()
                self._end_drained()
            except Exception as error:  # noqa: BLE001 - the ticker must outlive any one failure
                now = time.monotonic()
                if now - self._tick_failed_at >= 60:
                    self._tick_failed_at = now
                    _LOGGER.exception("[keyproxy] Ticker step failed (%s); continuing.", type(error).__name__)

    def _end_drained(self) -> None:
        if self.draining_since is None or self._exited or self.on_shutdown is None:
            return
        in_flight = sum(slot.in_flight for slot in self.pool.slots)
        if in_flight and time.monotonic() - self.draining_since < self.drain:
            return
        self._exited = True
        if in_flight:
            _LOGGER.error("[keyproxy] %d request(s) still upstream after the %g s drain; they are cancelled.",
                          in_flight, self.drain)
        self.on_shutdown()

    def request_shutdown(self) -> bool:
        """Schedule :meth:`shutdown` on the proxy's loop from a signal handler or another thread.

        Returns:
            False when a shutdown was already requested (the caller may then stop harder).
        """
        if self._shutdown_requested:
            return False
        self._shutdown_requested = True
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self.shutdown)
        else:
            self.shutdown()
        return True

    def shutdown(self) -> None:
        """Refuse queued and new requests and questions; requests already upstream finish.

        The listener stays open while they drain (health answers ``closing``), so a ``stop`` that waits for the
        health route to go away waits for the drain; the ticker then calls :attr:`on_shutdown`.
        """
        self.pool.close()
        self.book.close()
        self._shutdown_requested = True
        if self.draining_since is None:
            self.draining_since = time.monotonic()

    async def aclose(self) -> None:
        """Stop the ticker, close the upstream client and the ledger."""
        if self._ticker is not None:
            self._ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ticker
        for client in self._clients.values():
            await (client.aclose() if isinstance(client, httpx.AsyncClient) else client.close())
        self._clients.clear()
        self.ledger.close()

    def authorized(self, header: str | None) -> bool:
        """Whether a request carries the local bearer token (constant-time comparison)."""
        return secrets.compare_digest((header or "").encode(), self._expected)

    # -- requests

    def _budget(self, header: str | None) -> float:
        try:
            requested = int(header) / 1000 if header is not None else self.max_wait
        except ValueError:
            requested = self.max_wait
        return max(0.0, min(self.max_wait, requested))

    async def forward(self, body: bytes, *, content_type: str | None, wait_header: str | None,
                      gone: asyncio.Future | None = None,
                      model_header: str | None = None) -> tuple[int, bytes, dict[str, str]]:
        """Send one chat request with the next eligible key, moving on after 401/403/429, within the waiting budget.

        Args:
            body: The client's request body, sent unchanged.
            content_type: The client's ``Content-Type``.
            wait_header: The client's :data:`WAIT_HEADER` value, if any.
            gone: Completes when the client disconnects; the request is then withdrawn or cancelled upstream.
            model_header: The client's :data:`MODEL_HEADER` (the requested model, for the ledger); without it the
                body is parsed for the model.

        Returns:
            ``(status, body, headers)`` of the provider's answer, or of the proxy's own JSON refusal (499 for a
            client that left, 502 when the gateway was unreachable, 503 with ``Retry-After`` when no key could send).
        """
        loop = asyncio.get_running_loop()
        digest = hashlib.sha256(body).hexdigest()[:16]
        model = model_header[:128] if model_header else _requested_model(body)
        if self.ends_at is not None and time.time() >= self.ends_at:
            return self._refuse(digest, model, 503, "keyproxy_window_ended", 0.0, 0)
        deadline = loop.time() + self._budget(wait_header)
        waited, attempts = 0.0, 0
        headers = {"content-type": content_type or "application/json"}
        while True:
            before = loop.time()
            slot, reason = await self.pool.acquire(deadline, gone)
            waited += loop.time() - before
            if slot is None:
                status = 499 if reason == "client_disconnected" else 503
                code = {"all_retired": "keyproxy_all_keys_retired", "all_paused": "keyproxy_all_keys_paused",
                        "shutdown": "keyproxy_shutting_down",
                        "wait_deadline": "keyproxy_all_keys_busy"}.get(reason, "keyproxy_" + reason)
                paused = self.pool.paused_for() if reason == "all_paused" else None
                return self._refuse(digest, model, status, code, waited, attempts,
                                    retry_after=None if paused is None else max(1, math.ceil(paused)))
            attempts += 1
            sent, sent_at = loop.time(), time.time()
            try:
                response = await self._send(slot, body, headers, gone)
            except _Disconnected:
                self.pool.release(slot, None, None)
                self._record(digest, model, slot, attempts, None, waited, loop.time() - sent, error="client_disconnected")
                return self._refuse(digest, model, 499, "keyproxy_client_disconnected", waited, attempts, record=False)
            except _UpstreamError as error:
                self.pool.release(slot, None, None)
                self._record(digest, model, slot, attempts, None, waited, loop.time() - sent, error=str(error))
                return self._refuse(digest, model, 502, "keyproxy_upstream_unreachable", waited, attempts, record=False)
            except BaseException:
                self.pool.release(slot, None, None)
                raise
            upstream_s = loop.time() - sent
            deadline += upstream_s   # only waiting counts against the budget
            slot.upstream_s += upstream_s
            status = response.status
            action = self.pool.release(slot, status, response.headers.get("retry-after"), sent_at=sent_at)
            self._record(digest, model, slot, attempts, status, waited, upstream_s, action=action,
                         content=response.content)
            if status in (401, 403, 429) and attempts < MAX_ATTEMPTS:
                continue
            if status in (401, 403, 429):
                return self._refuse(digest, model, 503, "keyproxy_attempts_exhausted", waited, attempts)
            out = dict(response.headers)
            out.update({KEY_HEADER: slot.name, WAIT_HEADER: str(round(waited * 1000)), ATTEMPTS_HEADER: str(attempts)})
            self._waits.append(waited)
            return status, response.content, out

    async def _post(self, slot: KeySlot, body: bytes, headers: dict[str, str]) -> _Answer:
        """One upstream exchange with ``slot``'s key; any transport failure becomes :class:`_UpstreamError`."""
        client = self._clients[slot.name]
        headers = {**headers, "authorization": "Bearer " + slot.secret}
        if isinstance(client, httpx.AsyncClient):
            try:
                response = await client.post(self.upstream, content=body, headers=headers)
            except httpx.HTTPError as error:
                raise _UpstreamError(type(error).__name__) from None
            return _Answer(response.status_code, {name: value for name, value in response.headers.items()
                                                  if name in _PASSED_HEADERS}, response.content)
        import aiohttp
        loop = asyncio.get_running_loop()
        for retry in (True, False):
            sent = loop.time()
            try:
                async with client.post(self.upstream, data=body, headers=headers, allow_redirects=False) as response:
                    return _Answer(response.status, {name.lower(): value for name, value in response.headers.items()
                                                     if name.lower() in _PASSED_HEADERS}, await response.read())
            except (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectorError, aiohttp.ClientOSError) as error:
                # A kept-alive connection the gateway already closed fails at once, before the provider saw the
                # request; one retry opens a fresh connection. A late failure is not retried (it may have been seen).
                if not (retry and loop.time() - sent < STALE_RETRY_S):
                    raise _UpstreamError(type(error).__name__) from None
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                raise _UpstreamError(type(error).__name__) from None
        raise AssertionError("unreachable")

    async def _send(self, slot: KeySlot, body: bytes, headers: dict[str, str],
                    gone: asyncio.Future | None) -> _Answer:
        task = asyncio.ensure_future(self._post(slot, body, headers))
        try:
            if gone is not None:
                await asyncio.wait({task, gone}, return_when=asyncio.FIRST_COMPLETED)
                if not task.done():
                    raise _Disconnected
            return await task
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task

    def _refuse(self, digest: str, model: str | None, status: int, code: str, waited: float, attempts: int,
                *, record: bool = True, retry_after: int | None = None) -> tuple[int, bytes, dict[str, str]]:
        if record:
            self.ledger.write({"t": round(time.time(), 3), "req": digest, "model": model, "st": status, "err": code,
                               "wait_ms": round(waited * 1000), "n": attempts})
            self._requests[str(status)] += 1
        self._waits.append(waited)
        payload = json.dumps({"error": {"code": code, "type": "keyproxy",
                                        "message": "The host key proxy could not send this request; no answer was produced."}})
        headers = {"content-type": "application/json", WAIT_HEADER: str(round(waited * 1000)), ATTEMPTS_HEADER: str(attempts)}
        if status == 503:
            headers["retry-after"] = str(retry_after if retry_after is not None
                                         else 5 if code == "keyproxy_shutting_down" else 30)
        return status, payload.encode(), headers

    def _record(self, digest: str, model: str | None, slot: KeySlot, attempt: int, status: int | None, waited: float,
                upstream_s: float, *, action: str | None = None, error: str | None = None, content: bytes = b"") -> None:
        now = time.time()
        row: dict[str, Any] = {"t": round(now, 3), "req": digest, "key": slot.name, "n": attempt, "st": status,
                               "wait_ms": round(waited * 1000), "ms": round(upstream_s * 1000), "model": model}
        if action:
            row["act"] = action
        if error:
            row["err"] = error
        parsed, served, usage = _usage(content) if content else (None, None, {})
        if served:
            row["served"] = served[:128]
        if status == 200 and usage:
            row["usage"] = usage
            bucket = self._usage.setdefault((quota_day(now), served or model or "unknown"), Counter())
            bucket.update(usage)
            bucket["calls"] += 1
        if status is not None and (status != 200 or (parsed is not None and parsed.get("error"))):
            event = provider_errors.event(status=status, body=parsed if parsed is not None else content[:4000],
                                          model=model or "", file_name=slot.name, key_id="", secret=slot.secret, now=now)
            row.update(code=event["code"], type=event["type"], msg=event["message"])
        self._requests[str(status) if status is not None else "none"] += 1
        self.ledger.write(row)

    # -- telemetry

    def health(self) -> dict[str, Any]:
        """Cheap counts: what liveness and launch checks read. No token is needed for it."""
        draining = None if self.draining_since is None else round(time.monotonic() - self.draining_since, 1)
        return {"status": "closing" if self.pool.closing else "ok", "pid": os.getpid(), "draining_s": draining,
                **self.pool.summary(),
                "window_end": None if self.ends_at is None else datetime.fromtimestamp(self.ends_at, QUOTA_TZ).isoformat(),
                "questions": self.book.summary()}

    def stats(self) -> dict[str, Any]:
        """Per-key state, status counts, recent waiting times and token usage by day and model.

        ``usage`` is ``{day: {model: {prompt_tokens, cached_tokens, completion_tokens, ..., calls}}}`` over 200
        answers (today's survives a restart, from the ledger), plus ``{day: {"all": totals}}``.
        """
        now = self.pool.clock()
        waits = sorted(self._waits)

        def quantile(fraction: float) -> float | None:
            return round(waits[min(len(waits) - 1, int(fraction * (len(waits) - 1)))], 3) if waits else None

        usage: dict[str, dict[str, dict[str, int]]] = {}
        for (day, model), counts in sorted(self._usage.items()):
            usage.setdefault(day, {})[model] = dict(counts)
        for models in usage.values():
            models["all"] = dict(sum((Counter(counts) for counts in models.values()), Counter()))
        return {**self.health(), "started_at": self.started_at, "requests_by_status": dict(self._requests),
                "wait_s": {"n": len(waits), "mean": round(sum(waits) / len(waits), 3) if waits else None,
                           "p50": quantile(0.5), "p90": quantile(0.9), "p99": quantile(0.99),
                           "max": round(waits[-1], 3) if waits else None},
                "usage": usage, "key_states": [slot.summary(now) for slot in self.pool.slots]}


async def _disconnect(receive: Callable[[], Awaitable[Mapping[str, Any]]]) -> None:
    """Return once the ASGI client disconnects; call only after the request body was read."""
    while (await receive()).get("type") != "http.disconnect":
        pass


def build_app(proxy: KeyProxy) -> Any:
    """The ASGI application: ``POST /v1/chat/completions``, ``GET /keyproxy/health|ping|stats``,
    ``POST /keyproxy/shutdown`` and ``POST /question/reserve|phase|release`` (:mod:`.question_book`).

    Everything else answers 404. Every route but health needs the bearer token; health carries only counts.
    """
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    def unauthorized() -> Response:
        return JSONResponse({"error": {"code": "keyproxy_unauthorized", "type": "keyproxy"}}, status_code=401)

    async def chat(request: Request) -> Response:
        if not proxy.authorized(request.headers.get("authorization")):
            return unauthorized()
        body = await request.body()
        gone = asyncio.ensure_future(_disconnect(request.receive))
        try:
            status, content, headers = await proxy.forward(
                body, content_type=request.headers.get("content-type"), wait_header=request.headers.get(WAIT_HEADER),
                gone=gone, model_header=request.headers.get(MODEL_HEADER))
        finally:
            gone.cancel()
        return Response(content, status_code=status, headers=headers)

    async def health(_: Request) -> Response:
        return JSONResponse(proxy.health())

    async def ping(request: Request) -> Response:
        if not proxy.authorized(request.headers.get("authorization")):
            return unauthorized()
        return JSONResponse(proxy.health())

    async def stats(request: Request) -> Response:
        if not proxy.authorized(request.headers.get("authorization")):
            return unauthorized()
        return JSONResponse(proxy.stats())

    async def shutdown(request: Request) -> Response:
        if not proxy.authorized(request.headers.get("authorization")):
            return unauthorized()
        proxy.shutdown()
        return JSONResponse({"status": "closing", "in_flight": proxy.pool.summary()["in_flight"]})

    def question(action: str) -> Callable[[Request], Awaitable[Response]]:
        async def route(request: Request) -> Response:
            if not proxy.authorized(request.headers.get("authorization")):
                return unauthorized()
            try:
                payload = json.loads(await request.body())
                if not isinstance(payload, dict):
                    raise QuestionError("the body must be a JSON object")
                if action == "reserve":
                    gone = asyncio.ensure_future(_disconnect(request.receive))
                    try:
                        result = await proxy.book.reserve(payload, gone)
                    finally:
                        gone.cancel()
                else:
                    result = getattr(proxy.book, action)(payload)
            except (QuestionError, ValueError, TypeError) as error:
                return JSONResponse({"error": {"code": "keyproxy_bad_question_request", "type": "keyproxy",
                                               "message": str(error)[:200]}}, status_code=400)
            return JSONResponse(result)
        return route

    async def missing(_: Request) -> Response:
        return JSONResponse({"error": {"code": "keyproxy_route_not_allowed", "type": "keyproxy"}}, status_code=404)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Any):
        await proxy.start()
        try:
            yield
        finally:
            await proxy.aclose()

    return Starlette(routes=[
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/keyproxy/health", health, methods=["GET"]),
        Route("/keyproxy/ping", ping, methods=["GET"]),
        Route("/keyproxy/stats", stats, methods=["GET"]),
        Route("/keyproxy/shutdown", shutdown, methods=["POST"]),
        Route("/question/reserve", question("reserve"), methods=["POST"]),
        Route("/question/phase", question("phase"), methods=["POST"]),
        Route("/question/release", question("release"), methods=["POST"]),
        Route("/{path:path}", missing, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
    ], lifespan=lifespan)


# ----------------------------------------------------------------------------------------------- host files


def ensure_token(state_dir: Path) -> str:
    """The local bearer token in ``<state>/token``, created once (mode 600) and kept across restarts.

    A missing, empty or short file (a crash while it was first written) is replaced: the new token is written to a
    temporary file, synced and renamed into place, so the file is never seen empty.
    """
    path = state_dir / "token"
    with contextlib.suppress(OSError):
        value = path.read_text(encoding="utf-8").strip()
        if len(value) >= MIN_TOKEN_CHARS:
            return value
    temporary = state_dir / f".token.{os.getpid()}.{secrets.token_hex(4)}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(secrets.token_urlsafe(32) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()
    return path.read_text(encoding="utf-8").strip()


class StateLock:
    """The lifetime lock of one proxy's state directory: ``flock`` on POSIX, ``msvcrt.locking`` on Windows.

    The operating system drops it when the holding process dies, so a crash never leaves a stale lock.

    Args:
        path: ``<state>/keyproxy.lock``.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file: Any = None

    def acquire(self) -> bool:
        """Take the lock without waiting; False when another process holds it."""
        handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._file = handle
        return True

    def release(self) -> None:
        """Drop the lock (closing the file releases it)."""
        if self._file is None:
            return
        with contextlib.suppress(OSError):
            if os.name == "nt":
                import msvcrt
                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
        self._file.close()
        self._file = None


def lock_held(state_dir: Path) -> bool:
    """Whether a proxy (serving, starting or draining) holds ``state_dir``'s lifetime lock right now."""
    if not state_dir.is_dir():
        return False
    lock = StateLock(state_dir / LOCK_NAME)
    try:
        if not lock.acquire():
            return True
    except OSError:
        return False
    lock.release()
    return False


def summarize_ledger(path: Path) -> dict[str, Any]:
    """Totals of one ledger file: requests by status and key, waiting times, and token usage by served model.

    Args:
        path: ``<state>/ledger/<date>.jsonl``.

    Returns:
        ``{"rows", "by_status", "by_key", "wait_ms", "usage"}``; the usage sums each 200 row's counts per model.
    """
    by_status: Counter = Counter()
    by_key: dict[str, Counter] = {}
    usage: dict[str, Counter] = {}
    waits: list[int] = []
    rows = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rows += 1
            status = str(row.get("st"))
            by_status[status] += 1
            if row.get("key"):
                by_key.setdefault(row["key"], Counter())[status] += 1
            if "usage" in row:
                bucket = usage.setdefault(row.get("served") or row.get("model") or "unknown", Counter())
                bucket.update(row["usage"])
                bucket["calls"] += 1
            if row.get("st") == 200 or row.get("err", "").startswith("keyproxy_"):
                waits.append(int(row.get("wait_ms") or 0))
    waits.sort()
    return {"rows": rows, "by_status": dict(by_status), "by_key": {key: dict(value) for key, value in sorted(by_key.items())},
            "wait_ms": {"n": len(waits), "p50": waits[len(waits) // 2] if waits else None,
                        "p90": waits[int(0.9 * (len(waits) - 1))] if waits else None, "max": waits[-1] if waits else None},
            "usage": {model: dict(value) for model, value in sorted(usage.items())}}


def ledger_errors(state_dir: Path, since: float) -> list[dict[str, Any]]:
    """Provider refusals recorded in the ledgers since ``since``, shaped like :meth:`.key_broker.SqliteKeyBroker.errors`."""
    events = []
    directory = state_dir / "ledger"
    first = quota_day(since)
    for path in sorted(directory.glob("*.jsonl")) if directory.exists() else ():
        if path.stem < first:
            continue
        with path.open(encoding="utf-8") as source:
            for line in source:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("t", 0) >= since and "code" in row:
                    events.append({"at": row["t"], "status": row.get("st"), "code": row["code"], "type": row.get("type", ""),
                                   "message": row.get("msg", ""), "model": row.get("model") or "",
                                   "file_name": row.get("key", ""), "key_id": ""})
    return events


# ----------------------------------------------------------------------------------------------- CLI


def _key_paths(args: argparse.Namespace) -> tuple[Path, ...]:
    pool = args.pool or os.environ.get("RUBIKSQL_GATEWAY_POOL")
    if args.key_dir is None and pool:
        config = json.loads(Path(pool).expanduser().read_text(encoding="utf-8-sig"))
        if isinstance(config, dict) and "key_files" in config:
            return validate_files(config["key_files"])
    key_dir = (args.key_dir or configured_key_dir()).expanduser().absolute()
    return validate_files([str(key_dir / name) for name in configured_key_names(key_dir)])


def _state_dir(args: argparse.Namespace, paths: tuple[Path, ...]) -> Path:
    from .lent_keys import is_key_path
    state = Path(args.state or os.environ.get(STATE_ENV) or default_state_dir(paths[0].parent)).expanduser().absolute()
    if is_key_path(state) or state.resolve() == paths[0].parent.resolve() or paths[0].parent.resolve() in state.resolve().parents:
        raise SystemExit("The key proxy state must be outside the lent key directory.")
    return state


def _cap_file(args: argparse.Namespace, state: Path) -> CapFile:
    from .lent_keys import CAP_ENV, CAP_FILE_ENV
    default = args.per_key if args.per_key is not None else int(os.environ.get(CAP_ENV) or DEFAULT_PER_KEY)
    if not 0 <= default <= MAX_PER_KEY:
        raise SystemExit(f"--per-key must be an integer from 0 to {MAX_PER_KEY}.")
    return CapFile(Path(args.cap_file or os.environ.get(CAP_FILE_ENV) or state / "per-key.cap").expanduser(), default)


def _endpoint(args: argparse.Namespace, state: Path, *, serving: bool = False) -> Endpoint:
    """The proxy this command talks to: ``--host``/``--port`` when serving, else :data:`URL_ENV` when set.

    The token is :data:`TOKEN_FILE_ENV` when set, else ``<state>/token``. A configured URL is validated
    (:func:`origin`) before any token is sent to it.
    """
    token = Path(os.environ.get(TOKEN_FILE_ENV) or state / "token").expanduser()
    configured = (os.environ.get(URL_ENV) or "").strip().rstrip("/")
    if not serving and args.host is None and configured:
        return Endpoint(origin(configured), token)
    host = args.host or "127.0.0.1"
    if not private_host(host):
        raise _error("The key proxy is reached on loopback or a private (VPC) IP only.")
    return Endpoint(f"http://{host if ':' not in host else '[' + host + ']'}:{args.port}", token)


def _raise_open_files() -> None:
    """Let one process hold thousands of client and upstream sockets (POSIX soft limit up to the hard limit)."""
    try:
        import resource
    except ImportError:   # Windows has no per-process descriptor limit to raise
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
    if soft != resource.RLIM_INFINITY and soft < wanted:
        resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))


def listener(host: str, port: int) -> socket.socket:
    """The proxy's bound listening socket, with TCP keepalive that accepted connections inherit.

    A client host cut off without a FIN (a VPC blip, a hard kill) is then detected within about two minutes: its
    connection fails, the request counts as disconnected and its key slot is freed instead of being held for the
    rest of the run.
    """
    sock = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name != "nt":   # on Windows SO_REUSEADDR would let a second process bind the same port
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for name, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 15), ("TCP_KEEPCNT", 4)):
            if hasattr(socket, name):
                with contextlib.suppress(OSError):
                    sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, name), value)
        sock.bind((host, port))
    except BaseException:
        sock.close()
        raise
    return sock


def serve(args: argparse.Namespace) -> int:
    """Run the proxy in the foreground until ``POST /keyproxy/shutdown``, SIGTERM or SIGINT has drained.

    Returns:
        0 after a drained shutdown, 1 when the server could not start, 3 when another proxy holds this state
        directory or already answers on the port.
    """
    import uvicorn

    args.host = args.host or "127.0.0.1"
    if not private_host(args.host):
        raise SystemExit("The key proxy listens on loopback or a private (VPC) IP only, never a public or "
                         "unspecified address.")
    if os.name != "nt":
        os.umask(0o077)
    paths = _key_paths(args)
    state = _state_dir(args, paths)
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = StateLock(state / LOCK_NAME)
    if not lock.acquire():
        print(f"[keyproxy] Another key proxy holds {state / LOCK_NAME} (serving, starting or draining); not "
              "starting another.", file=sys.stderr)
        return 3
    try:
        endpoint = _endpoint(args, state, serving=True)
        with contextlib.suppress(OSError, ValueError):
            endpoint.health(timeout=1.0)
            print(f"[keyproxy] Port {args.port} already answers as a key proxy; not starting another.", file=sys.stderr)
            return 3
        _raise_open_files()
        marks = Marks(state / "marks.jsonl")
        pool = KeyPool(load_slots(paths, marks.load(time.time())), _cap_file(args, state), marks)
        book = QuestionBook(pool.capacity, override=args.question_capacity, warmup=args.question_warmup)
        proxy = KeyProxy(pool, ensure_token(state), state, max_wait=args.max_wait, ends_at=window_end(), book=book,
                         drain=args.drain)

        class Server(uvicorn.Server):
            def handle_exit(self, sig: int, frame: Any) -> None:
                # The first signal drains like ``stop`` (listener open, health ``closing``); a second one is
                # uvicorn's own: stop now (a second SIGINT forces).
                if not proxy.request_shutdown():
                    super().handle_exit(sig, frame)

        config = uvicorn.Config(build_app(proxy), host=args.host, port=args.port, log_level="warning",
                                access_log=False, timeout_keep_alive=75, backlog=8192,
                                timeout_graceful_shutdown=args.drain)
        server = Server(config)

        def end() -> None:
            server.config.timeout_graceful_shutdown = 5   # drained already: only idle connections remain
            server.should_exit = True
        proxy.on_shutdown = end
        sock = listener(args.host, args.port)
        runtime = state / "keyproxy.json"
        runtime.write_text(json.dumps({"pid": os.getpid(), "url": endpoint.url, "started_at": time.time()}) + "\n",
                           encoding="utf-8")
        summary = pool.summary()
        print(f"[keyproxy] Serving {endpoint.url}: {summary['usable_keys']} usable of {summary['keys']} keys, "
              f"{summary['retired']} retired, {summary['paused']} paused, per_key={summary['per_key']}.",
              file=sys.stderr, flush=True)
        try:
            server.run(sockets=[sock])
        finally:
            sock.close()
            with contextlib.suppress(OSError):
                runtime.unlink()
        return 0 if server.started else 1
    finally:
        lock.release()


def _serve_args(args: argparse.Namespace, state: Path) -> list[str]:
    """The options ``start`` hands to ``supervise`` and ``supervise`` to ``serve``."""
    options = ["--state", str(state), "--host", args.host, "--port", str(args.port), "--max-wait", str(args.max_wait),
               "--drain", str(args.drain), "--question-warmup", str(args.question_warmup)]
    for option, value in (("--key-dir", args.key_dir), ("--pool", args.pool), ("--cap-file", args.cap_file),
                          ("--per-key", args.per_key), ("--question-capacity", args.question_capacity)):
        if value is not None:
            options += [option, str(value)]
    return options


def supervise(args: argparse.Namespace) -> int:
    """Run ``serve`` as a child and restart it after a crash; ``start`` runs this detached.

    ``serve`` is restarted only when it had served for :data:`SUPERVISE_MIN_UP_S` and then died with an unexpected
    code (an OOM kill, an uncaught error), at most :data:`SUPERVISE_RESTARTS` times in :data:`SUPERVISE_WINDOW_S`.
    A drained shutdown (0), a refused start (2, 3), a quick failure or a stop signal to the supervisor ends it.

    Returns:
        The last ``serve`` exit code.
    """
    import signal

    paths = _key_paths(args)
    state = _state_dir(args, paths)
    args.host = args.host or "127.0.0.1"
    command = [sys.executable, "-m", "rubiksql_hb.keyproxy", "serve", *_serve_args(args, state)]
    child: subprocess.Popen | None = None
    stopping = False

    def forward(sig: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.send_signal(sig)
    if os.name != "nt":
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, forward)
    restarts: deque[float] = deque()
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    while True:
        started = time.monotonic()
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, **flags)
        code = child.wait()
        now = time.monotonic()
        if stopping or code in (0, 2, 3) or now - started < SUPERVISE_MIN_UP_S:
            return code
        while restarts and restarts[0] < now - SUPERVISE_WINDOW_S:
            restarts.popleft()
        if len(restarts) >= SUPERVISE_RESTARTS:
            print(f"[keyproxy] serve died {len(restarts) + 1} times within {SUPERVISE_WINDOW_S:g} s (last code "
                  f"{code}); not restarting.", file=sys.stderr, flush=True)
            return code
        restarts.append(now)
        print(f"[keyproxy] serve exited with code {code} after {now - started:.0f} s; restarting in 2 s.",
              file=sys.stderr, flush=True)
        time.sleep(2.0)


def start(args: argparse.Namespace) -> int:
    """Start the supervised proxy detached, logging to ``<state>/keyproxy.log``; return once health answers.

    Returns:
        0 when it started or a healthy proxy already answers, 1 when it failed or another proxy is still starting
        or draining on this state directory (wait for ``stop`` to finish).
    """
    paths = _key_paths(args)
    state = _state_dir(args, paths)
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    args.host = args.host or "127.0.0.1"
    endpoint = _endpoint(args, state, serving=True)
    try:
        health: dict[str, Any] | None = endpoint.health(timeout=1.0)
    except (OSError, ValueError):
        health = None
    if health is not None and health.get("status") == "closing":
        print(f"[keyproxy] The proxy at {endpoint.url} is draining ({health.get('in_flight')} in flight); wait for "
              "it to stop before starting another.", file=sys.stderr)
        return 1
    if health is not None:
        print(json.dumps({**health, "status": "already_running", "url": endpoint.url}))
        return 0
    if lock_held(state):
        print(f"[keyproxy] Another key proxy holds {state / LOCK_NAME} (starting or draining); not starting "
              "another.", file=sys.stderr)
        return 1
    command = [sys.executable, "-m", "rubiksql_hb.keyproxy", "supervise", *_serve_args(args, state)]
    flags = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
    with (state / "keyproxy.log").open("ab") as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, close_fds=True, **flags)
    deadline = time.monotonic() + args.ready_timeout
    while time.monotonic() < deadline:
        if child.poll() is not None:
            print(f"[keyproxy] serve exited with {child.returncode}; see {state / 'keyproxy.log'}.", file=sys.stderr)
            return 1
        with contextlib.suppress(OSError, ValueError):
            print(json.dumps({**endpoint.health(timeout=1.0), "status": "started", "url": endpoint.url,
                              "supervisor_pid": child.pid}))
            return 0
        time.sleep(0.2)
    print(f"[keyproxy] Not ready after {args.ready_timeout:g} s; see {state / 'keyproxy.log'}.", file=sys.stderr)
    return 1


def stop(args: argparse.Namespace) -> int:
    """Ask the running proxy to shut down and wait until it has drained and exited.

    The proxy keeps answering ``closing`` while in-flight requests drain, so "stopped" is printed only once health
    no longer answers and (on the proxy's own runner) its state lock is free: no paid request is still upstream.
    """
    paths = _key_paths(args)
    state = _state_dir(args, paths)
    endpoint = _endpoint(args, state)
    request = urllib.request.Request(endpoint.url + "/keyproxy/shutdown", data=b"", method="POST",
                                     headers={"Authorization": "Bearer " + endpoint.token})
    try:
        with endpoint.open(request, timeout=5.0) as response:
            print(response.read().decode("utf-8"), flush=True)
    except (OSError, ValueError):
        if not lock_held(state):
            print(json.dumps({"status": "not_running", "url": endpoint.url}))
            return 0
    deadline = time.monotonic() + args.drain + 30
    last_report = time.monotonic()
    while time.monotonic() < deadline:
        try:
            health: dict[str, Any] | None = endpoint.health(timeout=2.0)
        except (OSError, ValueError):
            health = None
        if health is None and not lock_held(state):
            print(json.dumps({"status": "stopped", "url": endpoint.url}))
            return 0
        if health is not None and time.monotonic() - last_report >= 30:
            last_report = time.monotonic()
            print(json.dumps({"status": "draining", "in_flight": health.get("in_flight"),
                              "draining_s": health.get("draining_s")}), flush=True)
        time.sleep(0.5)
    print(f"[keyproxy] Still running after {args.drain + 30:g} s.", file=sys.stderr)
    return 1


def _print_usage(args: argparse.Namespace, state: Path) -> int:
    """Token usage of one Asia/Shanghai day by model: the live proxy's counters, else that day's ledger here."""
    day = args.day or quota_day(time.time())
    endpoint = _endpoint(args, state)
    try:
        live = endpoint.call("/keyproxy/stats", timeout=10.0)
    except (OSError, ValueError, RuntimeError):
        live = None
    if live is not None and day in live.get("usage", {}):
        print(json.dumps({"day": day, "source": "live", "url": endpoint.url, "usage": live["usage"][day]}, indent=2))
        return 0
    path = state / "ledger" / f"{day}.jsonl"
    summary = summarize_ledger(path) if path.exists() else {"rows": 0, "usage": {}}
    print(json.dumps({"day": day, "source": "ledger" if path.exists() else "none", **summary}, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """``serve``, ``supervise``, ``start``, ``stop``, ``status``, ``stats``, ``usage`` and ``import-marks``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("serve", "supervise", "start", "stop", "status", "stats", "usage",
                                            "import-marks"))
    parser.add_argument("--state", type=Path, help=f"state directory (default {STATE_ENV} or beside the key directory)")
    parser.add_argument("--key-dir", type=Path, help="lent key directory (default from RUBIKSQL_GATEWAY_POOL, "
                                                     "else RUBIKSQL_LENT_KEY_DIR or the current usable-key directory)")
    parser.add_argument("--pool", type=Path, help="nonsecret lent pool JSON (default RUBIKSQL_GATEWAY_POOL)")
    parser.add_argument("--host", help="serve/start: loopback or private IP to bind (default 127.0.0.1); other "
                                       f"commands: the proxy to query (default {URL_ENV}, else 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--per-key", type=int, help="per-key cap while no cap file exists (default "
                                                    f"RUBIKSQL_KEY_MAX_INFLIGHT, else {DEFAULT_PER_KEY})")
    parser.add_argument("--cap-file", type=Path, help="live per-key cap file (default RUBIKSQL_KEY_CAP_FILE, "
                                                      "else <state>/per-key.cap)")
    parser.add_argument("--max-wait", type=float, default=DEFAULT_MAX_WAIT_S, help="longest in-proxy wait, seconds")
    parser.add_argument("--drain", type=int, default=int(DEFAULT_DRAIN_S),
                        help="seconds a shutdown waits for in-flight requests")
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    parser.add_argument("--question-capacity", type=int, help="serve: question-start slot capacity replacing the "
                                                              "keys' (eval processes' RUBIKSQL_QUESTION_START_CAPACITY wins)")
    parser.add_argument("--question-warmup", type=float, default=WARMUP_S,
                        help="serve: seconds after start before any question is admitted (running questions' owners "
                             "republish their reservations first)")
    parser.add_argument("--day", help="usage: the Asia/Shanghai date to summarize (default today)")
    parser.add_argument("--file", type=Path, help="import-marks: a key_broker export-marks JSON file")
    parser.add_argument("--from-broker", type=Path, help="import-marks: the host broker SQLite file (read-only)")
    parser.add_argument("--retire", nargs="*", default=(), help="import-marks: approved file names to retire")
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            return serve(args)
        if args.command == "supervise":
            return supervise(args)
        if args.command == "start":
            return start(args)
        if args.command == "stop":
            return stop(args)
        paths = _key_paths(args)
        state = _state_dir(args, paths)
        if args.command in ("status", "stats"):
            endpoint = _endpoint(args, state)
            try:
                payload = endpoint.health(timeout=5.0) if args.command == "status" else endpoint.call(
                    "/keyproxy/stats", timeout=10.0)
            except (OSError, ValueError, RuntimeError) as error:
                print(json.dumps({"status": "not_running", "url": endpoint.url, "error": type(error).__name__}))
                return 1
            print(json.dumps(payload, indent=2))
            return 0
        if args.command == "usage":
            return _print_usage(args, state)
        if any(name not in {path.name for path in paths} for name in args.retire):
            parser.error("--retire takes approved file names such as user-1.key")
        now = time.time()
        rows = [{"at": now, "key": name, "kind": "retired", "until": None} for name in args.retire]
        if args.file is not None:
            rows += marks_from_export(json.loads(args.file.read_text(encoding="utf-8")), now)
        if args.from_broker is not None:
            rows += broker_marks(args.from_broker, now)
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        Marks(state / "marks.jsonl").append(rows)
        restored = Marks(state / "marks.jsonl").load(now)
        print(json.dumps({"imported": len(rows), "retired": sorted(restored.retired), "paused": sorted(restored.paused),
                          "note": "a running proxy applies imported marks at its next start"}, indent=2))
        return 0
    except (OSError, ValueError, sqlite3.Error, RuntimeError) as error:
        parser.exit(2, f"[keyproxy] {args.command} failed ({type(error).__name__}); key contents withheld.\n")


if __name__ == "__main__":
    raise SystemExit(main())
