"""SDK request boundary for the lent file pool, including cancellable admission and backoff.

Without a key proxy every physical request leases a key from the host broker (:func:`acquire_lease`), sends with it and
releases it. With ``RUBIKSQL_KEYPROXY_URL`` set (:mod:`.keyproxy`) the request goes to the host's local key proxy
instead, which picks the key and applies the owner's 401/403/429 rules; this side then takes no SQLite lease at all and
keeps only the response-model verification, the launch-claim check and the window end.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from functools import partial
import logging
import math
import random
import time
from typing import Any

from . import provider_errors, session_timing
from .experiment_gateway import GatewayConfigurationError, GatewayLease, GatewayUnavailableError
from .gateway_chat import GatewayTransportError, _verify_response_model
from .prompt_cache import response_usage, wire_request
from .key_broker import file_key_id
from .lent_keys import FileLease, LentKeyPool, retry_delay

MAX_ATTEMPTS = 43  # One complete invalid/quota sweep plus eight transient retries.
_ACTIVE_LEASE: ContextVar[FileLease | None] = ContextVar("lent_wire_lease", default=None)
#: Broker transactions of the async path run on these threads, never on an event loop. One admission or release
#: is a host-wide SQLite write (``BEGIN IMMEDIATE``, FULL fsync) that can wait on every other process's writer; on
#: the loop that wait stalled every session sharing it, tool round trips included. Releases get their own threads
#: so a queue of admission attempts never delays the release that would free a slot.
_ADMISSION_EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="lent-admission")
_RELEASE_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="lent-release")


_LOGGER = logging.getLogger(__name__)


def _report_failed_release(release: Future) -> None:
    """Warn, without the failure's text, when a release thread could not finish its reservation."""
    if not release.cancelled() and release.exception() is not None:
        _LOGGER.warning("[lent-release] A lease release failed (%s); its reservation is held until its lifetime "
                        "expires. Details withheld.", type(release.exception()).__name__)


def _release(pool: LentKeyPool, token: str, status: int | None, retry_after: float = 0.0) -> Future | None:
    """Finish one reservation on a release thread; inline when the executor no longer accepts work.

    Returns:
        The release's future, or ``None`` when it already ran inline (interpreter shutdown).
    """
    try:
        release = _RELEASE_EXECUTOR.submit(pool.broker.finish, token, status, retry_after=retry_after)
    except RuntimeError:   # concurrent.futures stops accepting work at interpreter exit
        try:
            pool.broker.finish(token, status, retry_after=retry_after)
        except Exception as error:  # noqa: BLE001 - reported without provider or key text
            _LOGGER.warning("[lent-release] A lease release failed (%s); its reservation is held until its "
                            "lifetime expires. Details withheld.", type(error).__name__)
        return None
    release.add_done_callback(_report_failed_release)
    return release


def _release_abandoned(pool: LentKeyPool, attempt: Future) -> None:
    """Release a reservation whose admission attempt finished after its caller went away."""
    if attempt.cancelled() or attempt.exception() is not None:
        return
    lease = attempt.result()
    if lease is not None:
        _release(pool, lease.token, None)


async def acquire_lease(pool: LentKeyPool, *, ttl: float) -> FileLease:
    """Wait for one physical request's reservation without holding the event loop in SQLite.

    Each attempt is :meth:`LentKeyPool.try_acquire` on an admission thread, unchanged; between unsuccessful attempts
    the caller sleeps 0.5–1.5 s as before. All attempts of one admission run in one copy of the caller's context, so
    an observer that keeps per-admission state in a context variable sees one wait, as it did inline.

    Args:
        pool: The lent file pool.
        ttl: Request absolute deadline plus crash-recovery margin, in seconds.

    Returns:
        The reservation with its runtime-only credential.

    Raises:
        asyncio.CancelledError: The caller was cancelled. An attempt still queued is cancelled and never runs; one
            already running finishes on its thread and a reservation it makes is released there, so cancellation (or
            closing an abandoned task) never leaks a slot or counts a phantom request.
    """
    from .llm_runtime import refuse_if_stopped

    loop = asyncio.get_running_loop()
    context = copy_context()
    waiting = False
    try:
        while True:
            refuse_if_stopped()
            attempt = _ADMISSION_EXECUTOR.submit(context.run, partial(pool.try_acquire, ttl=ttl))
            try:
                lease = await asyncio.shield(asyncio.wrap_future(attempt, loop=loop))
            except BaseException:
                # Cancelled, or closed with its abandoned task (GeneratorExit): an attempt still queued never runs,
                # and one already running releases the reservation it makes.
                attempt.cancel()
                attempt.add_done_callback(partial(_release_abandoned, pool))
                raise
            if lease is not None:
                return lease
            if not waiting:
                # Counted in memory only; the host broker learns it from the counter's own publisher thread.
                waiting = True
                pool.waiting.change(+1)
            await asyncio.sleep(random.uniform(0.5, 1.5))
    finally:
        if waiting:
            pool.waiting.change(-1)


async def release_lease(pool: LentKeyPool, lease: FileLease, status: int | None, retry_after: float = 0.0) -> None:
    """Finish one reservation on a release thread; a cancelled caller returns early but the release still completes.

    Args:
        pool: The lent file pool.
        lease: The reservation to release.
        status: HTTP status, or ``None`` for a transport or cancellation failure.
        retry_after: Validated cooldown seconds for HTTP 429.
    """
    release = _release(pool, lease.token, status, retry_after)
    if release is not None:
        await asyncio.shield(asyncio.wrap_future(release, loop=asyncio.get_running_loop()))


class _Phases:
    """Time one physical request's admission, wire and release phases into the enclosing session timing.

    Admission runs from the start of the attempt (waiting for a key slot, then any configured pacing) until the
    request is sent; wire runs from send until the SDK returns or raises; release is the slot's return. Each phase is
    recorded once, also when the attempt fails or is cancelled part-way.

    Args:
        model: Requested model name.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._started = time.monotonic()
        self._sent: float | None = None
        self._wired = False

    def end_admission(self) -> None:
        """Close the admission phase (idempotent); the wire phase starts now."""
        if self._sent is None:
            self._sent = time.monotonic()
            session_timing.record_model_phase("admission", self.model, self._sent - self._started)

    def end_wire(self) -> None:
        """Close the wire phase (idempotent)."""
        if self._sent is not None and not self._wired:
            self._wired = True
            session_timing.record_model_phase("wire", self.model, time.monotonic() - self._sent)

    def release(self, elapsed: float) -> None:
        """Record the slot's release time."""
        session_timing.record_model_phase("release", self.model, elapsed)


class _SecretFilter(logging.Filter):
    """Suppress SDK debug bodies and redact credentials from adapter diagnostics in this context."""

    def filter(self, record: logging.LogRecord) -> bool:
        lease = _ACTIVE_LEASE.get()
        if lease is None:
            return True
        if record.levelno <= logging.DEBUG:
            return False
        record.msg = record.getMessage().replace(lease.secret, "[redacted]")
        record.args = ()
        if record.exc_info:
            record.exc_info = None
            record.exc_text = "provider exception details withheld"
        return True


_SECRET_FILTER = _SecretFilter()


def _failed(status: int | None, error: Exception) -> GatewayTransportError:
    """The sanitized error of a failed request, carrying the HTTP status the provider answered with.

    ``answered_status`` (a 4xx, or a 200 whose body is an error) marks a deterministic answer, which
    `llm_runtime._is_overloaded` does not count or retry as an unanswered request; only a request
    with no answer, or a 5xx, stays an overload.
    """
    failed = GatewayTransportError("Lent gateway request failed; provider details withheld.")
    # Not ``status_code``: account classification (experiment_gateway.credential_failure) reads that,
    # and this path deliberately leaves account handling to the broker's 401/403/429 rules.
    failed.answered_status = status if status is not None else getattr(error, "answered_status", None)
    from .experiment_gateway import credential_failure
    failed.provider_failure = (getattr(error, "provider_failure", False)
                               or failed.answered_status is None
                               or failed.answered_status in (401, 403, 429)
                               or credential_failure(error, status_code=status,
                                                     body=provider_errors.error_body(error)) is not None)
    return failed


def _filter_sdk_loggers() -> None:
    """Attach the credential filter to the SDK loggers that may render bodies at DEBUG, before our sanitizing."""
    names = {"openai._base_client", "openai._client", "openai._response", "httpx", "httpcore"}
    names.update(name for name in list(logging.Logger.manager.loggerDict)
                 if name.startswith(("openai.", "httpx.", "httpcore.", "hpack.", "h2.")))
    for name in names:
        logger = logging.getLogger(name)
        if _SECRET_FILTER not in logger.filters:
            logger.addFilter(_SECRET_FILTER)


def _replace_hooks(transport: Any, request_hook: Any, response_hook: Any | None) -> None:
    """Install this pool's request (and response) hook, replacing the ones an earlier install left on the transport."""
    hooks = transport.event_hooks
    previous = getattr(transport, "_rubiksql_file_hooks", ())
    for kind, hook in (("request", request_hook), ("response", response_hook)):
        kept = [item for item in hooks.get(kind, []) if item not in previous]
        hooks[kind] = kept + ([hook] if hook is not None else [])
    transport._rubiksql_file_hooks = tuple(hook for hook in (request_hook, response_hook) if hook is not None)


class _ProxyCall:
    """One proxied request: its model (sent as a header) and the seconds the key proxy reported waiting for a key."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.waited: float | None = None


_PROXY_CALL: ContextVar[_ProxyCall | None] = ContextVar("lent_proxy_call", default=None)


def _proxy_timeout(value: Any, wait: float) -> Any:
    """The SDK timeout of a proxied request: the caller's read budget plus the in-proxy waiting budget."""
    import httpx
    if isinstance(value, httpx.Timeout):
        return value if value.read is None else httpx.Timeout(
            connect=value.connect, read=value.read + wait, write=value.write, pool=value.pool)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return httpx.Timeout(float(value) + wait, connect=min(float(value), 10.0))
    return value


def _install_proxy(client: Any, recipe: Any, pool: LentKeyPool, *, asynchronous: bool, create: Any) -> None:
    """Route this SDK client's chat requests to the host's key proxy; no broker lease is taken on this side.

    The request hook rewrites only the approved gateway's chat route to the proxy's route and replaces the
    placeholder credential with the proxy token. The response hook reads how long the proxy waited for a key,
    recorded as the request's admission phase (the rest of the round trip is its wire phase). Every non-2xx answer,
    including the proxy's own 503 when no key could send, raises the sanitized transient failure that the caller's
    retry ladder already handles; nothing is fabricated. A proxy that cannot be connected to (restarting, or down)
    is retried for ``proxy.connect`` seconds with a short backoff; then the request fails the same way. It never
    falls back to a key outside the proxy.

    Raises:
        GatewayConfigurationError: The transport cannot carry hooks, a request targets another route, or the SDK
            transport would reach the loopback proxy through an environment HTTP proxy (add ``127.0.0.1`` to
            ``NO_PROXY``).
    """
    from .experiment_gateway import provider_claim
    from .keyproxy import WAIT_HEADER, connect_delay, connect_failed

    proxy = pool.proxy
    completions = client.chat.completions
    client.max_retries = 0
    transport = getattr(client, "_client", None)
    if not isinstance(getattr(transport, "event_hooks", None), dict):
        raise GatewayConfigurationError("The file-key SDK transport cannot enforce request-scoped authorization.")
    lease = FileLease("keyproxy", "keyproxy", proxy.token)   # lets the log filter redact the local token
    gateway = recipe.pool.base_url.rstrip("/") + "/chat/completions"
    import httpx
    target = httpx.URL(proxy.chat_url)
    routed = getattr(transport, "_transport_for_url", None)
    if callable(routed) and routed(target) is not getattr(transport, "_transport", None):
        raise GatewayConfigurationError("The SDK would reach the key proxy through an HTTP proxy; add 127.0.0.1 to NO_PROXY.")

    def route(request: Any) -> None:
        if str(request.url.copy_with(query=None)) != gateway:
            raise GatewayConfigurationError("The key proxy carries only the approved gateway's chat route.")
        pool.check_window()
        call = _PROXY_CALL.get()
        request.url = target
        request.headers["Host"] = target.netloc.decode("ascii")
        request.headers.update(proxy.headers(call.model if call is not None else None))

    def observe(response: Any) -> None:
        call = _PROXY_CALL.get()
        waited = response.headers.get(WAIT_HEADER)
        if call is not None and waited is not None and waited.isdigit():
            call.waited = int(waited) / 1000

    async def async_route(request: Any) -> None:
        route(request)

    async def async_observe(response: Any) -> None:
        observe(response)

    _replace_hooks(transport, async_route if asynchronous else route, async_observe if asynchronous else observe)

    def prepare(kwargs: dict[str, Any]) -> dict[str, Any]:
        if kwargs.get("stream"):
            raise GatewayConfigurationError("Lent file-key chat requires request/response mode.")
        pool.check_window()
        provider_claim()
        return {**wire_request(kwargs, recipe.model),
                "timeout": _proxy_timeout(kwargs.get("timeout", getattr(client, "timeout", None)), proxy.wait)}

    def settle(response: Any, model: str) -> Any:
        _verify_response_model(response, model, secret=proxy.token)
        if (getattr(response, "model_extra", None) or {}).get("error"):
            refused = GatewayTransportError("Gateway returned an unsuccessful response; provider details withheld.")
            refused.answered_status = 200   # an answer, not an overload (see _failed)
            refused.provider_failure = True
            raise refused
        return response_usage(response)

    def failed(error: Exception) -> Exception:
        status = getattr(error, "status_code", None)
        body = provider_errors.error_body(error)
        if provider_errors.model_refusal(status, body):
            return GatewayConfigurationError("Gateway refused the configured model; credentials were not retired.")
        if isinstance(body, dict) and isinstance(body.get("error"), dict):
            body = body["error"]   # the SDK may or may not have unwrapped the error envelope already
        if status == 401 and isinstance(body, dict) and body.get("code") == "keyproxy_unauthorized":
            return GatewayConfigurationError("The key proxy refused this host's token; copy the proxy's current token "
                                             "file and restart the client.")
        if connect_failed(error):
            unreachable = GatewayTransportError(f"Key proxy unreachable for {proxy.connect:g} s; no key outside the "
                                                "proxy is used.")
            unreachable.answered_status = None
            unreachable.provider_failure = True
            return unreachable
        return _failed(status, error)

    class _Reconnect:
        """The bounded reconnect budget of one request (``proxy.connect`` seconds from its first failure)."""

        def __init__(self) -> None:
            self.attempt = 0
            self.until: float | None = None

        def delay(self, error: Exception) -> float | None:
            """Seconds to wait before sending again, or ``None`` when the error is final."""
            if not connect_failed(error):
                return None
            now = time.monotonic()
            self.until = self.until if self.until is not None else now + proxy.connect
            if now >= self.until:
                return None
            if self.attempt == 0:
                _LOGGER.warning("[keyproxy] Cannot connect to the key proxy at %s; retrying for up to %g s.",
                                proxy.url, proxy.connect)
            delay = min(connect_delay(self.attempt), self.until - now)
            self.attempt += 1
            return delay

    def phases(model: str, started: float, call: _ProxyCall) -> None:
        elapsed = time.monotonic() - started
        waited = min(elapsed, call.waited or 0.0)
        session_timing.record_model_phase("admission", model, waited)
        session_timing.record_model_phase("wire", model, elapsed - waited)
        session_timing.record_model_phase("release", model, 0.0)

    def synchronous_create(**kwargs: Any) -> Any:
        from .llm_runtime import refuse_if_stopped
        refuse_if_stopped()
        model = str(kwargs.get("model") or recipe.model)
        call, started, reconnect = _ProxyCall(model), time.monotonic(), _Reconnect()
        tokens = _PROXY_CALL.set(call), _ACTIVE_LEASE.set(lease)
        try:
            while True:
                try:
                    response = create(**prepare(kwargs))
                    break
                except (GatewayConfigurationError, GatewayUnavailableError):
                    raise
                except Exception as error:
                    delay = reconnect.delay(error)
                    if delay is None:
                        raise failed(error) from None
                pool.backoff(delay)
        finally:
            _PROXY_CALL.reset(tokens[0])
            _ACTIVE_LEASE.reset(tokens[1])
            phases(model, started, call)
        return settle(response, model)

    async def asynchronous_create(**kwargs: Any) -> Any:
        from .llm_runtime import refuse_if_stopped
        refuse_if_stopped()
        model = str(kwargs.get("model") or recipe.model)
        call, started, reconnect = _ProxyCall(model), time.monotonic(), _Reconnect()
        tokens = _PROXY_CALL.set(call), _ACTIVE_LEASE.set(lease)
        try:
            while True:
                try:
                    response = await create(**prepare(kwargs))
                    break
                except (GatewayConfigurationError, GatewayUnavailableError):
                    raise
                except Exception as error:
                    delay = reconnect.delay(error)
                    if delay is None:
                        raise failed(error) from None
                await pool.abackoff(delay)
        finally:
            _PROXY_CALL.reset(tokens[0])
            _ACTIVE_LEASE.reset(tokens[1])
            phases(model, started, call)
        return settle(response, model)

    completions.create = asynchronous_create if asynchronous else synchronous_create


def install(client: Any, recipe: Any, *, asynchronous: bool) -> None:
    """Wrap the SDK's create boundary; each physical request owns one expiring lease, or goes to the key proxy.

    With the pool's key proxy configured (``pool.proxy``) the wrapper is :func:`_install_proxy`'s instead.

    Args:
        client: Request-local SDK client, with redirects already disabled.
        recipe: Approved runtime binding with a file-backed owner.
        asynchronous: Whether create returns an awaitable.
    """
    from .llm_runtime import _CHAT_DEADLINE_S, refuse_if_stopped

    owner = recipe.pool.owner
    pool: LentKeyPool = owner.file_keys
    completions = client.chat.completions
    create = getattr(completions, "_rubiksql_file_create", None)
    if create is None:
        create = completions.create
        completions._rubiksql_file_create = create
    client.max_retries = 0
    timeout = getattr(client, "timeout", None)
    timeouts = [getattr(timeout, name, None) for name in ("connect", "read", "write", "pool")]
    finite_timeouts = [float(value) for value in timeouts if isinstance(value, (int, float)) and math.isfinite(value)]
    lease_ttl = max([_CHAT_DEADLINE_S, *finite_timeouts]) + 60
    # These SDK loggers may render response bodies at DEBUG, before our boundary sanitizes exceptions.
    _filter_sdk_loggers()
    if pool.proxy is not None:
        _install_proxy(client, recipe, pool, asynchronous=asynchronous, create=create)
        return
    transport = getattr(client, "_client", None)
    hooks = getattr(transport, "event_hooks", None)
    if not isinstance(hooks, dict):
        raise GatewayConfigurationError("The file-key SDK transport cannot enforce request-scoped authorization.")

    def authorize(request: Any) -> None:
        active = _ACTIVE_LEASE.get()
        if active is None:
            raise GatewayConfigurationError("A physical file-key request has no active lease.")
        pool.check_window()
        request.headers["Authorization"] = "Bearer " + active.secret

    async def async_authorize(request: Any) -> None:
        authorize(request)

    _replace_hooks(transport, async_authorize if asynchronous else authorize, None)

    def prepare(kwargs: dict[str, Any]) -> dict[str, Any]:
        pool.check_window()
        # Inject authorization at the HTTP hook, after SDK option diagnostics. No key enters SDK options.
        return wire_request(kwargs, recipe.model)

    def report(status: int | None, body: Any, kwargs: dict[str, Any], lease: FileLease) -> None:
        # Sanitized: status, provider code/type/message (redacted), model, file name, opaque id; never the request.
        provider_errors.report(pool.broker, provider_errors.event(
            status=status, body=body, model=str(kwargs.get("model") or recipe.model), file_name=lease.file_name,
            key_id=file_key_id(lease.file_name), secret=lease.secret))

    def verify(response: Any, kwargs: dict[str, Any], lease: FileLease) -> None:
        _verify_response_model(response, str(kwargs.get("model") or recipe.model), secret=lease.secret)
        extra = getattr(response, "model_extra", None) or {}
        if extra.get("error"):
            report(200, extra, kwargs, lease)
            refused = GatewayTransportError("Gateway returned an unsuccessful response; provider details withheld.")
            refused.answered_status = 200   # an answer, not an overload (see _failed)
            refused.provider_failure = True
            raise refused

    def before_request(lease: FileLease, kwargs: dict[str, Any]) -> None:
        admitted = GatewayLease(pool.names.index(lease.file_name), lease.secret, recipe.pool.base_url, recipe.experiment_id)
        recipe.pool.before_request(admitted, recipe.model, kwargs)

    def failure(error: Exception, kwargs: dict[str, Any], lease: FileLease) -> tuple[int | None, float]:
        status = getattr(error, "status_code", None)
        if provider_errors.model_refusal(status, provider_errors.error_body(error)):
            raise GatewayConfigurationError("Gateway refused the configured model; credentials were not retired.") from None
        response = getattr(error, "response", None)
        delay = retry_delay(getattr(response, "headers", None)) if status == 429 else 0.0
        if isinstance(status, int) and status >= 400:
            report(status, provider_errors.error_body(error), kwargs, lease)
        return status, delay

    def exhausted() -> GatewayUnavailableError:
        return GatewayUnavailableError(recipe.experiment_id, pool.broker.path)

    def synchronous_create(**kwargs: Any) -> Any:
        # Project production chat is request/response. Fail closed instead of dropping a lease before a stream ends.
        if kwargs.get("stream"):
            raise GatewayConfigurationError("Lent file-key chat requires request/response mode.")
        model = str(kwargs.get("model") or recipe.model)
        for _ in range(MAX_ATTEMPTS):
            refuse_if_stopped()
            phase = _Phases(model)
            try:
                lease = pool.acquire(ttl=lease_ttl)
            except BaseException:
                phase.end_admission()
                raise
            active = _ACTIVE_LEASE.set(lease)
            status, delay = None, 0.0
            try:
                before_request(lease, kwargs)
                phase.end_admission()
                try:
                    response = create(**prepare(kwargs))
                finally:
                    phase.end_wire()
                status = 200
                verify(response, kwargs, lease)
                return response_usage(response)
            except (GatewayConfigurationError, GatewayUnavailableError):
                raise
            except Exception as error:
                status, delay = failure(error, kwargs, lease)
                if status not in (401, 403, 429):
                    raise _failed(status, error) from None
            finally:
                phase.end_admission()
                _ACTIVE_LEASE.reset(active)
                released = time.monotonic()
                try:
                    pool.broker.finish(lease.token, status, retry_after=delay)
                finally:
                    phase.release(time.monotonic() - released)
            if delay:
                pool.backoff(delay)
        raise exhausted()

    async def asynchronous_create(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            raise GatewayConfigurationError("Lent file-key chat requires request/response mode.")
        model = str(kwargs.get("model") or recipe.model)
        for _ in range(MAX_ATTEMPTS):
            phase = _Phases(model)
            try:
                lease = await acquire_lease(pool, ttl=lease_ttl)
            except BaseException:
                phase.end_admission()
                raise
            active = _ACTIVE_LEASE.set(lease)
            status, delay = None, 0.0
            try:
                # This route normally has no pacing; execute configured pacing outside the event loop.
                await asyncio.to_thread(before_request, lease, kwargs)
                phase.end_admission()
                try:
                    response = await create(**prepare(kwargs))
                finally:
                    phase.end_wire()
                status = 200
                verify(response, kwargs, lease)
                return response_usage(response)
            except (GatewayConfigurationError, GatewayUnavailableError):
                raise
            except Exception as error:
                status, delay = failure(error, kwargs, lease)
                if status not in (401, 403, 429):
                    raise _failed(status, error) from None
            finally:
                phase.end_admission()
                _ACTIVE_LEASE.reset(active)
                released = time.monotonic()
                try:
                    await release_lease(pool, lease, status, delay)
                finally:
                    phase.release(time.monotonic() - released)
            if delay:
                await pool.abackoff(delay)
        raise exhausted()

    completions.create = asynchronous_create if asynchronous else synchronous_create
