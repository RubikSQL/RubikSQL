"""Synchronous chat transport with shared experiment account rotation."""
from __future__ import annotations

from .provider_errors import model_refusal

from contextlib import contextmanager
import io
import itertools
import json
import logging
import os
import socket
import ssl
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, Iterator, Mapping
import urllib.error
import urllib.request

from .experiment_gateway import (
    DEEPSEEK_CANDIDATE_MODEL,
    DEEPSEEK_CANDIDATE_SERVED_MODEL,
    GatewayConfigurationError,
    GatewayPool,
    GatewayUnavailableError,
    credential_failure,
    require_deepseek_model,
)

GATEWAY_ERRORS = (GatewayConfigurationError, GatewayUnavailableError)
_LOGGER = logging.getLogger(__name__)
_MISSING_MODEL = object()
_TLS_LOCK = threading.Lock()
_TLS_CACHE: tuple[tuple[Any, ...], ssl.SSLContext] | None = None
_TRANSPORT = threading.local()
_HTTP_LOCK = threading.Lock()
_HTTP_IN_FLIGHT = 0
_HTTP_SEQUENCE = 0


def _reset_transport_after_fork() -> None:
    global _TLS_LOCK, _TLS_CACHE, _TRANSPORT, _HTTP_LOCK, _HTTP_IN_FLIGHT, _HTTP_SEQUENCE
    _TLS_LOCK = threading.Lock()
    _TLS_CACHE = None
    _TRANSPORT = threading.local()
    _HTTP_LOCK = threading.Lock()
    _HTTP_IN_FLIGHT = _HTTP_SEQUENCE = 0


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_transport_after_fork)


def _verify_response_model(response: Any, requested: str, *, secret: str) -> None:
    """Refuse a reported model substitution before callers consume its answer.

    Older endpoints omit ``model`` entirely. SDK completion objects can also expose
    an unset field as ``None``, so use their field-presence metadata when available.
    Only model identifiers enter diagnostics; provider bodies and credentials do not.
    """
    if isinstance(response, Mapping):
        served = response.get("model", _MISSING_MODEL)
    else:
        fields = getattr(response, "model_fields_set", None)
        if fields is not None and "model" not in fields:
            return
        served = getattr(response, "model", _MISSING_MODEL)
    if served is _MISSING_MODEL:
        return

    def label(value: Any) -> str:
        if not isinstance(value, str):
            return "<invalid model id>"
        return repr((value.replace(secret, "[redacted]") if secret else value)[:256])

    # The October alias alone may be answered by its pinned dated build. Every
    # other request, the September DeepSeek id included, keeps exact identity checking.
    expected = DEEPSEEK_CANDIDATE_SERVED_MODEL if requested == DEEPSEEK_CANDIDATE_MODEL else requested
    if served != expected:
        message = f"Gateway served model mismatch: requested={label(requested)}; served={label(served)}."
        _LOGGER.error(message)
        raise GatewayConfigurationError(message) from None
    _LOGGER.debug("Gateway model verified: requested=%s; served=%s.", label(requested), label(served))


class GatewayTransportError(ConnectionError):
    """A retryable transport failure whose message contains no raw request or credential."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "gateway redirects are forbidden", headers, fp)


def _opener() -> urllib.request.OpenerDirector:
    """Reuse verified TLS state and thread-owned handlers, retaining urllib proxy discovery."""
    global _TLS_CACHE
    tls_key = (os.getpid(), *(os.environ.get(name) for name in ("SSL_CERT_FILE", "SSL_CERT_DIR", "SSLKEYLOGFILE")))
    cached_tls = _TLS_CACHE
    if cached_tls is None or cached_tls[0] != tls_key:
        with _TLS_LOCK:
            cached_tls = _TLS_CACHE
            if cached_tls is None or cached_tls[0] != tls_key:
                context = ssl.create_default_context()
                # Match the default urllib/http.client HTTPS context on Python 3.11–3.13.
                context.set_alpn_protocols(["http/1.1"])
                if context.post_handshake_auth is not None:
                    context.post_handshake_auth = True
                cached_tls = _TLS_CACHE = (tls_key, context)
    context = cached_tls[1]
    proxies = urllib.request.getproxies()
    key = (tls_key, tuple(sorted(proxies.items())))
    cached = getattr(_TRANSPORT, "opener", None)
    if cached is None or cached[0] != key:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler(proxies), urllib.request.HTTPSHandler(context=context), _NoRedirect(),
        )
        _TRANSPORT.opener = (key, opener)
    return _TRANSPORT.opener[1]


def _http_event(event: str, attempt: int, in_flight: int, **fields: Any) -> None:
    print("[gateway-http] " + json.dumps({
        "event": event, "attempt": attempt, "pid": os.getpid(), "tid": threading.get_ident(),
        "in_flight": in_flight, **fields,
    }), file=sys.stderr, flush=True)


@contextmanager
def _open(request: urllib.request.Request, *, timeout: float) -> Iterator[Any]:
    """Open/read one request without sharing credentials or forwarding them on redirects."""
    global _HTTP_IN_FLIGHT, _HTTP_SEQUENCE
    opener = _opener()
    diagnostic = os.environ.get("RUBIKSQL_GATEWAY_HTTP_DIAGNOSTICS") == "1"
    started = time.perf_counter() if diagnostic else 0.0
    attempt = 0
    if diagnostic:
        with _HTTP_LOCK:
            _HTTP_SEQUENCE += 1
            attempt = _HTTP_SEQUENCE
            _HTTP_IN_FLIGHT += 1
            active = _HTTP_IN_FLIGHT
    status = None
    outcome = "response"
    try:
        if diagnostic:
            _http_event("http_start", attempt, active)
        try:
            with opener.open(request, timeout=timeout) as response:
                status = response.status
                yield response
        except urllib.error.HTTPError as error:
            status = error.code
            outcome = "http_error"
            # Keep the bounded error-body read inside the wire attempt; complete() still
            # classifies/sanitizes it after transport accounting has finished.
            try:
                raw = error.read(65536)
            except Exception:
                raw = b""
            finally:
                error.close()
            raise urllib.error.HTTPError(error.url, error.code, error.msg, error.headers, io.BytesIO(raw)) from None
        except (TimeoutError, socket.timeout):
            outcome = "timeout"
            raise
        except Exception:
            outcome = "transport_error"
            raise
    finally:
        if diagnostic:
            elapsed = time.perf_counter() - started
            with _HTTP_LOCK:
                _HTTP_IN_FLIGHT -= 1
                active = _HTTP_IN_FLIGHT
            _http_event("http_end", attempt, active, elapsed_sec=round(elapsed, 6), status=status, outcome=outcome)


def complete(body: Mapping[str, Any], *, timeout: float = 120.0) -> dict[str, Any]:
    """Make one chat request, moving to another account after an account-level rejection.

    A rejection ``experiment_gateway.credential_failure`` classifies (HTTP 401, 402 or 403, or
    account wording such as arrears or a suspended account) retires that account and retries the
    same payload on the experiment's or database group's next account.

    Args:
        body: Original OpenAI-compatible payload, including vendor request fields. Its ``model`` is a project
            id (a Qwen id or ``experiment_gateway.DEEPSEEK_MODEL``); the request sends the selected route's id
            for it (``ProviderPool.model_id``) and every other field unchanged.
        timeout: Per-attempt socket timeout. Callers retain their transient retry policy.

    Returns:
        The unmodified decoded response, including usage.

    Raises:
        GatewayConfigurationError: The pool or experiment identity is unavailable, or the response
            reports a model other than the requested route model.
        GatewayUnavailableError: Every configured account is unavailable; its checkpoint is retained.
        urllib.error.HTTPError: A sanitized noncredential HTTP failure for the caller's retry policy.
        GatewayTransportError: A sanitized connection, timeout or response-decoding failure.
    """
    model = str(body.get("model") or "")
    if model.startswith("deepseek") or model == DEEPSEEK_CANDIDATE_MODEL:
        require_deepseek_model(model, candidate=True)
        provider = "deepseek"
    elif model.startswith("qwen"):
        provider = "qwen"
    else:
        raise GatewayConfigurationError("Chat model does not belong to an approved project provider.")
    from .prompt_cache import wire_request

    # Validate serialization before leasing, as before; an invalid body must not
    # advance the account allocation. Cache hints are added only after routing.
    payload = json.dumps({**body, "model": model}, ensure_ascii=False).encode("utf-8")
    pool, lease = GatewayPool.from_env().lease_provider(provider)
    wire_model = pool.model_id(model)
    wire = wire_request({**body, "model": wire_model})
    if wire_model != model or wire.get("messages") is not body.get("messages"):
        payload = json.dumps(wire, ensure_ascii=False).encode("utf-8")
    if getattr(getattr(pool, "owner", None), "file_keys", None) is not None:
        return _complete_files(pool, wire_model, body, payload, timeout=timeout)
    while True:
        pool.before_request(lease, wire_model, body)
        request = urllib.request.Request(
            lease.base_url.rstrip("/") + "/chat/completions", data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {lease.secret}"},
        )
        try:
            with _open(request, timeout=timeout) as response:
                status = response.status
                raw = response.read()
            try:
                result = json.loads(raw)
            except ValueError:
                result = raw
            _verify_response_model(result, wire_model, secret=lease.secret)
            if model_refusal(status, result):
                raise GatewayConfigurationError("Gateway refused the configured model; credentials were not retired.") from None
            # Classify the provider answer before decoding/MIME failures or envelope
            # normalization can hide it. Decoded completions keep answer text opaque.
            reason = credential_failure(status_code=status, body=result)
            if reason:
                try:
                    pool.rotate(
                        lease.experiment_id, lease.index, reason,
                        failed_revision=lease.credential_revision,
                        **({"group_id": lease.group_id} if lease.group_id is not None else {}),
                    )
                except GATEWAY_ERRORS as blocked:
                    raise blocked from None
                lease = pool.lease()
                continue
            if not isinstance(result, dict):
                raise ValueError("chat response must be an object")
            if result.get("error"):
                raise GatewayTransportError("gateway returned an unsuccessful response")
            return result
        except urllib.error.HTTPError as error:
            try:
                raw = error.read(65536).decode("utf-8", "replace")
            except Exception:
                raw = ""
            try:
                result = json.loads(raw)
            except ValueError:
                result = raw
            _verify_response_model(result, wire_model, secret=lease.secret)
            if model_refusal(error.code, result):
                raise GatewayConfigurationError("Gateway refused the configured model; credentials were not retired.") from None
            reason = credential_failure(error, status_code=error.code, body=raw)
            if reason:
                try:
                    pool.rotate(
                        lease.experiment_id, lease.index, reason,
                        failed_revision=lease.credential_revision,
                        **({"group_id": lease.group_id} if lease.group_id is not None else {}),
                    )
                except GATEWAY_ERRORS as blocked:
                    raise blocked from None
                lease = pool.lease()
                continue
            # Preserve only known nonsecret parameter names needed by rulegen's compatibility retry.
            params = [name for name in ("temperature", "top_p", "seed") if name in raw] if 400 <= error.code < 500 else []
            detail = ("gateway rate limit" if error.code == 429 else "gateway request failed") + ("; unsupported parameter " + ", ".join(params) if params else "")
            headers = {}
            retry_after = (error.headers or {}).get("Retry-After", "")
            if retry_after.replace(".", "", 1).isdigit():
                headers["Retry-After"] = retry_after
            safe = urllib.error.HTTPError(pool.base_url.rstrip("/") + "/chat/completions", error.code, detail, headers,
                                          io.BytesIO(detail.encode("utf-8")))
            raise safe from None
        except (TimeoutError, socket.timeout):
            raise GatewayTransportError("gateway request timed out") from None
        except (OSError, ValueError):
            raise GatewayTransportError("gateway connection or response decoding failed") from None


def _report_file_error(keys: Any, lease: Any, status: int | None, body: Any, model: str) -> None:
    """Record a sanitized provider error answer of the lent urllib route (:mod:`.provider_errors`)."""
    from . import provider_errors
    from .key_broker import file_key_id
    provider_errors.report(keys.broker, provider_errors.event(status=status, body=body, model=model,
        file_name=lease.file_name, key_id=file_key_id(lease.file_name), secret=lease.secret))


def _complete_files(pool: Any, model: str, body: Mapping[str, Any], payload: bytes, *, timeout: float) -> dict[str, Any]:
    """Keep one expiring reservation per urllib request and apply the file owner's exact status rules."""
    from .lent_chat import MAX_ATTEMPTS
    from .lent_keys import retry_delay
    from .experiment_gateway import GatewayLease
    keys = pool.owner.file_keys
    if keys.proxy is not None:
        return _complete_proxy(keys, model, payload, timeout=timeout)
    for _ in range(MAX_ATTEMPTS):
        lease = keys.acquire(ttl=timeout + 60)
        status, delay = None, 0.0
        try:
            admitted = GatewayLease(keys.names.index(lease.file_name), lease.secret, pool.base_url, pool.owner.experiment_id or "file-key-request")
            pool.before_request(admitted, model, body)
            keys.check_window()
            request = urllib.request.Request(pool.base_url.rstrip("/") + "/chat/completions", data=payload,
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + lease.secret})
            with _open(request, timeout=timeout) as response:
                status = response.status
                if status == 429:
                    delay = retry_delay(response.headers)
                raw = response.read()
            if status not in (401, 403, 429):
                result = json.loads(raw)
                _verify_response_model(result, model, secret=lease.secret)
                if model_refusal(status, result):
                    raise GatewayConfigurationError("Gateway refused the configured model; credentials were not retired.")
                if not isinstance(result, dict) or result.get("error") or not 200 <= status < 300:
                    _report_file_error(keys, lease, status, result, model)
                    raise GatewayTransportError("Gateway returned an unsuccessful response; details withheld.")
                return result
            _report_file_error(keys, lease, status, raw, model)
        except urllib.error.HTTPError as error:
            status = error.code
            if status == 429:
                delay = retry_delay(error.headers)
            try:
                body = error.read(65536)
            except (OSError, ValueError):
                body = b""
            error.close()
            _report_file_error(keys, lease, status, body, model)
            if model_refusal(status, body):
                raise GatewayConfigurationError("Gateway refused the configured model; credentials were not retired.") from None
            if status not in (401, 403, 429):
                raise GatewayTransportError("Lent gateway HTTP request failed; details withheld.") from None
        except (GatewayConfigurationError, GatewayUnavailableError):
            raise
        except (OSError, ValueError):
            raise GatewayTransportError("Lent gateway connection or decoding failed; details withheld.") from None
        finally:
            keys.broker.finish(lease.token, status, retry_after=delay)
        if delay:
            keys.backoff(delay)
    raise GatewayUnavailableError(pool.owner.experiment_id or "file-key-request", keys.broker.path)


def _complete_proxy(keys: Any, model: str, payload: bytes, *, timeout: float) -> dict[str, Any]:
    """Send one urllib request through the key proxy (:mod:`.keyproxy`), which picks the key.

    The proxy applies the owner's 401/403/429 rules and waits for a free key; this side keeps the window end, the
    launch-claim check and the response-model verification. Any non-2xx answer, including the proxy's own 503 when no
    key could send, is a sanitized transient failure for the caller's retry policy. A proxy that cannot be connected to
    is retried for ``proxy.connect`` seconds, then the request fails the same way; no key outside the proxy is used.
    """
    from .experiment_gateway import provider_claim
    from .keyproxy import connect_delay, connect_failed
    from .llm_runtime import refuse_if_stopped
    proxy = keys.proxy
    until: float | None = None
    for attempt in itertools.count():
        keys.check_window()
        provider_claim()
        request = urllib.request.Request(proxy.chat_url, data=payload,
                                         headers={"Content-Type": "application/json", **proxy.headers(model)})
        try:
            with proxy.open(request, timeout=timeout + proxy.wait) as response:
                raw = response.read()
            result = json.loads(raw)
            break
        except urllib.error.HTTPError as error:
            try:
                body = error.read(65536)
            except (OSError, ValueError):
                body = b""
            error.close()
            if error.code == 401 and b"keyproxy_unauthorized" in body:
                raise GatewayConfigurationError("The key proxy refused this host's token.") from None
            if model_refusal(error.code, body):
                raise GatewayConfigurationError("Gateway refused the configured model; credentials were not retired.") from None
            raise GatewayTransportError("Key proxy request failed; details withheld.") from None
        except (OSError, ValueError) as error:
            now = time.monotonic()
            until = until if until is not None else now + proxy.connect
            if not connect_failed(error):
                raise GatewayTransportError("Key proxy connection or decoding failed; details withheld.") from None
            if now >= until:
                raise GatewayTransportError(f"Key proxy unreachable for {proxy.connect:g} s; no key outside the "
                                            "proxy is used.") from None
            delay = min(connect_delay(attempt), until - now)
            while delay > 0:
                refuse_if_stopped()
                time.sleep(min(delay, 1.0))
                delay -= 1.0
    _verify_response_model(result, model, secret=proxy.token)
    if not isinstance(result, dict) or result.get("error"):
        raise GatewayTransportError("Gateway returned an unsuccessful response; details withheld.")
    return result


def _objects(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{name: _objects(item) for name, item in value.items()})
    if isinstance(value, list):
        return [_objects(item) for item in value]
    return value


class GatewayChatClient:
    """The small SDK-shaped surface used by profiling, with no startup credential retained.

    Args:
        timeout: Per-request socket timeout; transient retries remain in the profiling caller.
    """

    def __init__(self, *, timeout: float = 120.0) -> None:
        self.timeout = timeout

    @property
    def chat(self) -> GatewayChatClient:
        """Expose the profiling caller's ``client.chat.completions.create`` shape."""
        return self

    @property
    def completions(self) -> GatewayChatClient:
        """Expose chat completion creation without constructing an SDK credential cache."""
        return self

    def create(self, **payload: Any) -> Any:
        """Forward all request fields and response fields through the pool transport.

        Args:
            **payload: SDK-shaped completion arguments; ``extra_body`` adds vendor fields.

        Returns:
            Attribute-accessible response fields, preserving choices and usage.
        """
        extra = payload.pop("extra_body", None) or {}
        return _objects(complete({**payload, **extra}, timeout=self.timeout))
