"""HeavenBase-backed chat and embedding transport for RubikSQL."""

from __future__ import annotations

import atexit
import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor
import random
import re
from collections.abc import AsyncIterator, Callable, Iterator
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field, replace
import http.client
import json
import os
import threading
import time
import urllib.error
from typing import Any, Literal

import heavenbase as hb
from heavenbase.utils import dumps_json, raise_mismatch
# Deliberate private import: the dmerge overwrite sentinel has no public alias in
# HeavenBase 0.1.2.2. Importing the constant (instead of hardcoding "__HB_OVERWRITE__")
# makes an upstream rename fail here at import time rather than silently reopening the
# machine-config leak the sentinel closes.
from heavenbase.utils.config.data import _OVERWRITE as _HB_OVERWRITE
# Deliberate private import (same rationale as _OVERWRITE above): per-authority client
# disposal is the exact teardown the watchdog needs and has no public alias in 0.1.2.3.
from heavenbase.llm.gateways._protocols import dispose_llm_clients
import httpx
# Reuse the SDK's SSE framing so fragmented events retain exactly its parsing semantics.
from openai._streaming import SSEDecoder

from . import deadline_events, rate_limits, session_timing
from .experiment_gateway import (CredentialFailure, GatewayConfigurationError, GatewayLease, GatewayPool, GatewayUnavailableError, ProviderPool,
                                 credential_failure, require_deepseek_model)
from .gateway_chat import GatewayTransportError, _verify_response_model
from .judge_majority import PASSES_ENV, configured_passes
from .key_broker import broker_enabled
from .prompt_cache import response_usage, wire_request


TRANSPORT_PROTOCOL = "volcengine_gateway_v1"


def selected_transport_protocol(*, base_protocol: str | None = None) -> str:
    """Return the cache namespace of the selected, validated chat routes.

    The historical Volcengine namespace is preserved for its existing cache rows.
    The official profile includes both approved endpoints because a pipeline can use
    Qwen and DeepSeek in different stages of one run. Account rotation does not move
    this identity.
    """
    protocol = TRANSPORT_PROTOCOL if base_protocol is None else base_protocol
    pool = GatewayPool.from_env()
    experiment_id = os.environ.get("RUBIKSQL_GATEWAY_EXPERIMENT_ID")
    if experiment_id:
        binding = pool.experiment_binding(experiment_id)
        if binding is not None and binding[0] != pool.profile:
            raise GatewayConfigurationError(
                "The experiment is bound to another provider profile; rebind it while paused before resuming."
            )
    if pool.profile == "volcengine":
        return protocol
    return (f"{protocol}|{pool.transport_identity('qwen')}|"
            f"{pool.transport_identity('deepseek')}")


LLMRole = Literal["candidate", "judge"]

_MAX_RETRIES = 2
_JUDGE_TIMEOUT = httpx.Timeout(600.0, connect=5.0)
#: Embedding calls are small and fast (p95 ~16s at 105-wide bursts), but they used to
#: ship with NO timeout at all - and on 08-26 a single black-holed OpenRouter socket
#: hung the entire knowledge build for 63 minutes at 0% CPU (heartbeat stalled_sec
#: 3781, one idle ESTABLISHED connection to Cloudflare, every thread parked). DashScope
#: never surfaced this; a WAN hop through a CDN does. With a real timeout the transport
#: retries take over and a dead socket costs minutes, not the run.
_EMBED_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
# Single owner of the embedding identity (embedding.py re-exports these). The model and
# the dimension are NOT knobs: they define the vector space every workspace was built in,
# and the wire request reads exactly these two values — an edit anywhere else is inert.
# DashScope text-embedding-v4 at 1024 dimensions since 2026-09-26 (the user's switch; before it
# text-embedding-v3, also 1024 wide): a workspace built with the old model cannot follow in
# place — its vectors are another space of the same width (embedding.assert_workspace_dim).
EMBEDDING_DIM = 1024
EMBEDDING_MODEL = "text-embedding-v4"

_ROUTES = ("qwen", "deepseek")
_MODEL_FAILURE: ContextVar[GatewayConfigurationError | None] = ContextVar("gateway_model_failure", default=None)


@contextlib.contextmanager
def _completion_model_guard() -> Iterator[None]:
    """Surface raw response identity errors outside the SDK's automatic retry boundary."""
    token = _MODEL_FAILURE.set(None)
    try:
        try:
            yield
        except Exception:
            if failure := _MODEL_FAILURE.get():
                raise failure from None
            raise
        if failure := _MODEL_FAILURE.get():
            raise failure from None
    finally:
        _MODEL_FAILURE.reset(token)


def _assert_no_custom_headers() -> None:
    """Refuse SDK environment headers which could override the leased Authorization."""
    if os.environ.get("OPENAI_CUSTOM_HEADERS"):
        raise GatewayConfigurationError("OPENAI_CUSTOM_HEADERS is not authorized for project API calls.")


def _disable_client_redirects(client: Any) -> None:
    """Pin the SDK's HTTP transport to its approved endpoint before sending a request."""
    # HB exposes the SDK through to_aclient(), but its vendored constructor has no
    # http_client option. OpenAI's only handle on that httpx client is private.
    # Fail closed if that seam changes; never silently restore SDK redirect defaults.
    transport = getattr(client, "_client", None)
    if transport is None or not hasattr(transport, "follow_redirects"):
        raise GatewayConfigurationError("The gateway SDK transport cannot disable endpoint redirects.")
    transport.follow_redirects = False


@dataclass(frozen=True, slots=True)
class _GatewayRuntime:
    """Bind an LLM to its experiment; credentials are selected anew for each attempt.

    ``model`` is the id the route receives (``ProviderPool.model_id``), not the project id.
    """

    provider: str
    model: str
    role: LLMRole
    experiment_id: str
    pool: ProviderPool = field(repr=False)
    group_id: str | None = None


class _GatewayAccountError(RuntimeError):
    """Carry a classified successful-HTTP rejection without retaining provider text."""

    def __init__(self, status_code: int, reason: CredentialFailure) -> None:
        self.status_code = status_code
        self.body = {"code": "invalid_api_key" if reason == "invalid" else "insufficient_quota"}
        super().__init__(f"Gateway account {reason}; HTTP status {status_code}; provider details withheld.")


def _install_provider_hook(client: Any, recipe: _GatewayRuntime, lease: GatewayLease, *, asynchronous: bool) -> None:
    """Check response identity and account errors, and gate physical SDK requests."""
    if getattr(getattr(recipe.pool, "owner", None), "file_keys", None) is not None:
        from .lent_chat import install
        install(client, recipe, asynchronous=asynchronous)
        return
    transport = getattr(client, "_client", None)
    hooks = getattr(transport, "event_hooks", None)
    if not isinstance(hooks, dict):
        raise GatewayConfigurationError("The chat transport cannot enforce its provider hooks.")

    def reject_account(status_code: int, body: Any) -> None:
        try:
            body = json.loads(body)
        except (ValueError, TypeError):
            pass
        reason = credential_failure(status_code=status_code, body=body)
        if reason is not None:
            raise _GatewayAccountError(status_code, reason)

    def requested_model(response: httpx.Response) -> str:
        # HB can share an SDK client between models. Its latest installed hook
        # must compare this response with this request, not the latest recipe.
        return str(json.loads(response.request.content)["model"])

    def reject_event(response: httpx.Response, decoder: SSEDecoder, chunk: bytes) -> None:
        for event in decoder.iter_bytes(iter((chunk,))):
            if event.data and not event.data.startswith("[DONE]"):
                try:
                    body = json.loads(event.data)
                except (ValueError, TypeError):
                    body = event.data
                _verify_response_model(body, requested_model(response), secret=lease.secret)
                reject_account(response.status_code, body)

    def defer_model_failure(response: httpx.Response, raw: bytes) -> bool:
        try:
            body = json.loads(raw)
        except (ValueError, TypeError):
            return False
        try:
            _verify_response_model(body, requested_model(response), secret=lease.secret)
        except GatewayConfigurationError as failure:
            _MODEL_FAILURE.set(failure)
            # This local response header is an SDK retry control, never sent to a
            # provider. The create wrapper raises the original configuration error.
            response.headers["x-should-retry"] = "false"
            return True
        return False

    def synchronous_response(response: httpx.Response) -> None:
        if not response.is_success or "text/event-stream" not in response.headers.get("content-type", "").lower():
            raw = response.read()
            if not defer_model_failure(response, raw) and response.is_success:
                reject_account(response.status_code, raw)
            return
        original = response.iter_bytes

        def iter_sse(chunk_size: int | None = None) -> Iterator[bytes]:
            decoder = SSEDecoder()
            # Inspect one decoded event at a time; never consume later events before
            # the SDK can deliver this one, or a late rejection could replay output.
            for chunk in decoder._iter_chunks(original(chunk_size)):
                reject_event(response, decoder, chunk)
                yield chunk

        response.iter_bytes = iter_sse

    async def asynchronous_response(response: httpx.Response) -> None:
        if not response.is_success or "text/event-stream" not in response.headers.get("content-type", "").lower():
            raw = await response.aread()
            if not defer_model_failure(response, raw) and response.is_success:
                reject_account(response.status_code, raw)
            return
        original = response.aiter_bytes

        async def iter_sse(chunk_size: int | None = None) -> AsyncIterator[bytes]:
            decoder = SSEDecoder()
            async for chunk in decoder._aiter_chunks(original(chunk_size)):
                reject_event(response, decoder, chunk)
                yield chunk

        response.aiter_bytes = iter_sse

    # Inspect identity before SDK status handling, and 2xx account errors before
    # normalization drops their provider status or envelope.
    previous_response = getattr(transport, "_rubiksql_response_hook", None)
    response_hook = asynchronous_response if asynchronous else synchronous_response
    hooks["response"] = [*filter(lambda item: item is not previous_response, hooks.get("response", [])), response_hook]
    transport._rubiksql_response_hook = response_hook

    # OpenAI retries exceptions raised by HTTP response hooks as connection errors.
    # Defer raw-body identity errors to this create wrapper, before HB can normalize
    # away the model or execute a tool. Also inspect SDK-shaped clients that bypass
    # HTTP hooks. Stream events are checked above, outside the SDK's retry loop.
    completions = client.chat.completions
    create = getattr(completions, "_rubiksql_create", None)
    if create is None:
        create = completions.create
        completions._rubiksql_create = create

    def synchronous_create(**kwargs: Any) -> Any:
        with _completion_model_guard():
            response = create(**wire_request(kwargs, recipe.model))
            if not kwargs.get("stream") and _MODEL_FAILURE.get() is None:
                _verify_response_model(response, str(kwargs.get("model") or recipe.model), secret=lease.secret)
            return response_usage(response, stream=bool(kwargs.get("stream")))

    async def asynchronous_create(**kwargs: Any) -> Any:
        with _completion_model_guard():
            response = await create(**wire_request(kwargs, recipe.model))
            if not kwargs.get("stream") and _MODEL_FAILURE.get() is None:
                _verify_response_model(response, str(kwargs.get("model") or recipe.model), secret=lease.secret)
            return response_usage(response, stream=bool(kwargs.get("stream")))

    completions.create = asynchronous_create if asynchronous else synchronous_create
    if not recipe.pool.owner.chat_pacing_enabled and "RUBIKSQL_PAID_LAUNCH" not in os.environ:
        return

    def payload(request: Any) -> Any:
        try:
            return json.loads(request.content)
        except (ValueError, TypeError):
            raise GatewayConfigurationError("The chat request cannot be paced before transport.") from None

    def synchronous_hook(request: Any) -> None:
        body = payload(request)
        with session_timing.measure("gate", "gateway_request_reservation"):
            recipe.pool.before_request(lease, str(body.get("model") or recipe.model), body)

    async def asynchronous_hook(request: Any) -> None:
        body = payload(request)
        with session_timing.measure("gate", "gateway_request_reservation"):
            await asyncio.get_running_loop().run_in_executor(None,
                lambda: recipe.pool.before_request(lease, str(body.get("model") or recipe.model), body))

    previous = getattr(transport, "_rubiksql_provider_hook", None)
    hook = asynchronous_hook if asynchronous else synchronous_hook
    hooks["request"] = [*filter(lambda item: item is not previous, hooks.get("request", [])), hook]
    transport._rubiksql_provider_hook = hook


@dataclass(slots=True)
class _RuntimeScope:
    """Own HeavenBase authority and route engines for one execution loop."""

    context: hb.Context
    engines: dict[tuple[str, ...], hb.LLMEngine]


_SCOPES: dict[
    tuple[int, asyncio.AbstractEventLoop | None],
    _RuntimeScope,
] = {}
_EMBEDDING_SCOPE: _RuntimeScope | None = None
_SCOPES_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class ModelTurn:
    """Store one provider-neutral assistant turn.

    Args:
        text (str): Visible assistant text without separate reasoning content.
        think (str): Provider reasoning text when exposed separately.
        tool_calls (list[dict[str, Any]]): OpenAI-shaped native tool calls.
        message (dict[str, Any]): Replay-safe assistant message.
        usage (dict[str, Any]): Normalized provider usage counters.
    """

    text: str
    think: str
    tool_calls: list[dict[str, Any]]
    message: dict[str, Any]
    usage: dict[str, Any]


def _scope_key() -> tuple[int, asyncio.AbstractEventLoop | None]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    return threading.get_ident(), loop


def _runtime_scope() -> _RuntimeScope:
    key = _scope_key()
    with _SCOPES_LOCK:
        scope = _SCOPES.get(key)
        if scope is None:
            scope = _RuntimeScope(hb.Context.load(), {})
            _SCOPES[key] = scope
        return scope


def _embedding_runtime_scope() -> _RuntimeScope:
    global _EMBEDDING_SCOPE
    with _SCOPES_LOCK:
        if _EMBEDDING_SCOPE is None:
            _EMBEDDING_SCOPE = _RuntimeScope(hb.Context.load(), {})
        return _EMBEDDING_SCOPE


def _close_current_llm_runtime() -> None:
    """Close the HeavenBase authority owned by the current thread/event loop."""
    with _SCOPES_LOCK:
        scope = _SCOPES.pop(_scope_key(), None)
    if scope is not None:
        scope.context.close()


async def _aclose_current_llm_runtime() -> None:
    """Close the current async authority and let scheduled SDK closes finish."""
    before = set(asyncio.all_tasks())
    _close_current_llm_runtime()
    current = asyncio.current_task()
    spawned = [
        task
        for task in asyncio.all_tasks()
        if task not in before and task is not current
    ]
    if spawned:
        await asyncio.gather(*spawned, return_exceptions=True)


@atexit.register
def _close_all_llm_runtimes() -> None:
    """Best-effort cleanup for sync worker scopes not owned by an MCP loop."""
    global _EMBEDDING_SCOPE
    with _SCOPES_LOCK:
        scopes = list(_SCOPES.values())
        _SCOPES.clear()
        if _EMBEDDING_SCOPE is not None:
            scopes.append(_EMBEDDING_SCOPE)
            _EMBEDDING_SCOPE = None
    for scope in scopes:
        scope.context.close()


def _engine_overlay(
    route_name: str,
    model_id: str,
    gateway: str,
    backend: str,
    api_key: str,
    base_url: str,
) -> dict[str, Any]:
    """Build the full-authority engine config for one route.

    The overlay is dmerged OVER the machine-level ``heavenbase.llm`` subtree, and the
    engine's args chain starts from the merged root's ``default_args``/``args``. A plain
    empty dict cannot mask machine keys (dmerge keeps them), so both root channels are
    pinned with the dmerge overwrite sentinel: whatever another project wrote under
    ``~/.heavenbase`` (``top_p``, ``think``, …) must never ride into RubikSQL payloads —
    that is exactly the silent provider-payload drift the migration plan lists as a stop
    condition.

    Args:
        route_name (str): RubikSQL public route name, e.g. ``"qwen"``.
        model_id (str): Provider-side model identifier.
        gateway (str): HeavenBase gateway name.
        backend (str): HeavenBase backend name.
        api_key (str): Non-empty provider credential.
        base_url (str): Provider endpoint base URL.

    Returns:
        dict[str, Any]: Config overlay for ``hb.LLMEngine``.
    """
    from .chat_keys import DeniedChatKey, denylist, refused
    if refused(api_key, "runtime chat credential", denylist()):
        raise DeniedChatKey("A denylisted chat credential cannot initialize a chat engine.")
    provider_id = f"rubiksql-{route_name}"
    model_key = f"rubiksql-{route_name}-model"
    preset_id = f"rubiksql-{route_name}"
    # One sentinel pair per node the engine's args chain reads (root, endpoint, gateway,
    # provider, model, provider record, preset): machine drift at ANY of them — shared
    # bundled ids like gateways.openai included, and colliding definitions of our own
    # rubiksql-* ids — must not reach the wire. Gateway headers are also owned here:
    # a machine Authorization header must never override the selected pool account.
    _masked = {"default_args": {_HB_OVERWRITE: True}, "args": {_HB_OVERWRITE: True}}
    return {
        "default_preset": preset_id,
        **_masked,
        "endpoints": {
            "openai": dict(_masked),
            "anthropic": dict(_masked),
        },
        "gateways": {
            gateway: {**_masked, "headers": {_HB_OVERWRITE: True}},
        },
        "presets": {
            preset_id: {
                "provider": provider_id,
                "gateway": gateway,
                "model": model_key,
                **_masked,
            },
        },
        "models": {
            model_key: {
                "default_provider": provider_id,
                **_masked,
                "providers": {
                    provider_id: {
                        "model_id": model_id,
                        **_masked,
                    },
                },
            },
        },
        "providers": {
            provider_id: {
                "api_key": api_key,
                "base_url": base_url,
                "backend": backend,
                "gateway": gateway,
                "allow_unknown_models": True,
                "require_api_key": True,
                **_masked,
            },
        },
    }


def _project_engine(
    route_name: str,
    model_id: str,
    gateway: str,
    backend: str,
    api_key: str,
    base_url: str,
    *,
    scope: _RuntimeScope | None = None,
) -> hb.LLMEngine:
    """Build or reuse one scope-local engine without inherited model defaults."""
    scope = scope or _runtime_scope()
    key = (
        route_name,
        model_id,
        gateway,
        backend,
        api_key,
        base_url,
    )
    with _SCOPES_LOCK:
        cached = scope.engines.get(key)
        if cached is not None:
            return cached

        engine = hb.LLMEngine(
            config=_engine_overlay(
                route_name,
                model_id,
                gateway,
                backend,
                api_key,
                base_url,
            ),
            context=scope.context,
        )
        scope.engines[key] = engine
        return engine


def _project_llm(
    *,
    route_name: str,
    model_id: str,
    gateway: str,
    endpoint: str,
    backend: str,
    api_key: str,
    base_url: str,
    args: dict[str, Any],
    scope: _RuntimeScope | None = None,
) -> hb.LLM:
    """Build one request runtime on a scope-cached project engine."""
    engine = _project_engine(
        route_name,
        model_id,
        gateway,
        backend,
        api_key,
        base_url,
        scope=scope,
    )
    preset_id = f"rubiksql-{route_name}"
    config_args = dict(args)
    runtime_timeout = config_args.get("timeout")
    if isinstance(runtime_timeout, httpx.Timeout):
        # OmegaConf accepts only primitive config values. Resolve the canonical
        # route first, then attach the SDK Timeout as a runtime-only spec arg.
        config_args["timeout"] = None
    llm = hb.LLM(
        preset=preset_id,
        endpoint=endpoint,
        engine=engine,
        cache=False,
        args=config_args,
    )
    if isinstance(runtime_timeout, httpx.Timeout):
        llm.spec = llm.spec.with_args({"timeout": runtime_timeout})
    return llm


def build_llm(
    provider: str,
    model: str,
    *,
    role: LLMRole,
    temperature: float | None = None,
    seed: int | None = None,
    experiment_id: str | None = None,
) -> hb.LLM:
    """Resolve one supported RubikSQL chat route through HeavenBase.

    Args:
        provider (str): Public RubikSQL provider name.
        model (str): Project model identifier: a Qwen id, the DeepSeek knowledge default, or
            ``experiment_gateway.DEEPSEEK_CANDIDATE_MODEL`` for candidates only. The request
            sends the selected route's id for it (``ProviderPool.model_id``).
        role (LLMRole): Request role. Supported values:
            - `candidate`: Candidate SQL agent with a 180-second default timeout.
            - `judge`: Judge or reviewer with the former SDK default pinned to
              5-second connect and 600-second read/write/pool timeouts.
        temperature (float | None): Optional sampling temperature.
        seed (int | None): Optional provider seed.
        experiment_id (str | None): Allocation identity in the configured pool, or ``None`` for
            ``RUBIKSQL_GATEWAY_EXPERIMENT_ID``. Every later attempt on the runtime (:func:`achat_turn`,
            :class:`SyncGatewayLLM`) leases under the same identity.

    Returns:
        hb.LLM: Configured HeavenBase LLM with its text cache disabled.

    Raises:
        GatewayConfigurationError: The configured account pool or experiment identity is missing.
        ValueError: The provider, role, or model is unsupported.
    """
    if provider == "claude":
        raise GatewayConfigurationError("Project Claude API calls are not authorized; use the approved experiment gateway.")
    provider = raise_mismatch(_ROUTES, provider, name="RubikSQL provider")
    role = raise_mismatch(("candidate", "judge"), role, name="LLM role")
    _assert_no_custom_headers()
    if "gemini" in model.casefold():
        raise ValueError(
            "Gemini models are not supported by the RubikSQL hb.LLM migration route"
        )
    if provider == "deepseek":
        require_deepseek_model(model, candidate=role == "candidate")
    if provider == "qwen" and not model.startswith("qwen"):
        raise GatewayConfigurationError("The Qwen route requires a Qwen model identifier.")
    pool, lease = GatewayPool.from_env().lease_provider(provider, experiment_id, advance=False)
    wire_model = pool.model_id(model)
    args: dict[str, Any] = {
        "temperature": temperature,
        "seed": seed,
        # Always request-response: HeavenBase streams a chat unless this is off, and
        # achat_turn reads one whole response (the only streaming consumer, the dmxapi
        # claude-thinking route, is retired).
        "stream_chat": False,
        "max_retries": 0 if getattr(getattr(pool, "owner", None), "file_keys", None) is not None else _MAX_RETRIES,
        "timeout": (
            float(os.environ.get("RUBIKSQL_LLM_TIMEOUT_SEC", "180") or "180")
            if role == "candidate"
            else _JUDGE_TIMEOUT
        ),
    }
    # Qwen keeps its provider-default reasoning; DeepSeek mining disables thinking.
    if provider == "deepseek" and role == "candidate":
        args["extra_body"] = {"thinking": {"type": "disabled"}}

    llm = _project_llm(
        route_name=f"{provider}-{role}",
        model_id=wire_model,
        gateway="openai",
        endpoint="openai",
        backend="openai",
        api_key=lease.secret,
        base_url=lease.base_url,
        args=args,
        scope=None,
    )
    llm._rubiksql_gateway = _GatewayRuntime(provider, wire_model, role, lease.experiment_id, pool, lease.group_id)
    return llm


def _tool_call(value: dict[str, Any]) -> dict[str, Any]:
    function = value.get("function") if isinstance(value.get("function"), dict) else {}
    arguments = function.get("arguments", value.get("arguments", ""))
    if isinstance(arguments, (dict, list)):
        arguments = dumps_json(arguments, indent=None)
    argument_text = "" if arguments is None else str(arguments)
    return {
        "id": str(value.get("id") or ""),
        "type": str(value.get("type") or "function"),
        "function": {
            "name": str(function.get("name") or value.get("name") or ""),
            "arguments": argument_text,
        },
    }


def _turn(
    text: str,
    think: str,
    tool_calls: list[dict[str, Any]],
    usage: dict[str, Any],
) -> ModelTurn:
    calls = [_tool_call(call) for call in tool_calls]
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = calls
    return ModelTurn(
        text=text,
        think=think,
        tool_calls=calls,
        message=message,
        usage=dict(usage),
    )


def merge_usage(current: dict[str, Any], delta: dict[str, Any]) -> None:
    """Fold one turn's usage counters into a running total, in place.

    Numeric counters add up; non-numeric metadata (model names, cache flags)
    keeps its first observed value. Callers accumulate across the turns of one
    agent run so the total is directly comparable across providers.

    Args:
        current (dict[str, Any]): Running totals, mutated in place.
        delta (dict[str, Any]): One turn's normalized usage counters.
    """
    for key, value in delta.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            previous = current.get(key)
            current[key] = (previous if isinstance(previous, (int, float)) and not isinstance(previous, bool) else 0) + value
        elif isinstance(value, dict):
            if not isinstance(current.get(key), dict):
                current[key] = {}
            merge_usage(current[key], value)
        elif key not in current:
            current[key] = value


#: Wall-clock deadline for ONE chat turn. The transport's own timeouts cannot be
#: trusted - run 9's sweep froze for 41 minutes in a zombie socket the configured
#: timeouts never fired on (the same tunnel pathology the embedding watchdog was
#: built for), and the per-question inner cancel could not unwind it either. The
#: judge gate queues OUTSIDE this call, so no legitimate rate-limit wait is under
#: this clock. 630s, not 300: the judge transport is configured with a 600s read
#: timeout (_JUDGE_TIMEOUT) and a deadline below it turns that configuration into
#: dead code while guillotining legal long-thinking turns (audited 08-27 - run 10
#: cut three whole candidate rounds on the historically slowest questions). 630
#: still sits far under the per-question guillotine, so one dead socket costs a
#: retry instead of the question.
_CHAT_DEADLINE_S = float(os.environ.get("RUBIKSQL_CHAT_DEADLINE_SEC", "630") or "630")

#: Extra in-turn attempts when the provider answers 429 - the tier refills within a
#: minute, so a rate limit is latency, not failure (see achat_turn's retry loop).
_RATE_LIMIT_RETRIES = int(os.environ.get("RUBIKSQL_RATELIMIT_RETRIES", "3") or "3")


#: Hook vote.py installs so judge turns are charged against the lane's sliding
#: window PER REQUEST (the 08-27 audit: a session admission hid its 1-3 actual
#: requests plus every 429 retry from the pacer). None means no pacing.
_JUDGE_TURN_PACER = None

#: Concurrent judge pipelines per question (RUBIKSQL_JUDGE_PASSES, `rubiksql_hb.judge_majority`).
#: Every per-process judge limit (this executor, vote.py's RPM window and session gate) is
#: stated per pass and scales with it, so N passes keep each pass's single-pass throughput.
JUDGE_PASSES = configured_passes(os.environ.get(PASSES_ENV))

# Long judge admission/pacing waits must not occupy the default executor used by
# HTTP request reservations. The worker bound changes scheduling, never lane policy.
# Waiters are served FIFO by ticket and only a running waiter holds a ticket, so a full
# executor queues later waiters without ever blocking the one being served.
_JUDGE_PACING_EXECUTOR = ThreadPoolExecutor(max_workers=32 * JUDGE_PASSES, thread_name_prefix="judge-pacing")

#: Stop check of the deadline-owned task running in this context (nl2sql._SharedSessionDeadline
#: binds it); None outside one. A deadline cancels its task once, and that one cancellation can
#: be consumed on the way: Python 3.11's asyncio.wait_for returns a reply that completed in the
#: same loop iteration. The check keeps a stopped task from starting any further work.
_STOP_CHECK: ContextVar[Callable[[], bool] | None] = ContextVar("rubiksql_stop_check", default=None)


@contextlib.contextmanager
def bind_stop_check(stopped: Callable[[], bool]) -> Iterator[None]:
    """Bind a deadline's stop check to this task context and the tasks it creates.

    A check already bound stays in force: the task counts as stopped once either says so (a candidate
    seat the cap cut inside a deadline-owned vote).

    Args:
        stopped: Thread-safe predicate that turns True once the deadline stopped the task.

    Yields:
        None while the check is bound.
    """
    outer = _STOP_CHECK.get()
    token = _STOP_CHECK.set(stopped if outer is None else lambda: stopped() or outer())
    try:
        yield
    finally:
        _STOP_CHECK.reset(token)


#: Monotonic time by which the work of this task context must be done (the candidate cap, the
#: judge deadline; `phase_budget`); ``None`` outside a deadline. A one-shot turn retry (an unanswered request,
#: unparseable tool-call arguments) runs only while at least :data:`_RETRY_MIN_LEFT_SEC` remain before it.
_RETRY_HORIZON: ContextVar[float | Callable[[], float | None] | None] = ContextVar("rubiksql_retry_horizon", default=None)
#: Seconds a one-shot turn retry needs before the horizon: a full candidate read timeout (180 s) plus margin.
_RETRY_MIN_LEFT_SEC = float(os.environ.get("RUBIKSQL_TURN_RETRY_MIN_LEFT_SEC", "200") or "200")
#: One-shot retries per turn: of a request the provider left unanswered (ReadTimeout, 5xx, lost
#: connection; C lost 63 whole seats to one such read), and of a reply whose tool-call arguments do not
#: parse (three C seats died on "Expecting ',' delimiter"). 0 disables both.
_TURN_RETRIES = int(os.environ.get("RUBIKSQL_TURN_RETRIES", "1") or "0")


@contextlib.contextmanager
def bind_retry_horizon(at: float | Callable[[], float | None] | None) -> Iterator[None]:
    """Bind the monotonic time a one-shot turn retry must leave room before, for this context's tasks.

    Args:
        at (float | Callable[[], float | None] | None): Monotonic deadline of the work, a function
            returning the current one (a horizon that moves while the work runs), or ``None``.

    Yields:
        None while the horizon is bound.
    """
    token = _RETRY_HORIZON.set(at)
    try:
        yield
    finally:
        _RETRY_HORIZON.reset(token)


def retry_horizon() -> float | None:
    """The bound retry horizon of this task context, resolved now; ``None`` without one."""
    horizon = _RETRY_HORIZON.get()
    return horizon() if callable(horizon) else horizon


def _retry_allowed(used: int, sdk_retries: int = 0) -> bool:
    """Whether a one-shot turn retry may run.

    It needs to be unused, the task not stopped, room before the horizon, and an SDK that retries
    nothing itself: with SDK retries (the account pool's ``max_retries``), one failed turn already
    made several physical requests, and a retry on top would outlast the room the guard assumes.
    """
    if used >= _TURN_RETRIES or sdk_retries or task_stopped():
        return False
    horizon = retry_horizon()
    return horizon is None or horizon - time.monotonic() >= _RETRY_MIN_LEFT_SEC


def _unparseable_tool_call(turn: "ModelTurn") -> bool:
    """Whether a reply carries tool-call arguments the agent loops cannot read.

    Mirrors ``nl2sql._parse_tool_args``: JSON, or a first complete JSON object followed by extra text.
    """
    for call in turn.tool_calls:
        raw = str((call.get("function") or {}).get("arguments") or "") or "{}"
        try:
            json.loads(raw)
            continue
        except json.JSONDecodeError:
            pass
        try:
            json.JSONDecoder().raw_decode(raw)
        except json.JSONDecodeError:
            return True
    return False


def task_stopped() -> bool:
    """Return whether the deadline that owns this task context has stopped it."""
    check = _STOP_CHECK.get()
    return check is not None and check()


def refuse_if_stopped() -> None:
    """Re-deliver a stopped task's cancellation before it starts provider or tool work.

    Raises:
        asyncio.CancelledError: The deadline that owns this task context has stopped it.
    """
    if task_stopped():
        raise asyncio.CancelledError("answering deadline stopped this task")


async def pace_judge(pacer: Callable[[], Any]) -> None:
    """Wait for a judge pacer on its dedicated executor with the caller's context.

    Args:
        pacer: Existing synchronous admission or per-request pacing operation.

    Raises:
        asyncio.CancelledError: The owning deadline stopped this task before the wait.
        Exception: The pacer's original failure, without conversion.
    """
    refuse_if_stopped()
    context = copy_context()
    await asyncio.get_running_loop().run_in_executor(_JUDGE_PACING_EXECUTOR, context.run, pacer)


def _is_judge_llm(llm: "hb.LLM") -> bool:
    """Whether this runtime is the judge lane (identified by its SDK transport).

    A structural check, deliberately not object identity: spec normalization
    (`with_args`) copies arg values, so `is _JUDGE_TIMEOUT` was always False on a
    real hb.LLM and the judge turn charge never fired - run 14 measured the judge
    lane fully unpaced while the mock-built unit test stayed green. The invariant
    that survives copying: only the judge role carries an httpx.Timeout in its
    spec args (candidates get a plain float).
    """
    return isinstance(llm.spec.args.get("timeout"), httpx.Timeout)


def _rate_lane(llm: Any) -> str:
    """The :mod:`.rate_limits` lane of a chat runtime: ``judge`` for the judge lane, ``chat`` for every other role.

    Structural like :func:`_is_judge_llm`, and tolerant of a runtime without a spec (a test double), which counts as
    ``chat``.
    """
    args = getattr(getattr(llm, "spec", None), "args", None)
    get = getattr(args, "get", None)
    return "judge" if get is not None and isinstance(get("timeout"), httpx.Timeout) else "chat"


#: Extra in-turn attempts when the transport itself dies mid-call. An outage is
#: latency with a horizon, not failure: the SDK's own reconnect attempts die within
#: seconds inside a real outage, so this layer waits 30/60/90s (jittered) - long
#: enough to ride out a router blip or proxy rotation, bounded so a dead config
#: still surfaces within one question budget. (08-27 rehearsal: a ~10-minute outage
#: during the retry sweep turned rescuable questions into placeholders.)
_CONN_RETRIES = 3


#: Outage-shaped transport deaths: the request never reached the provider, or the
#: transfer died before a response existed, so an in-turn retry cannot double-bill.
#: ConnectTimeout is the one that matters most in practice - the judge lane's
#: connect=5s makes it the canonical dead-network shape, and the OpenAI SDK wraps
#: it in an APITimeoutError whose text ("Request timed out.") names no connection
#: at all, which is exactly what the old string-only match leaked.
_OUTAGE_TRANSPORT_TYPES = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.RemoteProtocolError,
    httpx.WriteError,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
)

#: Read-side deaths are DELIBERATELY excluded: by the time a read times out the
#: request has almost certainly reached the provider and already burned its full
#: read budget (600s on the judge lane), so retrying inside the turn doubles the
#: spend on a shape the outer retry ladder already owns. They stay False even when
#: their text says "connection reset".
_READ_TRANSPORT_TYPES = (httpx.ReadTimeout, httpx.ReadError)


def _is_connection_error(exc: BaseException) -> bool:
    """Whether an exception is transport-level connection loss (outage-shaped).

    Classifies by type along the ``__cause__``/``__context__`` chain first - the
    SDK wraps the real httpx failure, and its own text can name nothing (e.g.
    APITimeoutError's "Request timed out."): connect-phase and mid-transfer
    transport deaths are outages; read-side deaths and anything carrying an HTTP
    ``status_code`` are not. Only when the chain yields no typed verdict does the
    canonical string matching apply, so an HTTP body merely echoing "connection
    error" cannot buy a retry once its exception is typed.
    """
    node: BaseException | None = exc
    seen: set[int] = set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        if getattr(node, "status_code", None) is not None:
            # An HTTP status means the request arrived; its handling lives elsewhere.
            return False
        if isinstance(node, _READ_TRANSPORT_TYPES):
            return False
        if isinstance(node, _OUTAGE_TRANSPORT_TYPES):
            return True
        node = node.__cause__ if node.__cause__ is not None else node.__context__
    text = str(exc).lower()
    return ("connection error" in text or "connection reset" in text
            or "server disconnected" in text or "peer closed connection" in text)


def _is_rate_limited(exc: BaseException) -> bool:
    """Whether an exception is the provider's rate-limit answer, whatever the SDK wrapped it in.

    Deliberately narrower than `"429" in text`: a context-overflow message quoting
    "143429 tokens" or a JSON decode error at "column 429" must not buy 150 seconds
    of pointless backoff (audited 08-27). The status number counts only when the
    text presents it AS a status.
    """
    text = str(exc).lower()
    return ("rate limit" in text
            or re.search(r"\berror code:?\s*429\b|\bstatus(?:\s+code)?:?\s*429\b"
                         r"|\b429 too many\b", text) is not None)


#: Transport deaths that mean the provider gave no answer: timeouts, lost or refused connections, a peer that closed
#: the stream (httpx for the SDK clients, the standard library for the urllib ones; a urllib ``HTTPError`` is a
#: ``URLError`` too, and its status decides first).
_UNANSWERED_TYPES = (
    TimeoutError,
    ConnectionError,
    http.client.HTTPException,
    urllib.error.URLError,
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


def _is_overloaded(exc: BaseException) -> bool:
    """Whether an exception is the provider leaving a request unanswered under load: an HTTP 5xx, a timeout or a lost
    connection — not a rate limit (:func:`_is_rate_limited`), not a request the provider refused.

    Classified along the ``__cause__`` / ``__context__`` chain, since the SDKs wrap the transport's own error: the
    first HTTP status found decides (``status_code`` of an SDK error, ``code`` of a urllib ``HTTPError``); otherwise a
    timeout, a network error or a urllib ``URLError`` (no HTTP answer at all) is an overload. The run logs count these
    beside the 429s (:func:`rate_limits.overloaded`).

    Args:
        exc (BaseException): What a transport attempt raised.

    Returns:
        bool: ``True`` for a 5xx answer or no answer at all.
    """
    if isinstance(exc, GatewayRequestError):
        return exc.overloaded
    node: BaseException | None = exc
    seen: set[int] = set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        status = getattr(node, "status_code", None)
        if status is None and isinstance(node, urllib.error.HTTPError):
            status = node.code
        if status is None:
            # The lent file-key path's sanitized error names the status it was answered with.
            status = getattr(node, "answered_status", None)
        if isinstance(status, int):
            return status >= 500
        if isinstance(node, _UNANSWERED_TYPES):
            return True
        node = node.__cause__ if node.__cause__ is not None else node.__context__
    return False


class GatewayRequestError(RuntimeError):
    """Expose only a safe failure category and HTTP status to retry/checkpoint callers."""

    def __init__(self, error: BaseException) -> None:
        status, node, seen = None, error, set()
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            value = getattr(node, "status_code", None) or getattr(node, "answered_status", None)
            if value is None:
                value = getattr(getattr(node, "response", None), "status_code", None)
            if value is None and isinstance(node, urllib.error.HTTPError):
                value = node.code
            if isinstance(value, int):
                status = value
                break
            node = node.__cause__ if node.__cause__ is not None else node.__context__
        self.status_code = status
        self.overloaded = _is_overloaded(error)
        from .provider_errors import error_body
        self.provider_failure = (getattr(error, "provider_failure", False)
                                 or status in (401, 402, 403, 429) or isinstance(status, int) and status >= 500
                                 or credential_failure(error, status_code=status, body=error_body(error)) is not None
                                 or self.overloaded or _is_connection_error(error))
        category = ("rate limit" if _is_rate_limited(error) else
                    "connection error" if _is_connection_error(error) else
                    "overloaded" if self.overloaded else "request rejected")
        suffix = f"; HTTP status {status}" if status is not None else ""
        super().__init__(f"Approved experiment gateway {category}{suffix}; provider details withheld.")


def provider_failure(error: BaseException) -> bool:
    """Identify an exhausted provider request without retaining its response body.

    Args:
        error: A model-call exception, before candidate or judge projection.

    Returns:
        Whether account exhaustion, refusal, throttling or transport failure left
        the call unanswered. Local question deadlines are classified by their owner.
    """
    if isinstance(error, GatewayUnavailableError) or getattr(error, "provider_failure", False):
        return True
    if isinstance(error, GatewayRequestError):
        return False
    status = getattr(error, "status_code", None) or getattr(error, "answered_status", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if status is None and isinstance(error, urllib.error.HTTPError):
        status = error.code
    return (status in (401, 402, 403, 429) or isinstance(status, int) and status >= 500
            or credential_failure(error) is not None
            or isinstance(error, (GatewayTransportError, httpx.TransportError)))


def _lease_request_llm(llm: hb.LLM, recipe: _GatewayRuntime) -> tuple[hb.LLM, GatewayLease]:
    """Lease the experiment's current account and bind a request-local copy of ``llm`` to it.

    HB resolves its SDK client from the spec, not from call keyword overrides, so every request gets its own spec
    and concurrent calls cannot mutate each other's credentials. A spec is a pure function of ``llm``, ``recipe`` and
    the lease's credential and endpoint, and an ``hb.LLM`` holds no per-call state, so the copy built for one
    credential and endpoint serves ``llm``'s later requests with the same pair. Building one resolves the whole HB
    configuration (OmegaConf), tens of milliseconds on the shared event loop per turn; the lent file pool's
    construction lease names one placeholder credential, so its turns now build once per run.
    """
    _assert_no_custom_headers()
    lease = recipe.pool.lease(recipe.experiment_id, **({"group_id": recipe.group_id} if recipe.group_id is not None else {}))
    built = llm.__dict__.setdefault("_rubiksql_request_llms", {})
    key = (recipe, lease.secret, lease.base_url)
    request_llm = built.get(key)
    if request_llm is None:
        request_llm = _project_llm(
            route_name=f"{recipe.provider}-{recipe.role}", model_id=recipe.model,
            gateway="openai", endpoint="openai", backend="openai",
            api_key=lease.secret, base_url=lease.base_url, args=dict(llm.spec.args),
        )
        request_llm._rubiksql_gateway = recipe
        built[key] = request_llm
    return request_llm, lease


def _rotate_on_credential_failure(recipe: _GatewayRuntime, lease: GatewayLease, error: BaseException) -> bool:
    """Retire the lease's credential when ``error`` is a classified account failure; return whether it did.

    The rotation names the exact request credential (index and revision): a delayed failure must not retire a
    replaced or explicitly restored account. The group repairs its membership
    (``GatewayPool.rotate``) without consuming the caller's next request turn.

    Raises:
        GatewayConfigurationError: The allocation state cannot record the rotation.
        GatewayUnavailableError: No configured account remains; the checkpoint is preserved.
    """
    reason = credential_failure(error)
    if reason is None:
        return False
    try:
        recipe.pool.rotate(recipe.experiment_id, lease.index, reason, failed_revision=lease.credential_revision,
                           **({"group_id": lease.group_id} if lease.group_id is not None else {}))
    except (GatewayConfigurationError, GatewayUnavailableError) as pool_error:
        raise pool_error from None
    return True


#: At most this many immediate moves to the next chat key after a rate limit, before the waiting ladder applies.
_KEY_SWITCHES = 8


def _next_key_after_rate_limit(recipe: _GatewayRuntime | None, error: BaseException, switched: int) -> bool:
    """Whether a rate-limited chat request should move straight to the next key of the host broker's order.

    With the host key broker on, each request takes the next key, so a 429 from one account is retried at once on
    the next one (up to :data:`_KEY_SWITCHES` times, and never more than the pool has other keys). With the broker
    off, or a single key, the existing waiting ladder applies unchanged.
    """
    if recipe is None or not broker_enabled() or not _is_rate_limited(error):
        return False
    owner = getattr(recipe.pool, "owner", recipe.pool)
    if getattr(owner, "file_keys", None) is not None:
        return False  # File-backed physical requests already waited for Retry-After.
    return switched < min(_KEY_SWITCHES, getattr(owner, "chat_account_count", 1) - 1)


async def achat_turn(
    llm: hb.LLM,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    system: str | None = None,
) -> ModelTurn:
    """Run one normalized HeavenBase chat turn without executing tools.

    The turn runs under :data:`_CHAT_DEADLINE_S` of wall clock. On expiry the
    current scope's HeavenBase authority is torn down - the wedged client dies
    with its pool, exactly like the embedding watchdog - and a plain
    ``TimeoutError`` reaches the caller's existing retry ladder, which rebuilds
    a fresh runtime on the next attempt.

    Two failures are retried once at once, on the next key of the broker's order, while the task
    is not stopped, the SDK retries nothing itself (the lent file-key path) and at least
    ``RUBIKSQL_TURN_RETRY_MIN_LEFT_SEC`` (200 s) remain before the bound retry horizon
    (`bind_retry_horizon`): a request the provider left unanswered (a read timeout, a 5xx or a
    lost connection, after the connection ladder), and a reply whose tool-call arguments do not
    parse. Neither produced an action, so the retry cannot repeat one; the discarded reply's usage
    is added to the returned turn's.

    Args:
        llm (hb.LLM): Runtime produced by `build_llm`.
        messages (list[dict[str, Any]]): Conversation history.
        tools (list[dict[str, Any]] | None): Non-executable provider tool schemas.
        system (str | None): Optional provider-neutral system prompt.

    Returns:
        ModelTurn: Replay-safe assistant text, reasoning, tools, and usage.

    Raises:
        asyncio.CancelledError: The owning deadline stopped this task before a request.
        GatewayConfigurationError: The allocation is invalid or the response reports another model.
        GatewayUnavailableError: Every configured provider account is unavailable.
    """
    recipe = getattr(llm, "_rubiksql_gateway", None)
    if isinstance(llm, hb.LLM) and not isinstance(recipe, _GatewayRuntime):
        raise GatewayConfigurationError("Chat runtime was not built with an approved experiment gateway allocation.")

    async def _one_turn(request_llm: hb.LLM) -> ModelTurn:
        if recipe is not None:
            client = request_llm.to_aclient()
            _disable_client_redirects(client)
            _install_provider_hook(client, recipe, lease, asynchronous=True)
            # The lent file-key path sets 0; the account pool keeps the SDK's own retries.
            sdk["retries"] = int(getattr(client, "max_retries", 0) or 0)
        fields = ["text", "think", "tool_calls", "usage"]
        response = await request_llm.achat(
            messages,
            include=fields,
            reduce=False,
            system=system,
            tools=tools,
            trace=None,  # Do not let HB persist raw provider errors before sanitization.
        )
        return _turn(
            str(response.get("text") or ""),
            str(response.get("think") or ""),
            list(response.get("tool_calls") or []),
            dict(response.get("usage") or {}),
        )

    rate_limited = 0
    conn_failed = 0
    switched = 0
    unanswered = 0
    unparseable = 0
    sdk = {"retries": 0}
    discarded_usage: dict[str, Any] = {}   # usage of a discarded unparseable reply, still billed
    lane = _rate_lane(llm)
    while True:
        # Checked before every send, in-turn retries included.
        refuse_if_stopped()
        pacer = _JUDGE_TURN_PACER
        if pacer is not None and _is_judge_llm(llm):
            # Every judge REQUEST pays the lane window - first sends and 429 retries
            # alike - on an executor thread so the shared loop keeps serving.
            with session_timing.measure("gate", "judge_turn_pacing"):
                await pace_judge(pacer)
        lease = None
        request_llm = llm
        if recipe is not None:
            with session_timing.measure("gate", "gateway_lease"):
                request_llm, lease = _lease_request_llm(llm, recipe)
        rate_limits.request(lane)
        try:
            with session_timing.measure("model"):
                turn = await asyncio.wait_for(_one_turn(request_llm), _CHAT_DEADLINE_S)
            if _unparseable_tool_call(turn) and _retry_allowed(unparseable, sdk["retries"]):
                # The loops would fail the whole seat on these arguments; ask again once instead.
                unparseable += 1
                merge_usage(discarded_usage, turn.usage)
                deadline_events.note("unparseable_retry")
                continue
            if discarded_usage:
                usage = dict(discarded_usage)
                merge_usage(usage, turn.usage)
                turn = replace(turn, usage=usage)
            return turn
        except asyncio.TimeoutError:
            rate_limits.overloaded(lane)
            # Targeted teardown (08-27 audit): dispose only THIS scope's cached SDK
            # clients - the wedged socket dies with its pool - while the Context
            # (workspaces, MCP session, engines) stays alive. The old whole-scope
            # close made one zombie socket cost every sibling on the shared loop its
            # entire runtime; now it costs them at most a lazy transport rebuild.
            with _SCOPES_LOCK:
                scope = _SCOPES.get(_scope_key())
            if scope is not None:
                with contextlib.suppress(Exception):
                    dispose_llm_clients(scope.context)
            error = TimeoutError(
                f"chat turn exceeded {_CHAT_DEADLINE_S:.0f}s wall clock; the scope's "
                "transport clients were disposed so the retry starts on fresh sockets"
            )
            error.provider_failure = True
            raise error from None
        except (GatewayConfigurationError, GatewayUnavailableError):
            raise
        except Exception as exc:  # noqa: BLE001 - classified for account rotation and transient retries
            if lease is not None and _rotate_on_credential_failure(recipe, lease, exc):
                continue
            # A 429 that survives the SDK's own retries used to kill the whole
            # question attempt (run 10: 29 three-strike failures across both slots,
            # every one a provider rate limit) - yet the tier refills within a
            # minute, so it is latency, not failure. Waiting out the bucket here
            # costs at most ~2.5 minutes against the question's 30-minute budget;
            # terminal gateway errors retain their category without the provider body.
            if _is_rate_limited(exc):
                rate_limits.rate_limited(lane)
            elif _is_overloaded(exc):
                rate_limits.overloaded(lane)
            if _next_key_after_rate_limit(recipe, exc, switched):
                switched += 1
                await asyncio.sleep(random.uniform(0.2, 1.0))
                continue
            if _is_rate_limited(exc) and rate_limited < _RATE_LIMIT_RETRIES:
                rate_limited += 1
                # Jittered: deterministic 25/50/75 made every co-throttled session
                # wake in lockstep and re-drain the bucket together (audited 08-27).
                await asyncio.sleep(min(25.0 * rate_limited, 75.0)
                                    * random.uniform(0.75, 1.25))
                continue
            if _is_connection_error(exc) and conn_failed < _CONN_RETRIES:
                conn_failed += 1
                await asyncio.sleep(min(30.0 * conn_failed, 120.0)
                                    * random.uniform(0.75, 1.25))
                continue
            if _is_overloaded(exc) and not _is_rate_limited(exc) and _retry_allowed(unanswered, sdk["retries"]):
                # No response at all (a read timeout, a 5xx, a lost connection): the turn produced
                # nothing, so asking again cannot repeat any action. One retry, at once, on the
                # next key of the broker's order, instead of failing the seat's whole run.
                unanswered += 1
                deadline_events.note("overload_retry")
                await asyncio.sleep(random.uniform(0.2, 1.0))
                continue
            if recipe is not None:
                raise GatewayRequestError(exc) from None
            raise


#: What a HeavenBase trace records for a failed provider call made through :class:`SyncGatewayLLM`.
_WITHHELD_TRACE_ERROR = {"error": "approved experiment gateway call failed; provider details withheld"}


def _withhold_trace_errors(trace: Callable[[Any], None]) -> Callable[[Any], None]:
    """Forward HeavenBase trace events, replacing each error event's provider text before it is recorded."""

    def emit(event: Any) -> None:
        trace(replace(event, payload=dict(_WITHHELD_TRACE_ERROR)) if event.kind == "error" else event)

    return emit


class SyncGatewayLLM:
    """Drive HeavenBase's own synchronous chat loop with gateway account rotation.

    :func:`achat_turn` runs one normalized turn for the project agents. A caller that hands its model to a
    HeavenBase loop instead (HeavenBase's dashboard assistant calls ``chat`` or ``stream`` and runs the tools
    itself) wraps a :func:`build_llm` runtime in this class. Each call leases the experiment's current account and
    runs on a request-local copy of that runtime with the SDK's redirects disabled, as :func:`achat_turn` does. A
    classified credential failure retires exactly that credential and, while the call has delivered nothing (no
    streamed chunk, no completed tool round), retries on the next account. Every other failure surfaces as
    :class:`GatewayRequestError`, which withholds the provider's details. A reported model mismatch
    raises :class:`GatewayConfigurationError` immediately, without retrying or rotating an account.

    Args:
        llm (hb.LLM): A :func:`build_llm` runtime; its route and request arguments serve every call.

    Attributes:
        trace (Callable[[], Callable[[Any], None]] | None): Trace factory. When set, each call passes
            ``trace=trace()`` to HeavenBase, which reports every provider request of the call to it; error events
            arrive with the provider's text withheld.

    Raises:
        GatewayConfigurationError: ``llm`` was not built by :func:`build_llm`.
    """

    def __init__(self, llm: hb.LLM) -> None:
        recipe = getattr(llm, "_rubiksql_gateway", None)
        if not isinstance(recipe, _GatewayRuntime):
            raise GatewayConfigurationError("Chat runtime was not built with an approved experiment gateway allocation.")
        self.llm = llm
        self.trace: Callable[[], Callable[[Any], None]] | None = None
        self._recipe = recipe

    def chat(self, messages: Any, **kwargs: Any) -> Any:
        """Run ``hb.LLM.chat`` on a leased account.

        Args:
            messages (Any): Conversation, as ``hb.LLM.chat`` takes it.
            **kwargs (Any): ``hb.LLM.chat`` options, ``on_round`` and ``trace`` included; ``stream=True``
                delegates to :meth:`stream`.

        Returns:
            Any: What ``hb.LLM.chat`` returns.

        Raises:
            GatewayConfigurationError: The pool or allocation is unusable, or the response reports another model.
            GatewayUnavailableError: Every configured account is unavailable; the checkpoint is preserved.
            GatewayRequestError: Any other failure, provider details withheld.
        """
        if kwargs.pop("stream", False):
            return self.stream(messages, **kwargs)
        caller_round = kwargs.pop("on_round", None)
        rounds = 0

        def on_round(response: Any) -> None:
            nonlocal rounds
            rounds += 1  # tools run after a completed round, so a retry from here could repeat them
            if caller_round is not None:
                caller_round(response)

        kwargs = self._traced(kwargs)
        switched = 0
        while True:
            request_llm, lease = self._lease()
            try:
                return request_llm.chat(messages, on_round=on_round, **kwargs)
            except (GatewayConfigurationError, GatewayUnavailableError):
                raise
            except Exception as error:  # noqa: BLE001 - classified for rotation, then always sanitized
                if _rotate_on_credential_failure(self._recipe, lease, error) and not rounds:
                    continue
                if not rounds and _next_key_after_rate_limit(self._recipe, error, switched):
                    switched += 1
                    time.sleep(random.uniform(0.2, 1.0))
                    continue
                raise GatewayRequestError(error) from None

    def stream(self, messages: Any, **kwargs: Any) -> Iterator[Any]:
        """Run ``hb.LLM.stream`` on a leased account.

        Args:
            messages (Any): Conversation, as ``hb.LLM.stream`` takes it.
            **kwargs (Any): ``hb.LLM.stream`` options, ``trace`` included.

        Yields:
            Any: The chunks ``hb.LLM.stream`` yields.

        Raises:
            GatewayConfigurationError: The pool or allocation is unusable, or the response reports another model.
            GatewayUnavailableError: Every configured account is unavailable; the checkpoint is preserved.
            GatewayRequestError: Any other failure, provider details withheld.
        """
        kwargs = self._traced(kwargs)
        while True:
            request_llm, lease = self._lease()
            chunks = request_llm.stream(messages, **kwargs)
            delivered = False
            try:
                for chunk in chunks:
                    delivered = True
                    yield chunk
                return
            except (GatewayConfigurationError, GatewayUnavailableError):
                raise
            except Exception as error:  # noqa: BLE001 - classified for rotation, then always sanitized
                if _rotate_on_credential_failure(self._recipe, lease, error) and not delivered:
                    continue
                raise GatewayRequestError(error) from None
            finally:
                chunks.close()

    def close(self) -> None:
        """Release the calling thread's HeavenBase authority and the SDK clients its calls opened."""
        _close_current_llm_runtime()

    def _lease(self) -> tuple[hb.LLM, GatewayLease]:
        request_llm, lease = _lease_request_llm(self.llm, self._recipe)
        client = request_llm.to_client()
        _disable_client_redirects(client)
        _install_provider_hook(client, self._recipe, lease, asynchronous=False)
        return request_llm, lease

    def _traced(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        trace = kwargs.pop("trace", None)
        if trace is None and self.trace is not None:
            trace = self.trace()
        return kwargs if trace is None else {**kwargs, "trace": _withhold_trace_errors(trace)}


__all__ = [
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "GatewayRequestError",
    "JUDGE_PASSES",
    "LLMRole",
    "ModelTurn",
    "SyncGatewayLLM",
    "TRANSPORT_PROTOCOL",
    "selected_transport_protocol",
    "achat_turn",
    "bind_stop_check",
    "build_llm",
    "merge_usage",
    "pace_judge",
    "refuse_if_stopped",
    "task_stopped",
]
