"""Qwen (DashScope) text embedding: ``text-embedding-v4`` at 1024 dimensions."""
import hashlib
from contextvars import copy_context
import json
import os
from pathlib import Path
import random
import threading
import time
import sqlite3
import httpx

# The model and the dimension are NOT knobs. embed() produces BOTH the vectors stored at
# build time (schema.py binds it via .compute on desc_vec / question_vec / value_vec /
# title_vec) AND the query vectors looked up against them, so overriding either at query
# time yields vectors from a different space and silently ruins every similarity. The
# dimension is additionally written into each Chroma collection when a workspace is built,
# and the model into the knowledge ledger of a workspace the pipeline's knowledge build made —
# that is what assert_workspace_dim checks (text-embedding-v3, the model until 2026-09-26, is
# 1024 wide too: only the ledger tells its vectors apart). The authorized DashScope
# endpoint is fixed too: an environment override must not send its credential elsewhere.
# llm_runtime owns the two constants (they pin the wire request); re-exported here for the
# scripts and modules that treat this module as the embedding API.
from . import rate_limits
from .experiment_gateway import (COORDINATOR_ENV, DEFAULT_POOL_PATH, GatewayConfigurationError, GatewayPool,
                                 GatewayUnavailableError, credential_failure, provider_claim)
from .llm_runtime import EMBEDDING_DIM, EMBEDDING_MODEL, _EMBED_TIMEOUT, _assert_no_custom_headers, _is_overloaded, _is_rate_limited

#: Test override: a monkeypatched `_llm` short-circuits the transport entirely, same
#: contract as when this module rode hb.LLM. Production leaves it None.
_llm = None
_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

#: One RAW OpenAI-SDK client PER THREAD - no HeavenBase engine, no shared client cache.
#: The 08-26 rehearsal exposed the stack this replaces: hb caches SDK clients by config
#: digest, so every embed in the process - seven parallel database registrations
#: included - funnelled through ONE httpx client, which serialises requests across
#: threads (measured raw: shared client 64.8s for 70 calls, per-thread clients 32.5s).
#: Worse, when a flaky tunnel socket made one call retry, the SDK's backoff slept
#: while the sibling threads queued behind the same client - the whole build froze
#: (0% CPU, stalled_sec 3781, threads parked on rlock_acquire). Embeddings need none
#: of hb's chat machinery: same wire call, thread-isolated failures, real parallelism.
_tls = threading.local()

#: Wall-clock deadline for ONE embedding attempt, and how many fresh-client attempts
#: one call gets. The httpx read timeout is configured (60s) yet a proxied tunnel read
#: was measured blocking for 20+ minutes past it (08-26, run 8: the local proxy's node
#: rotated mid-request, the socket went zombie, and both registration workers froze in
#: `ssl.read` with the armed timeout never firing). The deadline is enforced OUTSIDE
#: the transport - a watchdog that gives the call 75s of wall clock (connect 10 + read
#: 60 + slack), then closes the whole client so the hung socket dies with its pool,
#: and retries on a freshly built client. It therefore does not depend on any
#: transport layer honoring its own configuration.
_DEADLINE_S = 75.0
_ATTEMPTS = 3

#: Extra attempts when the provider answers with a rate limit, and the pause before each: the chat and transfer
#: lanes' ladder, min(15 s x k, 60 s) jittered by +-25%, or the provider's Retry-After when it sends one (at most
#: :data:`_RETRY_AFTER_MAX_S`). A rate limit is latency, not failure — DashScope may enforce its per-minute quota over
#: one-second windows, so a burst of the build's 7 x 10 parallel embeddings meets 429s the quota would allow over a
#: minute — and the callers that embed (registration, the column docs, the rule and synonym writers, apply_rules) have
#: no retry of their own: a 429 that escaped here used to fail a whole database's stage after a paid mining.
_RATE_LIMIT_RETRIES = 4
_RETRY_AFTER_MAX_S = 120.0

#: The knowledge ledger inside a workspace directory (``scripts/knowledge_build.py`` ``LEDGER_FILE``, which a test keeps
#: equal): its stage records name the embedding model their vectors were made with (``inputs.embedding``,
#: ``<model>/<dimension>``).
_LEDGER_FILE = "knowledge_ledger.json"


def _embedding_key():
    """Resolve managed credentials first; absent legacy configuration alone permits env fallback."""
    provider_claim()
    pool_path = Path(os.environ.get("RUBIKSQL_GATEWAY_POOL") or DEFAULT_POOL_PATH)
    state_path = Path(os.environ.get("RUBIKSQL_GATEWAY_STATE") or pool_path.with_suffix(".state.sqlite3"))
    remote = any(name in os.environ for name in COORDINATOR_ENV)
    location = ((os.environ.get(COORDINATOR_ENV[0], ""),
                 str(Path(os.environ[COORDINATOR_ENV[1]]).expanduser().resolve()) if os.environ.get(COORDINATOR_ENV[1]) else "")
                if remote else (str(pool_path.resolve()), str(state_path.resolve())))
    pool = getattr(_tls, "gateway_pool", None)
    if pool is not None and getattr(_tls, "gateway_location", None) != location:
        raise GatewayConfigurationError("Embedding configuration changed; resume in a fresh process after the paused update.")
    if pool is None and (remote or os.environ.get("RUBIKSQL_GATEWAY_POOL") or os.environ.get("RUBIKSQL_GATEWAY_STATE")
                         or pool_path.exists() or state_path.exists()):
        pool = GatewayPool.from_env()
        _tls.gateway_pool, _tls.gateway_location = pool, location
    lease = pool.embedding_lease() if pool is not None else None
    _tls.embedding_lease = lease
    key = lease.secret if lease is not None else None
    if remote and not key:
        raise GatewayConfigurationError("The gateway coordinator has no managed embedding credential for this database.")
    if key is None:
        key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY env var not set and no managed embedding key configured")
    return key


def _raw_client():
    """Return this thread's client for the key the next embedding request takes.

    Each request takes the next key of the managed embedding-only pool (``GatewayPool.embedding_lease``), so a
    thread keeps one client per pool index. A different key at an index this thread already used means the
    private configuration changed under the running process: that refuses, like any paused maintenance.
    """
    _assert_no_custom_headers()
    base_url = os.environ.get("RUBIKSQL_EMBED_BASE_URL", _DEFAULT_BASE_URL)
    if base_url != _DEFAULT_BASE_URL:
        raise RuntimeError("Embedding endpoint override is not authorized; DashScope remains the sole embedding route")
    api_key = _embedding_key()
    fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    lease = getattr(_tls, "embedding_lease", None)
    slot = lease.index if lease is not None else None
    clients = getattr(_tls, "clients", None)
    if clients is None:
        clients = _tls.clients = {}
    cached = clients.get(slot)
    if cached is not None and cached[0] != fingerprint:
        _tls.client = cached[1]
        _drop_client()
        raise GatewayConfigurationError("Embedding credentials changed; resume in a fresh process after the paused update.")
    if cached is None:
        from openai import OpenAI
        client = OpenAI(
            base_url=_DEFAULT_BASE_URL,
            api_key=api_key,
            timeout=_EMBED_TIMEOUT,
            max_retries=0,   # the deadline wrapper below owns retries
            http_client=httpx.Client(follow_redirects=False),
        )
        cached = clients[slot] = (fingerprint, client)
    _tls.client = cached[1]
    return cached[1]


def _drop_client():
    """Forget and close the client the thread used last - a hung socket dies with its pool."""
    client = getattr(_tls, "client", None)
    _tls.client = None
    clients = getattr(_tls, "clients", None) or {}
    for slot, (_, cached) in list(clients.items()):
        if cached is client:
            del clients[slot]
    if client is not None:
        try:
            client.close()
        except Exception:  # noqa: BLE001 - already discarding it
            pass


def _deadline_call(fn):
    """Run ``fn`` with a hard wall-clock deadline, transport-agnostic."""
    outcome = {}
    done = threading.Event()

    def _run():
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - relayed to the caller below
            outcome["error"] = exc
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True, name="embed-deadline").start()
    if not done.wait(_DEADLINE_S):
        raise TimeoutError(
            f"embedding call still silent after {_DEADLINE_S:.0f}s wall clock")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _rate_limit_pause(exc: BaseException, attempt: int) -> float:
    """Seconds to wait before retrying after a rate limit: the provider's Retry-After, else the jittered ladder.

    Args:
        exc (BaseException): The rate-limit error; its HTTP response (an SDK error's ``response``, anywhere on its
            cause chain) may carry ``Retry-After`` in seconds.
        attempt (int): The retry about to be made, from 1.

    Returns:
        float: ``Retry-After`` capped at :data:`_RETRY_AFTER_MAX_S`, or min(15 s x attempt, 60 s) x U(0.75, 1.25).
    """
    node, seen = exc, set()
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        headers = getattr(getattr(node, "response", None), "headers", None) or getattr(node, "headers", None)
        value = headers.get("retry-after") if hasattr(headers, "get") else None
        try:
            if value is not None and float(value) >= 0:
                return min(float(value), _RETRY_AFTER_MAX_S)
        except (TypeError, ValueError):
            pass   # an HTTP-date or garbage: the ladder decides
        node = node.__cause__ if node.__cause__ is not None else node.__context__
    return min(15.0 * attempt, 60.0) * random.uniform(0.75, 1.25)


def _get_llm():
    if _llm is not None:            # test override
        return _llm
    return _SdkEmbedder(_raw_client())


class _SdkEmbedder:
    """Adapter keeping the historical `llm.embed(text, dimensions=...)` seam."""

    __slots__ = ("_client",)

    def __init__(self, client):
        self._client = client

    def embed(self, text, *, dimensions):
        failed = limited = switched = blocked = 0
        while True:
            # The first attempt uses the key _get_llm took; every retry takes the next key of the pool.
            # The thread's client for a key is reused unless _drop_client below discarded it.
            client, self._client = (self._client, None) if self._client is not None else (_raw_client(), None)
            provider_claim()
            managed_pool = getattr(_tls, "gateway_pool", None)
            lease = getattr(_tls, "embedding_lease", None)
            accounts = getattr(managed_pool, "embedding_account_count", 1) if lease is not None else 1
            if managed_pool is not None and lease is not None:
                managed_pool.pace_embedding(lease, EMBEDDING_MODEL, text)
            rate_limits.request("embedding")
            try:
                return _deadline_call(
                    lambda: self._create(client, text, dimensions))
            except Exception as exc:  # noqa: BLE001 - bounded retry, then re-raised
                _drop_client()       # never reuse a connection that failed or hung
                if isinstance(exc, (GatewayConfigurationError, GatewayUnavailableError)):
                    raise
                reason = credential_failure(exc)
                if reason and managed_pool is not None and managed_pool.paired and lease is not None:
                    # 401/403 retire the key, daily exhaustion skips it; the next attempt takes the next key.
                    managed_pool.block_embedding(lease, reason)
                    if blocked < accounts:
                        blocked += 1
                        continue
                if _is_rate_limited(exc):
                    # the quota's own ladder, apart from the attempts a failure gets below
                    rate_limits.rate_limited("embedding")
                    if switched < accounts - 1:
                        # another key's own quota may serve at once: move to the next key after a short pause
                        switched += 1
                        time.sleep(random.uniform(0.1, 0.5))
                        continue
                    if limited >= _RATE_LIMIT_RETRIES:
                        raise
                    limited += 1
                    switched = 0
                    time.sleep(_rate_limit_pause(exc, limited))
                    continue
                if _is_overloaded(exc):
                    rate_limits.overloaded("embedding")
                failed += 1
                if failed >= _ATTEMPTS:
                    raise
                time.sleep(min(2 ** (failed - 1), 4.0))

    @staticmethod
    def _create(client, text, dimensions):
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL, input=text, dimensions=dimensions)
        if isinstance(text, list):
            return [row.embedding for row in sorted(resp.data, key=lambda r: r.index)]
        return resp.data[0].embedding


def embed(text: str) -> list[float]:
    """Embed a single text string using DashScope ``text-embedding-v4`` (1024-dim, ``EMBEDDING_MODEL``).

    Args:
        text (str): Text to embed. Empty or whitespace-only input returns a zero vector
            instead of calling the API.

    Returns:
        list[float]: An ``EMBEDDING_DIM``-long embedding.

    Raises:
        RuntimeError: No managed embedding key or legacy ``DASHSCOPE_API_KEY`` is configured,
            or the managed configuration is invalid or changed since this process began.
    """
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIM
    return list(
        _get_llm().embed(
            text.strip(),
            dimensions=EMBEDDING_DIM,
        )
    )


def embed_many(texts: list[str], *, workers: int = 1, embed_fn=None) -> list[list[float]]:
    """Embed several texts, ``workers`` at a time, in order: the vectors :func:`embed` returns one by one.

    Each worker thread keeps its own client (see ``_tls``), so a slow or retried call holds up only its own text.

    Args:
        texts (list[str]): Texts to embed; empty ones get the zero vector without a call, as in :func:`embed`.
        workers (int): Calls in flight at once (at least 1).
        embed_fn (Callable[[str], list[float]] | None): The one-text embedding to map, :func:`embed` by default; a
            caller passes the name it imported, so a test that replaces that name still sees every call.

    Returns:
        list[list[float]]: One vector per text, in the order of ``texts``.
    """
    fn = embed_fn or embed
    texts = list(texts)
    if workers <= 1 or len(texts) <= 1:
        return [fn(text) for text in texts]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(workers, len(texts))) as pool:
        context = copy_context()
        futures = [pool.submit(context.copy().run, fn, text) for text in texts]
        return [future.result() for future in futures]


def title_text(title: str) -> str:
    """The text a rule title's vector embeds: its identifier separators read as spaces (see :func:`embed_title`).

    Args:
        title: Stable rule title.

    Returns:
        str: ``title`` with ``_`` and ``-`` replaced by spaces.
    """
    return title.replace("_", " ").replace("-", " ")


def embed_title(title: str) -> list[float]:
    """Embed a rule title after expanding its identifier separators.

    Rule titles are stable snake_case identifiers such as
    ``difference_computation``. Replacing underscores and hyphens with spaces gives the
    embedding model the human words while leaving the stored title and identity unchanged.

    Args:
        title: Stable rule title.

    Returns:
        list[float]: An ``EMBEDDING_DIM``-long embedding.
    """
    return embed(title_text(title))


def rule_with_example(desc: str, example: str = "") -> str:
    """Compose the semantic rule text from its instruction and optional example.

    The example plays the same role as a question while ``desc`` plays the role of its
    explanatory hint. Keeping the label explicit prevents the embedding model from treating
    an illustrative question as another instruction.

    Args:
        desc: Reusable rule instruction.
        example: Optional natural-language example question or compact pattern.

    Returns:
        str: ``example + "\\n[Rule]: " + desc`` when an example exists, otherwise
        the stripped description.
    """
    instruction = str(desc or "").strip()
    illustration = str(example or "").strip()
    if not illustration:
        return instruction
    if not instruction:
        return illustration
    return f"{illustration}\n[Rule]: {instruction}"


def embed_rule_desc(desc: str, example: str = "") -> list[float]:
    """Embed one rule's instruction together with its optional example.

    Args:
        desc: Reusable rule instruction.
        example: Optional illustrative question or compact pattern.

    Returns:
        list[float]: An ``EMBEDDING_DIM``-long embedding of the composed rule text.
    """
    return embed(rule_with_example(desc, example))


def assert_workspace_dim(ws_dir: str) -> None:
    """Fail loudly when a built workspace's vectors are not ``EMBEDDING_MODEL``'s at ``EMBEDDING_DIM``.

    Chroma records a fixed dimension per collection at build time. If the shipped constant
    and the stored width ever disagree — a workspace built by another version, a partial
    rebuild, a hand-copied chroma_db — every near() lookup compares vectors across two
    different spaces and returns confident nonsense: no exception, no empty result, just
    wrong neighbours for the whole run. A width says nothing about the model, though:
    ``text-embedding-v3`` (the model until 2026-09-26) and ``text-embedding-v4`` are both 1024
    wide. A workspace the pipeline's knowledge build made keeps a ledger
    (``knowledge_ledger.json``) whose stage records name the model their vectors were made with,
    and a record that names another model than ``EMBEDDING_MODEL`` at ``EMBEDDING_DIM`` is the
    same mismatch. This is the one mismatch worth a startup check, because nothing downstream
    can detect it.

    A workspace with no Chroma database yet (nothing built) is not an error and is ignored, and
    so is one without a ledger (built before the knowledge ledger, or by another tool), whose
    model no file records.

    Args:
        ws_dir (str): Workspace directory, i.e. the parent of ``chroma_db/``.

    Raises:
        RuntimeError: A collection's stored dimension differs from ``EMBEDDING_DIM``, or the
            ledger records a vector-writing stage built with another embedding model.
    """
    path = os.path.join(ws_dir, "chroma_db", "chroma.sqlite3")
    if not os.path.exists(path):
        return
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = conn.execute("SELECT name, dimension FROM collections").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return          # not a chroma db we can read — the backend will report it properly
    bad = [(name, dim) for name, dim in rows if dim is not None and dim != EMBEDDING_DIM]
    if bad:
        detail = ", ".join(f"{name}={dim}" for name, dim in bad)
        raise RuntimeError(
            f"embedding dimension mismatch in {path}: collection(s) {detail} were built at a "
            f"different width than EMBEDDING_DIM={EMBEDDING_DIM}. Every vector lookup against "
            f"them would compare across two embedding spaces and silently return wrong "
            f"neighbours. Build a fresh knowledge root with scripts/run_official_pipeline.py "
            f"--stop-after-examples, or delete this workspace directory and build it again that "
            f"way: the knowledge build refuses to follow an embedding change in place, and a "
            f"workspace scripts/setup_dbs.py alone rebuilt carries no ledger record it can follow.")
    current = f"{EMBEDDING_MODEL}/{EMBEDDING_DIM}"
    other = sorted({(stage, str((record.get("inputs") or {}).get("embedding")))
                    for stage, record in _ledger_stages(ws_dir).items()
                    if (record.get("inputs") or {}).get("embedding") not in (None, current)})
    if other:
        detail = ", ".join(f"{stage} with {model}" for stage, model in other)
        raise RuntimeError(
            f"embedding model mismatch in {os.path.join(ws_dir, _LEDGER_FILE)}: the workspace's "
            f"vectors were made by {detail}, and this code embeds with {current} (the same width "
            f"for text-embedding-v3 and -v4, so Chroma cannot tell). Every vector lookup would "
            f"compare across two embedding spaces and silently return wrong neighbours. Build a "
            f"fresh knowledge root with scripts/run_official_pipeline.py --stop-after-examples: "
            f"the knowledge build refuses to follow an embedding model change in place.")


def unrecorded_embedding(ws_dir: str) -> bool:
    """Whether a built workspace's vectors carry no record of the embedding model that made them.

    :func:`assert_workspace_dim` lets such a workspace open, since nothing tells its model: one built before the
    knowledge ledger (every such workspace was embedded with ``text-embedding-v3``, the model until 2026-09-26) or by
    ``scripts/setup_dbs.py`` alone. A caller about to spend on answers over it says so (``eval_evolution.py``, the
    evolution loop).

    Args:
        ws_dir (str): Workspace directory, i.e. the parent of ``chroma_db/``.

    Returns:
        bool: ``True`` when the workspace holds a Chroma database and no stage record of its ledger names an embedding
            model; ``False`` for a workspace with nothing built yet.
    """
    if not os.path.exists(os.path.join(ws_dir, "chroma_db", "chroma.sqlite3")):
        return False
    return not any((record.get("inputs") or {}).get("embedding") for record in _ledger_stages(ws_dir).values())


def _ledger_stages(ws_dir: str) -> dict:
    """The stage records of a workspace's knowledge ledger; ``{}`` when it has none or it cannot be read."""
    try:
        with open(os.path.join(ws_dir, _LEDGER_FILE), encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    stages = data.get("stages") if isinstance(data, dict) else None
    return {name: record for name, record in (stages or {}).items() if isinstance(record, dict)}
