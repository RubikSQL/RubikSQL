"""
nl2sql — NL→SQL agentic pipeline, multi-provider.

Uses the rubiksql_hb.server MCP module as the single source of truth for tools.
Tool schemas are fetched from MCP at runtime; execution goes through MCP.
No duplicate Python function wiring.

Usage:
    result = nl2sql("有哪些商品", "customers_demo")
    result = nl2sql("有哪些商品", "customers_demo", provider="qwen")
"""

import asyncio
import atexit
import concurrent.futures as _futures
from contextlib import closing, contextmanager, nullcontext
import contextvars
from functools import partial
import hashlib
import uuid
import json
import os
import re
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from typing import Iterator, Optional

from dotenv import load_dotenv
load_dotenv(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"),
    override=False,
)

from mcp import ClientSession, MCPError
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.types import CONNECTION_CLOSED

from .experiment_gateway import DEEPSEEK_CANDIDATE_MODEL, GatewayConfigurationError, GatewayUnavailableError
from .llm_runtime import (
    EMBEDDING_MODEL,
    TRANSPORT_PROTOCOL,
    selected_transport_protocol,
    ModelTurn,
    _aclose_current_llm_runtime,
    _close_current_llm_runtime,
    achat_turn,
    bind_stop_check,
    build_llm,
    merge_usage,
    refuse_if_stopped,
    task_stopped,
)
from .model_render import MODEL_PROTOCOL, binary_json_default, model_tool_content, render_rules, restore_sql_results, table_display
from .prompts import RenderRecorder, current as _current_prompt_release
from .question import question_hints, question_with_hint
from .sql_action import SQL_ACTIONS, missing_rank_output, publish_rank_result
from . import count_columns, hint_conflicts, limit_repeats
from . import sql_host as sql_tools
from .knowledge_failures import degraded_knowledge_result, degraded_record
from . import turn_memo as _turn_memo
from . import cache_lineage
from . import deadline_events
from . import phase_budget
from . import session_timing
from . import tool_contract
from .turn_memo import RunClock, TurnMemo

# ── Model-facing text ─────────────────────────────────────────────────────────
# Every prompt text a candidate, judge or reviewer receives is a surface of the pinned
# hb.Prompt release (src/rubiksql_hb/prompts/store, RUBIKSQL_PROMPT_RELEASE): this module
# and vote.py only decide WHEN a text is injected and WHERE it goes, and they ask for texts by
# surface (``_PROMPTS.surface("candidate", "system", ...)``), never by row, so a release that
# keeps a role's prompt as one document and r0's composition of fragments serve the same
# calls. The release resolves once per process; a pin that does not load raises
# PromptStoreError here, on purpose — there is no fallback to the baseline (eval_bird once ran
# a whole arm on the wrong prompts because a missing patch fell through silently).
# ``_PROMPTS.surface`` memoises recent renders (a 512-entry LRU keyed by surface, argument
# values and types), so the globals derived below are usually the same object across reloads;
# what the code relies on is their bytes — routes captures these globals from this module
# after each reload, which is what keeps its ``is`` contract — and every model-visible render
# is counted on the RenderRecorder bound to the current agent run, under the row that rendered.
_PROMPTS = _current_prompt_release()


# ── LLM response cache (SQLite) ───────────────────────────────────────────────

_CACHE_LOCK = threading.Lock()
_NO_CACHE = False  # set via set_no_cache(): bypass cache READS only (still writes/refreshes)
_CACHE_SKIP_BARREN = False  # set via set_cache_skip_barren(): reuse only entries that got SQL
_CACHE_KEEP_EXISTING = False  # protect stored rows except this run's degraded answers


def set_no_cache(flag: bool) -> None:
    """Switch the LLM cache into force-refresh mode, process-wide.

    Args:
        flag (bool): When True, cache reads are bypassed so every call regenerates fresh —
            but the fresh result is STILL written back, refreshing any stale entry. This is
            force-refresh, NOT write-disable.
    """
    global _NO_CACHE
    _NO_CACHE = flag


def set_cache_skip_barren(flag: bool) -> None:
    """Reuse cached runs that produced SQL; regenerate the ones that produced none.

    The blunt alternative is :func:`set_no_cache`, and on a re-run of a question that already
    failed it throws away the wrong half of the work. The cache key is per (question, seed)
    and `_cache_set` only runs on the success path, so a candidate that TIMED OUT was never
    stored — force-refresh buys nothing there. What IS stored is a candidate that ran to
    completion and still ended with no usable SQL; replaying that one hands the retry the
    same dead end forever, which is the case force-refresh exists for.

    Those two are opposites, and `--no-cache` cannot tell them apart: it also discards the
    candidates that DID produce good SQL and whose only problem was a judge that timed out
    afterwards. Re-running those is pure waste — the situation is identical, so the result
    would be identical.

    So split on content instead of on the flag: an entry with a non-empty ``sql`` is reused,
    an entry without one is treated as a miss and regenerated.

    Args:
        flag (bool): When True, cached entries carrying no SQL are ignored on read. Entries
            that carry SQL are still served. Writes are unaffected either way.
    """
    global _CACHE_SKIP_BARREN
    _CACHE_SKIP_BARREN = flag


def set_cache_keep_existing(flag: bool) -> None:
    """Protect healthy cache rows while allowing degraded answers to recover.

    The shared LLM cache is also the replay source of every later run of the same epoch: a
    judge-only child replays its parent's candidates from it, and the outcome verifies each
    replay against the parent's immutable source. A counted run that wrote a fresh sample over
    such a row (as a ``--no-cache`` retry sweep did on 2026-10-23, replacing 680 parent
    candidates) would silently change what every later child replays. Counted runs therefore
    write only rows that do not exist yet, except that a healthy recovery may replace a
    degraded answer from any producer. Other fresh results still reach
    the immutable ``llm_source`` archive and the run's own result rows.

    Args:
        flag (bool): When True, healthy writes may replace degraded rows by comparing stored
            bytes. All other existing rows are kept and reported as not written,
            which also skips that run's candidate turn-index rows. When False (the default),
            a write replaces the stored row, as force-refresh needs.

    Returns:
        None: This function does not return a value.
    """
    global _CACHE_KEEP_EXISTING
    _CACHE_KEEP_EXISTING = flag


def _cache_db_path() -> str:
    """Return the LLM cache database path (``RUBIKSQL_CACHE_DB``, else under ``RUBIKSQL_DATA_DIR``).

    The turn memo's tables share this file, so the rule lives in :mod:`turn_memo` and both
    readers resolve the same path.

    Args:
        None.

    Returns:
        str: Absolute path of ``llm_cache.db``.
    """
    return _turn_memo.cache_db_path()


def _get_cache_conn() -> sqlite3.Connection:
    # RUBIKSQL_CACHE_DB may point into a not-yet-created sandbox dir; sqlite cannot
    # create parent dirs, and _cache_get/_cache_set swallow the failure — caching
    # would silently no-op for the whole run.
    conn = _turn_memo.connect_cache(_cache_db_path())
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key  TEXT PRIMARY KEY,
                response   TEXT NOT NULL,
                created_at REAL DEFAULT (unixepoch('now'))
            )
        """)
        # Memo tables are created only by writes, preserving frozen-cache reads.
        conn.commit()
    except BaseException:
        conn.close()
        raise
    return conn


#: Most synonym notes to append. Only phrases the question actually contains are appended, so
#: the cap almost never binds (the widest DB stores 60); it exists because "more context is
#: better" is false here — widening the example menu 12 → 20 measured **−2.21pp, ten databases
#: each −1** (T3, pure attention dilution). A question hitting more than this many stored
#: phrases is a question whose wording already matches the schema.
_SYNONYM_HINT_CAP = 12


def _mentions(low_question: str, phrase: str) -> bool:
    r"""Does `low_question` use `phrase` as a word, rather than merely contain its letters?

    Plain substring matching over-fires badly at the lengths these phrases actually have: of the
    332 mined for the 1534-question run, 27 are six characters or shorter and 27 are a single
    word. `'tag'` then hits "advantage", `'post'` hits "postal code", `'eyes'` hits "monkeys",
    `'SM'` hits "SMall". A false note is worse than a missing one — it invites the model to bind
    a column that has nothing to do with the question.

    Boundaries are checked with lookarounds instead of `\b` because these phrases are not all
    word-shaped: `"anti-SM"`, `"aCL IgG"` and `"Percent (%) Eligible"` contain punctuation, and
    `\b` before "(" would never match. `(?<!\w)` / `(?!\w)` only demand that the phrase not be
    glued to a letter or digit, which is exactly the over-firing being ruled out.

    Only the LEFT boundary is required; the phrase may continue on the right. Measured over the
    1534-question set against the 332 mined phrases, counting every raw-substring occurrence and
    hand-labelling the noise:

        plain substring   1440 useful   15 noise   98.97%
        left boundary     1417 useful    0 noise   100.00%   ← this
        left + right      1402 useful    0 noise   100.00%

    All 15 noise cases are left-glued — `'tag'` inside "percen*tag*e", `'Tin'` inside
    "dis*tin*ct", `'AST'` inside "le*ast*", `'converted mana'` inside "*un*converted mana"
    (which means the opposite). Adding a right boundary buys no precision and costs 15 real
    hits, because the right side is where legitimate morphology lives: `'powers'` in
    "superpowers", `'hero name'` in "superhero named", `'single bond'` in "single bonded",
    `'Owner'` in "owneruserid", `'phone number'` in "telephone number".

    Args:
        low_question (str): The question, already lowercased.
        phrase (str): The stored synonym, any case.

    Returns:
        bool: True when the phrase starts at a word boundary.
    """
    return re.search(rf"(?<!\w){re.escape(phrase.lower())}", low_question) is not None


#: Cosine floor for the semantic pass. Set to 0 to disable it.
#:
#: 0.65 is measured, not guessed. Over `superhero` (22 phrases x 40 sampled questions), the
#: LITERAL hits — 100% precise by construction — score min 0.516 / median 0.627 / max 0.770.
#: 0.65 sits just above that median, so a phrase the question does not word literally has to
#: look MORE like the question than a typical literal hit does before it is reported. What that
#: admits, per question: 6.3 extra at 0.50, 2.8 at 0.60, **1.2 at 0.65**, 0.4 at 0.70 — and the
#: ones it admits are the paraphrases the literal pass structurally cannot reach:
#:   0.797  "superpowers"       <- "Which power do superheroes have the most of?"
#:   0.760  "name of superhero" <- "Name the tallest superhero."
#: Picking a floor without measuring is how search_values ended up dropping 28 of the 37
#: literals it existed to find; this one has a stated basis and a way to re-derive it.
#: The floor is the release's ``synonym_near`` knob, and 0.65 — the code default, what a release
#: without the knob runs — is text-embedding-v3's basis. Under text-embedding-v4 the same 22 phrases
#: (the synonyms the 2026-07-27 run stored), against 40 sampled questions, score their literal hits
#: median 0.603 (v3 0.657 on that sample) and 0.65 admits 0.33 a question, a quarter of v3's 1.32;
#: over the 332 stored synonyms of the 11 dev databases the median falls from 0.635 to 0.540 and
#: 0.65 admits 28% of v3's share. The value recalibrated for v4 by the same measurement is in
#: docs/architecture.md (*Cosine floors*).
_SYNONYM_NEAR = _PROMPTS.knob("synonym_near")   # config.knobs; RUBIKSQL_SYNONYM_NEAR still overrides

#: Phrase embeddings, per (process, database). The phrases never change inside a run, so they are
#: embedded once and reused; only the question costs a call per lookup.
_SYN_VEC_CACHE: dict[str, list[tuple[str, list[float]]]] = {}
_SYN_VEC_LOCK = threading.Lock()
_SYN_VEC_STARTS: dict[str, threading.Lock] = {}


def _knowledge_db_path(base: str, db_id: str) -> str:
    """Return one per-database knowledge SQLite path without importing HeavenBase."""
    return os.path.join(base, db_id, f"{db_id}.db")


def _stored_synonym_vectors(db_id: str) -> list[tuple[str, list[float]]] | None:
    """Phrase embeddings as stored by ``SynonymEntity.synonym_vec``, or None if unavailable.

    Read from the workspace SQLite because that is what this module can reach: the semantic pass
    runs inside the candidate process, and `open_workspace` costs 16–55 s there — which is why
    `synonym_vec` is stored to `main` as well as to chroma. heavenbase serialises a Vector into
    the inline column as JSON, so it comes back as text.

    Args:
        db_id (str): Database whose phrases to read.

    Returns:
        list[tuple[str, list[float]]] | None: ``(lowercased phrase, vector)`` pairs, or None when
        the workspace, the table, or the column is absent — a workspace built before the field
        existed. None means "fall back to embedding at runtime", which is different from an empty
        list ("this database has no synonyms"), and the caller acts on that difference.
    """
    for base in (_BIRD_DBS_DIR, os.path.join(_BASE_DATA_DIR, "bird_dbs")):
        ws_db = _knowledge_db_path(base, db_id)
        if os.path.exists(ws_db):
            break
    else:
        return None
    try:
        conn = sqlite3.connect(f"file:{ws_db}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT synonym, synonym_vec FROM synonym_entity WHERE db_id = ? "
                "ORDER BY object_id", (db_id,)
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return None                      # no such column on a pre-field workspace
    out: list[tuple[str, list[float]]] = []
    for syn, raw in rows:
        if not syn or not raw:
            continue
        try:
            vec = json.loads(raw) if isinstance(raw, str) else list(raw)
        except Exception:
            continue
        if vec:
            out.append((syn.lower(), vec))
    # Every row lacking a vector is the pre-backfill state; treat that as "nothing stored" so the
    # caller embeds instead of silently searching a fraction of the table.
    return out if len(out) == len([r for r in rows if r[0]]) else None


def _synonym_near(db_id: str, question: str, rows: list, *, exclude: set) -> set:
    """Stored phrases whose MEANING is close to the question, beyond the ones it words literally.

    The literal pass cannot fire on a paraphrase — "how much does the hero weigh" never reaches
    the stored phrase "weight". This is the complement: cosine over the same phrases, returning
    only what the literal pass missed, so the two are a union and this can never remove a hit.

    ON by default, at the cosine floor the release's ``synonym_near`` knob gives (see
    :data:`_SYNONYM_NEAR`; a floor of 0, or ``RUBIKSQL_SYNONYM_NEAR=0``, disables it). It costs one
    embedding call per question — for the question only, since the phrases' vectors are read from
    the workspace — and the floor was picked from the distance distribution of the phrases the
    literal pass already hits (under text-embedding-v3: min .516 / median .627 / max .770, hence
    0.65), not measured against scores; docs/architecture.md (*Cosine floors*) has its
    text-embedding-v4 recalibration.

    Args:
        db_id (str): Database whose phrases to search.
        question (str): The question, unmodified.
        rows (list): ``(synonym, canonical, table, col)`` tuples already read for this database.
        exclude (set): Lowercased phrases the literal pass already matched; never re-reported.

    Returns:
        set: Lowercased phrases above the floor. Empty when the pass is off, when the embedding
        API is unreachable, or when nothing clears the floor — a semantic nicety must never be
        the reason a question fails to get an answer.
    """
    if _SYNONYM_NEAR <= 0:
        return set()
    try:
        import numpy as np
        from .embedding import embed
        with _SYN_VEC_LOCK:
            start_lock = _SYN_VEC_STARTS.setdefault(db_id, threading.Lock())
        with session_timing.locked(start_lock, "synonym_vectors"):
            cached = _SYN_VEC_CACHE.get(db_id)
            if cached is None:
                # Prefer the vectors already computed by SynonymEntity at upsert.
                cached = _stored_synonym_vectors(db_id)
                if cached is None:
                    # Pre-vector workspaces earn each phrase once per process, even
                    # when concurrent questions reach the cold cache together.
                    cached = [(s.lower(), embed(s)) for s, _c, _t, _co in rows if s]
                _SYN_VEC_CACHE[db_id] = cached
        if not cached:
            return set()
        qv = np.asarray(embed(question), dtype="float32")
        qn = float(np.linalg.norm(qv)) or 1.0
        out = set()
        for phrase, vec in cached:
            if phrase in exclude:
                continue
            v = np.asarray(vec, dtype="float32")
            vn = float(np.linalg.norm(v)) or 1.0
            if float(v @ qv) / (vn * qn) >= _SYNONYM_NEAR:
                out.add(phrase)
        return out
    except Exception:
        return set()


def _log_prompt(verbose: bool, system: str, user_content: str) -> None:
    """Mirror the initial model inputs and label display-only metadata.

    Args:
        verbose (bool): Whether to print the model-facing trace.
        system (str): The system prompt as sent.
        user_content (str): The complete first user message as sent.
    """
    _log(
        verbose,
        "说明：带此前缀的行以及 ▶/◀ 箭头都是额外显示，真实模型上下文中没有；"
        "箭头后的正文或结构化字段才是模型实际收发的内容。",
    )
    _log(verbose, f"system ({len(system)} chars)")
    _trace(verbose, "▶", system)
    _log(verbose, f"user ({len(user_content)} chars)")
    _trace(verbose, "▶", user_content)


def _synonym_context(db_id: str, question: str) -> str:
    """Vocabulary notes for whichever stored synonyms this question actually uses.

    The synonym table answers "what does the user's word mean in this schema" — mined from each
    question's own ``evidence`` by the pipeline's synonyms stage. It used to reach the model
    through a mandatory ``expand_synonyms`` step that ran ``str.replace`` over the question, and
    that measured **−1.00pp** on top of two-stage examples: rewriting the question destroys the
    input every downstream vector lookup depends on, so ``get_examples`` stopped seeing what the
    user actually asked. The step was removed; the table then went unread, because the surviving
    path (``query_knowledge(entity="synonym")``) is optional and the model never chose it — over
    the 1534-question run it queried synonyms **0 times** out of 7,630 candidate runs.

    So this pushes instead of waiting to be pulled, and **appends rather than rewrites**: the
    question reaches the model and every retrieval verbatim, with the notes as a separate block
    below it. Read straight from the workspace SQLite (same access pattern as
    :func:`_db_knowledge_fingerprint`) — no MCP round-trip, no LLM call, no tool budget.

    Args:
        db_id (str): Database whose workspace to read.
        question (str): The question, unmodified. Matching is case-insensitive substring.

    Returns:
        str: A block to append to the user message, starting with a newline, or ``""`` when the
        database has no synonyms, none of them appear in the question, or the workspace is absent.
        Never raises: a missing or unreadable table must not fail a question that would otherwise
        answer fine, and the caller cannot distinguish "no synonyms" from "read failed" anyway —
        the difference is reported on stderr by the synonyms stage, not here.
    """
    hits = _synonym_hits(db_id, question)
    return "" if hits is None else _PROMPTS.surface("shared", "synonyms", hits=hits)


# Cold synonym/embedding waits must leave the default executor available to the
# provider's HTTP request reservations, just like judge pacing in llm_runtime.
_SYNONYM_EXECUTOR = _futures.ThreadPoolExecutor(max_workers=32, thread_name_prefix="synonym-context")


async def _async_synonym_context(db_id: str, question: str) -> str:
    """Prepare synonym retrieval off-loop and render on the owning recorder task.

    Args:
        db_id: Database whose synonym table and phrase vectors are read.
        question: Unmodified question to match and embed.

    Returns:
        The same note as :func:`_synonym_context`. Cancelling the await cannot
        render later into the abandoned session's prompt recorder.

    Raises:
        asyncio.CancelledError: The owning deadline stopped this task; nothing is embedded.
    """
    refuse_if_stopped()
    with session_timing.measure("gate", "synonym_context"):
        context = contextvars.copy_context()
        hits = await asyncio.get_running_loop().run_in_executor(
            _SYNONYM_EXECUTOR, context.run, _synonym_hits, db_id, question)
    return "" if hits is None else _PROMPTS.surface("shared", "synonyms", hits=hits)


def _synonym_hits(db_id: str, question: str) -> list[dict] | None:
    """Read and match synonym rows without rendering model-facing text."""
    for base in (_BIRD_DBS_DIR, os.path.join(_BASE_DATA_DIR, "bird_dbs")):
        ws_db = _knowledge_db_path(base, db_id)
        if os.path.exists(ws_db):
            break
    else:
        return None
    try:
        conn = sqlite3.connect(f"file:{ws_db}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT synonym, canonical, table_name, col_name FROM synonym_entity "
                "WHERE db_id = ? ORDER BY object_id", (db_id,)
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        return None

    low = question.lower()
    literal = {s.lower() for s, _c, _t, _co in rows if s and _mentions(low, s)}
    near = _synonym_near(db_id, question, rows, exclude=literal)

    # Retrieval, matching, dedup and the cap stay here; the frame around the hits — the
    # heading, the per-hit line with its near-match mark, and the reference-not-instruction
    # trailer — is the shared.synonyms surface, which renders "" for an empty list.
    hits: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for syn, canonical, table, col in rows:
        if not syn or not canonical:
            continue
        sl = syn.lower()
        if sl in literal:
            is_near = False
        elif sl in near:
            is_near = True
        else:
            continue
        target = f"{table}.{col}" if table and col else str(canonical)
        key = (sl, target)
        if key in seen:                      # same phrase mined twice for one column
            continue
        seen.add(key)
        hits.append({"syn": str(syn), "target": target, "near": is_near})
        if len(hits) >= _SYNONYM_HINT_CAP:
            break
    return hits


def _db_knowledge_fingerprint(db_id: str) -> str:
    """Return the unchanged cache identity from the actual selected readonly workspace."""
    from pathlib import Path
    from .knowledge_identity import knowledge_fingerprint

    for base in (_BIRD_DBS_DIR, os.path.join(_BASE_DATA_DIR, "bird_dbs")):
        database = _knowledge_db_path(base, db_id)
        if os.path.exists(database):
            return knowledge_fingerprint(Path(database), embedding_model=EMBEDDING_MODEL)
    return ""

def _cache_payload(model: str, messages: list, seed: Optional[int], db_id: str,
                   disabled_tools: set[str] | None, route_extras: dict | None,
                   slot: int | None = None) -> dict:
    """Build the candidate cache-key payload; `_cache_key` and `_cache_key_pair` hash it.

    Args:
        model (str): Candidate model name.
        messages (list): The keyed request, ``[{"db_id", "question"}]``.
        seed (Optional[int]): Sampling seed, ``None`` when the provider has none.
        db_id (str): Database whose knowledge fingerprint enters the key.
        disabled_tools (set[str] | None): Tools withheld from this run.
        route_extras (dict | None): Per-candidate route terms (provider, temperature, style ...).
        slot (int | None): The candidate's position in the pool; a slot whose own document
            (``candidate.s<k>``) differs from the shared one keys on its own keyspace.

    Returns:
        dict: The key terms, including this candidate route's implementation digest.
            ``prompt`` and ``ns`` are present only when they apply.
    """
    payload_dict = {"model": model,
                    "system": _SYSTEM,
                    "messages": messages,
                    "seed": seed,
                    "kb": _db_knowledge_fingerprint(db_id),
                    "disabled": sorted(disabled_tools) if disabled_tools else None,
                    # Bump when the SHAPE of the user message changes for reasons the other keys
                    # cannot see. `kb` already tracks the synonym table's CONTENT, so a re-mined
                    # table invalidates on its own — but "are matching synonyms appended to the
                    # question at all" is a code-level behaviour, and every cache entry written
                    # before _synonym_context existed holds an answer produced WITHOUT them.
                    # Without this marker those entries replay and the change measures as no-op.
                    # The same trap exists one layer down, for a change to what a TOOL RETURNS:
                    # Rule/example retrievals now return session-local R/E ids, and example
                    # expansion reads a retained payload instead of querying storage again.
                    # Old transcripts carry persistent ids that are invalid under this contract.
                    # rule_tools_v2: `get_rules(k)` returns k rules, not 2k. `k` used to be
                    # the per-branch width AND the trim, so a parameter promising 10 returned
                    # 20. The default moved 10 -> 20 in the same change, so the candidate is
                    # still handed 20 rules and only the parameter's meaning moved. The bump
                    # stays because the rule BLOCK itself changed (render_v6 drops `tags`),
                    # and because `kb` cannot see either: the stored rule rows are untouched,
                    # which is exactly the "derivation code changed, inputs did not" case its
                    # own docstring disclaims. Without it a cached run replays a pre-change
                    # transcript and measures both changes as nil.
                    # final_sql_budget_v1: run_sql_base and submit_final_sql now execute under
                    # the 30s official limit instead of the 20-minute exploration budget, so a
                    # pathologically slow statement comes back as a cancellation where it used
                    # to come back as rows. That is precisely the "a change to what a TOOL
                    # RETURNS" case above, and `kb` cannot see it — the stored knowledge is
                    # untouched and only the executor's budget moved. Without the bump a
                    # cached run replays a transcript in which the slow SQL was accepted, and
                    # the change measures as nil on exactly the questions it exists for.
                    "msg_shape": (
                        "syn_ctx_v1|virtual_knowledge_ids_v1|question_hint_v1|"
                        "knowledge_text_v2|rule_tools_v2|rule_scope_v1|final_sql_budget_v1"
                    ),
                    "model_protocol": MODEL_PROTOCOL,
                    "transport_protocol": selected_transport_protocol(base_protocol=TRANSPORT_PROTOCOL)}
    # HET1 heterogeneous routes (per-candidate temperature / prompt style) get their own
    # cache namespace only when set. ``MODEL_PROTOCOL`` above deliberately invalidates
    # every pre-text-action transcript; there is no old-key fallback or migration path.
    if route_extras:
        payload_dict["route"] = dict(sorted(route_extras.items()))
    # The loop knobs join this list because _RUN_SQL_MAX_ROWS changes the JSON rows the
    # model reads; _TIME_NUDGE_SEC, _LOOP_GUARD/_LOOP_GUARD_TIERS and
    # _EMPTY_PROBE/_EMPTY_PROBE_AT change whether and when a transcript intervention fires.
    # Unset means absent within the current
    # protocol namespace; compatibility with older protocol keys is intentionally irrelevant.
    # Only list a knob that changes what the CANDIDATE does. A judge-only change must NOT be
    # listed here: namespacing the candidate key on it would make the arm re-generate its whole
    # candidate pool, so it would differ from baseline in TWO ways (judge behaviour AND a freshly
    # sampled pool) and no delta could be attributed. Judge-side switches belong on the judge
    # cache key (vote.py) instead.
    # A knob that gets PROMOTED to production leaves this list with its flag: once the behaviour
    # is unconditional there is no second arm to namespace apart from, and keeping the dead name
    # here would hand anyone who still has it exported a silently different key for identical
    # behaviour. RUBIKSQL_QUERY_KB and RUBIKSQL_NO_SYNONYM_STEP were removed on those grounds.
    # RUBIKSQL_FINAL_SQL_TIMEOUT_SEC and RUBIKSQL_MAX_SLOW_SQL_REJECTS join on the same
    # grounds: the budget decides whether a slow statement comes back as rows or as a
    # cancellation, and the cap decides how many rewrites the model is asked for — both
    # change what the CANDIDATE does. (The 1200s exploration budget stays off this list: it
    # is high enough never to fire, so it alters no transcript.) RUBIKSQL_SYNONYM_NEAR and
    # RUBIKSQL_SYMBOL_NEAR join on the same grounds: the synonym floor decides the synonym
    # block of the candidate's message and the glyph floor which values search_values
    # returns, while the keyed request is only {db_id, question}; the release's own values
    # of both move the `prompt` term below, an override of either moved nothing. Each is read
    # only when SET, so an unset env leaves the default namespace exactly where it was and no
    # operator override can silently share a cache with the production default.
    #
    # ANY future behaviour flag belongs here, and the failure mode if it does not is not a
    # wrong number but a silent one. A cache entry is one WHOLE question-run keyed on
    # {db_id, question} + model + seed + kb + protocols + route, and a hit returns before
    # the tool loop, so `get_rules` never runs. A retrieval-scoring change touches no rule
    # ROW, so `_db_knowledge_fingerprint` does not move — an arm testing it would replay the
    # baseline's answers and score identically, and the conclusion would be "the change does
    # nothing" when the change never happened. Measured on 08-11, before it could cost an
    # experiment. `build_run_identity` needs the same entry for the same reason, or
    # `--resume` skips the eval stage outright and hides the replay a second time.
    for _ev in ("RUBIKSQL_SEARCH_COLUMNS_PROFILE",
                "RUBIKSQL_RUN_SQL_MAX_ROWS",
                "RUBIKSQL_MAX_TURNS", "RUBIKSQL_MAX_SUBMIT_NUDGES",
                "RUBIKSQL_TIME_NUDGE_SEC", "RUBIKSQL_LOOP_GUARD", "RUBIKSQL_LOOP_GUARD_TIERS",
                "RUBIKSQL_EMPTY_PROBE", "RUBIKSQL_EMPTY_PROBE_AT",
                "RUBIKSQL_FINAL_SQL_TIMEOUT_SEC", "RUBIKSQL_MAX_SLOW_SQL_REJECTS",
                "RUBIKSQL_SYNONYM_NEAR", "RUBIKSQL_SYMBOL_NEAR"):
        _v = os.environ.get(_ev)
        if _v is not None:
            payload_dict[_ev.lower()] = _v
    # Prompt release: the key hashes `system` above, but a candidate also receives texts
    # the system prompt cannot see — nudges, notes, grain corrections, the hook and synonym
    # frames, the route block. `prompt` is the keyspace of this role/route's closure, and
    # it is present ONLY when the static texts of that closure's surfaces differ from r0's
    # (Release.surface_identity): r0 itself and every refactor-only release — the
    # one-document re-cut included — leave the key byte-identical, and a judge-only change
    # never re-keys the candidate pool (the pairing principle above). `ns` is the sampling
    # universe (RUBIKSQL_CACHE_NS), present only when set. Both are absent-by-default so
    # json.dumps(sort_keys=True) emits exactly today's bytes when neither applies. A candidate
    # slot with a document of its own keys on that slot's keyspace, so a slot document moves
    # the keys of its slot and no other; a slot whose texts equal the shared document's keys
    # exactly as before.
    _prompt_extra = _PROMPTS.key_extra("candidate", (route_extras or {}).get("style"), slot)
    if _prompt_extra:
        payload_dict["prompt"] = _prompt_extra
    _cache_ns = os.environ.get("RUBIKSQL_CACHE_NS", "")
    if _cache_ns:
        payload_dict["ns"] = _cache_ns
    contract = tool_contract.role_contract(
        "candidate", route=(route_extras or {}).get("style") or "default",
        disabled=disabled_tools,
    )
    payload_dict["implementation"] = contract["digest"]
    return payload_dict


def _hash_cache_payload(payload_dict: dict) -> str:
    """The sha256 of a key payload in its canonical JSON form (sorted keys, unicode kept)."""
    payload = json.dumps(payload_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _cache_key(model: str, messages: list, seed: Optional[int],
               db_id: str = "",
               disabled_tools: set[str] | None = None,
               route_extras: dict | None = None,
               *, base: bool = False, slot: int | None = None) -> str:
    """Return the candidate cache key: the sha256 of the canonical key payload.

    Args:
        model (str): Candidate model name.
        messages (list): The keyed request, ``[{"db_id", "question"}]``.
        seed (Optional[int]): Sampling seed, ``None`` when the provider has none.
        db_id (str): Database whose knowledge fingerprint enters the key.
        disabled_tools (set[str] | None): Tools withheld from this run.
        route_extras (dict | None): Per-candidate route terms (provider, temperature, style ...).
        base (bool): Return the turn memo's BASE key instead: the same payload without its
            ``prompt`` term, i.e. the key r0 rules produce for this run, under which runs that
            agree on everything but prompt content share recorded turns (turn_memo.py).
        slot (int | None): The candidate's position in the pool (see `_cache_payload`).

    Returns:
        str: 64 hex characters.
    """
    payload_dict = _cache_payload(model, messages, seed, db_id, disabled_tools, route_extras, slot)
    if base:
        payload_dict.pop("prompt", None)
    return _hash_cache_payload(payload_dict)


def _cache_key_pair(model: str, messages: list, seed: Optional[int], db_id: str,
                    disabled_tools: set[str] | None, route_extras: dict | None,
                    slot: int | None = None) -> tuple[str, str]:
    """Return one run's cache key and its turn-memo base key from a single payload build.

    The payload reads the database's knowledge fingerprint (a scan of its rules and
    examples), so the loop asks for both keys at once instead of building the payload
    twice; under r0 with no ``prompt`` term the two keys are the same string.

    Args:
        model (str): Candidate model name.
        messages (list): The keyed request, ``[{"db_id", "question"}]``.
        seed (Optional[int]): Sampling seed, ``None`` when the provider has none.
        db_id (str): Database whose knowledge fingerprint enters the key.
        disabled_tools (set[str] | None): Tools withheld from this run.
        route_extras (dict | None): Per-candidate route terms (provider, temperature, style ...).
        slot (int | None): The candidate's position in the pool (see `_cache_payload`).

    Returns:
        tuple[str, str]: ``(cache key, base key)``, each 64 hex characters.
    """
    payload_dict = _cache_payload(model, messages, seed, db_id, disabled_tools, route_extras, slot)
    key = _hash_cache_payload(payload_dict)
    if "prompt" not in payload_dict:
        return key, key
    return key, _hash_cache_payload({name: value for name, value in payload_dict.items() if name != "prompt"})


# One line per run, not one per call: a broken cache fails on every single call, and the
# point is only to make a dead cache visible in the log instead of invisible.
_cache_broken = False


def _cache_report_broken(op: str, exc: BaseException) -> None:
    """Announce, once per process, that the LLM cache is not working.

    A cache failure never costs correctness — a failed read is a miss, a failed write is a
    re-paid call — so both paths keep swallowing. But an entire multi-hour run can execute with
    the cache dead (read-only volume, full disk, `database is locked` under the parallel
    workers, a corrupt file) and look identical in the logs, which is how the missing-parent-dir
    bug noted above stayed hidden for a whole run.
    """
    global _cache_broken
    if _cache_broken:
        return
    _cache_broken = True
    print(
        f"{_LOG_META_PREFIX} [cache] LLM cache {op} FAILED "
        f"({type(exc).__name__}: {exc}) at {_cache_db_path()} — this run is proceeding "
        "WITHOUT cache; every call is being re-paid. Further cache errors are not reported.",
        file=sys.stderr,
        flush=True,
    )


def _cache_answer_record(response: dict) -> dict:
    """Return the producing role's record from a candidate or judge/reviewer answer."""
    return response.get("judge_debug", response)


_CACHE_PARENT_POOL: dict[str, dict] = {}


def set_cache_parent_pool(parent_run: str | None, *, parent_identity: dict | None = None) -> None:
    """Bind the intentional paired parent pool; moved candidate keys get no exemption.

    Args:
        parent_run: Parent directory supplied by the paired evaluation launcher, or None.
        parent_identity: Paired parent identity from the verified launch receipt.
    """
    _CACHE_PARENT_POOL.clear()
    if not parent_run or not parent_identity:
        return
    from pathlib import Path
    from .prompts import evolve, load_release
    from .prompts.gate import changed_surfaces, keyspace_moves

    release = load_release(parent_identity["release"], allow_retired=True)
    if (any(label.startswith("candidate") for label in keyspace_moves(_PROMPTS, release))
            or changed_surfaces(_PROMPTS, release) & evolve.KNOWLEDGE_SURFACES):
        return
    parent = evolve.read_run(Path(parent_run))
    if ({"run": parent.name, "release": parent.release, "namespace": parent.ns,
         "created_at": parent.created_at, "manifest_sha256": parent.manifest_sha,
         "result_files": parent.result_files} != parent_identity):
        raise ValueError("cache parent pool differs from the admitted paired parent")
    for row in parent.rows.values():
        for label, unit in evolve._units(row).items():
            proof = unit.get("source_proof")
            if label.startswith("c") and cache_lineage.valid_descriptor(proof, attributed=True):
                _CACHE_PARENT_POOL[proof["id"]] = unit


def _cache_parent_replay(response: dict) -> bool:
    """Exempt only the exact immutable candidate that the bound parent measured."""
    from .prompts.evolve import replays_parent_unit

    proof = response.get("cache_source") or {}
    unit = _CACHE_PARENT_POOL.get(proof.get("id"))
    record = _cache_answer_record(response)
    return bool(unit and not cache_lineage.owned_by_evaluation(record)
                and replays_parent_unit(unit, response, slot=unit["index"] + 1))


def _cache_get(key: str) -> Optional[dict]:
    if _NO_CACHE:
        return None
    with session_timing.locked(_CACHE_LOCK, "cache_lock"), session_timing.measure("gate", "cache_read"):
        try:
            with closing(_get_cache_conn()) as conn:
                row = conn.execute("SELECT response FROM llm_cache WHERE cache_key = ?", (key,)).fetchone()
            hit = restore_sql_results(json.loads(row[0])) if row else None
        except (sqlite3.Error, json.JSONDecodeError, OSError) as e:
            _cache_report_broken("read", e)
            return None
    # Under skip-barren, an entry that reached the end without SQL is a miss: serving it
    # would hand a retry the same dead end it is being run to escape. See set_cache_skip_barren.
    if hit is not None and _CACHE_SKIP_BARREN and not (hit.get("sql") or "").strip():
        return None
    # Keep turn replay enabled: matching prefixes still execute tools live, so a
    # recovered knowledge result changes the next request instead of replaying a bad answer.
    if hit is not None and degraded_record(_cache_answer_record(hit)) and not _cache_parent_replay(hit):
        return None
    return hit


def _cache_set(key: str, response: dict) -> bool:
    """Write one cache row and say whether it landed.

    ``_NO_CACHE`` only suppresses reads: fresh results still write to the cache under
    the selected replacement policy. Used by candidates, judges and reviewers. The return
    value gates candidate turn-index rows, so a kept row or a stand-in that swallows the
    write also suppresses that candidate's index update.

    Under `set_cache_keep_existing` (counted runs) the write is ``INSERT OR IGNORE``,
    except for healthy recovery of a degraded answer from any producer.
    Replacement compares the stored bytes so another writer's row cannot be overwritten.

    Args:
        key (str): Cache key.
        response (dict): The JSON-serialisable record.

    Returns:
        bool: True when the row was written; False when the cache is broken or, under
            `set_cache_keep_existing`, when the stored row remains protected.
    """
    verb = "INSERT OR IGNORE" if _CACHE_KEEP_EXISTING else "INSERT OR REPLACE"
    with session_timing.locked(_CACHE_LOCK, "cache_lock"), session_timing.measure("gate", "cache_write"):
        try:
            with closing(_get_cache_conn()) as conn:
                payload = json.dumps(response, ensure_ascii=False, default=binary_json_default)
                written = conn.execute(f"{verb} INTO llm_cache (cache_key, response) VALUES (?, ?)",
                                       (key, payload)).rowcount
                record = _cache_answer_record(response)
                if (_CACHE_KEEP_EXISTING and not written and (response.get("sql") or "").strip()
                        and not response.get("error") and not record.get("error") and not degraded_record(record)
                        and cache_lineage.owned_by_evaluation(record)):
                    row = conn.execute("SELECT response FROM llm_cache WHERE cache_key = ?", (key,)).fetchone()
                    if row is not None and degraded_record(_cache_answer_record(json.loads(row[0]))):
                        written = conn.execute(
                            "UPDATE llm_cache SET response = ? WHERE cache_key = ? AND response = ?",
                            (payload, key, row[0]),
                        ).rowcount
                conn.commit()
            return written > 0
        except (sqlite3.Error, json.JSONDecodeError, OSError) as e:
            _cache_report_broken("write", e)
            return False


def _turn_memo_report_broken(op: str, exc: BaseException) -> None:
    """Announce, once per process, that the turn memo's tables cannot be used.

    The memo has its own alarm because its failures are not the LLM cache's: a read-only
    cache file or a ``database is locked`` on an index write leaves every ``llm_cache`` hit
    intact and only stops recording and replay. Routing it through `_cache_report_broken`
    would announce a dead cache that is serving fine and silence the cache's own alarm.
    `turn_memo.TurnStore` calls this at most once.
    """
    print(
        f"{_LOG_META_PREFIX} [cache] turn memo {op} FAILED "
        f"({type(exc).__name__}: {exc}) at {_cache_db_path()} — this run records and replays "
        "no turns from here on; the LLM cache itself is unaffected. Further memo errors are "
        "not reported.",
        file=sys.stderr,
        flush=True,
    )


#: The turn memo's tables, behind the same connection factory and lock as the LLM cache
#: but with their own one-shot alarm (turn_memo.py). Recording is unconditional; lookups
#: are gated below.
_TURN_STORE = _turn_memo.TurnStore(_get_cache_conn, lock=_CACHE_LOCK, report=_turn_memo_report_broken)


def _backfill_turn_index(cache_key: str, base_key: str, record: dict, tools: frozenset[str]) -> bool:
    """Write the turn index of a whole-run cache hit when the stored transcript has none.

    A root run replayed from an ``llm_cache`` written before the memo existed (or on another
    machine) would otherwise leave no replay source for the proposals evaluated against it:
    every child unit would report ``no-source`` and be re-sampled. The rows are rebuilt from
    the stored ``messages`` with unknown clock readings (`turn_memo.transcript_rows`).

    Args:
        cache_key (str): The hit's cache key (the ``llm_cache`` row holding ``messages``).
        base_key (str): The run's base key, the same payload without its ``prompt`` term.
        record (dict): The cached record.
        tools (frozenset[str]): The route's exposed tool names. The key carries the role
            contract digest, so the recording run exposed exactly these.

    Returns:
        bool: Whether rows were written now.
    """
    messages = record.get("messages")
    if _TURN_STORE is None or not isinstance(messages, list):
        return False
    ns = os.environ.get("RUBIKSQL_CACHE_NS", "")
    replayable = partial(_replayable_calls, tools=tools, candidate=True)
    rows = [row for row in _turn_memo.transcript_rows(base_key, ns, messages)
            if row.idx == _turn_memo.START_IDX
            or _turn_memo.reply_replayable(messages[row.idx], replayable)]
    if len(rows) == 1:   # no assistant message: nothing to index
        return False
    if _TURN_STORE.lookup(rows[0].turn_key) is not None:   # the chain-start row: already indexed
        return False
    source = record.get("cache_source")
    ref = cache_lineage.REF + source["id"] if cache_lineage.valid_descriptor(source) else _turn_memo.REF_CACHE + cache_key
    return _TURN_STORE.write_turns(ref, ns, rows)


def _turn_lookup_allowed(role: str, route: str | None, slot: int | None = None) -> bool:
    """Whether a run of this role/route may replay recorded turns.

    Replay is worth doing only when the role's cache key carries a ``prompt`` term — its
    closure differs from r0's, so the whole-run cache misses by construction — and never
    under ``--no-cache`` or skip-barren, which exist to force fresh generation. Under r0
    with no namespace this is always False, so nothing is looked up and provider calls are
    exactly what they were.

    Args:
        role (str): ``candidate``, ``judge`` or ``reviewer``.
        route (str | None): The route, ``None`` for the default.
        slot (int | None): A candidate's position in the pool, whose own document may move its key alone.

    Returns:
        bool: The lookup switch for a :class:`turn_memo.TurnMemo`.
    """
    if _NO_CACHE or _CACHE_SKIP_BARREN:
        return False
    try:
        return bool(_PROMPTS.key_extra(role, route, slot))
    except KeyError:
        return False


def _replayed_turn(message: dict, think: str) -> ModelTurn:
    """Rebuild a provider turn from a recorded assistant message; the replay cost nothing.

    Args:
        message (dict): The recorded ``{"role": "assistant", "content", "tool_calls"?}`` message.
        think (str): The recorded reasoning text, empty when none was kept.

    Returns:
        ModelTurn: With the message as sent, its native tool calls, and usage ``{}``.
    """
    return ModelTurn(
        text=str(message.get("content") or ""),
        think=think,
        tool_calls=list(message.get("tool_calls") or []),
        message=message,
        usage={},
    )


def _candidate_turn_memo(base_key: str, clock: RunClock, recorder: RenderRecorder,
                         style: str | None, slot: int | None = None, *, tools: frozenset[str]) -> TurnMemo:
    """Create the turn memo of one openai-shaped candidate run.

    Args:
        base_key (str): The run's base key, ``_cache_key(..., base=True)``.
        clock (RunClock): The run's logical clock.
        recorder (RenderRecorder): The run's render recorder, read for divergence attribution.
        style (str | None): The candidate route.
        slot (int | None): The candidate's position in the pool.
        tools (frozenset[str]): The tool names this run's loop exposes and dispatches.

    Returns:
        TurnMemo: Recording always; replaying when :func:`_turn_lookup_allowed` says so.
    """
    return TurnMemo(
        _TURN_STORE,
        base_key=base_key,
        ns=os.environ.get("RUBIKSQL_CACHE_NS", ""),
        clock=clock,
        rebuild=_replayed_turn,
        lookup=_turn_lookup_allowed("candidate", style, slot),
        recorder=recorder,
        release=_PROMPTS,
        role="candidate",
        route=style,
        slot=slot,
        validate_calls=partial(_replayable_calls, tools=tools, candidate=True),
    )


@contextmanager
def _candidate_failure_record(record: dict, recorder: RenderRecorder) -> Iterator[None]:
    """Attach an interrupted candidate's observed evidence to its original exception.

    The gather boundary reads ``candidate_record``; direct callers still receive the
    same exception. Cancellation and process interrupts propagate without conversion.
    No failed answer or unfinished turn is entered into the success/replay cache.
    """
    started = time.monotonic()
    timing = session_timing.current()
    def snapshot():
        current = {key: value for key, value in record.items() if key not in ("memo", "clock", "completed_record")}
        if timing is not None:
            current["timing"] = timing.snapshot()
        memo = record.get("memo")
        if memo is not None:
            recorder.set_replay(**memo.replay_fields())
            current["reasoning"] = memo.thinks()
        current["prompt_identity"] = recorder.snapshot()
        return current
    observer = cache_lineage.observe(record.get("execution", {}), snapshot)
    try:
        yield
    except (Exception, asyncio.CancelledError) as error:
        # Attach observed data first. Clock, render and archive metadata are
        # secondary: none may erase the fragment or replace the original error.
        memo = record.pop("memo", None)
        clock = record.pop("clock", None)
        record.pop("completed_record", None)
        record.update(duration_sec=round(time.monotonic() - started, 3),
                      sql="", transcript_status="interrupted", error=str(error) or type(error).__name__,
                      capture_reason="async_cancel" if isinstance(error, asyncio.CancelledError) else "agent_error")
        record["timing"] = session_timing.current_snapshot()
        from .llm_runtime import provider_failure
        if provider_failure(error):
            record["provider_failure"] = True
        record.setdefault("reasoning", [])
        error.candidate_record = record
        try:
            if memo is not None:
                record["reasoning"] = memo.thinks()
                recorder.set_replay(**memo.replay_fields())
            record["prompt_identity"] = recorder.snapshot()
            if clock is not None:
                record["duration_sec"] = round(clock.real_elapsed(), 3)
        except Exception as recording_error:
            record["recording_error"] = str(recording_error)
            error.add_note(f"candidate metadata could not be retained: {type(recording_error).__name__}: {recording_error}")
        cache_lineage.retain_source(_TURN_STORE, record)
        raise
    finally:
        completed = record.pop("completed_record", None)
        cache_lineage.finish(observer, completed if completed is not None else
                             record if record.get("transcript_status") == "interrupted" else None)


# ── MCP server subprocess ─────────────────────────────────────────────────────
# Uses `python -m rubiksql_hb.server` — works whether run from project root
# or imported from an installed package.

_BASE_DATA_DIR = os.environ.get("RUBIKSQL_DATA_DIR", os.path.expanduser("~/.rubiksql_hb"))
# bird_dbs/ may live next to this package's src root (dev layout) or under _BASE_DATA_DIR
_BIRD_DBS_DIR = os.environ.get("RUBIKSQL_BIRD_DBS_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "bird_dbs",
)

#: The environment variable the answering servers read their retrieval generation token from
#: (`server._retrieval_open_kwargs`).
RETRIEVAL_VERSION_ENV = "RUBIKSQL_RETRIEVAL_VERSION"
#: A `RETRIEVAL_VERSION_ENV` value that asks for each database's own knowledge fingerprint.
RETRIEVAL_VERSION_PER_DB = "fingerprint"


def _server_retrieval_version(env: dict[str, str], db_id: str | None) -> None:
    """Resolve the per-database retrieval token one answering server receives, in place.

    One answering process starts servers for several databases from one environment, but
    each server serves one database, and HB keeps its versioned candidate cache inside that
    server's own workspace. ``RUBIKSQL_RETRIEVAL_VERSION=fingerprint`` therefore becomes
    ``kb-<fingerprint>`` for each server, the same knowledge fingerprint the candidate,
    judge and reviewer cache keys already carry, so the token names the generation the
    server reads. The token itself never enters a cache key. A database without its own
    workspace gets no token and keeps live reads. Any other value passes through unchanged.

    Args:
        env (dict[str, str]): The server's environment, updated in place.
        db_id (str | None): The database the server serves; ``None`` for the shared workspace.
    """
    if (env.get(RETRIEVAL_VERSION_ENV) or "").strip() != RETRIEVAL_VERSION_PER_DB:
        return
    fingerprint = _db_knowledge_fingerprint(db_id) if db_id else ""
    if fingerprint:
        env[RETRIEVAL_VERSION_ENV] = f"kb-{fingerprint}"
    else:
        env.pop(RETRIEVAL_VERSION_ENV, None)


# Per-db HeavenBase directories live under bird_dbs/{db_id}/
# Fall back to the shared workspace when no per-db directory exists.
def _mcp_params(db_id: str | None = None) -> StdioServerParameters:
    from pathlib import Path
    from .prompts.snapshot import verify_snapshot, snapshot_env
    from .prompts.binding import binding_arguments
    arguments = ["-m", "rubiksql_hb.server", *binding_arguments()]
    snapshot = verify_snapshot(Path(os.environ["RUBIKSQL_PROMPT_SNAPSHOT"]))
    execution_env = {**os.environ, **snapshot_env(snapshot, bundle=True)}
    execution_env.pop("RUBIKSQL_PROMPT_CONTROL", None)
    if db_id:
        # check project-local bird_dbs/ first, then _BASE_DATA_DIR/bird_dbs/
        for base in (_BIRD_DBS_DIR, os.path.join(_BASE_DATA_DIR, "bird_dbs")):
            per_db_dir = os.path.join(base, db_id)
            if os.path.isdir(per_db_dir):
                break
        else:
            per_db_dir = None
        if per_db_dir and os.path.isdir(per_db_dir):
            env = {**execution_env,
                   "RUBIKSQL_DATA_DIR": per_db_dir,
                   "RUBIKSQL_WORKSPACE_ID": db_id,
                   "CHROMA_PATH": os.path.join(per_db_dir, "chroma_db")}
            _server_retrieval_version(env, db_id)
            return StdioServerParameters(command=sys.executable,
                                         args=arguments, env=env)
    _server_retrieval_version(execution_env, None)
    return StdioServerParameters(command=sys.executable,
                                 args=arguments,
                                 env=execution_env)


# ── Shared per-db MCP session ─────────────────────────────────────────────────
# One background thread + event loop keeps an MCP server subprocess alive for each
# (db_id, server slot). All worker threads submit coroutines to that loop via
# run_coroutine_threadsafe — used by both single-agent (nl2sql) and vote (nl2sql_vote)
# modes. RUBIKSQL_MCP_SERVERS_PER_DB (default 1) sets how many servers one database gets
# in this process; each question is routed to one of them (`server_slot`), so every
# candidate, judge pass, reviewer call and close_session of that question meets one server.

#: The environment variable naming how many MCP server subprocesses each database gets.
SERVERS_PER_DB_ENV = "RUBIKSQL_MCP_SERVERS_PER_DB"
#: Upper bound accepted by :func:`servers_per_db`; each server holds its own workspace copy.
SERVERS_PER_DB_MAX = 64


def servers_per_db(raw: str | None) -> int:
    """The MCP server count per database in one answering process, from ``RUBIKSQL_MCP_SERVERS_PER_DB``.

    Tool bodies run in the server's threads, so one server spends at most about one core on
    its Python work (the GIL). Several servers per database spread one database's tool work
    over several cores. Unset or empty keeps one server, the behaviour before the setting
    existed. Otherwise the value, stripped of surrounding whitespace, must be a decimal
    integer from 1 to :data:`SERVERS_PER_DB_MAX`; anything else fails at import (startup)
    rather than falling back, so a mistyped launch never runs at an unintended width.

    Args:
        raw (str | None): The environment value, or ``None`` when the variable is unset.

    Returns:
        int: The configured count.

    Raises:
        ValueError: ``raw`` is nonempty and not a decimal integer in ``[1, SERVERS_PER_DB_MAX]``.
    """
    if raw is None or not raw.strip():
        return 1
    value = raw.strip()
    if not (value.isascii() and value.isdigit()) or not 1 <= int(value) <= SERVERS_PER_DB_MAX:
        raise ValueError(f"{SERVERS_PER_DB_ENV} must be an integer from 1 to {SERVERS_PER_DB_MAX}, got {raw!r}")
    return int(value)


#: Servers per database for this process, read once at import (see :func:`servers_per_db`).
_SERVERS_PER_DB = servers_per_db(os.environ.get(SERVERS_PER_DB_ENV))
#: The server slot `_get_shared_session` / `_evict_shared_session` act on in this context.
#: Entry points enter `server_scope` for their question's slot; a coroutine running on a
#: session's loop carries that session's slot, so in-loop reacquisition (a vote replacing
#: its dead session) targets the same server.
_SERVER_SLOT: contextvars.ContextVar[int] = contextvars.ContextVar("rubiksql_hb.mcp_server_slot", default=0)


def server_slots() -> range:
    """Every server slot of one database in this process: ``range(RUBIKSQL_MCP_SERVERS_PER_DB)``."""
    return range(_SERVERS_PER_DB)


def server_slot(route: str) -> int:
    """The server slot one question's work is pinned to.

    Stable across processes and runs (SHA-256, not the salted builtin ``hash``), so a retry
    or a recovery attempt of the same question reaches the same slot. With one server per
    database the slot is always 0 and nothing is hashed.

    Args:
        route (str): The question's routing text (its question with any hint appended).

    Returns:
        int: A slot in :func:`server_slots`.
    """
    if _SERVERS_PER_DB == 1:
        return 0
    return int.from_bytes(hashlib.sha256(route.encode("utf-8")).digest()[:8], "big") % _SERVERS_PER_DB


@contextmanager
def server_scope(slot: int) -> Iterator[None]:
    """Make `_get_shared_session` and `_evict_shared_session` act on ``slot`` in this context.

    Args:
        slot (int): A slot in :func:`server_slots`.

    Raises:
        ValueError: ``slot`` is outside :func:`server_slots`.
    """
    if slot not in server_slots():
        raise ValueError(f"MCP server slot {slot} is outside 0..{_SERVERS_PER_DB - 1}")
    token = _SERVER_SLOT.set(slot)
    try:
        yield
    finally:
        _SERVER_SLOT.reset(token)


def _session_key(db_id: str, slot: int) -> str:
    """The registry key of one database's server slot; slot 0 keeps the bare ``db_id``."""
    return db_id if slot == 0 else f"{db_id}#{slot}"


# Budget for one MCP server subprocess to spawn and finish initialize(). Generous: a cold
# start on a loaded machine imports heavenbase + chromadb and opens a workspace. It exists to
# bound a HANG, not to police slowness — every other wait in this module is bounded, this one
# was not. <= 0 restores the old unbounded wait.
_SESSION_START_TIMEOUT_SEC = float(os.environ.get("RUBIKSQL_SESSION_START_TIMEOUT_SEC", "180"))

#: Guards every session's in-flight run count and retirement flag (`_SharedSession.run`,
#: `retire_dead_session`); held only for those two updates.
_SESSION_STATE_LOCK = threading.Lock()


class _SessionSubmissionRejected(RuntimeError):
    """The acquired session retired before admitting this coroutine; no work ran."""


class _SharedSession:
    """Owns a background event loop + MCP session for one db_id's server slot."""

    #: Set when a coroutine run here failed because the MCP server or its transport is gone
    #: (`is_dead_session_error`); the question boundary then retires the session.
    dead = False
    #: `run` calls still waiting on this session. A retired session stops only when the last
    #: one returns, so a sibling question's in-flight work is never cut off with its loop.
    _active_runs = 0
    _retired = False
    #: The database's server slot this session serves (`server_slot`); 0 with one server.
    _slot = 0
    _task = None
    _stopping = False

    def __init__(self, db_id: str, slot: int = 0) -> None:
        self._db_id   = db_id
        self._slot    = slot
        self._session: ClientSession | None = None
        self._loop:    asyncio.AbstractEventLoop | None = None
        self._thread:  threading.Thread | None = None
        self._ready    = threading.Event()
        self._error:   Exception | None = None
        self._cm_outer = None
        self._cm_inner = None
        self._sentinel: asyncio.Future | None = None
        #: The `_start` task, so `stop` can cancel a start that has not reached its sentinel.
        self._task: asyncio.Task | None = None
        #: Set by `stop`; a start that sees it never serves (see `_start`).
        self._stopping = False

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        # Assigned before the loop runs it: a `stop` that reads None here is ahead of `_start`,
        # which then sees `_stopping` before it spawns anything.
        self._task = self._loop.create_task(self._start())
        self._loop.run_until_complete(self._task)

    async def _start(self) -> None:
        try:
            if self._stopping:
                raise asyncio.CancelledError()
            self._cm_outer = stdio_client(_mcp_params(self._db_id))
            read, write = await self._cm_outer.__aenter__()
            self._cm_inner = ClientSession(read, write)
            self._session  = await self._cm_inner.__aenter__()
            await self._session.initialize()
            if self._stopping:
                # `start` gave up (its readiness budget expired) while initialize() was still
                # running. Nothing registered this session, so serving would leave a server
                # nobody can reach or stop: tear it down instead.
                raise asyncio.CancelledError()
            self._ready.set()
            # Keep the loop alive until stop() cancels the sentinel.
            self._sentinel = self._loop.create_future()
            try:
                await self._sentinel
            except asyncio.CancelledError:
                pass
        except asyncio.CancelledError:
            # `stop` cancelled a start that had not reached its sentinel; the teardown below
            # still reaps the subprocess. Swallowed: this task is the loop's only work.
            self._error = RuntimeError(f"MCP session start for db_id={self._db_id!r} was stopped")
            self._ready.set()
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            # Best-effort teardown in the SAME task that entered the CMs (anyio cancel
            # scopes require this) so the MCP subprocess is reaped instead of leaked.
            for cm in (self._cm_inner, self._cm_outer):
                if cm is not None:
                    try:
                        await cm.__aexit__(None, None, None)
                    except Exception:
                        pass
            await _aclose_current_llm_runtime()

    def start(self, timeout: float | None = None) -> None:
        """Spawn the background loop and block until the MCP session is initialized.

        Bounded on purpose: ``_ready`` is set only after ``session.initialize()`` returns,
        and ClientSession is constructed without ``read_timeout_seconds``, so an MCP
        subprocess that starts but never answers would otherwise park this thread forever.

        Args:
            timeout (float | None): Seconds to wait for readiness; None uses
                ``_SESSION_START_TIMEOUT_SEC``. ``<= 0`` waits indefinitely (the old
                behaviour, kept as an escape hatch).

        Raises:
            RuntimeError: The session did not become ready within the budget.
            Exception: Whatever ``_start`` failed with, re-raised unchanged.
        """
        budget = _SESSION_START_TIMEOUT_SEC if timeout is None else timeout
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=budget if budget and budget > 0 else None):
            self.stop(timeout=1.0)      # reap the half-started subprocess/thread
            raise RuntimeError(
                f"MCP session for db_id={self._db_id!r} did not become ready within "
                f"{budget}s. The server subprocess started but never finished initialize() "
                f"— check that `python -m rubiksql_hb.server` runs, and raise "
                f"RUBIKSQL_SESSION_START_TIMEOUT_SEC if this machine is simply slow.")
        if self._error:
            raise self._error

    def run(self, coro, timeout: float | None = None, *, deadline: "_SharedSessionDeadline | None" = None,
            until: Callable[[], float | None] | None = None) -> object:
        """Submit a coroutine to the background loop and block until done.

        With `timeout`, a hung coroutine (e.g. an API/MCP read that never returns) is
        REALLY cancelled: run_coroutine_threadsafe's future.cancel() propagates into the
        async task, unwinding every await in the loop. Without this, .result() blocked
        forever and the caller's thread-level timeout could only abandon (not stop) it.
        ``until`` adds a moving absolute monotonic deadline that the coroutine itself may
        move (a vote's phase deadline, `phase_budget.VotePhases.deadline`): the wait re-reads
        it at least every ``_MOVING_DEADLINE_POLL_SEC`` and cancels the task the same way once
        it, or ``timeout`` when both are given, has passed.
        With ``deadline``, that owner replaces ``timeout`` and the wait includes
        cancellation cleanup; its proxy future is not cancelled while the owner joins.
        The wait ends ``_DEADLINE_RELEASE_GRACE_SEC`` after the owner's ``join_by``:
        a final cancellation then releases this thread from a task that cannot finish.

        A coroutine that fails because the MCP server or its transport is gone
        (`is_dead_session_error`) marks this session ``dead``; `retire_dead_session` then
        takes it out of the registry, and a retired session stops when its last run returns.
        Retirement rejects and closes coroutines not yet admitted. Callers may reacquire
        the session and construct a new coroutine only for that rejected submission.
        """
        with _SESSION_STATE_LOCK:
            if self._retired:
                coro.close()
                raise _SessionSubmissionRejected(f"MCP session for db_id={self._db_id!r} retired before submission")
            self._active_runs += 1
        try:
            return self._wait_on_loop(coro, timeout, deadline, until)
        except BaseException as error:
            if is_dead_session_error(error):
                self.dead = True
            raise
        finally:
            with _SESSION_STATE_LOCK:
                self._active_runs -= 1
                drained = self._retired and self._active_runs == 0
            if drained:
                self.stop()

    def _wait_on_loop(self, coro, timeout: float | None, deadline: "_SharedSessionDeadline | None",
                      until: Callable[[], float | None] | None = None) -> object:
        """Schedule ``coro`` on the session loop and wait for it as `run` documents."""
        submitted = time.monotonic()
        timing = session_timing.SessionTiming("shared_session")
        queue_span = timing.start_span("queue_wait", submitted)
        ownership = threading.Lock()
        started = False
        closed = False

        def close_unstarted(_future=None):
            nonlocal closed
            with ownership:
                if not started and not closed:
                    coro.close()
                    closed = True
                    timing.end_span(queue_span, "queue_wait", time.monotonic() - submitted)
                    timing.finish()

        async def measured():
            nonlocal started
            with ownership:
                if closed:
                    raise asyncio.CancelledError()
                started = True
            timing.end_span(queue_span, "queue_wait", time.monotonic() - submitted)
            _SERVER_SLOT.set(self._slot)    # this task's own context: reacquisition stays on this server
            with session_timing.scope("shared_session", timing=timing):
                result = None
                error = None
                try:
                    result = await deadline.wrap(coro) if deadline is not None else await coro
                    return result
                except (Exception, asyncio.CancelledError) as exc:
                    error = exc
                    raise
                finally:
                    timing.finish()
                    if isinstance(result, dict):
                        result["shared_session_timing"] = timing.snapshot()
                    if error is not None:
                        error.shared_session_timing = timing.snapshot()
                        for attribute in ("candidate_record", "vote_record"):
                            record = getattr(error, attribute, None)
                            if isinstance(record, dict):
                                record["shared_session_timing"] = timing.snapshot()

        wrapper = measured()
        try:
            fut = asyncio.run_coroutine_threadsafe(wrapper, self._loop)
        except Exception:
            wrapper.close()
            close_unstarted()
            raise
        fut.add_done_callback(close_unstarted)
        try:
            if deadline is None and until is not None:
                fixed = None if timeout is None else submitted + timeout
                return self._wait_until(fut, until if fixed is None else
                                        lambda: min(fixed, at) if (at := until()) is not None else fixed)
            # A deadline owns cancellation and drains the task before this future
            # completes. Cancelling the proxy future would acknowledge too early, so
            # this wait outlasts the owner's own join and only then releases the thread.
            return fut.result(timeout if deadline is None else
                              max(0.0, deadline.join_by - time.monotonic()) + _DEADLINE_RELEASE_GRACE_SEC)
        except _futures.TimeoutError:
            if fut.done():
                # Two TimeoutErrors share this class: the wait's own (message-less,
                # raised by Future.result) and one the coroutine raised itself (the
                # question deadline). A future that is done by now either completed
                # between the wait expiring and this check - its value must not be
                # dropped - or raised: result() returns the former and re-raises the
                # latter unchanged. A bare `raise` here could surface the wait's
                # empty TimeoutError in place of either.
                return fut.result()
            fut.cancel()
            if deadline is not None:
                error = TimeoutError("shared-session task still unwinding after its deadline owner stopped joining")
            else:
                # Keep the "cancelled after " prefix: the tail readout and replay tools
                # (scripts/readout_question_tail.py, replay_question_tail.py) match it.
                error = TimeoutError(f"shared-session coroutine cancelled after {timeout}s" if until is None else
                                     f"shared-session coroutine cancelled after "
                                     f"{time.monotonic() - submitted:.0f}s at its phase deadline")
                error.shared_session_timeout = True
            error.shared_session_timing = timing.snapshot()
            raise error

    @staticmethod
    def _wait_until(fut, until: Callable[[], float | None]) -> object:
        """Wait for ``fut`` until the moving deadline ``until()`` has passed (``None``: no deadline yet).

        Raises:
            concurrent.futures.TimeoutError: The deadline passed with the future unfinished.
        """
        while True:
            at = until()
            left = None if at is None else at - time.monotonic()
            if left is not None and left <= 0:
                raise _futures.TimeoutError()
            try:
                return fut.result(_MOVING_DEADLINE_POLL_SEC if left is None else min(left, _MOVING_DEADLINE_POLL_SEC))
            except _futures.TimeoutError:
                if fut.done():
                    return fut.result()

    def stop(self, timeout: float = 5.0) -> None:
        """Make `_start` run its teardown, then join the thread.

        A served session has a sentinel, and cancelling it ends `_start`. A start still before
        its sentinel (spawning the server or inside ``initialize()``, as when `start` timed
        out) has none: its task is cancelled instead, and `_stopping` stops a start that has
        not begun. Either way the subprocess is reaped, never left running unregistered.
        Best-effort and idempotent — safe to call on an already-dead session.
        """
        self._stopping = True
        loop, sentinel, task = self._loop, self._sentinel, self._task
        if loop is not None:
            try:
                if sentinel is not None:
                    loop.call_soon_threadsafe(lambda: not sentinel.done() and sentinel.cancel())
                elif task is not None:
                    loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass  # loop already stopped/closed
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    @property
    def session(self) -> ClientSession:
        return self._session


#: How often `_SharedSession.run` re-reads a moving deadline (``until``); the wait still ends at the
#: deadline itself whenever the deadline holds still.
_MOVING_DEADLINE_POLL_SEC = 1.0

#: How long `_SharedSession.run` keeps waiting after its deadline owner's ``join_by``, so the
#: owner's own wait always expires first and records the outcome.
_DEADLINE_RELEASE_GRACE_SEC = 5.0


class _SharedSessionDeadline:
    """Own one submission's absolute deadline, single cancellation and bounded join.

    Construct before worker/session acquisition. At ``expires`` the task is cancelled
    once, so an outer expiry cannot interrupt its cancellation cleanup a second time.
    From then on :meth:`stopped` holds: the task starts no model request, judge pacing,
    synonym embedding or tool dispatch, even after consuming that cancellation, and its
    vote hands each ``close_session`` to the loop instead of awaiting the server. The
    owner joins its worker after :meth:`cancel` until ``join_by``, acquisition included;
    ``run`` releases the worker thread ``_DEADLINE_RELEASE_GRACE_SEC`` later.

    Args:
        expires: Monotonic time at which the task is stopped.
        join_by: Monotonic time until which the owner waits for the stopped worker.
    """

    def __init__(self, expires: float, join_by: float) -> None:
        self.expires = expires
        self.join_by = max(expires, join_by)
        self._lock = threading.Lock()
        self._cancelled = False
        self._task: asyncio.Task | None = None

    def stopped(self) -> bool:
        """Return whether expiry or the owner has stopped this submission; any thread may ask."""
        return self._cancelled or time.monotonic() >= self.expires

    def cancel(self) -> None:
        """Request cancellation on the owning loop, or prevent a delayed start."""
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
            if self._task is not None:
                self._task.get_loop().call_soon_threadsafe(self._task.cancel)

    async def wrap(self, coro):
        """Run within the remaining budget and finish unwinding before reporting expiry."""
        loop = asyncio.get_running_loop()
        with self._lock:
            if self.stopped():
                coro.close()
                raise TimeoutError("shared-session deadline expired before task start")
            self._task = asyncio.current_task()
        timer = loop.call_later(max(0.0, self.expires - time.monotonic()), self.cancel)
        try:
            with bind_stop_check(self.stopped):
                return await coro
        except asyncio.CancelledError as error:
            raise TimeoutError("shared-session deadline expired; recovery stopped") from error
        finally:
            timer.cancel()
            with self._lock:
                self._task = None


#: Session cleanup handed to the loop by stopped tasks, referenced until it finishes.
_DETACHED_CLOSES: set[asyncio.Task] = set()


def _detach_close_session(session: ClientSession, session_id: str) -> bool:
    """Hand a stopped task's ``close_session`` to the loop instead of awaiting it.

    The server runs ``close_session`` only after that session's in-flight tool, and an
    exploratory ``run_sql`` or embedding-backed retrieval can outlast any caller wait.
    A task its deadline stopped must not wait for it. The detached call still deletes
    the session file once the tool finishes; ``sweep_sessions`` reclaims the files of a
    server that exits first. This cleanup is the one request a stopped task still sends.

    Args:
        session: The MCP session that owns ``session_id``.
        session_id: The stopped run's server-side session.

    Returns:
        bool: Whether the close was detached; False leaves the caller's awaited close.
    """
    if not task_stopped():
        return False

    async def close() -> None:
        try:
            await session.call_tool("close_session", {"session_id": session_id})
        except Exception:
            pass

    task = asyncio.get_running_loop().create_task(close())
    _DETACHED_CLOSES.add(task)
    task.add_done_callback(_DETACHED_CLOSES.discard)
    return True


_shared_sessions: dict[str, _SharedSession] = {}
# Guards the two dicts ONLY — never held across a session startup. The previous shape did
# hold it across `s.start()`, so one db whose MCP subprocess hung at initialize() parked
# every other db's threads on this same lock: not one wrong answer, the whole run frozen
# until each question burned its outer per-question timeout.
_shared_sessions_lock = threading.Lock()
# Per-db_id startup locks, so concurrent first-requests for the SAME db still build exactly
# one session per slot while DIFFERENT dbs start independently. The lock is per database,
# not per slot: one database's servers open the same workspace directory, so they open it
# one at a time.
_shared_session_starts: dict[str, threading.Lock] = {}


def _get_shared_session(db_id: str) -> _SharedSession:
    """Return this db's shared MCP session for the current server slot, building it on first use.

    The slot is the one `server_scope` set in this context (0 outside any scope, the only
    slot with one server per database).

    Args:
        db_id (str): Database whose MCP server subprocess backs the session.

    Returns:
        _SharedSession: The live session; the same object for every caller of this db_id
            and slot.

    Raises:
        RuntimeError: Startup exceeded ``_SESSION_START_TIMEOUT_SEC``.
        Exception: Whatever the session's own startup failed with.
    """
    slot = _SERVER_SLOT.get()
    key = _session_key(db_id, slot)
    with _shared_sessions_lock:
        existing = _shared_sessions.get(key)
        if existing is not None:
            return existing
        start_lock = _shared_session_starts.setdefault(db_id, threading.Lock())

    with start_lock:                     # serializes THIS db only
        with _shared_sessions_lock:
            existing = _shared_sessions.get(key)   # another thread may have won the race
            if existing is not None:
                return existing
        # Built OUTSIDE both locks. A failure propagates without ever being published, so a
        # half-started session can never be handed to a later caller.
        s = _SharedSession(db_id, slot)
        s.start()
        with _shared_sessions_lock:
            _shared_sessions[key] = s
        return s


def _evict_shared_session(db_id: str) -> None:
    """Remove the current slot's session, whatever it is, and tear it down.

    Only for an owner that knows nothing else uses the slot (the preflight's teardown). Never
    call it from a coroutine on a session's own loop: stopping that session stops the loop
    under every task it runs. A question that met a dead session uses `retire_dead_session`,
    which removes only sessions marked dead.
    """
    with _shared_sessions_lock:
        s = _shared_sessions.pop(_session_key(db_id, _SERVER_SLOT.get()), None)
    if s is not None:
        try:
            s.stop()
        except Exception:
            pass


def retire_dead_session(db_id: str) -> bool:
    """Replace this database's shared MCP sessions at the next request where a run found them dead.

    Only registered sessions that `_SharedSession.run` marked ``dead`` are removed, so a
    thread whose question failed on an older session can never evict the fresh replacement
    another question already started. Every server slot of the database is checked: each
    slot's ``dead`` flag is its own, so a live slot is never touched. A retired session stops
    when its last in-flight run returns: sibling questions still waiting on it fail fast
    against the closed connection instead of being stranded on a stopped event loop.

    Args:
        db_id (str): Database whose shared session a question attempt just failed on.

    Returns:
        bool: True when at least one registered session was dead and has been retired; the
            next `_get_shared_session` call for its slot starts a fresh server.
    """
    retired = []
    with _shared_sessions_lock:
        for slot in server_slots():
            key = _session_key(db_id, slot)
            session = _shared_sessions.get(key)
            if session is not None and session.dead:
                del _shared_sessions[key]
                retired.append(session)
    for session in retired:
        with _SESSION_STATE_LOCK:
            session._retired = True
            drained = session._active_runs == 0
        if drained:
            session.stop()
    return bool(retired)


@atexit.register
def _stop_all_shared_sessions() -> None:
    """Tear down every live MCP session at process exit (clean subprocess shutdown)."""
    with _shared_sessions_lock:
        sessions = list(_shared_sessions.values())
        _shared_sessions.clear()
    for s in sessions:
        try:
            s.stop()
        except Exception:
            pass
    _close_current_llm_runtime()


# Exceptions that mean "the shared MCP session/subprocess died" → evict + retry once.
def _dead_session_excs() -> tuple:
    excs: list = [ConnectionError, EOFError, BrokenPipeError]
    try:
        import anyio
        excs += [anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream]
    except Exception:
        pass
    return tuple(excs)


_DEAD_SESSION_EXCS = _dead_session_excs()
_DEAD_SESSION_EXC_NAMES = frozenset(e.__name__ for e in _DEAD_SESSION_EXCS)


class _DeadSessionCandidatesError(RuntimeError):
    """A live candidate pool failed after losing its shared MCP session."""


def is_dead_session_error(error: BaseException) -> bool:
    """Whether an exception means the shared MCP session's server or transport is gone.

    The transport exceptions in ``_DEAD_SESSION_EXCS`` qualify, and so does the MCP SDK's own
    report: mcp 2 answers every request after the server process exits (or its pipes close)
    with ``MCPError`` code ``CONNECTION_CLOSED`` instead of a transport exception. Any other
    ``MCPError`` is an ordinary request failure on a live session. Voting preserves this
    classification in ``_DeadSessionCandidatesError`` when gathering turned the original
    exceptions into candidate records.

    Args:
        error (BaseException): An exception raised by work that used a shared session.

    Returns:
        bool: True when the session cannot serve another request.
    """
    return (isinstance(error, (*_DEAD_SESSION_EXCS, _DeadSessionCandidatesError))
            or (isinstance(error, MCPError) and error.code == CONNECTION_CLOSED))


# Two-stage example retrieval (issue #8): promoted to DEFAULT ON (must stay in lockstep
# with server.py — both sides shape the tool set / system prompt). =0 reverts to the
# legacy single-stage flow. Evidence: PLAN §5.14.
# query_knowledge is UNCONDITIONAL (promoted 07-26). Every other knowledge channel the candidate
# has is top-k-by-similarity — get_examples, get_rules (top-k, k rules out), search_values (top-10
# over 0.45) — so without it nothing is enumerable or filterable and the synonym table is
# invisible. It sits in the allowlist and is introduced in the Phase 2 tool list; it must stay in
# lockstep with server.py (which registers it) and vote.py (judge side, also unconditional).
#
# The mandatory expand_synonyms step was REMOVED from Phase 1 (07-26). It was step 1 of a Phase 1
# headed "always required, in this order", but synonym_entity holds ZERO rows in all 11 hidden
# workspaces and nothing populated it: save_synonym is its only writer and it appeared in no
# model's allowlist and in no pipeline stage. resolve_synonyms is a plain str.replace loop over
# those rows, so with none it returned the question verbatim — a full run paid 1534 x 5 = ~7,670
# tool round-trips, and the first turn of every candidate's Phase 1, for an identity function.
# The MANDATORY STEP is what went: resolve_synonyms, the save_synonym MCP tool and SynonymEntity
# all stay, because an empty table is a content gap, not a missing capability — the pipeline's
# synonym-mining stage fills it. The judge side needed no change: _JUDGE_TOOLS never contained
# expand_synonyms.
# §12 doc-literalism guard (arm-level switch, no runtime trigger): documentation
# is for choosing tables/columns, never for changing SQL shape. Targets the four
# forensically-confirmed failure modes of the desc condition.

_GRAIN_AUDIT = os.environ.get("RUBIKSQL_GRAIN_AUDIT", "") == "1"
_GRAIN_GATE_VERSION = "v7"
_GRAIN_STYLE_BLOCK = _PROMPTS.surface("candidate", "route.grain")
# L1 arm: guard against unproductive debug loops (slow questions burn 90-114 tool calls
# re-running near-identical SQL; measured on hb2_subset4). Soft guidance only — long
# PRODUCTIVE chains exist (Q00640: 114 calls, correct), so no hard turn cap.
# L1 convergence guard — ON by default (RUBIKSQL_LOOP_GUARD=0 disables). Fires the nudge below
# at the tool-call tiers and flags identical re-runs. Catches the "explore forever, never submit"
# failure EARLIER than the wall-clock time nudge (call-count vs elapsed time); the two compose.
_LOOP_GUARD = _PROMPTS.knob("loop_guard")
_LOOP_GUARD_TIERS = _PROMPTS.knob("loop_guard_tiers")
# Empty-result probe — ON by default (RUBIKSQL_EMPTY_PROBE=0 disables). The "hunt for data that
# does not exist" loop starts the moment a well-formed query returns 0 rows and the model reads
# that as "my SQL is wrong". Fires once, on the Nth empty run_sql result.
#
# The note asks for CONJUNCT-level diagnosis, not per-literal checks. Measured on Q00944
# ("2008 Australian Grand Prix", which the DB does not have): every literal passes on its own —
# 'Australian Grand Prix' exists (1985-1995) and 2008 exists — and ONLY the combination is
# missing. A per-literal check therefore "confirms" the filters are fine and pushes the model
# back into debugging its own (correct) SQL; that question burned the full 30-min wall through
# empty-probe + loop-guard + time-nudge and still never submitted. Locating WHICH conjunct
# empties the result is the diagnosis that actually resolves the ambiguity.
_EMPTY_PROBE = _PROMPTS.knob("empty_probe")
_EMPTY_PROBE_AT = _PROMPTS.knob("empty_probe_at")
# fire on the 2nd empty result (the 1st can be ordinary exploration)
# The note itself is the candidate.note.empty surface (slot n); the L1 progress check is
# candidate.nudge.converge (slot n). Both render where they fire, below.
from .tool_bundle import process_bundle as _process_tool_bundle
from .tool_definitions import select_definitions as _select_tool_definitions

# Detached views for host route declarations; the pinned native rows own selection.
_NL2SQL_TOOLS_DEFAULT = tool_contract._selection("candidate", "default", None, None)
_NL2SQL_TOOLS_REQUIRED = frozenset(record["name"] for record in _select_tool_definitions(_process_tool_bundle().definitions, "candidate")
                                  if "candidate" in record.get("required_for", ()))
_GRAIN_AUDIT_TOOLS = frozenset(record["name"] for record in _select_tool_definitions(
    _process_tool_bundle().definitions, "candidate", "grain")) - _NL2SQL_TOOLS_DEFAULT

def _candidate_route(style: str | None):
    """Resolve one candidate style to its registered Route.

    ``style=None`` is the default route's zero-overhead fast path: no registry
    lookup, no hooks, byte-identical legacy behavior. Any other style must be
    registered in ``routes.REGISTRY`` with its feature bit satisfied.

    Args:
        style: Candidate style name (e.g. "grain") or None.

    Returns:
        Route | None: The registered route, or None for the default path.

    Raises:
        ValueError: ``style`` names no registered route.
        RuntimeError: The route's feature bit is missing (e.g. grain without
            RUBIKSQL_GRAIN_AUDIT=1).
    """
    if style is None:
        return None
    from . import routes  # deferred: routes imports this module's frozen pieces
    route = routes.get(style)
    route.require()
    return route


def _nl2sql_tools(
    disabled: set[str] | None = None,
    style: str | None = None,
) -> frozenset[str]:
    """Return the tools enabled for one candidate route."""
    return tool_contract._selection("candidate", style or "default", disabled, None)


def _route_style_block(style: str | None) -> str:
    """Return the prompt suffix for one candidate route."""
    route = _candidate_route(style)
    return "" if route is None else route.candidate_block


_GRAIN_AGGREGATE_RE = re.compile(r"\b(?:COUNT|SUM|AVG)\s*\(", re.IGNORECASE)
_GRAIN_PERCENT_RE = re.compile(r"\bpercent(?:age)?\b", re.IGNORECASE)
_GRAIN_COUNT_DISTINCT_RE = re.compile(r"\bCOUNT\s*\(\s*DISTINCT\b", re.IGNORECASE)
_GRAIN_EXISTS_RE = re.compile(r"\bEXISTS\s*\(", re.IGNORECASE)
_GRAIN_PERCENT_SCALE_RE = re.compile(r"(?:\*\s*100(?:\.0)?\b|\b100(?:\.0)?\s*\*)", re.IGNORECASE)
_GRAIN_PERCENT_OUTPUT_RE = re.compile(
    r"\b(?:what(?:\s+(?:is|was|are|were))?|calculate|compute|find|determine|report)\b"
    r"[^?]{0,140}\bpercent(?:age)?\b|\bhow\s+many\s+percent\b|"
    r"\bdeviation\s+in\s+percentage\b",
    re.IGNORECASE,
)
_GRAIN_ROW_MEASURE_INTENT_RE = re.compile(
    r"\b(?:average|race|lap|transaction|event|record|measurement|reading|"
    r"consumption|consumed|income|revenue|sales|amount\s+received)\b",
    re.IGNORECASE,
)
_GRAIN_MAX_CORRECTIONS_PER_REASON = 3
_GRAIN_ENTITY_ALIAS_RE = re.compile(
    r"(?i)([A-Za-z_][\w$]*(?:\s*\.\s*[A-Za-z_][\w$]*)?)\s+AS\s+"
    r"(?:[`\"\[])?answer_entity_id(?:[`\"\]])?"
)
_GRAIN_TABLE_ALIAS_RE = re.compile(
    r"(?i)\b(?:FROM|JOIN)\s+(?:`([^`]+)`|\"([^\"]+)\"|\[([^\]]+)\]|([A-Za-z_][\w$]*))"
    r"(?:\s+(?:AS\s+)?([A-Za-z_][\w$]*))?"
)
_SQL_RESERVED = frozenset({
    "on", "where", "left", "right", "inner", "outer", "full", "cross", "join",
    "group", "order", "limit", "having", "union", "except", "intersect",
})


def _grain_submit_error(
    sql: str,
    audit: dict | None,
    question: str = "",
) -> tuple[str, dict] | None:
    """Require diagnostic evidence without dictating row- versus entity-grain semantics."""
    if not (_GRAIN_AGGREGATE_RE.search(sql or "") or _GRAIN_PERCENT_RE.search(question or "")):
        return None
    if not audit:
        return "missing_audit", {"error": _PROMPTS.surface("shared", "grain.missing_audit")}
    if (_GRAIN_PERCENT_OUTPUT_RE.search(question or "")
            and not _GRAIN_PERCENT_SCALE_RE.search(sql or "")):
        return "percentage_scale", {"error": _PROMPTS.surface("shared", "grain.percentage_scale")}
    if _grain_requires_entity_stage(question, audit):
        upper = (sql or "").upper()
        nested_entity_stage = upper.count("SELECT") >= 2 and (
            "GROUP BY" in upper or re.search(r"\bSELECT\s+DISTINCT\b", upper)
        )
        has_entity_stage = bool(
            _GRAIN_COUNT_DISTINCT_RE.search(sql or "")
            or _GRAIN_EXISTS_RE.search(sql or "")
            or nested_entity_stage
        )
        if not has_entity_stage:
            return "entity_intent_fanout", {"error": _PROMPTS.surface("shared", "grain.entity_fanout")}
    return None


def _grain_gate_correction(
    counters: dict[str, int],
    reason: str,
    correction: dict,
) -> dict:
    """Allow bounded iterative repairs for one semantic gate reason."""
    attempt = counters.get(reason, 0) + 1
    counters[reason] = attempt
    if attempt > _GRAIN_MAX_CORRECTIONS_PER_REASON:
        raise RuntimeError(correction["error"])
    return {**correction, "grain_gate_attempt": attempt}


def _grain_requires_entity_stage(question: str, audit: dict) -> bool:
    """Return whether wording makes entity-level counting high confidence."""
    if audit.get("audit_status") != "fanout_detected":
        return False
    q = question or ""
    if _GRAIN_ROW_MEASURE_INTENT_RE.search(q):
        return False
    entity = str(audit.get("declared_answer_entity", ""))
    if not entity:
        return False
    forms = sorted(_grain_word_forms(entity), key=len, reverse=True)
    direct = any(re.search(
        rf"\b(?:percentage|percent|number|count)\s+of\b[^?;,.!]{{0,90}}\b{re.escape(form)}\b",
        q, re.IGNORECASE,
    ) for form in forms)
    how_many = any(re.search(
        rf"\bhow\s+many\b[^?;,.!]{{0,60}}\b{re.escape(form)}\b",
        q, re.IGNORECASE,
    ) for form in forms)
    entity_then_prevalence = any(re.search(
        rf"\b{re.escape(form)}\b[^?]{{0,140}}\bpercent(?:age)?\b[^?]{{0,80}}\bwho\s+has\b",
        q, re.IGNORECASE,
    ) for form in forms)
    return direct or how_many or entity_then_prevalence


def _grain_word_forms(table_name: str) -> set[str]:
    """Return conservative singular/plural forms for a schema table name."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", table_name).replace("_", " ").lower()
    words = re.findall(r"[a-z0-9]+", spaced)
    if not words:
        return set()
    last = words[-1]
    irregular_singular = {
        "people": "person", "men": "man", "women": "woman", "children": "child",
        "heroes": "hero", "analyses": "analysis", "indices": "index",
    }
    singular = irregular_singular.get(
        last,
        last[:-3] + "y" if last.endswith("ies") else last[:-2] if last.endswith("oes")
        else last[:-1] if last.endswith("s") and not last.endswith("ss") else last,
    )
    if singular.endswith("y") and len(singular) > 1 and singular[-2] not in "aeiou":
        plural = singular[:-1] + "ies"
    elif singular.endswith(("s", "x", "z", "ch", "sh", "o")):
        plural = singular + "es"
    else:
        plural = singular + "s"
    prefix = words[:-1]
    forms = {spaced, " ".join([*prefix, singular]), " ".join([*prefix, plural])}
    if len(words) == 1:
        forms.update({last, singular, plural})
    return {form for form in forms if form}


_GRAIN_MENTION_FUNCTION_WORDS = frozenset({
    "of", "in", "on", "at", "by", "with", "from", "for", "to", "per", "as", "and",
    "or", "that", "which", "who", "whose", "whom", "where", "when", "whether",
    "is", "are", "was", "were", "has", "have", "had", "do", "does", "did", "than",
    "the", "a", "an", "each", "every", "all", "any", "there", "not", "no", "but",
})


def _grain_expected_entity(question: str, schema: dict | None) -> dict | None:
    """Infer a high-confidence answer-entity table from question wording.

    A table name acting as a compound-noun MODIFIER is not a mention of that
    entity: "triple-bond" is about bonds' type, not the bond population, and
    "account holders" asks about people. Such hits drove 14% of the old gate's
    rejections (Q00286, Q00128), so a match is dropped when it is glued to a
    hyphen or immediately followed by a content word it modifies.
    """
    if not isinstance(schema, dict) or not isinstance(schema.get("tables"), list):
        return None
    q = (question or "").lower()
    focus_starts = [m.end() for m in re.finditer(
        r"\b(?:percentage|percent|number|count|average)\s+of(?:\s+the)?\s+|\bhow\s+many\s+",
        q,
    )]

    def _is_modifier(start: int, end: int) -> bool:
        if start > 0 and q[start - 1] == "-":
            return True
        follower = re.match(r"[ ]+([a-z]+)(?![\w])", q[end:])
        return bool(follower) and follower.group(1) not in _GRAIN_MENTION_FUNCTION_WORDS

    ranked = []
    mentioned = []
    for table in schema["tables"]:
        name = str(table.get("table", ""))
        positions = []
        for form in _grain_word_forms(name):
            positions.extend(
                m.start() for m in re.finditer(rf"\b{re.escape(form)}\b", q)
                if not _is_modifier(m.start(), m.end())
            )
        if not positions:
            continue
        mentioned.append(table)
        focus_distances = [p - start for start in focus_starts for p in positions if 0 <= p - start <= 100]
        score = 1000 - min(focus_distances) if focus_distances else 1
        ranked.append((score, table))
    focused = [item for item in ranked if item[0] > 1]
    if focused:
        focused.sort(key=lambda item: item[0], reverse=True)
        if len(focused) == 1 or focused[0][0] > focused[1][0]:
            return {**focused[0][1], "_schema_tables": schema["tables"]}
    return {**mentioned[0], "_schema_tables": schema["tables"]} if len(mentioned) == 1 else None


def _grain_entity_error(
    source_sql: str, expected: dict | None, entity_key: str = "",
) -> str | None:
    """Advise when the declared answer key does not look like the answer table's key.

    Formerly a hard gate, retired by the executed triage of all 293 rejections it
    issued across the 08-06/08-07 runs: wrong 163 times against right 46. Two of its
    three teeth actively lost points — demanding the literal ``AS answer_entity_id``
    spelling (66% of rejections; 96 blocked a column that WAS the gold answer entity)
    and enforcing its own table inference, which is wrong on 26% of decidable
    questions and, because the same text hits all five candidates identically,
    collapsed whole pools (Q00298/Q01454 reproduce gold exactly at the grain it
    blocked). The inference is now an ADVISORY NOTE the hooks attach to the audit
    result: where it was right it still warns; where it is wrong it no longer
    forbids (user verdict 08-07). The key to judge comes from the explicit
    ``AS answer_entity_id`` alias when present, else from the caller's declared
    ``entity_key`` — never from a spelling requirement.
    """
    if not expected:
        return None
    source = source_sql or ""
    table_name = str(expected.get("table", ""))
    match = _GRAIN_ENTITY_ALIAS_RE.search(source)
    if match:
        expression = re.sub(r"\s+", "", match.group(1))
    else:
        key = (entity_key or "").strip().strip('`"[]')
        if not key:
            return None
        declared = re.search(
            rf"(?i)([A-Za-z_][\w$]*(?:\s*\.\s*[A-Za-z_][\w$]*)?)\s+AS\s+"
            rf"(?:[`\"\[])?{re.escape(key)}(?:[`\"\]])?",
            source,
        )
        expression = re.sub(r"\s+", "", declared.group(1)) if declared else key
    qualifier, _, column = expression.rpartition(".")
    aliases: dict[str, str] = {}
    for table_match in _GRAIN_TABLE_ALIAS_RE.finditer(source):
        raw_table = next((value for value in table_match.groups()[:4] if value), "")
        alias = table_match.group(5) or raw_table
        if alias.lower() in _SQL_RESERVED:
            alias = raw_table
        aliases[alias.lower()] = raw_table.lower()
        aliases[raw_table.lower()] = raw_table.lower()
    schema_tables = expected.get("_schema_tables", [])
    table_lookup = {
        str(table.get("table", "")).casefold(): table for table in schema_tables
    }
    expected_name = table_name.casefold()
    expected_keys = {str(value).casefold() for value in expected.get("pks", [])}
    for table in schema_tables:
        for item in table.get("columns", []):
            ref = str(item.get("fk_ref", "") or "")
            ref_table, dot, ref_col = ref.rpartition(".")
            if dot and ref_table.casefold() == expected_name:
                expected_keys.add(ref_col.casefold())

    def _is_entity_key(actual_table: str, actual_column: str) -> bool:
        actual_name = actual_table.casefold()
        actual_col = actual_column.casefold()
        if actual_name == expected_name:
            return actual_col in expected_keys
        table = table_lookup.get(actual_name, {})
        item = next((
            value for value in table.get("columns", [])
            if str(value.get("name", "")).casefold() == actual_col
        ), None)
        if not item:
            return False
        ref_table, dot, ref_col = str(item.get("fk_ref", "") or "").rpartition(".")
        return bool(
            dot and ref_table.casefold() == expected_name and ref_col.casefold() in expected_keys
        )

    if qualifier:
        if qualifier.casefold() not in aliases:
            # A derived-table alias — `(SELECT ...) AS b` — that the FROM/JOIN regex
            # cannot bind. The gate cannot reason about what `b` holds, so it says
            # nothing rather than guessing (it used to hard-reject here, Q00615).
            if re.search(rf"(?i)\)\s*(?:AS\s+)?{re.escape(qualifier)}\b", source):
                return None
        actual_table = aliases.get(qualifier.casefold(), qualifier)
        if _is_entity_key(actual_table, column):
            return None
    else:
        source_tables = set(aliases.values())
        eligible = [name for name in source_tables if _is_entity_key(name, column)]
        # Exactly one source table's column is a valid key of the expected entity —
        # that reading wins. The old extra demand that no OTHER table anywhere in the
        # statement merely CONTAIN the name vetoed valid keys over subquery tables
        # the regex swept up (Q00639).
        if len(eligible) == 1:
            return None
    pks = ", ".join(str(value) for value in expected.get("pks", [])) or "its stable key"
    return _PROMPTS.surface(
        "shared", "grain.entity_note",
        table_name=table_name, expression=str(expression), pks=pks,
    )


# Shared word-shape classifiers, reused by router.py's grain text-exclusion rules.
_COHORT_EXCLUDED_RE = re.compile(
    r"\b(?:change|deviation|growth|increase[sd]?|decrease[sd]?|grew|grown|average)\b",
    re.IGNORECASE,
)
_COHORT_DISJOINT_RATIO_RE = re.compile(
    r"\bratios?\s+(?:of\b[^.?;]{0,80}?\b(?:to|against|versus|vs\.?)\b|between\b)",
    re.IGNORECASE,
)
_COHORT_ENTITY_COLLAPSE_RE = re.compile(
    r"\bdistinct\b|\bunique\b|\bever\b|\bat\s+any\s+time\b|\bat\s+least\s+once\b|"
    r"\blatest\b|\bmost\s+recent\b",
    re.IGNORECASE,
)


# ── Provider configs ──────────────────────────────────────────────────────────

_PROVIDERS = {
    "claude": {
        "default_model": "claude-sonnet-4-6",
    },
    "qwen": {
        "default_model": "qwen3.8-max",
    },
    "deepseek": {
        "default_model": DEEPSEEK_CANDIDATE_MODEL,
    },
}

# Count-based loop knobs. They are knobs of the prompt release (config.knobs, the table in
# prompts/entries.py): a release value other than the default moves the candidate keyspace,
# and an explicitly set RUBIKSQL_* variable still overrides it for an experiment, entering the
# _cache_key namespace (see there) as it always did. Defaults are the submitted values, so a
# release that carries them is byte-identical. _RUN_SQL_MAX_ROWS and
# _EMPTY_PROBE_AT/_LOOP_GUARD_TIERS change MODEL-VISIBLE content — without the key terms,
# changing one would replay a baseline cache produced under the old value.
_RUN_SQL_MAX_ROWS  = _PROMPTS.knob("run_sql_max_rows")    # cap run_sql rows
_MAX_TURNS         = _PROMPTS.knob("max_turns")           # runaway guard
_MAX_SUBMIT_NUDGES = _PROMPTS.knob("max_submit_nudges")   # re-prompt count
# The candidate system prompt: the three-phase workflow, its tool notes, Checks A and B, the
# hint / knowledge-priority / SQL-precision doctrine, intermediates, multi-part questions, the
# shared SQL-result note and the result-cap note — the system message of the candidate's
# document (readable in prompts/store/baseline_text/); the one runtime slot is the row cap,
# so RUBIKSQL_RUN_SQL_MAX_ROWS keeps moving the bytes and the cache key.
_SYSTEM = _PROMPTS.surface("candidate", "system", max_rows=_RUN_SQL_MAX_ROWS)


def _candidate_system_content(style: str | None, style_block: str, slot: int | None = None) -> str:
    """Assemble one candidate run's system prompt inside the run's bound recorder.

    The bytes are the protocol the cache key and the fingerprint line read, ``_SYSTEM +
    style_block``; both pieces are rendered again here, through the release, so the run's
    ``prompt_identity.fired`` names the rows the model saw (``Release.surface`` is memoised,
    so the same objects come back and no byte moves). Without this the gate's cost estimate
    and the exposure views read a system-prompt change as reaching no unit. A route the
    release does not provide (an ad-hoc test route) keeps the registry's block and records
    nothing for it. A slot whose own document the release carries (``candidate.s<k>``) renders
    that document's system message in place of ``_SYSTEM``.

    Args:
        style (str | None): The candidate route, ``None`` for the default route.
        style_block (str): ``route.candidate_block`` of that route, ``""`` for the default.
        slot (int | None): The candidate's position in the pool, ``None`` outside a pool.

    Returns:
        str: The system prompt exactly as ``_SYSTEM + style_block`` (the slot's own system
            message in place of ``_SYSTEM`` when it has one).
    """
    system = _PROMPTS.slot_surface(slot, "system", max_rows=_RUN_SQL_MAX_ROWS)
    if style and _PROMPTS.has_surface("candidate", f"route.{style}"):
        style_block = _PROMPTS.surface("candidate", f"route.{style}")
    return system + style_block


async def _candidate_user_content(question: str, db_id: str, rule_triggers: "_RuleTriggerState",
                                  slot: int | None = None) -> str:
    """Assemble one candidate run's first user message inside the run's bound recorder.

    The message is the ``candidate.user`` surface: the question, its synonym notes APPENDED —
    never substituted into it; see _synonym_context for why rewriting it measured -1.00pp — and
    the rule hooks the question itself fires. The two frames render first, through their own
    surfaces and in that order (the hook match records what it served), then the surface places
    them: the second message of the candidate's document, or under r0 the code's fallback
    ``{question}{synonyms}{rule_hooks}``, which is the concatenation 9140477 wrote here.

    Args:
        question (str): The question as the candidate receives it (hint appended).
        db_id (str): The database whose synonym table is read.
        rule_triggers (_RuleTriggerState): The run's hook state; the question's matches are recorded on it.
        slot (int | None): The candidate's position in the pool; its own document places the frames when it has one.

    Returns:
        str: The first user message exactly as sent.
    """
    synonyms = await _async_synonym_context(db_id, question)
    rule_hooks = await rule_triggers.match(question) + hint_conflict_frame(question)
    return _PROMPTS.slot_surface(slot, "user", question=question, synonyms=synonyms, rule_hooks=rule_hooks)


def hint_conflict_frame(question: str) -> str:
    """Render the ``shared.hint_conflict`` frame for each hint literal that contradicts the question.

    The frame follows the rule hooks in the candidate's first message and in the judge's and reviewer's user
    message. A release without the row, or a question where :func:`hint_conflicts.detect` finds nothing, renders
    the empty string, so those messages stay byte-identical.

    Args:
        question (str): The canonical question, hint appended.

    Returns:
        str: The rendered frames, or ``""``.
    """
    if not _PROMPTS.has("shared.hint_conflict"):
        return ""
    return "".join(_PROMPTS.surface("shared", "hint_conflict", kind=conflict.kind, hint_literal=conflict.hint_literal,
                                    question_literal=conflict.question_literal)
                   for conflict in hint_conflicts.detect(question))


async def _result_shape_note(session: ClientSession, db_id: str, sql: str, result: object, *,
                             probe_allowed: bool = True) -> tuple[str, dict | None]:
    """Check the bounced final SQL's executed rows for a repeat inside its LIMIT window.

    Runs only when the release carries ``candidate.note.result_shape``. A single-table query spends one
    cardinality probe through ``run_sql_base`` (:mod:`limit_repeats`); a join needs none. The probe is skipped,
    and the note stays silent, when ``probe_allowed`` is false (little time left before the time nudge).

    Args:
        session (ClientSession): The candidate's MCP session, for the probe.
        db_id (str): The database the SQL ran on.
        sql (str): The bounced final SQL.
        result (object): Its executed result (``columns``, ``rows``).
        probe_allowed (bool): Whether the run still has time for the probe.

    Returns:
        tuple[str, dict | None]: The rendered note (``""`` when silent) and the protocol record (``None`` when the
            shape never qualified). The record says whether the probe ran, its counts, and its error or timeout.
    """
    if not _PROMPTS.has("candidate.note.result_shape") or not isinstance(result, dict):
        return "", None
    repeat = limit_repeats.find(sql, result.get("columns") or [], result.get("rows") or [])
    if repeat is None:
        return "", None
    probe = None
    record = {"version": limit_repeats.IMPLEMENTATION_VERSION, "limit": repeat.limit,
              "rows": [repeat.first, repeat.second], "column": repeat.column, "join": repeat.join,
              "probe_ran": False, "probe": None, "probe_error": None, "probe_timeout": False}
    if not repeat.join and repeat.probe is not None:
        if probe_allowed:
            probe = await _raw_call(session, "run_sql_base", {"sql": repeat.probe}, db_id)
            record["probe_ran"] = True
            if isinstance(probe, dict) and probe.get("error"):
                record["probe_error"] = str(probe["error"])[:300]
                record["probe_timeout"] = bool(sql_tools.is_timeout_error(probe))
            elif isinstance(probe, dict):
                record["probe"] = probe.get("rows")
        else:
            record["probe_skipped"] = "time"
    fired = limit_repeats.fires(repeat, probe)
    record["fired"] = fired
    if not fired:
        return "", record
    return _PROMPTS.surface("candidate", "note.result_shape", limit=repeat.limit, first=repeat.first,
                            second=repeat.second), record


#: The result-shape probe needs this much time left before the time nudge (seconds).
_SHAPE_PROBE_MARGIN_SEC = 120


# Time-budget nudge — once a candidate has burned this many wall-clock seconds (default 20 min,
# under the 30-min per-question wall), inject ONE user message telling the model to stop exploring
# and submit its best SQL now. Fires at most once per candidate; RUBIKSQL_TIME_NUDGE_SEC=0 disables.
_TIME_NUDGE_SEC = _PROMPTS.knob("time_nudge_sec")
# The nudge text is the candidate.nudge.time surface (slot mins), rendered where it fires.
# RUBIKSQL_LLM_TIMEOUT_SEC is resolved by llm_runtime and supplied to HeavenBase. Raise it
# for monster questions whose single generation legitimately needs longer (candidates run
# concurrently, so this stays inside the 30-min wall).
#   measured 2026-07-25 (qwen3.7-plus, real replayed prompts): median turn 9.1s, slowest real
#   turn 77.5s, throughput 35.8-54.6 tok/s. A worst-case LEGITIMATE generation of the full
#   max_tokens=4096 at the slowest observed throughput needs 4096/35.8 = 114s — the old 120s
#   default left only 5% headroom, so a long-but-valid completion could be killed (the Q00153
#   "all 5 candidates timed out" signature). 180s = 58% headroom over that worst case, while a
#   hung request still costs only 10% of the 1800s per-question budget (600s would cost a third).
# Hard in-loop deadline. The time nudge only ASKS the model to finalize and it can ignore that;
# when it does, the candidate cap outside cuts the seat with NOTHING — the ckpt3 census lost
# 3 questions x 3 retries that way under the old 30-min wall. Past this deadline we leave the
# loop and let _maybe_salvage commit the last run_sql_base-verified SQL, turning a total loss
# into a best effort. 0 disables. The default sits phase_budget.SEAT_SALVAGE_MARGIN_SEC (5 min)
# under the candidate budget: 1700 s of the seat's own clock for the 2000 s budget
# (`phase_budget.seat_hard_deadline`); RUBIKSQL_HARD_DEADLINE_SEC sets it explicitly. A seat also
# leaves its loop at the vote's own salvage point before the cap (`_seat_out_of_time`), which
# holds for a late top-up seat and for a candidate budget a short ceiling scaled down.
_HARD_DEADLINE_SEC = (int(os.environ["RUBIKSQL_HARD_DEADLINE_SEC"] or "0")
                      if "RUBIKSQL_HARD_DEADLINE_SEC" in os.environ else phase_budget.seat_hard_deadline())


def _seat_out_of_time(clock: RunClock) -> bool:
    """Whether a candidate seat must leave its loop now and salvage its last verified SQL.

    Past its own hard deadline (``_HARD_DEADLINE_SEC`` of its run clock), or past the bound vote's
    salvage point (`phase_budget.VotePhases.seat_salvage_at`, a margin before the candidate cap).
    ``RUBIKSQL_HARD_DEADLINE_SEC=0`` disables both.

    Args:
        clock (RunClock): The seat's run clock.

    Returns:
        bool: The seat is out of time.
    """
    if not _HARD_DEADLINE_SEC:
        return False
    if clock.elapsed() > _HARD_DEADLINE_SEC:
        return True
    phases = phase_budget.current()
    return phases is not None and phases.seat_salvage_due()

# A2 nudge — appended as a user turn when the model ends its turn (stop/end_turn)
# without ever calling submit_final_sql AND there's no run_sql_base SQL to salvage.
# The pipeline scores ONLY the rows submit_final_sql executes, so a plain-text
# answer scores zero; this reminds the model to actually submit.
_SUBMIT_NUDGE = _PROMPTS.surface("candidate", "nudge.submit")

# Final-check bounce (unconditional production behaviour, 07-30). The first SUCCESSFUL
# submit_final_sql is executed but NOT accepted: the model receives _FINAL_CHECK_NOTE and
# must submit again. The note carries the mechanical output-hygiene rules, delivered at the
# moment a final SQL exists. MOST of them left _SYSTEM; item 1 (output columns) did NOT —
# its seven sub-bullets are byte-identical in both places, so that rule alone is delivered
# twice, once early and once at the bounce. An audit caught this after the accounting below
# had claimed a clean move for a year; the accounting is kept as written because it records
# the INTENT, and this paragraph records what the code actually does. Construction-semantics
# rules (DISTINCT, per-row vs
# aggregate, superlative/LIMIT, HAVING, ratio granularity, JOIN defaults) stay in _SYSTEM —
# they shape the first draft and a post-submit fix would be a rewrite, not a touch-up.
# Loss-bounding: the bounced SQL already executed against the base tables (identical
# conditions to run_sql_base), so it is registered for A1 salvage — a model that never
# resubmits ends exactly as a pre-bounce run did. The bounce is suppressed once the time
# nudge / final loop-guard tier has fired: those sessions are being told to submit NOW and
# a "check once more" instruction would contradict it. No opt-out knob by design decision
# (07-30): reverting means reverting the commit, matching the promoted-knob doctrine in
# _cache_key. Design + review record: docs/FINAL_CHECK_PROMPT_REFACTOR_PLAN.md.
# Accounting of _SYSTEM text moved into the note (nothing is summarized away):
#   Check B item 1 header + sub-bullets        → item 1 (printf bullet → item 5)
#   IMPORTANT-output-columns bullet 1           → item 1 header + "GROUP BY columns" clause
#   IMPORTANT-output-columns bullet 2           → covered verbatim by item 1's rank-by bullet
#   IMPORTANT-output-columns bullet 3           → dropped: byte-duplicate of Check B DISTINCT
#                                                 item's second sentence, which stays early
#   Check B item 5 bullets 1-2 (IS NOT NULL / extra joins) → items 2-3
#   IMPORTANT-SQL-precision ORDER BY / ROUND bullets       → items 4-5
_FINAL_CHECK_NOTE = _PROMPTS.surface("candidate", "note.final_check")


# ── Helpers ───────────────────────────────────────────────────────────────────

_LOG_META_PREFIX = "[LOG ONLY]"


def _log(verbose: bool, *args: object) -> None:
    """Print display-only metadata with an unmistakable provenance label."""
    if verbose:
        message = " ".join(str(arg) for arg in args)
        for line in message.splitlines() or [""]:
            print(f"{_LOG_META_PREFIX} {line}".rstrip(), flush=True)


def _trace(verbose: bool, arrow: str, content: str) -> None:
    """Print one exact conversation payload with a direction arrow."""
    if verbose and content:
        print(arrow)
        print(content, flush=True)


def _trace_tool_call(
    verbose: bool,
    name: str,
    arguments: str | dict[str, object],
) -> None:
    """Mirror one structured tool call."""
    value = (
        arguments
        if isinstance(arguments, str)
        else json.dumps(arguments, ensure_ascii=False)
    )
    _trace(verbose, "◀", f"{name}\n{value}")


def _tool_result_content(
    verbose: bool,
    tool_name: str,
    result: object,
    inputs: object,
) -> str:
    """Render one tool result and log the exact content sent to the model.

    Arguments that decoded to something other than an object never execute: the loop or
    dispatch answers them with an error, which renders without them instead of raising.
    """
    content = model_tool_content(
        tool_name,
        result,
        inputs if isinstance(inputs, dict) else None,
        max_rows=_RUN_SQL_MAX_ROWS,
    )
    if _PROMPTS.has("shared.count_column"):
        # A documented thresholded count column gets its one-line reading note (count_columns.py).
        content += "".join(_PROMPTS.surface("shared", "count_column", column=name)
                           for name in count_columns.threshold_count_columns(tool_name, result))
    _log(verbose, f"result: {tool_name} ({len(content)} chars)")
    _trace(verbose, "▶", content)
    return content


# ── timeout salvage ──────────────────────────────────────────────────────────
# A question that hits the per-question wall used to lose EVERYTHING: the future is
# discarded, so a candidate that had already produced a working SQL at minute 5 still
# lands as a 'SELECT NULL' placeholder scoring 0. Candidates publish their latest
# run_sql_base-verified SQL here (same execution path as submit_final_sql), so the
# caller can fall back to real SQL instead of a guaranteed zero. LATEST WINS: each
# successful verify overwrites, so a first wrong attempt is replaced by the later fix.
_SALVAGE_LOCK = threading.Lock()
_SALVAGE: dict[tuple[str, str], str] = {}


def _salvage_key(db_id: str, question: str) -> tuple[str, str]:
    return (db_id or "", hashlib.sha256((question or "").encode()).hexdigest()[:16])


def record_salvage_sql(db_id: str, question: str, sql: str) -> None:
    """Publish the latest verified SQL for this question (called by the agent loops)."""
    if not sql or not sql.strip():
        return
    with _SALVAGE_LOCK:
        _SALVAGE[_salvage_key(db_id, question)] = sql


def take_salvage_sql(db_id: str, question: str) -> str | None:
    """Pop the salvaged SQL recorded for this question, if any candidate ever verified one.

    Args:
        db_id (str): Database the question was asked against.
        question (str): The question text, exactly as passed to :func:`nl2sql`.

    Returns:
        str | None: The salvaged SQL, removed from the store so a second call returns None;
        None when no candidate ever verified SQL for this question.
    """
    with _SALVAGE_LOCK:
        return _SALVAGE.pop(_salvage_key(db_id, question), None)


def _strip_db_id(schema: dict) -> dict:
    """Remove db_id / session_id from the tool schema — the agent injects both itself, the
    model never sets them."""
    _injected = {"db_id", "session_id", "original_question"}
    s = dict(schema or {})
    props = {k: v for k, v in s.get("properties", {}).items() if k not in _injected}
    required = [r for r in s.get("required", []) if r not in _injected]
    return {**s, "properties": props, "required": required}


def _normalize_tool_name(raw: str, known: set[str]) -> str:
    """Strip gateway-added namespace prefixes from a tool-call name.

    Some OpenAI-compatible gateways namespace function names returned by the
    model, which the MCP layer rejects as unknown tools. If the raw name is not a
    known tool but stripping a leading
    'namespace<sep>' segment yields one, use the stripped name. Separators seen
    in the wild are ':', '.', and '__'. Our own MCP
    tool names contain none of these, so this only ever fires on namespaced names.
    """
    if not raw or raw in known:
        return raw
    for sep in (":", ".", "__"):
        if sep in raw:
            tail = raw.rsplit(sep, 1)[-1]
            if tail in known:
                return tail
    return raw


def _parse_tool_args(name: str, raw: str) -> dict:
    """json.loads for a tool call's arguments, with salvage for gateway-garbled
    streams: dmxapi has been seen delivering 'Extra data' arguments — a complete
    JSON object followed by more text (a re-emit of the same object, or a second
    call's object concatenated into the same slot). Take the first complete
    object and log loudly; raising would fail the whole question. Arguments that
    don't even start with one valid object still raise, as before."""
    raw = raw or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        obj, end = json.JSONDecoder().raw_decode(raw)   # raises if nothing valid
        rest = raw[end:].strip()
        dup = " (re-emit of same object)" if rest == raw[:end].strip() else ""
        print(
            f"{_LOG_META_PREFIX} [ARG SALVAGE] {name}: kept first JSON object, "
            f"dropped {len(rest)} extra chars{dup}: {rest[:120]!r}",
            flush=True,
        )
        return obj


# ── Per-run session: server-side intermediate tables ──────────────────────────
# save_sql_as_table materialises an intermediate into THIS run's own session file on the
# server (see tools.sql). run_sql sees base tables + this run's intermediates; the model
# references them by plain name during exploration, then inlines them into ONE self-contained
# base-table SQL for the final answer (verifying it with run_sql_base, which — like
# submit_final_sql — sees the base DB ONLY). submit_final_sql runs against the base DB ONLY,
# so a SQL still referencing an intermediate fails ('no such table') — forcing a genuinely
# self-contained, reproducible answer. run_sql / run_sql_base / submit_final_sql are PURE
# executors; all inlining is the model's job.
#
# Each run uses a fresh uuid session_id, injected into the session tools like db_id, so the
# concurrent candidates/workers that share the one MCP process per db never collide (each has
# its own session file). The file is deleted at end-of-run (close_session).
_SESSION_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("_SESSION_ID", default=None)


def _recorder_for(role: str, route: str | None, slot: int | None = None) -> RenderRecorder:
    """Create the render recorder of one agent run.

    Every production route is declared in the entries table, so ``_PROMPTS.recorder`` names
    its keyspace. A route the table does not declare — an ad-hoc test route, or a name the
    route registry itself rejects one step later — is recorded under its own name with an
    empty keyspace instead of failing here, ahead of the registry's own error; its cache key
    still fails loudly through ``_PROMPTS.key_extra``.

    Args:
        role (str): ``candidate``, ``judge`` or ``reviewer``.
        route (str | None): Route name, ``None`` for the default route.
        slot (int | None): A candidate's position in the pool; its keyspace is the slot's.

    Returns:
        RenderRecorder: An unbound recorder for this run.
    """
    try:
        return _PROMPTS.recorder(role, route, slot)
    except KeyError:
        return RenderRecorder(_PROMPTS.id, _PROMPTS.proposal, role, route, "")


def _candidate_recorder(style: str | None, slot: int | None = None) -> RenderRecorder:
    """Return the render recorder of the current candidate run.

    ``_run_with_session`` binds one recorder per run on the task context; the provider
    loops read it back here so their renders and message digests land on the same
    snapshot. A loop entered directly (tests, one-off probes) has none bound, so a fresh
    recorder for this role/route is bound in place — the run still gets an identity.

    Args:
        style (str | None): The candidate route, ``None`` for the default route.
        slot (int | None): The candidate's position in the pool.

    Returns:
        RenderRecorder: The recorder bound to the current context.
    """
    recorder = RenderRecorder.current()
    if recorder is None:
        recorder = _recorder_for("candidate", style, slot)
        recorder.bind()
    return recorder


#: Marker written in place of ``prompt_identity`` when a cache entry predates the recorder:
#: the replay is real, but it must not claim the current release as its provenance.
_LEGACY_CACHE_IDENTITY = {"release": None, "provenance": "legacy-cache"}

_QUESTION_HINTS: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "_QUESTION_HINTS",
    default=(),
)
_ORIGINAL_QUESTION: contextvars.ContextVar[str] = contextvars.ContextVar("_ORIGINAL_QUESTION", default="")
#: The most recent successful ``run_sql_base`` of this run, as ``(stripped sql, result)``.
#:
#: ``submit_final_sql`` runs the committed SQL against the base tables — the same statement,
#: the same connection conditions, the same read-only benchmark database — so a model that
#: verifies and then submits the identical text executes it twice, and three times once the
#: final-check bounce asks for a second submission. What the first execution is actually
#: for is the proof that the statement runs: it gates the bounce and licenses the salvage
#: registration. When the text is byte-identical to one already proven, that proof exists
#: and its rows are the rows submit would return.
#:
#: Run-scoped like the two above, and reset in :func:`_run_one`, so a cached proof can
#: never outlive the question or reach a sibling candidate. Reuse additionally demands an
#: exact match on the stripped statement — anything else executes.
_VERIFIED_BASE: contextvars.ContextVar[Optional[tuple[str, dict]]] = contextvars.ContextVar(
    "_VERIFIED_BASE",
    default=None,
)

# How many times one run may be told its SQL exceeded the 30s final-answer budget before the
# harness stops asking for a rewrite.
#
# The comparison that sets this number is NOT "rewrite attempts vs a working answer" — it is
# "rewrite attempts vs a GUARANTEED ZERO". A statement over the official limit scores 0 with
# certainty, so every attempt is played against a loss that is already banked, and any faster
# rewrite is weakly better. That asymmetry argues for a generous cap, and the cost is bounded
# and small: each rejection burns at most one budget (30s) of CPU, so ten of them is 300s
# against a _HARD_DEADLINE_SEC of 1700 (default) — under 20% of the run, in the 2-of-1751 questions that reach
# this path at all.
#
# AT THE CAP the harness stops the rewrite conversation and lets the run end into A1 salvage.
# It does NOT accept the slow statement. That would bank the certain zero the whole mechanism
# exists to avoid, and salvage is strictly better: it can only hold SQL that PASSED this same
# 30s budget (record_salvage_sql is reached solely from a successful run_sql_base / a
# successful bounce), so its worst case is an earlier fast answer and, when the run verified
# nothing fast, no SQL at all — which scores the same 0 the slow query would have scored
# officially. Nothing is given up by refusing it.
_MAX_SLOW_SQL_REJECTS = int(os.environ.get("RUBIKSQL_MAX_SLOW_SQL_REJECTS", "10"))

#: Count of final-answer statements THIS run had cancelled for exceeding the 30s budget.
#:
#: Run-scoped like _VERIFIED_BASE and reset beside it, so a candidate can neither inherit a
#: sibling's count nor carry its own past the question. A one-element list rather than an int
#: because the counter is incremented from :func:`_call_mcp`, which every provider loop and the
#: judge share — rebinding a ContextVar there would only mutate that call's context.
_SLOW_SQL_REJECTS: contextvars.ContextVar[list] = contextvars.ContextVar(
    "_SLOW_SQL_REJECTS",
    default=[0],
)


def slow_sql_rejects() -> int:
    """How many final-answer statements this run had cancelled for exceeding the budget."""
    return _SLOW_SQL_REJECTS.get()[0]


def slow_sql_cap_reached() -> bool:
    """True once this run has spent its rewrite attempts (see _MAX_SLOW_SQL_REJECTS)."""
    return slow_sql_rejects() >= _MAX_SLOW_SQL_REJECTS


# tool name → schema property names; populated whenever tool schemas are fetched
# (_fetch_tools here, judge setup in vote.py). Used by _call_mcp to repair
# near-miss argument names from gateways — see _normalize_tool_args.
_TOOL_PROPS: dict[str, frozenset] = {}


def tool_input_schema(tool) -> dict:
    """Return one MCP tool's input schema under either field spelling.

    mcp 1.x names it ``inputSchema``; 2.x renamed it to ``input_schema`` and kept no
    read-side alias. Which one the installed client uses is not a detail this repo can pin
    to the version its own venv happens to hold: a fresh ``uv sync`` resolves 2.x, and
    reading the 1.x name there raises AttributeError inside every candidate. That surfaces
    as "all 5 candidates failed" — the whole eval dies with the cause one attribute deep in
    a traceback the collector swallows, which has now cost two full-run launches.

    Args:
        tool: An MCP ``Tool`` from either client generation.

    Returns:
        dict: JSON Schema for the tool's arguments, or ``{}`` when the tool declares none.
    """
    schema = getattr(tool, "input_schema", None)
    if schema is None:
        schema = getattr(tool, "inputSchema", None)
    return schema or {}


def tool_result_is_error(result) -> bool:
    """Return whether an MCP call failed, under either field spelling.

    Same 1.x/2.x rename as :func:`tool_input_schema` (``isError`` → ``is_error``), and the
    more dangerous of the two: an AttributeError here fires on the FIRST tool call, while
    reading the wrong name without raising would silently treat every failed call as a
    success and feed the model error text as data.
    """
    for attribute in ("is_error", "isError"):
        value = getattr(result, attribute, None)
        if value is not None:
            return bool(value)
    return False


def _register_tool_props(tools) -> None:
    for t in tools:
        props = tool_input_schema(t).get("properties", {})
        _TOOL_PROPS[t.name] = frozenset(props)


def _normalize_tool_args(name: str, inputs: dict, *, quiet: bool = False) -> dict:
    """Clean the arguments a model emitted before dispatch:

      1. Rename near-miss names — models may call get_col_info with 'table' instead
         of 'table_name'; a sent key k is renamed to schema key p when k isn't
         in the schema, p wasn't also sent, and stripping a '_name' suffix makes them
         equal (table→table_name, col→col_name).
      2. DROP arguments the tool never declared — some models tack a dummy arg
         ('_noargs'='unused', 'random_string', …) onto a no-argument tool like
         get_db_info. Passing it through makes the strict MCP layer reject the ENTIRE
         call with pydantic 'unexpected keyword argument', so the model only sees a
         failure it can't diagnose (and can't load the schema). Dropping it lets the
         real call run.

    Calls whose args are already all valid pass through untouched. Before live tool
    registration (cache backfill), use the same pinned declaration's properties.
    ``quiet`` suppresses repair diagnostics during replay eligibility checks.
    """
    if not isinstance(inputs, dict):
        return inputs  # the shared argument validator reports the ordinary error
    props = _TOOL_PROPS.get(name)
    if props is None:
        declaration = next((record for record in tool_contract.declaration_tools()
                            if record["name"] == name), None)
        props = frozenset(declaration["input_schema"].get("properties", {})) if declaration else None
    if props is None or all(k in props for k in inputs):
        return inputs

    def base(s: str) -> str:
        return s[:-5] if s.endswith("_name") else s

    # Known near-miss spellings models emit for the 'sql' parameter. Dropping these (the
    # pre-existing behaviour) let submit_final_sql run with sql="" — which the server
    # executes as a no-op WITHOUT error — silently submitting an empty final answer.
    _SQL_ALIASES = {"query", "sql_query", "final_sql", "sql_text", "statement"}

    out: dict = {}
    for k, v in inputs.items():
        if k in props:
            out[k] = v
            continue
        if k in _SQL_ALIASES and "sql" in props and "sql" not in inputs and "sql" not in out:
            if not quiet:
                print(
                    f"{_LOG_META_PREFIX} [ARG FIX] {name}: renamed {k!r} → 'sql'",
                    flush=True,
                )
            out["sql"] = v
            continue
        matches = [p for p in props if p != k and base(p) == base(k)]
        if len(matches) == 1 and matches[0] not in inputs and matches[0] not in out:
            out[matches[0]] = v
        elif not quiet:
            print(
                f"{_LOG_META_PREFIX} [ARG DROP] {name}: "
                f"dropped undeclared arg {k!r}={v!r}",
                flush=True,
            )
    return out


def _forward_question(name: str, inputs: dict, hints: tuple[str, ...] = ()) -> dict:
    """Return ``inputs`` with the question text :func:`_call_mcp` forwards.

    A ``question`` of any type reaches the executor as ``question_with_hint`` of its text,
    and a falsy one as empty text, so the executor always receives a string.

    Args:
        name (str): Declared tool name.
        inputs (dict): Repaired model arguments.
        hints (tuple[str, ...]): The run's question hints.

    Returns:
        dict: ``inputs`` unchanged when the tool takes no question or none was sent,
            otherwise a copy carrying the forwarded text.
    """
    if "question" not in inputs or "question" not in tool_contract.callable_parameters(name):
        return inputs
    return {**inputs, "question": question_with_hint(str(inputs.get("question") or ""), hints)}


def _tool_argument_error(name: str, inputs: object) -> str | None:
    """Validate the arguments the host forwards for one model call.

    Dispatch and replay eligibility share this check. It receives the arguments after the
    dispatcher's repairs and run-state replacements, applies the question rewrite every
    dispatch performs later, and validates the result with the executor's lax adapter.
    Final SQL must also be non-empty text. Nothing the pre-validation dispatcher executed is
    rejected; a non-string ``sql`` now returns this error instead of raising.

    Args:
        name (str): Normalized name of a declared tool.
        inputs (object): Repaired model arguments, including any run-state replacement.

    Returns:
        str | None: The ordinary invalid-arguments error, or ``None``.
    """
    if isinstance(inputs, dict):
        inputs = _forward_question(name, inputs)
    error = tool_contract.argument_error(name, inputs)
    if error is not None:
        return error
    if name == "submit_final_sql":
        sql = inputs.get("sql") if isinstance(inputs, dict) else None
        if not isinstance(sql, str) or not sql.strip():
            return "Invalid arguments for submit_final_sql: 'sql' must be a non-empty string"
    return None


def _replay_tool_args_valid(name: str, inputs: object, *, candidate: bool = False) -> bool:
    """Whether dispatch would accept a recorded call's arguments.

    The dispatcher's name and argument repairs run first, then :func:`_tool_argument_error`.
    Only certain rejections count. A name outside the pinned declaration has no argument
    contract: the loops answer it with an ordinary error before any dispatch, so the reply
    keeps replaying. An argument the dispatcher overwrites from run state that the transcript
    does not record is not judged. Keeping a reply that then fails costs one replayed failure;
    rejecting a reply that executed would regenerate its turn and break counted-child pairing.

    Args:
        name (str): Recorded tool name, possibly gateway-namespaced.
        inputs (object): Decoded recorded arguments; anything but an object is rejected.
        candidate (bool): Judge the call as :func:`_dispatch_capability` dispatches it. That
            dispatcher overwrites ``get_rules.exclude_ids`` with the run's triggered rule
            ids whenever any rule fired; the judge and reviewer forward it unchanged.

    Returns:
        bool: ``False`` only when dispatch certainly rejects the arguments.
    """
    known = {record["name"] for record in tool_contract.declaration_tools()}
    name = _normalize_tool_name(name, known)
    if name not in known:
        return True
    inputs = _normalize_tool_args(name, inputs, quiet=True)
    if candidate and name == "get_rules" and isinstance(inputs, dict):
        inputs = {key: value for key, value in inputs.items() if key != "exclude_ids"}
    return _tool_argument_error(name, inputs) is None


def _replayable_calls(calls: list[tuple[str, object]], *, tools: frozenset[str],
                      candidate: bool = False) -> bool:
    """Whether dispatch accepts every recorded call of one reply that it certainly reaches.

    The loops answer a name the run does not expose with an ordinary error before dispatch.
    Once a ``submit_final_sql`` call is dispatched, an accepted submit leaves the rest of the
    reply unexecuted; acceptance depends on the final-check bounce, route gates and live
    results, which the transcript does not settle, so no later call can disqualify the
    reply. Every other call is dispatched and must pass :func:`_replay_tool_args_valid`.
    Decoding failures were already rejected by :func:`turn_memo.reply_replayable`.

    Args:
        calls (list[tuple[str, object]]): Recorded names and decoded arguments, in reply order.
        tools (frozenset[str]): The names the run's loop exposes and dispatches.
        candidate (bool): Judge calls as :func:`_dispatch_capability` dispatches them.

    Returns:
        bool: ``False`` only when a certainly dispatched call has rejected arguments.
    """
    for raw_name, arguments in calls:
        name = _normalize_tool_name(raw_name, tools)
        if name not in tools:
            continue
        if not _replay_tool_args_valid(name, arguments, candidate=candidate):
            return False
        if name == "submit_final_sql":
            return True
    return True


async def _raw_call(session: ClientSession, name: str, inputs: dict, db_id: str):
    """Bare MCP tool call (no temp-table rewrite/recording); inputs used verbatim.

    On an MCP-level error (e.g. pydantic arg validation, bad SQL) return the error
    TEXT to the model as {"error": ...} instead of swallowing it as []. The model
    can then self-correct (fix the arg name / SQL) on the next turn — returning []
    left it staring at an empty result with no idea why. The submit/run_sql call
    sites all gate on `isinstance(result, dict) and "error" not in result`, so an
    error dict is still correctly treated as a non-result there. A task its deadline
    stopped dispatches nothing (asyncio.CancelledError), even after consuming its one
    cancellation."""
    refuse_if_stopped()
    if not tool_contract.dispatch_allowed(name):
        return {"error": f"Unauthorized tool: {name}"}
    properties = tool_contract.callable_parameters(name)
    bound = {key: value for key, value in inputs.items() if key not in {"db_id", "session_id", "original_question"}}
    if "db_id" in properties:
        bound["db_id"] = db_id
    if "session_id" in properties and _SESSION_ID.get():
        bound["session_id"] = _SESSION_ID.get()
    if "original_question" in properties and _ORIGINAL_QUESTION.get():
        bound["original_question"] = _ORIGINAL_QUESTION.get()
    # The full round trip includes MCP transport and the server's dispatch queue.
    with session_timing.measure("tool", name):
        from rubiksql_hb.example_mask import QUESTION_META, READERS
        if name in READERS and "original_question" not in properties and _ORIGINAL_QUESTION.get():
            result = await session.call_tool(name, bound, meta={QUESTION_META: _ORIGINAL_QUESTION.get()})
        else:
            result = await session.call_tool(name, bound)
    if tool_result_is_error(result):
        err = result.content[0].text if result.content else "(no error text)"
        print(f"{_LOG_META_PREFIX} [MCP ERROR] {name}: {err}", flush=True)
        return {"error": err}
    out: object = []
    if result.content and result.content[0].type == "text":
        try:
            out = json.loads(result.content[0].text)
            if name in {"run_sql", "run_sql_base", "submit_final_sql"}:
                out = restore_sql_results(out)
        except (json.JSONDecodeError, ValueError):
            out = result.content[0].text
    return out


class _RuleTriggerState:
    """Conversation-local rule injection state.

    Only new user and assistant natural-language text is passed to :meth:`match`. Tool
    results and the blocks produced by :meth:`match` never enter the matcher. A rule id is
    recorded on first match, so the same rule cannot be injected twice in one run.
    """

    def __init__(self, session: ClientSession, db_id: str) -> None:
        self.session = session
        self.db_id = db_id
        self.triggered_ids: set[str] = set()
        self.triggered_titles: set[str] = set()
        self.degraded_retrieval = False

    async def match(self, text: str) -> str:
        """Return one injection block containing every newly matched rule."""
        if not text.strip():
            return ""
        result = await _call_mcp(
            self.session,
            "match_rule_triggers",
            {"text": text, "exclude_ids": sorted(self.triggered_ids)},
            self.db_id,
        )
        self.degraded_retrieval |= degraded_knowledge_result("match_rule_triggers", result)
        if not isinstance(result, list):
            return ""
        rules = [
            rule for rule in result
            if isinstance(rule, dict) and rule.get("virtual_id") and "error" not in rule
            # Titles are the only identity stable ACROSS sessions: the judge's baseline
            # 15 are retrieved in a throwaway knowledge session whose virtual ids never
            # reach this one, so id-based exclude_ids cannot see them.
            and str(rule.get("title") or "") not in self.triggered_titles
        ]
        if not rules:
            return ""
        self.triggered_ids.update(str(rule["virtual_id"]) for rule in rules)
        self.triggered_titles.update(str(rule.get("title") or "") for rule in rules)
        body = render_rules(
            rules,
            heading=_PROMPTS.surface("shared", "rules_heading.triggered"),
            explain_scope_conflict=True,
        )
        return _PROMPTS.surface("shared", "rule_hooks", body=body)

    def mark_served(self, result: object) -> None:
        """Record rules the model just received via ``get_rules`` as already served.

        The dedup used to be one-way: delivered hooks were stripped from later
        ``get_rules`` results, but a rule the model FETCHED itself stayed eligible for
        hook delivery and could arrive twice in one conversation. Serving is serving,
        whichever door it came through."""
        if not isinstance(result, list):
            return
        for row in result:
            if not isinstance(row, dict):
                continue
            if row.get("virtual_id"):
                self.triggered_ids.add(str(row["virtual_id"]))
            if row.get("title"):
                self.triggered_titles.add(str(row["title"]))

    def without_triggered(self, result: object) -> object:
        """Remove already injected rules from a model-requested ``get_rules`` result."""
        if not isinstance(result, list) or not self.triggered_titles:
            return result
        return [
            row for row in result
            if not isinstance(row, dict)
            or str(row.get("title") or "") not in self.triggered_titles
        ]


async def _submit_final(session: ClientSession, inputs: dict, db_id: str):
    """Finalize: submit the model's SQL VERBATIM against the base DB only (submit_final_sql
    does NOT see this run's intermediate tables). The committed SQL must therefore be
    self-contained over the base tables — one still referencing an intermediate fails here
    ('no such table') and the agent loop lets the model inline it and retry. _final_sql is
    exactly what ran. The agent loop ends on the first successful submit."""
    if not tool_contract.dispatch_allowed("submit_final_sql"):
        return {"error": "Unauthorized tool: submit_final_sql"}
    error = _tool_argument_error("submit_final_sql", inputs)
    if error is not None:
        return {"error": error}
    sql = inputs["sql"]
    verified = _VERIFIED_BASE.get()
    if verified is not None and verified[0] == sql.strip():
        # Already proven this run, on the same statement against the same base tables.
        # `final` is what submit_sql adds on top of exec_sql, so the reused payload is
        # what a re-execution would have returned.
        out = {**verified[1], "final": True, "_final_sql": sql}
        return out
    submit_inputs = {"sql": sql}
    out = await _raw_call(session, "submit_final_sql", submit_inputs, db_id)
    if isinstance(out, dict) and "error" not in out:
        out["_final_sql"] = sql
    return out


async def _call_mcp(session: ClientSession, name: str, inputs: dict, db_id: str):
    """Forward a tool call, injecting db_id and any run-scoped session id.

    SQL tools use the session for intermediate tables. Knowledge tools use it for short,
    stable R/E ids and cached example payloads. ``run_sql`` and ``submit_final_sql`` remain
    pure executors; submit goes through :func:`_submit_final` only to record committed SQL.
    """
    if not tool_contract.dispatch_allowed(name):
        return {"error": f"Unauthorized tool: {name}"}
    fixed = _normalize_tool_args(name, inputs)
    if fixed != inputs:
        print(
            f"{_LOG_META_PREFIX} [ARG FIX] {name}: "
            f"{sorted(inputs)} → {sorted(fixed)}",
            flush=True,
        )
        inputs = fixed
    error = _tool_argument_error(name, inputs)
    if error is not None:
        return {"error": error}
    inputs = _forward_question(name, inputs, _QUESTION_HINTS.get())
    if name == "submit_final_sql":
        result = await _submit_final(session, inputs, db_id)
        _note_slow_sql(name, inputs, result)
        return result
    result = await _raw_call(session, name, inputs, db_id)
    if name == "run_sql_base" and isinstance(result, dict) and "error" not in result:
        # Remember what this proved, so an identical submit does not prove it again. Set
        # here rather than in the agent loops because run_sql_base reaches MCP from both
        # the text-action path and the native tool-call path, and a proof missed on one
        # of them would silently cost an execution rather than fail.
        _VERIFIED_BASE.set((str(inputs.get("sql") or "").strip(), result))
    _note_slow_sql(name, inputs, result)
    return result


def _note_slow_sql(name: str, inputs: dict, result: object) -> None:
    """Count and ANNOUNCE a final-answer statement the executor cancelled for being too slow.

    Counted here rather than in the provider loops because run_sql_base and submit_final_sql
    reach MCP from the candidate loops AND from the judge, and a rejection missed on one of
    those paths would silently raise the cap for that run.

    Every rejection prints immediately. A retry loop that stays silent until its last attempt
    is indistinguishable from a hang while it is happening, and a run that quietly spends ten
    30-second budgets is exactly the shape worth seeing as it develops rather than afterwards.
    """
    if name not in ("run_sql_base", "submit_final_sql") or not sql_tools.is_timeout_error(result):
        return
    counter = _SLOW_SQL_REJECTS.get()
    counter[0] += 1
    sql = " ".join(str(inputs.get("sql") or "").split())
    print(
        f"{_LOG_META_PREFIX} [SLOW SQL {counter[0]}/{_MAX_SLOW_SQL_REJECTS}] {name} exceeded "
        f"the {int(sql_tools._FINAL_SQL_TIMEOUT_SEC)}s final-answer budget and was rejected"
        + ("; no further rewrites will be requested — falling back to salvage"
           if slow_sql_cap_reached() else "")
        + f": {sql[:200]}",
        flush=True,
    )


async def _fetch_tools(
    session: ClientSession,
    disabled: set[str] | None = None,
    style: str | None = None,
):
    """Fetch NL→SQL tool schemas from MCP server."""
    all_tools = (await session.list_tools()).tools
    _register_tool_props(all_tools)
    active = _nl2sql_tools(disabled, style)
    selected = [t for t in all_tools if t.name in active]
    tool_contract.assert_live_contract(
        all_tools, "candidate", route=style or "default", disabled=disabled,
    )
    return selected


def _model_mcp_tools(tools: list[object]) -> list[object]:
    """Return every capability as a provider-native tool schema.

    sql_native_v2 restores the pre-7f02fdc transport: SQL tools are model-visible
    native tools again. The 7/27 archive measured 0 JSON-argument failures across
    31,486 native SQL calls (67% multi-line), so the escaping risk the text protocol
    guarded against never materialized, while the protocol itself cost 5.5% of
    candidate turns and 15.2% of judge turns in rejections.
    """
    return list(tools)


async def _dispatch_capability(
    session: ClientSession,
    hooks,
    rule_triggers: _RuleTriggerState,
    name: str,
    inputs: dict,
    db_id: str,
) -> tuple[dict, object]:
    """Execute one validated runtime capability through the shared hook pipeline."""
    if not tool_contract.dispatch_allowed(name, calling="model"):
        return inputs, {"error": f"Unauthorized tool: {name}"}
    inputs = _normalize_tool_args(name, inputs)
    if name == "get_rules" and rule_triggers.triggered_ids and isinstance(inputs, dict):
        # Exclude at retrieval, not after it. Ranking would otherwise spend slots on rules
        # the conversation already injected in full — and spend them preferentially, since
        # _text_score's trigger term rewards the same match that injected them.
        inputs = {**inputs, "exclude_ids": sorted(rule_triggers.triggered_ids)}
    # Validate what is forwarded, after the replacement, and before any route hook reads it.
    error = _tool_argument_error(name, inputs)
    if error is not None:
        return inputs, {"error": error}
    result = await hooks.before_tool(name, inputs) if hooks is not None else None
    if result is None:
        result = await _call_mcp(session, name, inputs, db_id)
    if hooks is not None:
        result = await hooks.after_tool(name, result)
    if name == "get_rules":
        # Retrieval-side exclusion is keyed on this run's R-ids; a rule injected under a
        # different id but the same title still has to go, so this stays as the backstop.
        result = rule_triggers.without_triggered(result)
        rule_triggers.mark_served(result)
    return inputs, result


def _action_result_content(
    verbose: bool,
    name: str,
    result: object,
    inputs: dict[str, object],
    extra: str = "",
) -> str:
    """Render and log the exact ordinary user message following one SQL action."""
    content = model_tool_content(
        name,
        result,
        inputs,
        max_rows=_RUN_SQL_MAX_ROWS,
    ) + extra
    _log(verbose, f"result: {name} ({len(content)} chars)")
    _trace(verbose, "▶", content)
    return content


# ── Provider loops ────────────────────────────────────────────────────────────

async def _maybe_salvage(
    session,
    db_id,
    last_sql_result,
    last_sql,
    last_base_sql,
    sql_actions,
    verbose,
):
    """A1 salvage: a model that verified its answer with run_sql_base and then
    replied in plain text (never calling submit_final_sql) leaves a perfectly good
    SQL on the floor. run_sql_base executes against the BASE TABLES ONLY — the exact
    same conditions as submit_final_sql — so submitting that SQL is semantically
    identical to the model having submitted it. Do that instead of failing the run.
    No-op if the model already submitted, or there's nothing to salvage."""
    if last_sql_result is not None or not last_base_sql:
        return last_sql_result, last_sql
    salv = await _submit_final(session, {"sql": last_base_sql}, db_id)
    if isinstance(salv, dict) and "error" not in salv:
        last_sql = salv.pop("_final_sql", last_base_sql)
        salv["_salvaged"] = "run_sql_base"
        sql_actions.append({
            "name": "submit_final_sql",
            "args": {"sql": last_sql},
            "status": "salvaged",
        })
        notice = (
            "[SALVAGE] model answered in text without submit; "
            "submitted its last run_sql_base SQL"
        )
        if verbose:
            _log(verbose, notice)
            _log(verbose, f"[SALVAGE SQL preview] {last_sql[:200]}")
        else:
            print(notice, flush=True)
        return salv, last_sql
    return last_sql_result, last_sql


async def _run_claude(question: str, model: str, db_id: str,
                      session: ClientSession, verbose: bool,
                      disabled_tools: set[str] | None = None,
                      style: str | None = None,
                      slot: int | None = None) -> dict:
    route = _candidate_route(style)
    style_block = route.candidate_block if route is not None else ""
    hooks = route.candidate_hooks(route, question) if route is not None else None
    contract = tool_contract.role_contract("candidate", route=style or "default", disabled=disabled_tools)
    mcp_tools = await _fetch_tools(session, disabled_tools, style)
    model_mcp_tools = _model_mcp_tools(mcp_tools)
    tools = [
        {"name": t.name, "description": t.description or "",
         "input_schema": _strip_db_id(tool_input_schema(t))}
        for t in model_mcp_tools
    ]
    known_tools = {t.name for t in mcp_tools}
    native_tools = known_tools
    if hooks is not None:
        with tool_contract.dispatch_context("candidate", style or "default", disabled_tools):
            await hooks.init(session, db_id, known_tools)

    # Keep the orchestration lifetime explicit even though HeavenBase owns and reuses
    # the underlying SDK client pool.
    with tool_contract.dispatch_context("candidate", style or "default", disabled_tools), nullcontext(build_llm("claude", model, role="candidate")) as llm:
        recorder = _candidate_recorder(style, slot)
        # Synonym notes are APPENDED, never substituted into the question — see
        # _synonym_context for why rewriting it measured -1.00pp.
        system_content = _candidate_system_content(style, style_block, slot)
        rule_triggers = _RuleTriggerState(session, db_id)
        user_content = await _candidate_user_content(question, db_id, rule_triggers, slot)
        recorder.message("system", system_content)
        recorder.message("user", user_content)
        _log_prompt(verbose, system_content, user_content)
        messages = [{"role": "user", "content": user_content}]
        last_sql_result = None
        last_sql        = None
        last_base_sql   = None   # A1: SQL of the most recent successful run_sql_base
        nudges          = 0      # A2: count of submit nudges issued
        turn = 0
        all_tool_calls: list[dict] = []
        sql_actions: list[dict] = []
        protocol_events: list[dict] = []
        # Logical clock (turn_memo.RunClock): the hard deadline, the bounce suppression and
        # the time nudge read elapsed(); duration_sec reads the wall clock. No replay on this
        # path (it has no cache), so the offset stays zero and elapsed() is the wall clock.
        clock = RunClock(time.monotonic)
        run_usage: dict = {}
        time_nudged = False
        final_checked  = False   # final-check bounce fired (at most once per run)
        finalcheck_sql = None    # SQL held at the bounce, for the changed/unchanged marker
        degraded_retrieval = rule_triggers.degraded_retrieval

        while True:
            turn += 1
            if turn > _MAX_TURNS:
                raise RuntimeError(f"Exceeded max turns ({_MAX_TURNS}) without a successful submit")
            if _seat_out_of_time(clock):
                break
            _log(verbose, "─" * 60)
            _log(verbose, f"[Turn {turn}] → {model}")

            response = await achat_turn(
                llm,
                messages,
                system=system_content,
                tools=tools,
            )
            merge_usage(run_usage, response.usage)
            messages.append(response.message)
            tool_blocks = list(response.tool_calls)
            # Native transport: SQL rides tool calls, so the assistant text is pure prose —
            # exactly what the rule-trigger matcher wants (no SQL body to exclude first).
            triggered_block = await rule_triggers.match(response.text or "")
            degraded_retrieval |= rule_triggers.degraded_retrieval
            if response.text:
                _trace(verbose, "◀", response.text)
            for block in tool_blocks:
                function = block.get("function") or {}
                _trace_tool_call(
                    verbose,
                    str(function.get("name") or ""),
                    str(function.get("arguments") or "{}"),
                )

            if tool_blocks:
                tool_results = []
                any_knowledge = False
                for block in tool_blocks:
                    function = block.get("function") or {}
                    raw_name = str(function.get("name") or "")
                    inputs = _parse_tool_args(
                        raw_name,
                        str(function.get("arguments") or "{}"),
                    )
                    tool_name = _normalize_tool_name(raw_name, known_tools)
                    block_id = str(block.get("id") or "")
                    if tool_name not in native_tools:
                        # The server still registers tools this route disabled, so an
                        # unguarded dispatch would execute past route policy (e.g. the
                        # dnp checkpoint's withheld get_col_info). Answer per-call,
                        # name the tool, never reach MCP.
                        result: object = {"error": f"Unknown tool: {tool_name}"}
                        all_tool_calls.append({"name": tool_name, "args": inputs})
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block_id,
                            "content": _tool_result_content(
                                verbose, tool_name, result, inputs,
                            ),
                        })
                        continue
                    if last_sql_result is not None:
                        # A submit in this same batch already finalized the run. Later
                        # calls stay unexecuted, but every tool_use still gets a reply
                        # so the transcript honors the provider contract.
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block_id,
                            "content": "final SQL already accepted this turn; "
                                       "call not executed",
                        })
                        continue
                    inputs, result = await _dispatch_capability(
                        session,
                        hooks,
                        rule_triggers,
                        tool_name,
                        inputs,
                        db_id,
                    )
                    if tool_name not in SQL_ACTIONS:
                        any_knowledge = True
                        if degraded_knowledge_result(tool_name, result):
                            degraded_retrieval = True
                        all_tool_calls.append({"name": tool_name, "args": inputs})
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block_id,
                            "content": _tool_result_content(
                                verbose, tool_name, result, inputs,
                            ),
                        })
                        continue
                    # SQL action semantics, unchanged from the text era — only the
                    # envelope moved (tool_result instead of a user message).
                    if (tool_name == "run_sql_base"
                            and isinstance(result, dict) and "error" not in result):
                        last_base_sql = inputs.get("sql", "") or last_base_sql
                        record_salvage_sql(db_id, question, last_base_sql)
                    if (not final_checked
                            and tool_name == "submit_final_sql"
                            and isinstance(result, dict) and "error" not in result
                            and not time_nudged
                            and not (_TIME_NUDGE_SEC
                                     and clock.elapsed() > _TIME_NUDGE_SEC)):
                        # Transactional bounce: the SQL just EXECUTED against the base
                        # tables (identical conditions to run_sql_base), so register it
                        # for A1 salvage and hand back the checklist instead of
                        # finalizing — a model that never resubmits still ends with
                        # this verified SQL submitted. An errored submit takes the
                        # normal error path and does not consume the bounce; past the
                        # time nudge the bounce is skipped entirely.
                        final_checked = True
                        finalcheck_sql = result.get("_final_sql") or inputs.get("sql", "")
                        last_base_sql = finalcheck_sql
                        record_salvage_sql(db_id, question, last_base_sql)
                        sql_actions.append({
                            "name": tool_name,
                            "args": inputs,
                            "status": "final_check",
                        })
                        protocol_events.append({
                            "turn": turn,
                            "status": "final_check",
                            "action": tool_name,
                        })
                        rank_missing = (_PROMPTS.has("candidate.nudge.rank_output")
                                        and missing_rank_output(question, finalcheck_sql, result.get("columns", [])))
                        protocol_events[-1]["rank_output_missing"] = rank_missing
                        final_check_note = _PROMPTS.surface("candidate", "note.final_check")
                        if rank_missing:
                            final_check_note += "\n" + _PROMPTS.surface("candidate", "nudge.rank_output")
                        shape_note, shape_record = await _result_shape_note(
                            session, db_id, finalcheck_sql, result,
                            probe_allowed=not (_TIME_NUDGE_SEC and clock.elapsed() > _TIME_NUDGE_SEC - _SHAPE_PROBE_MARGIN_SEC))
                        if shape_record is not None:
                            protocol_events[-1]["result_shape"] = shape_record
                        if shape_note:
                            final_check_note += "\n" + shape_note
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block_id,
                            "content": final_check_note,
                        })
                        _trace(verbose, "▶", final_check_note)
                        continue
                    if (tool_name == "submit_final_sql"
                            and isinstance(result, dict) and "error" not in result):
                        result = dict(result)
                        last_sql = result.pop("_final_sql", inputs.get("sql", ""))
                        last_sql_result = result
                    sql_actions.append({
                        "name": tool_name,
                        "args": inputs,
                        "status": "error"
                        if isinstance(result, dict) and "error" in result
                        else "ok",
                    })
                    protocol_events.append({
                        "turn": turn,
                        "status": "ok",
                        "action": tool_name,
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block_id,
                        "content": _action_result_content(
                            verbose, tool_name, result, inputs,
                        ),
                    })
                if any_knowledge:
                    protocol_events.append({"turn": turn, "status": "native"})
                if triggered_block and last_sql_result is None:
                    # No delivery on a run-ending turn: an accepted submit means no
                    # further model turn will ever read the block.
                    tool_results.append({"type": "text", "text": triggered_block})
                    _trace(verbose, "▶", triggered_block)
                if (_TIME_NUDGE_SEC and not time_nudged and last_sql_result is None
                        and clock.elapsed() > _TIME_NUDGE_SEC):
                    time_nudged = True
                    _nmin = int(clock.elapsed() // 60)
                    time_content = _PROMPTS.surface("candidate", "nudge.time", mins=_nmin)
                    tool_results.append({"type": "text", "text": time_content})
                    _trace(verbose, "▶", time_content)
                messages.append({"role": "user", "content": tool_results})
                if last_sql_result is not None:
                    break
                if slow_sql_cap_reached():
                    # Spent every rewrite attempt on statements over the 30s final-answer
                    # budget (_MAX_SLOW_SQL_REJECTS). Stop asking and let A1 salvage below
                    # decide: it can only hold SQL that PASSED that same budget, so this
                    # ends on an earlier fast answer or on none — never on the slow one.
                    break
                continue

            if not tool_blocks:
                protocol_events.append({"turn": turn, "status": "no_action"})
                if hooks is not None:
                    correction = await hooks.on_stop(last_base_sql)
                    if correction is not None:
                        last_base_sql = None
                        correction_content = correction["error"] + triggered_block
                        messages.append({
                            "role": "user",
                            "content": correction_content,
                        })
                        _trace(verbose, "▶", correction_content)
                        continue
                # A2: model ended its turn without submitting. If there's a run_sql_base
                # SQL to salvage (A1, handled after the loop), stop and salvage; otherwise
                # nudge it to actually submit, up to _MAX_SUBMIT_NUDGES times.
                if last_sql_result is None and last_base_sql is None and nudges < _MAX_SUBMIT_NUDGES:
                    nudges += 1
                    nudge_content = _PROMPTS.surface("candidate", "nudge.submit") + triggered_block
                    messages.append({
                        "role": "user",
                        "content": nudge_content,
                    })
                    _trace(verbose, "▶", nudge_content)
                    continue
                break

        last_sql_result, last_sql = await _maybe_salvage(
            session, db_id, last_sql_result, last_sql, last_base_sql, sql_actions, verbose)
        if last_sql_result is None:
            raise RuntimeError("Model did not call submit_final_sql or all attempts failed")
        result = {**last_sql_result, "sql": last_sql,
                  "implementation_context": {"context": contract["context"], "digest": contract["digest"]},
                  "tool_calls": all_tool_calls,
                  "sql_actions": sql_actions,
                  "protocol_events": protocol_events,
                  "usage": run_usage,
                  "duration_sec": round(clock.real_elapsed(), 3),
                  "final_check": {
                      "bounced": final_checked,
                      "changed": bool(final_checked and last_sql != finalcheck_sql),
                  },
                  "messages": messages,
                  # Present only when set: the row-level flag eval_bird copies out, and
                  # deliberately part of the cached record — a replayed degraded run is
                  # still a degraded answer.
                  **({"degraded_retrieval": True} if degraded_retrieval else {})}
        if hooks is not None:
            result["route_debug"] = hooks.route_debug(style_block)
        result = publish_rank_result(question, result, [], enabled=False)
        recorder.set_replay(turns=turn)
        result["prompt_identity"] = recorder.snapshot()
        return result


async def _run_openai_compat(question: str, model: str, provider: str,
                              db_id: str, session: ClientSession, verbose: bool,
                              seed: int | None = None,
                              disabled_tools: set[str] | None = None,
                              temperature: float | None = None,
                              style: str | None = None,
                              slot: int | None = None) -> dict:
    route = _candidate_route(style)
    style_block = route.candidate_block if route is not None else ""
    # key = sha256(model + system_prompt + per-db rules/examples + db_id + question + seed + disabled_tools)
    route_extras = {"provider": provider}
    if temperature is not None:
        route_extras["t"] = temperature
    if route is not None:
        route_extras.update(route.cache_extras(db_id, question))
    contract = tool_contract.role_contract("candidate", route=style or "default", disabled=disabled_tools)
    ck, base_ck = _cache_key_pair(model, [{"db_id": db_id, "question": question}], seed, db_id,
                                  disabled_tools, route_extras or None, slot)
    cached = _cache_get(ck)
    if cached is not None:
        # A replay makes no API call: the stored counters belong to the run that first
        # paid for them, and returning them would bill this run for tokens and seconds
        # it never spent — inverting exactly the arm-vs-arm cost comparison the lab's
        # usage fields exist to support. Report the replay as unmeasured, and say why.
        # An entry written before the recorder existed carries no identity; it is marked
        # as such rather than credited to the release this process happens to run.
        # The stored transcript is a replay source for later releases; index it now if
        # the run that wrote it never did (turn_memo.py).
        _backfill_turn_index(ck, base_ck, cached, frozenset(contract["context"]["tools"]))
        replayed = cache_lineage.replay(
            {**cached, "prompt_identity": cached.get("prompt_identity") or dict(_LEGACY_CACHE_IDENTITY)},
            role="candidate", slot=slot if slot is not None else 1, capture=True,
        )
        replayed["consumer_implementation_context"] = {"context": contract["context"], "digest": contract["digest"]}
        return replayed

    mcp_tools = await _fetch_tools(session, disabled_tools, style)
    model_mcp_tools = _model_mcp_tools(mcp_tools)
    tools = [
        {"type": "function", "function": {
            "name": t.name, "description": t.description or "",
            "parameters": _strip_db_id(tool_input_schema(t)),
        }}
        for t in model_mcp_tools
    ]
    known_tools = {t.name for t in mcp_tools}
    native_tools = known_tools
    hooks = route.candidate_hooks(route, question) if route is not None else None
    if hooks is not None:
        with tool_contract.dispatch_context("candidate", style or "default", disabled_tools):
            await hooks.init(session, db_id, known_tools)

    recorder = _candidate_recorder(style, slot)
    system_content = _candidate_system_content(style, style_block, slot)
    attempt = {"messages": [], "usage": {}, "tool_calls": [], "sql_actions": [], "protocol_events": [],
               "cache_key": ck, "memo_base_key": base_ck,
               "implementation_context": {"context": contract["context"], "digest": contract["digest"]},
               "execution": cache_lineage.identity(db=db_id, role="candidate", slot=slot if slot is not None else 1,
                                                    release=_PROMPTS.id, knowledge_fingerprint=_db_knowledge_fingerprint(db_id))}
    with tool_contract.dispatch_context("candidate", style or "default", disabled_tools), _candidate_failure_record(attempt, recorder), nullcontext(build_llm(
        provider,
        model,
        role="candidate",
        seed=seed,
        temperature=temperature,
    )) as llm:
        rule_triggers = _RuleTriggerState(session, db_id)
        user_content = await _candidate_user_content(question, db_id, rule_triggers, slot)
        recorder.message("system", system_content)
        recorder.message("user", user_content)
        _log_prompt(verbose, system_content, user_content)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
        ]
        attempt["messages"] = messages
        last_sql_result = None
        last_sql        = None
        last_base_sql   = None   # A1: SQL of the most recent successful run_sql_base
        nudges          = 0      # A2: count of submit nudges issued
        turn            = 0
        all_tool_calls: list[dict] = []   # accumulated tool calls for diagnostics
        sql_actions: list[dict] = []
        protocol_events: list[dict] = []
        guard_tier      = 0      # L1: next convergence-nudge tier to fire
        seen_sql: dict[tuple, int] = {}   # L1: (tool, normalized sql) -> run count
        # Logical clock (turn_memo.RunClock): the three loop reads below use elapsed(), which
        # a replayed turn aligns to the source run's reading so the time nudge fires on the
        # same turn with the same {mins}; duration_sec reads the real wall clock.
        clock = RunClock(time.monotonic)
        memo = _candidate_turn_memo(base_ck, clock, recorder, style, slot, tools=frozenset(known_tools))
        run_usage: dict = {}
        attempt.update(usage=run_usage, tool_calls=all_tool_calls, sql_actions=sql_actions,
                       protocol_events=protocol_events, memo=memo, clock=clock)
        time_nudged     = False
        empty_hits      = 0      # L2: count of empty run_sql results (empty-probe trigger)
        empty_probed    = False
        final_checked   = False  # final-check bounce fired (at most once per run)
        finalcheck_sql  = None   # SQL held at the bounce, for the changed/unchanged marker
        degraded_retrieval = rule_triggers.degraded_retrieval

        while True:
            turn += 1
            if turn > _MAX_TURNS:
                raise RuntimeError(f"Exceeded max turns ({_MAX_TURNS}) without a successful submit")
            if _seat_out_of_time(clock):
                # Out of time: stop asking and salvage below rather than let the candidate
                # cap cut this candidate with no answer at all.
                break
            _log(verbose, "─" * 60)
            _log(verbose, f"[Turn {turn}] → {model}")

            # The turn memo replays a recorded reply when this request's bytes match one
            # (turn_memo.py); otherwise it awaits the provider. `achat_turn` is looked up
            # on the module at call time, so a patched transport stays in charge.
            response = await memo.turn(lambda: achat_turn(llm, messages, tools=tools), messages)
            merge_usage(run_usage, response.usage)
            messages.append(response.message)
            native_calls = list(response.tool_calls)
            # Native transport: SQL rides tool calls, so the assistant text is pure prose —
            # exactly what the rule-trigger matcher wants (no SQL body to exclude first).
            triggered_block = await rule_triggers.match(response.text or "")
            degraded_retrieval |= rule_triggers.degraded_retrieval
            if response.text:
                _trace(verbose, "◀", response.text)
            for tc in native_calls:
                function = tc.get("function") or {}
                _trace_tool_call(
                    verbose,
                    str(function.get("name") or ""),
                    str(function.get("arguments") or "{}"),
                )

            if native_calls:
                any_knowledge = False
                for tc in native_calls:
                    function = tc.get("function") or {}
                    raw_name = str(function.get("name") or "")
                    inputs = _parse_tool_args(
                        raw_name,
                        str(function.get("arguments") or "{}"),
                    )
                    tool_name = _normalize_tool_name(raw_name, known_tools)
                    call_id = str(tc.get("id") or "")
                    if tool_name not in native_tools:
                        # Same guard as the anthropic loop: the server still registers
                        # tools this route disabled — answer per-call, never reach MCP.
                        result: object = {"error": f"Unknown tool: {tool_name}"}
                        all_tool_calls.append({"name": tool_name, "args": inputs})
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": _tool_result_content(
                                verbose, tool_name, result, inputs,
                            ),
                        })
                        continue
                    if last_sql_result is not None:
                        # A submit in this same batch already finalized the run. Later
                        # calls stay unexecuted, but every tool_call still gets a reply
                        # so the transcript honors the provider contract.
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": "final SQL already accepted this turn; "
                                       "call not executed",
                        })
                        continue
                    inputs, result = await _dispatch_capability(
                        session,
                        hooks,
                        rule_triggers,
                        tool_name,
                        inputs,
                        db_id,
                    )
                    if tool_name not in SQL_ACTIONS:
                        any_knowledge = True
                        if degraded_knowledge_result(tool_name, result):
                            degraded_retrieval = True
                        all_tool_calls.append({"name": tool_name, "args": inputs})
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": _tool_result_content(
                                verbose, tool_name, result, inputs,
                            ),
                        })
                        continue
                    # SQL action semantics — text-era logic verbatim, only the envelope
                    # moved (a role:"tool" reply instead of a user message).
                    if tool_name in {"run_sql", "run_sql_base"} and isinstance(result, dict):
                        if _LOOP_GUARD and isinstance(inputs, dict) and isinstance(inputs.get("sql") or "", str):
                            key = (
                                tool_name,
                                " ".join((inputs.get("sql") or "").split()).lower(),
                            )
                            seen_sql[key] = seen_sql.get(key, 0) + 1
                            if seen_sql[key] >= 2:
                                result = {**result, "_note": _PROMPTS.surface(
                                    "candidate", "note.repeat_query", n=seen_sql[key])}
                        if tool_name == "run_sql_base" and "error" not in result:
                            last_base_sql = inputs.get("sql", "") or last_base_sql
                            record_salvage_sql(db_id, question, last_base_sql)
                        rows = result.get("rows", [])
                        if _EMPTY_PROBE and not empty_probed and not rows and "error" not in result:
                            empty_hits += 1
                            if empty_hits >= _EMPTY_PROBE_AT:
                                empty_probed = True
                                result = {**result, "_note": _PROMPTS.surface(
                                    "candidate", "note.empty", n=empty_hits)}
                    if (not final_checked
                            and tool_name == "submit_final_sql"
                            and isinstance(result, dict) and "error" not in result
                            and not time_nudged
                            and not (_TIME_NUDGE_SEC
                                     and clock.elapsed() > _TIME_NUDGE_SEC)
                            and guard_tier < len(_LOOP_GUARD_TIERS)):
                        # Transactional bounce: the SQL just EXECUTED against the base tables
                        # (identical conditions to run_sql_base), so register it for A1 salvage
                        # and hand back the checklist instead of finalizing — a model that never
                        # resubmits still ends with this verified SQL submitted. An errored
                        # submit takes the normal error path below and does not consume the
                        # bounce; past the time nudge or the last convergence tier the bounce
                        # is skipped entirely (those sessions are being told to submit NOW).
                        final_checked = True
                        finalcheck_sql = result.get("_final_sql") or inputs.get("sql", "")
                        last_base_sql = finalcheck_sql
                        record_salvage_sql(db_id, question, last_base_sql)
                        sql_actions.append({
                            "name": tool_name,
                            "args": inputs,
                            "status": "final_check",
                            "audit_status": result.get("audit_status"),
                        })
                        protocol_events.append({
                            "turn": turn,
                            "status": "final_check",
                            "action": tool_name,
                        })
                        rank_missing = (_PROMPTS.has("candidate.nudge.rank_output")
                                        and missing_rank_output(question, finalcheck_sql, result.get("columns", [])))
                        protocol_events[-1]["rank_output_missing"] = rank_missing
                        final_check_note = _PROMPTS.surface("candidate", "note.final_check")
                        if rank_missing:
                            final_check_note += "\n" + _PROMPTS.surface("candidate", "nudge.rank_output")
                        shape_note, shape_record = await _result_shape_note(
                            session, db_id, finalcheck_sql, result,
                            probe_allowed=not (_TIME_NUDGE_SEC and clock.elapsed() > _TIME_NUDGE_SEC - _SHAPE_PROBE_MARGIN_SEC))
                        if shape_record is not None:
                            protocol_events[-1]["result_shape"] = shape_record
                        if shape_note:
                            final_check_note += "\n" + shape_note
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": final_check_note,
                        })
                        _trace(verbose, "▶", final_check_note)
                        continue
                    if (tool_name == "submit_final_sql"
                            and isinstance(result, dict) and "error" not in result):
                        result = dict(result)
                        last_sql = result.pop("_final_sql", inputs.get("sql", ""))
                        last_sql_result = result
                    sql_actions.append({
                        "name": tool_name,
                        "args": inputs,
                        "status": "error"
                        if isinstance(result, dict) and "error" in result
                        else "ok",
                        "audit_status": result.get("audit_status")
                        if isinstance(result, dict) else None,
                    })
                    protocol_events.append({
                        "turn": turn,
                        "status": "ok",
                        "action": tool_name,
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _action_result_content(
                            verbose, tool_name, result, inputs,
                        ),
                    })
                if any_knowledge:
                    protocol_events.append({"turn": turn, "status": "native"})
                if triggered_block and last_sql_result is None:
                    # Same run-ending guard as the two-stage loop above.
                    messages.append({"role": "user", "content": triggered_block})
                    _trace(verbose, "▶", triggered_block)
                interaction_count = len(all_tool_calls) + len(sql_actions)
                if (last_sql_result is None and _LOOP_GUARD
                        and guard_tier < len(_LOOP_GUARD_TIERS)
                        and interaction_count >= _LOOP_GUARD_TIERS[guard_tier]):
                    converge_content = _PROMPTS.surface("candidate", "nudge.converge", n=interaction_count)
                    messages.append({"role": "user", "content": converge_content})
                    _trace(verbose, "▶", converge_content)
                    guard_tier += 1
                if (_TIME_NUDGE_SEC and not time_nudged and last_sql_result is None
                        and clock.elapsed() > _TIME_NUDGE_SEC):
                    time_nudged = True
                    _nmin = int(clock.elapsed() // 60)
                    time_content = _PROMPTS.surface("candidate", "nudge.time", mins=_nmin)
                    messages.append({"role": "user", "content": time_content})
                    _trace(verbose, "▶", time_content)
                if last_sql_result is not None:
                    break
                if slow_sql_cap_reached():
                    # Same verdict as the claude loop: rewrite attempts spent, stop asking,
                    # let A1 salvage end the run on SQL that passed the 30s budget or on none.
                    break
                continue


            if not native_calls:
                protocol_events.append({"turn": turn, "status": "no_action"})
                if hooks is not None:
                    correction = await hooks.on_stop(last_base_sql)
                    if correction is not None:
                        last_base_sql = None
                        correction_content = correction["error"] + triggered_block
                        messages.append({
                            "role": "user",
                            "content": correction_content,
                        })
                        _trace(verbose, "▶", correction_content)
                        continue
                # A2: stopped without submitting. Salvage run_sql_base SQL if any (A1,
                # after the loop); else nudge it to submit, up to _MAX_SUBMIT_NUDGES.
                if last_sql_result is None and last_base_sql is None and nudges < _MAX_SUBMIT_NUDGES:
                    nudges += 1
                    nudge_content = _PROMPTS.surface("candidate", "nudge.submit") + triggered_block
                    messages.append({
                        "role": "user",
                        "content": nudge_content,
                    })
                    _trace(verbose, "▶", nudge_content)
                    continue
                break

        last_sql_result, last_sql = await _maybe_salvage(
            session, db_id, last_sql_result, last_sql, last_base_sql, sql_actions, verbose)
        if last_sql_result is None:
            raise RuntimeError("Model did not call submit_final_sql or all attempts failed")

        result = {**last_sql_result, "sql": last_sql,
                  "cache_key": ck, "memo_base_key": base_ck,
                  "implementation_context": attempt["implementation_context"],
                  "tool_calls": all_tool_calls,
                  "sql_actions": sql_actions,
                  "protocol_events": protocol_events,
                  "usage": run_usage,
                  "duration_sec": round(clock.real_elapsed(), 3),
                  "final_check": {
                      "bounced": final_checked,
                      "changed": bool(final_checked and last_sql != finalcheck_sql),
                  },
                  "messages": messages,
                  "reasoning": memo.thinks(),
                  # Present only when set: the row-level flag eval_bird copies out, and
                  # deliberately part of the cached record — a replayed degraded run is
                  # still a degraded answer.
                  **({"degraded_retrieval": True} if degraded_retrieval else {})}
        if hooks is not None:
            result["route_debug"] = hooks.route_debug(style_block)
        result = publish_rank_result(question, result, [], enabled=False)
        # The render identity travels WITH the cached value: a replay reads the record made
        # the moment this run was written, never the release of the process replaying it.
        recorder.set_replay(**memo.replay_fields(turn))
        result["prompt_identity"] = recorder.snapshot()
        # The index rows describe the row just written: one per assistant message, keyed by
        # the request's hash chain, so a later run under a changed release can replay them.
        result["execution"] = attempt["execution"]
        result["transcript_status"] = "complete"
        result["timing"] = session_timing.current_snapshot()
        attempt["completed_record"] = result
        source = cache_lineage.retain_source(_TURN_STORE, result)
        if _cache_set(ck, result):
            memo.write_candidate_index(ck, messages, **({"source": source} if source is not None else {}))
        return result


# ── Async entrypoints ─────────────────────────────────────────────────────────

async def _run_with_session(question: str, db_id: str, session: ClientSession,
                             provider: str, model: str, verbose: bool,
                             seed: int | None = None,
                             disabled_tools: set[str] | None = None,
                             temperature: float | None = None,
                             style: str | None = None,
                             slot: int | None = None) -> dict:
    """Run one nl2sql call on an already-initialised MCP session.

    Each run gets a fresh session_id (its own server-side session file for intermediate
    tables — isolated from sibling candidates sharing the db); the file is dropped in
    finally. ``slot`` is the candidate's position in a vote's pool (1-based, in the order of
    the candidates spec): a slot whose own document the release carries (``candidate.s<k>``)
    renders it, and only that slot's cache keys move; ``None`` runs the shared document."""
    with session_timing.scope("candidate") as timing:
        sid = uuid.uuid4().hex
        session_token = _SESSION_ID.set(sid)
        # Each task owns both its prompt recorder and live timing collector.
        recorder = _recorder_for("candidate", style, slot)
        recorder_token = recorder.bind()
        hint_token = _QUESTION_HINTS.set(question_hints(question))
        question_token = _ORIGINAL_QUESTION.set(question)
        verified_token = _VERIFIED_BASE.set(None)
        slow_token = _SLOW_SQL_REJECTS.set([0])
        result = None
        error = None
        try:
            if provider == "claude":
                # Accepted-and-discarded arguments are worse than unsupported ones: a route
                # declaring ("claude", model, {"t": 0.7}) would silently run at the provider
                # default. Anthropic has no seed parameter at all.
                if temperature is not None or seed is not None:
                    raise ValueError("provider='claude' does not support seed/temperature; drop "
                                     "them from the sub-agent spec or use an openai-compat provider")
                result = await _run_claude(question, model, db_id, session, verbose,
                                           disabled_tools=disabled_tools, style=style, slot=slot)
            else:
                result = await _run_openai_compat(
                    question, model, provider, db_id, session, verbose,
                    seed=seed, disabled_tools=disabled_tools,
                    temperature=temperature, style=style, slot=slot)
            return result
        except (Exception, asyncio.CancelledError) as exc:
            error = exc
            raise
        finally:
            try:
                # A seat the candidate cap cut must not hold the judge phase behind its
                # in-flight tool: the server closes the session only after that tool.
                if not _detach_close_session(session, sid):
                    with session_timing.measure("tool", "close_session"):
                        await session.call_tool("close_session", {"session_id": sid})
            except Exception:
                pass
            finally:
                timing.finish()
                if isinstance(result, dict):
                    result["timing"] = timing.snapshot()
                if error is not None:
                    record = getattr(error, "candidate_record", None)
                    if isinstance(record, dict):
                        record["timing"] = timing.snapshot()
                    else:
                        # Timing alone must not replace AttemptCapture's transcript fallback.
                        error.candidate_timing = timing.snapshot()
                _VERIFIED_BASE.reset(verified_token)
                _SLOW_SQL_REJECTS.reset(slow_token)
                _QUESTION_HINTS.reset(hint_token)
                _ORIGINAL_QUESTION.reset(question_token)
                recorder.unbind(recorder_token)
                _SESSION_ID.reset(session_token)


async def _run_one(question: str, db_id: str, session: ClientSession,
                   provider: str, model: str, verbose: bool,
                   disabled_tools: set[str] | None = None) -> dict:
    """One single-agent question on an already-initialised (shared) session. _run_with_session
    gives this run its own session_id (a private server-side session file for any intermediate
    tables it saves) and drops that file in its finally (close_session) — per-run cleanup."""
    return await _run_with_session(question, db_id, session, provider, model, verbose,
                                   disabled_tools=disabled_tools)


def _exc_to_result(r: object) -> dict:
    """Normalise a gather() outcome to a result dict. For an exception, record a non-empty
    error (some dead-session excs stringify to ''), its type and terminal-session
    classification before the exception is lost. MCP codes distinguish connection closure
    from ordinary request failures even when their type and message are identical.
    Contract drift, invalid provider configuration and exhausted accounts abort instead
    of becoming failed candidates."""
    if isinstance(r, (tool_contract.ToolContractDrift, GatewayConfigurationError, GatewayUnavailableError)):
        raise r
    if isinstance(r, dict):
        return r
    evidence = getattr(r, "candidate_record", None)
    from .llm_runtime import provider_failure
    timing = getattr(r, "candidate_timing", None)
    return {**(evidence if isinstance(evidence, dict) else {}),
            **({"timing": timing} if isinstance(timing, dict) else {}),
            "sql": "", "rows": [], "row_count": 0,
            "error": str(r) or type(r).__name__, "_exc_type": type(r).__name__,
            "_dead_session": isinstance(r, BaseException) and is_dead_session_error(r),
            **({"provider_failure": True} if isinstance(r, BaseException) and provider_failure(r) else {}),
            **({"_mcp_error_code": r.code} if isinstance(r, MCPError) else {}),
            **({"shared_session_timing": r.shared_session_timing} if hasattr(r, "shared_session_timing") else {})}


#: Error text of a candidate `_gather_candidates` stopped at the candidate cap.
CANDIDATE_CUT_ERROR = "candidate stopped at the candidate budget"


def _candidate_completed(task: asyncio.Task) -> bool:
    """Whether a finished candidate task produced a usable answer (SQL, no error)."""
    if not task.done() or task.cancelled() or task.exception() is not None:
        return False
    result = task.result()
    return isinstance(result, dict) and not result.get("error") and bool((result.get("sql") or "").strip())


async def _gather_candidates(coros: list, cap_at: float | None = None, *,
                             slots: list[int | None] | None = None, topup: bool = False) -> list:
    """Run candidate coroutines concurrently until all finish or the candidate cap; results in input order.

    Without ``cap_at`` every candidate runs to its end, as ``asyncio.gather(...,
    return_exceptions=True)`` would. With it, the candidates still running at monotonic time
    ``cap_at`` (the vote's candidate deadline, `phase_budget.VotePhases.candidate_deadline`)
    are cancelled and reported as `CANDIDATE_CUT_ERROR`, and the judge works with the ones
    that finished. The cut ends the candidate phase at the cut time, and the cut seats hand
    their session cleanup to the loop (`_detach_close_session`) instead of waiting behind an
    in-flight tool, so the judge phase starts at once. A cancelled gather (the vote stopped)
    marks every running candidate stopped, cancels it and waits for its unwinding, its
    cleanup detached the same way. Each seat's start, end and outcome (``done``, ``failed``,
    ``cut``, or ``in_flight`` when the vote stopped it) go into the bound vote's phase record.

    Args:
        coros (list): Candidate coroutines, in pool order.
        cap_at (float | None): Monotonic time of the candidate cap, or ``None``.
        slots (list[int | None] | None): Each coroutine's pool slot, for the phase record.
        topup (bool): Whether this gather is a quorum top-up, for the phase record.

    Returns:
        list: Each candidate's result dict or exception, in pool order.
    """
    phases = phase_budget.current()
    labels = slots if slots is not None else [index + 1 for index in range(len(coros))]
    seats = ([phases.open("seats", slot=slot, round="topup" if topup else "first") for slot in labels]
             if phases is not None else [])
    stopped = threading.Event()   # set at the cut or the vote's stop; the seats' stop check
    # Bound before the tasks exist: each task copies this context when it is created.
    with bind_stop_check(stopped.is_set):
        tasks = [asyncio.ensure_future(coro) for coro in coros]
    if not tasks:
        return []
    for task, seat in zip(tasks, seats):
        task.add_done_callback(lambda t, seat=seat: phases.close(
            seat, "in_flight" if t.cancelled() else "done" if _candidate_completed(t) else "failed",
            at=phases.stopped_at() if t.cancelled() else None))
    cut: set[asyncio.Task] = set()
    try:
        # asyncio.wait, unlike gather, leaves the seats running when the vote stops this wait:
        # the handler below marks them stopped first, so they detach their cleanup.
        _done, pending = await asyncio.wait(
            tasks, timeout=None if cap_at is None else max(0.0, cap_at - time.monotonic()))
        cut = set(pending)
        cut_at = time.monotonic()
        for task, seat in zip(tasks, seats):
            if task in cut:
                phases.close(seat, "cut", at=cut_at)
        if cut:
            phase_budget.end_candidates(at=cut_at)   # the judge clock starts at the cut, not after cleanup
            stopped.set()
        for task in cut:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    except BaseException:
        stop_at = phases.stopped_at() if phases is not None else None
        for task, seat in zip(tasks, seats):
            if not task.done():
                phases.close(seat, "in_flight", at=stop_at)   # the vote stopped it, not a failure
        stopped.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    if cut:
        deadline_events.note("candidate_cap_cut")
        deadline_events.note("candidate_seats_cut", len(cut))
        print(f"{_LOG_META_PREFIX} [candidate-cap] {len(tasks) - len(cut)} candidate(s) finished; "
              f"stopped {len(cut)} still running at the candidate budget",
              file=sys.stderr, flush=True)
    results: list = []
    for task in tasks:
        if task in cut:
            results.append(asyncio.CancelledError(CANDIDATE_CUT_ERROR))
        elif task.cancelled():
            results.append(asyncio.CancelledError())
        else:
            results.append(task.exception() or task.result())
    return results


async def _gather_in_session(questions: list[str], db_id: str,
                             provider: str, model: str, verbose: bool,
                             session: ClientSession,
                             seeds: list[int | None] | None = None,
                             disabled_tools: set[str] | None = None,
                             slots: list[int | None] | None = None,
                             cap_at: float | None = None, topup: bool = False) -> list[dict]:
    """Run N questions concurrently on an already-initialised MCP session.

    ``slots`` gives each run its position in a vote's candidate pool (see
    `_run_with_session`); ``None`` runs every question on the shared candidate document.
    ``cap_at`` and ``topup`` are `_gather_candidates`'s."""
    if seeds is None:
        seeds = [None] * len(questions)
    if slots is None:
        slots = [None] * len(questions)
    results = await _gather_candidates(
        [_run_with_session(q, db_id, session, provider, model, verbose,
                           seed=s, disabled_tools=disabled_tools, slot=k)
         for q, s, k in zip(questions, seeds, slots)],
        cap_at, slots=slots if any(k is not None for k in slots) else None, topup=topup,
    )
    return [_exc_to_result(r) for r in results]


async def _gather_mixed_in_session(question: str, db_id: str,
                                    sub_agents: list[tuple[str, str]],
                                    verbose: bool,
                                    session: ClientSession,
                                    seeds: list[int | None] | None = None,
                                    disabled_tools: set[str] | None = None,
                                    slots: list[int | None] | None = None,
                                    cap_at: float | None = None, topup: bool = False) -> list[dict]:
    """Run sub-agents concurrently with mixed provider/model assignments.

    ``slots`` gives each sub-agent its position in the pool (1-based, spec order) by default;
    a top-up of failed members passes their original positions. ``cap_at`` and ``topup`` are
    `_gather_candidates`'s."""
    n = len(sub_agents)
    if seeds is None:
        seeds = list(range(1, n + 1))
    if slots is None:
        slots = list(range(1, n + 1))
    def _route(spec):
        # (provider, model) legacy pairs and (provider, model, opts) HET1 triples both accepted
        opts = spec[2] if len(spec) > 2 else {}
        return spec[0], spec[1], opts.get("t"), opts.get("style")
    results = await _gather_candidates(
        [_run_with_session(question, db_id, session, p, m, verbose,
                           seed=s, disabled_tools=disabled_tools,
                           temperature=t, style=st, slot=k)
         for spec, s, k in zip(sub_agents, seeds, slots)
         for p, m, t, st in [_route(spec)]],
        cap_at, slots=slots, topup=topup,
    )
    return [_exc_to_result(r) for r in results]


# ── Public API ────────────────────────────────────────────────────────────────


def nl2sql(question: str, db_id: str,
           provider: str = "deepseek", model: str | None = None,
           verbose: bool = False,
           disabled_tools: set[str] | None = None,
           hint: str | list[str] | tuple[str, ...] | None = None) -> dict:
    """Translate a natural-language question to SQL and execute it.

    Tools are fetched from and executed via the rubiksql_hb.server MCP module over a shared
    per-db session, so repeated calls for one db_id reuse a single server subprocess.

    Args:
        question (str): Natural-language question.
        db_id (str): Registered database name (e.g. ``'customers_demo'``).
        provider (str): One of ``"deepseek"`` (default), ``"qwen"``, ``"claude"``,
            ``"deepseek"``.
        model (str | None): Override the default model for the chosen provider; None uses
            the provider's ``default_model``.
        verbose (bool): Mirror complete model-facing payloads with ``▶``/``◀`` and print
            display-only diagnostics under the explicit ``[LOG ONLY]`` prefix.
        disabled_tools (set[str] | None): Optional tool names to hide from the LLM, e.g.
            ``{"search_values"}``. The required tools (get_rules, get_examples,
            get_db_info, submit_final_sql) cannot be disabled.
        hint: Optional evidence/preference text. It is canonically appended to ``question``
            and inherited by model-issued rule/example retrieval queries.

    Returns:
        dict: The final SQL execution result — ``columns``, ``rows``, ``row_count``,
        ``final`` — merged with ``sql`` (the SQL that ran), ``tool_calls`` (normalized native
        knowledge/schema calls), ``sql_actions`` (plain-text SQL actions), and ``messages``
        (the entire real model transcript, which is large). ``route_debug`` is present when
        a route hook is active, ``_salvaged`` when the answer came from the salvage path
        rather than an explicit submit action.

    Raises:
        ValueError: ``provider`` is not a registered provider.
        RuntimeError: the provider's API key is missing, the model exhausted the turn cap,
            or the loop ended with no submitted and no salvageable SQL.
    """
    cfg = _PROVIDERS.get(provider)
    if cfg is None:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_PROVIDERS)}")
    model = model or cfg["default_model"]
    question = question_with_hint(question, hint)
    # Use the per-db shared MCP session (same as vote mode): the question's server slot of
    # db_id, reused across concurrent questions. Retry once on ANY dead-session error (a closed/
    # broken stream or EOF from a crashed subprocess): `run` marked that session dead, and
    # retiring only dead sessions never stops a replacement another question already started.
    def _run() -> dict:
        while True:
            shared = _get_shared_session(db_id)
            try:
                return shared.run(_run_one(question, db_id, shared.session, provider, model, verbose,
                                           disabled_tools=disabled_tools))
            except _SessionSubmissionRejected:
                continue

    with server_scope(server_slot(question)):
        try:
            return _run()
        except _DEAD_SESSION_EXCS:
            retire_dead_session(db_id)
            return _run()


def nl2sql_batch(questions: list[str], db_id: str,
                 provider: str = "deepseek", model: str | None = None,
                 verbose: bool = False,
                 seeds: list[int | None] | None = None,
                 disabled_tools: set[str] | None = None,
                 hints: list[str | list[str] | tuple[str, ...] | None] | None = None) -> list[dict]:
    """Run multiple questions concurrently, sharing one MCP server session.

    Eliminates per-call MCP startup overhead and SQLite lock contention.

    Args:
        questions (list[str]): Natural-language questions to run in parallel.
        db_id (str): Registered database name.
        provider (str): LLM provider; see :func:`nl2sql` for the accepted values.
        model (str | None): Override the default model for the chosen provider.
        verbose (bool): Mirror complete model-facing payloads with ``▶``/``◀`` and print
            display-only diagnostics under an explicit ``不在模型上下文`` prefix.
        seeds (list[int | None] | None): One seed per question, for reproducibility.
        disabled_tools (set[str] | None): Optional tool names to hide from the LLM
            (see :func:`nl2sql`).
        hints: Optional hint value per question. Its length must match ``questions``.

    Returns:
        list[dict]: One entry per question, in the order given. Each is the same dict
        :func:`nl2sql` returns, or ``{"error": ...}`` for a question that failed — a failing
        question aborts its siblings on tool-contract drift, inconsistent provider
        configuration, or an unavailable provider account pool.

    Raises:
        ValueError: ``provider`` is not a registered provider.
        tool_contract.ToolContractDrift: A served tool contract differs from its declarations.
        GatewayConfigurationError: The approved provider configuration or served model differs.
        GatewayUnavailableError: No provider account remains available.
    """
    cfg = _PROVIDERS.get(provider)
    if cfg is None:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_PROVIDERS)}")
    model = model or cfg["default_model"]
    if hints is not None and len(hints) != len(questions):
        raise ValueError("hints must have the same length as questions")
    questions = [
        question_with_hint(question, hints[index] if hints is not None else None)
        for index, question in enumerate(questions)
    ]
    # Use the per-db shared MCP session (same as nl2sql / nl2sql_vote): one server slot of
    # db_id, chosen from the whole batch, reused across all calls. The N questions run
    # concurrently on that one session. Retry once on a dead session.
    used: list[_SharedSession] = []

    def _run() -> list[dict]:
        while True:
            shared = _get_shared_session(db_id)
            used.append(shared)
            try:
                return shared.run(_gather_in_session(
                    questions, db_id, provider, model, verbose, shared.session,
                    seeds=seeds, disabled_tools=disabled_tools))
            except _SessionSubmissionRejected:
                continue
    with server_scope(server_slot("\n".join(questions))):
        try:
            results = _run()
        except _DEAD_SESSION_EXCS:
            # Session died before/at scheduling (`run` marked it dead) — rebuild and retry the batch.
            retire_dead_session(db_id)
            return _run()
        # _gather_in_session swallows per-question exceptions (return_exceptions), so a session
        # that died mid-gather surfaces only as dead-session _exc_type tags. Mark the session this
        # batch used dead, retire it, and retry the batch once on a fresh session
        # (already-succeeded questions hit the LLM cache).
        if any(r.get("_exc_type") in _DEAD_SESSION_EXC_NAMES for r in results):
            used[-1].dead = True
            retire_dead_session(db_id)
            results = _run()
        return results


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "claude"
    print(f"Provider: {provider}\n")
    for q in ["有哪些商品", "有多少 VIP 客户", "总收入"]:
        print(f"Q: {q}")
        r = nl2sql(q, "customers_demo", provider=provider, verbose=True)
        print("最终结果:", json.dumps(r, ensure_ascii=False, indent=2), "\n")
