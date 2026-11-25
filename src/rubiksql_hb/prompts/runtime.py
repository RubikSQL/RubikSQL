"""Release loading with full integrity checks, pin resolution, surfaces, keyspaces and render identity.

A process resolves ``RUBIKSQL_PROMPT_RELEASE`` once (``current()``), verifies every row of
the release against its content key and its regenerated capsule source, pre-parses every
spec and document, and then renders surfaces (``Release.surface``) and entries
(``Release.text``) with memoisation. Nothing here opens a HeavenBase workspace or executes a
capsule: the verified rows are replayed by the stdlib interpreter.

Prompt identity is the static text of the surfaces: a keyspace hashes, for every surface of its
closure, the text as written — a text row's ``meta.text``, a document's message, a composition's
normal form printed as a document (:func:`doc.document_of`), the code's fallback document of a
surface no row provides — so two releases that cut the same texts into different rows share
every keyspace (:func:`same_prompts`). A candidate slot with a document of its own
(``candidate.s<k>``) has a keyspace of its own, the candidate closure with that document's texts, and a
knob of ``config.knobs`` whose value differs from its default joins the keyspaces it declares.

Two release formats load (:data:`RELEASE_FORMATS`). A ``rubiksql.closure/1`` release carries its knobs, rule
corpus and tools as prompt carrier rows. A ``rubiksql.closure/2`` release carries native prompt rows only and
references exact Tool, rule and parameter-set versions (:mod:`entity_model`); its knobs and corpus come from
those verified rows, never from code defaults, and its data surfaces keep the exact legacy texts, so a
representation-only migration changes no model input, keyspace or surface identity.
"""
from __future__ import annotations

import contextvars
from copy import deepcopy
import json
import os
import threading
from collections import OrderedDict
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from . import entity_model as M
from . import entries as E
from .canon import SCHEMA_PROMPT, canon, content_key, escape, sha256_hex, template_code
from .doc import DocError, Document, compile_document, document_of, parse_document
from .policy import KINDS, OUTPUT_MAX_CHARS, PARAM_MAX_CHARS, PARTS_MAX_DEPTH, PROPOSAL_RE, RELEASE_RE, PolicyError
from .policy import check_content_key, check_document, check_identifier, check_meta_keys, check_name, check_text
from .spec import (
    INT_MAX_CHARS,
    Composer,
    RenderContext,
    Span,
    SpecError,
    TextLeaf,
    check_params,
    check_type,
    compile_spec,
    parse_spec,
    spans_of,
)
from .store import (
    ENV_STORE,
    SCHEMA_OBJECT,
    SCHEMA_PROPOSAL,
    SCHEMA_RELEASE,
    BaselinePointer,
    PromptStoreError,
    Store,
    check_env_file,
    proposal_id,
    release_document_id,
)

__all__ = [
    "ENV_ALLOW_RETIRED",
    "ENV_CACHE_NS",
    "ENV_RELEASE",
    "KNOWN_RULES",
    "MEMO_SIZE",
    "RELEASE_FORMATS",
    "SCHEMA_RENDER",
    "Release",
    "RenderRecorder",
    "Span",
    "baseline_pointer",
    "current",
    "load_release",
    "reset_current",
    "resolve_baseline",
    "same_prompts",
]

ENV_RELEASE = "RUBIKSQL_PROMPT_RELEASE"
ENV_ALLOW_RETIRED = "RUBIKSQL_PROMPT_ALLOW_RETIRED"
ENV_CACHE_NS = "RUBIKSQL_CACHE_NS"
SCHEMA_RENDER = "rubiksql.render/1"
#: Closure rule versions this runtime can compute.
KNOWN_RULES = (E.CLOSURE_RULES, M.CLOSURE_V2)
#: The release file schema each closure rules version is published under; any other pair is refused.
RELEASE_FORMATS: Mapping[str, str] = MappingProxyType({E.CLOSURE_RULES: SCHEMA_RELEASE,
                                                       M.CLOSURE_V2: M.SCHEMA_RELEASE_V2})
#: Render memo entries kept per release.
MEMO_SIZE = 512

_LAYER_KEYS = frozenset({"type", "priority", "source", "source_hash", "qualname", "filename", "global_refs"})
_TRUE_VALUES = ("1", "true", "yes")


def _freeze(value: Any) -> Any:
    """Hashable memo key of a render argument; the type travels with the value so ``1``, ``True`` and ``1.0`` never share an entry."""
    if isinstance(value, Mapping):
        return ("map", tuple(sorted((key, _freeze(item)) for key, item in value.items())))
    if isinstance(value, (list, tuple)):
        return ("list", tuple(_freeze(item) for item in value))
    return (type(value).__name__, value)


def _args_digest(params: Mapping[str, Any]) -> str:
    return sha256_hex(canon(params))[:16]


# ---------------------------------------------------------------------------------------
# Render identity


_RECORDER: contextvars.ContextVar["RenderRecorder | None"] = contextvars.ContextVar("rubiksql_prompt_recorder", default=None)


class RenderRecorder:
    """Records which entries one agent run rendered, for ``prompt_identity`` snapshots.

    Bind one recorder per candidate, judge or reviewer run on the task's context; every
    ``Release.surface`` and ``Release.text`` call inside that context is counted under the
    entry that actually rendered (a role document once per message it rendered, a code
    fallback not at all) with a digest of its arguments.
    """

    def __init__(self, release: str, proposal: str | None, role: str, route: str | None, keyspace: str) -> None:
        """Create an unbound recorder.

        Args:
            release (str): Release id being rendered.
            proposal (str | None): Proposal id when the pin is a proposal.
            role (str): ``candidate``, ``judge`` or ``reviewer``.
            route (str | None): Route name or ``None``.
            keyspace (str): The 16-hex keyspace of that role/route under this release.
        """
        self.release = release
        self.proposal = proposal
        self.role = role
        self.route = route
        self.keyspace = keyspace
        self.fired: dict[str, dict[str, Any]] = {}
        self.msg: dict[str, str | None] = {"system": None, "user": None}
        self.replay: dict[str, Any] = {"turns": 0, "replayed": 0, "diverged_at": None, "cause": None, "source": None}

    @staticmethod
    def current() -> "RenderRecorder | None":
        """Return the recorder bound to the current context, if any.

        Args:
            None.

        Returns:
            RenderRecorder | None: The bound recorder or ``None``.
        """
        return _RECORDER.get()

    def bind(self) -> contextvars.Token:
        """Bind this recorder to the current context.

        Args:
            None.

        Returns:
            contextvars.Token: Pass it to :meth:`unbind` to restore the previous binding.
        """
        return _RECORDER.set(self)

    def unbind(self, token: contextvars.Token) -> None:
        """Restore the binding that preceded :meth:`bind`.

        Args:
            token (contextvars.Token): The token :meth:`bind` returned.

        Returns:
            None: This method does not return a value.
        """
        _RECORDER.reset(token)

    def __enter__(self) -> "RenderRecorder":
        self._token = self.bind()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.unbind(self._token)

    def record(self, entry: str, params: Mapping[str, Any]) -> None:
        """Count one render of an entry.

        Args:
            entry (str): Entry name.
            params (Mapping[str, Any]): Render arguments; only their digest is kept.

        Returns:
            None: This method does not return a value.
        """
        digest = _args_digest(params)
        slot = self.fired.setdefault(entry, {"n": 0, "args": []})
        slot["n"] += 1
        if digest not in slot["args"]:
            slot["args"].append(digest)

    def message(self, kind: str, text: str) -> None:
        """Record the digest of a whole assembled message.

        Args:
            kind (str): ``"system"`` or ``"user"``.
            text (str): The message bytes as sent.

        Returns:
            None: This method does not return a value.

        Raises:
            ValueError: If ``kind`` is not ``system`` or ``user``.
        """
        if kind not in self.msg:
            raise ValueError(f"message kind must be 'system' or 'user', got {kind!r}")
        self.msg[kind] = sha256_hex(text)[:16]

    def set_replay(self, **fields: Any) -> None:
        """Update replay counters, divergence, and optional per-turn immutable ``sources``.

        Args:
            **fields (Any): Replay fields to overwrite.

        Returns:
            None: This method does not return a value.

        Raises:
            KeyError: If a field is neither a replay counter/source nor ``sources``.
        """
        for key, value in fields.items():
            if key not in self.replay and key != "sources":
                raise KeyError(f"unknown replay field {key!r}; expected one of {sorted(self.replay)}")
            self.replay[key] = deepcopy(value) if key == "sources" else value

    def snapshot(self) -> dict[str, Any]:
        """Serialize the recorder as a ``rubiksql.render/1`` document.

        Args:
            None.

        Returns:
            dict[str, Any]: ``{"schema", "release", "proposal", "role", "route", "keyspace", "fired", "msg", "replay"}``.
        """
        return {
            "schema": SCHEMA_RENDER,
            "release": self.release,
            "proposal": self.proposal,
            "role": self.role,
            "route": self.route,
            "keyspace": self.keyspace,
            "fired": {entry: {"n": data["n"], "args": list(data["args"])} for entry, data in sorted(self.fired.items())},
            "msg": dict(self.msg),
            "replay": deepcopy(self.replay),
        }


# ---------------------------------------------------------------------------------------
# Release


class Release:
    """One verified, immutable release: its rows pre-parsed, its surfaces and entries renderable by name.

    Attributes:
        id (str): Release id.
        proposal (str | None): Proposal id when loaded through a proposal pin.
        refs (Mapping[str, str]): Row name to full content key of every prompt row in the closure: a closure/1
            release's data carriers included, a closure/2 release's native prompt rows only.
        entries (tuple[str, ...]): Entry names the release lists, sorted (its dropped entries included).
        rules (str): Closure rules version the release was published under (:data:`KNOWN_RULES`).
        content_refs (Mapping[str, Any] | None): A closure/2 release's typed references (``rubiksql.release-refs/1``);
            ``None`` for closure/1.
        typed_rows (Mapping[str, Mapping[str, Mapping[str, Any]]]): The verified canonical Tool, rule and
            parameter-set rows those references name, by entity id and object id; empty for closure/1.
        manifest (Mapping[str, Any]): The verified release file body.
        origin (str): ``"package"`` or ``"overlay:<dir>"``.
        resolved_from (str): ``"baseline@<blob sha>"`` (the pin was ``baseline``), ``"env=baseline@<blob sha>"``
            (an explicit pin naming the release ``BASELINE.json`` points at, as a launcher passes ``baseline``
            down to its children), ``"env"`` (any other explicit pin) or ``"explicit"`` (:func:`load_release`).
        allow_retired (bool): Whether the retired check was waived for this load.
        unknown_entries (tuple[str, ...]): Entries the release lists that this code does not know.
        dropped_entries (tuple[str, ...]): Entries the release lists that this code has dropped
            (:data:`entries.DROPPED_ENTRIES`): never rendered, providing no surface and in no keyspace.
        warnings (tuple[str, ...]): Contract warnings raised by surfaces without parameters.
    """

    def __init__(
        self,
        *,
        id: str,
        proposal: str | None,
        refs: Mapping[str, str],
        entries: Iterable[str],
        rules: str,
        origin: str,
        resolved_from: str,
        allow_retired: bool,
        objects: Mapping[str, Mapping[str, Any]],
        renderables: Mapping[str, TextLeaf | Composer],
        unknown_entries: Iterable[str],
        dropped_entries: Iterable[str],
        warnings: Iterable[str],
        providers: Mapping[str, tuple[str, int]],
        identity: Mapping[str, str],
        anchor: "Release | None",
        knobs: Mapping[str, Any] | None = None,
        corpus: str | None = None,
        values: Mapping[str, Any] | None = None,
        content_refs: Mapping[str, Any] | None = None,
        typed_rows: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self.id = id
        self.proposal = proposal
        self.refs: Mapping[str, str] = MappingProxyType(dict(refs))
        self.entries = tuple(sorted(entries))
        self.rules = rules
        self.content_refs: Mapping[str, Any] | None = (None if content_refs is None
                                                       else MappingProxyType(deepcopy(dict(content_refs))))
        self.typed_rows: Mapping[str, Mapping[str, Mapping[str, Any]]] = MappingProxyType(
            {entity: MappingProxyType(dict(rows)) for entity, rows in (typed_rows or {}).items()})
        self.manifest: Mapping[str, Any] = MappingProxyType(deepcopy(dict(manifest or {})))
        self._values = deepcopy(dict(values or {}))
        self.origin = origin
        self.resolved_from = resolved_from
        self.allow_retired = allow_retired
        self.unknown_entries = tuple(sorted(unknown_entries))
        self.dropped_entries = tuple(sorted(dropped_entries))
        self.warnings = tuple(warnings)
        self._objects = dict(objects)
        self._renderables = dict(renderables)
        self._providers = dict(providers)
        self._identity = dict(identity)
        self._fallbacks = {
            sid: TextLeaf(sid, "0" * 16, surface.fallback, list(surface.params))
            for sid, surface in E.SURFACES.items()
            if surface.fallback is not None and sid not in self._providers
        }
        self._knobs: Mapping[str, Any] = MappingProxyType(dict(knobs or {}))
        self._corpus = corpus
        self._keyspaces = {pair: self._keyspace(pair) for pair in E.KEYSPACES}
        self._slot_keyspaces = {
            (route, slot): self._keyspace(("candidate", route), slot)
            for slot in range(1, E.CANDIDATE_SLOTS + 1)
            if f"candidate.s{slot}.system" in self._providers
            for route in E.ROUTES["candidate"]
        }
        self._anchor = anchor
        self._anchor_keyspaces = None if anchor is None else {pair: anchor.keyspace(*pair) for pair in E.KEYSPACES}
        self._memo: OrderedDict[tuple[str, ...], str] = OrderedDict()
        self._lock = threading.Lock()

    def _keyspace(self, pair: tuple[str, str | None], slot: int | None = None) -> str:
        """Hash the identity texts of a keyspace's closure, a slot document's texts in place of the candidate document's, and the moved knobs."""
        texts = {sid: self._identity[sid] for sid in E.closure(*pair) if sid in self._identity}
        if slot is not None:
            texts["candidate.system"] = self._identity[f"candidate.s{slot}.system"]
            texts["candidate.user"] = self._identity[f"candidate.s{slot}.user"]
        for name, value in self._knobs.items():
            knob = E.KNOBS[name]
            if pair in knob.visible and value != knob.normalize(knob.default):
                texts[f"knob.{name}"] = canon(value)
        return sha256_hex(canon(texts))[:16]

    def has(self, entry: str) -> bool:
        """Tell whether the release carries an entry it can render.

        Args:
            entry (str): Entry name.

        Returns:
            bool: ``True`` when ``text(entry, ...)`` can render; ``False`` for a dropped entry, which is listed only.
        """
        return entry in self._renderables and entry in self.entries and entry not in self.dropped_entries

    def _require(self, entry: str) -> None:
        """Refuse an entry the release does not carry or lists as dropped."""
        if entry in self.dropped_entries:
            raise KeyError(f"release {self.id} lists {entry!r} as a dropped entry: it is never rendered (entries.DROPPED_ENTRIES)")
        if not self.has(entry):
            raise KeyError(f"release {self.id} has no entry {entry!r}")

    def has_surface(self, role: str, name: str) -> bool:
        """Tell whether the release provides a surface, through a row or the code's fallback.

        Args:
            role (str): ``candidate``, ``judge``, ``reviewer`` or ``shared``.
            name (str): The rest of the surface id, e.g. ``system``, ``route.grain``, ``synonyms``.

        Returns:
            bool: ``True`` when ``surface(role, name, ...)`` can render.
        """
        sid = f"{role}.{name}"
        return sid in self._providers or sid in self._fallbacks

    def provider(self, surface: str) -> tuple[str, int] | None:
        """The entry that provides a surface and the message it renders.

        Args:
            surface (str): Surface id, e.g. ``judge.user``.

        Returns:
            tuple[str, int] | None: ``(entry name, message index)``; ``None`` when no row provides it (absent, or the
                code's fallback renders it).
        """
        return self._providers.get(surface)

    def surface_identity(self) -> Mapping[str, str]:
        """The identity text of every surface the release provides.

        A text row's surface is its ``meta.text``, a document's surface one message of its ``meta.text``, a
        composition's surface its normal form printed as a document, a surface no row provides the code's
        fallback document (the r0 layout's candidate first user message, ``{question}{synonyms}{rule_hooks}``,
        which is exactly the one-document release's second candidate message). Slot and tag markers stay as
        written, so a release that re-cuts the same texts into other rows has the same identity, while any
        change of text, order or whitespace moves it.

        Args:
            None.

        Returns:
            Mapping[str, str]: Surface id to identity text; the keyspaces hash these.
        """
        return MappingProxyType(self._identity)

    def row(self, name: str) -> Mapping[str, Any]:
        """Return the exported HB row of one release member, as stored.

        Args:
            name (str): Row name.

        Returns:
            Mapping[str, Any]: The ``row`` block of the object file.

        Raises:
            KeyError: If the release has no such row.
        """
        return self._objects[name]["row"]

    def renderable(self, name: str) -> TextLeaf | Composer:
        """Return the pre-parsed executor of one row (entry or part).

        Args:
            name (str): Row name.

        Returns:
            TextLeaf | Composer: The row's executor (a :class:`doc.Document` for a doc row).

        Raises:
            KeyError: If the release has no such row.
        """
        return self._renderables[name]

    def composers(self) -> tuple[Composer, ...]:
        """All spec and doc rows of the release, for coverage obligations.

        Args:
            None.

        Returns:
            tuple[Composer, ...]: Sorted by name.
        """
        return tuple(item for _name, item in sorted(self._renderables.items()) if isinstance(item, Composer))

    def entry_params(self, entry: str) -> Mapping[str, Any]:
        """Declared parameter types of an entry as this release carries it.

        Args:
            entry (str): Entry name.

        Returns:
            Mapping[str, Any]: Parameter name to type expression (known entries follow the code table).

        Raises:
            KeyError: If the release has no such entry.
        """
        if entry not in self.entries:
            raise KeyError(f"release {self.id} has no entry {entry!r}")
        known = E.ENTRIES.get(entry)
        if known is not None:
            return dict(known.params)
        renderable = self._renderables[entry]
        return dict(renderable.params) if isinstance(renderable, Composer) else {slot: "str" for slot in renderable.slots}

    def _render(self, entry: str, params: Mapping[str, Any], ctx: RenderContext) -> list:
        renderable = self._renderables[entry]
        return renderable.render_chunks(ctx, params)

    def _memoised(self, key: tuple[str, ...], render: Any) -> str:
        with self._lock:
            hit = self._memo.get(key)
            if hit is not None:
                self._memo.move_to_end(key)
        if hit is None:
            hit = render()
            with self._lock:
                self._memo[key] = hit
                if len(self._memo) > MEMO_SIZE:
                    self._memo.popitem(last=False)
        return hit

    def surface(self, role: str, name: str, /, **params: Any) -> str:
        """Render one surface and record the entry that rendered it on the bound recorder.

        The producers ask for texts by surface only, so both layouts serve them: under a one-document
        release ``surface("judge", "system")`` renders the first message of the ``judge`` document, under
        ``surface("candidate", "user", ...)`` renders the
        second message of the ``candidate`` document,         ``{question}{synonyms}{rule_hooks}`` (recorded as nothing, since no row rendered it). Identical
        calls are memoised like :meth:`text`; the memo keys carry the argument types.

        Args:
            role (str): ``candidate``, ``judge``, ``reviewer`` or ``shared``.
            name (str): The rest of the surface id: ``system``, ``user``, ``nudge.time``, ``synonyms`` ...
            **params (Any): Arguments matching the surface's declared parameters exactly.

        Returns:
            str: The rendered text of that one message.

        Raises:
            KeyError: If the surface is unknown to the code or the release does not provide it; use
                :meth:`has_surface` for an optional route block.
            RenderError: If the arguments do not match the declared types or the output exceeds the hard cap.
        """
        sid = f"{role}.{name}"
        declared = E.SURFACES.get(sid)
        if declared is None:
            raise KeyError(f"unknown surface {sid!r}; known: {sorted(E.SURFACES)}")
        provider = self._providers.get(sid)
        if provider is None and sid not in self._fallbacks:
            raise KeyError(f"release {self.id} provides no surface {sid!r}")

        def render() -> str:
            check_params(declared.params, params, sid)
            if provider is None:
                executor: TextLeaf | Composer = self._fallbacks[sid]
            else:
                executor = self._renderables[provider[0]]
                if isinstance(executor, Document):
                    executor = executor.messages[provider[1]]
            return "".join(text for text, _src in executor.render_chunks(RenderContext(), params))

        text = self._memoised(("surface", sid, _freeze(params)), render)
        recorder = _RECORDER.get()
        if recorder is not None and provider is not None:
            recorder.record(provider[0], params)
        return text

    def text(self, entry: str, /, **params: Any) -> str:
        """Render an entry whole and record it on the bound recorder.

        A doc entry renders every message, joined by U+001E (:data:`doc.MESSAGE_SEPARATOR`); use
        :meth:`messages` for the list, or :meth:`surface` for one message. Identical ``(entry, params)``
        calls — same argument values and types — return the same ``str`` object while the entry stays in
        the 512-entry LRU memo; per-question renders can evict a static entry, after which a re-render is an
        equal but distinct object. A memo hit skips ``check_params`` only because the key already proved an
        identically typed call passed it. Callers rely on byte equality, not ``is``: the routes identity
        contract holds because routes captures nl2sql's globals after each reload, not because of this memo.

        Args:
            entry (str): Entry name.
            **params (Any): Arguments matching the entry's declared parameters exactly.

        Returns:
            str: The rendered text.

        Raises:
            KeyError: If the release has no such entry or lists it as dropped.
            RenderError: If the arguments do not match the declared types (a string parameter may be any
                length) or the output exceeds the ``OUTPUT_MAX_CHARS`` hard cap.
        """
        self._require(entry)

        def render() -> str:
            check_params(self.entry_params(entry), params, entry)
            return "".join(text for text, _src in self._render(entry, params, RenderContext()))

        text = self._memoised(("entry", entry, _freeze(params)), render)
        recorder = _RECORDER.get()
        if recorder is not None:
            recorder.record(entry, params)
        return text

    def messages(self, entry: str, /, **params: Any) -> list[str]:
        """Render an entry as the messages the model receives: one per ``{{message}}``-separated part of a document.

        Args:
            entry (str): Entry name.
            **params (Any): Arguments matching the entry's declared parameters exactly.

        Returns:
            list[str]: The messages in order; a text or spec row renders one.

        Raises:
            KeyError: If the release has no such entry or lists it as dropped.
            RenderError: If the arguments do not match the declared types or the output exceeds the hard cap.
        """
        self._require(entry)
        check_params(self.entry_params(entry), params, entry)
        renderable = self._renderables[entry]
        ctx = RenderContext()
        if isinstance(renderable, Document):
            out = ["".join(text for text, _src in message.render_chunks(ctx, {key: params[key] for key in message.params})) for message in renderable.messages]
        else:
            out = ["".join(text for text, _src in renderable.render_chunks(ctx, params))]
        recorder = _RECORDER.get()
        if recorder is not None:
            recorder.record(entry, params)
        return out

    def trace(self, entry: str, /, **params: Any) -> list[Span]:
        """Render an entry and return the provenance of every output range.

        Args:
            entry (str): Entry name.
            **params (Any): Arguments matching the entry's declared parameters exactly.

        Returns:
            list[Span]: Spans in output order; each names the row, version and node that produced the bytes.

        Raises:
            KeyError: If the release has no such entry or lists it as dropped.
        """
        self._require(entry)
        check_params(self.entry_params(entry), params, entry)
        return spans_of(self._render(entry, params, RenderContext()))

    def _plain(self, name: str) -> str:
        """Render a row without parameters and without recording it."""
        return "".join(text for text, _src in self._renderables[name].render_chunks(RenderContext(), {}))

    def keyspace(self, role: str, route: str | None, slot: int | None = None) -> str:
        """The 16-hex keyspace of one role/route: a hash of the identity texts of its closure's surfaces.

        A candidate slot whose document the release carries (``candidate.s<k>``) has the candidate closure's keyspace
        with that document's two messages in place of the shared candidate document's: equal to the shared keyspace
        when the texts are equal, its own otherwise. Any other slot is the shared candidate keyspace.

        Args:
            role (str): ``candidate``, ``judge`` or ``reviewer``.
            route (str | None): Route name or ``None``.
            slot (int | None): The candidate's position in the pool, counted from 1 (candidates only).

        Returns:
            str: Keyspace value.

        Raises:
            KeyError: If ``(role, route)`` is not one of the keyspaces.
            ValueError: If a slot is given for a role other than ``candidate``.
        """
        if (role, route) not in self._keyspaces:
            raise KeyError(f"unknown keyspace {E.route_label(role, route)!r}")
        if slot is not None and role != "candidate":
            raise ValueError(f"only candidates run in slots; {role} got slot {slot}")
        return self._slot_keyspaces.get((route, slot)) or self._keyspaces[(role, route)]

    def key_extra(self, role: str, route: str | None, slot: int | None = None) -> str:
        """The cache-key extra of one role/route (and candidate slot): empty when its keyspace equals the epoch root's.

        Args:
            role (str): ``candidate``, ``judge`` or ``reviewer``.
            route (str | None): Route name or ``None``.
            slot (int | None): The candidate's position in the pool; a slot with a document of its own keys on its
                own keyspace, so only that slot's keys move.

        Returns:
            str: ``""`` when this release is r0 or the keyspace equals the epoch root's, else the 16-hex keyspace. Without an r0
                anchor (root missing) every keyspace counts as moved, so caches are
                never shared with an unknown baseline.
        """
        value = self.keyspace(role, route, slot)
        if self._anchor is None:
            return ""
        if self._anchor_keyspaces is not None and self._anchor_keyspaces.get((role, route)) == value:
            return ""
        return value

    def keyspaces(self) -> Mapping[str, str]:
        """Every keyspace of the release by label: the role/route keyspaces, then the slots' own ones.

        Two releases send the same prompts exactly when these mappings are equal (:func:`same_prompts`).

        Args:
            None.

        Returns:
            Mapping[str, str]: ``candidate``, ``candidate/grain``, … and ``candidate[s<k>]``-style labels for each slot
                document whose texts differ from the shared candidate document's.
        """
        out = {E.route_label(*pair): value for pair, value in self._keyspaces.items()}
        out.update((E.route_label("candidate", route, slot), self._slot_keyspaces[(route, slot)]) for route, slot in self._own_slots())
        return MappingProxyType(out)

    def _own_slots(self) -> list[tuple[str | None, int]]:
        """``(route, slot)`` of every slot keyspace that differs from the shared candidate keyspace, by slot, then route."""
        order = E.ROUTES["candidate"]
        return sorted(
            (key for key, value in self._slot_keyspaces.items() if value != self._keyspaces[("candidate", key[0])]),
            key=lambda key: (key[1], order.index(key[0])),
        )

    def moved_keyspaces(self) -> tuple[str, ...]:
        """Labels of the keyspaces whose ``key_extra`` is non-empty, the slots' own keyspaces included.

        Args:
            None.

        Returns:
            tuple[str, ...]: e.g. ``("candidate", "candidate/grain")`` or ``("candidate[s2]", "candidate/grain[s2]")``;
                empty when everything equals the epoch root.
        """
        moved = [E.route_label(*pair) for pair in E.KEYSPACES if self.key_extra(*pair)]
        moved.extend(E.route_label("candidate", route, slot) for route, slot in self._own_slots() if self.key_extra("candidate", route, slot))
        return tuple(moved)

    def recorder(self, role: str, route: str | None, slot: int | None = None) -> RenderRecorder:
        """Create a recorder for one agent run under this release.

        Args:
            role (str): ``candidate``, ``judge`` or ``reviewer``.
            route (str | None): Route name or ``None``.
            slot (int | None): The candidate's position in the pool; its keyspace is the slot's (:meth:`keyspace`).

        Returns:
            RenderRecorder: Unbound; call :meth:`RenderRecorder.bind` in the run's task.
        """
        return RenderRecorder(self.id, self.proposal, role, route, self.keyspace(role, route, slot))

    def slot_surface(self, slot: int | None, name: str, /, **params: Any) -> str:
        """Render a candidate message for one candidate slot: its own document's when the release carries it.

        Args:
            slot (int | None): The candidate's position in the pool, counted from 1; ``None`` for a run outside a pool.
            name (str): ``system`` or ``user``.
            **params (Any): The surface's arguments.

        Returns:
            str: ``candidate.s<slot>.<name>`` when the release provides it, else ``candidate.<name>``; the bound
                recorder counts the row that rendered.
        """
        if slot is not None and f"candidate.s{slot}.{name}" in self._providers:
            return self.surface("candidate", f"s{slot}.{name}", **params)
        return self.surface("candidate", name, **params)

    def chat(self, entry: str, /, **params: Any) -> list[dict[str, str]]:
        """Render a two-message document as chat messages: a generator document's request.

        Args:
            entry (str): Entry name, e.g. ``gen.rules_general.map``; its surfaces end in ``.system`` and ``.user``.
            **params (Any): Arguments matching the entry's declared parameters exactly.

        Returns:
            list[dict[str, str]]: ``[{"role": "system", "content": …}, {"role": "user", "content": …}]``.

        Raises:
            KeyError: If the release does not carry the entry (a release published before the generator documents).
        """
        known = E.ENTRIES.get(entry)
        if known is None or [surface.rsplit(".", 1)[1] for surface in known.surfaces] != ["system", "user"]:
            raise KeyError(f"{entry!r} is not an entry of a system and a user message")
        system, user = self.messages(entry, **params)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def knob(self, name: str) -> Any:
        """The value of one pipeline knob in this process.

        An explicitly set environment variable wins (the knob's ``env``, read exactly as the code always read it, except
        that an empty ``RUBIKSQL_SYNONYM_NEAR`` is refused: it read as 0.65, a floor BASELINE no longer carries);
        otherwise the release's explicit value (its ``config.knobs`` row, or a closure/2 release's parameter set);
        otherwise the knob's default, which is what every release published before the knobs moved into the release
        ran with.

        Args:
            name (str): Knob name, e.g. ``max_turns`` (:data:`entries.KNOBS`).

        Returns:
            Any: The value, normalized (:meth:`entries.Knob.normalize`): an ``int``, ``float``, ``bool`` or a tuple.

        Raises:
            KeyError: If the name is not a knob.
            ValueError: If the environment variable is set to a value its parser refuses.
        """
        known = E.KNOBS.get(name)
        if known is None:
            raise KeyError(f"unknown knob {name!r}; known: {sorted(E.KNOBS)}")
        if known.env is not None and known.env in os.environ and known.parse is not None:
            return known.normalize(known.parse(os.environ[known.env]))
        if name in self._knobs:
            return self._knobs[name]
        return known.normalize(known.default)

    def knob_sources(self) -> Mapping[str, str]:
        """Where each knob's value comes from in this process.

        Args:
            None.

        Returns:
            Mapping[str, str]: Knob name to ``env`` (an explicitly set variable overrides the release), ``release``
                (an explicit value of the release's ``config.knobs`` or parameter set) or ``default``.
        """
        def source(knob: E.Knob) -> str:
            if knob.env is not None and knob.env in os.environ:
                return "env"
            return "release" if knob.name in self._knobs else "default"

        return MappingProxyType({name: source(knob) for name, knob in E.KNOBS.items()})

    def knob_settings(self) -> dict[str, Any]:
        """The value the release gives every knob, whatever the environment sets.

        Args:
            None.

        Returns:
            dict[str, Any]: Knob name to the release's explicit value, else its default, normalized as :meth:`knob`
                returns it.
        """
        return {name: self._knobs[name] if name in self._knobs else knob.normalize(knob.default) for name, knob in E.KNOBS.items()}

    def parameter_values(self) -> dict[str, Any]:
        """The explicit parameter values the release carries, exactly as stored.

        Args:
            None.

        Returns:
            dict[str, Any]: Knob name to its JSON value: a closure/1 release's ``config.knobs`` object (empty when it
                carries none), a closure/2 release's parameter-set ``values``. Omitted knobs keep their code defaults;
                no environment override is applied. A detached copy.
        """
        return deepcopy(self._values)

    def rules_general(self) -> Mapping[str, Any]:
        """The general rule corpus the release carries, parsed.

        A closure/1 release carries it as its ``rules.general`` row; a closure/2 release's corpus is reconstructed
        exactly from its ordered rule versions and corpus metadata (:func:`entity_model.reconstruct_corpus`), so both
        parse the same ``general_rules.json`` spelling.

        Args:
            None.

        Returns:
            Mapping[str, Any]: ``{"version", "generation", "rules", "provenance"}``, parsed afresh on every call, so a
                caller may change its copy.

        Raises:
            KeyError: If the release carries no corpus (a closure/1 release published before the corpus moved into it).
        """
        if self._corpus is None:
            raise KeyError(f"release {self.id} carries no {E.RULES_ENTRY!r}")
        return json.loads(self._corpus)

    def banner(self, **extra: str) -> str:
        """The ``[prompt-release]`` log line.

        ``unused_entries`` counts the entries the release lists that this code never renders: unknown ones and dropped
        ones. ``knobs_env`` names the knobs an environment variable overrides, when any does.

        Args:
            **extra (str): Consumer fields appended in order, e.g. ``system="995a19c6"``.

        Returns:
            str: One line starting with ``[prompt-release]``.
        """
        moved = self.moved_keyspaces()
        keyspace = "v0" if not moved else "moved(" + ",".join(moved) + ")"
        fields = [
            f"release={self.id}",
            f"proposal={self.proposal or '-'}",
            f"resolved={self.resolved_from}",
            "store=snapshot",
            f"ns={os.environ.get(ENV_CACHE_NS) or '-'}",
            f"allow_retired={int(self.allow_retired)}",
            f"keyspace={keyspace}",
            f"unused_entries={len(self.unknown_entries) + len(self.dropped_entries)}",
        ]
        if self.warnings:
            fields.append(f"warnings={len(self.warnings)}")
        overridden = [name for name, source in self.knob_sources().items() if source == "env"]
        if overridden:
            fields.append(f"knobs_env={','.join(overridden)}")
        fields.extend(f"{key}={value}" for key, value in extra.items())
        return "[prompt-release] " + " ".join(fields)


# ---------------------------------------------------------------------------------------
# Loading


class _Loader:
    """Runs the load-time check list of runtime §4 for one release and builds the ``Release``.

    ``check_revoked=False`` is the anchor mode: the epoch root anchor is loaded only to compare keyspaces
    and is never rendered, so a revoked archived-only row must not stop the release that replaced it from
    loading (revocation still applies, unconditionally, to every release that is rendered).
    """

    def __init__(self, store: Store, *, allow_retired: bool, check_revoked: bool = True, policy: bool = True) -> None:
        self.store = store
        self.allow_retired = allow_retired
        # Stored-history verification checks typed rows and legacy carriers for integrity, not current policy.
        self.policy = policy
        self.revoked = store.revoked() if check_revoked else {}
        self.origins: set[str] = set()

    def fail(self, where: str, check: str, detail: str) -> PromptStoreError:
        return PromptStoreError(f"{where}: {check}: {detail}")

    def load(self, ref: str, *, resolved_from: str, anchor: Release | None) -> Release:
        proposal: str | None = None
        rid = ref
        if PROPOSAL_RE.fullmatch(ref):
            proposal = ref
            loaded = self.store.proposal(ref)
            self.origins.add(loaded.origin)
            body = loaded.data
            if not isinstance(body, Mapping) or body.get("schema") != SCHEMA_PROPOSAL:
                raise self.fail(loaded.relative, "schema", f"expected {SCHEMA_PROPOSAL!r}")
            if body.get("id") != ref or proposal_id(body) != ref:
                raise self.fail(loaded.relative, "id", "recomputed proposal id does not match")
            rid = body.get("release")
            if not isinstance(rid, str) or not RELEASE_RE.fullmatch(rid):
                raise self.fail(loaded.relative, "release", f"proposal names an invalid release {rid!r}")
        elif not RELEASE_RE.fullmatch(ref):
            raise PromptStoreError(f"{ref!r} is not a release (r<16hex>) or proposal (p<16hex>) id")

        manifest = self.store.release(rid)
        self.origins.add(manifest.origin)
        data = manifest.data
        schema = data.get("schema") if isinstance(data, Mapping) else None
        formats = [f"{name} with {closure}" for closure, name in RELEASE_FORMATS.items()]
        if schema not in RELEASE_FORMATS.values():
            raise self.fail(manifest.relative, "schema", f"unknown release schema {schema!r}; known formats {formats}")
        refs = data.get("refs")
        entries = data.get("entries")
        rules = data.get("rules")
        if not isinstance(refs, Mapping) or not refs:
            raise self.fail(manifest.relative, "refs", "must be a non-empty mapping of row name to content key")
        if not isinstance(entries, list) or not entries or len(set(entries)) != len(entries):
            raise self.fail(manifest.relative, "entries", "must be a non-empty list of distinct row names")
        if rules not in KNOWN_RULES:
            raise self.fail(manifest.relative, "rules", f"unknown closure rules {rules!r}; known {list(KNOWN_RULES)}")
        if RELEASE_FORMATS[rules] != schema:
            raise self.fail(manifest.relative, "rules",
                            f"release schema {schema!r} cannot carry closure rules {rules!r}; known formats {formats}")
        try:
            valid = data.get("id") == rid and release_document_id(data) == rid
        except (KeyError, TypeError, ValueError) as error:
            raise self.fail(manifest.relative, "id", f"the release id cannot be recomputed: {error}") from error
        if not valid:
            raise self.fail(manifest.relative, "id", "recomputed release id does not match the file name")
        if rules == M.CLOSURE_V2:
            try:
                M.check_release_document(data)
            except M.EntityModelError as error:
                raise self.fail(manifest.relative, "content_refs", str(error)) from error
        elif data.get("content_refs") is not None:
            raise self.fail(manifest.relative, "content_refs", f"a {E.CLOSURE_RULES} release carries no typed references")
        if not self.allow_retired and self.store.is_retired(rid):
            raise self.fail(manifest.relative, "retired", f"release {rid} is retired; set {ENV_ALLOW_RETIRED}=1 only to reproduce an old experiment")
        for name in entries:
            if name not in refs:
                raise self.fail(manifest.relative, "entries", f"entry {name!r} is not in refs")

        objects: dict[str, Mapping[str, Any]] = {}
        renderables: dict[str, TextLeaf | Composer] = {}
        pending: dict[str, Mapping[str, Any]] = {}
        for name in sorted(refs):
            key = refs[name]
            where = f"objects/{name}"
            try:
                check_name(name)
                check_content_key(key)
            except PolicyError as error:
                raise self.fail(where, "ref", str(error)) from error
            loaded = self.store.object(name, key[:16])
            self.origins.add(loaded.origin)
            row = self.check_row(loaded, name, key)
            objects[name] = loaded.data
            meta = row["manifest"]["meta"]
            if meta["kind"] == "text":
                renderables[name] = self.check_text_row(loaded.relative, row, meta)
            elif meta["kind"] == "doc":
                renderables[name] = self.check_doc_row(loaded.relative, row, meta)
            else:
                pending[name] = meta
                self.check_parts_mapping(loaded.relative, meta, refs)

        self.resolve_specs(pending, objects, refs, renderables)

        reachable = set(entries)
        stack = list(entries)
        while stack:
            item = renderables[stack.pop()]
            if isinstance(item, Composer):
                for slot, child in item.parts.items():
                    if child.name not in reachable:
                        reachable.add(child.name)
                        stack.append(child.name)
        orphans = sorted(set(refs) - reachable)
        if orphans:
            raise self.fail(manifest.relative, "closure", f"refs carry rows no entry reaches: {orphans}")

        unknown: list[str] = []
        dropped: list[str] = []
        warnings: list[str] = []
        providers: dict[str, tuple[str, int]] = {}
        for name in sorted(entries):
            known = E.ENTRIES.get(name)
            renderable = renderables[name]
            if known is None:
                unknown.append(name)
                continue
            self.check_entry_shape(name, known, renderable)
            bound = renderable.bound(self.text_bounds(known)) if isinstance(renderable, TextLeaf) else renderable.bound()
            if bound > OUTPUT_MAX_CHARS:
                # a spec bound saturates at the cap, so the exact figure is not known here
                raise self.fail(f"entry {name}", "bound", f"static output bound exceeds {OUTPUT_MAX_CHARS} characters")
            if isinstance(renderable, Composer):
                try:
                    renderable.normal_form()
                except SpecError as error:
                    raise self.fail(f"entry {name}", "normal form", str(error)) from error
            for index, surface in enumerate(known.surfaces):
                if surface in providers:
                    raise self.fail(manifest.relative, "entries", f"surface {surface!r} is provided by both {providers[surface][0]!r} and {name!r}; a release carries one layout per surface")
                providers[surface] = (name, index)
        missing = [surface for surface in E.required_surfaces() if surface not in providers]
        if missing:
            raise self.fail(manifest.relative, "entries", f"required surfaces are absent: {missing}")
        identity = {surface: declared.fallback for surface, declared in E.SURFACES.items() if declared.fallback is not None and surface not in providers}
        for surface, (name, index) in sorted(providers.items(), key=lambda item: item[1]):
            renderable = renderables[name]
            identity[surface] = self.identity_of(name, renderable, index)
            declared = E.SURFACES[surface]
            if declared.params or not declared.contracts:
                continue
            executor = renderable.messages[index] if isinstance(renderable, Document) else renderable
            rendered = executor.render()
            for contract in declared.contracts:
                violation = contract.check(rendered)
                if violation is None:
                    continue
                if contract.severity == "error":
                    raise self.fail(f"entry {name}", f"contract {contract.name}", violation)
                warnings.append(f"{name}: {contract.name}: {violation}")

        typed_rows: dict[str, dict[str, Mapping[str, Any]]] = {}
        if rules == M.CLOSURE_V2:
            closure = self.typed_closure(manifest.relative, data)
            values, corpus = dict(closure.values), closure.corpus_text
            # The data surfaces keep the exact legacy texts, so the surface identity equals the closure/1 release's.
            identity.update(M.legacy_texts(closure))
            typed_rows = {M.TOOL_ENTITY: {row["object_id"]: row for row in closure.tool_rows},
                          M.RULE_ENTITY: {row["object_id"]: row for row in closure.rule_rows},
                          M.PARAMETER_ENTITY: {closure.parameter_row["object_id"]: closure.parameter_row}}
        else:
            values, corpus = self.check_data_rows(manifest.relative, providers, renderables)
        # Integrity-only history reads retain exact values below, without projecting them into
        # current pipeline declarations or role keyspaces. These releases are never executed.
        knobs = {name: E.KNOBS[name].normalize(value) for name, value in values.items()} if self.policy else {}
        origin = "package" if self.origins == {"package"} else sorted(item for item in self.origins if item != "package")[0]
        return Release(
            id=rid,
            proposal=proposal,
            refs=refs,
            entries=entries,
            rules=rules,
            origin=origin,
            resolved_from=resolved_from,
            allow_retired=self.allow_retired,
            objects=objects,
            renderables=renderables,
            unknown_entries=unknown,
            dropped_entries=dropped,
            warnings=warnings,
            providers=providers,
            identity=identity,
            anchor=anchor,
            knobs=knobs,
            corpus=corpus,
            values=values,
            content_refs=data["content_refs"] if rules == M.CLOSURE_V2 else None,
            typed_rows=typed_rows,
            manifest=data,
        )

    def typed_closure(self, where: str, data: Mapping[str, Any]) -> M.TypedClosure:
        """Resolve and verify a closure/2 release's typed references against the store's canonical rows.

        Nothing falls back: a missing, altered or unresolvable Tool, rule or parameter-set reference refuses the
        release, so its knobs never silently revert to code defaults and its corpus never to another version.

        Args:
            where (str): The release manifest's path, for messages.
            data (Mapping[str, Any]): The release file body, already checked by
                :func:`entity_model.check_release_document`.

        Returns:
            entity_model.TypedClosure: The verified Tool rows, rule versions, reconstructed corpus and parameter
                values.

        Raises:
            PromptStoreError: If a reference, stored row, rule order, corpus layout, source pin or value differs.
        """
        try:
            return M.resolve_closure(data["content_refs"], self.store.typed_index(),
                                     capsule_rows={row["object_id"]: row for row in self.store.native("sys-capsule")},
                                     policy=self.policy)
        except (KeyError, TypeError, ValueError) as error:
            raise self.fail(where, "content_refs", str(error)) from error

    def check_data_rows(
        self, where: str, providers: Mapping[str, tuple[str, int]], renderables: Mapping[str, TextLeaf | Composer]
    ) -> tuple[dict[str, Any], str | None]:
        """Parse and check the data rows a closure/1 release carries: ``config.knobs`` and ``rules.general``.

        A data row is a text row whose text is JSON in one fixed spelling, so equal content always has the same
        content key. Execution loads validate the knobs against current declarations, types and ranges;
        integrity-only history reads keep the recorded JSON object without current knob policy. The text is
        written as :func:`entries.knobs_text` writes it; an execution knob it leaves out keeps its default. The rule corpus
        is an object with exactly ``version`` (a string), ``generation`` and ``provenance`` (objects) and ``rules``
        (a list of objects), written as ``json.dumps(corpus, ensure_ascii=False, indent=2)`` — the bytes of
        ``resources/generated_rules/general_rules.json``, which promotion regenerates from the row.

        Args:
            where (str): The release manifest's path, for messages.
            providers (Mapping[str, tuple[str, int]]): Surface to its providing entry.
            renderables (Mapping[str, TextLeaf | Composer]): The release's executors.

        Returns:
            tuple[dict[str, Any], str | None]: The explicit knob values the release carries, as written (the loader
                normalizes them), and the rule corpus's text (``None`` when the release carries none).

        Raises:
            PromptStoreError: If a data row is not a text row, is not JSON in its fixed spelling, names an unknown
                knob, carries a value outside a knob's type or range, or the corpus lacks or mistypes a key.
        """
        values: dict[str, Any] = {}
        corpus: str | None = None
        if E.KNOBS_ENTRY in providers:
            data, text = self.data_json(where, providers[E.KNOBS_ENTRY][0], renderables)
            if not isinstance(data, dict):
                raise self.fail(f"entry {E.KNOBS_ENTRY}", "knobs", "the text must be a JSON object of knob values")
            if self.policy:
                unknown = sorted(set(data) - set(E.KNOBS))
                if unknown:
                    raise self.fail(f"entry {E.KNOBS_ENTRY}", "knobs", f"unknown knob(s) {unknown}; the code declares {sorted(E.KNOBS)}")
                problems = [problem for name, value in sorted(data.items()) if (problem := E.KNOBS[name].violation(value)) is not None]
                if problems:
                    raise self.fail(f"entry {E.KNOBS_ENTRY}", "knobs", "; ".join(problems))
            if E.knobs_text(data) != text:
                raise self.fail(f"entry {E.KNOBS_ENTRY}", "knobs", "the text must be spelled json.dumps(values, indent=2, sort_keys=True, ensure_ascii=False), with no trailing newline")
            values = dict(data)
        if E.RULES_ENTRY in providers:
            data, text = self.data_json(where, providers[E.RULES_ENTRY][0], renderables)
            shape = {"version": str, "generation": dict, "provenance": dict, "rules": list}
            if not isinstance(data, dict) or set(data) != set(shape) or any(not isinstance(data[key], kind) for key, kind in shape.items()):
                raise self.fail(f"entry {E.RULES_ENTRY}", "rules", "the text must be a JSON object with exactly a string version, generation and provenance objects and a rules list")
            if any(not isinstance(rule, dict) for rule in data["rules"]):
                raise self.fail(f"entry {E.RULES_ENTRY}", "rules", "every rule must be a JSON object")
            if json.dumps(data, ensure_ascii=False, indent=2) != text:
                raise self.fail(f"entry {E.RULES_ENTRY}", "rules", "the text must be spelled json.dumps(corpus, ensure_ascii=False, indent=2), the bytes of general_rules.json")
            corpus = text
        return values, corpus

    def data_json(self, where: str, name: str, renderables: Mapping[str, TextLeaf | Composer]) -> tuple[Any, str]:
        """The parsed JSON of a data row and its text; a data row must be a text row."""
        renderable = renderables[name]
        if not isinstance(renderable, TextLeaf):
            raise self.fail(f"entry {name}", "data", f"a data row is a text row; the release carries a {renderable.kind} row")
        try:
            return json.loads(renderable.text), renderable.text
        except json.JSONDecodeError as error:
            raise self.fail(f"entry {name}", "data", f"the text is not JSON: {error}") from error

    def identity_of(self, name: str, renderable: TextLeaf | Composer, index: int) -> str:
        """The identity text of one surface a row provides: the text as written, slot and tag markers included.

        The text is an identity because every kind of row reads it the same way: a text row may not contain a
        document tag (:data:`policy.TAG_START_RE`), so a text that is both a valid text row and a valid document
        message renders the same bytes as either. A composition prints as a document (:func:`doc.document_of`).
        One that uses a construct no document can write — a ``where`` filter, a ``not`` or ``eq`` test, a ``!r``
        part slot, a declared parameter it never uses — is identified by its canonical normal form behind U+001E,
        a text no row can equal (neither a document nor a text row may contain U+001E): no one-document release
        can be the same prompts as it until the construct is rewritten.

        Args:
            name (str): The providing entry.
            renderable (TextLeaf | Composer): Its executor.
            index (int): The message the surface is.

        Returns:
            str: A document's message, a text row's text, or a composition's document (or tagged normal form).
        """
        if isinstance(renderable, Document):
            return renderable.segments[index]
        if isinstance(renderable, TextLeaf):
            return renderable.text
        form = renderable.normal_form()
        try:
            return document_of(form, renderable.params)
        except DocError:
            return "\x1enf:" + canon(form)

    @staticmethod
    def part_names(composer: Composer) -> set[str]:
        """Names of every row a composition includes as a part, at any depth."""
        names: set[str] = set()
        stack = list(composer.parts.values())
        while stack:
            part = stack.pop()
            if part.name not in names:
                names.add(part.name)
                if isinstance(part, Composer):
                    stack.extend(part.parts.values())
        return names

    @staticmethod
    def text_bounds(entry: E.Entry) -> dict[str, int]:
        return {slot: PARAM_MAX_CHARS if declared == "str" else INT_MAX_CHARS for slot, declared in entry.params.items()}

    def check_row(self, loaded: Any, name: str, key: str) -> Mapping[str, Any]:
        where = loaded.relative
        data = loaded.data
        if not isinstance(data, Mapping) or data.get("schema") != SCHEMA_OBJECT or not isinstance(data.get("row"), Mapping):
            raise self.fail(where, "schema", f"expected {{'schema': {SCHEMA_OBJECT!r}, 'row': {{...}}}}")
        row = data["row"]
        if row.get("name") != name:
            raise self.fail(where, "name", f"row.name {row.get('name')!r} differs from the ref {name!r}")
        if row.get("version") != key[:16]:
            raise self.fail(where, "version", f"row.version {row.get('version')!r} is not content_key[:16] {key[:16]!r}")
        manifest = row.get("manifest")
        if not isinstance(manifest, Mapping):
            raise self.fail(where, "manifest", "row.manifest must be a mapping")
        meta = manifest.get("meta")
        try:
            check_meta_keys(meta)
        except PolicyError as error:
            raise self.fail(where, "meta", str(error)) from error
        if meta["schema"] != SCHEMA_PROMPT:
            raise self.fail(where, "meta.schema", f"expected {SCHEMA_PROMPT!r}, got {meta['schema']!r}")
        if meta["kind"] not in KINDS:
            raise self.fail(where, "meta.kind", f"expected one of {list(KINDS)}, got {meta['kind']!r}")
        if meta["content_key"] != key:
            raise self.fail(where, "meta.content_key", "differs from the ref")
        try:
            recomputed = content_key(name, meta)
        except (KeyError, TypeError, ValueError) as error:
            raise self.fail(where, "content_key", f"meta lacks the fields of a {meta['kind']} row ({error})") from error
        if recomputed != key:
            raise self.fail(where, "content_key", f"recomputed {recomputed} differs from the ref {key}")
        if key in self.revoked:  # empty in anchor mode; otherwise no switch overrides revocation
            raise self.fail(where, "revoked", f"content key is revoked by decision {self.revoked[key]!r}; no switch overrides revocation")
        reason = manifest.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise self.fail(where, "reason", "manifest.reason must be a non-empty string")
        return row

    def check_capsule(self, where: str, row: Mapping[str, Any], expected_source: str, qualname: str) -> None:
        capsule = row.get("capsule")
        if not isinstance(capsule, Mapping):
            raise self.fail(where, "capsule", "row.capsule must be a mapping")
        layers = capsule.get("restore_layers")
        if not isinstance(layers, list) or len(layers) != 1 or not isinstance(layers[0], Mapping):
            raise self.fail(where, "capsule.restore_layers", "exactly one restore layer is required")
        layer = layers[0]
        if set(layer) != _LAYER_KEYS:
            raise self.fail(where, "capsule.restore_layers", f"layer keys {sorted(layer)} differ from {sorted(_LAYER_KEYS)}")
        shape = {"type": "source", "priority": 10, "qualname": qualname, "filename": "<capsule-code>", "global_refs": {}}
        for field, value in shape.items():
            if layer.get(field) != value:
                raise self.fail(where, f"capsule.restore_layers.{field}", f"expected {value!r}, got {layer.get(field)!r}")
        source = layer.get("source")
        if source != expected_source:
            raise self.fail(where, "source", "capsule source differs from the source regenerated from meta")
        summary = capsule.get("source")
        if not isinstance(summary, Mapping) or summary.get("source") != source:
            raise self.fail(where, "capsule.source", "capsule.source.source differs from the restore layer")
        digest = sha256_hex(source)
        fingerprints = capsule.get("fingerprints")
        hashes = (layer.get("source_hash"), summary.get("source_hash"), fingerprints.get("source_hash") if isinstance(fingerprints, Mapping) else None)
        if any(value != digest for value in hashes):
            raise self.fail(where, "source_hash", "the three source hashes must all equal sha256(source)")
        trust = capsule.get("trust")
        if not isinstance(trust, Mapping) or trust.get("verify_source_hash") is not True:
            raise self.fail(where, "capsule.trust", "verify_source_hash must be true")

    def check_doc_row(self, where: str, row: Mapping[str, Any], meta: Mapping[str, Any]) -> Document:
        """Verify a doc row: policy, grammar, declared parameters, and the capsule re-derived from ``meta.text``.

        Args:
            where (str): Object path for messages.
            row (Mapping[str, Any]): The exported row.
            meta (Mapping[str, Any]): Its ``manifest.meta``.

        Returns:
            Document: The row's executor; a known entry's messages declare the parameters of its surfaces.

        Raises:
            PromptStoreError: If meta lacks ``text``/``params`` or stores a spec, parts or slots, the document fails its
                policy or grammar or has no tag (a text row's shape), or the capsule source differs from the source
                compiled from the document.
        """
        if "text" not in meta or "params" not in meta:
            raise self.fail(where, "meta", "a doc row needs meta.text and meta.params")
        stray = [key for key in ("spec", "parts", "slots") if key in meta]
        if stray:
            raise self.fail(where, "meta", f"a doc row stores no {stray}; they derive from meta.text")
        if not isinstance(meta["params"], Mapping):
            raise self.fail(where, "meta.params", "must map parameter names to types")
        try:
            check_document(meta["text"])
            parsed = parse_document(meta["text"], meta["params"])
            source = compile_document(row["name"], meta["text"], meta["params"])
        except (PolicyError, DocError) as error:
            raise self.fail(where, "document", str(error)) from error
        self.check_capsule(where, row, source, "compose")
        known = E.ENTRIES.get(row["name"])
        surfaces = known.surfaces if known is not None else ()
        message_params = [E.SURFACES[surface].params for surface in surfaces] if surfaces and len(surfaces) == len(parsed.messages) else None
        try:
            return Document(row["name"], row["version"], meta["text"], parsed, message_params)
        except SpecError as error:
            raise self.fail(where, "document", str(error)) from error

    def check_text_row(self, where: str, row: Mapping[str, Any], meta: Mapping[str, Any]) -> TextLeaf:
        if "text" not in meta or "slots" not in meta:
            raise self.fail(where, "meta", "a text row needs meta.text and meta.slots")
        try:
            check_text(meta["text"], meta["slots"], name=row["name"])
        except PolicyError as error:
            raise self.fail(where, "text policy", str(error)) from error
        expected = template_code(row["name"], escape(meta["text"], meta["slots"]), [])
        self.check_capsule(where, row, expected, "render")
        return TextLeaf(row["name"], row["version"], meta["text"], meta["slots"])

    def check_parts_mapping(self, where: str, meta: Mapping[str, Any], refs: Mapping[str, str]) -> None:
        for field in ("spec", "parts", "params"):
            if field not in meta:
                raise self.fail(where, "meta", f"a spec row needs meta.{field}")
        parts = meta["parts"]
        if not isinstance(parts, Mapping):
            raise self.fail(where, "meta.parts", "must map slot names to row names")
        for slot, target in parts.items():
            try:
                check_identifier(slot, "part slot")
                check_name(target)
            except PolicyError as error:
                raise self.fail(where, "meta.parts", str(error)) from error
            if target not in refs:
                raise self.fail(where, "meta.parts", f"part {slot!r} names {target!r}, which is not in the release refs")
        if not isinstance(meta["params"], Mapping):
            raise self.fail(where, "meta.params", "must map parameter names to types")

    def resolve_specs(
        self,
        pending: Mapping[str, Mapping[str, Any]],
        objects: Mapping[str, Mapping[str, Any]],
        refs: Mapping[str, str],
        renderables: dict[str, TextLeaf | Composer],
    ) -> None:
        state: dict[str, int] = {}

        def build(name: str, chain: tuple[str, ...]) -> None:
            if name in renderables:
                return
            if state.get(name) == 1:
                cycle = " -> ".join(chain + (name,))
                raise self.fail(f"objects/{name}", "parts", f"parts form a cycle: {cycle}")
            state[name] = 1
            meta = pending[name]
            where = f"objects/{name}/{refs[name][:16]}.json"
            for target in meta["parts"].values():
                build(target, chain + (name,))
            parts = {slot: renderables[target] for slot, target in meta["parts"].items()}
            documents = sorted(target for target in meta["parts"].values() if isinstance(renderables[target], Document))
            if documents:
                raise self.fail(where, "parts", f"doc rows cannot be parts: {documents}")
            sigs = {slot: part.signature() for slot, part in parts.items()}
            try:
                spec = parse_spec(meta["spec"], params=meta["params"], part_sigs=sigs)
            except SpecError as error:
                raise self.fail(where, "spec", str(error)) from error
            row = objects[name]["row"]
            composer = Composer(name, row["version"], spec, parts)
            if composer.depth > PARTS_MAX_DEPTH:
                raise self.fail(where, "parts", f"part nesting depth {composer.depth} exceeds {PARTS_MAX_DEPTH}")
            expected = compile_spec(name, meta["spec"], meta["parts"], meta["params"])
            self.check_capsule(where, row, expected, "compose")
            renderables[name] = composer
            state[name] = 2

        for name in sorted(pending):
            build(name, ())

    def check_entry_shape(self, name: str, known: E.Entry, renderable: TextLeaf | Composer) -> None:
        """Check an entry's row against the code's contract: its message count and its parameters.

        The row's kind is the release's choice — a template prompt when the text has no blocks, a document when
        it has messages or blocks, a composition in the r0 layout — as long as the interface the call sites use
        holds: a document renders one message per surface the entry provides, a spec or doc row declares exactly
        the code's parameter types, a text row takes them as scalar slots.

        Args:
            name (str): Entry name.
            known (E.Entry): The code-facing contract.
            renderable (TextLeaf | Composer): The release's row for the entry.

        Returns:
            None: This method does not return a value.

        Raises:
            PromptStoreError: If the message count or the parameters differ from the code's, or a text row would have
                to take a list or record parameter.
        """
        messages = len(renderable.messages) if isinstance(renderable, Document) else 1
        expected = max(len(known.surfaces), 1)
        if messages != expected:
            provides = ", ".join(known.surfaces) or "one data document"
            raise self.fail(f"entry {name}", "messages", f"the code expects {expected} message(s) ({provides}), the {renderable.kind} row renders {messages}")
        declared = {param: check_type(value) for param, value in known.params.items()}
        if isinstance(renderable, Composer):
            if dict(renderable.params) != declared:
                raise self.fail(f"entry {name}", "params", f"release declares {dict(renderable.params)}, the code expects {declared}")
            return
        if set(renderable.slots) != set(known.params):
            raise self.fail(f"entry {name}", "slots", f"release declares slots {sorted(renderable.slots)}, the code expects {sorted(known.params)}")
        structured = sorted(param for param, value in declared.items() if value not in ("str", "int", "bool"))
        if structured:
            raise self.fail(f"entry {name}", "kind", f"the code passes structured parameters {structured}; a text row takes scalar slots only")


def _store(stores: "Store | Sequence[str | os.PathLike[str]] | None") -> Store:
    if isinstance(stores, Store):
        return stores
    if stores is None:
        return Store.from_env()
    return Store(list(stores))


def baseline_pointer(*, stores: "Store | Sequence[str | os.PathLike[str]] | None" = None) -> BaselinePointer:
    """Read the package ``BASELINE.json`` pointer.

    Args:
        stores (Store | Sequence[str | os.PathLike[str]] | None): A store, overlay directories, or ``None`` for
            ``RUBIKSQL_PROMPT_STORE``; the pointer is read from the package layer regardless.

    Returns:
        BaselinePointer: The pointed release id and the pointer file's git blob sha.

    Raises:
        PromptStoreError: If the authority current-reference file is absent (no ``BASELINE.json``) or the pointer is malformed.
    """
    return _store(stores).baseline()


def resolve_baseline(*, stores: "Store | Sequence[str | os.PathLike[str]] | None" = None) -> str:
    """Resolve ``baseline`` to a concrete release id for the launcher to pass to child processes.

    Args:
        stores (Store | Sequence[str | os.PathLike[str]] | None): See :func:`baseline_pointer`.

    Returns:
        str: The release id ``BASELINE.json`` points at.

    Raises:
        PromptStoreError: If r0 has not been exported or the pointer is malformed.
    """
    return baseline_pointer(stores=stores).release


def _load_anchor(store: Store, rid: str) -> Release:
    """Load the current epoch root keyspace anchor, allowing retired and revoked rows.

    The anchor is never sent to a model; it supplies the keyspace values ``key_extra`` compares against and the
    exact root keyspace components.
    A store that cannot supply it is a broken store, and the error says so by name instead of pointing at a
    row the pinned release does not contain.

    Args:
        store (Store): The layered store to read the anchor from.
        rid (str): The anchor release id (the current epoch root).

    Returns:
        Release: The verified anchor.

    Raises:
        PromptStoreError: If the anchor is missing from every layer or fails a content check.
    """
    try:
        return _Loader(store, allow_retired=True, check_revoked=False).load(rid, resolved_from="explicit", anchor=None)
    except PromptStoreError as error:
        raise PromptStoreError(
            f"{error}\nrelease {rid} is the keyspace anchor, loaded only to compare keyspaces and never "
            f"rendered; every store and every package must ship releases/{rid}.json together with the objects it references"
        ) from error


def _default_anchor_id(store: Store, ref: str) -> str:
    """Resolve the sole current-epoch root anchor, refusing archived content."""
    root = store.v0_root()
    if root is None:
        raise PromptStoreError("snapshot has no current epoch root")
    rid = store.proposal(ref).data.get("release") if PROPOSAL_RE.fullmatch(ref) else ref
    release = store.release(rid).data
    if release.get("epoch") != store.release(root).data["epoch"]:
        raise PromptStoreError(f"release {rid} is outside this authority epoch")
    return root


def load_release(ref: str, *, stores: "Store | None" = None, allow_retired: bool = False,
                 anchor: "Release | str | None" = None) -> Release:
    """Load an exact current-epoch release or proposal from a snapshot/read view.

    Args:
        ref: Exact release/proposal id.
        stores: Explicit authority read view; otherwise the process snapshot.
        allow_retired: Allow a retained historical release within this epoch.
        anchor: Explicit current root when a reader already loaded it.

    Returns:
        Release: Fully checked content-addressed prompt documents.

    Raises:
        PromptStoreError: Invalid content or an out-of-epoch reference.
    """
    store = _store(stores)
    root = _default_anchor_id(store, ref)
    rid = store.proposal(ref).data.get("release") if PROPOSAL_RE.fullmatch(ref) else ref
    reference = anchor if isinstance(anchor, Release) else _load_anchor(store, anchor or root) if rid != root else None
    return _Loader(store, allow_retired=allow_retired).load(ref, resolved_from="explicit", anchor=reference)


_CURRENT: dict[tuple[str, ...], Release] = {}
_CURRENT_LOCK = threading.Lock()


def _env_tuple() -> tuple[str, ...]:
    return (os.environ.get(ENV_RELEASE, "").strip() or "baseline",
            os.environ.get("RUBIKSQL_PROMPT_SNAPSHOT", ""),
            os.environ.get(ENV_STORE, ""),
            os.environ.get(ENV_ALLOW_RETIRED, "").strip().lower())


def _env_provenance(store: Store, ref: str) -> str:
    """Provenance of an explicit pin: ``env=baseline@<blob sha>`` when it names the pointed release, else ``env``.

    Launchers resolve ``baseline`` once and pass the concrete id to every child, so a child's manifest
    would otherwise record a bare ``env`` and lose which ``BASELINE.json`` the run followed.

    Args:
        store (Store): The layered store whose package pointer is consulted.
        ref (str): The pinned release or proposal id.

    Returns:
        str: The ``resolved_from`` value.
    """
    try:
        pointer = store.baseline()
    except PromptStoreError:
        return "env"  # the pin is authoritative; the pointer only annotates where it came from
    return f"env=baseline@{pointer.blob_sha}" if pointer.release == ref else "env"


def current() -> Release:
    """Load the process's immutable verified snapshot pin, with no Git fallback."""
    from pathlib import Path
    from .binding import binding_document, load_binding
    from .snapshot import ENV_SNAPSHOT, SnapshotError, _verified_snapshot, reject_execution_controls

    check_env_file()
    try:
        reject_execution_controls(os.environ)
    except SnapshotError as exc:
        raise PromptStoreError(str(exc)) from exc
    source = os.environ.get(ENV_SNAPSHOT)
    if not source:
        raise PromptStoreError("no verified prompt snapshot; bootstrap the authority head before importing consumers")
    path = Path(source).absolute()
    environment = _env_tuple()
    snapshot = _verified_snapshot(path, pin=environment[0])
    key = environment + (str(path), canon(binding_document(load_binding())))
    hit = _CURRENT.get(key)
    if hit is not None:
        return hit
    with _CURRENT_LOCK:
        # Only release construction is serialized; hot lookups have already
        # checked the snapshot inventory/stat identity without this cold lock.
        hit = _CURRENT.get(key)
        if hit is not None:
            return hit
        ref = key[0]
        if ref == "baseline":
            ref = snapshot.release
        store = Store(package=snapshot.path)
        rid = store.proposal(ref).data.get("release") if PROPOSAL_RE.fullmatch(ref) else ref
        if rid != snapshot.release:
            raise PromptStoreError("process requested a release different from its snapshot pin")
        release = load_release(ref, stores=store)
        _CURRENT[key] = release
        return release


def reset_current() -> None:
    """Forget releases, comparisons and semantic admissions, so the next call fully verifies the store.

    Args:
        None.

    Returns:
        None: This function does not return a value.
    """
    from .snapshot import _reset_verified_content

    with _CURRENT_LOCK:
        _CURRENT.clear()
        _SAME_PROMPTS.clear()
        _reset_verified_content()


#: ``same_prompts`` verdicts by ``(release id, other id, store layers)``; content-addressed ids never change meaning.
_SAME_PROMPTS: dict[tuple[str, str, tuple[str, ...]], bool] = {}


def _reset_current_after_fork() -> None:
    """Drop inherited releases and a lock another parent thread may hold."""
    global _CURRENT_LOCK
    _CURRENT_LOCK = threading.Lock()
    _CURRENT.clear()
    _SAME_PROMPTS.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_current_after_fork)


def same_prompts(release: Release, other: Any, stores: "Store | Sequence[str | os.PathLike[str]] | None" = None) -> bool:
    """Whether ``other`` sends the model exactly the prompts ``release`` sends.

    Two releases are the same prompts when their keyspaces — the static texts of every surface of
    each role/route closure, and of each candidate slot's own document (``Release.keyspaces``) — are
    equal, however the texts are cut into rows: the
    equal current documents share their prompt keyspaces. Every consumer that decides "same run",
    "same control" or "same identity" asks this instead of comparing ids, so a refactor-only
    promotion never re-answers a result directory, archives a pipeline stage or breaks a pairing.
    A release that is not in the stores, or fails its checks, is not the same.

    Args:
        release (Release): The loaded release the comparison is made from.
        other (Any): The other release's id (``r…``) or a proposal id (``p…``); anything else is not the same.
        stores (Store | Sequence[str | os.PathLike[str]] | None): Store layers ``other`` loads from;
            ``RUBIKSQL_PROMPT_STORE`` by default.

    Returns:
        bool: ``True`` for the same id or equal keyspaces.
    """
    if other == release.id:
        return True
    if not isinstance(other, str) or not (RELEASE_RE.fullmatch(other) or PROPOSAL_RE.fullmatch(other)):
        return False
    store = _store(stores)
    key = (release.id, other, tuple(str(layer.path) for layer in store.layers))
    with _CURRENT_LOCK:
        cached = _SAME_PROMPTS.get(key)
    if cached is None:
        try:
            loaded = load_release(other, stores=store, allow_retired=True)
            cached = loaded.keyspaces() == release.keyspaces()
        except PromptStoreError:
            cached = False
        with _CURRENT_LOCK:
            _SAME_PROMPTS[key] = cached
    return cached
