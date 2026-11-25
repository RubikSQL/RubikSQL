"""Real prompt inputs rebuilt from eval result rows, kept only when a release re-renders them byte-exactly.

A run directory holds ``pass1/<db_id>.json`` (or ``<db_id>.json``) files whose ``results``
rows carry each candidate's transcript (``candidates_debug[].messages``), the judge's user
message (``judge_debug.user_msg``) and, when the evaluator records it, the judge's transcript
(``judge_debug.messages``). The builder recovers the surface and the parameters behind every
model-visible message it can: static surfaces by equality, integer slots by a bounded search,
single string slots by template inversion, and structured surfaces such as ``judge.user`` by
sentinel inversion (render with placeholders for the unknown strings, align the literal
segments, read the values from the gaps). Samples name surfaces, not rows, so a corpus rebuilt
under one layout renders under a release of the other; an older corpus that names root entries is
read through the surface table. Prompt text that travels as a ``tool`` message is harvested too: a
whole tool message that is not a tool-result envelope (the final-check bounce, the judge's
literal guard) and every ``notes`` item inside an envelope's JSON body (the repeat-query and
empty-result notes). A sample is kept only if rendering the recovered parameters under the
given release reproduces the message byte-for-byte, so the corpus never contains a guess.
Corpora live outside git (``RUBIKSQL_PROMPT_CORPUS_DIR`` or ``~/.rubiksql_hb/prompt_corpus``);
the gate reports ``k of N changed`` between two releases and names the changed entries the
corpus never reaches.
"""
from __future__ import annotations

import difflib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import entries as E
from .canon import canon, sha256_hex
from .doc import Document
from .spec import Composer, RenderContext, RenderError, TextLeaf
from .store import PACKAGE_STORE_DIR, PromptStoreError, canonical_bytes

__all__ = [
    "ENV_CORPUS_DIR",
    "SCHEMA_CORPUS",
    "Corpus",
    "CorpusStats",
    "Sample",
    "SampleDiff",
    "build",
    "default_dir",
    "invert_by_sentinels",
    "load",
    "read_rows",
    "render_sample",
    "save",
    "stats",
    "tool_texts",
]

SCHEMA_CORPUS = "rubiksql.corpus/1"
ENV_CORPUS_DIR = "RUBIKSQL_PROMPT_CORPUS_DIR"

_ROUTE_SURFACE = {"grain": "candidate.route.grain"}
_GROUP_RE = re.compile(r"Candidate (\d+):\s*\[result-group ([A-Z]+): (?:(\d+) candidate\(s\) return this exact result|unique result)\]")
#: A rendered tool result: ``<run_sql_result>\n<compact JSON or ERROR line>\n</run_sql_result>`` (model_render).
_TOOL_ENVELOPE_RE = re.compile(r"^<(?P<tag>[a-z_]+)_result>\n(?P<body>.*)\n</(?P=tag)_result>$", re.DOTALL)
#: Judge surfaces that never arrive as a later user or tool message: the system prompt and the first user message.
_JUDGE_NOT_MESSAGES = frozenset({"judge.system", "judge.user", "shared.synonyms", "shared.rule_hooks"})
#: Candidate surfaces that never arrive as a later user or tool message.
_CANDIDATE_NOT_MESSAGES = frozenset({"candidate.system", "candidate.user", "candidate.route.grain", "shared.synonyms", "shared.rule_hooks"})
#: Sentinel wrapping: U+2400 (SYMBOL FOR NULL) is printable, so ``repr`` (a ``!r`` slot) keeps it verbatim
#: and only adds quotes, which then belong to the literal segments around the value.
_SENTINEL = "␀"
_SENTINEL_RE = re.compile(f"{_SENTINEL}U([0-9]+){_SENTINEL}")


class Unknown:
    """Marker for a string parameter :func:`invert_by_sentinels` must recover from the text."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "Unknown"


UNKNOWN = Unknown()


@dataclass(frozen=True)
class Sample:
    """One real model-visible message and the entry renders that produce it.

    Attributes:
        qid (str): Question id.
        db_id (str): Database id.
        role (str): ``candidate``, ``judge`` or ``reviewer``.
        route (str | None): Route the agent ran under, when recoverable.
        message (str): ``system``, ``user``, ``tool`` (a whole tool message) or ``note`` (a ``notes`` item inside a
            tool-result envelope).
        pieces (tuple[tuple[str, Mapping[str, Any]], ...]): ``(surface, params)`` in concatenation order; a corpus
            built before surfaces names root entries, which render as the surface they provide.
        sha16 (str): First 16 hex of the sha256 of the message bytes.
        prefix (str): Text placed before the rendered pieces; empty for samples built from surfaces (an older
            corpus kept the first message's question here).
    """

    qid: str
    db_id: str
    role: str
    route: str | None
    message: str
    pieces: tuple[tuple[str, Mapping[str, Any]], ...]
    sha16: str
    prefix: str = ""

    def to_json(self) -> dict[str, Any]:
        """Serialise the sample.

        Returns:
            dict[str, Any]: JSON-compatible mapping.
        """
        return {"qid": self.qid, "db_id": self.db_id, "role": self.role, "route": self.route, "message": self.message, "pieces": [[entry, dict(params)] for entry, params in self.pieces], "sha16": self.sha16, "prefix": self.prefix}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Sample":
        """Rebuild a sample.

        Args:
            data (Mapping[str, Any]): Output of :meth:`to_json`.

        Returns:
            Sample: The sample.
        """
        return cls(str(data["qid"]), str(data["db_id"]), str(data["role"]), data.get("route"), str(data["message"]), tuple((str(entry), dict(params)) for entry, params in data["pieces"]), str(data["sha16"]), str(data.get("prefix", "")))


@dataclass
class Corpus:
    """A set of samples rebuilt from one run under one release.

    Attributes:
        run (str): Run directory name.
        release (str): Release that re-rendered every sample byte-exactly.
        samples (list[Sample]): Kept samples.
        dropped (dict[str, int]): Messages not rebuilt, by reason.
    """

    run: str
    release: str
    samples: list[Sample] = field(default_factory=list)
    dropped: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Serialise the corpus.

        Returns:
            dict[str, Any]: ``{"schema": "rubiksql.corpus/1", "run", "release", "samples", "dropped"}``.
        """
        return {"schema": SCHEMA_CORPUS, "run": self.run, "release": self.release, "samples": [sample.to_json() for sample in self.samples], "dropped": dict(sorted(self.dropped.items()))}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Corpus":
        """Rebuild a corpus.

        Args:
            data (Mapping[str, Any]): Output of :meth:`to_json`.

        Returns:
            Corpus: The corpus.

        Raises:
            PromptStoreError: If the schema tag is wrong.
        """
        if not isinstance(data, Mapping) or data.get("schema") != SCHEMA_CORPUS:
            raise PromptStoreError(f"corpus: expected schema {SCHEMA_CORPUS!r}")
        return cls(str(data["run"]), str(data["release"]), [Sample.from_json(item) for item in data.get("samples", [])], dict(data.get("dropped", {})))


# ---------------------------------------------------------------------------------------
# Rows and storage


def read_rows(run_dir: Path) -> list[Mapping[str, Any]]:
    """Read every result row of a run directory.

    Args:
        run_dir (Path): Directory holding ``pass1/<db>.json`` or ``<db>.json`` files with a ``results`` list.

    Returns:
        list[Mapping[str, Any]]: Rows in file order.

    Raises:
        PromptStoreError: If no result file is found.
    """
    run_dir = Path(run_dir)
    directory = run_dir / "pass1" if (run_dir / "pass1").is_dir() else run_dir
    rows: list[Mapping[str, Any]] = []
    found = False
    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("."):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, Mapping) and isinstance(data.get("results"), list):
            found = True
            rows.extend(row for row in data["results"] if isinstance(row, Mapping))
    if not found:
        raise PromptStoreError(f"{run_dir}: no result files (<db_id>.json with a 'results' list) under it or its pass1/")
    return rows


def default_dir() -> Path:
    """The local corpus directory, never inside the repository.

    Returns:
        Path: ``$RUBIKSQL_PROMPT_CORPUS_DIR``, else ``$RUBIKSQL_DATA_DIR/prompt_corpus``, else ``~/.rubiksql_hb/prompt_corpus``.
    """
    explicit = os.environ.get(ENV_CORPUS_DIR)
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("RUBIKSQL_DATA_DIR") or os.path.expanduser("~/.rubiksql_hb")
    return Path(base) / "prompt_corpus"


def save(corpus: Corpus, path: Path | None = None) -> Path:
    """Write a corpus as canonical JSON to local storage.

    Args:
        corpus (Corpus): The corpus.
        path (Path | None): Destination file; ``default_dir()/<run>.json`` when omitted.

    Returns:
        Path: The written file.

    Raises:
        PromptStoreError: If the destination lies inside the git-tracked package store.
    """
    path = Path(path) if path is not None else default_dir() / f"{corpus.run}.json"
    try:
        path.resolve().relative_to(PACKAGE_STORE_DIR.resolve())
    except ValueError:
        pass
    else:
        raise PromptStoreError(f"{path}: a corpus is local-only and must never be written into the package store")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(canonical_bytes(corpus.to_json()))
    temp.replace(path)
    return path


def load(path: Path) -> Corpus:
    """Read a corpus file.

    Args:
        path (Path): File written by :func:`save`.

    Returns:
        Corpus: The corpus.
    """
    return Corpus.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------------------
# Inversion


def _executor(release: Any, surface: str) -> TextLeaf | Composer | None:
    """The executor that renders one surface under a release: a row, a document's message, or ``None``."""
    provided = release.provider(surface)
    if provided is None:
        return None
    renderable = release.renderable(provided[0])
    return renderable.messages[provided[1]] if isinstance(renderable, Document) else renderable


def _surface_of(name: str) -> str | None:
    """A piece's surface: the name itself, or the one surface an root entry of an older corpus provides."""
    if name in E.SURFACES:
        return name
    entry = E.ENTRIES.get(name)
    return entry.surfaces[0] if entry is not None and len(entry.surfaces) == 1 else None


def _text(release: Any, surface: str, params: Mapping[str, Any]) -> str | None:
    try:
        return release.surface(*surface.split(".", 1), **params) if release.has_surface(*surface.split(".", 1)) else None
    except (KeyError, RenderError, ValueError, TypeError):
        return None


def render_sample(release: Any, sample: Sample) -> str | None:
    """Render a sample's pieces under a release and concatenate them.

    Args:
        release (Release): The release.
        sample (Sample): The sample.

    Returns:
        str | None: The prefix plus the text, or ``None`` when the release lacks one of the surfaces or rejects
            the parameters.
    """
    out = [sample.prefix]
    for name, params in sample.pieces:
        surface = _surface_of(name)
        text = _text(release, surface, params) if surface is not None else None
        if text is None:
            return None
        out.append(text)
    return "".join(out)


def _with_sentinels(value: Any, slots: list[tuple[str, ...]], path: tuple[str, ...]) -> Any:
    if isinstance(value, Unknown):
        slots.append(path)
        return f"{_SENTINEL}U{len(slots) - 1}{_SENTINEL}"
    if isinstance(value, Mapping):
        return {key: _with_sentinels(item, slots, path + (str(key),)) for key, item in value.items()}
    if isinstance(value, list):
        return [_with_sentinels(item, slots, path + (str(index),)) for index, item in enumerate(value)]
    return value


def _assign(params: Any, path: tuple[str, ...], value: str) -> Any:
    if not path:
        return value
    head, rest = path[0], path[1:]
    if isinstance(params, list):
        index = int(head)
        return [(_assign(item, rest, value) if position == index else item) for position, item in enumerate(params)]
    return {key: (_assign(item, rest, value) if key == head else item) for key, item in params.items()}


def invert_by_sentinels(renderable: TextLeaf | Composer, params: Mapping[str, Any], text: str) -> dict[str, Any] | None:
    """Recover unknown string parameters of a render whose other parameters are known.

    The renderable is rendered with a unique sentinel in place of every :data:`UNKNOWN`; the
    output splits into literal segments around the sentinels. The real text must start with the
    first segment; every later segment is located leftmost, and the gap before it becomes the
    unknown's value. The result is verified by re-rendering, so an ambiguous alignment fails
    rather than returning a guess.

    Args:
        renderable (TextLeaf | Composer): Executor of the entry.
        params (Mapping[str, Any]): Parameters with :data:`UNKNOWN` where a string value is to be recovered
            (top-level or nested inside lists and records).
        text (str): The real message bytes.

    Returns:
        dict[str, Any] | None: Fully known parameters that re-render to ``text``, or ``None``.
    """
    slots: list[tuple[str, ...]] = []
    seeded = _with_sentinels(dict(params), slots, ())
    try:
        rendered = "".join(piece for piece, _src in renderable.render_chunks(RenderContext(), seeded))
    except (RenderError, ValueError, TypeError, KeyError):
        return None
    if not slots:
        return dict(seeded) if rendered == text else None
    pieces = _SENTINEL_RE.split(rendered)
    literals = pieces[0::2]
    order = [int(index) for index in pieces[1::2]]
    if not text.startswith(literals[0]):
        return None
    position = len(literals[0])
    values: dict[int, str] = {}
    last = len(order) - 1
    for step, (slot_index, literal) in enumerate(zip(order, literals[1:])):
        if step == last:
            # the final literal is anchored at the end: a value may itself contain it
            found = len(text) - len(literal) if text.endswith(literal) and len(text) - len(literal) >= position else -1
        elif literal:
            found = text.find(literal, position)
        else:
            found = -1  # two adjacent unknowns have no boundary
        if found < 0:
            return None
        value = text[position:found]
        if slot_index in values and values[slot_index] != value:
            return None
        values[slot_index] = value
        position = found + len(literal)
    if position != len(text):
        return None
    result: Any = dict(params)
    for index, path in enumerate(slots):
        result = _assign(result, path, values.get(index, ""))
    try:
        check = "".join(piece for piece, _src in renderable.render_chunks(RenderContext(), result))
    except (RenderError, ValueError, TypeError, KeyError):
        return None
    return result if check == text else None


def _invert_leaf(leaf: TextLeaf, text: str, int_range: range, surface: str | None = None) -> dict[str, Any] | None:
    """Recover the slots of a text leaf: none, ints by search, or exactly one string by sentinel inversion."""
    if not leaf.slots:
        return {} if leaf.render() == text else None
    declared = E.SURFACES[surface].params if surface in E.SURFACES else {}
    kinds = {slot: ("str" if declared.get(slot, "str") == "str" else "int") for slot in leaf.slots}
    ints = [slot for slot, kind in kinds.items() if kind == "int"]
    strs = [slot for slot, kind in kinds.items() if kind == "str"]
    if len(ints) > 2 or len(strs) > 1:
        return None
    from itertools import product

    for combo in product(int_range, repeat=len(ints)):
        params: dict[str, Any] = dict(zip(ints, combo))
        for slot in strs:
            params[slot] = UNKNOWN
        found = invert_by_sentinels(leaf, params, text)
        if found is not None:
            return found
    return None


def tool_texts(content: str) -> list[tuple[str, str]]:
    """The prompt text a ``tool`` message carries, with the sample kind it would be recorded as.

    A message that is not a tool-result envelope is prompt text in its own right (``"tool"``: the
    final-check bounce, the judge's literal guard); an envelope contributes each string of the
    ``notes`` list in its JSON body (``"note"``: the repeat-query and empty-result notes) and nothing
    when it has none or its body is an ``ERROR:`` line.

    Args:
        content (str): The message content as sent to the model.

    Returns:
        list[tuple[str, str]]: ``(kind, text)`` pairs, possibly empty.
    """
    match = _TOOL_ENVELOPE_RE.fullmatch(content)
    if match is None:
        return [("tool", content)]
    try:
        body = json.loads(match.group("body"))
    except ValueError:
        return []
    notes = body.get("notes") if isinstance(body, Mapping) else None
    return [("note", note) for note in notes if isinstance(note, str)] if isinstance(notes, list) else []


class _Builder:
    def __init__(self, release: Any, *, max_rows: Sequence[int], int_range: range) -> None:
        self.release = release
        self.max_rows = tuple(max_rows)
        self.int_range = int_range
        self.candidate_user_surfaces = sorted((set(E.closure("candidate", None)) | set(E.closure("candidate", "grain"))) - _CANDIDATE_NOT_MESSAGES)
        self.judge_message_surfaces = sorted(set(E.closure("judge", None)) - _JUDGE_NOT_MESSAGES)

    def system(self, text: str) -> tuple[tuple[tuple[str, Mapping[str, Any]], ...], str | None] | None:
        """Match a candidate system message: ``candidate.system(max_rows)`` alone or plus one route block."""
        for rows in self.max_rows:
            base = _text(self.release, "candidate.system", {"max_rows": rows})
            if base is None:
                continue
            if text == base:
                return ((("candidate.system", {"max_rows": rows}),), None)
            for route, surface in _ROUTE_SURFACE.items():
                block = _text(self.release, surface, {})
                if block is not None and text == base + block:
                    return ((("candidate.system", {"max_rows": rows}), (surface, {})), route)
        return None

    def first_user(self, text: str, question: str) -> tuple[tuple[str, Mapping[str, Any]], ...] | None:
        """Match the first user message, ``candidate.user`` with the question; only the frames of no hits and no hooks are invertible."""
        synonyms = _text(self.release, "shared.synonyms", {"hits": []})
        if synonyms is None:
            return None
        params = {"question": question, "synonyms": synonyms, "rule_hooks": ""}
        if _text(self.release, "candidate.user", params) == text:
            return (("candidate.user", params),)
        return None

    def _one_surface(self, text: str, surfaces: Sequence[str]) -> tuple[tuple[str, Mapping[str, Any]], ...] | None:
        """Match a whole message against one surface: a text leaf by inversion, a parameter-less composition by equality."""
        for surface in surfaces:
            executor = _executor(self.release, surface)
            if executor is None:
                continue
            if isinstance(executor, TextLeaf):
                params = _invert_leaf(executor, text, self.int_range, surface)
                if params is not None:
                    return ((surface, params),)
            elif not executor.params and _text(self.release, surface, {}) == text:
                return ((surface, {}),)
        return None

    def judge_message(self, text: str) -> tuple[tuple[str, Mapping[str, Any]], ...] | None:
        """Match a judge message after the first: the submit nudge or the literal guard."""
        return self._one_surface(text, self.judge_message_surfaces)

    def user(self, text: str) -> tuple[tuple[str, Mapping[str, Any]], ...] | None:
        """Match a later candidate message (user or tool): one candidate-visible surface, alone or followed by the rule-hook frame."""
        hooks = "shared.rule_hooks"
        found = self._one_surface(text, self.candidate_user_surfaces)
        if found is not None:
            return found
        hook_leaf = _executor(self.release, hooks)
        if not isinstance(hook_leaf, TextLeaf):
            return None
        for surface in self.candidate_user_surfaces:
            executor = _executor(self.release, surface)
            if executor is None or (isinstance(executor, Composer) and executor.params):
                continue
            declared = E.SURFACES[surface].params
            if isinstance(executor, TextLeaf) and any(declared.get(slot, "str") == "str" for slot in executor.slots):
                continue
            candidates = [{}] if not getattr(executor, "slots", ()) else None
            if candidates is None:
                from itertools import product

                slots = list(executor.slots)
                candidates = [dict(zip(slots, combo)) for combo in product(self.int_range, repeat=len(slots))] if len(slots) <= 1 else []
            for params in candidates:
                head = _text(self.release, surface, params)
                if head is None or not text.startswith(head) or len(head) >= len(text):
                    continue
                body = _invert_leaf(hook_leaf, text[len(head):], self.int_range, hooks)
                if body is not None:
                    return ((surface, params), (hooks, body))
        return None

    def judge_user(self, row: Mapping[str, Any], text: str) -> tuple[tuple[str, Mapping[str, Any]], ...] | None:
        """Invert the judge user message from the row's question, candidate summaries and result-group notes."""
        surface = "judge.user"
        renderable = _executor(self.release, surface)
        if not isinstance(renderable, Composer):
            return None
        groups = {int(match.group(1)): (match.group(2), int(match.group(3) or 1)) for match in _GROUP_RE.finditer(text)}
        candidates = []
        for index, item in enumerate(row.get("candidates") or [], 1):
            if not isinstance(item, Mapping):
                return None
            error = str(item.get("error") or "")
            label, size = groups.get(index, ("", 0))
            candidates.append({
                "error": error,
                "group_size": size if not error else 0,
                "group_label": label if not error else "",
                "sql_fenced": "" if error else UNKNOWN,
                "n_rows": int(item.get("row_count") or 0) if not error else 0,
                "preview": "" if error else UNKNOWN,
            })
        question = str(row.get("question") or "")
        for synonyms, rules in ((UNKNOWN, UNKNOWN), (UNKNOWN, ""), ("", UNKNOWN), ("", "")):
            params = {"question": question, "synonyms": synonyms, "candidates": candidates, "rules": rules}
            found = invert_by_sentinels(renderable, params, text)
            if found is not None:
                return ((surface, found),)
        return None


def build(run_dir: Path, release: Any, *, max_rows: Sequence[int] = (64, 200), int_range: range = range(0, 121)) -> Corpus:
    """Rebuild every real prompt input of a run that the release re-renders byte-exactly.

    Args:
        run_dir (Path): Run directory (see :func:`read_rows`).
        release (Release): The release the run is attributed to (root for pre-migration runs).
        max_rows (Sequence[int]): ``max_rows`` values to try for the candidate system prompt.
        int_range (range): Search range for integer slots (minutes, counters).

    Returns:
        Corpus: Kept samples plus the count of dropped messages by reason.
    """
    run_dir = Path(run_dir)
    corpus = Corpus(run_dir.name, release.id)
    builder = _Builder(release, max_rows=max_rows, int_range=int_range)

    def keep(sample: Sample) -> None:
        text = render_sample(release, sample)
        if text is not None and sha256_hex(text)[:16] == sample.sha16:
            corpus.samples.append(sample)
        else:
            corpus.dropped["re-render-mismatch"] = corpus.dropped.get("re-render-mismatch", 0) + 1

    def drop(reason: str) -> None:
        corpus.dropped[reason] = corpus.dropped.get(reason, 0) + 1

    for row in read_rows(run_dir):
        qid, db_id = str(row.get("id")), str(row.get("db_id", ""))
        question = str(row.get("question") or "")
        for debug in row.get("candidates_debug") or []:
            messages = debug.get("messages") if isinstance(debug, Mapping) else None
            if not isinstance(messages, list):
                drop("candidate-transcript-absent")
                continue
            route: str | None = None
            user_seen = 0
            for message in messages:
                if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
                    continue
                content, role = message["content"], message.get("role")
                if role == "system":
                    found = builder.system(content)
                    if found is None:
                        drop("candidate-system-unmatched")
                        continue
                    pieces, route = found
                    keep(Sample(qid, db_id, "candidate", route, "system", pieces, sha256_hex(content)[:16]))
                elif role == "user":
                    user_seen += 1
                    pieces = builder.first_user(content, question) if user_seen == 1 else builder.user(content)
                    if pieces is None:
                        drop("candidate-first-user-not-invertible" if user_seen == 1 else "candidate-user-unmatched")
                        continue
                    keep(Sample(qid, db_id, "candidate", route, "user", pieces, sha256_hex(content)[:16]))
                elif role == "tool":
                    for kind, text in tool_texts(content):
                        pieces = builder.user(text)
                        if pieces is None:
                            drop(f"candidate-{kind}-unmatched")
                            continue
                        keep(Sample(qid, db_id, "candidate", route, kind, pieces, sha256_hex(text)[:16]))
        judge = row.get("judge_debug")
        if isinstance(judge, Mapping) and isinstance(judge.get("user_msg"), str):
            pieces = builder.judge_user(row, judge["user_msg"])
            if pieces is None:
                drop("judge-user-not-invertible")
            else:
                keep(Sample(qid, db_id, "judge", judge.get("style"), "user", pieces, sha256_hex(judge["user_msg"])[:16]))
        elif isinstance(judge, Mapping):
            drop("judge-user-absent")
        if isinstance(judge, Mapping) and isinstance(judge.get("messages"), list):
            user_seen = 0
            for message in judge["messages"]:
                if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
                    continue
                content, role = message["content"], message.get("role")
                if role == "user":
                    user_seen += 1
                    if user_seen == 1:
                        continue  # the first user message is user_msg, handled above
                    texts = [("user", content)]
                elif role == "tool":
                    texts = tool_texts(content)
                else:
                    continue
                for kind, text in texts:
                    pieces = builder.judge_message(text)
                    if pieces is None:
                        drop(f"judge-{kind}-unmatched")
                        continue
                    keep(Sample(qid, db_id, "judge", judge.get("style"), kind, pieces, sha256_hex(text)[:16]))
        if isinstance(row.get("reviewer_debug"), Mapping):
            drop("reviewer-transcript-absent")
    return corpus


# ---------------------------------------------------------------------------------------
# Stats


@dataclass(frozen=True)
class SampleDiff:
    """One sample whose bytes differ between two releases."""

    sample: Sample
    text_a: str
    text_b: str
    diff: str


@dataclass(frozen=True)
class CorpusStats:
    """``k of N changed`` between two releases, with samples.

    Attributes:
        n (int): Samples rendered under both releases.
        k (int): Samples whose bytes differ.
        skipped (int): Samples one release could not render (surface absent or parameters rejected).
        per_entry (Mapping[str, tuple[int, int]]): Leading surface to ``(n, k)``.
        samples (tuple[SampleDiff, ...]): Up to the requested number of changed samples with unified diffs.
        uncovered (tuple[str, ...]): Changed surfaces no rendered sample reaches: the corpus says nothing about them.
    """

    n: int
    k: int
    skipped: int
    per_entry: Mapping[str, tuple[int, int]]
    samples: tuple[SampleDiff, ...]
    uncovered: tuple[str, ...] = ()

    def summary(self) -> str:
        """One line for the gate report.

        Returns:
            str: ``"k / N real inputs change"`` plus the per-surface breakdown, and the changed surfaces without coverage.
        """
        parts = ", ".join(f"{entry} {k}/{n}" for entry, (n, k) in sorted(self.per_entry.items()) if k)
        text = f"{self.k} / {self.n} real inputs change" + (f" ({parts})" if parts else "") + (f"; {self.skipped} skipped" if self.skipped else "")
        if self.uncovered:
            text += "; no real-input coverage for " + ", ".join(self.uncovered)
        return text


def stats(corpus: Corpus, release_a: Any, release_b: Any, *, samples: int = 3, changed: Iterable[str] = ()) -> CorpusStats:
    """Count the corpus samples whose bytes differ between two releases.

    Args:
        corpus (Corpus): Samples to render.
        release_a (Release): The parent release.
        release_b (Release): The child release.
        samples (int): How many changed samples to return with diffs.
        changed (Iterable[str]): Surfaces whose static text differs between the releases; those no rendered sample
            reaches are reported as uncovered instead of hiding behind ``0 / N``.

    Returns:
        CorpusStats: The counts, the per-surface breakdown, the sample diffs and the uncovered changed surfaces.
    """
    n = k = skipped = 0
    per_entry: dict[str, list[int]] = {}
    diffs: list[SampleDiff] = []
    reached: set[str] = set()
    for sample in corpus.samples:
        text_a, text_b = render_sample(release_a, sample), render_sample(release_b, sample)
        if text_a is None or text_b is None:
            skipped += 1
            continue
        n += 1
        reached.update(E.surfaces_of(name for name, _params in sample.pieces))
        lead = (_surface_of(sample.pieces[0][0]) or sample.pieces[0][0]) if sample.pieces else "?"
        bucket = per_entry.setdefault(lead, [0, 0])
        bucket[0] += 1
        if text_a != text_b:
            k += 1
            bucket[1] += 1
            if len(diffs) < samples:
                diff = "".join(difflib.unified_diff(text_a.splitlines(True), text_b.splitlines(True), f"{sample.qid} {release_a.id}", f"{sample.qid} {release_b.id}"))
                diffs.append(SampleDiff(sample, text_a, text_b, diff))
    return CorpusStats(n, k, skipped, {entry: (n_e, k_e) for entry, (n_e, k_e) in per_entry.items()}, tuple(diffs), tuple(sorted(set(changed) - reached)))
