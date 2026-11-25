"""Field, name, reason and size policies every prompt row must satisfy.

The policies are pure functions over strings and JSON values. The runtime applies them at
load time to every row of a release, and the author-side gate applies them to agent
proposals before anything is published. A text row's field policy is checked on the
executable template (``canon.escape(text, slots)``) with ``string.Formatter().parse`` so no
attribute, index, positional or format-spec field can ever reach ``str.format``; a doc row's
text is checked here for its encoding and size and parsed by :mod:`doc`'s closed grammar.
"""
from __future__ import annotations

import keyword
import re
import string
from typing import Any, Iterable, Mapping

from .canon import escape, unescape

__all__ = [
    "CONTENT_KEY_RE",
    "DATA_OWNERS",
    "DATA_TEXT_MAX_BYTES",
    "DOC_MAX_CHARS",
    "EACH_MAX",
    "KINDS",
    "LIT_MAX_CHARS",
    "META_KEYS",
    "NAME_MAX_CHARS",
    "NAME_SEGMENT_RE",
    "NF_MAX_NODES",
    "OUTPUT_MAX_CHARS",
    "PARAM_MAX_CHARS",
    "PARTS_MAX_DEPTH",
    "PROPOSAL_RE",
    "REASON_MIN_CHARS",
    "RELEASE_RE",
    "RESERVED_IDENTIFIERS",
    "RUN_NAME_RE",
    "SEP_MAX_CHARS",
    "SLOT_RE",
    "SPEC_MAX_BYTES",
    "SPEC_MAX_DEPTH",
    "SPEC_MAX_NODES",
    "TAG_START_RE",
    "TEXT_MAX_BYTES",
    "VERSION_RE",
    "PolicyError",
    "check_content_key",
    "check_document",
    "check_identifier",
    "check_meta_keys",
    "check_name",
    "check_proposal_id",
    "check_reason",
    "check_release_id",
    "check_run_name",
    "check_template",
    "check_text",
    "check_version",
    "text_max_bytes",
]

#: One dotted-name segment, the same rule HeavenBase's ``check_id`` enforces on prompt names.
NAME_SEGMENT_RE = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")
#: A slot name, spec identifier or param path segment.
SLOT_RE = re.compile(r"^[a-z][a-z0-9_]*$")
#: Text that starts like a document tag (``rubiksql.doc/1``): ``{{`` then, after optional spaces or tabs, a
#: section sigil (``#``, ``/``, ``^``), a Mustache comment, partial or unescape sigil (``!``, ``>``, ``&``), or
#: ``message``/``else``. In a doc row it must be a well-formed tag, so a Mustache habit such as ``{{ #if x }}``
#: or ``{{! note }}`` is refused instead of reaching the model as text; a text row has no tags, so it may not
#: contain one: the two grammars then read every text they both accept the same way, which is what lets a
#: surface's identity be its text as written, whichever kind of row carries it.
TAG_START_RE = re.compile(r"\{\{[ \t]*(?:[#/^!>&]|message|else)")
#: HB ``version`` column of a prompt row: the first 16 hex digits of its content key.
VERSION_RE = re.compile(r"^[0-9a-f]{16}$")
#: A release id.
RELEASE_RE = re.compile(r"^r[0-9a-f]{16}$")
#: A proposal id.
PROPOSAL_RE = re.compile(r"^p[0-9a-f]{16}$")
#: A full content key.
CONTENT_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
#: A name that becomes the stem of ``attestations/<name>.json`` (``gate --record NAME``, such as ``r0-gate``).
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
#: Compiled loop-accumulator names, which a spec identifier must not shadow.
_ACCUMULATOR_RE = re.compile(r"^v[0-9]+$")

#: Identifiers a slot or spec variable may never use: HB's translation arguments, the
#: compiled ``compose(parts, params)`` signature, the three builtins the compiled code calls
#: (``str`` formats a document's int and bool slots), and every Python keyword.
RESERVED_IDENTIFIERS = frozenset({"tr", "lang", "parts", "params", "enumerate", "len", "str"} | set(keyword.kwlist))
#: Closed key table of ``manifest.meta``; it avoids every ``Prompt(...)`` constructor argument
#: because HB feeds meta back into the constructor on load.
META_KEYS = frozenset({"schema", "kind", "text", "slots", "spec", "parts", "params", "content_key", "origin", "author", "parent", "proposal"})
#: The row kinds: ``text`` (a template prompt), ``spec`` (a composition of parts, r0's assembly rows) and
#: ``doc`` (one document with message boundaries and blocks, a function prompt compiled from its text).
KINDS = ("text", "spec", "doc")

NAME_MAX_CHARS = 128
TEXT_MAX_BYTES = 64 * 1024
#: First name segments of the data rows: text rows whose JSON the code reads (``config.knobs``, ``rules.general``)
#: rather than a text a model receives. The general rule corpus alone is larger than a prompt text may be.
DATA_OWNERS = ("config", "rules", "tools")
#: Longest text a data row may hold, in bytes.
DATA_TEXT_MAX_BYTES = 1024 * 1024
#: Longest document a doc row may hold, in characters. The 16 KiB ``SPEC_MAX_BYTES`` applies to stored
#: specs only; the spec a document derives is bounded by this size and the node, depth and ``each.max`` limits.
DOC_MAX_CHARS = 256 * 1024
SPEC_MAX_BYTES = 16 * 1024
SPEC_MAX_DEPTH = 8
SPEC_MAX_NODES = 1500
LIT_MAX_CHARS = 64
SEP_MAX_CHARS = 8
EACH_MAX = 64
PARTS_MAX_DEPTH = 6
#: The per-string assumption of the static output bound: every string parameter or slot site is
#: charged this many characters, and the gate holds agent-written cells to it. It is checked
#: statically only — the runtime render path accepts a data parameter (a result preview, a rule
#: block) of any length, exactly as the f-strings of 9140477 did, and enforces ``OUTPUT_MAX_CHARS``
#: alone.
PARAM_MAX_CHARS = 64 * 1024
#: Static output bound the gate requires and hard cap the interpreter enforces, in characters.
OUTPUT_MAX_CHARS = 4 * 1024 * 1024
#: Most logical nodes a normal form may expand to. Parts are inlined at every reference, so a
#: spec that references a part many times per level fans out geometrically; the bound stops the
#: fan-out when its output grows, this stops it when the output does not (zero-length loops and
#: branches), before the digest has to serialise it. r0's largest form is a few thousand nodes.
NF_MAX_NODES = 1 << 20
REASON_MIN_CHARS = 20

#: Words that say nothing about why a version exists; a reason made only of these is boilerplate.
_BOILERPLATE_WORDS = frozenset({
    "a", "an", "the", "this", "that", "some", "and", "or", "of", "to", "for", "in", "on", "with", "it",
    "update", "updated", "updates", "updating", "fix", "fixes", "fixed", "fixing", "change", "changes", "changed",
    "improve", "improved", "improves", "improvement", "improvements", "better", "tweak", "tweaks", "tweaked",
    "minor", "small", "misc", "wip", "todo", "test", "testing", "initial", "version", "new", "cleanup", "refactor",
    "prompt", "prompts", "text", "wording", "edit", "edits", "edited", "rewrite", "rewritten", "polish", "polished",
})


class PolicyError(ValueError):
    """A prompt row, name, reason or template violates a policy; the message names the rule."""


def check_identifier(value: Any, label: str = "identifier") -> str:
    """Validate a slot name, spec variable or param path segment.

    Args:
        value (Any): Candidate identifier.
        label (str): Role named in the error message.

    Returns:
        str: The identifier unchanged.

    Raises:
        PolicyError: If it is not ``^[a-z][a-z0-9_]*$``, is reserved (``tr``, ``lang``, ``parts``,
            ``params``, ``enumerate``, ``len``, a Python keyword) or looks like a compiled accumulator ``v0``.
    """
    if not isinstance(value, str) or not SLOT_RE.fullmatch(value):
        raise PolicyError(f"invalid {label} {value!r}; use lower-case letters, digits and '_' starting with a letter")
    if value in RESERVED_IDENTIFIERS or _ACCUMULATOR_RE.fullmatch(value):
        raise PolicyError(f"reserved {label} {value!r}")
    return value


def check_name(name: Any) -> str:
    """Validate a dotted prompt row name before it is used as a path component or HB name.

    Args:
        name (Any): Candidate name such as ``rubiksql.judge.user``.

    Returns:
        str: The name unchanged.

    Raises:
        PolicyError: If any segment fails the HB id rule, the name is empty or exceeds 128 characters.
    """
    if not isinstance(name, str) or not name or len(name) > NAME_MAX_CHARS:
        raise PolicyError(f"invalid prompt name {name!r}; expected 1..{NAME_MAX_CHARS} characters of dotted segments")
    for segment in name.split("."):
        if not NAME_SEGMENT_RE.fullmatch(segment):
            raise PolicyError(f"invalid prompt name {name!r}: segment {segment!r} must match {NAME_SEGMENT_RE.pattern}")
    return name


def check_version(version: Any) -> str:
    """Validate a row version (16 hex digits) before it becomes a file name.

    Args:
        version (Any): Candidate version.

    Returns:
        str: The version unchanged.

    Raises:
        PolicyError: If it is not exactly 16 lower-case hex digits.
    """
    if not isinstance(version, str) or not VERSION_RE.fullmatch(version):
        raise PolicyError(f"invalid prompt version {version!r}; expected 16 hex digits")
    return version


def check_release_id(ref: Any) -> str:
    """Validate a release id (``r`` + 16 hex digits).

    Args:
        ref (Any): Candidate release id.

    Returns:
        str: The id unchanged.

    Raises:
        PolicyError: If the id does not match ``^r[0-9a-f]{16}$``.
    """
    if not isinstance(ref, str) or not RELEASE_RE.fullmatch(ref):
        raise PolicyError(f"invalid release id {ref!r}; expected 'r' followed by 16 hex digits")
    return ref


def check_proposal_id(ref: Any) -> str:
    """Validate a proposal id (``p`` + 16 hex digits).

    Args:
        ref (Any): Candidate proposal id.

    Returns:
        str: The id unchanged.

    Raises:
        PolicyError: If the id does not match ``^p[0-9a-f]{16}$``.
    """
    if not isinstance(ref, str) or not PROPOSAL_RE.fullmatch(ref):
        raise PolicyError(f"invalid proposal id {ref!r}; expected 'p' followed by 16 hex digits")
    return ref


def check_run_name(name: Any) -> str:
    """Validate a run or attestation name before it becomes a file name under ``attestations/``.

    Args:
        name (Any): Candidate name such as ``r0-gate`` or ``exp_full1534_20260808``.

    Returns:
        str: The name unchanged.

    Raises:
        PolicyError: If it is not a str matching ``^[A-Za-z0-9][A-Za-z0-9._-]*$`` (no separators, so no
            path can be built from it).
    """
    if not isinstance(name, str) or not RUN_NAME_RE.fullmatch(name):
        raise PolicyError(f"invalid run name {name!r}; expected letters, digits, '.', '_' and '-' starting with a letter or digit")
    return name


def check_content_key(key: Any) -> str:
    """Validate a full 64-hex content key.

    Args:
        key (Any): Candidate content key.

    Returns:
        str: The key unchanged.

    Raises:
        PolicyError: If it is not exactly 64 lower-case hex digits.
    """
    if not isinstance(key, str) or not CONTENT_KEY_RE.fullmatch(key):
        raise PolicyError(f"invalid content key {key!r}; expected 64 hex digits")
    return key


def check_template(template: str, slots: Iterable[str]) -> None:
    """Check an executable template's fields with ``string.Formatter().parse``.

    Args:
        template (str): The ``str.format`` template as stored in the capsule source.
        slots (Iterable[str]): Declared slot names.

    Returns:
        None: This function does not return a value.

    Raises:
        PolicyError: If the template is malformed, names a field that is not a declared slot, uses a
            positional, auto-numbered, attribute (``x.y``) or index (``x[0]``) field, uses a conversion
            other than none, ``!r`` or ``!s``, carries a format spec, or omits a declared slot.
    """
    declared = list(slots)
    seen: set[str] = set()
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as error:
        raise PolicyError(f"malformed template: {error}") from error
    for _literal, field_name, format_spec, conversion in parsed:
        if field_name is None:
            continue
        if not SLOT_RE.fullmatch(field_name):
            raise PolicyError(f"template field {field_name!r} is not a plain slot name (positional, attribute and index fields are forbidden)")
        if field_name in RESERVED_IDENTIFIERS:
            raise PolicyError(f"template field {field_name!r} is a reserved name")
        if field_name not in declared:
            raise PolicyError(f"template field {field_name!r} is not a declared slot; declared: {declared}")
        if conversion not in (None, "r", "s"):
            raise PolicyError(f"template field {field_name!r} uses conversion {conversion!r}; only !r and !s are allowed")
        if format_spec:
            raise PolicyError(f"template field {field_name!r} carries a format spec {format_spec!r}; format specs are forbidden")
        seen.add(field_name)
    missing = [slot for slot in declared if slot not in seen]
    if missing:
        raise PolicyError(f"declared slots never used in the template: {missing}")


def text_max_bytes(name: str | None) -> int:
    """The longest text a text row of this name may hold, in bytes.

    Args:
        name (str | None): The row name; ``None`` when the caller has none, which gets the prompt-text limit.

    Returns:
        int: :data:`DATA_TEXT_MAX_BYTES` for a data row (first segment in :data:`DATA_OWNERS`), else :data:`TEXT_MAX_BYTES`.
    """
    return DATA_TEXT_MAX_BYTES if name is not None and name.split(".", 1)[0] in DATA_OWNERS else TEXT_MAX_BYTES


def check_text(text: Any, slots: Any, *, name: str | None = None) -> str:
    """Apply the full text-row policy and return the executable template.

    Args:
        text (Any): Human-readable text of the row (``meta.text``).
        slots (Any): Declared slot names (``meta.slots``), a list of unique identifiers.
        name (str | None): The row name, which sets the size limit (:func:`text_max_bytes`).

    Returns:
        str: ``escape(text, slots)``, the template HeavenBase stores.

    Raises:
        PolicyError: If the text is not a str, is not UTF-8 encodable, contains ``\\r``, NUL, U+001E or the start
            of a document tag (:data:`TAG_START_RE`), exceeds 64 KiB (1 MiB for a data row), a slot name is invalid or
            duplicated, the escaped template fails :func:`check_template`, or escape/unescape does not round-trip.
    """
    if not isinstance(text, str):
        raise PolicyError(f"text must be a str, got {type(text).__name__}")
    if not isinstance(slots, list) or any(not isinstance(slot, str) for slot in slots):
        raise PolicyError("slots must be a list of strings")
    if len(set(slots)) != len(slots):
        raise PolicyError(f"duplicate slot names in {slots}")
    for slot in slots:
        check_identifier(slot, "slot")
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise PolicyError(f"text is not UTF-8 encodable: {error}") from error
    limit = text_max_bytes(name)
    if len(encoded) > limit:
        raise PolicyError(f"text is {len(encoded)} bytes; the limit is {limit}")
    if "\r" in text:
        raise PolicyError("text contains a carriage return; only LF newlines are allowed")
    if "\x00" in text:
        raise PolicyError("text contains a NUL character")
    if "\x1e" in text:
        raise PolicyError("text contains U+001E (RECORD SEPARATOR), the character a document's {{message}} renders as")
    tag = TAG_START_RE.search(text)
    if tag is not None:
        raise PolicyError(f"text contains {tag.group(0)!r}, the start of a document tag; a text row has no tags, so a text with blocks or message boundaries is a doc row")
    template = escape(text, slots)
    check_template(template, slots)
    if unescape(template, slots) != text:
        raise PolicyError("escape/unescape round trip does not reproduce the text")
    return template


def check_document(text: Any, *, what: str = "document") -> str:
    """Apply the encoding and size policy of a doc row's text (``meta.text``).

    The grammar itself (tags, slots, blocks) is checked by :func:`doc.parse_document`; this is the part
    that needs no parameter declarations. The document grammar applies the same character rules to the
    decoded value of an ``{{#each}}`` tag's ``sep`` attribute, whose JSON escapes could otherwise carry what
    the document itself may not.

    Args:
        text (Any): The document as stored (or a decoded ``sep`` value).
        what (str): What the text is, for the error messages.

    Returns:
        str: The text unchanged.

    Raises:
        PolicyError: If the text is not a str, is not UTF-8 encodable, exceeds ``DOC_MAX_CHARS`` characters,
            contains a carriage return or NUL, or contains U+001E, the character ``{{message}}`` renders as.
    """
    if not isinstance(text, str):
        raise PolicyError(f"{what} must be a str, got {type(text).__name__}")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise PolicyError(f"{what} is not UTF-8 encodable: {error}") from error
    if len(text) > DOC_MAX_CHARS:
        raise PolicyError(f"{what} is {len(text)} characters; the limit is {DOC_MAX_CHARS}")
    if "\r" in text:
        raise PolicyError(f"{what} contains a carriage return; only LF newlines are allowed")
    if "\x00" in text:
        raise PolicyError(f"{what} contains a NUL character")
    if "\x1e" in text:
        raise PolicyError(f"{what} contains U+001E (RECORD SEPARATOR), the character {{{{message}}}} renders as; a message boundary is a {{{{message}}}} line")
    return text


def check_reason(reason: Any, parent_reason: str | None = None) -> str:
    """Apply the publication reason policy.

    Args:
        reason (Any): ``manifest.reason`` of the row being published.
        parent_reason (str | None): The parent version's reason, which this one must not repeat.

    Returns:
        str: The stripped reason.

    Raises:
        PolicyError: If the reason is not a non-empty str of at least 20 characters with three distinct
            words, uses only boilerplate vocabulary ("improve the prompt wording"), or equals the parent's reason.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise PolicyError("reason must be a non-empty string")
    stripped = reason.strip()
    if len(stripped) < REASON_MIN_CHARS:
        raise PolicyError(f"reason is {len(stripped)} characters; at least {REASON_MIN_CHARS} are required")
    words = {word for word in re.findall(r"[a-zA-Z一-鿿]+", stripped.lower())}
    if len(words) < 3:
        raise PolicyError("reason must contain at least three distinct words")
    if words <= _BOILERPLATE_WORDS:
        raise PolicyError(f"reason {stripped!r} is boilerplate; say why this version exists")
    if parent_reason is not None and stripped.lower() == parent_reason.strip().lower():
        raise PolicyError("reason repeats the parent version's reason")
    return stripped


def check_meta_keys(meta: Mapping[str, Any]) -> None:
    """Reject a ``manifest.meta`` with keys outside the closed table.

    Args:
        meta (Mapping[str, Any]): The row's ``manifest.meta``.

    Returns:
        None: This function does not return a value.

    Raises:
        PolicyError: If meta is not a mapping, has an unknown key, or lacks ``schema``/``kind``/``content_key``.
    """
    if not isinstance(meta, Mapping):
        raise PolicyError(f"manifest.meta must be a mapping, got {type(meta).__name__}")
    unknown = sorted(set(meta) - META_KEYS)
    if unknown:
        raise PolicyError(f"manifest.meta has keys outside the closed table: {unknown}")
    for required in ("schema", "kind", "content_key"):
        if required not in meta:
            raise PolicyError(f"manifest.meta lacks {required!r}")
