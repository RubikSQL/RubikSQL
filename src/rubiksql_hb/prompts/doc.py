"""rubiksql.doc/1: the document format of doc rows — its parser, its compiler and its pretty-printer.

A *document* is one readable text for everything a role receives from one row: the system
message, the first user message, and the dynamic blocks inside them. It is plain UTF-8 text
with two kinds of markup:

- **Slots.** ``{name}`` for a declared parameter, ``{var.field}`` for a field of a loop variable
  or of a record parameter, ``{i}`` for a loop index. Only names in scope are slots; every
  other brace is literal text, exactly as in a text row. A slot's value must be a ``str``,
  ``int`` or ``bool`` (the last two print as ``str()`` does, like a text row's slot).
- **Tags.** ``{{message}}``, ``{{#each …}}`` / ``{{/each}}``, ``{{#if …}}`` / ``{{else}}`` /
  ``{{/if}}``. A tag that stands alone on its line — nothing else on the line, a line break or
  the end of the text after it — is removed together with that line break (Mustache's
  "standalone" rule, no surrounding whitespace allowed), so a document keeps one construct per
  line and stays readable; a tag anywhere else is inline and removed alone.

  - ``{{#each <list> as <var> [index=<name>] [start=0|1] max=<int> [sep="<JSON string>"]}}``
    renders its body once per element, at most ``max`` times (the static bound needs the cap),
    with ``sep`` between two renderings; ``index`` names the position, counted from ``start``.
  - ``{{#if <ref>}}`` tests the value for truth (a non-empty string or list, a non-zero int,
    ``true``); ``{{#if <ref> > <int>}}`` compares an int. ``{{else}}`` is optional.
  - ``{{message}}`` separates two messages. It stands alone on its line at the top level, and the
    line break before its line belongs to it (as a MIME multipart boundary's CRLF does), so the
    text above the line is the previous message exactly and the text below it the next. It
    renders as one U+001E, which a document may therefore never contain literally.

The grammar is closed: an unknown or malformed tag (any text that starts like one,
:data:`policy.TAG_START_RE` — Mustache's ``{{ #if x }}``, ``{{! note }}``, ``{{> partial }}``,
``{{&x}}``, ``{{^x}}`` included), an unbalanced block, an ``{{else}}`` or ``{{message}}`` out of
place, a reference to an undeclared field or a value that cannot be text, a bad or repeated
attribute, a double-braced slot, a declared parameter no message uses — each is a
:class:`DocError` naming the line, except the last, whose parameter appears on no line. A
document derives a ``rubiksql.spec/1`` tree (literals unbounded, ``{{message}}`` the literal
U+001E). Its node and nesting bounds are counted on that tree — an ``{{#if}}`` is one level,
a body of several items another, an ``{{#each}}`` two — so a nesting error names the line of
the innermost block on the offending path (a node-count error names none). The spec
interpreter renders the tree, and :func:`spec.compile_spec` turns it into the ``compose``
source HeavenBase executes; the runtime re-derives that source from ``meta.text`` on every
load, exactly as it does for a spec row.

:func:`document_of` prints a normal form as a document and parses the result back, refusing any
form it cannot print exactly, so a composition of parts and the document written from it have
the same text: that text is the identity a keyspace hashes (:meth:`runtime.Release.keyspace`).
A text row carries no tags and may contain neither text that starts like one
(:data:`policy.TAG_START_RE`) nor U+001E, so a text that is valid both as a text row and as a
document message renders the same bytes under either grammar.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .policy import EACH_MAX, SEP_MAX_CHARS, TAG_START_RE, PolicyError, check_document, check_identifier
from .spec import Composer, SpecError, Type, check_type, compile_spec, parse_spec, type_name

__all__ = [
    "MESSAGE_SEPARATOR",
    "SCHEMA_DOC",
    "DocError",
    "Document",
    "ParsedDocument",
    "compile_document",
    "document_of",
    "parse_document",
]

SCHEMA_DOC = "rubiksql.doc/1"
#: What ``{{message}}`` renders as: one U+001E (RECORD SEPARATOR). A document never contains it literally.
MESSAGE_SEPARATOR = "\x1e"

_IDENT = r"[a-z][a-z0-9_]*"
_REF = rf"{_IDENT}(?:\.{_IDENT})?"
#: A well-formed tag. Tags never span lines; a ``sep`` value may hold braces inside its quotes.
_TAG_RE = re.compile(r'\{\{(?:(?P<boundary>message)|(?P<otherwise>else)|/(?P<close>each|if)|#(?P<open>each|if) (?P<args>(?:[^"}\n]|"(?:[^"\\\n]|\\.)*")*))\}\}')
_SLOT_RE = re.compile(rf"\{{(?P<ref>{_REF})(?P<conv>![a-z])?\}}")
_DOUBLE_SLOT_RE = re.compile(rf"\{{\{{(?P<ref>{_REF})\}}\}}")
_IF_ARGS_RE = re.compile(rf"^(?P<ref>{_REF})(?:\s*>\s*(?P<value>-?[0-9]+))?$")
_EACH_ARGS_RE = re.compile(rf'^(?P<source>{_REF})\s+as\s+(?P<var>{_IDENT})(?P<attrs>(?:\s+[a-z_]+=(?:"(?:[^"\\]|\\.)*"|[^\s"]+))*)$')
_ATTR_RE = re.compile(r'([a-z_]+)=("(?:[^"\\]|\\.)*"|[^\s"]+)')
_INT_RE = re.compile(r"^-?[0-9]+$")
_SCALARS = ("str", "int", "bool")
_EACH_ATTRS = ("index", "start", "max", "sep")
_INDEX_NAMES = ("i", "j", "k")


class DocError(ValueError):
    """A document violates the grammar or its parameter declarations; the message starts with the line."""


def _error(line: int, message: str) -> DocError:
    return DocError(f"line {line}: {message}" if line else message)


# ---------------------------------------------------------------------------------------
# Tokens


@dataclass
class _Text:
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class _Slot:
    ref: str
    conv: str | None
    raw: str
    line: int
    double: bool = False


@dataclass(frozen=True)
class _Tag:
    kind: str
    args: str
    raw: str
    line: int


def _tokenize(text: str) -> tuple[list[Any], list[tuple[int, int]]]:
    """Split a document into literal text, slot candidates and tags; apply the standalone and boundary rules.

    Returns:
        tuple[list[Any], list[tuple[int, int]]]: The tokens, and per ``{{message}}`` the source range its line
            takes, from the line break before it (when there is one) to the end of the line break after it.
    """
    tokens: list[Any] = []
    boundaries: list[tuple[int, int]] = []
    buffer: list[str] = []
    start = 0
    pos = 0
    line = 1
    size = len(text)

    def flush(end: int) -> None:
        if buffer:
            tokens.append(_Text("".join(buffer), start, end))
            buffer.clear()

    while pos < size:
        char = text[pos]
        if char == "{" and text.startswith("{{", pos):
            match = _TAG_RE.match(text, pos)
            if match is not None:
                flush(pos)
                end = match.end()
                standalone = (pos == 0 or text[pos - 1] == "\n") and (end == size or text[end] == "\n")
                if match.group("boundary"):
                    kind = "message"
                elif match.group("otherwise"):
                    kind = "else"
                elif match.group("close"):
                    kind = "/" + match.group("close")
                else:
                    kind = "#" + match.group("open")
                if kind == "message":
                    if not standalone:
                        raise _error(line, "{{message}} must stand alone on its line")
                    # the line break before the boundary line belongs to the boundary
                    previous = tokens[-1] if tokens else None
                    if isinstance(previous, _Text) and previous.end == pos and previous.value.endswith("\n"):
                        previous.value = previous.value[:-1]
                        if not previous.value:
                            tokens.pop()
                    boundaries.append((pos - 1 if pos else 0, end + 1 if end < size else end))
                tokens.append(_Tag(kind, match.group("args") or "", match.group(0), line))
                pos = end
                if standalone and end < size:
                    pos += 1
                    line += 1
                start = pos
                continue
            if TAG_START_RE.match(text, pos):
                raise _error(line, f"malformed tag {text[pos:pos + 40].splitlines()[0]!r}")
            double = _DOUBLE_SLOT_RE.match(text, pos)
            if double is not None:
                flush(pos)
                tokens.append(_Slot(double.group("ref"), None, double.group(0), line, True))
                pos = start = double.end()
                continue
        if char == "{":
            slot = _SLOT_RE.match(text, pos)
            if slot is not None:
                flush(pos)
                tokens.append(_Slot(slot.group("ref"), slot.group("conv"), slot.group(0), line))
                pos = start = slot.end()
                continue
        if char == MESSAGE_SEPARATOR:
            raise _error(line, "U+001E is what {{message}} renders as; a document may not contain it")
        if not buffer:
            start = pos
        buffer.append(char)
        if char == "\n":
            line += 1
        pos += 1
    flush(size)
    return tokens, boundaries


# ---------------------------------------------------------------------------------------
# Parser


def _seq(items: list[Any]) -> Any:
    if not items:
        return {"lit": ""}
    return items[0] if len(items) == 1 else {"seq": items}


class _Parser:
    def __init__(self, tokens: list[Any], params: Mapping[str, Type]) -> None:
        self.tokens = tokens
        self.params = params
        self.index = 0
        self.used: set[str] = set()
        #: Line of each block's derived node, by ``id``: a bound error on the derived spec is reported there.
        self.lines: dict[int, int] = {}

    def messages(self) -> tuple[list[list[Any]], list[set[str]]]:
        messages: list[list[Any]] = []
        uses: list[set[str]] = []
        while True:
            self.used = set()
            items, closer = self.items(dict(self.params), frozenset({"message"}))
            messages.append(items)
            uses.append(self.used)
            if closer is None:
                return messages, uses

    def items(self, scope: dict[str, Type], closers: frozenset[str]) -> tuple[list[Any], _Tag | None]:
        out: list[Any] = []
        while self.index < len(self.tokens):
            token = self.tokens[self.index]
            if isinstance(token, _Tag):
                if token.kind in closers:
                    self.index += 1
                    return out, token
                if token.kind == "message":
                    raise _error(token.line, "{{message}} separates top-level messages; it cannot sit inside a block")
                if token.kind == "else":
                    raise _error(token.line, "{{else}} belongs directly inside an {{#if}} block, once")
                if token.kind.startswith("/"):
                    raise _error(token.line, f"{token.raw} closes no open {{{{#{token.kind[1:]}}}}} block")
                self.index += 1
                out.append(self.block(token, scope))
                continue
            self.index += 1
            if isinstance(token, _Text):
                out.append({"lit": token.value})
                continue
            head = token.ref.split(".")[0]
            if head not in scope:
                out.append({"lit": token.raw})
                continue
            if token.double:
                raise _error(token.line, f"{token.raw}: a slot takes single braces, {{{token.ref}}}")
            if token.conv:
                raise _error(token.line, f"{token.raw}: conversions such as {token.conv} are not supported in a document")
            kind = self.resolve(token.ref, scope, token.line)
            if kind not in _SCALARS:
                raise _error(token.line, f"{token.raw} is a {type_name(kind)}; only str, int and bool values render as text")
            out.append({"param": token.ref})
        return out, None

    def resolve(self, ref: str, scope: Mapping[str, Type], line: int) -> Type:
        segments = ref.split(".")
        head = segments[0]
        if head not in scope:
            raise _error(line, f"{head!r} is not a declared parameter or a loop variable in scope; declared {sorted(self.params)}")
        if head in self.params:
            self.used.add(head)
        current = scope[head]
        if len(segments) == 2:
            if not isinstance(current, Mapping) or "record" not in current:
                raise _error(line, f"{ref!r}: {head!r} is a {type_name(current)}, which has no field {segments[1]!r}")
            fields = current["record"]
            if segments[1] not in fields:
                raise _error(line, f"{ref!r}: {head!r} has no field {segments[1]!r}; fields {sorted(fields)}")
            current = fields[segments[1]]
        return current

    def fresh(self, name: str, scope: Mapping[str, Type], line: int, label: str) -> str:
        try:
            check_identifier(name, label)
        except PolicyError as error:
            raise _error(line, str(error)) from error
        if name in scope:
            raise _error(line, f"{label} {name!r} must be a fresh name; it is already declared")
        return name

    def opened(self, tag: _Tag, node: dict[str, Any]) -> dict[str, Any]:
        self.lines[id(node)] = tag.line
        return node

    def block(self, tag: _Tag, scope: dict[str, Type]) -> Any:
        if tag.kind == "#if":
            match = _IF_ARGS_RE.match(tag.args.strip())
            if match is None:
                raise _error(tag.line, f"{tag.raw}: expected {{{{#if <ref>}}}} or {{{{#if <ref> > <int>}}}}")
            ref = match.group("ref")
            kind = self.resolve(ref, scope, tag.line)
            if match.group("value") is not None:
                if kind != "int":
                    raise _error(tag.line, f"{tag.raw}: '>' compares an int, {ref!r} is a {type_name(kind)}")
                test: Any = {"gt": [ref, int(match.group("value"))]}
            else:
                if kind not in _SCALARS and not (isinstance(kind, Mapping) and "list" in kind):
                    raise _error(tag.line, f"{tag.raw}: a truth test needs a str, int, bool or list, {ref!r} is a {type_name(kind)}")
                test = {"truthy": ref}
            then, closer = self.items(scope, frozenset({"else", "/if"}))
            if closer is None:
                raise _error(tag.line, f"{tag.raw} is never closed with {{{{/if}}}}")
            otherwise: list[Any] = []
            if closer.kind == "else":
                otherwise, closer = self.items(scope, frozenset({"/if"}))
                if closer is None:
                    raise _error(tag.line, f"{tag.raw} is never closed with {{{{/if}}}}")
            return self.opened(tag, {"if": {"test": test, "then": _seq(then), "else": _seq(otherwise)}})
        match = _EACH_ARGS_RE.match(tag.args.strip())
        if match is None:
            raise _error(tag.line, f"{tag.raw}: expected {{{{#each <list> as <var> max=<int> ...}}}}")
        source = match.group("source")
        kind = self.resolve(source, scope, tag.line)
        if not (isinstance(kind, Mapping) and "list" in kind):
            raise _error(tag.line, f"{tag.raw}: {source!r} is a {type_name(kind)}, {{{{#each}}}} needs a list")
        var = self.fresh(match.group("var"), scope, tag.line, "loop variable")
        attrs: dict[str, str] = {}
        for key, value in _ATTR_RE.findall(match.group("attrs")):
            if key not in _EACH_ATTRS:
                raise _error(tag.line, f"{tag.raw}: unknown attribute {key!r}; allowed {list(_EACH_ATTRS)}")
            if key in attrs:
                raise _error(tag.line, f"{tag.raw}: attribute {key!r} is given twice")
            attrs[key] = value
        if "max" not in attrs:
            raise _error(tag.line, f"{tag.raw}: max=<int> is required (the static iteration cap)")
        if not _INT_RE.fullmatch(attrs["max"]) or not 1 <= int(attrs["max"]) <= EACH_MAX:
            raise _error(tag.line, f"{tag.raw}: max must be an int in 1..{EACH_MAX}")
        each: dict[str, Any] = {"in": source, "as": var, "max": int(attrs["max"])}
        inner = dict(scope)
        inner[var] = kind["list"]
        if "index" in attrs:
            index = self.fresh(attrs["index"], inner, tag.line, "index variable")
            inner[index] = "int"
            each["index"] = index
            start = attrs.get("start", "0")
            if start not in ("0", "1"):
                raise _error(tag.line, f"{tag.raw}: start must be 0 or 1")
            each["start"] = int(start)
        elif "start" in attrs:
            raise _error(tag.line, f"{tag.raw}: start is meaningless without index=<name>")
        sep = ""
        if "sep" in attrs:
            raw = attrs["sep"]
            try:
                sep = json.loads(raw) if raw.startswith('"') else None
            except ValueError:
                sep = None
            if not isinstance(sep, str):
                raise _error(tag.line, f"{tag.raw}: sep must be a double-quoted JSON string, got {raw}")
            if len(sep) > SEP_MAX_CHARS:
                raise _error(tag.line, f"{tag.raw}: sep has {len(sep)} characters; the limit is {SEP_MAX_CHARS}")
            try:
                check_document(sep, what="sep")   # a JSON escape must not carry what the document may not
            except PolicyError as error:
                raise _error(tag.line, f"{tag.raw}: {error}") from error
        body, closer = self.items(inner, frozenset({"/each"}))
        if closer is None:
            raise _error(tag.line, f"{tag.raw} is never closed with {{{{/each}}}}")
        each["body"] = _seq(body)
        return self.opened(tag, {"join": {"sep": sep, "items": {"each": each}}})


def _typed(params: Mapping[str, Any]) -> dict[str, Type]:
    if not isinstance(params, Mapping):
        raise DocError("params must be a mapping of parameter name to type")
    typed: dict[str, Type] = {}
    for name, value in params.items():
        try:
            check_identifier(name, "param name")
            typed[name] = check_type(value, f"params.{name}")
        except (PolicyError, SpecError) as error:
            raise DocError(str(error)) from error
    return typed


@dataclass(frozen=True)
class ParsedDocument:
    """A document parsed against its parameter declarations.

    Attributes:
        params (Mapping[str, Type]): The declared parameter types, normalized.
        spec (Mapping[str, Any]): The derived spec of the whole document: the messages' items in order, each boundary
            the literal U+001E. This is what the capsule source is compiled from.
        messages (tuple[Mapping[str, Any], ...]): The derived spec of each message, rendered one at a time.
        segments (tuple[str, ...]): The source text of each message, markers as written: the identity text.
        uses (tuple[tuple[str, ...], ...]): Per message, the declared parameters it references, in declaration order.
        tagged (bool): Whether the document has any tag; without one it has a text row's shape.
    """

    params: Mapping[str, Type]
    spec: Mapping[str, Any]
    messages: tuple[Mapping[str, Any], ...]
    segments: tuple[str, ...]
    uses: tuple[tuple[str, ...], ...]
    tagged: bool


def parse_document(text: str, params: Mapping[str, Any], *, bounded: bool = True) -> ParsedDocument:
    """Parse a document against its declared parameters.

    Args:
        text (str): The document (``meta.text``), already through :func:`policy.check_document`.
        params (Mapping[str, Any]): Declared parameter types (``meta.params``) in the spec type language.
        bounded (bool): Hold the derived spec to the node-count and nesting-depth limits (every stored document
            is); :func:`document_of` lifts them to verify the document of an existing composition.

    Returns:
        ParsedDocument: The derived specs, the per-message source segments and what each message uses.

    Raises:
        DocError: On any grammar or declaration violation, naming the line; also when a declared parameter is used by
            no message (no line to name), or the derived spec breaks a spec bound — a nesting error names the line of
            the innermost block on the offending path, a node-count error no line.
    """
    typed = _typed(params)
    tokens, boundaries = _tokenize(text)
    parser = _Parser(tokens, typed)
    items, uses = parser.messages()
    unused = [name for name in typed if not any(name in used for used in uses)]
    if unused:
        raise DocError(f"declared parameters no message uses: {unused}")
    segments: list[str] = []
    cursor = 0
    for before, after in boundaries:
        segments.append(text[cursor:before])
        cursor = after
    segments.append(text[cursor:])
    whole: list[Any] = []
    for index, message in enumerate(items):
        if index:
            whole.append({"lit": MESSAGE_SEPARATOR})
        whole.extend(message)
    spec = {"seq": whole}
    messages = tuple({"seq": message} for message in items)
    try:
        parse_spec(spec, params=typed, derived=True, bounded=bounded)
    except SpecError as error:
        line = _block_line(spec, str(error), parser.lines)
        raise _error(line, f"the document's derived spec breaks a bound{' inside the block opened on this line' if line else ''}: {error}") from error
    return ParsedDocument(
        typed,
        spec,
        messages,
        tuple(segments),
        tuple(tuple(name for name in typed if name in used) for used in uses),
        any(isinstance(token, _Tag) for token in tokens),
    )


def _block_line(spec: Mapping[str, Any], message: str, lines: Mapping[int, int]) -> int:
    """The line of the innermost block on the derived-spec path a bound error names; 0 when it names no block."""
    path = message.split(":", 1)[0]
    if not path.startswith("$"):
        return 0   # a node-count error names no path
    node: Any = spec
    line = 0
    for key, index in re.findall(r"\.([a-z]+)(?:\[(\d+)\])?", path):
        if not isinstance(node, Mapping) or key not in node:
            break
        node = node[key]
        if index:
            if not isinstance(node, list) or int(index) >= len(node):
                break
            node = node[int(index)]
        line = lines.get(id(node), line)
    return line


def compile_document(name: str, text: str, params: Mapping[str, Any]) -> str:
    """Generate the ``compose(parts, params)`` source HeavenBase stores for a doc row.

    A doc row is the function prompt of a document with tags. A document without any tag — no message
    boundary, no block — has a text row's shape and is published as one (a template prompt), so it is refused
    here, where every path that creates or verifies a doc row derives the row's source: one text is one kind
    of row and one content key.

    Args:
        name (str): Row name, quoted into the docstring.
        text (str): The document.
        params (Mapping[str, Any]): Declared parameter types.

    Returns:
        str: The source :func:`spec.compile_spec` emits for the document's derived spec; ``parts`` is unused.

    Raises:
        DocError: If the document does not parse or has no tag.
    """
    parsed = parse_document(text, params)
    if not parsed.tagged:
        raise DocError("a doc row needs a tag, a {{message}} line or a block; a text without tags is a text row (op: text)")
    return compile_spec(name, parsed.spec, {}, dict(params), derived=True, origin=f"meta.text ({SCHEMA_DOC})")


# ---------------------------------------------------------------------------------------
# Document rows


class Document(Composer):
    """A doc row: one document, rendered whole (messages joined by U+001E) or one message at a time.

    Attributes:
        text (str): The document.
        segments (tuple[str, ...]): The source text of each message.
        messages (tuple[Composer, ...]): One executor per message; each declares the parameters of its surface.
    """

    kind = "doc"

    def __init__(self, name: str, version: str, text: str, parsed: ParsedDocument, message_params: Sequence[Mapping[str, Any]] | None = None) -> None:
        """Build the executors of a parsed document.

        Args:
            name (str): Row name.
            version (str): Row version.
            text (str): The document.
            parsed (ParsedDocument): Output of :func:`parse_document` for ``text``.
            message_params (Sequence[Mapping[str, Any]] | None): Declared parameters of each message — the parameters
                of the surface it provides, which a message may use a subset of; the parameters each message uses
                when ``None``.

        Raises:
            SpecError: If the count of ``message_params`` differs from the document's messages, or a message uses a
                parameter its declaration lacks.
        """
        super().__init__(name, version, parse_spec(parsed.spec, params=parsed.params, derived=True), {})
        self.text = text
        self.segments = parsed.segments
        declared = list(message_params) if message_params is not None else [{param: parsed.params[param] for param in uses} for uses in parsed.uses]
        if len(declared) != len(parsed.messages):
            raise SpecError(f"{name}: the document has {len(parsed.messages)} message(s), {len(declared)} expected")
        executors = []
        for index, (spec, params) in enumerate(zip(parsed.messages, declared)):
            try:
                executors.append(Composer(name, version, parse_spec(spec, params=params, derived=True), {}))
            except SpecError as error:
                raise SpecError(f"{name}: message {index + 1} uses a parameter outside {sorted(params)}: {error}") from error
        self.messages = tuple(executors)


# ---------------------------------------------------------------------------------------
# Pretty-printer


def _merge(items: list[Any]) -> Any:
    """Flatten sequences and merge adjacent literals, exactly as the normal form does."""
    flat: list[Any] = []
    fragments: list[str] = []

    def flush() -> None:
        if fragments:
            flat.append(["lit", "".join(fragments)])
            fragments.clear()

    for item in items:
        for piece in item[1] if item[0] == "seq" else [item]:
            if piece[0] == "lit":
                if piece[1]:
                    fragments.append(piece[1])
                continue
            flush()
            flat.append(piece)
    flush()
    if not flat:
        return ["lit", ""]
    return flat[0] if len(flat) == 1 else ["seq", flat]


def _fold(form: Any) -> Any:
    """The normal form with every join over a static list folded into text, as a document writes it."""
    head = form[0]
    if head in ("lit", "param", "field", "var", "idx"):
        return form
    if head == "seq":
        return _merge([_fold(item) for item in form[1]])
    if head == "if":
        then, otherwise = _fold(form[2]), _fold(form[3])
        return then if then == otherwise else ["if", form[1], then, otherwise]
    if head == "join":
        sep, items = form[1], form[2]
        if items[0] == "list":
            pieces: list[Any] = []
            for index, item in enumerate(items[1]):
                if index and sep:
                    pieces.append(["lit", sep])
                pieces.append(_fold(item))
            return _merge(pieces)
        if items[0] == "each":
            return ["join", sep, ["each", *items[1:5], _fold(items[5])]]
        raise DocError("a join over a concatenation of lists has no document form")
    if head == "repr":
        raise DocError("a !r conversion has no document form")
    raise DocError(f"normal form node {head!r} has no document form")


def _singular(name: str) -> str:
    if name.endswith("ies") and len(name) > 3:
        return name[:-3] + "y"
    if name.endswith("s") and not name.endswith("ss") and len(name) > 1:
        return name[:-1]
    return name + "_item"


class _Printer:
    def __init__(self, params: Mapping[str, Type]) -> None:
        self.params = params
        self.out: list[str] = []
        self.loops: list[tuple[str, str | None]] = []

    def taken(self) -> set[str]:
        names = set(self.params)
        for var, index in self.loops:
            names.add(var)
            if index is not None:
                names.add(index)
        return names

    def fresh(self, candidates: Iterable[str]) -> str:
        taken = self.taken()
        for name in candidates:
            try:
                check_identifier(name)
            except PolicyError:
                continue
            if name not in taken:
                return name
        raise DocError("no fresh loop variable name is available")

    def emit(self, text: str) -> None:
        if text:
            self.out.append(text)

    def tag(self, text: str) -> None:
        # standalone when the source so far ends a line; its own line break is then removed by the parser
        self.emit(text + "\n" if not self.out or self.out[-1].endswith("\n") else text)

    def ref(self, form: Any) -> str:
        head = form[0]
        if head == "param":
            return str(form[1])
        if head == "var":
            return self.loops[form[1]][0]
        if head == "idx":
            index = self.loops[form[1]][1]
            if index is None:
                raise DocError("an index reference outside an indexed loop")
            return index
        if head == "field":
            return f"{self.ref(form[1])}.{form[2]}"
        raise DocError(f"normal form node {head!r} is not a reference")

    def test(self, form: Any) -> str:
        if form[0] == "truthy":
            return self.ref(form[1])
        if form[0] == "gt" and form[1][0] != "len":
            return f"{self.ref(form[1])} > {form[2]}"
        raise DocError(f"the test {form[0]!r} has no document form")

    def node(self, form: Any) -> None:
        head = form[0]
        if head == "lit":
            self.emit(form[1])
        elif head == "seq":
            for item in form[1]:
                self.node(item)
        elif head in ("param", "field", "var", "idx"):
            self.emit("{" + self.ref(form) + "}")
        elif head == "if":
            self.tag("{{#if " + self.test(form[1]) + "}}")
            self.node(form[2])
            if form[3] != ["lit", ""]:
                self.tag("{{else}}")
                self.node(form[3])
            self.tag("{{/if}}")
        elif head == "join":
            _each, source, start, where, maximum, body = form[2]
            if where is not None:
                raise DocError("a loop with a where filter has no document form")
            name = str(source[1]) if source[0] == "param" else str(source[2]) if source[0] == "field" else "item"
            var = self.fresh([_singular(name), name + "_item"] + [f"{_singular(name)}{number}" for number in range(2, 10)])
            index = None
            if start is not None:
                depth = len(self.loops)
                index = self.fresh(([_INDEX_NAMES[depth]] if depth < len(_INDEX_NAMES) else []) + ["idx"] + [f"idx{number}" for number in range(2, 10)])
            attrs = (f" index={index} start={start}" if index is not None else "") + f" max={maximum}"
            if form[1]:
                attrs += " sep=" + json.dumps(form[1], ensure_ascii=False)
            self.tag("{{#each " + self.ref(source) + " as " + var + attrs + "}}")
            self.loops.append((var, index))
            self.node(body)
            self.loops.pop()
            self.tag("{{/each}}")
        else:
            raise DocError(f"normal form node {head!r} has no document form")


def document_of(form: Any, params: Mapping[str, Any]) -> str:
    """Print a normal form as a document, verified to parse back to the same form.

    Literals print verbatim, parameter references as slots, a join over a static list as its text, a
    loop as ``{{#each}}`` with a variable named after the list (``candidates`` → ``candidate``) and an
    index named ``i``, ``j``, ``k`` by depth, a branch as ``{{#if}}``; a tag stands alone on its line
    whenever the text before it ends a line. The result is parsed under every declared parameter — the
    declarations a doc row carrying it would store — and its normal form compared with the input's, so a
    form with no exact document is refused, never misprinted: a literal that reads as a slot or a tag (a
    literal ``{p}`` of a declared ``p`` included, used by the form or not), a declared parameter the form
    never uses, a ``!r`` conversion, a ``where`` filter, a ``len`` test. Two forms that print the same
    document under the same declarations therefore fold to the same form and render the same bytes.

    Args:
        form (Any): A normal form (:meth:`spec.Composer.normal_form`, :meth:`spec.TextLeaf.normal_form`).
        params (Mapping[str, Any]): Every parameter type the form was declared with.

    Returns:
        str: The document of one message.

    Raises:
        DocError: If the form has no exact document form.
    """
    typed = _typed(params)
    folded = _fold(form)
    printer = _Printer(typed)
    printer.node(folded)
    text = "".join(printer.out)
    try:
        parsed = parse_document(text, typed, bounded=False)
        check = Composer("document_of", "0", parse_spec(parsed.messages[0], params=parsed.params, derived=True, bounded=False), {}).normal_form() if len(parsed.messages) == 1 else None
    except (DocError, SpecError) as error:
        raise DocError(f"the normal form has no exact document form: {error}") from error
    if check != folded:
        raise DocError("the normal form has no exact document form: the printed document parses back to a different form")
    return text
