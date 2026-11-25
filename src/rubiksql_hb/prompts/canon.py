"""Canonical JSON, content keys and the template escape shared by every prompt identity.

Every identity in the prompt system — a row's ``content_key``, a release id, a keyspace,
a render-argument digest — is the sha256 of the canonical JSON produced by :func:`canon`.
``escape`` turns the human-readable text of a text row into the executable ``str.format``
template HeavenBase stores, and :func:`template_code` is a byte-for-byte copy of HeavenBase's
private ``_template_code`` so the runtime can regenerate a text row's capsule source without
importing heavenbase; ``tests/test_prompt_policy.py`` pins the copy to the installed wheel.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, List, Mapping

__all__ = [
    "SCHEMA_PROMPT",
    "canon",
    "content_key",
    "escape",
    "sha256_hex",
    "template_code",
    "unescape",
]

#: Schema tag hashed into every row content key.
SCHEMA_PROMPT = "rubiksql.prompt/1"

#: One ``{slot}``, ``{slot!r}`` or ``{slot!s}`` field candidate in human-readable text.
_FIELD_RE = re.compile(r"\{([a-z][a-z0-9_]*)(![rs])?\}")
#: Tokens of the human-readable form: a field candidate, or one lone brace.
_TEXT_TOKEN_RE = re.compile(r"\{[a-z][a-z0-9_]*(?:![rs])?\}|\{|\}")
#: Tokens of the executable form: a doubled brace, a field candidate, or one lone brace.
_TEMPLATE_TOKEN_RE = re.compile(r"\{\{|\}\}|\{[a-z][a-z0-9_]*(?:![rs])?\}|\{|\}")


def canon(obj: Any) -> str:
    """Serialize a JSON value in the one canonical form every prompt identity hashes.

    Args:
        obj (Any): JSON-compatible value (str, int, bool, None, list, dict).

    Returns:
        str: ``json.dumps`` output with sorted keys, no whitespace and non-ASCII kept verbatim.
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_hex(data: str | bytes) -> str:
    """Return the sha256 hex digest of a string (UTF-8) or of raw bytes.

    Args:
        data (str | bytes): Text, encoded as UTF-8 before hashing, or bytes hashed as given.

    Returns:
        str: 64 lower-case hex characters.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def content_key(name: str, meta: Mapping[str, Any]) -> str:
    """Compute the 64-hex content key of a prompt row from its name and ``manifest.meta``.

    The key covers only what changes the rendered bytes: the kind, the name, and the text
    plus slots of a text row, the spec plus parts plus params of a spec row, or the text plus
    params of a doc row. Description, reason, lineage and timestamps stay outside, so
    republishing identical content is idempotent.

    Args:
        name (str): Dotted row name, already validated by the caller.
        meta (Mapping[str, Any]): The row's ``manifest.meta``; ``kind`` selects which keys are hashed.

    Returns:
        str: sha256 hex of ``canon({"schema": "rubiksql.prompt/1", "kind", "name", ...})``.

    Raises:
        ValueError: If ``meta["kind"]`` is not ``"text"``, ``"spec"`` or ``"doc"``.
    """
    kind = meta.get("kind")
    body: dict[str, Any] = {"schema": SCHEMA_PROMPT, "kind": kind, "name": name}
    if kind == "text":
        body["text"] = meta["text"]
        body["slots"] = list(meta["slots"])
    elif kind == "spec":
        body["spec"] = meta["spec"]
        body["parts"] = dict(meta["parts"])
        body["params"] = dict(meta["params"])
    elif kind == "doc":
        body["text"] = meta["text"]
        body["params"] = dict(meta["params"])
    else:
        raise ValueError(f"prompt row {name!r} has unknown kind {kind!r}; expected 'text', 'spec' or 'doc'")
    return sha256_hex(canon(body))


def escape(text: str, slots: Iterable[str]) -> str:
    """Turn human-readable prompt text into the ``str.format`` template HeavenBase executes.

    Every ``{slot}``, ``{slot!r}`` or ``{slot!s}`` whose name is a declared slot stays a field;
    every other brace is doubled so it renders literally. The result is what
    ``Prompt(template, ...)`` receives and what :func:`template_code` embeds.

    Args:
        text (str): Human-readable text with single-brace slot references.
        slots (Iterable[str]): Declared slot names; only these become format fields.

    Returns:
        str: Executable template; ``unescape(escape(text, slots), slots) == text`` always holds.
    """
    declared = set(slots)
    out: list[str] = []
    pos = 0
    for match in _TEXT_TOKEN_RE.finditer(text):
        out.append(text[pos:match.start()])
        token = match.group(0)
        field = _FIELD_RE.fullmatch(token)
        if field is not None and field.group(1) in declared:
            out.append(token)
        elif field is not None:
            out.append("{{" + token[1:-1] + "}}")
        else:
            out.append(token * 2)
        pos = match.end()
    out.append(text[pos:])
    return "".join(out)


def unescape(template: str, slots: Iterable[str]) -> str:
    """Recover the human-readable text from an executable template.

    Args:
        template (str): A template produced by :func:`escape` or written by HeavenBase.
        slots (Iterable[str]): Declared slot names; fields naming anything else are rejected.

    Returns:
        str: Human-readable text with doubled braces collapsed and declared fields kept verbatim.

    Raises:
        ValueError: If the template has a lone brace or a field that is not a declared slot.
    """
    declared = set(slots)
    out: list[str] = []
    pos = 0
    for match in _TEMPLATE_TOKEN_RE.finditer(template):
        out.append(template[pos:match.start()])
        token = match.group(0)
        if token == "{{" or token == "}}":
            out.append(token[0])
        elif token in ("{", "}"):
            raise ValueError(f"template has a lone {token!r} at offset {match.start()}")
        else:
            field = _FIELD_RE.fullmatch(token)
            if field is None or field.group(1) not in declared:
                raise ValueError(f"template field {token!r} is not a declared slot")
            out.append(token)
        pos = match.end()
    out.append(template[pos:])
    return "".join(out)


def template_code(name: str, template: str, tr_keys: List[str]) -> str:
    """Generate the ``render`` source HeavenBase stores for a template prompt.

    This is a byte-for-byte copy of ``heavenbase.extensions.prompt.prompt._template_code`` from
    the pinned wheel; the runtime regenerates every text row's capsule source with it and
    compares byte-for-byte, so the copy must track the installed function exactly.

    Args:
        name (str): Prompt name quoted into the error messages of the generated function.
        template (str): Executable ``str.format`` template, as produced by :func:`escape`.
        tr_keys (List[str]): Argument names the generated function translates; always ``[]`` here.

    Returns:
        str: Python source of a ``render(*args, tr=str, **kwargs)`` function.
    """
    return f'''def render(*args, tr=str, **kwargs):
    """Render a string prompt template."""
    values = dict(kwargs)
    if args:
        if len(args) == 1 and hasattr(args[0], "items"):
            values = dict(args[0]) | values
        else:
            raise TypeError("Template prompts accept keyword args or one mapping positional arg")
    template = tr({template!r})
    for key in {list(tr_keys)!r}:
        if key in values:
            values[key] = tr(str(values[key]))
    try:
        return str(template).format(**values)
    except KeyError as error:
        raise KeyError(f"Prompt {name!r} missing template placeholder {{error.args[0]!r}}") from error
    except Exception as error:
        raise ValueError(f"Prompt {name!r} template render failed: {{error}}") from error
'''
