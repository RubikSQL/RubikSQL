"""Pure verification of native prompt rows before any HB execution or bootstrap."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any

from .canon import content_key, sha256_hex, template_code
from .doc import DocError, check_document, compile_document
from .policy import PolicyError, check_meta_keys, check_name, check_text, check_version
from .spec import compile_spec


class PromptIntegrityError(ValueError):
    """A native prompt's source, mirrors or original fingerprints disagree."""


def _native_json(value: Any) -> Any:
    """Match the pinned HB JSON adapter, without loading HB or restoring code."""
    if isinstance(value, tuple):
        return {"__obj_type__": "tuple", "__obj_data__": [_native_json(item) for item in value]}
    if isinstance(value, set):
        return {"__obj_type__": "set", "__obj_data__": [_native_json(item) for item in sorted(value, key=repr)]}
    if isinstance(value, datetime):
        return {"__obj_type__": "datetime", "__obj_data__": value.timestamp()}
    if isinstance(value, date):
        return {"__obj_type__": "date", "__obj_data__": datetime(value.year, value.month, value.day).timestamp()}
    if isinstance(value, Decimal):
        return {"__obj_type__": "decimal", "__obj_data__": str(value)}
    if isinstance(value, int) and (value > 1 << 53 or value < -(1 << 53)):
        return {"__obj_type__": "bigint", "__obj_data__": str(value)}
    if value is Ellipsis:
        return {"__obj_type__": "ellipsis"}
    if isinstance(value, Mapping):
        return {str(key): _native_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_native_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if callable(value) and hasattr(value, "__module__") and hasattr(value, "__qualname__"):
        return {"__obj_type__": "callable-ref", "__obj_data__": f"{value.__module__}.{value.__qualname__}"}
    return value


def native_hash(value: Any, *, capsule: bool = False) -> str:
    """Hash exact native data, retaining timestamps and all semantic metadata.

    ``capsule=True`` reconstructs only the pinned template renderer's three
    ``tr=str`` defaults. HB's JSON decoder represents callable references as
    strings, while its original native fingerprint includes the reference tag.
    Ordinary JSON string defaults remain verifiable through the direct hash.
    """
    adapted = _native_json(value)
    if capsule and isinstance(adapted, dict):
        adapted = deepcopy(adapted)
        defaults = []
        properties = adapted.get("input_schema", {}).get("properties", {})
        if isinstance(properties.get("tr"), dict):
            defaults.append(properties["tr"])
        for parameter in adapted.get("signature", {}).get("parameters", []):
            if isinstance(parameter, dict) and parameter.get("name") == "tr":
                defaults.append(parameter)
                if isinstance(parameter.get("schema"), dict):
                    defaults.append(parameter["schema"])
        for item in defaults:
            if item.get("default") == "builtins.str":
                item["default"] = {"__obj_type__": "callable-ref", "__obj_data__": "builtins.str"}
    if value is None or isinstance(value, (str, int, float, bool)):
        encoded = str(value)
    else:
        # HB compact=True removes indentation, but retains Python's separators.
        encoded = json.dumps(adapted, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def check_prompt_source(row: Mapping[str, Any]) -> str:
    """Regenerate the only permitted source and verify content identity and trust."""
    try:
        name, version = check_name(row.get("name")), check_version(row.get("version"))
        meta = dict((row.get("manifest") or {}).get("meta") or {})
        check_meta_keys(meta)
        kind = meta.get("kind")
        if kind == "text":
            expected = template_code(name, check_text(meta.get("text"), meta.get("slots"), name=name), [])
            qualname = "render"
        elif kind == "spec":
            expected = compile_spec(name, meta["spec"], meta["parts"], meta["params"])
            qualname = "compose"
        elif kind == "doc":
            if any(key in meta for key in ("spec", "parts", "slots")):
                raise PromptIntegrityError(f"{name}: a doc row stores no spec, parts or slots")
            check_document(meta.get("text"))
            expected = compile_document(name, meta["text"], meta["params"])
            qualname = "compose"
        else:
            raise PromptIntegrityError(f"{name}: unknown row kind {kind!r}")
        if content_key(name, meta) != meta.get("content_key") or version != str(meta.get("content_key"))[:16]:
            raise PromptIntegrityError(f"{name}: content key or version does not match the row's meta")
        capsule = row.get("capsule") or {}
        layers = capsule.get("restore_layers") or []
        if len(layers) != 1:
            raise PromptIntegrityError(f"{name}: expected exactly one restore layer, found {len(layers)}")
        layer = layers[0]
        shape = {"type": "source", "priority": 10, "filename": "<capsule-code>", "global_refs": {}, "qualname": qualname}
        for key, value in shape.items():
            if layer.get(key) != value:
                raise PromptIntegrityError(f"{name}: restore layer {key} differs from {value!r}")
        if layer.get("source") != expected:
            raise PromptIntegrityError(f"{name}: stored capsule source differs from the source regenerated from meta")
        digest = sha256_hex(expected)
        if any(item.get("source_hash") != digest for item in (layer, capsule.get("source") or {}, capsule.get("fingerprints") or {})):
            raise PromptIntegrityError(f"{name}: the three source hashes must equal sha256(source)")
        if capsule.get("source", {}).get("source") != expected:
            raise PromptIntegrityError(f"{name}: capsule source differs from its restore layer")
        if (capsule.get("trust") or {}).get("verify_source_hash") is not True:
            raise PromptIntegrityError(f"{name}: capsule.trust.verify_source_hash must be true")
        return str(kind)
    except (PolicyError, DocError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, PromptIntegrityError):
            raise
        raise PromptIntegrityError(f"{row.get('name')}: {error}") from error


def check_prompt_row(row: Mapping[str, Any]) -> str:
    """Verify source, full native mirrors and original manifest/capsule hashes."""
    kind = check_prompt_source(row)
    manifest = row["manifest"]
    capsule = row["capsule"]
    for key, expected in (("namespace", "prompt"), ("name", row["name"]), ("version", row["version"])):
        if capsule.get(key) != expected:
            raise PromptIntegrityError(f"{row['name']}: native capsule {key} differs")
    if capsule.get("identity", {}).get("name") != row["name"]:
        raise PromptIntegrityError(f"{row['name']}: native capsule identity name differs")
    def native_id(*parts: str) -> str:
        encoded = json.dumps(_native_json(tuple(parts)), sort_keys=True, ensure_ascii=False)
        return str(int(hashlib.md5(encoded.encode("utf-8")).hexdigest(), 16)).zfill(42)

    if row.get("object_id") != native_id("prompt", row["name"], row["version"]) or capsule.get("id") != native_id("capsule", "prompt", row["name"], row["version"]):
        raise PromptIntegrityError(f"{row['name']}: native prompt or capsule identity differs")
    for key in ("object_id", "name", "version", "description", "tags", "dicts", "main_lang"):
        if row.get(key) != manifest.get(key):
            raise PromptIntegrityError(f"{row['name']}: native row and manifest {key} differ")
    fingerprints = row.get("fingerprints")
    if not isinstance(fingerprints, Mapping) or set(fingerprints) != {"manifest", "capsule"}:
        raise PromptIntegrityError(f"{row['name']}: native fingerprints are incomplete")
    capsule_hashes = {native_hash(row["capsule"])}
    if kind == "text":
        capsule_hashes.add(native_hash(row["capsule"], capsule=True))
    if fingerprints["manifest"] != native_hash(manifest) or fingerprints["capsule"] not in capsule_hashes:
        raise PromptIntegrityError(f"{row['name']}: native manifest or capsule fingerprint differs")
    return kind
