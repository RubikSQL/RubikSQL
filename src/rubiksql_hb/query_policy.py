"""HeavenBase's ``code_reduce`` never runs for a query spec a model wrote (user decision 2026-10-01).

``code_reduce`` is a HeavenBase aggregate that evaluates a caller-written Python expression, with full builtins, in
the process that executes the query (``heavenbase.query.aggregate._compiled_reduction``), and
``heavenbase.query.allow_code_reduce`` is ``true`` by default. Every spec this package hands to HeavenBase comes from a
model, so two layers keep it out:

- :func:`refuse_code_reduce` refuses a spec that names it, in every spelling HeavenBase resolves, before HeavenBase
  sees the spec. The knowledge readers ``tools.knowledge.query_records`` and ``query_frame`` call it, so
  ``query_knowledge``, ``query_spec`` and ``query_lab`` answer with their usual ``error`` result.
- :func:`code_reduce_disabled` turns the switch off for this process while it serves. ``server.main`` and
  ``lab_server.main`` serve inside it, so HeavenBase itself refuses ``code_reduce``, with its own "disabled" error, on
  every path those processes run, its native ``query`` tool included. Nothing is written to any HeavenBase home.
"""
from __future__ import annotations

import json
import threading
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from heavenbase.utils.config import ConfigManager, ConfigSnapshot

__all__ = ["ALLOW_CODE_REDUCE", "code_reduce_disabled", "code_reduce_key", "refuse_code_reduce"]

#: HeavenBase's switch for ``code_reduce``, read when one of its folds runs (``heavenbase.query.aggregate``).
ALLOW_CODE_REDUCE = "heavenbase.query.allow_code_reduce"

#: The final configuration layer :func:`code_reduce_disabled` binds. HeavenBase merges it without mutating it.
_OFF = {"heavenbase": {"query": {"allow_code_reduce": False}}}

_LOCK = threading.Lock()
_OPEN = 0  # code_reduce_disabled blocks open in this process; the outermost one installs and removes the layer


def code_reduce_key(spec: Any) -> str | None:
    """Return the first key in ``spec`` that HeavenBase resolves to its ``code_reduce`` aggregate, or ``None``.

    HeavenBase matches an operation name case-insensitively, around surrounding space and through its aliases
    (``$code_reduce``), so every key is resolved with HeavenBase's own ``normalize_op`` instead of compared by spelling;
    the two registered spellings also match directly, so the check holds when the operation registry cannot be read.
    Mappings and sequences are walked to any depth, and a string holding JSON is decoded and walked as well, however
    many times it was encoded: HeavenBase's native ``query`` decodes a spec given as JSON text.

    Args:
        spec: A query spec or filter as the caller wrote it: a mapping, a sequence, JSON text or ``None``.

    Returns:
        str | None: The key as written, or ``None`` when no key names ``code_reduce``.

    Examples:
        >>> code_reduce_key({"aggregate": {"n": {"$CODE_REDUCE": {"field": "x", "code": "len"}}}})
        '$CODE_REDUCE'
        >>> code_reduce_key({"aggregate": {"n": {"count": {"field": "x"}}}}) is None
        True
    """
    pending = deque([spec])
    while pending:
        node = pending.popleft()
        if isinstance(node, str):
            pending.append(_json_value(node))
        elif isinstance(node, Mapping):
            for key, value in node.items():
                if _names_code_reduce(key):
                    return str(key)
                pending.append(value)
        elif isinstance(node, (list, tuple, set, frozenset)):
            pending.extend(node)
    return None


def refuse_code_reduce(spec: Any) -> None:
    """Refuse a query spec or filter that names HeavenBase's ``code_reduce`` anywhere (:func:`code_reduce_key`).

    Args:
        spec: A query spec or filter as the caller wrote it: a mapping, a sequence, JSON text or ``None``.

    Raises:
        ValueError: ``spec`` names ``code_reduce``; the message names the key and the reducers to use instead.
    """
    key = code_reduce_key(spec)
    if key is not None:
        raise ValueError(f"{key!r} is refused: code_reduce runs a caller-written expression in the server process; "
                         "use a registered reducer such as count, min, max, sum or avg")


@contextmanager
def code_reduce_disabled() -> Iterator[None]:
    """Refuse HeavenBase's ``code_reduce`` everywhere in this process while the block runs, persisting nothing.

    ``ConfigManager.set`` would write the off value into the HeavenBase home, machine-wide or private, for every
    process to read. HeavenBase's one non-persisting layer is the final ``overrides`` layer of
    ``ConfigManager.snapshot``, through which ``get``, ``load`` and ``to_dict`` all read. The block binds :data:`_OFF`
    as that layer on ``CM_HVNB``, the ConfigManager of the default Context that every workspace here executes under,
    where HeavenBase reads :data:`ALLOW_CODE_REDUCE` when a ``code_reduce`` fold runs. Blocks nest; leaving the
    outermost one restores the stored configuration.

    Yields:
        None: Inside the block, every ``code_reduce`` this process runs fails with HeavenBase's own
        ``code_reduce ... is disabled`` ``ValueError``.

    Raises:
        RuntimeError: HeavenBase still reads the switch as on, so it no longer reads configuration through
            ``ConfigManager.snapshot``; the block refuses to start rather than serve with ``code_reduce`` enabled.
    """
    from heavenbase.utils.config import CM_HVNB

    global _OPEN
    with _LOCK:
        if _OPEN == 0:
            CM_HVNB.snapshot = _snapshot_with_off(CM_HVNB)
        _OPEN += 1
    try:
        if CM_HVNB.get(ALLOW_CODE_REDUCE, default=False) is not False:
            raise RuntimeError(f"{ALLOW_CODE_REDUCE} still reads as enabled with the off layer bound; "
                               "HeavenBase no longer reads its configuration through ConfigManager.snapshot")
        yield
    finally:
        with _LOCK:
            _OPEN -= 1
            if _OPEN == 0:
                vars(CM_HVNB).pop("snapshot", None)


def _snapshot_with_off(manager: ConfigManager) -> Callable[..., ConfigSnapshot]:
    """Return ``manager``'s own ``snapshot`` with :data:`_OFF` merged last into whatever ``overrides`` it is given."""
    from heavenbase.utils.config import dmerge

    snapshot = type(manager).snapshot

    def snapshot_with_off(overrides: Mapping[str, Any] | None = None, policy: Any = None,
                          refresh: bool = False) -> ConfigSnapshot:
        return snapshot(manager, dmerge([overrides, _OFF]) if overrides else _OFF, policy, refresh)

    return snapshot_with_off


def _names_code_reduce(key: Any) -> bool:
    """Whether HeavenBase resolves ``key`` to the ``code_reduce`` operation."""
    from heavenbase.filters.ops import normalize_op

    token = str(key).strip().lower()
    if token in ("code_reduce", "$code_reduce"):
        return True
    try:
        return normalize_op(token) == "code_reduce"
    except (LookupError, ValueError):  # not an operation name
        return False


def _json_value(text: str) -> Any:
    """``text`` decoded when it holds a JSON object, array or encoded string, otherwise ``None``."""
    body = text.strip()
    if not body or body[0] not in "{[\"":
        return None
    try:
        return json.loads(body)
    except (ValueError, RecursionError):  # not JSON; HeavenBase cannot decode it into a spec either
        return None
