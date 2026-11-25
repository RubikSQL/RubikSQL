"""rubiksql.spec/1: the declarative composition language of spec rows and its only executor.

A spec row composes text rows (its *parts*) into one model-visible message. The grammar is a
closed JSON data structure with no attribute access, calls, recursion or loop-carried state,
so every spec has a static output bound and can be interpreted by this module with the
standard library alone. :func:`compile_spec` emits the fixed-shape ``compose(parts, params)``
source HeavenBase stores for reference execution; the runtime regenerates it on every load
and compares byte-for-byte.

Grammar::

    Node  := {"lit": str<=64}
           | {"part": slot, "args": {name: Node}}       # args omitted means {}
           | {"param": path}                            # "x" | "x.f" | loop var "c" | "c.f"
           | {"seq": [Node...]}
           | {"if": {"test": Test, "then": Node, "else": Node}}
           | {"join": {"sep": str<=8, "items": List}}
    List  := {"list": [Node...]} | {"concat": [List...]}
           | {"each": {"in": path, "as": id, "index": id?, "start": 0|1, "where": Test?, "max": 1..64, "body": Node}}
    Test  := {"truthy": path} | {"not": Test} | {"eq": [path, scalar]}
           | {"gt"|"ge"|"lt"|"le": [path | {"len": path}, int]}

Parameter types are JSON values: ``"str"``, ``"int"``, ``"bool"``, ``{"list": T}`` or
``{"record": {field: T}}``. In a stored spec a ``param`` used as text must be a ``str``; ``int``
and ``bool`` values may only flow into part arguments (formatted by ``str.format``) and tests.
A *derived* spec — the tree :mod:`doc` builds from a document — may also use an ``int`` or
``bool`` as text (rendered with ``str()``, the formatting a text row's slot applies), and its
literals are not held to the 64-character limit: a document's prose lives in its literals.
"""
from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Union

from .canon import canon, escape
from .policy import (
    EACH_MAX,
    LIT_MAX_CHARS,
    NF_MAX_NODES,
    OUTPUT_MAX_CHARS,
    PARAM_MAX_CHARS,
    SEP_MAX_CHARS,
    SPEC_MAX_BYTES,
    SPEC_MAX_DEPTH,
    SPEC_MAX_NODES,
    PolicyError,
    check_identifier,
)

__all__ = [
    "ANY_SCALAR",
    "INT_MAX_CHARS",
    "SCHEMA_SPEC",
    "Chunk",
    "Composer",
    "Coverage",
    "PartSig",
    "RenderContext",
    "RenderError",
    "Span",
    "Spec",
    "SpecError",
    "TextLeaf",
    "Type",
    "check_params",
    "check_type",
    "check_value",
    "compile_spec",
    "parse_spec",
    "spans_of",
    "type_name",
]

SCHEMA_SPEC = "rubiksql.spec/1"
#: Characters the static bound charges for one formatted int or bool.
INT_MAX_CHARS = 20
#: The value :meth:`Composer.bound` reports once a bound is known to exceed the cap: the exact figure
#: is not computed, because the computation stops as soon as the cap is crossed.
_SATURATED = OUTPUT_MAX_CHARS + 1
#: Type language: a scalar name or a one-key mapping ``{"list": T}`` / ``{"record": {...}}``.
Type = Union[str, Mapping[str, Any]]
#: ``want`` marker for part arguments of text rows: any of str, int, bool.
ANY_SCALAR = "any-scalar"
_SCALARS = ("str", "int", "bool")


class SpecError(ValueError):
    """A spec violates the grammar, the static typing or a size bound; the message names the node path."""


class RenderError(ValueError):
    """A render received parameters outside their declared types or exceeded a size cap."""


# ---------------------------------------------------------------------------------------
# Types


def check_type(value: Any, where: str = "type") -> Type:
    """Validate a parameter type expression and return it in normalized form.

    Args:
        value (Any): ``"str"``, ``"int"``, ``"bool"``, ``{"list": T}`` or ``{"record": {name: T}}``.
        where (str): Location named in the error message.

    Returns:
        Type: The same type with mappings copied as plain dicts.

    Raises:
        SpecError: If the expression is not one of the supported forms or a record field name is invalid.
    """
    if isinstance(value, str):
        if value not in _SCALARS:
            raise SpecError(f"{where}: unknown type {value!r}; expected one of {list(_SCALARS)}")
        return value
    if isinstance(value, Mapping) and len(value) == 1:
        (key, inner), = value.items()
        if key == "list":
            return {"list": check_type(inner, f"{where}.list")}
        if key == "record":
            if not isinstance(inner, Mapping) or not inner:
                raise SpecError(f"{where}: record type needs a non-empty field mapping")
            fields = {}
            for field, field_type in inner.items():
                try:
                    check_identifier(field, "record field")
                except PolicyError as error:
                    raise SpecError(f"{where}: {error}") from error
                fields[field] = check_type(field_type, f"{where}.{field}")
            return {"record": fields}
    raise SpecError(f"{where}: unknown type expression {value!r}")


def type_name(value: Type) -> str:
    """Render a type expression for error messages, e.g. ``list[{syn: str, near: bool}]``.

    Args:
        value (Type): A normalized type expression.

    Returns:
        str: Compact human-readable spelling.
    """
    if isinstance(value, str):
        return value
    if "list" in value:
        return f"list[{type_name(value['list'])}]"
    fields = ", ".join(f"{k}: {type_name(v)}" for k, v in value["record"].items())
    return "{" + fields + "}"


def check_value(value: Any, expected: Type, where: str, *, param_max_chars: int | None = None) -> None:
    """Check one render argument against its declared type.

    Args:
        value (Any): The argument as passed by the caller.
        expected (Type): The declared type.
        where (str): Dotted location named in the error message.
        param_max_chars (int | None): Longest accepted string, for a static check that enforces the
            ``PARAM_MAX_CHARS`` assumption on authored cells. ``None``, the runtime default, accepts any
            length: data parameters such as a result preview are as long as the data makes them, and the
            only runtime size limit is the ``OUTPUT_MAX_CHARS`` hard cap of :class:`RenderContext`.

    Returns:
        None: This function does not return a value.

    Raises:
        RenderError: If the value's exact type does not match (``bool`` is not an ``int`` here), a string
            is longer than a given ``param_max_chars``, a list element mismatches, or a record's keys differ.
    """
    if expected == "str":
        if type(value) is not str:
            raise RenderError(f"{where}: expected str, got {type(value).__name__}")
        if param_max_chars is not None and len(value) > param_max_chars:
            raise RenderError(f"{where}: string of {len(value)} characters exceeds the {param_max_chars} cap")
        return
    if expected == "int":
        if type(value) is not int:
            raise RenderError(f"{where}: expected int, got {type(value).__name__}")
        return
    if expected == "bool":
        if type(value) is not bool:
            raise RenderError(f"{where}: expected bool, got {type(value).__name__}")
        return
    if "list" in expected:
        if not isinstance(value, (list, tuple)):
            raise RenderError(f"{where}: expected a list, got {type(value).__name__}")
        for index, item in enumerate(value):
            check_value(item, expected["list"], f"{where}[{index}]", param_max_chars=param_max_chars)
        return
    fields = expected["record"]
    if not isinstance(value, Mapping):
        raise RenderError(f"{where}: expected a record with fields {sorted(fields)}, got {type(value).__name__}")
    if set(value) != set(fields):
        raise RenderError(f"{where}: record keys {sorted(value)} differ from the declared {sorted(fields)}")
    for field, field_type in fields.items():
        check_value(value[field], field_type, f"{where}.{field}", param_max_chars=param_max_chars)


def check_params(declared: Mapping[str, Type], values: Mapping[str, Any], where: str, *, param_max_chars: int | None = None) -> None:
    """Check a whole render argument mapping against the declared parameters.

    Args:
        declared (Mapping[str, Type]): Declared parameter types.
        values (Mapping[str, Any]): Arguments passed to the render.
        where (str): Row or entry name for the error message.
        param_max_chars (int | None): Longest accepted string value; ``None`` (the runtime default) sets no
            per-value limit, see :func:`check_value`.

    Returns:
        None: This function does not return a value.

    Raises:
        RenderError: If the argument names differ from the declared ones or any value fails :func:`check_value`.
    """
    if set(values) != set(declared):
        missing = sorted(set(declared) - set(values))
        extra = sorted(set(values) - set(declared))
        raise RenderError(f"{where}: parameters differ from the declaration (missing {missing}, unexpected {extra})")
    for name, expected in declared.items():
        check_value(values[name], expected, f"{where}.{name}", param_max_chars=param_max_chars)


# ---------------------------------------------------------------------------------------
# Parsed nodes


@dataclass(frozen=True, slots=True)
class Lit:
    """A literal of at most 64 characters."""

    text: str
    path: str


@dataclass(frozen=True, slots=True)
class Param:
    """A reference to a declared parameter or loop variable, optionally with one field.

    ``text`` marks an ``int`` or ``bool`` used as text in a derived spec; it renders as ``str(value)``.
    """

    segments: tuple[str, ...]
    type: Type
    path: str
    text: bool = False


@dataclass(frozen=True, slots=True)
class Part:
    """A call of one part with keyword arguments, sorted by name."""

    slot: str
    args: tuple[tuple[str, "Node"], ...]
    path: str


@dataclass(frozen=True, slots=True)
class Seq:
    """Concatenation of string nodes."""

    items: tuple["Node", ...]
    path: str


@dataclass(frozen=True, slots=True)
class If:
    """A two-way branch whose arms share one type."""

    test: "Test"
    then: "Node"
    else_: "Node"
    type: Type
    path: str


@dataclass(frozen=True, slots=True)
class Join:
    """``sep.join(items)`` over a list node."""

    sep: str
    items: "ListNode"
    path: str


@dataclass(frozen=True, slots=True)
class ListLit:
    """An explicit list of string nodes."""

    items: tuple["Node", ...]
    path: str


@dataclass(frozen=True, slots=True)
class Concat:
    """Concatenation of list nodes."""

    items: tuple["ListNode", ...]
    path: str


@dataclass(frozen=True, slots=True)
class Each:
    """A bounded loop over a list parameter producing one string per kept element.

    ``zero_unreachable`` is set when an enclosing ``if`` already asserts the source list truthy on
    this branch, so a zero-iteration render cannot exist and coverage does not demand one.
    """

    source: Param
    var: str
    index: str | None
    start: int
    where: "Test | None"
    max: int
    body: "Node"
    path: str
    zero_unreachable: bool = False


@dataclass(frozen=True, slots=True)
class Truthy:
    """``bool(value)``."""

    operand: Param


@dataclass(frozen=True, slots=True)
class Not:
    """Negation of a test."""

    test: "Test"


@dataclass(frozen=True, slots=True)
class Eq:
    """Equality against a scalar literal of the operand's type."""

    operand: Param
    scalar: str | int | bool


@dataclass(frozen=True, slots=True)
class Cmp:
    """An ordered comparison of an int, or of a length, against an int literal."""

    op: str
    operand: Param
    length: bool
    value: int


Node = Union[Lit, Param, Part, Seq, If, Join]
ListNode = Union[ListLit, Concat, Each]
Test = Union[Truthy, Not, Eq, Cmp]

_CMP_OPS = ("gt", "ge", "lt", "le")
_CMP_PY = {"gt": ">", "ge": ">=", "lt": "<", "le": "<="}


def _node_type(node: Node) -> Type:
    if isinstance(node, Param):
        return "str" if node.text else node.type
    if isinstance(node, If):
        return node.type
    return "str"


@dataclass(frozen=True)
class PartSig:
    """What a part accepts: ``kind`` and its parameter types (``None`` = any scalar, for text slots)."""

    kind: str
    params: Mapping[str, Type | None]


@dataclass(frozen=True)
class Spec:
    """A validated, typed spec ready for interpretation, bounding, normalization and compilation."""

    root: Node
    params: Mapping[str, Type]
    node_count: int
    depth: int
    slots: tuple[str, ...]

    def branch_points(self) -> list[tuple[str, str]]:
        """List the structural coverage obligations of this spec.

        Args:
            None.

        Returns:
            list[tuple[str, str]]: ``(node_path, kind)`` pairs with kind ``"if"``, ``"each"`` or ``"where"``.
        """
        found: list[tuple[str, str]] = []

        def visit(node: Any) -> None:
            if isinstance(node, If):
                found.append((node.path, "if"))
                visit(node.then)
                visit(node.else_)
            elif isinstance(node, Part):
                for _name, arg in node.args:
                    visit(arg)
            elif isinstance(node, (Seq, ListLit, Concat)):
                for item in node.items:
                    visit(item)
            elif isinstance(node, Join):
                visit(node.items)
            elif isinstance(node, Each):
                found.append((node.path, "each-nonzero" if node.zero_unreachable else "each"))
                if node.where is not None:
                    found.append((node.path, "where"))
                visit(node.body)

        visit(self.root)
        return found


class _Parser:
    def __init__(self, params: Mapping[str, Type], part_sigs: Mapping[str, PartSig] | None, part_names: Iterable[str] | None, derived: bool = False, bounded: bool = True) -> None:
        self.params = params
        self.part_sigs = part_sigs
        self.part_names = None if part_names is None else set(part_names)
        self.derived = derived
        self.bounded = bounded
        self.nodes = 0
        self.max_depth = 0
        self.slots: list[str] = []

    def count(self, depth: int, path: str) -> None:
        self.nodes += 1
        if self.bounded and self.nodes > SPEC_MAX_NODES:
            raise SpecError(f"spec has more than {SPEC_MAX_NODES} nodes")
        if self.bounded and depth > SPEC_MAX_DEPTH:
            raise SpecError(f"{path}: nesting deeper than {SPEC_MAX_DEPTH}")
        self.max_depth = max(self.max_depth, depth)

    @staticmethod
    def keys(obj: Any, path: str, allowed: Iterable[str], required: Iterable[str]) -> Mapping[str, Any]:
        if not isinstance(obj, Mapping):
            raise SpecError(f"{path}: expected a JSON object, got {type(obj).__name__}")
        unknown = sorted(set(obj) - set(allowed))
        if unknown:
            raise SpecError(f"{path}: unknown keys {unknown}; allowed {sorted(allowed)}")
        missing = sorted(set(required) - set(obj))
        if missing:
            raise SpecError(f"{path}: missing keys {missing}")
        return obj

    def ident(self, value: Any, path: str, label: str) -> str:
        try:
            return check_identifier(value, label)
        except PolicyError as error:
            raise SpecError(f"{path}: {error}") from error

    def ref(self, value: Any, env: Mapping[str, Type], path: str) -> Param:
        if not isinstance(value, str) or not value:
            raise SpecError(f"{path}: param path must be a non-empty string")
        segments = tuple(value.split("."))
        if len(segments) > 2:
            raise SpecError(f"{path}: param path {value!r} may have at most one field access")
        for segment in segments:
            self.ident(segment, path, "param path segment")
        head = segments[0]
        if head in env:
            current: Type = env[head]
        elif head in self.params:
            current = self.params[head]
        else:
            raise SpecError(f"{path}: unknown param {head!r}; declared {sorted(self.params)}, in scope {sorted(env)}")
        if len(segments) == 2:
            if not isinstance(current, Mapping) or "record" not in current:
                raise SpecError(f"{path}: {head!r} has type {type_name(current)}, which has no field {segments[1]!r}")
            fields = current["record"]
            if segments[1] not in fields:
                raise SpecError(f"{path}: {head!r} has no field {segments[1]!r}; fields {sorted(fields)}")
            current = fields[segments[1]]
        return Param(segments, current, path)

    def node(self, obj: Any, path: str, env: Mapping[str, Type], depth: int, want: Any, guards: frozenset = frozenset()) -> Node:
        self.count(depth, path)
        if not isinstance(obj, Mapping) or not obj:
            raise SpecError(f"{path}: a node must be a non-empty JSON object")
        head = "part" if "part" in obj else next(iter(obj))
        if head == "lit":
            self.keys(obj, path, ["lit"], ["lit"])
            text = obj["lit"]
            if not isinstance(text, str):
                raise SpecError(f"{path}: lit must be a string")
            if len(text) > LIT_MAX_CHARS and not self.derived:
                raise SpecError(f"{path}: lit has {len(text)} characters; the limit is {LIT_MAX_CHARS} (prose belongs in a text row)")
            node: Node = Lit(text, path)
        elif head == "param":
            self.keys(obj, path, ["param"], ["param"])
            node = self.ref(obj["param"], env, path)
        elif head == "part":
            self.keys(obj, path, ["part", "args"], ["part"])
            slot = self.ident(obj.get("part"), path, "part slot")
            args_obj = obj.get("args", {})
            if not isinstance(args_obj, Mapping):
                raise SpecError(f"{path}: args must be a JSON object")
            if self.part_names is not None and slot not in self.part_names:
                raise SpecError(f"{path}: part {slot!r} is not in meta.parts {sorted(self.part_names)}")
            sig = None if self.part_sigs is None else self.part_sigs.get(slot)
            if self.part_sigs is not None and sig is None:
                raise SpecError(f"{path}: part {slot!r} is not in meta.parts {sorted(self.part_sigs)}")
            if sig is not None and set(args_obj) != set(sig.params):
                raise SpecError(f"{path}: args {sorted(args_obj)} must equal the parameters {sorted(sig.params)} of part {slot!r}")
            args = []
            for arg_name in sorted(args_obj):
                self.ident(arg_name, path, "arg name")
                arg_want: Any = None
                if sig is not None:
                    arg_want = ANY_SCALAR if sig.params[arg_name] is None else sig.params[arg_name]
                args.append((arg_name, self.node(args_obj[arg_name], f"{path}.part.args.{arg_name}", env, depth + 1, arg_want, guards)))
            if slot not in self.slots:
                self.slots.append(slot)
            node = Part(slot, tuple(args), path)
        elif head == "seq":
            self.keys(obj, path, ["seq"], ["seq"])
            items = obj["seq"]
            if not isinstance(items, list):
                raise SpecError(f"{path}: seq must be a list")
            node = Seq(tuple(self.node(item, f"{path}.seq[{i}]", env, depth + 1, "str", guards) for i, item in enumerate(items)), path)
        elif head == "if":
            inner = self.keys(obj["if"] if isinstance(obj, Mapping) and set(obj) == {"if"} else None, path, ["test", "then", "else"], ["test", "then", "else"])
            test = self.test(inner["test"], f"{path}.if.test", env, depth + 1)
            then_guards, else_guards = guards, guards
            if isinstance(test, Truthy):
                then_guards = guards | {test.operand.segments}
            elif isinstance(test, Not) and isinstance(test.test, Truthy):
                else_guards = guards | {test.test.operand.segments}
            then = self.node(inner["then"], f"{path}.if.then", env, depth + 1, want, then_guards)
            else_ = self.node(inner["else"], f"{path}.if.else", env, depth + 1, want, else_guards)
            then_type, else_type = _node_type(then), _node_type(else_)
            if then_type != else_type:
                raise SpecError(f"{path}: if branches have types {type_name(then_type)} and {type_name(else_type)}")
            node = If(test, then, else_, then_type, path)
        elif head == "join":
            inner = self.keys(obj["join"] if set(obj) == {"join"} else None, path, ["sep", "items"], ["sep", "items"])
            sep = inner["sep"]
            if not isinstance(sep, str):
                raise SpecError(f"{path}: join.sep must be a string")
            if len(sep) > SEP_MAX_CHARS:
                raise SpecError(f"{path}: join.sep has {len(sep)} characters; the limit is {SEP_MAX_CHARS}")
            node = Join(sep, self.list(inner["items"], f"{path}.join.items", env, depth + 1, guards), path)
        else:
            raise SpecError(f"{path}: unknown node type {head!r}")
        if self.derived and want == "str" and isinstance(node, Param) and node.type in ("int", "bool"):
            node = Param(node.segments, node.type, node.path, True)
        actual = _node_type(node)
        if want == ANY_SCALAR:
            if actual not in _SCALARS:
                raise SpecError(f"{path}: a text slot argument must be str, int or bool, got {type_name(actual)}")
        elif want is not None and actual != want:
            raise SpecError(f"{path}: expected {type_name(want)}, got {type_name(actual)}")
        return node

    def list(self, obj: Any, path: str, env: Mapping[str, Type], depth: int, guards: frozenset = frozenset()) -> ListNode:
        self.count(depth, path)
        if not isinstance(obj, Mapping) or len(obj) != 1:
            raise SpecError(f"{path}: a list node must be a JSON object with exactly one key")
        head = next(iter(obj))
        if head == "list":
            items = obj["list"]
            if not isinstance(items, list):
                raise SpecError(f"{path}: list must be a JSON array")
            return ListLit(tuple(self.node(item, f"{path}.list[{i}]", env, depth + 1, "str", guards) for i, item in enumerate(items)), path)
        if head == "concat":
            items = obj["concat"]
            if not isinstance(items, list):
                raise SpecError(f"{path}: concat must be a JSON array")
            return Concat(tuple(self.list(item, f"{path}.concat[{i}]", env, depth + 1, guards) for i, item in enumerate(items)), path)
        if head == "each":
            inner = self.keys(obj["each"], path, ["in", "as", "index", "start", "where", "max", "body"], ["in", "as", "max", "body"])
            source = self.ref(inner["in"], env, f"{path}.each.in")
            if not isinstance(source.type, Mapping) or "list" not in source.type:
                raise SpecError(f"{path}.each.in: {inner['in']!r} has type {type_name(source.type)}, expected a list")
            var = self.ident(inner["as"], f"{path}.each.as", "loop variable")
            index = None if "index" not in inner else self.ident(inner["index"], f"{path}.each.index", "index variable")
            if index == var or var in env or (index is not None and index in env) or var in self.params or (index is not None and index in self.params):
                raise SpecError(f"{path}: loop variables {var!r}/{index!r} must be fresh names")
            start = inner.get("start", 0)
            if type(start) is not int or start not in (0, 1):
                raise SpecError(f"{path}.each.start: must be 0 or 1")
            if "start" in inner and index is None:
                raise SpecError(f"{path}.each.start: start is meaningless without an index variable")
            maximum = inner["max"]
            if type(maximum) is not int or not 1 <= maximum <= EACH_MAX:
                raise SpecError(f"{path}.each.max: must be an int in 1..{EACH_MAX}")
            inner_env = dict(env)
            inner_env[var] = source.type["list"]
            if index is not None:
                inner_env[index] = "int"
            where = None if "where" not in inner else self.test(inner["where"], f"{path}.each.where", inner_env, depth + 1)
            body = self.node(inner["body"], f"{path}.each.body", inner_env, depth + 1, "str")
            return Each(source, var, index, start, where, maximum, body, path, source.segments in guards)
        raise SpecError(f"{path}: unknown list node type {head!r}")

    def test(self, obj: Any, path: str, env: Mapping[str, Type], depth: int) -> Test:
        self.count(depth, path)
        if not isinstance(obj, Mapping) or len(obj) != 1:
            raise SpecError(f"{path}: a test must be a JSON object with exactly one key")
        head = next(iter(obj))
        value = obj[head]
        if head == "truthy":
            operand = self.ref(value, env, path)
            if operand.type not in _SCALARS and not (isinstance(operand.type, Mapping) and "list" in operand.type):
                raise SpecError(f"{path}: truthy needs a scalar or list, got {type_name(operand.type)}")
            return Truthy(operand)
        if head == "not":
            return Not(self.test(value, f"{path}.not", env, depth + 1))
        if head == "eq":
            if not isinstance(value, list) or len(value) != 2:
                raise SpecError(f"{path}: eq needs [path, scalar]")
            operand = self.ref(value[0], env, path)
            scalar = value[1]
            scalar_type = "bool" if type(scalar) is bool else "int" if type(scalar) is int else "str" if isinstance(scalar, str) else None
            if scalar_type is None:
                raise SpecError(f"{path}: eq scalar must be a str, int or bool literal")
            if operand.type != scalar_type:
                raise SpecError(f"{path}: eq compares {type_name(operand.type)} with a {scalar_type} literal")
            return Eq(operand, scalar)
        if head in _CMP_OPS:
            if not isinstance(value, list) or len(value) != 2 or type(value[1]) is not int:
                raise SpecError(f"{path}: {head} needs [path | {{len: path}}, int]")
            left = value[0]
            if isinstance(left, Mapping):
                self.keys(left, path, ["len"], ["len"])
                operand = self.ref(left["len"], env, f"{path}.len")
                if operand.type != "str" and not (isinstance(operand.type, Mapping) and "list" in operand.type):
                    raise SpecError(f"{path}: len needs a str or list, got {type_name(operand.type)}")
                return Cmp(head, operand, True, value[1])
            operand = self.ref(left, env, path)
            if operand.type != "int":
                raise SpecError(f"{path}: {head} compares an int, got {type_name(operand.type)}")
            return Cmp(head, operand, False, value[1])
        raise SpecError(f"{path}: unknown test type {head!r}")


def parse_spec(
    spec: Any,
    *,
    params: Mapping[str, Any],
    part_sigs: Mapping[str, PartSig] | None = None,
    part_names: Iterable[str] | None = None,
    derived: bool = False,
    bounded: bool = True,
) -> Spec:
    """Validate a spec's grammar, size bounds and static types.

    Args:
        spec (Any): The JSON value of ``meta.spec``.
        params (Mapping[str, Any]): Declared parameter types (``meta.params``), validated with :func:`check_type`.
        part_sigs (Mapping[str, PartSig] | None): Signatures of the parts by slot. When given, every
            ``part`` node's args are checked against them; when ``None`` only the grammar is checked.
        part_names (Iterable[str] | None): Slot names to accept when signatures are unavailable.
        derived (bool): The spec was derived from a document (:mod:`doc`): the 16 KiB ``SPEC_MAX_BYTES`` and the
            64-character literal limit do not apply, and an ``int`` or ``bool`` may be used as text. The node,
            depth and ``each.max`` limits apply either way.
        bounded (bool): Enforce the node-count and nesting-depth limits. Only :func:`doc.document_of` passes
            ``False``, to verify the document it prints for an existing composition: identity must not fail
            where a row, its parts inlined, legitimately nests deeper than one authored spec may.

    Returns:
        Spec: The typed tree with its node count, depth and the slots it uses.

    Raises:
        SpecError: On any grammar, typing or bound violation; the message starts with the node path.
    """
    if not isinstance(params, Mapping):
        raise SpecError("params must be a mapping of name to type")
    typed: dict[str, Type] = {}
    for name, value in params.items():
        try:
            check_identifier(name, "param name")
        except PolicyError as error:
            raise SpecError(str(error)) from error
        typed[name] = check_type(value, f"params.{name}")
    if not derived:
        size = len(canon(spec).encode("utf-8"))
        if size > SPEC_MAX_BYTES:
            raise SpecError(f"spec is {size} bytes; the limit is {SPEC_MAX_BYTES}")
    parser = _Parser(typed, part_sigs, part_names, derived, bounded)
    root = parser.node(spec, "$", {}, 1, "str")
    return Spec(root, typed, parser.nodes, parser.max_depth, tuple(parser.slots))


# ---------------------------------------------------------------------------------------
# Interpretation


@dataclass(frozen=True, slots=True)
class Span:
    """Provenance of one contiguous range of rendered output."""

    start: int
    end: int
    name: str
    version: str
    node: str


#: One rendered piece and its provenance ``(row name, row version, node path)``.
Chunk = tuple[str, tuple[str, str, str]]


class Coverage:
    """Structural coverage accumulated across renders: if arms, each iteration classes, where outcomes."""

    def __init__(self) -> None:
        self.hits: dict[tuple[str, str, str], set[str]] = {}

    def hit(self, row: str, path: str, kind: str, outcome: str) -> None:
        """Record one structural outcome.

        Args:
            row (str): Spec row name.
            path (str): Node path inside that row.
            kind (str): ``"if"``, ``"each"`` or ``"where"``.
            outcome (str): ``"then"``/``"else"``, ``"0"``/``"1"``/``"2+"`` or ``"true"``/``"false"``.

        Returns:
            None: This method does not return a value.
        """
        self.hits.setdefault((row, path, kind), set()).add(outcome)

    def missing(self, composers: Iterable["Composer"]) -> list[str]:
        """List the obligations the given spec rows still leave unmet.

        The gate requires both arms of every ``if``, a zero-iteration and a two-or-more-iteration
        render of every ``each`` (only the latter when the loop sits under an ``if`` that already
        asserts its list truthy), and both outcomes of every ``where``.

        Args:
            composers (Iterable[Composer]): Spec rows whose obligations are checked.

        Returns:
            list[str]: Human-readable ``"row path kind:outcome"`` lines, sorted.
        """
        required = {"if": ("then", "else"), "each": ("0", "2+"), "each-nonzero": ("2+",), "where": ("true", "false")}
        gaps = []
        for composer in composers:
            for path, kind in composer.spec.branch_points():
                seen = self.hits.get((composer.name, path, "each" if kind == "each-nonzero" else kind), set())
                for outcome in required[kind]:
                    if outcome not in seen:
                        gaps.append(f"{composer.name} {path} {kind}:{outcome}")
        return sorted(gaps)


class RenderContext:
    """Per-render limits and observers shared by every row taking part in one render.

    ``max_chars`` is the one size limit the runtime enforces (trust boundary §1: the static bound
    the gate computes and "the same hard cap at run time"). ``param_max_chars`` is ``None`` unless a
    static check asks for the per-string assumption to be enforced on authored values.
    """

    __slots__ = ("max_chars", "param_max_chars", "coverage", "total")

    def __init__(self, *, max_chars: int = OUTPUT_MAX_CHARS, param_max_chars: int | None = None, coverage: Coverage | None = None) -> None:
        self.max_chars = max_chars
        self.param_max_chars = param_max_chars
        self.coverage = coverage
        self.total = 0

    def chunk(self, text: str, name: str, version: str, node: str) -> Chunk:
        """Account one output piece against the hard cap.

        Args:
            text (str): Rendered text.
            name (str): Producing row name.
            version (str): Producing row version.
            node (str): Node path, or ``"text"`` for a whole text row.

        Returns:
            Chunk: The piece with its provenance.

        Raises:
            RenderError: If the running total would exceed ``max_chars``.
        """
        self.total += len(text)
        if self.total > self.max_chars:
            raise RenderError(f"render exceeded the {self.max_chars}-character hard cap at {name}:{node}")
        return (text, (name, version, node))


def spans_of(chunks: Sequence[Chunk]) -> list[Span]:
    """Turn chunks into absolute spans, merging adjacent pieces from the same node.

    Args:
        chunks (Sequence[Chunk]): Rendered pieces in output order.

    Returns:
        list[Span]: Non-empty spans covering the output exactly.
    """
    spans: list[Span] = []
    position = 0
    for text, (name, version, node) in chunks:
        if not text:
            continue
        end = position + len(text)
        if spans and (spans[-1].name, spans[-1].version, spans[-1].node) == (name, version, node) and spans[-1].end == position:
            spans[-1] = Span(spans[-1].start, end, name, version, node)
        else:
            spans.append(Span(position, end, name, version, node))
        position = end
    return spans


NF = Any


class _NfBuild:
    """Bookkeeping of one normal-form computation: exact node counts and the size cap.

    A normal form is a DAG — every reference to a part shares the part's memoised form — so
    counting by object identity keeps the cost proportional to the distinct nodes, and checking
    the running count while a ``seq`` or ``concat`` is flattened stops a fan-out before it can
    allocate past ``NF_MAX_NODES``.
    """

    __slots__ = ("_sizes",)

    def __init__(self) -> None:
        # id -> (form, size); the form is kept so its id cannot be reused while the build lives
        self._sizes: dict[int, tuple[NF, int]] = {}

    def size(self, form: NF) -> int:
        cached = self._sizes.get(id(form))
        if cached is not None:
            return cached[1]
        total = 1 + sum(self.size(item) for item in form if isinstance(item, list))
        self._sizes[id(form)] = (form, total)
        return total

    def check(self, total: int, where: str) -> None:
        if total > NF_MAX_NODES:
            raise SpecError(f"{where}: normal form expands to more than {NF_MAX_NODES} nodes")


class TextLeaf:
    """A text row: human-readable text plus slots, rendered through its escaped template."""

    kind = "text"

    def __init__(self, name: str, version: str, text: str, slots: Sequence[str]) -> None:
        self.name = name
        self.version = version
        self.text = text
        self.slots = tuple(slots)
        self.template = escape(text, slots)
        self.params: Mapping[str, Type | None] = {slot: None for slot in self.slots}
        self.depth = 1
        self._fields = [(literal, field, conversion) for literal, field, _spec, conversion in string.Formatter().parse(self.template)]

    def signature(self) -> PartSig:
        """Describe this row as a part.

        Args:
            None.

        Returns:
            PartSig: ``kind="text"`` with every slot accepting any scalar.
        """
        return PartSig("text", self.params)

    def render_chunks(self, ctx: RenderContext, values: Mapping[str, Any]) -> list[Chunk]:
        """Render into one chunk attributed to this row.

        Args:
            ctx (RenderContext): Limits and observers of the current render.
            values (Mapping[str, Any]): Slot values; str, int or bool, one per declared slot.

        Returns:
            list[Chunk]: A single chunk.

        Raises:
            RenderError: If the slot names differ from the declaration, a value is not a scalar, a string
                exceeds a per-value cap the context sets, or the output exceeds the context's hard cap.
        """
        if set(values) != set(self.slots):
            raise RenderError(f"{self.name}: slots differ from the declaration {list(self.slots)}; got {sorted(values)}")
        for slot, value in values.items():
            if type(value) not in (str, int, bool):
                raise RenderError(f"{self.name}.{slot}: slot values must be str, int or bool, got {type(value).__name__}")
            if ctx.param_max_chars is not None and type(value) is str and len(value) > ctx.param_max_chars:
                raise RenderError(f"{self.name}.{slot}: string of {len(value)} characters exceeds the {ctx.param_max_chars} cap")
        return [ctx.chunk(self.template.format(**values), self.name, self.version, "text")]

    def render(self, ctx: RenderContext | None = None, **values: Any) -> str:
        """Render the row to a string.

        Args:
            ctx (RenderContext | None): Limits and observers; a default context when omitted.
            **values (Any): Slot values.

        Returns:
            str: The rendered text, byte-identical to HeavenBase executing the stored template.
        """
        return "".join(text for text, _src in self.render_chunks(ctx or RenderContext(), values))

    def bound(self, arg_bounds: Mapping[str, int]) -> int:
        """Static upper bound of the rendered length given per-slot argument bounds.

        Args:
            arg_bounds (Mapping[str, int]): Longest possible argument per slot, in characters.

        Returns:
            int: Literal characters plus each field's bound (``!r`` charged at four times plus two).
        """
        total = 0
        for literal, field, conversion in self._fields:
            total += len(literal)
            if field is not None:
                total += arg_bounds[field] * 4 + 2 if conversion == "r" else arg_bounds[field]
        return total

    def normal_form(self, args: Mapping[str, NF], arg_types: Mapping[str, Type]) -> NF:
        """Normal form of this row with its slots substituted by argument normal forms.

        Args:
            args (Mapping[str, NF]): Normal form of the argument node per slot.
            arg_types (Mapping[str, Type]): Static type of each argument, so ``!r`` on an int folds away.

        Returns:
            NF: A nameless JSON-able tree.
        """
        return self._nf_sized(args, arg_types, _NfBuild())

    def _nf_sized(self, args: Mapping[str, NF], arg_types: Mapping[str, Type], build: _NfBuild) -> NF:
        items: list[NF] = []
        for literal, field, conversion in self._fields:
            if literal:
                items.append(["lit", literal])
            if field is not None:
                node = args[field]
                if conversion == "r" and arg_types[field] == "str":
                    node = ["repr", node]
                items.append(node)
        return _nf_seq(items, build, self.name)


class Composer:
    """A spec row bound to its parts: the runtime's executor for one composition.

    Static bounds and normal forms are memoised per argument tuple: a spec that references a
    part many times costs one traversal of the part per distinct argument shape instead of one
    per reference, which is what keeps a deep fan-out of part references polynomial.
    """

    kind = "spec"

    def __init__(self, name: str, version: str, spec: Spec, parts: Mapping[str, "TextLeaf | Composer"]) -> None:
        self.name = name
        self.version = version
        self.spec = spec
        self.parts = dict(parts)
        self.params: Mapping[str, Type] = dict(spec.params)
        self.depth = 1 + max((part.depth for part in self.parts.values()), default=0)
        self._bound_memo: dict[tuple[tuple[str, int], ...], int] = {}
        self._nf_memo: dict[tuple[str | None, str | None, int], NF] = {}

    def signature(self) -> PartSig:
        """Describe this row as a part.

        Args:
            None.

        Returns:
            PartSig: ``kind="spec"`` with its declared parameter types.
        """
        return PartSig("spec", self.params)

    def render_chunks(self, ctx: RenderContext, values: Mapping[str, Any]) -> list[Chunk]:
        """Interpret the spec against argument values.

        Args:
            ctx (RenderContext): Limits and observers of the current render.
            values (Mapping[str, Any]): Arguments matching ``meta.params`` exactly.

        Returns:
            list[Chunk]: Output pieces in order with per-piece provenance.

        Raises:
            RenderError: On a parameter type mismatch or when a size cap is exceeded.
        """
        check_params(self.params, values, self.name, param_max_chars=ctx.param_max_chars)
        return self._render(self.spec.root, dict(values), ctx)

    def render(self, ctx: RenderContext | None = None, **values: Any) -> str:
        """Render the composition to a string.

        Args:
            ctx (RenderContext | None): Limits and observers; a default context when omitted.
            **values (Any): Arguments matching ``meta.params``.

        Returns:
            str: The rendered text.
        """
        return "".join(text for text, _src in self.render_chunks(ctx or RenderContext(), values))

    def trace(self, ctx: RenderContext | None = None, **values: Any) -> tuple[str, list[Span]]:
        """Render and return the text with the provenance of every range.

        Args:
            ctx (RenderContext | None): Limits and observers; a default context when omitted.
            **values (Any): Arguments matching ``meta.params``.

        Returns:
            tuple[str, list[Span]]: The rendered text and its spans, computed from the interpreter's own
                structure without inserting sentinel characters.
        """
        chunks = self.render_chunks(ctx or RenderContext(), values)
        return "".join(text for text, _src in chunks), spans_of(chunks)

    # -- interpreter -----------------------------------------------------------------

    def _resolve(self, ref: Param, env: Mapping[str, Any]) -> Any:
        value = env[ref.segments[0]]
        return value[ref.segments[1]] if len(ref.segments) == 2 else value

    def _test(self, test: Test, env: Mapping[str, Any]) -> bool:
        if isinstance(test, Truthy):
            return bool(self._resolve(test.operand, env))
        if isinstance(test, Not):
            return not self._test(test.test, env)
        if isinstance(test, Eq):
            return self._resolve(test.operand, env) == test.scalar
        left = self._resolve(test.operand, env)
        if test.length:
            left = len(left)
        if test.op == "gt":
            return left > test.value
        if test.op == "ge":
            return left >= test.value
        if test.op == "lt":
            return left < test.value
        return left <= test.value

    def _value(self, node: Node, env: Mapping[str, Any], ctx: RenderContext) -> Any:
        if isinstance(node, Param):
            return self._resolve(node, env)
        if isinstance(node, Lit):
            return node.text
        if isinstance(node, If):
            taken = self._test(node.test, env)
            if ctx.coverage is not None:
                ctx.coverage.hit(self.name, node.path, "if", "then" if taken else "else")
            return self._value(node.then if taken else node.else_, env, ctx)
        return "".join(text for text, _src in self._render(node, env, ctx))

    def _render(self, node: Node, env: Mapping[str, Any], ctx: RenderContext) -> list[Chunk]:
        if isinstance(node, Lit):
            return [ctx.chunk(node.text, self.name, self.version, node.path)]
        if isinstance(node, Param):
            value = self._resolve(node, env)
            return [ctx.chunk(str(value) if node.text else value, self.name, self.version, node.path)]
        if isinstance(node, Part):
            values = {arg_name: self._value(arg, env, ctx) for arg_name, arg in node.args}
            return self.parts[node.slot].render_chunks(ctx, values)
        if isinstance(node, Seq):
            chunks: list[Chunk] = []
            for item in node.items:
                chunks.extend(self._render(item, env, ctx))
            return chunks
        if isinstance(node, If):
            taken = self._test(node.test, env)
            if ctx.coverage is not None:
                ctx.coverage.hit(self.name, node.path, "if", "then" if taken else "else")
            return self._render(node.then if taken else node.else_, env, ctx)
        chunks = []
        for index, item in enumerate(self._list(node.items, env, ctx)):
            if index and node.sep:
                chunks.append(ctx.chunk(node.sep, self.name, self.version, node.path))
            chunks.extend(item)
        return chunks

    def _list(self, node: ListNode, env: Mapping[str, Any], ctx: RenderContext) -> list[list[Chunk]]:
        if isinstance(node, ListLit):
            return [self._render(item, env, ctx) for item in node.items]
        if isinstance(node, Concat):
            items: list[list[Chunk]] = []
            for part in node.items:
                items.extend(self._list(part, env, ctx))
            return items
        items = []
        source = self._resolve(node.source, env)
        for position, element in enumerate(source[: node.max], node.start):
            scope = dict(env)
            scope[node.var] = element
            if node.index is not None:
                scope[node.index] = position
            if node.where is not None:
                kept = self._test(node.where, scope)
                if ctx.coverage is not None:
                    ctx.coverage.hit(self.name, node.path, "where", "true" if kept else "false")
                if not kept:
                    continue
            items.append(self._render(node.body, scope, ctx))
        if ctx.coverage is not None:
            ctx.coverage.hit(self.name, node.path, "each", "0" if not items else "1" if len(items) == 1 else "2+")
        return items

    # -- static bound ----------------------------------------------------------------

    def bound(self, arg_bounds: Mapping[str, int] | None = None) -> int:
        """Static upper bound of the rendered length in characters.

        Every string parameter site is charged ``PARAM_MAX_CHARS`` (or the caller's bound when this
        row is used as a part), every int or bool site ``INT_MAX_CHARS``, every ``each`` body its
        ``max`` multiplicity; ``if`` takes the larger arm. A bound above ``OUTPUT_MAX_CHARS`` is
        reported as ``OUTPUT_MAX_CHARS + 1``: the traversal stops as soon as the cap is crossed,
        and the result is memoised per argument-bound tuple.

        Args:
            arg_bounds (Mapping[str, int] | None): Bounds of scalar parameters when rendered as a part;
                omitted for an entry, where declared types alone set the bounds.

        Returns:
            int: The bound, exact when it is at most ``OUTPUT_MAX_CHARS``; it never underestimates a
                render that passes :func:`check_params` with the assumed per-string length.
        """
        key = tuple(sorted((arg_bounds or {}).items()))
        hit = self._bound_memo.get(key)
        if hit is None:
            hit = self._bound_node(self.spec.root, dict(key))
            self._bound_memo[key] = hit
        return hit

    def _param_bound(self, ref: Param, given: Mapping[str, int]) -> int:
        if len(ref.segments) == 1 and ref.segments[0] in given:
            return given[ref.segments[0]]
        return PARAM_MAX_CHARS if ref.type == "str" else INT_MAX_CHARS

    def _bound_node(self, node: Node, given: Mapping[str, int]) -> int:
        if isinstance(node, Lit):
            return len(node.text)
        if isinstance(node, Param):
            return self._param_bound(node, given)
        if isinstance(node, Part):
            return min(_SATURATED, self.parts[node.slot].bound({arg_name: self._bound_node(arg, given) for arg_name, arg in node.args}))
        if isinstance(node, Seq):
            total = 0
            for item in node.items:
                total += self._bound_node(item, given)
                if total > OUTPUT_MAX_CHARS:
                    return _SATURATED
            return total
        if isinstance(node, If):
            then = self._bound_node(node.then, given)
            return _SATURATED if then > OUTPUT_MAX_CHARS else max(then, self._bound_node(node.else_, given))
        chars, count = self._bound_list(node.items, given)
        return min(_SATURATED, chars + len(node.sep) * max(count - 1, 0))

    def _bound_list(self, node: ListNode, given: Mapping[str, int]) -> tuple[int, int]:
        if isinstance(node, ListLit):
            chars = 0
            for item in node.items:
                chars += self._bound_node(item, given)
                if chars > OUTPUT_MAX_CHARS:
                    return _SATURATED, len(node.items)
            return chars, len(node.items)
        if isinstance(node, Concat):
            chars = count = 0
            for item in node.items:
                item_chars, item_count = self._bound_list(item, given)
                chars += item_chars
                count += item_count
                if chars > OUTPUT_MAX_CHARS:
                    return _SATURATED, count
            return chars, count
        return min(_SATURATED, self._bound_node(node.body, given) * node.max), node.max

    # -- normal form -----------------------------------------------------------------

    def normal_form(self, args: Mapping[str, NF] | None = None, arg_types: Mapping[str, Type] | None = None, depth: int = 0) -> NF:
        """Canonical semantic form: parts inlined, slots substituted, names erased, constants folded.

        Renaming rows or slots, splitting or merging text rows, and moving literal text between a
        ``lit`` and a text row leave the normal form unchanged; any change that can alter the output
        (a literal, a condition, a bound, structure) changes it.

        The result is memoised per ``(args, arg_types, depth)`` and shared between callers: treat it as
        read-only.

        Args:
            args (Mapping[str, NF] | None): Normal forms substituted for this row's parameters when it
                is used as a part; an entry uses ``["param", name]`` for each declared parameter.
            arg_types (Mapping[str, Type] | None): Static types of those arguments.
            depth (int): Loop nesting depth already in scope, which numbers loop variables canonically.

        Returns:
            NF: A JSON-able tree free of row, slot and variable names.

        Raises:
            SpecError: If the form would expand to more than ``NF_MAX_NODES`` logical nodes.
        """
        return self._nf_sized(args, arg_types, depth, _NfBuild())

    def _nf_sized(self, args: Mapping[str, NF] | None, arg_types: Mapping[str, Type] | None, depth: int, build: _NfBuild) -> NF:
        key = (None if args is None else canon(dict(args)), None if arg_types is None else canon(dict(arg_types)), depth)
        form = self._nf_memo.get(key)
        if form is None:
            env: dict[str, tuple[NF, Type]] = {}
            for name, declared in self.params.items():
                if args is not None:
                    env[name] = (args[name], arg_types[name] if arg_types else declared)
                else:
                    env[name] = (["param", name], declared)
            form = self._nf(self.spec.root, env, depth, build)
            build.check(build.size(form), self.name)
            self._nf_memo[key] = form
        return form

    def _nf_ref(self, ref: Param, env: Mapping[str, tuple[NF, Type]]) -> NF:
        base, _type = env[ref.segments[0]]
        return ["field", base, ref.segments[1]] if len(ref.segments) == 2 else base

    def _nf_test(self, test: Test, env: Mapping[str, tuple[NF, Type]]) -> NF:
        if isinstance(test, Truthy):
            return ["truthy", self._nf_ref(test.operand, env)]
        if isinstance(test, Not):
            return ["not", self._nf_test(test.test, env)]
        if isinstance(test, Eq):
            return ["eq", self._nf_ref(test.operand, env), test.scalar]
        operand = self._nf_ref(test.operand, env)
        return [test.op, ["len", operand] if test.length else operand, test.value]

    def _nf(self, node: Node, env: Mapping[str, tuple[NF, Type]], depth: int, build: _NfBuild) -> NF:
        if isinstance(node, Lit):
            return ["lit", node.text]
        if isinstance(node, Param):
            return self._nf_ref(node, env)
        if isinstance(node, Part):
            arg_forms = {arg_name: self._nf(arg, env, depth, build) for arg_name, arg in node.args}
            arg_types = {arg_name: _node_type(arg) for arg_name, arg in node.args}
            part = self.parts[node.slot]
            if isinstance(part, TextLeaf):
                return part._nf_sized(arg_forms, arg_types, build)
            return part._nf_sized(arg_forms, arg_types, depth, build)
        if isinstance(node, Seq):
            return _nf_seq([self._nf(item, env, depth, build) for item in node.items], build, self.name)
        if isinstance(node, If):
            then, else_ = self._nf(node.then, env, depth, build), self._nf(node.else_, env, depth, build)
            if then == else_:
                return then
            form = ["if", self._nf_test(node.test, env), then, else_]
            build.check(build.size(form), self.name)
            return form
        items = self._nf_list(node.items, env, depth, build)
        if items[0] == "list":
            if not items[1]:
                return ["lit", ""]
            if len(items[1]) == 1:
                return items[1][0]
        return ["join", node.sep, items]

    def _nf_list(self, node: ListNode, env: Mapping[str, tuple[NF, Type]], depth: int, build: _NfBuild) -> NF:
        if isinstance(node, ListLit):
            forms: list[NF] = []
            total = 2  # the list node and its item container
            for item in node.items:
                form = self._nf(item, env, depth, build)
                total += build.size(form)
                build.check(total, self.name)
                forms.append(form)
            return ["list", forms]
        if isinstance(node, Concat):
            # adjacent lists merge; the merged list is built in a buffer this call owns, never by
            # mutating a piece, because pieces are shared with memoised forms
            flat: list[NF] = []
            merged: list[NF] | None = None
            total = 2
            for item in node.items:
                form = self._nf_list(item, env, depth, build)
                for piece in form[1] if form[0] == "concat" else [form]:
                    if piece[0] == "list" and not piece[1]:
                        continue
                    if piece[0] == "list":
                        total += build.size(piece) - (2 if merged is not None else 0)
                        build.check(total, self.name)
                        if merged is None:
                            merged = list(piece[1])
                            flat.append(["list", merged])
                        else:
                            merged.extend(piece[1])
                    else:
                        total += build.size(piece)
                        build.check(total, self.name)
                        merged = None
                        flat.append(piece)
            if not flat:
                return ["list", []]
            return flat[0] if len(flat) == 1 else ["concat", flat]
        scope = dict(env)
        element_type = node.source.type["list"]
        scope[node.var] = (["var", depth], element_type)
        if node.index is not None:
            scope[node.index] = (["idx", depth], "int")
        where = None if node.where is None else self._nf_test(node.where, scope)
        start = None if node.index is None else node.start
        form = ["each", self._nf_ref(node.source, env), start, where, node.max, self._nf(node.body, scope, depth + 1, build)]
        build.check(build.size(form), self.name)
        return form


def _nf_seq(items: list[NF], build: _NfBuild, where: str) -> NF:
    # nested seqs flatten and adjacent literals merge; the merged literal is joined once from the
    # fragments this call collected, so a long run of literals costs linear time
    flat: list[NF] = []
    fragments: list[str] = []
    total = 1

    def flush() -> None:
        if fragments:
            flat.append(["lit", "".join(fragments)])
            fragments.clear()

    for item in items:
        for piece in item[1] if item[0] == "seq" else [item]:
            if piece[0] == "lit":
                if piece[1]:
                    if not fragments:
                        total += 1
                    fragments.append(piece[1])
                continue
            flush()
            total += build.size(piece)
            build.check(total, where)
            flat.append(piece)
    flush()
    if not flat:
        return ["lit", ""]
    return flat[0] if len(flat) == 1 else ["seq", flat]


# ---------------------------------------------------------------------------------------
# Compiler


class _Emitter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.counter = 0

    def line(self, indent: int, text: str) -> None:
        self.lines.append("    " * indent + text)

    def fresh(self) -> str:
        name = f"v{self.counter}"
        self.counter += 1
        return name

    @staticmethod
    def ref(param: Param, loop_vars: set[str]) -> str:
        head = param.segments[0]
        base = head if head in loop_vars else f'params[{head!r}]'
        return f"{base}[{param.segments[1]!r}]" if len(param.segments) == 2 else base

    def test(self, test: Test, loop_vars: set[str]) -> str:
        if isinstance(test, Truthy):
            return self.ref(test.operand, loop_vars)
        if isinstance(test, Not):
            return f"not ({self.test(test.test, loop_vars)})"
        if isinstance(test, Eq):
            return f"{self.ref(test.operand, loop_vars)} == {test.scalar!r}"
        left = self.ref(test.operand, loop_vars)
        if test.length:
            left = f"len({left})"
        return f"{left} {_CMP_PY[test.op]} {test.value!r}"

    def expr(self, node: Node, indent: int, loop_vars: set[str]) -> str:
        if isinstance(node, Lit):
            return repr(node.text)
        if isinstance(node, Param):
            return f"str({self.ref(node, loop_vars)})" if node.text else self.ref(node, loop_vars)
        if isinstance(node, Part):
            args = ", ".join(f"{arg_name}={self.expr(arg, indent, loop_vars)}" for arg_name, arg in node.args)
            return f'parts[{node.slot!r}]({args})'
        if isinstance(node, Seq):
            if not node.items:
                return "''"
            if len(node.items) == 1:
                return self.expr(node.items[0], indent, loop_vars)
            return '"".join([' + ", ".join(self.expr(item, indent, loop_vars) for item in node.items) + "])"
        if isinstance(node, If):
            return f"({self.expr(node.then, indent, loop_vars)} if {self.test(node.test, loop_vars)} else {self.expr(node.else_, indent, loop_vars)})"
        return f"{node.sep!r}.join({self.list(node.items, indent, loop_vars)})"

    def list(self, node: ListNode, indent: int, loop_vars: set[str]) -> str:
        if isinstance(node, ListLit):
            return "[" + ", ".join(self.expr(item, indent, loop_vars) for item in node.items) + "]"
        if isinstance(node, Concat):
            if not node.items:
                return "[]"
            return "(" + " + ".join(self.list(item, indent, loop_vars) for item in node.items) + ")"
        name = self.fresh()
        self.line(indent, f"{name} = []")
        source = f"{self.ref(node.source, loop_vars)}[:{node.max}]"
        inner_vars = set(loop_vars) | {node.var}
        if node.index is not None:
            inner_vars.add(node.index)
            self.line(indent, f"for {node.index}, {node.var} in enumerate({source}, {node.start}):")
        else:
            self.line(indent, f"for {node.var} in {source}:")
        if node.where is not None:
            self.line(indent + 1, f"if not ({self.test(node.where, inner_vars)}):")
            self.line(indent + 2, "continue")
        body = node.body
        if isinstance(body, If):
            self.line(indent + 1, f"if {self.test(body.test, inner_vars)}:")
            then = self.expr(body.then, indent + 2, inner_vars)
            self.line(indent + 2, f"{name}.append({then})")
            self.line(indent + 1, "else:")
            else_ = self.expr(body.else_, indent + 2, inner_vars)
            self.line(indent + 2, f"{name}.append({else_})")
        else:
            value = self.expr(body, indent + 1, inner_vars)
            self.line(indent + 1, f"{name}.append({value})")
        return name


def compile_spec(name: str, spec: Any, parts: Mapping[str, str], params: Mapping[str, Any], *, derived: bool = False, origin: str = f"meta.spec ({SCHEMA_SPEC})") -> str:
    """Generate the fixed-shape ``compose(parts, params)`` source HeavenBase stores for a spec or doc row.

    The output is a pure function of its inputs: loops become ``for`` statements over
    ``params[...][:max]`` filling accumulators ``v0, v1, ...``, everything else becomes expressions
    over ``parts[slot](...)`` calls, ``str(...)`` of a derived spec's int and bool text, ``"".join``
    and conditional expressions. Only the grammar is checked here; the runtime compiles after full
    typing and compares the stored source byte-for-byte.

    Args:
        name (str): Row name, quoted into the docstring.
        spec (Any): The JSON value of ``meta.spec``, or the spec :func:`doc.parse_document` derived from ``meta.text``.
        parts (Mapping[str, str]): ``meta.parts`` (slot to row name); every ``part`` node must use one of its slots.
        params (Mapping[str, Any]): ``meta.params`` (parameter types).
        derived (bool): Parse ``spec`` as a derived spec (see :func:`parse_spec`).
        origin (str): What the docstring says the source was compiled from; spec rows keep the default, so their
            stored sources never move.

    Returns:
        str: Python source of one function ``compose(parts, params)`` ending with a newline.

    Raises:
        SpecError: If the spec fails grammar, bound or parameter-type validation.
    """
    parsed = parse_spec(spec, params=params, part_names=parts, derived=derived)
    emitter = _Emitter()
    emitter.line(0, "def compose(parts, params):")
    emitter.line(1, f'"""{name} · compiled from {origin} · do not edit"""')
    result = emitter.expr(parsed.root, 1, set())
    emitter.line(1, f"return {result}")
    return "\n".join(emitter.lines) + "\n"
