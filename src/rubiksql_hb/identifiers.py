"""Structured source identifiers with reversible stable IDs and SQL rendering."""

from __future__ import annotations

__all__ = [
    "ColumnIdentifier",
    "IdentifierDialect",
    "RelationIdentifier",
    "normalize_identifier",
    "quote_identifier",
]

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

IdentifierDialect = Literal["oracle", "sqlite", "trino"]

_DIALECTS = frozenset({"oracle", "sqlite", "trino"})
_RELATION_FIELDS = frozenset({"catalog", "schema", "relation"})


def normalize_identifier(value: object, *, field: str) -> str:
    """Trim and validate one source identifier component.

    Args:
        value (object): Provider identifier component to validate.
        field (str): Caller-facing field name used in validation errors.

    Returns:
        str: Identifier with surrounding whitespace trimmed and all remaining
            spelling, case, punctuation, and embedded dots preserved.

    Raises:
        TypeError: If `value` is not a string.
        ValueError: If the trimmed component is empty or contains NUL, CR, or LF.
    """
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} cannot be empty")
    if any(char in normalized for char in ("\x00", "\r", "\n")):
        raise ValueError(f"{field} cannot contain control characters")
    return normalized


def quote_identifier(value: object, *, dialect: IdentifierDialect | str) -> str:
    """Quote one identifier component for a supported source dialect.

    Oracle, SQLite, and Trino all use ANSI double quotes for delimited
    identifiers. The dialect remains explicit so callers cannot accidentally
    quote an already joined path or silently render an unsupported provider.

    Args:
        value (object): Single unqualified provider identifier component.
        dialect (IdentifierDialect | str): SQL dialect. Supported values are
            `oracle`, `sqlite`, and `trino`.

    Returns:
        str: Always-delimited SQL identifier with embedded quotes doubled.

    Raises:
        TypeError: If `value` is not a string.
        ValueError: If `value` is invalid or `dialect` is unsupported.
    """
    component = normalize_identifier(value, field="identifier")
    normalized_dialect = normalize_identifier(dialect, field="dialect").casefold()
    if normalized_dialect not in _DIALECTS:
        choices = ", ".join(sorted(_DIALECTS))
        raise ValueError(f"unsupported identifier dialect {normalized_dialect!r}; expected one of: {choices}")
    return '"' + component.replace('"', '""') + '"'


def _encode_component(value: str) -> str:
    if "." not in value and '"' not in value:
        return value
    return '"' + value.replace('"', '""') + '"'


def _encode_path(parts: tuple[str, ...]) -> str:
    return ".".join(_encode_component(part) for part in parts)


def _decode_path(value: object, *, field: str) -> tuple[str, ...]:
    text = normalize_identifier(value, field=field)
    parts: list[str] = []
    index = 0
    while index < len(text):
        if text[index] == '"':
            index += 1
            component: list[str] = []
            while index < len(text):
                if text[index] != '"':
                    component.append(text[index])
                    index += 1
                    continue
                if index + 1 < len(text) and text[index + 1] == '"':
                    component.append('"')
                    index += 2
                    continue
                index += 1
                break
            else:
                raise ValueError(f"{field} contains an unterminated quoted component")
            if index < len(text) and text[index] != ".":
                raise ValueError(f"{field} must separate quoted identifier components with dots")
            raw = "".join(component)
        else:
            end = text.find(".", index)
            end = len(text) if end < 0 else end
            raw = text[index:end]
            if '"' in raw:
                raise ValueError(f"{field} must quote the complete component containing a double quote")
            index = end
        parts.append(normalize_identifier(raw, field=f"{field} component"))
        if index == len(text):
            break
        index += 1
        if index == len(text):
            raise ValueError(f"{field} cannot end with an empty component")
    return tuple(parts)


@dataclass(frozen=True, slots=True)
class RelationIdentifier:
    """Identify one source relation through separately owned components.

    `stable_id` deliberately omits `catalog`: a `SourceSpec` fixes one catalog,
    and profile rows are keyed by `source_id + stable_id`. `route_id` includes
    the catalog for source routing and diagnostics. This keeps existing
    `schema.relation` IDs unchanged while preventing a dotted component from
    being confused with a qualification boundary.

    Args:
        relation (str): Provider relation component.
        schema (str | None): Provider schema component.
        catalog (str | None): Provider catalog component fixed by the source.
    """

    relation: str
    schema: str | None = None
    catalog: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relation",
            normalize_identifier(self.relation, field="relation"),
        )
        for field in ("schema", "catalog"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(
                    self,
                    field,
                    normalize_identifier(value, field=field),
                )
        if self.catalog is not None and self.schema is None:
            raise ValueError("catalog-qualified relations require a schema")

    @property
    def stable_id(self) -> str:
        """Return the canonical source-scoped relation identifier."""
        parts = (self.relation,) if self.schema is None else (self.schema, self.relation)
        return _encode_path(parts)

    @property
    def route_id(self) -> str:
        """Return the canonical fully routed relation identifier."""
        return _encode_path(tuple(part for part in (self.catalog, self.schema, self.relation) if part is not None))

    @classmethod
    def from_stable_id(
        cls,
        value: object,
        *,
        catalog: str | None = None,
    ) -> "RelationIdentifier":
        """Decode a legacy or quoted-component stable relation ID.

        Args:
            value (object): One-part relation or `schema.relation` stable ID.
                Components containing dots use SQL-style double-quote escaping,
                for example `"sales.eu".orders`.
            catalog (str | None): Source-fixed catalog to retain for SQL routing.

        Returns:
            RelationIdentifier: Structured, normalized relation identity.

        Raises:
            TypeError: If `value` is not a string.
            ValueError: If the stable ID is malformed or has more than two
                source-scoped components.
        """
        parts = _decode_path(value, field="relation stable_id")
        if len(parts) == 1:
            return cls(parts[0], catalog=catalog)
        if len(parts) == 2:
            return cls(parts[1], schema=parts[0], catalog=catalog)
        raise ValueError("relation stable_id must contain relation or schema.relation; " "quote components that contain dots")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RelationIdentifier":
        """Build a relation identifier from its registry-v2 object.

        Args:
            data (Mapping[str, object]): Mapping with `relation` and optional
                `schema` and `catalog` components.

        Returns:
            RelationIdentifier: Structured, normalized relation identity.

        Raises:
            TypeError: If the value is not a string-keyed mapping.
            ValueError: If fields are unknown, missing, or invalid.
        """
        if not isinstance(data, Mapping):
            raise TypeError(f"relation identifier must be a mapping, got {type(data).__name__}")
        if not all(isinstance(key, str) for key in data):
            raise TypeError("relation identifier keys must be strings")
        unknown = set(data) - _RELATION_FIELDS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"relation identifier contains unknown fields: {names}")
        if "relation" not in data:
            raise ValueError("relation identifier requires relation")
        return cls(
            relation=data["relation"],  # type: ignore[arg-type]
            schema=data.get("schema"),  # type: ignore[arg-type]
            catalog=data.get("catalog"),  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, str]:
        """Return the canonical registry-v2 representation.

        Args:
            None.

        Returns:
            dict[str, str]: JSON-compatible structured relation components.
        """
        data = {"relation": self.relation}
        if self.schema is not None:
            data["schema"] = self.schema
        if self.catalog is not None:
            data["catalog"] = self.catalog
        return data

    def to_sql(self, dialect: IdentifierDialect | str) -> str:
        """Render a fully qualified relation by quoting every component.

        Args:
            dialect (IdentifierDialect | str): SQL dialect. Oracle rejects a
                catalog component; SQLite and Trino render every present part.

        Returns:
            str: Qualified SQL relation with each component independently quoted.

        Raises:
            ValueError: If the dialect is unsupported or Oracle receives a catalog.
        """
        normalized_dialect = normalize_identifier(dialect, field="dialect").casefold()
        if normalized_dialect == "oracle" and self.catalog is not None:
            raise ValueError("Oracle relation identifiers cannot declare a catalog")
        parts = tuple(part for part in (self.catalog, self.schema, self.relation) if part is not None)
        return ".".join(quote_identifier(part, dialect=normalized_dialect) for part in parts)

    def column(self, name: str) -> "ColumnIdentifier":
        """Create a column identity owned by this relation.

        Args:
            name (str): Provider column component.

        Returns:
            ColumnIdentifier: Structured source column identity.
        """
        return ColumnIdentifier(relation=self, column=name)


@dataclass(frozen=True, slots=True)
class ColumnIdentifier:
    """Identify one source column through a structured relation and name.

    Args:
        relation (RelationIdentifier): Relation that owns the column.
        column (str): Provider column component.
    """

    relation: RelationIdentifier
    column: str

    def __post_init__(self) -> None:
        if not isinstance(self.relation, RelationIdentifier):
            raise TypeError("column relation must be a RelationIdentifier")
        object.__setattr__(
            self,
            "column",
            normalize_identifier(self.column, field="column"),
        )

    @property
    def stable_id(self) -> str:
        """Return the canonical source-scoped column identifier."""
        parts = (self.relation.relation, self.column) if self.relation.schema is None else (self.relation.schema, self.relation.relation, self.column)
        return _encode_path(parts)

    @property
    def route_id(self) -> str:
        """Return the canonical fully routed column identifier."""
        return _encode_path(
            tuple(
                part
                for part in (
                    self.relation.catalog,
                    self.relation.schema,
                    self.relation.relation,
                    self.column,
                )
                if part is not None
            )
        )

    @classmethod
    def from_stable_id(
        cls,
        value: object,
        *,
        catalog: str | None = None,
    ) -> "ColumnIdentifier":
        """Decode a legacy or quoted-component stable column ID.

        Args:
            value (object): `relation.column` or `schema.relation.column` stable
                ID. Components that contain dots or double quotes must use
                canonical double quoting.
            catalog (str | None): Source-fixed catalog retained for SQL routing.

        Returns:
            ColumnIdentifier: Structured, normalized column identity.

        Raises:
            TypeError: If `value` is not a string.
            ValueError: If the stable ID does not contain exactly two or three
                source-scoped components, or a catalog is paired with an
                unqualified relation.
        """
        parts = _decode_path(value, field="column stable_id")
        if len(parts) == 2:
            if catalog is not None:
                raise ValueError("catalog-qualified column stable_id requires " "schema.relation.column")
            relation = RelationIdentifier(parts[0])
            return cls(relation=relation, column=parts[1])
        if len(parts) == 3:
            relation = RelationIdentifier(parts[1], schema=parts[0], catalog=catalog)
            return cls(relation=relation, column=parts[2])
        raise ValueError("column stable_id must contain relation.column or " "schema.relation.column; quote components that contain dots")

    def to_sql(
        self,
        dialect: IdentifierDialect | str,
        *,
        qualified: bool = True,
    ) -> str:
        """Render this column by quoting every selected component.

        Args:
            dialect (IdentifierDialect | str): SQL dialect used for rendering.
            qualified (bool): Whether to include the routed relation. When false,
                render only the column component for an already scoped query.

        Returns:
            str: Quoted column SQL.
        """
        column = quote_identifier(self.column, dialect=dialect)
        if not qualified:
            return column
        return f"{self.relation.to_sql(dialect)}.{column}"
