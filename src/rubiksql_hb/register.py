"""Register external database schema metadata in the RubikSQL knowledge base."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib

import heavenbase as hb
import sqlalchemy as sa
from sqlalchemy import inspect as sa_inspect

from .embedding import embed, embed_many
from .identifiers import RelationIdentifier
from .schema import ColumnEntity, EnumValueEntity, TableEntity


# The BIRD line's id rule, kept verbatim. The enterprise line hashes any part containing
# `__` or a leading/trailing `_` so distinct provider names can never share an id; adopting
# that here would rewrite 480 enum-value and 11 column object_ids across the 11 dev
# databases and force every workspace rebuild, so the readable rule stays.
def _safe_id(*parts: str, max_len: int = 60) -> str:
    """Readable ``object_id`` for a schema row, e.g. ``financial__loan__amount``.

    HeavenBase's default id is seeded from the row's ``name`` alone, which is not unique for
    schema metadata — a column named ``amount`` exists in many tables of many databases — so
    these rows carry an explicit object_id built from the full qualifying path. Overlong
    paths collapse to their md5 so the id stays a legal identifier.
    """
    raw = "__".join(parts)
    if len(raw) <= max_len:
        return raw
    return hashlib.md5(raw.encode()).hexdigest()


def _schema_names(schemas: Iterable[str]) -> tuple[str | None, ...]:
    """Normalize an optional schema allowlist without ever enumerating a whole catalog."""
    normalized = tuple(
        dict.fromkeys(str(name).strip() for name in schemas if str(name).strip())
    )
    return normalized or (None,)


def _qualified_table(schema_name: str | None, table_name: str) -> str:
    """Return the SQL name stored in knowledge metadata."""
    return f"{schema_name}.{table_name}" if schema_name else table_name


def _relation_identifier(value: object) -> RelationIdentifier:
    """Normalize one public relation input without splitting provider names."""
    if isinstance(value, RelationIdentifier):
        return value
    if isinstance(value, Mapping):
        return RelationIdentifier.from_dict(value)
    return RelationIdentifier.from_stable_id(value)


def _relation_names(
    inspector: sa.Inspector,
    schema_name: str | None,
    *,
    include_views: bool,
) -> tuple[str, ...]:
    """Return tables, and views only on request, while preserving quote metadata.

    ``quoted_name("orders", True)`` and plain ``"orders"`` compare equal as
    strings, but Oracle can expose them as distinct physical objects (quoted
    lowercase versus conventional unquoted uppercase). Deduplicating through a
    plain string set would silently drop one of them.

    Views are reflected only when ``include_views`` is set. An unfiltered local
    URL registers tables alone: ``scripts/setup_dbs.py`` gates a workspace on the
    ``sqlite_master`` rows of ``type='table'``, so a registered view would leave
    every rebuild "incomplete", and a view over a dropped table fails key
    reflection with an OperationalError instead of the tolerated NoSuchTableError.
    """
    tables = inspector.get_table_names(schema=schema_name)
    if not include_views:
        return tuple(tables)
    try:
        views = inspector.get_view_names(schema=schema_name)
    except NotImplementedError:
        views = []
    return (*tables, *views)


def _reflection_name(dialect: object, physical_name: str) -> str:
    """Convert one physical name to the dialect's Inspector representation."""
    dialect_name = getattr(dialect, "name", "")
    if dialect_name == "trino":
        # Trino identifiers are case-insensitive even when delimited, while its
        # SQLAlchemy Inspector filters information_schema with a case-sensitive
        # string parameter. Query through one component-level semantic spelling.
        return physical_name.casefold()
    if dialect_name != "oracle":
        return physical_name
    return dialect.normalize_name(physical_name)  # type: ignore[attr-defined, no-any-return]


def _reflection_key(dialect: object, value: object) -> tuple[str, bool]:
    """Return one dialect-semantic Inspector lookup key."""
    if getattr(dialect, "name", "") == "trino":
        # Quoting never makes a Trino identifier case-sensitive.
        return str(value).casefold(), False
    return str(value), getattr(value, "quote", None) is True


def _physical_name(dialect: object, reflected_name: object) -> str:
    """Recover one physical provider name from reflected dialect metadata."""
    text = str(reflected_name)
    if getattr(dialect, "name", "") != "oracle":
        return text
    if getattr(reflected_name, "quote", None) is True:
        return text
    return str(  # type: ignore[attr-defined]
        dialect.denormalize_name(reflected_name)
    )


def _constraint_metadata(
    inspector: sa.Inspector,
    method: str,
    table_name: str,
    schema_name: str | None,
) -> dict | list:
    """Read optional key metadata, tolerating dialects that omit it for views."""
    try:
        return getattr(inspector, method)(table_name, schema=schema_name)
    except (NotImplementedError, sa.exc.NoSuchTableError):
        return {} if method == "get_pk_constraint" else []


def register_db(
    db_id: str,
    url: str | hb.Database,
    workspace: hb.HeavenBase,
    desc: str = "",
    *,
    schemas: Iterable[str] = (),
    relations: Iterable[str | RelationIdentifier | Mapping[str, object]] = (),
    include_statistics: bool | None = None,
    workers: int = 1,
) -> dict[str, int]:
    """Reflect a source and write table/column metadata into a HeavenBase workspace.

    Args:
        db_id: Logical source id stored on every knowledge row.
        url: A legacy SQLAlchemy URL or a runtime-only HeavenBase database facade.
        workspace: Knowledge workspace receiving the reflected rows.
        desc: Reserved database-level description retained for API compatibility.
        schemas: Optional schema allowlist. When present, stored table names are qualified.
        relations: Relation allowlist. External sources require non-empty,
            structured or canonical schema-qualified names; the legacy local path
            also accepts bare names. Views are registered only through this
            allowlist; an unfiltered local URL reflects tables alone, as the BIRD
            line did.
        include_statistics: Run exact row counts and bounded distinct-value probes. None
            preserves the historical URL behavior and selects metadata-only reflection for a
            HeavenBase Database target.
        workers: Embedding calls in flight at once for the column and enum-value vectors (the
            vectors are the same whatever the width).

    Returns:
        Counts of table, column, and enum-value rows written.

    Notes:
        Oracle and Hive registration should pass a HeavenBase Database and leave
        include_statistics false. That keeps onboarding proportional to schema size rather
        than business-data size. A later, explicitly budgeted profiling job may enrich the
        metadata without changing the live source's read-only boundary.
    """
    del desc  # there is no database-level registration field yet
    external = isinstance(url, hb.Database)
    schema_names = _schema_names(schemas)
    relation_inputs = tuple(relations)
    allowed_relations: set[str]
    relation_refs: tuple[RelationIdentifier, ...] = ()
    if external:
        relation_refs = tuple(
            _relation_identifier(value) for value in relation_inputs
        )
        if not relation_refs:
            raise ValueError(
                "external source registration requires a non-empty relation allowlist"
            )
        if any(relation.schema is None for relation in relation_refs):
            raise ValueError(
                "external source relations require separate schema and relation components"
            )
        allowed_relations = {relation.stable_id for relation in relation_refs}
        if len(allowed_relations) != len(relation_refs):
            raise ValueError(
                "external source relation allowlist cannot contain duplicates"
            )
    else:
        allowed_relations = {
            str(name).strip()
            for name in relation_inputs
            if str(name).strip()
        }

    collect_stats = (not external) if include_statistics is None else bool(include_statistics)
    engine = url.engine if external else sa.create_engine(url)
    inspector = sa_inspect(engine)
    dialect = getattr(inspector, "dialect", engine.dialect)
    if external:
        configured_schema_identities = tuple(
            _reflection_key(dialect, _reflection_name(dialect, name))
            for name in schema_names
            if name is not None
        )
        if len(configured_schema_identities) != len(
            set(configured_schema_identities)
        ):
            raise ValueError(
                "external schema allowlist is ambiguous after dialect normalization"
            )
        configured_schema_keys = set(configured_schema_identities)
        relation_schema_keys = {
            _reflection_key(
                dialect,
                _reflection_name(dialect, relation.schema),
            )
            for relation in relation_refs
            if relation.schema is not None
        }
        if configured_schema_keys and not relation_schema_keys <= configured_schema_keys:
            raise ValueError(
                "external source relation schemas must be included in the schema allowlist"
            )
    allowed_by_identity: dict[
        tuple[tuple[str, bool], tuple[str, bool]],
        RelationIdentifier,
    ] = {}
    if external:
        reflection_schemas: dict[tuple[str, bool], str] = {}
        for relation in relation_refs:
            if relation.schema is None:  # validated above; keeps narrowing explicit
                raise RuntimeError("external relation lost its configured schema")
            reflected_schema = _reflection_name(dialect, relation.schema)
            reflected_relation = _reflection_name(dialect, relation.relation)
            schema_key = _reflection_key(dialect, reflected_schema)
            identity = (
                schema_key,
                _reflection_key(dialect, reflected_relation),
            )
            previous = allowed_by_identity.get(identity)
            if previous is not None and previous != relation:
                raise ValueError(
                    "external relation allowlist is ambiguous after dialect normalization"
                )
            allowed_by_identity[identity] = relation
            reflection_schemas.setdefault(schema_key, reflected_schema)
        # Reflect only schemas that own an allowlisted relation. Oracle receives
        # normalized plain lowercase for conventional unquoted uppercase names,
        # while quoted/mixed names retain the dialect's quote metadata.
        schema_names = tuple(reflection_schemas.values())
    matched_relations: set[str] = set()

    table_rows: list[TableEntity] = []
    column_rows: list[ColumnEntity] = []
    enum_rows: list[EnumValueEntity] = []

    try:
        for schema_name in schema_names:
            # External sources always carry an allowlist, so this reads "views only when
            # something names them"; the unfiltered local sqlite path takes tables alone.
            relation_names = _relation_names(
                inspector,
                schema_name,
                include_views=bool(allowed_relations),
            )
            for table_name in relation_names:
                reflected_table_name = table_name
                table_name = str(reflected_table_name)
                qualified_table = _qualified_table(schema_name, table_name)
                requested: RelationIdentifier | None = None
                if external:
                    if schema_name is None:
                        raise RuntimeError(
                            "external source reflection lost its configured schema"
                        )
                    requested = allowed_by_identity.get(
                        (
                            _reflection_key(dialect, schema_name),
                            _reflection_key(dialect, reflected_table_name),
                        )
                    )
                    if requested is None:
                        continue
                    if requested.stable_id in matched_relations:
                        raise ValueError(
                            "source reflection returned the same allowlisted relation more than once"
                        )
                    qualified_table = requested.stable_id
                    matched_relations.add(requested.stable_id)
                else:
                    matched = {table_name, qualified_table} & allowed_relations
                    if allowed_relations and not matched:
                        continue
                    matched_relations.update(matched)
                pk_info = _constraint_metadata(
                    inspector,
                    "get_pk_constraint",
                    reflected_table_name,
                    schema_name,
                )
                fk_infos = _constraint_metadata(
                    inspector,
                    "get_foreign_keys",
                    reflected_table_name,
                    schema_name,
                )
                col_infos = inspector.get_columns(
                    reflected_table_name,
                    schema=schema_name,
                )
                raw_pks = list(pk_info.get("constrained_columns") or [])
                pks = (
                    [_physical_name(dialect, name) for name in raw_pks]
                    if external
                    else raw_pks
                )

                fk_map: dict[str, str] = {}
                for fk in fk_infos:
                    ref_schema = fk.get("referred_schema")
                    for local_col, ref_col in zip(
                        fk["constrained_columns"],
                        fk["referred_columns"],
                    ):
                        if external:
                            if requested is None or requested.schema is None:
                                raise RuntimeError(
                                    "external reflection lost its structured relation"
                                )
                            local_name = _physical_name(dialect, local_col)
                            referred_schema = (
                                requested.schema
                                if ref_schema is None
                                else _physical_name(dialect, ref_schema)
                            )
                            referred = RelationIdentifier(
                                relation=_physical_name(
                                    dialect,
                                    fk["referred_table"],
                                ),
                                schema=referred_schema,
                            )
                            fk_map[local_name] = referred.column(
                                _physical_name(dialect, ref_col)
                            ).stable_id
                        else:
                            ref_table = str(fk["referred_table"])
                            qualified_ref = _qualified_table(ref_schema, ref_table)
                            fk_map[str(local_col)] = f"{qualified_ref}.{ref_col}"

                reflected_table = None
                n_rows = None
                if collect_stats:
                    # A lexical table clause, never ``sa.Table(autoload_with=engine)``: autoload
                    # re-reflects the table and resolves its foreign keys recursively, so a
                    # REFERENCES to a table that no longer exists raises NoSuchTableError and
                    # a column-less REFERENCES to a parent without a matching primary key
                    # raises ArgumentError. Neither declaration may stop registration, and the
                    # Inspector metadata above is all the statistics need.
                    reflected_table = sa.table(reflected_table_name, schema=schema_name)
                    with engine.connect() as connection:
                        n_rows = connection.execute(
                            sa.select(sa.func.count()).select_from(reflected_table)
                        ).scalar()

                table_rows.append(TableEntity.from_dict({
                    "object_id": _safe_id(db_id, qualified_table),
                    "db_id": db_id,
                    "name": qualified_table,
                    "n_rows": n_rows,
                    "pks": pks,
                }))

                for column_info in col_infos:
                    reflected_column_name = column_info["name"]
                    column_name = (
                        _physical_name(dialect, reflected_column_name)
                        if external
                        else str(reflected_column_name)
                    )
                    original_type = str(column_info["type"])
                    type_upper = original_type.upper()
                    categorical = any(
                        token in type_upper
                        for token in ("TEXT", "VARCHAR", "CHAR", "STRING")
                    )
                    column_rows.append(ColumnEntity.from_dict({
                        "object_id": _safe_id(db_id, qualified_table, column_name),
                        "db_id": db_id,
                        "table_name": qualified_table,
                        "name": column_name,
                        "datatype_orig": original_type,
                        "datatype_anno": "CATEGORICAL" if categorical else None,
                        "is_pk": column_name in pks,
                        "fk_ref": fk_map.get(column_name),
                    }))

                    if not collect_stats or not categorical or column_name in pks:
                        continue
                    if reflected_table is None:
                        raise RuntimeError("statistics registration requires a reflected table")
                    source_column = sa.column(reflected_column_name, column_info["type"])
                    with engine.connect() as connection:
                        values = connection.execute(
                            sa.select(source_column)
                            .select_from(reflected_table)
                            .where(source_column.is_not(None))
                            .distinct()
                            .limit(200)
                        ).fetchall()
                    distinct_values = [row[0] for row in values if row[0] is not None]
                    if len(distinct_values) <= 100:
                        for value in distinct_values:
                            enum_rows.append(EnumValueEntity.from_dict({
                                "object_id": _safe_id(
                                    db_id,
                                    qualified_table,
                                    column_name,
                                    value,
                                ),
                                "db_id": db_id,
                                "table_name": qualified_table,
                                "col_name": column_name,
                                "value": str(value),
                            }))

        missing_relations = allowed_relations - matched_relations
        if missing_relations:
            missing = ", ".join(sorted(missing_relations))
            raise ValueError(
                "source relation allowlist was not fully reflected; missing: " + missing
            )

        # Compute vectors before entering HeavenBase's workspace lifecycle lock. HB skips a
        # compute hook when the derived value is supplied, so one slow embedding request cannot
        # hold the shared workspace lock around every other registration.
        table_dicts = [row.to_dict() for row in table_rows]
        column_dicts = [row.to_dict() for row in column_rows]
        enum_dicts = [row.to_dict() for row in enum_rows]
        column_texts = [
            row.get("desc") or (
                f"{row['table_name']}.{row['name']} ({row.get('datatype_orig') or 'unknown'})"
                if external
                else ""
            )
            for row in column_dicts
        ]
        for row, vector in zip(column_dicts, embed_many(column_texts, workers=workers, embed_fn=embed)):
            row["desc_vec"] = vector
        enum_texts = [row.get("value") or "" for row in enum_dicts]
        for row, vector in zip(enum_dicts, embed_many(enum_texts, workers=workers, embed_fn=embed)):
            row["value_vec"] = vector

        workspace.upsert_many(TableEntity, table_dicts)
        workspace.upsert_many(ColumnEntity, column_dicts)
        workspace.upsert_many(EnumValueEntity, enum_dicts)
    finally:
        # An HB Database owns a shared engine registry entry and stays live for execution.
        if not external:
            engine.dispose()

    return {
        "tables": len(table_rows),
        "columns": len(column_rows),
        "enum_values": len(enum_rows),
    }
