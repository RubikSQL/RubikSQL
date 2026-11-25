"""Bind HB's query facade to existing SQLite storage without publishing metadata."""
from __future__ import annotations

from collections.abc import Sequence
import re
import threading
from typing import Any

from heavenbase.backends.sql.sqlite import SQLiteBackend
from heavenbase.backends.sql.strategies.sparse_gram import SQLSparseGramAdapter
from heavenbase.database import Database


_DDL_TOKENS = re.compile(r'''"(?:[^"]|"")*"|'(?:[^']|'')*'|`(?:[^`]|``)*`|\[[^\]]*\]|--[^\n]*|/\*[\s\S]*?\*/|\w+|[^\s]''')


def _identifier(token: str) -> str:
    if token.startswith('"'):
        return token[1:-1].replace('""', '"')
    if token.startswith("`"):
        return token[1:-1].replace("``", "`")
    return token[1:-1] if token.startswith("[") else token


def _column_collations(ddl: str) -> dict[str, str | None]:
    """Read column-level COLLATE declarations; skip nested checks and literals.

    SQLAlchemy's SQLite reflection drops collation. The native publisher's DDL
    remains authoritative for that part of each column contract.
    """
    result: dict[str, str | None] = {}
    depth, part = 0, []

    def finish() -> None:
        if part:
            collations = [part[index + 1] for index in range(1, len(part) - 1)
                          if part[index].upper() == "COLLATE"]
            if len(collations) > 1:
                raise ValueError("column declares more than one collation")
            result[_identifier(part[0])] = _identifier(collations[0]).upper() if collations else None

    for token in _DDL_TOKENS.findall(ddl):
        if token.startswith(("--", "/*")):
            continue
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth == 0:
                finish()
                break
        elif token == "," and depth == 1:
            finish()
            part = []
        elif depth == 1:
            part.append(token)
    return result


def _physical_type(column: Any, dialect: Any) -> tuple[str, str | None]:
    """Separate a compiled physical type from its explicit SQLite collation."""
    compiled = str(column.type.compile(dialect=dialect)).upper()
    match = re.fullmatch(r'(.*?)\s+COLLATE\s+("(?:[^"]|"")*"|[A-Z_][A-Z0-9_]*)', compiled)
    return (match[1], _identifier(match[2])) if match else (compiled, None)


class _ReadOnlyDatabase(Database):
    """Preserve physical read mode and destructive-operation guards when cloned."""

    _connection_lock = threading.Lock()

    @property
    def engine(self) -> Any:
        from sqlalchemy import event
        from .control import PromptControlError

        engine = super().engine
        uri = self.spec.database + "?mode=ro"
        # The pinned HB engine profile strips file:/// to an ordinary path. Its
        # pool still owns connections; this hook supplies sqlite3's actual URI.
        with self._connection_lock:
            installed = getattr(engine, "_rubiksql_readonly_uri", None)
            if installed is None:
                def connect(dialect: Any, record: Any, args: Any, kwargs: Any) -> None:
                    args[0] = uri
                    kwargs["uri"] = True

                event.listen(engine, "do_connect", connect)
                engine._rubiksql_readonly_uri = uri
            elif installed != uri:
                raise PromptControlError("read database connection differs from its bound file URI")
        return engine

    def clone(self) -> _ReadOnlyDatabase:
        """Return an independent guarded facade over this same captured spec."""
        return type(self)(autocreate=False, spec=self.spec,
                          resolver=self._module_resolver, config=self._config)

    def drop(self) -> None:
        from .control import PromptControlError

        raise PromptControlError("the bound read database refuses storage deletion")

    def drop_database(self, *, force: bool = False) -> None:
        from .control import PromptControlError

        raise PromptControlError("the bound read database refuses storage deletion")


class _ReadOnlySparseGramAdapter(SQLSparseGramAdapter):
    """Keep native field/config binding while verifying existing index storage.

    The pinned adapter's ensure hooks otherwise create tables, write fingerprints
    and rebuild postings. These hooks only inspect the owner's declared layout.
    """

    def _ensure_index(self) -> None:
        index = self._need_index()
        if self._ensured_index is index:
            return
        for table in (index.meta, index.keywords, index.postings):
            self.backend._validate_read_table(table, indexes=True)
        self._ensured_index = index

    def _ensure_fingerprint(self, entity_id: str, field_name: str, config: Any) -> None:
        from .control import PromptControlError

        key = f"fingerprint:{entity_id}:{field_name}"
        if key in self._fingerprint_checked:
            return
        if self._need_index().get_meta(key) != config.fingerprint():
            raise PromptControlError(f"read workspace sparse index fingerprint differs for {entity_id}.{field_name}")
        self._fingerprint_checked.add(key)


class ReadOnlySQLiteBackend(SQLiteBackend):
    """Reflect established tables; bootstrap accepts only transient schema metadata.

    The pinned HB facade publishes foundation/metaschema while binding schemas.
    Those schema-only mutations are discarded during construction. Canonical DML
    is always refused, and after binding every mutation is refused. The connection
    itself is also mode=ro/query_only, including direct database operations.
    """

    def __init__(self, name: str, database: str, *, ws_id: str, resolver: Any, config: Any) -> None:
        self._binding = True
        super().__init__(name, database, ws_id=ws_id, resolver=resolver, config=config,
                         params={"mode": "ro"}, autocreate=False,
                         pragmas=["PRAGMA query_only=ON", "PRAGMA busy_timeout=60000"])

    def finish_binding(self) -> None:
        """End the schema-only construction scope before exposing the facade."""
        self._binding = False

    def _connect_database(self) -> None:
        super()._connect_database()
        database = self.db
        # Resolving the ordinary facade captures HB's canonical DBSpec without
        # opening a connection. Keep that spec and its Context authorities.
        self.db = _ReadOnlyDatabase(autocreate=False, spec=database.spec,
                                   resolver=database._module_resolver, config=database._config)

    def drop(self) -> None:
        from .control import PromptControlError

        raise PromptControlError("the bound read workspace refuses storage deletion")

    def _install_strategy_adapters(self) -> None:
        self._strategy_adapters.clear()
        _ReadOnlySparseGramAdapter(self).register()

    def upsert(self, rows: Sequence[Any]) -> None:
        from .control import PromptControlError

        if self._binding and all(row.entity_id in {"sys-catalog", "sys-metaschema"} for row in rows):
            return
        raise PromptControlError("the bound read workspace refuses every row mutation")

    def _validate_read_table(self, declared: Any, *, indexes: bool = False) -> None:
        """Validate physical storage while retaining the owner's column codecs."""
        from sqlalchemy import MetaData, Table, inspect
        from .control import PromptControlError

        inspector = inspect(self.db.engine)
        name = declared.name
        if not inspector.has_table(name):
            raise PromptControlError(f"read workspace is missing table {name}")
        table = Table(name, MetaData(), autoload_with=self.db.engine)
        actual, required = set(table.c.keys()), set(declared.c.keys())
        if required != actual:
            raise PromptControlError(f"read workspace table {name} has incompatible columns: "
                                     f"missing={sorted(required - actual)}, extra={sorted(actual - required)}")
        ddl = self.db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=:name",
                              {"name": name}, readonly=True).scalar()
        try:
            collations = _column_collations(ddl)
        except (TypeError, ValueError) as error:
            raise PromptControlError(f"read workspace table {name} has incompatible collation declarations") from error
        for column in declared.c:
            held = table.c[column.name]
            expected_type, expected_collation = _physical_type(column, self.db.engine.dialect)
            if (_physical_type(held, self.db.engine.dialect)[0] != expected_type
                    or collations.get(column.name) != expected_collation
                    or bool(held.primary_key) != bool(column.primary_key)
                    or bool(held.nullable) != bool(column.nullable)):
                raise PromptControlError(f"read workspace column {name}.{column.name} is incompatible")
        if indexes:
            held_indexes = {item["name"]: item for item in inspector.get_indexes(name)}
            for index in declared.indexes:
                held = held_indexes.get(index.name)
                if (held is None or bool(held.get("unique")) != bool(index.unique)
                        or held.get("column_names") != [column.name for column in index.columns]):
                    raise PromptControlError(f"read workspace index {index.name} is missing or incompatible")

    def ensure_plan(self, schema: Any, storage_plan: Any) -> None:
        """Bind existing tables and native sparse indexes without storage mutations."""
        from sqlalchemy import Column, Integer, Table
        from heavenbase.backends.sql.base import _side_present_column
        from .control import PromptControlError

        main = self._main_table_fields(storage_plan.bindings)
        side = self._side_table_fields(storage_plan.bindings)
        fields = {binding.field_name for binding in main}
        expected = [self._column("object_id", schema.fields["object_id"].dtype, primary_key=True, nullable=False)]
        expected.extend(self._column(field.name, field.dtype, nullable=not field.required)
                        for field in schema.fields.values() if field.name != "object_id" and field.name in fields)
        expected.extend(Column(_side_present_column(binding.field_name), Integer(), nullable=True) for binding in side)
        name = self._table_name(schema.entity_id)
        declared = self.metadata.tables.get(name)
        if declared is None:
            declared = Table(name, self.metadata, *expected)
        self._validate_read_table(declared)
        self.tables[schema.entity_id] = declared
        for binding in side:
            field = schema.get(binding.field_name)
            spec = self._side_spec(schema.entity_id, field)
            self._validate_read_table(spec.table)
            self._side_specs.setdefault(schema.entity_id, {})[field.name] = spec
        for binding in self._my_bindings(storage_plan.bindings):
            if self.supports_strategy(schema, binding):
                adapter = self.strategy_adapter(binding.strategy)
                if adapter.identifier != SQLSparseGramAdapter.identifier:
                    raise PromptControlError(f"read workspace cannot bind unsupported indexed strategy for {schema.entity_id}")
                adapter.ensure(schema, binding)
