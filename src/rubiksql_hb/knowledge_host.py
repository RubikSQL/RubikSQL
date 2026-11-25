"""Knowledge management tools for examples, synonyms, rules, and exact entity reads."""
import re
import sys
import traceback
import unicodedata
from collections.abc import Mapping
from typing import Any

import heavenbase as hb

from .embedding import embed, embed_rule_desc, embed_title, rule_with_example
from .knowledge_failures import RULES_RETRIEVAL_FAILED_ERROR
from .query_policy import refuse_code_reduce
from .retrieval import near_max_top_k, search_vectors
from .schema import (
    ColumnEntity, EnumValueEntity, ExampleEntity, RuleEntity, SynonymEntity, TableEntity,
    normalize_rule_scope,
)




def degraded_rules_result(result: object) -> bool:
    """Whether a ``get_rules`` answer is this module's retrieval-failure payload.

    This rules-specific predicate retains the original payload contract. Runtime
    consumers use ``knowledge_failures.degraded_knowledge_result`` for every knowledge
    tool, and ``knowledge_failures.degraded_record`` for retained transcripts.

    True only for the exact payload the except-branch above emits — never for a
    legitimately empty or small result, and never for other error rows (a server-level
    ``"get_rules failed: …"`` is a caller/session bug, not a knowledge outage), so
    degraded-state keyed on it cannot misfire on healthy zero-matches.

    Args:
        result (object): A decoded ``get_rules`` tool result — usually a list of rule
            rows, but any shape a transport error can produce is accepted.

    Returns:
        bool: True when at least one row carries :data:`RULES_RETRIEVAL_FAILED_ERROR`.
    """
    return isinstance(result, list) and any(
        isinstance(row, dict) and row.get("error") == RULES_RETRIEVAL_FAILED_ERROR
        for row in result
    )


def _cte_names(body: str) -> set[str]:
    """Lower-cased names the statement's own ``WITH`` header defines.

    A CTE normally never survives :func:`tables_in_sql`' ``known`` filter, since its name is not
    a real table — that filter is the only defence, and it fails exactly when the author names a
    CTE after a table that does exist. Then every reference to the CTE is reported as a table the
    statement may never read, which is the worse error of the two: the example menu sends the
    model to a part of the schema the example does not touch.

    Walks the header rather than pattern-matching it, because ``name AS ( ... )`` bodies nest and
    a non-greedy regex stops at the first inner ``)``. Column lists (``name(a, b) AS (...)``) and
    ``WITH RECURSIVE`` are handled; anything it cannot parse ends the walk, yielding what was read
    so far — an under-read degrades to today's behaviour, while an over-read would drop real
    tables.

    The one case this trades away: SQLite resolves a non-recursive CTE's own name inside its body
    to the outer scope, so ``WITH cards AS (SELECT * FROM cards)`` does read the real ``cards``,
    and excluding the name loses it. Distinguishing the two positions needs scope analysis rather
    than a scan, and a miss is the lesser error. Measured over the shipped corpora — 1534 gold
    statements (102 with a ``WITH``) and the 1277 stored examples (none) — the collision this
    guards against occurs 0 times, so it changes nothing on real data in either direction today.

    Args:
        body (str): The statement with comments and string literals already blanked.

    Returns:
        set[str]: Lower-cased CTE names, empty when there is no ``WITH`` header.
    """
    head = re.search(r"\bWITH\s+(?:RECURSIVE\s+)?", body, flags=re.I)
    if not head:
        return set()
    names, i, n = set(), head.end(), len(body)
    while i < n:
        entry = re.match(r'\s*[`"\[]?([A-Za-z_]\w*)[`"\]]?\s*(?:\([^)]*\)\s*)?AS\s*(?:MATERIALIZED\s*|'
                         r'NOT\s+MATERIALIZED\s*)?\(', body[i:], flags=re.I)
        if not entry:
            break
        names.add(entry.group(1).lower())
        depth, j = 0, i + entry.end() - 1          # 停在这个 CTE 体的开括号上
        while j < n:
            if body[j] == "(":
                depth += 1
            elif body[j] == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        comma = re.match(r"\s*,", body[j + 1:])
        if not comma:
            break
        i = j + 1 + comma.end()
    return names


def tables_in_sql(sql: str, known: set[str]) -> list[str]:
    """Table names a SQL statement actually reads, verified against the database's own schema.

    Matches identifiers that follow FROM/JOIN/INTO/UPDATE and keeps only the ones ``known``
    confirms are real tables. That filter is what makes this safe to run on model-written SQL:
    aliases (``account AS T1``), CTE names (``WITH champion_time AS ...``) and subquery labels
    all appear in the same syntactic position, and none of them are in ``known``, so none
    survive. Case is normalised to the schema's own spelling, because SQLite matches table
    names case-insensitively while the menu should display them the way the schema writes them.

    Derived tables are read through: ``FROM (SELECT ... FROM cards ...)`` reports ``cards``,
    while the subquery's own select list is never mined for names, so a column that happens to
    share a table's name cannot be reported as a table the statement does not read.

    Args:
        sql (str): The statement to scan. Comments and string literals are stripped first so a
            table name mentioned inside a quoted value cannot be picked up.
        known (set[str]): Real table names for this database, e.g. from ``sqlite_master`` or
            ``TableEntity``. An empty set yields an empty list — never a guess.

    Returns:
        list[str]: The distinct tables, in the schema's spelling. Treat the ORDER AS UNDEFINED —
        every FROM position is read before any JOIN/INTO/UPDATE position, so a main-query JOIN
        partner lands after tables that only appear in subqueries::

            FROM a JOIN b ON ... WHERE x IN (SELECT * FROM c)   ->  ['a', 'c', 'b']

        This used to be documented as "ordered by first appearance, so the most structurally
        important table of a JOIN chain tends to come first" — measured false on 48/1534 gold
        statements (3.1%) and 29/1277 stored examples (2.3%). Nothing consumes the order (the
        only readers are ``read_examples``' two projections, which pass the list straight into
        the menu), so the sequence was left alone and the promise removed instead: ``tables[0]``
        is NOT the driving table, and sorting or slicing this list means nothing.

    Example:
        >>> tables_in_sql("SELECT T1.a FROM account AS T1 JOIN loan T2 ON ...", {"account", "loan"})
        ['account', 'loan']
        >>> tables_in_sql("SELECT T.a FROM ( SELECT a FROM account ) AS T", {"account", "loan"})
        ['account']
    """
    if not sql or not known:
        return []
    canon = {t.lower(): t for t in known}
    body = re.sub(r"--[^\n]*|/\*.*?\*/", " ", sql, flags=re.S)
    # `"` is SQL's identifier quote AND, in SQLite, a fallback string quote — the same two
    # characters mean opposite things here. Blanket-stripping `"..."` as a literal deleted
    # double-quoted TABLE names outright (`FROM "Match"` -> [], and `"order"` is the only way to
    # write financial's reserved-word table at all). Blanket-keeping them would let a genuine
    # literal leak a table name into the menu, which is the worse error. So decide by position:
    # right after FROM/JOIN/INTO/UPDATE it is an identifier — rewrite the quotes to backticks,
    # which the identifier patterns below already accept — and everywhere else it stays a literal
    # and is blanked. 66 of the 1277 shipped example SQLs contain a double quote; none of them in
    # identifier position, so this changes nothing on shipped data and only stops future losses.
    body = re.sub(r"(?i)(\b(?:FROM|JOIN|INTO|UPDATE)\s+)\"([^\"]+)\"", r"\1`\2`", body)
    body = re.sub(r"'[^']*'|\"[^\"]*\"", " ", body)
    out: list[str] = []
    defined_here = _cte_names(body)

    def take(name: str) -> None:
        if name.lower() in defined_here:
            return
        real = canon.get(name.lower())
        if real and real not in out:
            out.append(real)

    # A FROM clause can list several tables by comma (`FROM account, loan` — an implicit cross
    # join, common in BIRD's gold), so the keyword anchor alone would only ever see the first.
    # Grab the whole clause up to the next structural keyword and read every comma-separated
    # entry's leading identifier; aliases after it are dropped by the `known` filter anyway.
    # The word-keywords need \b so `ONE` does not read as `ON`; the punctuation and end-of-string
    # anchors must stay OUTSIDE it, because \b before $ never matches a string ending in space —
    # which silently made a trailing `FROM account` (no WHERE, e.g. after a comment was stripped)
    # yield nothing at all.
    # FROM stops the clause too, so a derived table cannot hide the statement's only real table:
    # in `FROM (SELECT name FROM cards WHERE ...)` the outer clause otherwise runs all the way to
    # WHERE, swallowing the inner FROM, and finditer resumes past it — `cards` was never seen.
    # The lookbehind is what keeps a QUOTED table whose name is a stop word from ending the clause
    # it starts: `` FROM `order` `` matched ORDER immediately after the backtick, so the capture
    # was the empty string and the clause yielded nothing — and `order` is reserved, so quoting it
    # is mandatory, which made financial's `order` table structurally unreportable. A comma list
    # headed by such a table lost its unquoted siblings too. Only the quote characters are excluded
    # (not \w, which \b already covers), so a real `ORDER BY` still terminates the clause.
    _STOP = r"FROM|WHERE|GROUP|ORDER|HAVING|LIMIT|UNION|EXCEPT|INTERSECT|JOIN|ON"
    for m in re.finditer(rf"\bFROM\s+(.*?)(?=(?<![`\"\[])\b(?:{_STOP})\b|[);]|$)",
                         body, flags=re.I | re.S):
        clause = m.group(1).lstrip(" \t\r\n(")
        # A subquery names no table *at this position* — its own FROM is a separate match now —
        # and its select list must not be read: a column sharing a table's name (this benchmark
        # has both a `tags` table and a `posts.Tags` column) would be reported as a table the
        # statement never touches. The paren is stripped rather than the clause skipped because
        # a parenthesised join, `FROM (account JOIN loan ON ...)`, does name one here.
        if re.match(r"(?:SELECT|VALUES|WITH)\b", clause, flags=re.I):
            continue
        for part in clause.split(","):
            ident = re.match(r"\s*[`\[]?([A-Za-z_][A-Za-z0-9_]*)", part)
            if ident:
                take(ident.group(1))
    for m in re.finditer(r"\b(?:JOIN|INTO|UPDATE)\s+[`\[]?([A-Za-z_][A-Za-z0-9_]*)[`\]]?",
                         body, flags=re.I):
        take(m.group(1))
    return out


def add_example(
    ws: hb.HeavenBase,
    db_id: str,
    question: str,
    sql: str,
    tables: list[str] | None = None,
    difficulty: str = "medium",
    name: str | None = None,
) -> str:
    """Store a new NL→SQL example in the knowledge base. Returns entity id.

    Pass an explicit `name` to make the record stable across updates —
    upsert with the same name overwrites the previous version.

    ``tables`` is derived from ``sql`` when omitted, against this workspace's own
    ``TableEntity`` names, so a runtime save lands the same two selection signals the build-time
    injector writes. It used to store ``[]``, which `get_examples`' two-stage menu then served as
    "this example touches no tables" — the build path had the identical hole and all 1277 shipped
    examples carried ``[]``.

    ``question`` is stored verbatim. Pass it in the shape a live question arrives in —
    ``question + "\\n[Hint]: " + evidence`` when there is evidence (``eval_bird`` builds exactly
    that, and ``inject_examples._question_with_hint`` stores exactly that) — because
    ``question_vec`` is computed from this string and is compared against the live query's
    vector. A bare question saved next to hint-bearing queries is noise on one side only.

    Args:
        ws (heavenbase.HeavenBase): Workspace to write into.
        db_id (str): Database this example belongs to.
        question (str): The question, hint block included when there is one (see above).
        sql (str): Validated SQL answering it.
        tables (list[str] | None): Tables the SQL reads. Derived from ``sql`` when None.
        difficulty (str): easy / medium / hard.
        name (str | None): Stable record name; an md5 of ``db_id:question`` when None.

    Returns:
        str: The entity id of the stored example, or ``""`` if the upsert returned nothing.
    """
    import hashlib
    if name is None:
        name = hashlib.md5(f"{db_id}:{question}".encode()).hexdigest()[:16]
    if tables is None:
        known = {r.get("name") for r in
                 ws.query(TableEntity).where({"db_id": {"$eq": db_id}}).execute().rows()
                 if r.get("name")}
        tables = tables_in_sql(sql, known)
    row = ExampleEntity.from_dict({
        "db_id":      db_id,
        "name":       name,
        "question":   question,
        "sql":        sql,
        "tables":     tables,
        "difficulty": difficulty,
    })
    row_dict = row.to_dict()
    # Computed here, outside HeavenBase's Context-wide write lock, so a runtime save
    # never holds every other worker's writes hostage to one embedding round-trip
    # (register.py's disease and fix; upstream Magolor/HeavenBase#93).
    row_dict["question_vec"] = embed(row_dict.get("question") or "")
    ids = ws.upsert_many(ExampleEntity, [row_dict])
    return ids[0] if ids else ""


def synonym_row(
    db_id: str,
    table_name: str,
    synonym: str,
    canonical: str,
    col_name: str = "",
) -> dict[str, Any]:
    """The row :func:`add_synonym` stores, without its vector.

    Its ``name`` is the md5 of ``db_id:table:column:synonym``, so writing the same mapping
    again replaces the row instead of adding one, and two rows of one name hold one synonym.

    Args:
        db_id: Database the synonym belongs to.
        table_name: Table the synonym applies to.
        synonym: The user's phrase.
        canonical: The schema name it maps to.
        col_name: The column, ``""`` for a table-level synonym.

    Returns:
        dict[str, Any]: The ``SynonymEntity`` row, ``synonym_vec`` not set.
    """
    import hashlib
    name = hashlib.md5(f"{db_id}:{table_name}:{col_name}:{synonym}".encode()).hexdigest()[:16]
    return SynonymEntity.from_dict({
        "db_id":      db_id,
        "name":       name,
        "table_name": table_name,
        "col_name":   col_name,
        "synonym":    synonym,
        "canonical":  canonical,
    }).to_dict()


def add_synonym(
    ws: hb.HeavenBase,
    db_id: str,
    table_name: str,
    synonym: str,
    canonical: str,
    col_name: str = "",
) -> str:
    """Store a synonym mapping for a table or column. Returns entity id."""
    row_dict = synonym_row(db_id, table_name, synonym, canonical, col_name)
    row_dict["synonym_vec"] = embed(row_dict.get("synonym") or "")  # unlocked, as in add_example
    ids = ws.upsert_many(SynonymEntity, [row_dict])
    return ids[0] if ids else ""


def save_rule(
    ws: hb.HeavenBase,
    db_id: str,
    title: str,
    desc: str,
    scope: str = "global",
    tags: list[str] | None = None,
    triggers: list[str] | None = None,
    example: str = "",
) -> str:
    """Store a rule under the stable identity ``(db_id, title)``.

    Args:
        ws: Workspace holding the rule.
        db_id: Database this rule applies to.
        title: Stable snake_case rule title.
        desc: Natural-language rule content.
        scope: ``global`` for cross-database rules or ``database`` for this ``db_id``.
        tags: Optional facets such as ``where`` or ``groupby``.
        triggers: Literal phrases that inject this rule into a conversation.
        example: Optional compact example; prefer leaving it empty.

    Returns:
        str: Stable entity id, or an empty string if no row was written.

    Raises:
        ValueError: ``title`` or ``scope`` violates the RuleEntity contract.
    """
    if re.fullmatch(r"[a-z][a-z0-9_]{0,254}", title) is None:
        raise ValueError(
            "rule title must be a lower snake_case identifier of at most 255 characters"
        )
    row = RuleEntity.from_dict({
        "db_id": db_id,
        "scope": normalize_rule_scope(scope),
        "title": title,
        "desc": str(desc or "").strip(),
        "tags": _normalize_rule_values(tags),
        "triggers": _normalize_rule_values(triggers),
        "example": str(example or "").strip(),
    })
    row_dict = row.to_dict()
    row_dict["title_vec"] = embed_title(row_dict.get("title") or "")     # unlocked, as in
    row_dict["desc_vec"] = embed_rule_desc(row_dict.get("desc") or "",   # add_example
                                           row_dict.get("example") or "")
    ids = ws.upsert_many(RuleEntity, [row_dict])
    return ids[0] if ids else ""


def update_rule(
    ws: hb.HeavenBase,
    db_id: str,
    entity_id: str,
    desc: str,
    scope: str = "",
    tags: list[str] | None = None,
    triggers: list[str] | None = None,
    example: str | None = None,
) -> dict[str, Any]:
    """Update mutable rule content while preserving its title-derived identity.

    The title cannot be changed by this operation because it is part of the stable id.
    HeavenBase recomputes ``desc_vec`` from ``example + desc`` when either input changes.
    The stored vector must be removed from the submitted row first: compute hooks skip a
    derived field that is already present.

    Args:
        ws: Workspace holding the rule.
        db_id: Database that owns the rule.
        entity_id: Stable rule id returned by :func:`save_rule`.
        desc: Replacement natural-language content.
        scope: Optional replacement scope; empty leaves it unchanged.
        tags: Optional replacement tags; ``None`` leaves them unchanged.
        triggers: Optional replacement triggers; ``None`` leaves them unchanged.
        example: Optional replacement example; ``None`` leaves it unchanged.

    Returns:
        dict[str, Any]: Updated rule projection, or an error mapping.
    """
    row = ws.get(entity_id, RuleEntity)
    if row is None or row.get("db_id") != db_id:
        return {"error": f"RuleEntity '{entity_id}' not found"}
    updated = {
        key: value for key, value in row.items()
        if not key.endswith("_vec")
    }
    updated["desc"] = desc
    if scope:
        updated["scope"] = normalize_rule_scope(scope)
    if tags is not None:
        updated["tags"] = _normalize_rule_values(tags)
    if triggers is not None:
        updated["triggers"] = _normalize_rule_values(triggers)
    if example is not None:
        updated["example"] = example
    ws.upsert(RuleEntity, updated)
    return _rule_projection(updated)


def _rule_projection(row: dict[str, Any]) -> dict[str, Any]:
    """Project a rule to prompt-safe public fields for readers and writers."""
    out = {
        "entity_id": row.get("object_id"),
        "scope": row.get("scope"),
        "title": row.get("title"),
        "desc": row.get("desc"),
        "tags": row.get("tags") or [],
    }
    if row.get("example"):
        out["example"] = row["example"]
    return out


def _normalize_rule_values(values: list[str] | None) -> list[str]:
    """Strip, deduplicate, and preserve the order of rule tags or triggers."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
        key = unicodedata.normalize("NFKC", text).casefold()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _normalize_rule_text(text: str) -> str:
    """Normalize text shared by lexical ranking and literal trigger matching."""
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"[^\w\u3400-\u9fff]+", " ", normalized)
    return " ".join(normalized.split())


def _trigger_matches(text: str, trigger: str) -> bool:
    """Return whether one literal trigger matches normalized conversation text."""
    haystack = _normalize_rule_text(text)
    needle = _normalize_rule_text(trigger)
    if not haystack or not needle:
        return False
    if re.search(r"[\u3400-\u9fff]", needle):
        return needle in haystack
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack) is not None


QUERY_ENTITIES: dict[str, type[hb.Entity]] = {
    "rule":       RuleEntity,
    "example":    ExampleEntity,
    "synonym":    SynonymEntity,
    "column":     ColumnEntity,
    "table":      TableEntity,
    "enum_value": EnumValueEntity,
}


QUERY_LIMIT_MAX = 200


def _drop_vectors(row: dict[str, Any]) -> dict[str, Any]:
    """Return ``row`` without any embedding field (every ``*_vec``).

    MEASURED, not hygiene. One RuleEntity row carrying a 1024-float vector at full
    repr precision — serialises to 45,482 characters, so a real database's 69 rules would be
    roughly 785K tokens and its 107 examples roughly 1.2M: a single call would blow any
    context window on exactly the entity types the model most wants to enumerate. Every
    curated retrieval path strips these for the same reason (see ``read_examples`` /
    ``get_rules``); this one cannot use ``.select()`` to do it (that would also drop
    ``object_id``), so it filters by name — which also covers any vector field added later.
    """
    return {k: v for k, v in row.items() if not k.endswith("_vec")}


def query_where_fields(cls: type[hb.Entity]) -> tuple[str, ...]:
    """Return the field names a :func:`query_knowledge` filter may name for ``cls``.

    The entity's own schema minus the embedding fields (equality on a 1024-float vector is
    meaningless — similarity is what the ``near()``-based tools above expose), with the row id
    presented as ``entity_id`` so filter and result speak the same word as every other
    knowledge tool.
    """
    return tuple("entity_id" if f == "object_id" else f
                 for f in cls.schema().fields if not f.endswith("_vec"))


def _normalize_where(cls: type[hb.Entity], where: Any) -> Any:
    """Validate one filter's field names and return it in storage vocabulary.

    Two jobs, both for a caller that cannot read the entity definitions: an unknown field is
    rejected with the list of the ones that exist (left to heavenbase the same typo surfaces as
    ``LookupError: No storage binding for rule-entity.scop``, which names nothing usable), and
    the model-facing ``entity_id`` is translated back to the stored ``object_id``. Operator keys
    (``$eq``, ``$or``, …) pass through and their operands are walked, so a nested ``$or`` over
    field keys is validated too.

    Raises:
        ValueError: A non-operator key is not a filterable field of ``cls``.
    """
    valid = query_where_fields(cls)
    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            out: dict[Any, Any] = {}
            for key, value in node.items():
                if isinstance(key, str) and key.startswith("$"):
                    out[key] = walk(value)   # logical operator — its operand carries field keys
                    continue                 # a field key's operand is {op: literal}, not fields
                base = str(key).split(".", 1)[0]
                if base not in valid and base != "object_id":
                    raise ValueError(
                        f"unknown field {key!r} for entity {cls.schema().entity_id!r} — "
                        f"filterable fields are: " + ", ".join(valid))
                out["object_id" if key == "entity_id" else key] = value
            return out
        if isinstance(node, (list, tuple)):
            return [walk(item) for item in node]
        return node
    return walk(where)


def query_records(
    ws: hb.HeavenBase,
    entities: Mapping[str, type[hb.Entity]],
    entity: str,
    where: dict[str, Any] | None = None,
    limit: int = 50,
    scope: tuple[str, str] | None = None,
    limit_max: int = QUERY_LIMIT_MAX,
) -> list[dict[str, Any]]:
    """Return rows of one registered entity matching an exact filter.

    The engine behind :func:`query_knowledge`, factored out so other workspaces — the lab
    analytics workspace — can mount the same model-friendly read path (field validation
    with usable errors, embedding stripping, a hard row cap) over their own entity
    registry without inheriting the knowledge surface's per-database scoping.

    Args:
        ws: Workspace holding the entity's rows.
        entities: Model-facing name to Entity class registry for this surface.
        entity: Friendly entity name; a key of ``entities``.
        where: Mongo-style filter mapping over that entity's own fields, or ``None``.
        limit: Caller's row cap, clamped into ``1 .. limit_max``.
        scope: Optional ``(field, value)`` equality this call must not escape, or
            ``None`` for an unscoped surface.
        limit_max: Hard ceiling on returned rows, whatever the caller asks.

    Returns:
        list[dict[str, Any]]: Matching rows with every embedding field removed and
        ``object_id`` renamed to ``entity_id``.

    Raises:
        ValueError: ``where`` names HeavenBase's ``code_reduce``
            (:func:`~rubiksql_hb.query_policy.refuse_code_reduce`), ``entity`` names no
            known entity type, or ``where`` filters a field that entity does not have; the
            last two messages list the valid names.
    """
    refuse_code_reduce(where)
    cls = entities.get((entity or "").strip().lower())
    if cls is None:
        raise ValueError(f"unknown entity {entity!r} — valid names are: "
                         + ", ".join(entities))
    # Clamp UP to 1, never down to 0: a limit=0 call answering with `[]` would be read as
    # "this database has no such knowledge", the one failure mode this module refuses to
    # produce silently (see _retrieval_failed).
    n = max(1, min(int(limit), limit_max))
    q = ws.query(cls)
    if where:
        q = q.where(_normalize_where(cls, where))
    if scope is not None:
        # The scope enters as its OWN clause, LAST: where() ANDs clauses, so a caller's
        # filter on the same field — or an $or spanning values — can only narrow this
        # scope, never widen it.
        field, value = scope
        q = q.where({field: {"$eq": value}})
    rows = q.limit(n).execute().rows()
    out: list[dict[str, Any]] = []
    for row in rows[:n]:   # belt and braces: the cap holds even if a backend ignores limit()
        rec = _drop_vectors(row)
        if "object_id" in rec:
            rec["entity_id"] = rec.pop("object_id")
        out.append(rec)
    return out


def _frame_vocab(cls: type[hb.Entity], spec: dict[str, Any]) -> dict[str, Any]:
    """Translate one :func:`query_frame` spec into storage vocabulary.

    The model speaks ``entity_id`` (every knowledge tool presents the row id that way);
    storage speaks ``object_id``. The filter additionally gets
    :func:`_normalize_where`'s field validation so a typo fails with the list of real
    fields instead of a storage-layer ``LookupError``. ``select`` / ``group_by`` /
    ``order`` entries and ``aggregate`` field references are renamed only — heavenbase
    validates them itself with usable messages.

    Args:
        cls: Resolved entity class.
        spec: Caller's ``query_json`` specification.

    Returns:
        dict[str, Any]: A translated copy; the input mapping is not mutated.
    """
    def col(name: str) -> str:
        return "object_id" if name == "entity_id" else name

    def rename_keys(node: Any) -> Any:
        """Rename non-operator dict keys through ``col`` at every nesting level."""
        if isinstance(node, dict):
            return {(key if str(key).startswith("$") else col(str(key))): rename_keys(value)
                    for key, value in node.items()}
        if isinstance(node, list):
            return [rename_keys(item) for item in node]
        return node

    # Only well-shaped clauses are touched: a scalar where a list belongs, or a non-string
    # entry, passes through untouched so heavenbase's own shape errors ("order clause must
    # be an array", "order entries must be strings") stay reachable and usable.
    body = dict(spec)
    if body.get("filter"):
        body["filter"] = _normalize_where(cls, body["filter"])
    for key in ("select", "group_by"):
        if isinstance(body.get(key), list):
            body[key] = [col(field) if isinstance(field, str) else field
                         for field in body[key]]
    if isinstance(body.get("order"), list):
        body["order"] = [
            ("-" + col(field[1:]) if field.startswith("-") else col(field))
            if isinstance(field, str) else field
            for field in body["order"]
        ]
    if isinstance(body.get("aggregate"), dict):
        body["aggregate"] = {
            out: {
                op: {**params, "field": col(params["field"])}
                if isinstance(params, dict) and isinstance(params.get("field"), str)
                else params
                for op, params in metric.items()
            } if isinstance(metric, dict) else metric
            for out, metric in body["aggregate"].items()
        }
    if isinstance(body.get("having"), dict):
        # having keys are group fields and metric aliases; group keys speak entity_id too.
        body["having"] = rename_keys(body["having"])
    return body


def query_frame(
    ws: hb.HeavenBase,
    entities: Mapping[str, type[hb.Entity]],
    entity: str,
    spec: dict[str, Any] | None = None,
    scope: tuple[str, str] | None = None,
    limit_max: int = QUERY_LIMIT_MAX,
) -> dict[str, Any]:
    """Run one full-spec workspace query and return a vector-free result frame.

    The aggregation-capable sibling of :func:`query_records`: same model-facing entity
    registry, same scoping contract, same embedding guardrail — but the whole
    ``query_json`` grammar (``select``, ``group_by`` + ``aggregate``, ``having``,
    ``order``, ``offset``) instead of a bare filter, so a caller can compute over the
    knowledge base rather than page through it.

    Args:
        ws: Workspace holding the entity's rows.
        entities: Model-facing name to Entity class registry for this surface.
        entity: Friendly entity name; a key of ``entities``.
        spec: ``query_json`` specification, or ``None`` for a plain scan. ``limit`` is
            clamped into ``1 .. limit_max``; field names may use ``entity_id``.
        scope: Optional ``(field, value)`` equality ANDed AROUND the caller's filter —
            an ``$or`` inside the filter can only narrow the scope, never widen it.
            ``None`` disables scoping.
        limit_max: Hard ceiling on returned rows, whatever the caller asks.

    Returns:
        dict[str, Any]: ``{"columns", "rows", "n", "truncated"}`` with every embedding
        field removed — whatever the spec selected — and ``object_id`` presented as
        ``entity_id``. ``truncated`` is True when more rows (or more GROUPS, for a
        ``group_by``) exist beyond the returned page; heavenbase applies the limit to
        group rows post-aggregation, so without this flag a capped breakdown would be
        indistinguishable from a complete one.

    Raises:
        ValueError: ``spec`` names HeavenBase's ``code_reduce`` anywhere
            (:func:`~rubiksql_hb.query_policy.refuse_code_reduce`), ``entity`` names no
            known entity type, or the filter names a field that entity does not have; the
            last two messages list the valid names.
    """
    refuse_code_reduce(spec)
    cls = entities.get((entity or "").strip().lower())
    if cls is None:
        raise ValueError(f"unknown entity {entity!r} — valid names are: "
                         + ", ".join(entities))
    body = _frame_vocab(cls, spec or {})
    # Clamp UP to 1 like query_records: a limit=0 answering with nothing would read as
    # "this database has no such knowledge".
    limit = body.get("limit")
    limit = max(1, min(50 if limit is None else int(limit), limit_max))
    body["limit"] = limit + 1   # one probe row past the page reveals truncation
    if scope is not None:
        field, value = scope
        clause = {field: {"$eq": value}}
        caller = body.get("filter")
        body["filter"] = {"$and": [caller, clause]} if caller else clause
    # Apply before projection/aggregation: historical pinned query_spec tools can
    # select SQL alone, so their result need not carry the question to mask later.
    from .example_mask import current_question
    from .question import normalized_question

    target = normalized_question(current_question()) if (entity or "").strip().lower() == "example" else ""
    if target:
        scan = ws.query(cls)
        if scope is not None:
            scan = scan.where({scope[0]: {"$eq": scope[1]}})
        questions = scan.select("question").execute().rows()
        excluded = sorted({row["question"] for row in questions if normalized_question(row.get("question")) == target})
        if excluded:
            # Use the same exclusion form as vector retrieval; $nin is not registered.
            clause = {"$not": {"question": {"$in": excluded}}}
            caller = body.get("filter")
            body["filter"] = {"$and": [caller, clause]} if caller else clause
    frame = ws.query_json(cls, body).execute()
    raw = frame.rows()
    rows: list[dict[str, Any]] = []
    for row in raw[:limit]:   # the cap holds even if a backend ignores limit
        rec = {k: v for k, v in row.items() if not str(k).endswith("_vec")}
        if "object_id" in rec:
            rec["entity_id"] = rec.pop("object_id")
        rows.append(rec)
    columns = ["entity_id" if col == "object_id" else col
               for col in frame.to_dict()["columns"] if not str(col).endswith("_vec")]
    return {"columns": columns, "rows": rows, "n": len(rows),
            "truncated": len(raw) > limit}


def query_knowledge(
    ws: hb.HeavenBase,
    db_id: str,
    entity: str,
    where: dict[str, Any] | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return knowledge-base records matching an exact filter.

    The structured counterpart to the top-k similarity readers above: same records, but
    filtered, enumerable and countable instead of ranked by cosine distance.

    Args:
        ws: Workspace holding this database's knowledge.
        db_id: The only database whose knowledge this call can read.
        entity: Friendly entity name; one of :data:`QUERY_ENTITIES`.
        where: Mongo-style filter mapping over that entity's own fields, or ``None``.
        limit: Caller's row cap, clamped into ``1 .. QUERY_LIMIT_MAX``.

    Returns:
        list[dict[str, Any]]: Matching rows with every embedding field removed and
        ``object_id`` renamed to ``entity_id`` — the same id ``expand_examples`` and
        ``update_rule`` accept.

    Raises:
        ValueError: ``entity`` names no known entity type, or ``where`` filters a field that
            entity does not have; both messages list the valid names.
    """
    return query_records(
        ws, QUERY_ENTITIES, entity, where=where, limit=limit, scope=("db_id", db_id),
    )
