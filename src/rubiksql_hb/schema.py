import heavenbase as hb

from .embedding import EMBEDDING_DIM, embed, embed_rule_desc, embed_title

RULE_SCOPES = ("global", "database")


def normalize_rule_scope(scope: str | None = None) -> str:
    """Return one canonical rule scope.

    Args:
        scope: Candidate scope. Empty values default to ``global``.

    Returns:
        str: Either ``global`` or ``database``.

    Raises:
        ValueError: The value is not part of the RuleEntity scope contract.
    """
    normalized = str(scope or "global").strip() or "global"
    if normalized not in RULE_SCOPES:
        allowed = ", ".join(RULE_SCOPES)
        raise ValueError(
            f"rule scope must be one of: {allowed}; got {normalized!r}"
        )
    return normalized


# Row ids follow the HeavenBase default: a row that carries no explicit ``object_id`` gets
# ``hash_id(<entity-id>, <name>)``, so name-seeded upserts stay idempotent without any local
# id code. Rows whose ``name`` alone is not unique (a column called `amount` exists in many
# tables) pass an explicit ``object_id`` at the call site — see register._safe_id.


class DatabaseEntity(hb.Entity):
    """Database-level overview for one external database.

    One row per database. ``profile_desc`` is the model-facing summary that has no
    natural home at table or column grain: which of several same-named columns
    (url, points, name) belongs to which table, which table is cumulative versus
    per-event, and any cross-table reading conventions. It renders at the top of
    ``get_db_info`` — passively, on every schema read, with no retrieval step —
    because the model needs it at the moment it scans the schema to pick tables,
    before it knows what to retrieve.
    """
    db_id        = hb.field(hb.ShortText,  desc="Database name").required()
    profile_desc = hb.field(hb.MediumText, desc="LLM database overview (cross-table "
                                                "disambiguation, grain map)").default("")


class TableEntity(hb.Entity):
    """Schema metadata for one table in an external database."""
    db_id        = hb.field(hb.ShortText,           desc="Database name").required()
    name         = hb.field(hb.ShortText,           desc="Table name").required()
    desc         = hb.field(hb.MediumText,          desc="Official docs: what this table stores").default("")
    # LLM layer, mirroring ColumnEntity: `desc` holds official documentation, profile_desc
    # the generated analysis (grain, sibling-table contrasts, derived-column semantics).
    # db_info/tab_info compose the two at serving time.
    profile_desc = hb.field(hb.MediumText,          desc="LLM table profile").default("")
    n_rows   = hb.field(hb.Integer,             desc="Approximate row count").optional()
    pks      = hb.field(hb.Array[hb.ShortText], desc="Primary key column names").default([])
    disabled = hb.field(hb.Boolean,             desc="Exclude from schema").default(False)


class ColumnEntity(hb.Entity):
    """Schema metadata for one column in an external database table."""
    db_id         = hb.field(hb.ShortText,  desc="Database name").required() \
                       .store(to="main", strategy="inline") \
                       .store(to="chroma", strategy="inline")
    table_name    = hb.field(hb.ShortText,  desc="Parent table name").required()
    name          = hb.field(hb.ShortText,  desc="Column name").required()
    datatype_orig = hb.field(hb.ShortText,  desc="Raw SQL type, e.g. VARCHAR(255)").optional()
    datatype_anno = hb.field(hb.ShortText,  desc="Semantic type: TEXT / CATEGORICAL / INTEGER / FLOAT / BOOLEAN").optional()
    desc          = hb.field(hb.MediumText, desc="What this column represents").default("")
    desc_vec      = hb.field(hb.Vector[EMBEDDING_DIM], desc="Embedding of desc for fuzzy column search") \
                       .store(to="chroma", strategy="vector") \
                       .compute(embed, inputs=["desc"])
    profile_desc  = hb.field(hb.LongText,  desc="LLM-generated summary of column statistics (null count, distinct count, min/max, top values, format)").default("")
    is_pk         = hb.field(hb.Boolean,    desc="Part of primary key").default(False)
    fk_ref        = hb.field(hb.ShortText,  desc="FK reference, e.g. users.id").optional()
    disabled      = hb.field(hb.Boolean,    desc="Exclude from schema").default(False)


class ExampleEntity(hb.Entity):
    """A historical NL→SQL example for few-shot retrieval."""
    db_id        = hb.field(hb.ShortText,  desc="Database this example belongs to").required() \
                      .store(to="main", strategy="inline") \
                      .store(to="chroma", strategy="inline")
    name         = hb.field(hb.ShortText,  desc="Short label for this example, used as id seed").required()
    question     = hb.field(hb.MediumText, desc="Natural language question").required()
    question_vec = hb.field(hb.Vector[EMBEDDING_DIM], desc="Embedding of question for similarity retrieval") \
                      .store(to="chroma", strategy="vector") \
                      .compute(embed, inputs=["question"])
    sql          = hb.field(hb.LongText,   desc="Correct SQL for the question").required()
    tables       = hb.field(hb.Array[hb.ShortText], desc="Tables referenced in the SQL").default([])
    difficulty   = hb.field(hb.ShortText,  desc="easy / medium / hard").default("medium")
    contract     = hb.field(hb.Json, desc="Independent semantic admission and example-only applicability scope").default({})


class SynonymEntity(hb.Entity):
    """A user-defined synonym mapping for a table or column."""
    db_id      = hb.field(hb.ShortText, desc="Database name").required()
    name       = hb.field(hb.ShortText, desc="Unique label for this synonym, used as id seed").required()
    table_name = hb.field(hb.ShortText, desc="Table the synonym applies to").required()
    col_name   = hb.field(hb.ShortText, desc="Column name, empty string if table-level synonym").default("")
    synonym    = hb.field(hb.ShortText, desc="Synonym or alias").required()
    # Embedded so a question can reach a mapping it does not word literally. The literal pass
    # (nl2sql._mentions) measured 1417 hits at 100% precision over the 1534-question set, but it
    # can only fire on the exact surface form — "how much does the hero weigh" never reaches the
    # stored phrase "weight". The vector is over `synonym`, not `canonical`, because the query
    # side is the user's wording, and the two sides must live in the same space.
    # Stored in BOTH backends on purpose. chroma is where a `.near()` would read it; `main` is
    # what `nl2sql._synonym_context` can actually reach — that path opens the workspace SQLite
    # read-only rather than paying `open_workspace`'s 16–55 s per process, so a chroma-only
    # vector is invisible to it and the phrases would be re-embedded at runtime every process.
    synonym_vec = hb.field(hb.Vector[EMBEDDING_DIM], desc="Embedding of the synonym phrase for semantic retrieval") \
                     .store(to="main", strategy="inline") \
                     .store(to="chroma", strategy="vector") \
                     .compute(embed, inputs=["synonym"])
    canonical  = hb.field(hb.ShortText, desc="Canonical name this synonym maps to").required()


class EnumValueEntity(hb.Entity):
    """A distinct categorical value found in a CATEGORICAL column."""
    db_id      = hb.field(hb.ShortText, desc="Database name").required() \
                    .store(to="main", strategy="inline") \
                    .store(to="chroma", strategy="inline")
    table_name = hb.field(hb.ShortText, desc="Table name").required()
    col_name   = hb.field(hb.ShortText, desc="Column name").required()
    value      = hb.field(hb.ShortText, desc="The distinct cell value").required()
    value_vec  = hb.field(hb.Vector[EMBEDDING_DIM], desc="Embedding of value for semantic search") \
                    .store(to="chroma", strategy="vector") \
                    .compute(embed, inputs=["value"])


class RuleEntity(hb.Entity):
    """A business rule or domain fact that governs how queries should be interpreted."""
    object_id = hb.field(hb.Identifier, desc="Stable id derived from db_id and title") \
        .compute(hb.utils.hash_id, inputs=["db_id", "title"])
    db_id = hb.field(hb.ShortText, desc="Database this rule applies to").required() \
        .store(to="main", strategy="inline") \
        .store(to="chroma", strategy="inline")
    scope = hb.field(
        hb.Categorical["global", "database"],
        desc=(
            "'global' for cross-database rules; 'database' for rules specific "
            "to this row's db_id"
        ),
    ).default("global")
    title = hb.field(hb.ShortText, desc="Stable snake_case rule title").required()
    desc = hb.field(hb.MediumText, desc="Reusable rule instruction in natural language").required()
    tags = hb.field(hb.Array[hb.ShortText], desc="Rule facets such as where or groupby") \
        .default([])
    triggers = hb.field(
        hb.Array[hb.ShortText],
        desc="Literal phrases that inject this rule during a conversation",
    ).default([])
    example = hb.field(hb.MediumText, desc="Optional illustrative question; prefer leaving empty") \
        .default("")
    title_vec = hb.field(
        hb.Vector[EMBEDDING_DIM],
        desc="Embedding of the normalized title for semantic retrieval",
    ).store(to="main", strategy="inline") \
        .store(to="chroma", strategy="vector") \
        .compute(embed_title, inputs=["title"])
    desc_vec = hb.field(
        hb.Vector[EMBEDDING_DIM],
        desc="Embedding of example + desc for semantic retrieval",
    ).store(to="main", strategy="inline") \
        .store(to="chroma", strategy="vector") \
        .compute(embed_rule_desc, inputs=["desc", "example"])
