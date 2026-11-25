"""Pinned implementation of check_functional_dependency."""
from typing import Any
import sqlalchemy as sa
from ..sql_host import _engine, _quote_ident


def fd_check(
    url: str,
    table: str,
    x_cols: list[str],
    y_cols: list[str],
) -> dict[str, Any]:
    """
    Check whether X functionally determines Y (X → Y) in a table.

    For every distinct combination of X values, there must be exactly one
    distinct combination of Y values.  Returns holds=True if the FD holds,
    False otherwise with a count of violating X-groups.

    ``table``, ``x_cols`` and ``y_cols`` arrive from the model via the
    ``check_functional_dependency`` tool, so every identifier is quoted through
    ``_quote_ident`` (which doubles embedded quotes) before it reaches the statement, and the
    engine is the read-only one — a malformed identifier can only produce a SQL error, never
    a write. The NULL substitute is the string literal ``''``; ``""`` is a *quoted identifier*
    that only resolves to a string by SQLite's legacy double-quote fallback.
    """
    x_expr = ", ".join(_quote_ident(c) for c in x_cols)
    # Concatenate Y columns into a single comparable string
    y_concat = " || '§' || ".join(f"COALESCE(CAST({_quote_ident(c)} AS TEXT), '')" for c in y_cols)

    sql = f"""
        SELECT COUNT(*) AS violation_groups
        FROM (
            SELECT {x_expr}
            FROM {_quote_ident(table)}
            GROUP BY {x_expr}
            HAVING COUNT(DISTINCT {y_concat}) > 1
        ) t
    """
    engine = _engine(url)
    try:
        with engine.connect() as conn:
            violations = conn.execute(sa.text(sql)).scalar() or 0
    except Exception as e:
        return {"error": str(e)}
    finally:
        engine.dispose()

    holds = violations == 0
    x_str = ", ".join(x_cols)
    y_str = ", ".join(y_cols)
    return {
        "holds": holds,
        "violation_groups": violations,
        "msg": (
            f"({x_str}) → ({y_str}) holds in '{table}'"
            if holds else
            f"({x_str}) → ({y_str}) violated: {violations} X-group(s) map to multiple Y values"
        ),
    }


def check_functional_dependency(db_id: str, table: str, x_cols: list[str], y_cols: list[str]) -> dict:
    """Check whether columns X functionally determine columns Y (X → Y) in a table."""
    from ..server import _db_url

    url = _db_url(db_id)
    if not url:
        return {"error": f"db_id '{db_id}' not registered."}
    return fd_check(url, table, x_cols, y_cols)
