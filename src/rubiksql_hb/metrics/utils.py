"""
Utility functions for metric calculations.
"""

from typing import Any, Dict, Tuple
from decimal import Decimal
import re


def strip_sql_comments(sql: str) -> str:
    """Remove SQL comments (``-- ...`` and ``/* ... */``).

    Inlined verbatim from ahvn.utils.db.db_utils (AgentHeaven-dev@7b6e6c1) so the
    metrics package carries no ahvn dependency.
    """
    if not sql:
        return ""
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


def norm_val(v: Any, ndigits: int | None = 3) -> Any:
    """
    Normalize a value for comparison.

    Args:
        v: Value to normalize.
        ndigits: Decimal places for rounding floats; None = NO rounding — raw value.
            BF/SF, the only metrics normalizing through here, use the default 3;
            EX never passes through this module (see ``metrics.official_ex``).

    Returns:
        Normalized value.
    """
    if ndigits is None:
        return v
    try:
        # bool is a subclass of int; keep it as-is so True/False aren't coerced to 1.0/0.0.
        if isinstance(v, bool):
            return v
        # Round through float so representation noise cannot split values the answer
        # treats as equal: the same 0.1 arriving as Decimal from one driver and as
        # float from another (Decimal("0.1") != 0.1 in Python), or a SUM whose last
        # bits differ between two arithmetically equivalent queries. int-vs-float is
        # NOT a motive — 118 == 118.0 already compares and hashes equal, so positional
        # tuples and Counter bags match unaided.
        if isinstance(v, (int, float, Decimal)):
            return round(float(v), ndigits)
    except Exception:
        pass
    return v


def norm_row(row: Dict[str, Any], ndigits: int | None = 3) -> Tuple[Any, ...]:
    """
    Normalize a row dictionary to a tuple for comparison.

    Args:
        row: Dictionary representing a database row.
        ndigits: Number of decimal places for rounding floats; None = no rounding.
            3 is the default all BF/SF callers use.

    Returns:
        Tuple of normalized values.
    """
    return tuple(norm_val(v, ndigits) for v in row.values())


def has_order_by(sql: str) -> bool:
    """Check if a SQL query contains ORDER BY clause."""
    s = strip_sql_comments(sql or "")
    return bool(re.search(r"\border\s+by\b", s, flags=re.IGNORECASE))
