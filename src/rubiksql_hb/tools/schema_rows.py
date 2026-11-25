"""Pinned implementation of schema_rows."""


def _col_row(c: dict) -> dict:
    """Strip internal fields from a column record for agent consumption."""
    return {
        "name":     c["name"],
        "type":     c.get("datatype_orig", ""),
        "is_pk":    c.get("is_pk", False),
        "fk_ref":   c.get("fk_ref", "") or "",
        "desc":     c.get("desc", "") or "",
    }


def _table_desc(row: dict) -> str:
    """Compose the served table description: official docs + LLM profile."""
    doc = (row.get("desc") or "").strip()
    prof = (row.get("profile_desc") or "").strip()
    if doc and prof:
        return f"{doc} — {prof}"
    return doc or prof
