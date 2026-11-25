"""Pinned implementation of save_example."""


def save_example(db_id: str, question: str, sql: str, tables: list[str] = None, difficulty: str = "medium") -> str:
    """Save a validated NL→SQL pair to the knowledge base for future retrieval.

    Pass `question` exactly as it reached you, hint block included — retrieval compares the
    stored question's embedding against the next question's, so a stripped-down paraphrase is
    matched against text nobody will ask. Omit `tables` and it is read out of the SQL.
    """
    from ..server import add_example, _ws_for

    return add_example(_ws_for(db_id), db_id, question, sql, tables=tables, difficulty=difficulty)
