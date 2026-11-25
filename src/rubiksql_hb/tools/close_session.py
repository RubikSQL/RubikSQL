"""Pinned implementation of close_session."""


def close_session(session_id: str, db_id: str = "") -> dict:
    """Drop this run's intermediates and retained virtual knowledge ids."""
    from ..server import _close_session, knowledge_sessions

    _close_session(session_id)
    knowledge_sessions.close(session_id)
    return {"closed": session_id}
