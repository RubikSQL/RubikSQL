"""Session-scoped virtual ids for model-facing rules and examples."""
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Literal


KnowledgeKind = Literal["example", "rule"]


@dataclass(frozen=True)
class _Record:
    """One retained knowledge record behind a short session-local id."""

    kind: KnowledgeKind
    db_id: str
    source_id: str
    payload: dict[str, Any]


@dataclass
class _Session:
    """Mutable virtual-id namespace owned by one agent run."""

    next_number: dict[KnowledgeKind, int] = field(
        default_factory=lambda: {"example": 1, "rule": 1},
    )
    source_to_virtual: dict[tuple[KnowledgeKind, str, str], str] = field(
        default_factory=dict,
    )
    virtual_to_record: dict[str, _Record] = field(default_factory=dict)


class KnowledgeSessions:
    """Own short, stable references to knowledge retained during agent sessions.

    Example and rule counters are independent: the first records retained in a session
    become ``E01`` and ``R01``. Re-retaining the same stored record for the same database
    reuses its existing virtual id, while newly seen records consume the next number.
    """

    _PREFIXES: dict[KnowledgeKind, str] = {"example": "E", "rule": "R"}

    def __init__(self) -> None:
        self._lock = RLock()
        self._sessions: dict[str, _Session] = {}

    def retain(
        self,
        session_id: str,
        kind: KnowledgeKind,
        db_id: str,
        source_id: str,
        payload: dict[str, Any],
    ) -> str:
        """Retain a full record and return its stable virtual id for this session.

        Args:
            session_id: Runtime-owned id for the current agent session.
            kind: Knowledge namespace, either ``example`` or ``rule``.
            db_id: Database whose record was retrieved.
            source_id: Persistent storage id, kept private from the model.
            payload: Full record to cache for later virtual-id operations.

        Returns:
            str: A short session-local id such as ``E01`` or ``R03``.

        Raises:
            ValueError: If a required identity component is empty.
        """
        if not session_id:
            raise ValueError("session_id is required for virtual knowledge ids")
        if not db_id:
            raise ValueError("db_id is required for virtual knowledge ids")
        if not source_id:
            raise ValueError("source_id is required for virtual knowledge ids")

        key = (kind, db_id, source_id)
        with self._lock:
            session = self._sessions.setdefault(session_id, _Session())
            virtual_id = session.source_to_virtual.get(key)
            if virtual_id is None:
                number = session.next_number[kind]
                session.next_number[kind] = number + 1
                virtual_id = f"{self._PREFIXES[kind]}{number:02d}"
                session.source_to_virtual[key] = virtual_id
            session.virtual_to_record[virtual_id] = _Record(
                kind=kind,
                db_id=db_id,
                source_id=source_id,
                payload=deepcopy(payload),
            )
            return virtual_id

    def resolve(
        self,
        session_id: str,
        kind: KnowledgeKind,
        db_id: str,
        virtual_id: str,
    ) -> dict[str, Any] | None:
        """Return a retained payload only when session, kind, and database all match."""
        normalized = str(virtual_id or "").strip().upper()
        with self._lock:
            session = self._sessions.get(session_id)
            record = session.virtual_to_record.get(normalized) if session else None
            if record is None or record.kind != kind or record.db_id != db_id:
                return None
            return deepcopy(record.payload)

    def source_id(
        self,
        session_id: str,
        kind: KnowledgeKind,
        db_id: str,
        virtual_id: str,
    ) -> str | None:
        """Resolve a virtual id to its private persistent id.

        This boundary exists for runtime-owned mutations such as ``update_rule``. The
        persistent id never needs to cross the MCP response back into model context.

        Args:
            session_id: Runtime-owned id for the current agent session.
            kind: Expected knowledge namespace.
            db_id: Expected database namespace.
            virtual_id: Model-facing id such as ``R03``.

        Returns:
            str | None: Persistent storage id when every namespace matches.
        """
        normalized = str(virtual_id or "").strip().upper()
        with self._lock:
            session = self._sessions.get(session_id)
            record = session.virtual_to_record.get(normalized) if session else None
            if record is None or record.kind != kind or record.db_id != db_id:
                return None
            return record.source_id

    def close(self, session_id: str) -> None:
        """Discard every virtual id and cached payload owned by a session."""
        if not session_id:
            return
        with self._lock:
            self._sessions.pop(session_id, None)


knowledge_sessions = KnowledgeSessions()
