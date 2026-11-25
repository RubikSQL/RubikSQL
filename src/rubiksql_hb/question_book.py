"""The key proxy's global question-start book: slot reservations and the admission queue of every host.

The per-host slot gate (:class:`.question_gate.SlotGate`) keeps its reservations in the host's SQLite broker, so
admission is first come, first served only among one host's processes. With the shared key proxy every eval process
on every runner sends through one proxy and one key pool, so admission must be global too. This book holds it in the
proxy's memory, on the proxy's event loop (no locks), behind three routes (:mod:`.keyproxy`):

* ``POST /question/reserve`` ``{owner, host, token, need, phase, since, wait, capacity}`` queues one question (or refreshes
  its ticket) and long-polls up to ``wait`` seconds (at most :data:`MAX_POLL_S`) for its admission. The answer is
  ``{"admitted": true, ...}`` once the question holds a reservation of ``need`` slots, else ``{"admitted": false}``
  and the client polls again with the same token, keeping its place. Repeating a reserve of an admitted token answers
  ``admitted`` again, so a lost answer never admits twice.
* ``POST /question/phase`` ``{owner, host, reservations: [[token, slots, phase], ...], capacity}`` sets the slots each
  running question reserves now (``S`` candidates, ``P`` judge, parked slots, ``0`` once final) and is the owner's
  heartbeat. A reservation the book does not know (after a proxy restart) is recreated from it, and the list is the
  owner's whole truth: a reservation of that owner it omits (one admitted by a reserve that arrived after its client
  had given up) is dropped once it is :data:`ORPHAN_GRACE_S` old.
* ``POST /question/release`` ``{owner, tokens}`` drops reservations and tickets: questions that finished or left.

Admission rotates across waiting owners, granting one question per owner per turn.
Within an owner, tickets are ordered by the client's ``since`` (its enqueue time,
so the order survives a proxy restart) and then by arrival. Owners at their host
capacity are skipped without holding back other owners. ``capacity`` is the
proxy's override (``--question-capacity``), else the pool's eligible keys times the live
per-key cap; lowering the cap file throttles every host at once. A host whose processes send the manual
``RUBIKSQL_QUESTION_START_CAPACITY`` (the smallest, when they differ; the difference is reported) reserves at most
that many slots in all, as its per-host gate did; its questions that do not fit wait without holding back other
hosts' questions, so a smoke run with a small capacity never throttles the fleet. Everything is soft state: an owner
silent for :data:`OWNER_TTL_S` (a process that died) loses its
reservations and tickets; a ticket not polled for :data:`TICKET_TTL_S` is dropped. A proxy that starts admits nothing
for ``warmup`` seconds, so the reservations of questions that were running before a restart are republished by their
owners' heartbeats before anyone is admitted against them.
"""
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math
import time
from typing import Any

#: A client's heartbeat (``/question/phase``) period; the owner and ticket lifetimes are multiples of it.
HEARTBEAT_S = 5.0
#: An owner (one eval process) not heard from for this long has exited: its reservations and tickets are dropped.
OWNER_TTL_S = 60.0
#: A ticket whose question has not polled for this long has left.
TICKET_TTL_S = 45.0
#: The longest single ``/question/reserve`` long-poll.
MAX_POLL_S = 30.0
#: Default start-up quiet period: two heartbeats plus margin.
WARMUP_S = 2 * HEARTBEAT_S + 2.0
#: A reservation its owner's heartbeat no longer lists is dropped once it is this old (a fresh admission may not
#: have reached the owner's list yet).
ORPHAN_GRACE_S = 30.0
PHASES = ("candidates", "judge", "done")


@dataclass(eq=False)
class _Ticket:
    owner: str
    need: int
    phase: str
    since: float
    seq: int
    seen: float
    waiters: list[asyncio.Future] = field(default_factory=list)


@dataclass(eq=False)
class _Held:
    owner: str
    slots: int
    phase: str
    seen: float
    #: When the book created it (an admission, or a heartbeat that republished it).
    created: float = 0.0


@dataclass
class _Owner:
    seen: float
    override: int | None
    host: str


class QuestionError(ValueError):
    """A malformed question-book request (the route answers 400)."""


def _owner(value: Any) -> str:
    if not isinstance(value, str) or not 0 < len(value) <= 128:
        raise QuestionError("owner must be a non-empty string of at most 128 characters")
    return value


def _token(value: Any) -> str:
    if not isinstance(value, str) or not 0 < len(value) <= 64:
        raise QuestionError("token must be a non-empty string of at most 64 characters")
    return value


def _count(value: Any, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= 1_000_000:
        raise QuestionError(f"{name} must be an integer of at least {minimum}")
    return value


def _phase(value: Any) -> str:
    if value not in PHASES:
        raise QuestionError(f"phase must be one of {', '.join(PHASES)}")
    return value


def _override(value: Any) -> int | None:
    return None if value is None else _count(value, "capacity", minimum=1)


def _host(value: Any, owner: str) -> str:
    if value is None:
        return owner
    if not isinstance(value, str) or not 0 < len(value) <= 128:
        raise QuestionError("host must be a non-empty string of at most 128 characters")
    return value


def _number(value: Any, name: str) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        raise QuestionError(f"{name} must be a number") from None
    if not math.isfinite(number):
        raise QuestionError(f"{name} must be a finite number")
    return number


class QuestionBook:
    """Global question reservations and the admission queue, owned by one event loop.

    Args:
        capacity: The computed slot capacity now (the pool's eligible keys times the live per-key cap).
        override: A manual fleet capacity configured on the proxy (``--question-capacity``).
        clock: Monotonic seconds (tests pass a fake clock).
        warmup: Seconds after creation during which nothing is admitted.
        owner_ttl: Seconds after which a silent owner's rows are dropped.
        ticket_ttl: Seconds after which an unpolled ticket is dropped.
    """

    def __init__(self, capacity: Callable[[], int], *, override: int | None = None,
                 clock: Callable[[], float] = time.monotonic, warmup: float = WARMUP_S,
                 owner_ttl: float = OWNER_TTL_S, ticket_ttl: float = TICKET_TTL_S) -> None:
        self._capacity = capacity
        self.override = override
        self.clock = clock
        self.ready_at = clock() + warmup
        self.owner_ttl = owner_ttl
        self.ticket_ttl = ticket_ttl
        self.admitted_total = 0
        self.closing = False
        self._seq = 0
        self._owners: dict[str, _Owner] = {}
        self._tickets: dict[str, _Ticket] = {}
        self._held: dict[str, _Held] = {}
        self._turns: deque[str] = deque()
        self._last_owner: str | None = None

    # -- capacity

    def capacity(self) -> tuple[int, str]:
        """``(slots, source)`` of the fleet: the proxy's override, else the keys'."""
        if self.override is not None:
            return self.override, "override"
        return max(0, int(self._capacity())), "keys"

    def host_caps(self) -> dict[str, int]:
        """Each host's manual capacity: the smallest ``RUBIKSQL_QUESTION_START_CAPACITY`` its live owners send."""
        caps: dict[str, int] = {}
        for owner in self._owners.values():
            if owner.override is not None:
                caps[owner.host] = min(caps.get(owner.host, owner.override), owner.override)
        return caps

    def _host_of(self, owner: str) -> str:
        record = self._owners.get(owner)
        return record.host if record is not None else owner

    def reserved(self) -> int:
        """Slots reserved by every live question."""
        return sum(held.slots for held in self._held.values())

    # -- requests

    def _touch(self, owner: str, override: int | None, host: str) -> float:
        now = self.clock()
        self._owners[owner] = _Owner(now, override, host)
        return now

    def _check_owner(self, row: _Ticket | _Held | None, owner: str) -> None:
        if row is not None and row.owner != owner:
            raise QuestionError("token belongs to another owner")

    async def reserve(self, payload: Mapping[str, Any], gone: asyncio.Future | None = None) -> dict[str, Any]:
        """Queue (or refresh) one question and wait up to its ``wait`` seconds for its admission.

        Args:
            payload: ``{owner, host, token, need, phase, since, wait, capacity}``; ``phase`` is the admitted
                question's first phase (``candidates``, or ``judge`` for a judge-only replay); ``host`` defaults to
                the owner.
            gone: Completes when the polling client disconnects; its ticket stays for the next poll.

        Returns:
            ``{"admitted": bool, **summary}``.

        Raises:
            QuestionError: A malformed request.
        """
        owner, token = _owner(payload.get("owner")), _token(payload.get("token"))
        need = _count(payload.get("need"), "need", minimum=1)
        phase = _phase(payload.get("phase", "candidates"))
        wait = min(MAX_POLL_S, max(0.0, _number(payload.get("wait"), "wait")))
        since = _number(payload.get("since"), "since")
        now = self._touch(owner, _override(payload.get("capacity")), _host(payload.get("host"), owner))
        held = self._held.get(token)
        self._check_owner(held, owner)
        if held is not None:
            held.seen = now
            return {"admitted": True, **self.summary()}
        if self.closing:
            return {"admitted": False, "closing": True, **self.summary()}
        ticket = self._tickets.get(token)
        self._check_owner(ticket, owner)
        if ticket is None:
            self._seq += 1
            ticket = self._tickets[token] = _Ticket(owner, need, phase, since, self._seq, now)
        ticket.need, ticket.phase, ticket.seen = need, phase, now
        self.admit()
        if token not in self._held and wait > 0:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            ticket.waiters.append(future)
            try:
                await asyncio.wait({future} if gone is None else {future, gone}, timeout=wait,
                                   return_when=asyncio.FIRST_COMPLETED)
            finally:
                if not future.done():
                    future.cancel()
                if future in ticket.waiters:
                    ticket.waiters.remove(future)
            ticket.seen = self.clock()
        return {"admitted": token in self._held, **self.summary()}

    def phase(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Set the owner's running reservations (its heartbeat); unknown tokens are recreated.

        Args:
            payload: ``{owner, host, reservations: [[token, slots, phase], ...], capacity}``.

        Raises:
            QuestionError: A malformed request.
        """
        owner = _owner(payload.get("owner"))
        rows = payload.get("reservations") or []
        if not isinstance(rows, list):
            raise QuestionError("reservations must be a list")
        parsed = []
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise QuestionError("each reservation is [token, slots, phase]")
            parsed.append((_token(row[0]), _count(row[1], "slots"), _phase(row[2])))
        now = self._touch(owner, _override(payload.get("capacity")), _host(payload.get("host"), owner))
        for token, slots, phase in parsed:
            held = self._held.get(token)
            self._check_owner(held, owner)
            if held is None:
                self._tickets.pop(token, None)
                self._held[token] = _Held(owner, slots, phase, now, created=now)
            else:
                held.slots, held.phase, held.seen = slots, phase, now
        listed = {token for token, _slots, _phase in parsed}
        for token, held in list(self._held.items()):
            if held.owner == owner and token not in listed and now - held.created >= ORPHAN_GRACE_S:
                del self._held[token]   # admitted after its client gave up: nothing would ever release it
        self.admit()
        return self.summary()

    def release(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Drop the owner's listed reservations and tickets.

        Args:
            payload: ``{owner, tokens}``.

        Raises:
            QuestionError: A malformed request.
        """
        owner = _owner(payload.get("owner"))
        tokens = payload.get("tokens") or []
        if not isinstance(tokens, list):
            raise QuestionError("tokens must be a list")
        known = self._owners.get(owner)
        self._touch(owner, known.override if known else None,
                    known.host if known else _host(payload.get("host"), owner))
        for token in map(_token, tokens):
            for table in (self._held, self._tickets):
                row = table.get(token)
                if row is not None and row.owner == owner:
                    del table[token]
                    for future in getattr(row, "waiters", ()):
                        if not future.done():
                            future.set_result(False)
        self.admit()
        return self.summary()

    # -- admission

    def admit(self) -> int:
        """Admit one question per waiting owner in turn; return the admission count.

        A question whose host is at its manual capacity is passed over (with the rest of that host's queue) without
        holding back the questions of other hosts queued behind it.
        """
        if self.closing or not self._tickets or self.clock() < self.ready_at:
            return 0
        total, _source = self.capacity()
        reserved = self.reserved()
        caps = self.host_caps()
        by_host: dict[str, int] = {}
        if caps:
            for held in self._held.values():
                host = self._host_of(held.owner)
                by_host[host] = by_host.get(host, 0) + held.slots
        blocked: set[str] = set()
        admitted = 0
        now = self.clock()
        queues: dict[str, deque[tuple[str, _Ticket]]] = {}
        for token, ticket in sorted(self._tickets.items(), key=lambda item: (item[1].since, item[1].seq)):
            queues.setdefault(ticket.owner, deque()).append((token, ticket))
        self._turns = deque(owner for owner in self._turns if owner in queues)
        self._turns.extend(owner for owner in queues if owner not in self._turns)
        if len(self._turns) > 1 and self._turns[0] == self._last_owner:
            self._turns.rotate(-1)
        skipped = 0
        while self._turns and skipped < len(self._turns):
            owner = self._turns[0]
            token, ticket = queues[owner][0]
            host = self._host_of(ticket.owner)
            if host in blocked or host in caps and by_host.get(host, 0) + ticket.need > caps[host]:
                blocked.add(host)
                self._turns.rotate(-1)
                skipped += 1
                continue
            if total - reserved < ticket.need:
                break
            skipped = 0
            del self._tickets[token]
            self._held[token] = _Held(ticket.owner, ticket.need, ticket.phase, now, created=now)
            queues[owner].popleft()
            self._last_owner = owner
            self._turns.popleft()
            if queues[owner]:
                self._turns.append(owner)
            reserved += ticket.need
            by_host[host] = by_host.get(host, 0) + ticket.need
            admitted += 1
            for future in ticket.waiters:
                if not future.done():
                    future.set_result(True)
        self.admitted_total += admitted
        return admitted

    def expire(self) -> None:
        """Drop the rows of silent owners and unpolled tickets, then admit what fits."""
        now = self.clock()
        gone = {owner for owner, record in self._owners.items() if now - record.seen >= self.owner_ttl}
        for owner in gone:
            del self._owners[owner]
        for token, ticket in list(self._tickets.items()):
            if ticket.owner in gone or (not ticket.waiters and now - ticket.seen >= self.ticket_ttl):
                del self._tickets[token]
        for token, held in list(self._held.items()):
            if held.owner in gone:
                del self._held[token]
        self.admit()

    def close(self) -> None:
        """Admit nothing more; pending long-polls answer ``admitted: false`` at once."""
        self.closing = True
        for ticket in self._tickets.values():
            for future in ticket.waiters:
                if not future.done():
                    future.set_result(False)

    def summary(self) -> dict[str, Any]:
        """Global gate counts, the same names as the per-host gate's cycle where they exist."""
        total, source = self.capacity()
        values: dict[str, set[int]] = {}
        for owner in self._owners.values():
            if owner.override is not None:
                values.setdefault(owner.host, set()).add(owner.override)
        conflicts = sorted({value for found in values.values() if len(found) > 1 for value in found})
        by_phase: dict[str, int] = {}
        for held in self._held.values():
            by_phase[held.phase] = by_phase.get(held.phase, 0) + held.slots
        return {"capacity": total, "capacity_source": source, "reserved": self.reserved(), "running": len(self._held),
                "queued": len(self._tickets), "queued_slots": sum(ticket.need for ticket in self._tickets.values()),
                "admitted_total": self.admitted_total, "owners": len(self._owners),
                "reserved_by_phase": by_phase, "warming": self.clock() < self.ready_at,
                "host_caps": self.host_caps() or None, "capacity_conflict": conflicts or None}
