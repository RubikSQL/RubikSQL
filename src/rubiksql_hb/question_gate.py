"""Hold question starts until the host has key slots for them, so slot waits stay outside question timers.

Every model request on the lent file pool waits for a key slot (``lent_chat.acquire_lease``). That wait used to happen
inside the question: a question started while the keys were saturated spent part of its timed budgets (the candidate
and judge phase budgets, `rubiksql_hb.phase_budget`) waiting for slots, and added no throughput. The gate moves that
wait in front of the question: the question's timers start only when it passes the gate, and the wait is recorded as
``gate_wait_sec``.

**Slots mode** (the default, ``RUBIKSQL_QUESTION_START_GATE_MODE=slots``) admits by reservation. Each candidate seat
and each judge pass is a sequential conversation with at most one model request in flight, so a question needs at most
``S`` request slots in its candidate phase (``S`` = candidate seats; top-ups only replace failed seats) and ``P`` in its
judge phase (``P`` = ``RUBIKSQL_JUDGE_PASSES``; the reviewer runs inside each pass). Every admitted question holds a
reservation in the host broker (:meth:`.key_broker.SqliteKeyBroker.gate_cycle`): ``S`` while it generates candidates,
``P`` from its judge start (:class:`Admission`, wired to :class:`.phase_budget.VotePhases`), and ``0`` once its answer
is final (the evaluator settles it after the last vote, retry or pool-guard fallback). A vote that returns keeps its
slots parked until then, so a retry, a completed-pool tail judge or the single-agent fallback reuses them instead of
taking slots another question was admitted into. A vote still running when its question exits (abandoned after the
question's wall) keeps its slots until it unwinds. A judge-only replay over saved candidates reserves ``P`` from the
start. A new question is admitted only when::

    capacity - sum(reservations) >= its need + the needs of every question queued before it

so admission is first come, first served across the host's processes and databases, and, while capacity is stable,
an admitted question never waits for a key slot. ``capacity`` is one number, recomputed on every cycle: the manual
``RUBIKSQL_QUESTION_START_CAPACITY`` (total request slots) when set, otherwise the eligible keys (neither cooling
after a 429, out of daily quota nor retired) times the per-key host cap (the live cap file or
``RUBIKSQL_KEY_MAX_INFLIGHT``, lowered by the cap controller's durable ceiling). The manual capacity works for any
provider: without the lent file pool it alone turns the gate on, on a host broker named by
``RUBIKSQL_KEY_BROKER_STATE`` (else beside the gateway pool file, else under ``~/.rubiksql-hb``). Every eval process
on a host must get the same value; a process that sees another live process admit against a different one logs it.

One daemon thread per process does all broker I/O, about once a second while the process has questions waiting or
running: it queues new questions, refreshes every reservation and ticket (their heartbeat), applies phase changes and
releases, and admits the queued questions that fit, all in one short transaction. Reservations and tickets of processes
that exited are ignored at once and deleted (:mod:`.process_identity`); rows not refreshed for
:data:`.key_broker.RESERVATION_TTL_S` / :data:`.key_broker.TICKET_TTL_S` are ignored too. When capacity falls below
the reservations (keys cooled or ran out), nothing is admitted until the reservations fall back under it; when it is
below one question's need (for example every key exhausted), questions keep waiting. Both cases log a rate-limited
line, and no answer is ever fabricated. Admission is never forced by waiting time (``RUBIKSQL_QUESTION_START_GATE_MAX_SEC``
applies to backlog mode only): while capacity covers one question, reservations drain as questions finish, so a
queued question cannot starve, and while it does not, an admitted question would only wait inside its timers. A
question also starts at once after the lent pool's window end, when one is configured (the provider route then refuses
as before), and after :data:`FAIL_OPEN_AFTER_S` seconds of consecutive broker failures other than busy locks (logged
once; such questions still join the process's reservations, which are published once the broker recovers). A cycle
right after a wall-clock jump admits nothing, because other processes' heartbeats may look stale for a moment.
Raising a running question's reservation (a judge phase needing more slots than its seats, a retry after a vote
failed in its judge phase) never blocks it: the excess shows as missing headroom and holds new questions back.

**With the key proxy** (``RUBIKSQL_KEYPROXY_URL``, :mod:`.keyproxy`) the same reservations live in the proxy instead
(:class:`ProxySlotGate`, :mod:`.question_book`), so admission is first come, first served across every host that
sends through it, against the pool's live capacity (available keys times the live per-key cap, or the proxy's
``--question-capacity``). The manual capacity then caps the slots its host's questions reserve, as the per-host gate
did, without throttling other hosts. A launch makes one authenticated call first, so a missing proxy or a stale token
copy refuses it; a token the proxy later refuses fails the waiting questions instead of holding them forever. Each
waiting question long-polls ``/question/reserve`` from its own thread; one thread per process
publishes reservation changes at once and every :data:`.question_book.HEARTBEAT_S` seconds (``/question/phase``,
which also rebuilds the book after a proxy restart) and returns finished questions' slots (``/question/release``).
While the proxy cannot be reached, questions keep waiting (logged, rate-limited) instead of starting unmetered,
because their model requests could not be sent either; running questions keep their reservations and republish them
once it answers. The proxy gate has no backlog mode.

**Backlog mode** (``RUBIKSQL_QUESTION_START_GATE_MODE=backlog``, the earlier rule) reads the host backlog
(:meth:`.key_broker.SqliteKeyBroker.backlog`): the admissions waiting on live processes (``HostCounter`` kind
``waiting``) and the leases in flight. The gate closes while::

    waiting > ratio * capacity                       (RUBIKSQL_QUESTION_START_GATE, default 0.5)
    or inflight >= inflight_ratio * capacity         (RUBIKSQL_QUESTION_START_GATE_INFLIGHT, default off)

where ``capacity = min(host cap, controller ceiling) * usable keys``, and reopens only once ``waiting`` falls to
:data:`REOPEN_FRACTION` of that threshold. Each broker read (one per second per process) grants this process a share
of the host's admission budget::

    share = (ratio * capacity - waiting) / requests per question / gate processes
    grant = max(1, share) - starts this process made in the last START_LAG_S seconds

(``RUBIKSQL_QUESTION_START_GATE_REQS``, default 3 requests per question). Gate processes are the live processes with
questions held at or running past their gate (``HostCounter`` kind ``gate_active``) or holding questions, whichever
count is larger. A process with no running question always admits one; the gate also admits after the window end,
when the backlog cannot be read, and after ``RUBIKSQL_QUESTION_START_GATE_MAX_SEC`` (default 1200 s) of waiting.

In both modes :meth:`stop` releases every waiter with :class:`QuestionStartStopped`; with
``RUBIKSQL_QUESTION_START_GATE=0``/``off``, or without the lent file pool and (in slots mode) without the manual
capacity, the gate is a no-op. In backlog mode the manual capacity replaces the computed one too. Questions held at
the gate, and questions held or running, are counted per process and published to the broker (``HostCounter`` kinds
``gate_held`` and ``gate_active``); the cap controller's host helper (``scripts/g28_capbal_host.py``) reports the held
total as ``gate_held``, the slots queued at the slot gate as ``gate_held_slots`` and the slots reserved by running
questions as ``gate_reserved``.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import random
import socket
import threading
import time
from typing import Any
from uuid import uuid4

GATE_ENV = "RUBIKSQL_QUESTION_START_GATE"
MODE_ENV = "RUBIKSQL_QUESTION_START_GATE_MODE"
CAPACITY_ENV = "RUBIKSQL_QUESTION_START_CAPACITY"
MODES = ("slots", "backlog")
DEFAULT_MODE = "slots"
INFLIGHT_ENV = "RUBIKSQL_QUESTION_START_GATE_INFLIGHT"
MAX_WAIT_ENV = "RUBIKSQL_QUESTION_START_GATE_MAX_SEC"
REQS_ENV = "RUBIKSQL_QUESTION_START_GATE_REQS"
DEFAULT_RATIO = 0.5
DEFAULT_MAX_WAIT_S = 1200.0
#: Key-slot requests one newly started question is assumed to add (candidates' first turns, judge later).
DEFAULT_REQS_PER_QUESTION = 3.0
#: A closed gate reopens only below this fraction of its closing threshold.
REOPEN_FRACTION = 0.8
#: Broker reads are shared by every waiter of a process for this long; each read grants a fresh start budget.
READ_INTERVAL_S = 1.0
#: A start counts against this process's next grants for this long: its first requests reach the published
#: ``waiting`` count only after the question's setup and the counter's publication tick.
START_LAG_S = 10.0
#: Slots mode: the gate thread's cycle period while the process has questions waiting or running.
CYCLE_INTERVAL_S = 1.0
#: Slots mode: a waiter re-checks its admission (and the stop flag) at least this often.
WAITER_POLL_S = 0.5
#: Slots mode: rate limit of the capacity warnings and the periodic host sample line.
LOG_INTERVAL_S = 60.0
#: Slots mode: consecutive broker failures (not busy locks) for this long start the waiting questions unmetered.
FAIL_OPEN_AFTER_S = 60.0
#: Slots mode: a finished question's reservation waits at most this long for a vote attempt still running.
DRAIN_LIMIT_S = 900.0
#: Slots mode: a wall-clock move this much larger or smaller than the monotonic one between cycles skips admission.
CLOCK_JUMP_S = 10.0
#: Proxy gate: the longest single ``/question/reserve`` long-poll (also how soon a stopped run releases its waiters).
PROXY_POLL_S = 10.0
_OFF = frozenset({"0", "off", "false", "no"})
_PHASES = ("candidates", "judge", "done")
_LOGGER = logging.getLogger(__name__)


class QuestionStartStopped(RuntimeError):
    """The run stopped while this question waited at the start gate; it never started."""


def _positive(name: str, raw: str | None, default: float | None, *, allow_off: bool) -> float | None:
    """Parse one gate variable: a finite positive number, ``off`` words when allowed, or the default when unset."""
    value = (raw or "").strip().lower()
    if not value:
        return default
    if allow_off and value in _OFF:
        return None
    try:
        number = float(value)
    except ValueError:
        number = float("nan")
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"{name} must be a finite positive number{' or off' if allow_off else ''}: {raw!r}")
    return number


@dataclass(frozen=True)
class QuestionDemand:
    """The request slots one question can have in flight at once.

    Attributes:
        candidates: Slots of its candidate phase (``S``, the candidate seats; 1 for a single-candidate question).
        judge: Slots of its judge phase (``P``, the judge passes; 0 without a judge).
        judge_only: The question judges saved candidates and has no candidate phase.
    """

    candidates: int = 1
    judge: int = 0
    judge_only: bool = False

    def __post_init__(self) -> None:
        if self.candidates < 1 or self.judge < 0 or self.need < 1:
            raise ValueError(f"invalid question demand: {self}")

    @property
    def need(self) -> int:
        """Slots reserved at admission: the judge phase's for a judge-only replay, else the candidate phase's."""
        return self.judge if self.judge_only else self.candidates

    @property
    def phase(self) -> str:
        """The phase a newly admitted question is in."""
        return "judge" if self.judge_only else "candidates"

    def slots(self, phase: str) -> int:
        """Slots of ``phase`` (``candidates``, ``judge`` or ``done``)."""
        if phase not in _PHASES:
            raise ValueError(f"unknown question phase: {phase!r}")
        return {"candidates": self.candidates, "judge": self.judge, "done": 0}[phase]


class Admission:
    """One question that passed the gate: its wait and, in slots mode, its live slot reservation.

    The reservation is a list of entries, one per vote attempt. It starts as one *parked* entry of the demand's
    :attr:`QuestionDemand.need`. Each vote attempt gets a phase listener from :meth:`attempt` and claims a parked
    entry when there is one (else it adds an entry), so overlapping attempts (a stopped vote still unwinding while a
    tail judge runs) are all counted. A running attempt's entry follows its phases (``candidates`` -> ``S``,
    ``judge`` -> ``P``); when the attempt reports ``done`` its entry is parked at its last value instead of dropping
    to 0, because the evaluator may still retry, judge the completed pool or run the single-agent pool guard: the
    next attempt reuses those slots without asking the gate again. :meth:`settle` drops the parked slots once the
    question's answer is final, and :meth:`close` (at question exit) drops them for good; an attempt still running
    after :meth:`close` keeps its slots until it reports ``done``. Changes are made in memory only and published by
    the gate's thread, so a listener may run on an event loop.

    Args:
        wait_sec: Seconds the question waited at the gate.
        demand: The question's demand; ``None`` for a gate that keeps no reservation.
        record: Admission telemetry stored with the result row (mode, need, capacity and reservations seen).
        on_change: Called after every reservation change (the gate wakes its thread).
    """

    def __init__(self, wait_sec: float = 0.0, demand: QuestionDemand | None = None, *,
                 record: Mapping[str, Any] | None = None, on_change: Callable[[], Any] | None = None) -> None:
        self.wait_sec = wait_sec
        self.demand = demand
        self.record = dict(record or {})
        self._on_change = on_change
        self._lock = threading.Lock()
        self._slots = [demand.need if demand is not None else 0]
        self._phases = [demand.phase if demand is not None else "candidates"]
        self._live = [False]
        self._closed = False

    @property
    def slots(self) -> int:
        """Slots currently reserved: running attempts' phases plus the parked slots."""
        with self._lock:
            return sum(self._slots)

    @property
    def phase(self) -> str:
        """The latest attempt's phase (``done`` once it returned)."""
        with self._lock:
            return self._phases[-1]

    @property
    def live(self) -> bool:
        """Whether some vote attempt is still running (claimed and not yet ``done``)."""
        with self._lock:
            return any(self._live)

    def _changed(self) -> None:
        if self._on_change is not None:
            self._on_change()

    def attempt(self) -> Callable[[str], None]:
        """A phase listener for one vote attempt, e.g. :class:`.phase_budget.VotePhases` ``listener``.

        The listener ignores every report after the attempt's ``done``, so a late or out-of-order report from an
        abandoned vote never changes the entry a later attempt took over.
        """
        with self._lock:
            index = next((i for i, live in enumerate(self._live) if not live and not self._closed), None)
            if index is None:
                self._slots.append(0)
                self._phases.append("candidates")
                self._live.append(True)
                index = len(self._slots) - 1
            else:
                self._live[index] = True
        finished = False

        def listener(phase: str) -> None:
            nonlocal finished
            if self.demand is None:
                return
            slots = self.demand.slots(phase)
            with self._lock:
                if finished:
                    return
                if phase != "done":
                    self._slots[index], self._phases[index] = slots, phase
                else:
                    finished = True
                    self._live[index], self._phases[index] = False, "done"
                    parked = [i for i, live in enumerate(self._live) if not live and i != index]
                    if self._closed:
                        self._slots[index] = 0
                    elif parked:  # keep one parked entry: the largest of the attempts that returned
                        keep = parked[0]
                        self._slots[keep] = max(self._slots[keep], self._slots[index])
                        self._slots[index] = 0
            self._changed()

        return listener

    def settle(self) -> None:
        """Drop the parked slots: the question's answer is final and no further attempt follows."""
        with self._lock:
            for i, live in enumerate(self._live):
                if not live:
                    self._slots[i], self._phases[i] = 0, "done"
        self._changed()

    def close(self) -> bool:
        """Settle for good at question exit; returns whether an attempt is still running (its slots stay)."""
        with self._lock:
            self._closed = True
        self.settle()
        return self.live


@dataclass
class _Waiter:
    """A question waiting at the slots gate; the gate's thread fills ``seq`` and ``admission``."""

    demand: QuestionDemand
    event: threading.Event
    seq: int | None = None
    admission: Admission | None = None


class SlotGate:
    """Admit questions by host slot reservations (slots mode; see the module documentation).

    Args:
        cycle: One broker cycle, :meth:`.key_broker.SqliteKeyBroker.gate_cycle` bound to the pool's files, live cap
            and capacity override; returns ``None`` when the broker is busy and may raise when it fails.
        cutoff: Wall-clock time after which questions start at once (the lent pool's window end; inf for none).
        wall: Wall clock compared with ``cutoff`` and used to detect a wall-clock jump between cycles.
        clock: Monotonic clock for the gate waits.
        held: Counter of questions held at the gate (``change(+1/-1)``), e.g. the broker's ``gate_held`` count.
        active: Counter of questions held at or running past the gate (``change(+1/-1)``, optional ``publish()``
            called when this process gets its first question), e.g. the broker's ``gate_active`` count; it keeps a
            healthy host (nothing held, nothing waiting) visible to the cap controller as a live broker.
        owner: This process's owner id; by default ``<pid>-<birth>-<nonce>`` (:mod:`.process_identity`).
        autostart: Start the broker thread on the first question; tests pass ``False`` and call :meth:`step`.
        interval: Seconds between the thread's cycles.
        poll: Seconds between a waiter's checks of its admission and the stop flag.
        fail_open_after: Seconds of consecutive broker failures (not busy locks) after which waiting questions
            start unmetered.
        drain_limit: Seconds a finished question's reservation is kept for a vote attempt still running.
    """

    def __init__(self, cycle: Callable[[Any], Any], *, cutoff: float = float("inf"),
                 wall: Callable[[], float] = time.time, clock: Callable[[], float] = time.monotonic,
                 held: Any = None, active: Any = None, owner: str | None = None, autostart: bool = True,
                 interval: float = CYCLE_INTERVAL_S, poll: float = WAITER_POLL_S,
                 fail_open_after: float = FAIL_OPEN_AFTER_S, drain_limit: float = DRAIN_LIMIT_S) -> None:
        from .process_identity import owner_tag

        self._cycle = cycle
        self.cutoff = cutoff
        self._wall = wall
        self._clock = clock
        self.held = held
        self.active = active
        self.owner = owner or f"{owner_tag()}-{uuid4().hex[:12]}"
        self._autostart = autostart
        self._interval = interval
        self._poll = poll
        self._fail_open_after = fail_open_after
        self._drain_limit = drain_limit
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._waiters: dict[str, _Waiter] = {}
        self._running: dict[str, Admission] = {}
        self._draining: dict[str, float] = {}
        self._leave: list[int] = []
        self._release: list[str] = []
        self._thread: threading.Thread | None = None
        self._entered = 0
        self._failing_since: float | None = None
        self._failure_warned = False
        self._last_cycle: tuple[float, float] | None = None
        self._logged: dict[str, float] = {}
        self.last: Any = None

    @property
    def running(self) -> int:
        """Questions of this process holding a reservation (finished questions still draining a vote included)."""
        with self._lock:
            return len(self._running)

    @property
    def waiting(self) -> int:
        """Questions of this process waiting at the gate."""
        with self._lock:
            return len(self._waiters)

    def stop(self) -> None:
        """Release every waiter with :class:`QuestionStartStopped`; later questions are refused too."""
        self._stop.set()
        self._wake.set()

    def _poke(self) -> None:
        self._wake.set()

    def _ensure_thread(self) -> None:
        if not self._autostart:
            return
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="question-gate", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while True:
            self._wake.wait(self._interval)
            self._wake.clear()
            try:
                self.step()
            except Exception as error:  # noqa: BLE001 - the gate thread must survive; step already fails open
                _LOGGER.warning("[question-gate] Gate cycle failed (%s).", type(error).__name__)

    def _clock_jumped(self) -> bool:
        """Whether the wall clock moved more than :data:`CLOCK_JUMP_S` against the monotonic clock since the last
        cycle (an NTP step, a resumed VM): other processes' rows may look stale until they refresh them."""
        wall, mono = self._wall(), self._clock()
        last, self._last_cycle = self._last_cycle, (wall, mono)
        return last is not None and abs((wall - last[0]) - (mono - last[1])) > CLOCK_JUMP_S

    def step(self) -> Any:
        """Run one broker cycle: queue, refresh, release and admit. Returns the cycle, or ``None`` when skipped."""
        from .key_broker import GateRequest

        with self._lock:
            now = self._clock()
            for token, since in list(self._draining.items()):
                admission = self._running.get(token)
                if admission is None or not admission.live or now - since >= self._drain_limit:
                    self._draining.pop(token)
                    if self._running.pop(token, None) is not None:
                        self._release.append(token)
            if not (self._waiters or self._running or self._leave or self._release):
                return None
            jumped = self._clock_jumped()
            request = GateRequest(
                owner=self.owner,
                enqueue=tuple((token, w.demand.need, w.demand.phase) for token, w in self._waiters.items()
                              if w.seq is None),
                waiting=tuple((token, w.seq, w.demand.need, w.demand.phase) for token, w in self._waiters.items()
                              if w.seq is not None),
                leave=tuple(self._leave),
                reservations=tuple((token, admission.slots, admission.phase)
                                   for token, admission in self._running.items()),
                release=tuple(self._release),
                admit=not jumped)
        if jumped and self._due("jump"):
            _LOGGER.warning("[question-gate] The wall clock jumped; no question starts in this cycle while the "
                            "host's other processes refresh their reservations.")
        try:
            result = self._cycle(request)
        except Exception as error:  # noqa: BLE001 - a broker failing for long must never hold questions
            self._failed(error)
            return None
        if result is None:
            return None  # busy writer: nothing changed, the next cycle resends everything
        with self._lock:
            self._failing_since = None
            del self._leave[:len(request.leave)]
            del self._release[:len(request.release)]
            for token, seq in result.enqueued.items():
                waiter = self._waiters.get(token)
                if waiter is not None:
                    waiter.seq = seq
                elif token not in result.admitted:
                    self._leave.append(seq)  # its question left while the ticket was being written
            for token in result.admitted:
                waiter = self._waiters.pop(token, None)
                if waiter is None:
                    self._release.append(token)  # its question left during the cycle that admitted it
                    continue
                admission = Admission(0.0, waiter.demand, on_change=self._poke, record={
                    "mode": "slots", "need": waiter.demand.need, "candidate_slots": waiter.demand.candidates,
                    "judge_slots": waiter.demand.judge, "judge_only": waiter.demand.judge_only,
                    "capacity": result.capacity, "capacity_source": result.capacity_source,
                    "reserved": result.reserved, "metered": True})
                self._running[token] = admission
                waiter.admission = admission
                waiter.event.set()
            needs = [w.demand.need for w in self._waiters.values()]
            running, waiting = len(self._running), len(self._waiters)
            self.last = result
        self._report(result, needs, running, waiting)
        return result

    def _failed(self, error: Exception) -> None:
        """Count one broker failure; after :attr:`_fail_open_after` seconds of them, start the waiters unmetered.

        A short failure (a disk hiccup, a cap-file read racing the controller's replace) only delays admission,
        like a busy lock. Questions started unmetered still join this process's reservations, so once the broker
        recovers they are published and hold new admissions back until they finish.
        """
        now = self._clock()
        with self._lock:
            if self._failing_since is None:
                self._failing_since = now
            if now - self._failing_since < self._fail_open_after:
                return
            waiters = list(self._waiters.items())
            self._waiters.clear()
            self._leave.extend(w.seq for _token, w in waiters if w.seq is not None)
            for token, waiter in waiters:
                waiter.admission = Admission(0.0, waiter.demand, on_change=self._poke, record={
                    "mode": "slots", "metered": False, "reason": "broker_unusable"})
                self._running[token] = waiter.admission
            warn, self._failure_warned = not self._failure_warned, True
        if warn:
            _LOGGER.warning("[question-gate] Broker unusable for the slot gate for %.0f s (%s); questions start "
                            "unmetered.", self._fail_open_after, type(error).__name__)
        for _token, waiter in waiters:
            waiter.event.set()

    def _due(self, name: str) -> bool:
        now = self._clock()
        if now - self._logged.get(name, float("-inf")) < LOG_INTERVAL_S:
            return False
        self._logged[name] = now
        return True

    def _report(self, result: Any, needs: list[int], running: int, waiting: int) -> None:
        """Rate-limited capacity warnings and the periodic host sample line."""
        self._failure_warned = False
        conflict = getattr(result, "capacity_conflict", None)
        if conflict and self._due("conflict"):
            _LOGGER.warning("[question-gate] Another live process on this host admits against a different capacity "
                            "(%s %s, here %s %s); give every eval process the same %s.", conflict[1], conflict[0],
                            result.capacity_source, result.capacity, CAPACITY_ENV)
        if needs and result.capacity < min(needs) and self._due("short"):
            _LOGGER.warning("[question-gate] Capacity %d request slots is below the %d one question needs; questions "
                            "wait at the gate (no answer fabricated). Keys cooling, exhausted or retired, a low key "
                            "cap, or %s.", result.capacity, min(needs), CAPACITY_ENV)
        elif result.reserved > result.capacity and self._due("over"):
            _LOGGER.warning("[question-gate] Capacity %d request slots fell below the %d reserved by running "
                            "questions; no question starts until reservations drain.", result.capacity, result.reserved)
        if self._due("sample"):
            print(f"[question-gate] host capacity={result.capacity} ({result.capacity_source}) "
                  f"reserved={result.reserved} running={result.running} queued={result.queued} "
                  f"queued_slots={result.queued_slots} inflight={result.inflight} waiting={result.waiting} "
                  f"admitted={result.admitted_total}; process running={running} waiting={waiting}", flush=True)

    def _enter(self) -> None:
        """Count a question as active; this process's first one is published at once."""
        with self._lock:
            self._entered += 1
            first = self._entered == 1
        if self.active is not None:
            self.active.change(+1)
            publish = getattr(self.active, "publish", None)
            if first and publish is not None:
                publish()

    def _exit(self) -> None:
        with self._lock:
            self._entered -= 1
        if self.active is not None:
            self.active.change(-1)

    @contextmanager
    def question(self, demand: QuestionDemand | None = None) -> Iterator[Admission]:
        """Wait until the host has slots for this question, then hold its reservation until it finishes.

        A vote attempt still running when the question exits (abandoned after the question's wall) keeps its
        reservation until it reports ``done``, for at most ``drain_limit`` seconds.

        Args:
            demand: The question's slot demand; one slot by default.

        Yields:
            The admission: its gate wait and its reservation, whose :meth:`Admission.attempt` listeners follow
            the vote's phases.

        Raises:
            QuestionStartStopped: :meth:`stop` was called before the question started.
        """
        demand = demand or QuestionDemand()
        started = self._clock()
        if self._stop.is_set():
            raise QuestionStartStopped("the run stopped before this question started")
        self._enter()
        try:
            if self._wall() >= self.cutoff:
                yield Admission(0.0, demand, record={"mode": "slots", "metered": False, "reason": "window_end"})
                return
            token = uuid4().hex
            waiter = _Waiter(demand, threading.Event())
            with self._lock:
                self._waiters[token] = waiter
            if self.held is not None:
                self.held.change(+1)
            self._ensure_thread()
            self._wake.set()
            admitted = False
            try:
                while not waiter.event.wait(self._poll):
                    if self._stop.is_set():
                        raise QuestionStartStopped("the run stopped before this question started")
                    if self._wall() >= self.cutoff:
                        break
                admitted = True
            finally:
                if self.held is not None:
                    self.held.change(-1)
                with self._lock:
                    if self._waiters.pop(token, None) is not None:
                        # Still queued: it leaves (stopped) or starts unmetered (window end).
                        if waiter.seq is not None:
                            self._leave.append(waiter.seq)
                        if admitted and waiter.admission is None:
                            waiter.admission = Admission(0.0, demand, record={"mode": "slots", "metered": False,
                                                                              "reason": "window_end"})
                    elif not admitted and self._running.pop(token, None) is not None:
                        self._release.append(token)  # admitted just as the run stopped: hand the slots back
                self._wake.set()
            admission = waiter.admission
            admission.wait_sec = self._clock() - started
            try:
                yield admission
            finally:
                live = admission.close()
                with self._lock:
                    if token in self._running:
                        if live:
                            self._draining[token] = self._clock()  # step releases it once the vote reports done
                        else:
                            del self._running[token]
                            self._release.append(token)
                self._wake.set()
        finally:
            self._exit()


class ProxySlotGate:
    """Admit questions by global slot reservations held in the key proxy (see the module documentation).

    Args:
        endpoint: The key proxy (:class:`.keyproxy.Endpoint`): its ``call(route, payload, timeout=...)``.
        capacity: The manual capacity this process sends (``RUBIKSQL_QUESTION_START_CAPACITY``), or ``None``.
        cutoff: Wall-clock time after which questions start at once (the lent pool's window end; inf for none).
        wall: Wall clock compared with ``cutoff``; also each question's enqueue time, its place in the global queue.
        clock: Monotonic clock for the gate waits.
        owner: This process's owner id; by default ``<pid>-<birth>-<nonce>`` (:mod:`.process_identity`).
        autostart: Start the publisher thread on the first question; tests pass ``False`` and call :meth:`publish`.
        heartbeat: Seconds between unprompted publications (the owner's heartbeat).
        poll: Seconds one ``/question/reserve`` long-poll may last.
        drain_limit: Seconds a finished question's reservation is kept for a vote attempt still running.
    """

    def __init__(self, endpoint: Any, *, capacity: int | None = None, cutoff: float = float("inf"),
                 wall: Callable[[], float] = time.time, clock: Callable[[], float] = time.monotonic,
                 owner: str | None = None, autostart: bool = True, heartbeat: float | None = None,
                 poll: float = PROXY_POLL_S, drain_limit: float = DRAIN_LIMIT_S) -> None:
        from .process_identity import owner_tag
        from .question_book import HEARTBEAT_S

        self.endpoint = endpoint
        self.capacity = capacity
        self.cutoff = cutoff
        self._wall = wall
        self._clock = clock
        self.owner = owner or f"{owner_tag()}-{uuid4().hex[:12]}"
        self._autostart = autostart
        self._heartbeat = HEARTBEAT_S if heartbeat is None else heartbeat
        self._poll = poll
        self._drain_limit = drain_limit
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._waiting: dict[str, QuestionDemand] = {}
        self._running: dict[str, Admission] = {}
        self._draining: dict[str, float] = {}
        self._release: list[str] = []
        self._thread: threading.Thread | None = None
        self._logged: dict[str, float] = {}
        self.host = socket.gethostname()[:128] or "unknown-host"
        self.last: dict[str, Any] | None = None

    @property
    def running(self) -> int:
        """Questions of this process holding a reservation (finished questions still draining a vote included)."""
        with self._lock:
            return len(self._running)

    @property
    def waiting(self) -> int:
        """Questions of this process waiting at the gate."""
        with self._lock:
            return len(self._waiting)

    def stop(self) -> None:
        """Release every waiter with :class:`QuestionStartStopped` (within one long-poll); later questions too."""
        self._stop.set()
        self._wake.set()

    def _poke(self) -> None:
        self._wake.set()

    def _due(self, name: str) -> bool:
        now = self._clock()
        if now - self._logged.get(name, float("-inf")) < LOG_INTERVAL_S:
            return False
        self._logged[name] = now
        return True

    def _unreachable(self, error: Exception) -> None:
        if isinstance(error, RuntimeError):   # the proxy refused the token or the request: retrying cannot help
            if self._due("refused"):
                _LOGGER.error("[question-gate] The key proxy refused this process (%s); no question can start.",
                              error)
            return
        if self._due("unreachable"):
            _LOGGER.warning("[question-gate] Key proxy unreachable for the question gate (%s); questions wait, "
                            "running questions keep their reservations.", type(error).__name__)

    def _ensure_thread(self) -> None:
        if not self._autostart:
            return
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="question-gate-proxy", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while True:
            self._wake.wait(self._heartbeat)
            self._wake.clear()
            try:
                self.publish()
            except Exception as error:  # noqa: BLE001 - the publisher must survive; the next cycle resends all
                self._unreachable(error)

    def publish(self) -> dict[str, Any] | None:
        """Return finished questions' slots, then publish every running reservation (the heartbeat).

        Returns:
            The proxy's gate summary, or ``None`` when this process holds nothing.

        Raises:
            OSError, ValueError: The proxy could not be reached; everything is resent next time.
        """
        with self._lock:
            now = self._clock()
            for token, since in list(self._draining.items()):
                admission = self._running.get(token)
                if admission is None or not admission.live or now - since >= self._drain_limit:
                    self._draining.pop(token)
                    if self._running.pop(token, None) is not None:
                        self._release.append(token)
            release = list(self._release)
        if release:
            self.endpoint.call("/question/release", {"owner": self.owner, "host": self.host, "tokens": release},
                               timeout=10.0)
            with self._lock:
                del self._release[:len(release)]
        with self._lock:
            rows = [[token, admission.slots, admission.phase] for token, admission in self._running.items()]
        if not rows and not release:
            return None
        result = self.endpoint.call("/question/phase", {"owner": self.owner, "host": self.host, "reservations": rows,
                                                        "capacity": self.capacity}, timeout=10.0)
        self.last = result
        self._report(result)
        return result

    def _report(self, result: Mapping[str, Any]) -> None:
        """Rate-limited capacity warnings and the periodic sample line."""
        if result.get("capacity_conflict") and self._due("conflict"):
            _LOGGER.warning("[question-gate] Processes of one host send different %s values %s; the proxy caps "
                            "that host at the smallest.", CAPACITY_ENV, result["capacity_conflict"])
        if result.get("reserved", 0) > result.get("capacity", 0) and self._due("over"):
            _LOGGER.warning("[question-gate] Capacity %s request slots fell below the %s reserved by running "
                            "questions; no question starts until reservations drain.", result.get("capacity"),
                            result.get("reserved"))
        if self._due("sample"):
            print(f"[question-gate] proxy capacity={result.get('capacity')} ({result.get('capacity_source')}) "
                  f"reserved={result.get('reserved')} running={result.get('running')} queued={result.get('queued')} "
                  f"queued_slots={result.get('queued_slots')} admitted={result.get('admitted_total')} "
                  f"owners={result.get('owners')}; process running={self.running} waiting={self.waiting}", flush=True)

    def _reserve(self, token: str, demand: QuestionDemand, since: float) -> dict[str, Any] | None:
        """One long-poll for this question's admission; ``None`` when the proxy could not be reached.

        Raises:
            GatewayConfigurationError: The proxy refused this process's token or request; the question fails
                instead of waiting forever.
        """
        try:
            return self.endpoint.call("/question/reserve", {
                "owner": self.owner, "host": self.host, "token": token, "need": demand.need, "phase": demand.phase,
                "since": since, "wait": self._poll, "capacity": self.capacity}, timeout=self._poll + 15.0)
        except (OSError, ValueError) as error:
            self._unreachable(error)
            return None
        except RuntimeError as error:
            self._unreachable(error)
            raise

    @contextmanager
    def question(self, demand: QuestionDemand | None = None) -> Iterator[Admission]:
        """Wait until the fleet has slots for this question, then hold its reservation until it finishes.

        Args:
            demand: The question's slot demand; one slot by default.

        Yields:
            The admission (see :meth:`SlotGate.question`).

        Raises:
            QuestionStartStopped: :meth:`stop` was called before the question started.
        """
        demand = demand or QuestionDemand()
        started = self._clock()
        if self._stop.is_set():
            raise QuestionStartStopped("the run stopped before this question started")
        if self._wall() >= self.cutoff:
            yield Admission(0.0, demand, record={"mode": "proxy", "metered": False, "reason": "window_end"})
            return
        token, since = uuid4().hex, self._wall()
        with self._lock:
            self._waiting[token] = demand
        self._ensure_thread()
        admission: Admission | None = None
        failures = 0
        try:
            while admission is None:
                if self._stop.is_set():
                    raise QuestionStartStopped("the run stopped before this question started")
                if self._wall() >= self.cutoff:
                    admission = Admission(0.0, demand, record={"mode": "proxy", "metered": False,
                                                                "reason": "window_end"})
                    break
                answer = self._reserve(token, demand, since)
                if answer is None:
                    failures += 1
                    self._stop.wait(min(10.0, 0.5 * 2 ** min(failures, 5)) * random.uniform(1.0, 1.25))
                    continue
                failures = 0
                if answer.get("admitted"):
                    admission = Admission(0.0, demand, on_change=self._poke, record={
                        "mode": "proxy", "need": demand.need, "candidate_slots": demand.candidates,
                        "judge_slots": demand.judge, "judge_only": demand.judge_only,
                        "capacity": answer.get("capacity"), "capacity_source": answer.get("capacity_source"),
                        "reserved": answer.get("reserved"), "metered": True})
                    with self._lock:
                        self._running[token] = admission
                elif answer.get("closing"):
                    self._stop.wait(min(5.0, self._poll))   # a proxy shutting down admits nothing; poll its successor
        finally:
            with self._lock:
                self._waiting.pop(token, None)
                if token not in self._running:
                    self._release.append(token)   # stopped or unmetered: drop the ticket
            self._wake.set()
        admission.wait_sec = self._clock() - started
        try:
            yield admission
        finally:
            live = admission.close()
            with self._lock:
                if token in self._running:
                    if live:
                        self._draining[token] = self._clock()   # publish releases it once the vote reports done
                    else:
                        del self._running[token]
                        self._release.append(token)
            self._wake.set()


class QuestionStartGate:
    """Admit question starts against a host key-slot backlog (backlog mode).

    Args:
        backlog: Returns ``waiting``, ``inflight``, ``capacity`` and optionally ``holders`` (processes holding
            questions) and ``gates`` (processes with questions held or running) for the host; may raise when
            unreadable.
        ratio: Close while waiting admissions exceed ``ratio * capacity``.
        inflight_ratio: Also close while in-flight leases reach ``inflight_ratio * capacity``; ``None`` disables.
        max_wait: Admit after this many seconds of waiting, whatever the backlog.
        cutoff: Wall-clock time after which the gate admits at once (the lent pool's window end; inf for none).
        reqs_per_question: Requests one started question is assumed to add, for the per-read start budget.
        clock: Monotonic clock; tests pass a controllable one.
        wall: Wall clock compared with ``cutoff``.
        sleep: Wait used between polls; receives the stop event and the seconds to wait.
        held: Counter of questions held at the gate (``change(+1/-1)``), e.g. the broker's ``gate_held`` count.
        active: Counter of questions held at or running past the gate (``change(+1/-1)``, optional ``publish()``
            called when this process gets its first question), e.g. the broker's ``gate_active`` count.
    """

    def __init__(self, backlog: Callable[[], Mapping[str, float]], *, ratio: float = DEFAULT_RATIO,
                 inflight_ratio: float | None = None, max_wait: float = DEFAULT_MAX_WAIT_S,
                 cutoff: float = float("inf"), reqs_per_question: float = DEFAULT_REQS_PER_QUESTION,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 sleep: Callable[[threading.Event, float], Any] | None = None, held: Any = None,
                 active: Any = None) -> None:
        self._backlog = backlog
        self.ratio = ratio
        self.inflight_ratio = inflight_ratio
        self.max_wait = max_wait
        self.cutoff = cutoff
        self.reqs_per_question = reqs_per_question
        self._clock = clock
        self._wall = wall
        self._sleep = sleep or (lambda event, seconds: event.wait(seconds))
        self.held = held
        self.active = active
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._running = 0
        self._active = 0
        self._starts: deque[float] = deque()
        self._read_at = float("-inf")
        self._closed = False
        self._tickets: float = 0
        self._warned = False

    @property
    def running(self) -> int:
        """Questions of this process that passed the gate and have not finished."""
        with self._lock:
            return self._running

    def stop(self) -> None:
        """Release every waiter with :class:`QuestionStartStopped`; later questions are refused too."""
        self._stop.set()

    def _read(self) -> None:
        """One broker read: update the open/closed state (with hysteresis) and grant this read's start budget."""
        try:
            state = self._backlog()
            capacity = max(0.0, float(state["capacity"]))
            threshold = self.ratio * capacity
            waiting = float(state["waiting"])
            saturated = (self.inflight_ratio is not None and capacity > 0
                         and state["inflight"] >= self.inflight_ratio * capacity)
            with self._lock:
                closed = self._closed
            if closed:
                closed = waiting > REOPEN_FRACTION * threshold or saturated
            else:
                closed = waiting > threshold or saturated
            processes = max(1, int(state.get("holders") or 0), int(state.get("gates") or 0))
            share = max(1, int(max(0.0, threshold - waiting) / self.reqs_per_question / processes))
            with self._lock:
                recent = self._recent_starts()
            tickets = 0 if closed else max(0, share - recent)
        except Exception as error:  # noqa: BLE001 - an unreadable backlog must never hold questions
            if not self._warned:
                self._warned = True
                _LOGGER.warning("[question-gate] Backlog unreadable (%s); admitting questions.", type(error).__name__)
            closed, tickets = False, float("inf")
        with self._lock:
            self._closed, self._tickets = closed, tickets

    def _recent_starts(self) -> int:
        """Starts of this process within :data:`START_LAG_S`; the caller holds the lock."""
        horizon = self._clock() - START_LAG_S
        while self._starts and self._starts[0] <= horizon:
            self._starts.popleft()
        return len(self._starts)

    def _start(self) -> None:
        """Count one admitted question as running and as a recent start; the caller holds the lock."""
        self._running += 1
        self._recent_starts()  # prune here too: an unreadable backlog never reaches the read's pruning
        self._starts.append(self._clock())

    def take_ticket(self) -> bool:
        """Take one start from the current read's budget; a new read happens at most once per second per process."""
        with self._lock:
            due = self._clock() - self._read_at >= READ_INTERVAL_S
            if due:
                self._read_at = self._clock()
        if due:
            self._read()
        with self._lock:
            if self._tickets >= 1:
                self._tickets -= 1
                return True
            return False

    def is_open(self) -> bool:
        """Whether the last read found the gate open (diagnostics; admission uses :meth:`take_ticket`)."""
        with self._lock:
            return not self._closed

    def _admit(self, waited_since: float) -> bool:
        if self._stop.is_set():
            raise QuestionStartStopped("the run stopped before this question started")
        with self._lock:
            if self._running == 0:
                self._start()
                return True
        if self._wall() >= self.cutoff or self._clock() - waited_since >= self.max_wait or self.take_ticket():
            with self._lock:
                self._start()
            return True
        return False

    def _enter(self) -> None:
        """Count a question as active; this process's first one is published at once, before any broker read."""
        with self._lock:
            self._active += 1
            first = self._active == 1
        if self.active is not None:
            self.active.change(+1)
            publish = getattr(self.active, "publish", None)
            if first and publish is not None:
                publish()

    def _leave(self) -> None:
        """Count a question as no longer active (finished, or stopped before it started)."""
        with self._lock:
            self._active -= 1
        if self.active is not None:
            self.active.change(-1)

    @contextmanager
    def question(self, demand: QuestionDemand | None = None) -> Iterator[Admission]:
        """Wait until a question may start, then hold its running slot until it finishes.

        Args:
            demand: Accepted for the common gate interface; backlog mode keeps no reservation.

        Yields:
            The admission; its ``wait_sec`` is the time spent waiting at the gate.

        Raises:
            QuestionStartStopped: :meth:`stop` was called before the question started.
        """
        started = self._clock()
        self._enter()
        try:
            if not self._admit(started):
                if self.held is not None:
                    self.held.change(+1)
                try:
                    while not self._admit(started):
                        self._sleep(self._stop, random.uniform(1.0, 2.0))
                finally:
                    if self.held is not None:
                        self.held.change(-1)
            try:
                yield Admission(self._clock() - started, record={"mode": "backlog"})
            finally:
                with self._lock:
                    self._running -= 1
        finally:
            self._leave()


class _OpenGate:
    """The no-op gate: the lent pool is not in use, or the gate is switched off."""

    running = 0

    def stop(self) -> None:
        """Nothing waits at a no-op gate."""

    @contextmanager
    def question(self, demand: QuestionDemand | None = None) -> Iterator[Admission]:
        """Start at once, with no reservation."""
        yield Admission()


Gate = SlotGate | ProxySlotGate | QuestionStartGate | _OpenGate
_PROCESS_GATES: dict[tuple[str, ...], Gate] = {}
_PROCESS_LOCK = threading.Lock()
_PROXY_URL_ENV = "RUBIKSQL_KEYPROXY_URL"
_GATE_INPUTS = ("RUBIKSQL_GATEWAY_POOL", GATE_ENV, MODE_ENV, CAPACITY_ENV, INFLIGHT_ENV, MAX_WAIT_ENV, REQS_ENV,
                _PROXY_URL_ENV)


def capacity_override(environ: Mapping[str, str] | None = None) -> int | None:
    """The manual slot capacity, ``RUBIKSQL_QUESTION_START_CAPACITY``: a positive integer, or ``None`` when unset.

    Args:
        environ: Environment to read; the process environment by default.

    Returns:
        The total request slots the gate may reserve on this host, for any provider.

    Raises:
        ValueError: The value is not a positive integer.
    """
    env = os.environ if environ is None else environ
    raw = (env.get(CAPACITY_ENV) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(f"{CAPACITY_ENV} must be a positive integer number of request slots: {raw!r}")
    return value


def gate_mode(environ: Mapping[str, str] | None = None) -> str:
    """The admission rule, ``RUBIKSQL_QUESTION_START_GATE_MODE``: ``slots`` (default) or ``backlog``.

    Raises:
        ValueError: Any other value.
    """
    env = os.environ if environ is None else environ
    mode = (env.get(MODE_ENV) or DEFAULT_MODE).strip().lower()
    if mode not in MODES:
        raise ValueError(f"{MODE_ENV} must be one of {', '.join(MODES)}: {env.get(MODE_ENV)!r}")
    return mode


def _lent_pool(environ: Mapping[str, str]) -> Any | None:
    """The lent file pool named by the gateway configuration, or ``None`` when another pool is in use.

    Raises:
        ValueError: The key proxy is configured (``RUBIKSQL_KEYPROXY_URL``) but no usable lent file pool is: every
            Rubik experiment shares the proxy's pool, so such a launch is refused rather than run ungated.
    """
    proxied = bool((environ.get(_PROXY_URL_ENV) or os.environ.get(_PROXY_URL_ENV) or "").strip())
    path = environ.get("RUBIKSQL_GATEWAY_POOL")
    config: Any = None
    if path:
        try:
            config = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            config = None
    if not isinstance(config, dict) or "key_files" not in config:
        if proxied:
            raise ValueError(f"{_PROXY_URL_ENV} is set but RUBIKSQL_GATEWAY_POOL names no lent file pool")
        return None
    from .lent_keys import LentKeyPool, validate_files
    try:
        return LentKeyPool(validate_files(config["key_files"]))
    except Exception as error:  # noqa: BLE001 - the provider route reports a bad pool; the gate only steps aside
        if proxied:
            raise ValueError(f"the lent pool for the key proxy is unusable: {error}") from None
        _LOGGER.warning("[question-gate] Lent pool unusable for the gate (%s); questions start ungated.",
                        type(error).__name__)
        return None


def _override_broker(environ: Mapping[str, str]) -> Any:
    """The host broker of a gate that runs on the manual capacity alone (no lent file pool).

    ``RUBIKSQL_KEY_BROKER_STATE`` when set; else the experiment gateway's default broker beside the pool file named
    by ``RUBIKSQL_GATEWAY_POOL`` (``<pool>.state.broker.sqlite3``, as `rubiksql_hb.experiment_gateway` places it),
    so the host's processes on one pool share it; else ``~/.rubiksql-hb/question-gate.sqlite3``.
    """
    from .key_broker import BROKER_STATE_ENV, SqliteKeyBroker

    configured = (environ.get(BROKER_STATE_ENV) or "").strip()
    pool = (environ.get("RUBIKSQL_GATEWAY_POOL") or "").strip()
    if configured:
        path = Path(configured).expanduser()
    elif pool:
        state = Path(pool).expanduser().with_suffix(".state.sqlite3")
        path = state.with_name(state.stem + ".broker.sqlite3")
    else:
        path = Path.home() / ".rubiksql-hb" / "question-gate.sqlite3"
    return SqliteKeyBroker.for_path(path)


class _LiveCap:
    """The lent pool's per-key cap read on every gate cycle; a failed read reuses the last good value.

    A read can fail for a moment (the cap controller replacing the file on Windows); a failure before any good
    read propagates, which the gate counts as a broker failure.
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self._last: int | None = None
        self._warned = False

    def __call__(self) -> int:
        try:
            self._last = self._pool.cap()
        except Exception as error:  # noqa: BLE001 - one unreadable moment must not stall or open the gate
            if self._last is None:
                raise
            if not self._warned:
                self._warned = True
                _LOGGER.warning("[question-gate] Cap file unreadable (%s); using the last cap %d.",
                                type(error).__name__, self._last)
        return self._last


def build_gate(environ: Mapping[str, str] | None = None) -> Gate:
    """Build the gate from the environment.

    Slots mode is on by default with the lent file pool, and with any other provider once
    ``RUBIKSQL_QUESTION_START_CAPACITY`` is set (the capacity then alone decides, on a host broker from
    :func:`_override_broker`). With the lent pool and the key proxy (``RUBIKSQL_KEYPROXY_URL``) the reservations
    live in the proxy (:class:`ProxySlotGate`). Backlog mode needs the lent pool and no key proxy. Anything else is a
    no-op. Every variable is validated even when the gate ends up a no-op, so a typo never hides.

    Args:
        environ: Environment to read; the process environment by default.

    Returns:
        The configured gate.

    Raises:
        ValueError: A gate variable holds an invalid value, or backlog mode is asked for with the key proxy.
    """
    env = os.environ if environ is None else environ
    ratio = _positive(GATE_ENV, env.get(GATE_ENV), DEFAULT_RATIO, allow_off=True)
    inflight_ratio = _positive(INFLIGHT_ENV, env.get(INFLIGHT_ENV), None, allow_off=True)
    max_wait = _positive(MAX_WAIT_ENV, env.get(MAX_WAIT_ENV), DEFAULT_MAX_WAIT_S, allow_off=False)
    reqs = _positive(REQS_ENV, env.get(REQS_ENV), DEFAULT_REQS_PER_QUESTION, allow_off=False)
    mode = gate_mode(env)
    capacity = capacity_override(env)
    if ratio is None:
        return _OpenGate()
    from .key_broker import host_counter

    pool = _lent_pool(env)
    if pool is None:
        if mode != "slots" or capacity is None:
            return _OpenGate()
        broker = _override_broker(env)
        return SlotGate(lambda request: broker.gate_cycle(request, files=(), cap=0, capacity=capacity),
                        held=host_counter(broker, "gate_held"), active=host_counter(broker, "gate_active"))

    cutoff = float("inf") if pool.ends_at is None else pool.ends_at
    if pool.proxy is not None:
        if mode != "slots":
            raise ValueError(f"{MODE_ENV}=backlog is per host; with the key proxy use slots")
        try:
            # One authenticated call: a missing proxy, a stale token copy or a wrong URL refuses the launch here
            # (after the bounded reconnect budget), not every question at the gate.
            pool.proxy.check()
        except RuntimeError as error:
            raise ValueError(str(error)) from None
        return ProxySlotGate(pool.proxy, capacity=capacity, cutoff=cutoff)
    held, active = host_counter(pool.broker, "gate_held"), host_counter(pool.broker, "gate_active")
    if mode == "slots":
        cap = _LiveCap(pool)
        return SlotGate(lambda request: pool.broker.gate_cycle(request, files=pool.names, cap=cap(),
                                                                capacity=capacity),
                        cutoff=cutoff, held=held, active=active)

    def backlog() -> dict[str, float]:
        state = pool.broker.backlog(pool.names)
        total = capacity if capacity is not None else min(pool.cap(), state["ceiling"]) * state["usable_keys"]
        return {"waiting": state["waiting"], "inflight": state["inflight"], "holders": state["holders"],
                "gates": state["gates"], "capacity": total}

    return QuestionStartGate(backlog, ratio=ratio, inflight_ratio=inflight_ratio, max_wait=max_wait,
                             cutoff=cutoff, reqs_per_question=reqs, held=held, active=active)


def process_gate() -> Gate:
    """This process's gate, built once per gate configuration from the environment."""
    key = tuple(os.environ.get(name, "") for name in _GATE_INPUTS)
    with _PROCESS_LOCK:
        gate = _PROCESS_GATES.get(key)
        if gate is None:
            gate = _PROCESS_GATES[key] = build_gate()
        return gate


def stop_all() -> None:
    """Release every waiter of every gate this process built (the run is stopping)."""
    with _PROCESS_LOCK:
        gates = list(_PROCESS_GATES.values())
    for gate in gates:
        gate.stop()
