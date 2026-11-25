"""The candidate and judge phase budgets of one vote, and the phase timing record it leaves.

A budgeted vote answers in two timed phases:

- **Candidate phase.** Its clock starts when the vote starts answering; the question-start gate wait
  (`rubiksql_hb.question_gate`) stays outside it. Candidates run until every seat has finished or
  ``RUBIKSQL_CANDIDATE_BUDGET_SEC`` (default 2000 s) has passed. Seats still running then are cut, and the judge
  works with every seat that finished. A quorum top-up may run only inside the same budget.
- **Judge phase.** Its clock starts when the candidate phase ends and covers everything after it: the concurrent
  judge passes, the reviewer, the hooks and the final answer, within ``RUBIKSQL_JUDGE_BUDGET_SEC`` (default 2400 s).
  At its deadline the vote is stopped, and the evaluator's checkpoint recovery or plurality net decides. A
  judge-only replay over saved candidates has no candidate phase: its judge phase starts with the vote.

A vote's overall budget (``nl2sql_vote(timeout_s=...)``) is a ceiling over both phases, counted from the
vote's submission (:class:`VotePhases` creation, after the gate), so a slow session acquisition comes out of the
judge phase and never pushes the vote past the evaluator's wait. The evaluator passes the question wall
(:func:`question_wall`) minus :data:`QUESTION_MARGIN_SEC`, which by default is exactly the sum of the two budgets.
A smaller ceiling (an explicit ``RUBIKSQL_QUESTION_TIMEOUT_SEC``, a short smoke) scales the candidate budget
down in proportion (:func:`effective_budgets`), so the judge keeps its proportional share; the evaluator logs
the effective budgets at launch, with a warning when a wall scales them. With ``RUBIKSQL_CANDIDATE_BUDGET_SEC=0`` there is no candidate cap: the ceiling alone bounds the vote, and the
evaluator's completed-pool recovery handles a vote stopped during generation.

:class:`VotePhases` owns one vote's phase clock and records, in its :meth:`VotePhases.snapshot`, the candidate phase
end, each seat's wall time (seats cut at the cap marked ``cut``), the judge start, each judge pass's and
reviewer's wall time (one still running when the vote stopped marked ``in_flight`` with its elapsed time).

Stdlib only, so the evaluator and the runtime modules import it without loading HeavenBase.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from typing import Any

__all__ = [
    "CANDIDATE_BUDGET_ENV", "JUDGE_BUDGET_ENV", "QUESTION_MARGIN_SEC", "QUESTION_TIMEOUT_ENV", "RETIRED_ENV",
    "SCHEMA", "SEAT_SALVAGE_MARGIN_SEC", "VotePhases", "bind", "candidate_budget", "current", "describe",
    "effective_budgets", "end_candidates", "judge_budget", "question_wall", "seat_hard_deadline", "track",
    "vote_budget",
]

SCHEMA = "rubiksql.vote-phases/1"
CANDIDATE_BUDGET_ENV = "RUBIKSQL_CANDIDATE_BUDGET_SEC"
JUDGE_BUDGET_ENV = "RUBIKSQL_JUDGE_BUDGET_SEC"
#: An explicit question wall; without it the wall is the two phase budgets plus :data:`QUESTION_MARGIN_SEC`.
QUESTION_TIMEOUT_ENV = "RUBIKSQL_QUESTION_TIMEOUT_SEC"
_DEFAULT_CANDIDATE_BUDGET_SEC = 2000.0
_DEFAULT_JUDGE_BUDGET_SEC = 2400.0
#: The evaluator's wait beyond the vote ceiling: the completed-pool tail judge (90 s) and evaluator overhead.
QUESTION_MARGIN_SEC = 120.0
#: A seat leaves its loop and salvages its last verified SQL this long before the candidate cap would cut it
#: (`nl2sql._HARD_DEADLINE_SEC`'s default): a turn plus its tool calls can take several minutes.
SEAT_SALVAGE_MARGIN_SEC = 300.0
#: The salvage margin is at most this share of a candidate budget, so a short (scaled) budget keeps most of it.
_SEAT_SALVAGE_MARGIN_SHARE = 0.15
#: The seat hard deadline when there is no candidate cap (the single-budget default it had before G31).
_UNCAPPED_SEAT_DEADLINE_SEC = 1500
#: Settings of the early straggler cut that G31 retired: the judge phase now has its own budget.
RETIRED_ENV = ("RUBIKSQL_STRAGGLER_RESERVE_SEC", "RUBIKSQL_STRAGGLER_MIN_DONE")

_LOGGER = logging.getLogger(__name__)
_CURRENT: ContextVar[VotePhases | None] = ContextVar("rubiksql_vote_phases", default=None)


def _seconds(name: str, default: float, *, allow_zero: bool) -> float:
    """A finite non-negative budget from the environment; a bad value is a configuration error."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number of seconds, got {raw!r}") from None
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'} and finite, got {raw!r}")
    return value


def candidate_budget() -> float | None:
    """The candidate phase budget in seconds (``RUBIKSQL_CANDIDATE_BUDGET_SEC``, default 2000).

    Returns:
        float | None: Seconds, or ``None`` when ``0`` disables the candidate cap.
    """
    return _seconds(CANDIDATE_BUDGET_ENV, _DEFAULT_CANDIDATE_BUDGET_SEC, allow_zero=True) or None


def judge_budget() -> float:
    """The judge phase budget in seconds (``RUBIKSQL_JUDGE_BUDGET_SEC``, default 2400)."""
    return _seconds(JUDGE_BUDGET_ENV, _DEFAULT_JUDGE_BUDGET_SEC, allow_zero=False)


def question_wall() -> float:
    """The evaluator's per-question wait for one attempt, in seconds.

    ``RUBIKSQL_QUESTION_TIMEOUT_SEC`` when set; otherwise both phase budgets plus :data:`QUESTION_MARGIN_SEC`
    (4520 s by default), so the vote's own deadlines always fire before the evaluator's wait.

    Returns:
        float: Seconds, always above :data:`QUESTION_MARGIN_SEC`.

    Raises:
        ValueError: An explicit wall does not leave the vote any time inside the margin.
    """
    wall = _seconds(QUESTION_TIMEOUT_ENV, (candidate_budget() or 0.0) + judge_budget() + QUESTION_MARGIN_SEC,
                    allow_zero=False)
    if wall <= QUESTION_MARGIN_SEC:
        raise ValueError(f"{QUESTION_TIMEOUT_ENV} must exceed the {QUESTION_MARGIN_SEC:g} s evaluator margin")
    return wall


def vote_budget() -> float:
    """The vote ceiling the evaluator passes as ``timeout_s``: :func:`question_wall` minus the margin."""
    return question_wall() - QUESTION_MARGIN_SEC


def effective_budgets(ceiling: float) -> tuple[float | None, float]:
    """The candidate budget and the least judge time of a generating vote with this ceiling.

    The configured budgets when they fit the ceiling; otherwise the candidate budget scales down in proportion,
    and the judge keeps its proportional share of the ceiling. Without a candidate cap the judge time is the
    judge budget, bounded by the ceiling. A judge-only vote has no candidate phase and keeps the whole judge
    budget within the ceiling.

    Args:
        ceiling (float): The vote ceiling (``timeout_s``), positive.

    Returns:
        tuple[float | None, float]: ``(candidate, judge)`` seconds; ``candidate`` is ``None`` without a cap.
    """
    cap, judge = candidate_budget(), judge_budget()
    if cap is not None and cap + judge > ceiling:
        cap = ceiling * cap / (cap + judge)
    return cap, min(judge, ceiling - (cap or 0.0))


def describe() -> str:
    """One launch-log line: the effective phase budgets and the question wall, and whether a wall scaled them.

    Returns:
        str: The line's text.

    Raises:
        ValueError: A budget setting is invalid (the launch should refuse).
    """
    wall = question_wall()
    cap, judge = effective_budgets(wall - QUESTION_MARGIN_SEC)
    configured = candidate_budget()
    line = (f"candidates {'uncapped' if cap is None else f'{cap:.0f}s'}, judge at least {judge:.0f}s, "
            f"question wall {wall:.0f}s")
    if cap is not None and configured is not None and cap < configured:
        line += (f" (WARNING: {QUESTION_TIMEOUT_ENV}={wall:g} scales the configured {configured:g}s/"
                 f"{judge_budget():g}s phase budgets down)")
    return line


def seat_hard_deadline() -> int:
    """The default in-loop hard deadline of one candidate seat, in seconds of its own clock.

    The candidate budget minus :data:`SEAT_SALVAGE_MARGIN_SEC` (1700 s by default): past it the seat leaves its
    loop and salvages its last verified SQL instead of being cut with nothing. Without a candidate cap it stays
    the pre-G31 1500 s.

    Returns:
        int: Seconds.
    """
    budget = candidate_budget()
    return _UNCAPPED_SEAT_DEADLINE_SEC if budget is None else max(1, int(budget - SEAT_SALVAGE_MARGIN_SEC))


class VotePhases:
    """One vote's phase clock and timing record; safe to read from any thread.

    The evaluator may construct it before submitting the vote so the record survives a vote stopped at its
    deadline; the vote calls :meth:`start` when it starts answering and :meth:`end_candidates` when its pool
    is final. Times in :meth:`snapshot` are seconds since :meth:`start`.

    Args:
        budget_sec (float | None): The vote ceiling (``timeout_s``); ``None`` for an unbudgeted vote, which
            records timing but has no deadline.
        now (Callable[[], float]): Monotonic time source.
        listener (Callable[[str], Any] | None): Told each phase transition once: ``candidates`` when a generating
            vote starts, ``judge`` when the judge phase starts (at once for a judge-only vote), ``done`` when the
            vote returns or unwinds. The question-start gate's reservation follows it
            (`rubiksql_hb.question_gate.Admission.attempt`); it must not block, and its errors are logged and
            ignored.
    """

    def __init__(self, budget_sec: float | None, *, now: Callable[[], float] = time.monotonic,
                 listener: Callable[[str], Any] | None = None) -> None:
        self._now = now
        self._listener = listener
        self._lock = threading.Lock()
        self.created = now()
        self.budget_sec = budget_sec if budget_sec and budget_sec > 0 else None
        self.candidate_budget_sec = effective_budgets(self.budget_sec)[0] if self.budget_sec else None
        # The configured judge budget; the ceiling clamp in `end_candidates` leaves a scaled vote its share.
        self.judge_budget_sec = judge_budget() if self.budget_sec else None
        self.started: float | None = None
        self.judge_only = False
        self.candidate_end: float | None = None
        self.judge_deadline: float | None = None
        self.ended: float | None = None
        self._entries: dict[str, list[dict[str, Any]]] = {"seats": [], "judge_passes": [], "reviewer": []}

    # ── the clock ─────────────────────────────────────────────────────────────

    def start(self, *, judge_only: bool) -> None:
        """Start the candidate phase now; a judge-only vote starts its judge phase instead.

        Args:
            judge_only (bool): The vote judges a saved pool and generates nothing.
        """
        with self._lock:
            self.started = self._now()
            self.judge_only = judge_only
        if judge_only:
            self.end_candidates()
        else:
            self._notify("candidates")

    def end_candidates(self, *, at: float | None = None) -> None:
        """End the candidate phase and start the judge phase's clock; later calls change nothing.

        Args:
            at (float | None): Monotonic end time (the cut, for a phase the cap ended); now when omitted.
        """
        with self._lock:
            if self.candidate_end is not None or self.started is None:
                return
            self.candidate_end = self._now() if at is None else at
            if self.budget_sec is not None:
                self.judge_deadline = min(self.candidate_end + self.judge_budget_sec, self.created + self.budget_sec)
        self._notify("judge")

    def finish(self) -> None:
        """Record that the vote returned or unwound; later calls change nothing."""
        with self._lock:
            if self.ended is not None:
                return
            self.ended = self._now()
        self._notify("done")

    def _notify(self, phase: str) -> None:
        """Tell the listener one phase transition; a failing listener never fails the vote.

        Nothing but ``done`` is told once the vote finished: an attempt the evaluator gave up before it started
        (:meth:`finish` first) must not take its reservation back by starting late.
        """
        if self._listener is None or (phase != "done" and self.ended is not None):
            return
        try:
            self._listener(phase)
        except Exception as error:  # noqa: BLE001 - phase bookkeeping must not break answering
            _LOGGER.warning("[phase-budget] Phase listener failed on %s (%s).", phase, type(error).__name__)

    @property
    def candidate_deadline(self) -> float | None:
        """Monotonic time of the candidate cap, or ``None`` without one (or before :meth:`start`).

        The candidate budget from the vote's start, never past the vote ceiling.
        """
        if self.started is None or self.candidate_budget_sec is None:
            return None
        return min(self.started + self.candidate_budget_sec, self.created + self.budget_sec)

    def seat_salvage_at(self) -> float | None:
        """Monotonic time past which a seat leaves its loop to salvage before the cap cuts it; ``None`` without a cap.

        :data:`SEAT_SALVAGE_MARGIN_SEC` before the candidate cap, or 15% of a shorter (scaled) candidate budget.
        Unlike the seat's own hard deadline it holds for a top-up seat that started late.
        """
        cap = self.candidate_deadline
        if cap is None:
            return None
        return cap - min(SEAT_SALVAGE_MARGIN_SEC, _SEAT_SALVAGE_MARGIN_SHARE * self.candidate_budget_sec)

    def seat_salvage_due(self) -> bool:
        """Whether :meth:`seat_salvage_at` has passed (never without a candidate cap)."""
        salvage_at = self.seat_salvage_at()
        return salvage_at is not None and self._now() > salvage_at

    def deadline(self) -> float | None:
        """The monotonic time the vote must be stopped by, as known now; ``None`` for an unbudgeted vote.

        Before the judge phase it is the latest the judge phase could end; from then on the judge deadline.
        It only moves earlier once the vote has started, so a waiter re-reading it never stops a vote early.
        """
        if self.budget_sec is None:
            return None
        if self.judge_deadline is not None:
            return self.judge_deadline
        ceiling = self.created + self.budget_sec
        cap = self.candidate_deadline
        return ceiling if cap is None else min(ceiling, cap + self.judge_budget_sec)

    def horizon(self) -> float | None:
        """The turn-retry horizon (`llm_runtime.bind_retry_horizon`): the current phase's deadline."""
        if self.candidate_end is None and self.candidate_deadline is not None:
            return self.candidate_deadline
        return self.deadline()

    # ── the record ────────────────────────────────────────────────────────────

    def open(self, kind: str, **labels: Any) -> dict[str, Any]:
        """Start timing one seat, judge pass or reviewer run.

        Args:
            kind (str): ``seats``, ``judge_passes`` or ``reviewer``.
            **labels: Identifying fields kept in the record (``slot``, ``round``, ``pass``).

        Returns:
            dict[str, Any]: The entry, for :meth:`close`.
        """
        entry = {**labels, "_start": self._now()}
        with self._lock:
            self._entries[kind].append(entry)
        return entry

    def stopped_at(self) -> float:
        """Now, or the judge deadline once it has passed: the end of a run the deadline stopped."""
        now = self._now()
        return now if self.judge_deadline is None else min(now, self.judge_deadline)

    def close(self, entry: dict[str, Any], status: str, *, at: float | None = None) -> None:
        """Record how a timed run ended; the first close wins.

        Args:
            entry (dict[str, Any]): An entry from :meth:`open`.
            status (str): ``done``, ``failed``, ``cut`` or ``in_flight``.
            at (float | None): Monotonic end time; now when omitted.
        """
        with self._lock:
            if "_end" not in entry:
                entry["_end"] = self._now() if at is None else at
                entry["status"] = status

    def snapshot(self) -> dict[str, Any]:
        """The phase timing record (schema ``rubiksql.vote-phases/1``), read now.

        A run still open is reported ``in_flight`` with its elapsed time up to now, or up to the judge deadline
        when the vote has passed it (the vote was stopped there).

        Returns:
            dict[str, Any]: Budgets, phase boundaries and per-run wall times, in seconds since :meth:`start`.
        """
        stop = self.stopped_at()
        with self._lock:
            if self.ended is not None:
                stop = min(stop, self.ended)   # a run left open when the vote returned or unwound
            origin = self.started if self.started is not None else self.created

            def rel(moment: float | None) -> float | None:
                return None if moment is None else round(moment - origin, 3)

            def runs(kind: str) -> list[dict[str, Any]]:
                out = []
                for entry in self._entries[kind]:
                    end = entry.get("_end", max(stop, entry["_start"]))
                    out.append({**{k: v for k, v in entry.items() if not k.startswith("_") and k != "status"},
                                "start_sec": rel(entry["_start"]), "wall_sec": round(end - entry["_start"], 3),
                                "status": entry.get("status", "in_flight")})
                return out

            seats = runs("seats")
            passes = runs("judge_passes")
            return {
                "schema": SCHEMA,
                "vote_budget_sec": self.budget_sec,
                "candidate_budget_sec": None if self.candidate_budget_sec is None else round(self.candidate_budget_sec, 3),
                "judge_budget_sec": self.judge_budget_sec,
                "judge_only": self.judge_only,
                "started": self.started is not None,
                "candidate_end_sec": rel(self.candidate_end),
                "candidate_cut": any(seat["status"] == "cut" for seat in seats),
                "judge_start_sec": min((p["start_sec"] for p in passes), default=None),
                "judge_deadline_sec": rel(self.judge_deadline),
                "ended_sec": rel(self.ended),
                "seats": seats,
                "judge_passes": passes,
                "reviewer": runs("reviewer"),
            }


@contextlib.contextmanager
def bind(phases: VotePhases | None) -> Iterator[None]:
    """Make ``phases`` the current vote's phases for this task context and the tasks it creates.

    Args:
        phases (VotePhases | None): The vote's phases; ``None`` binds none.

    Yields:
        None while bound.
    """
    token = _CURRENT.set(phases)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current() -> VotePhases | None:
    """The bound vote's phases, or ``None`` outside a vote."""
    return _CURRENT.get()


def end_candidates(*, at: float | None = None) -> None:
    """End the bound vote's candidate phase (now, or at ``at``); a no-op outside a vote."""
    phases = _CURRENT.get()
    if phases is not None:
        phases.end_candidates(at=at)


@contextlib.contextmanager
def track(kind: str, **labels: Any) -> Iterator[None]:
    """Time the enclosed judge pass or reviewer run in the bound vote's record; a no-op outside a vote.

    A normal exit records ``done``; a cancellation (the vote stopped at its deadline) ``in_flight``, ending at the
    judge deadline when it has passed; any other exception ``failed``.

    Args:
        kind (str): ``judge_passes`` or ``reviewer``.
        **labels: Identifying fields such as ``pass``.

    Yields:
        None while the run is timed.
    """
    phases = _CURRENT.get()
    if phases is None:
        yield
        return
    entry = phases.open(kind, **labels)
    status, at = "failed", None
    try:
        yield
        status = "done"
    except asyncio.CancelledError:
        status, at = "in_flight", phases.stopped_at()
        raise
    finally:
        phases.close(entry, status, at=at)
