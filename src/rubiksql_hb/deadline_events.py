"""Deadline-mechanics events of one answering attempt, and the run-level deadline counters.

A live vote runs against two phase budgets (`rubiksql_hb.phase_budget`): candidates are cut at the
candidate cap, a quorum top-up may be skipped near that cap, the judge phase ends at its own
deadline, a model turn may be retried once on another key. None of these leave a reliable trace
in the row when the vote is later cancelled, so the evaluator binds a plain dict per attempt
(:func:`bind`) and every mechanism adds its count to it (:func:`note`). The dict is shared by
reference with the vote's event-loop thread, so the evaluator reads it whichever way the attempt
ended.

:func:`counters` turns scored rows into the run summary that every arm reports: how the finals
were decided (inline judge passes, checkpoint recovery, plurality, abandoned), finals within
:data:`LATE_FINAL_MARGIN_SEC` of their own phase deadline, seats lost to provider overload, forced judge
passes, skipped top-ups, and the mean
seconds per model call of each model, so arms that ran in different provider windows can be
compared.

Stdlib only, so the evaluator and the runtime modules import it without loading HeavenBase.
"""
from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextvars import ContextVar
from typing import Any

__all__ = ["EVENTS", "LATE_FINAL_MARGIN_SEC", "bind", "counters", "late_final", "merge", "note", "summary"]

#: Every event a mechanism may note, in report order.
EVENTS = (
    "candidate_cap_cut",         # a candidate phase ended by its budget with seats still running
    "candidate_seats_cut",       # seats stopped at the candidate cap
    "straggler_cut_quorum",      # (before G31) a cut at the straggler cap with a quorum done
    "straggler_cut_floor",       # (before G31) a cut at the grace floor (cap + floor) with two diverse seats done
    "straggler_seats_cut",       # (before G31) seats stopped by either straggler cut
    "seat_cap_exit",             # a seat that left its loop just before a due cut
    "seat_cap_salvaged",         # ... and submitted its last verified SQL
    "topup_skipped",             # a quorum top-up skipped within the reserve of the cap
    "topup_ran",                 # a quorum top-up that ran
    "overload_retry",            # an unanswered model request retried once on the next key
    "unparseable_retry",         # a model turn with malformed tool-call arguments retried once
    "forced_submit",             # a judge pass told to submit, with only submit_final_sql offered
    "reviewer_deadline_skip",    # a review skipped because the judge deadline was too near
)

#: A final decided within this many seconds of its own phase deadline ran into the wall.
LATE_FINAL_MARGIN_SEC = 100.0
#: Rows recorded before the phase budgets (no ``deadline.phases``): a final at or after this many seconds of the
#: old 1800 s question budget ran into the wall.
_LEGACY_LATE_FINAL_SEC = 1780.0

_EVENTS: ContextVar[dict[str, int] | None] = ContextVar("rubiksql_deadline_events", default=None)
_LOCK = threading.Lock()


@contextlib.contextmanager
def bind(events: dict[str, int] | None) -> Iterator[None]:
    """Collect this task context's events, and those of the tasks it creates, into ``events``.

    Args:
        events (dict[str, int] | None): The attempt's event dict; ``None`` collects nothing.

    Yields:
        None while the dict is bound.
    """
    token = _EVENTS.set(events)
    try:
        yield
    finally:
        _EVENTS.reset(token)


def note(name: str, count: int = 1) -> None:
    """Add ``count`` to event ``name`` of the bound attempt; a no-op when none is bound.

    Args:
        name (str): One of :data:`EVENTS`.
        count (int): The amount to add.
    """
    events = _EVENTS.get()
    if events is None or count == 0:
        return
    with _LOCK:
        events[name] = events.get(name, 0) + count


def merge(*parts: Mapping[str, int] | None) -> dict[str, int]:
    """Sum event dicts (the attempts of one question), keeping only non-zero events.

    Args:
        *parts (Mapping[str, int] | None): Event dicts; ``None`` entries are skipped.

    Returns:
        dict[str, int]: Event to total, in :data:`EVENTS` order and then by name.
    """
    total: dict[str, int] = {}
    for part in parts:
        with _LOCK:   # an abandoned attempt's loop thread may still note into its dict
            items = list((part or {}).items())
        for name, value in items:
            total[name] = total.get(name, 0) + int(value or 0)
    order = {name: index for index, name in enumerate(EVENTS)}
    return {name: total[name] for name in sorted(total, key=lambda n: (order.get(n, len(order)), n)) if total[name]}


def _decision(row: Mapping[str, Any]) -> str:
    """How a row's final was decided: the categories the G28/G30 readouts count."""
    if row.get("placeholder"):
        return "abandoned"
    if row.get("judge_checkpoint_recovery") or row.get("decider_source") == "tail_judge":
        return "checkpoint"
    if row.get("decider_source") == "fallback_candidate":
        return "plurality"
    if row.get("pool_guard_fallback"):
        return "pool_guard"
    return "inline"


def _overloaded(record: Any) -> bool:
    """A seat or pass the provider left unanswered: the gateway's overload or connection-error category."""
    error = record.get("error") if isinstance(record, Mapping) else None
    return isinstance(error, str) and ("overloaded" in error or "connection error" in error.lower())


def _judge_started(row: Mapping[str, Any]) -> bool:
    """Whether a judge ran for this row: inline passes, or a judge record (or attempt) with a transcript.

    The evaluator's candidate net stamps every live plurality row with a judge record whose
    ``transcript_status`` is ``not_run``; that stub does not count.
    """
    if row.get("judge_passes"):
        return True
    debug = row.get("judge_debug")
    records = [debug, *(debug.get("attempts") or [])] if isinstance(debug, Mapping) else []
    return any(isinstance(record, Mapping) and record.get("transcript_status") not in (None, "not_run")
               for record in records)


def _judge_fallback(row: Mapping[str, Any]) -> bool:
    """Whether the vote's own judge ended in its plurality fallback (``judge_debug.mode`` ``fallback*``)."""
    debug = row.get("judge_debug")
    return isinstance(debug, Mapping) and str(debug.get("mode") or "").startswith("fallback")


def late_final(row: Mapping[str, Any]) -> bool:
    """Whether the row's final ran into its wall: decided within :data:`LATE_FINAL_MARGIN_SEC` of its phase deadline.

    A row without a phase record (an older run) keeps the old rule exactly, a duration at or after 1780 s. Otherwise
    a completed-pool tail judge always ran after the wall, and the last started attempt's phase record
    (``deadline.phases``, `phase_budget.VotePhases.snapshot`) decides: a vote that
    reached its judge phase is late when it never returned or returned within the margin of the judge deadline; one
    that never left a budgeted candidate phase ran into the vote's wall. A vote that never started (its session
    failed to start) decided nothing late.

    Args:
        row (Mapping[str, Any]): An eval_bird result row.

    Returns:
        bool: The final was late.
    """
    deadline = row.get("deadline") if isinstance(row.get("deadline"), Mapping) else {}
    if "phases" not in deadline:
        return float(row.get("duration_sec") or 0.0) >= _LEGACY_LATE_FINAL_SEC
    if row.get("tail_judge_recovery"):
        return True   # the evaluator judged a completed pool after the vote ran into its wall
    phases = [p for p in deadline.get("phases") or [] if isinstance(p, Mapping) and p.get("started")]
    if not phases:
        return False
    last = phases[-1]
    judge_deadline, ended = last.get("judge_deadline_sec"), last.get("ended_sec")
    if judge_deadline is not None:
        return ended is None or float(ended) >= float(judge_deadline) - LATE_FINAL_MARGIN_SEC
    return last.get("vote_budget_sec") is not None and last.get("candidate_end_sec") is None


def _forced(record: Any) -> bool:
    debug = record.get("judge_debug") if isinstance(record, Mapping) else None
    return isinstance(debug, Mapping) and bool(debug.get("forced_submit"))


def _judge_records(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The row's judge pass records: every inline pass, else the single recorded judge."""
    passes = [p for p in row.get("judge_passes") or [] if isinstance(p, Mapping)]
    if passes:
        return passes
    debug = row.get("judge_debug")
    return [{"judge_debug": debug, "error": debug.get("error")}] if isinstance(debug, Mapping) else []


def counters(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The run's deadline counters from its scored rows.

    Args:
        rows (Iterable[Mapping[str, Any]]): eval_bird result rows (the ``results`` of every
            database file of one run).

    Returns:
        dict[str, Any]: ``questions``; ``decisions`` (inline, checkpoint, plurality, abandoned,
            pool_guard); ``non_inline``; ``late_finals`` (:func:`late_final`); ``plurality_after_judge`` (plurality finals whose judge had
            started: no judge-stage checkpoint survived); ``inline_judge_fallback`` (inline finals,
            counted in ``inline``, where the vote's own judge fell back to the plurality candidate:
            ``plurality`` counts only the evaluator's candidate net); ``overload_dead_seats``
            (candidate seats and judge passes that ended with a gateway overload or connection error); ``forced_passes`` and
            ``forced_questions``; ``checkpoint_forced`` (checkpoint finals taken from a forced
            pass); ``events`` (the summed attempt events, :data:`EVENTS`); and
            ``seconds_per_call`` (model to ``{"calls", "mean_sec"}``, from each row's
            ``deadline.model_calls``).
    """
    decisions = {name: 0 for name in ("inline", "checkpoint", "plurality", "abandoned", "pool_guard")}
    late = overload = forced = forced_questions = checkpoint_forced = questions = judged_plurality = 0
    judge_fallback = 0
    events: list[Mapping[str, int]] = []
    calls: dict[str, list[float]] = {}
    for row in rows:
        questions += 1
        decision = _decision(row)
        decisions[decision] += 1
        if decision == "plurality" and _judge_started(row):
            judged_plurality += 1
        if decision == "inline" and _judge_fallback(row):
            judge_fallback += 1
        if late_final(row):
            late += 1
        seats = [c for c in row.get("candidates_debug") or [] if isinstance(c, Mapping)]
        judges = _judge_records(row)
        overload += sum(_overloaded(record) or _overloaded(record.get("judge_debug")) for record in [*seats, *judges])
        row_forced = sum(_forced(record) for record in judges)
        forced += row_forced
        forced_questions += bool(row_forced)
        if row.get("judge_checkpoint_recovery") and _forced(row):
            checkpoint_forced += 1
        deadline = row.get("deadline") if isinstance(row.get("deadline"), Mapping) else {}
        events.append(deadline.get("events") or {})
        for model, (count, seconds) in (deadline.get("model_calls") or {}).items():
            into = calls.setdefault(model, [0, 0.0])
            into[0] += int(count)
            into[1] += float(seconds)
    return {
        "questions": questions,
        "decisions": decisions,
        "non_inline": questions - decisions["inline"],
        "plurality_after_judge": judged_plurality,
        "inline_judge_fallback": judge_fallback,
        "late_finals": late,
        "overload_dead_seats": overload,
        "forced_passes": forced,
        "forced_questions": forced_questions,
        "checkpoint_forced": checkpoint_forced,
        "events": merge(*events),
        "seconds_per_call": {model: {"calls": count, "mean_sec": round(seconds / count, 2) if count else None}
                             for model, (count, seconds) in sorted(calls.items())},
    }


def summary(result: Mapping[str, Any]) -> str:
    """One run-log line for :func:`counters`' result."""
    d = result["decisions"]
    events = result["events"]
    calls = ", ".join(f"{model} {v['mean_sec']}s/call ({v['calls']})" for model, v in result["seconds_per_call"].items())
    return (f"{result['questions']} questions; non-inline {result['non_inline']} (checkpoint {d['checkpoint']}, "
            f"plurality {d['plurality']} ({result['plurality_after_judge']} after a judge started), "
            f"in-judge fallback {result['inline_judge_fallback']}, abandoned {d['abandoned']}, pool-guard {d['pool_guard']}); "
            f"finals within {LATE_FINAL_MARGIN_SEC:.0f}s of their deadline {result['late_finals']}; overload-dead seats "
            f"{result['overload_dead_seats']}; forced passes {result['forced_passes']} "
            f"({result['forced_questions']} questions, {result['checkpoint_forced']} checkpoint finals); "
            f"candidate caps {events.get('candidate_cap_cut', 0)} ({events.get('candidate_seats_cut', 0)} seats); "
            f"top-ups skipped {events.get('topup_skipped', 0)} / ran {events.get('topup_ran', 0)}; "
            f"events {dict(events)}; {calls or 'no model timing'}")
