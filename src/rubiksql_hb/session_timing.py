"""Bounded live timing for agent sessions and their shared-loop submission.

Only elapsed monotonic durations and bounded operation names are retained. Nested
role scopes contribute to the shared submission while sibling tasks keep separate
role counters. Replayed responses never create model measurements.

``model`` is a whole chat turn as the caller waits for it. The lent file-key route also
splits each physical request into its phases, by model (:func:`record_model_phase`):
``model_admission`` (waiting for a key slot, plus configured pacing), ``model_wire`` (the
HTTP request from send to response) and ``model_release`` (returning the slot).
"""

from __future__ import annotations

from _thread import LockType
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import threading
import time
from typing import Iterator, Literal


SCHEMA = "rubiksql.session-timing/1"
MAX_NAMES = 64
MAX_ACTIVE = 256
_OTHER = "__other__"
Category = Literal["model", "tool", "gate", "queue_wait", "model_admission", "model_wire", "model_release"]
#: Physical-request phases of one model call; each is also counted per model name.
MODEL_PHASES = ("admission", "wire", "release")
_CATEGORIES = ("model", "tool", "gate", "queue_wait", *(f"model_{phase}" for phase in MODEL_PHASES))
_NAMED = ("tool", "gate", *(f"model_{phase}" for phase in MODEL_PHASES))


@dataclass
class _Latency:
    count: int = 0
    sum_sec: float = 0.0
    max_sec: float = 0.0

    def add(self, elapsed: float) -> None:
        self.count += 1
        self.sum_sec += elapsed
        self.max_sec = max(self.max_sec, elapsed)

    def snapshot(self) -> dict[str, int | float]:
        return {"count": self.count, "sum_sec": round(self.sum_sec, 6), "max_sec": round(self.max_sec, 6)}


class SessionTiming:
    """Collect one scope's wall time and aggregate operation latencies.

    Args:
        role: Candidate, judge, reviewer, or shared-session scope being measured.
    """

    def __init__(self, role: str) -> None:
        self.role = role
        self._started = time.monotonic()
        self._finished: float | None = None
        self._lock = threading.Lock()
        self._totals = {name: _Latency() for name in _CATEGORIES}
        self._names: dict[str, dict[str, _Latency]] = {name: {} for name in _NAMED}
        self._active: dict[object, tuple[Category, str, float]] = {}
        self._untracked_active = 0

    @staticmethod
    def _add_named(counters: dict[str, _Latency], name: str, elapsed: float) -> None:
        key = name[:128]
        if key not in counters and len(counters) >= MAX_NAMES:
            key = _OTHER
        counters.setdefault(key, _Latency()).add(elapsed)

    def _record(self, category: Category, elapsed: float, name: str) -> None:
        self._totals[category].add(elapsed)
        if category in self._names:
            self._add_named(self._names[category], name, elapsed)

    def record(self, category: Category, elapsed: float, name: str = "") -> None:
        """Add one completed operation, including failures and cancellation.

        Args:
            category: Model request, MCP tool round trip, gate wait, loop queue wait, or one
                physical-request phase of a model call.
            elapsed: Real elapsed seconds measured with the monotonic clock.
            name: Tool, gate or model name; ignored for unnamed model and queue counters.
        """
        elapsed = max(0.0, elapsed)
        with self._lock:
            self._record(category, elapsed, name)

    def start_span(self, category: Category, started: float, name: str = "") -> object | None:
        """Start a bounded in-flight measurement so a stalled operation is visible.

        Args:
            category: Aggregate to receive the measurement.
            started: Monotonic start time.
            name: Tool or gate name, when applicable.

        Returns:
            Span token, or None when the active-span bound is reached. Completion
            still records the exact elapsed duration for an untracked span.
        """
        with self._lock:
            if len(self._active) >= MAX_ACTIVE:
                self._untracked_active += 1
                return None
            token = object()
            self._active[token] = category, name[:128], started
            return token

    def end_span(self, token: object | None, category: Category, elapsed: float, name: str = "") -> None:
        """Complete a span atomically, preventing snapshot double counting.

        Args:
            token: Token returned by :meth:`start_span`.
            category: Aggregate receiving the completed measurement.
            elapsed: Real elapsed seconds since the span began.
            name: Tool or gate name, when applicable.
        """
        with self._lock:
            if token is None:
                self._untracked_active -= 1
            else:
                self._active.pop(token, None)
            self._record(category, max(0.0, elapsed), name)

    def finish(self) -> None:
        """Freeze wall time when this scope exits; repeated calls are harmless."""
        with self._lock:
            if self._finished is None:
                self._finished = time.monotonic()

    def snapshot(self) -> dict:
        """Return detached, JSON-shaped live measurements for this scope.

        Returns:
            The versioned timing schema with wall time, count/sum/max aggregates,
            bounded per-tool and per-gate maps, and the per-model request phases
            (``models``: model -> admission/wire/release). In-flight operations
            contribute their elapsed duration once; completion replaces that partial duration.
        """
        with self._lock:
            end = self._finished if self._finished is not None else time.monotonic()
            totals = {name: _Latency(counter.count, counter.sum_sec, counter.max_sec) for name, counter in self._totals.items()}
            names = {category: {name: _Latency(counter.count, counter.sum_sec, counter.max_sec)
                                for name, counter in counters.items()} for category, counters in self._names.items()}
            for category, name, started in self._active.values():
                elapsed = max(0.0, end - started)
                totals[category].add(elapsed)
                if category in names:
                    self._add_named(names[category], name, elapsed)
            return {
                "schema": SCHEMA,
                "role": self.role,
                "wall_sec": round(max(0.0, end - self._started), 6),
                **{name: counter.snapshot() for name, counter in totals.items()},
                "tools": {name: counter.snapshot() for name, counter in names["tool"].items()},
                "gates": {name: counter.snapshot() for name, counter in names["gate"].items()},
                "models": _model_phases(names),
                "in_flight": len(self._active) + self._untracked_active,
                "in_flight_untracked": self._untracked_active,
            }


def _model_phases(names: dict[str, dict[str, _Latency]]) -> dict[str, dict[str, dict[str, int | float]]]:
    """Pivot the per-phase name maps into ``{model: {phase: count/sum/max}}``, every phase present."""
    models = sorted({model for phase in MODEL_PHASES for model in names[f"model_{phase}"]})
    return {model: {phase: names[f"model_{phase}"].get(model, _Latency()).snapshot() for phase in MODEL_PHASES}
            for model in models}


_ACTIVE: ContextVar[tuple[SessionTiming, ...]] = ContextVar("rubiksql_session_timing", default=())


@contextmanager
def scope(role: str, *, timing: SessionTiming | None = None) -> Iterator[SessionTiming]:
    """Bind a fresh timing scope to this task, preserving ancestor aggregation.

    Args:
        role: Role whose operations this scope records.
        timing: Existing collector when submission wall time starts before task entry.

    Yields:
        Collector whose snapshot can be attached to a result or failure record.
    """
    timing = SessionTiming(role) if timing is None else timing
    token = _ACTIVE.set((*_ACTIVE.get(), timing))
    try:
        yield timing
    finally:
        timing.finish()
        _ACTIVE.reset(token)


def current() -> SessionTiming | None:
    """Return the innermost scope, or None outside a measured session."""
    active = _ACTIVE.get()
    return active[-1] if active else None


def current_snapshot() -> dict | None:
    """Return a detached snapshot of the current scope, if one is active."""
    timing = current()
    return timing.snapshot() if timing is not None else None


@contextmanager
def measure(category: Category, name: str = "") -> Iterator[None]:
    """Measure an operation in every enclosing scope without retaining samples.

    Args:
        category: Model request, tool round trip, gate wait, or loop queue wait.
        name: Tool or gate name, when applicable.

    Yields:
        None. The original return value or exception is unaffected.
    """
    active = _ACTIVE.get()
    started = time.monotonic()
    spans = [(timing, timing.start_span(category, started, name)) for timing in active]
    try:
        yield
    finally:
        elapsed = time.monotonic() - started
        for timing, token in spans:
            timing.end_span(token, category, elapsed, name)


def record_model_phase(phase: str, model: str, elapsed: float) -> None:
    """Add one physical model request's phase to every enclosing scope, by model.

    Args:
        phase: ``admission``, ``wire`` or ``release`` (:data:`MODEL_PHASES`).
        model: Requested model name; bounded like tool names.
        elapsed: Real elapsed seconds measured with the monotonic clock.

    Raises:
        ValueError: Unknown phase.
    """
    if phase not in MODEL_PHASES:
        raise ValueError(f"unknown model request phase: {phase}")
    for timing in _ACTIVE.get():
        timing.record(f"model_{phase}", elapsed, model)  # type: ignore[arg-type]


@contextmanager
def locked(lock: LockType, name: str) -> Iterator[None]:
    """Acquire an existing lock while measuring only its admission wait.

    Args:
        lock: Existing lock whose ownership and release policy are preserved.
        name: Bounded diagnostic gate name.

    Yields:
        None while the caller owns the lock.
    """
    with measure("gate", name):
        lock.acquire()
    try:
        yield
    finally:
        lock.release()
