"""Small tracing contracts that keep observability optional."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from time import time
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class TraceEvent:
    run_id: str
    kind: str
    step: int
    details: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time)


class TraceSink(Protocol):
    def emit(self, event: TraceEvent) -> None: ...


class InMemoryTraceSink:
    """Thread-safe trace sink for tests, demos, and local inspection."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._lock = Lock()

    def emit(self, event: TraceEvent) -> None:
        with self._lock:
            self._events.append(event)

    def events(self) -> tuple[TraceEvent, ...]:
        with self._lock:
            return tuple(self._events)

