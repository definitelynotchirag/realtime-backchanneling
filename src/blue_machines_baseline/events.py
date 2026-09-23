"""Small, dependency-light JSONL lifecycle recorder."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Clock = Callable[[], float]
WallClock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class LifecycleEvent:
    """One event in a baseline session timeline."""

    name: str
    timestamp: str
    elapsed_ms: float
    data: Mapping[str, Any]
    scenario_id: str | None = None
    mode: str | None = None
    run_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "timestamp": self.timestamp,
            "elapsed_ms": self.elapsed_ms,
            "data": dict(self.data),
        }
        if self.scenario_id is not None:
            result["scenario_id"] = self.scenario_id
        if self.mode is not None:
            result["mode"] = self.mode
        if self.run_id is not None:
            result["run_id"] = self.run_id
        return result


class EventRecorder:
    """Record structured events in memory and, optionally, append them to JSONL."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        clock: Clock = time.monotonic,
        wall_clock: WallClock = _utc_now,
        scenario_id: str | None = None,
        mode: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self._path = path
        self._clock = clock
        self._wall_clock = wall_clock
        self._started_at = clock()
        self._scenario_id = scenario_id
        self._mode = mode
        self._run_id = run_id
        self._events: list[LifecycleEvent] = []
        self._lock = threading.Lock()
        self._file = None
        self._closed = False
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("a", encoding="utf-8")

    @property
    def events(self) -> tuple[LifecycleEvent, ...]:
        with self._lock:
            return tuple(self._events)

    @property
    def closed(self) -> bool:
        return self._closed

    def record(self, name: str, **data: Any) -> LifecycleEvent:
        """Record an event with a monotonic offset and an ISO-8601 UTC timestamp."""

        if not name.strip():
            raise ValueError("event name must not be empty")
        event = LifecycleEvent(
            name=name,
            timestamp=self._wall_clock().isoformat(),
            elapsed_ms=round((self._clock() - self._started_at) * 1000, 3),
            data=data,
            scenario_id=self._scenario_id,
            mode=self._mode,
            run_id=self._run_id,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot record an event after the recorder is closed")
            self._events.append(event)
            if self._file is not None:
                self._file.write(json.dumps(event.as_dict(), sort_keys=True) + "\n")
                self._file.flush()
        return event

    def close(self) -> None:
        """Flush and close the file; safe to call more than once."""

        with self._lock:
            if self._closed:
                return
            if self._file is not None:
                self._file.flush()
                self._file.close()
            self._closed = True

    def __enter__(self) -> EventRecorder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
