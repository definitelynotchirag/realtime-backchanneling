import json
from datetime import UTC, datetime

import pytest

from blue_machines_baseline.events import EventRecorder


def test_recorder_keeps_order_and_monotonic_elapsed_time(tmp_path) -> None:
    now = [100.0]
    wall = [datetime(2026, 1, 1, tzinfo=UTC)]

    def clock() -> float:
        return now[0]

    def wall_clock() -> datetime:
        return wall[0]

    path = tmp_path / "events.jsonl"
    recorder = EventRecorder(path, clock=clock, wall_clock=wall_clock)
    recorder.record("user_speech_started")
    now[0] += 0.250
    recorder.record("user_speech_ended", next_state="listening")
    recorder.close()

    assert [event.name for event in recorder.events] == [
        "user_speech_started",
        "user_speech_ended",
    ]
    assert recorder.events[1].elapsed_ms == 250.0
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["name"] == "user_speech_started"


def test_recorder_rejects_events_after_close() -> None:
    recorder = EventRecorder()
    recorder.close()

    with pytest.raises(RuntimeError, match="after the recorder is closed"):
        recorder.record("session_stopped")


def test_recorder_close_is_idempotent(tmp_path) -> None:
    recorder = EventRecorder(tmp_path / "events.jsonl")
    recorder.record("session_started")
    recorder.close()
    recorder.close()

    assert recorder.closed


def test_recorder_stamps_experiment_labels_on_every_jsonl_event() -> None:
    recorder = EventRecorder(scenario_id="middle_pause", mode="backchannel", run_id="r1")
    event = recorder.record("session_started")

    assert event.as_dict()["scenario_id"] == "middle_pause"
    assert event.as_dict()["mode"] == "backchannel"
    assert event.as_dict()["run_id"] == "r1"
