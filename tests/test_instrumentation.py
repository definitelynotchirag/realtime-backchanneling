import asyncio
from types import SimpleNamespace

from blue_machines_baseline.agent import attach_backchanneling, attach_instrumentation
from blue_machines_baseline.events import EventRecorder


class FakeSession:
    def __init__(self) -> None:
        self.callbacks = {}
        self.say_calls = []

    def on(self, event_name):
        def register(callback):
            self.callbacks[event_name] = callback
            return callback

        return register

    def say(self, text, **kwargs):
        self.say_calls.append((text, kwargs))
        return SimpleNamespace(interrupt=lambda **_kwargs: None)


class FakeBackchannel:
    active = True


def test_instrumentation_records_speech_boundaries_and_the_transcript_text() -> None:
    """The transcript text is recorded, reversing an earlier deliberate omission.

    The event carried only counts so that user speech never reached the log, and the
    console's transcript panel has been empty ever since - it asks for a field the worker
    never sent. The panel is the reason the text is kept now. Worth knowing before this
    runs on a real conversation: the live log is gitignored, and the committed evidence
    holds scripted lines only.
    """
    session = FakeSession()
    recorder = EventRecorder()
    attach_instrumentation(session, recorder)

    session.callbacks["user_state_changed"](
        SimpleNamespace(old_state="listening", new_state="speaking")
    )
    session.callbacks["user_input_transcribed"](
        SimpleNamespace(transcript="a short sentence", is_final=True)
    )
    session.callbacks["user_state_changed"](
        SimpleNamespace(old_state="speaking", new_state="listening")
    )
    session.callbacks["agent_state_changed"](
        SimpleNamespace(old_state="thinking", new_state="speaking")
    )
    session.callbacks["agent_state_changed"](
        SimpleNamespace(old_state="speaking", new_state="idle")
    )

    names = [event.name for event in recorder.events]
    assert names == [
        "user_speech_started",
        "stt_transcript",
        "user_speech_ended",
        "agent_response_started",
        "agent_response_ended",
    ]
    assert recorder.events[1].data == {
        "transcript": "a short sentence",
        "is_final": True,
        "character_count": len("a short sentence"),
        "word_count": 3,
    }


def test_instrumentation_ignores_late_metrics_after_recorder_close() -> None:
    session = FakeSession()
    recorder = EventRecorder()
    attach_instrumentation(session, recorder)
    recorder.close()

    session.callbacks["metrics_collected"](
        SimpleNamespace(metrics=SimpleNamespace(type="llm", ttft=0.1))
    )

    assert recorder.events == ()


def test_instrumentation_exposes_eot_signal_alongside_provider_metric() -> None:
    session = FakeSession()
    recorder = EventRecorder()
    attach_instrumentation(session, recorder)

    session.callbacks["metrics_collected"](
        SimpleNamespace(
            metrics=SimpleNamespace(
                type="eou_metrics",
                end_of_utterance_delay=0.15,
                transcription_delay=0.04,
            )
        )
    )

    assert [event.name for event in recorder.events] == ["pipeline_metric", "eot_signal"]
    assert recorder.events[1].data["end_of_utterance_delay"] == 0.15


def test_backchannel_speech_is_not_counted_as_an_assistant_response() -> None:
    session = FakeSession()
    recorder = EventRecorder()
    attach_instrumentation(session, recorder, FakeBackchannel())

    session.callbacks["agent_state_changed"](
        SimpleNamespace(old_state="thinking", new_state="speaking")
    )

    assert [event.name for event in recorder.events] == ["backchannel_agent_speaking"]


def test_final_public_transcript_suppresses_pending_backchannel() -> None:
    async def run() -> list[str]:
        session = FakeSession()
        recorder = EventRecorder()

        attach_backchanneling(
            session,
            SimpleNamespace(
                backchannel_texts=("mm-hmm",),
                backchannel_delay_seconds=1.4,
                backchannel_cooldown_seconds=4.0,
                collision_window_seconds=0.5,
                eot_detector="final_transcript",
            ),
            recorder,
            enabled=True,
        )
        session.callbacks["user_state_changed"](
            SimpleNamespace(old_state="listening", new_state="speaking")
        )
        session.callbacks["user_state_changed"](
            SimpleNamespace(old_state="speaking", new_state="listening")
        )
        session.callbacks["user_input_transcribed"](
            SimpleNamespace(transcript="finished thought", is_final=True)
        )
        await asyncio.sleep(0)
        return [event.name for event in recorder.events]

    assert asyncio.run(run()) == ["backchannel_suppressed_eot"]


def test_final_streaming_segment_does_not_end_a_still_active_user_turn() -> None:
    async def run() -> list[str]:
        session = FakeSession()
        recorder = EventRecorder()

        engine = attach_backchanneling(
            session,
            SimpleNamespace(
                backchannel_texts=("mm-hmm",),
                backchannel_delay_seconds=0.01,
                backchannel_cooldown_seconds=0.0,
                collision_window_seconds=0.5,
                eot_detector="final_transcript",
            ),
            recorder,
            enabled=True,
        )
        session.callbacks["user_state_changed"](
            SimpleNamespace(old_state="listening", new_state="speaking")
        )
        session.callbacks["user_input_transcribed"](
            SimpleNamespace(transcript="one completed streaming segment", is_final=True)
        )
        await asyncio.sleep(0.03)
        await engine.aclose()
        return [event.name for event in recorder.events]

    names = asyncio.run(run())
    assert "backchannel_suppressed_eot" not in names
    assert "backchannel_started" in names
