import asyncio
from types import SimpleNamespace

from blue_machines_baseline.agent import attach_backchanneling
from blue_machines_baseline.backchannel import BackchannelEngine
from blue_machines_baseline.config import Settings
from blue_machines_baseline.events import EventRecorder


class FakeHandle:
    def __init__(self) -> None:
        self.interrupted = False

    def interrupt(self, *, force: bool = False) -> object:
        self.interrupted = True
        return self


class CompletingHandle(FakeHandle):
    def __init__(self) -> None:
        super().__init__()
        self.callback = None

    def add_done_callback(self, callback) -> None:
        self.callback = callback

    def complete(self) -> None:
        assert self.callback is not None
        self.callback(self)


class FakeSession:
    def __init__(self) -> None:
        self.callbacks: dict[str, object] = {}
        self.say_calls: list[dict[str, object]] = []
        self.handle = FakeHandle()

    def on(self, event: str, callback=None):
        def register(handler):
            self.callbacks[event] = handler
            return handler

        return register(callback) if callback is not None else register

    def say(self, text: str, **kwargs: object) -> FakeHandle:
        self.say_calls.append({"text": text, **kwargs})
        return self.handle


def test_backchannel_plays_after_delay_and_is_cancelled_on_floor_change() -> None:
    async def scenario() -> None:
        handle = FakeHandle()
        engine = BackchannelEngine(lambda: handle, delay_seconds=0.01, cooldown_seconds=0)

        engine.user_started()
        await asyncio.sleep(0.03)
        assert handle.interrupted is False

        engine.user_stopped()
        assert handle.interrupted is True
        await engine.aclose()

    asyncio.run(scenario())


def test_backchannel_does_not_play_after_short_user_turn() -> None:
    async def scenario() -> None:
        played = False

        def play() -> FakeHandle:
            nonlocal played
            played = True
            return FakeHandle()

        engine = BackchannelEngine(play, delay_seconds=0.03)
        engine.user_started()
        await asyncio.sleep(0.005)
        engine.user_stopped()
        await asyncio.sleep(0.04)

        assert played is False
        await engine.aclose()

    asyncio.run(scenario())


def test_backchannel_starts_when_agent_finishes_speaking() -> None:
    async def scenario() -> None:
        played = False

        def play() -> FakeHandle:
            nonlocal played
            played = True
            return FakeHandle()

        engine = BackchannelEngine(play, delay_seconds=0.01)
        engine.set_agent_busy(True)
        engine.user_started()
        await asyncio.sleep(0.02)
        assert played is False

        engine.set_agent_busy(False)
        await asyncio.sleep(0.03)
        assert played is True
        await engine.aclose()

    asyncio.run(scenario())


def test_backchannel_does_not_self_cancel_when_play_emits_agent_speaking() -> None:
    async def scenario() -> None:
        played = False
        engine: BackchannelEngine | None = None

        def play() -> FakeHandle:
            nonlocal played
            played = True
            assert engine is not None
            engine.set_agent_busy(True)
            return FakeHandle()

        engine = BackchannelEngine(play, delay_seconds=0.01)
        engine.user_started()
        await asyncio.sleep(0.03)

        assert played is True
        await engine.aclose()

    asyncio.run(scenario())


def test_backchannel_cooldown_blocks_repeated_turn_acknowledgements_until_elapsed() -> None:
    async def scenario() -> None:
        handles: list[FakeHandle] = []

        def play() -> FakeHandle:
            handle = FakeHandle()
            handles.append(handle)
            return handle

        engine = BackchannelEngine(play, delay_seconds=0.005, cooldown_seconds=0.05)
        engine.user_started()
        await asyncio.sleep(0.02)
        engine.user_stopped()

        engine.user_started()
        await asyncio.sleep(0.02)
        assert len(handles) == 1
        engine.user_stopped()

        await asyncio.sleep(0.06)
        engine.user_started()
        await asyncio.sleep(0.02)
        assert len(handles) == 2
        await engine.aclose()

    asyncio.run(scenario())


def test_backchannel_suppresses_an_acknowledgement_when_eot_is_likely() -> None:
    async def scenario() -> None:
        played = False

        def play() -> FakeHandle:
            nonlocal played
            played = True
            return FakeHandle()

        engine = BackchannelEngine(play, delay_seconds=0.02, eot_threshold=0.8)
        engine.user_started()
        engine.update_eot_probability(0.91)
        await asyncio.sleep(0.04)

        assert played is False
        await engine.aclose()

    asyncio.run(scenario())


def test_semantic_backchannel_waits_for_approval_then_uses_remaining_delay() -> None:
    async def scenario() -> None:
        played = False

        def play() -> FakeHandle:
            nonlocal played
            played = True
            return FakeHandle()

        engine = BackchannelEngine(
            play,
            delay_seconds=0.03,
            cooldown_seconds=0,
            semantic_required=True,
        )
        engine.user_started()
        await asyncio.sleep(0.02)
        assert played is False

        engine.set_semantic_approval(True)
        await asyncio.sleep(0.025)
        assert played is True
        await engine.aclose()

    asyncio.run(scenario())


def test_semantic_rejection_cancels_a_pending_acknowledgement() -> None:
    async def scenario() -> None:
        played = False

        def play() -> FakeHandle:
            nonlocal played
            played = True
            return FakeHandle()

        engine = BackchannelEngine(
            play,
            delay_seconds=0.02,
            cooldown_seconds=0,
            semantic_required=True,
        )
        engine.user_started()
        engine.set_semantic_approval(True)
        await asyncio.sleep(0.005)
        engine.set_semantic_approval(False)
        await asyncio.sleep(0.03)

        assert played is False
        await engine.aclose()

    asyncio.run(scenario())


def test_failed_tts_playback_does_not_leave_policy_stuck() -> None:
    async def scenario() -> None:
        attempts = 0

        def play() -> FakeHandle:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("TTS unavailable")
            return FakeHandle()

        engine = BackchannelEngine(play, delay_seconds=0.005, cooldown_seconds=0)
        engine.user_started()
        await asyncio.sleep(0.02)
        engine.user_stopped()
        engine.user_started()
        await asyncio.sleep(0.02)

        assert attempts == 2
        await engine.aclose()

    asyncio.run(scenario())


def test_rapid_floor_transitions_leave_only_the_latest_turn_eligible() -> None:
    async def scenario() -> None:
        handles: list[FakeHandle] = []

        def play() -> FakeHandle:
            handle = FakeHandle()
            handles.append(handle)
            return handle

        engine = BackchannelEngine(play, delay_seconds=0.015, cooldown_seconds=0)
        engine.user_started()
        await asyncio.sleep(0.004)
        engine.user_stopped()
        engine.user_started()
        await asyncio.sleep(0.004)
        engine.user_stopped()
        engine.user_started()
        await asyncio.sleep(0.03)

        assert len(handles) == 1
        await engine.aclose()

    asyncio.run(scenario())


def test_shutdown_interrupts_active_acknowledgement_and_prevents_late_playback() -> None:
    async def scenario() -> None:
        handle = FakeHandle()
        played = False

        def play() -> FakeHandle:
            nonlocal played
            played = True
            return handle

        engine = BackchannelEngine(play, delay_seconds=0.005)
        engine.user_started()
        await asyncio.sleep(0.02)
        await engine.aclose()
        await asyncio.sleep(0.02)

        assert played is True
        assert handle.interrupted is True

    asyncio.run(scenario())


def test_completed_acknowledgement_is_not_later_counted_as_cancelled() -> None:
    async def scenario() -> None:
        handle = CompletingHandle()
        events: list[str] = []
        engine = BackchannelEngine(
            lambda: handle,
            delay_seconds=0,
            cooldown_seconds=0,
            on_event=lambda name, **_data: events.append(name),
        )
        engine.user_started()
        await asyncio.sleep(0.01)
        handle.complete()
        engine.user_stopped()

        assert handle.interrupted is False
        assert "backchannel_completed" in events
        assert "backchannel_cancelled" not in events
        await engine.aclose()

    asyncio.run(scenario())


def test_semantic_approval_cannot_overlap_an_active_acknowledgement() -> None:
    async def scenario() -> None:
        handles: list[CompletingHandle] = []
        events: list[tuple[str, dict]] = []

        def play() -> CompletingHandle:
            handle = CompletingHandle()
            handles.append(handle)
            return handle

        engine = BackchannelEngine(
            play,
            delay_seconds=0,
            cooldown_seconds=0,
            semantic_required=True,
            max_acknowledgements_per_turn=2,
            on_event=lambda name, **data: events.append((name, data)),
        )
        engine.user_started()
        engine.set_semantic_approval(True)
        await asyncio.sleep(0.01)
        assert len(handles) == 1

        # A withdrawn approval must take back an audible cue: the handle is
        # interrupted with an explicit reason rather than left on the floor.
        engine.set_semantic_approval(False)
        assert handles[0].interrupted is True
        assert ("backchannel_cancelled", {"reason": "semantic_rejected"}) in events

        # Only after that cancellation may a new approval start a second cue;
        # re-approving while the first cue was active did not overlap them.
        engine.set_semantic_approval(True)
        await asyncio.sleep(0.01)
        assert len(handles) == 2
        assert handles[0].interrupted is True

        # A completed cue is never reported as cancelled, and the per-turn cap
        # still applies: two acknowledgements were used, so re-approving cannot
        # start a third one in this turn.
        handles[1].complete()
        engine.set_semantic_approval(False)
        engine.set_semantic_approval(True)
        await asyncio.sleep(0.01)
        assert len(handles) == 2
        assert handles[1].interrupted is False
        await engine.aclose()

    asyncio.run(scenario())


def test_livekit_hook_keeps_backchannel_out_of_chat_context() -> None:
    async def scenario() -> None:
        session = FakeSession()
        settings = Settings.from_env(
            {
                "LIVEKIT_URL": "wss://example.livekit.cloud",
                "LIVEKIT_API_KEY": "lk_api_key",
                "LIVEKIT_API_SECRET": "lk_api_secret",
                "GROQ_API_KEY": "groq_api_key",
                "GEMINI_API_KEY": "gemini_api_key",
                "BACKCHANNEL_TEXT": "mm-hmm",
                "BACKCHANNEL_DELAY_SECONDS": "0.01",
                "BACKCHANNEL_COOLDOWN_SECONDS": "0",
            }
        )
        recorder = EventRecorder()
        engine = attach_backchanneling(session, settings, recorder)

        session.callbacks["user_state_changed"](SimpleNamespace(new_state="speaking"))
        session.callbacks["agent_state_changed"](SimpleNamespace(new_state="thinking"))
        await asyncio.sleep(0.03)

        assert session.say_calls == [
            {"text": "mm-hmm", "allow_interruptions": False, "add_to_chat_ctx": False}
        ]

        session.callbacks["agent_state_changed"](SimpleNamespace(new_state="speaking"))
        assert session.handle.interrupted is False

        session.callbacks["user_state_changed"](SimpleNamespace(new_state="listening"))
        assert session.handle.interrupted is True
        await engine.aclose()

    asyncio.run(scenario())


def _engine_with_events(
    handle: FakeHandle, **kwargs: object
) -> tuple[BackchannelEngine, list[tuple[str, dict]]]:
    events: list[tuple[str, dict]] = []
    engine = BackchannelEngine(
        lambda: handle,
        delay_seconds=0,
        cooldown_seconds=0,
        on_event=lambda name, **data: events.append((name, data)),
        **kwargs,
    )
    return engine, events


def _reasons(events: list[tuple[str, dict]]) -> list[str]:
    return [data["reason"] for name, data in events if name == "backchannel_cancelled"]


def test_collision_is_reported_when_the_user_yields_right_after_a_cue_became_audible() -> None:
    async def scenario() -> None:
        handle = CompletingHandle()
        engine, events = _engine_with_events(handle, collision_window_seconds=0.5)
        engine.user_started()
        await asyncio.sleep(0.01)
        engine.mark_audio_started()
        engine.user_stopped()

        collisions = [data for name, data in events if name == "backchannel_collision"]
        assert len(collisions) == 1
        assert collisions[0]["window_ms"] == 500
        assert 0 <= collisions[0]["ms_since_audible"] <= 500
        # The cue is still stopped, and the interruption is attributed.
        assert handle.interrupted is True
        assert _reasons(events) == ["user_stopped"]
        await engine.aclose()

    asyncio.run(scenario())


def test_no_collision_when_the_user_yields_long_after_the_cue_became_audible() -> None:
    async def scenario() -> None:
        handle = CompletingHandle()
        engine, events = _engine_with_events(handle, collision_window_seconds=0.02)
        engine.user_started()
        await asyncio.sleep(0.01)
        engine.mark_audio_started()
        await asyncio.sleep(0.05)
        engine.user_stopped()

        assert [name for name, _ in events if name == "backchannel_collision"] == []
        assert _reasons(events) == ["user_stopped"]
        await engine.aclose()

    asyncio.run(scenario())


def test_a_cue_that_completed_is_never_reported_as_a_collision() -> None:
    async def scenario() -> None:
        handle = CompletingHandle()
        engine, events = _engine_with_events(handle, collision_window_seconds=0.5)
        engine.user_started()
        await asyncio.sleep(0.01)
        engine.mark_audio_started()
        handle.complete()
        engine.user_stopped()

        assert [name for name, _ in events if name == "backchannel_collision"] == []
        assert _reasons(events) == []
        await engine.aclose()

    asyncio.run(scenario())


def test_collision_is_reported_at_most_once_per_cue() -> None:
    async def scenario() -> None:
        handle = CompletingHandle()
        engine, events = _engine_with_events(handle, collision_window_seconds=1.0)
        engine.user_started()
        await asyncio.sleep(0.01)
        engine.mark_audio_started()
        engine.user_stopped()
        engine.user_stopped()

        assert [name for name, _ in events if name == "backchannel_collision"] == [
            "backchannel_collision"
        ]
        await engine.aclose()

    asyncio.run(scenario())


def test_every_reachable_cancellation_path_reports_its_reason() -> None:
    async def scenario() -> None:
        # `agent_busy` is deliberately absent: while a cue is audible, the
        # engine cannot tell the cue's own "agent speaking" event from a real
        # response, so it ignores that transition instead of self-cancelling
        # (covered by test_backchannel_does_not_self_cancel_when_play_emits_agent_speaking).
        cases = {
            "user_stopped": lambda engine: engine.user_stopped(),
            "disabled": lambda engine: engine.set_enabled(False),
            "shutdown": lambda engine: engine.aclose(),
            "semantic_rejected": lambda engine: engine.set_semantic_approval(False),
            "mode_switch": lambda engine: engine.set_semantic_required(True),
        }
        for expected, trigger in cases.items():
            handle = CompletingHandle()
            engine, events = _engine_with_events(handle)
            engine.user_started()
            await asyncio.sleep(0.01)
            engine.mark_audio_started()
            if expected == "semantic_rejected":
                engine.set_semantic_approval(True)
            result = trigger(engine)
            if result is not None:
                await result
            assert _reasons(events) == [expected], f"{expected} path reported {_reasons(events)}"

    asyncio.run(scenario())


def test_completed_cue_reports_its_audible_duration() -> None:
    async def scenario() -> None:
        handle = CompletingHandle()
        engine, events = _engine_with_events(handle)
        engine.user_started()
        await asyncio.sleep(0.01)
        engine.mark_audio_started()
        await asyncio.sleep(0.02)
        handle.complete()

        completed = [data for name, data in events if name == "backchannel_completed"]
        assert len(completed) == 1
        assert completed[0]["duration_ms"] >= 20
        await engine.aclose()

    asyncio.run(scenario())


def test_cue_that_never_became_audible_reports_no_duration() -> None:
    async def scenario() -> None:
        handle = CompletingHandle()
        engine, events = _engine_with_events(handle)
        engine.user_started()
        await asyncio.sleep(0.01)
        handle.complete()

        completed = [data for name, data in events if name == "backchannel_completed"]
        assert len(completed) == 1
        assert completed[0]["duration_ms"] is None
        await engine.aclose()

    asyncio.run(scenario())


class RotatingSession(FakeSession):
    """A session whose cues can finish, so several can play in one test."""

    def __init__(self) -> None:
        super().__init__()
        self.handles: list[CompletingHandle] = []

    def say(self, text: str, **kwargs: object) -> CompletingHandle:
        self.say_calls.append({"text": text, **kwargs})
        handle = CompletingHandle()
        self.handles.append(handle)
        return handle


def test_the_timer_policy_rotates_through_its_configured_cues() -> None:
    """Without a classifier to choose, the timer policy still varies its cue.

    One sound repeated for a whole conversation is what makes an acknowledgement
    mechanical, so the configured cues are used in rotation.
    """

    async def scenario() -> None:
        session = RotatingSession()
        settings = Settings.from_env(
            {
                "LIVEKIT_URL": "wss://example.livekit.cloud",
                "LIVEKIT_API_KEY": "lk_api_key",
                "LIVEKIT_API_SECRET": "lk_api_secret",
                "GROQ_API_KEY": "groq_api_key",
                "GEMINI_API_KEY": "gemini_api_key",
                "BACKCHANNEL_TEXT": "mm-hmm,mm,hmm",
                "BACKCHANNEL_DELAY_SECONDS": "0.01",
                "BACKCHANNEL_COOLDOWN_SECONDS": "0",
            }
        )
        engine = attach_backchanneling(session, settings, EventRecorder())

        for _ in range(3):
            session.callbacks["user_state_changed"](SimpleNamespace(new_state="speaking"))
            session.callbacks["agent_state_changed"](SimpleNamespace(new_state="thinking"))
            await asyncio.sleep(0.03)
            session.handles[-1].complete()
            session.callbacks["user_state_changed"](SimpleNamespace(new_state="listening"))
            session.callbacks["agent_state_changed"](SimpleNamespace(new_state="listening"))

        assert [call["text"] for call in session.say_calls] == ["mm-hmm", "mm", "hmm"]
        await engine.aclose()

    asyncio.run(scenario())


def test_a_timer_policy_with_one_cue_still_repeats_it() -> None:
    """The default configuration and the committed evidence use a single cue."""

    async def scenario() -> None:
        session = RotatingSession()
        settings = Settings.from_env(
            {
                "LIVEKIT_URL": "wss://example.livekit.cloud",
                "LIVEKIT_API_KEY": "lk_api_key",
                "LIVEKIT_API_SECRET": "lk_api_secret",
                "GROQ_API_KEY": "groq_api_key",
                "GEMINI_API_KEY": "gemini_api_key",
                "BACKCHANNEL_DELAY_SECONDS": "0.01",
                "BACKCHANNEL_COOLDOWN_SECONDS": "0",
            }
        )
        engine = attach_backchanneling(session, settings, EventRecorder())

        for _ in range(2):
            session.callbacks["user_state_changed"](SimpleNamespace(new_state="speaking"))
            session.callbacks["agent_state_changed"](SimpleNamespace(new_state="thinking"))
            await asyncio.sleep(0.03)
            session.handles[-1].complete()
            session.callbacks["user_state_changed"](SimpleNamespace(new_state="listening"))
            session.callbacks["agent_state_changed"](SimpleNamespace(new_state="listening"))

        assert [call["text"] for call in session.say_calls] == ["mm-hmm", "mm-hmm"]
        await engine.aclose()

    asyncio.run(scenario())
