"""Runnable Phase 1 LiveKit voice-agent baseline."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference, room_io
from livekit.plugins import elevenlabs, google, groq, openai, silero
from typesafe_sdk import AsyncTypeSafeClient

from . import gemini_tts, groq_interim_stt
from .backchannel import BackchannelEngine
from .benchmark import parse_run_context
from .config import JEV_INTERIM_STT_ERROR, ConfigurationError, Settings
from .eot_detector import EotDetector
from .events import EventRecorder
from .jev import JevClassifier, JevTurnController

logger = logging.getLogger("blue-machines-baseline")


def load_backchannel_clips() -> dict[str, rtc.AudioFrame]:
    """Load pre-generated acknowledgement clips once per worker process."""

    clip_dir = Path(__file__).resolve().parents[2] / "assets" / "backchannels"
    clips: dict[str, rtc.AudioFrame] = {}
    for cue_text, filename in {
        "mm-hmm": "mm-hmm.wav",
        "uh-huh": "uh-huh.wav",
        "I see": "i-see.wav",
    }.items():
        path = clip_dir / filename
        if not path.exists():
            continue
        with wave.open(str(path), "rb") as source:
            clips[cue_text] = rtc.AudioFrame(
                data=source.readframes(source.getnframes()),
                sample_rate=source.getframerate(),
                num_channels=source.getnchannels(),
                samples_per_channel=source.getnframes(),
            )
    return clips


async def _cached_clip_stream(frame: rtc.AudioFrame) -> AsyncIterator[rtc.AudioFrame]:
    yield frame


class BaselineAssistant(Agent):
    """A deliberately small assistant used as the normal-agent comparison point."""

    def __init__(self, *, instructions: str, greet: bool = True) -> None:
        super().__init__(instructions=instructions)
        # A scripted benchmark run knows what the user is about to say, and every
        # greeting costs a speech request as well as wall-clock time. Callers can
        # turn it off; the live demo keeps it.
        self._greet = greet

    async def on_enter(self) -> None:
        if not self._greet:
            return
        await self.session.generate_reply(
            instructions="Greet the user briefly and ask how you can help today."
        )


def _metric_summary(metric: Any) -> dict[str, Any]:
    """Keep useful scalar timing fields without serializing provider internals."""

    fields = (
        "type",
        "duration",
        "ttft",
        "ttfb",
        "eou_delay",
        "end_of_utterance_delay",
        "transcription_delay",
        "on_user_turn_completed_delay",
        "total_duration",
        "detection_delay",
        "prediction_duration",
        "audio_duration",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "characters_count",
    )
    result: dict[str, Any] = {"metric_type": type(metric).__name__}
    for field in fields:
        value = getattr(metric, field, None)
        if isinstance(value, (str, int, float, bool)) or value is None:
            if value is not None:
                result[field] = value
    return result


def create_stt(settings: Settings) -> inference.STT | groq.STT | groq_interim_stt.STT:
    """Create streaming LiveKit STT, with Groq retained as a final-only fallback."""

    if settings.stt_provider == "livekit_inference":
        return inference.STT(
            model=settings.livekit_stt_model,
            language="en",
            api_key=settings.livekit_api_key.get_secret_value(),
            api_secret=settings.livekit_api_secret.get_secret_value(),
        )

    if settings.groq_api_key is None:
        raise ConfigurationError(
            f"GROQ_API_KEY is required when STT_PROVIDER={settings.stt_provider}"
        )
    if settings.stt_provider == "groq_interim":
        return groq_interim_stt.STT(
            model=settings.groq_stt_model,
            api_key=settings.groq_api_key.get_secret_value(),
            interim_interval_seconds=settings.groq_interim_interval_seconds,
        )
    return groq.STT(
        model=settings.groq_stt_model,
        api_key=settings.groq_api_key.get_secret_value(),
    )


def create_llm(settings: Settings) -> google.LLM | openai.LLM | groq.LLM:
    """Create the configured Gemini or OpenRouter LLM client."""

    if settings.llm_provider == "gemini":
        if settings.gemini_api_key is None:
            raise ConfigurationError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        return google.LLM(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key.get_secret_value(),
        )

    if settings.llm_provider == "groq":
        if settings.groq_api_key is None:
            raise ConfigurationError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        return groq.LLM(
            model=settings.groq_llm_model,
            api_key=settings.groq_api_key.get_secret_value(),
        )

    if settings.openrouter_api_key is None:
        raise ConfigurationError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")

    return openai.LLM.with_openrouter(
        model=settings.openrouter_model,
        api_key=settings.openrouter_api_key.get_secret_value(),
        base_url=settings.openrouter_base_url,
        fallback_models=list(settings.openrouter_fallback_models) or None,
    )


def create_tts(
    settings: Settings,
) -> inference.TTS | elevenlabs.TTS | gemini_tts.TTS | groq.TTS:
    """Create LiveKit Inference TTS by default, with explicit direct providers."""

    if settings.tts_provider == "livekit_inference":
        return inference.TTS(
            model=settings.livekit_tts_model,
            voice=settings.livekit_tts_voice,
            language="en",
            api_key=settings.livekit_api_key.get_secret_value(),
            api_secret=settings.livekit_api_secret.get_secret_value(),
        )

    if settings.tts_provider == "groq_tts":
        if settings.groq_api_key is None:
            raise ConfigurationError("GROQ_API_KEY is required when TTS_PROVIDER=groq_tts")
        # Native Groq speech (Orpheus). Groq blocks it until the account accepts
        # the model's terms of use, which surfaces as a clear 400 from the API.
        return groq.TTS(
            model=settings.groq_tts_model,
            voice=settings.groq_tts_voice,
            api_key=settings.groq_api_key.get_secret_value(),
        )

    if settings.tts_provider == "gemini_tts":
        if settings.gemini_api_key is None:
            raise ConfigurationError("GEMINI_API_KEY is required when TTS_PROVIDER=gemini_tts")
        return gemini_tts.TTS(
            model=settings.gemini_tts_model,
            voice=settings.gemini_tts_voice,
            api_key=settings.gemini_api_key.get_secret_value(),
        )

    if settings.elevenlabs_api_key is None:
        raise ConfigurationError("ELEVENLABS_API_KEY is required when TTS_PROVIDER=elevenlabs")
    return elevenlabs.TTS(
        voice_id=settings.elevenlabs_voice_id,
        model=settings.elevenlabs_tts_model,
        api_key=settings.elevenlabs_api_key.get_secret_value(),
    )


def attach_instrumentation(
    session: AgentSession,
    recorder: EventRecorder,
    backchannel: BackchannelEngine | None = None,
) -> None:
    """Attach public AgentSession listeners for the Phase 1 baseline timeline."""

    def record_if_open(name: str, **data: Any) -> None:
        """Ignore provider events that arrive after LiveKit begins shutdown."""

        if recorder.closed:
            return
        try:
            recorder.record(name, **data)
        except RuntimeError:
            # A provider callback can race with the recorder's close operation.
            # Late telemetry should never make LiveKit report an event failure.
            if not recorder.closed:
                raise

    @session.on("user_state_changed")
    def _on_user_state_changed(event: Any) -> None:
        old_state = getattr(event, "old_state", None)
        new_state = getattr(event, "new_state", None)
        if new_state == "speaking":
            record_if_open("user_speech_started", previous_state=old_state)
        elif old_state == "speaking" and new_state in {"listening", "away"}:
            record_if_open("user_speech_ended", next_state=new_state)

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(event: Any) -> None:
        transcript = getattr(event, "transcript", "") or ""
        record_if_open(
            "stt_transcript",
            is_final=bool(getattr(event, "is_final", False)),
            character_count=len(transcript),
            word_count=len(transcript.split()),
        )

    @session.on("agent_state_changed")
    def _on_agent_state_changed(event: Any) -> None:
        old_state = getattr(event, "old_state", None)
        new_state = getattr(event, "new_state", None)
        if new_state == "speaking" and old_state != "speaking":
            record_if_open(
                "backchannel_agent_speaking"
                if backchannel and backchannel.active
                else "agent_response_started",
                previous_state=old_state,
            )
        elif old_state == "speaking" and new_state != "speaking":
            record_if_open("agent_response_ended", next_state=new_state)

    @session.on("metrics_collected")
    def _on_metrics_collected(event: Any) -> None:
        metric = getattr(event, "metrics", event)
        summary = _metric_summary(metric)
        record_if_open("pipeline_metric", **summary)
        metric_type = str(summary.get("type", summary.get("metric_type", ""))).lower()
        if "eou" in metric_type or "eot" in metric_type:
            record_if_open("eot_signal", **summary)

    @session.on("error")
    def _on_error(event: Any) -> None:
        error = getattr(event, "error", event)
        record_if_open("session_error", error_type=type(error).__name__)


def attach_backchanneling(
    session: AgentSession,
    settings: Settings,
    recorder: EventRecorder,
    *,
    enabled: bool = True,
    semantic_required: bool = False,
    jev_controller_ref: list[JevTurnController] | None = None,
    eot_detector_ref: list[EotDetector] | None = None,
    cue_state: dict[str, str] | None = None,
    locked_mode: str | None = None,
    room: Any | None = None,
    cached_clips: dict[str, rtc.AudioFrame] | None = None,
) -> BackchannelEngine:
    """Attach the experimental acknowledgement policy to a running session."""

    def record_event(name: str, **data: Any) -> None:
        if recorder.closed:
            return
        try:
            recorder.record(name, **data)
        except RuntimeError:
            if not recorder.closed:
                raise
        logger.debug("backchannel event: %s", name)

    selected_cue = cue_state if cue_state is not None else {"text": settings.backchannel_text}

    def play_cue() -> Any:
        cue_text = selected_cue["text"]
        clip = cached_clips.get(cue_text) if cached_clips else None
        kwargs: dict[str, Any] = {
            # LiveKit delays interruptible speech until user silence. A backchannel
            # must play over ongoing user speech, so the engine force-interrupts it
            # itself when the user yields the floor.
            "allow_interruptions": False,
            "add_to_chat_ctx": False,
        }
        if clip is not None:
            kwargs["audio"] = _cached_clip_stream(clip)
        return session.say(cue_text, **kwargs)

    engine = BackchannelEngine(
        play_cue,
        delay_seconds=settings.backchannel_delay_seconds,
        # Jev mode can safely acknowledge more often because it already gates
        # each cue semantically; keep the timer-only policy unchanged.
        cooldown_seconds=(
            min(settings.backchannel_cooldown_seconds, 1.8)
            if semantic_required
            else settings.backchannel_cooldown_seconds
        ),
        enabled=enabled,
        semantic_required=semantic_required,
        max_acknowledgements_per_turn=5 if semantic_required else 1,
        collision_window_seconds=settings.collision_window_seconds,
        on_event=record_event,
    )

    def current_jev_controller() -> JevTurnController | None:
        return jev_controller_ref[0] if jev_controller_ref else None

    user_is_speaking = False

    detector: EotDetector | None = None
    detector_tasks: list[asyncio.Task[Any]] = []
    if settings.eot_detector == "livekit_inference" and room is not None:

        def on_eot_prediction(probability: float, threshold: float | None) -> None:
            record_event(
                "eot_prediction",
                probability=round(probability, 4),
                # The detector's own boundary when it publishes one, otherwise the
                # threshold the policy actually compares against.
                threshold=threshold if threshold is not None else engine.eot_threshold,
                source="turn_detector",
                model=detector.model,
            )
            engine.update_eot_probability(probability)

        def on_eot_unavailable(reason: str) -> None:
            record_event("eot_detector_unavailable", reason=reason)

        detector = EotDetector(
            on_prediction=on_eot_prediction,
            on_unavailable=on_eot_unavailable,
        )
        if eot_detector_ref is not None:
            eot_detector_ref.append(detector)

        async def feed_user_audio(track: Any) -> None:
            stream = rtc.AudioStream(track)
            try:
                async for event in stream:
                    if detector is not None:
                        detector.push_audio(event.frame)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - the audio tap must never break the room
                logger.debug("eot audio tap stopped", exc_info=True)

        @room.on("track_subscribed")
        def _on_track_subscribed(track: Any, _publication: Any, participant: Any) -> None:
            if track.kind != rtc.TrackKind.KIND_AUDIO:
                return
            if participant.identity == room.local_participant.identity:
                return
            detector_tasks.append(asyncio.create_task(feed_user_audio(track)))

        async def start_detector() -> None:
            if await detector.start() and user_is_speaking:
                detector.begin_turn()
            if detector.threshold is not None:
                engine.set_eot_threshold(detector.threshold)

        detector_tasks.append(asyncio.create_task(start_detector()))

    @session.on("user_state_changed")
    def _on_user_state_changed(event: Any) -> None:
        nonlocal user_is_speaking
        new_state = getattr(event, "new_state", None)
        if new_state == "speaking":
            user_is_speaking = True
            engine.user_started()
            if detector is not None:
                detector.begin_turn()
            if (controller := current_jev_controller()) is not None:
                controller.user_started()
        elif new_state in {"listening", "away"}:
            user_is_speaking = False
            if detector is not None:
                detector.end_turn()
            if (controller := current_jev_controller()) is not None:
                controller.user_stopped()
            engine.user_stopped()

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(event: Any) -> None:
        is_final = bool(getattr(event, "is_final", False))
        transcript = getattr(event, "transcript", "") or ""
        engine.update_transcript(is_final=is_final)
        if (controller := current_jev_controller()) is not None:
            controller.on_transcript(transcript, is_final=is_final)
        # The public turn detector above supplies a real probability while the
        # user is speaking. A final transcript is still the safe boundary when
        # the detector is disabled or unavailable, so keep the binary fallback:
        # never start an acknowledgement after STT says the turn is complete.
        if is_final and not user_is_speaking:
            engine.update_eot_probability(1.0)

    if room is not None:

        @room.on("data_received")
        def _on_data_received(packet: Any) -> None:
            if getattr(packet, "topic", None) != "blue-machines-control":
                return
            try:
                payload = json.loads(bytes(getattr(packet, "data", b"")).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                return
            if payload.get("type") == "backchannel_toggle" and isinstance(
                payload.get("enabled"), bool
            ):
                selected_mode = "backchannel" if payload["enabled"] else "baseline"
            elif payload.get("type") == "experiment_mode" and payload.get("mode") in {
                "baseline",
                "backchannel",
                "jev_backchannel",
            }:
                selected_mode = payload["mode"]
            else:
                return

            if locked_mode is not None and selected_mode != locked_mode:
                record_event(
                    "experiment_mode_change_rejected",
                    requested_mode=selected_mode,
                    locked_mode=locked_mode,
                )
                return

            controller = current_jev_controller()
            if selected_mode == "jev_backchannel" and controller is None:
                record_event("jev_mode_unavailable", reason="missing_typesafe_configuration")
                return
            if controller is not None:
                controller.set_enabled(selected_mode == "jev_backchannel")
            if selected_mode != "jev_backchannel":
                selected_cue["text"] = settings.backchannel_text
            engine.set_semantic_required(selected_mode == "jev_backchannel")
            engine.set_enabled(selected_mode != "baseline")
            record_event("experiment_mode_changed", mode=selected_mode)

    @session.on("agent_state_changed")
    def _on_agent_state_changed(event: Any) -> None:
        new_state = getattr(event, "new_state", None)
        engine.set_agent_busy(new_state == "speaking")

    return engine


def attach_backchannel_audio_instrumentation(
    session: AgentSession, engine: BackchannelEngine
) -> None:
    """Use the output sink's playback marker for decision-to-audio latency."""

    audio = session.output.audio
    if audio is None:
        return

    def _on_playback_started(_event: Any) -> None:
        engine.mark_audio_started()

    audio.on("playback_started", _on_playback_started)


async def entrypoint(ctx: JobContext) -> None:
    """Run one room job using settings loaded inside the child process.

    LiveKit starts each room job in a separate process. Keeping this callback at
    module scope makes it importable and pickle-safe for that process boundary.
    """

    settings = Settings.from_env()
    run_context = parse_run_context(getattr(ctx.job, "metadata", None))
    mode = (
        run_context.mode
        if run_context is not None
        else ("backchannel" if settings.backchannel_enabled else "baseline")
    )
    backchannel_enabled = mode in {"backchannel", "jev_backchannel"}
    semantic_required = mode == "jev_backchannel"
    stt = create_stt(settings)
    if semantic_required and not stt.capabilities.interim_results:
        raise ConfigurationError(JEV_INTERIM_STT_ERROR.format(provider=settings.stt_provider))
    if semantic_required and settings.typesafe_api_key is None:
        raise ConfigurationError("TYPESAFE_API_KEY is required when mode=jev_backchannel")
    llm = create_llm(settings)
    tts = create_tts(settings)

    recorder = EventRecorder(
        settings.event_log_path,
        scenario_id=run_context.scenario_id if run_context else None,
        mode=mode,
        run_id=run_context.run_id if run_context else None,
    )
    recorder.record(
        "session_started",
        room_name=ctx.room.name,
        backchannel_enabled=backchannel_enabled,
        scenario_id=run_context.scenario_id if run_context else None,
        run_id=run_context.run_id if run_context else None,
    )
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=stt,
        llm=llm,
        tts=tts,
    )
    jev_controller: JevTurnController | None = None
    jev_controller_ref: list[JevTurnController] = []
    eot_detector_ref: list[EotDetector] = []
    cue_state = {"text": settings.backchannel_text}
    cached_clips = load_backchannel_clips()
    backchannel = attach_backchanneling(
        session,
        settings,
        recorder,
        enabled=backchannel_enabled,
        semantic_required=semantic_required,
        jev_controller_ref=jev_controller_ref,
        eot_detector_ref=eot_detector_ref,
        cue_state=cue_state,
        locked_mode=mode,
        room=ctx.room,
        cached_clips=cached_clips,
    )
    if settings.typesafe_api_key is not None and stt.capabilities.interim_results:
        classifier = JevClassifier(
            AsyncTypeSafeClient(
                api_key=settings.typesafe_api_key.get_secret_value(),
                model=settings.jev_model,
                timeout=settings.jev_timeout_seconds,
            ),
            model=settings.jev_model,
            approval_threshold=settings.jev_approval_threshold,
            helpful_threshold=settings.jev_helpful_threshold,
            helpful_threshold_semantic=settings.jev_helpful_threshold_semantic,
            expects_answer_ceiling=settings.jev_expects_answer_ceiling,
            expects_answer_ceiling_semantic=settings.jev_expects_answer_ceiling_semantic,
        )
        jev_controller = JevTurnController(
            backchannel,
            classifier,
            min_interval_seconds=settings.jev_min_interval_seconds,
            timeout_seconds=settings.jev_timeout_seconds,
            enabled=semantic_required,
            max_approved_per_turn=5 if semantic_required else 1,
            min_words_between_cues=7,
            on_event=lambda name, **data: (
                recorder.record(name, **data) if not recorder.closed else None
            ),
            on_cue_selected=lambda text: cue_state.__setitem__("text", text),
        )
        jev_controller_ref.append(jev_controller)
    attach_instrumentation(session, recorder, backchannel)

    async def finalize() -> None:
        if jev_controller is not None:
            await jev_controller.aclose()
        if backchannel is not None:
            await backchannel.aclose()
        for detector in eot_detector_ref:
            await detector.aclose()
        if not recorder.closed:
            recorder.record("session_stopped")
            recorder.close()

    ctx.add_shutdown_callback(finalize)
    try:
        await session.start(
            agent=BaselineAssistant(
                instructions=settings.agent_instructions,
                greet=run_context.greet if run_context is not None else True,
            ),
            room=ctx.room,
            room_options=room_io.RoomOptions(),
        )
        attach_backchannel_audio_instrumentation(session, backchannel)
    except Exception as exc:
        if not recorder.closed:
            recorder.record("session_error", error_type=type(exc).__name__)
        await finalize()
        logger.exception("Voice session failed; check provider and LiveKit configuration")
        raise


def build_server(settings: Settings) -> AgentServer:
    """Build a configured worker without reading secrets at import time."""

    server = AgentServer(
        ws_url=settings.livekit_url,
        api_key=settings.livekit_api_key.get_secret_value(),
        api_secret=settings.livekit_api_secret.get_secret_value(),
    )
    server.rtc_session(entrypoint, agent_name=settings.livekit_agent_name)
    return server


def main() -> None:
    """Validate configuration and hand control to the official LiveKit CLI."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    cli.run_app(build_server(settings))


if __name__ == "__main__":
    main()
