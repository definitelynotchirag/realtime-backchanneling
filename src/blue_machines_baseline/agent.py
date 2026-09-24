"""Runnable Phase 1 LiveKit voice-agent baseline."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import wave
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference, room_io
from livekit.plugins import elevenlabs, google, groq, openai, silero
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from . import (
    deepgram_stt,
    deepgram_tts,
    gemini_tts,
    groq_interim_stt,
    livekit_endpoints,
    openrouter_tts,
)
from .backchannel import BackchannelEngine
from .benchmark import parse_run_context
from .config import JEV_INTERIM_STT_ERROR, ConfigurationError, Settings
from .cue_audio import (
    CueRotation,
    cue_phrases,
    cue_slug,
    read_cached_cues,
    synthesize_cue_frames,
    write_cached_cues,
)
from .cue_bank import CUE_BANK_BY_STYLE
from .eot_detector import EotDetector
from .events import EventRecorder
from .jev import JevClassifier, JevTurnController

logger = logging.getLogger("blue-machines-baseline")


BACKCHANNEL_CLIP_DIR = Path(__file__).resolve().parents[2] / "assets" / "backchannels"

CUE_TEXTS: tuple[str, ...] = tuple(cue for phrases in CUE_BANK_BY_STYLE.values() for cue in phrases)
"""Every acknowledgement the policy may choose from."""


def load_backchannel_clips(
    cue_texts: Sequence[str] = CUE_TEXTS,
) -> dict[str, rtc.AudioFrame]:
    """Load the committed acknowledgement clips for the cues that have one.

    This is the fallback path: with `BACKCHANNEL_CLIP_SOURCE=assets` it is the only
    path, and otherwise it covers sessions where the speech provider cannot render the
    cues at startup.
    """

    clips: dict[str, rtc.AudioFrame] = {}
    for cue_text in cue_texts:
        path = BACKCHANNEL_CLIP_DIR / f"{cue_slug(cue_text)}.wav"
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


async def synthesize_backchannel_clips(
    settings: Settings,
    cue_texts: Sequence[str] = CUE_TEXTS,
    *,
    timeout_seconds: float = 8.0,
) -> dict[str, rtc.AudioFrame]:
    """Render the cue bank in the configured voice, once, before the session starts.

    Cheap to hold (a few hundred kilobytes) and the only way a cue stays a cue: from
    cache it is audible in ~20 ms, whereas synthesizing per cue costs the provider's
    time-to-first-audio and lands the acknowledgement after the speaker has moved on.
    """

    provider = create_tts(settings)
    try:
        return await asyncio.wait_for(
            synthesize_cue_frames(provider, cue_phrases(cue_texts)), timeout=timeout_seconds
        )
    finally:
        close = getattr(provider, "aclose", None)
        if callable(close):
            await close()


def cue_cache_key(settings: Settings) -> str:
    """Which voice the cached cues belong to.

    Every room runs in its own process, so this is what keeps a sweep from paying the
    provider once per room.
    """

    voice = getattr(create_tts(settings), "model", "") or settings.scenario_tts_model or "default"
    return f"{settings.tts_provider}-{voice}".replace("/", "-").replace(":", "-")


async def load_cue_clips(settings: Settings, recorder: EventRecorder) -> dict[str, rtc.AudioFrame]:
    """The cue audio this session will play from.

    Precedence: cues already rendered for this voice, then a fresh render from the
    configured provider, then the committed clips. Whatever the first two are missing
    is filled from the committed bank, so the policy always has audio to choose from and
    never falls through to a one-second on-demand synthesis.
    """

    assets = load_backchannel_clips()
    if settings.backchannel_clip_source == "assets":
        recorder.record("backchannel_clips_ready", source="assets", cue_count=len(assets))
        return assets

    clips = dict(assets)
    started = time.monotonic()
    missing: list[str] = []
    try:
        cache_key = cue_cache_key(settings)
        cached = read_cached_cues(cache_key, CUE_TEXTS)
        clips.update(cached)
        missing = [cue for cue in CUE_TEXTS if cue not in cached]
        if missing:
            produced = await synthesize_backchannel_clips(settings, missing)
            write_cached_cues(cache_key, produced)
            clips.update(produced)
    except (Exception, asyncio.CancelledError) as exc:  # noqa: BLE001 - any failure degrades
        if isinstance(exc, asyncio.CancelledError):
            raise
        logger.warning(
            "could not render every acknowledgement cue with %s (%s: %s); the "
            "committed clips cover the rest",
            settings.tts_provider,
            type(exc).__name__,
            str(exc)[:160],
        )
        recorder.record(
            "backchannel_clips_ready",
            source="assets",
            cue_count=len(clips),
            fallback_reason=type(exc).__name__,
        )
        return clips

    recorder.record(
        "backchannel_clips_ready",
        source="tts",
        cue_count=len(clips),
        rendered=len(missing),
        provider=settings.tts_provider,
        duration_ms=round((time.monotonic() - started) * 1000, 1),
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
            # Short on purpose: this is synthesised once per room.
            instructions="Greet the user in four words or fewer."
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


def create_stt(
    settings: Settings,
) -> inference.STT | groq.STT | groq_interim_stt.STT | deepgram_stt.STT:
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
    if settings.stt_provider == "deepgram":
        if settings.deepgram_api_key is None:
            raise ConfigurationError("DEEPGRAM_API_KEY is required when STT_PROVIDER=deepgram")
        return deepgram_stt.STT(
            model=settings.deepgram_stt_model,
            language=settings.deepgram_stt_language,
            endpointing_ms=settings.deepgram_stt_endpointing_ms,
            api_key=settings.deepgram_api_key.get_secret_value(),
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


def create_scenario_tts(
    settings: Settings,
) -> (
    inference.TTS
    | elevenlabs.TTS
    | gemini_tts.TTS
    | groq.TTS
    | openrouter_tts.TTS
    | deepgram_tts.TTS
):
    """The configured speech provider, speaking as the user for scripted runs.

    The driver synthesizes the user's side of a scenario with it. Same provider and
    credentials as :func:`create_tts`; only the voice can differ, so the two
    speakers in one room are distinguishable and the user's side is audible in the
    stack actually under test rather than in whatever voice a committed clip was
    rendered with months ago.

    Unset overrides mean "use the agent's voice". For Deepgram the voice *is* the
    model (``aura-2-thalia-en``), so ``scenario_tts_model`` is the lever there;
    providers that separate the two use ``scenario_tts_voice``.
    """

    voice = settings.scenario_tts_voice
    model = settings.scenario_tts_model

    if settings.tts_provider == "livekit_inference":
        return inference.TTS(
            model=settings.livekit_tts_model,
            voice=voice or settings.livekit_tts_voice,
            language="en",
            api_key=settings.livekit_api_key.get_secret_value(),
            api_secret=settings.livekit_api_secret.get_secret_value(),
        )

    if settings.tts_provider == "groq_tts":
        if settings.groq_api_key is None:
            raise ConfigurationError("GROQ_API_KEY is required when TTS_PROVIDER=groq_tts")
        return groq.TTS(
            model=settings.groq_tts_model,
            voice=voice or settings.groq_tts_voice,
            api_key=settings.groq_api_key.get_secret_value(),
        )

    if settings.tts_provider == "deepgram_tts":
        if settings.deepgram_api_key is None:
            raise ConfigurationError("DEEPGRAM_API_KEY is required when TTS_PROVIDER=deepgram_tts")
        return deepgram_tts.TTS(
            model=model or settings.deepgram_tts_model,
            api_key=settings.deepgram_api_key.get_secret_value(),
        )

    if settings.tts_provider == "openrouter_tts":
        if settings.openrouter_api_key is None:
            raise ConfigurationError(
                "OPENROUTER_API_KEY is required when TTS_PROVIDER=openrouter_tts"
            )
        return openrouter_tts.TTS(
            model=settings.openrouter_tts_model,
            voice=voice or settings.openrouter_tts_voice,
            api_key=settings.openrouter_api_key.get_secret_value(),
        )

    if settings.tts_provider == "gemini_tts":
        if settings.gemini_api_key is None:
            raise ConfigurationError("GEMINI_API_KEY is required when TTS_PROVIDER=gemini_tts")
        return gemini_tts.TTS(
            model=settings.gemini_tts_model,
            voice=voice or settings.gemini_tts_voice,
            api_key=settings.gemini_api_key.get_secret_value(),
        )

    if settings.elevenlabs_api_key is None:
        raise ConfigurationError("ELEVENLABS_API_KEY is required when TTS_PROVIDER=elevenlabs")
    return elevenlabs.TTS(
        voice_id=settings.elevenlabs_voice_id,
        model=settings.elevenlabs_tts_model,
        api_key=settings.elevenlabs_api_key.get_secret_value(),
    )


def create_tts(
    settings: Settings,
) -> (
    inference.TTS
    | elevenlabs.TTS
    | gemini_tts.TTS
    | groq.TTS
    | openrouter_tts.TTS
    | deepgram_tts.TTS
):
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

    if settings.tts_provider == "deepgram_tts":
        if settings.deepgram_api_key is None:
            raise ConfigurationError("DEEPGRAM_API_KEY is required when TTS_PROVIDER=deepgram_tts")
        return deepgram_tts.TTS(
            model=settings.deepgram_tts_model,
            api_key=settings.deepgram_api_key.get_secret_value(),
        )

    if settings.tts_provider == "openrouter_tts":
        if settings.openrouter_api_key is None:
            raise ConfigurationError(
                "OPENROUTER_API_KEY is required when TTS_PROVIDER=openrouter_tts"
            )
        return openrouter_tts.TTS(
            model=settings.openrouter_tts_model,
            voice=settings.openrouter_tts_voice,
            api_key=settings.openrouter_api_key.get_secret_value(),
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

    speaking_episode_was_cue = False

    @session.on("agent_state_changed")
    def _on_agent_state_changed(event: Any) -> None:
        nonlocal speaking_episode_was_cue
        old_state = getattr(event, "old_state", None)
        new_state = getattr(event, "new_state", None)
        if new_state == "speaking" and old_state != "speaking":
            speaking_episode_was_cue = bool(backchannel and backchannel.active)
            record_if_open(
                "backchannel_agent_speaking"
                if speaking_episode_was_cue
                else "agent_response_started",
                previous_state=old_state,
            )
        elif old_state == "speaking" and new_state != "speaking":
            # A cue that opened as an acknowledgement closes as one: the engine
            # reports its own completion, and labelling it an agent response would
            # make a cue look like an answer to anything reading the event stream.
            if speaking_episode_was_cue:
                speaking_episode_was_cue = False
                return
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
        record_if_open(
            "session_error",
            error_type=type(error).__name__,
            # The provider's own refusal is the actionable part ("no credits", "rate
            # limited"), and without it the operator is left guessing which limit was hit.
            error_message=str(error)[:300],
        )


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

    # A timer policy has no classifier to pick between the configured cues, so it cycles
    # through them; repeating one sound for a whole conversation is what makes an
    # acknowledgement sound mechanical. Only cues with audio are offered, so a cue never
    # falls through to an on-demand synthesis that would cost a second instead of 20 ms.
    timer_cues = [
        text for text in settings.backchannel_texts if not cached_clips or text in cached_clips
    ]
    if cached_clips and len(timer_cues) < len(settings.backchannel_texts):
        logger.warning(
            "no cue audio for %s; those cues stay out of the timer policy's rotation",
            ", ".join(text for text in settings.backchannel_texts if text not in cached_clips),
        )
    rotation = CueRotation(timer_cues or settings.backchannel_texts)
    selected_cue = (
        cue_state if cue_state is not None else {"text": rotation.current, "from_jev": False}
    )

    def play_cue() -> Any:
        # Jev names the cue it earned by classifying the turn; the timer policy has
        # nothing to consult, so it takes the next cue in its rotation.
        if selected_cue.get("from_jev"):
            cue_text = selected_cue["text"]
            selected_cue["from_jev"] = False
        else:
            cue_text = rotation.next()
            selected_cue["text"] = cue_text
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
                selected_cue["text"] = rotation.current
                selected_cue["from_jev"] = False
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
        # The provider stack, so a report built from these events names the stack it
        # measured instead of relying on when the log happened to be written.
        stt_provider=settings.stt_provider,
        llm_provider=settings.llm_provider,
        tts_provider=settings.tts_provider,
        eot_detector=settings.eot_detector,
        # The spoken style matters as much as the providers: a vague instruction is
        # how a voice agent ends up reading a seven hundred word essay out loud.
        agent_instructions=settings.agent_instructions,
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
    cue_state: dict[str, Any] = {"text": settings.backchannel_texts[0], "from_jev": False}
    # Rendered before the session starts, from the configured voice. Held in memory so
    # a cue is audible ~20 ms after the policy decides on one.
    cached_clips = await load_cue_clips(settings, recorder)
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
                # The SDK retries 429/5xx responses with backoff, which pushes a call
                # past the deadline the policy is waiting on. For a decision about a
                # partial transcript a retry is also stale by definition: the next
                # interim produces a fresher call, and the controller coalesces those.
                retry=RetryPolicy(max_retries=0),
            ),
            model=settings.jev_model,
            available_cues=set(cached_clips),
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
            on_cue_selected=lambda text: cue_state.update({"text": text, "from_jev": True}),
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

    settings, endpoint = livekit_endpoints.apply_endpoint(settings)
    if endpoint is not None:
        logger.info("livekit project: %s (%s)", endpoint.label, endpoint.url)
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
