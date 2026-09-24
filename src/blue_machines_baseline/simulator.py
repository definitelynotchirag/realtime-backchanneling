"""Headless benchmark driver: real rooms, scripted speech, no human at the mic.

The assignment asks for repeatable scenarios so the same interaction can be run
through the baseline and the backchannel configuration, and explicitly says not
to depend on speaking into a microphone twice. This driver does that:

1. ``--generate-audio`` renders one speech clip per scenario with the configured
   text-to-speech provider and stores it under ``assets/scenarios`` together
   with a sidecar describing the text and its measured length.
2. A run mode that, for every (scenario, mode, repeat), opens a real LiveKit
   room, dispatches the worker with the usual ``scenario_id``/``mode``/``run_id``
   metadata, joins as a participant, and publishes the clip as microphone audio -
   inserting the scenario's scripted pauses - then waits for the agent's answer.

The worker must be running separately, for example::

    uv run blue-machines-agent dev

Timing measurements come from the worker's own event log, not from this driver.
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import contextlib
import io
import json
import logging
import sys
import time
import uuid
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv
from livekit import api as livekit_api
from livekit import rtc

from . import livekit_endpoints
from .agent import create_scenario_tts
from .benchmark import SCENARIO_BY_ID, SCENARIOS, BenchmarkScenario
from .config import ConfigurationError, Settings
from .eot_detector import frame_rms

logger = logging.getLogger("blue-machines-scenario")

CLIP_SAMPLE_RATE = 16000
CLIP_CHANNELS = 1
PUBLISH_RATE = 16000
FRAME_MS = 20
WORDS_PER_SECOND = 2.5
"""Rough spoken rate used to size an utterance for a scenario's target length."""

NO_AGENT_GRACE_SECONDS = 20.0
"""How long to wait for the worker to join the room before giving up."""

GREETING_QUIET_SECONDS = 1.0
"""Agent audio silence that marks the end of its greeting before the clip plays."""

NO_GREETING_GRACE_SECONDS = 2.5
"""How long to let the worker settle when the greeting is disabled."""

RESPONSE_QUIET_SECONDS = 1.0
"""Agent audio silence used as a fallback completion signal."""

AGENT_SPEECH_RMS = 150.0
"""Frame RMS above which the agent's published audio counts as speech."""

TERMINAL_EVENTS = {"agent_response_ended", "session_stopped", "session_error"}


class EventLogTail:
    """Follow the worker's event log for one run's terminal events.

    The agent streams silence continuously, so audible activity cannot tell the
    driver when the answer finished. The worker's own event log can, and it is
    the same contract the analyzer reads.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset = path.stat().st_size if path.exists() else 0
        self._partial = ""
        self.response_seen = False
        """Whether a response started after the tail was opened.

        The tail is opened once the clip has finished playing, so a response
        recorded here is the agent's answer to the user's turn. Agent *audio* is
        not a substitute: in backchannel mode the cached cue is agent audio too,
        and it plays even when the speech provider rejects the answer.
        """

    def poll(self, run_id: str) -> str | None:
        """Return the first terminal event name seen for this run, if any."""

        if not self._path.exists():
            return None
        with self._path.open(encoding="utf-8") as stream:
            stream.seek(self._offset)
            chunk = stream.read()
            self._offset = stream.tell()
        if not chunk:
            return None
        text = self._partial + chunk
        lines = text.splitlines()
        self._partial = "" if text.endswith("\n") or not lines else lines.pop()
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("run_id") != run_id:
                continue
            name = record.get("name")
            if name == "agent_response_started":
                self.response_seen = True
                continue
            if name == "agent_response_ended" and not self.response_seen:
                # Anything else that made noise (a cached cue, say) ending is not
                # the answer; treating it as one ends the run before the reply.
                continue
            if isinstance(name, str) and name in TERMINAL_EVENTS:
                return name
        return None


UTTERANCES: dict[str, str] = {
    "short_answer": "Doing great.",
    "long_monologue": (
        "So last week we finally shipped the migration and it was a much bigger job than "
        "anyone expected because the schema had drifted over two years and every service "
        "assumed something slightly different about the same three columns."
    ),
    "approaching_end_of_turn": "Anyway that is basically the whole story for now.",
    "middle_pause": (
        "The interesting part is that the first attempt looked fine, and then we noticed "
        "the numbers were slightly off whenever two workers overlapped."
    ),
    "fast_speaker": (
        "We ran the numbers again and switched the queue and the retries dropped "
        "immediately so honestly it was worth the rewrite."
    ),
    "noisy_audio": (
        "Sorry about the background noise here, I am calling from the office floor and "
        "the team is demoing something right behind me."
    ),
    "multiple_backchannels": (
        "Let me walk you through the whole thing from the start. We had three services "
        "talking to one queue, and the retry policy was copied from an older project "
        "that assumed at-least-once delivery, which was not true any more. So whenever "
        "a worker died mid-batch the whole batch came back, and the duplicates piled up "
        "until the reconciliation job started timing out as well."
    ),
    "stop_before_ack": "Yes, exactly.",
}


@dataclass(frozen=True)
class ScenarioClip:
    """A rendered utterance plus the sidecar's measured metadata."""

    scenario_id: str
    path: Path
    text: str
    duration_seconds: float
    pause_durations_seconds: tuple[float, ...]


def utterance_for(scenario: BenchmarkScenario) -> str:
    """Return the scripted utterance for a scenario, with its target length noted."""

    return UTTERANCES.get(scenario.scenario_id, scenario.description)


def clip_paths(directory: Path, scenario_id: str) -> tuple[Path, Path]:
    return directory / f"{scenario_id}.wav", directory / f"{scenario_id}.json"


def pcm_to_wav(pcm: bytes, *, source_rate: int, target_rate: int = CLIP_SAMPLE_RATE) -> bytes:
    """Convert raw mono 16-bit PCM to a WAV container at the target rate."""

    if source_rate != target_rate:
        pcm, _ = audioop.ratecv(pcm, 2, 1, source_rate, target_rate, None)
    with io.BytesIO() as raw:
        with wave.open(raw, "wb") as handle:
            handle.setnchannels(CLIP_CHANNELS)
            handle.setsampwidth(2)
            handle.setframerate(target_rate)
            handle.writeframes(pcm)
        return raw.getvalue()


def read_clip(path: Path) -> list[rtc.AudioFrame]:
    """Read a WAV clip as 16 kHz mono frames."""

    if not path.exists():
        raise FileNotFoundError(f"missing scenario clip {path}; run with --generate-audio first")
    with wave.open(str(path)) as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    if channels > 1:
        raw = audioop.tomono(raw, width, 0.5, 0.5)
    if rate != PUBLISH_RATE:
        raw, _ = audioop.ratecv(raw, width, 1, rate, PUBLISH_RATE, None)
    samples = memoryview(raw).cast("h")
    step = PUBLISH_RATE * FRAME_MS // 1000
    return [
        rtc.AudioFrame(
            data=bytes(samples[index : index + step]),
            sample_rate=PUBLISH_RATE,
            num_channels=1,
            samples_per_channel=step,
        )
        for index in range(0, len(samples) - step + 1, step)
    ]


def silence_frames(seconds: float) -> list[rtc.AudioFrame]:
    count = max(0, int(seconds * 1000 / FRAME_MS))
    step = PUBLISH_RATE * FRAME_MS // 1000
    return [
        rtc.AudioFrame(
            data=b"\x00\x00" * step,
            sample_rate=PUBLISH_RATE,
            num_channels=1,
            samples_per_channel=step,
        )
        for _ in range(count)
    ]


def with_pauses(frames: list[rtc.AudioFrame], pauses: tuple[float, ...]) -> list[rtc.AudioFrame]:
    """Insert the scenario's silence windows evenly inside the clip.

    A pause in the middle of a sentence is what makes the policy's pause and
    end-of-turn behaviour observable, so the silence is placed between equal
    chunks of speech rather than only at the end.
    """

    if not pauses:
        return frames
    chunks = len(pauses) + 1
    size = max(1, len(frames) // chunks)
    result: list[rtc.AudioFrame] = []
    for index in range(chunks):
        start = index * size
        stop = len(frames) if index == chunks - 1 else start + size
        result.extend(frames[start:stop])
        if index < len(pauses):
            result.extend(silence_frames(pauses[index]))
    return result


def split_for_pauses(text: str, pause_count: int) -> list[str]:
    """Split an utterance into the segments a scenario's pauses separate.

    A pause in the middle of a sentence is what makes the policy's pause and
    end-of-turn behaviour observable. Splitting the *text* rather than the rendered
    frames puts each pause at a word boundary, and - unlike counting frames - it
    works while the audio is still being produced, which is what lets the driver
    synthesize the user's side live. If the utterance has fewer words than segments,
    the extra pauses are dropped rather than spoken as empty audio.
    """

    words = text.split()
    if pause_count <= 0 or len(words) <= 1:
        return [text]
    segments = min(pause_count + 1, len(words))
    size = len(words) // segments
    remainder = len(words) % segments
    result: list[str] = []
    index = 0
    for position in range(segments):
        take = size + (1 if position < remainder else 0)
        result.append(" ".join(words[index : index + take]))
        index += take
    return result


class PublishRateFrames:
    """Convert provider audio into the fixed frames the room publishes.

    Providers choose their own sample rate - Deepgram speaks at 24 kHz while the
    AudioSource is created at 16 kHz - so live audio has to be converted as it
    arrives. ``audioop.ratecv`` keeps its resampler state between calls and the
    leftover samples are carried into the next frame, so nothing is dropped and the
    user does not wait for the whole utterance to be converted before being heard.
    """

    def __init__(self, *, sample_width: int = 2) -> None:
        self._sample_width = sample_width
        self._state: object | None = None
        self._carry = b""

    @property
    def frame_bytes(self) -> int:
        return (PUBLISH_RATE * FRAME_MS // 1000) * self._sample_width

    def push(self, frame: rtc.AudioFrame) -> list[rtc.AudioFrame]:
        data = bytes(frame.data)
        if frame.num_channels > 1:
            data = audioop.tomono(data, self._sample_width, 0.5, 0.5)
        if frame.sample_rate != PUBLISH_RATE:
            data, self._state = audioop.ratecv(
                data,
                self._sample_width,
                1,
                frame.sample_rate,
                PUBLISH_RATE,
                self._state,
            )
        pcm = self._carry + data
        step = self.frame_bytes
        frames = [
            self._frame(pcm[offset : offset + step])
            for offset in range(0, len(pcm) - step + 1, step)
        ]
        self._carry = pcm[len(frames) * step :]
        return frames

    def flush(self) -> list[rtc.AudioFrame]:
        """Emit the tail, padded with silence, so the last word is not clipped."""

        if not self._carry:
            return []
        padded = self._carry + b"\x00" * (self.frame_bytes - len(self._carry))
        self._carry = b""
        return [self._frame(padded)]

    def _frame(self, chunk: bytes) -> rtc.AudioFrame:
        return rtc.AudioFrame(
            data=chunk,
            sample_rate=PUBLISH_RATE,
            num_channels=1,
            samples_per_channel=len(chunk) // self._sample_width,
        )


class SpeechProvider(Protocol):
    """What the driver needs from a text-to-speech provider.

    Structural rather than nominal so a test can hand it a fake, and so any
    LiveKit TTS adapter satisfies it without the driver importing them all.
    """

    def synthesize(self, text: str) -> AsyncIterator[Any]: ...


class FrameSink(Protocol):
    """The part of ``rtc.AudioSource`` the live path uses."""

    async def capture_frame(self, frame: rtc.AudioFrame) -> None: ...


async def play_live_utterance(
    source: FrameSink,
    tts_provider: SpeechProvider,
    scenario: BenchmarkScenario,
    *,
    frame_ms: int = FRAME_MS,
) -> float:
    """Speak the scenario's utterance as the user, synthesizing it as it goes.

    The audio is captured into the room frame by frame as the provider produces it,
    so the user starts speaking a provider time-to-first-audio after the run starts
    rather than after the whole utterance is synthesized. Returns the seconds of
    audio captured, pauses included.
    """

    segments = split_for_pauses(utterance_for(scenario), len(scenario.pause_durations_seconds))
    pauses = scenario.pause_durations_seconds
    converter = PublishRateFrames()
    captured = 0.0

    async def capture(frames: list[rtc.AudioFrame]) -> None:
        nonlocal captured
        for frame in frames:
            await source.capture_frame(frame)
            captured += frame.samples_per_channel / max(1, frame.sample_rate)
            await asyncio.sleep(frame_ms / 1000)

    for position, segment in enumerate(segments):
        produced = 0
        async for event in tts_provider.synthesize(segment):
            frame = getattr(event, "frame", None)
            if frame is None:
                continue
            produced += 1
            await capture(converter.push(frame))
        await capture(converter.flush())
        if produced == 0:
            raise RuntimeError(
                f"{scenario.scenario_id}: the speech provider returned no audio for {segment!r}"
            )
        if position < len(segments) - 1 and position < len(pauses):
            await capture(silence_frames(pauses[position]))
    return round(captured, 3)


async def generate_audio(settings: Settings, scenario_ids: list[str]) -> list[ScenarioClip]:
    """Render one clip per scenario with the configured speech provider."""

    directory = settings.scenario_audio_dir
    directory.mkdir(parents=True, exist_ok=True)
    tts = create_scenario_tts(settings)

    clips: list[ScenarioClip] = []
    for scenario_id in scenario_ids:
        scenario = SCENARIO_BY_ID[scenario_id]
        text = utterance_for(scenario)
        frames = [event.frame async for event in tts.synthesize(text)]
        if not frames:
            raise RuntimeError(f"no audio returned for scenario {scenario_id}")
        combined = rtc.combine_audio_frames(frames)
        pcm = bytes(combined.data)
        wav_path, sidecar_path = clip_paths(directory, scenario_id)
        wav_path.write_bytes(pcm_to_wav(pcm, source_rate=combined.sample_rate))
        duration = round(combined.samples_per_channel / combined.sample_rate, 3)
        sidecar = {
            "scenario_id": scenario_id,
            "text": text,
            "provider": settings.tts_provider,
            "voice": (
                settings.scenario_tts_voice or settings.scenario_tts_model or "provider default"
            ),
            "model": getattr(tts, "model", settings.scenario_tts_model or ""),
            "duration_seconds": duration,
            "target_duration_seconds": scenario.speech_duration_seconds,
            "pause_durations_seconds": list(scenario.pause_durations_seconds),
        }
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
        clips.append(
            ScenarioClip(
                scenario_id=scenario_id,
                path=wav_path,
                text=text,
                duration_seconds=duration,
                pause_durations_seconds=scenario.pause_durations_seconds,
            )
        )
        logger.info(
            "generated %s (%.2fs, target %.2fs)",
            wav_path,
            duration,
            scenario.speech_duration_seconds,
        )
    return clips


def load_clip(scenario_id: str, directory: Path) -> ScenarioClip:
    wav_path, sidecar_path = clip_paths(directory, scenario_id)
    if not wav_path.exists():
        raise FileNotFoundError(
            f"missing scenario clip {wav_path}; run with --generate-audio first"
        )
    meta = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    return ScenarioClip(
        scenario_id=scenario_id,
        path=wav_path,
        text=str(meta.get("text", "")),
        duration_seconds=float(meta.get("duration_seconds", 0.0)),
        pause_durations_seconds=tuple(meta.get("pause_durations_seconds", ())),
    )


def build_token(
    settings: Settings, *, scenario_id: str, mode: str, run_id: str, greet: bool = False
) -> tuple[str, str]:
    """Mint a participant token that explicitly dispatches the worker."""

    room_name = f"blue-machines-{scenario_id}-{run_id}"
    metadata = json.dumps(
        {"scenario_id": scenario_id, "mode": mode, "run_id": run_id, "greet": greet},
        separators=(",", ":"),
    )
    token = (
        livekit_api.AccessToken(
            settings.livekit_api_key.get_secret_value(),
            settings.livekit_api_secret.get_secret_value(),
        )
        .with_identity(f"scenario-{uuid.uuid4().hex[:8]}")
        .with_name("Blue Machines scenario driver")
        .with_grants(
            livekit_api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .with_room_config(
            livekit_api.RoomConfiguration(
                agents=[
                    livekit_api.RoomAgentDispatch(
                        agent_name=settings.livekit_agent_name,
                        metadata=metadata,
                    )
                ]
            )
        )
    )
    return room_name, token.to_jwt()


@dataclass
class RunOutcome:
    scenario_id: str
    mode: str
    run_id: str
    room_name: str
    agent_joined: bool
    agent_spoke: bool
    agent_answered: bool
    elapsed_seconds: float


async def drive_one(
    settings: Settings,
    *,
    scenario_id: str,
    mode: str,
    repeat: int,
    max_wait: float,
    greet: bool = False,
    live_tts: SpeechProvider | None = None,
) -> RunOutcome:
    """Run one scripted scenario through a real room.

    With ``live_tts`` the user's line is synthesized as the run starts and streamed
    into the room; otherwise the committed clip for the scenario is replayed. Either
    way the agent hears the same words through the same microphone track, and the
    timing that matters is measured from when the user's audio stops.
    """

    run_id = f"{scenario_id}-{mode}-{repeat:02d}-{uuid.uuid4().hex[:6]}"
    room_name, token = build_token(
        settings, scenario_id=scenario_id, mode=mode, run_id=run_id, greet=greet
    )
    scenario = SCENARIO_BY_ID[scenario_id]
    frames: list[rtc.AudioFrame] = []
    if live_tts is None:
        clip = load_clip(scenario_id, settings.scenario_audio_dir)
        frames = with_pauses(read_clip(clip.path), clip.pause_durations_seconds)

    room = rtc.Room()
    state = {"agent_joined": False, "last_agent_audio": 0.0, "agent_spoke": False}
    tasks: list[asyncio.Task[None]] = []
    started = time.monotonic()

    async def drain(track: rtc.Track) -> None:
        stream = rtc.AudioStream(track)
        try:
            async for event in stream:
                # The agent publishes continuously, silence included, so frame
                # arrival says nothing about whether it is speaking. Only audio
                # above the level of a quiet room counts as speech.
                if frame_rms(event.frame) <= AGENT_SPEECH_RMS:
                    continue
                state["last_agent_audio"] = time.monotonic()
                state["agent_spoke"] = True
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a dropped agent track is not fatal
            logger.debug("agent audio tap stopped", exc_info=True)

    @room.on("participant_connected")
    def _on_participant(participant: rtc.RemoteParticipant) -> None:
        if participant.identity != room.local_participant.identity:
            state["agent_joined"] = True

    @room.on("track_subscribed")
    def _on_track(
        track: rtc.Track,
        _pub: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if participant.identity == room.local_participant.identity:
            return
        state["agent_joined"] = True
        tasks.append(asyncio.create_task(drain(track)))

    try:
        await room.connect(settings.livekit_url, token)
        source = rtc.AudioSource(PUBLISH_RATE, 1)
        track = rtc.LocalAudioTrack.create_audio_track("scenario-mic", source)
        await room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )

        # The worker is dispatched into the room and greets on entry, which takes
        # a few seconds. Publishing the clip before that happens loses it
        # entirely: an earlier version played to an empty room, so the agent
        # never transcribed anything and no turn was ever recorded.
        join_deadline = started + NO_AGENT_GRACE_SECONDS
        while time.monotonic() < join_deadline and not state["agent_joined"]:
            await asyncio.sleep(0.1)
        if not state["agent_joined"]:
            logger.warning("no worker joined %s", room_name)

        quiet_deadline = time.monotonic() + max_wait
        joined_at = time.monotonic()
        while time.monotonic() < quiet_deadline:
            await asyncio.sleep(0.1)
            last = state["last_agent_audio"]
            if state["agent_spoke"] and last and time.monotonic() - last > GREETING_QUIET_SECONDS:
                break
            # With the greeting turned off there is nothing to wait for: a short
            # grace is enough to let the worker finish attaching, and then the
            # clip can play instead of idling out the whole deadline.
            if (
                not greet
                and state["agent_joined"]
                and time.monotonic() - joined_at > NO_GREETING_GRACE_SECONDS
            ):
                break

        # Opened before the clip plays: a fast answer can start before the first
        # poll, and a tail opened afterwards would miss that event entirely and
        # report a successful run as unanswered.
        tail = EventLogTail(settings.event_log_path)
        if live_tts is not None:
            logger.info("synthesizing %s live into %s", scenario_id, room_name)
            spoken = await play_live_utterance(source, live_tts, scenario)
            logger.info("%s: synthesized %.2fs of user audio", scenario_id, spoken)
        else:
            logger.info(
                "playing the %s clip into %s (%.2fs)",
                scenario_id,
                room_name,
                len(frames) * FRAME_MS / 1000,
            )
            for frame in frames:
                await source.capture_frame(frame)
                await asyncio.sleep(FRAME_MS / 1000)
        playback_finished_at = time.monotonic()

        deadline = playback_finished_at + max_wait
        while time.monotonic() < deadline:
            await asyncio.sleep(0.2)
            if tail.poll(run_id):
                break
            # Silence only counts as "finished" once the agent has started its
            # answer. Any agent audio is not enough: a cached acknowledgement is
            # agent audio too, so a cue that plays just before the user stops used
            # to look like a reply that had already gone quiet, and the run ended
            # while the model was still generating the actual answer.
            if (
                tail.response_seen
                and state["last_agent_audio"] > playback_finished_at
                and time.monotonic() - state["last_agent_audio"] > RESPONSE_QUIET_SECONDS
            ):
                break
        return RunOutcome(
            scenario_id=scenario_id,
            mode=mode,
            run_id=run_id,
            room_name=room_name,
            agent_joined=state["agent_joined"],
            agent_spoke=state["agent_spoke"],
            agent_answered=tail.response_seen,
            elapsed_seconds=round(time.monotonic() - started, 2),
        )
    finally:
        # Teardown order matters: releasing the audio source and letting the
        # drains observe the closed stream comes before disconnecting the room.
        # Tearing a live capture down together with the transport is what aborts
        # the Rust side of the SDK ("panic in a function that cannot unwind").
        with contextlib.suppress(Exception):
            await source.aclose()
        for task in tasks:
            task.cancel()
        if tasks:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2.0)
        await room.disconnect()
        await _delete_room(settings, room_name)


async def _delete_room(settings: Settings, room_name: str) -> None:
    client = livekit_api.LiveKitAPI(
        settings.livekit_url.replace("wss://", "https://").replace("ws://", "http://"),
        settings.livekit_api_key.get_secret_value(),
        settings.livekit_api_secret.get_secret_value(),
    )
    try:
        await client.room.delete_room(livekit_api.DeleteRoomRequest(room=room_name))
    except Exception:  # noqa: BLE001 - cleanup is best effort
        logger.debug("could not delete room %s", room_name, exc_info=True)
    finally:
        await client.aclose()


async def run_benchmark(
    settings: Settings,
    *,
    scenario_ids: list[str],
    modes: list[str],
    repeats: int,
    max_wait: float,
    greet: bool = False,
    max_silent_runs: int = 3,
    live_audio: bool = False,
) -> list[RunOutcome]:
    outcomes: list[RunOutcome] = []
    silent_streak = 0
    provider: SpeechProvider | None = None

    def speech_provider() -> SpeechProvider:
        """Built on first use: a sweep that replays clips never needs one."""

        nonlocal provider
        if provider is None:
            provider = create_scenario_tts(settings)
        return provider

    for scenario_id in scenario_ids:
        has_clip = clip_paths(settings.scenario_audio_dir, scenario_id)[0].exists()
        synthesize_live = live_audio or not has_clip
        if synthesize_live and not has_clip and not live_audio:
            logger.info("no clip for %s; synthesizing the user's line live", scenario_id)
        live_tts = speech_provider() if synthesize_live else None
        for mode in modes:
            for repeat in range(1, repeats + 1):
                outcome = await drive_one(
                    settings,
                    scenario_id=scenario_id,
                    mode=mode,
                    repeat=repeat,
                    max_wait=max_wait,
                    greet=greet,
                    live_tts=live_tts,
                )
                outcomes.append(outcome)
                print(
                    f"{outcome.scenario_id:24s} {outcome.mode:16s} {outcome.run_id:32s} "
                    f"agent={'yes' if outcome.agent_joined else 'NO ':3s} "
                    f"answered={'yes' if outcome.agent_answered else 'NO ':3s} "
                    f"audio={'yes' if outcome.agent_spoke else 'NO ':3s} "
                    f"{outcome.elapsed_seconds:6.1f}s",
                    flush=True,
                )
                silent_streak = 0 if outcome.agent_answered else silent_streak + 1
                if silent_streak >= max_silent_runs:
                    # A run with no answer is not a measurement: it is a provider
                    # failure (quota, rate limit) or a broken worker. Stop instead
                    # of filling the log with empty runs.
                    print(
                        f"stopping: {silent_streak} consecutive runs produced no agent answer "
                        "- check provider quota and the worker log",
                        flush=True,
                    )
                    return outcomes
    return outcomes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive the benchmark scenarios through real rooms."
    )
    parser.add_argument(
        "--generate-audio",
        action="store_true",
        help="render the scenario utterances with the configured TTS provider and exit",
    )
    parser.add_argument(
        "--scenarios",
        default="all",
        help="comma-separated scenario ids, or 'all'",
    )
    parser.add_argument(
        "--modes",
        default="baseline,backchannel",
        help="comma-separated modes (baseline, backchannel, jev_backchannel)",
    )
    parser.add_argument("--repeats", type=int, default=1, help="runs per (scenario, mode)")
    parser.add_argument(
        "--max-wait", type=float, default=90.0, help="seconds to wait for the agent's answer"
    )
    parser.add_argument(
        "--live-audio",
        action="store_true",
        help=(
            "synthesize the user's line per run with the configured speech provider "
            "instead of replaying the committed clip"
        ),
    )
    parser.add_argument(
        "--greet",
        action="store_true",
        help="let the worker greet on entry (costs one speech request per run)",
    )
    parser.add_argument(
        "--max-silent-runs",
        type=int,
        default=3,
        help="stop the sweep after this many consecutive runs produced no agent audio",
    )
    return parser.parse_args(argv)


def resolve_scenarios(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return [scenario.scenario_id for scenario in SCENARIOS]
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in ids if item not in SCENARIO_BY_ID]
    if unknown:
        raise SystemExit(f"unknown scenario id(s): {', '.join(unknown)}")
    return ids


def main(argv: list[str] | None = None) -> None:
    """Generate scenario audio or drive the benchmark through real rooms."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv()
    args = parse_args(argv)
    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    settings, endpoint = livekit_endpoints.apply_endpoint(settings)
    if endpoint is not None:
        print(f"livekit project: {endpoint.url} ({endpoint.label})")
    scenario_ids = resolve_scenarios(args.scenarios)

    if args.generate_audio:
        clips = asyncio.run(generate_audio(settings, scenario_ids))
        for clip in clips:
            print(f"{clip.scenario_id:24s} {clip.duration_seconds:6.2f}s  {clip.path}")
        return

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    print(
        f"driving {len(scenario_ids)} scenario(s) x {len(modes)} mode(s) x {args.repeats} repeat(s)"
    )
    print("the worker must be running: uv run blue-machines-agent dev")
    scenario_voice = settings.scenario_tts_model or settings.scenario_tts_voice or "agent voice"
    if args.live_audio:
        print(
            f"user audio: live, synthesized per run with {settings.tts_provider} "
            f"({scenario_voice}); clips in {settings.scenario_audio_dir} are ignored"
        )
    else:
        print(
            f"user audio: clips from {settings.scenario_audio_dir} "
            "(pass --live-audio to synthesize them with the configured provider)"
        )
    outcomes = asyncio.run(
        run_benchmark(
            settings,
            scenario_ids=scenario_ids,
            modes=modes,
            repeats=args.repeats,
            max_wait=args.max_wait,
            greet=args.greet,
            max_silent_runs=args.max_silent_runs,
            live_audio=args.live_audio,
        )
    )
    # Cue audio is agent audio, so "produced audio" and "answered" are different
    # questions; a sweep is only healthy if the second one is true.
    spoke = sum(1 for outcome in outcomes if outcome.agent_spoke)
    answered = sum(1 for outcome in outcomes if outcome.agent_answered)
    print(f"{spoke}/{len(outcomes)} runs produced agent audio, {answered}/{len(outcomes)} answered")


if __name__ == "__main__":
    main()
