"""Groq speech-to-text with interim transcripts.

Groq's Whisper endpoint transcribes in batches, so the bundled plugin reports it
as non-streaming: the pipeline only learns what the user said once their turn has
ended. That is fine for answering, and fatal for the Jev policy, which decides
whether to acknowledge *while* the user is still speaking and therefore needs
partial transcripts as they arrive.

This adapter keeps the same batch endpoint but adds a streaming cadence on top:
audio accumulates per turn, and every ``interim_interval_seconds`` the buffer so
far is transcribed and emitted as an interim transcript. At the turn boundary the
buffer is transcribed once more and emitted as the final transcript, which is the
accurate one (it covers the whole turn rather than a prefix).

The cost is a provider request per interval instead of one per turn; the benefit
is a semantic policy that can run on a provider that does not stream natively.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections.abc import Callable
from dataclasses import dataclass

from livekit import rtc
from livekit.agents import stt, vad
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import AudioBuffer

logger = logging.getLogger("blue-machines-groq-interim")

API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
REQUEST_TIMEOUT_SECONDS: float = 30.0
SAMPLE_WIDTH_BYTES = 2
USER_AGENT = "blue-machines-baseline/0.1 (LiveKit backchannel benchmark)"
DEFAULT_INTERIM_INTERVAL_SECONDS: float = 3.0
"""How much audio to accumulate before asking for a partial transcript.

Groq allows 20 requests per minute on the free tier, so a cadence much faster
than this spends the whole budget on interims; the final transcript of every turn
must always fit inside it.
"""

DEFAULT_MAX_REQUESTS_PER_MINUTE: int = 18
"""Self-imposed ceiling, deliberately below Groq's 20 RPM for the free tier."""

MIN_SEGMENT_SECONDS: float = 0.3
"""Segments shorter than this are not worth a provider request."""


class GroqInterimSTTError(RuntimeError):
    """Raised when Groq cannot transcribe the buffered audio."""


def wav_bytes_from_frames(frames: list[rtc.AudioFrame]) -> tuple[bytes, float]:
    """Encode frames as a 16-bit PCM WAV plus the audio's duration in seconds."""

    if not frames:
        return b"", 0.0
    if len(frames) == 1:
        frame = frames[0]
    else:
        frame = rtc.combine_audio_frames(frames)
    data = frame.data
    payload = bytes(data) if isinstance(data, memoryview) else bytes(data)
    duration = frame.samples_per_channel / max(1, frame.sample_rate)
    with io.BytesIO() as raw:
        with wave.open(raw, "wb") as handle:
            handle.setnchannels(frame.num_channels)
            handle.setsampwidth(SAMPLE_WIDTH_BYTES)
            handle.setframerate(frame.sample_rate)
            handle.writeframes(payload)
        return raw.getvalue(), duration


def _post_transcription(
    *, api_key: str, model: str, audio: bytes, language: str | None, timeout: float
) -> dict[str, object]:
    """Send one WAV payload to Groq and return the decoded JSON response."""

    boundary = f"----blue-machines-{uuid.uuid4().hex}"
    parts: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        header = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
        parts.append(f"{header}{value}\r\n".encode())

    parts.append(
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="segment.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
    )
    parts.append(audio)
    parts.append(b"\r\n")
    add_field("model", model)
    add_field("response_format", "json")
    if language:
        add_field("language", language)
    parts.append(f"--{boundary}--\r\n".encode())

    request = urllib.request.Request(
        API_URL,
        data=b"".join(parts),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            # Groq sits behind Cloudflare, which rejects the default urllib
            # user agent with HTTP 403 / error 1010.
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:200]
        raise GroqInterimSTTError(f"Groq transcription HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise GroqInterimSTTError(f"Groq unreachable: {exc.reason}") from exc
    return payload if isinstance(payload, dict) else {}


_SHARED_VAD: list[vad.VAD] = []
"""One silero VAD per process; loading it per stream would be wasteful."""


def _default_vad() -> vad.VAD:
    """Return the process-wide VAD, loading silero on first use."""

    if not _SHARED_VAD:
        from livekit.plugins import silero

        _SHARED_VAD.append(silero.VAD.load())
    return _SHARED_VAD[0]


@dataclass(frozen=True)
class _Options:
    model: str
    language: str | None
    api_key: str
    http_timeout: float
    interim_interval_seconds: float
    max_requests_per_minute: int


class _RequestBudget:
    """Sliding-window limiter shared by every stream of one process.

    Interim snapshots are optional - skipping one costs a little context for the
    semantic policy - while the final transcript of a turn is not. Spending the
    provider's per-minute budget on interims would starve the finals, so requests
    are admitted only while the window has room.
    """

    def __init__(self, max_per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self._max = max(1, max_per_minute)
        self._clock = clock
        self._stamps: list[float] = []

    def allow(self) -> bool:
        now = self._clock()
        self._stamps = [stamp for stamp in self._stamps if now - stamp < 60.0]
        if len(self._stamps) >= self._max:
            return False
        self._stamps.append(now)
        return True


class STT(stt.STT):
    """Batch transcription with VAD segmentation and interim snapshots."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "whisper-large-v3-turbo",
        language: str | None = "en",
        interim_interval_seconds: float = DEFAULT_INTERIM_INTERVAL_SECONDS,
        http_timeout: float = REQUEST_TIMEOUT_SECONDS,
        max_requests_per_minute: int = DEFAULT_MAX_REQUESTS_PER_MINUTE,
        request_budget: _RequestBudget | None = None,
        vad: vad.VAD | None = None,
    ) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for the groq_interim provider")
        # The pipeline forwards frames continuously and never flushes this stream
        # at a turn boundary, so turns are segmented with a VAD - the same way the
        # SDK's own StreamAdapter segments a batch provider.
        self._vad = vad or _default_vad()
        self._opts = _Options(
            model=model,
            language=language,
            api_key=api_key,
            http_timeout=http_timeout,
            interim_interval_seconds=max(0.4, interim_interval_seconds),
            max_requests_per_minute=max_requests_per_minute,
        )
        self._budget = request_budget or _RequestBudget(max_requests_per_minute)

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "groq"

    @property
    def vad(self) -> vad.VAD:
        """The VAD used to segment turns for this stream."""

        return self._vad

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        return _InterimStream(
            stt=self, opts=self._opts, conn_options=conn_options, vad_stream=self._vad
        )

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        frames = buffer if isinstance(buffer, list) else [buffer]
        audio, duration = wav_bytes_from_frames(frames)
        text = await self._transcribe(audio, language=language)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(language="en", text=text, start_time=0.0, end_time=duration)
            ],
        )

    async def _transcribe(self, audio: bytes, *, language: str | None = None) -> str:
        if not audio:
            return ""
        payload = await asyncio.to_thread(
            _post_transcription,
            api_key=self._opts.api_key,
            model=self._opts.model,
            audio=audio,
            language=language or self._opts.language,
            timeout=self._opts.http_timeout,
        )
        return str(payload.get("text", "")).strip()


class _InterimStream(stt.RecognizeStream):
    """Turn audio into interim transcripts, then one final transcript."""

    def __init__(
        self,
        *,
        stt: STT,
        opts: _Options,
        conn_options: APIConnectOptions,
        vad_stream: vad.VAD,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._stt = stt
        self._opts = opts
        self._vad = vad_stream

    def _emit(self, event_type: stt.SpeechEventType, text: str, duration: float) -> None:
        """Emit a transcript event.

        Empty text is still an event: a final transcript with no text is the
        honest outcome of a failed or silent segment, and the pipeline needs it to
        close the user's turn. Dropping it would leave the turn open forever.
        """

        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=event_type,
                alternatives=[
                    stt.SpeechData(language="en", text=text, start_time=0.0, end_time=duration)
                ],
            )
        )

    def _emit_marker(self, event_type: stt.SpeechEventType, duration: float = 0.0) -> None:
        """Emit a lifecycle marker (start/end of speech) without a transcript."""

        self._event_ch.send_nowait(
            stt.SpeechEvent(
                type=event_type,
                alternatives=[stt.SpeechData(language="en", text="", end_time=duration)],
            )
        )

    def _duration(self, frames: list[rtc.AudioFrame]) -> float:
        return sum(frame.samples_per_channel / max(1, frame.sample_rate) for frame in frames)

    async def _run(self) -> None:
        """Segment with VAD, snapshot interims while speaking, finalize per turn.

        The pipeline forwards frames continuously and never flushes this stream at
        a turn boundary (``Agent.stt_node`` only pushes frames), so segmenting
        cannot rely on ``flush()``. It follows ``stt.StreamAdapter``'s approach
        instead: run a VAD over the forwarded audio and treat each end-of-speech
        as the end of a turn, which is what produces a final transcript per turn.
        Interims are snapshots taken during the segment - the part StreamAdapter
        cannot do, and the reason this adapter exists at all.
        """

        vad_stream = self._vad.stream()
        segment: list[rtc.AudioFrame] = []
        speaking = False
        last_interim_at = time.monotonic()
        interim_task: asyncio.Task[None] | None = None

        async def emit_interim(frames: list[rtc.AudioFrame]) -> None:
            audio, duration = wav_bytes_from_frames(frames)
            try:
                text = await self._stt._transcribe(audio)
            except GroqInterimSTTError as exc:
                # Dropping an interim is cheap; the next snapshot covers the same
                # speech plus more, and the final transcript is unaffected.
                logger.debug("interim transcript failed: %s", exc)
                return
            self._emit(stt.SpeechEventType.INTERIM_TRANSCRIPT, text, duration)

        async def forward_input() -> None:
            nonlocal last_interim_at, interim_task
            async for item in self._input_ch:
                if isinstance(item, self._FlushSentinel):
                    vad_stream.flush()
                    continue
                vad_stream.push_frame(item)
                if not speaking:
                    continue
                segment.append(item)
                if interim_task is not None and not interim_task.done():
                    continue
                if time.monotonic() - last_interim_at < self._opts.interim_interval_seconds:
                    continue
                if self._duration(segment) < MIN_SEGMENT_SECONDS:
                    continue
                if not self._stt._budget.allow():
                    continue
                last_interim_at = time.monotonic()
                # Transcription happens in the background so forwarding audio is
                # never blocked by a provider round trip.
                interim_task = asyncio.create_task(emit_interim(list(segment)))
            vad_stream.end_input()

        async def recognize() -> None:
            nonlocal segment, speaking, last_interim_at
            async for event in vad_stream:
                if event.type == vad.VADEventType.START_OF_SPEECH:
                    speaking = True
                    segment = []
                    last_interim_at = time.monotonic()
                    self._emit_marker(stt.SpeechEventType.START_OF_SPEECH)
                elif event.type == vad.VADEventType.END_OF_SPEECH:
                    speaking = False
                    frames = list(event.frames) or segment
                    segment = []
                    duration = self._duration(frames)
                    self._emit_marker(stt.SpeechEventType.END_OF_SPEECH, duration)
                    text = ""
                    if frames:
                        audio, _ = wav_bytes_from_frames(frames)
                        try:
                            text = await self._stt._transcribe(audio)
                        except GroqInterimSTTError as exc:
                            logger.warning("final transcript failed: %s", exc)
                    # Emitted even when empty: the turn must close either way.
                    self._emit(stt.SpeechEventType.FINAL_TRANSCRIPT, text, duration)

        tasks = [
            asyncio.create_task(forward_input(), name="groq_interim_forward"),
            asyncio.create_task(recognize(), name="groq_interim_recognize"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            if interim_task is not None and not interim_task.done():
                interim_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await interim_task
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await vad_stream.aclose()
