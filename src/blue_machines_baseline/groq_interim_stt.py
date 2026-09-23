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
import io
import json
import logging
import time
import urllib.error
import urllib.request
import uuid
import wave
from dataclasses import dataclass

from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import AudioBuffer

logger = logging.getLogger("blue-machines-groq-interim")

API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
REQUEST_TIMEOUT_SECONDS: float = 30.0
SAMPLE_WIDTH_BYTES = 2
USER_AGENT = "blue-machines-baseline/0.1 (LiveKit backchannel benchmark)"
DEFAULT_INTERIM_INTERVAL_SECONDS: float = 1.2
"""How much audio to accumulate before asking for a partial transcript."""

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


@dataclass(frozen=True)
class _Options:
    model: str
    language: str | None
    api_key: str
    http_timeout: float
    interim_interval_seconds: float


class STT(stt.STT):
    """Batch transcription that also emits interim transcripts on a cadence."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "whisper-large-v3-turbo",
        language: str | None = "en",
        interim_interval_seconds: float = DEFAULT_INTERIM_INTERVAL_SECONDS,
        http_timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for the groq_interim provider")
        self._opts = _Options(
            model=model,
            language=language,
            api_key=api_key,
            http_timeout=http_timeout,
            interim_interval_seconds=max(0.4, interim_interval_seconds),
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "groq"

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        return _InterimStream(stt=self, opts=self._opts, conn_options=conn_options)

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

    def __init__(self, *, stt: STT, opts: _Options, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._stt = stt
        self._opts = opts

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
        segment: list[rtc.AudioFrame] = []
        speaking = False
        last_interim_at = time.monotonic()

        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                if segment:
                    audio, duration = wav_bytes_from_frames(segment)
                    try:
                        text = await self._stt._transcribe(audio)
                    except GroqInterimSTTError as exc:
                        # A failed final still ends the segment: the pipeline must
                        # not be left waiting for a transcript that will not come.
                        logger.warning("final transcript failed: %s", exc)
                        text = ""
                    self._emit_marker(stt.SpeechEventType.END_OF_SPEECH, duration)
                    self._emit(stt.SpeechEventType.FINAL_TRANSCRIPT, text, duration)
                    segment = []
                speaking = False
                last_interim_at = time.monotonic()
                continue

            if not isinstance(data, rtc.AudioFrame):
                continue
            segment.append(data)
            if not speaking:
                speaking = True
                last_interim_at = time.monotonic()
                self._emit_marker(stt.SpeechEventType.START_OF_SPEECH)
                continue

            elapsed = time.monotonic() - last_interim_at
            if elapsed < self._opts.interim_interval_seconds:
                continue
            duration = self._duration(segment)
            if duration < MIN_SEGMENT_SECONDS:
                continue
            last_interim_at = time.monotonic()
            audio, duration = wav_bytes_from_frames(segment)
            try:
                text = await self._stt._transcribe(audio)
            except GroqInterimSTTError as exc:
                logger.debug("interim transcript failed: %s", exc)
                continue
            self._emit(stt.SpeechEventType.INTERIM_TRANSCRIPT, text, duration)

        # The input channel closed without a flush: emit the tail so a turn that
        # ends by teardown still produces a transcript.
        if segment:
            audio, duration = wav_bytes_from_frames(segment)
            try:
                text = await self._stt._transcribe(audio)
            except GroqInterimSTTError as exc:
                logger.debug("trailing transcript failed: %s", exc)
                text = ""
            self._emit(stt.SpeechEventType.FINAL_TRANSCRIPT, text, duration)
