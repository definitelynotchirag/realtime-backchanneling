"""Deepgram streaming speech-to-text with native interim transcripts.

Deepgram's live endpoint is a websocket: audio goes up as binary frames, results
come back as JSON, and the service marks each result with ``is_final`` (this text
will not change again) and ``speech_final`` (the speaker stopped). That is exactly
what a turn-taking pipeline needs, and it is why this adapter exists: the batch
providers can only report what the user said after their turn has ended, so the
semantic policy in ``jev.py`` had nothing to work with until then.

Audio is sent as 16 kHz linear16 mono, which the SDK's stream resamples for us.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass

import aiohttp
from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import AudioBuffer

logger = logging.getLogger("blue-machines-deepgram-stt")

DEFAULT_BASE_URL = "wss://api.deepgram.com/v1/listen"
DEFAULT_MODEL = "nova-3"
DEFAULT_LANGUAGE = "en"
DEFAULT_ENDPOINTING_MS = 300
"""Silence Deepgram waits for before closing a transcript. The pipeline's own
end-of-turn logic is separate; this only decides when words stop changing."""

SAMPLE_RATE = 16000
NUM_CHANNELS = 1
CONNECT_TIMEOUT_SECONDS = 10.0


class DeepgramSTTError(RuntimeError):
    """Raised when Deepgram cannot transcribe the audio."""


@dataclass(frozen=True)
class _Options:
    api_key: str
    model: str
    language: str
    endpointing_ms: int
    base_url: str


def events_from_message(payload: dict[str, object], *, language: str) -> list[stt.SpeechEvent]:
    """Translate one Deepgram message into the events the pipeline consumes.

    Kept pure so the mapping is testable without a socket: ``is_final`` marks a
    transcript that will not change again, ``speech_final`` marks the end of the
    user's turn.
    """

    if payload.get("type") != "Results":
        return []
    channel = payload.get("channel")
    alternatives = channel.get("alternatives") if isinstance(channel, dict) else None
    first = alternatives[0] if isinstance(alternatives, list) and alternatives else {}
    text = str(first.get("transcript", "")).strip() if isinstance(first, dict) else ""
    if not text:
        return []

    start = float(payload.get("start") or 0.0)
    duration = float(payload.get("duration") or 0.0)
    confidence = float(first.get("confidence") or 0.0) if isinstance(first, dict) else 0.0
    event_type = (
        stt.SpeechEventType.FINAL_TRANSCRIPT
        if payload.get("is_final")
        else stt.SpeechEventType.INTERIM_TRANSCRIPT
    )
    events = [
        stt.SpeechEvent(
            type=event_type,
            alternatives=[
                stt.SpeechData(
                    language=language,
                    text=text,
                    confidence=confidence,
                    start_time=start,
                    end_time=start + duration,
                )
            ],
        )
    ]
    if payload.get("speech_final"):
        events.append(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
    return events


def _listen_url(opts: _Options) -> str:
    query = "&".join(
        [
            f"model={opts.model}",
            f"language={opts.language}",
            "interim_results=true",
            "smart_format=true",
            f"endpointing={opts.endpointing_ms}",
            "encoding=linear16",
            f"sample_rate={SAMPLE_RATE}",
            f"channels={NUM_CHANNELS}",
        ]
    )
    return f"{opts.base_url}?{query}"


class STT(stt.STT):
    """Streaming transcription over Deepgram's websocket API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        language: str = DEFAULT_LANGUAGE,
        endpointing_ms: int = DEFAULT_ENDPOINTING_MS,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY is required for the deepgram provider")
        self._opts = _Options(
            api_key=api_key,
            model=model,
            language=language,
            endpointing_ms=endpointing_ms,
            base_url=base_url,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "deepgram"

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.RecognizeStream:
        # Passing sample_rate lets the SDK resample whatever the room provides.
        return _Stream(
            stt=self, opts=self._opts, conn_options=conn_options, sample_rate=SAMPLE_RATE
        )

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        """Batch transcription, for callers that are not streaming."""

        frames = buffer if isinstance(buffer, list) else [buffer]
        if not frames:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(language=self._opts.language, text="")],
            )
        audio, _duration = _wav_bytes(frames)
        headers = {
            "Authorization": f"Token {self._opts.api_key}",
            "Content-Type": "audio/wav",
        }
        url = self._opts.base_url.replace("wss://", "https://")
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{url}?model={self._opts.model}&smart_format=true",
                data=audio,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=conn_options.timeout),
            ) as response:
                if response.status != 200:
                    detail = await response.text()
                    raise DeepgramSTTError(
                        f"Deepgram listen HTTP {response.status}: {detail[:200]}"
                    )
                payload = await response.json()
        alternatives = payload.get("results", {}).get("channels", [{}])[0].get("alternatives", [])
        text = str(alternatives[0].get("transcript", "")).strip() if alternatives else ""
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=self._opts.language, text=text)],
        )


def _wav_bytes(frames: list[rtc.AudioFrame]) -> tuple[bytes, float]:
    import io
    import wave

    frame = frames[0] if len(frames) == 1 else rtc.combine_audio_frames(frames)
    data = frame.data
    payload = bytes(data) if isinstance(data, memoryview) else bytes(data)
    duration = frame.samples_per_channel / max(1, frame.sample_rate)
    with io.BytesIO() as raw:
        with wave.open(raw, "wb") as handle:
            handle.setnchannels(frame.num_channels)
            handle.setsampwidth(2)
            handle.setframerate(frame.sample_rate)
            handle.writeframes(payload)
        return raw.getvalue(), duration


class _Stream(stt.RecognizeStream):
    """Audio up as binary frames, results down as JSON."""

    def __init__(
        self,
        *,
        stt: STT,
        opts: _Options,
        conn_options: APIConnectOptions,
        sample_rate: int,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=sample_rate)
        self._stt = stt
        self._opts = opts
        self._started_speaking = False
        # Usage is reported per final transcript as the audio it covered, which is what
        # the SDK turns into STTMetrics for the session. Without it the pipeline has no
        # speech-to-text usage at all for this provider.
        self._audio_seconds = 0.0
        self._reported_audio_seconds = 0.0

    async def _run(self) -> None:
        try:
            session = aiohttp.ClientSession()
        except Exception as exc:  # noqa: BLE001 - surface as a clear provider error
            raise DeepgramSTTError(f"Deepgram session failed: {exc}") from exc
        try:
            async with session.ws_connect(
                _listen_url(self._opts),
                headers={"Authorization": f"Token {self._opts.api_key}"},
                timeout=aiohttp.ClientWSTimeout(ws_close=CONNECT_TIMEOUT_SECONDS),
                max_msg_size=0,
            ) as ws:
                sender = asyncio.create_task(self._send_audio(ws), name="deepgram_stt_send")
                try:
                    async for message in ws:
                        if message.type == aiohttp.WSMsgType.TEXT:
                            self._handle_text(message.data)
                        elif message.type == aiohttp.WSMsgType.ERROR:
                            logger.warning("deepgram socket error: %s", ws.exception())
                            break
                finally:
                    sender.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await sender
        finally:
            await session.close()

    def push_frame(self, frame: rtc.AudioFrame) -> None:
        # Deepgram's live API returns no usage, so the audio the stream is given is
        # counted here and reported per final transcript; the SDK turns that into the
        # session's speech-to-text usage.
        self._audio_seconds += frame.samples_per_channel / max(1, frame.sample_rate)
        super().push_frame(frame)

    async def _send_audio(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                await ws.send_str(json.dumps({"type": "Finalize"}))
                continue
            if isinstance(item, rtc.AudioFrame):
                await ws.send_bytes(bytes(item.data))
        # The channel closed: ask Deepgram to flush what it has and close.
        try:
            await ws.send_str(json.dumps({"type": "Finalize"}))
            await ws.send_str(json.dumps({"type": "CloseStream"}))
        except (ConnectionError, RuntimeError):
            logger.debug("deepgram socket already closed while shutting down")

    def _handle_text(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("deepgram sent unparsable message")
            return
        if not isinstance(payload, dict):
            return
        events = events_from_message(payload, language=self._opts.language)
        for event in events:
            if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT and not self._started_speaking:
                self._started_speaking = True
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )
            if event.type == stt.SpeechEventType.END_OF_SPEECH:
                self._started_speaking = False
            self._event_ch.send_nowait(event)
            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                self._event_ch.send_nowait(self._usage_event())

    def _usage_event(self) -> stt.SpeechEvent:
        """The audio covered by the transcript that has just been reported."""

        covered = self._audio_seconds - self._reported_audio_seconds
        self._reported_audio_seconds = self._audio_seconds
        return stt.SpeechEvent(
            type=stt.SpeechEventType.RECOGNITION_USAGE,
            recognition_usage=stt.RecognitionUsage(audio_duration=max(0.0, covered)),
        )
