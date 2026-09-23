"""Groq speech-to-text over the OpenAI-compatible REST endpoint.

The bundled Groq plugin talks to a realtime websocket that Groq does not serve
for its Whisper models, so this adapter posts each utterance to the batch
endpoint instead. It is deliberately non-streaming: the framework segments the
user's speech with VAD and calls ``recognize`` once per utterance.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import urllib.error
import urllib.request
import uuid
import wave
from typing import Any

from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given

logger = logging.getLogger("blue-machines-groq-stt")

API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
REQUEST_TIMEOUT_SECONDS: float = 30.0
SAMPLE_WIDTH_BYTES = 2
USER_AGENT = "blue-machines-baseline/0.1 (LiveKit backchannel benchmark)"


class GroqSTTError(RuntimeError):
    """Raised when Groq cannot transcribe an utterance."""


def wav_bytes_from_buffer(buffer: AudioBuffer) -> tuple[bytes, float]:
    """Encode recorded frames as a 16-bit PCM WAV plus its duration in seconds."""

    frames = buffer if isinstance(buffer, list) else [buffer]
    if not frames:
        return b"", 0.0
    if len(frames) == 1:
        frame = frames[0]
    elif all(
        frames[0].sample_rate == item.sample_rate and frames[0].num_channels == item.num_channels
        for item in frames
    ):
        frame = rtc.combine_audio_frames(frames)
    else:
        resampler = rtc.AudioResampler(
            input_rate=frames[0].sample_rate, output_rate=frames[0].sample_rate
        )
        combined = [resampler.push(f) for f in frames]
        flat = [item for group in combined for item in group]
        frame = rtc.combine_audio_frames(flat or frames)

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
) -> dict[str, Any]:
    """Send one WAV payload to Groq and return the decoded JSON response."""

    boundary = f"----blue-machines-{uuid.uuid4().hex}"
    parts: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
            ).encode()
        )

    parts.append(
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="utterance.wav"\r\n'
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
    body = b"".join(parts)

    request = urllib.request.Request(
        API_URL,
        data=body,
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
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:200]
        raise GroqSTTError(f"Groq transcription HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise GroqSTTError(f"Groq unreachable: {exc.reason}") from exc


class STT(stt.STT):
    """Batch transcription through Groq's OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "whisper-large-v3-turbo",
        language: str | None = "en",
        http_timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False))
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for the groq_rest provider")
        self._api_key = api_key
        self._model_name = model
        self._language = language
        self._http_timeout = http_timeout

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "groq"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        audio, duration = wav_bytes_from_buffer(buffer)
        if not audio:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(language="en", text="")],
            )
        payload = await asyncio.to_thread(
            _post_transcription,
            api_key=self._api_key,
            model=self._model_name,
            audio=audio,
            language=language if is_given(language) else self._language,
            timeout=self._http_timeout,
        )
        text = str(payload.get("text", "")).strip()
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(language="en", text=text, start_time=0.0, end_time=duration)
            ],
        )
