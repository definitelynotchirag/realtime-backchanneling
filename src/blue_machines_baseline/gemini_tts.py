"""Gemini text-to-speech provider for the LiveKit agent pipeline.

LiveKit Inference is the project's default TTS, and ElevenLabs is the direct
alternative, but neither is usable in every environment (quota, billing). This
adapter speaks the Gemini API directly so the benchmark can still run against a
real speech provider.

The API returns one base64 PCM blob per request, so this is a chunked
(non-streaming) provider: ``synthesize`` yields the whole utterance's frames.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import urllib.error
import urllib.request
from typing import Any

from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

logger = logging.getLogger("blue-machines-gemini-tts")

SAMPLE_RATE: int = 24000
"""Gemini TTS returns 24 kHz mono signed 16-bit PCM."""

NUM_CHANNELS: int = 1

API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
REQUEST_TIMEOUT_SECONDS: float = 30.0


class GeminiTTSError(RuntimeError):
    """Raised when the Gemini speech endpoint cannot be used."""


def _synthesize_blocking(
    *,
    api_key: str,
    model: str,
    voice: str,
    text: str,
    base_url: str,
    timeout: float,
) -> bytes:
    """Call the Gemini TTS endpoint and return raw PCM bytes."""

    body = json.dumps(
        {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
            },
            "model": model,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/models/{model}:generateContent?key={api_key}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:200]
        raise GeminiTTSError(f"Gemini TTS HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise GeminiTTSError(f"Gemini TTS unreachable: {exc.reason}") from exc

    try:
        encoded = payload["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GeminiTTSError(f"Gemini TTS returned no audio: {str(payload)[:200]}") from exc
    return base64.b64decode(encoded)


class TTS(tts.TTS):
    """A LiveKit TTS provider backed by the Gemini speech endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-3.8-flash-tts",
        voice: str = "Kore",
        base_url: str = API_BASE_URL,
        http_timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for the gemini_tts provider")
        self._api_key = api_key
        self._model_name = model
        self._voice = voice
        self._base_url = base_url.rstrip("/")
        self._http_timeout = http_timeout

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "gemini"

    @property
    def voice(self) -> str:
        return self._voice

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return _ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _ChunkedStream(tts.ChunkedStream):
    """One request per utterance: the endpoint returns the full waveform."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        try:
            pcm = await asyncio.to_thread(
                _synthesize_blocking,
                api_key=self._tts._api_key,
                model=self._tts._model_name,
                voice=self._tts.voice,
                text=self.input_text,
                base_url=self._tts._base_url,
                timeout=self._tts._http_timeout,
            )
        except GeminiTTSError as exc:
            logger.warning("gemini tts failed: %s", exc)
            raise
        if not pcm:
            raise GeminiTTSError("Gemini TTS returned an empty waveform")

        output_emitter.initialize(
            request_id=f"gemini-tts-{id(self):x}",
            sample_rate=self._tts.sample_rate,
            num_channels=self._tts.num_channels,
            mime_type="audio/pcm",
        )
        output_emitter.push(pcm)
        output_emitter.flush()
