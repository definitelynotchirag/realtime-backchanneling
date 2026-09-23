"""Text-to-speech through OpenRouter's speech endpoint.

OpenRouter serves several speech models, including a free tier of Deepgram's
Flux TTS, and this project already holds an OpenRouter key for the LLM failover
path. Using it for speech means one key can cover both roles - useful when the
dedicated speech providers run out of quota (Groq's Orpheus allows 3600 speech
tokens per day; a Gemini free project has its own daily cap).

The endpoint returns raw audio rather than JSON, and declares the encoding in the
response's ``Content-Type`` (for example ``audio/pcm;rate=24000;channels=1``).
That header is parsed rather than assumed, so a change of voice or model cannot
silently produce audio at the wrong rate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
import uuid

from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

logger = logging.getLogger("blue-machines-openrouter-tts")

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepgram/flux-tts:free"
DEFAULT_VOICE = "flux-alexis-en"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CHANNELS = 1
REQUEST_TIMEOUT_SECONDS: float = 45.0
USER_AGENT = "blue-machines-baseline/0.1 (LiveKit backchannel benchmark)"


class OpenRouterTTSError(RuntimeError):
    """Raised when OpenRouter cannot synthesize the utterance."""


def parse_audio_format(content_type: str | None) -> tuple[int, int]:
    """Read the sample rate and channel count out of a Content-Type header.

    ``audio/pcm;rate=24000;channels=1`` becomes ``(24000, 1)``. Missing or
    unparsable parameters fall back to the provider's documented defaults.
    """

    if not content_type:
        return DEFAULT_SAMPLE_RATE, DEFAULT_CHANNELS
    parameters: dict[str, str] = {}
    for part in content_type.split(";")[1:]:
        key, _, value = part.partition("=")
        parameters[key.strip().lower()] = value.strip()
    try:
        rate = int(parameters["rate"])
    except (KeyError, ValueError):
        rate = DEFAULT_SAMPLE_RATE
    try:
        channels = int(parameters["channels"])
    except (KeyError, ValueError):
        channels = DEFAULT_CHANNELS
    return rate, channels


def _synthesize_blocking(
    *,
    api_key: str,
    base_url: str,
    model: str,
    voice: str,
    text: str,
    timeout: float,
) -> tuple[bytes, int, int]:
    """Call the speech endpoint and return raw PCM plus its declared format."""

    body = json.dumps(
        {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": "pcm",
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/audio/speech",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            audio = response.read()
            content_type = response.headers.get("Content-Type")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:200]
        raise OpenRouterTTSError(f"OpenRouter speech HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise OpenRouterTTSError(f"OpenRouter unreachable: {exc.reason}") from exc
    if not audio:
        raise OpenRouterTTSError("OpenRouter returned no audio")
    rate, channels = parse_audio_format(content_type)
    return audio, rate, channels


class TTS(tts.TTS):
    """A LiveKit TTS provider backed by OpenRouter's speech endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        base_url: str = DEFAULT_BASE_URL,
        http_timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=DEFAULT_SAMPLE_RATE,
            num_channels=DEFAULT_CHANNELS,
        )
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for the openrouter_tts provider")
        self._api_key = api_key
        self._model_name = model
        self._voice = voice
        self._base_url = base_url
        self._http_timeout = http_timeout

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "openrouter"

    @property
    def voice(self) -> str:
        return self._voice

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return _ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _ChunkedStream(tts.ChunkedStream):
    """One request per utterance: the endpoint returns the whole waveform."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        audio, sample_rate, channels = await asyncio.to_thread(
            _synthesize_blocking,
            api_key=self._tts._api_key,
            base_url=self._tts._base_url,
            model=self._tts._model_name,
            voice=self._tts.voice,
            text=self.input_text,
            timeout=self._tts._http_timeout,
        )
        output_emitter.initialize(
            request_id=f"openrouter-tts-{uuid.uuid4().hex[:8]}",
            sample_rate=sample_rate,
            num_channels=channels,
            mime_type="audio/pcm",
        )
        output_emitter.push(audio)
        output_emitter.flush()
