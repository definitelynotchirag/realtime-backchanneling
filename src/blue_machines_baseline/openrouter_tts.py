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
from typing import Any

from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

logger = logging.getLogger("blue-machines-openrouter-tts")

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepgram/flux-tts:free"
DEFAULT_VOICE = "flux-alexis-en"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CHANNELS = 1
REQUEST_TIMEOUT_SECONDS: float = 45.0
CHUNK_BYTES: int = 9600
"""Read size: 200 ms of 24 kHz mono 16-bit audio per read."""
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


def _request_body(*, model: str, voice: str, text: str) -> bytes:
    return json.dumps(
        {"model": model, "input": text, "voice": voice, "response_format": "pcm"}
    ).encode()


def _open_speech(*, api_key: str, base_url: str, body: bytes, timeout: float) -> Any:
    """Open the speech request and return the live response.

    The endpoint streams its audio: for a sentence whose synthesis takes ten
    seconds the first bytes arrive in about one and a half, so the caller decides
    whether to read the body incrementally (lower latency) or as a whole.
    """

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
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:200]
        raise OpenRouterTTSError(f"OpenRouter speech HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise OpenRouterTTSError(f"OpenRouter unreachable: {exc.reason}") from exc


def _synthesize_blocking(
    *, api_key: str, base_url: str, model: str, voice: str, text: str, timeout: float
) -> tuple[bytes, int, int]:
    """Read the whole utterance at once (used by batch/smoke paths)."""

    response = _open_speech(
        api_key=api_key,
        base_url=base_url,
        body=_request_body(model=model, voice=voice, text=text),
        timeout=timeout,
    )
    try:
        audio = response.read()
        content_type = response.headers.get("Content-Type")
    finally:
        response.close()
    if not audio:
        raise OpenRouterTTSError("OpenRouter returned no audio")
    rate, channels = parse_audio_format(content_type)
    return audio, rate, channels


class _StreamFormat:
    """Marker carrying the audio format read from the response headers."""

    def __init__(self, sample_rate: int, channels: int) -> None:
        self.sample_rate = sample_rate
        self.channels = channels


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
    """Stream one utterance: push audio as the endpoint produces it.

    Buffering the whole response first is what made this adapter slow - the audio
    is available long before the last byte (1.5 s versus 10.5 s for a long
    sentence), so playback starts while synthesis is still finishing.
    """

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        loop = asyncio.get_running_loop()
        items: asyncio.Queue[object] = asyncio.Queue()
        done = object()

        def produce() -> None:
            """Read the response in chunks, handing each one to the event loop."""

            try:
                response = _open_speech(
                    api_key=self._tts._api_key,
                    base_url=self._tts._base_url,
                    body=_request_body(
                        model=self._tts._model_name, voice=self._tts.voice, text=self.input_text
                    ),
                    timeout=self._tts._http_timeout,
                )
                try:
                    rate, channels = parse_audio_format(response.headers.get("Content-Type"))
                    loop.call_soon_threadsafe(items.put_nowait, _StreamFormat(rate, channels))
                    while True:
                        chunk = response.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        loop.call_soon_threadsafe(items.put_nowait, chunk)
                finally:
                    response.close()
            except Exception as exc:  # noqa: BLE001 - forwarded to the awaiting task
                loop.call_soon_threadsafe(items.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(items.put_nowait, done)

        producer = asyncio.create_task(asyncio.to_thread(produce))
        initialised = False
        pushed = False
        try:
            while True:
                item = await items.get()
                if item is done:
                    break
                if isinstance(item, BaseException):
                    if pushed:
                        # Audio is already playing; a truncated answer is better
                        # than replaying the utterance from the start.
                        logger.warning("speech stream ended early: %s", item)
                        break
                    raise item
                if isinstance(item, _StreamFormat):
                    output_emitter.initialize(
                        request_id=f"openrouter-tts-{uuid.uuid4().hex[:8]}",
                        sample_rate=item.sample_rate,
                        num_channels=item.channels,
                        mime_type="audio/pcm",
                    )
                    initialised = True
                    continue
                if isinstance(item, bytes):
                    if not initialised:
                        raise OpenRouterTTSError("audio arrived before the format header")
                    output_emitter.push(item)
                    pushed = True
        finally:
            await producer
        if not initialised:
            raise OpenRouterTTSError("OpenRouter returned no audio")
        output_emitter.flush()
