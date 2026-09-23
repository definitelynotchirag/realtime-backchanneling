"""Text-to-speech through Deepgram's speak endpoint.

Deepgram streams the audio back and documents that you can play it as soon as the
first byte arrives, which is exactly what the agent pipeline wants: the first word
is audible while the rest of the reply is still being synthesised.

Two details differ from the OpenRouter route and are handled here:

- the audio is requested as ``linear16`` and announced as ``audio/l16;rate=24000``
  rather than ``audio/pcm``, so the format is read from the header rather than
  assumed;
- the voice is part of the model id (``aura-2-thalia-en``), so one setting covers
  both voice and language.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions

from . import tts_http

logger = logging.getLogger("blue-machines-deepgram-tts")

parse_audio_format = tts_http.parse_audio_format
"""Re-exported so the adapter's format handling is testable where it is used."""

DEFAULT_BASE_URL = "https://api.deepgram.com/v1"
DEFAULT_MODEL = "aura-2-thalia-en"
SAMPLE_RATE = 24000
USER_AGENT = "blue-machines-baseline/0.1 (LiveKit backchannel benchmark)"

REQUEST_TIMEOUT_SECONDS: float = 45.0
"""Also the read timeout between chunks while the response streams."""


class DeepgramTTSError(RuntimeError):
    """Raised when Deepgram cannot synthesize the utterance."""


def _request_body(text: str) -> bytes:
    return json.dumps({"text": text}).encode()


def _open_speech(
    *, api_key: str, base_url: str, model: str, text: str, sample_rate: int, timeout: float
) -> tts_http.StreamedResponse:
    """Open the speak request; the caller reads the audio as it streams."""

    query = f"model={model}&encoding=linear16&sample_rate={sample_rate}&container=none"
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/speak?{query}",
        data=_request_body(text),
        headers={
            # Deepgram authenticates with its own scheme, not Bearer.
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:200]
        raise DeepgramTTSError(f"Deepgram speech HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise DeepgramTTSError(f"Deepgram unreachable: {exc.reason}") from exc


class TTS(tts.TTS):
    """A LiveKit TTS provider backed by Deepgram's streaming speak endpoint."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        sample_rate: int = SAMPLE_RATE,
        http_timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY is required for the deepgram_tts provider")
        self._api_key = api_key
        self._model_name = model
        self._base_url = base_url
        self._sample_rate = sample_rate
        self._http_timeout = http_timeout

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def provider(self) -> str:
        return "deepgram"

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return _ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _ChunkedStream(tts.ChunkedStream):
    """Stream one utterance: audio is forwarded as Deepgram produces it."""

    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        await tts_http.stream_pcm_response(
            lambda: _open_speech(
                api_key=self._tts._api_key,
                base_url=self._tts._base_url,
                model=self._tts._model_name,
                text=self.input_text,
                sample_rate=self._tts._sample_rate,
                timeout=self._tts._http_timeout,
            ),
            output_emitter,
            request_id_prefix="deepgram-tts",
            provider="Deepgram",
        )
