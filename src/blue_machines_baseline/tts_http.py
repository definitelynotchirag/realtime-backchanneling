"""Shared plumbing for HTTP text-to-speech providers that stream PCM.

Both the OpenRouter and Deepgram adapters speak the same protocol shape: POST a
JSON body, receive a chunked audio response, and read it incrementally so playback
can begin before the utterance is complete. This module owns that machinery - the
provider adapters only supply how to open the request and how to name the audio.

The provider's own ``Content-Type`` declares the format (``audio/pcm;rate=24000``,
``audio/l16;rate=24000``); it is parsed rather than assumed, so changing voice or
model cannot silently play at the wrong rate.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from typing import Any, Protocol

from livekit.agents import tts

logger = logging.getLogger("blue-machines-tts-http")

DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CHANNELS = 1
CHUNK_BYTES = 4800
"""Read size: 100 ms of 24 kHz mono 16-bit audio per read."""

FRAME_MS = 100
"""Frame size handed to the audio emitter.

The emitter buffers incoming bytes into fixed frames before releasing them, so a
large frame delays the first audible sample by that much. 100 ms keeps the
latency contribution small while staying a sensible frame for the pipeline.
"""


class StreamedResponse(Protocol):
    """The part of an HTTP response this module reads."""

    headers: Any

    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...


class PcmStreamError(RuntimeError):
    """Raised when a speech provider returns no usable audio."""


class PcmStreamFormat:
    """Marker carrying the audio format read from the response headers."""

    def __init__(self, sample_rate: int, channels: int) -> None:
        self.sample_rate = sample_rate
        self.channels = channels


def parse_audio_format(content_type: str | None) -> tuple[int, int]:
    """Read the sample rate and channel count out of a Content-Type header.

    ``audio/pcm;rate=24000;channels=1`` and ``audio/l16;rate=24000`` both become
    ``(24000, 1)``. Missing or unparsable parameters fall back to the defaults.
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


async def stream_pcm_response(
    open_response: Callable[[], StreamedResponse],
    output_emitter: tts.AudioEmitter,
    *,
    request_id_prefix: str,
    provider: str,
) -> None:
    """Read an audio response in chunks, feeding frames out as they arrive.

    ``open_response`` is called on a worker thread because the request is blocking.
    Audio already handed to the emitter is never replayed: if the stream fails
    halfway through, the caller gets a truncated answer rather than a duplicated
    one, which is the better failure for speech.
    """

    loop = asyncio.get_running_loop()
    items: asyncio.Queue[object] = asyncio.Queue()
    done = object()

    def produce() -> None:
        try:
            response = open_response()
            try:
                rate, channels = parse_audio_format(response.headers.get("Content-Type"))
                loop.call_soon_threadsafe(items.put_nowait, PcmStreamFormat(rate, channels))
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
                    logger.warning("%s stream ended early: %s", provider, item)
                    break
                raise item
            if isinstance(item, PcmStreamFormat):
                output_emitter.initialize(
                    request_id=f"{request_id_prefix}-{uuid.uuid4().hex[:8]}",
                    sample_rate=item.sample_rate,
                    num_channels=item.channels,
                    mime_type="audio/pcm",
                    frame_size_ms=FRAME_MS,
                )
                initialised = True
                continue
            if isinstance(item, bytes):
                if not initialised:
                    raise PcmStreamError("audio arrived before the format header")
                output_emitter.push(item)
                pushed = True
    finally:
        await producer
    if not initialised:
        raise PcmStreamError(f"{provider} returned no audio")
    output_emitter.flush()
