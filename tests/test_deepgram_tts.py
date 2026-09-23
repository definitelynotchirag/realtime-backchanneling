import asyncio

import pytest
from livekit import rtc

from blue_machines_baseline import deepgram_tts


def pcm_bytes(samples: int, value: int = 1000) -> bytes:
    import struct

    return struct.pack(f"<{samples}h", *([value] * samples))


class FakeResponse:
    def __init__(self, *, content_type: str = "audio/l16;rate=24000", chunks=()) -> None:
        self.headers = {"Content-Type": content_type}
        self._chunks = list(chunks)
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""

    def close(self) -> None:
        self.closed = True


def synthesize(tts_client) -> list[rtc.AudioFrame]:
    async def collect() -> list[rtc.AudioFrame]:
        return [event.frame async for event in tts_client.synthesize("hello")]

    return asyncio.run(collect())


def test_linear16_header_supplies_the_sample_rate() -> None:
    # Deepgram announces linear16 as audio/l16;rate=..., not audio/pcm.
    assert deepgram_tts.parse_audio_format("audio/l16;rate=24000") == (24000, 1)
    assert deepgram_tts.parse_audio_format("audio/l16;rate=16000;channels=2") == (16000, 2)


def test_audio_streams_at_the_announced_rate(monkeypatch) -> None:
    response = FakeResponse(chunks=[pcm_bytes(2400), pcm_bytes(2400)])
    monkeypatch.setattr(deepgram_tts, "_open_speech", lambda **_kwargs: response)
    tts = deepgram_tts.TTS(api_key="key")

    frames = synthesize(tts)

    assert frames
    assert all(frame.sample_rate == 24000 for frame in frames)
    samples = sum(frame.samples_per_channel for frame in frames)
    # The emitter pads to a frame boundary (+240 samples at 24 kHz); nothing may be lost.
    assert samples >= 4800
    assert samples - 4800 <= 240
    assert response.closed  # the response is always released


def test_the_voice_is_part_of_the_model_id() -> None:
    tts = deepgram_tts.TTS(api_key="key", model="aura-2-helena-en")

    assert tts.model == "aura-2-helena-en"
    assert tts.provider == "deepgram"


def test_failures_are_surfaced(monkeypatch) -> None:
    def boom(**_kwargs: object) -> FakeResponse:
        raise deepgram_tts.DeepgramTTSError("Deepgram speech HTTP 401: invalid credentials")

    monkeypatch.setattr(deepgram_tts, "_open_speech", boom)
    tts = deepgram_tts.TTS(api_key="key")

    with pytest.raises(deepgram_tts.DeepgramTTSError):
        synthesize(tts)


def test_an_api_key_is_required() -> None:
    with pytest.raises(ValueError):
        deepgram_tts.TTS(api_key="")
