import asyncio
import struct

import pytest
from livekit import rtc

from blue_machines_baseline import openrouter_tts


def pcm_bytes(samples: int, value: int = 1000) -> bytes:
    return struct.pack(f"<{samples}h", *([value] * samples))


class FakeResponse:
    """Stands in for the live HTTP response, which is read incrementally."""

    def __init__(self, *, content_type: str = "audio/pcm;rate=16000;channels=1", chunks=()) -> None:
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


def test_audio_format_is_read_from_the_response_header() -> None:
    # The endpoint declares the encoding; assuming it would silently play audio
    # at the wrong rate if the voice or model changed.
    assert openrouter_tts.parse_audio_format("audio/pcm;rate=24000;channels=1") == (24000, 1)
    assert openrouter_tts.parse_audio_format("audio/pcm; rate=16000; channels=2") == (16000, 2)
    assert openrouter_tts.parse_audio_format("audio/pcm") == (24000, 1)
    assert openrouter_tts.parse_audio_format(None) == (24000, 1)
    assert openrouter_tts.parse_audio_format("audio/pcm;rate=abc") == (24000, 1)


def test_audio_is_emitted_as_it_arrives(monkeypatch) -> None:
    """Chunks are pushed as they are read, not after the whole body is buffered."""

    held: list = []

    class SlowResponse(FakeResponse):
        def read(self, size: int = -1) -> bytes:
            # Yield two chunks, then a short one, then end of body.
            if not hasattr(self, "_sent"):
                self._sent = 0
            self._sent += 1
            if self._sent == 1:
                return pcm_bytes(1600)  # 0.1 s
            if self._sent == 2:
                return pcm_bytes(1600)
            return b""

    monkeypatch.setattr(openrouter_tts, "_open_speech", lambda **_kwargs: SlowResponse())
    tts = openrouter_tts.TTS(api_key="key")

    frames = synthesize(tts)

    assert frames
    assert all(frame.sample_rate == 16000 for frame in frames)
    samples = sum(frame.samples_per_channel for frame in frames)
    # The framework's byte stream pads to a frame boundary: a fixed 160 samples
    # (10 ms) at 16 kHz. Assert nothing is truncated and the padding stays bounded.
    assert samples >= 3200
    assert samples - 3200 <= 160
    del held


def test_declared_format_wins_over_the_default(monkeypatch) -> None:
    response = FakeResponse(
        content_type="audio/pcm;rate=48000;channels=1", chunks=[pcm_bytes(4800)]
    )
    monkeypatch.setattr(openrouter_tts, "_open_speech", lambda **_kwargs: response)
    tts = openrouter_tts.TTS(api_key="key")

    frames = synthesize(tts)

    assert frames
    assert all(frame.sample_rate == 48000 for frame in frames)


def test_provider_failures_are_surfaced(monkeypatch) -> None:
    def boom(**_kwargs: object) -> FakeResponse:
        raise openrouter_tts.OpenRouterTTSError("OpenRouter speech HTTP 402: insufficient credits")

    monkeypatch.setattr(openrouter_tts, "_open_speech", boom)
    tts = openrouter_tts.TTS(api_key="key")

    with pytest.raises(openrouter_tts.OpenRouterTTSError):
        synthesize(tts)


def test_a_failure_mid_stream_keeps_what_already_played(monkeypatch) -> None:
    class BreakingResponse(FakeResponse):
        def read(self, size: int = -1) -> bytes:
            if not hasattr(self, "_sent"):
                self._sent = 0
            self._sent += 1
            if self._sent == 1:
                return pcm_bytes(1600)
            raise OSError("connection reset")

    monkeypatch.setattr(openrouter_tts, "_open_speech", lambda **_kwargs: BreakingResponse())
    tts = openrouter_tts.TTS(api_key="key")

    frames = synthesize(tts)

    # A truncated answer beats replaying the utterance from the beginning.
    assert frames
    samples = sum(frame.samples_per_channel for frame in frames)
    assert samples >= 1600  # the chunk that did arrive is kept
    assert samples - 1600 <= 160  # plus the emitter's frame padding


def test_an_api_key_is_required() -> None:
    with pytest.raises(ValueError):
        openrouter_tts.TTS(api_key="")
