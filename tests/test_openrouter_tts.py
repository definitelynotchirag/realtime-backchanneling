import asyncio
import struct

import pytest
from livekit import rtc

from blue_machines_baseline import openrouter_tts


def pcm_bytes(samples: int, value: int = 1000) -> bytes:
    return struct.pack(f"<{samples}h", *([value] * samples))


def test_audio_format_is_read_from_the_response_header() -> None:
    # The endpoint declares the encoding; assuming it would silently play audio
    # at the wrong rate if the voice or model changed.
    assert openrouter_tts.parse_audio_format("audio/pcm;rate=24000;channels=1") == (24000, 1)
    assert openrouter_tts.parse_audio_format("audio/pcm; rate=16000; channels=2") == (16000, 2)
    assert openrouter_tts.parse_audio_format("audio/pcm") == (24000, 1)
    assert openrouter_tts.parse_audio_format(None) == (24000, 1)
    assert openrouter_tts.parse_audio_format("audio/pcm;rate=abc") == (24000, 1)


def test_synthesis_uses_the_declared_format(monkeypatch) -> None:
    monkeypatch.setattr(
        openrouter_tts,
        "_synthesize_blocking",
        lambda **_kwargs: (pcm_bytes(4800), 16000, 1),  # 0.3 s at 16 kHz
    )
    tts = openrouter_tts.TTS(api_key="key")

    async def collect() -> list[rtc.AudioFrame]:
        return [event.frame async for event in tts.synthesize("hello")]

    frames = asyncio.run(collect())

    assert frames
    assert all(frame.sample_rate == 16000 for frame in frames)
    assert all(frame.num_channels == 1 for frame in frames)
    assert sum(frame.samples_per_channel for frame in frames) >= 4800
    assert tts.provider == "openrouter"
    assert tts.capabilities.streaming is False


def test_provider_failures_are_surfaced(monkeypatch) -> None:
    def boom(**_kwargs: object) -> tuple[bytes, int, int]:
        raise openrouter_tts.OpenRouterTTSError("OpenRouter speech HTTP 402: insufficient credits")

    monkeypatch.setattr(openrouter_tts, "_synthesize_blocking", boom)
    tts = openrouter_tts.TTS(api_key="key")

    async def collect() -> list[rtc.AudioFrame]:
        return [event.frame async for event in tts.synthesize("hello")]

    with pytest.raises(openrouter_tts.OpenRouterTTSError):
        asyncio.run(collect())


def test_an_api_key_is_required() -> None:
    with pytest.raises(ValueError):
        openrouter_tts.TTS(api_key="")
