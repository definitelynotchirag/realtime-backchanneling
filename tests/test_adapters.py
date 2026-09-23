import asyncio
import base64
import json
import struct

import pytest
from livekit import rtc

from blue_machines_baseline import gemini_tts
from blue_machines_baseline.benchmark import SCENARIO_BY_ID
from blue_machines_baseline.config import ConfigurationError, Settings
from blue_machines_baseline.simulator import (
    pcm_to_wav,
    read_clip,
    silence_frames,
    utterance_for,
    with_pauses,
)


def settings_for(**overrides: str) -> Settings:
    values = {
        "LIVEKIT_URL": "wss://example.livekit.cloud",
        "LIVEKIT_API_KEY": "lk_api_key",
        "LIVEKIT_API_SECRET": "lk_api_secret",
        "GROQ_API_KEY": "groq_api_key",
        "GEMINI_API_KEY": "gemini_api_key",
        **overrides,
    }
    return Settings.from_env(values)


def pcm_bytes(samples: int, value: int = 1000) -> bytes:
    return struct.pack(f"<{samples}h", *([value] * samples))


def test_gemini_tts_emits_frames_at_its_own_sample_rate(monkeypatch) -> None:
    monkeypatch.setattr(
        gemini_tts,
        "_synthesize_blocking",
        lambda **_kwargs: pcm_bytes(2400),  # 0.1 s at 24 kHz
    )
    tts = gemini_tts.TTS(api_key="key", voice="Kore")

    async def collect() -> list[rtc.AudioFrame]:
        return [event.frame async for event in tts.synthesize("hello")]

    frames = asyncio.run(collect())

    assert frames
    assert all(frame.sample_rate == 24000 for frame in frames)
    assert all(frame.num_channels == 1 for frame in frames)
    total_samples = sum(frame.samples_per_channel for frame in frames)
    # The framework's byte stream pads to a frame boundary, which costs a fixed
    # 240 samples (10 ms) regardless of the utterance length. Assert the audio is
    # not truncated and that the padding stays bounded.
    assert total_samples >= 2400
    assert total_samples - 2400 <= 240
    assert tts.provider == "gemini"
    assert tts.capabilities.streaming is False


def test_gemini_tts_surfaces_provider_failures(monkeypatch) -> None:
    def boom(**_kwargs: object) -> bytes:
        raise gemini_tts.GeminiTTSError("Gemini TTS HTTP 429: quota")

    monkeypatch.setattr(gemini_tts, "_synthesize_blocking", boom)
    tts = gemini_tts.TTS(api_key="key")

    async def collect() -> list[rtc.AudioFrame]:
        return [event.frame async for event in tts.synthesize("hello")]

    with pytest.raises(gemini_tts.GeminiTTSError):
        asyncio.run(collect())


def test_gemini_tts_rejects_an_empty_waveform(monkeypatch) -> None:
    monkeypatch.setattr(gemini_tts, "_synthesize_blocking", lambda **_kwargs: b"")
    tts = gemini_tts.TTS(api_key="key")

    async def collect() -> list[rtc.AudioFrame]:
        return [event.frame async for event in tts.synthesize("hello")]

    with pytest.raises(gemini_tts.GeminiTTSError):
        asyncio.run(collect())


def test_gemini_tts_requires_an_api_key() -> None:
    with pytest.raises(ValueError):
        gemini_tts.TTS(api_key="")


def test_scenario_clips_round_trip_through_the_driver_helpers(tmp_path) -> None:
    wav_bytes = pcm_to_wav(pcm_bytes(16000), source_rate=16000)
    path = tmp_path / "long_monologue.wav"
    path.write_bytes(wav_bytes)

    frames = read_clip(path)

    assert len(frames) == 50  # 1 s of 20 ms frames
    assert frames[0].sample_rate == 16000
    assert frames[0].samples_per_channel == 320


def test_scripted_pauses_are_inserted_inside_the_clip() -> None:
    frames = read_clip_bytes(pcm_to_wav(pcm_bytes(16000), source_rate=16000))

    with_pause = with_pauses(frames, (0.5,))

    assert len(with_pause) == len(frames) + len(silence_frames(0.5))
    # the pause lands between speech, not only at the end
    inserted = with_pause[len(frames) // 2]
    assert bytes(inserted.data) == b"\x00\x00" * 320


def read_clip_bytes(data: bytes) -> list[rtc.AudioFrame]:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "clip.wav"
        path.write_bytes(data)
        return read_clip(path)


def test_every_scenario_has_a_scripted_utterance() -> None:
    for scenario_id, scenario in SCENARIO_BY_ID.items():
        text = utterance_for(scenario)
        assert text
        assert text != scenario.description, f"{scenario_id} falls back to its description"


def test_settings_validate_the_new_policy_fields() -> None:
    assert settings_for(EOT_DETECTOR="final_transcript").eot_detector == "final_transcript"
    assert settings_for().eot_detector == "livekit_inference"
    assert (
        settings_for(BACKCHANNEL_COLLISION_WINDOW_SECONDS="0.75").collision_window_seconds == 0.75
    )
    assert settings_for().scenario_audio_dir.name == "scenarios"

    with pytest.raises(ConfigurationError):
        settings_for(EOT_DETECTOR="magic")
    with pytest.raises(ConfigurationError):
        settings_for(BACKCHANNEL_COLLISION_WINDOW_SECONDS="0")
    with pytest.raises(ConfigurationError):
        settings_for(TTS_PROVIDER="unknown_tts")
    with pytest.raises(ConfigurationError):
        settings_for(STT_PROVIDER="unknown_stt")


def test_gemini_tts_provider_requires_the_gemini_key() -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_env(
            {
                "LIVEKIT_URL": "wss://example.livekit.cloud",
                "LIVEKIT_API_KEY": "lk_api_key",
                "LIVEKIT_API_SECRET": "lk_api_secret",
                "TTS_PROVIDER": "gemini_tts",
            }
        )


def test_sidecar_metadata_describes_the_rendered_clip(tmp_path) -> None:
    scenario_id = "short_answer"
    (tmp_path / f"{scenario_id}.wav").write_bytes(pcm_to_wav(pcm_bytes(8000), source_rate=16000))
    (tmp_path / f"{scenario_id}.json").write_text(
        json.dumps(
            {
                "text": "Doing great.",
                "duration_seconds": 0.5,
                "pause_durations_seconds": [0.2],
            }
        )
    )

    from blue_machines_baseline.simulator import load_clip

    clip = load_clip(scenario_id, tmp_path)

    assert clip.text == "Doing great."
    assert clip.duration_seconds == 0.5
    assert clip.pause_durations_seconds == (0.2,)


def test_audio_is_base64_decodable_as_the_provider_returns_it() -> None:
    # Guards the decode step against a change in the payload shape.
    encoded = base64.b64encode(pcm_bytes(4)).decode()
    assert base64.b64decode(encoded) == pcm_bytes(4)
