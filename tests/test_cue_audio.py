"""Cue audio: how a phrase becomes a playable frame, and what happens when it cannot.

The property under test is that a cue stays a *cue*: short, at a known level, in one
shape whichever provider rendered it, and never silently swapped for a much slower
on-demand synthesis.
"""

from __future__ import annotations

import asyncio
import wave
from array import array
from dataclasses import dataclass, field

import pytest
from livekit import rtc

from blue_machines_baseline import agent
from blue_machines_baseline.cue_audio import (
    CUE_SAMPLE_RATE,
    cue_frame,
    cue_phrases,
    cue_slug,
    normalize_level,
    synthesize_cue_frames,
    trim_silence,
)
from blue_machines_baseline.events import EventRecorder


def _tone(samples: int, *, rate: int, value: int = 1000) -> array:
    return array("h", [value] * samples)


def _frame(samples: array, *, rate: int) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=rate,
        num_channels=1,
        samples_per_channel=len(samples),
    )


def test_cue_slugs_are_file_safe_and_stable() -> None:
    assert cue_slug("I see") == "i-see"
    assert cue_slug("mm-hmm") == "mm-hmm"
    assert cue_slug("makes sense") == "makes-sense"


def test_mm_hmm_is_respelled_but_keeps_its_cue_name() -> None:
    """A literal "mm-hmm" gains a trailing hum on the provider, so it is spoken as
    "mm-hm" - the cue name, the file name and the event log keep the canonical text."""

    assert cue_phrases(["mm-hmm", "I see"]) == {"mm-hmm": "mm-hm", "I see": "I see"}


def test_padding_is_trimmed_and_the_cue_is_normalized() -> None:
    speech, padding = 480, 2000  # 10 ms of speech inside ~42 ms of padding either side
    padded = (
        _tone(padding, rate=48000, value=0)
        + _tone(speech, rate=48000, value=8000)
        + _tone(padding, rate=48000, value=0)
    )

    trimmed = trim_silence(padded, sample_rate=48000)
    levelled = normalize_level(trimmed)

    margin = 48000 * 20 // 1000  # KEEP_SILENCE_MS either side
    assert len(trimmed) == speech + 2 * margin
    assert len(trimmed) < len(padded)
    assert max(abs(sample) for sample in levelled) == pytest.approx(0.70 * 32767, rel=0.01)


def test_a_very_quiet_cue_is_not_amplified_without_limit() -> None:
    """Levelling a near-silent render turns its noise floor into a hiss."""

    levelled = normalize_level(_tone(480, rate=48000, value=200))

    assert max(abs(sample) for sample in levelled) == 800  # the 4x cap, not 0.70 peak


def test_a_provider_frame_becomes_one_canonical_cue_frame() -> None:
    """Deepgram speaks at 24 kHz and the cues are held at 48 kHz."""

    cue = cue_frame(_frame(_tone(4800, rate=24000), rate=24000), phrase="uh-huh")

    assert cue.sample_rate == CUE_SAMPLE_RATE
    assert cue.num_channels == 1
    assert cue.samples_per_channel == pytest.approx(4800 * 2, rel=0.02)


def test_a_cue_that_is_too_long_is_rejected() -> None:
    """A long acknowledgement holds the floor instead of acknowledging."""

    too_long = _frame(_tone(CUE_SAMPLE_RATE * 2, rate=CUE_SAMPLE_RATE), rate=CUE_SAMPLE_RATE)

    with pytest.raises(ValueError, match="holds the floor"):
        cue_frame(too_long, phrase="go on")


def test_silent_audio_is_rejected_rather_than_cached() -> None:
    """A silent cue would play nothing while the log claimed an audible cue."""

    with pytest.raises(ValueError, match="no audible audio"):
        cue_frame(_frame(_tone(4800, rate=48000, value=0), rate=48000), phrase="mm-hm")


@dataclass
class FakeProvider:
    """Yields one short tone per phrase, and can be made to fail."""

    error: Exception | None = None
    calls: list[str] = field(default_factory=list)

    def synthesize(self, text: str):  # noqa: ANN201 - mirrors the adapter shape
        self.calls.append(text)
        return self._stream(text)

    async def _stream(self, text: str):
        if self.error is not None:
            raise self.error
        yield type("Event", (), {"frame": _frame(_tone(4800, rate=24000), rate=24000)})()


def test_every_cue_in_the_bank_is_synthesized() -> None:
    async def scenario() -> None:
        provider = FakeProvider()

        cues = await synthesize_cue_frames(provider, cue_phrases(["mm-hmm", "go on"]))

        assert set(cues) == {"mm-hmm", "go on"}  # keyed by cue text, not by phrase
        assert provider.calls == ["mm-hm", "go on"]
        assert all(frame.sample_rate == CUE_SAMPLE_RATE for frame in cues.values())

    asyncio.run(scenario())


def test_a_failing_provider_falls_back_to_the_committed_clips(tmp_path, monkeypatch) -> None:
    """The cues must keep working when the speech provider does not.

    This is also the reason the clips are committed at all: a rate-limited provider
    silences replies but must not silence acknowledgements.
    """

    async def failing(*_args, **_kwargs):
        raise RuntimeError("provider is rate limited")

    monkeypatch.setattr(agent, "synthesize_backchannel_clips", failing)
    recorder = EventRecorder(tmp_path / "events.jsonl", scenario_id="s", mode="backchannel")

    clips = asyncio.run(agent.load_cue_clips(agent.Settings.from_env(_settings_env()), recorder))

    assert {"mm-hmm", "uh-huh", "I see"} <= set(clips)
    import json

    events = [line for line in (tmp_path / "events.jsonl").read_text().splitlines() if line.strip()]
    ready = [json.loads(line) for line in events if "clips_ready" in line]
    assert ready and ready[-1]["data"]["source"] == "assets"
    assert ready[-1]["data"]["fallback_reason"] == "RuntimeError"


def test_the_asset_source_never_calls_the_provider(tmp_path, monkeypatch) -> None:
    async def unexpected(*_args, **_kwargs):
        raise AssertionError("the assets source must not synthesize")

    monkeypatch.setattr(agent, "synthesize_backchannel_clips", unexpected)
    settings = agent.Settings.from_env({**_settings_env(), "BACKCHANNEL_CLIP_SOURCE": "assets"})
    recorder = EventRecorder(tmp_path / "events.jsonl", scenario_id="s", mode="backchannel")

    clips = asyncio.run(agent.load_cue_clips(settings, recorder))

    assert clips  # the committed bank
    assert len(clips) >= 10


def _settings_env() -> dict[str, str]:
    return {
        "LIVEKIT_URL": "wss://example.livekit.cloud",
        "LIVEKIT_API_KEY": "k",
        "LIVEKIT_API_SECRET": "s",
        "GEMINI_API_KEY": "g",
    }


def test_every_bank_cue_has_a_committed_clip() -> None:
    """The fallback is only a fallback if it covers the whole bank."""

    missing = [
        cue
        for cue in agent.CUE_TEXTS
        if not (agent.BACKCHANNEL_CLIP_DIR / f"{cue_slug(cue)}.wav").exists()
    ]

    assert missing == []


def test_the_committed_clips_are_short_enough_to_be_acknowledgements() -> None:
    for cue in agent.CUE_TEXTS:
        path = agent.BACKCHANNEL_CLIP_DIR / f"{cue_slug(cue)}.wav"
        with wave.open(str(path)) as handle:
            seconds = handle.getnframes() / handle.getframerate()
            assert handle.getframerate() == CUE_SAMPLE_RATE
            assert handle.getnchannels() == 1
        assert 0.15 < seconds < 1.5, f"{cue} is {seconds:.2f}s"


def test_cues_already_rendered_for_this_voice_skip_the_provider(tmp_path, monkeypatch) -> None:
    """Every room is its own process, so this is what stops a sweep paying per room.

    Ten concurrent calls cost 1.4-5.0 s per session measured; with the cache primed the
    session makes no call at all for the cues.
    """

    from livekit import rtc

    from blue_machines_baseline.cue_audio import cue_phrases, write_cached_cues

    monkeypatch.setenv("BACKCHANNEL_CUE_CACHE_DIR", str(tmp_path / "cue-cache"))
    settings = agent.Settings.from_env(
        {**_settings_env(), "TTS_PROVIDER": "deepgram_tts", "DEEPGRAM_API_KEY": "d"}
    )
    key = agent.cue_cache_key(settings)
    frames = {
        cue: rtc.AudioFrame(
            data=b"\x01\x02" * 2400, sample_rate=48000, num_channels=1, samples_per_channel=2400
        )
        for cue in agent.CUE_TEXTS
    }
    write_cached_cues(key, frames)
    assert (tmp_path / "cue-cache").exists()

    async def unexpected(*_args, **_kwargs):
        raise AssertionError("a primed cache must not call the provider")

    monkeypatch.setattr(agent, "synthesize_backchannel_clips", unexpected)
    recorder = EventRecorder(tmp_path / "events.jsonl", scenario_id="s", mode="backchannel")

    clips = asyncio.run(agent.load_cue_clips(settings, recorder))

    assert set(clips) >= set(agent.CUE_TEXTS)
    import json

    ready = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if "clips_ready" in line
    ]
    assert ready[-1]["data"]["source"] == "tts"
    assert ready[-1]["data"]["rendered"] == 0
    assert cue_phrases(["mm-hmm"])  # the respelling is part of the cache key
