"""Acknowledgement audio: the cue bank, and how a cue becomes playable frames.

A backchannel is only a backchannel if it is *fast*. Playing a cue from a cached
frame takes ~20 ms from the policy's decision to audible speech; synthesizing it on
demand costs the provider's time-to-first-audio (measured 0.9-2.0 s for these
phrases on Deepgram Aura-2), by which time the speaker has often moved on or
stopped - and a cue that lands as the user yields is the failure this whole
experiment is built to measure.

So cues are synthesized once per worker process at startup with the configured
voice, trimmed and levelled here, and kept in memory. The committed WAVs stay as
the fallback when the provider is unavailable, which is also why the trim and
normalize helpers live here rather than in the script that writes them.
"""

from __future__ import annotations

import asyncio
import audioop
import hashlib
import logging
import os
import wave
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from livekit import rtc

logger = logging.getLogger("blue-machines-cue-audio")

CUE_SAMPLE_RATE = 48000
"""The rate cues are held at, so a cue has one shape regardless of which provider
rendered it: Deepgram speaks at 24 kHz, Gemini at 48 kHz, and the room plays out at
48 kHz."""

TARGET_PEAK = 0.70
"""Cues are normalized to a fixed peak so they sit at a consistent level against the
agent's replies instead of inheriting each provider's loudness."""

KEEP_SILENCE_MS = 20
"""Margin kept around the trimmed cue. Providers pad both ends, and an untrimmed
"mm-hmm" can run well over a second, at which point it holds the floor instead of
acknowledging."""

MAX_CUE_SECONDS = 1.5
"""A cue longer than this is not an acknowledgement. Treated as a provider fault."""

SILENT_PEAK = int(0.02 * 32767)
"""Below this the cue is silence, not a quiet acknowledgement. Caching it would play
nothing while the event log reported an audible cue."""


class SpeechProvider(Protocol):
    """The part of a LiveKit TTS adapter this module uses."""

    def synthesize(self, text: str) -> Any: ...


def cue_slug(text: str) -> str:
    """The file-safe name for a cue phrase: "I see" -> "i-see"."""

    return "-".join(text.lower().replace(",", "").split())


def trim_silence(samples: array, *, sample_rate: int, keep_ms: int = KEEP_SILENCE_MS) -> array:
    """Drop leading and trailing silence, keeping a short margin."""

    threshold = max(1, int(0.02 * max((abs(sample) for sample in samples), default=0)))
    first = next((index for index, sample in enumerate(samples) if abs(sample) > threshold), None)
    if first is None:
        return samples
    last = next(index for index, sample in enumerate(reversed(samples)) if abs(sample) > threshold)
    margin = int(sample_rate * keep_ms / 1000)
    start = max(0, first - margin)
    stop = min(len(samples), len(samples) - last + margin)
    return samples[start:stop]


def normalize_level(samples: array, *, peak: float = TARGET_PEAK, max_gain: float = 4.0) -> array:
    """Scale the cue to a fixed peak, with the gain capped so noise is not amplified."""

    loudest = max((abs(sample) for sample in samples), default=0)
    if not loudest:
        return samples
    gain = min(peak * 32767 / loudest, max_gain)
    return array("h", (max(-32768, min(32767, round(sample * gain))) for sample in samples))


def to_mono(samples: array, *, channels: int) -> array:
    if channels <= 1:
        return samples
    return array("h", audioop.tomono(samples.tobytes(), 2, 0.5, 0.5))


def to_sample_rate(
    samples: array, *, source_rate: int, target_rate: int = CUE_SAMPLE_RATE
) -> array:
    if source_rate == target_rate or not samples:
        return samples
    converted, _ = audioop.ratecv(samples.tobytes(), 2, 1, source_rate, target_rate, None)
    return array("h", converted)


def cue_frame(frame: rtc.AudioFrame, *, phrase: str) -> rtc.AudioFrame:
    """Turn one provider frame into the single cached frame a cue is played from."""

    samples = to_mono(array("h", bytes(frame.data)), channels=frame.num_channels)
    samples = to_sample_rate(samples, source_rate=frame.sample_rate)
    samples = normalize_level(trim_silence(samples, sample_rate=CUE_SAMPLE_RATE))
    if not samples or max((abs(sample) for sample in samples), default=0) < SILENT_PEAK:
        raise ValueError(f"{phrase!r} rendered no audible audio")
    seconds = len(samples) / CUE_SAMPLE_RATE
    if seconds > MAX_CUE_SECONDS:
        raise ValueError(
            f"{phrase!r} rendered {seconds:.2f}s of audio; an acknowledgement that long "
            "holds the floor instead of acknowledging"
        )
    return rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=CUE_SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=len(samples),
    )


async def synthesize_cue_frames(
    provider: SpeechProvider, phrases: Mapping[str, str]
) -> dict[str, rtc.AudioFrame]:
    """Synthesize every cue phrase concurrently, keyed by the cue text.

    Concurrently because these calls are latency-bound: eleven cues one after another
    would add eleven seconds to session setup, whereas together they cost roughly the
    slowest single call.
    """

    async def one(cue_text: str, phrase: str) -> tuple[str, rtc.AudioFrame]:
        frames = [event.frame async for event in provider.synthesize(phrase) if event.frame]
        if not frames:
            raise ValueError(f"{phrase!r} rendered no audio")
        combined = frames[0] if len(frames) == 1 else rtc.combine_audio_frames(frames)
        return cue_text, cue_frame(combined, phrase=phrase)

    results = await asyncio.gather(*(one(text, phrase) for text, phrase in phrases.items()))
    return dict(results)


def cue_phrases(cue_texts: Sequence[str]) -> dict[str, str]:
    """The text to speak for each cue. Kept separate so a phrase can be respelled
    without renaming its cue (a literal "mm-hmm" gains a trailing hum on some
    providers, so it is spoken as "mm-hm")."""

    respellings = {"mm-hmm": "mm-hm"}
    return {text: respellings.get(text, text) for text in cue_texts}


DEFAULT_CACHE_DIR = Path(".cache/backchannels")


def cue_cache_dir() -> Path:
    return Path(os.environ.get("BACKCHANNEL_CUE_CACHE_DIR", str(DEFAULT_CACHE_DIR))).expanduser()


def _cache_path(cache_key: str, cue_text: str) -> Path:
    phrase = cue_phrases([cue_text])[cue_text]
    digest = hashlib.sha256(f"{phrase}|{cue_text}".encode()).hexdigest()[:8]
    return cue_cache_dir() / f"{cache_key}-{cue_slug(cue_text)}-{digest}.wav"


def read_cached_cues(cache_key: str, cue_texts: Sequence[str]) -> dict[str, rtc.AudioFrame]:
    """Cued audio already rendered for this voice and bank, if any.

    Every room runs in its own process, so without this a sweep would pay the
    provider once per room - ten calls and a couple of seconds each time, for audio
    that cannot have changed because the voice and the phrases are in the key.
    """

    found: dict[str, rtc.AudioFrame] = {}
    for cue_text in cue_texts:
        path = _cache_path(cache_key, cue_text)
        if not path.exists():
            continue
        try:
            with wave.open(str(path)) as source:
                found[cue_text] = rtc.AudioFrame(
                    data=source.readframes(source.getnframes()),
                    sample_rate=source.getframerate(),
                    num_channels=source.getnchannels(),
                    samples_per_channel=source.getnframes(),
                )
        except (wave.Error, OSError) as exc:
            logger.warning("ignoring unreadable cue cache entry %s (%s)", path, exc)
    return found


def write_cached_cues(cache_key: str, cues: Mapping[str, rtc.AudioFrame]) -> None:
    directory = cue_cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    for cue_text, frame in cues.items():
        path = _cache_path(cache_key, cue_text)
        try:
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(frame.sample_rate)
                output.writeframes(bytes(frame.data))
        except OSError as exc:  # a cache that cannot be written is not an error
            logger.warning("could not cache cue %s (%s)", path, exc)
