#!/usr/bin/env python3
"""Generate short backchannel WAV assets with the configured worker voice."""

from __future__ import annotations

import asyncio
import wave
from array import array
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents.utils import http_context

from blue_machines_baseline.agent import create_tts
from blue_machines_baseline.config import Settings

PHRASES = {
    # The provider tends to pronounce a literal "mm-hmm" with an extra
    # trailing hum. This phonetic spelling keeps the audible cue short.
    "mm-hmm": "mm-hm",
    "uh-huh": "uh-huh",
    "i-see": "I see",
}


async def generate() -> None:
    load_dotenv()
    settings = Settings.from_env()
    provider = create_tts(settings)
    output_dir = Path("assets/backchannels")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        async with http_context.open():
            for filename, phrase in PHRASES.items():
                audio = await provider.synthesize(phrase).collect()
                _write_wav(output_dir / f"{filename}.wav", audio)
                print(f"Generated {output_dir / f'{filename}.wav'}")
    finally:
        await provider.aclose()


def _write_wav(path: Path, frame: object) -> None:
    if not isinstance(frame, rtc.AudioFrame):
        raise TypeError("TTS provider returned an unexpected audio frame")
    samples = array("h", bytes(frame.data))
    samples = _trim_silence(samples)
    peak = max((abs(sample) for sample in samples), default=0)
    if peak:
        gain = min(0.70 * 32767 / peak, 4.0)
        samples = array("h", (max(-32768, min(32767, round(sample * gain))) for sample in samples))

    with wave.open(str(path), "wb") as output:
        output.setnchannels(frame.num_channels)
        output.setsampwidth(2)
        output.setframerate(frame.sample_rate)
        output.writeframes(samples.tobytes())


def _trim_silence(samples: array, *, keep_ms: int = 20, sample_rate: int = 48000) -> array:
    """Drop leading and trailing silence, keeping a short margin.

    A backchannel is a short cue: providers that pad the waveform produce a
    "mm-hmm" of well over a second, which no longer behaves like an
    acknowledgement and can hold the floor far too long.
    """

    threshold = max(1, int(0.02 * max((abs(sample) for sample in samples), default=0)))
    first = next((index for index, sample in enumerate(samples) if abs(sample) > threshold), None)
    if first is None:
        return samples
    last = next(index for index, sample in enumerate(reversed(samples)) if abs(sample) > threshold)
    margin = int(sample_rate * keep_ms / 1000)
    start = max(0, first - margin)
    stop = min(len(samples), len(samples) - last + margin)
    return samples[start:stop]


if __name__ == "__main__":
    asyncio.run(generate())
