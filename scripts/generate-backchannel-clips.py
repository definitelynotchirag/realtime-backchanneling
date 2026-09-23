#!/usr/bin/env python3
"""Generate short backchannel WAV assets with the configured worker voice."""

from __future__ import annotations

import asyncio
import audioop
import json
import wave
from array import array
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents.utils import http_context

from blue_machines_baseline.agent import create_tts
from blue_machines_baseline.config import Settings

CUE_SAMPLE_RATE = 48000
"""Cue clips are written at the rate the room plays out, so the audio path sees the
same shape whichever provider rendered them. Deepgram speaks at 24 kHz, Gemini at
48 kHz, and these files were 48 kHz before - a cue that plays back with its own rate
would be the only thing on that path behaving differently from the replies."""


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

    manifest: dict[str, object] = {
        "provider": settings.tts_provider,
        "model": getattr(provider, "model", ""),
        "sample_rate": CUE_SAMPLE_RATE,
        "cues": {},
    }
    try:
        async with http_context.open():
            for filename, phrase in PHRASES.items():
                audio = await provider.synthesize(phrase).collect()
                duration = _write_wav(output_dir / f"{filename}.wav", audio, phrase=phrase)
                manifest["cues"][filename] = {"phrase": phrase, "duration_seconds": duration}
                print(f"Generated {output_dir / f'{filename}.wav'} ({duration:.3f}s)")
    finally:
        await provider.aclose()
    # The cues are pre-generated, so what produced them is not visible anywhere else.
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _write_wav(path: Path, frame: object, *, phrase: str) -> float:
    if not isinstance(frame, rtc.AudioFrame):
        raise TypeError("TTS provider returned an unexpected audio frame")
    samples = array("h", bytes(frame.data))
    samples = _trim_silence(samples, sample_rate=frame.sample_rate)
    if frame.num_channels > 1:
        mono = audioop.tomono(samples.tobytes(), 2, 0.5, 0.5)
        samples = array("h", mono)
    if frame.sample_rate != CUE_SAMPLE_RATE:
        converted, _ = audioop.ratecv(
            samples.tobytes(), 2, 1, frame.sample_rate, CUE_SAMPLE_RATE, None
        )
        samples = array("h", converted)
    peak = max((abs(sample) for sample in samples), default=0)
    if peak:
        gain = min(0.70 * 32767 / peak, 4.0)
        samples = array("h", (max(-32768, min(32767, round(sample * gain))) for sample in samples))

    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(CUE_SAMPLE_RATE)
        output.writeframes(samples.tobytes())
    if not samples or len(samples) / CUE_SAMPLE_RATE > 1.5:
        raise RuntimeError(
            f"{phrase!r} rendered {len(samples) / CUE_SAMPLE_RATE:.2f}s of audio; an "
            "acknowledgement that long holds the floor instead of acknowledging"
        )
    return round(len(samples) / CUE_SAMPLE_RATE, 3)


def _trim_silence(samples: array, *, sample_rate: int, keep_ms: int = 20) -> array:
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
