#!/usr/bin/env python3
"""Write the acknowledgement clips the policy falls back to.

The worker renders this same bank at session start with the configured voice
(`BACKCHANNEL_CLIP_SOURCE=tts`, the default); these files are what plays when that
render fails, and the only source when the source is set to `assets` - the
deterministic choice for a measurement sweep. Run this after changing the cue bank in
`jev.py`, so the fallback covers every cue the policy can choose.

    uv run python scripts/generate-backchannel-clips.py [cue ...]

The trim, level and length rules live in `blue_machines_baseline.cue_audio`, shared
with the runtime path, so a cue has the same shape whichever way it was produced.
"""

from __future__ import annotations

import asyncio
import json
import sys
import wave

from dotenv import load_dotenv
from livekit import rtc

from blue_machines_baseline.agent import BACKCHANNEL_CLIP_DIR, CUE_TEXTS, create_tts
from blue_machines_baseline.config import Settings
from blue_machines_baseline.cue_audio import cue_frame, cue_phrases, cue_slug


async def generate(cue_texts: tuple[str, ...]) -> None:
    load_dotenv()
    settings = Settings.from_env()
    provider = create_tts(settings)
    output_dir = BACKCHANNEL_CLIP_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    phrases = cue_phrases(cue_texts)

    manifest: dict[str, object] = {
        "provider": settings.tts_provider,
        "model": getattr(provider, "model", ""),
        "cues": {},
    }
    try:
        for cue_text in cue_texts:
            phrase = phrases[cue_text]
            frames = [event.frame async for event in provider.synthesize(phrase) if event.frame]
            if not frames:
                raise RuntimeError(f"{phrase!r} returned no audio")
            combined = frames[0] if len(frames) == 1 else rtc.combine_audio_frames(frames)
            cue = cue_frame(combined, phrase=phrase)
            path = output_dir / f"{cue_slug(cue_text)}.wav"
            with wave.open(str(path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(cue.sample_rate)
                output.writeframes(bytes(cue.data))
            seconds = round(cue.samples_per_channel / cue.sample_rate, 3)
            manifest["cues"][cue_text] = {"phrase": phrase, "duration_seconds": seconds}
            print(f"{cue_text:14s} {seconds:5.3f}s  {path}")
    finally:
        await provider.aclose()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    requested = tuple(sys.argv[1:]) or CUE_TEXTS
    unknown = [cue for cue in requested if cue not in CUE_TEXTS]
    if unknown:
        raise SystemExit(f"unknown cue(s): {', '.join(unknown)}; bank is {', '.join(CUE_TEXTS)}")
    asyncio.run(generate(requested))
