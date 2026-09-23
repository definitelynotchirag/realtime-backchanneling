import asyncio
import struct

import pytest
from livekit import rtc
from livekit.agents import stt

from blue_machines_baseline import groq_interim_stt


def frame(seconds: float = 0.2, sample_rate: int = 16000) -> rtc.AudioFrame:
    samples = int(sample_rate * seconds)
    return rtc.AudioFrame(
        data=struct.pack(f"<{samples}h", *([1000] * samples)),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


def test_interims_are_emitted_on_a_cadence_and_the_final_covers_the_whole_turn(monkeypatch) -> None:
    calls: list[int] = []

    async def fake_transcribe(self, audio: bytes, *, language=None) -> str:
        calls.append(len(audio))
        return f"partial-{len(calls)}"

    monkeypatch.setattr(groq_interim_stt.STT, "_transcribe", fake_transcribe)
    stt_client = groq_interim_stt.STT(api_key="key", interim_interval_seconds=0.4)

    async def run() -> list:
        stream = stt_client.stream()
        events = []

        async def read() -> None:
            async for event in stream:
                events.append(event)

        task = asyncio.create_task(read())
        for _ in range(12):  # 2.4 s of audio, spread over more than the cadence
            stream.push_frame(frame())
            await asyncio.sleep(0.05)
        stream.flush()
        await asyncio.sleep(0.05)
        stream.end_input()
        await asyncio.wait_for(task, timeout=5)
        await stream.aclose()
        return events

    events = asyncio.run(run())
    kinds = [event.type for event in events]

    assert stt_client.capabilities.interim_results is True
    assert stt_client.capabilities.streaming is True
    interim = [kind for kind in kinds if kind == stt.SpeechEventType.INTERIM_TRANSCRIPT]
    assert interim, f"expected interim transcripts, got {kinds}"
    assert kinds[-1] == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert kinds[-2] == stt.SpeechEventType.END_OF_SPEECH
    assert len(calls) > 1  # one request per interval, plus the final


def test_a_provider_failure_does_not_break_the_stream(monkeypatch) -> None:
    async def failing(self, audio: bytes, *, language=None) -> str:
        raise groq_interim_stt.GroqInterimSTTError("Groq transcription HTTP 429")

    monkeypatch.setattr(groq_interim_stt.STT, "_transcribe", failing)
    stt_client = groq_interim_stt.STT(api_key="key", interim_interval_seconds=0.4)

    async def run() -> list:
        stream = stt_client.stream()
        events = []

        async def read() -> None:
            async for event in stream:
                events.append(event)

        task = asyncio.create_task(read())
        for _ in range(4):
            stream.push_frame(frame())
            await asyncio.sleep(0.005)
        stream.flush()
        stream.end_input()
        await asyncio.wait_for(task, timeout=5)
        await stream.aclose()
        return events

    events = asyncio.run(run())

    # The turn still closes: a failed transcript must not stall the pipeline.
    assert events[-1].type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert events[-1].alternatives[0].text == ""


def test_wav_encoding_reports_the_segment_duration() -> None:
    audio, duration = groq_interim_stt.wav_bytes_from_frames([frame(0.2), frame(0.2)])

    assert audio.startswith(b"RIFF")
    assert duration == pytest.approx(0.4, abs=0.01)
