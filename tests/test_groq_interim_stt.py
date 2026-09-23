import asyncio
import struct

import pytest
from livekit import rtc
from livekit.agents import stt, vad

from blue_machines_baseline import groq_interim_stt


def frame(seconds: float = 0.2, sample_rate: int = 16000) -> rtc.AudioFrame:
    samples = int(sample_rate * seconds)
    return rtc.AudioFrame(
        data=struct.pack(f"<{samples}h", *([1000] * samples)),
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=samples,
    )


class FakeVadStream:
    """Emits one turn: speech starts on the first frame, ends after ``turn_frames``."""

    def __init__(self, *, turn_frames: int = 5) -> None:
        self._turn_frames = turn_frames
        self._queue: asyncio.Queue[vad.VADEvent] = asyncio.Queue()
        self._frames: list[rtc.AudioFrame] = []
        self._speaking = False
        self.flushes = 0
        self.closed = False

    def push_frame(self, item: rtc.AudioFrame) -> None:
        if not self._speaking:
            self._speaking = True
            self._queue.put_nowait(_event(vad.VADEventType.START_OF_SPEECH))
        self._frames.append(item)
        if len(self._frames) >= self._turn_frames:
            self._end_turn()

    def flush(self) -> None:
        self.flushes += 1

    def end_input(self) -> None:
        if self._speaking:
            self._end_turn()
        self._queue.put_nowait(None)  # type: ignore[arg-type]

    async def aclose(self) -> None:
        self.closed = True

    def _end_turn(self) -> None:
        self._speaking = False
        self._queue.put_nowait(_event(vad.VADEventType.END_OF_SPEECH, frames=tuple(self._frames)))
        self._frames = []

    def __aiter__(self):
        return self

    async def __anext__(self) -> vad.VADEvent:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item


def _event(event_type: vad.VADEventType, *, frames: tuple = ()) -> vad.VADEvent:
    event = vad.VADEvent.__new__(vad.VADEvent)
    object.__setattr__(event, "type", event_type)
    object.__setattr__(event, "frames", frames)
    object.__setattr__(event, "silence_duration", 0.5)
    object.__setattr__(event, "inference_duration", 0.01)
    return event


class FakeVad:
    def __init__(self, *, turn_frames: int = 5) -> None:
        self.stream_instance = FakeVadStream(turn_frames=turn_frames)

    def stream(self, **_kwargs) -> FakeVadStream:
        return self.stream_instance


def collect_events(stt_client, frames_to_push, *, pause: float, flush: bool = False):
    async def run():
        stream = stt_client.stream()
        events = []

        async def read() -> None:
            async for event in stream:
                events.append(event)

        task = asyncio.create_task(read())
        for item in frames_to_push:
            stream.push_frame(item)
            await asyncio.sleep(pause)
        if flush:
            stream.flush()
        stream.end_input()
        await asyncio.wait_for(task, timeout=10)
        await stream.aclose()
        return events

    return asyncio.run(run())


def test_capabilities_advertise_streaming_interims() -> None:
    stt_client = groq_interim_stt.STT(api_key="key", vad=FakeVad())

    assert stt_client.capabilities.streaming is True
    assert stt_client.capabilities.interim_results is True
    assert stt_client.provider == "groq"


def test_a_turn_ends_with_a_final_transcript(monkeypatch) -> None:
    calls: list[int] = []

    async def fake_transcribe(self, audio: bytes, *, language=None) -> str:
        calls.append(len(audio))
        return f"text-{len(calls)}"

    monkeypatch.setattr(groq_interim_stt.STT, "_transcribe", fake_transcribe)
    # The turn must outlast the interim cadence (0.4 s) for a snapshot to fire.
    stt_client = groq_interim_stt.STT(
        api_key="key", interim_interval_seconds=0.4, vad=FakeVad(turn_frames=12)
    )

    events = collect_events(stt_client, [frame() for _ in range(12)], pause=0.05)
    kinds = [event.type for event in events]

    assert stt.SpeechEventType.START_OF_SPEECH in kinds
    assert stt.SpeechEventType.END_OF_SPEECH in kinds
    assert stt.SpeechEventType.INTERIM_TRANSCRIPT in kinds
    # The turn closes with a final transcript: this is what the pipeline needs to
    # commit the user's turn, and what a flush-only implementation never produced.
    assert kinds[-1] == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert events[-1].alternatives[0].text == f"text-{len(calls)}"


def test_each_vad_turn_produces_its_own_final(monkeypatch) -> None:
    texts = iter(["first", "second"])

    async def fake_transcribe(self, audio: bytes, *, language=None) -> str:
        return next(texts, "extra")

    monkeypatch.setattr(groq_interim_stt.STT, "_transcribe", fake_transcribe)
    stt_client = groq_interim_stt.STT(
        api_key="key", interim_interval_seconds=5.0, vad=FakeVad(turn_frames=4)
    )

    events = collect_events(stt_client, [frame() for _ in range(8)], pause=0.01)
    finals = [
        event.alternatives[0].text
        for event in events
        if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    ]

    assert finals == ["first", "second"]


def test_a_failed_transcription_still_closes_the_turn(monkeypatch) -> None:
    async def failing(self, audio: bytes, *, language=None) -> str:
        raise groq_interim_stt.GroqInterimSTTError("Groq transcription HTTP 429")

    monkeypatch.setattr(groq_interim_stt.STT, "_transcribe", failing)
    stt_client = groq_interim_stt.STT(
        api_key="key", interim_interval_seconds=0.4, vad=FakeVad(turn_frames=4)
    )

    events = collect_events(stt_client, [frame() for _ in range(4)], pause=0.05)

    assert events[-1].type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert events[-1].alternatives[0].text == ""


def test_wav_encoding_reports_the_segment_duration() -> None:
    audio, duration = groq_interim_stt.wav_bytes_from_frames([frame(0.2), frame(0.2)])

    assert audio.startswith(b"RIFF")
    assert duration == pytest.approx(0.4, abs=0.01)


def test_request_budget_admits_requests_up_to_the_limit_then_refuses() -> None:
    now = [0.0]
    budget = groq_interim_stt._RequestBudget(3, clock=lambda: now[0])

    assert [budget.allow() for _ in range(3)] == [True, True, True]
    assert budget.allow() is False  # the window is full

    now[0] = 61.0  # the window slides
    assert budget.allow() is True


def test_interims_are_skipped_once_the_budget_is_spent(monkeypatch) -> None:
    calls: list[int] = []

    async def fake_transcribe(self, audio: bytes, *, language=None) -> str:
        calls.append(len(audio))
        return "partial"

    monkeypatch.setattr(groq_interim_stt.STT, "_transcribe", fake_transcribe)
    # Spend the only slot first: the interim must be refused, and the final must
    # still be transcribed because finals are never dropped.
    budget = groq_interim_stt._RequestBudget(1, clock=lambda: 0.0)
    assert budget.allow() is True
    stt_client = groq_interim_stt.STT(
        api_key="key",
        interim_interval_seconds=0.4,
        request_budget=budget,
        vad=FakeVad(turn_frames=12),
    )

    events = collect_events(stt_client, [frame() for _ in range(12)], pause=0.05)
    kinds = [event.type for event in events]

    assert stt.SpeechEventType.INTERIM_TRANSCRIPT not in kinds
    assert kinds[-1] == stt.SpeechEventType.FINAL_TRANSCRIPT  # finals are never dropped
    assert len(calls) == 1
