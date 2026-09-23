"""The driver's live user-audio path: synthesize the user's line as the run starts.

The committed clips are rendered once and replayed, which is what makes a sweep
byte-comparable. This path exists so the same runs can be driven with the speech
provider that is actually configured - the user's side then comes from the stack
under test instead of from whatever voice a clip was rendered with months ago.
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass, field

import pytest
from livekit import rtc

from blue_machines_baseline.benchmark import SCENARIO_BY_ID
from blue_machines_baseline.simulator import (
    FRAME_MS,
    PUBLISH_RATE,
    PublishRateFrames,
    parse_args,
    play_live_utterance,
    silence_frames,
    split_for_pauses,
    utterance_for,
)

SPEECH = b"\x01\x02" * 320  # 20 ms of 16 kHz mono, distinguishable from silence


@dataclass
class Syllable:
    """What a TTS adapter yields: an event that may carry a frame."""

    frame: rtc.AudioFrame | None


def _spoken(frames: int) -> list[Syllable]:
    return [
        Syllable(
            rtc.AudioFrame(data=SPEECH, sample_rate=16000, num_channels=1, samples_per_channel=320)
        )
        for _ in range(frames)
    ]


@dataclass
class RecordingSource:
    """Stands in for rtc.AudioSource, recording what the driver captured."""

    captured: list[bytes] = field(default_factory=list)

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self.captured.append(bytes(frame.data))


@dataclass
class FakeProvider:
    """Yields frames per segment, and can be told to wait before its last segment."""

    frames_per_segment: int
    released: asyncio.Event | None = None
    wanted: list[str] = field(default_factory=list)

    def synthesize(self, text: str):  # noqa: ANN201 - mirrors the adapter's shape
        self.wanted.append(text)
        return self._stream(text)

    async def _stream(self, text: str):
        for event in _spoken(self.frames_per_segment):
            yield event


def test_a_scenario_without_pauses_is_one_segment() -> None:
    assert split_for_pauses("Doing great.", 0) == ["Doing great."]


def test_the_pause_splits_the_sentence_at_a_word_boundary() -> None:
    scenario = SCENARIO_BY_ID["middle_pause"]

    segments = split_for_pauses(utterance_for(scenario), len(scenario.pause_durations_seconds))

    assert len(segments) == len(scenario.pause_durations_seconds) + 1
    assert " ".join(segments) == utterance_for(scenario)
    assert all(segment.strip() for segment in segments)
    # No word is cut in half, which frame-count splitting could not promise.
    words = set(utterance_for(scenario).split())
    assert all(word in words for segment in segments for word in segment.split())


def test_more_pauses_than_words_drops_the_extra_pauses() -> None:
    assert split_for_pauses("Yes, exactly.", 5) == ["Yes,", "exactly."]


def test_the_utterance_is_captured_with_pauses_between_the_segments() -> None:
    async def scenario() -> None:
        source = RecordingSource()
        provider = FakeProvider(frames_per_segment=3)
        middle_pause = SCENARIO_BY_ID["middle_pause"]

        captured_seconds = await play_live_utterance(source, provider, middle_pause)

        pause_frames = len(silence_frames(middle_pause.pause_durations_seconds[0]))
        assert pause_frames > 0  # otherwise this test cannot see the pause
        assert len(provider.wanted) == 2  # the scenario has one pause
        assert len(source.captured) == 6 + pause_frames
        # Speech, the pause, then speech again - in that order.
        assert source.captured[:3] == [SPEECH] * 3
        assert set(source.captured[3 : 3 + pause_frames]) == {b"\x00\x00" * 320}
        assert source.captured[3 + pause_frames :] == [SPEECH] * 3
        assert captured_seconds == pytest.approx(6 * 0.02 + pause_frames * 0.02, abs=0.001)

    asyncio.run(scenario())


def test_capture_is_paced_at_the_frame_interval() -> None:
    """Real-time pacing is what makes the room see a person talking, not a dump."""

    async def scenario() -> None:
        source = RecordingSource()
        provider = FakeProvider(frames_per_segment=4)
        started = asyncio.get_running_loop().time()

        await play_live_utterance(source, provider, SCENARIO_BY_ID["short_answer"])

        elapsed = asyncio.get_running_loop().time() - started
        assert len(source.captured) == 4
        assert elapsed >= 4 * FRAME_MS / 1000 * 0.8

    asyncio.run(scenario())


def test_the_first_segment_is_captured_before_the_last_one_is_synthesized() -> None:
    """The whole point of this path: no waiting for a full utterance buffer."""

    async def scenario() -> None:
        first_captured = asyncio.Event()
        release = asyncio.Event()

        class StreamingProvider:
            async def _stream(self, text: str):
                yield Syllable(
                    rtc.AudioFrame(
                        data=SPEECH, sample_rate=16000, num_channels=1, samples_per_channel=320
                    )
                )
                if text.startswith("we noticed"):
                    # The second segment may only be produced once the first has
                    # already reached the room.
                    await release.wait()
                yield Syllable(
                    rtc.AudioFrame(
                        data=SPEECH, sample_rate=16000, num_channels=1, samples_per_channel=320
                    )
                )

            def synthesize(self, text: str):
                return self._stream(text)

        class ObservingSource(RecordingSource):
            async def capture_frame(self, frame: rtc.AudioFrame) -> None:
                await super().capture_frame(frame)
                first_captured.set()

        middle_pause = SCENARIO_BY_ID["middle_pause"]
        task = asyncio.create_task(
            play_live_utterance(ObservingSource(), StreamingProvider(), middle_pause)
        )
        await asyncio.wait_for(first_captured.wait(), timeout=2.0)
        assert not task.done(), "the driver waited for the whole utterance before speaking"
        release.set()
        await task

    asyncio.run(scenario())


def test_a_provider_that_returns_no_audio_is_an_error() -> None:
    async def scenario() -> None:
        class SilentProvider:
            async def _stream(self, text: str):
                if False:  # pragma: no cover - makes this an async generator
                    yield None

            def synthesize(self, text: str):
                return self._stream(text)

        with pytest.raises(RuntimeError, match="no audio"):
            await play_live_utterance(
                RecordingSource(), SilentProvider(), SCENARIO_BY_ID["short_answer"]
            )

    asyncio.run(scenario())


def test_the_driver_defaults_to_clips_and_opts_into_live_audio() -> None:
    assert parse_args([]).live_audio is False
    assert parse_args(["--live-audio"]).live_audio is True


def _tone(samples: int, rate: int) -> rtc.AudioFrame:
    """A constant non-zero signal, so resampling cannot invent silence."""

    return rtc.AudioFrame(
        data=struct.pack(f"<{samples}h", *([1000] * samples)),
        sample_rate=rate,
        num_channels=1,
        samples_per_channel=samples,
    )


def test_provider_audio_is_converted_to_the_rate_the_room_publishes() -> None:
    """Deepgram speaks at 24 kHz; an AudioSource is created at one rate.

    Handing a 24 kHz frame to a 16 kHz source fails with
    "sample_rate and num_channels don't match", which is how this path first broke.
    """

    converter = PublishRateFrames()
    frames = converter.push(_tone(2400, 24000))  # 100 ms at 24 kHz

    assert frames, "no frames produced"
    assert {frame.sample_rate for frame in frames} == {PUBLISH_RATE}
    assert {frame.num_channels for frame in frames} == {1}
    assert all(frame.samples_per_channel == PUBLISH_RATE * FRAME_MS // 1000 for frame in frames)
    # 100 ms in, 100 ms out: five whole 20 ms frames, nothing dropped.
    assert len(frames) == 5


def test_a_partial_frame_is_carried_rather_than_dropped() -> None:
    """Short frames must not lose their tail, or the last word gets clipped."""

    converter = PublishRateFrames()
    produced: list[rtc.AudioFrame] = []
    for _ in range(3):
        produced.extend(converter.push(_tone(240, 24000)))  # 10 ms each: half a frame
    produced.extend(converter.flush())

    audible = sum(
        1 for frame in produced for sample in memoryview(bytes(frame.data)).cast("h") if sample != 0
    )
    step = PUBLISH_RATE * FRAME_MS // 1000
    assert audible == 3 * 240 * PUBLISH_RATE // 24000  # 30 ms of speech in, 30 ms out
    assert all(frame.sample_rate == PUBLISH_RATE for frame in produced)
    # Whole frames only: a partial tail is padded with silence, never handed over.
    assert all(frame.samples_per_channel == step for frame in produced)
