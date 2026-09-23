import asyncio
import struct
from types import SimpleNamespace

from livekit import rtc

from blue_machines_baseline.eot_detector import SILENCE_WINDOW_SECONDS, EotDetector


class FakeStream:
    def __init__(self, *, probability: float = 0.9, fail_prediction: bool = False) -> None:
        self.frames: list[rtc.AudioFrame] = []
        self.predictions = 0
        self.cancelled = 0
        self.flushes: list[str | None] = []
        self.closed = False
        self._probability = probability
        self._fail_prediction = fail_prediction

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        self.frames.append(frame)

    def predict(self) -> asyncio.Future:
        self.predictions += 1
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        if self._fail_prediction:
            future.set_exception(RuntimeError("detector exploded"))
        else:
            future.set_result(
                SimpleNamespace(end_of_turn_probability=self._probability, type="eot_prediction")
            )
        return future

    def cancel_inference(self) -> None:
        self.cancelled += 1

    def flush(self, reason: str | None = None) -> None:
        self.flushes.append(reason)

    async def aclose(self) -> None:
        self.closed = True


class FakeDetector:
    def __init__(self, stream: FakeStream, *, threshold: float | None = 0.36) -> None:
        self.stream_instance = stream
        self._threshold = threshold
        self.streams = 0

    async def unlikely_threshold(self, language) -> float | None:
        return self._threshold

    def stream(self) -> FakeStream:
        self.streams += 1
        return self.stream_instance


def frame(rms: int, seconds: float = 0.02, sample_rate: int = 16000) -> rtc.AudioFrame:
    samples = max(1, int(sample_rate * seconds))
    amplitude = max(0, min(32767, rms))
    data = struct.pack(f"<{samples}h", *([amplitude] * samples))
    return rtc.AudioFrame(
        data=data, sample_rate=sample_rate, num_channels=1, samples_per_channel=samples
    )


def test_start_resolves_the_detector_threshold_and_opens_one_stream() -> None:
    async def scenario() -> None:
        stream = FakeStream()
        detector = FakeDetector(stream)
        eot = EotDetector(on_prediction=lambda *_args: None, detector=detector)

        assert await eot.start() is True
        assert eot.available is True
        assert eot.threshold == 0.36
        assert detector.streams == 1
        await eot.aclose()

    asyncio.run(scenario())


def test_prediction_is_requested_only_after_the_silence_window_and_is_reported() -> None:
    async def scenario() -> None:
        stream = FakeStream(probability=0.82)
        seen: list[tuple[float, float | None]] = []
        eot = EotDetector(
            on_prediction=lambda probability, threshold: seen.append((probability, threshold)),
            detector=FakeDetector(stream),
        )
        await eot.start()
        eot.begin_turn()

        for _ in range(3):  # loud audio: the user is still talking
            eot.push_audio(frame(rms=4000))
        await asyncio.sleep(0)
        assert stream.predictions == 0

        silence_frames = int(SILENCE_WINDOW_SECONDS / 0.02) + 1
        for _ in range(silence_frames):
            eot.push_audio(frame(rms=0))
        await asyncio.sleep(0.05)

        assert stream.predictions == 1
        assert seen == [(0.82, 0.36)]
        assert len(stream.frames) == 3 + silence_frames
        await eot.aclose()

    asyncio.run(scenario())


def test_speech_resuming_cancels_an_in_flight_prediction() -> None:
    async def scenario() -> None:
        stream = FakeStream()
        eot = EotDetector(on_prediction=lambda *_args: None, detector=FakeDetector(stream))
        await eot.start()
        eot.begin_turn()

        for _ in range(int(SILENCE_WINDOW_SECONDS / 0.02) + 1):
            eot.push_audio(frame(rms=0))
        eot.push_audio(frame(rms=5000))
        await asyncio.sleep(0)

        assert stream.cancelled >= 1
        await eot.aclose()

    asyncio.run(scenario())


def test_a_failing_prediction_does_not_raise_or_report_anything() -> None:
    async def scenario() -> None:
        stream = FakeStream(fail_prediction=True)
        seen: list[tuple[float, float | None]] = []
        unavailable: list[str] = []
        eot = EotDetector(
            on_prediction=lambda probability, threshold: seen.append((probability, threshold)),
            on_unavailable=unavailable.append,
            detector=FakeDetector(stream),
        )
        await eot.start()
        eot.begin_turn()
        for _ in range(int(SILENCE_WINDOW_SECONDS / 0.02) + 1):
            eot.push_audio(frame(rms=0))
        await asyncio.sleep(0.05)

        assert seen == []
        assert unavailable == []
        await eot.aclose()

    asyncio.run(scenario())


def test_an_unstartable_detector_reports_unavailable_once_and_stays_inert() -> None:
    async def scenario() -> None:
        class BrokenDetector:
            async def unlikely_threshold(self, language):
                raise RuntimeError("gateway unreachable")

            def stream(self):  # pragma: no cover - never reached
                raise AssertionError("stream must not be opened")

        unavailable: list[str] = []
        eot = EotDetector(
            on_prediction=lambda *_args: None,
            on_unavailable=unavailable.append,
            detector=BrokenDetector(),
        )

        assert await eot.start() is False
        assert eot.available is False
        assert len(unavailable) == 1
        assert "gateway unreachable" in unavailable[0]

        # Pushing audio without a stream must not raise, and forcing a second
        # failure must not report twice.
        eot.begin_turn()
        eot.push_audio(frame(rms=0))
        eot.push_audio(frame(rms=5000))
        assert len(unavailable) == 1
        await eot.aclose()

    asyncio.run(scenario())


def test_turn_end_stops_requesting_predictions_and_flushes_the_stream() -> None:
    async def scenario() -> None:
        stream = FakeStream()
        eot = EotDetector(on_prediction=lambda *_args: None, detector=FakeDetector(stream))
        await eot.start()
        eot.begin_turn()
        eot.end_turn()

        for _ in range(int(SILENCE_WINDOW_SECONDS / 0.02) + 2):
            eot.push_audio(frame(rms=0))
        await asyncio.sleep(0.02)

        assert stream.predictions == 0
        assert stream.flushes == ["turn_committed"]
        await eot.aclose()

    asyncio.run(scenario())
