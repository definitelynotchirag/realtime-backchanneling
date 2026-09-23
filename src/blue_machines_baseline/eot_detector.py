"""Real end-of-turn probability for the backchannel policy.

The policy needs to know how likely it is that the user is about to finish.
LiveKit's ``AgentSession`` does not surface its own prediction as a public
event, but the turn detector that backs it is public, so this module taps the
user's microphone audio and runs that detector directly.

It is deliberately fail-safe: if the detector cannot start or a prediction
fails, the failure is reported once and the policy keeps working on its own
final-transcript fallback.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

from livekit import rtc

logger = logging.getLogger("blue-machines-eot-detector")

SAMPLE_RATE: int = 16000
"""Sample rate the detector stream is fed with."""

SILENCE_WINDOW_SECONDS: float = 0.2
"""Mirrors the SDK's own rule: a prediction needs this much recent silence."""

PREDICTION_TIMEOUT_SECONDS: float = 1.5
"""A prediction that does not answer in time is abandoned, not awaited."""

MIN_PREDICTION_INTERVAL_SECONDS: float = 0.4
"""Lower bound on how often the detector is asked for a prediction."""

SILENCE_RMS_THRESHOLD: float = 120.0
"""int16 frame RMS below which the frame counts as silence."""


class TurnDetectionEvent(Protocol):
    """The part of the SDK's event this module reads."""

    @property
    def end_of_turn_probability(self) -> float: ...


class TurnDetectorStream(Protocol):
    """The public stream surface used here."""

    def push_audio(self, frame: rtc.AudioFrame) -> None: ...

    def predict(self) -> asyncio.Future[TurnDetectionEvent]: ...

    def cancel_inference(self) -> None: ...

    def flush(self, reason: str | None = None) -> None: ...

    async def aclose(self) -> None: ...


class TurnDetectorLike(Protocol):
    """The public detector surface used here."""

    async def unlikely_threshold(self, language: Any) -> float | None: ...

    def stream(self) -> TurnDetectorStream: ...


def frame_rms(frame: rtc.AudioFrame) -> float:
    """Return the RMS amplitude of a 16-bit frame.

    ``AudioFrame.data`` is already an int16 view in this SDK version, but a
    bytes-like buffer is accepted too so the helper stays usable in tests.
    """

    data = frame.data
    if data is None:
        return 0.0
    if isinstance(data, (bytes, bytearray)):
        samples = memoryview(data).cast("h")
    elif getattr(data, "format", "") == "h":
        samples = data
    else:
        samples = memoryview(data).cast("h")
    if not samples:
        return 0.0
    total = 0
    for sample in samples:
        total += sample * sample
    return (total / len(samples)) ** 0.5


class EotDetector:
    """Run the public turn detector over the user's audio and report predictions.

    ``on_prediction`` receives ``(probability, threshold)`` for every answered
    prediction and ``on_unavailable`` receives a short reason exactly once, the
    first time the detector cannot be used at all.
    """

    def __init__(
        self,
        *,
        on_prediction: Callable[[float, float | None], None],
        on_unavailable: Callable[[str], None] | None = None,
        detector: TurnDetectorLike | None = None,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._on_prediction = on_prediction
        self._on_unavailable = on_unavailable or (lambda _reason: None)
        self._detector = detector
        self._sample_rate = sample_rate
        self._stream: TurnDetectorStream | None = None
        self._threshold: float | None = None
        self._prediction_task: asyncio.Task[None] | None = None
        self._closed = False
        self._unavailable_reported = False
        self._turn_open = False
        self._silence_seconds = 0.0
        self._last_prediction_at = float("-inf")
        self._clock = time.monotonic

    @property
    def threshold(self) -> float | None:
        """The detector's own unlikely-turn boundary, once it is known."""

        return self._threshold

    @property
    def model(self) -> str | None:
        """Which turn-detector model is answering, so the event can name it."""

        stream = self._stream
        return str(getattr(stream, "model", "")) or None if stream is not None else None

    @property
    def available(self) -> bool:
        return self._stream is not None and not self._closed

    async def start(self) -> bool:
        """Open the detector stream. Returns False when the detector is unusable."""

        if self._closed:
            return False
        try:
            detector = self._detector
            if detector is None:
                from livekit.agents import inference

                detector = inference.TurnDetector()
            self._detector = detector
            self._threshold = await detector.unlikely_threshold(None)
            self._stream = detector.stream()
        except Exception as exc:  # noqa: BLE001 - the detector is strictly optional
            self._report_unavailable(f"{type(exc).__name__}: {exc}")
            return False
        logger.info("eot detector ready (threshold=%s)", self._threshold)
        return True

    def begin_turn(self) -> None:
        """Start watching a new user turn."""

        self._turn_open = True
        self._silence_seconds = 0.0
        self._cancel_prediction()
        self._last_prediction_at = float("-inf")

    def end_turn(self) -> None:
        """Stop watching the turn and drop any prediction still in flight."""

        self._turn_open = False
        self._silence_seconds = 0.0
        self._cancel_prediction()
        stream = self._stream
        if stream is not None:
            try:
                stream.flush(reason="turn_committed")
            except Exception:  # noqa: BLE001 - flushing is best effort
                logger.debug("eot detector flush failed", exc_info=True)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        """Feed one microphone frame and ask for a prediction when warranted."""

        stream = self._stream
        if stream is None or self._closed:
            return
        try:
            stream.push_audio(frame)
        except Exception as exc:  # noqa: BLE001 - never break the audio path
            self._report_unavailable(f"{type(exc).__name__}: {exc}")
            return

        if not self._turn_open:
            return
        frame_seconds = frame.samples_per_channel / max(1, frame.sample_rate)
        if frame_rms(frame) <= SILENCE_RMS_THRESHOLD:
            self._silence_seconds += frame_seconds
        else:
            self._silence_seconds = 0.0
            self._cancel_prediction()
            return
        if self._silence_seconds < SILENCE_WINDOW_SECONDS:
            return
        if self._prediction_task is not None and not self._prediction_task.done():
            return
        now = self._clock()
        if now - self._last_prediction_at < MIN_PREDICTION_INTERVAL_SECONDS:
            return
        self._last_prediction_at = now
        self._prediction_task = asyncio.create_task(self._run_prediction())

    async def aclose(self) -> None:
        """Release the detector stream during shutdown."""

        self._closed = True
        self._turn_open = False
        self._cancel_prediction()
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                await stream.aclose()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                logger.debug("eot detector close failed", exc_info=True)

    async def _run_prediction(self) -> None:
        stream = self._stream
        if stream is None:
            return
        try:
            event = await asyncio.wait_for(stream.predict(), timeout=PREDICTION_TIMEOUT_SECONDS)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a failed prediction is not fatal
            logger.debug("eot prediction failed: %s", exc)
            return
        probability = float(getattr(event, "end_of_turn_probability", 0.0))
        try:
            self._on_prediction(probability, self._threshold)
        except Exception:  # noqa: BLE001 - a consumer error must not kill the tap
            logger.exception("eot prediction consumer failed")

    def _cancel_prediction(self) -> None:
        task = self._prediction_task
        if task is not None and not task.done():
            task.cancel()
        self._prediction_task = None
        stream = self._stream
        if stream is not None and self._turn_open:
            try:
                stream.cancel_inference()
            except Exception:  # noqa: BLE001 - superseding is best effort
                logger.debug("eot cancel_inference failed", exc_info=True)

    def _report_unavailable(self, reason: str) -> None:
        if self._unavailable_reported:
            return
        self._unavailable_reported = True
        logger.warning("eot detector unavailable: %s", reason)
        try:
            self._on_unavailable(reason)
        except Exception:  # noqa: BLE001 - reporting must not raise
            logger.exception("eot unavailable consumer failed")
