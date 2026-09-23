"""Safe, deliberately small backchannel timing policy."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

logger = logging.getLogger("blue-machines-backchannel")


class BackchannelHandle(Protocol):
    """The small part of a LiveKit SpeechHandle needed by this policy."""

    def interrupt(self, *, force: bool = False) -> object: ...


class BackchannelEngine:
    """Schedule one short acknowledgement and cancel it on a floor change.

    The engine knows nothing about LiveKit or the LLM. The caller supplies a
    function that plays an acknowledgement and receives a handle that can be
    interrupted. This keeps timing behavior deterministic and easy to test.
    """

    def __init__(
        self,
        play: Callable[[], BackchannelHandle],
        *,
        delay_seconds: float = 1.4,
        cooldown_seconds: float = 4.0,
        enabled: bool = True,
        semantic_required: bool = False,
        max_acknowledgements_per_turn: int = 1,
        eot_threshold: float = 0.8,
        collision_window_seconds: float = 0.5,
        on_event: Callable[..., None] | None = None,
    ) -> None:
        if delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        if not 0 <= eot_threshold <= 1:
            raise ValueError("eot_threshold must be between 0 and 1")
        if max_acknowledgements_per_turn < 1:
            raise ValueError("max_acknowledgements_per_turn must be at least 1")
        if collision_window_seconds <= 0:
            raise ValueError("collision_window_seconds must be positive")
        self._play = play
        self._delay_seconds = delay_seconds
        self._cooldown_seconds = cooldown_seconds
        self._enabled = enabled
        self._semantic_required = semantic_required
        self._max_acknowledgements_per_turn = max_acknowledgements_per_turn
        self._semantic_approved = False
        self._eot_threshold = eot_threshold
        self._on_event = on_event or (lambda _name, **_data: None)
        self._speaking = False
        self._agent_busy = False
        self._acknowledgements_this_turn = 0
        self._generation = 0
        self._last_played_at = float("-inf")
        self._pending_task: asyncio.Task[None] | None = None
        self._active_handle: BackchannelHandle | None = None
        self._starting_backchannel = False
        self._awaiting_audio_start = False
        self._eot_probability: float | None = None
        self._clock = time.monotonic
        self._turn_started_at: float | None = None
        self._collision_window_seconds = collision_window_seconds
        # Monotonic time at which the active acknowledgement became audible. It
        # backs both the collision signal (user yielding right after a cue became
        # audible) and the cue duration reported on completion.
        self._audible_at: float | None = None
        self._collision_reported = False

    def _emit(self, name: str, **data: Any) -> None:
        """Emit one lifecycle event, with optional structured metadata."""

        self._on_event(name, **data)

    def user_started(self) -> None:
        """Start watching a new user turn."""

        self._speaking = True
        self._turn_started_at = self._clock()
        self._acknowledgements_this_turn = 0
        self._semantic_approved = False
        self._eot_probability = None
        self._awaiting_audio_start = False
        self._generation += 1
        self._cancel_pending()
        self._schedule_if_eligible()

    def user_stopped(self) -> None:
        """Yield the floor and immediately stop any acknowledgement."""

        self._speaking = False
        self._turn_started_at = None
        self._semantic_approved = False
        self._generation += 1
        self._cancel_pending()
        self._report_collision_if_imminent()
        self._interrupt_active(reason="user_stopped")

    def _report_collision_if_imminent(self) -> None:
        """Flag an acknowledgement the user talked over into their turn end.

        An acknowledgement that is still audible when the user yields - within
        the collision window of the moment it became audible - is the failure
        mode this experiment is trying to measure, so it is reported once per
        cue rather than being silently cancelled.
        """

        if self._audible_at is None or self._collision_reported:
            return
        ms_since_audible = (self._clock() - self._audible_at) * 1000
        if ms_since_audible > self._collision_window_seconds * 1000:
            return
        self._collision_reported = True
        self._emit(
            "backchannel_collision",
            ms_since_audible=round(ms_since_audible, 3),
            window_ms=int(self._collision_window_seconds * 1000),
        )

    def set_agent_busy(self, busy: bool) -> None:
        """Prevent acknowledgements while the agent owns the floor."""

        self._clear_finished_active()
        # The acknowledgement itself also produces an agent "speaking" event.
        # Do not treat that event as a reason to interrupt the acknowledgement.
        if busy and (self._active_handle is not None or self._starting_backchannel):
            return
        self._agent_busy = busy
        if busy:
            self._generation += 1
            self._cancel_pending()
            self._interrupt_active(reason="agent_busy")
        else:
            self._schedule_if_eligible()

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable acknowledgements for the current room."""

        if self._enabled == enabled:
            return
        self._enabled = enabled
        self._emit("backchannel_enabled" if enabled else "backchannel_disabled")
        if not enabled:
            self._generation += 1
            self._cancel_pending()
            self._interrupt_active(reason="disabled")
        else:
            self._schedule_if_eligible()

    def set_semantic_required(self, required: bool) -> None:
        """Switch between timer-only and externally approved acknowledgements."""

        if self._semantic_required == required:
            return
        self._semantic_required = required
        self._semantic_approved = False
        self._generation += 1
        self._cancel_pending()
        self._interrupt_active(reason="mode_switch")
        self._emit("backchannel_semantic_required" if required else "backchannel_timer_policy")
        self._schedule_if_eligible()

    def set_semantic_approval(self, approved: bool) -> None:
        """Apply the latest semantic decision without weakening the safety gates."""

        if self._semantic_approved == approved:
            return
        self._semantic_approved = approved
        self._emit("backchannel_semantic_approved" if approved else "backchannel_semantic_rejected")
        if approved:
            self._schedule_if_eligible()
        else:
            self._generation += 1
            self._cancel_pending()
            # A withdrawn approval must also take back a cue that is already
            # audible; relying on the caller to stop the floor is implicit coupling.
            self._interrupt_active(reason="semantic_rejected")

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def awaiting_audio_start(self) -> bool:
        return self._awaiting_audio_start

    @property
    def active(self) -> bool:
        """Whether the current agent speech belongs to the acknowledgement."""

        return self._active_handle is not None or self._starting_backchannel

    @property
    def eot_probability(self) -> float | None:
        """Latest public end-of-turn estimate known to the policy."""

        return self._eot_probability

    @property
    def eot_threshold(self) -> float:
        """The end-of-turn probability at or above which a cue is suppressed."""

        return self._eot_threshold

    def update_eot_probability(self, probability: float | None) -> None:
        """Suppress a pending acknowledgement when a turn is likely complete."""

        self._eot_probability = probability
        if probability is not None and probability >= self._eot_threshold:
            self._generation += 1
            self._cancel_pending()
            self._on_event("backchannel_suppressed_eot")

    def set_eot_threshold(self, threshold: float) -> None:
        """Adopt a detector's own calibrated end-of-turn boundary.

        The default 0.8 only makes sense for the binary final-transcript
        fallback. A real detector publishes its own unlikely-turn threshold, and
        gating on that value is what makes the probability stream meaningful.
        """

        if not 0 <= threshold <= 1:
            raise ValueError("eot_threshold must be between 0 and 1")
        self._eot_threshold = threshold

    def update_transcript(self, *, is_final: bool) -> None:
        """Record that STT has produced a cue for the active turn.

        The policy intentionally does not require a transcript before scheduling: a
        provider can be late while VAD still has a reliable speaking signal. The
        callback exists so deployments can add stricter transcript gating without
        changing the engine's lifecycle contract.
        """

        if self._speaking and not is_final:
            self._emit("backchannel_interim_transcript")

    def mark_audio_started(self) -> None:
        """Mark the point at which the acknowledgement reached the audio output."""

        if self._awaiting_audio_start:
            self._awaiting_audio_start = False
            self._audible_at = self._clock()
            self._collision_reported = False
            self._emit("backchannel_audio_started")

    async def aclose(self) -> None:
        """Cancel all policy work during room shutdown."""

        self._speaking = False
        self._generation += 1
        pending_task = self._pending_task
        self._cancel_pending()
        self._interrupt_active(reason="shutdown")
        if pending_task is not None:
            await asyncio.gather(pending_task, return_exceptions=True)

    def _cancel_pending(self) -> None:
        if self._pending_task is not None and not self._pending_task.done():
            self._pending_task.cancel()
        self._pending_task = None

    def _schedule_if_eligible(self) -> None:
        if (
            self._enabled
            and self._speaking
            and not self._agent_busy
            and self._acknowledgements_this_turn < self._max_acknowledgements_per_turn
            and self._pending_task is None
            # A cue remains active until LiveKit reports its handle complete.
            # Do not let a new semantic approval create overlapping playback.
            and self._active_handle is None
            and not self._starting_backchannel
            and (not self._semantic_required or self._semantic_approved)
        ):
            elapsed = (
                max(0.0, self._clock() - self._turn_started_at)
                if self._turn_started_at is not None
                else 0.0
            )
            wait_seconds = max(0.0, self._delay_seconds - elapsed)
            logger.info("backchannel timer scheduled for %.2fs", wait_seconds)
            self._pending_task = asyncio.create_task(
                self._wait_and_play(self._generation, wait_seconds), name="backchannel-wait"
            )

    def _clear_finished_active(self) -> None:
        if self._active_handle is None:
            return
        done = getattr(self._active_handle, "done", None)
        if callable(done) and done():
            self._active_handle = None

    def _interrupt_active(self, *, reason: str) -> None:
        if self._active_handle is None:
            return
        try:
            self._active_handle.interrupt(force=True)
            self._emit("backchannel_cancelled", reason=reason)
        except Exception:
            logger.exception("Failed to interrupt backchannel speech")
        finally:
            self._awaiting_audio_start = False
            self._active_handle = None
            self._audible_at = None
            self._collision_reported = False

    async def _wait_and_play(self, generation: int, wait_seconds: float) -> None:
        try:
            await asyncio.sleep(wait_seconds)
            logger.info("backchannel timer fired")
            now = self._clock()
            if (
                generation != self._generation
                or not self._enabled
                or not self._speaking
                or self._agent_busy
                or self._acknowledgements_this_turn >= self._max_acknowledgements_per_turn
                or (self._semantic_required and not self._semantic_approved)
                or (
                    self._eot_probability is not None
                    and self._eot_probability >= self._eot_threshold
                )
                or now - self._last_played_at < self._cooldown_seconds
            ):
                logger.info("backchannel timer skipped: state changed before playback")
                return

            logger.info("backchannel invoking playback")
            self._emit("backchannel_decision")
            self._starting_backchannel = True
            try:
                handle = self._play()
            finally:
                self._starting_backchannel = False
            logger.info("backchannel playback handle created")
            self._active_handle = handle
            add_done_callback = getattr(handle, "add_done_callback", None)
            if callable(add_done_callback):
                add_done_callback(self._on_handle_done)
            self._acknowledgements_this_turn += 1
            self._last_played_at = now
            self._awaiting_audio_start = True
            self._emit("backchannel_started")

            if (
                generation != self._generation
                or not self._enabled
                or not self._speaking
                or self._agent_busy
            ):
                self._interrupt_active(reason="post_play_race")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to start backchannel speech")
        finally:
            if self._pending_task is asyncio.current_task():
                self._pending_task = None

    def _on_handle_done(self, handle: BackchannelHandle) -> None:
        if self._active_handle is not handle:
            return
        audible_at = self._audible_at
        self._active_handle = None
        self._awaiting_audio_start = False
        self._audible_at = None
        self._collision_reported = False
        # The handle completing means playout finished, so the audible duration is
        # the only honest end-to-end length of the cue.
        duration_ms = None if audible_at is None else round((self._clock() - audible_at) * 1000, 3)
        self._emit("backchannel_completed", duration_ms=duration_ms)
