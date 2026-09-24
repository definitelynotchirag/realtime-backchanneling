"""Asynchronous Jev gate for semantic backchannel decisions."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Protocol

from typesafe_sdk import Choice, Noul

from .backchannel import BackchannelEngine
from .cue_bank import CUE_BANK_BY_STYLE, CUE_TEXT_BY_STYLE


class JevClient(Protocol):
    async def system_one(self, state: object, questions: object, **kwargs: object) -> object: ...


@dataclass(frozen=True)
class JevDecision:
    """Small, auditable decision returned by the Jev classifier."""

    approved: bool
    turn_stage: str
    continuing_probability: float
    acknowledgement_helpful_probability: float
    expects_answer_probability: float
    confidence: float
    speech_type: str
    cue_style: str
    cue_text: str


def decision_withdraws_approval(decision: object) -> bool:
    """Whether a decision retracts an already-granted approval.

    The model re-reads the partial transcript on a short cadence, so most
    snapshots are only "not helpful yet" - thin evidence, low confidence.
    Retracting on those killed cues that a moment earlier were judged useful:
    live sessions cancelled roughly one approved cue per audible one. Only a
    stop signal retracts an approval: the turn is ending, or the speech
    function says the user is asking something or is unclear.

    The classifier result is duck-typed (tests and alternate clients return
    plain objects), so a response that omits the fields retracts nothing.
    """

    if getattr(decision, "turn_stage", None) in {"nearing_end", "complete"}:
        return True
    speech_type = getattr(decision, "speech_type", None)
    if speech_type is None:
        return False
    return speech_type not in SPEECH_TYPE_TO_CUE_STYLE


SPEECH_TYPE_TO_CUE_STYLE = {
    "plain_continuation": "neutral",
    "list_or_story": "following",
    "explanation_context": "understanding",
}


class JevClassifier:
    """Ask Jev whether a short listener cue is useful during the current turn."""

    def __init__(
        self,
        client: JevClient,
        *,
        model: str = "jev-latest",
        approval_threshold: float = 0.55,
        helpful_threshold: float = 0.50,
        helpful_threshold_semantic: float = 0.52,
        expects_answer_ceiling: float = 0.60,
        expects_answer_ceiling_semantic: float = 0.50,
        available_cues: Collection[str] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._available_cues = set(available_cues) if available_cues is not None else None
        self._cue_index: dict[str, int] = {}
        self._last_chosen_cue: str | None = None
        self._approval_threshold = approval_threshold
        # How helpful a cue must look before it is played, and how likely the user
        # must be *not* to expect an answer. The semantic pair is the stricter one:
        # "uh-huh" and "I see" claim more than a neutral "mm-hmm" does.
        self._helpful_threshold = helpful_threshold
        self._helpful_threshold_semantic = helpful_threshold_semantic
        self._expects_answer_ceiling = expects_answer_ceiling
        self._expects_answer_ceiling_semantic = expects_answer_ceiling_semantic

    async def classify(
        self, transcript: str, context: Mapping[str, object] | None = None
    ) -> JevDecision:
        state: dict[str, object] = {"partial_transcript": transcript}
        if context:
            state["timing"] = dict(context)
        response = await self._client.system_one(
            state,
            {
                "turn_stage": Choice(
                    criteria={
                        "continuing": "The speaker is clearly continuing their current thought.",
                        "nearing_end": "The speaker is wrapping up but has not fully yielded.",
                        "complete": "The speaker expects the assistant to answer now.",
                    },
                    instructions="Classify the current partial voice turn conservatively.",
                ),
                "ack_helpful": Noul(
                    instructions=(
                        "Probability that a very short listener acknowledgement would feel helpful "
                        "and would not interrupt the speaker."
                    )
                ),
                "expects_answer": Noul(
                    instructions="Probability that the speaker expects a substantive answer now."
                ),
                "speech_type": Choice(
                    criteria={
                        "plain_continuation": (
                            "The speaker is continuing normally without a list, explanation, "
                            "question, or explicit signal that more is coming."
                        ),
                        "list_or_story": (
                            "The speaker is clearly listing steps or details, or telling a "
                            "story with more events still coming."
                        ),
                        "explanation_context": (
                            "The speaker has just explained a reason, context, or situation "
                            "and is continuing."
                        ),
                        "question_or_end": (
                            "The speaker is asking a direct question, yielding the turn, or "
                            "clearly finishing. Do not backchannel."
                        ),
                        "uncertain_or_ambiguous": (
                            "The speaker is emotional, making an unclear claim, or the right "
                            "listener response is ambiguous. Do not backchannel."
                        ),
                    },
                    instructions=(
                        "Classify what the speaker is doing, not just the topic. Prefer "
                        "question_or_end or uncertain_or_ambiguous whenever a cue could feel "
                        "intrusive. Do not infer list_or_story from ordinary continuous speech. "
                        "Do not use a positive acknowledgement to agree with an unverified claim."
                    ),
                ),
            },
            model=self._model,
        )
        answers = getattr(response, "answers", {})
        stage = answers["turn_stage"]
        helpful = float(answers["ack_helpful"].noul)
        expects_answer = float(answers["expects_answer"].noul)
        probabilities = getattr(stage, "probabilities", {})
        continuing = float(probabilities.get("continuing", 0.0))
        confidence = float(getattr(stage, "confidence", 0.0))
        choice = str(getattr(stage, "choice", "complete"))
        speech_type = str(getattr(answers["speech_type"], "choice", "uncertain_or_ambiguous"))
        cue_style = SPEECH_TYPE_TO_CUE_STYLE.get(speech_type, "neutral")
        cue_text, playable = self._choose_cue(cue_style)
        # A cue with no audio cannot be played from cache, and letting it fall through
        # to on-demand synthesis would turn a 20 ms cue into a one second one - the
        # trade this design exists to avoid. No audio therefore means no cue.
        cue_is_allowed = speech_type in SPEECH_TYPE_TO_CUE_STYLE and playable
        # Story and explanation cues carry more meaning than a neutral "mm-hmm",
        # so they keep a stricter bar than plain continuation - but both bars sit
        # where the classifier's probabilities actually fall. An earlier version
        # demanded 0.55 helpfulness and 0.45 "expects an answer" for those styles,
        # which rejected almost every "uh-huh"/"I see" the model offered (the
        # measured distribution clusters at 0.52-0.58) and left only the neutral
        # cue audible.
        neutral_continuation = speech_type == "plain_continuation"
        helpful_threshold = (
            self._helpful_threshold if neutral_continuation else self._helpful_threshold_semantic
        )
        answer_threshold = (
            self._expects_answer_ceiling
            if neutral_continuation
            else self._expects_answer_ceiling_semantic
        )
        approved = (
            choice == "continuing"
            and continuing >= self._approval_threshold
            and helpful >= helpful_threshold
            and expects_answer <= answer_threshold
            and confidence >= (0.45 if neutral_continuation else 0.5)
            and cue_is_allowed
        )
        return JevDecision(
            approved=approved,
            turn_stage=choice,
            continuing_probability=continuing,
            acknowledgement_helpful_probability=helpful,
            expects_answer_probability=expects_answer,
            confidence=confidence,
            speech_type=speech_type,
            cue_style=cue_style,
            cue_text=cue_text,
        )

    def _choose_cue(self, cue_style: str) -> tuple[str, bool]:
        """Pick the next phrase for a style, and whether it can actually be played.

        Rotation rather than a second model call: which of a style's phrases is used is
        a matter of variety, because they are interchangeable by construction. When no
        phrase for the style has audio, the style's default is still named so the
        decision reads sensibly, and `playable` is False so it is never approved.
        """

        default = CUE_TEXT_BY_STYLE.get(cue_style, CUE_TEXT_BY_STYLE["neutral"])
        candidates = [
            phrase
            for phrase in CUE_BANK_BY_STYLE.get(cue_style, ())
            if self._available_cues is None or phrase in self._available_cues
        ]
        if not candidates:
            return default, False
        index = self._cue_index.get(cue_style, 0) % len(candidates)
        chosen = candidates[index]
        if chosen == self._last_chosen_cue and len(candidates) > 1:
            index = (index + 1) % len(candidates)
            chosen = candidates[index]
        self._cue_index[cue_style] = index + 1
        self._last_chosen_cue = chosen
        return chosen, True

    async def aclose(self) -> None:
        close = getattr(self._client, "aclose", None)
        if callable(close):
            await close()


class JevClassifierProtocol(Protocol):
    async def classify(
        self, transcript: str, context: Mapping[str, object] | None = None
    ) -> JevDecision: ...


class JevTurnController:
    """Throttle Jev calls and reject results that outlive the user's turn."""

    def __init__(
        self,
        engine: BackchannelEngine,
        classifier: JevClassifierProtocol,
        *,
        min_interval_seconds: float = 0.75,
        timeout_seconds: float = 2.5,
        enabled: bool = True,
        max_approved_per_turn: int = 1,
        min_words_between_cues: int = 18,
        on_event: Callable[..., None] | None = None,
        on_cue_selected: Callable[[str], None] | None = None,
    ) -> None:
        if max_approved_per_turn < 1:
            raise ValueError("max_approved_per_turn must be at least 1")
        if min_words_between_cues < 1:
            raise ValueError("min_words_between_cues must be at least 1")
        self._engine = engine
        self._classifier = classifier
        self._min_interval_seconds = min_interval_seconds
        self._timeout_seconds = timeout_seconds
        self._enabled = enabled
        self._max_approved_per_turn = max_approved_per_turn
        self._min_words_between_cues = min_words_between_cues
        self._on_event = on_event or (lambda _name, **_data: None)
        self._on_cue_selected = on_cue_selected or (lambda _text: None)
        self._speaking = False
        self._generation = 0
        self._last_requested_at = float("-inf")
        self._approved_count = 0
        self._last_approved_word_count = 0
        self._turn_started_at = 0.0
        self._last_transcript_word_count = 0
        self._last_partial_at = 0.0
        self._recent_partials: deque[str] = deque(maxlen=3)
        self._pending_snapshot: tuple[str, dict[str, object]] | None = None
        self._last_cue_text: str | None = None
        self._task: asyncio.Task[None] | None = None
        self._clock = time.monotonic

    def set_enabled(self, enabled: bool) -> None:
        if self._enabled == enabled:
            return
        self._enabled = enabled
        if not enabled:
            self._invalidate("disabled")

    def user_started(self) -> None:
        self._generation += 1
        self._speaking = True
        self._approved_count = 0
        self._last_approved_word_count = 0
        self._turn_started_at = self._clock()
        self._last_transcript_word_count = 0
        self._last_partial_at = self._turn_started_at
        self._recent_partials.clear()
        self._pending_snapshot = None
        self._engine.set_semantic_approval(False)
        self._cancel_task("new_turn")

    def user_stopped(self) -> None:
        self._speaking = False
        self._pending_snapshot = None
        self._engine.set_semantic_approval(False)
        self._invalidate("turn_ended")

    def on_transcript(self, transcript: str, *, is_final: bool) -> None:
        if is_final:
            self._engine.set_semantic_approval(False)
            self._invalidate("final_transcript")
            return
        word_count = len(transcript.split())
        if (
            not self._enabled
            or not self._speaking
            or self._approved_count >= self._max_approved_per_turn
            or word_count < 4
            or (
                self._approved_count > 0
                and word_count - self._last_approved_word_count < self._min_words_between_cues
            )
        ):
            return
        now = self._clock()
        if self._task is not None and not self._task.done():
            if word_count >= self._last_transcript_word_count + 3:
                self._pending_snapshot = (transcript, {"word_count": word_count})
            return
        if now - self._last_requested_at < self._min_interval_seconds:
            self._pending_snapshot = (transcript, {"word_count": word_count})
            return
        self._last_requested_at = now
        new_words = max(0, word_count - self._last_transcript_word_count)
        previous_partial_at = self._last_partial_at
        context = {
            "speech_duration_ms": round((now - self._turn_started_at) * 1000, 1),
            "new_words_since_last_check": new_words,
            "milliseconds_since_previous_partial": round(
                max(0.0, now - previous_partial_at) * 1000, 1
            ),
            "end_of_turn_probability": self._engine.eot_probability,
            "recent_partial_transcripts": list(self._recent_partials),
            "last_cue_text": self._last_cue_text,
        }
        self._last_transcript_word_count = word_count
        self._last_partial_at = now
        self._recent_partials.append(transcript)
        generation = self._generation
        self._on_event("jev_request_started", word_count=word_count)
        self._task = asyncio.create_task(
            self._classify(generation, transcript, now, context),
            name="jev-backchannel-decision",
        )

    async def aclose(self) -> None:
        self._speaking = False
        self._invalidate("session_closed")
        classifier_close = getattr(self._classifier, "aclose", None)
        if callable(classifier_close):
            await classifier_close()

    def _invalidate(self, reason: str) -> None:
        self._generation += 1
        self._cancel_task(reason)

    def _cancel_task(self, reason: str | None = None) -> None:
        # A cancelled classification costs a provider call and answers nothing, so it
        # is reported rather than dropped: without it the request count in the log does
        # not add up to the decisions, timeouts and errors it produced.
        if self._task is not None and not self._task.done():
            if reason is not None:
                self._on_event("jev_decision_cancelled", reason=reason)
            self._task.cancel()
        self._task = None

    async def _classify(
        self,
        generation: int,
        transcript: str,
        started_at: float,
        context: Mapping[str, object],
    ) -> None:
        try:
            decision = await asyncio.wait_for(
                self._classifier.classify(transcript, context), timeout=self._timeout_seconds
            )
            if generation != self._generation or not self._speaking or not self._enabled:
                self._on_event("jev_decision_stale")
                return
            if decision.approved:
                self._on_cue_selected(decision.cue_text)
                self._last_cue_text = decision.cue_text
                self._approved_count += 1
                self._last_approved_word_count = len(transcript.split())
                self._engine.set_semantic_approval(True)
            elif decision_withdraws_approval(decision):
                # A stop signal retracts a standing approval; a snapshot that
                # is merely not helpful yet leaves it in place, so the cue the
                # model already blessed can still fire.
                self._engine.set_semantic_approval(False)
            self._on_event(
                "jev_decision",
                approved=decision.approved,
                turn_stage=decision.turn_stage,
                continuing_probability=decision.continuing_probability,
                acknowledgement_helpful_probability=(decision.acknowledgement_helpful_probability),
                expects_answer_probability=decision.expects_answer_probability,
                confidence=decision.confidence,
                speech_type=decision.speech_type,
                cue_style=decision.cue_style,
                cue_text=decision.cue_text,
                cue_number=self._approved_count if decision.approved else None,
                latency_ms=round((self._clock() - started_at) * 1000, 3),
            )
        except TimeoutError:
            self._engine.set_semantic_approval(False)
            self._on_event("jev_decision_timeout")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._engine.set_semantic_approval(False)
            self._on_event("jev_decision_error", error_type=type(exc).__name__)
        finally:
            if self._task is asyncio.current_task():
                self._task = None
            pending = self._pending_snapshot
            self._pending_snapshot = None
            if pending is not None and self._speaking:
                self.on_transcript(pending[0], is_final=False)
