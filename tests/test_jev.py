import asyncio
from types import SimpleNamespace

from blue_machines_baseline.backchannel import BackchannelEngine
from blue_machines_baseline.jev import (
    CUE_BANK_BY_STYLE,
    CUE_TEXT_BY_STYLE,
    JevClassifier,
    JevTurnController,
)


class FakeHandle:
    def interrupt(self, *, force: bool = False) -> object:
        return self


class FakeClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[object, object]] = []

    async def system_one(self, state: object, questions: object, **_kwargs: object) -> object:
        self.calls.append((state, questions))
        return self.response


def test_jev_classifier_approves_only_a_helpful_continuing_turn() -> None:
    async def scenario() -> None:
        response = SimpleNamespace(
            answers={
                "turn_stage": SimpleNamespace(
                    choice="continuing",
                    confidence=0.91,
                    probabilities={"continuing": 0.88, "nearing_end": 0.09, "complete": 0.03},
                ),
                "ack_helpful": SimpleNamespace(noul=0.84),
                "expects_answer": SimpleNamespace(noul=0.08),
                "speech_type": SimpleNamespace(
                    choice="list_or_story",
                    confidence=0.86,
                    probabilities={"plain_continuation": 0.1, "list_or_story": 0.8},
                ),
            }
        )
        client = FakeClient(response)
        classifier = JevClassifier(client, approval_threshold=0.65)

        decision = await classifier.classify(
            "I started by mapping the requirements and then I looked at the audio path"
        )

        assert decision.approved is True
        assert decision.turn_stage == "continuing"
        assert decision.continuing_probability == 0.88
        assert decision.cue_text == "uh-huh"
        assert len(client.calls) == 1

    asyncio.run(scenario())


def test_jev_classifier_allows_a_reasonable_live_backchannel() -> None:
    async def scenario() -> None:
        response = SimpleNamespace(
            answers={
                "turn_stage": SimpleNamespace(
                    choice="continuing",
                    confidence=0.78,
                    probabilities={"continuing": 0.72, "nearing_end": 0.2, "complete": 0.08},
                ),
                "ack_helpful": SimpleNamespace(noul=0.58),
                "expects_answer": SimpleNamespace(noul=0.25),
                "speech_type": SimpleNamespace(
                    choice="plain_continuation",
                    confidence=0.7,
                    probabilities={"plain_continuation": 0.7, "list_or_story": 0.2},
                ),
            }
        )

        decision = await JevClassifier(FakeClient(response)).classify(
            "I am still explaining the second part of the approach"
        )

        assert decision.approved is True
        assert decision.cue_text == "mm-hmm"

    asyncio.run(scenario())


def _decision(**overrides: object) -> SimpleNamespace:
    base = {
        "approved": True,
        "turn_stage": "continuing",
        "continuing_probability": 0.9,
        "acknowledgement_helpful_probability": 0.9,
        "expects_answer_probability": 0.0,
        "confidence": 0.9,
        "speech_type": "plain_continuation",
        "cue_style": "neutral",
        "cue_text": "mm-hmm",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_a_merely_unhelpful_snapshot_keeps_a_granted_approval() -> None:
    """A cautious re-read must not cancel the cue the model already blessed."""

    async def scenario() -> None:
        calls = 0

        class Classifier:
            async def classify(self, _transcript: str, _context=None):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return _decision()
                return _decision(
                    approved=False,
                    continuing_probability=0.62,
                    acknowledgement_helpful_probability=0.48,
                    expects_answer_probability=0.4,
                    confidence=0.5,
                )

        played = False

        def play() -> FakeHandle:
            nonlocal played
            played = True
            return FakeHandle()

        engine = BackchannelEngine(
            play, delay_seconds=0.05, cooldown_seconds=0, semantic_required=True
        )
        controller = JevTurnController(
            engine,
            Classifier(),
            min_interval_seconds=0,
            timeout_seconds=1,
            max_approved_per_turn=5,
            min_words_between_cues=1,
        )

        engine.user_started()
        controller.user_started()
        controller.on_transcript(
            "this explanation keeps going with plenty of words in it", is_final=False
        )
        await asyncio.sleep(0.02)
        controller.on_transcript(
            "this explanation keeps going with plenty of words in it still", is_final=False
        )
        await asyncio.sleep(0.09)

        assert calls >= 2
        assert played is True
        await controller.aclose()
        await engine.aclose()

    asyncio.run(scenario())


def test_a_question_snapshot_retracts_a_granted_approval() -> None:
    async def scenario() -> None:
        calls = 0

        class Classifier:
            async def classify(self, _transcript: str, _context=None):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return _decision()
                return _decision(approved=False, speech_type="question_or_end")

        played = False

        def play() -> FakeHandle:
            nonlocal played
            played = True
            return FakeHandle()

        engine = BackchannelEngine(
            play, delay_seconds=0.05, cooldown_seconds=0, semantic_required=True
        )
        controller = JevTurnController(
            engine,
            Classifier(),
            min_interval_seconds=0,
            timeout_seconds=1,
            max_approved_per_turn=5,
            min_words_between_cues=1,
        )

        engine.user_started()
        controller.user_started()
        controller.on_transcript(
            "this explanation keeps going with plenty of words in it", is_final=False
        )
        await asyncio.sleep(0.02)
        controller.on_transcript(
            "this explanation keeps going with plenty of words in it still", is_final=False
        )
        await asyncio.sleep(0.09)

        assert calls >= 2
        assert played is False
        await controller.aclose()
        await engine.aclose()

    asyncio.run(scenario())


def test_jev_classifier_maps_each_safe_style_to_a_distinct_phrase() -> None:
    assert CUE_TEXT_BY_STYLE == {
        "neutral": "mm-hmm",
        "following": "uh-huh",
        "understanding": "I see",
    }


def test_jev_controller_never_applies_a_result_after_user_stops() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        class SlowClassifier:
            async def classify(self, _transcript: str, _context=None):
                started.set()
                await release.wait()
                return SimpleNamespace(
                    approved=True,
                    turn_stage="continuing",
                    continuing_probability=0.9,
                    acknowledgement_helpful_probability=0.9,
                    expects_answer_probability=0.0,
                    confidence=0.9,
                    speech_type="plain_continuation",
                    cue_style="neutral",
                    cue_text="mm-hmm",
                )

        played = False

        def play() -> FakeHandle:
            nonlocal played
            played = True
            return FakeHandle()

        engine = BackchannelEngine(
            play,
            delay_seconds=0,
            cooldown_seconds=0,
            semantic_required=True,
        )
        controller = JevTurnController(
            engine,
            SlowClassifier(),
            min_interval_seconds=0,
            timeout_seconds=1,
        )

        engine.user_started()
        controller.user_started()
        controller.on_transcript(
            "I am continuing with a long explanation about the implementation",
            is_final=False,
        )
        await started.wait()
        controller.user_stopped()
        engine.user_stopped()
        release.set()
        await asyncio.sleep(0.02)

        assert played is False
        await controller.aclose()
        await engine.aclose()

    asyncio.run(scenario())


def test_jev_controller_stops_classifying_after_one_approved_cue() -> None:
    async def scenario() -> None:
        calls = 0

        class ApprovingClassifier:
            async def classify(self, _transcript: str, _context=None):
                nonlocal calls
                calls += 1
                return SimpleNamespace(
                    approved=True,
                    turn_stage="continuing",
                    continuing_probability=0.9,
                    acknowledgement_helpful_probability=0.9,
                    expects_answer_probability=0.0,
                    confidence=0.9,
                    speech_type="list_or_story",
                    cue_style="following",
                    cue_text="uh-huh",
                )

        engine = BackchannelEngine(
            lambda: FakeHandle(),
            delay_seconds=0,
            cooldown_seconds=0,
            semantic_required=True,
        )
        controller = JevTurnController(
            engine,
            ApprovingClassifier(),
            min_interval_seconds=0,
            timeout_seconds=1,
        )
        engine.user_started()
        controller.user_started()
        controller.on_transcript("this explanation is still continuing now", is_final=False)
        await asyncio.sleep(0.02)
        controller.on_transcript(
            "this explanation is still continuing now with more detail", is_final=False
        )
        await asyncio.sleep(0.02)

        assert calls == 1
        await controller.aclose()
        await engine.aclose()

    asyncio.run(scenario())


def test_jev_controller_allows_one_more_cue_after_a_long_continuation() -> None:
    async def scenario() -> None:
        calls = 0

        class ApprovingClassifier:
            async def classify(self, _transcript: str, _context=None):
                nonlocal calls
                calls += 1
                return SimpleNamespace(
                    approved=True,
                    turn_stage="continuing",
                    continuing_probability=0.9,
                    acknowledgement_helpful_probability=0.9,
                    expects_answer_probability=0.0,
                    confidence=0.9,
                    speech_type="list_or_story",
                    cue_style="following",
                    cue_text="uh-huh",
                )

        engine = BackchannelEngine(
            lambda: FakeHandle(),
            delay_seconds=0,
            cooldown_seconds=0,
            semantic_required=True,
            max_acknowledgements_per_turn=2,
        )
        controller = JevTurnController(
            engine,
            ApprovingClassifier(),
            min_interval_seconds=0,
            timeout_seconds=1,
            max_approved_per_turn=2,
            min_words_between_cues=3,
        )
        engine.user_started()
        controller.user_started()
        controller.on_transcript("this explanation is still continuing now", is_final=False)
        await asyncio.sleep(0.02)
        controller.on_transcript(
            "this explanation is still continuing now with more detail for the next point",
            is_final=False,
        )
        await asyncio.sleep(0.02)

        assert calls == 2
        await controller.aclose()
        await engine.aclose()

    asyncio.run(scenario())


def test_jev_timeout_fails_closed_without_playing_audio() -> None:
    async def scenario() -> None:
        played = False
        events: list[str] = []

        class SlowClassifier:
            async def classify(self, _transcript: str, _context=None):
                await asyncio.sleep(1)

        def play() -> FakeHandle:
            nonlocal played
            played = True
            return FakeHandle()

        engine = BackchannelEngine(
            play,
            delay_seconds=0,
            cooldown_seconds=0,
            semantic_required=True,
        )
        controller = JevTurnController(
            engine,
            SlowClassifier(),
            min_interval_seconds=0,
            timeout_seconds=0.01,
            on_event=lambda name, **_data: events.append(name),
        )
        engine.user_started()
        controller.user_started()
        controller.on_transcript("this is a continuing explanation now", is_final=False)
        await asyncio.sleep(0.03)

        assert played is False
        assert "jev_decision_timeout" in events
        await controller.aclose()
        await engine.aclose()

    asyncio.run(scenario())


def test_jev_controller_accounts_for_a_classification_it_cancels() -> None:
    """A call cancelled when the turn ends costs a request and answers nothing.

    Without this event the request count in the log never reconciles with the
    decisions, timeouts and errors it produced, which hides both wasted spend and
    how often a decision was thrown away for arriving too late.
    """

    async def scenario() -> None:
        events: list[tuple[str, dict]] = []
        started = asyncio.Event()
        release = asyncio.Event()

        class SlowClassifier:
            async def classify(self, _transcript: str, _context=None):
                started.set()
                await release.wait()
                return SimpleNamespace(approved=True)

        engine = BackchannelEngine(
            play=lambda: FakeHandle(),
            delay_seconds=0,
            cooldown_seconds=0,
            semantic_required=True,
        )
        controller = JevTurnController(
            engine,
            SlowClassifier(),
            min_interval_seconds=0,
            timeout_seconds=5,
            on_event=lambda name, **data: events.append((name, data)),
        )
        engine.user_started()
        controller.user_started()
        controller.on_transcript("this is a continuing explanation now", is_final=False)
        await started.wait()
        controller.user_stopped()
        release.set()
        await asyncio.sleep(0.02)

        assert ("jev_decision_cancelled", {"reason": "turn_ended"}) in events
        # The cancelled call must not be reported as a decision.
        assert not [name for name, _ in events if name == "jev_decision"]
        await controller.aclose()
        await engine.aclose()

    asyncio.run(scenario())


def test_jev_error_fails_closed_without_playing_audio() -> None:
    async def scenario() -> None:
        played = False
        events: list[str] = []

        class BrokenClassifier:
            async def classify(self, _transcript: str, _context=None):
                raise RuntimeError("provider unavailable")

        def play() -> FakeHandle:
            nonlocal played
            played = True
            return FakeHandle()

        engine = BackchannelEngine(
            play,
            delay_seconds=0,
            cooldown_seconds=0,
            semantic_required=True,
        )
        controller = JevTurnController(
            engine,
            BrokenClassifier(),
            min_interval_seconds=0,
            timeout_seconds=1,
            on_event=lambda name, **_data: events.append(name),
        )
        engine.user_started()
        controller.user_started()
        controller.on_transcript("this is a continuing explanation now", is_final=False)
        await asyncio.sleep(0.02)

        assert played is False
        assert "jev_decision_error" in events
        await controller.aclose()
        await engine.aclose()

    asyncio.run(scenario())


def test_each_style_has_more_than_one_phrase_to_vary() -> None:
    assert all(len(bank) >= 2 for bank in CUE_BANK_BY_STYLE.values())


def _answer(*, speech_type: str, helpful: float = 0.9, expects: float = 0.1) -> SimpleNamespace:
    """A classifier response that approves, for a given speech type."""

    return SimpleNamespace(
        answers={
            "turn_stage": SimpleNamespace(
                choice="continuing", confidence=0.9, probabilities={"continuing": 0.9}
            ),
            "ack_helpful": SimpleNamespace(noul=helpful),
            "expects_answer": SimpleNamespace(noul=expects),
            "speech_type": SimpleNamespace(
                choice=speech_type, confidence=0.9, probabilities={speech_type: 0.9}
            ),
        }
    )


def test_the_classifier_never_repeats_the_cue_it_just_used() -> None:
    """Two approvals of the same style should not both say "mm-hmm"."""

    async def scenario() -> None:
        classifier = JevClassifier(FakeClient(_answer(speech_type="plain_continuation")))

        first = (await classifier.classify("a partial transcript that is continuing")).cue_text
        second = (await classifier.classify("another partial transcript")).cue_text

        assert first != second
        assert {first, second} <= set(CUE_BANK_BY_STYLE["neutral"])

    asyncio.run(scenario())


def test_a_cue_without_audio_is_not_approved() -> None:
    """Falling through to on-demand synthesis would turn a 20 ms cue into a 1 s one."""

    async def scenario() -> None:
        classifier = JevClassifier(
            FakeClient(_answer(speech_type="list_or_story")),
            available_cues={"mm-hmm", "I see"},  # nothing from the "following" bank
        )

        decision = await classifier.classify("so first this, then that, and then the other")

        assert decision.cue_style == "following"
        assert decision.approved is False
        # The decision still names the style's default cue; it is simply not playable.
        assert decision.cue_text == CUE_BANK_BY_STYLE["following"][0]

    asyncio.run(scenario())


def test_the_classifier_uses_the_playable_phrase_in_its_style() -> None:
    async def scenario() -> None:
        classifier = JevClassifier(
            FakeClient(_answer(speech_type="list_or_story")),
            available_cues={"uh-huh"},
        )

        decision = await classifier.classify("so first this, then that, and then the other")

        assert decision.approved is True
        assert decision.cue_text == "uh-huh"

    asyncio.run(scenario())


def test_the_cue_bank_covers_ten_ways_to_acknowledge_without_agreeing() -> None:
    """The vocabulary is grouped by what a cue claims, and nothing in it agrees.

    Words like "yes" or "exactly" would have the agent endorse a claim it has not
    checked, which is why the bank stops at acknowledgements a listener makes about
    *hearing* someone rather than about the truth of what they said.
    """

    phrases = [cue for bank in CUE_BANK_BY_STYLE.values() for cue in bank]

    assert len(phrases) >= 10
    assert len(set(phrases)) == len(phrases)
    assert {"mm-hmm", "uh-huh", "I see"} <= set(phrases)
    banned = {"yes", "yeah", "sure", "exactly", "of course", "correct", "agree"}
    assert not banned & set(phrases)
    assert all(len(bank) >= 2 for bank in CUE_BANK_BY_STYLE.values())
    # The default cue per style is what older logs and docs refer to.
    assert all(CUE_TEXT_BY_STYLE[style] == bank[0] for style, bank in CUE_BANK_BY_STYLE.items())
