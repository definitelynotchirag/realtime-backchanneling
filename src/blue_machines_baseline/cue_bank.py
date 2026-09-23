"""The acknowledgement vocabulary, and what each cue claims.

Kept dependency-free so settings can validate against it without importing the audio
stack, and so the safety reasoning lives next to the words it constrains.

The grouping is the safety property. Phrases inside a group are interchangeable, so a
policy can vary the wording without changing what is asserted; a policy may only use a
group whose claim it has actually checked.
"""

from __future__ import annotations

CUE_BANK_BY_STYLE: dict[str, tuple[str, ...]] = {
    # "I am still listening." Claims nothing about the content.
    "neutral": ("mm-hmm", "mm", "hmm"),
    # "Stay with me." Used while the speaker lists steps or tells a story.
    "following": ("uh-huh", "go on", "keep going"),
    # "That landed." Acknowledges comprehension of the telling, never agreement with it.
    "understanding": ("I see", "got it", "okay", "makes sense"),
}

CUE_TEXT_BY_STYLE: dict[str, str] = {
    style: phrases[0] for style, phrases in CUE_BANK_BY_STYLE.items()
}
"""The default cue for each style, chosen when a style's other phrases have no audio."""

CUE_TEXTS: tuple[str, ...] = tuple(cue for phrases in CUE_BANK_BY_STYLE.values() for cue in phrases)
"""Every acknowledgement there is."""

NEUTRAL_CUES: tuple[str, ...] = CUE_BANK_BY_STYLE["neutral"]
"""The only group a policy without a semantic gate may use.

The timer policy cues on a schedule while the user is still speaking; it has not
classified anything, so "got it" or "makes sense" from it would claim that something
landed when nothing checked that it did, and "go on" fired mid-sentence reads as an
interruption rather than an invitation. Jev earns the stronger cues by classifying the
turn first.
"""

AGREEMENT_WORDS: frozenset[str] = frozenset(
    {"yes", "yeah", "right", "sure", "exactly", "of course", "correct", "agree", "yep"}
)
"""Words that read as agreement and are therefore not in the bank.

"right" is included here deliberately: it means "correct" often enough that a listener
cannot tell it apart from endorsement, and an acknowledgement must never endorse a claim
the agent has not checked.
"""
