"""
Regression coverage for `turn_signals.is_utterance_likely_incomplete` --
the trailing-word heuristic behind the Twilio-only "incomplete final grace
window" fix (VOICE_STT_INCOMPLETE_FINAL_GRACE_MS in app/core/config.py).

Root cause this addresses: raising Deepgram's blanket silence-based
endpointing (DEEPGRAM_STT_ENDPOINTING_MS) helps overall, but a caller who
speaks a long sentence with a natural mid-clause pause ("I wanted to ask
about the pricing and... the onboarding process") can still get cut off,
because ANY fixed silence threshold has to be short enough to keep short
replies snappy. This heuristic targets the specific case a blanket
threshold can't: text that reads as unfinished regardless of how long the
pause was.
"""

from __future__ import annotations

from app.voice.turn_signals import is_utterance_likely_incomplete


class TestCompleteUtterances:
    def test_short_complete_reply_is_not_flagged(self):
        assert is_utterance_likely_incomplete("Yes.") is False
        assert is_utterance_likely_incomplete("No thanks") is False

    def test_terminal_punctuation_overrides_trailing_word(self):
        # Deepgram's own smart_format punctuation is a stronger "done"
        # signal than a heuristic trailing-word guess.
        assert is_utterance_likely_incomplete("I want to talk about pricing.") is False
        assert is_utterance_likely_incomplete("Can you help me with that?") is False
        assert is_utterance_likely_incomplete("That's amazing!") is False

    def test_normal_sentence_ending_in_noun_is_not_flagged(self):
        assert (
            is_utterance_likely_incomplete("I'd like to book an appointment")
            is False
        )

    def test_empty_or_whitespace_is_not_flagged(self):
        assert is_utterance_likely_incomplete("") is False
        assert is_utterance_likely_incomplete("   ") is False
        assert is_utterance_likely_incomplete(None) is False  # type: ignore[arg-type]


class TestIncompleteUtterances:
    def test_trailing_conjunction(self):
        assert is_utterance_likely_incomplete("I wanted to ask about pricing and") is True

    def test_trailing_preposition(self):
        assert is_utterance_likely_incomplete("Can you send that to") is True

    def test_trailing_article(self):
        assert is_utterance_likely_incomplete("I need to speak with the") is True

    def test_trailing_filler(self):
        assert is_utterance_likely_incomplete("So I was thinking, um") is True

    def test_trailing_comma_without_terminal_punctuation(self):
        assert is_utterance_likely_incomplete("First my name is John,") is True


class TestConservativeScope:
    """Guards against false positives that would delay genuinely finished
    short replies -- the heuristic must stay narrow."""

    def test_word_that_merely_contains_a_flagged_substring_is_not_flagged(self):
        # "and" is flagged, but a real word ending differently must not be.
        assert is_utterance_likely_incomplete("I live in Portland") is False

    def test_case_insensitive_match_still_conservative(self):
        assert is_utterance_likely_incomplete("Let's talk about AND") is True
