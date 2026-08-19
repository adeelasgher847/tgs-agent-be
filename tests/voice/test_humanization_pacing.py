"""
Phase 4B-1: provider-agnostic pacing decisions on HumanizationDecision.pacing.

Pacing is metadata only — no text rewriting, no silence, no sleeps, no audio
change. Covers punctuation-aware sentence-boundary detection (decimals,
currency, phone numbers, dates, URLs must not be miscounted as sentence
boundaries), partial/mid-stream chunks, failure fallback, the feature flag,
and Twilio/LiveKit decision parity.
"""
from __future__ import annotations

import app.voice.humanization_engine as humanization_engine
from app.voice.humanization_engine import (
    PacingHint,
    SentenceEndingType,
    analyze_response,
)
from app.voice.turn_signals import UserMood


# ---------------------------------------------------------------------------
# Empty / short text
# ---------------------------------------------------------------------------


def test_empty_text_neutral_pacing():
    d = analyze_response("")
    assert d.pacing == PacingHint()


def test_short_text_marked_short():
    d = analyze_response("Okay.")
    assert d.pacing.is_short_utterance is True
    assert d.pacing.sentence_count == 1
    assert d.pacing.ending_type == SentenceEndingType.STATEMENT


# ---------------------------------------------------------------------------
# Single vs multiple sentences
# ---------------------------------------------------------------------------


def test_single_sentence():
    d = analyze_response("This is a complete sentence.")
    assert d.pacing.sentence_count == 1
    assert d.pacing.has_multiple_sentences is False


def test_multiple_sentences():
    d = analyze_response("First sentence. Second sentence. Third one too.")
    assert d.pacing.sentence_count == 3
    assert d.pacing.has_multiple_sentences is True
    assert d.pacing.has_internal_pause_opportunity is True


# ---------------------------------------------------------------------------
# Questions / exclamations
# ---------------------------------------------------------------------------


def test_question_ending_type():
    d = analyze_response("Would tomorrow at noon work for you?")
    assert d.pacing.ending_type == SentenceEndingType.QUESTION
    assert d.pacing.sentence_count == 1


def test_exclamation_ending_type():
    d = analyze_response("That's fantastic news!")
    assert d.pacing.ending_type == SentenceEndingType.EXCLAMATION


# ---------------------------------------------------------------------------
# Partial / mid-stream chunks
# ---------------------------------------------------------------------------


def test_partial_chunk_no_terminal_punctuation():
    d = analyze_response("Let me check on that for")
    assert d.pacing.sentence_count == 0
    assert d.pacing.ending_type == SentenceEndingType.NONE
    assert d.pacing.has_multiple_sentences is False


def test_partial_chunk_ending_mid_word_after_prior_sentence():
    d = analyze_response("Sure, I can help. Let me look into")
    # One completed sentence, one trailing partial fragment — still counts
    # only the true boundary, and the chunk as a whole has no terminal
    # punctuation at its very end.
    assert d.pacing.sentence_count == 1
    assert d.pacing.ending_type == SentenceEndingType.NONE


# ---------------------------------------------------------------------------
# Punctuation-inside-content safety: URLs, decimals, currency, phone, dates
# ---------------------------------------------------------------------------


def test_url_period_not_counted_as_sentence_boundary():
    d = analyze_response("Visit example.com for more info.")
    assert d.pacing.sentence_count == 1


def test_decimal_period_not_counted_as_sentence_boundary():
    d = analyze_response("The rate is 3.14 percent this month.")
    assert d.pacing.sentence_count == 1


def test_currency_period_not_counted_as_sentence_boundary():
    d = analyze_response("That will be $42.50 total.")
    assert d.pacing.sentence_count == 1


def test_phone_number_periods_not_counted_as_sentence_boundaries():
    d = analyze_response("Call us at 555.123.4567 today.")
    assert d.pacing.sentence_count == 1


def test_date_periods_not_counted_as_sentence_boundaries():
    d = analyze_response("The date is 3.5.2024 for the appointment.")
    assert d.pacing.sentence_count == 1


def test_multiple_real_sentences_with_embedded_decimal():
    d = analyze_response("The total is $42.50. Does that work for you?")
    assert d.pacing.sentence_count == 2
    assert d.pacing.has_multiple_sentences is True
    assert d.pacing.ending_type == SentenceEndingType.QUESTION


# ---------------------------------------------------------------------------
# Internal pause opportunity (comma/semicolon/colon, reused from tts_flush)
# ---------------------------------------------------------------------------


def test_single_sentence_with_comma_has_pause_opportunity():
    d = analyze_response("Well, let me check on that for you.")
    assert d.pacing.has_multiple_sentences is False
    assert d.pacing.has_internal_pause_opportunity is True


def test_single_short_sentence_without_comma_has_no_pause_opportunity():
    d = analyze_response("Sure thing.")
    assert d.pacing.has_internal_pause_opportunity is False


# ---------------------------------------------------------------------------
# Failure fallback
# ---------------------------------------------------------------------------


def test_pacing_failure_falls_back_to_neutral_pacing_only(monkeypatch):
    """
    A pacing-analysis failure must degrade only pacing to neutral — mood,
    tone-adapted text, and everything else in the decision must still be
    computed normally.
    """
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated pacing failure")

    monkeypatch.setattr(humanization_engine, "_pacing_hint", _boom)

    d = analyze_response(
        "This is ridiculous, but I can still help.", user_text="This is ridiculous"
    )
    assert d.pacing == PacingHint()
    # The rest of the decision is unaffected by the isolated pacing failure.
    assert d.mood == UserMood.FRUSTRATED
    assert d.text  # tone adaptation still ran


def test_whole_engine_failure_still_yields_neutral_pacing(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated engine failure")

    monkeypatch.setattr("app.voice.turn_signals.build_turn_context", _boom)

    d = analyze_response("Some response text.", user_text="hi")
    assert d.pacing == PacingHint()


# ---------------------------------------------------------------------------
# Feature disabled -> neutral pacing
# ---------------------------------------------------------------------------


def test_disabled_flag_yields_neutral_pacing(monkeypatch):
    monkeypatch.setattr(
        humanization_engine.settings, "VOICE_ENABLE_HUMANIZATION_ENGINE", False
    )
    d = analyze_response("First sentence. Second sentence. A question?")
    assert d.pacing == PacingHint()


# ---------------------------------------------------------------------------
# Text remains unchanged by pacing analysis
# ---------------------------------------------------------------------------


def test_pacing_analysis_does_not_alter_text():
    original = "The total is $42.50. Does that work for you?"
    d = analyze_response(original, use_ssml=True)
    # Pacing must never rewrite amounts/questions. This string has no canned
    # formal phrases, so tone_adapter must also leave it byte-for-byte intact.
    assert d.text == original


# ---------------------------------------------------------------------------
# Twilio/LiveKit parity: identical input -> identical pacing decision
# ---------------------------------------------------------------------------


def test_identical_input_gives_identical_pacing_for_both_transports():
    """
    There is exactly one pacing implementation (analyze_response), called
    from the single shared TtsPipeline._process_chunk for both transports
    (see Phase 4A) — so any two callers passing the same arguments must get
    an identical PacingHint, proving no transport-specific divergence.
    """
    text = "Sure, I can help with that. Would 3 PM work for you?"
    d_twilio_like = analyze_response(text, user_text="hi", use_ssml=True, stt_confidence=0.9)
    d_livekit_like = analyze_response(text, user_text="hi", use_ssml=True, stt_confidence=0.9)
    assert d_twilio_like.pacing == d_livekit_like.pacing
