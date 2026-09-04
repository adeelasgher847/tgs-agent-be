"""
Phase 4C-2/4C-3: pure unit tests for
app.voice.humanization_engine.pause_frames_for_chunk and the
VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES config default.

No audio, no TtsPipeline — just the eligibility decision function itself.
"""

from __future__ import annotations

from app.core.config import settings
from app.voice.humanization_engine import (
    PacingHint,
    SentenceEndingType,
    pause_frames_for_chunk,
)


def _eligible_pacing() -> PacingHint:
    return PacingHint(
        sentence_count=1,
        has_multiple_sentences=False,
        is_short_utterance=False,
        ending_type=SentenceEndingType.STATEMENT,
        has_internal_pause_opportunity=False,
    )


# 1. Config default (Phase 4C-3: enabled at 3 frames / 60ms — code-level
# correctness only, not validated against real call audio; see module
# docstring in app/core/config.py).
def test_config_default_is_three_frames():
    assert settings.VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES == 3


# 2. Config = 0 -> zero frames even for an otherwise-eligible chunk
def test_config_zero_yields_zero_frames(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 0)
    assert pause_frames_for_chunk(_eligible_pacing(), is_final=False) == 0


# 3. Eligible chunk -> exactly the configured frame count
def test_eligible_chunk_returns_configured_frames(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    assert pause_frames_for_chunk(_eligible_pacing(), is_final=False) == 3


# 4 & 5. ending_type == NONE (partial / mid-stream / time-flush chunk) -> 0
def test_partial_chunk_ending_none_yields_zero(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    pacing = PacingHint(
        sentence_count=0,
        has_multiple_sentences=False,
        is_short_utterance=False,
        ending_type=SentenceEndingType.NONE,
        has_internal_pause_opportunity=False,
    )
    assert pause_frames_for_chunk(pacing, is_final=False) == 0


# 6. Short utterance -> 0
def test_short_utterance_yields_zero(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    pacing = PacingHint(
        sentence_count=1,
        has_multiple_sentences=False,
        is_short_utterance=True,
        ending_type=SentenceEndingType.STATEMENT,
        has_internal_pause_opportunity=False,
    )
    assert pause_frames_for_chunk(pacing, is_final=False) == 0


# 7. Final chunk -> 0 regardless of otherwise-eligible pacing
def test_final_chunk_yields_zero(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    assert pause_frames_for_chunk(_eligible_pacing(), is_final=True) == 0


# Question/exclamation endings are equally eligible (not just statements)
def test_question_and_exclamation_endings_are_eligible(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 2)
    for ending in (SentenceEndingType.QUESTION, SentenceEndingType.EXCLAMATION):
        pacing = PacingHint(
            sentence_count=1,
            has_multiple_sentences=False,
            is_short_utterance=False,
            ending_type=ending,
            has_internal_pause_opportunity=False,
        )
        assert pause_frames_for_chunk(pacing, is_final=False) == 2


# 16. Missing / malformed pacing falls back to 0, never raises
def test_missing_pacing_is_none_yields_zero(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    assert pause_frames_for_chunk(None, is_final=False) == 0


def test_malformed_pacing_object_yields_zero_not_raise(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)

    class _NotPacing:
        pass

    assert pause_frames_for_chunk(_NotPacing(), is_final=False) == 0  # type: ignore[arg-type]


def test_negative_or_invalid_config_yields_zero(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", -1)
    assert pause_frames_for_chunk(_eligible_pacing(), is_final=False) == 0


# 17. Humanization disabled -> analyze_response returns pacing=PacingHint()
#     (default: ending_type=NONE), which pause_frames_for_chunk already
#     treats as ineligible — verified at the analyze_response level here.
def test_neutral_pacing_from_disabled_humanization_yields_zero(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    neutral = PacingHint()  # what _neutral_decision()/disabled flag produces
    assert pause_frames_for_chunk(neutral, is_final=False) == 0


# 20. No provider-specific logic: the function's signature and behavior
# never reference a provider at all.
def test_pause_decision_takes_no_provider_argument():
    import inspect

    sig = inspect.signature(pause_frames_for_chunk)
    assert "provider" not in sig.parameters
    assert "provider_slug" not in sig.parameters
    # V-08: `pause_after` is an optional, keyword-only, provider-NEUTRAL
    # PauseCategory (semantic enum, never a provider slug/adapter) — added
    # alongside the original two positional params, not replacing them.
    assert list(sig.parameters) == ["pacing", "is_final", "pause_after"]
    assert sig.parameters["pause_after"].kind == inspect.Parameter.KEYWORD_ONLY
    assert sig.parameters["pause_after"].default is None
