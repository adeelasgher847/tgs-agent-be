"""
Regression coverage for VoiceOrchestrator._resolve_initial_endpointing_ms and
the VOICE_STT_ENDPOINTING_MODE default.

Root cause (confirmed via three separate real production call logs): the
default mode was "aggressive", which computes
max(80, int(DEEPGRAM_STT_ENDPOINTING_MS * 0.55)) clamped to 400 -- against
this repo's own DEEPGRAM_STT_ENDPOINTING_MS=350, that works out to ~192ms.
Deepgram routinely marked speech_final mid-sentence on an ordinary
conversational pause between clauses/words at that threshold (a single
continuous caller thought split into up to 3 separate "final" transcripts
within ~2 seconds in one real call), so the agent started responding
before the caller had actually finished speaking -- worse on longer
sentences, which have proportionally more chances to hit a >192ms gap.

Fix: default changed to "normal", which uses DEEPGRAM_STT_ENDPOINTING_MS
(350ms) directly -- this repo's own originally-documented deliberate value
for that setting, not a newly-invented number. These tests lock in that
default and the exact math for all three modes so a future change here is
deliberate, not an accidental regression back to the aggressive default.
"""

from __future__ import annotations

from app.core.config import settings
from app.voice.voice_orchestrator import _resolve_initial_endpointing_ms


def test_default_mode_is_normal_not_aggressive():
    """The specific regression this fix addresses: the default must not be
    'aggressive' (192ms effective), which caused premature mid-sentence
    turn-taking on real Twilio calls."""
    assert settings.VOICE_STT_ENDPOINTING_MODE == "normal"


def test_base_endpointing_raised_past_first_fix_after_further_reports():
    """350ms (this session's first fix) still wasn't patient enough --
    callers kept getting cut off on ordinary pauses between clauses of one
    longer sentence. Locks in the follow-up increase to 450/700ms so a
    future edit here is deliberate, not an accidental regression back
    toward the too-aggressive end of the range."""
    assert settings.DEEPGRAM_STT_ENDPOINTING_MS == 450
    assert settings.DEEPGRAM_STT_ENDPOINTING_MS_EXTENDED == 700
    # EXTENDED must stay strictly more patient than normal conversation.
    assert settings.DEEPGRAM_STT_ENDPOINTING_MS_EXTENDED > settings.DEEPGRAM_STT_ENDPOINTING_MS


def test_normal_mode_uses_base_endpointing_directly(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_STT_ENDPOINTING_MODE", "normal", raising=False)
    monkeypatch.setattr(settings, "DEEPGRAM_STT_ENDPOINTING_MS", 350, raising=False)
    assert _resolve_initial_endpointing_ms() == 350


def test_extended_mode_uses_max_of_base_and_extended(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_STT_ENDPOINTING_MODE", "extended", raising=False)
    monkeypatch.setattr(settings, "DEEPGRAM_STT_ENDPOINTING_MS", 350, raising=False)
    monkeypatch.setattr(settings, "DEEPGRAM_STT_ENDPOINTING_MS_EXTENDED", 500, raising=False)
    assert _resolve_initial_endpointing_ms() == 500


def test_aggressive_mode_still_available_for_explicit_opt_in(monkeypatch):
    """The aggressive formula itself is untouched -- only the default
    changed -- so a deployment that explicitly wants faster/choppier
    turn-taking can still opt in."""
    monkeypatch.setattr(settings, "VOICE_STT_ENDPOINTING_MODE", "aggressive", raising=False)
    monkeypatch.setattr(settings, "DEEPGRAM_STT_ENDPOINTING_MS", 350, raising=False)
    assert _resolve_initial_endpointing_ms() == 192


def test_unknown_mode_falls_back_to_base():
    """Defensive: an unrecognized mode string must behave like 'normal',
    not silently apply the aggressive reduction."""
    import app.voice.voice_orchestrator as vo_module

    class _FakeSettings:
        VOICE_STT_ENDPOINTING_MODE = "some-typo"
        DEEPGRAM_STT_ENDPOINTING_MS = 350
        DEEPGRAM_STT_ENDPOINTING_MS_EXTENDED = 500

    original = vo_module.settings
    vo_module.settings = _FakeSettings()
    try:
        assert _resolve_initial_endpointing_ms() == 350
    finally:
        vo_module.settings = original


def test_new_default_meaningfully_safer_than_old_aggressive_default():
    """Guardrail against silently reverting to the too-aggressive value:
    the effective threshold with today's actual settings must be
    comfortably above the ~192ms that caused the regression, without
    drifting all the way up to the 'extended' (spelling/email) tolerance."""
    effective = _resolve_initial_endpointing_ms()
    assert effective >= 300
    assert effective < settings.DEEPGRAM_STT_ENDPOINTING_MS_EXTENDED
