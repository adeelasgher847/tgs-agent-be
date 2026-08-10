from app.voice.humanization_engine import HumanizationDecision, analyze_response
from app.voice.tts_provider_capabilities import (
    build_voice_settings_overlay,
    get_capabilities,
)

# ---------------------------------------------------------------------------
# ElevenLabs
# ---------------------------------------------------------------------------


def test_elevenlabs_capability_detection():
    caps = get_capabilities("elevenlabs")
    assert caps.provider_slug == "elevenlabs"
    assert caps.supports_streaming is True
    assert caps.supports_speaking_rate is True
    assert caps.supports_stability_control is True
    assert caps.supports_native_expressive_tags is True
    # Not supported on the live streaming path today, regardless of docs.
    assert caps.supports_ssml is False
    assert caps.supports_pitch is False
    assert caps.supports_pause_control is False


def test_elevenlabs_stability_hint_mapping():
    decision = analyze_response(
        "I understand, let's get that fixed for you.",
        user_text="This is unacceptable, I want a refund now",
    )
    assert decision.tts_stability_hint is not None
    overlay = build_voice_settings_overlay("elevenlabs", decision)
    assert overlay == {"stability": decision.tts_stability_hint}


def test_elevenlabs_stability_hint_is_clamped_to_valid_range():
    decision = HumanizationDecision(
        text="hi",
        mood=decision_mood(),
        response_emotion="neutral",
        pacing=pacing_stub(),
        acknowledgement=ack_stub(),
        filler=filler_stub(),
        tts_stability_hint=5.0,  # out of ElevenLabs' valid [0.0, 1.0] range
    )
    overlay = build_voice_settings_overlay("elevenlabs", decision)
    assert overlay["stability"] == 1.0


def test_elevenlabs_default_stability_behavior_when_no_hint():
    decision = HumanizationDecision(
        text="hi",
        mood=decision_mood(),
        response_emotion="neutral",
        pacing=pacing_stub(),
        acknowledgement=ack_stub(),
        filler=filler_stub(),
        tts_stability_hint=None,
    )
    overlay = build_voice_settings_overlay("elevenlabs", decision)
    # No "stability" key at all -> caller's/service's existing default (0.45
    # in ElevenLabsService._default_voice_settings) remains untouched.
    assert "stability" not in overlay
    assert overlay == {}


def test_elevenlabs_invalid_hint_type_falls_back_safely():
    decision = HumanizationDecision(
        text="hi",
        mood=decision_mood(),
        response_emotion="neutral",
        pacing=pacing_stub(),
        acknowledgement=ack_stub(),
        filler=filler_stub(),
        tts_stability_hint="not-a-number",  # type: ignore[arg-type]
    )
    overlay = build_voice_settings_overlay("elevenlabs", decision)
    assert overlay == {}


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------


def test_google_supported_capabilities():
    caps = get_capabilities("google")
    assert caps.supports_streaming is True
    assert caps.supports_speaking_rate is True


def test_google_unsupported_capability_behavior():
    caps = get_capabilities("google")
    assert caps.supports_stability_control is False
    assert caps.supports_pitch is False
    assert caps.supports_ssml is False
    assert caps.supports_native_expressive_tags is False
    assert caps.supports_pause_control is False


def test_google_overlay_never_emits_stability_or_pitch():
    decision = analyze_response(
        "Let's take care of that right away.",
        user_text="This is unacceptable, I want a refund now",
    )
    overlay = build_voice_settings_overlay("google", decision)
    # Capability-gated: Google has no stability/pitch concept, so the
    # overlay must stay empty even though the decision carries a hint.
    assert overlay == {}


# ---------------------------------------------------------------------------
# Rime
# ---------------------------------------------------------------------------


def test_rime_supported_capabilities():
    caps = get_capabilities("rime")
    assert caps.supports_streaming is True
    assert caps.supports_speaking_rate is True


def test_rime_unsupported_expressive_controls_remain_unsupported():
    caps = get_capabilities("rime")
    assert caps.supports_stability_control is False
    assert caps.supports_pitch is False
    assert caps.supports_ssml is False
    assert caps.supports_native_expressive_tags is False
    assert caps.supports_pause_control is False


def test_rime_overlay_stays_empty_even_with_stability_hint():
    decision = analyze_response(
        "Got it, one moment.",
        user_text="This is unacceptable, I want a refund now",
    )
    overlay = build_voice_settings_overlay("rime", decision)
    assert overlay == {}


# ---------------------------------------------------------------------------
# General / fallback safety
# ---------------------------------------------------------------------------


def test_unknown_provider_returns_all_false_capabilities():
    caps = get_capabilities("some_future_provider")
    assert caps.provider_slug == "unknown"
    assert caps.supports_streaming is False
    assert caps.supports_stability_control is False


def test_missing_provider_slug_does_not_raise():
    caps = get_capabilities(None)
    assert caps.provider_slug == "unknown"
    overlay = build_voice_settings_overlay(None, None)
    assert overlay == {}


def test_missing_decision_returns_empty_overlay_safely():
    overlay = build_voice_settings_overlay("elevenlabs", None)
    assert overlay == {}


def test_provider_slug_lookup_is_case_and_whitespace_insensitive():
    a = get_capabilities("ElevenLabs")
    b = get_capabilities("  elevenlabs  ")
    c = get_capabilities("elevenlabs")
    assert a == b == c


# ---------------------------------------------------------------------------
# Local stubs (avoid depending on internal HumanizationDecision defaults)
# ---------------------------------------------------------------------------


def decision_mood():
    from app.voice.turn_signals import UserMood

    return UserMood.NEUTRAL


def pacing_stub():
    from app.voice.humanization_engine import PacingHint

    return PacingHint()


def ack_stub():
    from app.voice.humanization_engine import AcknowledgementHint

    return AcknowledgementHint()


def filler_stub():
    from app.voice.humanization_engine import FillerHint

    return FillerHint()
