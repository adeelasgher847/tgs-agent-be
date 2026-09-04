"""
V-08 acceptance criterion: adding a new TTS provider must require ONLY

  (a) a new `TTSProviderCapabilities` entry in
      `app.voice.tts_provider_capabilities._CAPABILITIES` declaring what it
      supports, and
  (b) provider-specific realization inside that same module's existing
      capability-gated functions (`build_voice_settings_overlay`,
      `apply_vocal_behavior_tag`, `apply_emphasis_word`)

with ZERO changes to `app.voice.humanization_intent` (semantic schema/LLM
wire format), `app.voice.humanization_engine` (policy/guardrails), or
either orchestrator (`app/routers/bidirectional_stream.py`,
`app/voice/conversation_orchestrator.py`).

This test proves that by registering a dummy provider purely via
monkeypatching `tts_provider_capabilities._CAPABILITIES` (never touching
production code, never adding a permanent fake provider) and then running
the existing, unmodified public functions of `humanization_intent`/
`humanization_engine`/`tts_provider_capabilities` against it — the same
functions every real provider already goes through.
"""

from __future__ import annotations

import app.voice.tts_provider_capabilities as caps_mod
from app.voice.humanization_engine import analyze_response
from app.voice.humanization_intent import build_segment_intent
from app.voice.tts_provider_capabilities import (
    TTSProviderCapabilities,
    apply_emphasis_word,
    apply_vocal_behavior_tag,
    build_voice_settings_overlay,
    get_capabilities,
)

_DUMMY_SLUG = "dummyprovider2099"


def _register_dummy_provider(monkeypatch, **capability_overrides) -> None:
    """Temporarily register a fake provider's capabilities for one test,
    via monkeypatch only — never mutates the real `_CAPABILITIES` dict
    beyond the test's lifetime."""
    base = dict(
        provider_slug=_DUMMY_SLUG,
        supports_streaming=True,
        supports_ssml=False,
        supports_speaking_rate=False,
        supports_pitch=False,
        supports_stability_control=False,
        supports_native_expressive_tags=False,
        supports_pause_control=False,
        supports_streaming_session=False,
    )
    base.update(capability_overrides)
    dummy_caps = TTSProviderCapabilities(**base)

    patched = dict(caps_mod._CAPABILITIES)
    patched[_DUMMY_SLUG] = dummy_caps
    monkeypatch.setattr(caps_mod, "_CAPABILITIES", patched)


def _decision_with_delivery(delivery):
    from app.voice.humanization_engine import (
        AcknowledgementHint,
        FillerHint,
        HumanizationDecision,
        PacingHint,
    )
    from app.voice.turn_signals import UserMood

    return HumanizationDecision(
        text=delivery.text if delivery else "hi",
        mood=UserMood.NEUTRAL,
        response_emotion="neutral",
        pacing=PacingHint(),
        acknowledgement=AcknowledgementHint(),
        filler=FillerHint(),
        tts_stability_hint=None,
        delivery=delivery,
    )


def test_dummy_provider_lookup_returns_registered_capabilities(monkeypatch):
    _register_dummy_provider(monkeypatch, supports_speaking_rate=True)
    caps = get_capabilities(_DUMMY_SLUG)
    assert caps.provider_slug == _DUMMY_SLUG
    assert caps.supports_speaking_rate is True
    assert caps.supports_stability_control is False


def test_dummy_provider_with_rate_capability_gets_rate_overlay_not_stability(monkeypatch):
    _register_dummy_provider(
        monkeypatch, supports_speaking_rate=True, supports_stability_control=False
    )
    intent = build_segment_intent("That's wonderful news!", emotion="upbeat")
    decision = _decision_with_delivery(intent)

    overlay = build_voice_settings_overlay(_DUMMY_SLUG, decision)

    from app.voice.tts_provider_capabilities import _SPEAKING_RATE_BY_EMOTION
    from app.voice.humanization_intent import DeliveryEmotion

    assert overlay == {"speaking_rate": _SPEAKING_RATE_BY_EMOTION[DeliveryEmotion.UPBEAT]}
    assert "stability" not in overlay


def test_dummy_provider_with_stability_capability_gets_stability_overlay(monkeypatch):
    _register_dummy_provider(
        monkeypatch, supports_speaking_rate=True, supports_stability_control=True
    )
    intent = build_segment_intent("Let's fix that for you.", emotion="calm")
    decision = _decision_with_delivery(intent)

    overlay = build_voice_settings_overlay(_DUMMY_SLUG, decision)

    from app.voice.tts_provider_capabilities import _ELEVEN_STABILITY_BY_EMOTION
    from app.voice.humanization_intent import DeliveryEmotion

    # Capability priority (stability wins over rate when both are declared)
    # is decided purely by build_voice_settings_overlay's existing
    # capability-gated branching — unchanged by adding this provider.
    assert overlay == {"stability": _ELEVEN_STABILITY_BY_EMOTION[DeliveryEmotion.CALM]}


def test_dummy_provider_with_no_expressive_capability_is_a_noop(monkeypatch):
    _register_dummy_provider(monkeypatch)  # every capability False except streaming
    intent = build_segment_intent("Sure, happy to help.", vocal_behavior="soft_chuckle")
    decision = _decision_with_delivery(intent)

    out = apply_vocal_behavior_tag("Sure, happy to help.", _DUMMY_SLUG, decision)

    assert out == "Sure, happy to help."
    assert "chuckle" not in out.lower()


def test_dummy_provider_with_native_expressive_tags_gets_realized_tag(monkeypatch):
    _register_dummy_provider(monkeypatch, supports_native_expressive_tags=True)
    intent = build_segment_intent("One moment please.", vocal_behavior="brief_sigh")
    decision = _decision_with_delivery(intent)

    out = apply_vocal_behavior_tag("One moment please.", _DUMMY_SLUG, decision)

    # Realizes using the same shared tag vocabulary every native-tag
    # provider (currently only ElevenLabs) uses — no new tag vocabulary
    # invented for this provider.
    assert out == "[sighs] One moment please."


def test_dummy_provider_emphasis_word_stays_the_documented_noop(monkeypatch):
    _register_dummy_provider(monkeypatch, supports_speaking_rate=True)
    intent = build_segment_intent("Confirmed for tomorrow.", emphasis_word="tomorrow")
    decision = _decision_with_delivery(intent)

    assert (
        apply_emphasis_word("Confirmed for tomorrow.", _DUMMY_SLUG, decision)
        == "Confirmed for tomorrow."
    )


def test_dummy_provider_never_requires_touching_semantic_or_policy_layers(monkeypatch):
    """
    End-to-end proof: registering the dummy provider (capability layer only)
    and then driving it through the UNMODIFIED, provider-agnostic
    humanization_intent + humanization_engine public API — the exact same
    call every real provider goes through — produces a normal, safe
    HumanizationDecision. Nothing here imports or references provider
    names inside humanization_intent.py or humanization_engine.py; the
    provider identity only ever reaches tts_provider_capabilities.py.
    """
    _register_dummy_provider(monkeypatch, supports_speaking_rate=True)

    decision = analyze_response(
        "Sure, I can help with that.",
        user_text="Can you help me book an appointment?",
    )
    overlay = build_voice_settings_overlay(_DUMMY_SLUG, decision)

    # A pre-V-08-shaped decision (no llm_intent) with a neutral response
    # still resolves cleanly against the new provider via the same
    # response_emotion fallback path every other rate-based provider uses.
    assert isinstance(overlay, dict)
    assert apply_vocal_behavior_tag(decision.text, _DUMMY_SLUG, decision) == decision.text
