"""
V-08: LLM-driven TTS delivery humanization ("Option C").

Covers:
- app.voice.humanization_intent (pure parsing/validation, wire-format
  extraction/stripping)
- app.voice.humanization_engine's llm_intent/guardrail_state extension to
  analyze_response (regression contract when the flag is off, guardrail
  cooldown/ceiling enforcement when it's on)
- app.voice.tts_provider_capabilities' provider realization
  (apply_vocal_behavior_tag, apply_emphasis_word, build_voice_settings_overlay)
- that both transports import and call the SAME shared functions (no
  duplicated/divergent parsing or prompt-building logic)
"""

import app.voice.humanization_engine as humanization_engine
from app.voice.humanization_engine import (
    VocalBehaviorGuardrailState,
    analyze_response,
    pause_frames_for_chunk,
)
from app.voice.humanization_intent import (
    DeliveryEmotion,
    PauseCategory,
    SegmentIntent,
    VocalBehavior,
    build_delivery_prompt_block,
    build_segment_intent,
    consume_delivery_tag,
    extract_pending_delivery_tag,
    parse_segment_intent_json,
    segment_intent_from_tag_attrs,
    strip_delivery_tags,
)
from app.voice.tts_provider_capabilities import (
    apply_emphasis_word,
    apply_vocal_behavior_tag,
    build_voice_settings_overlay,
)

# ---------------------------------------------------------------------------
# 1. Pure parsing/validation (app.voice.humanization_intent)
# ---------------------------------------------------------------------------


def test_valid_segment_intent_parses_correctly():
    intent = build_segment_intent(
        "Sure, I can help with tomorrow.",
        emotion="warm",
        vocal_behavior="soft_chuckle",
        pause_after="breath",
        emphasis_word="tomorrow",
        confidence=0.9,
    )
    assert intent.text == "Sure, I can help with tomorrow."
    assert intent.emotion == DeliveryEmotion.WARM
    assert intent.vocal_behavior == VocalBehavior.SOFT_CHUCKLE
    assert intent.pause_after == PauseCategory.BREATH
    assert intent.emphasis_word == "tomorrow"
    assert intent.confidence == 0.9


def test_malformed_json_falls_back_to_plain_text_with_neutral_defaults():
    intent = parse_segment_intent_json(
        "{not valid json!!", fallback_text="Hello there."
    )
    assert intent.text == "Hello there."
    assert intent.emotion == DeliveryEmotion.NEUTRAL
    assert intent.vocal_behavior == VocalBehavior.NONE
    assert intent.pause_after == PauseCategory.NONE
    assert intent.emphasis_word is None


def test_json_non_dict_payload_falls_back_safely():
    intent = parse_segment_intent_json("[1, 2, 3]", fallback_text="Fallback text.")
    assert intent.text == "Fallback text."
    assert intent.emotion == DeliveryEmotion.NEUTRAL


def test_missing_text_in_json_uses_fallback_text():
    intent = parse_segment_intent_json(
        '{"emotion": "warm"}', fallback_text="Real text."
    )
    assert intent.text == "Real text."
    assert intent.emotion == DeliveryEmotion.WARM


def test_unknown_enum_values_fall_back_to_neutral_none_never_raise():
    intent = build_segment_intent(
        "Some response text.",
        emotion="ecstatic",  # not a real DeliveryEmotion
        vocal_behavior="cackle",  # not a real VocalBehavior
        pause_after="dramatic",  # not a real PauseCategory
    )
    assert intent.emotion == DeliveryEmotion.NEUTRAL
    assert intent.vocal_behavior == VocalBehavior.NONE
    assert intent.pause_after == PauseCategory.NONE
    # Text is still preserved even though every enum was garbage.
    assert intent.text == "Some response text."


def test_empty_text_collapses_whole_segment_to_neutral_defaults():
    intent = build_segment_intent(
        "",
        emotion="warm",
        vocal_behavior="soft_chuckle",
        pause_after="breath",
        emphasis_word="hi",
    )
    assert intent.text == ""
    assert intent.emotion == DeliveryEmotion.NEUTRAL
    assert intent.vocal_behavior == VocalBehavior.NONE
    assert intent.pause_after == PauseCategory.NONE
    assert intent.emphasis_word is None


def test_none_text_never_raises():
    intent = build_segment_intent(None, emotion="warm")  # type: ignore[arg-type]
    assert intent.text == ""
    assert intent.emotion == DeliveryEmotion.NEUTRAL


def test_emphasis_word_not_substring_gets_dropped_not_failed():
    intent = build_segment_intent(
        "Your appointment is confirmed for tomorrow.",
        emphasis_word="Wednesday",  # not present in the text
    )
    assert intent.emphasis_word is None
    # The rest of the segment is untouched — dropping emphasis never fails
    # the whole segment.
    assert intent.text == "Your appointment is confirmed for tomorrow."


def test_emphasis_word_case_insensitive_substring_match_is_kept():
    intent = build_segment_intent(
        "Your appointment is confirmed for Tomorrow.",
        emphasis_word="tomorrow",
    )
    assert intent.emphasis_word == "tomorrow"


def test_at_most_one_vocal_behavior_enforced_by_enum_shape():
    # SegmentIntent.vocal_behavior is a single enum value, not a list — the
    # schema itself makes "more than one per segment" structurally
    # impossible to represent.
    intent = SegmentIntent(text="hi")
    assert isinstance(intent.vocal_behavior, VocalBehavior)


# ---------------------------------------------------------------------------
# Inline [DELIVERY ...] wire-format extraction
# ---------------------------------------------------------------------------


def test_extract_complete_delivery_tag_strips_it_from_buffer():
    buf = '[DELIVERY emotion=warm behavior=soft_chuckle pause=breath emphasis="tomorrow"] Sure thing.'
    attrs, cleaned = extract_pending_delivery_tag(buf)
    assert attrs == {
        "emotion": "warm",
        "behavior": "soft_chuckle",
        "pause": "breath",
        "emphasis": "tomorrow",
    }
    assert cleaned == " Sure thing."
    assert "[DELIVERY" not in cleaned


def test_extract_incomplete_delivery_tag_leaves_buffer_untouched():
    buf = "[DELIVERY emotion=warm behavior=soft"  # never closed yet
    attrs, cleaned = extract_pending_delivery_tag(buf)
    assert attrs is None
    assert cleaned == buf


def test_extract_no_tag_present_is_a_noop():
    buf = "Just a normal sentence with no tag."
    attrs, cleaned = extract_pending_delivery_tag(buf)
    assert attrs is None
    assert cleaned == buf


def test_consume_delivery_tag_behaves_identically_to_the_underlying_extractor():
    # consume_delivery_tag is the transport-facing entry point — a thin
    # alias, not a re-derivation of the parsing logic.
    buf = "[DELIVERY emotion=warm] Hello."
    assert consume_delivery_tag(buf) == extract_pending_delivery_tag(buf)


def test_segment_intent_from_tag_attrs_maps_wire_format_field_names():
    intent = segment_intent_from_tag_attrs(
        "Let's get that fixed.",
        {
            "emotion": "calm",
            "behavior": "brief_sigh",
            "pause": "thinking",
            "emphasis": "fixed",
        },
    )
    assert intent.emotion == DeliveryEmotion.CALM
    assert intent.vocal_behavior == VocalBehavior.BRIEF_SIGH
    assert intent.pause_after == PauseCategory.THINKING
    assert intent.emphasis_word == "fixed"


def test_segment_intent_from_tag_attrs_handles_none_attrs():
    intent = segment_intent_from_tag_attrs("Just speaking normally.", None)
    assert intent.emotion == DeliveryEmotion.NEUTRAL
    assert intent.vocal_behavior == VocalBehavior.NONE


def test_extract_pending_delivery_tag_drains_all_complete_tags_in_one_call():
    """
    Code-review finding #2: two complete tags landing in the SAME chunk must
    BOTH be stripped by a single extraction call, not just the first.
    """
    buf = (
        "[DELIVERY emotion=warm behavior=none pause=none] Sure! "
        "[DELIVERY emotion=neutral behavior=none pause=none] Let me check your account."
    )
    attrs, cleaned = extract_pending_delivery_tag(buf)
    assert "[DELIVERY" not in cleaned.upper()
    assert cleaned == " Sure!  Let me check your account."
    # The LAST (most recent) tag's attrs win — it's the one closest to the
    # not-yet-flushed text.
    assert attrs == {"emotion": "neutral", "behavior": "none", "pause": "none"}


def test_extract_pending_delivery_tag_three_tags_all_stripped():
    buf = "[DELIVERY emotion=warm]A[DELIVERY emotion=calm]B[DELIVERY emotion=upbeat]C"
    attrs, cleaned = extract_pending_delivery_tag(buf)
    assert cleaned == "ABC"
    assert attrs == {"emotion": "upbeat"}


def test_consume_delivery_tag_also_drains_multiple_tags_in_one_call():
    buf = "[DELIVERY emotion=warm]Hi. [DELIVERY emotion=calm]Bye."
    attrs, cleaned = consume_delivery_tag(buf)
    assert "[DELIVERY" not in cleaned.upper()
    assert attrs == {"emotion": "calm"}


def test_strip_delivery_tags_handles_dangling_unclosed_tag_at_end_of_stream():
    """
    Code-review finding #1: an unclosed/truncated tag at end-of-stream
    (e.g. max_tokens cutoff, or the model never closing the bracket) must
    never survive strip_delivery_tags — the literal brackets/attrs must
    never reach spoken text or the transcript.
    """
    buf = "Sure, here is the info. [DELIVERY emotion=warm behavior=soft_chuckle pause=breath"
    cleaned = strip_delivery_tags(buf)
    assert "[DELIVERY" not in cleaned.upper()
    assert "emotion=warm" not in cleaned
    assert cleaned.strip() == "Sure, here is the info."


def test_strip_delivery_tags_handles_dangling_tag_with_no_preceding_text():
    buf = "[DELIVERY emotion="
    cleaned = strip_delivery_tags(buf)
    assert "[DELIVERY" not in cleaned.upper()


def test_strip_delivery_tags_handles_both_complete_and_trailing_dangling_tag():
    buf = "[DELIVERY emotion=warm] Sure thing. [DELIVERY pause=brea"
    cleaned = strip_delivery_tags(buf)
    assert "[DELIVERY" not in cleaned.upper()
    assert "Sure thing." in cleaned


def test_extract_pending_delivery_tag_leaves_a_still_incomplete_trailing_tag_alone():
    # Sanity check: the STREAMING extractor must still NOT strip a genuinely
    # incomplete tag (mid-stream, more chunks expected) — only
    # strip_delivery_tags (the end-of-stream/defense-in-depth cleanup) treats
    # an unclosed tag as dangling garbage.
    buf = "Sure, here is the info. [DELIVERY emotion=warm behavior=soft"
    attrs, cleaned = extract_pending_delivery_tag(buf)
    assert attrs is None
    assert cleaned == buf


def test_booking_mixin_strip_control_tokens_strips_dangling_delivery_tag():
    """
    End-to-end safety net for the Twilio transport: BookingMixin's shared
    `_strip_control_tokens_for_tts` chokepoint (used by `_prepare_tts_text`
    and every early-detection call site in bidirectional_stream.py) must
    never let a dangling/unclosed [DELIVERY ...] fragment reach spoken text.
    """
    from app.voice.booking_mixin import BookingMixin

    truncated = "Sure, here is the info. [DELIVERY emotion=warm behavior=soft_chuckle pause=breath"
    cleaned = BookingMixin._strip_control_tokens_for_tts(truncated)
    assert "[DELIVERY" not in cleaned.upper()
    assert "Sure, here is the info." in cleaned


def test_booking_mixin_strip_control_tokens_strips_multiple_complete_tags():
    from app.voice.booking_mixin import BookingMixin

    two_tags = (
        "[DELIVERY emotion=warm behavior=none pause=none] Sure! "
        "[DELIVERY emotion=neutral behavior=none pause=none] Let me check your account."
    )
    cleaned = BookingMixin._strip_control_tokens_for_tts(two_tags)
    assert "[DELIVERY" not in cleaned.upper()
    assert "Sure!" in cleaned
    assert "Let me check your account." in cleaned


def test_strip_delivery_tags_removes_all_occurrences():
    text = "[DELIVERY emotion=warm] Hi there. [DELIVERY pause=breath] More text."
    cleaned = strip_delivery_tags(text)
    assert "[DELIVERY" not in cleaned
    assert "Hi there." in cleaned and "More text." in cleaned


def test_delivery_prompt_block_empty_when_disabled():
    assert build_delivery_prompt_block(enabled=False) == ""


def test_delivery_prompt_block_present_when_enabled():
    block = build_delivery_prompt_block(enabled=True)
    assert block.startswith("# DELIVERY")
    assert "emotion=" in block
    # Semantic enums only — never a provider-specific tag or a millisecond value.
    assert "chuckles" not in block
    assert "ms" not in block


def test_delivery_prompt_block_reads_settings_flag_by_default(monkeypatch):
    monkeypatch.setattr(
        humanization_engine.settings, "VOICE_ENABLE_LLM_HUMANIZATION", True
    )
    import app.voice.humanization_intent as hi

    monkeypatch.setattr(hi.settings, "VOICE_ENABLE_LLM_HUMANIZATION", True)
    assert build_delivery_prompt_block() != ""
    monkeypatch.setattr(hi.settings, "VOICE_ENABLE_LLM_HUMANIZATION", False)
    assert build_delivery_prompt_block() == ""


# ---------------------------------------------------------------------------
# 2. Deterministic policy layer (app.voice.humanization_engine.analyze_response)
# ---------------------------------------------------------------------------


def test_flag_off_analyze_response_ignores_llm_intent_entirely(monkeypatch):
    monkeypatch.setattr(
        humanization_engine.settings, "VOICE_ENABLE_LLM_HUMANIZATION", False
    )
    intent = build_segment_intent(
        "Sure, I can help with that.",
        emotion="upbeat",
        vocal_behavior="soft_chuckle",
        pause_after="breath",
    )
    without = analyze_response("Sure, I can help with that.")
    with_intent = analyze_response("Sure, I can help with that.", llm_intent=intent)
    # Regression contract: every pre-existing field is byte-identical
    # regardless of whether an llm_intent was passed, whenever the flag is
    # off — and `delivery` is always None.
    assert with_intent.text == without.text
    assert with_intent.mood == without.mood
    assert with_intent.response_emotion == without.response_emotion
    assert with_intent.pacing == without.pacing
    assert with_intent.acknowledgement == without.acknowledgement
    assert with_intent.filler == without.filler
    assert with_intent.tts_stability_hint == without.tts_stability_hint
    assert with_intent.metadata == without.metadata
    assert with_intent.delivery is None
    assert without.delivery is None


def test_flag_off_no_llm_intent_passed_is_also_unaffected(monkeypatch):
    monkeypatch.setattr(
        humanization_engine.settings, "VOICE_ENABLE_LLM_HUMANIZATION", False
    )
    d = analyze_response("Your appointment is confirmed for tomorrow at noon.")
    assert d.delivery is None


def test_neutral_factual_response_stays_neutral_even_with_flag_on(monkeypatch):
    """The most important 'don't over-humanize' case: a neutral delivery
    intent (the LLM's default / most-common output) must not introduce any
    vocal_behavior/pause/emotion effect."""
    monkeypatch.setattr(
        humanization_engine.settings, "VOICE_ENABLE_LLM_HUMANIZATION", True
    )
    intent = build_segment_intent("Your balance is $42.50.")
    d = analyze_response(
        "Your balance is $42.50.",
        llm_intent=intent,
        guardrail_state=VocalBehaviorGuardrailState(),
    )
    assert d.delivery is not None
    assert d.delivery.emotion == DeliveryEmotion.NEUTRAL
    assert d.delivery.vocal_behavior == VocalBehavior.NONE
    assert d.delivery.pause_after == PauseCategory.NONE


def test_numeric_factual_utterance_never_reaches_for_an_effect(monkeypatch):
    monkeypatch.setattr(
        humanization_engine.settings, "VOICE_ENABLE_LLM_HUMANIZATION", True
    )
    intent = build_segment_intent("That will be 3 units at 19 dollars each.")
    d = analyze_response(
        "That will be 3 units at 19 dollars each.",
        llm_intent=intent,
        guardrail_state=VocalBehaviorGuardrailState(),
    )
    assert d.delivery.vocal_behavior == VocalBehavior.NONE


def test_short_acknowledgement_never_reaches_for_an_effect(monkeypatch):
    monkeypatch.setattr(
        humanization_engine.settings, "VOICE_ENABLE_LLM_HUMANIZATION", True
    )
    intent = build_segment_intent("Got it.")
    d = analyze_response(
        "Got it.", llm_intent=intent, guardrail_state=VocalBehaviorGuardrailState()
    )
    assert d.delivery.vocal_behavior == VocalBehavior.NONE


def test_valid_llm_delivery_intent_flows_through_to_decision(monkeypatch):
    monkeypatch.setattr(
        humanization_engine.settings, "VOICE_ENABLE_LLM_HUMANIZATION", True
    )
    intent = build_segment_intent(
        "Oh, that's great to hear!",
        emotion="upbeat",
        pause_after="emphasis",
    )
    d = analyze_response(
        "Oh, that's great to hear!",
        llm_intent=intent,
        guardrail_state=VocalBehaviorGuardrailState(),
    )
    assert d.delivery.emotion == DeliveryEmotion.UPBEAT
    assert d.delivery.pause_after == PauseCategory.EMPHASIS


def test_malformed_llm_intent_object_never_blocks_response_text(monkeypatch):
    monkeypatch.setattr(
        humanization_engine.settings, "VOICE_ENABLE_LLM_HUMANIZATION", True
    )

    class _Boom:
        vocal_behavior = "not-a-real-enum-member"

    d = analyze_response(
        "This response must still be spoken.",
        llm_intent=_Boom(),  # type: ignore[arg-type]
        guardrail_state=VocalBehaviorGuardrailState(),
    )
    # The engine degrades `delivery` to None on any unexpected error but
    # NEVER drops/blocks the actual response text.
    assert d.text == "This response must still be spoken."
    assert d.delivery is None


def test_cooldown_guardrail_suppresses_repeat_vocal_behavior():
    state = VocalBehaviorGuardrailState()
    intent = build_segment_intent(
        "Ha, that's a funny one.", vocal_behavior="soft_chuckle"
    )

    import app.voice.humanization_engine as he

    d1 = he._apply_delivery_guardrails(intent, state)
    assert d1.vocal_behavior == VocalBehavior.SOFT_CHUCKLE

    # Same behavior requested again immediately after — cooldown must
    # suppress it (never two chuckles back to back).
    d2 = he._apply_delivery_guardrails(intent, state)
    assert d2.vocal_behavior == VocalBehavior.NONE


def test_cooldown_expires_after_enough_segments():
    state = VocalBehaviorGuardrailState()
    intent = build_segment_intent("Ha, funny.", vocal_behavior="soft_chuckle")
    neutral = build_segment_intent("Neutral segment.")

    import app.voice.humanization_engine as he

    first = he._apply_delivery_guardrails(intent, state)
    assert first.vocal_behavior == VocalBehavior.SOFT_CHUCKLE
    for _ in range(he._VOCAL_BEHAVIOR_COOLDOWN_SEGMENTS):
        he._apply_delivery_guardrails(neutral, state)
    later = he._apply_delivery_guardrails(intent, state)
    assert later.vocal_behavior == VocalBehavior.SOFT_CHUCKLE


def test_call_ceiling_guardrail_caps_total_vocal_behaviors():
    import app.voice.humanization_engine as he

    state = VocalBehaviorGuardrailState()
    neutral = build_segment_intent("Neutral segment.")
    used = 0
    for _ in range(he._VOCAL_BEHAVIOR_CALL_CEILING + 3):
        intent = build_segment_intent("Ha.", vocal_behavior="soft_chuckle")
        result = he._apply_delivery_guardrails(intent, state)
        if result.vocal_behavior != VocalBehavior.NONE:
            used += 1
        # Burn through the cooldown between attempts.
        for _ in range(he._VOCAL_BEHAVIOR_COOLDOWN_SEGMENTS):
            he._apply_delivery_guardrails(neutral, state)
    assert used == he._VOCAL_BEHAVIOR_CALL_CEILING


def test_guardrails_never_raise_on_unexpected_state():
    import app.voice.humanization_engine as he

    intent = build_segment_intent("Ha.", vocal_behavior="soft_chuckle")
    # A guardrail_state=None input must degrade gracefully, not raise.
    result = he._apply_delivery_guardrails(intent, None)
    assert result.vocal_behavior == VocalBehavior.SOFT_CHUCKLE


def test_pause_after_category_overrides_pacing_when_flag_on(monkeypatch):
    monkeypatch.setattr(
        humanization_engine.settings, "VOICE_ENABLE_LLM_HUMANIZATION", True
    )
    frames = pause_frames_for_chunk(
        None, is_final=False, pause_after=PauseCategory.THINKING
    )
    assert frames == humanization_engine._PAUSE_CATEGORY_FRAMES[PauseCategory.THINKING]


def test_pause_after_ignored_when_flag_off(monkeypatch):
    monkeypatch.setattr(
        humanization_engine.settings, "VOICE_ENABLE_LLM_HUMANIZATION", False
    )
    # pause_after is non-NONE, but the flag is off — must fall back to the
    # pacing-only path (None pacing => 0), never the category frame count.
    frames = pause_frames_for_chunk(
        None, is_final=False, pause_after=PauseCategory.THINKING
    )
    assert frames == 0


def test_pause_after_never_applies_on_final_chunk(monkeypatch):
    monkeypatch.setattr(
        humanization_engine.settings, "VOICE_ENABLE_LLM_HUMANIZATION", True
    )
    frames = pause_frames_for_chunk(
        None, is_final=True, pause_after=PauseCategory.THINKING
    )
    assert frames == 0


def test_pause_frames_for_chunk_default_none_matches_pre_v08_signature():
    # Calling with the old 2-positional-arg signature must still work
    # exactly as before (pacing=None, is_final=True => 0).
    assert pause_frames_for_chunk(None, True) == 0


# ---------------------------------------------------------------------------
# 3. Provider realization (app.voice.tts_provider_capabilities)
# ---------------------------------------------------------------------------


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


def test_elevenlabs_vocal_behavior_realized_as_native_bracket_tag():
    intent = build_segment_intent("Sure, happy to help.", vocal_behavior="soft_chuckle")
    decision = _decision_with_delivery(intent)
    out = apply_vocal_behavior_tag("Sure, happy to help.", "elevenlabs", decision)
    assert out == "[chuckles] Sure, happy to help."


def test_google_vocal_behavior_is_a_noop_never_simulated_with_text():
    intent = build_segment_intent("Sure, happy to help.", vocal_behavior="soft_chuckle")
    decision = _decision_with_delivery(intent)
    out = apply_vocal_behavior_tag("Sure, happy to help.", "google", decision)
    assert out == "Sure, happy to help."
    assert "*" not in out and "chuckle" not in out.lower()


def test_rime_vocal_behavior_is_a_noop():
    intent = build_segment_intent("One moment please.", vocal_behavior="brief_sigh")
    decision = _decision_with_delivery(intent)
    out = apply_vocal_behavior_tag("One moment please.", "rime", decision)
    assert out == "One moment please."


def test_hume_and_xai_vocal_behavior_is_a_noop():
    intent = build_segment_intent("Let me check.", vocal_behavior="hesitation")
    decision = _decision_with_delivery(intent)
    assert (
        apply_vocal_behavior_tag("Let me check.", "hume", decision) == "Let me check."
    )
    assert apply_vocal_behavior_tag("Let me check.", "xai", decision) == "Let me check."


def test_vocal_behavior_none_is_always_a_noop():
    intent = build_segment_intent("Neutral text.")
    decision = _decision_with_delivery(intent)
    assert (
        apply_vocal_behavior_tag("Neutral text.", "elevenlabs", decision)
        == "Neutral text."
    )


def test_vocal_behavior_tag_no_decision_is_a_noop():
    assert apply_vocal_behavior_tag("Some text.", "elevenlabs", None) == "Some text."


def test_emphasis_word_realization_is_a_documented_noop_everywhere():
    intent = build_segment_intent("Confirmed for tomorrow.", emphasis_word="tomorrow")
    decision = _decision_with_delivery(intent)
    for provider in ("elevenlabs", "google", "rime", "hume", "xai", None):
        assert apply_emphasis_word("Confirmed for tomorrow.", provider, decision) == (
            "Confirmed for tomorrow."
        )


def test_elevenlabs_stability_overlay_uses_llm_emotion_when_present():
    intent = build_segment_intent("Let's fix that for you.", emotion="calm")
    decision = _decision_with_delivery(intent)
    overlay = build_voice_settings_overlay("elevenlabs", decision)
    from app.voice.tts_provider_capabilities import _ELEVEN_STABILITY_BY_EMOTION

    assert overlay["stability"] == _ELEVEN_STABILITY_BY_EMOTION[DeliveryEmotion.CALM]


def test_elevenlabs_stability_falls_back_to_caller_mood_hint_when_delivery_neutral():
    decision = analyze_response(
        "I understand, let's get that fixed for you.",
        user_text="This is unacceptable, I want a refund now",
    )
    overlay = build_voice_settings_overlay("elevenlabs", decision)
    assert overlay == {"stability": decision.tts_stability_hint}


def test_google_speaking_rate_uses_llm_emotion_when_present():
    intent = build_segment_intent("That's wonderful news!", emotion="upbeat")
    decision = _decision_with_delivery(intent)
    overlay = build_voice_settings_overlay("google", decision)
    from app.voice.tts_provider_capabilities import _SPEAKING_RATE_BY_EMOTION

    assert overlay["speaking_rate"] == _SPEAKING_RATE_BY_EMOTION[DeliveryEmotion.UPBEAT]


def test_google_speaking_rate_falls_back_to_response_emotion_heuristic():
    # No delivery at all (pre-V-08 shape) — must reproduce the exact
    # previously-duplicated inline happy/sad/uncertain/confident mapping.
    decision = analyze_response("Amazing, I love it, that's great!")
    assert decision.response_emotion == "happy"
    overlay = build_voice_settings_overlay("google", decision)
    assert overlay["speaking_rate"] == 1.03


def test_google_speaking_rate_neutral_response_emotion_omits_key():
    decision = analyze_response("The appointment is at 3pm.")
    assert decision.response_emotion == "neutral"
    overlay = build_voice_settings_overlay("google", decision)
    assert "speaking_rate" not in overlay


# ---------------------------------------------------------------------------
# 9. Both transports call the SAME shared functions (no duplicated logic)
# ---------------------------------------------------------------------------


def test_both_transports_import_the_identical_shared_functions():
    import app.routers.bidirectional_stream as twilio_mod
    import app.voice.conversation_orchestrator as browser_mod

    assert twilio_mod.consume_delivery_tag is browser_mod.consume_delivery_tag
    assert twilio_mod.consume_delivery_tag is consume_delivery_tag
    assert (
        twilio_mod.segment_intent_from_tag_attrs
        is browser_mod.segment_intent_from_tag_attrs
    )
    assert twilio_mod.segment_intent_from_tag_attrs is segment_intent_from_tag_attrs
    assert (
        twilio_mod.build_delivery_prompt_block
        is browser_mod.build_delivery_prompt_block
    )
    assert twilio_mod.build_delivery_prompt_block is build_delivery_prompt_block
    assert twilio_mod.strip_delivery_tags is browser_mod.strip_delivery_tags


def test_both_transports_produce_identical_policy_decision_for_equivalent_input(
    monkeypatch,
):
    """
    Proves no duplicated/divergent per-transport logic was introduced: given
    the same raw tag attrs + segment text + guardrail state, the SAME shared
    parse -> policy pipeline both transports call produces the SAME decision
    regardless of which transport's imported symbols are used.
    """
    monkeypatch.setattr(
        humanization_engine.settings, "VOICE_ENABLE_LLM_HUMANIZATION", True
    )
    import app.routers.bidirectional_stream as twilio_mod
    import app.voice.conversation_orchestrator as browser_mod

    raw_attrs = {"emotion": "warm", "behavior": "soft_chuckle", "pause": "breath"}
    text = "Sure, happy to help with that."

    twilio_intent = twilio_mod.segment_intent_from_tag_attrs(text, raw_attrs)
    browser_intent = browser_mod.segment_intent_from_tag_attrs(text, raw_attrs)
    assert twilio_intent == browser_intent

    state_a = VocalBehaviorGuardrailState()
    state_b = VocalBehaviorGuardrailState()
    decision_a = analyze_response(
        text, llm_intent=twilio_intent, guardrail_state=state_a
    )
    decision_b = analyze_response(
        text, llm_intent=browser_intent, guardrail_state=state_b
    )
    assert decision_a.delivery == decision_b.delivery
    assert decision_a.text == decision_b.text
