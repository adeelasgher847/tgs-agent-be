from app.voice.tone_adapter import tone_adapter
from app.voice.turn_signals import (
    UserMood,
    build_turn_context,
    build_user_signals_block,
    detect_mood,
)


def test_detect_mood_urgent():
    assert detect_mood("This is an emergency, please help", 0.9) == UserMood.URGENT


def test_detect_mood_angry():
    assert detect_mood("I am so angry and frustrated with this", 0.8) == UserMood.ANGRY


def test_detect_mood_frustrated():
    assert detect_mood("This is not working at all", 0.7) == UserMood.FRUSTRATED


def test_detect_mood_happy():
    assert detect_mood("Thank you so much, that is great", 0.9) == UserMood.HAPPY


def test_build_turn_context_booking_phase():
    ctx = build_turn_context("I want to book tomorrow", 0.9, booking_context_active=True)
    assert ctx.conversation_phase == "booking"


def test_build_user_signals_block_contains_mood():
    ctx = build_turn_context("urgent: need help now", 0.85)
    block = build_user_signals_block(ctx)
    assert "USER_SIGNALS" in block
    assert "inferred_mood" in block
    assert UserMood.URGENT.value in block or "urgent" in block


def test_tone_adapter_strips_chipper_leading_when_sad():
    ctx = build_turn_context("I feel so sad", 0.8)
    assert ctx.mood == UserMood.SAD
    out = tone_adapter("Great! I hear you.", ctx, use_ssml=False)
    assert not out.lower().startswith("great!")


def test_tone_adapter_unchanged_for_neutral_hours():
    ctx = build_turn_context("What are your hours?", 0.9)
    text = "We are open nine to five."
    assert tone_adapter(text, ctx, use_ssml=False) == text


def test_tone_adapter_shapes_plain_text_when_ssml_flag_true():
    ctx = build_turn_context("What are your hours?", 0.9)
    out = tone_adapter("Great! I can help with that.", ctx, use_ssml=True)
    assert not out.lower().startswith("great!")


def test_tone_adapter_skips_ssml_markup():
    ctx = build_turn_context("What are your hours?", 0.9)
    ssml = "<speak>Great! I can help with that.</speak>"
    assert tone_adapter(ssml, ctx, use_ssml=True) == ssml


def test_tone_adapter_spoken_rewrites():
    ctx = build_turn_context("What are your hours?", 0.9)
    out = tone_adapter(
        "How may I assist you? I would be happy to check.",
        ctx,
        use_ssml=True,
    )
    assert "How may I assist" not in out
    assert "How can I help" in out
    assert "I'd be happy to" in out


def test_tone_adapter_contracts_i_am():
    ctx = build_turn_context("hello", 0.9)
    out = tone_adapter("I am here to help.", ctx, use_ssml=True)
    assert out.startswith("I'm")
