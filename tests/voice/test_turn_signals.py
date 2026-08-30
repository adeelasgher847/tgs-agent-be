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


def test_tone_adapter_unchanged_for_neutral():
    ctx = build_turn_context("What are your hours?", 0.9)
    text = "We are open nine to five."
    assert tone_adapter(text, ctx, use_ssml=False) == text


# --- TTS stability: one deliberate baseline, not a per-mood switch ---
#
# Previously _tts_stability_for_mood returned one of FOUR distinct values
# keyed off a coarse regex mood classifier, re-applied fresh every turn —
# read as inconsistent/unnatural tone, since the classifier can misfire on
# neutral text (e.g. "cancel"/"refund" trip the frustrated/angry keyword
# lists on a completely calm request). Fixed to one shared baseline for
# neutral/happy/sad (the vast majority of turns), with a single, higher
# ("calmer", not "more dynamic") override reserved for genuinely
# unmistakable caller distress.


def test_stability_shares_one_baseline_for_neutral_happy_and_sad():
    neutral = build_turn_context("What are your hours?", 0.9)
    happy = build_turn_context("Thank you so much, that is great", 0.9)
    sad = build_turn_context("I feel so sad", 0.8)
    assert neutral.mood == UserMood.NEUTRAL
    assert happy.mood == UserMood.HAPPY
    assert sad.mood == UserMood.SAD
    assert neutral.tts_stability_hint == happy.tts_stability_hint == sad.tts_stability_hint


def test_stability_is_higher_not_lower_for_distress_moods():
    """De-escalation calls for a CALMER (higher-stability) agent voice, not
    a more dynamic/unstable one -- the previous behavior had this backwards
    (angry/frustrated/urgent mapped to the LOWEST stability value)."""
    neutral = build_turn_context("What are your hours?", 0.9)
    angry = build_turn_context("I am so angry and frustrated with this", 0.8)
    urgent = build_turn_context("This is an emergency, please help", 0.9)
    frustrated = build_turn_context("This is not working at all", 0.7)

    assert angry.tts_stability_hint > neutral.tts_stability_hint
    assert urgent.tts_stability_hint > neutral.tts_stability_hint
    assert frustrated.tts_stability_hint > neutral.tts_stability_hint
    assert angry.tts_stability_hint == urgent.tts_stability_hint == frustrated.tts_stability_hint


def test_stability_values_stay_within_elevenlabs_conversational_range():
    """Guardrail against regressing into either extreme -- ElevenLabs'
    own guidance places conversational-agent stability roughly in
    0.30-0.85; both this codebase's baseline and its distress override
    should sit comfortably inside that band, not at either edge."""
    neutral = build_turn_context("What are your hours?", 0.9)
    angry = build_turn_context("I am so angry and frustrated with this", 0.8)
    assert 0.30 <= neutral.tts_stability_hint <= 0.85
    assert 0.30 <= angry.tts_stability_hint <= 0.85
