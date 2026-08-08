import app.voice.humanization_engine as humanization_engine
from app.voice.humanization_engine import analyze_response
from app.voice.turn_signals import UserMood


def test_empty_text_returns_neutral_decision():
    d = analyze_response("")
    assert d.text == ""
    assert d.mood == UserMood.NEUTRAL
    assert d.acknowledgement.eligible is False
    assert d.filler.allowed is False
    assert d.pacing.sentence_count == 0


def test_none_like_whitespace_text_returns_neutral_decision():
    d = analyze_response("   ")
    assert d.text.strip() == ""
    assert d.mood == UserMood.NEUTRAL


def test_short_text_marks_short_utterance():
    d = analyze_response("Okay, sure.")
    assert d.pacing.is_short_utterance is True
    assert d.pacing.has_multiple_sentences is False


def test_normal_sentence_single_boundary():
    d = analyze_response("Your appointment is confirmed for tomorrow at noon.")
    assert d.text == "Your appointment is confirmed for tomorrow at noon."
    assert d.pacing.sentence_count == 1
    assert d.pacing.has_multiple_sentences is False
    assert d.pacing.is_short_utterance is False


def test_multiple_sentences_detected():
    d = analyze_response("Sure, I can help with that. Let me check the schedule.")
    assert d.pacing.sentence_count == 2
    assert d.pacing.has_multiple_sentences is True


def test_question_counts_as_sentence_boundary():
    d = analyze_response("Would tomorrow at noon work for you?")
    assert d.pacing.sentence_count == 1
    assert d.pacing.has_multiple_sentences is False


def test_emotional_text_reuses_detect_emotion():
    d = analyze_response("I'm so sorry, unfortunately that slot isn't available.")
    assert d.response_emotion == "sad"


def test_acknowledgement_candidate_eligible():
    d = analyze_response(
        "Let me pull that up for you.",
        user_text="Can you tell me more about your pricing options",
    )
    assert d.acknowledgement.eligible is True


def test_acknowledgement_not_eligible_for_short_user_text():
    d = analyze_response("Sure thing.", user_text="hi")
    assert d.acknowledgement.eligible is False


def test_acknowledgement_not_eligible_for_emotional_skip_phrase():
    d = analyze_response(
        "I understand, let's sort this out.",
        user_text="I have an emergency and need help right now please",
    )
    assert d.acknowledgement.eligible is False


def test_filler_is_never_allowed_in_this_phase():
    d = analyze_response("Umm, let me think about that for a moment please.")
    assert d.filler.allowed is False


def test_feature_disabled_returns_neutral_regardless_of_input(monkeypatch):
    monkeypatch.setattr(
        humanization_engine.settings, "VOICE_ENABLE_HUMANIZATION_ENGINE", False
    )
    original_text = "This is great news, I'm thrilled to help!"
    d = analyze_response(
        original_text,
        user_text="Can you tell me more about your pricing options",
    )
    # Disabled means "no humanization decisions" — not "discard the text".
    # The original LLM text must remain usable by the caller.
    assert d.text == original_text
    assert d.mood == UserMood.NEUTRAL
    assert d.response_emotion == "neutral"
    assert d.acknowledgement.eligible is False


def test_engine_never_raises_on_internal_error(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(humanization_engine, "_acknowledgement_hint", _boom)
    d = analyze_response("Some normal response text.", user_text="a fairly long user query here")
    # Falls back to the safe neutral decision instead of propagating.
    assert d.mood == UserMood.NEUTRAL
    assert d.acknowledgement.eligible is False


def test_malformed_input_type_does_not_crash():
    # A caller could pass a non-str-like object; the engine must degrade
    # safely rather than raise, since humanization must never block TTS.
    d = analyze_response(None)  # type: ignore[arg-type]
    assert d.text == ""
    assert d.mood == UserMood.NEUTRAL


def test_mood_is_reused_from_turn_signals_not_reimplemented():
    d = analyze_response(
        "I understand this is urgent, let me help right away.",
        user_text="This is an emergency, I need help immediately",
        stt_confidence=0.9,
    )
    assert d.mood == UserMood.URGENT


def test_tts_stability_hint_surfaced_from_turn_context():
    d = analyze_response(
        "Let's get this sorted for you.",
        user_text="This is unacceptable, I want a refund now",
    )
    assert d.tts_stability_hint is not None
