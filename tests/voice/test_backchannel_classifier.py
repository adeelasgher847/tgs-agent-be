from app.voice.backchannel_classifier import (
    TurnClassification,
    classify_turn,
    has_explicit_interruption_intent,
    is_known_non_actionable_backchannel,
    is_pure_acoustic_filler,
    normalize_transcript,
)


def test_normalize_transcript():
    assert normalize_transcript("Hey. Hi.") == "hey hi"
    assert normalize_transcript("  What's that?!  ") == "whats that"
    assert normalize_transcript("Good morning, how are you?") == "good morning how are you"
    assert normalize_transcript("") == ""
    assert normalize_transcript(None) == ""


def test_is_pure_acoustic_filler():
    assert is_pure_acoustic_filler("uh") is True
    assert is_pure_acoustic_filler("um") is True
    assert is_pure_acoustic_filler("uh huh") is True
    assert is_pure_acoustic_filler("hey") is False
    assert is_pure_acoustic_filler("stop") is False


def test_is_known_non_actionable_backchannel():
    # Greetings
    assert is_known_non_actionable_backchannel("Hey") is True
    assert is_known_non_actionable_backchannel("Hi") is True
    assert is_known_non_actionable_backchannel("Hey. Hi.") is True
    assert is_known_non_actionable_backchannel("Hi there") is True
    assert is_known_non_actionable_backchannel("Good morning") is True
    assert is_known_non_actionable_backchannel("Good afternoon") is True
    assert is_known_non_actionable_backchannel("Good evening") is True

    # Affirmations
    assert is_known_non_actionable_backchannel("Yeah yeah") is True
    assert is_known_non_actionable_backchannel("Okay okay") is True
    assert is_known_non_actionable_backchannel("Thank you") is True
    assert is_known_non_actionable_backchannel("Got it") is True
    assert is_known_non_actionable_backchannel("Sure") is True
    assert is_known_non_actionable_backchannel("Alright") is True
    assert is_known_non_actionable_backchannel("Okay. And") is True
    assert is_known_non_actionable_backchannel("Uh huh") is True

    # Legitimate short requests (MUST NOT be classified as backchannel!)
    assert is_known_non_actionable_backchannel("Tell me") is False
    assert is_known_non_actionable_backchannel("Help me") is False
    assert is_known_non_actionable_backchannel("Call John") is False
    assert is_known_non_actionable_backchannel("Book appointment") is False
    assert is_known_non_actionable_backchannel("Transfer me") is False
    assert is_known_non_actionable_backchannel("What now") is False
    assert is_known_non_actionable_backchannel("What's that") is False
    assert is_known_non_actionable_backchannel("Who is that") is False
    assert is_known_non_actionable_backchannel("Where exactly") is False
    assert is_known_non_actionable_backchannel("How much") is False
    assert is_known_non_actionable_backchannel("Tell me more") is False
    assert is_known_non_actionable_backchannel("Random unseen phrase") is False


def test_has_explicit_interruption_intent():
    assert has_explicit_interruption_intent("Stop") is True
    assert has_explicit_interruption_intent("Stop please") is True
    assert has_explicit_interruption_intent("Hold on") is True
    assert has_explicit_interruption_intent("Wait please") is True
    assert has_explicit_interruption_intent("Pause please") is True
    assert has_explicit_interruption_intent("Cancel that") is True
    assert has_explicit_interruption_intent("Stop talking") is True
    assert has_explicit_interruption_intent("Be quiet") is True
    assert has_explicit_interruption_intent("Shut up") is True
    assert has_explicit_interruption_intent("Cancel order") is True
    assert has_explicit_interruption_intent("Stop billing") is True

    # Non-command phrases
    assert has_explicit_interruption_intent("Hey. Hi.") is False
    assert has_explicit_interruption_intent("Tell me") is False
    assert has_explicit_interruption_intent("How much") is False


def test_classify_turn_when_tts_playing():
    # 1. Explicit interruption commands -> BARGE_IN
    assert (
        classify_turn("Stop please", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )
    assert (
        classify_turn("Hold on", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )
    assert (
        classify_turn("Wait please", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )
    assert (
        classify_turn("Cancel that", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )
    assert (
        classify_turn("Be quiet", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )
    assert (
        classify_turn("Stop", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )

    # 2. Known conversational backchannels -> SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    assert (
        classify_turn("Hey", 1.00, is_tts_playing=True)
        == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    )
    assert (
        classify_turn("Hi", 1.00, is_tts_playing=True)
        == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    )
    assert (
        classify_turn("Hey. Hi.", 1.00, is_tts_playing=True)
        == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    )
    assert (
        classify_turn("Hi there", 1.00, is_tts_playing=True)
        == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    )
    assert (
        classify_turn("Good morning", 1.00, is_tts_playing=True)
        == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    )
    assert (
        classify_turn("Yeah yeah", 1.00, is_tts_playing=True)
        == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    )
    assert (
        classify_turn("Okay okay", 1.00, is_tts_playing=True)
        == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    )
    assert (
        classify_turn("Thank you", 1.00, is_tts_playing=True)
        == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    )
    assert (
        classify_turn("Got it", 1.00, is_tts_playing=True)
        == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    )
    assert (
        classify_turn("Sure", 1.00, is_tts_playing=True)
        == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    )
    assert (
        classify_turn("Alright", 1.00, is_tts_playing=True)
        == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    )
    assert (
        classify_turn("Okay. And", 1.00, is_tts_playing=True)
        == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    )

    # 3. Legitimate short requests & unknown phrases spoken over TTS -> BARGE_IN (never dropped!)
    assert (
        classify_turn("Tell me", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )
    assert (
        classify_turn("Help me", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )
    assert (
        classify_turn("Call John", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )
    assert (
        classify_turn("Book appointment", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )
    assert (
        classify_turn("Transfer me", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )
    assert (
        classify_turn("What now", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )
    assert (
        classify_turn("What's that", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )
    assert (
        classify_turn("Who is that", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )
    assert (
        classify_turn("Where exactly", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )
    assert (
        classify_turn("How much", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )
    assert (
        classify_turn("Tell me more", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )
    assert (
        classify_turn("Unknown phrase widget", 0.90, is_tts_playing=True)
        == TurnClassification.BARGE_IN
    )


def test_classify_turn_when_tts_silent():
    # When agent is silent / waiting for caller, all genuine phrases are NORMAL_USER_TURN
    assert (
        classify_turn("Hey", 1.00, is_tts_playing=False)
        == TurnClassification.NORMAL_USER_TURN
    )
    assert (
        classify_turn("Hi", 1.00, is_tts_playing=False)
        == TurnClassification.NORMAL_USER_TURN
    )
    assert (
        classify_turn("Hey. Hi.", 1.00, is_tts_playing=False)
        == TurnClassification.NORMAL_USER_TURN
    )
    assert (
        classify_turn("Tell me", 1.00, is_tts_playing=False)
        == TurnClassification.NORMAL_USER_TURN
    )
    assert (
        classify_turn("Book appointment", 1.00, is_tts_playing=False)
        == TurnClassification.NORMAL_USER_TURN
    )
    assert (
        classify_turn("How much", 1.00, is_tts_playing=False)
        == TurnClassification.NORMAL_USER_TURN
    )

    # Pure acoustic fillers are still suppressed
    assert (
        classify_turn("uh", 1.00, is_tts_playing=False)
        == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    )
    assert (
        classify_turn("um", 1.00, is_tts_playing=False)
        == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    )
