from app.voice.backchannel_classifier import (
    TurnClassification,
    classify_turn,
    classify_turn_detailed,
    has_actionable_shape,
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


# ── Shape heuristic (supporting signal only) ──────────────────────────────────


def test_has_actionable_shape_interrogative_openers():
    assert has_actionable_shape("What now") is True
    assert has_actionable_shape("Who is that") is True
    assert has_actionable_shape("How much") is True
    assert has_actionable_shape("Is it ready") is True
    assert has_actionable_shape("Can you help") is True


def test_has_actionable_shape_imperative_verbs():
    assert has_actionable_shape("Call John") is True
    assert has_actionable_shape("Book appointment") is True
    assert has_actionable_shape("Please cancel") is True
    assert has_actionable_shape("I need help") is True
    assert has_actionable_shape("Repeat") is True


def test_has_actionable_shape_negative():
    assert has_actionable_shape("Blue widget") is False
    assert has_actionable_shape("One more") is False
    assert has_actionable_shape("Try this") is False
    assert has_actionable_shape("") is False


# ── classify_turn_detailed: word count / confidence are no longer sufficient
#    alone -- unclassified-fallback branch requires shape OR candidate evidence
#    (or a materially higher confidence bar as the never-silently-drop safety net) ──


def test_unclassified_branch_requires_supporting_signal_reason_threading():
    # Explicit interruption -> reason reflects that branch specifically.
    _, reason = classify_turn_detailed("Stop please", 0.90, is_tts_playing=True)
    assert reason == "explicit_interruption"

    # Known backchannel -> reason reflects that branch.
    _, reason = classify_turn_detailed("Hey there", 1.00, is_tts_playing=True)
    assert reason == "known_backchannel"

    # Unknown phrase with shape match (imperative verb) -> corroborated barge-in.
    classification, reason = classify_turn_detailed("Tell me", 0.30, is_tts_playing=True)
    assert classification == TurnClassification.BARGE_IN
    assert reason == "unknown_actionable_shape"

    # Unknown phrase with speech-candidate evidence (no shape match) -> corroborated barge-in.
    classification, reason = classify_turn_detailed(
        "Blue widget", 0.30, is_tts_playing=True, speech_candidate_age_ms=150.0
    )
    assert classification == TurnClassification.BARGE_IN
    assert reason == "unknown_actionable_candidate"

    # Unknown phrase, no shape, no candidate, but high confidence -> still never
    # silently dropped (safety net), just with a materially higher bar than before.
    classification, reason = classify_turn_detailed("Blue widget", 0.90, is_tts_playing=True)
    assert classification == TurnClassification.BARGE_IN
    assert reason == "unknown_actionable_high_confidence"


def test_weak_evidence_unknown_phrase_is_suppressed_not_blindly_barged_in():
    """
    The actual production bug: word count + marginal confidence alone used to
    be sufficient to barge in on ANY 2+-word STT hit. A marginal-confidence,
    shape-less, candidate-less phrase (the realistic false-positive pattern —
    misheard line noise/background chatter) must now require corroboration
    before interrupting the caller's active TTS.
    """
    classification, reason = classify_turn_detailed(
        "some garbled noise", 0.30, is_tts_playing=True
    )
    assert classification == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    assert reason == "weak_evidence_suppressed"


def test_single_word_unknown_phrase_never_raises_and_reflects_evidence():
    """1-word unclassified phrases must never be a hard drop/crash -- they
    always resolve to a valid classification, and shape/candidate evidence
    (not word count) decides which one, PROVIDED the caller has actually
    configured min_words=1 (the single-word branch is gated on the
    configured min_words, not on the incoming utterance's word_count --
    see test_single_word_unclassified_never_bypasses_configured_min_words
    below for the min_words=2 production-default case).

    min_confidence is set above min_confidence_1w here specifically to make
    the earlier word_count >= min_words branch NOT already claim the case
    (it requires confidence >= min_confidence too), isolating the
    single-word branch's own gating/evidence logic for this test -- at the
    real production defaults (min_confidence=0.26 < min_confidence_1w=0.36)
    the multi-word branch always wins once min_words=1, since word_count>=1
    is trivially true; that's expected and unrelated to this fix.
    """
    # Shape match ("repeat" is an imperative action verb) -> corroborated barge-in.
    classification, reason = classify_turn_detailed(
        "Repeat", 0.40, is_tts_playing=True, min_words=1, min_confidence=0.90
    )
    assert classification == TurnClassification.BARGE_IN
    assert reason == "unknown_actionable_shape_1w"

    # No shape, no candidate, marginal confidence -> held back (suppressed),
    # but the call itself never raises and always returns a definite result.
    classification, reason = classify_turn_detailed(
        "blah", 0.40, is_tts_playing=True, min_words=1, min_confidence=0.90
    )
    assert classification == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    assert reason == "weak_evidence_suppressed_1w"

    # Same phrase, but with candidate evidence -> corroborated barge-in instead
    # of being dropped.
    classification, reason = classify_turn_detailed(
        "blah",
        0.40,
        is_tts_playing=True,
        min_words=1,
        min_confidence=0.90,
        speech_candidate_age_ms=100.0,
    )
    assert classification == TurnClassification.BARGE_IN
    assert reason == "unknown_actionable_candidate_1w"


def test_single_word_unclassified_never_bypasses_configured_min_words():
    """
    At the production default (min_words=2), a 1-word unclassified utterance
    must NOT be able to reach the single-word BARGE_IN branch regardless of
    confidence or shape/candidate evidence -- it must fall straight through
    to below_word_count_threshold, exactly matching pre-fix behavior. The
    single-word branch is only reachable when min_words is explicitly
    configured to 1; word_count alone must never widen its own gate.
    """
    # Shape match + high confidence + candidate evidence -- would all trigger
    # BARGE_IN if word_count==1 alone were sufficient to enter the branch.
    classification, reason = classify_turn_detailed(
        "Repeat",
        0.99,
        is_tts_playing=True,
        min_words=2,
        speech_candidate_age_ms=50.0,
    )
    assert classification == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
    assert reason == "below_word_count_threshold"


def test_classifier_task_brief_phrase_matrix_while_tts_playing():
    """
    The exact 8-phrase classifier matrix from the barge-in/VAD task brief,
    all evaluated while TTS is playing:
      - Pure backchannels/acknowledgements -> suppressed (agent keeps talking).
      - Short interruption/attention phrases -> barge-in.
      - Question/actionable-shaped short phrases -> barge-in.
      - Clearly actionable longer phrases -> barge-in.
    """
    suppress = ["uh huh", "mm hmm", "yes", "okay"]
    for phrase in suppress:
        assert (
            classify_turn(phrase, 0.90, is_tts_playing=True)
            == TurnClassification.SUPPRESS_NON_ACTIONABLE_BACKCHANNEL
        ), f"expected suppress for {phrase!r}"

    interruption = ["wait", "stop", "hold on"]
    for phrase in interruption:
        assert (
            classify_turn(phrase, 0.90, is_tts_playing=True)
            == TurnClassification.BARGE_IN
        ), f"expected barge-in for {phrase!r}"

    question_shaped = ["can you", "can you please", "are you there", "sorry what"]
    for phrase in question_shaped:
        assert (
            classify_turn(phrase, 0.90, is_tts_playing=True)
            == TurnClassification.BARGE_IN
        ), f"expected barge-in for {phrase!r}"

    actionable_longer = [
        "I need help",
        "book me an appointment",
        "tell me about yourself",
    ]
    for phrase in actionable_longer:
        assert (
            classify_turn(phrase, 0.90, is_tts_playing=True)
            == TurnClassification.BARGE_IN
        ), f"expected barge-in for {phrase!r}"


def test_classify_turn_backward_compatible_signature_accepts_candidate_kwarg():
    # classify_turn() itself keeps returning only the enum (existing call
    # sites/tests are unaffected), with the new kwarg optional and unused
    # by default.
    assert (
        classify_turn(
            "Tell me", 0.30, is_tts_playing=True, speech_candidate_age_ms=None
        )
        == TurnClassification.BARGE_IN
    )
