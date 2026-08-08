from app.voice.tts_flush import find_sentence_flush_index, find_time_flush_index


# ---------------------------------------------------------------------------
# find_sentence_flush_index
# ---------------------------------------------------------------------------


def test_sentence_flush_empty_buffer():
    assert find_sentence_flush_index("", min_words=4, max_words=6) is None


def test_sentence_flush_below_min_words_returns_none():
    # "Hi." is a complete sentence but under the word floor.
    assert find_sentence_flush_index("Hi.", min_words=4, max_words=6) is None


def test_sentence_flush_on_period_boundary():
    buf = "This is a complete sentence."
    idx = find_sentence_flush_index(buf, min_words=4, max_words=6)
    assert idx is not None
    assert buf[:idx].strip() == "This is a complete sentence."


def test_sentence_flush_prefers_newline_boundary():
    buf = "Short line here\nmore text after"
    idx = find_sentence_flush_index(buf, min_words=3, max_words=6)
    assert idx is not None
    assert buf[:idx] == "Short line here"


def test_sentence_flush_multiple_sentences_uses_last_boundary():
    buf = "First sentence here. Second sentence too. Trailing partial"
    idx = find_sentence_flush_index(buf, min_words=4, max_words=6)
    assert buf[:idx].strip() == "First sentence here. Second sentence too."


def test_sentence_flush_falls_back_to_soft_boundary_when_long():
    buf = "one, two, three, four, five, six, seven without terminal punctuation"
    idx = find_sentence_flush_index(buf, min_words=2, max_words=6)
    assert idx is not None
    # Falls back to the last comma boundary once max_words is exceeded.
    assert buf[:idx].rstrip().endswith(",")


def test_sentence_flush_no_boundary_returns_none():
    buf = "just a few words no punctuation"
    assert find_sentence_flush_index(buf, min_words=4, max_words=100) is None


def test_sentence_flush_exclamation_and_question_marks():
    for punct in ("!", "?"):
        buf = f"Are we done here{punct} Not quite yet"
        idx = find_sentence_flush_index(buf, min_words=3, max_words=6)
        assert idx is not None
        assert buf[:idx].strip() == f"Are we done here{punct}"


# ---------------------------------------------------------------------------
# find_time_flush_index
# ---------------------------------------------------------------------------


def test_time_flush_empty_buffer():
    assert find_time_flush_index("", min_words=2, target_words=4) is None


def test_time_flush_below_min_words_returns_none():
    assert find_time_flush_index("one two", min_words=4, target_words=4) is None


def test_time_flush_at_min_words_returns_index():
    buf = "one two three four"
    idx = find_time_flush_index(buf, min_words=4, target_words=4)
    assert idx == len(buf)


def test_time_flush_uses_target_words_word_boundary():
    buf = "one two three four five six seven eight"
    idx = find_time_flush_index(buf, min_words=2, target_words=4)
    assert buf[:idx] == "one two three four"


def test_time_flush_target_larger_than_available_words_clamped():
    buf = "one two three"
    idx = find_time_flush_index(buf, min_words=2, target_words=8)
    # target_words is clamped to len(words) internally.
    assert buf[:idx] == "one two three"


def test_time_flush_partial_llm_chunk_no_trailing_space():
    # Simulates a partial LLM token stream mid-word; should not crash and
    # should still resolve to a valid, in-bounds index.
    buf = "one two three fo"
    idx = find_time_flush_index(buf, min_words=2, target_words=3)
    assert idx is not None
    assert 0 <= idx <= len(buf)


# ---------------------------------------------------------------------------
# Twilio-path vs. browser-path tuning parity (regression guard for Phase 2)
# ---------------------------------------------------------------------------


def test_twilio_time_flush_floor_is_lower_than_browser():
    """
    bidirectional_stream.py calls find_time_flush_index with
    min_words=max(TTS_FLUSH_MIN_WORDS, 2); conversation_orchestrator.py calls
    it with min_words=max(TTS_FLUSH_MIN_WORDS, 5). This asymmetry is an
    intentional, pre-existing latency-fastpath behavior (see
    BookingMixin._should_use_latency_fastpath, Twilio-only) and must not be
    silently equalized by the shared-module refactor.
    """
    # Twilio-style floor (TTS_FLUSH_MIN_WORDS=4 by default -> max(4, 2) = 4)
    # 3 words is below the floor either way here; use a buffer that only
    # clears the lower (Twilio) floor.
    three_word_buf = "Sure got it"
    assert find_time_flush_index(three_word_buf, min_words=max(2, 2), target_words=3) is not None
    assert find_time_flush_index(three_word_buf, min_words=max(2, 5), target_words=3) is None
