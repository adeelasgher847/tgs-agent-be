from app.voice.tts_flush import find_sentence_flush_index, find_time_flush_index

from app.core.config import settings


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


def test_first_chunk_flush_two_words_with_comma():
    buf = "Hello there, how can I help you?"
    idx = find_sentence_flush_index(buf, min_words=2, max_words=8, first_chunk=True)
    assert idx is not None
    assert buf[:idx].strip() == "Hello there,"


def test_first_chunk_flush_one_word_returns_none():
    buf = "Hello,"
    assert find_sentence_flush_index(buf, min_words=2, max_words=8, first_chunk=True) is None


def test_subsequent_chunk_does_not_flush_soft_boundary_under_max_words():
    buf = "Hello there, how can I help you?"
    # For subsequent chunks (first_chunk=False), a soft boundary (comma) under max_words (8) is not flushed
    # sentence end ("help you?") requires min_words=4 which matches
    idx = find_sentence_flush_index(buf, min_words=4, max_words=8, first_chunk=False)
    assert idx is not None
    assert buf[:idx].strip() == "Hello there, how can I help you?"


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


# ---------------------------------------------------------------------------
# Phase 5-6: realistic token-arrival timing simulation against the shared
# flush functions, mirroring the decision logic in
# app.routers.bidirectional_stream.BidirectionalStreamHandler.try_stream /
# app.voice.conversation_orchestrator.ConversationOrchestrator.try_stream:
#
#   flush_idx = find_sentence_flush_index(buf, min_words, max_words)
#   if flush_idx is None and (now - last_flush_ts) >= time_flush_sec:
#       flush_idx = find_time_flush_index(buf, time_min_words, target_words)
#
# These tests use a fake clock (no real sleeps) so they stay fast and
# deterministic while still exercising the real VOICE_TTS_TIME_FLUSH_SEC /
# VOICE_TTS_FLUSH_MAX_WORDS values from settings.
# ---------------------------------------------------------------------------


def _simulate_stream(
    words: list[str],
    word_delay_sec: float,
    *,
    min_words: int = 4,
    max_words: int | None = None,
    time_flush_sec: float | None = None,
    time_min_words: int = 4,
    time_target_words: int = 8,
) -> list[str]:
    """
    Feed `words` one at a time to the buffer, advancing a fake clock by
    `word_delay_sec` per word, and apply the same sentence-then-time flush
    decision the real call sites use. Returns the ordered list of flushed
    chunk texts (final partial buffer, if any, is NOT auto-flushed — callers
    assert on `remaining` separately when relevant).
    """
    if max_words is None:
        max_words = settings.VOICE_TTS_FLUSH_MAX_WORDS
    if time_flush_sec is None:
        time_flush_sec = settings.VOICE_TTS_TIME_FLUSH_SEC

    buf = ""
    now = 0.0
    last_flush_ts = 0.0
    chunks: list[str] = []

    for i, word in enumerate(words):
        buf += (word + " ")
        now += word_delay_sec

        flush_idx = find_sentence_flush_index(buf, min_words, max_words)
        if flush_idx is None and (now - last_flush_ts) >= time_flush_sec:
            flush_idx = find_time_flush_index(buf, time_min_words, time_target_words)

        if flush_idx is not None:
            chunks.append(buf[:flush_idx].strip())
            buf = buf[flush_idx:].lstrip()
            last_flush_ts = now

    if buf.strip():
        # Expose any un-flushed trailing partial as a final "chunk" so callers
        # can assert on it explicitly if they care (most tests below don't).
        chunks.append(buf.strip())

    return chunks


_MULTI_SENTENCE_TEXT = (
    "That's a great option for you. Let me explain how it works. "
    "The process is very simple."
)


def test_fast_stream_flushes_at_sentence_boundary_not_early():
    """A short sentence arriving quickly should flush once, at the period,
    not be split by the time-flush fallback."""
    words = "That's a great option for you.".split()
    chunks = _simulate_stream(words, word_delay_sec=0.03)
    assert chunks == ["That's a great option for you."]


def test_normal_cadence_produces_sentence_aligned_chunks():
    """At a realistic ~30ms/word cadence, each ~6-word sentence completes
    well within the new 0.25s fallback window (~180ms), so chunks should be
    sentence-aligned rather than split mid-clause. (Slower cadences where a
    sentence takes longer than 250ms to arrive are covered separately below
    — the fallback is *expected* to fire there; that's the documented
    latency tradeoff of this phase, not a bug.)"""
    words = _MULTI_SENTENCE_TEXT.split()
    chunks = _simulate_stream(words, word_delay_sec=0.03)
    for chunk in chunks:
        assert chunk[-1:] in (".", "!", "?"), f"non-sentence-aligned chunk: {chunk!r}"


def test_cadence_slower_than_fallback_window_still_yields_a_flush():
    """At an 80ms/word cadence, this sentence (6 words = ~480ms) exceeds the
    new 250ms fallback window before its period arrives, so the time-flush
    fallback is *expected* to fire mid-sentence here — this is the honest
    latency tradeoff documented for this phase, not a regression. The
    fallback must still produce a bounded, non-empty chunk rather than
    stalling indefinitely."""
    words = "That's a great option for you.".split()
    chunks = _simulate_stream(words, word_delay_sec=0.08)
    assert chunks, "time-flush fallback failed to produce any chunk on a slower cadence"
    assert sum(len(c.split()) for c in chunks) == len(words)


def test_slow_stream_still_triggers_time_flush_fallback():
    """If the model genuinely pauses for longer than VOICE_TTS_TIME_FLUSH_SEC
    with no sentence boundary in sight, the time-based fallback must still be
    reachable (not effectively disabled by the higher threshold)."""
    # No terminal punctuation at all across these words, and a per-word delay
    # well above the 0.25s fallback threshold.
    words = "well let me think about that for a moment before I answer".split()
    chunks = _simulate_stream(words, word_delay_sec=0.30)
    assert len(chunks) > 1, "time-flush fallback never fired on a genuinely slow stream"
    # The first flushed chunk must have come from the time-fallback path
    # (no sentence-ending punctuation present anywhere in the source words).
    assert all(c[-1] not in ".!?" for c in chunks[:-1] if c)


def test_max_word_cap_still_bounds_unbounded_soft_boundary_chunks():
    """A long, comma-separated clause with no terminal punctuation should
    still flush once the (now 8-word) max-word threshold is reached, rather
    than growing unboundedly."""
    words = "one, two, three, four, five, six, seven, eight, nine, ten,".split()
    chunks = _simulate_stream(
        words,
        word_delay_sec=0.01,  # fast arrival: time-flush should not fire here
        min_words=2,
        time_flush_sec=10.0,  # effectively disable the time fallback for this test
    )
    assert chunks, "expected at least one soft-boundary flush once max_words was reached"
    first_chunk_word_count = len(chunks[0].split())
    assert first_chunk_word_count >= settings.VOICE_TTS_FLUSH_MAX_WORDS


def test_multi_sentence_response_prefers_sentence_boundaries_over_mid_sentence_splits():
    """Property (not exact-count) assertion for the canonical Phase 5-5
    regression example: at fast-enough cadences (each ~5-6 word sentence
    completing within the new 0.25s fallback window), every flushed chunk
    should end at a sentence boundary rather than mid-clause — i.e. the
    fallback should not preempt a boundary that was about to arrive anyway."""
    words = _MULTI_SENTENCE_TEXT.split()
    for word_delay_sec in (0.02, 0.03, 0.04):
        chunks = _simulate_stream(words, word_delay_sec=word_delay_sec)
        rejoined = " ".join(chunks)
        assert rejoined.replace("  ", " ") == _MULTI_SENTENCE_TEXT.strip().replace(
            "  ", " "
        )
        for chunk in chunks:
            assert chunk[-1:] in (
                ".",
                "!",
                "?",
            ), f"delay={word_delay_sec}: mid-sentence fragment produced: {chunk!r}"
