"""
Shared LLM-output-to-TTS flush-boundary decisions.

Both call transports (Twilio's BidirectionalStreamHandler and the browser's
ConversationOrchestrator) buffer streamed LLM tokens and decide when a prefix
of that buffer is safe to hand to TtsPipeline.queue_tts(). This module holds
that pure text-decision logic; callers keep owning their own tuning knobs
(word-count floors, target word counts) so existing per-transport latency
behavior is unchanged.
"""

from __future__ import annotations

import re

_SENTENCE_END_RE = re.compile(r"([.!?])(\s+|$)")
_SOFT_BOUNDARY_RE = re.compile(r"([,;:])(\s+|$)")

# Public aliases so other modules (e.g. app.voice.humanization_engine's
# pacing analysis) can reuse the same punctuation-aware boundary detection
# instead of a naive `[.!?]` scan that would misfire on decimals, currency,
# phone numbers, and URLs (a bare "." inside "42.50" or "example.com" is
# never followed by whitespace/end, so these patterns correctly skip it).
SENTENCE_END_RE = _SENTENCE_END_RE
SOFT_BOUNDARY_RE = _SOFT_BOUNDARY_RE


def find_sentence_flush_index(
    buf: str,
    min_words: int,
    max_words: int,
    *,
    first_chunk: bool = False,
) -> int | None:
    """
    Return an index (end-exclusive) where `buf` can safely be flushed to TTS.

    Prefers sentence boundaries (newline, or ., !, ? followed by whitespace/end).
    When first_chunk is True, selects the earliest valid boundary (sentence or soft)
    meeting min_words so initial speech audio starts as early as possible.
    Falls back to a soft boundary once the buffer reaches `max_words`, so a long clause
    without terminal punctuation still flushes.
    Returns None if no boundary meeting `min_words` is found.
    """
    if not buf:
        return None

    nl = buf.find("\n")
    if nl != -1:
        prefix = buf[:nl].strip()
        if len(prefix.split()) >= min_words:
            return nl

    if first_chunk:
        first_soft = None
        for m in _SOFT_BOUNDARY_RE.finditer(buf):
            p = buf[: m.end(1)].strip()
            if len(p.split()) >= min_words:
                first_soft = m.end(1)
                break

        first_sent = None
        for m in _SENTENCE_END_RE.finditer(buf):
            p = buf[: m.end(1)].strip()
            if len(p.split()) >= min_words:
                first_sent = m.end(1)
                break

        if first_soft is not None and first_sent is not None:
            return min(first_soft, first_sent)
        if first_soft is not None:
            return first_soft
        if first_sent is not None:
            return first_sent

    last_boundary = None
    for m in _SENTENCE_END_RE.finditer(buf):
        last_boundary = m.end(1)

    if last_boundary is not None:
        prefix = buf[:last_boundary].strip()
        if len(prefix.split()) >= min_words:
            return last_boundary

    words = buf.split()
    if len(words) >= max_words:
        last_soft = None
        for m in _SOFT_BOUNDARY_RE.finditer(buf):
            last_soft = m.end(1)
        if last_soft is not None:
            prefix = buf[:last_soft].strip()
            if len(prefix.split()) >= min_words:
                return last_soft

    return None


def find_time_flush_index(buf: str, min_words: int, target_words: int) -> int | None:
    """
    Time-based flush (Vapi-style): once punctuation is delayed, flush on a
    safe word boundary near `target_words` so the caller starts speaking
    fast instead of waiting for a full sentence to accumulate.

    Returns None if `buf` has fewer than `min_words` words, or if no clean
    word boundary can be matched.
    """
    if not buf:
        return None
    words = buf.split()
    if len(words) < min_words:
        return None

    count = min(target_words, len(words))
    m = re.match(rf"^(?:\S+\s+){{{count - 1}}}\S+", buf)
    if not m:
        return None
    return m.end()
