"""
Regression coverage: when `_stream_tts_chunk`'s incremental streaming branch
is given a prefetched ASYNC ITERATOR (`prefetched_bytes` supporting
`__aiter__` -- e.g. a provider's `async_stream_synthesize()` output, as used
for ElevenLabs/Rime/Hume) and that iterator fails mid-stream, the code must
fall back to a genuine fresh non-streaming resynthesis (generate_mulaw_tts),
not reuse the now-exhausted/errored async_generator object as if it were
batch `bytes`.

Root cause (confirmed via real production logs when Hume's account ran out
of credits mid-call): the except-handler's fallback path unconditionally
did `audio_bytes = prefetched_bytes` whenever `_use_prefetched` was still
True -- but `prefetched_bytes` here IS the same async_generator that
`stream_mulaw_from_audio_iter` had just failed to fully drain. An
async_generator object is truthy (so the `if audio_bytes:` check passed)
but has no `len()`, so (when the jitter buffer wasn't already primed) the
very next line (`apply_micro_fade_in`) raised `TypeError: object of type
'async_generator' has no len()`. `_stream_tts_chunk` has its own outer
try/except that catches and logs this rather than propagating it, so the
call didn't crash -- but the fallback synthesis never ran, silently
dropping that turn's TTS audio entirely instead of recovering via a
different, working synthesis path.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.agent_runtime import ResolvedTtsRuntime
from app.utils.audio_utils import MULAW_FRAME_BYTES


def _fake_tts_runtime(adapter_slug: str = "hume", voice_external_id: str | None = "voice-1"):
    return ResolvedTtsRuntime(
        adapter_slug=adapter_slug,
        voice_external_id=voice_external_id,
        language="en",
        settings_json={},
        used_ticket_tts=False,
    )


def _twilio_handler():
    from app.voice.tts_stream_mixin import TtsStreamMixin

    h = object.__new__(TtsStreamMixin)
    h._tts_cancel = asyncio.Event()
    h._elevenlabs_prev_tts_text = ""
    h.agent = MagicMock()
    h.agent.language = "en"
    h.agent.voice_type = "female"
    h.db = None
    h.stream_sid = "MZtest"
    h._tts_lock = asyncio.Lock()
    h.is_speaking = False
    h._is_tts_playing = False
    h._twilio_buffer_primed = True
    h._is_background_audio_enabled = lambda: False
    h._resolve_voice_volume = lambda: 1.0
    h.websocket = MagicMock()
    h.websocket.send_json = AsyncMock()
    return h


async def _failing_prefetched_iter():
    """Simulates a provider's async_stream_synthesize() output that starts
    yielding real audio, then fails mid-stream (e.g. Hume's "Exhausted
    credit balance" error arriving after the connection was already
    established and the first chunk queued)."""
    yield bytes([0x10]) * MULAW_FRAME_BYTES
    raise RuntimeError("Hume TTS error None: Exhausted credit balance.")


@pytest.mark.asyncio
async def test_failed_prefetched_iterator_falls_back_to_fresh_batch_synthesis():
    h = _twilio_handler()
    fallback_audio = bytes([0x20]) * MULAW_FRAME_BYTES * 3

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=_fake_tts_runtime()
    ), patch(
        "app.voice.tts_stream_mixin.generate_mulaw_tts",
        new=AsyncMock(return_value=fallback_audio),
    ) as mock_batch_synth:
        # Must not raise -- the whole point of the fallback.
        await h._stream_tts_chunk(
            "A complete sentence with real content.",
            is_final=True,
            prefetched_bytes=_failing_prefetched_iter(),
        )

    # Fresh non-streaming synthesis was actually invoked as the fallback...
    mock_batch_synth.assert_awaited_once()
    # ...and its audio was genuinely sent to Twilio (not silently dropped).
    assert h.websocket.send_json.await_count >= 1


@pytest.mark.asyncio
async def test_failed_prefetched_iterator_falls_back_even_on_unprimed_buffer_path():
    """Same scenario as the first test, but with _twilio_buffer_primed=False
    -- the exact condition that reaches apply_micro_fade_in() and produced
    `TypeError: object of type 'async_generator' has no len()` in
    production. Note: _stream_tts_chunk has its own outer try/except that
    catches and logs any exception rather than propagating it (so "does it
    raise" is never a meaningful assertion here) -- the real bug was that
    this TypeError, caught internally, aborted the fallback before it ever
    reached generate_mulaw_tts, silently dropping the turn's audio
    entirely. This asserts the fallback audio is genuinely produced and
    sent instead."""
    h = _twilio_handler()
    h._twilio_buffer_primed = False
    fallback_audio = bytes([0x20]) * MULAW_FRAME_BYTES * 3

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=_fake_tts_runtime()
    ), patch(
        "app.voice.tts_stream_mixin.generate_mulaw_tts",
        new=AsyncMock(return_value=fallback_audio),
    ) as mock_batch_synth:
        await h._stream_tts_chunk(
            "A complete sentence with real content.",
            is_final=True,
            prefetched_bytes=_failing_prefetched_iter(),
        )

    mock_batch_synth.assert_awaited_once()
    assert h.websocket.send_json.await_count >= 1


@pytest.mark.asyncio
async def test_barge_in_during_streaming_failure_still_skips_fallback_cleanly():
    """If cancel/hangup happened right as the stream failed, the existing
    early-return guard must still apply -- this fix must not force a
    fallback synthesis call after a legitimate cancellation."""
    h = _twilio_handler()
    h._tts_cancel.set()

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=_fake_tts_runtime()
    ), patch(
        "app.voice.tts_stream_mixin.generate_mulaw_tts", new=AsyncMock()
    ) as mock_batch_synth:
        await h._stream_tts_chunk(
            "A complete sentence with real content.",
            is_final=True,
            prefetched_bytes=_failing_prefetched_iter(),
        )

    mock_batch_synth.assert_not_awaited()
