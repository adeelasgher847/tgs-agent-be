"""
Tests for:
  1. The RAG interim-prefetch minimum-character-length gate
     (VOICE_RAG_PREFETCH_MIN_CHARS) added alongside the existing
     word-count/confidence gate on both transports — production defaults
     (VOICE_RAG_PREFETCH_MIN_WORDS=1, VOICE_RAG_PREFETCH_MIN_CONFIDENCE=0.05)
     let essentially any single-word interim fragment ("hi", "is", "no")
     fire a background embedding call on every qualifying interim update.
  2. Frame-level barge-in correctness: no MULAW frame is ever sent to
     Twilio after the cancel Event is set mid-stream.
  3. Rapid double `_cancel_inflight_llm_response()` is idempotent.
"""
from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.utils.audio_utils import MULAW_FRAME_BYTES, stream_mulaw_bytes_over_twilio
from tests.voice.test_barge_in_vad_and_telemetry import DummyWebSocket, _create_test_handler
from tests.voice.test_livekit_rag_prefetch import _real_handler_with_kb


# ─────────────────────────────────────────────────────────────────────────────
# RAG prefetch min-chars gate — Twilio path (bidirectional_stream.py)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_twilio_rag_prefetch_rejects_text_below_min_chars():
    handler = _create_test_handler()
    assert handler._rag_prefetch_min_chars == 4  # production default
    handler._is_tts_playing = False

    del handler.__dict__["_prefetch_rag_context"]
    handler._prefetch_rag_context = AsyncMock(return_value=("", {}))

    # "no" is 2 chars — a single qualifying word (word_count=1 >= min_words=1)
    # that must now be rejected by the new min-chars gate.
    await handler._maybe_process_interim("no", 0.90)
    assert handler._rag_prefetch_task is None
    handler._prefetch_rag_context.assert_not_called()


@pytest.mark.asyncio
async def test_twilio_rag_prefetch_fires_at_exact_min_chars_boundary():
    handler = _create_test_handler()
    handler._is_tts_playing = False

    del handler.__dict__["_prefetch_rag_context"]
    handler._prefetch_rag_context = AsyncMock(return_value=("", {}))

    # "book" is exactly 4 chars == VOICE_RAG_PREFETCH_MIN_CHARS default.
    assert len("book") == handler._rag_prefetch_min_chars
    await handler._maybe_process_interim("book", 0.90)
    assert handler._rag_prefetch_task is not None
    await handler._rag_prefetch_task


@pytest.mark.asyncio
async def test_twilio_rag_prefetch_fires_for_ordinary_multiword_query():
    """Sanity: the new gate must not regress the common case."""
    handler = _create_test_handler()
    handler._is_tts_playing = False

    del handler.__dict__["_prefetch_rag_context"]
    handler._prefetch_rag_context = AsyncMock(return_value=("", {}))

    await handler._maybe_process_interim("what are your hours", 0.90)
    assert handler._rag_prefetch_task is not None
    await handler._rag_prefetch_task


# ─────────────────────────────────────────────────────────────────────────────
# RAG prefetch min-chars gate — LiveKit/browser path
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_browser_rag_prefetch_rejects_text_below_min_chars():
    h = _real_handler_with_kb([uuid.uuid4()])
    assert h._rag_prefetch_min_chars == 4

    retrieve_mock = AsyncMock(return_value=("KB", 5.0))
    with patch(
        "app.services.kb_retrieval_service.retrieve_kb_context_for_turn", new=retrieve_mock
    ), patch("app.utils.redis_client.get_redis", return_value=None):
        await h._maybe_process_interim("no", 0.9)  # word_count=1, len=2

    assert h._rag_prefetch_task is None
    retrieve_mock.assert_not_called()


@pytest.mark.asyncio
async def test_browser_rag_prefetch_fires_for_ordinary_query():
    h = _real_handler_with_kb([uuid.uuid4()])

    retrieve_mock = AsyncMock(return_value=("KB", 5.0))
    with patch(
        "app.services.kb_retrieval_service.retrieve_kb_context_for_turn", new=retrieve_mock
    ), patch("app.utils.redis_client.get_redis", return_value=None):
        await h._maybe_process_interim("what are your hours", 0.9)
        assert h._rag_prefetch_task is not None
        await h._rag_prefetch_task

    retrieve_mock.assert_awaited_once()


# ─────────────────────────────────────────────────────────────────────────────
# Frame-level: no frame sent after cancel is set mid-stream
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_media_frame_sent_after_cancel_set_mid_stream():
    ws = DummyWebSocket()
    cancel = asyncio.Event()

    # 10 frames worth of audio (200ms). We'll cancel after the 3rd frame is
    # sent by hooking the websocket's send_json.
    audio = bytes([0xFF]) * MULAW_FRAME_BYTES * 10

    real_send_json = ws.send_json
    sent_count = 0

    async def _hooked_send_json(data):
        nonlocal sent_count
        await real_send_json(data)
        sent_count += 1
        if sent_count == 3:
            cancel.set()

    ws.send_json = _hooked_send_json

    await stream_mulaw_bytes_over_twilio(
        websocket=ws,
        stream_sid="MZ_test",
        audio_bytes=audio,
        pace_20ms=False,  # no sleeping — deterministic, fast test
        cancel=cancel,
        prime_frames=0,
    )

    media_frames = [m for m in ws.sent_json if m.get("event") == "media"]
    # Exactly 3 frames should have been sent (cancel observed at the top of
    # the loop on the 4th iteration, before any further send_json call).
    assert len(media_frames) == 3
    assert cancel.is_set()


@pytest.mark.asyncio
async def test_no_frames_sent_at_all_when_cancel_already_set_before_call():
    ws = DummyWebSocket()
    cancel = asyncio.Event()
    cancel.set()

    audio = bytes([0xFF]) * MULAW_FRAME_BYTES * 5
    await stream_mulaw_bytes_over_twilio(
        websocket=ws,
        stream_sid="MZ_test",
        audio_bytes=audio,
        pace_20ms=False,
        cancel=cancel,
        prime_frames=0,
    )
    media_frames = [m for m in ws.sent_json if m.get("event") == "media"]
    assert media_frames == []


# ─────────────────────────────────────────────────────────────────────────────
# Rapid double _cancel_inflight_llm_response() — idempotency
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rapid_double_cancel_inflight_llm_response_is_idempotent():
    handler = _create_test_handler()
    del handler.__dict__["_cancel_inflight_llm_response"]

    handler.stream_sid = "MZ_test"
    handler._tts_pipeline = MagicMock()
    handler._tts_pipeline.cancel_current_and_clear_queue = AsyncMock()
    handler._send_twilio_clear_event = AsyncMock()

    async def _never_finishes():
        await asyncio.sleep(5)

    handler._llm_response_task = asyncio.create_task(_never_finishes())
    handler._rag_prefetch_task = asyncio.create_task(_never_finishes())
    handler._rag_prefetch_user_text = "some text"
    handler._speculative_prefetch_task = asyncio.create_task(_never_finishes())

    # First cancel.
    await handler._cancel_inflight_llm_response()
    assert handler._llm_response_task is None
    assert handler._rag_prefetch_task is None
    assert handler._rag_prefetch_user_text == ""
    assert handler._speculative_prefetch_task is None

    # Second, immediate call — no exception, no duplicate/unsafe side effects.
    await handler._cancel_inflight_llm_response()

    assert handler._tts_pipeline.cancel_current_and_clear_queue.call_count == 2
    assert handler._send_twilio_clear_event.call_count == 2
    assert handler._llm_response_task is None
    assert handler._rag_prefetch_task is None
    assert handler._speculative_prefetch_task is None
