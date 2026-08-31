"""
Phase 4C-2: TtsPipeline._process_chunk threading decision.pacing through to
_stream_tts_chunk — the shared point for both transports.

Covers: pacing is passed without recomputation, queue_tts() stays
non-blocking with pacing enabled, synthesis is not delayed, and each chunk
in a multi-chunk turn gets its own independent pause decision.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.voice.tts_pipeline import TtsPipeline
from app.voice.tts_stream_mixin import TtsStreamMixin


class _FakeHandler:
    def __init__(self):
        self._tts_cancel = asyncio.Event()
        self._current_turn_user_text = ""
        self._current_turn_stt_confidence = 0.0
        self.prefetch_calls: list[Dict[str, Any]] = []
        self.stream_calls: list[Dict[str, Any]] = []

    async def _prefetch_tts_audio(self, task: Dict[str, Any]):
        self.prefetch_calls.append(dict(task))
        return b"\xff" * 160

    async def _stream_tts_chunk(
        self,
        text,
        use_ssml=False,
        is_final=False,
        prefetched_bytes=None,
        pacing=None,
        previous_text=None,
    ):
        self.stream_calls.append(
            {"text": text, "is_final": is_final, "pacing": pacing}
        )


async def _queue_and_wait(pipeline: TtsPipeline, chunk_id: int, **task_kwargs):
    task = {"chunk_id": chunk_id, "use_ssml": False, "is_final": False, **task_kwargs}
    await pipeline.queue_tts(task)
    t = pipeline._synthesis_tasks.get(chunk_id)
    if t is not None:
        await t


# 18 & decision flow-through: multiple eligible chunks each get their own
# pacing hint (not recomputed by _stream_tts_chunk, not shared/stale).
@pytest.mark.asyncio
async def test_multiple_chunks_each_receive_their_own_pacing_decision(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 2)
    handler = _FakeHandler()
    handler._current_turn_user_text = "hi"
    pipeline = TtsPipeline(handler)

    await pipeline.queue_tts(
        {"text": "First sentence here.", "chunk_id": 0, "use_ssml": False, "is_final": False}
    )
    t0 = pipeline._synthesis_tasks.get(0)
    if t0:
        await t0
    await pipeline.queue_tts(
        {"text": "Second sentence follows.", "chunk_id": 1, "use_ssml": False, "is_final": True}
    )
    t1 = pipeline._synthesis_tasks.get(1)
    if t1:
        await t1

    assert len(handler.stream_calls) == 2
    pacing_0 = handler.stream_calls[0]["pacing"]
    pacing_1 = handler.stream_calls[1]["pacing"]
    assert pacing_0 is not None
    assert pacing_1 is not None
    # Each chunk's pacing reflects its own text, not a stale/shared object.
    assert pacing_0.ending_type.value == "statement"
    assert pacing_1.ending_type.value == "statement"


@pytest.mark.asyncio
async def test_analyze_response_called_exactly_once_per_chunk_not_recomputed():
    """
    _process_chunk must compute the decision once and pass its .pacing
    through — never call analyze_response a second time for the same chunk
    (e.g. inside the _stream_tts_chunk call path).
    """
    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    call_count = {"n": 0}
    from app.voice import tts_pipeline as tts_pipeline_module

    original = tts_pipeline_module.analyze_response

    def _counting_wrapper(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    with patch("app.voice.tts_pipeline.analyze_response", side_effect=_counting_wrapper):
        await _queue_and_wait(pipeline, 0, text="A complete sentence here.", is_final=False)

    assert call_count["n"] == 1


# --- humanization_decision kwarg: Twilio-only, never sent to the browser/
# LiveKit transport's separate _stream_tts_chunk (which does not accept it) ---


class _FakeTwilioShapedHandler(TtsStreamMixin):
    """isinstance(_, TtsStreamMixin) must be True -- mirrors how
    BidirectionalStreamHandler actually inherits TtsStreamMixin -- while
    overriding _stream_tts_chunk itself so this test only exercises
    TtsPipeline's kwarg-forwarding decision, not the mixin's real synthesis
    internals (covered separately in tests/voice/test_pacing_twilio_livekit.py)."""

    def __init__(self):
        self._tts_cancel = asyncio.Event()
        self._current_turn_user_text = ""
        self._current_turn_stt_confidence = 0.0
        self.stream_calls: list[Dict[str, Any]] = []

    async def _prefetch_tts_audio(self, task: Dict[str, Any]):
        return b"\xff" * 160

    async def _stream_tts_chunk(self, text, **kwargs):
        self.stream_calls.append({"text": text, **kwargs})


@pytest.mark.asyncio
async def test_humanization_decision_forwarded_to_twilio_shaped_handler():
    handler = _FakeTwilioShapedHandler()
    pipeline = TtsPipeline(handler)

    await _queue_and_wait(pipeline, 0, text="A complete sentence here.", is_final=True)

    assert len(handler.stream_calls) == 1
    assert "humanization_decision" in handler.stream_calls[0]
    assert handler.stream_calls[0]["humanization_decision"] is not None


@pytest.mark.asyncio
async def test_humanization_decision_not_forwarded_to_non_twilio_handler():
    """Regression guard: LiveKitBrowserCallHandler's OWN, separate
    _stream_tts_chunk does not accept humanization_decision -- forwarding it
    unconditionally would TypeError on every chunk of every browser/demo
    call. _FakeHandler here is not a TtsStreamMixin instance, mirroring that
    transport's shape."""
    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    await _queue_and_wait(pipeline, 0, text="A complete sentence here.", is_final=True)

    assert len(handler.stream_calls) == 1


# 9 & 10. Pacing does not delay synthesis; queue_tts() remains non-blocking.
@pytest.mark.asyncio
async def test_queue_tts_still_returns_before_synthesis_completes_with_pacing_enabled(
    monkeypatch,
):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    handler = _FakeHandler()
    still_running = asyncio.Event()
    release = asyncio.Event()

    async def _slow_prefetch(task):
        still_running.set()
        await release.wait()
        return b"\xff" * 160

    handler._prefetch_tts_audio = _slow_prefetch
    pipeline = TtsPipeline(handler)

    await pipeline.queue_tts(
        {"text": "A complete sentence here.", "chunk_id": 0, "use_ssml": False, "is_final": False}
    )

    # queue_tts already returned; synthesis (and therefore any pacing
    # decision downstream of it) is still in flight in the background.
    await asyncio.wait_for(still_running.wait(), timeout=1.0)
    assert not release.is_set()

    release.set()
    t = pipeline._synthesis_tasks.get(0)
    if t is not None:
        await t


@pytest.mark.asyncio
async def test_final_chunk_never_receives_pause_eligible_pacing_flow(monkeypatch):
    """End-to-end sanity: is_final chunk flows through with pacing attached,
    but eligibility itself is enforced downstream (pause_frames_for_chunk),
    not by TtsPipeline stripping is_final chunks' pacing."""
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    await _queue_and_wait(pipeline, 0, text="A complete sentence here.", is_final=True)

    assert handler.stream_calls[0]["is_final"] is True
    assert handler.stream_calls[0]["pacing"] is not None


# ---------------------------------------------------------------------------
# Phase 4C-3: exercised against the actual live default (3 frames), not a
# monkeypatched value — these prove the config that will really run in
# staging/production, not just that the mechanism works when configured.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_realistic_multi_sentence_response_twilio_shaped_handler():
    """
    Simulates a typical 3-sentence agent turn, flushed as 3 chunks (as the
    real flush logic would), through the shared TtsPipeline with the actual
    live VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES default — no monkeypatch.
    """
    assert settings.VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES == 3  # sanity on the live default

    handler = _FakeHandler()
    handler._current_turn_user_text = "Can you help me with my account please"
    pipeline = TtsPipeline(handler)

    sentences = [
        ("Sure, I can help with that.", False),
        ("Let me check your account.", False),
        ("Would you like me to proceed?", True),
    ]
    for i, (text, is_final) in enumerate(sentences):
        await _queue_and_wait(pipeline, i, text=text, is_final=is_final)

    assert len(handler.stream_calls) == 3
    # First two are non-final, real sentence endings, not short -> eligible.
    assert handler.stream_calls[0]["pacing"].ending_type.value == "statement"
    assert handler.stream_calls[1]["pacing"].ending_type.value == "statement"
    # Third is_final -> pause_frames_for_chunk will return 0 regardless of pacing.
    assert handler.stream_calls[2]["is_final"] is True
    assert handler.stream_calls[2]["pacing"].ending_type.value == "question"


@pytest.mark.asyncio
async def test_realistic_multi_sentence_response_livekit_shaped_handler():
    """Same scenario, second independent fake handler instance (simulating
    the browser/LiveKit transport) — proves no per-transport divergence."""
    handler = _FakeHandler()
    handler._current_turn_user_text = "Can you help me with my account please"
    pipeline = TtsPipeline(handler)

    sentences = [
        ("Sure, I can help with that.", False),
        ("Let me check your account.", False),
        ("Would you like me to proceed?", True),
    ]
    for i, (text, is_final) in enumerate(sentences):
        await _queue_and_wait(pipeline, i, text=text, is_final=is_final)

    assert len(handler.stream_calls) == 3
    assert handler.stream_calls[0]["pacing"].ending_type.value == "statement"
    assert handler.stream_calls[1]["pacing"].ending_type.value == "statement"
    assert handler.stream_calls[2]["is_final"] is True


@pytest.mark.asyncio
async def test_disabled_humanization_yields_zero_pacing_even_with_live_default(monkeypatch):
    """
    VOICE_ENABLE_HUMANIZATION_ENGINE=False must still result in zero pacing
    end-to-end, even though VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES now
    defaults to 3 (not 0) — the humanization flag is the outer gate.
    """
    monkeypatch.setattr(settings, "VOICE_ENABLE_HUMANIZATION_ENGINE", False)
    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    await _queue_and_wait(
        pipeline, 0, text="Sure, I can help with that. Let me check.", is_final=False
    )

    from app.voice.humanization_engine import pause_frames_for_chunk

    pacing = handler.stream_calls[0]["pacing"]
    assert pacing is not None  # neutral decision still carries a (neutral) pacing hint
    assert pause_frames_for_chunk(pacing, is_final=False) == 0
