"""
Phase 6-3: TtsPipeline routing to a persistent ElevenLabs WebSocket
`stream-input` session (app.services.elevenlabs_ws_session), gated behind
VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED + the elevenlabs-only
TTSProviderCapabilities.supports_streaming_session capability.

Mirrors the real-TtsPipeline-lifecycle test convention established in
tests/voice/test_previous_text_continuity.py and
tests/voice/test_pacing_tts_pipeline.py: a lightweight `_FakeHandler` stands
in for both real transports (TtsStreamMixin / LiveKitBrowserCallHandler),
since neither transport's `_stream_tts_chunk` was modified for this
feature — proving the WS-derived audio async iterator flows through the
EXACT SAME `_stream_tts_chunk` call site an HTTP-derived iterator would.

The ElevenLabsWebSocketSession class itself (protocol-level behavior: single
writer serialization, closed-session rejection, voice_settings filtering) is
covered separately in tests/voice/test_elevenlabs_ws_session.py.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.agent_runtime import ResolvedTtsRuntime
from app.services.elevenlabs_ws_session import ElevenLabsWebSocketSessionError
from app.voice.tts_pipeline import TtsPipeline


def _fake_tts_runtime(
    adapter_slug: str = "elevenlabs",
    voice_external_id: str | None = "voice-1",
    settings_json: Dict[str, Any] | None = None,
) -> ResolvedTtsRuntime:
    return ResolvedTtsRuntime(
        adapter_slug=adapter_slug,
        voice_external_id=voice_external_id,
        language="en",
        settings_json=settings_json or {},
        used_ticket_tts=False,
    )


class _FakeHandler:
    """Mirrors _FakeHandler from tests/voice/test_previous_text_continuity.py
    / test_pacing_tts_pipeline.py — stands in for either real transport."""

    def __init__(self):
        self._tts_cancel = asyncio.Event()
        self._current_turn_user_text = ""
        self._current_turn_stt_confidence = 0.0
        self.agent = MagicMock()
        self.db = None
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
        # Drain an async-iterator prefetched_bytes exactly like both real
        # transports' stream_mulaw_from_audio_iter()/_publish_mulaw_stream()
        # do — proving the WS-derived iterator is consumable via the
        # unchanged, transport-owned consumption mechanism.
        drained = b""
        if prefetched_bytes is not None and hasattr(prefetched_bytes, "__aiter__"):
            async for piece in prefetched_bytes:
                drained += piece
        self.stream_calls.append(
            {
                "text": text,
                "is_final": is_final,
                "prefetched_bytes": prefetched_bytes,
                "drained_bytes": drained,
                "previous_text": previous_text,
            }
        )


async def _wait_chunk(pipeline: TtsPipeline, chunk_id: int) -> None:
    t = pipeline._synthesis_tasks.get(chunk_id)
    if t is not None:
        await t


def _make_mock_session_class(audio_chunks: list[bytes] | None = None):
    """Build a mock ElevenLabsWebSocketSession CLASS whose instances behave
    like a real session for pipeline-level tests, without touching a real
    network connection.

    Realism detail that matters for multi-chunk-turn tests: a REAL
    ElevenLabs WS session never completes `iter_audio()` until the server
    actually reports `isFinal` — which only happens after `finalize()`
    (flush) has been sent. If this mock yielded audio immediately regardless
    of finalize, the owner chunk's Phase 3 (and its finally-block session
    cleanup) could race ahead of later relay chunks still queued to run,
    since nothing here does real network I/O to force a scheduler yield.
    Gating `iter_audio()` on `finalize()` having been called reproduces the
    real timing relationship without needing a real WebSocket.
    """
    audio_chunks = audio_chunks if audio_chunks is not None else [b"\xff" * 160]
    created_instances: list[MagicMock] = []

    def _factory(**kwargs):
        inst = MagicMock()
        inst.init_kwargs = kwargs
        finalized = asyncio.Event()

        async def _do_finalize():
            finalized.set()

        inst.start = AsyncMock()
        inst.send_text = AsyncMock()
        inst.finalize = AsyncMock(side_effect=_do_finalize)
        inst.aclose = AsyncMock()
        inst.is_closed = False

        async def _iter_audio():
            await finalized.wait()
            for c in audio_chunks:
                yield c

        inst.iter_audio = MagicMock(side_effect=lambda: _iter_audio())
        created_instances.append(inst)
        return inst

    mock_cls = MagicMock(side_effect=_factory)
    mock_cls.created_instances = created_instances
    return mock_cls


# ---------------------------------------------------------------------------
# 1. Flag OFF -> exact existing HTTP path (regression)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_uses_http_path_unchanged(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(
        settings, "VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED", False
    )

    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    with patch(
        "app.core.agent_runtime.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ):
        await pipeline.queue_tts(
            {"text": "Hello there.", "chunk_id": 0, "is_final": True}
        )
        await _wait_chunk(pipeline, 0)

    assert len(handler.prefetch_calls) == 1
    assert len(handler.stream_calls) == 1
    assert pipeline._eleven_ws_session is None


# ---------------------------------------------------------------------------
# 2. Flag ON + elevenlabs + capability -> WS path used
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_on_elevenlabs_routes_to_ws_session(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(
        settings, "VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED", True
    )

    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    mock_cls = _make_mock_session_class([b"\xff" * 160, b"\xff" * 80])

    with patch(
        "app.core.agent_runtime.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ), patch("app.services.elevenlabs_ws_session.ElevenLabsWebSocketSession", mock_cls):
        await pipeline.queue_tts(
            {"text": "Hello there.", "chunk_id": 0, "is_final": True}
        )
        await _wait_chunk(pipeline, 0)

    # HTTP path never invoked for this chunk.
    assert handler.prefetch_calls == []
    assert len(handler.stream_calls) == 1
    assert mock_cls.call_count == 1
    inst = mock_cls.created_instances[0]
    inst.start.assert_awaited_once()
    inst.finalize.assert_awaited_once()  # single-chunk turn is also the final chunk
    # is_final forced True for the WS owner regardless of chunk's own flag.
    assert handler.stream_calls[0]["is_final"] is True
    assert handler.stream_calls[0]["drained_bytes"] == b"\xff" * 160 + b"\xff" * 80
    # Session released after the turn completes.
    assert pipeline._eleven_ws_session is None


# ---------------------------------------------------------------------------
# 3 & 4. Google / Rime unchanged regardless of flag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_slug", ["google", "rime"])
async def test_non_elevenlabs_providers_never_route_to_ws(monkeypatch, provider_slug):
    from app.core.config import settings

    monkeypatch.setattr(
        settings, "VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED", True
    )

    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    mock_cls = _make_mock_session_class()

    with patch(
        "app.core.agent_runtime.resolve_tts_runtime",
        return_value=_fake_tts_runtime(provider_slug, voice_external_id="v-1"),
    ), patch("app.services.elevenlabs_ws_session.ElevenLabsWebSocketSession", mock_cls):
        await pipeline.queue_tts(
            {"text": "Some text.", "chunk_id": 0, "is_final": True}
        )
        await _wait_chunk(pipeline, 0)

    assert mock_cls.call_count == 0
    assert len(handler.prefetch_calls) == 1
    assert pipeline._eleven_ws_session is None


# ---------------------------------------------------------------------------
# 5. Lazy session creation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_not_created_until_first_chunk_needs_synthesis(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(
        settings, "VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED", True
    )

    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)
    mock_cls = _make_mock_session_class()

    # Pre-seed the audio cache so chunk 0 is a cache HIT -> no synthesis path
    # (neither HTTP nor WS) should be attempted for it at all.
    pipeline._audio_cache[pipeline._cache_key("Cached greeting.")] = b"\xff" * 40

    with patch(
        "app.core.agent_runtime.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ), patch("app.services.elevenlabs_ws_session.ElevenLabsWebSocketSession", mock_cls):
        await pipeline.queue_tts(
            {"text": "Cached greeting.", "chunk_id": 0, "is_final": False}
        )
        await _wait_chunk(pipeline, 0)
        assert (
            mock_cls.call_count == 0
        )  # still not created — cache hit skipped synthesis entirely

        await pipeline.queue_tts(
            {"text": "New uncached text.", "chunk_id": 1, "is_final": True}
        )
        await _wait_chunk(pipeline, 1)
        assert (
            mock_cls.call_count == 1
        )  # created lazily on the first chunk that actually needs it


# ---------------------------------------------------------------------------
# 6 & 7. Multiple chunks share one session, in order, no reordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_chunks_share_one_session_in_order(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(
        settings, "VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED", True
    )

    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)
    mock_cls = _make_mock_session_class()

    with patch(
        "app.core.agent_runtime.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ), patch("app.services.elevenlabs_ws_session.ElevenLabsWebSocketSession", mock_cls):
        await pipeline.queue_tts(
            {"text": "First chunk.", "chunk_id": 0, "is_final": False}
        )
        await pipeline.queue_tts(
            {"text": "Second chunk.", "chunk_id": 1, "is_final": False}
        )
        await pipeline.queue_tts(
            {"text": "Third chunk.", "chunk_id": 2, "is_final": True}
        )
        await _wait_chunk(pipeline, 0)
        await _wait_chunk(pipeline, 1)
        await _wait_chunk(pipeline, 2)

    assert mock_cls.call_count == 1  # ONE session for the whole turn
    inst = mock_cls.created_instances[0]

    # Chunk 0's text is relayed via start(initial_text=...), chunks 1 & 2 via send_text().
    assert inst.start.await_args.kwargs["initial_text"] == "First chunk."
    sent_texts = [c.args[0] for c in inst.send_text.await_args_list]
    assert sent_texts == ["Second chunk.", "Third chunk."]
    inst.finalize.assert_awaited_once()

    # Only the owner (chunk 0) actually streamed audio; chunks 1 & 2 were
    # pure text-relays with nothing further to do.
    assert len(handler.stream_calls) == 1


# ---------------------------------------------------------------------------
# 9. Final flush sent exactly once at true turn-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_called_exactly_once(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(
        settings, "VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED", True
    )

    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)
    mock_cls = _make_mock_session_class()

    with patch(
        "app.core.agent_runtime.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ), patch("app.services.elevenlabs_ws_session.ElevenLabsWebSocketSession", mock_cls):
        # Chunk 0 (the owner, is_final=False) blocks in Phase 3 awaiting
        # audio until finalize() is called — so we must NOT await its
        # completion before sending the true turn-final marker below (it
        # would hang, exactly like a real server would never emit isFinal
        # without a flush). This mirrors realistic turn timing.
        await pipeline.queue_tts(
            {"text": "Only chunk.", "chunk_id": 0, "is_final": False}
        )
        # Give chunk 0's task a beat to actually run and establish the
        # session (queue_tts() only SCHEDULES the chunk's task — it doesn't
        # run until the loop yields — mirroring the real, non-zero gap that
        # always exists in production between queuing a chunk and the LLM
        # stream loop producing its next one).
        for _ in range(50):
            if pipeline._eleven_ws_session is not None:
                break
            await asyncio.sleep(0.01)
        assert pipeline._eleven_ws_session is not None
        # True turn-final marker arrives as a SEPARATE, later empty-text
        # queue_tts() call (the edge case documented in queue_tts()).
        await pipeline.queue_tts({"text": "", "chunk_id": 1, "is_final": True})
        await _wait_chunk(pipeline, 0)

    inst = mock_cls.created_instances[0]
    inst.finalize.assert_awaited_once()


# ---------------------------------------------------------------------------
# 11 & 12. New turn always creates a fresh session; normal reset is surgical
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_turn_never_reuses_prior_turn_session(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(
        settings, "VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED", True
    )

    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)
    mock_cls = _make_mock_session_class()

    with patch(
        "app.core.agent_runtime.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ), patch("app.services.elevenlabs_ws_session.ElevenLabsWebSocketSession", mock_cls):
        # NOTE: the "chunk_id" field in the task dict is caller-supplied,
        # cosmetic metadata only — TtsPipeline assigns its OWN internal
        # chunk id from self._next_chunk_id (which reset_previous_text_
        # continuity() deliberately does NOT reset, matching pre-existing,
        # unrelated behavior). Track the real internal id via
        # pipeline._next_chunk_id - 1 rather than assuming it restarts at 0.
        await pipeline.queue_tts({"text": "Turn one.", "chunk_id": 0, "is_final": True})
        await _wait_chunk(pipeline, pipeline._next_chunk_id - 1)
        assert mock_cls.call_count == 1
        assert pipeline._eleven_ws_session is None  # released after normal completion

        # Normal (non-barge-in) turn-start hook — must not disturb _turn_id/gates.
        turn_id_before = pipeline._turn_id
        gate_before = pipeline._playback_events
        pipeline.reset_previous_text_continuity()
        assert pipeline._turn_id == turn_id_before
        assert pipeline._playback_events is gate_before
        assert pipeline._eleven_ws_session is None

        await pipeline.queue_tts({"text": "Turn two.", "chunk_id": 0, "is_final": True})
        await _wait_chunk(pipeline, pipeline._next_chunk_id - 1)

    assert mock_cls.call_count == 2  # a brand new session for turn two
    assert mock_cls.created_instances[0] is not mock_cls.created_instances[1]


# ---------------------------------------------------------------------------
# 13. Barge-in closes the session and cancels internal tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_barge_in_closes_ws_session(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(
        settings, "VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED", True
    )

    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    release_gate = asyncio.Event()

    def _factory(**kwargs):
        inst = MagicMock()
        inst.start = AsyncMock()
        inst.send_text = AsyncMock()
        inst.finalize = AsyncMock()
        inst.aclose = AsyncMock()

        async def _iter_audio():
            # Block "mid-stream" until the barge-in fires, simulating audio
            # still arriving when the caller interrupts.
            await release_gate.wait()
            yield b"\xff" * 160

        inst.iter_audio = MagicMock(side_effect=lambda: _iter_audio())
        return inst

    mock_cls = MagicMock(side_effect=_factory)

    with patch(
        "app.core.agent_runtime.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ), patch("app.services.elevenlabs_ws_session.ElevenLabsWebSocketSession", mock_cls):
        await pipeline.queue_tts(
            {"text": "About to be interrupted.", "chunk_id": 0, "is_final": True}
        )
        # Give the owner task a beat to open the session and start awaiting
        # iter_audio() (blocked on release_gate).
        await asyncio.sleep(0.05)
        assert pipeline._eleven_ws_session is not None
        session_instance = pipeline._eleven_ws_session

        await pipeline.cancel_current_and_clear_queue()

    # Barge-in must have closed the session and released the pipeline's
    # reference so the NEXT turn gets a fresh one.
    session_instance.aclose.assert_awaited()
    assert pipeline._eleven_ws_session is None

    release_gate.set()  # let any lingering generator unwind harmlessly


# ---------------------------------------------------------------------------
# 14. No stale audio crosses into the next turn after cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_stale_audio_crosses_into_next_turn_after_cancel(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(
        settings, "VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED", True
    )

    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    release_gate = asyncio.Event()

    def _factory_turn1(**kwargs):
        inst = MagicMock()
        inst.start = AsyncMock()
        inst.send_text = AsyncMock()
        inst.finalize = AsyncMock()
        inst.aclose = AsyncMock()

        async def _iter_audio():
            await release_gate.wait()
            yield b"\xaa" * 160  # distinctive "turn 1" audio marker

        inst.iter_audio = MagicMock(side_effect=lambda: _iter_audio())
        return inst

    mock_cls = MagicMock(side_effect=_factory_turn1)

    with patch(
        "app.core.agent_runtime.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ), patch("app.services.elevenlabs_ws_session.ElevenLabsWebSocketSession", mock_cls):
        await pipeline.queue_tts(
            {"text": "Turn one, about to barge in.", "chunk_id": 0, "is_final": True}
        )
        await asyncio.sleep(0.05)
        turn1_task = pipeline._synthesis_tasks.get(0)

        # Barge-in fires WHILE turn 1's owner task is still blocked awaiting audio.
        await pipeline.cancel_current_and_clear_queue()

        # NOW release the stale generator — this simulates the Phase 6-2
        # "bounded trailing audio window" arriving after cancel. It must
        # never reach handler.stream_calls for the wrong turn.
        release_gate.set()
        if turn1_task is not None:
            try:
                await asyncio.wait_for(turn1_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        handler._tts_cancel.clear()  # real callers always clear before next turn

        mock_cls.side_effect = lambda **kwargs: _make_mock_session_class(
            [b"\xbb" * 160]
        ).side_effect(**kwargs)
        await pipeline.queue_tts(
            {"text": "Turn two, clean.", "chunk_id": 0, "is_final": True}
        )
        await _wait_chunk(pipeline, 0)

    # Only turn 2's audio (0xBB marker) should ever have reached the stream —
    # turn 1's 0xAA marker must never appear, proving the turn_id guard
    # discarded the stale/cancelled owner task's audio.
    all_drained = b"".join(c.get("drained_bytes", b"") for c in handler.stream_calls)
    assert b"\xaa" * 160 not in all_drained
    assert any(
        b"\xbb" * 160 in c.get("drained_bytes", b"") for c in handler.stream_calls
    )


# ---------------------------------------------------------------------------
# 15. WS failure-to-open falls back to HTTP safely, no duplicate audio
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ws_open_failure_falls_back_to_http_no_duplicate_audio(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(
        settings, "VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED", True
    )

    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    def _factory(**kwargs):
        inst = MagicMock()
        inst.start = AsyncMock(side_effect=ElevenLabsWebSocketSessionError("boom"))
        return inst

    mock_cls = MagicMock(side_effect=_factory)

    with patch(
        "app.core.agent_runtime.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ), patch("app.services.elevenlabs_ws_session.ElevenLabsWebSocketSession", mock_cls):
        await pipeline.queue_tts(
            {"text": "Fallback please.", "chunk_id": 0, "is_final": True}
        )
        await _wait_chunk(pipeline, 0)

    assert mock_cls.call_count == 1  # attempted once
    assert len(handler.prefetch_calls) == 1  # fell back to HTTP
    assert len(handler.stream_calls) == 1  # exactly one stream, no duplicate
    assert pipeline._eleven_ws_failed_this_turn is True
    assert pipeline._eleven_ws_session is None


@pytest.mark.asyncio
async def test_ws_open_failure_scopes_fallback_to_whole_turn(monkeypatch):
    """Every subsequent chunk in the SAME turn also falls back to HTTP,
    rather than re-attempting the WS session per chunk."""
    from app.core.config import settings

    monkeypatch.setattr(
        settings, "VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED", True
    )

    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    def _factory(**kwargs):
        inst = MagicMock()
        inst.start = AsyncMock(side_effect=ElevenLabsWebSocketSessionError("boom"))
        return inst

    mock_cls = MagicMock(side_effect=_factory)

    with patch(
        "app.core.agent_runtime.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ), patch("app.services.elevenlabs_ws_session.ElevenLabsWebSocketSession", mock_cls):
        await pipeline.queue_tts({"text": "First.", "chunk_id": 0, "is_final": False})
        await _wait_chunk(pipeline, 0)
        await pipeline.queue_tts({"text": "Second.", "chunk_id": 1, "is_final": True})
        await _wait_chunk(pipeline, 1)

    assert mock_cls.call_count == 1  # only ONE open attempt for the whole turn
    assert len(handler.prefetch_calls) == 2  # both chunks fell back to HTTP


# ---------------------------------------------------------------------------
# 17. No task leaks across a multi-turn sequence including one cancellation
# (mirrors Phase 6-2's asyncio.all_tasks() before/after methodology)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_task_leaks_across_multi_turn_sequence_with_cancellation(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(
        settings, "VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED", True
    )

    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    with patch(
        "app.core.agent_runtime.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ):
        # Turn 1: normal.
        mock_cls_1 = _make_mock_session_class([b"\xff" * 80])
        with patch(
            "app.services.elevenlabs_ws_session.ElevenLabsWebSocketSession", mock_cls_1
        ):
            await pipeline.queue_tts(
                {"text": "Turn one.", "chunk_id": 0, "is_final": True}
            )
            await _wait_chunk(pipeline, 0)

        pipeline.reset_previous_text_continuity()

        tasks_before = set(asyncio.all_tasks())

        # Turn 2: cancelled mid-flight.
        release_gate = asyncio.Event()

        def _factory_2(**kwargs):
            inst = MagicMock()
            inst.start = AsyncMock()
            inst.send_text = AsyncMock()
            inst.finalize = AsyncMock()
            inst.aclose = AsyncMock()

            async def _iter_audio():
                await release_gate.wait()
                yield b"\xff" * 40

            inst.iter_audio = MagicMock(side_effect=lambda: _iter_audio())
            return inst

        mock_cls_2 = MagicMock(side_effect=_factory_2)
        with patch(
            "app.services.elevenlabs_ws_session.ElevenLabsWebSocketSession", mock_cls_2
        ):
            await pipeline.queue_tts(
                {"text": "Turn two, cancel me.", "chunk_id": 0, "is_final": True}
            )
            await asyncio.sleep(0.05)
            await pipeline.cancel_current_and_clear_queue()
            release_gate.set()
            handler._tts_cancel.clear()

        # Turn 3: normal, after cancellation.
        mock_cls_3 = _make_mock_session_class([b"\xff" * 80])
        with patch(
            "app.services.elevenlabs_ws_session.ElevenLabsWebSocketSession", mock_cls_3
        ):
            await pipeline.queue_tts(
                {"text": "Turn three.", "chunk_id": 0, "is_final": True}
            )
            await _wait_chunk(pipeline, 0)

        # Give any background aclose()/cleanup tasks a beat to finish.
        await asyncio.sleep(0.2)
        tasks_after = set(asyncio.all_tasks())

    leaked = {t for t in (tasks_after - tasks_before) if not t.done()}
    assert not leaked, f"leaked tasks: {leaked}"


# ---------------------------------------------------------------------------
# 16. No transport-specific WS handling — structural guarantee
# ---------------------------------------------------------------------------


def test_no_transport_specific_ws_session_handling():
    """
    Both real transports (Twilio's TtsStreamMixin, LiveKit's
    LiveKitBrowserCallHandler) must consume WS-sourced audio through their
    EXISTING, unmodified `_stream_tts_chunk`/prefetched_bytes mechanism —
    the routing decision lives entirely in TtsPipeline._process_chunk. This
    is enforced structurally: neither transport module imports the new
    session module at all.
    """
    import app.voice.tts_stream_mixin as twilio_mixin
    import app.voice.livekit_browser_call_handler as livekit_handler

    assert "elevenlabs_ws_session" not in open(twilio_mixin.__file__).read()
    assert "elevenlabs_ws_session" not in open(livekit_handler.__file__).read()


# ---------------------------------------------------------------------------
# Code-review follow-up (blocking bug fix): a mid-session WS write failure
# must surface as a real, visible error at the TtsPipeline level too — not
# look like a normal, silent turn completion just because iter_audio()
# happened to end.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mid_turn_write_failure_logs_error_not_silent_completion(
    monkeypatch, caplog
):
    from app.core.config import settings

    monkeypatch.setattr(
        settings, "VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED", True
    )

    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    def _factory(**kwargs):
        inst = MagicMock()
        inst.start = AsyncMock()
        inst.send_text = AsyncMock()
        inst.finalize = AsyncMock()
        inst.aclose = AsyncMock()
        # Simulates ElevenLabsWebSocketSession's own behavior after a write
        # failure: iter_audio() ends (unblocked) with zero audio, and
        # write_failed becomes True — this is exactly what TtsPipeline must
        # detect and surface, since to _stream_tts_chunk this looks
        # identical to a normal (if empty) turn completion.
        inst.write_failed = True

        async def _iter_audio():
            return
            yield b""  # pragma: no cover - unreachable, keeps this an async generator

        inst.iter_audio = MagicMock(side_effect=lambda: _iter_audio())
        return inst

    mock_cls = MagicMock(side_effect=_factory)

    with patch(
        "app.core.agent_runtime.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ), patch("app.services.elevenlabs_ws_session.ElevenLabsWebSocketSession", mock_cls):
        with caplog.at_level("ERROR"):
            await pipeline.queue_tts(
                {
                    "text": "Turn cut short by a write failure.",
                    "chunk_id": 0,
                    "is_final": True,
                }
            )
            await _wait_chunk(pipeline, 0)

    error_records = [r for r in caplog.records if r.levelname == "ERROR"]
    assert any(
        "ElevenLabs WS session failed mid-turn" in r.message for r in error_records
    ), f"expected a mid-turn WS failure ERROR log, got: {[r.message for r in error_records]}"
