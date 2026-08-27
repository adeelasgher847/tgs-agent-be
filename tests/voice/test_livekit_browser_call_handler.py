"""
Unit tests for the browser-only (no Twilio) LiveKit voice agent path.

See app/voice/livekit_browser_call_handler.py — powers the "Share Demo Link"
feature (POST /api/v1/sdk/demo/{token}/call-token). These tests exercise the
new adapter/handler and lifecycle-wiring logic without a live LiveKit server
or real audio (not feasible in a unit test) — see
tests/api/test_sdk_demo_link.py::TestDemoCallTokenAgentJoin for the HTTP-level
scheduling coverage.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.agent_runtime import ResolvedSttRuntime
from app.voice.livekit_browser_call_handler import (
    LiveKitBrowserCallHandler,
    _LiveKitAgentAudioPublisher,
    _forced_browser_stt_runtime,
    _start_browser_call_recording,
    _stop_browser_call_recording,
    run_livekit_browser_call,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _base_handler() -> LiveKitBrowserCallHandler:
    db = MagicMock()
    call_session = MagicMock()
    call_session.id = uuid.uuid4()
    call_session.user_id = uuid.uuid4()
    call_session.tenant_id = uuid.uuid4()
    call_session.call_metadata = {}
    call_session.call_transcript = []

    agent = MagicMock()
    agent.id = uuid.uuid4()
    agent.name = "Demo Agent"
    agent.language = "en"
    agent.voice_type = "female"
    agent.greeting_message = "Hi, thanks for trying the demo!"
    agent.first_message = None

    return LiveKitBrowserCallHandler(db=db, call_session=call_session, agent=agent, call_flow=None)


# ─────────────────────────────────────────────────────────────────────────────
# Construction / attribute-surface wiring
# ─────────────────────────────────────────────────────────────────────────────

class TestHandlerConstruction:
    def test_constructs_against_voice_orchestrator_and_tts_pipeline_unmodified(self):
        """
        The whole point of this handler is that VoiceOrchestrator / TtsPipeline
        construct against it with zero modification to those classes.
        """
        from app.voice.voice_orchestrator import VoiceOrchestrator
        from app.voice.tts_pipeline import TtsPipeline

        h = _base_handler()
        vo = VoiceOrchestrator(h)

        assert h._tts_pipeline is not None
        assert isinstance(h._tts_pipeline, TtsPipeline)
        assert vo.tts_pipeline is h._tts_pipeline

    def test_conversation_orchestrator_attached(self):
        from app.voice.conversation_orchestrator import ConversationOrchestrator

        h = _base_handler()
        assert isinstance(h._conversation, ConversationOrchestrator)
        assert h._conversation._h is h

    def test_stream_sid_never_none_and_never_crashes_consumers(self):
        h = _base_handler()
        assert h.stream_sid is not None
        assert h.call_session_id in h.stream_sid


# ─────────────────────────────────────────────────────────────────────────────
# Out-of-scope stubs — must never crash, never actually act
# ─────────────────────────────────────────────────────────────────────────────

class TestOutOfScopeStubs:
    def test_update_booking_memory_is_a_noop(self):
        h = _base_handler()
        assert h._update_booking_memory_from_user_turn("book me for 3pm") is None

    @pytest.mark.asyncio
    async def test_end_call_after_agent_request_does_not_raise_or_set_stop(self):
        h = _base_handler()
        await h._end_call_after_agent_request()
        assert not h._stop_event.is_set()

    @pytest.mark.asyncio
    async def test_transfer_after_agent_request_does_not_raise(self):
        h = _base_handler()
        await h._transfer_after_agent_request()

    def test_schedule_recreate_stt_for_email_collection_is_a_noop(self):
        h = _base_handler()
        assert h._schedule_recreate_stt_for_email_collection("please spell your email") is None


# ─────────────────────────────────────────────────────────────────────────────
# Barge-in / turn handling
# ─────────────────────────────────────────────────────────────────────────────

class TestTurnHandling:
    @pytest.mark.asyncio
    async def test_interim_ignored_when_tts_not_playing(self):
        h = _base_handler()
        h._is_tts_playing = False
        h._cancel_inflight_llm_response = AsyncMock()
        await h._maybe_process_interim("stop that", 0.9)
        h._cancel_inflight_llm_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_interim_barge_in_cancels_when_confident_and_playing(self):
        h = _base_handler()
        h._is_tts_playing = True
        h._cancel_inflight_llm_response = AsyncMock()
        await h._maybe_process_interim("please stop now", 0.9)
        h._cancel_inflight_llm_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_interim_barge_in_rejects_low_confidence_filler(self):
        h = _base_handler()
        h._is_tts_playing = True
        h._cancel_inflight_llm_response = AsyncMock()
        await h._maybe_process_interim("um", 0.05)
        h._cancel_inflight_llm_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_transcript_persists_and_triggers_turn(self):
        h = _base_handler()
        h._add_to_transcript = AsyncMock()
        h._complete_llm_turn_after_stt_final = AsyncMock()

        await h._process_transcript("what are your hours", 0.8)

        h._add_to_transcript.assert_awaited_once_with(
            "client", "what are your hours", "speech", 0.8
        )
        h._complete_llm_turn_after_stt_final.assert_awaited_once_with(
            "what are your hours", 0.8
        )

    @pytest.mark.asyncio
    async def test_process_transcript_ignores_empty_text(self):
        h = _base_handler()
        h._add_to_transcript = AsyncMock()
        await h._process_transcript("   ", 0.9)
        h._add_to_transcript.assert_not_called()

    @pytest.mark.asyncio
    async def test_complete_llm_turn_delegates_to_conversation_orchestrator(self):
        h = _base_handler()
        h._conversation.generate_and_stream_response = AsyncMock()

        await h._complete_llm_turn_after_stt_final("hello", 0.9)

        h._conversation.generate_and_stream_response.assert_awaited_once_with(
            "hello", 0.9, is_greeting=False
        )
        assert h._llm_response_task is None  # cleared after completion

    @pytest.mark.asyncio
    async def test_generate_and_stream_response_delegates(self):
        h = _base_handler()
        h._conversation.generate_and_stream_response = AsyncMock()
        await h.generate_and_stream_response("hi", 1.0, is_greeting=True)
        h._conversation.generate_and_stream_response.assert_awaited_once_with(
            "hi", 1.0, is_greeting=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# Barge-in: LiveKit native AudioSource queue clearing
# ─────────────────────────────────────────────────────────────────────────────

class TestCancelInflightClearsLiveKitAudioQueue:
    """
    Regression coverage for the LiveKit-transport equivalent of Twilio's
    `{"event": "clear"}` message: cancelling an in-flight LLM/TTS turn must
    also flush frames already pushed into rtc.AudioSource's own internal
    playout queue (up to 1000ms buffered), unconditionally — not only when
    publish_mulaw()'s own per-call loop happens to observe cancel.is_set().
    """

    @pytest.mark.asyncio
    async def test_clears_audio_source_queue_when_publisher_has_buffered_audio(self):
        h = _base_handler()
        h._tts_pipeline = AsyncMock()

        publisher = MagicMock()
        publisher._source = MagicMock()
        h._agent_publisher = publisher

        await h._cancel_inflight_llm_response()

        publisher._source.clear_queue.assert_called_once()
        h._tts_pipeline.cancel_current_and_clear_queue.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_noop_and_does_not_raise_when_publisher_missing(self):
        h = _base_handler()
        h._tts_pipeline = AsyncMock()
        h._agent_publisher = None  # never constructed / cancellation before any audio

        await h._cancel_inflight_llm_response()  # must not raise

        h._tts_pipeline.cancel_current_and_clear_queue.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_noop_and_does_not_raise_when_publisher_source_not_yet_created(self):
        h = _base_handler()
        h._tts_pipeline = AsyncMock()

        publisher = MagicMock()
        publisher._source = None  # connect() never ran / disconnect() already cleared it
        h._agent_publisher = publisher

        await h._cancel_inflight_llm_response()  # must not raise

    @pytest.mark.asyncio
    async def test_clear_queue_exception_is_swallowed(self):
        h = _base_handler()
        h._tts_pipeline = AsyncMock()

        publisher = MagicMock()
        publisher._source = MagicMock()
        publisher._source.clear_queue.side_effect = RuntimeError("ffi handle disposed")
        h._agent_publisher = publisher

        await h._cancel_inflight_llm_response()  # must not raise

    @pytest.mark.asyncio
    async def test_repeated_barge_ins_are_idempotent_and_error_free(self):
        h = _base_handler()
        h._tts_pipeline = AsyncMock()

        publisher = MagicMock()
        publisher._source = MagicMock()
        h._agent_publisher = publisher

        await h._cancel_inflight_llm_response()
        await h._cancel_inflight_llm_response()
        await h._cancel_inflight_llm_response()

        assert publisher._source.clear_queue.call_count == 3

    @pytest.mark.asyncio
    async def test_default_handler_has_no_publisher_and_cancel_still_completes(self):
        """
        Real pre-existing construction path (LiveKitBrowserCallHandler.__init__
        sets self._agent_publisher = None until run_livekit_browser_call() wires
        it up) — exercises the guard via the handler's own default state rather
        than a synthetic None assignment.
        """
        h = _base_handler()
        assert h._agent_publisher is None
        h._tts_pipeline = AsyncMock()

        await h._cancel_inflight_llm_response()  # must not raise

        h._tts_pipeline.cancel_current_and_clear_queue.assert_awaited_once()


# ─────────────────────────────────────────────────────────────────────────────
# TTS synthesis + playback
# ─────────────────────────────────────────────────────────────────────────────

def _fake_tts_runtime(adapter_slug: str, voice_external_id: str | None = "voice-1"):
    from app.core.agent_runtime import ResolvedTtsRuntime

    return ResolvedTtsRuntime(
        adapter_slug=adapter_slug,
        voice_external_id=voice_external_id,
        language="en",
        settings_json={},
        used_ticket_tts=False,
    )


class TestTtsSynthesisAndPlayback:
    @pytest.mark.asyncio
    async def test_prefetch_tts_audio_returns_none_for_empty_text(self):
        h = _base_handler()
        assert await h._prefetch_tts_audio({"text": "  "}) is None

    @pytest.mark.asyncio
    async def test_prefetch_tts_audio_uses_true_streaming_api_not_generate_mulaw_tts(self):
        """
        The main functional fix: _prefetch_tts_audio must resolve the
        provider's real streaming API (async_stream_synthesize) and return
        an async iterator — never buffer the whole utterance via
        generate_mulaw_tts() first (that was the pre-fix behavior).
        """
        h = _base_handler()

        async def fake_stream(**kwargs):
            yield b"\x01" * 80
            yield b"\x02" * 80

        mock_adapter = MagicMock()
        mock_adapter.async_stream_synthesize = fake_stream

        with patch(
            "app.core.agent_runtime.resolve_tts_runtime",
            return_value=_fake_tts_runtime("rime"),
        ), patch(
            "app.utils.tts_adapter.get_tts_adapter", return_value=mock_adapter
        ), patch(
            "app.voice.livekit_browser_call_handler.generate_mulaw_tts",
            new=AsyncMock(side_effect=AssertionError("must not buffer via generate_mulaw_tts")),
        ):
            result = await h._prefetch_tts_audio({"text": "Hello there"})

        assert hasattr(result, "__aiter__"), "expected an async iterator, not a whole-blob bytes object"
        chunks = [c async for c in result]
        assert chunks == [b"\x01" * 80, b"\x02" * 80]

    @pytest.mark.asyncio
    async def test_prefetch_tts_audio_falls_back_to_generate_mulaw_tts_on_setup_failure(self):
        """If provider streaming setup itself raises, fall back to whole-utterance synthesis
        rather than dropping the chunk entirely."""
        h = _base_handler()
        with patch(
            "app.core.agent_runtime.resolve_tts_runtime",
            side_effect=RuntimeError("boom"),
        ), patch(
            "app.voice.livekit_browser_call_handler.generate_mulaw_tts",
            new=AsyncMock(return_value=b"\xff" * 160),
        ) as mock_gen:
            audio = await h._prefetch_tts_audio({"text": "Hello there"})

        assert audio == b"\xff" * 160
        mock_gen.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stream_tts_chunk_drops_when_no_publisher(self):
        h = _base_handler()
        h._agent_publisher = None
        await h._stream_tts_chunk("hello", is_final=True)
        assert h.is_speaking is False
        assert h._is_tts_playing is False

    @pytest.mark.asyncio
    async def test_stream_tts_chunk_publishes_prefetched_bytes(self):
        h = _base_handler()
        publisher = MagicMock()
        publisher.connected = True
        publisher.publish_mulaw = AsyncMock()
        h._agent_publisher = publisher

        await h._stream_tts_chunk("hello", is_final=True, prefetched_bytes=b"\xff" * 160)

        publisher.publish_mulaw.assert_awaited_once()
        args, kwargs = publisher.publish_mulaw.call_args
        assert args[0] == b"\xff" * 160
        assert h.is_speaking is False  # reset in finally
        assert h._is_tts_playing is False

    @pytest.mark.asyncio
    async def test_stream_tts_chunk_publishes_incrementally_as_provider_chunks_arrive(self):
        """
        Regression test for the main functional gap: when prefetched_bytes is
        a true streaming async iterator (the TtsPipeline flow), each provider
        chunk must be published to the room as it arrives — not buffered
        until the whole utterance's audio has finished generating.
        """
        h = _base_handler()
        publisher = MagicMock()
        publisher.connected = True
        publish_calls: list[bytes] = []

        async def _record_publish(data: bytes, cancel=None):
            publish_calls.append(bytes(data))

        publisher.publish_mulaw = AsyncMock(side_effect=_record_publish)
        h._agent_publisher = publisher

        produced: list[int] = []

        async def fake_provider_iter():
            # Two full frames' worth of provider audio, produced one at a time —
            # if the implementation buffered until StopAsyncIteration, we'd only
            # ever see a single publish_mulaw call at the very end.
            from app.utils.audio_utils import MULAW_FRAME_BYTES

            for i in range(2):
                produced.append(i)
                yield bytes([i]) * MULAW_FRAME_BYTES

        with patch("app.core.agent_runtime.resolve_tts_runtime", return_value=_fake_tts_runtime("rime")):
            await h._stream_tts_chunk(
                "hello there friend", is_final=True, prefetched_bytes=fake_provider_iter()
            )

        assert publisher.publish_mulaw.await_count >= 2, (
            "expected multiple incremental publish_mulaw calls, not one whole-blob call"
        )
        from app.utils.audio_utils import MULAW_FRAME_BYTES
        assert publish_calls[0] == bytes([0]) * MULAW_FRAME_BYTES
        assert publish_calls[1] == bytes([1]) * MULAW_FRAME_BYTES

    @pytest.mark.asyncio
    async def test_stream_tts_chunk_synthesizes_when_not_prefetched(self):
        h = _base_handler()
        publisher = MagicMock()
        publisher.connected = True
        publisher.publish_mulaw = AsyncMock()
        h._agent_publisher = publisher

        async def fake_stream(**kwargs):
            yield b"\xff" * 160

        mock_adapter = MagicMock()
        mock_adapter.async_stream_synthesize = fake_stream

        with patch(
            "app.core.agent_runtime.resolve_tts_runtime",
            return_value=_fake_tts_runtime("rime"),
        ), patch("app.utils.tts_adapter.get_tts_adapter", return_value=mock_adapter):
            await h._stream_tts_chunk("hello there", is_final=False)

        publisher.publish_mulaw.assert_awaited()

    @pytest.mark.asyncio
    async def test_stream_tts_chunk_respects_cancel(self):
        h = _base_handler()
        h._tts_cancel.set()
        publisher = MagicMock()
        publisher.connected = True
        publisher.publish_mulaw = AsyncMock()
        h._agent_publisher = publisher

        await h._stream_tts_chunk("hello", prefetched_bytes=b"\xff" * 160)

        publisher.publish_mulaw.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# TTS telemetry — LiveKit-path equivalent of the Twilio/TtsStreamMixin
# authoritative-marking coverage in
# tests/voice/test_barge_in_vad_and_telemetry.py::
#   test_relayed_style_chunk_populates_tts_first_audio_and_first_playback
# ─────────────────────────────────────────────────────────────────────────────

class TestPublishMulawStreamTelemetry:
    """
    LiveKitBrowserCallHandler mixes in only CallControlMixin (see the class
    definition) -- it does NOT mix in TtsStreamMixin, and maintains its own
    `_is_tts_playing` flag plus its own `_publish_mulaw_stream`/
    `_stream_tts_chunk` implementations that mirror (but do not share code
    with) TtsStreamMixin's Twilio-path streaming. `_publish_mulaw_stream`
    now marks `VoiceTurnMetrics.tts_first_audio_mono`/`first_playback_mono`
    at the real `publisher.publish_mulaw()` frame-write point, exactly like
    `TtsStreamMixin._stream_tts_chunk`'s `send_frame()` closure does for the
    Twilio path -- this exercises the real, unmodified method (via the real
    `_stream_tts_chunk` streaming branch that calls it), not a
    reimplementation.
    """

    @pytest.mark.asyncio
    async def test_publish_mulaw_stream_marks_tts_first_audio_and_first_playback(self):
        from app.utils.audio_utils import MULAW_FRAME_BYTES
        from app.voice.metrics import VoiceTurnMetrics

        h = _base_handler()
        h._voice_metrics = VoiceTurnMetrics()
        h._voice_metrics.start_generation()
        assert h._voice_metrics.tts_first_audio_mono is None
        assert h._voice_metrics.first_playback_mono is None

        publisher = MagicMock()
        publisher.connected = True
        publish_calls: list[bytes] = []
        playing_during_publish: list[bool] = []

        async def _record_publish(data: bytes, cancel=None):
            playing_during_publish.append(h._is_tts_playing)
            publish_calls.append(bytes(data))

        publisher.publish_mulaw = AsyncMock(side_effect=_record_publish)
        h._agent_publisher = publisher

        async def fake_provider_iter():
            yield bytes([0xAB]) * MULAW_FRAME_BYTES
            yield bytes([0xCD]) * MULAW_FRAME_BYTES

        assert h._is_tts_playing is False

        with patch(
            "app.core.agent_runtime.resolve_tts_runtime", return_value=_fake_tts_runtime("rime")
        ):
            await h._stream_tts_chunk(
                "hello there this streams incrementally",
                is_final=True,
                prefetched_bytes=fake_provider_iter(),
            )

        # "cycles correctly": True for every frame actually published, then
        # reset back to False once the chunk (and the whole method) finishes
        # -- the same lifecycle TtsStreamMixin's send_frame()/finally provides
        # on the Twilio path, and the flag LiveKit's own barge-in gate
        # (_maybe_process_interim / _on_speech_started_candidate) reads.
        assert len(playing_during_publish) >= 2
        assert all(playing_during_publish)
        assert h._is_tts_playing is False

        assert h._voice_metrics.tts_first_audio_mono is not None
        assert h._voice_metrics.first_playback_mono is not None
        # Both marks derive from the same frame-transmission event (two
        # sequential time.perf_counter() calls microseconds apart), not a
        # genuine gate-wait delta.
        assert (
            abs(
                h._voice_metrics.first_playback_mono
                - h._voice_metrics.tts_first_audio_mono
            )
            < 0.001
        )

    @pytest.mark.asyncio
    async def test_publish_mulaw_stream_direct_call_marks_telemetry_on_final_padded_frame(self):
        """
        Calls `_publish_mulaw_stream` itself directly (not via
        `_stream_tts_chunk`) with a sub-frame-sized final remainder, so the
        only publish is the padded-remainder branch at the bottom of the
        method -- confirms that branch also marks telemetry, not only the
        full-frame loop branch.
        """
        from app.voice.metrics import VoiceTurnMetrics

        h = _base_handler()
        h._voice_metrics = VoiceTurnMetrics()
        h._voice_metrics.start_generation()

        publisher = MagicMock()
        publisher.publish_mulaw = AsyncMock()

        async def small_iter():
            yield b"\xAB" * 10  # smaller than MULAW_FRAME_BYTES -- only the padded tail publishes

        await h._publish_mulaw_stream(publisher, small_iter(), h._tts_cancel)

        publisher.publish_mulaw.assert_awaited_once()
        assert h._voice_metrics.tts_first_audio_mono is not None
        assert h._voice_metrics.first_playback_mono is not None


# ─────────────────────────────────────────────────────────────────────────────
# Transcript persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestAddToTranscript:
    @pytest.mark.asyncio
    async def test_no_call_session_is_a_noop(self):
        h = _base_handler()
        h.call_session = None
        # Should not raise even though db/transcript_service would fail if called.
        await h._add_to_transcript("agent", "hello")

    @pytest.mark.asyncio
    async def test_persists_via_transcript_service(self):
        h = _base_handler()
        added = MagicMock()
        with patch(
            "app.voice.livekit_browser_call_handler.transcript_service"
        ) as mock_ts:
            mock_ts.add_and_broadcast_message = AsyncMock(return_value=added)
            mock_ts.get_conversation_array = MagicMock(return_value=[{"role": "agent"}])
            await h._add_to_transcript("agent", "Hello there", "greeting")

        mock_ts.add_and_broadcast_message.assert_awaited_once()
        call_kwargs = mock_ts.add_and_broadcast_message.call_args.kwargs
        assert call_kwargs["role"] == "agent"
        assert call_kwargs["message"] == "Hello there"
        assert call_kwargs["message_type"] == "greeting"
        assert call_kwargs["call_session_id"] == h.call_session.id
        # Non-deferred write commits the denormalized transcript array.
        h.db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_deferred_write_skips_denormalized_commit(self):
        h = _base_handler()
        with patch(
            "app.voice.livekit_browser_call_handler.transcript_service"
        ) as mock_ts:
            mock_ts.add_and_broadcast_message = AsyncMock(return_value=MagicMock())
            await h._add_to_transcript("client", "hi", "speech", defer_post_write=True)

        h.db.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_swallows_persistence_errors(self):
        h = _base_handler()
        with patch(
            "app.voice.livekit_browser_call_handler.transcript_service"
        ) as mock_ts:
            mock_ts.add_and_broadcast_message = AsyncMock(side_effect=RuntimeError("db down"))
            # Must not raise.
            await h._add_to_transcript("client", "hi")


# ─────────────────────────────────────────────────────────────────────────────
# STT runtime override
# ─────────────────────────────────────────────────────────────────────────────

def test_forced_browser_stt_runtime_overrides_wire_format_only():
    resolved = ResolvedSttRuntime(
        provider_slug="deepgram",
        model_id="nova-3",
        language_code="en",
        sample_rate_hz=8000,
        encoding="MULAW",
        silence_threshold_ms=1500,
        api_config={"foo": "bar"},
    )
    forced = _forced_browser_stt_runtime(resolved)

    assert forced.sample_rate_hz == 16000
    assert forced.encoding == "LINEAR16"
    # Everything else (provider/model/language/api_config) is preserved.
    assert forced.provider_slug == "deepgram"
    assert forced.model_id == "nova-3"
    assert forced.language_code == "en"
    assert forced.api_config == {"foo": "bar"}


# ─────────────────────────────────────────────────────────────────────────────
# _LiveKitAgentAudioPublisher
# ─────────────────────────────────────────────────────────────────────────────

class TestLiveKitAgentAudioPublisher:
    @pytest.mark.asyncio
    async def test_connect_returns_false_when_livekit_disabled(self):
        publisher = _LiveKitAgentAudioPublisher("room_" + str(uuid.uuid4()))
        with patch("app.voice.livekit_browser_call_handler.settings") as mock_settings:
            mock_settings.LIVEKIT_ENABLED = False
            connected = await publisher.connect()
        assert connected is False
        assert publisher.connected is False

    @pytest.mark.asyncio
    async def test_connect_uses_distinct_identity_from_audio_subscriber(self):
        room_name = f"room_{uuid.uuid4()}"
        publisher = _LiveKitAgentAudioPublisher(room_name)

        mock_rtc = MagicMock()
        mock_room_instance = MagicMock()
        mock_room_instance.connect = AsyncMock()
        mock_room_instance.local_participant.publish_track = AsyncMock()
        mock_rtc.Room.return_value = mock_room_instance

        mock_livekit_service = MagicMock()
        mock_livekit_service.generate_agent_token.return_value = "jwt.token"
        mock_livekit_service._get_credentials.return_value = ("https://lk.example.com", "k", "s")

        with patch("app.voice.livekit_browser_call_handler.settings") as mock_settings, \
             patch.dict("sys.modules", {"livekit": MagicMock(rtc=mock_rtc)}), \
             patch("app.services.livekit_service.livekit_service", mock_livekit_service):
            mock_settings.LIVEKIT_ENABLED = True
            connected = await publisher.connect()

        assert connected is True
        assert publisher.connected is True
        mock_livekit_service.generate_agent_token.assert_called_once_with(
            room_name, identity=f"agent-tts-{room_name}"
        )

    @pytest.mark.asyncio
    async def test_publish_mulaw_noop_when_not_connected(self):
        publisher = _LiveKitAgentAudioPublisher("room_x")
        # Should return immediately without raising or touching self._source.
        await publisher.publish_mulaw(b"\xff" * 160)

    @pytest.mark.asyncio
    async def test_publish_mulaw_stops_on_cancel(self):
        publisher = _LiveKitAgentAudioPublisher("room_x")
        publisher._connected = True
        mock_source = MagicMock()
        mock_source.capture_frame = AsyncMock()
        mock_source.clear_queue = MagicMock()
        publisher._source = mock_source

        cancel = asyncio.Event()
        cancel.set()

        mock_rtc = MagicMock()
        mock_rtc.AudioFrame.create.return_value = MagicMock(data=bytearray(320))

        with patch.dict("sys.modules", {"livekit": MagicMock(rtc=mock_rtc)}):
            await publisher.publish_mulaw(b"\xff" * 320, cancel=cancel)

        mock_source.capture_frame.assert_not_called()
        mock_source.clear_queue.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_mulaw_failure_is_logged_at_warning_not_swallowed(self, caplog):
        """
        Regression: capture_frame() failures were previously caught and
        logged only at DEBUG — invisible in normal/production logs. This
        meant TTS synthesis could succeed and be reported as "turn
        complete" while the actual publish to the browser silently failed
        with zero audible output and zero trace of why. Must now be visible
        at WARNING.
        """
        publisher = _LiveKitAgentAudioPublisher("room_x")
        publisher._connected = True
        mock_source = MagicMock()
        mock_source.capture_frame = AsyncMock(side_effect=RuntimeError("source closed"))
        publisher._source = mock_source

        mock_rtc = MagicMock()
        mock_rtc.AudioFrame.create.return_value = MagicMock(data=bytearray(320))

        with patch.dict("sys.modules", {"livekit": MagicMock(rtc=mock_rtc)}), \
             caplog.at_level(logging.WARNING, logger="tgs_agent"):
            await publisher.publish_mulaw(b"\xff" * 320)

        assert any(
            "publish_mulaw failed" in record.message for record in caplog.records
        ), "expected a WARNING-level log for the swallowed publish failure"


# ─────────────────────────────────────────────────────────────────────────────
# run_livekit_browser_call — full lifecycle wiring (fully mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunLiveKitBrowserCall:
    @pytest.mark.asyncio
    async def test_noop_when_livekit_disabled(self):
        with patch("app.voice.livekit_browser_call_handler.settings") as mock_settings:
            mock_settings.LIVEKIT_ENABLED = False
            # Must return immediately without touching the DB at all.
            await run_livekit_browser_call(uuid.uuid4())

    @pytest.mark.asyncio
    async def test_aborts_cleanly_when_call_session_missing(self):
        call_session_id = uuid.uuid4()
        mock_db = MagicMock()
        mock_db.close = MagicMock()

        with patch("app.voice.livekit_browser_call_handler.settings") as mock_settings, \
             patch("app.db.session.SessionLocal", return_value=mock_db), \
             patch(
                 "app.voice.livekit_browser_call_handler._load_browser_call_context",
                 new=AsyncMock(return_value=(None, None, None)),
             ), \
             patch("app.services.call_session_service.call_session_service") as mock_svc:
            mock_settings.LIVEKIT_ENABLED = True
            await run_livekit_browser_call(call_session_id)

        # No terminal-status update should fire for a call session that was
        # never found — nothing to finalize.
        mock_svc.update_call_session_status.assert_not_called()
        mock_db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_agent_finalizes_as_failed_not_completed(self):
        """
        A loaded call_session but no agent must NOT be finalized as
        "completed" — no audio/conversation ever happened. Also matters
        because update_call_session_status(status="completed") fires
        CRM write-back scheduling; a "failed" status doesn't.
        """
        call_session_id = uuid.uuid4()
        call_session = MagicMock()
        call_session.id = call_session_id
        mock_db = MagicMock()

        with patch("app.voice.livekit_browser_call_handler.settings") as mock_settings, \
             patch("app.db.session.SessionLocal", return_value=mock_db), \
             patch(
                 "app.voice.livekit_browser_call_handler._load_browser_call_context",
                 new=AsyncMock(return_value=(call_session, None, None)),
             ), \
             patch("app.services.call_session_service.call_session_service") as mock_svc:
            mock_settings.LIVEKIT_ENABLED = True
            await run_livekit_browser_call(call_session_id)

        mock_svc.update_call_session_status.assert_called_once_with(
            mock_db, call_session_id, "failed", ended_reason="agent_join_failed"
        )

    @pytest.mark.asyncio
    async def test_publisher_connect_failure_finalizes_as_failed_not_completed(self):
        """Same as above, but the abort happens after the agent is loaded —
        TTS publisher never connects, so no greeting/conversation happened."""
        call_session_id = uuid.uuid4()
        call_session = MagicMock()
        call_session.id = call_session_id
        agent = MagicMock()
        agent.id = uuid.uuid4()

        mock_db = MagicMock()
        publisher_instance = MagicMock()
        publisher_instance.connect = AsyncMock(return_value=False)
        publisher_instance.disconnect = AsyncMock()

        with patch("app.voice.livekit_browser_call_handler.settings") as mock_settings, \
             patch("app.db.session.SessionLocal", return_value=mock_db), \
             patch(
                 "app.voice.livekit_browser_call_handler._load_browser_call_context",
                 new=AsyncMock(return_value=(call_session, agent, None)),
             ), \
             patch(
                 "app.voice.voice_orchestrator.VoiceOrchestrator",
                 return_value=MagicMock(
                     shutdown=AsyncMock(), _is_gemini_live=False, _is_openai_realtime=False,
                 ),
             ), \
             patch(
                 "app.voice.livekit_browser_call_handler._LiveKitAgentAudioPublisher",
                 return_value=publisher_instance,
             ), \
             patch("app.services.call_session_service.call_session_service") as mock_svc:
            mock_settings.LIVEKIT_ENABLED = True
            await run_livekit_browser_call(call_session_id)

        publisher_instance.connect.assert_awaited_once()
        mock_svc.update_call_session_status.assert_called_once_with(
            mock_db, call_session_id, "failed", ended_reason="agent_join_failed"
        )

    @pytest.mark.asyncio
    async def test_full_happy_path_greets_and_finalizes_on_disconnect(self):
        """
        Full lifecycle with every LiveKit/DB boundary mocked: connects the
        TTS publisher, wires STT, greets, waits for the (immediately-set)
        stop event, then tears everything down and marks the call completed.
        """
        call_session_id = uuid.uuid4()
        call_session = MagicMock()
        call_session.id = call_session_id
        call_session.call_flow_id = None
        call_session.call_metadata = {}
        agent = MagicMock()
        agent.id = uuid.uuid4()
        agent.stt_provider_slug = None
        agent.stt_model_id = None

        mock_db = MagicMock()

        publisher_instance = MagicMock()
        publisher_instance.connect = AsyncMock(return_value=True)
        publisher_instance.disconnect = AsyncMock()
        publisher_instance._room = MagicMock()

        mock_subscriber_instance = MagicMock()
        mock_subscriber_instance.run = AsyncMock()
        mock_subscriber_instance.stop = AsyncMock()

        mock_vo_instance = MagicMock()
        mock_vo_instance.shutdown = AsyncMock()
        mock_vo_instance.stt_event_bus = MagicMock()
        mock_vo_instance._on_interim = AsyncMock()
        mock_vo_instance._on_final = AsyncMock()
        # Explicit, not auto-Mock-truthy: this test exercises the normal
        # (non-Gemini-Live/non-OpenAI-Realtime) STT/TTS path — a bare
        # MagicMock() would make `_is_gemini_live` / `_is_openai_realtime`
        # auto-created truthy child Mocks, wrongly entering one of those
        # native-audio branches and awaiting an un-awaitable
        # `_start_gemini_live_session` / `_start_openai_realtime_session` Mock.
        mock_vo_instance._is_gemini_live = False
        mock_vo_instance._is_openai_realtime = False

        async def _immediately_stop_and_greet(self, *_args, **_kwargs):
            # Simulate handler.generate_and_stream_response for the greeting,
            # then let the run loop fall through by setting the stop event
            # (as if the caller disconnected right after the greeting).
            self._stop_event.set()

        with patch("app.voice.livekit_browser_call_handler.settings") as mock_settings, \
             patch("app.db.session.SessionLocal", return_value=mock_db), \
             patch(
                 "app.voice.livekit_browser_call_handler._load_browser_call_context",
                 new=AsyncMock(return_value=(call_session, agent, None)),
             ), \
             patch("app.voice.voice_orchestrator.VoiceOrchestrator", return_value=mock_vo_instance), \
             patch(
                 "app.voice.livekit_browser_call_handler._LiveKitAgentAudioPublisher",
                 return_value=publisher_instance,
             ), \
             patch("app.core.agent_runtime.resolve_stt_runtime") as mock_resolve_stt, \
             patch("app.voice.stt_pipeline.SttPipeline.from_runtime_config", return_value=MagicMock()), \
             patch(
                 "app.voice.livekit_audio_subscriber.LiveKitAudioSubscriber",
                 return_value=mock_subscriber_instance,
             ), \
             patch("app.services.call_session_service.call_session_service") as mock_svc, \
             patch(
                 "app.voice.livekit_browser_call_handler._start_browser_call_recording",
                 new=AsyncMock(return_value=None),
             ) as mock_start_rec, \
             patch(
                 "app.voice.livekit_browser_call_handler._stop_browser_call_recording",
                 new=AsyncMock(),
             ) as mock_stop_rec, \
             patch.object(
                 LiveKitBrowserCallHandler,
                 "generate_and_stream_response",
                 new=_immediately_stop_and_greet,
             ):
            mock_settings.LIVEKIT_ENABLED = True
            mock_resolve_stt.return_value = ResolvedSttRuntime(
                provider_slug="deepgram",
                model_id="nova-3",
                language_code="en",
                sample_rate_hz=8000,
                encoding="MULAW",
                silence_threshold_ms=1500,
                api_config={},
            )

            await asyncio.wait_for(run_livekit_browser_call(call_session_id), timeout=5.0)

        publisher_instance.connect.assert_awaited_once()
        publisher_instance.disconnect.assert_awaited_once()
        mock_vo_instance.shutdown.assert_awaited_once()
        mock_svc.update_call_session_status.assert_called_once_with(
            mock_db, call_session_id, "completed"
        )
        mock_db.close.assert_called_once()
        # Recording is gated (start attempted, teardown always runs even when
        # nothing was actually started — mirrors Twilio's fail-open convention).
        mock_start_rec.assert_awaited_once_with(mock_db, call_session)
        mock_stop_rec.assert_awaited_once_with(call_session_id, None)

    @pytest.mark.asyncio
    async def test_max_call_duration_cap_force_ends_and_finalizes_with_reason(self):
        """
        If the caller/room never signals _stop_event (e.g. a dead network
        with no disconnect event ever firing), the hard max-duration safety
        cap must still force the call through the normal finally cleanup —
        including actually disconnecting the publisher's LiveKit room
        connection — and finalize the call_session as "completed" with a
        distinct, identifiable ended_reason instead of hanging forever.
        """
        call_session_id = uuid.uuid4()
        call_session = MagicMock()
        call_session.id = call_session_id
        call_session.call_flow_id = None
        call_session.call_metadata = {}
        agent = MagicMock()
        agent.id = uuid.uuid4()
        agent.stt_provider_slug = None
        agent.stt_model_id = None

        mock_db = MagicMock()

        publisher_instance = MagicMock()
        publisher_instance.connect = AsyncMock(return_value=True)
        publisher_instance.disconnect = AsyncMock()
        publisher_instance._room = MagicMock()

        mock_subscriber_instance = MagicMock()
        mock_subscriber_instance.run = AsyncMock()
        mock_subscriber_instance.stop = AsyncMock()

        mock_vo_instance = MagicMock()
        mock_vo_instance.shutdown = AsyncMock()
        mock_vo_instance.stt_event_bus = MagicMock()
        mock_vo_instance._on_interim = AsyncMock()
        mock_vo_instance._on_final = AsyncMock()
        mock_vo_instance._is_gemini_live = False
        mock_vo_instance._is_openai_realtime = False

        async def _greet_but_never_stop(self, *_args, **_kwargs):
            # Unlike the happy-path test, this never sets _stop_event —
            # simulates the caller's tab/network dying without any
            # participant_disconnected/disconnected room event ever firing.
            return None

        with patch("app.voice.livekit_browser_call_handler.settings") as mock_settings, \
             patch("app.voice.livekit_browser_call_handler._MAX_CALL_DURATION_SEC", 0.01), \
             patch("app.db.session.SessionLocal", return_value=mock_db), \
             patch(
                 "app.voice.livekit_browser_call_handler._load_browser_call_context",
                 new=AsyncMock(return_value=(call_session, agent, None)),
             ), \
             patch("app.voice.voice_orchestrator.VoiceOrchestrator", return_value=mock_vo_instance), \
             patch(
                 "app.voice.livekit_browser_call_handler._LiveKitAgentAudioPublisher",
                 return_value=publisher_instance,
             ), \
             patch("app.core.agent_runtime.resolve_stt_runtime") as mock_resolve_stt, \
             patch("app.voice.stt_pipeline.SttPipeline.from_runtime_config", return_value=MagicMock()), \
             patch(
                 "app.voice.livekit_audio_subscriber.LiveKitAudioSubscriber",
                 return_value=mock_subscriber_instance,
             ), \
             patch("app.services.call_session_service.call_session_service") as mock_svc, \
             patch(
                 "app.voice.livekit_browser_call_handler._start_browser_call_recording",
                 new=AsyncMock(return_value=None),
             ), \
             patch(
                 "app.voice.livekit_browser_call_handler._stop_browser_call_recording",
                 new=AsyncMock(),
             ), \
             patch.object(
                 LiveKitBrowserCallHandler,
                 "generate_and_stream_response",
                 new=_greet_but_never_stop,
             ):
            mock_settings.LIVEKIT_ENABLED = True
            mock_resolve_stt.return_value = ResolvedSttRuntime(
                provider_slug="deepgram",
                model_id="nova-3",
                language_code="en",
                sample_rate_hz=8000,
                encoding="MULAW",
                silence_threshold_ms=1500,
                api_config={},
            )

            await asyncio.wait_for(run_livekit_browser_call(call_session_id), timeout=5.0)

        # The cap must actually force-end the call, not just stop waiting:
        # the publisher's LiveKit room connection is disconnected via the
        # same finally-block path used for a normal disconnect.
        publisher_instance.disconnect.assert_awaited_once()
        mock_vo_instance.shutdown.assert_awaited_once()
        mock_svc.update_call_session_status.assert_called_once_with(
            mock_db, call_session_id, "completed", ended_reason="max_duration_exceeded"
        )
        mock_db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_agent_join_failure_is_caught_not_raised(self):
        """An exception anywhere in the join/conversation loop must never
        propagate out as an unhandled background-task exception."""
        call_session_id = uuid.uuid4()
        mock_db = MagicMock()

        with patch("app.voice.livekit_browser_call_handler.settings") as mock_settings, \
             patch("app.db.session.SessionLocal", return_value=mock_db), \
             patch(
                 "app.voice.livekit_browser_call_handler._load_browser_call_context",
                 new=AsyncMock(side_effect=RuntimeError("DB exploded")),
             ):
            mock_settings.LIVEKIT_ENABLED = True
            # Must not raise.
            await run_livekit_browser_call(call_session_id)

        mock_db.close.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Browser-call recording (room-composite egress on the call's own native room)
# ─────────────────────────────────────────────────────────────────────────────

class TestBrowserCallRecording:
    def _call_session(self):
        call_session = MagicMock()
        call_session.id = uuid.uuid4()
        call_session.tenant_id = uuid.uuid4()
        call_session.end_time = None
        call_session.call_metadata = {}
        call_session.call_type = "web"
        call_session.recording_error = False
        return call_session

    @pytest.mark.asyncio
    async def test_start_recording_skips_when_disabled(self):
        """Gate: get_recording_enabled_for_call() returning False must skip
        entirely — no egress call, no metadata write, no commit."""
        call_session = self._call_session()
        db = MagicMock()

        with patch(
            "app.services.recording_config_service.get_recording_enabled_for_call",
            return_value=False,
        ), patch(
            "app.services.livekit_recording_service.livekit_recording_service"
        ) as mock_svc:
            egress_id = await _start_browser_call_recording(db, call_session)

        assert egress_id is None
        mock_svc.start_room_recording.assert_not_called()
        db.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_recording_writes_call_metadata_shape_matching_twilio(self):
        """
        call_metadata["recording"] must be {"egress_id":..., "gcs_path":...} —
        the exact shape call_recording_upload_service._get_recording_meta reads,
        so no changes are needed there to pick up browser-call recordings.
        """
        call_session = self._call_session()
        db = MagicMock()

        with patch(
            "app.services.recording_config_service.get_recording_enabled_for_call",
            return_value=True,
        ), patch(
            "app.services.livekit_recording_service.livekit_recording_service"
        ) as mock_rec_svc, patch(
            "app.services.s3_recording_service.build_s3_key", return_value="path/to/recording.opus"
        ):
            mock_rec_svc.start_room_recording = AsyncMock(return_value="EG_123")
            egress_id = await _start_browser_call_recording(db, call_session)

        assert egress_id == "EG_123"
        assert call_session.call_metadata["recording"] == {
            "egress_id": "EG_123",
            "gcs_path": "path/to/recording.opus",
        }
        db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_recording_is_fail_open_on_egress_error(self):
        """A recording-start failure must never raise — the call must proceed
        unaffected (fail-open, same convention as Twilio's _start_livekit_recording).

        It must, however, mark call_session.recording_error=True (root-cause
        fix): previously a genuine egress-start exception left both
        recording_s3_path and recording_error unset, making the failure
        indistinguishable from "still processing" at
        GET /api/v1/recordings/{id} — a permanent, silent 404."""
        call_session = self._call_session()
        db = MagicMock()

        with patch(
            "app.services.recording_config_service.get_recording_enabled_for_call",
            return_value=True,
        ), patch(
            "app.services.livekit_recording_service.livekit_recording_service"
        ) as mock_rec_svc:
            mock_rec_svc.start_room_recording = AsyncMock(side_effect=RuntimeError("egress API down"))
            egress_id = await _start_browser_call_recording(db, call_session)

        assert egress_id is None
        assert call_session.recording_error is True
        db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_recording_marks_error_when_egress_id_falsy_without_exception(self):
        """The third, previously-silent failure branch: start_room_recording()
        returns without raising but with a falsy egress_id. Must be logged
        and must mark recording_error=True — not just silently return None."""
        call_session = self._call_session()
        db = MagicMock()

        with patch(
            "app.services.recording_config_service.get_recording_enabled_for_call",
            return_value=True,
        ), patch(
            "app.services.livekit_recording_service.livekit_recording_service"
        ) as mock_rec_svc, patch(
            "app.services.s3_recording_service.build_s3_key", return_value="path/to/recording.opus"
        ), patch(
            "app.voice.livekit_browser_call_handler.logger"
        ) as mock_logger:
            mock_rec_svc.start_room_recording = AsyncMock(return_value="")
            egress_id = await _start_browser_call_recording(db, call_session)

        assert egress_id is None
        assert call_session.recording_error is True
        assert "recording" not in (call_session.call_metadata or {})
        db.commit.assert_called_once()
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_recording_disabled_does_not_mark_recording_error(self):
        """Recording disabled is an expected, non-error state — must not set
        recording_error (GET /recordings already reports a distinct 404 for
        it via get_recording_enabled_for_call at query time)."""
        call_session = self._call_session()
        db = MagicMock()

        with patch(
            "app.services.recording_config_service.get_recording_enabled_for_call",
            return_value=False,
        ):
            egress_id = await _start_browser_call_recording(db, call_session)

        assert egress_id is None
        assert call_session.recording_error is False
        db.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_recording_noop_when_no_egress_id(self):
        with patch(
            "app.services.livekit_recording_service.livekit_recording_service"
        ) as mock_rec_svc, patch(
            "app.services.call_recording_upload_service.schedule_recording_upload"
        ) as mock_schedule:
            await _stop_browser_call_recording(uuid.uuid4(), None)

        mock_rec_svc.stop_room_recording.assert_not_called()
        mock_schedule.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_recording_stops_egress_then_schedules_upload(self):
        call_session_id = uuid.uuid4()

        with patch(
            "app.services.livekit_recording_service.livekit_recording_service"
        ) as mock_rec_svc, patch(
            "app.services.call_recording_upload_service.schedule_recording_upload"
        ) as mock_schedule:
            mock_rec_svc.stop_room_recording = AsyncMock(return_value=True)
            await _stop_browser_call_recording(call_session_id, "EG_123")

        mock_rec_svc.stop_room_recording.assert_awaited_once_with("EG_123")
        mock_schedule.assert_called_once_with(call_session_id)

    @pytest.mark.asyncio
    async def test_stop_recording_still_schedules_upload_when_egress_stop_fails(self):
        """Fail-open: even if stop_room_recording() raises, still schedule the
        upload job — LiveKit may already be tearing down the egress on its own."""
        call_session_id = uuid.uuid4()

        with patch(
            "app.services.livekit_recording_service.livekit_recording_service"
        ) as mock_rec_svc, patch(
            "app.services.call_recording_upload_service.schedule_recording_upload"
        ) as mock_schedule:
            mock_rec_svc.stop_room_recording = AsyncMock(side_effect=RuntimeError("boom"))
            await _stop_browser_call_recording(call_session_id, "EG_123")

        mock_schedule.assert_called_once_with(call_session_id)


class TestGetRecordingEnabledForCallBrowserGating:
    """
    get_recording_enabled_for_call's phone-number lookup can never resolve
    for browser demo calls (assistant_phone_number is the literal sentinel
    "web_agent", not a real Twilio number) — call_type == "web" now
    short-circuits to a dedicated process-wide flag instead of falling
    through to the NumberConfiguration/PhoneNumber join, which would always
    return False for these calls.
    """

    def test_web_call_type_short_circuits_to_flag_when_enabled(self):
        from app.services.recording_config_service import get_recording_enabled_for_call

        call_session = MagicMock()
        call_session.call_type = "web"
        call_session.call_flow_id = None
        db = MagicMock()

        with patch(
            "app.services.recording_config_service.settings"
        ) as mock_settings:
            mock_settings.VOICE_BROWSER_DEMO_RECORDING_ENABLED = True
            assert get_recording_enabled_for_call(db, call_session) is True

        db.execute.assert_not_called()

    def test_web_call_type_short_circuits_to_flag_when_disabled_by_default(self):
        from app.services.recording_config_service import get_recording_enabled_for_call

        call_session = MagicMock()
        call_session.call_type = "web"
        call_session.call_flow_id = None
        db = MagicMock()

        with patch(
            "app.services.recording_config_service.settings"
        ) as mock_settings:
            mock_settings.VOICE_BROWSER_DEMO_RECORDING_ENABLED = False
            assert get_recording_enabled_for_call(db, call_session) is False

        db.execute.assert_not_called()

    def test_non_web_call_type_still_uses_phone_number_lookup(self):
        """Twilio calls must be entirely unaffected by the new branch."""
        from app.services.recording_config_service import get_recording_enabled_for_call

        call_session = MagicMock()
        call_session.call_type = "inbound"
        call_session.call_flow_id = None
        call_session.assistant_phone_number = "+15551234567"
        call_session.tenant_id = uuid.uuid4()
        db = MagicMock()
        db.execute.return_value.scalar_one_or_none.return_value = None

        assert get_recording_enabled_for_call(db, call_session) is False
        db.execute.assert_called_once()
