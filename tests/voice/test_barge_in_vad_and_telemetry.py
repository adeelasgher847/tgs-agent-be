"""
State-transition tests for the VAD/speech-candidate barge-in architecture fix
and the TTS-telemetry authoritative-source fix.

Covers the playback x speech matrix from the task brief:
  1. idle + backchannel-shaped text -> normal turn processed
  2. playing + known backchannel final -> TTS continues, no cancel
  3. playing + explicit command -> TTS cut, cancel + Twilio clear fire
  4. playing + unknown 2-word phrase with speech-candidate evidence -> TTS cut
  5. playing + unknown 1-word phrase -> never silently dropped (valid result)
  6. SpeechStarted alone (no interim yet) while playing -> candidate recorded,
     TTS NOT cut
  7. TTS naturally finishes between SpeechStarted and final arriving -> no
     invalid double-cancel, final still processed
  8. rapid double barge-in -> idempotent cancellation, no stale state
  9. ElevenLabs WS "relayed" path (a chunk that never itself calls
     _prefetch_tts_audio because prefetched_bytes is already an iterator, and
     never re-enters TtsPipeline's synthesis marks) still populates
     tts_first_audio/first_playback via the real _stream_tts_chunk frame path.
"""
from __future__ import annotations

import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.agent_runtime import ResolvedTtsRuntime
from app.routers.bidirectional_stream import BidirectionalStreamHandler
from app.voice.metrics import VoiceTurnMetrics


class DummyWebSocket:
    def __init__(self) -> None:
        self.sent_json: list[dict] = []

    async def send_text(self, data: str) -> None:
        pass

    async def send_json(self, data: dict) -> None:
        self.sent_json.append(data)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        pass


def _create_test_handler() -> BidirectionalStreamHandler:
    handler = BidirectionalStreamHandler(
        websocket=DummyWebSocket(),
        call_session_id=str(uuid.uuid4()),
        agent_id=str(uuid.uuid4()),
        db=None,
    )
    handler.call_session = MagicMock()
    handler.call_session.id = uuid.uuid4()
    handler.call_session.tenant_id = uuid.uuid4()
    handler.agent = MagicMock()
    handler.agent.language = "en"
    handler._enable_interim_llm = False
    handler._barge_in_min_words = 2
    handler._barge_in_min_conf = 0.26
    handler._barge_in_min_conf_1w = 0.36
    handler._barge_in_dead_zone_ms = 600
    handler._speech_candidate_window_ms = 200

    handler._cancel_inflight_llm_response = AsyncMock()
    handler._complete_llm_turn_after_stt_final = AsyncMock()
    handler._prefetch_rag_context = AsyncMock()
    handler._run_speculative_tts_prefetch = AsyncMock()
    handler._add_to_transcript = AsyncMock()
    handler._send_in_progress_status = AsyncMock()
    handler._check_and_end_call_if_goodbye = AsyncMock(return_value=False)
    handler._check_and_end_call_if_voicemail = AsyncMock(return_value=False)
    handler._check_and_handle_call_screener = AsyncMock(return_value=False)
    handler._check_and_handle_ivr_and_hold = AsyncMock(return_value=False)
    handler._check_and_handle_anti_bot = AsyncMock(return_value=False)
    handler._check_and_handle_compliance_monitoring = AsyncMock(return_value=False)
    handler._send_quick_acknowledgement = AsyncMock()
    return handler


# ── 1. idle + backchannel-shaped text -> normal turn processed ──────────────


@pytest.mark.asyncio
async def test_idle_backchannel_shaped_text_processed_as_normal_turn():
    handler = _create_test_handler()
    handler._is_tts_playing = False

    await handler._process_transcript("Hey", 1.00)

    # Not gated by the barge-in branch at all -- reaches the normal completion path.
    assert handler._complete_llm_turn_after_stt_final.call_count == 1
    assert handler._cancel_inflight_llm_response.call_count == 0


# ── 2. playing + known backchannel final -> TTS continues, no cancel ────────


@pytest.mark.asyncio
async def test_playing_known_backchannel_final_does_not_cut_tts():
    handler = _create_test_handler()
    handler._is_tts_playing = True

    await handler._process_transcript("Hey there", 1.00)

    assert handler._cancel_inflight_llm_response.call_count == 0
    assert handler._complete_llm_turn_after_stt_final.call_count == 0
    assert handler._is_tts_playing is True


# ── 3. playing + explicit command -> TTS cut, clear + LLM cancel fire ───────


@pytest.mark.asyncio
async def test_playing_explicit_command_cuts_tts_and_cancels():
    handler = _create_test_handler()
    handler._is_tts_playing = True
    handler._tts_play_start_ts = time.perf_counter() - 1.0  # dead-zone passed

    await handler._maybe_process_interim("Stop please", 0.95)

    assert handler._cancel_inflight_llm_response.call_count == 1
    assert handler._is_tts_playing is False

    # _send_twilio_clear_event lives INSIDE the real _cancel_inflight_llm_response,
    # which _create_test_handler() mocks by default -- restore the real bound
    # method here so this half of the test verifies the un-mocked implementation.
    real_handler = _create_test_handler()
    del real_handler.__dict__["_cancel_inflight_llm_response"]
    real_handler._is_tts_playing = True
    real_handler._tts_pipeline = MagicMock()
    real_handler._tts_pipeline.cancel_current_and_clear_queue = AsyncMock()
    real_handler._send_twilio_clear_event = AsyncMock()
    await real_handler._process_transcript("Stop please", 0.95)
    assert real_handler._send_twilio_clear_event.call_count == 1
    assert real_handler._tts_pipeline.cancel_current_and_clear_queue.call_count == 1


# ── 4. playing + unknown 2-word phrase with speech-candidate evidence ───────


@pytest.mark.asyncio
async def test_speech_candidate_evidence_corroborates_low_confidence_unknown_phrase():
    handler = _create_test_handler()
    handler._is_tts_playing = True
    handler._tts_play_start_ts = time.perf_counter() - 1.0

    # Without candidate evidence: marginal confidence, no shape -> suppressed.
    assert handler._should_barge_in_on_stt("blue widget", 0.30) is False
    assert handler._last_classification_reason == "weak_evidence_suppressed"

    # SpeechStarted (VAD onset) recorded while TTS is playing...
    handler._on_speech_started_candidate()
    assert handler._speech_candidate_active is True

    # ...now the SAME marginal-confidence unknown phrase is corroborated -> barge-in.
    assert handler._should_barge_in_on_stt("blue widget", 0.30) is True
    assert handler._last_classification_reason == "unknown_actionable_candidate"

    await handler._process_transcript("blue widget", 0.30)
    assert handler._cancel_inflight_llm_response.call_count == 1


# ── 5. playing + unknown 1-word phrase -> never silently dropped ────────────


@pytest.mark.asyncio
async def test_unknown_1word_phrase_never_raises_and_resolves_definitely():
    """
    At the production default (handler._barge_in_min_words == 2), a 1-word
    unclassified utterance must NEVER reach the single-word BARGE_IN branch
    -- it must always resolve to SUPPRESS with "below_word_count_threshold",
    matching pre-fix behavior exactly, regardless of confidence or
    corroborating speech-candidate evidence. The single-word branch is only
    reachable when min_words is explicitly configured to 1 (word_count alone
    must never bypass the caller-configured min_words gate). The result is
    still always a definite classification -- never a raise/crash/None.
    """
    handler = _create_test_handler()
    assert handler._barge_in_min_words == 2
    handler._is_tts_playing = True

    result_no_evidence = handler._should_barge_in_on_stt("blah", 0.40)
    assert result_no_evidence is False
    assert handler._last_classification_reason == "below_word_count_threshold"

    # Even with fresh speech-candidate evidence, a 1-word utterance still
    # cannot barge in while min_words is configured to 2.
    handler._on_speech_started_candidate()
    result_with_evidence = handler._should_barge_in_on_stt("blah", 0.40)
    assert result_with_evidence is False
    assert handler._last_classification_reason == "below_word_count_threshold"


# ── 6. SpeechStarted alone (no interim yet) while playing ───────────────────


@pytest.mark.asyncio
async def test_speech_started_alone_records_candidate_without_cancelling_tts():
    handler = _create_test_handler()
    handler._is_tts_playing = True

    handler._on_speech_started_candidate()

    assert handler._speech_candidate_active is True
    assert handler._speech_candidate_age_ms() is not None
    assert handler._cancel_inflight_llm_response.call_count == 0
    assert handler._is_tts_playing is True  # untouched


@pytest.mark.asyncio
async def test_speech_started_while_idle_is_ignored():
    handler = _create_test_handler()
    handler._is_tts_playing = False

    handler._on_speech_started_candidate()

    assert handler._speech_candidate_active is False


@pytest.mark.asyncio
async def test_speech_candidate_expires_after_window():
    handler = _create_test_handler()
    handler._is_tts_playing = True
    handler._speech_candidate_window_ms = 50

    handler._on_speech_started_candidate()
    assert handler._speech_candidate_age_ms() is not None

    handler._speech_candidate_ts = time.perf_counter() - 1.0  # 1s ago, way past 50ms
    assert handler._speech_candidate_age_ms() is None


# ── 7. TTS naturally finishes between SpeechStarted and final arriving ──────


@pytest.mark.asyncio
async def test_tts_finishes_naturally_between_speech_started_and_final_no_double_cancel():
    handler = _create_test_handler()
    handler._is_tts_playing = True

    # SpeechStarted arrives while TTS is still playing.
    handler._on_speech_started_candidate()
    assert handler._speech_candidate_active is True

    # TTS finishes naturally (send_frame's finally block flips this False)
    # BEFORE the final transcript arrives.
    handler._is_tts_playing = False

    await handler._process_transcript("hello there how are you", 0.90)

    # Not playing anymore -> normal-turn path, no barge-in cancel at all.
    assert handler._cancel_inflight_llm_response.call_count == 0
    assert handler._complete_llm_turn_after_stt_final.call_count == 1


# ── 8. rapid double barge-in -> idempotent, no stale state ──────────────────


@pytest.mark.asyncio
async def test_rapid_double_barge_in_is_idempotent_and_clears_candidate_state():
    handler = _create_test_handler()
    handler._is_tts_playing = True
    handler._tts_play_start_ts = time.perf_counter() - 1.0
    handler._on_speech_started_candidate()

    await handler._maybe_process_interim("Stop please", 0.95)
    assert handler._cancel_inflight_llm_response.call_count == 1
    assert handler._speech_candidate_active is False  # reset on resolution

    # A second, near-simultaneous interim for the same/next utterance arrives.
    # TTS is already stopped (_is_tts_playing False) so the barge-in branch
    # isn't re-entered — no duplicate cancel, no crash, no stale candidate.
    handler._is_tts_playing = True  # new turn's TTS starts speaking
    handler._tts_play_start_ts = time.perf_counter() - 1.0
    await handler._maybe_process_interim("Stop please", 0.95)
    assert handler._cancel_inflight_llm_response.call_count == 2
    assert handler._speech_candidate_active is False


# ── 9. ElevenLabs WS "relayed" path still populates telemetry ───────────────


def _fake_tts_runtime(adapter_slug: str = "elevenlabs"):
    return ResolvedTtsRuntime(
        adapter_slug=adapter_slug,
        voice_external_id="voice-1",
        language="en",
        settings_json={},
        used_ticket_tts=False,
    )


async def _fake_audio_iter():
    # Small mu-law-shaped payload -- content doesn't matter, only that frames
    # actually reach send_frame().
    yield bytes([0xFF]) * 160
    yield bytes([0xFF]) * 160


@pytest.mark.asyncio
async def test_relayed_style_chunk_populates_tts_first_audio_and_first_playback():
    """
    Simulates the ElevenLabs WS "relayed" scenario at the level that actually
    matters for telemetry correctness: a chunk whose audio bytes were NOT
    obtained via TtsPipeline._process_chunk's own synthesis marks (i.e. no
    mark_tts_first_audio()/mark_first_playback() call happened upstream of
    _stream_tts_chunk) -- prefetched_bytes is handed in directly, exactly like
    TtsPipeline does today for both the "owner" and (implicitly, via the owner
    chunk's continuous iterator) the "relayed" case. Only the real frame-
    transmission point (send_frame inside _stream_tts_chunk) should be the
    source of these marks now.
    """
    handler = _create_test_handler()
    handler._voice_metrics = VoiceTurnMetrics()
    handler._voice_metrics.start_generation()
    assert handler._voice_metrics.tts_first_audio_mono is None
    assert handler._voice_metrics.first_playback_mono is None

    handler.stream_sid = "MZ_test"
    handler._stream_sid_ready.set()

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ):
        await handler._stream_tts_chunk(
            "hello there this is a relayed chunk",
            use_ssml=False,
            is_final=True,
            prefetched_bytes=_fake_audio_iter(),
        )

    assert handler._is_tts_playing is False  # reset in the finally block after streaming
    assert handler._voice_metrics.tts_first_audio_mono is not None
    assert handler._voice_metrics.first_playback_mono is not None
    # Per the fix: both marks derive from the same frame-transmission event
    # (two sequential time.perf_counter() calls a few microseconds apart,
    # not a genuine gate-wait delta) -- assert they are effectively identical,
    # not separated by any real processing time.
    assert (
        abs(
            handler._voice_metrics.first_playback_mono
            - handler._voice_metrics.tts_first_audio_mono
        )
        < 0.001
    )


@pytest.mark.asyncio
async def test_batch_prefetched_bytes_path_populates_tts_first_audio_and_first_playback():
    """
    Regression test: an LRU phrase-cache hit (or any utterance below
    VOICE_TTS_STREAM_MIN_WORDS with no prefetch) hands _stream_tts_chunk
    plain `bytes` via prefetched_bytes, NOT an async iterator. That makes
    use_streaming_tts False regardless of word count, so the call falls
    through to the batch path (stream_mulaw_bytes_over_twilio) below the
    streaming branch -- which never runs send_frame() and previously never
    set _is_tts_playing / called mark_tts_first_audio()/mark_first_playback().
    This must now populate both marks too, from the batch path itself.
    """
    handler = _create_test_handler()
    handler._voice_metrics = VoiceTurnMetrics()
    handler._voice_metrics.start_generation()
    assert handler._voice_metrics.tts_first_audio_mono is None
    assert handler._voice_metrics.first_playback_mono is None

    handler.stream_sid = "MZ_test"
    handler._stream_sid_ready.set()

    # Plain bytes (cache-hit shape), not an async iterator -- forces the
    # batch/non-streaming code path regardless of word count.
    cached_audio = bytes([0xFF]) * 320

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ):
        await handler._stream_tts_chunk(
            "ok",
            use_ssml=False,
            is_final=True,
            prefetched_bytes=cached_audio,
        )

    assert handler._is_tts_playing is False  # reset in the finally block
    assert handler._voice_metrics.tts_first_audio_mono is not None
    assert handler._voice_metrics.first_playback_mono is not None
    assert (
        abs(
            handler._voice_metrics.first_playback_mono
            - handler._voice_metrics.tts_first_audio_mono
        )
        < 0.001
    )


@pytest.mark.asyncio
async def test_short_reply_no_prefetch_below_stream_min_words_populates_telemetry():
    """
    Same regression as above, via the other route into the batch path: no
    prefetched_bytes at all, and word_count < VOICE_TTS_STREAM_MIN_WORDS
    (default 2), so use_streaming_tts is False and the call falls through to
    generate_mulaw_tts() + stream_mulaw_bytes_over_twilio() (batch path).
    """
    handler = _create_test_handler()
    handler._voice_metrics = VoiceTurnMetrics()
    handler._voice_metrics.start_generation()

    handler.stream_sid = "MZ_test"
    handler._stream_sid_ready.set()

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ), patch(
        "app.voice.tts_stream_mixin.generate_mulaw_tts",
        new=AsyncMock(return_value=bytes([0xFF]) * 320),
    ):
        await handler._stream_tts_chunk(
            "ok",
            use_ssml=False,
            is_final=True,
            prefetched_bytes=None,
        )

    assert handler._is_tts_playing is False
    assert handler._voice_metrics.tts_first_audio_mono is not None
    assert handler._voice_metrics.first_playback_mono is not None
