"""
Phase 4C-2: real-handler pacing tests for both transports.

Covers: Twilio and LiveKit both execute the shared pause decision correctly,
the existing final-utterance silence drain is unaffected/not stacked with
the new pause, LiveKit keeps _is_tts_playing True through the pause (so
barge-in stays active), cancellation stops additional silence frames
mid-pause, and a pacing/provider failure never breaks playback.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import settings
from app.core.agent_runtime import ResolvedTtsRuntime
from app.utils.audio_utils import MULAW_FRAME_BYTES
from app.voice.humanization_engine import PacingHint, SentenceEndingType


def _fake_tts_runtime(adapter_slug: str = "rime", voice_external_id: str | None = "voice-1"):
    return ResolvedTtsRuntime(
        adapter_slug=adapter_slug,
        voice_external_id=voice_external_id,
        language="en",
        settings_json={},
        used_ticket_tts=False,
    )


def _eligible_pacing() -> PacingHint:
    return PacingHint(
        sentence_count=1,
        has_multiple_sentences=False,
        is_short_utterance=False,
        ending_type=SentenceEndingType.STATEMENT,
        has_internal_pause_opportunity=False,
    )


def _ineligible_pacing() -> PacingHint:
    return PacingHint(
        sentence_count=0,
        has_multiple_sentences=False,
        is_short_utterance=False,
        ending_type=SentenceEndingType.NONE,
        has_internal_pause_opportunity=False,
    )


def _short_acknowledgement_pacing() -> PacingHint:
    """A real sentence ending ('Okay.') but too short to warrant a pause."""
    return PacingHint(
        sentence_count=1,
        has_multiple_sentences=False,
        is_short_utterance=True,
        ending_type=SentenceEndingType.STATEMENT,
        has_internal_pause_opportunity=False,
    )


async def _audio_iter(num_frames: int = 2):
    for i in range(num_frames):
        yield bytes([0x10]) * MULAW_FRAME_BYTES


# ---------------------------------------------------------------------------
# Twilio (TtsStreamMixin)
# ---------------------------------------------------------------------------


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
    h._twilio_buffer_primed = True  # skip priming frames to simplify counting
    h._is_background_audio_enabled = lambda: False
    h._resolve_voice_volume = lambda: 1.0
    h.websocket = MagicMock()
    h.websocket.send_json = AsyncMock()
    return h


@pytest.mark.asyncio
async def test_twilio_eligible_chunk_sends_configured_extra_silence_frames(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    h = _twilio_handler()

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=_fake_tts_runtime()
    ):
        await h._stream_tts_chunk(
            "A complete sentence.",
            is_final=False,
            prefetched_bytes=_audio_iter(2),
            pacing=_eligible_pacing(),
        )

    # 2 real audio frames + 3 pause frames = 5 total media sends.
    assert h.websocket.send_json.await_count == 5
    payloads = [c.args[0] for c in h.websocket.send_json.await_args_list]
    assert all(p["event"] == "media" for p in payloads)


@pytest.mark.asyncio
async def test_twilio_pause_frames_are_strictly_appended_after_real_audio(monkeypatch):
    """
    Requirement: pacing never happens INSIDE a chunk's audio, only at the
    boundary after it. Decode each sent frame's base64 payload and confirm
    the real-audio bytes (0x10) all come first, and the silence bytes
    (0xFF) only appear as a contiguous trailing run.
    """
    import base64

    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    h = _twilio_handler()

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=_fake_tts_runtime()
    ):
        await h._stream_tts_chunk(
            "A complete sentence.",
            is_final=False,
            prefetched_bytes=_audio_iter(2),
            pacing=_eligible_pacing(),
        )

    frames = [
        base64.b64decode(c.args[0]["media"]["payload"])
        for c in h.websocket.send_json.await_args_list
    ]
    assert len(frames) == 5
    # First 2 frames are the real audio content (0x10 bytes).
    assert frames[0] == bytes([0x10]) * MULAW_FRAME_BYTES
    assert frames[1] == bytes([0x10]) * MULAW_FRAME_BYTES
    # Last 3 frames are the appended silence (0xFF), strictly after.
    assert frames[2] == bytes([0xFF]) * MULAW_FRAME_BYTES
    assert frames[3] == bytes([0xFF]) * MULAW_FRAME_BYTES
    assert frames[4] == bytes([0xFF]) * MULAW_FRAME_BYTES


@pytest.mark.asyncio
async def test_twilio_ineligible_partial_chunk_sends_no_extra_frames(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    h = _twilio_handler()

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=_fake_tts_runtime()
    ):
        await h._stream_tts_chunk(
            "mid sentence fragment",
            is_final=False,
            prefetched_bytes=_audio_iter(2),
            pacing=_ineligible_pacing(),
        )

    # Only the 2 real audio frames — no pause appended.
    assert h.websocket.send_json.await_count == 2


@pytest.mark.asyncio
async def test_twilio_short_acknowledgement_sends_no_extra_frames(monkeypatch):
    """A real sentence ending ('Okay.') but flagged short — must not pause."""
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    h = _twilio_handler()

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=_fake_tts_runtime()
    ):
        await h._stream_tts_chunk(
            "Okay.",
            is_final=False,
            prefetched_bytes=_audio_iter(1),
            pacing=_short_acknowledgement_pacing(),
        )

    assert h.websocket.send_json.await_count == 1


@pytest.mark.asyncio
async def test_twilio_pacing_introduces_no_extra_resolve_tts_runtime_calls(monkeypatch):
    """
    Phase 4C-3 requirement: pacing must not add provider/runtime-resolution
    calls. resolve_tts_runtime() is called once per _stream_tts_chunk
    invocation regardless of whether a pause is appended afterward.
    """
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)

    call_counts = {}
    for label, pacing in (("eligible", _eligible_pacing()), ("ineligible", _ineligible_pacing())):
        h = _twilio_handler()
        with patch(
            "app.voice.tts_stream_mixin.resolve_tts_runtime",
            return_value=_fake_tts_runtime(),
        ) as mock_resolve:
            await h._stream_tts_chunk(
                "A complete sentence.",
                is_final=False,
                prefetched_bytes=_audio_iter(2),
                pacing=pacing,
            )
            call_counts[label] = mock_resolve.call_count

    assert call_counts["eligible"] == call_counts["ineligible"] == 1


@pytest.mark.asyncio
async def test_twilio_config_zero_sends_no_extra_frames_even_if_eligible(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 0)
    h = _twilio_handler()

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=_fake_tts_runtime()
    ):
        await h._stream_tts_chunk(
            "A complete sentence.",
            is_final=False,
            prefetched_bytes=_audio_iter(2),
            pacing=_eligible_pacing(),
        )

    assert h.websocket.send_json.await_count == 2


@pytest.mark.asyncio
async def test_twilio_final_chunk_drain_unchanged_no_pause_stacking(monkeypatch):
    """
    The existing 60ms (3-frame) end-of-turn silence drain must still fire
    exactly as before, and the new pacing must add ZERO extra frames on top
    of it (is_final short-circuits pause_frames_for_chunk to 0).
    """
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    h = _twilio_handler()

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=_fake_tts_runtime()
    ):
        await h._stream_tts_chunk(
            "Final sentence.",
            is_final=True,
            prefetched_bytes=_audio_iter(2),
            pacing=_eligible_pacing(),  # even if content looks "eligible"
        )

    # 2 real audio frames + exactly 3 drain frames (unchanged existing
    # behavior) — NOT 2 + 3 + 3 (which would mean the new pause stacked).
    assert h.websocket.send_json.await_count == 5


@pytest.mark.asyncio
async def test_twilio_cancellation_during_pause_stops_extra_frames(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 5)
    h = _twilio_handler()

    sent = {"n": 0}
    real_send_json = h.websocket.send_json

    async def _send_and_cancel_after_real_audio(*args, **kwargs):
        sent["n"] += 1
        # Cancel right after the 2 real audio frames have gone out, before
        # any pause frame would be sent.
        if sent["n"] == 2:
            h._tts_cancel.set()
        return await real_send_json(*args, **kwargs)

    h.websocket.send_json = AsyncMock(side_effect=_send_and_cancel_after_real_audio)

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=_fake_tts_runtime()
    ):
        await h._stream_tts_chunk(
            "A complete sentence.",
            is_final=False,
            prefetched_bytes=_audio_iter(2),
            pacing=_eligible_pacing(),
        )

    # Only the 2 real frames were sent — cancellation stopped the pause
    # loop before any of the 5 configured silence frames went out.
    assert h.websocket.send_json.await_count == 2


@pytest.mark.asyncio
async def test_twilio_pause_decision_failure_does_not_break_playback(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    h = _twilio_handler()

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=_fake_tts_runtime()
    ), patch(
        "app.voice.tts_stream_mixin.pause_frames_for_chunk",
        side_effect=RuntimeError("boom"),
    ):
        await h._stream_tts_chunk(
            "A complete sentence.",
            is_final=False,
            prefetched_bytes=_audio_iter(2),
            pacing=_eligible_pacing(),
        )

    # Real audio still played despite the pause-decision blowing up.
    assert h.websocket.send_json.await_count == 2


@pytest.mark.asyncio
async def test_twilio_malformed_pacing_falls_back_to_normal_playback(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    h = _twilio_handler()

    class _NotPacing:
        pass

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=_fake_tts_runtime()
    ):
        await h._stream_tts_chunk(
            "A complete sentence.",
            is_final=False,
            prefetched_bytes=_audio_iter(2),
            pacing=_NotPacing(),  # type: ignore[arg-type]
        )

    assert h.websocket.send_json.await_count == 2


@pytest.mark.asyncio
async def test_twilio_batch_fallback_path_with_pacing_does_not_raise(monkeypatch):
    """
    Regression guard: MULAW_FRAME_BYTES must be resolvable in the
    batch/non-streaming code path too (used for short phrases / when the
    streaming setup fails), not only inside the streaming branch's own
    local import. Force that path by using text short enough that
    use_streaming_tts is False and no prefetched iterator is supplied, so
    _stream_tts_chunk falls through to generate_mulaw_tts + the Phase 4C-2
    pause code added to the batch path. Previously, this combination would
    have raised UnboundLocalError.
    """
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 2)
    monkeypatch.setattr(settings, "VOICE_TTS_STREAM_MIN_WORDS", 5)
    h = _twilio_handler()

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=_fake_tts_runtime()
    ), patch(
        "app.voice.tts_stream_mixin.generate_mulaw_tts",
        new=AsyncMock(return_value=bytes([0x10]) * MULAW_FRAME_BYTES),
    ):
        # "One word" is well under stream_min_words=5 and no prefetched_bytes
        # is given, so use_streaming_tts is False -> batch path.
        await h._stream_tts_chunk(
            "One word",
            is_final=False,
            prefetched_bytes=None,
            pacing=_eligible_pacing(),
        )

    # 1 real audio "frame" (whole utterance sent as one blob via
    # stream_mulaw_bytes_over_twilio, which itself frames it) + 2 pause
    # frames appended after — the key assertion is simply that this ran
    # to completion without raising.
    assert h.websocket.send_json.await_count == 1 + 2


@pytest.mark.asyncio
async def test_twilio_streaming_fallback_applies_humanization_overlay(monkeypatch):
    """
    Root-cause regression: when _prefetch_tts_audio did NOT already hand back
    synthesized audio (no prefetched_bytes given here, forcing this
    function's own live-synthesis branch), the ElevenLabs stability overlay
    from the turn's HumanizationDecision must still be applied -- previously
    this branch built provider_settings from only the static
    tts_runtime.settings_json, silently dropping the same stability value
    _prefetch_tts_audio would have applied, so a turn's tone/stability could
    diverge depending on which internal path happened to synthesize it.
    """
    from app.voice.humanization_engine import HumanizationDecision, PacingHint
    from app.voice.turn_signals import UserMood

    monkeypatch.setattr(settings, "VOICE_TTS_STREAM_MIN_WORDS", 2)
    h = _twilio_handler()

    captured: dict = {}

    class _FakeAdapter:
        async def async_stream_synthesize(self, text, voice_external_id, settings_json):
            captured["settings_json"] = settings_json
            yield bytes([0x10]) * MULAW_FRAME_BYTES

    decision = HumanizationDecision(
        text="A complete sentence with real content.",
        mood=UserMood.NEUTRAL,
        response_emotion="neutral",
        pacing=PacingHint(),
        acknowledgement=None,
        filler=None,
        tts_stability_hint=0.5,
    )

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime",
        return_value=_fake_tts_runtime(adapter_slug="elevenlabs"),
    ), patch(
        "app.voice.tts_stream_mixin.get_tts_adapter", return_value=_FakeAdapter()
    ):
        await h._stream_tts_chunk(
            "A complete sentence with real content.",
            is_final=False,
            prefetched_bytes=None,
            humanization_decision=decision,
        )

    assert captured.get("settings_json", {}).get("stability") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# LiveKit (LiveKitBrowserCallHandler)
# ---------------------------------------------------------------------------


def _livekit_handler():
    from app.voice.livekit_browser_call_handler import LiveKitBrowserCallHandler

    db = MagicMock()
    call_session = MagicMock()
    call_session.id = "cs-1"
    agent = MagicMock()
    agent.language = "en"
    agent.voice_type = "female"
    agent.greeting_message = "hi"
    agent.first_message = None
    h = LiveKitBrowserCallHandler(db=db, call_session=call_session, agent=agent, call_flow=None)

    publisher = MagicMock()
    publisher.connected = True
    publisher.publish_mulaw = AsyncMock()
    h._agent_publisher = publisher
    return h, publisher


@pytest.mark.asyncio
async def test_livekit_eligible_chunk_publishes_configured_extra_silence_frames(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    h, publisher = _livekit_handler()

    await h._stream_tts_chunk(
        "A complete sentence.",
        is_final=False,
        prefetched_bytes=_audio_iter(2),
        pacing=_eligible_pacing(),
    )

    # 2 real audio frames + 3 pause frames = 5 publish_mulaw calls.
    assert publisher.publish_mulaw.await_count == 5


@pytest.mark.asyncio
async def test_livekit_uses_same_pause_frame_count_as_twilio(monkeypatch):
    """Same pacing hint, same config -> both transports append the same count."""
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 4)

    h, publisher = _livekit_handler()
    await h._stream_tts_chunk(
        "A complete sentence.",
        is_final=False,
        prefetched_bytes=_audio_iter(1),
        pacing=_eligible_pacing(),
    )
    livekit_total = publisher.publish_mulaw.await_count

    twilio = _twilio_handler()
    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime", return_value=_fake_tts_runtime()
    ):
        await twilio._stream_tts_chunk(
            "A complete sentence.",
            is_final=False,
            prefetched_bytes=_audio_iter(1),
            pacing=_eligible_pacing(),
        )
    twilio_total = twilio.websocket.send_json.await_count

    assert livekit_total == twilio_total == 1 + 4


@pytest.mark.asyncio
async def test_livekit_ineligible_chunk_publishes_no_extra_frames(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    h, publisher = _livekit_handler()

    await h._stream_tts_chunk(
        "mid sentence fragment",
        is_final=False,
        prefetched_bytes=_audio_iter(2),
        pacing=_ineligible_pacing(),
    )

    assert publisher.publish_mulaw.await_count == 2


@pytest.mark.asyncio
async def test_livekit_short_acknowledgement_publishes_no_extra_frames(monkeypatch):
    """A real sentence ending ('Okay.') but flagged short — must not pause."""
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    h, publisher = _livekit_handler()

    await h._stream_tts_chunk(
        "Okay.",
        is_final=False,
        prefetched_bytes=_audio_iter(1),
        pacing=_short_acknowledgement_pacing(),
    )

    assert publisher.publish_mulaw.await_count == 1


@pytest.mark.asyncio
async def test_livekit_pacing_introduces_no_extra_resolve_tts_runtime_calls(monkeypatch):
    """
    Phase 4C-3 requirement: pacing must not add provider/runtime-resolution
    calls. As of Phase 4D-2, _stream_tts_chunk's streaming-publish branch no
    longer calls resolve_tts_runtime itself for the (now-removed) post-
    playback ElevenLabs previous_text write — that continuity value is
    captured synchronously by TtsPipeline.queue_tts() instead (see
    app.voice.tts_pipeline.TtsPipeline._last_queued_text) and no longer
    requires a runtime lookup here. Verify an eligible (paused) chunk still
    makes exactly the same number of calls as an ineligible (unpaused) one,
    i.e. pacing adds zero either way.
    """
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)

    call_counts = {}
    for label, pacing in (("eligible", _eligible_pacing()), ("ineligible", _ineligible_pacing())):
        h, publisher = _livekit_handler()
        with patch("app.core.agent_runtime.resolve_tts_runtime") as mock_resolve:
            mock_resolve.return_value = _fake_tts_runtime()
            await h._stream_tts_chunk(
                "A complete sentence.",
                is_final=False,
                prefetched_bytes=_audio_iter(2),
                pacing=pacing,
            )
            call_counts[label] = mock_resolve.call_count

    assert call_counts["eligible"] == call_counts["ineligible"] == 0


@pytest.mark.asyncio
async def test_livekit_final_chunk_gets_no_pause(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    h, publisher = _livekit_handler()

    await h._stream_tts_chunk(
        "Final sentence.",
        is_final=True,
        prefetched_bytes=_audio_iter(2),
        pacing=_eligible_pacing(),
    )

    assert publisher.publish_mulaw.await_count == 2


@pytest.mark.asyncio
async def test_livekit_is_tts_playing_stays_true_through_the_pause(monkeypatch):
    """
    Barge-in on LiveKit gates on `_is_tts_playing` (see _maybe_process_interim).
    It must remain True for every publish call, including the pause frames,
    and only flip False after _stream_tts_chunk fully returns.
    """
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    h, publisher = _livekit_handler()

    observed_flags: list[bool] = []
    real_publish = publisher.publish_mulaw

    async def _record_flag(*args, **kwargs):
        observed_flags.append(h._is_tts_playing)
        return await real_publish(*args, **kwargs)

    publisher.publish_mulaw = AsyncMock(side_effect=_record_flag)

    assert h._is_tts_playing is False
    await h._stream_tts_chunk(
        "A complete sentence.",
        is_final=False,
        prefetched_bytes=_audio_iter(2),
        pacing=_eligible_pacing(),
    )

    # 2 real + 3 pause = 5 calls, every single one observed _is_tts_playing=True.
    assert len(observed_flags) == 5
    assert all(observed_flags)
    # Only after the whole call returns does it flip back to False.
    assert h._is_tts_playing is False


@pytest.mark.asyncio
async def test_livekit_cancellation_during_pause_stops_extra_frames(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 5)
    h, publisher = _livekit_handler()

    sent = {"n": 0}

    async def _publish_and_cancel_after_real_audio(*args, **kwargs):
        sent["n"] += 1
        if sent["n"] == 2:
            h._tts_cancel.set()

    publisher.publish_mulaw = AsyncMock(side_effect=_publish_and_cancel_after_real_audio)

    await h._stream_tts_chunk(
        "A complete sentence.",
        is_final=False,
        prefetched_bytes=_audio_iter(2),
        pacing=_eligible_pacing(),
    )

    assert publisher.publish_mulaw.await_count == 2


@pytest.mark.asyncio
async def test_livekit_pause_decision_failure_does_not_break_playback(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    h, publisher = _livekit_handler()

    with patch(
        "app.voice.livekit_browser_call_handler.pause_frames_for_chunk",
        side_effect=RuntimeError("boom"),
    ):
        await h._stream_tts_chunk(
            "A complete sentence.",
            is_final=False,
            prefetched_bytes=_audio_iter(2),
            pacing=_eligible_pacing(),
        )

    assert publisher.publish_mulaw.await_count == 2


@pytest.mark.asyncio
async def test_livekit_malformed_pacing_falls_back_to_normal_playback(monkeypatch):
    monkeypatch.setattr(settings, "VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES", 3)
    h, publisher = _livekit_handler()

    class _NotPacing:
        pass

    await h._stream_tts_chunk(
        "A complete sentence.",
        is_final=False,
        prefetched_bytes=_audio_iter(2),
        pacing=_NotPacing(),  # type: ignore[arg-type]
    )

    assert publisher.publish_mulaw.await_count == 2


# ---------------------------------------------------------------------------
# No provider-specific behavior
# ---------------------------------------------------------------------------


def test_no_provider_slug_referenced_in_pacing_call_sites():
    """
    Static guard: neither handler's pause-insertion code passes a provider
    slug/adapter into the pacing decision — it only ever receives
    (pacing, is_final).
    """
    import inspect

    import app.voice.tts_stream_mixin as twilio_mod
    import app.voice.livekit_browser_call_handler as livekit_mod

    twilio_src = inspect.getsource(twilio_mod._stream_tts_chunk if hasattr(
        twilio_mod, "_stream_tts_chunk"
    ) else twilio_mod.TtsStreamMixin._stream_tts_chunk)
    livekit_src = inspect.getsource(livekit_mod.LiveKitBrowserCallHandler._stream_tts_chunk)

    for src in (twilio_src, livekit_src):
        assert "pause_frames_for_chunk(pacing, is_final)" in src
