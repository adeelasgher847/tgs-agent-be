"""
Phase 3C: integration of the humanization engine + provider-capability layer
into the shared TTS preparation path.

Covers:
  1. Humanization disabled -> original TTS behavior (no overlay applied)
  2. Humanization enabled -> engine is called at the shared TtsPipeline point
  3. Humanization failure -> original text still reaches TTS
  4. Provider capability failure -> normal TTS continues
  5. ElevenLabs valid stability hint -> overlay applied
  6. ElevenLabs no hint -> existing default preserved (no override key added)
  7. Google -> no unsupported stability setting reaches the provider call
  8. Rime -> no unsupported stability setting reaches the provider call
  9. Twilio and browser handlers funnel through the same TtsPipeline /
     humanization entry point (no duplicated implementation)
  10. queue_tts() still returns before synthesis completes (incremental start)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

import app.voice.humanization_engine as humanization_engine
from app.core.agent_runtime import ResolvedTtsRuntime
from app.voice.tts_pipeline import TtsPipeline


def _fake_tts_runtime(
    adapter_slug: str, voice_external_id: str | None = "voice-1", settings_json=None
):
    return ResolvedTtsRuntime(
        adapter_slug=adapter_slug,
        voice_external_id=voice_external_id,
        language="en",
        settings_json=settings_json or {},
        used_ticket_tts=False,
    )


class _FakeHandler:
    """Minimal double for TtsPipeline._handler — records what it's called with."""

    def __init__(self):
        self._tts_cancel = asyncio.Event()
        self._current_turn_user_text = ""
        self.prefetch_calls: list[Dict[str, Any]] = []
        self.stream_calls: list[Dict[str, Any]] = []

    async def _prefetch_tts_audio(self, task: Dict[str, Any]):
        # Record a shallow copy so later task mutation doesn't affect assertions.
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
            {
                "text": text,
                "use_ssml": use_ssml,
                "is_final": is_final,
                "prefetched_bytes": prefetched_bytes,
                "pacing": pacing,
            }
        )


async def _drive_single_chunk(
    handler: _FakeHandler, text: str, user_text: str = ""
) -> None:
    handler._current_turn_user_text = user_text
    pipeline = TtsPipeline(handler)
    await pipeline.queue_tts(
        {"text": text, "chunk_id": 0, "use_ssml": False, "is_final": True}
    )
    # queue_tts spawns the chunk as an asyncio.Task; wait for it directly rather
    # than sleeping, so the test is not timing-dependent.
    task = pipeline._synthesis_tasks.get(0)
    if task is not None:
        await task


# ---------------------------------------------------------------------------
# 1 & 2: feature flag + shared entry point
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_humanization_enabled_engine_called_at_shared_point():
    handler = _FakeHandler()
    await _drive_single_chunk(
        handler,
        "Sure, let's get that scheduled for you.",
        user_text="Can you help me book an appointment please",
    )

    assert len(handler.prefetch_calls) == 1
    decision = handler.prefetch_calls[0].get("_humanization_decision")
    assert decision is not None
    assert decision.text == "Sure, let's get that scheduled for you."


@pytest.mark.asyncio
async def test_humanization_disabled_returns_neutral_decision_and_no_overlay(
    monkeypatch,
):
    monkeypatch.setattr(
        humanization_engine.settings, "VOICE_ENABLE_HUMANIZATION_ENGINE", False
    )
    handler = _FakeHandler()
    await _drive_single_chunk(
        handler,
        "Sure, let's get that scheduled for you.",
        user_text="This is unacceptable, I want a refund now",
    )

    decision = handler.prefetch_calls[0].get("_humanization_decision")
    assert decision is not None
    assert decision.tts_stability_hint is None

    from app.voice.tts_provider_capabilities import build_voice_settings_overlay

    assert build_voice_settings_overlay("elevenlabs", decision) == {}


# ---------------------------------------------------------------------------
# 3: humanization failure never blocks TTS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_humanization_engine_failure_still_reaches_tts(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated engine failure")

    monkeypatch.setattr("app.voice.tts_pipeline.analyze_response", _boom)

    handler = _FakeHandler()
    await _drive_single_chunk(handler, "Original response text stays intact.")

    assert len(handler.prefetch_calls) == 1
    task = handler.prefetch_calls[0]
    # Original text unchanged and still forwarded to synthesis.
    assert task["text"] == "Original response text stays intact."
    assert task["_humanization_decision"] is None
    # Streaming still happened — TTS was not blocked by the failure.
    assert len(handler.stream_calls) == 1


# ---------------------------------------------------------------------------
# 10: queue_tts() returns before synthesis completes (incremental streaming)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_queue_tts_returns_before_synthesis_completes():
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
        {
            "text": "Some longer response chunk here.",
            "chunk_id": 0,
            "use_ssml": False,
            "is_final": True,
        }
    )

    # queue_tts already returned; synthesis is still in flight in the background.
    await asyncio.wait_for(still_running.wait(), timeout=1.0)
    assert not release.is_set()

    release.set()
    task = pipeline._synthesis_tasks.get(0)
    if task is not None:
        await task


# ---------------------------------------------------------------------------
# 9: Twilio and browser paths share one TtsPipeline/engine implementation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_twilio_and_browser_handlers_share_same_process_chunk_implementation():
    twilio_like = _FakeHandler()
    browser_like = _FakeHandler()

    pipeline_a = TtsPipeline(twilio_like)
    pipeline_b = TtsPipeline(browser_like)

    # Exactly one TtsPipeline implementation backs both transports — not two
    # parallel humanization code paths.
    assert pipeline_a._process_chunk.__func__ is pipeline_b._process_chunk.__func__

    await _drive_single_chunk(twilio_like, "Response for the Twilio-style handler.")
    await _drive_single_chunk(browser_like, "Response for the browser-style handler.")

    assert twilio_like.prefetch_calls[0]["_humanization_decision"] is not None
    assert browser_like.prefetch_calls[0]["_humanization_decision"] is not None


# ---------------------------------------------------------------------------
# 4, 5, 6: ElevenLabs overlay wiring inside TtsStreamMixin._prefetch_tts_audio
# ---------------------------------------------------------------------------


def _mixin_handler():
    from app.voice.tts_stream_mixin import TtsStreamMixin

    h = object.__new__(TtsStreamMixin)
    h._tts_cancel = asyncio.Event()
    h._elevenlabs_prev_tts_text = ""
    h.agent = MagicMock()
    h.agent.language = "en"
    h.agent.voice_type = "female"
    h.agent.tts_voice = None
    h.db = None
    return h


@pytest.mark.asyncio
async def test_elevenlabs_valid_stability_hint_is_applied():
    h = _mixin_handler()
    decision = humanization_engine.analyze_response(
        "Let's take care of that for you.",
        user_text="This is unacceptable, I want a refund now",
    )
    assert decision.tts_stability_hint is not None

    captured: Dict[str, Any] = {}

    async def fake_stream(**kwargs):
        captured.update(kwargs)
        return
        yield  # pragma: no cover - makes this an async generator

    mock_adapter = MagicMock()
    mock_adapter.async_stream_synthesize = fake_stream

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ), patch("app.voice.tts_stream_mixin.get_tts_adapter", return_value=mock_adapter):
        result = await h._prefetch_tts_audio(
            {
                "text": "Let's take care of that for you.",
                "_humanization_decision": decision,
            }
        )
        if hasattr(result, "__aiter__"):
            async for _ in result:
                pass

    assert captured.get("settings_json", {}).get("stability") == pytest.approx(
        decision.tts_stability_hint
    )


@pytest.mark.asyncio
async def test_elevenlabs_no_hint_preserves_default_behavior():
    h = _mixin_handler()

    captured: Dict[str, Any] = {}

    async def fake_stream(**kwargs):
        captured.update(kwargs)
        return
        yield  # pragma: no cover

    mock_adapter = MagicMock()
    mock_adapter.async_stream_synthesize = fake_stream

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ), patch("app.voice.tts_stream_mixin.get_tts_adapter", return_value=mock_adapter):
        # No "_humanization_decision" key at all — simulates a chunk that
        # never went through TtsPipeline (or a decision-less fallback).
        result = await h._prefetch_tts_audio({"text": "Thanks for calling."})
        if hasattr(result, "__aiter__"):
            async for _ in result:
                pass

    assert "stability" not in captured.get("settings_json", {})


@pytest.mark.asyncio
async def test_provider_capability_failure_does_not_block_tts():
    h = _mixin_handler()

    captured: Dict[str, Any] = {}

    async def fake_stream(**kwargs):
        captured.update(kwargs)
        return
        yield  # pragma: no cover

    mock_adapter = MagicMock()
    mock_adapter.async_stream_synthesize = fake_stream

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ), patch(
        "app.voice.tts_stream_mixin.get_tts_adapter", return_value=mock_adapter
    ), patch(
        "app.voice.tts_stream_mixin.build_voice_settings_overlay",
        side_effect=RuntimeError("capability lookup exploded"),
    ):
        result = await h._prefetch_tts_audio(
            {"text": "Still works fine.", "_humanization_decision": object()}
        )
        if hasattr(result, "__aiter__"):
            async for _ in result:
                pass

    # Synthesis still happened with normal settings — no "stability" key,
    # and the call was never aborted by the capability-layer exception.
    assert captured.get("settings_json") is not None
    assert "stability" not in captured["settings_json"]


# ---------------------------------------------------------------------------
# 7: Google never receives a stability setting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_google_receives_no_stability_setting():
    h = _mixin_handler()
    decision = humanization_engine.analyze_response(
        "Let's take care of that for you.",
        user_text="This is unacceptable, I want a refund now",
    )

    captured: Dict[str, Any] = {}

    async def fake_google_stream(**kwargs):
        captured.update(kwargs)
        return
        yield  # pragma: no cover

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime",
        return_value=_fake_tts_runtime("google", voice_external_id=None),
    ), patch(
        "app.voice.tts_stream_mixin.build_voice_settings_overlay"
    ) as spy_overlay, patch(
        "app.voice.tts_stream_mixin.google_tts_service.stream_text_to_speech",
        side_effect=fake_google_stream,
    ):
        result = await h._prefetch_tts_audio(
            {
                "text": "Let's take care of that for you.",
                "_humanization_decision": decision,
            }
        )
        if hasattr(result, "__aiter__"):
            async for _ in result:
                pass

    # Google's branch never builds a provider_settings dict, so no
    # "stability"/"voice_settings" key is ever forwarded to the Google TTS
    # call — but as of V-08's speaking-rate consolidation, the SAME
    # build_voice_settings_overlay() IS now called for Google too (to
    # resolve `speaking_rate`, replacing the old duplicated inline
    # emotion->rate map), rather than never invoked.
    spy_overlay.assert_called_once_with("google", decision)
    assert "stability" not in captured
    assert "voice_settings" not in captured


# ---------------------------------------------------------------------------
# 8: Rime never receives a stability setting (capability-gated empty overlay)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rime_receives_no_stability_setting():
    h = _mixin_handler()
    decision = humanization_engine.analyze_response(
        "Got it, one moment.",
        user_text="This is unacceptable, I want a refund now",
    )

    captured: Dict[str, Any] = {}

    async def fake_stream(**kwargs):
        captured.update(kwargs)
        return
        yield  # pragma: no cover

    mock_adapter = MagicMock()
    mock_adapter.async_stream_synthesize = fake_stream

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime",
        return_value=_fake_tts_runtime("rime", voice_external_id="mistv2_Wildflower"),
    ), patch("app.voice.tts_stream_mixin.get_tts_adapter", return_value=mock_adapter):
        result = await h._prefetch_tts_audio(
            {"text": "Got it, one moment.", "_humanization_decision": decision}
        )
        if hasattr(result, "__aiter__"):
            async for _ in result:
                pass

    assert "stability" not in captured.get("settings_json", {})
