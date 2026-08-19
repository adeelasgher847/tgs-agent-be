"""
xAI Grok TTS integration unit tests — structural mirror of
tests/voice/test_hume_tts_and_barge_in.py, adapted for xAI's WebSocket
streaming protocol (one wss://api.x.ai/v1/tts connection per pipeline chunk,
JSON messages: text.delta/text.done outbound, audio.delta/audio.done/error
inbound, base64-encoded mulaw/8kHz audio — no PCM/resampling involved).

No real network calls — `websockets.asyncio.client.connect` is always mocked.

Tests cover:
  1.  XaiTtsService.stream_text_to_speech — successful streaming, outbound
      message shape (text.delta + text.done), chunk correctness.
  2.  WebSocket connection — URL/query params (codec=mulaw, sample_rate=8000,
      optimize_streaming_latency), Authorization header auth.
  3.  Incremental streaming — multiple audio.delta chunks yielded as they
      arrive, not buffered into one.
  4.  audio.delta handling — base64 decode -> raw mulaw bytes.
  5.  audio.done handling — clean stream termination.
  6.  mulaw/8kHz output end-to-end — no resampling artifacts, no
      PCMStreamDownsampler involvement.
  7.  Cancellation/barge-in — task cancellation during streaming closes the
      WS cleanly.
  8.  Errors — `error`-type message raises XaiTtsServiceError; abnormal
      connection drop before audio.done raises XaiTtsServiceError; connect
      failure/timeout; idle-recv timeout.
  9.  Empty/invalid responses — empty text short-circuits (no connect);
      malformed JSON frame / stray binary frame don't crash the loop.
  10. Provider registration — get_tts_adapter("xai") returns XaiTTSAdapter;
      TtsProviderEnum.xai exists; capabilities entry present; catalog seed
      includes "xai".
"""
from __future__ import annotations

import asyncio
import base64
import json
from unittest.mock import patch

import pytest


def _audio_delta(raw_bytes: bytes) -> str:
    return json.dumps({"type": "audio.delta", "delta": base64.b64encode(raw_bytes).decode()})


def _audio_done() -> str:
    return json.dumps({"type": "audio.done", "trace_id": "trace-123"})


def _error_msg(message: str) -> str:
    return json.dumps({"type": "error", "message": message})


class _FakeXaiWebSocket:
    """Minimal stand-in for a `websockets` client connection object."""

    def __init__(self, messages: list, recv_delay: float = 0.0):
        self._messages = list(messages)
        self.sent: list[str] = []
        self.closed = False
        self._recv_delay = recv_delay

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def recv(self):
        if self._recv_delay:
            await asyncio.sleep(self._recv_delay)
        if not self._messages:
            raise RuntimeError("no more fake messages queued")
        item = self._messages.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self) -> None:
        self.closed = True


def _patch_xai_connect(fake_ws: _FakeXaiWebSocket, captured: dict | None = None):
    async def _fake_connect(url, *args, **kwargs):
        if captured is not None:
            captured["url"] = url
            captured["kwargs"] = kwargs
        return fake_ws

    return patch("websockets.asyncio.client.connect", side_effect=_fake_connect)


def _run_stream(svc, **kwargs) -> list[bytes]:
    async def _run():
        chunks = []
        async for chunk in svc.stream_text_to_speech(**kwargs):
            chunks.append(chunk)
        return chunks

    return asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────────────────
# 1-4. Successful streaming / chunk correctness / audio.delta decode
# ─────────────────────────────────────────────────────────────────────────────


class TestXaiTtsServiceStreaming:
    def test_stream_yields_expected_mulaw_bytes(self):
        from app.services.xai_tts_service import XaiTtsService

        chunk1 = b"\xff\xfe\xfd\xfc"
        chunk2 = b"\x01\x02\x03\x04"
        messages = [_audio_delta(chunk1), _audio_delta(chunk2), _audio_done()]
        fake_ws = _FakeXaiWebSocket(messages)

        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"), _patch_xai_connect(fake_ws):
            svc = XaiTtsService()
            chunks = _run_stream(svc, text="Hello there", voice="eve")

        # No resampling: bytes come through byte-for-byte from audio.delta.
        assert chunks == [chunk1, chunk2]
        assert fake_ws.closed is True

    def test_incremental_chunks_are_yielded_as_they_arrive_not_buffered(self):
        """Multiple audio.delta messages must surface as multiple yields,
        not be collected into a single chunk before yielding."""
        from app.services.xai_tts_service import XaiTtsService

        parts = [b"aaa", b"bbb", b"ccc"]
        messages = [_audio_delta(p) for p in parts] + [_audio_done()]
        fake_ws = _FakeXaiWebSocket(messages)

        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"), _patch_xai_connect(fake_ws):
            svc = XaiTtsService()
            chunks = _run_stream(svc, text="Hi", voice="eve")

        assert chunks == parts

    def test_outbound_message_shape_is_text_delta_then_text_done(self):
        from app.services.xai_tts_service import XaiTtsService

        fake_ws = _FakeXaiWebSocket([_audio_delta(b"xyz"), _audio_done()])
        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"), _patch_xai_connect(fake_ws):
            svc = XaiTtsService()
            _run_stream(svc, text="  Hello there  ", voice="eve", speed=1.2)

        assert len(fake_ws.sent) == 2
        first = json.loads(fake_ws.sent[0])
        second = json.loads(fake_ws.sent[1])
        assert first == {"type": "text.delta", "delta": "Hello there"}  # stripped
        assert second == {"type": "text.done"}

    def test_audio_done_terminates_stream_cleanly(self):
        from app.services.xai_tts_service import XaiTtsService

        fake_ws = _FakeXaiWebSocket([_audio_delta(b"only-chunk"), _audio_done()])
        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"), _patch_xai_connect(fake_ws):
            svc = XaiTtsService()
            chunks = _run_stream(svc, text="Hi", voice="eve")

        assert chunks == [b"only-chunk"]
        assert fake_ws.closed is True


# ─────────────────────────────────────────────────────────────────────────────
# 2. WebSocket connection — URL/query params + auth header
# ─────────────────────────────────────────────────────────────────────────────


class TestXaiTtsServiceConnection:
    def test_connect_uses_mulaw_8000_and_aggressive_latency_by_default(self):
        from app.services.xai_tts_service import XaiTtsService

        fake_ws = _FakeXaiWebSocket([_audio_done()])
        captured: dict = {}
        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"), _patch_xai_connect(fake_ws, captured):
            svc = XaiTtsService()
            _run_stream(svc, text="Hi", voice="eve")

        url = captured["url"]
        assert "codec=mulaw" in url
        assert "sample_rate=8000" in url
        assert "optimize_streaming_latency=2" in url
        assert "voice=eve" in url

    def test_connect_sends_bearer_auth_header(self):
        from app.services.xai_tts_service import XaiTtsService

        fake_ws = _FakeXaiWebSocket([_audio_done()])
        captured: dict = {}
        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"), _patch_xai_connect(fake_ws, captured):
            svc = XaiTtsService()
            _run_stream(svc, text="Hi", voice="eve")

        headers = captured["kwargs"].get("additional_headers")
        assert headers == {"Authorization": "Bearer test-xai-key"}

    def test_missing_api_key_raises_before_connecting(self):
        from app.services.xai_tts_service import XaiTtsService, XaiTtsServiceError

        with patch("app.core.config.settings.XAI_API_KEY", ""), patch(
            "websockets.asyncio.client.connect"
        ) as mock_connect:
            svc = XaiTtsService()
            with pytest.raises(XaiTtsServiceError):
                _run_stream(svc, text="Hi", voice="eve")
        mock_connect.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Errors / timeouts / disconnects
# ─────────────────────────────────────────────────────────────────────────────


class TestXaiTtsServiceErrors:
    def test_error_message_raises_xai_tts_service_error(self):
        from app.services.xai_tts_service import XaiTtsService, XaiTtsServiceError

        fake_ws = _FakeXaiWebSocket([_error_msg("bad voice id")])
        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"), _patch_xai_connect(fake_ws):
            svc = XaiTtsService()
            with pytest.raises(XaiTtsServiceError, match="bad voice id"):
                _run_stream(svc, text="Hi", voice="eve")
        assert fake_ws.closed is True

    def test_abnormal_close_before_audio_done_raises(self):
        from app.services.xai_tts_service import XaiTtsService, XaiTtsServiceError

        fake_ws = _FakeXaiWebSocket([_audio_delta(b"partial"), ConnectionError("dropped")])
        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"), _patch_xai_connect(fake_ws):
            svc = XaiTtsService()
            with pytest.raises(XaiTtsServiceError):
                _run_stream(svc, text="Hi", voice="eve")
        assert fake_ws.closed is True

    def test_connect_failure_raises_xai_tts_service_error(self):
        from app.services.xai_tts_service import XaiTtsService, XaiTtsServiceError

        async def _fail_connect(url, *args, **kwargs):
            raise OSError("connection refused")

        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"), patch(
            "websockets.asyncio.client.connect", side_effect=_fail_connect
        ):
            svc = XaiTtsService()
            with pytest.raises(XaiTtsServiceError, match="failed to open"):
                _run_stream(svc, text="Hi", voice="eve")

    def test_idle_recv_timeout_raises(self):
        from app.services.xai_tts_service import (
            XaiTtsService,
            XaiTtsServiceError,
            _WS_RECV_TIMEOUT_S,
        )

        fake_ws = _FakeXaiWebSocket([_audio_delta(b"x")], recv_delay=_WS_RECV_TIMEOUT_S + 1)
        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"), _patch_xai_connect(fake_ws), patch(
            "app.services.xai_tts_service._WS_RECV_TIMEOUT_S", 0.05
        ):
            svc = XaiTtsService()
            with pytest.raises(XaiTtsServiceError, match="idle timeout"):
                _run_stream(svc, text="Hi", voice="eve")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Cancellation / barge-in
# ─────────────────────────────────────────────────────────────────────────────


class TestXaiTtsServiceCancellation:
    def test_cancelling_consumer_closes_ws_and_propagates(self):
        from app.services.xai_tts_service import XaiTtsService

        fake_ws = _FakeXaiWebSocket([_audio_delta(b"a"), _audio_delta(b"b"), _audio_done()], recv_delay=0.2)

        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"), _patch_xai_connect(fake_ws):
            svc = XaiTtsService()

            async def _consume():
                async for _ in svc.stream_text_to_speech(text="Hi", voice="eve"):
                    pass

            async def _run():
                task = asyncio.create_task(_consume())
                await asyncio.sleep(0.05)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            asyncio.run(_run())

        assert fake_ws.closed is True


# ─────────────────────────────────────────────────────────────────────────────
# 9. Empty/invalid responses
# ─────────────────────────────────────────────────────────────────────────────


class TestXaiTtsServiceEdgeCases:
    def test_empty_text_yields_nothing_and_does_not_connect(self):
        from app.services.xai_tts_service import XaiTtsService

        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"), patch(
            "websockets.asyncio.client.connect"
        ) as mock_connect:
            svc = XaiTtsService()
            chunks = _run_stream(svc, text="   ", voice="eve")

        assert chunks == []
        mock_connect.assert_not_called()

    def test_malformed_json_frame_does_not_crash_the_loop(self):
        from app.services.xai_tts_service import XaiTtsService

        fake_ws = _FakeXaiWebSocket(["not-json-{{{", _audio_delta(b"ok"), _audio_done()])
        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"), _patch_xai_connect(fake_ws):
            svc = XaiTtsService()
            chunks = _run_stream(svc, text="Hi", voice="eve")

        assert chunks == [b"ok"]

    def test_stray_binary_frame_is_skipped_not_crashed_on(self):
        from app.services.xai_tts_service import XaiTtsService

        fake_ws = _FakeXaiWebSocket([b"\x00\x01\x02", _audio_delta(b"ok"), _audio_done()])
        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"), _patch_xai_connect(fake_ws):
            svc = XaiTtsService()
            chunks = _run_stream(svc, text="Hi", voice="eve")

        assert chunks == [b"ok"]

    def test_unrecognized_message_type_is_ignored(self):
        from app.services.xai_tts_service import XaiTtsService

        fake_ws = _FakeXaiWebSocket(
            [json.dumps({"type": "audio.clear"}), _audio_delta(b"ok"), _audio_done()]
        )
        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"), _patch_xai_connect(fake_ws):
            svc = XaiTtsService()
            chunks = _run_stream(svc, text="Hi", voice="eve")

        assert chunks == [b"ok"]


# ─────────────────────────────────────────────────────────────────────────────
# synthesize() — batch/cache-path collection
# ─────────────────────────────────────────────────────────────────────────────


class TestXaiTtsServiceSynthesize:
    def test_synthesize_collects_full_bytes(self):
        from app.services.xai_tts_service import XaiTtsService

        fake_ws = _FakeXaiWebSocket([_audio_delta(b"ab"), _audio_delta(b"cd"), _audio_done()])
        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"), _patch_xai_connect(fake_ws):
            svc = XaiTtsService()
            result = asyncio.run(svc.synthesize(text="Hi", voice="eve"))

        assert result == b"abcd"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Provider registration / configuration
# ─────────────────────────────────────────────────────────────────────────────


class TestXaiTTSAdapterRegistration:
    def test_get_tts_adapter_returns_xai_adapter(self):
        from app.utils.tts_adapter import XaiTTSAdapter, get_tts_adapter

        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"):
            adapter = get_tts_adapter("xai")
        assert isinstance(adapter, XaiTTSAdapter)

    def test_adapter_construction_fails_fast_without_api_key(self):
        from app.utils.tts_adapter import get_tts_adapter

        with patch("app.core.config.settings.XAI_API_KEY", ""):
            with pytest.raises(ValueError):
                get_tts_adapter("xai")

    def test_adapter_default_voice_is_eve(self):
        from app.services.xai_tts_service import XAI_DEFAULT_VOICE
        from app.utils.tts_adapter import XaiTTSAdapter

        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"):
            adapter = XaiTTSAdapter()
        assert adapter._DEFAULT_VOICE == XAI_DEFAULT_VOICE == "eve"

    def test_adapter_stream_synthesize_raises_not_implemented(self):
        from app.utils.tts_adapter import XaiTTSAdapter

        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"):
            adapter = XaiTTSAdapter()
        with pytest.raises(NotImplementedError):
            adapter.stream_synthesize(text="hi", voice_external_id="eve")

    def test_adapter_list_voices_and_normalize(self):
        from app.utils.tts_adapter import XaiTTSAdapter

        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"):
            adapter = XaiTTSAdapter()
        voices = adapter.list_voices()
        assert any(v.get("voice_id") == "eve" for v in voices)
        normalized = adapter.normalize_voice_payload({"voice_id": "eve", "name": "Eve"})
        assert normalized["external_voice_id"] == "eve"
        assert normalized["sample_rate_hz"] == 8000

    def test_adapter_async_stream_synthesize_passthrough(self):
        from app.utils.tts_adapter import XaiTTSAdapter

        fake_ws = _FakeXaiWebSocket([_audio_delta(b"abc"), _audio_done()])
        with patch("app.core.config.settings.XAI_API_KEY", "test-xai-key"), _patch_xai_connect(fake_ws):
            adapter = XaiTTSAdapter()

            async def _run():
                chunks = []
                async for chunk in adapter.async_stream_synthesize(
                    text="Hi", voice_external_id="eve", settings_json={"speed": 1.1}
                ):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(_run())
        assert chunks == [b"abc"]

    def test_tts_provider_enum_has_xai(self):
        from app.schemas.agent import TtsProviderEnum

        assert TtsProviderEnum.xai == "xai"

    def test_capabilities_entry_present_for_xai(self):
        from app.voice.tts_provider_capabilities import get_capabilities

        caps = get_capabilities("xai")
        assert caps.provider_slug == "xai"
        assert caps.supports_streaming is True
        assert caps.supports_speaking_rate is True
        assert caps.supports_ssml is False
        assert caps.supports_streaming_session is False

    def test_catalog_seed_includes_xai(self):
        from app.services.tts_catalog_service import TTSCatalogService

        # ensure_default_provider() is DB-backed; assert against the
        # SEED_PROVIDERS class constant directly (the actual seed data,
        # not a text search over the method's source) instead of hitting a
        # real DB, mirroring how this module has no existing unit test that
        # does that.
        slugs = [spec["slug"] for spec in TTSCatalogService.SEED_PROVIDERS]
        assert "xai" in slugs
        xai_spec = next(s for s in TTSCatalogService.SEED_PROVIDERS if s["slug"] == "xai")
        assert xai_spec["is_active"] is True
        assert xai_spec["supports_streaming"] is True

    def test_verify_xai_api_key_configured_raises_when_missing(self):
        from app.services.tts_catalog_service import TTSCatalogService

        with patch("app.core.config.settings.XAI_API_KEY", ""):
            with pytest.raises(ValueError):
                TTSCatalogService.verify_xai_api_key_configured()

    def test_verify_xai_api_key_configured_passes_when_set(self):
        from app.services.tts_catalog_service import TTSCatalogService

        with patch("app.core.config.settings.XAI_API_KEY", "real-key"):
            TTSCatalogService.verify_xai_api_key_configured()  # must not raise


class TestAgentRuntimeXaiDefaults:
    def test_ticket_triad_allows_xai_without_explicit_voice(self):
        from app.core.agent_runtime import _ticket_tts_triad

        agent = type(
            "FakeAgent",
            (),
            {"tts_provider_slug": "xai", "tts_language": "en", "tts_voice_external_id": None},
        )()
        assert _ticket_tts_triad(agent) is True
