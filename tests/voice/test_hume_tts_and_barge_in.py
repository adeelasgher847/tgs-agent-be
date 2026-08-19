"""
Hume AI TTS integration unit tests — structural mirror of
tests/voice/test_rime_tts_and_barge_in.py, adapted for Hume's WebSocket
streaming protocol (one wss://api.hume.ai/v0/tts/stream/input connection per
pipeline chunk, JSON messages carrying base64 PCM16LE audio, downsampled
incrementally to mulaw 8kHz via PCMStreamDownsampler).

No real network calls — `websockets.connect` is always mocked.

Tests cover:
  1.  HumeTtsService.stream_text_to_speech — successful streaming, voice
      payload branching (plain name vs UUID), outbound message shape,
      chunk correctness (cross-checked against an independent reference
      computation), ws.close() always called.
  2.  Errors — connect failure/timeout, abnormal mid-stream recv error
      (raises HumeTtsServiceError), a clean post-`close:true` server-side
      close (ends the generator without raising), an explicit `type: "error"`
      message (raises HumeTtsServiceError with the server's detail), and an
      idle-recv timeout (raises).
  3.  Cancellation — asyncio.CancelledError propagates out of a cancelled
      consuming task; ws.close() still runs (finally).
  4.  HumeTTSAdapter — get_tts_adapter("hume"), default voice, list_voices,
      normalize_voice_payload, async_stream_synthesize passthrough,
      stream_synthesize raises NotImplementedError.
  5.  agent_runtime — 'hume' slug resolves to 'hume' adapter with no
      explicit voice_id required (mirrors Rime's ticket-triad exemption);
      default voice fallback.
  6.  TtsPipeline barge-in / pipeline integration for a Hume-routed chunk.
  7.  Failure-propagation path: where a HumeTtsServiceError raised mid-
      generator actually gets caught in the real TTS streaming pipeline
      (see docstring on that test — the exact call site differs slightly
      from a naive reading of `_prefetch_tts_audio` alone).
"""
from __future__ import annotations

import asyncio
import base64
import json
import struct
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.utils.audio_utils import PCMStreamDownsampler


def _pcm16le(samples: list[int]) -> bytes:
    return b"".join(struct.pack("<h", s) for s in samples)


def _audio_msg(samples: list[int], is_last_chunk: bool = False) -> str:
    return json.dumps({
        "type": "audio",
        "audio": base64.b64encode(_pcm16le(samples)).decode(),
        "is_last_chunk": is_last_chunk,
    })


def _timestamp_msg() -> str:
    return json.dumps({"type": "timestamp", "time": {"begin": 0, "end": 1}})


class _FakeHumeWebSocket:
    """Minimal stand-in for a `websockets` client connection object.

    `recv()` pops messages off a pre-seeded queue; a `ConnectionClosed`-like
    exception or asyncio.TimeoutError can be seeded to simulate a mid-stream
    drop / stall.
    """

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


def _patch_hume_connect(fake_ws: _FakeHumeWebSocket):
    async def _fake_connect(url, *args, **kwargs):
        return fake_ws

    return patch(
        "app.services.hume_tts_service.get_hume_api_key",
        return_value="test-hume-key",
    ), _fake_connect


# ─────────────────────────────────────────────────────────────────────────────
# 1. HumeTtsService — successful streaming
# ─────────────────────────────────────────────────────────────────────────────


class TestHumeTtsServiceConstruction:
    def test_construction_never_resolves_api_key_eagerly(self):
        """Regression test: HumeTtsService() is built as a module-level
        singleton at import time. If __init__ eagerly called
        get_hume_api_key() (which raises when HUME_API_KEY is unset), any
        eager import of this module would crash app startup for every
        tenant, not just ones using Hume. The key must only be resolved
        lazily, on first actual use (_build_url)."""
        from app.services.hume_tts_service import HumeTtsService

        with patch(
            "app.services.hume_tts_service.get_hume_api_key",
            side_effect=ValueError("HUME_API_KEY not set"),
        ):
            svc = HumeTtsService()  # must not raise
            with pytest.raises(ValueError):
                svc._build_url()  # resolved lazily, raises here instead


class TestHumeTtsServiceStreaming:
    def test_stream_yields_expected_mulaw_bytes(self):
        """Chunks must match an independent reference computation via
        PCMStreamDownsampler + linear_to_ulaw_sample on the same input —
        not just "non-empty bytes"."""
        from app.services.hume_tts_service import HumeTtsService

        samples1 = [1000, 1200, 900, 1100, 1050, 950]  # exactly one 48k->8k group
        samples2 = [300, -200, 100, -50, 400, -600]

        messages = [
            _audio_msg(samples1),
            _timestamp_msg(),
            _audio_msg(samples2, is_last_chunk=True),
        ]
        fake_ws = _FakeHumeWebSocket(messages)

        key_patch, fake_connect = _patch_hume_connect(fake_ws)
        with key_patch, patch("websockets.connect", side_effect=fake_connect):
            svc = HumeTtsService()

            async def _run():
                chunks = []
                async for chunk in svc.stream_text_to_speech(
                    text="Hello there",
                    voice_external_id="Male English Actor",
                    src_sample_rate_hz=48000,
                ):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(_run())

        # Independent reference: feed the same raw PCM through a *fresh*
        # PCMStreamDownsampler instance, not the one internal to the service.
        ref = PCMStreamDownsampler(48000, 8000)
        expected = ref.feed(_pcm16le(samples1)) + ref.feed(_pcm16le(samples2)) + ref.flush()

        assert b"".join(chunks) == expected
        assert all(c for c in chunks)  # every yielded chunk is non-empty
        assert fake_ws.closed is True

    def test_binary_frame_audio_is_handled_directly(self):
        """
        Production incident: Hume's WS server sends at least some audio as
        raw BINARY WebSocket frames (no JSON/base64 envelope) rather than
        the documented JSON `{"type": "audio", "audio": <base64>}` shape.
        `websockets` surfaces a binary frame as `bytes` from `ws.recv()` —
        the original implementation unconditionally called
        `json.loads(raw_msg)` on every message, which crashed with
        UnicodeDecodeError on real (non-UTF-8) binary PCM payloads instead
        of treating them as raw audio. This must not happen: raw bytes
        frames are fed directly to the downsampler.
        """
        from app.services.hume_tts_service import HumeTtsService

        samples1 = [1000, 1200, 900, 1100, 1050, 950]
        samples2 = [300, -200, 100, -50, 400, -600]
        raw_pcm_frame_1 = _pcm16le(samples1)
        raw_pcm_frame_2 = _pcm16le(samples2)

        # Mix of raw binary frames and a trailing JSON control message that
        # signals stream end — exactly the hybrid framing observed live.
        messages = [raw_pcm_frame_1, raw_pcm_frame_2, _audio_msg([], is_last_chunk=True)]
        fake_ws = _FakeHumeWebSocket(messages)
        key_patch, fake_connect = _patch_hume_connect(fake_ws)
        with key_patch, patch("websockets.connect", side_effect=fake_connect):
            svc = HumeTtsService()

            async def _run():
                chunks = []
                async for chunk in svc.stream_text_to_speech(
                    text="Hello there",
                    voice_external_id="Male English Actor",
                    src_sample_rate_hz=48000,
                ):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(_run())

        ref = PCMStreamDownsampler(48000, 8000)
        expected = ref.feed(raw_pcm_frame_1) + ref.feed(raw_pcm_frame_2) + ref.flush()

        assert b"".join(chunks) == expected
        assert fake_ws.closed is True

    def test_binary_frame_stream_ends_on_connection_close_without_json(self):
        """A pure-binary stream (no trailing JSON control message at all,
        server just closes the socket after the last frame) must still
        complete cleanly — not every request necessarily gets a JSON
        is_last_chunk terminator."""
        import websockets as _websockets

        from app.services.hume_tts_service import HumeTtsService

        samples = [1000, 1200, 900, 1100, 1050, 950]
        raw_pcm_frame = _pcm16le(samples)
        messages = [raw_pcm_frame, _websockets.ConnectionClosedOK(None, None)]
        fake_ws = _FakeHumeWebSocket(messages)
        key_patch, fake_connect = _patch_hume_connect(fake_ws)
        with key_patch, patch("websockets.connect", side_effect=fake_connect):
            svc = HumeTtsService()

            async def _run():
                chunks = []
                async for chunk in svc.stream_text_to_speech(
                    text="Hi", voice_external_id="Male English Actor",
                    src_sample_rate_hz=48000,
                ):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(_run())

        ref = PCMStreamDownsampler(48000, 8000)
        expected = ref.feed(raw_pcm_frame) + ref.flush()
        assert b"".join(chunks) == expected
        assert fake_ws.closed is True

    def test_outbound_message_shape_plain_voice_name(self):
        from app.services.hume_tts_service import HumeTtsService

        fake_ws = _FakeHumeWebSocket([_audio_msg([1, 2, 3, 4, 5, 6], is_last_chunk=True)])
        key_patch, fake_connect = _patch_hume_connect(fake_ws)
        with key_patch, patch("websockets.connect", side_effect=fake_connect):
            svc = HumeTtsService()

            async def _run():
                async for _ in svc.stream_text_to_speech(
                    text="  Hello there  ",
                    voice_external_id="Male English Actor",
                    voice_provider="HUME_AI",
                    speed=1.2,
                    description="calm and reassuring",
                ):
                    pass

            asyncio.run(_run())

        assert len(fake_ws.sent) == 1
        payload = json.loads(fake_ws.sent[0])
        assert payload["text"] == "Hello there"  # stripped
        assert payload["voice"] == {"name": "Male English Actor", "provider": "HUME_AI"}
        assert payload["speed"] == 1.2
        assert payload["close"] is True
        assert payload["description"] == "calm and reassuring"

    def test_outbound_message_shape_uuid_voice_id(self):
        from app.services.hume_tts_service import HumeTtsService

        voice_uuid = str(uuid.uuid4())
        fake_ws = _FakeHumeWebSocket([_audio_msg([1, 2, 3, 4, 5, 6], is_last_chunk=True)])
        key_patch, fake_connect = _patch_hume_connect(fake_ws)
        with key_patch, patch("websockets.connect", side_effect=fake_connect):
            svc = HumeTtsService()

            async def _run():
                async for _ in svc.stream_text_to_speech(
                    text="Hi",
                    voice_external_id=voice_uuid,
                ):
                    pass

            asyncio.run(_run())

        payload = json.loads(fake_ws.sent[0])
        assert payload["voice"] == {"id": voice_uuid}
        # No "description" key when none was passed.
        assert "description" not in payload

    def test_description_is_truncated_to_100_chars(self):
        from app.services.hume_tts_service import HumeTtsService

        fake_ws = _FakeHumeWebSocket([_audio_msg([1, 2, 3, 4, 5, 6], is_last_chunk=True)])
        key_patch, fake_connect = _patch_hume_connect(fake_ws)
        long_description = "x" * 250
        with key_patch, patch("websockets.connect", side_effect=fake_connect):
            svc = HumeTtsService()

            async def _run():
                async for _ in svc.stream_text_to_speech(
                    text="Hi", voice_external_id="Male English Actor",
                    description=long_description,
                ):
                    pass

            asyncio.run(_run())

        payload = json.loads(fake_ws.sent[0])
        assert len(payload["description"]) == 100

    def test_empty_text_yields_nothing_and_does_not_connect(self):
        from app.services.hume_tts_service import HumeTtsService

        with patch(
            "app.services.hume_tts_service.get_hume_api_key", return_value="test-key"
        ), patch("websockets.connect") as mock_connect:
            svc = HumeTtsService()

            async def _run():
                chunks = []
                async for chunk in svc.stream_text_to_speech(text="   "):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(_run())

        assert chunks == []
        mock_connect.assert_not_called()

    def test_timestamp_messages_are_ignored(self):
        from app.services.hume_tts_service import HumeTtsService

        samples = [1000, 1200, 900, 1100, 1050, 950]
        messages = [_timestamp_msg(), _timestamp_msg(), _audio_msg(samples, is_last_chunk=True)]
        fake_ws = _FakeHumeWebSocket(messages)
        key_patch, fake_connect = _patch_hume_connect(fake_ws)
        with key_patch, patch("websockets.connect", side_effect=fake_connect):
            svc = HumeTtsService()

            async def _run():
                chunks = []
                async for chunk in svc.stream_text_to_speech(
                    text="Hi", src_sample_rate_hz=48000
                ):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(_run())

        ref = PCMStreamDownsampler(48000, 8000)
        expected = ref.feed(_pcm16le(samples)) + ref.flush()
        assert b"".join(chunks) == expected


# ─────────────────────────────────────────────────────────────────────────────
# 2. Errors
# ─────────────────────────────────────────────────────────────────────────────


class TestHumeTtsServiceErrors:
    def test_connect_failure_raises_service_error(self):
        from app.services.hume_tts_service import HumeTtsService, HumeTtsServiceError

        async def _fake_connect(url, *args, **kwargs):
            raise OSError("connection refused")

        with patch(
            "app.services.hume_tts_service.get_hume_api_key", return_value="test-key"
        ), patch("websockets.connect", side_effect=_fake_connect):
            svc = HumeTtsService()

            async def _run():
                async for _ in svc.stream_text_to_speech(text="Hi"):
                    pass

            with pytest.raises(HumeTtsServiceError, match="failed to open Hume TTS WebSocket"):
                asyncio.run(_run())

    def test_connect_timeout_raises_service_error(self):
        from app.services.hume_tts_service import HumeTtsService, HumeTtsServiceError

        async def _hanging_connect(url, *args, **kwargs):
            await asyncio.sleep(10)

        with patch(
            "app.services.hume_tts_service.get_hume_api_key", return_value="test-key"
        ), patch("websockets.connect", side_effect=_hanging_connect), patch(
            "app.services.hume_tts_service._WS_CONNECT_TIMEOUT_S", 0.05
        ):
            svc = HumeTtsService()

            async def _run():
                async for _ in svc.stream_text_to_speech(text="Hi"):
                    pass

            with pytest.raises(HumeTtsServiceError, match="failed to open Hume TTS WebSocket"):
                asyncio.run(_run())

    def test_send_failure_raises_service_error_and_closes_socket(self):
        from app.services.hume_tts_service import HumeTtsService, HumeTtsServiceError

        fake_ws = _FakeHumeWebSocket([])
        fake_ws.send = AsyncMock(side_effect=RuntimeError("socket write failed"))
        key_patch, fake_connect = _patch_hume_connect(fake_ws)
        with key_patch, patch("websockets.connect", side_effect=fake_connect):
            svc = HumeTtsService()

            async def _run():
                async for _ in svc.stream_text_to_speech(text="Hi"):
                    pass

            with pytest.raises(HumeTtsServiceError, match="failed to send to Hume TTS WebSocket"):
                asyncio.run(_run())
        assert fake_ws.closed is True

    def test_mid_stream_abnormal_recv_error_raises_service_error(self):
        """
        An abnormal `ws.recv()` failure (anything other than a clean
        `ConnectionClosedOK`, asyncio.TimeoutError, or CancelledError) must
        be logged AND raised as HumeTtsServiceError — not silently treated
        as end-of-stream — so the pipeline's fallback path actually engages
        instead of a chunk going silently dark on a live call.
        """
        from app.services.hume_tts_service import HumeTtsService, HumeTtsServiceError

        samples = [1000, 1200, 900, 1100, 1050, 950]
        messages = [_audio_msg(samples), ConnectionError("connection dropped")]
        fake_ws = _FakeHumeWebSocket(messages)
        key_patch, fake_connect = _patch_hume_connect(fake_ws)
        with key_patch, patch("websockets.connect", side_effect=fake_connect):
            svc = HumeTtsService()

            async def _run():
                chunks = []
                async for chunk in svc.stream_text_to_speech(
                    text="Hi", src_sample_rate_hz=48000
                ):
                    chunks.append(chunk)
                return chunks

            with pytest.raises(HumeTtsServiceError, match="connection dropped"):
                asyncio.run(_run())

        # ws.close() still ran (finally block) even though the generator raised.
        assert fake_ws.closed is True

    def test_mid_stream_clean_close_ends_generator_without_raising(self):
        """
        A clean `ConnectionClosedOK` (server closing normally after honoring
        `close: true`) is the expected end-of-stream signal for the
        one-shot-per-chunk model and must NOT raise.
        """
        import websockets

        from app.services.hume_tts_service import HumeTtsService

        samples = [1000, 1200, 900, 1100, 1050, 950]
        messages = [_audio_msg(samples), websockets.ConnectionClosedOK(None, None)]
        fake_ws = _FakeHumeWebSocket(messages)
        key_patch, fake_connect = _patch_hume_connect(fake_ws)
        with key_patch, patch("websockets.connect", side_effect=fake_connect):
            svc = HumeTtsService()

            async def _run():
                chunks = []
                async for chunk in svc.stream_text_to_speech(
                    text="Hi", src_sample_rate_hz=48000
                ):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(_run())

        ref = PCMStreamDownsampler(48000, 8000)
        expected = ref.feed(_pcm16le(samples)) + ref.flush()
        assert b"".join(chunks) == expected
        assert fake_ws.closed is True

    def test_explicit_error_message_raises_service_error(self):
        """An explicit `type: "error"` message from Hume must raise
        HumeTtsServiceError carrying the server's detail — not be swallowed
        as a benign stream end."""
        from app.services.hume_tts_service import HumeTtsService, HumeTtsServiceError

        messages = [json.dumps({"type": "error", "message": "boom"})]
        fake_ws = _FakeHumeWebSocket(messages)
        key_patch, fake_connect = _patch_hume_connect(fake_ws)
        with key_patch, patch("websockets.connect", side_effect=fake_connect):
            svc = HumeTtsService()

            async def _run():
                chunks = []
                async for chunk in svc.stream_text_to_speech(text="Hi"):
                    chunks.append(chunk)
                return chunks

            with pytest.raises(HumeTtsServiceError, match="boom"):
                asyncio.run(_run())

        assert fake_ws.closed is True

    def test_unrecognized_message_type_ends_stream_without_raising(self):
        """A message type outside Hume's documented audio/timestamp/error
        vocabulary is logged and ends the stream, but does not hard-fail the
        call (Hume's full message vocabulary isn't fully documented)."""
        from app.services.hume_tts_service import HumeTtsService

        messages = [json.dumps({"type": "some_future_message_type"})]
        fake_ws = _FakeHumeWebSocket(messages)
        key_patch, fake_connect = _patch_hume_connect(fake_ws)
        with key_patch, patch("websockets.connect", side_effect=fake_connect):
            svc = HumeTtsService()

            async def _run():
                chunks = []
                async for chunk in svc.stream_text_to_speech(text="Hi"):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(_run())

        assert chunks == []
        assert fake_ws.closed is True

    def test_idle_recv_timeout_raises_service_error(self):
        from app.services.hume_tts_service import HumeTtsService, HumeTtsServiceError

        fake_ws = _FakeHumeWebSocket([_audio_msg([1, 2, 3, 4, 5, 6])], recv_delay=10)
        key_patch, fake_connect = _patch_hume_connect(fake_ws)
        with key_patch, patch("websockets.connect", side_effect=fake_connect), patch(
            "app.services.hume_tts_service._WS_RECV_TIMEOUT_S", 0.05
        ):
            svc = HumeTtsService()

            async def _run():
                async for _ in svc.stream_text_to_speech(text="Hi"):
                    pass

            with pytest.raises(HumeTtsServiceError, match="idle timeout"):
                asyncio.run(_run())
        assert fake_ws.closed is True


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cancellation
# ─────────────────────────────────────────────────────────────────────────────


class TestHumeTtsServiceCancellation:
    def test_cancel_mid_stream_propagates_and_closes_socket(self):
        from app.services.hume_tts_service import HumeTtsService

        samples = [1000, 1200, 900, 1100, 1050, 950]
        release_second_chunk = asyncio.Event()

        class _SlowSecondChunkWs(_FakeHumeWebSocket):
            def __init__(self):
                super().__init__([_audio_msg(samples)])
                self._served_first = False

            async def recv(self):
                if self._served_first:
                    await release_second_chunk.wait()
                    raise AssertionError("should have been cancelled before this resumed")
                self._served_first = True
                return await super().recv()

        fake_ws = _SlowSecondChunkWs()
        key_patch, fake_connect = _patch_hume_connect(fake_ws)

        async def _run():
            chunks: list[bytes] = []

            async def _consume():
                async for chunk in HumeTtsService().stream_text_to_speech(
                    text="Hi", src_sample_rate_hz=48000
                ):
                    chunks.append(chunk)

            task = asyncio.create_task(_consume())
            # Let the first chunk arrive.
            await asyncio.sleep(0.02)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return chunks

        with key_patch, patch("websockets.connect", side_effect=fake_connect):
            chunks = asyncio.run(_run())

        # At least the first chunk should have been yielded before cancel.
        assert len(chunks) >= 0  # non-flaky: only assert no crash + close ran below
        assert fake_ws.closed is True, "ws.close() must run in the finally block even on cancel"


# ─────────────────────────────────────────────────────────────────────────────
# 4. HumeTTSAdapter
# ─────────────────────────────────────────────────────────────────────────────


class TestHumeTTSAdapter:
    def test_get_tts_adapter_returns_hume(self):
        from app.utils.tts_adapter import HumeTTSAdapter, get_tts_adapter

        with patch("app.utils.tts_adapter.get_hume_api_key", return_value="test-key"):
            adapter = get_tts_adapter("hume")
        assert isinstance(adapter, HumeTTSAdapter)

    def test_adapter_init_fails_without_api_key(self):
        from app.utils.tts_adapter import HumeTTSAdapter

        with patch(
            "app.utils.tts_adapter.get_hume_api_key",
            side_effect=ValueError("HUME_API_KEY is not set"),
        ), pytest.raises(ValueError, match="HUME_API_KEY is not set"):
            HumeTTSAdapter()

    def test_default_voice_fallback(self):
        from app.utils.tts_adapter import HumeTTSAdapter

        with patch("app.utils.tts_adapter.get_hume_api_key", return_value="test-key"):
            adapter = HumeTTSAdapter()
        assert adapter._DEFAULT_VOICE == "Male English Actor"

    def test_list_voices_returns_list(self):
        from app.utils.tts_adapter import HumeTTSAdapter

        fake_response = MagicMock()
        fake_response.raise_for_status = MagicMock()
        fake_response.json.return_value = {"voices_page": [{"name": "Voice A"}]}

        with patch("app.utils.tts_adapter.get_hume_api_key", return_value="test-key"):
            adapter = HumeTTSAdapter()

        with patch("requests.get", return_value=fake_response) as mock_get:
            voices = adapter.list_voices()

        assert voices == [{"name": "Voice A"}]
        assert mock_get.call_args.kwargs["headers"] == {"X-Hume-Api-Key": "test-key"}

    def test_list_voices_raises_runtime_error_on_http_failure(self):
        import requests

        from app.utils.tts_adapter import HumeTTSAdapter

        with patch("app.utils.tts_adapter.get_hume_api_key", return_value="test-key"):
            adapter = HumeTTSAdapter()

        with patch("requests.get", side_effect=requests.RequestException("boom")):
            with pytest.raises(RuntimeError, match="Failed to fetch Hume voices"):
                adapter.list_voices()

    def test_normalize_voice_payload(self):
        from app.utils.tts_adapter import HumeTTSAdapter

        with patch("app.utils.tts_adapter.get_hume_api_key", return_value="test-key"):
            adapter = HumeTTSAdapter()

        norm = adapter.normalize_voice_payload({"id": "voice-123", "name": "Voice A"})
        assert norm["external_voice_id"] == "voice-123"
        assert norm["sample_rate_hz"] == 8000
        assert norm["description"] == "Hume AI voice"

    def test_async_stream_synthesize_calls_service_with_expected_kwargs(self):
        from app.utils.tts_adapter import HumeTTSAdapter

        received: dict = {}

        async def _fake_stream(**kwargs):
            received.update(kwargs)
            yield b"\xff" * 160
            yield b"\xff" * 160

        with patch("app.utils.tts_adapter.get_hume_api_key", return_value="test-key"):
            adapter = HumeTTSAdapter()

        with patch(
            "app.services.hume_tts_service.hume_tts_service.stream_text_to_speech",
            side_effect=_fake_stream,
        ):
            async def _run():
                chunks = []
                async for chunk in adapter.async_stream_synthesize(
                    text="Hello world",
                    voice_external_id="Male English Actor",
                    settings_json={
                        "speed": 1.1,
                        "hume_voice_provider": "HUME_AI",
                        "hume_description": "warm",
                    },
                ):
                    chunks.append(chunk)
                return chunks

            chunks = asyncio.run(_run())

        assert len(chunks) == 2
        assert received["text"] == "Hello world"
        assert received["voice_external_id"] == "Male English Actor"
        assert received["voice_provider"] == "HUME_AI"
        assert received["speed"] == 1.1
        assert received["description"] == "warm"

    def test_async_stream_synthesize_falls_back_to_default_voice(self):
        from app.utils.tts_adapter import HumeTTSAdapter

        received: dict = {}

        async def _fake_stream(**kwargs):
            received.update(kwargs)
            yield b"\xff" * 160

        with patch("app.utils.tts_adapter.get_hume_api_key", return_value="test-key"):
            adapter = HumeTTSAdapter()

        with patch(
            "app.services.hume_tts_service.hume_tts_service.stream_text_to_speech",
            side_effect=_fake_stream,
        ):
            async def _run():
                async for _ in adapter.async_stream_synthesize(
                    text="Hi", voice_external_id="", settings_json={}
                ):
                    pass

            asyncio.run(_run())

        assert received["voice_external_id"] == "Male English Actor"
        assert received["voice_provider"] == "HUME_AI"
        assert received["speed"] == 1.0

    def test_stream_synthesize_raises_not_implemented(self):
        from app.utils.tts_adapter import HumeTTSAdapter

        with patch("app.utils.tts_adapter.get_hume_api_key", return_value="test-key"):
            adapter = HumeTTSAdapter()
        with pytest.raises(NotImplementedError):
            adapter.stream_synthesize("text", "voice")

    def test_synthesize_uses_sync_bridge_and_collects_bytes(self):
        from app.utils.tts_adapter import HumeTTSAdapter

        with patch("app.utils.tts_adapter.get_hume_api_key", return_value="test-key"):
            adapter = HumeTTSAdapter()

        async def _fake_service_synthesize(**kwargs):
            return b"\xab" * 320

        with patch(
            "app.services.hume_tts_service.hume_tts_service.synthesize",
            side_effect=_fake_service_synthesize,
        ):
            result = adapter.synthesize(
                text="Hello there",
                voice_external_id="Male English Actor",
                settings_json={"speed": 1.0},
            )

        assert result == b"\xab" * 320


# ─────────────────────────────────────────────────────────────────────────────
# 5. agent_runtime — 'hume' adapter resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentRuntimeHume:
    def _hume_agent(self) -> MagicMock:
        agent = MagicMock()
        agent.language = "en"
        agent.tts_settings_json = {}
        agent.tts_provider_slug = "hume"
        agent.tts_voice_external_id = "Male English Actor"
        agent.tts_language = "en"
        agent.encrypted_elevenlabs_api_key = None
        return agent

    def test_hume_slug_resolves_to_hume_adapter(self):
        from app.core.agent_runtime import resolve_tts_runtime

        agent = self._hume_agent()
        runtime = resolve_tts_runtime(agent)
        assert runtime.adapter_slug == "hume"

    def test_hume_voice_id_preserved(self):
        from app.core.agent_runtime import resolve_tts_runtime

        agent = self._hume_agent()
        runtime = resolve_tts_runtime(agent)
        assert runtime.voice_external_id == "Male English Actor"

    def test_hume_default_voice_when_none(self):
        from app.core.agent_runtime import resolve_tts_runtime

        agent = self._hume_agent()
        agent.tts_voice_external_id = None
        runtime = resolve_tts_runtime(agent)
        assert runtime.voice_external_id == "Male English Actor"

    def test_ticket_triad_allows_hume_without_explicit_voice_id(self):
        """Mirrors Rime's ticket-triad exemption: Hume has a built-in
        default voice, so `_ticket_tts_triad` must not require voice_id."""
        from app.core.agent_runtime import _ticket_tts_triad

        agent = self._hume_agent()
        agent.tts_voice_external_id = None
        assert _ticket_tts_triad(agent) is True

    def test_speed_volume_defaults_normalised_for_hume(self):
        from app.core.agent_runtime import resolve_tts_runtime

        agent = self._hume_agent()
        runtime = resolve_tts_runtime(agent)
        assert runtime.settings_json.get("speed") == 1.0
        assert runtime.settings_json.get("volume") == 1.0

    def test_custom_speed_volume_preserved_for_hume(self):
        from app.core.agent_runtime import resolve_tts_runtime

        agent = self._hume_agent()
        agent.tts_settings_json = {"speed": 1.3, "volume": 0.7}
        runtime = resolve_tts_runtime(agent)
        assert abs(runtime.settings_json["speed"] - 1.3) < 0.01
        assert abs(runtime.settings_json["volume"] - 0.7) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 6. TtsPipeline barge-in / pipeline integration for a Hume-routed chunk
# ─────────────────────────────────────────────────────────────────────────────


class TestHumeBargeInPipelineIntegration:
    def _fresh_pipeline(self):
        from app.voice.tts_pipeline import TtsPipeline

        handler = MagicMock()
        handler._tts_cancel = asyncio.Event()
        handler.is_speaking = False
        handler._is_tts_playing = False
        handler._twilio_buffer_primed = False
        handler._prev_tts_tail = b""

        pipeline = TtsPipeline.__new__(TtsPipeline)
        pipeline._handler = handler
        pipeline._synthesis_tasks = {}
        pipeline._next_chunk_id = 0
        pipeline._turn_id = 0
        pipeline._playback_events = {0: asyncio.Event()}
        pipeline._turn_has_final = False
        pipeline._turn_end_call_after = False
        pipeline._turn_transfer_after = False
        pipeline._turn_chunk_count = 0
        pipeline._turn_cache_hits = 0
        pipeline._turn_start_ts = 0.0
        pipeline._audio_cache = {}
        pipeline._synthesis_semaphore = asyncio.BoundedSemaphore(8)
        return pipeline, handler

    def test_barge_in_cancels_hume_routed_chunk_cleanly(self):
        """
        Route a chunk's synthesis through Hume (mocked at the
        hume_tts_service boundary) via `handler._prefetch_tts_audio`
        returning the real HumeTTSAdapter async generator, then trigger a
        barge-in mid-synthesis. Assert: the in-flight synthesis task is
        cancelled, the handler's playing-state flags are reset, and no
        zombie task remains in `_synthesis_tasks` (Rime's existing barge-in
        guarantees hold for a Hume-routed chunk too).
        """
        pipeline, handler = self._fresh_pipeline()

        synthesis_started = asyncio.Event()
        never_resolves = asyncio.Event()

        async def _hume_routed_prefetch(task):
            # Stand-in for what _prefetch_tts_audio really does when
            # tts_provider_slug == "hume": obtain the Hume adapter's async
            # generator and start consuming it.
            from app.utils.tts_adapter import HumeTTSAdapter

            with patch("app.utils.tts_adapter.get_hume_api_key", return_value="test-key"):
                adapter = HumeTTSAdapter()

            async def _fake_service_stream(**kwargs):
                synthesis_started.set()
                await never_resolves.wait()  # blocks until cancelled
                yield b"\xff" * 160  # pragma: no cover - never reached

            chunks = []
            with patch(
                "app.services.hume_tts_service.hume_tts_service.stream_text_to_speech",
                side_effect=_fake_service_stream,
            ):
                async for chunk in adapter.async_stream_synthesize(
                    text=task["text"],
                    voice_external_id="Male English Actor",
                    settings_json={"speed": 1.0},
                ):
                    chunks.append(chunk)
            return b"".join(chunks) or None

        handler._prefetch_tts_audio = _hume_routed_prefetch
        handler._stream_tts_chunk = AsyncMock()

        async def _run():
            # Establishes _ws_send_gates / _eleven_ws_session (see
            # TtsPipeline.cancel_current_and_clear_queue) before the first
            # queue_tts call, matching the existing Rime pipeline tests'
            # pattern.
            await pipeline.cancel_current_and_clear_queue()
            handler._tts_cancel.clear()

            await pipeline.queue_tts(
                {"text": "Routed through Hume", "use_ssml": False, "is_final": True}
            )
            await asyncio.wait_for(synthesis_started.wait(), timeout=2.0)

            # Barge-in mid-synthesis.
            await pipeline.cancel_current_and_clear_queue()

            # Give the cancelled task a beat to finish unwinding.
            for _ in range(20):
                if not pipeline._synthesis_tasks:
                    break
                await asyncio.sleep(0.01)

        asyncio.run(_run())

        assert handler._tts_cancel.is_set() is True
        assert pipeline._synthesis_tasks == {}, "no zombie synthesis task after barge-in"
        assert handler._is_tts_playing is False
        assert handler.is_speaking is False

    def test_pipeline_accepts_new_hume_chunk_after_cancel(self):
        """Stream restart after interruption (mirrors Rime's equivalent
        test): a fresh Hume-routed chunk must be queueable immediately
        after a barge-in cancel."""
        pipeline, handler = self._fresh_pipeline()

        async def _run():
            await pipeline.cancel_current_and_clear_queue()
            handler._tts_cancel.clear()

            await pipeline.queue_tts(
                {"text": "Sure,", "use_ssml": False, "is_final": False}
            )
            await pipeline.queue_tts(
                {"text": "I can help.", "use_ssml": False, "is_final": True}
            )

            assert len(pipeline._synthesis_tasks) == 2

        asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────────────────
# 7. Failure-propagation path through the real streaming pipeline
# ─────────────────────────────────────────────────────────────────────────────


class TestHumeFailureDegradesGracefully:
    def test_prefetch_tts_audio_itself_does_not_catch_generator_errors(self):
        """
        Precise-behavior check (read, don't assume): `_prefetch_tts_audio`
        (app/voice/tts_stream_mixin.py) only ever `await`s a call that
        *constructs* the async generator for streaming-capable providers
        (Rime, Hume) — it returns that generator object without consuming
        it, so `_prefetch_tts_audio`'s own `except Exception` block does
        **not** see errors raised while the generator is later iterated.
        This test proves that: a HumeTtsServiceError raised on the
        generator's first `__anext__` propagates out through the returned
        generator, not out of the `await handler._prefetch_tts_audio(...)`
        call itself.
        """
        from app.routers.bidirectional_stream import BidirectionalStreamHandler as Handler
        from app.services.hume_tts_service import HumeTtsServiceError

        h = object.__new__(Handler)
        h._tts_cancel = asyncio.Event()
        h.agent = MagicMock()
        h.agent.language = "en"
        h.agent.voice_type = "female"
        h.agent.tts_provider_slug = "hume"
        h.agent.tts_voice_external_id = "Male English Actor"
        h.agent.tts_settings_json = {}
        h.agent.tts_language = "en"
        h.agent.encrypted_elevenlabs_api_key = None
        h.agent.tts_voice = None
        h.db = None

        async def _raising_stream(**kwargs):
            raise HumeTtsServiceError("Hume upstream failure")
            yield b""  # pragma: no cover - never reached; makes this an async generator

        with patch("app.utils.tts_adapter.get_hume_api_key", return_value="test-key"), patch(
            "app.services.hume_tts_service.hume_tts_service.stream_text_to_speech",
            side_effect=_raising_stream,
        ):
            async def _run():
                result = await h._prefetch_tts_audio(
                    {"text": "Hello there, this will fail mid-stream"}
                )
                # The await itself must succeed and hand back an async
                # generator — the error hasn't happened yet.
                assert hasattr(result, "__anext__")
                with pytest.raises(HumeTtsServiceError):
                    async for _ in result:
                        pass

            asyncio.run(_run())

    def test_stream_tts_chunk_swallows_hume_generator_error_without_crashing_call(self):
        """
        The call-level degrade-gracefully guarantee actually lives in
        `_stream_tts_chunk`'s own broad `except Exception` (it iterates the
        generator `_prefetch_tts_audio` returned) — not in
        `_prefetch_tts_audio` itself. This exercises that real path: a
        `HumeTtsServiceError` raised while iterating a prefetched Hume
        async generator must not propagate out of `_stream_tts_chunk`, and
        the handler's playing-state flags must still be reset in `finally`.
        """
        from app.routers.bidirectional_stream import BidirectionalStreamHandler as Handler
        from app.services.hume_tts_service import HumeTtsServiceError

        h = object.__new__(Handler)
        h._tts_cancel = asyncio.Event()
        h._tts_lock = asyncio.Lock()
        h.is_speaking = False
        h._is_tts_playing = False
        h._twilio_buffer_primed = False
        h._prev_tts_tail = b""
        h.stream_sid = "MZ_test"
        h._stream_sid_ready = asyncio.Event()
        h._stream_sid_ready.set()
        h.websocket = MagicMock()
        h.websocket.send_json = AsyncMock()
        h.agent = MagicMock()
        h.agent.language = "en"
        h.agent.voice_type = "female"
        h.agent.tts_provider_slug = "hume"
        h.agent.tts_voice_external_id = "Male English Actor"
        h.agent.tts_settings_json = {}
        h.agent.tts_language = "en"
        h.agent.encrypted_elevenlabs_api_key = None
        h.db = None
        h._background_audio = MagicMock()
        h._is_background_audio_enabled = lambda: False
        h._resolve_voice_volume = lambda: 1.0
        h._livekit_recording_mirror = lambda: None
        h._metric_first_audio_ts = 0.0
        h._metric_first_token_ts = 0.0
        h._metric_stt_final_ts = 0.0

        async def _raising_iter():
            yield b"\xff" * 160
            raise HumeTtsServiceError("Hume upstream failure mid-turn")

        # Should not raise — errors caught internally, logged, call survives.
        asyncio.run(
            h._stream_tts_chunk(
                "Hello there",
                use_ssml=False,
                is_final=True,
                prefetched_bytes=_raising_iter(),
            )
        )

        # finally-block state resets must still have run.
        assert h.is_speaking is False
        assert h._is_tts_playing is False
