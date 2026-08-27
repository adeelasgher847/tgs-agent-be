"""
Regression tests for two production bugs that made browser LiveKit demo
calls (and, for bug 2, potentially the Twilio bridge path too) go silent:

1. Rime TTS sync-bridge path crashes with "RuntimeError: Event loop is
   closed" because RimeTtsService (a module-level singleton) caches a single
   httpx.AsyncClient across calls, while RimeTTSAdapter.synthesize()'s
   no-running-loop fallback drives each call through a brand new
   asyncio.run() — which creates *and tears down* a fresh event loop every
   time. httpx/httpcore's connection pool keeps idle keep-alive connections
   bound to whichever loop was running when they were created; reusing the
   client on a second asyncio.run() loop touches the (now-closed) first
   loop and raises "Event loop is closed".

2. _LiveKitAgentAudioPublisher.publish_mulaw() (livekit_browser_call_handler.py)
   assigns raw `bytes` (buffer format 'B') into `frame.data[:]`, but
   `rtc.AudioFrame.data` is a memoryview already cast to format 'h'
   (int16) — see venv/lib/python3.12/site-packages/livekit/rtc/audio_frame.py.
   Assigning a differently-formatted buffer into a memoryview slice raises
   "ValueError: memoryview assignment: lvalue and rvalue have different
   structures", even though both sides have the same total byte length.
"""
from __future__ import annotations

import asyncio
import http.server
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Bug 1: Rime httpx.AsyncClient reused across asyncio.run() loops
# ─────────────────────────────────────────────────────────────────────────────

class _KeepAliveHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP/1.1 keep-alive server standing in for the real Rime API.

    A real TCP connection (not a MockTransport) is required to reproduce the
    bug: httpcore's connection pool only touches the anyio/asyncio transport
    objects bound to the loop that created them when actually closing/reusing
    a socket-backed connection.
    """

    protocol_version = "HTTP/1.1"

    def do_POST(self):
        # Must fully drain the request body before responding, otherwise the
        # leftover bytes corrupt request framing on the next request reusing
        # this same keep-alive connection.
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length:
            self.rfile.read(content_length)

        body = b"\xff" * 160
        self.send_response(200)
        self.send_header("Content-Type", "audio/x-mulaw")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args, **kwargs):  # noqa: D401 - silence test server logs
        pass


@pytest.fixture
def local_rime_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _KeepAliveHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{server.server_address[0]}:{server.server_address[1]}/"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def _drive_two_sequential_asyncio_run_calls(base_url: str):
    """Mimics production: two worker-thread invocations of
    RimeTTSAdapter.synthesize()'s asyncio.run(coro) fallback, both hitting
    the *same* RimeTtsService singleton (and therefore the same cached
    httpx.AsyncClient)."""
    import app.services.rime_tts_service as rime_mod

    service = rime_mod.RimeTtsService.__new__(rime_mod.RimeTtsService)
    service._clients = {}
    service._api_key = "test-key"

    with patch.object(rime_mod, "_RIME_TTS_URL", base_url):
        async def _one_call():
            chunks = []
            async for chunk in service.stream_text_to_speech(text="hello there"):
                chunks.append(chunk)
            return chunks

        asyncio.run(_one_call())  # loop #1: creates + caches the client, then closes
        return asyncio.run(_one_call())  # loop #2: reuses the cached client


class TestRimeClientEventLoopSafety:
    def test_shared_client_survives_across_asyncio_run_calls(self, local_rime_server):
        """After the fix, a second asyncio.run() call reusing the singleton's
        client must succeed instead of raising 'Event loop is closed'."""
        chunks = _drive_two_sequential_asyncio_run_calls(local_rime_server)
        assert b"".join(chunks) == b"\xff" * 160

    def test_rime_tts_adapter_synthesize_survives_two_worker_thread_calls(
        self, local_rime_server
    ):
        """End-to-end through the real call path exercised in production:
        RimeTTSAdapter.synthesize() invoked twice back-to-back off the main
        thread (as run_in_executor does), sharing the module-level
        rime_tts_service singleton."""
        import concurrent.futures

        import app.services.rime_tts_service as rime_mod
        from app.utils.tts_adapter import RimeTTSAdapter

        # Reset the real singleton's cached clients so this test starts clean
        # regardless of test ordering.
        rime_mod.rime_tts_service._clients = {}

        with patch.object(rime_mod, "_RIME_TTS_URL", local_rime_server), patch.object(
            rime_mod.rime_tts_service, "_api_key", "test-key"
        ):
            adapter = RimeTTSAdapter.__new__(RimeTTSAdapter)

            def _call():
                return adapter.synthesize(
                    text="hello there",
                    voice_external_id="mistv2_Wildflower",
                    settings_json={"speed": 1.0},
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                first = pool.submit(_call).result()
                second = pool.submit(_call).result()

        assert first == b"\xff" * 160
        assert second == b"\xff" * 160


# ─────────────────────────────────────────────────────────────────────────────
# Bug 2: publish_mulaw() memoryview format mismatch
# ─────────────────────────────────────────────────────────────────────────────

class TestPublishMulawBufferFormat:
    @pytest.mark.asyncio
    async def test_publish_mulaw_writes_pcm_into_int16_frame_without_raising(self, caplog):
        """Reproduces the exact production traceback: rtc.AudioFrame.data is
        a memoryview already cast to format 'h' (int16); assigning a plain
        `bytes` object (format 'B') into it must not raise
        'memoryview assignment: lvalue and rvalue have different structures'.

        Uses the real installed `livekit.rtc.AudioFrame` (not a mock) so the
        test fails against the buggy code for the same reason production
        did, and passes once publish_mulaw() writes through a same-format
        view. publish_mulaw() catches and only logs exceptions (never
        re-raises), so the regression is asserted via: (a) capture_frame
        actually got called with real audio data, and (b) no
        "publish_mulaw failed" warning was logged.
        """
        from app.voice.livekit_browser_call_handler import _LiveKitAgentAudioPublisher

        publisher = _LiveKitAgentAudioPublisher("room-1")
        publisher._connected = True
        publisher._source = MagicMock()
        publisher._source.capture_frame = AsyncMock()

        # One full MULAW_FRAME_BYTES chunk of non-zero audio bytes (0x00 = max negative in ulaw).
        from app.utils.audio_utils import MULAW_FRAME_BYTES

        mulaw_bytes = bytes([0x00]) * MULAW_FRAME_BYTES

        # No mocking of `livekit.rtc` here — we want the real AudioFrame
        # buffer-protocol behavior to be exercised.
        with caplog.at_level("WARNING"):
            await publisher.publish_mulaw(mulaw_bytes)

        assert not any(
            "publish_mulaw failed" in record.message for record in caplog.records
        )
        publisher._source.capture_frame.assert_awaited_once()
        frame_arg = publisher._source.capture_frame.call_args[0][0]
        # frame.data must contain the converted PCM16 samples, not be left
        # empty/zeroed because the assignment silently failed.
        assert bytes(frame_arg.data.cast("B")) != bytes(len(frame_arg.data.cast("B")))
