"""
Unit tests for app/services/gemini_live_service.py — GeminiLiveSession.

Follows tests/voice/test_vertex_llm.py's established pattern for mocking
google.genai: tests/conftest.py stubs sys.modules["google.genai"] as a bare
MagicMock() for the whole test session (so unrelated tests don't need real
GCP credentials/SDK installed); tests here that need the real installed
google-genai==2.3.0 types (SpeechConfig, LiveConnectConfig, Blob, etc.) use
the same _real_google_genai() context-manager technique.
"""
from __future__ import annotations

import asyncio
import contextlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.llm_models import GEMINI_LIVE_MODELS
from app.services.gemini_live_service import (
    GeminiLiveSession,
    _ensure_api_key_client,
)
from app.services.vertex_gemini_service import VertexLlmError, VertexLlmErrorType


@contextlib.contextmanager
def _real_google_genai():
    """Temporarily undo conftest's google/google.genai MagicMock stub so
    `import google.genai...` resolves to the real installed package for the
    duration of the `with` block, then restores the stub. Mirrors
    test_vertex_llm.py's identically-named helper."""
    saved = {k: v for k, v in sys.modules.items() if k == "google" or k.startswith("google.")}
    for k in list(saved):
        del sys.modules[k]
    try:
        import google.genai.types as real_types
        import google.genai.errors as real_errors

        yield real_types, real_errors
    finally:
        for k in list(sys.modules):
            if k == "google" or k.startswith("google."):
                del sys.modules[k]
        sys.modules.update(saved)


class _FakeAsyncSession:
    """Stands in for google.genai.live.AsyncSession."""

    def __init__(self, messages=None):
        self._messages = messages or []
        self.send_realtime_input = AsyncMock()
        self.send_client_content = AsyncMock()
        self.send_tool_response = AsyncMock()

    async def receive(self):
        for m in self._messages:
            yield m


class _FakeConnectCM:
    """Stands in for the object returned by client.aio.live.connect(...) —
    an async context manager yielding a session (real SDK decorates connect
    with @contextlib.asynccontextmanager)."""

    def __init__(self, session):
        self._session = session
        self.aexit_called = False

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        self.aexit_called = True
        return False


def _make_fake_client(session: _FakeAsyncSession):
    connect_cm = _FakeConnectCM(session)
    client = MagicMock()
    client.aio.live.connect = MagicMock(return_value=connect_cm)
    return client, connect_cm


# ─────────────────────────────────────────────────────────────────────────
# Construction / model validation
# ─────────────────────────────────────────────────────────────────────────


class TestGeminiLiveSessionConstruction:
    @pytest.mark.parametrize("model_name", list(GEMINI_LIVE_MODELS.keys()))
    def test_known_models_construct_successfully(self, model_name):
        session = GeminiLiveSession(model_name)
        assert session.model_name == model_name

    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError, match="not a Gemini Live native-audio model"):
            GeminiLiveSession("gemini-2.5-flash")

    def test_unknown_model_error_lists_valid_models(self):
        with pytest.raises(ValueError) as exc_info:
            GeminiLiveSession("totally-made-up-model")
        for model_name in GEMINI_LIVE_MODELS:
            assert model_name in str(exc_info.value)


# ─────────────────────────────────────────────────────────────────────────
# Client construction per auth mode
# ─────────────────────────────────────────────────────────────────────────


class TestBuildClientPerAuthMode:
    @pytest.mark.parametrize(
        "model_name",
        [
            "gemini-live-2.5-flash-native-audio",
            "gemini-live-2.5-flash-preview-native-audio-09-2025",
        ],
    )
    def test_vertex_models_use_ensure_vertex_client_with_us_central1(self, model_name):
        """Both Vertex-auth models must route through the reused
        _ensure_vertex_client(location) — never GEMINI_API_KEY."""
        session = GeminiLiveSession(model_name)
        mock_ensure = MagicMock(return_value="fake_vertex_client")

        with patch("app.services.gemini_live_service._ensure_vertex_client", mock_ensure):
            client = session._build_client()

        assert client == "fake_vertex_client"
        mock_ensure.assert_called_once_with("us-central1")

    def test_api_key_model_uses_ensure_api_key_client(self):
        session = GeminiLiveSession("gemini-3.1-flash-live-preview")
        mock_ensure = MagicMock(return_value="fake_api_key_client")

        with patch("app.services.gemini_live_service._ensure_api_key_client", mock_ensure):
            client = session._build_client()

        assert client == "fake_api_key_client"
        mock_ensure.assert_called_once_with()

    def test_ensure_api_key_client_reuses_settings_gemini_api_key(self, monkeypatch):
        import app.services.gemini_live_service as mod

        monkeypatch.setattr(mod, "_api_key_client", None)
        monkeypatch.setattr(mod.settings, "GEMINI_API_KEY", "test-api-key-value")

        fake_client_cls = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        fake_genai = SimpleNamespace(Client=fake_client_cls)
        monkeypatch.setitem(sys.modules, "google.genai", MagicMock())
        monkeypatch.setattr(sys.modules["google"], "genai", fake_genai, raising=False)

        client = _ensure_api_key_client()

        fake_client_cls.assert_called_once_with(api_key="test-api-key-value")
        assert client.api_key == "test-api-key-value"
        monkeypatch.setattr(mod, "_api_key_client", None)

    def test_ensure_api_key_client_caches_across_calls(self, monkeypatch):
        import app.services.gemini_live_service as mod

        monkeypatch.setattr(mod, "_api_key_client", None)
        monkeypatch.setattr(mod.settings, "GEMINI_API_KEY", "test-api-key-value")

        fake_client_cls = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
        fake_genai = SimpleNamespace(Client=fake_client_cls)
        monkeypatch.setattr(sys.modules["google"], "genai", fake_genai, raising=False)

        client1 = _ensure_api_key_client()
        client2 = _ensure_api_key_client()

        assert client1 is client2
        fake_client_cls.assert_called_once()
        monkeypatch.setattr(mod, "_api_key_client", None)

    def test_ensure_api_key_client_raises_when_no_api_key_configured(self, monkeypatch):
        import app.services.gemini_live_service as mod

        monkeypatch.setattr(mod, "_api_key_client", None)
        monkeypatch.setattr(mod.settings, "GEMINI_API_KEY", "")

        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            _ensure_api_key_client()


# ─────────────────────────────────────────────────────────────────────────
# start() — connects and launches the receive loop
# ─────────────────────────────────────────────────────────────────────────


class TestStart:
    def test_start_connects_and_launches_receive_loop(self):
        session_obj = _FakeAsyncSession(messages=[])
        client, connect_cm = _make_fake_client(session_obj)

        gls = GeminiLiveSession("gemini-3.1-flash-live-preview")

        async def _run():
            with patch.object(gls, "_build_client", return_value=client):
                await gls.start(
                    system_instruction="be helpful",
                    on_audio_chunk=AsyncMock(),
                    on_interrupted=AsyncMock(),
                )
            await asyncio.sleep(0)  # let receive loop run to completion
            await gls.close()

        asyncio.run(_run())

        client.aio.live.connect.assert_called_once()
        assert client.aio.live.connect.call_args.kwargs["model"] == "gemini-3.1-flash-live-preview"
        assert connect_cm.aexit_called is True

    def test_start_failure_raises_vertex_llm_error(self):
        gls = GeminiLiveSession("gemini-3.1-flash-live-preview")

        with patch.object(gls, "_build_client", side_effect=RuntimeError("boom")):
            with pytest.raises(VertexLlmError):
                asyncio.run(
                    gls.start(
                        system_instruction="hi",
                        on_audio_chunk=AsyncMock(),
                        on_interrupted=AsyncMock(),
                    )
                )

    def test_connect_exception_translated_to_vertex_llm_error(self):
        client = MagicMock()
        client.aio.live.connect = MagicMock(side_effect=RuntimeError("quota exceeded"))
        gls = GeminiLiveSession("gemini-3.1-flash-live-preview")

        with patch.object(gls, "_build_client", return_value=client):
            with pytest.raises(VertexLlmError) as exc_info:
                asyncio.run(
                    gls.start(
                        system_instruction="hi",
                        on_audio_chunk=AsyncMock(),
                        on_interrupted=AsyncMock(),
                    )
                )
        assert exc_info.value.error_type == VertexLlmErrorType.QUOTA


# ─────────────────────────────────────────────────────────────────────────
# send_audio — wraps the right Blob/mime_type
# ─────────────────────────────────────────────────────────────────────────


class TestSendAudio:
    def test_send_audio_wraps_blob_with_correct_mime_type(self):
        with _real_google_genai():
            session_obj = _FakeAsyncSession()
            gls = GeminiLiveSession("gemini-3.1-flash-live-preview")
            gls._session = session_obj

            asyncio.run(gls.send_audio(b"\x01\x02\x03\x04"))

            session_obj.send_realtime_input.assert_awaited_once()
            blob = session_obj.send_realtime_input.call_args.kwargs["audio"]
            assert blob.data == b"\x01\x02\x03\x04"
            assert blob.mime_type == "audio/pcm;rate=16000"

    def test_send_text_wraps_content_and_turn_complete(self):
        with _real_google_genai():
            session_obj = _FakeAsyncSession()
            gls = GeminiLiveSession("gemini-3.1-flash-live-preview")
            gls._session = session_obj

            asyncio.run(gls.send_text("hello", turn_complete=False))

            session_obj.send_client_content.assert_awaited_once()
            kwargs = session_obj.send_client_content.call_args.kwargs
            assert kwargs["turn_complete"] is False
            turns = kwargs["turns"]
            assert turns.role == "user"
            assert turns.parts[0].text == "hello"

    def test_send_tool_response_passthrough(self):
        gls = GeminiLiveSession("gemini-3.1-flash-live-preview")
        session_obj = _FakeAsyncSession()
        gls._session = session_obj

        fake_responses = [SimpleNamespace(name="check_availability")]
        asyncio.run(gls.send_tool_response(fake_responses))

        session_obj.send_tool_response.assert_awaited_once_with(function_responses=fake_responses)


# ─────────────────────────────────────────────────────────────────────────
# _receive_loop — demuxing each LiveServerMessage variant
# ─────────────────────────────────────────────────────────────────────────


def _msg(**kwargs) -> SimpleNamespace:
    """Build a fake LiveServerMessage with only the requested attrs set;
    all others resolve to None via getattr(..., default) in the source."""
    base = dict(
        setup_complete=None,
        server_content=None,
        tool_call=None,
        tool_call_cancellation=None,
        go_away=None,
        session_resumption_update=None,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


class TestReceiveLoopDemux:
    def _run_with_messages(self, messages, **callback_kwargs):
        session_obj = _FakeAsyncSession(messages=messages)
        gls = GeminiLiveSession("gemini-3.1-flash-live-preview")
        gls._session = session_obj
        for name, cb in callback_kwargs.items():
            setattr(gls, name, cb)

        asyncio.run(gls._receive_loop())
        return gls

    def test_audio_chunk_demuxed_to_on_audio_chunk(self):
        part = SimpleNamespace(inline_data=SimpleNamespace(data=b"\x00\x01", mime_type="audio/pcm;rate=24000"))
        model_turn = SimpleNamespace(parts=[part])
        server_content = SimpleNamespace(model_turn=model_turn, interrupted=False, input_transcription=None, output_transcription=None)
        on_audio_chunk = AsyncMock()

        self._run_with_messages([_msg(server_content=server_content)], on_audio_chunk=on_audio_chunk)

        on_audio_chunk.assert_awaited_once_with(b"\x00\x01")

    def test_interrupted_demuxed_to_on_interrupted(self):
        server_content = SimpleNamespace(model_turn=None, interrupted=True, input_transcription=None, output_transcription=None)
        on_interrupted = AsyncMock()

        self._run_with_messages([_msg(server_content=server_content)], on_interrupted=on_interrupted)

        on_interrupted.assert_awaited_once()

    def test_input_transcript_demuxed(self):
        transcript = SimpleNamespace(text="hello there", finished=True)
        server_content = SimpleNamespace(model_turn=None, interrupted=False, input_transcription=transcript, output_transcription=None)
        on_input_transcript = AsyncMock()

        self._run_with_messages([_msg(server_content=server_content)], on_input_transcript=on_input_transcript)

        on_input_transcript.assert_awaited_once_with("hello there", True)

    def test_output_transcript_demuxed(self):
        transcript = SimpleNamespace(text="hi, how can I help", finished=False)
        server_content = SimpleNamespace(model_turn=None, interrupted=False, input_transcription=None, output_transcription=transcript)
        on_output_transcript = AsyncMock()

        self._run_with_messages([_msg(server_content=server_content)], on_output_transcript=on_output_transcript)

        on_output_transcript.assert_awaited_once_with("hi, how can I help", False)

    def test_tool_call_demuxed(self):
        function_calls = [SimpleNamespace(name="check_availability", args={"date": "today"})]
        tool_call = SimpleNamespace(function_calls=function_calls)
        on_tool_call = AsyncMock()

        self._run_with_messages([_msg(tool_call=tool_call)], on_tool_call=on_tool_call)

        on_tool_call.assert_awaited_once_with(function_calls)

    def test_go_away_logs_and_takes_no_action(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            self._run_with_messages([_msg(go_away=SimpleNamespace())])
        assert any("go_away" in rec.message for rec in caplog.records)

    def test_setup_complete_logged_no_callback_invoked(self):
        on_audio_chunk = AsyncMock()
        self._run_with_messages([_msg(setup_complete=SimpleNamespace())], on_audio_chunk=on_audio_chunk)
        on_audio_chunk.assert_not_awaited()

    def test_sdk_exception_translated_to_vertex_llm_error_and_routed_to_on_error(self):
        class _BadMessage:
            @property
            def setup_complete(self):
                raise RuntimeError("quota exceeded mid-stream")

        on_error = AsyncMock()
        self._run_with_messages([_BadMessage()], on_error=on_error)

        on_error.assert_awaited_once()
        err = on_error.call_args.args[0]
        assert isinstance(err, VertexLlmError)
        assert err.error_type == VertexLlmErrorType.QUOTA

    def test_error_without_on_error_callback_is_logged_not_raised(self, caplog):
        import logging

        class _BadMessage:
            @property
            def setup_complete(self):
                raise RuntimeError("boom")

        with caplog.at_level(logging.ERROR):
            self._run_with_messages([_BadMessage()])  # no on_error set
        assert any("GeminiLiveSession" in rec.message for rec in caplog.records)


# ─────────────────────────────────────────────────────────────────────────
# close() — idempotent, never raises
# ─────────────────────────────────────────────────────────────────────────


class TestClose:
    def test_close_is_idempotent(self):
        gls = GeminiLiveSession("gemini-3.1-flash-live-preview")
        asyncio.run(gls.close())
        asyncio.run(gls.close())  # must not raise second time

    def test_close_cancels_receive_task_and_exits_connect_cm(self):
        session_obj = _FakeAsyncSession(messages=[])
        client, connect_cm = _make_fake_client(session_obj)
        gls = GeminiLiveSession("gemini-3.1-flash-live-preview")

        async def _run():
            with patch.object(gls, "_build_client", return_value=client):
                await gls.start(
                    system_instruction="hi",
                    on_audio_chunk=AsyncMock(),
                    on_interrupted=AsyncMock(),
                )
            await gls.close()

        asyncio.run(_run())
        assert connect_cm.aexit_called is True
        assert gls._receive_task is None

    def test_close_never_raises_even_if_session_teardown_errors(self):
        gls = GeminiLiveSession("gemini-3.1-flash-live-preview")

        class _BadCM:
            async def __aexit__(self, *exc):
                raise RuntimeError("teardown boom")

        gls._connect_cm = _BadCM()
        # Must not raise.
        asyncio.run(gls.close())

    def test_close_never_raises_even_if_receive_task_already_dead(self):
        gls = GeminiLiveSession("gemini-3.1-flash-live-preview")

        async def _dead_task_coro():
            raise RuntimeError("already crashed")

        async def _setup_and_close():
            task = asyncio.create_task(_dead_task_coro())
            await asyncio.sleep(0)  # let it crash
            gls._receive_task = task
            await gls.close()  # must not raise/propagate task's exception

        asyncio.run(_setup_and_close())
