"""Deepgram STT model-selection tests: Nova model forwarding + Flux (v2/listen)."""
import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from app.services.deepgram_stt_service import DeepgramSTTService
from app.voice.stt_pipeline import SttPipeline
from deepgram.listen.v2.types.listen_v2fatal_error import ListenV2FatalError
from deepgram.listen.v2.types.listen_v2turn_info import ListenV2TurnInfo
from deepgram.listen.v2.types.listen_v2turn_info_words_item import ListenV2TurnInfoWordsItem


def _turn_info(event: str, transcript: str = "", words=None) -> ListenV2TurnInfo:
    return ListenV2TurnInfo(
        type="TurnInfo",
        request_id="req-1",
        sequence_id=1,
        event=event,
        turn_index=0,
        audio_window_start=0.0,
        audio_window_end=1.0,
        transcript=transcript,
        words=words or [],
        end_of_turn_confidence=0.9,
    )


def _word(text: str, confidence: float) -> ListenV2TurnInfoWordsItem:
    return ListenV2TurnInfoWordsItem(word=text, confidence=confidence)


# ── create_streaming_session dispatch ─────────────────────────────────────────


def test_create_streaming_session_dispatches_flux_by_model_prefix():
    svc = DeepgramSTTService()
    svc._client = object()  # only needs to be truthy for this dispatch check

    flux_session = svc.create_streaming_session(model="flux-general-en")
    assert isinstance(flux_session, DeepgramSTTService.FluxStreamingSTTSession)

    multi_session = svc.create_streaming_session(model="flux-general-multi")
    assert isinstance(multi_session, DeepgramSTTService.FluxStreamingSTTSession)

    nova_session = svc.create_streaming_session(model="nova-2")
    assert isinstance(nova_session, DeepgramSTTService.StreamingSTTSession)

    default_session = svc.create_streaming_session(model=None)
    assert isinstance(default_session, DeepgramSTTService.StreamingSTTSession)
    assert default_session._model == "nova-3"


# ── Nova (v1/listen) model forwarding ──────────────────────────────────────────


def test_nova_session_connects_with_resolved_model(monkeypatch):
    captured = {}

    @contextmanager
    def fake_connect(**kwargs):
        captured.update(kwargs)
        conn = SimpleNamespace(
            on=lambda *_: None,
            start_listening=lambda: None,
        )
        yield conn

    fake_client = SimpleNamespace(listen=SimpleNamespace(v1=SimpleNamespace(connect=fake_connect)))

    svc = DeepgramSTTService()
    svc._client = fake_client
    session = svc.create_streaming_session(model="nova-2", encoding="MULAW", sample_rate=8000)
    session.finish()  # audio_q gets a None sentinel so sender_loop closes immediately
    session._run_blocking_stream()

    assert captured.get("model") == "nova-2"


# ── Flux (v2/listen) model forwarding ──────────────────────────────────────────


def test_flux_session_connects_via_v2_with_eot_config(monkeypatch):
    captured = {}

    @contextmanager
    def fake_connect(**kwargs):
        captured.update(kwargs)
        conn = SimpleNamespace(
            on=lambda *_: None,
            start_listening=lambda: None,
        )
        yield conn

    fake_client = SimpleNamespace(listen=SimpleNamespace(v2=SimpleNamespace(connect=fake_connect)))

    svc = DeepgramSTTService()
    svc._client = fake_client
    session = svc.create_streaming_session(
        model="flux-general-en",
        encoding="MULAW",
        sample_rate=8000,
        api_config={"eot_timeout_ms": 5000},
    )
    assert isinstance(session, DeepgramSTTService.FluxStreamingSTTSession)
    session.finish()
    session._run_blocking_stream()

    assert captured.get("model") == "flux-general-en"
    assert captured.get("encoding") == "mulaw"
    assert captured.get("eot_timeout_ms") == 5000


# ── Flux TurnInfo event translation ────────────────────────────────────────────


def _make_flux_session() -> "DeepgramSTTService.FluxStreamingSTTSession":
    return DeepgramSTTService.FluxStreamingSTTSession(
        client=SimpleNamespace(),
        language_code="en",
        encoding="MULAW",
        sample_rate=8000,
        model="flux-general-en",
    )


def test_flux_update_event_emits_interim_result():
    session = _make_flux_session()

    on_message = _extract_on_message(session)
    on_message(_turn_info("Update", transcript="hel", words=[_word("hel", 0.5)]))

    result = session._results_q.get_nowait()
    assert result == {"transcript": "hel", "confidence": 0.5, "is_final": False}


def test_flux_end_of_turn_event_emits_final_result():
    session = _make_flux_session()

    on_message = _extract_on_message(session)
    words = [_word("hello", 0.9), _word("there", 0.7)]
    on_message(_turn_info("EndOfTurn", transcript="hello there", words=words))

    result = session._results_q.get_nowait()
    assert result["transcript"] == "hello there"
    assert result["is_final"] is True
    assert result["confidence"] == pytest.approx(0.8)


@pytest.mark.parametrize("event", ["StartOfTurn", "EagerEndOfTurn", "TurnResumed"])
def test_flux_intermediate_turn_events_do_not_emit_results(event):
    session = _make_flux_session()

    on_message = _extract_on_message(session)
    on_message(_turn_info(event, transcript="ignored", words=[_word("ignored", 0.9)]))

    assert session._results_q.empty()


def test_flux_fatal_error_emits_error_result():
    session = _make_flux_session()

    on_message = _extract_on_message(session)
    on_message(ListenV2FatalError(type="FatalError", sequence_id=1, code="ERR", description="boom"))

    result = session._results_q.get_nowait()
    assert result["error"] == "boom"
    assert result["is_final"] is True


def _extract_on_message(session):
    """Run _run_blocking_stream against a fake v2 connection just far enough to
    capture the on_message closure it registers, without opening a real socket."""
    from deepgram.core.events import EventType

    captured = {}

    def fake_on(event, callback):
        captured[event] = callback

    @contextmanager
    def fake_connect(**_kwargs):
        yield SimpleNamespace(on=fake_on, start_listening=lambda: None)

    session._client = SimpleNamespace(listen=SimpleNamespace(v2=SimpleNamespace(connect=fake_connect)))
    session.finish()  # closes audio_q immediately so sender_loop exits without blocking
    session._run_blocking_stream()
    # _run_blocking_stream already completed (sender_loop saw the finish() sentinel)
    # and pushed its {"done": True} sentinel -- drain it so tests only see the
    # results they push via the returned on_message callback.
    while not session._results_q.empty():
        session._results_q.get_nowait()
    return captured[EventType.MESSAGE]


# ── recreate_with_endpointing no-op for Flux ──────────────────────────────────


def test_recreate_with_endpointing_is_noop_for_flux(monkeypatch):
    async def on_interim(_, __):
        return None

    async def on_final(_, __):
        return None

    async def _run():
        pipeline = SttPipeline(
            language_code="en",
            on_interim=on_interim,
            on_final=on_final,
            provider_slug="deepgram",
            model_id="flux-general-en",
        )
        pipeline._endpointing_ms = 350
        await pipeline.recreate_with_endpointing(900)
        # Endpointing must be left untouched -- Flux ignores app-side endpointing.
        assert pipeline._endpointing_ms == 350

    asyncio.run(_run())


# ── SttPipeline forwards model_id + api_config to the Deepgram service ───────


def test_stt_pipeline_forwards_model_and_api_config(monkeypatch):
    captured = {}

    def fake_create_streaming_session(**kwargs):
        captured.update(kwargs)

        class FakeSession:
            async def start(self):
                return None

            async def get_result(self):
                await asyncio.sleep(1)
                return {}

            def push_audio(self, _):
                return None

            def finish(self):
                return None

        return FakeSession()

    from app.services import deepgram_stt_service as dg_module

    monkeypatch.setattr(
        dg_module.deepgram_stt_service,
        "create_streaming_session",
        fake_create_streaming_session,
    )

    async def on_interim(_, __):
        return None

    async def on_final(_, __):
        return None

    async def _run():
        pipeline = SttPipeline(
            language_code="en",
            on_interim=on_interim,
            on_final=on_final,
            provider_slug="deepgram",
            model_id="nova-2",
            api_config={"api_model": "nova-2"},
        )
        await pipeline.feed_audio_chunk(b"\x00")
        await pipeline.aclose()

    asyncio.run(_run())

    assert captured.get("model") == "nova-2"
    assert captured.get("api_config") == {"api_model": "nova-2"}
