"""Deepgram Nova (v1/listen) SpeechStarted (VAD onset) event wiring.

Verifies:
1. `vad_events="true"` is passed to the connect() call.
2. A ListenV1SpeechStarted message is translated to a minimal
   {"speech_started": True} result (no transcript text) and does not
   disturb pending-transcript bookkeeping used by the UtteranceEnd fallback.
"""
from contextlib import contextmanager
from types import SimpleNamespace

from deepgram.core.events import EventType
from deepgram.listen.v1.types.listen_v1results import ListenV1Results
from deepgram.listen.v1.types.listen_v1speech_started import ListenV1SpeechStarted

from app.services.deepgram_stt_service import DeepgramSTTService


def _speech_started() -> ListenV1SpeechStarted:
    return ListenV1SpeechStarted.model_construct(type="SpeechStarted", channel=[0], timestamp=0.0)


def _results(transcript: str, confidence: float, speech_final: bool, is_final: bool = False):
    return ListenV1Results.model_construct(
        channel={"alternatives": [{"transcript": transcript, "confidence": confidence}]},
        speech_final=speech_final,
        is_final=is_final,
    )


def _make_session_with_on_message():
    captured = {}

    def fake_on(event, callback):
        captured[event] = callback

    @contextmanager
    def fake_connect(**_kwargs):
        yield SimpleNamespace(on=fake_on, start_listening=lambda: None)

    session = DeepgramSTTService.StreamingSTTSession(
        client=SimpleNamespace(),
        language_code="en",
        encoding="MULAW",
        sample_rate=8000,
        interim_results=True,
        single_utterance=False,
        model="nova-2",
    )
    session._client = SimpleNamespace(listen=SimpleNamespace(v1=SimpleNamespace(connect=fake_connect)))
    session.finish()
    session._run_blocking_stream()
    while not session._results_q.empty():
        session._results_q.get_nowait()
    return session, captured[EventType.MESSAGE]


def test_connect_call_includes_vad_events_true():
    captured = {}

    @contextmanager
    def fake_connect(**kwargs):
        captured.update(kwargs)
        yield SimpleNamespace(on=lambda *_: None, start_listening=lambda: None)

    svc = DeepgramSTTService()
    svc._client = SimpleNamespace(listen=SimpleNamespace(v1=SimpleNamespace(connect=fake_connect)))
    session = svc.create_streaming_session(model="nova-2", encoding="MULAW", sample_rate=8000)
    session.finish()
    session._run_blocking_stream()

    # SDK urlencodes Python bools wrong -- must be the string "true", not True.
    assert captured.get("vad_events") == "true"


def test_speech_started_pushes_minimal_result():
    session, on_message = _make_session_with_on_message()

    on_message(_speech_started())

    result = session._results_q.get_nowait()
    assert result == {"speech_started": True}
    assert session._results_q.empty()


def test_speech_started_does_not_disturb_pending_transcript_state():
    """SpeechStarted must be a pure passthrough event -- it must not clear or
    mutate the UtteranceEnd fallback's pending-transcript bookkeeping."""
    session, on_message = _make_session_with_on_message()

    on_message(_results("hello there", 0.7, speech_final=False))
    session._results_q.get_nowait()  # drain interim
    assert session._pending_transcript == "hello there"

    on_message(_speech_started())
    session._results_q.get_nowait()  # drain speech_started

    # Pending transcript state (used by the UtteranceEnd fallback) is untouched.
    assert session._pending_transcript == "hello there"
    assert session._pending_confidence == 0.7
