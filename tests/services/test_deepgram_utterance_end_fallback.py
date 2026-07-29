"""Deepgram Nova (v1/listen) UtteranceEnd fallback finalization.

Deepgram recommends pairing silence-based endpointing/speech_final with the
word-timing-based UtteranceEnd event for real-time conversational agents:
endpointing alone can fail to fire on phone-line noise (static, hold music,
cross-talk), leaving a turn open indefinitely. UtteranceEnd is the fallback.
"""
from contextlib import contextmanager
from types import SimpleNamespace

from deepgram.core.events import EventType
from deepgram.listen.v1.types.listen_v1results import ListenV1Results
from deepgram.listen.v1.types.listen_v1utterance_end import ListenV1UtteranceEnd

from app.services.deepgram_stt_service import DeepgramSTTService


def _results(
    transcript: str, confidence: float, speech_final: bool, is_final: bool = False
) -> ListenV1Results:
    # model_construct bypasses validation of the many unrelated required fields
    # (metadata, duration, etc.) that on_message never reads.
    return ListenV1Results.model_construct(
        channel={"alternatives": [{"transcript": transcript, "confidence": confidence}]},
        speech_final=speech_final,
        is_final=is_final,
    )


def _utterance_end() -> ListenV1UtteranceEnd:
    return ListenV1UtteranceEnd.model_construct(type="UtteranceEnd")


def _make_session_with_on_message():
    """Build a StreamingSTTSession and run just enough of _run_blocking_stream
    to capture its on_message closure, without opening a real socket."""
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


def test_utterance_end_finalizes_pending_interim_when_speech_final_never_arrives():
    session, on_message = _make_session_with_on_message()

    on_message(_results("hello there", 0.7, speech_final=False))
    assert session._results_q.get_nowait() == {
        "transcript": "hello there",
        "confidence": 0.7,
        "is_final": False,
    }

    on_message(_utterance_end())

    result = session._results_q.get_nowait()
    assert result == {"transcript": "hello there", "confidence": 0.7, "is_final": True}
    assert session._results_q.empty()


def test_utterance_end_is_noop_after_speech_final_already_fired():
    session, on_message = _make_session_with_on_message()

    on_message(_results("hello there", 0.7, speech_final=False))
    session._results_q.get_nowait()  # drain interim

    on_message(_results("hello there", 0.9, speech_final=True))
    final_result = session._results_q.get_nowait()
    assert final_result["is_final"] is True

    on_message(_utterance_end())

    # Nothing pending -- speech_final already closed this utterance, no double-final.
    assert session._results_q.empty()


def test_utterance_end_accumulates_multiple_finalized_segments():
    """A long utterance can be settled (is_final=true) in several segments before
    speech_final ever arrives. Each segment's transcript is scoped to that segment
    only, not cumulative -- the fallback must concatenate them rather than keep
    only the last one, or noisy-line calls (exactly the case this feature targets)
    would lose the earlier words of the sentence.
    """
    session, on_message = _make_session_with_on_message()

    on_message(_results("hello there", 0.7, speech_final=False, is_final=True))
    session._results_q.get_nowait()  # drain interim

    on_message(_results("how are you", 0.8, speech_final=False, is_final=False))
    session._results_q.get_nowait()  # drain interim

    on_message(_utterance_end())

    result = session._results_q.get_nowait()
    assert result["transcript"] == "hello there how are you"
    assert result["is_final"] is True


def test_utterance_end_with_no_pending_transcript_emits_nothing():
    session, on_message = _make_session_with_on_message()

    on_message(_utterance_end())

    assert session._results_q.empty()


def test_nova_connect_includes_utterance_end_ms():
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

    assert captured.get("utterance_end_ms") == 1000
