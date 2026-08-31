"""Deepgram Nova (v1/listen) vad_events / SpeechStarted soft-duck signal.

Regression coverage for the barge-in soft-duck feature: `vad_events="true"`
must be included on the Nova-3 `listen.v1.connect()` call, and a
`ListenV1SpeechStarted` message must push `{"speech_started": True}` through
`on_message` -- deliberately without any transcript/confidence, so callers
downstream (SttPipeline._reader_loop) can distinguish it from a normal
interim/final result and only use it for a low-risk "soft" action, never a
hard barge-in cancel.
"""
from contextlib import contextmanager
from types import SimpleNamespace

from deepgram.core.events import EventType
from deepgram.listen.v1.types.listen_v1speech_started import ListenV1SpeechStarted

from app.services.deepgram_stt_service import DeepgramSTTService


def _speech_started() -> ListenV1SpeechStarted:
    # model_construct bypasses validation of fields on_message never reads.
    return ListenV1SpeechStarted.model_construct(type="SpeechStarted")


def _make_session_with_on_message():
    """Build a StreamingSTTSession and run just enough of _run_blocking_stream
    to capture its on_message closure and the kwargs passed to connect(),
    without opening a real socket."""
    captured_kwargs = {}
    captured_handlers = {}

    def fake_on(event, callback):
        captured_handlers[event] = callback

    @contextmanager
    def fake_connect(**kwargs):
        captured_kwargs.update(kwargs)
        yield SimpleNamespace(on=fake_on, start_listening=lambda: None)

    session = DeepgramSTTService.StreamingSTTSession(
        client=SimpleNamespace(),
        language_code="en",
        encoding="MULAW",
        sample_rate=8000,
        interim_results=True,
        single_utterance=False,
        model="nova-3",
    )
    session._client = SimpleNamespace(listen=SimpleNamespace(v1=SimpleNamespace(connect=fake_connect)))
    session.finish()
    session._run_blocking_stream()
    while not session._results_q.empty():
        session._results_q.get_nowait()
    return session, captured_handlers[EventType.MESSAGE], captured_kwargs


def test_nova_connect_includes_vad_events_true():
    """Nova-3 connect() must request vad_events="true" so SpeechStarted
    events are emitted at all -- without this param, Deepgram never sends
    them and the soft-duck feature would be silently dead."""
    _, _, captured_kwargs = _make_session_with_on_message()

    assert captured_kwargs.get("vad_events") == "true"


def test_speech_started_message_pushes_speech_started_result():
    """A ListenV1SpeechStarted message must push exactly
    {"speech_started": True} -- no transcript/confidence keys, so downstream
    consumers can't mistake it for a normal interim/final result."""
    session, on_message, _ = _make_session_with_on_message()

    on_message(_speech_started())

    result = session._results_q.get_nowait()
    assert result == {"speech_started": True}
    assert session._results_q.empty()


def test_speech_started_message_does_not_touch_pending_transcript_state():
    """SpeechStarted fires independently of the transcript accumulation
    machinery (pending_transcript / pending_finalized_prefix used by the
    UtteranceEnd fallback) -- it must not perturb that state."""
    session, on_message, _ = _make_session_with_on_message()
    session._pending_transcript = "hello there"
    session._pending_confidence = 0.7
    session._pending_finalized_prefix = "hello there"

    on_message(_speech_started())
    session._results_q.get_nowait()  # drain the speech_started result

    assert session._pending_transcript == "hello there"
    assert session._pending_confidence == 0.7
    assert session._pending_finalized_prefix == "hello there"
