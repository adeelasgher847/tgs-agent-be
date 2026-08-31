"""Regression coverage for SttPipeline._reader_loop's `speech_started`
short-circuit (Deepgram Nova-3 `vad_events`/SpeechStarted soft-duck signal).

Verifies:
  - A {"speech_started": True} result emits SttSpeechStartedEvent on the
    event bus and invokes the optional on_speech_started callback.
  - It is NOT treated as a transcript (no on_interim/on_final call, no
    SttInterimEvent/SttFinalEvent emitted).
  - An exception raised inside the on_speech_started callback is caught and
    does not crash the reader loop -- subsequent results still get processed.
"""

from __future__ import annotations

import asyncio

from app.voice.stt_events import SttFinalEvent, SttInterimEvent, SttSpeechStartedEvent
from app.voice.stt_pipeline import SttPipeline


class _QueueSession:
    """Fake STT session whose get_result() drains a pre-seeded queue of
    result dicts, one per call -- mirrors DeepgramSTTService's real session
    interface closely enough for _reader_loop's needs."""

    def __init__(self, results: list[dict]):
        self._results = list(results)

    async def get_result(self) -> dict:
        if self._results:
            return self._results.pop(0)
        # Reader loop must not spin once the fixture is exhausted.
        await asyncio.sleep(10)
        return {}


def _make_pipeline(results: list[dict], on_speech_started=None):
    seen = {"finals": [], "interims": []}

    async def on_final(transcript, confidence):
        seen["finals"].append((transcript, confidence))

    async def on_interim(transcript, confidence):
        seen["interims"].append((transcript, confidence))

    pipeline = SttPipeline(
        language_code="en",
        on_interim=on_interim,
        on_final=on_final,
        on_speech_started=on_speech_started,
    )
    pipeline._stt_session = _QueueSession(results)
    return pipeline, seen


async def _run_reader_briefly(pipeline: SttPipeline, seconds: float = 0.3):
    task = asyncio.create_task(pipeline._reader_loop())
    await asyncio.sleep(seconds)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_speech_started_result_emits_event_and_invokes_callback():
    calls = []

    async def on_speech_started():
        calls.append(True)

    pipeline, seen = _make_pipeline(
        [{"speech_started": True}], on_speech_started=on_speech_started
    )

    emitted = []

    async def _capture(event):
        emitted.append(event)

    pipeline.event_bus.subscribe(_capture)

    await _run_reader_briefly(pipeline)

    assert calls == [True]
    assert len(emitted) == 1
    assert isinstance(emitted[0], SttSpeechStartedEvent)
    assert emitted[0].type == "speech_started"


async def test_speech_started_result_is_not_treated_as_transcript():
    pipeline, seen = _make_pipeline([{"speech_started": True}])

    emitted = []

    async def _capture(event):
        emitted.append(event)

    pipeline.event_bus.subscribe(_capture)

    await _run_reader_briefly(pipeline)

    assert seen["finals"] == []
    assert seen["interims"] == []
    assert not any(isinstance(e, (SttInterimEvent, SttFinalEvent)) for e in emitted)


async def test_speech_started_with_no_callback_is_a_noop():
    """on_speech_started is optional -- when None, the reader loop must
    still emit the event and continue without crashing."""
    pipeline, seen = _make_pipeline([{"speech_started": True}], on_speech_started=None)

    emitted = []

    async def _capture(event):
        emitted.append(event)

    pipeline.event_bus.subscribe(_capture)

    await _run_reader_briefly(pipeline)

    assert len(emitted) == 1
    assert isinstance(emitted[0], SttSpeechStartedEvent)


async def test_speech_started_callback_exception_does_not_crash_reader_loop():
    """An exception inside on_speech_started must be caught -- the reader
    loop keeps draining subsequent results (e.g. the real final that
    follows the SpeechStarted VAD onset)."""

    async def on_speech_started():
        raise RuntimeError("boom")

    pipeline, seen = _make_pipeline(
        [
            {"speech_started": True},
            {"transcript": "hello there", "confidence": 0.9, "is_final": True},
        ],
        on_speech_started=on_speech_started,
    )

    await _run_reader_briefly(pipeline)

    assert seen["finals"] == [("hello there", 0.9)]


async def test_multiple_speech_started_results_each_invoke_callback():
    calls = []

    async def on_speech_started():
        calls.append(True)

    pipeline, seen = _make_pipeline(
        [{"speech_started": True}, {"speech_started": True}],
        on_speech_started=on_speech_started,
    )

    await _run_reader_briefly(pipeline)

    assert calls == [True, True]
