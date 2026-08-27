"""SttPipeline._reader_loop -> SttSpeechStartedEvent propagation.

Verifies the acoustic/VAD-onset result shape pushed by
DeepgramSTTService.StreamingSTTSession (`{"speech_started": True}`) is
translated into a typed SttSpeechStartedEvent on SttPipeline's event bus,
without invoking the interim/final callbacks and without disturbing normal
transcript event flow.
"""
from __future__ import annotations

import asyncio

import pytest

from app.voice.stt_events import (
    SttEventBus,
    SttFinalEvent,
    SttInterimEvent,
    SttSpeechStartedEvent,
)
from app.voice.stt_pipeline import SttPipeline


class _FakeSttSession:
    """Minimal stand-in for DeepgramSTTService.StreamingSTTSession."""

    def __init__(self, results: list[dict]) -> None:
        self._results = list(results)

    async def get_result(self) -> dict:
        if self._results:
            return self._results.pop(0)
        # Block forever once drained (reader loop awaits get_result in a tight
        # while True loop) -- the test cancels the reader task explicitly.
        await asyncio.sleep(3600)
        return {}


@pytest.mark.asyncio
async def test_speech_started_result_emits_typed_event_only():
    events: list = []
    bus = SttEventBus()

    async def _capture(event):
        events.append(event)

    bus.subscribe(_capture)

    interim_calls = []
    final_calls = []

    async def on_interim(t, c):
        interim_calls.append((t, c))

    async def on_final(t, c):
        final_calls.append((t, c))

    pipeline = SttPipeline(
        language_code="en",
        on_interim=on_interim,
        on_final=on_final,
        event_bus=bus,
    )
    pipeline._stt_session = _FakeSttSession([{"speech_started": True}])
    pipeline._reader_task = asyncio.create_task(pipeline._reader_loop())

    await asyncio.sleep(0.05)
    pipeline._reader_task.cancel()
    try:
        await pipeline._reader_task
    except asyncio.CancelledError:
        pass

    assert len(events) == 1
    assert isinstance(events[0], SttSpeechStartedEvent)
    assert events[0].type == "speech_started"
    # SpeechStarted must never invoke transcript callbacks -- no text, no turn.
    assert interim_calls == []
    assert final_calls == []


@pytest.mark.asyncio
async def test_speech_started_interleaved_with_normal_interim_final_flow():
    events: list = []
    bus = SttEventBus()

    async def _capture(event):
        events.append(event)

    bus.subscribe(_capture)

    async def on_interim(t, c):
        pass

    async def on_final(t, c):
        pass

    pipeline = SttPipeline(
        language_code="en",
        on_interim=on_interim,
        on_final=on_final,
        event_bus=bus,
    )
    pipeline._stt_session = _FakeSttSession(
        [
            {"speech_started": True},
            {"transcript": "hello", "confidence": 0.5, "is_final": False},
            {"transcript": "hello there", "confidence": 0.9, "is_final": True},
        ]
    )
    pipeline._reader_task = asyncio.create_task(pipeline._reader_loop())

    await asyncio.sleep(0.05)
    pipeline._reader_task.cancel()
    try:
        await pipeline._reader_task
    except asyncio.CancelledError:
        pass

    event_types = [type(e) for e in events]
    assert event_types == [SttSpeechStartedEvent, SttInterimEvent, SttFinalEvent]
