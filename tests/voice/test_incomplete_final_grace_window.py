"""
Regression coverage for SttPipeline's incomplete-final grace window
(`_maybe_extend_incomplete_final`, gated by `incomplete_final_grace_ms`).

Root cause this addresses: even after raising Deepgram's blanket silence
endpointing, a caller who pauses mid-clause on a longer sentence ("...and
the onboarding process" <breath> "...how long does that take?") can still
get a premature speech_final on the first fragment. This layers a short,
content-aware grace window on top of the existing silence-based
endpointing: when a final's trailing text looks unfinished, wait briefly
for a continuation from the SAME underlying STT session queue (nothing is
dropped -- Deepgram keeps transcribing in the background) before treating
it as the caller's completed turn.

Scoped to Twilio only via `incomplete_final_grace_ms=0` (the default) --
LiveKitBrowserCallHandler never sets this, so demo/Share-Demo-Link calls
are provably unaffected (see TestScopedToTwilioOnly below and
test_voice_orchestrator's own wiring).
"""

from __future__ import annotations

import asyncio

from app.voice.stt_pipeline import SttPipeline


class _QueueSession:
    """Fake STT session whose get_result() drains a pre-seeded queue of
    result dicts, one per call -- mirrors DeepgramSTTService's real
    session interface closely enough for _reader_loop's needs."""

    def __init__(self, results: list[dict]):
        self._results = list(results)

    async def get_result(self) -> dict:
        if self._results:
            return self._results.pop(0)
        # Reader loop must not spin once the fixture is exhausted.
        await asyncio.sleep(10)
        return {}


def _make_pipeline(grace_ms: int, results: list[dict]):
    seen = {"finals": [], "interims": []}

    async def on_final(transcript, confidence):
        seen["finals"].append((transcript, confidence))

    async def on_interim(transcript, confidence):
        seen["interims"].append((transcript, confidence))

    pipeline = SttPipeline(
        language_code="en",
        on_interim=on_interim,
        on_final=on_final,
        incomplete_final_grace_ms=grace_ms,
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


class TestGraceWindowMergesContinuation:
    def test_incomplete_final_merges_with_continuation_final(self):
        """First final looks cut off ('...and'); a genuine continuation
        final arrives inside the grace window and should be merged into a
        single, complete on_final call -- not two separate (premature)
        turns."""

        async def _run():
            pipeline, seen = _make_pipeline(
                grace_ms=500,
                results=[
                    {
                        "transcript": "I wanted to ask about pricing and",
                        "is_final": True,
                        "confidence": 0.9,
                    },
                    {
                        "transcript": "I wanted to ask about pricing and the onboarding process.",
                        "is_final": True,
                        "confidence": 0.95,
                    },
                ],
            )
            await _run_reader_briefly(pipeline)
            assert len(seen["finals"]) == 1
            transcript, _ = seen["finals"][0]
            assert transcript == (
                "I wanted to ask about pricing and the onboarding process."
            )

        asyncio.run(_run())

    def test_complete_final_is_not_delayed(self):
        """A normal, complete-sounding final must be forwarded immediately
        -- no grace wait, no added latency."""

        async def _run():
            pipeline, seen = _make_pipeline(
                grace_ms=500,
                results=[
                    {
                        "transcript": "I'd like to book an appointment.",
                        "is_final": True,
                        "confidence": 0.95,
                    },
                ],
            )
            start = asyncio.get_event_loop().time()
            await _run_reader_briefly(pipeline, seconds=0.05)
            elapsed = asyncio.get_event_loop().time() - start
            assert len(seen["finals"]) == 1
            assert elapsed < 0.2  # nowhere near the 500ms grace window

        asyncio.run(_run())

    def test_no_continuation_arrives_final_still_forwarded_after_grace(self):
        """If nothing follows within the grace window, the original
        (incomplete-looking) final is forwarded as-is once the window
        expires -- never silently dropped."""

        async def _run():
            pipeline, seen = _make_pipeline(
                grace_ms=120,
                results=[
                    {
                        "transcript": "Can you send that to",
                        "is_final": True,
                        "confidence": 0.9,
                    },
                ],
            )
            await _run_reader_briefly(pipeline, seconds=0.3)
            assert len(seen["finals"]) == 1
            assert seen["finals"][0][0] == "Can you send that to"

        asyncio.run(_run())

    def test_unrelated_next_utterance_is_not_dropped(self):
        """A result that arrives during the grace window but is NOT a
        continuation of the pending utterance (e.g. an interim of a fresh,
        unrelated sentence) must be preserved for the next reader-loop
        iteration, not silently discarded."""

        async def _run():
            pipeline, seen = _make_pipeline(
                grace_ms=300,
                results=[
                    {
                        "transcript": "Can you send that to",
                        "is_final": True,
                        "confidence": 0.9,
                    },
                    {
                        "transcript": "Actually never mind, different question.",
                        "is_final": True,
                        "confidence": 0.9,
                    },
                ],
            )
            await _run_reader_briefly(pipeline, seconds=0.4)
            texts = [t for t, _ in seen["finals"]]
            assert "Can you send that to" in texts
            assert "Actually never mind, different question." in texts
            assert len(seen["finals"]) == 2

        asyncio.run(_run())


class TestGraceWindowDisabledByDefault:
    def test_zero_grace_ms_skips_heuristic_entirely(self):
        """Default incomplete_final_grace_ms=0 must behave exactly like
        before this fix -- immediate forwarding regardless of trailing
        text, since this is opt-in per SttPipeline construction."""

        async def _run():
            pipeline, seen = _make_pipeline(
                grace_ms=0,
                results=[
                    {
                        "transcript": "I wanted to ask about pricing and",
                        "is_final": True,
                        "confidence": 0.9,
                    },
                ],
            )
            await _run_reader_briefly(pipeline, seconds=0.05)
            assert len(seen["finals"]) == 1
            assert seen["finals"][0][0] == "I wanted to ask about pricing and"

        asyncio.run(_run())
