"""
Phase 4D-2: ElevenLabs `previous_text` continuity race-condition fix.

Bug (Phase 4D confirmed): TtsPipeline.queue_tts() spawns each chunk's
synthesis as an asyncio.Task immediately so chunk N+1 can prefetch/synthesize
while chunk N is still playing. Both `_prefetch_tts_audio` implementations
(TtsStreamMixin for Twilio, LiveKitBrowserCallHandler for LiveKit) used to
read a shared instance attribute (`self._elevenlabs_prev_tts_text`) that was
only mutated AFTER a chunk's audio had fully streamed — post-playback. Since
chunk N+1 typically prefetches while chunk N is still mid-playback, chunk
N+1's ElevenLabs request was built with a STALE previous_text (usually the
turn-start empty value), not chunk N's actual text.

Fix: TtsPipeline now tracks the last QUEUED chunk's text as a plain instance
attribute (`_last_queued_text`), updated synchronously inside queue_tts()
BEFORE the chunk's task is spawned, and threads the captured value through
via `task["_previous_text"]`. Both `_prefetch_tts_audio` implementations now
read `task.get("_previous_text")` instead of the racy shared attribute.

Covers:
  1. First chunk receives previous_text=None.
  2. Chunk N+1 receives chunk N's ACTUAL text even when chunk N has not
     finished playback yet (proves the race is fixed).
  3. Chunk 3 receives chunk 2's text (3-chunk chain).
  4. Back-to-back queue_tts() calls (no await yield in between) preserve
     correct ordering.
  5. Parallel prefetching remains intact — chunk N+1 can run concurrently
     with chunk N still pending.
  6. New turn resets continuity (via _reset_turn_state()).
  7. Barge-in/cancellation does not leak previous text into the next turn.
  8. Quick-ack does not inherit unrelated previous-turn text; the first main
     response chunk DOES inherit the quick-ack's text (documented existing
     playback-order intent — quick-ack genuinely plays immediately before
     the first main chunk).
  9. Twilio (TtsStreamMixin) and LiveKit (LiveKitBrowserCallHandler)
     `_prefetch_tts_audio` both read continuity from the task dict.
  10. Existing synthesis-failure fallback path still functions with the new
      plumbing (task["_previous_text"] is still captured even when
      synthesis errors).
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.agent_runtime import ResolvedTtsRuntime
from app.voice.tts_pipeline import TtsPipeline


def _fake_tts_runtime(adapter_slug: str = "elevenlabs", voice_external_id: str | None = "voice-1"):
    return ResolvedTtsRuntime(
        adapter_slug=adapter_slug,
        voice_external_id=voice_external_id,
        language="en",
        settings_json={},
        used_ticket_tts=False,
    )


class _FakeHandler:
    """Mirrors the _FakeHandler pattern used across tests/voice/test_pacing_tts_pipeline.py."""

    def __init__(self):
        self._tts_cancel = asyncio.Event()
        self._current_turn_user_text = ""
        self._current_turn_stt_confidence = 0.0
        self.prefetch_calls: list[Dict[str, Any]] = []
        self.stream_calls: list[Dict[str, Any]] = []
        self._prefetch_gate: asyncio.Event | None = None
        self._fail_prefetch = False

    async def _prefetch_tts_audio(self, task: Dict[str, Any]):
        self.prefetch_calls.append(dict(task))
        if self._prefetch_gate is not None:
            await self._prefetch_gate.wait()
        if self._fail_prefetch:
            raise RuntimeError("simulated synthesis failure")
        return b"\xff" * 160

    async def _stream_tts_chunk(
        self,
        text,
        use_ssml=False,
        is_final=False,
        prefetched_bytes=None,
        pacing=None,
        previous_text=None,
    ):
        self.stream_calls.append(
            {
                "text": text,
                "is_final": is_final,
                "prefetched_bytes": prefetched_bytes,
                "previous_text": previous_text,
            }
        )


async def _wait_chunk(pipeline: TtsPipeline, chunk_id: int) -> None:
    t = pipeline._synthesis_tasks.get(chunk_id)
    if t is not None:
        await t


# ---------------------------------------------------------------------------
# 1: first chunk gets previous_text=None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_chunk_receives_none_previous_text():
    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    await pipeline.queue_tts({"text": "First sentence.", "chunk_id": 0, "is_final": True})
    await _wait_chunk(pipeline, 0)

    assert handler.prefetch_calls[0]["_previous_text"] is None
    assert handler.stream_calls[0]["previous_text"] is None


# ---------------------------------------------------------------------------
# 2: chunk N+1 sees chunk N's text even while chunk N hasn't finished playing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chunk_n_plus_1_sees_chunk_n_text_before_chunk_n_playback_finishes():
    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    # Block chunk 0's "playback" (its _prefetch_tts_audio call) on an unset
    # gate, simulating chunk 0 still being mid-stream when chunk 1 is queued.
    chunk0_gate = asyncio.Event()
    handler._prefetch_gate = chunk0_gate

    await pipeline.queue_tts({"text": "Chunk zero text.", "chunk_id": 0, "is_final": False})

    # Chunk 0's task is now running and blocked inside _prefetch_tts_audio
    # (i.e. still "playing" from the caller's perspective). Queue chunk 1
    # RIGHT NOW, before releasing chunk 0.
    handler._prefetch_gate = None  # chunk 1 should not block
    await pipeline.queue_tts({"text": "Chunk one text.", "chunk_id": 1, "is_final": True})
    await _wait_chunk(pipeline, 1)

    # Proven BEFORE chunk 0 is released: chunk 1's captured previous_text is
    # already correct — it did not need to wait for chunk 0's post-playback
    # state update (which, under the old buggy design, hadn't happened yet).
    assert handler.prefetch_calls[1]["_previous_text"] == "Chunk zero text."
    assert not chunk0_gate.is_set()

    # Release chunk 0 so its task can complete cleanly.
    chunk0_gate.set()
    await _wait_chunk(pipeline, 0)


# ---------------------------------------------------------------------------
# 3: 3-chunk chain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chunk_3_receives_chunk_2_text_in_a_three_chunk_chain():
    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    for i, text in enumerate(texts):
        await pipeline.queue_tts({"text": text, "chunk_id": i, "is_final": i == len(texts) - 1})
        await _wait_chunk(pipeline, i)

    assert handler.prefetch_calls[0]["_previous_text"] is None
    assert handler.prefetch_calls[1]["_previous_text"] == "First sentence."
    assert handler.prefetch_calls[2]["_previous_text"] == "Second sentence."


# ---------------------------------------------------------------------------
# 4: back-to-back queue_tts() calls with no await yield between them
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_back_to_back_queue_tts_calls_preserve_ordering_without_yielding():
    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    # queue_tts() captures/updates _last_queued_text synchronously before it
    # ever awaits anything that could yield control back to the loop, so
    # calling it three times in immediate succession (each `await` here
    # resumes instantly — queue_tts never suspends before that point) must
    # still produce a correct chain.
    tasks = [
        pipeline.queue_tts({"text": "Alpha.", "chunk_id": 0, "is_final": False}),
        pipeline.queue_tts({"text": "Bravo.", "chunk_id": 1, "is_final": False}),
        pipeline.queue_tts({"text": "Charlie.", "chunk_id": 2, "is_final": True}),
    ]
    for coro in tasks:
        await coro

    for i in range(3):
        await _wait_chunk(pipeline, i)

    assert handler.prefetch_calls[0]["_previous_text"] is None
    assert handler.prefetch_calls[1]["_previous_text"] == "Alpha."
    assert handler.prefetch_calls[2]["_previous_text"] == "Bravo."


# ---------------------------------------------------------------------------
# 5: parallel prefetching remains intact (not serialized)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_prefetching_is_not_serialized_by_the_fix():
    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    chunk0_running = asyncio.Event()
    chunk0_release = asyncio.Event()
    chunk1_running = asyncio.Event()
    chunk1_release = asyncio.Event()

    async def _prefetch(task: Dict[str, Any]):
        handler.prefetch_calls.append(dict(task))
        if task["text"] == "Chunk zero text.":
            chunk0_running.set()
            await chunk0_release.wait()
        else:
            chunk1_running.set()
            await chunk1_release.wait()
        return b"\xff" * 160

    handler._prefetch_tts_audio = _prefetch

    await pipeline.queue_tts({"text": "Chunk zero text.", "chunk_id": 0, "is_final": False})
    await asyncio.wait_for(chunk0_running.wait(), timeout=1.0)

    await pipeline.queue_tts({"text": "Chunk one text.", "chunk_id": 1, "is_final": True})
    # Chunk 1's prefetch starts and runs CONCURRENTLY with chunk 0's still
    # pending prefetch — proving synthesis parallelism is untouched by the
    # previous_text fix (chunk 0 has not been released yet).
    await asyncio.wait_for(chunk1_running.wait(), timeout=1.0)
    assert not chunk0_release.is_set()

    chunk0_release.set()
    chunk1_release.set()
    await _wait_chunk(pipeline, 0)
    await _wait_chunk(pipeline, 1)

    assert handler.prefetch_calls[1]["_previous_text"] == "Chunk zero text."


# ---------------------------------------------------------------------------
# 6: new turn resets continuity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_new_turn_reset_clears_previous_text_for_first_chunk_of_next_turn():
    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    await pipeline.queue_tts({"text": "Turn one, last chunk.", "chunk_id": 0, "is_final": True})
    await _wait_chunk(pipeline, 0)

    # Simulate the turn boundary: cancel_current_and_clear_queue() is what
    # every real caller invokes before starting a new turn (barge-in guard
    # in _complete_llm_turn_after_stt_final / LiveKit's on-transcript path).
    await pipeline.cancel_current_and_clear_queue()

    await pipeline.queue_tts({"text": "Turn two, first chunk.", "chunk_id": 0, "is_final": True})
    await _wait_chunk(pipeline, 0)

    assert handler.prefetch_calls[-1]["_previous_text"] is None


# ---------------------------------------------------------------------------
# 7: barge-in mid-chunk does not leak text into the next turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_barge_in_cancellation_does_not_leak_previous_text():
    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    gate = asyncio.Event()
    handler._prefetch_gate = gate

    # Chunk 0 is queued and stuck inside its prefetch (not yet finished).
    await pipeline.queue_tts({"text": "About to be interrupted.", "chunk_id": 0, "is_final": False})

    # Barge-in fires before chunk 0 completes.
    await pipeline.cancel_current_and_clear_queue()
    gate.set()  # let the now-cancelled/zombie task unwind if it wasn't cancelled fast enough
    handler._tts_cancel.clear()  # real callers always clear before the next turn

    handler._prefetch_gate = None
    await pipeline.queue_tts({"text": "New turn after barge-in.", "chunk_id": 0, "is_final": True})
    await _wait_chunk(pipeline, 0)

    assert handler.prefetch_calls[-1]["_previous_text"] is None
    assert handler.prefetch_calls[-1]["text"] == "New turn after barge-in."


# ---------------------------------------------------------------------------
# 8: quick-ack continuity behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quick_ack_does_not_inherit_unrelated_previous_turn_text():
    """
    Real callers always reset the turn (cancel_current_and_clear_queue, per
    _cancel_inflight_llm_response) before queuing a new turn's quick-ack —
    see conversation_orchestrator.generate_and_stream_response(), which
    resets _elevenlabs_prev_tts_text/_tts_cancel BEFORE calling
    send_quick_acknowledgement(). So the quick-ack, as the first chunk
    queued in a fresh turn, must get previous_text=None regardless of what
    the previous turn's last chunk said.
    """
    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    await pipeline.queue_tts({"text": "Previous turn's final words.", "chunk_id": 0, "is_final": True})
    await _wait_chunk(pipeline, 0)

    await pipeline.cancel_current_and_clear_queue()  # turn boundary reset
    # Real callers always clear cancel_event before starting the next turn
    # (e.g. conversation_orchestrator.generate_and_stream_response /
    # _complete_llm_turn_after_stt_final) — mirror that here.
    handler._tts_cancel.clear()

    await pipeline.queue_tts(
        {"text": "Got it", "chunk_id": "quick_ack", "is_acknowledgement": True, "is_final": False}
    )
    await _wait_chunk(pipeline, 0)  # quick-ack becomes chunk_id 0 of the new turn

    assert handler.prefetch_calls[-1]["_previous_text"] is None
    assert handler.prefetch_calls[-1]["text"] == "Got it"


@pytest.mark.asyncio
async def test_main_response_first_chunk_inherits_quick_ack_text():
    """
    Documented existing architectural intent: quick-ack is queued as its own
    TtsPipeline chunk BEFORE the main response chunks (see
    ConversationOrchestrator.generate_and_stream_response -> send_quick_
    acknowledgement, called before the LLM streaming loop queues any main
    chunks). Nothing in the codebase special-cases "is_acknowledgement" to
    exclude it from continuity tracking, and the quick-ack genuinely plays
    immediately before the first main chunk — so the first main chunk
    correctly inherits the quick-ack's text as previous_text once queued.
    """
    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    await pipeline.queue_tts(
        {"text": "Got it", "chunk_id": "quick_ack", "is_acknowledgement": True, "is_final": False}
    )
    await _wait_chunk(pipeline, 0)

    await pipeline.queue_tts({"text": "Here is your answer.", "chunk_id": 1, "is_final": True})
    await _wait_chunk(pipeline, 1)

    assert handler.prefetch_calls[0]["_previous_text"] is None
    assert handler.prefetch_calls[1]["_previous_text"] == "Got it"


# ---------------------------------------------------------------------------
# 9: Twilio and LiveKit both read continuity from the task dict
# ---------------------------------------------------------------------------


def _twilio_mixin_handler():
    from app.voice.tts_stream_mixin import TtsStreamMixin

    h = object.__new__(TtsStreamMixin)
    h._tts_cancel = asyncio.Event()
    h._elevenlabs_prev_tts_text = ""  # legacy attribute — must NOT be read anymore
    h.agent = MagicMock()
    h.agent.language = "en"
    h.agent.voice_type = "female"
    h.agent.tts_voice = None
    h.db = None
    return h


def _livekit_mixin_handler():
    from app.voice.livekit_browser_call_handler import LiveKitBrowserCallHandler

    db = MagicMock()
    call_session = MagicMock()
    call_session.id = "cs-1"
    agent = MagicMock()
    agent.language = "en"
    agent.voice_type = "female"
    agent.greeting_message = "hi"
    agent.first_message = None
    agent.tts_voice = None
    h = LiveKitBrowserCallHandler(db=db, call_session=call_session, agent=agent, call_flow=None)
    h._elevenlabs_prev_tts_text = "STALE-SHOULD-NOT-BE-USED"  # legacy attribute
    return h


async def _drain(result):
    if hasattr(result, "__aiter__"):
        async for _ in result:
            pass


@pytest.mark.asyncio
async def test_twilio_prefetch_reads_previous_text_from_task_not_stale_attribute():
    h = _twilio_mixin_handler()
    h._elevenlabs_prev_tts_text = "STALE-SHOULD-NOT-BE-USED"

    captured: Dict[str, Any] = {}

    async def fake_stream(**kwargs):
        captured.update(kwargs)
        return
        yield  # pragma: no cover

    mock_adapter = MagicMock()
    mock_adapter.async_stream_synthesize = fake_stream

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ), patch("app.voice.tts_stream_mixin.get_tts_adapter", return_value=mock_adapter):
        result = await h._prefetch_tts_audio(
            {"text": "Chunk one text.", "_previous_text": "Chunk zero text."}
        )
        await _drain(result)

    assert captured.get("settings_json", {}).get("previous_text") == "Chunk zero text."


@pytest.mark.asyncio
async def test_livekit_prefetch_reads_previous_text_from_task_not_stale_attribute():
    h = _livekit_mixin_handler()

    captured: Dict[str, Any] = {}

    async def fake_stream(**kwargs):
        captured.update(kwargs)
        return
        yield  # pragma: no cover

    mock_adapter = MagicMock()
    mock_adapter.async_stream_synthesize = fake_stream

    with patch(
        "app.core.agent_runtime.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ), patch("app.utils.tts_adapter.get_tts_adapter", return_value=mock_adapter):
        result = await h._prefetch_tts_audio(
            {"text": "Chunk one text.", "_previous_text": "Chunk zero text."}
        )
        await _drain(result)

    assert captured.get("settings_json", {}).get("previous_text") == "Chunk zero text."


@pytest.mark.asyncio
async def test_twilio_prefetch_omits_previous_text_when_task_has_none():
    h = _twilio_mixin_handler()

    captured: Dict[str, Any] = {}

    async def fake_stream(**kwargs):
        captured.update(kwargs)
        return
        yield  # pragma: no cover

    mock_adapter = MagicMock()
    mock_adapter.async_stream_synthesize = fake_stream

    with patch(
        "app.voice.tts_stream_mixin.resolve_tts_runtime",
        return_value=_fake_tts_runtime("elevenlabs"),
    ), patch("app.voice.tts_stream_mixin.get_tts_adapter", return_value=mock_adapter):
        result = await h._prefetch_tts_audio({"text": "First chunk.", "_previous_text": None})
        await _drain(result)

    assert "previous_text" not in captured.get("settings_json", {})


# ---------------------------------------------------------------------------
# 10: existing synthesis-failure fallback still works with the new plumbing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesis_failure_still_streams_with_prefetched_bytes_none():
    handler = _FakeHandler()
    handler._fail_prefetch = True
    pipeline = TtsPipeline(handler)

    await pipeline.queue_tts({"text": "First chunk.", "chunk_id": 0, "is_final": False})
    await _wait_chunk(pipeline, 0)

    await pipeline.queue_tts({"text": "Second chunk.", "chunk_id": 1, "is_final": True})
    await _wait_chunk(pipeline, 1)

    # _process_chunk swallows the synthesis exception (audio_bytes=None) and
    # still calls _stream_tts_chunk so playback/fallback logic can run.
    assert len(handler.stream_calls) == 2
    assert handler.stream_calls[0]["prefetched_bytes"] is None
    # Continuity tracking is unaffected by synthesis failure — chunk 1 still
    # correctly captured chunk 0's text (captured synchronously at queue
    # time, independent of whether chunk 0's synthesis ever succeeds).
    assert handler.stream_calls[1]["previous_text"] == "First chunk."


# ---------------------------------------------------------------------------
# Phase 4D-2b (code review follow-up): TtsPipeline.reset_previous_text_
# continuity() must fire UNCONDITIONALLY every turn, not just on the barge-in
# path (cancel_current_and_clear_queue -> _reset_turn_state). On LiveKit,
# LiveKitBrowserCallHandler._process_transcript() only calls
# _cancel_inflight_llm_response()/cancel_current_and_clear_queue() when
# `self._is_tts_playing` is True — i.e. only on an actual barge-in. On a
# normal (non-barge-in) turn it goes straight to
# _complete_llm_turn_after_stt_final() -> generate_and_stream_response()
# without ever resetting _last_queued_text via that path. The fix restores
# the equivalent-guarantee reset directly in
# ConversationOrchestrator.generate_and_stream_response() (shared by both
# transports) via the new TtsPipeline.reset_previous_text_continuity().
# ---------------------------------------------------------------------------


def test_reset_previous_text_continuity_touches_only_last_queued_text():
    """
    The dedicated method must be surgical: clear _last_queued_text only,
    leaving gate-chain / _turn_id / is_speaking state (which must NOT be
    disturbed outside of an actual cancel) completely untouched.
    """
    handler = _FakeHandler()
    pipeline = TtsPipeline(handler)

    pipeline._last_queued_text = "some previously queued text"
    pipeline._turn_id = 5
    existing_gate = asyncio.Event()
    pipeline._playback_events = {7: existing_gate}
    pipeline._synthesis_tasks = {7: object()}  # sentinel — must survive untouched

    pipeline.reset_previous_text_continuity()

    assert pipeline._last_queued_text is None
    assert pipeline._turn_id == 5
    assert pipeline._playback_events == {7: existing_gate}
    assert pipeline._synthesis_tasks == {7: pipeline._synthesis_tasks[7]}
    assert 7 in pipeline._synthesis_tasks


class _StubLlmService:
    """Fake LLM boundary — mirrors the real service.stream_text(...) async
    generator signature/kwargs used by ConversationOrchestrator.try_stream."""

    def __init__(self, sentence: str = "Default reply."):
        self.next_sentence = sentence

    async def stream_text(self, **kwargs):
        yield self.next_sentence


class _FalsyCallSession:
    """
    A call_session that is falsy (`__bool__` -> False) so every
    `if self._h.call_session and self._h.db:`-guarded DB-dependent
    prompt-building branch inside ConversationOrchestrator.
    generate_and_stream_response() (CRM context, KB, caller memory, HubSpot
    field mapping, business-knowledge fetch) is skipped exactly the way it
    would be for a call with no persisted session — keeping this test's only
    real, unmocked dependencies at the LLM and TTS-provider boundaries,
    while still exercising the REAL turn-boundary control flow.
    """

    def __init__(self, session_id: str):
        self.id = session_id
        self.tenant_id = None
        self.call_transcript = None
        self.call_metadata = None
        self.user_id = None

    def __bool__(self) -> bool:
        return False


def _real_livekit_handler_for_turn_flow():
    """
    Build a REAL LiveKitBrowserCallHandler + REAL TtsPipeline +
    ConversationOrchestrator (LiveKitBrowserCallHandler.__init__ constructs
    `self._conversation = ConversationOrchestrator(self)` itself — not
    mocked). Only `_prefetch_tts_audio` (the TTS-provider boundary) is
    replaced with a lightweight recorder; `_stream_tts_chunk` runs for real
    against a mocked LiveKit publisher.
    """
    from app.voice.livekit_browser_call_handler import LiveKitBrowserCallHandler

    call_session = _FalsyCallSession("cs-turn-flow-1")
    h = LiveKitBrowserCallHandler(db=MagicMock(), call_session=call_session, agent=None, call_flow=None)
    h._tts_pipeline = TtsPipeline(h)

    publisher = MagicMock()
    publisher.connected = True
    publisher.publish_mulaw = AsyncMock()
    h._agent_publisher = publisher

    prefetch_calls: list[Dict[str, Any]] = []

    async def _fake_prefetch(task: Dict[str, Any]):
        prefetch_calls.append(dict(task))
        return b"\xff" * 160

    h._prefetch_tts_audio = _fake_prefetch
    return h, prefetch_calls


@pytest.mark.asyncio
async def test_livekit_real_turn_boundary_normal_turn_resets_previous_text(monkeypatch):
    """
    THE regression test the code review required: drives TWO full,
    non-barge-in turns through the REAL LiveKitBrowserCallHandler entry
    point — _process_transcript() -> _complete_llm_turn_after_stt_final() ->
    generate_and_stream_response() -> ConversationOrchestrator ->
    TtsPipeline.queue_tts() — proving the fix fires even though
    cancel_current_and_clear_queue() is NEVER invoked between these two
    turns (LiveKit only calls it when `_is_tts_playing` is True at
    _process_transcript() entry — i.e. only on an actual barge-in). Only the
    LLM and TTS-provider boundaries are mocked; everything else is real.
    """
    # Quick-ack is probability-gated; force it off so chunk numbering/text is
    # deterministic and unrelated to what this test is proving.
    monkeypatch.setattr("app.voice.conversation_orchestrator.random.random", lambda: 2.0)

    h, prefetch_calls = _real_livekit_handler_for_turn_flow()

    llm = _StubLlmService()
    monkeypatch.setattr("app.core.agent_runtime.llm_service_for_provider", lambda slug: llm)

    # ── Turn 1: normal turn, agent was silent beforehand (no barge-in) ────
    assert h._is_tts_playing is False
    llm.next_sentence = "Turn one full reply text right here."
    await h._process_transcript("First user turn, please help me today.", 0.9)

    assert len(prefetch_calls) >= 1, "turn 1 must have queued at least one real TtsPipeline chunk"
    # Turn 1 finished playing (real _stream_tts_chunk's finally block resets
    # this) — confirms turn 2 below is a genuine non-barge-in turn, matching
    # _process_transcript()'s `if self._is_tts_playing:` guard being False.
    assert h._is_tts_playing is False
    turn1_last_text = prefetch_calls[-1]["text"]
    assert turn1_last_text  # sanity: turn 1 actually queued real, non-empty text

    # ── Turn 2: normal turn — NO barge-in occurred, so
    # cancel_current_and_clear_queue() was never called between turns. ─────
    turn2_start_idx = len(prefetch_calls)
    llm.next_sentence = "Turn two full reply text right here, unrelated to turn one."
    await h._process_transcript("Second user turn, something else entirely.", 0.9)

    assert len(prefetch_calls) > turn2_start_idx, "turn 2 must have queued at least one chunk"
    # THE bug this regression test guards against: without the fix, turn 2's
    # first chunk would incorrectly inherit turn 1's last chunk text as
    # previous_text, because nothing would have reset
    # TtsPipeline._last_queued_text between these two non-barge-in turns.
    assert prefetch_calls[turn2_start_idx]["_previous_text"] is None
    assert prefetch_calls[turn2_start_idx]["text"] != turn1_last_text


@pytest.mark.asyncio
async def test_livekit_real_turn_boundary_barge_in_turn_still_resets_previous_text(monkeypatch):
    """
    Companion case: when a genuine barge-in DOES occur between turns (agent
    still speaking when the next final transcript arrives),
    cancel_current_and_clear_queue() fires as before and must still reset
    correctly — proving the barge-in path keeps working alongside the new
    unconditional per-turn reset.
    """
    monkeypatch.setattr("app.voice.conversation_orchestrator.random.random", lambda: 2.0)

    h, prefetch_calls = _real_livekit_handler_for_turn_flow()

    llm = _StubLlmService()
    monkeypatch.setattr("app.core.agent_runtime.llm_service_for_provider", lambda slug: llm)

    llm.next_sentence = "Turn one full reply text right here."
    await h._process_transcript("First user turn, please help me today.", 0.9)
    assert h._is_tts_playing is False

    # Simulate the agent still speaking when the caller's next final
    # transcript arrives — this is what actually drives
    # _process_transcript()'s `if self._is_tts_playing:` -> barge-in branch.
    h._is_tts_playing = True

    turn2_start_idx = len(prefetch_calls)
    llm.next_sentence = "Turn two full reply text right here, unrelated to turn one."
    await h._process_transcript("Second user turn, interrupting.", 0.9)

    assert len(prefetch_calls) > turn2_start_idx
    assert prefetch_calls[turn2_start_idx]["_previous_text"] is None
