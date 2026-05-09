"""
TtsPipeline: parallel TTS synthesis with gapless event-chain playback.

Architecture (V2-style, ported from feature/VoiceOrchestrator TTSStreamManager):
- One asyncio.Task per text chunk (no long-running worker loops to poll)
- Synthesis starts IMMEDIATELY in each task — runs in parallel with the previous
  chunk's Twilio frame streaming because the event loop runs both concurrently
- Ordered playback enforced via _playback_events[chunk_id] gate chain:
    chunk 0 gate  — set immediately in queue_tts (first chunk plays right away)
    chunk N gate  — set by chunk N-1's finally block once streaming ends
  This guarantees gapless, in-order audio with zero inter-chunk polling overhead.
- _turn_has_final / _turn_end_call_after: set at queue_tts() time so
  end_call_after fires correctly even if the final-chunk task errors out first.
- LRU audio cache (256 entries): repeated phrases (acks, etc.) skip TTS API.
- BoundedSemaphore(_MAX_CONCURRENT_SYNTHESIS=8): caps in-flight TTS API calls
  so a fast LLM can't OOM the process with 20+ simultaneous synthesis requests.

Race-condition safety:
- _turn_id (monotonically incremented on every _reset_turn_state call) is
  captured by each task at creation time and checked in the finally block before
  any mutation of _playback_events or _synthesis_tasks.  This prevents zombie
  tasks (orphaned by the 250ms cancel timeout) from corrupting the new turn's
  gate chain — a real scenario when ElevenLabs sync IO can't be interrupted.
- gate.wait() is wrapped in asyncio.wait_for(_GATE_WAIT_TIMEOUT_S) so a stuck
  gate never permanently stalls a downstream chunk.  Recovery: log an error and
  proceed — _tts_lock in _stream_tts_chunk serialises concurrent attempts.

Public API (unchanged — handler calls these without modification):
    queue_tts(task)                   — enqueue a text chunk
    cancel_current_and_clear_queue()  — barge-in: cancel all, reset state
    shutdown()                        — graceful teardown
    cancel_event   (property)         — handler's _tts_cancel Event
    is_speaking    (property)         — True while any task is active
    _worker_task   (property)         — backward-compat alias (always None)
    clear_audio_cache()               — evict all cached bytes
"""

import asyncio
import time
from typing import Any, Dict, Optional

from app.core.logger import logger

# ── Tuning constants ──────────────────────────────────────────────────────────

# Cap simultaneous TTS API calls so a burst of LLM chunks doesn't exhaust the
# thread pool or network connections.
_MAX_CONCURRENT_SYNTHESIS: int = 8

# How long cancel_current_and_clear_queue() waits for tasks to finish.
# Must be short enough to keep barge-in snappy; ElevenLabs sync IO can't be
# cancelled — tasks stuck there will be orphaned after this timeout.
_CANCEL_GATHER_TIMEOUT_S: float = 0.25

# Maximum time a chunk may wait for its playback gate before we give up on
# streaming this chunk (still unblock the next chunk in finally). Long utterances
# can legitimately play >8s; proceeding mid-wait caused audible glitches.
_GATE_WAIT_TIMEOUT_S: float = 45.0

# Log a WARNING when synthesis takes longer than this (TTS API latency alert).
_SLOW_SYNTHESIS_WARN_S: float = 3.0

# Log a WARNING when a gate is held open longer than this (synthesis not
# overlapping playback — prefetch ineffective for this chunk).
_SLOW_GATE_WARN_S: float = 1.0


class TtsPipeline:

    _CACHE_MAX: int = 256

    def __init__(self, handler: Any) -> None:
        self._handler = handler

        # Task registry: chunk_id → asyncio.Task
        self._synthesis_tasks: Dict[int, asyncio.Task] = {}
        self._next_chunk_id: int = 0

        # Monotonically increasing turn counter.  Incremented on every
        # _reset_turn_state() call (i.e. every barge-in).  Each task captures
        # the current value at creation time and checks it in the finally block
        # before touching _playback_events or _synthesis_tasks — preventing
        # zombie tasks from corrupting a new turn's gate chain.
        self._turn_id: int = 0

        # Ordered-playback gate chain.
        # _playback_events[0] is an unset Event created here (and in _reset_turn_state).
        # queue_tts sets gate 0 when chunk 0 is enqueued.
        # Each task sets gate chunk_id+1 in its finally block — releasing the next
        # task immediately after streaming ends, with no polling or sleep.
        self._playback_events: Dict[int, asyncio.Event] = {0: asyncio.Event()}

        # Turn-level completion flags — set at queue_tts() time, NOT per-task.
        # Decouples "which task finishes last" from "which task had is_final=True",
        # so end_call_after fires reliably even when the final chunk's TTS API
        # call errors out before earlier chunks finish.
        self._turn_has_final: bool = False
        self._turn_end_call_after: bool = False
        self._turn_transfer_after: bool = False

        # Per-turn observability counters (reset in _reset_turn_state).
        self._turn_chunk_count: int = 0
        self._turn_cache_hits: int = 0
        self._turn_start_ts: float = 0.0

        # LRU audio cache: normalised text → raw μ-law bytes
        self._audio_cache: Dict[str, bytes] = {}

        # Backpressure: limits simultaneous TTS API calls.
        # Acquired inside _process_chunk before synthesis; released after.
        # queue_tts() always returns instantly — tasks queue up inside themselves.
        self._synthesis_semaphore: asyncio.BoundedSemaphore = asyncio.BoundedSemaphore(
            _MAX_CONCURRENT_SYNTHESIS
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def _worker_task(self) -> None:
        """Backward-compat property — no persistent worker loop in V2 style."""
        return None

    @property
    def cancel_event(self) -> asyncio.Event:
        return self._handler._tts_cancel  # type: ignore[attr-defined]

    @property
    def is_speaking(self) -> bool:
        return bool(self._synthesis_tasks)

    async def queue_tts(self, task: Dict[str, Any]) -> None:
        """
        Enqueue a text chunk for parallel synthesis + ordered playback.

        Spawns an asyncio.Task immediately so TTS synthesis starts in the
        background while the previous chunk is still streaming to Twilio.
        Returns instantly — never blocks the caller.

        task dict fields:
          text (str), use_ssml (bool), is_final (bool),
          end_call_after (bool, optional), transfer_after (bool, optional)
        """
        text = (task.get("text") or "").strip()
        is_final = bool(task.get("is_final", False))
        if not text:
            if self.cancel_event.is_set():
                return
            if is_final:
                self._turn_has_final = True
                self._turn_end_call_after = bool(task.get("end_call_after", False))
                self._turn_transfer_after = self._turn_transfer_after or bool(
                    task.get("transfer_after", False)
                )
                if not self._synthesis_tasks:
                    asyncio.create_task(self._run_turn_completion_hooks())
            return
        if self.cancel_event.is_set():
            return

        chunk_id = self._next_chunk_id
        self._next_chunk_id += 1

        # Track turn start time on the first chunk so we can log total duration.
        if chunk_id == 0:
            self._turn_start_ts = time.perf_counter()
            self._turn_chunk_count = 0
            self._turn_cache_hits = 0

        self._turn_chunk_count += 1

        # Gate for this chunk must exist before the task starts.
        # For chunk 0 after _reset_turn_state it was pre-created (unset).
        # For chunk N>0 it was pre-created by chunk N-1's queue_tts call below.
        # The defensive branch handles any unexpected re-sequencing.
        if chunk_id not in self._playback_events:
            e = asyncio.Event()
            e.set()  # allow immediate play if gate was somehow missing
            self._playback_events[chunk_id] = e
            logger.warning(
                "[TTS] missing playback gate for chunk %d — "
                "reset_turn_state may have been skipped",
                chunk_id,
            )

        # Pre-create the gate that will block chunk N+1 until this chunk
        # finishes streaming.  Unconditional: a fresh unset gate is always
        # correct here because chunk N+1 must wait for chunk N regardless of
        # whether a stale entry (from a previous turn) existed.
        self._playback_events[chunk_id + 1] = asyncio.Event()

        # First chunk of a turn: release gate 0 immediately so playback can
        # start as soon as synthesis finishes.  For all later chunks, the
        # gate is released by the previous task's finally block.
        if chunk_id == 0:
            self._playback_events[0].set()

        if is_final:
            self._turn_has_final = True
            self._turn_end_call_after = bool(task.get("end_call_after", False))
            self._turn_transfer_after = self._turn_transfer_after or bool(
                task.get("transfer_after", False)
            )

        # Snapshot turn_id so the task can detect if it's been orphaned by a
        # subsequent barge-in that reset the pipeline while the task was stuck
        # in uncancellable sync IO (ElevenLabs).
        task_turn_id = self._turn_id

        t = asyncio.create_task(
            self._process_chunk(chunk_id, task, task_turn_id),
            name=f"tts_chunk_{chunk_id}",
        )
        self._synthesis_tasks[chunk_id] = t
        # NOTE: do NOT use add_done_callback to pop from _synthesis_tasks.
        # The "turn is done" check in _process_chunk's finally block must pop
        # the task itself BEFORE evaluating whether the dict is empty.
        # A callback fires after the coroutine returns (after finally), making
        # the empty-dict check unreliable.

    async def cancel_current_and_clear_queue(self) -> None:
        """
        Barge-in: cancel every in-flight synthesis task and reset turn state.
        Idempotent and safe to call concurrently.
        """
        if not self.cancel_event.is_set():
            self.cancel_event.set()

        tasks = list(self._synthesis_tasks.values())
        for t in tasks:
            if not t.done():
                t.cancel()

        if tasks:
            # Short timeout so barge-in remains snappy even when a task is stuck
            # inside a blocking ElevenLabs sync iterator (no await → uninterruptible).
            # Tasks that survive the timeout become zombies; _turn_id increment
            # (inside _reset_turn_state below) prevents them from corrupting the
            # new turn's gate chain when they eventually exit.
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=_CANCEL_GATHER_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                still_running = sum(1 for t in tasks if not t.done())
                logger.warning(
                    "[TTS] cancel gather timed out after %.0fms — "
                    "%d task(s) orphaned (zombie-safe via turn_id guard)",
                    _CANCEL_GATHER_TIMEOUT_S * 1000,
                    still_running,
                )

        self._reset_turn_state()

        if hasattr(self._handler, "is_speaking"):
            self._handler.is_speaking = False
        if hasattr(self._handler, "_twilio_buffer_primed"):
            self._handler._twilio_buffer_primed = False
        if hasattr(self._handler, "_prev_tts_tail"):
            self._handler._prev_tts_tail = b""

    def clear_audio_cache(self) -> None:
        """Discard all cached audio bytes (e.g., after voice/provider change)."""
        self._audio_cache.clear()

    async def shutdown(self) -> None:
        """
        Gracefully cancel every task and wait for full teardown.
        Called by VoiceOrchestrator.shutdown().
        """
        try:
            if not self.cancel_event.is_set():
                self.cancel_event.set()
        except Exception:
            pass

        tasks = list(self._synthesis_tasks.values())
        for t in tasks:
            if not t.done():
                t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._synthesis_tasks.clear()
        self._audio_cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_turn_state(self) -> None:
        """
        Reset all per-turn state for a fresh agent turn.
        Called by cancel_current_and_clear_queue (barge-in).

        Incrementing _turn_id is what makes zombie-task protection work:
        any task that was orphaned by the cancel timeout will see a mismatch
        between its captured task_turn_id and self._turn_id in the finally
        block and skip all pipeline mutations.
        """
        self._synthesis_tasks.clear()
        self._next_chunk_id = 0
        self._turn_has_final = False
        self._turn_end_call_after = False
        self._turn_transfer_after = False
        self._turn_chunk_count = 0
        self._turn_cache_hits = 0
        self._turn_start_ts = 0.0
        # Increment BEFORE replacing the gate dict so any zombie task that
        # woke up between the clear() above and the assignment below still
        # sees the old turn_id and bails out.
        self._turn_id += 1
        # Replace the entire gate dict with a single fresh unset gate for chunk 0.
        # Tasks waiting on old gate objects are either already cancelled (gather
        # above ran to completion) or orphaned zombies whose finally blocks will
        # be turn_id-guarded and skip gate mutations.
        self._playback_events = {0: asyncio.Event()}

    async def _run_turn_completion_hooks(self) -> None:
        """Invoke transfer/end-call handlers after a finalized turn (incl. empty-text finals)."""
        try:
            if self.cancel_event.is_set():
                self._turn_has_final = False
                self._turn_end_call_after = False
                self._turn_transfer_after = False
                return
            if not self._turn_has_final:
                return
            transfer = self._turn_transfer_after
            end_call = self._turn_end_call_after
            self._turn_has_final = False
            self._turn_end_call_after = False
            self._turn_transfer_after = False
            if (
                transfer
                and hasattr(self._handler, "_transfer_after_agent_request")
            ):
                try:
                    await self._handler._transfer_after_agent_request()
                except Exception as exc:
                    logger.warning("[TTS] transfer_after error: %s", exc)
            elif (
                end_call
                and hasattr(self._handler, "_end_call_after_agent_request")
            ):
                try:
                    await self._handler._end_call_after_agent_request()
                except Exception as exc:
                    logger.warning("[TTS] end_call_after error: %s", exc)
        except Exception as exc:
            logger.warning("[TTS] turn completion hooks error: %s", exc)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(text: str) -> str:
        return text.lower().strip()

    def _get_cached(self, key: str) -> Optional[bytes]:
        audio = self._audio_cache.get(key)
        if audio is not None:
            del self._audio_cache[key]
            self._audio_cache[key] = audio  # refresh LRU position
        return audio

    def _put_cached(self, key: str, audio: bytes) -> None:
        if key in self._audio_cache:
            del self._audio_cache[key]
        elif len(self._audio_cache) >= self._CACHE_MAX:
            del self._audio_cache[next(iter(self._audio_cache))]
        self._audio_cache[key] = audio

    # ------------------------------------------------------------------
    # Core task: synthesis + ordered playback
    # ------------------------------------------------------------------

    async def _process_chunk(
        self,
        chunk_id: int,
        task: Dict[str, Any],
        task_turn_id: int,
    ) -> None:
        """
        Single asyncio.Task per TTS chunk — the heart of the parallel pipeline.

        Phase 1 — Synthesis (runs immediately in the background):
          Starts as soon as queue_tts spawns this task.  The event loop runs
          _prefetch_tts_audio concurrently with the previous chunk's frame
          streaming, so audio is typically ready before the gate opens.
          BoundedSemaphore limits concurrent API calls to _MAX_CONCURRENT_SYNTHESIS.

        Phase 2 — Ordered playback gate (asyncio.wait_for(gate.wait(), timeout)):
          Blocks until the previous chunk's finally block sets this gate,
          guaranteeing strict in-order audio without any polling or sleep.
          If synthesis finished before the gate opens (the common case), playback
          starts the instant the event loop processes the gate.set() call.
          Timeout (_GATE_WAIT_TIMEOUT_S) avoids infinite hang on zombie gates.
          On timeout we skip streaming this chunk only (no overlap with audio
          still playing from the previous chunk); finally still unblocks the next.

        Phase 3 — Streaming:
          Calls handler._stream_tts_chunk with pre-synthesised bytes.
          The handler checks cancel_event per 20ms MULAW frame so barge-in
          stops audio within one frame period.

        Finally (always runs — error, cancel, or normal completion):
          Guard: skip all pipeline mutations if task_turn_id != self._turn_id.
          This is the zombie-task protection — see _reset_turn_state docstring.
          If guard passes:
            1. Sets gate chunk_id+1 unconditionally so the next chunk is never
               permanently blocked by this chunk's error or cancellation.
            2. Pops this chunk from _synthesis_tasks.
            3. When _synthesis_tasks is empty AND _turn_has_final is set, fires
               end_call_after.  Using _turn_has_final (set at queue_tts time)
               rather than a per-task parameter ensures correctness even when
               the is_final chunk's TTS call errors before earlier chunks end.
        """
        try:
            if self.cancel_event.is_set():
                return

            text = task.get("text", "")
            use_ssml = task.get("use_ssml", False)
            is_final = task.get("is_final", False)

            # ── Phase 1: Synthesis ─────────────────────────────────────────────
            # Runs in parallel with previous chunks' Twilio frame streaming.
            t0 = time.perf_counter()
            cache_key = self._cache_key(text)
            audio_bytes: Any = self._get_cached(cache_key)

            if audio_bytes is not None:
                logger.debug("[TTS] cache hit chunk %d '%.25s'", chunk_id, text)
                # Track cache hits for the per-turn summary (turn_id guard not
                # needed here — only increments, worst case slightly off on zombie).
                self._turn_cache_hits += 1
            else:
                async with self._synthesis_semaphore:
                    if self.cancel_event.is_set():
                        return
                    try:
                        audio_bytes = await self._handler._prefetch_tts_audio(task)  # type: ignore[attr-defined]
                    except asyncio.CancelledError:
                        raise  # propagate — do not swallow
                    except Exception as exc:
                        logger.warning(
                            "[TTS] synthesis error chunk %d: %s", chunk_id, exc
                        )
                        audio_bytes = None

                synth_elapsed = time.perf_counter() - t0
                if synth_elapsed > _SLOW_SYNTHESIS_WARN_S:
                    logger.warning(
                        "[TTS] slow synthesis chunk %d: %.2fs (threshold %.1fs)",
                        chunk_id, synth_elapsed, _SLOW_SYNTHESIS_WARN_S,
                    )
                else:
                    logger.debug(
                        "[TTS] synthesis chunk %d done in %.2fs", chunk_id, synth_elapsed
                    )

            if isinstance(audio_bytes, bytes):
                self._put_cached(cache_key, audio_bytes)

            # ── Phase 2: Ordered playback gate ────────────────────────────────
            # Wait for the previous chunk to finish streaming before emitting frames.
            # Gate for chunk 0 is set immediately in queue_tts.
            # Gate for chunk N>0 is set by chunk N-1's finally block.
            #
            # Wrapped in wait_for so a permanently stuck gate (zombie scenario)
            # does not hang the worker forever. On timeout: skip Phase 3 for this
            # chunk only — do not stream audio that could overlap the prior chunk.
            gate = self._playback_events.get(chunk_id)
            if gate is not None:
                gate_t0 = time.perf_counter()
                try:
                    await asyncio.wait_for(gate.wait(), timeout=_GATE_WAIT_TIMEOUT_S)
                except asyncio.TimeoutError:
                    logger.error(
                        "[TTS] chunk %d gate timed out after %.0fs — "
                        "skipping this chunk's playback (turn_id=%d task_turn_id=%d)",
                        chunk_id,
                        _GATE_WAIT_TIMEOUT_S,
                        self._turn_id,
                        task_turn_id,
                    )
                    return
                # CancelledError is intentionally NOT caught here — it propagates
                # to the outer handler so finally still runs (gate N+1 gets set).
                gate_elapsed = time.perf_counter() - gate_t0
                if gate_elapsed > _SLOW_GATE_WARN_S:
                    logger.warning(
                        "[TTS] chunk %d waited %.2fs for gate — "
                        "synthesis did not finish before playback caught up",
                        chunk_id, gate_elapsed,
                    )

            # Turn may have advanced during gate wait (barge-in); never stream stale audio.
            if self._turn_id != task_turn_id:
                return

            if self.cancel_event.is_set():
                return

            # ── Phase 3: Stream frames to Twilio ─────────────────────────────
            # handler._stream_tts_chunk checks cancel_event per 20ms frame so
            # barge-in takes effect within one frame period even mid-chunk.
            await self._handler._stream_tts_chunk(  # type: ignore[attr-defined]
                text,
                use_ssml=use_ssml,
                is_final=is_final,
                prefetched_bytes=audio_bytes,
            )

        except asyncio.CancelledError:
            pass  # Cancelled by cancel_current_and_clear_queue or shutdown
        except Exception as exc:
            logger.error("[TTS] chunk %d fatal: %s", chunk_id, exc, exc_info=True)
        finally:
            # ── Zombie guard ──────────────────────────────────────────────────
            # If _reset_turn_state was called while this task was stuck in
            # uncancellable sync IO (ElevenLabs), _turn_id will have advanced.
            # In that case, touching _playback_events or _synthesis_tasks would
            # corrupt the new turn's gate chain — skip all mutations.
            #
            # NOTE: intentionally using if/else rather than `return` here.
            # `return` in a finally block suppresses pending CancelledError,
            # which would make the task appear "done normally" to the event loop
            # and break gather()-based cancellation tracking.
            if self._turn_id != task_turn_id:
                logger.warning(
                    "[TTS] zombie chunk %d (task_turn=%d current_turn=%d) — "
                    "skipping pipeline mutations",
                    chunk_id, task_turn_id, self._turn_id,
                )
            else:
                # ── Unblock next chunk ────────────────────────────────────────
                # Set gate chunk_id+1 unconditionally — even on error or cancel.
                # This guarantees no downstream chunk ever hangs because an earlier
                # one errored, was cancelled, or produced no audio.
                next_gate = self._playback_events.get(chunk_id + 1)
                if next_gate is not None:
                    next_gate.set()
                # Remove the consumed gate to prevent unbounded dict growth.
                self._playback_events.pop(chunk_id, None)

                # Remove this task from the registry BEFORE checking emptiness.
                # (add_done_callback is intentionally NOT used — the callback fires
                # after the coroutine returns, i.e. AFTER this finally block, making
                # the empty-dict check below unreliable.)
                self._synthesis_tasks.pop(chunk_id, None)

                # ── Turn completion ───────────────────────────────────────────
                # Fire end_call when every task in this turn has finished AND a
                # final chunk was enqueued (_turn_has_final, set at queue_tts time).
                if not self._synthesis_tasks and self._turn_has_final:
                    turn_elapsed = (
                        time.perf_counter() - self._turn_start_ts
                        if self._turn_start_ts else 0.0
                    )
                    logger.info(
                        "[TTS] turn complete: %d chunk(s), %d cache hit(s), %.2fs total",
                        self._turn_chunk_count,
                        self._turn_cache_hits,
                        turn_elapsed,
                    )
                    await self._run_turn_completion_hooks()
