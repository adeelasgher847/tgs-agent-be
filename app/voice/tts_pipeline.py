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
from typing import Any, Dict

from app.core.logger import logger
from app.voice.humanization_engine import analyze_response

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

        # ── ElevenLabs `previous_text` continuity (Phase 4D-2 fix) ─────────
        # Tracks the text of the LAST chunk QUEUED (not the last chunk that
        # finished PLAYING). Captured synchronously inside queue_tts() before
        # the next chunk's task is spawned, so it can never race with
        # playback completion the way a post-playback-mutated attribute can.
        # This is now the single authoritative source for a queued chunk's
        # ElevenLabs `previous_text` — see queue_tts() and _reset_turn_state().
        self._last_queued_text: str | None = None

        # Backpressure: limits simultaneous TTS API calls.
        # Acquired inside _process_chunk before synthesis; released after.
        # queue_tts() always returns instantly — tasks queue up inside themselves.
        self._synthesis_semaphore: asyncio.BoundedSemaphore = asyncio.BoundedSemaphore(
            _MAX_CONCURRENT_SYNTHESIS
        )

        # ── Phase 6-3: ElevenLabs persistent WebSocket streaming session ───
        # Turn-scoped. Lazily created by the first chunk that routes to it
        # (see _try_elevenlabs_ws_route) — never created eagerly, never
        # reused across turns. Gated behind
        # settings.VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED +
        # TTSProviderCapabilities.supports_streaming_session; fully inert
        # (always None, never consulted) for Google/Rime/HTTP-mode ElevenLabs.
        self._eleven_ws_session: Any = None
        self._eleven_ws_owner_chunk_id: int | None = None
        # Set for the remainder of a turn once a WS-session open attempt has
        # failed, so every subsequent chunk in the SAME turn falls back to
        # the normal HTTP path too (turn-level fallback granularity — see
        # module docstring on _try_elevenlabs_ws_route for why).
        self._eleven_ws_failed_this_turn: bool = False
        # Dedicated gate chain — mirrors _playback_events' proven design —
        # that serializes ENTRY into _try_elevenlabs_ws_route in strict
        # chunk_id order. Necessary because chunks are independent
        # asyncio.Tasks (see queue_tts): without this, two chunks could both
        # observe `self._eleven_ws_session is None` concurrently and both
        # attempt to become the owner, or a later chunk's text could reach
        # the session before an earlier chunk's. Deliberately SEPARATE from
        # _playback_events: that chain only advances once a chunk's audio
        # has fully STREAMED (which, for the WS owner, can take the whole
        # turn) — reusing it here would serialize text-sending behind
        # audio-playback completion and defeat the entire point of
        # incremental WS synthesis.
        self._ws_send_gates: Dict[int, asyncio.Event] = {0: asyncio.Event()}

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
                # Phase 6-3: the true turn-final marker can legitimately carry
                # no text (LLM's trailing buffer was already fully flushed by
                # an earlier real chunk). If an ElevenLabs WS session is open
                # for this turn, it still needs its one-time flush signal here
                # — otherwise the owner chunk's iter_audio() would wait
                # indefinitely for the server's `isFinal` (bounded only by the
                # server's own inactivity_timeout as a backstop). Fire-and-
                # forget: queue_tts() must never block/await network I/O.
                if self._eleven_ws_session is not None:
                    asyncio.create_task(self._eleven_ws_session.finalize())
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

        # ── Capture ElevenLabs `previous_text` continuity synchronously ────
        # Must happen HERE — before this chunk's task is spawned — not via a
        # value mutated after some OTHER chunk finishes playing. This is what
        # prevents chunk N+1's prefetch (which can run concurrently with
        # chunk N still playing) from ever racing chunk N's post-playback
        # state update: there is no post-playback writer to race anymore.
        # `text` here is the raw, pre-humanization chunk text (humanization
        # mutates task["text"] later, inside the spawned task) — acceptable
        # since only minor wording changes are possible there.
        task["_previous_text"] = self._last_queued_text
        self._last_queued_text = text

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

        # Phase 6-3: same defensive/pre-create pattern as the playback gate
        # above, but for the SEPARATE _ws_send_gates chain (see __init__ for
        # why it must be independent of _playback_events).
        if chunk_id not in self._ws_send_gates:
            e = asyncio.Event()
            e.set()
            self._ws_send_gates[chunk_id] = e
        self._ws_send_gates[chunk_id + 1] = asyncio.Event()

        # First chunk of a turn: release gate 0 immediately so playback can
        # start as soon as synthesis finishes.  For all later chunks, the
        # gate is released by the previous task's finally block.
        if chunk_id == 0:
            self._playback_events[0].set()
            self._ws_send_gates[0].set()

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
        if hasattr(self._handler, "_is_tts_playing"):
            self._handler._is_tts_playing = False
        if hasattr(self._handler, "_twilio_buffer_primed"):
            self._handler._twilio_buffer_primed = False
        if hasattr(self._handler, "_prev_tts_tail"):
            self._handler._prev_tts_tail = b""

    def reset_previous_text_continuity(self) -> None:
        """
        Reset ElevenLabs `previous_text` continuity tracking for a fresh turn,
        WITHOUT touching gate-chain / _turn_id / is_speaking state.

        Callers must invoke this unconditionally at the start of EVERY new
        agent turn — not just on barge-in — mirroring the old (pre-Phase
        4D-2) unconditional `self._h._elevenlabs_prev_tts_text = ""` reset
        that used to run at the top of
        ConversationOrchestrator.generate_and_stream_response() for both
        transports.

        cancel_current_and_clear_queue() (-> _reset_turn_state()) already
        resets _last_queued_text too, but it ONLY runs on an actual
        barge-in/cancel path. On a normal (non-barge-in) turn — e.g. the
        common LiveKitBrowserCallHandler._process_transcript() path when the
        agent already finished speaking — nothing else clears
        _last_queued_text between turns, so without this dedicated reset the
        first chunk of a brand-new turn would incorrectly inherit the
        previous turn's last chunk as its `previous_text`. This method is
        the surgical fix: it clears only the one field relevant here,
        leaving gate/turn-id/is_speaking state (which must NOT be disturbed
        outside of an actual cancel) untouched.
        """
        self._last_queued_text = None
        # Phase 6-3: a fresh turn must NEVER inherit a prior turn's
        # ElevenLabs WS session. This mirrors the surgical, narrow-scope
        # reset pattern above — only the WS session reference/flag are
        # touched, gate-chain/_turn_id/is_speaking are untouched. Any
        # session object still referenced here is closed defensively
        # (best-effort, fire-and-forget) — normally already None/closed by
        # the owning chunk's own finally-block cleanup by the time a new
        # turn starts; this guards the rare edge case where a turn ended
        # without ever sending a final/flush chunk (see module docstring on
        # _try_elevenlabs_ws_route).
        self._defensive_close_ws_session()

    def _defensive_close_ws_session(self) -> None:
        """Best-effort, non-blocking close of any still-referenced ElevenLabs
        WS session, then clear turn-scoped WS state. Safe to call whether or
        not a session exists, and safe to call more than once (aclose() is
        idempotent).

        Uses getattr(..., None) rather than direct attribute access: some
        existing tests construct TtsPipeline via TtsPipeline.__new__(...) and
        hand-populate only the specific attributes they exercise, bypassing
        __init__ (see tests/voice/test_rime_tts_and_barge_in.py). Those
        pipelines never set the Phase 6-3 WS fields at all — direct access
        here would raise AttributeError for them even though the WS session
        feature is fully inert for their scenarios.
        """
        session = getattr(self, "_eleven_ws_session", None)
        self._eleven_ws_session = None
        self._eleven_ws_owner_chunk_id = None
        self._eleven_ws_failed_this_turn = False
        if session is not None:
            try:
                asyncio.create_task(session.aclose())
            except RuntimeError:
                # No running loop (defensive — should not happen on these
                # call sites, which only ever run inside the voice pipeline's
                # own event loop).
                pass

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
        except (
            Exception
        ):  # noqa: S110 - defensive; Event.set() during shutdown must not block teardown
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
        # New turn (or barge-in) must not inherit the previous turn's last
        # chunk text as `previous_text` for its first chunk.
        self._last_queued_text = None
        # Phase 6-3: barge-in must close/clear the ElevenLabs WS session too
        # — the next turn always gets a fresh session, and the cancelled
        # owner chunk's own finally-block cleanup (see _process_chunk) races
        # harmlessly with this: aclose() is idempotent, and whichever runs
        # first wins.
        self._defensive_close_ws_session()
        # Increment BEFORE replacing the gate dict so any zombie task that
        # woke up between the clear() above and the assignment below still
        # sees the old turn_id and bails out.
        self._turn_id += 1
        # Replace the entire gate dict with a single fresh unset gate for chunk 0.
        # Tasks waiting on old gate objects are either already cancelled (gather
        # above ran to completion) or orphaned zombies whose finally blocks will
        # be turn_id-guarded and skip gate mutations.
        self._playback_events = {0: asyncio.Event()}
        # Phase 6-3: reset the independent WS-send-ordering gate chain too.
        self._ws_send_gates = {0: asyncio.Event()}

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
            if transfer and hasattr(self._handler, "_transfer_after_agent_request"):
                try:
                    await self._handler._transfer_after_agent_request()
                except Exception as exc:
                    logger.warning("[TTS] transfer_after error: %s", exc)
            elif end_call and hasattr(self._handler, "_end_call_after_agent_request"):
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

    def _get_cached(self, key: str) -> bytes | None:
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
    # Phase 6-3: ElevenLabs WebSocket streaming-session routing
    # ------------------------------------------------------------------

    def _advance_ws_send_gate(self, chunk_id: int) -> None:
        """
        Release the WS-send-ordering gate for chunk_id + 1 and drop
        chunk_id's own consumed entry. MUST be called exactly once per
        chunk, promptly — i.e. right after this chunk's routing DECISION is
        made (owner/relayed/not-applicable), never deferred until Phase 3
        (audio streaming) completes. The WS owner's Phase 3 can legitimately
        block for the ENTIRE turn (awaiting more audio until finalize()), so
        deferring this release until then would deadlock every later chunk
        waiting on this same chain — see call sites for the two places this
        must fire from: the cache-hit branch (which never calls
        _try_elevenlabs_ws_route at all) and _try_elevenlabs_ws_route's own
        finally (every other chunk).
        """
        next_gate = self._ws_send_gates.get(chunk_id + 1)
        if next_gate is not None:
            next_gate.set()
        self._ws_send_gates.pop(chunk_id, None)

    async def _try_elevenlabs_ws_route(
        self,
        chunk_id: int,
        task: Dict[str, Any],
        text: str,
        is_final: bool,
        decision: Any,
    ) -> tuple[str | None, Any]:
        """
        Decide whether this chunk should be routed into a persistent
        ElevenLabs `stream-input` WebSocket session instead of the default
        one-HTTP-request-per-chunk path, and perform that routing if so.

        Returns (route, audio_iter_or_None):
          ("owner", async_iterator)  — this chunk opened/owns the turn's WS
              session. `async_iterator` yields the WHOLE turn's audio (not
              just this chunk's) — see the module docstring on
              app.services.elevenlabs_ws_session for why a 1:1 chunk-to-
              audio-segment mapping isn't available under `auto_mode`.
              Caller should treat `async_iterator` exactly like an
              HTTP-derived streaming iterator for Phase 2/3 below.
          ("relayed", None)          — this chunk's text was forwarded into
              an already-open session owned by an earlier chunk. Nothing
              else to do for THIS chunk.
          (None, None)               — not applicable: flag off, capability
              unavailable for the resolved provider, or the session failed
              to OPEN for this turn (safe to fall back to HTTP — no WS audio
              has played yet). A failure while relaying into an
              already-open session instead raises, so the caller can treat
              it as a chunk-level error (mirrors an HTTP synthesis failure;
              deliberately NOT a fallback, to avoid duplicate audio from a
              half-WS-half-HTTP turn — see Phase 6-3 task scope).
        """
        from app.core.config import settings as _app_settings

        if not _app_settings.VOICE_TTS_ELEVENLABS_STREAMING_SESSION_ENABLED:
            return None, None

        handler = self._handler
        agent = getattr(handler, "agent", None)
        if agent is None:
            return None, None

        try:
            from app.core.agent_runtime import resolve_tts_runtime

            tts_runtime = resolve_tts_runtime(agent, db=getattr(handler, "db", None))
        except Exception as exc:
            logger.debug(
                "[TTS] chunk %d: could not resolve tts runtime for WS routing: %s",
                chunk_id,
                exc,
            )
            return None, None

        from app.voice.tts_provider_capabilities import get_capabilities

        caps = get_capabilities(tts_runtime.adapter_slug)
        if not caps.supports_streaming_session:
            return None, None

        # ── Serialize entry in strict chunk_id order ───────────────────────
        # See __init__'s _ws_send_gates docstring for why this dedicated gate
        # chain (not _playback_events) is required: without it, concurrent
        # chunk tasks could race on `self._eleven_ws_session is None` and
        # either double-open a session or relay text out of order.
        #
        # CRITICAL: the release below (_advance_ws_send_gate) fires
        # immediately after the ROUTING DECISION — never deferred until
        # Phase 3 (audio streaming). The WS owner's Phase 3 can legitimately
        # block for the entire turn (awaiting more audio until finalize()),
        # so releasing this gate only after Phase 3 would deadlock every
        # later chunk waiting on this same chain.
        send_gate = self._ws_send_gates.get(chunk_id)
        if send_gate is not None:
            try:
                await asyncio.wait_for(send_gate.wait(), timeout=_GATE_WAIT_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.error(
                    "[TTS] chunk %d: ws send-gate timed out — falling back to HTTP",
                    chunk_id,
                )
                # Still advance the chain — a permanently-stuck gate must
                # never permanently block every later chunk in the turn.
                self._advance_ws_send_gate(chunk_id)
                return None, None

        try:
            return await self._route_elevenlabs_ws_locked(
                chunk_id, task, text, is_final, decision, tts_runtime
            )
        finally:
            self._advance_ws_send_gate(chunk_id)

    async def _route_elevenlabs_ws_locked(
        self,
        chunk_id: int,
        task: Dict[str, Any],
        text: str,
        is_final: bool,
        decision: Any,
        tts_runtime: Any,
    ) -> tuple[str | None, Any]:
        """
        The actual routing decision/action, always called with exclusive,
        in-chunk-order access (see _ws_send_gates in the caller). Split out
        of _try_elevenlabs_ws_route purely so the gate-release `finally`
        above stays simple and always fires exactly once.
        """
        # ── Non-owner: an active session already exists for this turn ─────
        if self._eleven_ws_session is not None:
            await self._eleven_ws_session.send_text(text)
            if is_final:
                await self._eleven_ws_session.finalize()
            return "relayed", None

        # Already failed to open earlier this turn: every subsequent chunk
        # falls back to HTTP too (turn-level fallback granularity).
        if self._eleven_ws_failed_this_turn:
            return None, None

        if not tts_runtime.voice_external_id:
            return None, None

        # ── This is the first chunk needing synthesis this turn: attempt to
        # become the session owner. Opened HERE (lazily), never eagerly. ───
        from app.services.elevenlabs_service import elevenlabs_service
        from app.services.elevenlabs_ws_session import (
            ElevenLabsWebSocketSession,
            ElevenLabsWebSocketSessionError,
        )
        from app.voice.tts_provider_capabilities import build_voice_settings_overlay

        provider_settings = dict(tts_runtime.settings_json or {})

        voice_settings: Dict[str, Any] = dict(
            elevenlabs_service._default_voice_settings()
        )
        for key in (
            "stability",
            "similarity_boost",
            "style",
            "use_speaker_boost",
            "speed",
        ):
            if key in provider_settings:
                voice_settings[key] = provider_settings[key]
        try:
            voice_settings.update(
                build_voice_settings_overlay(tts_runtime.adapter_slug, decision)
            )
        except Exception as exc:
            logger.debug(
                "[TTS] chunk %d: humanization overlay skipped for WS session: %s",
                chunk_id,
                exc,
            )

        api_key_override = None
        for key in ("elevenlabs_api_key", "xi_api_key"):
            val = provider_settings.get(key)
            if val:
                api_key_override = str(val).strip()
                break

        session = ElevenLabsWebSocketSession(
            voice_external_id=tts_runtime.voice_external_id,
            model_id=provider_settings.get("model", "eleven_flash_v2_5"),
            output_format=provider_settings.get("output_format", "ulaw_8000"),
            voice_settings=voice_settings,
            api_key_override=api_key_override,
        )
        try:
            await session.start(initial_text=text)
        except ElevenLabsWebSocketSessionError as exc:
            self._eleven_ws_failed_this_turn = True
            logger.warning(
                "[TTS] chunk %d: ElevenLabs WS session failed to open — "
                "falling back to HTTP for this turn: %s",
                chunk_id,
                exc,
            )
            return None, None

        self._eleven_ws_session = session
        self._eleven_ws_owner_chunk_id = chunk_id

        if is_final:
            try:
                await session.finalize()
            except Exception as exc:
                logger.warning(
                    "[TTS] chunk %d: ElevenLabs WS finalize failed: %s", chunk_id, exc
                )

        task["_ws_owner"] = True
        return "owner", session.iter_audio()

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

            # ── Humanization decision (shared point for both call transports) ──
            # Cheap, synchronous, in-memory only (see app.voice.humanization_engine
            # docstring) — never blocks or adds latency to synthesis below. The
            # decision is attached to `task` so handler._prefetch_tts_audio can
            # fold provider-safe hints (e.g. ElevenLabs stability) into the
            # settings it already builds, without TtsPipeline knowing about any
            # provider itself. `decision.text` (tone_adapter output) becomes the
            # single final text used for both synthesis and streaming below —
            # this is the ONE place tone adaptation runs for both transports, so
            # callers (bidirectional_stream.py, conversation_orchestrator.py)
            # must not also call tone_adapter() themselves. Failure here must
            # never prevent TTS — on any error the decision is absent and the
            # original `text` is used exactly as before this integration.
            decision = None
            try:
                _user_text = getattr(self._handler, "_current_turn_user_text", "") or ""
                _stt_confidence = (
                    getattr(self._handler, "_current_turn_stt_confidence", 0.0) or 0.0
                )
                decision = analyze_response(
                    text,
                    user_text=_user_text,
                    stt_confidence=_stt_confidence,
                    use_ssml=bool(use_ssml),
                    is_final=bool(is_final),
                )
                task["_humanization_decision"] = decision
                if decision and decision.text:
                    text = decision.text
                    task["text"] = text
            except Exception as exc:
                logger.debug(
                    "[TTS] humanization decision skipped for chunk %d: %s",
                    chunk_id,
                    exc,
                )
                task["_humanization_decision"] = None
                decision = None

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
                # Phase 6-3: a cache hit skips _try_elevenlabs_ws_route
                # entirely (no synthesis needed), so it must release the
                # WS-send gate itself, immediately — not deferred to this
                # chunk's own Phase 3 completion (see _advance_ws_send_gate's
                # docstring for why promptness matters here).
                self._advance_ws_send_gate(chunk_id)
            else:
                async with self._synthesis_semaphore:
                    if self.cancel_event.is_set():
                        return
                    logger.debug(
                        "[TTS] generation started chunk %d '%.25s'", chunk_id, text
                    )

                    # ── Phase 6-3: ElevenLabs WS streaming-session routing ──
                    # Gated behind flag + capability (see
                    # _try_elevenlabs_ws_route). "relayed" means this chunk's
                    # text was forwarded into an already-open session owned
                    # by an earlier chunk — nothing else to do for THIS
                    # chunk, so return immediately (finally still runs).
                    # "owner" means this chunk opened/owns the turn's session
                    # — audio_bytes becomes its audio async iterator and
                    # Phase 2/3 below run completely unchanged, exactly as
                    # they would for an HTTP-derived async iterator.
                    ws_route = None
                    try:
                        ws_route, ws_audio_iter = await self._try_elevenlabs_ws_route(
                            chunk_id, task, text, is_final, decision
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        # Mid-turn WS failure on a "relay" (non-owner) chunk:
                        # per Phase 6-3 scope this surfaces as a chunk-level
                        # error exactly like an HTTP synthesis failure would
                        # — no fallback, no risk of duplicate audio (no WS
                        # audio has necessarily even played yet for THIS
                        # chunk's text).
                        logger.warning(
                            "[TTS] chunk %d: ElevenLabs WS routing error: %s",
                            chunk_id,
                            exc,
                        )
                        return

                    if ws_route == "relayed":
                        return
                    if ws_route == "owner":
                        audio_bytes = ws_audio_iter
                    else:
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
                        chunk_id,
                        synth_elapsed,
                        _SLOW_SYNTHESIS_WARN_S,
                    )
                else:
                    logger.debug(
                        "[TTS] synthesis chunk %d done in %.2fs",
                        chunk_id,
                        synth_elapsed,
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
                        chunk_id,
                        gate_elapsed,
                    )

            # Turn may have advanced during gate wait (barge-in); never stream stale audio.
            if self._turn_id != task_turn_id:
                return

            if self.cancel_event.is_set():
                return

            # ── Phase 3: Stream frames to Twilio ─────────────────────────────
            # handler._stream_tts_chunk checks cancel_event per 20ms frame so
            # barge-in takes effect within one frame period even mid-chunk.
            # `pacing` (Phase 4C-2) is the SAME PacingHint already computed
            # above by analyze_response() — never recomputed here — passed
            # through so the handler can optionally append a small trailing
            # silence after eligible sentence-boundary chunks. `decision` may
            # be None if humanization failed/disabled above; that's a valid,
            # safe input (see humanization_engine.pause_frames_for_chunk).
            #
            # Phase 6-3: when this chunk OWNS the ElevenLabs WS session
            # (task["_ws_owner"]), `audio_bytes` is that session's
            # iter_audio() — a single continuous async iterator representing
            # the WHOLE turn's audio (auto_mode does not expose per-chunk
            # audio boundaries — see elevenlabs_ws_session module docstring),
            # not just this one chunk's text. `is_final` is therefore forced
            # True here regardless of this chunk's own task["is_final"], so
            # the transport's tail-drain/fade-out logic runs once the WHOLE
            # turn's audio is exhausted — exactly the semantics a single
            # is_final=True HTTP chunk would already have.
            effective_is_final = True if task.get("_ws_owner") else is_final
            await self._handler._stream_tts_chunk(  # type: ignore[attr-defined]
                text,
                use_ssml=use_ssml,
                is_final=effective_is_final,
                prefetched_bytes=audio_bytes,
                pacing=decision.pacing if decision is not None else None,
                previous_text=task.get("_previous_text"),
            )

            # Phase 6-3: distinguish "WS owner's iter_audio() ended because
            # the server sent isFinal (normal)" from "it was force-unblocked
            # because the session's write path failed mid-turn" — the two
            # look identical to _stream_tts_chunk (both just end the async
            # iterator), so without this check a write failure would look
            # like a perfectly normal, silent turn completion in the logs.
            if task.get("_ws_owner") and getattr(
                self._eleven_ws_session, "write_failed", False
            ):
                logger.error(
                    "[TTS] chunk %d: ElevenLabs WS session failed mid-turn "
                    "(write error) — turn's remaining audio was truncated",
                    chunk_id,
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
                    chunk_id,
                    task_turn_id,
                    self._turn_id,
                )
            else:
                # ── Phase 6-3: release the owned ElevenLabs WS session ──────────
                # Runs on every exit path (normal completion, error, or cancel)
                # since it's in `finally`. Fire-and-forget close (asyncio.create_
                # task) so a slow WS close handshake (Phase 6-2 measured
                # ~220-350ms) never delays gate advancement / turn-completion
                # detection below. Guarded by the turn_id check above — a zombie
                # owner task (barge-in already ran _reset_turn_state, which
                # closes the session itself) skips this entirely, avoiding a
                # double-close race; aclose() is idempotent regardless.
                if task.get("_ws_owner") and self._eleven_ws_session is not None:
                    _session_to_close = self._eleven_ws_session
                    self._eleven_ws_session = None
                    self._eleven_ws_owner_chunk_id = None
                    try:
                        asyncio.create_task(_session_to_close.aclose())
                    except RuntimeError:
                        pass

                # ── Unblock next chunk ────────────────────────────────────────
                # Set gate chunk_id+1 unconditionally — even on error or cancel.
                # This guarantees no downstream chunk ever hangs because an earlier
                # one errored, was cancelled, or produced no audio.
                next_gate = self._playback_events.get(chunk_id + 1)
                if next_gate is not None:
                    next_gate.set()
                # Remove the consumed gate to prevent unbounded dict growth.
                self._playback_events.pop(chunk_id, None)
                # NOTE: the separate Phase 6-3 _ws_send_gates chain is
                # deliberately NOT released here. It's released promptly
                # either inside _try_elevenlabs_ws_route's own finally (right
                # after the ROUTING DECISION, before Phase 3 audio streaming
                # even starts) or immediately in the cache-hit branch above —
                # never deferred to this point, since the WS owner's Phase 3
                # can legitimately block for the WHOLE turn, which would
                # deadlock every later chunk still waiting on this chain.

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
                        if self._turn_start_ts
                        else 0.0
                    )
                    logger.info(
                        "[TTS] turn complete: %d chunk(s), %d cache hit(s), %.2fs total",
                        self._turn_chunk_count,
                        self._turn_cache_hits,
                        turn_elapsed,
                    )
                    await self._run_turn_completion_hooks()
