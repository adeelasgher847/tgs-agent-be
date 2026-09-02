"""
TTS Streaming Mixin for BidirectionalStreamHandler.
Handles background audio, TTS chunk streaming, prefetch, and audio delivery to Twilio.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict

from app.core.agent_runtime import resolve_tts_runtime
from app.core.config import settings
from app.core.logger import logger
from app.services.bidirectional_stream_service import generate_mulaw_tts
from app.services.credit_service import credit_service
from app.services.google_tts_service import google_tts_service
from app.utils.audio_utils import (
    MULAW_FRAME_BYTES,
    apply_volume_fade,
    crossfade_mulaw_segments,
    stream_mulaw_bytes_over_twilio,
)
from app.utils.tts_adapter import get_tts_adapter
from app.utils.tts_preprocessing import detect_emotion
from app.utils.ssml_utils import smart_chunk_text
from app.utils.eleven_tts_text import prepare_tts_text_for_provider
from app.voice.humanization_engine import pause_frames_for_chunk
from app.voice.tts_provider_capabilities import build_voice_settings_overlay
from app.routers.general_websocket import broadcast_call_status_update

if TYPE_CHECKING:
    from app.voice.humanization_engine import PacingHint


# LiveKit recording-mirror agent-track keep-alive tuning.
#
# WebRTC AudioSource/RTP tracks (see LiveKitTwilioPublisher._source in
# livekit_twilio_bridge.py) are built around continuous, steady frame
# delivery to maintain the track's media clock. The agent's mirror track is
# only fed real frames during the narrow, bursty windows when TTS is
# actively streaming to Twilio (paced at 20ms by stream_mulaw_bytes_over_twilio)
# — between turns (silence, LLM thinking time, caller talking) it previously
# went silent for multi-second stretches. Confirmed via real S3 recording
# analysis (ffprobe/ffmpeg) that LiveKit's room-composite egress does not
# reliably keep such a gappy track mixed into the final recording, unlike the
# caller's mirror track which Twilio feeds continuously every ~20ms for the
# entire call. This keep-alive loop fills every gap with a silent μ-law frame
# so the agent's AudioSource never goes quiet for more than one tick.
_AGENT_MIRROR_KEEPALIVE_INTERVAL_S = 0.02  # 20ms, matches Twilio's own frame cadence
_AGENT_MIRROR_KEEPALIVE_GAP_THRESHOLD_S = 0.025  # 25ms — small margin over one tick
# Bounded queue depth for real frames awaiting the mirror writer (see
# _agent_mirror_keepalive_loop): ~5s of audio at one 20ms frame per slot.
# Generous enough to absorb any realistic synthesis burst/stall without
# growing unbounded; if ever exceeded, the OLDEST queued frame is dropped
# to make room — a lost recording frame is inconsequential, growing memory
# or blocking the enqueuer is not (see _mirror_enqueue).
_AGENT_MIRROR_QUEUE_MAXSIZE = 250


class TtsStreamMixin:
    """TTS streaming and audio delivery methods for BidirectionalStreamHandler."""

    def _livekit_recording_mirror(self):
        """Return agent TTS mirror callback when LiveKit egress recording is active.

        The returned callback is a fast, non-blocking ENQUEUE onto
        `self._agent_mirror_queue` — it never itself calls the publisher or
        awaits any network/pacing operation. The actual `pub.publish_mulaw()`
        call (which wraps LiveKit's rate-limited `AudioSource.capture_frame`)
        happens exclusively inside `_agent_mirror_keepalive_loop`, the single
        dedicated background writer for this call.

        This split exists because `capture_frame()` is a self-pacing API: it
        can legitimately take longer than one 20ms tick to maintain its own
        real-time delivery to the recording track. Previously this callback
        awaited `publish_mulaw()` directly, INLINE, before the caller-facing
        Twilio send in both `stream_mulaw_bytes_over_twilio` and
        `_stream_tts_chunk`'s `send_frame` — coupling the live call's audio
        pacing to the recording mirror's independent pacing clock. Confirmed
        via real-recording frame-level analysis: ~5.5% of frames showed a
        brief volume dip during continuous speech, consistent with this
        coupling occasionally delaying a live frame's send. Enqueueing here
        instead means the Twilio-facing call sites' `await mirror_mulaw(...)`
        now awaits a synchronous, in-memory queue operation (microseconds,
        no I/O) — it can never block, delay, or otherwise affect the audio
        sent to the caller, regardless of how the LiveKit publish is paced.
        """
        pub = getattr(self, "_lk_agent_publisher", None)
        if not (pub and getattr(pub, "connected", False)):
            return None

        async def _mirror_enqueue(mulaw_bytes: bytes) -> None:
            queue: asyncio.Queue | None = getattr(self, "_agent_mirror_queue", None)
            if queue is None:
                return
            try:
                queue.put_nowait(mulaw_bytes)
            except asyncio.QueueFull:
                # Drop the oldest queued frame to make room rather than
                # growing unbounded or blocking the caller — see
                # _AGENT_MIRROR_QUEUE_MAXSIZE.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    queue.put_nowait(mulaw_bytes)
                except asyncio.QueueFull:
                    pass

        return _mirror_enqueue

    async def _agent_mirror_keepalive_loop(self) -> None:
        """
        Background task: the SOLE writer to the agent recording-mirror
        publisher for this call — both real TTS frames (drained from
        `self._agent_mirror_queue`, enqueued by `_livekit_recording_mirror`'s
        callback) and silence keep-alive frames are published from here,
        never from the Twilio-facing send path directly. This guarantees:

        1. The caller-facing Twilio audio path never awaits anything that
           touches the LiveKit publisher — the recording mirror truly cannot
           block or delay live playback (see _livekit_recording_mirror's
           docstring for the frame-drop evidence that motivated this).
        2. Exactly one coroutine ever calls `pub.publish_mulaw()` — removing
           any possibility of this loop's own keep-alive silence frame
           racing/interleaving with a real frame pushed concurrently by a
           second producer (both now flow through this single loop, in
           order).

        Wakes up every ~20ms; if a real frame is queued, publishes the
        OLDEST one (FIFO — never drops queued content to "catch up", since
        `publish_mulaw`'s own pacing already absorbs momentary bursts via
        the queue acting as a buffer, exactly like `stream_mulaw_bytes_over_
        twilio`'s own pace_state does for the Twilio-facing send). Otherwise,
        publishes a silent frame UNLESS a real frame was published within
        the last `_AGENT_MIRROR_KEEPALIVE_GAP_THRESHOLD_S` — i.e. it is a
        no-op during active speech and only fills the gaps between turns.
        """
        silence_frame = bytes([0xFF]) * MULAW_FRAME_BYTES
        try:
            while True:
                await asyncio.sleep(_AGENT_MIRROR_KEEPALIVE_INTERVAL_S)

                pub = getattr(self, "_lk_agent_publisher", None)
                if not (pub and getattr(pub, "connected", False)):
                    # Publisher torn down / not yet connected — stay alive but idle
                    # so the task doesn't need to be recreated on reconnect races;
                    # it is explicitly cancelled at call teardown regardless.
                    continue

                queue: asyncio.Queue | None = getattr(self, "_agent_mirror_queue", None)
                real_frame: bytes | None = None
                if queue is not None and not queue.empty():
                    try:
                        real_frame = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        real_frame = None

                if real_frame is not None:
                    self._agent_mirror_last_real_frame_ts = time.monotonic()
                    try:
                        await pub.publish_mulaw(real_frame)
                    except Exception as exc:
                        logger.debug(
                            "[LiveKitBridge] agent mirror real frame publish failed: %s",
                            exc,
                        )
                    continue

                last_real = getattr(self, "_agent_mirror_last_real_frame_ts", 0.0)
                elapsed = time.monotonic() - last_real
                if elapsed < _AGENT_MIRROR_KEEPALIVE_GAP_THRESHOLD_S:
                    # A real frame was just pushed this tick window — don't double up.
                    continue

                try:
                    await pub.publish_mulaw(silence_frame)
                except Exception as exc:
                    logger.debug(
                        "[LiveKitBridge] agent mirror keep-alive frame failed: %s", exc
                    )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.debug("[LiveKitBridge] agent mirror keep-alive loop exited: %s", exc)

    def _start_agent_mirror_keepalive(self) -> None:
        """Start the keep-alive/writer task once the agent recording-mirror publisher is up."""
        existing = getattr(self, "_lk_agent_keepalive_task", None)
        if existing and not existing.done():
            return
        self._agent_mirror_last_real_frame_ts = 0.0
        self._agent_mirror_queue = asyncio.Queue(maxsize=_AGENT_MIRROR_QUEUE_MAXSIZE)
        self._lk_agent_keepalive_task = asyncio.create_task(
            self._agent_mirror_keepalive_loop()
        )

    async def _stop_agent_mirror_keepalive(self) -> None:
        """Cancel the keep-alive/writer task cleanly (call teardown / recording stop)."""
        task = getattr(self, "_lk_agent_keepalive_task", None)
        self._lk_agent_keepalive_task = None
        self._agent_mirror_queue = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug(
                    "[LiveKitBridge] agent mirror keep-alive task cleanup: %s", exc
                )

    async def _send_twilio_clear_event(self) -> None:
        """
        Tell Twilio to discard any audio already buffered/sent for this stream.

        Required by the Media Streams protocol on barge-in: cancelling our own
        in-flight TTS task only stops *us* from sending more frames — it does
        not stop Twilio from continuing to play frames it already received.
        See: https://www.twilio.com/docs/voice/media-streams/websocket-messages
        """
        stream_sid = getattr(self, "stream_sid", None)
        if not stream_sid:
            return
        try:
            await self.websocket.send_json(
                {
                    "event": "clear",
                    "streamSid": stream_sid,
                }
            )
        except RuntimeError:
            # WebSocket already closed (hangup) — nothing to clear.
            pass
        except Exception as e:
            logger.warning("Failed to send Twilio 'clear' event: %s", e)

    async def _start_background_audio_with_delay(self):
        """Start background loop after call stabilizes (dev-branch behavior)."""
        try:
            if not self._is_background_audio_enabled():
                return
            self._background_audio.set_user_level(self._resolve_background_volume())
            await self._background_audio.start_loop_if_enabled(delay_seconds=3.0)
        except Exception as e:
            logger.error(
                "Error in _start_background_audio_with_delay: %s", e, exc_info=True
            )

    def _is_background_audio_enabled(self) -> bool:
        """
        Enable ambient background only when:
        - agent TTS provider is ElevenLabs
        - tts_settings_json.background_enabled is explicitly true (opt-in)
        - tts_settings_json.background_profile is "office" (or omitted)
        """
        if not self.agent:
            return False
        tts_provider_slug = resolve_tts_runtime(
            self.agent, db=getattr(self, "db", None)
        ).adapter_slug
        if tts_provider_slug != "elevenlabs":
            return False

        settings_json = dict(getattr(self.agent, "tts_settings_json", None) or {})
        enabled_raw = settings_json.get("background_enabled", False)
        if isinstance(enabled_raw, str):
            enabled = enabled_raw.strip().lower() in {"true", "1", "on", "yes"}
        else:
            enabled = bool(enabled_raw)
        if not enabled:
            return False

        profile = (
            str(settings_json.get("background_profile") or "office").strip().lower()
        )
        return profile == "office"

    def _resolve_background_volume(self) -> float:
        """
        Resolve ambient volume from tts_settings_json.background_volume.
        Input range is 0..100 from UI slider; default is 50.
        Returns normalized linear gain in 0.0..1.0.
        """
        if not self.agent:
            return 0.5
        settings_json = dict(getattr(self.agent, "tts_settings_json", None) or {})
        raw = settings_json.get("background_volume", 50)
        try:
            pct = float(raw)
        except (TypeError, ValueError):
            pct = 50.0
        pct = max(0.0, min(100.0, pct))
        return pct / 100.0

    def _apply_soft_duck(self, gain: float) -> float:
        """
        Scale `gain` down while a soft-duck window is active (see
        BidirectionalStreamHandler._on_stt_speech_started, set from
        Deepgram's vad_events/SpeechStarted -- pure VAD onset, ahead of the
        real confidence-gated barge-in decision). Twilio-only: the LiveKit
        browser handler has no equivalent gain pipeline to hook into
        (see livekit_browser_call_handler.py), so this is a no-op there via
        the plain `getattr` default below.
        """
        duck_until = getattr(self, "_soft_duck_until_mono", 0.0)
        if duck_until and time.monotonic() < duck_until:
            duck_gain = getattr(self, "_soft_duck_gain", 0.35)
            return gain * duck_gain
        return gain

    def _base_voice_gain_from_runtime(self, runtime) -> float:
        """
        Compute the (non-duck) base voice gain from an ALREADY-RESOLVED
        ``ResolvedTtsRuntime``. Pure attribute reads -- cheap, no DB access.

        Callers that have already called ``resolve_tts_runtime()`` earlier
        in the same code path (e.g. `_stream_tts_chunk` resolves it once up
        front to pick the provider) should reuse that result via this
        method instead of calling `_resolve_voice_volume_base()`, which
        would resolve the runtime a second time (and, for a BYO ElevenLabs
        agent, decrypt the stored key a second time -- an extra Postgres
        round-trip for no reason).

        - Google TTS outputs at nominal telephony level (~-18 dBFS RMS) -> baseline 1.0x.
        - ElevenLabs ulaw_8000 outputs at ~-25.7 to -26.3 dBFS RMS with -6.2 dBFS peak headroom.
          Applying a 1.8x baseline gain (+5.1 dB) brings ElevenLabs to ~-20.6 dBFS RMS,
          matching PSTN speech levels while maintaining safe peak headroom (-1.7 dBFS).
        - User-configured volume slider scales on top of the provider baseline.
        """
        try:
            user_vol = float(runtime.settings_json.get("volume", 1.0))
            provider_slug = (runtime.adapter_slug or "").lower()
            baseline = 1.8 if provider_slug == "elevenlabs" else 1.0
            return max(0.0, user_vol * baseline)
        except Exception:
            return 1.0

    def _resolve_voice_volume_base(self) -> float:
        """
        Resolve TTS voice volume (linear gain) from agent settings with
        provider-aware telephony baseline leveling -- WITHOUT the soft-duck
        adjustment (see _resolve_voice_volume() for that).

        Deliberately NOT cheap enough to call at frame (20ms) granularity:
        ``resolve_tts_runtime()`` can decrypt a BYO ElevenLabs key, which
        hits Postgres (pgp_sym_decrypt). Callers that need duck-reactive
        gain inside a frame-streaming loop should call this once per
        chunk/utterance (or reuse an already-resolved runtime via
        `_base_voice_gain_from_runtime()` if one is already in scope) and
        re-apply only `_apply_soft_duck()` (cheap: attribute reads + a
        monotonic-clock comparison) on every frame.
        """
        if not self.agent:
            return 1.0
        try:
            runtime = resolve_tts_runtime(self.agent, db=getattr(self, "db", None))
            return self._base_voice_gain_from_runtime(runtime)
        except Exception:
            return 1.0

    def _resolve_voice_volume(self) -> float:
        """
        Resolve TTS voice volume (linear gain) from agent settings with
        provider-aware telephony baseline leveling, then apply the current
        soft-duck window (see _apply_soft_duck()) on top.

        NOTE: this recomputes the (non-cheap, see _resolve_voice_volume_base)
        base gain every call. Frame-streaming loops should NOT call this
        per-frame -- instead resolve the base gain once per chunk and call
        `self._apply_soft_duck(base_gain)` per frame so the duck window
        reacts immediately mid-chunk without repeating the expensive
        resolution (and possible BYO-key DB decrypt) at 50 Hz.
        """
        return self._apply_soft_duck(self._resolve_voice_volume_base())

    async def _stream_tts_chunk(
        self,
        text: str,
        use_ssml: bool = False,
        is_final: bool = False,
        prefetched_bytes: Any = None,
        pacing: "PacingHint | None" = None,
        previous_text: str | None = None,
        humanization_decision: Any = None,
    ):
        """
        Generate and stream a single TTS chunk (used by parallel pipeline worker).
        Simplified version without the complex prefix/suffix splitting.

        `pacing` (Phase 4C-2, optional): the HumanizationDecision.pacing hint
        already computed once by TtsPipeline._process_chunk — never
        recomputed here. When VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES > 0 and
        this is an eligible non-final, real-sentence-boundary chunk, a small
        trailing silence is appended after this chunk's real audio (see
        app.voice.humanization_engine.pause_frames_for_chunk). Defaults to
        None and is fully inert when omitted or when the config is 0.
        Note: Does NOT clear cancel flag - respects barge-in for entire queue.

        `previous_text` (Phase 4D-2, optional): the previously QUEUED chunk's
        text, captured synchronously by TtsPipeline.queue_tts() — used to set
        ElevenLabs' `previous_text` continuity field ONLY in the rare
        fallback path below where `_prefetch_tts_audio` did not already
        return an async iterator (e.g. it errored/returned None). The common
        path never reaches this: `_prefetch_tts_audio` reads the same value
        directly from the task dict.

        `humanization_decision` (optional): the SAME HumanizationDecision
        already computed once by TtsPipeline._process_chunk (via
        analyze_response()), passed through so the humanization voice-
        settings overlay (e.g. ElevenLabs stability) can be applied in the
        same rare fallback path noted above for `previous_text` -- when
        `_prefetch_tts_audio` did not already hand back synthesized audio,
        this function performs its own live synthesis and must apply the
        identical overlay `_prefetch_tts_audio` would have, or that turn's
        tone/stability silently diverges from every other chunk. Never
        recomputed here — passing None (humanization disabled/failed
        upstream) is a valid, safe input (build_voice_settings_overlay
        already tolerates it).

        Args:
            text: Text or SSML to convert to speech
            use_ssml: Whether text contains SSML markup
        """
        try:
            if not text or not text.strip():
                return

            # If stream isn't ready yet (race at call start), wait once on an event
            # instead of polling so we avoid up-to-1s spin latency.
            if not self.stream_sid:
                if self._tts_cancel.is_set():
                    return
                try:
                    await asyncio.wait_for(self._stream_sid_ready.wait(), timeout=0.35)
                except asyncio.TimeoutError:
                    return
                if not self.stream_sid:
                    return

            # Check if already cancelled before acquiring lock
            if self._tts_cancel.is_set():
                return

            async with self._tts_lock:
                self.is_speaking = True
                clean = text.strip()
                self._current_speaking_agent_text = (
                    getattr(self, "_current_speaking_agent_text", "") + " " + clean
                ).strip()
                try:
                    lang = (
                        self.agent.language
                        if self.agent and self.agent.language
                        else "en"
                    )
                    voice = (
                        self.agent.voice_type
                        if self.agent and self.agent.voice_type
                        else "female"
                    )
                    tts_runtime = resolve_tts_runtime(
                        self.agent, db=getattr(self, "db", None)
                    )
                    tts_provider_slug = tts_runtime.adapter_slug

                    # If audio was pre-generated by the parallel prefetch pipeline, skip the
                    # TTS API call entirely and fall through to the batch playback path.
                    _use_prefetched = (
                        prefetched_bytes is not None and not self._tts_cancel.is_set()
                    )
                    _is_prefetched_iter = _use_prefetched and hasattr(
                        prefetched_bytes, "__aiter__"
                    )

                    # Prefer true streaming TTS for longer responses (real-time playback).
                    # Keep cache-friendly path for very short phrases (e.g. quick ack).
                    word_count = len(clean.split())
                    stream_min_words = max(
                        1, int(getattr(settings, "VOICE_TTS_STREAM_MIN_WORDS", 2) or 2)
                    )
                    use_streaming_tts = (
                        word_count >= stream_min_words and not _use_prefetched
                    ) or _is_prefetched_iter
                    if use_streaming_tts and not self._tts_cancel.is_set():
                        try:
                            import base64
                            import time

                            # MULAW_FRAME_BYTES is imported at module level (see top of
                            # file) — deliberately NOT re-imported locally here. A local
                            # import of a name anywhere in a function makes that name
                            # local for the function's ENTIRE body (Python scoping), so
                            # importing it only inside this streaming branch previously
                            # would have made the batch/fallback path below (which also
                            # references MULAW_FRAME_BYTES for Phase 4C-2 pacing) raise
                            # UnboundLocalError whenever this streaming branch never ran.
                            from app.utils.audio_utils import (
                                apply_micro_fade_in,
                                apply_micro_fade_out,
                            )

                            # LiveKit recording mirror: this incremental
                            # (async_stream_synthesize) streaming branch sends
                            # frames directly to Twilio via send_frame below,
                            # unlike the bulk-buffer path (generate_mulaw_tts +
                            # stream_mulaw_bytes_over_twilio, further down in
                            # this same function) which already mirrors every
                            # frame it sends. This branch is the DOMINANT path
                            # for any response of 2+ words (VOICE_TTS_STREAM_
                            # MIN_WORDS), so without this it silently never
                            # mirrors the majority of real agent speech into
                            # the LiveKit recording egress -- confirmed via a
                            # real S3 recording where several agent turns were
                            # completely silent (ffprobe/volumedetect showed
                            # near-noise-floor levels, -40 to -70dB, at exactly
                            # those turns' timestamps) while short/simple
                            # responses (which take the bulk path) had normal
                            # speech levels.
                            _recording_mirror = self._livekit_recording_mirror()

                            async def send_frame(
                                frame: bytes, pace: bool = True, state: dict = None
                            ):
                                if not frame:
                                    return
                                if self._tts_cancel.is_set() or not self.stream_sid:
                                    return
                                if self._is_background_audio_enabled():
                                    frame = self._background_audio.mix_tts_frame(frame)
                                if _recording_mirror:
                                    try:
                                        await _recording_mirror(frame)
                                    except Exception:  # noqa: S110 - best-effort recording mirror; must not disrupt playback
                                        pass
                                payload = base64.b64encode(frame).decode("utf-8")
                                try:
                                    await self.websocket.send_json(
                                        {
                                            "event": "media",
                                            "streamSid": self.stream_sid,
                                            "media": {"payload": payload},
                                        }
                                    )
                                except RuntimeError:
                                    # WebSocket already closed (hangup). Stop sending immediately.
                                    self._tts_cancel.set()
                                    return
                                # Mark audio as actively playing on the first real frame sent.
                                if not getattr(self, "_is_tts_playing", False):
                                    self._is_tts_playing = True
                                    _first_audio_ts = time.perf_counter()
                                    self._metric_first_audio_ts = _first_audio_ts
                                    # Record for barge-in dead zone (see _maybe_process_interim)
                                    self._tts_play_start_ts = _first_audio_ts
                                    _first_token_ts = getattr(
                                        self, "_metric_first_token_ts", 0.0
                                    )
                                    if _first_token_ts > 0:
                                        _ttfa_ms = (
                                            _first_audio_ts - _first_token_ts
                                        ) * 1000
                                        logger.info(
                                            "[Metrics] llm_first_token_to_first_audio_chunk=%.0f ms",
                                            _ttfa_ms,
                                        )
                                    # End-to-end latency: caller speech → first agent audio out.
                                    # Parsed by scripts/latency_p95.py to compute staging p95.
                                    _stt_final_ts = getattr(
                                        self, "_metric_stt_final_ts", 0.0
                                    )
                                    if _stt_final_ts > 0:
                                        _e2e_ms = (
                                            _first_audio_ts - _stt_final_ts
                                        ) * 1000
                                        logger.info(
                                            "[Metrics] stt_final_to_first_audio=%.0f ms",
                                            _e2e_ms,
                                        )
                                if not pace:
                                    return
                                # Pacing with drift correction (shared state)
                                if state is None:
                                    return
                                if state["first"]:
                                    state["first"] = False
                                    state["next_send"] = (
                                        time.perf_counter() + state["send_interval"]
                                    )
                                    return
                                state["next_send"] += state["send_interval"]
                                now = time.perf_counter()
                                sleep_dur = state["next_send"] - now
                                if sleep_dur > 0:
                                    await asyncio.sleep(sleep_dur)
                                elif sleep_dur < -0.03:
                                    state["next_send"] = time.perf_counter()

                            # Resolve the (non-duck) base gain once per utterance, from
                            # the `tts_runtime` already resolved above (avoids a second
                            # resolve_tts_runtime() call / a second BYO-ElevenLabs-key
                            # Postgres decrypt). The duck-reactive part is re-applied
                            # fresh below, inside the `async for chunk_bytes in
                            # audio_iter` loop, so a SpeechStarted event landing
                            # mid-chunk immediately affects whichever provider chunks
                            # are still to be sent for this utterance -- see
                            # _apply_soft_duck() / _resolve_voice_volume().
                            base_voice_gain = self._base_voice_gain_from_runtime(
                                tts_runtime
                            )

                            async def stream_mulaw_from_audio_iter(audio_iter):
                                """
                                Consume an async iterator of MULAW bytes and stream as 20ms frames.
                                Uses:
                                - Optional jitter-buffer priming (first speak only)
                                - Single crossfade bridge at chunk boundary (prev tail + next head)
                                - Tail holdback (20ms) between chunks to avoid clicks/distortion
                                - User-configurable voice gain applied per provider chunk (not on
                                  priming / silence-drain frames, which stay at 0xFF). The
                                  soft-duck portion of this gain is re-resolved on every
                                  iteration of the loop below (cheap: monotonic-clock check),
                                  so a barge-in SpeechStarted event takes effect on the next
                                  provider chunk still to arrive, not just on the next
                                  full TTS-flush segment.
                                """
                                if self._is_background_audio_enabled():
                                    self._background_audio.set_user_level(
                                        self._resolve_background_volume()
                                    )

                                pace_state = {
                                    "send_interval": 0.02,
                                    "first": True,
                                    "next_send": time.perf_counter(),
                                }

                                # Prime Twilio jitter buffer once per utterance (2 frames = 40ms, paced so
                                # they arrive at proper 20ms intervals and actually fill the buffer).
                                if not self._twilio_buffer_primed:
                                    silent = bytes([0xFF]) * MULAW_FRAME_BYTES
                                    prime_frames = max(
                                        0,
                                        int(
                                            getattr(
                                                settings, "VOICE_TTS_PRIME_FRAMES", 1
                                            )
                                            or 1
                                        ),
                                    )
                                    for _ in range(prime_frames):
                                        if self._tts_cancel.is_set():
                                            return
                                        await send_frame(
                                            silent, pace=True, state=pace_state
                                        )

                                # Frame buffers
                                byte_buf = bytearray()
                                pending_frames = []

                                # No longer using crossfade bridge as it causes robotic stutter.
                                # Whether we've applied fade-in for this utterance
                                fade_needed = not self._twilio_buffer_primed

                                async for chunk_bytes in audio_iter:
                                    if self._tts_cancel.is_set():
                                        return
                                    if not chunk_bytes:
                                        continue
                                    # Re-resolve the duck-reactive gain on every provider
                                    # chunk (cheap -- see _apply_soft_duck()) instead of
                                    # reusing a value captured once before streaming
                                    # started, so mid-utterance SpeechStarted events are
                                    # audible immediately rather than only on the next
                                    # TTS-flush segment.
                                    voice_gain = self._apply_soft_duck(base_voice_gain)
                                    if voice_gain != 1.0:
                                        chunk_bytes = apply_volume_fade(
                                            chunk_bytes, voice_gain
                                        )
                                    byte_buf.extend(chunk_bytes)

                                    # Convert bytes to 20ms frames
                                    while len(byte_buf) >= MULAW_FRAME_BYTES:
                                        frame = bytes(byte_buf[:MULAW_FRAME_BYTES])
                                        del byte_buf[:MULAW_FRAME_BYTES]
                                        pending_frames.append(frame)

                                        # Send oldest frame
                                        out = pending_frames.pop(0)
                                        if fade_needed and out:
                                            out = apply_micro_fade_in(
                                                out, duration_ms=25.0
                                            )
                                            fade_needed = False
                                        await send_frame(
                                            out, pace=True, state=pace_state
                                        )

                                # End of streaming responses: handle remainder
                                if self._tts_cancel.is_set():
                                    return

                                if is_final:
                                    # Flush any partial remainder (pad with silence so we
                                    # always send aligned 20ms (160-byte) frames to Twilio).
                                    if byte_buf:
                                        pad = MULAW_FRAME_BYTES - (
                                            len(byte_buf) % MULAW_FRAME_BYTES
                                        )
                                        if pad != MULAW_FRAME_BYTES:
                                            byte_buf.extend(b"\xff" * pad)
                                        while len(byte_buf) >= MULAW_FRAME_BYTES:
                                            pending_frames.append(
                                                bytes(byte_buf[:MULAW_FRAME_BYTES])
                                            )
                                            del byte_buf[:MULAW_FRAME_BYTES]

                                    # Send all remaining frames. The very last audio frame
                                    # gets a 25 ms linear fade-out to remove the abrupt
                                    # cut/click that callers otherwise hear at the end of
                                    # an utterance (especially over MULAW @ 8 kHz).
                                    if pending_frames:
                                        last_idx = len(pending_frames) - 1
                                        for idx, out in enumerate(pending_frames):
                                            if self._tts_cancel.is_set():
                                                break
                                            if fade_needed and out:
                                                out = apply_micro_fade_in(
                                                    out, duration_ms=25.0
                                                )
                                                fade_needed = False
                                            if idx == last_idx and out:
                                                out = apply_micro_fade_out(
                                                    out, duration_ms=25.0
                                                )
                                            await send_frame(
                                                out, pace=True, state=pace_state
                                            )
                                        pending_frames.clear()

                                    # Drain Twilio's playout jitter buffer with a short
                                    # MULAW silence tail (3×20ms = 60ms). Without this,
                                    # the last 40–80 ms of speech are sometimes clipped
                                    # because the WebSocket / RTP path closes before the
                                    # final media frame finishes playing.
                                    if not self._tts_cancel.is_set():
                                        silence_drain = (
                                            bytes([0xFF]) * MULAW_FRAME_BYTES
                                        )
                                        for _ in range(3):
                                            if self._tts_cancel.is_set():
                                                break
                                            await send_frame(
                                                silence_drain,
                                                pace=True,
                                                state=pace_state,
                                            )

                                    self._prev_tts_tail = b""
                                else:
                                    # Non-final chunk: send all remaining frames (no tail holdback).
                                    # Holding back 1 frame for a crossfade bridge sounds good in theory,
                                    # but between chunks there is always a TTS API generation gap
                                    # (200–500 ms) during which Twilio's buffer drains to zero.
                                    # Crossfading a stale 20 ms tail with fresh audio after that gap
                                    # creates an audible click/stutter that is worse than a clean cut.
                                    if byte_buf:
                                        pad = MULAW_FRAME_BYTES - (
                                            len(byte_buf) % MULAW_FRAME_BYTES
                                        )
                                        if pad != MULAW_FRAME_BYTES:
                                            byte_buf.extend(b"\xff" * pad)
                                        while len(byte_buf) >= MULAW_FRAME_BYTES:
                                            pending_frames.append(
                                                bytes(byte_buf[:MULAW_FRAME_BYTES])
                                            )
                                            del byte_buf[:MULAW_FRAME_BYTES]

                                    for out in pending_frames:
                                        if fade_needed and out:
                                            out = apply_micro_fade_in(
                                                out, duration_ms=25.0
                                            )
                                            fade_needed = False
                                        await send_frame(
                                            out, pace=True, state=pace_state
                                        )
                                    self._prev_tts_tail = b""

                                    # Phase 4C-2: optional small trailing silence after a
                                    # non-final chunk that ends at a real sentence boundary,
                                    # so multi-sentence responses get a brief human-like
                                    # breath. Fully inert unless
                                    # VOICE_TTS_INTERSENTENCE_PAUSE_FRAMES > 0 and the chunk
                                    # is eligible (see pause_frames_for_chunk). Reuses the
                                    # SAME 0xFF silence-frame bytes and paced send_frame()
                                    # mechanism as jitter-buffer priming/end-of-turn drain
                                    # above — every frame still goes through send_frame()'s
                                    # existing _tts_cancel check.
                                    for _ in range(
                                        pause_frames_for_chunk(pacing, is_final)
                                    ):
                                        if self._tts_cancel.is_set():
                                            break
                                        await send_frame(
                                            bytes([0xFF]) * MULAW_FRAME_BYTES,
                                            pace=True,
                                            state=pace_state,
                                        )

                                self._twilio_buffer_primed = True

                            # Stream text in near real-time from provider.
                            # For Google: use native async streaming API.
                            streaming_text = prepare_tts_text_for_provider(
                                clean, tts_provider_slug
                            )
                            if not streaming_text or not streaming_text.strip():
                                return
                            if _is_prefetched_iter:
                                audio_iter = prefetched_bytes
                            elif tts_provider_slug and tts_provider_slug not in (
                                "google",
                                "",
                            ):
                                external_voice_id = tts_runtime.voice_external_id
                                if not external_voice_id:
                                    tts_voice = (
                                        getattr(self.agent, "tts_voice", None)
                                        if self.agent
                                        else None
                                    )
                                    external_voice_id = getattr(
                                        tts_voice, "external_voice_id", None
                                    )
                                if (
                                    not external_voice_id
                                    and tts_provider_slug == "rime"
                                ):
                                    external_voice_id = "mistv2_Wildflower"
                                elif (
                                    not external_voice_id
                                    and tts_provider_slug == "hume"
                                ):
                                    from app.services.hume_tts_service import (
                                        HUME_DEFAULT_VOICE,
                                    )

                                    external_voice_id = HUME_DEFAULT_VOICE
                                elif (
                                    not external_voice_id and tts_provider_slug == "xai"
                                ):
                                    from app.services.xai_tts_service import (
                                        XAI_DEFAULT_VOICE,
                                    )

                                    external_voice_id = XAI_DEFAULT_VOICE
                                if not external_voice_id:
                                    raise ValueError(
                                        "TTS voice is not configured for streaming."
                                    )
                                adapter = get_tts_adapter(tts_provider_slug)
                                provider_settings = dict(tts_runtime.settings_json)
                                if tts_provider_slug == "elevenlabs":
                                    provider_settings.setdefault(
                                        "output_format", "ulaw_8000"
                                    )
                                    _prev_text = (previous_text or "").strip()
                                    if _prev_text:
                                        provider_settings["previous_text"] = _prev_text[
                                            -500:
                                        ]
                                elif tts_provider_slug == "rime":
                                    # Rime uses async_stream_synthesize — no output_format key needed
                                    # (mulaw 8 kHz is the default in RimeTTSAdapter).
                                    pass
                                else:
                                    # xai/hume/google ignore this key — codec/sample_rate are fixed
                                    # (mulaw/8kHz) in their own adapters, not read from provider_settings.
                                    provider_settings.setdefault(
                                        "output_format", "ulaw_8000"
                                    )

                                # Fold in any humanization overlay (e.g. ElevenLabs
                                # stability hint), exactly like _prefetch_tts_audio
                                # does -- this is the rare fallback path where THIS
                                # function performs live synthesis itself rather
                                # than reusing _prefetch_tts_audio's already-
                                # synthesized bytes, so without this the chunk's
                                # tone/stability would silently diverge from every
                                # other chunk in the turn (root cause of a real
                                # production report: agent tone reading
                                # inconsistent/unnatural on some responses).
                                # Isolated try/except: a humanization failure must
                                # never prevent this chunk's TTS request from going
                                # out with normal settings.
                                try:
                                    provider_settings.update(
                                        build_voice_settings_overlay(
                                            tts_provider_slug, humanization_decision
                                        )
                                    )
                                except Exception as exc:
                                    logger.debug(
                                        "[TTS] humanization overlay skipped (streaming fallback): %s",
                                        exc,
                                    )

                                # Prefer async streaming for providers that support it (Rime, ElevenLabs).
                                if hasattr(adapter, "async_stream_synthesize"):
                                    _cancel_ref = self._tts_cancel

                                    async def _async_stream_adapter(
                                        _adapter=adapter,
                                        _text=streaming_text,
                                        _vid=external_voice_id,
                                        _cfg=provider_settings,
                                        _cancel=_cancel_ref,
                                    ):
                                        async for (
                                            chunk
                                        ) in _adapter.async_stream_synthesize(
                                            text=_text,
                                            voice_external_id=_vid,
                                            settings_json=_cfg,
                                        ):
                                            if _cancel.is_set():
                                                break
                                            if chunk:
                                                yield chunk

                                    audio_iter = _async_stream_adapter()
                                else:
                                    sync_iter = adapter.stream_synthesize(
                                        text=streaming_text,
                                        voice_external_id=external_voice_id,
                                        settings_json=provider_settings,
                                    )

                                    async def _async_iter_from_sync(sync_source):
                                        iterator = iter(sync_source)
                                        sentinel = object()
                                        while True:
                                            chunk = await asyncio.to_thread(
                                                next, iterator, sentinel
                                            )
                                            if chunk is sentinel:
                                                break
                                            yield chunk

                                    audio_iter = _async_iter_from_sync(sync_iter)
                            else:
                                # Reduce robotic feel (streaming-safe): tiny emotion-based speaking rate adjustments
                                # Keep this subtle to avoid uncanny/unstable cadence.
                                emo = detect_emotion(streaming_text)
                                speaking_rate = 1.0
                                if emo == "happy":
                                    speaking_rate = 1.03
                                elif emo == "sad":
                                    speaking_rate = 0.97
                                elif emo == "uncertain":
                                    speaking_rate = 0.98
                                elif emo == "confident":
                                    speaking_rate = 1.01

                                tts_voice = (
                                    getattr(self.agent, "tts_voice", None)
                                    if self.agent
                                    else None
                                )
                                google_voice_name = getattr(
                                    tts_voice, "external_voice_id", None
                                )
                                audio_iter = google_tts_service.stream_text_to_speech(
                                    text=streaming_text,
                                    language=lang,
                                    voice_type=voice,
                                    speaking_rate=speaking_rate,
                                    output_format="mulaw",
                                    use_chirp3_hd=True,
                                    sample_rate_hz=8000,
                                    voice_name_override=google_voice_name,
                                )

                            await stream_mulaw_from_audio_iter(audio_iter)
                            # NOTE (Phase 4D-2): previously wrote streaming_text into
                            # self._elevenlabs_prev_tts_text here, post-playback, as the
                            # "previous_text" source for the NEXT chunk. That write raced
                            # the next chunk's prefetch (which typically starts while THIS
                            # chunk is still playing) and is no longer read by anything —
                            # TtsPipeline.queue_tts() now captures the next chunk's
                            # previous_text synchronously at queue time instead. See
                            # app.voice.tts_pipeline.TtsPipeline._last_queued_text.
                            return  # streaming path complete
                        except Exception as e:
                            logger.warning(
                                "⚠️ Streaming TTS failed, falling back to non-streaming: %s", e
                            )

                            # If call ended / barge-in occurred, never fall back to batch TTS.
                            if self._tts_cancel.is_set() or not self.stream_sid:
                                self._prev_tts_tail = b""
                                return

                            # `prefetched_bytes` was an unconsumed/partially-
                            # consumed async iterator (e.g. Hume's
                            # async_stream_synthesize output) that streaming
                            # just failed to fully drain -- it is NOT valid
                            # batch audio and must never be reused as
                            # `audio_bytes` below (that async_generator object
                            # is truthy and has no len(), so the fade-in call
                            # a few lines down would crash with "object of
                            # type 'async_generator' has no len()" instead of
                            # actually falling back). Force a real, fresh
                            # non-streaming resynthesis via generate_mulaw_tts
                            # instead of pretending the failed stream is bytes.
                            if _is_prefetched_iter:
                                _use_prefetched = False

                    # Generate TTS audio (Google TTS auto-detects SSML)
                    if self._tts_cancel.is_set() or not self.stream_sid:
                        self._prev_tts_tail = b""
                        return
                    if _use_prefetched:
                        audio_bytes = prefetched_bytes
                    else:
                        audio_bytes = await generate_mulaw_tts(
                            text=clean,
                            lang=lang,
                            voice=voice,
                            use_chirp3_hd=True,
                            speaking_rate=1.0,
                            use_ssml=use_ssml,
                            add_office_bg=False,
                            agent=self.agent,
                            db=getattr(self, "db", None),
                        )

                    if self._tts_cancel.is_set():
                        self._prev_tts_tail = b""
                        return

                    # Stream TTS to Twilio (clean mu-law; crossfade + fade-in above)
                    if audio_bytes and not self._tts_cancel.is_set():
                        # Apply fade-in only at the start of the utterance to avoid "phat" / pop
                        from app.utils.audio_utils import (
                            apply_micro_fade_in,
                            apply_micro_fade_out,
                        )

                        # Crossfade bridge disabled to prevent robotic stutter/distortion

                        # Hold back a tail for the NEXT chunk (only when not final)
                        next_tail = b""
                        to_play = audio_bytes

                        to_stream = to_play

                        if not self._twilio_buffer_primed and to_stream:
                            to_stream = apply_micro_fade_in(to_stream, duration_ms=25.0)
                            logger.debug(
                                "🔊 Applied micro fade-in to first TTS audio (25ms)"
                            )

                        # Apply a 25 ms fade-out only on the FINAL chunk so the listener
                        # never hears an abrupt cut at the end of an utterance. We do
                        # this BEFORE the optional background mix so the bed isn't
                        # accidentally faded with the voice.
                        if is_final and to_stream:
                            to_stream = apply_micro_fade_out(
                                to_stream, duration_ms=25.0
                            )

                        # User-configurable TTS voice gain (uniform across providers).
                        # Applied on speech bytes only; jitter priming + silence drain
                        # frames keep their 0xFF mulaw silence below.
                        #
                        # This whole buffer is pre-generated and handed to
                        # stream_mulaw_bytes_over_twilio() to be paced out at 20ms/frame,
                        # so baking a single gain value in here (before any real time has
                        # elapsed) would freeze the soft-duck state as of NOW, before the
                        # send loop even starts -- exactly the bug this fix addresses. So
                        # we only bake in the non-duck base gain (provider baseline + user
                        # volume slider -- static for the whole call, and the only part
                        # worth resolving once since it can hit Postgres for a BYO
                        # ElevenLabs key). The duck-reactive multiplier is re-resolved
                        # fresh on every frame at actual send time via frame_gain_fn below.
                        # Reuses the `tts_runtime` already resolved above in this same
                        # function (avoids a second resolve_tts_runtime()/BYO-key-decrypt
                        # call).
                        base_voice_gain = self._base_voice_gain_from_runtime(tts_runtime)
                        if to_stream and base_voice_gain != 1.0:
                            to_stream = apply_volume_fade(to_stream, base_voice_gain)

                        # Mix with ambient bed only when explicitly enabled for office profile.
                        if self._is_background_audio_enabled():
                            self._background_audio.set_user_level(
                                self._resolve_background_volume()
                            )
                            to_stream = self._background_audio.mix_with_background(
                                to_stream
                            )

                        # Prime Twilio jitter buffer once for first speak only.
                        prime_frames = (
                            0
                            if self._twilio_buffer_primed
                            else max(
                                0,
                                int(
                                    getattr(settings, "VOICE_TTS_PRIME_FRAMES", 1) or 1
                                ),
                            )
                        )

                        await stream_mulaw_bytes_over_twilio(
                            websocket=self.websocket,
                            stream_sid=self.stream_sid,
                            audio_bytes=to_stream,
                            pace_20ms=True,
                            cancel=self._tts_cancel,
                            prime_frames=prime_frames,
                            mirror_mulaw=self._livekit_recording_mirror(),
                            # Cheap per-frame duck multiplier (1.0 outside the duck
                            # window) -- see _apply_soft_duck(). Resolved fresh at
                            # actual send time so a SpeechStarted event landing
                            # mid-buffer immediately ducks the frames still to be sent.
                            frame_gain_fn=lambda: self._apply_soft_duck(1.0),
                        )
                        self._twilio_buffer_primed = True

                        # Drain Twilio's playout jitter buffer with a 60ms MULAW silence
                        # tail on the final chunk so the last word doesn't get clipped
                        # by the WebSocket / RTP shutdown that can follow immediately
                        # afterwards (e.g. agent [END_CALL]). This is symmetric with
                        # the priming we apply at the start of an utterance.
                        if is_final and not self._tts_cancel.is_set():
                            try:
                                silence_drain = bytes([0xFF]) * MULAW_FRAME_BYTES * 3
                                await stream_mulaw_bytes_over_twilio(
                                    websocket=self.websocket,
                                    stream_sid=self.stream_sid,
                                    audio_bytes=silence_drain,
                                    pace_20ms=True,
                                    cancel=self._tts_cancel,
                                    prime_frames=0,
                                )
                            except Exception as drain_err:
                                logger.debug(
                                    "Trailing silence drain failed (non-fatal): %s",
                                    drain_err,
                                )
                        elif not self._tts_cancel.is_set():
                            # Phase 4C-2: optional small trailing silence after a
                            # non-final chunk ending at a real sentence boundary —
                            # same eligibility rule and 0xFF frame bytes as the
                            # streaming path above, applied here for this
                            # (non-streaming / batch-fallback) code path too, so
                            # eligibility depends only on chunk content, not on
                            # which internal path happened to handle it.
                            try:
                                pause_frames = pause_frames_for_chunk(pacing, is_final)
                                if pause_frames > 0:
                                    await stream_mulaw_bytes_over_twilio(
                                        websocket=self.websocket,
                                        stream_sid=self.stream_sid,
                                        audio_bytes=bytes([0xFF])
                                        * MULAW_FRAME_BYTES
                                        * pause_frames,
                                        pace_20ms=True,
                                        cancel=self._tts_cancel,
                                        prime_frames=0,
                                    )
                            except Exception as pause_err:
                                logger.debug(
                                    "Inter-sentence pause failed (non-fatal): %s",
                                    pause_err,
                                )

                        # Update crossfade tail state
                        if self._tts_cancel.is_set():
                            self._prev_tts_tail = b""
                        else:
                            self._prev_tts_tail = (
                                b"" if is_final else (next_tail or b"")
                            )
                finally:
                    if self._tts_cancel.is_set():
                        self._prev_tts_tail = b""
                    self.is_speaking = False
                    self._is_tts_playing = False

        except Exception as e:
            logger.error("Error in _stream_tts_chunk: %s", e, exc_info=True)

    async def _stream_live_audio_chunk(self, mulaw_bytes: bytes) -> None:
        """
        Minimal outbound-audio primitive for the Gemini Live (native-audio
        speech-to-speech) path — bypasses TtsPipeline/_stream_tts_chunk
        entirely (there is no synthesized-text chunk here, just raw MULAW
        bytes already converted from Gemini's PCM16/24kHz output by
        VoiceOrchestrator._on_gemini_live_audio_chunk). Reuses the same
        low-level paced-frame-send primitive
        (``stream_mulaw_bytes_over_twilio``) the existing TTS send loops use,
        so Twilio's 20ms/160-byte MULAW framing/message format is not
        reimplemented here.

        Cancellation is gated on whichever native-audio provider's own
        minimal barge-in flag is active for this call —
        ``self._voice_orchestrator._gemini_live_cancel`` (see
        VoiceOrchestrator._on_gemini_live_interrupted) or
        ``self._voice_orchestrator._openai_realtime_cancel`` (see
        VoiceOrchestrator._on_openai_realtime_interrupted) — never on
        ``self._tts_cancel``, since no TtsPipeline task exists for either of
        these calls. Both cancel Events are always constructed
        unconditionally in ``VoiceOrchestrator.__init__`` regardless of
        which (if either) provider is active for this call, so resolving by
        ``_is_openai_realtime`` here is safe even before either provider's
        session has actually started.
        """
        if not mulaw_bytes or not self.stream_sid:
            return
        vo = self._voice_orchestrator
        if getattr(vo, "_is_openai_realtime", False):
            cancel = getattr(vo, "_openai_realtime_cancel", None)
        else:
            cancel = getattr(vo, "_gemini_live_cancel", None)
        if cancel is not None and cancel.is_set():
            return
        try:
            await stream_mulaw_bytes_over_twilio(
                websocket=self.websocket,
                stream_sid=self.stream_sid,
                audio_bytes=mulaw_bytes,
                pace_20ms=True,
                cancel=cancel,
                prime_frames=0,
                mirror_mulaw=self._livekit_recording_mirror(),
            )
        except Exception as exc:
            logger.error(
                "[GeminiLive] _stream_live_audio_chunk failed: %s", exc, exc_info=True
            )

    async def _prefetch_tts_audio(self, task: Dict[str, Any]) -> bytes | None:
        """
        Generate TTS audio bytes in the background WITHOUT acquiring _tts_lock
        and WITHOUT streaming to Twilio.

        Called by TtsPipeline._prefetch_worker while the previous chunk is
        still playing, so the audio is ready (or nearly ready) by the time
        _playback_worker needs it — eliminating the inter-chunk TTS TTFB gap.

        Returns raw μ-law bytes on success, None on cancellation or error.
        """
        try:
            text = task.get("text", "")

            if not text or not text.strip():
                return None
            if self._tts_cancel.is_set():
                return None

            clean = text.strip()
            lang = self.agent.language if self.agent and self.agent.language else "en"
            voice = (
                self.agent.voice_type
                if self.agent and self.agent.voice_type
                else "female"
            )
            tts_runtime = resolve_tts_runtime(self.agent, db=getattr(self, "db", None))
            tts_provider_slug = tts_runtime.adapter_slug

            streaming_text = prepare_tts_text_for_provider(clean, tts_provider_slug)
            if not streaming_text or not streaming_text.strip():
                return None

            if tts_provider_slug and tts_provider_slug not in ("google", ""):
                external_voice_id = tts_runtime.voice_external_id
                if not external_voice_id:
                    tts_voice = (
                        getattr(self.agent, "tts_voice", None) if self.agent else None
                    )
                    external_voice_id = getattr(tts_voice, "external_voice_id", None)
                if not external_voice_id and tts_provider_slug == "rime":
                    external_voice_id = "mistv2_Wildflower"
                elif not external_voice_id and tts_provider_slug == "hume":
                    from app.services.hume_tts_service import HUME_DEFAULT_VOICE

                    external_voice_id = HUME_DEFAULT_VOICE
                elif not external_voice_id and tts_provider_slug == "xai":
                    from app.services.xai_tts_service import XAI_DEFAULT_VOICE

                    external_voice_id = XAI_DEFAULT_VOICE
                if not external_voice_id:
                    return None
                adapter = get_tts_adapter(tts_provider_slug)
                provider_settings = dict(tts_runtime.settings_json)
                if tts_provider_slug == "elevenlabs":
                    provider_settings.setdefault("output_format", "ulaw_8000")
                    # Phase 4D-2: read the previously QUEUED chunk's text — captured
                    # synchronously by TtsPipeline.queue_tts() at enqueue time — not
                    # a shared instance attribute mutated after some other chunk's
                    # playback completes (that was the source of the stale
                    # `previous_text` race: chunk N+1's prefetch commonly runs
                    # concurrently with chunk N still playing).
                    previous_text = (task.get("_previous_text") or "").strip()
                    if previous_text:
                        provider_settings["previous_text"] = previous_text[-500:]
                elif tts_provider_slug == "rime":
                    # Rime adapter handles format internally; no output_format key needed.
                    pass
                else:
                    # xai/hume/google ignore this key — codec/sample_rate are fixed
                    # (mulaw/8kHz) in their own adapters, not read from provider_settings.
                    provider_settings.setdefault("output_format", "ulaw_8000")

                # Fold in any humanization overlay (e.g. ElevenLabs stability hint)
                # computed by TtsPipeline._process_chunk. Capability-gated and a
                # no-op for providers/decisions with nothing safe to apply — see
                # app.voice.tts_provider_capabilities.build_voice_settings_overlay.
                # Isolated try/except: a humanization failure must never prevent
                # this chunk's TTS request from going out with normal settings.
                try:
                    provider_settings.update(
                        build_voice_settings_overlay(
                            tts_provider_slug, task.get("_humanization_decision")
                        )
                    )
                except Exception as exc:
                    logger.debug("[TTS] humanization overlay skipped: %s", exc)

                # Use true async streaming for providers that support it (Rime, ElevenLabs).
                if hasattr(adapter, "async_stream_synthesize"):
                    _cancel_ref = self._tts_cancel

                    async def _async_provider_iter(
                        _adapter=adapter,
                        _text=streaming_text,
                        _vid=external_voice_id,
                        _cfg=provider_settings,
                        _cancel=_cancel_ref,
                    ):
                        async for chunk in _adapter.async_stream_synthesize(
                            text=_text,
                            voice_external_id=_vid,
                            settings_json=_cfg,
                        ):
                            if _cancel.is_set():
                                break
                            if chunk:
                                yield chunk

                    return _async_provider_iter()

                sync_iter = adapter.stream_synthesize(
                    text=streaming_text,
                    voice_external_id=external_voice_id,
                    settings_json=provider_settings,
                )

                async def _async_iter_from_sync(sync_source):
                    iterator = iter(sync_source)
                    sentinel = object()
                    while True:
                        if self._tts_cancel.is_set():
                            break
                        chunk = await asyncio.to_thread(next, iterator, sentinel)
                        if chunk is sentinel:
                            break
                        if chunk:
                            yield chunk

                return _async_iter_from_sync(sync_iter)

            else:
                # Google: stream and collect
                emo = detect_emotion(streaming_text)
                speaking_rate = {
                    "happy": 1.03,
                    "sad": 0.97,
                    "uncertain": 0.98,
                    "confident": 1.01,
                }.get(emo, 1.0)
                tts_voice = (
                    getattr(self.agent, "tts_voice", None) if self.agent else None
                )
                google_voice_name = getattr(tts_voice, "external_voice_id", None)
                audio_iter = google_tts_service.stream_text_to_speech(
                    text=streaming_text,
                    language=lang,
                    voice_type=voice,
                    speaking_rate=speaking_rate,
                    output_format="mulaw",
                    use_chirp3_hd=True,
                    sample_rate_hz=8000,
                    voice_name_override=google_voice_name,
                )

                async def _checked_async_iter(source_iter):
                    async for chunk in source_iter:
                        if self._tts_cancel.is_set():
                            break
                        if chunk:
                            yield chunk

                return _checked_async_iter(audio_iter)

        except Exception as exc:
            logger.warning(
                "[TTS] _prefetch_tts_audio failed for '%s…': %s", text[:30], exc
            )
            return None

    async def stream_tts_response(self, text: str):
        """Fast-first TTS with barge-in: cancellable streaming with prefix-first strategy.

        Enhanced with sentence-aware chunking for natural pauses.
        """
        try:
            if not text or not text.strip():
                return
            async with self._tts_lock:
                self._tts_cancel.clear()
                self.is_speaking = True
                try:
                    lang = (
                        self.agent.language
                        if self.agent and self.agent.language
                        else "en"
                    )
                    voice = (
                        self.agent.voice_type
                        if self.agent and self.agent.voice_type
                        else "female"
                    )
                    clean = text.strip()

                    # Smart chunking at sentence boundaries (10 words for natural flow)
                    prefix, suffix = smart_chunk_text(clean, max_words=10)

                    # Begin generating suffix in parallel (if any)
                    suffix_task = (
                        asyncio.create_task(
                            generate_mulaw_tts(
                                text=suffix,
                                lang=lang,
                                voice=voice,
                                use_chirp3_hd=True,
                                speaking_rate=1.0,
                                add_office_bg=False,
                                agent=self.agent,
                                db=getattr(self, "db", None),
                            )
                        )
                        if suffix
                        else None
                    )

                    # Generate prefix audio immediately
                    prefix_audio = await generate_mulaw_tts(
                        text=prefix,
                        lang=lang,
                        voice=voice,
                        use_chirp3_hd=True,
                        speaking_rate=1.0,
                        add_office_bg=False,
                        agent=self.agent,
                        db=getattr(self, "db", None),
                    )

                    # Apply the (non-duck) base voice gain BEFORE crossfade split so
                    # both prefix_main and the crossfaded suffix join are scaled
                    # consistently across all providers. The duck-reactive multiplier
                    # is intentionally NOT baked in here -- both prefix_main and the
                    # suffix/tail are streamed out later via stream_mulaw_bytes_over_twilio(
                    # frame_gain_fn=...), which re-resolves it fresh per frame at actual
                    # send time so it reacts to SpeechStarted events mid-buffer instead
                    # of freezing the duck state as of now, before any audio has played.
                    base_voice_gain = self._resolve_voice_volume_base()
                    if prefix_audio and base_voice_gain != 1.0:
                        prefix_audio = apply_volume_fade(prefix_audio, base_voice_gain)

                    # Hold back 50ms for crossfade with next chunk (smooth transitions)
                    overlap_bytes = 400  # 50ms at 8kHz
                    if len(prefix_audio) > overlap_bytes:
                        prefix_main = prefix_audio[:-overlap_bytes]
                        prefix_tail = prefix_audio[-overlap_bytes:]
                    else:
                        prefix_main = prefix_audio
                        prefix_tail = b""

                    # Stream main part immediately
                    if prefix_main:
                        # Apply micro fade-in to the very first part of the response
                        if not self._twilio_buffer_primed:
                            from app.utils.audio_utils import apply_micro_fade_in

                            prefix_main = apply_micro_fade_in(
                                prefix_main, duration_ms=25.0
                            )
                            logger.debug(
                                "🔊 Applied micro fade-in to initial prefix chunk"
                            )

                        await stream_mulaw_bytes_over_twilio(
                            websocket=self.websocket,
                            stream_sid=self.stream_sid,
                            audio_bytes=prefix_main,
                            pace_20ms=True,
                            cancel=self._tts_cancel,
                            prime_frames=0 if self._twilio_buffer_primed else 3,
                            mirror_mulaw=self._livekit_recording_mirror(),
                            frame_gain_fn=lambda: self._apply_soft_duck(1.0),
                        )
                        self._twilio_buffer_primed = True

                    # Stream remainder when ready and not cancelled
                    if suffix_task and not self._tts_cancel.is_set():
                        try:
                            suffix_audio = await suffix_task
                        except Exception:
                            suffix_audio = b""

                        if suffix_audio and base_voice_gain != 1.0:
                            suffix_audio = apply_volume_fade(
                                suffix_audio, base_voice_gain
                            )

                        if not self._tts_cancel.is_set():
                            if suffix_audio:
                                # Crossfade boundary to eliminate clicks
                                if prefix_tail and len(suffix_audio) > overlap_bytes:
                                    merged = crossfade_mulaw_segments(
                                        prefix_tail, suffix_audio, overlap_bytes
                                    )
                                else:
                                    merged = (prefix_tail or b"") + suffix_audio

                                await stream_mulaw_bytes_over_twilio(
                                    websocket=self.websocket,
                                    stream_sid=self.stream_sid,
                                    audio_bytes=merged,
                                    pace_20ms=True,
                                    cancel=self._tts_cancel,
                                    prime_frames=0,
                                    mirror_mulaw=self._livekit_recording_mirror(),
                                    frame_gain_fn=lambda: self._apply_soft_duck(1.0),
                                )
                            else:
                                # No suffix - flush held tail
                                if prefix_tail:
                                    await stream_mulaw_bytes_over_twilio(
                                        websocket=self.websocket,
                                        stream_sid=self.stream_sid,
                                        audio_bytes=prefix_tail,
                                        pace_20ms=True,
                                        cancel=self._tts_cancel,
                                        prime_frames=0,
                                        mirror_mulaw=self._livekit_recording_mirror(),
                                        frame_gain_fn=lambda: self._apply_soft_duck(1.0),
                                    )
                finally:
                    self.is_speaking = False

        except Exception as e:
            logger.error("Error in stream_tts_response: %s", e, exc_info=True)

    def _split_into_sentences(self, text: str) -> list:
        """
        Split text into sentences for streaming
        NOTE: This function is now deprecated with word-by-word streaming
        Kept for potential fallback or future use
        """
        import re

        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    async def send_audio_to_twilio(self, audio_data: bytes):
        """Send audio chunk to Twilio for immediate playback (legacy method)"""
        try:
            # Use new 20ms chunked streaming method
            await stream_mulaw_bytes_over_twilio(
                websocket=self.websocket,
                stream_sid=self.stream_sid,
                audio_bytes=audio_data,
                pace_20ms=True,
                mirror_mulaw=self._livekit_recording_mirror(),
            )

        except Exception as e:
            logger.error("Error in send_audio_to_twilio: %s", e)

    async def _send_in_progress_status(self, transcript: str, confidence: float):
        """Send in-progress status when confident word is detected"""
        try:
            if not self.call_session:
                return

            try:
                # Outbound lifecycle uses "connected"; inbound/web keep "in-progress".
                is_outbound = (self.call_session.call_type or "").lower() == "outbound"
                live_status = "connected" if is_outbound else "in-progress"
                if self.call_session.status != live_status:
                    was_connected = self.call_session.status == "connected"
                    self.call_session.status = live_status

                    # Set start time when confident speech is detected
                    if not self.call_session.start_time:
                        self.call_session.start_time = datetime.now(timezone.utc)

                    self.db.commit()

                    # Status Webhook — "connect" event for outbound calls. Inbound
                    # calls fire this from voice.py's Twilio status-callback handler
                    # instead; this is the outbound equivalent of that same
                    # transition, since outbound calls report "connected" here (on
                    # first confident speech) rather than via a Twilio callback.
                    if is_outbound and not was_connected:
                        try:
                            from app.models.call_flow import CallFlow

                            _call_flow = (
                                self.db.query(CallFlow)
                                .filter(
                                    CallFlow.id == self.call_session.call_flow_id,
                                    CallFlow.tenant_id == self.call_session.tenant_id,
                                )
                                .first()
                                if self.call_session.call_flow_id
                                else None
                            )
                            if _call_flow and _call_flow.status_webhook_enabled:
                                from app.services.system_webhook_service import (
                                    schedule_status_webhook,
                                )

                                schedule_status_webhook(
                                    self.call_session.id, "call.connected"
                                )
                        except Exception as exc:
                            logger.debug(
                                "Status webhook (outbound connect) dispatch skipped: %s",
                                exc,
                            )

                await broadcast_call_status_update(
                    call_session_id=str(self.call_session.id),
                    status=live_status,
                    metadata={
                        "call_sid": self.call_sid,
                        "stream_sid": self.stream_sid,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "message": "connected",
                        "event": "confident_speech_detected",
                        "detected_word": transcript,
                        "confidence": confidence,
                    },
                )

                # 🎯 START CREDIT MONITORING - Start billing when connected status is sent (first media packet + connected status)
                try:
                    if (
                        self.call_session
                        and str(self.call_session.id)
                        not in credit_service._active_monitors
                    ):
                        # Pass current DB session (credit service will create its own for async task)
                        asyncio.create_task(
                            credit_service.start_credit_monitoring(
                                db=self.db,
                                call_session_id=self.call_session.id,
                                tenant_id=self.call_session.tenant_id,
                                agent_id=self.call_session.agent_id,
                            )
                        )
                except Exception as e:
                    logger.debug("Could not start credit monitoring: %s", e)

            except Exception as e:
                logger.error("Error in _send_in_progress_status inner loop: %s", e)

            except Exception as e:
                logger.error(
                    "Error updating call status in _send_in_progress_status: %s", e
                )

        except Exception as e:
            logger.error("Error in _send_in_progress_status: %s", e, exc_info=True)
