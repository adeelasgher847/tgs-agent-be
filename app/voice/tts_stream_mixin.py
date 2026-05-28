"""
TTS Streaming Mixin for BidirectionalStreamHandler.
Handles background audio, TTS chunk streaming, prefetch, and audio delivery to Twilio.
"""
from __future__ import annotations

import asyncio
import base64
import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from app.core.agent_runtime import resolve_tts_runtime
from app.core.config import settings
from app.core.logger import logger
from app.services.bidirectional_stream_service import generate_mulaw_tts
from app.services.credit_service import credit_service
from app.services.google_tts_service import google_tts_service
from app.utils.audio_utils import stream_mulaw_bytes_over_twilio, crossfade_mulaw_segments
from app.utils.tts_adapter import get_tts_adapter
from app.utils.tts_preprocessing import detect_emotion
from app.utils.ssml_utils import strip_ssml_tags, smart_chunk_text
from app.utils.eleven_tts_text import prepare_tts_text_for_provider
from app.routers.general_websocket import broadcast_call_status_update

if TYPE_CHECKING:
    pass


class TtsStreamMixin:
    """TTS streaming and audio delivery methods for BidirectionalStreamHandler."""

    async def _start_background_audio_with_delay(self):
        """Start background loop after call stabilizes (dev-branch behavior)."""
        try:
            if not self._is_background_audio_enabled():
                return
            self._background_audio.set_user_level(self._resolve_background_volume())
            await self._background_audio.start_loop_if_enabled(delay_seconds=3.0)
        except Exception as e:
            logger.error(f"Error in _start_background_audio_with_delay: {e}", exc_info=True)

    def _is_background_audio_enabled(self) -> bool:
        """
        Enable ambient background only when:
        - agent TTS provider is ElevenLabs
        - tts_settings_json.background_enabled is not explicitly false
        - tts_settings_json.background_profile is "office" (or omitted)
        """
        if not self.agent:
            return False
        tts_provider_slug = resolve_tts_runtime(self.agent).adapter_slug
        if tts_provider_slug != "elevenlabs":
            return False

        settings_json = dict(getattr(self.agent, "tts_settings_json", None) or {})
        enabled_raw = settings_json.get("background_enabled", True)
        if isinstance(enabled_raw, str):
            enabled = enabled_raw.strip().lower() not in {"false", "0", "off", "no"}
        else:
            enabled = bool(enabled_raw)
        if not enabled:
            return False

        profile = str(settings_json.get("background_profile") or "office").strip().lower()
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

    async def _stream_tts_chunk(self, text: str, use_ssml: bool = False, is_final: bool = False, prefetched_bytes: Any = None):
        """
        Generate and stream a single TTS chunk (used by parallel pipeline worker).
        Simplified version without the complex prefix/suffix splitting.
        Note: Does NOT clear cancel flag - respects barge-in for entire queue.
        
        Args:
            text: Text or SSML to convert to speech
            use_ssml: Whether text contains SSML markup
        """
        try:
            from datetime import datetime, timezone
            
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
                try:
                    lang = self.agent.language if self.agent and self.agent.language else "en"
                    voice = self.agent.voice_type if self.agent and self.agent.voice_type else "female"
                    clean = text.strip()
                    tts_runtime = resolve_tts_runtime(self.agent)
                    tts_provider_slug = tts_runtime.adapter_slug

                    # If audio was pre-generated by the parallel prefetch pipeline, skip the
                    # TTS API call entirely and fall through to the batch playback path.
                    _use_prefetched = prefetched_bytes is not None and not self._tts_cancel.is_set()
                    _is_prefetched_iter = _use_prefetched and hasattr(prefetched_bytes, "__aiter__")

                    # Prefer true streaming TTS for longer responses (real-time playback).
                    # Keep cache-friendly path for very short phrases (e.g. quick ack).
                    word_count = len(clean.split())
                    stream_min_words = max(
                        1, int(getattr(settings, "VOICE_TTS_STREAM_MIN_WORDS", 2) or 2)
                    )
                    use_streaming_tts = (word_count >= stream_min_words and not _use_prefetched) or _is_prefetched_iter
                    if use_streaming_tts and not self._tts_cancel.is_set():
                        try:
                            import base64
                            import time
                            from app.utils.audio_utils import (
                                apply_micro_fade_in,
                                apply_micro_fade_out,
                                build_crossfade_bridge,
                                MULAW_FRAME_BYTES,
                            )

                            # We crossfade at chunk boundaries with a single 20ms overlap for speed.
                            overlap_bytes = MULAW_FRAME_BYTES  # 160 bytes (20ms)

                            async def send_frame(frame: bytes, pace: bool = True, state: dict = None):
                                if not frame:
                                    return
                                if self._tts_cancel.is_set() or not self.stream_sid:
                                    return
                                if self._is_background_audio_enabled():
                                    frame = self._background_audio.mix_tts_frame(frame)
                                payload = base64.b64encode(frame).decode("utf-8")
                                try:
                                    await self.websocket.send_json({
                                        "event": "media",
                                        "streamSid": self.stream_sid,
                                        "media": {"payload": payload}
                                    })
                                except RuntimeError:
                                    # WebSocket already closed (hangup). Stop sending immediately.
                                    self._tts_cancel.set()
                                    return
                                # Mark audio as actively playing on the first real frame sent.
                                if not getattr(self, "_is_tts_playing", False):
                                    self._is_tts_playing = True
                                    _first_audio_ts = time.perf_counter()
                                    self._metric_first_audio_ts = _first_audio_ts
                                    _first_token_ts = getattr(self, "_metric_first_token_ts", 0.0)
                                    if _first_token_ts > 0:
                                        _ttfa_ms = (_first_audio_ts - _first_token_ts) * 1000
                                        logger.info(
                                            "[Metrics] llm_first_token_to_first_audio_chunk=%.0f ms",
                                            _ttfa_ms,
                                        )
                                    # End-to-end latency: caller speech → first agent audio out.
                                    # Parsed by scripts/latency_p95.py to compute staging p95.
                                    _stt_final_ts = getattr(self, "_metric_stt_final_ts", 0.0)
                                    if _stt_final_ts > 0:
                                        _e2e_ms = (_first_audio_ts - _stt_final_ts) * 1000
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
                                    state["next_send"] = time.perf_counter() + state["send_interval"]
                                    return
                                state["next_send"] += state["send_interval"]
                                now = time.perf_counter()
                                sleep_dur = state["next_send"] - now
                                if sleep_dur > 0:
                                    await asyncio.sleep(sleep_dur)
                                elif sleep_dur < -0.03:
                                    state["next_send"] = time.perf_counter()

                            async def stream_mulaw_from_audio_iter(audio_iter):
                                """
                                Consume an async iterator of MULAW bytes and stream as 20ms frames.
                                Uses:
                                - Optional jitter-buffer priming (first speak only)
                                - Single crossfade bridge at chunk boundary (prev tail + next head)
                                - Tail holdback (20ms) between chunks to avoid clicks/distortion
                                """
                                if self._is_background_audio_enabled():
                                    self._background_audio.set_user_level(self._resolve_background_volume())

                                pace_state = {"send_interval": 0.02, "first": True, "next_send": time.perf_counter()}

                                # Prime Twilio jitter buffer once per utterance (2 frames = 40ms, paced so
                                # they arrive at proper 20ms intervals and actually fill the buffer).
                                if not self._twilio_buffer_primed:
                                    silent = bytes([0xFF]) * MULAW_FRAME_BYTES
                                    prime_frames = max(
                                        0, int(getattr(settings, "VOICE_TTS_PRIME_FRAMES", 1) or 1)
                                    )
                                    for _ in range(prime_frames):
                                        if self._tts_cancel.is_set():
                                            return
                                        await send_frame(silent, pace=True, state=pace_state)

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
                                    byte_buf.extend(chunk_bytes)

                                    # Convert bytes to 20ms frames
                                    while len(byte_buf) >= MULAW_FRAME_BYTES:
                                        frame = bytes(byte_buf[:MULAW_FRAME_BYTES])
                                        del byte_buf[:MULAW_FRAME_BYTES]
                                        pending_frames.append(frame)

                                        # Send oldest frame
                                        out = pending_frames.pop(0)
                                        if fade_needed and out:
                                            out = apply_micro_fade_in(out, duration_ms=25.0)
                                            fade_needed = False
                                        await send_frame(out, pace=True, state=pace_state)

                                # End of streaming responses: handle remainder
                                if self._tts_cancel.is_set():
                                    return

                                if is_final:
                                    # Flush any partial remainder (pad with silence so we
                                    # always send aligned 20ms (160-byte) frames to Twilio).
                                    if byte_buf:
                                        pad = MULAW_FRAME_BYTES - (len(byte_buf) % MULAW_FRAME_BYTES)
                                        if pad != MULAW_FRAME_BYTES:
                                            byte_buf.extend(b"\xFF" * pad)
                                        while len(byte_buf) >= MULAW_FRAME_BYTES:
                                            pending_frames.append(bytes(byte_buf[:MULAW_FRAME_BYTES]))
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
                                                out = apply_micro_fade_in(out, duration_ms=25.0)
                                                fade_needed = False
                                            if idx == last_idx and out:
                                                out = apply_micro_fade_out(out, duration_ms=25.0)
                                            await send_frame(out, pace=True, state=pace_state)
                                        pending_frames.clear()

                                    # Drain Twilio's playout jitter buffer with a short
                                    # MULAW silence tail (3×20ms = 60ms). Without this,
                                    # the last 40–80 ms of speech are sometimes clipped
                                    # because the WebSocket / RTP path closes before the
                                    # final media frame finishes playing.
                                    if not self._tts_cancel.is_set():
                                        silence_drain = bytes([0xFF]) * MULAW_FRAME_BYTES
                                        for _ in range(3):
                                            if self._tts_cancel.is_set():
                                                break
                                            await send_frame(silence_drain, pace=True, state=pace_state)

                                    self._prev_tts_tail = b""
                                else:
                                    # Non-final chunk: send all remaining frames (no tail holdback).
                                    # Holding back 1 frame for a crossfade bridge sounds good in theory,
                                    # but between chunks there is always a TTS API generation gap
                                    # (200–500 ms) during which Twilio's buffer drains to zero.
                                    # Crossfading a stale 20 ms tail with fresh audio after that gap
                                    # creates an audible click/stutter that is worse than a clean cut.
                                    if byte_buf:
                                        pad = MULAW_FRAME_BYTES - (len(byte_buf) % MULAW_FRAME_BYTES)
                                        if pad != MULAW_FRAME_BYTES:
                                            byte_buf.extend(b"\xFF" * pad)
                                        while len(byte_buf) >= MULAW_FRAME_BYTES:
                                            pending_frames.append(bytes(byte_buf[:MULAW_FRAME_BYTES]))
                                            del byte_buf[:MULAW_FRAME_BYTES]

                                    for out in pending_frames:
                                        if fade_needed and out:
                                            out = apply_micro_fade_in(out, duration_ms=25.0)
                                            fade_needed = False
                                        await send_frame(out, pace=True, state=pace_state)
                                    self._prev_tts_tail = b""

                                self._twilio_buffer_primed = True

                            # Stream text in near real-time from provider.
                            # For Google: use native async streaming API.
                            # For ElevenLabs: use HTTP chunk streaming via adapter.
                            streaming_text = strip_ssml_tags(clean) if use_ssml or clean.lstrip().startswith("<speak>") else clean
                            streaming_text = prepare_tts_text_for_provider(
                                streaming_text, tts_provider_slug
                            )
                            if not streaming_text or not streaming_text.strip():
                                return
                            if _is_prefetched_iter:
                                audio_iter = prefetched_bytes
                            elif tts_provider_slug and tts_provider_slug not in ("google", ""):
                                external_voice_id = tts_runtime.voice_external_id
                                if not external_voice_id:
                                    tts_voice = getattr(self.agent, "tts_voice", None) if self.agent else None
                                    external_voice_id = getattr(tts_voice, "external_voice_id", None)
                                if not external_voice_id and tts_provider_slug == "rime":
                                    external_voice_id = "mistv2_Wildflower"
                                if not external_voice_id:
                                    raise ValueError("TTS voice is not configured for streaming.")
                                adapter = get_tts_adapter(tts_provider_slug)
                                provider_settings = dict(tts_runtime.settings_json)
                                if tts_provider_slug == "elevenlabs":
                                    provider_settings.setdefault("output_format", "ulaw_8000")
                                    previous_text = (self._elevenlabs_prev_tts_text or "").strip()
                                    if previous_text:
                                        provider_settings["previous_text"] = previous_text[-500:]
                                elif tts_provider_slug == "rime":
                                    # Rime uses async_stream_synthesize — no output_format key needed
                                    # (mulaw 8 kHz is the default in RimeTTSAdapter).
                                    pass
                                else:
                                    provider_settings.setdefault("output_format", "ulaw_8000")

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
                                        async for chunk in _adapter.async_stream_synthesize(
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
                                            chunk = await asyncio.to_thread(next, iterator, sentinel)
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

                                tts_voice = getattr(self.agent, "tts_voice", None) if self.agent else None
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

                            await stream_mulaw_from_audio_iter(audio_iter)
                            if tts_provider_slug == "elevenlabs" and not self._tts_cancel.is_set():
                                self._elevenlabs_prev_tts_text = streaming_text[-500:]
                            return  # streaming path complete
                        except Exception as e:
                            logger.warning(f"⚠️ Streaming TTS failed, falling back to non-streaming: {e}")

                            # If call ended / barge-in occurred, never fall back to batch TTS.
                            if self._tts_cancel.is_set() or not self.stream_sid:
                                self._prev_tts_tail = b""
                                return
                    
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
                            build_crossfade_bridge,
                        )

                        # Crossfade bridge disabled to prevent robotic stutter/distortion
                        overlap_bytes = 0

                        # Hold back a tail for the NEXT chunk (only when not final)
                        next_tail = b""
                        to_play = audio_bytes

                        to_stream = to_play

                        if not self._twilio_buffer_primed and to_stream:
                            to_stream = apply_micro_fade_in(to_stream, duration_ms=25.0)
                            logger.debug("🔊 Applied micro fade-in to first TTS audio (25ms)")

                        # Apply a 25 ms fade-out only on the FINAL chunk so the listener
                        # never hears an abrupt cut at the end of an utterance. We do
                        # this BEFORE the optional background mix so the bed isn't
                        # accidentally faded with the voice.
                        if is_final and to_stream:
                            to_stream = apply_micro_fade_out(to_stream, duration_ms=25.0)

                        # Mix with ambient bed only when explicitly enabled for office profile.
                        if self._is_background_audio_enabled():
                            self._background_audio.set_user_level(self._resolve_background_volume())
                            to_stream = self._background_audio.mix_with_background(to_stream)
                        
                        # Prime Twilio jitter buffer once for first speak only.
                        prime_frames = 0 if self._twilio_buffer_primed else max(
                            0, int(getattr(settings, "VOICE_TTS_PRIME_FRAMES", 1) or 1)
                        )
                        
                        await stream_mulaw_bytes_over_twilio(
                            websocket=self.websocket,
                            stream_sid=self.stream_sid,
                            audio_bytes=to_stream,
                            pace_20ms=True,
                            cancel=self._tts_cancel,
                            prime_frames=prime_frames,
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

                        # Update crossfade tail state
                        if self._tts_cancel.is_set():
                            self._prev_tts_tail = b""
                        else:
                            self._prev_tts_tail = b"" if is_final else (next_tail or b"")
                finally:
                    if self._tts_cancel.is_set():
                        self._prev_tts_tail = b""
                    self.is_speaking = False
                    self._is_tts_playing = False

        except Exception as e:
            logger.error(f"Error in _stream_tts_chunk: {e}", exc_info=True)

    async def _prefetch_tts_audio(self, task: Dict[str, Any]) -> Optional[bytes]:
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
            use_ssml = task.get("use_ssml", False)

            if not text or not text.strip():
                return None
            if self._tts_cancel.is_set():
                return None

            clean = text.strip()
            lang = self.agent.language if self.agent and self.agent.language else "en"
            voice = self.agent.voice_type if self.agent and self.agent.voice_type else "female"
            tts_runtime = resolve_tts_runtime(self.agent)
            tts_provider_slug = tts_runtime.adapter_slug

            streaming_text = strip_ssml_tags(clean) if use_ssml or clean.lstrip().startswith("<speak>") else clean
            streaming_text = prepare_tts_text_for_provider(streaming_text, tts_provider_slug)

            if not streaming_text or not streaming_text.strip():
                return None

            if tts_provider_slug and tts_provider_slug not in ("google", ""):
                external_voice_id = tts_runtime.voice_external_id
                if not external_voice_id:
                    tts_voice = getattr(self.agent, "tts_voice", None) if self.agent else None
                    external_voice_id = getattr(tts_voice, "external_voice_id", None)
                if not external_voice_id and tts_provider_slug == "rime":
                    external_voice_id = "mistv2_Wildflower"
                if not external_voice_id:
                    return None
                adapter = get_tts_adapter(tts_provider_slug)
                provider_settings = dict(tts_runtime.settings_json)
                if tts_provider_slug == "elevenlabs":
                    provider_settings.setdefault("output_format", "ulaw_8000")
                    previous_text = (self._elevenlabs_prev_tts_text or "").strip()
                    if previous_text:
                        provider_settings["previous_text"] = previous_text[-500:]
                elif tts_provider_slug == "rime":
                    # Rime adapter handles format internally; no output_format key needed.
                    pass
                else:
                    provider_settings.setdefault("output_format", "ulaw_8000")

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
                speaking_rate = {"happy": 1.03, "sad": 0.97, "uncertain": 0.98, "confident": 1.01}.get(emo, 1.0)
                tts_voice = getattr(self.agent, "tts_voice", None) if self.agent else None
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
            logger.warning("[TTS] _prefetch_tts_audio failed for '%s…': %s", text[:30], exc)
            return None

    async def stream_tts_response(self, text: str):
        """Fast-first TTS with barge-in: cancellable streaming with prefix-first strategy.
        
        Enhanced with sentence-aware chunking for natural pauses.
        """
        try:
            from datetime import datetime, timezone
            
            if not text or not text.strip():
                return
            async with self._tts_lock:
                self._tts_cancel.clear()
                self.is_speaking = True
                try:
                    lang = self.agent.language if self.agent and self.agent.language else "en"
                    voice = self.agent.voice_type if self.agent and self.agent.voice_type else "female"
                    clean = text.strip()

                    # Smart chunking at sentence boundaries (10 words for natural flow)
                    prefix, suffix = smart_chunk_text(clean, max_words=10)

                    # Begin generating suffix in parallel (if any)
                    suffix_task = asyncio.create_task(
                        generate_mulaw_tts(
                            text=suffix,
                            lang=lang,
                            voice=voice,
                            use_chirp3_hd=True,
                            speaking_rate=1.0,
                            add_office_bg=False,
                            agent=self.agent,
                        )
                    ) if suffix else None

                    # Generate prefix audio immediately
                    prefix_audio = await generate_mulaw_tts(
                        text=prefix,
                        lang=lang,
                        voice=voice,
                        use_chirp3_hd=True,
                        speaking_rate=1.0,
                        add_office_bg=False,
                        agent=self.agent,
                    )

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
                            prefix_main = apply_micro_fade_in(prefix_main, duration_ms=25.0)
                            logger.debug("🔊 Applied micro fade-in to initial prefix chunk")

                        await stream_mulaw_bytes_over_twilio(
                            websocket=self.websocket,
                            stream_sid=self.stream_sid,
                            audio_bytes=prefix_main,
                            pace_20ms=True,
                            cancel=self._tts_cancel,
                            prime_frames=0 if self._twilio_buffer_primed else 3,
                        )
                        self._twilio_buffer_primed = True

                    # Stream remainder when ready and not cancelled
                    if suffix_task and not self._tts_cancel.is_set():
                        try:
                            suffix_audio = await suffix_task
                        except Exception:
                            suffix_audio = b""
                        
                        if not self._tts_cancel.is_set():
                            if suffix_audio:
                                # Crossfade boundary to eliminate clicks
                                if prefix_tail and len(suffix_audio) > overlap_bytes:
                                    merged = crossfade_mulaw_segments(prefix_tail, suffix_audio, overlap_bytes)
                                else:
                                    merged = (prefix_tail or b"") + suffix_audio
                                
                                await stream_mulaw_bytes_over_twilio(
                                    websocket=self.websocket,
                                    stream_sid=self.stream_sid,
                                    audio_bytes=merged,
                                    pace_20ms=True,
                                    cancel=self._tts_cancel,
                                    prime_frames=0,
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
                                    )
                finally:
                    self.is_speaking = False
        
        except Exception as e:
            logger.error(f"Error in stream_tts_response: {e}", exc_info=True)
    
    def _split_into_sentences(self, text: str) -> list:
        """
        Split text into sentences for streaming
        NOTE: This function is now deprecated with word-by-word streaming
        Kept for potential fallback or future use
        """
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
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
            )
        
        except Exception as e:
            logger.error(f"Error in send_audio_to_twilio: {e}")
    
    async def _send_in_progress_status(self, transcript: str, confidence: float):
        """Send in-progress status when confident word is detected"""
        try:
            if not self.call_session:
                return
            
            try:
                if self.call_session.status != "in-progress":
                    self.call_session.status = "in-progress"
                    
                    # Set start time when confident speech is detected
                    if not self.call_session.start_time:
                        self.call_session.start_time = datetime.now(timezone.utc)
                    
                    self.db.commit()
                
                # Broadcast "in-progress" event (confident word detected)
                await broadcast_call_status_update(
                    call_session_id=str(self.call_session.id),
                    status="in-progress",
                    metadata={
                        "call_sid": self.call_sid,
                        "stream_sid": self.stream_sid,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "message": "connected",
                        "event": "confident_speech_detected",
                        "detected_word": transcript,
                        "confidence": confidence
                    }
                )
                
                # 🎯 START CREDIT MONITORING - Start billing when connected status is sent (first media packet + connected status)
                try:
                    if self.call_session and str(self.call_session.id) not in credit_service._active_monitors:
                        # Pass current DB session (credit service will create its own for async task)
                        asyncio.create_task(credit_service.start_credit_monitoring(
                            db=self.db,
                            call_session_id=self.call_session.id,
                            tenant_id=self.call_session.tenant_id,
                            agent_id=self.call_session.agent_id
                        ))
                except Exception as e:
                    logger.debug(f"Could not start credit monitoring: {e}")
                    
            except Exception as e:
                logger.error(f"Error in _send_in_progress_status inner loop: {e}")
                    
            except Exception as e:
                logger.error(f"Error updating call status in _send_in_progress_status: {e}")
        
        except Exception as e:
            logger.error(f"Error in _send_in_progress_status: {e}", exc_info=True)
    
