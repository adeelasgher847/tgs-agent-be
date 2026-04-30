"""
Bidirectional WebSocket for Real-time Voice AI
Handles both STT (incoming audio) and TTS (outgoing audio) simultaneously
Target latency: 400–500ms (Vapi-style).

STT → LLM → TTS FLOW:
- Twilio sends audio every 20ms (MULAW 8kHz). We push each chunk to Deepgram STT.
- By default, LLM+TTS runs on **final** STT only (`VOICE_ENABLE_INTERIM_LLM=False`), matching
  stable one-reply-per-utterance behavior. Deepgram emits many more partials than classic
  Google STT; early interim LLM is opt-in and gated by min word count + confidence.
- Optional: first qualifying interim can start one LLM+TTS run (see `VOICE_ENABLE_INTERIM_LLM`).
- When final arrives after an interim run, we regenerate only if `_should_regenerate_on_final` says so
  (same-utterance extensions skip a second LLM).

PARALLEL TTS PIPELINE (Vapi-style):
- User Speech → STT Interim → LLM Chunk 1 → TTS Chunk 1 Playing
                             ↓ LLM Chunk 2 → TTS Chunk 2 Generating (parallel)
                             ↓ LLM Chunk 3 → TTS Chunk 3 Queued
- TTS generation and playback happen in parallel; new TTS patches embed with current
  playback via crossfade (build_crossfade_bridge) — no distortion, no sudden buffer noise.

NATURAL CONVERSATION FEATURES (Vapi-Style):
1. SSML Support:
   - <prosody> tags with varied rate (95%, 98%, 100%, 102%, 105%)
   - <prosody> tags with varied pitch (-1st, 0st, +1st, +2st)
   - <break> tags for natural pauses (150-200ms)
   - <audio> tags for breath sounds (DISABLED by default - can cause distortion)
   - Example: <prosody rate="95%" pitch="+1st">Alright.</prosody><break time="180ms"/>

2. Micro-Pauses & Hesitation Fillers:
   - Hesitation patterns: "Hmm <break time='120ms'/> I think..."
   - "Uh", "Well", "Let me see" WITH breaks (15% chance)
   - SPOKEN boundary fillers at EVERY 10-word mid-chunk break (100%):
     * "<break time='80-100ms'/><prosody>uhh/umm/uh/hmm</prosody>" pattern
     * ALWAYS added between chunks to eliminate tak-tak distortion
     * No silent gaps - natural spoken connectors keep audio flowing
   - Example: <speak>Hmm <break time="120ms"/> I think I can help with that.</speak>

3. Turn-Taking & Barge-In:
   - ENABLED - Agent stops immediately when user starts speaking
   - Detection: 2+ words (interim confidence can be noisy; barge-in should not depend on confidence)
   - Checked FIRST before interim gating (highest priority!)
   - TTS queue cleared (prevents old audio from resuming)
   - Barge-in still uses interim; normal replies prefer final when interim LLM is off

4. Persona & Variability:
   - Subtle prosody variations (95%-105% rate, ±1 semitone pitch)
   - Randomized breath/pause durations
   - Consistent voice persona from agent configuration

SMART CHUNKING WITH OVERLAP:
- 10 words per chunk for balanced quality + performance
- Major punctuation (. ! ? ;) triggers immediate chunk
- Minor punctuation (, : —) with 5+ words triggers chunk
- OVERLAP TECHNIQUE: Last 2 words of chunk 1 + filler + first words of chunk 2
  Example: Chunk 1: "Hello how are you doing today I'm doing" + SAVE("great thank")
           Chunk 2: "great thank" + "uhh" + "you for asking how"
  Result: Seamless transition with spoken fillers, no tak-tak distortion!

CACHING & LOW-LATENCY STRATEGIES:
1. Auto-Greeting on Connect:
   - Agent speaks FIRST when call connects (no waiting for user!)
   - Uses agent's first_message or default: "hello how are you"
   - Bypasses LLM entirely for instant greeting (<200ms)
   - Eliminates awkward silence at call start

2. Pre-cached Common Phrases (disabled — implementation commented in code):
   - 36+ common phrases were pre-generated at connect to warm the MULAW TTS cache
   - Greetings, acknowledgements, confirmations; <50ms when cache hit
   - Re-enable by uncommenting asyncio.create_task(self._precache_common_phrases()) and the method

3. Quick Acknowledgement Pattern (5-Word Rule + Probability):
   - Eligible when user says 5+ words; then only ~38% chance we send "Got it" (more natural).
   - Never used for emotional/serious content (help, emergency, problem, etc.).
   - Short ack plays first; full response streams in parallel.
   - Example: "Got it" → "checking that now..." (full reply)

4. Adaptive Max Tokens:
   - Yes/No queries: 15 tokens (ultra-fast)
   - Short queries (1-3 words): 25 tokens (fast)
   - Medium queries (4-7 words): 35 tokens (balanced)
   - Complex queries: Full configured tokens
   - 30-60% faster LLM generation for simple queries

5. TTS Client Pre-warming:
   - Google TTS client initialized at startup
   - Avoids first-call penalty (~500ms saved)

TTS GAPLESS / NO DISTORTION (Vapi-style):
- Micro fade-in (25ms) on first frame to avoid clicks/pops.
- Crossfade at chunk boundaries (build_crossfade_bridge) so new TTS embeds with
  currently playing audio — no tak-tak, no network delay spikes.
- Jitter buffer primed with 3×20ms silence (60ms) once per utterance; no sudden
  buffer underrun noise.
- Google TTS streaming: https://cloud.google.com/text-to-speech/docs/create-audio-text-streaming
  (Chirp 3: HD, MULAW 8kHz, StreamingSynthesizeConfig.)
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
import json
import base64
import asyncio
from typing import Any, Optional, Dict, Iterable, List
import time
from datetime import datetime, timezone, date
import uuid
import sys
import math
import re
from app.core.logger import logger

# Deepgram STT is used via SttPipeline (app/voice/stt_pipeline.py).

from app.services.call_session_service import call_session_service
from app.services.agent_service import agent_service
from app.services.voice_logging_service import VoiceLoggingService
from app.services.transcript_service import transcript_service
from app.services.gemini_service import gemini_service
from app.services.openai_service import openai_service
from app.services.groq_service import groq_service
from app.services.rag_service import rag_service
from app.services.credit_service import credit_service
from app.services.twilio_service import twilio_service
from app.services.voice_twilio_utils import get_twilio_credentials_for_call
from app.services.google_tts_service import google_tts_service
from app.services.tts_adapter import get_tts_adapter
from app.voice.tone_adapter import tone_adapter
from app.voice.turn_signals import TurnContext, build_turn_context, build_user_signals_block
from app.utils.tts_preprocessing import detect_emotion
from app.core.config import settings
from app.routers.general_websocket import broadcast_call_status_update
from app.utils.tts_preprocessing import preprocess_for_tts, quick_clean
from app.voice.stt_pipeline import SttPipeline
from app.voice.tts_pipeline import TtsPipeline
from app.voice.background_audio import BackgroundAudioManager
from app.voice.conversation_orchestrator import (
    VOICE_TUNABLES,
    ConversationOrchestrator,
    should_send_quick_ack,
)
from app.voice.voice_orchestrator import VoiceOrchestrator
from app.voice.rag_context import build_rag_context_block, build_rag_context_block_with_trace
from app.voice.tts_only_session import TtsOnlySession

# Import utilities and services
from app.utils.audio_utils import (
    ulaw_to_linear_sample,
    stream_mulaw_bytes_over_twilio,
    crossfade_mulaw_segments,
    build_crossfade_bridge,
    MULAW_FRAME_BYTES,
)
from app.utils.ssml_utils import (
    strip_ssml_tags,
    add_natural_ssml,
    clean_text_for_tts,
    smart_chunk_text
)
from app.services.bidirectional_stream_service import (
    generate_mulaw_tts,
    build_streaming_twiml,
    build_tts_only_twiml,
)
from app.utils.eleven_tts_text import (
    build_elevenlabs_audio_tag_prompt_block,
    get_elevenlabs_voice_prompt_rule_lines,
    prepare_tts_text_for_provider,
    supports_elevenlabs_audio_tags,
)


router = APIRouter()

# Vapi-style customEndpointingRules analogue: when the agent asks for email, we either
# defer a longer Deepgram endpointing for the first STT session or reconnect once with
# DEEPGRAM_STT_ENDPOINTING_MS_EXTENDED so spelling pauses do not split finals prematurely.
_EMAIL_AGENT_PROMPT_FOR_EXTENDED_STT_RE = re.compile(
    r"(?i)(?:"
    r"(?:provide|share|send|give)\s+(?:us\s+)?(?:your\s+)?(?:e-?mail\s+address|e-?mail|email)|"
    r"(?:what(?:'s|\s+is)|may\s+i\s+have|can\s+i\s+(?:have|get))\s+(?:your\s+)?(?:e-?mail|email)(?:\s+address)?|"
    r"(?:your\s+)?(?:e-?mail|email)\s+address(?:,?\s*please)?|"
    r"\bspell\b.*\b(?:e-?mail|email)|\b(?:e-?mail|email)\b.*\bspell\b"
    r")",
)


class BidirectionalStreamHandler:
    """Handles real-time bidirectional voice streaming (400–500ms target, Vapi-style gapless TTS)."""

    # Expose tunables on the class so they remain easy to discover in-context,
    # while the actual values live in VOICE_TUNABLES above.
    STT_INTERIM_INTERVAL_MS = VOICE_TUNABLES.stt_interim_interval_ms

    # Quick acknowledgement: 5-word rule + probability (Vapi+ naturalness)
    QUICK_ACK_MIN_WORDS = VOICE_TUNABLES.quick_ack.min_words
    QUICK_ACK_PROBABILITY = VOICE_TUNABLES.quick_ack.probability
    QUICK_ACK_SKIP_PHRASES = VOICE_TUNABLES.quick_ack.skip_phrases

    # Conversation context: keep the prompt small for latency (voice calls)
    HISTORY_MAX_MESSAGES = VOICE_TUNABLES.history_max_messages

    # Incremental TTS: flush when we have a complete thought/sentence
    TTS_FLUSH_MIN_WORDS = VOICE_TUNABLES.tts_flush_min_words
    TTS_FLUSH_MAX_WORDS = VOICE_TUNABLES.tts_flush_max_words  # keep chunks short for fast TTS start
    
    def __init__(
        self,
        websocket: WebSocket,
        call_session_id: str,
        agent_id: str,
        db: Session
    ):
        self.websocket = websocket
        self.call_session_id = call_session_id
        self.agent_id = agent_id
        self.db = db
        
        # STT (Input) state - Deepgram streaming with endpointing (speech_final)
        self.stream_sid = None
        self.call_sid = None
        self.current_speech = ""
        self._stt_pipeline: Optional[SttPipeline] = None
        # Interim → early LLM (optional; default off in settings for Deepgram stability)
        self._last_interim_text = ""
        self._last_interim_sent_ts = 0.0
        self._enable_interim_llm: bool = bool(
            getattr(settings, "VOICE_ENABLE_INTERIM_LLM", False)
        )
        self._min_interim_words: int = max(
            1, int(getattr(settings, "VOICE_MIN_INTERIM_WORDS", 4))
        )
        self._min_interim_confidence: float = float(
            getattr(settings, "VOICE_MIN_INTERIM_CONFIDENCE", 0.52)
        )
        self._min_interim_interval_sec = self.STT_INTERIM_INTERVAL_MS / 1000.0
        
        # TTS (Output) state - Parallel Pipeline
        self.is_speaking = False
        self._tts_cancel = asyncio.Event()   # barge-in cancel signal
        self._tts_lock = asyncio.Lock()      # serialize TTS streams
        self._tts_worker_task = None         # Backwards-compatible handle to pipeline worker
        self._tts_generation_tasks = []      # Track parallel TTS generation (reserved)
        self._prev_tts_tail = b""            # Last streamed audio tail for crossfade bridge
        self._tts_overlap_bytes = 400        # 50ms overlap at 8kHz (Vapi's approach for smooth transitions)
        self._twilio_buffer_primed = False   # Track if jitter buffer has been primed
        self._tts_pipeline: Optional[TtsPipeline] = None
        self._elevenlabs_prev_tts_text = ""
        self._use_ssml = True                # Enable SSML by default
        
        # Session data
        self.call_session = None
        self.agent = None
        self._last_offered_calendar_slots: List[datetime] = []
        self._last_requested_calendar_date: Optional[date] = None
        self._last_selected_calendar_slot: Optional[datetime] = None
        self._load_session_data()
        
        # User pickup detection (VAPI-style: actual user audio = user picked up)
        self._user_picked_up = False
        self._first_media_received = False
        self._in_progress_sent = False  # Track if in-progress status has been sent
        self._skip_audio_until = None  # Timestamp until which to skip audio (system messages)
        self._audio_level_samples = []  # Track audio levels to detect actual user audio
        _r = int(getattr(settings, "VOICE_MIN_AUDIO_RMS_FOR_PICKUP", 70) or 70)
        self._min_audio_level_threshold = max(20, min(250, _r))  # linear RMS; clamp to avoid misconfig
        self._stt_min_final_confidence: float = float(
            getattr(settings, "VOICE_STT_MIN_FINAL_CONFIDENCE", 0.26) or 0.26
        )
        self._stt_min_final_confidence = max(0.15, min(0.45, self._stt_min_final_confidence))
        self._enable_soft_final_fallback: bool = bool(
            getattr(settings, "VOICE_STT_ENABLE_SOFT_FINAL_FALLBACK", True)
        )
        self._stt_soft_min_final_confidence: float = float(
            getattr(settings, "VOICE_STT_SOFT_MIN_FINAL_CONFIDENCE", 0.16) or 0.16
        )
        self._stt_soft_min_final_confidence = max(0.10, min(0.35, self._stt_soft_min_final_confidence))
        self._stt_soft_min_words: int = int(
            getattr(settings, "VOICE_STT_SOFT_MIN_WORDS", 2) or 2
        )
        self._stt_soft_min_words = max(1, min(6, self._stt_soft_min_words))
        self._barge_in_min_conf: float = float(
            getattr(settings, "VOICE_BARGE_IN_MIN_CONFIDENCE", 0.26) or 0.26
        )
        self._barge_in_min_conf = max(0.15, min(0.5, self._barge_in_min_conf))
        self._barge_in_min_conf_1w: float = float(
            getattr(settings, "VOICE_BARGE_IN_MIN_CONFIDENCE_1W", 0.52) or 0.52
        )
        self._barge_in_min_conf_1w = max(0.4, min(0.75, self._barge_in_min_conf_1w))
        self._audio_samples_needed = 10  # Need 10 consecutive non-silent samples (200ms) to confirm user audio
        
        # Goodbye detection state
        self._call_ended = False  # Track if call has been ended due to goodbye detection

        # STT feed gate: set to False on stop/disconnect so media frames are not pushed
        # to Deepgram after the call ends (prevents silence-frame leak post call-end).
        self._stt_active = True

        # Signals the main receive loop to break cleanly when the call ends internally
        # (goodbye phrase, [END_CALL] token, voicemail, etc.) so the WebSocket closes
        # without waiting for Twilio to send a `stop` event.
        self._stop_event = asyncio.Event()
        # Post-call appointment finalization (exactly one task per handler lifetime)
        self._post_call_orchestration_scheduled = False
        # Longer Deepgram endpointing after agent asks for email (one-time upgrade per call).
        self._email_stt_endpointing_upgraded = False
        self._stt_deferred_endpointing_ms: Optional[int] = None

        # One response per turn: first interim that passes gates starts the LLM (dev-style: commit agent at stream end)
        self._turn_response_started = False  # True after first interim triggers LLM for this turn
        self._turn_response_seed_text = ""
        # In-flight LLM+TTS; wrapped in a task so barge-in can cancel while we await (like dev, but cancelable)
        self._llm_response_task: Optional[asyncio.Task] = None
        self._auto_greeting_sent = False
        self._recording_started = False

        # Deepgram can emit the same is_final twice within a short window — avoid triple LLM/transcript
        self._stt_last_final_raw: str = ""
        self._stt_last_final_monotonic: float = 0.0
        # Widened from 2.5s to 6.0s: Twilio sometimes re-endpoints the same short utterance
        # ("Hello?", "Yes", "Okay") up to ~5s apart, producing two finals with identical text
        # that both trigger the LLM → duplicate agent replies. One STT final per user turn is the goal.
        self._STT_DEDUP_FINAL_WINDOW_SEC: float = 6.0

        # Turn coordinator: remember the last few (user_norm, agent_norm, ts) pairs so we can
        #   (a) short-circuit a generate when the same user turn repeats within 15s, and
        #   (b) suppress duplicate agent transcript writes within 25s (belt-and-suspenders so
        #       the visible transcript never shows the same agent line twice).
        # Bounded to the last 5 entries — O(1) cost, no added latency.
        self._recent_agent_pairs: list[tuple[str, str, float]] = []
        self._DUP_USER_TURN_WINDOW_SEC: float = 15.0
        self._AGENT_LINE_DEDUP_WINDOW_SEC: float = 25.0
        self._RECENT_AGENT_PAIRS_MAX: int = 5

        # Background audio manager (dev-branch style embedded ambience loop).
        self._background_audio = BackgroundAudioManager(
            websocket=self.websocket,
            get_stream_sid=lambda: self.stream_sid,
            is_speaking_flag=lambda: self.is_speaking,
        )
        asyncio.create_task(self._background_audio.load_from_base64_async())

        # ── OLD direct pipeline wiring (replaced by VoiceOrchestrator below) ──
        # self._tts_pipeline = TtsPipeline(self)
        # self._tts_worker_task = self._tts_pipeline._worker_task
        # self._conversation = ConversationOrchestrator(self)
        # ─────────────────────────────────────────────────────────────────────

        # Pre-cache common phrases in background for instant responses (disabled; uncomment to re-enable)
        # asyncio.create_task(self._precache_common_phrases())

        # ── Voice Orchestration Layer ─────────────────────────────────────────
        # VoiceOrchestrator owns SttPipeline + TtsPipeline lifecycle and
        # coordinates the full turn flow (pickup → STT → barge-in → LLM → TTS).
        # It writes _tts_pipeline and _tts_worker_task back onto this handler
        # so all existing methods that reference self._tts_pipeline keep working.
        self._voice_orchestrator = VoiceOrchestrator(self)
    
    def _load_session_data(self):
        """Load call session and agent data"""
        try:
            session_uuid = uuid.UUID(self.call_session_id)
            self.call_session = call_session_service.get_call_session_by_id(self.db, session_uuid)
            
            if self.call_session and self.agent_id:
                agent_uuid = uuid.UUID(self.agent_id)
                self.agent = agent_service.get_agent_by_id(
                    self.db,
                    agent_uuid,
                    self.call_session.tenant_id
                )
                # Lazy safety net: ensure prompt KB exists for older agents
                # created before auto-ingest rollout.
                agent_service.ensure_agent_prompt_ingested(self.db, self.agent)
        except Exception as e:
            logger.error(f"Error loading session data: {e}", exc_info=True)
    
    async def _precache_common_phrases(self):
        """
        Pre-generate and cache common phrases for instant playback.
        Runs in background during initialization.
        """
        try:
            # Common phrases for instant responses (greetings, confirmations, acknowledgements)
            common_phrases = [
                # Greetings
                "Hello",
                "Hi there",
                "Hi",
                "Good morning",
                "Good afternoon",
                "Good evening",
                
                # Acknowledgements (Quick feedback)
                "Got it",
                "I see",
                "Okay",
                "Sure",
                "Alright",
                "Perfect",
                "Great",
                "Understood",
                
                # Confirmations
                "Yes",
                "No",
                "Absolutely",
                "Of course",
                
                # Thinking/Processing
                "Let me check that",
                "One moment please",
                "Just a second",
                "Let me see",
                
                # Transitions
                "Thank you",
                "Thanks",
                "You're welcome",
                
                # Closings
                "Goodbye",
                "Have a great day",
                "Thank you for calling",
                "Talk to you later",
            ]
            
            lang = self.agent.language if self.agent and self.agent.language else "en"
            voice = self.agent.voice_type if self.agent and self.agent.voice_type else "female"
            
            for phrase in common_phrases:
                try:
                    # Generate and cache (async, non-blocking)
                    await generate_mulaw_tts(
                        text=phrase,
                        lang=lang,
                        voice=voice,
                        use_chirp3_hd=True,
                        speaking_rate=1.0,
                        use_ssml=False
                    )
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"Error in precache_common_phrases: {e}")
    
    async def handle_media_message(self, message: dict):
        """
        Handle incoming MULAW audio from Twilio.

        The heavy lifting (pickup detection, grace period, STT pipeline lifecycle,
        barge-in, and STT callbacks) now lives in VoiceOrchestrator.on_audio_chunk().
        This method is intentionally thin — it just decodes the base64 payload and
        hands the raw bytes to the orchestrator.

        ── OLD direct pipeline code (now owned by VoiceOrchestrator) ────────────
        # if not self._user_picked_up:  ... RMS pickup detection ...
        # if self._skip_audio_until ... grace period ...
        # if not self._stt_active: return
        # if self._stt_pipeline is None: ... SttPipeline(...) ...
        # await self._stt_pipeline.feed_audio_chunk(audio_data)
        # ────────────────────────────────────────────────────────────────────────
        """
        try:
            payload = message.get("media", {}).get("payload")
            if not payload:
                return

            audio_data = base64.b64decode(payload)
            await self._voice_orchestrator.on_audio_chunk(audio_data)

        except Exception as e:
            logger.error(f"Error handling media message: {e}", exc_info=True)

    def _schedule_recreate_stt_for_email_collection(self, agent_text: str) -> None:
        """
        Defer STT session reconnect to the next event-loop tick.
        Delegates to VoiceOrchestrator which owns the SttPipeline lifecycle.

        ── OLD direct implementation (now in VoiceOrchestrator) ─────────────────
        # async def _deferred(): await self._maybe_recreate_stt_for_email_collection(text)
        # asyncio.create_task(_deferred())
        # ────────────────────────────────────────────────────────────────────────
        """
        self._voice_orchestrator.schedule_stt_recreate_for_email(agent_text)

    # _maybe_recreate_stt_for_email_collection is now handled inside
    # VoiceOrchestrator._maybe_upgrade_stt_for_email — kept as a no-op stub
    # for any call-site that might reference it directly.
    async def _maybe_recreate_stt_for_email_collection(self, agent_text: str) -> None:
        await self._voice_orchestrator._maybe_upgrade_stt_for_email(agent_text)
    
    # Removed chunk-based STT processing; relying on Deepgram streaming endpointing

    async def _cancel_inflight_llm_response(self) -> None:
        """Stop background LLM+TTS for this turn (barge-in or final regen)."""
        t = self._llm_response_task
        self._llm_response_task = None
        if t and not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        if self._tts_pipeline:
            await self._tts_pipeline.cancel_current_and_clear_queue()

    async def _complete_llm_turn_after_stt_final(self, transcript: str, confidence: float) -> None:
        """
        Run after the user's final message is in the DB (dev-style):
        - If interim already ran: regenerate only if final text differs; else let interim run finish, no second LLM.
        - No interim: full generate from final only.
        """
        if self._turn_response_started:
            should_regenerate = self._should_regenerate_on_final(transcript)
            self._turn_response_started = False
            self._turn_response_seed_text = ""
            self._last_interim_text = ""
            if should_regenerate:
                await self._cancel_inflight_llm_response()
                self._tts_cancel.clear()
                await self.generate_and_stream_response(
                    transcript,
                    confidence,
                    is_greeting=False,
                )
            else:
                if self._llm_response_task and not self._llm_response_task.done():
                    try:
                        await self._llm_response_task
                    except asyncio.CancelledError:
                        pass
                self._llm_response_task = None
            return

        self._turn_response_seed_text = ""
        self._last_interim_text = ""
        self._tts_cancel.clear()
        await self.generate_and_stream_response(
            transcript,
            confidence,
            is_greeting=False,
        )

    def _should_accept_final_transcript(self, transcript: str, confidence: float) -> bool:
        """
        Primary gate for Deepgram final transcripts:
        - Keep strong confidence threshold for normal flow.
        - Optional soft fallback accepts likely-human speech at lower confidence so
          soft callers are not dropped mid-call.
        """
        text = (transcript or "").strip()
        if not text:
            return False
        if confidence >= self._stt_min_final_confidence:
            return True
        if not self._enable_soft_final_fallback:
            return False
        if confidence < self._stt_soft_min_final_confidence:
            return False
        words = text.split()
        if len(words) < self._stt_soft_min_words:
            return False
        alpha_chars = sum(1 for ch in text if ch.isalpha())
        if alpha_chars < 3:
            return False
        # Ignore pure filler-ish very short low-confidence utterances.
        low = re.sub(r"[^a-z ]+", "", text.lower()).strip()
        filler = {"uh", "um", "hmm", "mm", "ah", "er", "huh", "hmm hmm", "uh huh"}
        if low in filler:
            return False
        return True
    
    async def _process_transcript(self, transcript: str, confidence: float):
        """Process a transcript (final result)"""
        try:
            if not self._should_accept_final_transcript(transcript, confidence):
                return

            # Skip duplicate finals (e.g. same "Hello?" endpointed multiple times) — Vapi-style single turn
            tstrip = (transcript or "").strip()
            _now = time.monotonic()
            if (
                tstrip
                and tstrip == self._stt_last_final_raw
                and (_now - self._stt_last_final_monotonic) < self._STT_DEDUP_FINAL_WINDOW_SEC
            ):
                logger.debug("STT: skipping duplicate final within %ss", self._STT_DEDUP_FINAL_WINDOW_SEC)
                return
            self._stt_last_final_raw = tstrip
            self._stt_last_final_monotonic = _now
            
            # 🎯 Check for goodbye words FIRST - end call if detected
            if await self._check_and_end_call_if_goodbye(transcript):
                return  # Stop processing - call is ending
            
            # 🎯 Check for voicemail detection - end call if detected
            if await self._check_and_end_call_if_voicemail(transcript):
                return  # Stop processing - call is ending
            
            # 🎯 Send "in-progress" status when confident word is detected (like "hello")
            # Only send once when we get a confident transcript with meaningful words
            if not self._in_progress_sent and confidence >= 0.1 and len(transcript.split()) > 0:
                await self._send_in_progress_status(transcript, confidence)
                self._in_progress_sent = True
            
            # Add to transcript (always)
            await self._add_to_transcript("client", transcript, "speech", confidence)
            self._update_booking_memory_from_user_turn(transcript)

            # Turn coordinator: if the same user turn was just handled (e.g. "Hello?" twice
            # across the pickup/first-second-endpoint boundary), do not generate a second
            # agent reply. The client line is still saved so the caller sees they repeated.
            user_turn_norm = self._normalize_turn_text(transcript)
            if self._has_recent_duplicate_reply_for(user_turn_norm):
                logger.info(
                    "TurnCoordinator: suppressing duplicate generate for user turn=%r (within %ss)",
                    transcript,
                    self._DUP_USER_TURN_WINDOW_SEC,
                )
                # Reset interim-turn flags so the next real user utterance is free to generate.
                self._turn_response_started = False
                self._turn_response_seed_text = ""
                self._last_interim_text = ""
                return

            await self._complete_llm_turn_after_stt_final(transcript, confidence)
            
        except Exception as e:
            logger.error(f"Error processing transcript: {e}", exc_info=True)

    async def _maybe_process_interim(self, transcript: str, confidence: float):
        """
        At most one early LLM+TTS run per user utterance when `VOICE_ENABLE_INTERIM_LLM` is True.
        Default is False: LLM runs on **final** STT only, avoiding double replies from Deepgram
        partials (e.g. "I'm" then "I'm feeling sad"). Barge-in still uses interim. When enabled,
        gates use `VOICE_MIN_INTERIM_WORDS` + `VOICE_MIN_INTERIM_CONFIDENCE`.
        """
        try:
            if not transcript:
                return

            word_count = len(transcript.split())

            # Barge-in: require real speech (not filler noise) while the agent is speaking.
            # The previous "any word" gate was triggering on "uh", "mm", and phantom short
            # STT hits, which cancelled good in-flight replies and produced the "arr arr"
            # stutter. Two thresholds keep both worlds (tuned via settings.* for soft speech):
            #   • ≥2 words → VOICE_BARGE_IN_MIN_CONFIDENCE
            #   • 1 word  → VOICE_BARGE_IN_MIN_CONFIDENCE_1W
            is_barge_in = self._tts_pipeline and self._tts_pipeline.is_speaking and (
                (word_count >= 2 and confidence >= self._barge_in_min_conf)
                or (word_count >= 1 and confidence >= self._barge_in_min_conf_1w)
            )
            if is_barge_in:
                await self._cancel_inflight_llm_response()
                self._turn_response_started = False
                self._turn_response_seed_text = ""
                self._last_interim_text = ""
                return

            # Final-only mode: do not start LLM from partials (barge-in already handled).
            if not self._enable_interim_llm:
                return

            # At most one interim LLM start per user turn; further partials are ignored
            if self._turn_response_started:
                return

            if self._should_defer_interim_response(transcript):
                return

            if confidence < self._min_interim_confidence or word_count < self._min_interim_words:
                return

            now = asyncio.get_event_loop().time()
            if (now - self._last_interim_sent_ts) < self._min_interim_interval_sec:
                return

            if self._last_interim_text and transcript.startswith(self._last_interim_text):
                advanced = transcript[len(self._last_interim_text) :].strip()
                if not advanced:
                    return

            self._last_interim_text = transcript
            self._last_interim_sent_ts = now
            self._turn_response_started = True
            self._turn_response_seed_text = transcript

            async def _run_interim() -> None:
                await self.generate_and_stream_response(
                    transcript,
                    confidence,
                    is_greeting=False,
                )

            if self._llm_response_task and not self._llm_response_task.done():
                self._llm_response_task.cancel()
                try:
                    await self._llm_response_task
                except asyncio.CancelledError:
                    pass
            self._llm_response_task = asyncio.create_task(_run_interim())
            try:
                await self._llm_response_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error in interim LLM task: {e}", exc_info=True)
            finally:
                self._llm_response_task = None
        except Exception as e:
            logger.error(f"Error processing interim: {e}")
    
    async def _send_quick_acknowledgement(self, user_text: str):
        """
        Send instant acknowledgement for longer queries while generating full response.
        Probability-based (QUICK_ACK_PROBABILITY) so we don't say "Got it" every time — more natural.
        Skips emotional/serious content so we never ack with "Got it" to e.g. "I have an emergency".
        """
        import random
        
        text = (user_text or "").strip()
        # First check if this text is even eligible for a quick acknowledgement
        if not should_send_quick_ack(text, VOICE_TUNABLES.quick_ack):
            return

        # Apply probability filter so we don't say "Got it" every single time
        if random.random() >= VOICE_TUNABLES.quick_ack.probability:
            return
        acks = [
            "Got it",
            "I see",
            "Okay",
            "Alright",
            "Sure",
            "Mm-hmm",
            "Oh, okay",
            "One moment",
            "Hang on a sec",
            "Let me check that",
        ]
        ack = random.choice(acks)
        if not self._tts_pipeline:
            return
        await self._tts_pipeline.queue_tts({
            "text": ack,
            "chunk_id": "quick_ack",
            "use_ssml": False,
            "is_acknowledgement": True,
            "is_final": False
        })
    
    async def generate_and_stream_response(
        self,
        user_text: str,
        confidence: float,
        is_greeting: bool = False,
    ):
        """
        Generate AI response and stream TTS in real-time WITH conversation history.
        Uses PARALLEL TTS PIPELINE (Vapi-style) for ultra-low latency.
        Agent reply is always committed to the transcript at end of stream (dev-style for interim and final).

        Args:
            user_text: User's input text (empty for greeting)
            confidence: STT confidence score
            is_greeting: If True, uses agent's first_message instead of calling LLM
        """
        try:
            from datetime import datetime, timezone
            import json
            
            # 👋 HANDLE AUTO-GREETING - Skip LLM, use pre-defined greeting
            if is_greeting:
                # Priority: greeting_message → first_message → skip (no hardcoded fallback)
                greeting_text = None
                if self.agent:
                    if getattr(self.agent, 'greeting_message', None):
                        greeting_text = self.agent.greeting_message.strip()
                    elif getattr(self.agent, 'first_message', None):
                        greeting_text = self.agent.first_message.strip()

                if not greeting_text:
                    # No greeting configured — wait for user to speak first
                    return

                # Add greeting to transcript
                await self._add_to_transcript("agent", greeting_text, "greeting")

                # Queue greeting TTS directly (skip LLM!)
                if not self._tts_pipeline:
                    return
                await self._tts_pipeline.queue_tts({
                    "text": greeting_text,
                    "chunk_id": "greeting",
                    "use_ssml": self._use_ssml,
                    "is_final": True
                })

                # Mark as not primed for the greeting
                self._twilio_buffer_primed = False

                return  # Done! No LLM needed for greeting
            
            # Reset TTS state for new response generation
            self._tts_cancel.clear()
            self._prev_tts_tail = b""           # Reset crossfade state so new response starts clean
            self._twilio_buffer_primed = False  # Ensure micro-fade and buffer priming for new utterance

            # Send quick acknowledgement for longer queries (instant from cache!)
            #await self._send_quick_acknowledgement(user_text)

            turn_context = build_turn_context(
                user_text,
                confidence,
                booking_context_active=self._is_booking_context_active(user_text),
                is_final=True,
            )

            # ------- RAG: build knowledge base context in voice layer -------
            tenant_uuid = self.call_session.tenant_id if self.call_session else None
            agent_uuid = self.agent.id if self.agent else None
            rag_agent_scope = None if (self.agent and self.agent.is_inbound_agent) else agent_uuid

            # Build KB context for the LLM with an explicit timeout so the voice
            # pipeline never hangs on embeddings/Pinecone.
            rag_context_block = ""
            rag_trace: dict = {}
            try:
                loop = asyncio.get_running_loop()

                def _build_rag():
                    return build_rag_context_block_with_trace(
                        user_text=user_text,
                        tenant_id=tenant_uuid,
                        agent_id=rag_agent_scope,
                    )

                rag_context_block, rag_trace = await asyncio.wait_for(
                    loop.run_in_executor(None, _build_rag),
                    timeout=settings.RAG_RETRIEVAL_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                rag_context_block, rag_trace = build_rag_context_block_with_trace(
                    user_text="",
                    tenant_id=tenant_uuid,
                    agent_id=rag_agent_scope,
                )
                rag_trace["status"] = "timeout"
                rag_trace["timeout"] = True
            except Exception as e:
                logger.error("RAG context build failed unexpectedly: %s", e, exc_info=True)
                rag_context_block, rag_trace = build_rag_context_block_with_trace(
                    user_text="",
                    tenant_id=tenant_uuid,
                    agent_id=rag_agent_scope,
                )
                rag_trace["status"] = "failure"
                rag_trace["error"] = str(e)

            inbound_kb_docs_context_block = ""
            business_knowledge_block = ""
            if self.agent and self.agent.is_inbound_agent and tenant_uuid and agent_uuid:
                try:
                    inbound_kb_docs_context_block = (
                        agent_service.build_inbound_kb_documents_context_block(
                            db=self.db,
                            inbound_agent_id=agent_uuid,
                            tenant_id=tenant_uuid,
                        )
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to build inbound KB docs block for agent %s: %s",
                        agent_uuid,
                        e,
                        exc_info=True,
                    )

            if tenant_uuid:
                try:
                    business_knowledge_block = (
                        agent_service.build_business_knowledge_context_block(
                            db=self.db,
                            tenant_id=tenant_uuid,
                            agent_id=agent_uuid,
                        )
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to build business knowledge block for agent %s: %s",
                        agent_uuid,
                        e,
                        exc_info=True,
                    )

            # One-line RAG summary (safe: no chunk text, no secrets)
            try:
                logger.info(
                    "RAG trace summary: status=%s timeout=%s initial=%s filtered=%s retrieve_error=%s",
                    rag_trace.get("status"),
                    rag_trace.get("timeout"),
                    rag_trace.get("initial_retrieved_count"),
                    rag_trace.get("filtered_count"),
                    rag_trace.get("retrieve_error"),
                )
            except Exception:
                # Logging must never break voice calls.
                pass
            
            # Build conversation context from transcript
            conversation_history = []
            if self.call_session and self.call_session.call_transcript:
                try:
                    conversation_history = json.loads(self.call_session.call_transcript) if isinstance(self.call_session.call_transcript, str) else self.call_session.call_transcript
                except:
                    conversation_history = []
            
            # Build history text - bounded filtered history for stable long-call memory
            history_text = ""
            if conversation_history:
                try:
                    history_lines = []
                    filtered = []
                    for msg in conversation_history:
                        if isinstance(msg, dict):
                            # Handle both 'content' and 'message' keys
                            role = msg.get('role', 'unknown')
                            content = msg.get('content') or msg.get('message', '')
                            message_type = msg.get('message_type', '')
                            
                            # Filter: Only include client and agent messages (skip system/greeting/status messages)
                            if content and role in ['client', 'agent'] and message_type not in ['greeting', 'system', 'status']:
                                filtered.append((role, content))

                    # Booking flows benefit from a slightly wider history window so
                    # the model keeps the already-collected service/date/slot in view.
                    max_msgs = getattr(self, "HISTORY_MAX_MESSAGES", 40)
                    if self._is_booking_context_active(user_text):
                        max_msgs = max(max_msgs, 39)
                    if len(filtered) > max_msgs:
                        filtered = filtered[-max_msgs:]

                    # Build history text from the bounded window
                    for role, content in filtered:
                        history_lines.append(f"{role.capitalize()}: {content}")

                    history_text = "\n".join(history_lines)
                except Exception:
                    history_text = ""
            
            booking_memory_block = self._build_booking_memory_block()

            # Build system prompt with agent personality + history
            agent_name = self.agent.name if self.agent and self.agent.name else "AI Assistant"
            agent_language = self.agent.language if self.agent and self.agent.language else "en"
            v_add = ""
            if self.call_session and self.call_session.call_metadata:
                voice_ctx = self.call_session.call_metadata.get("voice_dynamic_context")
                if isinstance(voice_ctx, dict):
                    v_add = (voice_ctx.get("system_prompt_addendum") or "").strip()
            v_block = (
                f"\n\n# THIS CALL — CANDIDATE & ROLE\n{v_add}\n"
                if v_add
                else ""
            )
            tts_provider = getattr(self.agent, "tts_provider", None) if self.agent else None
            tts_provider_slug = (getattr(tts_provider, "slug", None) or "").lower()
            elevenlabs_audio_tags_enabled = supports_elevenlabs_audio_tags(tts_provider_slug)
            if elevenlabs_audio_tags_enabled:
                output_plain_text_rule, no_ssml_rule_base, no_ssml_rule = (
                    get_elevenlabs_voice_prompt_rule_lines()
                )
            else:
                output_plain_text_rule = (
                    "- OUTPUT PLAIN TEXT ONLY: Do NOT output SSML, XML, or any tags. "
                    "Prosody is handled by the system."
                )
                no_ssml_rule_base = (
                    "4. NO SSML: Do NOT output <speak>, <prosody>, or any XML tags. Plain text only."
                )
                no_ssml_rule = "3. NO SSML: Plain text only. No <speak>, <prosody>, or XML."
            elevenlabs_audio_tag_block = build_elevenlabs_audio_tag_prompt_block(tts_provider_slug)

            _greeting_msg = (getattr(self.agent, 'greeting_message', None) or "").strip() if self.agent else ""
            greeting_instruction_block = (
                f"\n- GREETING: When the caller says hi, hello, or any greeting, respond with exactly: {_greeting_msg}\n"
                if _greeting_msg else ""
            )

            # Base prompt for phone conversations (voice-first, plain text only, no SSML)
            base_prompt = f"""# ROLE
You are {agent_name}, having a real-time phone call with a human.
{v_block}
# STYLE & TONE
- VOICE-FIRST: Your output is for Text-to-Speech. Use short, punchy sentences.
- NATURAL: Use natural fillers/interjections ONLY when they fit the emotion: "umm", "hmm", "oh", "alright", "hang on", "one moment" (max one per response).
- CONCISE: Max 20 words per response unless explaining something complex.
- NO ROBOT TALK: Avoid "As an AI" or formal greetings. Use "Hey," "Hi," or "Hello."
{output_plain_text_rule}
- TEXT HYGIENE: Avoid "..." (use a comma or short sentence). Avoid slashes like "FastAPI/ML" (say "FastAPI and ML").{greeting_instruction_block}
# CONVERSATION STATE
Previous conversation:
{history_text}

{booking_memory_block}
{rag_context_block}
{inbound_kb_docs_context_block}
# CRITICAL RULES
1. NO REPETITION: If the history shows you asked a question, move to the next point.
2. HANDLING SILENCE: If the user says something vague, ask a clarifying question.
3. TERMINATION: When the objective is met, say a friendly goodbye and end your response with exactly [END_CALL].
4. BUSINESS FACTS: For any question about the business name, address, phone, email, website, services, or pricing — answer using AUTHORITATIVE BUSINESS FACTS below. Never say you don't know if the answer is there.
{no_ssml_rule_base}

{elevenlabs_audio_tag_block}

{business_knowledge_block}
# CALENDAR ASSIST
- Collect details naturally. Do not tell the caller the appointment is confirmed, booked, or held during this call; the server finalizes scheduling after the call when checks pass.
- To list availability emit exactly: [CHECK_SLOTS:date=YYYY-MM-DD] (ISO date or the date the caller asked about).
- When they choose a slot the system offered, you may emit on one line: [BOOK_APPOINTMENT:name=<spoken name>,slot=<exact offered ISO datetime>,reason=<short reason with no commas>]. That line is only a machine hint; the server does not store name or email from it.
- Put each calendar token on ONE line; always end with ]. Field order: name, optional phone/email, slot, reason.
- If they change their mind, run [CHECK_SLOTS:...] again, then a new [BOOK_APPOINTMENT:...] with the new slot.
- Only use times from slots this call already returned; never pick a time in the past (see CURRENT DATE & TIME).

# GOAL
Continue the conversation based on the history above. Be {agent_name}."""
            
            # Use agent's custom system prompt if available, otherwise use base prompt
            if self.agent and self.agent.system_prompt:
                # Agent has custom system prompt - use it with context (voice-first, plain text)
                system_prompt = f"""# ROLE
You are {agent_name}, having a real-time phone call. You speak {agent_language} naturally.

# CUSTOM INSTRUCTIONS
{self.agent.system_prompt}
{v_block}
# STYLE & TONE
- VOICE-FIRST: Output is for Text-to-Speech. Use short sentences (max 20 words unless explaining).
- NATURAL: Use natural fillers/interjections ONLY when they fit the emotion: "umm", "hmm", "oh", "alright", "hang on", "one moment" (max one per response).
{output_plain_text_rule}
- TEXT HYGIENE: Avoid "..." (use a comma or short sentence). Avoid slashes like "FastAPI/ML" (say "FastAPI and ML").{greeting_instruction_block}
# CONVERSATION STATE
Previous conversation:
{history_text}

{booking_memory_block}
{rag_context_block}
{inbound_kb_docs_context_block}
# CRITICAL RULES
1. NO REPETITION: Do not repeat questions already asked. Move to the next point.
2. TERMINATION: When all objectives from your custom instructions are complete, say a friendly goodbye and end your response with exactly [END_CALL].
3. BUSINESS FACTS: For any question about the business name, address, phone, email, website, services, or pricing — answer using AUTHORITATIVE BUSINESS FACTS below. Never say you don't know if the answer is there.
{no_ssml_rule}

{elevenlabs_audio_tag_block}

{business_knowledge_block}
# CALENDAR ASSIST
- Collect details naturally. Do not tell the caller the appointment is confirmed, booked, or held during this call; the server finalizes scheduling after the call when checks pass.
- To list availability emit exactly: [CHECK_SLOTS:date=YYYY-MM-DD].
- When they choose a slot the system offered, you may emit on one line: [BOOK_APPOINTMENT:name=<spoken name>,slot=<exact offered ISO datetime>,reason=<short reason with no commas>]. That line is only a machine hint; the server does not store name or email from it.
- Put each calendar token on ONE line; always end with ]. Field order: name, optional phone/email, slot, reason.
- If they change their mind, run [CHECK_SLOTS:...] again, then a new [BOOK_APPOINTMENT:...] with the new slot.
- Only use times from slots this call already returned; never pick a time in the past (see CURRENT DATE & TIME).

# GOAL
Follow your custom instructions. Continue from the history above. Be {agent_name}."""
            elif self.agent and self.agent.model and self.agent.model.system_prompt:
                # Model has system prompt - use it (voice-first, plain text)
                system_prompt = f"""# ROLE
You are {agent_name}, having a real-time phone call. You speak {agent_language} naturally.

# MODEL INSTRUCTIONS
{self.agent.model.system_prompt}
{v_block}
# STYLE & TONE
- VOICE-FIRST: Output is for Text-to-Speech. Use short sentences (max 20 words unless explaining).
- NATURAL: Use fillers like "uhm," "well," "I see" occasionally.
{output_plain_text_rule}{greeting_instruction_block}
# CONVERSATION STATE
Previous conversation:
{history_text}

{booking_memory_block}
{rag_context_block}
{inbound_kb_docs_context_block}
# CRITICAL RULES
1. NO REPETITION: Do not repeat questions. Move to the next point.
2. TERMINATION: When all objectives are complete, say a friendly goodbye and end your response with exactly [END_CALL].
3. BUSINESS FACTS: For any question about the business name, address, phone, email, website, services, or pricing — answer using AUTHORITATIVE BUSINESS FACTS below. Never say you don't know if the answer is there.
{no_ssml_rule}

{elevenlabs_audio_tag_block}

{business_knowledge_block}
# CALENDAR ASSIST
- Collect details naturally. Do not tell the caller the appointment is confirmed, booked, or held during this call; the server finalizes scheduling after the call when checks pass.
- To list availability emit exactly: [CHECK_SLOTS:date=YYYY-MM-DD].
- When they choose a slot the system offered, you may emit on one line: [BOOK_APPOINTMENT:name=<spoken name>,slot=<exact offered ISO datetime>,reason=<short reason with no commas>]. That line is only a machine hint; the server does not store name or email from it.
- Put each calendar token on ONE line; always end with ]. Field order: name, optional phone/email, slot, reason.
- If they change their mind, run [CHECK_SLOTS:...] again, then a new [BOOK_APPOINTMENT:...] with the new slot.
- Only use times from slots this call already returned; never pick a time in the past (see CURRENT DATE & TIME).

# GOAL
Follow the model instructions. Continue from the history above. Be {agent_name}."""
            else:
                # Use base prompt
                system_prompt = base_prompt

            # Prepend current date/time so the agent knows what "today", "tomorrow",
            # and "past slots" mean. Also injected into appointment booking flow.
            _now_local = datetime.now(timezone.utc)
            _now_str = _now_local.strftime("%A, %B %d, %Y at %I:%M %p UTC")
            _user_signals = build_user_signals_block(turn_context)
            system_prompt = (
                f"# CURRENT DATE & TIME\nNow: {_now_str}\n\n"
                f"{_user_signals}\n\n"
                + system_prompt
            )

            # Get agent's configured model and provider
            llm_service = None
            model_name = "gemini-1.5-flash"  # Default fallback
            api_key = None
            temperature = 0.5
            max_tokens = 100
            booking_intent_turn = self._is_booking_intent_turn(user_text)
            
            if self.agent and self.agent.model:
                model_name = self.agent.model.model_name
                
                # Decrypt API key if available
                if self.agent.model.api_key:
                    try:
                        from app.core.security import decrypt_api_key
                        api_key = decrypt_api_key(self.agent.model.api_key)
                    except Exception as e:
                        logger.error(f"Failed to decrypt agent API key: {e}")
                        api_key = None  # Will fallback to settings.OPENAI_API_KEY or settings.GOOGLE_API_KEY
                else:
                    api_key = None  # Will use global key from .env
                
                # Use agent-specific config if available
                if self.agent.agent_temperature is not None:
                    temperature = self.agent.agent_temperature / 100.0  # Convert 0-100 to 0-1
                elif self.agent.model.temperature is not None:
                    temperature = self.agent.model.temperature / 100.0
                
                if self.agent.agent_max_tokens:
                    max_tokens = self.agent.agent_max_tokens
                elif self.agent.model.max_tokens:
                    max_tokens = self.agent.model.max_tokens

                # Booking turns need enough completion budget for action token emission.
                if booking_intent_turn:
                    max_tokens = max(max_tokens, 180)
                
                # Select service based on provider
                if self.agent.provider:
                    provider_name = self.agent.provider.name.lower()
                    if "openai" in provider_name:
                        llm_service = openai_service
                    elif "gemini" in provider_name or "google" in provider_name:
                        llm_service = gemini_service
                    elif "groq" in provider_name:
                        llm_service = groq_service
                    else:
                        # Default to Gemini
                        llm_service = gemini_service
                else:
                    llm_service = gemini_service
            else:
                # Fallback to Gemini
                llm_service = gemini_service
            
            # Stream LLM output and QUEUE for PARALLEL TTS PIPELINE (Vapi-style)
            chunk_counter = 0
            logger.info(f"🧠 Calling LLM ({llm_service.__class__.__name__ if hasattr(llm_service, '__class__') else 'Service'}) for response to: '{user_text[:20]}...'")
            
            async def try_stream(service, model: str, api_key_override: str = None) -> str:
                nonlocal chunk_counter
                import re
                import time

                response_accum = ""
                tts_buffer = ""
                end_call_after = False
                first_tts_chunk = True
                last_flush_ts = time.perf_counter()
                deferred_memory_scheduled = False

                def _schedule_deferred_memory_once() -> None:
                    nonlocal deferred_memory_scheduled
                    if deferred_memory_scheduled:
                        return
                    deferred_memory_scheduled = True
                    asyncio.create_task(
                        self._deferred_conversation_memory_update(turn_context, user_text)
                    )

                def _find_flush_index(buf: str):
                    """
                    Return an index (end-exclusive) where we can safely flush.
                    Prefer sentence boundaries. Fallback to comma/semicolon if buffer is getting long.
                    """
                    if not buf:
                        return None

                    # Prefer sentence boundaries: ., !, ? followed by whitespace/newline/end
                    last_boundary = None
                    for m in re.finditer(r"([.!?])(\s+|$)", buf):
                        last_boundary = m.end(1)

                    if last_boundary is not None:
                        prefix = buf[:last_boundary].strip()
                        if len(prefix.split()) >= self.TTS_FLUSH_MIN_WORDS:
                            return last_boundary

                    # If the buffer is getting long, allow a softer boundary split
                    words = buf.split()
                    if len(words) >= self.TTS_FLUSH_MAX_WORDS:
                        last_soft = None
                        for m in re.finditer(r"([,;:])(\s+|$)", buf):
                            last_soft = m.end(1)
                        if last_soft is not None:
                            prefix = buf[:last_soft].strip()
                            if len(prefix.split()) >= self.TTS_FLUSH_MIN_WORDS:
                                return last_soft

                    return None

                def _find_time_flush_index(buf: str):
                    """
                    Time-based flush (Vapi-style): if punctuation is delayed, flush on a safe word boundary
                    so we can start speaking fast. Returns an index (end-exclusive) or None.
                    """
                    if not buf:
                        return None
                    words = buf.split()
                    if len(words) < max(self.TTS_FLUSH_MIN_WORDS, 5):
                        return None

                    # Flush around ~6-8 words to start speaking quickly.
                    target_words = min(8, len(words))
                    m = re.match(rf"^(?:\\S+\\s+){{{target_words - 1}}}\\S+", buf)
                    if not m:
                        return None
                    return m.end()

                async for chunk in service.stream_text(
                    prompt=user_text,
                    system_prompt=system_prompt,
                    model_name=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key_override
                ):
                    if not chunk:
                        continue
                    # If barge-in requested, stop generating
                    if self._tts_cancel.is_set():
                        break

                    response_accum += chunk
                    tts_buffer += chunk

                    # Detect END_CALL early (may appear late, but handle if it appears mid-stream)
                    if "[END_CALL]" in response_accum:
                        end_call_after = True
                        # Remove it from TTS buffer immediately so it never gets spoken
                        tts_buffer = self._strip_control_tokens_for_tts(tts_buffer)

                    # Remove OUTCOME tokens from any in-flight buffer (never spoken)
                    if "[OUTCOME:" in tts_buffer:
                        tts_buffer = self._strip_control_tokens_for_tts(tts_buffer)

                    # Avoid spoken "final confirmation" before backend booking succeeds.
                    if "[BOOK_APPOINTMENT:" in response_accum:
                        tts_buffer = self._prepare_tts_text(tts_buffer)

                    # Flush complete thoughts early for faster perceived latency
                    flush_idx = _find_flush_index(tts_buffer)
                    # If punctuation-based flush isn't available, do a time-based flush (~150ms)
                    if flush_idx is None:
                        now_ts = time.perf_counter()
                        if (now_ts - last_flush_ts) >= 0.15:
                            flush_idx = _find_time_flush_index(tts_buffer)
                    if flush_idx is not None and not self._tts_cancel.is_set():
                        flush_text = tts_buffer[:flush_idx].strip()
                        tts_buffer = tts_buffer[flush_idx:].lstrip()

                        flush_text = self._prepare_tts_text(flush_text)
                        if flush_text:
                            if self._use_ssml:
                                clean_text = strip_ssml_tags(flush_text)
                                enhanced_text = preprocess_for_tts(
                                    clean_text,
                                    start_break_ms=0,
                                    between_sentence_break_ms=100,
                                )
                            else:
                                enhanced_text = quick_clean(flush_text)

                            enhanced_text = tone_adapter(enhanced_text, turn_context, self._use_ssml)

                            chunk_counter += 1
                            if self._tts_pipeline:
                                await self._tts_pipeline.queue_tts({
                                    "text": enhanced_text,
                                    "chunk_id": chunk_counter,
                                    "use_ssml": self._use_ssml,
                                    "is_final": False,
                                })
                                _schedule_deferred_memory_once()
                            first_tts_chunk = False
                            last_flush_ts = time.perf_counter()

                # Flush any remaining text as the FINAL chunk
                full_response = response_accum.strip()
                end_call_after = end_call_after or ("[END_CALL]" in full_response)

                # Never speak control tokens
                tts_buffer = self._prepare_tts_text(tts_buffer)

                if tts_buffer and not self._tts_cancel.is_set():
                    if self._use_ssml:
                        clean_text = strip_ssml_tags(tts_buffer)
                        enhanced_text = preprocess_for_tts(
                            clean_text,
                            start_break_ms=0,
                            between_sentence_break_ms=150,
                        )
                    else:
                        enhanced_text = quick_clean(tts_buffer)

                    enhanced_text = tone_adapter(enhanced_text, turn_context, self._use_ssml)

                    chunk_counter += 1
                    if self._tts_pipeline:
                        await self._tts_pipeline.queue_tts({
                            "text": enhanced_text,
                            "chunk_id": chunk_counter,
                            "use_ssml": self._use_ssml,
                            "is_final": True,
                            "end_call_after": end_call_after,
                        })
                        _schedule_deferred_memory_once()

                return response_accum.strip()

            final_text = None
            try:
                # Use agent's configured model and service
                final_text = await try_stream(llm_service, model_name, api_key)
            except Exception as e:
                logger.warning(f"⚠️ Primary LLM failed ({model_name}): {e}. Attempting fallback...")
                # Fallback: try alternate service
                try:
                    if llm_service == openai_service:
                        # Fallback to Gemini
                        final_text = await try_stream(gemini_service, "gemini-1.5-flash", None)
                    else:
                        # Fallback to OpenAI
                        final_text = await try_stream(openai_service, "gpt-3.5-turbo", None)
                except Exception as e:
                    logger.warning(f"⚠️ Secondary LLM fallback failed: {e}. Attempting VoiceLoggingService fallback...")
                    # Last fallback: non-streaming fast response via VoiceLoggingService
                    try:
                        final_text = await VoiceLoggingService.generate_agent_response(
                            speech_text=user_text,
                            confidence=confidence,
                            agent=self.agent,
                            db=self.db,
                            call_session_id=self.call_session.id if self.call_session else None
                        )
                        # Queue fallback response
                        if final_text and not self._tts_cancel.is_set():
                            safe_tts_text = self._prepare_tts_text(final_text)
                            out_text = tone_adapter(safe_tts_text.strip(), turn_context, self._use_ssml)
                            if not out_text:
                                out_text = "One moment please."
                            chunk_counter += 1
                            if self._tts_pipeline and out_text:
                                await self._tts_pipeline.queue_tts({
                                    "text": out_text,
                                    "use_ssml": self._use_ssml,
                                    "is_final": True,
                                })
                                asyncio.create_task(
                                    self._deferred_conversation_memory_update(turn_context, user_text)
                                )
                    except Exception as e:
                        logger.warning(f"⚠️ VoiceLoggingService fallback failed: {e}. Using ultimate fallback.")
                        # Ultimate fallback response
                        final_text = "I apologize, I'm having trouble responding right now. Could you please repeat that?"
                        utt = tone_adapter(final_text, turn_context, self._use_ssml)
                        chunk_counter += 1
                        if self._tts_pipeline:
                            await self._tts_pipeline.queue_tts({
                                "text": utt,
                                "chunk_id": chunk_counter,
                                "use_ssml": self._use_ssml,
                                "is_final": True
                            })
                            asyncio.create_task(
                                self._deferred_conversation_memory_update(turn_context, user_text)
                            )

            if final_text:
                # Two-step reliability: if booking intent exists but token is missing, run action extraction.
                if booking_intent_turn and not self._has_calendar_token(final_text):
                    extracted_token = await self._extract_calendar_action_token(
                        llm_service=llm_service,
                        model_name=model_name,
                        api_key=api_key,
                        user_text=user_text,
                        assistant_text=final_text,
                        history_text=history_text,
                        temperature=temperature,
                    )
                    if extracted_token:
                        logger.info("Action extraction fallback emitted token: %s", extracted_token[:140])
                        final_text = f"{final_text}\n{extracted_token}"

                # Strip control tokens from transcript (never saved to history)
                transcript_text = self._strip_control_tokens_for_tts(final_text).replace("[END_CALL]", "").strip()
                if "[BOOK_APPOINTMENT:" in final_text:
                    transcript_text = self._strip_premature_booking_confirmation(transcript_text)
                if transcript_text:
                    await self._add_to_transcript(
                        "agent",
                        transcript_text,
                        "agent_response",
                        message_metadata={
                            "user_text": user_text,
                            "rag_trace": rag_trace,
                        },
                    )
                    self._schedule_recreate_stt_for_email_collection(transcript_text)

                # Handle calendar tokens (fire-and-forget after TTS is already queued)
                if re.search(r"\[\s*CHECK_SLOTS\s*:", final_text, flags=re.IGNORECASE):
                    asyncio.create_task(self._handle_check_slots_token(final_text))
                elif re.search(r"\[\s*BOOK_APPOINTMENT\s*:", final_text, flags=re.IGNORECASE):
                    asyncio.create_task(self._handle_book_appointment_token(final_text))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error in generate_and_stream_response: {e}", exc_info=True)

    async def _deferred_conversation_memory_update(
        self, turn_context: TurnContext, user_text: str
    ) -> None:
        """
        Non-blocking hook after first TTS is queued. Extend with embeddings or summaries
        without adding latency to STT → LLM → TTS.
        """
        try:
            logger.debug(
                "deferred turn context: mood=%s phase=%s is_final=%s (user chars=%d)",
                turn_context.mood_label(),
                turn_context.conversation_phase,
                turn_context.is_final,
                len((user_text or "")),
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("deferred conversation memory update failed: %s", e, exc_info=True)
    
    # ── Calendar token handlers ───────────────────────────────────────────────

    @staticmethod
    def _normalize_calendar_slot_key(value: str) -> str:
        value = (value or "").strip().lower().replace(".", "")
        return re.sub(r"\s+", " ", value)

    def _cache_calendar_slots(self, slots: List) -> None:
        self._last_offered_calendar_slots = [slot.slot_start for slot in slots]
        self._last_selected_calendar_slot = None

    @staticmethod
    def _has_calendar_token(text: str) -> bool:
        if not text:
            return False
        return bool(re.search(r"\[\s*(?:CHECK_SLOTS|BOOK_APPOINTMENT)\s*:", text, flags=re.IGNORECASE))

    def _is_booking_intent_turn(self, user_text: str, model_text: str = "") -> bool:
        """Conservative booking-intent detector for token-budget and fallback extraction."""
        haystack = f"{user_text or ''} {model_text or ''}".lower()
        if not haystack.strip():
            return False
        booking_keywords = (
            "book", "booking", "schedule", "appointment", "reschedule", "slot", "available slot",
            "am", "pm", "a.m", "p.m", "date", "time", "tomorrow", "today",
        )
        return any(k in haystack for k in booking_keywords)

    def _is_booking_context_active(self, user_text: str = "") -> bool:
        return bool(
            self._last_offered_calendar_slots
            or self._last_requested_calendar_date
            or self._last_selected_calendar_slot
            or self._is_booking_intent_turn(user_text)
        )

    @staticmethod
    def _normalize_turn_text(text: str) -> str:
        cleaned = re.sub(r"[^\w\s:]", " ", (text or "").lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _has_recent_duplicate_reply_for(self, user_norm: str) -> bool:
        """
        True if a committed agent reply already handled this exact user turn within
        `_DUP_USER_TURN_WINDOW_SEC`. Prevents the "user says 'Hello?' twice → agent
        repeats the same greeting" failure mode. O(5) cost.
        """
        if not user_norm:
            return False
        now = time.monotonic()
        for u_norm, _a_norm, ts in self._recent_agent_pairs:
            if (now - ts) < self._DUP_USER_TURN_WINDOW_SEC and u_norm and u_norm == user_norm:
                return True
        return False

    def _is_duplicate_agent_line(self, user_text: Optional[str], agent_text: str) -> bool:
        """
        Transcript-level guard: within `_AGENT_LINE_DEDUP_WINDOW_SEC`, the same agent
        line (normalized) is treated as a duplicate even if the user turn differs. This
        is the final safety net that stops the visible transcript from ever showing the
        same agent message twice in quick succession.
        """
        if not agent_text:
            return False
        a_norm = self._normalize_turn_text(agent_text)
        if not a_norm:
            return False
        now = time.monotonic()
        u_norm = self._normalize_turn_text(user_text or "")
        for prev_u, prev_a, ts in self._recent_agent_pairs:
            if (now - ts) >= self._AGENT_LINE_DEDUP_WINDOW_SEC:
                continue
            if prev_a == a_norm:
                # Same agent line recently spoken — duplicate regardless of user turn.
                return True
            # Very similar (same prefix ≥ 90%) on a non-trivial reply — treat as dup too.
            if len(a_norm) > 30 and (a_norm.startswith(prev_a) or prev_a.startswith(a_norm)):
                shorter, longer = sorted((a_norm, prev_a), key=len)
                if shorter and len(shorter) / max(len(longer), 1) >= 0.9:
                    # And the user turn matches (or was empty) — safe to dedupe.
                    if not u_norm or not prev_u or prev_u == u_norm:
                        return True
        return False

    def _remember_agent_turn(self, user_text: Optional[str], agent_text: str) -> None:
        """Append (user_norm, agent_norm, ts) and bound the buffer to the last few entries."""
        if not agent_text:
            return
        a_norm = self._normalize_turn_text(agent_text)
        if not a_norm:
            return
        u_norm = self._normalize_turn_text(user_text or "")
        self._recent_agent_pairs.append((u_norm, a_norm, time.monotonic()))
        if len(self._recent_agent_pairs) > self._RECENT_AGENT_PAIRS_MAX:
            self._recent_agent_pairs = self._recent_agent_pairs[-self._RECENT_AGENT_PAIRS_MAX :]

    def _should_defer_interim_response(self, transcript: str) -> bool:
        """
        Avoid early LLM responses for short/ambiguous booking clarifications.
        This keeps the low-latency path for normal conversation while waiting for
        a final STT transcript when the caller is choosing dates/times or correcting us.
        """
        if not self._is_booking_context_active(transcript):
            return False

        normalized = self._normalize_turn_text(transcript)
        if not normalized:
            return False

        words = normalized.split()
        if len(words) <= 4:
            return True

        if re.fullmatch(r"\d{1,2}(?::\d{2})?\s*(?:a\s*m|am|p\s*m|pm)?", normalized):
            return True

        clarification_markers = (
            " am",
            " pm",
            " a m",
            " p m",
            "slot",
            "time",
            "date",
            "spell",
            "spelled",
            "already",
            "wrong",
            "available",
        )
        return any(marker in f" {normalized} " for marker in clarification_markers)

    def _is_natural_continuation_of_seed(self, final_norm: str, seed_norm: str) -> bool:
        """
        True when the final text is the same utterance as the interim, with a bit more
        at the end (user was still talking). In that case we should NOT run a second
        LLM+TTS (Vapi-style: one reply per barge-in segment, no double-audio / distortion).
        """
        if not seed_norm or not final_norm or final_norm == seed_norm:
            return False
        if not final_norm.startswith(seed_norm):
            return False
        # Very short interims: always allow final to replace (e.g. "I need" -> "I need a refund")
        seed_words = seed_norm.split()
        if len(seed_words) < 3:
            return False
        extra = final_norm[len(seed_norm) :].strip()
        if not extra:
            return True
        # New semantic / correction content → second response is appropriate
        if re.search(
            r"\b(refund|refunds|help|emergency|cancel|complaint|dispute|manager|operator|"
            r"supervisor|wrong|problem|lawyer|sue|angry|escalat)\b",
            extra,
        ):
            return False
        extra_words = extra.split()
        if len(extra_words) > 6:
            return False
        return True

    def _should_regenerate_on_final(self, final_transcript: str) -> bool:
        """
        If an interim run used partial STT, decide whether a final run with full text
        is needed. Skip regeneration when the final is a natural extension of the seed
        (avoids back-to-back TTS and sounds much closer to Vapi).
        """
        if not self._turn_response_seed_text:
            return False

        final_norm = self._normalize_turn_text(final_transcript)
        seed_norm = self._normalize_turn_text(self._turn_response_seed_text)
        if not final_norm:
            return True
        if not seed_norm:
            return True

        if final_norm == seed_norm:
            if not self._is_booking_context_active(final_transcript):
                return False
            final_slot = self._resolve_cached_calendar_slot(final_transcript)
            seed_slot = self._resolve_cached_calendar_slot(self._turn_response_seed_text)
            if final_slot and seed_slot and final_slot != seed_slot:
                return True
            return False

        # Booking: slot/date resolution changed — re-run
        if self._is_booking_context_active(final_transcript) or self._is_booking_context_active(
            self._turn_response_seed_text
        ):
            final_slot = self._resolve_cached_calendar_slot(final_transcript)
            seed_slot = self._resolve_cached_calendar_slot(self._turn_response_seed_text)
            if final_slot != seed_slot:
                return True
            correction_markers = ("wrong", "no no", "not ", "already", "spell", "11 00", "11 am")
            if any(marker in final_norm for marker in correction_markers):
                return True

        # Same line still being dictated — one spoken reply is enough
        if self._is_natural_continuation_of_seed(final_norm, seed_norm):
            return False

        # STT word-level revision (e.g. "Alex Carlton" → "Alex Carter"): a long common
        # prefix with only the trailing 1–2 words changed is almost always Deepgram
        # correcting a mishear, not the user adding new intent. Skipping regen avoids
        # the "first reply uses wrong name, second reply uses right name" double-TTS.
        seed_words = seed_norm.split()
        final_words = final_norm.split()
        if seed_words and final_words and len(seed_words) >= 4:
            common_prefix = 0
            for sw, fw in zip(seed_words, final_words):
                if sw == fw:
                    common_prefix += 1
                else:
                    break
            if (
                common_prefix >= len(seed_words) - 1
                and abs(len(final_words) - len(seed_words)) <= 1
            ):
                return False

        return True

    def _update_booking_memory_from_user_turn(self, transcript: str) -> None:
        if not transcript or not self._last_offered_calendar_slots:
            return
        resolved_slot = self._resolve_cached_calendar_slot(transcript)
        if resolved_slot is not None:
            self._last_selected_calendar_slot = resolved_slot

    def _build_booking_memory_block(self) -> str:
        if not self._is_booking_context_active():
            return ""

        lines = [
            "# BOOKING MEMORY",
            "Use this deterministic booking memory before asking repeated questions.",
        ]
        if self._last_requested_calendar_date is not None:
            lines.append(
                f"- Date already discussed: {self._last_requested_calendar_date.strftime('%A, %B %d, %Y')}."
            )
        if self._last_offered_calendar_slots:
            offered = ", ".join(
                slot.strftime("%I:%M %p").lstrip("0")
                for slot in self._last_offered_calendar_slots[:8]
            )
            lines.append(f"- Last offered slots: {offered}.")
        if self._last_selected_calendar_slot is not None:
            lines.append(
                "- Current caller-selected slot candidate: "
                f"{self._last_selected_calendar_slot.strftime('%A, %B %d at %I:%M %p')}."
            )
        lines.append(
            "- If the caller gives a short clarification like '11', '11 a.m.', or corrects you, "
            "resolve it against the last offered slots before asking again."
        )
        lines.append(
            "- If appointment type/date/slot is already present here or in recent history, "
            "do not ask for it again; move to the next missing field."
        )
        return "\n".join(lines)

    @staticmethod
    def _strip_premature_booking_confirmation(text: str) -> str:
        """
        Remove assistant self-confirmations so final confirmation comes only after backend success.
        """
        if not text:
            return ""
        patterns = [
            r"(?i)\b(?:great|done|perfect|all set)[^.!?]*\b(?:appointment)[^.!?]*\b(?:scheduled|confirmed|booked)\b[^.!?]*[.!?]?",
            r"(?i)\byour appointment[^.!?]*\b(?:scheduled|confirmed|booked)\b[^.!?]*[.!?]?",
            r"(?i)\ba confirmation message will be sent to you shortly[^.!?]*[.!?]?",
            r"(?i)\bwe look forward to seeing you[^.!?]*[.!?]?",
        ]
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _strip_control_tokens_for_tts(text: str) -> str:
        """
        Remove control/action tokens from text before it is spoken.
        Handles bracketed and malformed/unbracketed variants.
        """
        if not text:
            return ""
        out = text
        # Bracketed canonical tokens
        out = out.replace("[END_CALL]", "")
        out = re.sub(r"\[OUTCOME:[^\]]+\]", "", out)
        out = re.sub(r"\[CHECK_SLOTS:[^\]]*\]", "", out)
        out = re.sub(r"\[BOOK_APPOINTMENT:[^\]]*\]", "", out)
        # Malformed bracket-open tokens without closing bracket
        out = re.sub(r"\[(?:OUTCOME|CHECK_SLOTS|BOOK_APPOINTMENT):[^\]\n\r]*", "", out)
        # Unbracketed control tails occasionally produced by model
        out = re.sub(
            r"(?im)\b(?:OUTCOME|CHECK_SLOTS|BOOK_APPOINTMENT)\s*:\s*[^\n\r]*",
            "",
            out,
        )
        return out

    @staticmethod
    def _looks_like_control_leak(text: str) -> bool:
        """
        Detect token-like technical fragments that should never be spoken.
        """
        if not text:
            return False
        t = text.lower()
        leak_patterns = (
            r"\bbook_appointment\b",
            r"\bcheck_slots\b",
            r"\boutcome\b",
            r"\bslot\s*=",
            r"\breason\s*=",
            r"\bname\s*=",
            r"\bemail\s*=",
            r"\bphone\s*=",
            r"\bclient phone number slot\b",
        )
        return any(re.search(p, t, flags=re.IGNORECASE) for p in leak_patterns)

    def _prepare_tts_text(self, text: str) -> str:
        """
        Final text gate before queueing TTS.
        """
        cleaned = self._strip_control_tokens_for_tts(text or "")
        cleaned = self._strip_premature_booking_confirmation(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if self._looks_like_control_leak(cleaned):
            logger.warning("TTSGuard: dropped token-like leak text=%r", cleaned[:180])
            return ""
        return cleaned

    async def _extract_calendar_action_token(
        self,
        *,
        llm_service,
        model_name: str,
        api_key: Optional[str],
        user_text: str,
        assistant_text: str,
        history_text: str,
        temperature: float,
    ) -> Optional[str]:
        """Second-pass action extraction. Returns one action token or None."""
        if not self.call_session:
            return None

        offered_slots = ", ".join(
            slot.strftime("%Y-%m-%d %H:%M")
            for slot in self._last_offered_calendar_slots[:16]
        )
        extraction_system_prompt = (
            "You suggest a single calendar hint line from a phone-call turn. "
            "Output is not authoritative; the server validates everything.\n"
            "Return exactly one line and nothing else:\n"
            "- [BOOK_APPOINTMENT:name=<placeholder>,slot=<slot>,reason=<reason>] "
            "(optional phone=...,email=...)\n"
            "- [CHECK_SLOTS:date=YYYY-MM-DD]\n"
            "- NONE\n"
            "Rules:\n"
            "1) If user selected a concrete offered slot, return BOOK_APPOINTMENT with slot.\n"
            "2) If user asked to check availability, return CHECK_SLOTS.\n"
            "3) If uncertain, return NONE.\n"
            "4) Keep reason short and without commas.\n"
        )
        extraction_prompt = (
            f"Now (UTC): {datetime.now(timezone.utc).isoformat()}\n\n"
            f"Recent history:\n{history_text or '(empty)'}\n\n"
            f"Latest user text:\n{user_text or '(empty)'}\n\n"
            f"Assistant draft text:\n{assistant_text or '(empty)'}\n\n"
            f"Offered slot starts (YYYY-MM-DD HH:MM):\n{offered_slots or '(none cached)'}\n"
        )

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: llm_service.generate_text(
                    prompt=extraction_prompt,
                    system_prompt=extraction_system_prompt,
                    model_name=model_name,
                    temperature=min(temperature, 0.15),
                    max_tokens=180,
                    api_key=api_key,
                ),
            )
            content = (result.get("content") or "").strip()
            if re.search(r"^\[\s*BOOK_APPOINTMENT\s*:", content, flags=re.IGNORECASE):
                return content.splitlines()[0].strip()
            if re.search(r"^\[\s*CHECK_SLOTS\s*:", content, flags=re.IGNORECASE):
                return content.splitlines()[0].strip()
            return None
        except Exception as e:
            logger.warning("Calendar action extraction pass failed: %s", e)
            return None

    def _resolve_cached_calendar_slot(self, slot_raw: str) -> Optional[datetime]:
        normalized = self._normalize_calendar_slot_key(slot_raw)
        if not normalized or not self._last_offered_calendar_slots:
            return None

        for slot_dt in self._last_offered_calendar_slots:
            candidates = {
                slot_dt.isoformat(),
                slot_dt.strftime("%Y-%m-%d %H:%M"),
                slot_dt.strftime("%Y-%m-%d %I:%M %p").lstrip("0"),
                slot_dt.strftime("%I:%M %p").lstrip("0"),
                slot_dt.strftime("%H:%M"),
            }
            if slot_dt.minute == 0:
                candidates.add(slot_dt.strftime("%I %p").lstrip("0"))

            normalized_candidates = {
                self._normalize_calendar_slot_key(candidate)
                for candidate in candidates
            }
            if normalized in normalized_candidates:
                return slot_dt

        try:
            parsed_dt = datetime.fromisoformat(slot_raw.replace("Z", "+00:00"))
        except ValueError:
            parsed_dt = None

        if parsed_dt is not None:
            for slot_dt in self._last_offered_calendar_slots:
                if parsed_dt.tzinfo is None:
                    offered_local = slot_dt.replace(tzinfo=None, second=0, microsecond=0)
                    parsed_local = parsed_dt.replace(second=0, microsecond=0)
                    if offered_local == parsed_local:
                        return slot_dt
                else:
                    offered_utc = slot_dt.astimezone(timezone.utc).replace(second=0, microsecond=0)
                    parsed_utc = parsed_dt.astimezone(timezone.utc).replace(second=0, microsecond=0)
                    if offered_utc == parsed_utc:
                        return slot_dt

        for fmt in ("%I:%M %p", "%I %p", "%H:%M"):
            try:
                parsed_time = datetime.strptime(slot_raw.strip(), fmt).time()
                matches = [
                    slot_dt
                    for slot_dt in self._last_offered_calendar_slots
                    if slot_dt.hour == parsed_time.hour and slot_dt.minute == parsed_time.minute
                ]
                if len(matches) == 1:
                    return matches[0]
            except ValueError:
                continue

        return None

    async def _handle_check_slots_token(self, llm_response: str):
        """
        Called when LLM emits [CHECK_SLOTS:date=<value>].
        Fetches available slots and speaks them directly via TTS (no second LLM call).
        """
        try:
            import re as _re
            from datetime import date as _date, timedelta as _td, datetime as _dt, timezone as _tz
            from zoneinfo import ZoneInfo as _ZI
            from app.services.calendar_service import calendar_service as _cal

            m = _re.search(r"\[CHECK_SLOTS:date=([^\]]+)\]", llm_response)
            if not m:
                return

            if not self.call_session:
                return

            tenant_id = self.call_session.tenant_id
            agent_id = self.agent.id if self.agent else None

            loop = asyncio.get_running_loop()
            tenant_tz_str = await loop.run_in_executor(
                None,
                lambda: _cal.get_tenant_timezone(self.db, tenant_id),
            )
            try:
                tenant_tz = _ZI(tenant_tz_str)
            except Exception:
                tenant_tz = _tz.utc

            today = _dt.now(tenant_tz).date()
            raw_date = m.group(1).strip().lower()

            if raw_date in ("today", "aaj"):
                target = today
            elif raw_date in ("tomorrow", "kal", "tomorrow's"):
                target = today + _td(days=1)
            else:
                try:
                    target = _date.fromisoformat(raw_date)
                except ValueError:
                    target = today + _td(days=1)

            result = await loop.run_in_executor(
                None,
                lambda: _cal.get_available_slots(self.db, tenant_id, target, agent_id),
            )
            self._cache_calendar_slots(result.slots)
            self._last_requested_calendar_date = target

            if not result.slots:
                msg = f"Sorry, there are no available slots on {target.strftime('%A, %B %d')}. Please try another date."
            else:
                slot_labels = ", ".join(s.slot_label for s in result.slots[:6])
                suffix = f" and {len(result.slots) - 6} more" if len(result.slots) > 6 else ""
                msg = (
                    f"On {target.strftime('%A, %B %d')}, these slots are available: "
                    f"{slot_labels}{suffix}. Which time works for you?"
                )

            await self._add_to_transcript("agent", msg, "calendar_slots")
            if self._tts_pipeline:
                await self._tts_pipeline.queue_tts({
                    "text": msg,
                    "chunk_id": "calendar_slots",
                    "use_ssml": False,
                    "is_final": True,
                })
        except Exception as e:
            logger.error("Error in _handle_check_slots_token: %s", e, exc_info=True)

    def _client_transcript_lines_newest_first(self, limit: int = 16) -> list[str]:
        """Recent client utterances (newest first) for voice email recovery."""
        conversation_history: list = []
        if self.call_session and self.call_session.call_transcript:
            try:
                raw = self.call_session.call_transcript
                conversation_history = (
                    json.loads(raw) if isinstance(raw, str) else raw
                )
            except Exception:
                conversation_history = []
        if not isinstance(conversation_history, list):
            return []
        out: list[str] = []
        for msg in reversed(conversation_history):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = (msg.get("content") or msg.get("message") or "").strip()
            message_type = msg.get("message_type", "")
            if (
                role == "client"
                and content
                and message_type not in ("greeting", "system", "status")
            ):
                out.append(content)
                if len(out) >= limit:
                    break
        return out

    async def _handle_book_appointment_token(self, llm_response: str):
        """
        LLM may emit [BOOK_APPOINTMENT:...] as a non-authoritative intent hint.
        Backend stores only slot / reason in call_metadata.booking_intent.
        Name and email from the token are ignored. No in-call reservation or appointment commit.
        Final booking runs in post_call_appointment_service after validation.
        """
        try:
            import re as _re
            from datetime import datetime as _dt

            from app.services.call_session_contact_state import persist_booking_intent_fields

            m = _re.search(r"\[BOOK_APPOINTMENT:([^\]]+)\]", llm_response)
            if m:
                raw = m.group(1)
            else:
                # Tolerate malformed token without closing bracket during live calls.
                m_fallback = _re.search(r"\[BOOK_APPOINTMENT:(.+)$", llm_response, flags=_re.DOTALL)
                if not m_fallback:
                    return
                raw = m_fallback.group(1).strip()
                logger.warning(
                    "BOOK_APPOINTMENT token missing closing bracket; using fallback parser. token_tail=%s",
                    raw[:300],
                )

            raw_single_line = " ".join((raw or "").split())

            # Robust parse: name, optional phone/email, slot, optional reason (commas in reason).
            strict = _re.search(
                r"name=(?P<name>.*?),\s*(?:phone=(?P<phone>.*?),\s*)?"
                r"(?:email=(?P<email>.*?),\s*)?slot=(?P<slot>.*?)(?:,\s*reason=(?P<reason>.*))?$",
                raw_single_line,
            )
            if strict:
                slot_raw = (strict.group("slot") or "").strip()
                reason_val = (strict.group("reason") or "").strip()
                reason = reason_val or None
            else:
                # Backward-compatible fallback for legacy/messy token shapes.
                def _get(key: str) -> str:
                    km = _re.search(rf"{key}=([^,\]]+)", raw_single_line)
                    return km.group(1).strip() if km else ""

                slot_raw = _get("slot")
                reason = _get("reason") or None

            if not slot_raw:
                logger.warning("BOOK_APPOINTMENT token missing slot: %s", raw_single_line[:500])
                return

            if not self.call_session:
                return

            slot_start = self._resolve_cached_calendar_slot(slot_raw)
            if slot_start is None:
                try:
                    slot_start = _dt.fromisoformat(slot_raw.replace("Z", "+00:00"))
                except ValueError:
                    logger.warning("BOOK_APPOINTMENT: invalid slot datetime: %s", slot_raw)
                    return

            slot_iso = slot_start.isoformat()

            persist_booking_intent_fields(
                self.db,
                self.call_session,
                slot_start_iso=slot_iso,
                appointment_reason=reason,
            )
            self._last_selected_calendar_slot = slot_start
            try:
                self.db.refresh(self.call_session)
            except Exception:
                pass

            msg = (
                "I've noted your preferred time. After we finish the call, our system will finalize "
                "your appointment if everything checks out. Anything else I can help with?"
            )

            await self._add_to_transcript("agent", msg, "calendar_booking")
            if self._tts_pipeline:
                await self._tts_pipeline.queue_tts({
                    "text": msg,
                    "chunk_id": "calendar_booking",
                    "use_ssml": False,
                    "is_final": True,
                })
        except Exception as e:
            logger.error("Error in _handle_book_appointment_token: %s", e, exc_info=True)

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
        tts_provider = getattr(self.agent, "tts_provider", None)
        tts_provider_slug = (getattr(tts_provider, "slug", None) or "").lower()
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

    async def _stream_tts_chunk(self, text: str, use_ssml: bool = False, is_final: bool = False):
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

            # If stream isn't ready yet (race at call start), wait briefly rather than dropping TTS.
            if not self.stream_sid:
                for _ in range(100):  # ~1s max
                    if self._tts_cancel.is_set():
                        return
                    if self.stream_sid:
                        break
                    await asyncio.sleep(0.01)
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
                    tts_provider_slug = None
                    if self.agent and getattr(self.agent, "tts_provider", None):
                        tts_provider_slug = (self.agent.tts_provider.slug or "").lower()

                    # Prefer true streaming TTS for longer responses (real-time playback).
                    # Keep cache-friendly path for very short phrases (e.g. quick ack).
                    word_count = len(clean.split())
                    use_streaming_tts = word_count >= 4
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
                                    for _ in range(2):
                                        if self._tts_cancel.is_set():
                                            return
                                        await send_frame(silent, pace=True, state=pace_state)

                                # Frame buffers
                                byte_buf = bytearray()
                                pending_frames = []

                                # Boundary crossfade with previous chunk tail (if any)
                                need_bridge = bool(self._prev_tts_tail)
                                bridge_sent = False

                                # Whether we've applied fade-in for this utterance
                                fade_needed = not self._twilio_buffer_primed

                                async for chunk_bytes in audio_iter:
                                    if self._tts_cancel.is_set():
                                        return
                                    if not chunk_bytes:
                                        continue
                                    byte_buf.extend(chunk_bytes)

                                    # Build and send boundary bridge once we have enough head audio
                                    if need_bridge and not bridge_sent and len(byte_buf) >= overlap_bytes:
                                        head = bytes(byte_buf[:overlap_bytes])
                                        bridge = build_crossfade_bridge(self._prev_tts_tail, head, overlap_bytes=overlap_bytes)
                                        self._prev_tts_tail = b""
                                        # Drop head overlap from normal stream (bridge already covers it)
                                        del byte_buf[:overlap_bytes]
                                        need_bridge = False
                                        bridge_sent = True

                                        # bridge length == overlap_bytes => exactly one frame; bed is
                                        # added in send_frame (ducked) like other frames.
                                        if fade_needed and bridge:
                                            bridge = apply_micro_fade_in(bridge, duration_ms=25.0)
                                            fade_needed = False
                                        if bridge:
                                            await send_frame(
                                                bridge[:MULAW_FRAME_BYTES],
                                                pace=True,
                                                state=pace_state,
                                            )

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

                                # If we never had a chance to build a bridge but still had prev tail,
                                # clear it to avoid carrying stale audio forward.
                                if need_bridge and self._prev_tts_tail:
                                    self._prev_tts_tail = b""

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
                            if tts_provider_slug and tts_provider_slug != "google":
                                tts_voice = getattr(self.agent, "tts_voice", None) if self.agent else None
                                external_voice_id = getattr(tts_voice, "external_voice_id", None)
                                if not external_voice_id:
                                    raise ValueError("TTS voice is not configured for streaming.")
                                adapter = get_tts_adapter(tts_provider_slug)
                                provider_settings = dict(getattr(self.agent, "tts_settings_json", None) or {})
                                if tts_provider_slug == "elevenlabs":
                                    provider_settings.setdefault("output_format", "ulaw_8000")
                                    previous_text = (self._elevenlabs_prev_tts_text or "").strip()
                                    if previous_text:
                                        # Maintain natural continuity across app-level chunked TTS requests.
                                        provider_settings["previous_text"] = previous_text[-500:]
                                else:
                                    provider_settings.setdefault("output_format", "ulaw_8000")
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

                        overlap_bytes = int(getattr(self, "_tts_overlap_bytes", 400) or 400)

                        # If we have a previous tail, create a crossfade bridge and drop the overlapped head
                        bridge = b""
                        head_drop = 0
                        if self._prev_tts_tail:
                            bridge = build_crossfade_bridge(self._prev_tts_tail, audio_bytes, overlap_bytes=overlap_bytes)
                            head_drop = min(overlap_bytes, len(audio_bytes))

                        body = audio_bytes[head_drop:]

                        # Hold back a tail for the NEXT chunk (only when not final)
                        next_tail = b""
                        if (not is_final) and overlap_bytes > 0 and len(body) > overlap_bytes:
                            to_play = body[:-overlap_bytes]
                            next_tail = body[-overlap_bytes:]
                        else:
                            to_play = body

                        to_stream = (bridge + to_play) if bridge else to_play

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
                        
                        # Prime Twilio jitter buffer (3 frames = 60ms) for first speak only — no sudden buffer noise
                        prime_frames = 0 if self._twilio_buffer_primed else 3
                        
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
        
        except Exception as e:
            logger.error(f"Error in _stream_tts_chunk: {e}", exc_info=True)
    
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
    
    async def _check_and_end_call_if_goodbye(self, transcript: str):
        """
        Check if transcript contains goodbye words and end call if detected.
        Returns True if call was ended, False otherwise.
        
        Goodbye keywords detected:
        - thanks for calling
        - thank you for calling
        - bye, bye bye, goodbye
        - see you, see ya
        - have a great day, have a nice day
        - take care
        - that's all, that's it
        - i'm done, i'm finished
        - all done, all set
        """
        if self._call_ended:
            return False  # Already ended
        
        # Goodbye keywords/phrases (case-insensitive)
        goodbye_keywords = [
            "bye",
            "bye bye",
            "goodbye",
            "good bye",
            "see you",
            "see ya",
            "have a great day",
            "have a nice day",
            "thanks bye",
            "thank you bye",
            "we're done",
            "we're finished"
        ]
        
        # Convert transcript to lowercase for case-insensitive matching
        transcript_lower = transcript.lower().strip()
        
        # Check if any goodbye keyword/phrase is present in transcript
        for keyword in goodbye_keywords:
            if keyword in transcript_lower:
                try:
                    # Mark as ended to prevent multiple calls
                    self._call_ended = True
                    
                    # Use shared status updater so CallLog + inbound CRM sync hooks run reliably.
                    if self.call_session:
                        updated = call_session_service.update_call_session_status(
                            self.db,
                            self.call_session.id,
                            "completed",
                            ended_reason="User said goodbye",
                        )
                        if updated:
                            self.call_session = updated
                    
                    # End Twilio call with DB-derived credentials (no env fallback).
                    if self.call_sid and self.call_session:
                        try:
                            account_sid, auth_token = get_twilio_credentials_for_call(
                                self.db, self.call_session
                            )
                            twilio_service.end_call_with_credentials(
                                self.call_sid, account_sid, auth_token
                            )
                        except Exception as end_err:
                            logger.warning(
                                "Could not end Twilio call with DB credentials "
                                "(call_sid=%s, session=%s): %s",
                                self.call_sid,
                                self.call_session.id if self.call_session else None,
                                end_err,
                            )
                    
                    # Broadcast call ended event
                    if self.call_session:
                        try:
                            await broadcast_call_status_update(
                                call_session_id=str(self.call_session.id),
                                status="completed",
                                metadata={
                                    "call_sid": self.call_sid,
                                    "stream_sid": self.stream_sid,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "message": "call_ended",
                                    "event": "goodbye_detected",
                                    "detected_phrase": keyword,
                                    "transcript": transcript,
                                    "reason": "User said goodbye"
                                }
                            )
                        except Exception as e:
                            logger.debug(f"WebSocket broadcast failed after goodbye: {e}")

                    # Shut down STT + LLM + TTS and signal the main loop to exit
                    asyncio.create_task(self._full_shutdown())
                    return True
                    
                except Exception as e:
                    logger.error(f"Error ending call after goodbye: {e}", exc_info=True)
                    return False
        
        return False
    
    async def _end_call_after_agent_request(self):
        """End the call when agent response contained [END_CALL] (after TTS has played).

        We deliberately wait a short grace period (~200ms) AFTER the streaming
        TTS path has finished pushing its trailing silence drain. Twilio's
        outbound media buffer plus carrier-side jitter buffers can otherwise
        drop the last 80–150 ms of the goodbye phrase when the WebSocket /
        media stream is torn down too aggressively. The grace is well below
        any human-perceptible "extra silence" but eliminates the clipped
        goodbye that production has been hitting.
        """
        if self._call_ended:
            return
        try:
            try:
                await asyncio.sleep(0.20)
            except asyncio.CancelledError:
                # If the surrounding task is being cancelled (e.g. global
                # shutdown), continue with hangup instead of raising —
                # there's no benefit to leaving the call in a half-ended
                # state.
                pass

            self._call_ended = True
            if self.call_session:
                updated = call_session_service.update_call_session_status(
                    self.db,
                    self.call_session.id,
                    "completed",
                    ended_reason="Agent sent [END_CALL]",
                )
                if updated:
                    self.call_session = updated
            if self.call_sid and self.call_session:
                try:
                    account_sid, auth_token = get_twilio_credentials_for_call(
                        self.db, self.call_session
                    )
                    twilio_service.end_call_with_credentials(
                        self.call_sid, account_sid, auth_token
                    )
                except Exception as end_err:
                    logger.warning(
                        "Could not end Twilio call with DB credentials "
                        "(call_sid=%s, session=%s): %s",
                        self.call_sid,
                        self.call_session.id if self.call_session else None,
                        end_err,
                    )
            if self.call_session:
                try:
                    await broadcast_call_status_update(
                        call_session_id=str(self.call_session.id),
                        status="completed",
                        metadata={
                            "call_sid": self.call_sid,
                            "stream_sid": self.stream_sid,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "message": "call_ended",
                            "event": "end_call_token",
                            "reason": "Agent sent [END_CALL]",
                        },
                    )
                except Exception as e:
                    logger.debug(f"WebSocket broadcast after [END_CALL]: {e}")

            # Shut down STT + LLM + TTS and signal the main loop to exit
            asyncio.create_task(self._full_shutdown())
        except Exception as e:
            logger.error(f"Error ending call after [END_CALL]: {e}", exc_info=True)
    
    async def _check_and_end_call_if_voicemail(self, transcript: str):
        """
        Check if transcript contains voicemail keywords and end call if detected.
        Returns True if call was ended, False otherwise.
        
        Voicemail keywords detected:
        - voicemail, voice mail
        - forwarded to voicemail
        - unavailable
        - no one is available, no 1 is available
        - record your message
        - press pound, press #, pound key
        - hang up
        - at the tone
        """
        if self._call_ended:
            return False  # Already ended
        
        # Voicemail keywords/phrases (case-insensitive)
        voicemail_keywords = [
            "forwarded to voicemail",
            "forwarded to voice mail",
            "record your message",
            "press #",
            "pound key",
            "hang up",
            "at the tone",
            "after the tone",
            "after the beep"
        ]
        
        # Convert transcript to lowercase for case-insensitive matching
        transcript_lower = transcript.lower().strip()
        
        # Check if any voicemail keyword/phrase is present in transcript
        for keyword in voicemail_keywords:
            if keyword in transcript_lower:
                try:
                    # Mark as ended to prevent multiple calls
                    self._call_ended = True
                    
                    # Use shared status updater so CallLog + inbound CRM sync hooks run reliably.
                    if self.call_session:
                        updated = call_session_service.update_call_session_status(
                            self.db,
                            self.call_session.id,
                            "completed",
                            ended_reason="Voicemail detected",
                        )
                        if updated:
                            self.call_session = updated
                    
                    # End Twilio call immediately with DB-derived credentials (no env fallback).
                    if self.call_sid and self.call_session:
                        try:
                            account_sid, auth_token = get_twilio_credentials_for_call(
                                self.db, self.call_session
                            )
                            twilio_service.end_call_with_credentials(
                                self.call_sid, account_sid, auth_token
                            )
                        except Exception as end_err:
                            logger.warning(
                                "Could not end Twilio call with DB credentials "
                                "(call_sid=%s, session=%s): %s",
                                self.call_sid,
                                self.call_session.id if self.call_session else None,
                                end_err,
                            )
                    
                    # Broadcast call ended event
                    if self.call_session:
                        try:
                            await broadcast_call_status_update(
                                call_session_id=str(self.call_session.id),
                                status="completed",
                                metadata={
                                    "call_sid": self.call_sid,
                                    "stream_sid": self.stream_sid,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "message": "call_ended",
                                    "event": "voicemail_detected",
                                    "detected_phrase": keyword,
                                    "transcript": transcript,
                                    "reason": "Voicemail detected"
                                }
                            )
                        except Exception as e:
                            logger.debug(f"WebSocket broadcast failed after voicemail detection: {e}")

                    # Shut down STT + LLM + TTS and signal the main loop to exit
                    asyncio.create_task(self._full_shutdown())
                    return True
                    
                except Exception as e:
                    logger.error(f"Error ending call after voicemail detection: {e}", exc_info=True)
                    return False
        
        return False
    
    async def _add_to_transcript(
        self,
        role: str,
        message: str,
        message_type: str = "speech",
        confidence: Optional[float] = None,
        message_metadata: Optional[dict] = None,
    ):
        """Add message to transcript (SSML tags are automatically stripped)"""
        try:
            if not self.call_session:
                return
            
            # Strip SSML tags before saving to transcript (keep only clean text)
            clean_message = strip_ssml_tags(message)

            # Final dedupe gate for spoken agent replies (agent_response / greeting only —
            # calendar_slots / calendar_booking are informational and must never be skipped).
            # If the same line was committed within the last ~25s we skip the DB write AND
            # the WebSocket broadcast so the user/dashboard never sees duplicate lines.
            if role == "agent" and message_type in {"agent_response", "greeting"}:
                user_text_meta = None
                if message_metadata:
                    user_text_meta = message_metadata.get("user_text") or message_metadata.get("query")
                if self._is_duplicate_agent_line(user_text_meta, clean_message):
                    logger.info(
                        "TranscriptDedupe: skipping duplicate agent line (type=%s, msg=%r)",
                        message_type,
                        clean_message[:80],
                    )
                    return

            added = await transcript_service.add_and_broadcast_message(
                db=self.db,
                call_session_id=self.call_session.id,
                role=role,
                message=clean_message,  # Save clean text without SSML
                message_type=message_type,
                agent_id=self.agent.id if self.agent else None,
                user_id=self.call_session.user_id,
                confidence=confidence,
                metadata=message_metadata
            )
            if added is None:
                return

            # Remember committed agent lines for future dedupe / turn-coordination.
            if role == "agent" and message_type in {"agent_response", "greeting"}:
                user_text_meta = None
                if message_metadata:
                    user_text_meta = message_metadata.get("user_text") or message_metadata.get("query")
                self._remember_agent_turn(user_text_meta, clean_message)
            
            # Update legacy field
            conversation = transcript_service.get_conversation_array(self.db, self.call_session.id)
            self.call_session.call_transcript = conversation
            self.db.commit()

            from app.services.call_session_contact_state import sync_contact_intake_after_message

            sync_contact_intake_after_message(
                self.db,
                self.call_session.id,
                role=role,
                message=clean_message,
            )
            try:
                self.db.refresh(self.call_session)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Error in _add_to_transcript: {e}", exc_info=True)
    
    async def handle_start_message(self, message: dict):
        """Handle stream start - Just WebSocket connection (NOT user pickup!)"""
        try:
            self.stream_sid = message.get("streamSid")
            # Notify orchestrator so it can perform any stream-SID-dependent setup.
            self._voice_orchestrator.set_stream_sid(self.stream_sid)
            start = message.get("start", {})
            self.call_sid = start.get("callSid")

            # Inbound calls use Connect/Stream, so start recording explicitly here.
            # This enables recording-status webhook -> call_session.recording_url persistence.
            if (
                self.call_sid
                and self.call_session
                and self.call_session.call_type == "inbound"
                and not self._recording_started
            ):
                try:
                    account_sid, auth_token = get_twilio_credentials_for_call(
                        self.db, self.call_session
                    )
                    started = twilio_service.start_recording_with_credentials(
                        self.call_sid, account_sid, auth_token
                    )
                    if started:
                        self._recording_started = True
                except Exception as rec_err:
                    logger.warning(
                        "Could not start inbound recording (call_sid=%s, session=%s): %s",
                        self.call_sid,
                        self.call_session.id if self.call_session else None,
                        rec_err,
                    )

            # DO NOT start credit monitoring or greeting here!
            # Wait for first media packet (user actually picks up - VAPI-style)

        except Exception as e:
            logger.error(f"Error in handle_start_message: {e}", exc_info=True)
    
    async def _handle_user_pickup(self):
        """Handle user pickup - called when actual user audio detected (not Twilio system messages)"""
        try:
            if self._user_picked_up:
                return  # Already handled
            
            self._user_picked_up = True
            
            # ❌ Credit monitoring moved to _send_in_progress_status() 
            # Credit deduction will start when connected status is sent (first media packet + connected status)
            
            # Don't send in-progress status here - wait for confident word detection
            # Status will be sent in _process_transcript() when confident transcript is detected

            # Start ambient background loop after brief stabilization delay.
            asyncio.create_task(self._start_background_audio_with_delay())
            
            # 👋 Send one-time immediate greeting after pickup for inbound calls.
            if (
                self.call_session
                and self.call_session.call_type == "inbound"
                and not self._auto_greeting_sent
            ):
                self._auto_greeting_sent = True
                asyncio.create_task(
                    self.generate_and_stream_response(
                        user_text="",
                        confidence=1.0,
                        is_greeting=True,
                    )
                )
        
        except Exception as e:
            logger.error(f"Error in _handle_user_pickup: {e}", exc_info=True)
    
    async def _full_shutdown(self) -> None:
        """
        Unified, idempotent shutdown for all pipelines (STT, LLM, TTS).

        Called from every call-end path:
          - Twilio `stop` event  (handle_stop_message)
          - User goodbye phrase  (_check_and_end_call_if_goodbye)
          - Agent [END_CALL]     (_end_call_after_agent_request)
          - Voicemail detected   (_check_and_end_call_if_voicemail)
          - WebSocket finally    (route-level cleanup)

        Sets _stop_event so the main receive loop breaks out immediately
        instead of hanging at `await websocket.receive_text()`.
        """
        # Idempotent guard — first caller wins, rest are no-ops
        if self._stop_event.is_set():
            return

        self._stop_event.set()
        self._stt_active = False

        # Stop background audio loop safely.
        try:
            await self._background_audio.stop_loop()
        except Exception:
            pass

        # ── VoiceOrchestrator handles LLM cancel + TTS shutdown + STT close ────
        # OLD direct code (now delegated to orchestrator):
        # t = self._llm_response_task; t.cancel(); self._llm_response_task = None
        # self._tts_cancel.set()
        # await self._tts_pipeline.shutdown()
        # await self._stt_pipeline.aclose()
        # ────────────────────────────────────────────────────────────────────────
        try:
            await self._voice_orchestrator.shutdown()
        except Exception:
            pass

        # Finalize voice appointment booking from transcript (exactly once per call handler)
        if not self._post_call_orchestration_scheduled:
            self._post_call_orchestration_scheduled = True
            asyncio.create_task(self._post_call_appointment_workflow())

    def _post_call_appointment_sync(self) -> None:
        from app.db.session import SessionLocal
        from app.services.post_call_appointment_service import post_call_appointment_service

        db = SessionLocal()
        try:
            post_call_appointment_service.process_call_session(
                db, uuid.UUID(self.call_session_id)
            )
        except Exception as e:
            logger.error("Post-call appointment processing failed: %s", e, exc_info=True)
        finally:
            db.close()

    async def _post_call_appointment_workflow(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._post_call_appointment_sync)
        except Exception as e:
            logger.error("Post-call appointment workflow error: %s", e, exc_info=True)

    async def handle_stop_message(self, message: dict):
        """Handle Twilio stream `stop` event — delegates to unified shutdown."""
        try:
            await self._full_shutdown()
        except Exception as e:
            logger.error(f"Error in handle_stop_message: {e}", exc_info=True)


async def _receive_or_stop(
    ws: WebSocket, stop_event: asyncio.Event
) -> Optional[str]:
    """
    Race websocket.receive_text() against an internal stop_event.

    Returns:
        str  — the raw text received from Twilio.
        None — stop_event fired first (call ended internally).

    Cancels the losing task cleanly so there are no dangling coroutines.
    """
    recv_task = asyncio.create_task(ws.receive_text())
    stop_task = asyncio.create_task(stop_event.wait())
    try:
        done, pending = await asyncio.wait(
            [recv_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        # Cancel and await the loser to suppress "task was destroyed" warnings
        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        if stop_task in done:
            # Internal shutdown — tell the caller to break the loop
            recv_task.cancel()
            return None
        return recv_task.result()
    except Exception:
        return None


@router.websocket("/ws/bidirectional/{callSessionId}/{agentId}")
async def bidirectional_stream_websocket(
    websocket: WebSocket,
    callSessionId: str,
    agentId: str
):
    """
    Bidirectional WebSocket for real-time voice AI
    
    Handles:
    - Incoming audio (STT) from Twilio
    - Outgoing audio (TTS) to Twilio
    - Real-time streaming for ultra-low latency
    
    Target: <3 seconds response time
    """
    # Accept connection
    try:
        await websocket.accept()
    except Exception as e:
        logger.error(f"Failed to accept bidirectional WebSocket: {e}")
        return
    
    # Get database session
    from app.db.session import SessionLocal
    db = SessionLocal()
    
    # Create handler
    handler = BidirectionalStreamHandler(
        websocket=websocket,
        call_session_id=callSessionId,
        agent_id=agentId,
        db=db
    )
    
    media_count = 0
    
    try:
        while True:
            # Race: receive next Twilio message OR stop_event (internal call end)
            data = await _receive_or_stop(websocket, handler._stop_event)

            if data is None:
                # Internal end-call path triggered _full_shutdown + set _stop_event
                logger.info(f"🛑 Internal stop event fired for session {callSessionId} — closing WebSocket")
                break

            message = json.loads(data)
            event = message.get("event")
            
            if event == "connected":
                pass
            
            elif event == "start":
                await handler.handle_start_message(message)
            
            elif event == "media":
                await handler.handle_media_message(message)
            
            elif event == "stop":
                await handler.handle_stop_message(message)
                break
            
            elif event == "mark":
                pass  # Synchronization marks
    
    except WebSocketDisconnect:
        logger.info(f"🔌 Bidirectional WebSocket disconnected for session {callSessionId}")
    
    except Exception as e:
        logger.error(f"Unexpected error in bidirectional WebSocket: {e}", exc_info=True)
    
    finally:
        # Ensure all pipelines are fully shut down (idempotent — safe if already done)
        if handler is not None:
            try:
                await handler._full_shutdown()
            except Exception as e:
                logger.debug(f"Pipeline cleanup in finally: {e}")

        # Explicitly close the WebSocket so Twilio gets an immediate close frame
        # instead of waiting for the TCP connection to time out.
        try:
            await websocket.close()
        except Exception:
            pass

        db.close()


@router.websocket("/ws/tts-only/{callSessionId}/{agentId}")
async def tts_only_websocket(
    websocket: WebSocket,
    callSessionId: str,
    agentId: str
):
    """
    TTS-ONLY WebSocket for streaming audio playback.
    Thin composition layer that delegates to TtsOnlySession.
    """
    try:
        await websocket.accept()
    except Exception:
        return

    from app.db.session import SessionLocal

    db = SessionLocal()
    session = TtsOnlySession(
        websocket=websocket,
        call_session_id=callSessionId,
        agent_id=agentId,
        db=db,
    )

    try:
        await session.run()
    except WebSocketDisconnect:
        logger.info(f"🔌 TTS-ONLY WebSocket disconnected for session {callSessionId}")
    except Exception as e:
        logger.error(f"Unexpected error in TTS-ONLY WebSocket: {e}", exc_info=True)
    finally:
        db.close()