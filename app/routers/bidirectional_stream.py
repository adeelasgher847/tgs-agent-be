"""
Bidirectional WebSocket for Real-time Voice AI
Handles both STT (incoming audio) and TTS (outgoing audio) simultaneously
Target latency: 400–500ms (Vapi-style).

STT → LLM → TTS FLOW:
- Twilio sends audio every 20ms (MULAW 8kHz). We push each chunk to Deepgram STT.
- First qualifying ~30ms interim can start one background LLM+TTS run (one per user utterance).
- LLM streams response → each flush (sentence/time ~200ms) → TTS chunk.
- When VAD yields the final STT, we either keep that interim result (if text matches) or
  replace it with a full run using the final transcript (no duplicate stacked replies).

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
   - Waits for final transcript before responding (no partial interruptions)

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
from app.utils.tts_preprocessing import detect_emotion
from app.core.config import settings
from app.routers.general_websocket import broadcast_call_status_update
from app.utils.tts_preprocessing import preprocess_for_tts, quick_clean
from app.voice.stt_pipeline import SttPipeline
from app.voice.tts_pipeline import TtsPipeline
from app.voice.conversation_orchestrator import (
    VOICE_TUNABLES,
    ConversationOrchestrator,
    should_send_quick_ack,
)
from app.voice.rag_context import build_rag_context_block, build_rag_context_block_with_trace
from app.voice.tts_only_session import TtsOnlySession

# Import utilities and services
from app.utils.audio_utils import (
    ulaw_to_linear_sample,
    stream_mulaw_bytes_over_twilio,
    crossfade_mulaw_segments,
    build_crossfade_bridge,
    MULAW_FRAME_BYTES,
    PCM16KStreamDownsampler,
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
from app.utils.eleven_tts_background import (
    BackgroundFrameMixer,
    LinearBackgroundMixer,
    parse_eleven_background_settings,
)
from app.utils.eleven_tts_text import (
    build_elevenlabs_audio_tag_prompt_block,
    prepare_tts_text_for_provider,
    supports_elevenlabs_audio_tags,
)


router = APIRouter()


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
        # Ultra-aggressive interim processing state (40% confidence)
        self._last_interim_text = ""
        self._last_interim_sent_ts = 0.0
        self._min_interim_words = 1  # speak sooner on shorter interim
        self._min_interim_confidence = 0.40  # ULTRA-AGGRESSIVE: process 40% confidence
        self._min_interim_interval_sec = self.STT_INTERIM_INTERVAL_MS / 1000.0  # 30ms: STT stream → LLM (400–500ms target)
        
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
        # Eleven TTS background: one mixer per call; keeps loop phase across sentence chunks
        self._eleven_bg_mixer: Optional[BackgroundFrameMixer] = None
        self._eleven_bg_mixer_key: Optional[tuple] = None  # (preset_id, rounded_level) or None
        # Continuous background task — runs for the full call duration (ElevenLabs only)
        self._bg_task: Optional[asyncio.Task] = None
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
        self._min_audio_level_threshold = 100  # Minimum audio level to consider as user audio (not silence/system noise)
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

        # One response per turn (Vapi-style): when we start LLM from interim, final only commits
        self._turn_response_started = False  # True after first interim triggers LLM for this turn
        self._turn_response_seed_text = ""
        # Single in-flight async LLM+TTS per user turn (interim is non-blocking for STT reader)
        self._llm_response_task: Optional[asyncio.Task] = None
        # If interim used commit=False, agent history is held here until final matching STT
        self._pending_interim_agent_transcript: Optional[Dict[str, Any]] = None
        self._auto_greeting_sent = False
        self._recording_started = False

        # Start parallel TTS pipeline worker via TtsPipeline facade
        self._tts_pipeline = TtsPipeline(self)
        self._tts_worker_task = self._tts_pipeline._worker_task

        # Pre-cache common phrases in background for instant responses (disabled; uncomment to re-enable)
        # asyncio.create_task(self._precache_common_phrases())

        # Conversation orchestrator encapsulating LLM + policy rules
        self._conversation = ConversationOrchestrator(self)
    
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

    def _extract_greeting_from_prompt(self) -> Optional[str]:
        """
        Extract explicit greeting from prompt text, if configured.
        Supported formats:
        - GREETING: Hello and welcome ...
        - FIRST_MESSAGE: Hello and welcome ...
        - OPENING: Hello and welcome ...
        - [GREETING:Hello and welcome ...]
        """
        candidate_prompts: List[str] = []
        if self.agent and getattr(self.agent, "system_prompt", None):
            candidate_prompts.append(self.agent.system_prompt or "")
        if (
            self.agent
            and getattr(self.agent, "model", None)
            and getattr(self.agent.model, "system_prompt", None)
        ):
            candidate_prompts.append(self.agent.model.system_prompt or "")

        patterns = [
            r"(?im)^\s*(?:GREETING|FIRST_MESSAGE|OPENING)\s*:\s*(.+?)\s*$",
            r"(?is)\[\s*GREETING\s*:\s*(.+?)\s*\]",
        ]

        for prompt_text in candidate_prompts:
            if not prompt_text:
                continue
            for pattern in patterns:
                match = re.search(pattern, prompt_text)
                if not match:
                    continue
                greeting_text = (match.group(1) or "").strip().strip('"').strip("'")
                if greeting_text:
                    return greeting_text
        return None

    # ------------------------------------------------------------------
    # (Disabled) Pre-warm MULAW TTS cache for common short phrases at connect.
    # To enable: uncomment the method and asyncio.create_task(...) in __init__.
    # ------------------------------------------------------------------
    #     async def _precache_common_phrases(self):
    #         """Pre-generate and cache common phrases for instant playback."""
    #         try:
    #             common_phrases = [
    #                 "Hello", "Hi there", "Hi", "Good morning", "Good afternoon", "Good evening",
    #                 "Got it", "I see", "Okay", "Sure", "Alright", "Perfect", "Great", "Understood",
    #                 "Yes", "No", "Absolutely", "Of course",
    #                 "Let me check that", "One moment please", "Just a second", "Let me see",
    #                 "Thank you", "Thanks", "You're welcome",
    #                 "Goodbye", "Have a great day", "Thank you for calling", "Talk to you later",
    #             ]
    #             lang = self.agent.language if self.agent and self.agent.language else "en"
    #             voice = self.agent.voice_type if self.agent and self.agent.voice_type else "female"
    #             for phrase in common_phrases:
    #                 try:
    #                     await generate_mulaw_tts(
    #                         text=phrase, lang=lang, voice=voice, use_chirp3_hd=True,
    #                         speaking_rate=1.0, use_ssml=False, agent=self.agent,
    #                     )
    #                 except Exception:
    #                     continue
    #         except Exception as e:
    #             logger.error(f"Error in precache_common_phrases: {e}")
    
    async def handle_media_message(self, message: dict):
        """Handle incoming audio from Twilio and feed to Deepgram streaming STT"""
        try:
            import time
            
            media = message.get("media", {})
            payload = media.get("payload")
            
            if not payload:
                return
            
            # Decode audio (MULAW from Twilio)
            audio_data = base64.b64decode(payload)
            
            # ✅ DETECT ACTUAL USER AUDIO (not Twilio system messages/music)
            if not self._first_media_received:
                self._first_media_received = True
            
            # Calculate audio level (RMS) to detect actual user audio vs silence/system noise
            if not self._user_picked_up:
                # Convert MULAW to linear and calculate RMS (Root Mean Square) audio level
                audio_level = 0
                if len(audio_data) > 0:
                    linear_samples = [ulaw_to_linear_sample(b) for b in audio_data]
                    # Calculate RMS
                    sum_squares = sum(s * s for s in linear_samples)
                    audio_level = int((sum_squares / len(linear_samples)) ** 0.5)
                
                # Track audio levels
                self._audio_level_samples.append(audio_level)
                if len(self._audio_level_samples) > self._audio_samples_needed * 2:
                    self._audio_level_samples.pop(0)  # Keep last 20 samples
                
                # Check if we have enough samples and enough non-silent audio (actual user audio)
                if len(self._audio_level_samples) >= self._audio_samples_needed:
                    non_silent_count = sum(1 for level in self._audio_level_samples[-self._audio_samples_needed:] if level > self._min_audio_level_threshold)
                    
                    if non_silent_count >= self._audio_samples_needed:
                        # Actual user audio detected! Set in-progress status
                        await self._handle_user_pickup()  # User actually picked up with real audio!
                        
                        # Skip first few seconds for STT (system messages might still be there)
                        self._skip_audio_until = time.time() + 3.0
                else:
                    return  # Don't process until we have actual user audio
            
            # Skip audio if still in grace period (system messages)
            if self._skip_audio_until and time.time() < self._skip_audio_until:
                return  # Don't send to STT - this is likely system message/ringing

            # Guard: stop feeding after call ends (e.g. Twilio keeps sending silence frames)
            if not self._stt_active:
                return

            # (Removed first-media DB marker for outbound gating)
            # Lazily create STT pipeline and push audio
            if self._stt_pipeline is None:
                language_code = (settings.DEEPGRAM_STT_LANGUAGE or "en").strip()
                self._stt_pipeline = SttPipeline(
                    language_code=language_code,
                    on_interim=self._maybe_process_interim,
                    on_final=self._process_transcript,
                    call_session_id=self.call_session_id,
                    agent_id=self.agent_id,
                )

            await self._stt_pipeline.feed_audio_chunk(audio_data)
        
        except Exception as e:
            logger.error(f"Error handling media message: {e}", exc_info=True)
    
    # Removed chunk-based STT processing; relying on Deepgram streaming endpointing

    async def _cancel_inflight_llm_response(self) -> None:
        """Stop background LLM+TTS for this turn (barge-in or final regen)."""
        self._pending_interim_agent_transcript = None
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

    async def _commit_pending_interim_agent_transcript(self) -> None:
        """Persist agent text from an interim (commit=False) run once the final STT matches."""
        p = self._pending_interim_agent_transcript
        if not p:
            return
        self._pending_interim_agent_transcript = None
        text = (p.get("text") or "").strip()
        if not text:
            return
        await self._add_to_transcript(
            "agent",
            text,
            "agent_response",
            message_metadata=p.get("metadata") or {},
        )

    async def _complete_llm_turn_after_stt_final(self, transcript: str, confidence: float) -> None:
        """
        Run after the user's final message is in the DB. Picks the correct LLM path:
        - Interim+final: regenerate if final text differs, else await interim task and commit pending history.
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
                    commit_agent_transcript=True,
                )
            else:
                if self._llm_response_task and not self._llm_response_task.done():
                    try:
                        await self._llm_response_task
                    except asyncio.CancelledError:
                        pass
                self._llm_response_task = None
                await self._commit_pending_interim_agent_transcript()
            return

        self._turn_response_seed_text = ""
        self._last_interim_text = ""
        self._tts_cancel.clear()
        await self.generate_and_stream_response(
            transcript,
            confidence,
            is_greeting=False,
            commit_agent_transcript=True,
        )
    
    async def _process_transcript(self, transcript: str, confidence: float):
        """Process a transcript (final result)"""
        try:
            if not transcript or confidence < 0.3:
                return
            
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

            await self._complete_llm_turn_after_stt_final(transcript, confidence)
            
        except Exception as e:
            logger.error(f"Error processing transcript: {e}", exc_info=True)

    async def _maybe_process_interim(self, transcript: str, confidence: float):
        """
        Start at most ONE early LLM+TTS run per user utterance (low latency).
        Does not block the STT reader (background task) so the final transcript can be processed
        while the model streams. Barge-in cancels the in-flight work.
        """
        try:
            if not transcript:
                return

            word_count = len(transcript.split())

            # Barge-in: any user words while the agent is speaking
            if self._tts_pipeline and self._tts_pipeline.is_speaking and word_count >= 1:
                await self._cancel_inflight_llm_response()
                self._turn_response_started = False
                self._turn_response_seed_text = ""
                self._last_interim_text = ""
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
                try:
                    await self.generate_and_stream_response(
                        transcript,
                        confidence,
                        is_greeting=False,
                        commit_agent_transcript=False,
                    )
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error in interim LLM task: {e}", exc_info=True)

            if self._llm_response_task and not self._llm_response_task.done():
                self._llm_response_task.cancel()
            self._llm_response_task = asyncio.create_task(_run_interim())
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
        commit_agent_transcript: bool = True,
    ):
        """
        Generate AI response and stream TTS in real-time WITH conversation history.
        Uses PARALLEL TTS PIPELINE (Vapi-style) for ultra-low latency.

        Args:
            user_text: User's input text (empty for greeting)
            confidence: STT confidence score
            is_greeting: If True, uses agent's first_message instead of calling LLM
            commit_agent_transcript: If False (interim-only run), do not write agent to DB
                until a matching final; store pending for later commit.
        """
        try:
            from datetime import datetime, timezone
            import json
            
            # 👋 HANDLE AUTO-GREETING - Skip LLM, use pre-defined greeting
            if is_greeting:
                # Get greeting from agent or use default
                if self.agent and hasattr(self.agent, 'first_message') and self.agent.first_message:
                    greeting_text = self.agent.first_message
                else:
                    prompt_greeting = self._extract_greeting_from_prompt()
                    if prompt_greeting:
                        greeting_text = prompt_greeting
                    elif self.call_session and self.call_session.call_type == "inbound":
                        greeting_text = "Thank you for calling. How may I assist you today?"
                    else:
                        greeting_text = "hello how are you"
                
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

            if commit_agent_transcript:
                self._pending_interim_agent_transcript = None
            
            # Send quick acknowledgement for longer queries (instant from cache!)
            await self._send_quick_acknowledgement(user_text)

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

            inbound_prompt_context_block = ""
            inbound_kb_docs_context_block = ""
            if self.agent and self.agent.is_inbound_agent and tenant_uuid and agent_uuid:
                try:
                    inbound_prompt_context_block = (
                        agent_service.build_inbound_prompt_context_block(
                            db=self.db,
                            inbound_agent_id=agent_uuid,
                            tenant_id=tenant_uuid,
                        )
                    )
                    inbound_kb_docs_context_block = (
                        agent_service.build_inbound_kb_documents_context_block(
                            db=self.db,
                            inbound_agent_id=agent_uuid,
                            tenant_id=tenant_uuid,
                        )
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to build inbound context blocks for agent %s: %s",
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
            tts_provider = getattr(self.agent, "tts_provider", None) if self.agent else None
            tts_provider_slug = (getattr(tts_provider, "slug", None) or "").lower()
            elevenlabs_audio_tags_enabled = supports_elevenlabs_audio_tags(tts_provider_slug)
            output_plain_text_rule = (
                "- OUTPUT PLAIN TEXT ONLY: Do NOT output SSML or XML. "
                "Sparse ElevenLabs bracketed audio tags like [breathes] are allowed when natural."
                if elevenlabs_audio_tags_enabled
                else "- OUTPUT PLAIN TEXT ONLY: Do NOT output SSML, XML, or any tags. Prosody is handled by the system."
            )
            no_ssml_rule_base = (
                "4. NO SSML: Do NOT output <speak>, <prosody>, or any XML tags. Plain text only. "
                "Sparse ElevenLabs bracketed audio tags like [breathes] are allowed when natural."
                if elevenlabs_audio_tags_enabled
                else "4. NO SSML: Do NOT output <speak>, <prosody>, or any XML tags. Plain text only."
            )
            no_ssml_rule = (
                "3. NO SSML: Plain text only. No <speak>, <prosody>, or XML. "
                "Sparse ElevenLabs bracketed audio tags like [breathes] are allowed when natural."
                if elevenlabs_audio_tags_enabled
                else "3. NO SSML: Plain text only. No <speak>, <prosody>, or XML."
            )
            elevenlabs_audio_tag_block = build_elevenlabs_audio_tag_prompt_block(tts_provider_slug)
            
            # Base prompt for phone conversations (voice-first, plain text only, no SSML)
            base_prompt = f"""# ROLE
You are {agent_name}, having a real-time phone call with a human.

# STYLE & TONE
- VOICE-FIRST: Your output is for Text-to-Speech. Use short, punchy sentences.
- NATURAL: Use natural fillers/interjections ONLY when they fit the emotion: "umm", "hmm", "oh", "alright", "hang on", "one moment" (max one per response).
- CONCISE: Max 20 words per response unless explaining something complex.
- NO ROBOT TALK: Avoid "As an AI" or formal greetings. Use "Hey," "Hi," or "Hello."
{output_plain_text_rule}
- TEXT HYGIENE: Avoid "..." (use a comma or short sentence). Avoid slashes like "FastAPI/ML" (say "FastAPI and ML").

# CONVERSATION STATE
Previous conversation:
{history_text}

{booking_memory_block}
{rag_context_block}
{inbound_prompt_context_block}
{inbound_kb_docs_context_block}

# CRITICAL RULES
1. NO REPETITION: If the history shows you asked a question, move to the next point.
2. HANDLING SILENCE: If the user says something vague, ask a clarifying question.
3. TERMINATION: When the objective is met, say a friendly goodbye and end your response with exactly [END_CALL].
{no_ssml_rule_base}

{elevenlabs_audio_tag_block}

# APPOINTMENT BOOKING
- If user wants to book/schedule an appointment: collect their name, phone number, reason, preferred date/time, and ask for email as optional for confirmations.
- If the user declines or does not provide email, continue booking without email (do not block scheduling).
- To check available slots emit exactly: [CHECK_SLOTS:date=YYYY-MM-DD] (use "tomorrow" or ISO date).
- Once user confirms a slot emit exactly: [BOOK_APPOINTMENT:name=<name>,phone=<phone>,email=<email if the user provided one; otherwise omit the email= field entirely>,slot=<exact offered ISO datetime or spoken slot label>,reason=<reason>]
- CRITICAL UX: Do NOT say "appointment confirmed/scheduled/booked" yourself. Emit the booking token and wait; backend will send final confirmation after actual DB success.
- CALENDAR TOKENS (CRITICAL): [CHECK_SLOTS:...] and [BOOK_APPOINTMENT:...] must be valid for the system to run. Put each token on ONE line. Always end with a closing ] — never omit it, truncate, wrap, or split across lines. Field order must be: name, phone, optional email, slot, reason. Example with email: [BOOK_APPOINTMENT:name=John Smith,phone=+15551234567,email=john@example.com,slot=2026-04-08T10:30:00,reason=Dental checkup]. Example without email: [BOOK_APPOINTMENT:name=John Smith,phone=+15551234567,slot=2026-04-08T10:30:00,reason=Dental checkup]
- Use a short reason with NO commas inside reason= (commas break parsing).
- If they already booked on this call and want a different time: offer [CHECK_SLOTS:...] again, then emit the same [BOOK_APPOINTMENT:...] with the new slot; the system reschedules automatically.
- Only book one of the slots that was just offered by the system.
- Never book a slot that is in the past (check CURRENT DATE & TIME above).
- Speak naturally; the system handles the actual booking silently.

# GOAL
Continue the conversation based on the history above. Be {agent_name}."""
            
            # Use agent's custom system prompt if available, otherwise use base prompt
            if self.agent and self.agent.system_prompt:
                # Agent has custom system prompt - use it with context (voice-first, plain text)
                system_prompt = f"""# ROLE
You are {agent_name}, having a real-time phone call. You speak {agent_language} naturally.

# CUSTOM INSTRUCTIONS
{self.agent.system_prompt}

# STYLE & TONE
- VOICE-FIRST: Output is for Text-to-Speech. Use short sentences (max 20 words unless explaining).
- NATURAL: Use natural fillers/interjections ONLY when they fit the emotion: "umm", "hmm", "oh", "alright", "hang on", "one moment" (max one per response).
{output_plain_text_rule}
- TEXT HYGIENE: Avoid "..." (use a comma or short sentence). Avoid slashes like "FastAPI/ML" (say "FastAPI and ML").

# CONVERSATION STATE
Previous conversation:
{history_text}

{booking_memory_block}
{rag_context_block}
{inbound_prompt_context_block}
{inbound_kb_docs_context_block}

# CRITICAL RULES
1. NO REPETITION: Do not repeat questions already asked. Move to the next point.
2. TERMINATION: When all objectives from your custom instructions are complete, say a friendly goodbye and end your response with exactly [END_CALL].
{no_ssml_rule}

{elevenlabs_audio_tag_block}

# APPOINTMENT BOOKING
- If user wants to book/schedule an appointment: collect their name, phone number, reason, preferred date/time, and ask for email as optional for confirmations.
- If the user declines or does not provide email, continue booking without email (do not block scheduling).
- To check available slots emit exactly: [CHECK_SLOTS:date=YYYY-MM-DD]
- Once user confirms a slot emit exactly: [BOOK_APPOINTMENT:name=<name>,phone=<phone>,email=<email if the user provided one; otherwise omit the email= field entirely>,slot=<exact offered ISO datetime or spoken slot label>,reason=<reason>]
- CRITICAL UX: Do NOT say "appointment confirmed/scheduled/booked" yourself. Emit the booking token and wait; backend will send final confirmation after actual DB success.
- CALENDAR TOKENS (CRITICAL): [CHECK_SLOTS:...] and [BOOK_APPOINTMENT:...] must be valid for the system to run. Put each token on ONE line. Always end with a closing ] — never omit it, truncate, wrap, or split across lines. Field order must be: name, phone, optional email, slot, reason. Example with email: [BOOK_APPOINTMENT:name=John Smith,phone=+15551234567,email=john@example.com,slot=2026-04-08T10:30:00,reason=Dental checkup]. Example without email: [BOOK_APPOINTMENT:name=John Smith,phone=+15551234567,slot=2026-04-08T10:30:00,reason=Dental checkup]
- Use a short reason with NO commas inside reason= (commas break parsing).
- If they already booked on this call and want a different time: run [CHECK_SLOTS:...] again, then the same [BOOK_APPOINTMENT:...] with the new slot; the system reschedules automatically.
- Only book one of the slots that was just offered by the system.
- Never book a slot in the past (see CURRENT DATE & TIME).

# GOAL
Follow your custom instructions. Continue from the history above. Be {agent_name}."""
            elif self.agent and self.agent.model and self.agent.model.system_prompt:
                # Model has system prompt - use it (voice-first, plain text)
                system_prompt = f"""# ROLE
You are {agent_name}, having a real-time phone call. You speak {agent_language} naturally.

# MODEL INSTRUCTIONS
{self.agent.model.system_prompt}

# STYLE & TONE
- VOICE-FIRST: Output is for Text-to-Speech. Use short sentences (max 20 words unless explaining).
- NATURAL: Use fillers like "uhm," "well," "I see" occasionally.
{output_plain_text_rule}

# CONVERSATION STATE
Previous conversation:
{history_text}

{booking_memory_block}
{rag_context_block}
{inbound_prompt_context_block}
{inbound_kb_docs_context_block}

# CRITICAL RULES
1. NO REPETITION: Do not repeat questions. Move to the next point.
2. TERMINATION: When all objectives are complete, say a friendly goodbye and end your response with exactly [END_CALL].
{no_ssml_rule}

{elevenlabs_audio_tag_block}

# APPOINTMENT BOOKING
- If user wants to book/schedule an appointment: collect their name, phone number, reason, preferred date/time, and ask for email as optional for confirmations.
- If the user declines or does not provide email, continue booking without email (do not block scheduling).
- To check available slots emit exactly: [CHECK_SLOTS:date=YYYY-MM-DD]
- Once user confirms a slot emit exactly: [BOOK_APPOINTMENT:name=<name>,phone=<phone>,email=<email if the user provided one; otherwise omit the email= field entirely>,slot=<exact offered ISO datetime or spoken slot label>,reason=<reason>]
- CRITICAL UX: Do NOT say "appointment confirmed/scheduled/booked" yourself. Emit the booking token and wait; backend will send final confirmation after actual DB success.
- CALENDAR TOKENS (CRITICAL): [CHECK_SLOTS:...] and [BOOK_APPOINTMENT:...] must be valid for the system to run. Put each token on ONE line. Always end with a closing ] — never omit it, truncate, wrap, or split across lines. Field order must be: name, phone, optional email, slot, reason. Example with email: [BOOK_APPOINTMENT:name=John Smith,phone=+15551234567,email=john@example.com,slot=2026-04-08T10:30:00,reason=Dental checkup]. Example without email: [BOOK_APPOINTMENT:name=John Smith,phone=+15551234567,slot=2026-04-08T10:30:00,reason=Dental checkup]
- Use a short reason with NO commas inside reason= (commas break parsing).
- If they already booked on this call and want a different time: run [CHECK_SLOTS:...] again, then the same [BOOK_APPOINTMENT:...] with the new slot; the system reschedules automatically.
- Only book one of the slots that was just offered by the system.
- Never book a slot in the past (see CURRENT DATE & TIME).

# GOAL
Follow the model instructions. Continue from the history above. Be {agent_name}."""
            else:
                # Use base prompt
                system_prompt = base_prompt

            # Prepend current date/time so the agent knows what "today", "tomorrow",
            # and "past slots" mean. Also injected into appointment booking flow.
            _now_local = datetime.now(timezone.utc)
            _now_str = _now_local.strftime("%A, %B %d, %Y at %I:%M %p UTC")
            system_prompt = (
                f"# CURRENT DATE & TIME\nNow: {_now_str}\n\n"
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

                def _strip_control_tokens(text: str) -> str:
                    # These are backend/system markers and must NEVER be spoken.
                    if not text:
                        return ""
                    text = text.replace("[END_CALL]", "")
                    text = re.sub(r"\[OUTCOME:[^\]]+\]", "", text)
                    text = re.sub(r"\[CHECK_SLOTS:[^\]]*\]", "", text)
                    text = re.sub(r"\[BOOK_APPOINTMENT:[^\]]*\]", "", text)
                    # Tolerate malformed tokens that miss a closing bracket.
                    text = re.sub(r"\[(?:OUTCOME|CHECK_SLOTS|BOOK_APPOINTMENT):[^\]\n\r]*", "", text)
                    return text

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
                        tts_buffer = _strip_control_tokens(tts_buffer)

                    # Remove OUTCOME tokens from any in-flight buffer (never spoken)
                    if "[OUTCOME:" in tts_buffer:
                        tts_buffer = _strip_control_tokens(tts_buffer)

                    # Avoid spoken "final confirmation" before backend booking succeeds.
                    if "[BOOK_APPOINTMENT:" in response_accum:
                        tts_buffer = self._strip_premature_booking_confirmation(_strip_control_tokens(tts_buffer))

                    # Flush complete thoughts early for faster perceived latency
                    flush_idx = _find_flush_index(tts_buffer)
                    # If punctuation-based flush isn't available, do a time-based flush (~200ms for 400–500ms latency)
                    if flush_idx is None:
                        now_ts = time.perf_counter()
                        if (now_ts - last_flush_ts) >= 0.20:
                            flush_idx = _find_time_flush_index(tts_buffer)
                    if flush_idx is not None and not self._tts_cancel.is_set():
                        flush_text = tts_buffer[:flush_idx].strip()
                        tts_buffer = tts_buffer[flush_idx:].lstrip()

                        flush_text = _strip_control_tokens(flush_text).strip()
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

                            chunk_counter += 1
                            if self._tts_pipeline:
                                await self._tts_pipeline.queue_tts({
                                "text": enhanced_text,
                                "chunk_id": chunk_counter,
                                "use_ssml": self._use_ssml,
                                "is_final": False,
                            })
                            first_tts_chunk = False
                            last_flush_ts = time.perf_counter()

                # Flush any remaining text as the FINAL chunk
                full_response = response_accum.strip()
                end_call_after = end_call_after or ("[END_CALL]" in full_response)

                # Never speak control tokens
                tts_buffer = _strip_control_tokens(tts_buffer).strip()

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

                    chunk_counter += 1
                    if self._tts_pipeline:
                        await self._tts_pipeline.queue_tts({
                        "text": enhanced_text,
                        "chunk_id": chunk_counter,
                        "use_ssml": self._use_ssml,
                        "is_final": True,
                        "end_call_after": end_call_after,
                    })

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
                            safe_tts_text = re.sub(r"\[OUTCOME:[^\]]+\]", "", final_text)
                            safe_tts_text = re.sub(r"\[CHECK_SLOTS:[^\]]*\]", "", safe_tts_text)
                            safe_tts_text = re.sub(r"\[BOOK_APPOINTMENT:[^\]]*\]", "", safe_tts_text)
                            safe_tts_text = re.sub(
                                r"\[(?:OUTCOME|CHECK_SLOTS|BOOK_APPOINTMENT):[^\]\n\r]*",
                                "",
                                safe_tts_text,
                            ).replace("[END_CALL]", "").strip()
                            chunk_counter += 1
                            if self._tts_pipeline:
                                await self._tts_pipeline.queue_tts({
                                    "text": safe_tts_text or final_text,
                                    "use_ssml": self._use_ssml,
                                    "is_final": True,
                                })
                    except Exception as e:
                        logger.warning(f"⚠️ VoiceLoggingService fallback failed: {e}. Using ultimate fallback.")
                        # Ultimate fallback response
                        final_text = "I apologize, I'm having trouble responding right now. Could you please repeat that?"
                        chunk_counter += 1
                        if self._tts_pipeline:
                            await self._tts_pipeline.queue_tts({
                            "text": final_text,
                            "chunk_id": chunk_counter,
                            "use_ssml": self._use_ssml,
                            "is_final": True
                        })

            if final_text:
                # Two-step reliability: if booking intent exists but token is missing, run action extraction.
                if commit_agent_transcript and booking_intent_turn and not self._has_calendar_token(final_text):
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
                transcript_text = re.sub(r"\[CHECK_SLOTS:[^\]]*\]", "", final_text)
                transcript_text = re.sub(r"\[BOOK_APPOINTMENT:[^\]]*\]", "", transcript_text)
                transcript_text = transcript_text.replace("[END_CALL]", "").strip()
                if "[BOOK_APPOINTMENT:" in final_text:
                    transcript_text = self._strip_premature_booking_confirmation(transcript_text)
                if transcript_text:
                    if commit_agent_transcript:
                        await self._add_to_transcript(
                            "agent",
                            transcript_text,
                            "agent_response",
                            message_metadata={
                                "user_text": user_text,
                                "rag_trace": rag_trace,
                            },
                        )
                    else:
                        self._pending_interim_agent_transcript = {
                            "text": transcript_text,
                            "metadata": {
                                "user_text": user_text,
                                "rag_trace": rag_trace,
                            },
                        }

                # Handle calendar tokens (fire-and-forget after TTS is already queued)
                if commit_agent_transcript:
                    if re.search(r"\[\s*CHECK_SLOTS\s*:", final_text, flags=re.IGNORECASE):
                        asyncio.create_task(self._handle_check_slots_token(final_text))
                    elif re.search(r"\[\s*BOOK_APPOINTMENT\s*:", final_text, flags=re.IGNORECASE):
                        asyncio.create_task(self._handle_book_appointment_token(final_text))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error in generate_and_stream_response: {e}", exc_info=True)
    
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

    def _should_regenerate_on_final(self, final_transcript: str) -> bool:
        """
        If an interim run used partial STT, decide whether a final run with full text
        is needed. Always regenerate when final normalized text differs from seed, plus
        booking slot / correction heuristics.
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
            # Text matches; only regenerate if a resolved calendar slot changed (booking)
            if not self._is_booking_context_active(final_transcript):
                return False
            final_slot = self._resolve_cached_calendar_slot(final_transcript)
            seed_slot = self._resolve_cached_calendar_slot(self._turn_response_seed_text)
            if final_slot and seed_slot and final_slot != seed_slot:
                return True
            return False

        if self._is_booking_context_active(final_transcript) or self._is_booking_context_active(
            self._turn_response_seed_text
        ):
            final_slot = self._resolve_cached_calendar_slot(final_transcript)
            seed_slot = self._resolve_cached_calendar_slot(self._turn_response_seed_text)
            if final_slot and seed_slot and final_slot != seed_slot:
                return True
            if final_norm.startswith(seed_norm) and len(final_norm) >= len(seed_norm) + 3:
                return True
            correction_markers = ("wrong", "no no", "not ", "already", "spell", "11 00", "11 am")
            if any(marker in final_norm for marker in correction_markers):
                return True

        # General conversation: final STT differs from what the interim used
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
            "You extract calendar actions from a phone-call turn.\n"
            "Return exactly one line and nothing else:\n"
            "- [BOOK_APPOINTMENT:name=<name>,phone=<phone>,slot=<slot>,reason=<reason>] "
            "(if the user gave an email, use: name=...,phone=...,email=...,slot=...,reason=...)\n"
            "- [CHECK_SLOTS:date=YYYY-MM-DD]\n"
            "- NONE\n"
            "Rules:\n"
            "1) If user selected a concrete slot that was offered, return BOOK_APPOINTMENT.\n"
            "2) If user asked to check availability, return CHECK_SLOTS.\n"
            "3) If uncertain or missing critical fields, return NONE.\n"
            "4) Keep reason short and without commas.\n"
            "5) Include email= only if the user clearly gave an address; field order is name, phone, optional email, slot, reason.\n"
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

    async def _repair_customer_email_with_llm(
        self,
        *,
        token_email_raw: Optional[str],
        transcript_client_lines_newest_first: list[str],
    ) -> Optional[str]:
        """
        Best-effort repair layer for noisy spoken emails.
        Returns a validated email or None. Never raises.
        """
        from app.utils.spoken_email import (
            build_email_repair_prompt,
            coerce_email_from_text,
            normalize_stored_email,
        )

        prompt = build_email_repair_prompt(
            token_email_raw=token_email_raw,
            transcript_client_lines_newest_first=transcript_client_lines_newest_first,
        )
        repair_system_prompt = (
            "You repair customer email addresses from noisy call transcripts. "
            "Prefer spelled-out transcript evidence over fused STT literals. "
            "If uncertain, return NONE. Output only the email or NONE."
        )
        loop = asyncio.get_running_loop()

        async def _call_repair(service, model_name: str) -> Optional[str]:
            def _run():
                return service.generate_text(
                    prompt=prompt,
                    system_prompt=repair_system_prompt,
                    model_name=model_name,
                    temperature=0.0,
                    max_tokens=60,
                )

            try:
                payload = await loop.run_in_executor(None, _run)
            except Exception as e:
                logger.warning(
                    "Email repair LLM failed via %s/%s: %s",
                    service.__class__.__name__,
                    model_name,
                    e,
                )
                return None

            content = (payload.get("content") or "").strip()
            if not content or content.upper() == "NONE":
                return None
            return normalize_stored_email(content) or coerce_email_from_text(content)

        if settings.GEMINI_API_KEY:
            repaired = await _call_repair(gemini_service, "gemini-1.5-flash")
            if repaired:
                return repaired

        if settings.OPENAI_API_KEY:
            repaired = await _call_repair(openai_service, "gpt-4o-mini")
            if repaired:
                return repaired

        return None

    def _merge_pending_email_note(
        self,
        existing_notes: Optional[str],
        *,
        pending_email: Optional[str],
        source: str,
        reason: str,
    ) -> Optional[str]:
        if not pending_email:
            return existing_notes

        entry = (
            f"Pending email verification: {pending_email} "
            f"(source: {source}; reason: {reason})"
        )
        base = (existing_notes or "").strip()
        if pending_email in base and "Pending email verification:" in base:
            return existing_notes
        if not base:
            return entry
        return f"{base}\n{entry}"

    async def _handle_book_appointment_token(self, llm_response: str):
        """
        Called when LLM emits [BOOK_APPOINTMENT:name=...,phone=...,optional email=...,slot=...,reason=...].
        Resolves customer_email from the token and/or recent client transcript (spoken-email STT recovery).
        Books the appointment and speaks the confirmation directly via TTS.
        """
        try:
            import re as _re
            from dataclasses import replace
            from datetime import datetime as _dt

            from app.services.calendar_service import calendar_service as _cal
            from app.utils.spoken_email import resolve_customer_email_for_booking

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

            token_email_raw: str | None = None
            # Robust parse: name, phone, optional email, slot, optional reason (commas in reason).
            strict = _re.search(
                r"name=(?P<name>.*?),\s*phone=(?P<phone>.*?),\s*(?:email=(?P<email>.*?),\s*)?"
                r"slot=(?P<slot>.*?)(?:,\s*reason=(?P<reason>.*))?$",
                raw_single_line,
            )
            if strict:
                customer_name = (strict.group("name") or "").strip()
                customer_phone = (strict.group("phone") or "").strip()
                token_email_raw = (strict.group("email") or "").strip() or None
                slot_raw = (strict.group("slot") or "").strip()
                reason_val = (strict.group("reason") or "").strip()
                reason = reason_val or None
            else:
                # Backward-compatible fallback for legacy/messy token shapes.
                def _get(key: str) -> str:
                    km = _re.search(rf"{key}=([^,\]]+)", raw_single_line)
                    return km.group(1).strip() if km else ""

                customer_name = _get("name")
                customer_phone = _get("phone")
                token_email_raw = _get("email") or None
                slot_raw = _get("slot")
                reason = _get("reason") or None

            if not customer_name or not customer_phone or not slot_raw:
                logger.warning("BOOK_APPOINTMENT token missing required fields: %s", raw_single_line[:500])
                return

            if not self.call_session:
                return

            transcript_lines = self._client_transcript_lines_newest_first()
            email_resolution = resolve_customer_email_for_booking(
                token_email_raw=token_email_raw,
                transcript_client_lines_newest_first=transcript_lines,
            )
            if (
                email_resolution.should_attempt_llm_repair
                and not email_resolution.verified_email
            ):
                repaired_email = await self._repair_customer_email_with_llm(
                    token_email_raw=token_email_raw,
                    transcript_client_lines_newest_first=transcript_lines,
                )
                if repaired_email and repaired_email != email_resolution.pending_email:
                    email_resolution = replace(
                        email_resolution,
                        pending_email=repaired_email,
                        source="llm_repaired_unverified",
                        trust_score=max(email_resolution.trust_score, 60),
                        should_attempt_llm_repair=False,
                        reason=(
                            f"{email_resolution.reason} "
                            "LLM repair produced a validated candidate; still pending confirmation."
                        ).strip(),
                    )

            verified_customer_email = email_resolution.verified_email
            # Demo override: store pending candidate when verified email isn't available.
            customer_email_for_storage = (
                verified_customer_email or email_resolution.pending_email
            )
            if verified_customer_email:
                logger.info(
                    "BOOK_APPOINTMENT: using verified customer_email source=%s trust=%s",
                    email_resolution.source,
                    email_resolution.trust_score,
                )
            elif email_resolution.pending_email:
                logger.info(
                    "BOOK_APPOINTMENT: keeping email pending verification source=%s trust=%s candidate=%s",
                    email_resolution.source,
                    email_resolution.trust_score,
                    email_resolution.pending_email,
                )

            slot_start = self._resolve_cached_calendar_slot(slot_raw)
            if slot_start is None:
                try:
                    slot_start = _dt.fromisoformat(slot_raw.replace("Z", "+00:00"))
                except ValueError:
                    logger.warning("BOOK_APPOINTMENT: invalid slot datetime: %s", slot_raw)
                    return

            tenant_id = self.call_session.tenant_id
            agent_id = self.agent.id if self.agent else None
            call_session_id = self.call_session.id

            loop = asyncio.get_running_loop()

            def _existing_for_call():
                return _cal.get_active_appointment_for_call_session(
                    self.db, tenant_id, call_session_id
                )

            existing = await loop.run_in_executor(None, _existing_for_call)
            merged_notes = self._merge_pending_email_note(
                existing.notes if existing else None,
                pending_email=email_resolution.pending_email,
                source=email_resolution.source,
                reason=email_resolution.reason,
            )

            try:
                if existing:
                    appt = await loop.run_in_executor(
                        None,
                        lambda: _cal.reschedule_appointment(
                            db=self.db,
                            tenant_id=tenant_id,
                            appointment_id=existing.id,
                            slot_start=slot_start,
                            customer_name=customer_name,
                            customer_phone=customer_phone,
                            appointment_reason=reason,
                            customer_email=customer_email_for_storage,
                            notes=merged_notes,
                        ),
                    )
                    msg = (
                        f"All set! I've moved your appointment to "
                        f"{appt.slot_start.strftime('%A, %B %d at %I:%M %p')}. "
                        f"Anything else I can help with?"
                    )
                    self._last_selected_calendar_slot = appt.slot_start
                else:
                    appt = await loop.run_in_executor(
                        None,
                        lambda: _cal.book_appointment(
                            db=self.db,
                            tenant_id=tenant_id,
                            customer_name=customer_name,
                            customer_phone=customer_phone,
                            slot_start=slot_start,
                            agent_id=agent_id,
                            call_session_id=call_session_id,
                            appointment_reason=reason,
                            customer_email=customer_email_for_storage,
                            notes=merged_notes,
                            created_via="voice_agent",
                        ),
                    )
                    msg = (
                        f"Done! Your appointment is confirmed for "
                        f"{appt.slot_start.strftime('%A, %B %d at %I:%M %p')}. "
                        f"Is there anything else I can help you with?"
                    )
                    self._last_selected_calendar_slot = appt.slot_start
                if email_resolution.pending_email and not verified_customer_email:
                    msg += (
                        " I captured an email candidate, but it will stay pending "
                        "verification until it's confirmed."
                    )
            except ValueError as ve:
                msg = f"{ve} Would you like to choose a different time?"

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

    def _get_or_create_eleven_bg_mixer(self) -> Optional[BackgroundFrameMixer]:
        """
        Reuse a single BackgroundFrameMixer for the whole call so the bed does not
        reset phase on every TTS chunk (sounds like continuous ambience).
        Recreates if preset or level in agent settings changes, or clears when disabled.
        """
        p = getattr(self.agent, "tts_provider", None) if self.agent else None
        slug = (getattr(p, "slug", None) or "").lower()
        if slug != "elevenlabs" or not self.agent:
            self._eleven_bg_mixer = None
            self._eleven_bg_mixer_key = None
            return None
        st = dict(getattr(self.agent, "tts_settings_json", None) or {})
        bid, blvl = parse_eleven_background_settings(st)
        if not bid:
            self._eleven_bg_mixer = None
            self._eleven_bg_mixer_key = None
            return None
        key = (bid, round(blvl, 3))
        if self._eleven_bg_mixer is not None and self._eleven_bg_mixer_key == key:
            return self._eleven_bg_mixer
        self._eleven_bg_mixer = BackgroundFrameMixer(bid, blvl)
        self._eleven_bg_mixer_key = key
        return self._eleven_bg_mixer

    async def _run_continuous_background(self) -> None:
        """
        Vapi-style: stream background audio frames for the entire call duration.

        - During TTS (is_speaking=True): TTS sender already mixes background via
          _voice_frame_mulaw — this loop skips to avoid doubling the frame rate.
        - During silence / STT (is_speaking=False): injects one background frame
          every 20 ms so the caller hears a continuous ambient bed.

        Uses perf_counter drift correction so background stays precisely at
        50 fps (20 ms/frame) even when the asyncio event loop is busy during
        heavy STT processing.
        """
        import base64 as _b64
        import time as _time
        from app.utils.audio_utils import MULAW_FRAME_BYTES as _FRM

        SILENT_MULAW = bytes([0xFF]) * _FRM  # mu-law silence carrier
        FRAME_INTERVAL = 0.02  # 20 ms per telephony frame

        next_send = _time.perf_counter() + FRAME_INTERVAL

        while not self._stop_event.is_set():
            # Drift-corrected sleep: sleep only the remaining time until next frame
            now = _time.perf_counter()
            sleep_dur = next_send - now
            if sleep_dur > 0:
                await asyncio.sleep(sleep_dur)
            elif sleep_dur < -0.10:
                # More than 100 ms behind (heavy load) — reset schedule to avoid
                # a burst of catch-up frames that would overwhelm Twilio's buffer.
                next_send = _time.perf_counter()

            next_send += FRAME_INTERVAL

            if self._stop_event.is_set() or not self.stream_sid:
                continue

            if self.is_speaking:
                # TTS pipeline is actively streaming and handles background
                # mixing internally.  Reset schedule so we resume cleanly the
                # instant is_speaking flips back to False.
                next_send = _time.perf_counter() + FRAME_INTERVAL
                continue

            mixer = self._get_or_create_eleven_bg_mixer()
            if mixer is None:
                # Not an ElevenLabs agent, or background explicitly disabled.
                continue

            bg_frame = mixer.mix_frame(SILENT_MULAW)
            try:
                payload = _b64.b64encode(bg_frame).decode("utf-8")
                await self.websocket.send_json({
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": payload},
                })
            except Exception:
                # WebSocket closed (hangup) — exit cleanly.
                break

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
                            from app.utils.audio_utils import apply_micro_fade_in, build_crossfade_bridge, MULAW_FRAME_BYTES

                            # We crossfade at chunk boundaries with a single 20ms overlap for speed.
                            overlap_bytes = MULAW_FRAME_BYTES  # 160 bytes (20ms)
                            # ElevenLabs optional background bed (set mixer_ref[0] before streaming).
                            mixer_ref: list[Optional[BackgroundFrameMixer]] = [None]

                            async def send_frame(frame: bytes, pace: bool = True, state: dict = None):
                                if not frame:
                                    return
                                if self._tts_cancel.is_set() or not self.stream_sid:
                                    return
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
                                def _voice_frame_mulaw(frame: bytes) -> bytes:
                                    m = mixer_ref[0]
                                    return m.mix_frame(frame) if m else frame

                                # Prime Twilio jitter buffer once per utterance (3 frames = 60ms for 400–500ms latency, no sudden noise)
                                if not self._twilio_buffer_primed:
                                    silent = bytes([0xFF]) * MULAW_FRAME_BYTES
                                    for _ in range(3):
                                        if self._tts_cancel.is_set():
                                            return
                                        await send_frame(silent, pace=False)

                                pace_state = {"send_interval": 0.02, "first": True, "next_send": time.perf_counter()}

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

                                        # bridge length == overlap_bytes => exactly one frame.
                                        # Send bridge as-is WITHOUT background mixing — it is
                                        # already a carefully blended crossfade; applying the
                                        # background mixer on top would double-process the signal
                                        # and cause the "chak-chak" distortion artefact.
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

                                        # Hold back 1 frame for crossfade tail unless final
                                        if not is_final and len(pending_frames) <= 1:
                                            continue

                                        # Send oldest frame
                                        out = pending_frames.pop(0)
                                        if fade_needed and out:
                                            out = apply_micro_fade_in(out, duration_ms=25.0)
                                            fade_needed = False
                                        await send_frame(_voice_frame_mulaw(out), pace=True, state=pace_state)

                                # End of streaming responses: handle remainder
                                if self._tts_cancel.is_set():
                                    return

                                # If we never had a chance to build a bridge but still had prev tail,
                                # clear it to avoid carrying stale audio forward.
                                if need_bridge and self._prev_tts_tail:
                                    self._prev_tts_tail = b""

                                if is_final:
                                    # Flush any partial remainder (pad with silence)
                                    if byte_buf:
                                        pad = MULAW_FRAME_BYTES - (len(byte_buf) % MULAW_FRAME_BYTES)
                                        if pad != MULAW_FRAME_BYTES:
                                            byte_buf.extend(b"\xFF" * pad)
                                        while len(byte_buf) >= MULAW_FRAME_BYTES:
                                            pending_frames.append(bytes(byte_buf[:MULAW_FRAME_BYTES]))
                                            del byte_buf[:MULAW_FRAME_BYTES]

                                    # Send all remaining frames
                                    for out in pending_frames:
                                        if fade_needed and out:
                                            out = apply_micro_fade_in(out, duration_ms=25.0)
                                            fade_needed = False
                                        await send_frame(_voice_frame_mulaw(out), pace=True, state=pace_state)
                                    pending_frames.clear()
                                    self._prev_tts_tail = b""
                                else:
                                    # Keep exactly 1 frame as tail (pad remainder into tail if needed)
                                    if byte_buf:
                                        pad = MULAW_FRAME_BYTES - (len(byte_buf) % MULAW_FRAME_BYTES)
                                        if pad != MULAW_FRAME_BYTES:
                                            byte_buf.extend(b"\xFF" * pad)
                                        while len(byte_buf) >= MULAW_FRAME_BYTES:
                                            pending_frames.append(bytes(byte_buf[:MULAW_FRAME_BYTES]))
                                            del byte_buf[:MULAW_FRAME_BYTES]

                                    # Keep last frame as tail; send earlier pending frames
                                    tail_frame = pending_frames[-1] if pending_frames else bytes([0xFF]) * MULAW_FRAME_BYTES
                                    frames_to_send = pending_frames[:-1]
                                    for out in frames_to_send:
                                        if fade_needed and out:
                                            out = apply_micro_fade_in(out, duration_ms=25.0)
                                            fade_needed = False
                                        await send_frame(_voice_frame_mulaw(out), pace=True, state=pace_state)
                                    self._prev_tts_tail = tail_frame

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
                            pcm_linear_mixer: Optional[LinearBackgroundMixer] = None
                            if tts_provider_slug and tts_provider_slug != "google":
                                tts_voice = getattr(self.agent, "tts_voice", None) if self.agent else None
                                external_voice_id = getattr(tts_voice, "external_voice_id", None)
                                if not external_voice_id:
                                    raise ValueError("TTS voice is not configured for streaming.")
                                adapter = get_tts_adapter(tts_provider_slug)
                                provider_settings = dict(getattr(self.agent, "tts_settings_json", None) or {})
                                if tts_provider_slug == "elevenlabs":
                                    bg_id, bg_level = parse_eleven_background_settings(provider_settings)
                                    if bg_id:
                                        # Request PCM so we can mix background in linear space and
                                        # encode to mu-law only once at the transport boundary.
                                        provider_settings["output_format"] = "pcm_16000"
                                        pcm_linear_mixer = LinearBackgroundMixer(bg_id, bg_level)
                                    else:
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

                                if pcm_linear_mixer is not None:
                                    downsampler = PCM16KStreamDownsampler()

                                    def _pcm_chunks_to_mulaw_chunks(sync_source):
                                        for raw_chunk in sync_source:
                                            samples_8k = downsampler.feed(raw_chunk)
                                            if samples_8k:
                                                yield pcm_linear_mixer.mix_linear_samples_to_ulaw(samples_8k)
                                        trailing = downsampler.flush()
                                        if trailing:
                                            yield pcm_linear_mixer.mix_linear_samples_to_ulaw(trailing)

                                    sync_iter = _pcm_chunks_to_mulaw_chunks(sync_iter)

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

                            mixer_ref[0] = None if pcm_linear_mixer is not None else self._get_or_create_eleven_bg_mixer()

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
                        from app.utils.audio_utils import apply_micro_fade_in
                        from app.utils.audio_utils import build_crossfade_bridge

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
        """End the call when agent response contained [END_CALL] (after TTS has played)."""
        if self._call_ended:
            return
        try:
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
            
            await transcript_service.add_and_broadcast_message(
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
            
            # Update legacy field
            conversation = transcript_service.get_conversation_array(self.db, self.call_session.id)
            self.call_session.call_transcript = conversation
            self.db.commit()
        
        except Exception as e:
            logger.error(f"Error in _add_to_transcript: {e}", exc_info=True)
    
    async def handle_start_message(self, message: dict):
        """Handle stream start - Just WebSocket connection (NOT user pickup!)"""
        try:
            self.stream_sid = message.get("streamSid")
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

            # Start continuous background loop (ElevenLabs agents only).
            # The loop is a no-op for non-ElevenLabs providers.
            if self._bg_task is None or self._bg_task.done():
                self._bg_task = asyncio.create_task(self._run_continuous_background())

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
        self._pending_interim_agent_transcript = None

        # Stop continuous background sender (stop_event already signals the loop;
        # cancel ensures we don't wait for the next 20ms sleep to elapse).
        if self._bg_task and not self._bg_task.done():
            self._bg_task.cancel()
        self._bg_task = None

        t = self._llm_response_task
        if t and not t.done():
            t.cancel()
        self._llm_response_task = None

        # Cancel any in-progress LLM streaming (orchestrator checks this flag)
        if not self._tts_cancel.is_set():
            self._tts_cancel.set()

        # Shutdown TTS pipeline worker (drains queue, cancels worker task)
        try:
            if self._tts_pipeline:
                await self._tts_pipeline.shutdown()
        except Exception:
            pass

        # Close STT / Deepgram WebSocket (sends CloseStream, closes socket,
        # waits up to 5 s for reader task to exit cleanly)
        try:
            if self._stt_pipeline:
                await self._stt_pipeline.aclose()
        except Exception:
            pass

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