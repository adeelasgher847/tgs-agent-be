"""
Bidirectional WebSocket for Real-time Voice AI
Handles both STT (incoming audio) and TTS (outgoing audio) simultaneously
Target latency: 400–500ms (Vapi-style).

STT → LLM → TTS FLOW:
- Twilio sends audio every 20ms (MULAW 8kHz). We push each chunk to Google STT.
- As soon as ~30ms of STT stream produces interim result → send to LLM (30ms throttle).
- LLM streams response → each flush (sentence/time ~200ms) → TTS chunk.
- When VAD detects user silent (final): the TTS we built from interim is already playing/queued;
  we do NOT start a second response (one response per turn = gapless, no duplicate).

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

3. Backchannels:
   - "mm-hmm", "I see", "okay", "right", "yeah", "got it"
   - Triggered during long user monologues (5-7+ seconds)
   - 30% random chance for naturalness

4. Turn-Taking & Barge-In:
   - ENABLED - Agent stops immediately when user starts speaking
   - Detection: 2+ words (Google interim confidence is unreliable, often 0.00)
   - Checked FIRST before interim gating (highest priority!)
   - TTS queue cleared (prevents old audio from resuming)
   - Waits for final transcript before responding (no partial interruptions)

5. Persona & Variability:
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

6. Ambient Background Noise:
   - DISABLED by default (caused distortion on some systems)
   - Can be enabled via self._use_ambient_noise = True
   - Subtle pink noise mixed with TTS audio (-46dB if enabled)
   - Note: Use with caution, may cause audio artifacts on certain setups

CACHING & LOW-LATENCY STRATEGIES:
1. Auto-Greeting on Connect:
   - Agent speaks FIRST when call connects (no waiting for user!)
   - Uses agent's first_message or default: "hello how are you"
   - Bypasses LLM entirely for instant greeting (<200ms)
   - Eliminates awkward silence at call start

2. Pre-cached Common Phrases:
   - 36+ common phrases pre-generated at startup
   - Greetings, acknowledgements, confirmations cached
   - <50ms response time for cached phrases (vs 500-2900ms generation)
   - Instant "Hello", "Got it", "Thank you" responses

3. Quick Acknowledgement Pattern (5-Word Rule + Probability):
   - Eligible when user says 5+ words; then only ~38% chance we send "Got it" (more natural).
   - Never used for emotional/serious content (help, emergency, problem, etc.).
   - Instant from cache when sent; then full response streams in parallel.
   - Example: "Got it" (50ms) → "checking that now..." (1500ms)

4. Adaptive Max Tokens:
   - Yes/No queries: 15 tokens (ultra-fast)
   - Short queries (1-3 words): 25 tokens (fast)
   - Medium queries (4-7 words): 35 tokens (balanced)
   - Complex queries: Full configured tokens
   - 30-60% faster LLM generation for simple queries

4. TTS Client Pre-warming:
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
from typing import Optional, Dict, Iterable
import time
from datetime import datetime, timezone
import uuid
import sys
import math
import re
from app.core.logger import logger

# Google built-in endpointing (VAD) will be used via streaming_recognize

from app.services.google_stt_service import google_stt_service
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
from app.services.google_tts_service import google_tts_service
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
from app.voice.rag_context import build_rag_context_block, build_rag_context_block_with_trace
from app.voice.tts_only_session import TtsOnlySession

# Import utilities and services
from app.utils.audio_utils import (
    decode_background_audio_from_base64,
    get_background_audio_chunk,
    apply_volume_fade,
    mix_audio_with_background,
    ulaw_to_linear_sample,
    linear_to_ulaw_sample,
    iter_mulaw_20ms_frames,
    stream_mulaw_bytes_over_twilio,
    crossfade_mulaw_segments,
    build_crossfade_bridge,
    add_ambient_noise_to_mulaw,
    MULAW_FRAME_BYTES
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
    build_tts_only_twiml
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
        
        # STT (Input) state - Google streaming_recognize with built-in VAD
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
        
        # Natural conversation state (backchannels & persona)
        self._user_speech_duration = 0.0    # Track user monologue duration
        self._last_backchannel_time = 0.0   # Prevent frequent backchannels
        self._last_user_speech_start = 0.0  # Track when user started speaking
        # Backchannels should be SHORT and natural (avoid long phrases that sound like interruptions)
        self._backchannel_phrases = [
            "mm-hmm",
            "uh-huh",
            "hmm",
            "I see",
            "okay",
            "alright",
            "right",
            "yeah",
            "got it",
            "oh, I see",
            "oh, okay",
        ]
        self._use_ssml = True                # Enable SSML by default
        
        # Session data
        self.call_session = None
        self.agent = None
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

        # One response per turn (Vapi-style): when we start LLM from interim, final only commits
        self._turn_response_started = False  # True after first interim triggers LLM for this turn
        self._auto_greeting_sent = False
        
        # Background audio manager (embedded MP3 / ambient noise)
        self._background_audio = BackgroundAudioManager(
            websocket=self.websocket,
            get_stream_sid=lambda: self.stream_sid,
            is_speaking_flag=lambda: self.is_speaking,
        )
        
        # Load background audio in background (non-blocking for fast initialization).
        # This prevents cold start delays on first call after deploy/sleep.
        asyncio.create_task(self._background_audio.load_from_base64_async())

        # Start parallel TTS pipeline worker via TtsPipeline facade
        self._tts_pipeline = TtsPipeline(self)
        self._tts_worker_task = self._tts_pipeline._worker_task
        
        # Pre-cache common phrases in background for instant responses
        asyncio.create_task(self._precache_common_phrases())

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
        """Handle incoming audio from Twilio and feed to Google streaming STT"""
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

            # (Removed first-media DB marker for outbound gating)
            # Lazily create STT pipeline and push audio
            if self._stt_pipeline is None:
                language = getattr(self.agent, "language", None)
                language_code = (language + "-US") if language == "en" else language
                self._stt_pipeline = SttPipeline(
                    language_code=language_code,
                    on_interim=self._maybe_process_interim,
                    on_final=self._process_transcript,
                )

            await self._stt_pipeline.feed_audio_chunk(audio_data)
        
        except Exception as e:
            logger.error(f"Error handling media message: {e}", exc_info=True)
    
    # Removed chunk-based STT processing; relying on Google streaming endpointing
    
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
            
            # Reset user speech timer (user finished speaking)
            self._last_user_speech_start = 0.0
            
            # Reset interim state (user finished, ready for new response)
            self._tts_cancel.clear()
            self._last_interim_text = ""

            # Add to transcript (always)
            await self._add_to_transcript("client", transcript, "speech", confidence)

            # One response per turn: if we already started LLM from interim, TTS is already playing/queued
            # When VAD detects user silent (final), we do NOT start a second response — gapless playback
            if self._turn_response_started:
                self._turn_response_started = False
                return  # TTS built from interim continues; no duplicate response

            # Generate and stream response (no interim was used for this turn, e.g. very short utterance)
            await self.generate_and_stream_response(transcript, confidence)
            
        except Exception as e:
            logger.error(f"Error processing transcript: {e}", exc_info=True)

    async def _maybe_inject_backchannel(self, transcript: str):
        """
        Inject backchannel responses during long user monologues.
        Triggered after 5-7 seconds of continuous user speech.
        """
        import random
        
        now = time.time()
        
        # Track user speech duration
        if not self._last_user_speech_start:
            self._last_user_speech_start = now
        
        speech_duration = now - self._last_user_speech_start
        time_since_last_backchannel = now - self._last_backchannel_time
        
        # Inject backchannel if:
        # 1. User has been speaking for 5-7+ seconds
        # 2. At least 3 seconds since last backchannel
        # 3. We're not currently speaking
        # 4. Random chance (30%) for naturalness
        should_backchannel = (
            speech_duration >= random.uniform(5.0, 7.0) and
            time_since_last_backchannel >= 3.0 and
            not self.is_speaking and
            random.random() < 0.3
        )
        
        if should_backchannel:
            backchannel = random.choice(self._backchannel_phrases)
            
            # Queue backchannel with minimal processing
            await self.tts_queue.put({
                "text": backchannel,
                "chunk_id": "backchannel",
                "is_backchannel": True,
                "is_final": True,
                "use_ssml": False
            })
            
            self._last_backchannel_time = now

    async def _maybe_process_interim(self, transcript: str, confidence: float):
        """
        ULTRA-AGGRESSIVE interim processing for minimal latency.
        Processes interim STT results with 40% confidence to start LLM generation ASAP.
        Also tracks user speech for backchannel injection.
        """
        try:
            if not transcript:
                return
            
            # Check for backchannel opportunity during long user speech
            await self._maybe_inject_backchannel(transcript)
            
            # Calculate word count for checks
            word_count = len(transcript.split())
            
            # ✅ BARGE-IN CHECK FIRST - Highest priority! Stop agent immediately!
            # Detection: 2+ words, agent currently speaking
            # NOTE: We check this BEFORE interim gate because barge-in should work
            # even with low confidence (Google interim often returns 0.00 confidence)
            if self._tts_pipeline and self._tts_pipeline.is_speaking and word_count >= 2:
                # Set cancel flag, clear queue, and mark agent as not speaking
                await self._tts_pipeline.cancel_current_and_clear_queue()
                
                # Don't process interim during barge-in - wait for final transcript
                return
            
            # Basic gating: confidence and minimum words (for LLM generation only)
            # NOTE: This comes AFTER barge-in check so that interruption works even with low confidence
            if confidence < self._min_interim_confidence or word_count < self._min_interim_words:
                return
            
            # Ultra-aggressive throttling: only 100ms between triggers
            now = asyncio.get_event_loop().time()
            if (now - self._last_interim_sent_ts) < self._min_interim_interval_sec:
                return
            
            # ULTRA-AGGRESSIVE: Process even small advances (no minimum word requirement)
            # This ensures we start LLM generation as soon as possible
            if self._last_interim_text and transcript.startswith(self._last_interim_text):
                advanced = transcript[len(self._last_interim_text):].strip()
                # Skip only if there's literally no new content
                if not advanced:
                    return
            
            # Passed heuristics → process immediately to start LLM generation (30ms-style trigger)
            self._last_interim_text = transcript
            self._last_interim_sent_ts = now
            self._turn_response_started = True  # One response per turn; final will not start a second one
            await self.generate_and_stream_response(transcript, confidence)
        except Exception as e:
            logger.error(f"Error processing interim: {e}")
    
    async def _send_quick_acknowledgement(self, user_text: str):
        """
        Send instant acknowledgement for longer queries while generating full response.
        Uses pre-cached phrases for <50ms latency.
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
    
    async def generate_and_stream_response(self, user_text: str, confidence: float, is_greeting: bool = False):
        """
        Generate AI response and stream TTS in real-time WITH conversation history.
        Uses PARALLEL TTS PIPELINE (Vapi-style) for ultra-low latency.
        
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
                # Get greeting from agent or use default
                if self.agent and hasattr(self.agent, 'first_message') and self.agent.first_message:
                    greeting_text = self.agent.first_message
                else:
                    if self.call_session and self.call_session.call_type == "inbound":
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

                    # Use only the most recent HISTORY_MAX_MESSAGES to keep prompt within model limits
                    max_msgs = getattr(self, "HISTORY_MAX_MESSAGES", 40)
                    if len(filtered) > max_msgs:
                        filtered = filtered[-max_msgs:]

                    # Build history text from the bounded window
                    for role, content in filtered:
                        history_lines.append(f"{role.capitalize()}: {content}")

                    history_text = "\n".join(history_lines)
                except Exception:
                    history_text = ""
            
            # Build system prompt with agent personality + history
            agent_name = self.agent.name if self.agent and self.agent.name else "AI Assistant"
            agent_language = self.agent.language if self.agent and self.agent.language else "en"
            
            # Base prompt for phone conversations (voice-first, plain text only, no SSML)
            base_prompt = f"""# ROLE
You are {agent_name}, having a real-time phone call with a human.

# STYLE & TONE
- VOICE-FIRST: Your output is for Text-to-Speech. Use short, punchy sentences.
- NATURAL: Use natural fillers/interjections ONLY when they fit the emotion: "umm", "hmm", "oh", "alright", "hang on", "one moment" (max one per response).
- CONCISE: Max 20 words per response unless explaining something complex.
- NO ROBOT TALK: Avoid "As an AI" or formal greetings. Use "Hey," "Hi," or "Hello."
- OUTPUT PLAIN TEXT ONLY: Do NOT output SSML, XML, or any tags. Prosody is handled by the system.
- TEXT HYGIENE: Avoid "..." (use a comma or short sentence). Avoid slashes like "FastAPI/ML" (say "FastAPI and ML").

# CONVERSATION STATE
Previous conversation:
{history_text}

{rag_context_block}
{inbound_prompt_context_block}
{inbound_kb_docs_context_block}

# CRITICAL RULES
1. NO REPETITION: If the history shows you asked a question, move to the next point.
2. HANDLING SILENCE: If the user says something vague, ask a clarifying question.
3. TERMINATION: When the objective is met, say a friendly goodbye and end your response with exactly [END_CALL].
4. NO SSML: Do NOT output <speak>, <prosody>, or any XML tags. Plain text only.

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
- OUTPUT PLAIN TEXT ONLY: Do NOT output SSML, XML, or tags. Prosody is handled by the system.
- TEXT HYGIENE: Avoid "..." (use a comma or short sentence). Avoid slashes like "FastAPI/ML" (say "FastAPI and ML").

# CONVERSATION STATE
Previous conversation:
{history_text}

{rag_context_block}
{inbound_prompt_context_block}
{inbound_kb_docs_context_block}

# CRITICAL RULES
1. NO REPETITION: Do not repeat questions already asked. Move to the next point.
2. TERMINATION: When all objectives from your custom instructions are complete, say a friendly goodbye and end your response with exactly [END_CALL].
3. NO SSML: Plain text only. No <speak>, <prosody>, or XML.

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
- OUTPUT PLAIN TEXT ONLY: Do NOT output SSML, XML, or tags. Prosody is handled by the system.

# CONVERSATION STATE
Previous conversation:
{history_text}

{rag_context_block}
{inbound_prompt_context_block}
{inbound_kb_docs_context_block}

# CRITICAL RULES
1. NO REPETITION: Do not repeat questions. Move to the next point.
2. TERMINATION: When all objectives are complete, say a friendly goodbye and end your response with exactly [END_CALL].
3. NO SSML: Plain text only. No <speak>, <prosody>, or XML.

# GOAL
Follow the model instructions. Continue from the history above. Be {agent_name}."""
            else:
                # Use base prompt
                system_prompt = base_prompt
            
            # Get agent's configured model and provider
            llm_service = None
            model_name = "gemini-1.5-flash"  # Default fallback
            api_key = None
            temperature = 0.5
            max_tokens = 100
            
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
                            chunk_counter += 1
                            await self.tts_queue.put({
                                "text": final_text,
                                "chunk_id": chunk_counter,
                                "use_ssml": self._use_ssml,
                                "is_final": True
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
                # Strip [END_CALL] from transcript so saved conversation is clean
                transcript_text = final_text.replace("[END_CALL]", "").strip()
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
        
        except Exception as e:
            logger.error(f"Error in generate_and_stream_response: {e}", exc_info=True)
    
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

                    # Prefer true streaming TTS for longer responses (real-time playback).
                    # Keep cache-friendly path for very short phrases (ack/backchannel).
                    word_count = len(clean.split())
                    use_streaming_tts = word_count >= 4  # speak sooner; streaming reduces time-to-first-audio
                    if use_streaming_tts and not self._tts_cancel.is_set():
                        try:
                            import base64
                            import time
                            from app.utils.audio_utils import apply_micro_fade_in, build_crossfade_bridge, MULAW_FRAME_BYTES

                            # We crossfade at chunk boundaries with a single 20ms overlap for speed.
                            overlap_bytes = MULAW_FRAME_BYTES  # 160 bytes (20ms)

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

                                        # bridge length == overlap_bytes => exactly one frame
                                        if fade_needed and bridge:
                                            bridge = apply_micro_fade_in(bridge, duration_ms=25.0)
                                            fade_needed = False
                                        if bridge:
                                            await send_frame(bridge[:MULAW_FRAME_BYTES], pace=True, state=pace_state)

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
                                        await send_frame(out, pace=True, state=pace_state)

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
                                        await send_frame(out, pace=True, state=pace_state)
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
                                        await send_frame(out, pace=True, state=pace_state)
                                    self._prev_tts_tail = tail_frame

                                self._twilio_buffer_primed = True

                            # Use Google StreamingSynthesize (bidirectional streaming) to reduce time-to-first-audio
                            # NEVER send SSML to streaming synthesize; strip tags to prevent them being spoken.
                            streaming_text = strip_ssml_tags(clean) if use_ssml or clean.lstrip().startswith("<speak>") else clean

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

                            audio_iter = google_tts_service.stream_text_to_speech(
                                text=streaming_text,
                                language=lang,
                                voice_type=voice,
                                speaking_rate=speaking_rate,
                                output_format="mulaw",
                                use_chirp3_hd=True,
                                sample_rate_hz=8000,
                            )

                            await stream_mulaw_from_audio_iter(audio_iter)
                            return  # streaming path complete
                        except Exception as e:
                            logger.warning(f"⚠️ Streaming TTS failed, falling back to non-streaming: {e}")

                            # If call ended / barge-in occurred, never fall back to batch TTS.
                            if self._tts_cancel.is_set() or not self.stream_sid:
                                self._prev_tts_tail = b""
                                return
                    
                    # Generate TTS audio (Google TTS auto-detects SSML)
                    # Note: add_office_bg=False because mixing is handled during streaming
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
                        add_office_bg=False  # Background mixing handled separately
                    )
                    
                    if self._tts_cancel.is_set():
                        self._prev_tts_tail = b""
                        return
                    
                    # Stream TTS CLEAN (no background mixing when AI is speaking)
                    # Background audio loop will automatically pause when is_speaking=True
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
                        
                        # Always stream clean TTS - background loop handles background separately
                        # Background loop pauses automatically when is_speaking=True
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
    
    async def _stream_mulaw_with_background(
        self,
        audio_bytes: bytes,
        cancel: Optional[asyncio.Event] = None
    ):
        """
        Stream TTS audio mixed with continuous background audio.
        Uses proper pacing with drift correction to prevent stuttering.
        """
        # Delegate mixing to BackgroundAudioManager, then reuse Twilio helper
        mixed_bytes = self._background_audio.mix_with_background(audio_bytes)
        await stream_mulaw_bytes_over_twilio(
            websocket=self.websocket,
            stream_sid=self.stream_sid,
            audio_bytes=mixed_bytes,
            pace_20ms=True,
            cancel=cancel,
        )
    
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
                        generate_mulaw_tts(text=suffix, lang=lang, voice=voice, use_chirp3_hd=True, speaking_rate=1.0, add_office_bg=True)
                    ) if suffix else None

                    # Generate prefix audio immediately
                    prefix_audio = await generate_mulaw_tts(text=prefix, lang=lang, voice=voice, use_chirp3_hd=True, speaking_rate=1.0, add_office_bg=True)

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
                    
                    # Update call session status to completed
                    if self.call_session:
                        self.call_session.status = "completed"
                        self.call_session.end_time = datetime.now(timezone.utc)
                        self.call_session.ended_reason = "User said goodbye"
                        
                        if self.call_session.start_time:
                            duration = (self.call_session.end_time - self.call_session.start_time).total_seconds()
                            self.call_session.duration = int(duration)
                        
                        self.db.commit()
                    
                    # End Twilio call
                    if self.call_sid:
                        twilio_service.end_call(self.call_sid)
                    
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
                            logger.debug(f"WebSocket close failed after goodbye: {e}")
                    
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
                self.call_session.status = "completed"
                self.call_session.end_time = datetime.now(timezone.utc)
                self.call_session.ended_reason = "Agent sent [END_CALL]"
                if self.call_session.start_time:
                    duration = (self.call_session.end_time - self.call_session.start_time).total_seconds()
                    self.call_session.duration = int(duration)
                self.db.commit()
            if self.call_sid:
                twilio_service.end_call(self.call_sid)
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
                    
                    # Update call session status to completed
                    if self.call_session:
                        self.call_session.status = "completed"
                        self.call_session.end_time = datetime.now(timezone.utc)
                        self.call_session.ended_reason = "Voicemail detected"
                        
                        if self.call_session.start_time:
                            duration = (self.call_session.end_time - self.call_session.start_time).total_seconds()
                            self.call_session.duration = int(duration)
                        
                        self.db.commit()
                    
                    # End Twilio call immediately
                    if self.call_sid:
                        twilio_service.end_call(self.call_sid)
                    
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
                            logger.debug(f"WebSocket close failed after voicemail detection: {e}")
                    
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
            
            # 🎵 START BACKGROUND AUDIO LOOP - User picked up, start after 3 seconds delay
            # Handler only decides *when* to start; manager owns implementation.
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
    
    async def _start_background_audio_with_delay(self):
        """
        Start background audio after 3 second delay to allow call to establish.
        This prevents cold start issues and ensures call is fully connected before background audio starts.
        """
        try:
            # Delegate to BackgroundAudioManager, preserving the 3-second delay
            await self._background_audio.start_loop_if_enabled(delay_seconds=3.0)
        except Exception as e:
            logger.error(f"Error in _start_background_audio_with_delay: {e}", exc_info=True)
    
    async def handle_stop_message(self, message: dict):
        """Handle stream stop"""
        try:
            # Stop TTS pipeline worker
            try:
                if self._tts_pipeline:
                    await self._tts_pipeline.shutdown()
            except (asyncio.TimeoutError, Exception):
                pass
            
            # Stop background audio streaming
            try:
                await self._background_audio.stop_loop()
            except Exception:
                pass
            
            # Close STT session
            try:
                if self._stt_pipeline:
                    self._stt_pipeline.finish_session()
            except Exception:
                pass
        
        except Exception as e:
            logger.error(f"Error in handle_stop_message: {e}", exc_info=True)


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
            # Receive message from Twilio
            data = await websocket.receive_text()
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