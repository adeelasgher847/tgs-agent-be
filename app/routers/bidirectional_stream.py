"""
Bidirectional WebSocket for Real-time Voice AI
Handles both STT (incoming audio) and TTS (outgoing audio) simultaneously
Optimized for ultra-low latency (<3s response time)

ULTRA-AGGRESSIVE INTERIM PROCESSING:
- Processes interim STT results with 40% confidence
- Starts LLM generation immediately (100ms throttle)
- Minimal latency for fastest possible response

PARALLEL TTS PIPELINE (Vapi-style):
- User Speech → STT Interim → LLM Chunk 1 → TTS Chunk 1 Playing
                             ↓ LLM Chunk 2 → TTS Chunk 2 Generating (parallel)
                             ↓ LLM Chunk 3 → TTS Chunk 3 Queued
- TTS generation and playback happen in parallel
- Significantly reduces total response time

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
from app.services.credit_service import credit_service
from app.services.twilio_service import twilio_service
from app.services.google_tts_service import google_tts_service
from app.utils.tts_preprocessing import detect_emotion
from app.core.config import settings
from app.routers.general_websocket import broadcast_call_status_update
from app.utils.tts_preprocessing import preprocess_for_tts, quick_clean

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
    """Handles real-time bidirectional voice streaming"""

    # ULTRA-AGGRESSIVE INTERIM PROCESSING:
    # Processes interim STT results with 30% confidence (Vapi-style)
    # Starts LLM generation immediately (immediate throttle)
    # Minimal latency for fastest possible response
    
    # Quick acknowledgement: 3-word rule + probability
    QUICK_ACK_MIN_WORDS = 3
    QUICK_ACK_PROBABILITY = 0.25  # More subtle acknowledgement
    QUICK_ACK_SKIP_PHRASES = (
        "help", "emergency", "urgent", "problem", "issue", "sad", "angry",
        "please help", "asap", "critical", "wrong", "broken"
    )

    # Conversation context: keep the prompt small
    HISTORY_MAX_MESSAGES = 10 

    # True Streaming: Flush tokens as they arrive
    TTS_FLUSH_MIN_WORDS = 1 
    
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
        
        # STT (Input) state
        self.stream_sid = None
        self.call_sid = None
        self.current_speech = ""
        self._stt_session = None
        self._stt_task = None
        
        # Aggressive interim state
        self._last_interim_text = ""
        self._last_interim_sent_ts = 0.0
        self._min_interim_words = 1 
        self._min_interim_confidence = 0.30  # EXTREMELY AGGRESSIVE
        self._min_interim_interval_sec = 0.05  # 50ms throttle
        
        # TTS (Output) state - True Streaming Pipeline
        self.tts_queue = asyncio.Queue()
        self.is_speaking = False
        self._tts_cancel = asyncio.Event() 
        self._tts_lock = asyncio.Lock()
        self._tts_worker_task = None
        self._prev_tts_tail = b""
        self._twilio_buffer_primed = False
        
        # Streaming context
        self._active_llm_stream = None
        self._active_tts_stream = None
        
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
        
        # Background audio state (embedded MP3)
        self._bg_audio_task = None
        self._bg_audio_offset = 0
        self._bg_audio_mulaw = None
        self._bg_audio_length = 0
        self._bg_audio_volume = 0.6  # 60% volume (-4.4dB) - increased for better audibility
        self._use_background_audio = False
        
        # Load background audio in background (non-blocking for fast initialization)
        # This prevents cold start delays on first call after deploy/sleep
        # FFmpeg conversion can take 2-5 seconds on cold start, so we do it async
        asyncio.create_task(self._load_background_audio_async())

        # Start parallel TTS pipeline worker
        self._tts_worker_task = asyncio.create_task(self._tts_pipeline_worker())
        
        # Pre-cache common phrases in background for instant responses
        asyncio.create_task(self._precache_common_phrases())
    
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
        except Exception as e:
            logger.error(f"Error loading session data: {e}", exc_info=True)
    
    async def _load_background_audio_async(self):
        """
        Load background audio asynchronously to avoid blocking initialization.
        This prevents cold start delays on first call after deploy/sleep.
        FFmpeg conversion can take 2-5 seconds on cold start.
        """
        try:
            # Run FFmpeg conversion in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            bg_audio_bytes, bg_audio_len = await loop.run_in_executor(
                None, 
                decode_background_audio_from_base64
            )
            
            if bg_audio_bytes and bg_audio_len > 0:
                self._bg_audio_mulaw = bg_audio_bytes
                self._bg_audio_length = bg_audio_len
                self._use_background_audio = True
        except Exception as e:
            # Continue without background audio - call won't crash
            logger.warning(f"Failed to load background audio: {e}")
    
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
    
    async def _tts_pipeline_worker(self):
        """
        Background worker for True Streaming Pipeline (Vapi-style).
        
        Pipes LLM tokens directly into Google TTS Streaming and then to Twilio.
        """
        try:
            while True:
                # Get next task from queue (triggered by STT interim/final)
                task = await self.tts_queue.get()
                
                if task is None:
                    break
                
                # Check for barge-in
                if self._tts_cancel.is_set():
                    self.tts_queue.task_done()
                    continue
                
                try:
                    text_input = task.get("text", "")
                    is_greeting = task.get("is_greeting", False)
                    is_acknowledgement = task.get("is_acknowledgement", False)
                    
                    if not text_input and not is_greeting:
                        self.tts_queue.task_done()
                        continue

                    # Mark as speaking
                    self.is_speaking = True
                    self._tts_cancel.clear()

                    # Greeting and Acknowledgements use pre-cached static TTS for speed
                    if is_greeting or is_acknowledgement:
                        await self._stream_static_tts(text_input)
                    else:
                        # Full response uses True Streaming Pipeline
                        await self._run_true_streaming_pipeline(text_input)
                
                except Exception as e:
                    logger.error(f"Error in True Streaming worker: {e}", exc_info=True)
                finally:
                    self.is_speaking = False
                    self.tts_queue.task_done()
        
        except Exception as e:
            logger.error(f"True Streaming worker error: {e}", exc_info=True)

    async def _run_true_streaming_pipeline(self, user_text: str):
        """
        The Core "Vapi Style" Pipeline:
        1. Gemini Stream (LLM) -> Text Tokens
        2. Text Tokens -> Google TTS StreamingSynthesize
        3. TTS Audio Chunks -> Twilio WebSocket
        """
        try:
            # Prepare prompts and context
            system_prompt = self._build_system_prompt()
            
            model_name = "gemini-1.5-flash"
            if self.agent and self.agent.model:
                model_name = self.agent.model.model_name
                
            # Start LLM stream
            llm_stream = gemini_service.stream_text(
                prompt=user_text,
                system_prompt=system_prompt,
                model_name=model_name
            )

            # Collect tokens and stream to TTS
            # Note: We group tokens into small words for more natural TTS chunks if needed,
            # but for "True Streaming" we send as soon as possible.
            
            current_sentence = ""
            async for token in llm_stream:
                if self._tts_cancel.is_set():
                    break
                
                current_sentence += token
                
                # If we have a complete thought or enough words, stream to TTS
                # (Google's StreamingSynthesize handles incremental text well)
                if any(p in token for p in ".!?\n") or len(current_sentence.split()) >= self.TTS_FLUSH_MIN_WORDS:
                    await self._stream_tts_chunk(current_sentence)
                    current_sentence = ""
            
            # Flush remaining
            if current_sentence and not self._tts_cancel.is_set():
                await self._stream_tts_chunk(current_sentence)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")

    async def _stream_tts_chunk(self, text: str, is_final: bool = False):
        """Stream a text chunk through Google TTS Streaming to Twilio"""
        try:
            if not text or not text.strip():
                return

            # Use Google TTS Bidirectional Streaming
            tts_stream = google_tts_service.stream_text_to_speech(
                text=text,
                language=self.agent.language if self.agent else "en",
                voice_type=self.agent.voice_type if self.agent else "female"
            )

            async for audio_chunk in tts_stream:
                if self._tts_cancel.is_set():
                    break
                
                # Send to Twilio
                await stream_mulaw_bytes_over_twilio(
                    self.websocket,
                    self.stream_sid,
                    audio_chunk,
                    pace_20ms=True,
                    cancel=self._tts_cancel
                )
        except Exception as e:
            logger.error(f"TTS chunk streaming error: {e}")

    async def _stream_static_tts(self, text: str):
        """Stream static pre-cached TTS audio (for greetings/acks)"""
        try:
            audio = await generate_mulaw_tts(
                text=text,
                lang=self.agent.language if self.agent else "en",
                voice=self.agent.voice_type if self.agent else "female"
            )
            await stream_mulaw_bytes_over_twilio(
                self.websocket,
                self.stream_sid,
                audio,
                pace_20ms=True,
                cancel=self._tts_cancel
            )
        except Exception as e:
            logger.error(f"Static TTS streaming error: {e}")

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
            # Lazily create a streaming session
            if self._stt_session is None:
                self._stt_session = google_stt_service.create_streaming_session(
                    language_code=(self.agent.language + "-US") if getattr(self.agent, "language", None) == "en" else None,
                    encoding="MULAW",
                    sample_rate=8000,
                    interim_results=True,
                    single_utterance=False,
                )

                async def consume_results():
                    try:
                        # Start underlying blocking stream in executor
                        await self._stt_session.start()
                    except Exception as e:
                        logger.error(f"STT session start error: {e}", exc_info=True)
                
                # Start the session in background and concurrently read results
                async def reader_loop():
                    while True:
                        result = await self._stt_session.get_result()
                        if not result:
                            continue
                        if result.get("error"):
                            continue
                        transcript = (result.get("transcript") or "").strip()
                        if not transcript:
                            continue
                        is_final = bool(result.get("is_final"))
                        confidence = float(result.get("confidence") or 0.0)
                        if is_final:
                            await self._process_transcript(transcript, confidence)
                        else:
                            # Process interim for ultra-low latency (Vapi-like)
                            await self._maybe_process_interim(transcript, confidence)

                # kick off background readers
                self._stt_task = asyncio.create_task(reader_loop())
                asyncio.create_task(consume_results())

            # Push audio to Google
            self._stt_session.push_audio(audio_data)
        
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
            
            # Add to transcript
            await self._add_to_transcript("client", transcript, "speech", confidence)
            
            # Generate and stream response
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
            if self.is_speaking and word_count >= 2:
                if not self._tts_cancel.is_set():
                    # Set cancel flag to stop streaming
                    self._tts_cancel.set()
                    
                    # 🆕 CLEAR TTS QUEUE - Remove ALL pending audio chunks!
                    # This prevents old audio from resuming after user finishes speaking
                    while not self.tts_queue.empty():
                        try:
                            self.tts_queue.get_nowait()
                            self.tts_queue.task_done()
                        except asyncio.QueueEmpty:
                            break
                    
                    # Mark agent as no longer speaking
                    self.is_speaking = False
                
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
            
            # Passed heuristics → process immediately to start LLM generation
            self._last_interim_text = transcript
            self._last_interim_sent_ts = now
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
        words = text.split()
        if len(words) < self.QUICK_ACK_MIN_WORDS:
            return
        lower = text.lower()
        for phrase in self.QUICK_ACK_SKIP_PHRASES:
            if phrase in lower:
                return
        if random.random() >= self.QUICK_ACK_PROBABILITY:
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
        await self.tts_queue.put({
            "text": ack,
            "chunk_id": "quick_ack",
            "use_ssml": False,
            "is_acknowledgement": True,
            "is_final": False
        })
    
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM based on agent configuration and history"""
        agent_name = self.agent.name if self.agent and self.agent.name else "AI Assistant"
        agent_language = self.agent.language if self.agent and self.agent.language else "en"
        
        # Base prompt for phone conversations
        base_prompt = f"""# ROLE
You are {agent_name}, a friendly and helpful phone assistant. 
You are speaking with a user over a real-time phone call.

# GUIDELINES
1. BE CONCISE: User is on a phone call. Keep responses short (1-3 sentences max).
2. BE NATURAL: Use natural conversational fillers like "um", "uh", "I see", "got it" occasionally.
3. NO SPECIAL CHARACTERS: Do not use markdown (bold, italics, lists, etc.). Speak in plain text only.
4. HANDLE INTERRUPTIONS: If the user interrupts, stop your current thought and address them.
5. NO REPETITION: Don't repeat what the user just said unless for confirmation.

# AGENT PERSONALITY/TASK
{self.agent.system_prompt if self.agent and self.agent.system_prompt else "Assist the user with their request politely."}

# LANGUAGE
Speak only in {agent_language}.
"""
        return base_prompt

    async def generate_and_stream_response(self, user_text: str, confidence: float, is_greeting: bool = False):
        """
        Generate AI response and stream TTS in real-time.
        Uses TRUE STREAMING PIPELINE for ultra-low latency.
        """
        try:
            #👋 HANDLE AUTO-GREETING
            if is_greeting:
                # Use getattr safely as first_message is not in the model but may be used in code
                text = getattr(self.agent, "first_message", "Hello, how can I help you today?")
                await self._add_to_transcript("agent", text, "greeting")
                await self.tts_queue.put({"text": text, "is_greeting": True})
                return

            # Reset cancel flag for new response
            self._tts_cancel.clear()
            self._twilio_buffer_primed = False
            
            # 🎯 Aggressive Response: Put the task in the queue immediately!
            # The tts_pipeline_worker will start the LLM stream and TTS stream.
            await self.tts_queue.put({"text": user_text, "is_final": True})

            # Send quick acknowledgement for longer queries
            asyncio.create_task(self._send_quick_acknowledgement(user_text))

        except Exception as e:
            logger.error(f"Error triggering response generation: {e}")
    async def _send_quick_acknowledgement(self, user_text: str):
        """Send a quick 'Got it' or 'I see' for longer user queries to reduce perceived latency"""
        import random
        
        # Criteria: Not a greeting, longer than N words
        words = user_text.split()
        if len(words) < self.QUICK_ACK_MIN_WORDS:
            return
            
        # Probability check
        if random.random() > self.QUICK_ACK_PROBABILITY:
            return
            
        # Skip for emotional/serious phrases
        lower_text = user_text.lower()
        if any(skip in lower_text for skip in self.QUICK_ACK_SKIP_PHRASES):
            return
            
        # Select random ack
        acks = ["Got it.", "I see.", "Okay.", "Right.", "Understood."]
        selected_ack = random.choice(acks)
        
        # Queue as a priority (is_acknowledgement=True skips LLM)
        await self.tts_queue.put({"text": selected_ack, "is_acknowledgement": True})
    async def _add_to_transcript(self, role: str, content: str, message_type: str = "chat"):
        """Add a message to the call transcript in the database"""
        try:
            from datetime import datetime, timezone
            import json
            
            # Build message object
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message_type": message_type
            }
            
            # Update session transcript
            if self.call_session:
                transcript = []
                if self.call_session.call_transcript:
                    try:
                        transcript = json.loads(self.call_session.call_transcript) if isinstance(self.call_session.call_transcript, str) else self.call_session.call_transcript
                    except:
                        transcript = []
                
                transcript.append(message)
                self.call_session.call_transcript = json.dumps(transcript)
                self.db.commit()
                
        except Exception as e:
            logger.error(f"Error adding to transcript: {e}")

    async def _handle_barge_in(self):
        """Handle user barge-in (interruption)"""
        try:
            # 1. Signal all active streams to stop
            self._tts_cancel.set()
            
            # 2. Stop Twilio playback immediately
            await self.websocket.send_json({
                "event": "clear",
                "streamSid": self.stream_sid
            })
            
            # 3. Log interruption
            logger.info("🚫 Barge-in detected: Clearing streams and playback")
            
        except Exception as e:
            logger.error(f"Error handling barge-in: {e}")

    
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
                                # Prime Twilio jitter buffer once per utterance
                                if not self._twilio_buffer_primed:
                                    silent = bytes([0xFF]) * MULAW_FRAME_BYTES
                                    for _ in range(5):
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
                        
                        # Prime Twilio's jitter buffer with 100ms (5 frames) of silence for the very first speak only
                        prime_frames = 0 if self._twilio_buffer_primed else 5
                        
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
    
    async def _stream_background_audio_loop(self):
        """
        Continuously stream background audio in a loop.
        PAUSES when TTS is speaking to avoid conflicts.
        Uses proper pacing with drift correction for smooth playback.
        """
        if not self._bg_audio_mulaw or self._bg_audio_length == 0:
            return
        
        send_interval = 0.02  # 20ms per frame
        frame_bytes = MULAW_FRAME_BYTES  # 160 bytes at 8kHz
        first = True
        next_send = time.perf_counter()
        
        try:
            while True:
                if not self.stream_sid:
                    await asyncio.sleep(0.1)
                    continue
                
                # PAUSE background audio when TTS is speaking (no noise when AI speaks)
                if self.is_speaking:
                    # Reset timing when pausing so resume is smooth
                    first = True
                    next_send = time.perf_counter()
                    await asyncio.sleep(0.01)  # High-frequency check (10ms) for instant pause
                    continue
                
                bg_chunk = get_background_audio_chunk(
                    self._bg_audio_offset,
                    frame_bytes,
                    self._bg_audio_mulaw,
                    self._bg_audio_length
                )
                
                self._bg_audio_offset = (self._bg_audio_offset + frame_bytes) % self._bg_audio_length
                
                payload = base64.b64encode(bg_chunk).decode("utf-8")
                await self.websocket.send_json({
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": payload}
                })
                
                # Proper pacing with drift correction (same as TTS streaming)
                if not first:
                    next_send += send_interval
                    now = time.perf_counter()
                    sleep_dur = next_send - now
                    if sleep_dur > 0:
                        await asyncio.sleep(sleep_dur)
                    elif sleep_dur < -0.03:
                        # We're late by >30ms; reset schedule to avoid cumulative jitter
                        next_send = time.perf_counter()
                else:
                    first = False
                    next_send = time.perf_counter() + send_interval
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in _stream_background_audio_loop: {e}")
    
    async def _stream_mulaw_with_background(
        self,
        audio_bytes: bytes,
        cancel: Optional[asyncio.Event] = None
    ):
        """
        Stream TTS audio mixed with continuous background audio.
        Uses proper pacing with drift correction to prevent stuttering.
        """
        send_interval = 0.02  # 20ms
        first = True
        next_send = time.perf_counter()
        
        for frame in iter_mulaw_20ms_frames(audio_bytes):
            if cancel and cancel.is_set():
                break
            
            if self._bg_audio_mulaw and self._bg_audio_length > 0:
                mixed_frame = mix_audio_with_background(
                    tts_audio=frame,
                    bg_audio=self._bg_audio_mulaw,
                    bg_length=self._bg_audio_length,
                    bg_offset=self._bg_audio_offset,
                    volume_level=self._bg_audio_volume
                )
                self._bg_audio_offset = (self._bg_audio_offset + len(frame)) % self._bg_audio_length
            else:
                mixed_frame = frame
            
            payload = base64.b64encode(mixed_frame).decode("utf-8")
            await self.websocket.send_json({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": payload}
            })
            
            # Proper pacing with drift correction (same as stream_mulaw_bytes_over_twilio)
            if not first:
                next_send += send_interval
                now = time.perf_counter()
                sleep_dur = next_send - now
                if sleep_dur > 0:
                    await asyncio.sleep(sleep_dur)
                elif sleep_dur < -0.03:
                    # We're late by >30ms; reset schedule to avoid cumulative jitter
                    next_send = time.perf_counter()
            else:
                first = False
                next_send = time.perf_counter() + send_interval
    
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
                            prime_frames=0 if self._twilio_buffer_primed else 5,
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
        confidence: Optional[float] = None
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
                confidence=confidence
            )
            
            # Update legacy field
            conversation = transcript_service.get_conversation_array(self.db, self.call_session.id)
            self.call_session.call_transcript = conversation
            self.db.commit()
        
        except Exception as e:
            logger.error(f"Error in _add_to_transcript: {e}", exc_info=True)
    
    async def handle_start_message(self, message: dict):
        """Handle stream start"""
        try:
            self.stream_sid = message.get("streamSid")
            start = message.get("start", {})
            self.call_sid = start.get("callSid")
            
            logger.info(f"🚀 Stream started: StreamSid={self.stream_sid}, CallSid={self.call_sid}")
            
            # 🔥 IMMEDIATE GREETING for outbound engagement
            # This ensures the user hears the bot as soon as they pick up.
            await self.generate_and_stream_response("", 1.0, is_greeting=True)
            
            # Start background audio if enabled
            if self._use_background_audio and self._bg_audio_mulaw:
                 asyncio.create_task(self._start_background_audio_with_delay())
                 
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
            # Delay prevents cold start issues and gives call time to establish
            if self._use_background_audio and self._bg_audio_mulaw and not self._bg_audio_task:
                asyncio.create_task(self._start_background_audio_with_delay())
        
        except Exception as e:
            logger.error(f"Error in _handle_user_pickup: {e}", exc_info=True)
    
    async def _start_background_audio_with_delay(self):
        """
        Start background audio after 3 second delay to allow call to establish.
        This prevents cold start issues and ensures call is fully connected before background audio starts.
        """
        try:
            # Wait 3 seconds for call to fully establish
            await asyncio.sleep(3.0)
            
            # Check again if background audio is ready and not already started
            if self._use_background_audio and self._bg_audio_mulaw and not self._bg_audio_task:
                self._bg_audio_task = asyncio.create_task(self._stream_background_audio_loop())
        except Exception as e:
            logger.error(f"Error in _start_background_audio_with_delay: {e}", exc_info=True)
    
    async def handle_stop_message(self, message: dict):
        """Handle stream stop"""
        try:
            # Stop TTS pipeline worker
            try:
                if self._tts_worker_task:
                    await self.tts_queue.put(None)  # Shutdown signal
                    await asyncio.wait_for(self._tts_worker_task, timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass
            
            # Stop background audio streaming
            try:
                if self._bg_audio_task:
                    self._bg_audio_task.cancel()
                    try:
                        await self._bg_audio_task
                    except asyncio.CancelledError:
                        pass
            except Exception:
                pass
            
            # Close STT session
            try:
                if self._stt_session:
                    self._stt_session.finish()
                if self._stt_task:
                    await asyncio.sleep(0)  # yield
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
    TTS-ONLY WebSocket for streaming audio playback
    
    Used with recording-based STT:
    - Recording callback sends TTS text via custom event
    - WebSocket streams audio in 20ms MULAW chunks
    - No STT handling (recording handles that)
    
    Flow:
    1. Connect to WebSocket
    2. Receive custom {"event": "play_tts", "text": "...", "lang": "en", "voice": "female"}
    3. Generate MULAW TTS
    4. Stream in 20ms chunks
    5. Send {"event": "tts_complete"} when done
    """
    try:
        await websocket.accept()
    except Exception:
        return
    
    # Get database session
    from app.db.session import SessionLocal
    db = SessionLocal()
    
    # Get agent info for voice settings
    agent = None
    call_session = None
    stream_sid = None
    
    try:
        session_uuid = uuid.UUID(callSessionId)
        call_session = call_session_service.get_call_session_by_id(db, session_uuid)
        
        if call_session and agentId:
            agent_uuid = uuid.UUID(agentId)
            agent = agent_service.get_agent_by_id(db, agent_uuid, call_session.tenant_id)
    except Exception as e:
        logger.error(f"Error loading session data for tts-only: {e}")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            event = message.get("event")
            
            if event == "connected":
                pass
            
            elif event == "start":
                stream_sid = message.get("streamSid")
                
                # Auto-retrieve and play pending TTS from call session metadata
                if call_session and call_session.call_metadata:
                    pending_tts = call_session.call_metadata.get("pending_tts")
                    if pending_tts:
                        text = pending_tts.get("text", "")
                        lang = pending_tts.get("lang", agent.language if agent else "en")
                        voice = pending_tts.get("voice", agent.voice_type if agent else "female")
                        
                        if text:
                            # Generate MULAW TTS with Chirp 3: HD
                            audio_bytes = await generate_mulaw_tts(
                                text=text,
                                lang=lang,
                                voice=voice,
                                use_chirp3_hd=True,
                                speaking_rate=1.0,
                                add_office_bg=True
                            )
                            
                            # Apply audio optimizations
                            from app.utils.audio_utils import apply_micro_fade_in
                            audio_bytes = apply_micro_fade_in(audio_bytes, duration_ms=25.0)

                            # Stream in 20ms chunks with 100ms jitter buffer priming
                            await stream_mulaw_bytes_over_twilio(
                                websocket=websocket,
                                stream_sid=stream_sid,
                                audio_bytes=audio_bytes,
                                pace_20ms=True,
                                prime_frames=5
                            )
                            
                            # Clear pending TTS
                            call_session.call_metadata.pop("pending_tts", None)
                            db.commit()
            
            elif event == "play_tts":
                # Custom event to trigger TTS playback
                text = message.get("text", "")
                lang = message.get("lang", agent.language if agent else "en")
                voice = message.get("voice", agent.voice_type if agent else "female")
                
                if text and stream_sid:
                    # Generate MULAW TTS with Chirp 3: HD
                    audio_bytes = await generate_mulaw_tts(
                        text=text,
                        lang=lang,
                        voice=voice,
                        use_chirp3_hd=True,
                        speaking_rate=1.0,
                        add_office_bg=True
                    )
                    
                    # Apply audio optimizations
                    from app.utils.audio_utils import apply_micro_fade_in
                    audio_bytes = apply_micro_fade_in(audio_bytes, duration_ms=25.0)

                    # Stream in 20ms chunks with 100ms jitter buffer priming
                    await stream_mulaw_bytes_over_twilio(
                        websocket=websocket,
                        stream_sid=stream_sid,
                        audio_bytes=audio_bytes,
                        pace_20ms=True,
                        prime_frames=5
                    )
                    
                    # Send completion event
                    await websocket.send_json({
                        "event": "tts_complete",
                        "text_length": len(text),
                        "audio_bytes": len(audio_bytes)
                    })
            
            elif event == "media":
                # Ignore incoming media (we're TTS-only)
                pass
            
            elif event == "stop":
                break
            
            elif event == "mark":
                pass  # Synchronization marks
    
    except WebSocketDisconnect:
        logger.info(f"🔌 TTS-only WebSocket disconnected for session {callSessionId}")
    
    except Exception as e:
        logger.error(f"Unexpected error in TTS-only WebSocket: {e}", exc_info=True)
    
    finally:
        db.close()