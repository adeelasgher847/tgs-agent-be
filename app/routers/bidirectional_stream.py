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
import struct

# Google built-in endpointing (VAD) will be used via streaming_recognize

from app.services.google_stt_service import google_stt_service
from app.services.google_tts_service import google_tts_service
from app.services.call_session_service import call_session_service
from app.services.agent_service import agent_service
from app.services.voice_logging_service import VoiceLoggingService
from app.services.transcript_service import transcript_service
from app.services.gemini_service import gemini_service
from app.services.openai_service import openai_service
from app.core.config import settings
from app.routers.tts_audio import audio_cache, generate_cache_key
from app.routers.general_websocket import broadcast_call_status_update

# Real-time TTS MULAW streaming constants
MULAW_SAMPLE_RATE_HZ = 8000  # Twilio-friendly
BYTES_PER_SECOND = MULAW_SAMPLE_RATE_HZ  # 8-bit mu-law => 1 byte per sample
CHUNK_DURATION_SEC = 0.02  # 20ms
MULAW_FRAME_BYTES = int(BYTES_PER_SECOND * CHUNK_DURATION_SEC)  # 160 bytes

router = APIRouter()


def iter_mulaw_20ms_frames(audio_bytes: bytes) -> Iterable[bytes]:
    """
    Yield 20ms mu-law frames (160 bytes at 8kHz).
    Pads the final frame with mu-law silence (0xFF) if needed.
    """
    if not audio_bytes:
        return
    total_len = len(audio_bytes)
    full_frames = total_len // MULAW_FRAME_BYTES
    remainder = total_len % MULAW_FRAME_BYTES

    offset = 0
    for _ in range(full_frames):
        yield audio_bytes[offset:offset + MULAW_FRAME_BYTES]
        offset += MULAW_FRAME_BYTES

    if remainder:
        last = bytearray(audio_bytes[offset:])
        last.extend(b'\xFF' * (MULAW_FRAME_BYTES - remainder))  # mu-law silence pad
        yield bytes(last)


async def stream_mulaw_bytes_over_twilio(
    websocket,
    stream_sid: str,
    audio_bytes: bytes,
    pace_20ms: bool = True,
    cancel: Optional[asyncio.Event] = None,
    prime_frames: int = 0,
):
    """
    Send mu-law audio to Twilio as 20ms 'media' frames.
    - Sends first frame immediately (early playback).
    - Optionally pace subsequent frames by ~20ms to match realtime.
    """
    first = True
    send_interval = 0.02  # 20ms
    next_send = time.perf_counter()
    # Optional: prime Twilio jitter buffer with mu-law silence frames
    if prime_frames and prime_frames > 0:
        silent_frame = bytes([0xFF]) * MULAW_FRAME_BYTES
        for _ in range(prime_frames):
            if cancel and cancel.is_set():
                break
            payload = base64.b64encode(silent_frame).decode("utf-8")
            await websocket.send_json({
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": payload}
            })
            # do not pace priming frames to quickly fill buffer
    for frame in iter_mulaw_20ms_frames(audio_bytes):
        if cancel and cancel.is_set():
            break
        payload = base64.b64encode(frame).decode("utf-8")
        await websocket.send_json({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload}
        })
        if not pace_20ms:
            continue
        if first:
            first = False
            next_send = time.perf_counter() + send_interval
            continue
        # Precise pacing with drift correction
        next_send += send_interval
        now = time.perf_counter()
        sleep_dur = next_send - now
        if sleep_dur > 0:
            await asyncio.sleep(sleep_dur)
        elif sleep_dur < -0.03:
            # We're late by >30ms; reset schedule to avoid cumulative jitter
            next_send = time.perf_counter()


def crossfade_mulaw_segments(prev_tail: bytes, next_head: bytes, overlap_bytes: int = None) -> bytes:
    """
    Crossfade two adjacent mu-law segments to eliminate clicks at boundaries.
    Uses audioop for fast, lightweight processing (no numpy needed).
    
    Args:
        prev_tail: Last portion of previous chunk
        next_head: Complete next chunk
        overlap_bytes: Overlap size (default: 160 bytes = 20ms at 8kHz)
        
    Returns:
        Blended audio bytes
    """
    if not prev_tail and not next_head:
        return b""
    if overlap_bytes is None:
        overlap_bytes = MULAW_FRAME_BYTES  # 160 bytes = 20ms
    
    # If either segment too short, concatenate safely
    if not prev_tail or len(prev_tail) < overlap_bytes:
        return (prev_tail or b"") + (next_head or b"")
    if not next_head or len(next_head) < overlap_bytes:
        return (prev_tail or b"") + (next_head or b"")
    
    try:
        import audioop
        import struct
        
        # Extract overlap regions
        prev_overlap = prev_tail[-overlap_bytes:]
        next_overlap = next_head[:overlap_bytes]
        
        # Convert mu-law to 16-bit linear PCM
        prev_lin = audioop.ulaw2lin(prev_overlap, 2)
        next_lin = audioop.ulaw2lin(next_overlap, 2)
        
        # Ensure equal length
        n = min(len(prev_lin), len(next_lin))
        prev_lin = prev_lin[:n]
        next_lin = next_lin[:n]
        
        # Linear crossfade: prev fades out (1→0), next fades in (0→1)
        mixed = bytearray(n)
        denom = max((n // 2) - 1, 1)  # Avoid division by zero
        
        for i in range(0, n, 2):
            t = i // 2
            fade_out = 1.0 - (t / denom) if denom > 0 else 0.0
            fade_out = max(0.0, min(1.0, fade_out))
            fade_in = 1.0 - fade_out
            
            s1 = struct.unpack('<h', prev_lin[i:i+2])[0]
            s2 = struct.unpack('<h', next_lin[i:i+2])[0]
            s = int(s1 * fade_out + s2 * fade_in)
            s = max(-32768, min(32767, s))
            mixed[i:i+2] = struct.pack('<h', s)
        
        # Convert back to mu-law
        mixed_mulaw = audioop.lin2ulaw(bytes(mixed), 2)
        
        # Return: (prev without overlap) + (blended) + (next without overlap)
        return prev_tail[:-overlap_bytes] + mixed_mulaw + next_head[overlap_bytes:]
        
    except Exception as e:
        print(f"⚠️ Crossfade failed, using direct join: {e}")
        return prev_tail + next_head


def smart_chunk_text(text: str, max_words: int = 15) -> tuple[str, str]:
    """
    Smart text chunking that splits at natural pauses for smoother speech.
    Prefers splitting at sentence boundaries to maintain natural flow.
    
    Args:
        text: Text to split
        max_words: Maximum words in prefix chunk
        
    Returns:
        (prefix, suffix) tuple
    """
    if not text or not text.strip():
        return "", ""
    
    text = text.strip()
    words = text.split()
    
    # If text is short enough, return as-is
    if len(words) <= max_words:
        return text, ""
    
    # Try to split at sentence boundaries (., !, ?)
    sentence_endings = ['. ', '! ', '? ']
    best_split = None
    
    for ending in sentence_endings:
        parts = text.split(ending)
        if len(parts) > 1:
            prefix_candidate = parts[0] + ending.strip()
            prefix_words = len(prefix_candidate.split())
            
            # Use this split if it's within our word limit
            if prefix_words <= max_words and prefix_words > max_words * 0.5:
                best_split = (prefix_candidate, text[len(prefix_candidate):].strip())
                break
    
    # If no good sentence split, try comma split
    if not best_split and ', ' in text:
        parts = text.split(', ', 1)
        prefix_candidate = parts[0] + ','
        prefix_words = len(prefix_candidate.split())
        
        if prefix_words <= max_words and prefix_words > 5:
            best_split = (prefix_candidate, parts[1].strip())
    
    # Fallback: split at word count
    if not best_split:
        prefix = " ".join(words[:max_words])
        suffix = " ".join(words[max_words:])
        best_split = (prefix, suffix)
    
    return best_split


async def generate_mulaw_tts(text: str, lang: str = "en", voice: str = "female", use_chirp3_hd: bool = True, speaking_rate: float = 0.95) -> bytes:
    """
    Generate mu-law (8kHz) TTS audio using Chirp 3: HD model.
    Optimized for word-by-word streaming with caching.
    """
    # Skip empty text
    if not text or not text.strip():
        return b''
    
    # Cache key aligned with existing cache strategy
    cache_key = generate_cache_key(text.strip(), lang, voice, use_chirp3_hd, "mulaw")

    if cache_key in audio_cache:
        return audio_cache[cache_key]

    # Use 8kHz MULAW for Twilio with Chirp 3: HD model - Optimized for small chunks
    audio_content = google_tts_service.text_to_speech(
        text=text.strip(),
        language=lang,
        voice_type=voice,
        speaking_rate=0.95,   # Fixed rate for consistent low latency
        pitch=0.0,
        output_format="mulaw",
        use_chirp3_hd=use_chirp3_hd
    )

    # Cache for instant reuse (especially useful for repeated words/phrases)
    audio_cache[cache_key] = audio_content
    return audio_content


def build_streaming_twiml(call_session_id: str, agent_id: str) -> str:
    """
    Replace <Play> with <Start><Stream> to enable realtime MULAW streaming.
    Configure Twilio edge/region via settings.TWILIO_EDGE if available.
    """
    from twilio.twiml.voice_response import VoiceResponse, Start, Stream, Parameter
    
    # Your public WebSocket endpoint that handles Twilio Media Streams:
    # Example: wss://your-domain.com/api/v1/stream/ws/bidirectional/{callSessionId}/{agentId}
    ws_url = f"{settings.WEBHOOK_BASE_URL.replace('http', 'ws')}/api/v1/stream/ws/bidirectional/{call_session_id}/{agent_id}"
    if ws_url.startswith("ws://"):
        ws_url = "wss://" + ws_url[len("ws://"):]  # enforce TLS for Twilio

    edge = getattr(settings, "TWILIO_EDGE", None)  # e.g., "ashburn", "singapore", "dublin"
    vr = VoiceResponse()
    with vr.start() as s:
        stream = Stream(url=ws_url, name=f"tts-stream-{agent_id}")
        # Forward region hint to your WS (for observability); set real Twilio edge via account/call config.
        if edge:
            stream.parameter(Parameter(name="edge", value=edge))
        s.append(stream)

    return str(vr)


def build_tts_only_twiml(call_session_id: str, agent_id: str, record_callback_url: str) -> str:
    """
    Build TwiML for TTS-only WebSocket streaming + Recording for next input
    
    Flow:
    1. Connect to TTS-only WebSocket
    2. WebSocket auto-plays pending TTS from metadata
    3. After playback, start recording for next user input
    
    Args:
        call_session_id: Call session UUID
        agent_id: Agent UUID
        record_callback_url: URL for recording callback
    
    Returns:
        TwiML string
    """
    from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
    
    # Build TTS-only WebSocket URL
    ws_url = f"{settings.WEBHOOK_BASE_URL.replace('http', 'ws')}/api/v1/voice/ws/tts-only/{call_session_id}/{agent_id}"
    if ws_url.startswith("ws://"):
        ws_url = "wss://" + ws_url[len("ws://"):]  # enforce TLS
    
    vr = VoiceResponse()
    
    # Connect to TTS-only WebSocket for streaming playback
    connect = Connect()
    stream = Stream(url=ws_url, name=f"tts-only-{agent_id}")
    connect.append(stream)
    vr.append(connect)
    
    # After TTS playback, start recording for next user input
    vr.record(
        action=record_callback_url,
        method='POST',
        timeout=3,  # Fast detection
        max_length=120,
        play_beep=False,
        trim='do-not-trim',
        recording_status_callback=record_callback_url,
        recording_status_callback_method='POST',
        transcribe=False
    )
    
    return str(vr)


class BidirectionalStreamHandler:
    """Handles real-time bidirectional voice streaming"""
    
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
        self._stt_session = None
        self._stt_task = None
        # Ultra-aggressive interim processing state (40% confidence)
        self._last_interim_text = ""
        self._last_interim_sent_ts = 0.0
        self._min_interim_words = 1  # speak sooner on shorter interim
        self._min_interim_confidence = 0.40  # ULTRA-AGGRESSIVE: process 40% confidence
        self._min_interim_interval_sec = 0.10  # ULTRA-AGGRESSIVE: 100ms throttle (was 200ms)
        
        # TTS (Output) state - Parallel Pipeline
        self.tts_queue = asyncio.Queue()     # Queue for parallel TTS processing
        self.is_speaking = False
        self._tts_cancel = asyncio.Event()   # barge-in cancel signal
        self._tts_lock = asyncio.Lock()      # serialize TTS streams
        self._tts_worker_task = None         # Background TTS worker
        self._tts_generation_tasks = []      # Track parallel TTS generation
        
        # Session data
        self.call_session = None
        self.agent = None
        self._load_session_data()
        
        # Pre-warm Google TTS client to avoid first-call penalty
        try:
            google_tts_service.get_client()
        except Exception as e:
            print(f"⚠️ TTS pre-warm failed: {e}")
            sys.stdout.flush()

        # Start parallel TTS pipeline worker
        self._tts_worker_task = asyncio.create_task(self._tts_pipeline_worker())
        
        print(f"✅ Bidirectional stream handler initialized (Google streaming STT + Streaming TTS)")
        print(f"⚡ ULTRA-AGGRESSIVE MODE: 40% confidence, 100ms throttle")
        print(f"🔄 PARALLEL TTS PIPELINE: Started background worker")
        sys.stdout.flush()
    
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
                print(f"✅ Loaded agent: {self.agent.name if self.agent else 'Unknown'}")
                sys.stdout.flush()
        except Exception as e:
            print(f"⚠️ Error loading session data: {e}")
            sys.stdout.flush()
    
    async def _tts_pipeline_worker(self):
        """
        Background worker for parallel TTS pipeline (Vapi-style).
        
        Processes TTS chunks from queue while new chunks are being generated:
        - LLM Chunk 1 → TTS generates → plays
        - LLM Chunk 2 → TTS generates (parallel) → queued
        - LLM Chunk 3 → TTS generates (parallel) → queued
        """
        print("🔄 TTS Pipeline Worker: Started")
        sys.stdout.flush()
        
        try:
            while True:
                # Get next TTS task from queue
                task = await self.tts_queue.get()
                
                # Check for shutdown signal
                if task is None:
                    print("🔄 TTS Pipeline Worker: Shutdown signal received")
                    sys.stdout.flush()
                    break
                
                # Check if cancelled (barge-in)
                if self._tts_cancel.is_set():
                    print("🛑 TTS Pipeline Worker: Skipping chunk due to barge-in")
                    sys.stdout.flush()
                    self.tts_queue.task_done()
                    continue
                
                try:
                    text = task.get("text", "")
                    chunk_id = task.get("chunk_id", "unknown")
                    
                    if not text or not text.strip():
                        self.tts_queue.task_done()
                        continue
                    
                    print(f"🔄 TTS Pipeline: Processing chunk {chunk_id}: '{text[:30]}...'")
                    sys.stdout.flush()
                    
                    # Generate and stream TTS (this is the blocking part)
                    await self._stream_tts_chunk(text)
                    
                    print(f"✅ TTS Pipeline: Completed chunk {chunk_id}")
                    sys.stdout.flush()
                    
                except Exception as e:
                    print(f"❌ TTS Pipeline Worker: Error processing chunk: {e}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                finally:
                    self.tts_queue.task_done()
        
        except Exception as e:
            print(f"❌ TTS Pipeline Worker: Fatal error: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    
    
    
    async def handle_media_message(self, message: dict):
        """Handle incoming audio from Twilio and feed to Google streaming STT"""
        try:
            media = message.get("media", {})
            payload = media.get("payload")
            
            if not payload:
                return
            
            # Decode audio (MULAW from Twilio)
            audio_data = base64.b64decode(payload)

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
                        print(f"❌ STT streaming start error: {e}")
                        sys.stdout.flush()
                    # Drain any remaining
                
                # Start the session in background and concurrently read results
                async def reader_loop():
                    while True:
                        result = await self._stt_session.get_result()
                        if not result:
                            continue
                        if result.get("error"):
                            print(f"❌ STT error: {result['error']}")
                            sys.stdout.flush()
                            continue
                        transcript = (result.get("transcript") or "").strip()
                        if not transcript:
                            continue
                        is_final = bool(result.get("is_final"))
                        confidence = float(result.get("confidence") or 0.0)
                        if is_final:
                            print(f"📝 Final STT: '{transcript}' ({confidence:.2f})")
                            sys.stdout.flush()
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
            print(f"❌ Error handling media: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    
    # Removed chunk-based STT processing; relying on Google streaming endpointing
    
    async def _process_transcript(self, transcript: str, confidence: float):
        """Process a transcript (final result)"""
        try:
            if not transcript or confidence < 0.3:
                print(f"⚠️ Skipping low-confidence transcript (confidence: {confidence:.2f})")
                sys.stdout.flush()
                return
            
            # Add to transcript
            await self._add_to_transcript("client", transcript, "speech", confidence)
            
            # Generate and stream response
            await self.generate_and_stream_response(transcript, confidence)
            
        except Exception as e:
            print(f"❌ Error processing transcript: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

    async def _maybe_process_interim(self, transcript: str, confidence: float):
        """
        ULTRA-AGGRESSIVE interim processing for minimal latency.
        Processes interim STT results with 40% confidence to start LLM generation ASAP.
        """
        try:
            if not transcript:
                return
            
            # Basic gating: confidence and minimum words (ULTRA-AGGRESSIVE)
            word_count = len(transcript.split())
            if confidence < self._min_interim_confidence or word_count < self._min_interim_words:
                # Still log interim for observability
                print(f"⌛ Interim (gated) [{confidence:.2f}]: '{transcript[:60]}...'")
                sys.stdout.flush()
                return
            
            # Barge-in: if we are speaking and user starts talking, cancel TTS immediately
            # Use lower threshold for barge-in since we're ultra-aggressive
            if self.is_speaking and confidence >= 0.45:  # ULTRA-AGGRESSIVE: 45% barge-in
                if not self._tts_cancel.is_set():
                    print(f"🛑 Barge-in: cancelling current TTS (confidence: {confidence:.2f})")
                    sys.stdout.flush()
                    self._tts_cancel.set()
                    # Clear TTS queue to stop all pending chunks
                    while not self.tts_queue.empty():
                        try:
                            self.tts_queue.get_nowait()
                            self.tts_queue.task_done()
                        except:
                            break
                    print("🛑 Barge-in: cleared TTS queue")
                    sys.stdout.flush()
                return
            
            # Ultra-aggressive throttling: only 100ms between triggers
            now = asyncio.get_event_loop().time()
            if (now - self._last_interim_sent_ts) < self._min_interim_interval_sec:
                print(f"⌛ Interim (throttled): '{transcript[:60]}...'")
                sys.stdout.flush()
                return
            
            # ULTRA-AGGRESSIVE: Process even small advances (no minimum word requirement)
            # This ensures we start LLM generation as soon as possible
            if self._last_interim_text and transcript.startswith(self._last_interim_text):
                advanced = transcript[len(self._last_interim_text):].strip()
                # Skip only if there's literally no new content
                if not advanced:
                    print(f"⌛ Interim (no-advance): '{transcript[:60]}...'")
                    sys.stdout.flush()
                    return
                # Process even single character advances for ultra-low latency
                print(f"⚡ ULTRA-AGGRESSIVE: Processing advance '{advanced}' (total: '{transcript[:60]}')")
                sys.stdout.flush()
            
            # Passed heuristics → process immediately to start LLM generation
            print(f"⚡⚡ ULTRA-AGGRESSIVE INTERIM [{confidence:.2f}]: '{transcript[:60]}'")
            sys.stdout.flush()
            self._last_interim_text = transcript
            self._last_interim_sent_ts = now
            await self.generate_and_stream_response(transcript, confidence)
        except Exception as e:
            print(f"❌ Error in interim processing: {e}")
            sys.stdout.flush()
    
    async def generate_and_stream_response(self, user_text: str, confidence: float):
        """
        Generate AI response and stream TTS in real-time WITH conversation history.
        Uses PARALLEL TTS PIPELINE (Vapi-style) for ultra-low latency.
        """
        try:
            from datetime import datetime, timezone
            import json
            
            # Reset cancel flag for new response generation
            self._tts_cancel.clear()
            
            print(f"🤖 Generating streaming response for: '{user_text}'")
            sys.stdout.flush()
            
            # Build conversation context from transcript
            conversation_history = []
            if self.call_session and self.call_session.call_transcript:
                try:
                    conversation_history = json.loads(self.call_session.call_transcript) if isinstance(self.call_session.call_transcript, str) else self.call_session.call_transcript
                except:
                    conversation_history = []
            
            # Build history text (last 6 messages for context) - handle different formats
            history_text = ""
            if conversation_history:
                try:
                    history_lines = []
                    for msg in conversation_history[-6:]:
                        if isinstance(msg, dict):
                            # Handle both 'content' and 'message' keys
                            role = msg.get('role', 'unknown')
                            content = msg.get('content') or msg.get('message', '')
                            if content:
                                history_lines.append(f"{role.capitalize()}: {content}")
                    history_text = "\n".join(history_lines)
                except Exception as e:
                    print(f"⚠️ Error building history text: {e}")
                    sys.stdout.flush()
                    history_text = ""
            
            # Build system prompt with agent personality + history
            agent_name = self.agent.name if self.agent and self.agent.name else "AI Assistant"
            system_prompt = f"""You are {agent_name}, a real person having a phone conversation.

Guidelines:
- Speak naturally and conversationally
- Keep responses brief (1-2 sentences)
- Remember previous conversation - don't repeat questions
- Be friendly and engaging
- Talk like a real human, not a robot

Previous conversation:
{history_text}

IMPORTANT: Use the conversation history above. Don't ask questions you already asked. Continue the conversation naturally."""
            
            # Get agent's configured model and provider
            llm_service = None
            model_name = "gemini-1.5-flash"  # Default fallback
            api_key = None
            temperature = 0.5
            max_tokens = 40
            
            if self.agent and self.agent.model:
                model_name = self.agent.model.model_name
                
                # Decrypt API key if available
                if self.agent.model.api_key:
                    try:
                        from app.core.security import decrypt_api_key
                        api_key = decrypt_api_key(self.agent.model.api_key)
                        print(f"🔑 Using decrypted model-specific API key")
                    except Exception as e:
                        print(f"⚠️ Failed to decrypt API key: {e}")
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
                        print(f"🤖 Using OpenAI: {model_name}")
                    elif "gemini" in provider_name or "google" in provider_name:
                        llm_service = gemini_service
                        print(f"🤖 Using Gemini: {model_name}")
                    else:
                        # Default to Gemini
                        llm_service = gemini_service
                        print(f"⚠️ Unknown provider '{provider_name}', defaulting to Gemini")
                else:
                    llm_service = gemini_service
                    print(f"⚠️ No provider configured, defaulting to Gemini")
            else:
                # Fallback to Gemini
                llm_service = gemini_service
                print(f"⚠️ No model configured for agent, using default Gemini")
            
            sys.stdout.flush()
            
            # Stream LLM output and QUEUE for PARALLEL TTS PIPELINE (Vapi-style)
            chunk_counter = 0
            
            async def try_stream(service, model: str, api_key_override: str = None) -> str:
                nonlocal chunk_counter
                response_accum = ""
                phrase_buf = ""
                last_flush = asyncio.get_event_loop().time()

                async for chunk in service.stream_text(
                    prompt=user_text,
                    system_prompt=system_prompt,  # NOW WITH HISTORY!
                    model_name=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    api_key=api_key_override
                ):
                    if not chunk:
                        continue
                    # If barge-in requested, stop generating more audio for this response
                    if self._tts_cancel.is_set():
                        print("🛑 Barge-in: aborting current LLM stream")
                        sys.stdout.flush()
                        break

                    response_accum += chunk
                    phrase_buf += chunk
                    word_count = len(phrase_buf.split())

                    # ULTRA-AGGRESSIVE: Start TTS on just 3-5 words OR punctuation OR timeout
                    now = asyncio.get_event_loop().time()
                    has_punct = any(p in phrase_buf for p in [".", "?", "!", ",", ";", "—", ":", "-"]) 

                    # ⚡⚡ ULTRA-AGGRESSIVE: 3-5 words OR punctuation OR 80ms timeout
                    if word_count >= 3 or has_punct or (now - last_flush) >= 0.08:
                        to_speak = phrase_buf.strip()
                        if to_speak and not self._tts_cancel.is_set():
                            # PARALLEL PIPELINE: Queue chunk instead of blocking
                            chunk_counter += 1
                            await self.tts_queue.put({
                                "text": to_speak,
                                "chunk_id": chunk_counter
                            })
                            print(f"🔄 Queued TTS chunk {chunk_counter} ({word_count} words): '{to_speak[:30]}...'")
                            sys.stdout.flush()
                        phrase_buf = ""
                        last_flush = now

                # flush any tail
                tail = phrase_buf.strip()
                if tail and not self._tts_cancel.is_set():
                    chunk_counter += 1
                    await self.tts_queue.put({
                        "text": tail,
                        "chunk_id": chunk_counter
                    })
                    print(f"🔄 Queued TTS chunk {chunk_counter} (final): '{tail[:30]}...'")
                    sys.stdout.flush()

                return response_accum.strip()

            final_text = None
            try:
                # Use agent's configured model and service
                final_text = await try_stream(llm_service, model_name, api_key)
            except Exception as e1:
                print(f"⚠️ Streaming with {model_name} failed: {e1}")
                sys.stdout.flush()
                # Fallback: try alternate service
                try:
                    if llm_service == openai_service:
                        # Fallback to Gemini
                        print("⚠️ Falling back to Gemini gemini-1.5-flash")
                        sys.stdout.flush()
                        final_text = await try_stream(gemini_service, "gemini-1.5-flash", None)
                    else:
                        # Fallback to OpenAI
                        print("⚠️ Falling back to OpenAI gpt-3.5-turbo")
                        sys.stdout.flush()
                        final_text = await try_stream(openai_service, "gpt-3.5-turbo", None)
                except Exception as e2:
                    print(f"⚠️ Streaming with fallback model failed: {e2}")
                    sys.stdout.flush()
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
                                "chunk_id": chunk_counter
                            })
                            print(f"🔄 Queued fallback TTS chunk {chunk_counter}")
                            sys.stdout.flush()
                    except Exception as e3:
                        print(f"⚠️ All fallbacks failed: {e3}")
                        sys.stdout.flush()
                        # Ultimate fallback response
                        final_text = "I apologize, I'm having trouble responding right now. Could you please repeat that?"
                        chunk_counter += 1
                        await self.tts_queue.put({
                            "text": final_text,
                            "chunk_id": chunk_counter
                        })

            if final_text:
                await self._add_to_transcript("agent", final_text, "agent_response")
        
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            sys.stdout.flush()
    
    async def _stream_tts_chunk(self, text: str):
        """
        Generate and stream a single TTS chunk (used by parallel pipeline worker).
        Simplified version without the complex prefix/suffix splitting.
        Note: Does NOT clear cancel flag - respects barge-in for entire queue.
        """
        try:
            from datetime import datetime, timezone
            
            if not text or not text.strip():
                return
            
            # Check if already cancelled before acquiring lock
            if self._tts_cancel.is_set():
                print(f"🛑 Skipping TTS chunk due to barge-in: '{text[:30]}...'")
                sys.stdout.flush()
                return
            
            async with self._tts_lock:
                self.is_speaking = True
                try:
                    lang = self.agent.language if self.agent and self.agent.language else "en"
                    voice = self.agent.voice_type if self.agent and self.agent.voice_type else "female"
                    clean = text.strip()
                    
                    print(f"🎵 Generating TTS for chunk: '{clean[:40]}...'")
                    sys.stdout.flush()
                    
                    # Generate TTS audio
                    tts_start = datetime.now(timezone.utc)
                    audio_bytes = await generate_mulaw_tts(
                        text=clean,
                        lang=lang,
                        voice=voice,
                        use_chirp3_hd=True,
                        speaking_rate=0.95
                    )
                    tts_gen_time = (datetime.now(timezone.utc) - tts_start).total_seconds()
                    print(f"⏱️ TTS generation: {tts_gen_time:.3f}s for '{clean[:20]}...'")
                    sys.stdout.flush()
                    
                    # Stream to Twilio immediately
                    if audio_bytes and not self._tts_cancel.is_set():
                        await stream_mulaw_bytes_over_twilio(
                            websocket=self.websocket,
                            stream_sid=self.stream_sid,
                            audio_bytes=audio_bytes,
                            pace_20ms=True,
                            cancel=self._tts_cancel,
                            prime_frames=1,
                        )
                        print(f"✅ Streamed {len(audio_bytes)} bytes")
                        sys.stdout.flush()
                finally:
                    self.is_speaking = False
        
        except Exception as e:
            print(f"❌ Error streaming TTS chunk: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    
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
                    print(f"🎵 Streaming TTS chunk: '{clean[:30]}...'")
                    sys.stdout.flush()

                    # Smart chunking at sentence boundaries (10 words for natural flow)
                    prefix, suffix = smart_chunk_text(clean, max_words=10)

                    # Begin generating suffix in parallel (if any)
                    suffix_task = asyncio.create_task(
                        generate_mulaw_tts(text=suffix, lang=lang, voice=voice, use_chirp3_hd=True, speaking_rate=0.95)
                    ) if suffix else None

                    # Generate prefix audio immediately
                    tts_start = datetime.now(timezone.utc)
                    prefix_audio = await generate_mulaw_tts(text=prefix, lang=lang, voice=voice, use_chirp3_hd=True, speaking_rate=0.95)
                    tts_gen_time = (datetime.now(timezone.utc) - tts_start).total_seconds()
                    print(f"⏱️ TTS(first) latency: {tts_gen_time:.3f}s for '{prefix[:20]}...'")
                    sys.stdout.flush()

                    # Hold back last 20ms for crossfade with next chunk
                    overlap_bytes = MULAW_FRAME_BYTES  # 160 bytes = 20ms
                    if len(prefix_audio) > overlap_bytes:
                        prefix_main = prefix_audio[:-overlap_bytes]
                        prefix_tail = prefix_audio[-overlap_bytes:]
                    else:
                        prefix_main = prefix_audio
                        prefix_tail = b""
                    
                    # Stream main part immediately
                    if prefix_main:
                        await stream_mulaw_bytes_over_twilio(
                            websocket=self.websocket,
                            stream_sid=self.stream_sid,
                            audio_bytes=prefix_main,
                            pace_20ms=True,
                            cancel=self._tts_cancel,
                            prime_frames=1,  # Smooth start with 20ms buffer
                        )

                    # Stream remainder when ready and not cancelled
                    if suffix_task and not self._tts_cancel.is_set():
                        try:
                            suffix_audio = await suffix_task
                        except Exception as e:
                            print(f"⚠️ TTS remainder generation failed: {e}")
                            sys.stdout.flush()
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
                                print(f"⚡ Streamed remainder with crossfade ({len(merged)} bytes)")
                                sys.stdout.flush()
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
            print(f"❌ Error streaming TTS chunk '{text[:20]}...': {e}")
            sys.stdout.flush()
    
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
            
            print(f"📤 Sent {len(audio_data)} bytes to Twilio (20ms chunks)")
            sys.stdout.flush()
        
        except Exception as e:
            print(f"❌ Error sending audio: {e}")
            sys.stdout.flush()
    
    async def _add_to_transcript(
        self,
        role: str,
        message: str,
        message_type: str = "speech",
        confidence: Optional[float] = None
    ):
        """Add message to transcript"""
        try:
            if not self.call_session:
                return
            
            await transcript_service.add_and_broadcast_message(
                db=self.db,
                call_session_id=self.call_session.id,
                role=role,
                message=message,
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
            print(f"❌ Error adding to transcript: {e}")
            sys.stdout.flush()
    
    async def handle_start_message(self, message: dict):
        """Handle stream start"""
        try:
            self.stream_sid = message.get("streamSid")
            start = message.get("start", {})
            self.call_sid = start.get("callSid")
            
            print("=" * 80)
            print(f"🎙️ BIDIRECTIONAL STREAM STARTED")
            print(f"Stream SID: {self.stream_sid}")
            print(f"Call SID: {self.call_sid}")
            print(f"Agent: {self.agent.name if self.agent else 'Unknown'}")
            print(f"📡 Real-time STT + TTS streaming enabled")
            print("=" * 80)
            sys.stdout.flush()
            
            # Broadcast "in-progress" status when media stream starts
            if self.call_session:
                try:
                    # Update call session status to "in-progress" if not already
                    if self.call_session.status != "in-progress":
                        self.call_session.status = "in-progress"
                        
                        # Set start time when call becomes in-progress
                        if not self.call_session.start_time:
                            self.call_session.start_time = datetime.now(timezone.utc)
                        
                        self.db.commit()
                        print(f"✅ Updated call session status to 'in-progress'")
                    
                    # Broadcast the in-progress status via WebSocket
                    await broadcast_call_status_update(
                        call_session_id=str(self.call_session.id),
                        status="in-progress",
                        metadata={
                            "call_sid": self.call_sid,
                            "stream_sid": self.stream_sid,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "message": "Connected"
                        }
                    )
                    print(f"✅ Broadcasted 'in-progress' status via WebSocket (media stream started)")
                except Exception as e:
                    print(f"❌ Failed to broadcast in-progress status: {e}")
                    import traceback
                    traceback.print_exc()
        
        except Exception as e:
            print(f"❌ Error handling start: {e}")
            sys.stdout.flush()
    
    async def handle_stop_message(self, message: dict):
        """Handle stream stop"""
        try:
            print("=" * 80)
            print(f"🛑 BIDIRECTIONAL STREAM STOPPED")
            print(f"Stream SID: {self.stream_sid}")
            print("=" * 80)
            sys.stdout.flush()
            
            # Stop TTS pipeline worker
            try:
                if self._tts_worker_task:
                    await self.tts_queue.put(None)  # Shutdown signal
                    await asyncio.wait_for(self._tts_worker_task, timeout=2.0)
                    print("✅ TTS pipeline worker stopped")
                    sys.stdout.flush()
            except asyncio.TimeoutError:
                print("⚠️ TTS pipeline worker shutdown timeout")
                sys.stdout.flush()
            except Exception as e:
                print(f"⚠️ Error stopping TTS worker: {e}")
                sys.stdout.flush()
            
            # Close STT session
            try:
                if self._stt_session:
                    self._stt_session.finish()
                if self._stt_task:
                    await asyncio.sleep(0)  # yield
            except Exception:
                pass
        
        except Exception as e:
            print(f"❌ Error handling stop: {e}")
            sys.stdout.flush()


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
    print("=" * 80)
    print(f"🎙️ Bidirectional WebSocket Connection")
    print(f"Call Session: {callSessionId}")
    print(f"Agent: {agentId}")
    print("=" * 80)
    sys.stdout.flush()
    
    # Accept connection
    try:
        await websocket.accept()
        print(f"✅ WebSocket accepted")
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ Failed to accept WebSocket: {e}")
        sys.stdout.flush()
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
                print("✅ Twilio connected")
                sys.stdout.flush()
            
            elif event == "start":
                await handler.handle_start_message(message)
            
            elif event == "media":
                media_count += 1
                if media_count % 100 == 0:
                    print(f"📦 Processed {media_count} media packets")
                    sys.stdout.flush()
                await handler.handle_media_message(message)
            
            elif event == "stop":
                await handler.handle_stop_message(message)
                break
            
            elif event == "mark":
                pass  # Synchronization marks
    
    except WebSocketDisconnect:
        print(f"📡 WebSocket disconnected")
        sys.stdout.flush()
    
    except Exception as e:
        print(f"❌ Error in bidirectional stream: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    
    finally:
        db.close()
        print(f"🔚 Bidirectional stream closed")
        sys.stdout.flush()


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
    print("=" * 80)
    print(f"🎵 TTS-ONLY WebSocket Connection")
    print(f"Call Session: {callSessionId}")
    print(f"Agent: {agentId}")
    print("=" * 80)
    sys.stdout.flush()
    
    try:
        await websocket.accept()
        print(f"✅ WebSocket accepted (TTS-only mode)")
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ Failed to accept WebSocket: {e}")
        sys.stdout.flush()
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
            if agent:
                print(f"✅ Agent: {agent.name}")
                sys.stdout.flush()
    except Exception as e:
        print(f"⚠️ Error loading agent: {e}")
        sys.stdout.flush()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            event = message.get("event")
            
            if event == "connected":
                print("✅ Twilio connected to TTS-only stream")
                sys.stdout.flush()
            
            elif event == "start":
                stream_sid = message.get("streamSid")
                print(f"🎙️ TTS Stream started - SID: {stream_sid}")
                sys.stdout.flush()
                
                # Auto-retrieve and play pending TTS from call session metadata
                if call_session and call_session.call_metadata:
                    pending_tts = call_session.call_metadata.get("pending_tts")
                    if pending_tts:
                        text = pending_tts.get("text", "")
                        lang = pending_tts.get("lang", agent.language if agent else "en")
                        voice = pending_tts.get("voice", agent.voice_type if agent else "female")
                        
                        if text:
                            print(f"🎵 Auto-playing pending TTS: '{text[:50]}...'")
                            sys.stdout.flush()
                            
                            # Generate MULAW TTS with Chirp 3: HD
                            audio_bytes = await generate_mulaw_tts(
                                text=text,
                                lang=lang,
                                voice=voice,
                                use_chirp3_hd=True
                            )
                            
                            # Stream in 20ms chunks
                            await stream_mulaw_bytes_over_twilio(
                                websocket=websocket,
                                stream_sid=stream_sid,
                                audio_bytes=audio_bytes,
                                pace_20ms=True
                            )
                            
                            # Clear pending TTS
                            call_session.call_metadata.pop("pending_tts", None)
                            db.commit()
                            
                            print(f"✅ Auto-playback complete, cleared pending TTS")
                            sys.stdout.flush()
            
            elif event == "play_tts":
                # Custom event to trigger TTS playback
                text = message.get("text", "")
                lang = message.get("lang", agent.language if agent else "en")
                voice = message.get("voice", agent.voice_type if agent else "female")
                
                if text and stream_sid:
                    print(f"🎵 Playing TTS: '{text[:50]}...'")
                    sys.stdout.flush()
                    
                    # Generate MULAW TTS with Chirp 3: HD
                    audio_bytes = await generate_mulaw_tts(
                        text=text,
                        lang=lang,
                        voice=voice,
                        use_chirp3_hd=True
                    )
                    
                    # Stream in 20ms chunks
                    await stream_mulaw_bytes_over_twilio(
                        websocket=websocket,
                        stream_sid=stream_sid,
                        audio_bytes=audio_bytes,
                        pace_20ms=True
                    )
                    
                    # Send completion event
                    await websocket.send_json({
                        "event": "tts_complete",
                        "text_length": len(text),
                        "audio_bytes": len(audio_bytes)
                    })
                    print(f"✅ TTS playback complete")
                    sys.stdout.flush()
            
            elif event == "media":
                # Ignore incoming media (we're TTS-only)
                pass
            
            elif event == "stop":
                print("🛑 TTS stream stopped")
                sys.stdout.flush()
                break
            
            elif event == "mark":
                pass  # Synchronization marks
    
    except WebSocketDisconnect:
        print(f"📡 WebSocket disconnected (TTS-only)")
        sys.stdout.flush()
    
    except Exception as e:
        print(f"❌ Error in TTS-only stream: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    
    finally:
        db.close()
        print(f"🔚 TTS-only stream closed")
        sys.stdout.flush()