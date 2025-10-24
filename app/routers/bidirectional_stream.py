"""
Bidirectional WebSocket for Real-time Voice AI
Handles both STT (incoming audio) and TTS (outgoing audio) simultaneously
Optimized for ultra-low latency (<3s response time)
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
import json
import base64
import asyncio
from typing import Optional, Dict, Iterable
from datetime import datetime, timezone
import uuid
import sys
import audioop
import math

# Using advanced adaptive energy-based VAD (pure Python, no dependencies)
# VAPI-style voice activity detection with noise floor estimation

from app.services.google_stt_service import google_stt_service
from app.services.google_tts_service import google_tts_service
from app.services.call_session_service import call_session_service
from app.services.agent_service import agent_service
from app.services.voice_logging_service import VoiceLoggingService
from app.services.transcript_service import transcript_service
from app.core.config import settings
from app.routers.tts_audio import audio_cache, generate_cache_key

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


async def stream_mulaw_bytes_over_twilio(websocket, stream_sid: str, audio_bytes: bytes, pace_20ms: bool = True):
    """
    Send mu-law audio to Twilio as 20ms 'media' frames.
    - Sends first frame immediately (early playback).
    - Optionally pace subsequent frames by ~20ms to match realtime.
    """
    first = True
    for frame in iter_mulaw_20ms_frames(audio_bytes):
        payload = base64.b64encode(frame).decode("utf-8")
        await websocket.send_json({
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload}
        })
        if first:
            first = False
            # Early playback: no initial sleep
        elif pace_20ms:
            await asyncio.sleep(CHUNK_DURATION_SEC)


async def generate_mulaw_tts(text: str, lang: str = "en", voice: str = "female", use_gemini_flash: bool = True) -> bytes:
    """
    Generate mu-law (8kHz) TTS audio using the existing Google TTS service.
    Caches audio for instant reuse.
    """
    # Cache key aligned with existing cache strategy
    cache_key = generate_cache_key(text, lang, voice, use_gemini_flash, "mulaw")

    if cache_key in audio_cache:
        return audio_cache[cache_key]

    # Use 8kHz MULAW for Twilio
    audio_content = google_tts_service.text_to_speech(
        text=text,
        language=lang,
        voice_type=voice,
        speaking_rate=1.0,   # clear at 8kHz MULAW
        pitch=0.0,
        output_format="mulaw",
        use_gemini_flash=use_gemini_flash
    )

    audio_cache[cache_key] = audio_content
    return audio_content


def build_streaming_twiml(call_session_id: str, agent_id: str) -> str:
    """
    Replace <Play> with <Start><Stream> to enable realtime MULAW streaming.
    Configure Twilio edge/region via settings.TWILIO_EDGE if available.
    """
    from twilio.twiml.voice_response import VoiceResponse, Start, Stream, Parameter
    
    # Your public WebSocket endpoint that handles Twilio Media Streams:
    # Example: wss://your-domain.com/api/v1/voice/ws/bidirectional/{callSessionId}/{agentId}
    ws_url = f"{settings.WEBHOOK_BASE_URL.replace('http', 'ws')}/api/v1/voice/ws/bidirectional/{call_session_id}/{agent_id}"
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
        
        # STT (Input) state
        self.audio_buffer = []
        self.stream_sid = None
        self.call_sid = None
        self.current_speech = ""
        self.speech_active = False
        self.silence_counter = 0
        
        # VAPI-style VAD settings: 0.9 seconds silence detection
        # Twilio sends 20ms packets (50 packets/second)
        # 0.9 seconds = 45 packets
        self.silence_threshold = 45  # 0.9 seconds of silence
        
        # Advanced Adaptive VAD (pure Python - works everywhere!)
        # Uses RMS energy with adaptive noise floor estimation
        self.frame_duration_ms = 20  # Twilio sends 20ms frames
        self.sample_rate = 8000  # Twilio MULAW is 8kHz
        
        # Adaptive VAD parameters
        self.noise_floor = 0.0  # Dynamic noise floor
        self.noise_samples = []  # Recent silence frames for noise estimation
        self.max_noise_samples = 10  # Track last 10 silence frames
        self.speech_multiplier = 3.0  # Speech must be 3x louder than noise
        self.min_speech_energy = 200  # Minimum absolute RMS for speech
        self.calibration_frames = 0  # Frames for initial calibration
        self.max_calibration_frames = 25  # Calibrate for 0.5 seconds
        
        # TTS (Output) state
        self.tts_queue = asyncio.Queue()
        self.is_speaking = False
        
        # Session data
        self.call_session = None
        self.agent = None
        self._load_session_data()
        
        print(f"✅ Bidirectional stream handler initialized (Adaptive VAD, 0.9s silence threshold)")
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
    
    def _calculate_rms_energy(self, audio_data: bytes) -> float:
        """Calculate RMS (Root Mean Square) energy of audio frame - accurate speech detection"""
        try:
            # Convert MULAW to LINEAR16 PCM for accurate energy calculation
            pcm_data = audioop.ulaw2lin(audio_data, 2)  # 2 = 16-bit samples
            
            # Calculate RMS energy from 16-bit PCM samples
            # Convert bytes to list of 16-bit integers
            samples = []
            for i in range(0, len(pcm_data), 2):
                if i + 1 < len(pcm_data):
                    # Convert two bytes to signed 16-bit integer (little-endian)
                    sample = int.from_bytes(pcm_data[i:i+2], byteorder='little', signed=True)
                    samples.append(sample)
            
            if not samples:
                return 0.0
            
            # Calculate RMS: sqrt(mean(sample^2))
            mean_square = sum(s * s for s in samples) / len(samples)
            rms = math.sqrt(mean_square)
            
            return rms
        
        except Exception as e:
            # Fallback: simple amplitude-based energy
            if len(audio_data) > 0:
                return sum(abs(b - 127) for b in audio_data) / len(audio_data)
            return 0.0
    
    def _update_noise_floor(self, energy: float):
        """Update adaptive noise floor estimation"""
        # Add to noise samples
        self.noise_samples.append(energy)
        
        # Keep only recent samples
        if len(self.noise_samples) > self.max_noise_samples:
            self.noise_samples.pop(0)
        
        # Calculate noise floor as average of recent low-energy frames
        if len(self.noise_samples) >= 3:
            # Use median to be robust against outliers
            sorted_samples = sorted(self.noise_samples)
            median_idx = len(sorted_samples) // 2
            self.noise_floor = sorted_samples[median_idx]
    
    async def handle_media_message(self, message: dict):
        """Handle incoming audio from Twilio (STT) - VAPI-style Adaptive VAD"""
        try:
            media = message.get("media", {})
            payload = media.get("payload")
            
            if not payload:
                return
            
            # Decode audio (MULAW from Twilio)
            audio_data = base64.b64decode(payload)
            self.audio_buffer.append(audio_data)
            
            # Calculate RMS energy of this frame
            energy = self._calculate_rms_energy(audio_data)
            
            # Initial calibration phase - estimate noise floor
            if self.calibration_frames < self.max_calibration_frames:
                self.calibration_frames += 1
                self._update_noise_floor(energy)
                
                if self.calibration_frames == self.max_calibration_frames:
                    print(f"🎛️ VAD calibrated - noise floor: {self.noise_floor:.1f} RMS")
                    sys.stdout.flush()
                return
            
            # Adaptive speech detection
            # Speech must be both:
            # 1. Above minimum absolute threshold (200 RMS)
            # 2. At least 3x louder than noise floor
            speech_threshold = max(self.min_speech_energy, self.noise_floor * self.speech_multiplier)
            is_speech = energy > speech_threshold
            
            if is_speech:
                # Speech detected
                self.silence_counter = 0
                if not self.speech_active:
                    self.speech_active = True
                    print(f"🎤 User started speaking (energy: {energy:.0f} RMS, threshold: {speech_threshold:.0f})...")
                    sys.stdout.flush()
            else:
                # Silence/noise detected
                if self.speech_active:
                    # User was speaking, now silent
                    self.silence_counter += 1
                    
                    # Debug logging
                    if self.silence_counter == 1:
                        print(f"🔇 Silence detected (energy: {energy:.0f}, threshold: {speech_threshold:.0f})...")
                        sys.stdout.flush()
                    elif self.silence_counter % 10 == 0:
                        remaining = self.silence_threshold - self.silence_counter
                        print(f"🔇 Silence continuing... ({self.silence_counter}/{self.silence_threshold}, {remaining} until processing)")
                        sys.stdout.flush()
                else:
                    # Not speaking yet, update noise floor
                    self._update_noise_floor(energy)
            
            # Process when 0.9 seconds of silence detected (VAPI-style)
            if self.speech_active and self.silence_counter >= self.silence_threshold:
                print(f"🔕 0.9s silence detected - processing speech and sending to STT → LLM → TTS...")
                sys.stdout.flush()
                await self.process_audio_buffer()
        
        except Exception as e:
            print(f"❌ Error handling media: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    
    async def process_audio_buffer(self):
        """Process accumulated audio with Google STT"""
        if not self.audio_buffer:
            return
        
        try:
            # Combine audio chunks
            combined_audio = b''.join(self.audio_buffer)
            self.audio_buffer = []
            self.speech_active = False
            self.silence_counter = 0
            
            if len(combined_audio) < 1000:
                return
            
            print(f"🎙️ Transcribing {len(combined_audio)} bytes...")
            sys.stdout.flush()
            
            # Get language from agent
            language_code = "en-US"
            if self.agent and hasattr(self.agent, 'language'):
                language_map = {
                    "en": "en-US", "es": "es-ES", "hi": "hi-IN",
                    "ar": "ar-SA", "zh": "zh-CN", "ur": "ur-PK"
                }
                language_code = language_map.get(self.agent.language, "en-US")
            
            # Transcribe
            result = google_stt_service.transcribe_audio_chunk_streaming(
                audio_content=combined_audio,
                language_code=language_code
            )
            
            if result.get("transcript"):
                transcript = result["transcript"].strip()
                confidence = result.get("confidence", 0.9)
                
                print(f"✅ Transcript: '{transcript}'")
                sys.stdout.flush()
                
                # Add to transcript
                await self._add_to_transcript("client", transcript, "speech", confidence)
                
                # Generate and stream response immediately!
                await self.generate_and_stream_response(transcript, confidence)
        
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    
    async def generate_and_stream_response(self, user_text: str, confidence: float):
        """Generate AI response and stream TTS in real-time"""
        try:
            print(f"🤖 Generating streaming response for: '{user_text}'")
            sys.stdout.flush()
            
            # Get AI response with streaming
            response_text = await VoiceLoggingService.generate_agent_response(
                speech_text=user_text,
                confidence=confidence,
                agent=self.agent,
                db=self.db,
                call_session_id=self.call_session.id if self.call_session else None
            )
            
            print(f"✅ AI Response: '{response_text}'")
            sys.stdout.flush()
            
            # Add to transcript
            await self._add_to_transcript("agent", response_text, "agent_response")
            
            # Stream TTS in chunks (sentence by sentence)
            await self.stream_tts_response(response_text)
        
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            sys.stdout.flush()
    
    async def stream_tts_response(self, text: str):
        """Stream TTS audio in 20ms chunks for immediate playback"""
        try:
            # Get agent voice settings
            lang = self.agent.language if self.agent and self.agent.language else "en"
            voice = self.agent.voice_type if self.agent and self.agent.voice_type else "female"
            
            print(f"🎵 Streaming TTS with 20ms chunks: '{text[:50]}...'")
            sys.stdout.flush()
            
            # Generate or fetch cached MULAW audio
            audio_bytes = await generate_mulaw_tts(text=text, lang=lang, voice=voice, use_gemini_flash=True)
            
            # Stream as 20ms frames with early playback
            await stream_mulaw_bytes_over_twilio(
                websocket=self.websocket,
                stream_sid=self.stream_sid,
                audio_bytes=audio_bytes,
                pace_20ms=True,
            )
            
            print(f"⚡ Streamed {len(audio_bytes)} bytes in 20ms chunks")
            sys.stdout.flush()
        
        except Exception as e:
            print(f"❌ Error streaming TTS: {e}")
            sys.stdout.flush()
    
    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences for streaming"""
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
            
            # Finalize any pending speech
            if self.current_speech.strip():
                await self.process_audio_buffer()
        
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
                            
                            # Generate MULAW TTS
                            audio_bytes = await generate_mulaw_tts(
                                text=text,
                                lang=lang,
                                voice=voice,
                                use_gemini_flash=True
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
                    
                    # Generate MULAW TTS
                    audio_bytes = await generate_mulaw_tts(
                        text=text,
                        lang=lang,
                        voice=voice,
                        use_gemini_flash=True
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

