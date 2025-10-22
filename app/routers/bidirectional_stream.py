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
import math  # For RMS energy calculation
from typing import Optional, Dict
from datetime import datetime, timezone
import uuid
import sys

from app.services.google_stt_service import google_stt_service
from app.services.google_tts_service import google_tts_service
from app.services.call_session_service import call_session_service
from app.services.agent_service import agent_service
from app.services.voice_logging_service import VoiceLoggingService
from app.services.transcript_service import transcript_service
from app.core.config import settings

router = APIRouter()


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
        self.silence_threshold = 50  # ~1 second silence
        
        # Dynamic energy tracking for better silence detection
        self.energy_history = []
        self.baseline_energy = 0
        self.peak_energy = 0
        
        # TTS (Output) state
        self.tts_queue = asyncio.Queue()
        self.is_speaking = False
        
        # Session data
        self.call_session = None
        self.agent = None
        self._load_session_data()
        
        print(f"✅ Bidirectional stream handler initialized for call {call_session_id}")
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
    
    async def handle_media_message(self, message: dict):
        """Handle incoming audio from Twilio (STT)"""
        try:
            media = message.get("media", {})
            payload = media.get("payload")
            
            if not payload:
                return
            
            # Decode audio
            audio_data = base64.b64decode(payload)
            self.audio_buffer.append(audio_data)
            
            # Dynamic silence detection based on audio energy
            try:
                if len(audio_data) > 0:
                    # Calculate RMS-like energy (Python 3.13 compatible)
                    sum_squares = sum(byte * byte for byte in audio_data)
                    rms_energy = math.sqrt(sum_squares / len(audio_data))
                    
                    # Track energy history for dynamic baseline
                    self.energy_history.append(rms_energy)
                    
                    # Keep last 100 samples (~2 seconds)
                    if len(self.energy_history) > 100:
                        self.energy_history.pop(0)
                    
                    # Update baseline and peak
                    if len(self.energy_history) >= 10:
                        sorted_energies = sorted(self.energy_history)
                        self.baseline_energy = sorted_energies[len(sorted_energies) // 4]  # 25th percentile
                        self.peak_energy = sorted_energies[-10]  # Top 10%
                    
                    # Dynamic threshold: speech if above baseline + 30% of range
                    energy_range = max(self.peak_energy - self.baseline_energy, 20)
                    speech_threshold = self.baseline_energy + (energy_range * 0.3)
                    
                    # Detect speech vs silence
                    if rms_energy > speech_threshold and rms_energy > 50:  # Minimum 50 to filter noise
                        self.silence_counter = 0
                        if not self.speech_active:
                            self.speech_active = True
                            print(f"🎤 User started speaking (energy: {int(rms_energy)}, threshold: {int(speech_threshold)})...")
                            sys.stdout.flush()
                    else:  # Silence detected
                        self.silence_counter += 1
                        if self.silence_counter == 1 and self.speech_active:
                            print(f"🔇 Silence starting (energy: {int(rms_energy)} < {int(speech_threshold)})...")
                            sys.stdout.flush()
                else:
                    self.silence_counter += 1
                    
            except Exception as e:
                # Fallback: use simple length check
                if len(audio_data) > 100:
                    self.silence_counter = 0
                    if not self.speech_active:
                        self.speech_active = True
                        print(f"⚠️ Energy check failed, using fallback: {e}")
                        sys.stdout.flush()
                else:
                    self.silence_counter += 1
            
            # Process when silence detected
            if self.speech_active and self.silence_counter >= self.silence_threshold:
                print(f"🔕 Silence detected - processing speech...")
                sys.stdout.flush()
                await self.process_audio_buffer()
        
        except Exception as e:
            print(f"❌ Error handling media: {e}")
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
        """Stream TTS audio in chunks for immediate playback"""
        try:
            # Get agent voice settings
            lang = self.agent.language if self.agent and self.agent.language else "en"
            voice = self.agent.voice_type if self.agent and self.agent.voice_type else "female"
            
            # Split into sentences for streaming
            sentences = self._split_into_sentences(text)
            
            print(f"🎵 Streaming {len(sentences)} sentence(s)...")
            sys.stdout.flush()
            
            for i, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                
                # Generate audio for this sentence
                audio_content = google_tts_service.text_to_speech(
                    text=sentence,
                    language=lang,
                    voice_type=voice,
                    speaking_rate=1.1,
                    pitch=0.0,
                    output_format="mulaw",  # Twilio format
                    use_gemini_flash=True  # Fast generation
                )
                
                # Send to Twilio immediately!
                await self.send_audio_to_twilio(audio_content)
                
                print(f"⚡ Streamed sentence {i+1}/{len(sentences)}")
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
        """Send audio chunk to Twilio for immediate playback"""
        try:
            # Encode audio as base64
            encoded_audio = base64.b64encode(audio_data).decode('utf-8')
            
            # Send media message to Twilio
            await self.websocket.send_json({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {
                    "payload": encoded_audio
                }
            })
            
            print(f"📤 Sent {len(audio_data)} bytes to Twilio")
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

