"""
WebSocket endpoint for handling Twilio Media Streams with Google Cloud STT
This replaces Twilio's built-in transcription with Google Cloud Speech-to-Text
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from sqlalchemy.orm import Session
import json
import base64
import asyncio
from typing import Optional, Dict
from datetime import datetime, timezone
import uuid

from app.api.deps import get_db
from app.services.google_stt_service import google_stt_service
from app.services.call_session_service import call_session_service
from app.services.agent_service import agent_service
from app.services.voice_logging_service import VoiceLoggingService
from app.routers.general_websocket import broadcast_transcript_update
from app.services.transcript_service import transcript_service
from app.services.twilio_service import twilio_service
from app.core.config import settings

router = APIRouter()


class TwilioMediaStreamHandler:
    """Handles Twilio Media Stream WebSocket connection and Google STT integration"""
    
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
        self.audio_buffer = []
        self.stream_sid = None
        self.call_sid = None
        self.is_connected = False
        self.current_speech = ""
        self.speech_active = False
        self.silence_counter = 0
        self.silence_threshold = 20  # Number of empty chunks before considering speech ended
        
        # Get call session and agent
        self.call_session = None
        self.agent = None
        self._load_session_data()
    
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
                print(f"✅ Loaded call session {self.call_session_id} and agent {self.agent.name if self.agent else 'Unknown'}")
        except Exception as e:
            print(f"⚠️ Error loading session data: {e}")
    
    async def handle_media_message(self, message: dict):
        """Handle incoming media message from Twilio"""
        try:
            # Extract audio payload (base64 encoded MULAW audio)
            media = message.get("media", {})
            payload = media.get("payload")
            
            if not payload:
                return
            
            # Decode base64 audio
            audio_data = base64.b64decode(payload)
            
            # Add to buffer
            self.audio_buffer.append(audio_data)
            
            # Reset silence counter when receiving audio
            if len(audio_data) > 0:
                self.silence_counter = 0
            else:
                self.silence_counter += 1
            
            # Process buffer when it reaches a certain size (adjust for latency vs accuracy)
            buffer_size_threshold = 10  # Process every 10 chunks
            
            if len(self.audio_buffer) >= buffer_size_threshold:
                await self.process_audio_buffer()
        
        except Exception as e:
            print(f"❌ Error handling media message: {e}")
            import traceback
            traceback.print_exc()
    
    async def process_audio_buffer(self):
        """Process accumulated audio buffer with Google STT"""
        if not self.audio_buffer:
            return
        
        try:
            # Combine audio chunks
            combined_audio = b''.join(self.audio_buffer)
            self.audio_buffer = []
            
            # Skip if audio is too short
            if len(combined_audio) < 100:
                return
            
            # Get language from agent
            language_code = "en-US"
            if self.agent and hasattr(self.agent, 'language'):
                # Map agent language to Google STT language codes
                language_map = {
                    "en": "en-US",
                    "es": "es-ES",
                    "hi": "hi-IN",
                    "ar": "ar-SA",
                    "zh": "zh-CN",
                    "ur": "ur-PK"
                }
                language_code = language_map.get(self.agent.language, "en-US")
            
            # Transcribe audio chunk
            result = google_stt_service.transcribe_audio_chunk(
                audio_content=combined_audio,
                language_code=language_code
            )
            
            if result.get("transcript"):
                transcript = result["transcript"].strip()
                confidence = result.get("confidence", 0.0)
                
                # Accumulate speech
                if transcript:
                    self.current_speech += " " + transcript
                    self.speech_active = True
                    self.silence_counter = 0
                    
                    print(f"🎤 Accumulated speech: '{self.current_speech.strip()}'")
            
            # Check if speech has ended (silence detected)
            if self.speech_active and self.silence_counter >= self.silence_threshold:
                await self.finalize_speech()
        
        except Exception as e:
            print(f"❌ Error processing audio buffer: {e}")
            import traceback
            traceback.print_exc()
    
    async def finalize_speech(self):
        """Finalize accumulated speech and trigger response generation"""
        if not self.current_speech.strip():
            self.speech_active = False
            self.current_speech = ""
            return
        
        try:
            final_transcript = self.current_speech.strip()
            print(f"✅ Final speech detected: '{final_transcript}'")
            
            # Reset speech state
            self.speech_active = False
            self.current_speech = ""
            self.silence_counter = 0
            
            # Add to transcript
            if self.call_session:
                await self._add_to_transcript(
                    role="client",
                    message=final_transcript,
                    message_type="speech",
                    confidence=0.9  # Approximate confidence
                )
                
                # Log voice interaction
                await VoiceLoggingService.log_voice_interaction(
                    db=self.db,
                    call_session_id=self.call_session.id,
                    interaction_type="speech_input",
                    speech_text=final_transcript,
                    confidence=0.9,
                    metadata={
                        "call_sid": self.call_sid,
                        "agent_id": str(self.agent.id) if self.agent else None,
                        "tenant_id": str(self.agent.tenant_id) if self.agent else None,
                        "source": "google_stt"
                    }
                )
                
                # Generate agent response
                response_text = await VoiceLoggingService.generate_agent_response(
                    speech_text=final_transcript,
                    confidence=0.9,
                    agent=self.agent,
                    db=self.db,
                    call_session_id=self.call_session.id
                )
                
                # Add agent response to transcript
                await self._add_to_transcript(
                    role="agent",
                    message=response_text,
                    message_type="agent_response"
                )
                
                # Send response back to Twilio via API call to update TwiML
                # Twilio Media Streams is one-way, so we need to interrupt the call with new TwiML
                print(f"📤 Agent response generated: '{response_text}'")
                
                # Store response in call session metadata
                if not self.call_session.call_metadata:
                    self.call_session.call_metadata = {}
                self.call_session.call_metadata["pending_response"] = response_text
                self.db.commit()
                
                # Trigger a TwiML update by redirecting the call
                # This will interrupt the current stream and play the response
                try:
                    if self.call_sid:
                        redirect_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events"
                        redirect_url += f"?agentId={self.agent_id}&callSessionId={self.call_session_id}"
                        
                        twilio_service.redirect_call(self.call_sid, redirect_url)
                        print(f"✅ Triggered TwiML redirect for call {self.call_sid}")
                except Exception as e:
                    print(f"⚠️ Failed to trigger TwiML redirect: {e}")
                    # Non-critical - the call will redirect on its own after the pause timeout
        
        except Exception as e:
            print(f"❌ Error finalizing speech: {e}")
            import traceback
            traceback.print_exc()
    
    async def _add_to_transcript(
        self,
        role: str,
        message: str,
        message_type: str = "speech",
        confidence: Optional[float] = None
    ):
        """Add message to transcript"""
        try:
            transcript_message = await transcript_service.add_and_broadcast_message(
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
            
            print(f"📝 Added to transcript: {role} - {message[:50]}...")
        
        except Exception as e:
            print(f"❌ Error adding to transcript: {e}")
    
    async def handle_start_message(self, message: dict):
        """Handle stream start message"""
        try:
            self.stream_sid = message.get("streamSid")
            start = message.get("start", {})
            self.call_sid = start.get("callSid")
            
            print("=" * 60)
            print(f"🎙️ MEDIA STREAM STARTED")
            print(f"Stream SID: {self.stream_sid}")
            print(f"Call SID: {self.call_sid}")
            print(f"Call Session ID: {self.call_session_id}")
            print(f"Agent: {self.agent.name if self.agent else 'Unknown'}")
            print("=" * 60)
            
            self.is_connected = True
        
        except Exception as e:
            print(f"❌ Error handling start message: {e}")
    
    async def handle_stop_message(self, message: dict):
        """Handle stream stop message"""
        try:
            print("=" * 60)
            print(f"🛑 MEDIA STREAM STOPPED")
            print(f"Stream SID: {self.stream_sid}")
            print(f"Call SID: {self.call_sid}")
            print("=" * 60)
            
            # Finalize any pending speech
            if self.current_speech.strip():
                await self.finalize_speech()
            
            self.is_connected = False
        
        except Exception as e:
            print(f"❌ Error handling stop message: {e}")


@router.websocket("/ws/media-stream")
async def media_stream_websocket(
    websocket: WebSocket,
    callSessionId: str = Query(...),
    agentId: str = Query(...),
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for Twilio Media Streams
    Receives real-time audio from Twilio and transcribes using Google Cloud STT
    """
    await websocket.accept()
    print(f"✅ WebSocket connection accepted for call session {callSessionId}")
    
    # Create handler
    handler = TwilioMediaStreamHandler(
        websocket=websocket,
        call_session_id=callSessionId,
        agent_id=agentId,
        db=db
    )
    
    try:
        while True:
            # Receive message from Twilio
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            event = message.get("event")
            
            if event == "connected":
                print("✅ Twilio Media Stream connected")
            
            elif event == "start":
                await handler.handle_start_message(message)
            
            elif event == "media":
                await handler.handle_media_message(message)
            
            elif event == "stop":
                await handler.handle_stop_message(message)
                break
            
            elif event == "mark":
                # Mark events can be used for synchronization
                pass
            
            else:
                print(f"⚠️ Unknown event type: {event}")
    
    except WebSocketDisconnect:
        print(f"📡 WebSocket disconnected for call session {callSessionId}")
        
        # Finalize any pending speech
        if handler.current_speech.strip():
            await handler.finalize_speech()
    
    except Exception as e:
        print(f"❌ Error in media stream WebSocket: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"🔚 WebSocket connection closed for call session {callSessionId}")

