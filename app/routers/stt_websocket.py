"""
WebSocket endpoint for handling Twilio Media Streams with Google Cloud STT
This replaces Twilio's built-in transcription with Google Cloud Speech-to-Text
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session
import json
import base64
import asyncio
from typing import Optional, Dict
from datetime import datetime, timezone
import uuid

from app.services.google_stt_service import google_stt_service
from app.services.call_session_service import call_session_service
from app.services.agent_service import agent_service
from app.services.voice_logging_service import VoiceLoggingService
from app.routers.general_websocket import broadcast_transcript_update, broadcast_call_status_update
from app.services.transcript_service import transcript_service
from app.services.twilio_service import twilio_service
from app.services.credit_service import credit_service
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
        self._first_media_received = False  # Track first media packet
        self._in_progress_sent = False  # Track if in-progress status has been sent
        self.current_speech = ""
        self.speech_active = False
        self.silence_counter = 0
        # Smart silence detection for end-of-speech
        # Twilio sends ~50 packets per second (20ms each)
        # We wait for actual silence before processing
        # 40-50 empty chunks = ~0.8-1 second of silence (Vapi-like speed)
        self.silence_threshold = 45  # ~1 second silence (optimal for clean speech detection)
        
        # Inactivity timeout - end call if no speech for 15 seconds
        self.last_activity_time = None
        self.inactivity_timeout = 15  # seconds
        self.has_received_any_speech = False
        
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
            # ✅ DETECT FIRST MEDIA PACKET = USER PICKED UP! (same as bidirectional_stream.py)
            if not self._first_media_received:
                self._first_media_received = True
                await self._handle_user_pickup()  # User actually picked up!
            
            # Extract audio payload (base64 encoded MULAW audio)
            media = message.get("media", {})
            payload = media.get("payload")
            
            if not payload:
                return
            
            # Decode base64 audio
            audio_data = base64.b64decode(payload)
            
            # Add to buffer
            self.audio_buffer.append(audio_data)
            
            # Track silence to detect end of speech (Smart approach - no duplicates!)
            if len(audio_data) > 0:
                self.silence_counter = 0
                if not self.speech_active:
                    self.speech_active = True
                    print(f"🎤 Speech started - buffering audio...")
                    import sys
                    sys.stdout.flush()
            else:
                self.silence_counter += 1
            
            # Check for inactivity timeout (no speech for 15 seconds)
            if self.last_activity_time:
                from datetime import datetime, timezone
                time_since_activity = (datetime.now(timezone.utc) - self.last_activity_time).total_seconds()
                if time_since_activity > self.inactivity_timeout and not self.has_received_any_speech:
                    print(f"⏱️ Inactivity timeout ({time_since_activity:.1f}s) - no speech detected, ending call...")
                    import sys
                    sys.stdout.flush()
                    
                    # Add timeout message
                    if self.call_session:
                        timeout_msg = "I didn't hear anything. Please call back when you're ready. Goodbye!"
                        await self._add_to_transcript(
                            role="agent",
                            message=timeout_msg,
                            message_type="timeout_end"
                        )
                        if not self.call_session.call_metadata:
                            self.call_session.call_metadata = {}
                        self.call_session.call_metadata["pending_response"] = timeout_msg
                        self.db.commit()
                    
                    # Trigger goodbye redirect
                    if self.call_sid:
                        redirect_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events"
                        redirect_url += f"?agentId={self.agent_id}&callSessionId={self.call_session_id}&timeout=true"
                        twilio_service.redirect_call(self.call_sid, redirect_url)
                        print(f"📞 Goodbye redirect triggered for inactivity")
                        sys.stdout.flush()
                    return
            
            # Only process when silence detected (avoids duplicates)
            # This way: User speaks → Silence → Process complete speech → Clean result!
            if self.speech_active and self.silence_counter >= self.silence_threshold:
                # User stopped speaking - process all accumulated audio
                import sys
                sys.stdout.flush()
                print(f"🔕 Silence detected ({self.silence_counter} empty chunks)")
                print(f"🔊 Processing complete speech: {len(self.audio_buffer)} audio chunks...")
                sys.stdout.flush()
                
                await self.process_audio_buffer()
                
                # Flush logs again after processing
                sys.stdout.flush()
        
        except Exception as e:
            print(f"❌ Error handling media message: {e}")
            import sys
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    
    async def process_audio_buffer(self):
        """Process accumulated audio buffer with Google STT (entire speech segment)"""
        if not self.audio_buffer:
            return
        
        try:
            import sys
            
            # Combine ALL audio chunks accumulated during speech
            combined_audio = b''.join(self.audio_buffer)
            self.audio_buffer = []
            
            # Reset speech state (ready for next speech segment)
            self.speech_active = False
            self.silence_counter = 0
            
            print(f"🎵 Combined {len(combined_audio)} bytes of complete speech")
            sys.stdout.flush()
            
            # Skip if audio buffer is too small (safety check)
            # Should have accumulated multiple seconds of speech
            if len(combined_audio) < 1000:
                print(f"⚠️ Skipping - audio too short: {len(combined_audio)} bytes")
                sys.stdout.flush()
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
            
            # Transcribe audio chunk using STREAMING API (Vapi-style)
            import sys
            from datetime import datetime, timezone
            
            print(f"🎙️ Sending {len(combined_audio)} bytes to Google Cloud STT (Vapi-style)...")
            sys.stdout.flush()
            
            result = await google_stt_service.transcribe_audio_chunk_streaming(
                audio_content=combined_audio,
                language_code=language_code
            )
            
            print(f"📝 STT Result: {result}")
            sys.stdout.flush()
            
            if result.get("transcript"):
                transcript = result["transcript"].strip()
                confidence = result.get("confidence", 0.0)
                
                # Add this speech segment to accumulated speech
                if transcript:
                    self.current_speech += " " + transcript
                    self.has_received_any_speech = True
                    
                    # Reset activity timer when speech is detected
                    self.last_activity_time = datetime.now(timezone.utc)
                    
                    # 🎯 Send "in-progress" status when confident word is detected (like "hello")
                    # Only send once when we get a confident transcript with meaningful words
                    if not self._in_progress_sent and confidence >= 0.1 and len(transcript.split()) > 0:
                        await self._send_in_progress_status(transcript, confidence)
                        self._in_progress_sent = True
                    
                    print(f"✅ Speech segment received: '{transcript}' (confidence: {confidence:.2f})")
                    print(f"🎤 Total accumulated speech: '{self.current_speech.strip()}'")
                    sys.stdout.flush()
            else:
                print(f"⚠️ No transcript in result")
                sys.stdout.flush()
            
            # Now that we have the complete speech segment, finalize it
            # This will generate LLM response and trigger agent reply
            if self.current_speech.strip():
                await self.finalize_speech()
        
        except Exception as e:
            print(f"❌ Error processing audio buffer: {e}")
            import traceback
            traceback.print_exc()
    
    async def finalize_speech(self):
        """Finalize accumulated speech and trigger response generation (Vapi-style speed)"""
        if not self.current_speech.strip():
            self.speech_active = False
            self.current_speech = ""
            return
        
        try:
            import sys
            import time
            
            # Start total response timer (Vapi-style metrics)
            total_start_time = time.time()
            
            final_transcript = self.current_speech.strip()
            print(f"✅ Final speech detected: '{final_transcript}'")
            sys.stdout.flush()
            
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
                
                # Generate agent response (Vapi-style: fast LLM call)
                import time
                start_time = time.time()
                
                response_text = await VoiceLoggingService.generate_agent_response(
                    speech_text=final_transcript,
                    confidence=0.9,
                    agent=self.agent,
                    db=self.db,
                    call_session_id=self.call_session.id
                )
                
                llm_time = time.time() - start_time
                print(f"⚡ LLM response generated in {llm_time:.2f}s (Vapi-style)")
                sys.stdout.flush()
                
                # Add agent response to transcript
                transcript_start = time.time()
                await self._add_to_transcript(
                    role="agent",
                    message=response_text,
                    message_type="agent_response"
                )
                transcript_time = time.time() - transcript_start
                
                # Send response back to Twilio via API call to update TwiML
                # Twilio Media Streams is one-way, so we need to interrupt the call with new TwiML
                print(f"📤 Agent response generated: '{response_text}'")
                
                # Store response in call session metadata
                if not self.call_session.call_metadata:
                    self.call_session.call_metadata = {}
                self.call_session.call_metadata["pending_response"] = response_text
                self.db.commit()
                
                # Trigger a TwiML update by redirecting the call (Vapi-style instant)
                # This will interrupt the current stream and play the response immediately
                try:
                    if self.call_sid:
                        # Vapi-style: Immediate redirect, no delay
                        redirect_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events"
                        redirect_url += f"?agentId={self.agent_id}&callSessionId={self.call_session_id}"
                        
                        redirect_start = time.time()
                        
                        # Try to redirect the call immediately
                        success = twilio_service.redirect_call(self.call_sid, redirect_url)
                        
                        redirect_time = time.time() - redirect_start
                        
                        if success:
                            print(f"✅ TwiML redirect triggered in {redirect_time:.2f}s")
                            sys.stdout.flush()
                            
                            # Calculate total response time (Vapi-style metrics)
                            total_time = time.time() - total_start_time
                            print("=" * 80)
                            print(f"⚡ VAPI-STYLE PERFORMANCE METRICS:")
                            print(f"   📊 LLM Generation: {llm_time:.2f}s")
                            print(f"   📊 Transcript Save: {transcript_time:.3f}s")
                            print(f"   📊 TwiML Redirect: {redirect_time:.3f}s")
                            print(f"   🎯 TOTAL RESPONSE TIME: {total_time:.2f}s")
                            print(f"   (Vapi target: ~1-2s)")
                            print("=" * 80)
                            sys.stdout.flush()
                        else:
                            print(f"⚠️ Call redirect failed - call may have ended")
                            sys.stdout.flush()
                except Exception as e:
                    print(f"⚠️ Failed to trigger TwiML redirect: {e}")
                    sys.stdout.flush()
                    # Non-critical - the call will redirect on its own after the pause timeout
        
        except Exception as e:
            print(f"❌ Error finalizing speech: {e}")
            import traceback
            traceback.print_exc()
    
    async def _send_in_progress_status(self, transcript: str, confidence: float):
        """Send in-progress status when confident word is detected"""
        try:
            from datetime import datetime, timezone
            from app.routers.voice import broadcast_call_status_update
            
            if not self.call_session:
                return
            
            print("=" * 80)
            print(f"🎯 CONFIDENT WORD DETECTED: '{transcript}' (confidence: {confidence:.2f})")
            print(f"✅ Sending 'in-progress' status now")
            print("=" * 80)
            import sys
            sys.stdout.flush()
            
            try:
                if self.call_session.status != "in-progress":
                    old_status = self.call_session.status
                    self.call_session.status = "in-progress"
                    
                    # Start time already set in _handle_user_pickup() (first media packet received)
                    # Only set if not already set (backup check)
                    if not self.call_session.start_time:
                        self.call_session.start_time = datetime.now(timezone.utc)
                        print(f"⚠️ Start time was not set in _handle_user_pickup() - setting now as backup")
                    
                    self.db.commit()
                    print(f"✅ Updated DB status: '{old_status}' → 'in-progress' (confident word detected)")
                
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
                print(f"✅ Broadcasted 'in-progress' status (confident word: '{transcript}')")
                
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
                        print(f"✅ Started credit monitoring for session {self.call_session.id} (billing starts when connected status sent)")
                        print(f"🔍 DEBUG: Credits will deduct every 10s while call is active")
                    else:
                        print(f"ℹ️ Credit monitoring already active for session {self.call_session.id if self.call_session else 'unknown'}")
                except Exception as e:
                    print(f"❌ Failed to start credit monitoring: {e}")
                    import traceback
                    traceback.print_exc()
                    
            except Exception as e:
                print(f"❌ Failed to send in-progress status: {e}")
                import traceback
                traceback.print_exc()
        
        except Exception as e:
            print(f"❌ Error in _send_in_progress_status: {e}")
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
            import sys
            from datetime import datetime, timezone
            
            self.stream_sid = message.get("streamSid")
            start = message.get("start", {})
            self.call_sid = start.get("callSid")
            
            # Start inactivity timer
            self.last_activity_time = datetime.now(timezone.utc)
            
            print("=" * 60)
            print(f"🎙️ MEDIA STREAM STARTED")
            print(f"Stream SID: {self.stream_sid}")
            print(f"Call SID: {self.call_sid}")
            print(f"Call Session ID: {self.call_session_id}")
            print(f"Agent: {self.agent.name if self.agent else 'Unknown'}")
            print(f"🎯 Listening for audio... (~1s silence = speech end)")
            print(f"⏱️ Inactivity timeout: {self.inactivity_timeout}s (auto-end if no speech)")
            print(f"📊 Silence threshold: {self.silence_threshold} chunks")
            print("=" * 60)
            sys.stdout.flush()
            
            self.is_connected = True
            
            # Don't set in-progress here - wait for first media packet (user actually picked up)
            # in-progress will be set in handle_media_message when first media packet arrives
        
        except Exception as e:
            print(f"❌ Error handling start message: {e}")
            import sys
            sys.stdout.flush()
    
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
    
    async def _handle_user_pickup(self):
        """Handle user pickup - called on first media packet (same as bidirectional_stream.py)"""
        try:
            from datetime import datetime, timezone
            
            print("=" * 80)
            print(f"🎉 FIRST MEDIA PACKET - USER PICKED UP!")
            print(f"✅ Audio stream active - User actually answered")
            print(f"⏳ Waiting for confident speech (like 'hello') before sending 'in-progress' status")
            print("=" * 80)
            import sys
            sys.stdout.flush()
            
            # 🎯 SET START TIME - When first media packet is received (same point as credit monitoring start)
            if self.call_session and not self.call_session.start_time:
                self.call_session.start_time = datetime.now(timezone.utc)
                self.db.commit()
                print(f"✅ Set call start_time: {self.call_session.start_time.isoformat()} (first media packet received)")
                sys.stdout.flush()
            
            # Don't send in-progress status here - wait for confident word detection
            # Status will be sent in process_audio_buffer() when confident transcript is detected
        
        except Exception as e:
            print(f"❌ Error in _handle_user_pickup: {e}")
            import traceback
            traceback.print_exc()


@router.websocket("/ws/media-stream/{callSessionId}/{agentId}")
async def media_stream_websocket(
    websocket: WebSocket,
    callSessionId: str,
    agentId: str
):
    """
    WebSocket endpoint for Twilio Media Streams
    Receives real-time audio from Twilio and transcribes using Google Cloud STT
    
    Path Parameters:
        callSessionId: UUID of the call session
        agentId: UUID of the agent
    """
    print("=" * 80)
    print(f"🎙️ STT WebSocket Connection Attempt")
    print(f"📋 Path Parameters: callSessionId={callSessionId}, agentId={agentId}")
    print("=" * 80)
    
    # Accept connection FIRST (before any DB operations to avoid 403)
    try:
        await websocket.accept()
        print(f"✅ WebSocket connection ACCEPTED for call session {callSessionId}")
    except Exception as e:
        print(f"❌ Failed to accept WebSocket: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get database session manually (after accepting connection)
    from app.db.session import SessionLocal
    db = SessionLocal()
    
    # Create handler
    handler = TwilioMediaStreamHandler(
        websocket=websocket,
        call_session_id=callSessionId,
        agent_id=agentId,
        db=db
    )
    
    # Counter for media messages
    media_message_count = 0
    
    try:
        while True:
            # Receive message from Twilio
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            event = message.get("event")
            
            if event == "connected":
                import sys
                print("✅ Twilio Media Stream connected")
                sys.stdout.flush()
            
            elif event == "start":
                import sys
                print(f"📡 Received START event")
                sys.stdout.flush()
                await handler.handle_start_message(message)
                sys.stdout.flush()
            
            elif event == "media":
                import sys
                media_message_count += 1
                # Log every 50th media message to track activity
                if media_message_count % 50 == 0:
                    print(f"📦 Media packets received: {media_message_count}")
                    sys.stdout.flush()
                await handler.handle_media_message(message)
            
            elif event == "stop":
                import sys
                print(f"📡 Received STOP event")
                sys.stdout.flush()
                await handler.handle_stop_message(message)
                sys.stdout.flush()
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
        # Close database session
        db.close()
        print(f"🔚 WebSocket connection closed for call session {callSessionId}")

