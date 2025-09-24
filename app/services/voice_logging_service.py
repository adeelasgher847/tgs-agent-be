"""
Voice Logging Service
Handles voice listening, speech recognition, and call logging
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session

from app.models.call_session import CallSession
from app.models.call_log import CallLog
from app.models.agent import Agent
from app.schemas.call_log import CallLogCreate, CallLogUpdate

class VoiceLoggingService:
    """Service for voice listening and call logging"""
    
    @staticmethod
    async def log_voice_interaction(
        db: Session,
        call_session_id: uuid.UUID,
        interaction_type: str,
        audio_data: Optional[bytes] = None,
        speech_text: Optional[str] = None,
        confidence: Optional[float] = None,
        duration: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log voice interaction during a call
        """
        try:
            print("=" * 60)
            print(f"🎤 LOGGING VOICE INTERACTION")
            print(f"🆔 Call Session: {call_session_id}")
            print(f"📝 Type: {interaction_type}")
            print(f"🗣️ Speech: {speech_text}")
            print(f"📊 Confidence: {confidence}")
            print(f"⏱️ Duration: {duration}")
            print("=" * 60)
            
            # Get call session
            call_session = db.query(CallSession).filter(
                CallSession.id == call_session_id
            ).first()
            
            if not call_session:
                raise Exception(f"Call session {call_session_id} not found")
            
            # Create voice interaction log
            voice_log = {
                "id": str(uuid.uuid4()),
                "call_session_id": str(call_session_id),
                "interaction_type": interaction_type,
                "speech_text": speech_text,
                "confidence": confidence,
                "duration": duration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {}
            }
            
            # Update call session with voice interaction
            if not call_session.call_metadata:
                call_session.call_metadata = {}
            
            if "voice_interactions" not in call_session.call_metadata:
                call_session.call_metadata["voice_interactions"] = []
            
            call_session.call_metadata["voice_interactions"].append(voice_log)
            
            # Update call transcript
            if speech_text:
                if not call_session.call_transcript:
                    call_session.call_transcript = []
                
                call_session.call_transcript.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "speaker": "customer" if interaction_type == "speech_input" else "agent",
                    "text": speech_text,
                    "confidence": confidence,
                    "duration": duration
                })
            
            db.commit()
            
            print(f"✅ Voice interaction logged successfully")
            return voice_log
            
        except Exception as e:
            print(f"❌ Error logging voice interaction: {e}")
            db.rollback()
            raise e
    
    @staticmethod
    async def process_speech_input(
        db: Session,
        call_session_id: uuid.UUID,
        speech_text: str,
        confidence: float,
        duration: float,
        agent_id: Optional[uuid.UUID] = None
    ) -> Dict[str, Any]:
        """
        Process speech input and generate response
        """
        try:
            print("=" * 60)
            print(f"🗣️ PROCESSING SPEECH INPUT")
            print(f"🆔 Call Session: {call_session_id}")
            print(f"📝 Speech: '{speech_text}'")
            print(f"📊 Confidence: {confidence}")
            print(f"⏱️ Duration: {duration}")
            print("=" * 60)
            
            # Log the speech input
            await VoiceLoggingService.log_voice_interaction(
                db=db,
                call_session_id=call_session_id,
                interaction_type="speech_input",
                speech_text=speech_text,
                confidence=confidence,
                duration=duration,
                metadata={
                    "agent_id": str(agent_id) if agent_id else None,
                    "processing_time": datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Get agent for response generation
            agent = None
            if agent_id:
                agent = db.query(Agent).filter(Agent.id == agent_id).first()
            
            # Generate response based on speech input
            response_text = await VoiceLoggingService.generate_agent_response(
                speech_text=speech_text,
                confidence=confidence,
                agent=agent
            )
            
            # Log the agent response
            await VoiceLoggingService.log_voice_interaction(
                db=db,
                call_session_id=call_session_id,
                interaction_type="agent_response",
                speech_text=response_text,
                confidence=1.0,  # Agent response is always 100% confident
                duration=len(response_text) * 0.05,  # Estimate duration
                metadata={
                    "agent_id": str(agent_id) if agent_id else None,
                    "response_type": "generated"
                }
            )
            
            print(f"✅ Speech processed, response generated: '{response_text}'")
            
            return {
                "response_text": response_text,
                "confidence": confidence,
                "processed": True,
                "agent_id": str(agent_id) if agent_id else None
            }
            
        except Exception as e:
            print(f"❌ Error processing speech input: {e}")
            raise e
    
    @staticmethod
    async def generate_agent_response(
        speech_text: str,
        confidence: float,
        agent: Optional[Agent] = None
    ) -> str:
        """
        Generate agent response based on speech input
        """
        try:
            print(f"🤖 Generating response for: '{speech_text}'")
            
            # Simple response generation (can be enhanced with AI)
            speech_lower = speech_text.lower()
            
            if "hello" in speech_lower or "hi" in speech_lower:
                response = "Hello! How can I help you today?"
            elif "help" in speech_lower:
                response = "I'm here to help you. What do you need assistance with?"
            elif "thank" in speech_lower:
                response = "You're welcome! Is there anything else I can help you with?"
            elif "goodbye" in speech_lower or "bye" in speech_lower:
                response = "Thank you for calling. Have a great day!"
            elif "price" in speech_lower or "cost" in speech_lower:
                response = "I can help you with pricing information. What product or service are you interested in?"
            elif "support" in speech_lower:
                response = "I'm here to provide support. What issue are you experiencing?"
            else:
                response = f"I understand you said '{speech_text}'. Let me help you with that."
            
            # Add agent name if available
            if agent and agent.name:
                response = f"Hello! This is {agent.name}. {response}"
            
            print(f"✅ Generated response: '{response}'")
            return response
            
        except Exception as e:
            print(f"❌ Error generating agent response: {e}")
            return "I'm sorry, I didn't understand that. Could you please repeat?"
    
    @staticmethod
    async def log_call_events(
        db: Session,
        call_session_id: uuid.UUID,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> None:
        """
        Log call events (ringing, answered, completed, etc.)
        """
        try:
            print("=" * 60)
            print(f"📞 LOGGING CALL EVENT")
            print(f"🆔 Call Session: {call_session_id}")
            print(f"📝 Event Type: {event_type}")
            print(f"📊 Event Data: {event_data}")
            print("=" * 60)
            
            # Get call session
            call_session = db.query(CallSession).filter(
                CallSession.id == call_session_id
            ).first()
            
            if not call_session:
                raise Exception(f"Call session {call_session_id} not found")
            
            # Update call session with event
            if not call_session.call_metadata:
                call_session.call_metadata = {}
            
            if "call_events" not in call_session.call_metadata:
                call_session.call_metadata["call_events"] = []
            
            event_log = {
                "id": str(uuid.uuid4()),
                "event_type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": event_data
            }
            
            call_session.call_metadata["call_events"].append(event_log)
            
            # Update call session status based on event
            if event_type == "call_answered":
                call_session.status = "active"
            elif event_type == "call_completed":
                call_session.status = "completed"
                call_session.end_time = datetime.now(timezone.utc)
            elif event_type == "call_failed":
                call_session.status = "failed"
                call_session.end_time = datetime.now(timezone.utc)
            
            db.commit()
            
            print(f"✅ Call event logged successfully")
            
        except Exception as e:
            print(f"❌ Error logging call event: {e}")
            db.rollback()
            raise e
    
    @staticmethod
    def get_call_voice_logs(
        db: Session,
        call_session_id: uuid.UUID
    ) -> List[Dict[str, Any]]:
        """
        Get voice logs for a specific call session
        """
        try:
            call_session = db.query(CallSession).filter(
                CallSession.id == call_session_id
            ).first()
            
            if not call_session:
                return []
            
            if not call_session.call_metadata or "voice_interactions" not in call_session.call_metadata:
                return []
            
            return call_session.call_metadata["voice_interactions"]
            
        except Exception as e:
            print(f"❌ Error getting call voice logs: {e}")
            return []
    
    @staticmethod
    def get_call_transcript(
        db: Session,
        call_session_id: uuid.UUID
    ) -> List[Dict[str, Any]]:
        """
        Get call transcript for a specific call session
        """
        try:
            call_session = db.query(CallSession).filter(
                CallSession.id == call_session_id
            ).first()
            
            if not call_session:
                return []
            
            return call_session.call_transcript or []
            
        except Exception as e:
            print(f"❌ Error getting call transcript: {e}")
            return []
