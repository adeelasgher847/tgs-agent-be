from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from datetime import datetime, timezone
import uuid
import asyncio

from app.models.transcript_message import TranscriptMessage
from app.models.call_session import CallSession
from app.routers.general_websocket import broadcast_transcript_update
from app.core.logger import logger
from app.services.dlp_service import redact_phi_if_hipaa

class TranscriptService:
    """Service for managing transcript messages"""

    @staticmethod
    def add_message(
        db: Session,
        call_session_id: uuid.UUID,
        role: str,
        message: str,
        message_type: str = "speech",
        agent_id: Optional[uuid.UUID] = None,
        user_id: Optional[uuid.UUID] = None,
        confidence: Optional[float] = None,
        duration: Optional[float] = None,
        response_time: Optional[float] = None,
        metadata: Optional[dict] = None,
        hipaa_enabled: bool = False,
    ) -> TranscriptMessage:
        """Add a new message to the transcript"""

        # Redact PHI before persistence when the flow is HIPAA-enabled
        message = redact_phi_if_hipaa(message, hipaa_enabled=hipaa_enabled)

        # Get the next sequence number for this call session
        last_message = db.query(TranscriptMessage).filter(
            TranscriptMessage.call_session_id == call_session_id
        ).order_by(TranscriptMessage.sequence_number.desc()).first()
        
        next_sequence = (last_message.sequence_number + 1) if last_message else 1
        
        transcript_message = TranscriptMessage(
            call_session_id=call_session_id,
            role=role,
            message=message,
            message_type=message_type,
            sequence_number=next_sequence,
            agent_id=agent_id,
            user_id=user_id,
            confidence=confidence,
            duration=duration,
            response_time=response_time,
            message_metadata=metadata or {}
        )
        
        db.add(transcript_message)
        db.commit()
        db.refresh(transcript_message)
        
        return transcript_message
    
    @staticmethod
    def get_messages_by_session(
        db: Session,
        call_session_id: uuid.UUID,
        limit: Optional[int] = None
    ) -> List[TranscriptMessage]:
        """Get all messages for a call session, ordered by sequence number"""
        
        query = db.query(TranscriptMessage).filter(
            TranscriptMessage.call_session_id == call_session_id
        ).order_by(TranscriptMessage.sequence_number.asc())
        
        if limit:
            query = query.limit(limit)
            
        return query.all()
    
    @staticmethod
    def get_conversation_array(
        db: Session,
        call_session_id: uuid.UUID
    ) -> List[dict]:
        """Get conversation as an array of message objects (for backward compatibility)"""
        
        messages = TranscriptService.get_messages_by_session(db, call_session_id)
        
        conversation = []
        for msg in messages:
            conversation.append({
                "role": msg.role,
                "message": msg.message,
                "timestamp": msg.created_at.isoformat(),
                "sequence_number": msg.sequence_number,
                "message_type": msg.message_type,
                "agent_id": str(msg.agent_id) if msg.agent_id else None,
                "user_id": str(msg.user_id) if msg.user_id else None,
                "confidence": msg.confidence,
                "duration": msg.duration,
                "response_time": msg.response_time,
                "metadata": msg.message_metadata
            })
        
        return conversation
    
    @staticmethod
    async def add_and_broadcast_message(
        db: Session,
        call_session_id: uuid.UUID,
        role: str,
        message: str,
        message_type: str = "speech",
        agent_id: Optional[uuid.UUID] = None,
        user_id: Optional[uuid.UUID] = None,
        confidence: Optional[float] = None,
        duration: Optional[float] = None,
        response_time: Optional[float] = None,
        metadata: Optional[dict] = None,
        hipaa_enabled: bool = False,
    ) -> Optional[TranscriptMessage]:
        """Add a message and broadcast the updated conversation to WebSocket"""
        
        # Filter: Ignore Twilio system messages (Vapi-style - clean transcripts!)
        twilio_system_messages = [
            "please hold while i try to connect you",
            "please hold while we connect you",
            "connecting you now",
            "please wait while we connect",
            "try to connect you",
            "connecting",
        ]
        
        message_lower = message.lower().strip()
        if any(sys_msg in message_lower for sys_msg in twilio_system_messages):
            logger.info(f"🚫 Filtered Twilio system message: '{message[:50]}...' (ignored, not saved)")
            # Return None or create a minimal object - don't save to DB
            # This prevents LLM from seeing Twilio's messages!
            return None
        
        # Add the message (only if not filtered)
        transcript_message = TranscriptService.add_message(
            db=db,
            call_session_id=call_session_id,
            role=role,
            message=message,
            message_type=message_type,
            agent_id=agent_id,
            user_id=user_id,
            confidence=confidence,
            duration=duration,
            response_time=response_time,
            metadata=metadata,
            hipaa_enabled=hipaa_enabled,
        )
        
        # Get the complete conversation
        conversation = TranscriptService.get_conversation_array(db, call_session_id)
        
        # Create the new message entry for broadcasting
        new_message = {
            "role": role,
            "message": message,
            "timestamp": transcript_message.created_at.isoformat(),
            "sequence_number": transcript_message.sequence_number,
            "message_type": message_type,
            "agent_id": str(transcript_message.agent_id) if transcript_message.agent_id else None,
            "user_id": str(transcript_message.user_id) if transcript_message.user_id else None,
            "confidence": confidence,
            "duration": duration,
            "response_time": response_time,
            "metadata": transcript_message.message_metadata
        }
        
        # Optional WebSocket broadcast (non-blocking - fire and forget)
        try:
            asyncio.create_task(broadcast_transcript_update(
                call_session_id=str(call_session_id),
                transcript=conversation,
                new_messages=[new_message]
            ))
            logger.debug(f"✅ WebSocket: Transcript update queued for session {call_session_id}")
        except Exception as e:
            logger.warning(f"⚠️ WebSocket broadcast failed (non-critical): {e}")
            # Don't let WebSocket failures affect transcript saving
        
        return transcript_message

# Create a singleton instance
transcript_service = TranscriptService()
