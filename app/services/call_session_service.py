"""
Call Session Service Module
Handles call session management including creation, updates, and retrieval
"""

from sqlalchemy.orm import Session
from app.models.call_session import CallSession
from app.models.user import User
from app.models.agent import Agent
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime
import json

class CallSessionService:
    """Service class for handling call session operations"""
    
    def create_call_session(self, db: Session, user_id: uuid.UUID, agent_id: uuid.UUID, 
                           tenant_id: uuid.UUID, twilio_call_sid: str = None,
                           from_number: str = None, to_number: str = None) -> CallSession:
        """
        Create a new call session
        
        Args:
            db: Database session
            user_id: User ID
            agent_id: Agent ID
            tenant_id: Tenant ID
            twilio_call_sid: Twilio call SID
            from_number: Caller number
            to_number: Called number
            
        Returns:
            CallSession object
        """
        
        call_session = CallSession(
            user_id=user_id,
            agent_id=agent_id,
            tenant_id=tenant_id,
            start_time=datetime.utcnow(),
            status="active",
            twilio_call_sid=twilio_call_sid,
            from_number=from_number,
            to_number=to_number,
            call_transcript=[],
            response_times=[]
        )
        
        db.add(call_session)
        db.commit()
        db.refresh(call_session)
        
        return call_session
    
    def get_call_session_by_id(self, db: Session, session_id: uuid.UUID) -> Optional[CallSession]:
        """
        Get call session by ID
        
        Args:
            db: Database session
            session_id: Session ID (UUID)
            
        Returns:
            CallSession object or None
        """
        return db.query(CallSession).filter(CallSession.id == session_id).first()
    
    def get_call_session_by_twilio_sid(self, db: Session, twilio_call_sid: str) -> Optional[CallSession]:
        """
        Get call session by Twilio call SID
        
        Args:
            db: Database session
            twilio_call_sid: Twilio call SID
            
        Returns:
            CallSession object or None
        """
        return db.query(CallSession).filter(CallSession.twilio_call_sid == twilio_call_sid).first()
    
    def update_call_session_status(self, db: Session, session_id: uuid.UUID, status: str) -> Optional[CallSession]:
        """
        Update call session status
        
        Args:
            db: Database session
            session_id: Session ID (UUID)
            status: New status
            
        Returns:
            Updated CallSession object or None
        """
        call_session = self.get_call_session_by_id(db, session_id)
        if call_session:
            call_session.status = status
            if status in ["completed", "failed", "busy"]:
                call_session.end_time = datetime.utcnow()
                if call_session.start_time:
                    duration = (call_session.end_time - call_session.start_time).total_seconds()
                    call_session.duration = int(duration)
            
            db.commit()
            db.refresh(call_session)
        
        return call_session
    
    def add_transcript_entry(self, db: Session, session_id: uuid.UUID, role: str, content: str, 
                           response_time: float = None) -> Optional[CallSession]:
        """
        Add a transcript entry to the call session
        
        Args:
            db: Database session
            session_id: Session ID (UUID)
            role: Role (user or assistant)
            content: Message content
            response_time: Response time in seconds
            
        Returns:
            Updated CallSession object or None
        """
        call_session = self.get_call_session_by_id(db, session_id)
        if call_session:
            # Initialize transcript if None
            if call_session.call_transcript is None:
                call_session.call_transcript = []
            
            # Add transcript entry
            transcript_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "role": role,
                "content": content
            }
            call_session.call_transcript.append(transcript_entry)
            
            # Add response time if provided
            if response_time is not None:
                if call_session.response_times is None:
                    call_session.response_times = []
                
                response_time_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "response_time": response_time
                }
                call_session.response_times.append(response_time_entry)
            
            db.commit()
            db.refresh(call_session)
        
        return call_session
    
    def get_call_sessions_by_user(self, db: Session, user_id: uuid.UUID, 
                                 limit: int = 50) -> List[CallSession]:
        """
        Get call sessions for a specific user
        
        Args:
            db: Database session
            user_id: User ID
            limit: Maximum number of results
            
        Returns:
            List of CallSession objects
        """
        return db.query(CallSession).filter(
            CallSession.user_id == user_id
        ).order_by(CallSession.created_at.desc()).limit(limit).all()
    
    def get_call_sessions_by_agent(self, db: Session, agent_id: uuid.UUID, 
                                  limit: int = 50) -> List[CallSession]:
        """
        Get call sessions for a specific agent
        
        Args:
            db: Database session
            agent_id: Agent ID
            limit: Maximum number of results
            
        Returns:
            List of CallSession objects
        """
        return db.query(CallSession).filter(
            CallSession.agent_id == agent_id
        ).order_by(CallSession.created_at.desc()).limit(limit).all()
    
    def get_call_sessions_by_tenant(self, db: Session, tenant_id: uuid.UUID, 
                                   limit: int = 100) -> List[CallSession]:
        """
        Get call sessions for a specific tenant
        
        Args:
            db: Database session
            tenant_id: Tenant ID
            limit: Maximum number of results
            
        Returns:
            List of CallSession objects
        """
        return db.query(CallSession).filter(
            CallSession.tenant_id == tenant_id
        ).order_by(CallSession.created_at.desc()).limit(limit).all()
    
    def get_call_session_stats(self, db: Session, session_id: uuid.UUID) -> Dict[str, Any]:
        """
        Get statistics for a call session
        
        Args:
            db: Database session
            session_id: Session ID (UUID)
            
        Returns:
            Dictionary with call session statistics
        """
        call_session = self.get_call_session_by_id(db, session_id)
        if not call_session:
            return {}
        
        # Calculate average response time
        avg_response_time = None
        if call_session.response_times:
            total_time = sum(entry.get("response_time", 0) for entry in call_session.response_times)
            avg_response_time = total_time / len(call_session.response_times)
        
        # Count messages by role
        user_messages = 0
        assistant_messages = 0
        if call_session.call_transcript:
            for entry in call_session.call_transcript:
                if entry.get("role") == "user":
                    user_messages += 1
                elif entry.get("role") == "assistant":
                    assistant_messages += 1
        
        return {
            "session_id": str(call_session.id),
            "status": call_session.status,
            "duration": call_session.duration,
            "start_time": call_session.start_time.isoformat() if call_session.start_time else None,
            "end_time": call_session.end_time.isoformat() if call_session.end_time else None,
            "total_messages": len(call_session.call_transcript) if call_session.call_transcript else 0,
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "average_response_time": avg_response_time,
            "total_response_time_entries": len(call_session.response_times) if call_session.response_times else 0
        }

# Global instance
call_session_service = CallSessionService()
