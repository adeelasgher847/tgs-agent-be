"""
Call Session Service Module
Handles call session management including creation, updates, and retrieval
"""

from sqlalchemy.orm import Session
from app.models.call_session import CallSession
from app.models.call_log import CallLog
from app.schemas.call_log import CallLogCreate
from typing import List, Dict, Optional, Any
import uuid
from datetime import datetime, timezone
import json
import asyncio
from app.core.logger import logger
from app.services.inbound_call_crm_sync_service import (
    schedule_inbound_crm_sync,
    tenant_has_active_inbound_crm,
)
from app.utils.arq_pool import get_arq_pool


def _fire_callback_enqueue(schedule_id: uuid.UUID, scheduled_at: datetime) -> None:
    """
    Schedule an ARQ ``execute_callback`` job on the running event loop.

    Design contract
    ---------------
    - Always fire-and-forget: never awaited, never blocks the caller.
    - Idempotent: ``_job_id=callback:<schedule_id>`` lets ARQ deduplicate
      concurrent enqueues for the same schedule row.
    - Resilient: if the ARQ pool is unavailable (Redis down, startup race)
      or there is no running event loop (rare sync-only caller), a warning is
      logged and the row stays with ``arq_job_id=NULL``.  The worker's
      ``on_startup`` hook (``startup_recover_callbacks``) will pick it up on
      the next restart and submit the job with the original ``_defer_until``
      time — so no retry is ever permanently lost.
    """
    pool = get_arq_pool()
    if pool is None:
        logger.warning(
            "ARQ pool not ready; callback schedule=%s will be recovered at next worker startup",
            schedule_id,
        )
        return

    async def _enqueue() -> None:
        try:
            await pool.enqueue_job(
                "execute_callback",
                str(schedule_id),
                _defer_until=scheduled_at,
                _job_id=f"callback:{schedule_id}",
            )
            logger.info(
                "callback_enqueued schedule_id=%s defer_until=%s",
                schedule_id,
                scheduled_at.isoformat(),
            )
        except Exception as exc:
            logger.error(
                "callback_enqueue failed for schedule=%s (startup recovery will retry): %s",
                schedule_id,
                exc,
            )

    try:
        asyncio.get_running_loop().create_task(_enqueue())
    except RuntimeError:
        # No running event loop — called from a sync-only context.
        # Startup recovery will handle it.
        logger.warning(
            "No event loop available; callback schedule=%s will be recovered at next worker startup",
            schedule_id,
        )


class CallSessionService:
    """Service class for handling call session operations"""
    
    def create_call_session(self, db: Session, user_id: uuid.UUID, agent_id: uuid.UUID,
                           tenant_id: uuid.UUID, twilio_call_sid: str = None,
                           from_number: str = None, to_number: str = None,
                           call_type: str = "inbound", assistant_phone_number: str = None,
                           customer_phone_number: str = None,
                           session_id: Optional[uuid.UUID] = None,
                           status: str = "active") -> CallSession:
        """
        Create a new call session and associated call log
        
        Args:
            db: Database session
            user_id: User ID
            agent_id: Agent ID
            tenant_id: Tenant ID
            twilio_call_sid: Twilio call SID
            from_number: Caller number
            to_number: Called number
            call_type: Type of call (inbound, outbound, web)
            assistant_phone_number: Assistant's phone number
            customer_phone_number: Customer's phone number
            
        Returns:
            CallSession object
        """
        
        call_session = CallSession(
            id=session_id if session_id is not None else uuid.uuid4(),
            user_id=user_id,
            agent_id=agent_id,
            tenant_id=tenant_id,
            start_time=datetime.utcnow(),
            status=status,
            call_type=call_type,
            twilio_call_sid=twilio_call_sid,
            from_number=from_number,
            to_number=to_number,
            assistant_phone_number=assistant_phone_number,
            customer_phone_number=customer_phone_number,
            call_transcript=[],
            response_times=[]
        )
        
        db.add(call_session)
        db.commit()
        db.refresh(call_session)
        
        # Create associated call log
        self._create_call_log_for_session(db, call_session)
        
        # Broadcast call session created event
        asyncio.create_task(self._broadcast_call_event(
            str(call_session.id), 
            "call_session_created", 
            {
                "call_session_id": str(call_session.id),
                "status": call_session.status,
                "call_type": call_session.call_type,
                "start_time": call_session.start_time.isoformat() if call_session.start_time else None
            }
        ))
        
        return call_session
    
    def _create_call_log_for_session(self, db: Session, call_session: CallSession) -> CallLog:
        """Create a call log entry for a call session"""
        # Generate a shortened call ID for display (like in Vapi dashboard)
        call_id = str(call_session.id)[:8] + "..."
        
        call_log_data = CallLogCreate(
            call_session_id=call_session.id,
            tenant_id=call_session.tenant_id,
            call_id=call_id,
            external_call_id=call_session.twilio_call_sid,
            call_type=call_session.call_type,
            assistant_phone_number=call_session.assistant_phone_number,
            customer_phone_number=call_session.customer_phone_number,
            start_time=call_session.start_time
        )
        
        call_log = CallLog(**call_log_data.dict())
        db.add(call_log)
        db.commit()
        db.refresh(call_log)
        
        return call_log
    
    def _update_call_log_for_session(self, db: Session, call_session: CallSession, 
                                   ended_reason: str = None, success_evaluation: str = None,
                                   cost: float = None, transferred: bool = None) -> Optional[CallLog]:
        """Update call log entry for a call session"""
        try:
            call_log = db.query(CallLog).filter(CallLog.call_session_id == call_session.id).first()
            
            if call_log:
                if ended_reason:
                    call_log.ended_reason = ended_reason
                if success_evaluation:
                    call_log.success_evaluation = success_evaluation
                if cost is not None:
                    call_log.cost = cost
                if transferred is not None:
                    call_log.transferred = transferred
                if call_session.end_time:
                    call_log.end_time = call_session.end_time
                if call_session.duration:
                    call_log.duration = call_session.duration
                
                call_log.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(call_log)
            
            return call_log

        except Exception as e:
            db.rollback()
            logger.error("DB error in _update_call_log_for_session (session=%s): %s", call_session.id, e, exc_info=True)
            return None
    
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
    
    def update_call_session_status(self, db: Session, session_id: uuid.UUID, status: str, 
                                 ended_reason: str = None, success_evaluation: str = None,
                                 cost: float = None, transferred: bool = None) -> Optional[CallSession]:
        """
        Update call session status and associated call log
        
        Args:
            db: Database session
            session_id: Session ID (UUID)
            status: New status
            ended_reason: Reason why call ended
            success_evaluation: Whether call was successful
            cost: Cost of the call
            
        Returns:
            Updated CallSession object or None
        """
        try:
            call_session = self.get_call_session_by_id(db, session_id)
            if call_session:
                call_session.status = status
                
                if ended_reason:
                    call_session.ended_reason = ended_reason
                if success_evaluation:
                    call_session.success_evaluation = success_evaluation
                if cost is not None:
                    call_session.cost = cost
                if transferred is not None:
                    call_session.transferred = transferred
                
                if status in ["completed", "failed", "busy", "no_answer"]:
                    call_session.end_time = datetime.now(timezone.utc)
                    if call_session.start_time:
                        duration = (call_session.end_time - call_session.start_time).total_seconds()
                        call_session.duration = int(duration)
                
                db.commit()
                db.refresh(call_session)

                self._update_call_log_for_session(
                    db, call_session, ended_reason, success_evaluation, cost, transferred
                )

                if (
                    (call_session.call_type or "").lower() == "inbound"
                    and status in ("completed", "failed", "busy")
                    and tenant_has_active_inbound_crm(db, call_session.tenant_id)
                ):
                    try:
                        schedule_inbound_crm_sync(call_session.id)
                    except Exception as sync_exc:  # pragma: no cover
                        logger.warning("Inbound CRM schedule failed (non-critical): %s", sync_exc)

                # HubSpot post-call write-back: create a Call engagement with the
                # transcript summary once the call has actually completed. Fire-and-forget
                # (fail open) — see app/services/hubspot_service.py::schedule_hubspot_writeback.
                if status == "completed":
                    try:
                        from app.services.hubspot_service import (
                            schedule_hubspot_writeback,
                            tenant_has_hubspot_connected,
                        )

                        if tenant_has_hubspot_connected(db, call_session.tenant_id):
                            schedule_hubspot_writeback(call_session.id)
                    except Exception as hubspot_exc:  # pragma: no cover
                        logger.warning(
                            "HubSpot write-back schedule failed (non-critical): %s", hubspot_exc
                        )

                # Smart Callback: schedule a retry for missed outbound calls.
                # maybe_schedule_callback writes the CallbackSchedule row (sync).
                # _fire_callback_enqueue then submits the ARQ job on the current
                # event loop so the retry fires exactly at scheduled_at without
                # any polling.  If the enqueue fails, startup_recover_callbacks
                # will re-submit it when the ARQ worker next starts.
                if status in ("no_answer", "busy"):
                    try:
                        from app.services.callback_scheduler_service import callback_scheduler_service
                        cb_schedule = callback_scheduler_service.maybe_schedule_callback(
                            db, call_session
                        )
                        if cb_schedule is not None:
                            _fire_callback_enqueue(cb_schedule.id, cb_schedule.scheduled_at)
                    except Exception as cb_exc:
                        logger.warning(
                            "Smart callback schedule failed (non-critical) session=%s: %s",
                            session_id,
                            cb_exc,
                        )

            return call_session

        except Exception as e:
            db.rollback()
            logger.error("DB error in update_call_session_status (session=%s status=%s): %s", session_id, status, e, exc_info=True)
            return None
    
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
    
    async def _broadcast_call_event(self, call_session_id: str, event_type: str, event_data: dict):
        """Broadcast a call event to WebSocket connections"""
        try:
            from app.routers.general_websocket import broadcast_call_event
            await broadcast_call_event(call_session_id, event_type, event_data)
        except Exception as e:
            logger.error(f"Error broadcasting call event: {e}")
    
    async def _broadcast_status_update(self, call_session_id: str, status: str, metadata: dict = None):
        """Broadcast call status update to WebSocket connections"""
        try:
            from app.routers.general_websocket import broadcast_call_status_update
            await broadcast_call_status_update(call_session_id, status, metadata)
        except Exception as e:
            logger.error(f"Error broadcasting status update: {e}")
    
    async def _broadcast_transcript_update(self, call_session_id: str, transcript: list, new_messages: list = None):
        """Broadcast transcript update to WebSocket connections"""
        try:
            from app.routers.general_websocket import broadcast_transcript_update
            await broadcast_transcript_update(call_session_id, transcript, new_messages)
        except Exception as e:
            logger.error(f"Error broadcasting transcript update: {e}")
    
    async def _broadcast_metadata_update(self, call_session_id: str, metadata: dict):
        """Broadcast call metadata update to WebSocket connections"""
        try:
            from app.routers.general_websocket import broadcast_call_metadata_update
            await broadcast_call_metadata_update(call_session_id, metadata)
        except Exception as e:
            logger.error(f"Error broadcasting metadata update: {e}")

# Global instance
call_session_service = CallSessionService()
