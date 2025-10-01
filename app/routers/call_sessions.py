"""
Call Sessions Router
Handles call session management and retrieval
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid

from app.api.deps import get_db, require_tenant
from app.models.user import User
from app.models.call_session import CallSession
from app.schemas.call_session import (
    CallSessionResponse, CallSessionStats, CallSessionList, CallSessionCreate
)
from app.services.call_session_service import call_session_service
from app.utils.response import create_success_response

router = APIRouter()

@router.get("/sessions", response_model=CallSessionList)
async def list_call_sessions(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    agent_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    List call sessions with optional filtering
    
    Users can only see their own call sessions within their tenant.
    """
    try:
        # Build query - filter by current user and tenant
        query = db.query(CallSession).filter(
            CallSession.tenant_id == user.current_tenant_id,
            CallSession.user_id == user.id
        )
        
        if agent_id:
            try:
                agent_uuid = uuid.UUID(agent_id)
                query = query.filter(CallSession.agent_id == agent_uuid)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid agent ID format")
        
        if status:
            query = query.filter(CallSession.status == status)
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        sessions = query.order_by(CallSession.created_at.desc()).offset(offset).limit(limit).all()
        
        # Convert to response models
        session_responses = []
        for session in sessions:
            session_responses.append(CallSessionResponse(
                id=session.id,
                user_id=session.user_id,
                agent_id=session.agent_id,
                tenant_id=session.tenant_id,
                status=session.status,
                twilio_call_sid=session.twilio_call_sid,
                from_number=session.from_number,
                to_number=session.to_number,
                start_time=session.start_time,
                end_time=session.end_time,
                duration=session.duration,
                call_transcript=session.call_transcript,
                response_times=session.response_times,
                created_at=session.created_at,
                updated_at=session.updated_at
            ))
        
        return CallSessionList(sessions=session_responses, total=total)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}", response_model=CallSessionResponse)
async def get_call_session(
    session_id: uuid.UUID,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get a specific call session by session ID
    """
    try:
        call_session = call_session_service.get_call_session_by_id(db, session_id)
        
        if not call_session:
            raise HTTPException(status_code=404, detail="Call session not found")
        
        # Check if user has access to this session (same tenant and same user)
        if call_session.tenant_id != user.current_tenant_id or call_session.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return CallSessionResponse(
            id=call_session.id,
            user_id=call_session.user_id,
            agent_id=call_session.agent_id,
            tenant_id=call_session.tenant_id,
            status=call_session.status,
            twilio_call_sid=call_session.twilio_call_sid,
            from_number=call_session.from_number,
            to_number=call_session.to_number,
            start_time=call_session.start_time,
            end_time=call_session.end_time,
            duration=call_session.duration,
            call_transcript=call_session.call_transcript,
            response_times=call_session.response_times,
            created_at=call_session.created_at,
            updated_at=call_session.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/stats", response_model=CallSessionStats)
async def get_call_session_stats(
    session_id: uuid.UUID,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get statistics for a specific call session
    """
    try:
        call_session = call_session_service.get_call_session_by_id(db, session_id)
        
        if not call_session:
            raise HTTPException(status_code=404, detail="Call session not found")
        
        # Check if user has access to this session (same tenant and same user)
        if call_session.tenant_id != user.current_tenant_id or call_session.user_id != user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        stats = call_session_service.get_call_session_stats(db, session_id)
        
        return CallSessionStats(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
