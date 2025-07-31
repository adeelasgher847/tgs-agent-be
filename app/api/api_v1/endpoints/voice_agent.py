from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.schemas.voice_agent import VoiceAgentCreate, VoiceAgentUpdate, VoiceAgentOut
from app.models.voice_agent import VoiceAgent
from app.api.deps import get_db, get_current_user_jwt
from app.models.user import User

router = APIRouter()


@router.post("/", response_model=VoiceAgentOut, status_code=status.HTTP_201_CREATED)
def create_voice_agent(voice_agent: VoiceAgentCreate, current_user: User = Depends(get_current_user_jwt), db: Session = Depends(get_db)):
    """Create a new voice agent"""
    db_voice_agent = VoiceAgent(**voice_agent.dict())
    db.add(db_voice_agent)
    db.commit()
    db.refresh(db_voice_agent)
    return db_voice_agent


@router.get("/{voice_agent_id}", response_model=VoiceAgentOut)
def get_voice_agent(voice_agent_id: int,current_user: User = Depends(get_current_user_jwt), db: Session = Depends(get_db)):
    """Get a specific voice agent by ID"""
    db_agent = db.query(VoiceAgent).filter(VoiceAgent.id == voice_agent_id).first()
    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Voice Agent not found"
        )
    return db_agent


@router.get("/", response_model=List[VoiceAgentOut])
def list_voice_agents(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    tenant_id: Optional[int] = Query(None, description="Filter by tenant ID"),
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """Get all voice agents with optional filtering"""
    query = db.query(VoiceAgent)
    
    if tenant_id is not None:
        query = query.filter(VoiceAgent.tenant_id == tenant_id)
    
    return query.offset(skip).limit(limit).all()


@router.put("/{voice_agent_id}", response_model=VoiceAgentOut)
def update_voice_agent(
    voice_agent_id: int,
    update_data: VoiceAgentUpdate,
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """Update a voice agent"""
    db_agent = db.query(VoiceAgent).filter(VoiceAgent.id == voice_agent_id).first()
    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Voice Agent not found"
        )
    
    update_dict = update_data.dict(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(db_agent, field, value)
    
    db.commit()
    db.refresh(db_agent)
    return db_agent


@router.delete("/{voice_agent_id}", response_model=VoiceAgentOut)
def delete_voice_agent(voice_agent_id: int, current_user: User = Depends(get_current_user_jwt), db: Session = Depends(get_db)):
    """Delete a voice agent"""
    db_agent = db.query(VoiceAgent).filter(VoiceAgent.id == voice_agent_id).first()
    if not db_agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Voice Agent not found"
        )
    
    db.delete(db_agent)
    db.commit()
    return db_agent