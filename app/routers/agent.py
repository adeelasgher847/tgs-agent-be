from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional
from app.schemas.agent import AgentCreate, AgentUpdate, AgentOut, AgentListResponse
from app.api.deps import get_db, require_tenant
from app.services.agent_service import agent_service
from app.models.user import User
import uuid

router = APIRouter()


@router.post("/", response_model=AgentOut, status_code=status.HTTP_201_CREATED)
def create_agent(
    agent_in: AgentCreate,
    user: User = Depends(require_tenant),  # ← Simple tenant enforcement
    db: Session = Depends(get_db)
):
    """Create a new agent"""
    return agent_service.create_agent(db, agent_in, user.current_tenant_id, user.id)


@router.get("/{agent_id}", response_model=AgentOut)
def get_agent(
    agent_id: uuid.UUID,
    user: User = Depends(require_tenant),  # ← Simple tenant enforcement
    db: Session = Depends(get_db)
):
    """Get a specific agent by ID"""
    agent = agent_service.get_agent_by_id(db, agent_id, user.current_tenant_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    return agent


@router.get("/", response_model=AgentListResponse)
def list_agents(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Records per page"),
    search: Optional[str] = Query(None, description="Search by name"),
    user: User = Depends(require_tenant),  # ← Simple tenant enforcement
    db: Session = Depends(get_db)
):
    """Get agents with pagination and search"""
    return agent_service.list_agents(db, user.current_tenant_id, page, limit, search)


@router.put("/{agent_id}", response_model=AgentOut)
def update_agent(
    agent_id: uuid.UUID,
    agent_update: AgentUpdate,
    user: User = Depends(require_tenant),  # ← Simple tenant enforcement
    db: Session = Depends(get_db)
):
    """Update an agent"""
    return agent_service.update_agent(db, agent_id, agent_update, user.current_tenant_id, user.id)


@router.delete("/{agent_id}")
def delete_agent(
    agent_id: uuid.UUID,
    user: User = Depends(require_tenant),  # ← Simple tenant enforcement
    db: Session = Depends(get_db)
):
    """Delete an agent"""
    agent_service.delete_agent(db, agent_id, user.current_tenant_id)
    return {"message": "Agent deleted successfully"}


@router.get("/search/{search_term}", response_model=list[AgentOut])
def search_agents(
    search_term: str,
    user: User = Depends(require_tenant),  # ← Simple tenant enforcement
    db: Session = Depends(get_db)
):
    """Search agents by name"""
    agents = agent_service.search_agents(db, user.current_tenant_id, search_term)
    return [AgentOut.model_validate(agent) for agent in agents] 