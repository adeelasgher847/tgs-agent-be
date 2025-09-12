from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional
from app.schemas.agent import AgentCreate, AgentUpdate, AgentOut, AgentListResponse, LanguageEnum, VoiceTypeEnum
from app.api.deps import get_db, get_current_user_jwt, require_member_or_admin, require_tenant, require_admin
from app.schemas.agent import AgentCreate, AgentUpdate, AgentOut, AgentListResponse
from app.schemas.base import SuccessResponse
from app.services.agent_service import agent_service
from app.models.user import User
from app.utils.response import create_success_response
import uuid

router = APIRouter()


@router.post("/", response_model=SuccessResponse[AgentOut], status_code=status.HTTP_201_CREATED)
def create_agent(
    agent_in: AgentCreate,
    tenant_user: User = Depends(require_tenant),  # ← First middleware: tenant validation
    admin_user: User = Depends(require_admin),    # ← Second middleware: admin validation
    db: Session = Depends(get_db)
):
    """Create a new agent"""
    # Both tenant_user and admin_user are validated by their respective middleware
    # We can use either one since they both represent the same user
    agent = agent_service.create_agent(db, agent_in, admin_user.current_tenant_id, admin_user.id)
    return create_success_response(agent, "Agent created successfully", status.HTTP_201_CREATED)


@router.get("/{agent_id}", response_model=SuccessResponse[AgentOut])
def get_agent(
    agent_id: uuid.UUID,
    tenant_user: User = Depends(require_tenant),  # ← First middleware: tenant validation
    user: User = Depends(require_member_or_admin),    # ← Second middleware: admin validation
    db: Session = Depends(get_db)
):
    """Get a specific agent by ID"""
    agent = agent_service.get_agent_by_id(db, agent_id, user.current_tenant_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    return create_success_response(agent, "Agent retrieved successfully")


@router.get("/", response_model=SuccessResponse[AgentListResponse])
def list_agents(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=100, description="Records per page"),
    search: Optional[str] = Query(None, description="Search by name"),
    tenant_user: User = Depends(require_tenant),  # ← First middleware: tenant validation
    user: User = Depends(require_member_or_admin),    # ← Second middleware: admin validation
    db: Session = Depends(get_db)
):
    """Get agents with pagination and search"""
    agents = agent_service.list_agents(db, user.current_tenant_id, page, limit, search)
    return create_success_response(agents, "Agents retrieved successfully")


@router.put("/{agent_id}", response_model=SuccessResponse[AgentOut])
def update_agent(
    agent_id: uuid.UUID,
    agent_update: AgentUpdate,
    tenant_user: User = Depends(require_tenant),  # ← First middleware: tenant validation
    admin_user: User = Depends(require_admin),    # ← Second middleware: admin validation
    db: Session = Depends(get_db)
):
    """Update an agent"""
    agent = agent_service.update_agent(db, agent_id, agent_update, admin_user.current_tenant_id, admin_user.id)
    return create_success_response(agent, "Agent updated successfully")


@router.delete("/{agent_id}", response_model=SuccessResponse[dict])
def delete_agent(
    agent_id: uuid.UUID,
    tenant_user: User = Depends(require_tenant),  # ← First middleware: tenant validation
    admin_user: User = Depends(require_admin),    # ← Second middleware: admin validation
    db: Session = Depends(get_db)
):
    """Delete an agent"""
    agent_service.delete_agent(db, agent_id, admin_user.current_tenant_id)
    return create_success_response({"id": str(agent_id)}, "Agent deleted successfully")


@router.get("/search/{search_term}", response_model=SuccessResponse[list[AgentOut]])
def search_agents(
    search_term: str,
    tenant_user: User = Depends(require_tenant),  # ← First middleware: tenant validation
    user: User = Depends(require_member_or_admin),    # ← Second middleware: admin validation
    db: Session = Depends(get_db)
):
    """Search agents by name"""
    agents = agent_service.search_agents(db, user.current_tenant_id, search_term)
    agent_list = [AgentOut.model_validate(agent) for agent in agents]
    return create_success_response(agent_list, f"Found {len(agent_list)} agents matching '{search_term}'") 

@router.get("/meta/voice-options")
def get_voice_options(
    user: User = Depends(get_current_user_jwt),
):
    return {
        "voice_types": [v.value for v in VoiceTypeEnum],
        "languages": [l.value for l in LanguageEnum],
    }    
    agent_list = [AgentOut.model_validate(agent) for agent in agents]
    return create_success_response(agent_list, f"Found {len(agent_list)} agents matching '{search_term}'") 
