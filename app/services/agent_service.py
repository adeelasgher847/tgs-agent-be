from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional, Dict, Any
from app.models.agent import Agent
from app.schemas.agent import AgentCreate, AgentUpdate, AgentOut, AgentListResponse
from fastapi import HTTPException, status
import uuid

class AgentService:
    """
    Agent service with business logic for agent operations
    """
    
    def create_agent(self, db: Session, agent_in: AgentCreate, tenant_id: uuid.UUID, user_id: uuid.UUID) -> Agent:
        """
        Create a new agent with tenant context and audit trail
        """
        # Add tenant_id and user audit fields to the agent data
        agent_data = agent_in.model_dump()
        agent_data['tenant_id'] = tenant_id
        agent_data['created_by'] = user_id
        agent_data['updated_by'] = user_id  # On creation, updated_by = created_by
        
        db_agent = Agent(**agent_data)
        db.add(db_agent)
        db.commit()
        db.refresh(db_agent)
        return db_agent
    
    def get_agent_by_id(self, db: Session, agent_id: uuid.UUID, tenant_id: uuid.UUID) -> Optional[Agent]:
        """
        Get agent by ID with tenant isolation
        """
        return db.query(Agent).filter(
            Agent.id == agent_id,
            Agent.tenant_id == tenant_id
        ).first()
    
    def list_agents(
        self, 
        db: Session, 
        tenant_id: uuid.UUID,
        page: int = 1,
        limit: int = 10,
        search: Optional[str] = None
    ) -> AgentListResponse:
        """
        List agents with pagination, search, and tenant isolation
        """
        # Calculate offset
        offset = (page - 1) * limit
        
        # Base query with tenant isolation
        query = db.query(Agent).filter(Agent.tenant_id == tenant_id)
        
        # Apply search filter
        if search:
            query = query.filter(func.lower(Agent.name).like(f"%{search.lower()}%"))
        
        # Get total count
        total = query.count()
        
        # Get paginated results
        agents = query.offset(offset).limit(limit).all()
        
        # Calculate pagination info
        total_pages = (total + limit - 1) // limit
        has_next = page * limit < total
        has_prev = page > 1
        
        return AgentListResponse(
            data=[AgentOut.model_validate(agent) for agent in agents],
            total=total,
            page=page,
            limit=limit,
            total_pages=total_pages,
            has_next=has_next,
            has_prev=has_prev
        )
    
    def update_agent(
        self, 
        db: Session, 
        agent_id: uuid.UUID, 
        agent_update: AgentUpdate, 
        tenant_id: uuid.UUID,
        user_id: uuid.UUID
    ) -> Agent:
        """
        Update agent with tenant isolation and audit trail
        """
        agent = self.get_agent_by_id(db, agent_id, tenant_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )
        
        update_dict = agent_update.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(agent, field, value)
        
        # Update the updated_by field
        agent.updated_by = user_id
        
        db.commit()
        db.refresh(agent)
        return agent
    
    def delete_agent(self, db: Session, agent_id: uuid.UUID, tenant_id: uuid.UUID) -> bool:
        """
        Delete agent with tenant isolation
        """
        agent = self.get_agent_by_id(db, agent_id, tenant_id)
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )
        
        db.delete(agent)
        db.commit()
        return True
    
    def get_agents_by_tenant(self, db: Session, tenant_id: uuid.UUID) -> List[Agent]:
        """
        Get all agents for a specific tenant
        """
        return db.query(Agent).filter(Agent.tenant_id == tenant_id).all()
    
    def search_agents(
        self, 
        db: Session, 
        tenant_id: uuid.UUID, 
        search_term: str
    ) -> List[Agent]:
        """
        Search agents by name within tenant
        """
        return db.query(Agent).filter(
            Agent.tenant_id == tenant_id,
            func.lower(Agent.name).like(f"%{search_term.lower()}%")
        ).all()

agent_service = AgentService() 