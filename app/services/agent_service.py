from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional, Dict, Any
from app.models.agent import Agent
from app.schemas.agent import AgentCreate, AgentUpdate, AgentOut, AgentListResponse
from fastapi import HTTPException, status

class AgentService:
    """
    Agent service with business logic for agent operations
    """
    
    def create_agent(self, db: Session, agent_in: AgentCreate, tenant_id: int) -> Agent:
        """
        Create a new agent with tenant context
        """
        # Add tenant_id to the agent data
        agent_data = agent_in.model_dump()
        agent_data['tenant_id'] = tenant_id
        
        db_agent = Agent(**agent_data)
        db.add(db_agent)
        db.commit()
        db.refresh(db_agent)
        return db_agent
    
    def get_agent_by_id(self, db: Session, agent_id: int, tenant_id: int) -> Optional[Agent]:
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
        tenant_id: int,
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
        
        # Apply search filter - handle empty strings and whitespace
        if search and search.strip():
            search_term = search.strip().lower()
            query = query.filter(func.lower(Agent.name).like(f"%{search_term}%"))
        
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
        agent_id: int, 
        agent_update: AgentUpdate, 
        tenant_id: int
    ) -> Agent:
        """
        Update agent with tenant isolation
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
        
        db.commit()
        db.refresh(agent)
        return agent
    
    def delete_agent(self, db: Session, agent_id: int, tenant_id: int) -> bool:
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
    
    def get_agents_by_tenant(self, db: Session, tenant_id: int) -> List[Agent]:
        """
        Get all agents for a specific tenant
        """
        return db.query(Agent).filter(Agent.tenant_id == tenant_id).all()
    
    def search_agents(
        self, 
        db: Session, 
        tenant_id: int, 
        search_term: str
    ) -> List[Agent]:
        """
        Search agents by name within tenant
        """
        if not search_term or not search_term.strip():
            return []
        
        clean_search_term = search_term.strip().lower()
        return db.query(Agent).filter(
            Agent.tenant_id == tenant_id,
            func.lower(Agent.name).like(f"%{clean_search_term}%")
        ).all()

agent_service = AgentService() 