from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func
from typing import List, Optional, Dict, Any
from app.models.agent import Agent
from app.models.model import Model
from app.schemas.agent import AgentCreate, AgentUpdate, AgentOut, AgentListResponse
from app.services.billing_service import BillingService
from fastapi import HTTPException, status
import uuid
from app.core.logger import logger

class AgentService:
    """
    Agent service with business logic for agent operations
    """
    
    def create_agent(self, db: Session, agent_in: AgentCreate, tenant_id: uuid.UUID, user_id: uuid.UUID) -> Agent:
        """
        Create a new agent with tenant context and audit trail
        """
        # Validate model_id if provided
        if agent_in.model_id:
            model = db.query(Model).filter(
                Model.id == agent_in.model_id,
                Model.archive == False  # Only allow active models
            ).first()
            if not model:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid model_id. Model not found or is archived."
                )

        # 🚨 CHECK AGENT LIMIT (MAX 5 AGENTS PER TENANT)
        agent_count = db.query(func.count(Agent.id)).filter(
            Agent.tenant_id == tenant_id,
            Agent.is_deleted == False
        ).scalar()
        
        if agent_count >= 5:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Agent limit reached. You can only create up to 5 agents per tenant."
            )

        # Check for duplicate name within tenant
        existing = db.query(Agent).filter(
            Agent.tenant_id == tenant_id,
            func.lower(Agent.name) == agent_in.name.strip().lower(),
            Agent.is_deleted == False
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Agent name must be unique within the tenant."
            )

        # Sanitize string fields
        agent_data = agent_in.model_dump()
        for field in ['name', 'system_prompt', 'fallback_response']:
            if field in agent_data and agent_data[field]:
                agent_data[field] = agent_data[field].strip()
        
        # Validate agent-specific model configuration fields
        if "agent_temperature" in agent_data and agent_data["agent_temperature"] is not None:
            temp = agent_data["agent_temperature"]
            if not (0 <= temp <= 100):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Agent temperature must be between 0 and 100."
                )
        
        if "agent_max_tokens" in agent_data and agent_data["agent_max_tokens"] is not None:
            tokens = agent_data["agent_max_tokens"]
            if tokens <= 0:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Agent max tokens must be greater than 0."
                )
        
        # Add tenant_id and user audit fields to the agent data
        agent_data['tenant_id'] = tenant_id
        agent_data['created_by'] = user_id
        agent_data['updated_by'] = user_id  # On creation, updated_by = created_by
        
        db_agent = Agent(**agent_data)
        db.add(db_agent)
        db.commit()
        db.refresh(db_agent)
        
        return db_agent
    
    def get_agent_by_id(self, db: Session, agent_id: uuid.UUID, tenant_id: uuid.UUID) -> Agent:
        """
        Get agent by ID with strict tenant isolation.
        Returns 403 if agent exists but belongs to different tenant.
        Returns 404 if agent doesn't exist at all.
        """
        # First, check if agent exists (regardless of tenant)
        agent = db.query(Agent).filter(
            Agent.id == agent_id,
            Agent.is_deleted == False
        ).first()
        
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )
        
        # If agent exists but belongs to different tenant, return 403
        if agent.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. You can only access agents within your current tenant."
            )
        
        return agent
    
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
        logger.debug(f"List agents for tenant: {tenant_id}")
        # Base query with tenant isolation
        query = db.query(Agent).filter(
            Agent.tenant_id == tenant_id,
            Agent.is_deleted == False
        )
        logger.debug(f"Query: {query}")

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
        agent_id: uuid.UUID, 
        agent_update: AgentUpdate, 
        tenant_id: uuid.UUID,
        user_id: uuid.UUID
    ) -> Agent:
        """
        Update agent with tenant isolation and audit trail
        """
        agent = self.get_agent_by_id(db, agent_id, tenant_id)  # This will handle 403/404 logic
        
        update_dict = agent_update.model_dump(exclude_unset=True)

        # Validate model_id if being updated
        if "model_id" in update_dict and update_dict["model_id"] is not None:
            model = db.query(Model).filter(
                Model.id == update_dict["model_id"],
                Model.archive == False  # Only allow active models
            ).first()
            if not model:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid model_id. Model not found or is archived."
                )

        # If name is being updated, check for duplicates
        if "name" in update_dict and update_dict["name"]:
            new_name = update_dict["name"].strip()
            existing = db.query(Agent).filter(
                Agent.tenant_id == tenant_id,
                func.lower(Agent.name) == new_name.lower(),
                Agent.id != agent_id,
                Agent.is_deleted == False
            ).first()
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Agent name must be unique within the tenant."
                )
            update_dict["name"] = new_name

        # Sanitize string fields
        for field in ['system_prompt', 'fallback_response']:
            if field in update_dict and update_dict[field]:
                update_dict[field] = update_dict[field].strip()

        # Validate agent-specific model configuration fields
        if "agent_temperature" in update_dict and update_dict["agent_temperature"] is not None:
            temp = update_dict["agent_temperature"]
            if not (0 <= temp <= 100):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Agent temperature must be between 0 and 100."
                )
        
        if "agent_max_tokens" in update_dict and update_dict["agent_max_tokens"] is not None:
            tokens = update_dict["agent_max_tokens"]
            if tokens <= 0:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Agent max tokens must be greater than 0."
                )

        for field, value in update_dict.items():
            setattr(agent, field, value)
        
        # Update the updated_by field
        agent.updated_by = user_id
        
        db.commit()
        db.refresh(agent)
        return agent
    
    def delete_agent(self, db: Session, agent_id: uuid.UUID, tenant_id: uuid.UUID) -> bool:
        """
        Soft delete agent with tenant isolation
        """
        agent = self.get_agent_by_id(db, agent_id, tenant_id)  # This will handle 403/404 logic
        
        # Soft delete
        agent.is_deleted = True
        
        db.commit()
        return True
    
    def get_agents_by_tenant(self, db: Session, tenant_id: uuid.UUID) -> List[Agent]:
        """
        Get all agents for a specific tenant
        """
        return db.query(Agent).filter(
            Agent.tenant_id == tenant_id,
            Agent.is_deleted == False
        ).all()
    
    def search_agents(
        self, 
        db: Session, 
        tenant_id: uuid.UUID, 
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
            Agent.is_deleted == False,
            func.lower(Agent.name).like(f"%{clean_search_term}%")
        ).all()
    
    def get_agent_effective_model_config(self, db: Session, agent_id: uuid.UUID, tenant_id: uuid.UUID) -> Dict[str, Any]:
        """
        Get the effective model configuration for an agent.
        Returns agent-specific values if set, otherwise falls back to model defaults.
        """
        agent = db.query(Agent).options(joinedload(Agent.model)).filter(
            Agent.id == agent_id,
            Agent.tenant_id == tenant_id,
            Agent.is_deleted == False
        ).first()
        
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )
        
        # If no model is assigned, return None
        if not agent.model:
            return {
                "model_id": None,
                "model_name": None,
                "temperature": None,
                "max_tokens": None,
                "system_prompt": agent.system_prompt
            }
        
        # Use agent-specific values if set, otherwise fall back to model defaults
        effective_config = {
            "model_id": agent.model_id,
            "model_name": agent.model.model_name,
            "temperature": agent.agent_temperature if agent.agent_temperature is not None else agent.model.temperature,
            "max_tokens": agent.agent_max_tokens if agent.agent_max_tokens is not None else agent.model.max_tokens,
            "system_prompt": (
                agent.system_prompt or 
                agent.model.system_prompt or 
                "You are a helpful AI assistant for phone calls."
            )
        }
        
        return effective_config

agent_service = AgentService()