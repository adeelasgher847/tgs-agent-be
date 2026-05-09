from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func
from typing import List, Optional, Dict, Any
from app.models.agent import Agent
from app.models.transfer_route import TransferRoute
from app.models.model import Model
from app.models.knowledge_base_document import KnowledgeBaseDocument
from app.models.business_knowledge import BusinessKnowledge
from app.models.tts_provider import TTSProvider
from app.models.tts_voice import TTSVoice
from app.schemas.agent import AgentCreate, AgentUpdate, AgentOut, AgentListResponse
from app.services.billing_service import BillingService
from app.services.embedding_service import embed_text_for_rag
from app.services.rag_service import rag_service
from app.core.config import settings
from fastapi import HTTPException, status
import uuid
import re
from app.core.logger import logger

class AgentService:
    """
    Agent service with business logic for agent operations
    """

    def _validate_tts_selection(
        self,
        db: Session,
        *,
        tts_provider_id: Optional[uuid.UUID],
        tts_voice_id: Optional[uuid.UUID],
    ) -> Dict[str, Any]:
        """
        Validate optional TTS provider/voice selection.
        Returns normalized ids where provider can be inferred from voice.
        """
        normalized = {
            "tts_provider_id": tts_provider_id,
            "tts_voice_id": tts_voice_id,
        }

        if not tts_provider_id and not tts_voice_id:
            return normalized

        provider = None
        if tts_provider_id:
            provider = db.query(TTSProvider).filter(TTSProvider.id == tts_provider_id).first()
            if not provider or not provider.is_active:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid tts_provider_id. Provider not found or inactive.",
                )

        if tts_voice_id:
            voice = db.query(TTSVoice).filter(TTSVoice.id == tts_voice_id).first()
            if not voice or not voice.is_active:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid tts_voice_id. Voice not found or inactive.",
                )

            if provider and voice.provider_id != provider.id:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Selected TTS voice does not belong to the selected provider.",
                )

            normalized["tts_provider_id"] = provider.id if provider else voice.provider_id
            normalized["tts_voice_id"] = voice.id
        elif provider:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="tts_voice_id is required when selecting a tts_provider_id.",
            )

        return normalized

    def _validate_tts_settings_payload(self, tts_settings_json: Optional[Dict[str, Any]]) -> None:
        if not tts_settings_json:
            return
        suspicious_key_pattern = re.compile(r"(api[_-]?key|token|secret|authorization|credential|xi[_-]?api[_-]?key)", re.IGNORECASE)

        def _walk(value: Any) -> bool:
            if isinstance(value, dict):
                for raw_key, nested_value in value.items():
                    if suspicious_key_pattern.search(str(raw_key or "")):
                        return True
                    if _walk(nested_value):
                        return True
            elif isinstance(value, list):
                return any(_walk(item) for item in value)
            return False

        if _walk(tts_settings_json):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="TTS provider credentials must not be passed in request payload.",
            )

        if "background_enabled" in tts_settings_json:
            raw_enabled = tts_settings_json.get("background_enabled")
            if isinstance(raw_enabled, bool):
                pass
            elif isinstance(raw_enabled, str):
                normalized = raw_enabled.strip().lower()
                if normalized not in {"true", "false", "1", "0", "on", "off", "yes", "no"}:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=(
                            "background_enabled must be a boolean or one of: "
                            "true/false, 1/0, on/off, yes/no."
                        ),
                    )
            elif raw_enabled is not None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        "background_enabled must be a boolean or one of: "
                        "true/false, 1/0, on/off, yes/no."
                    ),
                )

        if "background_profile" in tts_settings_json:
            profile = str(tts_settings_json.get("background_profile") or "").strip().lower()
            if profile and profile not in {"office", "cafe", "call_center", "none"}:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="background_profile must be one of: office, cafe, call_center, none.",
                )

        if "background_volume" in tts_settings_json:
            raw_volume = tts_settings_json.get("background_volume")
            try:
                volume = float(raw_volume)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="background_volume must be a number between 0 and 100.",
                )
            if volume < 0 or volume > 100:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="background_volume must be between 0 and 100.",
                )

    def _validate_transfer_route_for_tenant(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        route_id: Optional[uuid.UUID],
    ) -> None:
        """Ensure transfer_route_id belongs to the same tenant (or is null)."""
        if route_id is None:
            return
        exists = (
            db.query(TransferRoute.id)
            .filter(
                TransferRoute.id == route_id,
                TransferRoute.tenant_id == tenant_id,
                TransferRoute.is_deleted == False,  # noqa: E712
            )
            .first()
        )
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="transfer_route_id not found or does not belong to this tenant.",
            )

    def _auto_ingest_agent_system_prompt(self, db: Session, agent: Agent) -> None:
        """
        Automatically ingest agent system_prompt into RAG (best-effort).
        This keeps KB setup zero-touch for users who only configure an agent prompt.
        """
        prompt_text = (agent.system_prompt or "").strip()
        if not prompt_text:
            return

        if not settings.PINECONE_API_KEY:
            logger.info(
                "Auto KB ingest skipped for agent_id=%s: PINECONE_API_KEY not configured",
                agent.id,
            )
            return

        # We need at least one embedding provider available.
        if not settings.GEMINI_API_KEY and not settings.OPENAI_API_KEY:
            logger.info(
                "Auto KB ingest skipped for agent_id=%s: no embedding provider key configured",
                agent.id,
            )
            return

        try:
            rag_service.ingest_document(
                tenant_id=agent.tenant_id,
                agent_id=agent.id,
                title=f"{agent.name} - System Prompt (Auto)",
                source_type="agent_system_prompt_auto",
                source_ref=f"agent-system-prompt:{agent.id}",
                full_text=prompt_text,
                embedding_func=embed_text_for_rag,
                version="v1",
                db_session=db,
                replace_existing=True,
            )
            logger.info(
                "Auto KB ingest success for agent_id=%s tenant_id=%s",
                agent.id,
                agent.tenant_id,
            )
        except Exception as e:
            # Never fail agent create/update because of KB ingestion.
            logger.warning(
                "Auto KB ingest failed for agent_id=%s: %s",
                agent.id,
                e,
                exc_info=True,
            )

    def ensure_agent_prompt_ingested(self, db: Session, agent: Agent) -> None:
        """
        Lazy safety net for existing agents: if auto KB doc is missing, ingest now.
        Best-effort and non-blocking for call/runtime flows.
        """
        if not agent:
            return

        source_ref = f"agent-system-prompt:{agent.id}"
        exists = (
            db.query(KnowledgeBaseDocument.id)
            .filter(
                KnowledgeBaseDocument.tenant_id == agent.tenant_id,
                KnowledgeBaseDocument.agent_id == agent.id,
                KnowledgeBaseDocument.source_type == "agent_system_prompt_auto",
                KnowledgeBaseDocument.source_ref == source_ref,
                KnowledgeBaseDocument.is_active == True,  # noqa: E712
            )
            .first()
        )
        if exists:
            return

        logger.info(
            "Auto KB document missing for agent_id=%s; triggering lazy ingest",
            agent.id,
        )
        self._auto_ingest_agent_system_prompt(db, agent)

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
        for field in ['name', 'system_prompt', 'fallback_response', 'greeting_message']:
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

        normalized_tts = self._validate_tts_selection(
            db,
            tts_provider_id=agent_data.get("tts_provider_id"),
            tts_voice_id=agent_data.get("tts_voice_id"),
        )
        agent_data["tts_provider_id"] = normalized_tts.get("tts_provider_id")
        agent_data["tts_voice_id"] = normalized_tts.get("tts_voice_id")
        self._validate_tts_settings_payload(agent_data.get("tts_settings_json"))
        self._validate_transfer_route_for_tenant(db, tenant_id, agent_data.get("transfer_route_id"))

        # Enforce one dedicated inbound agent per tenant.
        if agent_data.get("is_inbound_agent"):
            existing_inbound_agent = db.query(Agent).filter(
                Agent.tenant_id == tenant_id,
                Agent.is_deleted == False,
                Agent.is_inbound_agent == True,
            ).first()
            if existing_inbound_agent:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Only one dedicated inbound agent is allowed per tenant.",
                )

        # Enforce one follow-up appointment agent per tenant.
        if agent_data.get("is_follow_up_agent"):
            existing_fu = db.query(Agent).filter(
                Agent.tenant_id == tenant_id,
                Agent.is_deleted == False,
                Agent.is_follow_up_agent == True,
            ).first()
            if existing_fu:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Only one follow-up appointment agent is allowed per tenant.",
                )
        
        db_agent = Agent(**agent_data)
        db.add(db_agent)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Agent role constraint violated (inbound or follow-up uniqueness per tenant).",
            )
        db.refresh(db_agent)
        self._auto_ingest_agent_system_prompt(db, db_agent)
        
        return db_agent
    
    def get_agent_by_id(self, db: Session, agent_id: uuid.UUID, tenant_id: uuid.UUID) -> Agent:
        """
        Get agent by ID with strict tenant isolation.
        Returns 403 if agent exists but belongs to different tenant.
        Returns 404 if agent doesn't exist at all.
        """
        # First, check if agent exists (regardless of tenant)
        agent = (
            db.query(Agent)
            .options(joinedload(Agent.transfer_route))
            .filter(
                Agent.id == agent_id,
                Agent.is_deleted == False,
            )
            .first()
        )
        
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

        # Enforce one dedicated inbound agent per tenant.
        if update_dict.get("is_inbound_agent") is True:
            existing_inbound_agent = db.query(Agent).filter(
                Agent.tenant_id == tenant_id,
                Agent.is_deleted == False,
                Agent.is_inbound_agent == True,
                Agent.id != agent_id,
            ).first()
            if existing_inbound_agent:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Only one dedicated inbound agent is allowed per tenant.",
                )

        # Enforce one follow-up appointment agent per tenant.
        if update_dict.get("is_follow_up_agent") is True:
            existing_fu = db.query(Agent).filter(
                Agent.tenant_id == tenant_id,
                Agent.is_deleted == False,
                Agent.is_follow_up_agent == True,
                Agent.id != agent_id,
            ).first()
            if existing_fu:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Only one follow-up appointment agent is allowed per tenant.",
                )

        normalized_tts = self._validate_tts_selection(
            db,
            tts_provider_id=update_dict.get("tts_provider_id", agent.tts_provider_id),
            tts_voice_id=update_dict.get("tts_voice_id", agent.tts_voice_id),
        )
        update_dict["tts_provider_id"] = normalized_tts.get("tts_provider_id")
        update_dict["tts_voice_id"] = normalized_tts.get("tts_voice_id")
        self._validate_tts_settings_payload(update_dict.get("tts_settings_json"))

        if "transfer_route_id" in update_dict:
            self._validate_transfer_route_for_tenant(
                db, tenant_id, update_dict.get("transfer_route_id")
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
        for field in ['system_prompt', 'fallback_response', 'greeting_message']:
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
        
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Agent role constraint violated (inbound or follow-up uniqueness per tenant).",
            )
        db.refresh(agent)
        self._auto_ingest_agent_system_prompt(db, agent)
        return agent

    def get_inbound_agent_knowledge_snapshot(
        self, db: Session, inbound_agent_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> Dict[str, Any]:
        """
        Returns a tenant-wide context snapshot for an inbound agent:
        - other active agents' prompts
        - active KB documents in the tenant
        """
        inbound_agent = self.get_agent_by_id(db, inbound_agent_id, tenant_id)
        if not inbound_agent.is_inbound_agent:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Requested agent is not marked as an inbound agent.",
            )

        agent_prompts = db.query(Agent).filter(
            Agent.tenant_id == tenant_id,
            Agent.is_deleted == False,
            Agent.id != inbound_agent_id,
        ).all()

        kb_documents = db.query(KnowledgeBaseDocument).filter(
            KnowledgeBaseDocument.tenant_id == tenant_id,
            KnowledgeBaseDocument.is_active == True,  # noqa: E712
        ).all()

        return {
            "inbound_agent_id": str(inbound_agent.id),
            "tenant_id": str(tenant_id),
            "agent_prompts": [
                {
                    "agent_id": str(agent.id),
                    "agent_name": agent.name,
                    "system_prompt": agent.system_prompt,
                }
                for agent in agent_prompts
                if agent.system_prompt
            ],
            "knowledge_documents": [
                {
                    "document_id": str(doc.id),
                    "title": doc.title,
                    "source_type": doc.source_type,
                    "source_ref": doc.source_ref,
                    "agent_id": str(doc.agent_id) if doc.agent_id else None,
                }
                for doc in kb_documents
            ],
        }

    def build_inbound_prompt_context_block(
        self, db: Session, inbound_agent_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> str:
        """
        Build a compact prompt block containing all other tenant agents' system prompts.
        Intended to be appended to the inbound agent's runtime system prompt.
        """
        snapshot = self.get_inbound_agent_knowledge_snapshot(
            db=db, inbound_agent_id=inbound_agent_id, tenant_id=tenant_id
        )
        prompts = snapshot.get("agent_prompts", [])

        if not prompts:
            return """
# TENANT AGENT PROMPT CONTEXT
No additional tenant agent prompts were found.
"""

        lines = [
            "# TENANT AGENT PROMPT CONTEXT",
            "You are the tenant's dedicated inbound agent.",
            "Use the following prompt intents from other tenant agents as reference context.",
            "Do not claim actions/capabilities unless supported by conversation context and KB.",
            "",
        ]
        for idx, item in enumerate(prompts, start=1):
            lines.append(f"[{idx}] Agent: {item.get('agent_name', 'Unknown')}")
            lines.append(item.get("system_prompt", ""))
            lines.append("")
        return "\n".join(lines)

    def build_inbound_kb_documents_context_block(
        self, db: Session, inbound_agent_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> str:
        """
        Build a compact context block listing active tenant KB documents for inbound agent use.
        """
        snapshot = self.get_inbound_agent_knowledge_snapshot(
            db=db, inbound_agent_id=inbound_agent_id, tenant_id=tenant_id
        )
        docs = snapshot.get("knowledge_documents", [])

        if not docs:
            return """
# TENANT KNOWLEDGE BASE DOCUMENTS
No active tenant knowledge base documents were found.
"""

        lines = [
            "# TENANT KNOWLEDGE BASE DOCUMENTS",
            "The following active tenant knowledge documents are available for this call context.",
            "Use this list with the retrieved KB chunk context above.",
            "",
        ]
        for idx, doc in enumerate(docs, start=1):
            lines.append(
                f"[{idx}] Title: {doc.get('title', 'Unknown')} | "
                f"Type: {doc.get('source_type', 'unknown')} | "
                f"Ref: {doc.get('source_ref', '')}"
            )
        return "\n".join(lines)
    
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

    def get_inbound_agent_by_tenant(self, db: Session, tenant_id: uuid.UUID) -> Optional[Agent]:
        """
        Get the dedicated inbound agent for a tenant.
        Returns None if no inbound agent is configured.
        """
        return (
            db.query(Agent)
            .filter(
                Agent.tenant_id == tenant_id,
                Agent.is_deleted == False,
                Agent.is_inbound_agent == True,
            )
            .first()
        )

    def get_follow_up_agent_by_tenant(self, db: Session, tenant_id: uuid.UUID) -> Optional[Agent]:
        """Tenant's single appointment follow-up / reminder agent, if configured."""
        return (
            db.query(Agent)
            .filter(
                Agent.tenant_id == tenant_id,
                Agent.is_deleted == False,
                Agent.is_follow_up_agent == True,
            )
            .first()
        )
    
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

    def get_business_knowledge_for_agent(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        agent_id: Optional[uuid.UUID] = None,
    ) -> List[BusinessKnowledge]:
        """
        Return active business knowledge records for the given tenant/agent.
        Agent-scoped records come first; tenant-wide records follow as fallback.
        Multiple active records are supported and all are returned.
        """
        records: List[BusinessKnowledge] = []

        if agent_id:
            agent_records = (
                db.query(BusinessKnowledge)
                .filter(
                    BusinessKnowledge.tenant_id == tenant_id,
                    BusinessKnowledge.agent_id == agent_id,
                    BusinessKnowledge.is_active == True,  # noqa: E712
                )
                .order_by(BusinessKnowledge.created_at)
                .all()
            )
            records.extend(agent_records)

        tenant_records = (
            db.query(BusinessKnowledge)
            .filter(
                BusinessKnowledge.tenant_id == tenant_id,
                BusinessKnowledge.agent_id == None,  # noqa: E711
                BusinessKnowledge.is_active == True,  # noqa: E712
            )
            .order_by(BusinessKnowledge.created_at)
            .all()
        )
        records.extend(tenant_records)

        return records

    def build_business_knowledge_context_block(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        agent_id: Optional[uuid.UUID] = None,
    ) -> str:
        """
        Build a prompt block containing active business knowledge for the agent.

        The block carries two things in one place (single source of truth):
        1. AUTHORITATIVE BUSINESS FACTS — verified spoken-form details the agent
           must read out when asked.
        2. BUSINESS SCOPE & SERVICE-AREA POLICY — strict rules that prevent the
           agent from inventing services we do not provide, and from accepting
           callers outside our service area (with a global/remote escape hatch).

        Returns an empty string when no knowledge is configured so existing
        prompts are not affected.
        """
        records = self.get_business_knowledge_for_agent(db, tenant_id, agent_id)
        if not records:
            return ""

        # ── Aggregate scope info across all active records ────────────────
        primary_services: List[str] = []
        secondary_services: List[str] = []
        specializations_list: List[str] = []
        service_area_texts: List[str] = []

        for rec in records:
            if rec.primary_service and rec.primary_service.strip():
                primary_services.append(rec.primary_service.strip())
            if rec.secondary_service and rec.secondary_service.strip():
                secondary_services.append(rec.secondary_service.strip())
            if rec.specializations and rec.specializations.strip():
                specializations_list.append(rec.specializations.strip())
            if rec.service_areas and rec.service_areas.strip():
                service_area_texts.append(rec.service_areas.strip())

        has_scope_info = bool(
            primary_services or secondary_services or specializations_list
        )
        has_service_area = bool(service_area_texts)

        # Detect "we serve everyone" coverage in the configured service-areas
        # text so we can give the LLM a deterministic hint instead of relying
        # only on its own interpretation. Keep this conservative — when in
        # doubt, fall back to the LLM reading the raw text.
        global_coverage_keywords = (
            "anywhere",
            "any where",
            "everywhere",
            "every where",
            "globally",
            "global",
            "worldwide",
            "world wide",
            "world-wide",
            "international",
            "internationally",
            "remote",
            "remotely",
            "online only",
            "online-only",
            "fully online",
            "virtual only",
            "virtually",
            "nationwide",
            "nation wide",
            "all over",
            "all states",
            "all countries",
            "any country",
            "any city",
            "any state",
        )
        service_area_blob = " | ".join(service_area_texts).lower()
        is_global_coverage = has_service_area and any(
            kw in service_area_blob for kw in global_coverage_keywords
        )

        lines: List[str] = [
            "# AUTHORITATIVE BUSINESS FACTS",
            "The following information is verified and authoritative for this business.",
            "ALWAYS use these facts when the caller asks about the business name, address, phone, email, website, services, areas served, or pricing.",
            "Say the details exactly as written — they are already in natural spoken form.",
            "This section overrides any conflicting or missing information elsewhere in the prompt.",
            "",
        ]

        for rec in records:
            if rec.business_name:
                lines.append(f"Business Name: {rec.business_name}")
            if rec.business_type:
                lines.append(f"Business Type: {rec.business_type}")
            if rec.business_description:
                lines.append(f"About: {rec.business_description}")
            if rec.address:
                lines.append(f"Address: {rec.address}")
            if rec.phone:
                lines.append(f"Phone: {rec.phone}")
            if rec.email:
                lines.append(f"Email: {rec.email}")
            if rec.website_url:
                lines.append(f"Website: {rec.website_url}")
            if rec.primary_service:
                lines.append(f"Primary Service(s): {rec.primary_service}")
            if rec.secondary_service:
                lines.append(f"Secondary Service(s): {rec.secondary_service}")
            if rec.specializations:
                lines.append(f"Specializations: {rec.specializations}")
            if rec.service_areas:
                lines.append(f"Service Areas: {rec.service_areas}")
            if rec.pricing_information:
                lines.append(f"Pricing: {rec.pricing_information}")
            if rec.additional_information:
                lines.append(f"Additional Info: {rec.additional_information}")
            lines.append("")

        # ── BUSINESS SCOPE & POLICY (strict, non-negotiable) ──────────────
        lines.append("# BUSINESS SCOPE & POLICY — STRICT RULES")
        lines.append(
            "These rules are non-negotiable and override general helpfulness. "
            "Follow them exactly, even if other parts of the prompt are silent."
        )
        lines.append("")

        # 1) Service scope — only offer what the business actually provides.
        lines.append("## 1) SERVICES WE OFFER (allowed scope)")
        if has_scope_info:
            if primary_services:
                lines.append(
                    f"- Primary services: {' | '.join(primary_services)}"
                )
            if secondary_services:
                lines.append(
                    f"- Secondary services: {' | '.join(secondary_services)}"
                )
            if specializations_list:
                lines.append(
                    f"- Specializations: {' | '.join(specializations_list)}"
                )
            lines.append("")
            lines.append("RULES:")
            lines.append(
                "- ONLY offer, quote, schedule, or take details for the services listed above. "
                "Treat anything outside this list as NOT offered by this business."
            )
            lines.append(
                "- If the caller asks for a service that is NOT in the allowed scope:"
            )
            lines.append(
                "  a) Politely say this business does not offer that specific service. "
                "Do NOT pretend, improvise, or promise it."
            )
            lines.append(
                "  b) In the SAME reply, briefly mention what this business actually does — "
                "lead with the primary services, then optionally add secondary services or "
                "specializations if relevant."
            )
            lines.append(
                "  c) Ask if any of those would help. If yes, continue the call normally. "
                "If they only want the unsupported service, thank them warmly, say a short "
                "goodbye, and end your response with exactly [END_CALL]."
            )
            lines.append(
                "- Do NOT invent prices, timelines, packages, guarantees, or capabilities for "
                "services that are not explicitly described in AUTHORITATIVE BUSINESS FACTS above."
            )
        else:
            lines.append(
                "- Service scope is not explicitly configured for this business. "
                "Do not make up specific services, prices, or capabilities. If asked what we do, "
                "say you can take a message and have the team follow up."
            )
        lines.append("")

        # 2) Service area — refuse / accept based on configured coverage.
        lines.append("## 2) SERVICE AREA (where we operate)")
        if has_service_area:
            lines.append(f"- Service Areas (verbatim): {' | '.join(service_area_texts)}")
            if is_global_coverage:
                lines.append(
                    "- COVERAGE: GLOBAL / REMOTE. The Service Areas text indicates the business "
                    "serves callers anywhere (globally, remotely, online, worldwide, or nationwide)."
                )
                lines.append("")
                lines.append("RULES:")
                lines.append(
                    "- NEVER refuse, redirect, or end the call based on the caller's location. "
                    "Treat every caller as in-area."
                )
                lines.append(
                    "- Do not ask the caller for their city/area solely to qualify them — only ask "
                    "for location if it is required to deliver the service."
                )
            else:
                lines.append(
                    "- COVERAGE: RESTRICTED. The Service Areas above are the ONLY locations this "
                    "business currently serves. Read the text carefully — it is authoritative."
                )
                lines.append("")
                lines.append("RULES:")
                lines.append(
                    "- When the caller mentions or implies a city, neighborhood, region, state, or "
                    "country, check whether it falls within the listed Service Areas. If you cannot "
                    "tell, ask once politely for the caller's location."
                )
                lines.append(
                    "- If the caller's location IS covered, proceed normally."
                )
                lines.append(
                    "- If the caller's location is NOT covered:"
                )
                lines.append(
                    "  a) Apologize warmly and clearly say this business does not currently provide "
                    "services in that area."
                )
                lines.append(
                    "  b) Briefly name the areas the business does cover (use the Service Areas "
                    "text above)."
                )
                lines.append(
                    "  c) Thank them for calling, say a short, friendly goodbye, and end your "
                    "response with exactly [END_CALL]."
                )
                lines.append(
                    "  d) Do NOT collect personal details, do NOT schedule, and do NOT take "
                    "payment for out-of-area callers."
                )
        else:
            lines.append(
                "- Service Areas are not configured for this business."
            )
            lines.append("")
            lines.append("RULES:")
            lines.append(
                "- Do NOT refuse the caller based on their location. Coverage is unspecified, so "
                "treat every caller as potentially in-area."
            )
            lines.append(
                "- If the caller asks where the business operates, say the service area is not "
                "specified on file and offer to take a message for the team to follow up."
            )
        lines.append("")

        # 3) Pricing & additional information — never fabricate.
        lines.append("## 3) PRICING & ADDITIONAL INFORMATION")
        lines.append(
            "- For pricing, only quote what is written under 'Pricing:' in AUTHORITATIVE BUSINESS "
            "FACTS. If pricing is not listed for the requested service, say it varies and offer to "
            "take their details for a follow-up — do NOT guess or invent a number."
        )
        lines.append(
            "- For policies, hours, requirements, guarantees, or anything else, only state what is "
            "written under 'Additional Info:' or elsewhere in AUTHORITATIVE BUSINESS FACTS. If "
            "something is not documented, say you don't have that information on hand and offer a "
            "follow-up — do NOT fabricate."
        )

        return "\n".join(lines)

    def build_call_policy_block(
        self,
        *,
        business_knowledge_block: str = "",
        transfer_route: Optional[TransferRoute] = None,
    ) -> str:
        """
        Top-of-prompt operational gates that take priority over style, tone,
        and any custom/model instructions later in the system prompt.

        Three gates, only the relevant ones are emitted:
        - Service Area Gate: only when business knowledge declares restricted
          coverage (we look for the COVERAGE: RESTRICTED marker emitted by
          ``build_business_knowledge_context_block``).
        - Booking Gate: always emitted because the calendar/booking flow is
          available on every call. Enforces the name/location/issue triad
          before any [BOOK_APPOINTMENT] hint.
        - Transfer Gate: only when an agent has a ``transfer_route``
          configured. Reinforces that [TRANSFER_CALL] is the only thing that
          actually triggers a transfer.

        Returning the gates as a single block (instead of scattering them
        across the prompt) keeps the policy enforceable on long calls where
        custom instructions and history would otherwise drown the rules out.
        """
        has_restricted_area = "COVERAGE: RESTRICTED" in (business_knowledge_block or "")
        has_transfer_route = transfer_route is not None

        lines: List[str] = [
            "# CALL POLICY (NON-NEGOTIABLE — APPLY IMMEDIATELY)",
            "These rules take priority over style/tone instructions and any custom or model "
            "instructions that appear later in this prompt. Apply them at every turn.",
            "",
        ]

        section = 1

        if has_restricted_area:
            lines.extend([
                f"## {section}. Service Area Gate",
                "- Before offering a slot, scheduling, or emitting [BOOK_APPOINTMENT], you MUST "
                "confirm the caller's location is within the Service Areas listed in "
                "AUTHORITATIVE BUSINESS FACTS.",
                "- If the caller's stated city/area is NOT in the listed Service Areas: apologize "
                "briefly, name the covered areas (use the verbatim text), and end your reply with "
                "exactly [END_CALL]. Do not propose slots, take further details, or transfer.",
                "- If the caller has not stated a location yet, ask for it BEFORE discussing "
                "scheduling. One question per turn.",
                "",
            ])
            section += 1

        lines.extend([
            f"## {section}. Booking Gate",
            "- Never emit [BOOK_APPOINTMENT] until you have clearly captured ALL of: (a) the "
            "caller's name, (b) a service location, and (c) a brief reason or issue for the visit.",
            "- If any of those are missing, your next reply must ask only the single missing one. "
            "Do not bundle multiple questions in a single turn.",
            "- Never tell the caller the appointment is confirmed, booked, or held during the "
            "call. The server finalizes scheduling after the call when checks pass.",
            "",
        ])
        section += 1

        if has_transfer_route:
            t_type = (getattr(transfer_route, "transfer_type", None) or "cold").lower()
            friendly = getattr(transfer_route, "friendly_name", None) or "human contact"
            lines.extend([
                f"## {section}. Transfer & Escalation Gate",
                f"- A human contact is configured for this agent ({friendly}; transfer type: "
                f"{t_type}).",
                "- Use [TRANSFER_CALL] ONLY for genuine emergencies, safety threats, or when the "
                "caller clearly needs a human and you cannot help.",
                "- Unless there is immediate danger to life, ask up to two short confirmation "
                "questions about the situation BEFORE you transfer.",
                "- A transfer is triggered ONLY when you emit [TRANSFER_CALL] at the end of your "
                "reply. Phrases like 'silent transfer' or 'connecting you' do nothing without "
                "that exact token.",
                "- If you use [TRANSFER_CALL], do not also use [END_CALL] in the same reply; "
                "transfer takes priority.",
            ])

        return "\n".join(lines).rstrip() + "\n"


agent_service = AgentService()