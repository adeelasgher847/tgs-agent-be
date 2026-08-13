from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func, select
from typing import List, Dict, Any
from app.models.agent import Agent
from app.models.phone_number import PhoneNumber
from app.models.transfer_route import TransferRoute
from app.models.model import Model
from app.models.knowledge_base_document import KnowledgeBase
from app.models.tts_provider import TTSProvider
from app.models.tts_voice import TTSVoice
from app.models.stt_provider import STTProvider
from app.models.stt_model import STTModel
from app.schemas.agent import (
    AgentCreate,
    AgentUpdate,
    AgentListResponse,
    TtsModelSchema,
    TtsProviderEnum,
    SttModelSchema,
    _DEFAULT_STT_PROVIDER,
    _DEFAULT_STT_MODEL_ID,
    _DEFAULT_STT_LANGUAGE_CODE,
    agent_to_out,
    normalize_tts_provider_slug,
)
from app.services.embedding_service import embed_text_for_rag
from app.services.rag_service import rag_service
from app.core.config import settings
from app.core.db_encryption import encrypt_elevenlabs_key
from app.repositories.agent_repository import AgentRepository
from fastapi import HTTPException, status
import uuid
import re
from app.core.logger import logger

class AgentService:
    """
    Agent service with business logic for agent operations
    """

    def _repo(self, db: Session) -> AgentRepository:
        return AgentRepository(db)

    def list_active_llm_model_names(self, db: Session) -> list[str]:
        rows = (
            db.query(Model.model_name)
            .filter(Model.archive == False)  # noqa: E712
            .order_by(Model.model_name)
            .all()
        )
        return [r[0] for r in rows]

    def _resolve_llm_model(self, db: Session, llm_model: str) -> Model:
        name = llm_model.strip()
        model = (
            db.query(Model)
            .filter(Model.model_name == name, Model.archive == False)  # noqa: E712
            .first()
        )
        if not model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"'{name}' is not a supported LLM model.",
            )
        return model

    def _resolve_tts_model(self, db: Session, tts: TtsModelSchema) -> dict[str, Any]:
        slug = normalize_tts_provider_slug(tts.provider.value)
        # BYO key is not a separate voice provider in our catalog; it only
        # changes runtime behavior (inject ElevenLabs API key).
        provider_lookup_slug = (
            "elevenlabs" if slug == TtsProviderEnum.elevenlabs_byo.value else slug
        )
        provider = (
            db.query(TTSProvider)
            .filter(
                TTSProvider.slug == provider_lookup_slug,
                TTSProvider.is_active == True,  # noqa: E712
            )
            .first()
        )
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid ttsModel.provider '{tts.provider.value}'. Provider not found or inactive.",
            )

        voice = (
            db.query(TTSVoice)
            .filter(
                TTSVoice.provider_id == provider.id,
                TTSVoice.external_voice_id == tts.voice_id,
                TTSVoice.is_active == True,  # noqa: E712
            )
            .first()
        )
        if not voice:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid ttsModel.voiceId '{tts.voice_id}' for provider '{tts.provider.value}'."
                ),
            )

        lang = tts.language.value
        if voice.language_code and voice.language_code.lower() != lang.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"ttsModel.language '{lang}' does not match voice language "
                    f"'{voice.language_code}'."
                ),
            )

        return {
            "tts_provider_slug": slug,
            "tts_voice_external_id": tts.voice_id,
            "tts_language": lang,
            "tts_provider_id": provider.id,
            "tts_voice_id": voice.id,
        }

    def _resolve_stt_model(
        self,
        db: Session,
        stt: SttModelSchema | None,
    ) -> Dict[str, Any]:
        """Validate and resolve STT provider + model to DB FK ids + slug triad.

        Falls back to deepgram/nova-3/en when stt is None (backward compat).
        modelId is the user-facing ID (e.g. 'nova-3', 'chirp-3'), never 'phone_call'.
        """
        if stt is None:
            slug = _DEFAULT_STT_PROVIDER
            model_id_str = _DEFAULT_STT_MODEL_ID
            lang = _DEFAULT_STT_LANGUAGE_CODE
        else:
            slug = stt.provider.value
            model_id_str = stt.model_id
            lang = stt.language_code

        provider = (
            db.query(STTProvider)
            .filter(
                STTProvider.slug == slug,
                STTProvider.is_active == True,  # noqa: E712
            )
            .first()
        )
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid sttModel.provider '{slug}'. Provider not found or inactive.",
            )

        model = (
            db.query(STTModel)
            .filter(
                STTModel.provider_id == provider.id,
                STTModel.external_model_id == model_id_str,
                STTModel.is_active == True,  # noqa: E712
            )
            .first()
        )
        if not model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid sttModel.modelId '{model_id_str}' "
                    f"for provider '{slug}'. Not found or inactive."
                ),
            )

        return {
            "stt_provider_slug": slug,
            "stt_model_external_id": model_id_str,
            "stt_language_code": lang,
            "stt_provider_id": provider.id,
            "stt_model_id": model.id,
        }

    def _encrypt_byo_key(self, raw_key: str, db: Session) -> str:
        try:
            return encrypt_elevenlabs_key(raw_key, db)
        except ValueError as exc:
            logger.error("ElevenLabs BYO key encryption failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not securely store the provided ElevenLabs key",
            )

    def _ticket_payload_from_create(self, db: Session, agent_in: AgentCreate) -> Dict[str, Any]:
        model = self._resolve_llm_model(db, agent_in.llm_model)
        tts_fields = self._resolve_tts_model(db, agent_in.tts_model)
        stt_fields = self._resolve_stt_model(db, agent_in.stt_model)
        encrypted_key: str | None = None
        if agent_in.tts_model.provider == TtsProviderEnum.elevenlabs_byo:
            encrypted_key = self._encrypt_byo_key(agent_in.eleven_labs_api_key or "", db)
        stt_settings: Dict[str, Any] | None = None
        if agent_in.stt_settings is not None:
            stt_settings = agent_in.stt_settings.model_dump(by_alias=False, exclude_none=True)
        return {
            "llm_model": model.model_name,
            "model_id": model.id,
            "provider_id": model.provider_id,
            "status": agent_in.status.value,
            "encrypted_elevenlabs_api_key": encrypted_key,
            "stt_settings_json": stt_settings,
            **tts_fields,
            **stt_fields,
        }

    def _apply_ticket_update(
        self,
        db: Session,
        agent_in: AgentUpdate,
        agent: Agent,
        update_dict: Dict[str, Any],
    ) -> None:
        if agent_in.llm_model is not None:
            model = self._resolve_llm_model(db, agent_in.llm_model)
            update_dict["llm_model"] = model.model_name
            update_dict["model_id"] = model.id
            update_dict["provider_id"] = model.provider_id
        if agent_in.status is not None:
            update_dict["status"] = agent_in.status.value
        if agent_in.tts_model is not None:
            update_dict.update(self._resolve_tts_model(db, agent_in.tts_model))
            if agent_in.tts_model.provider != TtsProviderEnum.elevenlabs_byo:
                update_dict["encrypted_elevenlabs_api_key"] = None
        if agent_in.stt_model is not None:
            update_dict.update(self._resolve_stt_model(db, agent_in.stt_model))
        if agent_in.stt_settings is not None:
            update_dict["stt_settings_json"] = agent_in.stt_settings.model_dump(
                by_alias=False, exclude_none=True
            )
        if agent_in.eleven_labs_api_key is not None:
            update_dict["encrypted_elevenlabs_api_key"] = self._encrypt_byo_key(
                agent_in.eleven_labs_api_key, db
            )
        if agent_in.tts_model is not None:
            new_is_byo = agent_in.tts_model.provider == TtsProviderEnum.elevenlabs_byo
            if (
                new_is_byo
                and not agent_in.eleven_labs_api_key
                and not agent.encrypted_elevenlabs_api_key
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="elevenLabsApiKey is required when ttsModel.provider is 'elevenlabs_byo'",
                )

    def has_active_phone_binding(self, db: Session, agent_id: uuid.UUID) -> bool:
        stmt = (
            select(PhoneNumber.id)
            .where(
                PhoneNumber.assistant_id == agent_id,
                PhoneNumber.status == "active",
            )
            .limit(1)
        )
        return db.execute(stmt).first() is not None

    def _validate_tts_settings_payload(self, tts_settings_json: Dict[str, Any] | None) -> None:
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

        # Validate speed / volume — accept flat or nested ("settings": {...}).
        # Ranges match agent_runtime clamps so the API rejects out-of-bounds
        # input rather than silently coercing it during call setup.
        nested = tts_settings_json.get("settings") if isinstance(tts_settings_json, dict) else None
        combined: Dict[str, Any] = {}
        if isinstance(nested, dict):
            combined.update(nested)
        combined.update({k: v for k, v in tts_settings_json.items() if k != "settings"})

        if "speed" in combined:
            try:
                speed = float(combined["speed"])
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"ttsSettingsJson.speed must be a number between "
                        f"{settings.TTS_SPEED_MIN} and {settings.TTS_SPEED_MAX}."
                    ),
                )
            if speed < settings.TTS_SPEED_MIN or speed > settings.TTS_SPEED_MAX:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"ttsSettingsJson.speed must be between "
                        f"{settings.TTS_SPEED_MIN} and {settings.TTS_SPEED_MAX}."
                    ),
                )

        if "volume" in combined:
            try:
                voice_volume = float(combined["volume"])
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"ttsSettingsJson.volume must be a number between "
                        f"{settings.TTS_VOLUME_MIN} and {settings.TTS_VOLUME_MAX}."
                    ),
                )
            if voice_volume < settings.TTS_VOLUME_MIN or voice_volume > settings.TTS_VOLUME_MAX:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"ttsSettingsJson.volume must be between "
                        f"{settings.TTS_VOLUME_MIN} and {settings.TTS_VOLUME_MAX}."
                    ),
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
        route_id: uuid.UUID | None,
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
        Migrated to pgvector schema: auto-ingest via the new pipeline.
        Calls _auto_ingest_agent_system_prompt which uses rag_service.ingest_document.
        """
        if not agent:
            return
        self._auto_ingest_agent_system_prompt(db, agent)

    def create_agent(
        self,
        db: Session,
        agent_in: AgentCreate,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
    ) -> Agent:
        """
        Create a new agent with tenant context and audit trail.
        Supports JWT users and API-key M2M (``user_id`` may be None).
        """
        repo = self._repo(db)

        ticket_data = self._ticket_payload_from_create(db, agent_in)

        # Sanitize string fields (exclude ticket-only nested objects)
        agent_data = agent_in.model_dump(
            exclude={"tts_model", "stt_model", "stt_settings", "eleven_labs_api_key", "llm_model", "status"}
        )
        agent_data.update(ticket_data)
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

        self._validate_tts_settings_payload(agent_data.get("tts_settings_json"))
        self._validate_transfer_route_for_tenant(db, tenant_id, agent_data.get("transfer_route_id"))

        try:
            db_agent = repo.create(agent_data)
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Agent role constraint violated (inbound or follow-up uniqueness per tenant).",
            )
        self._auto_ingest_agent_system_prompt(db, db_agent)

        return db_agent
    
    def get_agent_by_id(self, db: Session, agent_id: uuid.UUID, tenant_id: uuid.UUID) -> Agent:
        """
        Get agent by ID with strict tenant isolation.
        Returns 404 if agent doesn't exist or belongs to a different workspace.
        """
        agent = self._repo(db).find_by_id(agent_id, load_transfer_route=True)

        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )
        
        if agent.tenant_id != tenant_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found",
            )

        return agent
    
    def list_agents(
        self, 
        db: Session, 
        tenant_id: uuid.UUID,
        page: int = 1,
        limit: int = 20,
        search: str | None = None
    ) -> AgentListResponse:
        """
        List agents with pagination, search, and tenant isolation
        """
        logger.debug("List agents for tenant: %s", tenant_id)
        agents, total = self._repo(db).find_by_workspace(
            tenant_id, page=page, limit=limit, search=search
        )

        return AgentListResponse(
            data=[agent_to_out(agent) for agent in agents],
            total=total,
            page=page,
            page_size=limit,
        )
    
    def update_agent(
        self, 
        db: Session, 
        agent_id: uuid.UUID, 
        agent_update: AgentUpdate, 
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
    ) -> Agent:
        """
        Update agent with tenant isolation and audit trail
        """
        agent = self.get_agent_by_id(db, agent_id, tenant_id)

        update_dict = agent_update.model_dump(
            exclude_unset=True,
            exclude={"tts_model", "stt_model", "stt_settings", "eleven_labs_api_key", "llm_model", "status"},
        )
        self._apply_ticket_update(db, agent_update, agent, update_dict)

        self._validate_tts_settings_payload(update_dict.get("tts_settings_json"))

        if "transfer_route_id" in update_dict:
            self._validate_transfer_route_for_tenant(
                db, tenant_id, update_dict.get("transfer_route_id")
            )

        if "name" in update_dict and update_dict["name"]:
            update_dict["name"] = update_dict["name"].strip()

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

        if user_id is not None:
            update_dict["updated_by"] = user_id

        try:
            agent = self._repo(db).update(agent, update_dict)
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Agent role constraint violated (inbound or follow-up uniqueness per tenant).",
            )
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
            ~Agent.is_deleted,
            Agent.id != inbound_agent_id,
        ).all()

        kb_documents = db.query(KnowledgeBase).filter(
            KnowledgeBase.workspace_id == tenant_id,
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
                    "title": doc.name,
                    "source_type": "knowledge_base",
                    "source_ref": str(doc.id),
                    "agent_id": None,
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
    
    def delete_agent(
        self,
        db: Session,
        agent_id: uuid.UUID,
        tenant_id: uuid.UUID,
        *,
        user_id: uuid.UUID | None = None,
    ) -> None:
        """Soft delete; raises 409 when an active phone number is still bound."""
        agent = self.get_agent_by_id(db, agent_id, tenant_id)

        if self.has_active_phone_binding(db, agent.id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    "Agent has an active phone number bound to it. "
                    "Unassign the phone number before deleting."
                ),
            )

        self._repo(db).soft_delete(agent, updated_by=user_id)
    
    def get_agents_by_tenant(self, db: Session, tenant_id: uuid.UUID) -> List[Agent]:
        """
        Get all agents for a specific tenant
        """
        return db.query(Agent).filter(
            Agent.tenant_id == tenant_id,
            ~Agent.is_deleted
        ).all()

    def get_inbound_agent_by_tenant(self, db: Session, tenant_id: uuid.UUID) -> Agent | None:
        """
        Get the dedicated inbound agent for a tenant.
        Returns None if no inbound agent is configured.
        """
        return (
            db.query(Agent)
            .filter(
                Agent.tenant_id == tenant_id,
                ~Agent.is_deleted,
                Agent.is_inbound_agent,
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
            ~Agent.is_deleted,
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
            ~Agent.is_deleted
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

    def build_call_policy_block(
        self,
        *,
        transfer_route: TransferRoute | None = None,
    ) -> str:
        """
        Top-of-prompt operational gates that take priority over style, tone,
        and any custom/model instructions later in the system prompt.

        One gate, emitted only when relevant:
        - Transfer Gate: only when an agent has a ``transfer_route``
          configured. Reinforces that [TRANSFER_CALL] is the only thing that
          actually triggers a transfer.

        Returning the gates as a single block (instead of scattering them
        across the prompt) keeps the policy enforceable on long calls where
        custom instructions and history would otherwise drown the rules out.
        """
        if transfer_route is None:
            return ""

        t_type = (getattr(transfer_route, "transfer_type", None) or "cold").lower()
        friendly = getattr(transfer_route, "friendly_name", None) or "human contact"
        lines: List[str] = [
            "# CALL POLICY (NON-NEGOTIABLE — APPLY IMMEDIATELY)",
            "These rules take priority over style/tone instructions and any custom or model "
            "instructions that appear later in this prompt. Apply them at every turn.",
            "",
            "## 1. Transfer & Escalation Gate",
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
        ]

        return "\n".join(lines).rstrip() + "\n"


agent_service = AgentService()