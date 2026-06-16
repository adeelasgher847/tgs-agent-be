from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer, Boolean, Index, text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base

class Agent(Base):    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False)
    name = Column(String(255), nullable=False)
    system_prompt = Column(Text, nullable=True)
    language = Column(String(50), nullable=True)
    voice_type = Column(String(100), nullable=True)
    fallback_response = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    # Nullable so machine-to-machine (API key) requests can create/update agents
    # without a resolved user_id; dashboard JWT flows still populate these.
    created_by = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=True)
    updated_by = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=True)
    model_id = Column(UUID(as_uuid=True), ForeignKey("model.id"), nullable=True)
    provider_id = Column(UUID(as_uuid=True), ForeignKey("provider.id"), nullable=True)  # Provider for filtering models
    provider_agent_id = Column(String(255), nullable=True)
    tts_provider_id = Column(UUID(as_uuid=True), ForeignKey("ttsprovider.id"), nullable=True)
    tts_voice_id = Column(UUID(as_uuid=True), ForeignKey("ttsvoice.id"), nullable=True)
    tts_settings_json = Column(JSON, nullable=True)

    # ── Ticket fields (Sprint 2 agent-management API contract) ───────────────
    # Lifecycle state surfaced to the frontend list/detail view.
    status = Column(String(20), nullable=False, default="pending", server_default="pending")
    # String identifier validated against app.core.llm_models.ALLOWED_LLM_MODELS.
    llm_model = Column(String(100), nullable=True)
    # ttsModel triad from the request body — kept as plain strings so the new
    # endpoint stays decoupled from the legacy TTSProvider/TTSVoice catalog.
    tts_provider_slug = Column(String(40), nullable=True)
    tts_voice_external_id = Column(String(255), nullable=True)
    tts_language = Column(String(20), nullable=True)
    # BYO ElevenLabs key — pgp_sym_encrypt at rest (base64 TEXT), never returned in GET responses.
    encrypted_elevenlabs_api_key = Column(Text, nullable=True)
    # Smart callback flag — agent proactively calls back when a slot opens.
    smart_callback = Column(Boolean, default=False, nullable=False, server_default="false")

    # ── Smart Callback Scheduler (Sprint 8) ───────────────────────────────────
    # Enables the ARQ-deferred retry loop for no_answer / busy calls.
    smart_callback_enabled = Column(Boolean, default=False, nullable=False, server_default="false")
    # Maximum number of retry attempts before the chain is exhausted.
    max_callback_attempts = Column(Integer, default=5, nullable=False, server_default="5")
    # Ordered list of {days, hours} gaps between attempts, e.g. [{days:0,hours:1},{days:1,hours:0}].
    callback_gap_schedule = Column(
        JSON, nullable=False, default=list, server_default="[]"
    )
    # IANA timezone used for business-hours enforcement, e.g. "America/New_York".
    callback_timezone = Column(Text, nullable=True)

    # ── STT model triad (mirrors TTS triad: provider_slug / external_id / language) ──
    stt_provider_slug = Column(String(40), nullable=True)
    stt_model_external_id = Column(String(255), nullable=True)
    stt_language_code = Column(String(20), nullable=True)
    stt_provider_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sttprovider.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    stt_model_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sttmodel.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    stt_settings_json = Column(JSON, nullable=True)

    # Agent-specific model configuration (overrides model defaults)
    agent_temperature = Column(Integer, nullable=True)  # Agent-specific temperature (0-100)
    agent_max_tokens = Column(Integer, nullable=True)   # Agent-specific max tokens
    
    # Custom greeting spoken at call start and in response to hi/hello
    greeting_message = Column(Text, nullable=True)

    # Soft delete
    is_deleted = Column(Boolean, default=False, nullable=False, server_default='false')
    # Dedicated inbound entry point agent (max one active per tenant)
    is_inbound_agent = Column(Boolean, default=False, nullable=False, server_default='false')
    # Appointment reminder / follow-up outbound agent (max one active per tenant)
    is_follow_up_agent = Column(Boolean, default=False, nullable=False, server_default='false')

    # Optional default human transfer route (tenant-scoped; see TransferRoute)
    transfer_route_id = Column(
        UUID(as_uuid=True),
        ForeignKey("transferroute.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    
    # Relationships
    tenant = relationship("Tenant", back_populates="agents")
    creator = relationship("User", foreign_keys=[created_by], back_populates="created_agents")
    updater = relationship("User", foreign_keys=[updated_by], back_populates="updated_agents")
    call_sessions = relationship("CallSession", back_populates="agent")
    transcript_messages = relationship("TranscriptMessage", back_populates="agent")
    appointments = relationship("Appointment", back_populates="agent")
    model = relationship("Model")
    provider = relationship("Provider")  # Provider relationship for filtering models
    tts_provider = relationship("TTSProvider", back_populates="agents")
    tts_voice = relationship("TTSVoice", back_populates="agents")
    stt_provider = relationship("STTProvider", back_populates="agents", foreign_keys=[stt_provider_id])
    stt_model = relationship("STTModel", back_populates="agents", foreign_keys=[stt_model_id])
    transfer_route = relationship("TransferRoute", back_populates="agents")
    callback_schedules = relationship("CallbackSchedule", back_populates="agent")

    __table_args__ = (
        Index("ix_agent_tenant_id", "tenant_id"),
        Index(
            "uq_agent_single_inbound_per_tenant",
            "tenant_id",
            unique=True,
            postgresql_where=text("is_inbound_agent = true AND is_deleted = false"),
        ),
        Index(
            "uq_agent_single_follow_up_per_tenant",
            "tenant_id",
            unique=True,
            postgresql_where=text("is_follow_up_agent = true AND is_deleted = false"),
        ),
    )