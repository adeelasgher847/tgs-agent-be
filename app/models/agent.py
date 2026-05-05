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
    created_by = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    updated_by = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    model_id = Column(UUID(as_uuid=True), ForeignKey("model.id"), nullable=True)
    provider_id = Column(UUID(as_uuid=True), ForeignKey("provider.id"), nullable=True)  # Provider for filtering models
    provider_agent_id = Column(String(255), nullable=True)
    tts_provider_id = Column(UUID(as_uuid=True), ForeignKey("ttsprovider.id"), nullable=True)
    tts_voice_id = Column(UUID(as_uuid=True), ForeignKey("ttsvoice.id"), nullable=True)
    tts_settings_json = Column(JSON, nullable=True)
    
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
    transfer_route = relationship("TransferRoute", back_populates="agents")

    __table_args__ = (
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