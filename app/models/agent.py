from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Integer
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
    provider_agent_id = Column(String(255), nullable=True)
    
    # Agent-specific model configuration (overrides model defaults)
    agent_temperature = Column(Integer, nullable=True)  # Agent-specific temperature (0-100)
    agent_max_tokens = Column(Integer, nullable=True)   # Agent-specific max tokens
    
    # Relationships
    tenant = relationship("Tenant", back_populates="agents")
    creator = relationship("User", foreign_keys=[created_by], back_populates="created_agents")
    updater = relationship("User", foreign_keys=[updated_by], back_populates="updated_agents")
    call_sessions = relationship("CallSession", back_populates="agent")
    model = relationship("Model")