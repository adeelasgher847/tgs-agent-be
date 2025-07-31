from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base_class import Base

class VoiceAgent(Base):
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenant.id" ), nullable=False)
    name = Column(String(255), nullable=False)
    system_prompt = Column(Text, nullable=True)
    language = Column(String(50), nullable=True)
    voice_type = Column(String(100), nullable=True)
    fallback_response = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship back to tenant
    tenants = relationship("Tenant", back_populates="voice_agents")