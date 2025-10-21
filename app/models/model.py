from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Boolean, Text , Float , event
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base
from app.services.pricing_service import pricing_service
class Model(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    provider_id = Column(UUID(as_uuid=True), ForeignKey("provider.id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    api_key = Column(String(500), nullable=True)  # Model-specific API key
    description = Column(Text, nullable=True)  # Free tokens, efficiency, pricing details
    system_prompt = Column(String(1000), nullable=True)
    temperature = Column(Integer, nullable=True)
    max_tokens = Column(Integer, nullable=True)
    archive = Column(Boolean, default=True)
    llm_cost_per_minute = Column(Float, nullable=True)
    twilio_cost_per_minute = Column(Float, default=0.0140)  # constant for all voice models
    total_cost_per_minute = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    provider = relationship("Provider", back_populates="models")

# ===== Event listeners =====
@event.listens_for(Model, "before_insert")
def set_pricing_before_insert(mapper, connection, target):
    pricing = pricing_service.get_pricing_for_model(target.model_name)
    target.llm_cost_per_minute = round(pricing["llm_cost_per_minute"] or 0.0, 6)
    target.total_cost_per_minute = round(pricing["total_cost_per_minute"] or 0.0, 6)
    target.twilio_cost_per_minute = pricing["twilio_cost_per_minute"]  # always 0.0140

@event.listens_for(Model, "before_update")
def set_pricing_before_update(mapper, connection, target):
    pricing = pricing_service.get_pricing_for_model(target.model_name)
    target.llm_cost_per_minute = pricing["llm_cost_per_minute"] or 0.0
    target.total_cost_per_minute = pricing["total_cost_per_minute"] or 0.0
    target.twilio_cost_per_minute = pricing["twilio_cost_per_minute"]