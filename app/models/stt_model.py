from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, JSON, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class STTModel(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    provider_id = Column(UUID(as_uuid=True), ForeignKey("sttprovider.id"), nullable=False, index=True)
    external_model_id = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=False)
    language_code = Column(String(20), nullable=False, default="en-US")
    sample_rate_hz = Column(Integer, nullable=False, default=16000)
    encoding = Column(String(20), nullable=False, default="LINEAR16")
    # Provider API mapping — NOT exposed in API responses
    metadata_json = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False, server_default="true", index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    provider = relationship("STTProvider", back_populates="models")
    agents = relationship("Agent", back_populates="stt_model", foreign_keys="Agent.stt_model_id")

    __table_args__ = (
        UniqueConstraint("provider_id", "external_model_id", name="uq_sttmodel_provider_external"),
    )
