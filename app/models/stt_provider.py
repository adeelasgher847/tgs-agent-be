from sqlalchemy import Boolean, Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class STTProvider(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    slug = Column(String(50), nullable=False, unique=True, index=True)
    display_name = Column(String(100), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, server_default="true")
    supports_streaming = Column(Boolean, default=True, nullable=False, server_default="true")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    models = relationship("STTModel", back_populates="provider", cascade="all, delete-orphan")
    agents = relationship("Agent", back_populates="stt_provider", foreign_keys="Agent.stt_provider_id")
