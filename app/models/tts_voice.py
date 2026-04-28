from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class TTSVoice(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    provider_id = Column(UUID(as_uuid=True), ForeignKey("ttsprovider.id"), nullable=False, index=True)
    external_voice_id = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=False)
    language_code = Column(String(20), nullable=True, index=True)
    gender = Column(String(32), nullable=True)
    accent = Column(String(64), nullable=True)
    description = Column(Text, nullable=True)
    preview_audio_url = Column(String(1000), nullable=True)
    sample_rate_hz = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False, server_default="true", index=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    provider = relationship("TTSProvider", back_populates="voices")
    agents = relationship("Agent", back_populates="tts_voice")

    __table_args__ = (
        UniqueConstraint("provider_id", "external_voice_id", name="uq_ttsvoice_provider_external"),
    )
