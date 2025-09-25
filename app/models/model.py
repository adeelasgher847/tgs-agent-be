from sqlalchemy import Column, String, Integer, DateTime, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base

class Model(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    first_message = Column(String, nullable=True)
    system_prompt = Column(String, nullable=True)
    files = Column(String, nullable=True)
    max_tokens = Column(Integer, nullable=True)
    temp = Column(Integer, nullable=True)

    # Proper foreign key to Provider
    provider_id = Column(UUID(as_uuid=True), ForeignKey("provider.id"), nullable=False)

    # Commonly used in codebase
    is_active = Column(Boolean, default=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship back to Provider
    provider = relationship("Provider", back_populates="models")
