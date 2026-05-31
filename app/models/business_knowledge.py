from __future__ import annotations

import uuid
from sqlalchemy import Column, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db.base_class import Base


class BusinessKnowledge(Base):
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False, index=True)
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agent.id"), nullable=True, index=True)

    # Admin-facing label to distinguish multiple records
    label = Column(String(255), nullable=False)

    # Core business identity — all stored in natural spoken form
    business_name = Column(String(255), nullable=True)
    business_type = Column(String(255), nullable=True)
    business_description = Column(Text, nullable=True)

    # Contact & location — spoken-word format (e.g. "one two three Main Street")
    address = Column(Text, nullable=True)
    phone = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True)
    website_url = Column(String(512), nullable=True)

    # Services
    primary_service = Column(Text, nullable=True)
    secondary_service = Column(Text, nullable=True)
    service_areas = Column(Text, nullable=True)
    specializations = Column(Text, nullable=True)

    # Additional context
    pricing_information = Column(Text, nullable=True)
    additional_information = Column(Text, nullable=True)

    # Soft delete flag (False = deleted)
    is_active = Column(Boolean, nullable=False, default=True, server_default="true")

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
