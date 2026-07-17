from sqlalchemy import (
    Column,
    String,
    DateTime,
    Integer,
    Boolean,
    ForeignKey,
    UniqueConstraint,
    CheckConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
from app.db.base_class import Base


class PhoneNumber(Base):
    """Phone number model — supports Twilio-provisioned and BYO/SIP external numbers."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False, index=True)

    # E.164 phone number e.g. +614xxxxxxxx
    phone_number = Column(String(20), nullable=False, index=True)
    label = Column(String(100), nullable=True)
    status = Column(String(20), nullable=False, default="active")  # active, inactive

    # Provider: 'twilio' (provisioned via API) | 'external' (BYO / SIP)
    provider = Column(String(20), nullable=False, default="twilio", server_default="twilio")

    # Twilio SID — set after purchase; null for external numbers
    twilio_phone_number_sid = Column(String(255), nullable=True, index=True)

    # Encrypted per-number credentials (legacy import path — kept for backward compat).
    # New provisioning uses Secret Manager credentials from app.core.secret_manager.
    twilio_account_sid = Column(String(255), nullable=True)
    twilio_auth_token = Column(String(500), nullable=True)  # encrypted at rest

    # SIP/BYO external number credentials — sip_password encrypted via encrypt_api_key() before persist
    sip_username = Column(String(255), nullable=True)
    sip_password = Column(String(500), nullable=True)  # JWT-encrypted at rest (see app.core.security)

    # Agent binding (FK agent.id). Column name assistant_id is legacy; APIs also accept/return agent_id.
    assistant_id = Column(UUID(as_uuid=True), ForeignKey("agent.id"), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    tenant = relationship("Tenant", back_populates="phone_numbers")
    configuration = relationship(
        "NumberConfiguration",
        back_populates="phone_number_obj",
        uselist=False,
        cascade="all, delete-orphan",
    )
    reputation = relationship(
        "PhoneNumberReputation",
        back_populates="phone_number",
        uselist=False,
        cascade="all, delete-orphan",
    )

    # Global unique constraint: a phone number can belong to only one tenant.
    __table_args__ = (
        UniqueConstraint("phone_number", name="uq_phone_number_global"),
        CheckConstraint(
            "provider IN ('twilio', 'external')",
            name="ck_phonenumber_provider",
        ),
    )

    def __repr__(self) -> str:
        return f"<PhoneNumber(id={self.id}, number={self.phone_number}, provider={self.provider})>"


class NumberConfiguration(Base):
    """
    Per-number configuration: recording, duration limits, business hours.
    Table name follows Base: NumberConfiguration -> numberconfiguration.
    """

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    phone_number_id = Column(
        UUID(as_uuid=True),
        ForeignKey("phonenumber.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    recording_enabled = Column(Boolean, nullable=False, default=False, server_default="false")
    max_duration_seconds = Column(Integer, nullable=False, default=3600, server_default="3600")

    # JSONB business hours format:
    # {"timezone": "Australia/Sydney", "schedule": [{"day": 0, "open": "09:00", "close": "17:00"}]}
    business_hours = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    phone_number_obj = relationship("PhoneNumber", back_populates="configuration")
