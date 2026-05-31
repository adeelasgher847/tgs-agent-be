from sqlalchemy import Column, String, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class TenantInboundCRMConfig(Base):
    """
    Per-tenant inbound call log → CRM sync (Trello first).
    Separate from global CRMConfig used for scheduled calls.
    """

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False, unique=True, index=True)

    provider = Column(String(20), nullable=False, server_default="trello")
    connection_type = Column(String(30), nullable=False, server_default="byo_credentials")

    encrypted_api_key = Column(Text, nullable=True)
    encrypted_api_token = Column(Text, nullable=True)

    container_id = Column(String(200), nullable=True, index=True)
    container_url = Column(String(500), nullable=True)
    default_list_id = Column(String(200), nullable=True)

    extra_config = Column(JSONB, nullable=True)

    is_enabled = Column(Boolean, nullable=False, server_default="false")

    created_by_user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    tenant = relationship("Tenant", backref=backref("inbound_crm_config", uselist=False))
    creator = relationship("User", foreign_keys=[created_by_user_id])

    def __repr__(self) -> str:  # pragma: no cover
        return f"<TenantInboundCRMConfig(tenant_id={self.tenant_id}, provider={self.provider})>"
