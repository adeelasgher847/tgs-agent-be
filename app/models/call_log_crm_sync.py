from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class CallLogCRMSync(Base):
    """Tracks CRM push for each call log (idempotent, auditable)."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    call_log_id = Column(UUID(as_uuid=True), ForeignKey("calllog.id"), nullable=False, unique=True, index=True)
    tenant_inbound_crm_config_id = Column(
        UUID(as_uuid=True), ForeignKey("tenantinboundcrmconfig.id"), nullable=False, index=True
    )

    external_item_id = Column(String(200), nullable=True, index=True)
    external_item_url = Column(String(800), nullable=True)

    sync_status = Column(String(20), nullable=False, server_default="pending")
    last_error = Column(Text, nullable=True)
    attempt_count = Column(Integer, nullable=False, server_default="0")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    call_log = relationship("CallLog", backref=backref("crm_sync", uselist=False))
    inbound_config = relationship("TenantInboundCRMConfig", backref="call_syncs")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<CallLogCRMSync(call_log_id={self.call_log_id}, status={self.sync_status})>"
