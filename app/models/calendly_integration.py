from sqlalchemy import Column, DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class CalendlyIntegration(Base):
    """Calendly OAuth connection for a workspace (tenant). One row per workspace."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False, unique=True, index=True)

    # AES-256-GCM ciphertext — see app/core/db_encryption.py::encrypt_calendly_token.
    access_token = Column(Text, nullable=True)
    refresh_token = Column(Text, nullable=True)
    token_expires_at = Column(DateTime(timezone=True), nullable=True)

    calendly_user_uri = Column(Text, nullable=True)         # e.g. https://api.calendly.com/users/<uuid>
    calendly_event_type_uri = Column(Text, nullable=True)   # e.g. https://api.calendly.com/event_types/<uuid>

    connected_by_user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    workspace = relationship("Tenant", foreign_keys=[workspace_id])
    connected_by_user = relationship("User", foreign_keys=[connected_by_user_id])

    __table_args__ = (
        UniqueConstraint("workspace_id", name="uq_calendly_integration_workspace"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<CalendlyIntegration(workspace_id={self.workspace_id})>"
