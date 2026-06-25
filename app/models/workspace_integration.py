from sqlalchemy import Column, String, DateTime, ForeignKey, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base_class import Base


class WorkspaceIntegration(Base):
    """Third-party CRM/integration connection for a workspace (tenant). One row per provider."""

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("tenant.id"), nullable=False, index=True)

    provider = Column(String(50), nullable=False)  # "hubspot"

    # pgp_sym_encrypt ciphertext (base64) — see app/core/db_encryption.py.
    access_token = Column(Text, nullable=True)
    refresh_token = Column(Text, nullable=True)
    token_expires_at = Column(DateTime(timezone=True), nullable=True)

    # SQLAlchemy's declarative Base reserves the `metadata` attribute name for the
    # schema MetaData registry, so the Python attribute is `extra_metadata` while
    # the underlying DB column stays literally named `metadata`.
    extra_metadata = Column("metadata", JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    workspace = relationship("Tenant", foreign_keys=[workspace_id])

    __table_args__ = (
        UniqueConstraint("workspace_id", "provider", name="uq_workspace_integration_provider"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<WorkspaceIntegration(workspace_id={self.workspace_id}, provider={self.provider})>"
