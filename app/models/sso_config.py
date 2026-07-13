"""SQLAlchemy model for SSO configuration per workspace."""
from __future__ import annotations

import uuid

from sqlalchemy import Boolean, CheckConstraint, Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base


class SsoConfig(Base):
    """Stores IdP settings for SAML 2.0 and OIDC SSO.

    Secrets (oidc_client_secret, idp_x509_certificate) are NEVER logged.
    oidc_client_secret is Fernet-encrypted at rest (see app.core.sso_crypto).
    """

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tenant.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    protocol = Column(String(10), nullable=False)  # 'saml' | 'oidc'

    # SAML fields
    idp_entity_id = Column(Text, nullable=True)
    idp_sso_url = Column(Text, nullable=True)
    idp_x509_certificate = Column(Text, nullable=True)  # full PEM — never log

    # OIDC fields
    oidc_client_id = Column(Text, nullable=True)
    oidc_client_secret = Column(Text, nullable=True)   # Fernet-encrypted — never log
    oidc_discovery_url = Column(Text, nullable=True)

    is_active = Column(Boolean, nullable=False, default=False, server_default="false")
    
    # Allowed email domains for auto-provisioning (e.g. ["acme.com", "acme.org"])
    allowed_email_domains = Column(
        postgresql_JSONB := __import__("sqlalchemy.dialects.postgresql", fromlist=["JSONB"]).JSONB,
        nullable=True,
        server_default="[]"
    )

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    workspace = relationship("Tenant", back_populates="sso_config")

    __table_args__ = (
        CheckConstraint("protocol IN ('saml', 'oidc')", name="chk_sso_protocol"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<SsoConfig(workspace_id={self.workspace_id}, protocol={self.protocol}, active={self.is_active})>"
