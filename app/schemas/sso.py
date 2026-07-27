import uuid
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, field_validator


class SsoConfigUpsert(BaseModel):
    """Payload for creating or updating an SSO configuration."""
    protocol: Literal['saml', 'oidc']
    
    # SAML
    idp_entity_id: str | None = None
    idp_sso_url: str | None = None
    idp_x509_certificate: str | None = None
    
    # OIDC
    oidc_client_id: str | None = None
    oidc_client_secret: str | None = None
    oidc_discovery_url: str | None = None
    
    is_active: bool = False
    
    # Allowed email domains for auto-provisioning
    allowed_email_domains: list[str] | None = None

    @field_validator("allowed_email_domains")
    @classmethod
    def validate_allowed_email_domains(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            return [d.lower().strip() for d in v if d.strip()]
        return v

    @field_validator("oidc_discovery_url")
    @classmethod
    def validate_oidc_discovery_url(cls, v: str | None) -> str | None:
        if v:
            from app.utils.ssrf import assert_public_url, SSRFBlockedError
            try:
                assert_public_url(v)
            except SSRFBlockedError as exc:
                raise ValueError(str(exc))
        return v



class SsoConfigOut(BaseModel):
    """Safe response containing the SSO configuration with secrets masked."""
    id: uuid.UUID
    workspace_id: uuid.UUID
    protocol: str
    
    idp_entity_id: str | None
    idp_sso_url: str | None
    idp_x509_certificate_truncated: str | None
    
    oidc_client_id: str | None
    oidc_client_secret: str = "***"
    oidc_discovery_url: str | None
    
    is_active: bool
    allowed_email_domains: list[str] | None = None
    created_at: datetime
    updated_at: datetime | None

    model_config = {
        "from_attributes": True
    }


class SsoTestResult(BaseModel):
    success: bool
    error: str | None = None
