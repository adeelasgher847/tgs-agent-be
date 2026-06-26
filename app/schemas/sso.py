import uuid
from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel


class SsoConfigUpsert(BaseModel):
    """Payload for creating or updating an SSO configuration."""
    protocol: Literal['saml', 'oidc']
    
    # SAML
    idp_entity_id: Optional[str] = None
    idp_sso_url: Optional[str] = None
    idp_x509_certificate: Optional[str] = None
    
    # OIDC
    oidc_client_id: Optional[str] = None
    oidc_client_secret: Optional[str] = None
    oidc_discovery_url: Optional[str] = None
    
    is_active: bool = False


class SsoConfigOut(BaseModel):
    """Safe response containing the SSO configuration with secrets masked."""
    id: uuid.UUID
    workspace_id: uuid.UUID
    protocol: str
    
    idp_entity_id: Optional[str]
    idp_sso_url: Optional[str]
    idp_x509_certificate_truncated: Optional[str]
    
    oidc_client_id: Optional[str]
    oidc_client_secret: str = "***"
    oidc_discovery_url: Optional[str]
    
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]

    model_config = {
        "from_attributes": True
    }


class SsoTestResult(BaseModel):
    success: bool
    error: Optional[str] = None
