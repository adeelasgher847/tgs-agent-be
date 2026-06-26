"""API endpoints for managing workspace SSO configurations."""

import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import httpx

from app.api.deps import get_db, require_admin, require_tenant
from app.models.sso_config import SsoConfig
from app.models.user import User
from app.schemas.sso import SsoConfigUpsert, SsoConfigOut, SsoTestResult
from app.core.sso_crypto import encrypt_secret

router = APIRouter()


@router.post("", response_model=SsoConfigOut)
def upsert_sso_config(
    payload: SsoConfigUpsert,
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Create or update the SSO configuration for the current workspace."""
    workspace_id = user.current_tenant_id
    config = db.query(SsoConfig).filter(SsoConfig.workspace_id == workspace_id).first()

    if not config:
        config = SsoConfig(workspace_id=workspace_id)
        db.add(config)
        
    config.protocol = payload.protocol
    config.is_active = payload.is_active
    
    if payload.protocol == 'saml':
        config.idp_entity_id = payload.idp_entity_id
        config.idp_sso_url = payload.idp_sso_url
        config.idp_x509_certificate = payload.idp_x509_certificate
    elif payload.protocol == 'oidc':
        config.oidc_client_id = payload.oidc_client_id
        config.oidc_discovery_url = payload.oidc_discovery_url
        if payload.oidc_client_secret and payload.oidc_client_secret != "***":
            config.oidc_client_secret = encrypt_secret(payload.oidc_client_secret)
            
    db.commit()
    db.refresh(config)
    
    return _to_out(config)

@router.get("", response_model=SsoConfigOut)
def get_sso_config(
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Get the current SSO configuration."""
    workspace_id = user.current_tenant_id
    config = db.query(SsoConfig).filter(SsoConfig.workspace_id == workspace_id).first()
    
    if not config:
        raise HTTPException(status_code=404, detail="SSO configuration not found.")
        
    return _to_out(config)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
def disable_sso(
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Disable SSO for the current workspace (sets is_active=false)."""
    workspace_id = user.current_tenant_id
    config = db.query(SsoConfig).filter(SsoConfig.workspace_id == workspace_id).first()
    
    if config:
        config.is_active = False
        db.commit()


@router.post("/test", response_model=SsoTestResult)
async def test_sso_connection(
    user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Test IdP connectivity (e.g. discovery URL or SAML endpoint)."""
    workspace_id = user.current_tenant_id
    config = db.query(SsoConfig).filter(SsoConfig.workspace_id == workspace_id).first()
    
    if not config:
        return SsoTestResult(success=False, error="SSO configuration not found.")
        
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            if config.protocol == 'oidc':
                if not config.oidc_discovery_url:
                    return SsoTestResult(success=False, error="Missing OIDC discovery URL.")
                res = await client.get(config.oidc_discovery_url)
                res.raise_for_status()
            elif config.protocol == 'saml':
                if not config.idp_sso_url:
                    return SsoTestResult(success=False, error="Missing IdP SSO URL.")
                res = await client.head(config.idp_sso_url, follow_redirects=True)
                res.raise_for_status()
                
        return SsoTestResult(success=True)
    except Exception as e:
        return SsoTestResult(success=False, error=str(e))


def _to_out(config: SsoConfig) -> SsoConfigOut:
    trunc_cert = None
    if config.idp_x509_certificate:
        cert_clean = config.idp_x509_certificate.replace("-----BEGIN CERTIFICATE-----", "").replace("-----END CERTIFICATE-----", "").strip()
        trunc_cert = cert_clean[:40] + "..." if len(cert_clean) > 40 else cert_clean
        
    return SsoConfigOut(
        id=config.id,
        workspace_id=config.workspace_id,
        protocol=config.protocol,
        idp_entity_id=config.idp_entity_id,
        idp_sso_url=config.idp_sso_url,
        idp_x509_certificate_truncated=trunc_cert,
        oidc_client_id=config.oidc_client_id,
        oidc_client_secret="***",
        oidc_discovery_url=config.oidc_discovery_url,
        is_active=config.is_active,
        created_at=config.created_at,
        updated_at=config.updated_at
    )
