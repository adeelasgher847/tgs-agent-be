"""Router for SAML and OIDC authentication flows."""

import secrets
import json
import base64
import time
import hmac
import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session
from onelogin.saml2.auth import OneLogin_Saml2_Auth
from onelogin.saml2.settings import OneLogin_Saml2_Settings

from authlib.integrations.httpx_client import AsyncOAuth2Client

from app.db.async_session import get_db
from app.models.tenant import Tenant
from app.models.sso_config import SsoConfig
from app.core.sso_crypto import decrypt_secret
from app.services.sso_service import find_or_create_user
from app.api.deps import issue_tokens_for_user
from app.core.config import settings

router = APIRouter()

_OIDC_DISCOVERY_CACHE: dict[str, tuple[float, dict]] = {}

async def _fetch_oidc_discovery(url: str) -> dict:
    now = time.time()
    if url in _OIDC_DISCOVERY_CACHE:
        expiry, data = _OIDC_DISCOVERY_CACHE[url]
        if now < expiry:
            return data
            
    async with httpx.AsyncClient() as hc:
        res = await hc.get(url)
        res.raise_for_status()
        data = res.json()
        
    _OIDC_DISCOVERY_CACHE[url] = (now + 3600.0, data)  # cache for 1 hour
    return data

def get_tenant_by_slug(db: Session, slug: str) -> Tenant:
    tenant = db.query(Tenant).filter(Tenant.workspace_slug == slug, Tenant.deleted_at.is_(None)).first()
    if not tenant:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    return tenant

def get_sso_config(db: Session, workspace_id) -> SsoConfig:
    config = db.query(SsoConfig).filter(SsoConfig.workspace_id == workspace_id, SsoConfig.is_active.is_(True)).first()
    if not config:
        raise HTTPException(status_code=404, detail="SSO not configured for this workspace.")
    return config

def prepare_saml_req(request: Request, config: SsoConfig, slug: str) -> dict:
    return {
        'https': 'on' if request.url.scheme == 'https' else 'off',
        'http_host': request.url.hostname,
        'server_port': request.url.port,
        'script_name': request.url.path,
        'get_data': dict(request.query_params),
        'post_data': {}, # We will manually pass the post data later
    }


def get_saml_settings(config: SsoConfig, slug: str) -> dict:
    base_url = settings.WEBHOOK_BASE_URL.rstrip('/')
    return {
        "strict": True,
        "debug": False,
        "sp": {
            "entityId": f"{base_url}/auth/saml/{slug}/metadata",
            "assertionConsumerService": {
                "url": f"{base_url}/auth/saml/{slug}/callback",
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
            },
            "NameIDFormat": "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress",
            "x509cert": "",
            "privateKey": ""
        },
        "idp": {
            "entityId": config.idp_entity_id,
            "singleSignOnService": {
                "url": config.idp_sso_url,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
            },
            "x509cert": config.idp_x509_certificate
        }
    }


@router.get("/auth/saml/{workspace_slug}/metadata")
def saml_metadata(
    workspace_slug: str,
    db: Session = Depends(get_db)
):
    tenant = get_tenant_by_slug(db, workspace_slug)
    config = get_sso_config(db, tenant.id)
    
    saml_settings = get_saml_settings(config, workspace_slug)
    saml_settings_obj = OneLogin_Saml2_Settings(saml_settings)
    metadata = saml_settings_obj.get_sp_metadata()
    errors = saml_settings_obj.validate_metadata(metadata)
    
    if len(errors) > 0:
        raise HTTPException(status_code=500, detail=f"Invalid SP metadata: {', '.join(errors)}")
        
    return Response(content=metadata, media_type="text/xml")


@router.get("/auth/saml/{workspace_slug}/login")
def saml_login(
    workspace_slug: str,
    request: Request,
    db: Session = Depends(get_db)
):
    tenant = get_tenant_by_slug(db, workspace_slug)
    config = get_sso_config(db, tenant.id)
    
    req = prepare_saml_req(request, config, workspace_slug)
    auth = OneLogin_Saml2_Auth(req, get_saml_settings(config, workspace_slug))
    
    relay_state = secrets.token_urlsafe(32)
    sso_built_url = auth.login(return_to=relay_state)
    
    res = Response(status_code=302, headers={"Location": sso_built_url})
    res.set_cookie(
        key="saml_state",
        value=relay_state,
        httponly=True,
        max_age=300,
        samesite="lax",
        secure=settings.ENVIRONMENT.lower() in ("staging", "production")
    )
    return res


@router.post("/auth/saml/{workspace_slug}/callback")
async def saml_callback(
    workspace_slug: str,
    request: Request,
    db: Session = Depends(get_db)
):
    tenant = get_tenant_by_slug(db, workspace_slug)
    config = get_sso_config(db, tenant.id)
    
    form_data = await request.form()
    
    # CSRF check using RelayState parameter from IdP and saml_state cookie
    relay_state_from_idp = form_data.get("RelayState", "")
    cookie_state = request.cookies.get("saml_state", "")
    if not relay_state_from_idp or not hmac.compare_digest(relay_state_from_idp, cookie_state):
        raise HTTPException(status_code=400, detail="Invalid RelayState — CSRF check failed")
        
    req = prepare_saml_req(request, config, workspace_slug)
    req['post_data'] = dict(form_data)
    
    auth = OneLogin_Saml2_Auth(req, get_saml_settings(config, workspace_slug))
    auth.process_response()
    
    errors = auth.get_errors()
    if not auth.is_authenticated() or errors:
        raise HTTPException(status_code=401, detail=f"SAML Authentication Failed: {', '.join(errors)}")
        
    email = auth.get_nameid()
    if not email:
        raise HTTPException(status_code=401, detail="SAML Response missing NameID (email).")
        
    user, role_info = find_or_create_user(db, email, tenant.id)
    token_resp = issue_tokens_for_user(db, user, tenant.id, role_info)
    
    frontend_url = settings.DASHBOARD_URL or "http://localhost:3000"
    res = Response(status_code=302, headers={"Location": f"{frontend_url}/dashboard"})
    res.set_cookie(
        key="access_token",
        value=token_resp.access_token,
        httponly=True,
        samesite="lax",
        secure=settings.ENVIRONMENT.lower() in ("staging", "production")
    )
    res.delete_cookie("saml_state")
    return res


@router.get("/auth/oidc/{workspace_slug}/login")
async def oidc_login(
    workspace_slug: str,
    request: Request,
    db: Session = Depends(get_db)
):
    tenant = get_tenant_by_slug(db, workspace_slug)
    config = get_sso_config(db, tenant.id)
    
    client_secret = decrypt_secret(config.oidc_client_secret)
    client = AsyncOAuth2Client(
        client_id=config.oidc_client_id,
        client_secret=client_secret,
    )
    
    # We need to fetch the authorization endpoint from discovery URL
    try:
        discovery_data = await _fetch_oidc_discovery(config.oidc_discovery_url)
    except Exception as exc:
        from app.core.logger import logger
        logger.error("Failed to fetch OIDC discovery document: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Failed to retrieve OIDC discovery document")
        
    authorization_endpoint = discovery_data.get('authorization_endpoint')
    if not authorization_endpoint:
        raise HTTPException(status_code=500, detail="Discovery URL missing authorization_endpoint")
        
    base_url = settings.WEBHOOK_BASE_URL.rstrip('/')
    redirect_uri = f"{base_url}/auth/oidc/{workspace_slug}/callback"
    
    state = secrets.token_urlsafe(32)
    uri, state = client.create_authorization_url(
        authorization_endpoint,
        redirect_uri=redirect_uri,
        scope='openid email profile',
        state=state
    )
    
    res = Response(status_code=302, headers={"Location": uri})
    res.set_cookie(key="oidc_state", value=state, httponly=True, max_age=300)
    return res


@router.get("/auth/oidc/{workspace_slug}/callback")
async def oidc_callback(
    workspace_slug: str,
    request: Request,
    db: Session = Depends(get_db)
):
    tenant = get_tenant_by_slug(db, workspace_slug)
    config = get_sso_config(db, tenant.id)
    
    state = request.query_params.get('state')
    cookie_state = request.cookies.get('oidc_state')
    
    if not state or state != cookie_state:
        raise HTTPException(status_code=400, detail="Invalid state parameter.")
        
    code = request.query_params.get('code')
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code.")
        
    client_secret = decrypt_secret(config.oidc_client_secret)
    base_url = settings.WEBHOOK_BASE_URL.rstrip('/')
    redirect_uri = f"{base_url}/auth/oidc/{workspace_slug}/callback"
    
    try:
        discovery_data = await _fetch_oidc_discovery(config.oidc_discovery_url)
    except Exception as exc:
        from app.core.logger import logger
        logger.error("Failed to fetch OIDC discovery document: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Failed to retrieve OIDC discovery document")
        
    token_endpoint = discovery_data.get('token_endpoint')
    
    client = AsyncOAuth2Client(
        client_id=config.oidc_client_id,
        client_secret=client_secret,
    )
    
    token = await client.fetch_token(
        token_endpoint,
        authorization_response=str(request.url),
        redirect_uri=redirect_uri,
        grant_type='authorization_code'
    )
    
    if 'id_token' not in token:
        raise HTTPException(status_code=401, detail="Missing id_token in response.")
        
    # authlib parses the id_token automatically
    userinfo = token.get('userinfo')
    if not userinfo:
        # Some providers require calling userinfo_endpoint
        userinfo_endpoint = discovery_data.get('userinfo_endpoint')
        if userinfo_endpoint:
            async with httpx.AsyncClient() as hc:
                res = await hc.get(
                    userinfo_endpoint,
                    headers={"Authorization": f"Bearer {token['access_token']}"}
                )
                res.raise_for_status()
                userinfo = res.json()
                
    if not userinfo or 'email' not in userinfo:
        raise HTTPException(status_code=401, detail="Email claim missing from IdP response.")
        
    email = userinfo['email']
    user, role_info = find_or_create_user(db, email, tenant.id)
    token_resp = issue_tokens_for_user(db, user, tenant.id, role_info)
    
    frontend_url = settings.DASHBOARD_URL or "http://localhost:3000"
    res = Response(status_code=302, headers={"Location": f"{frontend_url}/dashboard"})
    res.set_cookie(
        key="access_token",
        value=token_resp.access_token,
        httponly=True,
        samesite="lax",
        secure=settings.ENVIRONMENT.lower() in ("staging", "production")
    )
    res.delete_cookie("oidc_state")
    return res
