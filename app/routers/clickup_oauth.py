"""
ClickUp OAuth 2.0 Integration Router
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session
from typing import Optional
import requests
import uuid
import json

from app.api.deps import get_db, require_owner
from app.models.user import User
from app.services.crm_config_service import CRMConfigService
from app.core.security import encrypt_api_key, decrypt_api_key, is_api_key_encrypted
from app.core.config import settings
from app.utils.response import create_success_response
from app.utils.n8n_webhook_verification import verify_n8n_webhook_secret_async
from app.schemas.base import SuccessResponse

router = APIRouter()

CLICKUP_AUTH_URL = "https://app.clickup.com/api"
CLICKUP_TOKEN_URL = "https://api.clickup.com/api/v2/oauth/token"


@router.get("/authorize")
async def clickup_authorize(
    user: User = Depends(require_owner),
    db: Session = Depends(get_db)
):
    """
    Generate ClickUp OAuth authorization URL.
    Returns URL that user should visit to authorize the app.
    """
    # Get ClickUp config to find client_id
    crm_config_service = CRMConfigService()
    clickup_config = crm_config_service.get_crm_config_by_type(db, "clickup")
    
    if not clickup_config:
        raise HTTPException(
            status_code=404,
            detail="ClickUp configuration not found. Please create CRM config first with client_id and client_secret."
        )
    
    # Get client_id from additional_config
    if not clickup_config.additional_config:
        raise HTTPException(
            status_code=400,
            detail="ClickUp client_id not found in additional_config. Please update CRM config with client_id and client_secret."
        )
    
    additional_config = json.loads(clickup_config.additional_config)
    
    # Handle double nesting if present
    if "additional_config" in additional_config and isinstance(additional_config.get("additional_config"), dict):
        additional_config = additional_config["additional_config"]
    
    client_id = additional_config.get("client_id")
    
    if not client_id:
        raise HTTPException(
            status_code=400,
            detail="ClickUp client_id not found. Please update CRM config with client_id in additional_config."
        )
    
    # Get redirect_uri from additional_config or use default
    redirect_uri = additional_config.get("redirect_uri") or f"{settings.WEBHOOK_BASE_URL}/api/v1/auth/clickup/callback"
    
    # Generate state (optional, for security)
    state = str(uuid.uuid4())
    
    # Store state in additional_config temporarily (or use session/cache)
    # For now, we'll just use it in the URL
    
    # Build authorization URL with required scopes
    # Scopes needed: read (to read teams/spaces), write (to create lists)
    scopes = "read write"
    auth_url = (
        f"{CLICKUP_AUTH_URL}?"
        f"client_id={client_id}&"
        f"redirect_uri={redirect_uri}&"
        f"scope={scopes}"
    )
    
    return create_success_response(
        data={
            "authorization_url": auth_url,
            "redirect_uri": redirect_uri,
            "instructions": "Visit the authorization_url to authorize the app. After authorization, you'll be redirected to the callback URL."
        },
        message="ClickUp authorization URL generated successfully"
    )


@router.get("/callback")
async def clickup_oauth_callback(
    code: str = Query(..., description="Authorization code from ClickUp"),
    state: Optional[str] = Query(None, description="State parameter (optional)"),
    db: Session = Depends(get_db)
):
    """
    ClickUp OAuth callback endpoint.
    Receives authorization code and exchanges it for access token.
    """
    # Get ClickUp config
    crm_config_service = CRMConfigService()
    clickup_config = crm_config_service.get_crm_config_by_type(db, "clickup")
    
    if not clickup_config:
        raise HTTPException(
            status_code=404,
            detail="ClickUp configuration not found"
        )
    
    # Get client_id and client_secret from additional_config
    if not clickup_config.additional_config:
        raise HTTPException(
            status_code=400,
            detail="ClickUp client credentials not found in additional_config"
        )
    
    additional_config = json.loads(clickup_config.additional_config)
    
    # Handle double nesting if present
    if "additional_config" in additional_config and isinstance(additional_config.get("additional_config"), dict):
        additional_config = additional_config["additional_config"]
    
    client_id = additional_config.get("client_id")
    client_secret_encrypted = additional_config.get("client_secret")
    redirect_uri = additional_config.get("redirect_uri") or f"{settings.WEBHOOK_BASE_URL}/api/v1/auth/clickup/callback"
    
    if not client_id or not client_secret_encrypted:
        raise HTTPException(
            status_code=400,
            detail="ClickUp client_id or client_secret not found in additional_config"
        )
    
    # Decrypt client_secret (check if encrypted first)
    try:
        if is_api_key_encrypted(client_secret_encrypted):
            client_secret = decrypt_api_key(client_secret_encrypted)
        else:
            # Already plain text, use as is
            client_secret = client_secret_encrypted
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to decrypt client_secret: {str(e)}"
        )
    
    # Exchange authorization code for access token
    try:
        token_data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code
        }
        
        response = requests.post(
            CLICKUP_TOKEN_URL,
            json=token_data,
            headers={"Content-Type": "application/json"},
            timeout=20
        )
        
        if response.status_code != 200:
            error_text = response.text[:500]
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to exchange code for token: {error_text}"
            )
        
        token_response = response.json()
        access_token = token_response.get("access_token")
        
        if not access_token:
            raise HTTPException(
                status_code=500,
                detail="Access token not found in ClickUp response"
            )
        
        # Encrypt and store access token in encrypted_api_key field
        encrypted_token = encrypt_api_key(access_token)
        clickup_config.encrypted_api_key = encrypted_token
        
        # Optionally store refresh token if provided
        if "refresh_token" in token_response:
            if not additional_config:
                additional_config = {}
            additional_config["refresh_token"] = encrypt_api_key(token_response["refresh_token"])
        
        # After storing access token, automatically fetch and store space_id
        try:
            print(f"🔍 Auto-detecting ClickUp space after OAuth...")
            
            # Create headers with access token
            auth_headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }
            
            # Get team first
            team_url = "https://api.clickup.com/api/v2/team"
            team_response = requests.get(team_url, headers=auth_headers, timeout=20)
            if team_response.status_code == 200:
                teams_data = team_response.json()
                teams = teams_data.get("teams", [])
                if teams:
                    team_id = teams[0].get("id", "")
                    team_name = teams[0].get("name", "Unknown")
                    print(f"✅ Found team: {team_name} (ID: {team_id})")
                    
                    # Get spaces for this team
                    spaces_url = f"https://api.clickup.com/api/v2/team/{team_id}/space"
                    spaces_response = requests.get(spaces_url, headers=auth_headers, timeout=20)
                    if spaces_response.status_code == 200:
                        spaces_data = spaces_response.json()
                        spaces = spaces_data.get("spaces", [])
                        if spaces:
                            space_id = spaces[0].get("id", "")
                            space_name = spaces[0].get("name", "Unknown")
                            
                            # Update additional_config with space_id (only if not already set)
                            if not additional_config:
                                additional_config = {}
                            if "space_id" not in additional_config or not additional_config.get("space_id"):
                                additional_config["space_id"] = space_id
                                print(f"✅ Auto-detected and stored ClickUp space: {space_name} (ID: {space_id})")
                            else:
                                print(f"ℹ️ Space ID already exists in config: {additional_config.get('space_id')}")
                    else:
                        print(f"⚠️ Failed to fetch spaces: HTTP {spaces_response.status_code}")
                else:
                    print(f"⚠️ No teams found for this access token")
            else:
                print(f"⚠️ Failed to fetch teams: HTTP {team_response.status_code}")
        except Exception as e:
            # Don't fail the OAuth flow if space detection fails
            print(f"⚠️ Failed to auto-detect space_id: {str(e)}. You can add it manually later.")
        
        # Update additional_config
        clickup_config.additional_config = json.dumps(additional_config)
        
        db.commit()
        db.refresh(clickup_config)
        
        return create_success_response(
            data={
                "message": "ClickUp OAuth authorization successful",
                "access_token_stored": True,
                "crm_config_id": str(clickup_config.id)
            },
            message="ClickUp access token stored successfully"
        )
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to exchange authorization code: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing OAuth callback: {str(e)}"
        )


@router.get("/token", include_in_schema=False)
async def get_clickup_token(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Get decrypted ClickUp OAuth token for n8n workflows.
    
    Authentication: X-N8N-Webhook-Secret header required
    
    Returns:
        {
            "success": true,
            "data": {
                "access_token": "decrypted_clickup_token"
            },
            "message": "ClickUp token retrieved successfully"
        }
    """
    # Verify webhook secret
    is_webhook = await verify_n8n_webhook_secret_async(request)
    
    if not is_webhook:
        raise HTTPException(
            status_code=401,
            detail="Authentication required: X-N8N-Webhook-Secret header"
        )
    
    # Get ClickUp config from DB
    crm_config_service = CRMConfigService()
    clickup_config = crm_config_service.get_crm_config_by_type(db, "clickup")
    
    if not clickup_config or not clickup_config.encrypted_api_key:
        raise HTTPException(
            status_code=404,
            detail="ClickUp OAuth token not found. Please complete OAuth authorization first."
        )
    
    # Decrypt token
    try:
        decrypted_token = decrypt_api_key(clickup_config.encrypted_api_key)
        
        if not decrypted_token:
            raise HTTPException(
                status_code=500,
                detail="Failed to decrypt ClickUp token"
            )
        
        return create_success_response(
            data={"access_token": decrypted_token},
            message="ClickUp token retrieved successfully"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to decrypt ClickUp token: {str(e)}"
        )

