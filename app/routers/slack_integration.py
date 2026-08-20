"""
Slack post-call-summary OAuth integration endpoints.

GET    /connect   — redirect to Slack's OAuth v2 consent page (admin-authenticated)
GET    /callback  — public; Slack redirects the browser here with no auth headers,
                     so the connecting tenant is recovered from the signed `state` param
DELETE ""         — revoke at Slack and delete the local connection
GET    ""         — connection status (team name, default channel)
GET    /channels  — list the tenant's available Slack channels (admin action, raises on error)
PUT    /channel   — set (or clear) the workspace's default Slack channel for post-call summaries
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_tenant, require_admin
from app.core.config import settings
from app.core.logger import logger
from app.schemas.base import SuccessResponse
from app.schemas.slack_integration import (
    SlackChannelOut,
    SlackDisconnectResponse,
    SlackIntegrationStatusOut,
    SlackSetChannelRequest,
)
from app.services import slack_service
from app.utils.response import create_success_response

router = APIRouter()


def _tenant_id(principal) -> uuid.UUID:
    return principal.current_tenant_id


@router.get("/connect", include_in_schema=False)
async def slack_connect(
    request: Request,
    principal=Depends(require_admin),
):
    """Redirect to Slack's OAuth v2 consent page. Scopes: channels:read, groups:read, chat:write.

    Frontend `fetch()` calls send `Accept: application/json` and can't follow
    a cross-origin 302 to Slack with our auth headers attached, so they get
    the URL back as JSON instead and navigate to it themselves. Normal browser
    navigation (no `Accept: application/json`) keeps the existing 302.
    """
    tenant_id = _tenant_id(principal)
    auth_url = slack_service.get_authorization_url(tenant_id)

    if request.headers.get("accept") == "application/json":
        return JSONResponse(content={"authorization_url": auth_url})

    return RedirectResponse(url=auth_url, status_code=status.HTTP_302_FOUND)


@router.get("/callback", include_in_schema=False)
async def slack_callback(
    code: str = Query(...),
    state: str = Query(...),
    db: Session = Depends(get_db),
):
    """
    Slack OAuth callback. No auth dependency — this is a top-level browser
    redirect from Slack, which cannot carry our JWT/API-key headers. The
    connecting tenant is recovered from the signed `state` param instead.
    """
    try:
        tenant_id = slack_service.verify_oauth_state(state)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    try:
        token_response = await slack_service.exchange_code_for_tokens(code)
    except Exception as exc:
        logger.warning(
            "Slack OAuth code exchange failed for tenant=%s: %s", tenant_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to exchange authorization code with Slack",
        )

    slack_service.upsert_integration(db, tenant_id, token_response)

    base = (settings.FRONTEND_URL or "").rstrip("/")
    redirect_to = (
        f"{base}/settings/integrations?slack=connected"
        if base
        else "/settings/integrations"
    )
    return RedirectResponse(url=redirect_to, status_code=status.HTTP_302_FOUND)


@router.delete("", response_model=SuccessResponse[SlackDisconnectResponse])
async def slack_disconnect(
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Revoke the bot token at Slack and delete the workspaceintegration row."""
    tenant_id = _tenant_id(principal)
    disconnected = await slack_service.disconnect(db, tenant_id)
    if not disconnected:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Slack is not connected for this workspace",
        )
    return create_success_response(
        SlackDisconnectResponse(disconnected=True),
        "Slack disconnected successfully",
    )


@router.get("", response_model=SuccessResponse[SlackIntegrationStatusOut])
async def slack_get_integration_status(
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Connection status and default channel for this workspace."""
    tenant_id = _tenant_id(principal)
    integration_status = slack_service.get_integration_status(db, tenant_id)
    return create_success_response(SlackIntegrationStatusOut(**integration_status))


@router.get("/channels", response_model=SuccessResponse[list[SlackChannelOut]])
async def slack_list_channels(
    principal=Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """List the tenant's available Slack channels (public + private the bot can see)."""
    tenant_id = _tenant_id(principal)
    try:
        channels = await slack_service.list_channels(db, tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.warning("Slack channel list failed for tenant=%s: %s", tenant_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to fetch Slack channels",
        )
    return create_success_response([SlackChannelOut(**c) for c in channels])


@router.put("/channel", response_model=SuccessResponse[SlackIntegrationStatusOut])
async def slack_set_channel(
    payload: SlackSetChannelRequest,
    principal=Depends(require_admin),
    db: Session = Depends(get_db),
):
    """Set (or clear, by omitting both fields) the workspace's default Slack channel."""
    tenant_id = _tenant_id(principal)
    if not slack_service.tenant_has_slack_connected(db, tenant_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Slack is not connected for this workspace",
        )

    slack_service.set_default_channel(
        db, tenant_id, payload.channel_id, payload.channel_name
    )
    integration_status = slack_service.get_integration_status(db, tenant_id)
    return create_success_response(
        SlackIntegrationStatusOut(**integration_status),
        "Default Slack channel updated successfully",
    )
