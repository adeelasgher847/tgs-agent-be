"""Public (unauthenticated) Web SDK endpoints.

POST /api/v1/sdk/public-call-token is the only route in the app that takes
no API key and no JWT — see _SKIP_PREFIXES in app/middleware/api_key_middleware.py.
Security is enforced inside the handler instead of via credentials:
  1. The target call flow must have public_access=True.
  2. The request's Origin header must match an allowed_domains entry for
     the workspace that owns the flow (or be localhost in development).
IP-based rate limiting (20/min) is enforced by RateLimitMiddleware before
the request reaches this handler — see _PUBLIC_TOKEN_POST_PATHS there.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.config import settings
from app.core.error_responses import build_api_error_payload
from app.core.logger import logger
from app.core.origin import is_localhost_origin, normalize_origin
from app.models.allowed_domain import AllowedDomain
from app.models.call_flow import CallFlow

router = APIRouter()


class PublicCallTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flow_id: uuid.UUID
    agent_id: uuid.UUID


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def _client_ip(request: Request) -> str:
    """Client IP, preferring X-Forwarded-For (load-balanced deployments)."""
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    client = request.client
    return client.host if client else "unknown"


def _error(request: Request, code: int, message: str, error_code: str) -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content=build_api_error_payload(
            code, message, error_code=error_code, request_id=_request_id(request)
        ),
    )


def _origin_allowed(db: Session, origin: Optional[str], workspace_id: uuid.UUID) -> bool:
    if not origin:
        return False
    if settings.ENVIRONMENT.lower() == "development" and is_localhost_origin(origin):
        return True
    normalized = normalize_origin(origin)
    existing = db.execute(
        select(AllowedDomain.id).where(
            AllowedDomain.workspace_id == workspace_id,
            AllowedDomain.domain == normalized,
        )
    ).first()
    return existing is not None


@router.post("/public-call-token")
def public_call_token(
    body: PublicCallTokenRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    origin = request.headers.get("origin")
    ip = _client_ip(request)

    flow = db.execute(
        select(CallFlow).where(
            CallFlow.id == body.flow_id,
            CallFlow.agent_id == body.agent_id,
            CallFlow.is_deleted == False,  # noqa: E712
        )
    ).scalar_one_or_none()

    if flow is None:
        logger.info(
            "Public token request: flow not found flow_id=%s agent_id=%s ip=%s origin=%s",
            body.flow_id, body.agent_id, ip, origin,
        )
        return _error(request, 404, "Call flow not found", "flow_not_found")

    if not flow.public_access:
        logger.info(
            "Public token request denied (public_access disabled) flow_id=%s ip=%s origin=%s",
            flow.id, ip, origin,
        )
        return _error(
            request,
            403,
            "Enable public access on this flow first.",
            "public_access_disabled",
        )

    if not _origin_allowed(db, origin, flow.tenant_id):
        logger.info(
            "Public token request denied (origin not allowed) flow_id=%s ip=%s origin=%s",
            flow.id, ip, origin,
        )
        return _error(
            request,
            403,
            "This domain is not whitelisted for the workspace.",
            "domain_not_allowed",
        )

    from app.services.livekit_service import livekit_service

    room_name = f"room_{uuid.uuid4()}"
    token = livekit_service.generate_caller_token(room_name)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=settings.LIVEKIT_TOKEN_TTL)

    logger.info(
        "Public token issued flow_id=%s ip=%s origin=%s room=%s",
        flow.id, ip, origin, room_name,
    )

    return {
        "livekit_token": token,
        "room_name": room_name,
        "flow_id": str(flow.id),
        "expires_at": expires_at.isoformat(),
    }
