"""Public (unauthenticated) Web SDK endpoints.

POST /api/v1/sdk/public-call-token and POST /api/v1/sdk/demo/{token}/call-token
are the only routes in the app that take no API key and no JWT — see
_SKIP_PREFIXES in app/middleware/api_key_middleware.py.

public-call-token security is enforced inside the handler instead of via
credentials:
  1. The target call flow must have public_access=True.
  2. The request's Origin header must match an allowed_domains entry for
     the workspace that owns the flow (or be localhost in development).
IP-based rate limiting (20/min) is enforced by RateLimitMiddleware before
the request reaches this handler — see _PUBLIC_TOKEN_POST_PATHS there.

demo/{token}/call-token is deliberately origin-unrestricted (no AllowedDomain
check) — access is controlled purely by possession of the opaque demo-link
token, its expiry/active flag, and its usage budgets. See
app/models/call_flow_demo_link.py for the full security rationale. It shares
the same CORS reflection, auth-skip, and rate-limit treatment as
public-call-token (see PublicSdkCorsMiddleware, _SKIP_PREFIXES, and
_PUBLIC_TOKEN_POST_PATHS respectively) since it is equally unauthenticated.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.config import settings
from app.core.error_responses import build_api_error_payload
from app.core.logger import logger
from app.core.origin import is_localhost_origin, normalize_origin
from app.models.allowed_domain import AllowedDomain
from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.call_flow_demo_link import CallFlowDemoLink
from app.models.call_flow_demo_link_visitor_usage import CallFlowDemoLinkVisitorUsage
from app.models.user import user_tenant_association
from app.services.call_session_service import call_session_service

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


def _origin_allowed(db: Session, origin: str | None, workspace_id: uuid.UUID) -> bool:
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


# ---------------------------------------------------------------------------
# Demo link — shareable, origin-unrestricted call token
# ---------------------------------------------------------------------------


class DemoCallTokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Opaque client-generated identifier the caller is expected to persist
    # (e.g. localStorage) and resend on every call so per-visitor minute
    # limits can be enforced across sessions. Not a FK to `user` — there is
    # no account behind an anonymous demo-link visitor.
    visitor_id: str = Field(min_length=1, max_length=255)


def _demo_error(request: Request, code: int, message: str, error_code: str) -> JSONResponse:
    return _error(request, code, message, error_code)


@router.post("/demo/{token}/call-token")
async def demo_call_token(
    token: str,
    body: DemoCallTokenRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    ip = _client_ip(request)

    demo_link = db.execute(
        select(CallFlowDemoLink).where(CallFlowDemoLink.token == token)
    ).scalar_one_or_none()

    if demo_link is None:
        logger.info("Demo call token request: link not found ip=%s", ip)
        return _demo_error(request, 404, "Demo link not found", "demo_link_not_found")

    if not demo_link.is_active:
        logger.info("Demo call token denied (inactive) demo_link_id=%s ip=%s", demo_link.id, ip)
        return _demo_error(request, 403, "This demo link is no longer active.", "demo_link_inactive")

    if demo_link.expires_at is not None:
        expires_at = demo_link.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at < datetime.now(timezone.utc):
            logger.info("Demo call token denied (expired) demo_link_id=%s ip=%s", demo_link.id, ip)
            return _demo_error(request, 410, "This demo link has expired.", "demo_link_expired")

    flow = db.execute(
        select(CallFlow).where(
            CallFlow.id == demo_link.call_flow_id,
            CallFlow.is_deleted == False,  # noqa: E712
            CallFlow.status == "active",
        )
    ).scalar_one_or_none()

    if flow is None:
        logger.info(
            "Demo call token denied (call flow unavailable) demo_link_id=%s ip=%s",
            demo_link.id, ip,
        )
        return _demo_error(
            request, 403, "This demo link's call flow is no longer available.", "call_flow_unavailable"
        )

    agent = db.execute(
        select(Agent).where(
            Agent.id == flow.agent_id,
            Agent.is_deleted == False,  # noqa: E712
        )
    ).scalar_one_or_none()

    if agent is None:
        logger.info(
            "Demo call token denied (agent unavailable) demo_link_id=%s agent_id=%s ip=%s",
            demo_link.id, flow.agent_id, ip,
        )
        return _demo_error(
            request, 403, "This demo link's agent is no longer available.", "agent_unavailable"
        )

    if demo_link.total_budget_minutes is not None and demo_link.total_minutes_used >= demo_link.total_budget_minutes:
        logger.info("Demo call token denied (budget exhausted) demo_link_id=%s ip=%s", demo_link.id, ip)
        return _demo_error(
            request, 403, "This demo link has used up its total call budget.", "demo_link_budget_exceeded"
        )

    visitor_id = body.visitor_id.strip()
    if not visitor_id:
        return _demo_error(request, 400, "visitor_id is required", "visitor_id_required")

    visitor_usage = db.execute(
        select(CallFlowDemoLinkVisitorUsage).where(
            CallFlowDemoLinkVisitorUsage.demo_link_id == demo_link.id,
            CallFlowDemoLinkVisitorUsage.visitor_id == visitor_id,
        )
    ).scalar_one_or_none()

    if visitor_usage is None:
        visitor_usage = CallFlowDemoLinkVisitorUsage(
            demo_link_id=demo_link.id,
            visitor_id=visitor_id,
            minutes_used=Decimal("0"),
        )
        db.add(visitor_usage)
        try:
            db.commit()
        except IntegrityError:
            # Two concurrent first-time requests for the same visitor_id —
            # the other request's insert won the unique constraint
            # (demo_link_id, visitor_id); fetch the row it created instead.
            db.rollback()
            visitor_usage = db.execute(
                select(CallFlowDemoLinkVisitorUsage).where(
                    CallFlowDemoLinkVisitorUsage.demo_link_id == demo_link.id,
                    CallFlowDemoLinkVisitorUsage.visitor_id == visitor_id,
                )
            ).scalar_one()
        else:
            db.refresh(visitor_usage)

    if (
        demo_link.per_user_limit_minutes is not None
        and visitor_usage.minutes_used >= demo_link.per_user_limit_minutes
    ):
        logger.info(
            "Demo call token denied (visitor limit) demo_link_id=%s visitor_id=%s ip=%s",
            demo_link.id, visitor_id, ip,
        )
        return _demo_error(
            request,
            403,
            "You have used up your allotted call time for this demo link.",
            "demo_link_visitor_limit_exceeded",
        )

    from app.services.livekit_service import livekit_service

    # CallSession.user_id is a NOT NULL FK to `user.id` — there is no real
    # user behind an anonymous demo-link visitor, so follow the same
    # convention already used for inbound Twilio calls (voice.py:
    # user_id=inbound_agent.created_by): attribute the session to whoever
    # created the underlying agent (fetched above). Fall back to the demo
    # link's own creator if the agent has no creator on record.
    owner_user_id = agent.created_by or demo_link.created_by_user_id
    if owner_user_id is None:
        # Last resort: every tenant/workspace has exactly one creator
        # (RBAC's is_creator flag), so this should always resolve even
        # when the agent/link have no recorded creator (e.g. API-key
        # originated).
        owner_user_id = db.execute(
            select(user_tenant_association.c.user_id).where(
                user_tenant_association.c.tenant_id == demo_link.tenant_id,
                user_tenant_association.c.is_creator == True,  # noqa: E712
            )
        ).scalar_one_or_none()
    if owner_user_id is None:
        logger.warning(
            "Demo call token denied (no owning user) demo_link_id=%s agent_id=%s ip=%s",
            demo_link.id, flow.agent_id, ip,
        )
        return _demo_error(
            request, 500, "This demo link is not configured correctly.", "demo_link_misconfigured"
        )

    # Room naming follows the platform-wide convention documented in
    # livekit_service.py ("room_{call_session.id}") rather than
    # public-call-token's unlinked room_{uuid4()} — this lets the demo-link
    # session show up in v2 active-calls/insights and, more importantly,
    # gives the end-of-call hook in CallSessionService.update_call_session_status
    # a CallSession row to finalize duration/minute-accounting against.
    call_session = call_session_service.create_call_session(
        db=db,
        user_id=owner_user_id,
        agent_id=flow.agent_id,
        tenant_id=demo_link.tenant_id,
        call_type="web",
        assistant_phone_number="web_agent",
        customer_phone_number="browser_demo_link",
    )
    call_session.call_flow_id = flow.id
    call_session.call_metadata = {
        **(call_session.call_metadata or {}),
        "demo_link": {
            "demo_link_id": str(demo_link.id),
            "visitor_id": visitor_id,
        },
    }
    db.commit()

    room_name = f"room_{call_session.id}"
    lk_token = livekit_service.generate_caller_token(room_name)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=settings.LIVEKIT_TOKEN_TTL)

    logger.info(
        "Demo call token issued demo_link_id=%s call_session_id=%s room=%s ip=%s",
        demo_link.id, call_session.id, room_name, ip,
    )

    # Spawn the agent's join/conversation loop as a detached background task —
    # never awaited here, so the HTTP response returns immediately regardless
    # of how long the call takes. All failures (agent-join, STT/TTS wiring,
    # LiveKit connectivity) are caught and logged inside
    # run_livekit_browser_call itself so this never becomes an unhandled task
    # exception. spawn_browser_call keeps a strong reference to the task for
    # its whole lifetime — asyncio.Task is only weakly held by the event
    # loop, and this task can run for the entire duration of a live call.
    # See app/voice/livekit_browser_call_handler.py.
    from app.voice.livekit_browser_call_handler import spawn_browser_call

    spawn_browser_call(call_session.id)

    return {
        "livekit_token": lk_token,
        "room_name": room_name,
        "flow_id": str(flow.id),
        "agent_id": str(flow.agent_id),
        "call_session_id": str(call_session.id),
        "demo_link_name": demo_link.name,
        "expires_at": expires_at.isoformat(),
    }
