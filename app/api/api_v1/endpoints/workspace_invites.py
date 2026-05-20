"""
Team invitation endpoints under /workspace prefix.

POST /api/v1/workspace/invite         — admin only, JWT required
GET  /api/v1/workspace/invitations    — admin only, JWT required
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin
from app.core.logger import logger
from app.models.invite import Invite
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.base import SuccessResponse
from app.schemas.workspace_invite import InviteCreate, InviteOut
from app.services.email_service import build_invite_email_content
from app.utils.response import create_success_response

router = APIRouter()


@router.post("/invite", response_model=SuccessResponse[InviteOut], status_code=201)
def invite_team_member(
    body: InviteCreate,
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> SuccessResponse[InviteOut]:
    tenant_id: uuid.UUID = admin.current_tenant_id

    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

    existing = (
        db.query(Invite)
        .filter(
            Invite.email == body.email,
            Invite.tenant_id == tenant_id,
            Invite.status == "pending",
        )
        .first()
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A pending invitation already exists for this email in this workspace",
        )

    token = str(uuid.uuid4())
    invite = Invite(
        email=body.email,
        tenant_id=tenant_id,
        invited_by=admin.id,
        token=token,
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        status="pending",
    )
    db.add(invite)
    db.commit()
    db.refresh(invite)

    inviter_name = f"{admin.first_name} {admin.last_name}".strip() or admin.email
    subject, html_body = build_invite_email_content(token, inviter_name, tenant.name)
    logger.info(
        "INVITE EMAIL (not sent)\nTo: %s\nSubject: %s\nBody:\n%s",
        body.email,
        subject,
        html_body,
    )

    return create_success_response(
        InviteOut.model_validate(invite),
        "Invitation sent successfully",
        status.HTTP_201_CREATED,
    )


@router.get("/invitations", response_model=SuccessResponse[list[InviteOut]])
def list_invitations(
    admin: User = Depends(require_admin),
    db: Session = Depends(get_db),
) -> SuccessResponse[list[InviteOut]]:
    invites = (
        db.query(Invite)
        .filter(
            Invite.tenant_id == admin.current_tenant_id,
            Invite.status == "pending",
        )
        .all()
    )
    out = [InviteOut.model_validate(i) for i in invites]
    return create_success_response(out, "Pending invitations retrieved")
