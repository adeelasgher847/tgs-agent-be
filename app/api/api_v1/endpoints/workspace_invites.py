"""
Team invitation endpoints under /workspace prefix.

POST /api/v1/workspace/invite         — admin only, JWT required
GET  /api/v1/workspace/invitations    — admin only, JWT required
"""
from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from typing import Union
from app.core.request_auth import ApiKeyPrincipal, is_api_key_principal
from app.api.deps import get_db, require_admin_or_api_key
from app.models.invite import Invite
from app.models.role import Role
from app.models.tenant import Tenant
from app.models.user import User, user_tenant_association
from app.schemas.base import SuccessResponse
from app.schemas.workspace_invite import InviteCreate, InviteOut
from app.services.email_service import email_service
from app.utils.response import create_success_response

router = APIRouter()


@router.post("/invite", response_model=SuccessResponse[InviteOut], status_code=201)
def invite_team_member(
    body: InviteCreate,
    admin: Union[User, ApiKeyPrincipal] = Depends(require_admin_or_api_key),
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

    # Validate role_id if provided
    role_id = None
    if body.role_id:
        role = db.query(Role).filter(Role.id == body.role_id).first()
        if not role:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Role with ID {body.role_id} not found"
            )
        if role.name in ("owner", "member"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot invite users with retired role '{role.name}'"
            )
        role_id = role.id

    # If caller is API Key principal, they do not have a User ID.
    # Map invited_by to the workspace creator's user ID to satisfy FK constraint.
    if is_api_key_principal(admin):
        invited_by_id = db.query(user_tenant_association.c.user_id).filter(
            user_tenant_association.c.tenant_id == tenant_id,
            user_tenant_association.c.is_creator == True
        ).scalar()
        if not invited_by_id:
            first_user = db.query(user_tenant_association.c.user_id).filter(
                user_tenant_association.c.tenant_id == tenant_id
            ).first()
            if first_user:
                invited_by_id = first_user[0]
        if not invited_by_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No admin or workspace member found to attribute this invite to"
            )
        inviter_name = tenant.name
    else:
        invited_by_id = admin.id
        inviter_name = f"{admin.first_name} {admin.last_name}".strip() or admin.email

    token = secrets.token_urlsafe(32)
    invite = Invite(
        email=body.email,
        tenant_id=tenant_id,
        invited_by=invited_by_id,
        role_id=role_id,
        token=token,
        expires_at=datetime.now(timezone.utc) + timedelta(days=7),
        status="pending",
    )

    db.add(invite)
    db.commit()
    db.refresh(invite)

    if not email_service.send_invite_email(
        email=body.email,
        invite_token=token,
        inviter_name=inviter_name,
        tenant_name=tenant.name,
    ):
        db.delete(invite)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send invitation email",
        )

    return create_success_response(
        InviteOut.model_validate(invite),
        "Invitation sent successfully",
        status.HTTP_201_CREATED,
    )


@router.get("/invitations", response_model=SuccessResponse[list[InviteOut]])
def list_invitations(
    admin: Union[User, ApiKeyPrincipal] = Depends(require_admin_or_api_key),
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

