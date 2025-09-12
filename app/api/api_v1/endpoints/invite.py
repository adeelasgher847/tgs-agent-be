from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.api.deps import get_db, require_tenant, require_admin
from app.services.role_service import is_admin_in_tenant
from app.services.email_service import email_service
from app.models.tenant import Tenant
from app.models.user import User
from app.models.invite import Invite
from datetime import datetime, timedelta
import uuid
import secrets

router = APIRouter()

@router.post("/invite")
def invite_team_member(
    email: str,
    tenant_user: User = Depends(require_tenant),  # First middleware: tenant validation
    admin_user: User = Depends(require_admin),    # Second middleware: admin validation
    db: Session = Depends(get_db)
):
    # Both tenant_user and admin_user are validated by their respective middleware
    # We can use either one since they both represent the same user
    tenant_id = admin_user.current_tenant_id
    
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    existing_invite = db.query(Invite).filter(
        Invite.email == email,
        Invite.tenant_id == tenant_id,
        Invite.status == "PENDING"
    ).first()
    
    if existing_invite:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User already has a pending invitation for this tenant"
        )
    
    inviter_name = f"{admin_user.first_name} {admin_user.last_name}"
    
    invite_token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=7)
    
    invite = Invite(
        email=email,
        tenant_id=tenant_id,
        invited_by=admin_user.id,
        token=invite_token,
        expires_at=expires_at
    )
    
    db.add(invite)
    db.commit()
    
    success = email_service.send_invite_email(
        email=email,
        invite_token=invite_token,
        inviter_name=inviter_name,
        tenant_name=tenant.name
    )
    
    if not success:
        db.delete(invite)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send invitation email"
        )
    
    return {"message": "Invitation sent successfully via Gmail", "invite_id": str(invite.id)}
