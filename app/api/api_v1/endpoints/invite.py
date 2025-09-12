from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.api.deps import get_db, require_tenant
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
    current_user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    tenant_id = current_user.current_tenant_id
    
    if not is_admin_in_tenant(db, current_user.id, tenant_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can invite team members"
        )
    
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
    
    inviter_name = f"{current_user.first_name} {current_user.last_name}"
    
    invite_token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(days=7)
    
    invite = Invite(
        email=email,
        tenant_id=tenant_id,
        invited_by=current_user.id,
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
