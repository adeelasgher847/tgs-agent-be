from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.api.deps import get_db
from app.models.invite import Invite
from app.models.user import User, user_tenant_association
from app.models.role import Role
from app.schemas.user import UserCreate, UserOut
from app.schemas.base import SuccessResponse
from app.core.security import get_password_hash, create_user_token, create_refresh_token_value, refresh_token_expires_at
from app.models.refresh_token import RefreshToken
from app.utils.response import create_success_response
from datetime import datetime, timezone
import uuid

router = APIRouter()

@router.post("/accept-invite", response_model=SuccessResponse[UserOut])
def accept_invite(
    token: str,
    password: str,
    db: Session = Depends(get_db)
):
    """
    Accept an invitation to join a tenant.
    
    Args:
        token: The invitation token
        password: User's chosen password (only required for new users)
    
    Returns:
        User details and access token
    """
    # Find the invitation by token
    invite = db.query(Invite).filter(
        Invite.token == token,
        Invite.status == "PENDING"
    ).first()
    
    if not invite:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid or expired invitation token"
        )
    
    # Check if invitation is expired
    if invite.expires_at < datetime.now(timezone.utc):
        # Mark invitation as expired
        invite.status = "EXPIRED"
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invitation has expired"
        )
    
    # Check if user already exists with this email
    existing_user = db.query(User).filter(User.email == invite.email).first()
    
    if existing_user:
        # User already exists, just add them to the new tenant
        user = existing_user
        
        # Check if user is already in this tenant
        from sqlalchemy import text
        result = db.execute(text("""
            SELECT COUNT(*) FROM user_tenant_association 
            WHERE user_id = :user_id AND tenant_id = :tenant_id
        """), {"user_id": str(user.id), "tenant_id": str(invite.tenant_id)})
        
        if result.scalar() > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User is already a member of this tenant."
            )
    else:
        # Create new user - use email prefix as name for now
        hashed_password = get_password_hash(password)
        email_prefix = invite.email.split('@')[0]
        user = User(
            first_name=email_prefix,
            last_name="User",
            email=invite.email,
            hashed_password=hashed_password,
            current_tenant_id=invite.tenant_id
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
    
    # Add user to tenant with member role (default for invited users)
    member_role = db.query(Role).filter(Role.name == "member").first()
    if not member_role:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Member role not found. Please contact administrator."
        )
    
    # Insert into user_tenant_association table
    db.execute(
        user_tenant_association.insert().values(
            user_id=user.id,
            tenant_id=invite.tenant_id,
            role_id=member_role.id
        )
    )
    
    # Update invitation status
    invite.status = "ACCEPTED"
    invite.accepted_at = datetime.now(timezone.utc)
    
    db.commit()
    
    # Create access token
    access_token = create_user_token(
        user_id=user.id,
        email=user.email,
        tenant_id=invite.tenant_id,
        role="member"
    )
    
    # Create refresh token
    rt_value = create_refresh_token_value()
    rt = RefreshToken(
        user_id=user.id,
        token=rt_value,
        expires_at=refresh_token_expires_at(),
        revoked=False
    )
    db.add(rt)
    db.commit()
    
    # Return user details
    user_out = UserOut(
        id=user.id,
        first_name=user.first_name,
        last_name=user.last_name,
        email=user.email,
        phone=user.phone,
        join_date=user.join_date,
        created_at=user.created_at,
        current_tenant_id=user.current_tenant_id
    )
    
    # Determine response message based on whether user was new or existing
    if existing_user:
        message = "Invitation accepted successfully. You have been added to the tenant."
    else:
        message = "Invitation accepted successfully. You are now a member of the tenant."
    
    return create_success_response(
        user_out, 
        message,
        status.HTTP_201_CREATED
    )
