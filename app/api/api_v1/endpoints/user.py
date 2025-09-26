from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy import update
from app.schemas.user import UserCreate, UserOut, UserProfile, UserUpdate, TenantMember, CreditInfo
from app.schemas.auth import LoginRequest, TokenResponse, RoleInfo, ForgotPasswordRequest, ForgotPasswordResponse, ResetPasswordRequest, ResetPasswordResponse
from app.schemas.auth import RefreshRequest
from app.schemas.base import SuccessResponse
from app.models.user import User, user_tenant_association
from app.models.password_reset import PasswordResetToken
from app.models.role import Role
from app.models.tenant import Tenant
from app.models.refresh_token import RefreshToken
from app.api.deps import get_db, get_current_user_jwt, require_member_or_admin, security
from app.core.security import verify_password, create_user_token, pwd_context, create_password_reset_token, get_password_hash
from app.core.security import create_refresh_token_value, refresh_token_expires_at
from app.core.security import is_token_expired, verify_token
from app.services.email_service import email_service
from app.services.credit_service import credit_service
from app.utils.response import create_success_response
from app.utils.rate_limiter import login_rate_limit
from datetime import datetime, timezone
from app.services.role_service import get_user_role_in_tenant
import uuid
import re

router = APIRouter()


@router.post("/register", response_model=SuccessResponse[UserOut])
def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    # Check if email already exists
    user = db.query(User).filter(User.email == user_in.email).first()
    if user:
        raise HTTPException(
            status_code=400, 
            detail={
                "field": "email",
                "message": "Email already registered",
                "error_type": "email_already_exists"
            }
        )    
    
    hashed_password = pwd_context.hash(user_in.password)
    db_user = User(
        email=user_in.email,
        first_name=user_in.first_name,
        last_name=user_in.last_name,
        phone=user_in.phone,
        hashed_password=hashed_password,
        join_date=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc)
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create tenant automatically with user's name as tenant name
    # Use first name + last name, but make it unique by adding email suffix if needed
    base_tenant_name = f"{user_in.first_name} {user_in.last_name}".strip()
    tenant_name = base_tenant_name
    
    # Check if tenant name already exists and make it unique
    counter = 1
    while db.query(Tenant).filter(Tenant.name == tenant_name).first():
        tenant_name = f"{base_tenant_name} ({user_in.email.split('@')[0]})"
        counter += 1
        if counter > 1:  # If still exists, add a number
            tenant_name = f"{base_tenant_name} ({user_in.email.split('@')[0]}) {counter}"
    
    # Generate schema name from tenant name
    schema_name = re.sub(r'[^a-zA-Z0-9]', '_', tenant_name.lower())
    schema_name = re.sub(r'_+', '_', schema_name).strip('_')
    schema_name = f"{schema_name}_schema"
    
    # Create new tenant
    db_tenant = Tenant(
        name=tenant_name,
        schema_name=schema_name,
        status="pending_payment"
    )
    
    db.add(db_tenant)
    db.commit()
    db.refresh(db_tenant)
    
    # Get owner role
    owner_role = db.query(Role).filter(Role.name == "owner").first()
    
    # Add user to tenant with owner role
    db_user.tenants.append(db_tenant)
    db.commit()
    
    # Update the role_id in the association table
    stmt = update(user_tenant_association).where(
        (user_tenant_association.c.user_id == db_user.id) &
        (user_tenant_association.c.tenant_id == db_tenant.id)
    ).values(role_id=owner_role.id)
    
    db.execute(stmt)
    
    # Set the new tenant as user's current tenant
    db_user.current_tenant_id = db_tenant.id
    
    db.commit()
    db.refresh(db_user)
    
    return create_success_response(db_user, "User registered successfully", status.HTTP_201_CREATED)


@router.post("/login", response_model=SuccessResponse[TokenResponse])
@login_rate_limit()
def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """
    User login endpoint that returns JWT token with role information as object.
    Uses the user's current_tenant_id if set, otherwise uses the first available tenant.
    Automatically assigns admin role if user has no role in current tenant.
    """
    # Find user by email
    user = db.query(User).filter(User.email == login_data.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "field": "email",
                "message": "Email not found in our system",
                "error_type": "email_not_found"
            }
        )
    
    # Verify password
    if not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "field": "password",
                "message": "Password is incorrect for this email",
                "error_type": "invalid_password"
            }
        )
    
    # Get user's tenant IDs
    tenant_ids = [tenant.id for tenant in user.tenants]
    
    # Determine which tenant to use
    current_tenant_id = None
    if user.current_tenant_id and user.current_tenant_id in tenant_ids:
        # Use the user's current tenant if it exists and user has access
        current_tenant_id = user.current_tenant_id
    elif tenant_ids:
        # If no current tenant set, use the first available tenant
        current_tenant_id = tenant_ids[0]
        # Update user's current_tenant_id
        user.current_tenant_id = current_tenant_id
        db.commit()
    
    # Get role information for the current tenant
    role_info = None
    current_role = None
    if current_tenant_id:
        from app.services.role_service import get_user_role_in_tenant, assign_role_to_user_tenant
        
        # Check if user has a role in this tenant
        role = get_user_role_in_tenant(db, user.id, current_tenant_id)
        
        if role:
            role_info = RoleInfo(
                id=role.id,
                name=role.name,
                description=role.description
            )
            current_role = role.name
    
    access_token = create_user_token(
        user_id=user.id,
        email=user.email,
        tenant_id=current_tenant_id,
        role=current_role
    )

    # Create refresh token (valid 7 days)
    rt_value = create_refresh_token_value()
    rt = RefreshToken(
        user_id=user.id,
        token=rt_value,
        expires_at=refresh_token_expires_at(),
        revoked=False
    )
    db.add(rt)
    db.commit()
    
    token_response = TokenResponse(
        access_token=access_token,
        user_id=user.id,
        email=user.email,
        tenant_id=current_tenant_id,
        tenant_ids=tenant_ids,
        role=role_info,
        refresh_token=rt_value
    )
    
    return create_success_response(token_response, "Login successful")


@router.post("/refresh")
def refresh_tokens(req: RefreshRequest, db: Session = Depends(get_db)):
    """
    Refresh endpoint:
    1) If access_token is provided and still valid -> return "still valid"
    2) If access_token expired but refresh_token valid -> issue new access_token only
       (reuse the last generated token if it exists)
    3) If refresh_token invalid/expired -> return 401
    """

    # 1) If access token is still valid
    if req.access_token:
        payload = verify_token(req.access_token)
        if payload and not is_token_expired(req.access_token):
            return {
                "status_code": 200,
                "message": "Access token still valid"
            }

    # 2) Validate refresh token
    rt = db.query(RefreshToken).filter(RefreshToken.token == req.refresh_token).first()
    if not rt or rt.revoked or rt.expires_at <= datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )

    user = db.query(User).filter(User.id == rt.user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    tenant_ids = [t.id for t in user.tenants]
    current_tenant_id = user.current_tenant_id if user.current_tenant_id in tenant_ids else (tenant_ids[0] if tenant_ids else None)

    role_info = None
    if current_tenant_id:
        role = get_user_role_in_tenant(db, user.id, current_tenant_id)
        if role:
            role_info = RoleInfo(id=role.id, name=role.name, description=role.description)

    # 3) Check if a new access token was already generated for this refresh token
    if rt.replaced_access_token and not is_token_expired(rt.replaced_access_token):
        new_access_token = rt.replaced_access_token
    else:
        new_access_token = create_user_token(
            user_id=user.id,
            email=user.email,
            tenant_id=current_tenant_id
        )
        rt.replaced_access_token = new_access_token
        db.add(rt)

    db.commit()

    return {
        "status_code": 200,
        "message": "Access token refreshed",
        "data": {
            "access_token": new_access_token,
            "token_type": "bearer",
            "user_id": user.id,
            "email": user.email,
            "tenant_id": current_tenant_id,
            "tenant_ids": tenant_ids,
            "role": role_info
        }
    }


@router.post("/logout", response_model=SuccessResponse[dict])
def logout(current_user: User = Depends(get_current_user_jwt), db: Session = Depends(get_db)):
    """
    Logout endpoint: revoke all active refresh tokens for the user.
    Note: Access JWTs are stateless and cannot be revoked server-side.
    """
    # Find and revoke all active refresh tokens for the user
    active_tokens = db.query(RefreshToken).filter(
        RefreshToken.user_id == current_user.id,
        RefreshToken.revoked == False,
        RefreshToken.expires_at > datetime.now(timezone.utc)
    ).all()
    
    for token in active_tokens:
        token.revoked = True
    
    db.commit()
    return create_success_response({"message": "Successfully logged out"}, "Logout successful")

@router.get("/token-info", response_model=SuccessResponse[dict])
def get_token_information(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get detailed information about the current token including expiration"""
    from app.core.security import get_token_info
    
    token = credentials.credentials
    token_info = get_token_info(token)
    
    if not token_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    
    return create_success_response({
        "token_info": token_info,
        "message": "Token expires in 15 minutes from creation"
    }, "Token information retrieved successfully")

@router.get("/check-token-expiration", response_model=SuccessResponse[dict])
def check_token_expiration(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Check if the current token is expired"""
    from app.core.security import is_token_expired
    
    token = credentials.credentials
    is_expired = is_token_expired(token)
    
    if is_expired:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    
    return create_success_response({
        "is_expired": False,
        "message": "Token is still valid"
    }, "Token validation successful")

@router.get("/my-tenants", response_model=SuccessResponse[dict])
def get_user_tenants(
    user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """
    Get all tenants associated with the current user for dropdown selection.
    Returns list of tenants with id and name, plus current tenant id.
    """
    # Get all tenants for the user
    user_tenants = user.tenants
    
    # Convert to simple format for dropdown
    tenant_list = [
        {
            "id": tenant.id,
            "name": tenant.name
        }
        for tenant in user_tenants
    ]
    
    return create_success_response({
        "tenants": tenant_list,
        "current_tenant_id": user.current_tenant_id
    }, "User tenants retrieved successfully")

@router.post("/forgot-password", response_model=SuccessResponse[ForgotPasswordResponse])
def forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """
    Send password reset email to user
    """
    # Find user by email
    user = db.query(User).filter(User.email == request.email).first()
    
    # Check if user exists
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "message": "Email address not found in our system.",
                "error_type": "email_not_found",
                "field":"email"
            }
        )
    
    # Invalidate any existing reset tokens for this user
    existing_tokens = db.query(PasswordResetToken).filter(
        PasswordResetToken.user_id == user.id,
        PasswordResetToken.used == False
    ).all()
    
    for token in existing_tokens:
        token.used = True
    
    # Create new reset token
    reset_token, expires_at = create_password_reset_token(user.id)
    
    # Save token to database
    db_reset_token = PasswordResetToken(
        user_id=user.id,
        token=reset_token,
        expires_at=expires_at,
        used=False
    )
    db.add(db_reset_token)
    db.commit()
    
    # Send email
    user_name = f"{user.first_name} {user.last_name}".strip()
    if not user_name:
        user_name = user.email
    
    email_sent = email_service.send_password_reset_email(
        email=user.email,
        reset_token=reset_token,
        user_name=user_name
    )
    
    if email_sent:
        return create_success_response(
            ForgotPasswordResponse(message="Password reset email sent successfully."),
            "Password reset email sent"
        )
    else:
        # If email fails, mark token as used
        db_reset_token.used = True
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": "Failed to send password reset email. Please try again later.",
                "error_type": "email_send_failed"
            }
        )

@router.post("/reset-password", response_model=SuccessResponse[ResetPasswordResponse])
def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    """
    Reset password using reset token and revoke all refresh tokens
    """
    # Find valid reset token
    reset_token = db.query(PasswordResetToken).filter(
        PasswordResetToken.token == request.token,
        PasswordResetToken.used == False,
        PasswordResetToken.expires_at > datetime.now(timezone.utc)
    ).first()
    
    if not reset_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Invalid or expired reset token.",
                "error_type": "invalid_reset_token"
            }
        )
    
    # Get user
    user = db.query(User).filter(User.id == reset_token.user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "User not found.",
                "error_type": "user_not_found"
            }
        )
    
    # Update password
    user.hashed_password = get_password_hash(request.new_password)
    
    # Mark token as used
    reset_token.used = True

    # Revoke all user's refresh tokens on password change
    active_rts = db.query(RefreshToken).filter(
        RefreshToken.user_id == user.id,
        RefreshToken.revoked == False,
        RefreshToken.expires_at > datetime.now(timezone.utc)
    ).all()
    
    for token in active_rts:
        token.revoked = True
    
    db.commit()
    
    return create_success_response(
        ResetPasswordResponse(message="Password reset successfully."),
        "Password reset successful"
    )


@router.get("/profile", response_model=SuccessResponse[UserProfile])
def get_user_profile(
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """
    Get complete user profile information including role and tenant details.
    Requires JWT Bearer token authentication.
    """
    # Fetch user with all related data
    user = db.query(User).filter(User.id == current_user.id).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get role information for the current tenant
    role_info = None
    if user.current_tenant_id:
        from app.services.role_service import get_user_role_in_tenant
        role = get_user_role_in_tenant(db, user.id, user.current_tenant_id)
        if role:
            role_info = RoleInfo(id=role.id, name=role.name, description=role.description)
    
    # Get credit information for the current tenant
    credit_info = None
    if user.current_tenant_id:
        try:
            credit_balance = credit_service.get_credit_balance(db, user.current_tenant_id)
            plan_pricing = credit_service.get_plan_pricing(db, user.current_tenant_id)
            
            credit_info = CreditInfo(
                credit_balance=credit_balance.credit_balance,
                plan_credits=credit_balance.plan_credits,
                plan_name=credit_balance.plan_name,
                plan_pricing={
                    "price_per_minute": plan_pricing["price_per_minute"],
                    "plan_id": plan_pricing["plan_id"],
                    "has_plan": plan_pricing["has_plan"]
                },
                can_make_call=credit_balance.credit_balance >= 1
            )
        except Exception as e:
            # If credit info fails, continue without it
            print(f"Warning: Could not fetch credit info for user {user.id}: {str(e)}")
    
    # Create user profile response
    user_profile = UserProfile(
        id=user.id,
        first_name=user.first_name,
        last_name=user.last_name,
        email=user.email,
        phone=user.phone,
        role_id=role_info.id if role_info else None,
        current_tenant_id=user.current_tenant_id,
        join_date=user.join_date,
        created_at=user.created_at,
        role=role_info,
        current_tenant=user.current_tenant,
        tenants=user.tenants,
        credit_info=credit_info
    )
    
    return create_success_response(
        user_profile,
        "User profile retrieved successfully"
    )


@router.put("/profile", response_model=SuccessResponse[UserProfile])
def update_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """
    Update user profile information.
    Requires JWT Bearer token authentication.
    Only updates fields that are provided in the request.
    """
    # Get the fields to update
    update_data = user_update.model_dump(exclude_unset=True)
    
    # Validate email if it's being updated
    if "email" in update_data:
        new_email = update_data["email"]
        if new_email != current_user.email:
            existing_user = db.query(User).filter(
                User.email == new_email,
                User.id != current_user.id
            ).first()
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already exists"
                )
    
    for field, value in update_data.items():
        setattr(current_user, field, value)
    
    try:
        db.commit()
        db.refresh(current_user)
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )
    
    # Get role information for the current tenant
    role_info = None
    if current_user.current_tenant_id:
        from app.services.role_service import get_user_role_in_tenant
        role = get_user_role_in_tenant(db, current_user.id, current_user.current_tenant_id)
        if role:
            role_info = RoleInfo(id=role.id, name=role.name, description=role.description)
    
    # Get credit information for the current tenant
    credit_info = None
    if current_user.current_tenant_id:
        try:
            credit_balance = credit_service.get_credit_balance(db, current_user.current_tenant_id)
            plan_pricing = credit_service.get_plan_pricing(db, current_user.current_tenant_id)
            
            credit_info = CreditInfo(
                credit_balance=credit_balance.credit_balance,
                plan_credits=credit_balance.plan_credits,
                plan_name=credit_balance.plan_name,
                plan_pricing={
                    "price_per_minute": plan_pricing["price_per_minute"],
                    "plan_id": plan_pricing["plan_id"],
                    "has_plan": plan_pricing["has_plan"]
                },
                can_make_call=credit_balance.credit_balance >= 1
            )
        except Exception as e:
            # If credit info fails, continue without it
            print(f"Warning: Could not fetch credit info for user {current_user.id}: {str(e)}")
    
    # Create updated user profile response
    user_profile = UserProfile(
        id=current_user.id,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        email=current_user.email,
        phone=current_user.phone,
        role_id=role_info.id if role_info else None,
        current_tenant_id=current_user.current_tenant_id,
        join_date=current_user.join_date,
        created_at=current_user.created_at,
        role=role_info,
        current_tenant=current_user.current_tenant,
        tenants=current_user.tenants,
        credit_info=credit_info
    )
    
    return create_success_response(
        user_profile,
        "User profile updated successfully"
    )


@router.get("/tenant-members", response_model=SuccessResponse[list[TenantMember]])
def get_tenant_members(
    current_user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db)
):
    """
    Get all members of the current tenant with their roles.
    Requires JWT Bearer token authentication and tenant membership.
    """
    if not current_user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tenant selected. Please set a current tenant."
        )
    
    # Get all users associated with the current tenant
    tenant_users = db.query(User).join(
        user_tenant_association
    ).filter(
        user_tenant_association.c.tenant_id == current_user.current_tenant_id
    ).all()
    
    # Build the response with role information for each user
    members = []
    for user in tenant_users:
        # Get role information for this user in the current tenant
        role = get_user_role_in_tenant(db, user.id, current_user.current_tenant_id)
        role_info = None
        if role:
            role_info = RoleInfo(
                id=role.id,
                name=role.name,
                description=role.description
            )
        
        member = TenantMember(
            id=user.id,
            first_name=user.first_name,
            last_name=user.last_name,
            email=user.email,
            role=role_info,
            join_date=user.join_date,
            created_at=user.created_at
        )
        members.append(member)
    
    return create_success_response(
        members,
        f"Retrieved {len(members)} tenant members successfully"
    )