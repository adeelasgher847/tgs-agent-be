from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.schemas.user import UserCreate, UserOut
from app.schemas.auth import LoginRequest, TokenResponse, RoleInfo, ForgotPasswordRequest, ForgotPasswordResponse, ResetPasswordRequest, ResetPasswordResponse
from app.schemas.auth import RefreshRequest
from app.schemas.base import SuccessResponse
from app.models.user import User
from app.models.password_reset import PasswordResetToken
from app.models.role import Role
from app.models.tenant import Tenant
from app.models.refresh_token import RefreshToken
from app.api.deps import get_db, get_current_user_jwt, security
from app.core.security import verify_password, create_user_token, pwd_context, create_password_reset_token, get_password_hash
from app.core.security import create_refresh_token_value, refresh_token_expires_at
from app.core.security import is_token_expired, verify_token
from app.services.email_service import email_service
from app.utils.response import create_success_response
from datetime import datetime, timezone
import uuid

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
    
    role_name = "admin"  
    
    # Validate role_id
    role = db.query(Role).filter(Role.name == role_name).first()
    if not role:
        raise HTTPException(
            status_code=400, 
            detail={
                "field": "role",
                "message": "Default role not found. Please contact administrator.",
                "error_type": "role_not_found"
            }
        )
    
    hashed_password = pwd_context.hash(user_in.password)
    db_user = User(
        email=user_in.email,
        role_id=role.id,
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
    return create_success_response(db_user, "User registered successfully", status.HTTP_201_CREATED)


@router.post("/login", response_model=SuccessResponse[TokenResponse])
def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """
    User login endpoint that returns JWT token with role information as object.
    Uses the user's current_tenant_id if set, otherwise uses the first available tenant.
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
    
    # Get role information as object
    role_info = None
    if user.role_id:
        role = db.query(Role).filter(Role.id == user.role_id).first()
        if role:
            role_info = RoleInfo(
                id=role.id,
                name=role.name,
                description=role.description
            )
    
    access_token = create_user_token(
        user_id=user.id,
        email=user.email,
        tenant_id=current_tenant_id
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
    if user.role_id:
        role = db.query(Role).filter(Role.id == user.role_id).first()
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
                "error_type": "email_not_found"
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