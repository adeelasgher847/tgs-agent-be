from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.schemas.user import UserCreate, UserOut
from app.schemas.auth import LoginRequest, TokenResponse, RoleInfo
from app.schemas.base import SuccessResponse
from app.models.user import User
from app.models.role import Role
from app.models.tenant import Tenant
from app.api.deps import get_db, get_current_user_jwt, security
from app.core.security import verify_password, create_user_token, pwd_context
from app.utils.response import create_success_response
from datetime import datetime, timezone
import uuid

router = APIRouter()


@router.post("/register", response_model=SuccessResponse[UserOut])
def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    # Check if email already exists
    user = db.query(User).filter(User.email == user_in.email).first()
    if user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    role_name = "admin"  
    
    # Validate role_id
    role = db.query(Role).filter(Role.name == role_name).first()
    if not role:
        raise HTTPException(
            status_code=400, 
            detail="Default role not found. Please contact administrator."
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
            detail="Incorrect email or password"
        )
    
    # Verify password
    if not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
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
    
    token_response = TokenResponse(
        access_token=access_token,
        user_id=user.id,
        email=user.email,
        tenant_id=current_tenant_id,
        tenant_ids=tenant_ids,
        role=role_info
    )
    
    return create_success_response(token_response, "Login successful")


@router.post("/logout", response_model=SuccessResponse[dict])
def logout():
    """
    Logout endpoint (client should discard token).
    Note: JWT tokens are stateless, so server-side logout requires token blacklisting.
    """
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
        "message": "Token expires in 30 minutes from creation"
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