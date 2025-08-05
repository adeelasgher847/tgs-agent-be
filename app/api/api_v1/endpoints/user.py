from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.schemas.user import UserCreate, UserOut
from app.schemas.auth import LoginRequest, TokenResponse
from app.models.user import User
from app.models.role import Role
from app.models.tenant import Tenant
from app.api.deps import get_db, get_current_user_jwt, security
from app.core.security import verify_password, create_user_token, pwd_context
from datetime import datetime, timezone

router = APIRouter()


@router.post("/register", response_model=UserOut)
def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    # Check if email already exists
    user = db.query(User).filter(User.email == user_in.email).first()
    if user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Set default role_id to "user" (ID: 2) - no longer from user input
    role_id = 2  # Default to "user" role
    
    # Validate role_id
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(
            status_code=400, 
            detail="Default role not found. Please contact administrator."
        )
    
    hashed_password = pwd_context.hash(user_in.password)
    db_user = User(
        email=user_in.email,
        role_id=role_id,
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
    return db_user


@router.post("/login", response_model=TokenResponse)
def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """
    User login endpoint that returns JWT token.
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
    
    access_token = create_user_token(
        user_id=user.id,
        email=user.email,
        tenant_id=current_tenant_id
    )
    
    return TokenResponse(
        access_token=access_token,
        user_id=user.id,
        email=user.email,
        tenant_id=current_tenant_id,
        tenant_ids=tenant_ids
    )


@router.post("/logout")
def logout():
    """
    Logout endpoint (client should discard token).
    Note: JWT tokens are stateless, so server-side logout requires token blacklisting.
    """
    return {"message": "Successfully logged out"}


@router.get("/token-info")
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
    
    return {
        "token_info": token_info,
        "message": "Token expires in 30 minutes from creation"
    }


@router.get("/check-token-expiration")
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
    
    return {
        "is_expired": False,
        "message": "Token is still valid"
    }