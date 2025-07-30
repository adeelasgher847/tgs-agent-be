from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.schemas.user import UserCreate, UserOut
from app.schemas.auth import LoginRequest, TokenResponse
from app.models.user import User
from app.models.role import Role
from app.api.deps import get_db
from app.core.security import verify_password, create_user_token, get_password_hash
from passlib.context import CryptContext
from datetime import datetime, timezone

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


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
    
    access_token = create_user_token(
        user_id=user.id,
        email=user.email,
        tenant_id=tenant_ids[0] if tenant_ids else None
    )
    
    return TokenResponse(
        access_token=access_token,
        user_id=user.id,
        email=user.email,
        tenant_id=tenant_ids[0] if tenant_ids else None,
        tenant_ids=tenant_ids
    )


@router.post("/logout")
def logout():
    """
    Logout endpoint (client should discard token).
    Note: JWT tokens are stateless, so server-side logout requires token blacklisting.
    """
    return {"message": "Successfully logged out"}