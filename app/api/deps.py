from app.db.session import SessionLocal
from typing import Generator, Union, Optional
from fastapi import Depends, HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.models.user import User
from app.core.security import verify_token
from app.schemas.auth import TokenData

security = HTTPBearer()

def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(
    x_user_id: int = Header(..., description="User ID for authentication"),
    db: Session = Depends(get_db)
) -> User:
    """
    Simple user authentication via header.
    In production, this would be replaced with JWT token validation.
    """
    user = db.query(User).filter(User.id == x_user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def get_current_user_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    JWT-based user authentication.
    Validates JWT token and returns the authenticated user.
    """
    token = credentials.credentials
    payload = verify_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: int = payload.get("user_id")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user

def get_current_user_with_tenants_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> tuple[User, TokenData]:
    """
    JWT-based user authentication with tenant information.
    Validates JWT token and returns the authenticated user along with token data.
    """
    token = credentials.credentials
    payload = verify_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: int = payload.get("user_id")
    email: str = payload.get("email")
    tenant_id: int = payload.get("tenant_id")

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Additional validation: check if email in token matches user's email
    if user.email != email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token email does not match user email",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get current tenant IDs from database (not from token)
    current_tenant_ids = [tenant.id for tenant in user.tenants]

    token_data = TokenData(
        user_id=user_id,
        email=email,
        tenant_ids=current_tenant_ids,
        tenant_id=tenant_id
    )

    return user, token_data 