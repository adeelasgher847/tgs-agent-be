from app.db.session import SessionLocal
from typing import Generator, Union
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
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
    include_tenants: bool = False
) -> Union[User, tuple[User, TokenData]]:
    """
    JWT-based user authentication with optional tenant context.
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

    # If tenant context is not needed, return just the user
    if not include_tenants:
        return user

    # If tenant context is needed, extract and return with token data
    email: str = payload.get("email")
    tenant_ids: list = payload.get("tenant_ids", [])
    current_tenant_id: int = payload.get("current_tenant_id")

    token_data = TokenData(
        user_id=user_id,
        email=email,
        tenant_ids=tenant_ids,
        current_tenant_id=current_tenant_id
    )

    return user, token_data

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
    tenant_ids: list = payload.get("tenant_ids", [])
    current_tenant_id: int = payload.get("current_tenant_id")

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

    token_data = TokenData(
        user_id=user_id,
        email=email,
        tenant_ids=tenant_ids,
        current_tenant_id=current_tenant_id
    )

    return user, token_data 