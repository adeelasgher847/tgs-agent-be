from app.db.session import SessionLocal
from typing import Generator
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.models.user import User, user_tenant_association
from app.models.tenant import Tenant
from app.core.security import verify_token
from app.services.role_service import is_admin_in_tenant
import uuid

security = HTTPBearer()


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user_jwt(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """JWT-based user authentication."""
    payload = verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id_str = payload.get("user_id")
    if not user_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = uuid.UUID(user_id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user ID format",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def require_tenant(
    user: User = Depends(get_current_user_jwt), 
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Ensure user has a current tenant set."""
    payload = verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    tenant_id = payload.get("tenant_id")
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tenant selected. Please set a current tenant."
        )
    
    try:
        user.current_tenant_id = uuid.UUID(tenant_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid tenant in token"
        )
    
    return user


def require_admin(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
) -> User:
    """Ensure user is an admin in their current tenant."""
    if not user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tenant selected. Please set a current tenant."
        )
    
    if not is_admin_in_tenant(db, user.id, user.current_tenant_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required for this operation"
        )
    
    # Ensure tenants relationship is loaded
    if not hasattr(user, 'tenants') or user.tenants is None:
        user.tenants = db.query(Tenant).join(user_tenant_association).filter(
            user_tenant_association.c.user_id == user.id
        ).all()
    
    return user


def require_member(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
) -> User:
    """Ensure user is a member (admin or regular member) in their current tenant."""
    if not user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tenant selected. Please set a current tenant."
        )
    
    # Check if user has any role in the tenant (admin or member)
    from app.models.user import user_tenant_association
    from app.models.role import Role
    
    result = db.query(user_tenant_association).join(Role).filter(
        user_tenant_association.c.user_id == user.id,
        user_tenant_association.c.tenant_id == user.current_tenant_id
    ).first()
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not a member of this tenant"
        )
    
    return user


def require_member_or_admin(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
) -> User:
    """Ensure user is either a member or admin in their current tenant."""
    if not user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tenant selected. Please set a current tenant."
        )
    
    # Check if user has any role in the tenant (admin or member)
    from app.models.user import user_tenant_association
    from app.models.role import Role
    
    result = db.query(user_tenant_association).join(Role).filter(
        user_tenant_association.c.user_id == user.id,
        user_tenant_association.c.tenant_id == user.current_tenant_id
    ).first()
    
    if not result:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not a member of this tenant"
        )
    
    return user