from app.db.session import SessionLocal
from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy.exc import InterfaceError
from app.models.user import User, user_tenant_association
from app.models.tenant import Tenant
from app.core.security import verify_token,create_user_token, create_refresh_token_value, refresh_token_expires_at
from app.models.refresh_token import RefreshToken
from app.schemas.auth import TokenResponse, RoleInfo
from app.services.role_service import is_admin_in_tenant
from app.services.role_service import get_user_product_in_tenant
import uuid

security = HTTPBearer()


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        try:
            db.close()
        except InterfaceError:
            # During app shutdown/reload, connection can already be gone.
            pass


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

# In-memory store to track credited Stripe Checkout sessions
_credited_session_ids: set[str] = set()

def is_session_already_credited(session_id: str) -> bool:
    return session_id in _credited_session_ids

def mark_session_credited(session_id: str) -> None:
    _credited_session_ids.add(session_id)


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


def get_optional_tenant_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Try to get user with tenant, but return None if authentication fails.
    Used for endpoints that support both JWT and webhook secret authentication."""
    if not credentials:
        return None
    
    try:
        payload = verify_token(credentials.credentials)
        if not payload:
            return None
        
        tenant_id = payload.get("tenant_id")
        if not tenant_id:
            return None
        
        user_id_str = payload.get("user_id")
        if not user_id_str:
            return None
        
        try:
            user_id = uuid.UUID(user_id_str)
            tenant_uuid = uuid.UUID(tenant_id)
        except ValueError:
            return None
        
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.current_tenant_id = tenant_uuid
        return user
    except:
        return None


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


def require_active_tenant(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
) -> User:
    """Ensure user's current tenant is active (not pending_payment)."""
    if not user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tenant selected. Please set a current tenant."
        )
    
    # Check tenant status
    tenant = db.query(Tenant).filter(Tenant.id == user.current_tenant_id).first()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found"
        )
    
    if tenant.status == "pending_payment":
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Insufficient credits. Please complete your payment to access this feature."
        )
    elif tenant.status != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Tenant is {tenant.status}. Please contact support."
        )
    
    return user


def require_owner(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
) -> User:
    """Ensure user is owner (only) in their current tenant."""
    if not user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tenant selected. Please set a current tenant."
        )
    
    # Get user's role in the current tenant
    from app.services.role_service import get_user_role_in_tenant
    role = get_user_role_in_tenant(db, user.id, user.current_tenant_id)
    
    if not role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not a member of this tenant"
        )
    
    # Check if user is owner (only)
    if role.name != "owner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Owner access required for this operation"
        )
    
    return user


def require_admin_or_owner(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
) -> User:
    """Ensure user is admin or owner in their current tenant."""
    if not user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tenant selected. Please set a current tenant."
        )
    
    # Get user's role in the current tenant
    from app.services.role_service import get_user_role_in_tenant
    role = get_user_role_in_tenant(db, user.id, user.current_tenant_id)
    
    if not role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not a member of this tenant"
        )
    
    # Check if user is admin or owner
    if role.name not in ["admin", "owner"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin or Owner access required for this operation"
        )
    
    return user


def issue_tokens_for_user(
    db: Session,
    user: User,
    current_tenant_id: Optional[uuid.UUID],
    role_info: Optional[RoleInfo]
) -> TokenResponse:
    """Issue access and refresh tokens for a user without password checks.
    Used for provider-based authentication flows.
    """
    access_token = create_user_token(
        user_id=user.id,
        email=user.email,
        tenant_id=current_tenant_id,
        role=role_info.name if role_info else None
    )

    rt_value = create_refresh_token_value()
    rt = RefreshToken(
        user_id=user.id,
        token=rt_value,
        expires_at=refresh_token_expires_at(),
        revoked=False
    )
    db.add(rt)
    db.commit()

    product_id = None
    if current_tenant_id:
        product = get_user_product_in_tenant(db, user.id, current_tenant_id)
        if product:
            product_id = product.id

    return TokenResponse(
        access_token=access_token,
        user_id=user.id,
        email=user.email,
        tenant_id=current_tenant_id,
        product_id=product_id,
        tenant_ids=[t.id for t in user.tenants],
        role=role_info,
        refresh_token=rt_value
    )


def require_active_subscription(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
) -> User:
    """Ensure user has at least one active paid CRM subscription with valid period."""
    from app.services.billing_service import BillingService
    if not BillingService.has_active_paid_subscription(db, user.id):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Access to CRM features requires an active paid subscription. Please subscribe to a plan for your CRM."
        )
    return user