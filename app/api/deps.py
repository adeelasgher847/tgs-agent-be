import hashlib

from app.db.session import SessionLocal
from typing import Generator, Optional, Union
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import InterfaceError
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user import User, user_tenant_association
from app.models.tenant import Tenant
from app.core.security import verify_token,create_user_token, create_refresh_token_value, refresh_token_expires_at
from app.core.request_auth import (
    AUTH_METHOD_API_KEY,
    AUTH_METHOD_JWT,
    ApiKeyPrincipal,
    get_auth_method,
    get_workspace_from_request,
)
from app.core.workspace import Workspace
from app.db.async_session import get_db as get_async_db
from app.models.api_key import Apikey
from app.models.refresh_token import RefreshToken
from app.schemas.auth import TokenResponse, RoleInfo
from app.services.role_service import is_admin_in_tenant, get_user_role_in_tenant
from app.services.role_service import get_user_product_in_tenant
import uuid

security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)

_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

_DEACTIVATED_USER_DETAIL = "User not found or account has been deactivated"


def get_active_user_by_id(db: Session, user_id: uuid.UUID) -> Optional[User]:
    """Load a user only when not soft-deleted (``deleted_at IS NULL``)."""
    return (
        db.query(User)
        .filter(User.id == user_id, User.deleted_at.is_(None))
        .first()
    )


def _reject_readonly_on_write(request: Request, role_name: str) -> None:
    """Block readonly role from mutating HTTP methods (GET remains allowed)."""
    if request.method in _WRITE_METHODS and role_name == "readonly":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Read-only access cannot modify resources",
        )


def get_workspace(request: Request) -> Workspace:
    """
    Return the workspace (tenant) attached by auth middleware.

    Use in route handlers: ``workspace: Workspace = Depends(get_workspace)``
    """
    workspace = get_workspace_from_request(request)
    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Workspace context not available",
        )
    return workspace


def get_workspace_api_key(request: Request) -> Workspace:
    """Workspace context for machine-to-machine routes (API key only, no JWT)."""
    workspace = get_workspace(request)
    if get_auth_method(request) != AUTH_METHOD_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint requires API key authentication",
        )
    return workspace


_UNAUTHORIZED_WORKSPACE = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail={"code": "unauthorized", "message": "Invalid or missing API key"},
)


async def get_current_workspace(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="x-api-key"),
    x_workspace_id: Optional[str] = Header(default=None, alias="x-workspace-id"),
    db: AsyncSession = Depends(get_async_db),
) -> Workspace:
    """v2 M2M auth — API key headers; reuses middleware state when already resolved."""
    raw_key = (x_api_key or "").strip()
    workspace_header = (x_workspace_id or "").strip()
    if not raw_key or not workspace_header:
        raise _UNAUTHORIZED_WORKSPACE

    try:
        workspace_id = uuid.UUID(workspace_header)
    except ValueError:
        raise _UNAUTHORIZED_WORKSPACE

    existing = get_workspace_from_request(request)
    if existing is not None and existing.id == workspace_id:
        if not existing.is_active:
            raise _UNAUTHORIZED_WORKSPACE
        return existing

    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    result = await db.execute(
        select(Apikey, Tenant)
        .join(Tenant, Apikey.tenant_id == Tenant.id)
        .where(
            Apikey.key_hash == key_hash,
            Apikey.tenant_id == workspace_id,
            Apikey.is_active.is_(True),
        )
    )
    row = result.first()

    if row is None:
        raise _UNAUTHORIZED_WORKSPACE

    api_key_obj, tenant = row
    workspace = Workspace.from_tenant(tenant)
    if not workspace.is_active:
        raise _UNAUTHORIZED_WORKSPACE

    request.state.workspace = workspace
    request.state.workspace_id = workspace.id
    request.state.auth_method = AUTH_METHOD_API_KEY
    request.state.api_key_id = api_key_obj.id

    return workspace


def _user_from_middleware_jwt(request: Request, db: Session) -> User:
    workspace = get_workspace(request)
    user = get_active_user_by_id(db, request.state.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_DEACTIVATED_USER_DETAIL,
            headers={"WWW-Authenticate": "Bearer"},
        )
    user.current_tenant_id = workspace.id
    return user


def _principal_from_middleware_api_key(request: Request) -> ApiKeyPrincipal:
    workspace = get_workspace(request)
    return ApiKeyPrincipal(
        current_tenant_id=workspace.id,
        api_key_id=request.state.api_key_id,
    )


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
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_optional),
    db: Session = Depends(get_db),
) -> User:
    """JWT-based user authentication (middleware-validated or Bearer header)."""
    method = get_auth_method(request)
    if method == AUTH_METHOD_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required for this operation",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if method == AUTH_METHOD_JWT:
        return _user_from_middleware_jwt(request, db)

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

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

    user = get_active_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_DEACTIVATED_USER_DETAIL,
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def require_tenant(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_optional),
    db: Session = Depends(get_db),
) -> Union[User, ApiKeyPrincipal]:
    """Ensure the request is scoped to a workspace (JWT user or API key)."""
    method = get_auth_method(request)
    if method == AUTH_METHOD_API_KEY:
        return _principal_from_middleware_api_key(request)
    if method == AUTH_METHOD_JWT:
        return _user_from_middleware_jwt(request, db)

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id_str = payload.get("user_id")
    tenant_id = payload.get("tenant_id")
    if not user_id_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tenant selected. Please set a current tenant.",
        )

    try:
        user_id = uuid.UUID(user_id_str)
        tenant_uuid = uuid.UUID(tenant_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid tenant in token",
        )

    user = get_active_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_DEACTIVATED_USER_DETAIL,
            headers={"WWW-Authenticate": "Bearer"},
        )

    user.current_tenant_id = tenant_uuid
    return user


def require_user_tenant(
    principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
) -> User:
    """Workspace access that must be a logged-in user (not API key)."""
    if isinstance(principal, ApiKeyPrincipal):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This operation requires a user session",
        )
    return principal


def require_write_access(
    request: Request,
    user: User = Depends(require_user_tenant),
    db: Session = Depends(get_db),
) -> User:
    """Tenant user; readonly role cannot use POST/PUT/PATCH/DELETE."""
    if user.current_tenant_id:
        role = get_user_role_in_tenant(db, user.id, user.current_tenant_id)
        if role:
            _reject_readonly_on_write(request, role.name)
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
        
        user = get_active_user_by_id(db, user_id)
        if user:
            user.current_tenant_id = tenant_uuid
        return user
    except Exception:
        return None


def require_admin(
    user: User = Depends(require_write_access),
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
    user: User = Depends(require_write_access),
    db: Session = Depends(get_db),
) -> User:
    """Ensure user is a tenant member; readonly may not use write HTTP methods."""
    if not user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tenant selected. Please set a current tenant.",
        )

    role = get_user_role_in_tenant(db, user.id, user.current_tenant_id)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not a member of this tenant",
        )

    return user


def require_member_or_admin(
    user: User = Depends(require_write_access),
    db: Session = Depends(get_db),
) -> User:
    """Ensure user is a tenant member; readonly may not use write HTTP methods."""
    if not user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tenant selected. Please set a current tenant.",
        )

    role = get_user_role_in_tenant(db, user.id, user.current_tenant_id)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not a member of this tenant",
        )

    return user


def require_active_workspace(
    workspace: Workspace = Depends(get_workspace),
) -> Workspace:
    """Ensure the resolved workspace is active (middleware-attached snapshot)."""
    if workspace.status == "pending_payment":
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Insufficient credits. Please complete your payment to access this feature.",
        )
    if workspace.status != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Tenant is {workspace.status}. Please contact support.",
        )
    return workspace


def require_active_tenant(
    user: User = Depends(require_user_tenant),
    workspace: Workspace = Depends(get_workspace),
) -> User:
    """Ensure user's current tenant is active (not pending_payment)."""
    if not user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tenant selected. Please set a current tenant.",
        )

    if user.current_tenant_id != workspace.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Workspace context does not match user tenant.",
        )

    if workspace.status == "pending_payment":
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Insufficient credits. Please complete your payment to access this feature.",
        )
    if workspace.status != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Tenant is {workspace.status}. Please contact support.",
        )

    return user


def require_owner(
    user: User = Depends(require_write_access),
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
    user: User = Depends(require_write_access),
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


# Roles that may configure workspace settings (but not manage users)
_CONFIG_ROLES = frozenset({"owner", "admin", "config"})

# All roles grant at least read access; readonly is the floor
_ANY_ROLE = frozenset({"owner", "admin", "member", "config", "readonly"})


def require_config(
    user: User = Depends(require_write_access),
    db: Session = Depends(get_db),
) -> User:
    """Ensure user has config-level access (owner, admin, or config role)."""
    if not user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tenant selected. Please set a current tenant.",
        )

    role = get_user_role_in_tenant(db, user.id, user.current_tenant_id)
    if not role or role.name not in _CONFIG_ROLES:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Config, Admin, or Owner access required for this operation",
        )

    return user


def require_readonly(
    user: User = Depends(require_user_tenant),
    db: Session = Depends(get_db),
) -> User:
    """Ensure user is a tenant member with at least readonly access."""
    if not user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tenant selected. Please set a current tenant.",
        )

    role = get_user_role_in_tenant(db, user.id, user.current_tenant_id)
    if not role or role.name not in _ANY_ROLE:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not a member of this tenant",
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
    user: User = Depends(require_user_tenant),
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