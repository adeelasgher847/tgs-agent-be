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
from app.services import role_service
from app.services import rbac_cache_service
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
    """Block read_only role from mutating HTTP methods (GET remains allowed).

    Superseded by the rank-based require_* dependencies below (which reject
    read_only on every method, not just writes) — kept for the handful of
    unit tests that exercise it directly and for require_write_access, which
    nothing in this codebase still depends on at the route level.
    """
    if request.method in _WRITE_METHODS and role_name == "read_only":
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


def _forbidden(role_required: str, user_role: Optional[str]) -> HTTPException:
    """Structured 403 per the RBAC matrix: detail.code/role_required/user_role/message."""
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail={
            "code": "forbidden",
            "role_required": role_required,
            "user_role": user_role or "none",
            "message": f"This action requires {role_required} role.",
        },
    )


def _not_a_member() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You are not a member of this tenant",
    )


def _resolve_effective_role(user: User, db: Session) -> Optional[str]:
    """Cached effective role for the user's current tenant (None = not a member)."""
    if not user.current_tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tenant selected. Please set a current tenant.",
        )
    return rbac_cache_service.get_effective_role(db, user.id, user.current_tenant_id)


def _attach_tenants(user: User, db: Session) -> User:
    if not hasattr(user, "tenants") or user.tenants is None:
        user.tenants = db.query(Tenant).join(user_tenant_association).filter(
            user_tenant_association.c.user_id == user.id
        ).all()
    return user


def _require_rank(required: str):
    """Build a dependency that passes when the caller's effective role outranks
    ``required`` in the admin > manager > config_only > read_only chain."""

    def _dependency(
        user: User = Depends(require_user_tenant),
        db: Session = Depends(get_db),
    ) -> User:
        role_name = _resolve_effective_role(user, db)
        if role_name is None:
            raise _not_a_member()
        if not role_service.has_rank(role_name, required):
            raise _forbidden(required, role_name)
        return _attach_tenants(user, db)

    _dependency.__name__ = f"require_{required}"
    return _dependency


# ─────────────────────────────────────────── canonical RBAC dependencies ──
# admin > manager > config_only > read_only. billing_only sits outside this
# chain (see require_billing) — see docs/rbac-matrix.md for the full matrix.
require_admin = _require_rank(role_service.ADMIN)
require_manager = _require_rank(role_service.MANAGER)
require_config = _require_rank(role_service.CONFIG_ONLY)
require_readonly = _require_rank(role_service.READ_ONLY)

def _require_rank_or_api_key(required: str):
    """Like _require_rank, but lets API-key (M2M) principals through untiered.

    ApiKeyPrincipal has no per-user role at all — it's a workspace-bound
    credential, not a member — and several existing v1 routes (call-flows,
    knowledge-base, allowed-domains) are exercised by both a dashboard JWT
    user and machine-to-machine API key callers today. Rejecting
    ApiKeyPrincipal here (as the plain rank dependencies do, via
    require_user_tenant) would silently break those M2M integrations.
    """

    def _dependency(
        principal: Union[User, ApiKeyPrincipal] = Depends(require_tenant),
        db: Session = Depends(get_db),
    ) -> Union[User, ApiKeyPrincipal]:
        if isinstance(principal, ApiKeyPrincipal):
            return principal
        role_name = _resolve_effective_role(principal, db)
        if role_name is None:
            raise _not_a_member()
        if not role_service.has_rank(role_name, required):
            raise _forbidden(required, role_name)
        return principal

    _dependency.__name__ = f"require_{required}_or_api_key"
    return _dependency


require_config_or_api_key = _require_rank_or_api_key(role_service.CONFIG_ONLY)
require_readonly_or_api_key = _require_rank_or_api_key(role_service.READ_ONLY)
require_admin_or_api_key = _require_rank_or_api_key(role_service.ADMIN)



def require_billing(
    user: User = Depends(require_user_tenant),
    db: Session = Depends(get_db),
) -> User:
    """billing_only, manager, and admin may call billing endpoints; config_only
    and read_only may not (billing_only is not part of the linear rank chain)."""
    role_name = _resolve_effective_role(user, db)
    if role_name is None:
        raise _not_a_member()
    if not role_service.can_access_billing(role_name):
        raise _forbidden(role_service.BILLING_ONLY, role_name)
    return user


# Legacy aliases — 'owner' and 'member' role *names* were retired in migration
# 9f3a2c7e5d41 (owner collapses into admin via is_creator; member becomes
# config_only), but ~80 existing call sites across the app (TalentSync resume/
# job/interview routers, CRM config, etc.) still import these names directly.
# Keeping them bound to the new rank-based dependencies avoids touching every
# one of those call sites while giving them correct hierarchy + caching +
# owner-override behavior for free.
#
# Known intentional behavior change: require_owner previously passed only the
# literal workspace creator; it now passes any admin-tier user, because the
# ticket's 5-role model has no creator-exclusive tier. Flagged in the RBAC
# matrix doc for the handful of routes that used require_owner specifically
# (scheduled_calls, inbound_crm, crm_config, clickup_oauth).
require_owner = require_admin
require_admin_or_owner = require_admin
require_member = require_readonly
require_member_or_admin = require_readonly


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