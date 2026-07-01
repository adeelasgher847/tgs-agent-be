import hashlib
from typing import Optional
import uuid

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps.db import get_async_db
from app.api.deps.auth import require_user_tenant
from app.core.request_auth import AUTH_METHOD_API_KEY, get_auth_method, get_workspace_from_request
from app.core.workspace import Workspace
from app.models.api_key import Apikey
from app.models.tenant import Tenant
from app.models.user import User

_UNAUTHORIZED_WORKSPACE = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail={"code": "unauthorized", "message": "Invalid or missing API key"},
)


def get_workspace(request: Request) -> Workspace:
    """Return the workspace (tenant) attached by auth middleware.

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
