from typing import Optional, Union
import uuid

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.api.deps.db import get_db, get_active_user_by_id
from app.core.security import verify_token
from app.core.request_auth import (
    AUTH_METHOD_API_KEY,
    AUTH_METHOD_JWT,
    ApiKeyPrincipal,
    get_auth_method,
    get_workspace_from_request,
)
from app.models.user import User

security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)

_DEACTIVATED_USER_DETAIL = "User not found or account has been deactivated"


def _user_from_middleware_jwt(request: Request, db: Session) -> User:
    # Inline the get_workspace dep to avoid importing workspace.py (would create a cycle)
    workspace = get_workspace_from_request(request)
    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Workspace context not available",
        )
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
    workspace = get_workspace_from_request(request)
    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Workspace context not available",
        )
    return ApiKeyPrincipal(
        current_tenant_id=workspace.id,
        api_key_id=request.state.api_key_id,
    )


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


def get_optional_tenant_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Return the authenticated user, or None when auth is absent/invalid.

    Used for endpoints that support both JWT and webhook secret auth.
    """
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
