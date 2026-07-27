from typing import Optional, Union

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.api.deps.auth import require_user_tenant, require_tenant
from app.api.deps.db import get_db
from app.core.request_auth import ApiKeyPrincipal
from app.models.user import User, user_tenant_association
from app.models.tenant import Tenant
from app.services import role_service, rbac_cache_service
from app.services.role_service import get_user_role_in_tenant

_WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _reject_readonly_on_write(request: Request, role_name: str) -> None:
    """Block read_only role from mutating HTTP methods (GET remains allowed).

    Superseded by the rank-based require_* dependencies (which reject read_only on
    every method) — kept for unit tests that exercise it directly.
    """
    if request.method in _WRITE_METHODS and role_name == "read_only":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Read-only access cannot modify resources",
        )


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
        user.tenants = (
            db.query(Tenant)
            .join(user_tenant_association)
            .filter(user_tenant_association.c.user_id == user.id)
            .all()
        )
    return user


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
    """Like _require_rank, but lets API-key (M2M) principals through untiered."""

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
    """billing_only, manager, and admin may call billing endpoints."""
    role_name = _resolve_effective_role(user, db)
    if role_name is None:
        raise _not_a_member()
    if not role_service.can_access_billing(role_name):
        raise _forbidden(role_service.BILLING_ONLY, role_name)
    return user


# Legacy aliases — 'owner' and 'member' role names were retired; these keep
# existing call sites working without touching every router file.
require_owner = require_admin
require_admin_or_owner = require_admin
require_member = require_readonly
require_member_or_admin = require_readonly


def require_active_subscription(
    user: User = Depends(require_user_tenant),
    db: Session = Depends(get_db),
) -> User:
    """Ensure user has at least one active paid CRM subscription with valid period."""
    from app.services.billing_service import BillingService
    if not BillingService.has_active_paid_subscription(db, user.id):
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Access to CRM features requires an active paid subscription. Please subscribe to a plan for your CRM.",
        )
    return user
