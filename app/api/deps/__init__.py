"""
app.api.deps — backward-compatible re-export shim.

All public names that were available from the old monolithic deps.py are
re-exported here so that every existing ``from app.api.deps import ...``
call site continues to work without modification.

Internal code should import from the sub-modules directly; this __init__ is
the compatibility surface for the rest of the application.
"""

from app.api.deps.db import (
    get_db,
    get_async_db,
    get_active_user_by_id,
)
from app.api.deps.auth import (
    security,
    security_optional,
    _DEACTIVATED_USER_DETAIL,
    _user_from_middleware_jwt,
    _principal_from_middleware_api_key,
    get_current_user_jwt,
    require_tenant,
    require_user_tenant,
    get_optional_tenant_user,
)
from app.api.deps.workspace import (
    get_workspace,
    get_workspace_api_key,
    get_current_workspace,
    require_active_workspace,
    require_active_tenant,
)
from app.api.deps.rbac import (
    _WRITE_METHODS,
    _reject_readonly_on_write,
    _forbidden,
    _not_a_member,
    _resolve_effective_role,
    _attach_tenants,
    _require_rank,
    _require_rank_or_api_key,
    require_write_access,
    require_admin,
    require_manager,
    require_config,
    require_readonly,
    require_billing,
    require_config_or_api_key,
    require_readonly_or_api_key,
    require_admin_or_api_key,
    require_owner,
    require_admin_or_owner,
    require_member,
    require_member_or_admin,
    require_active_subscription,
)
from app.api.deps.tokens import issue_tokens_for_user

__all__ = [
    # db
    "get_db",
    "get_async_db",
    "get_active_user_by_id",
    # auth
    "security",
    "security_optional",
    "_DEACTIVATED_USER_DETAIL",
    "_user_from_middleware_jwt",
    "_principal_from_middleware_api_key",
    "get_current_user_jwt",
    "require_tenant",
    "require_user_tenant",
    "get_optional_tenant_user",
    # workspace
    "get_workspace",
    "get_workspace_api_key",
    "get_current_workspace",
    "require_active_workspace",
    "require_active_tenant",
    # rbac
    "_WRITE_METHODS",
    "_reject_readonly_on_write",
    "_forbidden",
    "_not_a_member",
    "_resolve_effective_role",
    "_attach_tenants",
    "_require_rank",
    "_require_rank_or_api_key",
    "require_write_access",
    "require_admin",
    "require_manager",
    "require_config",
    "require_readonly",
    "require_billing",
    "require_config_or_api_key",
    "require_readonly_or_api_key",
    "require_admin_or_api_key",
    "require_owner",
    "require_admin_or_owner",
    "require_member",
    "require_member_or_admin",
    "require_active_subscription",
    # tokens
    "issue_tokens_for_user",
]
