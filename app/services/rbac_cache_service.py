"""Redis-backed cache for per-(user, workspace) RBAC role lookups.

Sits in front of role_service.get_membership_role_name() to cut DB hits on
high-frequency endpoints. Cache key: rbac:{user_id}:{workspace_id}, TTL 60s.

Fails open to a direct DB read whenever Redis is unavailable — caching is a
performance optimization here, not a correctness dependency.
"""
from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy.orm import Session

from app.services.role_service import get_membership_role_name
from app.utils.redis_client import get_redis_sync

TTL_SECONDS = 60

# Sentinel cached for "user has no membership row at all" — distinguishes a
# real cache miss (None from redis.get) from a confirmed non-member.
_NOT_A_MEMBER = "__not_a_member__"


def _cache_key(user_id: uuid.UUID, workspace_id: uuid.UUID) -> str:
    return f"rbac:{user_id}:{workspace_id}"


def get_effective_role(
    db: Session, user_id: uuid.UUID, workspace_id: uuid.UUID
) -> Optional[str]:
    """Cached resolution of the effective role name; None means not a member."""
    redis_client = get_redis_sync()
    key = _cache_key(user_id, workspace_id)

    if redis_client is not None:
        try:
            cached = redis_client.get(key)
        except Exception:
            cached = None
        if cached is not None:
            return None if cached == _NOT_A_MEMBER else cached

    role_name = get_membership_role_name(db, user_id, workspace_id)

    if redis_client is not None:
        try:
            redis_client.set(key, role_name or _NOT_A_MEMBER, ex=TTL_SECONDS)
        except Exception:
            pass

    return role_name


def invalidate(user_id: uuid.UUID, workspace_id: uuid.UUID) -> None:
    """Drop the cached role for a (user, workspace) pair — call on any role change."""
    redis_client = get_redis_sync()
    if redis_client is None:
        return
    try:
        redis_client.delete(_cache_key(user_id, workspace_id))
    except Exception:
        pass
