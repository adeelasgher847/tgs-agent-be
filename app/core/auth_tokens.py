"""Shared JWT bearer extraction and validation."""
from __future__ import annotations

import uuid
from typing import Optional

from app.core.security import verify_token


def extract_bearer_token(authorization_header: Optional[str]) -> Optional[str]:
    if not authorization_header:
        return None
    parts = authorization_header.strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def resolve_jwt_auth(token: str) -> Optional[dict]:
    """
    Validate JWT and return auth context dict.

    Keys: ``user_id``, ``workspace_id`` (from token ``tenant_id``).
    Returns ``None`` when the token is invalid or missing required claims.
    """
    payload = verify_token(token)
    if not payload:
        return None

    user_id_str = payload.get("user_id")
    tenant_id_str = payload.get("tenant_id")
    if not user_id_str or not tenant_id_str:
        return None

    try:
        user_id = uuid.UUID(str(user_id_str))
        workspace_id = uuid.UUID(str(tenant_id_str))
    except ValueError:
        return None

    return {"user_id": user_id, "workspace_id": workspace_id}
