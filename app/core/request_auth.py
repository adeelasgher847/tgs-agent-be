"""Request-scoped authentication helpers (JWT + API key)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal, Optional

from app.core.workspace import Workspace
from app.models.user import User

AuthMethod = Literal["api_key", "jwt"]

AUTH_METHOD_API_KEY: AuthMethod = "api_key"
AUTH_METHOD_JWT: AuthMethod = "jwt"


@dataclass
class ApiKeyPrincipal:
    """Minimal stand-in for ``User`` on machine-to-machine API key requests."""

    current_tenant_id: uuid.UUID
    api_key_id: uuid.UUID
    id: None = None
    email: str = ""


def is_api_key_principal(obj: object) -> bool:
    return isinstance(obj, ApiKeyPrincipal)


def get_auth_method(request) -> Optional[AuthMethod]:
    return getattr(request.state, "auth_method", None)


def get_workspace_id_from_request(request) -> Optional[uuid.UUID]:
    workspace = get_workspace_from_request(request)
    if workspace is not None:
        return workspace.id
    return getattr(request.state, "workspace_id", None)


def get_workspace_from_request(request) -> Optional[Workspace]:
    return getattr(request.state, "workspace", None)
