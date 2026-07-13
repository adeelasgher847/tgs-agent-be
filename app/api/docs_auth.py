"""HTTP Basic auth for GET /api/docs (single fixed credentials from settings)."""
from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.core.config import settings

_http_basic = HTTPBasic(auto_error=False)
_DOCS_REALM = "TGS API Docs"


def _docs_credentials_configured() -> bool:
    return bool(settings.API_DOCS_USERNAME and settings.API_DOCS_PASSWORD)


def verify_api_docs_basic(
    credentials: HTTPBasicCredentials | None = Depends(_http_basic),
) -> None:
    """
    Browser-friendly docs gate: one username/password pair from env/secrets.

    Triggers the native sign-in dialog via ``WWW-Authenticate: Basic``.
    """
    if not settings.API_DOCS_ENABLED:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    if not _docs_credentials_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API docs credentials are not configured",
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": f'Basic realm="{_DOCS_REALM}"'},
        )

    username_ok = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        settings.API_DOCS_USERNAME.encode("utf-8"),
    )
    password_ok = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        settings.API_DOCS_PASSWORD.encode("utf-8"),
    )
    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": f'Basic realm="{_DOCS_REALM}"'},
        )
