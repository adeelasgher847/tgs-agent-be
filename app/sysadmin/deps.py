"""
SysAdmin dependency injection — guards that verify sysadmin JWTs or API keys
and reject all regular tenant tokens with 401.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.sysadmin.security import decode_sysadmin_token, verify_api_key_hash

_bearer = HTTPBearer(auto_error=False)

_SCHEMA_RE = re.compile(r"^(public|tenant_[0-9a-f]{32})$")


def get_sysadmin_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _get_admin_from_token(token: str, db: Session) -> "SysAdminUser":  # type: ignore[name-defined]
    from app.models.sysadmin_user import SysAdminUser

    payload = decode_sysadmin_token(token)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired sysadmin token")

    admin_id = payload.get("sub")
    admin = db.get(SysAdminUser, uuid.UUID(admin_id))
    if not admin or not admin.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sysadmin account not found or inactive")
    return admin


def _get_admin_from_api_key(raw_key: str, db: Session) -> "SysAdminUser":  # type: ignore[name-defined]
    from sqlalchemy import select

    from app.models.sysadmin_user import SysAdminApiKey, SysAdminUser

    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    stmt = select(SysAdminApiKey).where(
        SysAdminApiKey.key_hash == key_hash,
        SysAdminApiKey.is_active.is_(True),
    )
    api_key_obj = db.execute(stmt).scalar_one_or_none()
    if not api_key_obj:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or revoked API key")
    if api_key_obj.expires_at and api_key_obj.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key expired")

    # Update last_used_at
    api_key_obj.last_used_at = datetime.now(timezone.utc)
    db.commit()

    admin = db.get(SysAdminUser, api_key_obj.admin_id)
    if not admin or not admin.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sysadmin account inactive")
    return admin


def get_current_sysadmin(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_sysadmin_db),
) -> "SysAdminUser":  # type: ignore[name-defined]
    """Resolve the authenticated sysadmin from Bearer JWT or API key."""
    token = None
    if credentials:
        token = credentials.credentials

    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    # API keys start with our prefix
    if token.startswith("tgssa_"):
        admin = _get_admin_from_api_key(token, db)
    else:
        admin = _get_admin_from_token(token, db)

    request.state.sysadmin = admin
    return admin


def require_super_admin(
    admin: "SysAdminUser" = Depends(get_current_sysadmin),  # type: ignore[name-defined]
) -> "SysAdminUser":  # type: ignore[name-defined]
    if admin.role != "SUPER_ADMIN":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="SUPER_ADMIN role required")
    return admin


def validate_tenant_schema(schema: str) -> str:
    """Prevent schema injection — only allow public or tenant_<32hex>."""
    if not _SCHEMA_RE.match(schema):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid tenant schema identifier")
    return schema
