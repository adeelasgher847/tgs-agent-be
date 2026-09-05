"""SysAdmin authentication routes — login + API key management."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from app.utils.rate_limiter import enforce_login_rate_limit
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.sysadmin.deps import get_current_sysadmin, get_sysadmin_db
from app.sysadmin.security import (
    _pwd_context,
    create_sysadmin_token,
    generate_api_key,
    hash_password,
    verify_password,
)

# Dummy hash used to prevent timing side-channel on unknown email
_DUMMY_HASH = _pwd_context.hash("dummy-password-for-timing-parity")

router = APIRouter(prefix="/auth", tags=["SysAdmin Auth"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    email: str


class ApiKeyCreateRequest(BaseModel):
    name: str
    expires_at: datetime | None = None


class ApiKeyResponse(BaseModel):
    id: uuid.UUID
    name: str
    key_prefix: str
    last_used_at: datetime | None
    expires_at: datetime | None
    is_active: bool
    created_at: datetime


class ApiKeyCreatedResponse(ApiKeyResponse):
    key: str  # shown once only


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

@router.post("/login", response_model=LoginResponse)
async def login(
    body: LoginRequest,
    request: Request,
    db: Session = Depends(get_sysadmin_db),
    _rl: None = Depends(enforce_login_rate_limit),
):
    from app.models.sysadmin_user import SysAdminUser
    from app.sysadmin.audit_service import record_audit

    stmt = select(SysAdminUser).where(SysAdminUser.email == body.email)
    admin = db.execute(stmt).scalar_one_or_none()

    # Always run bcrypt to prevent timing oracle on valid vs invalid email
    candidate_hash = admin.hashed_password if admin else _DUMMY_HASH
    password_ok = verify_password(body.password, candidate_hash)

    if not admin or not password_ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not admin.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account inactive")

    admin.last_login_at = datetime.now(timezone.utc)
    db.commit()

    token = create_sysadmin_token(admin.id, admin.email, admin.role)

    record_audit(
        db=db,
        admin=admin,
        action="LOGIN",
        source_page="LoginPage",
        ip_address=request.client.host if request.client else None,
    )

    return LoginResponse(access_token=token, role=admin.role, email=admin.email)


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------

@router.post("/api-keys", response_model=ApiKeyCreatedResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    body: ApiKeyCreateRequest,
    request: Request,
    db: Session = Depends(get_sysadmin_db),
    admin=Depends(get_current_sysadmin),
):
    from app.models.sysadmin_user import SysAdminApiKey
    from app.sysadmin.audit_service import record_audit

    raw_key, prefix, key_hash = generate_api_key()

    api_key = SysAdminApiKey(
        admin_id=admin.id,
        name=body.name,
        key_hash=key_hash,
        key_prefix=prefix,
        expires_at=body.expires_at,
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    record_audit(
        db=db,
        admin=admin,
        action="API_KEY_CREATE",
        source_page="ApiKeysPage",
        details={"key_name": body.name, "key_prefix": prefix},
        ip_address=request.client.host if request.client else None,
    )

    return ApiKeyCreatedResponse(
        id=api_key.id,
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        last_used_at=api_key.last_used_at,
        expires_at=api_key.expires_at,
        is_active=api_key.is_active,
        created_at=api_key.created_at,
        key=raw_key,
    )


@router.get("/api-keys", response_model=list[ApiKeyResponse])
async def list_api_keys(
    db: Session = Depends(get_sysadmin_db),
    admin=Depends(get_current_sysadmin),
):
    from app.models.sysadmin_user import SysAdminApiKey

    stmt = select(SysAdminApiKey).where(SysAdminApiKey.admin_id == admin.id).order_by(SysAdminApiKey.created_at.desc())
    keys = db.execute(stmt).scalars().all()
    return [
        ApiKeyResponse(
            id=k.id,
            name=k.name,
            key_prefix=k.key_prefix,
            last_used_at=k.last_used_at,
            expires_at=k.expires_at,
            is_active=k.is_active,
            created_at=k.created_at,
        )
        for k in keys
    ]


@router.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: uuid.UUID,
    request: Request,
    db: Session = Depends(get_sysadmin_db),
    admin=Depends(get_current_sysadmin),
):
    from app.models.sysadmin_user import SysAdminApiKey
    from app.sysadmin.audit_service import record_audit

    key_obj = db.get(SysAdminApiKey, key_id)
    if not key_obj or key_obj.admin_id != admin.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")

    key_obj.is_active = False
    db.commit()

    record_audit(
        db=db,
        admin=admin,
        action="API_KEY_REVOKE",
        source_page="ApiKeysPage",
        details={"key_name": key_obj.name, "key_prefix": key_obj.key_prefix},
        ip_address=request.client.host if request.client else None,
    )
