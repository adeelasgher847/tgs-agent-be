"""Workspace API key lifecycle: generate, mask, persist, revoke."""
from __future__ import annotations

import hashlib
import secrets
import uuid
from typing import List, Optional, Tuple

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.api_key import Apikey

RAW_KEY_PREFIX = "sk_"


def sha256_hex(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def generate_raw_api_key() -> str:
    return f"{RAW_KEY_PREFIX}{secrets.token_urlsafe(32)}"


def mask_api_key(raw: str) -> str:
    """Return a display-safe masked form, e.g. ``sk_ab12••••wxyz``."""
    if len(raw) <= 12:
        return raw[:4] + "••••"
    return f"{raw[:8]}••••{raw[-4:]}"


def create_api_key(
    db: Session,
    *,
    tenant_id: uuid.UUID,
    name: str,
) -> Tuple[Apikey, str]:
    """Persist a new key and return ``(record, raw_key)``."""
    raw_key = generate_raw_api_key()
    record = Apikey(
        tenant_id=tenant_id,
        name=name.strip(),
        key_prefix=mask_api_key(raw_key),
        key_hash=sha256_hex(raw_key),
        is_active=True,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record, raw_key


def list_api_keys(db: Session, *, tenant_id: uuid.UUID) -> List[Apikey]:
    return (
        db.query(Apikey)
        .filter(Apikey.tenant_id == tenant_id)
        .order_by(Apikey.created_at.desc())
        .all()
    )


def get_api_key_for_tenant(
    db: Session,
    *,
    key_id: uuid.UUID,
    tenant_id: uuid.UUID,
) -> Optional[Apikey]:
    return (
        db.query(Apikey)
        .filter(Apikey.id == key_id, Apikey.tenant_id == tenant_id)
        .first()
    )


def revoke_api_key(db: Session, record: Apikey) -> Apikey:
    if not record.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="API key is already revoked",
        )
    record.is_active = False
    db.commit()
    db.refresh(record)
    return record


def to_api_key_out(record: Apikey) -> dict:
    return {
        "id": record.id,
        "name": record.name,
        "workspace_id": record.tenant_id,
        "masked_key": record.key_prefix,
        "is_active": record.is_active,
        "created_at": record.created_at,
        "last_used_at": record.last_used_at,
    }
