"""
SysAdmin-specific security — separate JWT secret and auth utilities.
These tokens are entirely isolated from tenant JWTs and cannot be used
against regular /api/v1 or /api/v2 endpoints.
"""
from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SYSADMIN_TOKEN_TYPE = "sysadmin"
SYSADMIN_APIKEY_TOKEN_TYPE = "sysadmin_apikey"
ACCESS_TOKEN_EXPIRE_MINUTES = 60  # 1-hour short-lived sessions


def _secret() -> str:
    key = getattr(settings, "SYSADMIN_JWT_SECRET", None) or settings.SECRET_KEY + ":sysadmin"
    return key


def _algorithm() -> str:
    return getattr(settings, "ALGORITHM", "HS256") or "HS256"


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def create_sysadmin_token(admin_id: uuid.UUID, email: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(admin_id),
        "email": email,
        "role": role,
        "type": SYSADMIN_TOKEN_TYPE,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _secret(), algorithm=_algorithm())


def decode_sysadmin_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, _secret(), algorithms=[_algorithm()])
        if payload.get("type") != SYSADMIN_TOKEN_TYPE:
            return None
        return payload
    except JWTError:
        return None


# ---------------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------------

def generate_api_key() -> tuple[str, str, str]:
    """Return (raw_key, key_prefix, key_hash)."""
    raw = "tgssa_" + secrets.token_urlsafe(40)
    prefix = raw[:8]
    key_hash = hashlib.sha256(raw.encode()).hexdigest()
    return raw, prefix, key_hash


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def verify_api_key_hash(raw: str, stored_hash: str) -> bool:
    return hashlib.sha256(raw.encode()).hexdigest() == stored_hash
