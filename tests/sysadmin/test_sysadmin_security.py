"""Unit tests for SysAdmin Portal security utilities."""
from __future__ import annotations

import time
import uuid

import pytest

from app.sysadmin.security import (
    create_sysadmin_token,
    decode_sysadmin_token,
    generate_api_key,
    hash_password,
    verify_password,
)


def test_hash_and_verify_password():
    hashed = hash_password("s3cr3t!")
    assert verify_password("s3cr3t!", hashed) is True
    assert verify_password("wrong", hashed) is False


def test_generate_api_key_format():
    raw, prefix, key_hash = generate_api_key()
    assert raw.startswith("tgssa_")
    assert prefix == raw[:8]
    assert len(key_hash) == 64  # sha256 hex digest


def test_generate_api_key_is_unique():
    keys = {generate_api_key()[0] for _ in range(20)}
    assert len(keys) == 20


def test_create_and_decode_sysadmin_token():
    admin_id = uuid.uuid4()
    token = create_sysadmin_token(admin_id, "admin@tgs.ai", "SUPER_ADMIN")
    payload = decode_sysadmin_token(token)

    assert payload is not None
    assert payload["sub"] == str(admin_id)
    assert payload["email"] == "admin@tgs.ai"
    assert payload["role"] == "SUPER_ADMIN"
    assert payload["type"] == "sysadmin"


def test_decode_token_wrong_type_rejected():
    from jose import jwt
    from app.sysadmin.security import _secret, _algorithm

    payload = {"sub": str(uuid.uuid4()), "type": "tenant", "exp": time.time() + 3600}
    bad_token = jwt.encode(payload, _secret(), algorithm=_algorithm())
    assert decode_sysadmin_token(bad_token) is None


def test_decode_token_tampered_signature_rejected():
    admin_id = uuid.uuid4()
    token = create_sysadmin_token(admin_id, "x@x.com", "ADMIN")
    tampered = token[:-4] + "AAAA"
    assert decode_sysadmin_token(tampered) is None
