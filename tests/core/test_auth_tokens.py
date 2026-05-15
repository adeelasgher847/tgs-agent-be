"""Tests for shared JWT bearer helpers."""
from __future__ import annotations

import uuid

from app.core.auth_tokens import extract_bearer_token, resolve_jwt_auth
from app.core.security import create_user_token


def test_extract_bearer_token():
    assert extract_bearer_token("Bearer abc.def.ghi") == "abc.def.ghi"
    assert extract_bearer_token("bearer token") == "token"
    assert extract_bearer_token("Basic xyz") is None
    assert extract_bearer_token(None) is None


def test_resolve_jwt_auth_valid():
    user_id = uuid.uuid4()
    tenant_id = uuid.uuid4()
    token = create_user_token(
        user_id=user_id,
        email="u@test.com",
        tenant_id=tenant_id,
        role="admin",
    )
    ctx = resolve_jwt_auth(token)
    assert ctx is not None
    assert ctx["user_id"] == user_id
    assert ctx["workspace_id"] == tenant_id


def test_resolve_jwt_auth_missing_tenant():
    token = create_user_token(
        user_id=uuid.uuid4(),
        email="u@test.com",
        tenant_id=None,
        role="admin",
    )
    assert resolve_jwt_auth(token) is None
