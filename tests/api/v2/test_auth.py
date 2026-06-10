"""
Tests for the v2 get_current_workspace dependency.

Covers all four required scenarios:
  - valid API key
  - invalid API key
  - missing API key / headers
  - workspace mismatch
"""
from __future__ import annotations

import hashlib
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_async_db, get_current_workspace, require_tenant
from app.core.exception_handlers import register_exception_handlers
from app.core.request_auth import ApiKeyPrincipal
from app.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TENANT_ID: uuid.UUID = uuid.uuid4()
RAW_KEY: str = f"tgs-{uuid.uuid4().hex}"
KEY_HASH: str = hashlib.sha256(RAW_KEY.encode()).hexdigest()


def _make_mock_row(
    *,
    tenant_id: uuid.UUID = TENANT_ID,
    is_active: bool = True,
) -> tuple:
    api_key = MagicMock()
    api_key.id = uuid.uuid4()
    api_key.key_hash = KEY_HASH
    api_key.tenant_id = tenant_id
    api_key.is_active = is_active

    tenant = MagicMock()
    tenant.id = tenant_id
    tenant.name = "Test WS"
    tenant.schema_name = "ws_test"
    tenant.status = "active"
    tenant.credits = 10.0
    tenant.stripe_customer_id = None
    tenant.stripe_subscription_id = None

    return (api_key, tenant)


def _db_override(row) -> callable:
    """Return a get_db override that yields a mock session returning *row*."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_result = MagicMock()
    mock_result.first.return_value = row
    mock_session.execute = AsyncMock(return_value=mock_result)

    async def _override() -> AsyncGenerator[AsyncSession, None]:
        yield mock_session

    return _override


def _assert_unauthorized(resp) -> None:
    assert resp.status_code == 401
    err = resp.json().get("error") or resp.json().get("detail")
    assert err is not None
    assert err["code"] == "unauthorized"
    assert err["message"] == "Invalid or missing API key"


def _client(db_override) -> TestClient:
    """Minimal app with production exception handlers and a protected endpoint."""
    mini = FastAPI()
    register_exception_handlers(mini)

    @mini.get("/protected")
    async def _protected(workspace: Workspace = Depends(get_current_workspace)):
        return {"workspace_id": str(workspace.id)}

    mini.dependency_overrides[get_async_db] = db_override
    return TestClient(mini, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestGetCurrentWorkspace:
    def test_valid_api_key_returns_200(self):
        row = _make_mock_row()
        client = _client(_db_override(row))

        resp = client.get(
            "/protected",
            headers={"x-api-key": RAW_KEY, "x-workspace-id": str(TENANT_ID)},
        )

        assert resp.status_code == 200
        assert resp.json()["workspace_id"] == str(TENANT_ID)

    def test_invalid_api_key_returns_401(self):
        # DB finds no row for an unknown key hash
        client = _client(_db_override(None))

        resp = client.get(
            "/protected",
            headers={"x-api-key": "bad-key", "x-workspace-id": str(TENANT_ID)},
        )

        _assert_unauthorized(resp)

    def test_missing_api_key_returns_401(self):
        # x-api-key header omitted entirely
        client = _client(_db_override(None))

        resp = client.get(
            "/protected",
            headers={"x-workspace-id": str(TENANT_ID)},
        )

        _assert_unauthorized(resp)

    def test_workspace_mismatch_returns_401(self):
        # The API key belongs to TENANT_ID, but the request claims a different workspace.
        # The query filters by tenant_id == x_workspace_id, so no row is returned.
        other_workspace = uuid.uuid4()
        client = _client(_db_override(None))

        resp = client.get(
            "/protected",
            headers={"x-api-key": RAW_KEY, "x-workspace-id": str(other_workspace)},
        )

        _assert_unauthorized(resp)

    def test_api_key_principal_via_require_tenant(self):
        row = _make_mock_row()
        mini = FastAPI()
        register_exception_handlers(mini)

        @mini.get("/protected")
        async def _protected(
            workspace: Workspace = Depends(get_current_workspace),
            principal=Depends(require_tenant),
        ):
            return {
                "workspace_id": str(workspace.id),
                "api_key": isinstance(principal, ApiKeyPrincipal),
            }

        mini.dependency_overrides[get_async_db] = _db_override(row)
        client = TestClient(mini, raise_server_exceptions=False)

        resp = client.get(
            "/protected",
            headers={"x-api-key": RAW_KEY, "x-workspace-id": str(TENANT_ID)},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["workspace_id"] == str(TENANT_ID)
        assert body["api_key"] is True

    def test_inactive_workspace_returns_401(self):
        row = _make_mock_row()
        row[1].status = "suspended"
        client = _client(_db_override(row))

        resp = client.get(
            "/protected",
            headers={"x-api-key": RAW_KEY, "x-workspace-id": str(TENANT_ID)},
        )

        _assert_unauthorized(resp)
