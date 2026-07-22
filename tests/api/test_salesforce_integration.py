"""
Tests for Salesforce OAuth integration router endpoints.

Router functions are called directly (mirroring tests/api/test_hubspot_integration.py),
with `db` mocked and external Salesforce calls patched at the service boundary.

Coverage:
  1.  GET /connect — redirects to Salesforce's OAuth consent page
  2.  GET /callback — exchanges mock code for tokens, stores them, redirects
  3.  GET /callback — invalid state -> 400; token exchange failure -> 502
  4.  GET /contact — correct response shape; not connected -> 400; not found -> 404
  5.  DELETE "" — disconnects; 404 when not connected
  6.  GET /api/v1/integrations — includes a salesforce entry with connected/connected_at
  7.  GET "" / PUT /settings / GET /sync-status — status and toggle round trip
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

_TENANT_ID = uuid.UUID("aa200000-0000-0000-0000-000000000001")


def _principal():
    p = MagicMock()
    p.current_tenant_id = _TENANT_ID
    return p


class TestConnect:
    @pytest.mark.anyio
    async def test_redirects_to_salesforce(self):
        from app.routers.salesforce_integration import salesforce_connect

        with (
            patch(
                "app.routers.salesforce_integration.salesforce_service.build_oauth_state",
                return_value="signed-state",
            ),
            patch(
                "app.routers.salesforce_integration.salesforce_service.build_authorization_url",
                return_value="https://login.salesforce.com/services/oauth2/authorize?client_id=abc&state=signed-state",
            ),
        ):
            response = await salesforce_connect(principal=_principal())

        assert response.status_code == 302
        assert response.headers["location"] == (
            "https://login.salesforce.com/services/oauth2/authorize?client_id=abc&state=signed-state"
        )


class TestCallback:
    @pytest.mark.anyio
    async def test_exchanges_code_and_stores_tokens(self):
        from app.routers.salesforce_integration import salesforce_callback

        db = MagicMock()
        token_response = {
            "access_token": "a",
            "refresh_token": "r",
            "instance_url": "https://acme.my.salesforce.com",
        }

        with (
            patch(
                "app.routers.salesforce_integration.salesforce_service.verify_oauth_state",
                return_value=_TENANT_ID,
            ),
            patch(
                "app.routers.salesforce_integration.salesforce_service.exchange_code_for_tokens",
                new=AsyncMock(return_value=token_response),
            ) as mock_exchange,
            patch(
                "app.routers.salesforce_integration.salesforce_service.upsert_tokens"
            ) as mock_upsert,
        ):
            response = await salesforce_callback(
                code="mock-auth-code", state="signed-state", db=db
            )

        mock_exchange.assert_awaited_once_with("mock-auth-code")
        mock_upsert.assert_called_once_with(db, _TENANT_ID, token_response)
        assert response.status_code == 302
        assert "salesforce=connected" in response.headers["location"]

    @pytest.mark.anyio
    async def test_invalid_state_returns_400(self):
        from app.routers.salesforce_integration import salesforce_callback

        with patch(
            "app.routers.salesforce_integration.salesforce_service.verify_oauth_state",
            side_effect=ValueError("Invalid or expired OAuth state"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await salesforce_callback(
                    code="mock-auth-code", state="bad-state", db=MagicMock()
                )

        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_token_exchange_failure_returns_502(self):
        from app.routers.salesforce_integration import salesforce_callback

        with (
            patch(
                "app.routers.salesforce_integration.salesforce_service.verify_oauth_state",
                return_value=_TENANT_ID,
            ),
            patch(
                "app.routers.salesforce_integration.salesforce_service.exchange_code_for_tokens",
                new=AsyncMock(side_effect=Exception("Salesforce 400")),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await salesforce_callback(
                    code="mock-auth-code", state="signed-state", db=MagicMock()
                )

        assert exc_info.value.status_code == 502


class TestContactEndpoint:
    @pytest.mark.anyio
    async def test_returns_correct_shape(self):
        from app.routers.salesforce_integration import salesforce_get_contact

        contact = {
            "id": "003xx000004TmiQAAS",
            "name": "Ada Lovelace",
            "account": "Acme Corp",
            "email": "ada@example.com",
        }

        with (
            patch(
                "app.routers.salesforce_integration.salesforce_service.tenant_has_salesforce_connected",
                return_value=True,
            ),
            patch(
                "app.routers.salesforce_integration.salesforce_service.get_contact_for_phone",
                new=AsyncMock(return_value=contact),
            ),
        ):
            result = await salesforce_get_contact(
                phone="+15550001111", principal=_principal(), db=MagicMock()
            )

        assert result.data.id == "003xx000004TmiQAAS"
        assert result.data.name == "Ada Lovelace"
        assert result.data.account == "Acme Corp"
        assert result.data.email == "ada@example.com"

    @pytest.mark.anyio
    async def test_not_connected_returns_400(self):
        from app.routers.salesforce_integration import salesforce_get_contact

        with patch(
            "app.routers.salesforce_integration.salesforce_service.tenant_has_salesforce_connected",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await salesforce_get_contact(
                    phone="+15550001111", principal=_principal(), db=MagicMock()
                )

        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_no_match_returns_404(self):
        from app.routers.salesforce_integration import salesforce_get_contact

        with (
            patch(
                "app.routers.salesforce_integration.salesforce_service.tenant_has_salesforce_connected",
                return_value=True,
            ),
            patch(
                "app.routers.salesforce_integration.salesforce_service.get_contact_for_phone",
                new=AsyncMock(return_value=None),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await salesforce_get_contact(
                    phone="+15550001111", principal=_principal(), db=MagicMock()
                )

        assert exc_info.value.status_code == 404


class TestIntegrationStatusEndpoint:
    @pytest.mark.anyio
    async def test_returns_status_shape(self):
        from app.routers.salesforce_integration import salesforce_get_integration_status

        connected_at = datetime(2026, 6, 1, tzinfo=timezone.utc)
        settings = {
            "connected": True,
            "connected_at": connected_at,
            "last_sync_at": "2026-06-02T00:00:00Z",
            "write_back_enabled": False,
        }

        with patch(
            "app.routers.salesforce_integration.salesforce_service.get_integration_settings",
            return_value=settings,
        ):
            result = await salesforce_get_integration_status(
                principal=_principal(), db=MagicMock()
            )

        assert result.data.connected is True
        assert result.data.connected_at == connected_at
        assert result.data.last_sync_at == "2026-06-02T00:00:00Z"
        assert result.data.write_back_enabled is False


class TestSettingsEndpoint:
    @pytest.mark.anyio
    async def test_updates_settings(self):
        from app.routers.salesforce_integration import salesforce_update_settings
        from app.schemas.salesforce_integration import SalesforceSettingsUpdateRequest

        payload = SalesforceSettingsUpdateRequest(write_back_enabled=False)

        with (
            patch(
                "app.routers.salesforce_integration.salesforce_service.tenant_has_salesforce_connected",
                return_value=True,
            ),
            patch(
                "app.routers.salesforce_integration.salesforce_service.update_integration_settings"
            ) as mock_update,
            patch(
                "app.routers.salesforce_integration.salesforce_service.get_integration_settings",
                return_value={
                    "connected": True,
                    "connected_at": datetime.now(timezone.utc),
                    "last_sync_at": None,
                    "write_back_enabled": False,
                },
            ),
        ):
            result = await salesforce_update_settings(
                payload=payload, principal=_principal(), db=MagicMock()
            )

        mock_update.assert_called_once_with(
            mock_update.call_args[0][0], _TENANT_ID, write_back_enabled=False
        )
        assert result.data.write_back_enabled is False

    @pytest.mark.anyio
    async def test_not_connected_returns_400(self):
        from app.routers.salesforce_integration import salesforce_update_settings
        from app.schemas.salesforce_integration import SalesforceSettingsUpdateRequest

        payload = SalesforceSettingsUpdateRequest(write_back_enabled=False)

        with patch(
            "app.routers.salesforce_integration.salesforce_service.tenant_has_salesforce_connected",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await salesforce_update_settings(
                    payload=payload, principal=_principal(), db=MagicMock()
                )

        assert exc_info.value.status_code == 400


class TestSyncStatusEndpoint:
    @pytest.mark.anyio
    async def test_returns_sync_status_shape(self):
        from app.routers.salesforce_integration import salesforce_sync_status

        sync_status = {
            "last_lookup_at": "2026-06-01T00:00:00Z",
            "last_write_back_at": "2026-06-01T00:05:00Z",
            "last_write_back_status": "success",
            "error_count_24h": 0,
        }

        with patch(
            "app.routers.salesforce_integration.salesforce_service.get_sync_status",
            return_value=sync_status,
        ):
            result = await salesforce_sync_status(principal=_principal(), db=MagicMock())

        assert result.data.last_lookup_at == "2026-06-01T00:00:00Z"
        assert result.data.last_write_back_status == "success"
        assert result.data.error_count_24h == 0


class TestDisconnectEndpoint:
    @pytest.mark.anyio
    async def test_disconnects_successfully(self):
        from app.routers.salesforce_integration import salesforce_disconnect

        with patch(
            "app.routers.salesforce_integration.salesforce_service.disconnect",
            new=AsyncMock(return_value=True),
        ):
            result = await salesforce_disconnect(principal=_principal(), db=MagicMock())

        assert result.data.disconnected is True
        assert result.data.provider == "salesforce"

    @pytest.mark.anyio
    async def test_not_connected_returns_404(self):
        from app.routers.salesforce_integration import salesforce_disconnect

        with patch(
            "app.routers.salesforce_integration.salesforce_service.disconnect",
            new=AsyncMock(return_value=False),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await salesforce_disconnect(principal=_principal(), db=MagicMock())

        assert exc_info.value.status_code == 404


class TestIntegrationListIncludesSalesforce:
    @pytest.mark.anyio
    async def test_list_integrations_includes_salesforce_status(self):
        from app.routers.integrations import list_integrations

        request = MagicMock()
        tenant = MagicMock()
        tenant.id = _TENANT_ID
        tenant.workspace_settings = {}

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = tenant

        connected_at = datetime(2026, 6, 1, tzinfo=timezone.utc)

        with (
            patch(
                "app.core.request_auth.get_workspace_from_request",
                return_value=MagicMock(id=_TENANT_ID),
            ),
            patch(
                "app.services.hubspot_service.get_connection_status",
                return_value=(False, None),
            ),
            patch(
                "app.services.salesforce_service.get_connection_status",
                return_value=(True, connected_at),
            ),
            patch(
                "app.services.salesforce_service.get_sync_status",
                return_value={
                    "last_lookup_at": None,
                    "last_write_back_at": "2026-06-02T00:00:00Z",
                    "last_write_back_status": "success",
                    "error_count_24h": 0,
                },
            ),
            patch(
                "app.services.ghl_service.get_connection_status",
                return_value=(False, None),
            ),
        ):
            result = await list_integrations(request=request, user=_principal(), db=db)

        salesforce_item = next(i for i in result.integrations if i.name == "salesforce")
        assert salesforce_item.connected is True
        assert salesforce_item.connected_at == connected_at
        assert salesforce_item.last_sync_at == "2026-06-02T00:00:00Z"
