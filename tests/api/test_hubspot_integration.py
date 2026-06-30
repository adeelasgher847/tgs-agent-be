"""
Tests for HubSpot OAuth integration router endpoints.

Router functions are called directly (mirroring tests/api/test_integrations.py),
with `db` mocked and external HubSpot calls patched at the service boundary.

Coverage:
  1.  GET /connect — redirects to HubSpot's OAuth consent page
  2.  GET /callback — exchanges mock code for tokens, stores them, redirects
  3.  GET /callback — invalid state -> 400; token exchange failure -> 502
  4.  GET /contact — correct response shape; not connected -> 400; not found -> 404
  5.  DELETE "" — disconnects; 404 when not connected
  6.  GET /api/v1/integrations — includes a hubspot entry with connected/connected_at
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

_TENANT_ID = uuid.UUID("aa100000-0000-0000-0000-000000000001")


def _principal():
    p = MagicMock()
    p.current_tenant_id = _TENANT_ID
    return p


class TestConnect:
    @pytest.mark.anyio
    async def test_redirects_to_hubspot(self):
        from app.routers.hubspot_integration import hubspot_connect

        with (
            patch(
                "app.routers.hubspot_integration.hubspot_service.build_oauth_state",
                return_value="signed-state",
            ),
            patch(
                "app.routers.hubspot_integration.hubspot_service.build_authorization_url",
                return_value="https://app.hubspot.com/oauth/authorize?client_id=abc&state=signed-state",
            ),
        ):
            response = await hubspot_connect(principal=_principal())

        assert response.status_code == 302
        assert response.headers["location"] == (
            "https://app.hubspot.com/oauth/authorize?client_id=abc&state=signed-state"
        )


class TestCallback:
    @pytest.mark.anyio
    async def test_exchanges_code_and_stores_tokens(self):
        from app.routers.hubspot_integration import hubspot_callback

        db = MagicMock()
        token_response = {"access_token": "a", "refresh_token": "r", "expires_in": 1800}

        with (
            patch(
                "app.routers.hubspot_integration.hubspot_service.verify_oauth_state",
                return_value=_TENANT_ID,
            ),
            patch(
                "app.routers.hubspot_integration.hubspot_service.exchange_code_for_tokens",
                new=AsyncMock(return_value=token_response),
            ) as mock_exchange,
            patch(
                "app.routers.hubspot_integration.hubspot_service.upsert_tokens"
            ) as mock_upsert,
        ):
            response = await hubspot_callback(code="mock-auth-code", state="signed-state", db=db)

        mock_exchange.assert_awaited_once_with("mock-auth-code")
        mock_upsert.assert_called_once_with(db, _TENANT_ID, token_response)
        assert response.status_code == 302
        assert "hubspot=connected" in response.headers["location"]

    @pytest.mark.anyio
    async def test_invalid_state_returns_400(self):
        from app.routers.hubspot_integration import hubspot_callback

        with patch(
            "app.routers.hubspot_integration.hubspot_service.verify_oauth_state",
            side_effect=ValueError("Invalid or expired OAuth state"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await hubspot_callback(code="mock-auth-code", state="bad-state", db=MagicMock())

        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_token_exchange_failure_returns_502(self):
        from app.routers.hubspot_integration import hubspot_callback

        with (
            patch(
                "app.routers.hubspot_integration.hubspot_service.verify_oauth_state",
                return_value=_TENANT_ID,
            ),
            patch(
                "app.routers.hubspot_integration.hubspot_service.exchange_code_for_tokens",
                new=AsyncMock(side_effect=Exception("HubSpot 400")),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await hubspot_callback(code="mock-auth-code", state="signed-state", db=MagicMock())

        assert exc_info.value.status_code == 502


class TestContactEndpoint:
    @pytest.mark.anyio
    async def test_returns_correct_shape(self):
        from app.routers.hubspot_integration import hubspot_get_contact

        contact = {
            "id": "100451",
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "company": "Acme Corp",
            "last_interaction_date": "2026-06-01T10:00:00Z",
        }

        with (
            patch(
                "app.routers.hubspot_integration.hubspot_service.tenant_has_hubspot_connected",
                return_value=True,
            ),
            patch(
                "app.routers.hubspot_integration.hubspot_service.get_contact_for_phone",
                new=AsyncMock(return_value=contact),
            ),
        ):
            result = await hubspot_get_contact(
                phone="+15550001111", principal=_principal(), db=MagicMock()
            )

        assert result.data.id == "100451"
        assert result.data.name == "Ada Lovelace"
        assert result.data.email == "ada@example.com"
        assert result.data.company == "Acme Corp"
        assert result.data.last_interaction_date == "2026-06-01T10:00:00Z"

    @pytest.mark.anyio
    async def test_not_connected_returns_400(self):
        from app.routers.hubspot_integration import hubspot_get_contact

        with patch(
            "app.routers.hubspot_integration.hubspot_service.tenant_has_hubspot_connected",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await hubspot_get_contact(
                    phone="+15550001111", principal=_principal(), db=MagicMock()
                )

        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_no_match_returns_404(self):
        from app.routers.hubspot_integration import hubspot_get_contact

        with (
            patch(
                "app.routers.hubspot_integration.hubspot_service.tenant_has_hubspot_connected",
                return_value=True,
            ),
            patch(
                "app.routers.hubspot_integration.hubspot_service.get_contact_for_phone",
                new=AsyncMock(return_value=None),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await hubspot_get_contact(
                    phone="+15550001111", principal=_principal(), db=MagicMock()
                )

        assert exc_info.value.status_code == 404


class TestDisconnectEndpoint:
    @pytest.mark.anyio
    async def test_disconnects_successfully(self):
        from app.routers.hubspot_integration import hubspot_disconnect

        with patch(
            "app.routers.hubspot_integration.hubspot_service.disconnect",
            new=AsyncMock(return_value=True),
        ):
            result = await hubspot_disconnect(principal=_principal(), db=MagicMock())

        assert result.data.disconnected is True
        assert result.data.provider == "hubspot"

    @pytest.mark.anyio
    async def test_not_connected_returns_404(self):
        from app.routers.hubspot_integration import hubspot_disconnect

        with patch(
            "app.routers.hubspot_integration.hubspot_service.disconnect",
            new=AsyncMock(return_value=False),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await hubspot_disconnect(principal=_principal(), db=MagicMock())

        assert exc_info.value.status_code == 404


class TestIntegrationListIncludesHubSpot:
    @pytest.mark.anyio
    async def test_list_integrations_includes_hubspot_status(self):
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
                return_value=(True, connected_at),
            ),
        ):
            result = await list_integrations(request=request, user=_principal(), db=db)

        hubspot_item = next(i for i in result.integrations if i.name == "hubspot")
        assert hubspot_item.connected is True
        assert hubspot_item.connected_at == connected_at
