"""
Tests for GoHighLevel (GHL) OAuth integration router endpoints.

Router functions are called directly (mirroring tests/api/test_salesforce_integration.py),
with `db` mocked and external GHL calls patched at the service boundary.

Coverage:
  1.  GET /connect — redirects to GHL's OAuth consent page
  2.  GET /callback — exchanges mock code for tokens, stores them, redirects
  3.  GET /callback — invalid state -> 400; token exchange failure -> 502
  4.  GET /contact — correct response shape; not connected -> 400; not found -> 404
  5.  POST /note — creates a note directly; not connected -> 400; upstream failure -> 502
  6.  DELETE "" — disconnects; 404 when not connected
  7.  GET /api/v1/integrations — includes a GoHighLevel entry with connected/connected_at
  8.  GET "" / PUT /settings / GET /sync-status — status and toggle round trip
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

_TENANT_ID = uuid.UUID("aa300000-0000-0000-0000-000000000001")


def _principal():
    p = MagicMock()
    p.current_tenant_id = _TENANT_ID
    return p


class TestConnect:
    @pytest.mark.anyio
    async def test_redirects_to_ghl(self):
        from app.routers.ghl_integration import ghl_connect

        with (
            patch(
                "app.routers.ghl_integration.ghl_service.build_oauth_state",
                return_value="signed-state",
            ),
            patch(
                "app.routers.ghl_integration.ghl_service.build_authorization_url",
                return_value="https://marketplace.gohighlevel.com/oauth/chooselocation?client_id=abc&state=signed-state",
            ),
        ):
            response = await ghl_connect(principal=_principal())

        assert response.status_code == 302
        assert response.headers["location"] == (
            "https://marketplace.gohighlevel.com/oauth/chooselocation?client_id=abc&state=signed-state"
        )


class TestCallback:
    @pytest.mark.anyio
    async def test_exchanges_code_and_stores_tokens(self):
        from app.routers.ghl_integration import ghl_callback

        db = MagicMock()
        token_response = {
            "access_token": "a",
            "refresh_token": "r",
            "expires_in": 3600,
            "locationId": "loc-1",
        }

        with (
            patch(
                "app.routers.ghl_integration.ghl_service.verify_oauth_state",
                return_value=_TENANT_ID,
            ),
            patch(
                "app.routers.ghl_integration.ghl_service.exchange_code_for_tokens",
                new=AsyncMock(return_value=token_response),
            ) as mock_exchange,
            patch(
                "app.routers.ghl_integration.ghl_service.upsert_tokens"
            ) as mock_upsert,
        ):
            response = await ghl_callback(code="mock-auth-code", state="signed-state", db=db)

        mock_exchange.assert_awaited_once_with("mock-auth-code")
        mock_upsert.assert_called_once_with(db, _TENANT_ID, token_response)
        assert response.status_code == 302
        assert "ghl=connected" in response.headers["location"]

    @pytest.mark.anyio
    async def test_invalid_state_returns_400(self):
        from app.routers.ghl_integration import ghl_callback

        with patch(
            "app.routers.ghl_integration.ghl_service.verify_oauth_state",
            side_effect=ValueError("Invalid or expired OAuth state"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await ghl_callback(code="mock-auth-code", state="bad-state", db=MagicMock())

        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_token_exchange_failure_returns_502(self):
        from app.routers.ghl_integration import ghl_callback

        with (
            patch(
                "app.routers.ghl_integration.ghl_service.verify_oauth_state",
                return_value=_TENANT_ID,
            ),
            patch(
                "app.routers.ghl_integration.ghl_service.exchange_code_for_tokens",
                new=AsyncMock(side_effect=Exception("GHL 400")),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await ghl_callback(code="mock-auth-code", state="signed-state", db=MagicMock())

        assert exc_info.value.status_code == 502


class TestContactEndpoint:
    @pytest.mark.anyio
    async def test_returns_correct_shape(self):
        from app.routers.ghl_integration import ghl_get_contact

        contact = {
            "id": "contact-1",
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "tags": ["vip"],
            "pipeline_stage": "Negotiation",
            "last_activity_date": "2026-06-01T00:00:00Z",
        }

        with (
            patch(
                "app.routers.ghl_integration.ghl_service.tenant_has_ghl_connected",
                return_value=True,
            ),
            patch(
                "app.routers.ghl_integration.ghl_service.get_contact_for_phone",
                new=AsyncMock(return_value=contact),
            ),
        ):
            result = await ghl_get_contact(
                phone="+61412345678", principal=_principal(), db=MagicMock()
            )

        assert result.data.id == "contact-1"
        assert result.data.name == "Ada Lovelace"
        assert result.data.tags == ["vip"]
        assert result.data.pipeline_stage == "Negotiation"

    @pytest.mark.anyio
    async def test_not_connected_returns_400(self):
        from app.routers.ghl_integration import ghl_get_contact

        with patch(
            "app.routers.ghl_integration.ghl_service.tenant_has_ghl_connected",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await ghl_get_contact(
                    phone="+61412345678", principal=_principal(), db=MagicMock()
                )

        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_no_match_returns_404(self):
        from app.routers.ghl_integration import ghl_get_contact

        with (
            patch(
                "app.routers.ghl_integration.ghl_service.tenant_has_ghl_connected",
                return_value=True,
            ),
            patch(
                "app.routers.ghl_integration.ghl_service.get_contact_for_phone",
                new=AsyncMock(return_value=None),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await ghl_get_contact(
                    phone="+61412345678", principal=_principal(), db=MagicMock()
                )

        assert exc_info.value.status_code == 404


class TestNoteEndpoint:
    @pytest.mark.anyio
    async def test_creates_note_successfully(self):
        from app.routers.ghl_integration import ghl_create_note
        from app.schemas.ghl_integration import GhlNoteCreateRequest

        payload = GhlNoteCreateRequest(contact_id="contact-1", content="Called about billing.")

        with (
            patch(
                "app.routers.ghl_integration.ghl_service.tenant_has_ghl_connected",
                return_value=True,
            ),
            patch(
                "app.routers.ghl_integration.ghl_service.get_valid_access_token",
                new=AsyncMock(return_value=("access-token", "loc-1")),
            ),
            patch(
                "app.routers.ghl_integration.ghl_service.create_note",
                new=AsyncMock(return_value={"id": "note-1"}),
            ) as mock_create,
        ):
            result = await ghl_create_note(payload=payload, principal=_principal(), db=MagicMock())

        mock_create.assert_awaited_once_with(
            "access-token", "contact-1", "Called about billing.", _TENANT_ID
        )
        assert result.data.contact_id == "contact-1"
        assert result.data.id == "note-1"

    @pytest.mark.anyio
    async def test_not_connected_returns_400(self):
        from app.routers.ghl_integration import ghl_create_note
        from app.schemas.ghl_integration import GhlNoteCreateRequest

        payload = GhlNoteCreateRequest(contact_id="contact-1", content="Called about billing.")

        with (
            patch(
                "app.routers.ghl_integration.ghl_service.tenant_has_ghl_connected",
                return_value=False,
            ),
            patch(
                "app.routers.ghl_integration.ghl_service.get_valid_access_token",
            ) as mock_get_token,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await ghl_create_note(payload=payload, principal=_principal(), db=MagicMock())

        assert exc_info.value.status_code == 400
        mock_get_token.assert_not_called()

    @pytest.mark.anyio
    async def test_connected_but_token_unavailable_returns_502(self):
        """
        Connected, but the token refresh failed (e.g. a transient GHL outage or
        a revoked refresh token) — must be distinguished from "never connected"
        so the client isn't told to redo the OAuth flow unnecessarily.
        """
        from app.routers.ghl_integration import ghl_create_note
        from app.schemas.ghl_integration import GhlNoteCreateRequest

        payload = GhlNoteCreateRequest(contact_id="contact-1", content="Called about billing.")

        with (
            patch(
                "app.routers.ghl_integration.ghl_service.tenant_has_ghl_connected",
                return_value=True,
            ),
            patch(
                "app.routers.ghl_integration.ghl_service.get_valid_access_token",
                new=AsyncMock(return_value=None),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await ghl_create_note(payload=payload, principal=_principal(), db=MagicMock())

        assert exc_info.value.status_code == 502

    @pytest.mark.anyio
    async def test_upstream_failure_returns_502(self):
        from app.routers.ghl_integration import ghl_create_note
        from app.schemas.ghl_integration import GhlNoteCreateRequest

        payload = GhlNoteCreateRequest(contact_id="contact-1", content="Called about billing.")

        with (
            patch(
                "app.routers.ghl_integration.ghl_service.tenant_has_ghl_connected",
                return_value=True,
            ),
            patch(
                "app.routers.ghl_integration.ghl_service.get_valid_access_token",
                new=AsyncMock(return_value=("access-token", "loc-1")),
            ),
            patch(
                "app.routers.ghl_integration.ghl_service.create_note",
                new=AsyncMock(side_effect=Exception("GHL 500")),
            ),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await ghl_create_note(payload=payload, principal=_principal(), db=MagicMock())

        assert exc_info.value.status_code == 502


class TestIntegrationStatusEndpoint:
    @pytest.mark.anyio
    async def test_returns_status_shape(self):
        from app.routers.ghl_integration import ghl_get_integration_status

        connected_at = datetime(2026, 6, 1, tzinfo=timezone.utc)
        settings = {
            "connected": True,
            "connected_at": connected_at,
            "last_sync_at": "2026-06-02T00:00:00Z",
            "write_back_enabled": False,
        }

        with patch(
            "app.routers.ghl_integration.ghl_service.get_integration_settings",
            return_value=settings,
        ):
            result = await ghl_get_integration_status(principal=_principal(), db=MagicMock())

        assert result.data.connected is True
        assert result.data.connected_at == connected_at
        assert result.data.last_sync_at == "2026-06-02T00:00:00Z"
        assert result.data.write_back_enabled is False


class TestSettingsEndpoint:
    @pytest.mark.anyio
    async def test_updates_settings(self):
        from app.routers.ghl_integration import ghl_update_settings
        from app.schemas.ghl_integration import GhlSettingsUpdateRequest

        payload = GhlSettingsUpdateRequest(write_back_enabled=False)

        with (
            patch(
                "app.routers.ghl_integration.ghl_service.tenant_has_ghl_connected",
                return_value=True,
            ),
            patch(
                "app.routers.ghl_integration.ghl_service.update_integration_settings"
            ) as mock_update,
            patch(
                "app.routers.ghl_integration.ghl_service.get_integration_settings",
                return_value={
                    "connected": True,
                    "connected_at": datetime.now(timezone.utc),
                    "last_sync_at": None,
                    "write_back_enabled": False,
                },
            ),
        ):
            result = await ghl_update_settings(
                payload=payload, principal=_principal(), db=MagicMock()
            )

        mock_update.assert_called_once_with(
            mock_update.call_args[0][0], _TENANT_ID, write_back_enabled=False
        )
        assert result.data.write_back_enabled is False

    @pytest.mark.anyio
    async def test_not_connected_returns_400(self):
        from app.routers.ghl_integration import ghl_update_settings
        from app.schemas.ghl_integration import GhlSettingsUpdateRequest

        payload = GhlSettingsUpdateRequest(write_back_enabled=False)

        with patch(
            "app.routers.ghl_integration.ghl_service.tenant_has_ghl_connected",
            return_value=False,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await ghl_update_settings(
                    payload=payload, principal=_principal(), db=MagicMock()
                )

        assert exc_info.value.status_code == 400


class TestSyncStatusEndpoint:
    @pytest.mark.anyio
    async def test_returns_sync_status_shape(self):
        from app.routers.ghl_integration import ghl_sync_status

        sync_status = {
            "last_lookup_at": "2026-06-01T00:00:00Z",
            "last_write_back_at": "2026-06-01T00:05:00Z",
            "last_write_back_status": "success",
            "last_ghl_error": None,
            "error_count_24h": 0,
        }

        with patch(
            "app.routers.ghl_integration.ghl_service.get_sync_status",
            return_value=sync_status,
        ):
            result = await ghl_sync_status(principal=_principal(), db=MagicMock())

        assert result.data.last_lookup_at == "2026-06-01T00:00:00Z"
        assert result.data.last_write_back_status == "success"
        assert result.data.error_count_24h == 0


class TestDisconnectEndpoint:
    @pytest.mark.anyio
    async def test_disconnects_successfully(self):
        from app.routers.ghl_integration import ghl_disconnect

        with patch(
            "app.routers.ghl_integration.ghl_service.disconnect",
            new=AsyncMock(return_value=True),
        ):
            result = await ghl_disconnect(principal=_principal(), db=MagicMock())

        assert result.data.disconnected is True
        assert result.data.provider == "gohighlevel"

    @pytest.mark.anyio
    async def test_not_connected_returns_404(self):
        from app.routers.ghl_integration import ghl_disconnect

        with patch(
            "app.routers.ghl_integration.ghl_service.disconnect",
            new=AsyncMock(return_value=False),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await ghl_disconnect(principal=_principal(), db=MagicMock())

        assert exc_info.value.status_code == 404


class TestIntegrationListIncludesGhl:
    @pytest.mark.anyio
    async def test_list_integrations_includes_ghl_status(self):
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
                return_value=(False, None),
            ),
            patch(
                "app.services.ghl_service.get_connection_status",
                return_value=(True, connected_at),
            ),
            patch(
                "app.services.ghl_service.get_sync_status",
                return_value={
                    "last_lookup_at": None,
                    "last_write_back_at": "2026-06-02T00:00:00Z",
                    "last_write_back_status": "success",
                    "last_ghl_error": None,
                    "error_count_24h": 0,
                },
            ),
        ):
            result = await list_integrations(request=request, user=_principal(), db=db)

        ghl_item = next(i for i in result.integrations if i.name == "GoHighLevel")
        assert ghl_item.connected is True
        assert ghl_item.connected_at == connected_at
        assert ghl_item.last_sync_at == "2026-06-02T00:00:00Z"
