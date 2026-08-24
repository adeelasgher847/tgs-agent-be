"""
Tests for Slack OAuth integration router endpoints.

Router functions are called directly (mirroring tests/api/test_hubspot_integration.py),
with `db` mocked and external Slack calls patched at the service boundary
(`app.routers.slack_integration.slack_service.<fn>`).

Coverage:
  1.  GET /connect — redirect mode + JSON mode (Accept: application/json)
  2.  GET /callback — exchanges code for tokens, upserts, redirects;
      invalid/expired state -> 400; token exchange failure -> 502
  3.  DELETE "" — disconnects; 404 when not connected
  4.  GET "" — connection status shape
  5.  GET /channels — success shape; raises -> 400 (ValueError) / 502 (other)
  6.  PUT /channel — sets default channel; 400 when not connected
  7.  Multi-tenant isolation — principal's tenant_id is always what's passed
      to the service layer
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

_TENANT_ID = uuid.UUID("aa100000-0000-0000-0000-000000000001")
_OTHER_TENANT_ID = uuid.UUID("aa200000-0000-0000-0000-000000000002")


def _principal(tenant_id: uuid.UUID = _TENANT_ID):
    p = MagicMock()
    p.current_tenant_id = tenant_id
    return p


def _request(accept: str = ""):
    req = MagicMock()
    req.headers = {"accept": accept} if accept else {}
    return req


class TestConnect:
    @pytest.mark.anyio
    async def test_redirects_to_slack(self):
        from app.routers.slack_integration import slack_connect

        with patch(
            "app.routers.slack_integration.slack_service.get_authorization_url",
            return_value="https://slack.com/oauth/v2/authorize?client_id=abc&state=signed-state",
        ) as mock_url:
            response = await slack_connect(request=_request(), principal=_principal())

        mock_url.assert_called_once_with(_TENANT_ID)
        assert response.status_code == 302
        assert response.headers["location"] == (
            "https://slack.com/oauth/v2/authorize?client_id=abc&state=signed-state"
        )

    @pytest.mark.anyio
    async def test_returns_json_when_accept_header_requests_it(self):
        from app.routers.slack_integration import slack_connect

        with patch(
            "app.routers.slack_integration.slack_service.get_authorization_url",
            return_value="https://slack.com/oauth/v2/authorize?client_id=abc&state=signed-state",
        ):
            response = await slack_connect(
                request=_request("application/json"), principal=_principal()
            )

        assert response.status_code == 200
        assert response.body == (
            b'{"authorization_url":"https://slack.com/oauth/v2/authorize?'
            b'client_id=abc&state=signed-state"}'
        )

    @pytest.mark.anyio
    async def test_uses_tenant_from_principal_not_a_hardcoded_value(self):
        """Multi-tenant isolation: /connect must build the state for the
        *requesting* tenant, never a different one."""
        from app.routers.slack_integration import slack_connect

        with patch(
            "app.routers.slack_integration.slack_service.get_authorization_url",
            return_value="https://slack.com/oauth/v2/authorize",
        ) as mock_url:
            await slack_connect(
                request=_request(), principal=_principal(_OTHER_TENANT_ID)
            )

        mock_url.assert_called_once_with(_OTHER_TENANT_ID)


class TestCallback:
    @pytest.mark.anyio
    async def test_exchanges_code_and_stores_tokens(self):
        from app.routers.slack_integration import slack_callback

        db = MagicMock()
        token_response = {
            "access_token": "xoxb-abc",
            "team": {"id": "T123", "name": "Acme Workspace"},
        }

        with (
            patch(
                "app.routers.slack_integration.slack_service.verify_oauth_state",
                return_value=_TENANT_ID,
            ),
            patch(
                "app.routers.slack_integration.slack_service.exchange_code_for_tokens",
                new=AsyncMock(return_value=token_response),
            ) as mock_exchange,
            patch(
                "app.routers.slack_integration.slack_service.upsert_integration"
            ) as mock_upsert,
        ):
            response = await slack_callback(
                code="mock-auth-code", state="signed-state", db=db
            )

        mock_exchange.assert_awaited_once_with("mock-auth-code")
        mock_upsert.assert_called_once_with(db, _TENANT_ID, token_response)
        assert response.status_code == 302
        assert "slack=connected" in response.headers["location"]

    @pytest.mark.anyio
    async def test_invalid_state_returns_400(self):
        from app.routers.slack_integration import slack_callback

        with (
            patch(
                "app.routers.slack_integration.slack_service.verify_oauth_state",
                side_effect=ValueError("Invalid or expired OAuth state"),
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await slack_callback(
                code="mock-auth-code", state="bad-state", db=MagicMock()
            )

        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_expired_state_returns_400(self):
        from app.routers.slack_integration import slack_callback

        with (
            patch(
                "app.routers.slack_integration.slack_service.verify_oauth_state",
                side_effect=ValueError("Invalid or expired OAuth state"),
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await slack_callback(
                code="mock-auth-code", state="expired-state", db=MagicMock()
            )

        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_token_exchange_failure_returns_502(self):
        from app.routers.slack_integration import slack_callback

        with (
            patch(
                "app.routers.slack_integration.slack_service.verify_oauth_state",
                return_value=_TENANT_ID,
            ),
            patch(
                "app.routers.slack_integration.slack_service.exchange_code_for_tokens",
                new=AsyncMock(
                    side_effect=Exception(
                        "Slack OAuth token exchange failed: invalid_code"
                    )
                ),
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await slack_callback(code="bad-code", state="signed-state", db=MagicMock())

        assert exc_info.value.status_code == 502

    @pytest.mark.anyio
    async def test_upsert_uses_tenant_recovered_from_state_not_any_other_tenant(self):
        """Multi-tenant isolation: the callback has no auth header — the only
        source of truth for which tenant owns this connection is the signed
        state param, and upsert must be scoped to exactly that tenant."""
        from app.routers.slack_integration import slack_callback

        db = MagicMock()
        token_response = {"access_token": "xoxb-abc", "team": {}}

        with (
            patch(
                "app.routers.slack_integration.slack_service.verify_oauth_state",
                return_value=_OTHER_TENANT_ID,
            ),
            patch(
                "app.routers.slack_integration.slack_service.exchange_code_for_tokens",
                new=AsyncMock(return_value=token_response),
            ),
            patch(
                "app.routers.slack_integration.slack_service.upsert_integration"
            ) as mock_upsert,
        ):
            await slack_callback(code="mock-auth-code", state="signed-state", db=db)

        mock_upsert.assert_called_once_with(db, _OTHER_TENANT_ID, token_response)


class TestDisconnectEndpoint:
    @pytest.mark.anyio
    async def test_disconnects_successfully(self):
        from app.routers.slack_integration import slack_disconnect

        with patch(
            "app.routers.slack_integration.slack_service.disconnect",
            new=AsyncMock(return_value=True),
        ) as mock_disconnect:
            result = await slack_disconnect(principal=_principal(), db=MagicMock())

        mock_disconnect.assert_awaited_once_with(
            mock_disconnect.call_args[0][0], _TENANT_ID
        )
        assert result.data.disconnected is True
        assert result.data.provider == "slack"

    @pytest.mark.anyio
    async def test_not_connected_returns_404(self):
        from app.routers.slack_integration import slack_disconnect

        with (
            patch(
                "app.routers.slack_integration.slack_service.disconnect",
                new=AsyncMock(return_value=False),
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await slack_disconnect(principal=_principal(), db=MagicMock())

        assert exc_info.value.status_code == 404

    @pytest.mark.anyio
    async def test_uses_requesting_tenant_id(self):
        from app.routers.slack_integration import slack_disconnect

        db = MagicMock()
        with patch(
            "app.routers.slack_integration.slack_service.disconnect",
            new=AsyncMock(return_value=True),
        ) as mock_disconnect:
            await slack_disconnect(principal=_principal(_OTHER_TENANT_ID), db=db)

        mock_disconnect.assert_awaited_once_with(db, _OTHER_TENANT_ID)


class TestStatusEndpoint:
    @pytest.mark.anyio
    async def test_returns_status_shape(self):
        from app.routers.slack_integration import slack_get_integration_status
        from datetime import datetime, timezone

        connected_at = datetime(2026, 6, 1, tzinfo=timezone.utc)
        status_data = {
            "connected": True,
            "connected_at": connected_at,
            "team_name": "Acme Workspace",
            "default_channel_id": "C1",
            "default_channel_name": "general",
        }

        with patch(
            "app.routers.slack_integration.slack_service.get_integration_status",
            return_value=status_data,
        ):
            result = await slack_get_integration_status(
                principal=_principal(), db=MagicMock()
            )

        assert result.data.connected is True
        assert result.data.connected_at == connected_at
        assert result.data.team_name == "Acme Workspace"
        assert result.data.default_channel_id == "C1"
        assert result.data.default_channel_name == "general"

    @pytest.mark.anyio
    async def test_not_connected_shape(self):
        from app.routers.slack_integration import slack_get_integration_status

        status_data = {
            "connected": False,
            "connected_at": None,
            "team_name": None,
            "default_channel_id": None,
            "default_channel_name": None,
        }

        with patch(
            "app.routers.slack_integration.slack_service.get_integration_status",
            return_value=status_data,
        ):
            result = await slack_get_integration_status(
                principal=_principal(), db=MagicMock()
            )

        assert result.data.connected is False
        assert result.data.team_name is None


class TestChannelsEndpoint:
    @pytest.mark.anyio
    async def test_returns_channel_list(self):
        from app.routers.slack_integration import slack_list_channels

        channels = [{"id": "C1", "name": "general"}, {"id": "C2", "name": "sales"}]

        with patch(
            "app.routers.slack_integration.slack_service.list_channels",
            new=AsyncMock(return_value=channels),
        ):
            result = await slack_list_channels(principal=_principal(), db=MagicMock())

        assert len(result.data) == 2
        assert result.data[0].id == "C1"
        assert result.data[0].name == "general"

    @pytest.mark.anyio
    async def test_not_connected_raises_400(self):
        from app.routers.slack_integration import slack_list_channels

        with (
            patch(
                "app.routers.slack_integration.slack_service.list_channels",
                new=AsyncMock(
                    side_effect=ValueError("Slack is not connected for this workspace")
                ),
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await slack_list_channels(principal=_principal(), db=MagicMock())

        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_slack_api_failure_raises_502_not_fail_open(self):
        """This is an admin action — must surface a clear error, not silently
        return an empty channel list."""
        from app.routers.slack_integration import slack_list_channels

        with (
            patch(
                "app.routers.slack_integration.slack_service.list_channels",
                new=AsyncMock(
                    side_effect=Exception(
                        "Slack conversations.list failed: invalid_auth"
                    )
                ),
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await slack_list_channels(principal=_principal(), db=MagicMock())

        assert exc_info.value.status_code == 502

    @pytest.mark.anyio
    async def test_uses_requesting_tenant_id(self):
        from app.routers.slack_integration import slack_list_channels

        db = MagicMock()
        with patch(
            "app.routers.slack_integration.slack_service.list_channels",
            new=AsyncMock(return_value=[]),
        ) as mock_list:
            await slack_list_channels(principal=_principal(_OTHER_TENANT_ID), db=db)

        mock_list.assert_awaited_once_with(db, _OTHER_TENANT_ID)


class TestSetChannelEndpoint:
    @pytest.mark.anyio
    async def test_sets_default_channel(self):
        from app.routers.slack_integration import slack_set_channel
        from app.schemas.slack_integration import SlackSetChannelRequest

        payload = SlackSetChannelRequest(channel_id="C1", channel_name="general")
        updated_status = {
            "connected": True,
            "connected_at": None,
            "team_name": "Acme Workspace",
            "default_channel_id": "C1",
            "default_channel_name": "general",
        }

        with (
            patch(
                "app.routers.slack_integration.slack_service.tenant_has_slack_connected",
                return_value=True,
            ),
            patch(
                "app.routers.slack_integration.slack_service.set_default_channel"
            ) as mock_set,
            patch(
                "app.routers.slack_integration.slack_service.get_integration_status",
                return_value=updated_status,
            ),
        ):
            result = await slack_set_channel(
                payload=payload, principal=_principal(), db=MagicMock()
            )

        mock_set.assert_called_once_with(
            mock_set.call_args[0][0], _TENANT_ID, "C1", "general"
        )
        assert result.data.default_channel_id == "C1"
        assert result.data.default_channel_name == "general"

    @pytest.mark.anyio
    async def test_clears_default_channel(self):
        from app.routers.slack_integration import slack_set_channel
        from app.schemas.slack_integration import SlackSetChannelRequest

        payload = SlackSetChannelRequest(channel_id=None, channel_name=None)
        updated_status = {
            "connected": True,
            "connected_at": None,
            "team_name": "Acme Workspace",
            "default_channel_id": None,
            "default_channel_name": None,
        }

        with (
            patch(
                "app.routers.slack_integration.slack_service.tenant_has_slack_connected",
                return_value=True,
            ),
            patch(
                "app.routers.slack_integration.slack_service.set_default_channel"
            ) as mock_set,
            patch(
                "app.routers.slack_integration.slack_service.get_integration_status",
                return_value=updated_status,
            ),
        ):
            result = await slack_set_channel(
                payload=payload, principal=_principal(), db=MagicMock()
            )

        mock_set.assert_called_once_with(
            mock_set.call_args[0][0], _TENANT_ID, None, None
        )
        assert result.data.default_channel_id is None

    @pytest.mark.anyio
    async def test_not_connected_returns_400(self):
        from app.routers.slack_integration import slack_set_channel
        from app.schemas.slack_integration import SlackSetChannelRequest

        payload = SlackSetChannelRequest(channel_id="C1", channel_name="general")

        with (
            patch(
                "app.routers.slack_integration.slack_service.tenant_has_slack_connected",
                return_value=False,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await slack_set_channel(
                payload=payload, principal=_principal(), db=MagicMock()
            )

        assert exc_info.value.status_code == 400

    @pytest.mark.anyio
    async def test_uses_requesting_tenant_id(self):
        from app.routers.slack_integration import slack_set_channel
        from app.schemas.slack_integration import SlackSetChannelRequest

        payload = SlackSetChannelRequest(channel_id="C1", channel_name="general")
        db = MagicMock()

        with (
            patch(
                "app.routers.slack_integration.slack_service.tenant_has_slack_connected",
                return_value=True,
            ) as mock_connected,
            patch(
                "app.routers.slack_integration.slack_service.set_default_channel"
            ) as mock_set,
            patch(
                "app.routers.slack_integration.slack_service.get_integration_status",
                return_value={
                    "connected": True,
                    "connected_at": None,
                    "team_name": None,
                    "default_channel_id": "C1",
                    "default_channel_name": "general",
                },
            ),
        ):
            await slack_set_channel(
                payload=payload, principal=_principal(_OTHER_TENANT_ID), db=db
            )

        mock_connected.assert_called_once_with(db, _OTHER_TENANT_ID)
        mock_set.assert_called_once_with(db, _OTHER_TENANT_ID, "C1", "general")


class TestSlackSetChannelRequestValidation:
    def test_both_fields_set_is_valid(self):
        from app.schemas.slack_integration import SlackSetChannelRequest

        req = SlackSetChannelRequest(channel_id="C1", channel_name="general")
        assert req.channel_id == "C1"

    def test_both_fields_omitted_is_valid(self):
        from app.schemas.slack_integration import SlackSetChannelRequest

        req = SlackSetChannelRequest()
        assert req.channel_id is None
        assert req.channel_name is None

    def test_only_channel_id_set_is_invalid(self):
        from app.schemas.slack_integration import SlackSetChannelRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SlackSetChannelRequest(channel_id="C1", channel_name=None)

    def test_only_channel_name_set_is_invalid(self):
        from app.schemas.slack_integration import SlackSetChannelRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SlackSetChannelRequest(channel_id=None, channel_name="general")


class TestIntegrationListIncludesSlack:
    @pytest.mark.anyio
    async def test_list_integrations_includes_slack_status(self):
        from datetime import datetime, timezone
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
                return_value=(False, None),
            ),
            patch(
                "app.services.slack_service.get_connection_status",
                return_value=(True, connected_at),
            ),
        ):
            result = await list_integrations(request=request, user=_principal(), db=db)

        slack_item = next(i for i in result.integrations if i.name == "slack")
        assert slack_item.connected is True
        assert slack_item.connected_at == connected_at
