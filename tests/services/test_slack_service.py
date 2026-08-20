"""
Tests for Slack post-call-summary OAuth integration service.

External HTTP calls (OAuth token endpoint, conversations.list, chat.postMessage,
auth.revoke) are mocked at the boundary (`_request_with_backoff`), mirroring
tests/services/test_hubspot_service.py's convention.

Coverage:
  1.  OAuth state: sign/verify round trip, tamper rejection, wrong purpose
      rejection, expired state rejection
  2.  Authorization URL: client_id/scope/state present
  3.  Token exchange: success shape; `ok: false` -> ValueError; HTTP failure -> raises
  4.  upsert_integration: creates a new row; updates an existing row (encrypts token,
      stores team metadata)
  5.  get_valid_access_token: decrypts and returns; None when not connected
  6.  list_channels: success shape (incl. pagination via cursor); raises (does not
      fail open) when not connected / Slack API error / ok:false
  7.  set_default_channel: persists; clears when both args None; raises when not
      connected
  8.  disconnect: revokes + deletes; returns False when not connected; revoke
      failure still deletes local row (best-effort)
  9.  tenant_has_slack_connected: True/False
  10. send_call_summary: posts to call-flow-override channel over workspace
      default; correct phone/summary/sentiment fields; fail-open on every
      failure path (no call_flow, disabled, not connected, no channel,
      decrypt failure, post failure); lazily computes analysis when not cached
  11. 429 backoff (shared helper, mirrors HubSpot's suite)
  12. Slack token encryption round trip (AES-256-GCM)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import slack_service

_TENANT_ID = uuid.UUID("aa100000-0000-0000-0000-000000000001")
_OTHER_TENANT_ID = uuid.UUID("aa200000-0000-0000-0000-000000000002")
_SESSION_ID = uuid.UUID("bb100000-0000-0000-0000-000000000002")


# ── OAuth state ────────────────────────────────────────────────────────────────


class TestOAuthState:
    def test_roundtrip(self):
        state = slack_service.build_oauth_state(_TENANT_ID)
        assert slack_service.verify_oauth_state(state) == _TENANT_ID

    def test_tampered_state_rejected(self):
        state = slack_service.build_oauth_state(_TENANT_ID)
        with pytest.raises(ValueError):
            slack_service.verify_oauth_state(state + "x")

    def test_wrong_purpose_rejected(self):
        from jose import jwt as jose_jwt
        from app.core.config import settings

        bad_state = jose_jwt.encode(
            {
                "tenant_id": str(_TENANT_ID),
                "purpose": "something_else",
                "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
            },
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        with pytest.raises(ValueError):
            slack_service.verify_oauth_state(bad_state)

    def test_expired_state_rejected(self):
        from jose import jwt as jose_jwt
        from app.core.config import settings

        expired_state = jose_jwt.encode(
            {
                "tenant_id": str(_TENANT_ID),
                "purpose": "slack_oauth_state",
                "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
            },
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        with pytest.raises(ValueError):
            slack_service.verify_oauth_state(expired_state)

    def test_missing_tenant_id_rejected(self):
        from jose import jwt as jose_jwt
        from app.core.config import settings

        bad_state = jose_jwt.encode(
            {
                "purpose": "slack_oauth_state",
                "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
            },
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        with pytest.raises(ValueError):
            slack_service.verify_oauth_state(bad_state)


class TestAuthorizationUrl:
    def test_contains_client_id_scope_and_state(self):
        with patch(
            "app.services.slack_service.get_slack_oauth_credentials",
            return_value=("client-123", "secret-456"),
        ):
            url = slack_service.get_authorization_url(_TENANT_ID)

        assert url.startswith(slack_service.AUTHORIZE_URL)
        assert "client_id=client-123" in url
        assert "channels%3Aread" in url or "channels:read" in url
        assert "chat%3Awrite" in url or "chat:write" in url
        assert "state=" in url


# ── Token exchange ──────────────────────────────────────────────────────────────


class TestExchangeCodeForTokens:
    @pytest.mark.anyio
    async def test_success_returns_payload(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "ok": True,
            "access_token": "xoxb-abc",
            "team": {"id": "T123", "name": "Acme Workspace"},
        }

        with (
            patch(
                "app.services.slack_service.get_slack_oauth_credentials",
                return_value=("client-123", "secret-456"),
            ),
            patch(
                "app.services.slack_service._request_with_backoff",
                new=AsyncMock(return_value=response),
            ),
        ):
            payload = await slack_service.exchange_code_for_tokens("mock-code")

        assert payload["access_token"] == "xoxb-abc"
        assert payload["team"]["name"] == "Acme Workspace"

    @pytest.mark.anyio
    async def test_ok_false_raises_value_error(self):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"ok": False, "error": "invalid_code"}

        with (
            patch(
                "app.services.slack_service.get_slack_oauth_credentials",
                return_value=("client-123", "secret-456"),
            ),
            patch(
                "app.services.slack_service._request_with_backoff",
                new=AsyncMock(return_value=response),
            ),
            pytest.raises(ValueError, match="invalid_code"),
        ):
            await slack_service.exchange_code_for_tokens("bad-code")

    @pytest.mark.anyio
    async def test_http_failure_raises(self):
        import httpx

        response = MagicMock()
        response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock(status_code=500)
            )
        )

        with (
            patch(
                "app.services.slack_service.get_slack_oauth_credentials",
                return_value=("client-123", "secret-456"),
            ),
            patch(
                "app.services.slack_service._request_with_backoff",
                new=AsyncMock(return_value=response),
            ),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await slack_service.exchange_code_for_tokens("mock-code")


# ── WorkspaceIntegration CRUD ────────────────────────────────────────────────


class TestUpsertIntegration:
    def test_creates_new_row_when_none_exists(self):
        db = MagicMock()
        token_response = {
            "access_token": "xoxb-abc",
            "team": {"id": "T123", "name": "Acme Workspace"},
        }

        with patch("app.services.slack_service.get_integration", return_value=None):
            row = slack_service.upsert_integration(db, _TENANT_ID, token_response)

        assert row.workspace_id == _TENANT_ID
        assert row.provider == "slack"
        assert row.access_token.startswith("slack_gcm1:")
        assert row.extra_metadata["team_id"] == "T123"
        assert row.extra_metadata["team_name"] == "Acme Workspace"
        db.add.assert_called_once_with(row)
        db.commit.assert_called_once()
        db.refresh.assert_called_once_with(row)

    def test_updates_existing_row_preserving_other_metadata(self):
        db = MagicMock()
        existing = MagicMock()
        existing.extra_metadata = {
            "default_channel_id": "C1",
            "default_channel_name": "general",
        }
        token_response = {
            "access_token": "xoxb-new",
            "team": {"id": "T999", "name": "New Team Name"},
        }

        with patch("app.services.slack_service.get_integration", return_value=existing):
            row = slack_service.upsert_integration(db, _TENANT_ID, token_response)

        assert row is existing
        assert row.extra_metadata["team_id"] == "T999"
        assert row.extra_metadata["team_name"] == "New Team Name"
        # Reconnect must not clobber an already-configured default channel.
        assert row.extra_metadata["default_channel_id"] == "C1"
        assert row.access_token.startswith("slack_gcm1:")


class TestGetValidAccessToken:
    @pytest.mark.anyio
    async def test_returns_decrypted_token(self):
        db = MagicMock()
        row = MagicMock()
        row.access_token = "gcm1:ciphertext"

        with (
            patch("app.services.slack_service.get_integration", return_value=row),
            patch(
                "app.services.slack_service.decrypt_slack_token",
                return_value="xoxb-plain",
            ) as mock_decrypt,
        ):
            token = await slack_service.get_valid_access_token(db, _TENANT_ID)

        assert token == "xoxb-plain"
        mock_decrypt.assert_called_once_with(row.access_token, db)

    @pytest.mark.anyio
    async def test_returns_none_when_not_connected(self):
        db = MagicMock()
        with patch("app.services.slack_service.get_integration", return_value=None):
            token = await slack_service.get_valid_access_token(db, _TENANT_ID)
        assert token is None

    @pytest.mark.anyio
    async def test_returns_none_when_row_has_no_token(self):
        db = MagicMock()
        row = MagicMock()
        row.access_token = None
        with patch("app.services.slack_service.get_integration", return_value=row):
            token = await slack_service.get_valid_access_token(db, _TENANT_ID)
        assert token is None

    @pytest.mark.anyio
    async def test_decrypt_failure_propagates(self):
        """Invalid/corrupted ciphertext (e.g. re-encrypted under a rotated key)
        must surface as an error rather than silently return a bad token."""
        db = MagicMock()
        row = MagicMock()
        row.access_token = "gcm1:corrupt"

        with (
            patch("app.services.slack_service.get_integration", return_value=row),
            patch(
                "app.services.slack_service.decrypt_slack_token",
                side_effect=ValueError("Slack token decryption failed"),
            ),
            pytest.raises(ValueError),
        ):
            await slack_service.get_valid_access_token(db, _TENANT_ID)


class TestTenantHasSlackConnected:
    def test_true_when_connected(self):
        db = MagicMock()
        with patch(
            "app.services.slack_service.get_integration", return_value=MagicMock()
        ):
            assert slack_service.tenant_has_slack_connected(db, _TENANT_ID) is True

    def test_false_when_not_connected(self):
        db = MagicMock()
        with patch("app.services.slack_service.get_integration", return_value=None):
            assert slack_service.tenant_has_slack_connected(db, _TENANT_ID) is False


class TestGetIntegrationStatus:
    def test_not_connected_shape(self):
        db = MagicMock()
        with patch("app.services.slack_service.get_integration", return_value=None):
            status = slack_service.get_integration_status(db, _TENANT_ID)

        assert status == {
            "connected": False,
            "connected_at": None,
            "team_name": None,
            "default_channel_id": None,
            "default_channel_name": None,
        }

    def test_connected_shape(self):
        db = MagicMock()
        row = MagicMock()
        row.created_at = datetime.now(timezone.utc)
        row.extra_metadata = {
            "team_name": "Acme Workspace",
            "default_channel_id": "C1",
            "default_channel_name": "general",
        }

        with patch("app.services.slack_service.get_integration", return_value=row):
            status = slack_service.get_integration_status(db, _TENANT_ID)

        assert status["connected"] is True
        assert status["team_name"] == "Acme Workspace"
        assert status["default_channel_id"] == "C1"
        assert status["default_channel_name"] == "general"


class TestSetDefaultChannel:
    def test_sets_channel(self):
        db = MagicMock()
        row = MagicMock()
        row.extra_metadata = {}

        with patch("app.services.slack_service.get_integration", return_value=row):
            slack_service.set_default_channel(db, _TENANT_ID, "C1", "general")

        assert row.extra_metadata["default_channel_id"] == "C1"
        assert row.extra_metadata["default_channel_name"] == "general"
        db.commit.assert_called_once()

    def test_clears_channel_when_both_none(self):
        db = MagicMock()
        row = MagicMock()
        row.extra_metadata = {
            "default_channel_id": "C1",
            "default_channel_name": "general",
        }

        with patch("app.services.slack_service.get_integration", return_value=row):
            slack_service.set_default_channel(db, _TENANT_ID, None, None)

        assert "default_channel_id" not in row.extra_metadata
        assert "default_channel_name" not in row.extra_metadata

    def test_raises_when_not_connected(self):
        db = MagicMock()
        with (
            patch("app.services.slack_service.get_integration", return_value=None),
            pytest.raises(ValueError),
        ):
            slack_service.set_default_channel(db, _TENANT_ID, "C1", "general")


# ── Disconnect ──────────────────────────────────────────────────────────────────


class TestDisconnect:
    @pytest.mark.anyio
    async def test_revokes_and_deletes_row(self):
        db = MagicMock()
        row = MagicMock()
        row.access_token = "gcm1:ciphertext"

        with (
            patch("app.services.slack_service.get_integration", return_value=row),
            patch(
                "app.services.slack_service.decrypt_slack_token",
                return_value="xoxb-plain",
            ),
            patch(
                "app.services.slack_service._request_with_backoff",
                new=AsyncMock(return_value=MagicMock(status_code=200)),
            ) as mock_request,
        ):
            result = await slack_service.disconnect(db, _TENANT_ID)

        assert result is True
        mock_request.assert_awaited_once()
        assert mock_request.call_args[0][0] == "POST"
        assert mock_request.call_args[0][1] == slack_service.REVOKE_URL
        db.delete.assert_called_once_with(row)
        db.commit.assert_called_once()

    @pytest.mark.anyio
    async def test_returns_false_when_not_connected(self):
        db = MagicMock()
        with patch("app.services.slack_service.get_integration", return_value=None):
            result = await slack_service.disconnect(db, _TENANT_ID)
        assert result is False
        db.delete.assert_not_called()

    @pytest.mark.anyio
    async def test_revoke_failure_still_deletes_local_row(self):
        """A Slack outage or already-revoked token during auth.revoke must not
        block the local disconnect (best-effort revoke)."""
        db = MagicMock()
        row = MagicMock()
        row.access_token = "gcm1:ciphertext"

        with (
            patch("app.services.slack_service.get_integration", return_value=row),
            patch(
                "app.services.slack_service.decrypt_slack_token",
                return_value="xoxb-plain",
            ),
            patch(
                "app.services.slack_service._request_with_backoff",
                new=AsyncMock(side_effect=Exception("network error")),
            ),
        ):
            result = await slack_service.disconnect(db, _TENANT_ID)

        assert result is True
        db.delete.assert_called_once_with(row)
        db.commit.assert_called_once()

    @pytest.mark.anyio
    async def test_decrypt_failure_during_revoke_still_deletes_local_row(self):
        db = MagicMock()
        row = MagicMock()
        row.access_token = "gcm1:corrupt"

        with (
            patch("app.services.slack_service.get_integration", return_value=row),
            patch(
                "app.services.slack_service.decrypt_slack_token",
                side_effect=ValueError("Slack token decryption failed"),
            ),
        ):
            result = await slack_service.disconnect(db, _TENANT_ID)

        assert result is True
        db.delete.assert_called_once_with(row)


# ── Channel listing (admin action — raises, does not fail open) ────────────────


class TestListChannels:
    @pytest.mark.anyio
    async def test_returns_channel_shape(self):
        db = MagicMock()
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "ok": True,
            "channels": [
                {"id": "C1", "name": "general"},
                {"id": "C2", "name": "sales"},
            ],
            "response_metadata": {"next_cursor": ""},
        }

        with (
            patch(
                "app.services.slack_service.get_valid_access_token",
                new=AsyncMock(return_value="xoxb-plain"),
            ),
            patch(
                "app.services.slack_service._request_with_backoff",
                new=AsyncMock(return_value=response),
            ),
        ):
            channels = await slack_service.list_channels(db, _TENANT_ID)

        assert channels == [
            {"id": "C1", "name": "general"},
            {"id": "C2", "name": "sales"},
        ]

    @pytest.mark.anyio
    async def test_paginates_across_cursor(self):
        db = MagicMock()
        page1 = MagicMock()
        page1.raise_for_status = MagicMock()
        page1.json.return_value = {
            "ok": True,
            "channels": [{"id": "C1", "name": "general"}],
            "response_metadata": {"next_cursor": "cursor-2"},
        }
        page2 = MagicMock()
        page2.raise_for_status = MagicMock()
        page2.json.return_value = {
            "ok": True,
            "channels": [{"id": "C2", "name": "sales"}],
            "response_metadata": {"next_cursor": ""},
        }

        with (
            patch(
                "app.services.slack_service.get_valid_access_token",
                new=AsyncMock(return_value="xoxb-plain"),
            ),
            patch(
                "app.services.slack_service._request_with_backoff",
                new=AsyncMock(side_effect=[page1, page2]),
            ) as mock_request,
        ):
            channels = await slack_service.list_channels(db, _TENANT_ID)

        assert [c["id"] for c in channels] == ["C1", "C2"]
        assert mock_request.await_count == 2
        assert mock_request.await_args_list[1].kwargs["params"]["cursor"] == "cursor-2"

    @pytest.mark.anyio
    async def test_pagination_stops_at_page_cap(self):
        """Regression test: a workspace with far more channels than the page
        cap must not turn GET /channels into an unbounded blocking fan-out —
        list_channels stops after _MAX_CHANNEL_LIST_PAGES pages even though
        the server keeps returning a next_cursor."""
        db = MagicMock()

        def _page(n: int) -> MagicMock:
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "ok": True,
                "channels": [{"id": f"C{n}", "name": f"chan-{n}"}],
                "response_metadata": {"next_cursor": f"cursor-{n + 1}"},
            }
            return resp

        # Server would keep paginating forever (always returns a next_cursor);
        # only _MAX_CHANNEL_LIST_PAGES requests should ever be made.
        pages = [_page(n) for n in range(1, slack_service._MAX_CHANNEL_LIST_PAGES + 2)]

        with (
            patch(
                "app.services.slack_service.get_valid_access_token",
                new=AsyncMock(return_value="xoxb-plain"),
            ),
            patch(
                "app.services.slack_service._request_with_backoff",
                new=AsyncMock(side_effect=pages),
            ) as mock_request,
        ):
            channels = await slack_service.list_channels(db, _TENANT_ID)

        assert mock_request.await_count == slack_service._MAX_CHANNEL_LIST_PAGES
        assert len(channels) == slack_service._MAX_CHANNEL_LIST_PAGES

    @pytest.mark.anyio
    async def test_raises_when_not_connected(self):
        db = MagicMock()
        with (
            patch(
                "app.services.slack_service.get_valid_access_token",
                new=AsyncMock(return_value=None),
            ),
            pytest.raises(ValueError, match="not connected"),
        ):
            await slack_service.list_channels(db, _TENANT_ID)

    @pytest.mark.anyio
    async def test_raises_on_ok_false(self):
        db = MagicMock()
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {"ok": False, "error": "invalid_auth"}

        with (
            patch(
                "app.services.slack_service.get_valid_access_token",
                new=AsyncMock(return_value="xoxb-plain"),
            ),
            patch(
                "app.services.slack_service._request_with_backoff",
                new=AsyncMock(return_value=response),
            ),
            pytest.raises(ValueError, match="invalid_auth"),
        ):
            await slack_service.list_channels(db, _TENANT_ID)

    @pytest.mark.anyio
    async def test_raises_on_http_error_does_not_fail_open(self):
        """Admin-facing settings action: unlike send_call_summary, a Slack
        outage here must raise, not silently return an empty list."""
        import httpx

        db = MagicMock()
        response = MagicMock()
        response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "500", request=MagicMock(), response=MagicMock(status_code=500)
            )
        )

        with (
            patch(
                "app.services.slack_service.get_valid_access_token",
                new=AsyncMock(return_value="xoxb-plain"),
            ),
            patch(
                "app.services.slack_service._request_with_backoff",
                new=AsyncMock(return_value=response),
            ),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await slack_service.list_channels(db, _TENANT_ID)


# ── Post-call summary send ──────────────────────────────────────────────────────


def _call_session(
    *,
    call_flow_id=uuid.uuid4(),
    tenant_id=_TENANT_ID,
    phone="+15550001111",
    status="completed",
    duration=125,
    metadata=None,
    transcript_summary=None,
):
    cs = MagicMock()
    cs.id = _SESSION_ID
    cs.tenant_id = tenant_id
    cs.call_flow_id = call_flow_id
    cs.customer_phone_number = phone
    cs.status = status
    cs.duration = duration
    cs.call_metadata = metadata if metadata is not None else {}
    cs.transcript_summary = transcript_summary
    return cs


def _call_flow(*, enabled=True, channel_id=None, channel_name=None):
    flow = MagicMock()
    flow.slack_summary_enabled = enabled
    flow.slack_channel_id = channel_id
    flow.slack_channel_name = channel_name
    return flow


def _integration(*, default_channel_id=None):
    row = MagicMock()
    row.access_token = "gcm1:ciphertext"
    row.extra_metadata = (
        {"default_channel_id": default_channel_id} if default_channel_id else {}
    )
    return row


def _mock_query_chain(db, results_by_model):
    """db.query(Model).filter(...).first() -> results_by_model[Model]"""

    def _query(model):
        q = MagicMock()
        q.filter.return_value = q
        q.first.return_value = results_by_model.get(model)
        return q

    db.query.side_effect = _query


class TestSendCallSummary:
    def _setup_db(self, db, *, call_session, call_flow):
        from app.models.call_flow import CallFlow
        from app.models.call_session import CallSession

        _mock_query_chain(db, {CallSession: call_session, CallFlow: call_flow})

    @pytest.mark.anyio
    async def test_posts_correct_phone_summary_and_sentiment(self):
        db = MagicMock()
        call_flow = _call_flow(enabled=True)
        call_session = _call_session(
            call_flow_id=call_flow,
            phone="+15550001111",
            metadata={
                "llm_call_analysis": {
                    "analysis": {
                        "summary": "Caller wants a demo.",
                        "sentiment": "positive",
                    }
                }
            },
        )
        self._setup_db(db, call_session=call_session, call_flow=call_flow)
        integration = _integration(default_channel_id="C1")

        with (
            patch(
                "app.services.slack_service.get_integration", return_value=integration
            ),
            patch(
                "app.services.slack_service.decrypt_slack_token",
                return_value="xoxb-plain",
            ),
            patch(
                "app.services.slack_service._post_message", new=AsyncMock()
            ) as mock_post,
        ):
            await slack_service._send_call_summary_async(db, _SESSION_ID)

        mock_post.assert_awaited_once()
        access_token, channel, text, blocks = mock_post.call_args[0]
        assert access_token == "xoxb-plain"
        assert channel == "C1"
        assert "+15550001111" in text
        summary_block = next(
            b for b in blocks if b.get("type") == "section" and "Summary" in str(b)
        )
        assert "Caller wants a demo." in str(summary_block)
        sentiment_field = next(
            f for f in blocks[1]["fields"] if "Sentiment" in f["text"]
        )
        assert "positive" in sentiment_field["text"]

    @pytest.mark.anyio
    async def test_call_flow_channel_override_wins_over_workspace_default(self):
        db = MagicMock()
        call_flow = _call_flow(
            enabled=True, channel_id="C-OVERRIDE", channel_name="sales"
        )
        call_session = _call_session(call_flow_id=call_flow)
        self._setup_db(db, call_session=call_session, call_flow=call_flow)
        integration = _integration(default_channel_id="C-DEFAULT")

        with (
            patch(
                "app.services.slack_service.get_integration", return_value=integration
            ),
            patch(
                "app.services.slack_service.decrypt_slack_token",
                return_value="xoxb-plain",
            ),
            patch(
                "app.services.slack_service._post_message", new=AsyncMock()
            ) as mock_post,
        ):
            await slack_service._send_call_summary_async(db, _SESSION_ID)

        channel = mock_post.call_args[0][1]
        assert channel == "C-OVERRIDE"

    @pytest.mark.anyio
    async def test_falls_back_to_workspace_default_channel(self):
        db = MagicMock()
        call_flow = _call_flow(enabled=True, channel_id=None, channel_name=None)
        call_session = _call_session(call_flow_id=call_flow)
        self._setup_db(db, call_session=call_session, call_flow=call_flow)
        integration = _integration(default_channel_id="C-DEFAULT")

        with (
            patch(
                "app.services.slack_service.get_integration", return_value=integration
            ),
            patch(
                "app.services.slack_service.decrypt_slack_token",
                return_value="xoxb-plain",
            ),
            patch(
                "app.services.slack_service._post_message", new=AsyncMock()
            ) as mock_post,
        ):
            await slack_service._send_call_summary_async(db, _SESSION_ID)

        channel = mock_post.call_args[0][1]
        assert channel == "C-DEFAULT"

    @pytest.mark.anyio
    async def test_no_channel_configured_skips_without_raising(self):
        """Neither a call-flow override nor a workspace default configured —
        must log and skip cleanly, not crash or post to an empty channel."""
        db = MagicMock()
        call_flow = _call_flow(enabled=True, channel_id=None, channel_name=None)
        call_session = _call_session(call_flow_id=call_flow)
        self._setup_db(db, call_session=call_session, call_flow=call_flow)
        integration = _integration(default_channel_id=None)

        with (
            patch(
                "app.services.slack_service.get_integration", return_value=integration
            ),
            patch(
                "app.services.slack_service._post_message", new=AsyncMock()
            ) as mock_post,
        ):
            await slack_service._send_call_summary_async(db, _SESSION_ID)

        mock_post.assert_not_called()

    @pytest.mark.anyio
    async def test_no_call_session_returns_quietly(self):
        db = MagicMock()
        from app.models.call_flow import CallFlow
        from app.models.call_session import CallSession

        _mock_query_chain(db, {CallSession: None, CallFlow: None})

        with patch(
            "app.services.slack_service._post_message", new=AsyncMock()
        ) as mock_post:
            await slack_service._send_call_summary_async(db, _SESSION_ID)

        mock_post.assert_not_called()

    @pytest.mark.anyio
    async def test_no_call_flow_id_returns_quietly(self):
        db = MagicMock()
        call_session = _call_session(call_flow_id=None)
        from app.models.call_flow import CallFlow
        from app.models.call_session import CallSession

        _mock_query_chain(db, {CallSession: call_session, CallFlow: None})

        with patch(
            "app.services.slack_service._post_message", new=AsyncMock()
        ) as mock_post:
            await slack_service._send_call_summary_async(db, _SESSION_ID)

        mock_post.assert_not_called()

    @pytest.mark.anyio
    async def test_call_flow_disabled_returns_quietly(self):
        db = MagicMock()
        call_flow = _call_flow(enabled=False)
        call_session = _call_session(call_flow_id=call_flow)
        self._setup_db(db, call_session=call_session, call_flow=call_flow)

        with patch(
            "app.services.slack_service._post_message", new=AsyncMock()
        ) as mock_post:
            await slack_service._send_call_summary_async(db, _SESSION_ID)

        mock_post.assert_not_called()

    @pytest.mark.anyio
    async def test_not_connected_returns_quietly(self):
        db = MagicMock()
        call_flow = _call_flow(enabled=True)
        call_session = _call_session(call_flow_id=call_flow)
        self._setup_db(db, call_session=call_session, call_flow=call_flow)

        with (
            patch("app.services.slack_service.get_integration", return_value=None),
            patch(
                "app.services.slack_service._post_message", new=AsyncMock()
            ) as mock_post,
        ):
            await slack_service._send_call_summary_async(db, _SESSION_ID)

        mock_post.assert_not_called()

    @pytest.mark.anyio
    async def test_decrypt_failure_fails_open(self):
        db = MagicMock()
        call_flow = _call_flow(enabled=True)
        call_session = _call_session(call_flow_id=call_flow)
        self._setup_db(db, call_session=call_session, call_flow=call_flow)
        integration = _integration(default_channel_id="C1")

        with (
            patch(
                "app.services.slack_service.get_integration", return_value=integration
            ),
            patch(
                "app.services.slack_service.decrypt_slack_token",
                side_effect=ValueError("Slack token decryption failed"),
            ),
            patch(
                "app.services.slack_service._post_message", new=AsyncMock()
            ) as mock_post,
        ):
            await slack_service._send_call_summary_async(
                db, _SESSION_ID
            )  # must not raise

        mock_post.assert_not_called()

    @pytest.mark.anyio
    async def test_post_message_failure_fails_open_via_sync_entrypoint(self):
        """send_call_summary (the sync entrypoint actually wired into the call-time
        hook) must never raise even when everything downstream fails."""
        db = MagicMock()

        with (
            patch("app.services.slack_service.SessionLocal", return_value=db),
            patch(
                "app.services.slack_service._send_call_summary_async",
                new=AsyncMock(
                    side_effect=Exception("Slack chat.postMessage failed: rate_limited")
                ),
            ),
        ):
            slack_service.send_call_summary(_SESSION_ID)  # must not raise

        db.close.assert_called_once()

    @pytest.mark.anyio
    async def test_lazily_computes_analysis_when_not_cached(self):
        db = MagicMock()
        call_flow = _call_flow(enabled=True)
        call_session = _call_session(call_flow_id=call_flow, metadata={})
        self._setup_db(db, call_session=call_session, call_flow=call_flow)
        integration = _integration(default_channel_id="C1")

        def _fake_ensure(db_arg, cs_arg, raise_on_no_transcript=False):
            cs_arg.call_metadata = {
                "llm_call_analysis": {
                    "analysis": {"summary": "Lazy summary.", "sentiment": "neutral"}
                }
            }

        with (
            patch(
                "app.services.slack_service.get_integration", return_value=integration
            ),
            patch(
                "app.services.slack_service.decrypt_slack_token",
                return_value="xoxb-plain",
            ),
            patch(
                "app.services.voice_analysis_service.ensure_call_summary_locked",
                side_effect=_fake_ensure,
            ) as mock_ensure,
            patch(
                "app.services.slack_service._post_message", new=AsyncMock()
            ) as mock_post,
        ):
            await slack_service._send_call_summary_async(db, _SESSION_ID)

        mock_ensure.assert_called_once()
        text_blocks = mock_post.call_args[0][3]
        summary_block = next(
            b for b in text_blocks if b.get("type") == "section" and "Summary" in str(b)
        )
        assert "Lazy summary." in str(summary_block)

    @pytest.mark.anyio
    async def test_analysis_generation_failure_fails_open_and_uses_fallback_summary(
        self,
    ):
        db = MagicMock()
        call_flow = _call_flow(enabled=True)
        call_session = _call_session(call_flow_id=call_flow, metadata={})
        self._setup_db(db, call_session=call_session, call_flow=call_flow)
        integration = _integration(default_channel_id="C1")

        with (
            patch(
                "app.services.slack_service.get_integration", return_value=integration
            ),
            patch(
                "app.services.slack_service.decrypt_slack_token",
                return_value="xoxb-plain",
            ),
            patch(
                "app.services.voice_analysis_service.ensure_call_summary_locked",
                side_effect=Exception("advisory lock contention"),
            ),
            patch(
                "app.services.slack_service._post_message", new=AsyncMock()
            ) as mock_post,
        ):
            await slack_service._send_call_summary_async(
                db, _SESSION_ID
            )  # must not raise

        mock_post.assert_awaited_once()
        text_blocks = mock_post.call_args[0][3]
        summary_block = next(
            b for b in text_blocks if b.get("type") == "section" and "Summary" in str(b)
        )
        assert "No summary available." in str(summary_block)

    @pytest.mark.anyio
    async def test_uses_transcript_summary_over_llm_analysis_when_present(self):
        db = MagicMock()
        call_flow = _call_flow(enabled=True)
        call_session = _call_session(
            call_flow_id=call_flow,
            transcript_summary="Explicit transcript summary.",
            metadata={
                "llm_call_analysis": {"analysis": {"summary": "Analysis summary."}}
            },
        )
        self._setup_db(db, call_session=call_session, call_flow=call_flow)
        integration = _integration(default_channel_id="C1")

        with (
            patch(
                "app.services.slack_service.get_integration", return_value=integration
            ),
            patch(
                "app.services.slack_service.decrypt_slack_token",
                return_value="xoxb-plain",
            ),
            patch(
                "app.services.slack_service._post_message", new=AsyncMock()
            ) as mock_post,
        ):
            await slack_service._send_call_summary_async(db, _SESSION_ID)

        text_blocks = mock_post.call_args[0][3]
        summary_block = next(
            b for b in text_blocks if b.get("type") == "section" and "Summary" in str(b)
        )
        assert "Explicit transcript summary." in str(summary_block)
        assert "Analysis summary." not in str(summary_block)


class TestScheduleSlackSummary:
    def test_dispatches_via_thread_when_no_running_loop(self):
        with (
            patch(
                "app.services.slack_service.asyncio.get_running_loop",
                side_effect=RuntimeError("no running loop"),
            ),
            patch("app.services.slack_service.threading.Thread") as mock_thread_cls,
        ):
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            slack_service.schedule_slack_summary(_SESSION_ID)

        mock_thread_cls.assert_called_once_with(
            target=slack_service.send_call_summary, args=(_SESSION_ID,), daemon=True
        )
        mock_thread.start.assert_called_once()

    @pytest.mark.anyio
    async def test_dispatches_via_create_task_when_loop_running(self):
        with patch(
            "app.services.slack_service.asyncio.create_task"
        ) as mock_create_task:
            slack_service.schedule_slack_summary(_SESSION_ID)

        mock_create_task.assert_called_once()


# ── Multi-tenant isolation ───────────────────────────────────────────────────────


class TestTenantIsolation:
    def test_get_integration_filters_by_tenant_and_provider(self):
        """The WorkspaceIntegration query must filter by both workspace_id AND
        provider — never return tenant B's Slack row for tenant A."""
        db = MagicMock()
        query = MagicMock()
        db.query.return_value = query
        query.filter.return_value = query
        query.first.return_value = MagicMock()

        slack_service.get_integration(db, _TENANT_ID)

        filter_args = query.filter.call_args[0]
        # Two filter clauses passed: workspace_id == tenant_id, provider == "slack"
        assert len(filter_args) == 2

    def test_get_integration_never_returns_other_tenants_row(self, db):
        """Real-DB proof: with two tenants each holding a Slack WorkspaceIntegration
        row, tenant A's lookup must return only tenant A's row/token, never tenant B's,
        even though both rows share provider="slack"."""
        from app.models.tenant import Tenant
        from app.models.workspace_integration import WorkspaceIntegration

        tenant_a = Tenant(
            name=f"SlackIsoA-{uuid.uuid4().hex[:8]}",
            schema_name=f"slack_iso_a_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        tenant_b = Tenant(
            name=f"SlackIsoB-{uuid.uuid4().hex[:8]}",
            schema_name=f"slack_iso_b_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add_all([tenant_a, tenant_b])
        db.commit()
        db.refresh(tenant_a)
        db.refresh(tenant_b)

        row_a = WorkspaceIntegration(
            workspace_id=tenant_a.id,
            provider="slack",
            access_token="gcm1:tenant-a-ciphertext",
            extra_metadata={"team_id": "TEAMA", "default_channel_id": "C_A"},
        )
        row_b = WorkspaceIntegration(
            workspace_id=tenant_b.id,
            provider="slack",
            access_token="gcm1:tenant-b-ciphertext",
            extra_metadata={"team_id": "TEAMB", "default_channel_id": "C_B"},
        )
        db.add_all([row_a, row_b])
        db.commit()

        found_for_a = slack_service.get_integration(db, tenant_a.id)
        found_for_b = slack_service.get_integration(db, tenant_b.id)

        assert found_for_a is not None
        assert found_for_a.id == row_a.id
        assert found_for_a.access_token == "gcm1:tenant-a-ciphertext"
        assert found_for_a.extra_metadata["default_channel_id"] == "C_A"

        assert found_for_b is not None
        assert found_for_b.id == row_b.id
        assert found_for_b.access_token == "gcm1:tenant-b-ciphertext"

        # Neither lookup ever leaks the other tenant's row.
        assert found_for_a.id != found_for_b.id
        assert found_for_a.workspace_id == tenant_a.id
        assert found_for_b.workspace_id == tenant_b.id

        db.delete(row_a)
        db.delete(row_b)
        db.delete(tenant_a)
        db.delete(tenant_b)
        db.commit()

    def test_list_channels_uses_tenant_specific_token(self):
        """Confirms list_channels resolves the access token via the tenant-scoped
        get_valid_access_token rather than any global/shared credential."""
        db = MagicMock()

        async def _fake_get_valid_access_token(db_arg, tenant_id_arg):
            assert tenant_id_arg == _TENANT_ID
            return "tenant-a-token"

        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.json.return_value = {
            "ok": True,
            "channels": [],
            "response_metadata": {},
        }

        async def _run():
            with (
                patch(
                    "app.services.slack_service.get_valid_access_token",
                    side_effect=_fake_get_valid_access_token,
                ),
                patch(
                    "app.services.slack_service._request_with_backoff",
                    new=AsyncMock(return_value=response),
                ) as mock_request,
            ):
                await slack_service.list_channels(db, _TENANT_ID)
            assert (
                mock_request.call_args.kwargs["headers"]["Authorization"]
                == "Bearer tenant-a-token"
            )

        import asyncio

        asyncio.run(_run())


# ── 429 backoff (shared helper) ──────────────────────────────────────────────────


class TestBackoff:
    @pytest.mark.anyio
    async def test_retries_on_429_then_succeeds(self):
        responses = [
            MagicMock(status_code=429, headers={}),
            MagicMock(status_code=200, headers={}),
        ]

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=responses)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
        ):
            response = await slack_service._request_with_backoff(
                "GET", "https://slack.com/api/x"
            )

        assert response.status_code == 200
        mock_sleep.assert_awaited_once()
        assert mock_sleep.call_args[0][0] == 1.0

    @pytest.mark.anyio
    async def test_honors_retry_after_header(self):
        responses = [
            MagicMock(status_code=429, headers={"Retry-After": "3"}),
            MagicMock(status_code=200, headers={}),
        ]
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=responses)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
        ):
            await slack_service._request_with_backoff("GET", "https://slack.com/api/x")

        assert mock_sleep.call_args[0][0] == 3.0

    @pytest.mark.anyio
    async def test_gives_up_after_max_retries(self):
        always_429 = MagicMock(status_code=429, headers={})
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=always_429)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("asyncio.sleep", new=AsyncMock()),
        ):
            response = await slack_service._request_with_backoff(
                "GET", "https://slack.com/api/x"
            )

        assert response.status_code == 429
        assert mock_client.request.await_count == slack_service._MAX_RETRIES + 1


# ── Token encryption round trip ───────────────────────────────────────────────


class TestSlackTokenEncryption:
    def test_roundtrip(self):
        from app.core.db_encryption import decrypt_slack_token, encrypt_slack_token

        db = MagicMock()
        ciphertext = encrypt_slack_token("xoxb-secret-token", db)
        assert ciphertext.startswith("slack_gcm1:")
        assert decrypt_slack_token(ciphertext, db) == "xoxb-secret-token"

    def test_decrypt_unrecognized_format_raises(self):
        from app.core.db_encryption import decrypt_slack_token

        db = MagicMock()
        with pytest.raises(ValueError, match="Unrecognized Slack token"):
            decrypt_slack_token("not-a-gcm-ciphertext", db)

    def test_decrypt_empty_raises(self):
        from app.core.db_encryption import decrypt_slack_token

        db = MagicMock()
        with pytest.raises(ValueError, match="ciphertext is empty"):
            decrypt_slack_token("", db)

    def test_decrypt_tampered_ciphertext_raises(self):
        from app.core.db_encryption import decrypt_slack_token, encrypt_slack_token

        db = MagicMock()
        ciphertext = encrypt_slack_token("xoxb-secret-token", db)
        tampered = (
            ciphertext[:-4] + ("A" if ciphertext[-4] != "A" else "B") + ciphertext[-3:]
        )
        with pytest.raises(ValueError, match="decryption failed"):
            decrypt_slack_token(tampered, db)
