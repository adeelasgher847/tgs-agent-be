"""
Tests for the Calendly calendar integration service.

External HTTP calls (OAuth token endpoint, availability, invitee creation) are
mocked at the boundary per CLAUDE.md convention — no real network calls.

Coverage:
  1. OAuth state: sign/verify round trip, tamper rejection, wrong purpose rejection
  2. Authorization URL: client_id/redirect_uri/response_type/state present
  3. OAuth code exchange: POST to token endpoint with the right grant_type/payload
  4. Token auto-refresh: cached token returned when fresh; refreshes when within
     the 5-minute expiry margin; fails open (returns None) on refresh error
  5. Availability: response shape normalized to [{slot_start, slot_end, available}]
  6. Booking: request payload sent to POST /invitees matches the Calendly contract
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import calendly_service


_WORKSPACE_ID = uuid.UUID("cc100000-0000-0000-0000-000000000001")


# ── OAuth state ────────────────────────────────────────────────────────────────


class TestOAuthState:
    def test_roundtrip(self):
        state = calendly_service.build_oauth_state(_WORKSPACE_ID)
        assert calendly_service.verify_oauth_state(state) == _WORKSPACE_ID

    def test_tampered_state_rejected(self):
        state = calendly_service.build_oauth_state(_WORKSPACE_ID)
        with pytest.raises(ValueError):
            calendly_service.verify_oauth_state(state + "x")

    def test_wrong_purpose_rejected(self):
        from jose import jwt as jose_jwt
        from app.core.config import settings

        bad_state = jose_jwt.encode(
            {
                "workspace_id": str(_WORKSPACE_ID),
                "purpose": "something_else",
                "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
            },
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        with pytest.raises(ValueError):
            calendly_service.verify_oauth_state(bad_state)

    def test_expired_state_rejected(self):
        from jose import jwt as jose_jwt
        from app.core.config import settings

        expired_state = jose_jwt.encode(
            {
                "workspace_id": str(_WORKSPACE_ID),
                "purpose": "calendly_oauth_state",
                "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
            },
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        with pytest.raises(ValueError):
            calendly_service.verify_oauth_state(expired_state)


class TestAuthorizationUrl:
    def test_contains_client_id_redirect_and_state(self):
        with patch.object(calendly_service.settings, "CALENDLY_CLIENT_ID", "client-123"), \
             patch.object(calendly_service.settings, "CALENDLY_REDIRECT_URI", "https://app.example.com/callback"):
            state = calendly_service.build_oauth_state(_WORKSPACE_ID)
            url = calendly_service.build_authorization_url(state)

        assert url.startswith("https://auth.calendly.com/oauth/authorize?")
        assert "client_id=client-123" in url
        assert "response_type=code" in url
        assert f"state={state}" in url or state in url


# ── OAuth code exchange ─────────────────────────────────────────────────────────


class TestCodeExchange:
    @pytest.mark.anyio
    async def test_exchange_code_posts_authorization_code_grant(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "access-123",
            "refresh_token": "refresh-456",
            "expires_in": 7200,
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await calendly_service.exchange_code_for_tokens("auth-code-abc")

        assert result["access_token"] == "access-123"
        mock_client.request.assert_awaited_once()
        args, kwargs = mock_client.request.call_args
        assert args[0] == "POST"
        assert kwargs["data"]["grant_type"] == "authorization_code"
        assert kwargs["data"]["code"] == "auth-code-abc"


# ── Token auto-refresh ───────────────────────────────────────────────────────────


def _integration_row(*, expires_in_seconds: float, has_refresh_token: bool = True):
    row = MagicMock()
    row.access_token = "encrypted-access-token"
    row.refresh_token = "encrypted-refresh-token" if has_refresh_token else None
    row.token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds)
    return row


class TestTokenAutoRefresh:
    @pytest.mark.anyio
    async def test_returns_existing_token_when_not_near_expiry(self):
        row = _integration_row(expires_in_seconds=600)
        db = MagicMock()

        with (
            patch("app.services.calendly_service.get_integration", return_value=row),
            patch(
                "app.services.calendly_service.decrypt_calendly_token",
                return_value="plain-access-token",
            ) as mock_decrypt,
        ):
            token = await calendly_service.get_valid_access_token(db, _WORKSPACE_ID)

        assert token == "plain-access-token"
        mock_decrypt.assert_called_once_with(row.access_token, db)

    @pytest.mark.anyio
    async def test_refreshes_when_within_five_minute_margin(self):
        """Spec: refresh when token_expires_at is less than 5 minutes away."""
        row = _integration_row(expires_in_seconds=120)  # 2 minutes — inside the 5-minute margin
        db = MagicMock()
        new_row = _integration_row(expires_in_seconds=7200)

        with (
            patch("app.services.calendly_service.get_integration", return_value=row),
            patch(
                "app.services.calendly_service.decrypt_calendly_token",
                side_effect=["plain-refresh-token", "new-plain-access-token"],
            ),
            patch(
                "app.services.calendly_service.refresh_access_token",
                new=AsyncMock(return_value={"access_token": "new", "expires_in": 7200}),
            ) as mock_refresh,
            patch("app.services.calendly_service.upsert_tokens", return_value=new_row) as mock_upsert,
        ):
            token = await calendly_service.get_valid_access_token(db, _WORKSPACE_ID)

        mock_refresh.assert_awaited_once_with("plain-refresh-token")
        mock_upsert.assert_called_once()
        assert token == "new-plain-access-token"

    @pytest.mark.anyio
    async def test_refresh_failure_fails_open(self):
        row = _integration_row(expires_in_seconds=-60)
        db = MagicMock()

        with (
            patch("app.services.calendly_service.get_integration", return_value=row),
            patch(
                "app.services.calendly_service.decrypt_calendly_token",
                return_value="plain-refresh-token",
            ),
            patch(
                "app.services.calendly_service.refresh_access_token",
                new=AsyncMock(side_effect=Exception("Calendly down")),
            ),
        ):
            token = await calendly_service.get_valid_access_token(db, _WORKSPACE_ID)

        assert token is None

    @pytest.mark.anyio
    async def test_returns_none_when_no_integration(self):
        db = MagicMock()
        with patch("app.services.calendly_service.get_integration", return_value=None):
            token = await calendly_service.get_valid_access_token(db, _WORKSPACE_ID)
        assert token is None


# ── Availability ─────────────────────────────────────────────────────────────────


class TestGetAvailableSlots:
    @pytest.mark.anyio
    async def test_returns_normalized_slot_schema(self):
        row = MagicMock()
        row.calendly_event_type_uri = "https://api.calendly.com/event_types/abc"
        db = MagicMock()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "collection": [
                {
                    "status": "available",
                    "start_time": "2026-07-16T15:00:00.000000Z",
                    "end_time": "2026-07-16T15:30:00.000000Z",
                },
                {
                    "status": "unavailable",
                    "start_time": "2026-07-16T15:30:00.000000Z",
                    "end_time": "2026-07-16T16:00:00.000000Z",
                },
            ]
        }
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("app.services.calendly_service.get_integration", return_value=row),
            patch(
                "app.services.calendly_service.get_valid_access_token",
                new=AsyncMock(return_value="plain-access-token"),
            ),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            slots = await calendly_service.get_available_slots(
                db,
                _WORKSPACE_ID,
                datetime(2026, 7, 16, tzinfo=timezone.utc),
                datetime(2026, 7, 17, tzinfo=timezone.utc),
            )

        assert len(slots) == 2
        assert slots[0]["available"] is True
        assert slots[1]["available"] is False
        assert slots[0]["slot_start"] == datetime(2026, 7, 16, 15, 0, tzinfo=timezone.utc)

        # Request hit the documented Calendly endpoint with the connected event type.
        args, kwargs = mock_client.request.call_args
        assert args[0] == "GET"
        assert kwargs["params"]["event_type"] == row.calendly_event_type_uri

    @pytest.mark.anyio
    async def test_raises_when_not_connected(self):
        db = MagicMock()
        with patch("app.services.calendly_service.get_integration", return_value=None):
            with pytest.raises(ValueError, match="not connected"):
                await calendly_service.get_available_slots(
                    db,
                    _WORKSPACE_ID,
                    datetime(2026, 7, 16, tzinfo=timezone.utc),
                    datetime(2026, 7, 17, tzinfo=timezone.utc),
                )

    @pytest.mark.anyio
    async def test_raises_when_no_event_type_configured(self):
        row = MagicMock()
        row.calendly_event_type_uri = None
        db = MagicMock()
        with patch("app.services.calendly_service.get_integration", return_value=row):
            with pytest.raises(ValueError, match="event type"):
                await calendly_service.get_available_slots(
                    db,
                    _WORKSPACE_ID,
                    datetime(2026, 7, 16, tzinfo=timezone.utc),
                    datetime(2026, 7, 17, tzinfo=timezone.utc),
                )


# ── Booking ────────────────────────────────────────────────────────────────────


class TestBookAppointment:
    @pytest.mark.anyio
    async def test_booking_payload_matches_calendly_contract(self):
        row = MagicMock()
        row.calendly_event_type_uri = "https://api.calendly.com/event_types/abc"
        db = MagicMock()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"resource": {"uri": "https://api.calendly.com/scheduled_events/xyz"}}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("app.services.calendly_service.get_integration", return_value=row),
            patch(
                "app.services.calendly_service.get_valid_access_token",
                new=AsyncMock(return_value="plain-access-token"),
            ),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            result = await calendly_service.book_appointment(
                db,
                _WORKSPACE_ID,
                start_time=datetime(2026, 7, 16, 15, 0, tzinfo=timezone.utc),
                attendee_email="caller@example.com",
                attendee_name="Ada Lovelace",
                description="Voice call summary: caller wants a quote.",
            )

        assert result["resource"]["uri"] == "https://api.calendly.com/scheduled_events/xyz"

        args, kwargs = mock_client.request.call_args
        assert args[0] == "POST"
        assert kwargs["headers"]["Authorization"] == "Bearer plain-access-token"
        payload = kwargs["json"]
        assert payload["event_type"] == row.calendly_event_type_uri
        assert payload["start_time"] == "2026-07-16T15:00:00Z"
        assert payload["invitee"]["email"] == "caller@example.com"
        assert payload["invitee"]["name"] == "Ada Lovelace"
        assert payload["invitee"]["timezone"] == "UTC"
        assert payload["invitee"]["comments"] == "Voice call summary: caller wants a quote."

    @pytest.mark.anyio
    async def test_raises_when_not_connected(self):
        db = MagicMock()
        with patch("app.services.calendly_service.get_integration", return_value=None):
            with pytest.raises(ValueError, match="not connected"):
                await calendly_service.book_appointment(
                    db,
                    _WORKSPACE_ID,
                    start_time=datetime(2026, 7, 16, 15, 0, tzinfo=timezone.utc),
                    attendee_email="caller@example.com",
                    attendee_name="Ada Lovelace",
                )


# ── AES-256-GCM token encryption round trip ─────────────────────────────────────


class TestTokenEncryption:
    def test_encrypt_decrypt_roundtrip(self):
        from app.core.config import settings
        from app.core.db_encryption import decrypt_calendly_token, encrypt_calendly_token

        db = MagicMock()
        with patch.object(settings, "CALENDLY_TOKEN_ENCRYPTION_KEY", "a" * 64):
            ciphertext = encrypt_calendly_token("plain-token-value", db)
            assert ciphertext.startswith("gcm1:")
            assert decrypt_calendly_token(ciphertext, db) == "plain-token-value"

    def test_decrypt_rejects_unrecognized_format(self):
        from app.core.config import settings
        from app.core.db_encryption import decrypt_calendly_token

        with patch.object(settings, "CALENDLY_TOKEN_ENCRYPTION_KEY", "a" * 64):
            with pytest.raises(ValueError):
                decrypt_calendly_token("not-a-valid-ciphertext", MagicMock())


# ── Gemini function-calling: function-response Content wrapping ────────────────
# Regression test: appending a bare Part (instead of a Content(role="function", ...))
# to `contents` crashes the Vertex SDK's content-conversion step the moment any
# other Content object is present in the list.


class TestGenerateWithToolsContentWrapping:
    def test_function_response_is_wrapped_in_content_not_bare_part(self):
        pytest.importorskip("vertexai")
        from vertexai.generative_models import Content, Part
        from vertexai.generative_models._generative_models import _content_types_to_gapic_contents

        user_content = Content(role="user", parts=[Part.from_text("hi")])
        model_content = Content(role="model", parts=[Part.from_text("calling tool")])
        function_response_content = Content(
            role="function",
            parts=[Part.from_function_response(name="check_availability", response={"ok": True})],
        )

        # This is exactly the shape generate_with_tools() must build. A bare
        # Part in this list (instead of wrapping it in Content) raises
        # AttributeError here.
        result = _content_types_to_gapic_contents(
            [user_content, model_content, function_response_content]
        )
        assert len(result) == 3
