"""
Tests for HubSpot CRM integration service.

External HTTP calls (OAuth token endpoint, CRM Search API, Engagements/Calls API) are
mocked at the boundary (`_request_with_backoff`) per CLAUDE.md convention.

Coverage:
  1.  OAuth state: sign/verify round trip, tamper rejection, wrong purpose rejection
  2.  Authorization URL: client_id/scope/state present
  3.  Token refresh: returns cached token when fresh; refreshes when expired;
      returns None when no refresh_token; fails open on refresh error
  4.  Contact lookup: correct {id, name, email, company, last_interaction_date} shape;
      Redis cache hit/miss/store; fails open on HubSpot error
  5.  CRM context block: cached on call_metadata after first fetch; fails open
  6.  Post-call write-back: creates a Call engagement with the Gemini summary
  7.  Disconnect: revokes + deletes; returns False when not connected
  8.  429 backoff: retries with exponential delay, honors Retry-After
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import hubspot_service


_TENANT_ID = uuid.UUID("aa100000-0000-0000-0000-000000000001")
_SESSION_ID = uuid.UUID("bb100000-0000-0000-0000-000000000002")


# ── OAuth state ────────────────────────────────────────────────────────────────


class TestOAuthState:
    def test_roundtrip(self):
        state = hubspot_service.build_oauth_state(_TENANT_ID)
        assert hubspot_service.verify_oauth_state(state) == _TENANT_ID

    def test_tampered_state_rejected(self):
        state = hubspot_service.build_oauth_state(_TENANT_ID)
        with pytest.raises(ValueError):
            hubspot_service.verify_oauth_state(state + "x")

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
            hubspot_service.verify_oauth_state(bad_state)

    def test_expired_state_rejected(self):
        from jose import jwt as jose_jwt
        from app.core.config import settings

        expired_state = jose_jwt.encode(
            {
                "tenant_id": str(_TENANT_ID),
                "purpose": "hubspot_oauth_state",
                "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
            },
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        with pytest.raises(ValueError):
            hubspot_service.verify_oauth_state(expired_state)


class TestAuthorizationUrl:
    def test_contains_client_id_scope_and_state(self):
        with patch(
            "app.services.hubspot_service.get_hubspot_oauth_credentials",
            return_value=("client-123", "secret-456"),
        ):
            state = hubspot_service.build_oauth_state(_TENANT_ID)
            url = hubspot_service.build_authorization_url(state)

        assert url.startswith(hubspot_service.AUTHORIZE_URL)
        assert "client_id=client-123" in url
        assert "crm.objects.contacts.read" in url
        assert "crm.objects.contacts.write" in url
        assert f"state={state}" in url


# ── Token refresh ──────────────────────────────────────────────────────────────


def _integration_row(*, expires_in_seconds: float, has_refresh_token: bool = True):
    row = MagicMock()
    row.access_token = "encrypted-access-token"
    row.refresh_token = "encrypted-refresh-token" if has_refresh_token else None
    row.token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds)
    return row


class TestTokenRefresh:
    @pytest.mark.anyio
    async def test_returns_existing_token_when_not_expired(self):
        row = _integration_row(expires_in_seconds=600)
        db = MagicMock()

        with (
            patch("app.services.hubspot_service.get_integration", return_value=row),
            patch(
                "app.services.hubspot_service.decrypt_hubspot_token",
                return_value="plain-access-token",
            ) as mock_decrypt,
        ):
            token = await hubspot_service.get_valid_access_token(db, _TENANT_ID)

        assert token == "plain-access-token"
        mock_decrypt.assert_called_once_with(row.access_token, db)

    @pytest.mark.anyio
    async def test_refreshes_when_expired(self):
        row = _integration_row(expires_in_seconds=-60)
        db = MagicMock()
        new_row = _integration_row(expires_in_seconds=1800)

        with (
            patch("app.services.hubspot_service.get_integration", return_value=row),
            patch(
                "app.services.hubspot_service.decrypt_hubspot_token",
                side_effect=["plain-refresh-token", "new-plain-access-token"],
            ),
            patch(
                "app.services.hubspot_service.refresh_access_token",
                new=AsyncMock(return_value={"access_token": "new", "expires_in": 1800}),
            ) as mock_refresh,
            patch("app.services.hubspot_service.upsert_tokens", return_value=new_row) as mock_upsert,
        ):
            token = await hubspot_service.get_valid_access_token(db, _TENANT_ID)

        mock_refresh.assert_awaited_once_with("plain-refresh-token")
        mock_upsert.assert_called_once()
        assert token == "new-plain-access-token"

    @pytest.mark.anyio
    async def test_returns_none_when_expired_and_no_refresh_token(self):
        row = _integration_row(expires_in_seconds=-60, has_refresh_token=False)
        db = MagicMock()

        with patch("app.services.hubspot_service.get_integration", return_value=row):
            token = await hubspot_service.get_valid_access_token(db, _TENANT_ID)

        assert token is None

    @pytest.mark.anyio
    async def test_returns_none_when_no_integration(self):
        db = MagicMock()
        with patch("app.services.hubspot_service.get_integration", return_value=None):
            token = await hubspot_service.get_valid_access_token(db, _TENANT_ID)
        assert token is None

    @pytest.mark.anyio
    async def test_refresh_failure_fails_open(self):
        row = _integration_row(expires_in_seconds=-60)
        db = MagicMock()

        with (
            patch("app.services.hubspot_service.get_integration", return_value=row),
            patch(
                "app.services.hubspot_service.decrypt_hubspot_token",
                return_value="plain-refresh-token",
            ),
            patch(
                "app.services.hubspot_service.refresh_access_token",
                new=AsyncMock(side_effect=Exception("HubSpot down")),
            ),
        ):
            token = await hubspot_service.get_valid_access_token(db, _TENANT_ID)

        assert token is None


# ── Contact lookup ──────────────────────────────────────────────────────────────


_RAW_CONTACT = {
    "id": "100451",
    "properties": {
        "firstname": "Ada",
        "lastname": "Lovelace",
        "email": "ada@example.com",
        "company": "Acme Corp",
        "notes_last_contacted": "2026-06-01T10:00:00Z",
        "lastmodifieddate": "2026-06-10T10:00:00Z",
    },
}


class TestContactShape:
    def test_contact_dict_from_hubspot_shape(self):
        contact = hubspot_service._contact_dict_from_hubspot(_RAW_CONTACT)
        assert contact == {
            "id": "100451",
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "company": "Acme Corp",
            "last_interaction_date": "2026-06-01T10:00:00Z",
        }

    def test_falls_back_to_lastmodifieddate(self):
        raw = {
            "id": "1",
            "properties": {"firstname": "A", "lastname": "B", "lastmodifieddate": "2026-01-01"},
        }
        contact = hubspot_service._contact_dict_from_hubspot(raw)
        assert contact["last_interaction_date"] == "2026-01-01"


class TestGetContactForPhone:
    @pytest.mark.anyio
    async def test_cache_hit_skips_hubspot_call(self):
        db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(
            return_value='{"id": "1", "name": "Cached Person", "email": null, "company": null, "last_interaction_date": null}'
        )

        with (
            patch("app.services.hubspot_service.get_redis", return_value=mock_redis),
            patch("app.services.hubspot_service.get_valid_access_token") as mock_token,
        ):
            contact = await hubspot_service.get_contact_for_phone(db, _TENANT_ID, "+15550001111")

        mock_token.assert_not_called()
        assert contact["name"] == "Cached Person"

    @pytest.mark.anyio
    async def test_cache_miss_fetches_and_stores(self):
        db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock()

        with (
            patch("app.services.hubspot_service.get_redis", return_value=mock_redis),
            patch(
                "app.services.hubspot_service.get_valid_access_token",
                new=AsyncMock(return_value="access-token"),
            ),
            patch(
                "app.services.hubspot_service.search_contact_by_phone",
                new=AsyncMock(return_value=_RAW_CONTACT),
            ),
        ):
            contact = await hubspot_service.get_contact_for_phone(db, _TENANT_ID, "+15550001111")

        assert contact["id"] == "100451"
        assert contact["name"] == "Ada Lovelace"
        mock_redis.set.assert_awaited_once()
        _, kwargs = mock_redis.set.call_args
        assert kwargs.get("ex") == 300

    @pytest.mark.anyio
    async def test_no_match_caches_not_found_sentinel(self):
        db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock()

        with (
            patch("app.services.hubspot_service.get_redis", return_value=mock_redis),
            patch(
                "app.services.hubspot_service.get_valid_access_token",
                new=AsyncMock(return_value="access-token"),
            ),
            patch(
                "app.services.hubspot_service.search_contact_by_phone",
                new=AsyncMock(return_value=None),
            ),
        ):
            contact = await hubspot_service.get_contact_for_phone(db, _TENANT_ID, "+15550001111")

        assert contact is None
        mock_redis.set.assert_awaited_once_with(
            hubspot_service._contact_cache_key(_TENANT_ID, "+15550001111"),
            hubspot_service._CONTACT_NOT_FOUND_SENTINEL,
            ex=300,
        )

    @pytest.mark.anyio
    async def test_fails_open_on_hubspot_error(self):
        """A HubSpot outage during contact search must never raise — call proceeds without CRM data."""
        db = MagicMock()

        with (
            patch("app.services.hubspot_service.get_redis", return_value=None),
            patch(
                "app.services.hubspot_service.get_valid_access_token",
                new=AsyncMock(return_value="access-token"),
            ),
            patch(
                "app.services.hubspot_service.search_contact_by_phone",
                new=AsyncMock(side_effect=Exception("HubSpot 500")),
            ),
        ):
            contact = await hubspot_service.get_contact_for_phone(db, _TENANT_ID, "+15550001111")

        assert contact is None

    @pytest.mark.anyio
    async def test_no_access_token_returns_none(self):
        db = MagicMock()
        with (
            patch("app.services.hubspot_service.get_redis", return_value=None),
            patch(
                "app.services.hubspot_service.get_valid_access_token",
                new=AsyncMock(return_value=None),
            ),
        ):
            contact = await hubspot_service.get_contact_for_phone(db, _TENANT_ID, "+15550001111")
        assert contact is None


# ── CRM context block (conversation_orchestrator injection) ───────────────────


def _call_session(*, phone="+15550001111", metadata=None):
    cs = MagicMock()
    cs.id = _SESSION_ID
    cs.tenant_id = _TENANT_ID
    cs.customer_phone_number = phone
    cs.call_metadata = metadata
    return cs


class TestCrmContextBlock:
    @pytest.mark.anyio
    async def test_returns_cached_value_without_refetching(self):
        db = MagicMock()
        call_session = _call_session(metadata={"hubspot_crm_context": "# CRM CONTEXT\ncached"})

        with patch("app.services.hubspot_service.tenant_has_hubspot_connected") as mock_connected:
            block = await hubspot_service.get_crm_context_block_for_call(db, call_session)

        mock_connected.assert_not_called()
        assert block == "# CRM CONTEXT\ncached"

    @pytest.mark.anyio
    async def test_fetches_and_caches_on_first_call(self):
        db = MagicMock()
        call_session = _call_session(metadata={})
        contact = {
            "id": "1",
            "name": "Ada Lovelace",
            "company": "Acme Corp",
            "last_interaction_date": "2026-06-01",
        }

        with (
            patch("app.services.hubspot_service.tenant_has_hubspot_connected", return_value=True),
            patch(
                "app.services.hubspot_service.get_contact_for_phone",
                new=AsyncMock(return_value=contact),
            ),
        ):
            block = await hubspot_service.get_crm_context_block_for_call(db, call_session)

        assert "CRM CONTEXT: Caller name: Ada Lovelace" in block
        assert "Company: Acme Corp" in block
        assert "Last interaction: 2026-06-01" in block
        assert call_session.call_metadata["hubspot_crm_context"] == block
        db.flush.assert_called_once()
        db.commit.assert_not_called()

    @pytest.mark.anyio
    async def test_fails_open_on_flush_error(self):
        # Mocks a failure in the db.flush() call to test that context generation
        # fails open and still returns the context block without throwing.
        db = MagicMock()
        db.flush.side_effect = Exception("Flush failed")
        # To support db.begin_nested() mock context manager
        nested_mock = MagicMock()
        db.begin_nested.return_value = nested_mock
        nested_mock.__enter__ = MagicMock()
        nested_mock.__exit__ = MagicMock()

        call_session = _call_session(metadata={})
        contact = {
            "id": "1",
            "name": "Ada Lovelace",
            "company": "Acme Corp",
            "last_interaction_date": "2026-06-01",
        }

        with (
            patch("app.services.hubspot_service.tenant_has_hubspot_connected", return_value=True),
            patch(
                "app.services.hubspot_service.get_contact_for_phone",
                new=AsyncMock(return_value=contact),
            ),
        ):
            block = await hubspot_service.get_crm_context_block_for_call(db, call_session)

        assert "CRM CONTEXT: Caller name: Ada Lovelace" in block
        db.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_not_connected_returns_empty_block(self):
        db = MagicMock()
        call_session = _call_session(metadata={})

        with patch("app.services.hubspot_service.tenant_has_hubspot_connected", return_value=False):
            block = await hubspot_service.get_crm_context_block_for_call(db, call_session)

        assert block == ""

    @pytest.mark.anyio
    async def test_fails_open_on_exception(self):
        db = MagicMock()
        call_session = _call_session(metadata={})

        with patch(
            "app.services.hubspot_service.tenant_has_hubspot_connected",
            side_effect=Exception("DB down"),
        ):
            block = await hubspot_service.get_crm_context_block_for_call(db, call_session)

        assert block == ""


# ── Post-call write-back ───────────────────────────────────────────────────────


class TestPostCallWriteback:
    @pytest.mark.anyio
    async def test_creates_engagement_with_summary(self):
        db = MagicMock()
        call_session = MagicMock()
        call_session.id = _SESSION_ID
        call_session.tenant_id = _TENANT_ID
        call_session.customer_phone_number = "+15550001111"
        call_session.start_time = datetime.now(timezone.utc)
        call_session.duration = 120
        call_session.call_type = "outbound"
        call_session.status = "completed"

        contact = {"id": "contact-1", "name": "Ada Lovelace"}

        with (
            patch(
                "app.services.hubspot_service.get_valid_access_token",
                new=AsyncMock(return_value="access-token"),
            ),
            patch(
                "app.services.hubspot_service.get_contact_for_phone",
                new=AsyncMock(return_value=contact),
            ),
            patch(
                "app.services.hubspot_service.generate_transcript_summary",
                return_value="Caller asked about pricing. Agent booked a follow-up demo.",
            ),
            patch(
                "app.services.hubspot_service.create_call_engagement",
                new=AsyncMock(return_value={"id": "engagement-1"}),
            ) as mock_create_engagement,
        ):
            await hubspot_service._run_post_call_writeback_async(db, call_session)

        mock_create_engagement.assert_awaited_once()
        _, kwargs = mock_create_engagement.call_args
        assert kwargs["direction"] == "OUTBOUND"
        assert kwargs["hs_status"] == "COMPLETED"
        assert kwargs["duration_seconds"] == 120
        assert "pricing" in kwargs["body_text"]
        assert mock_create_engagement.call_args[0][1] == "contact-1"

    @pytest.mark.anyio
    async def test_skips_when_no_matching_contact(self):
        db = MagicMock()
        call_session = MagicMock()
        call_session.tenant_id = _TENANT_ID
        call_session.customer_phone_number = "+15550001111"
        call_session.id = _SESSION_ID

        with (
            patch(
                "app.services.hubspot_service.get_valid_access_token",
                new=AsyncMock(return_value="access-token"),
            ),
            patch(
                "app.services.hubspot_service.get_contact_for_phone",
                new=AsyncMock(return_value=None),
            ),
            patch("app.services.hubspot_service.create_call_engagement") as mock_create,
        ):
            await hubspot_service._run_post_call_writeback_async(db, call_session)

        mock_create.assert_not_called()

    def test_run_post_call_writeback_skips_when_not_connected(self):
        """Sync entrypoint must not touch HubSpot when the tenant has no connection."""
        db = MagicMock()
        call_session = MagicMock()
        call_session.tenant_id = _TENANT_ID
        call_session.customer_phone_number = "+15550001111"
        db.query.return_value.filter.return_value.first.return_value = call_session

        with (
            patch("app.services.hubspot_service.SessionLocal", return_value=db),
            patch("app.services.hubspot_service.tenant_has_hubspot_connected", return_value=False),
            patch("asyncio.run") as mock_asyncio_run,
        ):
            hubspot_service.run_post_call_writeback(_SESSION_ID)

        mock_asyncio_run.assert_not_called()
        db.close.assert_called_once()

    def test_call_status_and_direction_mapping(self):
        assert hubspot_service._hs_call_status("no_answer") == "NO_ANSWER"
        assert hubspot_service._hs_call_status("unknown") == "COMPLETED"
        assert hubspot_service._hs_call_direction("inbound") == "INBOUND"
        assert hubspot_service._hs_call_direction("outbound") == "OUTBOUND"


# ── Disconnect ──────────────────────────────────────────────────────────────────


class TestDisconnect:
    @pytest.mark.anyio
    async def test_revokes_and_deletes_row(self):
        db = MagicMock()
        row = MagicMock()
        row.refresh_token = "encrypted-refresh-token"

        with (
            patch("app.services.hubspot_service.get_integration", return_value=row),
            patch(
                "app.services.hubspot_service.decrypt_hubspot_token",
                return_value="plain-refresh-token",
            ),
            patch(
                "app.services.hubspot_service._request_with_backoff",
                new=AsyncMock(return_value=MagicMock(status_code=204)),
            ) as mock_request,
        ):
            result = await hubspot_service.disconnect(db, _TENANT_ID)

        assert result is True
        mock_request.assert_awaited_once()
        assert mock_request.call_args[0][0] == "DELETE"
        db.delete.assert_called_once_with(row)
        db.commit.assert_called_once()

    @pytest.mark.anyio
    async def test_returns_false_when_not_connected(self):
        db = MagicMock()
        with patch("app.services.hubspot_service.get_integration", return_value=None):
            result = await hubspot_service.disconnect(db, _TENANT_ID)
        assert result is False
        db.delete.assert_not_called()

    @pytest.mark.anyio
    async def test_revoke_failure_still_deletes_local_row(self):
        """HubSpot revoke endpoint being down must not block local disconnect."""
        db = MagicMock()
        row = MagicMock()
        row.refresh_token = "encrypted-refresh-token"

        with (
            patch("app.services.hubspot_service.get_integration", return_value=row),
            patch(
                "app.services.hubspot_service.decrypt_hubspot_token",
                return_value="plain-refresh-token",
            ),
            patch(
                "app.services.hubspot_service._request_with_backoff",
                new=AsyncMock(side_effect=Exception("network error")),
            ),
        ):
            result = await hubspot_service.disconnect(db, _TENANT_ID)

        assert result is True
        db.delete.assert_called_once_with(row)


# ── 429 backoff ─────────────────────────────────────────────────────────────────


class TestBackoff:
    @pytest.mark.anyio
    async def test_retries_on_429_then_succeeds(self):
        responses = [MagicMock(status_code=429, headers={}), MagicMock(status_code=200, headers={})]

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=responses)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
        ):
            response = await hubspot_service._request_with_backoff("GET", "https://api.hubapi.com/x")

        assert response.status_code == 200
        mock_sleep.assert_awaited_once()
        assert mock_sleep.call_args[0][0] == 1.0  # first backoff: base * 2**0

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
            await hubspot_service._request_with_backoff("GET", "https://api.hubapi.com/x")

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
            response = await hubspot_service._request_with_backoff("GET", "https://api.hubapi.com/x")

        assert response.status_code == 429
        assert mock_client.request.await_count == hubspot_service._MAX_RETRIES + 1


class TestPhoneNormalization:
    def test_normalize_to_e164(self):
        # Already E.164
        assert hubspot_service.normalize_to_e164("+15550001111") == "+15550001111"
        # US formats
        assert hubspot_service.normalize_to_e164("5550001111") == "+15550001111"
        assert hubspot_service.normalize_to_e164("15550001111") == "+15550001111"
        assert hubspot_service.normalize_to_e164("+1 (555) 000-1111") == "+15550001111"
        # International format
        assert hubspot_service.normalize_to_e164("+44 7123 456789") == "+447123456789"
        assert hubspot_service.normalize_to_e164("447123456789") == "+447123456789"
        # Non-numeric / weird formats
        assert hubspot_service.normalize_to_e164("") == ""
        assert hubspot_service.normalize_to_e164("abc") == "abc"

    def test_get_phone_search_values(self):
        # Test exact match search variations generated for a formatted US phone number
        vals = hubspot_service._get_phone_search_values("+1 (555) 000-1111")
        # E.164 (+15550001111), raw digits (15550001111), US national digits (5550001111), raw stripped input
        assert "+15550001111" in vals
        assert "15550001111" in vals
        assert "5550001111" in vals
        assert "+1 (555) 000-1111" in vals
