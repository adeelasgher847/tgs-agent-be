"""
Tests for GoHighLevel (GHL) CRM integration service.

External HTTP calls (OAuth token endpoint, Contacts API, Notes API) are mocked
at the boundary (`_request_with_backoff`) per CLAUDE.md convention.

Coverage:
  1.  OAuth state: sign/verify round trip, tamper rejection, wrong purpose rejection
  2.  Authorization URL: client_id/scope/state present
  3.  Token refresh: returns cached (token, location_id) when fresh; refreshes when
      within 5 min of expiry; persists location_id; returns None when no refresh_token;
      fails open
  4.  Contact lookup: correct {id, name, email, tags, pipeline_stage, last_activity_date}
      shape; E.164-then-local fallback; Redis cache hit/miss/store; fails open on GHL error
  5.  CRM context block: exact "CRM CONTEXT (GoHighLevel): ..." format, cached on
      call_metadata after first fetch; fails open
  6.  Post-call write-back: creates a note with duration/outcome/Gemini summary;
      skips when write-back disabled; records/clears last_ghl_error; retries once
  7.  Disconnect: deletes local row; returns False when not connected
  8.  429 backoff: retries with exponential delay, honors Retry-After
  9.  Phone normalization / local-format fallback
  10. Redis-backed rate limiter: no-ops without Redis; waits when over budget
  11. ARQ enqueue: schedule_ghl_writeback enqueues the job; fails open without a pool
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import ghl_service


_TENANT_ID = uuid.UUID("aa300000-0000-0000-0000-000000000001")
_SESSION_ID = uuid.UUID("bb300000-0000-0000-0000-000000000002")


# ── OAuth state ────────────────────────────────────────────────────────────────


class TestOAuthState:
    def test_roundtrip(self):
        state = ghl_service.build_oauth_state(_TENANT_ID)
        assert ghl_service.verify_oauth_state(state) == _TENANT_ID

    def test_tampered_state_rejected(self):
        state = ghl_service.build_oauth_state(_TENANT_ID)
        with pytest.raises(ValueError):
            ghl_service.verify_oauth_state(state + "x")

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
            ghl_service.verify_oauth_state(bad_state)

    def test_expired_state_rejected(self):
        from jose import jwt as jose_jwt
        from app.core.config import settings

        expired_state = jose_jwt.encode(
            {
                "tenant_id": str(_TENANT_ID),
                "purpose": "ghl_oauth_state",
                "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
            },
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        with pytest.raises(ValueError):
            ghl_service.verify_oauth_state(expired_state)


class TestAuthorizationUrl:
    def test_contains_client_id_scope_and_state(self):
        with patch(
            "app.services.ghl_service.get_ghl_oauth_credentials",
            return_value=("client-123", "secret-456"),
        ):
            state = ghl_service.build_oauth_state(_TENANT_ID)
            url = ghl_service.build_authorization_url(state)

        assert url.startswith("https://marketplace.gohighlevel.com/oauth/chooselocation")
        assert "client_id=client-123" in url
        assert "contacts.readonly" in url
        assert f"state={state}" in url


# ── Token refresh ──────────────────────────────────────────────────────────────


def _integration_row(*, expires_in_seconds: float, has_refresh_token: bool = True, location_id="loc-1"):
    row = MagicMock()
    row.access_token = "encrypted-access-token"
    row.refresh_token = "encrypted-refresh-token" if has_refresh_token else None
    row.token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds)
    row.extra_metadata = {"location_id": location_id} if location_id else {}
    return row


class TestTokenRefresh:
    @pytest.mark.anyio
    async def test_returns_existing_token_when_not_near_expiry(self):
        row = _integration_row(expires_in_seconds=600)
        db = MagicMock()

        with (
            patch("app.services.ghl_service.get_integration", return_value=row),
            patch(
                "app.services.ghl_service.decrypt_ghl_token",
                return_value="plain-access-token",
            ) as mock_decrypt,
        ):
            result = await ghl_service.get_valid_access_token(db, _TENANT_ID)

        assert result == ("plain-access-token", "loc-1")
        mock_decrypt.assert_called_once_with(row.access_token, db)

    @pytest.mark.anyio
    async def test_refreshes_when_within_5_minutes_of_expiry(self):
        row = _integration_row(expires_in_seconds=120)
        db = MagicMock()
        new_row = _integration_row(expires_in_seconds=3600, location_id="loc-1")

        with (
            patch("app.services.ghl_service.get_integration", return_value=row),
            patch(
                "app.services.ghl_service.decrypt_ghl_token",
                side_effect=["plain-refresh-token", "new-plain-access-token"],
            ),
            patch(
                "app.services.ghl_service.refresh_access_token",
                new=AsyncMock(
                    return_value={
                        "access_token": "new",
                        "refresh_token": "new-refresh",
                        "expires_in": 3600,
                        "locationId": "loc-1",
                    }
                ),
            ) as mock_refresh,
            patch("app.services.ghl_service.upsert_tokens", return_value=new_row) as mock_upsert,
        ):
            result = await ghl_service.get_valid_access_token(db, _TENANT_ID)

        mock_refresh.assert_awaited_once_with("plain-refresh-token")
        mock_upsert.assert_called_once()
        assert result == ("new-plain-access-token", "loc-1")

    @pytest.mark.anyio
    async def test_returns_none_when_expired_and_no_refresh_token(self):
        row = _integration_row(expires_in_seconds=-60, has_refresh_token=False)
        db = MagicMock()

        with patch("app.services.ghl_service.get_integration", return_value=row):
            result = await ghl_service.get_valid_access_token(db, _TENANT_ID)

        assert result is None

    @pytest.mark.anyio
    async def test_returns_none_when_no_integration(self):
        db = MagicMock()
        with patch("app.services.ghl_service.get_integration", return_value=None):
            result = await ghl_service.get_valid_access_token(db, _TENANT_ID)
        assert result is None

    @pytest.mark.anyio
    async def test_returns_none_when_no_location_id_stored(self):
        row = _integration_row(expires_in_seconds=600, location_id=None)
        db = MagicMock()
        with patch("app.services.ghl_service.get_integration", return_value=row):
            result = await ghl_service.get_valid_access_token(db, _TENANT_ID)
        assert result is None

    @pytest.mark.anyio
    async def test_refresh_failure_fails_open(self):
        row = _integration_row(expires_in_seconds=120)
        db = MagicMock()

        with (
            patch("app.services.ghl_service.get_integration", return_value=row),
            patch(
                "app.services.ghl_service.decrypt_ghl_token",
                return_value="plain-refresh-token",
            ),
            patch(
                "app.services.ghl_service.refresh_access_token",
                new=AsyncMock(side_effect=Exception("GHL down")),
            ),
        ):
            result = await ghl_service.get_valid_access_token(db, _TENANT_ID)

        assert result is None


class TestForceRefreshAccessToken:
    @pytest.mark.anyio
    async def test_always_refreshes_even_when_not_expired(self):
        row = _integration_row(expires_in_seconds=3600)
        db = MagicMock()
        new_row = _integration_row(expires_in_seconds=3600)

        with (
            patch("app.services.ghl_service.get_integration", return_value=row),
            patch(
                "app.services.ghl_service.decrypt_ghl_token",
                side_effect=["plain-refresh-token", "new-token"],
            ),
            patch(
                "app.services.ghl_service.refresh_access_token",
                new=AsyncMock(return_value={"access_token": "new", "locationId": "loc-1"}),
            ) as mock_refresh,
            patch("app.services.ghl_service.upsert_tokens", return_value=new_row),
        ):
            result = await ghl_service._force_refresh_access_token(db, _TENANT_ID)

        mock_refresh.assert_awaited_once()
        assert result == ("new-token", "loc-1")

    @pytest.mark.anyio
    async def test_fails_open_when_refresh_errors(self):
        row = _integration_row(expires_in_seconds=3600)
        db = MagicMock()

        with (
            patch("app.services.ghl_service.get_integration", return_value=row),
            patch(
                "app.services.ghl_service.decrypt_ghl_token",
                return_value="plain-refresh-token",
            ),
            patch(
                "app.services.ghl_service.refresh_access_token",
                new=AsyncMock(side_effect=Exception("down")),
            ),
        ):
            result = await ghl_service._force_refresh_access_token(db, _TENANT_ID)

        assert result is None

    @pytest.mark.anyio
    async def test_falls_back_to_get_valid_access_token_when_no_refresh_token(self):
        row = _integration_row(expires_in_seconds=3600, has_refresh_token=False)
        db = MagicMock()

        with (
            patch("app.services.ghl_service.get_integration", return_value=row),
            patch(
                "app.services.ghl_service.decrypt_ghl_token",
                return_value="plain-access-token",
            ),
        ):
            result = await ghl_service._force_refresh_access_token(db, _TENANT_ID)

        assert result == ("plain-access-token", "loc-1")


# ── Contact lookup ──────────────────────────────────────────────────────────────


_RAW_CONTACT = {
    "id": "contact-1",
    "firstName": "Ada",
    "lastName": "Lovelace",
    "email": "ada@example.com",
    "tags": ["vip", "lead"],
    "pipelineStage": "Negotiation",
    "lastActivity": "2026-06-01T00:00:00Z",
}


class TestContactShape:
    def test_contact_dict_from_ghl_shape(self):
        contact = ghl_service._contact_dict_from_ghl(_RAW_CONTACT)
        assert contact == {
            "id": "contact-1",
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "tags": ["vip", "lead"],
            "pipeline_stage": "Negotiation",
            "last_activity_date": "2026-06-01T00:00:00Z",
        }

    def test_missing_name_fields_resolve_to_none(self):
        raw = {"id": "1", "email": "a@b.com"}
        contact = ghl_service._contact_dict_from_ghl(raw)
        assert contact["name"] is None
        assert contact["tags"] == []


class TestLocalFormatFallback:
    def test_au_number_strips_country_code_and_prepends_zero(self):
        assert ghl_service.local_format_fallback("+61412345678") == "0412345678"

    def test_uk_number(self):
        assert ghl_service.local_format_fallback("+447911123456") == "07911123456"

    def test_nanp_number_has_no_local_fallback(self):
        assert ghl_service.local_format_fallback("+15550001111") is None

    def test_empty_input(self):
        assert ghl_service.local_format_fallback("") is None


class TestSearchContactByPhone:
    @pytest.mark.anyio
    async def test_e164_match_found_on_first_try(self):
        response = MagicMock()
        response.json.return_value = {"contacts": [_RAW_CONTACT]}
        response.raise_for_status = MagicMock()

        with (
            patch("app.services.ghl_service.check_rate_limit", new=AsyncMock()),
            patch(
                "app.services.ghl_service._request_with_backoff",
                new=AsyncMock(return_value=response),
            ) as mock_request,
        ):
            contact = await ghl_service.search_contact_by_phone(
                "token", "loc-1", "+61412345678", _TENANT_ID
            )

        assert contact == _RAW_CONTACT
        assert mock_request.await_count == 1
        assert mock_request.call_args.kwargs["params"]["phone"] == "+61412345678"

    @pytest.mark.anyio
    async def test_falls_back_to_local_format_when_e164_has_no_match(self):
        empty_response = MagicMock()
        empty_response.json.return_value = {"contacts": []}
        empty_response.raise_for_status = MagicMock()
        match_response = MagicMock()
        match_response.json.return_value = {"contacts": [_RAW_CONTACT]}
        match_response.raise_for_status = MagicMock()

        with (
            patch("app.services.ghl_service.check_rate_limit", new=AsyncMock()),
            patch(
                "app.services.ghl_service._request_with_backoff",
                new=AsyncMock(side_effect=[empty_response, match_response]),
            ) as mock_request,
        ):
            contact = await ghl_service.search_contact_by_phone(
                "token", "loc-1", "+61412345678", _TENANT_ID
            )

        assert contact == _RAW_CONTACT
        assert mock_request.await_count == 2
        assert mock_request.call_args_list[0].kwargs["params"]["phone"] == "+61412345678"
        assert mock_request.call_args_list[1].kwargs["params"]["phone"] == "0412345678"

    @pytest.mark.anyio
    async def test_no_match_returns_none(self):
        empty_response = MagicMock()
        empty_response.json.return_value = {"contacts": []}
        empty_response.raise_for_status = MagicMock()

        with (
            patch("app.services.ghl_service.check_rate_limit", new=AsyncMock()),
            patch(
                "app.services.ghl_service._request_with_backoff",
                new=AsyncMock(return_value=empty_response),
            ),
        ):
            contact = await ghl_service.search_contact_by_phone(
                "token", "loc-1", "+15550001111", _TENANT_ID
            )

        assert contact is None


class TestGetContactForPhone:
    @pytest.mark.anyio
    async def test_cache_hit_skips_ghl_call(self):
        db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(
            return_value='{"id": "1", "name": "Cached Person", "email": null, "tags": [], "pipeline_stage": null, "last_activity_date": null}'
        )

        with (
            patch("app.services.ghl_service.get_redis", return_value=mock_redis),
            patch("app.services.ghl_service.get_valid_access_token") as mock_token,
        ):
            contact = await ghl_service.get_contact_for_phone(db, _TENANT_ID, "+15550001111")

        mock_token.assert_not_called()
        assert contact["name"] == "Cached Person"

    @pytest.mark.anyio
    async def test_cache_miss_fetches_and_stores(self):
        db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock()

        with (
            patch("app.services.ghl_service.get_redis", return_value=mock_redis),
            patch(
                "app.services.ghl_service.get_valid_access_token",
                new=AsyncMock(return_value=("access-token", "loc-1")),
            ),
            patch(
                "app.services.ghl_service.search_contact_by_phone",
                new=AsyncMock(return_value=_RAW_CONTACT),
            ),
        ):
            contact = await ghl_service.get_contact_for_phone(db, _TENANT_ID, "+15550001111")

        assert contact["id"] == "contact-1"
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
            patch("app.services.ghl_service.get_redis", return_value=mock_redis),
            patch(
                "app.services.ghl_service.get_valid_access_token",
                new=AsyncMock(return_value=("access-token", "loc-1")),
            ),
            patch(
                "app.services.ghl_service.search_contact_by_phone",
                new=AsyncMock(return_value=None),
            ),
        ):
            contact = await ghl_service.get_contact_for_phone(db, _TENANT_ID, "+15550001111")

        assert contact is None
        mock_redis.set.assert_awaited_once_with(
            ghl_service._contact_cache_key(_TENANT_ID, "+15550001111"),
            ghl_service._CONTACT_NOT_FOUND_SENTINEL,
            ex=300,
        )

    @pytest.mark.anyio
    async def test_fails_open_on_ghl_error(self):
        """A GHL outage during contact search must never raise — call proceeds without CRM data."""
        db = MagicMock()

        with (
            patch("app.services.ghl_service.get_redis", return_value=None),
            patch(
                "app.services.ghl_service.get_valid_access_token",
                new=AsyncMock(return_value=("access-token", "loc-1")),
            ),
            patch(
                "app.services.ghl_service.search_contact_by_phone",
                new=AsyncMock(side_effect=Exception("GHL 500")),
            ),
        ):
            contact = await ghl_service.get_contact_for_phone(db, _TENANT_ID, "+15550001111")

        assert contact is None

    @pytest.mark.anyio
    async def test_no_access_token_returns_none(self):
        db = MagicMock()

        with (
            patch("app.services.ghl_service.get_redis", return_value=None),
            patch(
                "app.services.ghl_service.get_valid_access_token",
                new=AsyncMock(return_value=None),
            ),
        ):
            contact = await ghl_service.get_contact_for_phone(db, _TENANT_ID, "+15550001111")

        assert contact is None


# ── CRM context block ────────────────────────────────────────────────────────────


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
        call_session = _call_session(
            metadata={"ghl_crm_context": "CRM CONTEXT (GoHighLevel): cached"}
        )

        with patch("app.services.ghl_service.tenant_has_ghl_connected") as mock_connected:
            block = await ghl_service.get_crm_context_block_for_call(db, call_session)

        mock_connected.assert_not_called()
        assert block == "CRM CONTEXT (GoHighLevel): cached"

    @pytest.mark.anyio
    async def test_fetches_and_caches_on_first_call(self):
        db = MagicMock()
        call_session = _call_session(metadata={})
        contact = {
            "id": "1",
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "tags": ["vip", "lead"],
            "pipeline_stage": "Negotiation",
        }

        with (
            patch("app.services.ghl_service.tenant_has_ghl_connected", return_value=True),
            patch(
                "app.services.ghl_service.get_contact_for_phone",
                new=AsyncMock(return_value=contact),
            ) as mock_get_contact,
        ):
            block = await ghl_service.get_crm_context_block_for_call(db, call_session)

        assert block == (
            "CRM CONTEXT (GoHighLevel): Name: Ada Lovelace, "
            "Tags: vip, lead, Pipeline: Negotiation"
        )
        assert call_session.call_metadata["ghl_crm_context"] == block
        db.flush.assert_called_once()
        db.commit.assert_not_called()
        assert mock_get_contact.call_args.kwargs.get("commit_lookup_timestamp") is False

    @pytest.mark.anyio
    async def test_not_connected_returns_empty_block(self):
        db = MagicMock()
        call_session = _call_session(metadata={})

        with patch("app.services.ghl_service.tenant_has_ghl_connected", return_value=False):
            block = await ghl_service.get_crm_context_block_for_call(db, call_session)

        assert block == ""

    @pytest.mark.anyio
    async def test_fails_open_on_exception(self):
        db = MagicMock()
        call_session = _call_session(metadata={})

        with patch(
            "app.services.ghl_service.tenant_has_ghl_connected",
            side_effect=Exception("DB down"),
        ):
            block = await ghl_service.get_crm_context_block_for_call(db, call_session)

        assert block == ""


# ── Post-call write-back ───────────────────────────────────────────────────────


class TestPostCallWriteback:
    @staticmethod
    def _writeback_settings(**overrides):
        settings = {
            "connected": True,
            "connected_at": datetime.now(timezone.utc),
            "last_sync_at": None,
            "write_back_enabled": True,
        }
        settings.update(overrides)
        return settings

    @pytest.mark.anyio
    async def test_creates_note_with_duration_outcome_and_summary(self):
        db = MagicMock()
        call_session = MagicMock()
        call_session.id = _SESSION_ID
        call_session.tenant_id = _TENANT_ID
        call_session.customer_phone_number = "+15550001111"
        call_session.duration = 120
        call_session.status = "completed"
        call_session.call_metadata = {}

        contact = {"id": "contact-1", "name": "Ada Lovelace"}

        with (
            patch(
                "app.services.ghl_service.get_integration_settings",
                return_value=self._writeback_settings(),
            ),
            patch(
                "app.services.ghl_service._force_refresh_access_token",
                new=AsyncMock(return_value=("access-token", "loc-1")),
            ) as mock_force_refresh,
            patch(
                "app.services.ghl_service.get_contact_for_phone",
                new=AsyncMock(return_value=contact),
            ),
            patch(
                "app.services.ghl_service.generate_transcript_summary",
                return_value="Caller asked about pricing. Agent booked a follow-up demo.",
            ),
            patch(
                "app.services.ghl_service.create_note",
                new=AsyncMock(return_value={"id": "note-1"}),
            ) as mock_create_note,
            patch("app.services.ghl_service.set_last_ghl_error") as mock_set_error,
        ):
            await ghl_service._run_post_call_writeback_async(db, call_session)

        mock_force_refresh.assert_awaited_once_with(db, _TENANT_ID)
        mock_create_note.assert_awaited_once()
        args, _kwargs = mock_create_note.call_args
        assert args[0] == "access-token"
        assert args[1] == "contact-1"
        assert "120s" in args[2]
        assert "completed" in args[2]
        assert "pricing" in args[2]
        assert args[3] == _TENANT_ID
        mock_set_error.assert_called_once_with(db, _TENANT_ID, None)

    @pytest.mark.anyio
    async def test_skips_when_write_back_disabled(self):
        db = MagicMock()
        call_session = MagicMock()
        call_session.tenant_id = _TENANT_ID
        call_session.customer_phone_number = "+15550001111"
        call_session.id = _SESSION_ID

        with (
            patch(
                "app.services.ghl_service.get_integration_settings",
                return_value=self._writeback_settings(write_back_enabled=False),
            ),
            patch("app.services.ghl_service._force_refresh_access_token") as mock_refresh,
            patch("app.services.ghl_service.create_note") as mock_create,
        ):
            await ghl_service._run_post_call_writeback_async(db, call_session)

        mock_refresh.assert_not_called()
        mock_create.assert_not_called()

    @pytest.mark.anyio
    async def test_skips_when_no_matching_contact(self):
        db = MagicMock()
        call_session = MagicMock()
        call_session.tenant_id = _TENANT_ID
        call_session.customer_phone_number = "+15550001111"
        call_session.id = _SESSION_ID

        with (
            patch(
                "app.services.ghl_service.get_integration_settings",
                return_value=self._writeback_settings(),
            ),
            patch(
                "app.services.ghl_service._force_refresh_access_token",
                new=AsyncMock(return_value=("access-token", "loc-1")),
            ),
            patch(
                "app.services.ghl_service.get_contact_for_phone",
                new=AsyncMock(return_value=None),
            ),
            patch("app.services.ghl_service.create_note") as mock_create,
        ):
            await ghl_service._run_post_call_writeback_async(db, call_session)

        mock_create.assert_not_called()

    @pytest.mark.anyio
    async def test_writeback_retries_once_then_records_last_ghl_error(self):
        call_order = []
        db = MagicMock()
        db.rollback.side_effect = lambda: call_order.append("rollback")
        call_session = MagicMock()
        call_session.id = _SESSION_ID
        call_session.tenant_id = _TENANT_ID
        call_session.customer_phone_number = "+15550001111"
        call_session.duration = 60
        call_session.status = "completed"
        call_session.call_metadata = {}

        contact = {"id": "contact-1", "name": "Ada Lovelace"}

        with (
            patch(
                "app.services.ghl_service.get_integration_settings",
                return_value=self._writeback_settings(),
            ),
            patch(
                "app.services.ghl_service._force_refresh_access_token",
                new=AsyncMock(return_value=("access-token", "loc-1")),
            ),
            patch(
                "app.services.ghl_service.get_contact_for_phone",
                new=AsyncMock(return_value=contact),
            ),
            patch(
                "app.services.ghl_service.generate_transcript_summary",
                return_value="Summary.",
            ),
            patch(
                "app.services.ghl_service.create_note",
                new=AsyncMock(side_effect=Exception("GHL 500")),
            ) as mock_create_note,
            patch(
                "asyncio.sleep",
                new=AsyncMock(side_effect=lambda *_a, **_k: call_order.append("sleep")),
            ) as mock_sleep,
            patch("app.services.ghl_service.record_write_back_failure") as mock_record_failure,
        ):
            await ghl_service._run_post_call_writeback_async(db, call_session)

        assert mock_create_note.await_count == 2
        mock_sleep.assert_awaited_once_with(ghl_service._WRITE_BACK_RETRY_DELAY_SECONDS)
        mock_record_failure.assert_called_once_with(db, _TENANT_ID, "GHL 500")
        db.rollback.assert_called_once()
        assert call_order == ["rollback", "sleep"]

    @pytest.mark.anyio
    async def test_arq_task_skips_when_not_connected(self):
        db = MagicMock()
        call_session = MagicMock()
        call_session.tenant_id = _TENANT_ID
        call_session.customer_phone_number = "+15550001111"
        db.query.return_value.filter.return_value.first.return_value = call_session

        with (
            patch("app.services.ghl_service.SessionLocal", return_value=db),
            patch("app.services.ghl_service.tenant_has_ghl_connected", return_value=False),
            patch("app.services.ghl_service._run_post_call_writeback_async") as mock_run,
        ):
            await ghl_service._post_call_writeback_arq_task({}, str(_SESSION_ID))

        mock_run.assert_not_called()
        db.close.assert_called_once()

    def test_schedule_writeback_enqueues_arq_job(self):
        mock_pool = MagicMock()
        mock_pool.enqueue_job = AsyncMock()

        with (
            patch("app.services.ghl_service.get_arq_pool", return_value=mock_pool),
            patch(
                "app.services.ghl_service.asyncio.get_running_loop",
                side_effect=RuntimeError("no running loop"),
            ),
            patch("app.services.ghl_service.asyncio.run") as mock_run,
        ):
            ghl_service.schedule_ghl_writeback(_SESSION_ID)

        mock_run.assert_called_once()

    def test_schedule_writeback_fails_open_without_arq_pool(self):
        with patch("app.services.ghl_service.get_arq_pool", return_value=None):
            # Must not raise even though no pool is available.
            ghl_service.schedule_ghl_writeback(_SESSION_ID)


# ── Disconnect ──────────────────────────────────────────────────────────────────


class TestDisconnect:
    @pytest.mark.anyio
    async def test_deletes_local_row(self):
        db = MagicMock()
        row = MagicMock()

        with patch("app.services.ghl_service.get_integration", return_value=row):
            result = await ghl_service.disconnect(db, _TENANT_ID)

        assert result is True
        db.delete.assert_called_once_with(row)
        db.commit.assert_called_once()

    @pytest.mark.anyio
    async def test_returns_false_when_not_connected(self):
        db = MagicMock()
        with patch("app.services.ghl_service.get_integration", return_value=None):
            result = await ghl_service.disconnect(db, _TENANT_ID)
        assert result is False
        db.delete.assert_not_called()


# ── Backoff ──────────────────────────────────────────────────────────────────────


class _FakeAsyncClient:
    def __init__(self, responses):
        self._responses = list(responses)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def request(self, method, url, **kwargs):
        return self._responses.pop(0)


class TestBackoff:
    @pytest.mark.anyio
    async def test_retries_on_429_then_succeeds(self):
        rate_limited = MagicMock(status_code=429, headers={})
        ok = MagicMock(status_code=200, headers={})

        with (
            patch(
                "app.services.ghl_service.httpx.AsyncClient",
                return_value=_FakeAsyncClient([rate_limited, ok]),
            ),
            patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
        ):
            response = await ghl_service._request_with_backoff("GET", "https://x")

        assert response.status_code == 200
        mock_sleep.assert_awaited_once_with(1.0)

    @pytest.mark.anyio
    async def test_honors_retry_after_header(self):
        rate_limited = MagicMock(status_code=429, headers={"Retry-After": "3"})
        ok = MagicMock(status_code=200, headers={})

        with (
            patch(
                "app.services.ghl_service.httpx.AsyncClient",
                return_value=_FakeAsyncClient([rate_limited, ok]),
            ),
            patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
        ):
            await ghl_service._request_with_backoff("GET", "https://x")

        mock_sleep.assert_awaited_once_with(3.0)

    @pytest.mark.anyio
    async def test_gives_up_after_max_retries(self):
        responses = [MagicMock(status_code=429, headers={}) for _ in range(6)]

        with (
            patch(
                "app.services.ghl_service.httpx.AsyncClient",
                return_value=_FakeAsyncClient(responses),
            ),
            patch("asyncio.sleep", new=AsyncMock()),
        ):
            response = await ghl_service._request_with_backoff("GET", "https://x")

        assert response.status_code == 429


# ── Rate limiter ──────────────────────────────────────────────────────────────────


class TestRateLimiter:
    @pytest.mark.anyio
    async def test_noop_without_redis(self):
        with patch("app.services.ghl_service.get_redis", return_value=None):
            # Must not raise.
            await ghl_service.check_rate_limit(_TENANT_ID)

    @pytest.mark.anyio
    async def test_waits_when_over_budget(self):
        mock_redis = AsyncMock()
        mock_redis.incr = AsyncMock(return_value=101)
        mock_redis.ttl = AsyncMock(return_value=4)

        with (
            patch("app.services.ghl_service.get_redis", return_value=mock_redis),
            patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
        ):
            await ghl_service.check_rate_limit(_TENANT_ID)

        mock_sleep.assert_awaited_once_with(4)

    @pytest.mark.anyio
    async def test_sets_expiry_on_first_request_in_window(self):
        mock_redis = AsyncMock()
        mock_redis.incr = AsyncMock(return_value=1)
        mock_redis.expire = AsyncMock()

        with patch("app.services.ghl_service.get_redis", return_value=mock_redis):
            await ghl_service.check_rate_limit(_TENANT_ID)

        mock_redis.expire.assert_awaited_once_with(
            f"ghl:rate_limit:{_TENANT_ID}", ghl_service._RATE_LIMIT_WINDOW_SECONDS
        )

    @pytest.mark.anyio
    async def test_under_budget_does_not_sleep(self):
        mock_redis = AsyncMock()
        mock_redis.incr = AsyncMock(return_value=5)

        with (
            patch("app.services.ghl_service.get_redis", return_value=mock_redis),
            patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
        ):
            await ghl_service.check_rate_limit(_TENANT_ID)

        mock_sleep.assert_not_called()

    @pytest.mark.anyio
    async def test_fails_open_on_redis_error(self):
        mock_redis = AsyncMock()
        mock_redis.incr = AsyncMock(side_effect=Exception("redis down"))

        with patch("app.services.ghl_service.get_redis", return_value=mock_redis):
            # Must not raise.
            await ghl_service.check_rate_limit(_TENANT_ID)


# ── Phone normalization ───────────────────────────────────────────────────────────


class TestPhoneNormalization:
    def test_normalize_to_e164(self):
        assert ghl_service.normalize_to_e164("5550001111") == "+15550001111"
        assert ghl_service.normalize_to_e164("15550001111") == "+15550001111"
        assert ghl_service.normalize_to_e164("+61412345678") == "+61412345678"
        assert ghl_service.normalize_to_e164("") == ""


# ── Sync status / settings ─────────────────────────────────────────────────────


class TestIntegrationSettings:
    def test_not_connected_returns_disconnected_defaults(self):
        db = MagicMock()
        with patch("app.services.ghl_service.get_integration", return_value=None):
            result = ghl_service.get_integration_settings(db, _TENANT_ID)
        assert result["connected"] is False
        assert result["write_back_enabled"] is True

    def test_update_settings_raises_when_not_connected(self):
        db = MagicMock()
        with patch("app.services.ghl_service.get_integration", return_value=None):
            with pytest.raises(ValueError):
                ghl_service.update_integration_settings(db, _TENANT_ID, write_back_enabled=False)


class TestSyncStatus:
    def test_returns_defaults_when_not_connected(self):
        db = MagicMock()
        with patch("app.services.ghl_service.get_integration", return_value=None):
            result = ghl_service.get_sync_status(db, _TENANT_ID)
        assert result == {
            "last_lookup_at": None,
            "last_write_back_at": None,
            "last_write_back_status": None,
            "last_ghl_error": None,
            "error_count_24h": 0,
        }


class TestSafeErrorMsg:
    def test_redacts_bearer_token(self):
        exc = Exception("failed with Bearer abc123.def456==")
        msg = ghl_service._safe_error_msg(exc)
        assert "abc123" not in msg
        assert "Bearer [redacted]" in msg
