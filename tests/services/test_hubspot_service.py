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
  6.  Post-call write-back: creates a Call engagement with the Gemini summary;
      skips when write-back disabled; records/clears last_write_back_error
  7.  Disconnect: revokes + deletes; returns False when not connected
  8.  429 backoff: retries with exponential delay, honors Retry-After
  9.  Field mappings: save/read from extra_metadata; settings toggles with defaults
  10. Field mapping value resolution + `{prompt_variable}` substitution into the prompt
  11. Transcript summary caching on call_metadata (Gemini called at most once per call)
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

    @pytest.mark.anyio
    async def test_defaults_to_committing_the_lookup_timestamp(self):
        """The standalone (non-live-call) callers — the /contact router endpoint and
        the post-call write-back job — must get a durable commit by default."""
        db = MagicMock()
        with (
            patch("app.services.hubspot_service.get_redis", return_value=None),
            patch(
                "app.services.hubspot_service.get_valid_access_token",
                new=AsyncMock(return_value=None),
            ),
            patch("app.services.hubspot_service._touch_last_lookup_at") as mock_touch,
        ):
            await hubspot_service.get_contact_for_phone(db, _TENANT_ID, "+15550001111")

        mock_touch.assert_called_once_with(db, _TENANT_ID, commit=True)

    @pytest.mark.anyio
    async def test_live_call_callers_opt_out_of_committing_the_lookup_timestamp(self):
        db = MagicMock()
        with (
            patch("app.services.hubspot_service.get_redis", return_value=None),
            patch(
                "app.services.hubspot_service.get_valid_access_token",
                new=AsyncMock(return_value=None),
            ),
            patch("app.services.hubspot_service._touch_last_lookup_at") as mock_touch,
        ):
            await hubspot_service.get_contact_for_phone(
                db, _TENANT_ID, "+15550001111", commit_lookup_timestamp=False
            )

        mock_touch.assert_called_once_with(db, _TENANT_ID, commit=False)


class TestTouchLastLookupAt:
    def test_commit_true_calls_db_commit(self):
        db = MagicMock()
        row = MagicMock()
        row.extra_metadata = {}

        with patch("app.services.hubspot_service.get_integration", return_value=row):
            hubspot_service._touch_last_lookup_at(db, _TENANT_ID, commit=True)

        db.commit.assert_called_once()
        assert "last_lookup_at" in row.extra_metadata

    def test_commit_false_never_commits_the_shared_session(self):
        """Must not call db.commit() on the shared live-call session — only the
        nested-savepoint flush pattern used elsewhere for call-time writes."""
        db = MagicMock()
        row = MagicMock()
        row.extra_metadata = {}
        nested_mock = MagicMock()
        db.begin_nested.return_value = nested_mock
        nested_mock.__enter__ = MagicMock()
        nested_mock.__exit__ = MagicMock()

        with patch("app.services.hubspot_service.get_integration", return_value=row):
            hubspot_service._touch_last_lookup_at(db, _TENANT_ID, commit=False)

        db.commit.assert_not_called()
        db.begin_nested.assert_called_once()
        db.flush.assert_called_once()
        assert "last_lookup_at" in row.extra_metadata

    def test_fails_open_on_error(self):
        db = MagicMock()
        with patch(
            "app.services.hubspot_service.get_integration", side_effect=Exception("DB down")
        ):
            hubspot_service._touch_last_lookup_at(db, _TENANT_ID, commit=True)  # must not raise


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
            ) as mock_get_contact,
        ):
            block = await hubspot_service.get_crm_context_block_for_call(db, call_session)

        assert "CRM CONTEXT: Caller name: Ada Lovelace" in block
        assert "Company: Acme Corp" in block
        assert "Last interaction: 2026-06-01" in block
        assert call_session.call_metadata["hubspot_crm_context"] == block
        db.flush.assert_called_once()
        db.commit.assert_not_called()
        # Must opt out of committing the shared live-call session — see
        # _touch_last_lookup_at's commit=False contract.
        assert mock_get_contact.call_args.kwargs.get("commit_lookup_timestamp") is False

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
    @staticmethod
    def _writeback_settings(**overrides):
        settings = {
            "connected": True,
            "connected_at": datetime.now(timezone.utc),
            "contact_lookup_enabled": True,
            "write_back_enabled": True,
            "field_mappings": [],
        }
        settings.update(overrides)
        return settings

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
        call_session.call_metadata = {}

        contact = {"id": "contact-1", "name": "Ada Lovelace"}

        with (
            patch(
                "app.services.hubspot_service.get_integration_settings",
                return_value=self._writeback_settings(),
            ),
            patch(
                "app.services.hubspot_service._force_refresh_access_token",
                new=AsyncMock(return_value="access-token"),
            ) as mock_force_refresh,
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
            patch(
                "app.services.hubspot_service.set_last_write_back_error"
            ) as mock_set_error,
        ):
            await hubspot_service._run_post_call_writeback_async(db, call_session)

        mock_force_refresh.assert_awaited_once_with(db, _TENANT_ID)
        mock_create_engagement.assert_awaited_once()
        _, kwargs = mock_create_engagement.call_args
        assert kwargs["direction"] == "OUTBOUND"
        assert kwargs["hs_status"] == "COMPLETED"
        assert kwargs["duration_seconds"] == 120
        assert "pricing" in kwargs["body_text"]
        assert mock_create_engagement.call_args[0][1] == "contact-1"
        mock_set_error.assert_called_once_with(db, _TENANT_ID, None)

    @pytest.mark.anyio
    async def test_skips_when_write_back_disabled(self):
        """Post-call write-back must not touch HubSpot when write_back_enabled is False."""
        db = MagicMock()
        call_session = MagicMock()
        call_session.tenant_id = _TENANT_ID
        call_session.customer_phone_number = "+15550001111"
        call_session.id = _SESSION_ID

        with (
            patch(
                "app.services.hubspot_service.get_integration_settings",
                return_value=self._writeback_settings(write_back_enabled=False),
            ),
            patch("app.services.hubspot_service._force_refresh_access_token") as mock_refresh,
            patch("app.services.hubspot_service.create_call_engagement") as mock_create,
        ):
            await hubspot_service._run_post_call_writeback_async(db, call_session)

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
                "app.services.hubspot_service.get_integration_settings",
                return_value=self._writeback_settings(),
            ),
            patch(
                "app.services.hubspot_service._force_refresh_access_token",
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

    @pytest.mark.anyio
    async def test_writeback_retries_once_after_5_minutes_then_records_structured_error(self):
        """A write-back that keeps failing retries once after 5 min, then records a
        structured {timestamp, error} object and never raises."""
        call_order = []
        db = MagicMock()
        db.rollback.side_effect = lambda: call_order.append("rollback")
        call_session = MagicMock()
        call_session.id = _SESSION_ID
        call_session.tenant_id = _TENANT_ID
        call_session.customer_phone_number = "+15550001111"
        call_session.start_time = datetime.now(timezone.utc)
        call_session.duration = 60
        call_session.call_type = "inbound"
        call_session.status = "completed"
        call_session.call_metadata = {}

        contact = {"id": "contact-1", "name": "Ada Lovelace"}

        with (
            patch(
                "app.services.hubspot_service.get_integration_settings",
                return_value=self._writeback_settings(),
            ),
            patch(
                "app.services.hubspot_service._force_refresh_access_token",
                new=AsyncMock(return_value="access-token"),
            ),
            patch(
                "app.services.hubspot_service.get_contact_for_phone",
                new=AsyncMock(return_value=contact),
            ),
            patch(
                "app.services.hubspot_service.generate_transcript_summary",
                return_value="Summary.",
            ),
            patch(
                "app.services.hubspot_service.create_call_engagement",
                new=AsyncMock(side_effect=Exception("HubSpot 500")),
            ) as mock_create_engagement,
            patch(
                "asyncio.sleep",
                new=AsyncMock(side_effect=lambda *_a, **_k: call_order.append("sleep")),
            ) as mock_sleep,
            patch(
                "app.services.hubspot_service.record_write_back_failure"
            ) as mock_record_failure,
        ):
            await hubspot_service._run_post_call_writeback_async(db, call_session)

        assert mock_create_engagement.await_count == 2  # initial attempt + one retry
        mock_sleep.assert_awaited_once_with(hubspot_service._WRITE_BACK_RETRY_DELAY_SECONDS)
        mock_record_failure.assert_called_once_with(db, _TENANT_ID, "HubSpot 500")
        # The DB connection must be released (rollback) before the 5-minute sleep,
        # not held open — earlier reads in this function leave an uncommitted
        # begin_nested()/flush() transaction on the session.
        db.rollback.assert_called_once()
        assert call_order == ["rollback", "sleep"]

    @pytest.mark.anyio
    async def test_writeback_retry_succeeds_on_second_attempt(self):
        """If the retry succeeds, the write-back is recorded as successful — no error persisted."""
        db = MagicMock()
        call_session = MagicMock()
        call_session.id = _SESSION_ID
        call_session.tenant_id = _TENANT_ID
        call_session.customer_phone_number = "+15550001111"
        call_session.start_time = datetime.now(timezone.utc)
        call_session.duration = 60
        call_session.call_type = "inbound"
        call_session.status = "completed"
        call_session.call_metadata = {}

        contact = {"id": "contact-1", "name": "Ada Lovelace"}

        with (
            patch(
                "app.services.hubspot_service.get_integration_settings",
                return_value=self._writeback_settings(),
            ),
            patch(
                "app.services.hubspot_service._force_refresh_access_token",
                new=AsyncMock(return_value="access-token"),
            ),
            patch(
                "app.services.hubspot_service.get_contact_for_phone",
                new=AsyncMock(return_value=contact),
            ),
            patch(
                "app.services.hubspot_service.generate_transcript_summary",
                return_value="Summary.",
            ),
            patch(
                "app.services.hubspot_service.create_call_engagement",
                new=AsyncMock(side_effect=[Exception("HubSpot 500"), {"id": "engagement-1"}]),
            ) as mock_create_engagement,
            patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
            patch(
                "app.services.hubspot_service.set_last_write_back_error"
            ) as mock_set_error,
            patch(
                "app.services.hubspot_service.record_write_back_failure"
            ) as mock_record_failure,
        ):
            await hubspot_service._run_post_call_writeback_async(db, call_session)

        assert mock_create_engagement.await_count == 2
        mock_sleep.assert_awaited_once()
        mock_record_failure.assert_not_called()
        mock_set_error.assert_called_once_with(db, _TENANT_ID, None)

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

    def test_schedule_writeback_never_blocks_caller_without_a_running_loop(self):
        """schedule_hubspot_writeback promises to never block the caller. Since
        write-back can now retry with a real 5-minute asyncio.sleep, the
        no-running-loop branch (e.g. a synchronous webhook handler calling
        CallSessionService.update_call_session_status) must hand off to a
        background thread rather than run inline."""
        with (
            patch(
                "app.services.hubspot_service.asyncio.get_running_loop",
                side_effect=RuntimeError("no running loop"),
            ),
            patch("app.services.hubspot_service.threading.Thread") as mock_thread_cls,
            patch("app.services.hubspot_service.run_post_call_writeback") as mock_run,
        ):
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            hubspot_service.schedule_hubspot_writeback(_SESSION_ID)

        mock_thread_cls.assert_called_once_with(
            target=mock_run, args=(_SESSION_ID,), daemon=True
        )
        mock_thread.start.assert_called_once()
        mock_run.assert_not_called()  # must not run synchronously on the caller's thread

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


# ── Field mapping / settings storage ────────────────────────────────────────────


class TestFieldMappingStorage:
    def test_save_and_read_field_mappings(self):
        db = MagicMock()
        row = MagicMock()
        row.extra_metadata = None

        mappings = [{"hubspot_field": "jobtitle", "prompt_variable": "job_title"}]

        with patch("app.services.hubspot_service.get_integration", return_value=row):
            hubspot_service.save_field_mappings(db, _TENANT_ID, mappings)

        assert row.extra_metadata == {"field_mappings": mappings}
        db.commit.assert_called_once()

    def test_save_field_mappings_preserves_other_metadata_keys(self):
        db = MagicMock()
        row = MagicMock()
        row.extra_metadata = {"contact_lookup_enabled": False}

        mappings = [{"hubspot_field": "jobtitle", "prompt_variable": "job_title"}]

        with patch("app.services.hubspot_service.get_integration", return_value=row):
            hubspot_service.save_field_mappings(db, _TENANT_ID, mappings)

        assert row.extra_metadata["contact_lookup_enabled"] is False
        assert row.extra_metadata["field_mappings"] == mappings

    def test_save_field_mappings_raises_when_not_connected(self):
        db = MagicMock()
        with patch("app.services.hubspot_service.get_integration", return_value=None):
            with pytest.raises(ValueError):
                hubspot_service.save_field_mappings(db, _TENANT_ID, [])

    def test_get_field_mappings_defaults_to_empty_list(self):
        db = MagicMock()
        row = MagicMock()
        row.extra_metadata = None
        with patch("app.services.hubspot_service.get_integration", return_value=row):
            assert hubspot_service.get_field_mappings(db, _TENANT_ID) == []


class TestIntegrationSettings:
    def test_returns_defaults_when_metadata_empty(self):
        db = MagicMock()
        row = MagicMock()
        row.extra_metadata = None
        row.created_at = datetime.now(timezone.utc)

        with patch("app.services.hubspot_service.get_integration", return_value=row):
            result = hubspot_service.get_integration_settings(db, _TENANT_ID)

        assert result["connected"] is True
        assert result["contact_lookup_enabled"] is True
        assert result["write_back_enabled"] is True
        assert result["field_mappings"] == []

    def test_returns_stored_overrides(self):
        db = MagicMock()
        row = MagicMock()
        row.extra_metadata = {
            "contact_lookup_enabled": False,
            "write_back_enabled": False,
            "field_mappings": [{"hubspot_field": "company", "prompt_variable": "company_name"}],
        }
        row.created_at = datetime.now(timezone.utc)

        with patch("app.services.hubspot_service.get_integration", return_value=row):
            result = hubspot_service.get_integration_settings(db, _TENANT_ID)

        assert result["contact_lookup_enabled"] is False
        assert result["write_back_enabled"] is False
        assert result["field_mappings"] == [
            {"hubspot_field": "company", "prompt_variable": "company_name"}
        ]

    def test_not_connected_returns_disconnected_defaults(self):
        db = MagicMock()
        with patch("app.services.hubspot_service.get_integration", return_value=None):
            result = hubspot_service.get_integration_settings(db, _TENANT_ID)

        assert result == {
            "connected": False,
            "connected_at": None,
            "contact_lookup_enabled": True,
            "write_back_enabled": True,
            "field_mappings": [],
        }

    def test_update_settings_persists_toggles(self):
        db = MagicMock()
        row = MagicMock()
        row.extra_metadata = {}

        with patch("app.services.hubspot_service.get_integration", return_value=row):
            hubspot_service.update_integration_settings(
                db, _TENANT_ID, contact_lookup_enabled=False, write_back_enabled=True
            )

        assert row.extra_metadata["contact_lookup_enabled"] is False
        assert row.extra_metadata["write_back_enabled"] is True
        db.commit.assert_called_once()

    def test_update_settings_raises_when_not_connected(self):
        db = MagicMock()
        with patch("app.services.hubspot_service.get_integration", return_value=None):
            with pytest.raises(ValueError):
                hubspot_service.update_integration_settings(
                    db, _TENANT_ID, contact_lookup_enabled=True, write_back_enabled=True
                )


class TestLastWriteBackError:
    def test_records_error(self):
        db = MagicMock()
        row = MagicMock()
        row.extra_metadata = {}

        with patch("app.services.hubspot_service.get_integration", return_value=row):
            hubspot_service.set_last_write_back_error(db, _TENANT_ID, "HubSpot 500")

        assert row.extra_metadata["last_write_back_error"] == "HubSpot 500"
        db.commit.assert_called_once()

    def test_clears_error_on_success(self):
        db = MagicMock()
        row = MagicMock()
        row.extra_metadata = {"last_write_back_error": "HubSpot 500"}

        with patch("app.services.hubspot_service.get_integration", return_value=row):
            hubspot_service.set_last_write_back_error(db, _TENANT_ID, None)

        assert "last_write_back_error" not in row.extra_metadata

    def test_noop_when_not_connected(self):
        db = MagicMock()
        with patch("app.services.hubspot_service.get_integration", return_value=None):
            hubspot_service.set_last_write_back_error(db, _TENANT_ID, "error")
        db.commit.assert_not_called()


# ── Field mapping resolution + prompt substitution ─────────────────────────────


class TestFieldMappingResolution:
    def test_resolves_from_raw_properties(self):
        contact = {
            "id": "1",
            "name": "Ada Lovelace",
            "raw_properties": {"jobtitle": "Engineer", "company": "Acme Corp"},
        }
        mappings = [{"hubspot_field": "jobtitle", "prompt_variable": "job_title"}]

        values = hubspot_service.resolve_field_mapping_values(contact, mappings)

        assert values == {"job_title": "Engineer"}

    def test_falls_back_to_top_level_contact_keys(self):
        contact = {"id": "1", "name": "Ada Lovelace", "raw_properties": {}}
        mappings = [{"hubspot_field": "name", "prompt_variable": "caller_name"}]

        values = hubspot_service.resolve_field_mapping_values(contact, mappings)

        assert values == {"caller_name": "Ada Lovelace"}

    def test_missing_or_null_values_resolve_to_empty_string(self):
        """A configured mapping whose field is missing/NULL on the contact still gets an
        entry (empty string) so the prompt placeholder is cleanly omitted, not left
        unreplaced or filled with 'null'/'undefined'."""
        contact = {"id": "1", "raw_properties": {"jobtitle": None, "company": ""}}
        mappings = [
            {"hubspot_field": "jobtitle", "prompt_variable": "job_title"},
            {"hubspot_field": "company", "prompt_variable": "company_name"},
            {"hubspot_field": "", "prompt_variable": "x"},
            {"hubspot_field": "y"},
        ]

        assert hubspot_service.resolve_field_mapping_values(contact, mappings) == {
            "job_title": "",
            "company_name": "",
        }

    def test_no_contact_still_resolves_all_mapped_variables_to_empty_string(self):
        mappings = [{"hubspot_field": "jobtitle", "prompt_variable": "job_title"}]
        assert hubspot_service.resolve_field_mapping_values(None, mappings) == {"job_title": ""}

    def test_apply_field_mapping_values_omits_placeholder_for_null_field(self):
        prompt = "Hello {caller_name}, your title is {job_title}."
        values = hubspot_service.resolve_field_mapping_values(
            {"id": "1", "raw_properties": {"jobtitle": None}},
            [
                {"hubspot_field": "firstname", "prompt_variable": "caller_name"},
                {"hubspot_field": "jobtitle", "prompt_variable": "job_title"},
            ],
        )
        result = hubspot_service.apply_field_mapping_values(prompt, values)
        assert result == "Hello , your title is ."
        assert "null" not in result.lower()
        assert "undefined" not in result.lower()

    def test_no_contact_or_no_mappings_returns_empty(self):
        assert hubspot_service.resolve_field_mapping_values(None, [{"a": "b"}]) == {}
        assert hubspot_service.resolve_field_mapping_values({"id": "1"}, []) == {}

    def test_apply_field_mapping_values_replaces_placeholders(self):
        prompt = "Hello {caller_name}, you work at {company_name}."
        values = {"caller_name": "Ada", "company_name": "Acme Corp"}

        result = hubspot_service.apply_field_mapping_values(prompt, values)

        assert result == "Hello Ada, you work at Acme Corp."

    def test_apply_field_mapping_values_noop_when_empty(self):
        prompt = "Hello {caller_name}."
        assert hubspot_service.apply_field_mapping_values(prompt, {}) == prompt


class TestGetFieldMappingValuesForCall:
    @pytest.mark.anyio
    async def test_returns_cached_value_without_refetching(self):
        db = MagicMock()
        call_session = _call_session(metadata={"hubspot_field_mapping_values": {"job_title": "Engineer"}})

        with patch("app.services.hubspot_service.tenant_has_hubspot_connected") as mock_connected:
            values = await hubspot_service.get_field_mapping_values_for_call(db, call_session)

        mock_connected.assert_not_called()
        assert values == {"job_title": "Engineer"}

    @pytest.mark.anyio
    async def test_resolves_and_caches_on_first_call(self):
        db = MagicMock()
        call_session = _call_session(metadata={})
        contact = {"id": "1", "raw_properties": {"jobtitle": "Engineer"}}
        integration_settings = {
            "connected": True,
            "contact_lookup_enabled": True,
            "write_back_enabled": True,
            "field_mappings": [{"hubspot_field": "jobtitle", "prompt_variable": "job_title"}],
        }

        with (
            patch("app.services.hubspot_service.tenant_has_hubspot_connected", return_value=True),
            patch(
                "app.services.hubspot_service.get_integration_settings",
                return_value=integration_settings,
            ),
            patch(
                "app.services.hubspot_service.get_contact_for_phone",
                new=AsyncMock(return_value=contact),
            ) as mock_get_contact,
        ):
            values = await hubspot_service.get_field_mapping_values_for_call(db, call_session)

        assert values == {"job_title": "Engineer"}
        assert call_session.call_metadata["hubspot_field_mapping_values"] == values
        assert mock_get_contact.call_args.kwargs.get("commit_lookup_timestamp") is False
        db.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_skips_lookup_when_contact_lookup_disabled(self):
        db = MagicMock()
        call_session = _call_session(metadata={})
        integration_settings = {
            "connected": True,
            "contact_lookup_enabled": False,
            "write_back_enabled": True,
            "field_mappings": [{"hubspot_field": "jobtitle", "prompt_variable": "job_title"}],
        }

        with (
            patch("app.services.hubspot_service.tenant_has_hubspot_connected", return_value=True),
            patch(
                "app.services.hubspot_service.get_integration_settings",
                return_value=integration_settings,
            ),
            patch("app.services.hubspot_service.get_contact_for_phone") as mock_get_contact,
        ):
            values = await hubspot_service.get_field_mapping_values_for_call(db, call_session)

        mock_get_contact.assert_not_called()
        assert values == {}

    @pytest.mark.anyio
    async def test_fails_open_on_exception(self):
        db = MagicMock()
        call_session = _call_session(metadata={})

        with patch(
            "app.services.hubspot_service.tenant_has_hubspot_connected",
            side_effect=Exception("DB down"),
        ):
            values = await hubspot_service.get_field_mapping_values_for_call(db, call_session)

        assert values == {}


# ── Transcript summary caching ──────────────────────────────────────────────────


class TestTranscriptSummaryCaching:
    def test_generates_and_caches_on_first_call(self):
        db = MagicMock()
        call_session = _call_session(metadata={})

        with patch(
            "app.services.hubspot_service.generate_transcript_summary",
            return_value="Caller asked about pricing.",
        ) as mock_generate:
            summary = hubspot_service.get_cached_transcript_summary(db, call_session)

        mock_generate.assert_called_once_with(db, call_session)
        assert summary == "Caller asked about pricing."
        assert call_session.call_metadata["hubspot_call_summary"] == summary

    def test_returns_cached_value_without_calling_gemini_again(self):
        db = MagicMock()
        call_session = _call_session(
            metadata={"hubspot_call_summary": "Cached summary."}
        )

        with patch(
            "app.services.hubspot_service.generate_transcript_summary"
        ) as mock_generate:
            summary = hubspot_service.get_cached_transcript_summary(db, call_session)

        mock_generate.assert_not_called()
        assert summary == "Cached summary."


class TestSafeErrorMsg:
    def test_redacts_bearer_token(self):
        from app.services.hubspot_service import _safe_error_msg
        exc = Exception("API failed: Bearer pat-na1-12345-abc-xyz-123456789")
        msg = _safe_error_msg(exc)
        assert "Bearer [redacted]" in msg
        assert "pat-na1-12345" not in msg

    def test_truncates_long_errors(self):
        from app.services.hubspot_service import _safe_error_msg
        exc = Exception("a" * 1000)
        msg = _safe_error_msg(exc)
        assert len(msg) == 500
        assert msg == "a" * 500


# ── Contact properties fetch (field-mapping schema validation) ─────────────────


class TestGetHubSpotContactProperties:
    @pytest.mark.anyio
    async def test_cache_hit_skips_hubspot_call(self):
        db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value='["jobtitle", "company"]')

        with (
            patch("app.services.hubspot_service.get_redis", return_value=mock_redis),
            patch("app.services.hubspot_service.get_valid_access_token") as mock_token,
        ):
            properties = await hubspot_service.get_hubspot_contact_properties(db, _TENANT_ID)

        mock_token.assert_not_called()
        assert properties == ["jobtitle", "company"]

    @pytest.mark.anyio
    async def test_cache_miss_fetches_and_caches_for_1_hour(self):
        db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock()

        response = MagicMock()
        response.json.return_value = {
            "results": [{"name": "jobtitle"}, {"name": "company"}, {"name": ""}]
        }
        response.raise_for_status = MagicMock()

        with (
            patch("app.services.hubspot_service.get_redis", return_value=mock_redis),
            patch(
                "app.services.hubspot_service.get_valid_access_token",
                new=AsyncMock(return_value="access-token"),
            ),
            patch(
                "app.services.hubspot_service._request_with_backoff",
                new=AsyncMock(return_value=response),
            ),
        ):
            properties = await hubspot_service.get_hubspot_contact_properties(db, _TENANT_ID)

        assert properties == ["jobtitle", "company"]
        mock_redis.set.assert_awaited_once()
        args, kwargs = mock_redis.set.call_args
        assert args[0] == hubspot_service._properties_cache_key(_TENANT_ID)
        assert kwargs.get("ex") == 3600

    @pytest.mark.anyio
    async def test_raises_when_not_connected(self):
        db = MagicMock()
        with (
            patch("app.services.hubspot_service.get_redis", return_value=None),
            patch(
                "app.services.hubspot_service.get_valid_access_token",
                new=AsyncMock(return_value=None),
            ),
        ):
            with pytest.raises(ValueError):
                await hubspot_service.get_hubspot_contact_properties(db, _TENANT_ID)


# ── Forced pre-writeback token refresh ──────────────────────────────────────────


class TestForceRefreshAccessToken:
    @pytest.mark.anyio
    async def test_always_refreshes_even_when_not_expired(self):
        """Must call refresh_access_token unconditionally, ignoring token_expires_at."""
        row = _integration_row(expires_in_seconds=1700)  # far from expiry
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
            patch("app.services.hubspot_service.upsert_tokens", return_value=new_row),
        ):
            token = await hubspot_service._force_refresh_access_token(db, _TENANT_ID)

        mock_refresh.assert_awaited_once_with("plain-refresh-token")
        assert token == "new-plain-access-token"

    @pytest.mark.anyio
    async def test_fails_open_when_refresh_errors(self):
        row = _integration_row(expires_in_seconds=1700)
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
            token = await hubspot_service._force_refresh_access_token(db, _TENANT_ID)

        assert token is None

    @pytest.mark.anyio
    async def test_falls_back_to_get_valid_access_token_when_no_refresh_token(self):
        row = _integration_row(expires_in_seconds=600, has_refresh_token=False)
        db = MagicMock()

        with (
            patch("app.services.hubspot_service.get_integration", return_value=row),
            patch(
                "app.services.hubspot_service.get_valid_access_token",
                new=AsyncMock(return_value="fallback-token"),
            ) as mock_fallback,
        ):
            token = await hubspot_service._force_refresh_access_token(db, _TENANT_ID)

        mock_fallback.assert_awaited_once_with(db, _TENANT_ID)
        assert token == "fallback-token"


# ── Sync status + write-back failure tracking / admin alerting ─────────────────


class TestSyncStatus:
    def test_returns_defaults_when_not_connected(self):
        db = MagicMock()
        with patch("app.services.hubspot_service.get_integration", return_value=None):
            result = hubspot_service.get_sync_status(db, _TENANT_ID)

        assert result == {
            "last_lookup_at": None,
            "last_write_back_at": None,
            "last_write_back_status": None,
            "error_count_24h": 0,
        }

    def test_returns_stored_metrics_and_filters_old_failures_outside_24h_window(self):
        db = MagicMock()
        row = MagicMock()
        now = datetime.now(timezone.utc)
        stale_ts = (now - timedelta(hours=30)).isoformat().replace("+00:00", "Z")
        recent_ts = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        row.extra_metadata = {
            "last_lookup_at": "2026-07-20T10:00:00Z",
            "last_write_back_at": "2026-07-20T10:05:00Z",
            "last_write_back_status": "failed",
            "write_back_failure_timestamps": [stale_ts, recent_ts, recent_ts],
        }

        with patch("app.services.hubspot_service.get_integration", return_value=row):
            result = hubspot_service.get_sync_status(db, _TENANT_ID)

        assert result["last_lookup_at"] == "2026-07-20T10:00:00Z"
        assert result["last_write_back_at"] == "2026-07-20T10:05:00Z"
        assert result["last_write_back_status"] == "failed"
        assert result["error_count_24h"] == 2  # stale_ts excluded


class TestRecordWriteBackFailure:
    def test_persists_structured_error_and_increments_counter(self):
        db = MagicMock()
        row = MagicMock()
        row.extra_metadata = {}

        with (
            patch("app.services.hubspot_service.get_integration", return_value=row),
            patch("app.services.hubspot_service._send_write_back_failure_alert") as mock_alert,
        ):
            hubspot_service.record_write_back_failure(db, _TENANT_ID, "HubSpot 500")

        error = row.extra_metadata["last_write_back_error"]
        assert error["error"] == "HubSpot 500"
        assert "timestamp" in error
        assert row.extra_metadata["last_write_back_status"] == "failed"
        assert len(row.extra_metadata["write_back_failure_timestamps"]) == 1
        mock_alert.assert_not_called()  # below threshold

    def test_sends_admin_alert_after_5_failures_in_24h(self):
        db = MagicMock()
        row = MagicMock()
        now = datetime.now(timezone.utc)
        existing = [(now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")] * 4
        row.extra_metadata = {"write_back_failure_timestamps": existing}

        with (
            patch("app.services.hubspot_service.get_integration", return_value=row),
            patch("app.services.hubspot_service._send_write_back_failure_alert") as mock_alert,
        ):
            hubspot_service.record_write_back_failure(db, _TENANT_ID, "HubSpot 500")

        assert len(row.extra_metadata["write_back_failure_timestamps"]) == 5
        mock_alert.assert_called_once()
        args = mock_alert.call_args[0]
        assert args[0] is db
        assert args[1] == _TENANT_ID
        assert args[2] == "HubSpot 500"
        assert args[4] == 5

    def test_noop_when_not_connected(self):
        db = MagicMock()
        with patch("app.services.hubspot_service.get_integration", return_value=None):
            hubspot_service.record_write_back_failure(db, _TENANT_ID, "err")
        db.commit.assert_not_called()

    def test_does_not_re_alert_on_every_failure_past_threshold(self):
        """Once a workspace is already over the 5-failure threshold, further
        failures must not re-send the admin alert on every single one."""
        db = MagicMock()
        row = MagicMock()
        now = datetime.now(timezone.utc)
        existing = [(now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")] * 5
        row.extra_metadata = {"write_back_failure_timestamps": existing}

        with (
            patch("app.services.hubspot_service.get_integration", return_value=row),
            patch("app.services.hubspot_service._send_write_back_failure_alert") as mock_alert,
        ):
            hubspot_service.record_write_back_failure(db, _TENANT_ID, "HubSpot 500")

        assert len(row.extra_metadata["write_back_failure_timestamps"]) == 6
        mock_alert.assert_not_called()


class TestSendWriteBackFailureAlert:
    def test_sends_email_to_workspace_admin(self):
        db = MagicMock()
        tenant = MagicMock()
        tenant.name = "Acme Corp"
        tenant.contact_email = "contact@acme.test"
        db.get.return_value = tenant

        with (
            patch(
                "app.services.data_export_service._get_workspace_admin_email",
                return_value="admin@acme.test",
            ),
            patch("app.services.email_service.email_service.send_generic_email") as mock_send,
        ):
            hubspot_service._send_write_back_failure_alert(
                db, _TENANT_ID, "HubSpot 500", "2026-07-20T17:50:31Z", 5
            )

        mock_send.assert_called_once()
        _, kwargs = mock_send.call_args
        assert kwargs["to_email"] == "admin@acme.test"
        assert "Acme Corp" in kwargs["subject"]
        assert "settings/integrations" in kwargs["html_body"]

    def test_falls_back_to_contact_email_when_no_admin(self):
        db = MagicMock()
        tenant = MagicMock()
        tenant.name = "Acme Corp"
        tenant.contact_email = "contact@acme.test"
        db.get.return_value = tenant

        with (
            patch(
                "app.services.data_export_service._get_workspace_admin_email",
                return_value=None,
            ),
            patch("app.services.email_service.email_service.send_generic_email") as mock_send,
        ):
            hubspot_service._send_write_back_failure_alert(
                db, _TENANT_ID, "HubSpot 500", "2026-07-20T17:50:31Z", 5
            )

        mock_send.assert_called_once()
        assert mock_send.call_args[1]["to_email"] == "contact@acme.test"

    def test_noop_when_no_email_found(self):
        db = MagicMock()
        db.get.return_value = None

        with (
            patch(
                "app.services.data_export_service._get_workspace_admin_email",
                return_value=None,
            ),
            patch("app.services.email_service.email_service.send_generic_email") as mock_send,
        ):
            hubspot_service._send_write_back_failure_alert(
                db, _TENANT_ID, "HubSpot 500", "2026-07-20T17:50:31Z", 5
            )

        mock_send.assert_not_called()

    def test_never_raises_on_internal_error(self):
        db = MagicMock()
        with patch(
            "app.services.data_export_service._get_workspace_admin_email",
            side_effect=Exception("DB down"),
        ):
            hubspot_service._send_write_back_failure_alert(
                db, _TENANT_ID, "HubSpot 500", "2026-07-20T17:50:31Z", 5
            )  # must not raise
