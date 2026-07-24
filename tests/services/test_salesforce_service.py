"""
Tests for Salesforce CRM integration service.

External HTTP calls (OAuth token endpoint, SOQL query API, sobjects/Task API) are
mocked at the boundary (`_request_with_backoff`) per CLAUDE.md convention.

Coverage:
  1.  OAuth state: sign/verify round trip, tamper rejection, wrong purpose rejection
  2.  Authorization URL: client_id/scope/state present
  3.  Token refresh: returns cached (token, instance_url) when fresh; refreshes when
      expired; persists instance_url; returns None when no refresh_token; fails open
  4.  Contact lookup: correct {id, name, account, email} shape; Redis cache
      hit/miss/store; fails open on Salesforce error
  5.  CRM context block: exact "CRM CONTEXT (Salesforce): ..." format, cached on
      call_metadata after first fetch; fails open
  6.  Post-call write-back: creates a Task with the Gemini summary and
      CallDurationInSeconds; skips when write-back disabled; records/clears
      last_write_back_error; retries once after 5 minutes
  7.  Disconnect: revokes + deletes; returns False when not connected
  8.  429 backoff: retries with exponential delay, honors Retry-After
  9.  Phone normalization / search value variations
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services import salesforce_service


_TENANT_ID = uuid.UUID("aa200000-0000-0000-0000-000000000001")
_SESSION_ID = uuid.UUID("bb200000-0000-0000-0000-000000000002")


# ── OAuth state ────────────────────────────────────────────────────────────────


class TestOAuthState:
    def test_roundtrip(self):
        state = salesforce_service.build_oauth_state(_TENANT_ID)
        assert salesforce_service.verify_oauth_state(state) == _TENANT_ID

    def test_tampered_state_rejected(self):
        state = salesforce_service.build_oauth_state(_TENANT_ID)
        with pytest.raises(ValueError):
            salesforce_service.verify_oauth_state(state + "x")

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
            salesforce_service.verify_oauth_state(bad_state)

    def test_expired_state_rejected(self):
        from jose import jwt as jose_jwt
        from app.core.config import settings

        expired_state = jose_jwt.encode(
            {
                "tenant_id": str(_TENANT_ID),
                "purpose": "salesforce_oauth_state",
                "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
            },
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        with pytest.raises(ValueError):
            salesforce_service.verify_oauth_state(expired_state)


class TestAuthorizationUrl:
    def test_contains_client_id_scope_and_state(self):
        with patch(
            "app.services.salesforce_service.get_salesforce_oauth_credentials",
            return_value=("client-123", "secret-456"),
        ):
            state = salesforce_service.build_oauth_state(_TENANT_ID)
            url = salesforce_service.build_authorization_url(state)

        assert url.startswith("https://login.salesforce.com/services/oauth2/authorize")
        assert "client_id=client-123" in url
        assert "scope=api" in url or "scope=api+refresh_token" in url
        assert f"state={state}" in url


# ── Token refresh ──────────────────────────────────────────────────────────────


def _integration_row(*, expires_in_seconds: float, has_refresh_token: bool = True, instance_url="https://acme.my.salesforce.com"):
    row = MagicMock()
    row.access_token = "encrypted-access-token"
    row.refresh_token = "encrypted-refresh-token" if has_refresh_token else None
    row.token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in_seconds)
    row.extra_metadata = {"instance_url": instance_url} if instance_url else {}
    return row


class TestTokenRefresh:
    @pytest.mark.anyio
    async def test_returns_existing_token_when_not_expired(self):
        row = _integration_row(expires_in_seconds=600)
        db = MagicMock()

        with (
            patch("app.services.salesforce_service.get_integration", return_value=row),
            patch(
                "app.services.salesforce_service.decrypt_salesforce_token",
                return_value="plain-access-token",
            ) as mock_decrypt,
        ):
            result = await salesforce_service.get_valid_access_token(db, _TENANT_ID)

        assert result == ("plain-access-token", "https://acme.my.salesforce.com")
        mock_decrypt.assert_called_once_with(row.access_token, db)

    @pytest.mark.anyio
    async def test_refreshes_when_expired(self):
        row = _integration_row(expires_in_seconds=-60)
        db = MagicMock()
        new_row = _integration_row(expires_in_seconds=1800, instance_url="https://acme.my.salesforce.com")

        with (
            patch("app.services.salesforce_service.get_integration", return_value=row),
            patch(
                "app.services.salesforce_service.decrypt_salesforce_token",
                side_effect=["plain-refresh-token", "new-plain-access-token"],
            ),
            patch(
                "app.services.salesforce_service.refresh_access_token",
                new=AsyncMock(
                    return_value={
                        "access_token": "new",
                        "instance_url": "https://acme.my.salesforce.com",
                    }
                ),
            ) as mock_refresh,
            patch("app.services.salesforce_service.upsert_tokens", return_value=new_row) as mock_upsert,
        ):
            result = await salesforce_service.get_valid_access_token(db, _TENANT_ID)

        mock_refresh.assert_awaited_once_with("plain-refresh-token")
        mock_upsert.assert_called_once()
        assert result == ("new-plain-access-token", "https://acme.my.salesforce.com")

    @pytest.mark.anyio
    async def test_returns_none_when_expired_and_no_refresh_token(self):
        row = _integration_row(expires_in_seconds=-60, has_refresh_token=False)
        db = MagicMock()

        with patch("app.services.salesforce_service.get_integration", return_value=row):
            result = await salesforce_service.get_valid_access_token(db, _TENANT_ID)

        assert result is None

    @pytest.mark.anyio
    async def test_returns_none_when_no_integration(self):
        db = MagicMock()
        with patch("app.services.salesforce_service.get_integration", return_value=None):
            result = await salesforce_service.get_valid_access_token(db, _TENANT_ID)
        assert result is None

    @pytest.mark.anyio
    async def test_returns_none_when_no_instance_url_stored(self):
        row = _integration_row(expires_in_seconds=600, instance_url=None)
        db = MagicMock()
        with patch("app.services.salesforce_service.get_integration", return_value=row):
            result = await salesforce_service.get_valid_access_token(db, _TENANT_ID)
        assert result is None

    @pytest.mark.anyio
    async def test_refresh_failure_fails_open(self):
        row = _integration_row(expires_in_seconds=-60)
        db = MagicMock()

        with (
            patch("app.services.salesforce_service.get_integration", return_value=row),
            patch(
                "app.services.salesforce_service.decrypt_salesforce_token",
                return_value="plain-refresh-token",
            ),
            patch(
                "app.services.salesforce_service.refresh_access_token",
                new=AsyncMock(side_effect=Exception("Salesforce down")),
            ),
        ):
            result = await salesforce_service.get_valid_access_token(db, _TENANT_ID)

        assert result is None


class TestForceRefreshAccessToken:
    @pytest.mark.anyio
    async def test_always_refreshes_even_when_not_expired(self):
        row = _integration_row(expires_in_seconds=600)
        db = MagicMock()
        new_row = _integration_row(expires_in_seconds=1800)

        with (
            patch("app.services.salesforce_service.get_integration", return_value=row),
            patch(
                "app.services.salesforce_service.decrypt_salesforce_token",
                side_effect=["plain-refresh-token", "new-token"],
            ),
            patch(
                "app.services.salesforce_service.refresh_access_token",
                new=AsyncMock(return_value={"access_token": "new", "instance_url": "https://acme.my.salesforce.com"}),
            ) as mock_refresh,
            patch("app.services.salesforce_service.upsert_tokens", return_value=new_row),
        ):
            result = await salesforce_service._force_refresh_access_token(db, _TENANT_ID)

        mock_refresh.assert_awaited_once()
        assert result == ("new-token", "https://acme.my.salesforce.com")

    @pytest.mark.anyio
    async def test_fails_open_when_refresh_errors(self):
        row = _integration_row(expires_in_seconds=600)
        db = MagicMock()

        with (
            patch("app.services.salesforce_service.get_integration", return_value=row),
            patch(
                "app.services.salesforce_service.decrypt_salesforce_token",
                return_value="plain-refresh-token",
            ),
            patch(
                "app.services.salesforce_service.refresh_access_token",
                new=AsyncMock(side_effect=Exception("down")),
            ),
        ):
            result = await salesforce_service._force_refresh_access_token(db, _TENANT_ID)

        assert result is None

    @pytest.mark.anyio
    async def test_falls_back_to_get_valid_access_token_when_no_refresh_token(self):
        row = _integration_row(expires_in_seconds=600, has_refresh_token=False)
        db = MagicMock()

        with (
            patch("app.services.salesforce_service.get_integration", return_value=row),
            patch(
                "app.services.salesforce_service.decrypt_salesforce_token",
                return_value="plain-access-token",
            ),
        ):
            result = await salesforce_service._force_refresh_access_token(db, _TENANT_ID)

        assert result == ("plain-access-token", "https://acme.my.salesforce.com")


# ── Contact lookup ──────────────────────────────────────────────────────────────


_RAW_CONTACT = {
    "Id": "003xx000004TmiQAAS",
    "Name": "Ada Lovelace",
    "Email": "ada@example.com",
    "Account": {"Name": "Acme Corp"},
}


class TestContactShape:
    def test_contact_dict_from_salesforce_shape(self):
        contact = salesforce_service._contact_dict_from_salesforce(_RAW_CONTACT)
        assert contact == {
            "id": "003xx000004TmiQAAS",
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "account": "Acme Corp",
        }

    def test_missing_account_resolves_to_none(self):
        raw = {"Id": "1", "Name": "A B", "Email": "a@b.com"}
        contact = salesforce_service._contact_dict_from_salesforce(raw)
        assert contact["account"] is None


class TestGetContactForPhone:
    @pytest.mark.anyio
    async def test_cache_hit_skips_salesforce_call(self):
        db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(
            return_value='{"id": "1", "name": "Cached Person", "email": null, "account": null}'
        )

        with (
            patch("app.services.salesforce_service.get_redis", return_value=mock_redis),
            patch("app.services.salesforce_service.get_valid_access_token") as mock_token,
        ):
            contact = await salesforce_service.get_contact_for_phone(db, _TENANT_ID, "+15550001111")

        mock_token.assert_not_called()
        assert contact["name"] == "Cached Person"

    @pytest.mark.anyio
    async def test_cache_miss_fetches_and_stores(self):
        db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.set = AsyncMock()

        with (
            patch("app.services.salesforce_service.get_redis", return_value=mock_redis),
            patch(
                "app.services.salesforce_service.get_valid_access_token",
                new=AsyncMock(return_value=("access-token", "https://acme.my.salesforce.com")),
            ),
            patch(
                "app.services.salesforce_service.search_contact_by_phone",
                new=AsyncMock(return_value=_RAW_CONTACT),
            ),
        ):
            contact = await salesforce_service.get_contact_for_phone(db, _TENANT_ID, "+15550001111")

        assert contact["id"] == "003xx000004TmiQAAS"
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
            patch("app.services.salesforce_service.get_redis", return_value=mock_redis),
            patch(
                "app.services.salesforce_service.get_valid_access_token",
                new=AsyncMock(return_value=("access-token", "https://acme.my.salesforce.com")),
            ),
            patch(
                "app.services.salesforce_service.search_contact_by_phone",
                new=AsyncMock(return_value=None),
            ),
        ):
            contact = await salesforce_service.get_contact_for_phone(db, _TENANT_ID, "+15550001111")

        assert contact is None
        mock_redis.set.assert_awaited_once_with(
            salesforce_service._contact_cache_key(_TENANT_ID, "+15550001111"),
            salesforce_service._CONTACT_NOT_FOUND_SENTINEL,
            ex=300,
        )

    @pytest.mark.anyio
    async def test_fails_open_on_salesforce_error(self):
        """A Salesforce outage during contact search must never raise — call proceeds without CRM data."""
        db = MagicMock()

        with (
            patch("app.services.salesforce_service.get_redis", return_value=None),
            patch(
                "app.services.salesforce_service.get_valid_access_token",
                new=AsyncMock(return_value=("access-token", "https://acme.my.salesforce.com")),
            ),
            patch(
                "app.services.salesforce_service.search_contact_by_phone",
                new=AsyncMock(side_effect=Exception("Salesforce 500")),
            ),
        ):
            contact = await salesforce_service.get_contact_for_phone(db, _TENANT_ID, "+15550001111")

        assert contact is None

    @pytest.mark.anyio
    async def test_no_access_token_returns_none(self):
        db = MagicMock()

        with (
            patch("app.services.salesforce_service.get_redis", return_value=None),
            patch(
                "app.services.salesforce_service.get_valid_access_token",
                new=AsyncMock(return_value=None),
            ),
        ):
            contact = await salesforce_service.get_contact_for_phone(db, _TENANT_ID, "+15550001111")

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
            metadata={"salesforce_crm_context": "CRM CONTEXT (Salesforce): cached"}
        )

        with patch(
            "app.services.salesforce_service.tenant_has_salesforce_connected"
        ) as mock_connected:
            block = await salesforce_service.get_crm_context_block_for_call(db, call_session)

        mock_connected.assert_not_called()
        assert block == "CRM CONTEXT (Salesforce): cached"

    @pytest.mark.anyio
    async def test_fetches_and_caches_on_first_call(self):
        db = MagicMock()
        call_session = _call_session(metadata={})
        contact = {"id": "1", "name": "Ada Lovelace", "account": "Acme Corp", "email": "ada@example.com"}

        with (
            patch(
                "app.services.salesforce_service.tenant_has_salesforce_connected", return_value=True
            ),
            patch(
                "app.services.salesforce_service.get_contact_for_phone",
                new=AsyncMock(return_value=contact),
            ) as mock_get_contact,
        ):
            block = await salesforce_service.get_crm_context_block_for_call(db, call_session)

        assert block == (
            "CRM CONTEXT (Salesforce): Name: Ada Lovelace, "
            "Account: Acme Corp, Email: ada@example.com"
        )
        assert call_session.call_metadata["salesforce_crm_context"] == block
        db.flush.assert_called_once()
        db.commit.assert_not_called()
        assert mock_get_contact.call_args.kwargs.get("commit_lookup_timestamp") is False

    @pytest.mark.anyio
    async def test_not_connected_returns_empty_block(self):
        db = MagicMock()
        call_session = _call_session(metadata={})

        with patch(
            "app.services.salesforce_service.tenant_has_salesforce_connected", return_value=False
        ):
            block = await salesforce_service.get_crm_context_block_for_call(db, call_session)

        assert block == ""

    @pytest.mark.anyio
    async def test_fails_open_on_exception(self):
        db = MagicMock()
        call_session = _call_session(metadata={})

        with patch(
            "app.services.salesforce_service.tenant_has_salesforce_connected",
            side_effect=Exception("DB down"),
        ):
            block = await salesforce_service.get_crm_context_block_for_call(db, call_session)

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
    async def test_creates_task_with_summary(self):
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
                "app.services.salesforce_service.get_integration_settings",
                return_value=self._writeback_settings(),
            ),
            patch(
                "app.services.salesforce_service._force_refresh_access_token",
                new=AsyncMock(return_value=("access-token", "https://acme.my.salesforce.com")),
            ) as mock_force_refresh,
            patch(
                "app.services.salesforce_service.get_contact_for_phone",
                new=AsyncMock(return_value=contact),
            ),
            patch(
                "app.services.salesforce_service.generate_transcript_summary",
                return_value="Caller asked about pricing. Agent booked a follow-up demo.",
            ),
            patch(
                "app.services.salesforce_service.create_call_task",
                new=AsyncMock(return_value={"id": "task-1"}),
            ) as mock_create_task,
            patch("app.services.salesforce_service.set_last_write_back_error") as mock_set_error,
        ):
            await salesforce_service._run_post_call_writeback_async(db, call_session)

        mock_force_refresh.assert_awaited_once_with(db, _TENANT_ID)
        mock_create_task.assert_awaited_once()
        args, kwargs = mock_create_task.call_args
        assert args[0] == "access-token"
        assert args[1] == "https://acme.my.salesforce.com"
        assert args[2] == "contact-1"
        assert kwargs["duration_seconds"] == 120
        assert "pricing" in kwargs["description"]
        assert kwargs["call_type"] == "outbound"
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
                "app.services.salesforce_service.get_integration_settings",
                return_value=self._writeback_settings(write_back_enabled=False),
            ),
            patch("app.services.salesforce_service._force_refresh_access_token") as mock_refresh,
            patch("app.services.salesforce_service.create_call_task") as mock_create,
        ):
            await salesforce_service._run_post_call_writeback_async(db, call_session)

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
                "app.services.salesforce_service.get_integration_settings",
                return_value=self._writeback_settings(),
            ),
            patch(
                "app.services.salesforce_service._force_refresh_access_token",
                new=AsyncMock(return_value=("access-token", "https://acme.my.salesforce.com")),
            ),
            patch(
                "app.services.salesforce_service.get_contact_for_phone",
                new=AsyncMock(return_value=None),
            ),
            patch("app.services.salesforce_service.create_call_task") as mock_create,
        ):
            await salesforce_service._run_post_call_writeback_async(db, call_session)

        mock_create.assert_not_called()

    @pytest.mark.anyio
    async def test_writeback_retries_once_after_5_minutes_then_records_structured_error(self):
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
                "app.services.salesforce_service.get_integration_settings",
                return_value=self._writeback_settings(),
            ),
            patch(
                "app.services.salesforce_service._force_refresh_access_token",
                new=AsyncMock(return_value=("access-token", "https://acme.my.salesforce.com")),
            ),
            patch(
                "app.services.salesforce_service.get_contact_for_phone",
                new=AsyncMock(return_value=contact),
            ),
            patch(
                "app.services.salesforce_service.generate_transcript_summary",
                return_value="Summary.",
            ),
            patch(
                "app.services.salesforce_service.create_call_task",
                new=AsyncMock(side_effect=Exception("Salesforce 500")),
            ) as mock_create_task,
            patch(
                "asyncio.sleep",
                new=AsyncMock(side_effect=lambda *_a, **_k: call_order.append("sleep")),
            ) as mock_sleep,
            patch("app.services.salesforce_service.record_write_back_failure") as mock_record_failure,
        ):
            await salesforce_service._run_post_call_writeback_async(db, call_session)

        assert mock_create_task.await_count == 2
        mock_sleep.assert_awaited_once_with(salesforce_service._WRITE_BACK_RETRY_DELAY_SECONDS)
        mock_record_failure.assert_called_once_with(db, _TENANT_ID, "Salesforce 500")
        db.rollback.assert_called_once()
        assert call_order == ["rollback", "sleep"]

    def test_run_post_call_writeback_skips_when_not_connected(self):
        db = MagicMock()
        call_session = MagicMock()
        call_session.tenant_id = _TENANT_ID
        call_session.customer_phone_number = "+15550001111"
        db.query.return_value.filter.return_value.first.return_value = call_session

        with (
            patch("app.services.salesforce_service.SessionLocal", return_value=db),
            patch(
                "app.services.salesforce_service.tenant_has_salesforce_connected",
                return_value=False,
            ),
            patch("asyncio.run") as mock_asyncio_run,
        ):
            salesforce_service.run_post_call_writeback(_SESSION_ID)

        mock_asyncio_run.assert_not_called()
        db.close.assert_called_once()

    def test_schedule_writeback_never_blocks_caller_without_a_running_loop(self):
        with (
            patch(
                "app.services.salesforce_service.asyncio.get_running_loop",
                side_effect=RuntimeError("no running loop"),
            ),
            patch("app.services.salesforce_service.threading.Thread") as mock_thread_cls,
            patch("app.services.salesforce_service.run_post_call_writeback") as mock_run,
        ):
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            salesforce_service.schedule_salesforce_writeback(_SESSION_ID)

        mock_thread_cls.assert_called_once_with(
            target=mock_run, args=(_SESSION_ID,), daemon=True
        )
        mock_thread.start.assert_called_once()
        mock_run.assert_not_called()


# ── Disconnect ──────────────────────────────────────────────────────────────────


class TestDisconnect:
    @pytest.mark.anyio
    async def test_revokes_and_deletes_row(self):
        db = MagicMock()
        row = MagicMock()
        row.refresh_token = "encrypted-refresh-token"

        mock_response = MagicMock()
        with (
            patch("app.services.salesforce_service.get_integration", return_value=row),
            patch(
                "app.services.salesforce_service.decrypt_salesforce_token",
                return_value="plain-refresh-token",
            ),
            patch(
                "app.services.salesforce_service._request_with_backoff",
                new=AsyncMock(return_value=mock_response),
            ) as mock_request,
        ):
            result = await salesforce_service.disconnect(db, _TENANT_ID)

        assert result is True
        db.delete.assert_called_once_with(row)
        db.commit.assert_called_once()
        mock_request.assert_awaited_once()
        assert mock_request.call_args[0][0] == "POST"
        assert "revoke" in mock_request.call_args[0][1]

    @pytest.mark.anyio
    async def test_returns_false_when_not_connected(self):
        db = MagicMock()
        with patch("app.services.salesforce_service.get_integration", return_value=None):
            result = await salesforce_service.disconnect(db, _TENANT_ID)
        assert result is False
        db.delete.assert_not_called()

    @pytest.mark.anyio
    async def test_revoke_failure_still_deletes_local_row(self):
        db = MagicMock()
        row = MagicMock()
        row.refresh_token = "encrypted-refresh-token"

        with (
            patch("app.services.salesforce_service.get_integration", return_value=row),
            patch(
                "app.services.salesforce_service.decrypt_salesforce_token",
                return_value="plain-refresh-token",
            ),
            patch(
                "app.services.salesforce_service._request_with_backoff",
                new=AsyncMock(side_effect=Exception("Salesforce down")),
            ),
        ):
            result = await salesforce_service.disconnect(db, _TENANT_ID)

        assert result is True
        db.delete.assert_called_once_with(row)
        db.commit.assert_called_once()


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
                "app.services.salesforce_service.httpx.AsyncClient",
                return_value=_FakeAsyncClient([rate_limited, ok]),
            ),
            patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
        ):
            response = await salesforce_service._request_with_backoff("GET", "https://x")

        assert response.status_code == 200
        mock_sleep.assert_awaited_once_with(1.0)

    @pytest.mark.anyio
    async def test_honors_retry_after_header(self):
        rate_limited = MagicMock(status_code=429, headers={"Retry-After": "3"})
        ok = MagicMock(status_code=200, headers={})

        with (
            patch(
                "app.services.salesforce_service.httpx.AsyncClient",
                return_value=_FakeAsyncClient([rate_limited, ok]),
            ),
            patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
        ):
            await salesforce_service._request_with_backoff("GET", "https://x")

        mock_sleep.assert_awaited_once_with(3.0)

    @pytest.mark.anyio
    async def test_gives_up_after_max_retries(self):
        responses = [MagicMock(status_code=429, headers={}) for _ in range(6)]

        with (
            patch(
                "app.services.salesforce_service.httpx.AsyncClient",
                return_value=_FakeAsyncClient(responses),
            ),
            patch("asyncio.sleep", new=AsyncMock()),
        ):
            response = await salesforce_service._request_with_backoff("GET", "https://x")

        assert response.status_code == 429


# ── Phone normalization ───────────────────────────────────────────────────────────


class TestCallTypeMapping:
    def test_maps_inbound_and_outbound(self):
        assert salesforce_service._sf_call_type("inbound") == "Inbound"
        assert salesforce_service._sf_call_type("Inbound") == "Inbound"
        assert salesforce_service._sf_call_type("outbound") == "Outbound"
        assert salesforce_service._sf_call_type(None) == "Outbound"
        assert salesforce_service._sf_call_type("") == "Outbound"


class TestPhoneNormalization:
    def test_normalize_to_e164(self):
        assert salesforce_service.normalize_to_e164("5550001111") == "+15550001111"
        assert salesforce_service.normalize_to_e164("15550001111") == "+15550001111"
        assert salesforce_service.normalize_to_e164("+15550001111") == "+15550001111"
        assert salesforce_service.normalize_to_e164("") == ""

    def test_get_phone_search_values(self):
        values = salesforce_service._get_phone_search_values("+15550001111")
        assert "+15550001111" in values
        assert "5550001111" in values
        assert "15550001111" in values


# ── Sync status / settings ─────────────────────────────────────────────────────


class TestIntegrationSettings:
    def test_not_connected_returns_disconnected_defaults(self):
        db = MagicMock()
        with patch("app.services.salesforce_service.get_integration", return_value=None):
            result = salesforce_service.get_integration_settings(db, _TENANT_ID)
        assert result["connected"] is False
        assert result["write_back_enabled"] is True

    def test_update_settings_raises_when_not_connected(self):
        db = MagicMock()
        with patch("app.services.salesforce_service.get_integration", return_value=None):
            with pytest.raises(ValueError):
                salesforce_service.update_integration_settings(db, _TENANT_ID, write_back_enabled=False)


class TestSyncStatus:
    def test_returns_defaults_when_not_connected(self):
        db = MagicMock()
        with patch("app.services.salesforce_service.get_integration", return_value=None):
            result = salesforce_service.get_sync_status(db, _TENANT_ID)
        assert result == {
            "last_lookup_at": None,
            "last_write_back_at": None,
            "last_write_back_status": None,
            "error_count_24h": 0,
        }


class TestSafeErrorMsg:
    def test_redacts_bearer_token(self):
        exc = Exception("failed with Bearer abc123.def456==")
        msg = salesforce_service._safe_error_msg(exc)
        assert "abc123" not in msg
        assert "Bearer [redacted]" in msg
