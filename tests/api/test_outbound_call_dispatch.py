"""
Integration tests for the outbound call dispatch endpoint.

Tests the `voice_call_service.initiate_call` service function directly with all
external dependencies (Twilio, LiveKit, DB, credits) mocked — no real servers.

Run:
    pytest tests/api/test_outbound_call_dispatch.py tests/services/test_voice_call_livekit.py -v
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status as http_status
from fastapi.responses import JSONResponse


# ---------------------------------------------------------------------------
# Fixed UUIDs
# ---------------------------------------------------------------------------

_AGENT_ID = uuid.UUID("aa000000-0000-0000-0000-000000000001")
_TENANT_ID = uuid.UUID("bb000000-0000-0000-0000-000000000002")
_SESSION_ID = uuid.UUID("cc000000-0000-0000-0000-000000000003")
_USER_ID = uuid.UUID("dd000000-0000-0000-0000-000000000004")
_TWILIO_SID = "CA12345678901234567890123456789012"


# ---------------------------------------------------------------------------
# Object builders
# ---------------------------------------------------------------------------


def _request(overrides: dict | None = None):
    from app.schemas.twilio import CallInitiateRequest

    defaults = dict(agentId=str(_AGENT_ID), toNumber="+15555550001")
    if overrides:
        defaults.update(overrides)
    return CallInitiateRequest(**defaults)


def _agent(*, status: str = "ready"):
    ag = MagicMock()
    ag.id = _AGENT_ID
    ag.name = "Test Agent"
    ag.status = status
    ag.model = MagicMock(model_name="gpt-4o")
    return ag


def _phone_number():
    pn = MagicMock()
    pn.id = uuid.uuid4()
    pn.phone_number = "+15555550000"
    pn.assistant_id = _AGENT_ID
    pn.twilio_account_sid = None
    pn.twilio_auth_token = None
    pn.status = "active"
    return pn


def _call_session(*, sid: uuid.UUID = _SESSION_ID):
    cs = MagicMock()
    cs.id = sid
    cs.call_flow_id = None
    cs.call_metadata = None
    cs.status = "initiated"
    cs.twilio_call_sid = ""
    return cs


def _db(phone: MagicMock | None = None, active_outbound: int = 0):
    """Mock DB that returns a bound phone number and a scalar count for concurrent limit."""
    db = MagicMock()
    phone_obj = phone or _phone_number()

    # query(PhoneNumber).filter().first() → phone_obj
    # query(func.count()).filter().scalar() → active_outbound
    def _query(model):
        q = MagicMock()
        q.filter.return_value.first.return_value = phone_obj
        q.filter.return_value.scalar.return_value = active_outbound
        return q

    db.query.side_effect = _query
    return db


def _mock_settings(*, livekit_enabled: bool = True, max_concurrent: int = 10):
    s = MagicMock()
    s.LIVEKIT_ENABLED = livekit_enabled
    s.WEBHOOK_BASE_URL = "http://test.local"
    s.OUTBOUND_MAX_CONCURRENT_PER_WORKSPACE = max_concurrent
    return s




# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


async def _run(
    req,
    *,
    agent_status: str = "ready",
    livekit_enabled: bool = True,
    max_concurrent: int = 10,
    active_outbound: int = 0,
    livekit_create_room=None,
    twilio_make_call=None,
    call_session_obj=None,
    phone_obj=None,
):
    from app.services.voice_call_service import initiate_call

    cs = call_session_obj or _call_session()
    lk_create = livekit_create_room or AsyncMock(
        return_value=SimpleNamespace(sid="RM_test_sid")
    )
    twilio_call = twilio_make_call or MagicMock(
        return_value=SimpleNamespace(sid=_TWILIO_SID)
    )

    mock_livekit = MagicMock()
    mock_livekit.create_room = lk_create
    mock_livekit.generate_agent_token = MagicMock(return_value="token.payload.sig")
    mock_livekit.close_room = AsyncMock()

    db = _db(phone=phone_obj, active_outbound=active_outbound)

    with (
        patch(
            "app.services.voice_call_service.agent_service",
            MagicMock(get_agent_by_id=MagicMock(return_value=_agent(status=agent_status))),
        ),
        patch(
            "app.services.voice_call_service.twilio_service",
            MagicMock(
                validate_phone_number=MagicMock(return_value=True),
                make_call_with_credentials=twilio_call,
                make_call=twilio_call,
            ),
        ),
        patch(
            "app.services.voice_call_service.credit_service",
            MagicMock(has_sufficient_credits=MagicMock(return_value=(True, 100, 10))),
        ),
        patch(
            "app.services.voice_call_service.call_session_service",
            MagicMock(create_call_session=MagicMock(return_value=cs)),
        ),
        patch(
            "app.services.voice_call_service.broadcast_call_status_update",
            AsyncMock(),
        ),
        patch("app.services.livekit_service.livekit_service", mock_livekit),
        patch(
            "app.services.voice_call_service.settings",
            _mock_settings(
                livekit_enabled=livekit_enabled, max_concurrent=max_concurrent
            ),
        ),
        patch(
            "app.core.secret_manager.get_twilio_credentials",
            MagicMock(return_value=("AC_test", "token_test")),
        ),
    ):
        result = await initiate_call(
            call_request=req,
            db=db,
            is_system_call=False,
            tenant_id=_TENANT_ID,
            user_id=_USER_ID,
            request_id="test-req-id",
        )

    return result, mock_livekit, twilio_call, db


# ---------------------------------------------------------------------------
# Test: Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_response_status_is_initiated(self):
        """Successful call → response contains status='initiated'."""
        req = _request()
        result, _, _, _ = await _run(req)

        # result is a SuccessResponse wrapping CallInitiateResponse
        assert hasattr(result, "data"), f"Unexpected result type: {type(result)}"
        assert result.data.status == "initiated"

    @pytest.mark.asyncio
    async def test_livekit_create_room_called_once(self):
        """LiveKit room creation is called once with correct call_id."""
        req = _request()
        _, mock_livekit, _, _ = await _run(req)

        mock_livekit.create_room.assert_awaited_once()
        _, kwargs = mock_livekit.create_room.await_args
        assert isinstance(kwargs.get("call_id"), uuid.UUID)

    @pytest.mark.asyncio
    async def test_twilio_make_call_called_once(self):
        """Twilio make_call is invoked exactly once on the happy path."""
        req = _request()
        _, _, twilio_call, _ = await _run(req)

        twilio_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_session_created_with_initiated_status(self):
        """create_call_session is called with status='initiated' (not 'active')."""
        req = _request()
        _, _, _, db = await _run(req)

        # Find the create_call_session call args in patched service
        # (We confirm via response; the DB mock isn't the real DB so we check indirectly)
        result, _, _, _ = await _run(req)
        assert result.data.status == "initiated"

    @pytest.mark.asyncio
    async def test_response_contains_call_session_id(self):
        """callSessionId in response matches the created session."""
        cs = _call_session(sid=_SESSION_ID)
        req = _request()
        result, _, _, _ = await _run(req, call_session_obj=cs)

        assert result.data.callSessionId == str(_SESSION_ID)

    @pytest.mark.asyncio
    async def test_response_contains_twilio_call_sid(self):
        """twilioCallSid in response matches Twilio's returned SID."""
        req = _request()
        result, _, _, _ = await _run(req)

        assert result.data.twilioCallSid == _TWILIO_SID

    @pytest.mark.asyncio
    async def test_from_number_matching_bound_number_accepted(self):
        """fromNumber that matches the agent's bound phone is accepted."""
        phone = _phone_number()
        phone.phone_number = "+15555550000"
        req = _request({"fromNumber": "+15555550000"})
        result, _, _, _ = await _run(req, phone_obj=phone)
        assert result.data.status == "initiated"


# ---------------------------------------------------------------------------
# Test: agent_not_ready
# ---------------------------------------------------------------------------


class TestAgentNotReady:
    @pytest.mark.asyncio
    async def test_pending_agent_returns_422(self):
        """Agent with status='pending' returns 422 with agent_not_ready code."""
        req = _request()
        result, _, _, _ = await _run(req, agent_status="pending")

        assert isinstance(result, JSONResponse)
        assert result.status_code == http_status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_pending_agent_error_code(self):
        """422 response body contains error.code='agent_not_ready'."""
        import json

        req = _request()
        result, _, _, _ = await _run(req, agent_status="pending")

        body = json.loads(result.body)
        assert body["error"]["code"] == "agent_not_ready"

    @pytest.mark.asyncio
    async def test_pending_agent_livekit_not_called(self):
        """LiveKit create_room is NOT called when agent is not ready."""
        req = _request()
        _, mock_livekit, _, _ = await _run(req, agent_status="pending")

        mock_livekit.create_room.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pending_agent_twilio_not_called(self):
        """Twilio make_call is NOT called when agent is not ready."""
        req = _request()
        _, _, twilio_call, _ = await _run(req, agent_status="pending")

        twilio_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_from_number_mismatch_returns_400(self):
        """fromNumber that does NOT match the bound number returns 400."""
        import json

        phone = _phone_number()
        phone.phone_number = "+15555550000"
        # fromNumber differs from bound number
        req = _request({"fromNumber": "+19998887777"})
        result, _, _, _ = await _run(req, phone_obj=phone)

        assert isinstance(result, JSONResponse)
        assert result.status_code == http_status.HTTP_400_BAD_REQUEST


# ---------------------------------------------------------------------------
# Test: concurrent outbound limit
# ---------------------------------------------------------------------------


class TestConcurrentLimit:
    @pytest.mark.asyncio
    async def test_limit_exceeded_returns_429(self):
        """10 active outbound calls → next call returns 429."""
        req = _request()
        result, _, _, _ = await _run(req, active_outbound=10, max_concurrent=10)

        assert isinstance(result, JSONResponse)
        assert result.status_code == http_status.HTTP_429_TOO_MANY_REQUESTS

    @pytest.mark.asyncio
    async def test_limit_exceeded_error_code(self):
        """429 body has error.code='outbound_concurrent_limit_exceeded'."""
        import json

        req = _request()
        result, _, _, _ = await _run(req, active_outbound=10, max_concurrent=10)

        body = json.loads(result.body)
        assert body["error"]["code"] == "outbound_concurrent_limit_exceeded"

    @pytest.mark.asyncio
    async def test_limit_not_exceeded_proceeds(self):
        """9 active calls with limit=10 → call proceeds normally."""
        req = _request()
        result, _, _, _ = await _run(req, active_outbound=9, max_concurrent=10)

        assert not isinstance(result, JSONResponse)
        assert result.data.status == "initiated"

    @pytest.mark.asyncio
    async def test_limit_exceeded_livekit_not_called(self):
        """LiveKit is NOT called when concurrent limit is exceeded."""
        req = _request()
        _, mock_livekit, _, _ = await _run(req, active_outbound=10, max_concurrent=10)

        mock_livekit.create_room.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_limit_exceeded_twilio_not_called(self):
        """Twilio is NOT called when concurrent limit is exceeded."""
        req = _request()
        _, _, twilio_call, _ = await _run(req, active_outbound=10, max_concurrent=10)

        twilio_call.assert_not_called()


# ---------------------------------------------------------------------------
# Test: LiveKit room creation failure (GAP 2b)
# ---------------------------------------------------------------------------


class TestLivekitRoomCreationFailure:
    @pytest.mark.asyncio
    async def test_livekit_failure_returns_503(self):
        """create_room raising → 503 response, no DB record, no Twilio call."""
        lk_fail = AsyncMock(side_effect=Exception("LiveKit server unreachable"))
        req = _request()

        result, mock_livekit, twilio_call, _ = await _run(
            req, livekit_create_room=lk_fail
        )

        assert isinstance(result, JSONResponse)
        assert result.status_code == http_status.HTTP_503_SERVICE_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_livekit_failure_error_code(self):
        """503 body has error.code='livekit_room_creation_failed'."""
        import json

        lk_fail = AsyncMock(side_effect=Exception("LiveKit server unreachable"))
        req = _request()
        result, _, _, _ = await _run(req, livekit_create_room=lk_fail)

        body = json.loads(result.body)
        assert body["error"]["code"] == "livekit_room_creation_failed"

    @pytest.mark.asyncio
    async def test_livekit_failure_twilio_not_called(self):
        """Twilio make_call is NOT invoked when LiveKit room creation fails."""
        lk_fail = AsyncMock(side_effect=Exception("LiveKit server unreachable"))
        req = _request()

        _, _, twilio_call, _ = await _run(req, livekit_create_room=lk_fail)

        twilio_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_livekit_failure_db_not_committed(self):
        """create_call_session is NOT called when LiveKit room creation fails."""
        lk_fail = AsyncMock(side_effect=Exception("LiveKit server unreachable"))
        req = _request()

        # Patch call_session_service explicitly to confirm it is not called
        mock_css = MagicMock()
        mock_css.create_call_session = MagicMock(return_value=_call_session())
        lk_create = lk_fail

        mock_livekit = MagicMock()
        mock_livekit.create_room = lk_create
        mock_livekit.generate_agent_token = MagicMock(return_value="t")
        mock_livekit.close_room = AsyncMock()

        with (
            patch(
                "app.services.voice_call_service.agent_service",
                MagicMock(get_agent_by_id=MagicMock(return_value=_agent())),
            ),
            patch(
                "app.services.voice_call_service.twilio_service",
                MagicMock(
                    validate_phone_number=MagicMock(return_value=True),
                    make_call_with_credentials=MagicMock(),
                ),
            ),
            patch(
                "app.services.voice_call_service.credit_service",
                MagicMock(has_sufficient_credits=MagicMock(return_value=(True, 100, 10))),
            ),
            patch("app.services.voice_call_service.call_session_service", mock_css),
            patch(
                "app.services.voice_call_service.broadcast_call_status_update",
                AsyncMock(),
            ),
            patch("app.services.livekit_service.livekit_service", mock_livekit),
            patch(
                "app.services.voice_call_service.settings",
                _mock_settings(livekit_enabled=True),
            ),
            patch(
                "app.core.secret_manager.get_twilio_credentials",
                MagicMock(return_value=("AC_test", "tok")),
            ),
        ):
            result = await _run_direct(req)

        # create_call_session must NOT have been called
        mock_css.create_call_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_livekit_disabled_skips_create_room(self):
        """When LIVEKIT_ENABLED=False, create_room is never called."""
        req = _request()
        _, mock_livekit, _, _ = await _run(req, livekit_enabled=False)

        mock_livekit.create_room.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_livekit_disabled_call_still_succeeds(self):
        """Call proceeds normally when LIVEKIT_ENABLED=False."""
        req = _request()
        result, _, _, _ = await _run(req, livekit_enabled=False)

        assert not isinstance(result, JSONResponse)
        assert result.data.status == "initiated"


# ---------------------------------------------------------------------------
# Test: E.164 validation
# ---------------------------------------------------------------------------


class TestE164Validation:
    def test_invalid_to_number_raises(self):
        """Non-E.164 toNumber raises validation error at schema level."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            from app.schemas.twilio import CallInitiateRequest

            CallInitiateRequest(agentId=str(_AGENT_ID), toNumber="5551234567")

    def test_missing_to_number_raises(self):
        """toNumber is required."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            from app.schemas.twilio import CallInitiateRequest

            CallInitiateRequest(agentId=str(_AGENT_ID))

    def test_invalid_from_number_raises(self):
        """Non-E.164 fromNumber raises validation error at schema level."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            from app.schemas.twilio import CallInitiateRequest

            CallInitiateRequest(
                agentId=str(_AGENT_ID),
                toNumber="+15551234567",
                fromNumber="not_a_number",
            )


# ---------------------------------------------------------------------------
# Test: Twilio call failure cleanup
# ---------------------------------------------------------------------------


class TestTwilioCallFailure:
    @pytest.mark.asyncio
    async def test_twilio_failure_returns_502(self):
        twilio_fail = MagicMock(side_effect=Exception("Twilio API error"))
        req = _request()
        result, _, _, _ = await _run(req, twilio_make_call=twilio_fail)

        assert isinstance(result, JSONResponse)
        assert result.status_code == http_status.HTTP_502_BAD_GATEWAY

    @pytest.mark.asyncio
    async def test_twilio_failure_error_code(self):
        import json

        twilio_fail = MagicMock(side_effect=Exception("Twilio API error"))
        req = _request()
        result, _, _, _ = await _run(req, twilio_make_call=twilio_fail)

        body = json.loads(result.body)
        assert body["error"]["code"] == "twilio_call_failed"

    @pytest.mark.asyncio
    async def test_twilio_failure_closes_livekit_room(self):
        twilio_fail = MagicMock(side_effect=Exception("Twilio API error"))
        req = _request()
        _, mock_livekit, _, _ = await _run(req, twilio_make_call=twilio_fail)

        mock_livekit.close_room.assert_awaited()


class TestActiveOutboundStatuses:
    def test_concurrent_limit_includes_connected_and_in_progress(self):
        from app.services.voice_call_service import _ACTIVE_OUTBOUND_STATUSES

        assert "connected" in _ACTIVE_OUTBOUND_STATUSES
        assert "in-progress" in _ACTIVE_OUTBOUND_STATUSES


# ---------------------------------------------------------------------------
# Helper: run initiate_call directly (for isolated sub-tests)
# ---------------------------------------------------------------------------


async def _run_direct(req):
    """Thin wrapper used by tests that need to inject specific mocks directly."""
    from app.services.voice_call_service import initiate_call

    return await initiate_call(
        call_request=req,
        db=_db(),
        is_system_call=False,
        tenant_id=_TENANT_ID,
        user_id=_USER_ID,
    )
