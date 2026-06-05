"""
Unit tests: LiveKit flow_id pass-through in voice_call_service.initiate_call.

Verifies that livekit_service.create_room receives the correct flow_id value
depending on whether callFlowId is present in the CallInitiateRequest.

All external dependencies (Twilio, DB, credits, LiveKit) are mocked — no
real servers are required.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixed UUIDs — stable across assertions
# ---------------------------------------------------------------------------

_AGENT_ID = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
_TENANT_ID = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000002")
_SESSION_ID = uuid.UUID("cccccccc-0000-0000-0000-000000000003")
_USER_ID = uuid.UUID("dddddddd-0000-0000-0000-000000000004")
_FLOW_ID = uuid.UUID("eeeeeeee-0000-0000-0000-000000000005")


# ---------------------------------------------------------------------------
# Object builders
# ---------------------------------------------------------------------------


def _call_request(**overrides):
    from app.schemas.twilio import CallInitiateRequest

    defaults: dict = dict(agentId=str(_AGENT_ID), userPhoneNumber="+15555550001")
    defaults.update(overrides)
    return CallInitiateRequest(**defaults)


def _agent():
    ag = MagicMock()
    ag.id = _AGENT_ID
    ag.name = "Test Agent"
    ag.model = MagicMock(model_name="gpt-4o")
    return ag


def _phone_number():
    pn = MagicMock()
    pn.id = uuid.uuid4()
    pn.phone_number = "+15555550000"
    pn.assistant_id = _AGENT_ID
    pn.twilio_account_sid = None
    pn.twilio_auth_token = None
    return pn


def _session(*, call_flow_id: uuid.UUID | None = None):
    cs = MagicMock()
    cs.id = _SESSION_ID
    cs.call_flow_id = call_flow_id
    cs.call_metadata = None
    return cs


def _db():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = _phone_number()
    return db


def _mock_settings(*, livekit_enabled: bool = True):
    s = MagicMock()
    s.LIVEKIT_ENABLED = livekit_enabled
    s.WEBHOOK_BASE_URL = "http://test.local"
    return s


# ---------------------------------------------------------------------------
# Helper: run initiate_call with all externals mocked, return create_room mock
# ---------------------------------------------------------------------------


async def _run(
    request,
    *,
    session_obj=None,
    livekit_enabled: bool = True,
) -> AsyncMock:
    """
    Call initiate_call with all external dependencies mocked.

    Returns the AsyncMock that was substituted for livekit_service.create_room
    so callers can assert on its invocation arguments.
    """
    from app.services.voice_call_service import initiate_call

    cs = session_obj or _session()
    lk_create_room = AsyncMock(return_value=SimpleNamespace(sid="RM_test"))

    mock_livekit = MagicMock()
    mock_livekit.create_room = lk_create_room
    mock_livekit.generate_agent_token = MagicMock(return_value="h.p.s")

    mock_http_req = MagicMock()
    mock_http_req.headers = {}

    mock_user = MagicMock()
    mock_user.current_tenant_id = _TENANT_ID
    mock_user.id = _USER_ID

    with (
        patch(
            "app.services.voice_call_service.verify_n8n_webhook_secret_async",
            AsyncMock(return_value=False),
        ),
        patch(
            "app.services.voice_call_service.agent_service",
            MagicMock(get_agent_by_id=MagicMock(return_value=_agent())),
        ),
        patch(
            "app.services.voice_call_service.twilio_service",
            MagicMock(
                validate_phone_number=MagicMock(return_value=True),
                make_call_with_credentials=MagicMock(
                    return_value=SimpleNamespace(sid="CA_test12345678")
                ),
            ),
        ),
        patch(
            "app.services.voice_call_service.credit_service",
            MagicMock(
                has_sufficient_credits=MagicMock(return_value=(True, 100, 10))
            ),
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
            _mock_settings(livekit_enabled=livekit_enabled),
        ),
        patch(
            "app.core.secret_manager.get_twilio_credentials",
            MagicMock(return_value=("AC_test", "token_test")),
        ),
    ):
        await initiate_call(request, mock_http_req, mock_user, _db())

    return lk_create_room


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlowIdPassThrough:
    @pytest.mark.asyncio
    async def test_flow_id_passed_when_call_flow_id_provided(self):
        """create_room receives the UUID from callFlowId when the field is set."""
        req = _call_request(callFlowId=str(_FLOW_ID))

        # Session starts with no flow_id; the service block will set it
        cs = _session(call_flow_id=None)

        # After the callFlowId block runs, call_session.call_flow_id becomes _FLOW_ID
        # We simulate this by letting the mock attribute be set naturally.
        mock = await _run(req, session_obj=cs)

        mock.assert_awaited_once()
        _, kwargs = mock.await_args
        assert kwargs["flow_id"] == _FLOW_ID, (
            f"Expected flow_id={_FLOW_ID!r}, got {kwargs['flow_id']!r}"
        )

    @pytest.mark.asyncio
    async def test_flow_id_none_when_call_flow_id_not_provided(self):
        """create_room receives flow_id=None when callFlowId is absent."""
        req = _call_request()  # no callFlowId
        cs = _session(call_flow_id=None)

        mock = await _run(req, session_obj=cs)

        mock.assert_awaited_once()
        _, kwargs = mock.await_args
        assert kwargs["flow_id"] is None, (
            f"Expected flow_id=None, got {kwargs['flow_id']!r}"
        )

    @pytest.mark.asyncio
    async def test_flow_id_from_existing_session_when_no_request_flow_id(self):
        """
        If the session already has call_flow_id (set by DB / prior code path)
        and callFlowId is not in the request, that value is forwarded.
        """
        req = _call_request()  # no callFlowId in request
        existing_flow = uuid.uuid4()
        cs = _session(call_flow_id=existing_flow)

        mock = await _run(req, session_obj=cs)

        mock.assert_awaited_once()
        _, kwargs = mock.await_args
        assert kwargs["flow_id"] == existing_flow

    @pytest.mark.asyncio
    async def test_create_room_not_called_when_livekit_disabled(self):
        """create_room must not be called when LIVEKIT_ENABLED=False."""
        req = _call_request(callFlowId=str(_FLOW_ID))

        mock = await _run(req, livekit_enabled=False)

        mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_call_flow_id_returns_400(self):
        """A non-UUID callFlowId must cause a 400 response, not crash."""
        from fastapi.responses import JSONResponse

        req = _call_request(callFlowId="not-a-valid-uuid")

        # _run wraps the HTTPException and returns a JSONResponse
        result = await _run_raw(req)
        assert isinstance(result, JSONResponse)
        assert result.status_code == 400


async def _run_raw(request) -> object:
    """Like _run but returns the raw return value of initiate_call."""
    from app.services.voice_call_service import initiate_call

    cs = _session(call_flow_id=None)
    mock_livekit = MagicMock()
    mock_livekit.create_room = AsyncMock(return_value=SimpleNamespace(sid="RM_test"))
    mock_livekit.generate_agent_token = MagicMock(return_value="h.p.s")

    mock_http_req = MagicMock()
    mock_http_req.headers = {}

    mock_user = MagicMock()
    mock_user.current_tenant_id = _TENANT_ID
    mock_user.id = _USER_ID

    with (
        patch(
            "app.services.voice_call_service.verify_n8n_webhook_secret_async",
            AsyncMock(return_value=False),
        ),
        patch(
            "app.services.voice_call_service.agent_service",
            MagicMock(get_agent_by_id=MagicMock(return_value=_agent())),
        ),
        patch(
            "app.services.voice_call_service.twilio_service",
            MagicMock(
                validate_phone_number=MagicMock(return_value=True),
                make_call_with_credentials=MagicMock(
                    return_value=SimpleNamespace(sid="CA_test12345678")
                ),
            ),
        ),
        patch(
            "app.services.voice_call_service.credit_service",
            MagicMock(
                has_sufficient_credits=MagicMock(return_value=(True, 100, 10))
            ),
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
            _mock_settings(livekit_enabled=True),
        ),
        patch(
            "app.core.secret_manager.get_twilio_credentials",
            MagicMock(return_value=("AC_test", "token_test")),
        ),
    ):
        return await initiate_call(request, mock_http_req, mock_user, _db())
