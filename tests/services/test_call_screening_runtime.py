"""Unit and runtime integration tests for Call Screening Detection.

Coverage:
  - Detection of automated call screener phrases (Google, iOS, Samsung, IVR)
  - Execution of hang_up action (ends call immediately with ended_reason='Call screener detected')
  - Execution of respond action (continues conversation without disconnecting)
  - Non-screener speech passes through normally
  - Fallback default behavior (respond) when no flow attached
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.voice.call_control_mixin import CallControlMixin


class DummyHostHandler(CallControlMixin):
    def __init__(self, db=None, call_session=None, call_flow=None, call_sid="CA123", stream_sid="MZ123"):
        self.db = db
        self.call_session = call_session
        self.call_flow = call_flow
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self._call_ended = False
        self._play_tts_message = AsyncMock()
        self._full_shutdown = AsyncMock()


class TestCallScreeningRuntime:
    @pytest.mark.parametrize(
        "phrase",
        [
            "The person you're calling is using a screening service, please say your name",
            "This is a screening service from google. Please state your name and why you're calling.",
            "I'm using google call screen to screen this call.",
            "Go ahead and say why you're calling.",
            "State your name and why you're calling.",
            "Who is calling and why?",
            "Please say your name and reason for calling after the tone.",
        ],
    )
    @pytest.mark.anyio
    async def test_screener_detected_with_hang_up_action_ends_call(self, phrase):
        flow = MagicMock(spec=CallFlow)
        flow.call_screening_action = "hang_up"

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)

        with (
            patch("app.voice.call_control_mixin.call_session_service.update_call_session_status") as mock_update,
            patch("app.voice.call_control_mixin.get_twilio_credentials_for_call", return_value=("AC123", "token")),
            patch("app.voice.call_control_mixin.twilio_service.end_call_with_credentials") as mock_end_call,
            patch("app.voice.call_control_mixin.broadcast_call_status_update") as mock_broadcast,
        ):
            result = await handler._check_and_handle_call_screener(phrase)

        assert result is True
        assert handler._call_ended is True
        mock_update.assert_called_once_with(
            handler.db, session.id, "completed", ended_reason="Call screener detected"
        )
        mock_end_call.assert_called_once_with("CA123", "AC123", "token")
        mock_broadcast.assert_called_once()

    @pytest.mark.parametrize(
        "phrase",
        [
            "The person you're calling is using a screening service, please say your name",
            "This is a screening service from google. Please state your name and why you're calling.",
            "I'm using google call screen to screen this call.",
            "Go ahead and say why you're calling.",
            "State your name and why you're calling.",
            "Who is calling and why?",
            "Please say your name and reason for calling after the tone.",
        ],
    )
    @pytest.mark.anyio
    async def test_screener_detected_with_respond_action_allows_call_to_proceed(self, phrase):
        flow = MagicMock(spec=CallFlow)
        flow.call_screening_action = "respond"

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        result = await handler._check_and_handle_call_screener(phrase)

        assert result is False
        assert handler._call_ended is False

    @pytest.mark.anyio
    async def test_normal_human_speech_does_not_trigger_screener_action(self):
        flow = MagicMock(spec=CallFlow)
        flow.call_screening_action = "hang_up"

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        result = await handler._check_and_handle_call_screener("Hello, this is John speaking. How can I help you today?")

        assert result is False
        assert handler._call_ended is False

    @pytest.mark.anyio
    async def test_fallback_when_no_flow_attached_defaults_to_respond(self):
        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = None

        handler = DummyHostHandler(call_session=session, call_flow=None)
        result = await handler._check_and_handle_call_screener("The person is using a screening service.")

        assert result is False
        assert handler._call_ended is False
