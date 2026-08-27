"""Unit and runtime integration tests for Call Screening Detection.

Coverage:
  - Detection of automated call screener phrases (Google, iOS, Samsung, IVR)
  - Execution of hang_up action (ends call immediately with ended_reason='Call screener detected')
  - Execution of respond action (continues conversation without disconnecting)
  - Non-screener speech passes through normally
  - Fallback default behavior (respond) when no flow attached
"""

from __future__ import annotations

import asyncio
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

    @pytest.mark.parametrize(
        "phrase",
        [
            "this is not a screening service",
            "we are not using a screening service",
            "this isn't a screening service from google",
            "no screening service is enabled",
        ],
    )
    @pytest.mark.anyio
    async def test_negated_screener_phrase_does_not_trigger_hangup(self, phrase):
        flow = MagicMock(spec=CallFlow)
        flow.call_screening_action = "hang_up"

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        result = await handler._check_and_handle_call_screener(phrase)

        assert result is False
        assert handler._call_ended is False

    @pytest.mark.anyio
    async def test_db_error_on_lazy_callflow_fetch_returns_false(self):
        from sqlalchemy.exc import SQLAlchemyError

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        mock_db = MagicMock()
        mock_db.execute.side_effect = SQLAlchemyError("Database connection dropped")
        mock_db.get.side_effect = SQLAlchemyError("Database connection dropped")

        handler = DummyHostHandler(db=mock_db, call_session=session, call_flow=None)
        result = await handler._check_and_handle_call_screener(
            "This is a screening service from google. Please state your name."
        )

        assert result is False
        assert handler._call_ended is False

    @pytest.mark.anyio
    async def test_livekit_browser_call_handler_call_screening_detection(self):
        from app.voice.livekit_browser_call_handler import LiveKitBrowserCallHandler

        flow = MagicMock(spec=CallFlow)
        flow.call_screening_action = "hang_up"

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()
        session.user_id = uuid.uuid4()
        session.agent_id = uuid.uuid4()
        session.tenant_id = uuid.uuid4()

        agent = MagicMock()
        agent.id = session.agent_id

        mock_db = MagicMock()

        handler = LiveKitBrowserCallHandler(
            db=mock_db, call_session=session, agent=agent, call_flow=flow
        )
        handler._add_to_transcript = AsyncMock()
        handler._update_booking_memory_from_user_turn = MagicMock()
        handler._complete_llm_turn_after_stt_final = AsyncMock()

        with (
            patch("app.voice.call_control_mixin.call_session_service.update_call_session_status") as mock_update,
            patch("app.voice.call_control_mixin.broadcast_call_status_update") as mock_broadcast,
        ):
            await handler._process_transcript(
                "This is a screening service from google. Please state your name and why you're calling.",
                confidence=0.95,
            )

        await asyncio.sleep(0.01)
        assert handler._call_ended is True
        assert handler._stop_event.is_set() is True
        mock_update.assert_called_once_with(
            mock_db, session.id, "completed", ended_reason="Call screener detected"
        )
        mock_broadcast.assert_called_once()
        handler._add_to_transcript.assert_not_called()
        handler._complete_llm_turn_after_stt_final.assert_not_called()
