"""Unit and runtime integration tests for IVR Phone Tree and DTMF Keypad handling.

Coverage:
  - DTMF disabled ignores keypad events
  - DTMF enabled buffers digits and flushes on debounce timer
  - DTMF max digits limit and exceeded attempts hangup
  - IVR disabled ignores menu phrases
  - IVR enabled with hang_up action ends call on phone tree detection
  - IVR enabled with dial_through action allows AI conversation
  - IVR wait on hold logs and proceeds without hanging up
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.voice.call_control_mixin import CallControlMixin


class DummyHostHandler(CallControlMixin):
    def __init__(
        self,
        db=None,
        call_session=None,
        call_flow=None,
        call_sid="CA_TEST_123",
        stream_sid="MZ_TEST_123",
    ):
        self.db = db
        self.call_session = call_session
        self.call_flow = call_flow
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self._call_ended = False
        self._full_shutdown = AsyncMock()
        self.processed_transcripts: list[str] = []

    async def _process_transcript(self, transcript: str, confidence: float = 1.0, **kwargs):
        self.processed_transcripts.append(transcript)


class TestDTMFKeypadRuntime:
    @pytest.mark.anyio
    async def test_dtmf_disabled_ignores_digit_messages(self):
        flow = MagicMock(spec=CallFlow)
        flow.dtmf_enabled = False

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        await handler.handle_dtmf_message({"event": "dtmf", "dtmf": {"digit": "1"}})

        assert not hasattr(handler, "_dtmf_buffer") or handler._dtmf_buffer == ""
        assert len(handler.processed_transcripts) == 0

    @pytest.mark.anyio
    async def test_dtmf_enabled_buffers_and_flushes_on_debounce(self):
        flow = MagicMock(spec=CallFlow)
        flow.dtmf_enabled = True
        flow.dtmf_button_press_delay = 0  # instantaneous for test
        flow.dtmf_max_digits = 10
        flow.dtmf_allow_caller_interruption = False

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)

        # Send multiple digits
        await handler.handle_dtmf_message({"event": "dtmf", "dtmf": {"digit": "1"}})
        await handler.handle_dtmf_message({"event": "dtmf", "dtmf": {"digit": "2"}})
        await handler.handle_dtmf_message({"event": "dtmf", "dtmf": {"digit": "3"}})

        # Wait for debounce flush task
        if handler._dtmf_debounce_task:
            await handler._dtmf_debounce_task

        assert "User input DTMF: 123" in handler.processed_transcripts

    @pytest.mark.anyio
    async def test_dtmf_exceeded_attempts_ends_call(self):
        flow = MagicMock(spec=CallFlow)
        flow.dtmf_enabled = True
        flow.dtmf_button_press_delay = 2
        flow.dtmf_max_digits = 3
        flow.dtmf_allowed_exceeded_attempts = 1
        flow.dtmf_exceeded_action = "end_call"
        flow.dtmf_allow_caller_interruption = True

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)

        with (
            patch("app.voice.call_control_mixin.call_session_service.update_call_session_status") as mock_update,
            patch("app.voice.call_control_mixin.get_twilio_credentials_for_call", return_value=("AC123", "tok")),
            patch("app.voice.call_control_mixin.twilio_service.end_call_with_credentials") as mock_end_call,
        ):
            # Attempt 1: exceeds 3 digits (4 digits)
            for d in ["1", "2", "3", "4"]:
                await handler.handle_dtmf_message({"event": "dtmf", "dtmf": {"digit": d}})
            assert handler._dtmf_exceeded_count == 1
            assert not handler._call_ended

            # Attempt 2: exceeds 3 digits again -> count becomes 2 > allowed (1) -> ends call
            for d in ["5", "6", "7", "8"]:
                await handler.handle_dtmf_message({"event": "dtmf", "dtmf": {"digit": d}})

            assert handler._dtmf_exceeded_count == 2
            assert handler._call_ended is True
            mock_update.assert_called_once_with(
                handler.db, session.id, "completed", ended_reason="DTMF input limit exceeded"
            )
            mock_end_call.assert_called_once_with("CA_TEST_123", "AC123", "tok")

    @pytest.mark.anyio
    async def test_dtmf_exceeded_action_continue_does_not_end_call(self):
        flow = MagicMock(spec=CallFlow)
        flow.dtmf_enabled = True
        flow.dtmf_button_press_delay = 2
        flow.dtmf_max_digits = 2
        flow.dtmf_allowed_exceeded_attempts = 1
        flow.dtmf_exceeded_action = "continue"
        flow.dtmf_allow_caller_interruption = True

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)

        for d in ["1", "2", "3"]:
            await handler.handle_dtmf_message({"event": "dtmf", "digit": d})

        assert handler._dtmf_exceeded_count == 1
        assert handler._call_ended is False
        assert handler._dtmf_buffer == ""

    @pytest.mark.anyio
    async def test_dtmf_top_level_digit_format(self):
        flow = MagicMock(spec=CallFlow)
        flow.dtmf_enabled = True
        flow.dtmf_button_press_delay = 0
        flow.dtmf_max_digits = 10
        flow.dtmf_allow_caller_interruption = False

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        await handler.handle_dtmf_message({"event": "dtmf", "digit": "9"})

        if handler._dtmf_debounce_task:
            await handler._dtmf_debounce_task

        assert "User input DTMF: 9" in handler.processed_transcripts

    @pytest.mark.anyio
    async def test_dtmf_already_ended_call_ignored(self):
        flow = MagicMock(spec=CallFlow)
        flow.dtmf_enabled = True

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        handler._call_ended = True

        await handler.handle_dtmf_message({"event": "dtmf", "digit": "1"})
        assert not hasattr(handler, "_dtmf_buffer") or handler._dtmf_buffer == ""

    @pytest.mark.anyio
    async def test_dtmf_end_call_message_played_on_hangup(self):
        flow = MagicMock(spec=CallFlow)
        flow.dtmf_enabled = True
        flow.dtmf_button_press_delay = 2
        flow.dtmf_max_digits = 1
        flow.dtmf_allowed_exceeded_attempts = 0
        flow.dtmf_exceeded_action = "end_call"
        flow.dtmf_end_call_message = "Limit reached. Goodbye."

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        handler._play_tts_message = AsyncMock()

        with (
            patch("app.voice.call_control_mixin.call_session_service.update_call_session_status"),
            patch("app.voice.call_control_mixin.get_twilio_credentials_for_call", return_value=("AC123", "tok")),
            patch("app.voice.call_control_mixin.twilio_service.end_call_with_credentials"),
            patch("app.voice.call_control_mixin.broadcast_call_status_update") as mock_broadcast,
        ):
            await handler.handle_dtmf_message({"event": "dtmf", "digit": "1"})
            await handler.handle_dtmf_message({"event": "dtmf", "digit": "2"})

        assert handler._call_ended is True
        handler._play_tts_message.assert_awaited_once_with("Limit reached. Goodbye.")
        mock_broadcast.assert_called_once()


class TestIVRAndHoldRuntime:
    @pytest.mark.anyio
    async def test_ivr_disabled_bypasses_phone_tree_detection(self):
        flow = MagicMock(spec=CallFlow)
        flow.ivr_enabled = False

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        ended = await handler._check_and_handle_ivr_and_hold("For English, press 1")
        assert ended is False
        assert handler._call_ended is False

    @pytest.mark.anyio
    async def test_ivr_already_ended_call_ignored(self):
        flow = MagicMock(spec=CallFlow)
        flow.ivr_enabled = True
        flow.ivr_action = "hang_up"

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        handler._call_ended = True

        ended = await handler._check_and_handle_ivr_and_hold("Press 1 for English")
        assert ended is False

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "phrase",
        [
            "Press 1 for customer support, press 2 for sales",
            "Please listen carefully as our menu options have changed",
            "To speak with a representative, please press 0",
            "Main menu: dial the extension of the party you wish to reach",
        ],
    )
    async def test_ivr_enabled_hang_up_action_ends_call(self, phrase):
        flow = MagicMock(spec=CallFlow)
        flow.ivr_enabled = True
        flow.ivr_action = "hang_up"

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)

        with (
            patch("app.voice.call_control_mixin.call_session_service.update_call_session_status") as mock_update,
            patch("app.voice.call_control_mixin.get_twilio_credentials_for_call", return_value=("AC123", "tok")),
            patch("app.voice.call_control_mixin.twilio_service.end_call_with_credentials") as mock_end_call,
            patch("app.voice.call_control_mixin.broadcast_call_status_update") as mock_broadcast,
        ):
            ended = await handler._check_and_handle_ivr_and_hold(phrase)

        assert ended is True
        assert handler._call_ended is True
        mock_update.assert_called_once_with(
            handler.db, session.id, "completed", ended_reason="IVR phone tree detected"
        )
        mock_end_call.assert_called_once_with("CA_TEST_123", "AC123", "tok")
        mock_broadcast.assert_called_once()

    @pytest.mark.anyio
    async def test_ivr_enabled_dial_through_action_allows_call_to_proceed(self):
        flow = MagicMock(spec=CallFlow)
        flow.ivr_enabled = True
        flow.ivr_action = "dial_through"

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        ended = await handler._check_and_handle_ivr_and_hold("Press 1 for English, press 2 for Spanish")
        assert ended is False
        assert handler._call_ended is False

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "hold_phrase",
        [
            "All representatives are currently busy, please stay on the line",
            "Your call is important to us. Please hold for the next available representative",
            "Thank you for holding, an agent will be with you shortly",
        ],
    )
    async def test_ivr_wait_on_hold_allows_call_to_proceed(self, hold_phrase):
        flow = MagicMock(spec=CallFlow)
        flow.ivr_enabled = True
        flow.ivr_wait_on_hold = True
        flow.ivr_max_hold_time = 180

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        ended = await handler._check_and_handle_ivr_and_hold(hold_phrase)
        assert ended is False
        assert handler._call_ended is False
