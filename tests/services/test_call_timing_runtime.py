"""Unit and runtime integration tests for Call Timing and Silence Detection watchdog.

Coverage:
  - Silence watchdog plays reminder message upon silence timeout
  - Silence watchdog cycles through custom reminder messages
  - Client speech cancels and resets silence watchdog timer
  - DTMF keypress cancels and resets silence watchdog timer
  - Silence watchdog terminates call after final reminder retries exhausted
  - Max call duration watchdog plays departure message and terminates call
  - Already-ended calls are safely ignored
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
    def __init__(
        self,
        db=None,
        call_session=None,
        call_flow=None,
        call_sid="CA_TIMING_123",
        stream_sid="MZ_TIMING_123",
    ):
        self.db = db
        self.call_session = call_session
        self.call_flow = call_flow
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self._call_ended = False
        self._is_tts_playing = False
        self._silence_watchdog_task = None
        self._max_duration_task = None
        self._silence_retry_count = 0
        self._full_shutdown = AsyncMock()
        self._play_tts_message = AsyncMock()
        self.processed_transcripts: list[str] = []

    async def _process_transcript(
        self, transcript: str, confidence: float = 1.0, **kwargs
    ):
        self._cancel_silence_watchdog()
        self.processed_transcripts.append(transcript)


class TestSilenceWatchdogRuntime:
    @pytest.mark.anyio
    async def test_silence_watchdog_plays_first_reminder(self):
        flow = MagicMock(spec=CallFlow)
        flow.silence_timeout = 0  # Instantaneous for test
        flow.reminder_retries = 1
        flow.end_call_after_reminder = 10
        flow.reminder_messages = ["Are you still with me?"]

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        handler._arm_silence_watchdog()

        # Allow loop to execute first reminder
        await asyncio.sleep(0.01)

        handler._play_tts_message.assert_awaited_once_with(
            "Are you still with me?"
        )
        assert handler._silence_retry_count == 1
        assert handler._call_ended is False

        handler._cancel_silence_watchdog()

    @pytest.mark.anyio
    async def test_silence_watchdog_cycles_custom_reminder_messages(self):
        flow = MagicMock(spec=CallFlow)
        flow.silence_timeout = 0
        flow.reminder_retries = 2
        flow.end_call_after_reminder = 10
        flow.reminder_messages = ["First reminder.", "Second reminder."]

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        handler._arm_silence_watchdog()

        await asyncio.sleep(0.02)

        assert handler._play_tts_message.await_count >= 2
        calls = [c[0][0] for c in handler._play_tts_message.call_args_list]
        assert "First reminder." in calls
        assert "Second reminder." in calls
        assert handler._silence_retry_count == 2

        handler._cancel_silence_watchdog()

    @pytest.mark.anyio
    async def test_user_speech_cancels_and_resets_silence_watchdog(self):
        flow = MagicMock(spec=CallFlow)
        flow.silence_timeout = 10
        flow.reminder_retries = 2
        flow.end_call_after_reminder = 10
        flow.reminder_messages = []

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        handler._arm_silence_watchdog()
        handler._silence_retry_count = 1

        assert handler._silence_watchdog_task is not None

        # User speaks
        await handler._process_transcript("Yes, I have a question.")

        assert handler._silence_retry_count == 0
        assert handler._silence_watchdog_task is None

    @pytest.mark.anyio
    async def test_dtmf_keypress_cancels_silence_watchdog(self):
        flow = MagicMock(spec=CallFlow)
        flow.dtmf_enabled = True
        flow.dtmf_button_press_delay = 2
        flow.dtmf_max_digits = 10
        flow.dtmf_allowed_exceeded_attempts = 5
        flow.dtmf_exceeded_action = "end_call"
        flow.silence_timeout = 10
        flow.reminder_retries = 2

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        handler._arm_silence_watchdog()
        handler._silence_retry_count = 1

        assert handler._silence_watchdog_task is not None

        # User presses DTMF key
        await handler.handle_dtmf_message({"event": "dtmf", "digit": "5"})

        assert handler._silence_retry_count == 0
        assert handler._silence_watchdog_task is None

        if handler._dtmf_debounce_task:
            handler._dtmf_debounce_task.cancel()

    @pytest.mark.anyio
    async def test_final_silence_timeout_terminates_call(self):
        flow = MagicMock(spec=CallFlow)
        flow.silence_timeout = 0
        flow.reminder_retries = 1
        flow.end_call_after_reminder = 0
        flow.reminder_messages = ["Are you still there?"]

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)

        with (
            patch(
                "app.voice.call_control_mixin.call_session_service.update_call_session_status"
            ) as mock_update,
            patch(
                "app.voice.call_control_mixin.get_twilio_credentials_for_call",
                return_value=("AC123", "token"),
            ),
            patch(
                "app.voice.call_control_mixin.twilio_service.end_call_with_credentials"
            ) as mock_end_call,
            patch(
                "app.voice.call_control_mixin.broadcast_call_status_update"
            ) as mock_broadcast,
        ):
            handler._arm_silence_watchdog()
            await asyncio.sleep(0.05)

        assert handler._call_ended is True
        mock_update.assert_called_once_with(
            handler.db,
            session.id,
            "completed",
            ended_reason="Silence timeout after reminders",
        )
        mock_end_call.assert_called_once_with("CA_TIMING_123", "AC123", "token")
        mock_broadcast.assert_called_once()

    @pytest.mark.anyio
    async def test_silence_watchdog_ignores_already_ended_call(self):
        flow = MagicMock(spec=CallFlow)
        flow.silence_timeout = 0
        flow.reminder_retries = 1

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        handler._call_ended = True

        handler._arm_silence_watchdog()
        assert handler._silence_watchdog_task is None


class TestMaxCallDurationRuntime:
    @pytest.mark.anyio
    async def test_max_duration_watchdog_plays_message_and_terminates(self):
        flow = MagicMock(spec=CallFlow)
        flow.max_call_duration = 0  # Instantaneous for test
        flow.max_duration_message = "Our time limit has arrived. Goodbye."

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)

        with (
            patch(
                "app.voice.call_control_mixin.call_session_service.update_call_session_status"
            ) as mock_update,
            patch(
                "app.voice.call_control_mixin.get_twilio_credentials_for_call",
                return_value=("AC123", "token"),
            ),
            patch(
                "app.voice.call_control_mixin.twilio_service.end_call_with_credentials"
            ) as mock_end_call,
            patch(
                "app.voice.call_control_mixin.broadcast_call_status_update"
            ) as mock_broadcast,
        ):
            await handler._max_duration_watchdog(max_seconds=0)

        assert handler._call_ended is True
        handler._play_tts_message.assert_awaited_once_with(
            "Our time limit has arrived. Goodbye."
        )
        mock_update.assert_called_once_with(
            handler.db,
            session.id,
            "completed",
            ended_reason="Max call duration reached",
        )
        mock_end_call.assert_called_once_with("CA_TIMING_123", "AC123", "token")
        mock_broadcast.assert_called_once()

    @pytest.mark.anyio
    async def test_max_duration_watchdog_ignores_already_ended_call(self):
        flow = MagicMock(spec=CallFlow)
        flow.max_call_duration = 0

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        handler._call_ended = True

        await handler._max_duration_watchdog(max_seconds=0)
        assert handler._play_tts_message.await_count == 0

    @pytest.mark.anyio
    async def test_silence_watchdog_skips_when_tts_is_playing(self):
        flow = MagicMock(spec=CallFlow)
        flow.silence_timeout = 0
        flow.reminder_retries = 1
        flow.end_call_after_reminder = 10
        flow.reminder_messages = ["Are you there?"]

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        handler._is_tts_playing = True  # Agent is currently speaking
        handler._arm_silence_watchdog()

        await asyncio.sleep(0.02)

        # Reminder should NOT have played because TTS was active
        assert handler._play_tts_message.await_count == 0
        assert handler._silence_retry_count == 0

        handler._cancel_silence_watchdog()

    @pytest.mark.anyio
    async def test_direct_task_cancellation_resets_silence_retry_count(self):
        flow = MagicMock(spec=CallFlow)
        flow.silence_timeout = 0.01
        flow.reminder_retries = 2
        flow.end_call_after_reminder = 10
        flow.reminder_messages = ["Are you there?"]

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        handler._arm_silence_watchdog()

        # Let it trigger at least one reminder
        await asyncio.sleep(0.03)
        assert handler._silence_retry_count >= 1

        # Direct task cancel (e.g. barge-in or unhandled coroutine cancel)
        if handler._silence_watchdog_task:
            handler._silence_watchdog_task.cancel()
            try:
                await handler._silence_watchdog_task
            except asyncio.CancelledError:
                pass

        # finally block in watchdog coroutine MUST reset retry_count to 0
        assert handler._silence_retry_count == 0

    @pytest.mark.anyio
    async def test_default_play_tts_message_queues_to_tts_pipeline(self):
        class RealMixinHandler(CallControlMixin):
            def __init__(self):
                self._tts_pipeline = MagicMock()
                self._tts_pipeline.queue_tts = AsyncMock()
                self._use_ssml = False
                self._twilio_buffer_primed = True

        handler = RealMixinHandler()
        await handler._play_tts_message("Testing default TTS queueing")

        handler._tts_pipeline.queue_tts.assert_awaited_once()
        task_arg = handler._tts_pipeline.queue_tts.call_args[0][0]
        assert task_arg["text"] == "Testing default TTS queueing"
        assert task_arg["is_final"] is True
        assert handler._twilio_buffer_primed is False

    @pytest.mark.anyio
    async def test_max_duration_watchdog_fallback_db_lookup(self):
        flow = MagicMock(spec=CallFlow)
        flow.max_call_duration = 0
        flow.max_duration_message = "Time is up from DB lookup."

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        db_mock = MagicMock()
        db_mock.execute.return_value.scalar_one_or_none.return_value = flow
        db_mock.get.return_value = flow

        # handler.call_flow is None, should look up via handler.db.get(CallFlow, ...)
        handler = DummyHostHandler(
            db=db_mock, call_session=session, call_flow=None
        )

        with (
            patch(
                "app.voice.call_control_mixin.call_session_service.update_call_session_status"
            ),
            patch(
                "app.voice.call_control_mixin.get_twilio_credentials_for_call",
                return_value=("AC123", "token"),
            ),
            patch(
                "app.voice.call_control_mixin.twilio_service.end_call_with_credentials"
            ),
            patch(
                "app.voice.call_control_mixin.broadcast_call_status_update"
            ),
        ):
            await handler._max_duration_watchdog(max_seconds=0)

        assert handler._call_ended is True
        handler._play_tts_message.assert_awaited_once_with(
            "Time is up from DB lookup."
        )
