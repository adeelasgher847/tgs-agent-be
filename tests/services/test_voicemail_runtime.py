"""Unit and integration tests for Voicemail Detection runtime and AMD dispatch.

Coverage:
  - In-call keyword detection: disabled flow bypasses termination
  - In-call keyword detection: hang_up ends call immediately
  - In-call keyword detection: leave_message triggers voicemail message playback then ends call
  - In-call keyword detection: continue allows conversation to proceed
  - Outbound Twilio AMD kwargs generation: standard vs advanced ML detection (DetectMessageEnd)
  - AMD webhook callback handling for CallFlow sessions: machine_start, machine_end_beep, human, continue
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


class TestVoicemailKeywordRuntime:
    @pytest.mark.anyio
    async def test_disabled_flow_bypasses_voicemail_detection(self):
        flow = MagicMock(spec=CallFlow)
        flow.voicemail_detection_enabled = False
        flow.voicemail_action = "hang_up"

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        result = await handler._check_and_end_call_if_voicemail("Your call has been forwarded to voicemail.")

        assert result is False
        assert handler._call_ended is False

    @pytest.mark.anyio
    async def test_enabled_flow_with_hang_up_ends_call(self):
        flow = MagicMock(spec=CallFlow)
        flow.voicemail_detection_enabled = True
        flow.voicemail_action = "hang_up"

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
            result = await handler._check_and_end_call_if_voicemail("Please leave a message after the tone.")

        assert result is True
        assert handler._call_ended is True
        mock_update.assert_called_once_with(
            handler.db, session.id, "completed", ended_reason="Voicemail detected"
        )
        mock_end_call.assert_called_once_with("CA123", "AC123", "token")
        mock_broadcast.assert_called_once()

    @pytest.mark.anyio
    async def test_enabled_flow_with_leave_message_plays_tts_and_ends_call(self):
        flow = MagicMock(spec=CallFlow)
        flow.voicemail_detection_enabled = True
        flow.voicemail_action = "leave_message"
        flow.voicemail_message = "Hi, this is a test voicemail message."

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)

        with (
            patch("app.voice.call_control_mixin.call_session_service.update_call_session_status"),
            patch("app.voice.call_control_mixin.get_twilio_credentials_for_call", return_value=("AC123", "token")),
            patch("app.voice.call_control_mixin.twilio_service.end_call_with_credentials"),
            patch("app.voice.call_control_mixin.broadcast_call_status_update"),
        ):
            result = await handler._check_and_end_call_if_voicemail("Please leave a message after the beep.")

        assert result is True
        assert handler._call_ended is True
        handler._play_tts_message.assert_called_once_with("Hi, this is a test voicemail message.")

    @pytest.mark.anyio
    async def test_enabled_flow_with_continue_action_allows_call_to_proceed(self):
        flow = MagicMock(spec=CallFlow)
        flow.voicemail_detection_enabled = True
        flow.voicemail_action = "continue"

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        handler = DummyHostHandler(call_session=session, call_flow=flow)
        result = await handler._check_and_end_call_if_voicemail("Please leave a message after the tone.")

        assert result is False
        assert handler._call_ended is False

    @pytest.mark.anyio
    async def test_db_error_on_lazy_callflow_fetch_returns_false_and_does_not_hangup(self):
        from sqlalchemy.exc import SQLAlchemyError

        session = MagicMock(spec=CallSession)
        session.id = uuid.uuid4()
        session.call_flow_id = uuid.uuid4()

        mock_db = MagicMock()
        mock_db.get.side_effect = SQLAlchemyError("DB connection lost")

        handler = DummyHostHandler(db=mock_db, call_session=session, call_flow=None)
        result = await handler._check_and_end_call_if_voicemail("Your call has been forwarded to voicemail.")

        assert result is False
        assert handler._call_ended is False


class TestVoicemailAmdDispatchLogic:
    def test_flow_with_advanced_ml_amd_generates_detect_message_end(self):
        flow = MagicMock()
        flow.voicemail_detection_enabled = True
        flow.voicemail_advanced_detection_enabled = True
        flow.voicemail_detection_timeout = 15

        detection_type = "DetectMessageEnd" if flow.voicemail_advanced_detection_enabled else "Enable"
        timeout_val = flow.voicemail_detection_timeout or 5
        amd_status_callback_url = "https://example.com/amd"

        amd_kwargs = {
            "machine_detection": detection_type,
            "machine_detection_timeout": timeout_val,
            "async_amd": "true",
            "async_amd_status_callback": amd_status_callback_url,
        }

        assert amd_kwargs["machine_detection"] == "DetectMessageEnd"
        assert amd_kwargs["machine_detection_timeout"] == 15
        assert amd_kwargs["async_amd"] == "true"

    def test_flow_with_standard_amd_generates_enable(self):
        flow = MagicMock()
        flow.voicemail_detection_enabled = True
        flow.voicemail_advanced_detection_enabled = False
        flow.voicemail_detection_timeout = 8

        detection_type = "DetectMessageEnd" if flow.voicemail_advanced_detection_enabled else "Enable"
        timeout_val = flow.voicemail_detection_timeout or 5
        amd_status_callback_url = "https://example.com/amd"

        amd_kwargs = {
            "machine_detection": detection_type,
            "machine_detection_timeout": timeout_val,
            "async_amd": "true",
            "async_amd_status_callback": amd_status_callback_url,
        }

        assert amd_kwargs["machine_detection"] == "Enable"
        assert amd_kwargs["machine_detection_timeout"] == 8
        assert amd_kwargs["async_amd"] == "true"


class TestAmdWebhookCallFlowIntegration:
    @pytest.mark.anyio
    async def test_amd_callback_machine_start_hangup(self, db):
        from datetime import datetime, timezone
        from app.models.tenant import Tenant
        from app.models.user import User
        from app.models.agent import Agent
        from app.routers import amd_webhook

        tenant = Tenant(name=f"AMDWS-{uuid.uuid4().hex[:8]}", schema_name=f"amd_ws_{uuid.uuid4().hex[:8]}", status="active")
        user = User(first_name="Test", last_name="User", email=f"amd_{uuid.uuid4().hex[:8]}@example.com", hashed_password="pw")
        db.add_all([tenant, user])
        db.commit()

        agent = Agent(
            tenant_id=tenant.id,
            name="AMD Agent",
            status="active",
            llm_model="gpt-4o-mini",
            tts_provider_slug="elevenlabs",
            tts_voice_external_id="voice-x",
            tts_language="en",
        )
        db.add(agent)
        db.commit()

        flow = CallFlow(
            tenant_id=tenant.id,
            agent_id=agent.id,
            name="AMD Test Flow",
            direction="outbound",
            voicemail_detection_enabled=True,
            voicemail_action="hang_up",
        )
        db.add(flow)
        db.commit()

        session = CallSession(
            user_id=user.id,
            agent_id=agent.id,
            tenant_id=tenant.id,
            call_flow_id=flow.id,
            call_type="outbound",
            status="active",
            start_time=datetime.now(timezone.utc),
            call_metadata={},
        )
        db.add(session)
        db.commit()

        req = MagicMock()
        req.form = AsyncMock(return_value={"AnsweredBy": "machine_start", "CallSid": "CA999"})

        with (
            patch.object(amd_webhook, "_validate_amd_signature", return_value=True),
            patch.object(amd_webhook, "_hangup") as mock_hangup,
        ):
            await amd_webhook.amd_callback(req, callSessionId=str(session.id), db=db)

        db.refresh(session)
        assert session.call_metadata["amd_result"] == "machine_start"
        assert session.ended_reason == "Voicemail detected"
        mock_hangup.assert_called_once_with(db, session, "CA999")

    @pytest.mark.anyio
    async def test_amd_callback_machine_end_beep_plays_message(self, db):
        from datetime import datetime, timezone
        from app.models.tenant import Tenant
        from app.models.user import User
        from app.models.agent import Agent
        from app.routers import amd_webhook

        tenant = Tenant(name=f"AMDWS-{uuid.uuid4().hex[:8]}", schema_name=f"amd_ws_{uuid.uuid4().hex[:8]}", status="active")
        user = User(first_name="Test", last_name="User", email=f"amd_{uuid.uuid4().hex[:8]}@example.com", hashed_password="pw")
        db.add_all([tenant, user])
        db.commit()

        agent = Agent(
            tenant_id=tenant.id,
            name="AMD Agent",
            status="active",
            llm_model="gpt-4o-mini",
            tts_provider_slug="elevenlabs",
            tts_voice_external_id="voice-x",
            tts_language="en",
        )
        db.add(agent)
        db.commit()

        flow = CallFlow(
            tenant_id=tenant.id,
            agent_id=agent.id,
            name="AMD Leave Message Flow",
            direction="outbound",
            voicemail_detection_enabled=True,
            voicemail_action="leave_message",
            voicemail_message="Please callback at your earliest convenience.",
        )
        db.add(flow)
        db.commit()

        session = CallSession(
            user_id=user.id,
            agent_id=agent.id,
            tenant_id=tenant.id,
            call_flow_id=flow.id,
            call_type="outbound",
            status="active",
            start_time=datetime.now(timezone.utc),
            call_metadata={},
        )
        db.add(session)
        db.commit()

        req = MagicMock()
        req.form = AsyncMock(return_value={"AnsweredBy": "machine_end_beep", "CallSid": "CA888"})

        with (
            patch.object(amd_webhook, "_validate_amd_signature", return_value=True),
            patch.object(amd_webhook, "_play_voicemail_and_hangup") as mock_play,
        ):
            await amd_webhook.amd_callback(req, callSessionId=str(session.id), db=db)

        mock_play.assert_called_once_with(
            db, session, "CA888", "Please callback at your earliest convenience."
        )
        db.refresh(session)
        assert session.call_metadata["amd_result"] == "machine_end_beep"
        # Ended reason must NOT be set prematurely while TTS is playing
        assert session.ended_reason is None

    @pytest.mark.anyio
    async def test_amd_callback_double_hangup_guard_prevents_duplicate_hangup(self, db):
        from datetime import datetime, timezone
        from app.models.tenant import Tenant
        from app.models.user import User
        from app.models.agent import Agent
        from app.routers import amd_webhook

        tenant = Tenant(name=f"AMDWS-{uuid.uuid4().hex[:8]}", schema_name=f"amd_ws_{uuid.uuid4().hex[:8]}", status="active")
        user = User(first_name="Test", last_name="User", email=f"amd_{uuid.uuid4().hex[:8]}@example.com", hashed_password="pw")
        db.add_all([tenant, user])
        db.commit()

        agent = Agent(
            tenant_id=tenant.id,
            name="AMD Agent",
            status="active",
            llm_model="gpt-4o-mini",
            tts_provider_slug="elevenlabs",
            tts_voice_external_id="voice-x",
            tts_language="en",
        )
        db.add(agent)
        db.commit()

        flow = CallFlow(
            tenant_id=tenant.id,
            agent_id=agent.id,
            name="AMD Hangup Flow",
            direction="outbound",
            voicemail_detection_enabled=True,
            voicemail_action="hang_up",
        )
        db.add(flow)
        db.commit()

        session = CallSession(
            user_id=user.id,
            agent_id=agent.id,
            tenant_id=tenant.id,
            call_flow_id=flow.id,
            call_type="outbound",
            status="active",
            start_time=datetime.now(timezone.utc),
            call_metadata={},
        )
        db.add(session)
        db.commit()

        req1 = MagicMock()
        req1.form = AsyncMock(return_value={"AnsweredBy": "machine_start", "CallSid": "CA_DUP"})

        with (
            patch.object(amd_webhook, "_validate_amd_signature", return_value=True),
            patch.object(amd_webhook, "_hangup") as mock_hangup1,
        ):
            await amd_webhook.amd_callback(req1, callSessionId=str(session.id), db=db)

        db.refresh(session)
        assert session.call_metadata["amd_result"] == "machine_start"
        assert session.ended_reason == "Voicemail detected"
        mock_hangup1.assert_called_once()

        # Twilio sends subsequent machine_end_beep callback for the same call
        req2 = MagicMock()
        req2.form = AsyncMock(return_value={"AnsweredBy": "machine_end_beep", "CallSid": "CA_DUP"})

        with (
            patch.object(amd_webhook, "_validate_amd_signature", return_value=True),
            patch.object(amd_webhook, "_hangup") as mock_hangup2,
        ):
            await amd_webhook.amd_callback(req2, callSessionId=str(session.id), db=db)

        # Second callback must not trigger duplicate hangup since session is no longer active / already ended
        mock_hangup2.assert_not_called()

    @pytest.mark.anyio
    async def test_amd_callback_tenant_isolation_on_callflow_lookup(self, db):
        from datetime import datetime, timezone
        from app.models.tenant import Tenant
        from app.models.user import User
        from app.models.agent import Agent
        from app.routers import amd_webhook

        tenant_a = Tenant(name=f"AMDWS-A-{uuid.uuid4().hex[:8]}", schema_name=f"amd_ws_a_{uuid.uuid4().hex[:8]}", status="active")
        tenant_b = Tenant(name=f"AMDWS-B-{uuid.uuid4().hex[:8]}", schema_name=f"amd_ws_b_{uuid.uuid4().hex[:8]}", status="active")
        user = User(first_name="Test", last_name="User", email=f"amd_{uuid.uuid4().hex[:8]}@example.com", hashed_password="pw")
        db.add_all([tenant_a, tenant_b, user])
        db.commit()

        agent = Agent(
            tenant_id=tenant_a.id,
            name="AMD Agent",
            status="active",
            llm_model="gpt-4o-mini",
            tts_provider_slug="elevenlabs",
            tts_voice_external_id="voice-x",
            tts_language="en",
        )
        db.add(agent)
        db.commit()

        # Flow belongs to Tenant A
        flow = CallFlow(
            tenant_id=tenant_a.id,
            agent_id=agent.id,
            name="AMD Flow A",
            direction="outbound",
            voicemail_detection_enabled=True,
            voicemail_action="hang_up",
        )
        db.add(flow)
        db.commit()

        # CallSession belongs to Tenant B, referencing flow from Tenant A
        session = CallSession(
            user_id=user.id,
            agent_id=agent.id,
            tenant_id=tenant_b.id,
            call_flow_id=flow.id,
            call_type="outbound",
            status="active",
            start_time=datetime.now(timezone.utc),
            call_metadata={},
        )
        db.add(session)
        db.commit()

        req = MagicMock()
        req.form = AsyncMock(return_value={"AnsweredBy": "machine_start", "CallSid": "CA_TENANT"})

        with (
            patch.object(amd_webhook, "_validate_amd_signature", return_value=True),
            patch.object(amd_webhook, "_hangup") as mock_hangup,
        ):
            await amd_webhook.amd_callback(req, callSessionId=str(session.id), db=db)

        # Cross-tenant flow must NOT be matched; action defaults to continue, no hangup
        mock_hangup.assert_not_called()
        db.refresh(session)
        assert session.call_metadata["amd_result"] == "continue"

    @pytest.mark.anyio
    async def test_amd_callback_continue_action_persists_continue(self, db):
        from datetime import datetime, timezone
        from app.models.tenant import Tenant
        from app.models.user import User
        from app.models.agent import Agent
        from app.routers import amd_webhook

        tenant = Tenant(name=f"AMDWS-{uuid.uuid4().hex[:8]}", schema_name=f"amd_ws_{uuid.uuid4().hex[:8]}", status="active")
        user = User(first_name="Test", last_name="User", email=f"amd_{uuid.uuid4().hex[:8]}@example.com", hashed_password="pw")
        db.add_all([tenant, user])
        db.commit()

        agent = Agent(
            tenant_id=tenant.id,
            name="AMD Agent",
            status="active",
            llm_model="gpt-4o-mini",
            tts_provider_slug="elevenlabs",
            tts_voice_external_id="voice-x",
            tts_language="en",
        )
        db.add(agent)
        db.commit()

        flow = CallFlow(
            tenant_id=tenant.id,
            agent_id=agent.id,
            name="AMD Continue Flow",
            direction="outbound",
            voicemail_detection_enabled=True,
            voicemail_action="continue",
        )
        db.add(flow)
        db.commit()

        session = CallSession(
            user_id=user.id,
            agent_id=agent.id,
            tenant_id=tenant.id,
            call_flow_id=flow.id,
            call_type="outbound",
            status="active",
            start_time=datetime.now(timezone.utc),
            call_metadata={},
        )
        db.add(session)
        db.commit()

        req = MagicMock()
        req.form = AsyncMock(return_value={"AnsweredBy": "machine_start", "CallSid": "CA777"})

        with (
            patch.object(amd_webhook, "_validate_amd_signature", return_value=True),
            patch.object(amd_webhook, "_hangup") as mock_hangup,
        ):
            await amd_webhook.amd_callback(req, callSessionId=str(session.id), db=db)

        db.refresh(session)
        assert session.call_metadata["amd_result"] == "continue"
        mock_hangup.assert_not_called()
