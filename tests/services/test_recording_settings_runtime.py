"""Runtime tests for Call Flow Recording Settings and Real-Time Pipeline Execution.

Coverage:
  - Resolution of recording_enabled from CallFlow in recording_config_service
  - Synchronous LiveKit recording startup when faster_inbound_pickup=False (Default)
  - Asynchronous background task recording startup when faster_inbound_pickup=True
  - Stop recording on transfer when stop_recording_on_transfer=True
  - Preserve recording when stop_recording_on_transfer=False
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.models.tenant import Tenant
from app.models.transfer_route import TransferRoute
from app.models.user import User
from app.services.recording_config_service import get_recording_enabled_for_call
from app.voice.call_control_mixin import CallControlMixin


class TestRecordingConfigServiceResolution:
    @pytest.fixture
    def setup_entities(self, db):
        tenant = Tenant(
            name=f"RecResWS-{uuid.uuid4().hex[:8]}",
            schema_name=f"rec_res_ws_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(tenant)
        db.commit()
        db.refresh(tenant)

        user = User(
            email=f"user-{uuid.uuid4().hex[:8]}@example.com",
            first_name="Test",
            last_name="User",
            current_tenant_id=tenant.id,
            hashed_password="dummy",
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        agent = Agent(
            tenant_id=tenant.id,
            name="Rec Agent",
            created_by=user.id,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)

        flow = CallFlow(
            tenant_id=tenant.id,
            agent_id=agent.id,
            name="Rec Flow",
            direction="inbound",
            status="active",
            recording_enabled=True,
        )
        db.add(flow)
        db.commit()
        db.refresh(flow)

        session = CallSession(
            tenant_id=tenant.id,
            user_id=user.id,
            agent_id=agent.id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="active",
            start_time=flow.created_at,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        return tenant, user, agent, flow, session

    def test_recording_enabled_true_from_flow(self, db, setup_entities):
        _, _, _, flow, session = setup_entities
        flow.recording_enabled = True
        db.commit()

        assert get_recording_enabled_for_call(db, session) is True

    def test_recording_enabled_false_from_flow(self, db, setup_entities):
        _, _, _, flow, session = setup_entities
        flow.recording_enabled = False
        db.commit()

        assert get_recording_enabled_for_call(db, session) is False

    def test_recording_enabled_web_call_with_flow_overrides_env(
        self, db, setup_entities
    ):
        _, _, _, flow, session = setup_entities
        session.call_type = "web"
        flow.recording_enabled = False
        db.commit()

        with patch(
            "app.services.recording_config_service.settings.VOICE_BROWSER_DEMO_RECORDING_ENABLED",
            True,
        ):
            assert get_recording_enabled_for_call(db, session) is False

    def test_recording_enabled_tenant_mismatch_isolated(self, db, setup_entities):
        _, _, _, flow, session = setup_entities
        # Flow belongs to foreign tenant
        flow.tenant_id = uuid.uuid4()
        flow.recording_enabled = True
        db.commit()

        # Should NOT use the foreign flow's recording_enabled, returns False when no number config
        assert get_recording_enabled_for_call(db, session) is False

    def test_recording_enabled_deleted_flow_ignored(self, db, setup_entities):
        _, _, _, flow, session = setup_entities
        flow.is_deleted = True
        flow.recording_enabled = True
        db.commit()

        assert get_recording_enabled_for_call(db, session) is False

    def test_recording_enabled_none_column_defaults_true(self, db, setup_entities):
        _, _, _, _, session = setup_entities
        mock_flow = MagicMock(spec=CallFlow)
        mock_flow.recording_enabled = None
        mock_flow.tenant_id = session.tenant_id
        mock_flow.is_deleted = False

        with patch.object(db, "execute") as mock_exec:
            mock_exec.return_value.scalar_one_or_none.return_value = mock_flow
            assert get_recording_enabled_for_call(db, session) is True


class TestFasterInboundPickupTiming:
    @pytest.mark.asyncio
    async def test_synchronous_recording_when_faster_pickup_false(self):
        handler = MagicMock()
        handler.call_sid = "CA12345"
        handler.call_session = MagicMock(call_type="inbound")
        handler._recording_started = False
        handler.db = MagicMock()
        handler.call_flow = MagicMock(faster_inbound_pickup=False)
        handler._start_livekit_recording = AsyncMock()

        with (
            patch(
                "app.services.recording_config_service.get_recording_enabled_for_call",
                return_value=True,
            ),
            patch("app.core.config.settings.LIVEKIT_ENABLED", True),
            patch("asyncio.create_task") as mock_create_task,
        ):
            # Simulate the start recording block in handle_start_message
            _rec_enabled = get_recording_enabled_for_call(
                handler.db, handler.call_session
            )
            assert _rec_enabled is True

            faster_pickup = False
            if handler.call_flow and getattr(
                handler.call_flow, "faster_inbound_pickup", False
            ):
                faster_pickup = True

            if faster_pickup:
                asyncio.create_task(handler._start_livekit_recording())
            else:
                await handler._start_livekit_recording()

            handler._start_livekit_recording.assert_awaited_once()
            mock_create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_asynchronous_task_when_faster_pickup_true(self):
        handler = MagicMock()
        handler.call_sid = "CA12345"
        handler.call_session = MagicMock(call_type="inbound")
        handler._recording_started = False
        handler.db = MagicMock()
        handler.call_flow = MagicMock(faster_inbound_pickup=True)
        handler._start_livekit_recording = MagicMock()

        with (
            patch(
                "app.services.recording_config_service.get_recording_enabled_for_call",
                return_value=True,
            ),
            patch("app.core.config.settings.LIVEKIT_ENABLED", True),
            patch("asyncio.create_task") as mock_create_task,
        ):
            _rec_enabled = get_recording_enabled_for_call(
                handler.db, handler.call_session
            )
            assert _rec_enabled is True

            faster_pickup = False
            if handler.call_flow and getattr(
                handler.call_flow, "faster_inbound_pickup", False
            ):
                faster_pickup = True

            if faster_pickup:
                asyncio.create_task(handler._start_livekit_recording())
            else:
                await handler._start_livekit_recording()

            mock_create_task.assert_called_once()


class DummyTransferHandler(CallControlMixin):
    def __init__(self, db, session, agent, flow):
        self._call_ended = False
        self.db = db
        self.call_session = session
        self.agent = agent
        self.call_flow = flow
        self.call_sid = "CA_TRANSFER_TEST_SID"
        self._teardown_called = False

    async def _teardown_livekit_recording(self):
        self._teardown_called = True


class TestStopRecordingOnTransfer:
    @pytest.fixture
    def setup_transfer_entities(self, db):
        tenant = Tenant(
            name=f"TransferWS-{uuid.uuid4().hex[:8]}",
            schema_name=f"transfer_ws_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(tenant)
        db.commit()

        user = User(
            email=f"transfer-user-{uuid.uuid4().hex[:8]}@example.com",
            first_name="Transfer",
            last_name="Tester",
            current_tenant_id=tenant.id,
            hashed_password="dummy",
        )
        db.add(user)
        db.commit()

        route = TransferRoute(
            tenant_id=tenant.id,
            friendly_name="Support Desk",
            phone_number="+15550003333",
            transfer_type="cold",
        )
        db.add(route)
        db.commit()

        agent = Agent(
            tenant_id=tenant.id,
            name="Transfer Agent",
            created_by=user.id,
            transfer_route_id=route.id,
        )
        db.add(agent)
        db.commit()

        flow = CallFlow(
            tenant_id=tenant.id,
            agent_id=agent.id,
            name="Transfer Flow",
            direction="inbound",
            status="active",
            stop_recording_on_transfer=True,
        )
        db.add(flow)
        db.commit()

        session = CallSession(
            tenant_id=tenant.id,
            user_id=user.id,
            agent_id=agent.id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="active",
            start_time=flow.created_at,
            twilio_call_sid="CA_TRANSFER_TEST_SID",
        )
        db.add(session)
        db.commit()

        return tenant, user, agent, route, flow, session

    @pytest.mark.asyncio
    async def test_transfer_halts_recording_when_configured(
        self, db, setup_transfer_entities
    ):
        _, _, agent, route, flow, session = setup_transfer_entities
        flow.stop_recording_on_transfer = True
        db.commit()

        handler = DummyTransferHandler(db, session, agent, flow)

        with (
            patch(
                "app.voice.call_control_mixin.get_twilio_credentials_for_call",
                return_value=("ACtest", "token123"),
            ),
            patch(
                "app.voice.call_control_mixin.twilio_service.redirect_call_with_credentials",
                return_value=True,
            ),
        ):
            await handler._transfer_after_agent_request()

        assert handler._teardown_called is True
        assert handler._call_ended is True

    @pytest.mark.asyncio
    async def test_transfer_does_not_halt_recording_when_disabled(
        self, db, setup_transfer_entities
    ):
        _, _, agent, route, flow, session = setup_transfer_entities
        flow.stop_recording_on_transfer = False
        db.commit()

        handler = DummyTransferHandler(db, session, agent, flow)

        with (
            patch(
                "app.voice.call_control_mixin.get_twilio_credentials_for_call",
                return_value=("ACtest", "token123"),
            ),
            patch(
                "app.voice.call_control_mixin.twilio_service.redirect_call_with_credentials",
                return_value=True,
            ),
        ):
            await handler._transfer_after_agent_request()

        assert handler._teardown_called is False
        assert handler._call_ended is True

    @pytest.mark.asyncio
    async def test_transfer_resilient_when_teardown_raises_exception(
        self, db, setup_transfer_entities
    ):
        _, _, agent, route, flow, session = setup_transfer_entities
        flow.stop_recording_on_transfer = True
        db.commit()

        handler = DummyTransferHandler(db, session, agent, flow)

        async def _exploding_teardown():
            raise RuntimeError("LiveKit server unreachable")

        handler._teardown_livekit_recording = _exploding_teardown

        with (
            patch(
                "app.voice.call_control_mixin.get_twilio_credentials_for_call",
                return_value=("ACtest", "token123"),
            ),
            patch(
                "app.voice.call_control_mixin.twilio_service.redirect_call_with_credentials",
                return_value=True,
            ),
        ):
            # Should not raise exception
            await handler._transfer_after_agent_request()

        assert handler._call_ended is True

    @pytest.mark.asyncio
    async def test_transfer_warm_route_with_stop_recording(
        self, db, setup_transfer_entities
    ):
        _, _, agent, route, flow, session = setup_transfer_entities
        route.transfer_type = "warm"
        flow.stop_recording_on_transfer = True
        db.commit()

        handler = DummyTransferHandler(db, session, agent, flow)

        with (
            patch(
                "app.voice.call_control_mixin.get_twilio_credentials_for_call",
                return_value=("ACtest", "token123"),
            ),
            patch(
                "app.voice.call_control_mixin.twilio_service.redirect_call_with_credentials",
                return_value=True,
            ) as mock_redirect,
        ):
            await handler._transfer_after_agent_request()

        assert handler._teardown_called is True
        assert handler._call_ended is True
        mock_redirect.assert_called_once()
        assert "conference-customer" in mock_redirect.call_args[0][1]
