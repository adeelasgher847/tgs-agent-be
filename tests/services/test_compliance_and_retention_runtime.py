"""Runtime integration tests for Anti-Bot / Fake Voice Detection, Compliance Monitoring,
and Data Retention purge execution.

Covers:
- Anti-bot signature detection and metadata flagging
- Immediate call termination on synthetic voice / bot detection
- Compliance monitoring policy violation flagging
- Data retention cleanup for expired transcripts, summaries, and S3 recordings
- Retention safety: non-expired calls untouched, core metadata preserved
"""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock, patch
import uuid

import pytest

from app.models.agent import Agent
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.models.tenant import Tenant
from app.models.transcript_message import TranscriptMessage
from app.models.user import User
from app.services.data_retention_service import purge_expired_call_data
from app.voice.call_control_mixin import CallControlMixin


class DummyVoiceHandler(CallControlMixin):
    def __init__(self, db, session, agent, flow):
        self._call_ended = False
        self.db = db
        self.call_session = session
        self.agent = agent
        self.call_flow = flow
        self.call_sid = "CA_BOT_TEST_SID"
        self.stream_sid = "MZ_BOT_STREAM_SID"
        self._teardown_called = False

    async def _full_shutdown(self):
        self._teardown_called = True


@pytest.fixture
def setup_runtime_entities(db):
    tenant = Tenant(
        name=f"RuntimePolicyWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"test_policy_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(tenant)
    db.commit()
    db.refresh(tenant)

    user = User(
        id=uuid.uuid4(),
        email=f"policy-user-{uuid.uuid4().hex[:8]}@example.com",
        first_name="Policy",
        last_name="User",
        hashed_password="dummy",
        current_tenant_id=tenant.id,
    )
    db.add(user)
    db.commit()

    agent = Agent(
        tenant_id=tenant.id,
        name="Policy Agent",
        created_by=user.id,
    )
    db.add(agent)
    db.commit()

    flow = CallFlow(
        tenant_id=tenant.id,
        agent_id=agent.id,
        name="Runtime Policy Flow",
        direction="inbound",
        status="active",
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
        start_time=datetime.datetime.now(datetime.timezone.utc),
        twilio_call_sid="CA_BOT_TEST_SID",
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return tenant, user, agent, flow, session


class TestAntiBotAndFakeVoiceRuntime:
    @pytest.mark.asyncio
    async def test_bot_detected_flagged_without_termination(
        self, db, setup_runtime_entities
    ):
        tenant, user, agent, flow, session = setup_runtime_entities
        flow.anti_bot_detection_enabled = True
        flow.terminate_on_fake_voice = False
        db.commit()

        handler = DummyVoiceHandler(db, session, agent, flow)

        ended = await handler._check_and_handle_anti_bot(
            "Hello, this is an automated call from your pharmacy"
        )
        assert ended is False
        assert handler._call_ended is False

        # Metadata was flagged
        assert session.call_metadata is not None
        assert session.call_metadata["anti_bot"]["detected"] is True
        assert (
            session.call_metadata["anti_bot"]["keyword"]
            == "this is an automated call"
        )

    @pytest.mark.asyncio
    async def test_bot_detected_terminates_call_when_configured(
        self, db, setup_runtime_entities
    ):
        tenant, user, agent, flow, session = setup_runtime_entities
        flow.anti_bot_detection_enabled = True
        flow.terminate_on_fake_voice = True
        db.commit()

        handler = DummyVoiceHandler(db, session, agent, flow)

        with (
            patch(
                "app.voice.call_control_mixin.get_twilio_credentials_for_call",
                return_value=("ACtest", "token123"),
            ),
            patch(
                "app.voice.call_control_mixin.twilio_service.end_call_with_credentials",
                return_value=True,
            ) as mock_end_call,
            patch(
                "app.voice.call_control_mixin.broadcast_call_status_update",
                new=AsyncMock(),
            ) as mock_broadcast,
        ):
            ended = await handler._check_and_handle_anti_bot(
                "Press 1 to speak with an agent or press 2 to hang up"
            )

        assert ended is True
        assert handler._call_ended is True
        mock_end_call.assert_called_once()
        mock_broadcast.assert_called_once()

        db.refresh(session)
        assert session.status == "completed"
        assert session.ended_reason == "Bot or fake voice detected"
        assert session.call_metadata["anti_bot"]["detected"] is True

    @pytest.mark.asyncio
    async def test_bot_detection_ignored_when_disabled(
        self, db, setup_runtime_entities
    ):
        tenant, user, agent, flow, session = setup_runtime_entities
        flow.anti_bot_detection_enabled = False
        flow.terminate_on_fake_voice = False
        db.commit()

        handler = DummyVoiceHandler(db, session, agent, flow)

        ended = await handler._check_and_handle_anti_bot(
            "This is an automated call from your bank"
        )
        assert ended is False
        assert handler._call_ended is False
        assert (
            session.call_metadata is None
            or "anti_bot" not in session.call_metadata
        )

    @pytest.mark.asyncio
    async def test_bot_detection_empty_or_ended_call_ignored(
        self, db, setup_runtime_entities
    ):
        tenant, user, agent, flow, session = setup_runtime_entities
        flow.anti_bot_detection_enabled = True
        flow.terminate_on_fake_voice = True
        db.commit()

        handler = DummyVoiceHandler(db, session, agent, flow)

        # Empty transcript
        assert await handler._check_and_handle_anti_bot("") is False
        assert await handler._check_and_handle_anti_bot("   ") is False

        # Already ended call
        handler._call_ended = True
        assert (
            await handler._check_and_handle_anti_bot(
                "this is an automated message"
            )
            is False
        )

    @pytest.mark.asyncio
    async def test_bot_detection_handles_twilio_error_gracefully(
        self, db, setup_runtime_entities
    ):
        tenant, user, agent, flow, session = setup_runtime_entities
        flow.anti_bot_detection_enabled = True
        flow.terminate_on_fake_voice = True
        db.commit()

        handler = DummyVoiceHandler(db, session, agent, flow)

        with (
            patch(
                "app.voice.call_control_mixin.get_twilio_credentials_for_call",
                return_value=("ACtest", "token123"),
            ),
            patch(
                "app.voice.call_control_mixin.twilio_service.end_call_with_credentials",
                side_effect=RuntimeError("Twilio connection timed out"),
            ),
            patch(
                "app.voice.call_control_mixin.broadcast_call_status_update",
                new=AsyncMock(),
            ),
        ):
            # Should not raise exception
            ended = await handler._check_and_handle_anti_bot(
                "this is an automated message"
            )
            assert ended is True

    @pytest.mark.asyncio
    async def test_bot_detection_ignores_agent_role_utterances(
        self, db, setup_runtime_entities
    ):
        tenant, user, agent, flow, session = setup_runtime_entities
        flow.anti_bot_detection_enabled = True
        flow.terminate_on_fake_voice = True
        db.commit()

        handler = DummyVoiceHandler(db, session, agent, flow)

        # Spoken by the AI agent itself (role='agent') - should not trigger self-termination
        ended = await handler._check_and_handle_anti_bot(
            "This is an automated call from customer support",
            role="agent",
        )
        assert ended is False
        assert handler._call_ended is False
        assert session.call_metadata is None or "anti_bot" not in session.call_metadata

    @pytest.mark.asyncio
    async def test_flow_resolution_blocks_cross_tenant_flow_tampering(
        self, db, setup_runtime_entities
    ):
        tenant, user, agent, flow, session = setup_runtime_entities

        # Create another tenant and flow
        other_tenant = Tenant(
            name="AttackerWS",
            schema_name="test_attacker_ws",
            status="active",
        )
        db.add(other_tenant)
        db.commit()

        other_agent = Agent(
            tenant_id=other_tenant.id,
            name="Attacker Agent",
            created_by=user.id,
        )
        db.add(other_agent)
        db.commit()

        other_flow = CallFlow(
            tenant_id=other_tenant.id,
            agent_id=other_agent.id,
            name="Other Tenant Flow",
            direction="inbound",
            status="active",
        )
        db.add(other_flow)
        db.commit()

        # Session belongs to `tenant`, but `call_flow_id` points to `other_flow.id`
        session.call_flow_id = other_flow.id
        db.commit()

        # Voice handler without cached call_flow
        handler = DummyVoiceHandler(db, session, agent, None)
        resolved = handler._resolve_flow()

        # Must not load other tenant's flow!
        assert resolved is None


class TestComplianceMonitoringRuntime:
    @pytest.mark.asyncio
    async def test_compliance_monitoring_flags_violations(
        self, db, setup_runtime_entities
    ):
        tenant, user, agent, flow, session = setup_runtime_entities
        flow.compliance_monitoring_enabled = True
        db.commit()

        handler = DummyVoiceHandler(db, session, agent, flow)

        await handler._check_and_handle_compliance_monitoring(
            "Please tell me your credit card cvv and wire money immediately"
        )

        assert handler._call_ended is False
        assert session.call_metadata is not None
        assert session.call_metadata["compliance"]["flagged"] is True
        violations = session.call_metadata["compliance"]["violations"]
        triggers = [v["trigger"] for v in violations]
        assert "tell me your credit card cvv" in triggers
        assert "wire money immediately" in triggers

    @pytest.mark.asyncio
    async def test_compliance_monitoring_accumulates_violations(
        self, db, setup_runtime_entities
    ):
        tenant, user, agent, flow, session = setup_runtime_entities
        flow.compliance_monitoring_enabled = True
        db.commit()

        handler = DummyVoiceHandler(db, session, agent, flow)

        await handler._check_and_handle_compliance_monitoring(
            "Please share your password"
        )
        await handler._check_and_handle_compliance_monitoring(
            "Also send gift cards"
        )

        assert session.call_metadata["compliance"]["flagged"] is True
        violations = session.call_metadata["compliance"]["violations"]
        assert len(violations) == 2
        triggers = [v["trigger"] for v in violations]
        assert "share your password" in triggers
        assert "send gift cards" in triggers

    @pytest.mark.asyncio
    async def test_compliance_monitoring_empty_transcript_ignored(
        self, db, setup_runtime_entities
    ):
        tenant, user, agent, flow, session = setup_runtime_entities
        flow.compliance_monitoring_enabled = True
        db.commit()

        handler = DummyVoiceHandler(db, session, agent, flow)
        await handler._check_and_handle_compliance_monitoring("")
        await handler._check_and_handle_compliance_monitoring("   ")
        assert (
            session.call_metadata is None
            or "compliance" not in session.call_metadata
        )

    @pytest.mark.asyncio
    async def test_compliance_monitoring_ignored_when_disabled(
        self, db, setup_runtime_entities
    ):
        tenant, user, agent, flow, session = setup_runtime_entities
        flow.compliance_monitoring_enabled = False
        db.commit()

        handler = DummyVoiceHandler(db, session, agent, flow)

        await handler._check_and_handle_compliance_monitoring(
            "Please wire money immediately"
        )

        assert (
            session.call_metadata is None
            or "compliance" not in session.call_metadata
        )


class TestDataRetentionPurgeServiceRuntime:
    def test_data_retention_purge_cleans_expired_data_and_preserves_metadata(
        self, db, setup_runtime_entities
    ):
        tenant, user, agent, flow, _ = setup_runtime_entities
        flow.retention_policy_enabled = True
        flow.retention_transcript_enabled = True
        flow.retention_transcript_days = 30
        flow.retention_summary_enabled = True
        flow.retention_summary_days = 30
        flow.retention_recording_enabled = True
        flow.retention_recording_days = 30
        db.commit()

        now = datetime.datetime.now(datetime.timezone.utc)
        expired_date = now - datetime.timedelta(days=45)
        recent_date = now - datetime.timedelta(days=10)

        # 1. Expired session
        expired_session = CallSession(
            tenant_id=tenant.id,
            user_id=user.id,
            agent_id=agent.id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="completed",
            start_time=expired_date,
            end_time=expired_date + datetime.timedelta(seconds=120),
            duration=120,
            cost=0.05,
            cost_currency="USD",
            twilio_call_sid="CA_EXPIRED_SID",
            from_number="+15551112222",
            to_number="+15553334444",
            call_transcript=[{"role": "user", "content": "Sensitive history"}],
            transcript_summary="User discussed sensitive medical record.",
            recording_s3_path="recordings/test/expired_audio.opus",
            recording_url="https://api.twilio.com/recordings/expired",
        )
        db.add(expired_session)

        # 2. Recent session (within retention window)
        recent_session = CallSession(
            tenant_id=tenant.id,
            user_id=user.id,
            agent_id=agent.id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="completed",
            start_time=recent_date,
            end_time=recent_date + datetime.timedelta(seconds=60),
            duration=60,
            cost=0.02,
            cost_currency="USD",
            twilio_call_sid="CA_RECENT_SID",
            from_number="+15551112222",
            to_number="+15553334444",
            call_transcript=[{"role": "user", "content": "Recent message"}],
            transcript_summary="Recent summary.",
            recording_s3_path="recordings/test/recent_audio.opus",
            recording_url="https://api.twilio.com/recordings/recent",
        )
        db.add(recent_session)
        db.commit()

        with patch(
            "app.services.s3_recording_service.delete_recording_object",
            return_value=True,
        ) as mock_delete_s3:
            res = purge_expired_call_data(
                db, tenant_id=tenant.id, flow_id=flow.id
            )

        assert res.purged_transcripts_count == 1
        assert res.purged_summaries_count == 1
        assert res.purged_recordings_count == 1
        assert res.purged_sessions_count == 1
        mock_delete_s3.assert_called_once_with(
            "recordings/test/expired_audio.opus"
        )

        # Verify expired session data is purged
        db.refresh(expired_session)
        assert expired_session.call_transcript is None
        assert expired_session.transcript_summary is None
        assert expired_session.recording_s3_path is None
        assert expired_session.recording_url is None

        # Verify core metadata preserved
        assert expired_session.twilio_call_sid == "CA_EXPIRED_SID"
        assert expired_session.duration == 120
        assert expired_session.cost == 0.05
        assert expired_session.from_number == "+15551112222"
        assert expired_session.to_number == "+15553334444"

        # Verify recent session is untouched
        db.refresh(recent_session)
        assert recent_session.call_transcript is not None
        assert recent_session.transcript_summary is not None
        assert recent_session.recording_s3_path is not None
        assert recent_session.recording_url is not None

    def test_retention_policy_disabled_does_not_purge(
        self, db, setup_runtime_entities
    ):
        tenant, user, agent, flow, _ = setup_runtime_entities
        flow.retention_policy_enabled = False
        db.commit()

        now = datetime.datetime.now(datetime.timezone.utc)
        expired_date = now - datetime.timedelta(days=45)

        session = CallSession(
            tenant_id=tenant.id,
            user_id=user.id,
            agent_id=agent.id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="completed",
            start_time=expired_date,
            call_transcript=[{"role": "user", "content": "Keep me"}],
        )
        db.add(session)
        db.commit()

        res = purge_expired_call_data(db, tenant_id=tenant.id, flow_id=flow.id)
        assert res.purged_transcripts_count == 0
        assert res.purged_sessions_count == 0

        db.refresh(session)
        assert session.call_transcript is not None

    def test_data_retention_purges_transcript_messages_table(
        self, db, setup_runtime_entities
    ):
        tenant, user, agent, flow, _ = setup_runtime_entities
        flow.retention_policy_enabled = True
        flow.retention_transcript_enabled = True
        flow.retention_transcript_days = 15
        db.commit()

        now = datetime.datetime.now(datetime.timezone.utc)
        expired_date = now - datetime.timedelta(days=20)

        session = CallSession(
            tenant_id=tenant.id,
            user_id=user.id,
            agent_id=agent.id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="completed",
            start_time=expired_date,
            call_transcript=[{"role": "user", "content": "JSON transcript"}],
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        # Add detailed TranscriptMessage turns
        msg1 = TranscriptMessage(
            call_session_id=session.id,
            role="user",
            message="Detailed turn 1",
            sequence_number=1,
            agent_id=agent.id,
            user_id=user.id,
        )
        msg2 = TranscriptMessage(
            call_session_id=session.id,
            role="assistant",
            message="Detailed turn 2",
            sequence_number=2,
            agent_id=agent.id,
            user_id=user.id,
        )
        db.add_all([msg1, msg2])
        db.commit()

        res = purge_expired_call_data(db, tenant_id=tenant.id, flow_id=flow.id)
        assert res.purged_transcripts_count == 1
        assert res.purged_sessions_count == 1

        db.refresh(session)
        assert session.call_transcript is None

        # Verify TranscriptMessage rows are completely deleted
        remaining_messages = (
            db.query(TranscriptMessage)
            .filter(TranscriptMessage.call_session_id == session.id)
            .all()
        )
        assert len(remaining_messages) == 0

    def test_data_retention_tenant_wide_purge_and_cross_tenant_isolation(
        self, db, setup_runtime_entities
    ):
        tenant_a, user_a, agent_a, flow_a, _ = setup_runtime_entities
        flow_a.retention_policy_enabled = True
        flow_a.retention_summary_enabled = True
        flow_a.retention_summary_days = 30
        db.commit()

        # Create Tenant B
        tenant_b = Tenant(
            name=f"TenantB-{uuid.uuid4().hex[:8]}",
            schema_name=f"tb_{uuid.uuid4().hex[:8]}",
            status="active",
        )
        db.add(tenant_b)
        db.commit()

        agent_b = Agent(
            tenant_id=tenant_b.id,
            name="Agent B",
            created_by=user_a.id,
        )
        db.add(agent_b)
        db.commit()

        flow_b = CallFlow(
            tenant_id=tenant_b.id,
            agent_id=agent_b.id,
            name="Flow B",
            direction="inbound",
            status="active",
            retention_policy_enabled=True,
            retention_summary_enabled=True,
            retention_summary_days=30,
        )
        db.add(flow_b)
        db.commit()

        now = datetime.datetime.now(datetime.timezone.utc)
        expired_date = now - datetime.timedelta(days=40)

        session_a = CallSession(
            tenant_id=tenant_a.id,
            user_id=user_a.id,
            agent_id=agent_a.id,
            call_flow_id=flow_a.id,
            call_type="inbound",
            status="completed",
            start_time=expired_date,
            transcript_summary="Tenant A summary",
        )
        session_b = CallSession(
            tenant_id=tenant_b.id,
            user_id=user_a.id,
            agent_id=agent_b.id,
            call_flow_id=flow_b.id,
            call_type="inbound",
            status="completed",
            start_time=expired_date,
            transcript_summary="Tenant B summary",
        )
        db.add_all([session_a, session_b])
        db.commit()

        # Tenant-wide purge for Tenant A (flow_id=None)
        res_a = purge_expired_call_data(db, tenant_id=tenant_a.id, flow_id=None)
        assert res_a.purged_summaries_count == 1
        assert res_a.purged_sessions_count == 1

        db.refresh(session_a)
        assert session_a.transcript_summary is None

        # Verify Tenant B session summary remains intact (cross-tenant isolation)
        db.refresh(session_b)
        assert session_b.transcript_summary == "Tenant B summary"

    def test_data_retention_s3_failure_continues_purge(
        self, db, setup_runtime_entities
    ):
        tenant, user, agent, flow, _ = setup_runtime_entities
        flow.retention_policy_enabled = True
        flow.retention_recording_enabled = True
        flow.retention_recording_days = 30
        db.commit()

        now = datetime.datetime.now(datetime.timezone.utc)
        expired_date = now - datetime.timedelta(days=40)

        session = CallSession(
            tenant_id=tenant.id,
            user_id=user.id,
            agent_id=agent.id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="completed",
            start_time=expired_date,
            recording_s3_path="recordings/missing/file.opus",
            recording_url="https://api.twilio.com/recordings/missing",
        )
        db.add(session)
        db.commit()

        with patch(
            "app.services.s3_recording_service.delete_recording_object",
            return_value=False,
        ) as mock_delete_s3:
            res = purge_expired_call_data(
                db, tenant_id=tenant.id, flow_id=flow.id
            )

        assert res.purged_recordings_count == 1
        assert res.purged_sessions_count == 1
        mock_delete_s3.assert_called_once_with("recordings/missing/file.opus")

        db.refresh(session)
        assert session.recording_s3_path is None
        assert session.recording_url is None

    def test_retention_purge_with_naive_datetime(
        self, db, setup_runtime_entities
    ):
        tenant, user, agent, flow, _ = setup_runtime_entities
        flow.retention_policy_enabled = True
        flow.retention_transcript_enabled = True
        flow.retention_transcript_days = 30
        db.commit()

        # Legacy naive datetime (without timezone info)
        naive_expired_date = datetime.datetime.utcnow() - datetime.timedelta(days=45)

        session = CallSession(
            tenant_id=tenant.id,
            user_id=user.id,
            agent_id=agent.id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="completed",
            start_time=naive_expired_date,
            call_transcript=[{"role": "user", "content": "Legacy naive transcript"}],
        )
        db.add(session)
        db.commit()

        res = purge_expired_call_data(
            db, tenant_id=tenant.id, flow_id=flow.id
        )
        assert res.purged_transcripts_count == 1
        assert res.purged_sessions_count == 1

        db.refresh(session)
        assert session.call_transcript is None

    def test_retention_purge_commits_db_before_s3_deletion(
        self, db, setup_runtime_entities
    ):
        tenant, user, agent, flow, _ = setup_runtime_entities
        flow.retention_policy_enabled = True
        flow.retention_recording_enabled = True
        flow.retention_recording_days = 30
        db.commit()

        now = datetime.datetime.now(datetime.timezone.utc)
        expired_date = now - datetime.timedelta(days=40)

        session = CallSession(
            tenant_id=tenant.id,
            user_id=user.id,
            agent_id=agent.id,
            call_flow_id=flow.id,
            call_type="inbound",
            status="completed",
            start_time=expired_date,
            recording_s3_path="recordings/order_test/file.opus",
            recording_url="https://api.twilio.com/recordings/order_test",
        )
        db.add(session)
        db.commit()

        # Even if S3 deletion raises an exception, DB changes must already be committed
        with patch(
            "app.services.s3_recording_service.delete_recording_object",
            side_effect=Exception("S3 network timeout"),
        ):
            res = purge_expired_call_data(
                db, tenant_id=tenant.id, flow_id=flow.id
            )

        assert res.purged_recordings_count == 1
        assert res.purged_sessions_count == 1

        db.refresh(session)
        # S3 path and recording_url must be cleared in DB
        assert session.recording_s3_path is None
        assert session.recording_url is None
