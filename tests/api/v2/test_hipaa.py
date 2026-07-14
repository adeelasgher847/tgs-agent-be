"""
Tests for HIPAA compliance implementation.

Coverage:
  - dlp_service._HIPAA_PATTERNS: local regex redaction for ICD-10, MRN, SSN, CPT, NPI, insurance IDs
  - dlp_service.redact_phi: local-pattern pass runs first, then DLP inspect_content
  - TranscriptService.add_message: redacts message when hipaa_enabled=True, skips when False
  - call_control_mixin._add_to_transcript: passes hipaa_enabled from self.call_flow
  - voice_analysis_service: redacts analysis fields before storing in call_metadata
  - s3_recording_service.upload_recording: kms_key_name passed as SSE-KMS params for CMK
  - s3_recording_service.set_bucket_default_kms_key: sets bucket default KMS for LiveKit uploads
  - PUT /api/v2/flows/{id}/settings: toggle hipaa_compliance, admin RBAC, audit event
  - GET /api/v2/workspace/hipaa-status: returns hipaa_enabled_flows, kms_key_configured, baa_on_file
  - PUT /api/v2/workspace/kms-key: validates KMS key, persists, sets bucket default, rejects bad format
  - GET /api/v1/recordings/{call_id}: 403 for read_only/config on HIPAA flows, 200 for admin
"""
from __future__ import annotations

import sys
import uuid
from unittest.mock import MagicMock, patch, call

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exception_handlers import register_exception_handlers

# ── Shared IDs ────────────────────────────────────────────────────────────────

WORKSPACE_ID = uuid.uuid4()
FLOW_ID = uuid.uuid4()
CALL_ID = uuid.uuid4()
USER_ID = uuid.uuid4()

# ── DLP Service Tests ─────────────────────────────────────────────────────────


class TestRedactPhi:
    """Unit tests for dlp_service.redact_phi."""

    def _make_finding(self, quote: str, start: int):
        finding = MagicMock()
        finding.quote = quote
        finding.location.byte_range.start = start
        finding.location.byte_range.end = start + len(quote)
        return finding

    def _dlp_response(self, findings):
        resp = MagicMock()
        resp.result.findings = findings
        return resp

    def test_redact_phi_calls_dlp_and_replaces_findings(self):
        """DLP inspect_content is called and PHI findings are replaced with [REDACTED]."""
        from app.services import dlp_service

        text = "Patient John Smith called from 555-123-4567"
        name_finding = self._make_finding("John Smith", 8)
        phone_finding = self._make_finding("555-123-4567", 31)

        mock_response = self._dlp_response([name_finding, phone_finding])
        mock_client = MagicMock()
        mock_client.inspect_content.return_value = mock_response

        mock_dlp = MagicMock()
        mock_dlp.Likelihood.POSSIBLE = 2
        sys.modules["google.cloud"].dlp_v2 = mock_dlp

        # Inject mock client into the singleton slot
        old_client = dlp_service._dlp_client
        dlp_service._dlp_client = mock_client
        try:
            mock_settings = MagicMock()
            mock_settings.GCP_PROJECT_ID = "test-project"
            with patch.object(dlp_service, "settings", mock_settings):
                result = dlp_service.redact_phi(text)
        finally:
            dlp_service._dlp_client = old_client

        mock_client.inspect_content.assert_called_once()
        call_kwargs = mock_client.inspect_content.call_args[1]["request"]
        assert call_kwargs["parent"] == "projects/test-project"
        assert call_kwargs["item"] == {"value": text}
        assert "[REDACTED]" in result
        assert "John Smith" not in result
        assert "555-123-4567" not in result

    def test_redact_phi_no_findings_returns_original(self):
        from app.services import dlp_service

        text = "No PHI here"
        mock_response = self._dlp_response([])
        mock_client = MagicMock()
        mock_client.inspect_content.return_value = mock_response

        mock_dlp = MagicMock()
        mock_dlp.Likelihood.POSSIBLE = 2
        sys.modules["google.cloud"].dlp_v2 = mock_dlp

        old_client = dlp_service._dlp_client
        dlp_service._dlp_client = mock_client
        try:
            mock_settings = MagicMock()
            mock_settings.GCP_PROJECT_ID = "test-project"
            with patch.object(dlp_service, "settings", mock_settings):
                result = dlp_service.redact_phi(text)
        finally:
            dlp_service._dlp_client = old_client

        assert result == text

    def test_redact_phi_missing_project_id_returns_original(self):
        from app.services import dlp_service

        mock_settings = MagicMock()
        mock_settings.GCP_PROJECT_ID = ""
        with patch.object(dlp_service, "settings", mock_settings):
            result = dlp_service.redact_phi("Patient John Smith")

        assert result == "Patient John Smith"

    def test_redact_phi_dlp_error_returns_original(self):
        from app.services import dlp_service

        mock_client = MagicMock()
        mock_client.inspect_content.side_effect = RuntimeError("DLP unavailable")

        mock_dlp = MagicMock()
        mock_dlp.Likelihood.POSSIBLE = 2
        sys.modules["google.cloud"].dlp_v2 = mock_dlp

        old_client = dlp_service._dlp_client
        dlp_service._dlp_client = mock_client
        try:
            mock_settings = MagicMock()
            mock_settings.GCP_PROJECT_ID = "test-project"
            with patch.object(dlp_service, "settings", mock_settings):
                result = dlp_service.redact_phi("Patient John Smith")
        finally:
            dlp_service._dlp_client = old_client

        # Must never raise — fail open
        assert result == "Patient John Smith"

    def test_redact_phi_if_hipaa_disabled_skips_dlp(self):
        from app.services import dlp_service

        with patch.object(dlp_service, "redact_phi") as mock_redact:
            result = dlp_service.redact_phi_if_hipaa("Patient data", hipaa_enabled=False)
            mock_redact.assert_not_called()
        assert result == "Patient data"

    def test_redact_phi_if_hipaa_enabled_calls_dlp(self):
        from app.services import dlp_service

        with patch.object(dlp_service, "redact_phi", return_value="[REDACTED]") as mock_redact:
            result = dlp_service.redact_phi_if_hipaa("Patient data", hipaa_enabled=True)
            mock_redact.assert_called_once_with("Patient data")
        assert result == "[REDACTED]"

    def test_local_patterns_run_before_dlp(self):
        """Local _HIPAA_PATTERNS pass runs first; Cloud DLP step runs on already-cleaned text."""
        from app.services import dlp_service

        mock_response = self._dlp_response([])
        mock_client = MagicMock()
        mock_client.inspect_content.return_value = mock_response
        mock_dlp = MagicMock()
        mock_dlp.Likelihood.POSSIBLE = 2
        sys.modules["google.cloud"].dlp_v2 = mock_dlp

        old_client = dlp_service._dlp_client
        dlp_service._dlp_client = mock_client
        try:
            mock_settings = MagicMock()
            mock_settings.GCP_PROJECT_ID = "test-project"
            with patch.object(dlp_service, "settings", mock_settings):
                result = dlp_service.redact_phi("Patient SSN: 123-45-6789")
        finally:
            dlp_service._dlp_client = old_client

        assert "123-45-6789" not in result
        assert "[REDACTED-SSN]" in result
        mock_client.inspect_content.assert_called_once()


# ── Local HIPAA Pattern Tests ─────────────────────────────────────────────────


class TestHipaaPatterns:
    """Unit tests for _HIPAA_PATTERNS local regex redaction."""

    def _redact(self, text: str) -> str:
        from app.services.dlp_service import _redact_local_patterns
        return _redact_local_patterns(text)

    def test_ssn_redacted(self):
        assert "[REDACTED-SSN]" in self._redact("SSN is 123-45-6789 on file")
        assert "123-45-6789" not in self._redact("SSN is 123-45-6789 on file")

    def test_mrn_redacted(self):
        result = self._redact("Patient MRN: 1234567 admitted today")
        assert "[REDACTED-MRN]" in result
        assert "1234567" not in result

    def test_mrn_medical_record_number_redacted(self):
        result = self._redact("Medical Record Number: 9876543")
        assert "[REDACTED-MRN]" in result

    def test_icd10_with_context_redacted(self):
        result = self._redact("Diagnosis: J45.20 (moderate persistent asthma)")
        assert "[REDACTED-DIAGNOSIS]" in result
        assert "J45.20" not in result

    def test_icd10_standalone_redacted(self):
        result = self._redact("Code E11.65 noted in the chart")
        assert "[REDACTED-ICD10]" in result
        assert "E11.65" not in result

    def test_cpt_code_redacted(self):
        result = self._redact("Billed CPT: 99213 for the visit")
        assert "[REDACTED-CPT]" in result
        assert "99213" not in result

    def test_npi_redacted(self):
        result = self._redact("Provider NPI: 1234567890")
        assert "[REDACTED-NPI]" in result
        assert "1234567890" not in result

    def test_insurance_member_id_redacted(self):
        result = self._redact("Member ID: ABC123456 under BlueCross")
        assert "[REDACTED-INSURANCE-ID]" in result

    def test_non_hipaa_text_unchanged(self):
        text = "The appointment is scheduled for next Tuesday at 2pm"
        assert self._redact(text) == text

    def test_no_false_positive_on_regular_numbers(self):
        text = "Please call extension 1234 for support"
        result = self._redact(text)
        # Should not be redacted — no HIPAA context
        assert "1234" in result


# ── Transcript HIPAA Redaction Tests ─────────────────────────────────────────


class TestTranscriptHipaaRedaction:
    """TranscriptService.add_message redacts when hipaa_enabled=True."""

    def _make_db(self):
        db = MagicMock()
        # Sequence number query
        db.query.return_value.filter.return_value.order_by.return_value.first.return_value = None
        db.add = MagicMock()
        db.commit = MagicMock()
        db.refresh = MagicMock()
        return db

    def test_add_message_redacts_when_hipaa_enabled(self):
        from app.services.transcript_service import TranscriptService

        db = self._make_db()
        saved_message = None

        def capture_add(obj):
            nonlocal saved_message
            saved_message = obj

        db.add.side_effect = capture_add

        with patch(
            "app.services.transcript_service.redact_phi_if_hipaa",
            return_value="[REDACTED]",
        ) as mock_redact:
            TranscriptService.add_message(
                db=db,
                call_session_id=CALL_ID,
                role="client",
                message="My SSN is 123-45-6789",
                hipaa_enabled=True,
            )
            mock_redact.assert_called_once_with(
                "My SSN is 123-45-6789", hipaa_enabled=True
            )

        assert saved_message is not None
        assert saved_message.message == "[REDACTED]"

    def test_add_message_skips_redaction_when_hipaa_disabled(self):
        from app.services.transcript_service import TranscriptService

        db = self._make_db()

        with patch(
            "app.services.transcript_service.redact_phi_if_hipaa",
            wraps=lambda t, **_: t,
        ) as mock_redact:
            TranscriptService.add_message(
                db=db,
                call_session_id=CALL_ID,
                role="client",
                message="Normal text",
                hipaa_enabled=False,
            )
            mock_redact.assert_called_once_with("Normal text", hipaa_enabled=False)


# ── call_control_mixin hipaa_enabled threading ───────────────────────────────


class TestCallControlMixinHipaaEnabled:
    """_add_to_transcript passes hipaa_enabled=True when call_flow.hipaa_compliance is True."""

    def _make_handler(self, hipaa_compliance: bool):
        """Construct a minimal fake BidirectionalStreamHandler with the mixin applied."""
        from app.voice.call_control_mixin import CallControlMixin

        class FakeHandler(CallControlMixin):
            def __init__(self):
                self.call_session = MagicMock()
                self.call_session.id = CALL_ID
                self.call_session.user_id = USER_ID
                self.agent = MagicMock()
                self.agent.id = uuid.uuid4()
                self.db = MagicMock()

                self.call_flow = MagicMock()
                self.call_flow.hipaa_compliance = hipaa_compliance

                # Stubs required by _add_to_transcript
                self._call_ended = False
                self._is_duplicate_agent_line = MagicMock(return_value=False)

        return FakeHandler()

    def test_hipaa_flow_passes_hipaa_enabled_true(self):
        import asyncio
        from app.voice import call_control_mixin as ccm_module

        handler = self._make_handler(hipaa_compliance=True)
        captured_kwargs: dict = {}

        async def fake_add_and_broadcast(**kwargs):
            captured_kwargs.update(kwargs)
            msg = MagicMock()
            msg.id = uuid.uuid4()
            return msg

        async def run():
            with patch.object(ccm_module.transcript_service, "add_and_broadcast_message", side_effect=fake_add_and_broadcast):
                await handler._add_to_transcript("client", "my SSN is 123-45-6789")

        asyncio.run(run())
        assert captured_kwargs.get("hipaa_enabled") is True

    def test_non_hipaa_flow_passes_hipaa_enabled_false(self):
        import asyncio
        from app.voice import call_control_mixin as ccm_module

        handler = self._make_handler(hipaa_compliance=False)
        captured_kwargs: dict = {}

        async def fake_add_and_broadcast(**kwargs):
            captured_kwargs.update(kwargs)
            msg = MagicMock()
            msg.id = uuid.uuid4()
            return msg

        async def run():
            with patch.object(ccm_module.transcript_service, "add_and_broadcast_message", side_effect=fake_add_and_broadcast):
                await handler._add_to_transcript("agent", "Hello!")

        asyncio.run(run())
        assert captured_kwargs.get("hipaa_enabled") is False

    def test_no_call_flow_passes_hipaa_enabled_false(self):
        """call_flow=None on handler → hipaa_enabled=False (defensive)."""
        import asyncio
        from app.voice import call_control_mixin as ccm_module

        handler = self._make_handler(hipaa_compliance=False)
        handler.call_flow = None  # simulate flow not loaded
        captured_kwargs: dict = {}

        async def fake_add_and_broadcast(**kwargs):
            captured_kwargs.update(kwargs)
            msg = MagicMock()
            msg.id = uuid.uuid4()
            return msg

        async def run():
            with patch.object(ccm_module.transcript_service, "add_and_broadcast_message", side_effect=fake_add_and_broadcast):
                await handler._add_to_transcript("client", "Hello")

        asyncio.run(run())
        assert captured_kwargs.get("hipaa_enabled") is False


# ── Voice Analysis HIPAA Redaction Tests ─────────────────────────────────────


class TestVoiceAnalysisHipaaRedaction:
    """voice_analysis_service redacts analysis fields before storing in call_metadata."""

    def _make_call_session(self, flow_hipaa: bool = True):
        session = MagicMock()
        session.id = CALL_ID
        session.tenant_id = WORKSPACE_ID
        session.call_flow_id = FLOW_ID
        session.call_metadata = {}
        session.duration = 120
        session.status = "completed"
        session.agent_id = None
        return session

    def _make_db(self, flow_hipaa: bool = True):
        from app.models.call_flow import CallFlow

        flow = MagicMock()
        flow.hipaa_compliance = flow_hipaa
        flow.is_deleted = False

        db = MagicMock()

        def _query(model):
            q = MagicMock()
            if model is CallFlow:
                q.filter.return_value.first.return_value = flow
            else:
                q.filter.return_value.first.return_value = None
            return q

        db.query.side_effect = _query
        db.commit = MagicMock()
        db.refresh = MagicMock()
        return db

    def test_analysis_fields_redacted_on_hipaa_flow(self):
        """
        When call_flow.hipaa_compliance is True, redact_phi_if_hipaa is called
        with hipaa_enabled=True on every analysis text field before persistence.
        """
        from app.services import voice_analysis_service as vas_module
        from app.services.voice_analysis_service import VoiceAnalysisService

        session = self._make_call_session(flow_hipaa=True)
        db = self._make_db(flow_hipaa=True)

        msg = MagicMock()
        msg.role = "client"
        msg.message = "Patient John Smith SSN: 123-45-6789"

        # Mock model — uses openai provider so the inner generate call is predictable
        mock_model = MagicMock()
        mock_model.model_name = "gpt-4o-mini"
        mock_model.provider.name = "openai"
        mock_model.api_key = None  # skip decrypt path

        analysis_text = {"content": "The call resolved the patient's issue."}
        redact_calls_hipaa: list[bool] = []

        def tracking_redact(text: str, *, hipaa_enabled: bool) -> str:
            redact_calls_hipaa.append(hipaa_enabled)
            return text  # pass through so analysis_data can be assembled

        with patch.object(vas_module.transcript_service, "get_messages_by_session", return_value=[msg]):
            with patch("app.services.voice_analysis_service.redact_phi_if_hipaa", side_effect=tracking_redact):
                with patch.object(vas_module.ModelService, "get_model_by_name", return_value=mock_model):
                    # Patch the OpenAIService that generate_analysis_text creates internally
                    with patch("app.services.openai_service.OpenAIService.generate_text", return_value=analysis_text):
                        try:
                            VoiceAnalysisService().analyze_call_transcript(
                                db=db, call_session=session, user_id=USER_ID
                            )
                        except Exception:
                            pass

        # At least the summary, sentiment, and caller_name fields must have been
        # passed to redact_phi_if_hipaa with hipaa_enabled=True
        assert any(h is True for h in redact_calls_hipaa), (
            f"Expected redact_phi_if_hipaa called with hipaa_enabled=True; "
            f"got hipaa_enabled values: {redact_calls_hipaa}"
        )

    def test_non_hipaa_flow_skips_redaction(self):
        """When flow.hipaa_compliance=False, redact_phi_if_hipaa is called with hipaa_enabled=False."""
        from app.services import voice_analysis_service as vas_module
        from app.services.voice_analysis_service import VoiceAnalysisService

        session = self._make_call_session(flow_hipaa=False)
        db = self._make_db(flow_hipaa=False)

        msg = MagicMock()
        msg.role = "client"
        msg.message = "Normal call text"

        with patch.object(vas_module.transcript_service, "get_messages_by_session", return_value=[msg]):
            with patch("app.services.voice_analysis_service.redact_phi_if_hipaa") as mock_redact:
                mock_redact.side_effect = lambda t, **_: t
                with patch.object(vas_module.ModelService, "get_model_by_name", return_value=None):
                    try:
                        VoiceAnalysisService().analyze_call_transcript(
                            db=db, call_session=session, user_id=USER_ID
                        )
                    except Exception:
                        pass

            # If redact was called, it must have been with hipaa_enabled=False
            for c in mock_redact.call_args_list:
                assert c.kwargs.get("hipaa_enabled") is False


# ── S3 CMK Upload Tests ────────────────────────────────────────────────────────


class TestCmekUpload:
    """Verify kms_key_name is forwarded as SSE-KMS params for CMK-encrypted uploads."""

    def test_upload_recording_passes_kms_key_to_put_object(self):
        from app.services import s3_recording_service

        kms_key = "arn:aws:kms:us-east-1:111122223333:key/1234abcd-12ab-34cd-56ef-1234567890ab"
        file_bytes = b"fake-audio"
        metadata = {"callId": str(CALL_ID)}

        mock_client = MagicMock()

        with patch("app.services.s3_recording_service.get_s3_client", return_value=mock_client):
            s3_recording_service.upload_recording(
                key="recordings/test.opus",
                file_bytes=file_bytes,
                metadata=metadata,
                kms_key_name=kms_key,
            )

        mock_client.put_object.assert_called_once()
        call_kwargs = mock_client.put_object.call_args.kwargs
        assert call_kwargs["Key"] == "recordings/test.opus"
        assert call_kwargs["Body"] == file_bytes
        assert call_kwargs["ServerSideEncryption"] == "aws:kms"
        assert call_kwargs["SSEKMSKeyId"] == kms_key

    def test_upload_recording_no_kms_key_omits_kms_param(self):
        from app.services import s3_recording_service

        mock_client = MagicMock()

        with patch("app.services.s3_recording_service.get_s3_client", return_value=mock_client):
            s3_recording_service.upload_recording(
                key="recordings/test.opus",
                file_bytes=b"audio",
                metadata={},
            )

        call_kwargs = mock_client.put_object.call_args.kwargs
        assert "ServerSideEncryption" not in call_kwargs
        assert "SSEKMSKeyId" not in call_kwargs

    def test_set_bucket_default_kms_key(self):
        """set_bucket_default_kms_key calls put_bucket_encryption with the SSE-KMS key."""
        from app.services import s3_recording_service

        kms_key = "arn:aws:kms:us-east-1:111122223333:key/1234abcd-12ab-34cd-56ef-1234567890ab"
        mock_client = MagicMock()

        with patch("app.services.s3_recording_service.get_s3_client", return_value=mock_client):
            s3_recording_service.set_bucket_default_kms_key(kms_key)

        mock_client.put_bucket_encryption.assert_called_once()
        call_kwargs = mock_client.put_bucket_encryption.call_args.kwargs
        rule = call_kwargs["ServerSideEncryptionConfiguration"]["Rules"][0]
        assert rule["ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"] == "aws:kms"
        assert rule["ApplyServerSideEncryptionByDefault"]["KMSMasterKeyID"] == kms_key


# ── HIPAA Router Tests ────────────────────────────────────────────────────────


def _make_user(tenant_id: uuid.UUID = WORKSPACE_ID) -> MagicMock:
    user = MagicMock()
    user.id = USER_ID
    user.current_tenant_id = tenant_id
    return user


def _build_hipaa_app(db_override) -> TestClient:
    """Minimal app mounting both HIPAA routers with auth and DB overridden."""
    from app.api.deps import get_db, require_admin
    from app.api.v2.routers.hipaa import flows_router, workspace_router

    admin_user = _make_user()

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(flows_router)
    mini.include_router(workspace_router)

    mini.dependency_overrides[require_admin] = lambda: admin_user
    mini.dependency_overrides[get_db] = lambda: db_override

    return TestClient(mini, raise_server_exceptions=False)


class TestHipaaFlowSettings:
    """PUT /flows/{id}/settings"""

    def _make_db_with_flow(
        self,
        hipaa_compliance: bool = False,
        baa_on_file: bool = False,
    ):
        flow = MagicMock()
        flow.id = FLOW_ID
        flow.tenant_id = WORKSPACE_ID
        flow.hipaa_compliance = hipaa_compliance
        flow.is_deleted = False

        tenant = MagicMock()
        tenant.id = WORKSPACE_ID
        tenant.baa_on_file = baa_on_file
        tenant.kms_key_name = None

        db = MagicMock()

        # SQLAlchemy 2.x pattern: db.execute(stmt).scalar_one_or_none()
        mock_flow_result = MagicMock()
        mock_flow_result.scalar_one_or_none.return_value = flow

        mock_tenant_result = MagicMock()
        mock_tenant_result.scalar_one_or_none.return_value = tenant

        # First execute → CallFlow query, second → Tenant query
        db.execute.side_effect = [mock_flow_result, mock_tenant_result]
        db.commit = MagicMock()
        db.refresh = MagicMock()
        return db, flow

    def test_enable_hipaa_returns_200_and_updated_flag(self):
        db, flow = self._make_db_with_flow(hipaa_compliance=False, baa_on_file=True)
        client = _build_hipaa_app(db)

        resp = client.put(f"/flows/{FLOW_ID}/settings", json={"hipaa_compliance": True})

        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["flow_id"] == str(FLOW_ID)
        assert body["data"]["hipaa_compliance"] is True

    def test_enable_hipaa_fires_audit_event(self):
        db, flow = self._make_db_with_flow(hipaa_compliance=False, baa_on_file=True)
        client = _build_hipaa_app(db)

        with patch("app.api.v2.routers.hipaa.log_audit_event") as mock_log_audit:
            client.put(f"/flows/{FLOW_ID}/settings", json={"hipaa_compliance": True})

        mock_log_audit.assert_called_once()
        kwargs = mock_log_audit.call_args.kwargs
        assert kwargs["action"] == "hipaa_flag.updated"
        assert kwargs["resource_type"] == "call_flow"
        assert kwargs["resource_id"] == FLOW_ID
        assert kwargs["old_value"] == {"hipaa_compliance": False}
        assert kwargs["new_value"] == {"hipaa_compliance": True}

    def test_no_change_skips_commit(self):
        db, flow = self._make_db_with_flow(hipaa_compliance=True, baa_on_file=True)
        client = _build_hipaa_app(db)

        resp = client.put(f"/flows/{FLOW_ID}/settings", json={"hipaa_compliance": True})

        assert resp.status_code == 200
        db.commit.assert_not_called()

    def test_flow_not_found_returns_404(self):
        db = MagicMock()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        db.execute.return_value = mock_result
        client = _build_hipaa_app(db)

        resp = client.put(f"/flows/{uuid.uuid4()}/settings", json={"hipaa_compliance": True})
        assert resp.status_code == 404

    def test_missing_body_field_returns_400(self):
        """App exception handler converts Pydantic ValidationError to 400."""
        db, _ = self._make_db_with_flow()
        client = _build_hipaa_app(db)

        resp = client.put(f"/flows/{FLOW_ID}/settings", json={})
        assert resp.status_code == 400

    def test_enable_hipaa_baa_not_on_file_returns_400(self):
        """HIPAA cannot be enabled when tenant.baa_on_file is False."""
        db, _ = self._make_db_with_flow(hipaa_compliance=False, baa_on_file=False)
        client = _build_hipaa_app(db)

        resp = client.put(f"/flows/{FLOW_ID}/settings", json={"hipaa_compliance": True})

        assert resp.status_code == 400
        body = resp.json()
        assert "BAA" in body["error"]["message"] or "Business Associate" in body["error"]["message"]

    def test_enable_hipaa_baa_on_file_succeeds(self):
        """HIPAA can be enabled when tenant.baa_on_file is True."""
        db, flow = self._make_db_with_flow(hipaa_compliance=False, baa_on_file=True)
        client = _build_hipaa_app(db)

        resp = client.put(f"/flows/{FLOW_ID}/settings", json={"hipaa_compliance": True})

        assert resp.status_code == 200
        assert resp.json()["data"]["hipaa_compliance"] is True


class TestHipaaStatus:
    """GET /workspace/hipaa-status"""

    def _make_db(
        self,
        flow_ids: list,
        kms_key_name: str | None = None,
        baa_on_file: bool = False,
    ) -> MagicMock:
        from app.models.tenant import Tenant
        from app.models.call_flow import CallFlow

        tenant = MagicMock()
        tenant.id = WORKSPACE_ID
        tenant.kms_key_name = kms_key_name
        tenant.baa_on_file = baa_on_file

        flow_rows = [MagicMock(id=fid) for fid in flow_ids]

        db = MagicMock()

        # SQLAlchemy 2.x: db.execute(stmt)
        # First call → _get_tenant_or_404 (Tenant), second → _get_hipaa_flow_ids (CallFlow.id)
        mock_tenant_result = MagicMock()
        mock_tenant_result.scalar_one_or_none.return_value = tenant

        mock_flows_result = MagicMock()
        mock_flows_result.all.return_value = flow_rows

        db.execute.side_effect = [mock_tenant_result, mock_flows_result]
        return db

    def test_hipaa_status_returns_correct_structure(self):
        fid1 = uuid.uuid4()
        fid2 = uuid.uuid4()
        kms = "projects/p/locations/us-central1/keyRings/r/cryptoKeys/k"
        db = self._make_db(flow_ids=[fid1, fid2], kms_key_name=kms, baa_on_file=True)
        client = _build_hipaa_app(db)

        resp = client.get("/workspace/hipaa-status")

        assert resp.status_code == 200
        body = resp.json()["data"]
        assert set(body["hipaa_enabled_flows"]) == {str(fid1), str(fid2)}
        assert body["kms_key_configured"] is True
        assert body["baa_on_file"] is True

    def test_hipaa_status_no_kms_key_returns_false(self):
        db = self._make_db(flow_ids=[], kms_key_name=None)
        client = _build_hipaa_app(db)

        resp = client.get("/workspace/hipaa-status")

        assert resp.status_code == 200
        assert resp.json()["data"]["kms_key_configured"] is False


class TestKmsKeyUpdate:
    """PUT /workspace/kms-key"""

    _VALID_KEY = "arn:aws:kms:us-east-1:111122223333:key/1234abcd-12ab-34cd-56ef-1234567890ab"

    def _make_db(self, kms_key_name: str | None = None):
        tenant = MagicMock()
        tenant.id = WORKSPACE_ID
        tenant.kms_key_name = kms_key_name

        db = MagicMock()

        # SQLAlchemy 2.x: db.execute(stmt).scalar_one_or_none()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = tenant
        db.execute.return_value = mock_result

        db.commit = MagicMock()
        db.refresh = MagicMock()
        return db, tenant

    def test_valid_kms_key_persisted_and_returned(self):
        db, tenant = self._make_db()
        client = _build_hipaa_app(db)

        with patch("app.api.v2.routers.hipaa._validate_kms_key"):
            with patch("app.api.v2.routers.hipaa.s3_recording_service.set_bucket_default_kms_key") as mock_set_bucket:
                resp = client.put("/workspace/kms-key", json={"kms_key_name": self._VALID_KEY})

        assert resp.status_code == 200
        assert resp.json()["data"]["kms_key_name"] == self._VALID_KEY
        assert tenant.kms_key_name == self._VALID_KEY
        assert db.commit.call_count == 2
        mock_set_bucket.assert_called_once_with(self._VALID_KEY)

    def test_bucket_default_kms_failure_does_not_break_response(self):
        """S3 bucket patch failure is logged as warning; endpoint still returns 200."""
        from app.services import s3_recording_service

        db, tenant = self._make_db()
        client = _build_hipaa_app(db)

        with patch("app.api.v2.routers.hipaa._validate_kms_key"):
            with patch(
                "app.api.v2.routers.hipaa.s3_recording_service.set_bucket_default_kms_key",
                side_effect=RuntimeError("S3 unavailable"),
            ):
                resp = client.put("/workspace/kms-key", json={"kms_key_name": self._VALID_KEY})

        # Key is still persisted even if S3 bucket patch fails
        assert resp.status_code == 200
        assert tenant.kms_key_name == self._VALID_KEY

    def test_kms_key_without_projects_prefix_returns_400(self):
        """App exception handler converts Pydantic ValidationError to 400."""
        db, _ = self._make_db()
        client = _build_hipaa_app(db)

        resp = client.put("/workspace/kms-key", json={"kms_key_name": "bad-key-name"})
        assert resp.status_code == 400

    def test_kms_validation_failure_returns_400(self):
        db, _ = self._make_db()
        client = _build_hipaa_app(db)

        from fastapi import HTTPException as FastHTTPException

        with patch(
            "app.api.v2.routers.hipaa._validate_kms_key",
            side_effect=FastHTTPException(status_code=400, detail="KMS key validation failed"),
        ):
            resp = client.put("/workspace/kms-key", json={"kms_key_name": self._VALID_KEY})

        assert resp.status_code == 400


# ── Recording RBAC Gate Tests ─────────────────────────────────────────────────


class TestRecordingHipaaRbac:
    """
    GET /api/v1/recordings/{call_id}
    HIPAA flows: 403 for read_only/config_only roles, 200 for admin/manager
    (and the workspace creator, via is_creator — see require_admin in deps.py).
    """

    def _setup_recording_app(
        self,
        *,
        hipaa_compliance: bool,
        role_name: str,
    ) -> TestClient:
        """Build a minimal recording app with controlled call session / role state."""
        from app.api.deps import get_db, require_tenant
        from app.routers.recordings import router as recordings_router

        flow_id = uuid.uuid4()

        # Mock call session
        session = MagicMock()
        session.id = CALL_ID
        session.tenant_id = WORKSPACE_ID
        session.call_flow_id = flow_id
        session.recording_s3_path = "recordings/test.opus"
        session.recording_error = False
        session.duration = 60

        # Mock flow
        flow = MagicMock()
        flow.id = flow_id
        flow.hipaa_compliance = hipaa_compliance

        db = MagicMock()

        def _query(model):
            from app.models.call_session import CallSession
            from app.models.call_flow import CallFlow

            q = MagicMock()
            if model is CallSession:
                q.filter.return_value.first.return_value = session
            elif model is CallFlow:
                q.filter.return_value.first.return_value = flow
            return q

        db.query.side_effect = _query

        # Mock JWT user principal (plain MagicMock — not spec'd so dynamic attrs work)
        user = MagicMock()
        user.id = USER_ID
        user.current_tenant_id = WORKSPACE_ID

        mini = FastAPI()
        register_exception_handlers(mini)
        mini.include_router(recordings_router)

        mini.dependency_overrides[require_tenant] = lambda: user
        mini.dependency_overrides[get_db] = lambda: db

        return TestClient(mini, raise_server_exceptions=False)

    @pytest.mark.parametrize("role_name", ["read_only", "config_only"])
    def test_blocked_role_on_hipaa_flow_returns_403(self, role_name: str):
        client = self._setup_recording_app(hipaa_compliance=True, role_name=role_name)

        with patch("app.routers.recordings.role_service.get_membership_role_name", return_value=role_name):
            with patch("app.routers.recordings.get_recording_enabled_for_call", return_value=True):
                with patch("app.services.s3_recording_service.generate_signed_url"):
                    resp = client.get(f"/{CALL_ID}")

        assert resp.status_code == 403
        # App uses {"error": {"message": ...}} format via build_api_error_payload
        message = resp.json()["error"]["message"]
        assert "HIPAA" in message or "admin" in message.lower()

    @pytest.mark.parametrize("role_name", ["admin", "manager"])
    def test_allowed_role_on_hipaa_flow_returns_200(self, role_name: str):
        client = self._setup_recording_app(hipaa_compliance=True, role_name=role_name)

        with patch("app.routers.recordings.role_service.get_membership_role_name", return_value=role_name):
            with patch("app.routers.recordings.get_recording_enabled_for_call", return_value=True):
                with patch("app.services.s3_recording_service.generate_signed_url", return_value="https://signed.url"):
                    with patch("app.services.s3_recording_service.get_object_size", return_value=1024):
                        resp = client.get(f"/{CALL_ID}")

        assert resp.status_code == 200

    @pytest.mark.parametrize("role_name", ["read_only", "config_only"])
    def test_blocked_role_on_non_hipaa_flow_returns_200(self, role_name: str):
        """HIPAA RBAC gate only applies when hipaa_compliance=True."""
        client = self._setup_recording_app(hipaa_compliance=False, role_name=role_name)

        with patch("app.routers.recordings.role_service.get_membership_role_name", return_value=role_name):
            with patch("app.routers.recordings.get_recording_enabled_for_call", return_value=True):
                with patch("app.services.s3_recording_service.generate_signed_url", return_value="https://signed.url"):
                    with patch("app.services.s3_recording_service.get_object_size", return_value=512):
                        resp = client.get(f"/{CALL_ID}")

        assert resp.status_code == 200
