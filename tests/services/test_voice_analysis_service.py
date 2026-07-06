"""Unit tests for voice_analysis_service — transcript_summary persistence.

Cross-session caller memory reads CallSession.transcript_summary, which must
be populated with the finalized, HIPAA-redacted summary at call-end analysis
time. Mirrors the mocking pattern from tests/api/v2/test_hipaa.py's
TestVoiceAnalysisHipaaRedaction.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

WORKSPACE_ID = uuid.uuid4()
FLOW_ID = uuid.uuid4()
CALL_ID = uuid.uuid4()
USER_ID = uuid.uuid4()


def _make_call_session():
    session = MagicMock()
    session.id = CALL_ID
    session.tenant_id = WORKSPACE_ID
    session.call_flow_id = FLOW_ID
    session.call_metadata = {}
    session.duration = 120
    session.status = "completed"
    session.agent_id = None
    session.transcript_summary = None
    return session


def _make_db():
    from app.models.call_flow import CallFlow

    flow = MagicMock()
    flow.hipaa_compliance = False
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


class TestTranscriptSummaryPersistence:
    def test_summary_persisted_to_transcript_summary_column(self):
        from app.services import voice_analysis_service as vas_module
        from app.services.voice_analysis_service import VoiceAnalysisService

        session = _make_call_session()
        db = _make_db()

        msg = MagicMock()
        msg.role = "client"
        msg.message = "I'd like to book an appointment for next Tuesday."

        mock_model = MagicMock()
        mock_model.model_name = "gpt-4o-mini"
        mock_model.provider.name = "openai"
        mock_model.api_key = None

        analysis_text = {
            "content": "Caller booked an appointment for next Tuesday.\nCALLER_NAME: Jane Doe"
        }

        with patch.object(vas_module.transcript_service, "get_messages_by_session", return_value=[msg]):
            with patch.object(vas_module.ModelService, "get_model_by_name", return_value=mock_model):
                with patch(
                    "app.services.openai_service.OpenAIService.generate_text",
                    return_value=analysis_text,
                ):
                    result = VoiceAnalysisService().analyze_call_transcript(
                        db=db, call_session=session, user_id=USER_ID
                    )

        assert session.transcript_summary == result["analysis"]["summary"]
        assert "Caller booked an appointment for next Tuesday." in session.transcript_summary
        # CALLER_NAME marker line must be stripped out of the persisted summary.
        assert "CALLER_NAME" not in session.transcript_summary
        db.commit.assert_called()

    def test_cached_analysis_path_does_not_touch_transcript_summary(self):
        """Cache-hit path returns early and must not raise or clobber transcript_summary."""
        from app.services import voice_analysis_service as vas_module
        from app.services.voice_analysis_service import VoiceAnalysisService

        session = _make_call_session()
        session.transcript_summary = "Pre-existing summary from a prior run."
        session.call_metadata = {
            "llm_call_analysis": {
                "analysis": {"summary": "cached summary"},
                "model_used": "gpt-4o-mini",
                "timestamp": "2026-07-01T00:00:00+00:00",
            }
        }

        with patch.object(vas_module.transcript_service, "get_messages_by_session", return_value=[]):
            result = VoiceAnalysisService().analyze_call_transcript(
                db=MagicMock(), call_session=session, user_id=USER_ID
            )

        assert result["is_cached"] is True
        assert session.transcript_summary == "Pre-existing summary from a prior run."
