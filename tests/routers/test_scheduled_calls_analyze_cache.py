"""Unit tests for analyze_call_transcript_internal's cache short-circuit.

app/routers/scheduled_calls.py::analyze_call_transcript_internal now checks
call_session.call_metadata["llm_call_analysis"] first (populated either by
the automatic post-call summary ARQ job or a prior manual "analyze
transcript" call) and returns that cached analysis instead of re-running
(paid) LLM calls. Falls through to the existing uncached LLM-calling logic
only when no cache exists.

Mirrors the MagicMock-db style used in tests/services/test_voice_analysis_service.py.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from app.routers.scheduled_calls import analyze_call_transcript_internal

_CALL_ID = uuid.uuid4()
_TENANT_ID = uuid.uuid4()
_AGENT_ID = uuid.uuid4()


def _make_call_session(*, call_metadata):
    session = MagicMock()
    session.id = _CALL_ID
    session.tenant_id = _TENANT_ID
    session.agent_id = None
    session.call_metadata = call_metadata
    return session


def _make_transcript_messages(count=3):
    messages = []
    for i in range(count):
        msg = MagicMock()
        msg.role = "client" if i % 2 == 0 else "agent"
        msg.message = f"message {i}"
        messages.append(msg)
    return messages


class TestAnalyzeCallTranscriptInternalCache:
    async def test_returns_cached_analysis_without_calling_any_llm(self):
        cached_block = {
            "analysis": {"summary": "cached summary", "sentiment": "positive"},
            "model_used": "gpt-4o-mini",
            "timestamp": "2026-08-01T00:00:00+00:00",
        }
        call_session = _make_call_session(
            call_metadata={"llm_call_analysis": cached_block}
        )
        transcript_messages = _make_transcript_messages(4)
        db = MagicMock()

        with (
            patch(
                "app.routers.scheduled_calls.transcript_service.get_messages_by_session",
                return_value=transcript_messages,
            ) as mock_get_messages,
            patch(
                "app.routers.scheduled_calls.model_service.get_model_by_name"
            ) as mock_get_model,
            patch("app.services.gemini_service.GeminiService.generate_text") as mock_gemini,
            patch("app.services.openai_service.OpenAIService.generate_text") as mock_openai,
            patch("app.services.groq_service.GroqService.generate_text") as mock_groq,
        ):
            result = await analyze_call_transcript_internal(
                db=db, call_session=call_session, user=MagicMock()
            )

        assert result == {
            "analysis": cached_block["analysis"],
            "model_used": cached_block["model_used"],
            "transcript_message_count": 4,
        }
        mock_get_messages.assert_called_once_with(db, _CALL_ID)
        mock_get_model.assert_not_called()
        mock_gemini.assert_not_called()
        mock_openai.assert_not_called()
        mock_groq.assert_not_called()

    async def test_cache_hit_transcript_message_count_matches_actual_rows(self):
        """transcript_message_count in the cached response reflects the live
        transcript row count, not anything stashed in the cache block."""
        cached_block = {
            "analysis": {"summary": "s", "sentiment": "neutral"},
            "model_used": "gemini-2.0-flash",
            "timestamp": "2026-08-01T00:00:00+00:00",
        }
        call_session = _make_call_session(
            call_metadata={"llm_call_analysis": cached_block}
        )
        transcript_messages = _make_transcript_messages(7)
        db = MagicMock()

        with patch(
            "app.routers.scheduled_calls.transcript_service.get_messages_by_session",
            return_value=transcript_messages,
        ):
            result = await analyze_call_transcript_internal(
                db=db, call_session=call_session, user=MagicMock()
            )

        assert result["transcript_message_count"] == 7

    async def test_no_cache_falls_through_to_llm_generated_analysis(self):
        """Sanity check: the cache-check must not accidentally short-circuit
        or break the existing no-cache LLM-calling path."""
        call_session = _make_call_session(call_metadata=None)
        transcript_messages = _make_transcript_messages(2)
        db = MagicMock()

        mock_model = MagicMock()
        mock_model.model_name = "gpt-4o-mini"
        mock_model.provider.name = "openai"
        mock_model.api_key = None

        summary_response = {"content": "A brief call summary."}
        sentiment_response = {"content": "Overall sentiment: positive."}
        recommendations_response = {"content": "1. Follow up promptly."}

        with (
            patch(
                "app.routers.scheduled_calls.transcript_service.get_messages_by_session",
                return_value=transcript_messages,
            ),
            patch(
                "app.routers.scheduled_calls.model_service.get_model_by_name",
                return_value=mock_model,
            ),
            patch(
                "app.services.openai_service.OpenAIService.generate_text",
                side_effect=[summary_response, sentiment_response, recommendations_response],
            ) as mock_openai,
        ):
            result = await analyze_call_transcript_internal(
                db=db, call_session=call_session, user=MagicMock()
            )

        assert result is not None
        assert result["analysis"]["summary"] == "A brief call summary."
        assert result["analysis"]["sentiment"] == "Overall sentiment: positive."
        assert result["model_used"] == "gpt-4o-mini"
        assert result["transcript_message_count"] == 2
        assert mock_openai.call_count == 3

    async def test_no_transcript_messages_returns_none_when_no_cache(self):
        call_session = _make_call_session(call_metadata=None)
        db = MagicMock()

        with patch(
            "app.routers.scheduled_calls.transcript_service.get_messages_by_session",
            return_value=[],
        ):
            result = await analyze_call_transcript_internal(
                db=db, call_session=call_session, user=MagicMock()
            )

        assert result is None
