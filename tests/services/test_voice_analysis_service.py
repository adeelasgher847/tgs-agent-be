"""Unit tests for voice_analysis_service — transcript_summary persistence,
and the automatic post-call summary ARQ job (schedule_call_summary_generation
/ _generate_call_summary_arq_task).

Cross-session caller memory reads CallSession.transcript_summary, which must
be populated with the finalized, HIPAA-redacted summary at call-end analysis
time. Mirrors the mocking pattern from tests/api/v2/test_hipaa.py's
TestVoiceAnalysisHipaaRedaction.

The ARQ scheduling tests mirror tests/services/test_ghl_service.py's
TestPostCallWriteback ARQ-enqueue tests (`schedule_ghl_writeback` /
`_post_call_writeback_arq_task`), since `schedule_call_summary_generation`
uses the exact same get_arq_pool() + asyncio.get_running_loop() pattern.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

WORKSPACE_ID = uuid.uuid4()
FLOW_ID = uuid.uuid4()
CALL_ID = uuid.uuid4()
USER_ID = uuid.uuid4()
_SESSION_ID = uuid.uuid4()


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

        with (
            patch.object(
                vas_module.transcript_service, "get_messages_by_session", return_value=[msg]
            ),
            patch.object(
                vas_module.ModelService, "get_model_by_name", return_value=mock_model
            ),
            patch(
                "app.services.openai_service.OpenAIService.generate_text",
                return_value=analysis_text,
            ),
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


# ── Automatic call summary: ARQ scheduling ──────────────────────────────────


class TestScheduleCallSummaryGeneration:
    def test_fails_open_without_arq_pool(self):
        """No pool available yet (worker not initialized) — must not raise."""
        from app.services import voice_analysis_service as vas_module

        with patch.object(vas_module, "get_arq_pool", return_value=None):
            vas_module.schedule_call_summary_generation(_SESSION_ID)

    def test_enqueues_generate_call_summary_job_with_stringified_id(self):
        """Happy path with no running event loop (e.g. called synchronously
        from CallSessionService.update_call_session_status): must fall
        through to asyncio.run(_enqueue()), which really invokes
        pool.enqueue_job with the job name + stringified call_session_id."""
        from app.services import voice_analysis_service as vas_module

        mock_pool = MagicMock()
        mock_pool.enqueue_job = AsyncMock()

        with (
            patch.object(vas_module, "get_arq_pool", return_value=mock_pool),
            patch(
                "app.services.voice_analysis_service.asyncio.get_running_loop",
                side_effect=RuntimeError("no running loop"),
            ),
        ):
            vas_module.schedule_call_summary_generation(_SESSION_ID)

        mock_pool.enqueue_job.assert_called_once_with(
            "generate_call_summary", str(_SESSION_ID)
        )

    def test_enqueue_failure_is_caught_and_does_not_propagate(self):
        """A Redis/enqueue error inside _enqueue() must be caught and logged
        — summary generation is best-effort and must never blow up the
        caller (CallSessionService.update_call_session_status)."""
        from app.services import voice_analysis_service as vas_module

        mock_pool = MagicMock()
        mock_pool.enqueue_job = AsyncMock(side_effect=RuntimeError("redis down"))

        with (
            patch.object(vas_module, "get_arq_pool", return_value=mock_pool),
            patch(
                "app.services.voice_analysis_service.asyncio.get_running_loop",
                side_effect=RuntimeError("no running loop"),
            ),
        ):
            # Must not raise.
            vas_module.schedule_call_summary_generation(_SESSION_ID)

        mock_pool.enqueue_job.assert_called_once()

    def test_creates_task_when_loop_already_running(self):
        """When called from inside an async context (a running event loop),
        must hand off via asyncio.create_task instead of asyncio.run."""
        from app.services import voice_analysis_service as vas_module

        mock_pool = MagicMock()
        mock_pool.enqueue_job = AsyncMock()

        with (
            patch.object(vas_module, "get_arq_pool", return_value=mock_pool),
            patch(
                "app.services.voice_analysis_service.asyncio.get_running_loop",
                return_value=MagicMock(),
            ),
            patch(
                "app.services.voice_analysis_service.asyncio.create_task"
            ) as mock_create_task,
        ):
            vas_module.schedule_call_summary_generation(_SESSION_ID)

        mock_create_task.assert_called_once()
        # Close the never-awaited coroutine (create_task is mocked out, so it
        # was never scheduled) to avoid a "coroutine was never awaited" warning.
        mock_create_task.call_args[0][0].close()


# ── Automatic call summary: ARQ job entrypoint ──────────────────────────────


class TestGenerateCallSummaryArqTask:
    async def test_session_not_found_returns_cleanly(self):
        """Session was deleted (or the id is stale) between enqueue and job
        run — must log a warning and return without raising."""
        from app.services import voice_analysis_service as vas_module

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None

        with (
            patch("app.services.voice_analysis_service.SessionLocal", return_value=db),
            patch.object(
                vas_module.voice_analysis_service, "analyze_call_transcript"
            ) as mock_analyze,
        ):
            await vas_module._generate_call_summary_arq_task({}, str(uuid.uuid4()))

        mock_analyze.assert_not_called()
        db.close.assert_called_once()

    async def test_session_found_calls_analyze_with_raise_on_no_transcript_false(self):
        from app.services import voice_analysis_service as vas_module

        call_session = MagicMock()
        call_session.user_id = USER_ID
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = call_session

        session_id = uuid.uuid4()
        with (
            patch("app.services.voice_analysis_service.SessionLocal", return_value=db),
            patch.object(
                vas_module.voice_analysis_service, "analyze_call_transcript"
            ) as mock_analyze,
        ):
            await vas_module._generate_call_summary_arq_task({}, str(session_id))

        mock_analyze.assert_called_once_with(
            db, call_session, user_id=USER_ID, raise_on_no_transcript=False
        )
        db.close.assert_called_once()

    async def test_analyze_exception_is_caught_and_does_not_propagate(self):
        """analyze_call_transcript can raise HTTPException (e.g. no model
        configured) -- the ARQ job must swallow it, not crash the worker."""
        from app.services import voice_analysis_service as vas_module

        call_session = MagicMock()
        call_session.user_id = USER_ID
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = call_session

        with (
            patch("app.services.voice_analysis_service.SessionLocal", return_value=db),
            patch.object(
                vas_module.voice_analysis_service,
                "analyze_call_transcript",
                side_effect=HTTPException(status_code=404, detail="No model available"),
            ),
        ):
            # Must not raise.
            await vas_module._generate_call_summary_arq_task({}, str(uuid.uuid4()))

        db.close.assert_called_once()

    async def test_malformed_call_session_id_is_caught_and_does_not_propagate(self):
        """uuid.UUID(call_session_id) raising ValueError for a malformed id
        must also be swallowed by the same outer try/except -- not just
        session-not-found and analyze-time exceptions."""
        from app.services import voice_analysis_service as vas_module

        db = MagicMock()

        with (
            patch("app.services.voice_analysis_service.SessionLocal", return_value=db),
        ):
            # Must not raise even though "not-a-uuid" fails uuid.UUID(...).
            await vas_module._generate_call_summary_arq_task({}, "not-a-uuid")

        db.close.assert_called_once()


# ── Automatic call summary: concurrency-safe advisory lock ─────────────────


class TestGenerateCallSummaryArqTaskLocking:
    async def test_prelock_cache_hit_never_attempts_lock(self):
        """If llm_call_analysis is already cached before the lock is even
        attempted, analyze_call_transcript and the lock helper are both
        skipped entirely."""
        from app.services import voice_analysis_service as vas_module

        call_session = _make_call_session()
        call_session.call_metadata = {
            "llm_call_analysis": {"analysis": {"summary": "cached"}}
        }
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = call_session

        with (
            patch("app.services.voice_analysis_service.SessionLocal", return_value=db),
            patch.object(
                vas_module.voice_analysis_service, "analyze_call_transcript"
            ) as mock_analyze,
            patch.object(
                vas_module, "_try_acquire_call_summary_lock"
            ) as mock_try_lock,
            patch.object(
                vas_module, "_release_call_summary_lock"
            ) as mock_release,
        ):
            await vas_module._generate_call_summary_arq_task({}, str(uuid.uuid4()))

        mock_analyze.assert_not_called()
        mock_try_lock.assert_not_called()
        mock_release.assert_not_called()
        db.close.assert_called_once()

    async def test_lock_not_acquired_skips_quietly_without_analyzing(self):
        """Simulated contention: another worker holds the lock -- this
        worker must return cleanly without calling analyze_call_transcript
        or attempting a release (it never acquired anything)."""
        from app.services import voice_analysis_service as vas_module

        call_session = _make_call_session()
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = call_session

        with (
            patch("app.services.voice_analysis_service.SessionLocal", return_value=db),
            patch.object(
                vas_module.voice_analysis_service, "analyze_call_transcript"
            ) as mock_analyze,
            patch.object(
                vas_module, "_try_acquire_call_summary_lock", return_value=False
            ) as mock_try_lock,
            patch.object(
                vas_module, "_release_call_summary_lock"
            ) as mock_release,
        ):
            await vas_module._generate_call_summary_arq_task({}, str(uuid.uuid4()))

        mock_try_lock.assert_called_once()
        mock_analyze.assert_not_called()
        mock_release.assert_not_called()
        db.close.assert_called_once()

    async def test_lock_acquired_no_cache_analyzes_then_releases(self):
        """Happy path: lock acquired, no cache present -- analyze_call_transcript
        runs exactly once and the lock is always released afterward."""
        from app.services import voice_analysis_service as vas_module

        call_session = _make_call_session()
        call_session.call_metadata = {}
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = call_session

        calls = []

        def _analyze(*args, **kwargs):
            calls.append("analyze")

        def _release(*args, **kwargs):
            calls.append("release")

        with (
            patch("app.services.voice_analysis_service.SessionLocal", return_value=db),
            patch.object(
                vas_module.voice_analysis_service,
                "analyze_call_transcript",
                side_effect=_analyze,
            ) as mock_analyze,
            patch.object(
                vas_module, "_try_acquire_call_summary_lock", return_value=True
            ) as mock_try_lock,
            patch.object(
                vas_module, "_release_call_summary_lock", side_effect=_release
            ) as mock_release,
        ):
            await vas_module._generate_call_summary_arq_task({}, str(uuid.uuid4()))

        mock_try_lock.assert_called_once()
        mock_analyze.assert_called_once_with(
            db, call_session, user_id=call_session.user_id, raise_on_no_transcript=False
        )
        mock_release.assert_called_once()
        assert calls == ["analyze", "release"]
        db.close.assert_called_once()

    async def test_double_checked_lock_cache_hit_skips_analyze_but_still_releases(self):
        """The race-closing case: lock acquired, but db.refresh() reveals a
        concurrent winner already committed the cached analysis while this
        worker waited for/acquired the lock -- analyze_call_transcript must
        NOT run, but the lock must still be released (not leaked)."""
        from app.services import voice_analysis_service as vas_module

        call_session = _make_call_session()
        call_session.call_metadata = {}

        def _refresh(obj):
            obj.call_metadata = {
                "llm_call_analysis": {"analysis": {"summary": "won by other worker"}}
            }

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = call_session
        db.refresh = MagicMock(side_effect=_refresh)

        with (
            patch("app.services.voice_analysis_service.SessionLocal", return_value=db),
            patch.object(
                vas_module.voice_analysis_service, "analyze_call_transcript"
            ) as mock_analyze,
            patch.object(
                vas_module, "_try_acquire_call_summary_lock", return_value=True
            ),
            patch.object(
                vas_module, "_release_call_summary_lock"
            ) as mock_release,
        ):
            await vas_module._generate_call_summary_arq_task({}, str(uuid.uuid4()))

        mock_analyze.assert_not_called()
        mock_release.assert_called_once()
        db.close.assert_called_once()

    async def test_lock_released_even_when_analyze_raises(self):
        """If analyze_call_transcript raises, the finally-block must still
        release the lock, and the outer try/except must swallow the
        exception so it doesn't propagate out of the ARQ job."""
        from app.services import voice_analysis_service as vas_module

        call_session = _make_call_session()
        call_session.call_metadata = {}
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = call_session

        with (
            patch("app.services.voice_analysis_service.SessionLocal", return_value=db),
            patch.object(
                vas_module.voice_analysis_service,
                "analyze_call_transcript",
                side_effect=RuntimeError("LLM provider error"),
            ),
            patch.object(
                vas_module, "_try_acquire_call_summary_lock", return_value=True
            ),
            patch.object(
                vas_module, "_release_call_summary_lock"
            ) as mock_release,
        ):
            # Must not raise -- outer try/except in the ARQ task catches it.
            await vas_module._generate_call_summary_arq_task({}, str(uuid.uuid4()))

        mock_release.assert_called_once()
        db.close.assert_called_once()

    def test_advisory_lock_helpers_no_op_on_sqlite(self, db):
        """Realistic smoke test against this repo's SQLite test db fixture
        (no mocking): the postgres-dialect guard must make both helpers
        no-op cleanly, proving the guard doesn't break anything on the
        dialect this whole suite actually runs against."""
        from app.services.voice_analysis_service import (
            _try_acquire_call_summary_lock,
            _release_call_summary_lock,
        )

        session_id = uuid.uuid4()

        assert _try_acquire_call_summary_lock(db, session_id) is True
        # Must complete without raising -- no return value contract.
        _release_call_summary_lock(db, session_id)


class TestPgAdvisoryKeyPair:
    def test_returns_deterministic_two_int_tuple(self):
        from app.services.voice_analysis_service import _pg_advisory_key_pair

        key_uuid = uuid.uuid4()

        result1 = _pg_advisory_key_pair(key_uuid)
        result2 = _pg_advisory_key_pair(key_uuid)

        assert result1 == result2
        assert len(result1) == 2
        assert all(isinstance(k, int) for k in result1)
