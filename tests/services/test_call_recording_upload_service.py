"""Unit tests for call_recording_upload_service — the ARQ-based recording
finalization job (schedule_recording_upload / _finalize_call_recording_arq_task).

This replaces the old one-shot in-process LiveKit egress check (which wrongly
marked essentially every real call's recording as permanently failed because
LiveKit is normally still ENDING, not FAILED, right after stop_egress) with a
proper retry-with-backoff chain via ARQ.

Mirrors the mocking pattern in tests/services/test_voice_analysis_service.py's
TestScheduleCallSummaryGeneration / TestGenerateCallSummaryArqTask, since
schedule_recording_upload / _finalize_call_recording_arq_task follow the same
get_arq_pool() + asyncio.get_running_loop() + ctx["redis"] shape.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

_SESSION_ID = uuid.uuid4()


def _make_call_session(*, recording_s3_path=None, egress_id="EG_1", gcs_path="s3://bucket/key.mp3"):
    session = MagicMock()
    session.id = _SESSION_ID
    session.tenant_id = uuid.uuid4()
    session.agent_id = uuid.uuid4()
    session.duration = 42
    session.recording_s3_path = recording_s3_path
    session.recording_error = False
    session.call_metadata = {"recording": {"egress_id": egress_id, "gcs_path": gcs_path}}
    return session


def _make_db(session):
    db = MagicMock()
    db.get.return_value = session
    db.commit = MagicMock()
    db.rollback = MagicMock()
    db.close = MagicMock()
    return db


def _egress_info(status: int):
    info = MagicMock()
    info.status = status
    return info


# ── schedule_recording_upload ───────────────────────────────────────────────


class TestScheduleRecordingUpload:
    def test_fails_open_without_arq_pool(self):
        """Pool not ready yet (worker not initialized) — must not raise."""
        from app.services import call_recording_upload_service as cru_module

        with patch("app.utils.arq_pool.get_arq_pool", return_value=None):
            cru_module.schedule_recording_upload(_SESSION_ID)

    def test_enqueues_finalize_job_with_stringified_id_and_attempt_1_no_running_loop(self):
        """Happy path with no running event loop must fall through to
        asyncio.run(_enqueue()), which really invokes pool.enqueue_job with
        the job name, stringified call_session_id, attempt=1 and the
        module's initial defer constant."""
        from app.services import call_recording_upload_service as cru_module

        mock_pool = MagicMock()
        mock_pool.enqueue_job = AsyncMock()

        with (
            patch("app.utils.arq_pool.get_arq_pool", return_value=mock_pool),
            patch(
                "app.services.call_recording_upload_service.asyncio.get_running_loop",
                side_effect=RuntimeError("no running loop"),
            ),
        ):
            cru_module.schedule_recording_upload(_SESSION_ID)

        mock_pool.enqueue_job.assert_called_once_with(
            "finalize_call_recording",
            str(_SESSION_ID),
            1,
            _defer_by=cru_module._INITIAL_FINALIZE_DELAY_SECONDS,
            _job_id=f"finalize_recording:{_SESSION_ID}:1",
        )

    def test_creates_task_when_loop_already_running(self):
        """Called from inside an async context — must hand off via
        asyncio.create_task instead of asyncio.run."""
        from app.services import call_recording_upload_service as cru_module

        mock_pool = MagicMock()
        mock_pool.enqueue_job = AsyncMock()

        with (
            patch("app.utils.arq_pool.get_arq_pool", return_value=mock_pool),
            patch(
                "app.services.call_recording_upload_service.asyncio.get_running_loop",
                return_value=MagicMock(),
            ),
            patch(
                "app.services.call_recording_upload_service.asyncio.create_task"
            ) as mock_create_task,
        ):
            cru_module.schedule_recording_upload(_SESSION_ID)

        mock_create_task.assert_called_once()
        # create_task is mocked out (never actually scheduled) — close the
        # coroutine to avoid a "coroutine was never awaited" warning.
        mock_create_task.call_args[0][0].close()

    def test_enqueue_failure_is_caught_and_does_not_propagate(self):
        """A Redis/enqueue error inside _enqueue() must be swallowed."""
        from app.services import call_recording_upload_service as cru_module

        mock_pool = MagicMock()
        mock_pool.enqueue_job = AsyncMock(side_effect=RuntimeError("redis down"))

        with (
            patch("app.utils.arq_pool.get_arq_pool", return_value=mock_pool),
            patch(
                "app.services.call_recording_upload_service.asyncio.get_running_loop",
                side_effect=RuntimeError("no running loop"),
            ),
        ):
            cru_module.schedule_recording_upload(_SESSION_ID)

        mock_pool.enqueue_job.assert_called_once()


# ── _finalize_call_recording_arq_task: idempotency / early-return guards ───


class TestFinalizeArqTaskGuards:
    async def test_session_not_found_returns_cleanly(self):
        from app.services import call_recording_upload_service as cru_module

        db = MagicMock()
        db.get.return_value = None

        with patch("app.db.session.SessionLocal", return_value=db):
            await cru_module._finalize_call_recording_arq_task({}, str(uuid.uuid4()), 1)

        db.close.assert_called_once()
        db.commit.assert_not_called()

    async def test_recording_not_enabled_returns_without_touching_status(self):
        from app.services import call_recording_upload_service as cru_module

        session = _make_call_session()
        db = _make_db(session)

        with (
            patch("app.db.session.SessionLocal", return_value=db),
            patch(
                "app.services.recording_config_service.get_recording_enabled_for_call",
                return_value=False,
            ),
        ):
            await cru_module._finalize_call_recording_arq_task({}, str(_SESSION_ID), 1)

        db.commit.assert_not_called()
        assert session.recording_error is False
        db.close.assert_called_once()

    async def test_already_has_recording_s3_path_skips_refinalize(self):
        from app.services import call_recording_upload_service as cru_module

        session = _make_call_session(recording_s3_path="s3://already/there.mp3")
        db = _make_db(session)

        with (
            patch("app.db.session.SessionLocal", return_value=db),
            patch(
                "app.services.recording_config_service.get_recording_enabled_for_call",
                return_value=True,
            ),
            patch(
                "app.services.livekit_recording_service.livekit_recording_service"
            ) as mock_lk,
        ):
            await cru_module._finalize_call_recording_arq_task({}, str(_SESSION_ID), 1)

        mock_lk.stop_room_recording.assert_not_called()
        mock_lk.get_egress_info.assert_not_called()
        db.commit.assert_not_called()
        db.close.assert_called_once()

    async def test_missing_egress_id_or_gcs_path_skips_without_calling_livekit(self):
        from app.services import call_recording_upload_service as cru_module

        session = _make_call_session()
        session.call_metadata = {"recording": {"egress_id": None, "gcs_path": None}}
        db = _make_db(session)

        with (
            patch("app.db.session.SessionLocal", return_value=db),
            patch(
                "app.services.recording_config_service.get_recording_enabled_for_call",
                return_value=True,
            ),
            patch(
                "app.services.livekit_recording_service.livekit_recording_service"
            ) as mock_lk,
        ):
            await cru_module._finalize_call_recording_arq_task({}, str(_SESSION_ID), 1)

        mock_lk.stop_room_recording.assert_not_called()
        mock_lk.get_egress_info.assert_not_called()
        db.close.assert_called_once()


# ── Core fix: ENDING must retry, not fail ───────────────────────────────────


class TestFinalizeArqTaskRetryOnInProgress:
    async def test_ending_status_retries_and_does_not_mark_error(self):
        """EGRESS_ENDING (status=2) on attempt 1 must schedule a retry via
        ctx['redis'].enqueue_job and must NOT mark recording_error."""
        from app.services import call_recording_upload_service as cru_module

        session = _make_call_session()
        db = _make_db(session)

        mock_lk = MagicMock()
        mock_lk.stop_room_recording = AsyncMock(return_value=True)
        mock_lk.get_egress_info = AsyncMock(return_value=_egress_info(2))

        mock_redis = MagicMock()
        mock_redis.enqueue_job = AsyncMock()
        ctx = {"redis": mock_redis}

        with (
            patch("app.db.session.SessionLocal", return_value=db),
            patch(
                "app.services.recording_config_service.get_recording_enabled_for_call",
                return_value=True,
            ),
            patch(
                "app.services.livekit_recording_service.livekit_recording_service",
                mock_lk,
            ),
        ):
            await cru_module._finalize_call_recording_arq_task(ctx, str(_SESSION_ID), 1)

        db.commit.assert_not_called()
        assert session.recording_error is False
        mock_redis.enqueue_job.assert_called_once_with(
            "finalize_call_recording",
            str(_SESSION_ID),
            2,
            _defer_by=cru_module._FINALIZE_RETRY_DELAY_SECONDS,
            _job_id=f"finalize_recording:{_SESSION_ID}:2",
        )

    async def test_get_egress_info_raising_is_treated_as_inconclusive_and_retries(self):
        """An exception fetching egress status must not be an immediate
        failure — it's treated the same as status=None, i.e. retried."""
        from app.services import call_recording_upload_service as cru_module

        session = _make_call_session()
        db = _make_db(session)

        mock_lk = MagicMock()
        mock_lk.stop_room_recording = AsyncMock(return_value=True)
        mock_lk.get_egress_info = AsyncMock(side_effect=RuntimeError("livekit api blip"))

        mock_redis = MagicMock()
        mock_redis.enqueue_job = AsyncMock()
        ctx = {"redis": mock_redis}

        with (
            patch("app.db.session.SessionLocal", return_value=db),
            patch(
                "app.services.recording_config_service.get_recording_enabled_for_call",
                return_value=True,
            ),
            patch(
                "app.services.livekit_recording_service.livekit_recording_service",
                mock_lk,
            ),
        ):
            await cru_module._finalize_call_recording_arq_task(ctx, str(_SESSION_ID), 1)

        db.commit.assert_not_called()
        assert session.recording_error is False
        mock_redis.enqueue_job.assert_called_once_with(
            "finalize_call_recording",
            str(_SESSION_ID),
            2,
            _defer_by=cru_module._FINALIZE_RETRY_DELAY_SECONDS,
            _job_id=f"finalize_recording:{_SESSION_ID}:2",
        )

    async def test_full_ending_ending_complete_chain_across_invocations(self):
        """Simulate the real-world regression scenario: three sequential ARQ
        invocations — ENDING (attempt 1) -> ENDING (attempt 2) -> COMPLETE
        (attempt 3) — using the same session/db. stop_room_recording must be
        called exactly once total (attempt 1 only); the final invocation
        must set recording_s3_path and clear recording_error."""
        from app.services import call_recording_upload_service as cru_module

        session = _make_call_session()
        db = _make_db(session)

        mock_lk = MagicMock()
        mock_lk.stop_room_recording = AsyncMock(return_value=True)
        mock_lk.get_egress_info = AsyncMock(
            side_effect=[_egress_info(2), _egress_info(2), _egress_info(3)]
        )

        mock_redis = MagicMock()
        mock_redis.enqueue_job = AsyncMock()
        ctx = {"redis": mock_redis}

        with (
            patch("app.db.session.SessionLocal", return_value=db),
            patch(
                "app.services.recording_config_service.get_recording_enabled_for_call",
                return_value=True,
            ),
            patch(
                "app.services.livekit_recording_service.livekit_recording_service",
                mock_lk,
            ),
            patch(
                "app.services.s3_recording_service.update_object_metadata"
            ) as mock_update_meta,
        ):
            await cru_module._finalize_call_recording_arq_task(ctx, str(_SESSION_ID), 1)
            await cru_module._finalize_call_recording_arq_task(ctx, str(_SESSION_ID), 2)
            await cru_module._finalize_call_recording_arq_task(ctx, str(_SESSION_ID), 3)

        mock_lk.stop_room_recording.assert_called_once()
        assert mock_lk.get_egress_info.await_count == 3
        mock_update_meta.assert_called_once()
        assert session.recording_s3_path == "s3://bucket/key.mp3"
        assert session.recording_error is False


# ── Terminal failure: FAILED / ABORTED never retries ────────────────────────


class TestFinalizeArqTaskTerminalFailure:
    async def test_failed_or_aborted_status_marks_error_immediately_no_retry(self):
        """EGRESS_FAILED (4) and EGRESS_ABORTED (5) must be immediate
        terminal failures — no retry, even on attempt==1."""
        from app.services import call_recording_upload_service as cru_module

        for status in (4, 5):
            session = _make_call_session()
            db = _make_db(session)

            mock_lk = MagicMock()
            mock_lk.stop_room_recording = AsyncMock(return_value=True)
            mock_lk.get_egress_info = AsyncMock(return_value=_egress_info(status))

            mock_redis = MagicMock()
            mock_redis.enqueue_job = AsyncMock()
            ctx = {"redis": mock_redis}

            with (
                patch("app.db.session.SessionLocal", return_value=db),
                patch(
                    "app.services.recording_config_service.get_recording_enabled_for_call",
                    return_value=True,
                ),
                patch(
                    "app.services.livekit_recording_service.livekit_recording_service",
                    mock_lk,
                ),
            ):
                await cru_module._finalize_call_recording_arq_task(
                    ctx, str(_SESSION_ID), 1
                )

            assert session.recording_error is True
            db.commit.assert_called_once()
            mock_redis.enqueue_job.assert_not_called()


# ── Retry exhaustion ─────────────────────────────────────────────────────────


class TestFinalizeArqTaskRetryExhaustion:
    async def test_max_attempts_reached_marks_error_instead_of_retrying(self):
        from app.services import call_recording_upload_service as cru_module

        session = _make_call_session()
        db = _make_db(session)

        mock_lk = MagicMock()
        mock_lk.stop_room_recording = AsyncMock(return_value=True)
        mock_lk.get_egress_info = AsyncMock(return_value=_egress_info(2))

        mock_redis = MagicMock()
        mock_redis.enqueue_job = AsyncMock()
        ctx = {"redis": mock_redis}

        with (
            patch("app.db.session.SessionLocal", return_value=db),
            patch(
                "app.services.recording_config_service.get_recording_enabled_for_call",
                return_value=True,
            ),
            patch(
                "app.services.livekit_recording_service.livekit_recording_service",
                mock_lk,
            ),
        ):
            await cru_module._finalize_call_recording_arq_task(
                ctx, str(_SESSION_ID), cru_module._MAX_FINALIZE_ATTEMPTS
            )

        assert session.recording_error is True
        db.commit.assert_called_once()
        mock_redis.enqueue_job.assert_not_called()


# ── stop_room_recording called only on attempt==1 ───────────────────────────


class TestFinalizeArqTaskStopEgressOnlyFirstAttempt:
    async def test_attempt_1_calls_stop_room_recording(self):
        from app.services import call_recording_upload_service as cru_module

        session = _make_call_session()
        db = _make_db(session)

        mock_lk = MagicMock()
        mock_lk.stop_room_recording = AsyncMock(return_value=True)
        mock_lk.get_egress_info = AsyncMock(return_value=_egress_info(2))

        mock_redis = MagicMock()
        mock_redis.enqueue_job = AsyncMock()
        ctx = {"redis": mock_redis}

        with (
            patch("app.db.session.SessionLocal", return_value=db),
            patch(
                "app.services.recording_config_service.get_recording_enabled_for_call",
                return_value=True,
            ),
            patch(
                "app.services.livekit_recording_service.livekit_recording_service",
                mock_lk,
            ),
        ):
            await cru_module._finalize_call_recording_arq_task(ctx, str(_SESSION_ID), 1)

        mock_lk.stop_room_recording.assert_called_once()

    async def test_attempt_greater_than_1_does_not_call_stop_room_recording(self):
        from app.services import call_recording_upload_service as cru_module

        session = _make_call_session()
        db = _make_db(session)

        mock_lk = MagicMock()
        mock_lk.stop_room_recording = AsyncMock(return_value=True)
        mock_lk.get_egress_info = AsyncMock(return_value=_egress_info(2))

        mock_redis = MagicMock()
        mock_redis.enqueue_job = AsyncMock()
        ctx = {"redis": mock_redis}

        with (
            patch("app.db.session.SessionLocal", return_value=db),
            patch(
                "app.services.recording_config_service.get_recording_enabled_for_call",
                return_value=True,
            ),
            patch(
                "app.services.livekit_recording_service.livekit_recording_service",
                mock_lk,
            ),
        ):
            await cru_module._finalize_call_recording_arq_task(ctx, str(_SESSION_ID), 2)

        mock_lk.stop_room_recording.assert_not_called()


# ── ARQ pool unavailable mid-chain ──────────────────────────────────────────


class TestFinalizeArqTaskNoPoolAvailable:
    async def test_no_redis_pool_falls_through_to_mark_error(self):
        """ctx.get('redis') is None during what would otherwise be a retry
        (non-terminal status, attempts remaining) — since we can't
        reschedule, this must fall through to _mark_recording_error rather
        than silently dropping the call."""
        from app.services import call_recording_upload_service as cru_module

        session = _make_call_session()
        db = _make_db(session)

        mock_lk = MagicMock()
        mock_lk.stop_room_recording = AsyncMock(return_value=True)
        mock_lk.get_egress_info = AsyncMock(return_value=_egress_info(2))

        ctx = {"redis": None}

        with (
            patch("app.db.session.SessionLocal", return_value=db),
            patch(
                "app.services.recording_config_service.get_recording_enabled_for_call",
                return_value=True,
            ),
            patch(
                "app.services.livekit_recording_service.livekit_recording_service",
                mock_lk,
            ),
        ):
            await cru_module._finalize_call_recording_arq_task(ctx, str(_SESSION_ID), 1)

        assert session.recording_error is True
        db.commit.assert_called_once()


# ── enqueue_job raising during retry falls through to mark_recording_error ─


class TestFinalizeArqTaskEnqueueRaisesDuringRetry:
    async def test_enqueue_job_raising_during_retry_falls_through_to_mark_recording_error(
        self,
    ):
        """A non-terminal status with attempts remaining should trigger a
        retry-reschedule; if ctx['redis'].enqueue_job itself raises (e.g. a
        Redis connection blip), the local try/except around that call must
        catch it and fall through to _mark_recording_error — not let the
        exception propagate to the outer catch-all and strand the call
        session with no path forward."""
        from app.services import call_recording_upload_service as cru_module

        session = _make_call_session()
        db = _make_db(session)

        mock_lk = MagicMock()
        mock_lk.stop_room_recording = AsyncMock(return_value=True)
        mock_lk.get_egress_info = AsyncMock(return_value=_egress_info(2))

        mock_redis = MagicMock()
        mock_redis.enqueue_job = AsyncMock(side_effect=RuntimeError("redis connection lost"))
        ctx = {"redis": mock_redis}

        with (
            patch("app.db.session.SessionLocal", return_value=db),
            patch(
                "app.services.recording_config_service.get_recording_enabled_for_call",
                return_value=True,
            ),
            patch(
                "app.services.livekit_recording_service.livekit_recording_service",
                mock_lk,
            ),
        ):
            # Must return cleanly — no exception should propagate out.
            await cru_module._finalize_call_recording_arq_task(ctx, str(_SESSION_ID), 1)

        mock_redis.enqueue_job.assert_called_once()
        assert session.recording_error is True
        db.commit.assert_called_once()
        db.close.assert_called_once()
