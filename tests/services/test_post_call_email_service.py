"""Unit tests for app/services/post_call_email_service.py.

Covers:
  - schedule_post_call_email_summary: ARQ enqueue happy path, fails open
    without a pool, fails open on enqueue error, hands off via
    asyncio.create_task when a loop is already running. Mirrors
    tests/services/test_voice_analysis_service.py's
    TestScheduleCallSummaryGeneration.
  - _send_post_call_email_summary_impl: recipient-list assembly (both
    toggles off/on/combined + dedupe), empty-recipients skip, cached vs.
    lazily-computed LLM analysis, and fail-open exception handling. Mirrors
    tests/services/test_voice_analysis_service.py's
    TestGenerateCallSummaryArqTask db-mocking pattern.

email_service.send_generic_email is always mocked — never a real AWS SES
call. ARQ pool interactions are mocked; no real Redis connection required.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch


WORKSPACE_ID = uuid.uuid4()
FLOW_ID = uuid.uuid4()
SESSION_ID = uuid.uuid4()
USER_ID = uuid.uuid4()


def _make_call_session(
    *,
    call_flow_id=FLOW_ID,
    call_metadata=None,
    transcript_summary=None,
    call_type="inbound",
    from_number=None,
    to_number=None,
):
    session = MagicMock()
    session.id = SESSION_ID
    session.tenant_id = WORKSPACE_ID
    session.call_flow_id = call_flow_id
    session.call_metadata = call_metadata
    session.transcript_summary = transcript_summary
    session.user_id = USER_ID
    # Explicit (not MagicMock-default) so _resolve_caller_display's string
    # comparisons/concatenation behave deterministically in tests.
    session.call_type = call_type
    session.from_number = from_number
    session.to_number = to_number
    return session


def _make_call_flow(
    *,
    email_summary_enabled=False,
    email_summary_recipients=None,
    summary_to_business_owner_enabled=False,
    post_call_analysis_variables=None,
):
    flow = MagicMock()
    flow.id = FLOW_ID
    flow.name = "Test Flow"
    flow.email_summary_enabled = email_summary_enabled
    flow.email_summary_recipients = email_summary_recipients or []
    flow.summary_to_business_owner_enabled = summary_to_business_owner_enabled
    flow.post_call_analysis_variables = post_call_analysis_variables or []
    flow.hipaa_compliance = False
    return flow


def _make_db(call_session, call_flow, owner_user=None, owner_user_id=None):
    """Mirrors the db.query.side_effect dispatch pattern from
    tests/services/test_voice_analysis_service.py::_make_db, extended with
    db.execute(...) for the is_creator owner lookup."""
    from app.models.call_flow import CallFlow
    from app.models.call_session import CallSession
    from app.models.user import User

    db = MagicMock()

    def _query(model):
        q = MagicMock()
        if model is CallSession:
            q.filter.return_value.first.return_value = call_session
        elif model is CallFlow:
            q.filter.return_value.first.return_value = call_flow
        elif model is User:
            q.filter.return_value.first.return_value = owner_user
        else:
            q.filter.return_value.first.return_value = None
        return q

    db.query.side_effect = _query

    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = owner_user_id
    db.execute.return_value = execute_result

    return db


# ── schedule_post_call_email_summary (ARQ enqueue) ──────────────────────────


class TestSchedulePostCallEmailSummary:
    def test_fails_open_without_arq_pool(self):
        from app.services import post_call_email_service as pces

        with patch.object(pces, "get_arq_pool", return_value=None):
            pces.schedule_post_call_email_summary(SESSION_ID)

    def test_enqueues_send_post_call_email_summary_job_with_stringified_id(self):
        from app.services import post_call_email_service as pces

        mock_pool = MagicMock()
        mock_pool.enqueue_job = AsyncMock()

        with (
            patch.object(pces, "get_arq_pool", return_value=mock_pool),
            patch(
                "app.services.post_call_email_service.asyncio.get_running_loop",
                side_effect=RuntimeError("no running loop"),
            ),
        ):
            pces.schedule_post_call_email_summary(SESSION_ID)

        mock_pool.enqueue_job.assert_called_once_with(
            "send_post_call_email_summary", str(SESSION_ID)
        )

    def test_enqueue_failure_is_caught_and_does_not_propagate(self):
        from app.services import post_call_email_service as pces

        mock_pool = MagicMock()
        mock_pool.enqueue_job = AsyncMock(side_effect=RuntimeError("redis down"))

        with (
            patch.object(pces, "get_arq_pool", return_value=mock_pool),
            patch(
                "app.services.post_call_email_service.asyncio.get_running_loop",
                side_effect=RuntimeError("no running loop"),
            ),
        ):
            pces.schedule_post_call_email_summary(SESSION_ID)

        mock_pool.enqueue_job.assert_called_once()

    def test_creates_task_when_loop_already_running(self):
        from app.services import post_call_email_service as pces

        mock_pool = MagicMock()
        mock_pool.enqueue_job = AsyncMock()

        with (
            patch.object(pces, "get_arq_pool", return_value=mock_pool),
            patch(
                "app.services.post_call_email_service.asyncio.get_running_loop",
                return_value=MagicMock(),
            ),
            patch(
                "app.services.post_call_email_service.asyncio.create_task"
            ) as mock_create_task,
        ):
            pces.schedule_post_call_email_summary(SESSION_ID)

        mock_create_task.assert_called_once()
        mock_create_task.call_args[0][0].close()


# ── _send_post_call_email_summary_impl ──────────────────────────────────────


class TestSendPostCallEmailSummaryImpl:
    def test_session_not_found_skips_cleanly(self):
        from app.services import post_call_email_service as pces

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_send.assert_not_called()
        db.close.assert_called_once()

    def test_both_toggles_off_sends_no_email(self):
        from app.services import post_call_email_service as pces

        call_session = _make_call_session()
        call_flow = _make_call_flow(
            email_summary_enabled=False, summary_to_business_owner_enabled=False
        )
        db = _make_db(call_session, call_flow)

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
            patch(
                "app.services.voice_analysis_service.voice_analysis_service.analyze_call_transcript"
            ) as mock_analyze,
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_send.assert_not_called()
        mock_analyze.assert_not_called()
        db.close.assert_called_once()

    def test_missing_call_flow_sends_no_email(self):
        from app.services import post_call_email_service as pces

        call_session = _make_call_session(call_flow_id=None)
        db = _make_db(call_session, call_flow=None)

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_send.assert_not_called()

    def test_only_email_summary_enabled_sends_to_configured_recipients_only(self):
        from app.services import post_call_email_service as pces

        call_session = _make_call_session(
            call_metadata={"llm_call_analysis": {"analysis": {"summary": "cached"}}}
        )
        call_flow = _make_call_flow(
            email_summary_enabled=True,
            email_summary_recipients=["a@example.com", "b@example.com"],
            summary_to_business_owner_enabled=False,
        )
        db = _make_db(call_session, call_flow, owner_user_id=None)

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
            patch(
                "app.services.voice_analysis_service.voice_analysis_service.analyze_call_transcript"
            ) as mock_analyze,
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_analyze.assert_not_called()  # analysis already cached
        mock_send.assert_called_once()
        kwargs = mock_send.call_args.kwargs
        assert kwargs["to_email"] == "a@example.com"
        assert kwargs["cc_emails"] == ["b@example.com"]

    def test_only_business_owner_toggle_sends_to_owner_email_only(self):
        from app.services import post_call_email_service as pces

        call_session = _make_call_session(
            call_metadata={"llm_call_analysis": {"analysis": {"summary": "cached"}}}
        )
        call_flow = _make_call_flow(
            email_summary_enabled=False,
            summary_to_business_owner_enabled=True,
        )
        owner_user_id = uuid.uuid4()
        owner_user = MagicMock()
        owner_user.id = owner_user_id
        owner_user.email = "owner@example.com"
        db = _make_db(
            call_session, call_flow, owner_user=owner_user, owner_user_id=owner_user_id
        )

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_send.assert_called_once()
        kwargs = mock_send.call_args.kwargs
        assert kwargs["to_email"] == "owner@example.com"
        assert kwargs["cc_emails"] is None

    def test_no_owner_row_found_and_only_owner_toggle_enabled_skips_send(self):
        """No is_creator row for the tenant (edge case / data drift) — must
        skip cleanly rather than erroring, since there are no recipients."""
        from app.services import post_call_email_service as pces

        call_session = _make_call_session(
            call_metadata={"llm_call_analysis": {"analysis": {"summary": "cached"}}}
        )
        call_flow = _make_call_flow(
            email_summary_enabled=False,
            summary_to_business_owner_enabled=True,
        )
        db = _make_db(call_session, call_flow, owner_user=None, owner_user_id=None)

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_send.assert_not_called()

    def test_both_toggles_on_dedupes_recipients(self):
        """Owner's email already present in the configured recipients list —
        must not be sent twice / appear twice in cc_emails."""
        from app.services import post_call_email_service as pces

        call_session = _make_call_session(
            call_metadata={"llm_call_analysis": {"analysis": {"summary": "cached"}}}
        )
        call_flow = _make_call_flow(
            email_summary_enabled=True,
            email_summary_recipients=["owner@example.com", "b@example.com"],
            summary_to_business_owner_enabled=True,
        )
        owner_user_id = uuid.uuid4()
        owner_user = MagicMock()
        owner_user.id = owner_user_id
        owner_user.email = "owner@example.com"
        db = _make_db(
            call_session, call_flow, owner_user=owner_user, owner_user_id=owner_user_id
        )

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_send.assert_called_once()
        kwargs = mock_send.call_args.kwargs
        assert kwargs["to_email"] == "owner@example.com"
        assert kwargs["cc_emails"] == ["b@example.com"]

    def test_both_toggles_on_distinct_recipients_combined(self):
        from app.services import post_call_email_service as pces

        call_session = _make_call_session(
            call_metadata={"llm_call_analysis": {"analysis": {"summary": "cached"}}}
        )
        call_flow = _make_call_flow(
            email_summary_enabled=True,
            email_summary_recipients=["a@example.com"],
            summary_to_business_owner_enabled=True,
        )
        owner_user_id = uuid.uuid4()
        owner_user = MagicMock()
        owner_user.id = owner_user_id
        owner_user.email = "owner@example.com"
        db = _make_db(
            call_session, call_flow, owner_user=owner_user, owner_user_id=owner_user_id
        )

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_send.assert_called_once()
        kwargs = mock_send.call_args.kwargs
        assert kwargs["to_email"] == "a@example.com"
        assert kwargs["cc_emails"] == ["owner@example.com"]

    def test_empty_final_recipient_list_skips_send_without_error(self):
        """email_summary_enabled with an empty recipients list, and no owner
        toggle — final list is empty, must skip without raising."""
        from app.services import post_call_email_service as pces

        call_session = _make_call_session(
            call_metadata={"llm_call_analysis": {"analysis": {"summary": "cached"}}}
        )
        call_flow = _make_call_flow(
            email_summary_enabled=True,
            email_summary_recipients=[],
            summary_to_business_owner_enabled=False,
        )
        db = _make_db(call_session, call_flow)

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_send.assert_not_called()

    def test_cached_analysis_is_reused_not_recomputed(self):
        from app.services import post_call_email_service as pces

        call_session = _make_call_session(
            call_metadata={
                "llm_call_analysis": {"analysis": {"summary": "already computed"}}
            }
        )
        call_flow = _make_call_flow(
            email_summary_enabled=True, email_summary_recipients=["a@example.com"]
        )
        db = _make_db(call_session, call_flow)

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email"),
            patch(
                "app.services.voice_analysis_service.voice_analysis_service.analyze_call_transcript"
            ) as mock_analyze,
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_analyze.assert_not_called()

    def test_missing_analysis_triggers_lazy_compute(self):
        from app.services import post_call_email_service as pces

        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow(
            email_summary_enabled=True, email_summary_recipients=["a@example.com"]
        )
        db = _make_db(call_session, call_flow)

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
            patch(
                "app.services.voice_analysis_service.voice_analysis_service.analyze_call_transcript"
            ) as mock_analyze,
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_analyze.assert_called_once_with(
            db, call_session, user_id=call_session.user_id, raise_on_no_transcript=False
        )
        mock_send.assert_called_once()

    def test_lazy_compute_acquires_and_releases_same_advisory_lock(self):
        """Lazy-compute must go through `ensure_call_summary_locked`, which
        serializes against the automatic `generate_call_summary` ARQ job via
        the same pg advisory lock helpers (keyed on call_session_id) --
        covers the race fix for duplicate paid LLM analysis calls."""
        from app.services import post_call_email_service as pces
        from app.services import voice_analysis_service as vas_module

        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow(
            email_summary_enabled=True, email_summary_recipients=["a@example.com"]
        )
        db = _make_db(call_session, call_flow)

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
            patch.object(
                vas_module.voice_analysis_service, "analyze_call_transcript"
            ) as mock_analyze,
            patch.object(
                vas_module, "_try_acquire_call_summary_lock", return_value=True
            ) as mock_try_lock,
            patch.object(vas_module, "_release_call_summary_lock") as mock_release,
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_try_lock.assert_called_once_with(db, call_session.id)
        mock_analyze.assert_called_once()
        mock_release.assert_called_once_with(db, call_session.id)
        mock_send.assert_called_once()

    def test_lock_unavailable_skips_recompute_but_still_sends(self):
        """Another worker (the automatic summary job) is holding the lock --
        this job must not call analyze_call_transcript itself, and should
        still degrade gracefully and send using whatever data is available."""
        from app.services import post_call_email_service as pces
        from app.services import voice_analysis_service as vas_module

        call_session = _make_call_session(call_metadata=None, transcript_summary="fallback")
        call_flow = _make_call_flow(
            email_summary_enabled=True, email_summary_recipients=["a@example.com"]
        )
        db = _make_db(call_session, call_flow)

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
            patch.object(
                vas_module.voice_analysis_service, "analyze_call_transcript"
            ) as mock_analyze,
            patch.object(
                vas_module, "_try_acquire_call_summary_lock", return_value=False
            ),
            patch.object(vas_module, "_release_call_summary_lock") as mock_release,
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_analyze.assert_not_called()
        mock_release.assert_not_called()  # never acquired, so nothing to release
        mock_send.assert_called_once()

    def test_double_checked_lock_reuses_racing_jobs_result_instead_of_recomputing(self):
        """Lock acquired, but the concurrent `generate_call_summary` ARQ job
        already committed its analysis while this job was waiting -- the
        post-lock `db.refresh` re-check must see that and skip a second
        (paid) LLM call, reusing the racing job's result."""
        from app.services import post_call_email_service as pces
        from app.services import voice_analysis_service as vas_module

        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow(
            email_summary_enabled=True, email_summary_recipients=["a@example.com"]
        )
        db = _make_db(call_session, call_flow)

        def _refresh_sets_cache(obj):
            obj.call_metadata = {
                "llm_call_analysis": {"analysis": {"summary": "won by other worker"}}
            }

        db.refresh.side_effect = _refresh_sets_cache

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
            patch.object(
                vas_module.voice_analysis_service, "analyze_call_transcript"
            ) as mock_analyze,
            patch.object(
                vas_module, "_try_acquire_call_summary_lock", return_value=True
            ),
            patch.object(vas_module, "_release_call_summary_lock") as mock_release,
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_analyze.assert_not_called()
        mock_release.assert_called_once_with(db, call_session.id)
        mock_send.assert_called_once()

    def test_lazy_compute_failure_is_non_fatal_and_still_sends(self):
        """Analysis generation failing must not block the email send — the
        email is built from whatever fields exist (transcript_summary /
        empty analysis_data)."""
        from app.services import post_call_email_service as pces

        call_session = _make_call_session(call_metadata=None, transcript_summary="fallback")
        call_flow = _make_call_flow(
            email_summary_enabled=True, email_summary_recipients=["a@example.com"]
        )
        db = _make_db(call_session, call_flow)

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
            patch(
                "app.services.voice_analysis_service.voice_analysis_service.analyze_call_transcript",
                side_effect=RuntimeError("LLM outage"),
            ),
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_send.assert_called_once()

    def test_exception_anywhere_in_body_is_swallowed_and_logged(self):
        """A totally unexpected exception (e.g. DB error resolving the call
        session) must never raise into the ARQ worker."""
        from app.services import post_call_email_service as pces

        db = MagicMock()
        db.query.side_effect = RuntimeError("db connection lost")

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.logger, "warning") as mock_warning,
        ):
            # Must not raise.
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_warning.assert_called()
        db.close.assert_called_once()

    def test_send_generic_email_exception_is_swallowed(self):
        from app.services import post_call_email_service as pces

        call_session = _make_call_session(
            call_metadata={"llm_call_analysis": {"analysis": {"summary": "cached"}}}
        )
        call_flow = _make_call_flow(
            email_summary_enabled=True, email_summary_recipients=["a@example.com"]
        )
        db = _make_db(call_session, call_flow)

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(
                pces.email_service,
                "send_generic_email",
                side_effect=RuntimeError("SES outage"),
            ),
            patch.object(pces.logger, "warning") as mock_warning,
        ):
            # Must not raise.
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_warning.assert_called()
        db.close.assert_called_once()


# ── Post-Call Analysis (custom extracted variables) email integration ───────


class TestPostCallAnalysisEmailIntegration:
    """Covers the email hand-off to post_call_analysis_service: extraction is
    triggered (best-effort, non-blocking) when the flow has variables
    configured and results aren't cached yet, and _build_email_content
    renders an "Extracted Details" section from whatever's cached."""

    def test_extraction_triggered_when_variables_configured_and_not_cached(self):
        from app.services import post_call_email_service as pces

        call_session = _make_call_session(
            call_metadata={"llm_call_analysis": {"analysis": {"summary": "cached"}}}
        )
        call_flow = _make_call_flow(
            email_summary_enabled=True,
            email_summary_recipients=["a@example.com"],
            post_call_analysis_variables=[
                {"name": "service_type", "description": "Extract the service type."}
            ],
        )
        db = _make_db(call_session, call_flow)

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
            patch(
                "app.services.post_call_analysis_service.ensure_post_call_analysis_locked"
            ) as mock_ensure_analysis,
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_ensure_analysis.assert_called_once_with(db, call_session, call_flow)
        mock_send.assert_called_once()

    def test_extraction_not_triggered_when_no_variables_configured(self):
        from app.services import post_call_email_service as pces

        call_session = _make_call_session(
            call_metadata={"llm_call_analysis": {"analysis": {"summary": "cached"}}}
        )
        call_flow = _make_call_flow(
            email_summary_enabled=True,
            email_summary_recipients=["a@example.com"],
            post_call_analysis_variables=[],
        )
        db = _make_db(call_session, call_flow)

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
            patch(
                "app.services.post_call_analysis_service.ensure_post_call_analysis_locked"
            ) as mock_ensure_analysis,
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_ensure_analysis.assert_not_called()
        mock_send.assert_called_once()

    def test_extraction_not_triggered_when_already_cached(self):
        from app.services import post_call_email_service as pces

        call_session = _make_call_session(
            call_metadata={
                "llm_call_analysis": {"analysis": {"summary": "cached"}},
                "post_call_analysis": {
                    "variables": {"service_type": "residential plumbing"},
                    "model_used": "gpt-4o-mini",
                    "timestamp": "2026-08-01T00:00:00+00:00",
                },
            }
        )
        call_flow = _make_call_flow(
            email_summary_enabled=True,
            email_summary_recipients=["a@example.com"],
            post_call_analysis_variables=[
                {"name": "service_type", "description": "Extract the service type."}
            ],
        )
        db = _make_db(call_session, call_flow)

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
            patch(
                "app.services.post_call_analysis_service.ensure_post_call_analysis_locked"
            ) as mock_ensure_analysis,
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_ensure_analysis.assert_not_called()
        mock_send.assert_called_once()

    def test_extraction_failure_is_non_fatal_and_email_still_sends(self):
        """A slow/failed custom extraction must never block the base email
        send — it's wrapped in its own try/except."""
        from app.services import post_call_email_service as pces

        call_session = _make_call_session(
            call_metadata={"llm_call_analysis": {"analysis": {"summary": "cached"}}}
        )
        call_flow = _make_call_flow(
            email_summary_enabled=True,
            email_summary_recipients=["a@example.com"],
            post_call_analysis_variables=[
                {"name": "service_type", "description": "Extract the service type."}
            ],
        )
        db = _make_db(call_session, call_flow)

        with (
            patch("app.services.post_call_email_service.SessionLocal", return_value=db),
            patch.object(pces.email_service, "send_generic_email") as mock_send,
            patch(
                "app.services.post_call_analysis_service.ensure_post_call_analysis_locked",
                side_effect=RuntimeError("LLM outage"),
            ),
        ):
            pces._send_post_call_email_summary_impl(SESSION_ID)

        mock_send.assert_called_once()

    def test_build_email_content_includes_extracted_details_when_present(self):
        from app.services.post_call_email_service import _build_email_content

        call_session = _make_call_session(
            call_metadata={
                "llm_call_analysis": {"analysis": {"summary": "cached"}},
                "post_call_analysis": {
                    "variables": {
                        "service_type": "residential plumbing",
                        "urgency": "high",
                    },
                    "model_used": "gpt-4o-mini",
                    "timestamp": "2026-08-01T00:00:00+00:00",
                },
            }
        )
        call_flow = _make_call_flow(email_summary_enabled=True)

        _subject, html_body = _build_email_content(call_session, call_flow)

        assert "Extracted Details" in html_body
        assert "Service Type" in html_body
        assert "residential plumbing" in html_body
        assert "Urgency" in html_body
        assert "high" in html_body

    def test_build_email_content_omits_extracted_details_when_not_configured(self):
        from app.services.post_call_email_service import _build_email_content

        call_session = _make_call_session(
            call_metadata={"llm_call_analysis": {"analysis": {"summary": "cached"}}}
        )
        call_flow = _make_call_flow(email_summary_enabled=True)

        _subject, html_body = _build_email_content(call_session, call_flow)

        assert "Extracted Details" not in html_body

    def test_build_email_content_escapes_html_in_extracted_variable_values(self):
        """LLM-extracted variable values are caller-influenced free text and
        must never be interpolated into the email HTML unescaped — a value
        containing markup/script-like text must render escaped, not raw."""
        from app.services.post_call_email_service import _build_email_content

        call_session = _make_call_session(
            call_metadata={
                "llm_call_analysis": {"analysis": {"summary": "cached"}},
                "post_call_analysis": {
                    "variables": {
                        "notes": "<script>alert('xss')</script>",
                        "quoted_request": "The caller said \"<b>urgent</b>\"",
                    },
                    "model_used": "gpt-4o-mini",
                    "timestamp": "2026-08-01T00:00:00+00:00",
                },
            }
        )
        call_flow = _make_call_flow(email_summary_enabled=True)

        _subject, html_body = _build_email_content(call_session, call_flow)

        assert "<script>alert('xss')</script>" not in html_body
        assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html_body
        assert "<b>urgent</b>" not in html_body
        assert "&lt;b&gt;urgent&lt;/b&gt;" in html_body

    def test_build_email_content_escapes_variable_name_defense_in_depth(self):
        """Humanized variable names are also escaped, defense-in-depth, even
        though names are already regex-constrained at the schema level."""
        from app.services.post_call_email_service import _build_email_content

        call_session = _make_call_session(
            call_metadata={
                "llm_call_analysis": {"analysis": {"summary": "cached"}},
                "post_call_analysis": {
                    "variables": {"service_type": "plumbing"},
                    "model_used": "gpt-4o-mini",
                    "timestamp": "2026-08-01T00:00:00+00:00",
                },
            }
        )
        call_flow = _make_call_flow(email_summary_enabled=True)

        _subject, html_body = _build_email_content(call_session, call_flow)

        assert "Service Type" in html_body


# ── Caller display, Recommendations removal, HappyAssist sign-off ──────────


class TestCallerDisplaySignOffAndRecommendationsRemoval:
    def test_web_call_type_shows_demo_link_label(self):
        from app.services.post_call_email_service import _resolve_caller_display

        call_session = _make_call_session(call_type="web", from_number=None, to_number=None)
        assert _resolve_caller_display(call_session) == "Demo Link Call"

    def test_inbound_call_shows_from_number(self):
        from app.services.post_call_email_service import _resolve_caller_display

        call_session = _make_call_session(
            call_type="inbound", from_number="+15551234567", to_number="+15557654321"
        )
        assert _resolve_caller_display(call_session) == "+15551234567"

    def test_outbound_call_shows_to_number(self):
        from app.services.post_call_email_service import _resolve_caller_display

        call_session = _make_call_session(
            call_type="outbound", from_number="+15557654321", to_number="+15551234567"
        )
        assert _resolve_caller_display(call_session) == "+15551234567"

    def test_inbound_call_missing_number_falls_back_to_unknown(self):
        from app.services.post_call_email_service import _resolve_caller_display

        call_session = _make_call_session(call_type="inbound", from_number=None)
        assert _resolve_caller_display(call_session) == "Unknown"

    def test_build_email_content_shows_phone_number_for_phone_call(self):
        from app.services.post_call_email_service import _build_email_content

        call_session = _make_call_session(
            call_metadata={"llm_call_analysis": {"analysis": {"caller_name": "Unknown"}}},
            call_type="inbound",
            from_number="+15551234567",
        )
        call_flow = _make_call_flow(email_summary_enabled=True)

        subject, html_body = _build_email_content(call_session, call_flow)

        assert "+15551234567" in html_body
        assert "+15551234567" in subject
        assert "<strong>Caller:</strong> Unknown</p>" not in html_body

    def test_build_email_content_shows_demo_link_call_for_web_call(self):
        from app.services.post_call_email_service import _build_email_content

        call_session = _make_call_session(call_type="web")
        call_flow = _make_call_flow(email_summary_enabled=True)

        subject, html_body = _build_email_content(call_session, call_flow)

        assert "Demo Link Call" in html_body
        assert "Demo Link Call" in subject

    def test_recommendations_never_appear_in_email(self):
        """Regression: the Recommendations section must be removed entirely,
        even when the LLM analysis includes recommendations."""
        from app.services.post_call_email_service import _build_email_content

        call_session = _make_call_session(
            call_metadata={
                "llm_call_analysis": {
                    "analysis": {
                        "summary": "cached",
                        "recommendations": [
                            "Offer a callback.",
                            "Confirm caller's issue is resolved.",
                        ],
                    }
                }
            }
        )
        call_flow = _make_call_flow(email_summary_enabled=True)

        _subject, html_body = _build_email_content(call_session, call_flow)

        assert "Recommendations" not in html_body
        assert "Offer a callback." not in html_body

    def test_sign_off_is_happyassist_not_voice_agent_team(self):
        from app.services.post_call_email_service import _build_email_content

        call_session = _make_call_session()
        call_flow = _make_call_flow(email_summary_enabled=True)

        _subject, html_body = _build_email_content(call_session, call_flow)

        assert "HappyAssist Team" in html_body
        assert "The Voice Agent Team" not in html_body
