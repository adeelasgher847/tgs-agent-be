"""Unit tests for app/services/post_call_analysis_service.py.

Covers:
  - schedule_run_post_call_analysis: ARQ enqueue happy path, fails open
    without a pool, fails open on enqueue error. Mirrors
    tests/services/test_post_call_email_service.py::TestSchedulePostCallEmailSummary.
  - ensure_post_call_analysis_locked: no-op when variables empty, no-op when
    already cached, lock acquire/release around extraction, double-checked-lock
    reuse, lock-unavailable degrades gracefully.
  - _run_extraction: no transcript → no-op; successful JSON response persists
    variables/model_used/timestamp; markdown-fenced response still parses;
    malformed JSON falls back to regex recovery; fully unrecoverable response
    persists nothing; HIPAA redaction applied only when call_flow.hipaa_compliance
    is true; model fallback chain order; mirrors onto CallLog when present.
  - _run_post_call_analysis_impl: session/flow/variables missing → no-op;
    any exception is swallowed, logged, never raised.
  - The lock key derivation is distinct from voice_analysis_service's
    (different key pair for the same UUID).

No real LLM/provider/Redis/Postgres calls — all mocked at the boundary.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

WORKSPACE_ID = uuid.uuid4()
FLOW_ID = uuid.uuid4()
SESSION_ID = uuid.uuid4()
AGENT_ID = uuid.uuid4()

VARIABLES = [
    {"name": "service_type", "description": "What service was requested."},
    {"name": "urgency", "description": "How urgent the request is."},
]


def _make_call_session(*, call_metadata=None, agent_id=None):
    session = MagicMock()
    session.id = SESSION_ID
    session.tenant_id = WORKSPACE_ID
    session.call_flow_id = FLOW_ID
    session.call_metadata = call_metadata
    session.agent_id = agent_id
    return session


def _make_call_flow(*, variables=None, model=None, hipaa=False):
    flow = MagicMock()
    flow.id = FLOW_ID
    flow.post_call_analysis_variables = VARIABLES if variables is None else variables
    flow.post_call_analysis_model = model
    flow.hipaa_compliance = hipaa
    return flow


def _make_db(call_session=None, call_flow=None, call_log=None):
    from app.models.call_flow import CallFlow
    from app.models.call_log import CallLog
    from app.models.call_session import CallSession

    db = MagicMock()

    def _query(model):
        q = MagicMock()
        if model is CallSession:
            q.filter.return_value.first.return_value = call_session
        elif model is CallFlow:
            q.filter.return_value.first.return_value = call_flow
        elif model is CallLog:
            q.filter.return_value.first.return_value = call_log
        else:
            q.filter.return_value.first.return_value = None
        return q

    db.query.side_effect = _query
    return db


# ── schedule_run_post_call_analysis (ARQ enqueue) ───────────────────────────


class TestScheduleRunPostCallAnalysis:
    def test_fails_open_without_arq_pool(self):
        from app.services import post_call_analysis_service as pcas

        with patch.object(pcas, "get_arq_pool", return_value=None):
            pcas.schedule_run_post_call_analysis(SESSION_ID)

    def test_enqueues_run_post_call_analysis_job_with_stringified_id(self):
        from app.services import post_call_analysis_service as pcas

        mock_pool = MagicMock()
        mock_pool.enqueue_job = AsyncMock()

        with (
            patch.object(pcas, "get_arq_pool", return_value=mock_pool),
            patch(
                "app.services.post_call_analysis_service.asyncio.get_running_loop",
                side_effect=RuntimeError("no running loop"),
            ),
        ):
            pcas.schedule_run_post_call_analysis(SESSION_ID)

        mock_pool.enqueue_job.assert_called_once_with(
            "run_post_call_analysis", str(SESSION_ID)
        )

    def test_enqueue_failure_is_caught_and_does_not_propagate(self):
        from app.services import post_call_analysis_service as pcas

        mock_pool = MagicMock()
        mock_pool.enqueue_job = AsyncMock(side_effect=RuntimeError("redis down"))

        with (
            patch.object(pcas, "get_arq_pool", return_value=mock_pool),
            patch(
                "app.services.post_call_analysis_service.asyncio.get_running_loop",
                side_effect=RuntimeError("no running loop"),
            ),
        ):
            pcas.schedule_run_post_call_analysis(SESSION_ID)

        mock_pool.enqueue_job.assert_called_once()

    def test_creates_task_when_loop_already_running(self):
        from app.services import post_call_analysis_service as pcas

        mock_pool = MagicMock()
        mock_pool.enqueue_job = AsyncMock()

        with (
            patch.object(pcas, "get_arq_pool", return_value=mock_pool),
            patch(
                "app.services.post_call_analysis_service.asyncio.get_running_loop",
                return_value=MagicMock(),
            ),
            patch(
                "app.services.post_call_analysis_service.asyncio.create_task"
            ) as mock_create_task,
        ):
            pcas.schedule_run_post_call_analysis(SESSION_ID)

        mock_create_task.assert_called_once()
        mock_create_task.call_args[0][0].close()


# ── Advisory-lock key derivation must differ from voice_analysis_service's ──


class TestLockKeyDerivationIsDistinct:
    def test_key_pair_differs_from_voice_analysis_services(self):
        from app.services import post_call_analysis_service as pcas
        from app.services import voice_analysis_service as vas

        session_id = uuid.uuid4()
        assert pcas._pg_advisory_key_pair(session_id) != vas._pg_advisory_key_pair(session_id)

    def test_key_pair_is_stable_for_same_uuid(self):
        from app.services import post_call_analysis_service as pcas

        session_id = uuid.uuid4()
        assert pcas._pg_advisory_key_pair(session_id) == pcas._pg_advisory_key_pair(session_id)


# ── ensure_post_call_analysis_locked ─────────────────────────────────────────


class TestEnsurePostCallAnalysisLocked:
    def test_noop_when_variables_empty(self):
        from app.services import post_call_analysis_service as pcas

        db = MagicMock()
        call_session = _make_call_session()
        call_flow = _make_call_flow(variables=[])

        with patch.object(pcas, "_try_acquire_post_call_analysis_lock") as mock_lock:
            pcas.ensure_post_call_analysis_locked(db, call_session, call_flow)

        mock_lock.assert_not_called()

    def test_noop_when_already_cached(self):
        from app.services import post_call_analysis_service as pcas

        db = MagicMock()
        call_session = _make_call_session(
            call_metadata={"post_call_analysis": {"variables": {"service_type": "x"}}}
        )
        call_flow = _make_call_flow()

        with patch.object(pcas, "_try_acquire_post_call_analysis_lock") as mock_lock:
            pcas.ensure_post_call_analysis_locked(db, call_session, call_flow)

        mock_lock.assert_not_called()

    def test_acquires_runs_extraction_and_releases_lock(self):
        from app.services import post_call_analysis_service as pcas

        db = MagicMock()
        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow()

        with (
            patch.object(pcas, "_try_acquire_post_call_analysis_lock", return_value=True) as mock_try,
            patch.object(pcas, "_release_post_call_analysis_lock") as mock_release,
            patch.object(pcas, "_run_extraction") as mock_run,
        ):
            pcas.ensure_post_call_analysis_locked(db, call_session, call_flow)

        mock_try.assert_called_once_with(db, call_session.id)
        mock_run.assert_called_once_with(db, call_session, call_flow)
        mock_release.assert_called_once_with(db, call_session.id)

    def test_lock_unavailable_degrades_gracefully_without_raising(self):
        from app.services import post_call_analysis_service as pcas

        db = MagicMock()
        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow()

        with (
            patch.object(pcas, "_try_acquire_post_call_analysis_lock", return_value=False),
            patch.object(pcas, "_release_post_call_analysis_lock") as mock_release,
            patch.object(pcas, "_run_extraction") as mock_run,
        ):
            pcas.ensure_post_call_analysis_locked(db, call_session, call_flow)

        mock_run.assert_not_called()
        mock_release.assert_not_called()

    def test_double_checked_lock_reuses_racing_workers_result(self):
        from app.services import post_call_analysis_service as pcas

        db = MagicMock()
        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow()

        def _refresh_sets_cache(obj):
            obj.call_metadata = {
                "post_call_analysis": {"variables": {"service_type": "won by other worker"}}
            }

        db.refresh.side_effect = _refresh_sets_cache

        with (
            patch.object(pcas, "_try_acquire_post_call_analysis_lock", return_value=True),
            patch.object(pcas, "_release_post_call_analysis_lock") as mock_release,
            patch.object(pcas, "_run_extraction") as mock_run,
        ):
            pcas.ensure_post_call_analysis_locked(db, call_session, call_flow)

        mock_run.assert_not_called()
        mock_release.assert_called_once_with(db, call_session.id)


# ── _run_extraction ───────────────────────────────────────────────────────


class TestRunExtraction:
    def test_no_transcript_is_a_noop(self):
        from app.services import post_call_analysis_service as pcas

        db = _make_db()
        call_session = _make_call_session()
        call_flow = _make_call_flow()

        with patch.object(
            pcas.transcript_service, "get_messages_by_session", return_value=[]
        ):
            pcas._run_extraction(db, call_session, call_flow)

        db.commit.assert_not_called()

    def _transcript_msgs(self):
        msg = MagicMock()
        msg.role = "client"
        msg.message = "I need residential plumbing help urgently."
        return [msg]

    def _mock_model(self, name="gpt-4o-mini"):
        model = MagicMock()
        model.model_name = name
        model.provider.name = "openai"
        model.api_key = None
        return model

    def test_successful_json_response_persists_variables(self):
        from app.services import post_call_analysis_service as pcas

        db = _make_db()
        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow()

        model = self._mock_model()
        content = '{"service_type": "residential plumbing", "urgency": "high"}'

        with (
            patch.object(
                pcas.transcript_service,
                "get_messages_by_session",
                return_value=self._transcript_msgs(),
            ),
            patch.object(pcas.agent_service, "get_agent_by_id", side_effect=Exception("no agent")),
            patch.object(pcas._model_service, "get_model_by_name", return_value=model),
            patch.object(pcas, "decrypt_api_key", return_value=None),
            patch.object(
                pcas, "generate_analysis_text", return_value={"content": content}
            ) as mock_gen,
        ):
            pcas._run_extraction(db, call_session, call_flow)

        assert call_session.call_metadata["post_call_analysis"]["variables"] == {
            "service_type": "residential plumbing",
            "urgency": "high",
        }
        assert call_session.call_metadata["post_call_analysis"]["model_used"] == "gpt-4o-mini"
        assert "timestamp" in call_session.call_metadata["post_call_analysis"]
        db.commit.assert_called_once()
        mock_gen.assert_called_once()

    def test_markdown_fenced_response_still_parses(self):
        from app.services import post_call_analysis_service as pcas

        db = _make_db()
        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow()

        model = self._mock_model()
        content = '```json\n{"service_type": "commercial plumbing", "urgency": "low"}\n```'

        with (
            patch.object(
                pcas.transcript_service,
                "get_messages_by_session",
                return_value=self._transcript_msgs(),
            ),
            patch.object(pcas.agent_service, "get_agent_by_id", side_effect=Exception("no agent")),
            patch.object(pcas._model_service, "get_model_by_name", return_value=model),
            patch.object(pcas, "decrypt_api_key", return_value=None),
            patch.object(pcas, "generate_analysis_text", return_value={"content": content}),
        ):
            pcas._run_extraction(db, call_session, call_flow)

        assert call_session.call_metadata["post_call_analysis"]["variables"] == {
            "service_type": "commercial plumbing",
            "urgency": "low",
        }

    def test_malformed_json_falls_back_to_regex_recovery(self):
        from app.services import post_call_analysis_service as pcas

        db = _make_db()
        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow()

        model = self._mock_model()
        # Not valid JSON (missing closing brace), but "service_type" is
        # regex-recoverable; "urgency" is not quoted the same way and is
        # absent entirely — must be backfilled with "not mentioned" rather
        # than silently dropped, so the persisted result always has one
        # entry per configured variable.
        content = '{"service_type": "residential plumbing", oops broken'

        with (
            patch.object(
                pcas.transcript_service,
                "get_messages_by_session",
                return_value=self._transcript_msgs(),
            ),
            patch.object(pcas.agent_service, "get_agent_by_id", side_effect=Exception("no agent")),
            patch.object(pcas._model_service, "get_model_by_name", return_value=model),
            patch.object(pcas, "decrypt_api_key", return_value=None),
            patch.object(pcas, "generate_analysis_text", return_value={"content": content}),
            patch.object(pcas.logger, "warning") as mock_warning,
        ):
            pcas._run_extraction(db, call_session, call_flow)

        variables = call_session.call_metadata["post_call_analysis"]["variables"]
        assert variables == {
            "service_type": "residential plumbing",
            "urgency": "not mentioned",
        }
        db.commit.assert_called_once()
        # A warning must be logged naming the backfilled variable(s).
        assert any(
            "urgency" in str(call.args)
            for call in mock_warning.call_args_list
        )

    def test_fully_unrecoverable_response_persists_nothing(self):
        from app.services import post_call_analysis_service as pcas

        db = _make_db()
        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow()

        model = self._mock_model()
        content = "I'm sorry, I cannot help with that."

        with (
            patch.object(
                pcas.transcript_service,
                "get_messages_by_session",
                return_value=self._transcript_msgs(),
            ),
            patch.object(pcas.agent_service, "get_agent_by_id", side_effect=Exception("no agent")),
            patch.object(pcas._model_service, "get_model_by_name", return_value=model),
            patch.object(pcas, "decrypt_api_key", return_value=None),
            patch.object(pcas, "generate_analysis_text", return_value={"content": content}),
        ):
            pcas._run_extraction(db, call_session, call_flow)

        assert call_session.call_metadata is None
        db.commit.assert_not_called()

    def test_hipaa_redaction_applied_when_flow_hipaa_enabled(self):
        from app.services import post_call_analysis_service as pcas

        db = _make_db()
        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow(hipaa=True)

        model = self._mock_model()
        content = '{"service_type": "call SSN 123-45-6789", "urgency": "high"}'

        with (
            patch.object(
                pcas.transcript_service,
                "get_messages_by_session",
                return_value=self._transcript_msgs(),
            ),
            patch.object(pcas.agent_service, "get_agent_by_id", side_effect=Exception("no agent")),
            patch.object(pcas._model_service, "get_model_by_name", return_value=model),
            patch.object(pcas, "decrypt_api_key", return_value=None),
            patch.object(pcas, "generate_analysis_text", return_value={"content": content}),
            patch.object(pcas, "redact_phi_if_hipaa", return_value="[REDACTED]") as mock_redact,
        ):
            pcas._run_extraction(db, call_session, call_flow)

        mock_redact.assert_called()
        for call in mock_redact.call_args_list:
            assert call.kwargs["hipaa_enabled"] is True
        assert call_session.call_metadata["post_call_analysis"]["variables"]["service_type"] == "[REDACTED]"

    def test_hipaa_redaction_not_applied_when_flow_hipaa_disabled(self):
        from app.services import post_call_analysis_service as pcas

        db = _make_db()
        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow(hipaa=False)

        model = self._mock_model()
        content = '{"service_type": "residential plumbing", "urgency": "high"}'

        with (
            patch.object(
                pcas.transcript_service,
                "get_messages_by_session",
                return_value=self._transcript_msgs(),
            ),
            patch.object(pcas.agent_service, "get_agent_by_id", side_effect=Exception("no agent")),
            patch.object(pcas._model_service, "get_model_by_name", return_value=model),
            patch.object(pcas, "decrypt_api_key", return_value=None),
            patch.object(pcas, "generate_analysis_text", return_value={"content": content}),
            patch.object(pcas, "redact_phi_if_hipaa", wraps=pcas.redact_phi_if_hipaa) as mock_redact,
        ):
            pcas._run_extraction(db, call_session, call_flow)

        for call in mock_redact.call_args_list:
            assert call.kwargs["hipaa_enabled"] is False
        assert (
            call_session.call_metadata["post_call_analysis"]["variables"]["service_type"]
            == "residential plumbing"
        )

    def test_model_fallback_chain_tries_flow_model_first(self):
        from app.services import post_call_analysis_service as pcas

        db = _make_db()
        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow(model="claude-3-5-sonnet")

        model = self._mock_model(name="claude-3-5-sonnet")
        content = '{"service_type": "x", "urgency": "y"}'

        with (
            patch.object(
                pcas.transcript_service,
                "get_messages_by_session",
                return_value=self._transcript_msgs(),
            ),
            patch.object(pcas.agent_service, "get_agent_by_id", side_effect=Exception("no agent")),
            patch.object(
                pcas._model_service, "get_model_by_name", return_value=model
            ) as mock_get_model,
            patch.object(pcas, "decrypt_api_key", return_value=None),
            patch.object(pcas, "generate_analysis_text", return_value={"content": content}),
        ):
            pcas._run_extraction(db, call_session, call_flow)

        # First candidate attempted must be the flow's configured model.
        first_call_args = mock_get_model.call_args_list[0]
        assert first_call_args.args[1] == "claude-3-5-sonnet"
        assert call_session.call_metadata["post_call_analysis"]["model_used"] == "claude-3-5-sonnet"

    def test_model_fallback_chain_falls_through_to_agents_model_then_hardcoded(self):
        from app.services import post_call_analysis_service as pcas

        db = _make_db()
        call_session = _make_call_session(call_metadata=None, agent_id=AGENT_ID)
        call_flow = _make_call_flow(model=None)

        agent = MagicMock()
        agent.model.model_name = "gemini-1.5-pro"

        gemini_model = self._mock_model(name="gemini-1.5-pro")
        content = '{"service_type": "x", "urgency": "y"}'

        with (
            patch.object(
                pcas.transcript_service,
                "get_messages_by_session",
                return_value=self._transcript_msgs(),
            ),
            patch.object(pcas.agent_service, "get_agent_by_id", return_value=agent),
            patch.object(
                pcas._model_service, "get_model_by_name", return_value=gemini_model
            ) as mock_get_model,
            patch.object(pcas, "decrypt_api_key", return_value=None),
            patch.object(pcas, "generate_analysis_text", return_value={"content": content}),
        ):
            pcas._run_extraction(db, call_session, call_flow)

        # call_flow.post_call_analysis_model is None, so the agent's model is
        # tried first among the candidates.
        first_call_args = mock_get_model.call_args_list[0]
        assert first_call_args.args[1] == "gemini-1.5-pro"

    def test_all_candidate_models_fail_persists_nothing(self):
        from app.services import post_call_analysis_service as pcas

        db = _make_db()
        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow()

        with (
            patch.object(
                pcas.transcript_service,
                "get_messages_by_session",
                return_value=self._transcript_msgs(),
            ),
            patch.object(pcas.agent_service, "get_agent_by_id", side_effect=Exception("no agent")),
            patch.object(pcas._model_service, "get_model_by_name", return_value=None),
        ):
            pcas._run_extraction(db, call_session, call_flow)

        assert call_session.call_metadata is None
        db.commit.assert_not_called()

    def test_mirrors_onto_call_log_when_matching_row_exists(self):
        from app.services import post_call_analysis_service as pcas

        call_log = MagicMock()
        call_log.call_metadata = None

        db = _make_db()
        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow()

        # Rebuild db with the CallLog row wired up.
        db = _make_db(call_log=call_log)

        model = self._mock_model()
        content = '{"service_type": "x", "urgency": "y"}'

        with (
            patch.object(
                pcas.transcript_service,
                "get_messages_by_session",
                return_value=self._transcript_msgs(),
            ),
            patch.object(pcas.agent_service, "get_agent_by_id", side_effect=Exception("no agent")),
            patch.object(pcas._model_service, "get_model_by_name", return_value=model),
            patch.object(pcas, "decrypt_api_key", return_value=None),
            patch.object(pcas, "generate_analysis_text", return_value={"content": content}),
        ):
            pcas._run_extraction(db, call_session, call_flow)

        assert call_log.call_metadata["post_call_analysis"]["variables"] == {
            "service_type": "x",
            "urgency": "y",
        }


# ── Partial-parse backfill (missing-key completeness invariant) ─────────────


class TestPartialParseBackfill:
    def _transcript_msgs(self):
        msg = MagicMock()
        msg.role = "client"
        msg.message = "I need residential plumbing help urgently."
        return [msg]

    def _mock_model(self, name="gpt-4o-mini"):
        model = MagicMock()
        model.model_name = name
        model.provider.name = "openai"
        model.api_key = None
        return model

    def test_partial_parse_backfills_missing_keys_and_logs_warning(self):
        """Model only complies for 1 of 3 configured variables — the other
        two must be backfilled with 'not mentioned' (not silently dropped),
        so the persisted result always has one entry per configured
        variable, and a warning naming the missing keys must be logged."""
        from app.services import post_call_analysis_service as pcas

        db = _make_db()
        call_session = _make_call_session(call_metadata=None)
        variables = [
            {"name": "service_type", "description": "What service was requested."},
            {"name": "urgency", "description": "How urgent the request is."},
            {"name": "budget", "description": "Customer's stated budget."},
        ]
        call_flow = _make_call_flow(variables=variables)

        model = self._mock_model()
        # Model only returned "service_type" — "urgency" and "budget" absent.
        content = '{"service_type": "residential plumbing"}'

        with (
            patch.object(
                pcas.transcript_service,
                "get_messages_by_session",
                return_value=self._transcript_msgs(),
            ),
            patch.object(pcas.agent_service, "get_agent_by_id", side_effect=Exception("no agent")),
            patch.object(pcas._model_service, "get_model_by_name", return_value=model),
            patch.object(pcas, "decrypt_api_key", return_value=None),
            patch.object(pcas, "generate_analysis_text", return_value={"content": content}),
            patch.object(pcas.logger, "warning") as mock_warning,
        ):
            pcas._run_extraction(db, call_session, call_flow)

        variables_persisted = call_session.call_metadata["post_call_analysis"]["variables"]
        assert variables_persisted == {
            "service_type": "residential plumbing",
            "urgency": "not mentioned",
            "budget": "not mentioned",
        }
        db.commit.assert_called_once()

        warning_calls = [str(call.args) for call in mock_warning.call_args_list]
        assert any(
            str(call_session.id) in text and "urgency" in text and "budget" in text
            for text in warning_calls
        )

    def test_total_parse_failure_still_persists_nothing(self):
        """Zero keys parsed at all (not even via regex fallback) for ANY
        variable — must still leave call_metadata untouched so a future
        attempt can retry cleanly, not persist an all-'not mentioned'
        result that looks superficially complete."""
        from app.services import post_call_analysis_service as pcas

        db = _make_db()
        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow()

        model = self._mock_model()
        content = "I'm sorry, I cannot help with that."

        with (
            patch.object(
                pcas.transcript_service,
                "get_messages_by_session",
                return_value=self._transcript_msgs(),
            ),
            patch.object(pcas.agent_service, "get_agent_by_id", side_effect=Exception("no agent")),
            patch.object(pcas._model_service, "get_model_by_name", return_value=model),
            patch.object(pcas, "decrypt_api_key", return_value=None),
            patch.object(pcas, "generate_analysis_text", return_value={"content": content}),
        ):
            pcas._run_extraction(db, call_session, call_flow)

        assert call_session.call_metadata is None
        db.commit.assert_not_called()


# ── _run_post_call_analysis_impl (ARQ task body) ─────────────────────────────


class TestRunPostCallAnalysisImpl:
    def test_session_not_found_is_a_noop(self):
        from app.services import post_call_analysis_service as pcas

        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None

        with (
            patch("app.services.post_call_analysis_service.SessionLocal", return_value=db),
            patch.object(pcas, "ensure_post_call_analysis_locked") as mock_ensure,
        ):
            pcas._run_post_call_analysis_impl(SESSION_ID)

        mock_ensure.assert_not_called()
        db.close.assert_called_once()

    def test_missing_call_flow_is_a_noop(self):
        from app.services import post_call_analysis_service as pcas
        from app.models.call_flow import CallFlow
        from app.models.call_session import CallSession

        call_session = _make_call_session(call_metadata=None)
        call_session.call_flow_id = None
        db = _make_db(call_session=call_session, call_flow=None)

        with (
            patch("app.services.post_call_analysis_service.SessionLocal", return_value=db),
            patch.object(pcas, "ensure_post_call_analysis_locked") as mock_ensure,
        ):
            pcas._run_post_call_analysis_impl(SESSION_ID)

        mock_ensure.assert_not_called()

    def test_empty_variables_is_a_noop(self):
        from app.services import post_call_analysis_service as pcas

        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow(variables=[])
        db = _make_db(call_session=call_session, call_flow=call_flow)

        with (
            patch("app.services.post_call_analysis_service.SessionLocal", return_value=db),
            patch.object(pcas, "ensure_post_call_analysis_locked") as mock_ensure,
        ):
            pcas._run_post_call_analysis_impl(SESSION_ID)

        mock_ensure.assert_not_called()

    def test_delegates_to_ensure_post_call_analysis_locked(self):
        from app.services import post_call_analysis_service as pcas

        call_session = _make_call_session(call_metadata=None)
        call_flow = _make_call_flow()
        db = _make_db(call_session=call_session, call_flow=call_flow)

        with (
            patch("app.services.post_call_analysis_service.SessionLocal", return_value=db),
            patch.object(pcas, "ensure_post_call_analysis_locked") as mock_ensure,
        ):
            pcas._run_post_call_analysis_impl(SESSION_ID)

        mock_ensure.assert_called_once_with(db, call_session, call_flow)
        db.close.assert_called_once()

    def test_exception_anywhere_is_swallowed_and_logged(self):
        from app.services import post_call_analysis_service as pcas

        db = MagicMock()
        db.query.side_effect = RuntimeError("db connection lost")

        with (
            patch("app.services.post_call_analysis_service.SessionLocal", return_value=db),
            patch.object(pcas.logger, "warning") as mock_warning,
        ):
            # Must not raise.
            pcas._run_post_call_analysis_impl(SESSION_ID)

        mock_warning.assert_called()
        db.close.assert_called_once()
