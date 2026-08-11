"""Unit test for the `run_post_call_analysis` ARQ job wrapper registered in
app/workers/batch_call_worker.py::WorkerSettings.functions.

The wrapper itself has no logic beyond parsing the UUID string and
delegating to post_call_analysis_service._run_post_call_analysis_impl —
this just guards against a future refactor silently dropping the
delegation or mangling args. Mirrors
tests/workers/test_batch_call_worker_send_post_call_email_summary.py.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from app.workers.batch_call_worker import run_post_call_analysis


class TestRunPostCallAnalysisWorkerWrapper:
    async def test_delegates_to_run_post_call_analysis_impl_with_parsed_uuid(self):
        ctx = {"some": "ctx"}
        session_id = uuid.uuid4()

        with patch(
            "app.services.post_call_analysis_service._run_post_call_analysis_impl",
            new_callable=MagicMock,
        ) as mock_impl:
            await run_post_call_analysis(ctx, str(session_id))

        mock_impl.assert_called_once_with(session_id)

    async def test_registered_in_worker_settings_functions(self):
        from app.workers.batch_call_worker import WorkerSettings

        assert run_post_call_analysis in WorkerSettings.functions
