"""Unit test for the `generate_call_summary` ARQ job wrapper registered in
app/workers/batch_call_worker.py::WorkerSettings.functions.

The wrapper itself has no logic beyond delegating to
voice_analysis_service._generate_call_summary_arq_task — this just guards
against a future refactor silently dropping the delegation or mangling args.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from app.workers.batch_call_worker import generate_call_summary


class TestGenerateCallSummaryWorkerWrapper:
    async def test_delegates_to_generate_call_summary_arq_task_with_same_args(self):
        ctx = {"some": "ctx"}
        call_session_id = "11111111-1111-1111-1111-111111111111"

        with patch(
            "app.services.voice_analysis_service._generate_call_summary_arq_task",
            new_callable=AsyncMock,
        ) as mock_task:
            await generate_call_summary(ctx, call_session_id)

        mock_task.assert_called_once_with(ctx, call_session_id)

    async def test_registered_in_worker_settings_functions(self):
        from app.workers.batch_call_worker import WorkerSettings

        assert generate_call_summary in WorkerSettings.functions
