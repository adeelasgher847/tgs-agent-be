"""Unit test for the `finalize_call_recording` ARQ job wrapper registered in
app/workers/batch_call_worker.py::WorkerSettings.functions.

The wrapper itself has no logic beyond delegating to
call_recording_upload_service._finalize_call_recording_arq_task — this just
guards against a future refactor silently dropping the delegation or
mangling args. Mirrors
tests/workers/test_batch_call_worker_generate_call_summary.py.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from app.workers.batch_call_worker import finalize_call_recording


class TestFinalizeCallRecordingWorkerWrapper:
    async def test_delegates_to_finalize_call_recording_arq_task_with_same_args(self):
        ctx = {"some": "ctx"}
        call_session_id = "11111111-1111-1111-1111-111111111111"

        with patch(
            "app.services.call_recording_upload_service._finalize_call_recording_arq_task",
            new_callable=AsyncMock,
        ) as mock_task:
            await finalize_call_recording(ctx, call_session_id, 3)

        mock_task.assert_called_once_with(ctx, call_session_id, 3)

    async def test_default_attempt_is_1(self):
        ctx = {"some": "ctx"}
        call_session_id = "11111111-1111-1111-1111-111111111111"

        with patch(
            "app.services.call_recording_upload_service._finalize_call_recording_arq_task",
            new_callable=AsyncMock,
        ) as mock_task:
            await finalize_call_recording(ctx, call_session_id)

        mock_task.assert_called_once_with(ctx, call_session_id, 1)

    async def test_registered_in_worker_settings_functions(self):
        from app.workers.batch_call_worker import WorkerSettings

        assert finalize_call_recording in WorkerSettings.functions
