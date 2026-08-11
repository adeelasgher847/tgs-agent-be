"""Unit test for the `send_post_call_email_summary` ARQ job wrapper
registered in app/workers/batch_call_worker.py::WorkerSettings.functions.

The wrapper itself has no logic beyond parsing the UUID string and
delegating to post_call_email_service._send_post_call_email_summary_impl —
this just guards against a future refactor silently dropping the
delegation or mangling args. Mirrors
tests/workers/test_batch_call_worker_generate_call_summary.py.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

from app.workers.batch_call_worker import send_post_call_email_summary


class TestSendPostCallEmailSummaryWorkerWrapper:
    async def test_delegates_to_send_post_call_email_summary_impl_with_parsed_uuid(self):
        ctx = {"some": "ctx"}
        session_id = uuid.uuid4()

        with patch(
            "app.services.post_call_email_service._send_post_call_email_summary_impl",
            new_callable=MagicMock,
        ) as mock_impl:
            await send_post_call_email_summary(ctx, str(session_id))

        mock_impl.assert_called_once_with(session_id)

    async def test_registered_in_worker_settings_functions(self):
        from app.workers.batch_call_worker import WorkerSettings

        assert send_post_call_email_summary in WorkerSettings.functions
