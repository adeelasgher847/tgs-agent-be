"""
API tests for v2 batch-calls endpoints.

These are unit/functional tests: DB is SQLite in-memory, GCS and initiate_call
are mocked at the service boundary.  No Postgres required.

Coverage:
  - POST /batch-calls — happy path, invalid CSV (missing column), variable mismatch
  - GET /batch-calls — list
  - GET /batch-calls/{id} — detail
  - GET /batch-calls/{id}/progress — live counts
  - GET /batch-calls/{id}/calls — paginated records
  - DELETE /batch-calls/{id} — cancel
  - Auth: missing / invalid API key returns 401
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import uuid
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_workspace, require_tenant
from app.core.exception_handlers import register_exception_handlers
from app.core.workspace import Workspace


# ── Auth helpers ──────────────────────────────────────────────────────────────

WORKSPACE_ID = uuid.uuid4()
RAW_KEY = f"tgs-{uuid.uuid4().hex}"
KEY_HASH = hashlib.sha256(RAW_KEY.encode()).hexdigest()
AUTH_HEADERS = {"x-api-key": RAW_KEY, "x-workspace-id": str(WORKSPACE_ID)}
AGENT_ID = uuid.uuid4()


def _mock_workspace() -> Workspace:
    ws = MagicMock(spec=Workspace)
    ws.id = WORKSPACE_ID
    ws.status = "active"
    ws.is_active = True
    return ws


def _mock_principal():
    """Minimal ApiKeyPrincipal stand-in — satisfies require_tenant override."""
    from app.core.request_auth import ApiKeyPrincipal

    return ApiKeyPrincipal(current_tenant_id=WORKSPACE_ID, api_key_id=uuid.uuid4())


# ── CSV fixtures ──────────────────────────────────────────────────────────────

def _make_csv(rows: list[dict], *, extra_cols: list[str] | None = None) -> bytes:
    cols = ["phone_number"] + (extra_cols or [])
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    return "\n".join(lines).encode("utf-8")


VALID_CSV = _make_csv([
    {"phone_number": "+15551111001"},
    {"phone_number": "+15551111002"},
    {"phone_number": "+15551111003"},
])

CSV_NO_PHONE_COL = b"name,email\nAlice,alice@example.com"

CSV_WITH_VARS = _make_csv(
    [{"phone_number": "+15551111001", "first_name": "Alice"}],
    extra_cols=["first_name"],
)


# ── App factory ───────────────────────────────────────────────────────────────

def _build_app(svc_mock) -> TestClient:
    """
    Build a minimal FastAPI app that mounts only the batch-calls router,
    with BatchCallService replaced by svc_mock.

    Overrides require_tenant and get_workspace (standard deps) plus both
    service factories so tests run without a real DB or auth middleware.
    """
    from app.api.v2.routers import batch_calls as bc_module

    ws = _mock_workspace()
    principal = _mock_principal()

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(bc_module.router)

    # Auth + workspace: short-circuit middleware-resolved deps
    mini.dependency_overrides[require_tenant] = lambda: principal
    mini.dependency_overrides[get_workspace] = lambda: ws

    # Service: return svc_mock for both read and write factories
    def _svc_override():
        yield svc_mock

    mini.dependency_overrides[bc_module._batch_service] = _svc_override
    mini.dependency_overrides[bc_module._batch_service_write] = _svc_override

    return TestClient(mini, raise_server_exceptions=False)


# ── POST /batch-calls ─────────────────────────────────────────────────────────

class TestCreateBatchJob:
    def _svc(self, job_out):
        svc = MagicMock()
        svc.create_batch_job.return_value = job_out
        svc.rotate_number_if_flagged = AsyncMock(return_value=None)
        return svc

    def _job_out(self, total: int = 3, voicemail_action: str = "skip", voicemail_message=None) -> dict:
        from app.schemas.batch_call import BatchJobOut
        from datetime import datetime, timezone

        return BatchJobOut(
            id=uuid.uuid4(),
            workspace_id=WORKSPACE_ID,
            agent_id=AGENT_ID,
            status="pending",
            total_count=total,
            waiting_count=total,
            active_count=0,
            completed_count=0,
            failed_count=0,
            voicemail_action=voicemail_action,
            voicemail_message=voicemail_message,
            s3_path=f"batch-files/{WORKSPACE_ID}/{uuid.uuid4()}.csv",
            scheduled_at=None,
            started_at=None,
            completed_at=None,
            created_at=datetime.now(timezone.utc),
        )

    @patch("app.api.v2.routers.batch_calls._enqueue_batch_job", new_callable=AsyncMock)
    def test_upload_valid_csv_returns_201(self, mock_enqueue):
        job = self._job_out(3)
        client = _build_app(self._svc(job))

        resp = client.post(
            "/batch-calls",
            data={"agent_id": str(AGENT_ID)},
            files={"file": ("test.csv", io.BytesIO(VALID_CSV), "text/csv")},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 201
        body = resp.json()
        assert body["status"] == "pending"
        assert body["total_count"] == 3
        assert body["workspace_id"] == str(WORKSPACE_ID)

    @patch("app.api.v2.routers.batch_calls._enqueue_batch_job", new_callable=AsyncMock)
    def test_upload_with_voicemail_leave_message_passes_fields_to_service(self, mock_enqueue):
        svc = self._svc(self._job_out(3, voicemail_action="leave_message", voicemail_message="Call us back"))
        client = _build_app(svc)

        resp = client.post(
            "/batch-calls",
            data={
                "agent_id": str(AGENT_ID),
                "voicemail_action": "leave_message",
                "voicemail_message": "Call us back",
            },
            files={"file": ("test.csv", io.BytesIO(VALID_CSV), "text/csv")},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 201
        assert resp.json()["voicemail_action"] == "leave_message"
        assert resp.json()["voicemail_message"] == "Call us back"
        _, kwargs = svc.create_batch_job.call_args
        assert kwargs["voicemail_action"] == "leave_message"
        assert kwargs["voicemail_message"] == "Call us back"

    @patch("app.api.v2.routers.batch_calls._enqueue_batch_job", new_callable=AsyncMock)
    def test_upload_defaults_voicemail_action_to_skip(self, mock_enqueue):
        svc = self._svc(self._job_out(3))
        client = _build_app(svc)

        resp = client.post(
            "/batch-calls",
            data={"agent_id": str(AGENT_ID)},
            files={"file": ("test.csv", io.BytesIO(VALID_CSV), "text/csv")},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 201
        _, kwargs = svc.create_batch_job.call_args
        assert kwargs["voicemail_action"] == "skip"

    @patch("app.api.v2.routers.batch_calls._enqueue_batch_job", new_callable=AsyncMock)
    def test_upload_invalid_voicemail_action_returns_422(self, mock_enqueue):
        svc = self._svc(self._job_out(3))
        client = _build_app(svc)

        resp = client.post(
            "/batch-calls",
            data={"agent_id": str(AGENT_ID), "voicemail_action": "bogus"},
            files={"file": ("test.csv", io.BytesIO(VALID_CSV), "text/csv")},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 422
        svc.create_batch_job.assert_not_called()

    @patch("app.api.v2.routers.batch_calls._enqueue_batch_job", new_callable=AsyncMock)
    def test_upload_invalid_csv_missing_phone_col_returns_422(self, mock_enqueue):
        from fastapi import HTTPException

        svc = MagicMock()
        svc.create_batch_job.side_effect = HTTPException(
            status_code=422,
            detail="CSV must contain a 'phone_number' column. Found columns: ['name', 'email']",
        )
        client = _build_app(svc)

        resp = client.post(
            "/batch-calls",
            data={"agent_id": str(AGENT_ID)},
            files={"file": ("bad.csv", io.BytesIO(CSV_NO_PHONE_COL), "text/csv")},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 422

    @patch("app.api.v2.routers.batch_calls._enqueue_batch_job", new_callable=AsyncMock)
    def test_prompt_variable_mismatch_returns_422(self, mock_enqueue):
        from fastapi import HTTPException

        svc = MagicMock()
        svc.create_batch_job.side_effect = HTTPException(
            status_code=422,
            detail="Agent prompt references variables not present as CSV columns: ['last_name']",
        )
        client = _build_app(svc)

        resp = client.post(
            "/batch-calls",
            data={"agent_id": str(AGENT_ID)},
            files={"file": ("vars.csv", io.BytesIO(CSV_WITH_VARS), "text/csv")},
            headers=AUTH_HEADERS,
        )

        assert resp.status_code == 422

    # ── File size validation (Fix 3) ──────────────────────────────────────────

    @patch("app.api.v2.routers.batch_calls._enqueue_batch_job", new_callable=AsyncMock)
    def test_pre_read_size_check_rejects_oversized_via_size_metadata(self, mock_enqueue):
        """
        When UploadFile.size metadata exceeds 20 MB the router must return 422
        without reading the file body or calling the service.
        """
        import asyncio
        from unittest.mock import AsyncMock as _AsyncMock, MagicMock, PropertyMock
        from fastapi import UploadFile
        from app.api.v2.routers.batch_calls import create_batch_job
        from app.core.workspace import Workspace

        mock_file = MagicMock(spec=UploadFile)
        mock_file.content_type = "text/csv"
        mock_file.size = 21 * 1024 * 1024  # 21 MB — over the limit
        mock_file.read = _AsyncMock(return_value=b"phone_number\n+15550001111\n")

        mock_workspace = MagicMock(spec=Workspace)
        mock_workspace.id = WORKSPACE_ID

        mock_svc = MagicMock()

        from fastapi import HTTPException as _HTTPException
        with pytest.raises(_HTTPException) as exc_info:
            asyncio.run(
                create_batch_job(
                    request=MagicMock(),
                    file=mock_file,
                    agent_id=AGENT_ID,
                    scheduled_at=None,
                    voicemail_action="skip",
                    voicemail_message=None,
                    workspace=mock_workspace,
                    db=MagicMock(),
                    svc=mock_svc,
                )
            )

        assert exc_info.value.status_code == 422
        assert "20 MB" in exc_info.value.detail
        mock_file.read.assert_not_called()
        mock_svc.create_batch_job.assert_not_called()

    @patch("app.api.v2.routers.batch_calls._enqueue_batch_job", new_callable=AsyncMock)
    def test_post_read_size_check_rejects_oversized_payload_without_metadata(self, mock_enqueue):
        """
        When UploadFile.size is None (no Content-Length) but the actual payload
        exceeds 20 MB, the router must still return 422 after reading.
        """
        import asyncio
        from unittest.mock import AsyncMock as _AsyncMock, MagicMock
        from fastapi import UploadFile, HTTPException as _HTTPException
        from app.api.v2.routers.batch_calls import create_batch_job
        from app.core.workspace import Workspace
        from app.services.batch_call_service import MAX_CSV_BYTES

        oversized_body = b"phone_number\n" + b"+15550001111\n" + b"x" * (MAX_CSV_BYTES + 1)

        mock_file = MagicMock(spec=UploadFile)
        mock_file.content_type = "text/csv"
        mock_file.size = None  # No size metadata available
        mock_file.read = _AsyncMock(return_value=oversized_body)

        mock_workspace = MagicMock(spec=Workspace)
        mock_workspace.id = WORKSPACE_ID

        mock_svc = MagicMock()

        with pytest.raises(_HTTPException) as exc_info:
            asyncio.run(
                create_batch_job(
                    request=MagicMock(),
                    file=mock_file,
                    agent_id=AGENT_ID,
                    scheduled_at=None,
                    voicemail_action="skip",
                    voicemail_message=None,
                    workspace=mock_workspace,
                    db=MagicMock(),
                    svc=mock_svc,
                )
            )

        assert exc_info.value.status_code == 422
        assert "20 MB" in exc_info.value.detail
        mock_file.read.assert_called_once()
        mock_svc.create_batch_job.assert_not_called()

    @patch("app.api.v2.routers.batch_calls._enqueue_batch_job", new_callable=AsyncMock)
    def test_valid_upload_unaffected_by_size_checks(self, mock_enqueue):
        """
        A valid CSV under the size limit must not be rejected by either size check.
        Regression guard for Fix 3.
        """
        import asyncio
        from unittest.mock import AsyncMock as _AsyncMock, MagicMock
        from fastapi import UploadFile
        from app.api.v2.routers.batch_calls import create_batch_job
        from app.core.workspace import Workspace
        from app.schemas.batch_call import BatchJobOut
        from datetime import datetime, timezone

        job = BatchJobOut(
            id=uuid.uuid4(),
            workspace_id=WORKSPACE_ID,
            agent_id=AGENT_ID,
            status="pending",
            total_count=2,
            waiting_count=2,
            active_count=0,
            completed_count=0,
            failed_count=0,
            voicemail_action="skip",
            voicemail_message=None,
            s3_path=f"batch-files/{WORKSPACE_ID}/x.csv",
            scheduled_at=None,
            started_at=None,
            completed_at=None,
            created_at=datetime.now(timezone.utc),
        )

        mock_file = MagicMock(spec=UploadFile)
        mock_file.content_type = "text/csv"
        mock_file.size = len(VALID_CSV)
        mock_file.read = _AsyncMock(return_value=VALID_CSV)

        mock_workspace = MagicMock(spec=Workspace)
        mock_workspace.id = WORKSPACE_ID

        mock_svc = MagicMock()
        mock_svc.create_batch_job.return_value = job
        mock_svc.rotate_number_if_flagged = AsyncMock(return_value=None)

        result = asyncio.run(
            create_batch_job(
                request=MagicMock(),
                file=mock_file,
                agent_id=AGENT_ID,
                scheduled_at=None,
                voicemail_action="skip",
                voicemail_message=None,
                workspace=mock_workspace,
                db=MagicMock(),
                svc=mock_svc,
            )
        )

        mock_svc.create_batch_job.assert_called_once()
        assert result.total_count == 2

    def test_missing_api_key_returns_401(self):
        from app.api.v2.routers import batch_calls as bc_module

        mini = FastAPI()
        register_exception_handlers(mini)
        mini.include_router(bc_module.router)

        # Override require_tenant to raise 401 (simulates unauthenticated request)
        def _unauth():
            raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Invalid or missing API key"})

        mini.dependency_overrides[require_tenant] = _unauth

        with TestClient(mini, raise_server_exceptions=False) as c:
            resp = c.post(
                "/batch-calls",
                data={"agent_id": str(AGENT_ID)},
                files={"file": ("t.csv", io.BytesIO(VALID_CSV), "text/csv")},
            )

        assert resp.status_code == 401


# ── GET /batch-calls ──────────────────────────────────────────────────────────

class TestListBatchJobs:
    def test_returns_paginated_list(self):
        from app.schemas.batch_call import PaginatedBatchJobs
        from datetime import datetime, timezone

        from app.schemas.batch_call import BatchJobOut

        job = BatchJobOut(
            id=uuid.uuid4(),
            workspace_id=WORKSPACE_ID,
            agent_id=AGENT_ID,
            status="completed",
            total_count=5,
            waiting_count=0,
            active_count=0,
            completed_count=5,
            failed_count=0,
            s3_path=None,
            scheduled_at=None,
            started_at=None,
            completed_at=None,
            created_at=datetime.now(timezone.utc),
        )
        svc = MagicMock()
        svc.list_batch_jobs.return_value = PaginatedBatchJobs(
            items=[job], total=1, page=1, page_size=20
        )

        client = _build_app(svc)
        resp = client.get("/batch-calls", headers=AUTH_HEADERS)

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert len(body["items"]) == 1


# ── GET /batch-calls/{id} ─────────────────────────────────────────────────────

class TestGetBatchJob:
    def test_returns_detail(self):
        from datetime import datetime, timezone

        from app.schemas.batch_call import BatchJobOut

        batch_id = uuid.uuid4()
        job_model = MagicMock()
        job_model.id = batch_id
        job_model.workspace_id = WORKSPACE_ID
        job_model.agent_id = AGENT_ID
        job_model.status = "processing"
        job_model.total_count = 10
        job_model.waiting_count = 7
        job_model.active_count = 3
        job_model.completed_count = 0
        job_model.failed_count = 0
        job_model.voicemail_action = "skip"
        job_model.voicemail_message = None
        job_model.s3_path = "batch-files/x/y.csv"
        job_model.scheduled_at = None
        job_model.started_at = None
        job_model.completed_at = None
        job_model.created_at = datetime.now(timezone.utc)

        svc = MagicMock()
        svc.get_batch_job.return_value = job_model
        client = _build_app(svc)

        resp = client.get(f"/batch-calls/{batch_id}", headers=AUTH_HEADERS)

        assert resp.status_code == 200
        assert resp.json()["status"] == "processing"

    def test_not_found_returns_404(self):
        from fastapi import HTTPException

        svc = MagicMock()
        svc.get_batch_job.side_effect = HTTPException(status_code=404, detail="not found")
        client = _build_app(svc)

        resp = client.get(f"/batch-calls/{uuid.uuid4()}", headers=AUTH_HEADERS)

        assert resp.status_code == 404


# ── GET /batch-calls/{id}/progress ───────────────────────────────────────────

class TestGetBatchJobProgress:
    def test_returns_live_counts(self):
        from app.schemas.batch_call import BatchJobProgress

        batch_id = uuid.uuid4()
        progress = BatchJobProgress(
            batch_id=batch_id,
            status="processing",
            waiting=5,
            active=2,
            completed=3,
            failed=0,
            total=10,
            percent_complete=30.0,
            voicemail_skipped=2,
            voicemail_message_left=1,
        )
        svc = MagicMock()
        svc.get_batch_job_progress.return_value = progress
        client = _build_app(svc)

        resp = client.get(f"/batch-calls/{batch_id}/progress", headers=AUTH_HEADERS)

        assert resp.status_code == 200
        body = resp.json()
        assert body["waiting"] == 5
        assert body["active"] == 2
        assert body["completed"] == 3
        assert body["total"] == 10
        assert body["percent_complete"] == 30.0
        assert body["voicemail_skipped"] == 2
        assert body["voicemail_message_left"] == 1


# ── GET /batch-calls/{id}/calls ───────────────────────────────────────────────

class TestListBatchCallRecords:
    def test_returns_paginated_records(self):
        from datetime import datetime, timezone

        from app.schemas.batch_call import BatchCallRecordOut, PaginatedBatchCallRecords

        batch_id = uuid.uuid4()
        rec = BatchCallRecordOut(
            id=uuid.uuid4(),
            batch_job_id=batch_id,
            phone_number="+15551112222",
            variables={"first_name": "Alice"},
            status="completed",
            call_id=None,
            attempts=1,
            last_error=None,
            next_attempt_at=None,
            created_at=datetime.now(timezone.utc),
            updated_at=None,
        )
        svc = MagicMock()
        svc.list_batch_call_records.return_value = PaginatedBatchCallRecords(
            items=[rec], total=1, page=1, page_size=50
        )
        client = _build_app(svc)

        resp = client.get(f"/batch-calls/{batch_id}/calls", headers=AUTH_HEADERS)

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["items"][0]["phone_number"] == "+15551112222"


# ── DELETE /batch-calls/{id} ──────────────────────────────────────────────────

class TestCancelBatchJob:
    def test_cancel_in_progress_job(self):
        from datetime import datetime, timezone

        from app.schemas.batch_call import BatchJobOut

        batch_id = uuid.uuid4()
        cancelled_job = BatchJobOut(
            id=batch_id,
            workspace_id=WORKSPACE_ID,
            agent_id=AGENT_ID,
            status="cancelled",
            total_count=10,
            waiting_count=0,
            active_count=2,
            completed_count=3,
            failed_count=0,
            s3_path=None,
            scheduled_at=None,
            started_at=None,
            completed_at=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        svc = MagicMock()
        svc.cancel_batch_job.return_value = cancelled_job
        client = _build_app(svc)

        resp = client.delete(f"/batch-calls/{batch_id}", headers=AUTH_HEADERS)

        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_cancel_already_completed_returns_409(self):
        from fastapi import HTTPException

        svc = MagicMock()
        svc.cancel_batch_job.side_effect = HTTPException(
            status_code=409, detail="BatchJob is already in terminal state 'completed'"
        )
        client = _build_app(svc)

        resp = client.delete(f"/batch-calls/{uuid.uuid4()}", headers=AUTH_HEADERS)

        assert resp.status_code == 409
