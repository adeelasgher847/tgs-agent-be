"""
BatchCallService — upload, CSV validation, job/record CRUD.

All DB writes are sync (Session) to stay consistent with the rest of the v1
service layer.  The v2 router calls this via run_in_executor when needed.
"""
from __future__ import annotations

import csv
import io
import re
import string
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from fastapi import HTTPException, status
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import logger
from app.models.agent import Agent
from app.models.batch_call_record import BatchCallRecord
from app.models.batch_job import BatchJob
from app.schemas.batch_call import (
    BatchCallRecordOut,
    BatchJobOut,
    BatchJobProgress,
    CsvRowError,
    PaginatedBatchCallRecords,
    PaginatedBatchJobs,
)
from app.services import batch_call_s3_service

MAX_CSV_ROWS = 10_000
MAX_CSV_BYTES = 20 * 1024 * 1024  # 20 MB
REQUIRED_COLUMN = "phone_number"

# Statuses that cannot be cancelled
_TERMINAL_STATUSES = frozenset({"completed", "cancelled", "failed"})

# Formatter variable pattern  {variable_name}
_VAR_PATTERN = re.compile(r"\{(\w+)\}")


class AllNumbersFlaggedError(Exception):
    """Raised when the agent's bound number is spam-flagged and no clean
    same-country replacement exists in the workspace's phone number pool."""


def _extract_prompt_vars(prompt: Optional[str]) -> List[str]:
    """Return all {variable} names referenced in the agent system prompt."""
    if not prompt:
        return []
    return _VAR_PATTERN.findall(prompt)


def _validate_csv(raw: bytes) -> Tuple[List[dict], List[CsvRowError]]:
    """
    Parse and validate CSV bytes.

    Returns (rows_as_dicts, row_errors).
    Raises HTTPException 422 for structural errors (encoding, missing column).
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV must be UTF-8 encoded",
        )

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV file is empty or has no header row",
        )

    headers = [h.strip() for h in reader.fieldnames]
    if REQUIRED_COLUMN not in headers:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"CSV must contain a '{REQUIRED_COLUMN}' column. Found columns: {headers}",
        )

    rows: List[dict] = []
    errors: List[CsvRowError] = []

    for i, row in enumerate(reader, start=2):  # row 1 = header
        stripped = {k.strip(): (v or "").strip() for k, v in row.items() if k}
        phone = stripped.get(REQUIRED_COLUMN, "")
        if not phone:
            errors.append(CsvRowError(row=i, error="phone_number is empty"))
            continue
        rows.append(stripped)

        if len(rows) > MAX_CSV_ROWS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"CSV exceeds maximum row limit of {MAX_CSV_ROWS}",
            )

    return rows, errors


class BatchCallService:
    def __init__(self, db: Session) -> None:
        self._db = db

    # ── Upload ────────────────────────────────────────────────────────────────

    def create_batch_job(
        self,
        workspace_id: uuid.UUID,
        agent_id: uuid.UUID,
        csv_bytes: bytes,
        scheduled_at: Optional[datetime] = None,
        voicemail_action: str = "skip",
        voicemail_message: Optional[str] = None,
    ) -> BatchJobOut:
        """
        Validate CSV, upload to GCS, persist BatchJob + BatchCallRecords.

        Raises HTTPException 422 on any validation failure.
        """
        if len(csv_bytes) > MAX_CSV_BYTES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"CSV file exceeds maximum size of 20 MB",
            )

        # ── Validate agent belongs to workspace ──────────────────────────────
        agent = (
            self._db.execute(
                select(Agent).where(
                    Agent.id == agent_id,
                    Agent.tenant_id == workspace_id,
                    Agent.is_deleted.is_(False),
                )
            )
            .scalars()
            .first()
        )
        if agent is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found in this workspace",
            )

        # ── Validate CSV ─────────────────────────────────────────────────────
        rows, row_errors = _validate_csv(csv_bytes)

        if not rows:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="CSV contains no valid data rows",
            )

        if row_errors:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": "CSV validation failed",
                    "errors": [e.model_dump() for e in row_errors],
                },
            )

        # ── Prompt variable validation ────────────────────────────────────────
        prompt_vars = _extract_prompt_vars(agent.system_prompt)
        if prompt_vars:
            csv_columns = set(rows[0].keys())
            missing = [v for v in prompt_vars if v not in csv_columns]
            if missing:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"Agent prompt references variables not present as CSV columns: "
                        f"{missing}. CSV columns: {sorted(csv_columns)}"
                    ),
                )

        # ── Persist job + records ─────────────────────────────────────────────
        batch_id = uuid.uuid4()
        gcs_key = batch_call_s3_service.build_batch_csv_gcs_key(workspace_id, batch_id)

        # Upload CSV to S3 (may raise — no DB side-effects yet)
        try:
            batch_call_s3_service.upload_batch_csv(gcs_key, csv_bytes, workspace_id, batch_id)
        except Exception as exc:
            logger.error("S3 CSV upload failed for batch %s: %s", batch_id, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to upload CSV to storage; please retry",
            )

        job = BatchJob(
            id=batch_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
            status="pending",
            total_count=len(rows),
            waiting_count=len(rows),
            active_count=0,
            completed_count=0,
            failed_count=0,
            s3_path=gcs_key,
            scheduled_at=scheduled_at,
            voicemail_action=voicemail_action,
            voicemail_message=voicemail_message,
        )
        self._db.add(job)
        self._db.flush()

        records = [
            BatchCallRecord(
                id=uuid.uuid4(),
                batch_job_id=batch_id,
                phone_number=row[REQUIRED_COLUMN],
                variables={k: v for k, v in row.items() if k != REQUIRED_COLUMN} or None,
                status="waiting",
                attempts=0,
            )
            for row in rows
        ]
        self._db.bulk_save_objects(records)
        self._db.commit()

        self._db.refresh(job)
        logger.info(
            "BatchJob %s created for workspace %s: %d records, scheduled_at=%s",
            batch_id, workspace_id, len(rows), scheduled_at,
        )
        return BatchJobOut.model_validate(job)

    # ── Read ──────────────────────────────────────────────────────────────────

    def list_batch_jobs(
        self,
        workspace_id: uuid.UUID,
        page: int = 1,
        page_size: int = 20,
    ) -> PaginatedBatchJobs:
        offset = (page - 1) * page_size

        total = (
            self._db.execute(
                select(func.count(BatchJob.id)).where(BatchJob.workspace_id == workspace_id)
            )
            .scalar_one()
        )

        jobs = (
            self._db.execute(
                select(BatchJob)
                .where(BatchJob.workspace_id == workspace_id)
                .order_by(BatchJob.created_at.desc())
                .offset(offset)
                .limit(page_size)
            )
            .scalars()
            .all()
        )

        return PaginatedBatchJobs(
            items=[BatchJobOut.model_validate(j) for j in jobs],
            total=total,
            page=page,
            page_size=page_size,
        )

    def get_batch_job(self, workspace_id: uuid.UUID, batch_id: uuid.UUID) -> BatchJob:
        job = (
            self._db.execute(
                select(BatchJob).where(
                    BatchJob.id == batch_id,
                    BatchJob.workspace_id == workspace_id,
                )
            )
            .scalars()
            .first()
        )
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"BatchJob {batch_id} not found",
            )
        return job

    def get_batch_job_progress(
        self, workspace_id: uuid.UUID, batch_id: uuid.UUID
    ) -> BatchJobProgress:
        job = self.get_batch_job(workspace_id, batch_id)
        total = job.total_count or 0
        completed = job.completed_count or 0
        pct = round((completed / total * 100), 1) if total > 0 else 0.0
        return BatchJobProgress(
            batch_id=job.id,
            status=job.status,
            waiting=job.waiting_count,
            active=job.active_count,
            completed=completed,
            failed=job.failed_count,
            total=total,
            percent_complete=pct,
            voicemail_skipped=job.voicemail_skipped_count or 0,
            voicemail_message_left=job.voicemail_message_left_count or 0,
        )

    def list_batch_call_records(
        self,
        workspace_id: uuid.UUID,
        batch_id: uuid.UUID,
        page: int = 1,
        page_size: int = 50,
    ) -> PaginatedBatchCallRecords:
        # Ensure job belongs to workspace
        self.get_batch_job(workspace_id, batch_id)

        offset = (page - 1) * page_size
        total = (
            self._db.execute(
                select(func.count(BatchCallRecord.id)).where(
                    BatchCallRecord.batch_job_id == batch_id
                )
            )
            .scalar_one()
        )
        records = (
            self._db.execute(
                select(BatchCallRecord)
                .where(BatchCallRecord.batch_job_id == batch_id)
                .order_by(BatchCallRecord.created_at.asc())
                .offset(offset)
                .limit(page_size)
            )
            .scalars()
            .all()
        )
        return PaginatedBatchCallRecords(
            items=[BatchCallRecordOut.model_validate(r) for r in records],
            total=total,
            page=page,
            page_size=page_size,
        )

    # ── Cancel ────────────────────────────────────────────────────────────────

    def cancel_batch_job(self, workspace_id: uuid.UUID, batch_id: uuid.UUID) -> BatchJobOut:
        """
        Cancel a batch job.  Already-connected calls complete naturally;
        waiting records are flipped to cancelled; the worker stops picking
        new records because it checks job.status before pickup.
        """
        job = self.get_batch_job(workspace_id, batch_id)

        if job.status in _TERMINAL_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"BatchJob is already in terminal state '{job.status}'",
            )

        # Cancel waiting records only (active ones may already be dialling)
        waiting_records = (
            self._db.execute(
                select(BatchCallRecord).where(
                    BatchCallRecord.batch_job_id == batch_id,
                    BatchCallRecord.status == "waiting",
                )
            )
            .scalars()
            .all()
        )
        cancelled_count = 0
        for rec in waiting_records:
            rec.status = "cancelled"
            cancelled_count += 1

        job.status = "cancelled"
        job.waiting_count = 0
        job.completed_at = datetime.now(timezone.utc)
        self._db.commit()

        self._db.refresh(job)
        logger.info(
            "BatchJob %s cancelled: %d waiting records cancelled", batch_id, cancelled_count
        )
        return BatchJobOut.model_validate(job)

    # ── Outbound number reputation / auto-rotation ────────────────────────────

    async def rotate_number_if_flagged(
        self,
        workspace_id: uuid.UUID,
        agent_id: uuid.UUID,
        batch_job_id: uuid.UUID,
    ) -> Optional[Tuple[str, str]]:
        """
        Check the reputation of the agent's bound phone number for this batch and,
        if it's spam-flagged, rotate the batch to a clean number from the
        workspace's pool (same country code, preferring the same area code).

        Returns (old_number, new_number) if a rotation happened, or None if the
        bound number is clean (or unbound — nothing to rotate here; call
        initiation will surface the "agent not ready" error as usual).

        Raises AllNumbersFlaggedError if the bound number is flagged and no
        clean same-country replacement exists in the pool.
        """
        from sqlalchemy import or_

        from app.models.phone_number import PhoneNumber
        from app.models.phone_number_reputation import PhoneNumberReputation
        from app.services.reputation_service import check_number_reputation
        from app.utils.phone_geo import get_area_code, get_country_code

        phone_number_obj = (
            self._db.query(PhoneNumber)
            .filter(
                PhoneNumber.assistant_id == agent_id,
                PhoneNumber.tenant_id == workspace_id,
                PhoneNumber.status == "active",
            )
            .first()
        )
        if phone_number_obj is None:
            return None

        reputation = (
            self._db.query(PhoneNumberReputation)
            .filter(PhoneNumberReputation.phone_number_id == phone_number_obj.id)
            .first()
        )
        if reputation is None:
            await check_number_reputation(self._db, phone_number_obj)
            reputation = (
                self._db.query(PhoneNumberReputation)
                .filter(PhoneNumberReputation.phone_number_id == phone_number_obj.id)
                .first()
            )

        if reputation is None or not reputation.spam_flagged:
            return None

        country_code = get_country_code(phone_number_obj.phone_number)
        area_code = get_area_code(phone_number_obj.phone_number)

        candidates = (
            self._db.query(PhoneNumber)
            .outerjoin(
                PhoneNumberReputation,
                PhoneNumberReputation.phone_number_id == PhoneNumber.id,
            )
            .filter(
                PhoneNumber.tenant_id == workspace_id,
                PhoneNumber.status == "active",
                PhoneNumber.id != phone_number_obj.id,
                PhoneNumber.assistant_id.is_(None),
                or_(
                    PhoneNumberReputation.id.is_(None),
                    PhoneNumberReputation.spam_flagged.is_(False),
                ),
            )
            .all()
        )
        same_country = [c for c in candidates if get_country_code(c.phone_number) == country_code]

        replacement: Optional[PhoneNumber] = None
        if area_code:
            same_area = [c for c in same_country if get_area_code(c.phone_number) == area_code]
            if same_area:
                replacement = same_area[0]
        if replacement is None and same_country:
            replacement = same_country[0]

        if replacement is None:
            logger.warning(
                "BatchJob %s: bound number %s is spam-flagged and no clean replacement "
                "exists in workspace %s",
                batch_job_id,
                phone_number_obj.phone_number,
                workspace_id,
            )
            job = self._db.get(BatchJob, batch_job_id)
            if job is not None:
                self._db.execute(
                    update(BatchCallRecord)
                    .where(
                        BatchCallRecord.batch_job_id == batch_job_id,
                        BatchCallRecord.status == "waiting",
                    )
                    .values(status="cancelled", last_error="all_numbers_flagged")
                )
                job.status = "failed"
                job.waiting_count = 0
                job.completed_at = datetime.now(timezone.utc)
                self._db.commit()
            raise AllNumbersFlaggedError()

        job = self._db.get(BatchJob, batch_job_id)
        job.actual_from_number = replacement.phone_number
        self._db.commit()

        logger.info(
            "BatchJob %s: rotated outbound number %s -> %s (spam_flagged)",
            batch_job_id,
            phone_number_obj.phone_number,
            replacement.phone_number,
        )
        return phone_number_obj.phone_number, replacement.phone_number
