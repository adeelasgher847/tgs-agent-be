"""
BatchCallWorkerService — SKIP LOCKED pickup, per-record dispatch, retry logic.

Called exclusively from the ARQ worker (app/workers/batch_call_worker.py).
Uses a *sync* SQLAlchemy Session; the ARQ job function runs sync DB work inside
asyncio by acquiring a connection from the sync pool.

Retry policy
  - no_answer | busy  → up to 3 attempts, 30-minute gap
  - invalid_number    → no retry; mark failed immediately
  - system error      → no retry; mark failed; do NOT bill
  - connected / voicemail → bill via BillingService, mark completed
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import text, update
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import logger
from app.models.batch_call_record import BatchCallRecord
from app.models.batch_job import BatchJob

MAX_ATTEMPTS = 3
RETRY_GAP_MINUTES = 30

# Twilio/call statuses that are billable (connected or answered by voicemail)
_BILLABLE_ENDED_REASONS = frozenset({"completed", "voicemail", "voicemail-detected"})

# Statuses that trigger a retry (not the caller's fault, call never connected)
_RETRY_STATUSES = frozenset({"no-answer", "busy", "no_answer"})

# Status that must never be retried
_NO_RETRY_STATUS = "invalid-number"


class BatchCallWorkerService:
    def __init__(self, db: Session) -> None:
        self._db = db

    # ── Record pickup (SKIP LOCKED) ───────────────────────────────────────────

    def pick_waiting_records(
        self,
        batch_job_id: uuid.UUID,
        limit: int,
    ) -> list[BatchCallRecord]:
        """
        Atomically pick up to `limit` waiting records with SKIP LOCKED.

        Marks them active before returning so concurrent workers never double-pick.
        Returns [] if the job is cancelled or no records are available.
        """
        # Abort early if job is cancelled
        job = self._db.get(BatchJob, batch_job_id)
        if job is None or job.status in ("cancelled", "completed", "failed"):
            return []

        now = datetime.now(timezone.utc)

        # Raw SQL for SKIP LOCKED — SQLAlchemy ORM does not expose it portably.
        rows = self._db.execute(
            text(
                """
                SELECT id FROM batchcallrecord
                WHERE batch_job_id = :job_id
                  AND status = 'waiting'
                  AND (next_attempt_at IS NULL OR next_attempt_at <= :now)
                ORDER BY created_at
                LIMIT :lim
                FOR UPDATE SKIP LOCKED
                """
            ),
            {"job_id": str(batch_job_id), "now": now, "lim": limit},
        ).fetchall()

        if not rows:
            return []

        record_ids = [r[0] for r in rows]

        # Flip to active in the same transaction
        self._db.execute(
            update(BatchCallRecord)
            .where(BatchCallRecord.id.in_(record_ids))
            .values(status="active", updated_at=now)
        )
        self._db.execute(
            update(BatchJob)
            .where(BatchJob.id == batch_job_id)
            .values(
                waiting_count=BatchJob.waiting_count - len(record_ids),
                active_count=BatchJob.active_count + len(record_ids),
                status="processing",
            )
        )
        # Set started_at only when not already set (first pickup)
        self._db.execute(
            update(BatchJob)
            .where(BatchJob.id == batch_job_id, BatchJob.started_at.is_(None))
            .values(started_at=now)
        )
        self._db.commit()

        return (
            self._db.query(BatchCallRecord)
            .filter(BatchCallRecord.id.in_(record_ids))
            .all()
        )

    # ── Dispatch ──────────────────────────────────────────────────────────────

    async def dispatch_record(
        self,
        record: BatchCallRecord,
        workspace_id: uuid.UUID,
        agent_id: uuid.UUID,
        agent_system_prompt: Optional[str],
    ) -> None:
        """
        Dispatch a single batch call record via the existing `initiate_call` path.

        Builds a synthetic Request with the N8N webhook secret so the auth branch
        in `initiate_call` resolves to the webhook path (no User required).
        """
        from app.schemas.twilio import CallInitiateRequest
        from app.services import voice_call_service as _vcs

        initiate_call = _vcs.initiate_call

        # Build prompt with variable substitution (validated at upload time)
        variables: dict = record.variables or {}
        prompt_override: Optional[str] = None
        if agent_system_prompt:
            if variables:
                try:
                    prompt_override = agent_system_prompt.format(**variables)
                except KeyError as exc:
                    logger.warning(
                        "Batch record %s: prompt variable substitution failed: %s — using raw prompt",
                        record.id,
                        exc,
                    )
                    prompt_override = agent_system_prompt
            else:
                prompt_override = agent_system_prompt

        call_request = CallInitiateRequest(
            agentId=str(agent_id),
            toNumber=record.phone_number,
            tenant_id=str(workspace_id),
            batch_call_record_id=str(record.id),
            batch_prompt_override=prompt_override,
        )

        # Build a minimal fake Starlette Request so initiate_call can read the
        # N8N webhook secret header and resolve is_webhook=True.
        secret = settings.N8N_WEBHOOK_SECRET
        if not secret:
            raise RuntimeError(
                "N8N_WEBHOOK_SECRET must be set before batch dispatch can run. "
                "Set it in your .env or Secret Manager."
            )
        fake_request = _build_fake_request(secret)

        try:
            result = await initiate_call(
                call_request=call_request,
                http_request=fake_request,
                user=None,
                db=self._db,
            )
        except Exception as exc:
            logger.error("Batch dispatch error for record %s: %s", record.id, exc)
            await self._mark_failed(record, str(exc), is_system_error=True)
            return

        # Interpret the response
        from fastapi.responses import JSONResponse

        if isinstance(result, JSONResponse):
            body = result.body
            import json as _json

            try:
                payload = _json.loads(body)
            except Exception:
                payload = {}

            error_code = payload.get("error", {}).get("code", "")
            if error_code == _NO_RETRY_STATUS or "invalid" in error_code:
                await self._mark_failed(record, error_code, is_system_error=False, no_retry=True)
            elif error_code in ("busy", "no-answer", "no_answer"):
                await self._schedule_retry(record, error_code)
            else:
                await self._mark_failed(record, error_code or str(payload), is_system_error=True)
            return

        # Success: SuccessResponse — extract call_id
        call_id: Optional[uuid.UUID] = None
        try:
            data = result.data
            if hasattr(data, "callId"):
                call_id = uuid.UUID(str(data.callId))
        except Exception:
            pass

        await self._mark_active_with_call(record, call_id)

    # ── Post-call state transitions ───────────────────────────────────────────

    async def handle_call_completion(
        self,
        record_id: uuid.UUID,
        ended_reason: str,
    ) -> None:
        """
        Called from the voice webhook handler once the call ends.

        Marks the record completed or schedules a retry, and updates job counters.
        Billing is handled inside this method for connected/voicemail calls.
        """
        record = self._db.get(BatchCallRecord, record_id)
        if record is None:
            return

        if ended_reason in _BILLABLE_ENDED_REASONS:
            await self._mark_completed(record, bill=True)
        elif ended_reason in _RETRY_STATUSES and record.attempts < MAX_ATTEMPTS:
            await self._schedule_retry(record, ended_reason)
        else:
            no_retry = _NO_RETRY_STATUS in ended_reason
            await self._mark_failed(
                record, ended_reason, is_system_error=False, no_retry=no_retry
            )

    # ── State helpers ─────────────────────────────────────────────────────────

    async def _mark_active_with_call(
        self,
        record: BatchCallRecord,
        call_id: Optional[uuid.UUID],
    ) -> None:
        """Record is active; Twilio call was initiated successfully."""
        now = datetime.now(timezone.utc)
        record.call_id = call_id
        record.attempts = (record.attempts or 0) + 1
        record.updated_at = now
        self._db.commit()

    async def _mark_completed(self, record: BatchCallRecord, bill: bool = False) -> None:
        now = datetime.now(timezone.utc)
        record.status = "completed"
        record.updated_at = now
        self._db.commit()

        if bill:
            try:
                await _bill_connected_call(record, self._db)
            except Exception as exc:
                logger.warning("Billing failed for batch record %s: %s", record.id, exc)

        _decrement_job_counter(self._db, record.batch_job_id, "active_count")
        _increment_job_counter(self._db, record.batch_job_id, "completed_count")
        self._db.commit()
        await _maybe_complete_job(self._db, record.batch_job_id)

    async def _mark_failed(
        self,
        record: BatchCallRecord,
        error: str,
        *,
        is_system_error: bool,
        no_retry: bool = False,
    ) -> None:
        now = datetime.now(timezone.utc)
        record.status = "failed"
        record.last_error = error
        record.updated_at = now
        self._db.commit()

        _decrement_job_counter(self._db, record.batch_job_id, "active_count")
        _increment_job_counter(self._db, record.batch_job_id, "failed_count")
        self._db.commit()
        await _maybe_complete_job(self._db, record.batch_job_id)

    async def _schedule_retry(self, record: BatchCallRecord, reason: str) -> None:
        if record.attempts >= MAX_ATTEMPTS:
            await self._mark_failed(
                record,
                f"max_attempts_exceeded ({reason})",
                is_system_error=False,
            )
            return

        now = datetime.now(timezone.utc)
        record.status = "waiting"
        record.last_error = reason
        record.attempts = (record.attempts or 0) + 1
        record.next_attempt_at = now + timedelta(minutes=RETRY_GAP_MINUTES)
        record.updated_at = now
        self._db.commit()

        # Keep waiting_count accurate: we moved from active → waiting (retry)
        _decrement_job_counter(self._db, record.batch_job_id, "active_count")
        _increment_job_counter(self._db, record.batch_job_id, "waiting_count")
        self._db.commit()

        logger.info(
            "Batch record %s scheduled for retry #%d at %s (reason: %s)",
            record.id,
            record.attempts,
            record.next_attempt_at,
            reason,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decrement_job_counter(db: Session, batch_job_id: uuid.UUID, field: str) -> None:
    job = db.get(BatchJob, batch_job_id)
    if job is None:
        return
    current = getattr(job, field) or 0
    setattr(job, field, max(0, current - 1))


def _increment_job_counter(db: Session, batch_job_id: uuid.UUID, field: str) -> None:
    job = db.get(BatchJob, batch_job_id)
    if job is None:
        return
    setattr(job, field, (getattr(job, field) or 0) + 1)


async def _maybe_complete_job(db: Session, batch_job_id: uuid.UUID) -> None:
    """Mark the job completed when no records remain in waiting or active state."""
    job = db.get(BatchJob, batch_job_id)
    if job is None:
        return
    if job.status in ("cancelled", "completed", "failed"):
        return

    remaining = (job.waiting_count or 0) + (job.active_count or 0)
    if remaining == 0:
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        logger.info("BatchJob %s completed", batch_job_id)

        # Fire batch.completed webhook (non-blocking)
        try:
            import asyncio as _asyncio
            from app.services.webhook_service import fire_webhooks

            _asyncio.create_task(
                fire_webhooks(
                    workspace_id=job.workspace_id,
                    event_type="batch.completed",
                    data={
                        "batch_job_id": str(batch_job_id),
                        "agent_id": str(job.agent_id) if job.agent_id else None,
                        "total_count": job.total_count,
                        "completed_count": job.completed_count,
                        "failed_count": job.failed_count,
                        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                    },
                )
            )
        except Exception as _wh_exc:
            logger.warning("batch.completed webhook fire failed: %s", _wh_exc)


async def _bill_connected_call(record: BatchCallRecord, db: Session) -> None:
    """Increment the workspace usage_record for a connected batch call."""
    if record.call_id is None:
        return

    from app.models.call_session import CallSession

    call_session = db.get(CallSession, record.call_id)
    if call_session is None:
        return

    try:
        from app.services.billing_service import BillingService

        BillingService.record_call_usage(
            db, call_session.tenant_id, user_id=call_session.user_id
        )
    except Exception as exc:
        logger.warning("BillingService.record_call_usage failed: %s", exc)


def _build_fake_request(webhook_secret: str):
    """
    Build a minimal Starlette Request that satisfies verify_n8n_webhook_secret_async.

    The function only reads the X-N8N-Webhook-Secret header and the body bytes.
    We provide the header; body resolves to b"".
    """
    from starlette.datastructures import Headers
    from starlette.requests import Request
    from starlette.types import Scope

    secret_bytes = webhook_secret.encode("latin-1") if webhook_secret else b""
    scope: Scope = {
        "type": "http",
        "method": "POST",
        "path": "/internal/batch",
        "query_string": b"",
        "headers": [
            (b"x-n8n-webhook-secret", secret_bytes),
            (b"content-type", b"application/json"),
        ],
        "state": {},
    }

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive=_receive)
