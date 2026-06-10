"""
Bridge Twilio call-events webhooks to the batch call record state machine.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.batch_call_record import BatchCallRecord

# Twilio terminal statuses we care about for batch lifecycle
_TWILIO_TO_ENDED_REASON = {
    "completed": "completed",
    "busy": "busy",
    "no-answer": "no-answer",
    "failed": "failed",
}


async def notify_batch_call_ended(
    db: Session,
    call_session_id: uuid.UUID,
    twilio_status: str,
) -> None:
    """
    If this call session belongs to an active batch record, advance its state
    (complete, retry, or fail) and update job counters / billing.
    """
    ended_reason = _TWILIO_TO_ENDED_REASON.get(twilio_status)
    if ended_reason is None:
        return

    record = (
        db.execute(
            select(BatchCallRecord).where(
                BatchCallRecord.call_id == call_session_id,
                BatchCallRecord.status == "active",
            )
        )
        .scalars()
        .first()
    )
    if record is None:
        return

    from app.services.batch_call_worker_service import BatchCallWorkerService

    svc = BatchCallWorkerService(db)
    try:
        await svc.handle_call_completion(record.id, ended_reason)
        logger.info(
            "Batch record %s updated from call session %s (twilio_status=%s)",
            record.id,
            call_session_id,
            twilio_status,
        )
    except Exception as exc:
        logger.warning(
            "Batch completion failed for record %s session %s: %s",
            record.id,
            call_session_id,
            exc,
            exc_info=True,
        )
