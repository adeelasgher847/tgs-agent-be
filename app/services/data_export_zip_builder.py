"""
Streams a workspace's exportable data into a ZIP file on local disk.

Each member is written incrementally via ZipFile.open(name, "w"), which
returns a writable stream — rows are read from the DB in chunks
(execution_options(yield_per=...)) and written member-by-member, so the
full export is never held in memory at once (per ticket technical notes).

Members:
  workspace.json      — workspace metadata
  calls.csv           — all CallSession rows for the workspace
  transcripts.json     — all TranscriptMessage rows for the workspace's calls
  kb_chunks.json       — all KbChunk rows for the workspace's knowledge bases
  audit_events.csv     — all AuditLog rows for the workspace
  batch_job_records.csv — all BatchCallRecord rows for the workspace's batch jobs
"""
from __future__ import annotations

import csv
import io
import json
import tempfile
import uuid
import zipfile
from typing import Any, Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.audit_log import AuditLog
from app.models.batch_call_record import BatchCallRecord
from app.models.batch_job import BatchJob
from app.models.call_session import CallSession
from app.models.knowledge_base_chunk import KbChunk
from app.models.knowledge_base_document import KnowledgeBase
from app.models.tenant import Tenant
from app.models.transcript_message import TranscriptMessage

_YIELD_PER = 500


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _write_json_array(zf: zipfile.ZipFile, arcname: str, rows: Iterable[dict]) -> None:
    with zf.open(arcname, "w") as stream:
        stream.write(b"[")
        first = True
        for row in rows:
            if not first:
                stream.write(b",")
            stream.write(json.dumps(row, default=_json_default).encode("utf-8"))
            first = False
        stream.write(b"]")


def _write_csv(
    zf: zipfile.ZipFile,
    arcname: str,
    header: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> None:
    with zf.open(arcname, "w") as stream:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(header)
        stream.write(buf.getvalue().encode("utf-8"))

        for row in rows:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(row)
            stream.write(buf.getvalue().encode("utf-8"))


def _workspace_metadata(db: Session, workspace_id: uuid.UUID) -> dict:
    tenant = db.get(Tenant, workspace_id)
    if tenant is None:
        return {}
    return {
        "id": str(tenant.id),
        "name": tenant.name,
        "status": tenant.status,
        "workspace_type": tenant.workspace_type,
        "created_at": tenant.created_at,
        "updated_at": tenant.updated_at,
    }


def _stream_call_sessions(db: Session, workspace_id: uuid.UUID) -> Iterable[CallSession]:
    result = db.execute(
        select(CallSession)
        .where(CallSession.tenant_id == workspace_id)
        .execution_options(yield_per=_YIELD_PER)
    )
    yield from result.scalars()


def _call_session_row(c: CallSession) -> list:
    return [
        str(c.id), str(c.tenant_id), str(c.agent_id), str(c.user_id),
        c.start_time, c.end_time, c.status, c.duration, c.call_type,
        c.success_evaluation, c.ended_reason, c.cost, c.cost_currency,
        c.from_number, c.to_number, c.twilio_call_sid,
    ]


def _stream_transcripts(db: Session, workspace_id: uuid.UUID) -> Iterable[TranscriptMessage]:
    result = db.execute(
        select(TranscriptMessage)
        .join(CallSession, TranscriptMessage.call_session_id == CallSession.id)
        .where(CallSession.tenant_id == workspace_id)
        .execution_options(yield_per=_YIELD_PER)
    )
    yield from result.scalars()


def _transcript_row(t: TranscriptMessage) -> dict:
    return {
        "id": str(t.id),
        "call_session_id": str(t.call_session_id),
        "role": t.role,
        "message": t.message,
        "message_type": t.message_type,
        "sequence_number": t.sequence_number,
        "created_at": t.created_at,
    }


def _stream_kb_chunks(db: Session, workspace_id: uuid.UUID) -> Iterable[KbChunk]:
    result = db.execute(
        select(KbChunk)
        .join(KnowledgeBase, KbChunk.kb_id == KnowledgeBase.id)
        .where(KnowledgeBase.workspace_id == workspace_id)
        .execution_options(yield_per=_YIELD_PER)
    )
    yield from result.scalars()


def _kb_chunk_row(k: KbChunk) -> dict:
    return {
        "id": str(k.id),
        "kb_id": str(k.kb_id),
        "file_id": str(k.file_id) if k.file_id else None,
        "content": k.content,
        "metadata": k.chunk_metadata,
        "created_at": k.created_at,
    }


def _stream_audit_events(db: Session, workspace_id: uuid.UUID) -> Iterable[AuditLog]:
    result = db.execute(
        select(AuditLog)
        .where(AuditLog.tenant_id == workspace_id)
        .execution_options(yield_per=_YIELD_PER)
    )
    yield from result.scalars()


def _audit_event_row(a: AuditLog) -> list:
    return [
        str(a.id), a.timestamp, a.action, a.resource_type,
        str(a.resource_id) if a.resource_id else "",
        str(a.user_id) if a.user_id else "", a.actor_api_key_prefix or "",
        str(a.ip_address) if a.ip_address else "",
        json.dumps(a.old_value) if a.old_value is not None else "",
        json.dumps(a.new_value) if a.new_value is not None else "",
    ]


def _stream_batch_records(db: Session, workspace_id: uuid.UUID) -> Iterable[BatchCallRecord]:
    result = db.execute(
        select(BatchCallRecord)
        .join(BatchJob, BatchCallRecord.batch_job_id == BatchJob.id)
        .where(BatchJob.workspace_id == workspace_id)
        .execution_options(yield_per=_YIELD_PER)
    )
    yield from result.scalars()


def _batch_record_row(b: BatchCallRecord) -> list:
    return [
        str(b.id), str(b.batch_job_id), b.phone_number, b.status,
        str(b.call_id) if b.call_id else "", b.attempts, b.last_error or "",
        b.created_at,
    ]


def build_export_zip(db: Session, workspace_id: uuid.UUID) -> str:
    """
    Build the full export ZIP on local disk and return its path.

    Caller is responsible for deleting the temp file once it has been
    uploaded to GCS.
    """
    fd, path = tempfile.mkstemp(suffix=".zip", prefix="data-export-")
    import os

    os.close(fd)

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        with zf.open("workspace.json", "w") as stream:
            stream.write(
                json.dumps(
                    _workspace_metadata(db, workspace_id), default=_json_default
                ).encode("utf-8")
            )

        _write_csv(
            zf,
            "calls.csv",
            [
                "id", "tenant_id", "agent_id", "user_id", "start_time", "end_time",
                "status", "duration", "call_type", "success_evaluation",
                "ended_reason", "cost", "cost_currency", "from_number", "to_number",
                "twilio_call_sid",
            ],
            (_call_session_row(c) for c in _stream_call_sessions(db, workspace_id)),
        )

        _write_json_array(
            zf,
            "transcripts.json",
            (_transcript_row(t) for t in _stream_transcripts(db, workspace_id)),
        )

        _write_json_array(
            zf,
            "kb_chunks.json",
            (_kb_chunk_row(k) for k in _stream_kb_chunks(db, workspace_id)),
        )

        _write_csv(
            zf,
            "audit_events.csv",
            [
                "id", "timestamp", "action", "resource_type", "resource_id",
                "actor_user_id", "actor_api_key_prefix", "ip_address",
                "old_value", "new_value",
            ],
            (_audit_event_row(a) for a in _stream_audit_events(db, workspace_id)),
        )

        _write_csv(
            zf,
            "batch_job_records.csv",
            [
                "id", "batch_job_id", "phone_number", "status", "call_id",
                "attempts", "last_error", "created_at",
            ],
            (_batch_record_row(b) for b in _stream_batch_records(db, workspace_id)),
        )

    return path
