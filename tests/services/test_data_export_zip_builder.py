"""
Unit test for the GDPR export ZIP builder (app/services/data_export_zip_builder.py).

Runs against the shared SQLite test DB (no Postgres required) — exercises the
real streaming SELECT + ZipFile.open(..., "w") write path end to end and
verifies every member ends up in the archive with the expected content.
"""
from __future__ import annotations

import csv
import io
import json
import os
import uuid
import zipfile
from datetime import datetime, timezone

from app.models.agent import Agent
from app.models.audit_log import AuditLog
from app.models.batch_call_record import BatchCallRecord
from app.models.batch_job import BatchJob
from app.models.call_session import CallSession
from app.models.knowledge_base_chunk import KbChunk
from app.models.knowledge_base_document import KnowledgeBase
from app.models.tenant import Tenant
from app.models.transcript_message import TranscriptMessage
from app.models.user import User
from app.services.data_export_zip_builder import build_export_zip


def test_build_export_zip_contains_all_expected_members(db):
    tenant = Tenant(name=f"ExportCo-{uuid.uuid4().hex[:8]}", schema_name="export_co")
    db.add(tenant)
    db.commit()
    db.refresh(tenant)

    user = User(
        email=f"export-{uuid.uuid4().hex[:8]}@example.com",
        hashed_password="hashed",
        first_name="Export",
        last_name="Tester",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    agent = Agent(tenant_id=tenant.id, name="Export Agent")
    db.add(agent)
    db.commit()
    db.refresh(agent)

    call = CallSession(
        user_id=user.id,
        agent_id=agent.id,
        tenant_id=tenant.id,
        start_time=datetime.now(timezone.utc),
        status="completed",
        from_number="+15550001111",
        to_number="+15550002222",
    )
    db.add(call)
    db.commit()
    db.refresh(call)

    transcript = TranscriptMessage(
        call_session_id=call.id,
        role="agent",
        message="Hello, how can I help?",
        sequence_number=1,
    )
    db.add(transcript)

    kb = KnowledgeBase(workspace_id=tenant.id, name="Export KB")
    db.add(kb)
    db.commit()
    db.refresh(kb)

    chunk = KbChunk(kb_id=kb.id, content="Some chunk content", chunk_metadata={"source": "test"})
    db.add(chunk)

    batch_job = BatchJob(workspace_id=tenant.id, agent_id=agent.id, status="completed")
    db.add(batch_job)
    db.commit()
    db.refresh(batch_job)

    batch_record = BatchCallRecord(batch_job_id=batch_job.id, phone_number="+15550003333", status="completed")
    db.add(batch_record)

    audit_row = AuditLog(
        tenant_id=tenant.id,
        user_id=user.id,
        action="agent.created",
        resource_type="agent",
        resource_id=agent.id,
    )
    db.add(audit_row)
    db.commit()

    zip_path = build_export_zip(db, tenant.id)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            assert names == {
                "workspace.json", "calls.csv", "transcripts.json",
                "kb_chunks.json", "audit_events.csv", "batch_job_records.csv",
            }

            workspace_meta = json.loads(zf.read("workspace.json"))
            assert workspace_meta["id"] == str(tenant.id)
            assert workspace_meta["name"] == tenant.name

            calls_rows = list(csv.reader(io.StringIO(zf.read("calls.csv").decode("utf-8"))))
            assert calls_rows[0][0] == "id"
            assert len(calls_rows) == 2  # header + 1 call
            assert str(call.id) in calls_rows[1]
            assert "+15550001111" in calls_rows[1]

            transcripts = json.loads(zf.read("transcripts.json"))
            assert len(transcripts) == 1
            assert transcripts[0]["message"] == "Hello, how can I help?"

            kb_chunks = json.loads(zf.read("kb_chunks.json"))
            assert len(kb_chunks) == 1
            assert kb_chunks[0]["content"] == "Some chunk content"

            audit_rows = list(csv.reader(io.StringIO(zf.read("audit_events.csv").decode("utf-8"))))
            assert audit_rows[0][0] == "id"
            assert len(audit_rows) == 2
            assert "agent.created" in audit_rows[1]

            batch_rows = list(csv.reader(io.StringIO(zf.read("batch_job_records.csv").decode("utf-8"))))
            assert batch_rows[0][0] == "id"
            assert len(batch_rows) == 2
            assert "+15550003333" in batch_rows[1]
    finally:
        os.remove(zip_path)


def test_build_export_zip_empty_workspace_produces_empty_collections(db):
    tenant = Tenant(name=f"EmptyCo-{uuid.uuid4().hex[:8]}", schema_name="empty_co")
    db.add(tenant)
    db.commit()
    db.refresh(tenant)

    zip_path = build_export_zip(db, tenant.id)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            assert json.loads(zf.read("transcripts.json")) == []
            assert json.loads(zf.read("kb_chunks.json")) == []

            calls_rows = list(csv.reader(io.StringIO(zf.read("calls.csv").decode("utf-8"))))
            assert len(calls_rows) == 1  # header only
    finally:
        os.remove(zip_path)
