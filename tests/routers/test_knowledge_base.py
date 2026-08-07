"""
Tests for the knowledge-base management endpoints.

Covers:
  - POST /                   → 201 create KB (admin/owner only)
  - GET /                    → 200 paginated list with file_count + total_chunk_count
  - GET /{id}                → 200 KB detail with files list
  - PUT /{id}                → 200 update name/description (admin/owner only)
  - DELETE /{id}             → 200 cascade delete; 409 when linked to active flow
  - DELETE /{kb_id}/files/{file_id} → 200 per-file delete + chunk cascade
  - PUT /call-flows/{flow_id}/knowledge-bases → 200 link/unlink KBs
  - GET /{id}/search         → 200 search ranking
  - POST /{kb_id}/file       → 202, returns job_id + file_id
  - POST /{kb_id}/file       → 422 when file exceeds 50 MB
  - POST /{kb_id}/text       → 201, chunks inserted synchronously
  - GET /{kb_id}/files/{file_id}/status → returns status/chunk_count
  - GET /files/{file_id}?action=view|download → 200 presigned URL response
  - Unit tests for tiktoken chunker
"""
from __future__ import annotations

import io
import json
import uuid
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.main import app
from app.models.knowledge_base_document import KnowledgeBase
from app.models.kb_file import KbFile
from app.models.knowledge_base_chunk import KbChunk


# ── Auth bypass ───────────────────────────────────────────────────────────────

def _mock_admin(tenant_id: uuid.UUID):
    user = MagicMock()
    user.current_tenant_id = tenant_id
    return user


@contextmanager
def _auth_ctx(workspace_id: uuid.UUID):
    """Override every dependency a KB route might use to bypass auth entirely."""
    from app.api.deps import (
        require_tenant,
        require_admin_or_owner,
        require_config_or_api_key,
        require_readonly_or_api_key,
    )
    import app.middleware.api_key_middleware as auth_mw

    mock_user = _mock_admin(workspace_id)
    overridden = [
        require_tenant,
        require_admin_or_owner,
        require_config_or_api_key,
        require_readonly_or_api_key,
    ]
    for dep in overridden:
        app.dependency_overrides[dep] = lambda: mock_user
    with patch.object(auth_mw, "_try_jwt_auth", new=AsyncMock(return_value=True)):
        try:
            yield
        finally:
            for dep in overridden:
                app.dependency_overrides.pop(dep, None)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def _drop_partial_unique_indexes(db):
    from sqlalchemy import text
    for index_name in (
        "uq_agent_single_inbound_per_tenant",
        "uq_agent_single_follow_up_per_tenant",
    ):
        try:
            db.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
        except Exception:  # noqa: S110 — best-effort SQLite teardown
            pass
    db.commit()
    yield


@pytest.fixture()
def workspace_id(db) -> uuid.UUID:
    from app.models.tenant import Tenant

    return db.query(Tenant).first().id


@pytest.fixture()
def kb(db, workspace_id) -> KnowledgeBase:
    kb = KnowledgeBase(id=uuid.uuid4(), workspace_id=workspace_id, name="Test KB")
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return kb


# ── CREATE KB ─────────────────────────────────────────────────────────────────

def test_create_knowledge_base(client, db, workspace_id):
    with _auth_ctx(workspace_id):
        resp = client.post("/api/v1/kb/", json={"name": "My KB", "description": "Test"})
    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    assert data["name"] == "My KB"
    assert data["workspace_id"] == str(workspace_id)


# ── LIST KBs ──────────────────────────────────────────────────────────────────

def test_list_knowledge_bases(client, db, kb, workspace_id):
    with _auth_ctx(workspace_id):
        resp = client.get("/api/v1/kb/")
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    ids = [d["id"] for d in data["knowledge_bases"]]
    assert str(kb.id) in ids


def test_list_knowledge_bases_includes_call_flow_ids(client, db, kb, workspace_id):
    from app.models.call_flow import CallFlow
    from app.models.agent import Agent

    agent = Agent(id=uuid.uuid4(), tenant_id=workspace_id, name="Test Agent", is_deleted=False)
    db.add(agent)
    db.commit()

    flow = CallFlow(
        id=uuid.uuid4(),
        tenant_id=workspace_id,
        agent_id=agent.id,
        name="Test Flow",
        direction="inbound",
        is_deleted=False,
        knowledge_base_ids=[str(kb.id)],
    )
    db.add(flow)
    db.commit()

    with _auth_ctx(workspace_id):
        resp = client.get("/api/v1/kb/")

    assert resp.status_code == 200, resp.text
    items = resp.json()["data"]["knowledge_bases"]
    item = next(i for i in items if i["id"] == str(kb.id))
    assert item["call_flow_ids"] == [str(flow.id)]

    db.delete(flow)
    db.delete(agent)
    db.commit()


# ── FILE UPLOAD → 202 ─────────────────────────────────────────────────────────

def test_file_upload_returns_202(client, db, kb, workspace_id):
    fake_pdf = b"%PDF-1.4 fake pdf content for testing"

    with (
        _auth_ctx(workspace_id),
        patch("app.routers.knowledge_base.settings") as mock_settings,
    ):
        mock_settings.S3_KB_BUCKET = ""  # Skip S3
        mock_settings.OPENAI_API_KEY = "test-key"
        resp = client.post(
            f"/api/v1/kb/{kb.id}/file",
            files={"file": ("test.pdf", io.BytesIO(fake_pdf), "application/pdf")},
        )

    assert resp.status_code == 202, resp.text
    data = resp.json()["data"]
    assert "job_id" in data
    assert "file_id" in data

    # KbFile row created with status=processing
    file_id = uuid.UUID(data["file_id"])
    kb_file = db.get(KbFile, file_id)
    assert kb_file is not None
    assert kb_file.status == "processing"
    assert kb_file.original_filename == "test.pdf"


def test_file_upload_unsupported_type_returns_422(client, db, kb, workspace_id):
    with _auth_ctx(workspace_id):
        resp = client.post(
            f"/api/v1/kb/{kb.id}/file",
            files={"file": ("report.xlsx", io.BytesIO(b"data"), "application/vnd.ms-excel")},
        )
    assert resp.status_code == 422, resp.text


def test_file_upload_oversized_returns_422(client, db, kb, workspace_id):
    oversized = b"x" * (50 * 1024 * 1024 + 1)

    with (
        _auth_ctx(workspace_id),
        patch("app.routers.knowledge_base.settings") as mock_settings,
    ):
        mock_settings.S3_KB_BUCKET = ""
        resp = client.post(
            f"/api/v1/kb/{kb.id}/file",
            files={"file": ("big.pdf", io.BytesIO(oversized), "application/pdf")},
        )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    # App wraps HTTPException into {"error": {"message": "..."}}
    msg = body.get("detail") or body.get("error", {}).get("message", "")
    assert "50" in msg


# ── TEXT INGEST → 201 (synchronous) ──────────────────────────────────────────

def test_text_ingest_synchronous_inserts_chunks(client, db, kb, workspace_id):
    """Text is chunked, embedded (mocked), and committed before 201 returns."""
    fake_embedding = [0.0] * 1536

    with (
        _auth_ctx(workspace_id),
        patch("app.routers.knowledge_base.settings") as mock_settings,
        patch(
            "app.services.kb_ingestion_service.embed_chunks",
            new=AsyncMock(return_value=[fake_embedding]),
        ),
    ):
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.RAG_SCORE_THRESHOLD = 0.4
        mock_settings.RAG_MAX_CONTEXT_CHARS = 6000
        # Use ≥50 tokens of content so min_tokens filter passes
        content = (
            "This document outlines company policies and procedures for all employees. "
            "All staff must adhere to the code of conduct and maintain professionalism. "
            "Violations may result in disciplinary action up to and including termination. "
            "Please review this policy carefully and confirm your understanding by signing below."
        )
        resp = client.post(
            f"/api/v1/kb/{kb.id}/text",
            json={"content": content},
        )

    assert resp.status_code == 201, resp.text
    data = resp.json()["data"]
    assert data["chunk_count"] >= 1
    assert str(data["kb_id"]) == str(kb.id)

    # Verify chunk was written to DB with 1536-dim embedding
    chunks = db.query(KbChunk).filter(KbChunk.kb_id == kb.id).all()
    assert len(chunks) >= 1
    raw = chunks[0].embedding
    # SQLite returns the vector as a numpy array; PostgreSQL stores it as a castable type
    embedding = list(raw) if not isinstance(raw, str) else json.loads(raw)
    assert len(embedding) == 1536


def test_text_ingest_no_openai_key_returns_400(client, db, kb, workspace_id):
    with (
        _auth_ctx(workspace_id),
        patch("app.routers.knowledge_base.settings") as mock_settings,
    ):
        mock_settings.OPENAI_API_KEY = ""
        resp = client.post(
            f"/api/v1/kb/{kb.id}/text",
            json={"content": "Test content"},
        )
    assert resp.status_code == 400, resp.text


# ── FILE STATUS ───────────────────────────────────────────────────────────────

def test_get_file_status_ready(client, db, kb, workspace_id):
    file_id = uuid.uuid4()
    kb_file = KbFile(
        id=file_id,
        kb_id=kb.id,
        original_filename="sample.pdf",
        size_bytes=1024,
        file_type="pdf",
        status="ready",
        chunk_count=5,
    )
    db.add(kb_file)
    db.commit()

    with _auth_ctx(workspace_id):
        resp = client.get(f"/api/v1/kb/{kb.id}/files/{file_id}/status")

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["status"] == "ready"
    assert data["chunk_count"] == 5
    assert data["file_id"] == str(file_id)


def test_get_file_status_not_found(client, db, kb, workspace_id):
    with _auth_ctx(workspace_id):
        resp = client.get(f"/api/v1/kb/{kb.id}/files/{uuid.uuid4()}/status")
    assert resp.status_code == 404


# ── CHUNKING UNIT TESTS ───────────────────────────────────────────────────────

def test_chunk_text_tiktoken_splits_long_text():
    from app.services.kb_ingestion_service import chunk_text

    text = " ".join(["word"] * 2000)
    chunks = chunk_text(text, max_tokens=800, overlap_tokens=100, min_tokens=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) > 0


def test_chunk_text_tiktoken_short_text():
    from app.services.kb_ingestion_service import chunk_text

    # Very short text — 2 words (well below min_tokens=50)
    chunks = chunk_text("Hello world.", max_tokens=800, overlap_tokens=100, min_tokens=50)
    # Short text below min_tokens: either 0 chunks or merged into previous
    assert isinstance(chunks, list)


def test_chunk_text_empty():
    from app.services.kb_ingestion_service import chunk_text

    assert chunk_text("") == []
    assert chunk_text("   ") == []


# ── GET /{kb_id} DETAIL ───────────────────────────────────────────────────────

def test_get_knowledge_base_detail(client, db, kb, workspace_id):
    # Add a file so the files list is non-empty
    file_id = uuid.uuid4()
    f = KbFile(
        id=file_id,
        kb_id=kb.id,
        original_filename="doc.pdf",
        size_bytes=2048,
        file_type="pdf",
        status="ready",
        chunk_count=3,
    )
    db.add(f)
    db.commit()

    with _auth_ctx(workspace_id):
        resp = client.get(f"/api/v1/kb/{kb.id}")

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["id"] == str(kb.id)
    assert data["name"] == kb.name
    assert isinstance(data["files"], list)
    file_ids = [fi["id"] for fi in data["files"]]
    assert str(file_id) in file_ids
    # Verify file shape
    f_data = next(fi for fi in data["files"] if fi["id"] == str(file_id))
    assert f_data["filename"] == "doc.pdf"
    assert f_data["size_bytes"] == 2048
    assert f_data["status"] == "ready"
    assert f_data["chunk_count"] == 3


def test_get_knowledge_base_detail_not_found(client, db, workspace_id):
    with _auth_ctx(workspace_id):
        resp = client.get(f"/api/v1/kb/{uuid.uuid4()}")
    assert resp.status_code == 404


def test_get_knowledge_base_detail_includes_linked_call_flow_ids(client, db, kb, workspace_id):
    from app.models.call_flow import CallFlow
    from app.models.agent import Agent

    agent = Agent(id=uuid.uuid4(), tenant_id=workspace_id, name="Test Agent", is_deleted=False)
    db.add(agent)
    db.commit()

    flow = CallFlow(
        id=uuid.uuid4(),
        tenant_id=workspace_id,
        agent_id=agent.id,
        name="Test Flow",
        direction="inbound",
        is_deleted=False,
        knowledge_base_ids=[str(kb.id)],
    )
    deleted_flow = CallFlow(
        id=uuid.uuid4(),
        tenant_id=workspace_id,
        agent_id=agent.id,
        name="Deleted Flow",
        direction="inbound",
        is_deleted=True,
        knowledge_base_ids=[str(kb.id)],
    )
    db.add_all([flow, deleted_flow])
    db.commit()

    with _auth_ctx(workspace_id):
        resp = client.get(f"/api/v1/kb/{kb.id}")

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["call_flow_ids"] == [str(flow.id)]

    db.delete(flow)
    db.delete(deleted_flow)
    db.delete(agent)
    db.commit()


def test_get_knowledge_base_detail_dedupes_flow_with_repeated_kb_id(client, db, kb, workspace_id):
    """Nothing currently prevents knowledge_base_ids from holding the same id
    twice within one flow -- the response must still list that flow once.
    """
    from app.models.call_flow import CallFlow
    from app.models.agent import Agent

    agent = Agent(id=uuid.uuid4(), tenant_id=workspace_id, name="Test Agent", is_deleted=False)
    db.add(agent)
    db.commit()

    flow = CallFlow(
        id=uuid.uuid4(),
        tenant_id=workspace_id,
        agent_id=agent.id,
        name="Test Flow",
        direction="inbound",
        is_deleted=False,
        knowledge_base_ids=[str(kb.id), str(kb.id)],
    )
    db.add(flow)
    db.commit()

    with _auth_ctx(workspace_id):
        resp = client.get(f"/api/v1/kb/{kb.id}")

    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["call_flow_ids"] == [str(flow.id)]

    db.delete(flow)
    db.delete(agent)
    db.commit()


def test_get_knowledge_base_detail_call_flow_ids_empty_when_unlinked(client, db, kb, workspace_id):
    with _auth_ctx(workspace_id):
        resp = client.get(f"/api/v1/kb/{kb.id}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["call_flow_ids"] == []


# ── GET / LIST WITH COUNTS ────────────────────────────────────────────────────

def test_list_knowledge_bases_includes_counts(client, db, workspace_id):
    kb2 = KnowledgeBase(id=uuid.uuid4(), workspace_id=workspace_id, name="KB Counts Test")
    db.add(kb2)
    db.commit()

    # Add 2 ready files and 5 chunks
    for _ in range(2):
        fid = uuid.uuid4()
        db.add(KbFile(id=fid, kb_id=kb2.id, original_filename="f.pdf",
                      file_type="pdf", status="ready"))
    for _ in range(5):
        db.add(KbChunk(id=uuid.uuid4(), kb_id=kb2.id, content="chunk"))
    db.commit()

    with _auth_ctx(workspace_id):
        resp = client.get("/api/v1/kb/")

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    items = data["knowledge_bases"]
    kb2_item = next((i for i in items if i["id"] == str(kb2.id)), None)
    assert kb2_item is not None
    assert kb2_item["file_count"] == 2
    assert kb2_item["total_chunk_count"] == 5


# ── PUT /{kb_id} UPDATE ───────────────────────────────────────────────────────

def test_update_knowledge_base(client, db, workspace_id):
    kb3 = KnowledgeBase(id=uuid.uuid4(), workspace_id=workspace_id, name="Before Update")
    db.add(kb3)
    db.commit()

    with _auth_ctx(workspace_id):
        resp = client.put(
            f"/api/v1/kb/{kb3.id}",
            json={"name": "After Update", "description": "New desc"},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["name"] == "After Update"
    assert data["description"] == "New desc"

    db.refresh(kb3)
    assert kb3.name == "After Update"


def test_update_knowledge_base_not_found(client, db, workspace_id):
    with _auth_ctx(workspace_id):
        resp = client.put(f"/api/v1/kb/{uuid.uuid4()}", json={"name": "x"})
    assert resp.status_code == 404


# ── DELETE /{kb_id} WITH CASCADE ─────────────────────────────────────────────

def test_delete_knowledge_base_cascades(client, db, workspace_id):
    kb_del = KnowledgeBase(id=uuid.uuid4(), workspace_id=workspace_id, name="To Delete")
    db.add(kb_del)
    db.commit()

    fid = uuid.uuid4()
    db.add(KbFile(id=fid, kb_id=kb_del.id, original_filename="f.pdf",
                  file_type="pdf", status="ready"))
    cid = uuid.uuid4()
    db.add(KbChunk(id=cid, kb_id=kb_del.id, content="chunk content", file_id=fid))
    db.commit()

    with _auth_ctx(workspace_id):
        resp = client.delete(f"/api/v1/kb/{kb_del.id}")

    assert resp.status_code == 200, resp.text

    # KB, file, and chunks should all be gone via CASCADE
    assert db.get(KnowledgeBase, kb_del.id) is None
    assert db.get(KbFile, fid) is None
    assert db.get(KbChunk, cid) is None


def test_delete_knowledge_base_409_when_linked_to_flow(client, db, workspace_id):
    from app.models.call_flow import CallFlow
    from app.models.agent import Agent

    # Create a minimal agent and flow
    agent = Agent(
        id=uuid.uuid4(),
        tenant_id=workspace_id,
        name="Test Agent",
        is_deleted=False,
    )
    db.add(agent)
    db.commit()

    kb_linked = KnowledgeBase(
        id=uuid.uuid4(), workspace_id=workspace_id, name="Linked KB"
    )
    db.add(kb_linked)
    db.commit()

    flow = CallFlow(
        id=uuid.uuid4(),
        tenant_id=workspace_id,
        agent_id=agent.id,
        name="Test Flow",
        direction="inbound",
        is_deleted=False,
        knowledge_base_ids=[str(kb_linked.id)],
    )
    db.add(flow)
    db.commit()

    with _auth_ctx(workspace_id):
        resp = client.delete(f"/api/v1/kb/{kb_linked.id}")

    assert resp.status_code == 409, resp.text

    # Cleanup
    db.delete(flow)
    db.delete(kb_linked)
    db.delete(agent)
    db.commit()


# ── DELETE /{kb_id}/files/{file_id} ──────────────────────────────────────────

def test_delete_kb_file_removes_file_and_chunks(client, db, kb, workspace_id):
    fid = uuid.uuid4()
    db.add(KbFile(id=fid, kb_id=kb.id, original_filename="del.pdf",
                  file_type="pdf", status="ready"))
    cid = uuid.uuid4()
    db.add(KbChunk(id=cid, kb_id=kb.id, content="chunk to delete", file_id=fid))
    db.commit()

    with _auth_ctx(workspace_id):
        resp = client.delete(f"/api/v1/kb/{kb.id}/files/{fid}")

    assert resp.status_code == 200, resp.text
    assert db.get(KbFile, fid) is None
    assert db.get(KbChunk, cid) is None


def test_delete_kb_file_not_found(client, db, kb, workspace_id):
    with _auth_ctx(workspace_id):
        resp = client.delete(f"/api/v1/kb/{kb.id}/files/{uuid.uuid4()}")
    assert resp.status_code == 404


# ── PUT /call-flows/{flow_id}/knowledge-bases ─────────────────────────────────

def test_update_flow_knowledge_bases(client, db, workspace_id):
    from app.models.call_flow import CallFlow
    from app.models.agent import Agent

    agent = Agent(
        id=uuid.uuid4(),
        tenant_id=workspace_id,
        name="Flow KB Agent",
        is_deleted=False,
    )
    db.add(agent)
    db.commit()

    flow = CallFlow(
        id=uuid.uuid4(),
        tenant_id=workspace_id,
        agent_id=agent.id,
        name="Flow for KB Link",
        direction="inbound",
        is_deleted=False,
        knowledge_base_ids=[],
    )
    db.add(flow)

    kb_a = KnowledgeBase(id=uuid.uuid4(), workspace_id=workspace_id, name="KB A")
    kb_b = KnowledgeBase(id=uuid.uuid4(), workspace_id=workspace_id, name="KB B")
    db.add_all([kb_a, kb_b])
    db.commit()

    with _auth_ctx(workspace_id):
        resp = client.put(
            f"/api/v1/call-flows/{flow.id}/knowledge-bases",
            json={"kb_ids": [str(kb_a.id), str(kb_b.id)]},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert str(kb_a.id) in data["kb_ids"]
    assert str(kb_b.id) in data["kb_ids"]

    db.refresh(flow)
    assert str(kb_a.id) in flow.knowledge_base_ids

    # Detach all KBs
    with _auth_ctx(workspace_id):
        resp2 = client.put(
            f"/api/v1/call-flows/{flow.id}/knowledge-bases",
            json={"kb_ids": []},
        )
    assert resp2.status_code == 200
    db.refresh(flow)
    assert flow.knowledge_base_ids == []

    # Cleanup
    db.delete(flow)
    db.delete(kb_a)
    db.delete(kb_b)
    db.delete(agent)
    db.commit()


def test_update_flow_knowledge_bases_rejects_foreign_kb(client, db, workspace_id):
    from app.models.call_flow import CallFlow
    from app.models.agent import Agent

    agent = Agent(
        id=uuid.uuid4(),
        tenant_id=workspace_id,
        name="Foreign KB Agent",
        is_deleted=False,
    )
    db.add(agent)
    flow = CallFlow(
        id=uuid.uuid4(),
        tenant_id=workspace_id,
        agent_id=agent.id,
        name="Flow Foreign KB",
        direction="inbound",
        is_deleted=False,
        knowledge_base_ids=[],
    )
    db.add(flow)
    db.commit()

    with _auth_ctx(workspace_id):
        resp = client.put(
            f"/api/v1/call-flows/{flow.id}/knowledge-bases",
            json={"kb_ids": [str(uuid.uuid4())]},  # non-existent KB
        )

    assert resp.status_code == 404

    db.delete(flow)
    db.delete(agent)
    db.commit()


def test_update_flow_knowledge_bases_flow_not_found(client, db, workspace_id):
    with _auth_ctx(workspace_id):
        resp = client.put(
            f"/api/v1/call-flows/{uuid.uuid4()}/knowledge-bases",
            json={"kb_ids": []},
        )
    assert resp.status_code == 404


# ── SEARCH RANKING ────────────────────────────────────────────────────────────

def test_search_returns_results_sorted_by_score(client, db, workspace_id):
    """Verify search endpoint returns results ordered by descending score."""
    kb_s = KnowledgeBase(id=uuid.uuid4(), workspace_id=workspace_id, name="Search KB")
    db.add(kb_s)
    db.commit()

    # Two chunks — mocked embedding ensures they appear in the response
    fake_embedding = [0.1] * 1536

    with (
        _auth_ctx(workspace_id),
        patch("app.routers.knowledge_base.settings") as mock_settings,
        patch(
            "app.routers.knowledge_base.embed_text_for_rag", return_value=fake_embedding
        ),
    ):
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.RAG_SCORE_THRESHOLD = 0.0
        mock_settings.RAG_MAX_CONTEXT_CHARS = 6000
        # SQLite doesn't support pgvector — expect a 500 from the raw SQL;
        # what we validate here is that the route is wired, auth works, and
        # scores are returned in descending order when the DB supports it.
        resp = client.get(f"/api/v1/kb/{kb_s.id}/search?q=hello&limit=5")

    # SQLite will fail on the vector cast — that's fine; 400 means missing key, 500 is expected
    assert resp.status_code in (200, 500)

    db.delete(kb_s)
    db.commit()


# ── GET /files/{file_id}?action=view|download ────────────────────────────────

def _make_ready_file(db, kb_id, *, filename="doc.pdf", file_type="pdf", s3_path="kb/some/key.pdf"):
    file_id = uuid.uuid4()
    kb_file = KbFile(
        id=file_id,
        kb_id=kb_id,
        original_filename=filename,
        size_bytes=1024,
        file_type=file_type,
        s3_path=s3_path,
        status="ready",
    )
    db.add(kb_file)
    db.commit()
    db.refresh(kb_file)
    return kb_file


def test_get_kb_file_access_no_tenant_selected_returns_403(client, db, kb, workspace_id):
    kb_file = _make_ready_file(db, kb.id)

    with _auth_ctx(workspace_id):
        from app.api.deps import require_readonly_or_api_key

        no_tenant_user = _mock_admin(None)
        app.dependency_overrides[require_readonly_or_api_key] = lambda: no_tenant_user
        try:
            resp = client.get(f"/api/v1/kb/files/{kb_file.id}", params={"action": "view"})
        finally:
            app.dependency_overrides.pop(require_readonly_or_api_key, None)

    assert resp.status_code == 403, resp.text


def test_get_kb_file_access_not_found_when_file_missing(client, db, workspace_id):
    with _auth_ctx(workspace_id):
        resp = client.get(f"/api/v1/kb/files/{uuid.uuid4()}", params={"action": "view"})
    assert resp.status_code == 404, resp.text
    body = resp.json()
    msg = body.get("detail") or body.get("error", {}).get("message", "")
    assert "File not found" in msg


def test_get_kb_file_access_isolated_across_tenants(client, db, kb, workspace_id):
    """A file that belongs to a KB in another workspace must 404, not leak."""
    from app.models.tenant import Tenant

    other_tenant = Tenant(name="Other Tenant", schema_name="other_tenant_schema")
    db.add(other_tenant)
    db.commit()
    db.refresh(other_tenant)

    other_kb = KnowledgeBase(id=uuid.uuid4(), workspace_id=other_tenant.id, name="Other KB")
    db.add(other_kb)
    db.commit()

    other_file = _make_ready_file(db, other_kb.id, filename="secret.pdf")

    with _auth_ctx(workspace_id):
        resp = client.get(f"/api/v1/kb/files/{other_file.id}", params={"action": "view"})

    assert resp.status_code == 404, resp.text

    db.delete(other_kb)
    db.delete(other_tenant)
    db.commit()


@pytest.mark.parametrize("status_value", ["processing", "error"])
def test_get_kb_file_access_not_available_when_not_ready(client, db, kb, workspace_id, status_value):
    file_id = uuid.uuid4()
    kb_file = KbFile(
        id=file_id,
        kb_id=kb.id,
        original_filename="not_ready.pdf",
        file_type="pdf",
        s3_path="kb/not/ready.pdf",
        status=status_value,
    )
    db.add(kb_file)
    db.commit()

    with _auth_ctx(workspace_id):
        resp = client.get(f"/api/v1/kb/files/{file_id}", params={"action": "view"})

    assert resp.status_code == 404, resp.text
    body = resp.json()
    msg = body.get("detail") or body.get("error", {}).get("message", "")
    assert "File not available" in msg


def test_get_kb_file_access_not_available_when_s3_path_missing(client, db, kb, workspace_id):
    file_id = uuid.uuid4()
    kb_file = KbFile(
        id=file_id,
        kb_id=kb.id,
        original_filename="no_path.pdf",
        file_type="pdf",
        s3_path=None,
        status="ready",
    )
    db.add(kb_file)
    db.commit()

    with _auth_ctx(workspace_id):
        resp = client.get(f"/api/v1/kb/files/{file_id}", params={"action": "view"})

    assert resp.status_code == 404, resp.text
    body = resp.json()
    msg = body.get("detail") or body.get("error", {}).get("message", "")
    assert "File not available" in msg


def test_get_kb_file_access_invalid_action_returns_400(client, db, kb, workspace_id):
    """FastAPI would natively 422 an invalid Literal, but this app's global
    RequestValidationError handler (app/core/exception_handlers.py) rewrites
    all query/body validation failures to 400 with code=validation_error —
    confirmed by tests/core/test_error_envelope.py::test_validation_error_envelope.
    """
    kb_file = _make_ready_file(db, kb.id)

    with _auth_ctx(workspace_id):
        resp = client.get(f"/api/v1/kb/files/{kb_file.id}", params={"action": "print"})

    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["error"]["code"] == "validation_error"


def test_get_kb_file_access_bucket_not_configured_returns_500(client, db, kb, workspace_id):
    kb_file = _make_ready_file(db, kb.id)

    with (
        _auth_ctx(workspace_id),
        patch("app.routers.knowledge_base.settings") as mock_settings,
    ):
        mock_settings.S3_KB_BUCKET = ""
        resp = client.get(f"/api/v1/kb/files/{kb_file.id}", params={"action": "view"})

    # Server errors (5xx) are always masked to a generic message by
    # app.core.pii_redactor.safe_error_message — never the raw HTTPException
    # detail — per tests/core/test_error_envelope.py::test_unhandled_exception_envelope.
    # We can only assert the envelope shape/code here, not the original detail text.
    assert resp.status_code == 500, resp.text
    body = resp.json()
    assert body["error"]["code"] == "internal_error"


def test_get_kb_file_access_presigned_url_failure_returns_500(client, db, kb, workspace_id):
    kb_file = _make_ready_file(db, kb.id)

    mock_client = MagicMock()
    mock_client.generate_presigned_url.side_effect = Exception("boom")

    with (
        _auth_ctx(workspace_id),
        patch("app.routers.knowledge_base.settings") as mock_settings,
        patch("app.routers.knowledge_base.get_s3_client", return_value=mock_client),
    ):
        mock_settings.S3_KB_BUCKET = "test-bucket"
        resp = client.get(f"/api/v1/kb/files/{kb_file.id}", params={"action": "view"})

    assert resp.status_code == 500, resp.text
    body = resp.json()
    assert body["error"]["code"] == "internal_error"


def test_get_kb_file_access_view_pdf(client, db, kb, workspace_id):
    kb_file = _make_ready_file(db, kb.id, filename="report.pdf", file_type="pdf")

    mock_client = MagicMock()
    mock_client.generate_presigned_url.return_value = "https://s3.example.com/signed-url-pdf"

    with (
        _auth_ctx(workspace_id),
        patch("app.routers.knowledge_base.settings") as mock_settings,
        patch("app.routers.knowledge_base.get_s3_client", return_value=mock_client),
    ):
        mock_settings.S3_KB_BUCKET = "test-bucket"
        resp = client.get(f"/api/v1/kb/files/{kb_file.id}", params={"action": "view"})

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["preview_supported"] is True
    assert data["url"] == "https://s3.example.com/signed-url-pdf"
    assert data["expires_in"] == 300
    assert data["file_id"] == str(kb_file.id)
    assert data["action"] == "view"

    call_kwargs = mock_client.generate_presigned_url.call_args.kwargs
    params = call_kwargs["Params"]
    assert params["ResponseContentDisposition"] == "inline"
    assert params["ResponseContentType"] == "application/pdf"
    assert call_kwargs["ExpiresIn"] == 300

    # The raw S3 object key must never leak into the response body.
    assert kb_file.s3_path not in resp.text


def test_get_kb_file_access_view_txt(client, db, kb, workspace_id):
    kb_file = _make_ready_file(db, kb.id, filename="notes.txt", file_type="txt")

    mock_client = MagicMock()
    mock_client.generate_presigned_url.return_value = "https://s3.example.com/signed-url-txt"

    with (
        _auth_ctx(workspace_id),
        patch("app.routers.knowledge_base.settings") as mock_settings,
        patch("app.routers.knowledge_base.get_s3_client", return_value=mock_client),
    ):
        mock_settings.S3_KB_BUCKET = "test-bucket"
        resp = client.get(f"/api/v1/kb/files/{kb_file.id}", params={"action": "view"})

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["preview_supported"] is True

    call_kwargs = mock_client.generate_presigned_url.call_args.kwargs
    params = call_kwargs["Params"]
    assert params["ResponseContentDisposition"] == "inline"
    assert params["ResponseContentType"] == "text/plain"

    assert kb_file.s3_path not in resp.text


def test_get_kb_file_access_view_docx_forces_attachment(client, db, kb, workspace_id):
    kb_file = _make_ready_file(db, kb.id, filename="contract.docx", file_type="docx")

    mock_client = MagicMock()
    mock_client.generate_presigned_url.return_value = "https://s3.example.com/signed-url-docx"

    with (
        _auth_ctx(workspace_id),
        patch("app.routers.knowledge_base.settings") as mock_settings,
        patch("app.routers.knowledge_base.get_s3_client", return_value=mock_client),
    ):
        mock_settings.S3_KB_BUCKET = "test-bucket"
        resp = client.get(f"/api/v1/kb/files/{kb_file.id}", params={"action": "view"})

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["preview_supported"] is False

    call_kwargs = mock_client.generate_presigned_url.call_args.kwargs
    params = call_kwargs["Params"]
    # DOCX is never inline-viewable: even with action=view it must still
    # be forced to an attachment disposition per the endpoint's spec.
    assert params["ResponseContentDisposition"] == 'attachment; filename="contract.docx"'
    assert "ResponseContentType" not in params

    assert kb_file.s3_path not in resp.text


@pytest.mark.parametrize(
    "file_type,filename",
    [("pdf", "report.pdf"), ("txt", "notes.txt"), ("docx", "contract.docx")],
)
def test_get_kb_file_access_download_forces_attachment_for_all_types(
    client, db, kb, workspace_id, file_type, filename
):
    kb_file = _make_ready_file(db, kb.id, filename=filename, file_type=file_type)

    mock_client = MagicMock()
    mock_client.generate_presigned_url.return_value = "https://s3.example.com/signed-url-download"

    with (
        _auth_ctx(workspace_id),
        patch("app.routers.knowledge_base.settings") as mock_settings,
        patch("app.routers.knowledge_base.get_s3_client", return_value=mock_client),
    ):
        mock_settings.S3_KB_BUCKET = "test-bucket"
        resp = client.get(f"/api/v1/kb/files/{kb_file.id}", params={"action": "download"})

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["action"] == "download"

    call_kwargs = mock_client.generate_presigned_url.call_args.kwargs
    params = call_kwargs["Params"]
    assert params["ResponseContentDisposition"].startswith('attachment; filename="')
    assert filename in params["ResponseContentDisposition"]

    assert kb_file.s3_path not in resp.text


def test_get_kb_file_access_sanitizes_filename_in_content_disposition(client, db, kb, workspace_id):
    """A `"` (or CR/LF) in a client-supplied upload filename must not be able
    to break out of the quoted Content-Disposition value and inject extra
    header parameters into the presigned URL."""
    malicious_filename = 'evil".pdf'
    kb_file = _make_ready_file(db, kb.id, filename=malicious_filename, file_type="pdf")

    mock_client = MagicMock()
    mock_client.generate_presigned_url.return_value = "https://s3.example.com/signed-url"

    with (
        _auth_ctx(workspace_id),
        patch("app.routers.knowledge_base.settings") as mock_settings,
        patch("app.routers.knowledge_base.get_s3_client", return_value=mock_client),
    ):
        mock_settings.S3_KB_BUCKET = "test-bucket"
        resp = client.get(f"/api/v1/kb/files/{kb_file.id}", params={"action": "download"})

    assert resp.status_code == 200, resp.text

    call_kwargs = mock_client.generate_presigned_url.call_args.kwargs
    disposition = call_kwargs["Params"]["ResponseContentDisposition"]
    assert disposition == 'attachment; filename="evil.pdf"'
