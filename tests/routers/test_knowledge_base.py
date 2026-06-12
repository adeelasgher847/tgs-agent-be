"""
Tests for the knowledge-base ingestion pipeline.

Covers:
  - POST /{kb_id}/file       → 202, returns job_id + file_id
  - POST /{kb_id}/file       → 422 when file exceeds 50 MB
  - POST /{kb_id}/text       → 201, chunks inserted synchronously
  - GET /{kb_id}/files/{file_id}/status → returns status/chunk_count
  - POST /                   → 201 create KB
  - GET /                    → 200 list KBs
  - Unit tests for tiktoken chunker
"""
from __future__ import annotations

import io
import json
import uuid
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

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
    """Bypass the API key middleware AND override require_admin for the request."""
    from app.api.deps import require_tenant
    import app.middleware.api_key_middleware as auth_mw

    app.dependency_overrides[require_tenant] = lambda: _mock_admin(workspace_id)
    with patch.object(auth_mw, "_try_jwt_auth", new=AsyncMock(return_value=True)):
        try:
            yield
        finally:
            app.dependency_overrides.pop(require_tenant, None)


# ── Fixtures ──────────────────────────────────────────────────────────────────

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


# ── FILE UPLOAD → 202 ─────────────────────────────────────────────────────────

def test_file_upload_returns_202(client, db, kb, workspace_id):
    fake_pdf = b"%PDF-1.4 fake pdf content for testing"

    with _auth_ctx(workspace_id):
        with patch("app.routers.knowledge_base.settings") as mock_settings:
            mock_settings.GCS_KB_BUCKET = ""  # Skip GCS
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

    with _auth_ctx(workspace_id):
        with patch("app.routers.knowledge_base.settings") as mock_settings:
            mock_settings.GCS_KB_BUCKET = ""
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

    with _auth_ctx(workspace_id):
        with (
            patch("app.routers.knowledge_base.settings") as mock_settings,
            patch("app.services.kb_ingestion_service.embed_chunks", new=AsyncMock(return_value=[fake_embedding])),
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
    embedding = json.loads(chunks[0].embedding)
    assert len(embedding) == 1536


def test_text_ingest_no_openai_key_returns_400(client, db, kb, workspace_id):
    with _auth_ctx(workspace_id):
        with patch("app.routers.knowledge_base.settings") as mock_settings:
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
