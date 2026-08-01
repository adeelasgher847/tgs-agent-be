"""
Regression tests for RagService.ingest_document embedding serialization.

Bug: KbChunk.embedding was being assigned a JSON-stringified vector
(`json.dumps(embedding)`) instead of a raw list[float]. pgvector's
Vector._to_db() (invoked by SQLAlchemy's bind_processor at INSERT time)
cannot parse a stringified vector and raises:

    ValueError: could not convert string to float: '[-0.023..., 0.014..., ...]'

These tests reproduce the failure at the point of stringification (without
needing a live Postgres+pgvector instance) and confirm the fix restores a
raw list[float] on the ORM object before it ever reaches the DB driver.
"""
from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock

import pytest
from pgvector.utils import Vector

from app.services.rag_service import RagService


FAKE_EMBEDDING = [0.1, 0.2, 0.3, 0.4, 0.5]


def _embedding_func(text: str):
    return FAKE_EMBEDDING


def _make_fake_db():
    db = MagicMock()
    db.add_all = MagicMock()
    db.commit = MagicMock()
    db.rollback = MagicMock()
    return db


def test_ingest_document_attaches_raw_list_embedding(monkeypatch):
    """
    RagService.ingest_document must attach embedding as a raw list[float]
    to KbChunk.embedding, not a JSON-stringified blob. A stringified value
    passes SQLAlchemy's ORM layer fine but blows up at INSERT time inside
    pgvector's bind_processor (see companion test below) -- so the
    assertion here is the actual root-cause guard.
    """
    service = RagService()
    db = _make_fake_db()

    fake_kb = MagicMock()
    fake_kb.id = uuid.uuid4()
    monkeypatch.setattr(service, "_get_or_create_auto_kb", lambda db, tenant_id: fake_kb)
    monkeypatch.setattr(service, "_delete_chunks_by_meta_source_ref", lambda *a, **kw: None)

    tenant_id = uuid.uuid4()
    service.ingest_document(
        tenant_id=tenant_id,
        agent_id=None,
        title="Doc",
        source_type="text",
        source_ref="ref-1",
        full_text="Some short text to embed for the knowledge base.",
        embedding_func=_embedding_func,
        db_session=db,
    )

    assert db.add_all.called
    chunk_rows = db.add_all.call_args[0][0]
    assert len(chunk_rows) >= 1
    chunk = chunk_rows[0]

    assert isinstance(chunk.embedding, list), (
        f"expected embedding to be a raw list[float], got {type(chunk.embedding)}: "
        f"{chunk.embedding!r}"
    )
    assert all(isinstance(x, (float, int)) for x in chunk.embedding)
    assert len(chunk.embedding) == len(FAKE_EMBEDDING)

    # Simulate exactly what pgvector's bind_processor does at INSERT time --
    # this must not raise.
    Vector._to_db(chunk.embedding, dim=len(FAKE_EMBEDDING))


def test_stringified_embedding_reproduces_insert_time_valueerror():
    """
    Pins down the exact production failure mode: feeding a JSON-stringified
    embedding (what the buggy code produced) through pgvector's Vector._to_db
    raises `ValueError: could not convert string to float`, matching the
    reported traceback.
    """
    stringified = json.dumps(FAKE_EMBEDDING)
    with pytest.raises(ValueError, match="could not convert string to float"):
        Vector._to_db(stringified, dim=len(FAKE_EMBEDDING))
