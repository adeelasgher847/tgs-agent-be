"""
Regression tests for RagService.retrieve() kb_id scoping.

Bug: the Visual Flow Editor's kb_lookup node let a flow author configure a
specific kb_id, but retrieve() had no way to filter by it -- every kb_lookup
node silently searched across *all* KBs bound to the agent's tenant. These
tests guard:

  1. When kb_id is supplied, the query filters on kb.id in addition to the
     existing kb.workspace_id (tenant) filter -- tenant scoping is never
     bypassed by kb_id alone.
  2. When kb_id is omitted (default), behavior is unchanged (unscoped across
     the tenant's KBs) -- backward compatible for non-flow RAG callers such
     as app/voice/rag_context.py and app/routers/knowledge_base.py.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from app.services.rag_service import RagService


FAKE_EMBEDDING = [0.1, 0.2, 0.3]


def _embedding_func(text: str):
    return FAKE_EMBEDDING


def _make_fake_db(rows=None):
    db = MagicMock()
    result = MagicMock()
    result.fetchall.return_value = rows or []
    db.execute.return_value = result
    return db


def test_retrieve_with_kb_id_filters_query_by_kb_and_still_scopes_tenant():
    service = RagService()
    db = _make_fake_db()
    tenant_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    kb_id = uuid.uuid4()

    service.retrieve(
        user_text="What is the returns policy?",
        tenant_id=tenant_id,
        agent_id=agent_id,
        embedding_func=_embedding_func,
        db_session=db,
        kb_id=kb_id,
    )

    assert db.execute.call_count == 1
    stmt, params = db.execute.call_args[0]
    sql_text = str(stmt)

    # kb_id filter present in the query.
    assert "kb.id = :kb_id" in sql_text
    assert params["kb_id"] == str(kb_id)

    # Tenant (workspace) filter is still applied unconditionally.
    assert "kb.workspace_id = :workspace_id" in sql_text
    assert params["workspace_id"] == str(tenant_id)


def test_retrieve_without_kb_id_is_unscoped_across_tenant_kbs():
    service = RagService()
    db = _make_fake_db()
    tenant_id = uuid.uuid4()
    agent_id = uuid.uuid4()

    service.retrieve(
        user_text="What is the pricing?",
        tenant_id=tenant_id,
        agent_id=agent_id,
        embedding_func=_embedding_func,
        db_session=db,
    )

    assert db.execute.call_count == 1
    stmt, params = db.execute.call_args[0]
    sql_text = str(stmt)

    assert "kb.id = :kb_id" not in sql_text
    assert "kb_id" not in params
    assert "kb.workspace_id = :workspace_id" in sql_text
    assert params["workspace_id"] == str(tenant_id)


def test_retrieve_with_kb_id_as_string_is_accepted():
    """kb_id may arrive as a string (e.g. from flow node config JSON)."""
    service = RagService()
    db = _make_fake_db()
    tenant_id = uuid.uuid4()
    kb_id = uuid.uuid4()

    service.retrieve(
        user_text="Question",
        tenant_id=tenant_id,
        agent_id=None,
        embedding_func=_embedding_func,
        db_session=db,
        kb_id=str(kb_id),
    )

    _, params = db.execute.call_args[0]
    assert params["kb_id"] == str(kb_id)
