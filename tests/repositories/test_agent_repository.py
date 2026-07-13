"""AgentRepository integration tests against in-memory SQLite."""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from app.models.tenant import Tenant
from app.repositories.agent_repository import AgentRepository


@pytest.fixture(scope="module", autouse=True)
def _drop_partial_unique_indexes(db):
    """SQLite ignores partial indexes — drop tenant-only unique constraints for tests."""
    for index_name in (
        "uq_agent_single_inbound_per_tenant",
        "uq_agent_single_follow_up_per_tenant",
    ):
        try:
            db.execute(text(f"DROP INDEX IF EXISTS {index_name}"))
        except Exception:  # noqa: BLE001
            pass
    db.commit()
    yield


@pytest.fixture
def tenant(db) -> Tenant:
    t = Tenant(
        name=f"RepoWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"repo_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.mark.usefixtures("db")
def test_create_find_by_id_and_soft_delete(db, tenant: Tenant):
    repo = AgentRepository(db)
    created = repo.create(
        {
            "tenant_id": tenant.id,
            "name": "Repo Agent",
            "status": "active",
            "llm_model": "gpt-4o-mini",
            "tts_provider_slug": "11labs",
            "tts_voice_external_id": "vX",
            "tts_language": "en",
        }
    )
    assert created.id is not None

    loaded = repo.find_by_id(created.id)
    assert loaded is not None
    assert loaded.name == "Repo Agent"

    repo.soft_delete(loaded)
    assert repo.find_by_id(created.id) is None


@pytest.mark.usefixtures("db")
def test_find_by_workspace_paginates(db, tenant: Tenant):
    repo = AgentRepository(db)
    for i in range(3):
        repo.create(
            {
                "tenant_id": tenant.id,
                "name": f"Paged {i}",
                "status": "active",
                "llm_model": "gpt-4o-mini",
                "tts_provider_slug": "11labs",
                "tts_voice_external_id": "v",
                "tts_language": "en",
            }
        )

    rows, total = repo.find_by_workspace(tenant.id, page=1, limit=2)
    assert total == 3
    assert len(rows) == 2
