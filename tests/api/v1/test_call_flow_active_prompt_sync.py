"""Unit tests for Call Flow active prompt sync with Agent.system_prompt and Live Voice prompt resolution."""

from __future__ import annotations

import uuid

import pytest

from app.models.agent import Agent
from app.models.tenant import Tenant
from app.schemas.call_flow import CallFlowCreate, CallFlowUpdate
from app.services.call_flow_service import call_flow_service


@pytest.fixture
def tenant(db) -> Tenant:
    t = Tenant(
        name=f"SyncWS-{uuid.uuid4().hex[:8]}",
        schema_name=f"sync_ws_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


def _create_test_agent(db, tenant_id: uuid.UUID) -> Agent:
    agent = Agent(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        name="Test Prompt Sync Agent",
        llm_model="gemini-1.5-flash",
        tts_provider_slug="rime",
        tts_voice_external_id="voice-1",
        tts_language="en",
        status="active",
        system_prompt="Initial Agent Prompt",
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)
    return agent


def test_create_flow_syncs_agent_system_prompt(db, tenant):
    agent = _create_test_agent(db, tenant.id)

    body = CallFlowCreate(
        name="Sync Flow 1",
        direction="inbound",
        agentId=agent.id,
        prompt="First Flow Prompt",
        notes="V1 Notes",
    )

    flow_out = call_flow_service.create_flow(db, tenant.id, body)

    db.refresh(agent)
    assert agent.system_prompt == "First Flow Prompt"
    assert flow_out["currentPromptId"] is not None


def test_update_flow_new_prompt_syncs_agent_system_prompt(db, tenant):
    agent = _create_test_agent(db, tenant.id)

    create_body = CallFlowCreate(
        name="Sync Flow 2",
        direction="inbound",
        agentId=agent.id,
        prompt="Initial Prompt",
        notes="Version 1",
    )
    flow_out = call_flow_service.create_flow(db, tenant.id, create_body)
    flow_id = uuid.UUID(flow_out["id"])

    # Update flow with new prompt text
    update_body = CallFlowUpdate(
        prompt="Updated Prompt Text",
        notes="Version 2",
    )
    updated_out = call_flow_service.update_flow(db, flow_id, tenant.id, update_body)

    db.refresh(agent)
    assert agent.system_prompt == "Updated Prompt Text"
    assert updated_out["currentPromptId"] != flow_out["currentPromptId"]


def test_update_flow_rollback_syncs_agent_system_prompt(db, tenant):
    agent = _create_test_agent(db, tenant.id)

    # Create initial version
    flow_out_1 = call_flow_service.create_flow(
        db,
        tenant.id,
        CallFlowCreate(
            name="Sync Flow 3",
            direction="inbound",
            agentId=agent.id,
            prompt="Prompt V1",
            notes="V1",
        ),
    )
    flow_id = uuid.UUID(flow_out_1["id"])
    v1_id = uuid.UUID(flow_out_1["currentPromptId"])

    # Create version 2
    call_flow_service.update_flow(
        db,
        flow_id,
        tenant.id,
        CallFlowUpdate(prompt="Prompt V2", notes="V2"),
    )

    db.refresh(agent)
    assert agent.system_prompt == "Prompt V2"

    # Roll back to V1
    flow_out_3 = call_flow_service.update_flow(
        db,
        flow_id,
        tenant.id,
        CallFlowUpdate(currentPromptId=v1_id),
    )

    db.refresh(agent)
    assert agent.system_prompt == "Prompt V1"
    assert flow_out_3["currentPromptId"] == str(v1_id)
