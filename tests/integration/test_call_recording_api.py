"""
Integration-style test for GET /api/v1/recordings/{call_id}.

Uses mocked GCS signing (no live bucket required). Skips when explicitly
running only live GCS tests without mocks — see RUN_GCS_RECORDING_INTEGRATION.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.models.agent import Agent
from app.models.call_session import CallSession
from app.models.phone_number import NumberConfiguration, PhoneNumber
from app.models.tenant import Tenant

_SIGNED_URL_RE = re.compile(
    r"^https://storage\.googleapis\.com/.+\?X-Goog-.*",
    re.IGNORECASE,
)


@pytest.mark.integration
def test_get_recording_returns_signed_url_format(client: TestClient, db):
    """Ticket: create call with recording path → GET → confirm signed URL format."""
    if os.environ.get("RUN_GCS_RECORDING_INTEGRATION", "").lower() not in ("1", "true", "yes"):
        pytest.skip("Set RUN_GCS_RECORDING_INTEGRATION=1 for live GCS signing tests")

    tenant = Tenant(name="Rec Int Tenant", schema_name="rec_int", status="active")
    db.add(tenant)
    db.commit()
    db.refresh(tenant)

    agent = Agent(name="Rec Agent", tenant_id=tenant.id, status="ready")
    db.add(agent)
    db.commit()
    db.refresh(agent)

    pn = PhoneNumber(
        phone_number="+61290000001",
        tenant_id=tenant.id,
        assistant_id=agent.id,
        status="active",
    )
    db.add(pn)
    db.commit()
    db.refresh(pn)

    db.add(NumberConfiguration(phone_number_id=pn.id, recording_enabled=True))
    db.commit()

    session = CallSession(
        user_id=uuid.uuid4(),
        agent_id=agent.id,
        tenant_id=tenant.id,
        start_time=datetime.now(timezone.utc),
        status="completed",
        call_type="outbound",
        duration=60,
        assistant_phone_number=pn.phone_number,
        recording_gcs_path=f"recordings/{tenant.id}/{uuid.uuid4()}/20260609.opus",
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    mock_url = (
        "https://storage.googleapis.com/my-bucket/recordings/x.opus"
        "?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Signature=abc"
    )
    with (
        patch(
            "app.services.gcs_recording_service.generate_signed_url",
            return_value=mock_url,
        ),
        patch("app.services.gcs_recording_service.get_object_size", return_value=1024),
        patch(
            "app.middleware.api_key_middleware._resolve_api_key",
            return_value={
                "api_key_id": str(uuid.uuid4()),
                "tenant_id": str(tenant.id),
                "key_is_active": True,
                "workspace": {
                    "id": str(tenant.id),
                    "name": tenant.name,
                    "schema_name": tenant.schema_name,
                    "status": "active",
                },
            },
        ),
    ):
        resp = client.get(
            f"/api/v1/recordings/{session.id}",
            headers={
                "X-API-Key": "integration-test-key",
                "X-Workspace-ID": str(tenant.id),
            },
        )

    assert resp.status_code == 200, resp.text
    url = resp.json()["data"]["url"]
    assert _SIGNED_URL_RE.match(url), url

    db.delete(session)
    db.query(NumberConfiguration).filter(NumberConfiguration.phone_number_id == pn.id).delete()
    db.delete(pn)
    db.delete(agent)
    db.delete(tenant)
    db.commit()
