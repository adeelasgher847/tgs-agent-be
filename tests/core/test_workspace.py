"""Tests for Workspace request context."""
from __future__ import annotations

import uuid

from app.core.workspace import Workspace


def test_from_mapping_and_cache_roundtrip():
    data = {
        "id": str(uuid.uuid4()),
        "name": "ACME",
        "schema_name": "acme_schema",
        "status": "active",
        "credits": 42.5,
        "stripe_customer_id": None,
        "stripe_subscription_id": None,
    }
    ws = Workspace.from_mapping(data)
    assert ws.name == "ACME"
    assert ws.is_active is True

    restored = Workspace.from_mapping(ws.to_cache_dict())
    assert restored == ws
