"""
Regression: log_audit_event() silently failed (caught, logged at WARNING,
rolled back) whenever old_value/new_value contained a datetime, UUID, or
other non-JSON-primitive value — e.g. a raw Pydantic .model_dump() dict, as
every call site in app/routers/call_flows.py passes. AuditLog.old_value/
new_value are plain `JSON` columns with no custom serializer, so
`TypeError: Object of type datetime is not JSON serializable` was raised at
flush time and swallowed, silently dropping the audit trail entry while the
underlying business operation (e.g. demo link creation) still succeeded.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

from app.services.audit_service import log_audit_event


def _fake_request() -> MagicMock:
    request = MagicMock()
    request.headers = {"x-forwarded-for": "", "user-agent": "pytest"}
    request.client = MagicMock(host="127.0.0.1")
    request.state = MagicMock(api_key_prefix=None)
    return request


def test_log_audit_event_with_datetime_in_new_value_does_not_fail():
    db = MagicMock()
    request = _fake_request()

    log_audit_event(
        db,
        request=request,
        tenant_id=uuid.uuid4(),
        action="call_flow.demo_link_created",
        resource_type="call_flow_demo_link",
        resource_id=uuid.uuid4(),
        new_value={
            "name": "demo",
            "expires_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        },
    )

    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.rollback.assert_not_called()

    entry = db.add.call_args[0][0]
    assert entry.new_value["name"] == "demo"
    assert entry.new_value["expires_at"] == "2026-01-01 00:00:00+00:00"


def test_log_audit_event_with_uuid_and_decimal_in_old_value_does_not_fail():
    db = MagicMock()
    request = _fake_request()

    some_id = uuid.uuid4()
    log_audit_event(
        db,
        request=request,
        tenant_id=uuid.uuid4(),
        action="call_flow.demo_link_updated",
        resource_type="call_flow_demo_link",
        old_value={"linked_id": some_id, "budget": Decimal("12.50")},
    )

    db.add.assert_called_once()
    db.commit.assert_called_once()
    db.rollback.assert_not_called()

    entry = db.add.call_args[0][0]
    assert entry.old_value["linked_id"] == str(some_id)
    assert entry.old_value["budget"] == "12.50"


def test_log_audit_event_with_none_values_is_a_noop_passthrough():
    db = MagicMock()
    request = _fake_request()

    log_audit_event(
        db,
        request=request,
        tenant_id=uuid.uuid4(),
        action="call_flow.demo_link_deleted",
        resource_type="call_flow_demo_link",
    )

    entry = db.add.call_args[0][0]
    assert entry.old_value is None
    assert entry.new_value is None
