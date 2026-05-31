"""Unit tests for workspace API key service helpers."""
from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from app.models.api_key import Apikey
from app.models.tenant import Tenant
from app.services.api_key_service import (
    RAW_KEY_PREFIX,
    create_api_key,
    generate_raw_api_key,
    get_api_key_for_tenant,
    list_api_keys,
    mask_api_key,
    revoke_api_key,
    sha256_hex,
    to_api_key_out,
)


@pytest.fixture
def tenant(db):
    suffix = uuid.uuid4().hex[:8]
    t = Tenant(name=f"API Key Corp {suffix}", schema_name=f"apikey_{suffix}", status="active")
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


class TestKeyHelpers:
    def test_generate_raw_key_has_prefix(self):
        raw = generate_raw_api_key()
        assert raw.startswith(RAW_KEY_PREFIX)
        assert len(raw) > len(RAW_KEY_PREFIX) + 20

    def test_mask_api_key_format(self):
        raw = "sk_abcdefghijklmnop"
        masked = mask_api_key(raw)
        assert masked.startswith(raw[:8])
        assert masked.endswith(raw[-4:])
        assert "••••" in masked

    def test_sha256_is_deterministic(self):
        assert sha256_hex("abc") == sha256_hex("abc")
        assert sha256_hex("abc") != sha256_hex("xyz")


@pytest.mark.usefixtures("db")
class TestApiKeyCrud:
    def test_create_stores_hash_not_raw(self, db, tenant):
        record, raw = create_api_key(db, tenant_id=tenant.id, name="CI")
        assert raw.startswith(RAW_KEY_PREFIX)
        assert record.key_hash == sha256_hex(raw)
        assert record.key_hash != raw
        assert record.key_prefix == mask_api_key(raw)
        assert record.is_active is True

    def test_list_scoped_to_tenant(self, db, tenant):
        s2 = uuid.uuid4().hex[:8]
        other = Tenant(name=f"Other-{s2}", schema_name=f"other_{s2}", status="active")
        db.add(other)
        db.commit()

        create_api_key(db, tenant_id=tenant.id, name="A")
        create_api_key(db, tenant_id=other.id, name="B")

        keys = list_api_keys(db, tenant_id=tenant.id)
        assert len(keys) == 1
        assert keys[0].name == "A"

    def test_revoke_sets_inactive(self, db, tenant):
        record, _ = create_api_key(db, tenant_id=tenant.id, name="Revoke me")
        revoked = revoke_api_key(db, record)
        assert revoked.is_active is False

    def test_revoke_twice_raises_400(self, db, tenant):
        record, _ = create_api_key(db, tenant_id=tenant.id, name="Twice")
        revoke_api_key(db, record)
        with pytest.raises(HTTPException) as exc:
            revoke_api_key(db, record)
        assert exc.value.status_code == 400

    def test_get_api_key_for_tenant_filters_workspace(self, db, tenant):
        record, _ = create_api_key(db, tenant_id=tenant.id, name="Mine")
        wrong_tenant = uuid.uuid4()
        assert get_api_key_for_tenant(db, key_id=record.id, tenant_id=wrong_tenant) is None
        found = get_api_key_for_tenant(db, key_id=record.id, tenant_id=tenant.id)
        assert found is not None
        assert found.id == record.id

    def test_to_api_key_out_shape(self, db, tenant):
        record, _ = create_api_key(db, tenant_id=tenant.id, name="Out")
        out = to_api_key_out(record)
        assert out["workspace_id"] == tenant.id
        assert out["masked_key"] == record.key_prefix
        assert "raw_key" not in out
