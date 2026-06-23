"""Unit tests for the Redis-backed RBAC role cache.

No real Redis is required in CI — a tiny in-memory fake stands in for the
sync redis client (mirrors the dict-and-TTL surface area we actually use:
get/set/delete). Falling open to a direct DB read when Redis is truly
unavailable is covered separately (get_redis_sync() returns None there).
"""
from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from app.services import rbac_cache_service
from app.models.role import Role
from app.models.tenant import Tenant
from app.models.user import User, user_tenant_association


class _FakeRedis:
    """Minimal get/set/delete fake — no TTL expiry simulation needed for these tests."""

    def __init__(self):
        self.store: dict[str, str] = {}
        self.calls: list[str] = []

    def get(self, key: str):
        self.calls.append(f"get:{key}")
        return self.store.get(key)

    def set(self, key: str, value: str, ex: int | None = None):
        self.calls.append(f"set:{key}")
        self.store[key] = value

    def delete(self, key: str):
        self.calls.append(f"delete:{key}")
        self.store.pop(key, None)


@pytest.fixture
def fake_redis():
    return _FakeRedis()


@pytest.fixture
def tenant(db) -> Tenant:
    t = Tenant(
        name=f"cache-{uuid.uuid4().hex[:8]}",
        schema_name=f"s_{uuid.uuid4().hex[:8]}",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def member(db, tenant) -> User:
    role = db.query(Role).filter(Role.name == "config_only").first()
    if role is None:
        role = Role(name="config_only", description="config_only")
        db.add(role)
        db.commit()
        db.refresh(role)

    user = User(
        email=f"member-{uuid.uuid4().hex[:8]}@example.com",
        first_name="Member",
        last_name="User",
        hashed_password="x",
        current_tenant_id=tenant.id,
    )
    db.add(user)
    db.flush()
    db.execute(
        user_tenant_association.insert().values(
            user_id=user.id, tenant_id=tenant.id, role_id=role.id, is_creator=False
        )
    )
    db.commit()
    db.refresh(user)
    return user


def test_cache_miss_then_hit(db, tenant, member, fake_redis):
    with patch("app.services.rbac_cache_service.get_redis_sync", return_value=fake_redis):
        first = rbac_cache_service.get_effective_role(db, member.id, tenant.id)
        assert first == "config_only"
        assert any(c.startswith("set:") for c in fake_redis.calls)

        fake_redis.calls.clear()
        second = rbac_cache_service.get_effective_role(db, member.id, tenant.id)
        assert second == "config_only"
        # Served from cache — no second "set" (no DB fallback re-populating it)
        assert not any(c.startswith("set:") for c in fake_redis.calls)


def test_cache_key_format(db, tenant, member, fake_redis):
    with patch("app.services.rbac_cache_service.get_redis_sync", return_value=fake_redis):
        rbac_cache_service.get_effective_role(db, member.id, tenant.id)
    assert f"rbac:{member.id}:{tenant.id}" in fake_redis.store


def test_invalidate_clears_cached_value(db, tenant, member, fake_redis):
    with patch("app.services.rbac_cache_service.get_redis_sync", return_value=fake_redis):
        rbac_cache_service.get_effective_role(db, member.id, tenant.id)
        assert fake_redis.store

        rbac_cache_service.invalidate(member.id, tenant.id)
        assert not fake_redis.store


def test_invalidate_then_reread_reflects_new_role(db, tenant, member, fake_redis):
    with patch("app.services.rbac_cache_service.get_redis_sync", return_value=fake_redis):
        assert rbac_cache_service.get_effective_role(db, member.id, tenant.id) == "config_only"

        # Simulate a role change happening underneath the cached entry.
        new_role = db.query(Role).filter(Role.name == "manager").first()
        if not new_role:
            new_role = Role(name="manager", description="manager")
            db.add(new_role)
            db.commit()
        db.execute(
            user_tenant_association.update()
            .where(
                user_tenant_association.c.user_id == member.id,
                user_tenant_association.c.tenant_id == tenant.id,
            )
            .values(role_id=new_role.id)
        )
        db.commit()

        # Still cached as the stale value until invalidated.
        assert rbac_cache_service.get_effective_role(db, member.id, tenant.id) == "config_only"

        rbac_cache_service.invalidate(member.id, tenant.id)
        assert rbac_cache_service.get_effective_role(db, member.id, tenant.id) == "manager"


def test_not_a_member_caches_sentinel_not_none(db, tenant, fake_redis):
    stranger_id = uuid.uuid4()
    with patch("app.services.rbac_cache_service.get_redis_sync", return_value=fake_redis):
        result = rbac_cache_service.get_effective_role(db, stranger_id, tenant.id)
    assert result is None
    # Confirms a real cache miss (None) is distinguishable from a cached
    # "not a member" sentinel on the next read — exercised by re-reading
    # with calls cleared and no DB activity required to get the same answer.
    fake_redis.calls.clear()
    with patch("app.services.rbac_cache_service.get_redis_sync", return_value=fake_redis):
        result2 = rbac_cache_service.get_effective_role(db, stranger_id, tenant.id)
    assert result2 is None
    assert not any(c.startswith("set:") for c in fake_redis.calls)


def test_redis_unavailable_falls_back_to_db(db, tenant, member):
    with patch("app.services.rbac_cache_service.get_redis_sync", return_value=None):
        result = rbac_cache_service.get_effective_role(db, member.id, tenant.id)
    assert result == "config_only"


def test_redis_errors_fail_open_to_db(db, tenant, member):
    class _ExplodingRedis:
        def get(self, key):
            raise ConnectionError("redis unreachable")

        def set(self, *a, **kw):
            raise ConnectionError("redis unreachable")

        def delete(self, *a, **kw):
            raise ConnectionError("redis unreachable")

    with patch("app.services.rbac_cache_service.get_redis_sync", return_value=_ExplodingRedis()):
        result = rbac_cache_service.get_effective_role(db, member.id, tenant.id)
        assert result == "config_only"
        rbac_cache_service.invalidate(member.id, tenant.id)  # must not raise
