"""
Real-Postgres integration tests for GDPR right-to-erasure.

Requires TEST_DATABASE_URL. The shared pg_engine schema (tests/conftest.py)
is built via Base.metadata.create_all, not Alembic, so the no_update_audit
append-only trigger from migration 20260618_audit_update_bypass does not
exist there automatically — _install_audit_trigger below applies the exact
same DDL once per session so this module can verify the real behaviour:
default updates are silently blocked, and the GDPR bypass GUC anonymizes
only the two actor columns regardless of what is requested.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

from app.models.audit_log import AuditLog
from app.models.batch_call_record import BatchCallRecord
from app.models.batch_job import BatchJob
from app.models.call_session import CallSession
from app.models.agent import Agent
from app.models.knowledge_base_chunk import KbChunk
from app.models.knowledge_base_document import KnowledgeBase
from app.models.phone_number import PhoneNumber
from app.models.role import Role
from app.models.tenant import Tenant
from app.models.user import User, user_tenant_association
from app.services.account_deletion_service import delete_workspace_account
from tests.conftest import _INTEGRATION_SKIP

pytestmark = [_INTEGRATION_SKIP, pytest.mark.integration]

_TRIGGER_SQL = """
DROP RULE IF EXISTS no_update_audit ON auditlog;

CREATE OR REPLACE FUNCTION _restrict_auditlog_update()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF current_setting('app.bypass_audit_update', true) = 'true' THEN
        NEW := OLD;
        NEW.user_id := NULL;
        NEW.actor_api_key_prefix := '[DELETED]';
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS no_update_audit ON auditlog;

CREATE TRIGGER no_update_audit
BEFORE UPDATE ON auditlog
FOR EACH ROW EXECUTE FUNCTION _restrict_auditlog_update();
"""


@pytest.fixture(scope="module", autouse=True)
def _install_audit_trigger(pg_engine):
    """Apply the production migration's DDL directly (pg_engine skips Alembic)."""
    with pg_engine.connect() as conn:
        conn.execute(text(_TRIGGER_SQL))
        conn.commit()


def _make_tenant(pg_session) -> Tenant:
    tenant = Tenant(name=f"GDPR-{uuid.uuid4().hex[:8]}", schema_name="gdpr_test")
    pg_session.add(tenant)
    pg_session.commit()
    pg_session.refresh(tenant)
    return tenant


def _make_user(pg_session, **overrides) -> User:
    user = User(
        email=overrides.get("email", f"{uuid.uuid4().hex[:10]}@example.com"),
        hashed_password="hashed",
        first_name=overrides.get("first_name", "Real"),
        last_name=overrides.get("last_name", "Name"),
        phone=overrides.get("phone", "+15559998888"),
    )
    pg_session.add(user)
    pg_session.commit()
    pg_session.refresh(user)
    return user


def _link_user_tenant(pg_session, user: User, tenant: Tenant) -> None:
    pg_session.execute(
        user_tenant_association.insert().values(user_id=user.id, tenant_id=tenant.id)
    )
    pg_session.commit()


class TestAuditUpdateTrigger:
    """Verifies the production-safety design of the GUC-gated trigger directly."""

    def test_update_without_bypass_is_silently_blocked(self, pg_session):
        tenant = _make_tenant(pg_session)
        row = AuditLog(tenant_id=tenant.id, action="agent.created", resource_type="agent")
        pg_session.add(row)
        pg_session.commit()
        pg_session.refresh(row)

        pg_session.execute(
            text("UPDATE auditlog SET action = 'tampered' WHERE id = :id"),
            {"id": str(row.id)},
        )
        pg_session.commit()
        pg_session.refresh(row)

        assert row.action == "agent.created"

    def test_bypass_forces_fixed_anonymization_regardless_of_requested_values(self, pg_session):
        """Even if the UPDATE asks to change action/old_value, the trigger only
        ever applies the fixed actor anonymization — proving the bypass cannot
        be abused to tamper with anything beyond its intended scope."""
        tenant = _make_tenant(pg_session)
        user_id = uuid.uuid4()
        row = AuditLog(
            tenant_id=tenant.id,
            user_id=user_id,
            actor_api_key_prefix=None,
            action="agent.created",
            resource_type="agent",
        )
        pg_session.add(row)
        pg_session.commit()
        pg_session.refresh(row)

        pg_session.execute(text("SET LOCAL app.bypass_audit_update = 'true'"))
        pg_session.execute(
            text(
                "UPDATE auditlog SET user_id = NULL, actor_api_key_prefix = '[DELETED]', "
                "action = 'tampered' WHERE id = :id"
            ),
            {"id": str(row.id)},
        )
        pg_session.commit()
        pg_session.refresh(row)

        assert row.user_id is None
        assert row.actor_api_key_prefix == "[DELETED]"
        assert row.action == "agent.created"  # untouched despite the SET clause


class TestDeleteWorkspaceAccount:
    def test_wipes_pii_and_soft_deletes_workspace(self, pg_session):
        tenant = _make_tenant(pg_session)
        other_tenant = _make_tenant(pg_session)

        sole_user = _make_user(pg_session, email="sole@example.com")
        _link_user_tenant(pg_session, sole_user, tenant)

        shared_user = _make_user(pg_session, email="shared@example.com")
        _link_user_tenant(pg_session, shared_user, tenant)
        _link_user_tenant(pg_session, shared_user, other_tenant)

        phone = PhoneNumber(tenant_id=tenant.id, phone_number="+15551230000", provider="twilio")
        pg_session.add(phone)

        agent = Agent(tenant_id=tenant.id, name="Agent")
        pg_session.add(agent)
        pg_session.commit()
        pg_session.refresh(agent)

        from datetime import datetime, timezone

        call = CallSession(
            user_id=sole_user.id,
            agent_id=agent.id,
            tenant_id=tenant.id,
            start_time=datetime.now(timezone.utc),
            status="completed",
            from_number="+15550001111",
            to_number="+15550002222",
        )
        pg_session.add(call)

        kb = KnowledgeBase(workspace_id=tenant.id, name="KB")
        pg_session.add(kb)
        pg_session.commit()
        pg_session.refresh(kb)

        chunk = KbChunk(kb_id=kb.id, content="secret content", chunk_metadata={})
        pg_session.add(chunk)

        batch_job = BatchJob(workspace_id=tenant.id, agent_id=agent.id, status="completed")
        pg_session.add(batch_job)
        pg_session.commit()
        pg_session.refresh(batch_job)

        batch_record = BatchCallRecord(batch_job_id=batch_job.id, phone_number="+15559990000", status="completed")
        pg_session.add(batch_record)

        audit_row = AuditLog(
            tenant_id=tenant.id,
            user_id=sole_user.id,
            actor_api_key_prefix=None,
            action="agent.created",
            resource_type="agent",
            resource_id=agent.id,
        )
        pg_session.add(audit_row)
        pg_session.commit()

        # Capture plain ids before the wipe: delete_workspace_account's own
        # db.commit() expires every object this session has loaded so far
        # (default expire_on_commit), so touching ORM attributes on these
        # instances afterwards would either silently re-select (masking what
        # we're trying to prove) or — once detached — raise
        # DetachedInstanceError. Re-querying by plain id sidesteps both.
        tenant_id, other_tenant_id = tenant.id, other_tenant.id
        sole_user_id, shared_user_id = sole_user.id, shared_user.id
        phone_id, call_id, chunk_id, audit_id = phone.id, call.id, chunk.id, audit_row.id

        delete_workspace_account(pg_session, tenant_id)

        from sqlalchemy import select as _select

        def _fresh(model, id_):
            return pg_session.execute(_select(model).where(model.id == id_)).scalar_one_or_none()

        wiped_user = _fresh(User, sole_user_id)
        assert wiped_user.email == "[DELETED@DELETED.COM]"
        assert wiped_user.first_name == "[DELETED]"
        assert wiped_user.last_name == "[DELETED]"
        assert wiped_user.phone is None
        assert wiped_user.deleted_at is not None

        untouched_user = _fresh(User, shared_user_id)
        assert untouched_user.email == "shared@example.com"
        assert untouched_user.deleted_at is None

        remaining_links = pg_session.execute(
            user_tenant_association.select().where(
                user_tenant_association.c.user_id == shared_user_id
            )
        ).all()
        assert len(remaining_links) == 1
        assert remaining_links[0].tenant_id == other_tenant_id

        wiped_phone = _fresh(PhoneNumber, phone_id)
        assert wiped_phone.phone_number == "[REDACTED]"

        wiped_call = _fresh(CallSession, call_id)
        assert wiped_call.from_number == "[REDACTED]"
        assert wiped_call.to_number == "[REDACTED]"

        assert _fresh(KbChunk, chunk_id) is None

        wiped_audit = _fresh(AuditLog, audit_id)
        assert wiped_audit is not None  # legal retention: row stays
        assert wiped_audit.user_id is None
        assert wiped_audit.actor_api_key_prefix == "[DELETED]"
        assert wiped_audit.action == "agent.created"  # only actor fields touched

        wiped_tenant = _fresh(Tenant, tenant_id)
        assert wiped_tenant.deleted_at is not None


class TestAccountDeletionHttp:
    """Exercises the real DELETE /api/v2/workspace/account endpoint with a
    genuine JWT admin session against Postgres."""

    def _admin_token_and_tenant(self, pg_session):
        from app.core.security import create_user_token

        tenant = _make_tenant(pg_session)
        admin_role = pg_session.query(Role).filter(Role.name == "admin").first()
        if admin_role is None:
            admin_role = Role(name="admin")
            pg_session.add(admin_role)
            pg_session.commit()
            pg_session.refresh(admin_role)

        user = _make_user(pg_session, email=f"admin-{uuid.uuid4().hex[:8]}@example.com")
        pg_session.execute(
            user_tenant_association.insert().values(
                user_id=user.id, tenant_id=tenant.id, role_id=admin_role.id
            )
        )
        pg_session.commit()

        token = create_user_token(user_id=user.id, email=user.email, tenant_id=tenant.id, role="admin")
        return token, tenant

    def test_wrong_phrase_returns_400(self, pg_client, pg_session):
        token, tenant = self._admin_token_and_tenant(pg_session)

        resp = pg_client.request(
            "DELETE",
            "/api/v2/workspace/account",
            headers={"Authorization": f"Bearer {token}"},
            json={"confirmation": "nope"},
        )
        assert resp.status_code == 400

        still_there = pg_session.get(Tenant, tenant.id)
        assert still_there.deleted_at is None

    # The 204 success path (real JWT auth -> real wipe) is covered by:
    #   - TestDeleteWorkspaceAccount.test_wipes_pii_and_soft_deletes_workspace
    #     (exercises delete_workspace_account directly against Postgres)
    #   - tests/api/v2/test_account_deletion.py::test_correct_phrase_returns_204_and_wipes
    #     (exercises the route's 204 contract with the service mocked)
    # Not duplicated here: routing a second real JWT request through
    # pg_client's auth middleware in the same process hits a pre-existing
    # lazy-Redis-reinit quirk in tests/conftest.py's pg_auth_middleware
    # fixture (unrelated to this feature) that makes multi-request JWT
    # flows order-dependent in this harness.
