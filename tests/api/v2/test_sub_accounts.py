import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api.deps import get_db, require_admin, get_current_workspace
from app.core.exception_handlers import register_exception_handlers
from app.models.plan import Plan
from app.models.subscription import Subscription
from app.models.tenant import Tenant
from app.models.user import User

@pytest.fixture
def agency_workspace(db) -> Tenant:
    t = Tenant(
        name=f"agency-{uuid.uuid4().hex[:8]}",
        schema_name=f"s_{uuid.uuid4().hex[:8]}",
        workspace_type="agency",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t

@pytest.fixture
def standalone_workspace(db) -> Tenant:
    t = Tenant(
        name=f"standalone-{uuid.uuid4().hex[:8]}",
        schema_name=f"s_{uuid.uuid4().hex[:8]}",
        workspace_type="standalone",
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t

def _client(db, workspace: Tenant, admin_user: User = None, override_get_admin=True) -> TestClient:
    from app.api.v2.routers.workspace import v2_router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(v2_router, prefix="/workspace")

    if override_get_admin:
        # Create a real DB user tied to this workspace to reflect true DB state
        if admin_user is None:
            admin_user = User(email=f"admin_{uuid.uuid4().hex[:8]}@test.com", current_tenant_id=workspace.id, first_name="A", last_name="B", hashed_password="X")
            db.add(admin_user)
            db.commit()
            db.refresh(admin_user)
            admin_user.tenants.append(workspace)
            db.commit()
            
        mini.dependency_overrides[require_admin] = lambda: admin_user
        async def mock_get_current_workspace():
            return workspace
        mini.dependency_overrides[get_current_workspace] = mock_get_current_workspace
    elif admin_user:
        mini.dependency_overrides[require_admin] = lambda: admin_user

    mini.dependency_overrides[get_db] = lambda: db
    return TestClient(mini, raise_server_exceptions=False)

def test_create_sub_account_success(db, agency_workspace):
    client = _client(db, agency_workspace)
    payload = {
        "name": "Test Sub Account",
        "contact_email": "sub@example.com"
    }

    res = client.post("/workspace/sub-accounts", json=payload)
    assert res.status_code == 201
    data = res.json()
    assert data["name"] == "Test Sub Account"
    assert data["contact_email"] == "sub@example.com"
    assert "api_key" in data

def test_create_sub_account_not_agency(db, standalone_workspace):
    client = _client(db, standalone_workspace)
    res = client.post("/workspace/sub-accounts", json={"name": "Sub", "contact_email": "x@x.com"})
    assert res.status_code == 403
    assert "agency workspaces" in res.json()["error"]["message"]

def test_sub_accounts_crud(db, agency_workspace):
    client = _client(db, agency_workspace)

    # 1. Create
    res = client.post("/workspace/sub-accounts", json={"name": "Sub 1", "contact_email": "x@x.com"})
    assert res.status_code == 201
    sub_id = res.json()["id"]

    # 2. List
    res = client.get("/workspace/sub-accounts")
    assert res.status_code == 200
    assert len(res.json()["data"]) >= 1

    # 3. Get
    res = client.get(f"/workspace/sub-accounts/{sub_id}")
    assert res.status_code == 200
    assert res.json()["id"] == sub_id

    # 4. Update
    res = client.put(f"/workspace/sub-accounts/{sub_id}", json={"name": "Sub 1 Updated"})
    assert res.status_code == 200
    assert res.json()["name"] == "Sub 1 Updated"

    # 5. Delete
    res = client.delete(f"/workspace/sub-accounts/{sub_id}")
    assert res.status_code == 204

def test_cross_workspace_isolation(db, agency_workspace):
    # Setup Agency B
    agency_b = Tenant(
        name=f"agency-b-{uuid.uuid4().hex[:8]}",
        schema_name=f"s_{uuid.uuid4().hex[:8]}",
        workspace_type="agency",
        status="active",
    )
    db.add(agency_b)
    db.commit()
    db.refresh(agency_b)
    
    # Sub account of Agency A
    sub_a = Tenant(
        name=f"sub-a-{uuid.uuid4().hex[:8]}",
        schema_name=f"s_{uuid.uuid4().hex[:8]}",
        workspace_type="standalone",
        parent_workspace_id=agency_workspace.id,
        status="active",
    )
    db.add(sub_a)
    db.commit()
    db.refresh(sub_a)

    # Client acting as Agency B
    client_b = _client(db, agency_b)
    
    # Try to access Sub A using Agency B's client context
    res = client_b.get(f"/workspace/sub-accounts/{sub_a.id}")
    assert res.status_code == 404
    # Our exception handler wraps errors under {"error": {"message": ...}}
    assert "Sub-account not found" in res.json()["error"]["message"]

def test_rbac_enforcement(db, agency_workspace):
    from app.api.v2.routers.workspace import v2_router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(v2_router, prefix="/workspace")

    async def mock_require_admin_fail():
        raise HTTPException(status_code=403, detail="Workspace context does not match user tenant.")

    mini.dependency_overrides[require_admin] = mock_require_admin_fail
    mini.dependency_overrides[get_db] = lambda: db
    client = TestClient(mini, raise_server_exceptions=False)

    res = client.post("/workspace/sub-accounts", json={"name": "Sub", "contact_email": "x@x.com"})
    assert res.status_code == 403
    assert "Workspace context does not match" in res.json()["error"]["message"]

def _activate_plan(db, tenant, *, max_subaccounts):
    suffix = uuid.uuid4().hex[:8]
    plan = Plan(
        name=f"plan_{suffix}",
        display_name="Plan",
        price_monthly=9900,
        crm_type=None,
        max_subaccounts=max_subaccounts,
        is_active=True,
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)

    now = datetime.now(timezone.utc)
    admin_user = User(
        email=f"owner_{suffix}@test.com",
        current_tenant_id=tenant.id,
        first_name="Owner",
        last_name="Test",
        hashed_password="X",
    )
    db.add(admin_user)
    db.commit()
    db.refresh(admin_user)

    sub = Subscription(
        user_id=admin_user.id,
        tenant_id=tenant.id,
        plan_id=plan.id,
        crm_type=None,
        status="active",
        current_period_start=now - timedelta(days=1),
        current_period_end=now + timedelta(days=29),
    )
    db.add(sub)
    db.commit()
    return plan


def test_create_sub_account_studio_tier_rejects_over_cap(db, agency_workspace):
    _activate_plan(db, agency_workspace, max_subaccounts=3)
    client = _client(db, agency_workspace)
    suffix = uuid.uuid4().hex[:8]

    for i in range(3):
        res = client.post(
            "/workspace/sub-accounts",
            json={"name": f"Sub {suffix}-{i}", "contact_email": f"sub{suffix}{i}@example.com"},
        )
        assert res.status_code == 201

    # 4th sub-account exceeds the Studio-tier cap of 3.
    res = client.post(
        "/workspace/sub-accounts",
        json={"name": f"Sub {suffix}-4", "contact_email": f"sub{suffix}4@example.com"},
    )
    assert res.status_code == 422
    assert "Sub-account limit reached" in res.json()["error"]["message"]


def test_create_sub_account_agency_tier_no_cap(db, agency_workspace):
    _activate_plan(db, agency_workspace, max_subaccounts=None)
    client = _client(db, agency_workspace)
    suffix = uuid.uuid4().hex[:8]

    for i in range(4):
        res = client.post(
            "/workspace/sub-accounts",
            json={"name": f"Sub {suffix}-{i}", "contact_email": f"sub{suffix}{i}@example.com"},
        )
        assert res.status_code == 201


def _owner_client(db, workspace: Tenant, owner_user: User) -> TestClient:
    """Client for the owner-only wallet-sharing endpoints — mirrors
    tests/api/v2/test_workspace_usage_breakdown.py's convention of exercising
    require_workspace_owner for real against the DB rather than overriding it."""
    from app.api.deps import require_user_tenant
    from app.api.v2.routers.workspace import v2_router

    mini = FastAPI()
    register_exception_handlers(mini)
    mini.include_router(v2_router, prefix="/workspace")
    mini.dependency_overrides[require_user_tenant] = lambda: owner_user
    mini.dependency_overrides[get_db] = lambda: db
    return TestClient(mini, raise_server_exceptions=False)


def _make_owner(db, tenant_id) -> User:
    from app.models.user import user_tenant_association

    suffix = uuid.uuid4().hex[:8]
    owner = User(
        email=f"owner-{suffix}@test.com",
        current_tenant_id=tenant_id,
        first_name="Owner",
        last_name="Test",
        hashed_password="X",
    )
    db.add(owner)
    db.flush()
    db.execute(
        user_tenant_association.insert().values(
            user_id=owner.id, tenant_id=tenant_id, role_id=None, is_creator=True,
        )
    )
    db.commit()
    db.refresh(owner)
    return owner


def test_wallet_sharing_toggle_updates_sub_account(db, agency_workspace):
    owner = _make_owner(db, agency_workspace.id)
    sub = Tenant(
        name=f"sub-{uuid.uuid4().hex[:8]}",
        schema_name=f"s_{uuid.uuid4().hex[:8]}",
        workspace_type="sub_account",
        parent_workspace_id=agency_workspace.id,
        status="active",
    )
    db.add(sub)
    db.commit()
    db.refresh(sub)

    client = _owner_client(db, agency_workspace, owner)
    res = client.put(
        f"/workspace/sub-accounts/{sub.id}/wallet-sharing",
        json={"using_master_wallet": True},
    )
    assert res.status_code == 200
    assert res.json()["using_master_wallet"] is True

    db.refresh(sub)
    assert sub.uses_master_wallet is True


def test_wallet_sharing_toggle_rejects_non_family_sub_account(db, agency_workspace):
    """A sub_account_id that isn't actually the caller's own sub-account must
    be rejected, not silently trusted."""
    owner = _make_owner(db, agency_workspace.id)

    other_agency = Tenant(
        name=f"agency-b-{uuid.uuid4().hex[:8]}",
        schema_name=f"s_{uuid.uuid4().hex[:8]}",
        workspace_type="agency",
        status="active",
    )
    db.add(other_agency)
    db.commit()
    db.refresh(other_agency)

    foreign_sub = Tenant(
        name=f"sub-{uuid.uuid4().hex[:8]}",
        schema_name=f"s_{uuid.uuid4().hex[:8]}",
        workspace_type="sub_account",
        parent_workspace_id=other_agency.id,
        status="active",
        # Explicit — SQLite's handling of the Boolean column's textual
        # server_default("false") is unreliable in this test harness, unlike
        # real Postgres.
        uses_master_wallet=False,
    )
    db.add(foreign_sub)
    db.commit()
    db.refresh(foreign_sub)

    client = _owner_client(db, agency_workspace, owner)
    res = client.put(
        f"/workspace/sub-accounts/{foreign_sub.id}/wallet-sharing",
        json={"using_master_wallet": True},
    )
    assert res.status_code == 404

    db.refresh(foreign_sub)
    assert foreign_sub.uses_master_wallet is False


def test_auto_link_new_workspaces_toggle(db, agency_workspace):
    owner = _make_owner(db, agency_workspace.id)
    client = _owner_client(db, agency_workspace, owner)

    res = client.put(
        "/workspace/auto-link-new-workspaces",
        json={"auto_link_new_workspaces": True},
    )
    assert res.status_code == 200
    assert res.json()["auto_link_new_workspaces"] is True

    db.refresh(agency_workspace)
    assert agency_workspace.auto_link_new_workspaces is True


def test_linked_workspaces_lists_parent_and_sub_accounts(db, agency_workspace):
    owner = _make_owner(db, agency_workspace.id)
    sub = Tenant(
        name=f"sub-{uuid.uuid4().hex[:8]}",
        schema_name=f"s_{uuid.uuid4().hex[:8]}",
        workspace_type="sub_account",
        parent_workspace_id=agency_workspace.id,
        status="active",
        uses_master_wallet=True,
    )
    db.add(sub)
    db.commit()
    db.refresh(sub)

    client = _owner_client(db, agency_workspace, owner)
    res = client.get("/workspace/linked-workspaces")
    assert res.status_code == 200
    body = res.json()
    ids = {row["id"]: row for row in body["workspaces"]}
    assert str(agency_workspace.id) in ids
    assert ids[str(agency_workspace.id)]["is_master"] is True
    assert str(sub.id) in ids
    assert ids[str(sub.id)]["using_master_wallet"] is True
    assert ids[str(sub.id)]["is_master"] is False


def test_auto_link_new_workspaces_true_defaults_new_sub_account_to_wallet_sharing_on(
    db, agency_workspace,
):
    agency_workspace.auto_link_new_workspaces = True
    db.commit()

    client = _client(db, agency_workspace)
    res = client.post(
        "/workspace/sub-accounts",
        json={"name": "Auto Linked Sub", "contact_email": "autolink@example.com"},
    )
    assert res.status_code == 201
    sub_id = res.json()["id"]

    sub = db.query(Tenant).filter(Tenant.id == uuid.UUID(sub_id)).first()
    assert sub.uses_master_wallet is True


def test_auto_link_new_workspaces_false_leaves_new_sub_account_own_wallet(
    db, agency_workspace,
):
    # Explicit — SQLite's handling of the Boolean column's textual
    # server_default("false") is unreliable in this test harness, unlike
    # real Postgres.
    agency_workspace.auto_link_new_workspaces = False
    db.commit()

    client = _client(db, agency_workspace)
    res = client.post(
        "/workspace/sub-accounts",
        json={"name": "Non Linked Sub", "contact_email": "nolink@example.com"},
    )
    assert res.status_code == 201
    sub_id = res.json()["id"]

    sub = db.query(Tenant).filter(Tenant.id == uuid.UUID(sub_id)).first()
    assert sub.uses_master_wallet is False


def test_create_member_role_post_alias(db, agency_workspace):
    user = User(email=f"test{uuid.uuid4().hex[:8]}@x.com", current_tenant_id=agency_workspace.id, first_name="A", last_name="B", hashed_password="X")
    db.add(user)
    db.commit()
    db.refresh(user)

    client = _client(db, agency_workspace, admin_user=user, override_get_admin=False)
    
    with patch("app.api.v2.routers.workspace.update_member_role") as mock_update:
        mock_update.return_value = {"role": "manager", "user_id": str(user.id), "workspace_id": str(agency_workspace.id)}
        res = client.post(f"/workspace/members/{user.id}/role", json={"role": "manager"})
        
    assert res.status_code == 200
    assert res.json()["role"] == "manager"
