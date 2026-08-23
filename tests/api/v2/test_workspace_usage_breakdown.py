"""Agency billing-dashboard reporting endpoints:

  GET /api/v2/workspace/usage/breakdown[.csv]
  GET /api/v2/workspace/usage/recent-activity
  GET /api/v2/workspace/usage/minutes-by-month[.csv]

Coverage:
  - Owner (is_creator=True on the parent/master tenant) sees the full
    breakdown across parent + direct sub-accounts.
  - A non-owner admin on the same parent tenant gets 403.
  - A user on a sub-account — even that sub-account's own creator — gets
    403 when hitting these endpoints from the sub-account's own tenant
    context.
  - Growth calculation returns None (not a divide-by-zero) when last month
    had $0 spend.
  - A standalone workspace with no sub-accounts returns a valid single-row
    response.
  - CSV endpoints return text/csv with the same numbers as the JSON
    endpoints.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.api.deps import get_db, require_user_tenant, require_workspace_owner
from app.core.exception_handlers import register_exception_handlers
from app.models.tenant import Tenant
from app.models.usage_record import UsageRecord
from app.models.user import User, user_tenant_association


def _make_tenant(db, *, workspace_type="standalone", parent_workspace_id=None, created_at=None) -> Tenant:
    suffix = uuid.uuid4().hex[:8]
    t = Tenant(
        id=uuid.uuid4(),
        name=f"Tenant {suffix}",
        schema_name=f"t_{suffix}",
        workspace_type=workspace_type,
        parent_workspace_id=parent_workspace_id,
        status="active",
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    if created_at is not None:
        db.query(Tenant).filter(Tenant.id == t.id).update({"created_at": created_at})
        db.commit()
        db.refresh(t)
    return t


def _make_member(db, tenant_id, *, is_creator: bool) -> User:
    suffix = uuid.uuid4().hex[:8]
    user = User(
        email=f"user-{suffix}@test.com",
        first_name="Test",
        last_name="User",
        hashed_password="x",
        current_tenant_id=tenant_id,
    )
    db.add(user)
    db.flush()
    db.execute(
        user_tenant_association.insert().values(
            user_id=user.id, tenant_id=tenant_id, role_id=None, is_creator=is_creator,
        )
    )
    db.commit()
    db.refresh(user)
    return user


def _usage(db, workspace_id, *, recorded_at, credits, minutes=Decimal("10"), call_id=None):
    rec = UsageRecord(
        id=uuid.uuid4(),
        workspace_id=workspace_id,
        call_id=call_id,
        billable_minutes=minutes,
        credits_charged=credits,
        recorded_at=recorded_at,
    )
    db.add(rec)
    db.commit()
    return rec


def _client(db, user: User) -> TestClient:
    from app.api.v2.routers.workspace import v2_router

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(v2_router, prefix="/workspace")

    # Only bypass the outer JWT resolution — require_workspace_owner itself
    # runs for real against the DB, so authorization behavior is genuinely
    # exercised rather than mocked away.
    app.dependency_overrides[require_user_tenant] = lambda: user
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def now():
    return datetime.now(timezone.utc)


@pytest.fixture
def parent(db, now):
    return _make_tenant(db, workspace_type="agency", created_at=now - timedelta(days=200))


@pytest.fixture
def sub_account(db, parent, now):
    return _make_tenant(
        db, workspace_type="sub_account", parent_workspace_id=parent.id, created_at=now - timedelta(days=40)
    )


@pytest.fixture
def owner(db, parent):
    return _make_member(db, parent.id, is_creator=True)


@pytest.fixture
def non_owner_admin(db, parent):
    return _make_member(db, parent.id, is_creator=False)


# ─────────────────────────────────────────────────────── authorization ──


class TestAuthorization:
    def test_non_owner_admin_gets_403(self, db, parent, sub_account, non_owner_admin):
        client = _client(db, non_owner_admin)
        for path in (
            "/workspace/usage/breakdown",
            "/workspace/usage/recent-activity",
            "/workspace/usage/minutes-by-month",
        ):
            res = client.get(path)
            assert res.status_code == 403, path

    def test_sub_account_creator_gets_403_from_sub_account_context(self, db, parent, sub_account, now):
        # This user IS the creator of the sub-account itself — is_creator=True
        # for sub_account.id — but must still be rejected because they're not
        # the master/agency workspace's owner.
        sub_creator = _make_member(db, sub_account.id, is_creator=True)
        client = _client(db, sub_creator)
        for path in (
            "/workspace/usage/breakdown",
            "/workspace/usage/recent-activity",
            "/workspace/usage/minutes-by-month",
        ):
            res = client.get(path)
            assert res.status_code == 403, path

    def test_not_a_member_gets_403(self, db, parent):
        stray = User(
            email=f"stray-{uuid.uuid4().hex[:8]}@test.com",
            first_name="S",
            last_name="T",
            hashed_password="x",
            current_tenant_id=parent.id,
        )
        db.add(stray)
        db.commit()
        db.refresh(stray)
        client = _client(db, stray)
        res = client.get("/workspace/usage/breakdown")
        assert res.status_code == 403

    def test_require_workspace_owner_unit(self, db, parent, owner, non_owner_admin):
        assert require_workspace_owner(user=owner, db=db) is owner
        with pytest.raises(HTTPException) as exc:
            require_workspace_owner(user=non_owner_admin, db=db)
        assert exc.value.status_code == 403


# ─────────────────────────────────────────────────────────── breakdown ──


class TestBreakdown:
    def test_owner_sees_full_breakdown_across_parent_and_sub_accounts(
        self, db, parent, sub_account, owner, now
    ):
        this_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        _usage(db, parent.id, recorded_at=this_month_start + timedelta(days=1), credits=Decimal("10.00"))
        _usage(db, sub_account.id, recorded_at=this_month_start + timedelta(days=1), credits=Decimal("5.00"))

        client = _client(db, owner)
        res = client.get("/workspace/usage/breakdown")
        assert res.status_code == 200
        data = res.json()

        assert data["workspace_count"] == 2
        names_to_master = {r["workspace_id"]: r["is_master"] for r in data["rows"]}
        assert names_to_master[str(parent.id)] is True
        assert names_to_master[str(sub_account.id)] is False

        this_month_by_id = {r["workspace_id"]: Decimal(str(r["this_month"])) for r in data["rows"]}
        assert this_month_by_id[str(parent.id)] == Decimal("10.00")
        assert this_month_by_id[str(sub_account.id)] == Decimal("5.00")

        assert Decimal(str(data["this_month_total"])) == Decimal("15.00")
        assert Decimal(str(data["totals"]["this_month"])) == Decimal("15.00")

    def test_growth_is_null_when_no_prior_month_spend(self, db, parent, owner, now):
        this_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        _usage(db, parent.id, recorded_at=this_month_start + timedelta(days=1), credits=Decimal("20.00"))

        client = _client(db, owner)
        res = client.get("/workspace/usage/breakdown")
        assert res.status_code == 200
        data = res.json()
        row = next(r for r in data["rows"] if r["workspace_id"] == str(parent.id))
        assert row["growth_percent"] is None
        assert data["this_month_growth_percent"] is None

    def test_standalone_workspace_with_no_sub_accounts_returns_single_row(self, db, now):
        standalone = _make_tenant(db, workspace_type="standalone", created_at=now - timedelta(days=10))
        owner = _make_member(db, standalone.id, is_creator=True)

        client = _client(db, owner)
        res = client.get("/workspace/usage/breakdown")
        assert res.status_code == 200
        data = res.json()
        assert data["workspace_count"] == 1
        assert len(data["rows"]) == 1
        assert data["rows"][0]["workspace_id"] == str(standalone.id)
        assert data["rows"][0]["is_master"] is True
        assert Decimal(str(data["rows"][0]["this_month"])) == Decimal("0")

    def test_breakdown_csv_matches_json_numbers(self, db, parent, sub_account, owner, now):
        this_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        _usage(db, parent.id, recorded_at=this_month_start + timedelta(days=1), credits=Decimal("10.00"))

        client = _client(db, owner)
        json_res = client.get("/workspace/usage/breakdown")
        csv_res = client.get("/workspace/usage/breakdown.csv")

        assert csv_res.status_code == 200
        assert csv_res.headers["content-type"].startswith("text/csv")
        assert "attachment; filename=" in csv_res.headers["content-disposition"]

        json_total = str(json_res.json()["this_month_total"])
        assert json_total in csv_res.text


# ────────────────────────────────────────────────────────── activity ──


class TestRecentActivity:
    def test_last_ten_across_family_newest_first(self, db, parent, sub_account, owner, now):
        for i in range(12):
            _usage(
                db,
                parent.id if i % 2 == 0 else sub_account.id,
                recorded_at=now - timedelta(hours=i),
                credits=Decimal(f"{i + 1}.00"),
                call_id=None,
            )

        client = _client(db, owner)
        res = client.get("/workspace/usage/recent-activity")
        assert res.status_code == 200
        items = res.json()["items"]
        assert len(items) == 10
        # Newest (i=0, credits=1.00) first.
        assert Decimal(str(items[0]["amount"])) == Decimal("-1.00")
        assert items[0]["type"] == "call_usage"
        assert items[0]["description"].startswith("Call ID - ")


# ────────────────────────────────────────────────────── minutes-by-month ──


class TestMinutesByMonth:
    def test_three_months_aggregated_across_family(self, db, parent, sub_account, owner, now):
        this_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        call_id = None
        _usage(
            db,
            parent.id,
            recorded_at=this_month_start + timedelta(days=1),
            credits=Decimal("3.00"),
            minutes=Decimal("15"),
            call_id=call_id,
        )
        _usage(
            db,
            sub_account.id,
            recorded_at=this_month_start + timedelta(days=1),
            credits=Decimal("2.00"),
            minutes=Decimal("5"),
            call_id=call_id,
        )

        client = _client(db, owner)
        res = client.get("/workspace/usage/minutes-by-month")
        assert res.status_code == 200
        months = res.json()["months"]
        assert len(months) == 3
        current = months[-1]
        assert Decimal(str(current["total_minutes"])) == Decimal("20")
        assert Decimal(str(current["total_cost"])) == Decimal("5.00")

    def test_minutes_csv_matches_json(self, db, parent, owner, now):
        this_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        _usage(db, parent.id, recorded_at=this_month_start + timedelta(days=1), credits=Decimal("4.00"), minutes=Decimal("8"))

        client = _client(db, owner)
        json_res = client.get("/workspace/usage/minutes-by-month")
        csv_res = client.get("/workspace/usage/minutes-by-month.csv")

        assert csv_res.status_code == 200
        assert csv_res.headers["content-type"].startswith("text/csv")
        current_month_label = json_res.json()["months"][-1]["label"]
        assert current_month_label in csv_res.text
