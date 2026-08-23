"""Tests for the tenant billing-dashboard endpoints in app/api/api_v1/endpoints/tenant.py:

- POST /tenants/billing/portal-session
- POST /tenants/plan/cancel
- GET  /tenants/billing/invoices

Follows the direct-function-call pattern used by
tests/api/test_tenant_credit_checkout.py rather than spinning up a TestClient,
since these endpoints have no route-level behavior beyond their dependencies.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.api_v1.endpoints.tenant import (
    cancel_tenant_plan,
    create_billing_portal_session,
    get_auto_recharge_settings,
    get_tenant_invoices,
    upsert_auto_recharge_settings,
)
from app.models.plan import Plan
from app.models.subscription import Subscription
from app.models.tenant import Tenant
from app.models.user import User
from app.models.wallet_auto_recharge_config import WalletAutoRechargeConfig
from app.schemas.tenant import AutoRechargeSettingsUpsert


def _fake_request() -> MagicMock:
    request = MagicMock()
    request.headers = {"x-forwarded-for": "", "user-agent": "pytest"}
    request.client = MagicMock(host="127.0.0.1")
    request.state = MagicMock(api_key_prefix=None)
    return request


@pytest.fixture
def tenant(db):
    suffix = uuid.uuid4().hex[:8]
    t = Tenant(
        id=uuid.uuid4(),
        name=f"Billing Dashboard Tenant {suffix}",
        schema_name=f"billing_dash_{suffix}",
        status="active",
        credits=Decimal("0"),
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@pytest.fixture
def user(db, tenant):
    suffix = uuid.uuid4().hex[:8]
    u = User(
        id=uuid.uuid4(),
        email=f"billing_dash_{suffix}@example.com",
        first_name="Test",
        last_name="User",
        hashed_password="x",
        current_tenant_id=tenant.id,
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@pytest.fixture
def plan(db):
    suffix = uuid.uuid4().hex[:8]
    p = Plan(
        id=uuid.uuid4(),
        name=f"studio_{suffix}",
        display_name="Studio",
        crm_type=None,
        included_minutes=100,
        monthly_credits=Decimal("50"),
        price_monthly=9900,
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


@pytest.fixture
def active_subscription(db, tenant, user, plan):
    sub = Subscription(
        id=uuid.uuid4(),
        user_id=user.id,
        plan_id=plan.id,
        tenant_id=tenant.id,
        stripe_subscription_id="sub_test_123",
        stripe_customer_id="cus_test_123",
        crm_type=None,
        status="active",
        cancel_at_period_end=False,
    )
    db.add(sub)
    db.commit()
    db.refresh(sub)
    return sub


# ---------------------------------------------------------------------------
# POST /tenants/billing/portal-session
# ---------------------------------------------------------------------------


class TestCreateBillingPortalSession:
    def test_lazily_creates_customer_and_returns_url(self, db, tenant, user):
        assert tenant.stripe_customer_id is None

        with patch(
            "app.api.api_v1.endpoints.tenant.StripeService.create_customer",
            return_value="cus_new_123",
        ) as mock_create_customer, patch(
            "app.api.api_v1.endpoints.tenant.StripeService.create_portal_session",
            return_value={"url": "https://billing.stripe.test/session_abc"},
        ) as mock_portal:
            result = create_billing_portal_session(
                current_user=user,
                admin_user=user,
                db=db,
            )

        mock_create_customer.assert_called_once()
        mock_portal.assert_called_once()
        assert mock_portal.call_args[0][0] == "cus_new_123"
        assert result.data.url == "https://billing.stripe.test/session_abc"

        db.refresh(tenant)
        assert tenant.stripe_customer_id == "cus_new_123"

    def test_reuses_existing_customer_id(self, db, tenant, user):
        tenant.stripe_customer_id = "cus_existing_456"
        db.commit()

        with patch(
            "app.api.api_v1.endpoints.tenant.StripeService.create_customer"
        ) as mock_create_customer, patch(
            "app.api.api_v1.endpoints.tenant.StripeService.create_portal_session",
            return_value={"url": "https://billing.stripe.test/session_def"},
        ) as mock_portal:
            result = create_billing_portal_session(
                current_user=user,
                admin_user=user,
                db=db,
            )

        mock_create_customer.assert_not_called()
        mock_portal.assert_called_once()
        assert mock_portal.call_args[0][0] == "cus_existing_456"
        assert result.data.url == "https://billing.stripe.test/session_def"

    def test_stripe_failure_returns_502(self, db, tenant, user):
        tenant.stripe_customer_id = "cus_existing_456"
        db.commit()

        with patch(
            "app.api.api_v1.endpoints.tenant.StripeService.create_portal_session",
            side_effect=Exception("stripe down"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                create_billing_portal_session(
                    current_user=user,
                    admin_user=user,
                    db=db,
                )
        assert exc_info.value.status_code == 502


# ---------------------------------------------------------------------------
# POST /tenants/plan/cancel
# ---------------------------------------------------------------------------


class TestCancelTenantPlan:
    def test_cancels_active_subscription_and_audit_logs(
        self, db, tenant, user, active_subscription
    ):
        request = _fake_request()

        with patch(
            "app.api.api_v1.endpoints.tenant.StripeService.cancel_subscription",
            return_value={"status": "active"},
        ) as mock_cancel:
            result = cancel_tenant_plan(
                request=request,
                current_user=user,
                admin_user=user,
                db=db,
            )

        mock_cancel.assert_called_once_with("sub_test_123", at_period_end=True)
        assert result.data.cancel_at_period_end is True

        db.refresh(active_subscription)
        assert active_subscription.cancel_at_period_end is True

        from app.models.audit_log import AuditLog

        audit_entry = (
            db.query(AuditLog)
            .filter(AuditLog.action == "billing.plan_canceled")
            .first()
        )
        assert audit_entry is not None
        assert audit_entry.tenant_id == tenant.id
        assert audit_entry.resource_id == active_subscription.id

    def test_no_active_subscription_returns_404(self, db, tenant, user):
        request = _fake_request()

        with pytest.raises(HTTPException) as exc_info:
            cancel_tenant_plan(
                request=request,
                current_user=user,
                admin_user=user,
                db=db,
            )
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# GET /tenants/billing/invoices
# ---------------------------------------------------------------------------


class TestGetTenantInvoices:
    def test_no_stripe_customer_returns_empty_list(self, db, tenant, user):
        assert tenant.stripe_customer_id is None
        result = get_tenant_invoices(current_user=user, billing_user=user, db=db)
        assert result.data.invoices == []

    def test_maps_stripe_invoice_response(self, db, tenant, user):
        tenant.stripe_customer_id = "cus_existing_789"
        db.commit()

        fake_invoice = {
            "id": "in_test_1",
            "number": "INV-0001",
            "description": None,
            "status": "paid",
            "currency": "usd",
            "amount_due": 9900,
            "amount_paid": 9900,
            "created": 1735689600,
            "hosted_invoice_url": "https://invoice.stripe.test/in_test_1",
            "invoice_pdf": "https://invoice.stripe.test/in_test_1.pdf",
            "lines": {"data": [{"description": "Studio plan - Aug"}]},
        }

        with patch(
            "app.api.api_v1.endpoints.tenant.StripeService.get_invoices",
            return_value=[fake_invoice],
        ) as mock_get_invoices:
            result = get_tenant_invoices(current_user=user, billing_user=user, db=db)

        mock_get_invoices.assert_called_once_with("cus_existing_789", limit=10)
        assert len(result.data.invoices) == 1
        invoice = result.data.invoices[0]
        assert invoice.id == "in_test_1"
        assert invoice.number == "INV-0001"
        assert invoice.description == "Studio plan - Aug"
        assert invoice.status == "paid"
        assert invoice.amount_due == Decimal("99")
        assert invoice.amount_paid == Decimal("99")
        assert invoice.hosted_invoice_url == "https://invoice.stripe.test/in_test_1"


# ---------------------------------------------------------------------------
# GET/PUT /tenants/billing/auto-recharge
# ---------------------------------------------------------------------------


class TestAutoRechargeSettings:
    def test_get_returns_defaults_when_no_config_exists(self, db, tenant, user):
        result = get_auto_recharge_settings(current_user=user, billing_user=user, db=db)
        assert result.data.enabled is False
        assert result.data.min_balance == Decimal("10.00")
        assert result.data.recharge_amount == Decimal("20.00")
        assert result.data.stripe_payment_method_id is None
        assert result.data.last_triggered_at is None

    def test_get_returns_existing_config(self, db, tenant, user):
        config = WalletAutoRechargeConfig(
            workspace_id=tenant.id,
            enabled=True,
            min_balance=Decimal("15.00"),
            recharge_amount=Decimal("30.00"),
            stripe_payment_method_id="pm_existing_123",
        )
        db.add(config)
        db.commit()

        result = get_auto_recharge_settings(current_user=user, billing_user=user, db=db)
        assert result.data.enabled is True
        assert result.data.min_balance == Decimal("15.00")
        assert result.data.recharge_amount == Decimal("30.00")
        assert result.data.stripe_payment_method_id == "pm_existing_123"

    def test_put_rejects_recharge_amount_not_positive(self):
        with pytest.raises(Exception):
            AutoRechargeSettingsUpsert(
                enabled=True,
                min_balance=Decimal("10"),
                recharge_amount=Decimal("0"),
                stripe_payment_method_id=None,
            )

    def test_put_rejects_negative_min_balance(self):
        with pytest.raises(Exception):
            AutoRechargeSettingsUpsert(
                enabled=True,
                min_balance=Decimal("-1"),
                recharge_amount=Decimal("10"),
                stripe_payment_method_id=None,
            )

    def test_put_rejects_payment_method_not_belonging_to_tenant(self, db, tenant, user):
        tenant.stripe_customer_id = "cus_owned_by_tenant"
        db.commit()

        payload = AutoRechargeSettingsUpsert(
            enabled=True,
            min_balance=Decimal("10"),
            recharge_amount=Decimal("20"),
            stripe_payment_method_id="pm_belongs_to_someone_else",
        )

        with patch(
            "app.api.api_v1.endpoints.tenant.StripeService.get_customer_payment_methods",
            return_value=[{"id": "pm_actually_owned_123"}],
        ):
            with pytest.raises(HTTPException) as exc_info:
                upsert_auto_recharge_settings(
                    payload=payload, current_user=user, admin_user=user, db=db
                )
        assert exc_info.value.status_code == 400

    def test_put_creates_config_with_valid_payment_method(self, db, tenant, user):
        tenant.stripe_customer_id = "cus_owned_by_tenant"
        db.commit()

        payload = AutoRechargeSettingsUpsert(
            enabled=True,
            min_balance=Decimal("10"),
            recharge_amount=Decimal("20"),
            stripe_payment_method_id="pm_actually_owned_123",
        )

        with patch(
            "app.api.api_v1.endpoints.tenant.StripeService.get_customer_payment_methods",
            return_value=[{"id": "pm_actually_owned_123"}],
        ):
            result = upsert_auto_recharge_settings(
                payload=payload, current_user=user, admin_user=user, db=db
            )

        assert result.data.enabled is True
        assert result.data.stripe_payment_method_id == "pm_actually_owned_123"

        config = (
            db.query(WalletAutoRechargeConfig)
            .filter(WalletAutoRechargeConfig.workspace_id == tenant.id)
            .first()
        )
        assert config is not None
        assert config.min_balance == Decimal("10")
        assert config.recharge_amount == Decimal("20")

    def test_put_updates_existing_config(self, db, tenant, user):
        tenant.stripe_customer_id = "cus_owned_by_tenant"
        config = WalletAutoRechargeConfig(
            workspace_id=tenant.id,
            enabled=False,
            min_balance=Decimal("10.00"),
            recharge_amount=Decimal("20.00"),
        )
        db.add(config)
        db.commit()

        # enabled=True now requires a payment method (see
        # AutoRechargeSettingsUpsert._validate_recharge_config).
        payload = AutoRechargeSettingsUpsert(
            enabled=True,
            min_balance=Decimal("25"),
            recharge_amount=Decimal("50"),
            stripe_payment_method_id="pm_actually_owned_123",
        )
        with patch(
            "app.api.api_v1.endpoints.tenant.StripeService.get_customer_payment_methods",
            return_value=[{"id": "pm_actually_owned_123"}],
        ):
            result = upsert_auto_recharge_settings(
                payload=payload, current_user=user, admin_user=user, db=db
            )
        assert result.data.enabled is True
        assert result.data.min_balance == Decimal("25")
        assert result.data.recharge_amount == Decimal("50")

        db.refresh(config)
        assert config.enabled is True
        assert config.min_balance == Decimal("25")

    def test_put_rejects_enabled_without_payment_method(self):
        with pytest.raises(Exception):
            AutoRechargeSettingsUpsert(
                enabled=True,
                min_balance=Decimal("10"),
                recharge_amount=Decimal("20"),
                stripe_payment_method_id=None,
            )

    def test_put_rejects_recharge_amount_too_small_relative_to_min_balance(self):
        with pytest.raises(Exception):
            AutoRechargeSettingsUpsert(
                enabled=True,
                min_balance=Decimal("100"),
                recharge_amount=Decimal("10"),  # < 0.5 * min_balance
                stripe_payment_method_id="pm_test_123",
            )

    def test_put_allows_recharge_amount_at_exactly_half_min_balance(self):
        # Boundary: recharge_amount == 0.5 * min_balance must be allowed, not rejected.
        payload = AutoRechargeSettingsUpsert(
            enabled=True,
            min_balance=Decimal("100"),
            recharge_amount=Decimal("50"),
            stripe_payment_method_id="pm_test_123",
        )
        assert payload.recharge_amount == Decimal("50")

    def test_put_allows_disabled_without_payment_method(self):
        # enabled=False must never require a payment method, regardless of ratio.
        payload = AutoRechargeSettingsUpsert(
            enabled=False,
            min_balance=Decimal("100"),
            recharge_amount=Decimal("1"),
            stripe_payment_method_id=None,
        )
        assert payload.enabled is False
