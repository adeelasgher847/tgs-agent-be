"""
SSO callback integration tests against real PostgreSQL.

Requires TEST_DATABASE_URL. Verifies that SSO user provisioning correctly
upserts User rows, creates user_tenant_association entries, and enforces
JSONB allowed_email_domains domain restrictions — behaviours that SQLite's
weak type system (no native JSONB, no CHECK constraints) cannot verify.

The SAML and OIDC HTTP-level tests mock only the IdP-specific cryptography
(OneLogin_Saml2_Auth, OIDC token exchange) while allowing all database
operations to execute against real PostgreSQL.

Coverage:
  1.  test_new_user_created_on_first_sso_login
  2.  test_existing_user_returned_on_repeat_sso_login
  3.  test_user_linked_to_workspace_after_sso
  4.  test_allowed_email_domains_blocks_disallowed_domain_in_postgres
  5.  test_allowed_email_domains_permits_matching_domain_in_postgres
  6.  test_empty_allowed_domains_list_permits_any_email
  7.  test_null_allowed_domains_permits_any_email
  8.  test_workspace_fk_constraint_on_sso_config
  9.  test_saml_callback_upserts_user_in_postgres
 10.  test_saml_callback_missing_sso_config_returns_404
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from app.models.role import Role
from app.models.sso_config import SsoConfig
from app.models.tenant import Tenant
from app.models.user import User, user_tenant_association
from app.services.api_key_service import create_api_key
from app.services.sso_service import find_or_create_user
from tests.conftest import _INTEGRATION_SKIP

pytestmark = [_INTEGRATION_SKIP, pytest.mark.integration]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tenant(pg_session, *, slug: str | None = None) -> Tenant:
    tenant = Tenant(
        name=f"SSO-{uuid.uuid4().hex[:8]}",
        schema_name=f"sso_test_{uuid.uuid4().hex[:6]}",
        status="active",
        workspace_slug=slug or f"sso-slug-{uuid.uuid4().hex[:8]}",
    )
    pg_session.add(tenant)
    pg_session.commit()
    pg_session.refresh(tenant)
    return tenant


def _make_sso_config(
    pg_session,
    workspace_id: uuid.UUID,
    *,
    allowed_domains: list[str] | None = None,
    is_active: bool = True,
) -> SsoConfig:
    config = SsoConfig(
        workspace_id=workspace_id,
        protocol="saml",
        is_active=is_active,
        idp_entity_id="https://idp.example.com/entity",
        idp_sso_url="https://idp.example.com/sso",
        idp_x509_certificate="MIIC...",
        allowed_email_domains=allowed_domains,
    )
    pg_session.add(config)
    pg_session.commit()
    pg_session.refresh(config)
    return config


def _ensure_roles(pg_session) -> None:
    """Seed the role rows find_or_create_user requires (read_only at minimum)."""
    for name in ("admin", "user", "read_only"):
        exists = pg_session.query(Role).filter(Role.name == name).first()
        if not exists:
            pg_session.add(Role(name=name))
    pg_session.commit()


def _user_tenant_link_exists(pg_session, user_id: uuid.UUID, tenant_id: uuid.UUID) -> bool:
    row = pg_session.execute(
        text(
            "SELECT 1 FROM user_tenant_association "
            "WHERE user_id = :uid AND tenant_id = :tid"
        ),
        {"uid": str(user_id), "tid": str(tenant_id)},
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Service-layer tests — call find_or_create_user directly against PG
# ---------------------------------------------------------------------------


class TestSsoFindOrCreateUserPostgres:
    """Verifies user-upsert logic executes correctly against real PostgreSQL."""

    def test_new_user_created_on_first_sso_login(self, pg_session):
        _ensure_roles(pg_session)
        tenant = _make_tenant(pg_session)
        _make_sso_config(pg_session, tenant.id, allowed_domains=None)

        email = f"newuser-{uuid.uuid4().hex[:8]}@example.com"
        user, role_info = find_or_create_user(pg_session, email, tenant.id)

        assert user.id is not None
        assert user.email == email
        assert user.first_name == "SSO"
        assert user.last_name == "User"
        # Password hash must be set (random unusable), not empty
        assert user.hashed_password and len(user.hashed_password) > 20

    def test_existing_user_returned_on_repeat_sso_login(self, pg_session):
        _ensure_roles(pg_session)
        tenant = _make_tenant(pg_session)
        _make_sso_config(pg_session, tenant.id)

        email = f"existing-{uuid.uuid4().hex[:8]}@example.com"
        user_first, _ = find_or_create_user(pg_session, email, tenant.id)
        user_second, _ = find_or_create_user(pg_session, email, tenant.id)

        # Same row, no duplicate
        assert user_first.id == user_second.id

        count = (
            pg_session.query(User)
            .filter(User.email == email)
            .count()
        )
        assert count == 1

    def test_user_linked_to_workspace_after_sso(self, pg_session):
        _ensure_roles(pg_session)
        tenant = _make_tenant(pg_session)
        _make_sso_config(pg_session, tenant.id)

        email = f"linked-{uuid.uuid4().hex[:8]}@example.com"
        user, _ = find_or_create_user(pg_session, email, tenant.id)

        assert _user_tenant_link_exists(pg_session, user.id, tenant.id)

    def test_allowed_email_domains_blocks_disallowed_domain_in_postgres(
        self, pg_session
    ):
        """JSONB allowed_email_domains enforcement persisted in and read from real PG."""
        _ensure_roles(pg_session)
        tenant = _make_tenant(pg_session)
        _make_sso_config(
            pg_session, tenant.id, allowed_domains=["acme.com", "acme.org"]
        )

        with pytest.raises(HTTPException) as exc_info:
            find_or_create_user(pg_session, f"hacker-{uuid.uuid4().hex[:4]}@gmail.com", tenant.id)

        assert exc_info.value.status_code == 403
        assert "not permitted" in exc_info.value.detail

    def test_allowed_email_domains_permits_matching_domain_in_postgres(
        self, pg_session
    ):
        _ensure_roles(pg_session)
        tenant = _make_tenant(pg_session)
        _make_sso_config(
            pg_session, tenant.id, allowed_domains=["acme.com"]
        )

        user, _ = find_or_create_user(
            pg_session, f"valid-{uuid.uuid4().hex[:6]}@acme.com", tenant.id
        )
        assert user.email.endswith("@acme.com")

    def test_empty_allowed_domains_list_permits_any_email(self, pg_session):
        """Empty list [] means "no restriction" — any domain should be allowed."""
        _ensure_roles(pg_session)
        tenant = _make_tenant(pg_session)
        _make_sso_config(pg_session, tenant.id, allowed_domains=[])

        user, _ = find_or_create_user(
            pg_session, f"anyone-{uuid.uuid4().hex[:6]}@random.io", tenant.id
        )
        assert user.id is not None

    def test_null_allowed_domains_permits_any_email(self, pg_session):
        """NULL allowed_email_domains means "no restriction"."""
        _ensure_roles(pg_session)
        tenant = _make_tenant(pg_session)
        _make_sso_config(pg_session, tenant.id, allowed_domains=None)

        user, _ = find_or_create_user(
            pg_session, f"wildcard-{uuid.uuid4().hex[:6]}@whatever.net", tenant.id
        )
        assert user.id is not None


# ---------------------------------------------------------------------------
# Model/constraint tests — verify PG enforces FK and CHECK constraints
# ---------------------------------------------------------------------------


class TestSsoConfigDbConstraints:
    """PG-specific constraint tests on the sso_config table."""

    def test_workspace_fk_constraint_on_sso_config(self, pg_session):
        """workspace_id must reference an existing tenant row."""
        ghost_workspace_id = uuid.uuid4()
        bad_config = SsoConfig(
            workspace_id=ghost_workspace_id,
            protocol="saml",
            is_active=False,
        )
        pg_session.add(bad_config)
        with pytest.raises(Exception):
            pg_session.flush()
        pg_session.rollback()

    def test_sso_config_unique_per_workspace(self, pg_session):
        """workspace_id has a unique constraint on sso_config."""
        tenant = _make_tenant(pg_session)
        pg_session.add(
            SsoConfig(workspace_id=tenant.id, protocol="saml", is_active=False)
        )
        pg_session.commit()

        pg_session.add(
            SsoConfig(workspace_id=tenant.id, protocol="oidc", is_active=False)
        )
        with pytest.raises(Exception):
            pg_session.flush()
        pg_session.rollback()

    def test_protocol_check_constraint_rejects_unknown_value(self, pg_session):
        """protocol column has CHECK IN ('saml', 'oidc')."""
        tenant = _make_tenant(pg_session)
        bad = SsoConfig(
            workspace_id=tenant.id,
            protocol="oauth1",  # not in ('saml', 'oidc')
            is_active=False,
        )
        pg_session.add(bad)
        with pytest.raises(Exception):
            pg_session.flush()
        pg_session.rollback()


# ---------------------------------------------------------------------------
# HTTP-level tests — exercise the SAML callback route end-to-end
# Mocks only the IdP-specific SAML processing; all DB operations hit real PG.
# ---------------------------------------------------------------------------


class TestSamlCallbackHttp:
    """
    HTTP-level SAML callback tests using pg_client.

    The route uses app.db.async_session.get_db for its database dependency.
    We patch only the SAML library auth object so that all user-upsert and
    token-issue logic runs against real PostgreSQL.
    """

    def test_saml_callback_missing_sso_config_returns_404(
        self, pg_client, pg_session
    ):
        """Workspace with no active SSO config returns 404 before SAML processing."""
        tenant = _make_tenant(pg_session, slug=f"no-sso-{uuid.uuid4().hex[:8]}")
        slug = tenant.workspace_slug

        relay = "relay_state_value_abc123"
        resp = pg_client.post(
            f"/auth/saml/{slug}/callback",
            data={"RelayState": relay, "SAMLResponse": "gibberish"},
            cookies={"saml_state": relay},
        )
        # No SsoConfig in PG → 404
        assert resp.status_code == 404

    def test_saml_callback_invalid_relay_state_returns_400(
        self, pg_client, pg_session
    ):
        """CSRF check: RelayState in form must match saml_state cookie."""
        tenant = _make_tenant(pg_session, slug=f"csrf-sso-{uuid.uuid4().hex[:8]}")
        _make_sso_config(pg_session, tenant.id)
        slug = tenant.workspace_slug

        resp = pg_client.post(
            f"/auth/saml/{slug}/callback",
            data={"RelayState": "form_state", "SAMLResponse": "any"},
            cookies={"saml_state": "different_state"},
        )
        assert resp.status_code == 400

    def test_saml_callback_upserts_user_in_postgres(
        self, pg_client, pg_session
    ):
        """With SAML auth mocked to succeed, the user is upserted into real PG."""
        _ensure_roles(pg_session)
        tenant = _make_tenant(pg_session, slug=f"saml-sso-{uuid.uuid4().hex[:8]}")
        _make_sso_config(pg_session, tenant.id)
        slug = tenant.workspace_slug

        test_email = f"saml-user-{uuid.uuid4().hex[:8]}@example.com"
        relay = "relay_state_test_value_xyz"

        mock_auth = MagicMock()
        mock_auth.is_authenticated.return_value = True
        mock_auth.get_errors.return_value = []
        mock_auth.get_nameid.return_value = test_email

        with patch("app.routers.sso_auth.OneLogin_Saml2_Auth", return_value=mock_auth):
            resp = pg_client.post(
                f"/auth/saml/{slug}/callback",
                data={"RelayState": relay, "SAMLResponse": "mocked_saml_response"},
                cookies={"saml_state": relay},
                follow_redirects=False,
            )

        # Successful SSO issues tokens and redirects to the dashboard
        assert resp.status_code in (200, 302, 307), resp.text

        # Verify the user row was upserted in PG
        created_user = (
            pg_session.query(User)
            .filter(User.email == test_email)
            .first()
        )
        assert created_user is not None, "User should be persisted in PG after SSO callback"
        assert _user_tenant_link_exists(pg_session, created_user.id, tenant.id)
