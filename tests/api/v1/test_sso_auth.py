import pytest
from fastapi import HTTPException
from app.services.sso_service import find_or_create_user
from app.models.sso_config import SsoConfig
from app.models.tenant import Tenant

def test_find_or_create_user_allowed_email_domains(db):
    import uuid
    # Setup a dummy tenant
    tenant = Tenant(
        name="Acme Test Workspace",
        workspace_slug="acme-test-sso",
        schema_name=f"test_schema_{uuid.uuid4().hex[:8]}",
        status="active"
    )
    db.add(tenant)
    db.commit()
    db.refresh(tenant)

    # Setup SsoConfig with allowed_email_domains restriction
    sso_config = SsoConfig(
        workspace_id=tenant.id,
        protocol="oidc",
        is_active=True,
        allowed_email_domains=["acme.com", "acme.org"]
    )
    db.add(sso_config)
    db.commit()

    # Login with valid email domain acme.com should succeed
    user, role_info = find_or_create_user(db, "john@acme.com", tenant.id)
    assert user.email == "john@acme.com"

    # Login with invalid email domain gmail.com should fail with 403 HTTPException
    with pytest.raises(HTTPException) as exc_info:
        find_or_create_user(db, "hacker@gmail.com", tenant.id)
    assert exc_info.value.status_code == 403
    assert "not permitted for this workspace" in exc_info.value.detail

    # Clean up
    db.delete(sso_config)
    db.delete(user)
    db.delete(tenant)
    db.commit()
