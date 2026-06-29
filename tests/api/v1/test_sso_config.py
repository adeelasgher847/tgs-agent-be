import pytest
import uuid
from app.models.sso_config import SsoConfig
from app.models.tenant import Tenant

def test_sso_config_table_name():
    # Verify the table name defaults to 'ssoconfig' per Base class auto-naming convention
    assert SsoConfig.__tablename__ == "ssoconfig"

def test_upsert_saml_config(db):
    # Fetch seed tenant
    tenant = db.query(Tenant).first()
    assert tenant is not None

    # Clean up existing SSO config if any
    existing = db.query(SsoConfig).filter_by(workspace_id=tenant.id).first()
    if existing:
        db.delete(existing)
        db.commit()

    # Create new SsoConfig
    sso_config = SsoConfig(
        workspace_id=tenant.id,
        protocol="saml",
        idp_entity_id="https://example.com/saml2",
        idp_sso_url="https://example.com/sso",
        idp_x509_certificate="PEM_DATA",
        is_active=True
    )
    db.add(sso_config)
    db.commit()
    db.refresh(sso_config)

    # Query back and assert
    fetched = db.query(SsoConfig).filter_by(workspace_id=tenant.id).first()
    assert fetched is not None
    assert fetched.protocol == "saml"
    assert fetched.idp_entity_id == "https://example.com/saml2"
    assert fetched.is_active is True

    # Clean up
    db.delete(fetched)
    db.commit()
