import pytest
from pydantic import ValidationError
from app.schemas.sso import SsoConfigUpsert

def test_sso_config_oidc_discovery_url_ssrf():
    # Attempting to upsert OIDC SSO config with private discovery URL should fail validation
    payload = {
        "protocol": "oidc",
        "oidc_client_id": "test-client-id",
        "oidc_client_secret": "test-secret",
        "oidc_discovery_url": "http://127.0.0.1/discovery",
        "is_active": True,
        "allowed_email_domains": ["acme.com"]
    }
    
    with pytest.raises(ValidationError) as exc_info:
        SsoConfigUpsert(**payload)
        
    assert "resolves to a blocked address" in str(exc_info.value)
