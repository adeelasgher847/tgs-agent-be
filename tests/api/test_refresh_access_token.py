"""
Regression tests for POST /api/v1/users/refresh.

Previously, reusing ``replaced_access_token`` only compared ``role``. If the user
switched ``current_tenant_id`` but kept the same role name in both tenants,
the cached JWT could still carry the old ``tenant_id`` claim.
"""

import uuid

import pytest
from jose import jwt
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.security import get_password_hash
from app.models.tenant import Tenant
from app.models.user import User
from app.services.role_service import assign_role_to_user_tenant

# Must match the hash applied in _ensure_known_password
_TEST_LOGIN_PASSWORD = "testpass"


@pytest.fixture(scope="module", autouse=True)
def _ensure_known_password(db):
    """conftest's bcrypt hash may not match across passlib/bcrypt versions."""
    u = db.query(User).filter(User.email == "test@example.com").first()
    if u is None:
        raise RuntimeError("test@example.com user missing from conftest db")
    u.hashed_password = get_password_hash(_TEST_LOGIN_PASSWORD)
    db.commit()
    yield


def _decode_access_token(token: str) -> dict:
    return jwt.decode(
        token,
        settings.SECRET_KEY,
        algorithms=[settings.ALGORITHM],
    )


@pytest.mark.usefixtures("client", "db")
class TestRefreshAccessToken:
    def test_refresh_returns_valid_jwt_claims(self, client: TestClient):
        """New access token from refresh must decode and include expected claims."""
        r = client.post(
            "/api/v1/users/login",
            json={"email": "test@example.com", "password": _TEST_LOGIN_PASSWORD},
        )
        assert r.status_code == 200, r.text
        login_data = r.json()["data"]
        refresh = login_data["refresh_token"]

        r2 = client.post(
            "/api/v1/users/refresh",
            json={"refresh_token": refresh},
        )
        assert r2.status_code == 200, r2.text
        body = r2.json()
        assert body["message"] == "Access token refreshed"
        access = body["data"]["access_token"]

        payload = _decode_access_token(access)
        assert payload.get("type") == "access"
        assert payload.get("email") == "test@example.com"
        assert payload.get("user_id") is not None
        assert "exp" in payload

    def test_refreshed_jwt_tenant_matches_current_tenant_after_switch(
        self,
        client: TestClient,
        db,
    ):
        """
        After switching current_tenant_id, the next refresh must mint a JWT whose
        tenant_id claim matches the new tenant (not a stale cached token).
        """
        user = db.query(User).filter(User.email == "test@example.com").first()
        assert user is not None
        t1 = user.tenants[0]

        t2 = Tenant(name="Second tenant", schema_name=f"schema_{uuid.uuid4().hex[:8]}")
        db.add(t2)
        db.commit()
        db.refresh(t2)

        user.tenants.append(t2)
        db.commit()

        assign_role_to_user_tenant(db, user.id, t1.id, "user")
        assign_role_to_user_tenant(db, user.id, t2.id, "user")

        user.current_tenant_id = t1.id
        db.commit()

        r = client.post(
            "/api/v1/users/login",
            json={"email": "test@example.com", "password": _TEST_LOGIN_PASSWORD},
        )
        assert r.status_code == 200, r.text
        refresh = r.json()["data"]["refresh_token"]

        r1 = client.post("/api/v1/users/refresh", json={"refresh_token": refresh})
        assert r1.status_code == 200, r1.text
        p1 = _decode_access_token(r1.json()["data"]["access_token"])
        assert p1.get("tenant_id") == str(t1.id)

        user.current_tenant_id = t2.id
        db.add(user)
        db.commit()

        r2 = client.post("/api/v1/users/refresh", json={"refresh_token": refresh})
        assert r2.status_code == 200, r2.text
        p2 = _decode_access_token(r2.json()["data"]["access_token"])
        assert p2.get("tenant_id") == str(t2.id), (
            "Access token from refresh must reflect current_tenant_id when role is unchanged; "
            f"expected tenant {t2.id}, got claim {p2.get('tenant_id')!r}"
        )
