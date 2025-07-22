from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.models.tenant import Tenant
from app.models.user import User

def test_create_tenant(client: TestClient):
    response = client.post(
        "/api/v1/tenants/",
        json={"name": "Test Tenant", "schema_name": "test_tenant_schema"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Tenant"
    assert "id" in data

def test_create_user(client: TestClient, db: Session):
    # First, create a tenant to associate the user with
    tenant = Tenant(name="Test Tenant for User", schema_name="user_tenant_schema")
    db.add(tenant)
    db.commit()
    db.refresh(tenant)

    response = client.post(
        "/api/v1/users/register",
        json={
            "email": "test@example.com",
            "password": "password",
            "tenant_id": tenant.id
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
    assert "id" in data
    assert "tenant_id" in data

    # Verify user is in the database
    user = db.query(User).filter(User.email == "test@example.com").first()
    assert user is not None
    assert user.tenant_id == tenant.id 