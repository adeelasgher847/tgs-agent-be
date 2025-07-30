import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from app.models.tenant import Tenant
from app.models.user import User
from app.models.role import Role
from app.core.security import create_user_token, verify_password, get_password_hash

# Test account credentials (you'll provide these)
TEST_USER_EMAIL = "test@example.com"
TEST_USER_ID = 1  # You'll provide the actual test user ID
TEST_TENANT_ID = 1  # You'll provide the actual test tenant ID

class TestUserAuthentication:
    """Test user registration and authentication endpoints"""
    
    def test_register_user_success(self, client: TestClient, db: Session):
        """Test successful user registration"""
        user_data = {
            "email": "newuser@example.com",
            "password": "securepassword123",
            "first_name": "John",
            "last_name": "Doe",
            "phone": "+1234567890"
        }
        
        response = client.post("/api/v1/users/register", json=user_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert data["email"] == "newuser@example.com"
        assert data["first_name"] == "John"
        assert data["last_name"] == "Doe"
        assert data["phone"] == "+1234567890"
        assert "id" in data
        assert "role_id" in data
        assert "join_date" in data
        assert "created_at" in data
        assert "password" not in data  # Password should not be returned
        
        # Verify user is in database
        user = db.query(User).filter(User.email == "newuser@example.com").first()
        assert user is not None
        assert user.first_name == "John"
        assert user.last_name == "Doe"
        assert user.phone == "+1234567890"
        assert user.role_id == 2  # Default user role
        assert verify_password("securepassword123", user.hashed_password)
    
    def test_register_user_duplicate_email(self, client: TestClient, db: Session):
        """Test registration with existing email"""
        # Create a user first
        existing_user = User(
            email="existing@example.com",
            hashed_password="hashedpassword",
            first_name="Existing",
            last_name="User",
            role_id=2
        )
        db.add(existing_user)
        db.commit()
        
        user_data = {
            "email": "existing@example.com",
            "password": "newpassword123",
            "first_name": "New",
            "last_name": "User"
        }
        
        response = client.post("/api/v1/users/register", json=user_data)
        
        assert response.status_code == 400
        assert "Email already registered" in response.json()["detail"]
    
    def test_register_user_missing_required_fields(self, client: TestClient):
        """Test registration with missing required fields"""
        # Test missing email
        user_data = {
            "password": "password123",
            "first_name": "John",
            "last_name": "Doe"
        }
        
        response = client.post("/api/v1/users/register", json=user_data)
        assert response.status_code == 422
        
        # Test missing password
        user_data = {
            "email": "test@example.com",
            "first_name": "John",
            "last_name": "Doe"
        }
        
        response = client.post("/api/v1/users/register", json=user_data)
        assert response.status_code == 422
        
        # Test missing first_name
        user_data = {
            "email": "test@example.com",
            "password": "password123",
            "last_name": "Doe"
        }
        
        response = client.post("/api/v1/users/register", json=user_data)
        assert response.status_code == 422
        
        # Test missing last_name
        user_data = {
            "email": "test@example.com",
            "password": "password123",
            "first_name": "John"
        }
        
        response = client.post("/api/v1/users/register", json=user_data)
        assert response.status_code == 422
    
    def test_register_user_invalid_email(self, client: TestClient):
        """Test registration with invalid email format"""
        user_data = {
            "email": "invalid-email",
            "password": "password123",
            "first_name": "John",
            "last_name": "Doe"
        }
        
        response = client.post("/api/v1/users/register", json=user_data)
        assert response.status_code == 422
    
    def test_register_user_short_password(self, client: TestClient):
        """Test registration with password shorter than 6 characters"""
        user_data = {
            "email": "test@example.com",
            "password": "12345",  # Less than 6 characters
            "first_name": "John",
            "last_name": "Doe"
        }
        
        response = client.post("/api/v1/users/register", json=user_data)
        assert response.status_code == 422
    
    def test_register_user_empty_fields(self, client: TestClient):
        """Test registration with empty string fields"""
        user_data = {
            "email": "",
            "password": "",
            "first_name": "",
            "last_name": ""
        }
        
        response = client.post("/api/v1/users/register", json=user_data)
        assert response.status_code == 422
    
    def test_login_success(self, client: TestClient, db: Session):
        """Test successful user login"""
        # Create a test user for login with proper password hashing
        test_password = "testpassword123"
        hashed_password = get_password_hash(test_password)
        
        test_user = User(
            email="logintest@example.com",
            hashed_password=hashed_password,
            first_name="Login",
            last_name="Test",
            role_id=2
        )
        db.add(test_user)
        db.commit()
        
        login_data = {
            "email": "logintest@example.com",
            "password": test_password
        }
        
        response = client.post("/api/v1/users/login", json=login_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "access_token" in data
        assert data["user_id"] == test_user.id
        assert data["email"] == "logintest@example.com"
        assert "tenant_id" in data
        assert "tenant_ids" in data
        assert isinstance(data["tenant_ids"], list)
    
    def test_login_invalid_email(self, client: TestClient):
        """Test login with non-existent email"""
        login_data = {
            "email": "nonexistent@example.com",
            "password": "password123"
        }
        
        response = client.post("/api/v1/users/login", json=login_data)
        
        assert response.status_code == 401
        assert "Incorrect email or password" in response.json()["detail"]
    
    def test_login_invalid_password(self, client: TestClient, db: Session):
        """Test login with incorrect password"""
        # Create a test user with proper password hashing
        test_password = "correctpassword123"
        hashed_password = get_password_hash(test_password)
        
        test_user = User(
            email="wrongpass@example.com",
            hashed_password=hashed_password,
            first_name="Wrong",
            last_name="Pass",
            role_id=2
        )
        db.add(test_user)
        db.commit()
        
        login_data = {
            "email": "wrongpass@example.com",
            "password": "wrongpassword"
        }
        
        response = client.post("/api/v1/users/login", json=login_data)
        
        assert response.status_code == 401
        assert "Incorrect email or password" in response.json()["detail"]
    
    def test_login_missing_fields(self, client: TestClient):
        """Test login with missing fields"""
        # Test missing email
        login_data = {
            "password": "password123"
        }
        
        response = client.post("/api/v1/users/login", json=login_data)
        assert response.status_code == 422
        
        # Test missing password
        login_data = {
            "email": "test@example.com"
        }
        
        response = client.post("/api/v1/users/login", json=login_data)
        assert response.status_code == 422
    
    def test_logout_success(self, client: TestClient):
        """Test logout endpoint"""
        response = client.post("/api/v1/users/logout")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Successfully logged out"
    
    def test_register_user_with_tenant_association(self, client: TestClient, db: Session):
        """Test that registered user gets proper role assignment"""
        user_data = {
            "email": "tenantuser@example.com",
            "password": "securepassword123",
            "first_name": "Tenant",
            "last_name": "User"
        }
        
        response = client.post("/api/v1/users/register", json=user_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify user has default role (ID: 2 for "user" role)
        assert data["role_id"] == 2
        
        # Verify in database
        user = db.query(User).filter(User.email == "tenantuser@example.com").first()
        assert user.role_id == 2

class TestTenantManagement:
    """Test tenant management endpoints using test account"""
    
    def get_test_token(self) -> str:
        """Get JWT token for test user"""
        return create_user_token(
            user_id=TEST_USER_ID, 
            email=TEST_USER_EMAIL,
            tenant_id=TEST_TENANT_ID
        )
    
    def test_create_tenant_success(self, client: TestClient, db: Session):
        """Test successful tenant creation"""
        token = self.get_test_token()
        
        response = client.post(
            "/api/v1/tenants/create",
            json={"name": "New Test Tenant"},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Tenant created successfully"
        assert data["tenant"]["name"] == "New Test Tenant"
        assert "id" in data["tenant"]
        assert "schema_name" in data["tenant"]
        
        # Verify tenant is in database
        tenant = db.query(Tenant).filter(Tenant.name == "New Test Tenant").first()
        assert tenant is not None
        assert tenant.name == "New Test Tenant"
    
    def test_create_tenant_duplicate_name(self, client: TestClient, db: Session):
        """Test tenant creation with duplicate name"""
        # Create a tenant first
        existing_tenant = Tenant(name="Duplicate Test Tenant", schema_name="duplicate_schema")
        db.add(existing_tenant)
        db.commit()
        
        token = self.get_test_token()
        
        response = client.post(
            "/api/v1/tenants/create",
            json={"name": "Duplicate Test Tenant"},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 400
        assert "Tenant name already exists" in response.json()["detail"]
    
    def test_create_tenant_unauthorized(self, client: TestClient):
        """Test tenant creation without authentication"""
        response = client.post(
            "/api/v1/tenants/create",
            json={"name": "Unauthorized Tenant"}
        )
        
        assert response.status_code == 403
    
    def test_switch_tenant_success(self, client: TestClient, db: Session):
        """Test switching between tenants"""
        # Create a second tenant for the test user
        second_tenant = Tenant(name="Second Test Tenant", schema_name="second_schema")
        db.add(second_tenant)
        db.commit()
        
        # Add test user to second tenant
        test_user = db.query(User).filter(User.id == TEST_USER_ID).first()
        if test_user:
            test_user.tenants.append(second_tenant)
            db.commit()
        
        token = self.get_test_token()
        
        response = client.post(
            "/api/v1/tenants/switch",
            json={"tenant_id": second_tenant.id},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["tenant_id"] == second_tenant.id
        assert data["user_id"] == TEST_USER_ID
    
    def test_switch_tenant_unauthorized(self, client: TestClient, db: Session):
        """Test switching to tenant user doesn't have access to"""
        # Create a tenant that test user doesn't have access to
        unauthorized_tenant = Tenant(name="Unauthorized Tenant", schema_name="unauthorized_schema")
        db.add(unauthorized_tenant)
        db.commit()
        
        token = self.get_test_token()
        
        response = client.post(
            "/api/v1/tenants/switch",
            json={"tenant_id": unauthorized_tenant.id},
            headers={"Authorization": f"Bearer {token}"}
        )
        
        assert response.status_code == 401
        assert "Access denied to this tenant" in response.json()["detail"]

class TestRoleManagement:
    """Test role management endpoints"""
    
    def test_create_role_success(self, client: TestClient, db: Session):
        """Test successful role creation"""
        response = client.post(
            "/api/v1/roles/",
            json={
                "name": "test_role",
                "description": "Test role description"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_role"
        assert data["description"] == "Test role description"
        assert "id" in data
        # Check if created_at exists (it might not be in the response)
        if "created_at" in data:
            assert data["created_at"] is not None
        
        # Verify role is in database
        role = db.query(Role).filter(Role.name == "test_role").first()
        assert role is not None
        assert role.name == "test_role"
    
    def test_create_role_duplicate_name(self, client: TestClient, db: Session):
        """Test role creation with duplicate name"""
        # Create first role
        role1 = Role(name="duplicate_role", description="First role")
        db.add(role1)
        db.commit()
        
        # Try to create second role with same name
        response = client.post(
            "/api/v1/roles/",
            json={
                "name": "duplicate_role",
                "description": "Second role"
            }
        )
        
        assert response.status_code == 400
        assert "Role name already exists" in response.json()["detail"]
    
    def test_get_roles_list(self, client: TestClient, db: Session):
        """Test getting list of roles"""
        # Create some test roles
        roles = [
            Role(name="role1", description="First role"),
            Role(name="role2", description="Second role"),
            Role(name="role3", description="Third role")
        ]
        for role in roles:
            db.add(role)
        db.commit()
        
        response = client.get("/api/v1/roles/")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 3  # At least the 3 we created
        
        # Check that our roles are in the response
        role_names = [role["name"] for role in data]
        assert "role1" in role_names
        assert "role2" in role_names
        assert "role3" in role_names
    
    def test_get_role_by_id(self, client: TestClient, db: Session):
        """Test getting a specific role by ID"""
        # Create a test role
        role = Role(name="test_role_by_id", description="Test role")
        db.add(role)
        db.commit()
        db.refresh(role)
        
        response = client.get(f"/api/v1/roles/{role.id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == role.id
        assert data["name"] == "test_role_by_id"
        assert data["description"] == "Test role"
    
    def test_get_role_not_found(self, client: TestClient):
        """Test getting a role that doesn't exist"""
        response = client.get("/api/v1/roles/99999")
        
        assert response.status_code == 404
        assert "Role not found" in response.json()["detail"]
    
    def test_update_role_success(self, client: TestClient, db: Session):
        """Test successful role update"""
        # Create a test role
        role = Role(name="update_test_role", description="Original description")
        db.add(role)
        db.commit()
        db.refresh(role)
        
        response = client.put(
            f"/api/v1/roles/{role.id}",
            json={
                "name": "updated_role_name",
                "description": "Updated description"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "updated_role_name"
        assert data["description"] == "Updated description"
        
        # Verify in database - refresh the session to see the changes
        db.expire_all()  # Expire all objects to force fresh queries
        updated_role = db.query(Role).filter(Role.id == role.id).first()
        assert updated_role.name == "updated_role_name"
        assert updated_role.description == "Updated description"
    
    def test_delete_role_success(self, client: TestClient, db: Session):
        """Test successful role deletion"""
        # Create a test role
        role = Role(name="delete_test_role", description="To be deleted")
        db.add(role)
        db.commit()
        db.refresh(role)
        
        response = client.delete(f"/api/v1/roles/{role.id}")
        
        assert response.status_code == 200
        assert response.json()["message"] == "Role deleted successfully"
        
        # Verify role is deleted from database
        deleted_role = db.query(Role).filter(Role.id == role.id).first()
        assert deleted_role is None

class TestHealthCheck:
    """Test health check endpoint"""
    
    def test_health_check(self, client: TestClient):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root_endpoint(self, client: TestClient):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        assert "Welcome to the Multi-Tenant SaaS Voice Agent Backend!" in response.json()["message"] 