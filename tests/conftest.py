import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.db.base import Base
from app.db.session import SessionLocal
from app.core.config import settings
from app.models.user import User
from app.models.role import Role
from app.models.tenant import Tenant

# Use a separate test database
TEST_DATABASE_URL = settings.DATABASE_URL + "_test"

engine = create_engine(TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the test database and tables
Base.metadata.create_all(bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

# Override the get_db dependency to use the test database
from app.api.deps import get_db
app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module")
def db():
    """Create a fresh database for the entire test module"""
    # Drop and recreate all tables
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    db = TestingSessionLocal()
    
    try:
        # Create test data
        # Create test role
        test_role = Role(name="test_user", description="Test user role")
        db.add(test_role)
        db.commit()
        db.refresh(test_role)
        
        # Create test tenant
        test_tenant = Tenant(name="Test Tenant", schema_name="test_tenant_schema")
        db.add(test_tenant)
        db.commit()
        db.refresh(test_tenant)
        
        # Create test user (you'll provide the actual test user data)
        test_user = User(
            email="test@example.com",
            hashed_password="$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4tbQJbqK8O",  # "testpassword123"
            role_id=test_role.id
        )
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        
        # Associate user with tenant
        test_user.tenants.append(test_tenant)
        db.commit()
        
        yield db
        
    finally:
        db.close()

@pytest.fixture(scope="module")
def client(db):
    """Create a test client that shares the same database session for the module"""
    with TestClient(app) as c:
        yield c 