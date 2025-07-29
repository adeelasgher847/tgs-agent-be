from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.schemas.tenant import TenantCreate, TenantCreateResponse, TenantOut
from app.schemas.auth import SwitchTenantRequest, TokenResponse
from app.models.tenant import Tenant
from app.models.user import User
from app.api.deps import get_db, get_current_user_jwt
from app.core.security import create_user_token
import re
from app.core.config import settings

router = APIRouter()

def generate_schema_name(tenant_name: str) -> str:
    """Generate a schema name from tenant name"""
    # Convert to lowercase, replace spaces/special chars with underscores
    schema_name = re.sub(r'[^a-zA-Z0-9]', '_', tenant_name.lower())
    # Remove multiple underscores and trailing/leading underscores
    schema_name = re.sub(r'_+', '_', schema_name).strip('_')
    return f"{schema_name}_schema"

@router.post("/create", response_model=TenantCreateResponse)
def create_tenant(tenant_in: TenantCreate, current_user: User = Depends(get_current_user_jwt), db: Session = Depends(get_db)):
    """
    Create a new tenant organization and associate the creator as its admin.
    
    Requirements:
    - Tenant name must be unique
    - Creator user is auto-linked to the tenant with role "admin"
    - Returns tenant_id and tenant details with updated token
    """
    # Check if tenant name already exists
    existing_tenant = db.query(Tenant).filter(Tenant.name == tenant_in.name).first()
    if existing_tenant:
        raise HTTPException(status_code=400, detail="Tenant name already exists")
    
    # Generate unique schema name
    schema_name = generate_schema_name(tenant_in.name)
    
    # Ensure schema name is unique
    existing_schema = db.query(Tenant).filter(Tenant.schema_name == schema_name).first()
    counter = 1
    original_schema = schema_name
    while existing_schema:
        schema_name = f"{original_schema}_{counter}"
        existing_schema = db.query(Tenant).filter(Tenant.schema_name == schema_name).first()
        counter += 1
    
    # Create new tenant
    db_tenant = Tenant(
        name=tenant_in.name,
        schema_name=schema_name
    )
    
    db.add(db_tenant)
    db.commit()
    db.refresh(db_tenant)
    
    # Add user to tenant's users list (many-to-many association)
    current_user.tenants.append(db_tenant)
    
    # Update user's role to admin (role_id = 1 for admin)
    current_user.role_id = settings.ADMIN_ROLE_ID 
    db.commit()
    db.refresh(current_user)
    
    # Convert SQLAlchemy model to Pydantic model
    tenant_out = TenantOut.model_validate(db_tenant)
    
    return TenantCreateResponse(
        tenant_id=db_tenant.id,
        message="Tenant created successfully",
        tenant=tenant_out
    )


@router.post("/switch", response_model=TokenResponse)
def switch_tenant(
    switch_data: SwitchTenantRequest,
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """
    Switch to a different tenant and return new JWT token.
    """
    # Get user's tenant IDs from database
    user_tenant_ids = [tenant.id for tenant in current_user.tenants]
    
    # Check if user has access to the requested tenant
    if switch_data.tenant_id not in user_tenant_ids:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access denied to this tenant"
        )
    
    # Create new token with updated tenant (no tenant_ids in token)
    access_token = create_user_token(
        user_id=current_user.id,
        email=current_user.email,
        tenant_id=switch_data.tenant_id  
    )
    
    return TokenResponse(
        access_token=access_token,
        user_id=current_user.id,
        email=current_user.email,
        tenant_id=switch_data.tenant_id,
        tenant_ids=user_tenant_ids
    )