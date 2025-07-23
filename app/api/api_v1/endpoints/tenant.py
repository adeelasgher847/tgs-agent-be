from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.schemas.tenant import TenantCreate, TenantCreateResponse
from app.models.tenant import Tenant
from app.models.user import User
from app.api.deps import get_db, get_current_user
import re

router = APIRouter()

def generate_schema_name(tenant_name: str) -> str:
    """Generate a schema name from tenant name"""
    # Convert to lowercase, replace spaces/special chars with underscores
    schema_name = re.sub(r'[^a-zA-Z0-9]', '_', tenant_name.lower())
    # Remove multiple underscores and trailing/leading underscores
    schema_name = re.sub(r'_+', '_', schema_name).strip('_')
    return f"{schema_name}_schema"

@router.post("/create", response_model=TenantCreateResponse)
def create_tenant(tenant_in: TenantCreate, current_user: User = Depends(get_current_user),db: Session = Depends(get_db)):
    """
    Create a new tenant organization and associate the creator as its admin.
    
    Requirements:
    - Tenant name must be unique
    - Creator user is auto-linked to the tenant with role "admin"
    - Returns tenant_id and tenant details
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
    
    # Create new tenant with current user as admin
    db_tenant = Tenant(
        name=tenant_in.name,
        schema_name=schema_name,
        admin_id=current_user.id
    )
    
    db.add(db_tenant)
    db.commit()
    db.refresh(db_tenant)
    
    # Auto-link creator to tenant (this would include role "admin" when role system is implemented)
    # For now, the admin relationship is tracked via admin_id field
    
    return TenantCreateResponse(
        tenant_id=db_tenant.id,
        message="Tenant created successfully",
        tenant=db_tenant
    )