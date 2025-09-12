from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.schemas.tenant import TenantCreate, TenantCreateResponse, TenantOut
from app.schemas.auth import SwitchTenantRequest, TokenResponse, RoleInfo
from app.schemas.base import SuccessResponse
from app.models.tenant import Tenant
from app.models.user import User
from app.models.role import Role
from app.api.deps import get_db, get_current_user_jwt
from app.core.security import create_user_token
from app.utils.response import create_success_response
import re
from app.core.config import settings
from app.models.user import user_tenant_association

from sqlalchemy import update
router = APIRouter()

def generate_schema_name(tenant_name: str) -> str:
    """Generate a schema name from tenant name"""
    # Convert to lowercase, replace spaces/special chars with underscores
    schema_name = re.sub(r'[^a-zA-Z0-9]', '_', tenant_name.lower())
    # Remove multiple underscores and trailing/leading underscores
    schema_name = re.sub(r'_+', '_', schema_name).strip('_')
    return f"{schema_name}_schema"

@router.post("/create", response_model=SuccessResponse[TenantCreateResponse])
def create_tenant(tenant_in: TenantCreate, current_user: User = Depends(get_current_user_jwt), db: Session = Depends(get_db)):
    """
    Create a new tenant organization and associate the creator as its admin.
    
    Requirements:
    - Tenant name must be unique
    - Creator user is auto-linked to the tenant with role "admin"
    - Sets the new tenant as user's current tenant
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
    
    # Get admin role by name
    admin_role = db.query(Role).filter(Role.name == settings.ADMIN_ROLE).first()
    if not admin_role:
        raise HTTPException(
            status_code=400, 
            detail="Admin role not found. Please contact administrator."
        )

    # Add user to tenant's users list (many-to-many association)
    current_user.tenants.append(db_tenant)
    
    # Commit the association first
    db.commit()
    
    # Update the role_id in the association table
    stmt = update(user_tenant_association).where(
        (user_tenant_association.c.user_id == current_user.id) &
        (user_tenant_association.c.tenant_id == db_tenant.id)
    ).values(role_id=admin_role.id)
    
    db.execute(stmt)
    
    # Set the new tenant as user's current tenant
    current_user.current_tenant_id = db_tenant.id
    
    db.commit()
    db.refresh(current_user)
    
    # Convert SQLAlchemy model to Pydantic model
    tenant_out = TenantOut.model_validate(db_tenant)
    
    tenant_response = TenantCreateResponse(
        tenant_id=db_tenant.id,
        tenant=tenant_out
    )
    
    return create_success_response(tenant_response, "Tenant created successfully", status.HTTP_201_CREATED)


@router.post("/switch", response_model=SuccessResponse[TokenResponse])
def switch_tenant(
    switch_data: SwitchTenantRequest,
    current_user: User = Depends(get_current_user_jwt),
    db: Session = Depends(get_db)
):
    """
    Switch to a different tenant and return new JWT token with role information.
    Also updates the user's current_tenant_id in the database.
    """
    # Get user's tenant IDs from database
    user_tenant_ids = [tenant.id for tenant in current_user.tenants]
    
    # Check if user has access to the requested tenant
    if switch_data.tenant_id not in user_tenant_ids:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access denied to this tenant"
        )
    
    # Update user's current_tenant_id in the database
    current_user.current_tenant_id = switch_data.tenant_id
    db.commit()
    db.refresh(current_user)
    
    # Get role information for the switched tenant
    role_info = None
    current_role = None
    from app.services.role_service import get_user_role_in_tenant
    role = get_user_role_in_tenant(db, current_user.id, switch_data.tenant_id)
    if role:
        role_info = RoleInfo(
            id=role.id,
            name=role.name,
            description=role.description
        )
        current_role = role.name
    
    # Create new token with updated tenant and role
    access_token = create_user_token(
        user_id=current_user.id,
        email=current_user.email,
        tenant_id=switch_data.tenant_id,
        role=current_role
    )

    # Create refresh token (valid 7 days)
    from app.core.security import create_refresh_token_value, refresh_token_expires_at
    from app.models.refresh_token import RefreshToken
    
    rt_value = create_refresh_token_value()
    rt = RefreshToken(
        user_id=current_user.id,
        token=rt_value,
        expires_at=refresh_token_expires_at(),
        revoked=False
    )
    db.add(rt)
    db.commit()
    
    token_response = TokenResponse(
        access_token=access_token,
        user_id=current_user.id,
        email=current_user.email,
        tenant_id=switch_data.tenant_id,
        tenant_ids=user_tenant_ids,
        role=role_info,
        refresh_token=rt_value
    )
    
    return create_success_response(token_response, "Tenant switched successfully")