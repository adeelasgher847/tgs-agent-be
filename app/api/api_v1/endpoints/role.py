from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session
from typing import List
from app.schemas.role import RoleCreate, RoleOut
from app.schemas.base import SuccessResponse
from app.models.role import Role
from app.models.user import User
from app.api.deps import get_db, require_admin_or_owner
from app.services.audit_service import log_audit_event
from app.utils.response import create_success_response
import uuid

router = APIRouter()

@router.post("/", response_model=SuccessResponse[RoleOut],include_in_schema=False)
def create_role(request: Request, role_in: RoleCreate, user: User = Depends(require_admin_or_owner), db: Session = Depends(get_db)):
    """Create a new role"""
    existing_role = db.query(Role).filter(Role.name == role_in.name).first()
    if existing_role:
        raise HTTPException(status_code=400, detail="Role name already exists")

    db_role = Role(**role_in.model_dump())
    db.add(db_role)
    db.commit()
    db.refresh(db_role)
    log_audit_event(
        db,
        request=request,
        tenant_id=user.current_tenant_id,
        action="rbac_role.created",
        resource_type="rbac_role",
        resource_id=db_role.id,
        new_value=role_in.model_dump(),
        actor_user_id=user.id,
    )
    return create_success_response(db_role, "Role created successfully", status.HTTP_201_CREATED)

@router.get("/", response_model=SuccessResponse[List[RoleOut]],include_in_schema=False)
def get_roles(skip: int = 0, limit: int = 100, user: User = Depends(require_admin_or_owner), db: Session = Depends(get_db)):
    """Get all roles"""
    roles = db.query(Role).offset(skip).limit(limit).all()
    return create_success_response(roles, "Roles retrieved successfully")

@router.get("/{role_id}", response_model=SuccessResponse[RoleOut])
def get_role(role_id: uuid.UUID, user: User = Depends(require_admin_or_owner), db: Session = Depends(get_db)):
    """Get a specific role by ID"""
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")
    return create_success_response(role, "Role retrieved successfully")

@router.put("/{role_id}", response_model=SuccessResponse[RoleOut],include_in_schema=False)
def update_role(role_id: uuid.UUID, role_in: RoleCreate, request: Request, user: User = Depends(require_admin_or_owner), db: Session = Depends(get_db)):
    """Update a role"""
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")

    if role_in.name != role.name:
        existing_role = db.query(Role).filter(Role.name == role_in.name).first()
        if existing_role:
            raise HTTPException(status_code=400, detail="Role name already exists")

    old_val = {"name": role.name}
    for field, value in role_in.model_dump().items():
        setattr(role, field, value)

    db.commit()
    db.refresh(role)
    log_audit_event(
        db,
        request=request,
        tenant_id=user.current_tenant_id,
        action="rbac_role.updated",
        resource_type="rbac_role",
        resource_id=role_id,
        old_value=old_val,
        new_value=role_in.model_dump(),
        actor_user_id=user.id,
    )
    return create_success_response(role, "Role updated successfully")

@router.delete("/{role_id}", response_model=SuccessResponse[dict],include_in_schema=False)
def delete_role(role_id: uuid.UUID, request: Request, user: User = Depends(require_admin_or_owner), db: Session = Depends(get_db)):
    """Delete a role"""
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(status_code=404, detail="Role not found")

    old_val = {"name": role.name}
    db.delete(role)
    db.commit()
    log_audit_event(
        db,
        request=request,
        tenant_id=user.current_tenant_id,
        action="rbac_role.deleted",
        resource_type="rbac_role",
        resource_id=role_id,
        old_value=old_val,
        actor_user_id=user.id,
    )
    return create_success_response({"id": str(role_id)}, "Role deleted successfully")