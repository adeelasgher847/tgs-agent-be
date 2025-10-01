from sqlalchemy.orm import Session
from sqlalchemy import Table, Column, UUID, ForeignKey
from app.models.role import Role
from app.models.user import user_tenant_association
import uuid

def assign_role_to_user_tenant(db: Session, user_id: uuid.UUID, tenant_id: uuid.UUID, role_name: str):
    role = db.query(Role).filter(Role.name == role_name).first()
    if not role:
        return None
    
    result = db.execute(
        user_tenant_association.update().where(
            (user_tenant_association.c.user_id == user_id) & 
            (user_tenant_association.c.tenant_id == tenant_id)
        ).values(role_id=role.id)
    )
    
    if result.rowcount == 0:
        db.execute(
            user_tenant_association.insert().values(
                user_id=user_id,
                tenant_id=tenant_id,
                role_id=role.id
            )
        )
    
    db.commit()
    return True

def get_user_role_in_tenant(db: Session, user_id: uuid.UUID, tenant_id: uuid.UUID):
    result = db.query(Role).join(
        user_tenant_association, Role.id == user_tenant_association.c.role_id
    ).filter(
        user_tenant_association.c.user_id == user_id,
        user_tenant_association.c.tenant_id == tenant_id
    ).first()
    
    return result

def is_admin_in_tenant(db: Session, user_id: uuid.UUID, tenant_id: uuid.UUID):
    role = get_user_role_in_tenant(db, user_id, tenant_id)
    return role and role.name == "admin"
