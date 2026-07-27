"""Service for handling SAML and OIDC Single Sign-On operations."""

import uuid
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.sso_config import SsoConfig
from app.models.user import User, user_tenant_association
from app.models.tenant import Tenant
from app.schemas.auth import RoleInfo
from app.core.security import get_password_hash
from app.services import role_service

# For authlib OIDC
from authlib.integrations.httpx_client import AsyncOAuth2Client

def get_sso_config(db: Session, workspace_id: uuid.UUID) -> Optional[SsoConfig]:
    return db.query(SsoConfig).filter(SsoConfig.workspace_id == workspace_id).first()


def find_or_create_user(db: Session, email: str, workspace_id: uuid.UUID) -> tuple[User, RoleInfo]:
    """Find a user by email from an SSO login, or create one if they don't exist.
    Ensure they are a member of the workspace (tenant).
    Returns (User, RoleInfo).
    """
    email = email.lower().strip()
    
    # Validate allowed email domains
    config = db.query(SsoConfig).filter(SsoConfig.workspace_id == workspace_id).first()
    if config and config.allowed_email_domains:
        email_domain = email.split("@", 1)[-1].lower().strip()
        allowed_domains = [d.lower().strip() for d in config.allowed_email_domains if d.strip()]

        if allowed_domains and email_domain not in allowed_domains:
            from fastapi import HTTPException
            raise HTTPException(
                status_code=403,
                detail=f"Email domain @{email_domain} is not permitted for this workspace",
            )
    
    # 1. Look for existing user
    user = db.query(User).filter(User.email == email, User.deleted_at.is_(None)).first()
    
    if not user:
        # Create new user
        # We don't have first/last name from basic assertion usually, so set a default.
        # Can extract from OIDC claims/SAML attributes later if needed.
        user = User(
            email=email,
            first_name="SSO",
            last_name="User",
            hashed_password=get_password_hash(uuid.uuid4().hex),  # Random unusable password
        )
        db.add(user)
        db.flush()
    else:
        # Touch last login? We could, but usually handled elsewhere or not strictly needed
        pass
        
    # 2. Ensure they are in the workspace
    # Check if association exists
    assoc = db.execute(
        text("SELECT 1 FROM user_tenant_association WHERE user_id = :uid AND tenant_id = :tid"),
        {"uid": str(user.id), "tid": str(workspace_id)}
    ).fetchone()
    
    if not assoc:
        # Add to workspace
        tenant = db.query(Tenant).filter(Tenant.id == workspace_id).first()
        if tenant:
            user.tenants.append(tenant)
            db.flush()
            
            # Default to read_only role
            from app.models.role import Role
            ro_role = db.query(Role).filter(Role.name == role_service.READ_ONLY).first()
            if ro_role:
                db.execute(
                    user_tenant_association.update()
                    .where(
                        (user_tenant_association.c.user_id == user.id) & 
                        (user_tenant_association.c.tenant_id == workspace_id)
                    )
                    .values(role_id=ro_role.id)
                )

    db.commit()
    db.refresh(user)
    
    # Get role info
    role = role_service.get_user_role_in_tenant(db, user.id, workspace_id)
    role_info = RoleInfo(id=role.id, name=role.name, description=role.description) if role else None
    
    return user, role_info
