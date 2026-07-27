from typing import Optional
import uuid

from sqlalchemy.orm import Session

from app.core.security import create_user_token, create_refresh_token_value, refresh_token_expires_at
from app.models.refresh_token import RefreshToken
from app.models.user import User
from app.schemas.auth import TokenResponse, RoleInfo
from app.services.role_service import get_user_product_in_tenant


def issue_tokens_for_user(
    db: Session,
    user: User,
    current_tenant_id: Optional[uuid.UUID],
    role_info: Optional[RoleInfo],
) -> TokenResponse:
    """Issue access and refresh tokens for a user without password checks.

    Used for provider-based authentication flows (e.g. Google OAuth, SSO).
    """
    access_token = create_user_token(
        user_id=user.id,
        email=user.email,
        tenant_id=current_tenant_id,
        role=role_info.name if role_info else None,
    )

    rt_value = create_refresh_token_value()
    rt = RefreshToken(
        user_id=user.id,
        token=rt_value,
        expires_at=refresh_token_expires_at(),
        revoked=False,
    )
    db.add(rt)
    db.commit()

    product_id = None
    if current_tenant_id:
        product = get_user_product_in_tenant(db, user.id, current_tenant_id)
        if product:
            product_id = product.id

    return TokenResponse(
        access_token=access_token,
        user_id=user.id,
        email=user.email,
        tenant_id=current_tenant_id,
        product_id=product_id,
        tenant_ids=[t.id for t in user.tenants],
        role=role_info,
        refresh_token=rt_value,
    )
