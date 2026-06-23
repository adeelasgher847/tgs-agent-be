from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import Table, Column, UUID, ForeignKey
from app.models.role import Role
from app.models.product import Product
from app.core.product_enums import ProductName
from app.models.user import user_tenant_association
import uuid

# ─────────────────────────────────────────────────────────── RBAC hierarchy ──
# Canonical roles reuse the existing `role` catalog + `user_tenant_association`
# tables rather than a dedicated rbac_roles table — see docs/rbac-matrix.md.

ADMIN = "admin"
MANAGER = "manager"
CONFIG_ONLY = "config_only"
READ_ONLY = "read_only"
BILLING_ONLY = "billing_only"

CANONICAL_ROLES = (ADMIN, MANAGER, CONFIG_ONLY, READ_ONLY, BILLING_ONLY)

# Linear inheritance chain: admin > manager > config_only > read_only.
# billing_only deliberately has no rank here — it is not part of this chain,
# it only ever satisfies require_billing (see BILLING_ALLOWED_ROLES below).
ROLE_RANK = {
    READ_ONLY: 1,
    CONFIG_ONLY: 2,
    MANAGER: 3,
    ADMIN: 4,
}

# admin/manager outrank billing_only and inherit into it; config_only/read_only
# do not get billing access even though they outrank nothing here.
BILLING_ALLOWED_ROLES = frozenset({ADMIN, MANAGER, BILLING_ONLY})


def has_rank(role_name: Optional[str], required: str) -> bool:
    """True if ``role_name`` is at or above ``required`` in the linear chain.

    Unranked names (None, or a role outside the chain like billing_only)
    are treated as rank 0 — they fail every chain check.
    """
    return ROLE_RANK.get(role_name, 0) >= ROLE_RANK[required]


def can_access_billing(role_name: Optional[str]) -> bool:
    return role_name in BILLING_ALLOWED_ROLES


def get_display_role_details(
    db: Session, user_id: uuid.UUID, tenant_id: uuid.UUID
) -> Optional[dict]:
    """Resolve display role details (name and description) for a (user, tenant).
    If the user is the workspace creator (is_creator=True), returns name='owner'.
    """
    row = (
        db.query(
            user_tenant_association.c.is_creator,
            user_tenant_association.c.role_id,
        )
        .filter(
            user_tenant_association.c.user_id == user_id,
            user_tenant_association.c.tenant_id == tenant_id,
        )
        .first()
    )
    if row is None:
        return None

    is_creator, role_id = row
    if is_creator:
        return {
            "name": "owner",
            "description": "Owner role with full access to tenant",
        }

    if role_id is None:
        return {
            "name": READ_ONLY,
            "description": "Read-only access; blocked from mutating endpoints",
        }

    role = db.query(Role).filter(Role.id == role_id).first()
    if role:
        return {
            "name": role.name,
            "description": role.description,
        }
    return {
        "name": READ_ONLY,
        "description": "Read-only access; blocked from mutating endpoints",
    }



def get_membership_role_name(
    db: Session, user_id: uuid.UUID, tenant_id: uuid.UUID
) -> Optional[str]:
    """Resolve the effective role name for a (user, tenant) pair.

    Returns ``None`` only when the user has no membership row at all for this
    tenant. A member with no role assigned yet defaults to ``read_only``
    (never rejected) per the RBAC matrix. The workspace creator's
    ``is_creator`` flag always resolves to ``admin``, regardless of whatever
    role_id is stored on the row.
    """
    # db.query(), not db.execute(select(...)) — some tests wrap the session
    # to intercept execute() generically for unrelated raw-SQL mocking
    # (e.g. pgvector search); .query() passes through untouched, matching
    # the convention of every other lookup in this module.
    row = (
        db.query(
            user_tenant_association.c.is_creator,
            user_tenant_association.c.role_id,
        )
        .filter(
            user_tenant_association.c.user_id == user_id,
            user_tenant_association.c.tenant_id == tenant_id,
        )
        .first()
    )
    if row is None:
        return None

    is_creator, role_id = row
    if is_creator:
        return ADMIN
    if role_id is None:
        return READ_ONLY

    role = db.query(Role).filter(Role.id == role_id).first()
    return role.name if role else READ_ONLY


def get_default_product_id(db: Session) -> uuid.UUID | None:
    product = db.query(Product).filter(Product.name == ProductName.TALENTSYNC.value).first()
    return product.id if product else None

def assign_role_to_user_tenant(db: Session, user_id: uuid.UUID, tenant_id: uuid.UUID, role_name: str):
    role = db.query(Role).filter(Role.name == role_name).first()
    if not role:
        return None
    default_product_id = get_default_product_id(db)
    
    result = db.execute(
        user_tenant_association.update().where(
            (user_tenant_association.c.user_id == user_id) & 
            (user_tenant_association.c.tenant_id == tenant_id)
        ).values(role_id=role.id, product_id=default_product_id)
    )
    
    if result.rowcount == 0:
        db.execute(
            user_tenant_association.insert().values(
                user_id=user_id,
                tenant_id=tenant_id,
                role_id=role.id,
                product_id=default_product_id,
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


def get_user_product_in_tenant(db: Session, user_id: uuid.UUID, tenant_id: uuid.UUID):
    result = db.query(Product).join(
        user_tenant_association, Product.id == user_tenant_association.c.product_id
    ).filter(
        user_tenant_association.c.user_id == user_id,
        user_tenant_association.c.tenant_id == tenant_id
    ).first()

    return result

def is_admin_in_tenant(db: Session, user_id: uuid.UUID, tenant_id: uuid.UUID) -> bool:
    return has_rank(get_membership_role_name(db, user_id, tenant_id), ADMIN)
