"""
Immutable workspace (tenant) context attached to ``request.state.workspace``.

SQLAlchemy ``Tenant`` rows must not be stored on ``request.state`` after the
middleware DB session closes — this dataclass is a detached, request-safe
snapshot used by all downstream handlers and dependencies.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional

from app.models.tenant import Tenant


@dataclass(frozen=True, slots=True)
class Workspace:
    """Workspace == tenant. Attached to ``request.state.workspace`` after auth."""

    id: uuid.UUID
    name: str
    schema_name: str
    status: str
    credits: float
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    parent_workspace_id: Optional[uuid.UUID] = None
    workspace_type: str = "standalone"
    contact_email: Optional[str] = None

    @classmethod
    def from_tenant(cls, tenant: Tenant) -> Workspace:
        return cls(
            id=tenant.id,
            name=tenant.name,
            schema_name=tenant.schema_name,
            status=tenant.status,
            credits=float(tenant.credits or 0),
            stripe_customer_id=tenant.stripe_customer_id,
            stripe_subscription_id=tenant.stripe_subscription_id,
            parent_workspace_id=tenant.parent_workspace_id,
            workspace_type=tenant.workspace_type,
            contact_email=tenant.contact_email,
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Workspace:
        credits_raw = data.get("credits", 0)
        if isinstance(credits_raw, Decimal):
            credits = float(credits_raw)
        else:
            credits = float(credits_raw or 0)

        return cls(
            id=uuid.UUID(str(data["id"])),
            name=str(data["name"]),
            schema_name=str(data["schema_name"]),
            status=str(data["status"]),
            credits=credits,
            stripe_customer_id=data.get("stripe_customer_id"),
            stripe_subscription_id=data.get("stripe_subscription_id"),
            parent_workspace_id=uuid.UUID(str(data["parent_workspace_id"])) if data.get("parent_workspace_id") else None,
            workspace_type=str(data.get("workspace_type", "standalone")),
            contact_email=data.get("contact_email"),
        )

    def to_cache_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "schema_name": self.schema_name,
            "status": self.status,
            "credits": self.credits,
            "stripe_customer_id": self.stripe_customer_id,
            "stripe_subscription_id": self.stripe_subscription_id,
            "parent_workspace_id": str(self.parent_workspace_id) if self.parent_workspace_id else None,
            "workspace_type": self.workspace_type,
            "contact_email": self.contact_email,
        }

    @property
    def is_active(self) -> bool:
        return self.status == "active"
