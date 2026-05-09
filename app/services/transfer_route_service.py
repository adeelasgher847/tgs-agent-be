from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.agent import Agent
from app.models.transfer_route import TransferRoute
from app.schemas.transfer_route import TransferRouteCreate, TransferRouteUpdate


class TransferRouteService:
    def list_for_tenant(self, db: Session, tenant_id: uuid.UUID) -> List[TransferRoute]:
        return (
            db.query(TransferRoute)
            .filter(
                TransferRoute.tenant_id == tenant_id,
                TransferRoute.is_deleted == False,  # noqa: E712
            )
            .order_by(TransferRoute.created_at.desc())
            .all()
        )

    def get(self, db: Session, route_id: uuid.UUID, tenant_id: uuid.UUID) -> Optional[TransferRoute]:
        return (
            db.query(TransferRoute)
            .filter(
                TransferRoute.id == route_id,
                TransferRoute.tenant_id == tenant_id,
                TransferRoute.is_deleted == False,  # noqa: E712
            )
            .first()
        )

    def create(self, db: Session, tenant_id: uuid.UUID, body: TransferRouteCreate) -> TransferRoute:
        row = TransferRoute(
            tenant_id=tenant_id,
            friendly_name=body.friendly_name.strip(),
            phone_number=body.phone_number,
            transfer_type=body.transfer_type.value,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    def update(
        self, db: Session, route_id: uuid.UUID, tenant_id: uuid.UUID, body: TransferRouteUpdate
    ) -> TransferRoute:
        row = self.get(db, route_id, tenant_id)
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transfer route not found")
        data = body.model_dump(exclude_unset=True)
        if "friendly_name" in data and data["friendly_name"] is not None:
            row.friendly_name = data["friendly_name"].strip()
        if "phone_number" in data and data["phone_number"] is not None:
            row.phone_number = data["phone_number"]
        if "transfer_type" in data and data["transfer_type"] is not None:
            row.transfer_type = (
                data["transfer_type"].value
                if hasattr(data["transfer_type"], "value")
                else str(data["transfer_type"])
            )
        db.commit()
        db.refresh(row)
        return row

    def soft_delete(self, db: Session, route_id: uuid.UUID, tenant_id: uuid.UUID) -> None:
        row = self.get(db, route_id, tenant_id)
        if not row:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Transfer route not found")
        row.is_deleted = True
        db.query(Agent).filter(Agent.transfer_route_id == row.id).update(
            {Agent.transfer_route_id: None},
            synchronize_session=False,
        )
        db.commit()


transfer_route_service = TransferRouteService()
