"""Allowed-domain (Web SDK whitelist) service."""
from __future__ import annotations

import uuid

from fastapi import HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.allowed_domain import AllowedDomain
from app.schemas.allowed_domain import AllowedDomainCreate, AllowedDomainOut

MAX_DOMAINS_PER_WORKSPACE = 20


class AllowedDomainService:
    def list_domains(self, db: Session, workspace_id: uuid.UUID) -> list[AllowedDomainOut]:
        rows = (
            db.execute(
                select(AllowedDomain)
                .where(AllowedDomain.workspace_id == workspace_id)
                .order_by(AllowedDomain.created_at.desc())
            )
            .scalars()
            .all()
        )
        return [AllowedDomainOut.model_validate(r) for r in rows]

    def create_domain(
        self, db: Session, workspace_id: uuid.UUID, body: AllowedDomainCreate
    ) -> AllowedDomainOut:
        count = db.execute(
            select(func.count())
            .select_from(AllowedDomain)
            .where(AllowedDomain.workspace_id == workspace_id)
        ).scalar_one()
        if count >= MAX_DOMAINS_PER_WORKSPACE:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Workspace already has the maximum of {MAX_DOMAINS_PER_WORKSPACE} allowed domains",
            )

        domain = AllowedDomain(
            workspace_id=workspace_id,
            domain=body.domain,
        )
        db.add(domain)
        db.commit()
        db.refresh(domain)
        return AllowedDomainOut.model_validate(domain)

    def delete_domain(self, db: Session, workspace_id: uuid.UUID, domain_id: uuid.UUID) -> None:
        domain = db.execute(
            select(AllowedDomain).where(
                AllowedDomain.id == domain_id,
                AllowedDomain.workspace_id == workspace_id,
            )
        ).scalar_one_or_none()
        if domain is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Allowed domain not found",
            )
        db.delete(domain)
        db.commit()


allowed_domain_service = AllowedDomainService()
