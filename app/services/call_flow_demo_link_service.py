"""Tenant-facing CRUD for shareable call-flow demo links.

Public consumption of a demo link (token lookup, minute-usage tracking) is a
separate, unauthenticated surface built elsewhere (app/routers/sdk.py) — this
service only covers the authenticated tenant-owner management API.
"""
from __future__ import annotations

import secrets
import uuid

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.call_flow import CallFlow
from app.models.call_flow_demo_link import CallFlowDemoLink
from app.schemas.call_flow_demo_link import (
    CallFlowDemoLinkCreate,
    CallFlowDemoLinkUpdate,
)

_TOKEN_BYTES = 32


class CallFlowDemoLinkService:
    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_flow_or_404(
        self, db: Session, flow_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> CallFlow:
        flow = db.execute(
            select(CallFlow).where(
                CallFlow.id == flow_id,
                CallFlow.tenant_id == tenant_id,
                CallFlow.is_deleted == False,  # noqa: E712
            )
        ).scalar_one_or_none()
        if flow is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Call flow {flow_id} not found",
            )
        return flow

    def _get_link_or_404(
        self, db: Session, link_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> CallFlowDemoLink:
        link = db.execute(
            select(CallFlowDemoLink).where(
                CallFlowDemoLink.id == link_id,
                CallFlowDemoLink.tenant_id == tenant_id,
            )
        ).scalar_one_or_none()
        if link is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Demo link {link_id} not found",
            )
        return link

    def build_share_url(self, token: str) -> str | None:
        base = (settings.FRONTEND_URL or "").rstrip("/")
        if not base:
            return None
        return f"{base}/demo/{token}"

    # ── Public API ────────────────────────────────────────────────────────

    def create_demo_link(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        flow_id: uuid.UUID,
        body: CallFlowDemoLinkCreate,
        created_by_user_id: uuid.UUID | None,
    ) -> CallFlowDemoLink:
        self._get_flow_or_404(db, flow_id, tenant_id)

        link = CallFlowDemoLink(
            tenant_id=tenant_id,
            call_flow_id=flow_id,
            name=body.name,
            token=secrets.token_urlsafe(_TOKEN_BYTES),
            expires_at=body.expires_at,
            per_user_limit_minutes=body.per_user_limit_minutes,
            total_budget_minutes=body.total_budget_minutes,
            created_by_user_id=created_by_user_id,
        )
        db.add(link)
        db.commit()
        db.refresh(link)
        return link

    def list_demo_links(
        self, db: Session, tenant_id: uuid.UUID, flow_id: uuid.UUID
    ) -> list[CallFlowDemoLink]:
        self._get_flow_or_404(db, flow_id, tenant_id)
        rows = db.execute(
            select(CallFlowDemoLink)
            .where(
                CallFlowDemoLink.call_flow_id == flow_id,
                CallFlowDemoLink.tenant_id == tenant_id,
            )
            .order_by(CallFlowDemoLink.created_at.desc())
        ).scalars().all()
        return list(rows)

    def update_demo_link(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        link_id: uuid.UUID,
        body: CallFlowDemoLinkUpdate,
    ) -> CallFlowDemoLink:
        link = self._get_link_or_404(db, link_id, tenant_id)

        updates = body.model_dump(exclude_unset=True)
        for field, value in updates.items():
            setattr(link, field, value)

        db.commit()
        db.refresh(link)
        return link

    def delete_demo_link(
        self, db: Session, tenant_id: uuid.UUID, link_id: uuid.UUID
    ) -> None:
        link = self._get_link_or_404(db, link_id, tenant_id)
        db.delete(link)
        db.commit()


call_flow_demo_link_service = CallFlowDemoLinkService()
