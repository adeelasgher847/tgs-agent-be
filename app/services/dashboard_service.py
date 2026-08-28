"""Composes the dashboard home-view summary from existing domain services.

No business logic lives here beyond simple aggregation/wiring — the actual
metric computation, credit lookup, and flow serialization are all delegated
to the services/repositories that already own them.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.call_session import CallSession
from app.models.kb_file import KbFile
from app.models.knowledge_base_document import KnowledgeBase
from app.repositories.call_flow_repository import CallFlowRepository
from app.schemas.call_flow import CallFlowListItem
from app.schemas.dashboard import DashboardKbItem, DashboardRecentCallItem, DashboardSummary
from app.services.call_flow_service import call_flow_service
from app.services.call_history_service import call_history_service
from app.services.credit_service import credit_service

RECENT_ITEMS_LIMIT = 4


class DashboardService:

    def get_dashboard_summary(self, db: Session, tenant_id: uuid.UUID) -> DashboardSummary:
        month_start = datetime.now(timezone.utc).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )

        metrics = call_history_service.get_metrics(db, tenant_id, date_from=month_start)
        credits = credit_service.get_tenant_credits(db, tenant_id)

        return DashboardSummary(
            monthly_spent=int(round(float(metrics.total_cost or 0))),
            monthly_calls=int(metrics.total_calls or 0),
            success_rate_percent=(
                int(round(float(metrics.success_rate_percent)))
                if metrics.success_rate_percent is not None
                else None
            ),
            credits=round(credits, 2),
            recent_call_flows=self._get_recent_call_flows(db, tenant_id),
            recent_knowledge_bases=self._get_recent_knowledge_bases(db, tenant_id),
            recent_calls=self._get_recent_calls(db, tenant_id),
        )

    # ── recent call flows ────────────────────────────────────────────────────

    def _get_recent_call_flows(
        self, db: Session, tenant_id: uuid.UUID
    ) -> list[CallFlowListItem]:
        repo = CallFlowRepository(db)
        flows, _total = repo.find_by_workspace(tenant_id, page=1, limit=RECENT_ITEMS_LIMIT)
        folder_ids_map = repo.find_folder_ids_map([flow.id for flow in flows])
        return [
            call_flow_service.flow_to_list_item_model(flow, folder_ids_map.get(flow.id, []))
            for flow in flows
        ]

    # ── recent knowledge bases ───────────────────────────────────────────────

    def _get_recent_knowledge_bases(
        self, db: Session, tenant_id: uuid.UUID
    ) -> list[DashboardKbItem]:
        kbs = (
            db.execute(
                select(KnowledgeBase)
                .where(KnowledgeBase.workspace_id == tenant_id)
                .order_by(KnowledgeBase.created_at.desc())
                .limit(RECENT_ITEMS_LIMIT)
            )
            .scalars()
            .all()
        )

        kb_ids = [kb.id for kb in kbs]
        file_counts: dict[uuid.UUID, int] = {}
        if kb_ids:
            rows = db.execute(
                select(KbFile.kb_id, func.count(KbFile.id))
                .where(KbFile.kb_id.in_(kb_ids), KbFile.status == "ready")
                .group_by(KbFile.kb_id)
            ).all()
            file_counts = {kb_id: count for kb_id, count in rows}

        return [
            DashboardKbItem(
                id=kb.id,
                name=kb.name,
                description=kb.description,
                file_count=file_counts.get(kb.id, 0),
                created_at=kb.created_at,
            )
            for kb in kbs
        ]

    # ── recent calls ─────────────────────────────────────────────────────────

    def _get_recent_calls(
        self, db: Session, tenant_id: uuid.UUID
    ) -> list[DashboardRecentCallItem]:
        calls = (
            db.execute(
                select(CallSession)
                .where(CallSession.tenant_id == tenant_id)
                .order_by(CallSession.created_at.desc())
                .limit(RECENT_ITEMS_LIMIT)
            )
            .scalars()
            .all()
        )

        return [
            DashboardRecentCallItem(
                id=call.id,
                status=call.status,
                summary=call.transcript_summary,
                call_type=call.call_type,
                duration=call.duration,
                cost=call.cost,
                created_at=call.created_at,
            )
            for call in calls
        ]


dashboard_service = DashboardService()
