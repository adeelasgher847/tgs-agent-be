from __future__ import annotations

import csv
import io
import math
import uuid
from datetime import datetime
from typing import AsyncIterator, List, Optional

from sqlalchemy import case, cast, func, select, text
from sqlalchemy.orm import Session

from app.models.agent import Agent
from app.models.batch_job import BatchJob
from app.models.call_flow import CallFlow
from app.models.call_session import CallSession
from app.schemas.call_history import (
    BatchCallMetrics,
    CallHistoryItem,
    CallHistoryList,
    CallHistoryMetrics,
    CallHistoryTimeSeriesPoint,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _base_filter(
    stmt,
    tenant_id: uuid.UUID,
    *,
    agent_id: Optional[uuid.UUID] = None,
    flow_id: Optional[uuid.UUID] = None,
    direction: Optional[str] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    status: Optional[str] = None,
):
    """Apply the common WHERE predicates to any statement that touches callsession."""
    stmt = stmt.where(CallSession.tenant_id == tenant_id)
    if agent_id:
        stmt = stmt.where(CallSession.agent_id == agent_id)
    if flow_id:
        stmt = stmt.where(CallSession.call_flow_id == flow_id)
    if direction:
        stmt = stmt.where(CallSession.call_type == direction)
    if date_from:
        stmt = stmt.where(CallSession.start_time >= date_from)
    if date_to:
        stmt = stmt.where(CallSession.start_time <= date_to)
    if status:
        stmt = stmt.where(CallSession.status == status)
    return stmt


# ── service ───────────────────────────────────────────────────────────────────

class CallHistoryService:

    # ── metrics ───────────────────────────────────────────────────────────────

    def get_metrics(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        *,
        agent_id: Optional[uuid.UUID] = None,
        flow_id: Optional[uuid.UUID] = None,
        direction: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        status: Optional[str] = None,
    ) -> CallHistoryMetrics:
        completed_expr = case((CallSession.status == "completed", 1), else_=None)
        failed_expr = case((CallSession.status == "failed", 1), else_=None)
        no_answer_expr = case((CallSession.status == "no_answer", 1), else_=None)

        stmt = select(
            func.count().label("total_calls"),
            func.count(completed_expr).label("completed"),
            func.count(failed_expr).label("failed"),
            func.count(no_answer_expr).label("no_answer"),
            func.avg(CallSession.duration).label("avg_duration_seconds"),
            func.sum(CallSession.duration).label("total_duration_seconds"),
            # success_rate computed entirely in Postgres
            func.round(
                100.0
                * func.count(completed_expr)
                / func.nullif(func.count(), 0),
                2,
            ).label("success_rate_percent"),
        ).select_from(CallSession)

        stmt = _base_filter(
            stmt,
            tenant_id,
            agent_id=agent_id,
            flow_id=flow_id,
            direction=direction,
            date_from=date_from,
            date_to=date_to,
            status=status,
        )

        row = db.execute(stmt).one()

        return CallHistoryMetrics(
            total_calls=row.total_calls or 0,
            completed=row.completed or 0,
            failed=row.failed or 0,
            no_answer=row.no_answer or 0,
            avg_duration_seconds=(
                float(round(row.avg_duration_seconds, 2))
                if row.avg_duration_seconds is not None
                else None
            ),
            total_duration_seconds=(
                int(row.total_duration_seconds)
                if row.total_duration_seconds is not None
                else None
            ),
            success_rate_percent=(
                float(row.success_rate_percent)
                if row.success_rate_percent is not None
                else None
            ),
        )

    # ── time-series ───────────────────────────────────────────────────────────

    def get_time_series(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        *,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        agent_id: Optional[uuid.UUID] = None,
        flow_id: Optional[uuid.UUID] = None,
        direction: Optional[str] = None,
    ) -> List[CallHistoryTimeSeriesPoint]:
        date_col = func.date(CallSession.start_time).label("date")
        completed_expr = case((CallSession.status == "completed", 1), else_=None)
        failed_expr = case((CallSession.status == "failed", 1), else_=None)

        stmt = (
            select(
                date_col,
                func.count().label("total"),
                func.count(completed_expr).label("completed"),
                func.count(failed_expr).label("failed"),
            )
            .select_from(CallSession)
            .group_by(date_col)
            .order_by(date_col)
        )

        stmt = _base_filter(
            stmt,
            tenant_id,
            agent_id=agent_id,
            flow_id=flow_id,
            direction=direction,
            date_from=date_from,
            date_to=date_to,
        )

        rows = db.execute(stmt).all()

        return [
            CallHistoryTimeSeriesPoint(
                date=str(row.date),
                total=row.total,
                completed=row.completed,
                failed=row.failed,
            )
            for row in rows
        ]

    # ── paginated list ────────────────────────────────────────────────────────

    _SORT_COLUMNS = {
        "started_at": CallSession.start_time,
        "duration": CallSession.duration,
        "status": CallSession.status,
    }

    def get_list(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        *,
        page: int = 1,
        per_page: int = 25,
        sort_by: str = "started_at",
        sort_dir: str = "desc",
        agent_id: Optional[uuid.UUID] = None,
        flow_id: Optional[uuid.UUID] = None,
        direction: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        status: Optional[str] = None,
    ) -> CallHistoryList:
        sort_col = self._SORT_COLUMNS.get(sort_by, CallSession.start_time)
        order_expr = sort_col.desc() if sort_dir == "desc" else sort_col.asc()

        # count
        count_stmt = select(func.count()).select_from(CallSession)
        count_stmt = _base_filter(
            count_stmt,
            tenant_id,
            agent_id=agent_id,
            flow_id=flow_id,
            direction=direction,
            date_from=date_from,
            date_to=date_to,
            status=status,
        )
        total: int = db.execute(count_stmt).scalar_one()

        # data — LEFT JOIN agent and call_flow for names
        stmt = (
            select(
                CallSession.id.label("call_id"),
                CallSession.call_type.label("direction"),
                CallSession.from_number,
                CallSession.to_number,
                Agent.name.label("agent_name"),
                CallFlow.name.label("flow_name"),
                CallSession.status,
                CallSession.duration.label("duration_seconds"),
                CallSession.start_time.label("started_at"),
                CallSession.end_time.label("ended_at"),
            )
            .select_from(CallSession)
            .outerjoin(Agent, Agent.id == CallSession.agent_id)
            .outerjoin(CallFlow, CallFlow.id == CallSession.call_flow_id)
            .order_by(order_expr)
            .offset((page - 1) * per_page)
            .limit(per_page)
        )

        stmt = _base_filter(
            stmt,
            tenant_id,
            agent_id=agent_id,
            flow_id=flow_id,
            direction=direction,
            date_from=date_from,
            date_to=date_to,
            status=status,
        )

        rows = db.execute(stmt).all()

        items = [
            CallHistoryItem(
                call_id=row.call_id,
                direction=row.direction,
                from_number=row.from_number,
                to_number=row.to_number,
                agent_name=row.agent_name,
                flow_name=row.flow_name,
                status=row.status,
                duration_seconds=row.duration_seconds,
                started_at=row.started_at,
                ended_at=row.ended_at,
            )
            for row in rows
        ]

        return CallHistoryList(
            items=items,
            total=total,
            page=page,
            per_page=per_page,
            pages=math.ceil(total / per_page) if total else 0,
        )

    # ── CSV streaming ─────────────────────────────────────────────────────────

    CSV_COLUMNS = [
        "call_id",
        "direction",
        "from_number",
        "to_number",
        "agent_name",
        "flow_name",
        "status",
        "duration_seconds",
        "started_at",
        "ended_at",
    ]

    def iter_export_csv(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        *,
        agent_id: Optional[uuid.UUID] = None,
        flow_id: Optional[uuid.UUID] = None,
        direction: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        status: Optional[str] = None,
    ):
        """Yield CSV lines (str) one at a time — header first, then one row per call."""
        stmt = (
            select(
                CallSession.id.label("call_id"),
                CallSession.call_type.label("direction"),
                CallSession.from_number,
                CallSession.to_number,
                Agent.name.label("agent_name"),
                CallFlow.name.label("flow_name"),
                CallSession.status,
                CallSession.duration.label("duration_seconds"),
                CallSession.start_time.label("started_at"),
                CallSession.end_time.label("ended_at"),
            )
            .select_from(CallSession)
            .outerjoin(Agent, Agent.id == CallSession.agent_id)
            .outerjoin(CallFlow, CallFlow.id == CallSession.call_flow_id)
            .order_by(CallSession.start_time.desc())
            .execution_options(yield_per=500)
        )

        stmt = _base_filter(
            stmt,
            tenant_id,
            agent_id=agent_id,
            flow_id=flow_id,
            direction=direction,
            date_from=date_from,
            date_to=date_to,
            status=status,
        )

        buf = io.StringIO()
        writer = csv.writer(buf)

        # header
        writer.writerow(self.CSV_COLUMNS)
        yield buf.getvalue()

        for row in db.execute(stmt):
            buf.seek(0)
            buf.truncate()
            writer.writerow([
                str(row.call_id),
                row.direction or "",
                row.from_number or "",
                row.to_number or "",
                row.agent_name or "",
                row.flow_name or "",
                row.status or "",
                row.duration_seconds if row.duration_seconds is not None else "",
                row.started_at.isoformat() if row.started_at else "",
                row.ended_at.isoformat() if row.ended_at else "",
            ])
            yield buf.getvalue()

    # ── batch metrics ─────────────────────────────────────────────────────────

    def get_batch_metrics(
        self,
        db: Session,
        tenant_id: uuid.UUID,
        *,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> BatchCallMetrics:
        stmt = select(
            func.count().label("total_batches"),
            func.round(
                func.avg(
                    BatchJob.completed_count
                    * 100.0
                    / func.nullif(BatchJob.total_count, 0)
                ),
                2,
            ).label("avg_completion_rate_percent"),
            func.coalesce(func.sum(BatchJob.total_count), 0).label("total_calls_dispatched"),
            func.coalesce(func.sum(BatchJob.failed_count), 0).label("total_failed"),
        ).select_from(BatchJob).where(BatchJob.workspace_id == tenant_id)

        if date_from:
            stmt = stmt.where(BatchJob.created_at >= date_from)
        if date_to:
            stmt = stmt.where(BatchJob.created_at <= date_to)

        row = db.execute(stmt).one()

        return BatchCallMetrics(
            total_batches=row.total_batches or 0,
            avg_completion_rate_percent=(
                float(row.avg_completion_rate_percent)
                if row.avg_completion_rate_percent is not None
                else None
            ),
            total_calls_dispatched=int(row.total_calls_dispatched),
            total_failed=int(row.total_failed),
        )


call_history_service = CallHistoryService()
