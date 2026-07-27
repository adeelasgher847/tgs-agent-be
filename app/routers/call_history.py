from __future__ import annotations

import uuid
from datetime import datetime, date
from typing import List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_tenant
from app.models.user import User
from app.schemas.base import SuccessResponse
from app.schemas.call_history import (
    BatchCallMetrics,
    CallHistoryItem,
    CallHistoryList,
    CallHistoryMetrics,
    CallHistoryTimeSeriesPoint,
)
from app.services.call_history_service import call_history_service
from app.utils.response import create_success_response

router = APIRouter()


# ── shared filter params (re-used across several endpoints) ───────────────────

def _common_filters(
    agent_id: Optional[uuid.UUID] = Query(None, description="Filter by agent ID"),
    flow_id: Optional[uuid.UUID] = Query(None, description="Filter by call flow ID"),
    direction: Optional[Literal["inbound", "outbound"]] = Query(
        None, description="Filter by call direction"
    ),
    date_from: Optional[datetime] = Query(
        None, description="ISO 8601 UTC lower bound for call start time"
    ),
    date_to: Optional[datetime] = Query(
        None, description="ISO 8601 UTC upper bound for call start time"
    ),
    status: Optional[str] = Query(
        None, description="Filter by call status (completed, failed, no_answer, …)"
    ),
):
    return {
        "agent_id": agent_id,
        "flow_id": flow_id,
        "direction": direction,
        "date_from": date_from,
        "date_to": date_to,
        "status": status,
    }


# ── GET /calls/history ────────────────────────────────────────────────────────

@router.get(
    "/history",
    response_model=SuccessResponse[CallHistoryList],
    summary="Paginated call history list",
)
async def get_call_history(
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=200),
    sort_by: Literal["started_at", "duration", "status"] = Query("started_at"),
    sort_dir: Literal["asc", "desc"] = Query("desc"),
    filters: dict = Depends(_common_filters),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """
    Paginated call history. Date filters expect ISO 8601 UTC timestamps.
    """
    try:
        result = call_history_service.get_list(
            db,
            user.current_tenant_id,
            page=page,
            per_page=per_page,
            sort_by=sort_by,
            sort_dir=sort_dir,
            **filters,
        )
        return create_success_response(result, "Call history retrieved successfully")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── GET /calls/history/metrics ────────────────────────────────────────────────

@router.get(
    "/history/metrics",
    response_model=SuccessResponse[CallHistoryMetrics],
    summary="Aggregated call metrics",
)
async def get_call_metrics(
    filters: dict = Depends(_common_filters),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """
    Returns aggregated totals and success rate. All computation is pushed to Postgres.
    Date filters expect ISO 8601 UTC timestamps.
    """
    try:
        metrics = call_history_service.get_metrics(
            db, user.current_tenant_id, **filters
        )
        return create_success_response(metrics, "Metrics retrieved successfully")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── GET /calls/history/time-series ────────────────────────────────────────────

@router.get(
    "/history/time-series",
    response_model=SuccessResponse[List[CallHistoryTimeSeriesPoint]],
    summary="Daily call counts for charts",
)
async def get_call_time_series(
    date_from: Optional[datetime] = Query(
        None, description="ISO 8601 UTC lower bound"
    ),
    date_to: Optional[datetime] = Query(
        None, description="ISO 8601 UTC upper bound"
    ),
    agent_id: Optional[uuid.UUID] = Query(None),
    flow_id: Optional[uuid.UUID] = Query(None),
    direction: Optional[Literal["inbound", "outbound"]] = Query(None),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """
    Returns one row per calendar day (UTC). Format matches Recharts LineChart:
    [{date:'2026-05-01', total:12, completed:10, failed:2}]
    """
    try:
        series = call_history_service.get_time_series(
            db,
            user.current_tenant_id,
            date_from=date_from,
            date_to=date_to,
            agent_id=agent_id,
            flow_id=flow_id,
            direction=direction,
        )
        return create_success_response(series, "Time-series retrieved successfully")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── POST /calls/history/export ────────────────────────────────────────────────

@router.post(
    "/history/export",
    summary="Stream CSV export of call history",
    response_class=StreamingResponse,
)
async def export_call_history_csv(
    filters: dict = Depends(_common_filters),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """
    Streams a CSV file directly from the database cursor — no in-memory buffering.
    Columns: call_id, direction, from_number, to_number, agent_name, flow_name,
             status, duration_seconds, started_at, ended_at.
    Date filters expect ISO 8601 UTC timestamps.
    """
    filename = f"calls-{date.today().isoformat()}.csv"

    def _generate():
        yield from call_history_service.iter_export_csv(
            db, user.current_tenant_id, **filters
        )

    return StreamingResponse(
        _generate(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── GET /batch-calls/metrics ──────────────────────────────────────────────────
# Registered on a separate prefix in api.py; kept in this file for locality.

batch_router = APIRouter()


@batch_router.get(
    "/metrics",
    response_model=SuccessResponse[BatchCallMetrics],
    summary="Batch call campaign metrics",
)
async def get_batch_call_metrics(
    date_from: Optional[datetime] = Query(None, description="ISO 8601 UTC lower bound"),
    date_to: Optional[datetime] = Query(None, description="ISO 8601 UTC upper bound"),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """
    Returns aggregated statistics across all batch call campaigns for this tenant.
    """
    try:
        metrics = call_history_service.get_batch_metrics(
            db,
            user.current_tenant_id,
            date_from=date_from,
            date_to=date_to,
        )
        return create_success_response(metrics, "Batch metrics retrieved successfully")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
