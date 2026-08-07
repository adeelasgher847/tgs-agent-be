from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_readonly_or_api_key
from app.schemas.base import SuccessResponse
from app.schemas.dashboard import DashboardSummary
from app.services.dashboard_service import dashboard_service
from app.utils.response import create_success_response

router = APIRouter()


@router.get("", response_model=SuccessResponse[DashboardSummary])
async def get_dashboard_summary(
    user=Depends(require_readonly_or_api_key),
    db: Session = Depends(get_db),
):
    """All data for the dashboard home view in a single response."""
    workspace_id = user.current_tenant_id
    if workspace_id is None:
        raise HTTPException(status_code=403, detail="No tenant selected")

    summary = dashboard_service.get_dashboard_summary(db, workspace_id)
    return create_success_response(summary, "Dashboard summary fetched")
