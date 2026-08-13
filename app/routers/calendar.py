from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import uuid

from app.api.deps import get_db, require_tenant
from app.models.user import User
from app.schemas.base import SuccessResponse
from typing import List
from app.schemas.calendar import BusinessHoursUpsert, BusinessHoursOut
from app.services.business_hours_service import BusinessHoursConflictError, business_hours_service
from app.utils.response import create_success_response

router = APIRouter()

# External API mapping (stable public contract):
# 0=Sunday ... 6=Saturday
# Internal storage/service mapping remains Python weekday:
# 0=Monday ... 6=Sunday
def _api_day_to_internal(day: int) -> int:
    return (day + 6) % 7


def _internal_day_to_api(day: int) -> int:
    return (day + 1) % 7


def _map_hours_payload_to_internal(payload: List[BusinessHoursUpsert]) -> List[BusinessHoursUpsert]:
    return [
        item.model_copy(update={"day_of_week": _api_day_to_internal(item.day_of_week)})
        for item in payload
    ]


def _map_hours_out_to_api(rows: List[BusinessHoursOut]) -> List[BusinessHoursOut]:
    return [
        row.model_copy(update={"day_of_week": _internal_day_to_api(row.day_of_week)})
        for row in rows
    ]


# ─── Business Hours (used by the Smart Callback Scheduler retry gate) ─────────

@router.get("/business-hours", response_model=SuccessResponse[List[BusinessHoursOut]])
def get_business_hours(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    hours = business_hours_service.get_business_hours(db, user.current_tenant_id)
    out = [BusinessHoursOut.model_validate(h) for h in hours]
    return create_success_response(data=_map_hours_out_to_api(out))


@router.post("/business-hours", response_model=SuccessResponse[List[BusinessHoursOut]], status_code=status.HTTP_201_CREATED)
def create_business_hours(
    payload: List[BusinessHoursUpsert],
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Create business hours for the tenant. Use PUT to update existing weekdays."""
    internal_payload = _map_hours_payload_to_internal(payload)
    try:
        hours = business_hours_service.create_business_hours(db, user.current_tenant_id, internal_payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except BusinessHoursConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Business hours already exist for one or more weekdays.",
                "day_of_week": [_internal_day_to_api(day) for day in exc.days],
            },
        ) from exc
    out = [BusinessHoursOut.model_validate(h) for h in hours]
    return create_success_response(data=_map_hours_out_to_api(out))


@router.put("/business-hours", response_model=SuccessResponse[List[BusinessHoursOut]])
def upsert_business_hours(
    payload: List[BusinessHoursUpsert],
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    """Set business hours for the tenant. Pass all 7 days at once (or just the ones you want to update)."""
    internal_payload = _map_hours_payload_to_internal(payload)
    hours = business_hours_service.upsert_business_hours(db, user.current_tenant_id, internal_payload)
    out = [BusinessHoursOut.model_validate(h) for h in hours]
    return create_success_response(data=_map_hours_out_to_api(out))


@router.delete("/business-hours/{business_hours_id}", response_model=SuccessResponse[dict])
def delete_business_hours(
    business_hours_id: uuid.UUID,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
):
    deleted = business_hours_service.delete_business_hours(db, business_hours_id, user.current_tenant_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Business hours not found")
    return create_success_response(data={"deleted": True, "id": str(business_hours_id)})
