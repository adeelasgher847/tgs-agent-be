from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_member_or_admin
from app.models.user import User
from app.schemas.base import SuccessResponse
from app.schemas.recruitment_dashboard import RecruitmentDashboardData
from app.services.recruitment_dashboard_service import recruitment_dashboard_service
from app.utils.response import create_success_response

router = APIRouter()


@router.get(
    "",
    response_model=SuccessResponse[RecruitmentDashboardData],
    summary="Recruitment dashboard (TalentSync-style)",
    description="Aggregated KPIs, funnel, upcoming AI interviews, and active job rows for the current tenant.",
)
def get_recruitment_dashboard(
    user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    try:
        data = recruitment_dashboard_service.build(db, user)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    return create_success_response(data, "Recruitment dashboard data retrieved successfully")
