import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin_or_owner, require_member_or_admin, require_tenant
from app.models.user import User
from app.schemas.base import SuccessResponse
from app.schemas.job_description import JobDescriptionCreateManual, JobDescriptionOut, JobDescriptionUpdate
from app.services.job_description_service import job_description_service
from app.utils.response import create_success_response

router = APIRouter()


@router.post("/manual", response_model=SuccessResponse[JobDescriptionOut], status_code=status.HTTP_201_CREATED)
def create_job_description_manual(
    payload: JobDescriptionCreateManual,
    tenant_user: User = Depends(require_tenant),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    jd = job_description_service.create_manual(
        db=db,
        payload=payload,
        tenant_id=admin_user.current_tenant_id,
        user_id=admin_user.id,
    )
    return create_success_response(jd, "Job description created successfully", status.HTTP_201_CREATED)


@router.get("/", response_model=SuccessResponse[list[JobDescriptionOut]])
def list_job_descriptions(
    tenant_user: User = Depends(require_tenant),
    user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    rows = job_description_service.list_by_tenant(db=db, tenant_id=user.current_tenant_id)
    return create_success_response(rows, "Job descriptions retrieved successfully")


@router.get("/{job_description_id}", response_model=SuccessResponse[JobDescriptionOut])
def get_job_description(
    job_description_id: uuid.UUID,
    tenant_user: User = Depends(require_tenant),
    user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    jd = job_description_service.get_by_id(db=db, job_description_id=job_description_id, tenant_id=user.current_tenant_id)
    return create_success_response(jd, "Job description retrieved successfully")


@router.put("/{job_description_id}", response_model=SuccessResponse[JobDescriptionOut])
def update_job_description(
    job_description_id: uuid.UUID,
    payload: JobDescriptionUpdate,
    tenant_user: User = Depends(require_tenant),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    jd = job_description_service.update(
        db=db,
        job_description_id=job_description_id,
        payload=payload,
        tenant_id=admin_user.current_tenant_id,
        user_id=admin_user.id,
    )
    return create_success_response(jd, "Job description updated successfully")
