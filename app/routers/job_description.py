import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_manager, require_readonly
from app.models.user import User
from app.schemas.base import SuccessResponse
from app.schemas.job_description import (
    JobDescriptionCreateManual,
    JobDescriptionListOut,
    JobDescriptionOut,
    JobDescriptionUpdate,
)
from app.services.job_description_service import job_description_service
from app.utils.response import create_success_response

router = APIRouter()


@router.post("", response_model=SuccessResponse[JobDescriptionOut], status_code=status.HTTP_201_CREATED)
def create_job_description_manual(
    payload: JobDescriptionCreateManual,
    user: User = Depends(require_manager),
    db: Session = Depends(get_db),
):
    jd = job_description_service.create_manual(
        db=db,
        payload=payload,
        tenant_id=user.current_tenant_id,
        user_id=user.id,
    )
    jd = job_description_service.process_upload(
        db=db,
        job_description_id=jd.id,
        tenant_id=user.current_tenant_id,
        user_id=user.id,
    )
    return create_success_response(
        JobDescriptionOut.model_validate(jd),
        "Job description created successfully",
        status.HTTP_201_CREATED,
    )


@router.post("/upload", response_model=SuccessResponse[JobDescriptionOut], status_code=status.HTTP_201_CREATED)
async def upload_job_description(
    file: UploadFile = File(..., description="JD file: pdf/docx/txt"),
    user: User = Depends(require_manager),
    db: Session = Depends(get_db),
):
    filename = file.filename or ""
    allowed = (".pdf", ".docx", ".txt")
    if not filename.lower().endswith(allowed):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file type")

    content = await file.read()
    jd = job_description_service.create_upload(
        db=db,
        filename=filename,
        file_content=content,
        tenant_id=user.current_tenant_id,
        user_id=user.id,
    )

    jd = job_description_service.process(
        db=db,
        job_description_id=jd.id,
        tenant_id=user.current_tenant_id,
        user_id=user.id,
    )

    return create_success_response(
        JobDescriptionOut.model_validate(jd),
        "Job description uploaded successfully",
        status.HTTP_201_CREATED,
    )


@router.get("", response_model=SuccessResponse[list[JobDescriptionListOut]])
def list_job_descriptions(
    user: User = Depends(require_readonly),
    db: Session = Depends(get_db),
):
    """List job descriptions"""
    if not user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")
    rows = job_description_service.list_by_tenant(db=db, tenant_id=user.current_tenant_id)
    out: list[JobDescriptionListOut] = []
    for jd in rows:
        job_description_service.normalize_for_read_response(jd)
        out.append(JobDescriptionListOut.model_validate(jd))
    return create_success_response(out, "Job descriptions retrieved successfully")


@router.patch("/{job_description_id}", response_model=SuccessResponse[JobDescriptionOut])
def update_job_description(
    job_description_id: uuid.UUID,
    payload: JobDescriptionUpdate,
    user: User = Depends(require_manager),
    db: Session = Depends(get_db),
):
    jd = job_description_service.update(
        db=db,
        job_description_id=job_description_id,
        payload=payload,
        tenant_id=user.current_tenant_id,
        user_id=user.id,
    )
    job_description_service.normalize_for_read_response(jd)
    return create_success_response(
        JobDescriptionOut.model_validate(jd),
        "Job description updated successfully",
    )


@router.delete("/{job_description_id}", response_model=SuccessResponse[dict])
def delete_job_description(
    job_description_id: uuid.UUID,
    user: User = Depends(require_manager),
    db: Session = Depends(get_db),
):
    job_description_service.delete(
        db=db,
        job_description_id=job_description_id,
        tenant_id=user.current_tenant_id,
    )
    return create_success_response(
        {"id": str(job_description_id)},
        "Job description deleted successfully",
    )


@router.get("/{job_description_id}", response_model=SuccessResponse[JobDescriptionListOut])
def get_job_description(
    job_description_id: uuid.UUID,
    user: User = Depends(require_readonly),
    db: Session = Depends(get_db),
):
    """Single job description: DB read only. No LLM."""
    if not user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")
    jd = job_description_service.get_by_id(
        db=db,
        job_description_id=job_description_id,
        tenant_id=user.current_tenant_id,
    )
    job_description_service.normalize_for_read_response(jd)
    return create_success_response(JobDescriptionListOut.model_validate(jd), "Job description fetched successfully")


@router.get("/{job_description_id}/status", response_model=SuccessResponse[dict])
def get_job_description_status(
    job_description_id: uuid.UUID,
    user: User = Depends(require_readonly),
    db: Session = Depends(get_db),
):
    if not user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")
    processing_status = job_description_service.get_status(
        db=db,
        job_description_id=job_description_id,
        tenant_id=user.current_tenant_id,
    )
    return create_success_response(
        {"id": str(job_description_id), "processing_status": processing_status},
        "Job description status retrieved successfully",
    )
