import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin_or_owner, require_member_or_admin, require_tenant
from app.models.user import User
from app.schemas.base import SuccessResponse
from app.schemas.job_description import JobDescriptionCreateManual, JobDescriptionOut
from app.services.job_description_service import job_description_service
from app.utils.response import create_success_response

router = APIRouter()


@router.post("/", response_model=SuccessResponse[JobDescriptionOut], status_code=status.HTTP_201_CREATED)
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
    jd = job_description_service.process(
        db=db,
        job_description_id=jd.id,
        tenant_id=admin_user.current_tenant_id,
        user_id=admin_user.id,
    )
    return create_success_response(jd, "Job description created successfully", status.HTTP_201_CREATED)


@router.post("/upload", response_model=SuccessResponse[JobDescriptionOut], status_code=status.HTTP_201_CREATED)
async def upload_job_description(
    file: UploadFile = File(..., description="JD file: pdf/docx/txt"),
    tenant_user: User = Depends(require_tenant),
    admin_user: User = Depends(require_admin_or_owner),
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
        tenant_id=admin_user.current_tenant_id,
        user_id=admin_user.id,
    )

    jd = job_description_service.process(
        db=db,
        job_description_id=jd.id,
        tenant_id=admin_user.current_tenant_id,
        user_id=admin_user.id,
    )

    return create_success_response(jd, "Job description uploaded successfully", status.HTTP_201_CREATED)


@router.get("/{job_description_id}", response_model=SuccessResponse[JobDescriptionOut])
def get_job_description(
    job_description_id: uuid.UUID,
    tenant_user: User = Depends(require_tenant),
    user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    jd = job_description_service.get_by_id(db=db, job_description_id=job_description_id, tenant_id=user.current_tenant_id)
    # Backward-compatibility: older rows may have NULL JSON fields.
    jd.required_skills = jd.required_skills or []
    jd.key_responsibilities = jd.key_responsibilities or []
    jd.required_certifications = jd.required_certifications or []
    jd.extracted_skills = jd.extracted_skills or []
    jd.keywords = jd.keywords or []
    jd.skill_weight_matrix = jd.skill_weight_matrix or {}
    jd.matching_criteria = jd.matching_criteria or {}
    return create_success_response(jd, "Job description retrieved successfully")


@router.post("/{job_description_id}/process", response_model=SuccessResponse[JobDescriptionOut])
def process_job_description(
    job_description_id: uuid.UUID,
    tenant_user: User = Depends(require_tenant),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    jd = job_description_service.process(
        db=db,
        job_description_id=job_description_id,
        tenant_id=admin_user.current_tenant_id,
        user_id=admin_user.id,
    )
    return create_success_response(jd, "Job description processed successfully")


@router.get("/{job_description_id}/status", response_model=SuccessResponse[dict])
def get_job_description_status(
    job_description_id: uuid.UUID,
    tenant_user: User = Depends(require_tenant),
    user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    processing_status = job_description_service.get_status(
        db=db,
        job_description_id=job_description_id,
        tenant_id=user.current_tenant_id,
    )
    return create_success_response(
        {"id": str(job_description_id), "processing_status": processing_status},
        "Job description status retrieved successfully",
    )
