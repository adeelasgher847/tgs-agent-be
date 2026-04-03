from __future__ import annotations

import logging
import uuid
from pathlib import Path
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin_or_owner, require_member_or_admin
from app.core.config import settings
from app.models.job_description import JobDescription
from app.models.resume import ParseStatus, Resume, UploadMode
from app.models.user import User
from app.schemas.base import SuccessResponse
from app.schemas.resume import (
    BatchShortlistItem,
    BatchShortlistPayload,
    MatchMode,
    MatchRequest,
    ParseMode,
    ParseStatusEnum,
    ParsedResume,
    ResumeListItem,
    ResumeStatusResponse,
)
from app.services.resume_matching_service import score_candidate
from app.services.resume_parse_service import run_parse_for_resume
from app.utils.fit_score_labels import explain_fit_score
from app.utils.response import create_success_response

router = APIRouter()
log = logging.getLogger(__name__)


def optional_batch_id_query(
    batch_id: str | None = Query(
        None,
        description=(
            "Optional. Filter by upload batch UUID from multi-upload. "
            "Omit this parameter or leave it empty to return all resumes for the tenant (up to `limit`)."
        ),
    ),
) -> UUID | None:
    """Avoid 422 when clients send batch_id= empty; treat as 'no filter'."""
    if batch_id is None:
        return None
    s = batch_id.strip()
    if not s:
        return None
    try:
        return UUID(s)
    except ValueError:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Invalid batch_id: expected a UUID, got {batch_id!r}",
        ) from None


def _ensure_upload_dir() -> Path:
    upload_root = Path(getattr(settings, "RESUME_UPLOAD_DIR", "./uploads/resumes"))
    upload_root.mkdir(parents=True, exist_ok=True)
    return upload_root


def _allowed_extensions() -> set[str]:
    exts = getattr(settings, "RESUME_ALLOWED_EXTENSIONS", None)
    if exts:
        return set(exts)
    return {".pdf", ".docx", ".txt"}


def _max_upload_bytes() -> int:
    return int(getattr(settings, "RESUME_MAX_UPLOAD_BYTES", 5 * 1024 * 1024))


async def _store_single_file_and_create_resume(
    file: UploadFile,
    tenant_id: uuid.UUID,
    db: Session,
    upload_mode: UploadMode = UploadMode.SINGLE,
    batch_id: uuid.UUID | None = None,
) -> Resume:
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Filename required")
    ext = Path(file.filename).suffix.lower()
    allowed = _allowed_extensions()
    if ext not in allowed:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Allowed types: {sorted(allowed)}",
        )

    upload_root = _ensure_upload_dir()

    temp_path = upload_root / f"_{uuid.uuid4()}{ext}"
    size = 0
    try:
        with temp_path.open("wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > _max_upload_bytes():
                    raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "File too large")
                buffer.write(chunk)

        resume = Resume(
            tenant_id=tenant_id,
            original_filename=file.filename,
            content_type=file.content_type or "application/octet-stream",
            storage_path="",
            status=ParseStatus.PENDING,
            upload_mode=upload_mode,
            batch_id=batch_id,
        )
        db.add(resume)
        db.flush()
        final_path = upload_root / f"{resume.id}{ext}"
        temp_path.rename(final_path)
        resume.storage_path = str(final_path.resolve())
        db.flush()
        return resume
    except HTTPException:
        temp_path.unlink(missing_ok=True)
        raise
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


@router.post(
    "/upload",
    response_model=SuccessResponse[dict],
    status_code=status.HTTP_201_CREATED,
)
async def upload_resume(
    file: UploadFile = File(...),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    """
    Upload a single resume file for the current tenant.
    """
    if not admin_user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")
    resume = await _store_single_file_and_create_resume(
        file,
        admin_user.current_tenant_id,
        db,
        upload_mode=UploadMode.SINGLE,
    )
    db.commit()
    db.refresh(resume)
    payload = {
        "resume_id": str(resume.id),
        "status": resume.status.value,
        "upload_mode": resume.upload_mode.value if resume.upload_mode else None,
        "batch_id": str(resume.batch_id) if resume.batch_id else None,
    }
    return create_success_response(payload, "Resume uploaded successfully", status.HTTP_201_CREATED)


@router.post(
    "/upload-batch",
    response_model=SuccessResponse[dict],
    status_code=status.HTTP_201_CREATED,
    include_in_schema=False,
)
async def upload_resumes_batch(
    files: list[UploadFile] = File(...),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    return await _upload_resumes_batch_impl(files, admin_user, db)


@router.post(
    "/upload-multiple",
    response_model=SuccessResponse[dict],
    status_code=status.HTTP_201_CREATED,
)
async def upload_resumes_multiple(
    files: list[UploadFile] = File(...),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    """
    Public batch endpoint for uploading multiple resumes.
    """
    return await _upload_resumes_batch_impl(files, admin_user, db)


async def _upload_resumes_batch_impl(
    files: list[UploadFile],
    admin_user: User,
    db: Session,
) -> SuccessResponse[dict]:
    """
    Upload multiple resume files in one request for the current tenant.
    """
    if not admin_user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")

    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "At least one file is required")

    results: list[dict] = []
    success_count = 0
    error_count = 0
    batch_id = uuid.uuid4()
    for f in files:
        try:
            resume = await _store_single_file_and_create_resume(
                f,
                admin_user.current_tenant_id,
                db,
                upload_mode=UploadMode.BATCH,
                batch_id=batch_id,
            )
            db.commit()
            db.refresh(resume)
            results.append(
                {
                    "original_filename": resume.original_filename,
                    "resume_id": str(resume.id),
                    "status": resume.status.value,
                    "upload_mode": resume.upload_mode.value,
                    "batch_id": str(resume.batch_id) if resume.batch_id else None,
                    "error": None,
                }
            )
            success_count += 1
        except HTTPException as exc:
            db.rollback()
            results.append(
                {
                    "original_filename": f.filename,
                    "resume_id": None,
                    "status": "FAILED",
                    "upload_mode": UploadMode.BATCH.value,
                    "batch_id": str(batch_id),
                    "error": str(exc.detail),
                }
            )
            error_count += 1
        except Exception as exc:
            db.rollback()
            results.append(
                {
                    "original_filename": f.filename,
                    "resume_id": None,
                    "status": "FAILED",
                    "upload_mode": UploadMode.BATCH.value,
                    "batch_id": str(batch_id),
                    "error": str(exc),
                }
            )
            error_count += 1

    payload = {
        "batch_id": str(batch_id),
        "items": results,
        "count": len(results),
        "success_count": success_count,
        "error_count": error_count,
    }
    return create_success_response(payload, "Resumes uploaded successfully", status.HTTP_201_CREATED)


@router.post(
    "/{resume_id}/parse",
    response_model=SuccessResponse[dict],
)
def parse(
    resume_id: UUID,
    parse_mode: ParseMode = Form(default=ParseMode.hybrid),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    if not admin_user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")
    res = db.query(Resume).filter(
        Resume.id == resume_id, Resume.tenant_id == admin_user.current_tenant_id
    ).first()
    if res is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Resume not found")
    res = run_parse_for_resume(db, resume_id, parse_mode=parse_mode)
    db.commit()
    db.refresh(res)
    payload = resume_detail_payload(res)
    return create_success_response(payload, "Resume parsed successfully")


@router.post(
    "/parse-batch",
    response_model=SuccessResponse[dict],
)
def parse_batch(
    resume_ids: list[str] = Form(...),
    parse_mode: ParseMode = Form(default=ParseMode.hybrid),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    """
    Parse multiple resumes in one call.
    Returns per-resume status without failing the whole batch.
    """
    if not admin_user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")
    parsed_resume_ids: list[UUID] = []
    for raw in resume_ids:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        for part in parts:
            try:
                parsed_resume_ids.append(UUID(part))
            except ValueError:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST,
                    f"Invalid resume_id: '{part}'",
                ) from None

    if not parsed_resume_ids:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "At least one resume_id is required")

    items: list[dict] = []
    for rid in parsed_resume_ids:
        res = db.query(Resume).filter(
            Resume.id == rid, Resume.tenant_id == admin_user.current_tenant_id
        ).first()
        if res is None:
            items.append(
                {
                    "resume_id": str(rid),
                    "status": "NOT_FOUND",
                    "confidence_score": None,
                    "warnings": [],
                    "error_message": "Resume not found",
                }
            )
            continue
        try:
            res = run_parse_for_resume(db, rid, parse_mode=parse_mode)
            db.commit()
            db.refresh(res)
            payload = resume_detail_payload(res)
            items.append(
                {
                    "resume_id": payload["resume_id"],
                    "status": payload["status"],
                    "confidence_score": payload["confidence_score"],
                    "warnings": payload["warnings"],
                    "error_message": payload["error_message"],
                }
            )
        except HTTPException as http_exc:
            db.rollback()
            items.append(
                {
                    "resume_id": str(rid),
                    "status": f"ERROR_{http_exc.status_code}",
                    "confidence_score": None,
                    "warnings": [],
                    "error_message": http_exc.detail,
                }
            )
        except Exception as exc:  # defensive
            db.rollback()
            items.append(
                {
                    "resume_id": str(rid),
                    "status": "ERROR",
                    "confidence_score": None,
                    "warnings": [],
                    "error_message": str(exc),
                }
            )

    payload = {
        "items": items,
        "count": len(items),
        "parse_mode": parse_mode.value,
    }
    return create_success_response(payload, "Batch parse completed")


@router.get(
    "",
    response_model=SuccessResponse[list[ResumeListItem]],
)
def list_resumes(
    limit: int = Query(50, ge=1, le=200, description="Maximum number of resumes to return"),
    batch_id: UUID | None = Depends(optional_batch_id_query),
    user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    if not user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")
    q = db.query(Resume).filter(Resume.tenant_id == user.current_tenant_id)
    if batch_id is not None:
        q = q.filter(Resume.batch_id == batch_id)
    q = q.order_by(Resume.created_at.desc()).limit(min(limit, 200))
    rows = q.all()
    items = [
        ResumeListItem(
            id=r.id,
            original_filename=r.original_filename,
            status=ParseStatusEnum(r.status.value),
            parse_confidence=r.parse_confidence,
            created_at=r.created_at,
            upload_mode=r.upload_mode.value if r.upload_mode else None,
            batch_id=r.batch_id,
        )
        for r in rows
    ]
    return create_success_response(items, "Resumes retrieved successfully")


@router.post(
    "/shortlist-by-batch",
    response_model=SuccessResponse[BatchShortlistPayload],
)
def shortlist_batch_against_job(
    batch_id: UUID = Query(..., description="Upload batch id from multi-upload response"),
    job_description_id: UUID = Query(..., description="Job description to score candidates against"),
    top_k: int | None = Query(
        None,
        ge=1,
        le=200,
        description="Return only the top K candidates after ranking (omit for all scored resumes)",
    ),
    min_overall_score: float | None = Query(
        None,
        ge=0.0,
        le=1.0,
        description="Drop candidates below this overall score before applying top_k",
    ),
    max_resumes: int = Query(
        500,
        ge=1,
        le=2000,
        description="Safety cap: maximum resumes from this batch to evaluate",
    ),
    match_mode: MatchMode | None = Query(
        None,
        description="rules | ai | hybrid; omit to use server RECRUIT_MATCH_MODE (default hybrid)",
    ),
    user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    if not user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")

    job = db.query(JobDescription).filter(
        JobDescription.id == job_description_id,
        JobDescription.tenant_id == user.current_tenant_id,
    ).first()
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job description not found")

    rows = (
        db.query(Resume)
        .filter(
            Resume.tenant_id == user.current_tenant_id,
            Resume.batch_id == batch_id,
        )
        .order_by(Resume.created_at.asc())
        .limit(max_resumes)
        .all()
    )

    if not rows:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No resumes found for this batch in the current tenant",
        )

    items_work: list[BatchShortlistItem] = []
    not_scored = 0

    for res in rows:
        if res.status != ParseStatus.READY or not res.parsed_json:
            not_scored += 1
            continue
        try:
            parsed = ParsedResume.model_validate(res.parsed_json)
            mm = match_mode.value if match_mode else None
            result = score_candidate(res.id, job, parsed, match_mode=mm)
            pct, fit_label, fit_summary = explain_fit_score(float(result.overall_score))
            items_work.append(
                BatchShortlistItem(
                    resume_id=res.id,
                    filename=res.original_filename,
                    score=float(result.overall_score),
                    match_percent=pct,
                    fit_label=fit_label,
                    fit_summary=fit_summary,
                )
            )
        except Exception:
            not_scored += 1

    items_work.sort(key=lambda x: (-x.score, str(x.resume_id)))

    if min_overall_score is not None:
        items_work = [it for it in items_work if it.score >= min_overall_score]

    if top_k is not None:
        items_work = items_work[:top_k]

    payload = BatchShortlistPayload(items=items_work, not_scored_count=not_scored)
    return create_success_response(payload, "Batch shortlist computed successfully")


@router.get(
    "/{resume_id}",
    response_model=SuccessResponse[dict],
)
def get_resume(
    resume_id: UUID,
    user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    if not user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")
    res = db.query(Resume).filter(
        Resume.id == resume_id, Resume.tenant_id == user.current_tenant_id
    ).first()
    if res is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Resume not found")
    payload = resume_detail_payload(res)
    return create_success_response(payload, "Resume retrieved successfully")


@router.get(
    "/{resume_id}/status",
    response_model=SuccessResponse[ResumeStatusResponse],
)
def get_status(
    resume_id: UUID,
    user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    if not user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")
    res = db.query(Resume).filter(
        Resume.id == resume_id, Resume.tenant_id == user.current_tenant_id
    ).first()
    if res is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Resume not found")
    body = ResumeStatusResponse(
        resume_id=res.id,
        status=ParseStatusEnum(res.status.value),
        parse_confidence=res.parse_confidence,
        parse_source=res.parse_source,
        warnings=list(res.warnings or []),
        error_message=res.error_message,
    )
    return create_success_response(body, "Resume status retrieved successfully")


@router.post(
    "/{resume_id}/match/{job_description_id}",
    response_model=SuccessResponse[dict],
)
def match_resume(
    resume_id: UUID,
    job_description_id: UUID,
    body: MatchRequest | None = None,
    user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    if not user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")
    res = db.query(Resume).filter(
        Resume.id == resume_id, Resume.tenant_id == user.current_tenant_id
    ).first()
    if res is None or res.status != ParseStatus.READY or not res.parsed_json:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Resume must be parsed and READY to match",
        )
    job = db.query(JobDescription).filter(
        JobDescription.id == job_description_id,
        JobDescription.tenant_id == user.current_tenant_id,
    ).first()
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job description not found")

    parsed = ParsedResume.model_validate(res.parsed_json)
    mm = body.match_mode.value if body and body.match_mode else None
    result = score_candidate(res.id, job, parsed, match_mode=mm)

    payload = {
        "match": result.model_dump(mode="json"),
        "billing_note": {"estimated_match_api_charge_usd": 0.0},
    }
    return create_success_response(payload, "Match computed successfully")


def resume_detail_payload(res: Resume) -> dict:
    parsed = res.parsed_json
    if res.parsed_json:
        try:
            parsed = ParsedResume.model_validate(res.parsed_json).model_dump(mode="json")
        except Exception:
            parsed = res.parsed_json
    return {
        "resume_id": str(res.id),
        "status": res.status.value,
        "upload_mode": res.upload_mode.value if res.upload_mode else None,
        "batch_id": str(res.batch_id) if res.batch_id else None,
        "original_filename": res.original_filename,
        "parsed": parsed,
        "confidence_score": res.parse_confidence,
        "warnings": list(res.warnings or []),
        "parser_version": res.parser_version,
        "model_name": res.model_name,
        "provider": res.provider,
        "error_message": res.error_message,
    }


