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
from fastapi.responses import FileResponse
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
    MatchResponse,
    ParseMode,
    ParseStatusEnum,
    ParsedResume,
    ResumeListItem,
    ResumeStatusResponse,
    ShortlistCriteriaResponse,
    ShortlistCriteriaUpdateRequest,
    ShortlistByBatchRequest,
    TopCandidatesRequest,
    TopCandidatesResponse,
)
from app.services.candidate_shortlisting_service import candidate_shortlisting_service
from app.services.resume_matching_service import score_candidate
from app.services.resume_parse_service import run_parse_for_resume
from app.services.resume_rules_parser import extract_location_from_text
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


def _match_result_to_payload(result: MatchResponse, *, verbose: bool) -> dict:
    if verbose:
        return {
            "match": result.model_dump(mode="json"),
            "billing_note": {"estimated_match_api_charge_usd": 0.0},
        }
    strengths = list(result.weighted_skill_hits.keys())[:3]
    gaps = list(result.missing_required_skills)[:3]
    return {
        "match": {
            "resume_id": str(result.resume_id),
            "job_description_id": str(result.job_description_id),
            "match_percent": result.overall_match_percent,
            "fit_label": result.overall_fit_label,
            "fit_summary": result.overall_fit_summary,
            "confidence_percent": result.match_confidence_percent,
            "confidence_label": result.match_confidence_label,
            "top_strengths": strengths,
            "top_gaps": gaps,
        }
    }


def _extract_candidate_summary(resume: Resume) -> tuple[str | None, list[str], float | None]:
    parsed = resume.parsed_json if isinstance(resume.parsed_json, dict) else {}
    profile = parsed.get("profile") if isinstance(parsed.get("profile"), dict) else {}
    location = profile.get("location")
    if not location:
        raw_text = parsed.get("raw_text") or resume.raw_text
        if isinstance(raw_text, str) and raw_text.strip():
            location = extract_location_from_text(raw_text)

    years_exp = parsed.get("years_experience_total")
    try:
        years_exp = float(years_exp) if years_exp is not None else None
    except (TypeError, ValueError):
        years_exp = None

    education_values: list[str] = []
    education = parsed.get("education")
    if isinstance(education, list):
        for item in education:
            if not isinstance(item, dict):
                continue
            degree = str(item.get("degree") or "").strip()
            institution = str(item.get("institution") or "").strip()
            combined = " - ".join(part for part in [degree, institution] if part)
            if combined:
                education_values.append(combined)

    return (str(location).strip() if location else None, education_values, years_exp)


def _extract_candidate_enrichment(
    resume: Resume,
) -> tuple[str | None, str | None, list[str], list[str], list[str], list[str], list[str]]:
    parsed = resume.parsed_json if isinstance(resume.parsed_json, dict) else {}
    profile = parsed.get("profile") if isinstance(parsed.get("profile"), dict) else {}

    title = None
    summary = None

    experience_items = parsed.get("experience") if isinstance(parsed.get("experience"), list) else []
    if experience_items:
        first = experience_items[0] if isinstance(experience_items[0], dict) else {}
        role = str(first.get("role") or "").strip()
        company = str(first.get("company") or "").strip()
        if role and company:
            title = f"{role} at {company}"
        elif role:
            title = role
        elif company:
            title = company

    raw_text = parsed.get("raw_text") if isinstance(parsed.get("raw_text"), str) else ""
    if raw_text:
        summary = " ".join(raw_text.split())[:260]

    skills: list[str] = []
    skills_raw = parsed.get("skills") if isinstance(parsed.get("skills"), list) else []
    for s in skills_raw:
        if isinstance(s, dict):
            name = str(s.get("name") or "").strip()
            if name:
                skills.append(name)

    experience: list[str] = []
    for item in experience_items:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        company = str(item.get("company") or "").strip()
        duration = str(item.get("duration") or "").strip()
        label = " | ".join(part for part in [role, company, duration] if part)
        if label:
            experience.append(label)

    languages: list[str] = []
    langs = parsed.get("languages") if isinstance(parsed.get("languages"), list) else []
    for lang in langs:
        text = str(lang).strip()
        if text:
            languages.append(text)

    projects: list[str] = []
    projects_raw = parsed.get("projects") if isinstance(parsed.get("projects"), list) else []
    for p in projects_raw:
        if not isinstance(p, dict):
            continue
        name = str(p.get("name") or "").strip()
        desc = str(p.get("description") or "").strip()
        project_label = " - ".join(part for part in [name, desc] if part)
        if project_label:
            projects.append(project_label)

    achievements: list[str] = []
    for item in experience_items:
        if not isinstance(item, dict):
            continue
        responsibilities = item.get("responsibilities")
        if not isinstance(responsibilities, list):
            continue
        for ach in responsibilities:
            text = str(ach).strip()
            if text:
                achievements.append(text)

    # Fallback title to profile name if experience role is absent.
    if not title:
        profile_name = str(profile.get("name") or "").strip()
        title = profile_name or None

    return (
        title,
        summary,
        skills[:30],
        experience[:20],
        languages[:15],
        achievements[:20],
        projects[:15],
    )


def _resolved_resume_file_path(storage_path: str) -> Path:
    """Ensure stored path is a real file under the configured upload root."""
    path = Path(storage_path).expanduser().resolve()
    upload_root = Path(
        getattr(settings, "RESUME_UPLOAD_DIR", "./uploads/resumes")
    ).expanduser().resolve()
    try:
        path.relative_to(upload_root)
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Resume file path is not allowed",
        ) from exc
    if not path.is_file():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Resume file not found on server",
        )
    return path


@router.get("/{resume_id}/download")
def download_resume(
    resume_id: UUID,
    user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    """
    Download the original uploaded resume file (tenant-scoped).
    """
    if not user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")

    res = (
        db.query(Resume)
        .filter(
            Resume.id == resume_id,
            Resume.tenant_id == user.current_tenant_id,
        )
        .first()
    )
    if res is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Resume not found")

    file_path = _resolved_resume_file_path(res.storage_path)
    media_type = res.content_type or "application/octet-stream"
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=res.original_filename,
    )


def _match_against_job(
    resume: Resume,
    job: JobDescription,
    *,
    match_mode: MatchMode,
) -> tuple[int | None, float | None, str | None]:
    """Compute match vs JD for list views. Returns (match_percent, overall_score, fit_label)."""
    if resume.status != ParseStatus.READY or not resume.parsed_json:
        return None, None, None
    try:
        parsed = ParsedResume.model_validate(resume.parsed_json)
    except Exception:
        return None, None, None
    try:
        result = score_candidate(
            resume.id,
            job,
            parsed,
            match_mode=match_mode.value,
        )
        return (
            result.overall_match_percent,
            result.overall_score,
            result.overall_fit_label,
        )
    except Exception as exc:
        log.warning("Match scoring failed for resume %s: %s", resume.id, exc)
        return None, None, None


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
    "/upload-multiple-and-match",
    response_model=SuccessResponse[dict],
    status_code=status.HTTP_201_CREATED,
)
async def upload_multiple_resumes_and_match(
    files: list[UploadFile] = File(..., description="Multiple resume files"),
    job_description_id: UUID = Form(
        ...,
        description="Job description UUID to score uploaded resumes against",
    ),
    parse_mode: ParseMode = Form(default=ParseMode.hybrid),
    match_mode: MatchMode | None = Form(default=None),
    verbose: bool = Query(
        default=False,
        description="When true, return full detailed scoring payload. Default returns concise match summary.",
    ),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    """
    Upload multiple resumes, parse each, and score against one job description.
    Returns per-file results without failing the whole batch.
    """
    if not admin_user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "At least one file is required")

    job = db.query(JobDescription).filter(
        JobDescription.id == job_description_id,
        JobDescription.tenant_id == admin_user.current_tenant_id,
    ).first()
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job description not found")

    # Single batch id ties all uploaded files together for traceability.
    batch_id = uuid.uuid4()
    items: list[dict] = []
    success_count = 0
    error_count = 0
    mm = match_mode.value if match_mode else None

    for f in files:
        original_filename = f.filename or "unknown"
        try:
            # Upload phase: persist file + create resume record.
            resume = await _store_single_file_and_create_resume(
                f,
                admin_user.current_tenant_id,
                db,
                upload_mode=UploadMode.BATCH,
                batch_id=batch_id,
            )
            resume.job_description_id = job_description_id
            db.commit()
            db.refresh(resume)
        except HTTPException as exc:
            db.rollback()
            items.append(
                {
                    "original_filename": original_filename,
                    "resume_id": None,
                    "status": "FAILED_UPLOAD",
                    "error": str(exc.detail),
                    "match": None,
                }
            )
            error_count += 1
            continue
        except Exception as exc:
            db.rollback()
            items.append(
                {
                    "original_filename": original_filename,
                    "resume_id": None,
                    "status": "FAILED_UPLOAD",
                    "error": str(exc),
                    "match": None,
                }
            )
            error_count += 1
            continue

        try:
            # Parse phase: convert raw file to structured candidate profile.
            parsed_resume_row = run_parse_for_resume(db, resume.id, parse_mode=parse_mode)
            db.commit()
            db.refresh(parsed_resume_row)
        except Exception as exc:
            db.rollback()
            items.append(
                {
                    "original_filename": original_filename,
                    "resume_id": str(resume.id),
                    "status": "FAILED_PARSE",
                    "error": str(exc),
                    "match": None,
                }
            )
            error_count += 1
            continue

        # Match phase requires parse-ready content.
        if parsed_resume_row.status != ParseStatus.READY or not parsed_resume_row.parsed_json:
            items.append(
                {
                    "original_filename": original_filename,
                    "resume_id": str(parsed_resume_row.id),
                    "status": "FAILED_PARSE",
                    "error": parsed_resume_row.error_message
                    or "Parsing did not complete successfully",
                    "match": None,
                }
            )
            error_count += 1
            continue

        try:
            # Match phase: score one resume against the requested JD.
            parsed = ParsedResume.model_validate(parsed_resume_row.parsed_json)
            result = score_candidate(parsed_resume_row.id, job, parsed, match_mode=mm)
            threshold = float(job.pass_match_threshold if job.pass_match_threshold is not None else 0.5)
            parsed_resume_row.overall_match_score = float(result.overall_score)
            parsed_resume_row.match_percent = int(result.overall_match_percent)
            parsed_resume_row.is_relevant = bool(result.overall_score >= threshold)
            parsed_resume_row.fit_label = "Relevant" if parsed_resume_row.is_relevant else "Irrelevant"
            db.commit()
            db.refresh(parsed_resume_row)
            items.append(
                {
                    "original_filename": original_filename,
                    "resume_id": str(parsed_resume_row.id),
                    "status": "MATCHED",
                    "error": None,
                    "match": _match_result_to_payload(result, verbose=verbose).get("match"),
                }
            )
            success_count += 1
        except Exception as exc:
            items.append(
                {
                    "original_filename": original_filename,
                    "resume_id": str(parsed_resume_row.id),
                    "status": "FAILED_MATCH",
                    "error": str(exc),
                    "match": None,
                }
            )
            error_count += 1

    payload = {
        "batch_id": str(batch_id),
        "job_description_id": str(job_description_id),
        "parse_mode": parse_mode.value,
        "match_mode": mm,
        "items": items,
        "count": len(items),
        "success_count": success_count,
        "error_count": error_count,
    }
    return create_success_response(
        payload,
        "Multiple resumes uploaded and matched",
        status.HTTP_201_CREATED,
    )


@router.get(
    "/by-job/{job_description_id}",
    response_model=SuccessResponse[list[ResumeListItem]],
)
def list_resumes_by_job_description(
    job_description_id: UUID,
    limit: int = Query(50, ge=1, le=200, description="Maximum number of resumes to return"),
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
            Resume.job_description_id == job_description_id,
        )
        .order_by(Resume.created_at.desc())
        .limit(min(limit, 200))
        .all()
    )
    items: list[ResumeListItem] = []
    threshold = float(job.pass_match_threshold if job.pass_match_threshold is not None else 0.5)
    for r in rows:
        location, education, years_exp = _extract_candidate_summary(r)
        title, summary, skills, experience, languages, achievements, projects = _extract_candidate_enrichment(r)
        os_ = r.overall_match_score
        mp = r.match_percent
        if r.is_relevant is not None:
            is_rel: bool | None = bool(r.is_relevant)
        elif os_ is not None:
            is_rel = bool(os_ >= threshold)
        else:
            is_rel = None
        fit_label = r.fit_label or ("Relevant" if is_rel is True else ("Irrelevant" if is_rel is False else None))
        items.append(
            ResumeListItem(
                id=r.id,
                original_filename=r.original_filename,
                status=ParseStatusEnum(r.status.value),
                parse_confidence=r.parse_confidence,
                location=location,
                education=education,
                experience_years=years_exp,
                title=title,
                summary=summary,
                skills=skills,
                experience=experience,
                languages=languages,
                achievements=achievements,
                projects=projects,
                match_percent=mp,
                overall_score=os_,
                overall_match_score=os_,
                fit_label=fit_label,
                is_relevant=is_rel,
                created_at=r.created_at,
                batch_id=r.batch_id,
                job_description_id=r.job_description_id,
            )
        )
    return create_success_response(items, "Resumes by job description retrieved successfully")


@router.get(
    "/resume-after-screening/{job_description_id}",
    response_model=SuccessResponse[list[ResumeListItem]],
)
def list_resumes_after_screening(
    job_description_id: UUID,
    limit: int = Query(50, ge=1, le=200, description="Maximum number of resumes to return"),
    user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    """
    Return only screened-in resumes where overall_score >= job pass threshold.
    """
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
            Resume.job_description_id == job_description_id,
        )
        .order_by(Resume.created_at.desc())
        .limit(min(limit, 200))
        .all()
    )

    threshold = float(job.pass_match_threshold if job.pass_match_threshold is not None else 0.5)
    items: list[ResumeListItem] = []
    for r in rows:
        location, education, years_exp = _extract_candidate_summary(r)
        title, summary, skills, experience, languages, achievements, projects = _extract_candidate_enrichment(r)
        os_ = r.overall_match_score
        mp = r.match_percent
        if os_ is None or os_ < threshold:
            continue

        items.append(
            ResumeListItem(
                id=r.id,
                original_filename=r.original_filename,
                status=ParseStatusEnum(r.status.value),
                parse_confidence=r.parse_confidence,
                location=location,
                education=education,
                experience_years=years_exp,
                title=title,
                summary=summary,
                skills=skills,
                experience=experience,
                languages=languages,
                achievements=achievements,
                projects=projects,
                match_percent=mp,
                overall_score=os_,
                overall_match_score=os_,
                fit_label=r.fit_label or "Relevant",
                is_relevant=True,
                created_at=r.created_at,
                batch_id=r.batch_id,
                job_description_id=r.job_description_id,
            )
        )

    return create_success_response(
        items,
        "Screened resumes retrieved successfully",
    )


# @router.post(
#     "/upload-batch",
#     response_model=SuccessResponse[dict],
#     status_code=status.HTTP_201_CREATED,
#     include_in_schema=False,
# )
# async def upload_resumes_batch(
#     files: list[UploadFile] = File(...),
#     admin_user: User = Depends(require_admin_or_owner),
#     db: Session = Depends(get_db),
# ):
#     return await _upload_resumes_batch_impl(files, admin_user, db)


# @router.post(
#     "/upload-multiple",
#     response_model=SuccessResponse[dict],
#     status_code=status.HTTP_201_CREATED,
# )
# async def upload_resumes_multiple(
#     files: list[UploadFile] = File(...),
#     admin_user: User = Depends(require_admin_or_owner),
#     db: Session = Depends(get_db),
# ):
#     """
#     Public batch endpoint for uploading multiple resumes.
#     """
#     return await _upload_resumes_batch_impl(files, admin_user, db)


# async def _upload_resumes_batch_impl(
#     files: list[UploadFile],
#     admin_user: User,
#     db: Session,
# ) -> SuccessResponse[dict]:
#     """
#     Upload multiple resume files in one request for the current tenant.
#     """
#     if not admin_user.current_tenant_id:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")

#     if not files:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, "At least one file is required")

#     results: list[dict] = []
#     success_count = 0
#     error_count = 0
#     batch_id = uuid.uuid4()
#     for f in files:
#         try:
#             resume = await _store_single_file_and_create_resume(
#                 f,
#                 admin_user.current_tenant_id,
#                 db,
#                 upload_mode=UploadMode.BATCH,
#                 batch_id=batch_id,
#             )
#             db.commit()
#             db.refresh(resume)
#             results.append(
#                 {
#                     "original_filename": resume.original_filename,
#                     "resume_id": str(resume.id),
#                     "status": resume.status.value,
#                     "upload_mode": resume.upload_mode.value,
#                     "batch_id": str(resume.batch_id) if resume.batch_id else None,
#                     "error": None,
#                 }
#             )
#             success_count += 1
#         except HTTPException as exc:
#             db.rollback()
#             results.append(
#                 {
#                     "original_filename": f.filename,
#                     "resume_id": None,
#                     "status": "FAILED",
#                     "upload_mode": UploadMode.BATCH.value,
#                     "batch_id": str(batch_id),
#                     "error": str(exc.detail),
#                 }
#             )
#             error_count += 1
#         except Exception as exc:
#             db.rollback()
#             results.append(
#                 {
#                     "original_filename": f.filename,
#                     "resume_id": None,
#                     "status": "FAILED",
#                     "upload_mode": UploadMode.BATCH.value,
#                     "batch_id": str(batch_id),
#                     "error": str(exc),
#                 }
#             )
#             error_count += 1

#     payload = {
#         "batch_id": str(batch_id),
#         "items": results,
#         "count": len(results),
#         "success_count": success_count,
#         "error_count": error_count,
#     }
#     return create_success_response(payload, "Resumes uploaded successfully", status.HTTP_201_CREATED)


# @router.post(
#     "/{resume_id}/parse",
#     response_model=SuccessResponse[dict],
# )
# def parse(
#     resume_id: UUID,
#     parse_mode: ParseMode = Form(default=ParseMode.hybrid),
#     admin_user: User = Depends(require_admin_or_owner),
#     db: Session = Depends(get_db),
# ):
#     if not admin_user.current_tenant_id:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")
#     res = db.query(Resume).filter(
#         Resume.id == resume_id, Resume.tenant_id == admin_user.current_tenant_id
#     ).first()
#     if res is None:
#         raise HTTPException(status.HTTP_404_NOT_FOUND, "Resume not found")
#     res = run_parse_for_resume(db, resume_id, parse_mode=parse_mode)
#     db.commit()
#     db.refresh(res)
#     payload = resume_detail_payload(res)
#     return create_success_response(payload, "Resume parsed successfully")


# @router.post(
#     "/parse-batch",
#     response_model=SuccessResponse[dict],
# )
# def parse_batch(
#     resume_ids: list[str] = Form(...),
#     parse_mode: ParseMode = Form(default=ParseMode.hybrid),
#     admin_user: User = Depends(require_admin_or_owner),
#     db: Session = Depends(get_db),
# ):
#     """
#     Parse multiple resumes in one call.
#     Returns per-resume status without failing the whole batch.
#     """
#     if not admin_user.current_tenant_id:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")
#     parsed_resume_ids: list[UUID] = []
#     for raw in resume_ids:
#         parts = [p.strip() for p in str(raw).split(",") if p.strip()]
#         for part in parts:
#             try:
#                 parsed_resume_ids.append(UUID(part))
#             except ValueError:
#                 raise HTTPException(
#                     status.HTTP_400_BAD_REQUEST,
#                     f"Invalid resume_id: '{part}'",
#                 ) from None

#     if not parsed_resume_ids:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, "At least one resume_id is required")

#     items: list[dict] = []
#     for rid in parsed_resume_ids:
#         res = db.query(Resume).filter(
#             Resume.id == rid, Resume.tenant_id == admin_user.current_tenant_id
#         ).first()
#         if res is None:
#             items.append(
#                 {
#                     "resume_id": str(rid),
#                     "status": "NOT_FOUND",
#                     "confidence_score": None,
#                     "warnings": [],
#                     "error_message": "Resume not found",
#                 }
#             )
#             continue
#         try:
#             res = run_parse_for_resume(db, rid, parse_mode=parse_mode)
#             db.commit()
#             db.refresh(res)
#             payload = resume_detail_payload(res)
#             items.append(
#                 {
#                     "resume_id": payload["resume_id"],
#                     "status": payload["status"],
#                     "confidence_score": payload["confidence_score"],
#                     "warnings": payload["warnings"],
#                     "error_message": payload["error_message"],
#                 }
#             )
#         except HTTPException as http_exc:
#             db.rollback()
#             items.append(
#                 {
#                     "resume_id": str(rid),
#                     "status": f"ERROR_{http_exc.status_code}",
#                     "confidence_score": None,
#                     "warnings": [],
#                     "error_message": http_exc.detail,
#                 }
#             )
#         except Exception as exc:  # defensive
#             db.rollback()
#             items.append(
#                 {
#                     "resume_id": str(rid),
#                     "status": "ERROR",
#                     "confidence_score": None,
#                     "warnings": [],
#                     "error_message": str(exc),
#                 }
#             )

#     payload = {
#         "items": items,
#         "count": len(items),
#         "parse_mode": parse_mode.value,
#     }
#     return create_success_response(payload, "Batch parse completed")


# @router.get(
#     "",
#     response_model=SuccessResponse[list[ResumeListItem]],
# )
# def list_resumes(
#     limit: int = Query(50, ge=1, le=200, description="Maximum number of resumes to return"),
#     batch_id: UUID | None = Depends(optional_batch_id_query),
#     user: User = Depends(require_member_or_admin),
#     db: Session = Depends(get_db),
# ):
#     if not user.current_tenant_id:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")
#     q = db.query(Resume).filter(Resume.tenant_id == user.current_tenant_id)
#     if batch_id is not None:
#         q = q.filter(Resume.batch_id == batch_id)
#     q = q.order_by(Resume.created_at.desc()).limit(min(limit, 200))
#     rows = q.all()
#     items = [
#         ResumeListItem(
#             id=r.id,
#             original_filename=r.original_filename,
#             status=ParseStatusEnum(r.status.value),
#             parse_confidence=r.parse_confidence,
#             created_at=r.created_at,
#             upload_mode=r.upload_mode.value if r.upload_mode else None,
#             batch_id=r.batch_id,
#         )
#         for r in rows
#     ]
#     return create_success_response(items, "Resumes retrieved successfully")


# @router.post(
#     "/shortlist-by-batch",
#     response_model=SuccessResponse[BatchShortlistPayload],
# )
# def shortlist_batch_against_job(
#     batch_id: UUID = Query(..., description="Upload batch id from multi-upload response"),
#     job_description_id: UUID = Query(..., description="Job description to score candidates against"),
#     top_k: int | None = Query(
#         None,
#         ge=1,
#         le=200,
#         description="Return only the top K candidates after ranking (omit for all scored resumes)",
#     ),
#     min_overall_score: float | None = Query(
#         None,
#         ge=0.0,
#         le=1.0,
#         description="Drop candidates below this overall score before applying top_k",
#     ),
#     max_resumes: int = Query(
#         500,
#         ge=1,
#         le=2000,
#         description="Safety cap: maximum resumes from this batch to evaluate",
#     ),
#     match_mode: MatchMode | None = Query(
#         None,
#         description="rules | ai | hybrid; omit to use server RECRUIT_MATCH_MODE (default hybrid)",
#     ),
#     user: User = Depends(require_member_or_admin),
#     db: Session = Depends(get_db),
# ):
#     if not user.current_tenant_id:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")

#     job = db.query(JobDescription).filter(
#         JobDescription.id == job_description_id,
#         JobDescription.tenant_id == user.current_tenant_id,
#     ).first()
#     if job is None:
#         raise HTTPException(status.HTTP_404_NOT_FOUND, "Job description not found")

#     rows = (
#         db.query(Resume)
#         .filter(
#             Resume.tenant_id == user.current_tenant_id,
#             Resume.batch_id == batch_id,
#         )
#         .order_by(Resume.created_at.asc())
#         .limit(max_resumes)
#         .all()
#     )

#     if not rows:
#         raise HTTPException(
#             status.HTTP_404_NOT_FOUND,
#             "No resumes found for this batch in the current tenant",
#         )

#     items_work: list[BatchShortlistItem] = []
#     not_scored = 0

#     for res in rows:
#         if res.status != ParseStatus.READY or not res.parsed_json:
#             not_scored += 1
#             continue
#         try:
#             parsed = ParsedResume.model_validate(res.parsed_json)
#             mm = match_mode.value if match_mode else None
#             result = score_candidate(res.id, job, parsed, match_mode=mm)
#             pct, fit_label, fit_summary = explain_fit_score(float(result.overall_score))
#             items_work.append(
#                 BatchShortlistItem(
#                     resume_id=res.id,
#                     filename=res.original_filename,
#                     score=float(result.overall_score),
#                     match_percent=pct,
#                     fit_label=fit_label,
#                     fit_summary=fit_summary,
#                 )
#             )
#         except Exception:
#             not_scored += 1

#     items_work.sort(key=lambda x: (-x.score, str(x.resume_id)))

#     if min_overall_score is not None:
#         items_work = [it for it in items_work if it.score >= min_overall_score]

#     if top_k is not None:
#         items_work = items_work[:top_k]

#     payload = BatchShortlistPayload(items=items_work, not_scored_count=not_scored)
#     return create_success_response(payload, "Batch shortlist computed successfully")


# @router.post(
#     "/shortlist",
#     response_model=SuccessResponse[TopCandidatesResponse],
# )
# def shortlist_top_candidates(
#     payload: TopCandidatesRequest,
#     user: User = Depends(require_member_or_admin),
#     db: Session = Depends(get_db),
# ):
#     if not user.current_tenant_id:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")

#     job = db.query(JobDescription).filter(
#         JobDescription.id == payload.job_description_id,
#         JobDescription.tenant_id == user.current_tenant_id,
#     ).first()
#     if job is None:
#         raise HTTPException(status.HTTP_404_NOT_FOUND, "Job description not found")

#     result = candidate_shortlisting_service.shortlist(
#         db,
#         tenant_id=user.current_tenant_id,
#         job=job,
#         batch_id=payload.batch_id,
#         top_k=payload.top_k,
#         min_overall_score=payload.min_overall_score,
#         max_resumes=payload.max_resumes,
#         match_mode=payload.match_mode.value if payload.match_mode else None,
#         include_excluded=payload.include_excluded,
#     )
#     return create_success_response(result, "Top candidates shortlisted successfully")


# @router.post(
#     "/shortlist/by-batch",
#     response_model=SuccessResponse[TopCandidatesResponse],
# )
# def shortlist_top_candidates_by_batch(
#     payload: ShortlistByBatchRequest,
#     user: User = Depends(require_member_or_admin),
#     db: Session = Depends(get_db),
# ):
#     if not user.current_tenant_id:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")

#     job = db.query(JobDescription).filter(
#         JobDescription.id == payload.job_description_id,
#         JobDescription.tenant_id == user.current_tenant_id,
#     ).first()
#     if job is None:
#         raise HTTPException(status.HTTP_404_NOT_FOUND, "Job description not found")

#     result = candidate_shortlisting_service.shortlist(
#         db,
#         tenant_id=user.current_tenant_id,
#         job=job,
#         batch_id=payload.batch_id,
#         top_k=payload.top_k,
#         min_overall_score=payload.min_overall_score,
#         max_resumes=payload.max_resumes,
#         match_mode=payload.match_mode.value if payload.match_mode else None,
#         include_excluded=payload.include_excluded,
#     )
#     return create_success_response(
#         result,
#         "Top candidates shortlisted successfully for the provided batch",
#     )


# @router.put(
#     "/shortlist/criteria/{job_description_id}",
#     response_model=SuccessResponse[ShortlistCriteriaResponse],
# )
# def update_shortlist_criteria(
#     job_description_id: UUID,
#     payload: ShortlistCriteriaUpdateRequest,
#     admin_user: User = Depends(require_admin_or_owner),
#     db: Session = Depends(get_db),
# ):
#     if not admin_user.current_tenant_id:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")

#     job = db.query(JobDescription).filter(
#         JobDescription.id == job_description_id,
#         JobDescription.tenant_id == admin_user.current_tenant_id,
#     ).first()
#     if job is None:
#         raise HTTPException(status.HTTP_404_NOT_FOUND, "Job description not found")

#     criteria_updates: dict = {}
#     if payload.scoring_dimensions is not None:
#         criteria_updates["scoring_dimensions"] = payload.scoring_dimensions
#     if payload.must_have_criteria is not None:
#         criteria_updates["must_have_criteria"] = payload.must_have_criteria
#     if payload.minimum_parse_confidence is not None:
#         criteria_updates["minimum_parse_confidence"] = payload.minimum_parse_confidence
#     if payload.minimum_profile_completeness is not None:
#         criteria_updates["minimum_profile_completeness"] = payload.minimum_profile_completeness

#     job = candidate_shortlisting_service.update_shortlist_criteria(
#         db,
#         job=job,
#         user_id=admin_user.id,
#         criteria_updates=criteria_updates,
#         skill_weight_matrix=payload.skill_weight_matrix,
#     )
#     response = ShortlistCriteriaResponse(
#         job_description_id=job.id,
#         matching_criteria=job.matching_criteria if isinstance(job.matching_criteria, dict) else {},
#         skill_weight_matrix=job.skill_weight_matrix if isinstance(job.skill_weight_matrix, dict) else {},
#         version=job.version or 1,
#     )
#     return create_success_response(response, "Shortlist scoring criteria updated successfully")


# @router.get(
#     "/{resume_id}",
#     response_model=SuccessResponse[dict],
# )
# def get_resume(
#     resume_id: UUID,
#     user: User = Depends(require_member_or_admin),
#     db: Session = Depends(get_db),
# ):
#     if not user.current_tenant_id:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")
#     res = db.query(Resume).filter(
#         Resume.id == resume_id, Resume.tenant_id == user.current_tenant_id
#     ).first()
#     if res is None:
#         raise HTTPException(status.HTTP_404_NOT_FOUND, "Resume not found")
#     payload = resume_detail_payload(res)
#     return create_success_response(payload, "Resume retrieved successfully")


# @router.get(
#     "/{resume_id}/status",
#     response_model=SuccessResponse[ResumeStatusResponse],
# )
# def get_status(
#     resume_id: UUID,
#     user: User = Depends(require_member_or_admin),
#     db: Session = Depends(get_db),
# ):
#     if not user.current_tenant_id:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")
#     res = db.query(Resume).filter(
#         Resume.id == resume_id, Resume.tenant_id == user.current_tenant_id
#     ).first()
#     if res is None:
#         raise HTTPException(status.HTTP_404_NOT_FOUND, "Resume not found")
#     body = ResumeStatusResponse(
#         resume_id=res.id,
#         status=ParseStatusEnum(res.status.value),
#         parse_confidence=res.parse_confidence,
#         parse_source=res.parse_source,
#         warnings=list(res.warnings or []),
#         error_message=res.error_message,
#     )
#     return create_success_response(body, "Resume status retrieved successfully")


# @router.post(
#     "/{resume_id}/match/{job_description_id}",
#     response_model=SuccessResponse[dict],
# )
# def match_resume(
#     resume_id: UUID,
#     job_description_id: UUID,
#     verbose: bool = Query(
#         default=False,
#         description="When true, return full detailed scoring payload. Default returns concise match summary.",
#     ),
#     body: MatchRequest | None = None,
#     user: User = Depends(require_member_or_admin),
#     db: Session = Depends(get_db),
# ):
#     if not user.current_tenant_id:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")
#     res = db.query(Resume).filter(
#         Resume.id == resume_id, Resume.tenant_id == user.current_tenant_id
#     ).first()
#     if res is None or res.status != ParseStatus.READY or not res.parsed_json:
#         raise HTTPException(
#             status.HTTP_400_BAD_REQUEST,
#             "Resume must be parsed and READY to match",
#         )
#     job = db.query(JobDescription).filter(
#         JobDescription.id == job_description_id,
#         JobDescription.tenant_id == user.current_tenant_id,
#     ).first()
#     if job is None:
#         raise HTTPException(status.HTTP_404_NOT_FOUND, "Job description not found")

#     parsed = ParsedResume.model_validate(res.parsed_json)
#     mm = body.match_mode.value if body and body.match_mode else None
#     result = score_candidate(res.id, job, parsed, match_mode=mm)
#     payload = _match_result_to_payload(result, verbose=verbose)
#     return create_success_response(payload, "Match computed successfully")


# def _match_result_to_payload(result: MatchResponse, *, verbose: bool) -> dict:
#     if verbose:
#         return {
#             "match": result.model_dump(mode="json"),
#             "billing_note": {"estimated_match_api_charge_usd": 0.0},
#         }
#     strengths = list(result.weighted_skill_hits.keys())[:3]
#     gaps = list(result.missing_required_skills)[:3]
#     return {
#         "match": {
#             "resume_id": str(result.resume_id),
#             "job_description_id": str(result.job_description_id),
#             "match_percent": result.overall_match_percent,
#             "fit_label": result.overall_fit_label,
#             "fit_summary": result.overall_fit_summary,
#             "confidence_percent": result.match_confidence_percent,
#             "confidence_label": result.match_confidence_label,
#             "top_strengths": strengths,
#             "top_gaps": gaps,
#         }
#     }


# def resume_detail_payload(res: Resume) -> dict:
#     parsed = res.parsed_json
#     if res.parsed_json:
#         try:
#             parsed = ParsedResume.model_validate(res.parsed_json).model_dump(mode="json")
#         except Exception:
#             parsed = res.parsed_json
#     return {
#         "resume_id": str(res.id),
#         "status": res.status.value,
#         "upload_mode": res.upload_mode.value if res.upload_mode else None,
#         "batch_id": str(res.batch_id) if res.batch_id else None,
#         "original_filename": res.original_filename,
#         "parsed": parsed,
#         "confidence_score": res.parse_confidence,
#         "warnings": list(res.warnings or []),
#         "parser_version": res.parser_version,
#         "model_name": res.model_name,
#         "provider": res.provider,
#         "error_message": res.error_message,
#     }




# @router.post(
#     "/upload-and-match",
#     response_model=SuccessResponse[dict],
#     status_code=status.HTTP_201_CREATED,
# )
# async def upload_resume_and_match(
#     file: UploadFile = File(...),
#     job_description_id: UUID = Form(
#         ...,
#         description="Job description UUID to score this resume against",
#     ),
#     parse_mode: ParseMode = Form(default=ParseMode.hybrid),
#     match_mode: MatchMode | None = Form(default=None),
#     verbose: bool = Query(
#         default=False,
#         description="When true, return full detailed scoring payload. Default returns concise match summary.",
#     ),
#     admin_user: User = Depends(require_admin_or_owner),
#     db: Session = Depends(get_db),
# ):
#     """
#     Upload a resume, parse it, and score it against a job description in a single request.
#     """
#     if not admin_user.current_tenant_id:
#         raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")

#     job = db.query(JobDescription).filter(
#         JobDescription.id == job_description_id,
#         JobDescription.tenant_id == admin_user.current_tenant_id,
#     ).first()
#     if job is None:
#         raise HTTPException(status.HTTP_404_NOT_FOUND, "Job description not found")

#     # 1) Persist uploaded file and create a pending Resume row.
#     resume = await _store_single_file_and_create_resume(
#         file,
#         admin_user.current_tenant_id,
#         db,
#         upload_mode=UploadMode.SINGLE,
#     )
#     db.commit()
#     db.refresh(resume)

#     # 2) Parse the stored resume into structured JSON used by the matcher.
#     try:
#         res = run_parse_for_resume(db, resume.id, parse_mode=parse_mode)
#     except ValueError:
#         raise HTTPException(status.HTTP_404_NOT_FOUND, "Resume not found")
#     db.commit()
#     db.refresh(res)

#     # Parsing is a hard prerequisite for matching.
#     if res.status != ParseStatus.READY or not res.parsed_json:
#         raise HTTPException(
#             status.HTTP_422_UNPROCESSABLE_ENTITY,
#             detail={
#                 "message": "Resume uploaded but parsing did not complete successfully",
#                 "resume": resume_detail_payload(res),
#             },
#         )

#     # 3) Score parsed resume against JD and shape response.
#     parsed = ParsedResume.model_validate(res.parsed_json)
#     mm = match_mode.value if match_mode else None
#     result = score_candidate(res.id, job, parsed, match_mode=mm)
#     match_payload = _match_result_to_payload(result, verbose=verbose)
#     payload = {"resume": resume_detail_payload(res), **match_payload}
#     return create_success_response(
#         payload,
#         "Resume uploaded, parsed, and matched successfully",
#         status.HTTP_201_CREATED,
#     )
