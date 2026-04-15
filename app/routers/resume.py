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
    MatchMode,
    MatchResponse,
    ParseMode,
    ParseStatusEnum,
    ParsedResume,
    ResumeListItem,
)
from app.services.resume_matching_service import score_candidate
from app.services.resume_parse_service import run_parse_for_resume
from app.utils.response import create_success_response

router = APIRouter()
log = logging.getLogger(__name__)


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


def _extract_candidate_phone(resume: Resume) -> str | None:
    parsed = resume.parsed_json if isinstance(resume.parsed_json, dict) else {}
    profile = parsed.get("profile") if isinstance(parsed.get("profile"), dict) else {}
    phone = str(profile.get("phone") or "").strip()
    return phone or None


def _extract_candidate_email(resume: Resume) -> str | None:
    parsed = resume.parsed_json if isinstance(resume.parsed_json, dict) else {}
    profile = parsed.get("profile") if isinstance(parsed.get("profile"), dict) else {}
    email = str(profile.get("email") or "").strip()
    return email or None


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
        phone = _extract_candidate_phone(r)
        email = _extract_candidate_email(r)
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
                phone=phone,
                email=email,
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
        phone = _extract_candidate_phone(r)
        email = _extract_candidate_email(r)
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
                phone=phone,
                email=email,
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
