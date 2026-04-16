from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin_or_owner, require_member_or_admin
from app.models.job_description import JobDescription
from app.models.call_session import CallSession
from app.models.resume import Resume
from app.models.resume_interview import ResumeInterview
from app.models.scheduled_call import ScheduledCall
from app.models.tenant_crm_config import CRMConfig
from app.models.user import User
from app.schemas.base import SuccessResponse
from app.schemas.resume_interview import (
    ResumeInterviewBulkScheduleRequest,
    ResumeInterviewBulkScheduleResponse,
    ResumeInterviewBulkScheduleResultItem,
    ResumeInterviewItem,
    ResumeInterviewSessionLinkItem,
    ResumeInterviewScheduleRequest,
    ResumeInterviewStatusUpdateRequest,
)
from app.services.scheduled_call_service import ScheduledCallService
from app.utils.response import create_success_response

router = APIRouter()
scheduled_call_service = ScheduledCallService()

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "NO_ANSWER", "CANCELLED", "REJECTED"}
ALLOWED_STATUS_TRANSITIONS = {
    "SCHEDULE_REQUESTED": {"SCHEDULED", "IN_PROGRESS", "SCHEDULE_FAILED", "CANCELLED"},
    "SCHEDULE_FAILED": {"SCHEDULE_REQUESTED", "CANCELLED"},
    "SCHEDULED": {"DIALING", "IN_PROGRESS", "CANCELLED"},
    "DIALING": {"IN_PROGRESS", "NO_ANSWER", "FAILED", "CANCELLED"},
    "IN_PROGRESS": {"COMPLETED", "FAILED", "NO_ANSWER"},
    "NO_ANSWER": {"SCHEDULED", "CANCELLED"},
    "FAILED": {"SCHEDULED", "CANCELLED"},
}


def _serialize_interview(row: ResumeInterview) -> ResumeInterviewItem:
    return ResumeInterviewItem(
        id=row.id,
        tenant_id=row.tenant_id,
        resume_id=row.resume_id,
        job_description_id=row.job_description_id,
        agent_id=row.agent_id,
        call_session_id=row.call_session_id,
        candidate_phone=row.candidate_phone,
        scheduled_at=row.scheduled_at,
        status=row.status,
        crm_type=row.crm_type,
        crm_item_id=row.crm_item_id,
        crm_batch_id=row.crm_batch_id,
        phone_number_id=row.phone_number_id,
        twilio_call_sid=row.twilio_call_sid,
        attempt_count=row.attempt_count,
        last_error=row.last_error,
        metadata_json=row.metadata_json,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_session_link_item(
    *,
    resume: Resume,
    interview: ResumeInterview | None,
    call_session: CallSession | None,
) -> ResumeInterviewSessionLinkItem:
    return ResumeInterviewSessionLinkItem(
        resume_id=resume.id,
        resume_filename=resume.original_filename,
        interview_id=interview.id if interview else None,
        interview_status=interview.status if interview else None,
        scheduled_at=interview.scheduled_at if interview else None,
        call_session_id=interview.call_session_id if interview else None,
        call_session_status=call_session.status if call_session else None,
        twilio_call_sid=interview.twilio_call_sid if interview else None,
        crm_item_id=interview.crm_item_id if interview else None,
        crm_batch_id=interview.crm_batch_id if interview else None,
    )


def _parse_utc_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Invalid call_time_utc format. Use ISO datetime (e.g. 2026-04-16T12:30:00Z).",
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_crm_config_id(db: Session, user: User, explicit: UUID | None) -> UUID:
    if explicit:
        return explicit

    # Prefer Trello as requested by product flow.
    trello_link = (
        db.query(ScheduledCall)
        .filter(
            ScheduledCall.user_id == user.id,
            ScheduledCall.tenant_crm_config_id.isnot(None),
            ScheduledCall.crm_type == "trello",
        )
        .order_by(ScheduledCall.created_at.desc())
        .first()
    )
    if trello_link and trello_link.tenant_crm_config_id:
        return trello_link.tenant_crm_config_id

    # Fallback: any linked CRM config for this user.
    any_link = (
        db.query(ScheduledCall)
        .filter(
            ScheduledCall.user_id == user.id,
            ScheduledCall.tenant_crm_config_id.isnot(None),
        )
        .order_by(ScheduledCall.created_at.desc())
        .first()
    )
    if any_link and any_link.tenant_crm_config_id:
        return any_link.tenant_crm_config_id

    # Final fallback: global Trello CRM config from DB.
    global_trello = (
        db.query(CRMConfig)
        .filter(CRMConfig.crm_type == "trello")
        .first()
    )
    if global_trello:
        return global_trello.id

    raise HTTPException(
        status.HTTP_400_BAD_REQUEST,
        "crm_config_id is missing and no linked/global CRM configuration was found",
    )


async def _create_scheduled_interview(
    *,
    db: Session,
    user: User,
    body: ResumeInterviewScheduleRequest,
) -> ResumeInterview:
    if not user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")

    resume = (
        db.query(Resume)
        .filter(Resume.id == body.resume_id, Resume.tenant_id == user.current_tenant_id)
        .first()
    )
    if resume is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Resume not found")

    if body.job_description_id:
        job = (
            db.query(JobDescription)
            .filter(
                JobDescription.id == body.job_description_id,
                JobDescription.tenant_id == user.current_tenant_id,
            )
            .first()
        )
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Job description not found")

    scheduled_at_utc = _parse_utc_datetime(body.call_time_utc)

    existing = (
        db.query(ResumeInterview)
        .filter(
            ResumeInterview.tenant_id == user.current_tenant_id,
            ResumeInterview.resume_id == body.resume_id,
            ResumeInterview.scheduled_at == scheduled_at_utc,
            ResumeInterview.status.notin_(TERMINAL_STATUSES),
        )
        .first()
    )
    if existing is not None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "An active interview already exists for this resume at the same time",
        )

    interview = ResumeInterview(
        tenant_id=user.current_tenant_id,
        resume_id=body.resume_id,
        job_description_id=body.job_description_id,
        agent_id=body.agent_id,
        candidate_phone=body.phone_number,
        scheduled_at=scheduled_at_utc,
        status="SCHEDULE_REQUESTED",
        phone_number_id=body.phone_number_id,
        metadata_json=body.metadata or {},
        created_by=user.id,
        updated_by=user.id,
    )
    db.add(interview)
    db.flush()

    try:
        resolved_crm_config_id = _resolve_crm_config_id(db, user, body.crm_config_id)
        schedule_result = await scheduled_call_service.create_single_scheduled_call(
            db=db,
            tenant_id=user.current_tenant_id,
            user_id=user.id,
            phone_number=body.phone_number,
            agent_id=body.agent_id,
            call_time_utc=body.call_time_utc,
            crm_config_id=resolved_crm_config_id,
            phone_number_id=str(body.phone_number_id) if body.phone_number_id else None,
        )
        interview.status = "IN_PROGRESS"
        interview.crm_item_id = schedule_result.get("item_id")
        interview.crm_batch_id = schedule_result.get("batch_id")
        interview.crm_type = schedule_result.get("crm_type")
        interview.last_error = None
        interview.updated_by = user.id
        db.commit()
    except HTTPException as exc:
        interview.status = "SCHEDULE_FAILED"
        interview.last_error = str(exc.detail)
        interview.updated_by = user.id
        db.commit()
        raise
    except Exception as exc:
        interview.status = "SCHEDULE_FAILED"
        interview.last_error = str(exc)
        interview.updated_by = user.id
        db.commit()
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to schedule interview")

    db.refresh(interview)
    return interview


@router.post(
    "/schedule",
    response_model=SuccessResponse[ResumeInterviewItem],
    status_code=status.HTTP_201_CREATED,
)
async def schedule_resume_interview(
    body: ResumeInterviewScheduleRequest,
    user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    interview = await _create_scheduled_interview(db=db, user=user, body=body)
    return create_success_response(
        _serialize_interview(interview),
        "Resume interview scheduled successfully",
        status.HTTP_201_CREATED,
    )


@router.post(
    "/schedule-bulk",
    response_model=SuccessResponse[ResumeInterviewBulkScheduleResponse],
    status_code=status.HTTP_201_CREATED,
)
async def schedule_resume_interviews_bulk(
    body: ResumeInterviewBulkScheduleRequest,
    user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    results: list[ResumeInterviewBulkScheduleResultItem] = []
    success_count = 0
    error_count = 0

    for item in body.items:
        try:
            interview = await _create_scheduled_interview(db=db, user=user, body=item)
            success_count += 1
            results.append(
                ResumeInterviewBulkScheduleResultItem(
                    resume_id=item.resume_id,
                    success=True,
                    interview=_serialize_interview(interview),
                    error=None,
                )
            )
        except HTTPException as exc:
            error_count += 1
            results.append(
                ResumeInterviewBulkScheduleResultItem(
                    resume_id=item.resume_id,
                    success=False,
                    interview=None,
                    error=str(exc.detail),
                )
            )
        except Exception as exc:
            error_count += 1
            results.append(
                ResumeInterviewBulkScheduleResultItem(
                    resume_id=item.resume_id,
                    success=False,
                    interview=None,
                    error=str(exc),
                )
            )

    payload = ResumeInterviewBulkScheduleResponse(
        total=len(body.items),
        success_count=success_count,
        error_count=error_count,
        items=results,
    )
    return create_success_response(
        payload,
        "Bulk resume interview scheduling completed",
        status.HTTP_201_CREATED,
    )


@router.get(
    "/by-resume/{resume_id}",
    response_model=SuccessResponse[list[ResumeInterviewItem]],
)
def list_resume_interviews(
    resume_id: UUID,
    limit: int = Query(50, ge=1, le=200),
    user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    if not user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")

    rows = (
        db.query(ResumeInterview)
        .filter(
            ResumeInterview.tenant_id == user.current_tenant_id,
            ResumeInterview.resume_id == resume_id,
        )
        .order_by(ResumeInterview.created_at.desc())
        .limit(limit)
        .all()
    )
    return create_success_response(
        [_serialize_interview(r) for r in rows],
        "Resume interview history retrieved successfully",
    )


@router.patch(
    "/{interview_id}/status",
    response_model=SuccessResponse[ResumeInterviewItem],
)
def update_resume_interview_status(
    interview_id: UUID,
    body: ResumeInterviewStatusUpdateRequest,
    user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    if not user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")

    interview = (
        db.query(ResumeInterview)
        .filter(
            ResumeInterview.id == interview_id,
            ResumeInterview.tenant_id == user.current_tenant_id,
        )
        .first()
    )
    if interview is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Resume interview not found")

    requested = body.status.upper().strip()
    current = (interview.status or "").upper().strip()
    if requested != current:
        allowed = ALLOWED_STATUS_TRANSITIONS.get(current, set())
        if requested not in allowed:
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"Invalid status transition from {current} to {requested}",
            )

    interview.status = requested
    if body.call_session_id:
        interview.call_session_id = body.call_session_id
    if body.twilio_call_sid:
        interview.twilio_call_sid = body.twilio_call_sid
    if body.last_error is not None:
        interview.last_error = body.last_error
    if body.increment_attempt:
        interview.attempt_count = int(interview.attempt_count or 0) + 1
    if body.metadata_patch:
        existing = interview.metadata_json or {}
        interview.metadata_json = {**existing, **body.metadata_patch}
    interview.updated_by = user.id

    db.commit()
    db.refresh(interview)
    return create_success_response(
        _serialize_interview(interview),
        "Resume interview status updated successfully",
    )


@router.get(
    "/session-link/by-resume/{resume_id}",
    response_model=SuccessResponse[ResumeInterviewSessionLinkItem],
)
def get_resume_session_link(
    resume_id: UUID,
    user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    if not user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")

    resume = (
        db.query(Resume)
        .filter(
            Resume.id == resume_id,
            Resume.tenant_id == user.current_tenant_id,
        )
        .first()
    )
    if resume is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Resume not found")

    interview = (
        db.query(ResumeInterview)
        .filter(
            ResumeInterview.tenant_id == user.current_tenant_id,
            ResumeInterview.resume_id == resume_id,
        )
        .order_by(ResumeInterview.created_at.desc())
        .first()
    )
    call_session = None
    if interview and interview.call_session_id:
        call_session = (
            db.query(CallSession)
            .filter(
                CallSession.id == interview.call_session_id,
                CallSession.tenant_id == user.current_tenant_id,
            )
            .first()
        )

    return create_success_response(
        _to_session_link_item(resume=resume, interview=interview, call_session=call_session),
        "Resume session link retrieved successfully",
    )


@router.get(
    "/session-link/by-job/{job_description_id}",
    response_model=SuccessResponse[list[ResumeInterviewSessionLinkItem]],
)
def list_resume_session_links_by_job(
    job_description_id: UUID,
    limit: int = Query(200, ge=1, le=500),
    user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    if not user.current_tenant_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Current tenant is required")

    job = (
        db.query(JobDescription)
        .filter(
            JobDescription.id == job_description_id,
            JobDescription.tenant_id == user.current_tenant_id,
        )
        .first()
    )
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job description not found")

    resumes = (
        db.query(Resume)
        .filter(
            Resume.tenant_id == user.current_tenant_id,
            Resume.job_description_id == job_description_id,
        )
        .order_by(Resume.created_at.desc())
        .limit(limit)
        .all()
    )

    items: list[ResumeInterviewSessionLinkItem] = []
    for resume in resumes:
        interview = (
            db.query(ResumeInterview)
            .filter(
                ResumeInterview.tenant_id == user.current_tenant_id,
                ResumeInterview.resume_id == resume.id,
            )
            .order_by(ResumeInterview.created_at.desc())
            .first()
        )
        call_session = None
        if interview and interview.call_session_id:
            call_session = (
                db.query(CallSession)
                .filter(
                    CallSession.id == interview.call_session_id,
                    CallSession.tenant_id == user.current_tenant_id,
                )
                .first()
            )
        items.append(
            _to_session_link_item(
                resume=resume,
                interview=interview,
                call_session=call_session,
            )
        )

    return create_success_response(items, "Resume session links by job retrieved successfully")
