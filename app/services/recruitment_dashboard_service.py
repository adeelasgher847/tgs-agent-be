from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.models.job_description import JobDescription
from app.models.resume import ParseStatus, Resume
from app.models.resume_interview import ResumeInterview
from app.models.tenant import Tenant
from app.models.user import User
from app.schemas.recruitment_dashboard import (
    AccountSnapshot,
    ActiveJobRow,
    RecruitmentDashboardData,
    RecruitmentFunnelRow,
    RecruitmentKpiBlock,
    UpcomingInterviewItem,
)

_CANCELLED = ("CANCELLED", "REJECTED")


def _initials(name: str) -> str:
    parts = re.split(r"\s+", (name or "").strip())
    if not parts or not parts[0]:
        return "?"
    if len(parts) == 1:
        s = parts[0]
        return (s[:2] if len(s) >= 2 else s[0]).upper()
    a, b = parts[0][0], parts[-1][0]
    return f"{a}{b}".upper()


def _candidate_name_from_resume(r: Resume) -> str:
    parsed = r.parsed_json if isinstance(r.parsed_json, dict) else {}
    prof = parsed.get("profile") if isinstance(parsed.get("profile"), dict) else {}
    n = (prof.get("name") or "").strip()
    if n:
        return n
    base = (r.original_filename or "Candidate").rsplit(".", 1)[0]
    return base or "Candidate"


def _open_roles_from_jd(matching_criteria: Any) -> int:
    if not isinstance(matching_criteria, dict):
        return 1
    for key in ("open_roles", "headcount", "openings", "slots"):
        v = matching_criteria.get(key)
        if v is not None:
            try:
                n = int(v)
                return max(1, n)
            except (TypeError, ValueError):
                continue
    return 1


def _human_posted_ago(created: datetime, now: datetime) -> str:
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    delta = now - created
    days = max(0, delta.days)
    if days == 0:
        return "Posted today"
    if days == 1:
        return "Posted 1 day ago"
    if days < 7:
        return f"Posted {days} days ago"
    weeks = days // 7
    if weeks == 1:
        return "Posted 1 week ago"
    if weeks < 5:
        return f"Posted {weeks} weeks ago"
    return created.strftime("Posted %b %d, %Y")


def _time_label_utc(d: datetime) -> str:
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    d_utc = d.astimezone(timezone.utc)
    today = datetime.now(timezone.utc).date()
    d_date = d_utc.date()
    t_str = d_utc.strftime("%I:%M %p")
    if t_str.startswith("0"):
        t_str = t_str[1:]
    if d_date == today:
        return f"Today {t_str} UTC"
    if d_date == today + timedelta(days=1):
        return f"Tomorrow {t_str} UTC"
    return f"{d_utc:%b %d, %I:%M %p} UTC"


def _month_bounds(anchor: datetime) -> tuple[datetime, datetime, datetime, datetime]:
    """Current month [cur_start, cur_end), previous month [prev_start, prev_end)."""
    y, m = anchor.year, anchor.month
    cur_start = datetime(y, m, 1, tzinfo=timezone.utc)
    if m == 12:
        next_m_start = datetime(y + 1, 1, 1, tzinfo=timezone.utc)
    else:
        next_m_start = datetime(y, m + 1, 1, tzinfo=timezone.utc)
    cur_end = next_m_start
    if m == 1:
        prev_start = datetime(y - 1, 12, 1, tzinfo=timezone.utc)
    else:
        prev_start = datetime(y, m - 1, 1, tzinfo=timezone.utc)
    prev_end = cur_start
    return cur_start, cur_end, prev_start, prev_end


class RecruitmentDashboardService:
    def build(self, db: Session, user: User) -> RecruitmentDashboardData:
        tenant_id = user.current_tenant_id
        if not tenant_id:
            raise ValueError("current_tenant_id is required")

        now = datetime.now(timezone.utc)
        cur_start, cur_end, prev_start, prev_end = _month_bounds(now)
        week_ago = now - timedelta(days=7)
        today_utc = now.date()

        tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
        if not tenant:
            raise ValueError("Tenant not found")

        # --- Resumes ---
        total_candidates = db.query(Resume).filter(Resume.tenant_id == tenant_id).count()
        screened_candidates = (
            db.query(Resume)
            .filter(Resume.tenant_id == tenant_id, Resume.status == ParseStatus.READY)
            .count()
        )
        new_this_week = (
            db.query(Resume)
            .filter(Resume.tenant_id == tenant_id, Resume.created_at >= week_ago)
            .count()
        )

        # --- Job descriptions (open positions / requisitions) ---
        open_positions = db.query(JobDescription).filter(JobDescription.tenant_id == tenant_id).count()
        jd_this_month = (
            db.query(JobDescription)
            .filter(
                JobDescription.tenant_id == tenant_id,
                JobDescription.created_at >= cur_start,
                JobDescription.created_at < cur_end,
            )
            .count()
        )
        jd_prev_month = (
            db.query(JobDescription)
            .filter(
                JobDescription.tenant_id == tenant_id,
                JobDescription.created_at >= prev_start,
                JobDescription.created_at < prev_end,
            )
            .count()
        )
        delta_jd = jd_this_month - jd_prev_month
        open_sub: str | None
        if delta_jd > 0:
            open_sub = f"+{delta_jd} since last month"
        elif delta_jd < 0:
            open_sub = f"{delta_jd} since last month"
        elif jd_this_month > 0:
            open_sub = f"{jd_this_month} new this month"
        else:
            open_sub = None

        total_sub = f"+{new_this_week} this week" if new_this_week else None

        # --- Interviews: scope to tenant ---
        all_iv = (
            db.query(ResumeInterview)
            .filter(ResumeInterview.tenant_id == tenant_id)
            .all()
        )
        not_cancel = [i for i in all_iv if i.status not in _CANCELLED]
        completed = [i for i in not_cancel if i.status == "COMPLETED"]

        def _is_offer(m: Any) -> bool:
            if not isinstance(m, dict):
                return False
            out = str(m.get("outcome", "")).lower()
            st = str(m.get("stage", "")).lower()
            if out == "offer" or st == "offer":
                return True
            return bool(m.get("offer") is True or m.get("offer_sent") is True)

        offer_rows = [i for i in not_cancel if _is_offer(i.metadata_json)]
        offer_resume_ids = {i.resume_id for i in offer_rows}
        offer_count = len(offer_resume_ids)

        with_interview = {i.resume_id for i in not_cancel}
        with_completed = {i.resume_id for i in completed}

        ready_ids = {
            r[0]
            for r in (
                db.query(Resume.id)
                .filter(Resume.tenant_id == tenant_id, Resume.status == ParseStatus.READY)
                .all()
            )
        }

        # Funnel: monotonic — sourcing ≥ screened ≥ technical ≥ manager ≥ offer
        n_sourcing = total_candidates
        n_screened = min(screened_candidates, n_sourcing) if n_sourcing else 0
        n_technical = len(with_interview & ready_ids)
        n_technical = min(n_technical, n_screened) if n_screened else n_technical
        n_manager = len(with_completed & ready_ids)
        n_manager = min(n_manager, n_technical) if n_technical else n_manager
        # Offers stay ≤ manager stage; if no one has completed a call yet, still cap by technical
        n_offer = min(offer_count, n_manager) if n_manager else min(offer_count, n_technical)

        interviews_today = sum(
            1
            for i in not_cancel
            if i.scheduled_at
            and i.scheduled_at.astimezone(timezone.utc).date() == today_utc
        )
        # KPI "Interviews scheduled" == candidates in the technical (screening) stage, same as funnel
        interviews_scheduled = n_technical
        int_sub = f"{interviews_today} today" if interviews_today else None

        awaiting = sum(
            1
            for i in offer_rows
            if (i.metadata_json or {}).get("awaiting_feedback") is True
            or (i.metadata_json or {}).get("feedback_pending") is True
        )
        off_sub = f"{awaiting} awaiting feedback" if awaiting else None

        pipeline = [
            RecruitmentFunnelRow(key="sourcing", label="Sourcing", count=n_sourcing),
            RecruitmentFunnelRow(key="screened", label="Screened", count=n_screened),
            RecruitmentFunnelRow(key="technical", label="Technical", count=n_technical),
            RecruitmentFunnelRow(key="manager", label="Manager", count=n_manager),
            RecruitmentFunnelRow(key="offer", label="Offer", count=n_offer),
        ]

        summary = RecruitmentKpiBlock(
            open_positions=open_positions,
            open_positions_subtitle=open_sub,
            total_candidates=total_candidates,
            total_candidates_subtitle=total_sub,
            interviews_scheduled=interviews_scheduled,
            interviews_today=interviews_today,
            interviews_scheduled_subtitle=int_sub,
            offers_sent=n_offer,
            offers_awaiting_feedback=awaiting,
            offers_subtitle=off_sub,
        )

        # Upcoming: future interviews (next 14 days), not cancelled
        future_cutoff = now + timedelta(days=14)
        upcoming_q = (
            db.query(ResumeInterview, Resume, JobDescription)
            .join(Resume, Resume.id == ResumeInterview.resume_id)
            .outerjoin(
                JobDescription,
                JobDescription.id == ResumeInterview.job_description_id,
            )
            .filter(
                ResumeInterview.tenant_id == tenant_id,
                Resume.tenant_id == tenant_id,
                ResumeInterview.status.notin_(_CANCELLED),
                ResumeInterview.scheduled_at >= now,
                ResumeInterview.scheduled_at <= future_cutoff,
            )
            .order_by(ResumeInterview.scheduled_at.asc())
            .limit(20)
        )
        upcoming: list[UpcomingInterviewItem] = []
        for interview, res, jd in upcoming_q:
            st = interview.scheduled_at
            if st is None or st.tzinfo is None:
                st_aware = (st or now).replace(tzinfo=timezone.utc) if st else now
            else:
                st_aware = st
            d_utc = st_aware.astimezone(timezone.utc).date()
            today_d = datetime.now(timezone.utc).date()
            name = _candidate_name_from_resume(res)
            upcoming.append(
                UpcomingInterviewItem(
                    resume_id=res.id,
                    interview_id=interview.id,
                    candidate_name=name,
                    candidate_initials=_initials(name),
                    job_title=jd.job_title if jd else None,
                    scheduled_at=st_aware,
                    time_label=_time_label_utc(st_aware),
                    is_today=d_utc == today_d,
                    is_tomorrow=d_utc == today_d + timedelta(days=1),
                )
            )

        # Active openings table
        jds = (
            db.query(JobDescription)
            .filter(JobDescription.tenant_id == tenant_id)
            .order_by(JobDescription.created_at.desc())
            .limit(50)
            .all()
        )
        active_openings: list[ActiveJobRow] = []
        for jd in jds:
            ap_count = (
                db.query(Resume)
                .filter(Resume.tenant_id == tenant_id, Resume.job_description_id == jd.id)
                .count()
            )
            active_openings.append(
                ActiveJobRow(
                    job_title=jd.job_title,
                    open_roles=_open_roles_from_jd(jd.matching_criteria),
                    applicant_count=ap_count,
                    posted_at=jd.created_at,
                    posted_ago=_human_posted_ago(jd.created_at, now),
                    job_description_id=jd.id,
                )
            )

        account = AccountSnapshot(
            user_id=user.id,
            email=user.email,
            tenant_id=tenant_id,
            tenant_name=tenant.name,
            credits=float(tenant.credits or 0),
        )

        return RecruitmentDashboardData(
            summary=summary,
            pipeline=pipeline,
            upcoming_interviews=upcoming,
            active_openings=active_openings,
            account=account,
        )


recruitment_dashboard_service = RecruitmentDashboardService()
