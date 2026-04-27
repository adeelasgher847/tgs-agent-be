"""
Builds voice-call prompt addenda from optional resume + job description IDs
for /voice/call/initiate. Safe to run on every call: never raises, logs-only on miss.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.job_description import JobDescription
from app.models.resume import Resume

# Keep voice prompts bounded for latency
_MAX_JD_CHARS = 2800
_MAX_RESUME_CHARS = 1200


def parse_optional_uuid(value: str | None) -> uuid.UUID | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return uuid.UUID(s)
    except (ValueError, TypeError, AttributeError):
        return None


def _name_from_raw_text_line(resume: Resume) -> str | None:
    """Best-effort when profile.name is missing: first line of raw_text often is the name."""
    t = (resume.raw_text or "").strip()
    if not t:
        return None
    first = t.split("\n", 1)[0].strip()
    if not first or len(first) > 64:
        return None
    fl = first.lower()
    if "@" in first or "http" in fl or "phone" in fl or "e-mail" in fl or "email" in fl:
        return None
    if re.match(r"^\d", first):  # starts with number (address etc.)
        return None
    # e.g. "PROFESSIONAL SUMMARY" — reject long ALL-CAPS headers
    if first.isupper() and len(first) > 20:
        return None
    if not re.match(r"^[A-Za-z][A-Za-z\s.'-]*$", first):
        return None
    words = first.split()
    if not words or len(words) > 5:
        return None
    return first.strip()


def _candidate_name_from_resume(resume: Resume) -> str | None:
    pj = resume.parsed_json
    if isinstance(pj, dict):
        prof = pj.get("profile")
        if isinstance(prof, dict) and prof.get("name"):
            n = str(prof.get("name", "")).strip()
            if n:
                return n
        n2 = pj.get("name")
        if n2 and str(n2).strip():
            return str(n2).strip()
    return _name_from_raw_text_line(resume)


def _job_text_for_prompt(job: JobDescription) -> str:
    title = (job.job_title or "").strip() or "Role"
    parts: list[str] = [f"Title: {title}"]
    if job.location:
        parts.append(f"Location: {str(job.location).strip()}")
    if job.employment_type:
        parts.append(f"Employment type: {str(job.employment_type).strip()}")
    if job.years_experience_min is not None or job.years_experience_max is not None:
        y0, y1 = job.years_experience_min, job.years_experience_max
        if y0 is not None and y1 is not None:
            parts.append(f"Experience range: {y0}–{y1} years")
        elif y0 is not None:
            parts.append(f"Minimum experience: {y0} years")
    if job.raw_text and str(job.raw_text).strip():
        raw = str(job.raw_text).strip()
        if len(raw) > _MAX_JD_CHARS:
            raw = f"{raw[:_MAX_JD_CHARS].rstrip()}..."
        parts.append("Full job description:\n" + raw)
    elif job.key_responsibilities:
        try:
            blob = json.dumps(job.key_responsibilities, ensure_ascii=False)
            if len(blob) > 2000:
                blob = f"{blob[:2000]}..."
            parts.append("Responsibilities (structured):\n" + blob)
        except Exception:
            pass
    return "\n\n".join(parts)


def _resume_excerpt(resume: Resume) -> str:
    if not resume.raw_text or not str(resume.raw_text).strip():
        return ""
    t = str(resume.raw_text).strip()
    if len(t) > _MAX_RESUME_CHARS:
        return f"{t[:_MAX_RESUME_CHARS].rstrip()}..."
    return t


def build_voice_interview_enrichment(
    db: Session,
    *,
    tenant_id: uuid.UUID,
    jd_id: uuid.UUID | None = None,
    resume_id: uuid.UUID | None = None,
    existing_jd_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Returns:
      - merged_jd_context: superset of client jd_context for persistence
      - voice_dynamic_context: { system_prompt_addendum, candidate_name?, job_title? } or None
    """
    out_merged: dict[str, Any] = {**(existing_jd_context or {})}
    if not jd_id and not resume_id:
        return {"merged_jd_context": out_merged, "voice_dynamic_context": None}

    resume: Resume | None = None
    job: JobDescription | None = None
    job_title_hint = str(out_merged.get("jd_title") or "").strip() or None

    if resume_id:
        resume = (
            db.query(Resume)
            .filter(Resume.id == resume_id, Resume.tenant_id == tenant_id)
            .first()
        )
        if not resume:
            logger.warning(
                "Voice interview: resume not found: %s (tenant %s)", resume_id, tenant_id
            )
        else:
            out_merged["resume_id"] = str(resume_id)
            cname = _candidate_name_from_resume(resume)
            if cname:
                out_merged["candidate_name"] = cname
            if not jd_id and resume.job_description_id:
                jd_id = resume.job_description_id

    if jd_id:
        job = (
            db.query(JobDescription)
            .filter(JobDescription.id == jd_id, JobDescription.tenant_id == tenant_id)
            .first()
        )
        if not job:
            logger.warning(
                "Voice interview: job description not found: %s (tenant %s)", jd_id, tenant_id
            )
        else:
            out_merged["jd_id"] = str(jd_id)
            jt = (job.job_title or "").strip()
            if jt:
                out_merged["jd_title"] = jt
                job_title_hint = jt
            if job.raw_text and str(job.raw_text).strip():
                summ = str(job.raw_text).strip()
                if len(summ) > 500:
                    summ = f"{summ[:500].rstrip()}..."
                out_merged["jd_summary"] = summ

    if resume and job and resume.job_description_id and resume.job_description_id != job.id:
        logger.info(
            "Voice interview: resume %s linked to JD %s; call uses requested JD %s for job text",
            resume.id,
            resume.job_description_id,
            job.id,
        )

    if not resume and not job:
        return {"merged_jd_context": out_merged, "voice_dynamic_context": None}

    candidate_name = (out_merged.get("candidate_name") or None) or (
        _candidate_name_from_resume(resume) if resume else None
    )
    if isinstance(candidate_name, str):
        candidate_name = candidate_name.strip() or None

    lines: list[str] = [
        "This call is a job interview or candidate screening. Use the information below for the full conversation; do not claim you cannot see it.",
    ]
    if candidate_name:
        lines.append(
            f"The candidate's name from the application record is: {candidate_name}. "
            "Greet them by name and use it naturally. Do not ask for their name unless it sounds wrong or they correct you."
        )
    else:
        lines.append(
            "The candidate's name is not in the system. Politely ask what name they go by, then continue."
        )

    if job:
        lines.append("Role you are hiring for and requirements:\n" + _job_text_for_prompt(job))
        lines.append(
            "Ask one question at a time. Tailor questions to this job: required skills, "
            "relevant experience, and scenarios that test fit. Avoid unrelated small talk. "
            "When discussing experience, connect answers back to the role above."
        )
    else:
        lines.append(
            "A full job description was not found in the system; keep questions aligned with your normal agent goals."
        )

    excerpt = _resume_excerpt(resume) if resume else ""
    if excerpt:
        lines.append(
            "Resume text on file (for your reference only; do not read it aloud verbatim; use it to ask smarter questions):\n"
            + excerpt
        )

    addendum = "\n\n".join(lines)
    vdc: dict[str, Any] = {
        "system_prompt_addendum": addendum,
        "job_title": (job.job_title if job else None) or job_title_hint,
    }
    if candidate_name:
        vdc["candidate_name"] = candidate_name

    return {"merged_jd_context": out_merged, "voice_dynamic_context": vdc}
