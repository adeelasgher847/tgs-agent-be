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


def _trim_to(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return f"{s[:max_chars].rstrip()}..."


def _candidate_resume_highlights(resume: Resume) -> str:
    """
    Create a short, grounded candidate summary for the LLM.
    We intentionally keep this as structured bullets (not raw JSON) so the model
    uses it for JD-aligned questions without hallucinating missing fields.
    """
    pj = resume.parsed_json if isinstance(resume.parsed_json, dict) else {}
    if not pj:
        return ""

    profile = pj.get("profile") if isinstance(pj.get("profile"), dict) else {}
    name = str(profile.get("name") or "").strip() or _candidate_name_from_resume(resume) or ""

    years_total = pj.get("years_experience_total")
    years_str = ""
    if years_total is not None:
        try:
            years_str = f"{float(years_total):g} years"
        except (TypeError, ValueError):
            years_str = str(years_total).strip()

    skills_raw = pj.get("skills") if isinstance(pj.get("skills"), list) else []
    skills: list[str] = []
    for s in skills_raw:
        if isinstance(s, dict):
            n = str(s.get("name") or "").strip()
            if n:
                skills.append(n)
        elif isinstance(s, str):
            s2 = s.strip()
            if s2:
                skills.append(s2)
    top_skills = ", ".join(skills[:10]) if skills else ""

    exp_raw = pj.get("experience") if isinstance(pj.get("experience"), list) else []
    recent_roles: list[str] = []
    for e in exp_raw[:5]:
        if not isinstance(e, dict):
            continue
        role = str(e.get("role") or "").strip()
        company = str(e.get("company") or "").strip()
        duration = str(e.get("duration") or "").strip()
        dur_bits = [b for b in [duration] if b]
        dur = dur_bits[0] if dur_bits else ""
        header = " - ".join([p for p in [role, company] if p]) or "Experience"
        if dur:
            header = f"{header} ({dur})"
        # Keep a couple responsibilities as "grounded evidence"
        resp = e.get("responsibilities")
        resp_bits: list[str] = []
        if isinstance(resp, list):
            for r in resp:
                if isinstance(r, str):
                    t = r.strip()
                    if t:
                        resp_bits.append(t)
        evidence = "; ".join(resp_bits[:2])
        if evidence:
            header = f"{header}: {evidence}"
        recent_roles.append(header)
    recent_roles_text = "\n".join(f"- {r}" for r in recent_roles[:4]) if recent_roles else ""

    projects_raw = pj.get("projects") if isinstance(pj.get("projects"), list) else []
    projects: list[str] = []
    for p in projects_raw[:5]:
        if not isinstance(p, dict):
            continue
        pn = str(p.get("name") or "").strip()
        desc = str(p.get("description") or "").strip()
        if pn or desc:
            projects.append(_trim_to(" - ".join([b for b in [pn, desc] if b]), 140))
    projects_text = "\n".join(f"- {p}" for p in projects[:3]) if projects else ""

    edu_raw = pj.get("education") if isinstance(pj.get("education"), list) else []
    edu: list[str] = []
    for ed in edu_raw[:5]:
        if not isinstance(ed, dict):
            continue
        degree = str(ed.get("degree") or "").strip()
        inst = str(ed.get("institution") or "").strip()
        year = ed.get("year")
        year_str2 = ""
        if year is not None:
            try:
                year_str2 = str(int(year))
            except (TypeError, ValueError):
                year_str2 = str(year).strip()
        bits = [b for b in [degree, inst, year_str2] if b]
        if bits:
            edu.append(_trim_to(" - ".join(bits), 120))
    edu_text = "\n".join(f"- {x}" for x in edu[:3]) if edu else ""

    langs_raw = pj.get("languages") if isinstance(pj.get("languages"), list) else []
    langs = [str(l).strip() for l in langs_raw if isinstance(l, str) and l.strip()]
    langs_text = ", ".join(langs[:6]) if langs else ""

    lines: list[str] = ["Candidate resume highlights (parsed):"]
    if name:
        lines.append(f"- Name: {name}")
    if years_str:
        lines.append(f"- Total experience: {years_str}")
    if top_skills:
        lines.append(f"- Top skills: {top_skills}")
    if recent_roles_text:
        lines.append("- Recent/most relevant roles & evidence:\n" + recent_roles_text)
    if projects_text:
        lines.append("- Projects:\n" + projects_text)
    if edu_text:
        lines.append("- Education:\n" + edu_text)
    if langs_text:
        lines.append(f"- Languages: {langs_text}")

    # Hard cap to avoid ballooning the prompt for long resumes.
    return _trim_to("\n".join(lines), 1200)


def _jd_requirements_summary(job: JobDescription) -> str:
    required_skills_raw = job.required_skills if isinstance(job.required_skills, list) else []
    required_skills: list[str] = [str(s).strip() for s in required_skills_raw if str(s).strip()]
    required_skills_text = ", ".join(required_skills[:10]) if required_skills else ""

    certs_raw = job.required_certifications if isinstance(job.required_certifications, list) else []
    certs: list[str] = [str(c).strip() for c in certs_raw if str(c).strip()]
    certs_text = ", ".join(certs[:6]) if certs else ""

    responsibilities_raw = job.key_responsibilities if isinstance(job.key_responsibilities, list) else []
    responsibilities: list[str] = []
    for r in responsibilities_raw[:8]:
        if isinstance(r, str) and r.strip():
            responsibilities.append(_trim_to(r.strip(), 120))
    responsibilities_text = "\n".join(f"- {r}" for r in responsibilities[:4]) if responsibilities else ""

    years_bits: list[str] = []
    if job.years_experience_min is not None:
        years_bits.append(f"min {job.years_experience_min} years")
    if job.years_experience_max is not None:
        years_bits.append(f"max {job.years_experience_max} years")
    years_text = ""
    if years_bits:
        years_text = ", ".join(years_bits)

    salary_bits: list[str] = []
    currency = (job.currency or "").strip()
    if job.salary_min is not None:
        salary_bits.append(f"min {job.salary_min} {currency}".strip())
    if job.salary_max is not None:
        salary_bits.append(f"max {job.salary_max} {currency}".strip())
    salary_text = ", ".join(salary_bits) if salary_bits else ""

    lines: list[str] = ["Key JD requirements (structured, when available):"]
    if years_text:
        lines.append(f"- Experience: {years_text}")
    if salary_text:
        lines.append(f"- Budget range: {salary_text}")
    if required_skills_text:
        lines.append(f"- Required skills: {required_skills_text}")
    if certs_text:
        lines.append(f"- Required certifications: {certs_text}")
    if responsibilities_text:
        lines.append("- Responsibilities:\n" + responsibilities_text)
    return _trim_to("\n".join(lines), 900)


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
    highlights_added = False
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
        # Provide the model a compact "grounding" view of candidate evidence and JD requirements.
        # This reduces generic Q&A and increases resume->JD alignment.
        jd_summary = _jd_requirements_summary(job)
        if jd_summary:
            lines.append(jd_summary)

        if resume:
            highlights = _candidate_resume_highlights(resume)
            if highlights:
                lines.append(highlights)
                highlights_added = True
        lines.append(
            "Screening flow (strict order; ask one question at a time): "
            "1) Name cross-check: confirm candidate full name from profile/resume. "
            "If mismatch, re-confirm once politely before moving on. "
            "2) Job intent and switch reason: ask why this role and why they want to switch now. "
            "3) Skill verification against JD: validate each major required skill using resume-grounded evidence. "
            "For each missing or weak required skill, ask short reason and current learning status. "
            "If critical JD skills clearly do not match, politely close the call and end with [END_CALL]. "
            "4) Ask exactly one analytical scenario question relevant to this JD and probe depth with one short follow-up. "
            "5) Salary expectation and budget alignment: ask candidate expected salary, then share JD budget range when available and confirm alignment/misalignment. "
            "Keep the conversation concise, professional, and evidence-based; avoid unrelated small talk."
        )
    else:
        lines.append(
            "A full job description was not found in the system; keep questions aligned with your normal agent goals."
        )

    excerpt = _resume_excerpt(resume) if resume else ""
    if excerpt and not highlights_added:
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
