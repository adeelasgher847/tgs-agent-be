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


def _trim_to(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def _name_from_raw_text_line(resume: Resume) -> str | None:
    """Best-effort when profile.name is missing: first line of raw_text often is the name."""
    raw_text = (resume.raw_text or "").strip()
    if not raw_text:
        return None
    first_line = raw_text.split("\n", 1)[0].strip()
    if not first_line or len(first_line) > 64:
        return None
    first_line_lower = first_line.lower()
    if (
        "@"
        in first_line
        or "http" in first_line_lower
        or "phone" in first_line_lower
        or "e-mail" in first_line_lower
        or "email" in first_line_lower
    ):
        return None
    if re.match(r"^\d", first_line):  # starts with number (address etc.)
        return None
    # e.g. "PROFESSIONAL SUMMARY" — reject long ALL-CAPS headers
    if first_line.isupper() and len(first_line) > 20:
        return None
    if not re.match(r"^[A-Za-z][A-Za-z\s.'-]*$", first_line):
        return None
    word_list = first_line.split()
    if not word_list or len(word_list) > 5:
        return None
    return first_line.strip()


def _candidate_name_from_resume(resume: Resume) -> str | None:
    parsed_json = resume.parsed_json
    if isinstance(parsed_json, dict):
        profile_dict = parsed_json.get("profile")
        if isinstance(profile_dict, dict) and profile_dict.get("name"):
            profile_name = str(profile_dict.get("name", "")).strip()
            if profile_name:
                return profile_name
        top_level_name = parsed_json.get("name")
        if top_level_name and str(top_level_name).strip():
            return str(top_level_name).strip()
    return _name_from_raw_text_line(resume)


def _job_text_for_prompt(job: JobDescription) -> str:
    job_title = (job.job_title or "").strip() or "Role"
    parts: list[str] = [f"Title: {job_title}"]
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
        raw_description = str(job.raw_text).strip()
        if len(raw_description) > _MAX_JD_CHARS:
            raw_description = f"{raw_description[:_MAX_JD_CHARS].rstrip()}..."
        parts.append("Full job description:\n" + raw_description)
    elif job.key_responsibilities:
        try:
            responsibilities_json = json.dumps(job.key_responsibilities, ensure_ascii=False)
            if len(responsibilities_json) > 2000:
                responsibilities_json = f"{responsibilities_json[:2000]}..."
            parts.append("Responsibilities (structured):\n" + responsibilities_json)
        except Exception:
            pass
    return "\n\n".join(parts)


def _resume_excerpt(resume: Resume) -> str:
    if not resume.raw_text or not str(resume.raw_text).strip():
        return ""
    resume_text = str(resume.raw_text).strip()
    if len(resume_text) > _MAX_RESUME_CHARS:
        return f"{resume_text[:_MAX_RESUME_CHARS].rstrip()}..."
    return resume_text


def _candidate_resume_highlights(resume: Resume) -> str:
    """
    Create a short, grounded candidate summary for the LLM.
    We intentionally keep this as structured bullets (not raw JSON) so the model
    uses it for JD-aligned questions without hallucinating missing fields.
    """
    parsed_json = resume.parsed_json if isinstance(resume.parsed_json, dict) else {}
    if not parsed_json:
        return ""

    profile_data = (
        parsed_json.get("profile") if isinstance(parsed_json.get("profile"), dict) else {}
    )
    name = str(profile_data.get("name") or "").strip() or _candidate_name_from_resume(resume) or ""

    years_total = parsed_json.get("years_experience_total")
    years_str = ""
    if years_total is not None:
        try:
            years_str = f"{float(years_total):g} years"
        except (TypeError, ValueError):
            years_str = str(years_total).strip()

    skills_raw = parsed_json.get("skills") if isinstance(parsed_json.get("skills"), list) else []
    skills: list[str] = []
    for skill_entry in skills_raw:
        if isinstance(skill_entry, dict):
            skill_name = str(skill_entry.get("name") or "").strip()
            if skill_name:
                skills.append(skill_name)
        elif isinstance(skill_entry, str):
            skill_text = skill_entry.strip()
            if skill_text:
                skills.append(skill_text)
    top_skills = ", ".join(skills[:10]) if skills else ""

    exp_raw = (
        parsed_json.get("experience") if isinstance(parsed_json.get("experience"), list) else []
    )
    recent_roles: list[str] = []
    for experience_entry in exp_raw[:5]:
        if not isinstance(experience_entry, dict):
            continue
        role = str(experience_entry.get("role") or "").strip()
        company = str(experience_entry.get("company") or "").strip()
        duration = str(experience_entry.get("duration") or "").strip()
        header = " - ".join([part for part in [role, company] if part]) or "Experience"
        if duration:
            header = f"{header} ({duration})"
        # Keep a couple responsibilities as "grounded evidence"
        resp = experience_entry.get("responsibilities")
        resp_bits: list[str] = []
        if isinstance(resp, list):
            for responsibility in resp:
                if isinstance(responsibility, str):
                    responsibility_text = responsibility.strip()
                    if responsibility_text:
                        resp_bits.append(responsibility_text)
        evidence = "; ".join(resp_bits[:2])
        if evidence:
            header = f"{header}: {evidence}"
        recent_roles.append(header)
    recent_roles_text = "\n".join(f"- {r}" for r in recent_roles[:4]) if recent_roles else ""

    projects_raw = parsed_json.get("projects") if isinstance(parsed_json.get("projects"), list) else []
    projects: list[str] = []
    for project_entry in projects_raw[:5]:
        if not isinstance(project_entry, dict):
            continue
        project_name = str(project_entry.get("name") or "").strip()
        project_description = str(project_entry.get("description") or "").strip()
        if project_name or project_description:
            projects.append(
                _trim_to(" - ".join([value for value in [project_name, project_description] if value]), 140)
            )
    projects_text = "\n".join(f"- {p}" for p in projects[:3]) if projects else ""

    edu_raw = parsed_json.get("education") if isinstance(parsed_json.get("education"), list) else []
    edu: list[str] = []
    for education_entry in edu_raw[:5]:
        if not isinstance(education_entry, dict):
            continue
        degree = str(education_entry.get("degree") or "").strip()
        institution = str(education_entry.get("institution") or "").strip()
        year = education_entry.get("year")
        year_text = ""
        if year is not None:
            try:
                year_text = str(int(year))
            except (TypeError, ValueError):
                year_text = str(year).strip()
        bits = [value for value in [degree, institution, year_text] if value]
        if bits:
            edu.append(_trim_to(" - ".join(bits), 120))
    edu_text = "\n".join(f"- {x}" for x in edu[:3]) if edu else ""

    languages_raw = (
        parsed_json.get("languages") if isinstance(parsed_json.get("languages"), list) else []
    )
    languages = [
        str(language).strip()
        for language in languages_raw
        if isinstance(language, str) and language.strip()
    ]
    langs_text = ", ".join(languages[:6]) if languages else ""

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

    certifications_raw = (
        job.required_certifications if isinstance(job.required_certifications, list) else []
    )
    certifications: list[str] = [
        str(certification).strip()
        for certification in certifications_raw
        if str(certification).strip()
    ]
    certs_text = ", ".join(certifications[:6]) if certifications else ""

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
    salary_currency = (job.currency or "").strip()
    if job.salary_min is not None:
        salary_bits.append(f"min {job.salary_min} {salary_currency}".strip())
    if job.salary_max is not None:
        salary_bits.append(f"max {job.salary_max} {salary_currency}".strip())
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
            job_title = (job.job_title or "").strip()
            if job_title:
                out_merged["jd_title"] = job_title
                job_title_hint = job_title
            if job.raw_text and str(job.raw_text).strip():
                job_summary = str(job.raw_text).strip()
                if len(job_summary) > 500:
                    job_summary = f"{job_summary[:500].rstrip()}..."
                out_merged["jd_summary"] = job_summary

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
            "0) Immediate opt-out/identity guard: if the person says they are not interested in this opportunity "
            "or says they are not the intended candidate/person, politely apologize and end the call immediately with [END_CALL]. "
            "1) Name cross-check: confirm candidate full name from profile/resume. "
            "If mismatch, re-confirm once politely; if still mismatched, politely close the call and end with [END_CALL]. "
            "2) Job intent and employment context: ask why this role. "
            "If the candidate is currently employed, ask why they want to switch now. "
            "3) Skill verification against JD: validate each major required skill using resume-grounded evidence. "
            "For each missing or weak required skill, ask short reason and current learning status. "
            "If critical JD skills clearly do not match, politely close the call and end with [END_CALL]. "
            "4) Ask exactly one simple analytical/logical question (general, not domain-heavy), then one short follow-up to understand thinking. "
            "Do not end the call only because the candidate gives a weak or wrong answer to this analytical question. "
            "5) Compensation discussion: ask candidate expected salary. "
            "If the candidate asks for budget, share JD budget range when available and confirm alignment/misalignment. "
            "Keep the conversation concise, professional, and evidence-based; avoid unrelated small talk. "
            "Call termination tokens (critical): Use [END_CALL] for every hang-up. "
            "Only when you have naturally completed the ENTIRE flow above through step 5 without an early rejection/opt-out/skill mismatch — "
            "i.e. successful screening — append the exact token [SCREENING_QUALIFIED] on its own immediately BEFORE [END_CALL] in your final reply (goodbye first, then tokens). "
            "Never output [SCREENING_QUALIFIED] for early closes, rejections, wrong candidate, or skill mismatch exits."
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
