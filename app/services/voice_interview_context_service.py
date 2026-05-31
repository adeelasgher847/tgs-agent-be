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
        # Persist on outbound jd_context so stream handler / qualification only run recruitment logic for JD calls.
        out_merged["recruitment_jd_screening"] = True
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
        _job_title_display = (job.job_title or "").strip() or "the open role"
        _name_step_detail = (
            f'Ask what name they go by or confirm their full name. You have "{candidate_name}" on file — '
            "use it naturally and ask them to confirm it is correct."
            if candidate_name
            else "Ask what name they go by or ask for their full name for this application."
        )
        lines.append(
            "RECRUITMENT CALL FLOW — FOLLOW IN ORDER:\n\n"
            "PRIORITY ZERO — BEFORE ANY SCREENING QUESTION:\n"
            "If the user says they are NOT interested, NOT available, wrong number, wrong person, wrong call, "
            "stop calling, don't call again, or clearly cannot continue — respond with ONE short polite sentence ONLY "
            "and end that same reply with [END_CALL]. Do not ask anything else.\n\n"
            "NOTE: On this outbound call a short intro may play automatically when they answer. "
            "If that already happened, do NOT repeat the full intro — acknowledge briefly if needed, then continue "
            "(confirm name / screening steps).\n\n"
            "1) OPENING (only if no automated intro played yet):\n"
            f"   Briefly introduce yourself as calling from hiring about screening for the {_job_title_display} role — "
            "one or two sentences. Then ask a clear permission check like: \"Is this still a good time for a 3-5 minute screening call?\"\n\n"
            f"2) NAME:\n   {_name_step_detail}\n\n"
            "3) INTEREST & AVAILABILITY — END ONLY FOR TRUE HARD-STOP:\n"
            "   HARD-STOP OVERRIDE: if candidate says \"I am not interested\" (or equivalent), immediately end the call in the same turn. "
            "Do not ask any follow-up, do not persuade, do not continue screening, and do not attempt to reschedule.\n"
            "   If they clearly say they are NOT interested in this opportunity, NOT interested in this role, or do NOT want to continue — "
            "respond with ONE short polite line (thank them / wish them well) and end the same reply with [END_CALL]. Do not persuade. Do not ask screening questions.\n"
            "   If they clearly say they are NOT available (cannot talk now, not available for this role/process, cannot proceed in a way that means \"stop\") — "
            "respond with ONE short polite line and end with [END_CALL]. Do not insist on continuing.\n"
            "   If they only say \"busy right now\" but are willing to continue later — offer quick reschedule/close politely; do not force full screening in that moment.\n\n"
            "4) ONLY IF STILL ON CALL AFTER STEPS 1-3 — COMPLETE role-specific screening:\n"
            "   Interview pacing is MANDATORY: ask exactly ONE question at a time, wait for the candidate's complete answer, then move on.\n"
            "   Never jump to the next topic while they are still answering. If answer is unclear/incomplete, ask exactly one short follow-up for the SAME topic, then continue.\n"
            "   Always acknowledge the current answer in one short phrase (for example: \"Got it\" / \"Thanks for clarifying\") before asking the next question.\n"
            "   Keep each question short and specific; avoid double questions in one turn.\n"
            "   Do not skip required sub-steps (a-e) unless a hard-stop condition from step 3 occurs.\n"
            "   a) Name cross-check: confirm candidate full name from profile/resume. "
            "If mismatch, re-confirm once politely; if still mismatched, politely close and end with [END_CALL].\n"
            "   b) Job intent and employment context: ask why this role; if currently employed, ask why they want to switch now.\n"
            "   c) Skill verification against JD: validate each major required skill using resume-grounded evidence. "
            "For each missing or weak required skill, ask short reason and current learning status. "
            "If critical JD skills clearly do not match, politely close and end with [END_CALL].\n"
            "   d) Ask exactly one simple analytical/logical question (general, not domain-heavy), then one short follow-up to understand thinking. "
            "Do not end the call only because the candidate gives a weak or wrong answer to this question.\n"
            "   e) Compensation: ask candidate expected salary. "
            "If the candidate asks for budget, share JD budget range when available and confirm alignment/misalignment.\n\n"
            "5) ENDING TOKENS (critical — never speak these aloud, they are system signals only):\n"
            "   Use [END_CALL] every time you hang up for any reason.\n"
            "   Only when you have naturally completed the ENTIRE flow through step 4e without any early rejection, opt-out, or skill mismatch — "
            "i.e. a successful screening — append [SCREENING_QUALIFIED] immediately BEFORE [END_CALL] in your final reply (goodbye first, then tokens). "
            "Never output [SCREENING_QUALIFIED] for early closes, rejections, wrong candidate, or skill mismatch exits.\n\n"
            "RULES: Never waste call time on someone who already said they are not interested or not available. "
            "Keep the conversation structured, professional, and evidence-based; prioritize complete candidate answers over speed; avoid unrelated small talk.\n\n"
            "SCREENING COMPLETENESS CHECK (BEFORE FINAL GOODBYE):\n"
            "- Confirm you have covered steps 4a through 4e in order.\n"
            "- If any required step was missed, ask that missing question first before ending.\n"
            "- Do not mark screening as successful unless all required steps are completed.\n\n"
            "NO REPEATED QUESTIONS:\n"
            "- Once a question has a usable answer, mark that topic as CLOSED and never ask the same question again in this call.\n"
            "- Do not re-ask an answered question with different wording (no paraphrased repeats).\n"
            "- Before each reply, mentally review the conversation history shown above. Never ask the same screening question twice "
            "(name, interest, salary expectation, a specific skill, or the same analytical puzzle).\n"
            "- If the candidate already answered a topic, acknowledge briefly (e.g. \"Got it, thanks\") and move to the NEXT topic in the flow.\n"
            "- Do not re-ask because they were brief or you forgot — check history first.\n"
            "- Step 4 sub-steps (a–e) each cover distinct ground once; do not circle back unless they volunteer new information."
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
