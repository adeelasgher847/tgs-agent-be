from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.resume import ParseStatus, Resume
from app.schemas.resume import ParseMode, ParsedResume
from app.services.openai_service import openai_service
from app.services.resume_extraction_service import ExtractionError, extract_text_from_file
from app.services.resume_merge_service import merge_parsed
from app.utils.resume_rules_parser import parse_rules

log = logging.getLogger(__name__)


def collect_warnings(parsed: ParsedResume) -> list[str]:
    w: list[str] = []
    if not parsed.profile.email:
        w.append("Email not detected")
    if not parsed.profile.phone:
        w.append("Phone not detected")
    if not parsed.skills:
        w.append("No structured skills extracted")
    if not parsed.experience:
        w.append("Experience section weak or missing")
    if not parsed.education:
        w.append("Education section weak or missing")
    return w


def _openai_pricing_per_1k(model: str) -> tuple[float, float]:
    m = model.lower()
    if "gpt-4o-mini" in m:
        return 0.00015, 0.0006
    if "gpt-4o" in m:
        return 0.0025, 0.01
    return 0.00015, 0.0006


def _parse_with_openai(raw_text: str, model_name: str) -> tuple[ParsedResume | None, int | None, int | None, float, float, str, str]:
    """
    Use existing OpenAIService to get structured JSON and map to ParsedResume.
    Returns (parsed, tokens_in, tokens_out, input_cost, output_cost, model_used, provider_used).
    """
    system_prompt = (
        "You extract structured candidate data from resume text.\n"
        "Return ONLY valid JSON matching this exact shape (no markdown, no commentary):\n"
        "{\n"
        '  "profile": {"name": string|null, "email": string|null, "phone": string|null, "location": string|null, "links": string[]},\n'
        '  "skills": [{"name": string, "confidence": number}],\n'
        '  "experience": [{"role": string|null, "company": string|null, "duration": string|null, "responsibilities": string[]}],\n'
        '  "education": [{"degree": string|null, "institution": string|null, "year": number|string|null}],\n'
        '  "certifications": [{"name": string, "issuer": string|null, "year": number|string|null}],\n'
        '  "projects": [{"name": string, "description": string|null, "technologies": string[]}],\n'
        '  "languages": string[],\n'
        '  "years_experience_total": number|null\n'
        "}\n"
        "Use null where unknown. Keep responsibilities concise bullet strings."
    )
    prompt = f"Resume text:\n---\n{raw_text[:24000]}\n---\n"

    resp = openai_service.generate_text(
        prompt=prompt,
        system_prompt=system_prompt,
        model_name=model_name,
        temperature=0.1,
        max_tokens=1500,
    )
    content = resp["content"]
    usage = resp.get("usage") or {}
    tokens_in = usage.get("prompt_tokens")
    tokens_out = usage.get("completion_tokens")
    in_rate, out_rate = _openai_pricing_per_1k(model_name)
    tin = tokens_in or 0
    tout = tokens_out or 0
    input_cost = (tin / 1000) * in_rate
    output_cost = (tout / 1000) * out_rate

    import json
    import re

    raw = content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```\s*$", "", raw)
    blob = json.loads(raw)

    from app.schemas.resume import (
        CertificationItem,
        EducationItem,
        ExperienceItem,
        ParsedResume,
        ParseSource,
        ProfileBlock,
        ProjectItem,
        SkillItem,
    )

    prof = blob.get("profile") or {}
    profile = ProfileBlock(
        name=prof.get("name"),
        email=prof.get("email"),
        phone=prof.get("phone"),
        location=prof.get("location"),
        links=list(prof.get("links") or []),
    )
    skills_raw = blob.get("skills") or []
    skills: list[SkillItem] = []
    for s in skills_raw:
        if isinstance(s, dict) and s.get("name"):
            conf = float(s.get("confidence") or 0.75)
            conf = max(0.0, min(1.0, conf))
            skills.append(SkillItem(name=str(s["name"]).strip(), source="AI", confidence=conf))
    exp_raw = blob.get("experience") or []
    experience: list[ExperienceItem] = []
    for e in exp_raw:
        if not isinstance(e, dict):
            continue
        experience.append(
            ExperienceItem(
                role=e.get("role"),
                company=e.get("company"),
                duration=e.get("duration"),
                responsibilities=list(e.get("responsibilities") or []),
            )
        )
    edu_raw = blob.get("education") or []
    education: list[EducationItem] = []
    for ed in edu_raw:
        if not isinstance(ed, dict):
            continue
        education.append(
            EducationItem(
                degree=ed.get("degree"),
                institution=ed.get("institution"),
                year=ed.get("year"),
            )
        )
    cert_raw = blob.get("certifications") or []
    certifications: list[CertificationItem] = []
    for c in cert_raw:
        if isinstance(c, dict) and c.get("name"):
            certifications.append(
                CertificationItem(
                    name=c["name"],
                    issuer=c.get("issuer"),
                    year=c.get("year"),
                )
            )
    proj_raw = blob.get("projects") or []
    projects: list[ProjectItem] = []
    for p in proj_raw:
        if isinstance(p, dict) and p.get("name"):
            projects.append(
                ProjectItem(
                    name=p["name"],
                    description=p.get("description"),
                    technologies=list(p.get("technologies") or []),
                )
            )
    languages = [str(x) for x in (blob.get("languages") or []) if x]
    yext = blob.get("years_experience_total")
    years_total = float(yext) if yext is not None else None

    parsed = ParsedResume(
        profile=profile,
        skills=skills,
        experience=experience,
        education=education,
        certifications=certifications,
        projects=projects,
        languages=languages,
        years_experience_total=years_total,
        raw_text=None,
        parse_confidence=0.88,
        parse_source=ParseSource.AI,
        parser_version="",
        model_name=model_name,
        provider="openai",
    )
    return parsed, tokens_in, tokens_out, input_cost, output_cost, model_name, "openai"


def run_parse_for_resume(
    session: Session,
    resume_id: UUID,
    parse_mode: ParseMode = ParseMode.hybrid,
) -> Resume:
    res = session.get(Resume, resume_id)
    if res is None:
        raise ValueError("Resume not found")

    res.status = ParseStatus.PROCESSING
    res.error_message = None
    session.flush()

    path = Path(res.storage_path)
    ext = Path(res.original_filename).suffix.lower()
    try:
        raw_text, extraction_cost = extract_text_from_file(
            path, res.content_type, ext
        )
    except ExtractionError as e:
        res.status = ParseStatus.FAILED
        res.error_message = str(e)
        session.flush()
        return res

    rules = parse_rules(raw_text, parser_version="1.0.0")
    rules.raw_text = raw_text

    use_llm = parse_mode in {ParseMode.llm, ParseMode.hybrid}
    use_ai = use_llm and bool(settings.OPENAI_API_KEY)

    llm_input_cost = 0.0
    llm_output_cost = 0.0
    tokens_in = None
    tokens_out = None
    ai_parsed = None
    model_used = None
    provider_used = "rules"

    model_for_call = "gpt-4o-mini"

    if use_ai and settings.OPENAI_API_KEY:
        try:
            (
                ai_parsed,
                tokens_in,
                tokens_out,
                llm_input_cost,
                llm_output_cost,
                model_used,
                provider_used,
            ) = _parse_with_openai(raw_text, model_for_call)
        except Exception as e:
            log.warning("LLM parse failed, rules-only fallback: %s", e, exc_info=False)

    if parse_mode == ParseMode.rules:
        merged = rules
    elif parse_mode == ParseMode.llm:
        merged = ai_parsed or rules
    else:
        merged = merge_parsed(rules, ai_parsed)
    merged.raw_text = raw_text
    merged.parser_version = "1.0.0"
    if model_used:
        merged.model_name = model_used
        merged.provider = provider_used

    warnings = collect_warnings(merged)
    res.raw_text = raw_text
    res.parsed_json = merged.model_dump(mode="json")
    res.warnings = warnings
    res.parse_confidence = merged.parse_confidence
    res.parse_source = merged.parse_source.value
    res.parser_version = merged.parser_version
    res.model_name = merged.model_name
    res.provider = merged.provider
    res.status = ParseStatus.READY
    session.flush()
    return res

