"""
LLM-based resume ↔ job description fit scoring.

Used together with rule-based scoring for hybrid matching; falls back if no API key or LLM errors.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from app.core.config import settings
from app.models.job_description import JobDescription
from app.schemas.resume import ParsedResume
from app.services.gemini_service import gemini_service
from app.services.openai_service import openai_service

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a senior technical recruiter. Your task is to score how well a candidate fits a specific open role.

Be strict and realistic:
- If the job is people-management, executive, or sales-led and the resume is only hands-on IC engineering (or the opposite mismatch), overall_fit must be low (typically under 0.35) unless the resume clearly proves relevant transferable leadership.
- If required tools/stacks differ materially with no evidence of equivalence, penalize skill_match.
- Do not inflate scores for generic overlap (e.g. both mention "team").
- Scores are probabilities of strong hire fit, not politeness.

Output ONLY valid JSON, no markdown fences, with this exact shape:
{
  "overall_fit": <float 0.0-1.0>,
  "skill_match": <float 0.0-1.0>,
  "role_alignment": <float 0.0-1.0>,
  "rationale": "<one or two factual sentences>",
  "red_flags": ["<short item>", ...]
}
"""


@dataclass
class AIMatchResult:
    overall_fit: float
    skill_match: float
    role_alignment: float
    rationale: str
    red_flags: list[str]
    model: str
    provider: str


def _truncate(text: str | None, max_len: int) -> str:
    if not text:
        return ""
    t = text.strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "…"


def _job_prompt_block(job: JobDescription) -> str:
    parts: list[str] = []
    parts.append(f"Title: {job.job_title or 'N/A'}")
    if job.required_skills:
        parts.append(f"Required skills: {json.dumps(job.required_skills, ensure_ascii=False)}")
    if job.keywords:
        parts.append(f"Keywords: {json.dumps(job.keywords, ensure_ascii=False)}")
    if job.years_experience_min is not None:
        parts.append(f"Min years experience: {job.years_experience_min}")
    if job.raw_text:
        parts.append(f"Full description:\n{_truncate(job.raw_text, 7000)}")
    elif job.key_responsibilities:
        parts.append(f"Responsibilities / details:\n{_truncate(json.dumps(job.key_responsibilities, ensure_ascii=False), 4000)}")
    return "\n\n".join(parts)


def _resume_prompt_block(parsed: ParsedResume) -> str:
    d = parsed.model_dump(mode="json")
    raw = d.get("raw_text")
    if raw and len(str(raw)) > 5000:
        d["raw_text"] = str(raw)[:5000] + "…[truncated]"
    ex = d.get("experience") or []
    if len(ex) > 8:
        d["experience"] = ex[:8] + [{"_note": f"{len(ex) - 8} additional roles omitted"}]
    sk = d.get("skills") or []
    if len(sk) > 60:
        d["skills"] = sk[:60] + [{"_note": f"{len(sk) - 60} more skills omitted"}]
    try:
        blob = json.dumps(d, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        blob = str(d)
    return _truncate(blob, 11000)


def _parse_llm_json(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if not text:
        return {}
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def _extract_scores(data: dict[str, Any]) -> AIMatchResult | None:
    if not data:
        return None
    overall = _clamp01(data.get("overall_fit"))
    skill = _clamp01(data.get("skill_match"))
    role = _clamp01(data.get("role_alignment"))
    if overall <= 0 and skill <= 0 and role <= 0:
        return None
    if overall <= 0:
        overall = round((skill + role) / 2, 4) if (skill or role) else 0.0
    rationale = str(data.get("rationale") or "").strip()[:1200]
    flags = data.get("red_flags")
    red_flags: list[str] = []
    if isinstance(flags, list):
        red_flags = [str(f).strip() for f in flags if str(f).strip()][:12]
    return AIMatchResult(
        overall_fit=round(overall, 4),
        skill_match=round(skill, 4),
        role_alignment=round(role, 4),
        rationale=rationale,
        red_flags=red_flags,
        model="",
        provider="",
    )


def llm_resume_job_fit(job: JobDescription, parsed: ParsedResume) -> AIMatchResult | None:
    """
    Call configured LLM(s) to score fit. Returns None if unavailable or parsing failed.
    """
    max_chars = int(getattr(settings, "RECRUIT_MATCH_MAX_PROMPT_CHARS", 14000))
    jd_block = _truncate(_job_prompt_block(job), max_chars)
    cv_block = _truncate(_resume_prompt_block(parsed), max_chars)
    user_content = f"""## JOB\n{jd_block}\n\n## CANDIDATE_RESUME_JSON\n{cv_block}"""

    temp = float(getattr(settings, "RECRUIT_MATCH_LLM_TEMPERATURE", 0.12))
    max_tokens = int(getattr(settings, "RECRUIT_MATCH_LLM_MAX_TOKENS", 600))
    provider_pref = str(getattr(settings, "RECRUIT_MATCH_LLM_PROVIDER", "auto")).lower()

    def try_openai() -> AIMatchResult | None:
        if not (settings.OPENAI_API_KEY or "").strip():
            return None
        model = str(getattr(settings, "RECRUIT_MATCH_OPENAI_MODEL", "gpt-4o-mini"))
        try:
            raw = openai_service.chat_completion(
                messages=[{"role": "user", "content": user_content}],
                system_prompt=_SYSTEM_PROMPT,
                model_name=model,
                temperature=temp,
                max_tokens=max_tokens,
            )
            data = _parse_llm_json(raw.get("content", ""))
            out = _extract_scores(data)
            if out:
                out.model = raw.get("model") or model
                out.provider = "openai"
            return out
        except Exception as exc:
            log.warning("OpenAI resume match failed: %s", exc)
            return None

    def try_gemini() -> AIMatchResult | None:
        if not (settings.GEMINI_API_KEY or "").strip():
            return None
        model = str(getattr(settings, "RECRUIT_MATCH_GEMINI_MODEL", "gemini-1.5-flash"))
        try:
            raw = gemini_service.chat_completion(
                messages=[{"role": "user", "content": user_content}],
                system_prompt=_SYSTEM_PROMPT,
                model_name=model,
                temperature=temp,
                max_tokens=max_tokens,
            )
            data = _parse_llm_json(raw.get("content", ""))
            out = _extract_scores(data)
            if out:
                out.model = model
                out.provider = "gemini"
            return out
        except Exception as exc:
            log.warning("Gemini resume match failed: %s", exc)
            return None

    if provider_pref == "openai":
        return try_openai() or try_gemini()
    if provider_pref == "gemini":
        return try_gemini() or try_openai()
    # auto
    return try_openai() or try_gemini()


def ai_match_enabled() -> bool:
    return bool((settings.OPENAI_API_KEY or "").strip() or (settings.GEMINI_API_KEY or "").strip())
