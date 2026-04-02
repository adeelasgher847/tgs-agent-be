from __future__ import annotations

import re
from uuid import UUID

from app.core.config import settings
from app.models.job_description import JobDescription
from app.schemas.resume import (
    MatchComponentScore,
    MatchResponse,
    ExperienceItem,
    ParsedResume,
    ProjectItem,
)
from app.services.resume_ai_match_service import ai_match_enabled, llm_resume_job_fit

_STOPWORDS = frozenset(
    """
    the a an and or to of for in on at by with from as is was are are be been being
    has have had do does did will would could should may might must can need our your
    their they them we us you this that these those it its not no yes all any some
    per plus into about over under more most less least other than then such both each
    every very also just only own same so if but what which who when where how why
    including include included well work working strong excellent good great team
    ability experience years year month months day days time responsibilities role
    looking seeking candidate candidates position opportunity company based remote
    hybrid full onsite etc eg ie via using use used skills skill
    """.split()
)

# Single-token signals (post tokenization / normalization)
_MANAGEMENT_TOKENS = frozenset(
    """
    manager management managing director executive supervision supervisor
    leadership people stakeholders stakeholder budget pmo governance
    chief officer superintendent president vice reports hiring strategic
    """.split()
)

_IC_TECH_TOKENS = frozenset(
    """
    frontend front-end backend back-end fullstack full-stack software web mobile
    react angular vue svelte nextjs next nuxt nodejs node typescript javascript
    python java kotlin swift golang rust ruby rails django flask fastapi spring
    engineer developer development devops kubernetes docker aws azure gcp cloud
    terraform ansible jenkins ci cd git sql nosql mongodb postgres redis graphql
    rest api microservice css html sass scss webpack redux webpack jest cypress
    ui ux design figma protobuf grpc kafka elasticsearch solr machine ml data
    scientist analytics etl pandas numpy tensorflow pytorch llm ai
    """.split()
)

_SALES_MARKETING_TOKENS = frozenset(
    """
    sales account quota revenue crm hubspot salesforce sdr bdr pipeline prospect
    marketing growth brand demand gen campaign seo sem content social media gtm
    """.split()
)

_TOKEN_SYNONYMS = {
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "reactjs": "react",
    "vuejs": "vue",
    "node": "nodejs",
}


def _norm_skill(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch in "+#.")


def _normalize_token(word: str) -> str:
    w = word.lower().strip()
    return _TOKEN_SYNONYMS.get(w, w)


def _collect_tokens(*text_parts: str | None) -> set[str]:
    blob = " ".join(p or "" for p in text_parts if p)
    out: set[str] = set()
    for raw in re.findall(r"[a-z0-9+#.]{2,}", blob.lower()):
        t = _normalize_token(raw)
        if t not in _STOPWORDS and len(t) >= 2:
            out.add(t)
    return out


def _jd_text_parts(job: JobDescription) -> list[str]:
    parts: list[str] = [job.job_title or "", job.raw_text or "", job.education_requirements or ""]
    rs = job.required_skills or []
    if isinstance(rs, list):
        parts.append(" ".join(str(x) for x in rs))
    kw = job.keywords or []
    if isinstance(kw, list):
        parts.append(" ".join(str(x) for x in kw))
    ex = job.extracted_skills or []
    if isinstance(ex, list):
        for item in ex:
            if isinstance(item, dict) and item.get("skill"):
                parts.append(str(item["skill"]))
            elif isinstance(item, str):
                parts.append(item)
    kr = job.key_responsibilities or []
    if isinstance(kr, list):
        parts.append(" ".join(str(x) for x in kr))
    elif isinstance(kr, str):
        parts.append(kr)
    return parts


def _resume_text_parts(parsed: ParsedResume) -> list[str]:
    parts: list[str] = [parsed.raw_text or ""]
    for s in parsed.skills:
        parts.append(s.name)
    for e in parsed.experience:
        parts.extend(_experience_texts(e))
    for p in parsed.projects:
        parts.extend(_project_texts(p))
    for c in parsed.certifications:
        parts.append(c.name)
    for d in parsed.education:
        if d.degree:
            parts.append(d.degree)
        if d.institution:
            parts.append(d.institution)
    for lang in parsed.languages:
        parts.append(str(lang))
    return parts


def _experience_texts(e: ExperienceItem) -> list[str]:
    texts = [e.role or "", e.company or ""]
    for r in e.responsibilities or []:
        texts.append(r)
    return texts


def _project_texts(p: ProjectItem) -> list[str]:
    texts = [p.name or "", p.description or ""]
    for t in p.technologies or []:
        texts.append(t)
    return texts


def _resume_blob_lower(parsed: ParsedResume) -> str:
    return " ".join(_resume_text_parts(parsed)).lower()


def _jd_token_coverage(jd_tokens: set[str], resume_tokens: set[str]) -> float:
    if not jd_tokens:
        return 0.45
    inter = jd_tokens & resume_tokens
    return len(inter) / len(jd_tokens)


def _title_alignment(job_title: str, resume_blob_lower: str, resume_tokens: set[str]) -> float:
    title = " ".join((job_title or "").lower().split())
    if len(title) < 2:
        return 0.45
    if title in resume_blob_lower:
        return 1.0
    title_tok = _collect_tokens(job_title)
    if not title_tok:
        return 0.45
    overlap = len(title_tok & resume_tokens) / len(title_tok)
    return max(0.0, min(1.0, overlap))


def _text_alignment_score(job: JobDescription, parsed: ParsedResume) -> float:
    jd_parts = _jd_text_parts(job)
    jd_tokens = _collect_tokens(*jd_parts)
    resume_parts = _resume_text_parts(parsed)
    resume_blob = " ".join(resume_parts).lower()
    resume_tokens = _collect_tokens(*resume_parts)

    cov = _jd_token_coverage(jd_tokens, resume_tokens)
    title_part = _title_alignment(job.job_title or "", resume_blob, resume_tokens)
    return round(min(1.0, 0.5 * cov + 0.5 * title_part), 4)


def _structured_skills_match(
    job: JobDescription,
    parsed: ParsedResume,
) -> tuple[float, list[str], dict[str, float]]:
    """Explicit required_skills list scoring; returns (score 0-1, missing, weighted_hits)."""
    required = [str(x) for x in (job.required_skills or [])]
    weight_map: dict[str, float] = {}
    for k, v in (job.skill_weight_matrix or {}).items():
        try:
            weight_map[_norm_skill(str(k))] = float(v)
        except (TypeError, ValueError):
            continue

    resume_skills = {_norm_skill(s.name) for s in parsed.skills}
    resume_blob = _resume_blob_lower(parsed)
    missing: list[str] = []
    weighted_hits: dict[str, float] = {}
    skill_score_sum = 0.0
    skill_weight_sum = 0.0

    for req in required:
        nk = _norm_skill(req)
        w = weight_map.get(nk, 1.0)
        skill_weight_sum += w
        if nk and nk in resume_skills:
            skill_score_sum += w
            weighted_hits[req] = w
        else:
            hit = any(nk and nk in rs for rs in resume_skills)
            if hit:
                skill_score_sum += w * 0.85
                weighted_hits[req] = w * 0.85
            elif nk and nk in resume_blob:
                skill_score_sum += w * 0.7
                weighted_hits[req] = w * 0.7
            else:
                missing.append(req.capitalize())

    if skill_weight_sum <= 0:
        # No explicit requirements → no structured list signal (avoid faking a perfect score)
        return 0.0, missing, weighted_hits
    return skill_score_sum / skill_weight_sum, missing, weighted_hits


def _mismatch_penalty(
    job: JobDescription,
    jd_tokens: set[str],
    resume_tokens: set[str],
    resume_blob_lower: str,
) -> float:
    """Reduce scores when JD domain and resume domain strongly disagree."""
    m = 1.0
    title_l = (job.job_title or "").lower()
    title_words = _collect_tokens(job.job_title)

    jd_m = jd_tokens & _MANAGEMENT_TOKENS
    jd_t = jd_tokens & _IC_TECH_TOKENS
    res_m = resume_tokens & _MANAGEMENT_TOKENS
    res_t = resume_tokens & _IC_TECH_TOKENS

    title_mgmt_hint = bool(title_words & _MANAGEMENT_TOKENS) or (
        "vice president" in title_l or "head of" in title_l
    )

    # Management / leadership JD vs hands-on IC resume (e.g. "Manager" vs "Frontend developer")
    if (len(jd_m) >= 1 or title_mgmt_hint) and len(res_m) == 0 and len(res_t) >= 2:
        m *= 0.36

    # Strong engineering JD vs sales-only resume
    jd_s = jd_tokens & _SALES_MARKETING_TOKENS
    res_s = resume_tokens & _SALES_MARKETING_TOKENS
    if len(jd_t) >= 3 and len(res_s) >= 3 and len(res_t) <= 1:
        m *= 0.4

    # Sales/marketing JD vs pure IC eng resume
    if len(jd_s) >= 2 and len(res_s) == 0 and len(res_t) >= 3:
        m *= 0.38

    # IC engineering JD vs management-heavy resume with little engineering signal
    if len(jd_t) >= 2 and len(res_t) <= 0 and len(res_m) >= 2:
        m *= 0.42

    # Title contains a distinctive word not present anywhere in resume (extra nudge)
    for marker in ("manager", "director", "sales", "account executive"):
        if marker in title_l and marker not in resume_blob_lower and len(title_l) > 4:
            m *= 0.72
            break

    return max(0.18, min(1.0, m))


def _blend_ai_rules(ai_val: float, rules_val: float, weight_ai: float) -> float:
    """Ensemble LLM + rules; dampen LLM when it is much more optimistic than rules."""
    w = max(0.0, min(1.0, weight_ai))
    blended = w * ai_val + (1 - w) * rules_val
    gap = ai_val - rules_val
    if gap > 0.42:
        blended = (w * 0.55) * ai_val + (1 - w * 0.55) * rules_val
    return round(min(1.0, max(0.0, blended)), 4)


def _score_candidate_rules(
    resume_id: UUID,
    job: JobDescription,
    parsed: ParsedResume,
) -> MatchResponse:
    required = [str(x) for x in (job.required_skills or [])]
    has_skill_list = bool(required)

    structured_score, missing, weighted_hits = _structured_skills_match(job, parsed)
    text_align = _text_alignment_score(job, parsed)

    if has_skill_list:
        fit_channel = round(0.52 * structured_score + 0.48 * text_align, 4)
    else:
        # No explicit list: do not assume perfect skill match — lean on JD text vs resume
        fit_channel = round(0.22 * structured_score + 0.78 * text_align, 4)

    jd_tokens = _collect_tokens(*_jd_text_parts(job))
    resume_parts = _resume_text_parts(parsed)
    resume_blob_lower = " ".join(resume_parts).lower()
    resume_tokens = _collect_tokens(*resume_parts)

    penalty = _mismatch_penalty(job, jd_tokens, resume_tokens, resume_blob_lower)

    criteria = job.matching_criteria or {}
    breakdown: list[MatchComponentScore] = []
    crit_scores: list[float] = []
    crit_weights: list[float] = []

    for name, spec in criteria.items():
        if not isinstance(spec, dict):
            continue
        weight = float(spec.get("weight", 1.0))
        crit_type = str(spec.get("type", "skill"))
        detail = ""
        matched = False
        score = 0.0
        resume_skills = {_norm_skill(s.name) for s in parsed.skills}
        if crit_type == "skill":
            target = str(spec.get("skill", ""))
            nk = _norm_skill(target)
            matched = nk in resume_skills or any(nk in rs for rs in resume_skills)
            score = 1.0 if matched else 0.0
            detail = f"Required skill `{target}` {'found' if matched else 'missing'}"
        elif crit_type == "years_experience":
            min_years = float(spec.get("min_years", 0))
            y = parsed.years_experience_total
            if y is None:
                score = 0.5
                detail = "Years of experience unknown; partial credit"
            else:
                matched = y >= min_years
                score = 1.0 if matched else max(0.0, min(1.0, y / max(min_years, 1e-6)))
                detail = f"Candidate ~{y}y vs min {min_years}y"
        else:
            score = 0.5
            detail = f"Unknown criterion type `{crit_type}` — neutral score"

        breakdown.append(
            MatchComponentScore(
                criterion=name,
                weight=weight,
                score=score,
                matched=matched,
                detail=detail,
            )
        )
        crit_scores.append(score * weight)
        crit_weights.append(weight)

    crit_component = (
        sum(crit_scores) / sum(crit_weights) if sum(crit_weights) > 0 else fit_channel
    )
    core = 0.58 * fit_channel + 0.42 * crit_component
    overall = round(min(1.0, max(0.0, core * penalty)), 4)

    return MatchResponse(
        resume_id=resume_id,
        job_description_id=job.id,
        overall_score=overall,
        skill_match_score=fit_channel,
        criteria_breakdown=breakdown,
        missing_required_skills=missing,
        weighted_skill_hits={k: round(v, 4) for k, v in weighted_hits.items()},
        match_source="rules",
    )


def score_candidate(
    resume_id: UUID,
    job: JobDescription,
    parsed: ParsedResume,
    *,
    match_mode: str | None = None,
) -> MatchResponse:
    """
    Score resume vs job. Uses RECRUIT_MATCH_MODE / API keys when match_mode is None.
    hybrid blends LLM judgement with rules; ai prefers LLM (rules kept in breakdown); rules skips LLM.
    """
    rules_mr = _score_candidate_rules(resume_id, job, parsed)
    configured = str(getattr(settings, "RECRUIT_MATCH_MODE", "hybrid") or "hybrid").lower().strip()
    mode = (match_mode or configured).lower().strip()
    if mode not in ("rules", "ai", "hybrid"):
        mode = "hybrid"

    if mode == "rules" or not ai_match_enabled():
        return rules_mr

    ai = llm_resume_job_fit(job, parsed)
    if ai is None:
        return rules_mr.model_copy(
            update={
                "match_source": "rules",
                "rules_baseline_overall": rules_mr.overall_score,
            }
        )

    w = float(getattr(settings, "RECRUIT_MATCH_AI_WEIGHT", 0.68))

    if mode == "ai":
        return MatchResponse(
            resume_id=resume_id,
            job_description_id=job.id,
            overall_score=ai.overall_fit,
            skill_match_score=ai.skill_match,
            criteria_breakdown=rules_mr.criteria_breakdown,
            missing_required_skills=rules_mr.missing_required_skills,
            weighted_skill_hits=rules_mr.weighted_skill_hits,
            match_source="ai",
            rules_baseline_overall=rules_mr.overall_score,
            ai_rationale=ai.rationale,
            ai_red_flags=ai.red_flags,
            ai_model=ai.model,
            ai_provider=ai.provider,
        )

    overall_b = _blend_ai_rules(ai.overall_fit, rules_mr.overall_score, w)
    skill_b = _blend_ai_rules(ai.skill_match, rules_mr.skill_match_score, w)

    return MatchResponse(
        resume_id=resume_id,
        job_description_id=job.id,
        overall_score=overall_b,
        skill_match_score=skill_b,
        criteria_breakdown=rules_mr.criteria_breakdown,
        missing_required_skills=rules_mr.missing_required_skills,
        weighted_skill_hits=rules_mr.weighted_skill_hits,
        match_source="hybrid",
        rules_baseline_overall=rules_mr.overall_score,
        ai_rationale=ai.rationale,
        ai_red_flags=ai.red_flags,
        ai_model=ai.model,
        ai_provider=ai.provider,
    )
