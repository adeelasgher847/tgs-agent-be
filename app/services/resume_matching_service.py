from __future__ import annotations

from uuid import UUID

from app.models.job_description import JobDescription
from app.schemas.resume import (
    MatchComponentScore,
    MatchResponse,
    ParsedResume,
)


def _norm_skill(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch in "+#.")


def score_candidate(
    resume_id: UUID,
    job: JobDescription,
    parsed: ParsedResume,
) -> MatchResponse:
    required = [str(x) for x in (job.required_skills or [])]
    weight_map: dict[str, float] = {}
    for k, v in (job.skill_weight_matrix or {}).items():
        try:
            weight_map[_norm_skill(str(k))] = float(v)
        except (TypeError, ValueError):
            continue

    resume_skills = {_norm_skill(s.name) for s in parsed.skills}
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
            # fuzzy contains
            hit = any(nk and nk in rs for rs in resume_skills)
            if hit:
                skill_score_sum += w * 0.85
                weighted_hits[req] = w * 0.85
            else:
                missing.append(req.capitalize())

    skill_match_score = skill_score_sum / skill_weight_sum if skill_weight_sum > 0 else 1.0

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
        sum(crit_scores) / sum(crit_weights) if sum(crit_weights) > 0 else skill_match_score
    )
    overall = 0.65 * skill_match_score + 0.35 * crit_component

    return MatchResponse(
        resume_id=resume_id,
        job_description_id=job.id,
        overall_score=round(min(1.0, overall), 4),
        skill_match_score=round(skill_match_score, 4),
        criteria_breakdown=breakdown,
        missing_required_skills=missing,
        weighted_skill_hits={k: round(v, 4) for k, v in weighted_hits.items()},
    )

