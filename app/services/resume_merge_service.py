from __future__ import annotations

from app.schemas.resume import ParsedResume, ParseSource, SkillItem


def merge_parsed(rules: ParsedResume, ai: ParsedResume | None) -> ParsedResume:
    if ai is None:
        out = rules.model_copy(deep=True)
        out.parse_source = ParseSource.RULES
        return out

    profile = rules.profile.model_copy(deep=True)
    if not profile.email and ai.profile.email:
        profile.email = ai.profile.email
    if not profile.phone and ai.profile.phone:
        profile.phone = ai.profile.phone
    if not profile.name and ai.profile.name:
        profile.name = ai.profile.name
    if not profile.location and ai.profile.location:
        profile.location = ai.profile.location
    merged_links = list(dict.fromkeys((profile.links or []) + (ai.profile.links or [])))
    profile.links = merged_links

    skills = _merge_skills(rules.skills, ai.skills)

    experience = ai.experience if len(ai.experience) >= len(rules.experience) else rules.experience
    education = ai.education if len(ai.education) >= len(rules.education) else rules.education

    certifications = ai.certifications or rules.certifications
    projects = ai.projects or rules.projects
    languages = list(dict.fromkeys((rules.languages or []) + (ai.languages or [])))

    y = ai.years_experience_total or rules.years_experience_total

    conf = (rules.parse_confidence * 0.35 + ai.parse_confidence * 0.65)

    return ParsedResume(
        profile=profile,
        skills=skills,
        experience=experience,
        education=education,
        certifications=certifications,
        projects=projects,
        languages=languages,
        years_experience_total=y,
        raw_text=rules.raw_text,
        parse_confidence=min(0.97, conf),
        parse_source=ParseSource.HYBRID,
        parser_version=rules.parser_version,
        model_name=ai.model_name,
        provider=ai.provider or rules.provider,
    )


def _merge_skills(rules_skills: list[SkillItem], ai_skills: list[SkillItem]) -> list[SkillItem]:
    by_name: dict[str, SkillItem] = {}
    for s in rules_skills:
        key = s.name.lower()
        by_name[key] = SkillItem(
            name=s.name,
            source="RULES",
            confidence=s.confidence,
        )
    for s in ai_skills:
        key = s.name.lower()
        if key not in by_name:
            by_name[key] = SkillItem(
                name=s.name,
                source="AI",
                confidence=s.confidence,
            )
        else:
            prev = by_name[key]
            c = max(prev.confidence, s.confidence)
            by_name[key] = SkillItem(
                name=prev.name,
                source="HYBRID",
                confidence=min(1.0, c + 0.05),
            )
    return sorted(by_name.values(), key=lambda x: (-x.confidence, x.name.lower()))

