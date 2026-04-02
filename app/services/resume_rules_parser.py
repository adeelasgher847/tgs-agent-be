from __future__ import annotations

import re

from app.schemas.resume import (
    EducationItem,
    ExperienceItem,
    ParsedResume,
    ParseSource,
    ProfileBlock,
    ProjectItem,
    SkillItem,
)

EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
)
PHONE_RE = re.compile(
    r"(?:(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})",
)
URL_RE = re.compile(
    r"https?://[^\s\)\]>\,\"']+",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
SECTION_HEADERS = re.compile(
    r"(?mi)^\s*(experience|work history|employment|education|skills|"
    r"projects|certifications|summary|objective)\s*:?\s*$",
)


def parse_rules(raw_text: str, parser_version: str) -> ParsedResume:
    profile = ProfileBlock(
        email=_first_match(EMAIL_RE, raw_text),
        phone=_normalize_phone(_first_match(PHONE_RE, raw_text)),
        links=_unique_links(URL_RE.findall(raw_text)),
        location=_guess_location_line(raw_text),
        name=_guess_name(raw_text),
    )
    skills = _skills_from_text(raw_text)
    experience = _experience_blocks(raw_text)
    education = _education_blocks(raw_text)

    confidences: list[float] = []
    if profile.email:
        confidences.append(0.95)
    if profile.phone:
        confidences.append(0.9)
    confidences.extend([0.75 for _ in skills])
    confidences.extend([0.55 for _ in experience])
    confidences.extend([0.55 for _ in education])
    parse_confidence = sum(confidences) / max(len(confidences), 1) if confidences else 0.35

    years_total = _estimate_years_experience(experience, raw_text)

    return ParsedResume(
        profile=profile,
        skills=skills,
        experience=experience,
        education=education,
        certifications=[],
        projects=[],
        languages=[],
        years_experience_total=years_total,
        raw_text=raw_text,
        parse_confidence=min(parse_confidence, 0.92),
        parse_source=ParseSource.RULES,
        parser_version=parser_version,
        model_name=None,
        provider="rules",
    )


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    m = pattern.search(text)
    return m.group(0).strip() if m else None


def _normalize_phone(raw: str | None) -> str | None:
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    if len(digits) < 10:
        return raw.strip()
    return raw.strip()


def _unique_links(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        u = u.rstrip(").,;")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:12]


def _guess_location_line(text: str) -> str | None:
    for line in text.splitlines()[:25]:
        line_stripped = line.strip()
        if not line_stripped or "@" in line_stripped:
            continue
        if EMAIL_RE.search(line_stripped):
            continue
        if len(line_stripped) < 80 and "," in line_stripped and any(
            ch.isalpha() for ch in line_stripped
        ):
            return line_stripped
    return None


def _guess_name(text: str) -> str | None:
    lines = [ln.strip() for ln in text.splitlines()[:8] if ln.strip()]
    if not lines:
        return None
    first = lines[0]
    if EMAIL_RE.search(first) or URL_RE.search(first):
        return None
    if 2 <= len(first) <= 80 and all(
        part.isalpha() or part in "-'." for part in first.replace(" ", "")
    ):
        return first
    return None


def _skills_from_text(text: str) -> list[SkillItem]:
    skill_keywords = {
        "python",
        "java",
        "javascript",
        "typescript",
        "react",
        "node",
        "fastapi",
        "django",
        "flask",
        "sql",
        "postgresql",
        "postgres",
        "mongodb",
        "redis",
        "docker",
        "kubernetes",
        "aws",
        "gcp",
        "azure",
        "git",
        "ci/cd",
        "tensorflow",
        "pytorch",
        "pandas",
        "numpy",
        "kafka",
        "spark",
        "graphql",
        "rest",
        "api",
        "go",
        "rust",
        "c++",
        "c#",
        ".net",
        "angular",
        "vue",
        "spring",
        "rails",
        "ruby",
        "php",
        "swift",
        "kotlin",
        "scala",
        "elasticsearch",
        "snowflake",
        "dba",
        "linux",
        "terraform",
        "ansible",
    }
    lower = text.lower()
    found: list[SkillItem] = []
    for kw in sorted(skill_keywords, key=len, reverse=True):
        if kw in lower:
            found.append(
                SkillItem(name=kw, source="RULES", confidence=0.72 if len(kw) > 3 else 0.6)
            )
    # de-dupe preserve order
    seen: set[str] = set()
    unique: list[SkillItem] = []
    for s in found:
        if s.name not in seen:
            seen.add(s.name)
            unique.append(s)
    return unique[:40]


def _experience_blocks(text: str) -> list[ExperienceItem]:
    lines = text.splitlines()
    items: list[ExperienceItem] = []
    buf: list[str] = []
    for ln in lines:
        if SECTION_HEADERS.match(ln.strip()) and "education" in ln.lower():
            break
        if re.match(
            r"^\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}",
            ln,
            re.I,
        ) or re.search(r"\d{4}\s*[–\-]\s*(Present|\d{4})", ln, re.I):
            if buf:
                items.append(_lines_to_experience(buf))
                buf = []
            buf.append(ln)
        elif buf:
            buf.append(ln)
    if buf:
        items.append(_lines_to_experience(buf))
    return items[:15]


def _lines_to_experience(lines: list[str]) -> ExperienceItem:
    header = lines[0]
    duration = None
    md = re.search(r"(\d{4}\s*[–\-]\s*(?:Present|\d{4})|20\d{2}|19\d{2})", header, re.I)
    if md:
        duration = md.group(1)
    rest = " ".join(lines[1:6])
    role_company = re.sub(
        r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}\s*[–\-]\s*(?:Present|\d{4})\s*",
        "",
        header,
        flags=re.I,
    ).strip()
    role = role_company
    company = None
    if "|" in role_company:
        parts = [p.strip() for p in role_company.split("|")]
        role, company = parts[0], parts[1] if len(parts) > 1 else None
    elif " at " in role_company.lower():
        idx = role_company.lower().find(" at ")
        role, company = role_company[:idx].strip(), role_company[idx + 4 :].strip()
    responsibilities = [ln.strip("•- \t") for ln in lines[1:] if ln.strip()][:8]
    return ExperienceItem(
        role=role or None,
        company=company,
        duration=duration,
        responsibilities=responsibilities,
    )


def _education_blocks(text: str) -> list[EducationItem]:
    section = _extract_section(text, "education")
    if not section:
        return []
    items: list[EducationItem] = []
    for para in re.split(r"\n{2,}", section):
        yr = None
        ym = YEAR_RE.search(para)
        if ym:
            try:
                yr = int(ym.group(0))
            except ValueError:
                yr = ym.group(0)
        lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
        degree = lines[0] if lines else None
        institution = lines[1] if len(lines) > 1 else None
        if degree:
            items.append(
                EducationItem(degree=degree, institution=institution, year=yr),
            )
    return items[:10]


def _extract_section(text: str, name: str) -> str | None:
    pattern = re.compile(rf"(?is)^\s*{name}\s*:?\s*$(.*?)(?=^\s*\w+\s*:?\s*$|\Z)", re.MULTILINE)
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _estimate_years_experience(
    experience: list[ExperienceItem],
    raw_text: str,
) -> float | None:
    year_vals = set()
    for m in YEAR_RE.finditer(raw_text):
        try:
            year_vals.add(int(m.group(0)))
        except ValueError:
            continue
    if len(year_vals) >= 2:
        return float(max(year_vals) - min(year_vals))
    if experience:
        return float(len(experience) * 1.5)
    return None

