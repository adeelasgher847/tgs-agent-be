from __future__ import annotations

import re
from functools import lru_cache

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
DATE_RANGE_RE = re.compile(
    r"(?i)\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}\s*[–\-]\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}|Present|Current)\b"
)
YEAR_RANGE_RE = re.compile(r"(?i)\b(19|20)\d{2}\s*[–\-]\s*(Present|Current|(19|20)\d{2})\b")
NUMERIC_MONTH_RANGE_RE = re.compile(
    r"(?i)\b(0?[1-9]|1[0-2])\s*/\s*(19|20)\d{2}\s*[–\-]\s*(Present|Current|(0?[1-9]|1[0-2])\s*/\s*(19|20)\d{2})\b"
)
SECTION_HEADERS = re.compile(
    r"(?mi)^\s*(experience|work history|employment|education|skills|"
    r"projects|certifications|summary|objective)\s*:?\s*$",
)

GLOBAL_LOCATION_HINTS = {
    "lahore",
    "karachi",
    "islamabad",
    "rawalpindi",
    "faisalabad",
    "multan",
    "peshawar",
    "quetta",
    "pakistan",
    "india",
    "bangladesh",
    "nepal",
    "sri lanka",
    "uae",
    "dubai",
    "abu dhabi",
    "saudi arabia",
    "riyadh",
    "doha",
    "qatar",
    "usa",
    "united states",
    "uk",
    "united kingdom",
    "canada",
    "australia",
    "germany",
    "france",
    "netherlands",
}


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
    location_tokens = {
        "lahore",
        "karachi",
        "islamabad",
        "pakistan",
        "india",
        "uae",
        "dubai",
        "saudi",
        "riyadh",
        "usa",
        "uk",
        "canada",
        "germany",
        "france",
        "remote",
        "onsite",
        "hybrid",
        "city",
        "country",
    }
    skill_like_tokens = {
        "react",
        "node",
        "nodejs",
        "express",
        "fastapi",
        "django",
        "flask",
        "python",
        "javascript",
        "typescript",
        "sql",
        "mongodb",
        "postgres",
        "aws",
        "docker",
        "kubernetes",
    }

    candidates: list[str] = []
    for line in text.splitlines()[:25]:
        line_stripped = line.strip()
        if not line_stripped or "@" in line_stripped:
            continue
        if EMAIL_RE.search(line_stripped):
            continue
        if len(line_stripped) > 80:
            continue
        lower = line_stripped.lower()
        if lower.startswith(("skills", "tech stack", "technologies", "experience", "education")):
            continue
        words = [w for w in re.split(r"[\s,/-]+", lower) if w]
        if not words:
            continue
        # Reject likely skill lists like "React, Express, Node.js, MySQL"
        skill_hits = sum(1 for w in words if w in skill_like_tokens)
        if skill_hits >= 2:
            continue
        # Prefer lines with explicit location hints
        if any(w in location_tokens for w in words):
            candidates.append(line_stripped)
            continue
        # Fallback for common "City, Country" style with low technical signal
        if "," in line_stripped and any(ch.isalpha() for ch in line_stripped) and skill_hits == 0:
            candidates.append(line_stripped)
    for candidate in candidates:
        if _is_globally_valid_location(candidate):
            return candidate
    return None


def extract_location_from_text(text: str) -> str | None:
    """Public helper for lightweight location extraction from resume text."""
    return _guess_location_line(text or "")


@lru_cache(maxsize=256)
def _is_globally_valid_location(value: str) -> bool:
    """
    Validate that a candidate location resolves as a real geographic place globally.
    Uses geopy/Nominatim with small timeout and cache to avoid repeated lookups.
    """
    query = (value or "").strip()
    if not query or len(query) < 3:
        return False
    try:
        from geopy.geocoders import Nominatim
    except Exception:
        # If geopy is unavailable for any reason, fail closed.
        return False

    lower_query = query.lower()
    if any(hint in lower_query for hint in GLOBAL_LOCATION_HINTS):
        return True

    try:
        geolocator = Nominatim(user_agent="tgs_resume_location_validator")
        result = geolocator.geocode(query, exactly_one=True, addressdetails=True, timeout=2)
    except Exception:
        result = None
    if not result:
        # Try geocoding meaningful tokens/subphrases as fallback for lines like
        # "KBWH PU Old Campus Lahore" where full text may not resolve directly.
        parts = [p.strip() for p in re.split(r"[,/|\-]+", query) if p.strip()]
        words = [w for w in re.split(r"\s+", query) if len(w) >= 4]
        token_candidates = parts + words[-3:]
        for token in token_candidates:
            t = token.strip()
            if not t:
                continue
            if any(hint in t.lower() for hint in GLOBAL_LOCATION_HINTS):
                return True
            try:
                res = geolocator.geocode(t, exactly_one=True, addressdetails=True, timeout=2)
            except Exception:
                continue
            if not res:
                continue
            raw = getattr(res, "raw", {}) or {}
            place_type = str(raw.get("type") or "").lower()
            if place_type in {
                "city",
                "town",
                "village",
                "hamlet",
                "suburb",
                "county",
                "state",
                "province",
                "region",
                "administrative",
                "country",
                "municipality",
            } or bool(raw.get("address")):
                return True
        return False

    raw = getattr(result, "raw", {}) or {}
    place_type = str(raw.get("type") or "").lower()
    allowed_types = {
        "city",
        "town",
        "village",
        "hamlet",
        "suburb",
        "county",
        "state",
        "province",
        "region",
        "administrative",
        "country",
        "municipality",
    }
    # Accept if geocoder classified this as a geographic place-like entity.
    return place_type in allowed_types or bool(raw.get("address"))


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
    in_experience_section = False
    for ln in lines:
        stripped = ln.strip()
        lower = stripped.lower()
        if SECTION_HEADERS.match(stripped):
            if "experience" in lower or "employment" in lower or "work history" in lower:
                in_experience_section = True
                continue
            if in_experience_section and ("education" in lower or "projects" in lower or "skills" in lower):
                break
        if SECTION_HEADERS.match(ln.strip()) and "education" in ln.lower():
            break
        has_date_marker = (
            re.match(
                r"^\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}",
                ln,
                re.I,
            )
            or YEAR_RANGE_RE.search(ln)
            or DATE_RANGE_RE.search(ln)
            or NUMERIC_MONTH_RANGE_RE.search(ln)
        )

        if has_date_marker or (in_experience_section and _looks_like_role_header(stripped)):
            if buf:
                items.append(_lines_to_experience(buf))
                buf = []
            buf.append(ln)
        elif buf and (in_experience_section or stripped):
            buf.append(ln)
    if buf:
        items.append(_lines_to_experience(buf))
    return items[:15]


def _lines_to_experience(lines: list[str]) -> ExperienceItem:
    header = lines[0]
    duration = None
    line_blob = " ".join(lines[:3])
    md = (
        DATE_RANGE_RE.search(line_blob)
        or YEAR_RANGE_RE.search(line_blob)
        or NUMERIC_MONTH_RANGE_RE.search(line_blob)
    )
    if not md:
        md = re.search(r"(20\d{2}|19\d{2})", header, re.I)
    if md:
        duration = md.group(0)
    role_company = DATE_RANGE_RE.sub("", header).strip()
    role_company = YEAR_RANGE_RE.sub("", role_company).strip()
    role_company = NUMERIC_MONTH_RANGE_RE.sub("", role_company).strip()
    role_company = re.sub(r"(?i)\b(achievements/tasks|achievements|tasks)\b", "", role_company).strip(" -|")
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


def _looks_like_role_header(line: str) -> bool:
    if not line:
        return False
    lower = line.lower().strip()
    if len(lower) < 5 or len(lower) > 120:
        return False
    if SECTION_HEADERS.match(lower):
        return False
    if EMAIL_RE.search(lower) or URL_RE.search(lower):
        return False
    role_keywords = (
        "engineer",
        "developer",
        "manager",
        "lead",
        "intern",
        "consultant",
        "architect",
        "analyst",
        "specialist",
        "officer",
    )
    return any(k in lower for k in role_keywords)


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

