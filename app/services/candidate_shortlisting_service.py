from __future__ import annotations

from collections import Counter
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session, load_only

from app.models.job_description import JobDescription
from app.models.resume import ParseStatus, Resume
from app.schemas.resume import ParsedResume, TopCandidateItem, TopCandidatesResponse
from app.services.resume_matching_service import score_candidate
from app.utils.fit_score_labels import explain_fit_score


class CandidateShortlistingService:
    @staticmethod
    def _normalize_weights(weight_map: dict[str, float] | None) -> dict[str, float]:
        if not isinstance(weight_map, dict):
            return {}
        cleaned: dict[str, float] = {}
        for k, v in weight_map.items():
            key = str(k).strip()
            if not key:
                continue
            try:
                cleaned[key] = max(0.0, float(v))
            except (TypeError, ValueError):
                continue
        total = sum(cleaned.values())
        if total <= 0:
            return {}
        return {k: round(v / total, 4) for k, v in cleaned.items()}

    @staticmethod
    def _profile_completeness(parsed: ParsedResume) -> float:
        checks = [
            bool(parsed.profile and parsed.profile.name),
            bool(parsed.profile and parsed.profile.email),
            bool(parsed.skills),
            bool(parsed.experience),
            bool(parsed.education),
            bool(parsed.projects),
            parsed.years_experience_total is not None,
        ]
        return round(sum(1 for c in checks if c) / len(checks), 4)

    @staticmethod
    def _must_have_missing(parsed: ParsedResume, required: list[str]) -> list[str]:
        if not required:
            return []
        blob = " ".join(
            [parsed.raw_text or ""]
            + [s.name for s in parsed.skills]
            + [e.role or "" for e in parsed.experience]
            + [e.company or "" for e in parsed.experience]
        ).lower()
        missing: list[str] = []
        for req in required:
            token = (req or "").strip().lower()
            if token and token not in blob:
                missing.append(req)
        return missing

    def update_shortlist_criteria(
        self,
        db: Session,
        *,
        job: JobDescription,
        user_id: UUID,
        criteria_updates: dict[str, Any],
        skill_weight_matrix: dict[str, float] | None,
    ) -> JobDescription:
        current_criteria = job.matching_criteria if isinstance(job.matching_criteria, dict) else {}
        merged = {**current_criteria, **criteria_updates}
        normalized_weights = self._normalize_weights(skill_weight_matrix)
        if normalized_weights:
            job.skill_weight_matrix = normalized_weights
        job.matching_criteria = merged
        job.version = (job.version or 1) + 1
        job.updated_by = user_id
        db.commit()
        db.refresh(job)
        return job

    def shortlist(
        self,
        db: Session,
        *,
        tenant_id: UUID,
        job: JobDescription,
        batch_id: UUID | None,
        top_k: int,
        min_overall_score: float,
        max_resumes: int,
        match_mode: str | None,
        include_excluded: bool,
    ) -> TopCandidatesResponse:
        q = (
            db.query(Resume)
            .options(
                load_only(
                    Resume.id,
                    Resume.original_filename,
                    Resume.parsed_json,
                    Resume.status,
                    Resume.parse_confidence,
                )
            )
            .filter(Resume.tenant_id == tenant_id)
            .order_by(Resume.created_at.desc())
        )
        if batch_id is not None:
            q = q.filter(Resume.batch_id == batch_id)
        q = q.limit(max_resumes)

        rows = q.all()
        excluded_reasons = Counter()
        included: list[TopCandidateItem] = []

        criteria = job.matching_criteria if isinstance(job.matching_criteria, dict) else {}
        min_parse_conf = float(criteria.get("minimum_parse_confidence", 0.0) or 0.0)
        min_profile_completeness = float(criteria.get("minimum_profile_completeness", 0.0) or 0.0)
        must_have = criteria.get("must_have_criteria") or []

        for res in rows:
            reasons: list[str] = []
            if res.status != ParseStatus.READY:
                reasons.append("resume_not_ready")
            if not res.parsed_json:
                reasons.append("missing_parsed_profile")
            if reasons:
                for r in reasons:
                    excluded_reasons[r] += 1
                continue

            try:
                parsed = ParsedResume.model_validate(res.parsed_json)
            except Exception:
                excluded_reasons["invalid_parsed_profile"] += 1
                continue

            completeness = self._profile_completeness(parsed)
            parse_conf = float(res.parse_confidence or parsed.parse_confidence or 0.0)
            if parse_conf < min_parse_conf:
                reasons.append("parse_confidence_below_threshold")
            if completeness < min_profile_completeness:
                reasons.append("profile_incomplete")
            missing_must_have = self._must_have_missing(parsed, must_have if isinstance(must_have, list) else [])
            if missing_must_have:
                reasons.append("failed_must_have_criteria")
            if reasons:
                for r in reasons:
                    excluded_reasons[r] += 1
                if include_excluded:
                    included.append(
                        TopCandidateItem(
                            resume_id=res.id,
                            filename=res.original_filename,
                            score=0.0,
                            rank=max(1, len(included) + 1),
                            match_percent=0,
                            fit_label="Irrelevant",
                            fit_summary="Excluded by shortlist filters",
                            profile_completeness=completeness,
                            parse_confidence=parse_conf,
                            exclusion_reasons=reasons,
                        )
                    )
                continue

            result = score_candidate(res.id, job, parsed, match_mode=match_mode)
            score = float(result.overall_score)
            if score < min_overall_score:
                excluded_reasons["score_below_threshold"] += 1
                continue

            pct, fit_label, fit_summary = explain_fit_score(score)
            included.append(
                TopCandidateItem(
                    resume_id=res.id,
                    filename=res.original_filename,
                    score=score,
                    rank=0,
                    match_percent=pct,
                    fit_label=fit_label,
                    fit_summary=fit_summary,
                    profile_completeness=completeness,
                    parse_confidence=parse_conf,
                    exclusion_reasons=[],
                )
            )

        included.sort(key=lambda x: (-x.score, str(x.resume_id)))
        shortlisted = included[:top_k]
        for idx, item in enumerate(shortlisted, start=1):
            item.rank = idx

        excluded_count = sum(excluded_reasons.values())
        return TopCandidatesResponse(
            items=shortlisted,
            scanned_count=len(rows),
            shortlisted_count=len(shortlisted),
            excluded_count=excluded_count,
            excluded_reasons_summary=dict(excluded_reasons),
        )


candidate_shortlisting_service = CandidateShortlistingService()
