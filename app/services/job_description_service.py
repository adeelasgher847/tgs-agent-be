from enum import Enum
from typing import Any, Optional
import uuid
import re
import zipfile
from io import BytesIO
import json
from decimal import Decimal

from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from pypdf import PdfReader

from app.core.config import settings
from app.core.logger import logger
from app.models.job_description import JobDescription
from app.models.resume import Resume
from app.models.user import user_tenant_association
from app.schemas.job_description import JobDescriptionCreateManual, JobDescriptionUpdate
from app.services.openai_service import openai_service
from app.services.gemini_service import gemini_service


class JobDescriptionService:
    _DEFAULT_SCORING_DIMENSIONS = [
        {"name": "skills_match", "weight": 0.40, "description": "Core and secondary skills alignment with role requirements."},
        {"name": "experience", "weight": 0.25, "description": "Relevant years and depth of hands-on experience."},
        {"name": "education_certifications", "weight": 0.15, "description": "Education and certifications meeting job baseline."},
        {"name": "location_employment_fit", "weight": 0.10, "description": "Location and employment type compatibility."},
        {"name": "keyword_context", "weight": 0.10, "description": "Domain keyword and responsibility context relevance."},
    ]

    @staticmethod
    def _threshold_percent_to_fraction(value: Any) -> float:
        try:
            v = float(value if value is not None else 50.0)
        except (TypeError, ValueError):
            v = 50.0
        v = max(1.0, min(100.0, v))
        return round(v / 100.0, 4)

    @staticmethod
    def _threshold_fraction_to_percent(value: Any) -> float:
        try:
            v = float(value if value is not None else 0.5)
        except (TypeError, ValueError):
            v = 0.5
        v = max(0.0, min(1.0, v))
        return round(v * 100.0, 2)

    def create_manual(
        self,
        db: Session,
        payload: JobDescriptionCreateManual,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> JobDescription:
        data = payload.model_dump()
        data["pass_match_threshold"] = self._threshold_percent_to_fraction(
            data.get("pass_match_threshold", 50.0)
        )

        # Persist manual form as raw text fallback when explicit raw text is not sent.
        if not data.get("raw_text"):
            data["raw_text"] = self._build_raw_text_from_manual_fields(data)

        db_jd = JobDescription(
            **data,
            tenant_id=tenant_id,
            extracted_skills=[],
            keywords=[],
            skill_weight_matrix={},
            matching_criteria={},
            processing_status="PENDING",
            version=1,
            created_by=user_id,
            updated_by=user_id,
        )
        db.add(db_jd)
        db.commit()
        db.refresh(db_jd)
        return db_jd

    def create_upload(
        self,
        db: Session,
        filename: str,
        file_content: bytes,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> JobDescription:
        raw_text = self._extract_text_from_upload(filename=filename, file_content=file_content)
        if not raw_text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract readable text from uploaded file",
            )

        inferred_job_title = self._infer_job_title(raw_text)

        db_jd = JobDescription(
            tenant_id=tenant_id,
            job_title=inferred_job_title,
            required_skills=[],
            years_experience_min=None,
            years_experience_max=None,
            education_requirements=None,
            location=None,
            salary_min=None,
            salary_max=None,
            currency=None,
            employment_type=None,
            key_responsibilities=[],
            required_certifications=[],
            raw_text=raw_text,
            extracted_skills=[],
            keywords=[],
            skill_weight_matrix={},
            matching_criteria={},
            processing_status="PENDING",
            version=1,
            created_by=user_id,
            updated_by=user_id,
        )
        db.add(db_jd)
        db.commit()
        db.refresh(db_jd)
        return db_jd

    def get_by_id(self, db: Session, job_description_id: uuid.UUID, tenant_id: uuid.UUID) -> JobDescription:
        jd = (
            db.query(JobDescription)
            .filter(
                JobDescription.id == job_description_id,
                JobDescription.tenant_id == tenant_id,
            )
            .first()
        )
        if not jd:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job description not found")
        return jd

    def get_by_id_in_tenants(
        self,
        db: Session,
        job_description_id: uuid.UUID,
        tenant_ids: list[uuid.UUID],
    ) -> JobDescription:
        if not tenant_ids:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job description not found")
        jd = (
            db.query(JobDescription)
            .filter(
                JobDescription.id == job_description_id,
                JobDescription.tenant_id.in_(tenant_ids),
            )
            .first()
        )
        if not jd:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job description not found")
        return jd

    @staticmethod
    def tenant_ids_for_user(db: Session, user_id: uuid.UUID) -> list[uuid.UUID]:
        rows = (
            db.query(user_tenant_association.c.tenant_id)
            .filter(user_tenant_association.c.user_id == user_id)
            .distinct()
            .all()
        )
        return [r[0] for r in rows]

    def list_by_tenant(self, db: Session, tenant_id: uuid.UUID) -> list[JobDescription]:
        return (
            db.query(JobDescription)
            .filter(JobDescription.tenant_id == tenant_id)
            .order_by(JobDescription.created_at.desc())
            .all()
        )

    def list_by_tenant_ids(self, db: Session, tenant_ids: list[uuid.UUID]) -> list[JobDescription]:
        if not tenant_ids:
            return []
        return (
            db.query(JobDescription)
            .filter(JobDescription.tenant_id.in_(tenant_ids))
            .order_by(JobDescription.created_at.desc())
            .all()
        )

    @staticmethod
    def _ensure_str_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            out: list[str] = []
            for x in value:
                if x is None:
                    continue
                s = str(x).strip()
                if s:
                    out.append(s)
            return out
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _normalize_extracted_skills_json(value: Any) -> list[dict[str, Any]]:
        if not value or not isinstance(value, list):
            return []
        out: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append({"skill": s[:120], "confidence": 0.5})
                continue
            if not isinstance(item, dict):
                continue
            skill = (item.get("skill") or item.get("name") or "").strip()
            if not skill:
                continue
            conf = item.get("confidence")
            if not isinstance(conf, (int, float)):
                conf = 0.5
            conf = max(0.0, min(1.0, float(conf)))
            out.append({"skill": skill[:120], "confidence": round(conf, 2)})
        return out

    def normalize_for_read_response(self, jd: JobDescription) -> None:
        """In-memory coercion so JobDescriptionOut validates (null JSON, enums, etc.). No LLM, no HTTP to AI providers."""
        jd.required_skills = self._ensure_str_list(jd.required_skills)
        jd.key_responsibilities = self._ensure_str_list(jd.key_responsibilities)
        jd.required_certifications = self._ensure_str_list(jd.required_certifications)
        jd.keywords = self._ensure_str_list(jd.keywords)
        jd.extracted_skills = self._normalize_extracted_skills_json(jd.extracted_skills)
        jd.skill_weight_matrix = jd.skill_weight_matrix if isinstance(jd.skill_weight_matrix, dict) else {}
        jd.matching_criteria = jd.matching_criteria if isinstance(jd.matching_criteria, dict) else {}

        if jd.currency is not None:
            c = str(jd.currency).strip().upper()
            jd.currency = c if len(c) >= 3 else None

        if jd.employment_type is not None:
            jd.employment_type = self._normalize_employment_type(jd.employment_type)

        title = (jd.job_title or "").strip()
        jd.job_title = (title[:255] if title else "Untitled")

        ps = str(jd.processing_status or "PENDING").strip().upper()
        if ps not in ("PENDING", "PROCESSING", "READY", "FAILED"):
            ps = "PENDING"
        jd.processing_status = ps

        jd.pass_match_threshold = self._threshold_fraction_to_percent(jd.pass_match_threshold)

    def update(
        self,
        db: Session,
        job_description_id: uuid.UUID,
        payload: JobDescriptionUpdate,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> JobDescription:
        jd = self.get_by_id(db, job_description_id, tenant_id)
        updates = payload.model_dump(exclude_unset=True)
        if "pass_match_threshold" in updates:
            updates["pass_match_threshold"] = self._threshold_percent_to_fraction(
                updates.get("pass_match_threshold")
            )
        for key, value in list(updates.items()):
            if isinstance(value, Enum):
                updates[key] = value.value

        for key, value in updates.items():
            setattr(jd, key, value)

        jd.version = (jd.version or 1) + 1
        jd.updated_by = user_id

        db.commit()
        db.refresh(jd)
        return jd

    def delete(
        self,
        db: Session,
        job_description_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> None:
        jd = self.get_by_id(db, job_description_id, tenant_id)
        # Detach dependent resumes before deleting JD to satisfy FK constraints.
        (
            db.query(Resume)
            .filter(
                Resume.job_description_id == job_description_id,
            )
            .update({Resume.job_description_id: None}, synchronize_session=False)
        )
        db.delete(jd)
        db.commit()

    def process_upload(
        self,
        db: Session,
        job_description_id: uuid.UUID,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> JobDescription:
        jd = self.get_by_id(db, job_description_id, tenant_id)
        jd.processing_status = "PROCESSING"
        jd.updated_by = user_id
        db.commit()
        db.refresh(jd)

        try:
            source_text = (jd.raw_text or "").strip()
            source_text_lower = source_text.lower()
            # Deterministic extraction FIRST to reduce LLM dependency and latency.
            deterministic_skills = self._extract_skills(source_text_lower)
            deterministic_keywords = self._extract_keywords(source_text_lower, deterministic_skills)
            deterministic_resp = self._extract_responsibilities_from_text(source_text)
            deterministic_sal_min, deterministic_sal_max, deterministic_currency = self._extract_salary_from_text(
                source_text
            )
            deterministic_exp_min, deterministic_exp_max = self._extract_experience_from_text(source_text)

            if jd.years_experience_min is None and deterministic_exp_min is not None:
                jd.years_experience_min = deterministic_exp_min
            if jd.years_experience_max is None and deterministic_exp_max is not None:
                jd.years_experience_max = deterministic_exp_max
            if jd.salary_min is None and deterministic_sal_min is not None:
                jd.salary_min = deterministic_sal_min
            if jd.salary_max is None and deterministic_sal_max is not None:
                jd.salary_max = deterministic_sal_max
            if not jd.currency and deterministic_currency:
                jd.currency = deterministic_currency
            if not jd.key_responsibilities and deterministic_resp:
                jd.key_responsibilities = deterministic_resp

            llm_data: dict[str, Any] = {}
            needs_llm = (
                not (jd.job_title or "").strip()
                or (jd.years_experience_min is None and jd.years_experience_max is None)
                or not jd.education_requirements
                or not jd.location
                or not jd.employment_type
                or (not jd.required_skills and not deterministic_skills)
                or not jd.key_responsibilities
                or (jd.salary_min is None and jd.salary_max is None)
            )
            if needs_llm:
                llm_data = self._extract_with_llm(source_text=source_text, jd=jd)
            llm_enriched = bool(llm_data)

            # Fill only remaining gaps with LLM.
            if not (jd.job_title or "").strip() and (llm_data.get("job_title") or "").strip():
                jd.job_title = (llm_data.get("job_title") or "").strip()[:255]
            if jd.years_experience_min is None and llm_data.get("years_experience_min") is not None:
                jd.years_experience_min = llm_data.get("years_experience_min")
            if jd.years_experience_max is None and llm_data.get("years_experience_max") is not None:
                jd.years_experience_max = llm_data.get("years_experience_max")
            if not jd.education_requirements and llm_data.get("education_requirements"):
                jd.education_requirements = llm_data.get("education_requirements")
            if not jd.location and llm_data.get("location"):
                jd.location = llm_data.get("location")
            if not jd.employment_type and llm_data.get("employment_type"):
                jd.employment_type = self._normalize_employment_type(llm_data.get("employment_type"))
            if not jd.required_certifications and llm_data.get("required_certifications"):
                jd.required_certifications = llm_data.get("required_certifications")
            if not jd.key_responsibilities and llm_data.get("key_responsibilities"):
                jd.key_responsibilities = self._dedupe_preserve(llm_data.get("key_responsibilities"))
            if jd.salary_min is None and llm_data.get("salary_min") is not None:
                jd.salary_min = self._safe_decimal(llm_data.get("salary_min"))
            if jd.salary_max is None and llm_data.get("salary_max") is not None:
                jd.salary_max = self._safe_decimal(llm_data.get("salary_max"))
            if not jd.currency and llm_data.get("currency"):
                jd.currency = self._normalize_currency(llm_data.get("currency"))

            llm_skill_objects = llm_data.get("skills") or []
            llm_skill_names = [
                (s.get("name") or "").strip().lower()
                for s in llm_skill_objects
                if (s.get("name") or "").strip()
            ]
            skills = jd.required_skills or deterministic_skills or llm_skill_names
            skills = self._dedupe_preserve(skills)

            keywords = (
                llm_data.get("keywords")
                or deterministic_keywords
                or self._extract_keywords(source_text_lower, skills)
            )
            keywords = self._dedupe_preserve(keywords)

            weight_matrix = self._build_skill_weight_matrix(
                skills=skills,
                llm_skill_objects=llm_skill_objects,
                source_text=source_text_lower,
            )

            extracted_skills = self._build_extracted_skills(
                skills=skills,
                llm_skill_objects=llm_skill_objects,
                source_text=source_text_lower,
            )

            scoring_dimensions = self._normalize_scoring_dimensions(
                llm_data.get("scoring_dimensions")
            )
            must_have_criteria = llm_data.get("must_have_criteria") or self._build_must_have_criteria(jd, skills)
            overall_confidence = self._compute_overall_confidence(extracted_skills, llm_data)
            matching_criteria = {
                "required_skills": skills,
                "minimum_years_experience": jd.years_experience_min,
                "maximum_years_experience": jd.years_experience_max,
                "education_requirements": jd.education_requirements,
                "location": jd.location,
                "employment_type": jd.employment_type,
                "keywords": keywords,
                "must_have_criteria": must_have_criteria,
                "scoring_dimensions": scoring_dimensions,
                "explainability": {
                    "weights_reasoning": "Skill weights are based on required/preferred signal, term emphasis in JD text, and semantic extraction confidence.",
                    "match_formula": "candidate_score = sum(dimension_score * weight) for each scoring dimension.",
                    "llm_enriched": llm_enriched,
                },
            }

            jd.required_skills = skills
            jd.extracted_skills = extracted_skills
            jd.keywords = keywords
            jd.skill_weight_matrix = weight_matrix
            jd.matching_criteria = matching_criteria
            jd.processing_status = "READY"
            jd.version = (jd.version or 1) + 1
            jd.updated_by = user_id
            db.commit()
            db.refresh(jd)
            self.normalize_for_read_response(jd)
            return jd
        except Exception as e:
            jd.processing_status = "FAILED"
            jd.version = (jd.version or 1) + 1
            jd.updated_by = user_id
            db.commit()
            db.refresh(jd)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    def process(
        self,
        db: Session,
        job_description_id: uuid.UUID,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> JobDescription:
        jd = self.get_by_id(db, job_description_id, tenant_id)
        jd.processing_status = "PROCESSING"
        jd.updated_by = user_id
        db.commit()
        db.refresh(jd)

        try:
            source_text = (jd.raw_text or "").strip()
            source_text_lower = source_text.lower()
            llm_data = self._extract_with_llm(source_text=source_text, jd=jd)
            llm_enriched = bool(llm_data)

            # Prefer explicit recruiter-provided values first, then LLM inference, then deterministic fallback.
            if not (jd.job_title or "").strip() and (llm_data.get("job_title") or "").strip():
                jd.job_title = (llm_data.get("job_title") or "").strip()[:255]
            if jd.years_experience_min is None and llm_data.get("years_experience_min") is not None:
                jd.years_experience_min = llm_data.get("years_experience_min")
            if jd.years_experience_max is None and llm_data.get("years_experience_max") is not None:
                jd.years_experience_max = llm_data.get("years_experience_max")
            if not jd.education_requirements and llm_data.get("education_requirements"):
                jd.education_requirements = llm_data.get("education_requirements")
            if not jd.location and llm_data.get("location"):
                jd.location = llm_data.get("location")
            if not jd.employment_type and llm_data.get("employment_type"):
                jd.employment_type = self._normalize_employment_type(llm_data.get("employment_type"))
            if not jd.required_certifications and llm_data.get("required_certifications"):
                jd.required_certifications = llm_data.get("required_certifications")
            if not jd.key_responsibilities and llm_data.get("key_responsibilities"):
                jd.key_responsibilities = self._dedupe_preserve(llm_data.get("key_responsibilities"))
            if jd.salary_min is None and llm_data.get("salary_min") is not None:
                jd.salary_min = self._safe_decimal(llm_data.get("salary_min"))
            if jd.salary_max is None and llm_data.get("salary_max") is not None:
                jd.salary_max = self._safe_decimal(llm_data.get("salary_max"))
            if not jd.currency and llm_data.get("currency"):
                jd.currency = self._normalize_currency(llm_data.get("currency"))

            # Fallback salary extraction from raw text when LLM did not return salary.
            if jd.salary_min is None and jd.salary_max is None:
                s_min, s_max, s_currency = self._extract_salary_from_text(source_text)
                jd.salary_min = s_min
                jd.salary_max = s_max
                if not jd.currency:
                    jd.currency = s_currency

            # Fallback responsibility extraction from raw text when LLM did not return.
            if not jd.key_responsibilities:
                jd.key_responsibilities = self._extract_responsibilities_from_text(source_text)

            llm_skill_objects = llm_data.get("skills") or []
            llm_skill_names = [
                (s.get("name") or "").strip().lower()
                for s in llm_skill_objects
                if (s.get("name") or "").strip()
            ]
            skills = jd.required_skills or llm_skill_names or self._extract_skills(source_text_lower)
            skills = self._dedupe_preserve(skills)

            keywords = (
                llm_data.get("keywords")
                or self._extract_keywords(source_text_lower, skills)
            )
            keywords = self._dedupe_preserve(keywords)

            weight_matrix = self._build_skill_weight_matrix(
                skills=skills,
                llm_skill_objects=llm_skill_objects,
                source_text=source_text_lower,
            )

            extracted_skills = self._build_extracted_skills(
                skills=skills,
                llm_skill_objects=llm_skill_objects,
                source_text=source_text_lower,
            )

            scoring_dimensions = self._normalize_scoring_dimensions(
                llm_data.get("scoring_dimensions")
            )
            must_have_criteria = llm_data.get("must_have_criteria") or self._build_must_have_criteria(jd, skills)
            overall_confidence = self._compute_overall_confidence(extracted_skills, llm_data)
            matching_criteria = {
                "required_skills": skills,
                "minimum_years_experience": jd.years_experience_min,
                "maximum_years_experience": jd.years_experience_max,
                "education_requirements": jd.education_requirements,
                "location": jd.location,
                "employment_type": jd.employment_type,
                "keywords": keywords,
                "must_have_criteria": must_have_criteria,
                "scoring_dimensions": scoring_dimensions,
                "explainability": {
                    "weights_reasoning": "Skill weights are based on required/preferred signal, term emphasis in JD text, and semantic extraction confidence.",
                    "match_formula": "candidate_score = sum(dimension_score * weight) for each scoring dimension.",
                    "llm_enriched": llm_enriched,
                },
            }

            jd.required_skills = skills
            jd.extracted_skills = extracted_skills
            jd.keywords = keywords
            jd.skill_weight_matrix = weight_matrix
            jd.matching_criteria = matching_criteria
            jd.processing_status = "READY"
            jd.version = (jd.version or 1) + 1
            jd.updated_by = user_id
            db.commit()
            db.refresh(jd)
            self.normalize_for_read_response(jd)
            return jd
        except Exception as e:
            jd.processing_status = "FAILED"
            jd.version = (jd.version or 1) + 1
            jd.updated_by = user_id
            db.commit()
            db.refresh(jd)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    def get_status(
        self,
        db: Session,
        job_description_id: uuid.UUID,
        tenant_id: uuid.UUID,
    ) -> str:
        jd = self.get_by_id(db, job_description_id, tenant_id)
        return jd.processing_status

    def get_status_in_tenants(
        self,
        db: Session,
        job_description_id: uuid.UUID,
        tenant_ids: list[uuid.UUID],
    ) -> str:
        jd = self.get_by_id_in_tenants(db, job_description_id, tenant_ids)
        return jd.processing_status

    @staticmethod
    def _dedupe_preserve(items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in items or []:
            norm = (item or "").strip()
            if not norm:
                continue
            key = norm.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(norm)
        return out

    def _extract_with_llm(self, source_text: str, jd: JobDescription) -> dict[str, Any]:
        if not source_text:
            return {}

        system_prompt = (
            "You are an expert recruiting analyst. Extract structured hiring requirements from a job description. "
            "Return only valid JSON."
        )
        user_prompt = f"""
Analyze the following job description and return STRICT JSON with this exact structure:
{{
  "job_title": string|null,
  "skills": [
    {{
      "name": string,
      "type": "required"|"preferred",
      "weight": number,
      "confidence": number,
      "rationale": string
    }}
  ],
  "keywords": [string],
  "years_experience_min": number|null,
  "years_experience_max": number|null,
  "education_requirements": string|null,
  "location": string|null,
  "employment_type": string|null,
  "required_certifications": [string],
  "key_responsibilities": [string],
  "salary_min": number|null,
  "salary_max": number|null,
  "currency": string|null,
  "must_have_criteria": [string],
  "scoring_dimensions": [
    {{"name": string, "weight": number, "description": string}}
  ],
  "overall_confidence": number
}}

Rules:
- Use confidence and weights in range 0.0 to 1.0.
- Output concise, clean values.
- Do not include markdown fences.

JOB DESCRIPTION:
{source_text[:12000]}
""".strip()

        # OpenAI first, Gemini fallback (if configured).
        providers = []
        if settings.OPENAI_API_KEY:
            providers.append(("openai", "gpt-4o-mini"))
        if settings.GEMINI_API_KEY:
            providers.append(("gemini", "gemini-1.5-flash"))

        if not providers:
            logger.warning(
                "Job description LLM skipped: set OPENAI_API_KEY and/or GEMINI_API_KEY in the environment."
            )

        for provider, model_name in providers:
            try:
                if provider == "openai":
                    resp = openai_service.chat_completion(
                        messages=[{"role": "user", "content": user_prompt}],
                        system_prompt=system_prompt,
                        model_name=model_name,
                        temperature=0.1,
                        max_tokens=1200,
                    )
                else:
                    resp = gemini_service.chat_completion(
                        messages=[{"role": "user", "content": user_prompt}],
                        system_prompt=system_prompt,
                        model_name=model_name,
                        temperature=0.1,
                        max_tokens=1200,
                    )
                parsed = self._parse_llm_json(resp.get("content", ""))
                if parsed:
                    return parsed
            except Exception as exc:
                logger.warning("JD LLM extraction failed via %s: %s", provider, exc)

        return {}

    @staticmethod
    def _parse_llm_json(content: str) -> dict[str, Any]:
        text = (content or "").strip()
        if not text:
            return {}
        # Handle optional markdown fenced JSON.
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(text[start : end + 1])
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    return {}
        return {}

    @staticmethod
    def _build_raw_text_from_manual_fields(data: dict[str, Any]) -> str:
        parts = [
            f"Job Title: {data.get('job_title') or ''}",
            f"Required Skills: {', '.join(data.get('required_skills') or [])}",
            (
                "Experience: "
                f"{data.get('years_experience_min') or ''}"
                f"{('-' + str(data.get('years_experience_max'))) if data.get('years_experience_max') is not None else ''} "
                "years"
            ),
            f"Education: {data.get('education_requirements') or ''}",
            f"Location: {data.get('location') or ''}",
            (
                "Salary: "
                f"{data.get('salary_min') or ''} - {data.get('salary_max') or ''} "
                f"{data.get('currency') or ''}"
            ),
            f"Employment Type: {data.get('employment_type') or ''}",
            f"Responsibilities: {', '.join(data.get('key_responsibilities') or [])}",
            f"Certifications: {', '.join(data.get('required_certifications') or [])}",
        ]
        return "\n".join(parts).strip()

    @staticmethod
    def _extract_text_from_upload(filename: str, file_content: bytes) -> str:
        name = (filename or "").lower()

        if name.endswith(".txt"):
            return file_content.decode("utf-8", errors="ignore")

        if name.endswith(".pdf"):
            reader = PdfReader(BytesIO(file_content))
            pages = [(p.extract_text() or "") for p in reader.pages]
            return "\n".join(pages).strip()

        if name.endswith(".docx"):
            with zipfile.ZipFile(BytesIO(file_content)) as docx_zip:
                xml_content = docx_zip.read("word/document.xml").decode("utf-8", errors="ignore")
            # Very lightweight stripping; enough for MVP extraction.
            text = re.sub(r"</w:p>", "\n", xml_content)
            text = re.sub(r"<[^>]+>", "", text)
            return re.sub(r"\n{2,}", "\n", text).strip()

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Allowed: pdf, docx, txt",
        )

    @staticmethod
    def _infer_job_title(raw_text: str) -> str:
        lines = (raw_text or "").strip().splitlines()
        for line in lines:
            candidate = line.strip()
            if candidate:
                return candidate[:255]
        return "Uploaded Job Description"

    @staticmethod
    def _extract_skills(text: str) -> list[str]:
        skill_bank = [
            "python",
            "fastapi",
            "django",
            "flask",
            "sql",
            "postgresql",
            "mysql",
            "docker",
            "kubernetes",
            "aws",
            "azure",
            "gcp",
            "javascript",
            "typescript",
            "react",
            "node.js",
            "redis",
        ]
        return [skill for skill in skill_bank if skill in text]

    @staticmethod
    def _extract_keywords(text: str, skills: list[str]) -> list[str]:
        tokens = [token.strip(".,:;()[]{}").lower() for token in text.split()]
        filtered = [t for t in tokens if len(t) > 2 and t.isascii()]
        top = []
        seen = set()
        for token in filtered:
            if token in seen:
                continue
            seen.add(token)
            top.append(token)
            if len(top) >= 20:
                break
        return list(dict.fromkeys([*skills, *top]))

    def _build_skill_weight_matrix(
        self,
        skills: list[str],
        llm_skill_objects: Optional[list[dict[str, Any]]] = None,
        source_text: str = "",
    ) -> dict[str, float]:
        if not skills:
            return {}
        llm_skill_objects = llm_skill_objects or []
        llm_map = {}
        for item in llm_skill_objects:
            name = (item.get("name") or "").strip()
            if not name:
                continue
            w = item.get("weight")
            if isinstance(w, (int, float)):
                llm_map[name.lower()] = float(w)

        # Blend: 70% LLM weight (if present) + 30% textual emphasis prior.
        raw: dict[str, float] = {}
        for idx, skill in enumerate(skills):
            key = skill.lower()
            llm_weight = llm_map.get(key)
            emphasis = 1.2 if f"must have {key}" in source_text or f"required: {key}" in source_text else 1.0
            rank_prior = (len(skills) - idx) / max(1, len(skills))
            if llm_weight is None:
                score = rank_prior * emphasis
            else:
                score = (0.7 * max(0.0, min(1.0, llm_weight))) + (0.3 * rank_prior * emphasis)
            raw[skill] = max(0.01, score)

        total = sum(raw.values()) or 1.0
        return {k: round(v / total, 4) for k, v in raw.items()}

    @staticmethod
    def _build_extracted_skills(
        skills: list[str],
        llm_skill_objects: list[dict[str, Any]],
        source_text: str,
    ) -> list[dict[str, Any]]:
        llm_conf: dict[str, float] = {}
        for item in llm_skill_objects:
            name = (item.get("name") or "").strip().lower()
            conf = item.get("confidence")
            if name and isinstance(conf, (int, float)):
                llm_conf[name] = max(0.0, min(1.0, float(conf)))

        out = []
        for skill in skills:
            key = skill.lower()
            if key in llm_conf:
                confidence = llm_conf[key]
            else:
                # Heuristic confidence fallback based on exact occurrences.
                occurrences = source_text.count(key)
                confidence = min(0.95, 0.55 + (occurrences * 0.08))
            out.append({"skill": skill, "confidence": round(confidence, 2)})
        return out

    def _build_must_have_criteria(self, jd: JobDescription, skills: list[str]) -> list[str]:
        criteria = []
        if jd.years_experience_min is not None:
            criteria.append(f"Minimum {jd.years_experience_min}+ years relevant experience.")
        for skill in skills[:5]:
            criteria.append(f"Demonstrated practical experience in {skill}.")
        for cert in (jd.required_certifications or [])[:3]:
            criteria.append(f"Must hold certification: {cert}.")
        return criteria

    def _normalize_scoring_dimensions(self, dims: Any) -> list[dict[str, Any]]:
        if not isinstance(dims, list) or not dims:
            return self._DEFAULT_SCORING_DIMENSIONS

        normalized = []
        for item in dims:
            if not isinstance(item, dict):
                continue
            name = (item.get("name") or "").strip().lower().replace(" ", "_")
            desc = (item.get("description") or "").strip() or "Scoring factor"
            weight = item.get("weight")
            if not name or not isinstance(weight, (int, float)):
                continue
            normalized.append({"name": name, "weight": max(0.0, float(weight)), "description": desc})

        if not normalized:
            return self._DEFAULT_SCORING_DIMENSIONS

        total = sum(d["weight"] for d in normalized)
        if total <= 0:
            return self._DEFAULT_SCORING_DIMENSIONS
        for d in normalized:
            d["weight"] = round(d["weight"] / total, 4)
        return normalized

    @staticmethod
    def _safe_decimal(value: Any) -> Optional[Decimal]:
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except Exception:
            return None

    @staticmethod
    def _normalize_currency(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().upper()
        if not text:
            return None
        if text in {"$", "USD"}:
            return "USD"
        if text in {"PKR", "RS", "RUPEES"}:
            return "PKR"
        if text in {"INR", "₹"}:
            return "INR"
        if text in {"EUR", "€"}:
            return "EUR"
        if text in {"GBP", "£"}:
            return "GBP"
        return text[:12]

    def _extract_salary_from_text(self, source_text: str) -> tuple[Optional[Decimal], Optional[Decimal], Optional[str]]:
        text = source_text or ""
        if not text:
            return None, None, None

        # Try to detect explicit currency tokens first.
        upper = text.upper()
        currency = None
        if "PKR" in upper:
            currency = "PKR"
        elif "USD" in upper or "$" in text:
            currency = "USD"
        elif "INR" in upper or "₹" in text:
            currency = "INR"
        elif "EUR" in upper or "€" in text:
            currency = "EUR"
        elif "GBP" in upper or "£" in text:
            currency = "GBP"

        # Range patterns like "60,000 - 90,000" or "$80k to $120k"
        range_match = re.search(
            r"(?i)(?:salary|compensation|ctc)?\s*[:\-]?\s*[$€£₹]?\s*([0-9][0-9,]*(?:\.\d+)?)\s*([kKmM]?)\s*(?:to|-)\s*[$€£₹]?\s*([0-9][0-9,]*(?:\.\d+)?)\s*([kKmM]?)",
            text,
        )
        if range_match:
            min_val = self._scaled_number(range_match.group(1), range_match.group(2))
            max_val = self._scaled_number(range_match.group(3), range_match.group(4))
            return min_val, max_val, currency

        # Single-value patterns like "Salary: 120000"
        single_match = re.search(
            r"(?i)(?:salary|compensation|ctc)\s*[:\-]?\s*[$€£₹]?\s*([0-9][0-9,]*(?:\.\d+)?)\s*([kKmM]?)",
            text,
        )
        if single_match:
            value = self._scaled_number(single_match.group(1), single_match.group(2))
            return value, value, currency

        return None, None, currency

    @staticmethod
    def _extract_experience_from_text(source_text: str) -> tuple[Optional[int], Optional[int]]:
        text = source_text or ""
        if not text.strip():
            return None, None

        range_match = re.search(
            r"(?i)\b(\d{1,2})\s*(?:\+?\s*)?(?:-|to)\s*(\d{1,2})\s*(?:\+?\s*)?(?:years?|yrs?)\b",
            text,
        )
        if range_match:
            min_years = int(range_match.group(1))
            max_years = int(range_match.group(2))
            if min_years <= max_years:
                return min_years, max_years
            return max_years, min_years

        min_only_match = re.search(
            r"(?i)\b(?:minimum|min\.?)\s*(\d{1,2})\s*(?:\+?\s*)?(?:years?|yrs?)\b",
            text,
        )
        if min_only_match:
            return int(min_only_match.group(1)), None

        plus_match = re.search(
            r"(?i)\b(\d{1,2})\s*\+\s*(?:years?|yrs?)\b",
            text,
        )
        if plus_match:
            return int(plus_match.group(1)), None

        return None, None

    @staticmethod
    def _extract_responsibilities_from_text(source_text: str) -> list[str]:
        text = source_text or ""
        if not text.strip():
            return []

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return []

        # 1) Try section-based extraction under common headers.
        header_patterns = (
            "responsibilities",
            "key responsibilities",
            "role responsibilities",
            "what you'll do",
            "what you will do",
            "job responsibilities",
        )
        out: list[str] = []
        capture = False
        for line in lines:
            lower = line.lower().rstrip(":")
            if any(h == lower for h in header_patterns):
                capture = True
                continue

            # Stop when another major section starts.
            if capture and any(
                lower.startswith(prefix)
                for prefix in (
                    "requirements",
                    "required skills",
                    "qualifications",
                    "education",
                    "experience",
                    "salary",
                    "location",
                    "benefits",
                )
            ):
                break

            if capture:
                cleaned = re.sub(r"^[\-\*\u2022\d\.\)\(]+\s*", "", line).strip()
                if cleaned:
                    out.append(cleaned)
                if len(out) >= 12:
                    break

        if out:
            return out

        # 2) Fallback: collect bullet-style lines from the whole document.
        bullets = []
        for line in lines:
            if re.match(r"^\s*[\-\*\u2022]\s+", line) or re.match(r"^\s*\d+[\.\)]\s+", line):
                cleaned = re.sub(r"^[\-\*\u2022\d\.\)\(]+\s*", "", line).strip()
                if cleaned and len(cleaned.split()) >= 3:
                    bullets.append(cleaned)
            if len(bullets) >= 12:
                break
        return bullets

    @staticmethod
    def _scaled_number(number_text: str, suffix: str) -> Optional[Decimal]:
        try:
            base = Decimal(number_text.replace(",", ""))
        except Exception:
            return None
        s = (suffix or "").lower()
        if s == "k":
            return base * Decimal("1000")
        if s == "m":
            return base * Decimal("1000000")
        return base

    @staticmethod
    def _normalize_employment_type(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().lower().replace("_", "-")
        text = re.sub(r"\s+", "-", text)
        aliases = {
            "fulltime": "full-time",
            "full-time": "full-time",
            "full-time-role": "full-time",
            "contract": "contract",
            "contractor": "contract",
            "remote": "remote",
            "hybrid": "hybrid",
        }
        if text in aliases:
            return aliases[text]
        return None

    @staticmethod
    def _compute_overall_confidence(extracted_skills: list[dict[str, Any]], llm_data: dict[str, Any]) -> float:
        llm_conf = llm_data.get("overall_confidence")
        if isinstance(llm_conf, (int, float)):
            base = max(0.0, min(1.0, float(llm_conf)))
        else:
            base = 0.7
        if not extracted_skills:
            return round(max(0.45, base - 0.15), 2)
        avg_skill_conf = sum(float(s.get("confidence", 0.0)) for s in extracted_skills) / len(extracted_skills)
        return round(max(0.0, min(1.0, (0.6 * base) + (0.4 * avg_skill_conf))), 2)


job_description_service = JobDescriptionService()
