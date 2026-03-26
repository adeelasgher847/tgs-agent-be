from typing import Any
import uuid
import re
import zipfile
from io import BytesIO

from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from pypdf import PdfReader

from app.models.job_description import JobDescription
from app.schemas.job_description import JobDescriptionCreateManual, JobDescriptionUpdate


class JobDescriptionService:
    def create_manual(
        self,
        db: Session,
        payload: JobDescriptionCreateManual,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
    ) -> JobDescription:
        data = payload.model_dump()

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

    def list_by_tenant(self, db: Session, tenant_id: uuid.UUID) -> list[JobDescription]:
        return (
            db.query(JobDescription)
            .filter(JobDescription.tenant_id == tenant_id)
            .order_by(JobDescription.created_at.desc())
            .all()
        )

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

        for key, value in updates.items():
            setattr(jd, key, value)

        jd.version = (jd.version or 1) + 1
        jd.updated_by = user_id

        db.commit()
        db.refresh(jd)
        return jd

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
            # Lightweight deterministic enrichment for now.
            source_text = (jd.raw_text or "").lower()
            skills = jd.required_skills or self._extract_skills(source_text)
            keywords = self._extract_keywords(source_text, skills)
            weight_matrix = self._build_skill_weight_matrix(skills)
            matching_criteria = {
                "required_skills": skills,
                "minimum_years_experience": jd.years_experience_min,
                "education_requirements": jd.education_requirements,
                "location": jd.location,
                "employment_type": jd.employment_type,
                "keywords": keywords,
            }

            jd.required_skills = skills
            jd.extracted_skills = [{"skill": skill, "confidence": 0.85} for skill in skills]
            jd.keywords = keywords
            jd.skill_weight_matrix = weight_matrix
            jd.matching_criteria = matching_criteria
            jd.processing_status = "READY"
            jd.version = (jd.version or 1) + 1
            jd.updated_by = user_id
            db.commit()
            db.refresh(jd)
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

    @staticmethod
    def _build_raw_text_from_manual_fields(data: dict[str, Any]) -> str:
        parts = [
            f"Job Title: {data.get('job_title') or ''}",
            f"Required Skills: {', '.join(data.get('required_skills') or [])}",
            f"Experience: {data.get('years_experience_min') or ''} years minimum",
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

    @staticmethod
    def _build_skill_weight_matrix(skills: list[str]) -> dict[str, float]:
        if not skills:
            return {}
        total = len(skills)
        return {skill: round((total - idx) / total, 2) for idx, skill in enumerate(skills)}


job_description_service = JobDescriptionService()
