from typing import Any
import uuid

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

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


job_description_service = JobDescriptionService()
