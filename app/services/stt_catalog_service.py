from __future__ import annotations

from typing import Optional
import uuid

from sqlalchemy.orm import Session

from app.models.stt_provider import STTProvider
from app.models.stt_model import STTModel


class STTCatalogService:

    def list_providers(self, db: Session, active_only: bool = True) -> list[STTProvider]:
        query = db.query(STTProvider)
        if active_only:
            query = query.filter(STTProvider.is_active == True)  # noqa: E712
        return query.order_by(STTProvider.display_name.asc()).all()

    def list_models(
        self,
        db: Session,
        provider_id: uuid.UUID,
        language_code: Optional[str] = None,
        active_only: bool = True,
    ) -> list[STTModel]:
        query = db.query(STTModel).filter(STTModel.provider_id == provider_id)
        if active_only:
            query = query.filter(STTModel.is_active == True)  # noqa: E712
        if language_code:
            query = query.filter(STTModel.language_code == language_code)
        return query.order_by(STTModel.display_name.asc()).all()

    def get_provider_by_slug(self, db: Session, slug: str) -> Optional[STTProvider]:
        return (
            db.query(STTProvider)
            .filter(STTProvider.slug == slug)
            .first()
        )

    def get_model_by_provider_and_external_id(
        self,
        db: Session,
        provider_id: uuid.UUID,
        external_model_id: str,
    ) -> Optional[STTModel]:
        return (
            db.query(STTModel)
            .filter(
                STTModel.provider_id == provider_id,
                STTModel.external_model_id == external_model_id,
            )
            .first()
        )


stt_catalog_service = STTCatalogService()
