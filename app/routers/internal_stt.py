"""
STT catalog API — read-only endpoints for frontend provider/model dropdowns.
Mirrors internal_tts.py pattern.
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_tenant
from app.models.user import User
from app.models.stt_provider import STTProvider
from app.models.stt_model import STTModel
from app.services.stt_catalog_service import stt_catalog_service


router = APIRouter()


# ── Response schemas ──────────────────────────────────────────────────────────

class STTProviderOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    slug: str
    display_name: str
    is_active: bool
    supports_streaming: bool


class STTModelOut(BaseModel):
    """Public STT model projection — never exposes metadata_json (internal API params)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    provider_id: uuid.UUID = Field(..., serialization_alias="providerId")
    external_model_id: str = Field(..., serialization_alias="modelId")
    display_name: str = Field(..., serialization_alias="displayName")
    language_code: str = Field(..., serialization_alias="languageCode")
    sample_rate_hz: int = Field(..., serialization_alias="sampleRateHz")
    encoding: str
    is_active: bool = Field(..., serialization_alias="isActive")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/providers", response_model=List[STTProviderOut])
def list_stt_providers(
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
) -> List[STTProvider]:
    """List active STT providers for agent configuration dropdowns."""
    return stt_catalog_service.list_providers(db, active_only=True)


@router.get("/providers/{provider_id}/models", response_model=List[STTModelOut])
def list_stt_models(
    provider_id: uuid.UUID,
    language_code: Optional[str] = None,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db),
) -> List[STTModel]:
    """List active STT models for a provider, optionally filtered by language_code."""
    provider = db.query(STTProvider).filter(STTProvider.id == provider_id).first()
    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="STT provider not found.",
        )
    return stt_catalog_service.list_models(
        db, provider_id=provider_id, language_code=language_code, active_only=True
    )
