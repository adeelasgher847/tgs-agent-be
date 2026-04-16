from typing import Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_member_or_admin, require_tenant
from app.models.user import User
from app.schemas.base import SuccessResponse
from app.schemas.tts import TTSProviderListOut, TTSVoiceListOut
from app.services.tts_catalog_service import tts_catalog_service
from app.utils.response import create_success_response


router = APIRouter()


@router.get("/providers", response_model=SuccessResponse[TTSProviderListOut])
def list_tts_providers(
    user: User = Depends(require_tenant),
    member_user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    providers = tts_catalog_service.list_providers(db, active_only=True)
    return create_success_response({"providers": providers}, "TTS providers retrieved successfully")


@router.get("/voices", response_model=SuccessResponse[TTSVoiceListOut])
def list_tts_voices(
    provider_id: uuid.UUID = Query(..., description="TTS provider ID"),
    language: Optional[str] = Query(None, description="Optional language code filter"),
    user: User = Depends(require_tenant),
    member_user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    provider = tts_catalog_service.get_provider_by_id(db, provider_id)
    if not provider or not provider.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Active TTS provider not found.",
        )

    voices = tts_catalog_service.list_voices(
        db,
        provider_id=provider_id,
        language_code=language,
        active_only=True,
    )
    return create_success_response({"voices": voices}, "TTS voices retrieved successfully")


