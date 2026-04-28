import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_admin_or_owner, require_tenant
from app.models.tts_provider import TTSProvider
from app.models.tts_voice import TTSVoice
from app.models.user import User
from app.schemas.base import SuccessResponse
from app.schemas.tts import TTSProviderCreate, TTSProviderOut, TTSVoiceCreate, TTSVoiceOut
from app.services.tts_catalog_service import tts_catalog_service
from app.utils.response import create_success_response


router = APIRouter()


@router.post("/providers", response_model=SuccessResponse[TTSProviderOut], status_code=status.HTTP_201_CREATED)
def create_tts_provider(
    payload: TTSProviderCreate,
    user: User = Depends(require_tenant),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    provider = TTSProvider(
        slug=payload.slug.strip().lower(),
        display_name=payload.display_name.strip(),
        is_active=payload.is_active,
        supports_streaming=payload.supports_streaming,
        supports_ssml=payload.supports_ssml,
    )
    db.add(provider)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provider slug already exists.",
        )
    db.refresh(provider)
    return create_success_response(provider, "TTS provider created successfully", status.HTTP_201_CREATED)


@router.delete("/providers/{provider_id}", response_model=SuccessResponse[dict])
def delete_tts_provider(
    provider_id: uuid.UUID,
    user: User = Depends(require_tenant),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    provider = db.query(TTSProvider).filter(TTSProvider.id == provider_id).first()
    if not provider:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="TTS provider not found.")

    db.query(TTSVoice).filter(TTSVoice.provider_id == provider_id).delete(synchronize_session=False)
    db.delete(provider)
    db.commit()
    return create_success_response({"deleted": True}, "TTS provider deleted successfully")


@router.post("/voices", response_model=SuccessResponse[TTSVoiceOut], status_code=status.HTTP_201_CREATED)
def create_tts_voice(
    payload: TTSVoiceCreate,
    user: User = Depends(require_tenant),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    provider = db.query(TTSProvider).filter(TTSProvider.id == payload.provider_id).first()
    if not provider:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="TTS provider not found.")

    voice = TTSVoice(
        provider_id=payload.provider_id,
        external_voice_id=payload.external_voice_id.strip(),
        display_name=payload.display_name.strip(),
        language_code=payload.language_code,
        gender=payload.gender,
        accent=payload.accent,
        description=payload.description,
        preview_audio_url=payload.preview_audio_url,
        sample_rate_hz=payload.sample_rate_hz,
        is_active=payload.is_active,
        metadata_json=payload.metadata_json,
    )
    db.add(voice)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Voice already exists for this provider.",
        )
    db.refresh(voice)
    return create_success_response(voice, "TTS voice created successfully", status.HTTP_201_CREATED)


@router.delete("/voices/{voice_id}", response_model=SuccessResponse[dict])
def delete_tts_voice(
    voice_id: uuid.UUID,
    user: User = Depends(require_tenant),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    voice = db.query(TTSVoice).filter(TTSVoice.id == voice_id).first()
    if not voice:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="TTS voice not found.")

    db.delete(voice)
    db.commit()
    return create_success_response({"deleted": True}, "TTS voice deleted successfully")


@router.post("/providers/{provider_slug}/sync-voices", response_model=SuccessResponse[dict])
def sync_provider_voices(
    provider_slug: str,
    user: User = Depends(require_tenant),
    admin_user: User = Depends(require_admin_or_owner),
    db: Session = Depends(get_db),
):
    try:
        result = tts_catalog_service.sync_provider_voices(db, provider_slug=provider_slug)
        return create_success_response(result, f"{provider_slug} voices synchronized successfully")
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to synchronize voices for the selected provider.",
        )
