from typing import Optional
import uuid
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
import requests
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_member_or_admin, require_tenant
from app.models.tts_voice import TTSVoice
from app.models.user import User
from app.schemas.base import SuccessResponse
from app.schemas.tts import (
    TTSProviderListOut,
    TTSVoiceListOut,
)
from app.services.google_tts_service import google_tts_service
from app.services.tts_catalog_service import tts_catalog_service
from app.utils.response import create_success_response


router = APIRouter()

GOOGLE_PREVIEW_TEXT = "Hi, this is a short Google voice preview."


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


@router.get("/voices/{voice_id}/preview")
def preview_tts_voice_audio(
    voice_id: uuid.UUID,
    user: User = Depends(require_tenant),
    member_user: User = Depends(require_member_or_admin),
    db: Session = Depends(get_db),
):
    voice = db.query(TTSVoice).filter(TTSVoice.id == voice_id, TTSVoice.is_active == True).first()  # noqa: E712
    if not voice:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Active TTS voice not found.")

    preview_audio_url = (voice.preview_audio_url or "").strip()
    provider_slug = (getattr(getattr(voice, "provider", None), "slug", "") or "").strip().lower()

    if not preview_audio_url:
        # Google voice previews do not always have a static preview URL.
        # In that case, synthesize a short preview clip on demand.
        if provider_slug == "google":
            try:
                audio_content = google_tts_service.text_to_speech(
                    text=GOOGLE_PREVIEW_TEXT,
                    speaking_rate=1.0,
                    output_format="mp3",
                    use_chirp3_hd=True,
                    voice_name_override=(voice.external_voice_id or "").strip() or None,
                )
                return Response(
                    content=audio_content,
                    media_type="audio/mpeg",
                    headers={
                        "Cache-Control": "private, max-age=120",
                        "X-Content-Type-Options": "nosniff",
                    },
                )
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Failed to synthesize Google preview audio.",
                )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Preview audio is not available for this voice.",
        )

    parsed = urlparse(preview_audio_url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid preview audio URL scheme.",
        )

    try:
        upstream = requests.get(preview_audio_url, timeout=12)
        upstream.raise_for_status()
    except requests.RequestException:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to fetch preview audio from provider.",
        )

    media_type = upstream.headers.get("content-type") or "audio/mpeg"
    return Response(
        content=upstream.content,
        media_type=media_type,
        headers={
            "Cache-Control": "private, max-age=120",
            "X-Content-Type-Options": "nosniff",
        },
    )


