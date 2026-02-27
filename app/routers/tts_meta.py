"""
TTS Meta router: lists of curated voices and background presets for UI dropdowns.
Used when creating agents: voice_id and (for ElevenLabs) background_noise_id / tts_background_preset_id from these lists.
"""

from fastapi import APIRouter, Query, Depends, HTTPException
from typing import Optional

from app.api.deps import require_tenant
from app.models.user import User
from app.services.elevenlabs_service import elevenlabs_service
from app.services.google_tts_service import google_tts_service
from app.core.logger import logger

router = APIRouter(prefix="/meta", tags=["TTS Meta"])


@router.get("/google-voices")
def get_google_voices(
    language: Optional[str] = Query(None, description="Filter by language code (e.g. en, es)"),
    user: User = Depends(require_tenant),
):
    """
    Curated Google TTS voices for dropdown (Neural2 / Chirp3).
    Returns: [{ id: voice_name, label, language, gender }].
    """
    try:
        data = google_tts_service.list_curated_voices(language=language)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.exception("Google voices list failed: %s", e)
        raise HTTPException(status_code=503, detail="Voice list temporarily unavailable")


@router.get("/elevenlabs-voices")
def get_elevenlabs_voices(
    language: Optional[str] = Query(None, description="Filter by language code (e.g. en, es)"),
    user: User = Depends(require_tenant),
):
    """
    Curated ElevenLabs voices for dropdown.
    Returns: [{ id, name, language, labels }].
    """
    try:
        data = elevenlabs_service.list_curated_voices(language=language)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.exception("ElevenLabs voices list failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail="ElevenLabs API temporarily unavailable; try again later",
        )


@router.get("/elevenlabs-backgrounds")
def get_elevenlabs_backgrounds(user: User = Depends(require_tenant)):
    """
    ElevenLabs background presets for dropdown.
    Returns: [{ id, name, description }].
    """
    try:
        data = elevenlabs_service.list_background_presets()
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.exception("ElevenLabs backgrounds list failed: %s", e)
        raise HTTPException(
            status_code=503,
            detail="ElevenLabs presets temporarily unavailable; try again later",
        )
