from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.schemas.base import SuccessResponse
from app.utils.response import create_success_response
from app.core.config import settings
from app.services.elevenlabs_service import elevenlabs_service
from app.core.logger import logger

router = APIRouter()

@router.get("/health", response_model=SuccessResponse[dict])
def health_check():
    return create_success_response({"status": "ok"}, "Health check successful") 


class ElevenLabsHealthRequest(BaseModel):
    voice_id: str = Field(..., min_length=3, max_length=128)
    text: str = Field("Hello from health test.", min_length=1, max_length=500)
    model_id: str = Field("eleven_flash_v2_5", min_length=1, max_length=64)
    output_format: str = Field("ulaw_8000", min_length=1, max_length=64)
    optimize_streaming_latency: int = Field(4, ge=0, le=4)
    check_voice_exists: bool = True


@router.post("/health/elevenlabs", response_model=SuccessResponse[dict])
def elevenlabs_health_check(payload: ElevenLabsHealthRequest):
    """
    Lightweight diagnostic endpoint for Render/prod troubleshooting.
    Safe output only: no full API key is returned.
    """
    api_key = (settings.ELEVENLABS_API_KEY or "").strip()
    key_tail = api_key[-6:] if len(api_key) >= 6 else ("*" * len(api_key))

    if not api_key:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY is missing.")

    response_payload = {
        "api_key_present": True,
        "api_key_length": len(api_key),
        "api_key_tail": key_tail,
        "voice_id": payload.voice_id,
        "model_id": payload.model_id,
        "output_format": payload.output_format,
    }

    # Optional pre-check: verify requested voice exists for current key/workspace.
    if payload.check_voice_exists:
        try:
            voices = elevenlabs_service.get_available_voices().get("voices", [])
            voice_ids = {v.get("voice_id") for v in voices if v.get("voice_id")}
            response_payload["voices_count"] = len(voice_ids)
            response_payload["voice_exists"] = payload.voice_id in voice_ids
        except Exception as e:
            logger.warning("ElevenLabs voice list failed in health check: %s", e)
            response_payload["voices_count"] = None
            response_payload["voice_exists"] = None
            response_payload["voice_check_error"] = str(e)

    try:
        audio = elevenlabs_service.text_to_speech(
            text=payload.text,
            voice_id=payload.voice_id,
            model_id=payload.model_id,
            output_format=payload.output_format,
            optimize_streaming_latency=payload.optimize_streaming_latency,
        )
        response_payload["tts_ok"] = True
        response_payload["audio_bytes"] = len(audio or b"")
        return create_success_response(response_payload, "ElevenLabs health test succeeded.")
    except Exception as e:
        logger.error("ElevenLabs health test failed: %s", e, exc_info=True)
        response_payload["tts_ok"] = False
        response_payload["error"] = str(e)
        raise HTTPException(status_code=502, detail=response_payload)