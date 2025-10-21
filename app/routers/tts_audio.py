"""
Google TTS Audio Router
Generates and serves Google TTS audio for Twilio calls
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
import hashlib
import base64
from urllib.parse import quote
from app.services.google_tts_service import google_tts_service
from app.core.config import settings

router = APIRouter()

# In-memory cache for generated audio (AGGRESSIVE CACHING)
audio_cache = {}
MAX_CACHE_SIZE = 500  # Cache up to 500 audio files for faster responses

# Common responses to pre-cache on startup
COMMON_RESPONSES = [
    "Hello",
    "Hi",
    "How can I help you?",
    "Could you repeat that?",
    "Sorry, I didn't catch that.",
    "Thank you for calling.",
    "Goodbye",
    "Please hold on.",
    "One moment please.",
    "I'm here to help.",
    "Can you say that again?",
    "I understand.",
    "Let me check that for you.",
    "Is there anything else?",
    "Have a great day!"
]


def generate_cache_key(text: str, language: str, voice_type: str, use_gemini: bool = False) -> str:
    """Generate unique cache key for audio"""
    content = f"{text}_{language}_{voice_type}_{use_gemini}"
    return hashlib.md5(content.encode()).hexdigest()


@router.get("/google-tts/audio")
async def serve_google_tts_audio(
    text: str = Query(..., description="Text to convert to speech"),
    lang: str = Query("en", description="Language code"),
    voice: str = Query("female", description="Voice type (male/female)"),
    gemini_flash: bool = Query(False, description="Use Gemini Flash TTS voices (ultra-fast)")
):
    """
    Generate and serve Google TTS audio on-the-fly for Twilio
    
    This endpoint is called by Twilio's <Play> verb during calls.
    Audio is cached to improve performance.
    
    Args:
        text: Text to speak
        lang: Language code (en, es, hi, ar, zh, ur)
        voice: Voice type (male or female)
        gemini_flash: Use Gemini Flash TTS voices (ultra-fast and high quality)
        
    Returns:
        Audio file as MP3
    """
    try:
        # Generate cache key
        cache_key = generate_cache_key(text, lang, voice, gemini_flash)
        
        # Check cache first
        if cache_key in audio_cache:
            voice_label = "Gemini Flash" if gemini_flash else "Neural2"
            print(f"✅ Serving cached Google TTS audio ({voice_label}): '{text[:50]}...'")
            audio_content = audio_cache[cache_key]
        else:
            # Generate new audio
            voice_label = "Gemini Flash" if gemini_flash else "Neural2"
            print(f"🎤 Generating Google TTS audio ({voice_label}): '{text[:50]}...' (lang={lang}, voice={voice})")
            
            audio_content = google_tts_service.text_to_speech(
                text=text,
                language=lang,
                voice_type=voice,
                speaking_rate=1.3,  # 30% faster for minimum latency
                pitch=0.0,
                output_format="mp3",
                use_gemini_flash=gemini_flash
            )
            
            # Cache it (with size limit)
            if len(audio_cache) >= MAX_CACHE_SIZE:
                # Remove oldest entry
                oldest_key = next(iter(audio_cache))
                audio_cache.pop(oldest_key)
                print(f"🗑️ Removed oldest cache entry (cache full)")
            
            audio_cache[cache_key] = audio_content
            print(f"💾 Cached Google TTS audio ({len(audio_content)} bytes)")
        
        # Return audio as MP3
        return Response(
            content=audio_content,
            media_type="audio/mpeg",
            headers={
                "Content-Type": "audio/mpeg",
                "Cache-Control": "public, max-age=86400",  # Cache for 24 hours
                "Content-Disposition": "inline"
            }
        )
        
    except Exception as e:
        print(f"❌ Error generating Google TTS audio: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@router.get("/google-tts/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    total_size = sum(len(audio) for audio in audio_cache.values())
    
    return {
        "cached_items": len(audio_cache),
        "max_cache_size": MAX_CACHE_SIZE,
        "total_bytes": total_size,
        "total_mb": round(total_size / (1024 * 1024), 2)
    }


@router.delete("/google-tts/cache/clear")
async def clear_cache():
    """Clear audio cache"""
    global audio_cache
    cache_size = len(audio_cache)
    audio_cache.clear()
    
    return {
        "message": "Cache cleared",
        "items_removed": cache_size
    }


@router.post("/google-tts/cache/warmup")
async def warmup_cache():
    """Pre-cache common responses for instant delivery"""
    cached_count = 0
    
    try:
        # Pre-cache common responses in English with both male and female voices
        for text in COMMON_RESPONSES:
            for voice_type in ["male", "female"]:
                for use_gemini in [True, False]:
                    cache_key = generate_cache_key(text, "en", voice_type, use_gemini)
                    
                    if cache_key not in audio_cache:
                        try:
                            audio_content = google_tts_service.text_to_speech(
                                text=text,
                                language="en",
                                voice_type=voice_type,
                                speaking_rate=1.3,
                                pitch=0.0,
                                output_format="mp3",
                                use_gemini_flash=use_gemini
                            )
                            
                            audio_cache[cache_key] = audio_content
                            cached_count += 1
                        except Exception as e:
                            print(f"⚠️ Failed to cache '{text}' ({voice_type}): {e}")
        
        return {
            "message": "Cache warmup completed",
            "cached_items": cached_count,
            "total_cache_size": len(audio_cache)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache warmup failed: {str(e)}")

