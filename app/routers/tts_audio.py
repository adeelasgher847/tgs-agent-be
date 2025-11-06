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

# In-memory cache for generated audio
audio_cache = {}
MAX_CACHE_SIZE = 200  # Cache up to 200 audio files


def generate_cache_key(text: str, language: str, voice_type: str, use_chirp3_hd: bool = False, format: str = "mp3") -> str:
    """Generate unique cache key for audio"""
    content = f"{text}_{language}_{voice_type}_{use_chirp3_hd}_{format}"
    return hashlib.md5(content.encode()).hexdigest()


@router.get("/google-tts/audio")
async def serve_google_tts_audio(
    text: str = Query(..., description="Text to convert to speech"),
    lang: str = Query("en", description="Language code"),
    voice: str = Query("female", description="Voice type (male/female)"),
    chirp3_hd: bool = Query(True, description="Use Chirp 3: HD model - ultra-realistic voices"),
    format: str = Query("mp3", description="Audio format (mp3, mulaw) - mulaw is faster for Twilio")
):
    """
    Generate and serve Google TTS audio on-the-fly for Twilio
    
    This endpoint is called by Twilio's <Play> verb during calls.
    Audio is cached to improve performance.
    
    Features:
    - Chirp 3: HD model - ultra-realistic, human-like voices
    - Telephony-optimized audio (volume boost + effects profile)
    - High-quality audio (24kHz MP3 / 8kHz MULAW)
    - Smart caching for performance
    
    Args:
        text: Text to speak
        lang: Language code (en, es, hi, ar, zh, ur)
        voice: Voice type (male or female)
        chirp3_hd: Use Chirp 3: HD model (default: True)
        format: Audio format (mp3 or mulaw)
        
    Returns:
        Audio file as MP3 or MULAW
    """
    try:
        # Validate format
        valid_formats = ["mp3", "mulaw"]
        if format not in valid_formats:
            format = "mp3"
        
        # Generate cache key
        cache_key = generate_cache_key(text, lang, voice, chirp3_hd, format)
        
        # Check cache first
        if cache_key in audio_cache:
            voice_label = "Chirp 3: HD" if chirp3_hd else "Neural2"
            print(f"✅ Serving cached Google TTS audio ({voice_label}): '{text[:50]}...'")
            audio_content = audio_cache[cache_key]
        else:
            # Generate new audio
            voice_label = "Chirp 3: HD" if chirp3_hd else "Neural2"
            print(f"🎤 Generating Google TTS audio ({voice_label}): '{text[:50]}...' (lang={lang}, voice={voice})")
            
            # Optimized speaking rate for natural conversation (slightly slower for clarity)
            rate = 0.95  # Slightly slower and more natural
            
            audio_content = google_tts_service.text_to_speech(
                text=text,
                language=lang,
                voice_type=voice,
                speaking_rate=rate,  # Optimized for natural conversation
                pitch=0.0,
                output_format=format,  # Use requested format (mp3 or mulaw)
                use_chirp3_hd=chirp3_hd
            )
            
            # Cache it (with size limit)
            if len(audio_cache) >= MAX_CACHE_SIZE:
                # Remove oldest entry
                oldest_key = next(iter(audio_cache))
                audio_cache.pop(oldest_key)
                print(f"🗑️ Removed oldest cache entry (cache full)")
            
            audio_cache[cache_key] = audio_content
            print(f"💾 Cached Google TTS audio ({len(audio_content)} bytes)")
        
        # Return audio with appropriate media type
        media_types = {
            "mp3": "audio/mpeg",
            "mulaw": "audio/x-mulaw"  # MULAW with proper MIME type
        }
        media_type = media_types.get(format, "audio/mpeg")
        
        # Set proper Content-Type with sample rate for MULAW
        if format == "mulaw":
            content_type_header = "audio/x-mulaw;rate=8000"  # Specify 8kHz for Twilio
        else:
            content_type_header = media_type
        
        # Return audio with optimized headers
        return Response(
            content=audio_content,
            media_type=media_type,
            headers={
                "Content-Type": content_type_header,  # Proper type with rate
                "Cache-Control": "public, max-age=31536000, immutable",  # Cache 1 year (aggressive!)
                "Content-Disposition": "inline",
                "X-Content-Type-Options": "nosniff",
                "Access-Control-Allow-Origin": "*"  # Allow cross-origin for Twilio
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

