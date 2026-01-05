"""
Service functions for bidirectional streaming.
Handles TTS generation and TwiML building.
"""

from app.services.google_tts_service import google_tts_service
from app.routers.tts_audio import audio_cache, generate_cache_key
from app.utils.audio_utils import add_ambient_noise_to_mulaw
from app.core.config import settings
from app.core.logger import logger


async def generate_mulaw_tts(text: str, lang: str = "en", voice: str = "female", use_chirp3_hd: bool = True, speaking_rate: float = 0.95, use_ssml: bool = False, add_office_bg: bool = False) -> bytes:
    """
    Generate mu-law (8kHz) TTS audio using Chirp 3: HD model.
    Optimized for word-by-word streaming with caching.
    
    Args:
        text: Text or SSML to convert to speech
        lang: Language code
        voice: Voice type (male/female)
        use_chirp3_hd: Use Chirp 3 HD model
        speaking_rate: Speech rate
        use_ssml: Whether text contains SSML markup
        add_office_bg: Add office background noise to audio (mixed at audio level)
    
    Note: Google TTS natively supports SSML. Text starting with <speak> is auto-detected.
    """
    # Skip empty text
    if not text or not text.strip():
        return b''
    
    try:
        # Cache key aligned with existing cache strategy (include ssml and office_bg flags)
        cache_key = generate_cache_key(text.strip(), lang, voice, use_chirp3_hd, "mulaw") + ("_ssml" if use_ssml else "") + ("_officebg" if add_office_bg else "")

        if cache_key in audio_cache:
            logger.debug(f"✅ Serving cached MULAW TTS ('{text[:30]}...')")
            return audio_cache[cache_key]

        # Use 8kHz MULAW for Twilio with Chirp 3: HD model - Optimized for small chunks
        # Google TTS auto-detects SSML if text starts with <speak>
        # Let SSML control prosody (use defaults when SSML present, don't override)
        logger.info(f"🎤 Generating fresh MULAW TTS ('{text[:30]}...') [chirp3_hd={use_chirp3_hd}, ssml={use_ssml}]")
        audio_content = google_tts_service.text_to_speech(
            text=text.strip(),
            language=lang,
            voice_type=voice,
            speaking_rate=1.0 if use_ssml else speaking_rate,  # Use 1.0 (default) for SSML to respect prosody tags
            pitch=0.0,  # Always 0, let SSML <prosody pitch> handle variations
            output_format="mulaw",
            use_chirp3_hd=use_chirp3_hd
        )

        # Mix office background noise if enabled (NO DOWNLOAD - generates programmatically!)
        if add_office_bg:
            audio_content = add_ambient_noise_to_mulaw(
                audio_content, 
                noise_level=0.06  # Office background noise (~-24dB) - audible but not distracting
            )
            logger.info(f"🔊 Added office background noise to TTS audio (noise_level: 0.06)")

        # Cache for instant reuse (especially useful for repeated words/phrases)
        audio_cache[cache_key] = audio_content
        return audio_content
    except Exception as e:
        logger.error(f"❌ Error in generate_mulaw_tts: {e}", exc_info=True)
        raise


def build_streaming_twiml(call_session_id: str, agent_id: str) -> str:
    """
    Replace <Play> with <Start><Stream> to enable realtime MULAW streaming.
    Configure Twilio edge/region via settings.TWILIO_EDGE if available.
    """
    from twilio.twiml.voice_response import VoiceResponse, Start, Stream, Parameter
    
    # Your public WebSocket endpoint that handles Twilio Media Streams:
    # Example: wss://your-domain.com/api/v1/stream/ws/bidirectional/{callSessionId}/{agentId}
    ws_url = f"{settings.WEBHOOK_BASE_URL.replace('http', 'ws')}/api/v1/stream/ws/bidirectional/{call_session_id}/{agent_id}"
    if ws_url.startswith("ws://"):
        ws_url = "wss://" + ws_url[len("ws://"):]  # enforce TLS for Twilio

    edge = getattr(settings, "TWILIO_EDGE", None)  # e.g., "ashburn", "singapore", "dublin"
    vr = VoiceResponse()
    with vr.start() as s:
        stream = Stream(url=ws_url, name=f"tts-stream-{agent_id}")
        # Forward region hint to your WS (for observability); set real Twilio edge via account/call config.
        if edge:
            stream.parameter(Parameter(name="edge", value=edge))
        s.append(stream)

    logger.debug(f"🛠️ Built streaming TwiML for session {call_session_id}")
    return str(vr)


def build_tts_only_twiml(call_session_id: str, agent_id: str, record_callback_url: str) -> str:
    """
    Build TwiML for TTS-only WebSocket streaming + Recording for next input
    
    Flow:
    1. Connect to TTS-only WebSocket
    2. WebSocket auto-plays pending TTS from metadata
    3. After playback, start recording for next user input
    
    Args:
        call_session_id: Call session UUID
        agent_id: Agent UUID
        record_callback_url: URL for recording callback
    
    Returns:
        TwiML string
    """
    from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
    
    # Build TTS-only WebSocket URL
    ws_url = f"{settings.WEBHOOK_BASE_URL.replace('http', 'ws')}/api/v1/voice/ws/tts-only/{call_session_id}/{agent_id}"
    if ws_url.startswith("ws://"):
        ws_url = "wss://" + ws_url[len("ws://"):]  # enforce TLS
    
    vr = VoiceResponse()
    
    # Connect to TTS-only WebSocket for streaming playback
    connect = Connect()
    stream = Stream(url=ws_url, name=f"tts-only-{agent_id}")
    connect.append(stream)
    vr.append(connect)
    
    # After TTS playback, start recording for next user input
    vr.record(
        action=record_callback_url,
        method='POST',
        timeout=3,  # Fast detection
        max_length=120,
        play_beep=False,
        trim='do-not-trim',
        recording_status_callback=record_callback_url,
        recording_status_callback_method='POST',
        transcribe=False
    )
    
    logger.debug(f"🛠️ Built TTS-only TwiML for session {call_session_id}")
    return str(vr)

