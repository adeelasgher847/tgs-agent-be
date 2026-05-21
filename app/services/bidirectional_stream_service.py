"""
Service functions for bidirectional streaming.
Handles TTS generation and TwiML building.
"""

from typing import Optional, Any

from app.services.google_tts_service import google_tts_service
from app.routers.tts_audio import audio_cache, generate_cache_key
from app.core.config import settings
from app.core.logger import logger
from app.core.agent_runtime import resolve_tts_runtime
from app.utils.tts_adapter import get_tts_adapter
from app.utils.eleven_tts_text import prepare_tts_text_for_provider
from app.utils.audio_utils import add_ambient_noise_to_mulaw


def _resolve_tts_provider_slug(agent: Optional[Any]) -> Optional[str]:
    if not agent:
        return None
    return resolve_tts_runtime(agent).adapter_slug


async def generate_mulaw_tts(
    text: str,
    lang: str = "en",
    voice: str = "female",
    use_chirp3_hd: bool = True,
    speaking_rate: float = 0.95,
    use_ssml: bool = False,
    add_office_bg: bool = False,
    agent: Optional[Any] = None,
) -> bytes:
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
        add_office_bg: Deprecated; ignored (kept for call-site compatibility).
    
    Note: Google TTS natively supports SSML. Text starting with <speak> is auto-detected.
    """
    # Skip empty text
    if not text or not text.strip():
        return b''

    try:
        # Cache key aligned with existing cache strategy (include ssml flag)
        provider_slug = _resolve_tts_provider_slug(agent) or "google"
        tts_text = prepare_tts_text_for_provider(text.strip(), provider_slug)
        if not tts_text:
            return b""

        tts_runtime = resolve_tts_runtime(agent) if agent else None
        selected_voice = voice
        if provider_slug != "google":
            if tts_runtime and tts_runtime.voice_external_id:
                selected_voice = tts_runtime.voice_external_id
            else:
                tts_voice = getattr(agent, "tts_voice", None) if agent else None
                selected_voice = getattr(tts_voice, "external_voice_id", None) or voice

        cache_key = (
            generate_cache_key(tts_text, lang, f"{provider_slug}:{selected_voice}", use_chirp3_hd, "mulaw")
            + ("_ssml" if use_ssml else "")
            + ("_officebg" if add_office_bg else "")
        )

        if cache_key in audio_cache:
            logger.debug(f"✅ Serving cached MULAW TTS ('{text[:30]}...')")
            return audio_cache[cache_key]

        if provider_slug == "google":
            google_voice_name = None
            if tts_runtime and tts_runtime.voice_external_id:
                google_voice_name = tts_runtime.voice_external_id
            else:
                tts_voice = getattr(agent, "tts_voice", None) if agent else None
                google_voice_name = getattr(tts_voice, "external_voice_id", None)
            # Use 8kHz MULAW for Twilio with Chirp 3: HD model - Optimized for small chunks
            # Google TTS auto-detects SSML if text starts with <speak>
            # Let SSML control prosody (use defaults when SSML present, don't override)
            logger.info(f"🎤 Generating fresh MULAW TTS ('{text[:30]}...') [provider=google, chirp3_hd={use_chirp3_hd}, ssml={use_ssml}]")
            audio_content = google_tts_service.text_to_speech(
                text=tts_text,
                language=lang,
                voice_type=voice,
                speaking_rate=1.0 if use_ssml else speaking_rate,  # Use 1.0 (default) for SSML to respect prosody tags
                pitch=0.0,  # Always 0, let SSML <prosody pitch> handle variations
                output_format="mulaw",
                use_chirp3_hd=use_chirp3_hd,
                voice_name_override=google_voice_name,
            )
        else:
            external_voice_id = None
            settings_json: dict = {}
            if tts_runtime:
                external_voice_id = tts_runtime.voice_external_id
                settings_json = dict(tts_runtime.settings_json)
            if not external_voice_id:
                tts_voice = getattr(agent, "tts_voice", None) if agent else None
                external_voice_id = getattr(tts_voice, "external_voice_id", None)
            if not external_voice_id:
                raise ValueError("TTS voice is not configured for the selected provider.")
            if not settings_json:
                settings_json = dict(getattr(agent, "tts_settings_json", None) or {})
            settings_json.setdefault("output_format", "ulaw_8000")
            adapter = get_tts_adapter(provider_slug)
            logger.info(f"🎤 Generating fresh MULAW TTS ('{text[:30]}...') [provider={provider_slug}]")
            audio_content = adapter.synthesize(
                text=tts_text,
                voice_external_id=external_voice_id,
                settings_json=settings_json,
            )

        if add_office_bg:
            audio_content = add_ambient_noise_to_mulaw(audio_content, noise_level=0.06)

        # Cache for instant reuse (especially useful for repeated words/phrases)
        audio_cache[cache_key] = audio_content
        return audio_content
    except Exception as e:
        logger.error(f"❌ Error in generate_mulaw_tts: {e}", exc_info=True)
        raise


def build_streaming_twiml(call_session_id: str, agent_id: str) -> str:
    """
    Build <Connect><Stream> TwiML for bidirectional media streaming.
    <Connect> keeps the call alive and enables two-way audio over WebSocket.
    """
    from twilio.twiml.voice_response import VoiceResponse, Connect, Stream, Parameter

    ws_url = f"{settings.WEBHOOK_BASE_URL.replace('http', 'ws')}/api/v1/stream/ws/bidirectional/{call_session_id}/{agent_id}"
    if ws_url.startswith("ws://"):
        ws_url = "wss://" + ws_url[len("ws://"):]

    edge = getattr(settings, "TWILIO_EDGE", None)

    vr = VoiceResponse()
    connect = Connect()
    stream = Stream(url=ws_url, name=f"tts-stream-{agent_id}")
    stream.parameter(Parameter(name="callSessionId", value=call_session_id))
    stream.parameter(Parameter(name="agentId", value=agent_id))
    if edge:
        stream.parameter(Parameter(name="edge", value=edge))
    connect.append(stream)
    vr.append(connect)

    logger.debug(f"Built bidirectional streaming TwiML for session {call_session_id}")
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
    from twilio.twiml.voice_response import VoiceResponse, Connect, Stream, Parameter
    
    # Build TTS-only WebSocket URL
    ws_url = f"{settings.WEBHOOK_BASE_URL.replace('http', 'ws')}/api/v1/voice/ws/tts-only/{call_session_id}/{agent_id}"
    if ws_url.startswith("ws://"):
        ws_url = "wss://" + ws_url[len("ws://"):]  # enforce TLS
    
    vr = VoiceResponse()
    
    # Connect to TTS-only WebSocket for streaming playback
    connect = Connect()
    stream = Stream(url=ws_url, name=f"tts-only-{agent_id}")
    edge = getattr(settings, "TWILIO_EDGE", None)
    if edge:
        stream.parameter(Parameter(name="edge", value=edge))
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

