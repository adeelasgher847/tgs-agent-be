"""
Fast Conversational AI Router using Twilio Gather + Google STT + LLM + TTS
Optimized for 3-4 second latency per turn
"""

from fastapi import APIRouter, Request, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse, Response
from sqlalchemy.orm import Session
from typing import Optional
from twilio.twiml.voice_response import VoiceResponse, Gather
from datetime import datetime, timezone
import uuid
import asyncio
import requests
import sys

from app.api.deps import get_db
from app.services.twilio_service import twilio_service
from app.services.agent_service import agent_service
from app.services.call_session_service import call_session_service
from app.services.google_stt_service import google_stt_service
from app.services.google_tts_service import google_tts_service
from app.services.voice_logging_service import VoiceLoggingService
from app.services.gemini_service import gemini_service
from app.services.openai_service import openai_service
from app.services.model_service import ModelService
from app.core.config import settings
from app.utils.twilio_validation import get_request_body
from app.routers.general_websocket import broadcast_transcript_update, broadcast_call_event
from urllib.parse import quote
import hashlib

router = APIRouter()
model_service = ModelService()

# Import TTS audio cache from tts_audio router for pre-generation optimization
from app.routers.tts_audio import audio_cache


def generate_cache_key(text: str, language: str, voice_type: str, use_gemini: bool = False) -> str:
    """Generate unique cache key for TTS audio (same as tts_audio.py)"""
    content = f"{text}_{language}_{voice_type}_{use_gemini}"
    return hashlib.md5(content.encode()).hexdigest()


def pre_generate_tts(text: str, language: str = "en", voice_type: str = "female", use_gemini_flash: bool = True) -> None:
    """
    Pre-generate TTS audio and cache it for instant playback
    Uses Gemini Flash TTS for ultra-fast generation (200-300ms)
    """
    try:
        cache_key = generate_cache_key(text, language, voice_type, use_gemini_flash)
        
        if cache_key not in audio_cache:
            # Generate audio with Gemini Flash
            voice_label = "Gemini Flash" if use_gemini_flash else "Neural2"
            audio_content = google_tts_service.text_to_speech(
                text=text,
                language=language,
                voice_type=voice_type,
                speaking_rate=1.3,  # 30% faster for minimum latency
                pitch=0.0,
                output_format="mp3",
                use_gemini_flash=use_gemini_flash
            )
            
            # Cache it
            audio_cache[cache_key] = audio_content
            print(f"⚡ Pre-cached TTS ({voice_label}): '{text[:30]}...' ({len(audio_content)} bytes)")
            sys.stdout.flush()
    except Exception as e:
        # Non-critical - will generate on-demand if pre-generation fails
        print(f"⚠️ TTS pre-cache failed: {e}")
        sys.stdout.flush()


def get_call_duration_realtime(call_session) -> str:
    """Get real-time call duration in human-readable format"""
    if not call_session or not call_session.start_time:
        return "00:00"
    
    current_time = datetime.now(timezone.utc)
    duration_seconds = (current_time - call_session.start_time).total_seconds()
    
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    
    return f"{minutes:02d}:{seconds:02d}"


def get_agent_voice(agent) -> str:
    """Get the appropriate Twilio voice based on agent's voice type and language"""
    if not agent:
        return "Polly.Joanna"
    
    voice_type = agent.voice_type or "female"
    language = agent.language or "en"
    
    voice_map = {
        "en": {"male": "Polly.Matthew", "female": "Polly.Joanna"},
        "es": {"male": "Polly.Miguel", "female": "Polly.Penelope"},
        "hi": {"male": "Polly.Aditi", "female": "Polly.Aditi"},
        "ar": {"male": "Polly.Zeina", "female": "Polly.Zeina"},
        "zh": {"male": "Polly.Zhiyu", "female": "Polly.Zhiyu"},
        "ur": {"male": "Polly.Aditi", "female": "Polly.Aditi"}
    }
    
    return voice_map.get(language, voice_map["en"]).get(voice_type, "Polly.Joanna")


def get_gather_language(agent) -> str:
    """Get language code for Twilio Gather based on agent language"""
    if not agent or not agent.language:
        return "en-US"
    
    language_map = {
        "en": "en-US",
        "es": "es-ES",
        "hi": "hi-IN",
        "ar": "ar-SA",
        "zh": "zh-CN",
        "ur": "ur-PK"
    }
    
    return language_map.get(agent.language, "en-US")


async def add_to_transcript(
    call_session, 
    role: str, 
    message: str, 
    db: Session, 
    message_type: str = "speech",
    confidence: Optional[float] = None
):
    """Add a message to the transcript"""
    try:
        from app.services.transcript_service import transcript_service
        
        transcript_message = await transcript_service.add_and_broadcast_message(
            db=db,
            call_session_id=call_session.id,
            role=role,
            message=message,
            message_type=message_type,
            confidence=confidence
        )
        
        # Update legacy call_transcript field
        conversation = transcript_service.get_conversation_array(db, call_session.id)
        call_session.call_transcript = conversation
        db.commit()
        
        return transcript_message
    except Exception as e:
        print(f"❌ Failed to add transcript message: {e}")
        import traceback
        traceback.print_exc()


@router.post("/gather/greeting", response_class=HTMLResponse, include_in_schema=False)
async def gather_greeting_webhook(
    request: Request,
    agentId: Optional[str] = Query(None),
    userId: Optional[str] = Query(None),
    callSessionId: Optional[str] = Query(None),
    body: str = Depends(get_request_body),
    db: Session = Depends(get_db)
):
    """
    Initial greeting webhook - returns TwiML with <Say> + <Gather input="speech">
    
    This is called when the call first connects (in-progress status).
    """
    print("=" * 80)
    print(f"🎤 GATHER GREETING WEBHOOK - Low Latency Flow")
    print(f"📞 Call Session: {callSessionId}")
    print(f"🤖 Agent: {agentId}")
    print(f"⏰ Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)
    sys.stdout.flush()
    
    try:
        # Parse form data
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "")
        call_status = form_data.get("CallStatus", "")
        
        print(f"📊 Call Status: {call_status}")
        print(f"📞 Call SID: {call_sid}")
        sys.stdout.flush()
        
        # Get call session and agent
        call_session = None
        agent = None
        agent_name = "AI Assistant"
        
        if callSessionId:
            try:
                session_uuid = uuid.UUID(callSessionId)
                call_session = call_session_service.get_call_session_by_id(db, session_uuid)
                
                if call_session and agentId:
                    agent = agent_service.get_agent_by_id(db, uuid.UUID(agentId), call_session.tenant_id)
                    if agent:
                        agent_name = agent.name
                        print(f"✅ Agent: {agent_name}")
                    sys.stdout.flush()
            except ValueError:
                print(f"⚠️ Invalid call session ID: {callSessionId}")
                sys.stdout.flush()
        
        # Create TwiML response
        response = VoiceResponse()
        agent_voice = get_agent_voice(agent)
        gather_language = get_gather_language(agent)
        
        # NO GREETING - User speaks first! 🎤
        # Just start listening immediately for better UX
        print(f"👂 Listening for user to speak first (no greeting)")
        sys.stdout.flush()
        
        # Log call start event (no greeting transcript)
        if call_session:
            try:
                asyncio.create_task(broadcast_call_event(
                    call_session_id=str(call_session.id),
                    event_type="call_started",
                    event_data={
                        "agent_name": agent_name,
                        "message": "Call connected - Listening for user input",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                ))
            except Exception as e:
                print(f"⚠️ Broadcast failed (non-critical): {e}")
        
        # Build callback URL for speech input
        callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/gather/speech-callback?agentId={agentId}&userId={userId}&callSessionId={callSessionId}"
        
        # Gather speech input with optimized settings for low latency
        # User speaks FIRST - no greeting
        gather = response.gather(
            input='speech',
            action=callback_url,
            method='POST',
            speechTimeout=0.8,  # VAPI-STYLE: Fast silence detection (800ms vs 1.5s)
            timeout=5,  # Quick timeout for responsive UX
            language=gather_language,
            enhanced=True,  # Use enhanced model for better accuracy
            profanity_filter=False,  # Don't filter for natural conversation
            speech_model='phone_call'  # Optimized for phone calls
        )
        
        # Timeout fallback (if user doesn't speak)
        text = "Hello? Are you there? Please speak so I can help you."
        lang = agent.language if agent and agent.language else "en"
        voice = agent.voice_type if agent and agent.voice_type else "female"
        tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}&gemini_flash=true"
        response.play(tts_url)
        
        # Give one more chance to speak
        gather_retry = response.gather(
            input='speech',
            action=callback_url,
            method='POST',
            speechTimeout=0.8,  # Fast detection on retry
            timeout=5,  # Shorter timeout for retry
            language=gather_language,
            enhanced=True,
            profanity_filter=False,
            speech_model='phone_call'
        )
        
        text = "I still can't hear you. Please call back. Goodbye!"
        lang = agent.language if agent and agent.language else "en"
        voice = agent.voice_type if agent and agent.voice_type else "female"
        tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}&gemini_flash=true"
        response.play(tts_url)
        response.hangup()
        
        print(f"✅ TwiML generated - User speaks FIRST (no greeting)")
        print(f"📝 TwiML: {str(response)[:300]}...")
        sys.stdout.flush()
        
        return HTMLResponse(str(response), media_type="application/xml")
    
    except Exception as e:
        print(f"❌ Error in greeting webhook: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        
        # Fallback response
        response = VoiceResponse()
        text = "Sorry, something went wrong. Please call back later. Goodbye!"
        tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang=en&voice=female"
        response.play(tts_url)
        response.hangup()
        return HTMLResponse(str(response), media_type="application/xml")


@router.post("/gather/speech-callback", response_class=HTMLResponse, include_in_schema=False)
async def gather_speech_callback_webhook(
    request: Request,
    agentId: Optional[str] = Query(None),
    userId: Optional[str] = Query(None),
    callSessionId: Optional[str] = Query(None),
    body: str = Depends(get_request_body),
    db: Session = Depends(get_db)
):
    """
    Speech callback webhook - receives Gather speech input
    
    Flow:
    1. Receive speech input from Twilio Gather
    2. Download audio recording
    3. Convert to format for Google STT (LINEAR16, 8000Hz)
    4. Transcribe with Google Cloud STT
    5. Pass transcript to LLM (Gemini/GPT)
    6. Generate AI response text
    7. Return TwiML with <Say> + <Gather> to continue conversation
    
    Target: 3-4 seconds total latency
    """
    print("=" * 80)
    print(f"🎙️ GATHER SPEECH CALLBACK - Processing User Input")
    print(f"📞 Call Session: {callSessionId}")
    print(f"🤖 Agent: {agentId}")
    print(f"⏰ Processing Start: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)
    sys.stdout.flush()
    
    processing_start_time = datetime.now(timezone.utc)
    
    try:
        # Parse form data from Twilio
        form_data = await request.form()
        
        call_sid = form_data.get("CallSid", "")
        speech_result = form_data.get("SpeechResult", "")  # Twilio's transcript (backup)
        confidence = form_data.get("Confidence", "0")
        recording_url = form_data.get("RecordingUrl", "")  # Audio recording URL
        
        print(f"📊 Twilio Speech Result: '{speech_result}'")
        print(f"📊 Twilio Confidence: {confidence}")
        print(f"🎵 Recording URL: {recording_url}")
        sys.stdout.flush()
        
        # Get call session and agent
        call_session = None
        agent = None
        agent_name = "AI Assistant"
        
        if callSessionId:
            try:
                session_uuid = uuid.UUID(callSessionId)
                call_session = call_session_service.get_call_session_by_id(db, session_uuid)
                
                if call_session and agentId:
                    agent = agent_service.get_agent_by_id(db, uuid.UUID(agentId), call_session.tenant_id)
                    if agent:
                        agent_name = agent.name
                        print(f"✅ Agent: {agent_name}")
                    sys.stdout.flush()
            except ValueError:
                print(f"⚠️ Invalid call session ID: {callSessionId}")
                sys.stdout.flush()
        
        # Get agent voice and language
        agent_voice = get_agent_voice(agent)
        gather_language = get_gather_language(agent)
        
        # Get real-time call duration
        call_duration = get_call_duration_realtime(call_session) if call_session else "00:00"
        print(f"⏱️ Real-time Call Duration: {call_duration}")
        sys.stdout.flush()
        
        # STEP 2: Download audio from Twilio (if available)
        transcript = ""
        stt_confidence = 0.0
        
        if recording_url:
            try:
                download_start = datetime.now(timezone.utc)
                
                # Get Twilio credentials for authenticated download
                client = twilio_service.get_client()
                account_sid = client.username
                auth_token = client.password
                
                # Build authenticated URL
                if not recording_url.startswith('http'):
                    auth_url = f"https://{account_sid}:{auth_token}@api.twilio.com{recording_url}.wav"
                else:
                    auth_url = recording_url.replace('https://api.twilio.com', f'https://{account_sid}:{auth_token}@api.twilio.com') + '.wav'
                
                print(f"📥 Downloading audio from Twilio...")
                sys.stdout.flush()
                
                # Download audio with reduced timeout for faster response
                audio_response = requests.get(auth_url, timeout=3)  # Reduced from 5s to 3s
                
                if audio_response.status_code == 200:
                    audio_content = audio_response.content
                    download_time = (datetime.now(timezone.utc) - download_start).total_seconds()
                    print(f"✅ Downloaded {len(audio_content)} bytes in {download_time:.2f}s")
                    sys.stdout.flush()
                    
                    # STEP 3 & 4: Convert and transcribe with Google STT
                    stt_start = datetime.now(timezone.utc)
                    
                    # Get language from agent
                    stt_language_code = "en-US"
                    if agent and hasattr(agent, 'language'):
                        language_map = {
                            "en": "en-US",
                            "es": "es-ES",
                            "hi": "hi-IN",
                            "ar": "ar-SA",
                            "zh": "zh-CN",
                            "ur": "ur-PK"
                        }
                        stt_language_code = language_map.get(agent.language, "en-US")
                    
                    print(f"🎙️ Transcribing with Google Cloud STT (language: {stt_language_code})...")
                    sys.stdout.flush()
                    
                    # Transcribe with Google STT
                    stt_result = google_stt_service.transcribe_audio_chunk_streaming(
                        audio_content=audio_content,
                        language_code=stt_language_code
                    )
                    
                    transcript = stt_result.get("transcript", "").strip()
                    stt_confidence = stt_result.get("confidence", 0.0)
                    stt_time = (datetime.now(timezone.utc) - stt_start).total_seconds()
                    
                    print(f"✅ Google STT: '{transcript}' (confidence: {stt_confidence:.2f}, time: {stt_time:.2f}s)")
                    sys.stdout.flush()
                else:
                    print(f"⚠️ Failed to download audio: HTTP {audio_response.status_code}")
                    sys.stdout.flush()
            
            except Exception as e:
                print(f"⚠️ Error processing audio: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
        
        # Fallback to Twilio's transcript if Google STT failed
        if not transcript and speech_result:
            transcript = speech_result
            stt_confidence = float(confidence)
            print(f"ℹ️ Using Twilio transcript as fallback: '{transcript}'")
            sys.stdout.flush()
        
        # Check if we have a valid transcript
        if not transcript:
            print(f"⚠️ No transcript available")
            sys.stdout.flush()
            
            # Ask user to repeat
            response = VoiceResponse()
            text = "Sorry, I didn't catch that. Could you repeat?"
            lang = agent.language if agent and agent.language else "en"
            voice = agent.voice_type if agent and agent.voice_type else "female"
            
            # Pre-generate TTS for instant playback
            pre_generate_tts(text, lang, voice)
            
            tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}&gemini_flash=true"
            response.play(tts_url)
            
            # Gather again
            callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/gather/speech-callback?agentId={agentId}&userId={userId}&callSessionId={callSessionId}"
            
            gather = response.gather(
                input='speech',
                action=callback_url,
                method='POST',
                speechTimeout=0.8,  # Fast silence detection
                timeout=5,  # Quick timeout
                language=gather_language,
                enhanced=True,
                profanity_filter=False,
                speech_model='phone_call'
            )
            
            text = "I'm still not hearing you. Please call back if you need help. Goodbye!"
            lang = agent.language if agent and agent.language else "en"
            voice = agent.voice_type if agent and agent.voice_type else "female"
            tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}&gemini_flash=true"
            response.play(tts_url)
            response.hangup()
            
            return HTMLResponse(str(response), media_type="application/xml")
        
        # STEP 5: Add user speech to transcript (non-blocking - fire and forget)
        if call_session:
            # Fire DB writes in background - don't wait for them
            asyncio.create_task(add_to_transcript(
                call_session,
                "client",
                transcript,
                db,
                message_type="speech",
                confidence=stt_confidence
            ))
            
            # Log voice interaction (non-blocking)
            asyncio.create_task(VoiceLoggingService.log_voice_interaction(
                db=db,
                call_session_id=call_session.id,
                interaction_type="speech_input",
                speech_text=transcript,
                confidence=stt_confidence,
                metadata={
                    "call_sid": call_sid,
                    "agent_id": str(agent.id) if agent else None,
                    "source": "google_stt" if recording_url else "twilio"
                }
            ))
        
        # STEP 6: Generate AI response using LLM (start immediately - don't wait for DB)
        llm_start = datetime.now(timezone.utc)
        
        print(f"🤖 Generating AI response...")
        sys.stdout.flush()
        
        # Use the voice logging service to generate response (handles Gemini/OpenAI)
        response_text = await VoiceLoggingService.generate_agent_response(
            speech_text=transcript,
            confidence=stt_confidence,
            agent=agent,
            db=db,
            call_session_id=call_session.id if call_session else None
        )
        
        llm_time = (datetime.now(timezone.utc) - llm_start).total_seconds()
        
        print(f"✅ AI Response: '{response_text}' (time: {llm_time:.2f}s)")
        sys.stdout.flush()
        
        # Add agent response to transcript (non-blocking - fire and forget)
        if call_session:
            asyncio.create_task(add_to_transcript(
                call_session,
                "agent",
                response_text,
                db,
                message_type="agent_response"
            ))
        
        # Calculate total processing latency
        processing_time = (datetime.now(timezone.utc) - processing_start_time).total_seconds()
        
        # Get updated real-time call duration
        call_duration_end = get_call_duration_realtime(call_session) if call_session else "00:00"
        
        print(f"⏱️ Processing Latency: {processing_time:.2f}s")
        print(f"📞 Call Duration (Real-time): {call_duration_end}")
        sys.stdout.flush()
        
        # Broadcast real-time duration update
        if call_session:
            try:
                asyncio.create_task(broadcast_call_event(
                    call_session_id=str(call_session.id),
                    event_type="duration_update",
                    event_data={
                        "duration": call_duration_end,
                        "processing_time": f"{processing_time:.2f}s",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                ))
            except Exception as e:
                print(f"⚠️ Duration broadcast failed (non-critical): {e}")
        
        # STEP 7: Pre-generate TTS audio (OPTIMIZATION - eliminates 1s delay)
        lang = agent.language if agent and agent.language else "en"
        voice = agent.voice_type if agent and agent.voice_type else "female"
        
        tts_start = datetime.now(timezone.utc)
        try:
            # Check if already cached
            cache_key = generate_cache_key(response_text, lang, voice)
            
            if cache_key not in audio_cache:
                # Pre-generate audio BEFORE sending TwiML
                print(f"⚡ Pre-generating TTS audio: '{response_text[:50]}...'")
                sys.stdout.flush()
                
                audio_content = google_tts_service.text_to_speech(
                    text=response_text,
                    language=lang,
                    voice_type=voice,
                    speaking_rate=1.1,  # 10% faster for quicker responses (sounds natural)
                    pitch=0.0,
                    output_format="mp3"
                )
                
                # Cache it for instant playback
                audio_cache[cache_key] = audio_content
                
                tts_time = (datetime.now(timezone.utc) - tts_start).total_seconds()
                print(f"✅ TTS pre-generated: {len(audio_content)} bytes in {tts_time:.2f}s (cached)")
                sys.stdout.flush()
            else:
                print(f"⚡ TTS already cached: '{response_text[:50]}...'")
                sys.stdout.flush()
                
        except Exception as e:
            print(f"⚠️ TTS pre-generation failed (will generate on-demand): {e}")
            sys.stdout.flush()
        
        # STEP 8: Create TwiML response with Google TTS + <Gather>
        response = VoiceResponse()
        
        # Say agent's response using Google TTS (now instant from cache)
        tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(response_text)}&lang={lang}&voice={voice}&gemini_flash=true"
        response.play(tts_url)
        
        # Check if this is a goodbye
        is_goodbye = VoiceLoggingService._is_completion_goodbye(response_text)
        if is_goodbye or "goodbye" in response_text.lower() or "bye" in response_text.lower():
            print(f"👋 Goodbye detected - ending call")
            sys.stdout.flush()
            response.hangup()
            return HTMLResponse(str(response), media_type="application/xml")
        
        # Continue conversation - gather next input with optimized timeout
        callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/gather/speech-callback?agentId={agentId}&userId={userId}&callSessionId={callSessionId}"
        
        gather = response.gather(
            input='speech',
            action=callback_url,
            method='POST',
            speechTimeout=0.8,  # VAPI-STYLE: 800ms silence detection for speed
            timeout=5,  # Quick timeout for responsive UX
            language=gather_language,
            enhanced=True,
            profanity_filter=False,
            speech_model='phone_call'
        )
        
        # Timeout fallback
        text = "Thank you for calling. Have a great day!"
        lang = agent.language if agent and agent.language else "en"
        voice = agent.voice_type if agent and agent.voice_type else "female"
        tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}&gemini_flash=true"
        response.play(tts_url)
        response.hangup()
        
        print(f"🔄 Continuing conversation - waiting for next user input")
        print(f"✅ Response TwiML generated")
        sys.stdout.flush()
        
        return HTMLResponse(str(response), media_type="application/xml")
    
    except Exception as e:
        print(f"❌ Error in speech callback webhook: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        
        # Fallback response
        response = VoiceResponse()
        text = "Sorry, I had trouble processing that. Could you try again?"
        tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang=en&voice=female"
        response.play(tts_url)
        
        # Try to gather again
        try:
            callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/gather/speech-callback?agentId={agentId}&userId={userId}&callSessionId={callSessionId}"
            
            gather = response.gather(
                input='speech',
                action=callback_url,
                method='POST',
                speechTimeout=0.8,  # Fast detection for error recovery
                timeout=5,  # Quick timeout
                language='en-US',
                enhanced=True,
                profanity_filter=False,
                speech_model='phone_call'
            )
            
            text = "If you're still having trouble, please call back. Goodbye!"
            tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang=en&voice=female"
            response.play(tts_url)
            response.hangup()
        except:
            text = "Please call back later. Goodbye!"
            tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang=en&voice=female"
            response.play(tts_url)
            response.hangup()
        
        return HTMLResponse(str(response), media_type="application/xml")


# Health check endpoint
@router.get("/gather/health")
async def health_check():
    """Health check for gather-based conversation flow"""
    return {
        "status": "healthy",
        "service": "voice-gather-conversation",
        "description": "Fast conversational AI using Twilio Gather + Google STT + LLM",
        "target_latency": "3-4 seconds per turn",
        "endpoints": {
            "greeting": "/api/v1/voice/gather/greeting",
            "speech_callback": "/api/v1/voice/gather/speech-callback"
        }
    }

