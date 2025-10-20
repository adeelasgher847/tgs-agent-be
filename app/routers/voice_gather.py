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
from app.services.voice_logging_service import VoiceLoggingService
from app.services.gemini_service import gemini_service
from app.services.openai_service import openai_service
from app.services.model_service import ModelService
from app.core.config import settings
from app.utils.twilio_validation import get_request_body
from app.routers.general_websocket import broadcast_transcript_update, broadcast_call_event

router = APIRouter()
model_service = ModelService()


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
        
        # Greeting message
        greeting_text = f"Hello! This is {agent_name}. How can I help you today?"
        response.say(greeting_text, voice=agent_voice)
        
        # Add greeting to transcript
        if call_session:
            await add_to_transcript(call_session, "agent", greeting_text, db, message_type="greeting")
            
            # Broadcast greeting event
            try:
                asyncio.create_task(broadcast_call_event(
                    call_session_id=str(call_session.id),
                    event_type="greeting",
                    event_data={
                        "agent_name": agent_name,
                        "greeting_text": greeting_text,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                ))
            except Exception as e:
                print(f"⚠️ Broadcast failed (non-critical): {e}")
        
        # Build callback URL for speech input
        callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/gather/speech-callback?agentId={agentId}&userId={userId}&callSessionId={callSessionId}"
        
        # Gather speech input with optimized settings for low latency
        gather = response.gather(
            input='speech',
            action=callback_url,
            method='POST',
            speechTimeout='auto',  # Auto-detect when user stops speaking
            timeout=10,  # Overall timeout (10 seconds of silence)
            language=gather_language,
            enhanced=True,  # Use enhanced model for better accuracy
            profanity_filter=False,  # Don't filter for natural conversation
            speech_model='phone_call'  # Optimized for phone calls
        )
        
        # Timeout fallback
        response.say("I didn't hear anything. Please call back if you need assistance. Goodbye!", voice=agent_voice)
        response.hangup()
        
        print(f"✅ Greeting TwiML generated with <Gather>")
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
        response.say("Sorry, something went wrong. Please call back later. Goodbye!")
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
    print(f"⏰ Start Time: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)
    sys.stdout.flush()
    
    start_time = datetime.now(timezone.utc)
    
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
                
                # Download audio
                audio_response = requests.get(auth_url, timeout=5)
                
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
            response.say("Sorry, I didn't catch that. Could you repeat?", voice=agent_voice)
            
            # Gather again
            callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/gather/speech-callback?agentId={agentId}&userId={userId}&callSessionId={callSessionId}"
            
            gather = response.gather(
                input='speech',
                action=callback_url,
                method='POST',
                speechTimeout='auto',
                timeout=10,
                language=gather_language,
                enhanced=True,
                profanity_filter=False,
                speech_model='phone_call'
            )
            
            response.say("I'm still not hearing you. Please call back if you need help. Goodbye!", voice=agent_voice)
            response.hangup()
            
            return HTMLResponse(str(response), media_type="application/xml")
        
        # STEP 5: Add user speech to transcript
        if call_session:
            await add_to_transcript(
                call_session,
                "client",
                transcript,
                db,
                message_type="speech",
                confidence=stt_confidence
            )
            
            # Log voice interaction
            await VoiceLoggingService.log_voice_interaction(
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
            )
        
        # STEP 6: Generate AI response using LLM
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
        
        # Add agent response to transcript
        if call_session:
            await add_to_transcript(
                call_session,
                "agent",
                response_text,
                db,
                message_type="agent_response"
            )
        
        # Calculate total latency
        total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        print(f"⏱️ Total Latency: {total_time:.2f}s")
        sys.stdout.flush()
        
        # STEP 7: Create TwiML response with <Say> + <Gather>
        response = VoiceResponse()
        
        # Say agent's response
        response.say(response_text, voice=agent_voice)
        
        # Check if this is a goodbye
        is_goodbye = VoiceLoggingService._is_completion_goodbye(response_text)
        if is_goodbye or "goodbye" in response_text.lower() or "bye" in response_text.lower():
            print(f"👋 Goodbye detected - ending call")
            sys.stdout.flush()
            response.hangup()
            return HTMLResponse(str(response), media_type="application/xml")
        
        # Continue conversation - gather next input
        callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/gather/speech-callback?agentId={agentId}&userId={userId}&callSessionId={callSessionId}"
        
        gather = response.gather(
            input='speech',
            action=callback_url,
            method='POST',
            speechTimeout='auto',  # Auto-detect silence
            timeout=10,  # 10 seconds of silence
            language=gather_language,
            enhanced=True,
            profanity_filter=False,
            speech_model='phone_call'
        )
        
        # Timeout fallback
        response.say("Thank you for calling. Have a great day!", voice=agent_voice)
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
        agent_voice = get_agent_voice(None)
        response.say("Sorry, I had trouble processing that. Could you try again?", voice=agent_voice)
        
        # Try to gather again
        try:
            callback_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/voice/gather/speech-callback?agentId={agentId}&userId={userId}&callSessionId={callSessionId}"
            
            gather = response.gather(
                input='speech',
                action=callback_url,
                method='POST',
                speechTimeout='auto',
                timeout=10,
                language='en-US',
                enhanced=True,
                profanity_filter=False,
                speech_model='phone_call'
            )
            
            response.say("If you're still having trouble, please call back. Goodbye!", voice=agent_voice)
            response.hangup()
        except:
            response.say("Please call back later. Goodbye!", voice=agent_voice)
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

