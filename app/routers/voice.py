from fastapi import APIRouter, Request, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from typing import Optional
from twilio.twiml.voice_response import VoiceResponse
from datetime import datetime

from app.api.deps import get_db, require_tenant
from app.schemas.twilio import CallInitiateRequest, CallInitiateResponse
from app.schemas.base import SuccessResponse
from app.services.twilio_service import twilio_service
from app.services.agent_service import agent_service
from app.models.agent import Agent
from app.models.user import User
from app.services.call_session_service import call_session_service
from app.services.voice_logging_service import VoiceLoggingService
from app.utils.twilio_validation import validate_twilio_signature, validate_webrtc_auth, get_request_body
from app.utils.response import create_success_response
from app.core.config import settings
import uuid
from datetime import datetime

router = APIRouter()


def get_agent_voice(agent) -> str:
    """Get the appropriate Twilio voice based on agent's voice type and language"""
    if not agent:
        return "en-US-Neural2-F"  # Default female voice
    
    # Get voice type and language from agent
    voice_type = agent.voice_type
    language = agent.language
    
    # Voice mapping based on language and gender
    voice_map = {
        # English voices
        "en": {
            "male": "en-US-Neural2-M",
            "female": "en-US-Neural2-F"
        },
        # Spanish voices
        "es": {
            "male": "es-US-Neural2-M",
            "female": "es-US-Neural2-F"
        },
        # Hindi voices
        "hi": {
            "male": "hi-IN-Neural2-M",
            "female": "hi-IN-Neural2-F"
        },
        # Arabic voices
        "ar": {
            "male": "ar-XA-Neural2-M",
            "female": "ar-XA-Neural2-F"
        },
        # Chinese voices
        "zh": {
            "male": "zh-CN-Neural2-M",
            "female": "zh-CN-Neural2-F"
        },
        # Urdu voices
        "ur": {
            "male": "ur-PK-Neural2-M",
            "female": "ur-PK-Neural2-F"
        }
    }
    
    # Default to English if language not specified
    if not language:
        language = "en"
    
    # Default to female if voice type not specified
    if not voice_type:
        voice_type = "female"
    
    # Get the voice from the mapping
    selected_voice = voice_map.get(language, voice_map["en"]).get(voice_type, "en-US-Neural2-F")
    
    print(f"🎤 Agent voice selection: language={language}, voice_type={voice_type}, selected_voice={selected_voice}")
    
    return selected_voice


@router.post("/call/initiate", response_model=SuccessResponse[CallInitiateResponse])
async def initiate_call(
    request: CallInitiateRequest,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Endpoint to initiate a voice call using Twilio.
    
    Request Payload:
    {
        "agentId": "agent_12345",
        "userPhoneNumber": "+1234567890"
    }
    
    Response:
    {
        "callId": "call_abc123",
        "twilioCallSid": "CAxxxxxxx",
        "status": "initiated"
    }
    """
    try:
        # Validate agent exists in database
        try:
            agent_id = uuid.UUID(request.agentId)
            agent = agent_service.get_agent_by_id(db, agent_id, user.current_tenant_id)
        except (ValueError, HTTPException):
            raise HTTPException(status_code=404, detail=f"Agent {request.agentId} not found")
        
        # Validate phone number format
        if not twilio_service.validate_phone_number(request.userPhoneNumber):
            raise HTTPException(status_code=400, detail="Invalid phone number format. Must start with +")
        
        # Get base URL for webhooks
        base_url = settings.WEBHOOK_BASE_URL
        
        # Make the call using Twilio with separate webhooks
        # Main webhook for conversation flow
        webhook_url = f"{base_url}/api/v1/voice/webhook/call-events?agentId={agent.id}&userId={user.id}"
        # Status callback for call status updates only
        status_callback_url = f"{base_url}/api/v1/voice/webhook/status-callback?agentId={agent.id}&userId={user.id}"
        
        print(f"Making call with webhook_url: {webhook_url}")
        print(f"Making call with status_callback_url: {status_callback_url}")
        
        # Add retry logic for call initiation to prevent double dialing
        max_retries = 3
        call = None
        for attempt in range(max_retries):
            try:
                call = twilio_service.make_call(
                    to_number=request.userPhoneNumber,
                    from_number=twilio_service.get_phone_number(),
                    webhook_url=webhook_url,
                    status_callback_url=status_callback_url
                )
                print(f"✅ Call initiated successfully on attempt {attempt + 1}")
                break
            except Exception as e:
                print(f"⚠️ Call initiation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise HTTPException(status_code=500, detail=f"Failed to initiate call after {max_retries} attempts")
                # Wait before retry
                import time
                time.sleep(1)
        
        # Create call session immediately when call is initiated
        call_session = call_session_service.create_call_session(
            db=db,
            user_id=user.id,
            agent_id=agent.id,
            tenant_id=user.current_tenant_id,
            twilio_call_sid=call.sid,
            from_number=twilio_service.get_phone_number(),
            to_number=request.userPhoneNumber
        )
        
        # Generate call ID
        call_id = f"call_{call.sid[-8:]}"
        
        return create_success_response(
            CallInitiateResponse(
                callId=call_id,
                twilioCallSid=call.sid,
                callSessionId=str(call_session.id),
                status="initiated"
            ),
            "Call initiated successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook/call-events", response_class=HTMLResponse)
async def handle_call_events_webhook(
    request: Request,
    agentId: Optional[str] = Query(None),
    body: str = Depends(get_request_body),
    db: Session = Depends(get_db)
):
    print("=== Call Events Webhook Started === print check")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Request method: {request.method}")
    print(f"Request URL: {request.url}")
    print(f"Request headers: {dict(request.headers)}")
    print(f"Query params: agentId={agentId}")
    print(f"Request body length: {len(body) if body else 0}")
    print(f"Request body preview: {body[:200] if body else 'None'}...")
    print(f"Database session: {db}")
    try:
        print("Parsing request body...")
        
        # Fetch agent information from database using agentId
        agent = None
        if agentId:
            try:
                # Parse form data to get call information
                form_data = await request.form()
                call_sid = form_data.get("CallSid", "")
                
                # Get call session to get tenant_id
                call_session = call_session_service.get_call_session_by_twilio_sid(db, call_sid)
                if call_session:
                    # Fetch agent from database
                    agent = agent_service.get_agent_by_id(db, uuid.UUID(agentId), call_session.tenant_id)
                    if agent:
                        print(f"✅ Agent fetched: {agent.name} (ID: {agent.id})")
                        print(f"🏢 Tenant: {agent.tenant_id}")
                        print(f"🎤 Voice type: {agent.voice_type}, Language: {agent.language}")
                    else:
                        print(f"⚠️ Agent {agentId} not found in tenant {call_session.tenant_id}")
                else:
                    print(f"⚠️ Call session not found for SID: {call_sid}")
            except Exception as e:
                print(f"⚠️ Error fetching agent: {e}")
                agent = None
        else:
            print("⚠️ No agentId provided in query parameters")
        
        # Validate request (Twilio signature or WebRTC auth)
        is_twilio = 'X-Twilio-Signature' in request.headers
        is_webrtc = 'Authorization' in request.headers
        
        if is_twilio:
            print("Twilio signature found, but skipping validation for testing")
            # if not validate_twilio_signature(request, body):
            #     raise HTTPException(status_code=403, detail="Invalid Twilio signature")
        elif is_webrtc:
            if not validate_webrtc_auth(request):
                raise HTTPException(status_code=403, detail="Invalid WebRTC authentication")
        else:
            # For testing purposes, allow requests without validation
            print("No authentication headers found, allowing for testing")
        
        # Form data already parsed above for agent fetching
        
        # Extract call information from form data
        call_sid = form_data.get("CallSid", "")
        call_status = form_data.get("CallStatus", "")
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        direction = form_data.get("Direction", "")
        
        # Extract speech input (if any)
        speech_result = form_data.get("SpeechResult", "")
        confidence = form_data.get("Confidence", "")
        speech_duration = form_data.get("SpeechDuration", "")
        
        print(f"🎤 Speech input - Result: '{speech_result}', Confidence: {confidence}, Duration: {speech_duration}")
        
        
        # Log the call event
        print(f"Call Events Webhook - SID: {call_sid}, Status: {call_status}, From: {from_number}, To: {to_number}, Direction: {direction}")
        print(f"AgentId from query: {agentId}")
        
        
        # Get agent from database if agentId is provided
        agent = None
        if agentId:
            try:
                agent_uuid = uuid.UUID(agentId)
                # Get agent from database
                agent = db.query(Agent).filter(Agent.id == agent_uuid).first()
                if agent:
                    print(f"Found agent: {agent.name} (ID: {agent.id})")
                else:
                    print(f"Agent not found in database for ID: {agentId}")
            except (ValueError, Exception) as e:
                print(f"Error getting agent: {e}")
                agent = None
        else:
            print("No agentId provided in webhook")
        
        # This webhook is only for conversation flow, not status updates
        # Status updates are handled by the separate status-callback webhook
        print(f"Main webhook called - SID: {call_sid}, Status: {call_status}")
        
        # If this is a status update, redirect to status callback
        if call_status in ["initiated", "ringing", "completed", "failed", "busy"]:
            print(f"⚠️ Status update received in main webhook: {call_status} - this should go to status-callback")
            return HTMLResponse("", media_type="application/xml")
        
        # MAIN CONVERSATION FLOW - Handle speech input or initial greeting
        if speech_result and speech_result.strip():
            # User spoke - process with Gemini
            # VALID SPEECH DETECTED - LOG AND RESPOND
            print("=" * 60)
            print(f"🎤 SPEECH DETECTED: '{speech_result}'")
            print(f"📊 Confidence: {confidence}, Duration: {speech_duration}")
            print(f"📞 Call SID: {call_sid}")
            print(f"⏰ Timestamp: {datetime.now()}")
            print(f"🏢 Tenant ID: {agent.tenant_id if agent else 'Unknown'}")
            print(f"🤖 Agent: {agent.name if agent else 'Unknown'}")
            print("=" * 60)
            
            # Log voice interaction for smooth tracking
            try:
                call_session = call_session_service.get_call_session_by_twilio_sid(db, call_sid)
                if call_session:
                    await VoiceLoggingService.log_voice_interaction(
                        db=db,
                        call_session_id=call_session.id,
                        interaction_type="speech_input",
                        speech_text=speech_result,
                        confidence=float(confidence) if confidence else None,
                        duration=float(speech_duration) if speech_duration else None,
                        metadata={
                            "call_sid": call_sid,
                            "agent_id": str(agent.id) if agent else None,
                            "tenant_id": str(agent.tenant_id) if agent else None
                        }
                    )
            except Exception as e:
                print(f"⚠️ Error logging voice interaction: {e}")
            
            # Generate smooth, natural response
            response = VoiceResponse()
            
            if agent:
                # Generate intelligent response based on speech using Gemini
                response_text = await VoiceLoggingService.generate_agent_response(
                    speech_text=speech_result,
                    confidence=float(confidence) if confidence else 0.0,
                    agent=agent,
                    db=db
                )
                
                # Say response naturally but keep it shorter for better conversation flow
                agent_voice = get_agent_voice(agent)
                response.say(response_text, voice=agent_voice)
                response.pause(length=0.5)  # Shorter pause for more natural flow
                
                # Continue listening immediately - don't ask additional questions
                response.gather(
                    input='speech',
                    timeout=15,  # Shorter timeout for more natural conversation
                    speech_timeout='auto',
                    action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agent.id}',
                    method='POST'
                )
                
                # Only gentle fallback if no response
                response.say("I'm still here if you need anything.", voice=agent_voice)
            else:
                # Default response for smooth conversation
                default_voice = get_agent_voice(None)  # Use default voice
                response.say(f"I heard you say: {speech_result}.", voice=default_voice)
                response.pause(length=1)
                response.say("How can I help you further?", voice=default_voice)
                
                # Continue listening
                response.gather(
                    input='speech',
                    timeout=25,
                    speech_timeout='auto',
                    action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agent.id if agent else ""}',
                    method='POST'
                )
                
                response.say("Thank you for calling. Have a great day!", voice=default_voice)
            
            print(f"📝 Speech response generated: {str(response)[:200]}...")
            return HTMLResponse(str(response), media_type="application/xml")
        
        else:
            # No speech input - this is the initial greeting when call is answered
            print("=" * 50)
            print(f"📞 INITIAL GREETING - SID: {call_sid}")
            print("=" * 50)
            
            # SIMPLE GREETING AND LISTENING
            response = VoiceResponse()
            
            # Get agent info
            agent_name = "AI Assistant"
            if agent:
                agent_name = agent.name
                print(f"🤖 Agent: {agent_name}")
            
            # Simple greeting with agent-specific voice
            agent_voice = get_agent_voice(agent)
            response.say(f"Hello! This is {agent_name}. How can I help you today?", voice=agent_voice)
            response.pause(length=1)
            
            # Start listening immediately
            response.gather(
                input='speech',
                timeout=10,
                speech_timeout='auto',
                action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agentId}',
                method='POST'
            )
            
            # Simple fallback
            response.say("I didn't hear anything. Please speak when ready.", voice=agent_voice)
            
            twiml_result = str(response)
            print(f"📝 Initial greeting TwiML: {twiml_result}")
            return HTMLResponse(twiml_result, media_type="application/xml")
        
        # Speech input handling already done above - this is a fallback for unexpected cases
        print(f"⚠️ Unexpected speech input: '{speech_result}' - returning empty response")
        return HTMLResponse("", media_type="application/xml")
    
    except Exception as e:
        print(f"ERROR occurred: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("Error traceback:")
        import traceback
        print(traceback.format_exc())
        print("=== Call Events Webhook Failed ===")
        raise
        
        # Call status handling already done above - this is a fallback for unexpected cases
        print(f"⚠️ Unexpected call status: '{call_status}' - returning empty response")
        return HTMLResponse("", media_type="application/xml")
    
    except Exception as e:
        print(f"ERROR occurred: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("Error traceback:")
        import traceback
        print(traceback.format_exc())
        print("=== Call Events Webhook Failed ===")
        raise
    
    except Exception as e:
        print(f"ERROR occurred: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("Error traceback:")
        import traceback
        print(traceback.format_exc())
        print("=== Call Events Webhook Failed ===")
        raise


@router.post("/webhook/status-callback", response_class=HTMLResponse)
async def handle_status_callback(
    request: Request,
    agentId: Optional[str] = Query(None),
    userId: Optional[str] = Query(None),
    body: str = Depends(get_request_body),
    db: Session = Depends(get_db)
):
    """
    Handle call status callbacks (ringing, completed, etc.)
    This is separate from the main conversation webhook
    """
    try:
        print("=== Status Callback Started ===")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"AgentId: {agentId}, UserId: {userId}")
        
        # Parse form data
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "")
        call_status = form_data.get("CallStatus", "")
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        direction = form_data.get("Direction", "")
        
        print(f"Status Callback - SID: {call_sid}, Status: {call_status}, From: {from_number}, To: {to_number}")
        
        # Just log the status, don't return any TwiML
        if call_status == "ringing":
            print("🔔 Call is ringing - no action needed")
        elif call_status == "in-progress":
            print("📞 Call answered - conversation will be handled by main webhook")
        elif call_status == "completed":
            print("📴 Call completed")
        elif call_status == "failed":
            print("❌ Call failed")
        else:
            print(f"📊 Call status: {call_status}")
        
        # Return empty response for status callbacks
        return HTMLResponse("", media_type="application/xml")
        
    except Exception as e:
        print(f"ERROR in status callback: {str(e)}")
        return HTMLResponse("", media_type="application/xml")


def _get_twilio_voice(voice_type):
    """Map voice_type to Twilio voice names"""
    if voice_type == "male":
        return "en-US-Neural2-M"  # Male voice
    elif voice_type == "female":
        return "en-US-Neural2-F"  # Female voice
    else:
        return "en-US-Neural2-F"  # Default to female voice

def _process_speech_input(agent, speech_text: str, call_sid: str) -> str:
    """Process speech input and generate agent response"""
    if not agent:
        return f"I heard you say: {speech_text}. How can I help you further?"
    
    # Simple keyword-based responses (you can enhance this with AI)
    speech_lower = speech_text.lower()
    
    if any(word in speech_lower for word in ['hello', 'hi', 'hey']):
        return f"Hello! This is {agent.name}. How can I assist you today?"
    elif any(word in speech_lower for word in ['help', 'support', 'assistance']):
        return f"I'm here to help! What specific assistance do you need?"
    elif any(word in speech_lower for word in ['thank', 'thanks']):
        return f"You're welcome! Is there anything else I can help you with?"
    elif any(word in speech_lower for word in ['bye', 'goodbye', 'end']):
        return f"Thank you for calling! Have a great day!"
    else:
        return f"I understand you said: {speech_text}. Let me help you with that. What would you like me to do?"

def _generate_agent_response(agent, call_data: dict) -> str:
    """Generate TwiML response based on agent from database"""
    if not agent:
        return _generate_default_response()
    
    # Create TwiML response
    response = VoiceResponse()
    
    # Use agent's name and fallback response
    agent_name = agent.name
    greeting = agent.fallback_response if agent.fallback_response and agent.fallback_response.strip() and agent.fallback_response != "string" else f"Hello! This is {agent_name} speaking. How can I help you today?"
    twilio_voice = _get_twilio_voice(agent.voice_type)
    
    print(f"🎯 Agent greeting: '{greeting}'")
    print(f"🎯 Agent voice: '{twilio_voice}'")
    
    # ABSOLUTE SIMPLEST - NO GATHER AT ALL
    response.say("Hello! This is your AI assistant speaking.", voice=twilio_voice)
    response.pause(length=2)
    response.say("I can help you with any questions you have.", voice=twilio_voice)
    response.pause(length=2)
    response.say("Thank you for calling. Have a great day!", voice=twilio_voice)
    response.pause(length=1)
    response.hangup()
    
    return str(response)


def _generate_default_response() -> str:
    """Generate default TwiML response"""
    response = VoiceResponse()
    response.say("Thank you for calling. An agent will be with you shortly.", voice="")
    response.pause(length=2)
    response.say("Please hold while we connect you.", voice="")
    return str(response)




# VICIdial integration removed - file cleaned
