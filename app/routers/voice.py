from fastapi import APIRouter, Request, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from typing import Optional
from twilio.twiml.voice_response import VoiceResponse

from app.api.deps import get_db, require_tenant
from app.schemas.twilio import CallInitiateRequest, CallInitiateResponse
from app.schemas.base import SuccessResponse
from app.services.twilio_service import twilio_service
from app.services.agent_service import agent_service
from app.models.agent import Agent
from app.models.user import User
from app.services.call_session_service import call_session_service
from app.utils.twilio_validation import validate_twilio_signature, validate_webrtc_auth, get_request_body
from app.utils.response import create_success_response
from app.core.config import settings
import uuid
from datetime import datetime
import logging
from app.core.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)

router = APIRouter()


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
        
        # Make the call using Twilio with main call events webhook
        webhook_url = f"{base_url}/api/v1/voice/webhook/call-events?agentId={agent.id}&userId={user.id}"
        status_callback_url = f"{base_url}/api/v1/voice/webhook/call-events?agentId={agent.id}&userId={user.id}"
        
        print(f"Making call with webhook_url: {webhook_url}")
        print(f"Making call with status_callback_url: {status_callback_url}")
        
        call = twilio_service.make_call(
            to_number=request.userPhoneNumber,
            from_number=twilio_service.get_phone_number(),
            webhook_url=webhook_url,
            status_callback_url=status_callback_url
        )
        
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
            logger.warning("No authentication headers found, allowing for testing")
        
        # Parse form data
        form_data = await request.form()
        
        # Extract call information
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
                    logger.warning(f"Agent not found in database for ID: {agentId}")
            except (ValueError, Exception) as e:
                print(f"Error getting agent: {e}")
                agent = None
        else:
            print("No agentId provided in webhook")
        
        # Handle speech input first (if present)
        if speech_result:
            print("=" * 50)
            print(f"🎤 SPEECH DETECTED: '{speech_result}'")
            print(f"📊 Confidence: {confidence}, Duration: {speech_duration}")
            print("=" * 50)
            
            # Process the speech input
            response = VoiceResponse()
            
            if agent:
                # Generate response based on speech input
                agent_response = _process_speech_input(agent, speech_result, call_sid)
                response.say(agent_response, voice=_get_twilio_voice(agent.voice_type))
            else:
                response.say(f"I heard you say: {speech_result}. How can I help you further?", voice="")
            
            # Add another gather to continue listening
            gather = response.gather(
                input='speech',
                timeout=10,
                speech_timeout='auto',
                action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agent.id if agent else ""}',
                method='POST'
            )
            gather.say("Please tell me more about how I can help you.", voice=_get_twilio_voice(agent.voice_type) if agent else "")
            
            # Fallback if no input
            response.say("I didn't catch that. Let me transfer you to a human agent.", voice=_get_twilio_voice(agent.voice_type) if agent else "")
            
            print(f"📝 Generated speech response TwiML: {str(response)[:200]}...")
            return HTMLResponse(str(response), media_type="application/xml")
        
        # Handle different call statuses and trigger agent logic
        print(f"Processing call status: '{call_status}' with direction: '{direction}'")
        
        if call_status == "initiated" and direction == "outbound-api":
            # Call has been initiated - just log and return empty response
            print(f"Call initiated - SID: {call_sid}")
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "ringing" and direction == "outbound-api":
            # Outbound call is ringing - trigger agent logic
            print("=" * 50)
            print(f"🔔 CALL IS RINGING - SID: {call_sid}")
            print("=" * 50)
            if agent:
                print(f"🤖 Generating agent response for agent: {agent.name}")
                # Generate agent-specific response using database agent
                twiml_response = _generate_agent_response(agent, {
                    'call_sid': call_sid,
                    'from_number': from_number,
                    'to_number': to_number,
                    'status': call_status,
                    'event_type': 'call_ringing'
                })
                print(f"📝 Generated TwiML response (first 300 chars):")
                print(twiml_response[:300])
                print("=" * 50)
                print("✅ RETURNING TwiML RESPONSE TO TWILIO")
                print("=" * 50)
                return HTMLResponse(twiml_response, media_type="application/xml")
            else:
                print("❌ No agent found, using default response")
                # Default response
                response = VoiceResponse()
                response.say("Hello! Thank you for answering our call.", voice="")
                response.say("An agent will be with you shortly.", voice="")
                default_twiml = str(response)
                print(f"📝 Default TwiML response: {default_twiml}")
                return HTMLResponse(default_twiml, media_type="application/xml")
        
        elif call_status == "in-progress":
            # Call is in progress - trigger agent logic
            if agent:
                # Generate agent-specific response for active call
                twiml_response = _generate_agent_response(agent, {
                    'call_sid': call_sid,
                    'from_number': from_number,
                    'to_number': to_number,
                    'status': call_status,
                    'event_type': 'call_in_progress'
                })
                return HTMLResponse(twiml_response, media_type="application/xml")
            else:
                response = VoiceResponse()
                response.say("Your call is now connected. How can we help you today?", voice="")
                return HTMLResponse(str(response), media_type="application/xml")
        
        elif call_status == "completed":
            # Call completed
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "failed":
            # Call failed - handle error
            logger.warning(f"Call failed - SID: {call_sid}")
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "busy":
            # Call busy - handle busy signal
            logger.warning(f"Call busy - SID: {call_sid}")
            return HTMLResponse("", media_type="application/xml")
        
        else:
            # Default response for other statuses
            logger.warning(f"Unhandled call status: '{call_status}' - using default response")
            response = VoiceResponse()
            agent_voice = agent.name if agent else ""
            response.say("Thank you for your call.", voice=agent_voice)
            return HTMLResponse(str(response), media_type="application/xml")
    
    except Exception as e:
        print(f"ERROR occurred: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("Error traceback:")
        import traceback
        print(traceback.format_exc())
        print("=== Call Events Webhook Failed ===")
        raise


def _get_twilio_voice(voice_type):
    """Map voice_type to Twilio voice names"""
    if voice_type == "male":
        return "en-US-Neural2-F"  # Male voice
    elif voice_type == "female":
        return "en-US-Neural2-E"  # Female voice
    else:
        return ""  # Default voice

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
    greeting = agent.fallback_response or f"Hello! This is {agent_name} speaking. How can I help you today?"
    twilio_voice = _get_twilio_voice(agent.voice_type)
    
    # Say the greeting with agent's voice
    response.say(greeting, voice=twilio_voice)
    
    # Add gather to collect user input
    gather = response.gather(
        input='speech',
        timeout=10,
        speech_timeout='auto',
        action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agent.id}',
        method='POST'
    )
    gather.say(f"Please tell me how I can assist you.", voice=twilio_voice)
    
    # Fallback if no input
    response.say("I didn't catch that. Let me transfer you to a human agent.", voice=twilio_voice)
    
    return str(response)


def _generate_default_response() -> str:
    """Generate default TwiML response"""
    response = VoiceResponse()
    response.say("Thank you for calling. An agent will be with you shortly.", voice="")
    response.pause(length=2)
    response.say("Please hold while we connect you.", voice="")
    return str(response)
