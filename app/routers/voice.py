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
        base_url = f"http://{settings.HOST}:{settings.PORT}"
        
        # Make the call using Twilio with main call events webhook
        call = twilio_service.make_call(
            to_number=request.userPhoneNumber,
            from_number=twilio_service.get_phone_number(),
            webhook_url=f"{base_url}/api/v1/voice/webhook/call-events?agentId={request.agentId}&userId={user.id}",
            status_callback_url=f"{base_url}/api/v1/voice/webhook/call-events"
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
    logger.info("=== Call Events Webhook Started ===")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request URL: {request.url}")
    logger.info(f"Request headers: {dict(request.headers)}")
    logger.info(f"Query params: agentId={agentId}")
    logger.info(f"Request body length: {len(body) if body else 0}")
    logger.info(f"Request body preview: {body[:200] if body else 'None'}...")
    logger.info(f"Database session: {db}")
    
    try:
        logger.info("Parsing request body...")
        # Validate request (Twilio signature or WebRTC auth)
        is_twilio = 'X-Twilio-Signature' in request.headers
        is_webrtc = 'Authorization' in request.headers
        
        if is_twilio:
            logger.info("Twilio signature found, but skipping validation for testing")
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
        
        
        # Log the call event
        logger.info(f"Call Events Webhook - SID: {call_sid}, Status: {call_status}, From: {from_number}, To: {to_number}, Direction: {direction}")
        logger.info(f"AgentId from query: {agentId}")
        
        # Get agent from database if agentId is provided
        agent = None
        if agentId:
            try:
                agent_uuid = uuid.UUID(agentId)
                # Get agent from database
                agent = db.query(Agent).filter(Agent.id == agent_uuid).first()
                if agent:
                    logger.info(f"Found agent: {agent.name} (ID: {agent.id})")
                else:
                    logger.warning(f"Agent not found in database for ID: {agentId}")
            except (ValueError, Exception) as e:
                logger.error(f"Error getting agent: {e}")
                agent = None
        else:
            logger.info("No agentId provided in webhook")
        
        # Handle different call statuses and trigger agent logic
        logger.info(f"Processing call status: '{call_status}' with direction: '{direction}'")
        
        if call_status == "ringing" and direction == "outbound-api":
            # Outbound call is ringing - trigger agent logic
            if agent:
                # Generate agent-specific response using database agent
                twiml_response = _generate_agent_response(agent, {
                    'call_sid': call_sid,
                    'from_number': from_number,
                    'to_number': to_number,
                    'status': call_status,
                    'event_type': 'call_ringing'
                })
                return HTMLResponse(twiml_response, media_type="application/xml")
            else:
                # Default response
                response = VoiceResponse()
                response.say("Hello! Thank you for answering our call.", voice="")
                response.say("An agent will be with you shortly.", voice="")
                return HTMLResponse(str(response), media_type="application/xml")
        
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
        logger.error(f"ERROR occurred: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error("Error traceback:")
        import traceback
        logger.error(traceback.format_exc())
        logger.error("=== Call Events Webhook Failed ===")
        raise


def _generate_agent_response(agent, call_data: dict) -> str:
    """Generate TwiML response based on agent from database"""
    if not agent:
        return _generate_default_response()
    
    # Create TwiML response
    response = VoiceResponse()
    
    # Map voice_type to Twilio voice names
    def get_twilio_voice(voice_type):
        if voice_type == "male":
            return "en-US-Neural2-F"  # Male voice
        elif voice_type == "female":
            return "en-US-Neural2-E"  # Female voice
        else:
            return ""  # Default voice
    
    # Use agent's name and fallback response
    agent_name = agent.name
    greeting = agent.fallback_response or f"Hello! This is {agent_name} speaking. How can I help you today?"
    twilio_voice = get_twilio_voice(agent.voice_type)
    
    # Say the greeting with agent's voice
    response.say(greeting, voice=twilio_voice)
    
    # Add gather to collect user input
    gather = response.gather(
        input='speech',
        timeout=10,
        speech_timeout='auto',
        action=f'/webhook/call-events?agentId={agent.id}',
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
