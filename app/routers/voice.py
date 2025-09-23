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
from app.utils.twilio_validation import validate_twilio_signature, validate_webrtc_auth, get_request_body
from app.utils.response import create_success_response
from app.core.config import settings
import uuid
from datetime import datetime

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
            print("No authentication headers found, allowing for testing")
        
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
                    print(f"Agent not found in database for ID: {agentId}")
            except (ValueError, Exception) as e:
                print(f"Error getting agent: {e}")
                agent = None
        else:
            print("No agentId provided in webhook")
        
        # Handle speech input - ROBUST MULTI-TENANT LOGGING
        if speech_result and speech_result.strip():
            # VALID SPEECH DETECTED - LOG AND RESPOND
            print("=" * 60)
            print(f"🎤 SPEECH DETECTED: '{speech_result}'")
            print(f"📊 Confidence: {confidence}, Duration: {speech_duration}")
            print(f"📞 Call SID: {call_sid}")
            print(f"⏰ Timestamp: {datetime.now()}")
            print(f"🏢 Tenant ID: {agent.tenant_id if agent else 'Unknown'}")
            print(f"🤖 Agent: {agent.name if agent else 'Unknown'}")
            print("=" * 60)
            
            response = VoiceResponse()
            response.say(f"Thank you! I heard you say: {speech_result}.", voice="en-US-Neural2-F")
            response.pause(length=1)
            response.say("How else can I help you today?", voice="en-US-Neural2-F")
            response.pause(length=2)
            response.say("Please speak again or I will end the call.", voice="en-US-Neural2-F")
            
            # Continue listening
            response.gather(
                input='speech',
                timeout=10,
                speech_timeout='auto',
                action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agent.id if agent else ""}',
                method='POST'
            )
            
            # Fallback
            response.say("Thank you for calling. Have a great day!", voice="en-US-Neural2-F")
            response.pause(length=1)
            response.hangup()
            
            print(f"📝 Speech response generated: {str(response)[:200]}...")
            return HTMLResponse(str(response), media_type="application/xml")
        
        elif speech_result == "" or speech_result is None:
            # NO SPEECH DETECTED - HANDLE GRACEFULLY
            print("=" * 60)
            print(f"🔇 NO SPEECH DETECTED")
            print(f"📞 Call SID: {call_sid}")
            print(f"⏰ Timestamp: {datetime.now()}")
            print(f"🏢 Tenant ID: {agent.tenant_id if agent else 'Unknown'}")
            print("=" * 60)
            
            response = VoiceResponse()
            response.say("I didn't hear anything. Let me try again.", voice="en-US-Neural2-F")
            response.pause(length=1)
            response.say("Please speak clearly into your phone.", voice="en-US-Neural2-F")
            response.pause(length=2)
            response.say("I will wait for your response.", voice="en-US-Neural2-F")
            
            # Try again with longer timeout
            response.gather(
                input='speech',
                timeout=15,
                speech_timeout='auto',
                action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agent.id if agent else ""}',
                method='POST'
            )
            
            # Final fallback
            response.say("I still didn't hear anything. Thank you for calling. Goodbye!", voice="en-US-Neural2-F")
            response.pause(length=1)
            response.hangup()
            
            print(f"📝 No speech response generated: {str(response)[:200]}...")
            return HTMLResponse(str(response), media_type="application/xml")
        
        # Handle different call statuses and trigger agent logic
        print(f"Processing call status: '{call_status}' with direction: '{direction}'")
        
        if call_status == "initiated" and direction == "outbound-api":
            # Call has been initiated - just log and return empty response
            print(f"Call initiated - SID: {call_sid}")
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "ringing" and direction == "outbound-api":
            # Outbound call is ringing - just log, don't play any audio
            print("=" * 50)
            print(f"🔔 CALL IS RINGING - SID: {call_sid}")
            print("=" * 50)
            # Return empty response - no audio should play while ringing
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "in-progress":
            # Call is in progress - person answered, play greeting
            print("=" * 50)
            print(f"📞 CALL ANSWERED - SID: {call_sid}")
            print("=" * 50)
            
            # MULTI-TENANT GREETING WITH ROBUST SPEECH LOGGING
            response = VoiceResponse()
            
            # Get agent info for tenant context
            agent_name = "AI Assistant"
            if agent:
                agent_name = agent.name
                print(f"🏢 Multi-tenant call for tenant: {agent.tenant_id}")
                print(f"🤖 Agent: {agent_name}")
            
            response.say(f"Hello! This is {agent_name} speaking.", voice="en-US-Neural2-F")
            response.pause(length=2)
            response.say("I can help you with any questions you have.", voice="en-US-Neural2-F")
            response.pause(length=2)
            response.say("Please speak clearly and I will respond to you.", voice="en-US-Neural2-F")
            response.pause(length=1)
            response.say("I am listening now.", voice="en-US-Neural2-F")
            
            # Robust gather for speech input
            response.gather(
                input='speech',
                timeout=15,
                speech_timeout='auto',
                action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agentId}',
                method='POST'
            )
            
            # Fallback if no input
            response.say("I didn't hear anything. Let me try one more time.", voice="en-US-Neural2-F")
            response.pause(length=1)
            response.say("Please speak now or I will end the call.", voice="en-US-Neural2-F")
            response.pause(length=1)
            response.hangup()
            
            twiml_result = str(response)
            print(f"📝 BULLETPROOF TwiML: {twiml_result}")
            print("=" * 50)
            print("✅ RETURNING BULLETPROOF TwiML TO TWILIO")
            print("=" * 50)
            return HTMLResponse(twiml_result, media_type="application/xml")
        
        elif call_status == "completed":
            # Call completed
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "failed":
            # Call failed - handle error
            print(f"Call failed - SID: {call_sid}")
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "busy":
            # Call busy - handle busy signal
            print(f"Call busy - SID: {call_sid}")
            return HTMLResponse("", media_type="application/xml")
        
        else:
            # Default response for other statuses
            print(f"Unhandled call status: '{call_status}' - using default response")
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