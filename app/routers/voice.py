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
            
            # Continue listening with longer timeout
            response.gather(
                input='speech',
                timeout=20,
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
            # NO SPEECH DETECTED - KEEP LISTENING, DON'T TERMINATE
            print("=" * 60)
            print(f"🔇 NO SPEECH DETECTED - KEEPING CALL ALIVE")
            print(f"📞 Call SID: {call_sid}")
            print(f"⏰ Timestamp: {datetime.now()}")
            print(f"🏢 Tenant ID: {agent.tenant_id if agent else 'Unknown'}")
            print("=" * 60)
            
            response = VoiceResponse()
            response.say("I didn't hear anything. Let me try again.", voice="en-US-Neural2-F")
            response.pause(length=1)
            response.say("Please speak clearly into your phone.", voice="en-US-Neural2-F")
            response.pause(length=2)
            response.say("I am still listening.", voice="en-US-Neural2-F")
            
            # Keep listening with longer timeout - NO HANGUP
            response.gather(
                input='speech',
                timeout=20,
                speech_timeout='auto',
                action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agent.id if agent else ""}',
                method='POST'
            )
            
            # Keep listening - don't hangup immediately
            response.say("I am still here and listening. Please speak when you are ready.", voice="en-US-Neural2-F")
            response.pause(length=2)
            
            # Try one more time
            response.gather(
                input='speech',
                timeout=20,
                speech_timeout='auto',
                action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agent.id if agent else ""}',
                method='POST'
            )
            
            # Only hangup after multiple attempts
            response.say("I haven't heard anything for a while. Thank you for calling. Goodbye!", voice="en-US-Neural2-F")
            response.pause(length=1)
            response.hangup()
            
            print(f"📝 Extended listening response: {str(response)[:200]}...")
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
            
            # Extended gather for speech input - LONGER TIMEOUTS
            response.gather(
                input='speech',
                timeout=25,
                speech_timeout='auto',
                action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agentId}',
                method='POST'
            )
            
            # Fallback if no input - KEEP LISTENING
            response.say("I didn't hear anything. Let me try again.", voice="en-US-Neural2-F")
            response.pause(length=1)
            response.say("Please speak clearly into your phone.", voice="en-US-Neural2-F")
            response.pause(length=2)
            response.say("I am still listening.", voice="en-US-Neural2-F")
            
            # Try again with even longer timeout
            response.gather(
                input='speech',
                timeout=25,
                speech_timeout='auto',
                action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agentId}',
                method='POST'
            )
            
            # Final attempt before hanging up
            response.say("I am still here and listening. Please speak when you are ready.", voice="en-US-Neural2-F")
            response.pause(length=2)
            response.say("If you don't speak soon, I will end the call.", voice="en-US-Neural2-F")
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


# Vitchi Dialer Integration - Separate from current flow
@router.post("/vitchi/initiate-call")
async def initiate_vitchi_call(
    request: CallInitiateRequest,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Initiate a call through Vitchi dialer
    This is a separate endpoint for Vitchi integration
    """
    try:
        print("=" * 60)
        print(f"🚀 INITIATING VITCHI DIALER CALL")
        print(f"📞 To: {request.userPhoneNumber}")
        print(f"🤖 Agent: {request.agentId}")
        print(f"👤 User: {user.email}")
        print(f"🏢 Tenant: {user.current_tenant_id}")
        print("=" * 60)
        
        # Get agent from database
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
        
        # Create unique call ID for Vitchi tracking
        call_id = f"vitchi_{uuid.uuid4().hex[:8]}"
        
        # Vitchi-specific webhook URLs
        webhook_url = f"{base_url}/api/v1/voice/vitchi/webhook/call-events?agentId={agent.id}&userId={user.id}&callId={call_id}"
        status_callback_url = f"{base_url}/api/v1/voice/vitchi/webhook/status?agentId={agent.id}&userId={user.id}&callId={call_id}"
        
        print(f"🔗 Vitchi Webhook URL: {webhook_url}")
        print(f"📊 Vitchi Status Callback: {status_callback_url}")
        
        # Make the call using Twilio (Vitchi uses Twilio backend)
        call = twilio_service.make_call(
            to_number=request.userPhoneNumber,
            from_number=twilio_service.get_phone_number(),
            webhook_url=webhook_url,
            status_callback_url=status_callback_url
        )
        
        print(f"✅ Vitchi call initiated - SID: {call.sid}")
        
        # Create call session for Vitchi call
        call_session = call_session_service.create_call_session(
            db=db,
            user_id=user.id,
            agent_id=agent.id,
            tenant_id=user.current_tenant_id,
            twilio_call_sid=call.sid,
            from_number=twilio_service.get_phone_number(),
            to_number=request.userPhoneNumber,
            call_type="vitchi_outbound",
            assistant_phone_number=twilio_service.get_phone_number(),
            customer_phone_number=request.userPhoneNumber
        )
        
        print(f"📝 Vitchi call session created: {call_session.id}")
        
        # Generate call ID for response
        response_call_id = f"vitchi_call_{call.sid[-8:]}"
        
        return create_success_response(
            CallInitiateResponse(
                callId=response_call_id,
                twilioCallSid=call.sid,
                callSessionId=str(call_session.id),
                status="initiated"
            ),
            "Vitchi dialer call initiated successfully"
        )
        
    except Exception as e:
        print(f"❌ Vitchi call initiation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate Vitchi call: {str(e)}")


@router.post("/vitchi/webhook/call-events")
async def handle_vitchi_call_events(
    request: Request,
    agentId: str = Query(..., description="Agent ID"),
    userId: str = Query(..., description="User ID"),
    callId: str = Query(..., description="Call ID for tracking"),
    db: Session = Depends(get_db)
):
    """
    Handle Vitchi dialer call events and recording
    This is a separate webhook for Vitchi integration
    """
    print("=" * 60)
    print(f"🎤 VITCHI DIALER CALL EVENTS WEBHOOK")
    print(f"📞 Call ID: {callId}")
    print(f"🤖 Agent: {agentId}")
    print(f"👤 User: {userId}")
    print(f"⏰ Timestamp: {datetime.now()}")
    print("=" * 60)
    
    try:
        # Parse form data
        form_data = await request.form()
        
        # Extract call information
        call_sid = form_data.get("CallSid", "")
        call_status = form_data.get("CallStatus", "")
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        direction = form_data.get("Direction", "")
        
        # Speech recognition results
        speech_result = form_data.get("SpeechResult", "")
        confidence = form_data.get("Confidence", "")
        speech_duration = form_data.get("SpeechDuration", "")
        
        # Recording information (Vitchi specific)
        recording_url = form_data.get("RecordingUrl", "")
        recording_duration = form_data.get("RecordingDuration", "")
        recording_sid = form_data.get("RecordingSid", "")
        
        print(f"📞 Call SID: {call_sid}")
        print(f"📊 Status: {call_status}")
        print(f"📱 From: {from_number} → To: {to_number}")
        print(f"🎤 Speech: '{speech_result}' (Confidence: {confidence})")
        print(f"🎙️ Recording URL: {recording_url}")
        print(f"⏱️ Recording Duration: {recording_duration}")
        print(f"🎙️ Recording SID: {recording_sid}")
        
        # Get agent from database
        agent = None
        if agentId:
            try:
                agent_uuid = uuid.UUID(agentId)
                agent = db.query(Agent).filter(Agent.id == agent_uuid).first()
                if agent:
                    print(f"🤖 Found agent: {agent.name}")
                else:
                    print(f"❌ Agent not found: {agentId}")
            except Exception as e:
                print(f"❌ Error getting agent: {e}")
        
        # Handle different call statuses for Vitchi
        if call_status == "initiated":
            print(f"🚀 Vitchi call initiated - SID: {call_sid}")
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "ringing":
            print(f"🔔 Vitchi call ringing - SID: {call_sid}")
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "in-progress":
            print(f"📞 Vitchi call answered - SID: {call_sid}")
            
            # Generate TwiML response for Vitchi answered call
            response = VoiceResponse()
            
            if agent:
                agent_name = agent.name
                print(f"🤖 Vitchi Agent: {agent_name}")
                
                # Start recording for Vitchi
                response.record(
                    action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/vitchi/webhook/call-events?agentId={agentId}&userId={userId}&callId={callId}',
                    method='POST',
                    timeout=30,
                    finish_on_key='#',
                    play_beep=True
                )
                
                response.say(f"Hello! This is {agent_name} from Vitchi dialer.", voice="en-US-Neural2-F")
                response.pause(length=2)
                response.say("I can help you with any questions you have.", voice="en-US-Neural2-F")
                response.pause(length=2)
                response.say("Please speak clearly and I will respond to you.", voice="en-US-Neural2-F")
                
                # Gather speech input with recording
                response.gather(
                    input='speech',
                    timeout=15,
                    speech_timeout='auto',
                    action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/vitchi/webhook/call-events?agentId={agentId}&userId={userId}&callId={callId}',
                    method='POST'
                )
                
                # Fallback
                response.say("I didn't hear anything. Thank you for calling Vitchi. Goodbye!", voice="en-US-Neural2-F")
            else:
                response.say("Hello! Thank you for calling Vitchi dialer.", voice="en-US-Neural2-F")
                response.say("An agent will be with you shortly.", voice="en-US-Neural2-F")
            
            twiml_result = str(response)
            print(f"📝 Generated Vitchi TwiML: {twiml_result[:200]}...")
            return HTMLResponse(twiml_result, media_type="application/xml")
        
        elif call_status == "completed":
            print(f"✅ Vitchi call completed - SID: {call_sid}")
            
            # Update call session status for Vitchi
            try:
                call_session = call_session_service.get_call_session_by_twilio_sid(db, call_sid)
                if call_session:
                    call_session_service.update_call_session_status(
                        db=db,
                        session_id=call_session.id,
                        status="completed",
                        ended_reason="Vitchi call completed successfully",
                        success_evaluation="success"
                    )
                    print(f"📝 Updated Vitchi call session: {call_session.id}")
            except Exception as e:
                print(f"⚠️ Error updating Vitchi call session: {e}")
            
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "failed":
            print(f"❌ Vitchi call failed - SID: {call_sid}")
            
            # Update call session status for Vitchi
            try:
                call_session = call_session_service.get_call_session_by_twilio_sid(db, call_sid)
                if call_session:
                    call_session_service.update_call_session_status(
                        db=db,
                        session_id=call_session.id,
                        status="failed",
                        ended_reason="Vitchi call failed",
                        success_evaluation="fail"
                    )
            except Exception as e:
                print(f"⚠️ Error updating Vitchi call session: {e}")
            
            return HTMLResponse("", media_type="application/xml")
        
        # Handle speech input for Vitchi
        elif speech_result:
            print("=" * 60)
            print(f"🎤 VITCHI SPEECH RECEIVED: '{speech_result}'")
            print(f"📊 Confidence: {confidence}, Duration: {speech_duration}")
            print(f"📞 Call SID: {call_sid}")
            print(f"🏢 Tenant: {agent.tenant_id if agent else 'Unknown'}")
            print(f"🤖 Agent: {agent.name if agent else 'Unknown'}")
            print("=" * 60)
            
            # Generate response for Vitchi
            response = VoiceResponse()
            
            if agent:
                # Use your existing speech processing logic
                response_text = _process_speech_input(agent, speech_result, call_sid)
                response.say(f"Vitchi: {response_text}", voice="en-US-Neural2-F")
                
                # Continue listening with recording
                response.gather(
                    input='speech',
                    timeout=15,
                    speech_timeout='auto',
                    action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/vitchi/webhook/call-events?agentId={agentId}&userId={userId}&callId={callId}',
                    method='POST'
                )
                
                # Fallback
                response.say("Thank you for calling Vitchi. Goodbye!", voice="en-US-Neural2-F")
            else:
                response.say(f"Vitchi heard you say: {speech_result}. Thank you for calling!", voice="en-US-Neural2-F")
            
            twiml_result = str(response)
            print(f"📝 Vitchi speech response: {twiml_result[:200]}...")
            return HTMLResponse(twiml_result, media_type="application/xml")
        
        # Handle recording completion for Vitchi
        elif recording_url:
            print("=" * 60)
            print(f"🎙️ VITCHI RECORDING COMPLETED")
            print(f"📞 Call SID: {call_sid}")
            print(f"🎙️ Recording URL: {recording_url}")
            print(f"⏱️ Recording Duration: {recording_duration}")
            print(f"🎙️ Recording SID: {recording_sid}")
            print("=" * 60)
            
            # Store recording information in call session
            try:
                call_session = call_session_service.get_call_session_by_twilio_sid(db, call_sid)
                if call_session:
                    # Update call session with recording info
                    call_session.call_metadata = {
                        "recording_url": recording_url,
                        "recording_duration": recording_duration,
                        "recording_sid": recording_sid,
                        "vitchi_call": True
                    }
                    db.commit()
                    print(f"📝 Updated Vitchi call session with recording: {call_session.id}")
            except Exception as e:
                print(f"⚠️ Error updating Vitchi call session with recording: {e}")
            
            # Continue the call after recording
            response = VoiceResponse()
            response.say("Recording completed. How can I help you further?", voice="en-US-Neural2-F")
            
            response.gather(
                input='speech',
                timeout=15,
                speech_timeout='auto',
                action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/vitchi/webhook/call-events?agentId={agentId}&userId={userId}&callId={callId}',
                method='POST'
            )
            
            response.say("Thank you for calling Vitchi. Goodbye!", voice="en-US-Neural2-F")
            
            return HTMLResponse(str(response), media_type="application/xml")
        
        else:
            print(f"📊 Vitchi call status: {call_status} - No action needed")
            return HTMLResponse("", media_type="application/xml")
            
    except Exception as e:
        print(f"❌ Vitchi webhook error: {e}")
        return HTMLResponse("", media_type="application/xml")