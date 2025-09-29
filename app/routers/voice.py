from fastapi import APIRouter, Request, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from typing import Optional
from twilio.twiml.voice_response import VoiceResponse
from datetime import datetime, timezone

from app.api.deps import get_db, require_tenant
from app.schemas.twilio import CallInitiateRequest, CallInitiateResponse
from app.schemas.base import SuccessResponse
from app.services.twilio_service import twilio_service
from app.services.agent_service import agent_service
from app.models.agent import Agent
from app.models.user import User
from app.models.call_session import CallSession
from app.services.call_session_service import call_session_service
from app.services.voice_logging_service import VoiceLoggingService
from app.utils.twilio_validation import validate_twilio_signature, validate_webrtc_auth, get_request_body
from app.utils.response import create_success_response
from app.core.config import settings
import uuid
from datetime import datetime, timezone

router = APIRouter()


def _add_to_transcript(call_session, message_type: str, content: str, timestamp: datetime = None):
    """Add a message to the call session transcript"""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    # Initialize transcript if it doesn't exist
    if not call_session.call_transcript:
        call_session.call_transcript = []
    
    # Add new message to transcript
    message = {
        "type": message_type,  # "user_speech" or "agent_response"
        "content": content,
        "timestamp": timestamp.isoformat()
    }
    
    call_session.call_transcript.append(message)
    print(f"📝 Added to transcript: {message_type} - {content[:50]}...")


def get_agent_voice(agent) -> str:
    """Get the appropriate Twilio voice based on agent's voice type and language"""
    if not agent:
        return "Polly.Joanna"  # Default female voice
    
    # Get voice type and language from agent
    voice_type = agent.voice_type
    language = agent.language
    
    # Voice mapping based on language and gender using correct Twilio voice names
    voice_map = {
        # English voices
        "en": {
            "male": "Polly.Matthew",
            "female": "Polly.Joanna"
        },
        # Spanish voices
        "es": {
            "male": "Polly.Miguel",
            "female": "Polly.Penelope"
        },
        # Hindi voices
        "hi": {
            "male": "Polly.Aditi",
            "female": "Polly.Aditi"
        },
        # Arabic voices
        "ar": {
            "male": "Polly.Zeina",
            "female": "Polly.Zeina"
        },
        # Chinese voices
        "zh": {
            "male": "Polly.Zhiyu",
            "female": "Polly.Zhiyu"
        },
        # Urdu voices
        "ur": {
            "male": "Polly.Aditi",
            "female": "Polly.Aditi"
        }
    }
    
    # Default to English if language not specified
    if not language:
        language = "en"
    
    # Default to female if voice type not specified
    if not voice_type:
        voice_type = "female"
    
    # Get the voice from the mapping
    selected_voice = voice_map.get(language, voice_map["en"]).get(voice_type, "Polly.Joanna")
    
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
        
        # Make the call using Twilio with main call events webhook
        webhook_url = f"{base_url}/api/v1/voice/webhook/call-events?agentId={agent.id}&userId={user.id}"
        status_callback_url = f"{base_url}/api/v1/voice/webhook/call-events?agentId={agent.id}&userId={user.id}"
        
        print(f"Making call with webhook_url: {webhook_url}")
        print(f"Making call with status_callback_url: {status_callback_url}")
        
        # Make the call using Twilio
        call = twilio_service.make_call(
            to_number=request.userPhoneNumber,
            from_number=twilio_service.get_phone_number(),
            webhook_url=webhook_url,
            status_callback_url=status_callback_url
        )
        print(f"✅ Call initiated successfully")
        
        # Create call session immediately when call is initiated
        call_session = call_session_service.create_call_session(
            db=db,
            user_id=user.id,
            agent_id=agent.id,
            tenant_id=user.current_tenant_id,
            twilio_call_sid=call.sid,
            from_number=twilio_service.get_phone_number(),
            to_number=request.userPhoneNumber,
            call_type="outbound"  # Agent is initiating the call, so it's outbound
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


@router.post("/webhook/call-events", response_class=HTMLResponse,include_in_schema=False)
async def handle_call_events_webhook(
    request: Request,
    agentId: Optional[str] = Query(None),
    body: str = Depends(get_request_body),
    db: Session = Depends(get_db)
):
    print("=== Call Events Webhook Started === print check")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
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
                    # Update call session status if call_status is provided
                    call_status = form_data.get("CallStatus", "")
                    if call_status:
                        call_session.status = call_status
                        
                        # Set start time when call becomes in-progress
                        if call_status == "in-progress" and not call_session.start_time:
                            call_session.start_time = datetime.now(timezone.utc)
                        
                        # Set end time and calculate duration when call completes
                        if call_status == "completed":
                            call_session.end_time = datetime.now(timezone.utc)
                            if call_session.start_time:
                                duration = (call_session.end_time - call_session.start_time).total_seconds()
                                call_session.duration = int(duration)
                            
                            # Save transcript to database when call completes
                            if call_session.call_transcript:
                                print(f"📝 Saving transcript with {len(call_session.call_transcript)} messages")
                                print(f"📝 Transcript content: {call_session.call_transcript}")
                        
                        db.commit()
                        print(f"✅ Updated call session {call_session.id} status to: {call_status}")
                    
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
        
        # Handle speech input - ROBUST MULTI-TENANT LOGGING
        if speech_result and speech_result.strip():
            # VALID SPEECH DETECTED - LOG AND RESPOND
            print("=" * 60)
            print(f"🎤 SPEECH DETECTED: '{speech_result}'")
            print(f"📊 Confidence: {confidence}, Duration: {speech_duration}")
            print(f"📞 Call SID: {call_sid}")
            print(f"⏰ Timestamp: {datetime.now(timezone.utc)}")
            print(f"🏢 Tenant ID: {agent.tenant_id if agent else 'Unknown'}")
            print(f"🤖 Agent: {agent.name if agent else 'Unknown'}")
            print("=" * 60)
            
            # Log voice interaction for smooth tracking
            try:
                call_session = call_session_service.get_call_session_by_twilio_sid(db, call_sid)
                if call_session:
                    # Add user speech to transcript
                    _add_to_transcript(call_session, "user_speech", speech_result)
                    
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
                
                # Add agent response to transcript
                if call_session:
                    _add_to_transcript(call_session, "agent_response", response_text)
                
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
        
        elif speech_result == "" or speech_result is None:
            # NO SPEECH DETECTED - KEEP LISTENING, DON'T TERMINATE
            print("=" * 60)
            print(f"🔇 NO SPEECH DETECTED - KEEPING CALL ALIVE")
            print(f"📞 Call SID: {call_sid}")
            print(f"⏰ Timestamp: {datetime.now(timezone.utc)}")
            print(f"🏢 Tenant ID: {agent.tenant_id if agent else 'Unknown'}")
            print("=" * 60)
            
            response = VoiceResponse()
            agent_voice = get_agent_voice(agent)
            response.say("Sorry, I didn't catch that.", voice=agent_voice)
            response.pause(length=0.5)
            response.say("Could you speak a bit louder?", voice=agent_voice)
            response.pause(length=1)
            response.say("I'm still here.", voice=agent_voice)
            
            # Keep listening with single gather - no multiple attempts
            response.gather(
                input='speech',
                timeout=20,
                speech_timeout='auto',
                action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agent.id if agent else ""}',
                method='POST'
            )
            
            # Gentle reminder only
            response.say("I'm still listening. Go ahead when you're ready.", voice=agent_voice)
            response.pause(length=1)
            
            # Final attempt with longer timeout
            response.gather(
                input='speech',
                timeout=30,
                speech_timeout='auto',
                action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agent.id if agent else ""}',
                method='POST'
            )
            
            # Only hangup after very long silence
            response.say("I haven't heard anything for a while. Thanks for calling!", voice=agent_voice)
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
            
            # Get agent info from database using agentId from query params
            agent = None
            agent_name = "AI Assistant"
            if agentId:
                try:
                    # Get call session to get tenant_id
                    call_session = call_session_service.get_call_session_by_twilio_sid(db, call_sid)
                    if call_session:
                        # Fetch agent from database
                        agent = agent_service.get_agent_by_id(db, uuid.UUID(agentId), call_session.tenant_id)
                        if agent:
                            agent_name = agent.name
                            print(f"🏢 Multi-tenant call for tenant: {agent.tenant_id}")
                            print(f"🤖 Agent: {agent_name}")
                        else:
                            print(f"⚠️ Agent {agentId} not found in tenant {call_session.tenant_id}")
                    else:
                        print(f"⚠️ Call session not found for SID: {call_sid}")
                except Exception as e:
                    print(f"⚠️ Error fetching agent: {e}")
                    agent = None
            
            # Natural, conversational greeting with agent-specific voice
            agent_voice = get_agent_voice(agent)
            greeting_text = f"Hi! This is {agent_name}. How are you doing today? What can I help you with?"
            response.say(f"Hi! This is {agent_name}.", voice=agent_voice)
            response.pause(length=0.5)  # Natural pause
            response.say("How are you doing today?", voice=agent_voice)
            response.pause(length=0.5)  # Natural pause
            response.say("What can I help you with?", voice=agent_voice)
            
            # Add initial greeting to transcript
            if call_session:
                _add_to_transcript(call_session, "agent_response", greeting_text)
            
            # Log call answered event
            try:
                if call_session:
                    await VoiceLoggingService.log_call_events(
                        db=db,
                        call_session_id=call_session.id,
                        event_type="call_answered",
                        event_data={
                            "call_sid": call_sid,
                            "agent_name": agent_name,
                            "agent_id": str(agent.id) if agent else None,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
            except Exception as e:
                print(f"⚠️ Error logging call answered event: {e}")
            
            # Use main webhook for conversation flow
            response.gather(
                input='speech',
                timeout=15,  # Reasonable timeout for initial greeting
                speech_timeout='auto',
                action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agentId}',
                method='POST'
            )
            
            # Gentle fallback if no input
            response.say("I didn't catch that. Could you please repeat?", voice=agent_voice)
            response.pause(length=1)
            
            # Try main webhook again
            response.gather(
                input='speech',
                timeout=20,
                speech_timeout='auto',
                action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agentId}',
                method='POST'
            )
            
            # Final gentle attempt
            response.say("I'm having trouble hearing you. Please call back when you have a moment. Thank you!", voice=agent_voice)
            
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


# def _get_twilio_voice(voice_type):
#     """Map voice_type to Twilio voice names"""
#     if voice_type == "male":
#         return "en-US-Neural2-M"  # Male voice
#     elif voice_type == "female":
#         return "en-US-Neural2-F"  # Female voice
#     else:
#         return "en-US-Neural2-F"  # Default to female voice

# def _process_speech_input(agent, speech_text: str, call_sid: str) -> str:
#     """Process speech input and generate agent response"""
#     if not agent:
#         return f"I heard you say: {speech_text}. How can I help you further?"
    
#     # Simple keyword-based responses (you can enhance this with AI)
#     speech_lower = speech_text.lower()
    
#     if any(word in speech_lower for word in ['hello', 'hi', 'hey']):
#         return f"Hello! This is {agent.name}. How can I assist you today?"
#     elif any(word in speech_lower for word in ['help', 'support', 'assistance']):
#         return f"I'm here to help! What specific assistance do you need?"
#     elif any(word in speech_lower for word in ['thank', 'thanks']):
#         return f"You're welcome! Is there anything else I can help you with?"
#     elif any(word in speech_lower for word in ['bye', 'goodbye', 'end']):
#         return f"Thank you for calling! Have a great day!"
#     else:
#         return f"I understand you said: {speech_text}. Let me help you with that. What would you like me to do?"

# def _generate_agent_response(agent, call_data: dict) -> str:
#     """Generate TwiML response based on agent from database"""
#     if not agent:
#         return _generate_default_response()
    
#     # Create TwiML response
#     response = VoiceResponse()
    
#     # Use agent's name and fallback response
#     agent_name = agent.name
#     greeting = agent.fallback_response if agent.fallback_response and agent.fallback_response.strip() and agent.fallback_response != "string" else f"Hello! This is {agent_name} speaking. How can I help you today?"
#     twilio_voice = _get_twilio_voice(agent.voice_type)
    
#     print(f"🎯 Agent greeting: '{greeting}'")
#     print(f"🎯 Agent voice: '{twilio_voice}'")
    
#     # ABSOLUTE SIMPLEST - NO GATHER AT ALL
#     response.say("Hello! This is your AI assistant speaking.", voice=twilio_voice)
#     response.pause(length=2)
#     response.say("I can help you with any questions you have.", voice=twilio_voice)
#     response.pause(length=2)
#     response.say("Thank you for calling. Have a great day!", voice=twilio_voice)
#     response.pause(length=1)
#     response.hangup()
    
#     return str(response)


@router.get("/dashboard/analytics", response_model=SuccessResponse[dict])
async def get_dashboard_analytics(
    agent_id: Optional[str] = Query(None, description="Filter by specific agent ID"),
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get dashboard analytics for the current tenant.
    Returns call statistics including number of calls and average duration.
    Optionally filter by specific agent ID.
    """
    try:
        tenant_id = user.current_tenant_id
        
        # Build base query for call sessions
        base_query = db.query(CallSession).filter(CallSession.tenant_id == tenant_id)
        
        # Apply agent filter if provided
        if agent_id:
            try:
                agent_uuid = uuid.UUID(agent_id)
                base_query = base_query.filter(CallSession.agent_id == agent_uuid)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid agent ID format")
        
        # Get all call sessions for the tenant (with optional agent filter)
        call_sessions = base_query.all()
        
        # Calculate statistics
        total_calls = len(call_sessions)
        
        # Filter completed calls for duration calculation
        completed_calls = [call for call in call_sessions if call.status == "completed" and call.duration is not None]
        
        # Calculate average duration
        if completed_calls:
            total_duration = sum(call.duration for call in completed_calls)
            average_duration = total_duration / len(completed_calls)
        else:
            average_duration = 0
        
        # Get calls by status
        status_counts = {}
        for call in call_sessions:
            status = call.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Get calls by type
        type_counts = {}
        for call in call_sessions:
            call_type = call.call_type
            type_counts[call_type] = type_counts.get(call_type, 0) + 1
        
        # Get agent-wise statistics (only if not filtering by specific agent)
        agent_stats = {}
        if not agent_id:
            # Get all agents for this tenant
            agents = db.query(Agent).filter(Agent.tenant_id == tenant_id).all()
            
            for agent in agents:
                agent_calls = [call for call in call_sessions if call.agent_id == agent.id]
                agent_completed = [call for call in agent_calls if call.status == "completed" and call.duration is not None]
                
                agent_avg_duration = 0
                if agent_completed:
                    agent_total_duration = sum(call.duration for call in agent_completed)
                    agent_avg_duration = agent_total_duration / len(agent_completed)
                
                agent_stats[str(agent.id)] = {
                    "agent_name": agent.name,
                    "total_calls": len(agent_calls),
                    "completed_calls": len(agent_completed),
                    "average_duration_seconds": round(agent_avg_duration, 2),
                    "average_duration_minutes": round(agent_avg_duration / 60, 2)
                }
        
        # Get recent calls (last 10)
        recent_calls = base_query.order_by(CallSession.created_at.desc()).limit(10).all()
        
        # Format recent calls data
        recent_calls_data = []
        for call in recent_calls:
            recent_calls_data.append({
                "id": str(call.id),
                "call_sid": call.twilio_call_sid,
                "agent_name": call.agent.name if call.agent else "Unknown",
                "status": call.status,
                "call_type": call.call_type,
                "duration": call.duration,
                "start_time": call.start_time.isoformat() if call.start_time else None,
                "end_time": call.end_time.isoformat() if call.end_time else None,
                "from_number": call.from_number,
                "to_number": call.to_number,
                "cost": call.cost,
                "recording_url": call.recording_url,
                "has_recording": call.recording_url is not None
            })
        
        # Prepare analytics data
        analytics_data = {
            "tenant_id": str(tenant_id),
            "filtered_by_agent": agent_id is not None,
            "agent_id": agent_id,
            "total_calls": total_calls,
            "completed_calls": len(completed_calls),
            "average_duration_seconds": round(average_duration, 2),
            "average_duration_minutes": round(average_duration / 60, 2),
            "status_breakdown": status_counts,
            "call_type_breakdown": type_counts,
            "agent_statistics": agent_stats,
            "recent_calls": recent_calls_data,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        message = f"Retrieved dashboard analytics for tenant {tenant_id}"
        if agent_id:
            message += f" filtered by agent {agent_id}"
        
        return create_success_response(analytics_data, message)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard analytics: {str(e)}")


@router.get("/call-recordings/{call_sid}", response_model=SuccessResponse[dict])
async def get_call_recordings(
    call_sid: str,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get call recordings for a specific call SID.
    Returns recording URLs and metadata.
    """
    try:
        # Verify the call belongs to the user's tenant
        call_session = db.query(CallSession).filter(
            CallSession.twilio_call_sid == call_sid,
            CallSession.tenant_id == user.current_tenant_id
        ).first()
        
        if not call_session:
            raise HTTPException(status_code=404, detail="Call not found or access denied")
        
        # Check if call has recordings
        has_recordings = twilio_service.has_call_recordings(call_sid)
        
        if not has_recordings:
            return create_success_response(
                {
                    "call_sid": call_sid,
                    "call_session_id": str(call_session.id),
                    "recordings": [],
                    "total_recordings": 0,
                    "message": "No recordings found for this call. Recordings may not be available yet or recording was not enabled.",
                    "generated_at": datetime.now(timezone.utc).isoformat()
                },
                f"No recordings found for call {call_sid}"
            )
        
        # Get recordings from Twilio
        recordings = twilio_service.get_call_recordings(call_sid)
        
        # Format recording data
        recording_data = []
        for recording in recordings:
            recording_info = twilio_service.get_recording_url(recording.sid)
            recording_data.append(recording_info)
        
        # Update call session with recording URL if available and completed
        if recording_data and not call_session.recording_url:
            # Use the first completed recording URL
            for recording_info in recording_data:
                if recording_info.get('url'):
                    call_session.recording_url = recording_info['url']
                    db.commit()
                    break
        
        response_data = {
            "call_sid": call_sid,
            "call_session_id": str(call_session.id),
            "recordings": recording_data,
            "total_recordings": len(recording_data),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        return create_success_response(
            response_data,
            f"Retrieved {len(recording_data)} recordings for call {call_sid}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get call recordings: {str(e)}")


@router.get("/call-session/{session_id}/recording", response_model=SuccessResponse[dict])
async def get_call_session_recording(
    session_id: str,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get recording URL for a specific call session.
    """
    try:
        # Get call session
        call_session = db.query(CallSession).filter(
            CallSession.id == session_id,
            CallSession.tenant_id == user.current_tenant_id
        ).first()
        
        if not call_session:
            raise HTTPException(status_code=404, detail="Call session not found")
        
        if not call_session.twilio_call_sid:
            raise HTTPException(status_code=404, detail="No Twilio call SID found for this session")
        
        # Check if call has recordings
        has_recordings = twilio_service.has_call_recordings(call_session.twilio_call_sid)
        
        if not has_recordings:
            return create_success_response(
                {
                    "recording_url": None, 
                    "message": "No recordings found for this call. Recordings may not be available yet or recording was not enabled.",
                    "recording_status": "not_available"
                },
                "No recordings available for this call session"
            )
        
        # Get recordings from Twilio
        recordings = twilio_service.get_call_recordings(call_session.twilio_call_sid)
        
        # Get the first recording info
        recording_info = twilio_service.get_recording_url(recordings[0].sid)
        
        # Update call session with recording URL if not already set and recording is completed
        if not call_session.recording_url and recording_info.get('url'):
            call_session.recording_url = recording_info['url']
            db.commit()
        
        response_data = {
            "call_session_id": str(call_session.id),
            "call_sid": call_session.twilio_call_sid,
            "recording_url": recording_info['url'],
            "recording_duration": recording_info['duration'],
            "recording_status": recording_info['status'],
            "recording_channels": recording_info['channels']
        }
        
        return create_success_response(
            response_data,
            f"Retrieved recording for call session {session_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get call session recording: {str(e)}")


@router.post("/webhook/recording-status")
async def handle_recording_status_webhook(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Handle Twilio recording status callbacks.
    This webhook is called when recording status changes (in-progress, completed, etc.)
    """
    try:
        form_data = await request.form()
        
        # Extract recording information
        recording_sid = form_data.get("RecordingSid")
        call_sid = form_data.get("CallSid")
        recording_status = form_data.get("RecordingStatus")
        recording_url = form_data.get("RecordingUrl")
        recording_duration = form_data.get("RecordingDuration")
        
        print("=" * 60)
        print(f"🎙️ RECORDING STATUS UPDATE")
        print(f"Recording SID: {recording_sid}")
        print(f"Call SID: {call_sid}")
        print(f"Status: {recording_status}")
        print(f"URL: {recording_url}")
        print(f"Duration: {recording_duration}")
        print("=" * 60)
        
        # Find the call session
        if call_sid:
            call_session = call_session_service.get_call_session_by_twilio_sid(db, call_sid)
            if call_session:
                # Update recording URL when recording is completed
                if recording_status == "completed" and recording_url:
                    call_session.recording_url = recording_url
                    db.commit()
                    print(f"✅ Updated call session {call_session.id} with recording URL")
                else:
                    print(f"📝 Recording status: {recording_status} - URL not ready yet")
            else:
                print(f"⚠️ Call session not found for SID: {call_sid}")
        
        # Return empty TwiML response
        return HTMLResponse("", media_type="application/xml")
        
    except Exception as e:
        print(f"⚠️ Error handling recording status webhook: {e}")
        return HTMLResponse("", media_type="application/xml")


# def _generate_default_response() -> str:
#     """Generate default TwiML response"""
#     response = VoiceResponse()
#     response.say("Thank you for calling. An agent will be with you shortly.", voice="")
#     response.pause(length=2)
#     response.say("Please hold while we connect you.", voice="")
#     return str(response)