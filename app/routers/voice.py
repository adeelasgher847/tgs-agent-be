from fastapi import APIRouter, Request, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from typing import Optional
from twilio.twiml.voice_response import VoiceResponse
from datetime import datetime, timezone
import random
import uuid

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
from app.routers.general_websocket import (
    broadcast_transcript_update,
    broadcast_call_status_update,
    broadcast_call_ended,
    broadcast_call_event,
    broadcast_system_notification
)

router = APIRouter()

# Array of human-like "didn't catch that" response phrases
DIDNT_CATCH_RESPONSES = [
    "Hmm, I missed that—mind saying it again?",
    "Didn't quite get that, can you repeat?",
    "I didn't hear you clearly, would you mind repeating?",
    "Can you say that again real quick?",
    "I might've misheard—could you repeat that?"
]

# Array of follow-up phrases for when the agent didn't catch something
FOLLOW_UP_RESPONSES = [
    "Could you repeat that for me?",
    "Mind saying that one more time?",
    "Can you try that again?",
    "Would you mind repeating that?",
    "Could you say that again?"
]


def _get_random_didnt_catch_response() -> str:
    """Get a random 'didn't catch that' response to make interactions feel more human"""
    return random.choice(DIDNT_CATCH_RESPONSES)


def _get_random_follow_up_response() -> str:
    """Get a random follow-up response to make interactions feel more human"""
    return random.choice(FOLLOW_UP_RESPONSES)


async def _add_to_transcript(call_session, role: str, message: str, db: Session, timestamp: datetime = None):
    """Add a message to the call session transcript and broadcast to WebSocket
    
    Args:
        call_session: The call session object
        role: Either "agent" or "client" 
        message: The message content
        db: Database session for committing changes
        timestamp: Optional timestamp, defaults to current time
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    # Initialize transcript if it doesn't exist
    if not call_session.call_transcript:
        call_session.call_transcript = []
    
    # Add new message to transcript in role-based format
    transcript_entry = {
        "role": role,  # "agent" or "client"
        "message": message,
        "timestamp": timestamp.isoformat()
    }
    
    call_session.call_transcript.append(transcript_entry)
    print(f"📝 Added to transcript: {role} - {message[:50]}...")
    
    # Commit the transcript changes to database
    try:
        db.commit()
        print(f"✅ Committed transcript changes to database for session {call_session.id}")
    except Exception as e:
        print(f"❌ Failed to commit transcript changes: {e}")
    
    # Broadcast transcript update to WebSocket
    try:
        await broadcast_transcript_update(
            call_session_id=str(call_session.id),
            transcript=call_session.call_transcript,
            new_messages=[transcript_entry]
        )
        print(f"✅ Broadcasted transcript update for session {call_session.id}")
    except Exception as e:
        print(f"❌ Failed to broadcast transcript update: {e}")


def _get_conversation_state(call_session):
    """Helper function to get conversation state"""
    if not call_session.call_metadata:
        call_session.call_metadata = {}
    if "conversation_state" not in call_session.call_metadata:
        call_session.call_metadata["conversation_state"] = {}
    return call_session.call_metadata["conversation_state"]


def _update_conversation_state(call_session, key: str, value):
    """Helper function to update conversation state"""
    state = _get_conversation_state(call_session)
    state[key] = value
    call_session.call_metadata["conversation_state"] = state


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
        
        # Create call session first so we can include the ID in webhook URLs
        call_session = call_session_service.create_call_session(
            db=db,
            user_id=user.id,
            agent_id=agent.id,
            tenant_id=user.current_tenant_id,
            twilio_call_sid="",  # Will be updated after call is made
            from_number=twilio_service.get_phone_number(),
            to_number=request.userPhoneNumber,
            call_type="outbound"  # Agent is initiating the call, so it's outbound
        )
        
        # Make the call using Twilio with call session ID in webhook URLs
        webhook_url = f"{base_url}/api/v1/voice/webhook/call-events?agentId={agent.id}&userId={user.id}&callSessionId={call_session.id}"
        status_callback_url = f"{base_url}/api/v1/voice/webhook/call-events?agentId={agent.id}&userId={user.id}&callSessionId={call_session.id}"
        
        print(f"Making call with webhook_url: {webhook_url}")
        print(f"Making call with status_callback_url: {status_callback_url}")
        
        # Broadcast call initiating event BEFORE making the call
        try:
            await broadcast_call_status_update(
                call_session_id=str(call_session.id),
                status="initiating",
                metadata={
                    "agent_id": str(agent.id),
                    "agent_name": agent.name,
                    "to_number": request.userPhoneNumber,
                    "from_number": twilio_service.get_phone_number(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            print(f"✅ Broadcasted call initiating event for session {call_session.id}")
        except Exception as e:
            print(f"❌ Failed to broadcast call initiating event: {e}")
        
        # Make the call using Twilio
        call = twilio_service.make_call(
            to_number=request.userPhoneNumber,
            from_number=twilio_service.get_phone_number(),
            webhook_url=webhook_url,
            status_callback_url=status_callback_url
        )
        print(f"✅ Call initiated successfully")
        
        # Update call session with Twilio SID
        call_session.twilio_call_sid = call.sid
        db.commit()
        print(f"✅ Updated call session {call_session.id} with Twilio SID: {call.sid}")
        
        # Broadcast call initiated event AFTER Twilio confirms
        try:
            await broadcast_call_status_update(
                call_session_id=str(call_session.id),
                status="initiated",
                metadata={
                    "call_sid": call.sid,
                    "agent_id": str(agent.id),
                    "agent_name": agent.name,
                    "to_number": request.userPhoneNumber,
                    "from_number": twilio_service.get_phone_number(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            print(f"✅ Broadcasted call initiated event for session {call_session.id}")
        except Exception as e:
            print(f"❌ Failed to broadcast call initiated event: {e}")
        
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
    userId: Optional[str] = Query(None),
    callSessionId: Optional[str] = Query(None),
    body: str = Depends(get_request_body),
    db: Session = Depends(get_db)
):
    print("🔥🔥🔥 WEBHOOK CALLED! 🔥🔥🔥")
    print("=== Call Events Webhook Started ===")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Request method: {request.method}")
    print(f"Request URL: {request.url}")
    print(f"Request headers: {dict(request.headers)}")
    print(f"Query params: agentId={agentId}, userId={userId}, callSessionId={callSessionId}")
    print(f"Request body length: {len(body) if body else 0}")
    print(f"Request body preview: {body[:200] if body else 'None'}...")
    print(f"Database session: {db}")
    
    # Test WebSocket broadcast at the start of webhook
    try:
        await broadcast_system_notification(
            notification_type="webhook_started",
            message=f"Webhook started for call session {callSessionId}",
            metadata={
                "agent_id": agentId,
                "user_id": userId,
                "call_session_id": callSessionId,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        print(f"✅ Test broadcast sent at webhook start")
    except Exception as e:
        print(f"❌ Test broadcast failed at webhook start: {e}")
        import traceback
        traceback.print_exc()
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
        
        # Get call session for this webhook call (prefer callSessionId from query param)
        call_session = None
        if callSessionId:
            try:
                session_uuid = uuid.UUID(callSessionId)
                call_session = call_session_service.get_call_session_by_id(db, session_uuid)
                if call_session:
                    print(f"✅ Found call session: {call_session.id} from query parameter")
                else:
                    print(f"⚠️ No call session found for ID: {callSessionId}")
            except ValueError:
                print(f"⚠️ Invalid call session ID format: {callSessionId}")
        elif call_sid:
            # Fallback to finding by Twilio SID
            call_session = call_session_service.get_call_session_by_twilio_sid(db, call_sid)
            if call_session:
                print(f"✅ Found call session: {call_session.id} for SID: {call_sid}")
            else:
                print(f"⚠️ No call session found for SID: {call_sid}")
        else:
            print(f"⚠️ No call session ID or SID provided")
        
        # Log the call event
        print(f"Call Events Webhook - SID: {call_sid}, Status: {call_status}, From: {from_number}, To: {to_number}, Direction: {direction}")
        print(f"AgentId from query: {agentId}")
        
        # Test WebSocket connection if we have a call session
        if call_session:
            try:
                await broadcast_call_status_update(
                    call_session_id=str(call_session.id),
                    status="webhook_test",
                    metadata={
                        "message": "Webhook is working",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "call_sid": call_sid
                    }
                )
                print(f"✅ Test broadcast sent to WebSocket for session {call_session.id}")
            except Exception as e:
                print(f"❌ Test broadcast failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Update call session status if we have a call session and status
        if call_session and call_status:
            print(f"🔄 Updating call session {call_session.id} status to: {call_status}")
            call_session.status = call_status
            
            # Set start time when call becomes in-progress
            if call_status == "in-progress" and not call_session.start_time:
                call_session.start_time = datetime.now(timezone.utc)
                print(f"⏰ Set start time for session {call_session.id}")
            
            # Set end time and calculate duration when call completes
            if call_status == "completed":
                call_session.end_time = datetime.now(timezone.utc)
                if call_session.start_time:
                    duration = (call_session.end_time - call_session.start_time).total_seconds()
                    call_session.duration = int(duration)
                    print(f"⏰ Set end time and duration ({duration}s) for session {call_session.id}")
                
                # Broadcast call ended event
                try:
                    await broadcast_call_ended(
                        call_session_id=str(call_session.id),
                        reason="completed",
                        duration=call_session.duration,
                        metadata={
                            "call_sid": call_sid,
                            "from_number": from_number,
                            "to_number": to_number,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    print(f"✅ Broadcasted call ended event for session {call_session.id}")
                except Exception as e:
                    print(f"❌ Failed to broadcast call ended event: {e}")
            
            # Commit the status update
            db.commit()
            print(f"✅ Updated call session {call_session.id} status to: {call_status}")
            
            # Broadcast status update to WebSocket
            try:
                print(f"🚀 ATTEMPTING to broadcast call status update: {call_status} for session {call_session.id}")
                await broadcast_call_status_update(
                    call_session_id=str(call_session.id),
                    status=call_status,
                    metadata={
                        "call_sid": call_sid,
                        "from_number": from_number,
                        "to_number": to_number,
                        "direction": direction,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "start_time": call_session.start_time.isoformat() if call_session.start_time else None,
                        "end_time": call_session.end_time.isoformat() if call_session.end_time else None,
                        "duration": call_session.duration
                    }
                )
                print(f"✅ Successfully broadcasted call status update: {call_status} for session {call_session.id}")
            except Exception as e:
                print(f"❌ Failed to broadcast call status update: {e}")
                import traceback
                traceback.print_exc()
            
            # Broadcast call ended event if call completed
            if call_status == "completed":
                try:
                    await broadcast_call_ended(
                        call_session_id=str(call_session.id),
                        reason="Call completed",
                        final_data={
                            "call_sid": call_sid,
                            "duration": call_session.duration,
                            "end_time": call_session.end_time.isoformat(),
                            "transcript": call_session.call_transcript or []
                        }
                    )
                    print(f"✅ Broadcasted call ended event for session {call_session.id}")
                except Exception as e:
                    print(f"❌ Failed to broadcast call ended event: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            if not call_session:
                print(f"⚠️ No call session found - cannot update status or broadcast")
            if not call_status:
                print(f"⚠️ No call status provided - cannot update status or broadcast")
        
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
                # Use the call session we already fetched (should be available from query param)
                if call_session:
                    # Add user speech to transcript
                    print(f"📝 Adding user speech to transcript for session {call_session.id}")
                    await _add_to_transcript(call_session, "client", speech_result, db)
                    print(f"✅ User speech added to transcript for session {call_session.id}")
                    
                    # Also broadcast speech input event directly
                    try:
                        await broadcast_call_event(
                            call_session_id=str(call_session.id),
                            event_type="speech_input",
                            event_data={
                                "speech_text": speech_result,
                                "confidence": float(confidence) if confidence else None,
                                "duration": float(speech_duration) if speech_duration else None,
                                "call_sid": call_sid
                            }
                        )
                        print(f"✅ Broadcasted speech input event for session {call_session.id}")
                    except Exception as e:
                        print(f"❌ Failed to broadcast speech input event: {e}")
                    
                    # Update conversation state with interaction count
                    conversation_state = _get_conversation_state(call_session)
                    interaction_count = conversation_state.get("interaction_count", 0) + 1
                    _update_conversation_state(call_session, "interaction_count", interaction_count)
                    _update_conversation_state(call_session, "last_user_input", speech_result)
                    db.commit()
                    
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
                # Generate intelligent response based on speech using Gemini with conversation context
                response_text = await VoiceLoggingService.generate_agent_response(
                    speech_text=speech_result,
                    confidence=float(confidence) if confidence else 0.0,
                    agent=agent,
                    db=db,
                    call_session_id=call_session.id if call_session else None
                )
                
                # Add agent response to transcript
                if call_session:
                    print(f"📝 Adding agent response to transcript for session {call_session.id}")
                    await _add_to_transcript(call_session, "agent", response_text, db)
                    print(f"✅ Agent response added to transcript for session {call_session.id}")
                    
                    # Also broadcast agent response event directly
                    try:
                        await broadcast_call_event(
                            call_session_id=str(call_session.id),
                            event_type="agent_response",
                            event_data={
                                "response_text": response_text,
                                "agent_id": str(agent.id) if agent else None,
                                "agent_name": agent.name if agent else None,
                                "call_sid": call_sid
                            }
                        )
                        print(f"✅ Broadcasted agent response event for session {call_session.id}")
                    except Exception as e:
                        print(f"❌ Failed to broadcast agent response event: {e}")
                
                # Say response naturally with conversational flow
                agent_voice = get_agent_voice(agent)
                response.say(response_text, voice=agent_voice)
                response.pause(length=0.3)  # Natural pause for conversation flow
                
                # Continue listening immediately for natural conversation
                response.gather(
                    input='speech',
                    timeout=12,  # Shorter timeout for more natural conversation
                    speech_timeout='auto',
                    action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agent.id}&userId={userId}&callSessionId={call_session.id}',
                    method='POST'
                )
                
                # Gentle, natural fallback
                response.say("I'm here if you want to talk about anything else.", voice=agent_voice)
            else:
                # Default response for smooth conversation
                default_voice = get_agent_voice(None)  # Use default voice
                response.say("Got it!", voice=default_voice)
                response.pause(length=0.3)
                response.say("What else would you like to talk about?", voice=default_voice)
                
                # Continue listening
                response.gather(
                    input='speech',
                    timeout=15,
                    speech_timeout='auto',
                    action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agent.id if agent else ""}&userId={userId}&callSessionId={call_session.id if call_session else ""}',
                    method='POST'
                )
                
                response.say("Thanks for calling! Take care!", voice=default_voice)
            
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
            
            # More natural "didn't catch that" response
            response.say(_get_random_didnt_catch_response(), voice=agent_voice)
            response.pause(length=0.3)
            response.say(_get_random_follow_up_response(), voice=agent_voice)
            
            # Keep listening with reasonable timeout
            response.gather(
                input='speech',
                timeout=15,
                speech_timeout='auto',
                action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agent.id if agent else ""}&userId={userId}&callSessionId={call_session.id if call_session else ""}',
                method='POST'
            )
            
            # Gentle reminder
            response.say("I'm still here. Go ahead when you're ready.", voice=agent_voice)
            response.pause(length=0.3)
            
            # Final attempt with longer timeout
            response.gather(
                input='speech',
                timeout=20,
                speech_timeout='auto',
                action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agent.id if agent else ""}&userId={userId}&callSessionId={call_session.id if call_session else ""}',
                method='POST'
            )
            
            # Only hangup after very long silence
            response.say("I haven't heard anything for a while. Thanks for calling! Have a great day!", voice=agent_voice)
            response.hangup()
            
            print(f"📝 Extended listening response: {str(response)[:200]}...")
            return HTMLResponse(str(response), media_type="application/xml")
        
        # Handle different call statuses and trigger agent logic
        print(f"Processing call status: '{call_status}' with direction: '{direction}'")
        
        if call_status == "initiated" and direction == "outbound-api":
            # Call has been initiated - just log and return empty response
            print(f"Call initiated - SID: {call_sid}")
            
            # Broadcast call initiated event
            if call_session:
                try:
                    await broadcast_call_status_update(
                        call_session_id=str(call_session.id),
                        status="initiated",
                        metadata={
                            "call_sid": call_sid,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    print(f"✅ Broadcasted call initiated event for session {call_session.id}")
                except Exception as e:
                    print(f"❌ Failed to broadcast call initiated event: {e}")
            
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "ringing" and direction == "outbound-api":
            # Outbound call is ringing - just log, don't play any audio
            print("=" * 50)
            print(f"🔔 CALL IS RINGING - SID: {call_sid}")
            print("=" * 50)
            
            # Broadcast call ringing event
            if call_session:
                try:
                    await broadcast_call_status_update(
                        call_session_id=str(call_session.id),
                        status="ringing",
                        metadata={
                            "call_sid": call_sid,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    print(f"✅ Broadcasted call ringing event for session {call_session.id}")
                except Exception as e:
                    print(f"❌ Failed to broadcast call ringing event: {e}")
            
            # Return empty response - no audio should play while ringing
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "in-progress":
            # Call is in progress - person answered, check if we already greeted
            print("=" * 50)
            print(f"📞 CALL IN PROGRESS - SID: {call_sid}")
            print("=" * 50)
            
            # Broadcast call in-progress event
            if call_session:
                try:
                    await broadcast_call_status_update(
                        call_session_id=str(call_session.id),
                        status="in-progress",
                        metadata={
                            "call_sid": call_sid,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    print(f"✅ Broadcasted call in-progress event for session {call_session.id}")
                except Exception as e:
                    print(f"❌ Failed to broadcast call in-progress event: {e}")
            
            # Get call session to check conversation state
            call_session = call_session_service.get_call_session_by_twilio_sid(db, call_sid)
            if not call_session:
                print(f"⚠️ Call session not found for SID: {call_sid}")
                return HTMLResponse("", media_type="application/xml")
            
            # Check if we already greeted this call
            conversation_state = _get_conversation_state(call_session)
            has_greeted = conversation_state.get("has_greeted", False)
            
            # Get agent info
            agent = None
            agent_name = "AI"
            if agentId:
                try:
                    agent = agent_service.get_agent_by_id(db, uuid.UUID(agentId), call_session.tenant_id)
                    if agent:
                        agent_name = agent.name
                        print(f"🏢 Multi-tenant call for tenant: {agent.tenant_id}")
                        print(f"🤖 Agent: {agent_name}")
                    else:
                        print(f"⚠️ Agent {agentId} not found in tenant {call_session.tenant_id}")
                except Exception as e:
                    print(f"⚠️ Error fetching agent: {e}")
                    agent = None
            
            # Only greet if we haven't greeted yet
            if not has_greeted:
                print("🎤 FIRST TIME GREETING - Playing welcome message")
                
                # Mark as greeted
                _update_conversation_state(call_session, "has_greeted", True)
                _update_conversation_state(call_session, "greeting_time", datetime.now(timezone.utc).isoformat())
                db.commit()
                
                # Natural, conversational greeting with agent-specific voice
                response = VoiceResponse()
                agent_voice = get_agent_voice(agent)
                
                # Professional, concise greeting
                response.say(f"Hello! This is {agent_name}. How can I help you today?", voice=agent_voice)
                
                # Add initial greeting to transcript
                greeting_text = f"Hello! This is {agent_name}. How can I help you today?"
                await _add_to_transcript(call_session, "agent", greeting_text, db)
                
                # Broadcast greeting event
                try:
                    await broadcast_call_event(
                        call_session_id=str(call_session.id),
                        event_type="greeting",
                        event_data={
                            "agent_name": agent_name,
                            "greeting_text": greeting_text,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    print(f"✅ Broadcasted greeting event for session {call_session.id}")
                except Exception as e:
                    print(f"❌ Failed to broadcast greeting event: {e}")
                
                # Log call answered event
                try:
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
                    timeout=15,
                    speech_timeout='auto',
                    action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agentId}&userId={userId}&callSessionId={call_session.id}',
                    method='POST'
                )
                
                # Gentle fallback if no input
                response.say(_get_random_didnt_catch_response(), voice=agent_voice)
                response.pause(length=0.5)
                
                # Try main webhook again
                response.gather(
                    input='speech',
                    timeout=20,
                    speech_timeout='auto',
                    action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agentId}&userId={userId}&callSessionId={call_session.id}',
                    method='POST'
                )
                
                # Final gentle attempt
                response.say("I'm having trouble hearing you clearly. Please call back when you have a moment. Thanks!", voice=agent_voice)
                
                twiml_result = str(response)
                print(f"📝 GREETING TwiML: {twiml_result}")
                return HTMLResponse(twiml_result, media_type="application/xml")
            else:
                print("🔄 ALREADY GREETED - Continuing conversation")
                # Already greeted, just continue listening
                response = VoiceResponse()
                agent_voice = get_agent_voice(agent)
                
                # Continue listening without repeating greeting
                response.gather(
                    input='speech',
                    timeout=15,
                    speech_timeout='auto',
                    action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agentId}&userId={userId}&callSessionId={call_session.id}',
                    method='POST'
                )
                
                # Gentle fallback
                response.say("I'm here if you want to talk about anything else.", voice=agent_voice)
                
                twiml_result = str(response)
                print(f"📝 CONTINUATION TwiML: {twiml_result}")
                return HTMLResponse(twiml_result, media_type="application/xml")
        
        elif call_status == "completed":
            # Call completed
            print(f"📞 CALL COMPLETED - SID: {call_sid}")
            
            # Broadcast call completed event (this is already handled above in the status update section)
            # The broadcast_call_ended is already called in the status update section above
            
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "failed":
            # Call failed - handle error
            print(f"Call failed - SID: {call_sid}")
            
            # Broadcast call failed event
            if call_session:
                try:
                    await broadcast_call_status_update(
                        call_session_id=str(call_session.id),
                        status="failed",
                        metadata={
                            "call_sid": call_sid,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    print(f"✅ Broadcasted call failed event for session {call_session.id}")
                    
                    # Also broadcast call ended event for failed calls
                    await broadcast_call_ended(
                        call_session_id=str(call_session.id),
                        reason="failed",
                        duration=0,
                        metadata={
                            "call_sid": call_sid,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    print(f"✅ Broadcasted call ended (failed) event for session {call_session.id}")
                except Exception as e:
                    print(f"❌ Failed to broadcast call failed event: {e}")
            
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "busy":
            # Call busy - handle busy signal
            print(f"Call busy - SID: {call_sid}")
            
            # Broadcast call busy event
            if call_session:
                try:
                    await broadcast_call_status_update(
                        call_session_id=str(call_session.id),
                        status="busy",
                        metadata={
                            "call_sid": call_sid,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    print(f"✅ Broadcasted call busy event for session {call_session.id}")
                    
                    # Also broadcast call ended event for busy calls
                    await broadcast_call_ended(
                        call_session_id=str(call_session.id),
                        reason="busy",
                        duration=0,
                        metadata={
                            "call_sid": call_sid,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    print(f"✅ Broadcasted call ended (busy) event for session {call_session.id}")
                except Exception as e:
                    print(f"❌ Failed to broadcast call busy event: {e}")
            
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "no-answer":
            # Call no-answer - handle no answer
            print(f"Call no-answer - SID: {call_sid}")
            
            # Broadcast call no-answer event
            if call_session:
                try:
                    await broadcast_call_status_update(
                        call_session_id=str(call_session.id),
                        status="no-answer",
                        metadata={
                            "call_sid": call_sid,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    print(f"✅ Broadcasted call no-answer event for session {call_session.id}")
                    
                    # Also broadcast call ended event for no-answer calls
                    await broadcast_call_ended(
                        call_session_id=str(call_session.id),
                        reason="no-answer",
                        duration=0,
                        metadata={
                            "call_sid": call_sid,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    print(f"✅ Broadcasted call ended (no-answer) event for session {call_session.id}")
                except Exception as e:
                    print(f"❌ Failed to broadcast call no-answer event: {e}")
            
            return HTMLResponse("", media_type="application/xml")
        
        else:
            # Default response for other statuses
            print(f"Unhandled call status: '{call_status}' - using default response")
            response = VoiceResponse()
            agent_voice = get_agent_voice(agent)
            response.say("Thanks for calling! Have a great day!", voice=agent_voice)
            return HTMLResponse(str(response), media_type="application/xml")
    
    except Exception as e:
        print(f"ERROR occurred: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print("Error traceback:")
        import traceback
        print(traceback.format_exc())
        print("=== Call Events Webhook Failed ===")
        raise



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


@router.get("/recording/{call_session_id}/access", response_model=SuccessResponse[dict])
async def get_recording_access(
    call_session_id: str,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Get direct access to call recording for authenticated users.
    Returns the authenticated Twilio recording URL.
    """
    try:
        # Get call session and verify user has access
        call_session = db.query(CallSession).filter(
            CallSession.id == call_session_id,
            CallSession.tenant_id == user.current_tenant_id
        ).first()
        
        if not call_session:
            raise HTTPException(status_code=404, detail="Call session not found or access denied")
        
        if not call_session.recording_url:
            raise HTTPException(status_code=404, detail="No recording available for this call")
        
        # Get Twilio credentials to create authenticated URL
        client = twilio_service.get_client()
        account_sid = client.username
        auth_token = client.password
        
        # Extract recording SID from the URL
        recording_sid = call_session.recording_url.split('/')[-1].replace('.mp3', '')
        
        # Create authenticated Twilio URL
        authenticated_url = f"https://{account_sid}:{auth_token}@api.twilio.com/2010-04-01/Accounts/{account_sid}/Recordings/{recording_sid}.mp3"
        
        # Return the URL in JSON response instead of redirecting
        return create_success_response(
            {
                "call_session_id": call_session_id,
                "recording_url": authenticated_url,
                "message": "Use the recording_url to access the call recording directly"
            },
            "Recording access URL generated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to access recording: {str(e)}")


# def _generate_default_response() -> str:
#     """Generate default TwiML response"""
#     response = VoiceResponse()
#     response.say("Thank you for calling. An agent will be with you shortly.", voice="")
#     response.pause(length=2)
#     response.say("Please hold while we connect you.", voice="")
#     return str(response)