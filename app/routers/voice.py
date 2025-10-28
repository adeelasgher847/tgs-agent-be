from fastapi import APIRouter, Request, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional
from twilio.twiml.voice_response import VoiceResponse, Start, Stream
from datetime import datetime, timezone
import random
import uuid
import sys
import requests
import asyncio

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
from app.services.transcript_service import transcript_service
from app.services.model_service import ModelService
from app.services.gemini_service import gemini_service
from app.services.credit_service import credit_service
from urllib.parse import quote
from app.routers.bidirectional_stream import build_streaming_twiml

router = APIRouter()

# Initialize services
model_service = ModelService()

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


async def _add_to_transcript(
    call_session, 
    role: str, 
    message: str, 
    db: Session, 
    message_type: str = "speech",
    agent_id: Optional[uuid.UUID] = None,
    user_id: Optional[uuid.UUID] = None,
    confidence: Optional[float] = None,
    duration: Optional[float] = None,
    response_time: Optional[float] = None,
    metadata: Optional[dict] = None
):
    """Add a message to the transcript using the new transcript service
    
    Args:
        call_session: The call session object
        role: Either "agent" or "client" 
        message: The message content
        db: Database session for committing changes
        message_type: Type of message (speech, timeout, error, etc.)
        confidence: Speech recognition confidence (0.0-1.0)
        duration: Message duration in seconds
        response_time: Time taken to generate response
        metadata: Additional message metadata
    """
    
    print(f"📝 Adding to transcript: {role} - {message[:50]}...")
    
    try:
        # Use the new transcript service
        transcript_message = await transcript_service.add_and_broadcast_message(
            db=db,
            call_session_id=call_session.id,
            role=role,
            message=message,
            message_type=message_type,
            agent_id=agent_id,
            user_id=user_id,
            confidence=confidence,
            duration=duration,
            response_time=response_time,
            metadata=metadata
        )
        
        print(f"✅ Added transcript message {transcript_message.id} for session {call_session.id}")
        
        # Also update the legacy call_transcript field for backward compatibility
        conversation = transcript_service.get_conversation_array(db, call_session.id)
        call_session.call_transcript = conversation
        db.commit()
        
        return transcript_message
        
    except Exception as e:
        print(f"❌ Failed to add transcript message: {e}")
        import traceback
        traceback.print_exc()
        raise


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


def get_gather_language(agent) -> str:
    """Get language code for Twilio Gather based on agent language"""
    if not agent or not agent.language:
        return "en-US"
    
    # Map agent language to Twilio supported languages
    language_map = {
        "en": "en-US",
        "es": "es-ES",
        "hi": "hi-IN",
        "ar": "ar-SA",
        "zh": "zh-CN",
        "ur": "ur-PK"
    }
    
    return language_map.get(agent.language, "en-US")


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


def add_media_stream_to_response(
    response: VoiceResponse,
    agent_id: str,
    call_session_id: str,
    track: str = "inbound_track"
) -> VoiceResponse:
    """
    Add media streaming to TwiML response for Google Cloud STT
    
    Args:
        response: VoiceResponse object
        agent_id: Agent ID
        call_session_id: Call session ID
        track: Which audio track to stream (inbound_track, outbound_track, both_tracks)
    
    Returns:
        Modified VoiceResponse with streaming enabled
    """
    # Build WebSocket URL for media streaming
    # Use wss:// for secure WebSocket connection
    ws_protocol = "wss" if "https" in settings.WEBHOOK_BASE_URL else "ws"
    ws_base = settings.WEBHOOK_BASE_URL.replace("https://", "").replace("http://", "")
    
    # Pass parameters as path segments instead of query params to avoid XML encoding issues
    ws_url = f"{ws_protocol}://{ws_base}/api/v1/stt/ws/media-stream/{call_session_id}/{agent_id}"
    
    print(f"🎙️ Adding media stream to TwiML: {ws_url}")
    
    # Start media streaming
    start = Start()
    stream = Stream(url=ws_url, track=track)
    start.append(stream)
    response.append(start)
    
    return response


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
        
        # Check credits before initiating call
        if not agent.model:
            raise HTTPException(status_code=400, detail="Agent does not have a model configured")
        
        model_name = agent.model.model_name
        has_sufficient, current_credits, required_credits = credit_service.has_sufficient_credits(
            db=db,
            tenant_id=user.current_tenant_id,
            model_name=model_name,
            estimated_minutes=1  # Check for at least 1 minute
        )
        
        if not has_sufficient:
            print(f"❌ Insufficient credits: {current_credits} < {required_credits}")
            raise HTTPException(
                status_code=402,  # Payment Required
                detail=f"Insufficient credits to initiate call. Current balance: {current_credits} credits, Required: {required_credits} credits. Model: {model_name}"
            )
        
        print(f"✅ Credit check passed: {current_credits} credits available, {required_credits} required for model {model_name}")
        
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
        
        # Don't broadcast "initiating" status - wait for proper webhook status
        # This ensures consistent status flow
        print(f"✅ Call initiated - waiting for proper webhook status updates")
        
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
        
        # Don't broadcast "initiated" here - let webhook handle all status updates
        print(f"✅ Call initiated - waiting for proper webhook status updates")
        
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
    timeout: Optional[str] = Query(None),
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
    
    # Optional WebSocket broadcast (non-blocking - fire and forget)
    try:
        asyncio.create_task(broadcast_system_notification(
            notification_type="webhook_started",
            message=f"Webhook started for call session {callSessionId}",
            metadata={
                "agent_id": agentId,
                "user_id": userId,
                "call_session_id": callSessionId,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ))
        print(f"✅ WebSocket broadcast queued at webhook start")
    except Exception as e:
        print(f"⚠️ WebSocket broadcast failed (non-critical): {e}")
        # Don't print traceback - this is not critical for call processing
    try:
        print("Parsing request body...")
        
        # Parse form data to get call information
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "")
        call_status = form_data.get("CallStatus", "")
        callback_event = form_data.get("CallStatusCallbackEvent", "")  # initiated | ringing | answered | completed
        answered_by = form_data.get("AnsweredBy", "")  # human | machine-start | etc.
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        direction = form_data.get("Direction", "")
        
        # Note: Speech input is now handled by Google Cloud STT via WebSocket
        # The old Twilio SpeechResult is no longer used
        # speech_result = form_data.get("SpeechResult", "")
        # confidence = form_data.get("Confidence", "")
        # speech_duration = form_data.get("SpeechDuration", "")
        
        print(f"🎤 Speech handling is now managed by Google Cloud STT WebSocket")
        
        # Get call session using callSessionId from query parameters (OPTIMIZED)
        call_session = None
        agent = None
        
        if callSessionId:
            try:
                session_uuid = uuid.UUID(callSessionId)
                call_session = call_session_service.get_call_session_by_id(db, session_uuid)
                if call_session:
                    print(f"✅ Found call session: {call_session.id} from query parameter")
                    
                    # Fetch agent using call session's tenant_id
                    if agentId:
                        agent = agent_service.get_agent_by_id(db, uuid.UUID(agentId), call_session.tenant_id)
                        if agent:
                            print(f"✅ Agent fetched: {agent.name} (ID: {agent.id})")
                            print(f"🏢 Tenant: {agent.tenant_id}")
                        else:
                            print(f"⚠️ Agent {agentId} not found in tenant {call_session.tenant_id}")
                else:
                    print(f"⚠️ No call session found for ID: {callSessionId}")
            except ValueError:
                print(f"⚠️ Invalid call session ID format: {callSessionId}")
        else:
            print(f"⚠️ No callSessionId provided in query parameters")
        
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
        
        # Log the call event with detailed Twilio information
        print("=" * 80)
        print(f"🔔 WEBHOOK RECEIVED - SID: {call_sid}")
        print(f"📊 CallStatus: {call_status}")
        print(f"📊 CallStatusCallbackEvent: {callback_event}")
        print(f"📊 AnsweredBy: {answered_by}")
        print(f"📊 Direction: {direction}")
        print(f"📊 From: {from_number} → To: {to_number}")
        print(f"📊 AgentId: {agentId}")
        print("=" * 80)
        
        # Test WebSocket connection if we have a call session (non-blocking - fire and forget)
        # if call_session:
        #     try:
        #         asyncio.create_task(broadcast_call_status_update(
        #             call_session_id=str(call_session.id),
        #             status="webhook_test",
        #             metadata={
        #                 "message": "Webhook is working",
        #                 "timestamp": datetime.now(timezone.utc).isoformat(),
        #                 "call_sid": call_sid
        #             }
        #         ))
        #         print(f"✅ Test broadcast queued to WebSocket for session {call_session.id}")
        #     except Exception as e:
        #         print(f"⚠️ Test broadcast failed (non-critical): {e}")
        
        # Status broadcasts will be handled in the main status update section below
        
        # Smart detection: Use multiple methods to detect when receiver picks up
        should_broadcast_connected = False
        
        # Method 1: Twilio answered event (most reliable)
        if call_session and callback_event == "answered":
            should_broadcast_connected = True
            print(f"✅ Method 1: Twilio answered event - broadcasting 'connected' status")
        
        # Method 2: in-progress + answered_by human (backup)
        elif call_session and call_status == "in-progress" and answered_by == "human":
            should_broadcast_connected = True
            print(f"✅ Method 2: in-progress + human answered - broadcasting 'connected' status")
        
        # Method 3: in-progress + previous status was ringing (fallback)
        elif call_session and call_status == "in-progress":
            previous_status = call_session.status
            if previous_status in ["ringing", "initiated"]:
                should_broadcast_connected = True
                print(f"✅ Method 3: in-progress + transition from {previous_status} - broadcasting 'connected' status")
            else:
                print(f"📊 Method 3: in-progress but previous status was {previous_status} - skipping")
        
        else:
            print(f"📊 No connection detected - callback_event: {callback_event}, call_status: {call_status}, answered_by: {answered_by}")
        
        # Update call session status if we have a call session and status
        if call_session and call_status:
            print(f"🔄 Updating call session {call_session.id} status to: {call_status}")
            
            # Map Twilio statuses to our internal statuses for better UX
            status_mapping = {
                "initiating": "initiating",
                "initiated": "initiated",
                "ringing": "ringing", 
                "in-progress": "connected",  # Map to "connected" for better UX
                "completed": "completed",
                "failed": "failed",
                "busy": "busy",
                "no-answer": "no-answer"
            }
            
            mapped_status = status_mapping.get(call_status, call_status)
            call_session.status = mapped_status
            
            # Set start time when call becomes connected (receiver picks up)
            if mapped_status == "connected" and not call_session.start_time:
                call_session.start_time = datetime.now(timezone.utc)
                print(f"⏰ Set start time for session {call_session.id}")
            
            # Set end time and calculate duration when call completes
            if mapped_status == "completed":
                call_session.end_time = datetime.now(timezone.utc)
                if call_session.start_time:
                    duration = (call_session.end_time - call_session.start_time).total_seconds()
                    call_session.duration = int(duration)
                    print(f"⏰ Set end time and duration ({duration}s) for session {call_session.id}")

                    # Broadcast call ended event (non-blocking - fire and forget)
                    try:
                        asyncio.create_task(broadcast_call_ended(
                            call_session_id=str(call_session.id),
                            reason="completed",
                            final_data={
                                "call_sid": call_sid,
                                "from_number": from_number,
                                "to_number": to_number,
                                "direction": direction,
                                "duration": call_session.duration,
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            }
                        ))
                        print(f"✅ Queued call ended event for session {call_session.id}")
                    except Exception as e:
                        print(f"⚠️ Failed to queue call ended event (non-critical): {e}")

                    # Stop credit monitoring when call completes
                    try:
                        credit_service.stop_credit_monitoring(call_session.id)
                        print(f"✅ Stopped credit monitoring for call session {call_session.id}")
                    except Exception as e:
                        print(f"⚠️ Failed to stop credit monitoring (non-critical): {e}")
                    
                    db.commit()

            # Broadcast status update to WebSocket for ALL statuses EXCEPT "connected" (handled separately)
            if mapped_status != "connected":
                try:
                    print(f"🚀 Broadcasting call status update: {mapped_status} for session {call_session.id}")

                    # Prepare comprehensive metadata
                    metadata = {
                        "from_number": from_number,
                        "to_number": to_number,
                        "direction": direction,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "start_time": call_session.start_time.isoformat() if call_session.start_time else None,
                        "end_time": call_session.end_time.isoformat() if call_session.end_time else None,
                        "duration": call_session.duration
                    }

                    # Add status-specific messages
                    if mapped_status == "initiating":
                        metadata["message"] = "Call is being initiated"
                    elif mapped_status == "initiated":
                        metadata["message"] = "Call has been initiated"
                    elif mapped_status == "ringing":
                        metadata["message"] = "Call is ringing"
                    elif mapped_status == "completed":
                        metadata["message"] = "Call has been completed"
                    elif mapped_status == "failed":
                        metadata["message"] = "Call failed"
                    elif mapped_status == "busy":
                        metadata["message"] = "Call busy"
                    elif mapped_status == "no-answer":
                        metadata["message"] = "No answer"

                    await broadcast_call_status_update(
                        call_session_id=str(call_session.id),
                        status=mapped_status,
                        metadata=metadata
                    )
                    print(f"✅ Call status update sent: {mapped_status} for session {call_session.id}")
                except Exception as e:
                    print(f"❌ Failed to broadcast call status update: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⏸️ Skipping generic broadcast for 'connected' - handled separately")

        # Also broadcast call ended event for completed calls (non-blocking - fire and forget)
        if mapped_status == "completed":
            asyncio.create_task(broadcast_call_ended(
                call_session_id=str(call_session.id),
                reason="Call completed",
                final_data={
                    "call_sid": call_sid,
                    "duration": call_session.duration,
                    "end_time": call_session.end_time.isoformat(),
                    "transcript": call_session.call_transcript or []
                }
            ))
            print(f"✅ Queued call ended event for session {call_session.id}")
        else:
            if not call_session:
                print(f"⚠️ No call session found - cannot update status or broadcast")
            if not call_status:
                print(f"⚠️ No call status provided - cannot update status or broadcast")
        
        # Speech input is now handled by Google Cloud STT via WebSocket
        # The WebSocket will transcribe audio and generate responses
        # This webhook now primarily handles call status updates and plays pending responses
        
        # Handle different call statuses and trigger agent logic
        print(f"Processing call status: '{call_status}' with direction: '{direction}'")
        
        if call_status == "initiated" and direction == "outbound-api":
            # Call has been initiated - broadcast "initiated" status
            print(f"Call initiated webhook received - SID: {call_sid}")
            
            # Broadcast "initiated" status
            if call_session:
                try:
                    await broadcast_call_status_update(
                        call_session_id=str(call_session.id),
                        status="initiated",
                        metadata={
                            "call_sid": call_sid,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "message": "Call has been initiated"
                        }
                    )
                    print(f"✅ Broadcasted call initiated status for session {call_session.id}")
                except Exception as e:
                    print(f"❌ Failed to broadcast call initiated status: {e}")
            
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "ringing" and direction == "outbound-api":
            # Outbound call is ringing - just log, don't play any audio
            print("=" * 50)
            print(f"🔔 CALL IS RINGING - SID: {call_sid}")
            print("=" * 50)
            
            # Broadcast call ringing event (non-blocking - fire and forget)
            if call_session:
                try:
                    asyncio.create_task(broadcast_call_status_update(
                        call_session_id=str(call_session.id),
                        status="ringing",
                        metadata={
                            "call_sid": call_sid,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    ))
                    print(f"✅ Broadcasted call ringing event for session {call_session.id}")
                except Exception as e:
                    print(f"❌ Failed to broadcast call ringing event: {e}")
            
            # Return empty response - no audio should play while ringing
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "in-progress":
            # Call is in progress - person answered, check if we already greeted
            print("=" * 50)
            print(f"📞 CALL IN PROGRESS - SID: {call_sid}")
            print(f"📊 CallStatusCallbackEvent: {callback_event}")
            print("=" * 50)
            
            # Use the should_broadcast_connected flag set earlier (before database update)
            print(f"📊 Using pre-calculated should_broadcast_connected: {should_broadcast_connected}")
            
            if should_broadcast_connected:
                print(f"✅ Call in progress - broadcasting 'connected' status")
                
                # Broadcast call connected event (non-blocking - fire and forget)
                if call_session:
                    try:
                        # Specific broadcast for in-progress
                        asyncio.create_task(broadcast_call_status_update(
                            call_session_id=str(call_session.id),
                            status="connected",  # Map to "connected" for better UX
                            metadata={
                                "call_sid": call_sid,
                                "direction": direction,
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            }
                        ))
                        print(f"✅ Broadcasted call connected event for session {call_session.id}")
                        
                        # ALSO do generic broadcast to ensure frontend gets it
                        asyncio.create_task(broadcast_call_status_update(
                            call_session_id=str(call_session.id),
                            status="connected",
                            metadata={
                                "from_number": from_number,
                                "to_number": to_number,
                                "direction": direction,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "message": "Call is now connected"
                            }
                        ))
                        print(f"✅ Broadcasted generic connected event for session {call_session.id}")
                        
                    except Exception as e:
                        print(f"❌ Failed to broadcast call connected event: {e}")
            else:
                print(f"⏸️ Skipping 'connected' broadcast - waiting for answered event (CallStatusCallbackEvent={callback_event})")
            
            # Start credit monitoring for the call (only when status is "in-progress")
            if call_session:
                try:
                    asyncio.create_task(credit_service.start_credit_monitoring(
                        db=db,
                        call_session_id=call_session.id,
                        tenant_id=call_session.tenant_id,
                        agent_id=call_session.agent_id
                    ))
                    print(f"✅ Started credit monitoring for call session {call_session.id} (30s intervals)")
                except Exception as e:
                    print(f"❌ Failed to start credit monitoring: {e}")
            
            # Get call session to check conversation state
            call_session = call_session_service.get_call_session_by_twilio_sid(db, call_sid)
            if not call_session:
                print(f"⚠️ Call session not found for SID: {call_sid}")
                return HTMLResponse("", media_type="application/xml")
            
            # Initialize conversation state for new calls (fixes infinite timeout loop)
            if not call_session.call_metadata:
                call_session.call_metadata = {}
            
            if "conversation_state" not in call_session.call_metadata:
                call_session.call_metadata["conversation_state"] = {
                    "has_greeted": False
                }
                db.commit()
                print("✅ Initialized conversation state for new call - STT will start properly")
            
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
            
            # Only start conversation if we haven't started yet
            if not has_greeted:
                # Check if we should use bidirectional streaming (TTS focus)
                use_streaming = getattr(settings, 'USE_BIDIRECTIONAL_STREAMING', True)
                
                if use_streaming:
                    # NEW: Bidirectional WebSocket streaming (USER SPEAKS FIRST!)
                    print("⚡ Using bidirectional streaming - USER SPEAKS FIRST (no agent greeting)")
                    
                    response = VoiceResponse()
                    response.redirect(
                        f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/gather/streaming?agentId={agentId}&userId={userId}&callSessionId={call_session.id}',
                        method='POST'
                    )
                    
                    # Mark as greeted (conversation started)
                    _update_conversation_state(call_session, "has_greeted", True)
                    _update_conversation_state(call_session, "greeting_time", datetime.now(timezone.utc).isoformat())
                    db.commit()
                    
                    print("✅ Redirecting to bidirectional streaming - TTS will stream in real-time")
                    return HTMLResponse(str(response), media_type="application/xml")
                
                else:
                    # Fallback: Use recording-based approach
                    print("🎤 Using recording-based approach - USER SPEAKS FIRST")
                    
                    # Mark as greeted
                    _update_conversation_state(call_session, "has_greeted", True)
                    _update_conversation_state(call_session, "greeting_time", datetime.now(timezone.utc).isoformat())
                    db.commit()
                    
                    # Start recording immediately - user speaks first
                    response = VoiceResponse()
                    recording_callback_url = f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/recording-callback?agentId={agentId}&userId={userId}&callSessionId={call_session.id}'
                    
                    response.record(
                        action=recording_callback_url,
                        method='POST',
                        timeout=3,  # Wait 3 seconds of silence (faster detection)
                        max_length=120,  # Max 2 minutes
                        play_beep=False,  # No beep for natural conversation
                        trim='do-not-trim',
                        recording_status_callback=recording_callback_url,
                        recording_status_callback_method='POST',
                        transcribe=False  # We use Google STT
                    )
                    
                    print("✅ Recording started - waiting for user to speak first")
                    return HTMLResponse(str(response), media_type="application/xml")
            else:
                print("🔄 ALREADY STARTED - Continuing with bidirectional streaming")
                
                # Check for pending response (shouldn't happen with streaming, but keep for safety)
                pending_response = None
                if call_session.call_metadata and "pending_response" in call_session.call_metadata:
                    pending_response = call_session.call_metadata.pop("pending_response")
                    db.commit()
                
                # Check if timeout redirect
                if timeout == "true":
                    print(f"⏱️ Timeout - ending call gracefully")
                    response = VoiceResponse()
                    text = pending_response if pending_response else "Thank you for calling. Goodbye!"
                    lang = agent.language if agent and agent.language else "en"
                    voice = agent.voice_type if agent and agent.voice_type else "female"
                    tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}"
                    response.play(tts_url)
                    response.hangup()
                    return HTMLResponse(str(response), media_type="application/xml")
                
                # If there's a pending response (edge case), play it
                if pending_response:
                    print(f"🎤 Playing pending response: {pending_response}")
                    response = VoiceResponse()
                    lang = agent.language if agent and agent.language else "en"
                    voice = agent.voice_type if agent and agent.voice_type else "female"
                    tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(pending_response)}&lang={lang}&voice={voice}"
                    response.play(tts_url)
                    
                    # Check if goodbye
                    is_goodbye = VoiceLoggingService._is_completion_goodbye(pending_response)
                    if is_goodbye:
                        response.hangup()
                        print(f"🛑 Goodbye detected - ending call")
                        return HTMLResponse(str(response), media_type="application/xml")
                    
                    # Continue with recording
                    recording_callback_url = f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/recording-callback?agentId={agentId}&userId={userId}&callSessionId={call_session.id}'
                    response.record(
                        action=recording_callback_url,
                        method='POST',
                        timeout=3,  # Faster speech detection
                        max_length=120,
                        play_beep=False,
                        trim='do-not-trim',
                        recording_status_callback=recording_callback_url,
                        recording_status_callback_method='POST',
                        transcribe=False
                    )
                    print(f"📝 Played pending response, continuing with recording")
                    return HTMLResponse(str(response), media_type="application/xml")
                
                # No pending response - redirect back to streaming
                response = VoiceResponse()
                response.redirect(
                    f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/gather/streaming?agentId={agentId}&userId={userId}&callSessionId={call_session.id}',
                    method='POST'
                )
                print(f"📝 Redirecting back to bidirectional streaming")
                return HTMLResponse(str(response), media_type="application/xml")
        
        elif call_status == "completed":
            # Call completed
            print(f"📞 CALL COMPLETED - SID: {call_sid}")
            
            # Broadcast call completed event (this is already handled above in the status update section)
            # The broadcast_call_ended is already called in the status update section above
            
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "failed":
            # Call failed - handle error
            print(f"Call failed - SID: {call_sid}")
            
            # Broadcast call failed event (non-blocking - fire and forget)
            if call_session:
                try:
                    asyncio.create_task(broadcast_call_status_update(
                        call_session_id=str(call_session.id),
                        status="failed",
                        metadata={
                            "call_sid": call_sid,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    ))
                    print(f"✅ Queued call failed event for session {call_session.id}")
                    
                    # Also broadcast call ended event for failed calls (non-blocking - fire and forget)
                    asyncio.create_task(broadcast_call_ended(
                        call_session_id=str(call_session.id),
                        reason="failed",
                        duration=0,
                        metadata={
                            "call_sid": call_sid,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    ))
                    print(f"✅ Queued call ended (failed) event for session {call_session.id}")
                except Exception as e:
                    print(f"❌ Failed to broadcast call failed event: {e}")
                
                # Stop credit monitoring when call fails
                try:
                    credit_service.stop_credit_monitoring(call_session.id)
                    print(f"✅ Stopped credit monitoring for failed call session {call_session.id}")
                except Exception as e:
                    print(f"⚠️ Failed to stop credit monitoring (non-critical): {e}")
            
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "busy":
            # Call busy - handle busy signal
            print(f"Call busy - SID: {call_sid}")
            
            # Broadcast call busy event (non-blocking - fire and forget)
            if call_session:
                try:
                    asyncio.create_task(broadcast_call_status_update(
                        call_session_id=str(call_session.id),
                        status="busy",
                        metadata={
                            "call_sid": call_sid,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    ))
                    print(f"✅ Queued call busy event for session {call_session.id}")
                    
                    # Also broadcast call ended event for busy calls (non-blocking - fire and forget)
                    asyncio.create_task(broadcast_call_ended(
                        call_session_id=str(call_session.id),
                        reason="busy",
                        duration=0,
                        metadata={
                            "call_sid": call_sid,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    ))
                    print(f"✅ Queued call ended (busy) event for session {call_session.id}")
                except Exception as e:
                    print(f"❌ Failed to broadcast call busy event: {e}")
                
                # Stop credit monitoring when call is busy
                try:
                    credit_service.stop_credit_monitoring(call_session.id)
                    print(f"✅ Stopped credit monitoring for busy call session {call_session.id}")
                except Exception as e:
                    print(f"⚠️ Failed to stop credit monitoring (non-critical): {e}")
            
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "no-answer":
            # Call no-answer - handle no answer
            print(f"Call no-answer - SID: {call_sid}")
            
            # Broadcast call no-answer event (non-blocking - fire and forget)
            if call_session:
                try:
                    asyncio.create_task(broadcast_call_status_update(
                        call_session_id=str(call_session.id),
                        status="no-answer",
                        metadata={
                            "call_sid": call_sid,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    ))
                    print(f"✅ Queued call no-answer event for session {call_session.id}")
                    
                    # Also broadcast call ended event for no-answer calls (non-blocking - fire and forget)
                    asyncio.create_task(broadcast_call_ended(
                        call_session_id=str(call_session.id),
                        reason="no-answer",
                        duration=0,
                        metadata={
                            "call_sid": call_sid,
                            "direction": direction,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    ))
                    print(f"✅ Queued call ended (no-answer) event for session {call_session.id}")
                except Exception as e:
                    print(f"❌ Failed to broadcast call no-answer event: {e}")
                
                # Stop credit monitoring when call has no-answer
                try:
                    credit_service.stop_credit_monitoring(call_session.id)
                    print(f"✅ Stopped credit monitoring for no-answer call session {call_session.id}")
                except Exception as e:
                    print(f"⚠️ Failed to stop credit monitoring (non-critical): {e}")
            
            return HTMLResponse("", media_type="application/xml")
        
        else:
            # Default response for other statuses
            print(f"Unhandled call status: '{call_status}' - using default response")
            response = VoiceResponse()
            text = "Thanks for calling! Have a great day!"
            lang = agent.language if agent and agent.language else "en"
            voice = agent.voice_type if agent and agent.voice_type else "female"
            tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}"
            response.play(tts_url)
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


@router.post("/webhook/recording-callback", response_class=HTMLResponse)
async def handle_recording_callback(
    request: Request,
    agentId: Optional[str] = Query(None),
    userId: Optional[str] = Query(None),
    callSessionId: Optional[str] = Query(None),
    body: str = Depends(get_request_body),
    db: Session = Depends(get_db)
):
    """
    VAPI-style Recording Callback Webhook
    
    When user stops speaking (silence detected), Twilio sends the recording here.
    We download it, transcribe with Google STT, generate LLM response, and return TwiML.
    
    This is the simple, synchronous approach similar to feature/openai branch.
    """
    print("=" * 80)
    print(f"🎙️ RECORDING CALLBACK WEBHOOK - VAPI-style")
    print(f"📞 Call Session: {callSessionId}")
    print(f"🤖 Agent: {agentId}")
    print("=" * 80)
    
    try:
        form_data = await request.form()
        
        # Extract recording details
        recording_url = form_data.get("RecordingUrl", "")
        recording_sid = form_data.get("RecordingSid", "")
        recording_duration = form_data.get("RecordingDuration", "0")
        call_sid = form_data.get("CallSid", "")
        recording_status = form_data.get("RecordingStatus", "")
        
        print(f"🎵 Recording URL: {recording_url}")
        print(f"📝 Recording SID: {recording_sid}")
        print(f"⏱️ Duration: {recording_duration}s")
        print(f"📊 Status: {recording_status}")
        sys.stdout.flush()
        
        # IMPORTANT: Twilio calls this webhook twice:
        # 1. 'action' callback (no status, has URL) - User finished speaking → PROCESS THIS for TTS
        # 2. 'recordingStatusCallback' (has status) - Recording processed → SKIP (just for logging)
        
        if recording_status:
            # This is a status callback, not the action callback
            # We don't need to return TTS here, just acknowledge
            print(f"ℹ️ Recording status callback (status={recording_status}) - acknowledging only, no TTS")
            sys.stdout.flush()
            return HTMLResponse("", media_type="application/xml")
        
        # If no recording URL at all, something is wrong
        if not recording_url:
            print(f"⚠️ No recording URL provided - cannot process")
            sys.stdout.flush()
            return HTMLResponse("", media_type="application/xml")
        
        # This is the 'action' callback - user finished speaking
        # Process this for TTS response
        print(f"✅ Action callback detected - processing for TTS response")
        sys.stdout.flush()
        
        # Get call session
        call_session = None
        agent = None
        
        if callSessionId:
            try:
                session_uuid = uuid.UUID(callSessionId)
                call_session = call_session_service.get_call_session_by_id(db, session_uuid)
                
                if call_session and agentId:
                    agent = agent_service.get_agent_by_id(db, uuid.UUID(agentId), call_session.tenant_id)
                    print(f"✅ Found call session and agent: {agent.name if agent else 'Unknown'}")
            except ValueError:
                print(f"⚠️ Invalid call session ID: {callSessionId}")
        
        # Process recording if available
        if recording_url and call_session:
            try:
                import requests
                
                # Get Twilio credentials for authenticated download
                client = twilio_service.get_client()
                account_sid = client.username
                auth_token = client.password
                
                # Build authenticated recording URL
                # Twilio recordings are usually at /Recordings/{RecordingSid}
                if not recording_url.startswith('http'):
                    # Relative URL - build full URL
                    auth_url = f"https://{account_sid}:{auth_token}@api.twilio.com{recording_url}.wav"
                else:
                    # Full URL - add auth
                    auth_url = recording_url.replace('https://api.twilio.com', f'https://{account_sid}:{auth_token}@api.twilio.com') + '.wav'
                
                print(f"📥 Downloading audio from Twilio...")
                
                # Download the recording
                audio_response = requests.get(auth_url, timeout=10)
                
                if audio_response.status_code != 200:
                    print(f"❌ Failed to download recording: HTTP {audio_response.status_code}")
                    raise Exception(f"Failed to download recording: HTTP {audio_response.status_code}")
                
                audio_content = audio_response.content
                print(f"✅ Downloaded {len(audio_content)} bytes of audio")
                
                # Get language from agent
                language_code = "en-US"
                if agent and hasattr(agent, 'language'):
                    language_map = {
                        "en": "en-US",
                        "es": "es-ES",
                        "hi": "hi-IN",
                        "ar": "ar-SA",
                        "zh": "zh-CN",
                        "ur": "ur-PK"
                    }
                    language_code = language_map.get(agent.language, "en-US")
                
                print(f"🎙️ Transcribing with Google Cloud STT (language: {language_code})...")
                
                # Transcribe with Google STT
                from app.services.google_stt_service import google_stt_service
                
                stt_result = await google_stt_service.transcribe_audio_chunk_streaming(
                    audio_content=audio_content,
                    language_code=language_code
                )
                
                transcript = stt_result.get("transcript", "").strip()
                confidence = stt_result.get("confidence", 0.0)
                
                print(f"📝 Google STT Transcript: '{transcript}'")
                print(f"📊 Confidence: {confidence:.2f}")
                
                # If we have a transcript, process it
                if transcript:
                    # Add user speech to transcript
                    await _add_to_transcript(
                        call_session,
                        "client",
                        transcript,
                        db,
                        message_type="speech",
                        confidence=confidence
                    )
                    
                    # Log voice interaction
                    await VoiceLoggingService.log_voice_interaction(
                        db=db,
                        call_session_id=call_session.id,
                        interaction_type="speech_input",
                        speech_text=transcript,
                        confidence=confidence,
                        duration=float(recording_duration) if recording_duration else None,
                        metadata={
                            "call_sid": call_sid,
                            "recording_sid": recording_sid,
                            "agent_id": str(agent.id) if agent else None,
                            "source": "google_stt"
                        }
                    )
                    
                    # Generate agent response using LLM
                    print(f"🤖 Generating agent response...")
                    response_text = await VoiceLoggingService.generate_agent_response(
                        speech_text=transcript,
                        confidence=confidence,
                        agent=agent,
                        db=db,
                        call_session_id=call_session.id
                    )
                    
                    print(f"✅ Agent response: '{response_text}'")
                    
                    # Add agent response to transcript
                    await _add_to_transcript(
                        call_session,
                        "agent",
                        response_text,
                        db,
                        message_type="agent_response"
                    )
                    
                    # Check if this is a goodbye
                    is_goodbye = VoiceLoggingService._is_completion_goodbye(response_text)
                    if is_goodbye:
                        print(f"🛑 Goodbye detected - ending call")
                        sys.stdout.flush()
                        response = VoiceResponse()
                        response.hangup()
                        twiml_str = str(response)
                        print(f"📤 Returning TwiML (goodbye): {twiml_str[:200]}...")
                        sys.stdout.flush()
                        return HTMLResponse(twiml_str, media_type="application/xml")
                    
                    # Store TTS text in call session metadata for WebSocket to retrieve
                    lang = agent.language if agent and agent.language else "en"
                    voice_type = agent.voice_type if agent and agent.voice_type else "female"
                    
                    if not call_session.call_metadata:
                        call_session.call_metadata = {}
                    
                    call_session.call_metadata["pending_tts"] = {
                        "text": response_text,
                        "lang": lang,
                        "voice": voice_type
                    }
                    db.commit()
                    
                    print(f"💾 Stored pending TTS in metadata: '{response_text[:50]}...'")
                    sys.stdout.flush()
                    
                    # Build TwiML for TTS-only WebSocket streaming + Recording
                    recording_callback_url = f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/recording-callback?agentId={agentId}&userId={userId}&callSessionId={callSessionId}'
                    
                    from app.routers.bidirectional_stream import build_tts_only_twiml
                    twiml_str = build_tts_only_twiml(
                        call_session_id=str(call_session.id),
                        agent_id=str(agent.id) if agent else agentId,
                        record_callback_url=recording_callback_url
                    )
                    
                    print(f"🎵 Returning TwiML with TTS WebSocket streaming")
                    print(f"📤 TwiML: {twiml_str[:200]}...")
                    sys.stdout.flush()
                    return HTMLResponse(twiml_str, media_type="application/xml")
                
                else:
                    # No transcript - ask user to repeat
                    print(f"⚠️ No transcript from Google STT")
                    response = VoiceResponse()
                    
                    # Natural "didn't catch that" response
                    text = _get_random_didnt_catch_response()
                    lang = agent.language if agent and agent.language else "en"
                    voice = agent.voice_type if agent and agent.voice_type else "female"
                    tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}"
                    response.play(tts_url)
                    
                    # Record again
                    recording_callback_url = f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/recording-callback?agentId={agentId}&userId={userId}&callSessionId={callSessionId}'
                    
                    response.record(
                        action=recording_callback_url,
                        method='POST',
                        timeout=3,  # Faster detection
                        max_length=60,
                        play_beep=False,
                        trim='do-not-trim',
                        recording_status_callback=recording_callback_url,
                        recording_status_callback_method='POST',
                        transcribe=False
                    )
                    
                    return HTMLResponse(str(response), media_type="application/xml")
            
            except Exception as e:
                print(f"❌ Error processing recording: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback response
                response = VoiceResponse()
                text = "Sorry, I had trouble hearing you. Could you please repeat that?"
                lang = agent.language if agent and agent.language else "en"
                voice = agent.voice_type if agent and agent.voice_type else "female"
                tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}"
                response.play(tts_url)
                
                recording_callback_url = f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/recording-callback?agentId={agentId}&userId={userId}&callSessionId={callSessionId}'
                
                response.record(
                    action=recording_callback_url,
                    method='POST',
                    timeout=3,  # Faster detection
                    max_length=60,
                    play_beep=False,
                    trim='do-not-trim',
                    recording_status_callback=recording_callback_url,
                    recording_status_callback_method='POST',
                    transcribe=False
                )
                
                return HTMLResponse(str(response), media_type="application/xml")
        
        # Fallback if no recording URL
        print(f"⚠️ No recording URL provided")
        response = VoiceResponse()
        text = "I didn't hear anything. Please try speaking again."
        lang = agent.language if agent and agent.language else "en"
        voice = agent.voice_type if agent and agent.voice_type else "female"
        tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}"
        response.play(tts_url)
        
        recording_callback_url = f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/recording-callback?agentId={agentId}&userId={userId}&callSessionId={callSessionId}'
        
        response.record(
            action=recording_callback_url,
            method='POST',
            timeout=3,  # Faster detection
            max_length=60,
            play_beep=False,
            trim='do-not-trim',
            recording_status_callback=recording_callback_url,
            recording_status_callback_method='POST',
            transcribe=False
        )
        
        return HTMLResponse(str(response), media_type="application/xml")
    
    except Exception as e:
        print(f"❌ Error in recording callback webhook: {e}")
        import traceback
        traceback.print_exc()
        
        # Ultimate fallback - use streaming TwiML if we have session info
        if call_session and agent:
            streaming_twiml = build_streaming_twiml(str(call_session.id), str(agent.id))
            return HTMLResponse(streaming_twiml, media_type="application/xml")
        else:
            # Fallback to simple response if no session info
            response = VoiceResponse()
            response.say("Sorry, something went wrong. Please try calling again later. Goodbye!")
            response.hangup()
            return HTMLResponse(str(response), media_type="application/xml")


@router.post("/webhook/gather-speech", response_class=HTMLResponse)
async def handle_gather_speech_webhook(
    request: Request,
    agentId: Optional[str] = Query(None),
    callSessionId: Optional[str] = Query(None),
    body: str = Depends(get_request_body),
    db: Session = Depends(get_db)
):
    """
    DEPRECATED: This endpoint was used for the old Gather-based approach.
    Now we use the simpler /webhook/recording-callback endpoint with <Record>.
    
    Keeping this for backward compatibility with feature/openai branch style.
    """
    print("=" * 80)
    print(f"⚠️ DEPRECATED: GATHER SPEECH WEBHOOK CALLED")
    print(f"Use /webhook/recording-callback instead")
    print("=" * 80)
    
    try:
        form_data = await request.form()
        
        call_sid = form_data.get("CallSid", "")
        recording_url = form_data.get("RecordingUrl", "")
        speech_result = form_data.get("SpeechResult", "")  # Twilio's transcription
        confidence = form_data.get("Confidence", "0")
        
        print(f"📞 Call SID: {call_sid}")
        print(f"🎤 Twilio Speech Result: {speech_result}")
        print(f"📊 Confidence: {confidence}")
        print(f"🎵 Recording URL: {recording_url}")
        
        # Get call session
        call_session = None
        if callSessionId:
            try:
                session_uuid = uuid.UUID(callSessionId)
                call_session = call_session_service.get_call_session_by_id(db, session_uuid)
                print(f"✅ Found call session: {call_session.id}")
            except ValueError:
                print(f"⚠️ Invalid call session ID: {callSessionId}")
        
        # Get agent
        agent = None
        if agentId and call_session:
            try:
                agent = agent_service.get_agent_by_id(db, uuid.UUID(agentId), call_session.tenant_id)
                print(f"✅ Agent: {agent.name}")
            except Exception as e:
                print(f"⚠️ Error fetching agent: {e}")
        
        # Download audio from Twilio recording
        if recording_url and call_session:
            try:
                import requests
                import base64
                
                # Get Twilio credentials
                client = twilio_service.get_client()
                account_sid = client.username
                auth_token = client.password
                
                # Download recording with authentication
                auth_url = f"https://{account_sid}:{auth_token}@api.twilio.com{recording_url}.wav"
                print(f"📥 Downloading audio from Twilio...")
                
                audio_response = requests.get(auth_url)
                audio_content = audio_response.content
                
                print(f"✅ Downloaded {len(audio_content)} bytes of audio")
                
                # Send to Google Cloud STT
                from app.services.google_stt_service import google_stt_service
                
                # Get language
                language_code = "en-US"
                if agent and hasattr(agent, 'language'):
                    language_map = {
                        "en": "en-US",
                        "es": "es-ES",
                        "hi": "hi-IN",
                        "ar": "ar-SA",
                        "zh": "zh-CN",
                        "ur": "ur-PK"
                    }
                    language_code = language_map.get(agent.language, "en-US")
                
                print(f"🎙️ Transcribing with Google Cloud STT (language: {language_code})...")
                
                # Transcribe with Google STT
                stt_result = await google_stt_service.transcribe_audio_chunk_streaming(
                    audio_content=audio_content,
                    language_code=language_code
                )
                
                google_transcript = stt_result.get("transcript", "")
                google_confidence = stt_result.get("confidence", 0.0)
                
                print(f"📝 Google STT Transcript: '{google_transcript}'")
                print(f"📊 Google STT Confidence: {google_confidence:.2f}")
                
                # Use Google transcript (more accurate)
                final_transcript = google_transcript if google_transcript else speech_result
                
                if final_transcript:
                    # Add to transcript
                    await _add_to_transcript(
                        call_session, 
                        "client", 
                        final_transcript, 
                        db,
                        message_type="speech",
                        confidence=google_confidence
                    )
                    
                    # Generate LLM response
                    response_text = await VoiceLoggingService.generate_agent_response(
                        speech_text=final_transcript,
                        confidence=google_confidence,
                        agent=agent,
                        db=db,
                        call_session_id=call_session.id
                    )
                    
                    # Add agent response to transcript
                    await _add_to_transcript(
                        call_session,
                        "agent",
                        response_text,
                        db,
                        message_type="agent_response"
                    )
                    
                    print(f"✅ Generated agent response: '{response_text}'")
                    
                    # Create response TwiML
                    response = VoiceResponse()
                    
                    # Say agent response using Google TTS
                    lang = agent.language if agent and agent.language else "en"
                    voice = agent.voice_type if agent and agent.voice_type else "female"
                    tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(response_text)}&lang={lang}&voice={voice}"
                    response.play(tts_url)
                    
                    # Check if goodbye
                    is_goodbye = VoiceLoggingService._is_completion_goodbye(response_text)
                    if is_goodbye:
                        response.hangup()
                        print(f"🛑 Goodbye detected - ending call")
                        return HTMLResponse(str(response), media_type="application/xml")
                    
                    # Continue conversation - gather next input
                    gather = response.gather(
                        input='speech',
                        timeout=10,
                        speech_timeout='auto',
                        action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/gather-speech?agentId={agentId}&callSessionId={call_session.id}',
                        method='POST',
                        enhanced=True,
                        profanity_filter=False,
                        language=get_gather_language(agent)
                    )
                    
                    # Fallback
                    text = "I didn't catch that. Please try again!"
                    lang = agent.language if agent and agent.language else "en"
                    voice = agent.voice_type if agent and agent.voice_type else "female"
                    tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}"
                    response.play(tts_url)
                    response.redirect(
                        f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/call-events?agentId={agentId}&callSessionId={call_session.id}',
                        method='POST'
                    )
                    
                    print(f"📝 Response TwiML: {str(response)[:200]}...")
                    return HTMLResponse(str(response), media_type="application/xml")
            
            except Exception as e:
                print(f"❌ Error processing gathered speech: {e}")
                import traceback
                traceback.print_exc()
        
        # Fallback response
        response = VoiceResponse()
        text = "I didn't hear you. Could you please repeat that?"
        lang = agent.language if agent and agent.language else "en"
        voice = agent.voice_type if agent and agent.voice_type else "female"
        tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}"
        response.play(tts_url)
        
        gather = response.gather(
            input='speech',
            timeout=10,
            speech_timeout='auto',
            action=f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/gather-speech?agentId={agentId}&callSessionId={call_session.id}',
            method='POST',
            enhanced=True,
            profanity_filter=False,
            language=get_gather_language(agent)
        )
        
        return HTMLResponse(str(response), media_type="application/xml")
    
    except Exception as e:
        print(f"❌ Error in gather speech webhook: {e}")
        import traceback
        traceback.print_exc()
        raise


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
                    
                    # Broadcast call status update when recording is completed (non-blocking - fire and forget)
                    try:
                        asyncio.create_task(broadcast_call_status_update(
                            call_session_id=str(call_session.id),
                            status="completed",
                            metadata={
                                "call_sid": call_sid,
                                "call_duration": recording_duration,
                                "message": "Call completed",
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            }
                        ))
                        print(f"✅ Queued recording completed status update for session {call_session.id}")
                    except Exception as e:
                        print(f"⚠️ Failed to queue recording completed status update (non-critical): {e}")
                else:
                    print(f"📝 Recording status: {recording_status} - URL not ready yet")
            else:
                print(f"⚠️ Call session not found for SID: {call_sid}")
        
        # Return empty TwiML response
        return HTMLResponse("", media_type="application/xml")
        
    except Exception as e:
        print(f"⚠️ Error handling recording status webhook: {e}")
        return HTMLResponse("", media_type="application/xml")


@router.post("/call/end", response_model=SuccessResponse[dict])
async def end_call(
    request: dict,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    End a call programmatically
    
    Request Payload:
    {
        "callSessionId": "uuid",
        "reason": "user_requested" | "agent_completed" | "timeout" | "error",
        "message": "Optional goodbye message"
    }
    """
    try:
        call_session_id = request.get("callSessionId")
        reason = request.get("reason", "user_requested")
        goodbye_message = request.get("message", "Thank you for calling! Have a great day!")
        
        if not call_session_id:
            raise HTTPException(status_code=400, detail="callSessionId is required")
        
        # Get call session
        try:
            session_uuid = uuid.UUID(call_session_id)
            call_session = call_session_service.get_call_session_by_id(db, session_uuid)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid callSessionId format")
        
        if not call_session:
            raise HTTPException(status_code=404, detail="Call session not found")
        
        # Verify user has access to this call session
        if call_session.tenant_id != user.current_tenant_id:
            raise HTTPException(status_code=403, detail="Access denied to this call session")
        
        # End the call using Twilio API if we have the call SID
        call_ended = False
        if call_session.twilio_call_sid:
            call_ended = twilio_service.end_call(call_session.twilio_call_sid)
        
        # Update call session status
        call_session.status = "completed"
        call_session.end_time = datetime.now(timezone.utc)
        call_session.ended_reason = reason
        
        if call_session.start_time:
            duration = (call_session.end_time - call_session.start_time).total_seconds()
            call_session.duration = int(duration)
        
        db.commit()
        
        # Add goodbye message to transcript
        if goodbye_message:
            await _add_to_transcript(
                call_session,
                "agent",
                goodbye_message,
                db,
                message_type="call_end",
                agent_id=call_session.agent_id,
                user_id=call_session.user_id
            )
        
        # Broadcast call ended event
        try:
            asyncio.create_task(broadcast_call_ended(
                call_session_id=str(call_session.id),
                reason=reason,
                final_data={
                    "call_sid": call_session.twilio_call_sid,
                    "duration": call_session.duration,
                    "end_time": call_session.end_time.isoformat(),
                    "transcript": call_session.call_transcript or []
                }
            ))
        except Exception as e:
            print(f"⚠️ Failed to broadcast call ended event: {e}")
        
        return SuccessResponse(
            data={
                "callSessionId": str(call_session.id),
                "status": "completed",
                "reason": reason,
                "duration": call_session.duration,
                "twilioEnded": call_ended
            },
            message="Call ended successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error ending call: {e}")
        raise HTTPException(status_code=500, detail="Failed to end call")


@router.get("/recording/{call_session_id}/access")
async def get_recording_access(
    call_session_id: str,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Stream call recording directly to user (NO Twilio login required!)
    Returns audio file that can be played directly in browser.
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
        
        # Get Twilio credentials to download recording server-side
        client = twilio_service.get_client()
        account_sid = client.username
        auth_token = client.password
        
        # Extract recording SID from the URL
        recording_sid = call_session.recording_url.split('/')[-1].replace('.mp3', '').replace('.wav', '')
        
        # Create authenticated Twilio URL for server-side download
        authenticated_url = f"https://{account_sid}:{auth_token}@api.twilio.com/2010-04-01/Accounts/{account_sid}/Recordings/{recording_sid}.mp3"
        
        print(f"📥 Streaming recording for call session: {call_session_id}")
        print(f"🎵 Recording SID: {recording_sid}")
        
        # Download recording from Twilio (server-side with auth)
        response = requests.get(authenticated_url, stream=True, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Failed to fetch recording: HTTP {response.status_code}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to fetch recording from Twilio: HTTP {response.status_code}"
            )
        
        print(f"✅ Streaming recording to user (no login required)")
        
        # Stream audio directly to user (NO authentication required on user's end!)
        return StreamingResponse(
            response.iter_content(chunk_size=8192),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"inline; filename=call_recording_{call_session_id}.mp3",
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                "Accept-Ranges": "bytes"  # Enable seeking in audio player
            }
        )
        
    except HTTPException:
        raise
    except requests.RequestException as e:
        print(f"❌ Network error fetching recording: {e}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        print(f"❌ Error streaming recording: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stream recording: {str(e)}")


@router.post("/transcript/analyze/{call_session_id}", response_model=SuccessResponse[dict])
async def analyze_call_transcript(
    call_session_id: str,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Analyze call transcript using Gemini for summary and sentiment analysis
    
    Args:
        call_session_id: UUID of the call session
        user: Current authenticated user
        db: Database session
        
    Returns:
        Analysis results including summary and sentiment
    """
    try:
        # Static model name for analysis
        model_name = "gemini-2.0-flash"
        
        # Validate call session ID
        try:
            session_uuid = uuid.UUID(call_session_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid call session ID format")
        
        # Get call session
        call_session = call_session_service.get_call_session_by_id(db, session_uuid)
        if not call_session:
            raise HTTPException(status_code=404, detail="Call session not found")
        
        # Check if user has access to this call session
        if call_session.user_id != user.id and call_session.tenant_id != user.current_tenant_id:
            raise HTTPException(status_code=403, detail="Access denied to this call session")
        
        # Get model information by static name
        model = model_service.get_model_by_name(db, model_name)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in database")
        
        print(f"🔍 Model found: {model.model_name}, Provider: {model.provider.name}")
        
        # Check if model is a Gemini model
        provider_name = (model.provider.name or "").strip().lower()
        print(f"🔍 Provider name (normalized): '{provider_name}'")
        
        if provider_name not in ("gemini", "google", "google-ai", "google ai", "gemini-1.5-flash", "gemini-2.0-flash"):
            print(f"❌ Provider '{provider_name}' not recognized as Gemini model")
            raise HTTPException(status_code=400, detail=f"Model must be a Gemini model for analysis. Provider: {model.provider.name}")
        
        print(f"✅ Model validated as Gemini model: {model.model_name}")
        
        # Get transcript messages
        transcript_messages = transcript_service.get_messages_by_session(db, session_uuid)
        print(f"🔍 Found {len(transcript_messages)} transcript messages for session {call_session_id}")
        
        if not transcript_messages:
            raise HTTPException(status_code=404, detail="No transcript messages found for this call session")
        
        # Format transcript for analysis
        transcript_text = ""
        for msg in transcript_messages:
            role_label = "Agent" if msg.role == "agent" else "Customer"
            transcript_text += f"{role_label}: {msg.message}\n"
        
        # Create analysis prompts
        summary_prompt = f"""
        Analyze this call transcript and provide a brief summary in 2-3 sentences.
        
        Call Transcript:
        {transcript_text}
        
        Provide only:
        - Brief call overview
        - Main topic/issue
        - Outcome/resolution
        
        Keep it concise and to the point.
        """
        
        sentiment_prompt = f"""
        Analyze the sentiment of this call transcript and provide a brief assessment.
        
        Call Transcript:
        {transcript_text}
        
        Provide only:
        - Overall sentiment (positive/negative/neutral)
        - Sentiment score (0-100)
        - Customer satisfaction level (high/medium/low)
        
        Keep it brief and concise.
        """
        
        # Get model API key
        model_api_key = None
        if model.api_key:
            from app.core.security import decrypt_api_key
            model_api_key = decrypt_api_key(model.api_key)
        
        # Perform analysis with Gemini
        try:
            # Generate summary
            summary_result = gemini_service.generate_text(
                prompt=summary_prompt,
                model_name=model.model_name,
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=200,  # Reduced for brief responses
                api_key=model_api_key
            )
            
            # Generate sentiment analysis
            sentiment_result = gemini_service.generate_text(
                prompt=sentiment_prompt,
                model_name=model.model_name,
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=150,  # Reduced for brief responses
                api_key=model_api_key
            )
            
            # Prepare response (hide model_id for security)
            analysis_result = {
                "call_session_id": call_session_id,
                "transcript_message_count": len(transcript_messages),
                "call_duration": call_session.duration,
                "call_status": call_session.status,
                "analysis": {
                    "summary": summary_result["content"].strip(),
                    "sentiment": sentiment_result["content"].strip()
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            print(f"✅ Transcript analysis completed for session {call_session_id}")
            return create_success_response(
                data=analysis_result,
                message="Transcript analysis completed successfully"
            )
            
        except Exception as e:
            print(f"❌ Error during Gemini analysis: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in transcript analysis endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# def _generate_default_response() -> str:
#     """Generate default TwiML response"""
#     response = VoiceResponse()
#     response.say("Thank you for calling. An agent will be with you shortly.", voice="")
#     response.pause(length=2)
#     response.say("Please hold while we connect you.", voice="")
#     return str(response)