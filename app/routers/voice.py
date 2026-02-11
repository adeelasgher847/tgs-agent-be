from fastapi import APIRouter, Request, HTTPException, Query, Depends, status, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import Optional
from twilio.twiml.voice_response import VoiceResponse
from datetime import datetime, timezone
import random
import uuid
import sys
import requests
import asyncio
import csv
import io
from app.core.logger import logger

from app.api.deps import get_db, require_tenant, get_optional_tenant_user
from app.schemas.twilio import CallInitiateRequest, CallInitiateResponse, CallInitiateErrorResponse
from app.schemas.base import SuccessResponse
from app.services.twilio_service import twilio_service
from app.services.agent_service import agent_service
from app.models.agent import Agent
from app.models.user import User
from app.utils.n8n_webhook_verification import verify_n8n_webhook_secret_async
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
from app.services.phone_number_service import phone_number_service

router = APIRouter()

# Initialize services
model_service = ModelService()

def get_twilio_credentials_for_call(db: Session, call_session: CallSession):
    """
    Get Twilio credentials for a call session.
    Priority: DB phone number credentials > Env credentials
    
    Returns:
        tuple: (account_sid, auth_token)
    """
    from app.models.phone_number import PhoneNumber
    from app.core.security import decrypt_api_key
    
    # Check if call was made with DB phone number
    if call_session.from_number:
        phone_number_obj = db.query(PhoneNumber).filter(
            PhoneNumber.phone_number == call_session.from_number,
            PhoneNumber.tenant_id == call_session.tenant_id,
            PhoneNumber.status == "active"
        ).first()
        
        if phone_number_obj and phone_number_obj.twilio_account_sid and phone_number_obj.twilio_auth_token:
            # ✅ Use DB credentials (decrypt both)
            account_sid = decrypt_api_key(phone_number_obj.twilio_account_sid)
            auth_token = decrypt_api_key(phone_number_obj.twilio_auth_token)
            logger.info(f"✅ Using DB credentials for recording (phone: {call_session.from_number})")
            return account_sid, auth_token
    
    # ✅ Fallback to env credentials
    client = twilio_service.get_client()
    account_sid = client.username
    auth_token = client.password
    logger.info(f"✅ Using env credentials for recording")
    return account_sid, auth_token

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
    
    logger.debug(f"📝 Adding to transcript: {role} - {message[:50]}...")
    
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
        
        logger.debug(f"✅ Added transcript message {transcript_message.id} for session {call_session.id}")
        
        # Also update the legacy call_transcript field for backward compatibility
        conversation = transcript_service.get_conversation_array(db, call_session.id)
        call_session.call_transcript = conversation
        db.commit()
        
        return transcript_message
        
    except Exception as e:
        logger.error(f"❌ Failed to add transcript message: {e}", exc_info=True)
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
    
    logger.debug(f"🎤 Agent voice selection: language={language}, voice_type={voice_type}, selected_voice={selected_voice}")
    
    return selected_voice

@router.post("/call/initiate", response_model=SuccessResponse[CallInitiateResponse])
async def initiate_call(
    call_request: CallInitiateRequest,
    http_request: Request,
    user: Optional[User] = Depends(get_optional_tenant_user),
    db: Session = Depends(get_db)
):
    """
    Endpoint to initiate a voice call using Twilio.
    
    Authentication: Either JWT token OR n8n webhook secret (X-N8N-Webhook-Secret header).
    If using webhook secret, provide tenant_id (and optionally user_id) in request body.
    
    Request Payload (JWT auth - Normal Call):
    {
        "agentId": "agent_12345",
        "userPhoneNumber": "+1234567890"
    }
    
    Request Payload (n8n webhook - Scheduled Call):
    {
        "agentId": "agent_12345",
        "userPhoneNumber": "+1234567890",
        "tenant_id": "tenant-uuid",
        "user_id": "user-uuid" (optional),
        
        // Legacy Monday.com fields (for backward compatibility)
        "board_id": "board_123",
        "monday_item_id": "item_456",
        "status_column_id": "status_col_789",
        "call_session_id_column_id": "session_col_012",
        
        // OR Generic CRM fields (for multi-CRM support)
        "crm_container_id": "board_123",  // board_id/list_id/project_id
        "crm_item_id": "item_456",  // item_id/task_id/issue_id/card_id
        "status_field_id": "status_col_789",  // status field ID
        "call_session_id_field_id": "session_col_012",  // call_session_id field ID
        "crm_type": "monday"  // "monday" | "clickup" | "jira" | "trello"
    }
    
    Response (includes both legacy and generic fields for n8n compatibility):
    {
        "callId": "call_abc123",
        "twilioCallSid": "CAxxxxxxx",
        "status": "initiated",
        "board_id": "board_123",  // Legacy field (if provided)
        "monday_item_id": "item_456",  // Legacy field (if provided)
        "crm_container_id": "board_123",  // Generic field (if provided)
        "crm_item_id": "item_456",  // Generic field (if provided)
        "crm_type": "monday"  // Generic field (if provided)
    }
    
    Note: Normal calls (JWT auth) don't need CRM fields. Only n8n scheduled calls need them.
    """
    try:
        # Verify authentication: either JWT token OR webhook secret
        is_webhook = await verify_n8n_webhook_secret_async(http_request)
        
        if is_webhook:
            # Webhook authentication - get tenant_id and user_id from request body
            if not call_request.tenant_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="tenant_id is required in request body when using webhook secret"
                )
            try:
                tenant_uuid = uuid.UUID(call_request.tenant_id)
                user_uuid = uuid.UUID(call_request.user_id) if call_request.user_id else None
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid UUID format for tenant_id or user_id"
                )
            tenant_id_filter = tenant_uuid
            user_id_filter = user_uuid
        else:
            # JWT authentication - get from user token
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required: JWT token or n8n webhook secret"
                )
            tenant_id_filter = user.current_tenant_id
            user_id_filter = user.id
        
        # Validate agent exists in database
        try:
            agent_id = uuid.UUID(call_request.agentId)
            agent = agent_service.get_agent_by_id(db, agent_id, tenant_id_filter)
        except (ValueError, HTTPException):
            raise HTTPException(status_code=404, detail=f"Agent {call_request.agentId} not found")
        
        # Validate phone number format
        if not twilio_service.validate_phone_number(call_request.userPhoneNumber):
            raise HTTPException(status_code=400, detail="Invalid phone number format. Must start with +")
        
        # Check credits before initiating call
        if not agent.model:
            raise HTTPException(status_code=400, detail="Agent does not have a model configured")
        
        model_name = agent.model.model_name
        has_sufficient, current_credits, required_credits = credit_service.has_sufficient_credits(
            db=db,
            tenant_id=tenant_id_filter,
            model_name=model_name,
            estimated_minutes=1  # Check for at least 1 minute
        )
        
        if not has_sufficient:
            logger.warning(f"❌ Insufficient credits: {current_credits} < {required_credits}")
            error_message = f"Insufficient credits to initiate call. Current balance: {current_credits} credits, Required: {required_credits} credits. Model: {model_name}"
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=error_message
            )
        
        logger.info(f"✅ Credit check passed: {current_credits} credits available, {required_credits} required for model {model_name}")
        
        # Get phone number and credentials - Priority: User Selected > Agent Assigned > Env
        from app.models.phone_number import PhoneNumber
        
        phone_number_obj = None
        from_number = None
        use_custom_credentials = False
        account_sid = None
        auth_token = None
        
        # Priority 1: Check if user explicitly selected a phone number (VAPI style)
        if call_request.phone_number_id:
            try:
                phone_number_uuid = uuid.UUID(call_request.phone_number_id)
                phone_number_obj = phone_number_service.get_phone_number_by_id(
                    db=db,
                    phone_number_id=phone_number_uuid,
                    tenant_id=tenant_id_filter
                )
                if phone_number_obj and phone_number_obj.status == "active":
                    logger.info(f"✅ Using user selected phone number: {phone_number_obj.phone_number} (ID: {phone_number_uuid})")
                elif phone_number_obj and phone_number_obj.status != "active":
                    # ✅ Phone number exists but is inactive - raise error
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Phone number {call_request.phone_number_id} is not active."
                    )
                else:
                    # ✅ Phone number not found or belongs to different tenant - raise error
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Phone number {call_request.phone_number_id} not found in your account."
                    )
            except HTTPException:
                raise
            except (ValueError, Exception) as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid phone_number_id format: {str(e)}"
                )
        
        # Priority 2: Check if agent has assigned phone number in DB
        if not phone_number_obj and agent.id:
            phone_number_obj = db.query(PhoneNumber).filter(
                PhoneNumber.assistant_id == agent.id,
                PhoneNumber.tenant_id == tenant_id_filter,
                PhoneNumber.status == "active"
            ).first()
            if phone_number_obj:
                logger.info(f"✅ Using agent's assigned phone number: {phone_number_obj.phone_number}")
        
        # Use selected phone number with credentials if available
        if phone_number_obj and phone_number_obj.twilio_account_sid and phone_number_obj.twilio_auth_token:
            # ✅ Use DB phone number with custom credentials (decrypt both)
            from_number = phone_number_obj.phone_number
            from app.core.security import decrypt_api_key
            account_sid = decrypt_api_key(phone_number_obj.twilio_account_sid)
            auth_token = decrypt_api_key(phone_number_obj.twilio_auth_token)
            use_custom_credentials = True
            logger.info(f"✅ Using DB phone number: {from_number} with custom credentials")
        else:
            # ✅ No fallback - user must have a phone number in DB
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No phone number found. Please create and assign a phone number in your account before making calls."
            )
        
        # Get base URL for webhooks
        base_url = settings.WEBHOOK_BASE_URL
        
        # Create call session first so we can include the ID in webhook URLs
        call_session = call_session_service.create_call_session(
            db=db,
            user_id=user_id_filter,
            agent_id=agent.id,
            tenant_id=tenant_id_filter,
            twilio_call_sid="",  # Will be updated after call is made
            from_number=from_number,  # ✅ Use selected phone number
            to_number=call_request.userPhoneNumber,
            call_type="outbound"  # Agent is initiating the call, so it's outbound
        )
        
        # Direct WebSocket streaming connection (Vapi-style - no intermediate messages!)
        # User speaks first, agent responds naturally
        webhook_url = f"{base_url}/api/v1/voice/gather/streaming?agentId={agent.id}&userId={user_id_filter}&callSessionId={call_session.id}"
        status_callback_url = f"{base_url}/api/v1/voice/webhook/call-events?agentId={agent.id}&userId={user_id_filter}&callSessionId={call_session.id}"
        
        logger.info(f"Making call with webhook_url: {webhook_url}")
        logger.info(f"Making call with status_callback_url: {status_callback_url}")
        
        # Optional WebSocket broadcast
        try:
            await broadcast_call_status_update(
                call_session_id=str(call_session.id),
                status="initiating",
                metadata={
                    "agent_id": str(agent.id),
                    "agent_name": agent.name,
                    "to_number": call_request.userPhoneNumber,
                    "from_number": from_number,  # ✅ Use selected phone number
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            logger.info(f"✅ WebSocket: Call initiating event sent")
        except Exception as e:
            logger.warning(f"⚠️ WebSocket broadcast failed (non-critical): {e}")
        
        # Make call with appropriate credentials
        if use_custom_credentials:
            # ✅ Use custom credentials from DB
            call = twilio_service.make_call_with_credentials(
                to_number=call_request.userPhoneNumber,
                from_number=from_number,
                webhook_url=webhook_url,
                status_callback_url=status_callback_url,
                account_sid=account_sid,
                auth_token=auth_token
            )
        else:
            # ✅ Use env credentials (current behavior)
            call = twilio_service.make_call(
                to_number=call_request.userPhoneNumber,
                from_number=from_number,
                webhook_url=webhook_url,
                status_callback_url=status_callback_url
            )
        logger.info(f"✅ Call initiated successfully")
        
        # Update call session with Twilio SID
        call_session.twilio_call_sid = call.sid
        db.commit()
        logger.info(f"✅ Updated call session {call_session.id} with Twilio SID: {call.sid}")
        
        # Broadcast call initiated event AFTER Twilio confirms
        try:
            await broadcast_call_status_update(
                call_session_id=str(call_session.id),
                status="initiated",
                metadata={
                    "call_sid": call.sid,
                    "agent_id": str(agent.id),
                    "agent_name": agent.name,
                    "to_number": call_request.userPhoneNumber,
                    "from_number": twilio_service.get_phone_number(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            logger.info(f"✅ Call initiated event sent for session {call_session.id}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to send call initiated event (non-critical): {e}")
        
        # Generate call ID
        call_id = f"call_{call.sid[-8:]}"
        
        # Determine which fields to echo back (prioritize generic fields, fallback to legacy)
        crm_container_id = call_request.crm_container_id or call_request.board_id
        crm_item_id = call_request.crm_item_id or call_request.monday_item_id
        status_field_id = call_request.status_field_id or call_request.status_column_id
        call_session_id_field_id = call_request.call_session_id_field_id or call_request.call_session_id_column_id
        
        return create_success_response(
            CallInitiateResponse(
                callId=call_id,
                twilioCallSid=call.sid,
                callSessionId=str(call_session.id),
                status="initiated",
                # Legacy Monday.com fields (for backward compatibility)
                board_id=call_request.board_id,  # Echo back if provided by n8n
                monday_item_id=call_request.monday_item_id,  # Echo back if provided by n8n
                status_column_id=call_request.status_column_id,  # Echo back if provided by n8n
                call_session_id_column_id=call_request.call_session_id_column_id,  # Echo back if provided by n8n
                # Generic CRM fields (for multi-CRM support)
                crm_container_id=crm_container_id,  # Echo back generic container ID
                crm_item_id=crm_item_id,  # Echo back generic item ID
                status_field_id=status_field_id,  # Echo back generic status field ID
                call_session_id_field_id=call_session_id_field_id,  # Echo back generic call_session_id field ID
                crm_type=call_request.crm_type  # Echo back CRM type if provided
            ),
            "Call initiated successfully"
        )
        
    except HTTPException as e:
        # Return error response with CRM metadata (same format as success response)
        # This allows n8n workflow to access CRM fields even on errors
        # Prioritize generic fields, fallback to legacy
        crm_container_id = call_request.crm_container_id or call_request.board_id
        crm_item_id = call_request.crm_item_id or call_request.monday_item_id
        status_field_id = call_request.status_field_id or call_request.status_column_id
        call_session_id_field_id = call_request.call_session_id_field_id or call_request.call_session_id_column_id
        
        error_response = CallInitiateErrorResponse(
            detail=e.detail,
            # Legacy Monday.com fields (for backward compatibility)
            board_id=call_request.board_id,
            monday_item_id=call_request.monday_item_id,
            status_column_id=call_request.status_column_id,
            call_session_id_column_id=call_request.call_session_id_column_id,
            # Generic CRM fields (for multi-CRM support)
            crm_container_id=crm_container_id,
            crm_item_id=crm_item_id,
            status_field_id=status_field_id,
            call_session_id_field_id=call_session_id_field_id,
            crm_type=call_request.crm_type
        )
        # Return JSONResponse with same status code as HTTPException
        return JSONResponse(
            status_code=e.status_code,
            content=error_response.dict(exclude_none=True)
        )
    except Exception as e:
        # Handle unexpected errors - also include metadata if available
        crm_container_id = call_request.crm_container_id or call_request.board_id
        crm_item_id = call_request.crm_item_id or call_request.monday_item_id
        status_field_id = call_request.status_field_id or call_request.status_column_id
        call_session_id_field_id = call_request.call_session_id_field_id or call_request.call_session_id_column_id
        
        error_response = CallInitiateErrorResponse(
            detail=str(e),
            # Legacy Monday.com fields (for backward compatibility)
            board_id=call_request.board_id,
            monday_item_id=call_request.monday_item_id,
            status_column_id=call_request.status_column_id,
            call_session_id_column_id=call_request.call_session_id_column_id,
            # Generic CRM fields (for multi-CRM support)
            crm_container_id=crm_container_id,
            crm_item_id=crm_item_id,
            status_field_id=status_field_id,
            call_session_id_field_id=call_session_id_field_id,
            crm_type=call_request.crm_type
        )
        return JSONResponse(
            status_code=500,
            content=error_response.dict(exclude_none=True)
        )
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
    logger.info("🔥🔥🔥 WEBHOOK CALLED! 🔥🔥🔥")
    logger.info("=== Call Events Webhook Started ===")
    logger.info(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Request URL: {request.url}")
    logger.info(f"Request headers: {dict(request.headers)}")
    logger.info(f"Query params: agentId={agentId}, userId={userId}, callSessionId={callSessionId}")
    logger.info(f"Request body length: {len(body) if body else 0}")
    logger.debug(f"Request body preview: {body[:200] if body else 'None'}...")
    logger.debug(f"Database session: {db}")
    
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
        logger.info(f"✅ WebSocket broadcast queued at webhook start")
    except Exception as e:
        logger.warning(f"⚠️ WebSocket broadcast failed (non-critical): {e}")
        # Don't print traceback - this is not critical for call processing
    try:
        logger.debug("Parsing request body...")
        
        # Parse form data to get call information
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "")
        call_status = form_data.get("CallStatus", "")
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        direction = form_data.get("Direction", "")
        
        # Note: Speech input is now handled by Google Cloud STT via WebSocket
        # The old Twilio SpeechResult is no longer used
        # speech_result = form_data.get("SpeechResult", "")
        # confidence = form_data.get("Confidence", "")
        # speech_duration = form_data.get("SpeechDuration", "")
        
        logger.info(f"🎤 Speech handling is now managed by Google Cloud STT WebSocket")
        
        # Get call session using callSessionId from query parameters (OPTIMIZED)
        call_session = None
        agent = None
        
        if callSessionId:
            try:
                session_uuid = uuid.UUID(callSessionId)
                call_session = call_session_service.get_call_session_by_id(db, session_uuid)
                if call_session:
                    logger.info(f"✅ Found call session: {call_session.id} from query parameter")
                    
                    # Fetch agent using call session's tenant_id
                    if agentId:
                        agent = agent_service.get_agent_by_id(db, uuid.UUID(agentId), call_session.tenant_id)
                        if agent:
                            logger.info(f"✅ Agent fetched: {agent.name} (ID: {agent.id})")
                            logger.info(f"🏢 Tenant: {agent.tenant_id}")
                        else:
                            logger.warning(f"⚠️ Agent {agentId} not found in tenant {call_session.tenant_id}")
                else:
                    logger.warning(f"⚠️ No call session found for ID: {callSessionId}")
            except ValueError:
                logger.warning(f"⚠️ Invalid call session ID format: {callSessionId}")
        else:
            logger.info(f"⚠️ No callSessionId provided in query parameters")
        
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
            logger.info("No authentication headers found, allowing for testing")
        
        # (Removed outbound in-progress gating based on AnsweredBy/has_media)

        # Log the call event
        logger.info(f"Call Events Webhook - SID: {call_sid}, Status: {call_status}, From: {from_number}, To: {to_number}, Direction: {direction}")
        logger.info(f"AgentId from query: {agentId}")
        
        # 🔍 DEBUG: Track all incoming statuses for troubleshooting
        logger.debug("=" * 60)
        logger.debug(f"🔍 DEBUG WEBHOOK RECEIVED:")
        logger.debug(f"   Status: '{call_status}'")
        logger.debug(f"   Direction: '{direction}'")
        logger.debug(f"   Call SID: {call_sid}")
        if call_session:
            logger.debug(f"   Current DB Status: '{call_session.status}'")
            logger.debug(f"   Call Session ID: {call_session.id}")
        else:
            logger.debug(f"   Call Session: Not found")
        logger.debug("=" * 60)
        
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
        #         logger.info(f"✅ Test broadcast queued to WebSocket for session {call_session.id}")
        #     except Exception as e:
        #         logger.warning(f"⚠️ Test broadcast failed (non-critical): {e}")
        
        # Status broadcasts will be handled in the main status update section below
        
        # Update call session status if we have a call session and status
        # ⚠️ SKIP automatic update for "answered" and "in-progress" - handled in specific handlers below
        # "in-progress" will ONLY be set when media streaming actually starts (first media packet in bidirectional_stream.py)
        if call_session and call_status and call_status not in ["answered", "in-progress"]:
            logger.info(f"🔄 Updating call session {call_session.id} status to: {call_status}")
            call_session.status = call_status
        elif call_session and call_status in ["answered", "in-progress"]:
            logger.debug(f"🔍 DEBUG: Skipping automatic status update for '{call_status}' - will be set when media streaming starts")
        
        # Set end time and calculate duration when call completes
        if call_session and call_status == "completed":
            call_session.end_time = datetime.now(timezone.utc)
            if call_session.start_time:
                duration = (call_session.end_time - call_session.start_time).total_seconds()
                call_session.duration = int(duration)
                logger.info(f"⏰ Set end time and duration ({duration}s) for session {call_session.id}")
                
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
                    logger.info(f"✅ Queued call ended event for session {call_session.id}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to queue call ended event (non-critical): {e}")
                
            
            # Update call session AND call log together (single commit)
            call_session_service.update_call_session_status(
                db, 
                call_session.id, 
                "completed",
                ended_reason="hung up"
            )
            logger.info(f"✅ Updated call session {call_session.id} status to: {call_status} with ended_reason: hung up")

        # Broadcast status update to WebSocket (SINGLE COMPREHENSIVE BROADCAST)
        if call_session and call_status:
            # Allow "in-progress" and "answered" statuses to be broadcasted
            broadcast_status = call_status
            if call_status == "answered":
                broadcast_status = "in-progress"
                logger.info(f"🔄 Treating 'answered' as 'in-progress' for immediate status update")

            try:
                logger.info(f"🚀 Broadcasting call status update: {broadcast_status} for session {call_session.id}")
                
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
                if broadcast_status == "ringing":
                    metadata["message"] = "Call is ringing"
                elif broadcast_status == "in-progress":
                    metadata["message"] = "Call is in progress"
                elif broadcast_status == "completed":
                    metadata["message"] = "Call has been completed"
                elif broadcast_status == "initiated":
                    metadata["message"] = "Call is being initiated"
                elif broadcast_status == "failed":
                    metadata["message"] = "Call failed to connect"
                elif broadcast_status == "busy":
                    metadata["message"] = "Line is busy"
                elif broadcast_status == "no-answer":
                    metadata["message"] = "No answer from the recipient"
                elif broadcast_status == "canceled":
                    metadata["message"] = "Call was canceled"
                
                await broadcast_call_status_update(
                    call_session_id=str(call_session.id),
                    status=broadcast_status,
                    metadata=metadata
                )
                logger.debug(f"✅ Call status update sent: {broadcast_status} for session {call_session.id}")
                
                # Terminate monitoring and broadcast final event for terminal statuses
                terminal_statuses = ["completed", "failed", "busy", "no-answer", "canceled"]
                if call_status in terminal_statuses:
                    # Stop credit monitoring
                    try:
                        credit_service.stop_credit_monitoring(call_session.id)
                        logger.info(f"✅ Stopped credit monitoring for session {call_session.id} (status: {call_status})")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to stop credit monitoring: {e}")

                    # Broadcast call ended event
                    asyncio.create_task(broadcast_call_ended(
                        call_session_id=str(call_session.id),
                        reason=call_status,
                        final_data={
                            "call_sid": call_sid,
                            "from_number": from_number,
                            "to_number": to_number,
                            "direction": direction,
                            "duration": call_session.duration if call_session.duration else 0,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "transcript": call_session.call_transcript or []
                        }
                    ))
                    logger.debug(f"✅ Queued call ended event ({call_status}) for session {call_session.id}")
                        
            except Exception as e:
                logger.error(f"❌ Failed to broadcast call status update: {e}", exc_info=True)
        else:
            if not call_session:
                logger.warning(f"⚠️ No call session found - cannot broadcast status")
            if not call_status:
                logger.warning(f"⚠️ No call status provided - cannot broadcast status")
        
        # Speech input is now handled by Google Cloud STT via WebSocket
        # The WebSocket will transcribe audio and generate responses
        # This webhook now primarily handles call status updates and plays pending responses
        
        # Handle different call statuses and trigger agent logic
        logger.info(f"Processing call status: '{call_status}' with direction: '{direction}'")
        
        if call_status == "initiated" and direction == "outbound-api":
            # Call has been initiated - just log and return empty response
            logger.info(f"Call initiated - SID: {call_sid}")
            
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "ringing" and direction == "outbound-api":
            # Outbound call is ringing - just log, don't play any audio
            logger.info(f"🔔 CALL IS RINGING - SID: {call_sid}")
            
            # Return empty response - no audio should play while ringing
            return HTMLResponse("", media_type="application/xml")

        elif call_status == "answered" and direction == "outbound-api":
            # Treat answered as in-progress for immediate feedback
            logger.info(f"✅ Call answered! Updating session status to in-progress.")
            
            if call_session:
                # Update call session status and start time
                call_session.status = "in-progress"
                if not call_session.start_time:
                    call_session.start_time = datetime.now(timezone.utc)
                db.commit()
                
                # Start credit monitoring as soon as call connects
                try:
                    if str(call_session.id) not in credit_service._active_monitors:
                        asyncio.create_task(credit_service.start_credit_monitoring(
                            db=db,
                            call_session_id=call_session.id,
                            tenant_id=call_session.tenant_id,
                            agent_id=call_session.agent_id
                        ))
                        logger.info(f"✅ Started credit monitoring for session {call_session.id}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to start credit monitoring: {e}")

            return HTMLResponse("", media_type="application/xml")

        elif call_status == "in-progress":
            # Ensure session is marked as in-progress and credit monitoring is running
            logger.info(f"ℹ️ IN-PROGRESS STATUS RECEIVED")
            
            if call_session:
                if call_session.status != "in-progress":
                    call_session.status = "in-progress"
                    if not call_session.start_time:
                        call_session.start_time = datetime.now(timezone.utc)
                    db.commit()
                
                # Ensure credit monitoring is running
                try:
                    if str(call_session.id) not in credit_service._active_monitors:
                        asyncio.create_task(credit_service.start_credit_monitoring(
                            db=db,
                            call_session_id=call_session.id,
                            tenant_id=call_session.tenant_id,
                            agent_id=call_session.agent_id
                        ))
                except Exception as e:
                    logger.warning(f"⚠️ Failed to check/start credit monitoring: {e}")
            
            return HTMLResponse("", media_type="application/xml")
        elif call_status == "completed":
            # Call completed
            logger.info(f"📞 CALL COMPLETED - SID: {call_sid}")
            
            # Broadcast call completed event (this is already handled above in the status update section)
            # The broadcast_call_ended is already called in the status update section above
            
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "failed":
            # Call failed - handle error
            logger.error(f"Call failed - SID: {call_sid}")
            
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "busy":
            # Call busy - handle busy signal
            logger.info(f"Call busy - SID: {call_sid}")
            
            return HTMLResponse("", media_type="application/xml")
        
        elif call_status == "no-answer":
            # Call no-answer - handle no answer
            logger.info(f"Call no-answer - SID: {call_sid}")
            
            return HTMLResponse("", media_type="application/xml")
        
        else:
            # Default response for other statuses
            logger.info(f"Unhandled call status: '{call_status}' - using default response")
            response = VoiceResponse()
            text = "Thanks for calling! Have a great day!"
            lang = agent.language if agent and agent.language else "en"
            voice = agent.voice_type if agent and agent.voice_type else "female"
            tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}"
            response.play(tts_url)
            return HTMLResponse(str(response), media_type="application/xml")
    
    except Exception as e:
        logger.error(f"ERROR occurred: {str(e)}", exc_info=True)
        logger.error("=== Call Events Webhook Failed ===")
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
    logger.info(f"🎙️ RECORDING CALLBACK WEBHOOK - VAPI-style")
    logger.debug(f"📞 Call Session: {callSessionId}")
    logger.debug(f"🤖 Agent: {agentId}")
    
    try:
        form_data = await request.form()
        
        # Extract recording details
        recording_url = form_data.get("RecordingUrl", "")
        recording_sid = form_data.get("RecordingSid", "")
        recording_duration = form_data.get("RecordingDuration", "0")
        call_sid = form_data.get("CallSid", "")
        recording_status = form_data.get("RecordingStatus", "")
        
        logger.debug(f"🎵 Recording URL: {recording_url}")
        logger.debug(f"📝 Recording SID: {recording_sid}")
        logger.debug(f"⏱️ Duration: {recording_duration}s")
        logger.debug(f"📊 Status: {recording_status}")
        
        # IMPORTANT: Twilio calls this webhook twice:
        # 1. 'action' callback (no status, has URL) - User finished speaking → PROCESS THIS for TTS
        # 2. 'recordingStatusCallback' (has status) - Recording processed → SKIP (just for logging)
        
        if recording_status:
            # This is a status callback, not the action callback
            # We don't need to return TTS here, just acknowledge
            logger.debug(f"ℹ️ Recording status callback (status={recording_status}) - acknowledging only, no TTS")
            return HTMLResponse("", media_type="application/xml")
        
        # If no recording URL at all, something is wrong
        if not recording_url:
            logger.warning(f"⚠️ No recording URL provided - cannot process")
            return HTMLResponse("", media_type="application/xml")
        
        # This is the 'action' callback - user finished speaking
        # Process this for TTS response
        logger.info(f"✅ Action callback detected - processing for TTS response")
        
        # Get call session
        call_session = None
        agent = None
        
        if callSessionId:
            try:
                session_uuid = uuid.UUID(callSessionId)
                call_session = call_session_service.get_call_session_by_id(db, session_uuid)
                
                if call_session and agentId:
                    agent = agent_service.get_agent_by_id(db, uuid.UUID(agentId), call_session.tenant_id)
                    logger.debug(f"✅ Found call session and agent: {agent.name if agent else 'Unknown'}")
            except ValueError:
                logger.warning(f"⚠️ Invalid call session ID: {callSessionId}")
        
        # Process recording if available
        if recording_url and call_session:
            try:
                import requests
                
                # ✅ Get Twilio credentials based on call session (DB or Env)
                account_sid, auth_token = get_twilio_credentials_for_call(db, call_session)
                
                # Build authenticated recording URL
                # Twilio recordings are usually at /Recordings/{RecordingSid}
                if not recording_url.startswith('http'):
                    # Relative URL - build full URL
                    auth_url = f"https://{account_sid}:{auth_token}@api.twilio.com{recording_url}.wav"
                else:
                    # Full URL - add auth
                    auth_url = recording_url.replace('https://api.twilio.com', f'https://{account_sid}:{auth_token}@api.twilio.com') + '.wav'
                
                logger.debug(f"📥 Downloading audio from Twilio...")
                
                # Download the recording
                audio_response = requests.get(auth_url, timeout=10)
                
                if audio_response.status_code != 200:
                    logger.error(f"❌ Failed to download recording: HTTP {audio_response.status_code}")
                    raise Exception(f"Failed to download recording: HTTP {audio_response.status_code}")
                
                audio_content = audio_response.content
                logger.debug(f"✅ Downloaded {len(audio_content)} bytes of audio")
                
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
                
                logger.debug(f"🎙️ Transcribing with Google Cloud STT (language: {language_code})...")
                
                # Transcribe with Google STT
                from app.services.google_stt_service import google_stt_service
                
                stt_result = await google_stt_service.transcribe_audio_chunk_streaming(
                    audio_content=audio_content,
                    language_code=language_code
                )
                
                transcript = stt_result.get("transcript", "").strip()
                confidence = stt_result.get("confidence", 0.0)
                
                logger.info(f"📝 Google STT Transcript: '{transcript}'")
                logger.debug(f"📊 Confidence: {confidence:.2f}")
                
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
                    logger.debug(f"🤖 Generating agent response...")
                    response_text = await VoiceLoggingService.generate_agent_response(
                        speech_text=transcript,
                        confidence=confidence,
                        agent=agent,
                        db=db,
                        call_session_id=call_session.id
                    )
                    
                    logger.info(f"✅ Agent response: '{response_text}'")
                    
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
                        logger.info(f"🛑 Goodbye detected - ending call")
                        response = VoiceResponse()
                        response.hangup()
                        twiml_str = str(response)
                        logger.debug(f"📤 Returning TwiML (goodbye): {twiml_str[:200]}...")
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
                    
                    logger.debug(f"💾 Stored pending TTS in metadata: '{response_text[:50]}...'")
                    
                    # Build TwiML for TTS-only WebSocket streaming + Recording
                    recording_callback_url = f'{settings.WEBHOOK_BASE_URL}/api/v1/voice/webhook/recording-callback?agentId={agentId}&userId={userId}&callSessionId={callSessionId}'
                    
                    from app.routers.bidirectional_stream import build_tts_only_twiml
                    twiml_str = build_tts_only_twiml(
                        call_session_id=str(call_session.id),
                        agent_id=str(agent.id) if agent else agentId,
                        record_callback_url=recording_callback_url
                    )
                    
                    logger.debug(f"🎵 Returning TwiML with TTS WebSocket streaming")
                    logger.debug(f"📤 TwiML: {twiml_str[:200]}...")
                    return HTMLResponse(twiml_str, media_type="application/xml")
                
                else:
                    # No transcript - ask user to repeat
                    logger.info(f"⚠️ No transcript from Google STT")
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
                logger.error(f"❌ Error processing recording: {e}", exc_info=True)
                
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
        logger.warning(f"⚠️ No recording URL provided")
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
        logger.error(f"❌ Error in recording callback webhook: {e}", exc_info=True)
        
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
    logger.warning(f"⚠️ DEPRECATED: GATHER SPEECH WEBHOOK CALLED")
    logger.warning(f"Use /webhook/recording-callback instead")
    
    try:
        form_data = await request.form()
        
        call_sid = form_data.get("CallSid", "")
        recording_url = form_data.get("RecordingUrl", "")
        speech_result = form_data.get("SpeechResult", "")  # Twilio's transcription
        confidence = form_data.get("Confidence", "0")
        
        logger.debug(f"📞 Call SID: {call_sid}")
        logger.debug(f"🎤 Twilio Speech Result: {speech_result}")
        logger.debug(f"📊 Confidence: {confidence}")
        logger.debug(f"🎵 Recording URL: {recording_url}")
        
        # Get call session
        call_session = None
        if callSessionId:
            try:
                session_uuid = uuid.UUID(callSessionId)
                call_session = call_session_service.get_call_session_by_id(db, session_uuid)
                logger.debug(f"✅ Found call session: {call_session.id}")
            except ValueError:
                logger.warning(f"⚠️ Invalid call session ID: {callSessionId}")
        
        # Get agent
        agent = None
        if agentId and call_session:
            try:
                agent = agent_service.get_agent_by_id(db, uuid.UUID(agentId), call_session.tenant_id)
                logger.debug(f"✅ Agent: {agent.name}")
            except Exception as e:
                logger.warning(f"⚠️ Error fetching agent: {e}")
        
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
                logger.debug(f"📥 Downloading audio from Twilio...")
                
                audio_response = requests.get(auth_url)
                audio_content = audio_response.content
                
                logger.debug(f"✅ Downloaded {len(audio_content)} bytes of audio")
                
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
                
                logger.debug(f"🎙️ Transcribing with Google Cloud STT (language: {language_code})...")
                
                # Transcribe with Google STT
                stt_result = await google_stt_service.transcribe_audio_chunk_streaming(
                    audio_content=audio_content,
                    language_code=language_code
                )
                
                google_transcript = stt_result.get("transcript", "")
                google_confidence = stt_result.get("confidence", 0.0)
                
                logger.info(f"📝 Google STT Transcript: '{google_transcript}'")
                logger.debug(f"📊 Google STT Confidence: {google_confidence:.2f}")
                
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
                    
                    logger.info(f"✅ Generated agent response: '{response_text}'")
                    
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
                        logger.info(f"🛑 Goodbye detected - ending call")
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
                    
                    logger.debug(f"📝 Response TwiML: {str(response)[:200]}...")
                    return HTMLResponse(str(response), media_type="application/xml")
            
            except Exception as e:
                logger.error(f"❌ Error processing gathered speech: {e}", exc_info=True)
        
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
        logger.error(f"❌ Error in gather speech webhook: {e}", exc_info=True)
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
        
        logger.info(f"🎙️ RECORDING STATUS UPDATE")
        logger.debug(f"Recording SID: {recording_sid}")
        logger.debug(f"Call SID: {call_sid}")
        logger.debug(f"Status: {recording_status}")
        logger.debug(f"URL: {recording_url}")
        logger.debug(f"Duration: {recording_duration}")
        
        # Find the call session
        if call_sid:
            call_session = call_session_service.get_call_session_by_twilio_sid(db, call_sid)
            if call_session:
                # Update recording URL when recording is completed
                if recording_status == "completed" and recording_url:
                    call_session.recording_url = recording_url
                    db.commit()
                    logger.info(f"✅ Updated call session {call_session.id} with recording URL")
                    
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
                        logger.debug(f"✅ Queued recording completed status update for session {call_session.id}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to queue recording completed status update (non-critical): {e}")
                else:
                    logger.debug(f"📝 Recording status: {recording_status} - URL not ready yet")
            else:
                logger.warning(f"⚠️ Call session not found for SID: {call_sid}")
        
        # Return empty TwiML response
        return HTMLResponse("", media_type="application/xml")
        
    except Exception as e:
        logger.warning(f"⚠️ Error handling recording status webhook: {e}")
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
        
        if call_session.start_time:
            duration = (call_session.end_time - call_session.start_time).total_seconds()
            call_session.duration = int(duration)
        
        # Update call session AND call log together (single commit)
        call_session_service.update_call_session_status(
            db, 
            call_session.id, 
            "completed",
            ended_reason="completed"
        )
        
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
            logger.warning(f"⚠️ Failed to broadcast call ended event: {e}")
        
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
        logger.error(f"❌ Error ending call: {e}", exc_info=True)
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
        
        # ✅ Get Twilio credentials based on call session (DB or Env)
        account_sid, auth_token = get_twilio_credentials_for_call(db, call_session)
        
        # Extract recording SID from the URL
        recording_sid = call_session.recording_url.split('/')[-1].replace('.mp3', '').replace('.wav', '')
        
        # Create authenticated Twilio URL for server-side download
        authenticated_url = f"https://{account_sid}:{auth_token}@api.twilio.com/2010-04-01/Accounts/{account_sid}/Recordings/{recording_sid}.mp3"
        
        logger.info(f"📥 Streaming recording for call session: {call_session_id}")
        logger.debug(f"🎵 Recording SID: {recording_sid}")
        
        # Download recording from Twilio (server-side with auth)
        response = requests.get(authenticated_url, stream=True, timeout=30)
        
        if response.status_code != 200:
            logger.error(f"❌ Failed to fetch recording: HTTP {response.status_code}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to fetch recording from Twilio: HTTP {response.status_code}"
            )
        
        logger.info(f"✅ Streaming recording to user (no login required)")
        
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
        logger.error(f"❌ Network error fetching recording: {e}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Error streaming recording: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to stream recording: {str(e)}")


@router.post("/transcript/analyze/{call_session_id}", response_model=SuccessResponse[dict])
async def analyze_call_transcript(
    call_session_id: str,
    user: User = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Analyze call transcript using LLM for summary, sentiment, and recommendations.
    
    Args:
        call_session_id: UUID of the call session
        user: Current authenticated user
        db: Database session
        
    Returns:
        Analysis results including:
        - summary: Brief call overview (2-3 sentences)
        - sentiment: Sentiment analysis with score
        - recommendations: Actionable recommendations based on agent's prompt/instructions (if agent has custom prompt)
    """
    try:
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
        
        # 🎯 FLEXIBLE MODEL SELECTION WITH FALLBACK
        # Priority: 1. Call's model, 2. Gemini 2.0 Flash, 3. Llama, 4. GPT-4o Mini
        
        # Try to use call's model first and get agent's prompt
        preferred_model = None
        agent = None
        agent_prompt = None
        if call_session.agent_id:
            try:
                agent = agent_service.get_agent_by_id(db, call_session.agent_id, call_session.tenant_id)
                if agent:
                    # Get agent's system prompt (priority: agent.system_prompt > model.system_prompt)
                    if agent.system_prompt:
                        agent_prompt = agent.system_prompt
                        logger.debug(f"📝 Using agent's custom system prompt ({len(agent_prompt)} chars)")
                    elif agent.model and agent.model.system_prompt:
                        agent_prompt = agent.model.system_prompt
                        logger.debug(f"📝 Using model's system prompt ({len(agent_prompt)} chars)")
                    
                    if agent and agent.model:
                        preferred_model = agent.model.model_name
                        logger.debug(f"🔍 Found call's model: {preferred_model}")
            except Exception as e:
                logger.warning(f"⚠️ Could not get call's model or agent prompt: {e}")
        
        # Fallback models in priority order
        fallback_models = [
            preferred_model,  # Call's model (if available)
            "gemini-2.0-flash",  # Preferred for analysis
            "llama-3.3-70b-versatile",  # Llama fallback
            "gpt-4o-mini"  # GPT fallback
        ]
        
        # Remove None values
        fallback_models = [m for m in fallback_models if m]
        
        model = None
        last_error = None
        
        # Try each model until one works
        for model_name in fallback_models:
            try:
                logger.debug(f"🔄 Trying model: {model_name}")
                model = model_service.get_model_by_name(db, model_name)
                if model:
                    logger.debug(f"✅ Model found: {model.model_name}, Provider: {model.provider.name}")
                    break
            except Exception as e:
                logger.warning(f"⚠️ Model {model_name} not available: {e}")
                last_error = e
                continue
        
        if not model:
            raise HTTPException(
                status_code=404, 
                detail=f"No available model found. Tried: {', '.join(fallback_models)}"
            )
        
        # Get transcript messages
        transcript_messages = transcript_service.get_messages_by_session(db, session_uuid)
        logger.debug(f"🔍 Found {len(transcript_messages)} transcript messages for session {call_session_id}")
        
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
        
        # Create recommendations prompt based on agent's instructions
        recommendations_prompt = f"""
Analyze this call transcript and provide 2-3 brief, actionable recommendations for the agent.

Call Transcript:
{transcript_text}

Agent's Instructions/Purpose:
{agent_prompt if agent_prompt else "No specific instructions provided. Use general best practices for customer service calls."}

IMPORTANT - Keep recommendations BRIEF and CONCISE:
- Provide only 2-3 recommendations maximum
- Each recommendation should be 1 sentence only (brief and to the point)
- Be specific and actionable
- Use friendly, conversational tone

Format your response as:
1. [Brief recommendation in 1 sentence]
2. [Next brief recommendation in 1 sentence]
3. [Optional third recommendation in 1 sentence]

Keep it concise - similar to summary format. Maximum 1 sentence per recommendation.
"""
        
        # Helper function to call appropriate service based on provider
        def generate_analysis_text(current_model, current_api_key, prompt: str, max_tokens: int = 200):
            """Generate text using the appropriate service based on provider"""
            provider_name = (current_model.provider.name or "").strip().lower()
            
            if provider_name in ("gemini", "google", "google-ai", "google ai", "gemini-1.5-flash", "gemini-2.0-flash"):
                # Use Gemini service
                from app.services.gemini_service import GeminiService
                service = GeminiService()
                return service.generate_text(
                    prompt=prompt,
                    model_name=current_model.model_name,
                    temperature=0.3,
                    max_tokens=max_tokens,
                    api_key=current_api_key
                )
            elif provider_name in ("openai", "gpt", "gpt-4o-mini", "gpt-4o", "gpt-4"):
                # Use OpenAI service
                from app.services.openai_service import OpenAIService
                service = OpenAIService()
                return service.generate_text(
                    prompt=prompt,
                    system_prompt="You are an AI assistant that analyzes call transcripts.",
                    model_name=current_model.model_name,
                    temperature=0.3,
                    max_tokens=max_tokens,
                    api_key=current_api_key
                )
            elif provider_name in ("groq", "llama", "llama-3.3-70b-versatile"):
                # Use Groq service (for Llama)
                from app.services.groq_service import GroqService
                service = GroqService()
                return service.generate_text(
                    prompt=prompt,
                    system_prompt="You are an AI assistant that analyzes call transcripts.",
                    model_name=current_model.model_name,
                    temperature=0.3,
                    max_tokens=max_tokens,
                    api_key=current_api_key
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported provider for analysis: {provider_name}"
                )
        
        # Perform analysis with automatic fallback on quota errors
        summary_result = None
        sentiment_result = None
        recommendations_result = None
        used_model = None
        last_error = None
        
        # Try each model until one succeeds
        for model_name in fallback_models:
            try:
                # Get model
                current_model = model_service.get_model_by_name(db, model_name)
                if not current_model:
                    continue
                
                # Get API key
                current_api_key = None
                if current_model.api_key:
                    from app.core.security import decrypt_api_key
                    current_api_key = decrypt_api_key(current_model.api_key)
                
                logger.debug(f"🔄 Attempting analysis with {current_model.model_name}...")
                
                # Generate summary
                summary_result = generate_analysis_text(current_model, current_api_key, summary_prompt, max_tokens=200)
                
                # Generate sentiment analysis
                sentiment_result = generate_analysis_text(current_model, current_api_key, sentiment_prompt, max_tokens=150)
                
                # Generate recommendations (only if agent has a prompt/instructions)
                if agent_prompt:
                    try:
                        recommendations_result = generate_analysis_text(
                            current_model, 
                            current_api_key, 
                            recommendations_prompt, 
                            max_tokens=300
                        )
                        logger.debug(f"✅ Recommendations generated")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to generate recommendations: {e}")
                        # Continue even if recommendations fail
                
                used_model = current_model.model_name
                logger.info(f"✅ Analysis successful with {used_model}")
                break
                
            except Exception as e:
                error_str = str(e)
                logger.warning(f"⚠️ Error with {model_name}: {e}")
                
                # Check if it's a quota error - try next model
                if "429" in error_str or "quota" in error_str.lower() or "exceeded" in error_str.lower():
                    logger.warning(f"⚠️ Quota exceeded for {model_name}, trying next model...")
                    last_error = e
                    continue
                else:
                    # Other errors - try next model anyway
                    last_error = e
                    continue
        
        # Check if we got required results
        if not summary_result or not sentiment_result:
            error_msg = f"Analysis failed with all models. Last error: {str(last_error)}"
            logger.error(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Prepare response (hide model_id for security)
        analysis_data = {
            "summary": summary_result["content"].strip(),
            "sentiment": sentiment_result["content"].strip()
        }
        
        # Add recommendations if available - parse into array format
        if recommendations_result:
            recommendations_text = recommendations_result["content"].strip()
            
            # Parse recommendations into array (extract numbered list items)
            import re
            recommendations_list = []
            
            # Split by newline and extract numbered items
            lines = recommendations_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Match patterns like "1. ", "2. ", "1.", "2.", etc.
                match = re.match(r'^\d+\.\s*(.+)$', line)
                if match:
                    recommendations_list.append(match.group(1).strip())
                # Also handle cases without numbers but with bullet points
                elif line.startswith('- ') or line.startswith('* '):
                    recommendations_list.append(line[2:].strip())
                # If no pattern matches but line is substantial, include it
                elif len(line) > 20 and not recommendations_list:
                    # First item might not have number
                    recommendations_list.append(line)
            
            # If parsing failed, use original text as single item
            if not recommendations_list:
                recommendations_list = [recommendations_text]
            
            analysis_data["recommendations"] = recommendations_list
            analysis_data["recommendations_text"] = recommendations_text  # Keep original for backward compatibility
        elif agent_prompt:
            # If agent has prompt but recommendations failed, indicate it
            analysis_data["recommendations"] = ["Unable to generate recommendations at this time."]
            analysis_data["recommendations_text"] = "Unable to generate recommendations at this time."
        
        analysis_result = {
            "call_session_id": call_session_id,
            "transcript_message_count": len(transcript_messages),
            "call_duration": call_session.duration,
            "call_status": call_session.status,
            "analysis": analysis_data,
            "model_used": used_model,  # Show which model was actually used
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"✅ Transcript analysis completed for session {call_session_id} using {used_model}")
        return create_success_response(
            data=analysis_result,
            message=f"Transcript analysis completed successfully using {used_model}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in transcript analysis endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# def _generate_default_response() -> str:
#     """Generate default TwiML response"""
#     response = VoiceResponse()
#     response.say("Thank you for calling. An agent will be with you shortly.", voice="")
#     response.pause(length=2)
#     response.say("Please hold while we connect you.", voice="")
#     return str(response)