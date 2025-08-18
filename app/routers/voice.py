from fastapi import APIRouter, Request, Form, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from typing import Optional
from twilio.twiml.voice_response import VoiceResponse

from app.api.deps import get_db, require_tenant
from app.schemas.twilio import (
    CallInitiateRequest, CallInitiateResponse, StatusResponse,
    CallResponse, MakeCallRequest, AgentRegistrationRequest,
    AgentRegistrationResponse, AgentListResponse
)
from app.schemas.base import SuccessResponse
from app.services.twilio_service import twilio_service
from app.services.voice_agent_service import voice_agent_manager
from app.utils.twilio_validation import validate_twilio_signature, validate_webrtc_auth, get_request_body
from app.utils.response import create_success_response
from app.core.config import settings

router = APIRouter()


@router.post("/call/initiate", response_model=SuccessResponse[CallInitiateResponse])
async def initiate_call(
    request: CallInitiateRequest,
    user: dict = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """
    Endpoint to initiate a voice call using Twilio.
    
    Request Payload:
    {
        "agentId": "agent_12345",
        "userPhoneNumber": "+1234567890"
    }
    """
    try:
        # Validate agent exists
        agent = voice_agent_manager.get_agent(request.agentId)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {request.agentId} not found")
        
        # Validate phone number format
        if not twilio_service.validate_phone_number(request.userPhoneNumber):
            raise HTTPException(status_code=400, detail="Invalid phone number format. Must start with +")
        
        # Get base URL for webhooks
        base_url = f"http://{settings.HOST}:{settings.PORT}"
        
        # Make the call using Twilio
        call = twilio_service.make_call(
            to_number=request.userPhoneNumber,
            from_number=twilio_service.get_phone_number(),
            webhook_url=f"{base_url}/api/v1/voice/webhook?agentId={request.agentId}",
            status_callback_url=f"{base_url}/api/v1/voice/status"
        )
        
        # Assign call to agent
        voice_agent_manager.assign_call_to_agent(call.sid, request.agentId)
        
        # Generate call ID
        call_id = f"call_{call.sid[-8:]}"
        
        return create_success_response(
            CallInitiateResponse(
                callId=call_id,
                twilioCallSid=call.sid,
                status="initiated"
            ),
            "Call initiated successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/call", response_model=SuccessResponse[CallResponse])
async def make_call(
    request: MakeCallRequest,
    user: dict = Depends(require_tenant),
    db: Session = Depends(get_db)
):
    """Make an outbound call using Twilio (JSON request)"""
    try:
        # Validate phone number format
        if not twilio_service.validate_phone_number(request.to_number):
            raise HTTPException(status_code=400, detail="Phone number must start with +")
        
        # Get base URL for webhooks
        base_url = f"http://{settings.HOST}:{settings.PORT}"
        
        # Use provided webhook URLs or defaults
        webhook_url = request.webhook_url or f"{base_url}/api/v1/voice/webhook"
        status_callback_url = request.status_callback_url or f"{base_url}/api/v1/voice/status"
        
        # Make the call using the service
        call = twilio_service.make_call(
            to_number=request.to_number,
            from_number=twilio_service.get_phone_number(),
            webhook_url=webhook_url,
            status_callback_url=status_callback_url
        )
        
        return create_success_response(
            CallResponse(
                success=True,
                call_sid=call.sid,
                to_number=request.to_number,
                from_number=twilio_service.get_phone_number(),
                status=call.status,
                message="Call initiated successfully"
            ),
            "Call initiated successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhook", response_class=HTMLResponse)
async def handle_call_events(
    request: Request,
    agentId: Optional[str] = Query(None),
    body: str = Depends(get_request_body)
):
    """
    Webhook endpoint to receive and handle Twilio or WebRTC voice call events in real time.
    """
    try:
        # Validate request (Twilio signature or WebRTC auth)
        is_twilio = 'X-Twilio-Signature' in request.headers
        is_webrtc = 'Authorization' in request.headers
        
        if is_twilio:
            if not validate_twilio_signature(request, body):
                raise HTTPException(status_code=403, detail="Invalid Twilio signature")
        elif is_webrtc:
            if not validate_webrtc_auth(request):
                raise HTTPException(status_code=403, detail="Invalid WebRTC authentication")
        else:
            # For testing purposes, allow requests without validation
            print("Warning: No authentication headers found, allowing for testing")
        
        # Parse form data
        form_data = await request.form()
        
        # Extract call information
        call_sid = form_data.get("CallSid", "")
        call_status = form_data.get("CallStatus", "")
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        direction = form_data.get("Direction", "")
        
        # Log the call event
        print(f"Call Event - SID: {call_sid}, Status: {call_status}, From: {from_number}, To: {to_number}, Direction: {direction}")
        
        # Handle different call statuses
        if call_status == "ringing" and direction == "outbound-api":
            # Outbound call is ringing
            if agentId:
                # Generate agent-specific response
                twiml_response = voice_agent_manager.generate_agent_response(agentId, {
                    'call_sid': call_sid,
                    'from_number': from_number,
                    'to_number': to_number,
                    'status': call_status
                })
                return HTMLResponse(twiml_response, media_type="application/xml")
            else:
                # Default response
                response = VoiceResponse()
                response.say("Hello! Thank you for answering our call.", voice="alice")
                response.say("An agent will be with you shortly.", voice="alice")
                return HTMLResponse(str(response), media_type="application/xml")
        
        elif call_status == "in-progress":
            # Call is in progress
            response = VoiceResponse()
            response.say("Your call is now connected. How can we help you today?", voice="alice")
            return HTMLResponse(str(response), media_type="application/xml")
        
        elif call_status == "completed":
            # Call completed
            voice_agent_manager.release_agent_from_call(call_sid)
            return HTMLResponse("", media_type="application/xml")
        
        else:
            # Default response for other statuses
            response = VoiceResponse()
            response.say("Thank you for your call.", voice="alice")
            return HTMLResponse(str(response), media_type="application/xml")
    
    except Exception as e:
        print(f"Error in call events webhook: {e}")
        # Return a simple response to avoid call failures
        response = VoiceResponse()
        response.say("Thank you for calling. Please try again later.", voice="alice")
        return HTMLResponse(str(response), media_type="application/xml")


@router.post("/gather", response_class=HTMLResponse)
async def handle_gather_input(request: Request, agentId: str = Query(...)):
    """Handle user input gathered from the call"""
    try:
        form_data = await request.form()
        speech_result = form_data.get("SpeechResult", "")
        confidence = form_data.get("Confidence", "0")
        
        print(f"Gather Input - Agent: {agentId}, Speech: {speech_result}, Confidence: {confidence}")
        
        # Create response based on user input
        response = VoiceResponse()
        
        if speech_result and float(confidence) > 0.5:
            # Process the speech input
            response.say(f"I understand you said: {speech_result}", voice="alice")
            response.say("Let me help you with that.", voice="alice")
            
            # Add more sophisticated logic here based on speech content
            if "help" in speech_result.lower():
                response.say("I'm here to help you. What specific assistance do you need?", voice="alice")
            elif "support" in speech_result.lower():
                response.say("I'll connect you with our support team.", voice="alice")
            else:
                response.say("Thank you for your input. An agent will assist you further.", voice="alice")
        else:
            response.say("I didn't catch that clearly. Let me transfer you to a human agent.", voice="alice")
        
        return HTMLResponse(str(response), media_type="application/xml")
    
    except Exception as e:
        print(f"Error in gather webhook: {e}")
        response = VoiceResponse()
        response.say("I'm having trouble processing your request. Let me transfer you.", voice="alice")
        return HTMLResponse(str(response), media_type="application/xml")


@router.post("/transfer", response_class=HTMLResponse)
async def handle_transfer(request: Request):
    """Handle call transfer to human agent"""
    try:
        response = VoiceResponse()
        response.say("Transferring you to a human agent. Please hold.", voice="alice")
        response.pause(length=2)
        response.say("Thank you for your patience.", voice="alice")
        return HTMLResponse(str(response), media_type="application/xml")
    
    except Exception as e:
        print(f"Error in transfer webhook: {e}")
        response = VoiceResponse()
        response.say("Thank you for calling. Goodbye.", voice="alice")
        return HTMLResponse(str(response), media_type="application/xml")


@router.post("/status", response_model=SuccessResponse[StatusResponse])
async def handle_call_status(request: Request):
    """Handle call status updates"""
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid", "")
        call_status = form_data.get("CallStatus", "")
        call_duration = form_data.get("CallDuration", "")
        
        print(f"Call Status Update - SID: {call_sid}, Status: {call_status}, Duration: {call_duration}")
        
        # Handle call completion
        if call_status == "completed":
            voice_agent_manager.release_agent_from_call(call_sid)
        
        return create_success_response(
            StatusResponse(status="received", message=f"Status update processed for call {call_sid}"),
            "Status update processed"
        )
    
    except Exception as e:
        print(f"Error in status webhook: {e}")
        return create_success_response(
            StatusResponse(status="error", message=str(e)),
            "Error processing status"
        )


@router.get("/agents", response_model=SuccessResponse[AgentListResponse])
async def list_agents(user: dict = Depends(require_tenant)):
    """List all registered agents and their status"""
    agent_status = voice_agent_manager.get_agent_status()
    return create_success_response(
        AgentListResponse(**agent_status),
        "Agent status retrieved successfully"
    )


@router.post("/agents/{agent_id}/register", response_model=SuccessResponse[AgentRegistrationResponse])
async def register_agent(
    agent_id: str,
    request: AgentRegistrationRequest,
    user: dict = Depends(require_tenant)
):
    """Register a new agent"""
    try:
        voice_agent_manager.register_agent(agent_id, request.capabilities)
        return create_success_response(
            AgentRegistrationResponse(success=True, message=f"Agent {agent_id} registered successfully"),
            f"Agent {agent_id} registered successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
