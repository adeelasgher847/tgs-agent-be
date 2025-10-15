"""
Voice Processing Router
Handles speech-to-text, OpenAI processing, and text-to-speech for voice calls
"""

from fastapi import APIRouter, Request, HTTPException, Query, Depends, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional
from twilio.twiml.voice_response import VoiceResponse
import io
import base64

from app.api.deps import get_db, require_tenant
from app.models.agent import Agent
from app.models.user import User
from app.models.call_session import CallSession
from app.services.openai_service import openai_service
from app.services.elevenlabs_service import elevenlabs_service
from app.services.call_session_service import call_session_service
from app.services.twilio_service import twilio_service
from app.utils.twilio_validation import validate_twilio_signature, get_request_body
from app.core.config import settings
import uuid

router = APIRouter()

@router.post("/webhook/voice-process", response_class=HTMLResponse)
async def process_voice_input(
    request: Request,
    agentId: Optional[str] = Query(None),
    sessionId: Optional[str] = Query(None),
    body: str = Depends(get_request_body),
    db: Session = Depends(get_db)
):
    """
    Process voice input: Speech-to-Text -> OpenAI -> Text-to-Speech
    
    This endpoint handles the complete voice processing pipeline:
    1. Receives speech input from Twilio
    2. Converts speech to text (using Twilio's built-in speech recognition)
    3. Sends text to OpenAI for processing
    4. Converts OpenAI response to speech
    5. Returns audio or TwiML response
    """
    try:
        # Validate Twilio signature
        if not validate_twilio_signature(request, body):
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")
        
        # Parse form data
        form_data = await request.form()
        
        # Extract call information
        call_sid = form_data.get("CallSid", "")
        speech_result = form_data.get("SpeechResult", "")
        confidence = form_data.get("Confidence", "0")
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        
        print(f"Voice Processing - Call SID: {call_sid}")
        print(f"Speech Result: {speech_result}")
        print(f"Confidence: {confidence}")
        print(f"Agent ID: {agentId}")
        print(f"Session ID: {sessionId}")
        
        # Get or create call session
        call_session = None
        if sessionId:
            try:
                session_uuid = uuid.UUID(sessionId)
                call_session = call_session_service.get_call_session_by_id(db, session_uuid)
            except ValueError:
                print(f"Invalid session ID format: {sessionId}")
        
        if not call_session and call_sid:
            call_session = call_session_service.get_call_session_by_twilio_sid(db, call_sid)
        
        # Get agent
        agent = None
        if agentId:
            try:
                agent_uuid = uuid.UUID(agentId)
                agent = db.query(Agent).filter(Agent.id == agent_uuid).first()
            except (ValueError, Exception) as e:
                print(f"Error getting agent: {e}")
        
        if not agent and call_session:
            agent = db.query(Agent).filter(Agent.id == call_session.agent_id).first()
        
        if not agent:
            # Default response if no agent found
            response = VoiceResponse()
            response.say("I'm sorry, but I couldn't find the agent configuration. Please try again later.")
            return HTMLResponse(str(response), media_type="application/xml")
        
        # Process speech input
        if speech_result and float(confidence) > 0.5:
            # Add user input to transcript
            if call_session:
                call_session_service.add_transcript_entry(
                    db, call_session.id, "user", speech_result
                )
            
            # Process with Gemini AI
            try:
                # Get conversation history for context
                conversation_history = []
                if call_session and call_session.call_transcript:
                    # Get last few messages for context (excluding the current one)
                    recent_messages = call_session.call_transcript[-6:]  # Last 6 messages
                    for msg in recent_messages:
                        if msg.get("role") in ["user", "assistant"]:
                            conversation_history.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })
                
                # Use Gemini if agent has a model_id, otherwise fall back to OpenAI
                if agent.model_id:
                    from app.services.gemini_service import gemini_service
                    from app.services.model_service import model_service
                    from app.core.security import decrypt_api_key
                    
                    # Get the model from database
                    model = model_service.get_model_by_id(db, agent.model_id)
                    if model and not model.archive and 'gemini' in model.model_name.lower():
                        # Use Gemini with agent-specific configuration
                        model_name = model.model_name
                        # Always ensure agent name is included in system prompt
                        agent_name = agent.name if agent.name else "AI Assistant"
                        agent_language = agent.language if agent.language else "English"
                        
                        if agent.system_prompt:
                            system_prompt = f"""You are {agent_name}, a real person taking phone calls.

Your identity:
- You are {agent_name}, a real human being
- You speak {agent_language} naturally
- You are having a phone conversation with a customer
- Always respond as {agent_name} - introduce yourself naturally when appropriate

Custom instructions:
{agent.system_prompt}

Guidelines for phone conversations:
- Speak naturally and conversationally like a real person
- Be friendly and professional
- Give complete, helpful answers
- If you don't understand something, ask for clarification politely
- Keep responses between 1-3 sentences for good conversation flow
- Be helpful and try to answer questions thoroughly
- Remember you are {agent_name} - introduce yourself naturally when appropriate

Always respond as {agent_name}, a real person, not as any kind of system or tool."""
                        elif model.system_prompt:
                            system_prompt = f"""You are {agent_name}, a real person taking phone calls.

Your identity:
- You are {agent_name}, a real human being
- You speak {agent_language} naturally
- You are having a phone conversation with a customer
- Always respond as {agent_name} - introduce yourself naturally when appropriate

Model instructions:
{model.system_prompt}

Guidelines for phone conversations:
- Speak naturally and conversationally like a real person
- Be friendly and professional
- Give complete, helpful answers
- If you don't understand something, ask for clarification politely
- Keep responses between 1-3 sentences for good conversation flow
- Be helpful and try to answer questions thoroughly
- Remember you are {agent_name} - introduce yourself naturally when appropriate

Always respond as {agent_name}, a real person, not as any kind of system or tool."""
                        else:
                            system_prompt = f"""You are {agent_name}, a real person taking phone calls.

Your identity:
- You are {agent_name}, a real human being
- You speak {agent_language} naturally
- You are having a phone conversation with a customer
- Be friendly, professional, and conversational like a real person

Guidelines for phone conversations:
- Speak naturally and conversationally
- Be friendly and professional
- Give complete, helpful answers
- If you don't understand something, ask for clarification politely
- Keep responses between 1-3 sentences for good conversation flow
- Be helpful and try to answer questions thoroughly
- Remember you are {agent_name} - introduce yourself naturally when appropriate

Always respond as {agent_name}, a real person, not as any kind of system or tool."""
                        # Use agent-specific temperature if set, otherwise fall back to model default
                        temperature = (
                            (agent.agent_temperature / 100.0) if agent.agent_temperature is not None 
                            else (model.temperature / 100.0) if model.temperature 
                            else 0.8
                        )
                        # Use agent-specific max tokens if set, otherwise fall back to model default
                        max_tokens = agent.agent_max_tokens if agent.agent_max_tokens is not None else (model.max_tokens or 300)
                        
                        # Use model-specific API key if available
                        api_key = None
                        if model.api_key:
                            try:
                                api_key = decrypt_api_key(model.api_key)
                            except Exception as e:
                                print(f"⚠️ Failed to decrypt model API key: {e}")
                        
                        # Generate response using Gemini
                        gemini_response = gemini_service.generate_text(
                            prompt=speech_result,
                            system_prompt=system_prompt,
                            model_name=model_name,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            api_key=api_key
                        )
                        
                        ai_response_text = gemini_response["content"]
                        response_time = gemini_response["response_time"]
                        print(f"✅ Used Gemini model: {model_name}")
                    else:
                        # Fall back to OpenAI
                        openai_response = openai_service.process_agent_conversation(
                            user_input=speech_result,
                            agent_system_prompt=agent.system_prompt or "You are a helpful assistant.",
                            conversation_history=conversation_history
                        )
                        ai_response_text = openai_response["response"]
                        response_time = openai_response["response_time"]
                        print("✅ Used OpenAI (fallback)")
                else:
                    # No model_id, use OpenAI
                    openai_response = openai_service.process_agent_conversation(
                        user_input=speech_result,
                        agent_system_prompt=agent.system_prompt or "You are a helpful assistant.",
                        conversation_history=conversation_history
                    )
                    ai_response_text = openai_response["response"]
                    response_time = openai_response["response_time"]
                    print("✅ Used OpenAI (no model_id)")
                
                # Add AI response to transcript
                if call_session:
                    call_session_service.add_transcript_entry(
                        db, call_session.id, "assistant", ai_response_text, response_time
                    )
                
                # Generate TwiML response with AI response
                response = VoiceResponse()
                
                # Use agent's voice type if configured
                voice_type = agent.voice_type or "alloy"
                
                # Map agent voice types to TTS voices
                tts_voice = _map_voice_type_to_tts(voice_type)
                
                # Say the AI response
                response.say(ai_response_text, voice=tts_voice)
                
                # Add gather for next user input
                gather = response.gather(
                    input='speech',
                    timeout=8,
                    speech_timeout='auto',
                    action=f'/voice/webhook/voice-process?agentId={agent.id}&sessionId={call_session.id if call_session else ""}',
                    method='POST'
                )
                gather.say("What else can I help you with?", voice=tts_voice)
                
                # Fallback if no input
                response.say("Sorry, I didn't catch that. Let me know if you need anything else.", voice=tts_voice)
                
                return HTMLResponse(str(response), media_type="application/xml")
                
            except Exception as e:
                print(f"Error processing with OpenAI: {e}")
                # Fallback response
                response = VoiceResponse()
                response.say("Sorry, I'm having some trouble right now. Could you try again?", voice="Polly.Joanna")
                
                # Add gather for retry
                gather = response.gather(
                    input='speech',
                    timeout=10,
                    speech_timeout='auto',
                    action=f'/voice/webhook/voice-process?agentId={agent.id}&sessionId={call_session.id if call_session else ""}',
                    method='POST'
                )
                gather.say("Please try again.", voice="")
                
                return HTMLResponse(str(response), media_type="application/xml")
        
        else:
            # Low confidence or no speech detected
            response = VoiceResponse()
            response.say("I didn't catch that clearly. Could you please repeat what you said?", voice="")
            
            # Add gather for retry
            gather = response.gather(
                input='speech',
                timeout=10,
                speech_timeout='auto',
                action=f'/voice/webhook/voice-process?agentId={agent.id}&sessionId={call_session.id if call_session else ""}',
                method='POST'
            )
            gather.say("Please speak clearly and try again.", voice="")
            
            return HTMLResponse(str(response), media_type="application/xml")
    
    except Exception as e:
        print(f"Error in voice processing webhook: {e}")
        # Return a simple response to avoid call failures
        response = VoiceResponse()
        response.say("I'm sorry, but I'm experiencing technical difficulties. Please try again later.", voice="")
        return HTMLResponse(str(response), media_type="application/xml")

@router.post("/webhook/voice-init", response_class=HTMLResponse)
async def initialize_voice_call(
    request: Request,
    agentId: Optional[str] = Query(None),
    userId: Optional[str] = Query(None),
    body: str = Depends(get_request_body),
    db: Session = Depends(get_db)
):
    """
    Initialize a voice call and create call session
    
    This endpoint is called when a call starts to:
    1. Create a call session
    2. Set up the initial voice interaction
    3. Start the speech recognition
    """
    try:
        # Validate Twilio signature
        if not validate_twilio_signature(request, body):
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")
        
        # Parse form data
        form_data = await request.form()
        
        # Extract call information
        call_sid = form_data.get("CallSid", "")
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        
        print(f"Voice Init - Call SID: {call_sid}")
        print(f"Agent ID: {agentId}")
        print(f"User ID: {userId}")
        
        # Get agent
        agent = None
        if agentId:
            try:
                agent_uuid = uuid.UUID(agentId)
                agent = db.query(Agent).filter(Agent.id == agent_uuid).first()
            except (ValueError, Exception) as e:
                print(f"Error getting agent: {e}")
        
        if not agent:
            # Default response if no agent found
            response = VoiceResponse()
            response.say("I'm sorry, but I couldn't find the agent configuration. Please try again later.")
            return HTMLResponse(str(response), media_type="application/xml")
        
        # Get existing call session or create new one if not found
        call_session = None
        if call_sid:
            call_session = call_session_service.get_call_session_by_twilio_sid(db, call_sid)
        
        if not call_session and userId:
            try:
                user_uuid = uuid.UUID(userId)
                call_session = call_session_service.create_call_session(
                    db=db,
                    user_id=user_uuid,
                    agent_id=agent.id,
                    tenant_id=agent.tenant_id,
                    twilio_call_sid=call_sid,
                    from_number=from_number,
                    to_number=to_number,
                    call_type="inbound",
                    assistant_phone_number=to_number,
                    customer_phone_number=from_number
                )
                print(f"Created call session: {call_session.id}")
            except (ValueError, Exception) as e:
                print(f"Error creating call session: {e}")
        
        # Generate initial TwiML response
        response = VoiceResponse()
        
        # Use agent's greeting or default
        greeting = agent.fallback_response or f"Hello! This is {agent.name}. How can I help you today?"
        voice_type = _map_voice_type_to_tts(agent.voice_type or "alloy")
        
        response.say(greeting, voice=voice_type)
        
        # Add gather for speech input
        gather = response.gather(
            input='speech',
            timeout=8,
            speech_timeout='auto',
            action=f'/voice/webhook/voice-process?agentId={agent.id}&sessionId={call_session.id if call_session else ""}',
            method='POST'
        )
        gather.say("Please tell me how I can assist you.", voice=voice_type)
        
        # Fallback if no input
        response.say("I didn't hear anything. Please let me know if you need assistance.", voice=voice_type)
        
        return HTMLResponse(str(response), media_type="application/xml")
    
    except Exception as e:
        print(f"Error in voice init webhook: {e}")
        # Return a simple response to avoid call failures
        response = VoiceResponse()
        response.say("Thank you for calling. An agent will be with you shortly.", voice="")
        return HTMLResponse(str(response), media_type="application/xml")

@router.post("/webhook/call-end", response_class=HTMLResponse)
async def handle_call_end(
    request: Request,
    sessionId: Optional[str] = Query(None),
    body: str = Depends(get_request_body),
    db: Session = Depends(get_db)
):
    """
    Handle call end and update call session status
    """
    try:
        # Validate Twilio signature
        if not validate_twilio_signature(request, body):
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")
        
        # Parse form data
        form_data = await request.form()
        
        # Extract call information
        call_sid = form_data.get("CallSid", "")
        call_status = form_data.get("CallStatus", "")
        
        print(f"Call End - Call SID: {call_sid}, Status: {call_status}")
        
        # Update call session status
        if sessionId:
            try:
                session_uuid = uuid.UUID(sessionId)
                call_session_service.update_call_session_status(db, session_uuid, call_status)
            except ValueError:
                print(f"Invalid session ID format: {sessionId}")
        elif call_sid:
            call_session = call_session_service.get_call_session_by_twilio_sid(db, call_sid)
            if call_session:
                call_session_service.update_call_session_status(db, call_session.id, call_status)
        
        return HTMLResponse("", media_type="application/xml")
    
    except Exception as e:
        print(f"Error in call end webhook: {e}")
        return HTMLResponse("", media_type="application/xml")

def _map_voice_type_to_tts(voice_type: str) -> str:
    """
    Map agent voice type to TTS voice
    
    Args:
        voice_type: Agent's voice type
        
    Returns:
        TTS voice identifier
    """
    voice_mapping = {
        "male": "Polly.Matthew",  # Male voice
        "female": "Polly.Joanna",  # Female voice
        "alloy": "alloy",  # OpenAI TTS
        "echo": "echo",  # OpenAI TTS
        "fable": "fable",  # OpenAI TTS
        "onyx": "onyx",  # OpenAI TTS
        "nova": "nova",  # OpenAI TTS
        "shimmer": "shimmer"  # OpenAI TTS
    }
    
    return voice_mapping.get(voice_type.lower(), "Polly.Joanna")
