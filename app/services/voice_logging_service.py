"""
Voice Logging Service
Handles voice listening, speech recognition, and call logging
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session

from app.models.call_session import CallSession
from app.models.call_log import CallLog
from app.models.agent import Agent
from app.schemas.call_log import CallLogCreate, CallLogUpdate

class VoiceLoggingService:
    """Service for voice listening and call logging"""
    
    @staticmethod
    async def log_voice_interaction(
        db: Session,
        call_session_id: uuid.UUID,
        interaction_type: str,
        audio_data: Optional[bytes] = None,
        speech_text: Optional[str] = None,
        confidence: Optional[float] = None,
        duration: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Log voice interaction during a call
        """
        try:
            print("=" * 60)
            print(f"🎤 LOGGING VOICE INTERACTION")
            print(f"🆔 Call Session: {call_session_id}")
            print(f"📝 Type: {interaction_type}")
            print(f"🗣️ Speech: {speech_text}")
            print(f"📊 Confidence: {confidence}")
            print(f"⏱️ Duration: {duration}")
            print("=" * 60)
            
            # Get call session
            call_session = db.query(CallSession).filter(
                CallSession.id == call_session_id
            ).first()
            
            if not call_session:
                raise Exception(f"Call session {call_session_id} not found")
            
            # Create voice interaction log
            voice_log = {
                "id": str(uuid.uuid4()),
                "call_session_id": str(call_session_id),
                "interaction_type": interaction_type,
                "speech_text": speech_text,
                "confidence": confidence,
                "duration": duration,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {}
            }
            
            # Update call session with voice interaction
            if not call_session.call_metadata:
                call_session.call_metadata = {}
            
            if "voice_interactions" not in call_session.call_metadata:
                call_session.call_metadata["voice_interactions"] = []
            
            call_session.call_metadata["voice_interactions"].append(voice_log)
            
            # Note: Transcript is now handled by _add_to_transcript function in voice.py
            # This prevents duplicate transcript entries with different formats
            
            db.commit()
            
            print(f"✅ Voice interaction logged successfully")
            return voice_log
            
        except Exception as e:
            print(f"❌ Error logging voice interaction: {e}")
            db.rollback()
            raise e
    
    @staticmethod
    async def process_speech_input(
        db: Session,
        call_session_id: uuid.UUID,
        speech_text: str,
        confidence: float,
        duration: float,
        agent_id: Optional[uuid.UUID] = None
    ) -> Dict[str, Any]:
        """
        Process speech input and generate response
        """
        try:
            print("=" * 60)
            print(f"🗣️ PROCESSING SPEECH INPUT")
            print(f"🆔 Call Session: {call_session_id}")
            print(f"📝 Speech: '{speech_text}'")
            print(f"📊 Confidence: {confidence}")
            print(f"⏱️ Duration: {duration}")
            print("=" * 60)
            
            # Log the speech input
            await VoiceLoggingService.log_voice_interaction(
                db=db,
                call_session_id=call_session_id,
                interaction_type="speech_input",
                speech_text=speech_text,
                confidence=confidence,
                duration=duration,
                metadata={
                    "agent_id": str(agent_id) if agent_id else None,
                    "processing_time": datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Get agent for response generation
            agent = None
            if agent_id:
                agent = db.query(Agent).filter(Agent.id == agent_id).first()
            
            # Generate response based on speech input
            response_text = await VoiceLoggingService.generate_agent_response(
                speech_text=speech_text,
                confidence=confidence,
                agent=agent
            )
            
            # Log the agent response
            await VoiceLoggingService.log_voice_interaction(
                db=db,
                call_session_id=call_session_id,
                interaction_type="agent_response",
                speech_text=response_text,
                confidence=1.0,  # Agent response is always 100% confident
                duration=len(response_text) * 0.05,  # Estimate duration
                metadata={
                    "agent_id": str(agent_id) if agent_id else None,
                    "response_type": "generated"
                }
            )
            
            print(f"✅ Speech processed, response generated: '{response_text}'")
            
            return {
                "response_text": response_text,
                "confidence": confidence,
                "processed": True,
                "agent_id": str(agent_id) if agent_id else None
            }
            
        except Exception as e:
            print(f"❌ Error processing speech input: {e}")
            raise e
    
    @staticmethod
    async def generate_agent_response(
        speech_text: str,
        confidence: float,
        agent: Optional[Agent] = None,
        db: Optional[Session] = None,
        call_session_id: Optional[uuid.UUID] = None
    ) -> str:
        """
        Generate agent response based on speech input using Gemini AI with conversation context
        """
        try:
            print(f"🤖 Generating Gemini response for: '{speech_text}'",agent,'agent info log')
            
            # If no agent or no model_id, fall back to simple responses
            if not agent or not agent.model_id:
                print("⚠️ No agent or model_id found, using fallback response")
                return await VoiceLoggingService._generate_fallback_response(speech_text, agent)
            
            # Import here to avoid circular imports
            from app.services.gemini_service import gemini_service
            from app.services.model_service import model_service
            from app.core.security import decrypt_api_key
            
            # Get the model from database
            model = model_service.get_model_by_id(db, agent.model_id)
            if not model:
                print("⚠️ Model not found, using fallback response")
                return await VoiceLoggingService._generate_fallback_response(speech_text, agent)
            
            # Check if model is active
            if model.archive:
                print("⚠️ Model is archived, using fallback response")
                return await VoiceLoggingService._generate_fallback_response(speech_text, agent)
            
            # Get model details with agent-specific overrides
            model_name = model.model_name
            
            # Create personalized system prompt with agent information
            agent_name = agent.name if agent.name else "AI Assistant"
            agent_language = agent.language if agent.language else "English"
            
            # Get conversation context if call_session_id is provided
            conversation_context = ""
            if call_session_id and db:
                try:
                    call_session = db.query(CallSession).filter(CallSession.id == call_session_id).first()
                    if call_session and call_session.call_transcript:
                        # Get last few interactions for context
                        recent_interactions = call_session.call_transcript[-6:]  # Last 6 interactions
                        if recent_interactions:
                            conversation_context = "\n\nPrevious conversation context:\n"
                            for interaction in recent_interactions:
                                speaker = "Customer" if interaction.get("speaker") == "customer" else "You"
                                text = interaction.get("text", "")
                                conversation_context += f"{speaker}: {text}\n"
                except Exception as e:
                    print(f"⚠️ Error getting conversation context: {e}")
            
            # Build personalized system prompt with conversation context
            personalized_system_prompt = f"""You are {agent_name}, a real person taking phone calls.

Your identity:
- You are {agent_name}, a real human being
- You speak {agent_language} naturally
- You are having a phone conversation with a customer
- Be friendly, professional, and conversational like a real person

Guidelines for phone conversations:
- Speak naturally and conversationally like a real person
- Be friendly and engaging
- Have natural conversations, not customer service interactions
- If you don't understand something, ask for clarification politely
- Keep responses between 1-3 sentences for good conversation flow
- Be conversational and engaging, not robotic or service-oriented
- Remember you are {agent_name} - only mention your name when specifically asked or in the initial greeting
- Use the conversation context to provide relevant, personalized responses
- Build on previous topics naturally
- Don't repeat information you've already shared unless asked
- Talk like a friend, not like a helper or assistant
- Don't constantly repeat your name - just talk naturally

Always respond as {agent_name}, a real person having a conversation, not as any kind of system or tool.{conversation_context}"""
            
            # Always ensure agent name is included in system prompt
            # If agent has custom system prompt, prepend agent identity to it
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
- Be friendly and engaging
- Have natural conversations, not customer service interactions
- If you don't understand something, ask for clarification politely
- Keep responses between 1-3 sentences for good conversation flow
- Be conversational and engaging, not robotic or service-oriented
- Remember you are {agent_name} - only mention your name when specifically asked or in the initial greeting
- Use the conversation context to provide relevant, personalized responses
- Build on previous topics naturally
- Don't repeat information you've already shared unless asked
- Talk like a friend, not like a helper or assistant
- Don't constantly repeat your name - just talk naturally

Always respond as {agent_name}, a real person having a conversation, not as any kind of system or tool.{conversation_context}"""
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
- Be friendly and engaging
- Have natural conversations, not customer service interactions
- If you don't understand something, ask for clarification politely
- Keep responses between 1-3 sentences for good conversation flow
- Be conversational and engaging, not robotic or service-oriented
- Remember you are {agent_name} - only mention your name when specifically asked or in the initial greeting
- Use the conversation context to provide relevant, personalized responses
- Build on previous topics naturally
- Don't repeat information you've already shared unless asked
- Talk like a friend, not like a helper or assistant
- Don't constantly repeat your name - just talk naturally

Always respond as {agent_name}, a real person having a conversation, not as any kind of system or tool.{conversation_context}"""
            else:
                system_prompt = personalized_system_prompt
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
                    # Continue with global key
            
            # Check if this is a Gemini model
            if 'gemini' not in model_name.lower():
                print(f"⚠️ Model {model_name} is not a Gemini model, using fallback response")
                return await VoiceLoggingService._generate_fallback_response(speech_text, agent)
            
            # Generate response using Gemini
            print(f"🔧 Gemini Config: model={model_name}, temp={temperature}, max_tokens={max_tokens}")
            print(f"🔧 Agent: {agent_name} (Language: {agent_language})")
            print(f"🔧 System Prompt: {system_prompt[:200]}...")
            print(f"🔧 User Prompt: {speech_text}")
            
            gemini_response = gemini_service.generate_text(
                prompt=speech_text,
                system_prompt=system_prompt,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key
            )
            
            response_text = gemini_response["content"]
            response_time = gemini_response["response_time"]
            
            print(f"✅ Gemini generated response in {response_time:.2f}s: '{response_text}'")
            return response_text
            
        except Exception as e:
            print(f"❌ Error generating Gemini response: {e}")
            # Fall back to simple response
            return await VoiceLoggingService._generate_fallback_response(speech_text, agent)
    
    @staticmethod
    async def _generate_fallback_response(speech_text: str, agent: Optional[Agent] = None) -> str:
        """
        Generate fallback response when Gemini is not available
        """
        try:
            speech_lower = speech_text.lower()
            agent_name = agent.name if agent and agent.name else "AI Assistant"
            
            if "hello" in speech_lower or "hi" in speech_lower:
                response = "How can I help you today?"
            elif "help" in speech_lower:
                response = "Sure! What's going on?"
            elif "thank" in speech_lower:
                response = "You're welcome! What else would you like to talk about?"
            elif "goodbye" in speech_lower or "bye" in speech_lower:
                response = "Thanks for calling! Take care!"
            elif "price" in speech_lower or "cost" in speech_lower:
                response = "I can talk about pricing info. What are you looking for?"
            elif "support" in speech_lower:
                response = "Sure! What's going on?"
            elif "name" in speech_lower or "who" in speech_lower:
                response = "I'm here to help you. What would you like to talk about?"
            elif "how are you" in speech_lower:
                response = "I'm doing great, thank you for asking! How are you doing today?"
            elif "what" in speech_lower and "do" in speech_lower:
                response = "I'm here to chat with you. What's on your mind?"
            else:
                response = "Got it! What else would you like to talk about?"
            
            print(f"✅ Generated fallback response: '{response}'")
            return response
            
        except Exception as e:
            print(f"❌ Error generating fallback response: {e}")
            # Use agent name in error response if available
            if agent and agent.name:
                return "Sorry, I didn't quite catch that. Could you repeat that?"
            else:
                return "Sorry, I didn't quite catch that. Could you repeat that?"
    
    @staticmethod
    async def log_call_events(
        db: Session,
        call_session_id: uuid.UUID,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> None:
        """
        Log call events (ringing, answered, completed, etc.)
        """
        try:
            print("=" * 60)
            print(f"📞 LOGGING CALL EVENT")
            print(f"🆔 Call Session: {call_session_id}")
            print(f"📝 Event Type: {event_type}")
            print(f"📊 Event Data: {event_data}")
            print("=" * 60)
            
            # Get call session
            call_session = db.query(CallSession).filter(
                CallSession.id == call_session_id
            ).first()
            
            if not call_session:
                raise Exception(f"Call session {call_session_id} not found")
            
            # Update call session with event
            if not call_session.call_metadata:
                call_session.call_metadata = {}
            
            if "call_events" not in call_session.call_metadata:
                call_session.call_metadata["call_events"] = []
            
            event_log = {
                "id": str(uuid.uuid4()),
                "event_type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": event_data
            }
            
            call_session.call_metadata["call_events"].append(event_log)
            
            # Update call session status based on event
            if event_type == "call_started":
                call_session.status = "in-progress"
            elif event_type == "call_ended":
                call_session.status = "completed"
                call_session.end_time = datetime.now(timezone.utc)
                if call_session.start_time:
                    duration = (call_session.end_time - call_session.start_time).total_seconds()
                    call_session.duration = int(duration)
            elif event_type == "call_failed":
                call_session.status = "failed"
                call_session.end_time = datetime.now(timezone.utc)
            
            db.commit()
            
            # Broadcast the event to WebSocket connections
            try:
                from app.routers.general_websocket import broadcast_call_event
                import asyncio
                asyncio.create_task(broadcast_call_event(
                    str(call_session_id), 
                    event_type, 
                    event_data
                ))
            except Exception as e:
                print(f"Error broadcasting call event: {e}")
            
            print(f"✅ Logged call event: {event_type} for session {call_session_id}")
            
        except Exception as e:
            print(f"❌ Error logging call event: {e}")
            raise
    
    @staticmethod
    def get_call_voice_logs(
        db: Session,
        call_session_id: uuid.UUID
    ) -> List[Dict[str, Any]]:
        """
        Get voice logs for a specific call session
        """
        try:
            call_session = db.query(CallSession).filter(
                CallSession.id == call_session_id
            ).first()
            
            if not call_session:
                return []
            
            if not call_session.call_metadata or "voice_interactions" not in call_session.call_metadata:
                return []
            
            return call_session.call_metadata["voice_interactions"]
            
        except Exception as e:
            print(f"❌ Error getting call voice logs: {e}")
            return []
    
    @staticmethod
    def get_call_transcript(
        db: Session,
        call_session_id: uuid.UUID
    ) -> List[Dict[str, Any]]:
        """
        Get call transcript for a specific call session
        """
        try:
            call_session = db.query(CallSession).filter(
                CallSession.id == call_session_id
            ).first()
            
            if not call_session:
                return []
            
            return call_session.call_transcript or []
            
        except Exception as e:
            print(f"❌ Error getting call transcript: {e}")
            return []
