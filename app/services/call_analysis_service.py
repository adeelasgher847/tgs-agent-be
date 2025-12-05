"""
Call Analysis Service
Handles call behavior analysis and early call termination
"""

from sqlalchemy.orm import Session
from typing import Optional
import json
import sys
import asyncio

from app.models.call_session import CallSession
from app.models.agent import Agent
from app.services.call_session_service import call_session_service
from app.services.twilio_service import twilio_service
from app.services.transcript_service import transcript_service
from app.services.gemini_service import gemini_service
from app.routers.general_websocket import broadcast_call_ended


def strip_ssml_tags(text: str) -> str:
    """
    Remove all SSML tags from text, keeping only the actual text content.
    Used for saving clean text to transcript.
    Handles both complete and incomplete SSML tags.
    """
    if not text:
        return ""
    
    import re
    # Remove complete SSML tags (<tag>content</tag> or <tag/>)
    text = re.sub(r'<[^>]+>', '', text)
    # Remove incomplete SSML tags (tags without closing >, like <break time="150ms)
    text = re.sub(r'<[^>]*', '', text)
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class CallAnalysisService:
    """Service for analyzing call behavior and managing early call termination"""
    
    @staticmethod
    async def check_for_early_disinterest(
        call_session: CallSession,
        user_text: str
    ) -> bool:
        """
        Check if user shows disinterest early in the call.
        Returns True if disinterest detected and call should end.
        Only triggers in first 2-3 user messages to avoid false positives.
        """
        try:
            if not user_text or not call_session:
                return False
            
            # Only check in first 2-3 user messages (early in call)
            conversation_history = []
            if call_session.call_transcript:
                try:
                    conversation_history = json.loads(call_session.call_transcript) if isinstance(call_session.call_transcript, str) else call_session.call_transcript
                except:
                    conversation_history = []
            
            # Count user messages (client role)
            user_message_count = sum(1 for msg in conversation_history if isinstance(msg, dict) and msg.get('role') == 'client')
            
            # Only check in first 2-3 user messages (0, 1, 2 = first 3 messages)
            if user_message_count >= 3:
                return False
            
            # Disinterest phrases (case-insensitive)
            disinterest_phrases = [
                "goodbye", "bye", "good bye",
                "i don't want to talk", "don't want to talk", "not interested",
                "have a great day", "have a nice day", "have a good day",
                "i'm not interested", "not interested", "no thanks",
                "i don't want", "don't want", "no thank you",
                "i'm busy", "i'm not available", "call me later",
                "wrong number", "not the right person"
            ]
            
            user_text_lower = user_text.lower().strip()
            
            # Check if any disinterest phrase is present
            for phrase in disinterest_phrases:
                if phrase in user_text_lower:
                    print(f"🚫 Early disinterest detected: '{phrase}' in '{user_text}'")
                    print(f"   User message count: {user_message_count}")
                    sys.stdout.flush()
                    return True
            
            return False
        
        except Exception as e:
            print(f"⚠️ Error checking for disinterest: {e}")
            sys.stdout.flush()
            return False
    
    @staticmethod
    async def check_if_all_objectives_complete(
        call_session: CallSession,
        agent: Agent,
        agent_response: str
    ) -> bool:
        """
        Check if all objectives/questions from system prompt are completed.
        Uses LLM to analyze conversation and determine completion.
        """
        try:
            if not call_session or not agent:
                return False
            
            # Get conversation history
            conversation_history = []
            if call_session.call_transcript:
                try:
                    conversation_history = json.loads(call_session.call_transcript) if isinstance(call_session.call_transcript, str) else call_session.call_transcript
                except:
                    conversation_history = []
            
            if len(conversation_history) < 4:  # Need at least 2 exchanges
                return False
            
            # Build conversation text
            conversation_text = ""
            for msg in conversation_history[-10:]:  # Last 10 messages
                if isinstance(msg, dict):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content') or msg.get('message', '')
                    if content and role in ['client', 'agent']:
                        conversation_text += f"{role.capitalize()}: {content}\n"
            
            # Get system prompt
            system_prompt = ""
            if agent.system_prompt:
                system_prompt = agent.system_prompt
            elif agent.model and agent.model.system_prompt:
                system_prompt = agent.model.system_prompt
            else:
                return False  # No system prompt to check
            
            if not system_prompt or len(system_prompt.strip()) < 20:
                return False
            
            # Use LLM to analyze if all objectives are complete
            analysis_prompt = f"""Analyze this phone conversation and determine if ALL objectives/questions from the system prompt have been completed.

System Prompt Objectives:
{system_prompt}

Conversation:
{conversation_text}

Latest Agent Response:
{agent_response}

Instructions:
- Review the system prompt objectives/questions carefully
- Check if ALL objectives have been addressed in the conversation
- Consider the conversation complete ONLY if:
  1. All questions from the system prompt have been asked AND answered
  2. All objectives/tasks have been completed
  3. The agent has provided all necessary information
  
Respond with ONLY one word: "YES" if all objectives are complete, or "NO" if any objectives remain incomplete."""

            try:
                # Use fast model for quick analysis (Gemini Flash)
                response = gemini_service.generate_text(
                    prompt=analysis_prompt,
                    system_prompt="You are an expert conversation analyzer. Analyze conversations and determine if objectives are complete. Respond with only YES or NO.",
                    model_name="gemini-1.5-flash",
                    temperature=0.3,
                    max_tokens=10
                )
                
                result = response.get("content", "").strip().upper()
                is_complete = "YES" in result
                
                if is_complete:
                    print(f"✅ All objectives/questions completed - call should end")
                    sys.stdout.flush()
                
                return is_complete
                
            except Exception as e:
                print(f"⚠️ Error checking objectives completion: {e}")
                sys.stdout.flush()
                return False
        
        except Exception as e:
            print(f"⚠️ Error in check_if_all_objectives_complete: {e}")
            sys.stdout.flush()
            return False
    
    @staticmethod
    async def check_for_rude_behavior(
        call_session: CallSession,
        user_text: str
    ) -> bool:
        """
        Check if user is showing rude or inappropriate behavior.
        Uses LLM to analyze user messages for rudeness.
        """
        try:
            if not user_text or not call_session:
                return False
            
            # Get recent user messages (last 3-4 messages)
            conversation_history = []
            if call_session.call_transcript:
                try:
                    conversation_history = json.loads(call_session.call_transcript) if isinstance(call_session.call_transcript, str) else call_session.call_transcript
                except:
                    conversation_history = []
            
            # Get last 3-4 user messages
            recent_user_messages = []
            for msg in conversation_history[-6:]:  # Check last 6 messages
                if isinstance(msg, dict) and msg.get('role') == 'client':
                    content = msg.get('content') or msg.get('message', '')
                    if content:
                        recent_user_messages.append(content)
            
            if len(recent_user_messages) < 1:
                return False
            
            # Build analysis text
            user_messages_text = "\n".join([f"- {msg}" for msg in recent_user_messages[-3:]])
            
            # Use LLM to analyze for rude behavior
            analysis_prompt = f"""Analyze these user messages from a phone call and determine if the user is being rude, disrespectful, or inappropriate.

User Messages:
{user_messages_text}

Latest User Message:
{user_text}

Check for:
- Swearing, profanity, or offensive language
- Disrespectful or hostile tone
- Aggressive or threatening language
- Inappropriate comments
- Repeated rude behavior patterns

Respond with ONLY one word: "YES" if the user is being rude/inappropriate, or "NO" if the user is being polite and respectful."""

            try:
                # Use fast model for quick analysis
                response = gemini_service.generate_text(
                    prompt=analysis_prompt,
                    system_prompt="You are an expert at detecting rude, disrespectful, or inappropriate behavior in conversations. Respond with only YES or NO.",
                    model_name="gemini-1.5-flash",
                    temperature=0.2,
                    max_tokens=10
                )
                
                result = response.get("content", "").strip().upper()
                is_rude = "YES" in result
                
                if is_rude:
                    print(f"🚫 Rude behavior detected in user messages")
                    print(f"   Messages: {user_messages_text[:100]}...")
                    sys.stdout.flush()
                
                return is_rude
                
            except Exception as e:
                print(f"⚠️ Error checking for rude behavior: {e}")
                sys.stdout.flush()
                return False
        
        except Exception as e:
            print(f"⚠️ Error in check_for_rude_behavior: {e}")
            sys.stdout.flush()
            return False
    
    @staticmethod
    async def end_call_early(
        db: Session,
        call_session: CallSession,
        call_sid: Optional[str] = None,
        reason: str = "user_disinterest",
        agent: Optional[Agent] = None
    ) -> None:
        """
        End the call early due to various reasons.
        Updates call session status and terminates Twilio call.
        """
        try:
            if not call_session:
                print("⚠️ Cannot end call: no call session")
                return
            
            # Map reasons to appropriate messages
            reason_messages = {
                "user_disinterest": "Thank you for your time. Have a great day!",
                "all_objectives_completed": "Thank you! I've completed everything. Have a great day!",
                "rude_user_behavior": "I'm sorry, but I cannot continue this conversation. Goodbye."
            }
            
            final_message = reason_messages.get(reason, "Thank you for your time. Have a great day!")
            
            print(f"🛑 Ending call early due to: {reason}")
            sys.stdout.flush()
            
            # Update call session status to completed
            call_session_service.update_call_session_status(
                db=db,
                session_id=call_session.id,
                status="completed",
                ended_reason=reason,
                success_evaluation="early_termination" if reason != "all_objectives_completed" else "completed"
            )
            
            # End Twilio call if we have call SID
            if call_sid:
                try:
                    twilio_service.end_call(call_sid)
                    print(f"✅ Twilio call {call_sid} terminated")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"⚠️ Error ending Twilio call: {e}")
                    sys.stdout.flush()
            
            # Add final message to transcript using transcript_service
            try:
                # Strip SSML tags if any
                clean_message = strip_ssml_tags(final_message)
                
                await transcript_service.add_and_broadcast_message(
                    db=db,
                    call_session_id=call_session.id,
                    role="agent",
                    message=clean_message,
                    message_type="call_ended",
                    agent_id=agent.id if agent else None,
                    user_id=call_session.user_id,
                    confidence=None
                )
                
                # Update legacy field
                conversation = transcript_service.get_conversation_array(db, call_session.id)
                call_session.call_transcript = conversation
                db.commit()
            except Exception as e:
                print(f"⚠️ Error adding final message to transcript: {e}")
                sys.stdout.flush()
            
            # Broadcast call ended event
            try:
                asyncio.create_task(broadcast_call_ended(
                    call_session_id=str(call_session.id),
                    reason=reason,
                    final_data={
                        "call_sid": call_sid,
                        "status": "completed",
                        "ended_reason": reason
                    }
                ))
            except Exception as e:
                print(f"⚠️ Failed to broadcast call ended: {e}")
                sys.stdout.flush()
            
            print(f"✅ Call ended successfully due to: {reason}")
            sys.stdout.flush()
        
        except Exception as e:
            print(f"❌ Error ending call early: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()

