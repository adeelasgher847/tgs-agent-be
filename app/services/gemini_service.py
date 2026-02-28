"""
Gemini Service Module
Handles all Gemini-related operations including text generation
"""

from google import genai
from app.core.config import settings
from typing import List, Dict, Optional, Any
import time
import json

class GeminiService:
    """Service class for handling Gemini operations"""
    
    def __init__(self):
        self._client = None
        self._model_cache = {}
        self._current_api_key = None
    
    def _extract_text_from_response(self, response):
        """Helper method to extract text from response (handles different API structures)"""
        if hasattr(response, 'text') and response.text:
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    return candidate.content.parts[0].text if hasattr(candidate.content.parts[0], 'text') else ""
        elif hasattr(response, 'content') and response.content:
            if hasattr(response.content, 'parts') and response.content.parts:
                return response.content.parts[0].text if hasattr(response.content.parts[0], 'text') else ""
        return ""
    
    def get_client(self, api_key: str = None):
        """Get or create Gemini client with specific API key"""
        # Use provided API key or fall back to global setting
        key_to_use = api_key or settings.GEMINI_API_KEY
        
        if not key_to_use:
            raise Exception("Gemini API key not found. Please provide an API key or set GEMINI_API_KEY in your config.")
        
        # Only recreate client if the API key has changed
        if self._current_api_key != key_to_use:
            # Use the official google-genai SDK
            self._client = genai.Client(api_key=key_to_use)
            self._current_api_key = key_to_use
        
        return self._client
    
    def generate_text(self, prompt: str, system_prompt: str = None, 
                     model_name: str = "gemini-1.5-flash", 
                     temperature: float = 0.7, 
                     max_tokens: int = 1000,
                     api_key: str = None) -> Dict[str, Any]:
        """
        Generate text using Gemini API
        """
        try:
            start_time = time.time()
            client = self.get_client(api_key)
            
            # Prepare config
            config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if system_prompt:
                 config["system_instruction"] = system_prompt

            # Generate content using correct SDK v1 pattern
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            response_text = self._extract_text_from_response(response)
            
            return {
                "content": response_text,
                "model": model_name,
                "response_time": response_time,
                "usage": {
                    "prompt_tokens": 0, # SDK doesn't always return this immediately
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(response_text.split())
                },
                "finish_reason": "stop"
            }
            
        except Exception as e:
            logger.error(f"Error in Gemini text generation: {e}")
            raise Exception(f"Error in Gemini text generation: {str(e)}")

    async def stream_text(self, prompt: str, system_prompt: str = None,
                          model_name: str = "gemini-1.5-flash",
                          temperature: float = 0.7,
                          max_tokens: int = 1000,
                          api_key: str = None):
        """Yield text chunks from Gemini as they arrive (streaming)."""
        import asyncio
        import threading
        from queue import Queue

        client = self.get_client(api_key)
        
        # Prepare config
        config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_prompt:
            config["system_instruction"] = system_prompt

        q: Queue = Queue(maxsize=100)
        SENTINEL = object()

        def producer():
            try:
                # Use models.generate_content_stream for real-time response
                response = client.models.generate_content_stream(
                    model=model_name,
                    contents=prompt,
                    config=config
                )
                for chunk in response:
                    try:
                        text = None
                        if hasattr(chunk, 'text') and chunk.text:
                            text = chunk.text
                        elif hasattr(chunk, 'candidates') and chunk.candidates:
                            text = chunk.candidates[0].content.parts[0].text if chunk.candidates[0].content.parts else None
                        
                        if text:
                            q.put(text)
                    except Exception:
                        continue
            except Exception as e:
                logger.error(f"Gemini Streaming Producer Error: {e}")
                q.put(("__error__", str(e)))
            finally:
                q.put(SENTINEL)

        threading.Thread(target=producer, daemon=True).start()
        loop = asyncio.get_event_loop()

        while True:
            chunk = await loop.run_in_executor(None, q.get)
            if chunk is SENTINEL:
                break
            if isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == "__error__":
                raise Exception(chunk[1])
            yield str(chunk)
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       system_prompt: str = None, 
                       model_name: str = "gemini-1.5-flash", 
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       api_key: str = None) -> Dict[str, Any]:
        """
        Generate chat completion using Gemini API
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: System prompt to use for the conversation
            model_name: Gemini model to use
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens for response
            api_key: Model-specific API key (optional)
            
        Returns:
            Dictionary with response content and metadata
        """
        try:
            start_time = time.time()
            client = self.get_client(api_key)
            
            # Prepare the conversation for SDK v1
            # Note: SDK v1 uses a slightly different structure for contents
            # For simplicity in this refactor, we'll format as a single string prompt 
            # if it's meant to be a simple conversion or use the proper structure.
            
            formatted_prompt = ""
            if system_prompt:
                formatted_prompt += f"System: {system_prompt}\n\n"
            
            for message in messages:
                role = "User" if message["role"] == "user" else "Assistant"
                formatted_prompt += f"{role}: {message['content']}\n\n"
            
            # Prepare config
            config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if system_prompt:
                 config["system_instruction"] = system_prompt

            response = client.models.generate_content(
                model=model_name,
                contents=formatted_prompt,
                config=config
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            response_text = self._extract_text_from_response(response)
            
            return {
                "content": response_text,
                "model": model_name,
                "response_time": response_time,
                "usage": {
                    "prompt_tokens": len(formatted_prompt.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(formatted_prompt.split()) + len(response_text.split())
                },
                "finish_reason": "stop"
            }
            
        except Exception as e:
            logger.error(f"Error in Gemini chat completion: {e}")
            raise Exception(f"Error in Gemini chat completion: {str(e)}")
    
    def process_agent_conversation(self, user_input: str, 
                             agent_system_prompt: str = "You are a helpful assistant.",
                             call_session=None,
                             db=None,
                             model_name: str = "gemini-1.5-flash",
                             temperature: float = 0.7,
                             max_tokens: int = 1000,
                             api_key: str = None) -> Dict[str, Any]:
        """
        Process agent conversation using Gemini API with DB context memory.
        """
        try:
            from datetime import datetime
            import json

            start_time = time.time()
            client = self.get_client(api_key)

            # 📜 Load previous transcript (if any)
            conversation_history = []
            if call_session and call_session.call_transcript:
                try:
                    conversation_history = json.loads(call_session.call_transcript)
                except:
                    conversation_history = []

            # 🗣️ Prepare message context
            history_lines = []
            for msg in conversation_history[-6:]:
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content') or msg.get('message', '')
                    if content:
                        history_lines.append(f"{role.capitalize()}: {content}")
            history_text = "\n".join(history_lines)

            full_prompt = (
                f"Previous conversation:\n{history_text}\n\n"
                f"User: {user_input}\n\n"
                f"Note: Never repeat any question already asked. Be human-like and polite."
            )

            # Prepare config
            config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if agent_system_prompt:
                 config["system_instruction"] = agent_system_prompt

            # 💬 Generate model response
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=config
            )

            # ⏱️ Timing
            end_time = time.time()
            response_time = end_time - start_time

            # 🪄 Update transcript in DB (use "message" key for consistency)
            response_text = self._extract_text_from_response(response)
            
            if call_session and db:
                conversation_history.append({"role": "user", "message": user_input})
                conversation_history.append({"role": "assistant", "message": response_text})
                call_session.call_transcript = json.dumps(conversation_history)
                call_session.updated_at = datetime.utcnow()
                db.commit()

            return {
                "response": response_text,
                "model": model_name,
                "response_time": response_time,
                "usage": {
                    "prompt_tokens": len(full_prompt.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(full_prompt.split()) + len(response_text.split())
                }
            }

        except Exception as e:
            logger.error(f"Error in Gemini agent conversation: {e}")
            raise Exception(f"Error in Gemini agent conversation: {str(e)}")
        
# Create a singleton instance
gemini_service = GeminiService()
