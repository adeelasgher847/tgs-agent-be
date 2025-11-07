"""
Gemini Service Module
Handles all Gemini-related operations including text generation
"""

import google.generativeai as genai
from app.core.config import settings
from typing import List, Dict, Optional, Any
import time
import json

class GeminiService:
    """Service class for handling Gemini operations"""
    
    def __init__(self):
        self._client = None
        self._model = None
        self._current_api_key = None
    
    def get_client(self, api_key: str = None):
        """Get or create Gemini client with specific API key"""
        # Use provided API key or fall back to global setting
        key_to_use = api_key or settings.GEMINI_API_KEY
        
        if not key_to_use:
            raise Exception("Gemini API key not found. Please provide an API key or set GEMINI_API_KEY in your config.")
        
        # Only reconfigure if the API key has changed
        if self._current_api_key != key_to_use:
            genai.configure(api_key=key_to_use)
            self._current_api_key = key_to_use
            self._client = genai
            self._model = None  # Reset model to force recreation with new key
        
        return self._client
    
    def get_model(self, model_name: str = "gemini-1.5-flash", api_key: str = None):
        """Get or create Gemini model instance with specific API key"""
        # Use provided API key or fall back to global setting
        key_to_use = api_key or settings.GEMINI_API_KEY
        
        if not key_to_use:
            raise Exception("Gemini API key not found. Please provide an API key or set GEMINI_API_KEY in your config.")
        
        # Only recreate model if API key has changed
        if self._current_api_key != key_to_use or self._model is None:
            client = self.get_client(key_to_use)
            self._model = client.GenerativeModel(model_name)
        
        return self._model
    
    def generate_text(self, prompt: str, system_prompt: str = None, 
                     model_name: str = "gemini-1.5-flash", 
                     temperature: float = 0.7, 
                     max_tokens: int = 1000,
                     api_key: str = None) -> Dict[str, Any]:
        """
        Generate text using Gemini API
        
        Args:
            prompt: The input prompt for text generation
            system_prompt: System prompt to set the context
            model_name: Gemini model to use
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens for response
            api_key: Model-specific API key (optional)
            
        Returns:
            Dictionary with response content and metadata
        """
        try:
            start_time = time.time()
            
            # Get model instance with specific API key
            model = self.get_model(model_name, api_key)
            
            # Prepare the full prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
            
            # Generate content
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return {
                "content": response.text,
                "model": model_name,
                "response_time": response_time,
                "usage": {
                    "prompt_tokens": len(full_prompt.split()),  # Approximate
                    "completion_tokens": len(response.text.split()),  # Approximate
                    "total_tokens": len(full_prompt.split()) + len(response.text.split())
                },
                "finish_reason": "stop"  # Gemini doesn't provide this directly
            }
            
        except Exception as e:
            raise Exception(f"Error in Gemini text generation: {str(e)}")

    async def stream_text(self, prompt: str, system_prompt: str = None,
                          model_name: str = "gemini-1.5-flash",
                          temperature: float = 0.7,
                          max_tokens: int = 1000,
                          api_key: str = None):
        """Yield text chunks from Gemini as they arrive (streaming).
        This returns an async iterator of strings.
        """
        import asyncio
        import threading
        from queue import Queue

        # Prepare prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"

        # Get model instance (thread-safe via global client)
        model = self.get_model(model_name, api_key)

        q: Queue = Queue()
        SENTINEL = object()

        def producer():
            try:
                response = model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                    stream=True,
                )
                for event in response:
                    try:
                        text = getattr(event, "text", None)
                        if text:
                            q.put(text)
                    except Exception:
                        # Skip malformed events
                        continue
            except Exception as e:
                q.put(("__error__", str(e)))
            finally:
                q.put(SENTINEL)

        threading.Thread(target=producer, daemon=True).start()

        loop = asyncio.get_event_loop()

        while True:
            chunk = await loop.run_in_executor(None, q.get)
            if isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == "__error__":
                # Raise so caller can handle fallback logic instead of speaking error text
                raise Exception(chunk[1])
            if chunk is SENTINEL:
                break
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
            
            # Get model instance with specific API key
            model = self.get_model(model_name, api_key)
            
            # Start a chat session
            chat = model.start_chat(history=[])
            
            # Prepare the conversation
            if system_prompt:
                # Send system prompt as first message
                chat.send_message(f"System: {system_prompt}")
            
            # Send all messages
            for message in messages:
                if message["role"] == "user":
                    chat.send_message(message["content"])
                elif message["role"] == "assistant":
                    # For assistant messages, we need to handle them differently
                    # Gemini doesn't support adding assistant messages to history directly
                    pass
            
            # Get the last user message
            last_user_message = None
            for message in reversed(messages):
                if message["role"] == "user":
                    last_user_message = message["content"]
                    break
            
            if not last_user_message:
                raise Exception("No user message found in the conversation")
            
            # Generate response
            response = chat.send_message(
                last_user_message,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return {
                "content": response.text,
                "model": model_name,
                "response_time": response_time,
                "usage": {
                    "prompt_tokens": len(" ".join([msg["content"] for msg in messages]).split()),
                    "completion_tokens": len(response.text.split()),
                    "total_tokens": len(" ".join([msg["content"] for msg in messages]).split()) + len(response.text.split())
                },
                "finish_reason": "stop"
            }
            
        except Exception as e:
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

        # 🧠 Get model instance
        model = self.get_model(model_name, api_key)

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
            f"System: {agent_system_prompt}\n\n"
            f"Previous conversation:\n{history_text}\n\n"
            f"User: {user_input}\n\n"
            f"Note: Never repeat any question already asked. Be human-like and polite."
        )

        # 💬 Generate model response
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )

        # ⏱️ Timing
        end_time = time.time()
        response_time = end_time - start_time

        # 🪄 Update transcript in DB (use "message" key for consistency)
        if call_session and db:
            conversation_history.append({"role": "user", "message": user_input})
            conversation_history.append({"role": "assistant", "message": response.text})
            call_session.call_transcript = json.dumps(conversation_history)
            call_session.updated_at = datetime.utcnow()
            db.commit()

        return {
            "response": response.text,
            "model": model_name,
            "response_time": response_time,
            "usage": {
                "prompt_tokens": len(full_prompt.split()),
                "completion_tokens": len(response.text.split()),
                "total_tokens": len(full_prompt.split()) + len(response.text.split())
            }
        }

    except Exception as e:
        raise Exception(f"Error in Gemini agent conversation: {str(e)}")
        
# Create a singleton instance
gemini_service = GeminiService()
