"""
OpenAI Service Module
Handles all OpenAI-related operations including text generation and chat completions
"""

from openai import OpenAI
from app.core.config import settings
from typing import List, Dict, Optional, Any
import time
import json

class OpenAIService:
    """Service class for handling OpenAI operations"""
    
    def __init__(self):
        self._clients = {}  # Store clients by API key
        self._current_api_key = None
    
    def get_client(self, api_key: str = None):
        """Get or create OpenAI client with specific API key"""
        # Use provided API key or fall back to global setting
        key_to_use = api_key or settings.OPENAI_API_KEY
        
        if not key_to_use:
            raise Exception("OpenAI API key not found. Please provide an API key or set OPENAI_API_KEY in your config.")
        
        # Return existing client or create new one for this API key
        if key_to_use not in self._clients:
            self._clients[key_to_use] = OpenAI(api_key=key_to_use)
        
        return self._clients[key_to_use]
    
    def generate_text(self, prompt: str, system_prompt: str = None, 
                     model_name: str = "gpt-3.5-turbo", 
                     temperature: float = 0.7, 
                     max_tokens: int = 1000,
                     api_key: str = None) -> Dict[str, Any]:
        """
        Generate text using OpenAI API
        
        Args:
            prompt: The input prompt for text generation
            system_prompt: System prompt to set the context
            model_name: OpenAI model to use
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens for response
            api_key: Model-specific API key (optional)
            
        Returns:
            Dictionary with response content and metadata
        """
        try:
            start_time = time.time()
            
            # Get client instance with specific API key
            client = self.get_client(api_key)
            
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Generate content
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "response_time": response_time,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            raise Exception(f"Error in OpenAI text generation: {str(e)}")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       system_prompt: str = None, 
                       model_name: str = "gpt-3.5-turbo", 
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       api_key: str = None) -> Dict[str, Any]:
        """
        Generate chat completion using OpenAI API
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: System prompt to use for the conversation
            model_name: OpenAI model to use
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens for response
            api_key: Model-specific API key (optional)
            
        Returns:
            Dictionary with response content and metadata
        """
        try:
            start_time = time.time()
            
            # Get client instance with specific API key
            client = self.get_client(api_key)
            
            # Prepare messages with system prompt
            api_messages = []
            if system_prompt:
                api_messages.append({"role": "system", "content": system_prompt})
            
            api_messages.extend(messages)
            
            # Generate chat completion
            response = client.chat.completions.create(
                model=model_name,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "response_time": response_time,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            raise Exception(f"Error in OpenAI chat completion: {str(e)}")
    
    def process_agent_conversation(self, user_input: str, 
                                 agent_system_prompt: str = "You are a helpful assistant.",
                                 conversation_history: List[Dict[str, str]] = None,
                                 model_name: str = "gpt-3.5-turbo",
                                 temperature: float = 0.7,
                                 max_tokens: int = 1000,
                                 api_key: str = None) -> Dict[str, Any]:
        """
        Process agent conversation using OpenAI API
        
        Args:
            user_input: Current user input
            agent_system_prompt: System prompt for the agent
            conversation_history: Previous conversation messages
            model_name: OpenAI model to use
            temperature: Temperature setting (0.0 to 1.0)
            max_tokens: Maximum tokens for response
            api_key: Model-specific API key (optional)
            
        Returns:
            Dictionary with response content and metadata
        """
        try:
            start_time = time.time()
            
            # Get client instance with specific API key
            client = self.get_client(api_key)
            
            # Prepare messages
            messages = []
            if agent_system_prompt:
                messages.append({"role": "system", "content": agent_system_prompt})
            
            # Add conversation history
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            # Generate response
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return {
                "response": response.choices[0].message.content,
                "model": response.model,
                "response_time": response_time,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            raise Exception(f"Error in OpenAI agent conversation: {str(e)}")
    
    async def generate_streaming_text(self, prompt: str, system_prompt: str = None, model_name: str = None, temperature: float = 0.7, max_tokens: int = 1000, api_key: str = None):
        """
        Generate streaming text response from OpenAI
        """
        try:
            from datetime import datetime
            import json
            import asyncio

            start_time = time.time()

            # 🧠 Get client
            client = self.get_client(api_key)

            # 🗣️ Prepare message context
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # 💬 Generate streaming response
            print(f"🎯 Starting OpenAI streaming response...")
            
            # Generate response with streaming
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True  # Enable streaming
            )
            
            # Stream the response
            response_text = ""
            chunk_count = 0
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    response_text += content
                    chunk_count += 1
                    
                    yield {
                        "content": content,
                        "chunk_id": chunk_count,
                        "is_final": False
                    }
            
            # Send final chunk
            yield {
                "content": "",
                "chunk_id": chunk_count + 1,
                "is_final": True,
                "response_time": time.time() - start_time
            }
            
        except Exception as e:
            print(f"❌ Error in OpenAI streaming service: {e}")
            import traceback
            traceback.print_exc()
            yield {
                "content": "",
                "error": str(e),
                "is_final": True
            }
    
    def text_to_speech(self, text: str, voice: str = "alloy", 
                      model: str = "tts-1", output_format: str = "mp3",
                      api_key: str = None) -> bytes:
        """
        Convert text to speech using OpenAI TTS API
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS model to use
            output_format: Output format (mp3, opus, aac, flac)
            api_key: Model-specific API key (optional)
            
        Returns:
            Audio data as bytes
        """
        try:
            # Get client instance with specific API key
            client = self.get_client(api_key)
            
            response = client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format=output_format
            )
            
            return response.content
            
        except Exception as e:
            raise Exception(f"Error in OpenAI text-to-speech: {str(e)}")
    
    def process_agent_conversation(self, user_input: str, agent_system_prompt: str, 
                                 conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process a conversation turn with an agent
        
        Args:
            user_input: User's speech input (transcribed text)
            agent_system_prompt: Agent's system prompt
            conversation_history: Previous conversation messages
            
        Returns:
            Dictionary with agent response and metadata
        """
        start_time = time.time()
        
        # Prepare messages
        messages = []
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_input})
        
        # Get response from OpenAI
        response = self.chat_completion(
            messages=messages,
            system_prompt=agent_system_prompt,
            max_tokens=100
        )
        
        response_time = time.time() - start_time
        
        return {
            "response": response["content"],
            "response_time": response_time,
            "usage": response["usage"],
            "model": response["model"]
        }

# Create a singleton instance
openai_service = OpenAIService()
