"""
Groq Service Module
Handles all Groq-related operations including text generation
Groq uses OpenAI-compatible API
"""

from openai import OpenAI
from app.core.config import settings
from typing import List, Dict, Optional, Any
import time
import json

class GroqService:
    """Service class for handling Groq operations"""
    
    def __init__(self):
        self._clients = {}  # Store clients by API key
        self._current_api_key = None
    
    def get_client(self, api_key: str = None):
        """Get or create Groq client with specific API key"""
        # Use provided API key or fall back to global setting
        key_to_use = api_key or settings.GROQ_API_KEY
        
        if not key_to_use:
            raise Exception("Groq API key not found. Please provide an API key or set GROQ_API_KEY in your config.")
        
        # Return existing client or create new one for this API key
        if key_to_use not in self._clients:
            self._clients[key_to_use] = OpenAI(
                api_key=key_to_use,
                base_url="https://api.groq.com/openai/v1"
            )
        
        return self._clients[key_to_use]
    
    def generate_text(self, prompt: str, system_prompt: str = None, 
                     model_name: str = "llama-3.3-70b-versatile", 
                     temperature: float = 0.7, 
                     max_tokens: int = 1000,
                     api_key: str = None) -> Dict[str, Any]:
        """
        Generate text using Groq API
        
        Args:
            prompt: The input prompt for text generation
            system_prompt: System prompt to set the context
            model_name: Groq model to use (default: llama-3.3-70b-versatile)
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
            raise Exception(f"Error in Groq text generation: {str(e)}")
    
    async def stream_text(self, prompt: str, system_prompt: str = None,
                          model_name: str = "llama-3.3-70b-versatile",
                          temperature: float = 0.7,
                          max_tokens: int = 1000,
                          api_key: str = None):
        """Yield text chunks from Groq as they arrive (streaming).
        This returns an async iterator of strings.
        """
        import asyncio
        
        try:
            # Get client instance with specific API key
            client = self.get_client(api_key)
            
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Create streaming response
            stream = await asyncio.to_thread(
                client.chat.completions.create,
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            # Stream chunks
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise Exception(f"Error in Groq text streaming: {str(e)}")
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       system_prompt: str = None, 
                       model_name: str = "llama-3.3-70b-versatile", 
                       temperature: float = 0.7,
                       max_tokens: int = 1000,
                       api_key: str = None) -> Dict[str, Any]:
        """
        Generate chat completion using Groq API
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: System prompt to use for the conversation
            model_name: Groq model to use
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
            raise Exception(f"Error in Groq chat completion: {str(e)}")
    
    def process_agent_conversation(self, user_input: str, 
                                 agent_system_prompt: str = "You are a helpful assistant.",
                                 conversation_history: List[Dict[str, str]] = None,
                                 model_name: str = "llama-3.3-70b-versatile",
                                 temperature: float = 0.7,
                                 max_tokens: int = 1000,
                                 api_key: str = None) -> Dict[str, Any]:
        """
        Process agent conversation using Groq API
        
        Args:
            user_input: Current user input
            agent_system_prompt: System prompt for the agent
            conversation_history: Previous conversation messages
            model_name: Groq model to use
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
            raise Exception(f"Error in Groq agent conversation: {str(e)}")

# Create a singleton instance
groq_service = GroqService()

