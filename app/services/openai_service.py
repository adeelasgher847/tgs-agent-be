"""
OpenAI Service Module
Handles all OpenAI-related operations including chat completions and text-to-speech
"""

import openai
from openai import OpenAI
from app.core.config import settings
from typing import List, Dict, Optional, Any
import time
import json

class OpenAIService:
    """Service class for handling OpenAI operations"""
    
    def __init__(self):
        self._client = None
    
    def get_client(self):
        """Get or create OpenAI client"""
        if self._client is None:
            api_key = settings.OPENAI_API_KEY
            
            if not api_key:
                raise Exception("OpenAI API key not found. Please set OPENAI_API_KEY in your config.")
            
            self._client = OpenAI(api_key=api_key)
        
        return self._client
    def chat_completion(self, messages: List[Dict[str, str]], system_prompt: str = None, 
                       model: str = "gpt-3.5-turbo", max_tokens: int = 150) -> Dict[str, Any]:
        """
        Generate chat completion using OpenAI API
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: System prompt to use for the conversation
            model: OpenAI model to use
            max_tokens: Maximum tokens for response
            
        Returns:
            Dictionary with response content and metadata
        """
        client = self.get_client()
        
        try:
            # Prepare messages with system prompt
            api_messages = []
            if system_prompt:
                api_messages.append({"role": "system", "content": system_prompt})
            
            api_messages.extend(messages)
            
            # Make API call
            response = client.chat.completions.create(
                model=model,
                messages=api_messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return {
                "content": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            raise Exception(f"Error in OpenAI chat completion: {str(e)}")
    
    def text_to_speech(self, text: str, voice: str = "alloy", 
                      model: str = "tts-1", output_format: str = "mp3") -> bytes:
        """
        Convert text to speech using OpenAI TTS API
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            model: TTS model to use
            output_format: Output format (mp3, opus, aac, flac)
            
        Returns:
            Audio data as bytes
        """
        client = self.get_client()
        
        try:
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
            max_tokens=200
        )
        
        response_time = time.time() - start_time
        
        return {
            "response": response["content"],
            "response_time": response_time,
            "usage": response["usage"],
            "model": response["model"]
        }

# Global instance
openai_service = OpenAIService()
