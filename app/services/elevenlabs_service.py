"""
ElevenLabs Service Module
Handles text-to-speech operations using ElevenLabs API
"""

import requests
from app.core.config import settings
from typing import Dict, Optional, Any
import time

class ElevenLabsService:
    """Service class for handling ElevenLabs operations"""
    
    def __init__(self):
        self._api_key = None
        self._base_url = "https://api.elevenlabs.io/v1"
    
    def get_api_key(self):
        """Get ElevenLabs API key"""
        if self._api_key is None:
            api_key = settings.ELEVENLABS_API_KEY
            
            if not api_key:
                raise Exception("ElevenLabs API key not found. Please set ELEVENLABS_API_KEY in your config.")
            
            self._api_key = api_key
        
        return self._api_key
    
    def text_to_speech(self, text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM", 
                      model_id: str = "eleven_monolingual_v1", 
                      output_format: str = "mp3") -> bytes:
        """
        Convert text to speech using ElevenLabs API
        
        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID
            model_id: ElevenLabs model ID
            output_format: Output format (mp3, opus, aac, flac)
            
        Returns:
            Audio data as bytes
        """
        api_key = self.get_api_key()
        
        try:
            url = f"{self._base_url}/text-to-speech/{voice_id}"
            
            headers = {
                "Accept": f"audio/{output_format}",
                "Content-Type": "application/json",
                "xi-api-key": api_key
            }
            
            data = {
                "text": text,
                "model_id": model_id,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                return response.content
            else:
                raise Exception(f"ElevenLabs API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Error in ElevenLabs text-to-speech: {str(e)}")
    
    def get_available_voices(self) -> Dict[str, Any]:
        """
        Get list of available voices
        
        Returns:
            Dictionary with available voices
        """
        api_key = self.get_api_key()
        
        try:
            url = f"{self._base_url}/voices"
            
            headers = {
                "xi-api-key": api_key
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"ElevenLabs API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Error getting ElevenLabs voices: {str(e)}")

# Global instance
elevenlabs_service = ElevenLabsService()
