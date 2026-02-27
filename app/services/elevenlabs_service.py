"""
ElevenLabs Service Module
Handles text-to-speech operations using ElevenLabs API
"""

import requests
import asyncio
import threading
from app.core.config import settings
from typing import Dict, Optional, Any, AsyncIterator, List
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

    def stream_mulaw_8000(
        self,
        text: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_multilingual_v2",
        language_code: Optional[str] = None,
        optimize_streaming_latency: Optional[int] = 2,
    ) -> bytes:
        """
        Generate 8kHz μ-law (ulaw_8000) audio optimized for Twilio.

        This uses ElevenLabs' /stream endpoint with output_format=ulaw_8000 so
        the result can be sent directly over Twilio Media Streams without
        additional re-encoding.
        """
        api_key = self.get_api_key()
        try:
            params = []
            params.append("output_format=ulaw_8000")
            if optimize_streaming_latency is not None:
                params.append(f"optimize_streaming_latency={optimize_streaming_latency}")
            query = "&".join(params)
            url = f"{self._base_url}/text-to-speech/{voice_id}/stream"
            if query:
                url = f"{url}?{query}"

            headers = {
                "Content-Type": "application/json",
                "xi-api-key": api_key,
            }

            body: Dict[str, Any] = {
                "text": text,
                "model_id": model_id,
            }
            if language_code:
                body["language_code"] = language_code

            # For now we read the whole body into memory; the audio is already
            # ulaw 8kHz and ready for Twilio.
            resp = requests.post(url, headers=headers, json=body)
            if resp.status_code == 200:
                return resp.content
            raise Exception(f"ElevenLabs stream_ulaw_8000 error: {resp.status_code} - {resp.text}")
        except Exception as e:
            raise Exception(f"Error in ElevenLabs stream_mulaw_8000: {str(e)}")

    async def stream_mulaw_8000_iter(
        self,
        text: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_multilingual_v2",
        language_code: Optional[str] = None,
        optimize_streaming_latency: Optional[int] = 3,
    ) -> AsyncIterator[bytes]:
        """
        Async iterator over 8kHz μ-law audio chunks from the ElevenLabs /stream endpoint.
        This is used for true low-latency streaming with Twilio:
        - output_format=ulaw_8000 (no extra transcoding)
        - small chunks are forwarded to our TTS pipeline as they arrive.
        """
        api_key = self.get_api_key()
        loop = asyncio.get_event_loop()
        queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()

        def _worker():
            try:
                params = []
                params.append("output_format=ulaw_8000")
                if optimize_streaming_latency is not None:
                    params.append(f"optimize_streaming_latency={optimize_streaming_latency}")
                query = "&".join(params)
                url = f"{self._base_url}/text-to-speech/{voice_id}/stream"
                if query:
                    url = f"{url}?{query}"

                headers = {
                    "Content-Type": "application/json",
                    "xi-api-key": api_key,
                }

                body: Dict[str, Any] = {
                    "text": text,
                    "model_id": model_id,
                }
                if language_code:
                    body["language_code"] = language_code

                with requests.post(url, headers=headers, json=body, stream=True) as resp:
                    if resp.status_code != 200:
                        asyncio.run_coroutine_threadsafe(
                            queue.put(None), loop
                        )
                        return
                    for chunk in resp.iter_content(chunk_size=1024):
                        if not chunk:
                            continue
                        asyncio.run_coroutine_threadsafe(
                            queue.put(chunk), loop
                        )
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)
            except Exception:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
    
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

    # Curated tags we consider suitable for call/phone/support/assistant use cases
    _CURATED_TAGS = frozenset({"call", "phone", "support", "assistant", "professional", "customer service"})

    def list_curated_voices(self, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Return a curated list of voices for UI dropdown: call/phone/support/assistant oriented.
        Uses get_available_voices(), filters by optional language and by tags (call, phone, support, assistant).
        Return shape: [{ id, name, language, labels }].
        """
        raw = self.get_available_voices()
        voices_list = raw.get("voices") if isinstance(raw, dict) else []
        if not isinstance(voices_list, list):
            return []

        result: List[Dict[str, Any]] = []
        for v in voices_list:
            if not isinstance(v, dict):
                continue
            voice_id = v.get("voice_id") or v.get("id")
            name = v.get("name") or ""
            labels = v.get("labels") or {}
            if isinstance(labels, dict):
                pass
            else:
                labels = {} if labels is None else {"raw": str(labels)}
            lang = (labels.get("language") or labels.get("locale") or "").strip() or None
            if not lang and isinstance(v.get("locale"), str):
                lang = v.get("locale")
            if language is not None:
                lang_normalized = (lang or "en").split("-")[0].split("_")[0].lower()
                if lang_normalized != language.lower().strip():
                    continue
            result.append({
                "id": voice_id,
                "name": name,
                "language": lang or "en",
                "labels": labels,
            })
        return result

    def list_background_presets(self) -> List[Dict[str, Any]]:
        """
        Return static list of ElevenLabs background presets for UI dropdown.
        Shape: [{ id, name, description }].
        """
        return [
            {"id": "el_office_1", "name": "Busy Call Center", "description": "Office / call center ambience"},
            {"id": "el_cafe_1", "name": "Cafe Ambience", "description": "Cafe background noise"},
            {"id": "none", "name": "No Background Noise", "description": "No background sound"},
        ]

# Global instance
elevenlabs_service = ElevenLabsService()
