"""
ElevenLabs Service Module
Handles text-to-speech operations using ElevenLabs API
"""

import asyncio
import httpx
import requests
from app.core.config import settings
from app.core.logger import logger
from typing import AsyncIterator, Dict, Any, Iterator

class ElevenLabsService:
    """Service class for handling ElevenLabs operations"""
    
    def __init__(self):
        self._api_key = None
        self._base_url = "https://api.elevenlabs.io/v1"
        self._session = requests.Session()
        self._async_clients: dict[int, httpx.AsyncClient] = {}

    def _get_async_client(self, timeout_sec: float = 25.0) -> httpx.AsyncClient:
        """
        Get or create a cached httpx.AsyncClient keyed by the active event loop.
        Pools TCP/TLS connections to avoid recreating handshakes per TTS chunk.
        """
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        client = self._async_clients.get(loop_id)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5.0, read=timeout_sec, write=10.0, pool=10.0),
                http2=False,
                limits=httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=20,
                    keepalive_expiry=30.0,
                ),
            )
            self._async_clients[loop_id] = client
        return client

    async def aclose(self) -> None:
        """Explicit lifecycle cleanup: close all cached AsyncClients."""
        for client in list(self._async_clients.values()):
            try:
                if not client.is_closed:
                    await client.aclose()
            except Exception:
                pass
        self._async_clients.clear()
    
    def get_api_key(self, override: str | None = None) -> str:
        """Get ElevenLabs API key (tenant BYO override or platform env)."""
        if override and str(override).strip():
            return str(override).strip()
        if self._api_key is None:
            env_key = (settings.ELEVENLABS_API_KEY or "").strip()
            api_key = env_key

            if not api_key:
                raise RuntimeError("ElevenLabs API key not found. Please set ELEVENLABS_API_KEY in your config.")
            
            self._api_key = api_key
        
        return self._api_key

    def _default_voice_settings(self) -> Dict[str, Any]:
        # Tuned defaults for low latency conversational calls.
        # Keep stability moderate for naturalness while preserving fast generation.
        return {
            "stability": 0.45,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
            "speed": 1.0,
        }

    def _build_tts_request(
        self,
        *,
        text: str,
        model_id: str,
        voice_settings: Dict[str, Any] | None,
        language_code: str | None = None,
        previous_text: str | None = None,
        next_text: str | None = None,
        previous_request_ids: list[str] | None = None,
        next_request_ids: list[str] | None = None,
        apply_text_normalization: str | None = None,
        apply_language_text_normalization: bool | None = None,
    ) -> Dict[str, Any]:
        settings_payload = self._default_voice_settings()
        if voice_settings:
            settings_payload.update(voice_settings)
        payload: Dict[str, Any] = {
            "text": text,
            "model_id": model_id,
            "voice_settings": settings_payload,
        }
        if language_code:
            payload["language_code"] = language_code
        if previous_text:
            payload["previous_text"] = previous_text
        if next_text:
            payload["next_text"] = next_text
        if previous_request_ids:
            payload["previous_request_ids"] = previous_request_ids[:3]
        if next_request_ids:
            payload["next_request_ids"] = next_request_ids[:3]
        if apply_text_normalization in {"auto", "on", "off"}:
            payload["apply_text_normalization"] = apply_text_normalization
        if apply_language_text_normalization is not None:
            payload["apply_language_text_normalization"] = bool(apply_language_text_normalization)
        return payload

    def _accept_header_for_output_format(self, output_format: str) -> str:
        if output_format.startswith("ulaw"):
            return "audio/basic"
        if output_format.startswith("mp3"):
            return "audio/mpeg"
        if output_format.startswith("ogg"):
            return "audio/ogg"
        if output_format.startswith("pcm"):
            return "audio/wav"
        return "application/octet-stream"

    def text_to_speech(
        self,
        text: str,
        voice_id: str,
        model_id: str = "eleven_flash_v2_5",
        output_format: str = "ulaw_8000",
        voice_settings: Dict[str, Any] | None = None,
        language_code: str | None = None,
        previous_text: str | None = None,
        next_text: str | None = None,
        previous_request_ids: list[str] | None = None,
        next_request_ids: list[str] | None = None,
        apply_text_normalization: str | None = None,
        apply_language_text_normalization: bool | None = None,
        optimize_streaming_latency: int = 4,
        request_timeout_seconds: int = 25,
        api_key_override: str | None = None,
    ) -> bytes:
        """
        Convert text to speech using ElevenLabs low-latency stream endpoint.

        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID
            model_id: ElevenLabs model ID (default: eleven_flash_v2_5 for low latency)
            output_format: ElevenLabs output format (default ulaw_8000 for telephony/Twilio)
            voice_settings: Optional provider-specific voice settings
            optimize_streaming_latency: Lower response time, range 0-4
            request_timeout_seconds: HTTP timeout

        Returns:
            Audio data as bytes
        """
        api_key = self.get_api_key(api_key_override)

        try:
            url = f"{self._base_url}/text-to-speech/{voice_id}/stream"
            safe_optimize = max(0, min(4, int(optimize_streaming_latency)))
            headers = {
                "Accept": self._accept_header_for_output_format(output_format),
                "Content-Type": "application/json",
                "xi-api-key": api_key
            }
            data = self._build_tts_request(
                text=text,
                model_id=model_id,
                voice_settings=voice_settings,
                language_code=language_code,
                previous_text=previous_text,
                next_text=next_text,
                previous_request_ids=previous_request_ids,
                next_request_ids=next_request_ids,
                apply_text_normalization=apply_text_normalization,
                apply_language_text_normalization=apply_language_text_normalization,
            )

            params = {
                "output_format": output_format,
                "optimize_streaming_latency": safe_optimize,
            }
            response = self._session.post(
                url,
                headers=headers,
                params=params,
                json=data,
                timeout=request_timeout_seconds,
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            raise RuntimeError("ElevenLabs text-to-speech request failed.") from exc

    def stream_text_to_speech(
        self,
        text: str,
        voice_id: str,
        model_id: str = "eleven_flash_v2_5",
        output_format: str = "ulaw_8000",
        voice_settings: Dict[str, Any] | None = None,
        language_code: str | None = None,
        previous_text: str | None = None,
        next_text: str | None = None,
        previous_request_ids: list[str] | None = None,
        next_request_ids: list[str] | None = None,
        apply_text_normalization: str | None = None,
        apply_language_text_normalization: bool | None = None,
        optimize_streaming_latency: int = 4,
        request_timeout_seconds: int = 25,
        chunk_size: int = 320,
        api_key_override: str | None = None,
    ) -> Iterator[bytes]:
        """
        Stream ElevenLabs synthesized audio as byte chunks.
        """
        api_key = self.get_api_key(api_key_override)
        safe_optimize = max(0, min(4, int(optimize_streaming_latency)))
        url = f"{self._base_url}/text-to-speech/{voice_id}/stream"
        headers = {
            "Accept": self._accept_header_for_output_format(output_format),
            "Content-Type": "application/json",
            "xi-api-key": api_key,
        }
        data = self._build_tts_request(
            text=text,
            model_id=model_id,
            voice_settings=voice_settings,
            language_code=language_code,
            previous_text=previous_text,
            next_text=next_text,
            previous_request_ids=previous_request_ids,
            next_request_ids=next_request_ids,
            apply_text_normalization=apply_text_normalization,
            apply_language_text_normalization=apply_language_text_normalization,
        )
        params = {
            "output_format": output_format,
            "optimize_streaming_latency": safe_optimize,
        }

        try:
            with self._session.post(
                url,
                headers=headers,
                params=params,
                json=data,
                timeout=request_timeout_seconds,
                stream=True,
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        yield chunk
        except requests.RequestException as exc:
            raise RuntimeError("ElevenLabs streaming TTS request failed.") from exc

    async def async_stream_text_to_speech(
        self,
        text: str,
        voice_id: str,
        model_id: str = "eleven_flash_v2_5",
        output_format: str = "ulaw_8000",
        voice_settings: Dict[str, Any] | None = None,
        language_code: str | None = None,
        previous_text: str | None = None,
        next_text: str | None = None,
        previous_request_ids: list | None = None,
        next_request_ids: list | None = None,
        apply_text_normalization: str | None = None,
        apply_language_text_normalization: bool | None = None,
        optimize_streaming_latency: int = 4,
        request_timeout_seconds: int = 25,
        chunk_size: int = 320,
        api_key_override: str | None = None,
    ) -> AsyncIterator[bytes]:
        """
        True async streaming via httpx.AsyncClient.
        Yields raw audio bytes without blocking the event loop.
        Used in the hot TTS path to eliminate sync-request stutter.
        """

        api_key = self.get_api_key(api_key_override)
        safe_optimize = max(0, min(4, int(optimize_streaming_latency)))
        url = f"{self._base_url}/text-to-speech/{voice_id}/stream"
        headers = {
            "Accept": self._accept_header_for_output_format(output_format),
            "Content-Type": "application/json",
            "xi-api-key": api_key,
        }
        data = self._build_tts_request(
            text=text,
            model_id=model_id,
            voice_settings=voice_settings,
            language_code=language_code,
            previous_text=previous_text,
            next_text=next_text,
            previous_request_ids=previous_request_ids,
            next_request_ids=next_request_ids,
            apply_text_normalization=apply_text_normalization,
            apply_language_text_normalization=apply_language_text_normalization,
        )
        params = {
            "output_format": output_format,
            "optimize_streaming_latency": safe_optimize,
        }
        client = self._get_async_client(timeout_sec=float(request_timeout_seconds))
        try:
            async with client.stream(
                "POST", url, headers=headers, params=params, json=data
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size):
                    if chunk:
                        yield chunk
        except Exception as exc:
            raise RuntimeError("ElevenLabs async streaming TTS request failed.") from exc

    def get_available_voices(self) -> Dict[str, Any]:
        """
        Get list of available voices
        
        Returns:
            Dictionary with available voices
        """
        api_key = None
        try:
            api_key = self.get_api_key()
        except Exception as e:
            logger.warning("ElevenLabs: No API key found or invalid key, attempting public fetch: %s", e)

        try:
            url = f"{self._base_url}/voices"
            
            headers = {}
            if api_key:
                headers["xi-api-key"] = api_key
            
            response = self._session.get(url, headers=headers, timeout=20)
            
            # If the request fails with 401 Unauthorized but we provided a key,
            # retry without the key to fetch public voices.
            if response.status_code == 401 and api_key:
                logger.warning("ElevenLabs: Configured API key was unauthorized (401). Retrying with public fetch...")
                headers.pop("xi-api-key", None)
                response = self._session.get(url, headers=headers, timeout=20)
                
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise RuntimeError("Failed to fetch ElevenLabs voices.") from exc

# Global instance
elevenlabs_service = ElevenLabsService()
