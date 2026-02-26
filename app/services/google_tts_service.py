"""
Google Cloud Text-to-Speech Service Module
Handles text-to-speech operations using Google Cloud TTS API
"""

from google.cloud import texttospeech
from google.cloud import texttospeech_v1
from app.core.config import settings
from typing import Optional, AsyncIterator
import os
import json
import re
from app.core.logger import logger
from google.api_core.client_options import ClientOptions


class GoogleTTSService:
    """Service class for handling Google Cloud Text-to-Speech operations"""
    
    def __init__(self):
        self._client = None
        self._async_client = None
        self._initialize_credentials()
    
    def _initialize_credentials(self):
        """Initialize Google Cloud credentials (same as STT service)"""
        if settings.GOOGLE_APPLICATION_CREDENTIALS:
            creds = settings.GOOGLE_APPLICATION_CREDENTIALS.strip()
            
            # Check if it's JSON content (more robust check)
            is_json = False
            try:
                # Try to parse as JSON
                json.loads(creds)
                is_json = True
            except (json.JSONDecodeError, ValueError):
                # Not JSON, treat as file path
                is_json = False
            
            if is_json:
                # It's JSON content - write to temporary file
                import tempfile
                try:
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                        f.write(creds)
                        temp_path = f.name
                    
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
                    logger.info(f"✅ Google TTS: Using credentials from JSON content (temp file: {temp_path})")
                except Exception as e:
                    logger.error(f"⚠️ Google TTS: Error creating temp file for JSON credentials: {e}")
            else:
                # It's a file path - check if file exists
                if os.path.exists(creds):
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
                    logger.info(f"✅ Google TTS: Using credentials from file: {creds}")
                else:
                    logger.warning(f"⚠️ Google TTS: Credentials file not found: {creds}")
    
    def get_client(self):
        """Get Google Cloud TTS client"""
        if self._client is None:
            try:
                endpoint = (settings.CLOUD_TTS_ENDPOINT or "").strip()
                client_options = ClientOptions(api_endpoint=endpoint) if endpoint else None
                self._client = texttospeech.TextToSpeechClient(client_options=client_options)
                logger.info("✅ Google Cloud Text-to-Speech client initialized")
            except Exception as e:
                logger.error(f"⚠️ Failed to initialize Google TTS client: {e}")
                logger.warning("⚠️ Text-to-Speech will not be available without proper credentials")
        
        return self._client

    def get_async_client(self):
        """Get Google Cloud TTS async client (for bidirectional streaming)."""
        if self._async_client is None:
            try:
                endpoint = (settings.CLOUD_TTS_ENDPOINT or "").strip()
                client_options = ClientOptions(api_endpoint=endpoint) if endpoint else None
                self._async_client = texttospeech_v1.TextToSpeechAsyncClient(client_options=client_options)
                logger.info("✅ Google Cloud Text-to-Speech ASYNC client initialized")
            except Exception as e:
                logger.error(f"⚠️ Failed to initialize Google TTS async client: {e}")
                logger.warning("⚠️ Streaming TTS will not be available without proper credentials")
        return self._async_client
    
    def get_voice_name(self, language: str = "en", voice_type: str = "female", use_chirp3_hd: bool = False) -> str:
        """
        Get appropriate Google TTS voice name based on language and gender
        BEST REALISTIC VOICES - Using Chirp 3: HD, Studio, Neural2 voices
        
        Args:
            language: Language code (en, es, hi, ar, zh, ur)
            voice_type: Voice gender (male or female)
            use_chirp3_hd: Use Chirp 3: HD voices (ultra-realistic and high quality)
            
        Returns:
            Google Cloud TTS voice name
        """
        # Optional exact voice override (useful for A/B testing naturalness)
        # Example: en-US-Chirp3-HD-Achernar
        if getattr(settings, "GOOGLE_TTS_VOICE_NAME", "").strip():
            return settings.GOOGLE_TTS_VOICE_NAME.strip()

        # Chirp 3: HD voices - ULTRA REALISTIC + PREMIUM QUALITY (Google's latest AI TTS model)
        if use_chirp3_hd:
            chirp3_hd_voice_map = {
                # English voices - Chirp 3: HD Model
                "en": {
                    "male": "en-US-Chirp3-HD-Achird",       # Chirp 3: HD Male - Friendly (Ultra-realistic)
                    "female": "en-US-Chirp3-HD-Achernar"    # Chirp 3: HD Female - Soft (Ultra-realistic)
                },
                # Spanish voices - Gemini Flash
                "es": {
                    "male": "es-US-Journey-D",       # Gemini Flash Male Spanish
                    "female": "es-US-Journey-F"      # Gemini Flash Female Spanish
                },
                # Hindi voices - Fallback to Neural2
                "hi": {
                    "male": "hi-IN-Neural2-B",       # Fast Male Hindi
                    "female": "hi-IN-Neural2-A"      # Fast Female Hindi
                },
                # Arabic voices - Fallback to Wavenet
                "ar": {
                    "male": "ar-XA-Wavenet-B",       # Male Arabic
                    "female": "ar-XA-Wavenet-A"      # Female Arabic
                },
                # Chinese voices - Fallback to Wavenet
                "zh": {
                    "male": "cmn-CN-Wavenet-B",      # Male Mandarin
                    "female": "cmn-CN-Wavenet-A"     # Female Mandarin
                },
                # Urdu - Fallback to Hindi
                "ur": {
                    "male": "hi-IN-Neural2-B",       # Fast Male Hindi
                    "female": "hi-IN-Neural2-A"      # Fast Female Hindi
                }
            }
            
            language = language if language in chirp3_hd_voice_map else "en"
            voice_type = voice_type if voice_type in ["male", "female"] else "female"
            return chirp3_hd_voice_map[language][voice_type]
        
        # Google Cloud TTS voice mapping (Standard Neural2 voices)
        # Using NEURAL2 voices for SPEED + QUALITY balance (2x faster than Studio!)
        voice_map = {
            # English voices - FAST + HIGH QUALITY (Neural2 - 60% faster than Studio)
            "en": {
                "male": "en-US-Neural2-A",       # Fast Male US English (0.5s vs 1.0s)
                "female": "en-US-Neural2-C"      # Fast Female US English (0.5s vs 1.0s)
            },
            # Spanish voices - FAST + HIGH QUALITY
            "es": {
                "male": "es-ES-Neural2-B",       # Fast Male Spanish
                "female": "es-ES-Neural2-A"      # Fast Female Spanish
            },
            # Hindi voices - FAST + HIGH QUALITY
            "hi": {
                "male": "hi-IN-Neural2-B",       # Fast Male Hindi
                "female": "hi-IN-Neural2-A"      # Fast Female Hindi
            },
            # Arabic voices - Wavenet (good balance)
            "ar": {
                "male": "ar-XA-Wavenet-B",       # Male Arabic
                "female": "ar-XA-Wavenet-A"      # Female Arabic
            },
            # Chinese voices - Wavenet (good balance)
            "zh": {
                "male": "cmn-CN-Wavenet-B",      # Male Mandarin
                "female": "cmn-CN-Wavenet-A"     # Female Mandarin
            },
            # Urdu (using Hindi voices as fallback)
            "ur": {
                "male": "hi-IN-Neural2-B",       # Fast Male Hindi
                "female": "hi-IN-Neural2-A"      # Fast Female Hindi
            }
        }
        
        # Default to English if language not found
        language = language if language in voice_map else "en"
        voice_type = voice_type if voice_type in ["male", "female"] else "female"
        
        return voice_map[language][voice_type]
    
    def get_language_code(self, language: str = "en") -> str:
        """
        Get language code for Google TTS
        
        Args:
            language: Short language code (en, es, hi, etc.)
            
        Returns:
            Full language code (en-US, es-ES, etc.)
        """
        language_code_map = {
            "en": "en-US",
            "es": "es-ES",
            "hi": "hi-IN",
            "ar": "ar-XA",
            "zh": "cmn-CN",
            "ur": "hi-IN"  # Using Hindi as fallback for Urdu
        }
        
        return language_code_map.get(language, "en-US")
    
    def text_to_speech(
        self, 
        text: str, 
        language: str = "en",
        voice_type: str = "female",
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
        output_format: str = "mp3",
        use_chirp3_hd: bool = False
    ) -> bytes:
        """
        Convert text to speech using Google Cloud TTS API with Chirp 3: HD model
        
        Args:
            text: Text to convert to speech
            language: Language code (en, es, hi, ar, zh, ur)
            voice_type: Voice gender (male or female)
            speaking_rate: Speech speed (0.25 to 4.0, default 1.0)
            pitch: Voice pitch (-20.0 to 20.0, default 0.0)
            output_format: Output format (mp3, linear16, ogg_opus, mulaw, alaw)
            use_chirp3_hd: Use Chirp 3: HD model (ultra-realistic and high quality)
            
        Returns:
            Audio data as bytes
        """
        try:
            client = self.get_client()
            
            # Set the text input to be synthesized
            # Auto-detect SSML if text starts with <speak>
            if text.strip().startswith('<speak>'):
                synthesis_input = texttospeech.SynthesisInput(ssml=text)
            else:
                synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Get voice name and language code
            voice_name = self.get_voice_name(language, voice_type, use_chirp3_hd)
            language_code = self.get_language_code(language)
            
            # Build the voice request
            voice = texttospeech.VoiceSelectionParams(
                name=voice_name,
                language_code=language_code
            )
            
            # Map output format to Google TTS format
            audio_encoding_map = {
                "mp3": texttospeech.AudioEncoding.MP3,
                "linear16": texttospeech.AudioEncoding.LINEAR16,
                "ogg_opus": texttospeech.AudioEncoding.OGG_OPUS,
                "mulaw": texttospeech.AudioEncoding.MULAW,
                "alaw": texttospeech.AudioEncoding.ALAW
            }
            
            audio_encoding = audio_encoding_map.get(output_format, texttospeech.AudioEncoding.MP3)
            
            # Select the type of audio file you want returned
            # Enhanced configuration for better quality and telephony optimization
            audio_config = texttospeech.AudioConfig(
                audio_encoding=audio_encoding,
                speaking_rate=speaking_rate,
                pitch=pitch,
                sample_rate_hertz=24000 if output_format == "mp3" else 8000,  # 24kHz for MP3 (better quality), 8kHz for MULAW
                effects_profile_id=["telephony-class-application"],  # Optimize for phone calls/Twilio
                volume_gain_db=0.0  # Reset to 0.0 to prevent digital clipping/distortion
            )
            
            # Perform the text-to-speech request
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Return the audio content
            return response.audio_content
            
        except Exception as e:
            raise Exception(f"Error in Google TTS: {str(e)}")

    async def stream_text_to_speech(
        self,
        text: str,
        language: str = "en",
        voice_type: str = "female",
        speaking_rate: float = 1.0,
        output_format: str = "mulaw",
        use_chirp3_hd: bool = True,
        sample_rate_hz: int = 8000,
    ) -> AsyncIterator[bytes]:
        """
        Bidirectional streaming TTS (StreamingSynthesize).
        Yields audio bytes chunks as they arrive to reduce latency.

        Notes:
        - Streaming synthesize is compatible with Chirp 3: HD voices.
        - For HD voices, SSML-like content may need to go through `markup`.
        """
        client = self.get_async_client()
        if client is None:
            raise Exception("Google TTS async client not available")

        # Streaming is only supported for specific voices (Chirp 3: HD per docs).
        voice_name = self.get_voice_name(language, voice_type, use_chirp3_hd=True if use_chirp3_hd else False)
        language_code = self.get_language_code(language)
        if use_chirp3_hd and "Chirp3" not in voice_name:
            raise Exception(f"Streaming TTS requires Chirp3-HD voice; got '{voice_name}'")

        # StreamingAudioConfig supports PCM, ALAW, MULAW, OGG_OPUS.
        encoding_map = {
            "mulaw": texttospeech_v1.AudioEncoding.MULAW,
            "alaw": texttospeech_v1.AudioEncoding.ALAW,
            "linear16": texttospeech_v1.AudioEncoding.LINEAR16,
            "pcm": texttospeech_v1.AudioEncoding.LINEAR16,  # alias
            "ogg_opus": texttospeech_v1.AudioEncoding.OGG_OPUS,
        }
        audio_encoding = encoding_map.get(output_format.lower(), texttospeech_v1.AudioEncoding.MULAW)

        streaming_config = texttospeech_v1.StreamingSynthesizeConfig(
            voice=texttospeech_v1.VoiceSelectionParams(
                name=voice_name,
                language_code=language_code,
            ),
            streaming_audio_config=texttospeech_v1.StreamingAudioConfig(
                audio_encoding=audio_encoding,
                sample_rate_hertz=int(sample_rate_hz),
                speaking_rate=float(speaking_rate),
            ),
        )

        # IMPORTANT:
        # In streaming_synthesize, HD "markup" is NOT SSML. If we send SSML tags here,
        # some voices may literally speak them (e.g. "less than prosody...").
        # So we ALWAYS stream plain text (strip any SSML/XML tags defensively).
        text_stripped = (text or "").strip()
        if "<" in text_stripped and ">" in text_stripped:
            text_stripped = re.sub(r"<[^>]+>", "", text_stripped)
            text_stripped = re.sub(r"\s+", " ", text_stripped).strip()

        async def request_generator():
            # First request must be config only
            yield texttospeech_v1.StreamingSynthesizeRequest(streaming_config=streaming_config)
            # Then input
            inp = texttospeech_v1.StreamingSynthesisInput(text=text_stripped)
            yield texttospeech_v1.StreamingSynthesizeRequest(input=inp)

        stream = await client.streaming_synthesize(requests=request_generator())
        async for response in stream:
            if response and getattr(response, "audio_content", None):
                yield response.audio_content


# Global instance
google_tts_service = GoogleTTSService()

