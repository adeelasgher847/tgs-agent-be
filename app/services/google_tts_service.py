"""
Google Cloud Text-to-Speech Service Module
Handles text-to-speech operations using Google Cloud TTS API
"""

from google.cloud import texttospeech
from app.core.config import settings
from typing import Optional
import os
import json
from app.core.logger import logger


class GoogleTTSService:
    """Service class for handling Google Cloud Text-to-Speech operations"""
    
    def __init__(self):
        self._client = None
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
                self._client = texttospeech.TextToSpeechClient()
                logger.info("✅ Google Cloud Text-to-Speech client initialized")
            except Exception as e:
                logger.error(f"⚠️ Failed to initialize Google TTS client: {e}")
                logger.warning("⚠️ Text-to-Speech will not be available without proper credentials")
        
        return self._client
    
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
                # Urdu - REALISTIC VOICES
                "ur": {
                    "male": "ur-PK-Wavenet-B",       # Male Urdu Pakistan
                    "female": "ur-PK-Wavenet-A"      # Female Urdu Pakistan
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
            # Urdu (using correct Wavenet voices)
            "ur": {
                "male": "ur-PK-Wavenet-B",       # Male Urdu
                "female": "ur-PK-Wavenet-A"      # Female Urdu
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
            "ur": "ur-PK"  # Fixed mapping for Urdu
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


# Global instance
google_tts_service = GoogleTTSService()

