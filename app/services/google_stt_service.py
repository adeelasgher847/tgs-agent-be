"""
Google Cloud Speech-to-Text Service for real-time transcription
Handles streaming audio from Twilio and returns transcriptions
"""

import os
import asyncio
import base64
from typing import Optional, Callable, Dict, Any
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1 import types
from app.core.config import settings
import json


class GoogleSTTService:
    """Service for handling Google Cloud Speech-to-Text streaming"""
    
    def __init__(self):
        """Initialize Google Speech-to-Text client"""
        # Set credentials from environment variable
        # Support both file path and JSON content string
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
                # It's JSON content - write to temporary  file
                import tempfile
                try:
                    # Create temporary file
                    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                        f.write(creds)
                        temp_path = f.name
                    
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_path
                    print(f"✅ Using Google Cloud credentials from JSON content (temp file: {temp_path})")
                except Exception as e:
                    print(f"⚠️ Error creating temp file for JSON credentials: {e}")
            else:
                # It's a file path - check if file exists
                if os.path.exists(creds):
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
                    print(f"✅ Using Google Cloud credentials from file: {creds}")
                else:
                    print(f"⚠️ Credentials file not found: {creds}")
        
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Speech client"""
        try:
            self.client = speech.SpeechClient()
            print("✅ Google Cloud Speech-to-Text client initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize Google Speech client: {e}")
            print("⚠️ Transcription will not be available without proper credentials")
    
    def get_streaming_config(
        self,
        language_code: str = None,
        sample_rate: int = None,
        encoding: str = None,
        enable_automatic_punctuation: bool = True,
        model: str = "phone_call",
        use_enhanced: bool = True,
        interim_results: bool = False
    ) -> types.StreamingRecognitionConfig:
        """
        Create streaming recognition configuration
        
        Args:
            language_code: BCP-47 language code (e.g., 'en-US', 'es-ES')
            sample_rate: Audio sample rate in Hz
            encoding: Audio encoding format
            enable_automatic_punctuation: Whether to add punctuation
            model: Recognition model to use
            use_enhanced: Whether to use enhanced model
        
        Returns:
            StreamingRecognitionConfig object
        """
        # Use defaults from settings if not provided
        language_code = language_code or settings.GOOGLE_STT_LANGUAGE_CODE
        sample_rate = sample_rate or settings.GOOGLE_STT_SAMPLE_RATE
        
        # Map encoding string to enum
        encoding_map = {
            "MULAW": speech.RecognitionConfig.AudioEncoding.MULAW,
            "LINEAR16": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "FLAC": speech.RecognitionConfig.AudioEncoding.FLAC,
            "AMR": speech.RecognitionConfig.AudioEncoding.AMR,
            "AMR_WB": speech.RecognitionConfig.AudioEncoding.AMR_WB,
            "OGG_OPUS": speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
            "SPEEX_WITH_HEADER_BYTE": speech.RecognitionConfig.AudioEncoding.SPEEX_WITH_HEADER_BYTE,
        }
        
        encoding_str = encoding or settings.GOOGLE_STT_ENCODING
        audio_encoding = encoding_map.get(encoding_str, speech.RecognitionConfig.AudioEncoding.MULAW)
        
        # Create recognition config (Vapi-style for phone calls)
        config = speech.RecognitionConfig(
            encoding=audio_encoding,
            sample_rate_hertz=sample_rate,
            language_code=language_code,
            enable_automatic_punctuation=enable_automatic_punctuation,
            model=model,
            use_enhanced=use_enhanced,
            # Add speech contexts for better phone call recognition
            speech_contexts=[
                speech.SpeechContext(
                    phrases=[
                        "hello", "hi", "hey",
                        "help", "assistance", "support",
                        "yes", "no", "okay", "sure",
                        "thank you", "thanks", "goodbye"
                    ],
                    boost=10.0
                )
            ],
        )
        
        # Create streaming config optimized for phone calls (Vapi-style)
        streaming_config = types.StreamingRecognitionConfig(
            config=config,
            interim_results=interim_results,  # Configurable interim results
            single_utterance=True,  # End stream after utterance (better for turn-based conversation)
        )
        
        return streaming_config
    
    async def transcribe_stream(
        self,
        audio_generator,
        language_code: str = None,
        on_interim_result: Optional[Callable] = None,
        on_final_result: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ):
        """
        Transcribe audio stream from Twilio
        
        Args:
            audio_generator: Async generator yielding audio chunks
            language_code: Language code for transcription
            on_interim_result: Callback for interim results
            on_final_result: Callback for final results
            on_error: Callback for errors
        """
        if not self.client:
            print("❌ Google Speech client not initialized")
            if on_error:
                await on_error("Google Speech client not initialized")
            return
        
        try:
            # Get streaming config
            streaming_config = self.get_streaming_config(language_code=language_code)
            
            # Create request generator
            async def request_generator():
                # First request with config
                yield types.StreamingRecognizeRequest(streaming_config=streaming_config)
                
                # Subsequent requests with audio
                async for audio_chunk in audio_generator:
                    if audio_chunk:
                        yield types.StreamingRecognizeRequest(audio_content=audio_chunk)
            
            # Start streaming recognition
            print("🎤 Starting Google Cloud STT streaming recognition...")
            
            # Create requests and get responses
            requests = request_generator()
            responses = self.client.streaming_recognize(requests)
            
            # Process responses
            for response in responses:
                if not response.results:
                    continue
                
                # Get the first result
                result = response.results[0]
                
                if not result.alternatives:
                    continue
                
                # Get the top alternative
                alternative = result.alternatives[0]
                transcript = alternative.transcript
                confidence = alternative.confidence if hasattr(alternative, 'confidence') else 0.0
                
                # Handle interim vs final results
                if result.is_final:
                    print(f"✅ Final transcript: '{transcript}' (confidence: {confidence:.2f})")
                    if on_final_result:
                        await on_final_result({
                            "transcript": transcript,
                            "confidence": confidence,
                            "is_final": True
                        })
                else:
                    print(f"⏳ Interim transcript: '{transcript}'")
                    if on_interim_result:
                        await on_interim_result({
                            "transcript": transcript,
                            "confidence": confidence,
                            "is_final": False
                        })
        
        except Exception as e:
            print(f"❌ Error in streaming transcription: {e}")
            import traceback
            traceback.print_exc()
            if on_error:
                await on_error(str(e))
    
    def create_streaming_request_generator(self, audio_queue):
        """
        Create a generator for streaming requests (Vapi-style)
        
        Args:
            audio_queue: Queue of audio chunks
            
        Yields:
            StreamingRecognizeRequest objects
        """
        # First request with config
        streaming_config = self.get_streaming_config()
        yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
        
        # Subsequent requests with audio
        while True:
            try:
                chunk = audio_queue.get_nowait()
                if chunk is None:  # Sentinel to stop
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except:
                break
    
    def transcribe_audio_chunk_streaming(
        self,
        audio_content: bytes,
        language_code: str = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio chunk using Google Cloud STT (Vapi-style approach)
        Uses simple recognize() API for short phone call audio chunks
        Fast and reliable - perfect for real-time phone conversations
        
        Args:
            audio_content: Raw audio bytes (MULAW format)
            language_code: Language code for transcription
        
        Returns:
            Dictionary with transcript, confidence, and is_final status
        """
        if not self.client:
            return {"error": "Google Speech client not initialized", "transcript": "", "confidence": 0.0}
        
        try:
            import sys
            language_code = language_code or settings.GOOGLE_STT_LANGUAGE_CODE
            
            # Create streaming config optimized for phone calls
            encoding_map = {
                "MULAW": speech.RecognitionConfig.AudioEncoding.MULAW,
                "LINEAR16": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            }
            audio_encoding = encoding_map.get(
                settings.GOOGLE_STT_ENCODING,
                speech.RecognitionConfig.AudioEncoding.MULAW
            )
            
            config = speech.RecognitionConfig(
                encoding=audio_encoding,
                sample_rate_hertz=settings.GOOGLE_STT_SAMPLE_RATE,
                language_code=language_code,
                enable_automatic_punctuation=True,
                model="phone_call",  # Phone call optimized model (Vapi uses this)
                use_enhanced=True,   # Enhanced model for better accuracy
                # Add common phrases for better recognition (Vapi-style)
                speech_contexts=[
                    speech.SpeechContext(
                        phrases=[
                            "hello", "hi", "hey",
                            "help", "assistance", "support",
                            "yes", "no", "okay", "sure",
                            "thank you", "thanks", "goodbye"
                        ],
                        boost=10.0  # Boost recognition of common phone words
                    )
                ],
            )
            
            # Use simple recognize() API instead of streaming
            # This is better for short phone call chunks (Vapi-style)
            audio = speech.RecognitionAudio(content=audio_content)
            
            print(f"🎙️ Sending {len(audio_content)} bytes to Google Cloud STT...")
            sys.stdout.flush()
            
            # Simple recognize() call - fast and reliable for short chunks
            response = self.client.recognize(config=config, audio=audio)
            
            print(f"📡 Received response from Google STT")
            sys.stdout.flush()
            
            # Process results
            if response.results:
                result = response.results[0]
                if result.alternatives:
                    alternative = result.alternatives[0]
                    transcript_text = alternative.transcript
                    confidence = alternative.confidence if hasattr(alternative, 'confidence') else 0.9
                    
                    print(f"📝 Transcript: '{transcript_text}' (confidence: {confidence:.2f})")
                    sys.stdout.flush()
                    
                    return {
                        "transcript": transcript_text,
                        "confidence": confidence,
                        "is_final": True
                    }
            
            print(f"⚠️ No transcript in response")
            sys.stdout.flush()
            return {"transcript": "", "confidence": 0.0, "is_final": True}
        
        except Exception as e:
            print(f"❌ Error in transcription: {e}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            return {"error": str(e), "transcript": "", "confidence": 0.0}


# Global service instance
google_stt_service = GoogleSTTService()

