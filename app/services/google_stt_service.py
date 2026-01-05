"""
Google Cloud Speech-to-Text Service for real-time transcription
Handles streaming audio from Twilio and returns transcriptions
"""

import os
import asyncio
import base64
import time
from typing import Optional, Callable, Dict, Any
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1 import types
from google.api_core import exceptions as gcp_exceptions
from app.core.config import settings
import json
from app.core.logger import logger


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
                    logger.info(f"✅ Using Google Cloud credentials from JSON content (temp file: {temp_path})")
                except Exception as e:
                    logger.error(f"⚠️ Error creating temp file for JSON credentials: {e}")
            else:
                # It's a file path - check if file exists
                if os.path.exists(creds):
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds
                    logger.info(f"✅ Using Google Cloud credentials from file: {creds}")
                else:
                    logger.warning(f"⚠️ Credentials file not found: {creds}")
        
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Speech client"""
        try:
            self.client = speech.SpeechClient()
            logger.info("✅ Google Cloud Speech-to-Text client initialized")
        except Exception as e:
            logger.error(f"⚠️ Failed to initialize Google Speech client: {e}")
            logger.warning("⚠️ Transcription will not be available without proper credentials")
    
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
            interim_results=True,  # Enable interim results for real-time transcription
            single_utterance=False,  # Keep stream alive for continuous speech
        )
        
        return streaming_config
    
    class StreamingSTTSession:
        """Manage a single long-lived Google streaming_recognize session.
        Uses Google built-in endpointing (VAD) to emit interim and final results.
        """
        def __init__(self, client: speech.SpeechClient, config: "GoogleSTTService", language_code: str = None, encoding: str = None, sample_rate: int = None, interim_results: bool = True, single_utterance: bool = False):
            import queue
            import threading
            self._client = client
            self._language_code = language_code or settings.GOOGLE_STT_LANGUAGE_CODE
            self._encoding = (encoding or settings.GOOGLE_STT_ENCODING)
            self._sample_rate = sample_rate or settings.GOOGLE_STT_SAMPLE_RATE
            self._interim_results = interim_results
            self._single_utterance = single_utterance
            self._audio_q: "queue.Queue[Optional[bytes]]" = queue.Queue()
            self._results_q: "queue.Queue[dict]" = queue.Queue()
            self._closed = False
            self._task_started = False
            self._thread: Optional[threading.Thread] = None

            # Build configs
            encoding_map = {
                "MULAW": speech.RecognitionConfig.AudioEncoding.MULAW,
                "LINEAR16": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            }
            audio_encoding = encoding_map.get(self._encoding, speech.RecognitionConfig.AudioEncoding.MULAW)

            self._recognition_config = speech.RecognitionConfig(
                encoding=audio_encoding,
                sample_rate_hertz=self._sample_rate,
                language_code=self._language_code,
                enable_automatic_punctuation=True,
                model="phone_call" if self._encoding == "MULAW" else "default",
                use_enhanced=False,
            )
            # Let Google do endpointing; keep stream open for multiple utterances
            self._streaming_config = types.StreamingRecognitionConfig(
                config=self._recognition_config,
                interim_results=self._interim_results,
                single_utterance=self._single_utterance,
            )

        def push_audio(self, audio_chunk: bytes) -> None:
            if self._closed:
                return
            self._audio_q.put(audio_chunk)

        def finish(self) -> None:
            if not self._closed:
                self._closed = True
                self._audio_q.put(None)  # sentinel

        async def start(self) -> None:
            if self._task_started:
                return
            self._task_started = True
            import threading
            self._thread = threading.Thread(target=self._run_blocking_stream, daemon=True)
            self._thread.start()

        def _run_blocking_stream(self):
            def request_iter():
                # Audio chunks only; config passed via API 'config' arg (helper signature)
                while True:
                    chunk = self._audio_q.get()
                    if chunk is None:
                        break
                    if not chunk:
                        continue
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)

            try:
                # Some builds route through SpeechHelpers which requires 'config' arg
                responses = self._client.streaming_recognize(
                    config=self._streaming_config,
                    requests=request_iter(),
                )
                for response in responses:
                    # Each response may contain multiple results; only the most recent is of interest
                    if not response.results:
                        continue
                    result = response.results[0]
                    if not result.alternatives:
                        continue
                    alt = result.alternatives[0]
                    payload = {
                        "transcript": alt.transcript or "",
                        "confidence": getattr(alt, "confidence", 0.0) or 0.0,
                        "is_final": bool(result.is_final),
                    }
                    self._results_q.put(payload)
            except Exception as e:
                self._results_q.put({"error": str(e), "transcript": "", "confidence": 0.0, "is_final": True})
            finally:
                # Signal completion
                self._results_q.put({"done": True})

        async def get_result(self) -> Dict[str, Any]:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._results_q.get)

    def create_streaming_session(
        self,
        language_code: str = None,
        encoding: str = None,
        sample_rate: int = None,
        interim_results: bool = True,
        single_utterance: bool = False
    ) -> "GoogleSTTService.StreamingSTTSession":
        if not self.client:
            raise Exception("Google Speech client not initialized")
        return GoogleSTTService.StreamingSTTSession(
            client=self.client,
            config=self,
            language_code=language_code,
            encoding=encoding,
            sample_rate=sample_rate,
            interim_results=interim_results,
            single_utterance=single_utterance,
        )
    
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
    
    async def transcribe_audio_chunk_streaming(
        self,
        audio_content: bytes,
        language_code: str = None,
        encoding: str = None,
        sample_rate: int = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Google Cloud STT
        Works with both MULAW (Media Streams) and WAV (Gather recordings)
        
        Args:
            audio_content: Raw audio bytes (MULAW or WAV format)
            language_code: Language code for transcription
            encoding: Audio encoding (MULAW, LINEAR16) - auto-detected if None
            sample_rate: Sample rate in Hz - auto-detected if None
        
        Returns:
            Dictionary with transcript, confidence, and is_final status
        """
        if not self.client:
            return {"error": "Google Speech client not initialized", "transcript": "", "confidence": 0.0}
        
        try:
            import sys
            language_code = language_code or settings.GOOGLE_STT_LANGUAGE_CODE
            
            # Default model and enhanced flag
            model = "default"
            use_enhanced = False
            
            # Auto-detect encoding and sample rate from audio file
            if encoding is None or sample_rate is None:
                # Check if WAV file (starts with RIFF header)
                if audio_content[:4] == b'RIFF':
                    # WAV file detected
                    encoding = "LINEAR16"
                    # Parse WAV header for sample rate (bytes 24-27)
                    sample_rate = int.from_bytes(audio_content[24:28], byteorder='little')
                    
                    # For 8kHz LINEAR16 recordings, use 'default' or 'command_and_search' model
                    # 'telephony' model is deprecated, 'phone_call' is for MULAW only
                    if sample_rate <= 8000:
                        model = "command_and_search"  # Best for short utterances at 8kHz
                        use_enhanced = False  # Enhanced not available for 8kHz
                    else:
                        model = "default"
                        use_enhanced = True
                    
                    logger.info(f"📊 Auto-detected: WAV file, {sample_rate}Hz, LINEAR16, model={model}, enhanced={use_enhanced}")
                else:
                    # Assume MULAW for Media Streams
                    encoding = settings.GOOGLE_STT_ENCODING
                    sample_rate = settings.GOOGLE_STT_SAMPLE_RATE
                    
                    # phone_call model is for MULAW streaming
                    model = "phone_call"
                    use_enhanced = False  # Enhanced not available for 8kHz MULAW
                    
                    logger.info(f"📊 Using config: MULAW, {sample_rate}Hz, model={model}, enhanced={use_enhanced}")
            else:
                # Manual configuration provided
                if sample_rate <= 8000:
                    model = "command_and_search" if encoding == "LINEAR16" else "phone_call"
                    use_enhanced = False
                else:
                    model = "default"
                    use_enhanced = True
            
            # Map encoding
            encoding_map = {
                "MULAW": speech.RecognitionConfig.AudioEncoding.MULAW,
                "LINEAR16": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            }
            audio_encoding = encoding_map.get(encoding, speech.RecognitionConfig.AudioEncoding.LINEAR16)
            
            config = speech.RecognitionConfig(
                encoding=audio_encoding,
                sample_rate_hertz=sample_rate,  # Use detected or provided sample rate
                language_code=language_code,
                enable_automatic_punctuation=True,
                model=model,  # Dynamic model based on audio format
                use_enhanced=use_enhanced,  # Only for 16kHz+ audio
                # Add common phrases for better recognition
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
            # This is better for short phone call chunks (VAPI-style)
            audio = speech.RecognitionAudio(content=audio_content)
            
            logger.info(f"🎙️ Sending {len(audio_content)} bytes to Google Cloud STT...")
            logger.debug(f"🔧 Config: encoding={encoding}, sample_rate={sample_rate}Hz, model={model}, enhanced={use_enhanced}")
            
            # Run Google STT call in thread pool to avoid blocking
            import concurrent.futures
            loop = asyncio.get_event_loop()
            
            def sync_recognize():
                return self.client.recognize(config=config, audio=audio)
            
            # Simple recognize() call - fast and reliable for short chunks
            response = await loop.run_in_executor(None, sync_recognize)
            
            logger.debug(f"📡 Received response from Google STT")
            
            # Process results
            if response.results:
                result = response.results[0]
                if result.alternatives:
                    alternative = result.alternatives[0]
                    transcript_text = alternative.transcript
                    confidence = alternative.confidence if hasattr(alternative, 'confidence') else 0.9
                    
                    logger.info(f"📝 Transcript: '{transcript_text}' (confidence: {confidence:.2f})")
                    
                    return {
                        "transcript": transcript_text,
                        "confidence": confidence,
                        "is_final": True
                    }
            
            logger.debug(f"⚠️ No transcript in response - empty audio or silence")
            return {"transcript": "", "confidence": 0.0, "is_final": True}
        
        except Exception as e:
            logger.error(f"❌ Error in transcription: {e}", exc_info=True)
            return {"error": str(e), "transcript": "", "confidence": 0.0}


# Global service instance
google_stt_service = GoogleSTTService()

