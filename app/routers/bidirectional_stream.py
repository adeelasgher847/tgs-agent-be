"""
bidirectional_stream.py — WebSocket Handler for VoiceOrchestrator V2

This handler delegates all voice logic (STT, LLM, TTS, Barge-in) to VoiceOrchestrator.
Keeps WebSocket + call session concerns separate from the audio pipeline.
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.agent_service import agent_service
from app.services.call_session_service import call_session_service
from app.services.voice_logging_service import VoiceLoggingService
from app.voice.orchestrator import VoiceOrchestrator
from app.voice.stt_stream_manager import EndpointingMode
from app.voice.background_audio import BackgroundAudioManager
from app.services.transcript_service import transcript_service

# Re-export TWiML builders for backwards compatibility with voice.py endpoints
from app.services.bidirectional_stream_service import (
    build_streaming_twiml,
    build_tts_only_twiml,
)

logger = logging.getLogger(__name__)

router = APIRouter()

import re

_EMAIL_AGENT_PROMPT_RE = re.compile(
    r"(?i)(?:"
    r"(?:provide|share|send|give)\s+(?:us\s+)?(?:your\s+)?(?:e-?mail\s+address|e-?mail|email)|"
    r"(?:what(?:'s|\s+is)|may\s+i\s+have|can\s+i\s+(?:have|get))\s+(?:your\s+)?(?:e-?mail|email)(?:\s+address)?|"
    r"(?:your\s+)?(?:e-?mail|email)\s+address(?:,?\s*please)?|"
    r"\bspell\b.*\b(?:e-?mail|email)|\b(?:e-?mail|email)\b.*\bspell\b"
    r")",
)


class BidirectionalStreamHandler:
    """
    WebSocket handler wrapping VoiceOrchestrator.

    Responsibilities (ONLY):
    - Parse Twilio WebSocket messages (connected, media, stop, mark)
    - Extract MULAW audio frames → pass to orchestrator
    - Forward orchestrator TTS frames → Twilio WebSocket
    - Handle session lifecycle (start/stop/cleanup)
    - Mix ambient background audio based on agent profile
    """

    def __init__(
        self,
        websocket: WebSocket,
        call_session_id: str,
        agent_id: str,
        db: Session,
    ) -> None:
        self.websocket = websocket
        self.call_session_id = call_session_id
        self.agent_id = agent_id
        self.db = db

        # Twilio stream metadata
        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None

        # Session data
        self.call_session = None
        self.agent = None
        self._load_session_data()

        # User pickup detection
        self._user_picked_up: bool = False
        self._first_media_received: bool = False
        self._skip_audio_until: Optional[float] = None
        self._auto_greeting_sent: bool = False
        self._call_ended: bool = False
        self._stop_event: asyncio.Event = asyncio.Event()

        # Background audio manager (dev-branch style embedded ambience loop).
        self._background_audio = BackgroundAudioManager(
            agent_id=self.agent_id,
            tenant_id=self.call_session.tenant_id if self.call_session else None,
            db=self.db,
        )
        asyncio.create_task(self._background_audio.load_from_base64_async())

        # Build agent config dict for orchestrator
        agent_config = self._build_agent_config()

        # Orchestrator — all voice logic lives here
        self._orchestrator: Optional[VoiceOrchestrator] = None
        self._agent_config = agent_config

        # Telemetry
        self._call_start_ts: float = time.monotonic()
        self._first_tts_latency_ms: Optional[int] = None
        self._total_turns: int = 0

        self._email_stt_upgraded: bool = False

    def _load_session_data(self) -> None:
        """Load call session and agent from DB."""
        try:
            session_uuid = uuid.UUID(self.call_session_id)
            self.call_session = call_session_service.get_call_session_by_id(
                self.db, session_uuid
            )
            if self.call_session and self.agent_id:
                agent_uuid = uuid.UUID(self.agent_id)
                self.agent = agent_service.get_agent_by_id(
                    self.db, agent_uuid, self.call_session.tenant_id
                )
                agent_service.ensure_agent_prompt_ingested(self.db, self.agent)
        except Exception as e:
            logger.error(f"[V2] Failed to load session data: {e}", exc_info=True)

    def _build_agent_config(self) -> Dict[str, Any]:
        """Build the agent_config dict passed to orchestrator managers."""
        return {
            "agent": self.agent,
            "call_session": self.call_session,
        }

    async def handle(self) -> None:
        """
        Main coroutine: drive the Twilio WebSocket for this call.
        """
        logger.info(
            f"[V2] Call started: call_session_id={self.call_session_id} "
            f"agent_id={self.agent_id}"
        )

        self._orchestrator = VoiceOrchestrator(
            call_id=self.call_session_id,
            agent_id=self.agent_id,
            agent_config=self._agent_config,
            send_twilio_frame_callback=self._send_twilio_audio_frame,
        )
        await self._orchestrator.start_call()

        try:
            await self._message_loop()
        except WebSocketDisconnect:
            logger.info(f"[V2] WebSocket disconnected: {self.call_session_id}")
        except Exception as e:
            logger.error(f"[V2] Unexpected error: {e}", exc_info=True)
        finally:
            await self._cleanup()

    async def _message_loop(self) -> None:
        async for raw_message in self.websocket.iter_text():
            if self._stop_event.is_set():
                break

            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                continue

            event_type = message.get("event", "")

            if event_type == "connected":
                await self._handle_connected(message)
            elif event_type == "start":
                await self._handle_start(message)
            elif event_type == "media":
                await self._handle_media(message)
            elif event_type == "stop":
                await self._handle_stop(message)
                break
            elif event_type == "mark":
                pass

    async def _handle_connected(self, message: Dict[str, Any]) -> None:
        logger.debug(f"[V2] Twilio connected: {self.call_session_id}")

    async def _handle_start(self, message: Dict[str, Any]) -> None:
        start_data = message.get("start", {})
        self.stream_sid = start_data.get("streamSid", "")
        self.call_sid = start_data.get("callSid", "")

        logger.info(
            f"[V2] Stream started: stream_sid={self.stream_sid} "
            f"call_sid={self.call_sid}"
        )

        if not self._auto_greeting_sent and self._orchestrator:
            self._auto_greeting_sent = True
            await self._orchestrator.play_greeting()

    async def _handle_media(self, message: Dict[str, Any]) -> None:
        media = message.get("media", {})
        payload = media.get("payload")
        if not payload:
            return

        try:
            audio_data = base64.b64decode(payload)
        except Exception:
            return

        if not self._first_media_received:
            self._first_media_received = True

        if not self._user_picked_up:
            if self._detect_user_audio(audio_data):
                self._user_picked_up = True
                self._skip_audio_until = time.monotonic() + 3.0
                logger.info(f"[V2] User pickup detected: {self.call_session_id}")
                asyncio.create_task(self._start_background_audio_with_delay())
            return

        if self._skip_audio_until and time.monotonic() < self._skip_audio_until:
            return

        if self._orchestrator and not self._call_ended:
            await self._orchestrator.process_twilio_frame(audio_data)

    async def _handle_stop(self, message: Dict[str, Any]) -> None:
        logger.info(f"[V2] Twilio stop received: {self.call_session_id}")
        self._stop_event.set()

    async def _send_twilio_audio_frame(self, mulaw_frame: bytes) -> None:
        if not self.stream_sid:
            return

        try:
            if self._is_background_audio_enabled():
                self._background_audio.set_user_level(self._resolve_background_volume())
                mulaw_frame = self._background_audio.mix_tts_frame(mulaw_frame)

            payload = base64.b64encode(mulaw_frame).decode("ascii")
            await self.websocket.send_json({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {
                    "payload": payload,
                },
            })
        except Exception as e:
            logger.debug(f"[V2] Frame send error (websocket may be closing): {e}")

    def _detect_user_audio(self, audio_data: bytes) -> bool:
        if not audio_data:
            return False

        min_rms = int(getattr(settings, "VOICE_MIN_AUDIO_RMS_FOR_PICKUP", 40))

        try:
            from app.utils.audio_utils import ulaw_to_linear_sample
            samples = [ulaw_to_linear_sample(b) for b in audio_data]
            rms = int((sum(s * s for s in samples) / len(samples)) ** 0.5)
            return rms >= min_rms
        except Exception:
            avg = sum(audio_data) / len(audio_data)
            return abs(avg - 127.5) > 20

    async def _start_background_audio_with_delay(self):
        try:
            if not self._is_background_audio_enabled():
                return
            self._background_audio.set_user_level(self._resolve_background_volume())
            await self._background_audio.start_loop_if_enabled(delay_seconds=3.0)
        except Exception as e:
            logger.error(f"Error in _start_background_audio_with_delay: {e}", exc_info=True)

    def _is_background_audio_enabled(self) -> bool:
        if not self.agent or not self.agent.tts_settings_json:
            return True
        settings_json = self.agent.tts_settings_json
        if isinstance(settings_json, str):
            try:
                settings_json = json.loads(settings_json)
            except Exception:
                settings_json = {}
        enabled_raw = settings_json.get("background_enabled", True)
        if str(enabled_raw).strip().lower() in ("false", "0"):
            return False
        profile = str(settings_json.get("background_profile") or "office").strip().lower()
        return profile == "office"

    def _resolve_background_volume(self) -> float:
        if not self.agent or not self.agent.tts_settings_json:
            return 50.0
        settings_json = self.agent.tts_settings_json
        if isinstance(settings_json, str):
            try:
                settings_json = json.loads(settings_json)
            except Exception:
                settings_json = {}
        raw = settings_json.get("background_volume", 50)
        try:
            return float(raw)
        except Exception:
            return 50.0

    def schedule_email_stt_upgrade(self, agent_text: str) -> None:
        if self._email_stt_upgraded or not self._orchestrator:
            return
        if not _EMAIL_AGENT_PROMPT_RE.search(agent_text or ""):
            return

        async def _deferred():
            try:
                await asyncio.sleep(0)
                if not self._email_stt_upgraded and self._orchestrator:
                    self._orchestrator.set_endpointing_mode(EndpointingMode.EXTENDED)
                    self._email_stt_upgraded = True
                    logger.info(
                        f"[V2] STT upgraded to extended endpointing (email collection)"
                    )
            except Exception as e:
                logger.debug(f"[V2] Email STT upgrade error: {e}")

        asyncio.create_task(_deferred())

    async def _cleanup(self) -> None:
        self._call_ended = True

        if self._orchestrator:
            try:
                call_summary = self._orchestrator.get_call_summary()
                await self._orchestrator.shutdown()

                try:
                    await self._background_audio.stop_loop()
                except Exception:
                    pass

                # Persist to database
                if self.call_session:
                    try:
                        conversation = transcript_service.get_conversation_array(self.db, self.call_session.id)
                        self.call_session.call_transcript = conversation
                        
                        ended_reason = "Call completed normally" if call_summary.get("state") == "completed" else "Call ended abruptly"
                        call_session_service.update_call_session_status(
                            db=self.db,
                            call_session_id=self.call_session.id,
                            status="completed",
                            ended_reason=ended_reason
                        )
                    except Exception as e:
                        logger.error(f"[V2] Database persistence error: {e}", exc_info=True)

                elapsed_ms = int((time.monotonic() - self._call_start_ts) * 1000)
                logger.info(
                    f"[V2] Call ended: {self.call_session_id} "
                    f"duration={elapsed_ms}ms "
                    f"turns={self._total_turns} "
                    f"state={call_summary.get('state')} "
                    f"messages={call_summary.get('message_count', 0)}"
                )
            except Exception as e:
                logger.error(f"[V2] Cleanup error: {e}", exc_info=True)


@router.websocket("/ws/bidirectional/{call_session_id}/{agent_id}")
async def bidirectional_stream_endpoint(
    websocket: WebSocket,
    call_session_id: str,
    agent_id: str,
) -> None:
    await websocket.accept()

    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        handler = BidirectionalStreamHandler(
            websocket=websocket,
            call_session_id=call_session_id,
            agent_id=agent_id,
            db=db,
        )
        await handler.handle()
    except Exception as e:
        logger.error(f"[V2] Endpoint error: {e}", exc_info=True)
    finally:
        db.close()