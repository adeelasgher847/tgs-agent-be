"""
bidirectional_stream_v2.py — Shadow WebSocket Handler for VoiceOrchestrator V2

Feature-flagged parallel route. When ENABLE_VOICE_ORCHESTRATOR_V2=true, this
handler is used instead of bidirectional_stream.py.

Gradual rollout strategy (plan.md):
  Week 1: 5% of calls → V2
  Week 2: 20% → V2
  Week 3: 50% → V2
  Week 4: 100% → V2 (V1 decommissioned)

This file implements the Twilio WebSocket protocol (same as V1) while delegating
all voice logic to VoiceOrchestrator. Keeps WebSocket + call session concerns
separate from audio pipeline.

API: /ws/bidirectional/v2/{callSessionId}/{agentId}
     Same URL shape as V1 — swap via router mount/flag, not API change.
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

logger = logging.getLogger(__name__)

router = APIRouter()

# Email agent prompt regex (mirrors V1 logic for STT extended endpointing)
import re

_EMAIL_AGENT_PROMPT_RE = re.compile(
    r"(?i)(?:"
    r"(?:provide|share|send|give)\s+(?:us\s+)?(?:your\s+)?(?:e-?mail\s+address|e-?mail|email)|"
    r"(?:what(?:'s|\s+is)|may\s+i\s+have|can\s+i\s+(?:have|get))\s+(?:your\s+)?(?:e-?mail|email)(?:\s+address)?|"
    r"(?:your\s+)?(?:e-?mail|email)\s+address(?:,?\s*please)?|"
    r"\bspell\b.*\b(?:e-?mail|email)|\b(?:e-?mail|email)\b.*\bspell\b"
    r")",
)


class BidirectionalStreamV2Handler:
    """
    WebSocket handler wrapping VoiceOrchestrator V2.

    Responsibilities (ONLY):
    - Parse Twilio WebSocket messages (connected, media, stop, mark)
    - Extract MULAW audio frames → pass to orchestrator
    - Forward orchestrator TTS frames → Twilio WebSocket
    - Handle session lifecycle (start/stop/cleanup)
    - Latency telemetry (compare V1 vs V2)

    NOT responsible for:
    - STT, LLM, TTS logic (delegated to orchestrator + managers)
    - Barge-in logic (delegated to BargeInController)
    - Prompt building (delegated to LLMStreamManager)
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

        # User pickup detection (mirrors V1 approach)
        self._user_picked_up: bool = False
        self._first_media_received: bool = False
        self._skip_audio_until: Optional[float] = None
        self._auto_greeting_sent: bool = False
        self._call_ended: bool = False
        self._stop_event: asyncio.Event = asyncio.Event()

        # Build agent config dict for orchestrator
        agent_config = self._build_agent_config()

        # Orchestrator — all voice logic lives here
        self._orchestrator: Optional[VoiceOrchestrator] = None
        self._agent_config = agent_config

        # Telemetry
        self._call_start_ts: float = time.monotonic()
        self._first_tts_latency_ms: Optional[int] = None
        self._total_turns: int = 0

        # Email STT extended endpointing (mirrors V1)
        self._email_stt_upgraded: bool = False

    # ------------------------------------------------------------------
    # Session loading
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Main WebSocket handler
    # ------------------------------------------------------------------

    async def handle(self) -> None:
        """
        Main coroutine: drive the Twilio WebSocket for this call.

        Runs until Twilio sends 'stop' or the orchestrator calls shutdown.
        """
        logger.info(
            f"[V2] Call started: call_session_id={self.call_session_id} "
            f"agent_id={self.agent_id}"
        )

        # Build and start orchestrator
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
        """
        Process incoming Twilio WebSocket messages.

        Twilio message types:
        - connected: Stream initialized
        - start: Call started, stream_sid + call_sid available
        - media: Audio frame (MULAW 8kHz, 20ms)
        - stop: Call ended by Twilio
        - mark: Playback marker (for sync)
        """
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
                pass  # Sync marker — no action needed in V2

    # ------------------------------------------------------------------
    # Twilio event handlers
    # ------------------------------------------------------------------

    async def _handle_connected(self, message: Dict[str, Any]) -> None:
        """WebSocket protocol connected."""
        logger.debug(f"[V2] Twilio connected: {self.call_session_id}")

    async def _handle_start(self, message: Dict[str, Any]) -> None:
        """Call started — extract stream metadata and send greeting."""
        start_data = message.get("start", {})
        self.stream_sid = start_data.get("streamSid", "")
        self.call_sid = start_data.get("callSid", "")

        logger.info(
            f"[V2] Stream started: stream_sid={self.stream_sid} "
            f"call_sid={self.call_sid}"
        )

        # Send greeting immediately (< 200ms target, bypasses LLM)
        if not self._auto_greeting_sent and self._orchestrator:
            self._auto_greeting_sent = True
            await self._orchestrator.play_greeting()

    async def _handle_media(self, message: Dict[str, Any]) -> None:
        """
        Audio frame from Twilio — decode and feed to orchestrator.

        Applies user pickup detection (mirrors V1 logic):
        - Wait for non-silent audio before enabling STT
        - Skip first 3 seconds after pickup (system messages)
        """
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

        # User pickup detection via RMS
        if not self._user_picked_up:
            if self._detect_user_audio(audio_data):
                self._user_picked_up = True
                self._skip_audio_until = time.monotonic() + 3.0
                logger.info(f"[V2] User pickup detected: {self.call_session_id}")
            return

        # Skip audio during grace period (system messages)
        if self._skip_audio_until and time.monotonic() < self._skip_audio_until:
            return

        # Feed to orchestrator (→ STT → LLM → TTS)
        if self._orchestrator and not self._call_ended:
            await self._orchestrator.process_twilio_frame(audio_data)

    async def _handle_stop(self, message: Dict[str, Any]) -> None:
        """Twilio ended the call."""
        logger.info(f"[V2] Twilio stop received: {self.call_session_id}")
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Twilio audio output
    # ------------------------------------------------------------------

    async def _send_twilio_audio_frame(self, mulaw_frame: bytes) -> None:
        """
        Send a 20ms MULAW frame to Twilio via WebSocket.

        Called by VoiceOrchestrator.on_tts_frame_ready().

        Format: Twilio Media Streams JSON message with base64-encoded audio.
        """
        if not self.stream_sid:
            return

        try:
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

    # ------------------------------------------------------------------
    # User pickup detection
    # ------------------------------------------------------------------

    def _detect_user_audio(self, audio_data: bytes) -> bool:
        """
        Detect real user audio vs Twilio system tones / silence.

        Uses simple RMS threshold (mirrors V1 logic).
        Returns True when non-silent audio is detected.
        """
        if not audio_data:
            return False

        min_rms = int(getattr(settings, "VOICE_MIN_AUDIO_RMS_FOR_PICKUP", 40))

        try:
            from app.utils.audio_utils import ulaw_to_linear_sample
            samples = [ulaw_to_linear_sample(b) for b in audio_data]
            rms = int((sum(s * s for s in samples) / len(samples)) ** 0.5)
            return rms >= min_rms
        except Exception:
            # Fallback: simple byte-level check
            avg = sum(audio_data) / len(audio_data)
            return abs(avg - 127.5) > 20  # 127/128 = MULAW silence

    # ------------------------------------------------------------------
    # Email STT endpointing (mirrors V1 behaviour)
    # ------------------------------------------------------------------

    def schedule_email_stt_upgrade(self, agent_text: str) -> None:
        """
        If agent just asked for email, upgrade STT to extended endpointing.

        Call this from on_llm_chunk or on_tts_complete when agent_text
        matches the email-ask pattern.
        """
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

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def _cleanup(self) -> None:
        """Graceful shutdown: stop orchestrator and persist call data."""
        self._call_ended = True

        if self._orchestrator:
            try:
                call_summary = self._orchestrator.get_call_summary()
                await self._orchestrator.shutdown()

                # Log telemetry comparison (V1 vs V2)
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


# ---------------------------------------------------------------------------
# FastAPI route — shadow handler (parallel to V1 route in bidirectional_stream.py)
# ---------------------------------------------------------------------------

@router.websocket("/ws/bidirectional/v2/{call_session_id}/{agent_id}")
async def bidirectional_stream_v2_endpoint(
    websocket: WebSocket,
    call_session_id: str,
    agent_id: str,
) -> None:
    """
    V2 shadow WebSocket endpoint for VoiceOrchestrator.

    Feature-flagged: only active when ENABLE_VOICE_ORCHESTRATOR_V2=true.
    Gradual rollout via call routing logic in bidirectional_stream.py.

    Same API shape as V1: /ws/bidirectional/{call_session_id}/{agent_id}
    """
    if not getattr(settings, "ENABLE_VOICE_ORCHESTRATOR_V2", False):
        logger.warning(
            "[V2] V2 endpoint called but ENABLE_VOICE_ORCHESTRATOR_V2=false. "
            "Check your routing configuration."
        )
        await websocket.close(code=1008, reason="V2 not enabled")
        return

    await websocket.accept()

    from app.db.session import SessionLocal

    db = SessionLocal()
    try:
        handler = BidirectionalStreamV2Handler(
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
