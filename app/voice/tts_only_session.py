import json
import uuid
from typing import Optional

from fastapi import WebSocket

from app.core.logger import logger
from app.services.call_session_service import call_session_service
from app.services.agent_service import agent_service
from app.services.bidirectional_stream_service import generate_mulaw_tts
from app.utils.audio_utils import stream_mulaw_bytes_over_twilio, apply_micro_fade_in


class TtsOnlySession:
    """
    Handles the /ws/tts-only WebSocket:
    - No STT; receives text via custom events and streams TTS audio.
    - Can auto-play pending TTS stored on the call session.
    """

    def __init__(
        self,
        websocket: WebSocket,
        call_session_id: str,
        agent_id: str,
        db,
    ):
        self.websocket = websocket
        self.call_session_id = call_session_id
        self.agent_id = agent_id
        self.db = db

        self.call_session = None
        self.agent = None
        self.stream_sid: Optional[str] = None

    def _load_session_data(self) -> None:
        try:
            session_uuid = uuid.UUID(self.call_session_id)
            self.call_session = call_session_service.get_call_session_by_id(self.db, session_uuid)

            if self.call_session and self.agent_id:
                agent_uuid = uuid.UUID(self.agent_id)
                self.agent = agent_service.get_agent_by_id(
                    self.db,
                    agent_uuid,
                    self.call_session.tenant_id,
                )
        except Exception as e:
            logger.error(f"Error loading session data for tts-only: {e}")

    async def _play_tts_text(self, text: str, lang: Optional[str], voice: Optional[str]) -> None:
        if not text or not self.stream_sid:
            return

        lang = lang or (self.agent.language if self.agent and self.agent.language else "en")
        voice = voice or (self.agent.voice_type if self.agent and self.agent.voice_type else "female")

        audio_bytes = await generate_mulaw_tts(
            text=text,
            lang=lang,
            voice=voice,
            use_chirp3_hd=True,
            speaking_rate=1.0,
            add_office_bg=True,
        )

        audio_bytes = apply_micro_fade_in(audio_bytes, duration_ms=25.0)

        await stream_mulaw_bytes_over_twilio(
            websocket=self.websocket,
            stream_sid=self.stream_sid,
            audio_bytes=audio_bytes,
            pace_20ms=True,
            prime_frames=3,
        )

    async def run(self) -> None:
        """
        Main receive loop for the TTS-only WebSocket.
        Mirrors the previous inline implementation in bidirectional_stream.py.
        """
        self._load_session_data()

        try:
            while True:
                data = await self.websocket.receive_text()
                message = json.loads(data)

                event = message.get("event")

                if event == "connected":
                    continue

                if event == "start":
                    self.stream_sid = message.get("streamSid")

                    # Auto-retrieve and play pending TTS from call session metadata
                    if self.call_session and getattr(self.call_session, "call_metadata", None):
                        pending_tts = self.call_session.call_metadata.get("pending_tts")
                        if pending_tts:
                            await self._play_tts_text(
                                text=pending_tts.get("text", ""),
                                lang=pending_tts.get("lang"),
                                voice=pending_tts.get("voice"),
                            )
                            self.call_session.call_metadata.pop("pending_tts", None)
                            self.db.commit()

                elif event == "play_tts":
                    text = message.get("text", "")
                    lang = message.get("lang")
                    voice = message.get("voice")
                    await self._play_tts_text(text=text, lang=lang, voice=voice)

                elif event == "stop":
                    break

        except Exception as e:
            logger.error(f"Unexpected error in TTS-only WebSocket: {e}", exc_info=True)

