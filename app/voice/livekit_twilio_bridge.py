"""
Twilio Media Stream → LiveKit room bridge (caller audio publish).

Twilio connects to ``/api/v1/livekit/{roomName}``. This module publishes inbound
μ-law frames into the LiveKit room as ``caller-{roomName}`` so agent-side
subscribers (e.g. LiveKitAudioSubscriber + Google STT) receive caller audio.

TTS and conversation logic stay on the same Twilio WebSocket via
BidirectionalStreamHandler — only inbound audio is mirrored to LiveKit.
"""

from __future__ import annotations

import asyncio
import struct
from typing import Any, Optional

from app.core.config import settings
from app.core.logger import logger
from app.services.livekit_service import livekit_service, _http_to_ws_url
from app.utils.audio_utils import MULAW_FRAME_BYTES, ulaw_to_linear_sample

# Twilio μ-law is 8 kHz mono — match LiveKit AudioSource rate.
_TWILIO_SAMPLE_RATE = 8000


def build_livekit_stream_ws_url(room_name: str) -> str:
    """TwiML <Stream url=…> for ticket-compliant LiveKit bridge path."""
    base = settings.WEBHOOK_BASE_URL.rstrip("/")
    ws_protocol = "wss" if base.startswith("https") else "ws"
    host = base.replace("https://", "").replace("http://", "")
    return f"{ws_protocol}://{host}/api/v1/livekit/{room_name}"


class LiveKitTwilioPublisher:
    """Publish Twilio inbound μ-law audio into a LiveKit room as the caller."""

    def __init__(self, room_name: str) -> None:
        self._room_name = room_name
        self._room: Any = None
        self._source: Any = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        if not settings.LIVEKIT_ENABLED:
            return False
        try:
            from livekit import rtc
        except ImportError:
            logger.error("[LiveKitBridge] livekit package not available")
            return False

        try:
            livekit_service._validate_room_name(self._room_name)
            url, _, _ = livekit_service._get_credentials()
            ws_url = _http_to_ws_url(url)
            token = livekit_service.generate_caller_token(self._room_name)

            self._room = rtc.Room()
            await self._room.connect(ws_url, token)

            self._source = rtc.AudioSource(_TWILIO_SAMPLE_RATE, 1)
            track = rtc.LocalAudioTrack.create_audio_track(
                "caller-audio", self._source
            )
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            await self._room.local_participant.publish_track(track, options)

            self._connected = True
            logger.info(
                "[LiveKitBridge] caller track published room=%s", self._room_name
            )
            return True
        except Exception as exc:
            logger.error(
                "[LiveKitBridge] connect failed room=%s: %s",
                self._room_name,
                exc,
                exc_info=True,
            )
            await self.disconnect()
            return False

    async def publish_mulaw(self, mulaw_bytes: bytes) -> None:
        """Push one or more Twilio μ-law bytes into the LiveKit caller track."""
        if not self._connected or not self._source or not mulaw_bytes:
            return

        try:
            from livekit import rtc
        except ImportError:
            return

        try:
            offset = 0
            total = len(mulaw_bytes)
            while offset < total:
                chunk = mulaw_bytes[offset : offset + MULAW_FRAME_BYTES]
                offset += MULAW_FRAME_BYTES
                if len(chunk) < MULAW_FRAME_BYTES:
                    chunk = chunk + bytes([0xFF]) * (MULAW_FRAME_BYTES - len(chunk))

                samples = [ulaw_to_linear_sample(b) for b in chunk]
                pcm = struct.pack(f"<{len(samples)}h", *samples)
                frame = rtc.AudioFrame.create(
                    _TWILIO_SAMPLE_RATE, 1, len(samples)
                )
                frame.data[:] = pcm
                await self._source.capture_frame(frame)
        except Exception as exc:
            logger.debug("[LiveKitBridge] publish_mulaw: %s", exc)

    async def disconnect(self) -> None:
        self._connected = False
        if self._room is not None:
            try:
                await self._room.disconnect()
            except Exception:
                pass
        self._room = None
        self._source = None
        logger.info("[LiveKitBridge] disconnected room=%s", self._room_name)
