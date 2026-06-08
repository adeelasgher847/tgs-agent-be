"""
LiveKit ↔ Twilio Media Stream bridge WebSocket.

Ticket TwiML::
    <Connect><Stream url="wss://{host}/api/v1/livekit/{roomName}"/></Connect>

Twilio audio is published into the pre-provisioned LiveKit room; the same
WebSocket also drives BidirectionalStreamHandler (STT/TTS/LLM).
"""

from __future__ import annotations

import base64
import json
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logger import logger
from app.db.session import SessionLocal
from app.routers.bidirectional_stream import (
    BidirectionalStreamHandler,
    _receive_or_stop,
)
from app.services.call_session_service import call_session_service
from app.services.livekit_service import livekit_service
from app.voice.livekit_twilio_bridge import LiveKitTwilioPublisher

router = APIRouter()


def _call_session_id_from_room(room_name: str) -> uuid.UUID | None:
    try:
        livekit_service._validate_room_name(room_name)
        return uuid.UUID(room_name.removeprefix("room_"))
    except (ValueError, AttributeError):
        return None


@router.websocket("/{room_name}")
async def livekit_twilio_bridge_websocket(
    websocket: WebSocket,
    room_name: str,
) -> None:
    """
    Twilio Media Stream endpoint — bridges caller audio into LiveKit and runs
    the voice agent on the same socket.
    """
    session_id = _call_session_id_from_room(room_name)
    if session_id is None:
        logger.warning("[LiveKitBridge] invalid room name: %s", room_name)
        await websocket.close(code=1008, reason="Invalid room name")
        return

    try:
        await websocket.accept()
    except Exception as exc:
        logger.error("[LiveKitBridge] accept failed: %s", exc)
        return

    db = SessionLocal()
    call_session = call_session_service.get_call_session_by_id(db, session_id)
    if not call_session:
        logger.warning("[LiveKitBridge] no call session for room=%s", room_name)
        await websocket.close(code=1008, reason="Call session not found")
        db.close()
        return

    agent_id = str(call_session.agent_id)
    call_session_id = str(call_session.id)

    lk_publisher = LiveKitTwilioPublisher(room_name)
    lk_ok = await lk_publisher.connect()

    handler = BidirectionalStreamHandler(
        websocket=websocket,
        call_session_id=call_session_id,
        agent_id=agent_id,
        db=db,
    )

    logger.info(
        "[LiveKitBridge] session=%s room=%s livekit_publish=%s",
        call_session_id,
        room_name,
        lk_ok,
    )

    try:
        while True:
            raw = await _receive_or_stop(websocket, handler._stop_event)
            if raw is None:
                logger.info(
                    "[LiveKitBridge] internal stop session=%s", call_session_id
                )
                break

            message = json.loads(raw)
            event = message.get("event")

            if event == "start":
                await handler.handle_start_message(message)
            elif event == "media":
                if lk_ok:
                    payload = message.get("media", {}).get("payload")
                    if payload:
                        try:
                            mulaw = base64.b64decode(payload)
                            await lk_publisher.publish_mulaw(mulaw)
                        except Exception as exc:
                            logger.debug(
                                "[LiveKitBridge] media decode/publish: %s", exc
                            )
                await handler.handle_media_message(message)
            elif event == "stop":
                await handler.handle_stop_message(message)
                break
            elif event in ("connected", "mark"):
                pass

    except WebSocketDisconnect:
        logger.info("[LiveKitBridge] disconnected session=%s", call_session_id)
    except Exception as exc:
        logger.error(
            "[LiveKitBridge] error session=%s: %s",
            call_session_id,
            exc,
            exc_info=True,
        )
    finally:
        try:
            await handler._full_shutdown()
        except Exception as exc:
            logger.debug("[LiveKitBridge] shutdown: %s", exc)
        await lk_publisher.disconnect()
        try:
            await websocket.close()
        except Exception:
            pass
        db.close()
