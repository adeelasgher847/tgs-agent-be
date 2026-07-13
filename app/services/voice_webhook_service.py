from datetime import datetime, timezone
import uuid
from typing import Optional

import asyncio
from fastapi import Request, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.core.logger import logger
from app.models.call_session import CallSession
from app.services.agent_service import agent_service
from app.services.voice_screening_qualification_service import maybe_qualify_resume_on_call_completed
from app.services.call_session_service import call_session_service
from app.services.credit_service import credit_service
from app.utils.twilio_validation import (
    validate_twilio_signature,
    validate_webrtc_auth,
)
from app.routers.general_websocket import (
    broadcast_system_notification,
    broadcast_call_status_update,
    broadcast_call_ended,
)
from app.core.config import settings
from app.services.voice_conversation_service import add_to_transcript
from app.services.voice_language_service import get_agent_voice
from app.utils.voice_twilio_utils import get_twilio_credentials_for_call
from app.services.voice_logging_service import VoiceLoggingService
from app.utils.response import create_success_response
from app.utils.twilio_validation import get_request_body
from app.services.twilio_service import twilio_service
from fastapi.responses import StreamingResponse
import requests
from app.routers.bidirectional_stream import build_streaming_twiml
from app.services.voice_phrase_service import get_random_didnt_catch_response
from urllib.parse import quote
from twilio.twiml.voice_response import VoiceResponse


async def handle_call_events_webhook(
    request: Request,
    agentId: Optional[str],
    userId: Optional[str],
    callSessionId: Optional[str],
    timeout: Optional[str],
    body: str,
    db: Session,
) -> HTMLResponse:
    """
    Extracted logic from `voice.handle_call_events_webhook`, behavior preserved.
    """
    logger.info("🔥🔥🔥 WEBHOOK CALLED! 🔥🔥🔥")
    logger.info("=== Call Events Webhook Started ===")
    logger.info(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    from app.core.pii_redactor import prepare_request_log_context

    logger.info(
        "Call events webhook started %s",
        prepare_request_log_context(
            request.method,
            request.url.path,
            request.headers,
            query_params={
                "agentId": agentId or "",
                "userId": userId or "",
                "callSessionId": callSessionId or "",
            },
            body_length=len(body) if body else 0,
        ),
    )

    try:
        try:
            asyncio.create_task(
                broadcast_system_notification(
                    notification_type="webhook_started",
                    message=f"Webhook started for call session {callSessionId}",
                    metadata={
                        "agent_id": agentId,
                        "user_id": userId,
                        "call_session_id": callSessionId,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            )
            logger.info("✅ WebSocket broadcast queued at webhook start")
        except Exception as e:  # pragma: no cover
            logger.warning("⚠️ WebSocket broadcast failed (non-critical): %s", e)

        logger.debug("Parsing request body...")

        form_data = await request.form()
        call_sid = form_data.get("CallSid", "")
        call_status = form_data.get("CallStatus", "")
        from_number = form_data.get("From", "")
        to_number = form_data.get("To", "")
        direction = form_data.get("Direction", "")

        logger.info(
            "🎤 Speech handling is now managed by Deepgram STT WebSocket"
        )

        call_session = None
        agent = None

        if callSessionId:
            try:
                session_uuid = uuid.UUID(callSessionId)
                call_session = call_session_service.get_call_session_by_id(
                    db, session_uuid
                )
                if call_session:
                    logger.info(
                        "✅ Found call session: %s from query parameter",
                        call_session.id,
                    )

                    if agentId:
                        agent = agent_service.get_agent_by_id(
                            db, uuid.UUID(agentId), call_session.tenant_id
                        )
                        if agent:
                            logger.info(
                                "✅ Agent fetched: %s (ID: %s)",
                                agent.name,
                                agent.id,
                            )
                            logger.info("🏢 Tenant: %s", agent.tenant_id)
                        else:
                            logger.warning(
                                "⚠️ Agent %s not found in tenant %s",
                                agentId,
                                call_session.tenant_id,
                            )
                else:
                    logger.warning(
                        "⚠️ No call session found for ID: %s", callSessionId
                    )
            except ValueError:
                logger.warning("⚠️ Invalid call session ID format: %s", callSessionId)
        else:
            logger.info("⚠️ No callSessionId provided in query parameters")

        is_twilio = "X-Twilio-Signature" in request.headers
        is_webrtc = "Authorization" in request.headers

        if is_twilio:
            logger.info(
                "Twilio signature found, but skipping validation for testing"
            )
        elif is_webrtc:
            if not validate_webrtc_auth(request):
                raise HTTPException(
                    status_code=403, detail="Invalid WebRTC authentication"
                )
        else:
            logger.info("No authentication headers found, allowing for testing")

        logger.info(
            "Call Events Webhook - SID: %s, Status: %s, From: %s, To: %s, Direction: %s",
            call_sid,
            call_status,
            from_number,
            to_number,
            direction,
        )
        logger.info("AgentId from query: %s", agentId)

        logger.debug("=" * 60)
        logger.debug("🔍 DEBUG WEBHOOK RECEIVED:")
        logger.debug("   Status: '%s'", call_status)
        logger.debug("   Direction: '%s'", direction)
        logger.debug("   Call SID: %s", call_sid)
        if call_session:
            logger.debug("   Current DB Status: '%s'", call_session.status)
            logger.debug("   Call Session ID: %s", call_session.id)
        else:
            logger.debug("   Call Session: Not found")
        logger.debug("=" * 60)

        if call_session and call_status and call_status not in [
            "answered",
            "in-progress",
        ]:
            logger.info(
                "🔄 Updating call session %s status to: %s",
                call_session.id,
                call_status,
            )
            call_session.status = call_status
        elif call_session and call_status in ["answered", "in-progress"]:
            logger.debug(
                "🔍 DEBUG: Skipping automatic status update for '%s' - will be set when media streaming starts",
                call_status,
            )

        if call_session and call_status == "completed":
            call_session.end_time = datetime.now(timezone.utc)
            if call_session.start_time:
                duration = (
                    call_session.end_time - call_session.start_time
                ).total_seconds()
                call_session.duration = int(duration)
                logger.info(
                    "⏰ Set end time and duration (%ss) for session %s",
                    duration,
                    call_session.id,
                )

                try:
                    asyncio.create_task(
                        broadcast_call_ended(
                            call_session_id=str(call_session.id),
                            reason="completed",
                            final_data={
                                "call_sid": call_sid,
                                "from_number": from_number,
                                "to_number": to_number,
                                "direction": direction,
                                "duration": call_session.duration,
                                "timestamp": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            },
                        )
                    )
                    logger.info(
                        "✅ Queued call ended event for session %s", call_session.id
                    )
                except Exception as e:  # pragma: no cover
                    logger.warning(
                        "⚠️ Failed to queue call ended event (non-critical): %s", e
                    )

                try:
                    credit_service.stop_credit_monitoring(call_session.id)
                    logger.info(
                        "✅ Stopped credit monitoring for call session %s",
                        call_session.id,
                    )
                except Exception as e:  # pragma: no cover
                    logger.warning(
                        "⚠️ Failed to stop credit monitoring (non-critical): %s", e
                    )

            call_session_service.update_call_session_status(
                db, call_session.id, "completed", ended_reason="hung up"
            )

            try:
                maybe_qualify_resume_on_call_completed(db, call_session.id)
            except Exception as mq_exc:  # pragma: no cover
                logger.warning(
                    "Resume screening qualify on call completed (fallback): %s",
                    mq_exc,
                    exc_info=True,
                )

            logger.info(
                "✅ Updated call session %s status to: %s with ended_reason: hung up",
                call_session.id,
                call_status,
            )

            if call_status == "in-progress":
                logger.info(
                    "ℹ️ Skipping 'in-progress' broadcast here - will be sent by media stream handler"
                )
            else:
                try:
                    logger.info(
                        "🚀 Broadcasting call status update: %s for session %s",
                        call_status,
                        call_session.id,
                    )

                    metadata = {
                        "from_number": from_number,
                        "to_number": to_number,
                        "direction": direction,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "start_time": call_session.start_time.isoformat()
                        if call_session.start_time
                        else None,
                        "end_time": call_session.end_time.isoformat()
                        if call_session.end_time
                        else None,
                        "duration": call_session.duration,
                    }

                    if call_status == "ringing":
                        metadata["message"] = "Call is ringing"
                    elif call_status == "completed":
                        metadata["message"] = "Call has been completed"

                    await broadcast_call_status_update(
                        call_session_id=str(call_session.id),
                        status=call_status,
                        metadata=metadata,
                    )
                    logger.debug(
                        "✅ Call status update sent: %s for session %s",
                        call_status,
                        call_session.id,
                    )

                    if call_status == "completed":
                        asyncio.create_task(
                            broadcast_call_ended(
                                call_session_id=str(call_session.id),
                                reason="Call completed",
                                final_data={
                                    "call_sid": call_sid,
                                    "duration": call_session.duration,
                                    "end_time": call_session.end_time.isoformat(),
                                    "transcript": call_session.call_transcript
                                    or [],
                                },
                            )
                        )
                        logger.debug(
                            "✅ Queued call ended event for session %s",
                            call_session.id,
                        )

                except Exception as e:  # pragma: no cover
                    logger.error(
                        "❌ Failed to broadcast call status update: %s", e, exc_info=True
                    )
        else:
            if not call_session:
                logger.warning(
                    "⚠️ No call session found - cannot update status or broadcast"
                )
            if not call_status:
                logger.warning(
                    "⚠️ No call status provided - cannot update status or broadcast"
                )

        logger.info(
            "Processing call status: '%s' with direction: '%s'",
            call_status,
            direction,
        )

        if call_status == "initiated" and direction == "outbound-api":
            logger.info("Call initiated - SID: %s", call_sid)

            if call_session:
                try:
                    asyncio.create_task(
                        broadcast_call_status_update(
                            call_session_id=str(call_session.id),
                            status="initiated",
                            metadata={
                                "check": "just checking",
                                "call_sid": call_sid,
                                "direction": direction,
                                "timestamp": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            },
                        )
                    )
                    logger.debug(
                        "✅ Broadcasted call initiated event for session %s",
                        call_session.id,
                    )
                except Exception as e:  # pragma: no cover
                    logger.error(
                        "❌ Failed to broadcast call initiated event: %s", e
                    )

            return HTMLResponse("", media_type="application/xml")

        elif call_status == "ringing" and direction == "outbound-api":
            logger.info("🔔 CALL IS RINGING - SID: %s", call_sid)

            if call_session:
                try:
                    asyncio.create_task(
                        broadcast_call_status_update(
                            call_session_id=str(call_session.id),
                            status="ringing",
                            metadata={
                                "call_sid": call_sid,
                                "direction": direction,
                                "timestamp": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            },
                        )
                    )
                    logger.debug(
                        "✅ Broadcasted call ringing event for session %s",
                        call_session.id,
                    )
                except Exception as e:  # pragma: no cover
                    logger.error(
                        "❌ Failed to broadcast call ringing event: %s", e
                    )

            return HTMLResponse("", media_type="application/xml")

        elif call_status == "answered" and direction == "outbound-api":
            logger.info(
                "ℹ️ ANSWERED STATUS RECEIVED (ignored - using first media packet instead)"
            )
            logger.debug(
                "🔍 DEBUG: Will wait for first media packet from WebSocket stream"
            )
            logger.debug(
                "🔍 DEBUG: User pickup detection happens in bidirectional_stream.py"
            )

            return HTMLResponse("", media_type="application/xml")

        elif call_status == "in-progress":
            logger.info(
                "ℹ️ IN-PROGRESS STATUS RECEIVED (ignored - using first media packet instead)"
            )
            logger.debug(
                "🔍 DEBUG: Media stream status from Twilio (not user pickup)"
            )
            logger.debug(
                "🔍 DEBUG: User pickup detection happens in bidirectional_stream.py"
            )

            return HTMLResponse("", media_type="application/xml")
        elif call_status == "completed":
            logger.info("📞 CALL COMPLETED - SID: %s", call_sid)
            return HTMLResponse("", media_type="application/xml")

        elif call_status == "failed":
            logger.error("Call failed - SID: %s", call_sid)

            if call_session:
                try:
                    asyncio.create_task(
                        broadcast_call_status_update(
                            call_session_id=str(call_session.id),
                            status="failed",
                            metadata={
                                "call_sid": call_sid,
                                "direction": direction,
                                "timestamp": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            },
                        )
                    )
                    logger.debug(
                        "✅ Queued call failed event for session %s", call_session.id
                    )

                    asyncio.create_task(
                        broadcast_call_ended(
                            call_session_id=str(call_session.id),
                            reason="failed",
                            final_data={
                                "call_sid": call_sid,
                                "direction": direction,
                                "duration": 0,
                                "timestamp": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            },
                        )
                    )
                except Exception as e:  # pragma: no cover
                    logger.error("❌ Failed to broadcast call failed event: %s", e)

                try:
                    credit_service.stop_credit_monitoring(call_session.id)
                    logger.debug(
                        "✅ Stopped credit monitoring for failed call session %s",
                        call_session.id,
                    )
                except Exception as e:  # pragma: no cover
                    logger.warning(
                        "⚠️ Failed to stop credit monitoring (non-critical): %s", e
                    )

            return HTMLResponse("", media_type="application/xml")

        elif call_status == "busy":
            logger.info("Call busy - SID: %s", call_sid)

            if call_session:
                try:
                    asyncio.create_task(
                        broadcast_call_status_update(
                            call_session_id=str(call_session.id),
                            status="busy",
                            metadata={
                                "call_sid": call_sid,
                                "direction": direction,
                                "timestamp": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            },
                        )
                    )
                    logger.debug(
                        "✅ Queued call busy event for session %s", call_session.id
                    )

                    asyncio.create_task(
                        broadcast_call_ended(
                            call_session_id=str(call_session.id),
                            reason="busy",
                            final_data={
                                "call_sid": call_sid,
                                "direction": direction,
                                "duration": 0,
                                "timestamp": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            },
                        )
                    )
                    logger.debug(
                        "✅ Queued call ended (busy) event for session %s",
                        call_session.id,
                    )
                except Exception as e:  # pragma: no cover
                    logger.error(
                        "❌ Failed to broadcast call busy event: %s", e
                    )

                try:
                    credit_service.stop_credit_monitoring(call_session.id)
                    logger.debug(
                        "✅ Stopped credit monitoring for busy call session %s",
                        call_session.id,
                    )
                except Exception as e:  # pragma: no cover
                    logger.warning(
                        "⚠️ Failed to stop credit monitoring (non-critical): %s", e
                    )

            return HTMLResponse("", media_type="application/xml")

        elif call_status == "no-answer":
            logger.info("Call no-answer - SID: %s", call_sid)

            if call_session:
                try:
                    asyncio.create_task(
                        broadcast_call_status_update(
                            call_session_id=str(call_session.id),
                            status="no-answer",
                            metadata={
                                "call_sid": call_sid,
                                "direction": direction,
                                "timestamp": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            },
                        )
                    )
                    logger.debug(
                        "✅ Queued call no-answer event for session %s",
                        call_session.id,
                    )

                    asyncio.create_task(
                        broadcast_call_ended(
                            call_session_id=str(call_session.id),
                            reason="no-answer",
                            final_data={
                                "call_sid": call_sid,
                                "direction": direction,
                                "duration": 0,
                                "timestamp": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                            },
                        )
                    )
                    logger.debug(
                        "✅ Queued call ended (no-answer) event for session %s",
                        call_session.id,
                    )
                except Exception as e:  # pragma: no cover
                    logger.error(
                        "❌ Failed to broadcast call no-answer event: %s", e
                    )

                try:
                    credit_service.stop_credit_monitoring(call_session.id)
                    logger.debug(
                        "✅ Stopped credit monitoring for no-answer call session %s",
                        call_session.id,
                    )
                except Exception as e:  # pragma: no cover
                    logger.warning(
                        "⚠️ Failed to stop credit monitoring (non-critical): %s", e
                    )

            return HTMLResponse("", media_type="application/xml")

        else:
            logger.info(
                "Unhandled call status: '%s' - using default response", call_status
            )
            response = VoiceResponse()
            text = "Thanks for calling! Have a great day!"
            lang = agent.language if agent and agent.language else "en"
            voice = agent.voice_type if agent and agent.voice_type else "female"
            tts_url = f"{settings.WEBHOOK_BASE_URL}/api/v1/tts/google-tts/audio?text={quote(text)}&lang={lang}&voice={voice}"
            response.play(tts_url)
            return HTMLResponse(str(response), media_type="application/xml")

    except Exception as e:  # pragma: no cover
        logger.error("ERROR occurred: %s", str(e), exc_info=True)
        logger.error("=== Call Events Webhook Failed ===")
        raise

