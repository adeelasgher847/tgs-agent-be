"""
Call Control Mixin for BidirectionalStreamHandler.
Handles call termination (goodbye, voicemail), transfer routing, and transcript recording.
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
import re
from typing import TYPE_CHECKING, Any
import uuid

from app.core.config import settings
from app.core.logger import logger
from app.routers.general_websocket import broadcast_call_status_update
from app.services.call_session_service import call_session_service
from app.services.transcript_service import transcript_service
from app.services.twilio_service import twilio_service
from app.services.voice_screening_qualification_service import (
    apply_resume_candidate_status_after_voice_screening,
)
from app.utils.ssml_utils import strip_ssml_tags
from app.utils.voice_twilio_utils import (
    get_twilio_credentials_for_call,
    twilio_caller_id_for_transfer_dial,
)

if TYPE_CHECKING:
    pass

# F-02 / F-06: Silence watchdog phrase pools
_SILENCE_REPROMPTS = [
    "Still there? Take your time.",
    "Just checking — are you still on the line?",
    "I'm still here whenever you're ready.",
]
_SILENCE_GOODBYES = [
    "It looks like we may have lost you — feel free to call back anytime. Goodbye!",
    "I'll go ahead and let you go — don't hesitate to call again. Take care!",
]


class CallControlMixin:
    """Call termination and transcript methods for BidirectionalStreamHandler."""

    async def _play_tts_message(self, text: str) -> None:
        """Enqueue a standalone spoken message to the TTS pipeline."""
        clean_text = (text or "").strip()
        if not clean_text:
            return
        tts_pipe = getattr(self, "_tts_pipeline", None)
        if tts_pipe is not None:
            if hasattr(tts_pipe, "queue_tts"):
                use_ssml = bool(getattr(self, "_use_ssml", False))
                await tts_pipe.queue_tts(
                    {
                        "text": clean_text,
                        "chunk_id": f"msg_{uuid.uuid4().hex[:8]}",
                        "use_ssml": use_ssml,
                        "is_final": True,
                    }
                )
                setattr(self, "_twilio_buffer_primed", False)
            elif hasattr(tts_pipe, "say"):
                await tts_pipe.say(clean_text)

    async def _check_and_end_call_if_goodbye(self, transcript: str):
        """
        Check if transcript contains goodbye words and end call if detected.
        Returns True if call was ended, False otherwise.

        Goodbye keywords detected:
        - thanks for calling
        - thank you for calling
        - bye, bye bye, goodbye
        - see you, see ya
        - have a great day, have a nice day
        - take care
        - that's all, that's it
        - i'm done, i'm finished
        - all done, all set
        """
        if self._call_ended:
            return False  # Already ended

        # Goodbye keywords/phrases (case-insensitive)
        goodbye_keywords = [
            "bye",
            "bye bye",
            "goodbye",
            "good bye",
            "see you",
            "see ya",
            "have a great day",
            "have a nice day",
            "thanks bye",
            "thank you bye",
            "we're done",
            "we're finished",
        ]

        # Convert transcript to lowercase for case-insensitive matching
        transcript_lower = transcript.lower().strip()

        # Check if any goodbye keyword/phrase is present in transcript
        for keyword in goodbye_keywords:
            if keyword in transcript_lower:
                try:
                    # Mark as ended to prevent multiple calls
                    self._call_ended = True

                    # Use shared status updater so CallLog + inbound CRM sync hooks run reliably.
                    if self.call_session:
                        updated = call_session_service.update_call_session_status(
                            self.db,
                            self.call_session.id,
                            "completed",
                            ended_reason="User said goodbye",
                        )
                        if updated:
                            self.call_session = updated

                    # End Twilio call with DB-derived credentials (no env fallback).
                    if self.call_sid and self.call_session:
                        try:
                            account_sid, auth_token = get_twilio_credentials_for_call(
                                self.db, self.call_session
                            )
                            twilio_service.end_call_with_credentials(
                                self.call_sid, account_sid, auth_token
                            )
                        except Exception as end_err:
                            logger.warning(
                                "Could not end Twilio call with DB credentials "
                                "(call_sid=%s, session=%s): %s",
                                self.call_sid,
                                self.call_session.id if self.call_session else None,
                                end_err,
                            )

                    # Broadcast call ended event
                    if self.call_session:
                        try:
                            await broadcast_call_status_update(
                                call_session_id=str(self.call_session.id),
                                status="completed",
                                metadata={
                                    "call_sid": self.call_sid,
                                    "stream_sid": self.stream_sid,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "message": "call_ended",
                                    "event": "goodbye_detected",
                                    "detected_phrase": keyword,
                                    "transcript": transcript,
                                    "reason": "User said goodbye",
                                },
                            )
                        except Exception as e:
                            logger.debug(
                                "WebSocket broadcast failed after goodbye: %s", e
                            )

                    # Shut down STT + LLM + TTS and signal the main loop to exit
                    asyncio.create_task(self._full_shutdown())
                    return True

                except Exception as e:
                    logger.error("Error ending call after goodbye: %s", e, exc_info=True)
                    return False

        return False

    async def _end_call_after_agent_request(self):
        """End the call when agent response contained [END_CALL] (after TTS has played).

        We deliberately wait a short grace period (~200ms) AFTER the streaming
        TTS path has finished pushing its trailing silence drain. Twilio's
        outbound media buffer plus carrier-side jitter buffers can otherwise
        drop the last 80–150 ms of the goodbye phrase when the WebSocket /
        media stream is torn down too aggressively. The grace is well below
        any human-perceptible "extra silence" but eliminates the clipped
        goodbye that production has been hitting.
        """
        if self._call_ended:
            return
        try:
            try:
                await asyncio.sleep(0.20)
            except asyncio.CancelledError:
                # If the surrounding task is being cancelled (e.g. global
                # shutdown), continue with hangup instead of raising —
                # there's no benefit to leaving the call in a half-ended
                # state.
                pass

            self._call_ended = True
            if self.call_session:
                if getattr(self, "_pending_resume_screening_qualify", False):
                    try:
                        apply_resume_candidate_status_after_voice_screening(
                            self.db, self.call_session
                        )
                    except (
                        Exception
                    ) as qual_exc:  # pragma: no cover - non-blocking for hangup
                        logger.warning(
                            "Voice screening qualify failed (session=%s): %s",
                            self.call_session.id,
                            qual_exc,
                            exc_info=True,
                        )
                    finally:
                        self._pending_resume_screening_qualify = False
                updated = call_session_service.update_call_session_status(
                    self.db,
                    self.call_session.id,
                    "completed",
                    ended_reason="Agent sent [END_CALL]",
                )
                if updated:
                    self.call_session = updated
            else:
                self._pending_resume_screening_qualify = False

            if self.call_sid and self.call_session:
                try:
                    account_sid, auth_token = get_twilio_credentials_for_call(
                        self.db, self.call_session
                    )
                    twilio_service.end_call_with_credentials(
                        self.call_sid, account_sid, auth_token
                    )
                except Exception as end_err:
                    logger.warning(
                        "Could not end Twilio call with DB credentials "
                        "(call_sid=%s, session=%s): %s",
                        self.call_sid,
                        self.call_session.id if self.call_session else None,
                        end_err,
                    )
            if self.call_session:
                try:
                    await broadcast_call_status_update(
                        call_session_id=str(self.call_session.id),
                        status="completed",
                        metadata={
                            "call_sid": self.call_sid,
                            "stream_sid": self.stream_sid,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "message": "call_ended",
                            "event": "end_call_token",
                            "reason": "Agent sent [END_CALL]",
                        },
                    )
                except Exception as e:
                    logger.debug("WebSocket broadcast after [END_CALL]: %s", e)

            # Shut down STT + LLM + TTS and signal the main loop to exit
            asyncio.create_task(self._full_shutdown())
        except Exception as e:
            logger.error("Error ending call after [END_CALL]: %s", e, exc_info=True)

    async def _transfer_after_agent_request(self):
        """Redirect live Twilio call to human transfer TwiML after TTS (cold Dial or warm Conference)."""
        if self._call_ended:
            return
        try:
            try:
                await asyncio.sleep(0.20)
            except asyncio.CancelledError:
                pass

            self._call_ended = True
            route = getattr(self.agent, "transfer_route", None) if self.agent else None
            if (
                not self.call_session
                or not route
                or getattr(route, "is_deleted", False)
            ):
                logger.warning(
                    "[TRANSFER_CALL] skipped: missing call_session or transfer_route "
                    "(session=%s)",
                    self.call_session.id if self.call_session else None,
                )
                # F-10: speak an apology before shutting down
                if getattr(self, "_tts_pipeline", None):
                    await self._tts_pipeline.queue_tts({"text": (
                        "I'm sorry, I wasn't able to connect you at this time. "
                        "Please try calling back and ask to speak with a team member directly. Goodbye!"
                    )})
                await asyncio.sleep(4.5)
                asyncio.create_task(self._full_shutdown())
                return

            if not self.call_sid:
                logger.warning("[TRANSFER_CALL] skipped: no Twilio call_sid")
                asyncio.create_task(self._full_shutdown())
                return

            meta = dict(self.call_session.call_metadata or {})
            meta["human_transfer"] = {
                "route_id": str(route.id),
                "friendly_name": route.friendly_name,
                "transfer_type": route.transfer_type,
            }
            self.call_session.call_metadata = meta
            self.db.commit()
            self.db.refresh(self.call_session)

            updated = call_session_service.update_call_session_status(
                self.db,
                self.call_session.id,
                "completed",
                ended_reason="Human transfer ([TRANSFER_CALL])",
                transferred=True,
            )
            if updated:
                self.call_session = updated

            # If call flow specifies stop_recording_on_transfer, halt recording before transfer handoff
            call_flow = getattr(self, "call_flow", None)
            if call_flow and getattr(call_flow, "stop_recording_on_transfer", False):
                logger.info(
                    "Halting call recording prior to transfer per call flow configuration"
                )
                if hasattr(self, "_teardown_livekit_recording"):
                    try:
                        await self._teardown_livekit_recording()
                    except Exception as rec_stop_err:
                        logger.warning(
                            "Error tearing down LiveKit recording on transfer: %s",
                            rec_stop_err,
                        )

            base = settings.WEBHOOK_BASE_URL.rstrip("/")
            sid_str = str(self.call_session.id)
            ttype = (route.transfer_type or "cold").lower()
            if ttype == "warm":
                redirect_url = (
                    f"{base}/api/v1/voice/webhook/transfer/conference-customer"
                    f"?callSessionId={sid_str}"
                )
            else:
                redirect_url = (
                    f"{base}/api/v1/voice/webhook/transfer/dial-cold"
                    f"?callSessionId={sid_str}"
                )

            try:
                account_sid, auth_token = get_twilio_credentials_for_call(
                    self.db, self.call_session
                )
                ok = twilio_service.redirect_call_with_credentials(
                    self.call_sid,
                    redirect_url,
                    account_sid,
                    auth_token,
                    method="POST",
                )
                if not ok:
                    logger.error(
                        "Transfer redirect failed call_sid=%s session=%s",
                        self.call_sid,
                        self.call_session.id,
                    )
                elif ttype == "warm":
                    from_num = twilio_caller_id_for_transfer_dial(self.call_session)
                    if not from_num:
                        logger.error(
                            "Warm transfer: no Twilio caller ID on session %s (type=%s)",
                            self.call_session.id,
                            self.call_session.call_type,
                        )
                    else:
                        sup_url = (
                            f"{base}/api/v1/voice/webhook/transfer/conference-supervisor"
                            f"?callSessionId={sid_str}"
                        )
                        status_cb = (
                            f"{base}/api/v1/voice/webhook/call-events"
                            f"?callSessionId={sid_str}"
                            f"&agentId={self.agent.id}&userId={self.call_session.user_id}"
                        )
                        try:
                            twilio_service.make_call_with_credentials(
                                to_number=route.phone_number,
                                from_number=from_num,
                                webhook_url=sup_url,
                                status_callback_url=status_cb,
                                account_sid=account_sid,
                                auth_token=auth_token,
                                record=False,
                            )
                        except Exception as dial_err:
                            logger.error(
                                "Warm transfer supervisor dial failed: %s",
                                dial_err,
                                exc_info=True,
                            )
            except Exception as redir_err:
                logger.error(
                    "Transfer redirect exception: %s",
                    redir_err,
                    exc_info=True,
                )

            if self.call_session:
                try:
                    await broadcast_call_status_update(
                        call_session_id=str(self.call_session.id),
                        status="completed",
                        metadata={
                            "call_sid": self.call_sid,
                            "stream_sid": self.stream_sid,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "message": "call_ended",
                            "event": "human_transfer",
                            "reason": "Agent sent [TRANSFER_CALL]",
                            "transfer_type": route.transfer_type,
                        },
                    )
                except Exception as br_err:
                    logger.debug("WebSocket broadcast after transfer: %s", br_err)

            asyncio.create_task(self._full_shutdown())
        except Exception as e:
            logger.error("Error during human transfer: %s", e, exc_info=True)

    def _resolve_flow(self) -> Any:
        """Resolve CallFlow with tenant-scoped filtering to prevent cross-tenant object access."""
        flow = getattr(self, "call_flow", None)
        if flow is not None:
            return flow
        sess = getattr(self, "call_session", None)
        db = getattr(self, "db", None)
        if sess and getattr(sess, "call_flow_id", None) and db:
            try:
                from app.models.call_flow import CallFlow
                from sqlalchemy import select

                stmt = select(CallFlow).where(
                    CallFlow.id == sess.call_flow_id,
                    CallFlow.is_deleted.is_(False),
                )
                if getattr(sess, "tenant_id", None):
                    stmt = stmt.where(CallFlow.tenant_id == sess.tenant_id)
                flow = db.execute(stmt).scalar_one_or_none()
                if flow:
                    self.call_flow = flow
                return flow
            except Exception as e:
                logger.warning(
                    "Flow fetch failed (call_flow_id=%s): %s",
                    getattr(sess, "call_flow_id", None),
                    e,
                )
        return None

    async def _check_and_end_call_if_voicemail(self, transcript: str):
        """
        Check if transcript contains voicemail keywords and handle voicemail action if detected.
        Returns True if call was ended, False otherwise.
        """
        if self._call_ended:
            return False  # Already ended

        # Check call_flow voicemail detection settings if attached
        has_flow_attached = bool(
            getattr(self, "call_session", None)
            and getattr(self.call_session, "call_flow_id", None)
        )
        flow = self._resolve_flow()
        if has_flow_attached and flow is None:
            return False

        # If flow has voicemail detection disabled -> bypass keyword check
        if flow is not None and not getattr(flow, "voicemail_detection_enabled", True):
            return False

        voicemail_action = (
            getattr(flow, "voicemail_action", "hang_up") if flow else "hang_up"
        )
        if voicemail_action == "continue":
            return False

        # Voicemail keywords/phrases (case-insensitive)
        voicemail_keywords = [
            "forwarded to voicemail",
            "forwarded to voice mail",
            "record your message",
            "press #",
            "pound key",
            "hang up",
            "at the tone",
            "after the tone",
            "after the beep",
        ]

        # Convert transcript to lowercase for case-insensitive matching
        transcript_lower = transcript.lower().strip()

        # Check if any voicemail keyword/phrase is present in transcript
        for keyword in voicemail_keywords:
            if keyword in transcript_lower:
                try:
                    # Mark as ended to prevent multiple calls
                    self._call_ended = True

                    # If action is leave_message, play voicemail message if configured
                    if voicemail_action == "leave_message" and flow:
                        voicemail_msg = (
                            getattr(flow, "voicemail_message", None) or ""
                        ).strip()
                        if voicemail_msg:
                            try:
                                await self._play_tts_message(voicemail_msg)
                            except Exception as tts_err:
                                logger.warning(
                                    "Could not play voicemail message before hangup: %s",
                                    tts_err,
                                )

                    # Use shared status updater so CallLog + inbound CRM sync hooks run reliably.
                    if self.call_session:
                        updated = call_session_service.update_call_session_status(
                            self.db,
                            self.call_session.id,
                            "completed",
                            ended_reason="Voicemail detected",
                        )
                        if updated:
                            self.call_session = updated

                    # End Twilio call immediately with DB-derived credentials (no env fallback).
                    if self.call_sid and self.call_session:
                        try:
                            account_sid, auth_token = get_twilio_credentials_for_call(
                                self.db, self.call_session
                            )
                            twilio_service.end_call_with_credentials(
                                self.call_sid, account_sid, auth_token
                            )
                        except Exception as end_err:
                            logger.warning(
                                "Could not end Twilio call with DB credentials "
                                "(call_sid=%s, session=%s): %s",
                                self.call_sid,
                                self.call_session.id if self.call_session else None,
                                end_err,
                            )

                    # Broadcast call ended event
                    if self.call_session:
                        try:
                            await broadcast_call_status_update(
                                call_session_id=str(self.call_session.id),
                                status="completed",
                                metadata={
                                    "call_sid": self.call_sid,
                                    "stream_sid": self.stream_sid,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "message": "call_ended",
                                    "event": "voicemail_detected",
                                    "detected_phrase": keyword,
                                    "transcript": transcript,
                                    "reason": "Voicemail detected",
                                },
                            )
                        except Exception as e:
                            logger.debug(
                                "WebSocket broadcast failed after voicemail detection: %s", e
                            )

                    # Shut down STT + LLM + TTS and signal the main loop to exit
                    asyncio.create_task(self._full_shutdown())
                    return True

                except Exception as e:
                    logger.error(
                        "Error ending call after voicemail detection: %s", e,
                        exc_info=True,
                    )
                    return False

        return False

    async def _check_and_handle_call_screener(self, transcript: str) -> bool:
        """
        Check if transcript contains automated call screening phrases and execute configured action.
        Returns True if call was ended (hang_up action), False otherwise (respond action / no screener).

        Supported call screening services:
        - Google Call Screen / Assistant Call Screening
        - Apple / iOS Call Screen
        - Samsung Smart Call / Bixby Call Assist
        - Automated IVR screening
        """
        if self._call_ended:
            return False

        screener_keywords = [
            "using a screening service",
            "screening service from google",
            "screening service",
            "google call screen",
            "go ahead and say why you're calling",
            "state your name and why you're calling",
            "who is calling and why",
            "please say your name and reason for calling",
        ]

        transcript_lower = transcript.lower().strip()
        for keyword in screener_keywords:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            match = re.search(pattern, transcript_lower)
            if match:
                # Check for negation immediately preceding the match (e.g. "not a screening service")
                start_pos = match.start()
                preceding_text = transcript_lower[
                    max(0, start_pos - 30) : start_pos
                ].strip()
                if re.search(
                    r"\b(?:not|no|never|isn't|aren't|wasn't|don't|doesn't)(?:\s+(?:a|an|using\s+a|using))?\s*$",
                    preceding_text,
                ):
                    continue

                flow = self._resolve_flow()
                action = (
                    getattr(flow, "call_screening_action", "respond")
                    if flow
                    else "respond"
                )

                if action == "hang_up":
                    try:
                        self._call_ended = True

                        if getattr(self, "call_session", None):
                            updated = call_session_service.update_call_session_status(
                                self.db,
                                self.call_session.id,
                                "completed",
                                ended_reason="Call screener detected",
                            )
                            if updated:
                                self.call_session = updated

                        if getattr(self, "call_sid", None) and self.call_session:
                            try:
                                account_sid, auth_token = (
                                    get_twilio_credentials_for_call(
                                        self.db, self.call_session
                                    )
                                )
                                twilio_service.end_call_with_credentials(
                                    self.call_sid, account_sid, auth_token
                                )
                            except Exception as end_err:
                                logger.warning(
                                    "Could not end Twilio call with DB credentials "
                                    "(call_sid=%s, session=%s): %s",
                                    self.call_sid,
                                    self.call_session.id if self.call_session else None,
                                    end_err,
                                )

                        if getattr(self, "call_session", None):
                            try:
                                await broadcast_call_status_update(
                                    call_session_id=str(self.call_session.id),
                                    status="completed",
                                    metadata={
                                        "call_sid": getattr(self, "call_sid", None),
                                        "stream_sid": getattr(self, "stream_sid", None),
                                        "timestamp": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "message": "call_ended",
                                        "event": "call_screener_detected",
                                        "detected_phrase": keyword,
                                        "transcript": transcript,
                                        "reason": "Call screener detected",
                                    },
                                )
                            except Exception as e:
                                logger.debug(
                                    "WebSocket broadcast failed after call screening hang up: %s",
                                    e,
                                )

                        asyncio.create_task(self._full_shutdown())
                        return True

                    except Exception as e:
                        logger.error(
                            "Error ending call after call screener detection: %s",
                            e,
                            exc_info=True,
                        )
                        return False
                else:
                    return False

        return False

    async def _check_and_handle_ivr_and_hold(self, transcript: str) -> bool:
        """Detect phone tree/IVR prompts or hold queues and handle accordingly."""
        if getattr(self, "_call_ended", False):
            return False

        if not transcript or not transcript.strip():
            return False

        flow = self._resolve_flow()
        if not flow or not getattr(flow, "ivr_enabled", False):
            return False

        ivr_keywords = [
            "press 1",
            "press 2",
            "press 3",
            "for english press",
            "menu options",
            "listen carefully as our menu options have changed",
            "main menu",
            "to speak with",
            "please press",
            "dial the extension",
            "press the pound key",
            "press star",
        ]

        hold_keywords = [
            "all of our agents are busy",
            "all representatives are currently busy",
            "your call is important to us",
            "please hold",
            "please remain on the line",
            "next available representative",
            "estimated wait time",
            "you are number in line",
            "thank you for holding",
            "please stay on the line",
        ]

        transcript_lower = transcript.lower().strip()

        # Check IVR menu keywords
        for keyword in ivr_keywords:
            if keyword in transcript_lower:
                action = getattr(flow, "ivr_action", "dial_through") or "dial_through"
                if action == "hang_up":
                    try:
                        self._call_ended = True

                        if self.call_session:
                            updated = call_session_service.update_call_session_status(
                                self.db,
                                self.call_session.id,
                                "completed",
                                ended_reason="IVR phone tree detected",
                            )
                            if updated:
                                self.call_session = updated

                        if self.call_sid and self.call_session:
                            try:
                                account_sid, auth_token = (
                                    get_twilio_credentials_for_call(
                                        self.db, self.call_session
                                    )
                                )
                                twilio_service.end_call_with_credentials(
                                    self.call_sid, account_sid, auth_token
                                )
                            except Exception as end_err:
                                logger.warning(
                                    "Could not end Twilio call with DB credentials "
                                    "(call_sid=%s, session=%s): %s",
                                    self.call_sid,
                                    self.call_session.id if self.call_session else None,
                                    end_err,
                                )

                        if self.call_session:
                            try:
                                await broadcast_call_status_update(
                                    call_session_id=str(self.call_session.id),
                                    status="completed",
                                    metadata={
                                        "call_sid": self.call_sid,
                                        "stream_sid": self.stream_sid,
                                        "timestamp": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "message": "call_ended",
                                        "event": "ivr_detected",
                                        "detected_phrase": keyword,
                                        "transcript": transcript,
                                        "reason": "IVR phone tree detected",
                                    },
                                )
                            except Exception as e:
                                logger.debug(
                                    "WebSocket broadcast failed after IVR detection: %s", e
                                )

                        asyncio.create_task(self._full_shutdown())
                        return True
                    except Exception as e:
                        logger.error(
                            "Error ending call after IVR detection: %s", e, exc_info=True
                        )
                        return False
                else:
                    logger.info(
                        "IVR phone tree detected ('%s'); action is %s, navigating menu",
                        keyword,
                        action,
                    )
                    return False

        # Check Hold queue keywords
        for keyword in hold_keywords:
            if keyword in transcript_lower:
                if getattr(flow, "ivr_wait_on_hold", False):
                    max_hold = getattr(flow, "ivr_max_hold_time", 120) or 120
                    logger.info(
                        "Hold queue detected ('%s'); waiting on hold up to %s seconds",
                        keyword,
                        max_hold,
                    )
                return False

        return False

    async def _check_and_handle_anti_bot(
        self, transcript: str, *, role: str = "user"
    ) -> bool:
        """
        Detect automated bots and synthetic voices, and terminate call if configured.

        Scoped strictly to caller utterances (role == 'user') to prevent the AI agent from
        false-positively triggering on its own greetings or disclosure scripts.
        """
        if getattr(self, "_call_ended", False):
            return False

        if role != "user":
            return False

        if not transcript or not transcript.strip():
            return False

        flow = self._resolve_flow()
        if not flow or not getattr(flow, "anti_bot_detection_enabled", False):
            return False

        bot_keywords = [
            "this is an automated call",
            "this is an automated message",
            "press 1 to speak",
            "press 1 to connect",
            "press 1 now",
            "press 2 to",
            "press 1 or press 2",
            "automated telephone banking",
            "you have reached the automated",
            "to hear this message again",
            "this is a pre-recorded message",
            "is a pre-recorded message",
            "synthetic voice detected",
            "robocall",
        ]

        transcript_lower = transcript.lower().strip()
        for keyword in bot_keywords:
            if keyword in transcript_lower:
                logger.info(
                    "Anti-bot detected automated caller signature on call %s: keyword=%r",
                    getattr(self, "call_sid", None),
                    keyword,
                )
                if getattr(self, "call_session", None):
                    meta = self.call_session.call_metadata or {}
                    if not isinstance(meta, dict):
                        meta = {}
                    meta["anti_bot"] = {
                        "detected": True,
                        "keyword": keyword,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    self.call_session.call_metadata = meta
                    if getattr(self, "db", None):
                        try:
                            from sqlalchemy.orm.attributes import flag_modified

                            flag_modified(self.call_session, "call_metadata")
                            self.db.commit()
                        except Exception as db_err:
                            logger.debug(
                                "Failed to persist anti-bot metadata: %s",
                                db_err,
                            )

                if getattr(flow, "terminate_on_fake_voice", False):
                    logger.info(
                        "Bot/fake voice detected on call %s — terminating immediately",
                        getattr(self, "call_sid", None),
                    )
                    try:
                        self._call_ended = True

                        if getattr(self, "call_session", None):
                            updated = call_session_service.update_call_session_status(
                                self.db,
                                self.call_session.id,
                                "completed",
                                ended_reason="Bot or fake voice detected",
                            )
                            if updated:
                                self.call_session = updated

                        if getattr(self, "call_sid", None) and getattr(
                            self, "call_session", None
                        ):
                            try:
                                account_sid, auth_token = (
                                    get_twilio_credentials_for_call(
                                        self.db, self.call_session
                                    )
                                )
                                twilio_service.end_call_with_credentials(
                                    self.call_sid, account_sid, auth_token
                                )
                            except Exception as end_err:
                                logger.warning(
                                    "Could not end Twilio call with DB credentials "
                                    "(call_sid=%s, session=%s): %s",
                                    self.call_sid,
                                    self.call_session.id if self.call_session else None,
                                    end_err,
                                )

                        if getattr(self, "call_session", None):
                            try:
                                await broadcast_call_status_update(
                                    call_session_id=str(self.call_session.id),
                                    status="completed",
                                    metadata={
                                        "call_sid": getattr(self, "call_sid", None),
                                        "stream_sid": getattr(self, "stream_sid", None),
                                        "timestamp": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "message": "call_ended",
                                        "event": "bot_detected",
                                        "detected_phrase": keyword,
                                        "transcript": transcript,
                                        "reason": "Bot or fake voice detected",
                                    },
                                )
                            except Exception as e:
                                logger.debug(
                                    "WebSocket broadcast failed after bot detection: %s",
                                    e,
                                )

                        asyncio.create_task(self._full_shutdown())
                        return True
                    except Exception as e:
                        logger.error(
                            "Error ending call after bot detection: %s",
                            e,
                            exc_info=True,
                        )
                        return False
                return False

        return False

    async def _check_and_handle_compliance_monitoring(self, transcript: str) -> None:
        """Monitor conversation against compliance policies and flag violations in metadata."""
        if not transcript or not transcript.strip():
            return

        flow = self._resolve_flow()
        if not flow or not getattr(flow, "compliance_monitoring_enabled", False):
            return

        compliance_triggers = [
            "unauthorized transaction",
            "wire money immediately",
            "share your password",
            "tell me your credit card cvv",
            "send gift cards",
            "social security number is",
            "my ssn is",
        ]

        transcript_lower = transcript.lower().strip()
        matched = []
        for trigger in compliance_triggers:
            if trigger in transcript_lower:
                matched.append(trigger)

        if matched and getattr(self, "call_session", None):
            logger.warning(
                "Compliance monitoring flagged potential policy violation on call %s: triggers=%r",
                getattr(self, "call_sid", None),
                matched,
            )
            meta = self.call_session.call_metadata or {}
            if not isinstance(meta, dict):
                meta = {}
            comp = meta.get("compliance", {"flagged": True, "violations": []})
            if not isinstance(comp, dict):
                comp = {"flagged": True, "violations": []}
            comp["flagged"] = True
            violations = comp.get("violations", [])
            if not isinstance(violations, list):
                violations = []
            for m in matched:
                violations.append(
                    {
                        "trigger": m,
                        "transcript_snippet": transcript[:200],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            comp["violations"] = violations
            meta["compliance"] = comp
            self.call_session.call_metadata = meta

            if getattr(self, "db", None):
                try:
                    from sqlalchemy.orm.attributes import flag_modified

                    flag_modified(self.call_session, "call_metadata")
                    self.db.commit()
                except Exception as db_err:
                    logger.debug("Failed to persist compliance metadata: %s", db_err)

    async def handle_dtmf_message(self, message: dict) -> None:
        """Handle incoming WebSocket DTMF event with debounce buffering and limit enforcement."""
        if getattr(self, "_call_ended", False):
            return

        flow = self._resolve_flow()
        if not flow or not getattr(flow, "dtmf_enabled", False):
            return

        digit = (message.get("dtmf") or {}).get("digit") or message.get("digit")
        if not digit:
            return

        if not hasattr(self, "_dtmf_buffer"):
            self._dtmf_buffer = ""
        if not hasattr(self, "_dtmf_debounce_task"):
            self._dtmf_debounce_task = None
        if not hasattr(self, "_dtmf_exceeded_count"):
            self._dtmf_exceeded_count = 0

        # Suppress STT acoustic processing during digit press if interruption not allowed
        if not getattr(flow, "dtmf_allow_caller_interruption", False):
            self._dtmf_suppress_stt = True

        self._cancel_silence_watchdog()

        if self._dtmf_debounce_task and not self._dtmf_debounce_task.done():
            self._dtmf_debounce_task.cancel()

        self._dtmf_buffer += str(digit)
        max_digits_raw = getattr(flow, "dtmf_max_digits", 50)
        max_digits = max_digits_raw if max_digits_raw is not None else 50

        if len(self._dtmf_buffer) > max_digits:
            self._dtmf_exceeded_count += 1
            allowed_raw = getattr(flow, "dtmf_allowed_exceeded_attempts", 10)
            allowed = allowed_raw if allowed_raw is not None else 10
            action = getattr(flow, "dtmf_exceeded_action", "end_call") or "end_call"
            if self._dtmf_exceeded_count >= allowed and action == "end_call":
                logger.warning(
                    "DTMF max digits exceeded %d times (allowed: %d) — ending call",
                    self._dtmf_exceeded_count,
                    allowed,
                )
                self._dtmf_buffer = ""
                self._dtmf_suppress_stt = False
                try:
                    self._call_ended = True

                    # Play configured end-call message before hangup if present
                    end_msg = (
                        getattr(flow, "dtmf_end_call_message", None) or ""
                    ).strip()
                    if end_msg:
                        try:
                            await self._play_tts_message(end_msg)
                        except Exception as tts_err:
                            logger.warning(
                                "Could not play DTMF end call message before hangup: %s",
                                tts_err,
                            )

                    if self.call_session:
                        updated = call_session_service.update_call_session_status(
                            self.db,
                            self.call_session.id,
                            "completed",
                            ended_reason="DTMF input limit exceeded",
                        )
                        if updated:
                            self.call_session = updated

                    if self.call_sid and self.call_session:
                        try:
                            account_sid, auth_token = get_twilio_credentials_for_call(
                                self.db, self.call_session
                            )
                            twilio_service.end_call_with_credentials(
                                self.call_sid, account_sid, auth_token
                            )
                        except Exception as end_err:
                            logger.warning(
                                "Could not end Twilio call on DTMF limit: %s", end_err
                            )

                    if self.call_session:
                        try:
                            await broadcast_call_status_update(
                                call_session_id=str(self.call_session.id),
                                status="completed",
                                metadata={
                                    "call_sid": self.call_sid,
                                    "stream_sid": self.stream_sid,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "message": "call_ended",
                                    "event": "dtmf_limit_exceeded",
                                    "reason": "DTMF input limit exceeded",
                                },
                            )
                        except Exception as e:
                            logger.debug(
                                "WebSocket broadcast failed after DTMF limit exceeded: %s", e
                            )

                    asyncio.create_task(self._full_shutdown())
                except Exception as e:
                    logger.error("Error ending call on DTMF limit: %s", e, exc_info=True)
                return
            else:
                self._dtmf_buffer = ""
                self._dtmf_suppress_stt = False
                return

        delay_raw = getattr(flow, "dtmf_button_press_delay", 2)
        delay = delay_raw if delay_raw is not None else 2
        self._dtmf_debounce_task = asyncio.create_task(
            self._flush_dtmf_buffer_after_delay(float(delay))
        )

    async def _flush_dtmf_buffer_after_delay(self, delay: float) -> None:
        """Wait debounce delay and flush collected DTMF digits to conversation processing."""
        try:
            await asyncio.sleep(delay)
            digits = getattr(self, "_dtmf_buffer", "")
            self._dtmf_buffer = ""
            if digits:
                logger.info("DTMF buffer flushed: %r", digits)
                await self._process_transcript(
                    f"User input DTMF: {digits}", confidence=1.0
                )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Error flushing DTMF buffer: %s", e, exc_info=True)
        finally:
            self._dtmf_suppress_stt = False

    def _cancel_silence_watchdog(self) -> None:
        """Cancel the active silence watchdog timer and reset retry count on user activity."""
        task = getattr(self, "_silence_watchdog_task", None)
        if task and not task.done():
            task.cancel()
        self._silence_watchdog_task = None
        self._silence_retry_count = 0

    def _arm_silence_watchdog(self) -> None:
        """Arm or re-arm background silence watchdog timer when agent finishes speaking."""
        if getattr(self, "_call_ended", False):
            return
        task = getattr(self, "_silence_watchdog_task", None)
        if task and not task.done():
            task.cancel()
        self._silence_watchdog_task = asyncio.create_task(self._silence_watchdog_loop())

    async def _silence_watchdog_loop(self) -> None:
        """Background loop that plays silence reminders and terminates after final timeout."""
        try:
            flow = self._resolve_flow()
            if not flow:
                return

            silence_timeout_raw = getattr(flow, "silence_timeout", 10)
            silence_timeout = (
                silence_timeout_raw if silence_timeout_raw is not None else 10
            )
            reminder_retries_raw = getattr(flow, "reminder_retries", 1)
            reminder_retries = (
                reminder_retries_raw if reminder_retries_raw is not None else 1
            )
            end_call_after_reminder_raw = getattr(flow, "end_call_after_reminder", 10)
            end_call_after_reminder = (
                end_call_after_reminder_raw
                if end_call_after_reminder_raw is not None
                else 10
            )
            reminder_messages = list(getattr(flow, "reminder_messages", []) or [])

            while not getattr(self, "_call_ended", False):
                if getattr(self, "_silence_retry_count", 0) < reminder_retries:
                    await asyncio.sleep(float(silence_timeout))
                    if getattr(self, "_call_ended", False):
                        break
                    # If agent is generating response or TTS is playing, skip reminder this tick
                    _llm_task = getattr(self, "_llm_response_task", None)
                    _is_busy = (
                        getattr(self, "_is_tts_playing", False)
                        or getattr(self, "_turn_response_started", False)
                        or getattr(self, "is_speaking", False)
                        or (_llm_task is not None and not _llm_task.done())
                    )
                    if _is_busy:
                        await asyncio.sleep(1)
                        continue

                    retry_idx = getattr(self, "_silence_retry_count", 0)
                    if (
                        retry_idx < len(reminder_messages)
                        and reminder_messages[retry_idx]
                    ):
                        msg = reminder_messages[retry_idx]
                    else:
                        msg = _SILENCE_REPROMPTS[retry_idx % len(_SILENCE_REPROMPTS)]

                    logger.info(
                        "Silence timeout (%ss) reached; playing reminder %d/%d: %r",
                        silence_timeout,
                        retry_idx + 1,
                        reminder_retries,
                        msg,
                    )
                    self._silence_retry_count = retry_idx + 1
                    try:
                        await self._play_tts_message(msg)
                    except Exception as play_err:
                        logger.warning(
                            "Error playing silence reminder TTS: %s", play_err
                        )
                else:
                    # Final reminder retry exhausted: wait end_call_after_reminder seconds then terminate
                    await asyncio.sleep(float(end_call_after_reminder))
                    if getattr(self, "_call_ended", False):
                        break
                    _llm_task = getattr(self, "_llm_response_task", None)
                    _is_busy = (
                        getattr(self, "_is_tts_playing", False)
                        or getattr(self, "_turn_response_started", False)
                        or getattr(self, "is_speaking", False)
                        or (_llm_task is not None and not _llm_task.done())
                    )
                    if _is_busy:
                        await asyncio.sleep(1)
                        continue

                    logger.info(
                        "Silence timeout after reminders (%ss) reached — ending call",
                        end_call_after_reminder,
                    )
                    try:
                        # Try to escalate to human agent before hanging up
                        transfer_route = getattr(
                            getattr(self, "agent", None), "transfer_route", None
                        )
                        if transfer_route:
                            try:
                                await self._play_tts_message(
                                    "Let me connect you with someone who can help. One moment."
                                )
                                await asyncio.sleep(2)
                            except Exception:
                                pass
                            await self._transfer_after_agent_request()
                            return
                        # No transfer route — speak a warm goodbye before hanging up
                        try:
                            await self._play_tts_message(random.choice(_SILENCE_GOODBYES))
                            await asyncio.sleep(3.5)
                        except Exception:
                            pass
                        self._call_ended = True
                        if self.call_session:
                            updated = call_session_service.update_call_session_status(
                                self.db,
                                self.call_session.id,
                                "completed",
                                ended_reason="Silence timeout after reminders",
                            )
                            if updated:
                                self.call_session = updated

                        if self.call_sid and self.call_session:
                            try:
                                account_sid, auth_token = (
                                    get_twilio_credentials_for_call(
                                        self.db, self.call_session
                                    )
                                )
                                twilio_service.end_call_with_credentials(
                                    self.call_sid, account_sid, auth_token
                                )
                            except Exception as end_err:
                                logger.warning(
                                    "Could not end Twilio call on silence timeout: %s",
                                    end_err,
                                )

                        if self.call_session:
                            try:
                                await broadcast_call_status_update(
                                    call_session_id=str(self.call_session.id),
                                    status="completed",
                                    metadata={
                                        "call_sid": self.call_sid,
                                        "stream_sid": self.stream_sid,
                                        "timestamp": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                        "message": "call_ended",
                                        "event": "silence_timeout",
                                        "reason": "Silence timeout after reminders",
                                    },
                                )
                            except Exception as b_err:
                                logger.debug(
                                    "WebSocket broadcast failed after silence timeout: %s", b_err
                                )

                        asyncio.create_task(self._full_shutdown())
                    except Exception as end_e:
                        logger.error(
                            "Error ending call on silence timeout: %s", end_e,
                            exc_info=True,
                        )
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Unexpected error in silence watchdog: %s", e, exc_info=True)
        finally:
            self._silence_retry_count = 0

    def _arm_max_duration_timer(self) -> None:
        """Arm hard timer for maximum call duration."""
        if getattr(self, "_call_ended", False):
            return
        flow = self._resolve_flow()
        if not flow:
            return

        max_dur_raw = getattr(flow, "max_call_duration", 1800)
        max_dur = max_dur_raw if max_dur_raw is not None else 1800

        task = getattr(self, "_max_duration_task", None)
        if task and not task.done():
            task.cancel()
        self._max_duration_task = asyncio.create_task(
            self._max_duration_watchdog(float(max_dur))
        )

    async def _max_duration_watchdog(self, max_seconds: float) -> None:
        """Sleep for max call duration and terminate call with optional departure message."""
        try:
            await asyncio.sleep(max_seconds)
            if getattr(self, "_call_ended", False):
                return

            logger.info("Max call duration (%ss) reached — ending call", max_seconds)
            flow = self._resolve_flow()
            departure_msg = (getattr(flow, "max_duration_message", None) or "").strip()
            if departure_msg:
                try:
                    await self._play_tts_message(departure_msg)
                except Exception as tts_err:
                    logger.warning("Error playing max duration message: %s", tts_err)

            self._call_ended = True
            if self.call_session:
                updated = call_session_service.update_call_session_status(
                    self.db,
                    self.call_session.id,
                    "completed",
                    ended_reason="Max call duration reached",
                )
                if updated:
                    self.call_session = updated

            if self.call_sid and self.call_session:
                try:
                    account_sid, auth_token = get_twilio_credentials_for_call(
                        self.db, self.call_session
                    )
                    twilio_service.end_call_with_credentials(
                        self.call_sid, account_sid, auth_token
                    )
                except Exception as end_err:
                    logger.warning(
                        "Could not end Twilio call on max duration limit: %s",
                        end_err,
                    )

            if self.call_session:
                try:
                    await broadcast_call_status_update(
                        call_session_id=str(self.call_session.id),
                        status="completed",
                        metadata={
                            "call_sid": self.call_sid,
                            "stream_sid": self.stream_sid,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "message": "call_ended",
                            "event": "max_duration_reached",
                            "reason": "Max call duration reached",
                        },
                    )
                except Exception as b_err:
                    logger.debug(
                        "WebSocket broadcast failed after max duration reached: %s", b_err
                    )

            asyncio.create_task(self._full_shutdown())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                "Unexpected error in max duration watchdog: %s", e, exc_info=True
            )

    async def _add_to_transcript(
        self,
        role: str,
        message: str,
        message_type: str = "speech",
        confidence: float | None = None,
        message_metadata: dict | None = None,
        defer_post_write: bool = False,
    ):
        """Add message to transcript (SSML tags are automatically stripped)"""
        try:
            if not self.call_session:
                return

            # Strip SSML tags before saving to transcript (keep only clean text)
            clean_message = strip_ssml_tags(message)

            # Final dedupe gate for spoken agent replies (agent_response / greeting only —
            # calendar_slots / calendar_booking are informational and must never be skipped).
            # If the same line was committed within the last ~25s we skip the DB write AND
            # the WebSocket broadcast so the user/dashboard never sees duplicate lines.
            if role == "agent" and message_type in {"agent_response", "greeting"}:
                user_text_meta = None
                if message_metadata:
                    user_text_meta = message_metadata.get(
                        "user_text"
                    ) or message_metadata.get("query")
                if self._is_duplicate_agent_line(user_text_meta, clean_message):
                    logger.info(
                        "TranscriptDedupe: skipping duplicate agent line (type=%s, msg=%r)",
                        message_type,
                        clean_message[:80],
                    )
                    return

            hipaa_enabled = bool(
                getattr(self, "call_flow", None)
                and getattr(self.call_flow, "hipaa_compliance", False)
            )

            added = await transcript_service.add_and_broadcast_message(
                db=self.db,
                call_session_id=self.call_session.id,
                role=role,
                message=clean_message,  # Save clean text without SSML
                message_type=message_type,
                agent_id=self.agent.id if self.agent else None,
                user_id=self.call_session.user_id,
                confidence=confidence,
                metadata=message_metadata,
                hipaa_enabled=hipaa_enabled,
            )
            if added is None:
                return

            # Mirror to Redis for live insights polling (key: call_transcript:{room_name}).
            # Only speech turns (client/agent) are useful for live analysis — skip system
            # messages, greeting meta, etc.
            if role in ("client", "agent") and message_type in (
                "speech",
                "agent_response",
                "greeting",
            ):
                try:
                    import json as _json
                    from app.utils.redis_client import get_redis

                    _redis = get_redis()
                    if _redis is not None:
                        _key = f"call_transcript:room_{self.call_session.id}"
                        _raw = await _redis.get(_key)
                        _turns: list = _json.loads(_raw) if _raw else []
                        _turns.append({"role": role, "text": clean_message})
                        await _redis.set(_key, _json.dumps(_turns), ex=7200)
                except Exception as _re:
                    logger.debug("Redis transcript mirror failed (non-fatal): %s", _re)

            # Remember committed agent lines for future dedupe / turn-coordination.
            if role == "agent" and message_type in {"agent_response", "greeting"}:
                user_text_meta = None
                if message_metadata:
                    user_text_meta = message_metadata.get(
                        "user_text"
                    ) or message_metadata.get("query")
                self._remember_agent_turn(user_text_meta, clean_message)

            # Keep in-memory history cache in sync so generate_and_stream_response
            # never needs to re-parse the call_transcript JSON.
            if role in ("client", "agent") and message_type not in (
                "greeting",
                "system",
                "status",
            ):
                self._conversation_history_cache.append((role, clean_message))

            if not defer_post_write:
                # Legacy denormalized transcript payload used by older read paths.
                conversation = transcript_service.get_conversation_array(
                    self.db, self.call_session.id
                )
                self.call_session.call_transcript = conversation
                self.db.commit()

            from app.services.call_session_contact_state import (
                sync_contact_intake_after_message,
            )

            sync_contact_intake_after_message(
                self.db,
                self.call_session.id,
                role=role,
                message=clean_message,
            )
            try:
                self.db.refresh(self.call_session)
            except Exception as exc:
                logger.debug(
                    "Failed to refresh call_session %s after transcript update: %s",
                    self.call_session.id,
                    exc,
                )

        except Exception as e:
            logger.error("Error in _add_to_transcript: %s", e, exc_info=True)
