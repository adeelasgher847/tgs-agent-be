"""
Call Control Mixin for BidirectionalStreamHandler.
Handles call termination (goodbye, voicemail), transfer routing, and transcript recording.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from app.core.config import settings
from app.core.logger import logger
from app.routers.general_websocket import broadcast_call_status_update
from app.services.call_session_service import call_session_service
from app.services.transcript_service import transcript_service
from app.services.twilio_service import twilio_service
from app.services.voice_screening_qualification_service import apply_resume_candidate_status_after_voice_screening
from app.utils.ssml_utils import strip_ssml_tags
from app.utils.voice_twilio_utils import get_twilio_credentials_for_call

if TYPE_CHECKING:
    pass


class CallControlMixin:
    """Call termination and transcript methods for BidirectionalStreamHandler."""

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
            "we're finished"
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
                                    "reason": "User said goodbye"
                                }
                            )
                        except Exception as e:
                            logger.debug(f"WebSocket broadcast failed after goodbye: {e}")

                    # Shut down STT + LLM + TTS and signal the main loop to exit
                    asyncio.create_task(self._full_shutdown())
                    return True
                    
                except Exception as e:
                    logger.error(f"Error ending call after goodbye: {e}", exc_info=True)
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
                        apply_resume_candidate_status_after_voice_screening(self.db, self.call_session)
                    except Exception as qual_exc:  # pragma: no cover - non-blocking for hangup
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
                    logger.debug(f"WebSocket broadcast after [END_CALL]: {e}")

            # Shut down STT + LLM + TTS and signal the main loop to exit
            asyncio.create_task(self._full_shutdown())
        except Exception as e:
            logger.error(f"Error ending call after [END_CALL]: {e}", exc_info=True)

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
            if not self.call_session or not route or getattr(route, "is_deleted", False):
                logger.warning(
                    "[TRANSFER_CALL] skipped: missing call_session or transfer_route "
                    "(session=%s)",
                    self.call_session.id if self.call_session else None,
                )
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
    
    async def _check_and_end_call_if_voicemail(self, transcript: str):
        """
        Check if transcript contains voicemail keywords and end call if detected.
        Returns True if call was ended, False otherwise.
        
        Voicemail keywords detected:
        - voicemail, voice mail
        - forwarded to voicemail
        - unavailable
        - no one is available, no 1 is available
        - record your message
        - press pound, press #, pound key
        - hang up
        - at the tone
        """
        if self._call_ended:
            return False  # Already ended
        
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
            "after the beep"
        ]
        
        # Convert transcript to lowercase for case-insensitive matching
        transcript_lower = transcript.lower().strip()
        
        # Check if any voicemail keyword/phrase is present in transcript
        for keyword in voicemail_keywords:
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
                                    "reason": "Voicemail detected"
                                }
                            )
                        except Exception as e:
                            logger.debug(f"WebSocket broadcast failed after voicemail detection: {e}")

                    # Shut down STT + LLM + TTS and signal the main loop to exit
                    asyncio.create_task(self._full_shutdown())
                    return True
                    
                except Exception as e:
                    logger.error(f"Error ending call after voicemail detection: {e}", exc_info=True)
                    return False
        
        return False
    
    async def _add_to_transcript(
        self,
        role: str,
        message: str,
        message_type: str = "speech",
        confidence: Optional[float] = None,
        message_metadata: Optional[dict] = None,
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
                    user_text_meta = message_metadata.get("user_text") or message_metadata.get("query")
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
                "speech", "agent_response", "greeting"
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
                    user_text_meta = message_metadata.get("user_text") or message_metadata.get("query")
                self._remember_agent_turn(user_text_meta, clean_message)

            # Keep in-memory history cache in sync so generate_and_stream_response
            # never needs to re-parse the call_transcript JSON.
            if role in ("client", "agent") and message_type not in ("greeting", "system", "status"):
                self._conversation_history_cache.append((role, clean_message))
            
            if not defer_post_write:
                # Legacy denormalized transcript payload used by older read paths.
                conversation = transcript_service.get_conversation_array(
                    self.db, self.call_session.id
                )
                self.call_session.call_transcript = conversation
                self.db.commit()

            from app.services.call_session_contact_state import sync_contact_intake_after_message

            sync_contact_intake_after_message(
                self.db,
                self.call_session.id,
                role=role,
                message=clean_message,
            )
            try:
                self.db.refresh(self.call_session)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Error in _add_to_transcript: {e}", exc_info=True)
    
