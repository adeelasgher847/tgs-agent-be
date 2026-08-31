"""
Booking & Calendar Token Mixin for BidirectionalStreamHandler.
Handles Calendly-backed booking (via Gemini function-calling), in-call
calendar-slot memory/dedup, and TTS-safe control-token stripping.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, date, timedelta, timezone
from typing import TYPE_CHECKING

from app.core.config import settings
from app.core.logger import logger
from app.utils.eleven_tts_text import strip_eleven_v3_style_tags_for_non_eleven_tts

if TYPE_CHECKING:
    pass

_RE_VOICE_END_CALL = re.compile(r"\[\s*END_CALL\s*\]", re.IGNORECASE)
_RE_VOICE_SCREENING_QUALIFIED = re.compile(
    r"\[\s*SCREENING_QUALIFIED\s*\]", re.IGNORECASE
)


class BookingMixin:
    """Calendar and appointment booking methods for BidirectionalStreamHandler."""

    # ── Calendly (Gemini function-calling) ──────────────────────────────────────

    def _calendly_enabled(self) -> bool:
        """True when this call's flow has `calendly_integration_enabled` set."""
        call_flow = getattr(self, "call_flow", None)
        flow_settings = getattr(call_flow, "settings", None) if call_flow else None
        if not isinstance(flow_settings, dict):
            return False
        return bool(flow_settings.get("calendly_integration_enabled"))

    async def _execute_calendly_tool_call(self, name: str, args: dict) -> dict:
        """
        Resolve a Gemini function_call (check_availability / book_appointment)
        against the Calendly service backend. Always returns a JSON-serializable
        dict — errors are surfaced to the model as {"error": "..."} so it can
        recover gracefully instead of crashing the turn.
        """
        from app.services import calendly_service

        if not self.call_session:
            return {"error": "no active call session"}
        tenant_id = self.call_session.tenant_id

        try:
            if name == "check_availability":
                # Calendly's event_type_available_times endpoint has no duration
                # parameter — slot length is fixed by the connected event type,
                # so a caller-requested duration cannot filter/size the results.
                date_str = (args.get("date") or "").strip()
                target = self._resolve_calendly_target_date(date_str)
                date_from = datetime.combine(
                    target, datetime.min.time(), tzinfo=timezone.utc
                )
                date_to = date_from + timedelta(days=1)
                slots = await calendly_service.get_available_slots(
                    self.db, tenant_id, date_from, date_to
                )
                self._last_offered_calendar_slots = [
                    s["slot_start"] for s in slots if s["available"]
                ]
                self._last_selected_calendar_slot = None
                self._last_requested_calendar_date = target
                return {
                    "date": target.isoformat(),
                    "slots": [
                        {
                            "slot_start": s["slot_start"].isoformat(),
                            "available": s["available"],
                        }
                        for s in slots
                    ],
                }

            if name == "book_appointment":
                slot_raw = (args.get("slot") or "").strip()
                attendee_email = (args.get("attendee_email") or "").strip()
                if not slot_raw or not attendee_email:
                    return {"error": "slot and attendee_email are required"}
                slot_dt = self._resolve_cached_calendar_slot(slot_raw)
                if slot_dt is None:
                    try:
                        slot_dt = datetime.fromisoformat(
                            slot_raw.replace("Z", "+00:00")
                        )
                    except ValueError:
                        return {"error": f"invalid slot datetime: {slot_raw}"}
                attendee_name = ""
                try:
                    from app.services.call_session_contact_state import (
                        get_contact_intake,
                    )

                    intake = get_contact_intake(self.call_session)
                    attendee_name = (intake.get("name") or "").strip()
                except Exception as exc:
                    logger.debug(
                        "Failed to get contact intake name for booking: %s", exc
                    )
                summary = (self.call_session.transcript_summary or "").strip()
                result = await calendly_service.book_appointment(
                    self.db,
                    tenant_id,
                    start_time=slot_dt,
                    attendee_email=attendee_email,
                    attendee_name=attendee_name or "Caller",
                    description=summary or None,
                )
                return {
                    "booked": True,
                    "slot_start": slot_dt.isoformat(),
                    "calendly_event": result.get("resource", result),
                }

            return {"error": f"unknown function: {name}"}
        except ValueError as ve:
            return {"error": str(ve)}
        except Exception as e:
            logger.error("Calendly tool call %s failed: %s", name, e, exc_info=True)
            return {"error": "internal error executing Calendly tool call"}

    @staticmethod
    def _resolve_calendly_target_date(raw_date: str) -> date:
        today = datetime.now(timezone.utc).date()
        low = (raw_date or "").strip().lower()
        if low in ("", "today"):
            return today
        if low == "tomorrow":
            return today + timedelta(days=1)
        try:
            return date.fromisoformat(low)
        except ValueError:
            return today + timedelta(days=1)

    async def _run_calendly_tool_turn(
        self,
        *,
        llm_service,
        user_text: str,
        system_prompt: str,
        conversation_history: list,
        model_name: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """
        Run one LLM turn with Gemini native tool-calling for Calendly-enabled
        agents. Completely bypasses the legacy [CHECK_SLOTS:...] /
        [BOOK_APPOINTMENT:...] regex-token pipeline — the model resolves
        availability/booking via function calls instead.
        """
        return await llm_service.generate_with_tools(
            prompt=user_text,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_executor=self._execute_calendly_tool_call,
        )

    # ── Calendar token handlers ───────────────────────────────────────────────

    @staticmethod
    def _normalize_calendar_slot_key(value: str) -> str:
        value = (value or "").strip().lower().replace(".", "")
        return re.sub(r"\s+", " ", value)

    def _is_booking_intent_turn(self, user_text: str, model_text: str = "") -> bool:
        """Conservative booking-intent detector for token-budget and fallback extraction."""
        haystack = f"{user_text or ''} {model_text or ''}".lower()
        if not haystack.strip():
            return False
        booking_keywords = (
            "book",
            "booking",
            "schedule",
            "appointment",
            "reschedule",
            "slot",
            "available slot",
            "am",
            "pm",
            "a.m",
            "p.m",
            "date",
            "time",
            "tomorrow",
            "today",
        )
        return any(k in haystack for k in booking_keywords)

    def _should_use_latency_fastpath(
        self, user_text: str, booking_intent_turn: bool
    ) -> bool:
        """
        Fast-path only for short/simple turns where heavy context is unlikely to help.
        Keeps booking/business-intent turns on the full context path.
        """
        if not bool(getattr(settings, "VOICE_ENABLE_LATENCY_FASTPATH", True)):
            return False
        if booking_intent_turn:
            return False
        text = (user_text or "").strip().lower()
        if not text:
            return False
        max_words = int(getattr(settings, "VOICE_FASTPATH_MAX_WORDS", 7) or 7)
        words = text.split()
        if len(words) > max_words:
            return False
        heavy_intent_markers = (
            "price",
            "pricing",
            "cost",
            "address",
            "phone",
            "email",
            "website",
            "service",
            "book",
            "booking",
            "appointment",
            "schedule",
            "slot",
            "quote",
            "area",
            "location",
            "city",
            "state",
            "zip",
            "county",
            "region",
        )
        return not any(k in text for k in heavy_intent_markers)

    def _is_booking_context_active(self, user_text: str = "") -> bool:
        return bool(
            self._last_offered_calendar_slots
            or self._last_requested_calendar_date
            or self._last_selected_calendar_slot
            or self._is_booking_intent_turn(user_text)
        )

    @staticmethod
    def _normalize_turn_text(text: str) -> str:
        cleaned = re.sub(r"[^\w\s:]", " ", (text or "").lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _has_recent_duplicate_reply_for(self, user_norm: str) -> bool:
        """
        True if a committed agent reply already handled this exact user turn within
        `_DUP_USER_TURN_WINDOW_SEC`. Prevents the "user says 'Hello?' twice → agent
        repeats the same greeting" failure mode. O(5) cost.
        """
        if not user_norm:
            return False
        now = time.monotonic()
        for u_norm, _a_norm, ts in self._recent_agent_pairs:
            if (
                (now - ts) < self._DUP_USER_TURN_WINDOW_SEC
                and u_norm
                and u_norm == user_norm
            ):
                return True
        return False

    def _is_duplicate_agent_line(self, user_text: str | None, agent_text: str) -> bool:
        """
        Transcript-level guard: within `_AGENT_LINE_DEDUP_WINDOW_SEC`, the same agent
        line (normalized) is treated as a duplicate even if the user turn differs. This
        is the final safety net that stops the visible transcript from ever showing the
        same agent message twice in quick succession.
        """
        if not agent_text:
            return False
        a_norm = self._normalize_turn_text(agent_text)
        if not a_norm:
            return False
        now = time.monotonic()
        u_norm = self._normalize_turn_text(user_text or "")
        for prev_u, prev_a, ts in self._recent_agent_pairs:
            if (now - ts) >= self._AGENT_LINE_DEDUP_WINDOW_SEC:
                continue
            if prev_a == a_norm:
                # Same agent line recently spoken — duplicate regardless of user turn.
                return True
            # Very similar (same prefix ≥ 90%) on a non-trivial reply — treat as dup too.
            if len(a_norm) > 30 and (
                a_norm.startswith(prev_a) or prev_a.startswith(a_norm)
            ):
                shorter, longer = sorted((a_norm, prev_a), key=len)
                if shorter and len(shorter) / max(len(longer), 1) >= 0.9:
                    # And the user turn matches (or was empty) — safe to dedupe.
                    if not u_norm or not prev_u or prev_u == u_norm:
                        return True
        return False

    def _is_agent_self_echo(self, transcript: str) -> bool:
        """
        True when an incoming STT closely matches text the agent is currently speaking
        or recently spoke.

        Phone-line sidetone and open-mic setups can feed the agent's TTS audio back
        into the STT stream as a transcript.
        """
        if not transcript:
            return False
        t_norm = self._normalize_turn_text(transcript)
        if not t_norm:
            return False
        t_words = t_norm.split()
        if not t_words:
            return False

        # 1. Live speaking check: if agent is currently speaking, check if incoming STT
        # text matches what the agent is actively uttering right now (even 2-3 words).
        current_speaking = getattr(self, "_current_speaking_agent_text", "")
        if current_speaking:
            curr_norm = self._normalize_turn_text(current_speaking)
            if curr_norm:
                if t_norm in curr_norm or curr_norm in t_norm:
                    return True
                curr_words = curr_norm.split()
                if len(curr_words) >= 2 and len(t_words) >= 2:
                    overlap_curr = len(set(t_words) & set(curr_words))
                    if (overlap_curr / len(t_words)) >= 0.75:
                        return True

        # 2. Historical check: last 12s of committed agent replies (requires >= 4 words to avoid false positives)
        recent_pairs = getattr(self, "_recent_agent_pairs", None)
        if not recent_pairs:
            return False
        if len(t_words) < 4:
            return False
        t_word_set = set(t_words)
        now = time.monotonic()
        for _u, a_norm, ts in recent_pairs:
            if (now - ts) > 12.0:
                continue
            if not a_norm:
                continue
            a_word_set = set(a_norm.split())
            if len(a_word_set) < 4:
                continue
            overlap = len(t_word_set & a_word_set)
            shorter = min(len(t_word_set), len(a_word_set))
            if shorter > 0 and (overlap / shorter >= 0.80):
                return True
        return False

    def _remember_agent_turn(self, user_text: str | None, agent_text: str) -> None:
        """Append (user_norm, agent_norm, ts) and bound the buffer to the last few entries."""
        if not agent_text:
            return
        a_norm = self._normalize_turn_text(agent_text)
        if not a_norm:
            return
        u_norm = self._normalize_turn_text(user_text or "")
        self._recent_agent_pairs.append((u_norm, a_norm, time.monotonic()))
        if len(self._recent_agent_pairs) > self._RECENT_AGENT_PAIRS_MAX:
            self._recent_agent_pairs = self._recent_agent_pairs[
                -self._RECENT_AGENT_PAIRS_MAX :
            ]

    def _should_defer_interim_response(self, transcript: str) -> bool:
        """
        Avoid early LLM responses for short/ambiguous booking clarifications.
        This keeps the low-latency path for normal conversation while waiting for
        a final STT transcript when the caller is choosing dates/times or correcting us.
        """
        if not self._is_booking_context_active(transcript):
            return False

        normalized = self._normalize_turn_text(transcript)
        if not normalized:
            return False

        words = normalized.split()
        if len(words) <= 4:
            return True

        if re.fullmatch(r"\d{1,2}(?::\d{2})?\s*(?:a\s*m|am|p\s*m|pm)?", normalized):
            return True

        clarification_markers = (
            " am",
            " pm",
            " a m",
            " p m",
            "slot",
            "time",
            "date",
            "spell",
            "spelled",
            "already",
            "wrong",
            "available",
        )
        return any(marker in f" {normalized} " for marker in clarification_markers)

    def _is_natural_continuation_of_seed(self, final_norm: str, seed_norm: str) -> bool:
        """
        True when the final text is the same utterance as the interim, with a bit more
        at the end (user was still talking). In that case we should NOT run a second
        LLM+TTS (Vapi-style: one reply per barge-in segment, no double-audio / distortion).
        """
        if not seed_norm or not final_norm or final_norm == seed_norm:
            return False
        if not final_norm.startswith(seed_norm):
            return False
        # Very short interims: always allow final to replace (e.g. "I need" -> "I need a refund")
        seed_words = seed_norm.split()
        if len(seed_words) < 3:
            return False
        extra = final_norm[len(seed_norm) :].strip()
        if not extra:
            return True
        # New semantic / correction content → second response is appropriate
        if re.search(
            r"\b(refund|refunds|help|emergency|cancel|complaint|dispute|manager|operator|"
            r"supervisor|wrong|problem|lawyer|sue|angry|escalat)\b",
            extra,
        ):
            return False
        extra_words = extra.split()
        if len(extra_words) > 6:
            return False
        return True

    def _should_regenerate_on_final(self, final_transcript: str) -> bool:
        """
        If an interim run used partial STT, decide whether a final run with full text
        is needed. Skip regeneration when the final is a natural extension of the seed
        (avoids back-to-back TTS and sounds much closer to Vapi).
        """
        if not self._turn_response_seed_text:
            return False

        final_norm = self._normalize_turn_text(final_transcript)
        seed_norm = self._normalize_turn_text(self._turn_response_seed_text)
        if not final_norm:
            return True
        if not seed_norm:
            return True

        if final_norm == seed_norm:
            if not self._is_booking_context_active(final_transcript):
                return False
            final_slot = self._resolve_cached_calendar_slot(final_transcript)
            seed_slot = self._resolve_cached_calendar_slot(
                self._turn_response_seed_text
            )
            if final_slot and seed_slot and final_slot != seed_slot:
                return True
            return False

        # Booking: slot/date resolution changed — re-run
        if self._is_booking_context_active(
            final_transcript
        ) or self._is_booking_context_active(self._turn_response_seed_text):
            final_slot = self._resolve_cached_calendar_slot(final_transcript)
            seed_slot = self._resolve_cached_calendar_slot(
                self._turn_response_seed_text
            )
            if final_slot != seed_slot:
                return True
            correction_markers = (
                "wrong",
                "no no",
                "not ",
                "already",
                "spell",
                "11 00",
                "11 am",
            )
            if any(marker in final_norm for marker in correction_markers):
                return True

        # Same line still being dictated — one spoken reply is enough
        if self._is_natural_continuation_of_seed(final_norm, seed_norm):
            return False

        # STT word-level revision (e.g. "Alex Carlton" → "Alex Carter"): a long common
        # prefix with only the trailing 1–2 words changed is almost always Deepgram
        # correcting a mishear, not the user adding new intent. Skipping regen avoids
        # the "first reply uses wrong name, second reply uses right name" double-TTS.
        seed_words = seed_norm.split()
        final_words = final_norm.split()
        if seed_words and final_words and len(seed_words) >= 4:
            common_prefix = 0
            for sw, fw in zip(seed_words, final_words):
                if sw == fw:
                    common_prefix += 1
                else:
                    break
            if (
                common_prefix >= len(seed_words) - 1
                and abs(len(final_words) - len(seed_words)) <= 1
            ):
                return False

        return True

    def _update_booking_memory_from_user_turn(self, transcript: str) -> None:
        if not transcript or not self._last_offered_calendar_slots:
            return
        resolved_slot = self._resolve_cached_calendar_slot(transcript)
        if resolved_slot is not None:
            self._last_selected_calendar_slot = resolved_slot

    def _build_booking_memory_block(self) -> str:
        if not self._is_booking_context_active():
            return ""

        lines = [
            "# BOOKING MEMORY",
            "Use this deterministic booking memory before asking repeated questions.",
        ]
        if self._last_requested_calendar_date is not None:
            lines.append(
                f"- Date already discussed: {self._last_requested_calendar_date.strftime('%A, %B %d, %Y')}."
            )
        if self._last_offered_calendar_slots:
            offered = ", ".join(
                slot.strftime("%I:%M %p").lstrip("0")
                for slot in self._last_offered_calendar_slots[:8]
            )
            lines.append(f"- Last offered slots: {offered}.")
        if self._last_selected_calendar_slot is not None:
            lines.append(
                "- Current caller-selected slot candidate: "
                f"{self._last_selected_calendar_slot.strftime('%A, %B %d at %I:%M %p')}."
            )
        lines.append(
            "- If the caller gives a short clarification like '11', '11 a.m.', or corrects you, "
            "resolve it against the last offered slots before asking again."
        )
        lines.append(
            "- If appointment type/date/slot is already present here or in recent history, "
            "do not ask for it again; move to the next missing field."
        )
        return "\n".join(lines)

    @staticmethod
    def _strip_premature_booking_confirmation(text: str) -> str:
        """
        Remove assistant self-confirmations so final confirmation comes only after backend success.

        The voice agent must NOT tell the caller their booking is confirmed
        during the call — booking confirmation is owned by the connected
        Calendly integration (see `_execute_calendly_tool_call`), not the
        agent's own words. If the LLM ignores that rule and says "your
        appointment is booked", "your plumbing service is booked", "you're
        all set", etc., we strip those sentences before they reach TTS so the
        caller never hears a false confirmation.
        """
        if not text:
            return ""
        patterns = [
            # "Great/Done/Perfect, your appointment is scheduled/confirmed/booked"
            r"(?i)\b(?:great|done|perfect|all\s+set|excellent|wonderful)[^.!?]*\b(?:appointment|booking|reservation|visit|service)\b[^.!?]*\b(?:scheduled|confirmed|booked|reserved|set|locked\s+in)\b[^.!?]*[.!?]?",
            # "Your appointment/booking/visit/service is booked/confirmed/scheduled"
            r"(?i)\byour\s+(?:[a-z]+\s+){0,4}?(?:appointment|booking|reservation|visit|service|slot|time)\b[^.!?]*\b(?:scheduled|confirmed|booked|reserved|set|locked\s+in)\b[^.!?]*[.!?]?",
            # "You're (all) booked / set / scheduled / confirmed"
            r"(?i)\byou'?re\s+(?:all\s+)?(?:booked|set|scheduled|confirmed|locked\s+in)\b[^.!?]*[.!?]?",
            # "I've (gone ahead and) booked you / scheduled you / locked it in"
            r"(?i)\bi'?(?:ve|have)\s+(?:gone\s+ahead\s+and\s+)?(?:booked|scheduled|reserved|locked\s+in|set\s+up|confirmed)\s+(?:you|your|it|the\s+(?:appointment|booking|visit|slot))\b[^.!?]*[.!?]?",
            # "We've booked you / We have you booked / We've got you scheduled"
            r"(?i)\bwe'?(?:ve|have)\s+(?:got\s+you\s+|you\s+)?(?:booked|scheduled|set\s+up|reserved|confirmed)\b[^.!?]*[.!?]?",
            # "Your X is booked/set/scheduled" (generic – safety net for unusual phrasings)
            r"(?i)\byour\s+(?:[a-z\-]+\s+){0,3}?(?:is|has\s+been)\s+(?:booked|scheduled|reserved|confirmed|locked\s+in|set\s+up)\b[^.!?]*[.!?]?",
            # Aftermath / closing pleasantries that imply success
            r"(?i)\ba\s+confirmation\s+(?:message|email|text)\s+(?:will\s+be\s+|has\s+been\s+)?sent[^.!?]*[.!?]?",
            r"(?i)\bwe\s+look\s+forward\s+to\s+seeing\s+you[^.!?]*[.!?]?",
            r"(?i)\bsee\s+you\s+(?:then|tomorrow|on\s+\w+)\b[^.!?]*[.!?]?",
        ]
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _strip_control_tokens_for_tts(text: str) -> str:
        """
        Remove control/action tokens from text before it is spoken.
        Handles bracketed and malformed/unbracketed variants.
        """
        if not text:
            return ""
        out = text
        # Bracketed canonical tokens (case-insensitive — LLMs vary)
        out = _RE_VOICE_END_CALL.sub("", out)
        out = _RE_VOICE_SCREENING_QUALIFIED.sub("", out)
        out = re.sub(r"\[\s*TRANSFER_CALL\s*\]", "", out, flags=re.IGNORECASE)
        out = re.sub(r"\[OUTCOME:[^\]]+\]", "", out)
        # Strip all known ElevenLabs-style audio tags and common variants so they
        # are never spoken as literal words regardless of TTS provider.
        out = strip_eleven_v3_style_tags_for_non_eleven_tts(out)
        # Citation-like bracket numbers such as [1], [2], ...
        out = re.sub(r"\[\s*\d{1,3}\s*\]", "", out)
        # Malformed bracket-open tokens without closing bracket
        out = re.sub(
            r"\[OUTCOME:[^\]\n\r]*",
            "",
            out,
        )
        # Unbracketed control tails occasionally produced by model
        out = re.sub(
            r"(?im)\bOUTCOME\s*:\s*[^\n\r]*",
            "",
            out,
        )
        return out

    @staticmethod
    def _looks_like_control_leak(text: str) -> bool:
        """
        Detect token-like technical fragments that should never be spoken.
        """
        if not text:
            return False
        t = text.lower()
        leak_patterns = (
            r"\boutcome\b",
            r"\bclient phone number slot\b",
        )
        return any(re.search(p, t, flags=re.IGNORECASE) for p in leak_patterns)

    def _prepare_tts_text(self, text: str) -> str:
        """
        Final text gate before queueing TTS.
        """
        from app.utils.eleven_tts_text import prepare_tts_text_for_provider

        cleaned = self._strip_control_tokens_for_tts(text or "")
        cleaned = self._strip_premature_booking_confirmation(cleaned)
        cleaned = prepare_tts_text_for_provider(cleaned, None)
        if self._looks_like_control_leak(cleaned):
            logger.warning("TTSGuard: dropped token-like leak text=%r", cleaned[:180])
            return ""
        return cleaned

    def _resolve_cached_calendar_slot(self, slot_raw: str) -> datetime | None:
        normalized = self._normalize_calendar_slot_key(slot_raw)
        if not normalized or not self._last_offered_calendar_slots:
            return None

        for slot_dt in self._last_offered_calendar_slots:
            candidates = {
                slot_dt.isoformat(),
                slot_dt.strftime("%Y-%m-%d %H:%M"),
                slot_dt.strftime("%Y-%m-%d %I:%M %p").lstrip("0"),
                slot_dt.strftime("%I:%M %p").lstrip("0"),
                slot_dt.strftime("%H:%M"),
            }
            if slot_dt.minute == 0:
                candidates.add(slot_dt.strftime("%I %p").lstrip("0"))

            normalized_candidates = {
                self._normalize_calendar_slot_key(candidate) for candidate in candidates
            }
            if normalized in normalized_candidates:
                return slot_dt

        try:
            parsed_dt = datetime.fromisoformat(slot_raw.replace("Z", "+00:00"))
        except ValueError:
            parsed_dt = None

        if parsed_dt is not None:
            for slot_dt in self._last_offered_calendar_slots:
                if parsed_dt.tzinfo is None:
                    offered_local = slot_dt.replace(
                        tzinfo=None, second=0, microsecond=0
                    )
                    parsed_local = parsed_dt.replace(second=0, microsecond=0)
                    if offered_local == parsed_local:
                        return slot_dt
                else:
                    offered_utc = slot_dt.astimezone(timezone.utc).replace(
                        second=0, microsecond=0
                    )
                    parsed_utc = parsed_dt.astimezone(timezone.utc).replace(
                        second=0, microsecond=0
                    )
                    if offered_utc == parsed_utc:
                        return slot_dt

        for fmt in ("%I:%M %p", "%I %p", "%H:%M"):
            try:
                parsed_time = datetime.strptime(slot_raw.strip(), fmt).time()
                matches = [
                    slot_dt
                    for slot_dt in self._last_offered_calendar_slots
                    if slot_dt.hour == parsed_time.hour
                    and slot_dt.minute == parsed_time.minute
                ]
                if len(matches) == 1:
                    return matches[0]
            except ValueError:
                continue

        return None

    def _client_transcript_lines_newest_first(self, limit: int = 16) -> list[str]:
        """Recent client utterances (newest first) for voice email recovery."""
        conversation_history: list = []
        if self.call_session and self.call_session.call_transcript:
            try:
                raw = self.call_session.call_transcript
                conversation_history = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                conversation_history = []
        if not isinstance(conversation_history, list):
            return []
        out: list[str] = []
        for msg in reversed(conversation_history):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = (msg.get("content") or msg.get("message") or "").strip()
            message_type = msg.get("message_type", "")
            if (
                role == "client"
                and content
                and message_type not in ("greeting", "system", "status")
            ):
                out.append(content)
                if len(out) >= limit:
                    break
        return out
